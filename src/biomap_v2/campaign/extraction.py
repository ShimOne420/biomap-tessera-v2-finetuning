# Scopo: estrazione frozen per chunk, somme/conteggi atomici e ripresa senza doppio conteggio.
# Nessun training: il modello Medium è verificato mediante SHA256 prima del caricamento.
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from biomap_v2.common import digest, save_json, WEIGHTS_SHA256
from biomap_v2.encoder import load_encoder
from biomap_v2.reference import encode_pixels
from .inputs import identity, legacy_raw
from .state import atomic_npz, cached, MEANS

# Adatta il batch in OOM senza cambiare punti, date o precisione.
def infer(model, raw, device, batch, amp=False):
    while True:
        try:
            with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type,enabled=amp):
                return encode_pixels(model,raw['s2'],raw['s2_doys'],raw['asc'],raw['asc_doys'],
                    raw['desc'],raw['desc_doys'],raw['mask'],batch_pixels=batch,device=device)
        except torch.cuda.OutOfMemoryError:
            if batch==1:raise
            batch=max(1,batch//2);torch.cuda.empty_cache()
            print(f'CUDA OOM: nuovo pixel_batch={batch}',flush=True)

# Non si usano mai righe di un memmap oltre il progresso confermato del vecchio encoder.
def old_embeddings(source, low, high):
    progress=source/'encoder_progress.json'
    if not progress.exists():return None
    p=json.loads(progress.read_text())
    from biomap_v2.dense.v2_support import PROTOCOL
    if p.get('checkpoint_sha256')!=WEIGHTS_SHA256 or p.get('protocol')!=PROTOCOL:return None
    if p.get('inputs_sha256')!=digest(source/'inputs_manifest.json'):raise ValueError('Manifest legacy modificato')
    if p.get('pixels_complete',0)<high:return None
    return np.array(np.load(source/'embeddings_fp32.npy',mmap_mode='r')[low:high],copy=True)

# Il registro è derivato dai chunk: un crash non può lasciare un accumulatore contato due volte.
def extract(args, spec, frame, profile=False):
    work=Path(args.work_root); ref=Path(args.reference); started=time.perf_counter()
    if not args.checkpoint or digest(args.checkpoint)!=WEIGHTS_SHA256:raise ValueError('Checkpoint Medium SHA256 non conforme')
    model=None;processed=0;reused=0;timing={'input_seconds':0.,'inference_seconds':0.,'write_seconds':0.}
    sums=np.zeros((len(frame),128));counts=np.zeros(len(frame),dtype=np.int64)
    batch_limit=args.profile_chunks if profile else None; chunk_number=0
    for area in spec['areas']:
        aoi=area['aoi_id']
        if args.aoi and aoi not in args.aoi:continue
        lookup=np.load(ref/aoi/'lookup.npy')
        for tile_name in area['tiles']:
            tile=ref/aoi/tile_name; source=Path(args.source_root)/aoi/tile_name
            if source.exists() and digest(source/'geometry.npz')!=digest(tile/'geometry.npz'):raise ValueError('Geometry sorgente discordante')
            with np.load(tile/'geometry.npz') as z: ids=lookup[z['cell_ids']]
            for low in range(0,len(ids),args.chunk_size):
                high=min(low+args.chunk_size,len(ids)); tag={**identity(tile,low,high),'checkpoint':WEIGHTS_SHA256,'amp':args.amp}
                rawpath=work/'inputs'/aoi/tile_name/f'{low}-{high}.npz'
                input_manifest=source/'inputs_manifest.json'
                tag['inputs_sha256']=digest(rawpath) if rawpath.exists() else (digest(input_manifest) if input_manifest.exists() else 'missing')
                dest=work/'chunks'/aoi/tile_name/f'{low}-{high}.npz'
                if not profile and cached(dest,tag):
                    with np.load(dest) as z: cell=z['cell'];s=z['sums'];n=z['counts']
                    sums[cell]+=s;counts[cell]+=n;reused+=high-low;continue
                t=time.perf_counter();emb=None if (args.amp or profile) else old_embeddings(source,low,high)
                if emb is None:
                    rawpath=work/'inputs'/aoi/tile_name/f'{low}-{high}.npz'
                    if cached(rawpath,identity(tile,low,high)):
                        with np.load(rawpath) as z:raw={k:z[k] for k in z.files}
                    else:raw=legacy_raw(source,low,high,work/'staging'/aoi/tile_name)
                    timing['input_seconds']+=time.perf_counter()-t;t=time.perf_counter()
                    if model is None:
                        cfg={'indicator':'msa','model':{'device':args.device,'checkpoint':str(args.checkpoint)}}
                        model,_,_=load_encoder(cfg,override=args.checkpoint);model.eval().requires_grad_(False)
                    available=raw['mask'].any(axis=1) | np.any(raw['asc']!=0,axis=(1,2)) | np.any(raw['desc']!=0,axis=(1,2))
                    if not available.all():raise ValueError('Pixel senza input informativo: supporto non ridotto silenziosamente')
                    emb=infer(model,raw,args.device,args.pixel_batch,args.amp)
                    timing['inference_seconds']+=time.perf_counter()-t
                if emb.shape!=(high-low,128) or not np.isfinite(emb).all():raise ValueError('Embedding invalidi')
                selected=ids[low:high];valid=selected>=0; cell,inverse=np.unique(selected[valid],return_inverse=True)
                s=np.zeros((len(cell),128));np.add.at(s,inverse,emb[valid].astype(np.float64));n=np.bincount(inverse,minlength=len(cell))
                t=time.perf_counter()
                if not profile:atomic_npz(dest,tag,cell=cell,sums=s,counts=n)
                timing['write_seconds']+=time.perf_counter()-t;sums[cell]+=s;counts[cell]+=n
                processed+=high-low;chunk_number+=1
                save_json(work/('profile_progress.json' if profile else 'extraction_progress.json'),{'pixels_new':processed,'pixels_reused':reused,'last_tile':f'{aoi}/{tile_name}','elapsed_seconds':time.perf_counter()-started})
                print(f'{aoi}/{tile_name}: {high}/{len(ids)}; nuovi {processed}, cache {reused}',flush=True)
                if batch_limit and chunk_number>=batch_limit:break
            if batch_limit and chunk_number>=batch_limit:break
        if batch_limit and chunk_number>=batch_limit:break
    elapsed=time.perf_counter()-started
    if profile:
        from .reporting import peak_ram
        ram=peak_ram()
        result={**timing,'seconds':elapsed,'pixels':processed,'pixels_per_second':processed/elapsed,
            'cells_complete':int(((counts==frame.n_valid)&(counts>0)).sum()),'ram_ru_maxrss_native':ram,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else None,
            'eta_seconds_linear':float((frame.n_valid.sum()-processed)/(processed/elapsed)),
            'confidence':'limited sample; includes cache reuse; not guaranteed inference ETA'}
        save_json(work/'profile.json',result);return result
    expected=frame.n_valid.to_numpy(); eligible=frame.aoi_id.isin(args.aoi) if args.aoi else np.ones(len(frame),bool)
    if not np.array_equal(counts[eligible],expected[eligible]):raise ValueError('Supporto incompleto: nessuna matrice completa pubblicata')
    result=frame.loc[eligible,['cell_id','aoi_id','partition','cv_fold','lon','lat','target','n_valid','n_total','valid_fraction']].copy()
    means=sums[eligible]/counts[eligible,None]
    for i,c in enumerate(MEANS):result[c]=means[:,i]
    destination=work/('v2_subset.parquet' if args.aoi else 'v2.parquet')
    result.to_parquet(destination,index=False)
    save_json(work/'extraction_complete.json',{'sha256':digest(destination),'amp':args.amp,'n_cells':len(result),'complete_campaign':len(result)==132900})
    return result
