# Scopo: valutazione finale OOF/Liguria su TUTTI i contributi densi con pesi adattati, protocollo nuovo.
# Fasi: checkpoint fold/refit verificati -> acquisizione r1 per tile -> embedding temporanei -> head per cella.
# Input: reference storico e run completo campionato; output: predizioni/metriche dense separate da quelle sampled.
# Parametri: dense_pixel_batch esplicito, reserve_gib, resume; uso dense-eval --dense-pixel-batch 256.
# Risorse: rete/GPU/disco per UNA tile raw e mappe di UNA AOI; nessuna serie universitaria preesistente usata.
# Ripresa: tile/AOI con SHA; dopo commit si rimuove SOLO staging transitorio del nuovo run, mai dati/checkpoint.
import json
import hashlib
import os
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from biomap_v2.campaign import streaming as historical
from biomap_spatial.engine import atomic_parquet,metrics_report
from biomap_spatial.support import windows
from .data import reference,catalogs
from .training import registry,build_model
from .io import claim_root,signature,digest,save_json,load_checkpoint,disk_budget,Progress,replace_with_retry

PROTOCOL='finetune-v2-dense-eval-r1'


def dense_plan(cfg):
    spec,frame=reference(cfg);ref=Path(cfg['paths']['reference']);area_pixels=[];maximum=0
    for area in spec['areas']:
        sizes=[int(json.loads((ref/area['aoi_id']/tile/'geometry.json').read_text(encoding='utf-8'))['pixels']) for tile in area['tiles']]
        area_pixels.append(sum(sizes));maximum=max(maximum,max(sizes,default=0))
    return {'protocol':PROTOCOL,'cells':len(frame),'pixel_contributions':int(frame.n_valid.sum()),'training_support':'sampled, unchanged',
        'inference_support':'full_historical','peak_staging_bytes_illustrative':maximum*2956*2+max(area_pixels,default=0)*128*4,
        'storage_basis':'2956 raw bytes/pixel reported by updated PC handoff, actual scene counts checked per tile',
        'wall_time_seconds':None,'time_reference':'PC reports full frozen encoding 8-9h and r1 acquisition about 2.5 days; adapted end-to-end NOT measured',
        'selection':'fold-specific checkpoints for development AOIs, refit for Liguria; no reselection on dense test'}


def predict_cell(model,parts,cell_id,device):
    """parts contain actual native embeddings/flat positions, never reconstructed means."""
    if model.aggregation=='dense_mean':
        value=sum(np.asarray(p['embeddings'],np.float64).sum(0) for p in parts)/sum(len(p['flat']) for p in parts)
        x=torch.tensor(value[None],dtype=torch.float32,device=device)
        return float(model.head({'mean':(x-model.embedding_mean)/model.embedding_std}).item())
    patches=[];masks=[]
    for p in parts:
        rr=p['flat']//p['width'];cc=p['flat']%p['width'];r0=rr.min();c0=cc.min()
        mask=np.zeros((int(rr.max()-r0+1),int(cc.max()-c0+1)),bool);mask[rr-r0,cc-c0]=True
        for r,c,m in windows(mask,64):
            take=np.flatnonzero((rr-r0>=r)&(rr-r0<r+m.shape[0])&(cc-c0>=c)&(cc-c0<c+m.shape[1]))
            h,w=m.shape;value=torch.zeros(h*w,128,device=device)
            emb=torch.tensor(np.array(p['embeddings'][take]),device=device)
            emb=(emb-model.embedding_mean)/model.embedding_std
            local=(rr[take]-r0-r)*w+cc[take]-c0-c
            value[torch.tensor(local,device=device)]=emb
            patches.append(value.T.reshape(128,h,w));masks.append(torch.tensor(m[None],device=device))
    h=max(p.shape[1] for p in patches);w=max(p.shape[2] for p in patches)
    x=torch.stack([F.pad(p,(0,w-p.shape[2],0,h-p.shape[1])) for p in patches])
    mask=torch.stack([F.pad(p,(0,w-p.shape[2],0,h-p.shape[1])) for p in masks])
    return float(model.head({'features':x,'mask':mask,'owners':torch.zeros(len(x),dtype=torch.long,device=device),
        'weights':mask.sum((1,2,3)).float(),'cell_ids':[cell_id]}).item())


def dense_evaluate(cfg,run_id,pixel_batch,resume=False):
    if cfg['experiment']['profile']!='benchmark' or not isinstance(pixel_batch,int) or pixel_batch<1:
        raise ValueError('dense-eval richiede benchmark e dense-pixel-batch positivo esplicito')
    source,run,identity=registry(cfg,run_id,True);spec,frame=reference(cfg);ref=Path(cfg['paths']['reference'])
    checkpoints={}
    for fold in [0,1,2,'refit']:
        folder=run/('refit' if fold=='refit' else f'fold_{fold}')
        best=json.loads((folder/'best.json').read_text(encoding='utf-8'));p=folder/best['file']
        if not p.is_file():raise ValueError('Completare CV e refit prima di dense-eval')
        checkpoints[str(fold)]={'path':str(p),'sha256':digest(p)}
    dense_id=signature({'protocol':PROTOCOL,'run':identity,'checkpoints':checkpoints,'preprocessing':historical.code_identity(),'precision':'fp32'})
    root=run/'dense-eval'
    if root.exists() and not resume:raise ValueError('Dense-eval esistente: usare --resume')
    claim_root(root,dense_id,'dense_evaluation');progress=Progress(root/'progress.json')
    save_json(root/'manifest.json',{'protocol':PROTOCOL,'identity':dense_id,'training_support':source.spec['support'],
        'inference_support':'full_historical','checkpoints':checkpoints,'pixel_batch':pixel_batch,'precision':'fp32',
        'selection':'unchanged sampled CV; dense OOF descriptive, Liguria never selection','staging_policy':'remove ONLY own temporary raw after tile commit and own maps after AOI commit'})
    started=time.perf_counter();done=0;parts_out=[];timings=historical.Timings();reader=historical.reader(timings,root/'reader')
    try:
        for area in spec['areas']:
            aoi=area['aoi_id'];cells=frame[frame.aoi_id==aoi].copy();out=root/'aoi'/f'{aoi}.parquet';marker=out.with_suffix('.json')
            if marker.exists():
                m=json.loads(marker.read_text(encoding='utf-8'))
                if m['identity']!=dense_id or digest(out)!=m['sha256']:raise ValueError('AOI dense corrotta/incompatibile')
                parts_out.append(pd.read_parquet(out));done+=len(cells)
                temporary=root/'tmp'/aoi
                if temporary.exists():historical.remove_temporary(temporary,root)
                continue
            fold='refit' if cells.partition.eq('test').all() else str(int(cells.cv_fold.iloc[0]))
            if cells.partition.eq('train').any() and cells.cv_fold.nunique()!=1:raise ValueError('AOI attraversa fold')
            model=build_model(cfg);state=load_checkpoint(checkpoints[fold]['path'],identity);model.load_state_dict(state['model'],strict=True)
            del state;model.eval();tiles={};members={int(i):[] for i in cells.reference_index};lookup=np.load(ref/aoi/'lookup.npy')
            for tile_name in area['tiles']:
                tile=ref/aoi/tile_name;folder=root/'tmp'/aoi/tile_name;folder.mkdir(parents=True,exist_ok=True)
                with np.load(tile/'geometry.npz') as z:flat=z['flat_indices'].copy();ids=lookup[z['cell_ids']]
                tag=signature({'dense':dense_id,'aoi':aoi,'tile':tile_name,'geometry':digest(tile/'geometry.npz'),'checkpoint':checkpoints[fold]['sha256']})
                features=folder/'features.npy';complete=folder/'features.json';pixels=historical.Pixels(tile)
                if complete.exists():
                    m=json.loads(complete.read_text(encoding='utf-8'))
                    if m['identity']!=tag or digest(features)!=m['sha256']:raise ValueError('Tile adattata corrotta')
                else:
                    streams,box,hashes=catalogs(cfg,{'aoi':aoi,'tile':tile_name,'positions':np.array([0])},root)
                    bpp=21*len(streams['s2'])+4*(len(streams['asc'])+len(streams['desc']))
                    disk_budget(root,cfg['storage']['reserve_gib'],len(flat)*(2*bpp+128*4)+1024**2)
                    raw=historical.acquire_points(pixels,streams,folder/'raw',{'protocol':PROTOCOL,'identity':tag,'catalogs':hashes},reader,cfg['acquisition']['io_workers'])
                    if historical.coverage(raw,len(flat))['pixels_without_any_input']:raise ValueError('Full support con pixel privo input: nessuna esclusione')
                    tmp=folder/('features.partial-'+uuid.uuid4().hex[:8]+'.npy')
                    mmap=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(flat),128));written=[]
                    with torch.no_grad():
                        for low in range(0,len(flat),pixel_batch):
                            high=min(low+pixel_batch,len(flat));r={k:v[low:high] if k in {'s2','mask','asc','desc'} else v for k,v in raw.items()}
                            emb=model.encoder.encode_raw(r,pixel_batch).float().cpu().numpy()
                            if not np.isfinite(emb).all():raise ValueError('Embedding adattato non finito')
                            mmap[low:high]=emb
                            written.append((low,high,hashlib.sha256(emb.tobytes()).hexdigest()))
                            progress.update(phase='dense_encode',done=done,total=len(frame),unit='cells',current=f'{aoi}/{tile_name}',tile_pixels_done=high,tile_pixels=len(flat),eta_seconds=None)
                    mmap.flush();del mmap,raw
                    with tmp.open('r+b') as durable:durable.flush();os.fsync(durable.fileno())
                    check=np.load(tmp,mmap_mode='r',allow_pickle=False)
                    if check.shape!=(len(flat),128) or check.dtype!=np.float32:raise ValueError('Rilettura tile adattata incompatibile')
                    for low,high,expected_sha in written:
                        if hashlib.sha256(np.asarray(check[low:high]).tobytes()).hexdigest()!=expected_sha:raise ValueError('Rilettura embedding diversa dal forward')
                    del check;sha=digest(tmp);replace_with_retry(tmp,features)
                    save_json(complete,{'identity':tag,'sha256':sha,'geometry':digest(tile/'geometry.npz'),'protocol':PROTOCOL})
                if (folder/'raw').exists():historical.remove_temporary(folder/'raw',root)
                tiles[tile_name]={'path':features,'flat':flat,'width':pixels.spec['width']}
                order=np.argsort(ids,kind='stable')
                for positions in np.split(order,np.flatnonzero(np.diff(ids[order]))+1):
                    if len(positions) and int(ids[positions[0]]) in members:members[int(ids[positions[0]])].append((tile_name,positions))
            predictions=[]
            with torch.no_grad():
                for row in cells.itertuples():
                    pieces=[]
                    for tile_name,take in members[int(row.reference_index)]:
                        tile=tiles[tile_name];mmap=np.load(tile['path'],mmap_mode='r',allow_pickle=False)
                        pieces.append({'embeddings':np.array(mmap[take]),'flat':tile['flat'][take],'width':tile['width']});del mmap
                    if sum(len(p['flat']) for p in pieces)!=row.n_valid:raise ValueError('Supporto denso diverso dallo storico')
                    predictions.append(predict_cell(model,pieces,row.cell_id,cfg['execution']['device']))
            if not np.isfinite(predictions).all():raise ValueError('Predizioni dense non finite')
            cells['prediction']=predictions;cells['model']='predictor';cells['fold']=-1 if fold=='refit' else int(fold)
            dummy=cells.copy();train=frame[frame.partition=='train']
            if fold!='refit':train=train[train.cv_fold!=int(fold)]
            dummy['prediction']=float(train.target.mean());dummy['model']='dummy_mean'
            result=pd.concat([cells,dummy],ignore_index=True);out.parent.mkdir(parents=True,exist_ok=True);atomic_parquet(out,result)
            reread=pd.read_parquet(out)
            if not reread.equals(result):raise ValueError('Rilettura predizioni dense discordante')
            save_json(marker,{'identity':dense_id,'sha256':digest(out),'cells':len(cells)})
            parts_out.append(result);done+=len(cells);del model
            historical.remove_temporary(root/'tmp'/aoi,root)
            elapsed=time.perf_counter()-started;rate=done/max(elapsed,1e-9)
            progress.update(force=True,phase='dense_evaluate',done=done,total=len(frame),unit='cells',rate_per_second=rate,eta_seconds=(len(frame)-done)/rate)
        all_rows=pd.concat(parts_out,ignore_index=True)
        for partition,name in [('train','oof_dense'),('test','liguria_dense')]:
            rows=all_rows[all_rows.partition==partition];atomic_parquet(root/(name+'_predictions.parquet'),rows);metrics_report(rows,root,name)
        progress.update(force=True,phase='complete',done=len(frame),total=len(frame),eta_seconds=0)
        return {'protocol':PROTOCOL,'cells':len(frame),'status':'COMPLETE','path':str(root)}
    except BaseException as error:
        progress.update(force=True,phase='interrupted',error=repr(error),eta_seconds=None);raise
    finally:reader.close()
