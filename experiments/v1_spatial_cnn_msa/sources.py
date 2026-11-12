# Scopo: audit e acquisizione esplicita dei soli prodotti V1.0, più target GLOBIO invariati.
# Fasi: membership -> target read-only -> inventario int8/scale/CRS -> download privato se richiesto.
# Input: reference/CSV/target già disponibili e v1-root; output: indice e sorgenti nella work-root V1.
# Parametri: pilot/intermediate/full; nessuna inferenza Sentinel o lettura di output V2 densi.
# Risorse: memmap CPU; rete solo fetch/targets espliciti, timeout e resume con Range/ETag.
# Ripresa: .partial e marker SHA; sorgenti esterne mai sovrascritte o cancellate.
import json
import os
from pathlib import Path
import time
import hashlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from runtime import Budget,digest,save_json,signature


def owned_sources(cfg):return Path(cfg['work_root'])/'sources'


def materialized(path):
    path=Path(path)
    if getattr(path.stat(),'st_flags',0)&0x40000000:raise FileNotFoundError('File iCloud non materializzato: '+str(path))
    return path


def target_values(cfg,frame):
    path=cfg['data'].get('targets_csv')
    reference=cfg['data'].get('reference')
    if not path and reference:
        root=Path(reference);marker=root/'reference.json'
        if not marker.is_file():raise FileNotFoundError('Reference non finalizzata: reference.json assente')
        spec=json.loads(materialized(marker).read_text())
        if spec.get('protocol')!='dense127909-r1':raise ValueError('Protocollo reference incompatibile')
        p=Path(reference)/'v1.parquet'
        if not p.is_file() or spec.get('files',{}).get('v1.parquet')!=digest(materialized(p)):raise ValueError('Reference V1 parquet mancante o checksum discordante')
        path=str(p)
    if path:
        p=materialized(path)
        f=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,dtype={'cell_id':str})
        if 'cell_id' not in f or 'target' not in f:raise ValueError('Target tabulari: richiesti cell_id,target')
        f['cell_id']=f.cell_id.astype(str)
        if f.cell_id.duplicated().any():raise ValueError('Target duplicati')
        for key in ['aoi_id','partition','cv_fold','lon','lat']:
            if key in f:
                joined=frame[['cell_id',key]].merge(f[['cell_id',key]],on='cell_id',how='left',suffixes=('_expected','_provided'),validate='one_to_one')
                left=joined[key+'_expected'];right=joined[key+'_provided']
                same=np.allclose(left,right,rtol=0,atol=1e-8) if key in ['lon','lat'] else left.eq(right).all()
                if not same:raise ValueError('Target tabulari con geometria/split diversi: '+key)
        needed=['cell_id','target']+[x for x in ['n_valid','n_total',*['mean_%03d'%i for i in range(128)]] if x in f]
        values=frame.merge(f[needed],on='cell_id',how='left',validate='one_to_one')
        proof={'source':str(p.resolve()),'sha256':digest(p),'formula':'msa_area / cellarea','year':2020,'kind':'reference_or_declared_csv'}
    else:
        chunks=[];evidence=[]
        for aoi,part in frame.groupby('aoi_id',sort=True):
            if reference and (Path(reference)/aoi/'target.tif').is_file():
                num=Path(reference)/aoi/'target.tif';den=None
            else:
                pointer=Path(cfg['work_root'])/'targets/source.json'
                if not pointer.is_file():raise FileNotFoundError('Target mancanti: --reference, --targets-csv oppure patch_builder.py targets')
                info=json.loads(pointer.read_text());num=Path(info['raster'])
                den=num.with_name('cellarea_km2_10sec.tif') if num.name=='msa_area_10sec.tif' else None
            from biomap_v2.target_data import sample_raster,check_grid
            materialized(num)
            if den:materialized(den)
            check_grid({'indicator':'msa'},num,den)
            n,inside,masked,aligned=sample_raster(num,part)
            if den:
                d,di,dm,da=sample_raster(den,part)
                if not (di&~dm&da&np.isfinite(d)&(d>0)).all():raise ValueError('Denominatore GLOBIO non valido')
                n=n/d
            if not (inside&~masked&aligned).all():raise ValueError('Target fuori copertura/NoData/non allineato')
            sub=part.copy();sub['target']=n;chunks.append(sub)
            evidence.append({'aoi':aoi,'numerator':str(num),'numerator_sha256':digest(num),'denominator_sha256':digest(den) if den else None})
        values=pd.concat(chunks).set_index('cell_id').loc[frame.cell_id].reset_index()
        proof={'formula':'msa_area / cellarea','year':2020,'rasters':evidence}
    if values.target.isna().any() or not np.isfinite(values.target).all() or not values.target.between(0,1).all():raise ValueError('Target assenti/non finiti/fuori [0,1]; nessuna imputazione')
    return values,proof


def fetch_targets(cfg,frame):
    # Adattatore locale: stesso downloader scientifico, namespace privato, nessuna modifica al modulo.
    import yaml
    from runtime import REPO
    from biomap_v2.target_data import ensure_target
    base=yaml.safe_load((REPO/'configs/msa_v1_liguria.yaml').read_text())
    root=Path(cfg['work_root']);base.update(data_root=str(root.parent),dataset_namespace=root.name)
    base['storage']=cfg['storage'].copy()
    return ensure_target(base,frame,None)


def expected_tiles(frame):
    result={}
    for aoi,part in frame.groupby('aoi_id',sort=True):
        names=set()
        for r in part.itertuples():
            for x in range(int(np.floor(r.west*10+1e-9)),int(np.ceil(r.east*10-1e-9))):
                for y in range(int(np.floor(r.south*10+1e-9)),int(np.ceil(r.north*10-1e-9))):names.add(f'grid_{x/10+.05:.2f}_{y/10+.05:.2f}')
        result[aoi]=sorted(names)
    return result


def area_root(cfg,aoi):
    root=Path(cfg['data']['v1_root']) if cfg['data']['v1_root'] else owned_sources(cfg)
    candidate=root/aoi/'tessera/2020'
    if candidate.is_dir() or not cfg['data']['v1_root']:return candidate
    if (root/'global_0.1_degree_representation/2020').is_dir():return root
    return candidate


def area_roots(cfg,aoi):
    roots=[area_root(cfg,aoi),owned_sources(cfg)/aoi/'tessera/2020']
    return list(dict.fromkeys(roots))


def choose_tile(cfg,aoi,stem):
    for base in area_roots(cfg,aoi):
        spec=tile_spec(base,stem)
        if all(Path(spec[k]).is_file() for k in ['embedding','scales','landmask']):return base,spec
    base=owned_sources(cfg)/aoi/'tessera/2020'
    return base,tile_spec(base,stem)


def tile_spec(base,stem):
    folder=base/'global_0.1_degree_representation/2020'/stem
    return {'name':stem,'embedding':str(folder/(stem+'.npy')),'scales':str(folder/(stem+'_scales.npy')),
            'landmask':str(base/'global_0.1_degree_tiff_all'/(stem+'.tiff'))}


def inspect_tile(spec,checksum=False):
    for key in ['embedding','scales','landmask']:
        if not Path(spec[key]).is_file():raise FileNotFoundError(spec[key])
        materialized(spec[key])
    emb=np.load(spec['embedding'],mmap_mode='r',allow_pickle=False);sc=np.load(spec['scales'],mmap_mode='r',allow_pickle=False)
    if emb.ndim!=3 or emb.shape[2]!=128 or emb.dtype!=np.int8 or sc.shape!=emb.shape[:2] or sc.dtype!=np.float32:raise ValueError('Tile non conforme a V1.0 int8/scales per pixel: '+spec['name'])
    with rasterio.open(spec['landmask']) as ds:
        if ds.shape!=emb.shape[:2] or ds.crs is None or not ds.crs.is_projected or not np.allclose(ds.res,[10,10],atol=1e-6,rtol=0) or ds.transform.b or ds.transform.d:raise ValueError('CRS/griglia V1 non conforme: '+spec['name'])
        extra={'height':ds.height,'width':ds.width,'channels':128,'crs':str(ds.crs),'transform':list(ds.transform)[:6],'resolution_m':list(ds.res)}
    identity={key:({'sha256':digest(spec[key])} if checksum else {'size':Path(spec[key]).stat().st_size,'mtime_ns':Path(spec[key]).stat().st_mtime_ns}) for key in ['embedding','scales','landmask']}
    return {**spec,**extra,'fingerprint':signature(identity),'files':identity}


def inventory(cfg,frame,deep=False):
    want=expected_tiles(frame);found=[];missing=[];bad=[]
    for aoi,names in want.items():
        # Extra historical tiles are retained: do not silently deduplicate historical contributions.
        available={p.parent.name for base in area_roots(cfg,aoi) for p in (base/'global_0.1_degree_representation/2020').glob('grid_*/*_scales.npy')}
        used_bases=set()
        for stem in sorted(set(names)|available):
            base,spec=choose_tile(cfg,aoi,stem)
            if not all(Path(spec[k]).is_file() for k in ['embedding','scales','landmask']):
                if stem in names:missing.append({'aoi_id':aoi,'tile':stem})
                continue
            used_bases.add(base)
            try:found.append({**inspect_tile(spec,deep),'aoi_id':aoi})
            except (ValueError,OSError) as error:bad.append({'aoi_id':aoi,'tile':stem,'error':str(error)})
        for base in used_bases:
            meta=base/'tessera_metadata.json'
            if meta.exists():
                m=json.loads(meta.read_text())
                if str(m.get('dataset_version'))!='1.0' or m.get('dataset_variant')!='vultr' or m.get('year')!=2020:bad.append({'aoi_id':aoi,'error':'Metadati prodotto non V1.0 vultr 2020'})
            else:bad.append({'aoi_id':aoi,'error':'tessera_metadata.json assente: versione del prodotto non verificabile'})
    return found,missing,bad


def download(url,path,budget,progress):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);marker=path.with_suffix(path.suffix+'.download.json')
    partial=path.with_suffix(path.suffix+'.partial');pending=marker.with_name(marker.name+'.pending')
    if pending.exists():
        info=json.loads(pending.read_text());candidate=path if path.exists() else partial
        if not candidate.exists() or digest(candidate)!=info['sha256']:raise ValueError('Transazione download incompleta/corrotta')
        if candidate==partial:os.replace(partial,path)
        os.replace(pending,marker)
    if path.exists():
        if not marker.exists() or json.loads(marker.read_text()).get('sha256')!=digest(path):raise ValueError('Download esistente non verificato: '+str(path))
        return
    session=Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    state=partial.with_suffix(partial.suffix+'.json')
    try:
        head=session.head(url,timeout=(10,60));head.raise_for_status()
        size=int(head.headers.get('Content-Length',0));etag=head.headers.get('ETag')
        prior=json.loads(state.read_text()) if state.exists() else {}
        offset=partial.stat().st_size if partial.exists() else 0
        if offset and (prior.get('etag')!=etag or prior.get('bytes')!=size):raise ValueError('Sorgente remota cambiata: usare una nuova work-root')
        budget.check(max(size-offset,0));save_json(state,{'etag':etag,'bytes':size,'url':url})
        headers={'Range':f'bytes={offset}-','If-Range':etag} if offset and etag else {}
        if offset==size and size>0:response=None
        else:
            response=session.get(url,headers=headers,stream=True,timeout=(10,60));response.raise_for_status()
            if response.status_code==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Risposta Range non corrispondente al file parziale')
            if response.status_code!=206:offset=0
            with response,partial.open('ab' if offset else 'wb') as stream:
                for data in response.iter_content(4*1024**2):
                    if shutil_free(path.parent)<budget.reserve+len(data):raise RuntimeError('Riserva disco raggiunta')
                    stream.write(data);offset+=len(data);progress.update(current_file=path.name,bytes_done=offset,bytes_total=size)
                stream.flush();os.fsync(stream.fileno())
        if size<=0 or partial.stat().st_size!=size:raise ValueError('Lunghezza download non verificabile o incompleta')
        # A single-part ETag is a content MD5; multipart/opaque ETags are not treated as checksums.
        tag=(etag or '').strip('"')
        if len(tag)==32 and all(x in '0123456789abcdefABCDEF' for x in tag):
            h=hashlib.md5()
            with partial.open('rb') as stream:
                for data in iter(lambda:stream.read(4*1024**2),b''):h.update(data)
            if h.hexdigest().lower()!=tag.lower():raise ValueError('MD5 del download discordante')
        sha=digest(partial);save_json(pending,{'sha256':sha,'bytes':size,'etag':etag,'url':url,'version':'1.0'})
        os.replace(partial,path);os.replace(pending,marker)
    finally:session.close()


def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free


def fetch_tiles(cfg,frame,progress):
    wanted=expected_tiles(frame);total=sum(len(x)*3 for x in wanted.values());progress.update(total=total,unit='files')
    done=0;budget=Budget(cfg)
    for aoi,names in wanted.items():
        base=owned_sources(cfg)/aoi/'tessera/2020'
        downloaded=False
        for stem in names:
            old_base,old=choose_tile(cfg,aoi,stem)
            if all(Path(old[k]).is_file() for k in ['embedding','scales','landmask']):
                if old_base==base and not (base/'tessera_metadata.json').exists():
                    for key in ['embedding','scales','landmask']:
                        marker=Path(old[key]).with_suffix(Path(old[key]).suffix+'.download.json')
                        if not marker.exists() or json.loads(marker.read_text()).get('sha256')!=digest(old[key]):raise ValueError('Provenienza download interrotto non verificabile')
                    downloaded=True
                inspect_tile(old);done+=3;progress.update(done=done);continue
            spec=tile_spec(base,stem)
            downloaded=True
            paths={'embedding':f"{cfg['data']['source_url']}/2020/{stem}/{stem}.npy",
                   'scales':f"{cfg['data']['source_url']}/2020/{stem}/{stem}_scales.npy",
                   'landmask':f"{cfg['data']['landmask_url']}/{stem}.tiff"}
            for kind,url in paths.items():
                progress.update(aoi=aoi,tile=stem);download(url,spec[kind],budget,progress);done+=1;progress.update(done=done)
        if downloaded:save_json(base/'tessera_metadata.json',{'dataset_version':'1.0','dataset_variant':'vultr','year':2020,'source_url_prefix':cfg['data']['source_url']})
