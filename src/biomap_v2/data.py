# Scopo: acquisire target ufficiali e serie annuali Sentinel nelle celle degli split.
# Fasi: target -> punti stratificati -> catalogo STAC -> letture COG -> cache per cella.
# Input: CSV geometrie, YAML, eventuale raster/cache SSD; output: NPZ, JSON e celle eleggibili.
# Parametri: prepare --limit N, --target-raster PATH, --import-cache PATH, --set sampling.*.
# Esempio: python scripts/probe_msa.py prepare --data-root E:\BioMAP --limit 12.
# Risorse: Internet e rasterio; cache bounded, disco con riserva; S1 può richiedere PC_SDK_SUBSCRIPTION_KEY.
# Ripresa: cataloghi/giorni/celle verificati con hash; nessun ricampionamento dopo errori rete.
from pathlib import Path
from datetime import datetime
import io
import json
import os
import shutil
import time
import uuid
import zipfile
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .common import budget, cells, digest, root, save_json
from functools import lru_cache
from types import SimpleNamespace

BANDS = ['B04','B02','B03','B08','B8A','B05','B06','B07','B11','B12']
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
PROTOCOL = 'stratified-native-cell-v2-power-rtc-v1'


def session():
    s=requests.Session()
    s.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    return s


def download(url, path, cfg, params=None):
    path=Path(path)
    partial=path.with_suffix(path.suffix+'.partial')
    path.parent.mkdir(parents=True,exist_ok=True)
    with session().get(url,params=params,stream=True,timeout=(20,180)) as response:
        response.raise_for_status()
        size=int(response.headers.get('Content-Length',0))
        budget(cfg,size)
        with partial.open('wb') as output:
            for chunk in response.iter_content(1024**2):
                budget(cfg,len(chunk));output.write(chunk)
    os.replace(partial,path)
    return path


def unzip_named(archive, names, destination, cfg):
    with zipfile.ZipFile(archive) as z:
        nested=next((n for n in z.namelist() if n.lower().endswith('.zip')),None)
        if nested:
            budget(cfg,z.getinfo(nested).file_size)
            with zipfile.ZipFile(io.BytesIO(z.read(nested))) as inner:
                return extract(inner,names,destination,cfg)
        return extract(z,names,destination,cfg)


def extract(z,names,destination,cfg):
    entries={Path(n).name:n for n in z.namelist()}
    for name in names:
        if name not in entries:raise ValueError(f'Archivio privo di {name}')
        target=destination/name
        budget(cfg,z.getinfo(entries[name]).file_size)
        with z.open(entries[name]) as source,target.open('wb') as output:shutil.copyfileobj(source,output)


def target_raster(cfg,frame,supplied=None):
    from .target_data import ensure_target
    return ensure_target(cfg,frame,supplied)


def targets(cfg,frame,path):
    from .target_data import validated_targets
    return validated_targets(cfg,frame,path)


def sample_points(cell,n):
    side=int(np.sqrt(n));offset=(np.arange(side)+.5)/side
    x,y=np.meshgrid(cell.west+offset*(cell.east-cell.west),cell.south+offset*(cell.north-cell.south))
    return np.column_stack([x.ravel(),y.ravel()])


def power_to_storage(power):
    # MPC gamma0 is intensity: amplitude=sqrt(power), so 20log10(amplitude)=10log10(power).
    out=np.zeros_like(power,dtype=np.int16);valid=np.isfinite(power)&(power>0)
    out[valid]=np.clip((10*np.log10(power[valid])+50)*200,0,32767).astype(np.int16)
    return out


@lru_cache(maxsize=8)
def aoi_bounds(path,mtime):
    frame=pd.read_csv(path)
    return {name:SimpleNamespace(west=float(f.west.min()),south=float(f.south.min()),east=float(f.east.max()),north=float(f.north.max())) for name,f in frame.groupby('aoi_id')}


def query(cfg,cell,collection,cache):
    if cfg.get('acquisition',{}).get('shared_catalogs'):
        geometry=Path(cfg['cells'])
        cell_bounds=aoi_bounds(str(geometry),geometry.stat().st_mtime_ns)[cell.aoi_id]
        folder=root(cfg)/'catalogs'/cell.aoi_id
        folder.mkdir(parents=True,exist_ok=True)
        cache=folder/f'{cfg["year"]}_{collection}.json'
    else:cell_bounds=cell
    if cache.exists():return json.loads(cache.read_text())
    body={'collections':[collection],'bbox':[cell_bounds.west,cell_bounds.south,cell_bounds.east,cell_bounds.north],
          'datetime':f'{cfg["year"]}-01-01T00:00:00Z/{cfg["year"]}-12-31T23:59:59Z','limit':500}
    s=session();url=STAC;method='POST';features=[]
    while url:
        r=s.request(method,url,json=body if method=='POST' else None,timeout=90);r.raise_for_status();data=r.json()
        features.extend(data['features'])
        link=next((x for x in data.get('links',[]) if x['rel']=='next'),None)
        if link:url=link['href'];method=link.get('method','GET');body=link.get('body',body)
        else:url=None
    save_json(cache,features)
    return features


class Reader:
    def __init__(self):self.http=session();self.tokens={}
    def signed(self,href):
        from urllib.parse import urlparse
        parsed=urlparse(href);account=parsed.netloc.split('.')[0];container=parsed.path.split('/')[1]
        key=(account,container)
        entry=self.tokens.get(key)
        if entry is None or time.time()-entry[0]>900:
            secret=os.environ.get('PC_SDK_SUBSCRIPTION_KEY')
            headers={'Ocp-Apim-Subscription-Key':secret} if secret else {}
            r=self.http.get(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}',headers=headers,timeout=45)
            if r.status_code in (401,403):raise RuntimeError('Accesso MPC negato: impostare PC_SDK_SUBSCRIPTION_KEY oppure importare cache; non salvare la chiave in Git')
            r.raise_for_status();self.tokens[key]=(time.time(),r.json()['token'])
        return href.split('?')[0]+'?'+self.tokens[key][1]
    def values(self,item,band,points):
        if band not in item['assets']:raise ValueError(f'Asset {band} assente nella scena {item["id"]}')
        href=self.signed(item['assets'][band]['href'])
        # Never log signed URLs; restrict GDAL network reads to requested COG blocks.
        for attempt in range(3):
          try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',GDAL_HTTP_MAX_RETRY='3'):
                with rasterio.open(href) as ds:
                    x,y=transform_coords(4326,ds.crs,points[:,0].tolist(),points[:,1].tolist())
                    result=np.array([float(v[0]) if not np.ma.is_masked(v[0]) else np.nan for v in ds.sample(zip(x,y),masked=True)])
            return result
          except rasterio.errors.RasterioError:
            self.tokens.clear()
            if attempt == 2:
                raise RuntimeError(f'Lettura COG fallita dopo 3 tentativi: scena {item["id"]}, banda {band}; riprovare prepare') from None
            time.sleep(2 ** attempt)


def acquire(cfg,cell,directory):
    points=sample_points(cell,cfg['sampling']['pixels_per_cell']);reader=Reader();streams={'s2':{},'asc':{},'desc':{}}
    for collection in ['sentinel-2-l2a','sentinel-1-rtc']:
        items=query(cfg,cell,collection,directory/f'{collection}.json')
        for item in items:
            props=item['properties'];day=props['datetime'][:10]
            source='s2' if collection=='sentinel-2-l2a' else ('asc' if props.get('sat:orbit_state')=='ascending' else 'desc')
            streams[source].setdefault(day,[]).append(item)
    outputs={};manifest={}
    for source,days in streams.items():
        bands=[];masks=[];doys=[];manifest[source]=[]
        for day,items in sorted(days.items()):
            cache=directory/f'{source}_{day}.npz';meta=cache.with_suffix('.json')
            if cache.exists():
                if not meta.exists() or digest(cache)!=json.loads(meta.read_text())['sha256']:raise ValueError(f'Cache giorno corrotta: {cache.name}')
                with np.load(cache) as z:arr=z['bands'];valid=z['valid']
            else:
                arr=np.zeros((len(points),10 if source=='s2' else 2),dtype=np.uint16 if source=='s2' else np.int16)
                valid=np.zeros(len(points),dtype=bool)
                for item in sorted(items,key=lambda i:(i['properties'].get('eo:cloud_cover',0),i['id'])):
                    if source=='s2':
                        scl=reader.values(item,'SCL',points)
                        clear=np.isfinite(scl)&~np.isin(scl,[0,1,2,3,8,9])
                        raw=np.stack([reader.values(item,b,points) for b in BANDS],axis=1)
                        take=clear&np.isfinite(raw).all(axis=1)&~valid
                        arr[take]=np.clip(raw[take],0,65535).astype(np.uint16);valid[take]=True
                    else:
                        raw=np.stack([reader.values(item,b,points) for b in ['vv','vh']],axis=1)
                        take=np.isfinite(raw).all(axis=1)&(raw>0).all(axis=1)&~valid
                        arr[take]=power_to_storage(raw[take]);valid[take]=True
                    if valid.all():break
                budget(cfg,arr.nbytes+valid.nbytes+4096)
                np.savez_compressed(cache,bands=arr,valid=valid)
                save_json(meta,{'sha256':digest(cache),'scene_ids':[i['id'] for i in items]})
            bands.append(arr);masks.append(valid)
            # Official public inference contract limits DOY to 365. 2020-12-31 maps to 365.
            doys.append(min(datetime.fromisoformat(day).timetuple().tm_yday,365))
            manifest[source].append({'date':day,'scene_ids':[i['id'] for i in items]})
        outputs[source]=np.stack(bands,axis=1) if bands else np.zeros((len(points),0,10 if source=='s2' else 2),dtype=np.float32)
        outputs[source+'_doys']=np.asarray(doys,dtype=np.int16)
        if source=='s2':outputs['mask']=np.stack(masks,axis=1) if masks else np.zeros((len(points),0),dtype=bool)
    good=outputs['mask'].sum(axis=1)>=cfg['sampling']['min_s2_observations']
    good &= (np.any(outputs['asc']!=0,axis=-1).sum(axis=1)+np.any(outputs['desc']!=0,axis=-1).sum(axis=1))>=cfg['sampling']['min_s1_observations']
    fraction=float(good.mean())
    if fraction<cfg['sampling']['min_valid_fraction']:
        save_json(directory/'excluded.json',{'reason':'insufficient_sentinel_coverage','valid_fraction':fraction})
        return None
    for key in ['s2','mask','asc','desc']:outputs[key]=outputs[key][good]
    outputs['points']=points[good]
    budget(cfg,sum(a.nbytes for a in outputs.values())+4096)
    path=directory/'series.npz';np.savez_compressed(path,**outputs)
    save_json(directory/'manifest.json',{'sha256':digest(path),'protocol':PROTOCOL,'year':cfg['year'],
        'pixels_per_cell':cfg['sampling']['pixels_per_cell'],'sampling':cfg['sampling'],
        'bounds':[cell.west,cell.south,cell.east,cell.north],'valid_fraction':fraction,'band_order':BANDS,
        'doy_policy':'actual day of year capped at 365','sar_units':'MPC intensity -> 10log10 -> +50 -> *200 int16',
        'scene_manifest':manifest})
    # Only delete our verified day intermediates after the complete series is durable.
    # Sources/imported originals are never deleted. This bounds inode count and cache size.
    with np.load(path) as verified:
        for key,value in outputs.items():
            if not np.array_equal(verified[key],value):raise ValueError('Consolidamento serie non verificato')
    for source,days in streams.items():
        for day in days:
            intermediate=directory/f'{source}_{day}.npz'
            intermediate.unlink(missing_ok=True)
            intermediate.with_suffix('.json').unlink(missing_ok=True)
    return fraction


def cell_directory(cfg,cell_id):
    return root(cfg)/'series'/cell_id.replace(':','_')


def prepare(cfg,args):
    frame=cells(cfg);all_cells=frame.copy();base=root(cfg);base.mkdir(parents=True,exist_ok=True);budget(cfg)
    # A limit samples within every partition, ensuring smoke runs still have train/val/test.
    if args.limit:
        if args.limit<3:raise ValueError('--limit minimo 3')
        parts=[p for p in ['train','validation','test'] if p in set(frame.partition)]
        sizes={p:args.limit//len(parts)+(i<args.limit%len(parts)) for i,p in enumerate(parts)}
        if cfg['evaluation'].get('mode')=='historical_groupkfold':
            if args.limit<4:raise ValueError('Smoke storico: minimo 4 celle per rappresentare tre fold e test')
            ntest=max(1,args.limit//2);ntrain=args.limit-ntest
            if ntrain<3:ntest=args.limit-3;ntrain=3
            pieces=[frame[(frame.partition=='train')&(frame.cv_fold==fold)].head(ntrain//3+(fold<ntrain%3)) for fold in range(3)]
            pieces.append(frame[frame.partition=='test'].head(ntest))
            frame=pd.concat(pieces,ignore_index=True)
        else:frame=pd.concat([frame[frame.partition==p].head(n) for p,n in sizes.items()],ignore_index=True)
    path=target_raster(cfg,all_cells,args.target_raster);frame=targets(cfg,frame,path)
    selected=[];excluded=[]
    for index,cell in enumerate(frame.itertuples()):
        directory=cell_directory(cfg,cell.cell_id);directory.mkdir(parents=True,exist_ok=True)
        if args.import_cache:
            source=Path(args.import_cache)/cfg['indicator']/'series'/directory.name
            if source.exists() and not (directory/'series.npz').exists():
                budget(cfg,sum(p.stat().st_size for p in source.rglob('*') if p.is_file()))
                shutil.copytree(source,directory,dirs_exist_ok=True)
        data=directory/'series.npz';meta=directory/'manifest.json'
        if data.exists():
            if not meta.exists():raise ValueError('Cache senza manifest: non usare array storici non verificati')
            info=json.loads(meta.read_text())
            if info['protocol']!=PROTOCOL or info['year']!=cfg['year'] or info.get('sampling')!=cfg['sampling'] or not np.allclose(info.get('bounds',[]),[cell.west,cell.south,cell.east,cell.north],atol=1e-10,rtol=0):
                raise ValueError('Cache incompatibile; scegliere una nuova --data-root')
            if digest(data)!=info['sha256']:raise ValueError('Cache cella corrotta')
            fraction=info['valid_fraction']
        else:fraction=acquire(cfg,cell,directory)
        if fraction is None:excluded.append({'cell_id':cell.cell_id,'reason':'insufficient_coverage'})
        else:selected.append({**cell._asdict(),'valid_fraction':fraction,'series_sha256':digest(data)})
        print(f'prepare {cfg["indicator"]}: {index+1}/{len(frame)} {cell.cell_id}',flush=True)
    eligible=pd.DataFrame(selected)
    if len(eligible)==0:raise ValueError('Nessuna cella eleggibile')
    filename='cells_smoke.csv' if args.limit else 'cells.csv'
    target=base/filename
    if target.exists():
        prior=pd.read_csv(target,dtype={'cell_id':str})
        if not prior.cell_id.tolist()==eligible.cell_id.tolist() or not np.allclose(prior.target,eligible.target,rtol=0,atol=1e-10):raise ValueError('Celle eleggibili/target cambiati: usare una nuova data-root per un nuovo studio')
    if cfg['evaluation'].get('mode')=='historical_groupkfold' and not args.limit:
        expected=cfg['evaluation']['expected_counts']
        actual=eligible.partition.value_counts().to_dict()
        if actual!=expected:
            destination=base/'diagnostics'/f'{__import__("datetime").datetime.now().strftime("%Y%m%dT%H%M%S%f")}_coverage'
            destination.mkdir(parents=True)
            eligible.to_csv(destination/'eligible.csv',index=False)
            save_json(destination/'coverage.json',{'expected':expected,'actual':actual,'excluded':excluded})
            raise ValueError(f'Copertura non completa: benchmark storico non pubblicato come equivalente; vedere {destination}')
    eligible.to_csv(target,index=False)
    save_json(target.with_suffix('.json'),{'protocol':PROTOCOL,'year':cfg['year'],'source_cells_sha256':digest(cfg['cells']),
        'eligible_sha256':digest(target),'target_source':json.loads((base/'targets/source.json').read_text()),
        'counts':eligible.partition.value_counts().to_dict(),'excluded':excluded,'config_sampling':cfg['sampling']})
    return target


def prepared(cfg,smoke=False):
    path=root(cfg)/('cells_smoke.csv' if smoke else 'cells.csv')
    if not path.exists():raise FileNotFoundError('Eseguire prepare prima di run; --smoke usa cells_smoke.csv')
    info=json.loads(path.with_suffix('.json').read_text())
    if digest(path)!=info['eligible_sha256'] or digest(cfg['cells'])!=info['source_cells_sha256']:
        raise ValueError('Manifest celle modificato dopo prepare')
    if info['config_sampling']!=cfg['sampling'] or info['year']!=cfg['year']:
        raise ValueError('Sampling/anno cambiati: nuova preparazione e data-root richieste')
    frame=pd.read_csv(path,dtype={'cell_id':str})
    historical=cfg['evaluation'].get('mode')=='historical_groupkfold'
    required={'train','test'} if historical else {'train','validation','test'}
    if set(frame.partition)!=required:raise ValueError(f'Partizioni richieste: {sorted(required)}')
    if historical:
        from .benchmark import validate_protocol
        validate_protocol(cfg,frame,smoke)
    for cell in frame.itertuples():
        if digest(cell_directory(cfg,cell.cell_id)/'series.npz')!=cell.series_sha256:raise ValueError('Serie modificata')
    return frame
