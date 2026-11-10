# Scopo: lettura bounded delle serie legacy o acquisizione API esplicita per chunk.
# Output: cache NPZ verificata nella work-root; sorgenti sempre read-only.
import gzip
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rasterio
from pyproj import Transformer
from biomap_v2.common import digest
from biomap_v2.data import Reader, BANDS, power_to_storage, query
from .state import atomic_npz, cached

# Identità fisica del chunk: non dipende da dove sono montati i dischi.
def identity(tile, low, high):
    return {'protocol':'dense127909-input-r1','geometry':digest(tile/'geometry.npz'),
            'low':low,'high':high,'year':2020,'bands':BANDS,
            'preprocessing_sha256':digest(Path(__file__)),
            'official_preprocessing_sha256':digest(Path(__file__).parents[1]/'reference.py')}

# Gzip viene decompresso una volta nella cache scrivibile, non nella directory sorgente.
def array(source, key, staging):
    path = source/f'{key}.npy'
    if path.exists():
        verify_archive_array(source,key,path,False)
        return np.load(path,mmap_mode='r',allow_pickle=False)
    packed = source/f'{key}.npy.gz'
    if not packed.exists(): raise FileNotFoundError(packed)
    verify_archive_array(source,key,packed,True)
    staging.mkdir(parents=True,exist_ok=True); dest = staging/f'{key}.npy'
    signature = {'source_sha256':digest(packed)}
    if not cached(dest, signature):
        expected_size=json.loads((source/'input_archive_manifest.json').read_text())['arrays'][key]['original_bytes']
        if shutil.disk_usage(staging).free < expected_size+20*1024**3:raise OSError('Spazio insufficiente per decompressione; sorgente preservata')
        tmp = dest.with_suffix('.partial')
        with gzip.open(packed,'rb') as src, tmp.open('wb') as out: shutil.copyfileobj(src,out)
        os.replace(tmp,dest)
        from biomap_v2.common import save_json
        save_json(dest.with_suffix('.json'),{'identity':signature,'sha256':digest(dest)})
    return np.load(dest,mmap_mode='r',allow_pickle=False)

# Legacy time-major -> punto-major; nessuna interpolazione dei 25 punti.
def legacy_raw(source, low, high, staging):
    manifest = json.loads((source/'inputs_manifest.json').read_text())
    if set(manifest.get('scene_groups',{})) != {'s2','asc','desc'}: raise ValueError('Catalogo legacy incompleto')
    result = {}
    for key in ['s2','mask','asc','desc']:
        a = array(source,key,staging)
        result[key] = np.array(a[:,low:high].T if key=='mask' else a[:,low:high].transpose(1,0,2),copy=True)
    for key in ['s2','asc','desc']:
        days = manifest['scene_groups'][key]
        complete = manifest.get('complete_days',{}).get(key,{})
        if any(x['date'] not in complete for x in days): raise ValueError('Giorni legacy incompleti')
        result[key+'_doys'] = np.array([min(datetime.fromisoformat(x['date']).timetuple().tm_yday,365) for x in days],dtype=np.int16)
    return result

# Centri originali V1, non una nuova griglia regolare nella cella.
def points(tile, low, high):
    spec = json.loads((tile/'geometry.json').read_text())
    with np.load(tile/'geometry.npz') as z: flat = z['flat_indices'][low:high]
    rr, cc = flat//spec['width'], flat%spec['width']
    x,y = rasterio.Affine(*spec['transform'])*(cc+.5,rr+.5)
    return np.column_stack(Transformer.from_crs(spec['crs'],4326,always_xy=True).transform(x,y))

# Cataloghi congelati e checkpoint giornalieri: la ripresa non riacquisisce giorni validi.
def acquire(tile, low, high, work):
    p = points(tile,low,high); folder = work/'api'/tile.parent.name/tile.name
    cfg = {'year':2020,'data_root':str(folder),'dataset_namespace':'catalogs','acquisition':{}}
    cell = SimpleNamespace(west=p[:,0].min(),east=p[:,0].max(),south=p[:,1].min(),north=p[:,1].max())
    streams = {k:{} for k in ['s2','asc','desc']}; ident = identity(tile,low,high)
    folder.mkdir(parents=True,exist_ok=True)
    for collection in ['sentinel-2-l2a','sentinel-1-rtc']:
        # Query per tile intera: un catalogo non può essere quello del primo chunk soltanto.
        full = json.loads((tile/'geometry.json').read_text())['bbox']
        cell = SimpleNamespace(west=full[0],south=full[1],east=full[2],north=full[3])
        for item in query(cfg,cell,collection,folder/f'{collection}.json'):
            props=item['properties']; day=props['datetime'][:10]
            source='s2' if collection=='sentinel-2-l2a' else ('asc' if props.get('sat:orbit_state')=='ascending' else 'desc')
            streams[source].setdefault(day,[]).append(item)
    reader=Reader(); result={}
    try:
        for source, days in streams.items():
            bands=[]; masks=[]; doys=[]
            for day, items in sorted(days.items()):
                path=folder/f'{low}-{high}'/f'{source}_{day}.npz'
                tag={**ident,'source':source,'day':day,'scenes':sorted(x['id'] for x in items)}
                if cached(path,tag):
                    with np.load(path) as z: a=z['bands']; valid=z['valid']
                else:
                    a=np.zeros((len(p),10 if source=='s2' else 2),dtype=np.uint16 if source=='s2' else np.int16);valid=np.zeros(len(p),bool)
                    for item in sorted(items,key=lambda x:(x['properties'].get('eo:cloud_cover',0),x['id'])):
                        names=BANDS if source=='s2' else ['vv','vh']
                        raw=np.stack([reader.values(item,b,p) for b in names],axis=1)
                        if source=='s2':
                            scl=reader.values(item,'SCL',p);good=np.isfinite(scl)&~np.isin(scl,[0,1,2,3,8,9])
                        else: good=(raw>0).all(axis=1)
                        take=good&np.isfinite(raw).all(axis=1)&~valid
                        a[take]=np.clip(raw[take],0,65535).astype(np.uint16) if source=='s2' else power_to_storage(raw[take])
                        valid[take]=True
                        if valid.all():break
                    atomic_npz(path,tag,bands=a,valid=valid)
                bands.append(a);masks.append(valid);doys.append(min(datetime.fromisoformat(day).timetuple().tm_yday,365))
            result[source]=np.stack(bands,axis=1) if bands else np.zeros((len(p),0,10 if source=='s2' else 2),dtype=np.uint16 if source=='s2' else np.int16)
            result[source+'_doys']=np.array(doys,dtype=np.int16)
            if source=='s2':result['mask']=np.stack(masks,axis=1) if masks else np.zeros((len(p),0),bool)
    finally:reader.close()
    return result

# Valida gli hash del vecchio archivio una volta per file/stat, non per ogni chunk.
from functools import lru_cache

@lru_cache(maxsize=1024)
def checked_hash(path, size, mtime):
    return digest(path)

def verify_archive_array(source,key,path,compressed):
    archive=source/'input_archive_manifest.json'
    if not archive.exists():raise ValueError('Array legacy senza provenienza/hash: acquisire in una work-root nuova')
    info=json.loads(archive.read_text())['arrays'][key]
    expected=info['compressed_sha256' if compressed else 'original_sha256']
    st=path.stat()
    if checked_hash(str(path),st.st_size,st.st_mtime_ns)!=expected:raise ValueError('Array Sentinel legacy corrotto')

# Un array allocato ma con giorni incompleti non è una serie annuale acquisita.
def legacy_ready(source):
    source=Path(source);p=source/'inputs_manifest.json'
    if not p.exists():return False
    m=json.loads(p.read_text());groups=m.get('scene_groups',{});complete=m.get('complete_days',{})
    if set(groups)!={'s2','asc','desc'}:return False
    if any(any(x['date'] not in complete.get(k,{}) for x in days) for k,days in groups.items()):return False
    return all((source/f'{k}.npy').exists() or (source/f'{k}.npy.gz').exists() for k in ['s2','mask','asc','desc'])
