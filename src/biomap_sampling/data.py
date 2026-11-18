# Scopo: adapter opt-in del reader V2 esistente, manifest geografico e cache compatta condivisa.
# Fasi: indice read-only -> piano shard -> estrazione unica via V2Source.patches -> momenti training-only.
# Input: r2/reference/index esistenti; output: nuove root con float32 non normalizzati e piani verificati.
# Parametri: shard celle, seed, subset AOI, mmap LRU; esempio prepare(source,cfg), cache(source,cfg).
# Risorse: CPU/disco; nessuna rete/encoder. Ripresa: shard atomici SHA, marker finale solo a lavoro completo.
from collections import OrderedDict
import json
from pathlib import Path
import zlib
import sqlite3
import uuid
import numpy as np
import pandas as pd
from biomap_spatial.contracts import ExperimentSpec,ModelConfig,ExperimentProtocol
from biomap_spatial.sources import V2Source
from .geometry import select,unpack,xy
from .state import *


def blob(value):return np.frombuffer(zlib.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode('utf-8')),np.uint8)
def unblob(array):return json.loads(zlib.decompress(array.tobytes()).decode('utf-8'))


def code_identity():
    paths=list((ROOT/'src/biomap_sampling').glob('*.py'))+[ROOT/'scripts/sampling_benchmark.py',ROOT/'src/biomap_finetune/data.py']
    from biomap_spatial.utils import code_identity as spatial_code
    return {**spatial_code(),**{p.relative_to(ROOT).as_posix():digest(p) for p in paths}}


def roots(cfg):
    p=cfg['paths'];protected=[p.get(k) for k in ['pixel_root','reference','index_root','historical_run']]+p.get('read_only_roots',[])
    values=[]
    for k in ['cache_root','output_root']:
        if not p.get(k):raise ValueError('Percorso mancante: '+k)
        values.append(separate_output(p[k],protected))
    separate_output(values[0],[values[1]])
    return values


def open_source(cfg):
    p=cfg['paths']
    if any(not p.get(k) for k in ['pixel_root','reference','index_root']):raise ValueError('pixel-root/reference/index-root richiesti')
    source_cfg={'root':p['pixel_root'],'reference':p['reference'],'kind':'v2_pixels','representation':'spatial',
        'feature_space_id':'geotessera_v2_medium_3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e',
        'protocol':'dense127909-stream-legacy-exact-r2-fast','resolution':[10,10],'max_patch_side':64,'mmap_limit':4}
    spec=ExperimentSpec(source_cfg,ModelConfig(),ExperimentProtocol(),cfg['training'],cfg['execution'])
    # Constructor verifies the existing index. Never call prepare_index or write into index_root.
    return V2Source(spec,p['index_root'])


def development(source):
    frame=source.frame[source.frame.partition=='train'].copy().sort_values(['aoi_id','cell_id']).reset_index(drop=True)
    if frame.empty or frame.cell_id.duplicated().any() or set(frame.cv_fold)!={0,1,2}:raise ValueError('Sviluppo richiede tre fold/celle uniche')
    if frame.aoi_id.eq('liguria_levante').any() or not frame.groupby('aoi_id').cv_fold.nunique().eq(1).all():raise ValueError('Leakage geografico')
    return frame


class DevelopmentView:
    """Membership guard around the existing reader; does not implement or modify raster reading."""
    def __init__(self,source):
        self.source=source;self.frame=development(source);self.ids=set(self.frame.cell_id)
        self.fingerprint=source.fingerprint;self.feature_space_id=source.feature_space_id
    def patches(self,cell_id):
        if cell_id not in self.ids:raise ValueError('Cella esterna allo sviluppo: accesso benchmark vietato')
        return self.source.patches(cell_id)


def subset_ids(frame,n,seed):
    n=min(n,len(frame));counts=frame.groupby('aoi_id').size();quota=counts*n/len(frame);base=np.floor(quota).astype(int)
    if n<len(counts):raise ValueError('FULL-SUBSET deve poter rappresentare ogni AOI')
    missing=n-int(base.sum())
    order=sorted(counts.index,key=lambda a:(-(quota[a]-base[a]),a))
    for a in order[:missing]:base[a]+=1
    result=[]
    from .geometry import stable
    for a,g in frame.groupby('aoi_id'):
        result.extend(sorted(g.cell_id,key=lambda c:stable(seed,c,a,0,0))[:int(base[a])])
    return set(result)


def metadata(source,cell_id):
    patches=[]
    for patch_id,key,row,col,h,w,packed in source.db.execute('SELECT id,tile,row0,col0,height,width,mask FROM patches WHERE cell_id=? ORDER BY id',(cell_id,)):
        grid=source.tiles[key]['grid'];patches.append({'tile':key,'row':row,'col':col,'source_index_patch_id':int(patch_id),
            'grid':{'height':grid.height,'width':grid.width,'transform':list(grid.transform),'crs':grid.crs},
            'mask':np.unpackbits(np.frombuffer(packed,np.uint8),count=h*w).reshape(h,w).astype(bool)})
    return patches


def attach_positions(source,patch):
    rr,cc=np.where(unpack(patch));rows=rr+patch['row'];cols=cc+patch['col'];flat=rows*patch['grid']['width']+cols
    _,original=source.tile(patch['tile']);positions=np.searchsorted(original,flat)
    if (positions>=len(original)).any() or not np.array_equal(original[positions],flat):raise ValueError('Pixel selezionato fuori geometria originale')
    # Geometry is immutable in this read-only source: hash once/tile, never once/cell/patch.
    if not hasattr(source,'_sampling_geometry_shas'):source._sampling_geometry_shas={}
    hashes=source._sampling_geometry_shas
    if patch['tile'] not in hashes:hashes[patch['tile']]=digest(source.tiles[patch['tile']]['geometry'])
    patch.update(positions=positions.tolist(),flat_indices=flat.tolist(),rows=rows.tolist(),cols=cols.tolist(),
        geometry_sha256=hashes[patch['tile']])


def identity(source,cfg):
    return signature({'protocol':PROTOCOL,'source':source.fingerprint,'sampling':cfg['sampling'],'code':code_identity()})


def own(root,tag,kind):
    root=Path(root);mark=root/'ownership.json';expected={'protocol':PROTOCOL,'identity':tag,'kind':kind}
    if mark.exists():
        if json.loads(mark.read_text(encoding='utf-8'))!=expected:raise ValueError('Root incompatibile: nessun overwrite')
    else:
        if root.exists() and any(root.iterdir()):raise ValueError('Root non vuota non posseduta dal benchmark')
        root.mkdir(parents=True,exist_ok=True);save_json(mark,expected)


def prepare(source,cfg,resume=False,frame=None,deadline=None):
    cache_root,_=roots(cfg);frame=development(source) if frame is None else frame.copy()
    # Frame is now resolved; explicit ID membership is part of the root identity also for smoke.
    tag=signature({'identity':identity(source,cfg),'cells':frame.cell_id.tolist()})
    if (cache_root/'ownership.json').exists() and not resume:raise ValueError('Cache esistente: --resume richiesto')
    own(cache_root,tag,'sampling_cache');chosen_subset=subset_ids(frame,cfg['sampling']['subset_cells'],cfg['sampling']['seed'])
    progress=Progress(cache_root/'progress.json');began=__import__('time').perf_counter();shards=[];step=cfg['cache']['cells_per_shard'];summary={m:{'pixels':0,'cells':0,'insufficient':0,'rf_positions':0} for m in cfg['sampling']['modes']}
    for low in range(0,len(frame),step):
        check_deadline(deadline);rows=frame.iloc[low:low+step];key=f'{low:07d}';folder=cache_root/'plan'/key;block_tag=signature({'root':tag,'key':key})
        if folder.exists():arrays,info=load_arrays(folder,block_tag);records=unblob(arrays['records'])
        else:
            records=[]
            for row in rows.itertuples():
                geometry=metadata(source,row.cell_id);modes={}
                for mode in cfg['sampling']['modes']:
                    if mode=='FULL-SUBSET' and row.cell_id not in chosen_subset:continue
                    value=select(geometry,mode,row.cell_id,float(row.lon),float(row.lat),cfg['sampling']['seed'],cfg['sampling']['min_valid_pixels'])
                    for patch in value['patches']:attach_positions(source,patch)
                    modes[mode]=value
                records.append({'cell_id':row.cell_id,'aoi_id':row.aoi_id,'fold':int(row.cv_fold),'modes':modes,
                    'source_fingerprint':source.fingerprint,'sampling_seed':cfg['sampling']['seed'],'algorithm':PROTOCOL,
                    'checkpoint_sha256':'3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e'})
            disk_budget(cache_root,cfg['execution']['reserve_gib'],len(blob(records))*3)
            info=commit_arrays(folder,{'records':blob(records)},block_tag,{'cells':len(rows)})
        for cell in records:
            for mode,value in cell['modes'].items():
                d=value['diagnostics'];summary[mode]['pixels']+=d['selected_pixels'];summary[mode]['cells']+=1
                summary[mode]['insufficient']+=int(d['insufficient']);summary[mode]['rf_positions']+=d['rf_complete_positions']
        shards.append({'key':key,'identity':block_tag,'marker_sha256':digest(folder/'complete.json')})
        done=min(low+step,len(frame));rate=done/max(__import__('time').perf_counter()-began,1e-9)
        progress.update(force=True,phase='prepare',cells_done=done,cells_total=len(frame),cells_per_second=rate,eta_seconds=(len(frame)-done)/rate)
    atomic_frame(cache_root/'cells.parquet',frame)
    manifest={'protocol':PROTOCOL,'identity':tag,'source':source.fingerprint,'code':code_identity(),'sampling':cfg['sampling'],
        'cells':len(frame),'cells_sha256':digest(cache_root/'cells.parquet'),'shards':shards,'summary':summary,
        'temporal_masks':'NOT_AVAILABLE_FROM_EMBEDDINGS; future authorized acquisition must validate historical preprocessing',
        'reference_checkpoint':'3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e'}
    save_json(cache_root/'plan.json',manifest);progress.update(force=True,phase='prepare_complete',eta_seconds=0)
    return manifest


class Plan:
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'plan.json').read_text(encoding='utf-8'))
        if digest(self.root/'cells.parquet')!=self.manifest['cells_sha256']:raise ValueError('Celle del piano corrotte')
        self.frame=pd.read_parquet(self.root/'cells.parquet');self.ids={str(c):i for i,c in enumerate(self.frame.cell_id)}
        self.step=None;self.records=OrderedDict()
        for block in self.manifest['shards']:
            if digest(self.root/'plan'/block['key']/'complete.json')!=block['marker_sha256']:raise ValueError('Manifest shard modificato')
    def blocks(self):
        for block in self.manifest['shards']:
            arrays,_=load_arrays(self.root/'plan'/block['key'],block['identity']);yield block,unblob(arrays['records'])
    def cell(self,cell_id):
        import bisect
        index=self.ids[cell_id];starts=[int(b['key']) for b in self.manifest['shards']];b=self.manifest['shards'][bisect.bisect_right(starts,index)-1]
        if b['key'] not in self.records:
            arrays,_=load_arrays(self.root/'plan'/b['key'],b['identity']);self.records[b['key']]=unblob(arrays['records'])
            while len(self.records)>4:self.records.popitem(last=False)
        self.records.move_to_end(b['key']);record=self.records[b['key']][index-int(b['key'])]
        if record['cell_id']!=cell_id or record['source_fingerprint']!=self.manifest['source']:raise ValueError('Identita del record diversa dal manifest')
        return record


def future_requests(cell,mode):
    """Pure bridge: exact tile/row order, patch masks and weights; NO acquisition or encoder calls."""
    value=cell['modes'][mode];groups=[];lookup={};patches=[]
    for patch in value['patches']:
        mask=unpack(patch)
        if int(mask.sum())!=patch['weight'] or len(patch['positions'])!=patch['weight']:raise ValueError('Maschere/pesi non coerenti')
        rr,cc=np.where(mask);flat=(rr+patch['row'])*patch['grid']['width']+cc+patch['col']
        if not np.array_equal(flat,patch['flat_indices']):raise ValueError('Coordinate/ordine non coerenti')
        key=patch['tile']
        if key not in lookup:lookup[key]=len(groups);groups.append({'tile':key,'grid':patch['grid'],'geometry_sha256':patch['geometry_sha256'],'positions':[]})
        group=groups[lookup[key]]
        if group['grid']!=patch['grid'] or group['geometry_sha256']!=patch['geometry_sha256']:raise ValueError('Griglia/geometry incompatibile')
        group['positions'].extend(patch['positions'])
    for group in groups:group['positions']=sorted(set(group['positions']))
    for patch in value['patches']:
        group=lookup[patch['tile']];positions=groups[group]['positions'];take=np.searchsorted(positions,patch['positions'])
        patches.append({**patch,'group':group,'take':take.tolist()})
    return {'cell_id':cell['cell_id'],'aoi_id':cell['aoi_id'],'fold':cell['fold'],'groups':groups,'patches':patches,
        'weight':sum(p['weight'] for p in patches),'approved_for_acquisition':False,'temporal_masks_verified':False,
        'source_fingerprint':cell['source_fingerprint'],'sampling_seed':cell['sampling_seed'],'checkpoint_sha256':cell['checkpoint_sha256']}


def verify_requests(request,reference):
    """Read-only round trip to original geometry: no re-sampling, scene query or Sentinel I/O."""
    for group in request['groups']:
        tile=Path(reference)/group['tile']
        if digest(tile/'geometry.npz')!=group['geometry_sha256']:raise ValueError('Geometria originale cambiata')
        with np.load(tile/'geometry.npz',allow_pickle=False) as z:flat=z['flat_indices'][group['positions']]
        grid=json.loads((tile/'geometry.json').read_text(encoding='utf-8'))
        if any(group['grid'][k]!=grid[k] for k in ['width','height','crs','transform']):raise ValueError('Griglia originale diversa')
        for p in request['patches']:
            if request['groups'][p['group']]['tile']!=group['tile']:continue
            if not np.array_equal(flat[p['take']],p['flat_indices']) or int(unpack(p).sum())!=p['weight']:raise ValueError('Mapping tile/pixel/maschere/pesi diverso')
    return True


def cache(source,cfg,resume=False,deadline=None):
    root,_=roots(cfg);plan=Plan(root)
    if source.fingerprint!=plan.manifest['source']:raise ValueError('Sorgente diversa dal piano')
    if (root/'cache.json').exists():
        result=json.loads((root/'cache.json').read_text(encoding='utf-8'))
        if result['identity']!=plan.manifest['identity']:raise ValueError('Cache di un piano diverso')
        for b in result['shards']:load_arrays(root/'features'/b['key'],b['identity'],mmap=True)
        if digest(root/result['index']['file'])!=result['index']['sha256']:raise ValueError('Indice cache corrotto')
        return result
    own(root,plan.manifest['identity'],'sampling_cache');markers=[];stats={m:[] for m in cfg['sampling']['modes']};base=plan.frame.set_index('cell_id')
    progress=Progress(root/'progress.json');began=__import__('time').perf_counter();done=0
    for block,records in plan.blocks():
        check_deadline(deadline);folder=root/'features'/block['key'];tag=signature({'plan':plan.manifest['identity'],'block':block['key']})
        if folder.exists():arrays,info=load_arrays(folder,tag,mmap=True)
        else:
            vectors=[];stored=[];offset=0
            for cell in records:
                original=source.patches(cell['cell_id']);union={};items=[]
                for value in cell['modes'].values():
                    for p in value['patches']:
                        src=original[p['source_patch']];rr,cc=np.where(unpack(p));r=rr+p['row']-src.metadata['row'];c=cc+p['col']-src.metadata['col']
                        if not src.mask[r,c].all():raise ValueError('Piano contiene pixel invalidi')
                        x=src.features[:,r,c].T
                        indices=[]
                        for k,position in enumerate(p['positions']):
                            key=(p['tile'],position)
                            if key not in union:union[key]=offset+len(items);items.append(x[k].copy())
                            elif not np.array_equal(items[union[key]-offset],x[k]):raise ValueError('Contributo ripetuto con valori diversi')
                            indices.append(union[key])
                        p['cache_indices']=indices
                vectors.extend(items);offset+=len(items);stored.append(cell)
            features=np.asarray(vectors,dtype=np.float32)
            disk_budget(root,cfg['execution']['reserve_gib'],features.nbytes*3+len(blob(stored))*3)
            info=commit_arrays(folder,{'features':features,'records':blob(stored)},tag,{'union_pixels':len(features)})
            arrays={'features':features,'records':blob(stored)}
        for cell in unblob(arrays['records']):
            for mode,value in cell['modes'].items():
                indices=np.concatenate([p['cache_indices'] for p in value['patches']]);x=np.array(arrays['features'][indices],dtype=np.float64)
                row=base.loc[cell['cell_id']].to_dict();row.update(cell_id=cell['cell_id'],n_valid=len(x),mean=x.mean(0).astype(np.float32),
                    variance=x.var(0).astype(np.float32),patches=len(value['patches']),sample_insufficient=value['diagnostics']['insufficient'])
                stats[mode].append(row)
        del arrays
        markers.append({'key':block['key'],'identity':tag,'marker_sha256':digest(folder/'complete.json')})
        done+=len(records);rate=done/max(__import__('time').perf_counter()-began,1e-9)
        progress.update(force=True,phase='cache',cells_done=done,cells_total=len(plan.frame),cells_per_second=rate,eta_seconds=(len(plan.frame)-done)/rate)
    frames={}
    for mode,rows in stats.items():
        path=root/('moments_'+mode+'.parquet');atomic_frame(path,pd.DataFrame(rows));frames[mode]={'file':path.name,'sha256':digest(path)}
    result={'identity':plan.manifest['identity'],'source':source.fingerprint,'shards':markers,'frames':frames,'index':make_index(root,markers),
        'seconds_finalizing_invocation':__import__('time').perf_counter()-began}
    save_json(root/'cache.json',result)
    progress.update(force=True,phase='cache_complete',cells_done=len(plan.frame),cells_total=len(plan.frame),eta_seconds=0)
    return result


def make_index(root,blocks):
    path=Path(root)/('cache_index_'+uuid.uuid4().hex[:8]+'.sqlite3');tmp=unique_tmp(path)
    db=sqlite3.connect(tmp)
    try:
        db.execute('CREATE TABLE cells(cell_id TEXT,mode TEXT,shard TEXT,patches BLOB,PRIMARY KEY(cell_id,mode))')
        for block in blocks:
            arrays,_=load_arrays(Path(root)/'features'/block['key'],block['identity'],verify=False,mmap=True)
            for cell in unblob(arrays['records']):
                for mode,value in cell['modes'].items():
                    patches=[{k:p[k] for k in ['height','width','mask','weight','cache_indices']} for p in value['patches']]
                    db.execute('INSERT INTO cells VALUES(?,?,?,?)',(cell['cell_id'],mode,block['key'],blob(patches).tobytes()))
            del arrays
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Indice SQLite cache non integro')
    finally:db.close()
    replace_with_retry(tmp,path);return {'file':path.name,'sha256':digest(path)}


class CachedSource:
    def __init__(self,root,mode,limit=4):
        self.root=Path(root);self.mode=mode;self.limit=limit;self.maps=OrderedDict();self.db=None
        self.info=json.loads((self.root/'cache.json').read_text(encoding='utf-8'));record=self.info['frames'][mode]
        if digest(self.root/record['file'])!=record['sha256']:raise ValueError('Momenti cache modificati')
        self.frame=pd.read_parquet(self.root/record['file'])
        self.blocks={b['key']:b for b in self.info['shards']}
        if digest(self.root/self.info['index']['file'])!=self.info['index']['sha256']:raise ValueError('Indice cache modificato')
        for block in self.info['shards']:
            folder=self.root/'features'/block['key']
            if digest(folder/'complete.json')!=block['marker_sha256']:raise ValueError('Cache manifest modificato')
            # Shards are SHA-verified once at opening; subsequent reads use immutable mapped arrays.
            load_arrays(folder,block['identity'],mmap=True)
        self.fingerprint=signature({'cache':self.info,'mode':mode});self.feature_space_id='sampling_frozen_v2_128';self.channels=128
    def patches(self,cell_id):
        from biomap_spatial.contracts import SpatialSample
        if self.db is None:self.db=sqlite3.connect((self.root/self.info['index']['file']).resolve().as_uri()+'?mode=ro',uri=True)
        row=self.db.execute('SELECT shard,patches FROM cells WHERE cell_id=? AND mode=?',(cell_id,self.mode)).fetchone()
        if row is None:raise ValueError('Cella non presente nella configurazione')
        key,packed=row;patches=json.loads(zlib.decompress(packed).decode('utf-8'));block=self.blocks[key]
        if key not in self.maps:
            self.maps[key]=load_arrays(self.root/'features'/key,block['identity'],verify=False,mmap=True)[0]['features']
            while len(self.maps)>self.limit:self.maps.popitem(last=False)
        self.maps.move_to_end(key);array=self.maps[key];result=[]
        for p in patches:
            mask=unpack(p);x=np.zeros((128,p['height'],p['width']),np.float32);x[:,mask]=np.array(array[p['cache_indices']]).T
            result.append(SpatialSample(x,mask,{'feature_space_id':self.feature_space_id,**p}))
        return result
    def close(self):
        self.maps.clear()
        if self.db is not None:self.db.close();self.db=None
    def __getstate__(self):
        state=self.__dict__.copy();state['maps']=OrderedDict();state['db']=None;state['frame']=self.frame.iloc[:0];return state


def full_validation_cache(source,frame,cfg,deadline=None):
    """Bounded scratch: ONE validation group, original FULL contribution policy, never production files."""
    import shutil
    root=Path(cfg['paths']['output_root'])/'validation_scratch';tag=signature({'source':source.fingerprint,'cells':sorted(frame.cell_id)})
    mark=root/'ownership.json'
    if mark.exists():
        record=json.loads(mark.read_text(encoding='utf-8'))
        if record.get('protocol')!=PROTOCOL or record.get('kind')!='full_validation_scratch':raise ValueError('Scratch non posseduto: non viene rimosso')
        if record['identity']!=tag:
            # Caller holds workspace lock and closes old view first. Only derived scratch is reclaimed.
            if root.is_symlink():raise ValueError('Scratch symlink rifiutato')
            shutil.rmtree(root)
    own(root,tag,'full_validation_scratch')
    if (root/'cache.json').exists():return CachedSource(root,'FULL',cfg['cache']['mmap_limit'])
    estimated=int(frame.n_valid.sum())*128*4
    if estimated>cfg['cache']['validation_max_gib']*1024**3:raise OSError('Gruppo validation oltre il budget scratch: aumentare esplicitamente il limite con spazio verificato')
    disk_budget(root,cfg['execution']['reserve_gib'],estimated*1.05+16*1024**2);blocks=[];step=cfg['cache']['cells_per_shard']
    for low in range(0,len(frame),step):
        check_deadline(deadline);key=f'{low:07d}';folder=root/'features'/key;block_id=signature({'root':tag,'key':key})
        if not folder.exists():
            vectors=[];records=[];offset=0
            for row in frame.iloc[low:low+step].itertuples():
                patches=[]
                for patch in source.patches(row.cell_id):
                    x=patch.features[:,patch.mask].T;vectors.extend(x.copy());n=len(x)
                    patches.append({'height':patch.mask.shape[0],'width':patch.mask.shape[1],'mask':__import__('biomap_sampling.geometry',fromlist=['encode_mask']).encode_mask(patch.mask),
                        'weight':n,'cache_indices':list(range(offset,offset+n))});offset+=n
                if sum(p['weight'] for p in patches)!=row.n_valid:raise ValueError('Validation full: supporto perso')
                records.append({'cell_id':row.cell_id,'modes':{'FULL':{'patches':patches}}})
            commit_arrays(folder,{'features':np.asarray(vectors,np.float32),'records':blob(records)},block_id)
        else:load_arrays(folder,block_id,mmap=True)
        blocks.append({'key':key,'identity':block_id,'marker_sha256':digest(folder/'complete.json')})
    path=root/'moments_FULL.parquet';atomic_frame(path,frame)
    save_json(root/'cache.json',{'identity':tag,'source':source.fingerprint,'shards':blocks,'frames':{'FULL':{'file':path.name,'sha256':digest(path)}},'index':make_index(root,blocks)})
    return CachedSource(root,'FULL',cfg['cache']['mmap_limit'])
