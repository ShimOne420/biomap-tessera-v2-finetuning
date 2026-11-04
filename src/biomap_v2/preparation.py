# Scopo: preparare lotti di celle vicine mantenendo mosaicking e serie del percorso originale.
# Fasi: valida cache esistenti -> raggruppa solo l'I/O -> legge bande in parallelo -> salva giorni/celle atomici.
# Input: identiche celle/25 punti, cataloghi, YAML; output: stesso series.npz/manifest.json, con profiling separato.
# Parametri: acquisition.engine=optimized, batch_cells=16, io_workers=4; --import-cache resta compatibile.
# Risorse: thread I/O limitati, cache nativa RAM/disco; nessuna operazione GPU in prepare.
# Ripresa: checksum, transazioni pending, riuso giorni parziali e blocco OS contro prepare concorrenti.
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import rasterio
from .common import digest,root


@contextmanager
def preparation_lock(cfg):
    directory=root(cfg);directory.mkdir(parents=True,exist_ok=True)
    handle=(directory/'prepare.lock').open('a+b')
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('Un altro prepare usa questa data-root: fermarlo prima di riprendere') from None
    try:yield
    finally:
        handle.seek(0)
        if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


def atomic_pair(path,meta,arrays,info,guard,profile):
    stage=path.with_name(path.name+'.stage');pending=meta.with_name(meta.name+'.pending')
    guard.npz(stage,**arrays)
    with profile.span('checksum'):info={**info,'sha256':digest(stage)}
    guard.json(pending,info)
    guard.replace(stage,path);guard.replace(pending,meta)
    return info


def recover_pair(path,meta,guard,profile):
    pending=meta.with_name(meta.name+'.pending');stage=path.with_name(path.name+'.stage')
    if pending.exists():
        info=json.loads(pending.read_text(encoding='utf-8'))
        candidate=path if path.exists() else stage
        with profile.span('checksum'):
            if not candidate.exists() or digest(candidate)!=info['sha256']:
                raise ValueError(f'Transazione cache incompleta/corrotta: {candidate}')
        if candidate==stage:guard.replace(stage,path)
        guard.replace(pending,meta);profile.count('cache_transactions_recovered')
    if path.exists() and not meta.exists():raise ValueError(f'Cache senza manifest: {path}; nessun dato sostituito')
    if meta.exists() and not path.exists():raise ValueError(f'Manifest senza dati: {path}')


def completed_cell(cfg,cell,directory,guard,profile):
    from .data import PROTOCOL
    path=directory/'series.npz';meta=directory/'manifest.json';recover_pair(path,meta,guard,profile)
    if not path.exists():return None
    with profile.span('cell_cache_validation'):
        info=json.loads(meta.read_text(encoding='utf-8'))
        if info['protocol']!=PROTOCOL or info['year']!=cfg['year'] or info.get('sampling')!=cfg['sampling'] or not np.allclose(info.get('bounds',[]),[cell.west,cell.south,cell.east,cell.north],rtol=0,atol=1e-10):
            raise ValueError(f'Cache incompatibile per {cell.cell_id}: non viene modificata')
        if digest(path)!=info['sha256']:raise ValueError(f'Cache cella corrotta: {path}')
    return {'fraction':info['valid_fraction'],'sha256':info['sha256'],'cached':True}


class BatchAcquirer:
    def __init__(self,cfg,profile,guard):
        from .data import Reader
        self.cfg=cfg;self.profile=profile;self.guard=guard;self.catalogs={}
        self.reader=Reader(cfg,profile,guard)
        # GDAL_CACHEMAX is process-global: set/restore it once, never from overlapping workers.
        self.environment=rasterio.Env(GDAL_CACHEMAX=64*1024**2)
        self.environment.__enter__()
        self.pool=ThreadPoolExecutor(max_workers=cfg['acquisition']['io_workers'],thread_name_prefix='sentinel-io')

    def close(self):
        try:
            self.pool.shutdown(wait=True,cancel_futures=True);self.reader.close()
        finally:self.environment.__exit__(None,None,None)

    def streams(self,cell,directory):
        from . import data
        key=cell.aoi_id
        if key in self.catalogs:return self.catalogs[key]
        streams={'s2':{},'asc':{},'desc':{}};fingerprints={}
        for collection in ['sentinel-2-l2a','sentinel-1-rtc']:
            cache=root(self.cfg)/'catalogs'/key/f'{self.cfg["year"]}_{collection}.json'
            previous=cache.stat().st_size if cache.exists() else 0
            stage='catalog_cache_load' if cache.exists() else 'stac_query'
            with self.profile.span(stage):items=data.query(self.cfg,cell,collection,directory/f'{collection}.json')
            self.profile.count(stage+'_calls')
            if cache.exists():
                self.guard.account(cache.stat().st_size-previous)
                fingerprints[collection]=digest(cache)
            for item in items:
                properties=item['properties'];day=properties['datetime'][:10]
                source='s2' if collection=='sentinel-2-l2a' else ('asc' if properties.get('sat:orbit_state')=='ascending' else 'desc')
                streams[source].setdefault(day,[]).append(item)
        self.catalogs[key]=streams
        self.guard.json(self.profile.directory/f'catalog_{key}.json',{'year':self.cfg['year'],'sha256':fingerprints})
        return streams

    def read_bands(self,item,bands,points):
        self.profile.count('band_tasks',len(bands))
        futures=[self.pool.submit(self.reader.values,item,band,points) for band in bands]
        try:return [future.result() for future in futures]
        except BaseException:
            for future in futures:future.cancel()
            raise

    def day(self,cells,directories,source,day,items,points):
        from .data import BANDS,power_to_storage
        n=self.cfg['sampling']['pixels_per_cell'];width=10 if source=='s2' else 2
        dtype=np.uint16 if source=='s2' else np.int16
        result={};missing=[];scene_ids=[item['id'] for item in items]
        for i,directory in enumerate(directories):
            path=directory/f'{source}_{day}.npz';meta=path.with_suffix('.json');recover_pair(path,meta,self.guard,self.profile)
            if path.exists():
                with self.profile.span('day_cache_read'):
                    info=json.loads(meta.read_text(encoding='utf-8'))
                    if info['sha256']!=digest(path) or info['scene_ids']!=scene_ids:raise ValueError(f'Cache giorno corrotta/incompatibile: {path}')
                    with np.load(path,allow_pickle=False) as arrays:arr=arrays['bands'];valid=arrays['valid']
                    if arr.shape!=(n,width) or arr.dtype!=dtype or valid.shape!=(n,) or not np.isin(valid,[0,1]).all():raise ValueError(f'Cache giorno non valida: {path}')
                result[i]=(arr,valid);self.profile.count('day_cache_hits')
            else:missing.append(i)
        if not missing:return result
        arr=np.zeros((len(missing),n,width),dtype=dtype);valid=np.zeros((len(missing),n),dtype=bool)
        for item in sorted(items,key=lambda i:(i['properties'].get('eo:cloud_cover',0),i['id'])):
            active=np.flatnonzero(~valid.all(axis=1))
            if not len(active):break
            selected=np.concatenate([points[missing[int(i)]] for i in active],axis=0)
            if source=='s2':
                values=self.read_bands(item,['SCL',*BANDS],selected)
                with self.profile.span('preprocessing'):
                    scl=values[0].reshape(len(active),n)
                    raw=np.stack(values[1:],axis=1).reshape(len(active),n,width)
                    clear=np.isfinite(scl)&~np.isin(scl,[0,1,2,3,8,9])
                    take=clear&np.isfinite(raw).all(axis=2)&~valid[active]
                    current=arr[active].copy();current[take]=np.clip(raw[take],0,65535).astype(np.uint16)
            else:
                values=self.read_bands(item,['vv','vh'],selected)
                with self.profile.span('preprocessing'):
                    raw=np.stack(values,axis=1).reshape(len(active),n,width)
                    take=np.isfinite(raw).all(axis=2)&(raw>0).all(axis=2)&~valid[active]
                    current=arr[active].copy();current[take]=power_to_storage(raw[take])
            arr[active]=current;valid[active]|=take
        for j,i in enumerate(missing):
            path=directories[i]/f'{source}_{day}.npz';meta=path.with_suffix('.json')
            atomic_pair(path,meta,{'bands':arr[j],'valid':valid[j]}, {'scene_ids':scene_ids},self.guard,self.profile)
            result[i]=(arr[j],valid[j])
        return result

    def batch(self,cells,directories):
        from .data import sample_points
        points=[sample_points(cell,self.cfg['sampling']['pixels_per_cell']) for cell in cells]
        streams=self.streams(cells[0],directories[0]);outputs=[{} for _ in cells];manifest={}
        n=self.cfg['sampling']['pixels_per_cell']
        for source,days in streams.items():
            bands=[[] for _ in cells];masks=[[] for _ in cells];doys=[];manifest[source]=[]
            for day,items in sorted(days.items()):
                self.profile.stage(f'{cells[0].aoi_id} {source} {day}, batch={len(cells)}')
                result=self.day(cells,directories,source,day,items,points)
                for i in range(len(cells)):bands[i].append(result[i][0]);masks[i].append(result[i][1])
                doys.append(min(datetime.fromisoformat(day).timetuple().tm_yday,365))
                manifest[source].append({'date':day,'scene_ids':[item['id'] for item in items]})
            for i in range(len(cells)):
                outputs[i][source]=np.stack(bands[i],axis=1) if bands[i] else np.zeros((n,0,10 if source=='s2' else 2),dtype=np.float32)
                outputs[i][source+'_doys']=np.asarray(doys,dtype=np.int16)
                if source=='s2':outputs[i]['mask']=np.stack(masks[i],axis=1) if masks[i] else np.zeros((n,0),dtype=bool)
        result={}
        for i,cell in enumerate(cells):
            result[cell.cell_id]=self.finish_cell(cell,directories[i],outputs[i],points[i],manifest,streams)
            self.profile.cell(excluded=result[cell.cell_id]['fraction'] is None)
        self.profile.flush();return result

    def finish_cell(self,cell,directory,outputs,points,manifest,streams):
        from .data import PROTOCOL,BANDS
        with self.profile.span('preprocessing'):
            good=outputs['mask'].sum(axis=1)>=self.cfg['sampling']['min_s2_observations']
            good &= (np.any(outputs['asc']!=0,axis=-1).sum(axis=1)+np.any(outputs['desc']!=0,axis=-1).sum(axis=1))>=self.cfg['sampling']['min_s1_observations']
            fraction=float(good.mean())
        if fraction<self.cfg['sampling']['min_valid_fraction']:
            self.guard.json(directory/'excluded.json',{'reason':'insufficient_sentinel_coverage','valid_fraction':fraction})
            return {'fraction':None,'sha256':None,'cached':False}
        for name in ['s2','mask','asc','desc']:outputs[name]=outputs[name][good]
        outputs['points']=points[good]
        meta={'protocol':PROTOCOL,'year':self.cfg['year'],'pixels_per_cell':self.cfg['sampling']['pixels_per_cell'],
            'sampling':self.cfg['sampling'],'bounds':[cell.west,cell.south,cell.east,cell.north],
            'valid_fraction':fraction,'band_order':BANDS,'doy_policy':'actual day of year capped at 365',
            'sar_units':'MPC intensity -> 10log10 -> +50 -> *200 int16','scene_manifest':manifest,'io_engine':'batched_native_v1'}
        info=atomic_pair(directory/'series.npz',directory/'manifest.json',outputs,meta,self.guard,self.profile)
        with self.profile.span('series_verification'):
            with np.load(directory/'series.npz',allow_pickle=False) as verified:
                for key,value in outputs.items():
                    if not np.array_equal(verified[key],value):raise ValueError('Consolidamento serie non verificato')
        for source,days in streams.items():
            for day in days:
                path=directory/f'{source}_{day}.npz'
                self.guard.remove_owned(path);self.guard.remove_owned(path.with_suffix('.json'))
        return {'fraction':fraction,'sha256':info['sha256'],'cached':False}


def acquire_optimized(cfg,frame,args,profile):
    from .data import cell_directory
    from .preparation_profile import SpaceGuard
    guard=SpaceGuard(cfg,profile);results={};pending=[]
    profile.stage('validating completed cells')
    for cell in frame.itertuples():
        directory=cell_directory(cfg,cell.cell_id);directory.mkdir(parents=True,exist_ok=True)
        if args.import_cache and not (directory/'series.npz').exists():
            source=Path(args.import_cache)/cfg['indicator']/'series'/directory.name
            if source.resolve()==directory.resolve():raise ValueError('Import cache coincide con la destinazione')
            if source.exists():
                size=sum(p.stat().st_size for p in source.rglob('*') if p.is_file());guard.check(size)
                before=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
                shutil.copytree(source,directory,dirs_exist_ok=True)
                guard.account(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())-before)
        info=completed_cell(cfg,cell,directory,guard,profile)
        if info is not None:results[cell.cell_id]=info;profile.cell(cached=True)
        else:pending.append(cell)
    if not pending:return results
    profile.start_work();profile.stage('acquiring observations')
    reader=BatchAcquirer(cfg,profile,guard)
    try:
        # I/O-only ordering: final CSV is restored to original source order by data.prepare.
        pending.sort(key=lambda c:(c.aoi_id,int(c.lon/.02),int(c.lat/.02),c.lon,c.lat,c.cell_id))
        groups=defaultdict(list)
        for cell in pending:groups[cell.aoi_id].append(cell)
        size=cfg['acquisition']['batch_cells']
        for _,group in groups.items():
            for start in range(0,len(group),size):
                batch=group[start:start+size]
                results.update(reader.batch(batch,[cell_directory(cfg,cell.cell_id) for cell in batch]))
    except BaseException:
        profile.stage('stopping; valid cache preserved');raise
    finally:reader.close()
    return results
