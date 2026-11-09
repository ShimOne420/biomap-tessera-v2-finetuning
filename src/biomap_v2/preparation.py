# SCOPO
#   Motore di acquisizione "optimized" di prepare: produce per ogni cella gli stessi series.npz e
#   manifest.json del percorso originale, riorganizzando soltanto letture e salvataggi.
#
# COME VIENE RICHIAMATO
#   Non si esegue da solo. data.prepare() -> data._prepare_run() -> acquire_optimized(), quando il YAML
#   ha acquisition.engine=optimized. Comandi utente:
#     python scripts/probe_msa.py prepare --config configs/msa_v1_liguria.yaml --data-root ..\data
#     python scripts/probe_msa.py prepare --config ... --data-root ... --aoi belluno_dolomiti
#     python scripts/probe_msa.py prepare --config ... --data-root ... --parallel-aois 15
#
# LOGICA, IN ORDINE
#   1. acquire_optimized valida le celle già complete (checksum e identità); non le riscrive mai.
#   2. Le celle mancanti si ordinano per vicinanza, si dividono per area e in lotti di batch_cells.
#   3. resume_completed_checkpoints finisce di salvare i lotti che avevano già tutti i giorni.
#   4. batch() crea una attività per ogni coppia sorgente/giorno (s2, asc, desc), eseguita in day_pool.
#   5. day() fa il mosaicking di quel giorno: scene ordinate per nuvolosità e id, arresto per singola
#      cella appena tutti i suoi punti sono validi. Le bande di una scena si leggono in parallelo in pool.
#   6. day_task salva il giorno in un solo file batch_checkpoints/<area>/<hash>/<sorgente>_<giorno>.npz.
#   7. consolidate rilegge i giorni a gruppi di CONSOLIDATION_CELLS, ricostruisce le serie annuali e chiama
#      finish_cell, che filtra i punti, scrive in modo atomico e rilegge per verifica.
#   8. A cella salvata si rimuovono checkpoint ed eventuali intermedi per cella del vecchio formato.
#
# ECCEZIONE DI COPERTURA (acquisition.low_coverage_cells)
#   Una cella sotto sampling.min_valid_fraction viene esclusa con excluded.json, che di norma è definitivo.
#   Se il suo cell_id è in acquisition.low_coverage_cells:
#   - completed_cell non considera definitivo excluded.json e rimette la cella tra quelle da acquisire;
#   - finish_cell la salva con i soli pixel validi (almeno uno), con low_coverage_accepted=true nel manifest,
#     e rinomina excluded.json in excluded.superseded.json per conservarne la storia.
#   Punti, scene e mosaicking restano identici: cambia solo la decisione finale di tenere la cella.
#
# PERCHÉ IL RISULTATO NON CAMBIA
#   Lotti, thread e processi decidono solo l'ordine dell'I/O. Punti, scene, ordine del mosaicking,
#   maschere, conversioni e ordine dei giorni nella serie restano gli stessi. Provato da
#   tests/test_preparation_performance.py e, su celle reali, da scripts/verify_prepare_equivalence.py.
#
# INPUT E OUTPUT
#   Input: celle e cataloghi congelati in <data-root>/<namespace>/catalogs, parametri del YAML.
#   Output: series/<cella>/series.npz e manifest.json, oppure excluded.json se la copertura è insufficiente.
#
# PARAMETRI (sezione acquisition del YAML, modificabili con --set)
#   batch_cells: celle per lotto; con un'area intera ogni blocco raster si legge una sola volta.
#   day_workers: giorni elaborati insieme. io_workers: bande lette insieme.
#
# RISORSE
#   Thread limitati dai due pool; poca RAM perché i giorni completati stanno su disco; nessuna GPU.
#
# RIPRESA DOPO UN'INTERRUZIONE
#   Rilanciare lo stesso comando: celle complete, checkpoint giornalieri e intermedi per cella del vecchio
#   formato vengono riusati. preparation_locks impedisce due prepare concorrenti sulle stesse celle.
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION,ThreadPoolExecutor,wait
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import numpy as np
import rasterio
from .common import digest,root

CONSOLIDATION_CELLS=512


@contextmanager
def preparation_lock(cfg,name='prepare'):
    directory=root(cfg);directory.mkdir(parents=True,exist_ok=True)
    handle=(directory/f'{name}.lock').open('a+b')
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


@contextmanager
def preparation_locks(cfg,names):
    # A full prepare holds every area lock, so it can never overlap a per-area worker.
    if not names:
        yield;return
    with preparation_lock(cfg,names[0]):
        with preparation_locks(cfg,names[1:]):yield


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


def cell_identity(cfg,cell):
    from .data import PROTOCOL
    return {'protocol':PROTOCOL,'year':cfg['year'],'sampling':cfg['sampling'],
        'bounds':[cell.west,cell.south,cell.east,cell.north]}


def matches_identity(cfg,cell,info):
    expected=cell_identity(cfg,cell)
    return info.get('protocol')==expected['protocol'] and info.get('year')==expected['year'] and info.get('sampling')==expected['sampling'] and np.allclose(info.get('bounds',[]),expected['bounds'],rtol=0,atol=1e-10)


def completed_cell(cfg,cell,directory,guard,profile):
    path=directory/'series.npz';meta=directory/'manifest.json';recover_pair(path,meta,guard,profile)
    if not path.exists():
        excluded=directory/'excluded.json'
        # Exclusions recorded with their identity are final: intermediates are no longer retained for them.
        if excluded.exists():
            info=json.loads(excluded.read_text(encoding='utf-8'))
            waived=cell.cell_id in set(cfg['acquisition'].get('low_coverage_cells',[]))
            # A cell listed in low_coverage_cells is acquired again and kept with its valid pixels.
            if 'protocol' in info and matches_identity(cfg,cell,info) and not waived:return {'fraction':None,'sha256':None,'cached':True}
        return None
    with profile.span('cell_cache_validation'):
        info=json.loads(meta.read_text(encoding='utf-8'))
        if not matches_identity(cfg,cell,info):
            raise ValueError(f'Cache incompatibile per {cell.cell_id}: non viene modificata')
        if digest(path)!=info['sha256']:raise ValueError(f'Cache cella corrotta: {path}')
    return {'fraction':info['valid_fraction'],'sha256':info['sha256'],'cached':True}


class BatchAcquirer:
    def __init__(self,cfg,profile,guard):
        from .data import Reader
        self.cfg=cfg;self.profile=profile;self.guard=guard;self.catalogs={}
        self.reader=Reader(cfg,profile,guard);self.stopping=threading.Event()
        # GDAL_CACHEMAX is process-global: set/restore it once, never from overlapping workers.
        self.environment=rasterio.Env(GDAL_CACHEMAX=64*1024**2)
        self.environment.__enter__()
        self.pool=ThreadPoolExecutor(max_workers=cfg['acquisition']['io_workers'],thread_name_prefix='sentinel-io')
        # Day tasks wait on band tasks, so they need their own pool to exclude deadlocks.
        self.day_pool=ThreadPoolExecutor(max_workers=cfg['acquisition']['day_workers'],thread_name_prefix='sentinel-day')

    def close(self):
        try:
            self.stopping.set()
            self.day_pool.shutdown(wait=True,cancel_futures=True)
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

    def day(self,directories,names,source,day,items,points):
        from .data import BANDS,power_to_storage
        n=self.cfg['sampling']['pixels_per_cell'];width=10 if source=='s2' else 2
        dtype=np.uint16 if source=='s2' else np.int16
        total=len(directories);full=np.zeros((total,n,width),dtype=dtype);full_valid=np.zeros((total,n),dtype=bool)
        missing=[];scene_ids=[item['id'] for item in items];stem=f'{source}_{day}'
        for i,directory in enumerate(directories):
            # Only cells interrupted under the earlier format still hold per-day files.
            present=names[i] and ({stem+'.npz',stem+'.json',stem+'.json.pending'}&names[i])
            if not present:
                missing.append(i);continue
            path=directory/f'{stem}.npz';meta=path.with_suffix('.json');recover_pair(path,meta,self.guard,self.profile)
            if path.exists():
                with self.profile.span('day_cache_read'):
                    info=json.loads(meta.read_text(encoding='utf-8'))
                    if info['sha256']!=digest(path) or info['scene_ids']!=scene_ids:raise ValueError(f'Cache giorno corrotta/incompatibile: {path}')
                    with np.load(path,allow_pickle=False) as arrays:arr=arrays['bands'];valid=arrays['valid']
                    if arr.shape!=(n,width) or arr.dtype!=dtype or valid.shape!=(n,) or not np.isin(valid,[0,1]).all():raise ValueError(f'Cache giorno non valida: {path}')
                full[i]=arr;full_valid[i]=valid;self.profile.count('day_cache_hits')
            else:missing.append(i)
        if not missing:return full,full_valid
        arr=np.zeros((len(missing),n,width),dtype=dtype);valid=np.zeros((len(missing),n),dtype=bool)
        for item in sorted(items,key=lambda i:(i['properties'].get('eo:cloud_cover',0),i['id'])):
            active=np.flatnonzero(~valid.all(axis=1))
            if not len(active):break
            if self.stopping.is_set():raise RuntimeError('Acquisizione interrotta: checkpoint conservati')
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
        full[missing]=arr;full_valid[missing]=valid
        return full,full_valid

    def checkpoint_base(self,aoi):
        return root(self.cfg)/'batch_checkpoints'/aoi

    def checkpoint_identity(self):
        from .data import PROTOCOL
        return {'protocol':PROTOCOL,'year':self.cfg['year'],'sampling':self.cfg['sampling']}

    def checkpoint(self,cells):
        ids=[cell.cell_id for cell in cells]
        identity=self.checkpoint_identity()
        name=hashlib.sha256(json.dumps([identity,ids],sort_keys=True).encode()).hexdigest()[:20]
        directory=self.checkpoint_base(cells[0].aoi_id)/name;manifest=directory/'manifest.json'
        if not manifest.exists():self.guard.json(manifest,{**identity,'cell_ids':ids})
        return directory,ids

    def day_task(self,checkpoint,directories,names,source,day,items,points):
        path=checkpoint/f'{source}_{day}.npz'
        if path.exists():
            self.profile.count('batch_day_hits');return
        bands,valid=self.day(directories,names,source,day,items,points)
        self.guard.npz(path,bands=bands,valid=valid,scene_ids=np.asarray([item['id'] for item in items]))

    def batch(self,cells,directories):
        from .data import sample_points
        pixels=self.cfg['sampling']['pixels_per_cell']
        points=[sample_points(cell,pixels) for cell in cells]
        streams=self.streams(cells[0],directories[0])
        checkpoint,ids=self.checkpoint(cells)
        names=[set(os.listdir(directory)) if directory.exists() else set() for directory in directories]
        tasks=[(source,day,items) for source,days in streams.items() for day,items in sorted(days.items())]
        total=len(tasks);aoi=cells[0].aoi_id
        self.profile.batch_days(0,total,len(cells))
        self.profile.stage(f'{aoi} giorni 0/{total}, batch={len(cells)}')
        futures={self.day_pool.submit(self.day_task,checkpoint,directories,names,source,day,items,points) for source,day,items in tasks}
        outstanding=set(futures)
        try:
            # Short waits keep Ctrl+C responsive on Windows; the first failure stops the batch.
            while outstanding:
                done,outstanding=wait(outstanding,timeout=1,return_when=FIRST_EXCEPTION)
                for future in done:future.result()
                finished=total-len(outstanding)
                self.profile.batch_days(finished,total,len(cells))
                self.profile.stage(f'{aoi} giorni {finished}/{total}, batch={len(cells)}')
        except BaseException:
            self.stopping.set()
            for future in futures:future.cancel()
            raise
        self.guard.json(checkpoint/'days_complete.json',{'days':total})
        self.profile.stage(f'{aoi} salvataggio celle, batch={len(cells)}')
        result=self.consolidate(cells,directories,checkpoint,ids,streams)
        self.remove_checkpoint(checkpoint)
        return result

    def consolidate(self,cells,directories,checkpoint,ids,streams):
        from .data import sample_points
        n=self.cfg['sampling']['pixels_per_cell'];index={cell_id:i for i,cell_id in enumerate(ids)};result={}
        manifest={source:[{'date':day,'scene_ids':[item['id'] for item in days[day]]} for day in sorted(days)] for source,days in streams.items()}
        doys={source:np.asarray([min(datetime.fromisoformat(day).timetuple().tm_yday,365) for day in sorted(days)],dtype=np.int16) for source,days in streams.items()}
        for start in range(0,len(cells),CONSOLIDATION_CELLS):
            chunk=cells[start:start+CONSOLIDATION_CELLS];positions=np.asarray([index[cell.cell_id] for cell in chunk])
            outputs=[{} for _ in chunk]
            for source,days in streams.items():
                width=10 if source=='s2' else 2;bands=[];masks=[]
                for day in sorted(days):
                    path=checkpoint/f'{source}_{day}.npz'
                    with np.load(path,allow_pickle=False) as stored:
                        if stored['scene_ids'].tolist()!=[item['id'] for item in days[day]]:raise ValueError(f'Checkpoint incompatibile col catalogo: {path}')
                        if stored['bands'].shape!=(len(ids),n,width) or stored['valid'].shape!=(len(ids),n):raise ValueError(f'Checkpoint non valido: {path}')
                        bands.append(stored['bands'][positions]);masks.append(stored['valid'][positions])
                stacked=np.stack(bands,axis=1) if bands else None
                stacked_mask=np.stack(masks,axis=1) if masks else None
                for i in range(len(chunk)):
                    outputs[i][source]=stacked[i].transpose(1,0,2) if bands else np.zeros((n,0,width),dtype=np.float32)
                    outputs[i][source+'_doys']=doys[source]
                    if source=='s2':outputs[i]['mask']=stacked_mask[i].T if masks else np.zeros((n,0),dtype=bool)
            for i,cell in enumerate(chunk):
                directory=directories[start+i];points=sample_points(cell,n)
                result[cell.cell_id]=self.finish_cell(cell,directory,outputs[i],points,manifest,streams)
                self.profile.cell(excluded=result[cell.cell_id]['fraction'] is None)
            self.profile.flush()
        return result

    def remove_checkpoint(self,checkpoint):
        if not checkpoint.exists():return
        for path in list(checkpoint.iterdir()):
            if path.is_file():self.guard.remove_owned(path)
        try:checkpoint.rmdir()
        except OSError:self.profile.count('checkpoint_cleanup_errors')

    def resume_completed_checkpoints(self,group,cell_directory):
        # Consolidation starts after every day is complete, so it may resume on a subset of the batch.
        base=self.checkpoint_base(group[0].aoi_id);results={}
        if not base.exists():return results
        identity=self.checkpoint_identity()
        for checkpoint in sorted(path for path in base.iterdir() if path.is_dir()):
            manifest=checkpoint/'manifest.json'
            if not manifest.exists() or not (checkpoint/'days_complete.json').exists():continue
            info=json.loads(manifest.read_text(encoding='utf-8'))
            if any(info.get(key)!=value for key,value in identity.items()):continue
            ids=info['cell_ids'];known=set(ids)
            subset=[cell for cell in group if cell.cell_id in known and cell.cell_id not in results]
            if subset:
                streams=self.streams(subset[0],cell_directory(self.cfg,subset[0].cell_id))
                results.update(self.consolidate(subset,[cell_directory(self.cfg,cell.cell_id) for cell in subset],checkpoint,ids,streams))
            self.remove_checkpoint(checkpoint)
        return results

    def clear_area_checkpoints(self,aoi):
        base=self.checkpoint_base(aoi)
        if not base.exists():return
        for checkpoint in list(base.iterdir()):
            if checkpoint.is_dir():self.remove_checkpoint(checkpoint)
        try:base.rmdir()
        except OSError:self.profile.count('checkpoint_cleanup_errors')

    def finish_cell(self,cell,directory,outputs,points,manifest,streams):
        from .data import PROTOCOL,BANDS
        with self.profile.span('preprocessing'):
            good=outputs['mask'].sum(axis=1)>=self.cfg['sampling']['min_s2_observations']
            good &= (np.any(outputs['asc']!=0,axis=-1).sum(axis=1)+np.any(outputs['desc']!=0,axis=-1).sum(axis=1))>=self.cfg['sampling']['min_s1_observations']
            fraction=float(good.mean())
        from .common import low_coverage_waived
        waived=fraction<self.cfg['sampling']['min_valid_fraction'] and low_coverage_waived(self.cfg,cell.cell_id,fraction)
        if fraction<self.cfg['sampling']['min_valid_fraction'] and not waived:
            self.guard.json(directory/'excluded.json',{'reason':'insufficient_sentinel_coverage','valid_fraction':fraction,**cell_identity(self.cfg,cell)})
            return {'fraction':None,'sha256':None,'cached':False}
        for name in ['s2','mask','asc','desc']:outputs[name]=outputs[name][good]
        outputs['points']=points[good]
        meta={'protocol':PROTOCOL,'year':self.cfg['year'],'pixels_per_cell':self.cfg['sampling']['pixels_per_cell'],
            'sampling':self.cfg['sampling'],'bounds':[cell.west,cell.south,cell.east,cell.north],
            'valid_fraction':fraction,'band_order':BANDS,'doy_policy':'actual day of year capped at 365',
            'sar_units':'MPC intensity -> 10log10 -> +50 -> *200 int16','scene_manifest':manifest,'io_engine':'batched_native_v1'}
        if waived:meta['low_coverage_accepted']=True
        info=atomic_pair(directory/'series.npz',directory/'manifest.json',outputs,meta,self.guard,self.profile)
        if waived and (directory/'excluded.json').exists():
            self.guard.replace(directory/'excluded.json',directory/'excluded.superseded.json')
        with self.profile.span('series_verification'):
            with np.load(directory/'series.npz',allow_pickle=False) as verified:
                for key,value in outputs.items():
                    if not np.array_equal(verified[key],value):raise ValueError('Consolidamento serie non verificato')
        # Remove only per-day files left by an interrupted run of the earlier format.
        leftovers=[name for name in os.listdir(directory) if name.endswith(('.npz','.json')) and name.startswith(('s2_','asc_','desc_'))]
        for name in leftovers:
            source,_,day=name.rpartition('.')[0].partition('_')
            if day in streams.get(source,{}):self.guard.remove_owned(directory/name)
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
        if info is not None:results[cell.cell_id]=info;profile.cell(cached=True,excluded=info['fraction'] is None)
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
        for aoi,group in groups.items():
            resumed=reader.resume_completed_checkpoints(group,cell_directory);results.update(resumed)
            group=[cell for cell in group if cell.cell_id not in resumed]
            for start in range(0,len(group),size):
                batch=group[start:start+size]
                results.update(reader.batch(batch,[cell_directory(cfg,cell.cell_id) for cell in batch]))
            # Batches interrupted under another lot size can no longer be matched: their cells are complete.
            reader.clear_area_checkpoints(aoi)
    except BaseException:
        profile.stage('stopping; valid cache preserved');raise
    finally:reader.close()
    return results
