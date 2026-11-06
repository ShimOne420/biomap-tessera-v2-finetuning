# SCOPO
#   Misura il lavoro di prepare (tempi, contatori, avanzamento) e controlla che le scritture restino
#   entro il budget disco. Non influisce sui valori delle serie.
#
# COME VIENE RICHIAMATO
#   Non si esegue da solo. data.prepare crea PreparationProfile(cfg, celle, etichetta_area);
#   preparation.acquire_optimized crea SpaceGuard(cfg, profilo) e lo passa a chi scrive su disco.
#
# LOGICA DI PreparationProfile
#   1. Crea diagnostics/<timestamp>[_<area>]_prepare e salva la configurazione effettiva.
#   2. span() somma i tempi per stadio; count() somma i contatori; cell() conta le celle concluse.
#   3. batch_days() registra i giorni completati nel lotto: con un lotto per area le celle si salvano solo
#      alla fine, quindi è questa la misura di avanzamento che legge parallel_prepare.
#   4. Un thread scrive progress.json e profile.json ogni progress_seconds e stampa una riga di stato.
#   5. finish() registra lo stato finale e scrive assets.csv. Dopo un'uccisione forzata lo stato resta
#      "running" e assets.csv manca.
#
# LOGICA DI SpaceGuard
#   1. All'avvio somma una volta i file della data-root; poi aggiorna il totale a ogni scrittura.
#   2. check() ferma il comando se si supera storage.max_gib o se lo spazio libero scende sotto reserve_gib.
#   3. storage.max_new_gib, impostato da --max-new-gib, assegna una quota fissa a ogni processo parallelo:
#      i processi non si vedono tra loro, ma la somma delle quote resta entro il limite globale.
#   4. json() e npz() scrivono su un file .partial e poi lo sostituiscono in modo atomico.
#
# INPUT E OUTPUT
#   Input: configurazione e numero di celle. Output: progress.json, profile.json, config.json, assets.csv.
#
# PARAMETRI
#   acquisition.progress_seconds, storage.max_gib, storage.reserve_gib, storage.max_new_gib.
#
# RISORSE
#   Un thread leggero; nessuna GPU; nessun URL firmato nei report.
#
# RIPRESA
#   Ogni esecuzione ha la propria cartella di report e ricostruisce il totale dal disco.
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import shutil
import threading
import time
import numpy as np
from .common import root, save_json


class PreparationProfile:
    def __init__(self,cfg,total,label=None):
        self.cfg=cfg;self.total=total;self.started=time.perf_counter();self.work_started=None
        self.completed=0;self.cached=0;self.new=0;self.excluded=0;self.phase='initializing';self.status='running'
        self.days_done=0;self.days_total=0;self.days_cells=0
        self.timings=defaultdict(lambda:{'seconds':0.,'calls':0});self.counters=defaultdict(int);self.assets={}
        self.lock=threading.RLock();self.writer=threading.RLock();self.stop=threading.Event()
        suffix=f'_{label}_prepare' if label else '_prepare'
        self.directory=root(cfg)/'diagnostics'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+suffix)
        self.directory.mkdir(parents=True,exist_ok=True)
        save_json(self.directory/'config.json',cfg)
        self.interval=cfg.get('acquisition',{}).get('progress_seconds',10)
        self.thread=threading.Thread(target=self._heartbeat,daemon=True);self.thread.start()

    @contextmanager
    def span(self,stage,asset=None):
        started=time.perf_counter()
        try:yield
        finally:
            elapsed=time.perf_counter()-started
            with self.lock:
                self.timings[stage]['seconds']+=elapsed;self.timings[stage]['calls']+=1
                if asset:
                    row=self.assets.setdefault(asset,{'asset_key':asset})
                    row[stage+'_seconds']=row.get(stage+'_seconds',0.)+elapsed

    def count(self,name,n=1,asset=None):
        with self.lock:
            self.counters[name]+=n
            if asset:
                row=self.assets.setdefault(asset,{'asset_key':asset});row[name]=row.get(name,0)+n

    def asset(self,key,item,band):
        with self.lock:
            self.assets.setdefault(key,{'asset_key':key}).update(scene_id=item['id'],band=band)

    def stage(self,phase):
        with self.lock:self.phase=phase

    def start_work(self):self.work_started=time.perf_counter()

    def batch_days(self,done,total,cells):
        with self.lock:self.days_done=done;self.days_total=total;self.days_cells=cells

    def cell(self,cached=False,excluded=False):
        with self.lock:
            self.completed+=1;self.cached+=int(cached);self.new+=int(not cached);self.excluded+=int(excluded)

    def snapshot(self):
        with self.lock:
            elapsed=time.perf_counter()-self.started
            work_elapsed=time.perf_counter()-self.work_started if self.work_started else 0
            rate=self.new/work_elapsed if self.new and work_elapsed else None
            remaining=max(0,self.total-self.completed)
            return {'status':self.status,'phase':self.phase,'total_cells':self.total,'completed_cells':self.completed,
                'cached_cells':self.cached,'new_cells':self.new,'excluded_cells':self.excluded,
                'elapsed_seconds':elapsed,'new_cells_per_hour':rate*3600 if rate else None,
                'eta_seconds':remaining/rate if rate else (0 if remaining==0 else None),
                'batch_days_done':self.days_done,'batch_days_total':self.days_total,'batch_cells':self.days_cells,
                'timings':{key:dict(value) for key,value in self.timings.items()},'counters':dict(self.counters),
                'timing_note':'worker seconds are sums and can exceed wall time; decoded bytes are not HTTP wire bytes',
                'report_directory':str(self.directory)}

    def flush(self,console=True):
        with self.writer:
            state=self.snapshot();save_json(self.directory/'profile.json',state);save_json(self.directory/'progress.json',state)
            if console:
                rate=f'{state["new_cells_per_hour"]:.1f}/h' if state['new_cells_per_hour'] else 'warming up'
                eta=f'{state["eta_seconds"]/3600:.2f}h' if state['eta_seconds'] is not None else 'unknown'
                print(f'prepare {state["completed_cells"]}/{self.total} | cached={state["cached_cells"]} new={state["new_cells"]} | {rate} ETA={eta} | {state["phase"]}',flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            try:self.flush()
            except OSError:
                # Profiling must not discard already acquired data when output storage fails.
                self.count('profile_write_errors')

    def finish(self,status):
        self.stop.set();self.thread.join(timeout=2)
        with self.lock:self.status=status;self.phase=status;rows=[dict(v) for v in self.assets.values()]
        self.flush()
        if rows:
            columns=sorted({key for row in rows for key in row})
            with (self.directory/'assets.csv').open('w',newline='',encoding='utf-8') as output:
                writer=csv.DictWriter(output,fieldnames=columns);writer.writeheader();writer.writerows(rows)


class SpaceGuard:
    def __init__(self,cfg,profile):
        self.base=Path(cfg['data_root']);self.base.mkdir(parents=True,exist_ok=True)
        self.profile=profile;self.lock=threading.RLock()
        self.cap=cfg['storage']['max_gib']*1024**3;self.reserve=cfg['storage']['reserve_gib']*1024**3
        with profile.span('disk_inventory'):
            self.used=0
            for path in self.base.rglob('*'):
                # The heartbeat atomically replaces its temporary files during a long inventory.
                try:
                    if path.is_file():self.used+=path.stat().st_size
                except OSError:continue
        # Reserve room for profiler + SQLite/WAL metadata, which do not call these writers.
        self.used+=32*1024**2
        # Parallel area workers each receive a fixed share, so their sum stays within the global cap.
        share=cfg['storage'].get('max_new_gib')
        if share is not None:self.cap=min(self.cap,self.used+share*1024**3)
        self.check(0)

    def check(self,n):
        with self.lock,self.profile.span('disk_budget'):
            if self.used+n>self.cap or shutil.disk_usage(self.base).free<self.reserve+n:
                raise RuntimeError('Budget disco insufficiente: dati validi conservati; verificare storage.max_gib e spazio libero')

    def account(self,delta):
        with self.lock:self.used+=delta

    def remove_owned(self,path):
        path=Path(path)
        if path.exists():
            size=path.stat().st_size;path.unlink();self.account(-size)

    def replace(self,tmp,path):
        old=path.stat().st_size if path.exists() else 0
        os.replace(tmp,path);self.account(-old)

    def json(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        encoded=json.dumps(value,indent=2,allow_nan=False,default=str).encode('utf-8')
        self.check(len(encoded));tmp=path.with_suffix(path.suffix+'.partial')
        previous=tmp.stat().st_size if tmp.exists() else 0
        with self.profile.span('cache_write'):
            tmp.write_bytes(encoded);self.account(len(encoded)-previous);self.replace(tmp,path)

    def npz(self,path,**arrays):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        estimated=sum(v.nbytes for v in arrays.values())*1.01+8192
        self.check(int(estimated));tmp=path.with_suffix(path.suffix+'.partial')
        previous=tmp.stat().st_size if tmp.exists() else 0
        with self.profile.span('cache_write'):
            with tmp.open('wb') as output:np.savez_compressed(output,**arrays)
            self.account(tmp.stat().st_size-previous);self.replace(tmp,path)
