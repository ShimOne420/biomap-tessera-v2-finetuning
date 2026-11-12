# Scopo: configurazione, isolamento dei percorsi, transazioni, budget e monitoraggio V1 spatial.
# Fasi: YAML/CLI -> membership congelata -> work-root privata -> status atomico con heartbeat.
# Input: config.yaml e flag; output: JSON/log solo in v1_spatial_msa, mai nei dati V2.
# Parametri: --data-root, --work-root, --preset, --reference, --v1-root, --targets-csv, --set.
# Risorse: CPU e filesystem; esempio: python preflight.py --data-root ../data --preset pilot.
# Ripresa: FileLock OS, JSON atomici e identità SHA; nessun processo esterno è terminato.
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import sys
import numpy as np
import pandas as pd
import yaml
from filelock import FileLock

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'src'))
PROTOCOL='v1.0-spatial-native-centers-cell-loss-r1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def code_identity():
    return {p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(HERE.glob('*.py'))}


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False,default=str)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def common_parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',default=str(HERE/'config.yaml'))
    p.add_argument('--data-root',default='../data')
    p.add_argument('--work-root',help='Nuova root privata; nome obbligatorio v1_spatial_msa')
    p.add_argument('--v1-root');p.add_argument('--reference');p.add_argument('--targets-csv')
    p.add_argument('--preset',choices=['pilot','intermediate','full'],default='pilot')
    p.add_argument('--set',action='append',default=[])
    return p


def configuration(args):
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    for expr in args.set:
        key,sep,value=expr.partition('=')
        if not sep:raise ValueError('--set richiede sezione.chiave=valore')
        at=cfg;parts=key.split('.')
        for item in parts[:-1]:at=at[item]
        if parts[-1] not in at:raise ValueError('Parametro sconosciuto: '+key)
        at[parts[-1]]=yaml.safe_load(value)
    cfg=deepcopy(cfg)
    work=Path(args.work_root or Path(args.data_root)/'v1_spatial_msa').expanduser().resolve()
    if work.name!='v1_spatial_msa':raise ValueError('La work-root deve terminare con v1_spatial_msa per evitare output V2')
    for key in ['v1_root','reference','targets_csv']:
        value=getattr(args,key,None) or cfg['data'].get(key)
        cfg['data'][key]=str(Path(value).expanduser().resolve()) if value else None
        if value:
            source=Path(cfg['data'][key])
            if work==source or source in work.parents:raise ValueError('Output dentro una sorgente: '+str(source))
    if cfg['data']['year']!=2020 or cfg['data']['channels']!=128 or str(cfg['data']['source_version'])!='1.0' or cfg['data']['source_variant']!='vultr':
        raise ValueError('Questo protocollo richiede V1.0 vultr 2020 a 128 canali')
    if cfg['data']['max_patch_side']<1 or cfg['training']['batch_cells']<1 or cfg['training']['epochs']<1 or cfg['training']['patience']<1:raise ValueError('Dimensioni/epoche/patience positive richieste')
    if not 0<cfg['data']['min_valid_fraction']<=1:raise ValueError('Copertura non valida')
    cfg.update(work_root=str(work),preset=args.preset,protocol=PROTOCOL)
    return cfg


def membership(cfg):
    path=REPO/'configs/msa_v1_liguria_cells.csv'
    ref=json.loads((REPO/'configs/msa_v1_liguria_reference.json').read_text())
    raw=path.read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest()!=ref['cells_sha256']:raise ValueError('Membership originale modificata')
    frame=pd.read_csv(path,dtype={'cell_id':str})
    if frame.cell_id.duplicated().any() or frame.partition.value_counts().to_dict()!={'train':127909,'test':4991}:raise ValueError('Conteggi/split non conformi')
    if frame[['lon','lat']].duplicated().any():raise ValueError('Celle fisiche duplicate fra aree: verificare leakage prima del training')
    expected=ref['zero_based_fold_by_aoi']
    if not frame[frame.partition=='train'].apply(lambda row:row.cv_fold==expected[row.aoi_id],axis=1).all():raise ValueError('Fold geografici alterati')
    if set(frame[frame.partition=='test'].aoi_id)!={'liguria_levante'}:raise ValueError('Test esterno alterato')
    if ((frame.west<12.7)&(frame.east>12.4)&(frame.south<43.2)&(frame.north>43.0)).any():raise ValueError('Umbria nord sigillata')
    if not np.allclose(frame.east-frame.west,1/360,atol=1e-8,rtol=0) or not np.allclose(frame.north-frame.south,1/360,atol=1e-8,rtol=0):raise ValueError('Celle non native 10 arcsec')
    if cfg['preset']!='full':
        prefix='pilot' if cfg['preset']=='pilot' else 'intermediate'
        pieces=[]
        for aoi,group in frame.groupby('aoi_id',sort=True):
            count=cfg['sampling'][prefix+'_liguria'] if aoi=='liguria_levante' else cfg['sampling'][prefix+'_per_aoi']
            order=group.sort_values(['lat','lon','cell_id'])
            take=np.unique(np.linspace(0,len(order)-1,min(count,len(order))).astype(int))
            pieces.append(order.iloc[take])
        frame=pd.concat(pieces)
    return frame.sort_index().reset_index(drop=True)


def cache_path(cfg):return Path(cfg['work_root'])/'cache'/cfg['preset']/'spatial.sqlite3'


def data_identity(cfg,frame):
    data={k:v for k,v in cfg['data'].items() if k not in {'allow_partial_cells','source_mmap_limit'}}
    columns=['cell_id','aoi_id','partition','cv_fold','lon','lat','west','south','east','north','target']
    return signature({'protocol':PROTOCOL,'data':data,'preset':cfg['preset'],'cells':frame[[k for k in columns if k in frame]].to_dict('records')})


class Budget:
    def __init__(self,cfg):
        self.root=Path(cfg['work_root']);self.root.mkdir(parents=True,exist_ok=True)
        self.cap=cfg['storage']['max_gib']*1024**3;self.reserve=cfg['storage']['reserve_gib']*1024**3
    def check(self,extra=0):
        used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and p.suffix not in {'.tmp','.lock'})
        free=shutil.disk_usage(self.root).free
        if used+extra>self.cap or free-extra<self.reserve:raise RuntimeError('Budget/spazio insufficiente; sorgenti e risultati preservati')
        return {'used_bytes':used,'free_bytes':free}


@contextmanager
def exclusive(cfg):
    root=Path(cfg['work_root']);root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'experiment.lock'),timeout=0):yield


class Progress:
    def __init__(self,cfg,phase,total=0):
        self.path=Path(cfg['work_root'])/'status.json';self.lock=threading.RLock();self.stop=threading.Event()
        self.state={'protocol':PROTOCOL,'phase':phase,'preset':cfg['preset'],'status':'running','pid':os.getpid(),
            'started_utc':datetime.now(timezone.utc).isoformat(),'done':0,'total':total,'unit':'items'}
        self.started=time.perf_counter();self.rate_start=self.started;self.rate_done=0
        self.flush();self.thread=threading.Thread(target=self._beat,daemon=True);self.thread.start()
    def update(self,**values):
        with self.lock:self.state.update(values)
    def reset_rate(self,done=0):self.rate_start=time.perf_counter();self.rate_done=done
    def flush(self):
        with self.lock:
            value=dict(self.state);now=time.perf_counter();value['elapsed_seconds']=now-self.started
            progressed=value.get('done',0)-self.rate_done;elapsed=now-self.rate_start
            rate=progressed/elapsed if progressed>0 and elapsed>0 else None
            value.update(rate_per_second=rate,eta_seconds=max(value.get('total',0)-value.get('done',0),0)/rate if rate else None,updated_utc=datetime.now(timezone.utc).isoformat())
            save_json(self.path,value)
    def _beat(self):
        while not self.stop.wait(5):
            try:self.flush()
            except OSError:pass
    def close(self,status='complete',**values):
        self.stop.set();self.thread.join();self.update(status=status,**values);self.flush()


@contextmanager
def tracked(cfg,phase,total=0):
    p=Progress(cfg,phase,total)
    try:yield p
    except KeyboardInterrupt:p.close('interrupted');raise
    except Exception as error:p.close('failed',error=f'{type(error).__name__}: {error}');raise
    else:p.close()
