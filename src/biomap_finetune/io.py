# Scopo: transazioni nuove, immutabili e riprendibili per raw NPY, checkpoint e avanzamento Windows.
# Fasi: verifica root -> temporanei unici -> fsync/rilettura/SHA -> replace con retry -> marker finale.
# Input: array/checkpoint propri del nuovo esperimento; output: solo nuove root, mai sorgenti legacy.
# Parametri: identita, reserve_gib, budget; uso: commit_arrays(path,arrays,identity).
# Risorse: CPU/disco; nessuna rete/GPU. Ripresa: marker assente non e completo, orphan parziali preservati.
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from biomap_spatial.utils import signature,separate_output,disk_budget,git_commit
from biomap_v2.common import digest,ROOT
from biomap_v2.dense_fast.io_safe import save_json,unique_tmp,replace_with_retry

PROTOCOL='finetune-v2-r1'


def checked_roots(cfg):
    paths=cfg['paths']
    protected=[paths.get(k) for k in ['reference','catalog_root','frozen_work','frozen_spatial_run','frozen_pixels']]
    protected+=paths.get('read_only_roots',[])
    roots=[Path(paths[k]).resolve() for k in ['raw_root','output_root'] if paths.get(k)]
    for root in roots:separate_output(root,protected)
    if len(roots)==2 and (roots[0]==roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents):
        raise ValueError('Raw e run devono essere root nuove separate, non annidate')
    return roots


def claim_root(root,identity,kind):
    root=Path(root)
    marker=root/'ownership.json'
    if marker.exists():
        record=json.loads(marker.read_text(encoding='utf-8'))
        if record!={'identity':identity,'kind':kind,'protocol':PROTOCOL}:raise ValueError('Root non vuota incompatibile: nessun overwrite')
    else:
        if root.exists() and any(root.iterdir()):raise ValueError('Root non vuota senza identita: rifiutata')
        root.mkdir(parents=True,exist_ok=True)
        save_json(marker,{'identity':identity,'kind':kind,'protocol':PROTOCOL})


def commit_arrays(path,arrays,identity,metadata=None):
    path=Path(path)
    if path.exists():return load_arrays(path,identity)[1]
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex[:10]);tmp.mkdir()
    files={}
    for name,value in arrays.items():
        value=np.asarray(value)
        if value.dtype.hasobject:raise ValueError('Array object non ammesso')
        p=tmp/(name+'.npy')
        with p.open('wb') as stream:
            np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        check=np.load(p,allow_pickle=False)
        if check.shape!=value.shape or check.dtype!=value.dtype or not np.array_equal(check,value,equal_nan=True):raise ValueError('Rilettura NPY discordante')
        files[name]={'sha256':digest(p),'shape':list(value.shape),'dtype':str(value.dtype)}
    save_json(tmp/'complete.json',{'identity':identity,'files':files,'metadata':metadata or {}})
    replace_with_retry(tmp,path)
    return json.loads((path/'complete.json').read_text(encoding='utf-8'))


def load_arrays(path,identity=None,verify=True,mmap=False):
    path=Path(path);marker=path/'complete.json'
    if not marker.is_file():raise ValueError('Blocco incompleto: marker assente')
    info=json.loads(marker.read_text(encoding='utf-8'))
    if identity is not None and info['identity']!=identity:raise ValueError('Identita blocco differente')
    arrays={}
    for name,record in info['files'].items():
        p=path/(name+'.npy')
        if verify and digest(p)!=record['sha256']:raise ValueError('Checksum input/prefisso discordante')
        arr=np.load(p,allow_pickle=False,mmap_mode='r' if mmap else None)
        if list(arr.shape)!=record['shape'] or str(arr.dtype)!=record['dtype']:raise ValueError('Shape/dtype cache non validi')
        arrays[name]=arr
    return arrays,info


def save_checkpoint(path,state):
    import torch
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Checkpoint immutabile gia presente: ripresa dal pointer o nuovo run-id, nessun overwrite')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=unique_tmp(path)
    with tmp.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    reread=torch.load(tmp,map_location='cpu',weights_only=False)
    if reread['identity']!=state['identity'] or reread['epoch']!=state['epoch']:raise ValueError('Rilettura checkpoint non valida')
    sha=digest(tmp);replace_with_retry(tmp,path)
    save_json(path.with_suffix('.json'),{'sha256':sha,'identity':state['identity'],'epoch':state['epoch']})


def load_checkpoint(path,identity):
    import torch
    path=Path(path);mark=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if mark['identity']!=identity or digest(path)!=mark['sha256']:raise ValueError('Checkpoint corrotto/incompatibile')
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value['identity']!=identity:raise ValueError('Identita interna checkpoint diversa')
    return value


class Progress:
    def __init__(self,path):self.path=Path(path);self.began=time.perf_counter();self.last=0.;self.state={}
    def update(self,force=False,**kw):
        self.state.update(kw);now=time.perf_counter();elapsed=now-self.began
        self.state.update(elapsed_seconds=elapsed,updated_utc=datetime.now(timezone.utc).isoformat())
        eta=self.state.get('eta_seconds')
        self.state['estimated_finish_utc']=(datetime.now(timezone.utc)+timedelta(seconds=eta)).isoformat() if eta is not None else None
        if force or now-self.last>=2:
            try:save_json(self.path,self.state)
            except OSError as error:print('Progress non scritto; checkpoint indipendente:',error,flush=True)
            self.last=now


def code_identity():
    paths=list((ROOT/'src/biomap_finetune').glob('*.py'))+[ROOT/'scripts/finetune_v2.py']
    paths+=[ROOT/'src/biomap_v2/student.py',ROOT/'src/biomap_spatial/model.py']
    from biomap_v2.campaign.streaming import code_identity as legacy_identity
    return {**legacy_identity(),**{str(p.relative_to(ROOT)):digest(p) for p in paths}}
