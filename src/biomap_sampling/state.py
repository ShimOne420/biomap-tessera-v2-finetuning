# Scopo: configurazione preregistrata e I/O del nuovo benchmark, con root di proprieta separate.
# Fasi: YAML/CLI -> gate risorse -> identita codice/dati -> transazioni e progress atomici.
# Input: config/percorsi espliciti; output: solo cache e run nuovi, mai indice o sorgenti.
# Parametri: reserve_gib, shard, worker spawn, budget; uso load_config(path,overrides).
# Risorse: metadati CPU/disco. Ripresa: marker verificati, nessun overwrite di payload validi.
from copy import deepcopy
import json
from pathlib import Path
import time
import math
import yaml
from biomap_spatial.utils import signature,separate_output,disk_budget,git_commit
from biomap_v2.common import ROOT,digest
from biomap_v2.dense_fast.io_safe import save_json,unique_tmp,replace_with_retry
from biomap_finetune.io import commit_arrays,load_arrays,save_checkpoint,load_checkpoint,Progress

PROTOCOL='sampling-v2-nested-r1'


def load_config(path,overrides=()):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for entry in overrides:
        key,sep,value=entry.partition('=')
        if not sep:raise ValueError('Override sezione.chiave=valore richiesto')
        section,name=key.split('.',1)
        if section not in cfg or name not in cfg[section]:raise ValueError('Parametro sconosciuto: '+key)
        cfg[section][name]=yaml.safe_load(value)
    if cfg['sampling']['modes']!=['S64','M128','S144','M64','FULL-SUBSET']:raise ValueError('Suite preregistrata non modificabile')
    if cfg['decision']['rule']!='measured_full_seed_noise' or not cfg['decision']['recommendation_only']:raise ValueError('Regola preregistrata richiesta')
    if cfg['training']['control_seeds']!=[42,43] or cfg['training']['confirmation_seeds']!=[42,43,44] or cfg['training']['initial_seed']!=42:raise ValueError('Seed preregistrati richiesti')
    for section,names in [('sampling',['subset_cells','min_valid_pixels']),('cache',['cells_per_shard','mmap_limit']),
        ('training',['epochs','patience','batch_cells','accumulate_cells']),('execution',['smoke_cells','smoke_seconds','screen_budget_hours'])]:
        for name in names:
            if cfg[section][name]<=0:raise ValueError('Parametro positivo richiesto: '+name)
    if cfg['training']['accumulate_cells']<cfg['training']['batch_cells'] or cfg['training']['num_workers'] not in range(5):raise ValueError('Batch/worker non validi')
    if cfg['execution']['smoke_seconds']>900 or cfg['execution']['reserve_gib']<0:raise ValueError('Budget non valido')
    for name in ['learning_rate','clip_grad','worker_timeout_seconds']:
        if cfg['training'][name]<=0 or not math.isfinite(cfg['training'][name]):raise ValueError('Parametro training non valido: '+name)
    if cfg['training']['weight_decay']<0 or not math.isfinite(cfg['training']['weight_decay']):raise ValueError('weight_decay non valido')
    if cfg['decision']['bootstrap_samples']<1 or cfg['cache']['validation_max_gib']<=0:raise ValueError('Budget bootstrap/cache non valido')
    return cfg


def roots(cfg,require=True):
    p=cfg['paths'];required=['pixel_root','reference','index_root','cache_root','output_root']
    if require and any(not p[x] for x in required):raise ValueError('Configurare '+', '.join(required))
    protected=[p.get(x) for x in ['pixel_root','reference','index_root','historical_run']]+p.get('read_only_roots',[])
    outputs=[Path(p[k]).resolve() for k in ['cache_root','output_root'] if p.get(k)]
    for out in outputs:separate_output(out,protected)
    if len(outputs)==2:separate_output(outputs[0],[outputs[1]])
    return outputs


def claim(root,identity,kind):
    root=Path(root);mark=root/'ownership.json';value={'protocol':PROTOCOL,'identity':identity,'kind':kind}
    if mark.exists():
        if json.loads(mark.read_text(encoding='utf-8'))!=value:raise ValueError('Root incompatibile: nessun overwrite')
    else:
        if root.exists() and any(root.iterdir()):raise ValueError('Root non vuota senza proprieta')
        root.mkdir(parents=True,exist_ok=True);save_json(mark,value)


def code_identity():
    paths=list((ROOT/'src/biomap_sampling').glob('*.py'))+[ROOT/'scripts/sampling_benchmark.py']
    paths += [ROOT/'src/biomap_spatial'/n for n in ['model.py','sources.py','support.py','batching.py','normalization.py']]
    paths += [ROOT/'src/biomap_finetune/data.py',ROOT/'src/biomap_v2/dense/evaluation.py']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Manifest immutabile differente')
    else:save_json(path,value)


def atomic_frame(path,frame):
    frame=frame.reset_index(drop=True)  # parquet index=False: row labels are not scientific cell identifiers.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        import pandas as pd
        if not pd.read_parquet(path).equals(frame):raise ValueError('Risultato esistente differente')
        return
    tmp=unique_tmp(path);frame.to_parquet(tmp,index=False)
    import pandas as pd
    if not pd.read_parquet(tmp).equals(frame):raise ValueError('Rilettura parquet differente')
    replace_with_retry(tmp,path)


class BudgetExpired(RuntimeError):pass


def check_deadline(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise BudgetExpired('Budget raggiunto; checkpoint validi conservati, usare --resume')
