# SCOPO: opzioni comuni E2, separazione fisica sorgente/lavoro e scritture atomiche.
# La sorgente è solo letta: nessuna chiamata legacy prepare/weights/doctor può scriverci.
# Tutte le CLI richiedono una work-root distinta, anche se la sorgente è un symlink.
# I parametri scientifici vengono salvati nel run e controllati durante il resume.
import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import yaml
from biomap_v2.common import ROOT, config as legacy_config, digest, save_json
from . import PROTOCOL


# Risolve symlink e impedisce sia una work-root interna alla sorgente sia il contrario.
def separate_roots(source, work):
    source, work = Path(source).expanduser().resolve(), Path(work).expanduser().resolve()
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('source-data-root e work-root devono essere cartelle separate, non annidate')
    return source, work


# Una sola interfaccia coerente per preparazione, training, valutazione e profiling.
def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default=str(ROOT/'configs/e2_msa.yaml'))
    p.add_argument('--source-data-root', required=True, help='Cartella esistente con cells.csv e series/')
    p.add_argument('--work-root', required=True, help='Nuova cartella scrivibile, separata dalla sorgente')
    p.add_argument('--checkpoint', help='Medium già scaricato; nessun download automatico')
    p.add_argument('--device', help='cuda:0 per il run, cpu per test tecnici')
    return p


# Carica il protocollo del probing senza eseguirlo; gli override sono limitati al device.
def configuration(args):
    source, work = separate_roots(args.source_data_root, args.work_root)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    legacy = legacy_config('msa', SimpleNamespace(config=str(ROOT/cfg['source_config']),
                                                 data_root=str(source.parent), set=[]))
    if source.name != legacy['dataset_namespace']:
        raise ValueError('source-data-root deve terminare con '+legacy['dataset_namespace'])
    if legacy['sampling']['pixels_per_cell'] != 25 or legacy['year'] != 2020:
        raise ValueError('E2 richiede campionamento 5x5 del 2020')
    checkpoint = Path(args.checkpoint or source.parent/'weights/student_medium.pt').expanduser().resolve()
    cfg.update(source=str(source), work=str(work), checkpoint=str(checkpoint), protocol=PROTOCOL)
    if args.device:
        cfg['runtime']['device'] = args.device
    for key in ['threads', 'pixel_batch']:
        if not isinstance(cfg['runtime'][key], int) or cfg['runtime'][key] < 1:
            raise ValueError(f'runtime.{key} deve essere un intero positivo')
    t = cfg['training']
    for key in ['warmup_epochs', 'epochs', 'patience', 'microbatch', 'effective_batch']:
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f'training.{key} deve essere un intero positivo')
    if t['seed'] != 42 or t['microbatch'] > t['effective_batch']:
        raise ValueError('Prima campagna: seed 42 e microbatch <= batch effettivo')
    for key in ['learning_rate', 'clip_grad']:
        if t[key] <= 0:
            raise ValueError(f'training.{key} deve essere positivo')
    if t['weight_decay'] < 0 or t['min_delta'] < 0:
        raise ValueError('Weight decay e min_delta non possono essere negativi')
    return cfg, legacy


# Le identità JSON usano ordine stabile e rifiutano NaN, senza dipendere dal percorso del PC.
def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


# Controllo globale della sola cartella di lavoro; non scandisce o modifica la data-root attiva.
def work_budget(cfg, additional=0):
    import shutil
    work = Path(cfg['work'])
    separate_roots(cfg['source'], work)
    work.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() and work not in p.resolve().parents for p in work.rglob('*')):
        raise ValueError('La work-root contiene un symlink esterno: isolamento non garantito')
    used = sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
    if used+additional > cfg['storage']['max_work_gib']*1024**3:
        raise RuntimeError('Budget work-root insufficiente; nessun dato sorgente cancellato')
    if shutil.disk_usage(work).free-additional < cfg['storage']['reserve_gib']*1024**3:
        raise RuntimeError('Riserva disco insufficiente nella work-root')


# Scrittura Torch atomica: un'interruzione lascia valido il checkpoint precedente.
def save_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


# Serializzazione DataFrame atomica per evitare file indicizzati a metà.
def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


# Un solo writer per work-root: nessun lock viene creato nei dati del probing.
@contextmanager
def command_context(args):
    from filelock import FileLock, Timeout
    cfg, legacy = configuration(args)
    work_budget(cfg)
    lock = FileLock(str(Path(cfg['work'])/'.e2_operation.lock'))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError('Un altro comando E2 usa questa work-root') from None
    try:
        yield cfg, legacy
    finally:
        lock.release()


# Tutte le CLI vincolano il run-id a una sottocartella semplice della work-root.
def run_directory(cfg, run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run-id: usare lettere, numeri, underscore o trattino')
    return Path(cfg['work'])/'runs'/run_id
