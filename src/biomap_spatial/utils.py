# Scopo: I/O atomico e identita del nuovo esperimento, con riuso delle primitive Windows esistenti.
# Fasi: hash/guardie percorsi -> file temporaneo fsync -> replace con retry; import B isolati.
# Input: percorsi locali e piccoli record; output: JSON/checkpoint privati, senza modificare le sorgenti.
# Parametri: cartella output separata, reserve_gib. Uso: import dal runner spatial_cnn.py.
# Risorse: CPU/disco; nessuna rete o modello encoder. Ripresa: marker finali scritti per ultimi.
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from biomap_v2.dense_fast.io_safe import save_json, unique_tmp, replace_with_retry
from biomap_v2.common import digest

ROOT = Path(__file__).resolve().parents[2]


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


@lru_cache(maxsize=2)
def legacy(name):
    if name not in {'model', 'dataset'}:
        raise ValueError('Import legacy non ammesso')
    spec = importlib.util.spec_from_file_location('biomap_spatial._legacy_' + name,
        ROOT / 'experiments/v1_spatial_cnn_msa' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def separate_output(out, sources):
    out = Path(out).resolve()
    for source in sources:
        if not source:
            continue
        source = Path(source).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError('Output e sorgenti devono essere separati e non annidati')
    return out


def disk_budget(path, reserve_gib=20, additional_bytes=0):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < reserve_gib * 1024**3 + additional_bytes:
        raise OSError('Spazio insufficiente: sorgenti e checkpoint preservati')


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    try:
        with tmp.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def code_identity():
    names = list((ROOT / 'src/biomap_spatial').glob('*.py'))
    names += [ROOT / 'experiments/v1_spatial_cnn_msa' / (n + '.py') for n in ['model', 'dataset']]
    names += [ROOT / 'src/biomap_v2/dense/evaluation.py', ROOT / 'scripts/spatial_cnn.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in names}


def git_commit():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'
