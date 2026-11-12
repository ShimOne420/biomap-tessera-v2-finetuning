# SCOPO
#   Scritture atomiche robuste su Windows per la pipeline r2-fast. Corregge la causa dei due arresti
#   della campagna r1 (11/11/2026, 14:06 e 20:09): os.replace su stream_progress.json fallito con
#   PermissionError [WinError 5] perché un altro processo stava leggendo il file.
# COME VIENE RICHIAMATO
#   from biomap_v2.dense_fast.io_safe import save_json, atomic_npz, cached, append_jsonl, replace_with_retry
# LOGICA IN ORDINE
#   1. Ogni scrittura usa un file temporaneo UNICO (pid + uuid) nella stessa cartella: due processi o
#      due thread non si sovrascrivono mai il temporaneo a vicenda.
#   2. Il temporaneo viene scritto, svuotato (flush) e sincronizzato su disco (fsync).
#   3. os.replace viene ritentato con attesa crescente su PermissionError/OSError di condivisione
#      (WinError 5 e 32), fino a `attempts` volte; poi l'errore viene rilanciato.
#   4. In caso di errore il temporaneo viene rimosso; il file di destinazione resta quello precedente,
#      intatto: un file completato non viene mai corrotto da un salvataggio fallito.
#   5. atomic_npz/cached: stesso contratto della r1 (payload, poi marker JSON con identità e SHA256;
#      identità diversa o checksum errato = errore, mai cache da sovrascrivere).
# INPUT E OUTPUT
#   Input: oggetti JSON o array NumPy. Output: file nella work-root indicata dal chiamante.
# PARAMETRI
#   attempts (default 30) e max_delay (default 2 s) per i ritentativi di os.replace.
# RISORSE
#   Disco; nessuna rete.
# RIPRESA
#   Nessuno stato proprio: i file sono completi oppure assenti.
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from biomap_v2.common import digest

_RETRYABLE_WINERRORS = {5, 32, 33}


def _retryable(error):
    if isinstance(error, PermissionError):
        return True
    return getattr(error, 'winerror', None) in _RETRYABLE_WINERRORS


def unique_tmp(path):
    path = Path(path)
    return path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex[:12]}.tmp')


def replace_with_retry(tmp, path, attempts=30, max_delay=2.0):
    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return attempt
        except OSError as error:
            if not _retryable(error) or attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def _write_tmp(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    try:
        with open(tmp, 'wb') as out:
            writer(out)
            out.flush()
            os.fsync(out.fileno())
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _commit(tmp, path):
    try:
        replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_json(path, value):
    data = json.dumps(value, indent=2, default=str).encode('utf-8')
    tmp = _write_tmp(path, lambda out: out.write(data))
    _commit(tmp, path)


def atomic_npz(path, identity, **arrays):
    path = Path(path)
    tmp = _write_tmp(path, lambda out: np.savez_compressed(out, **arrays))
    try:
        with np.load(tmp, allow_pickle=False) as z:
            if set(z.files) != set(arrays):
                raise ValueError('Transazione NPZ incompleta')
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _commit(tmp, path)
    save_json(path.with_suffix('.json'), {'identity': identity, 'sha256': digest(path)})


def cached(path, identity):
    path = Path(path)
    meta = path.with_suffix('.json')
    if not meta.exists():
        return False
    info = json.loads(meta.read_text())
    if info['identity'] != identity or not path.exists() or digest(path) != info['sha256']:
        raise ValueError(f'Cache incompatibile/corrotta: {path}')
    return True


def append_jsonl(path, record, attempts=30):
    line = json.dumps(record, default=str) + '\n'
    delay = 0.05
    for attempt in range(attempts):
        try:
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(line)
            return
        except OSError as error:
            if not _retryable(error) or attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
