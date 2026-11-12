# SCOPO
#   Lettura di file di una campagna in corso con la condivisione più ampia possibile (Read|Write|Delete).
#   ATTENZIONE (verificato il 12/11/2026): su Windows os.replace fallisce con PermissionError [WinError 5]
#   se il file di DESTINAZIONE è aperto da un altro processo, ANCHE con condivisione Delete. Quindi:
#   - sicuro per file mai sostituiti: stream_log.jsonl (solo aggiunte), tile.json/tile.npz completati,
#     cataloghi api/, geometry e reference, last_invocation.json durante il run;
#   - NON usare mai su file che la campagna sostituisce di continuo (stream_progress.json).
# COME VIENE RICHIAMATO
#   from biomap_v2.dense_fast.shared_read import read_bytes, read_text, sha256_file, load_npz
# LOGICA IN ORDINE
#   1. CreateFileW con FILE_SHARE_READ|WRITE|DELETE, sola lettura; handle chiuso subito dopo la lettura.
#   2. Su sistemi non Windows usa la lettura normale (nessun blocco equivalente).
# INPUT E OUTPUT
#   Input: percorso. Output: bytes / testo / SHA256 / dizionario di array da un .npz.
# PARAMETRI
#   Nessuno.
# RISORSE
#   Un handle per volta, letto a blocchi da 1 MiB per l'hash.
# RIPRESA
#   Funzioni pure.
import hashlib
import io
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def open_shared(path):
    if os.name != 'nt':
        with open(path, 'rb') as stream:
            yield stream
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        error = ctypes.get_last_error()
        raise FileNotFoundError(error, f'Impossibile aprire {path}') if error in (2, 3) else OSError(error, f'Impossibile aprire {path}')
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    with open(fd, 'rb') as stream:
        yield stream


def read_bytes(path):
    with open_shared(path) as stream:
        return stream.read()


def read_text(path):
    return read_bytes(path).decode('utf-8', errors='replace')


def sha256_file(path):
    h = hashlib.sha256()
    with open_shared(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    with np.load(io.BytesIO(read_bytes(path)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def exists(path):
    return Path(path).exists()
