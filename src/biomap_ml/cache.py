# SCOPO: produrre una sola volta gli embedding per punto dell'encoder congelato.
# INPUT: indice auditato, serie NPZ esistenti e checkpoint Medium con SHA256 verificato.
# OUTPUT: SQLite in work-root/cache, con una griglia, maschera e pesi per ogni cella.
# RIPRESA: transazioni per 64 celle, identità del dataset e checksum per ogni record.
# La cache media 128D del probing non viene usata come falsa immagine 5x5.
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from biomap_v2.common import digest, save_json
from biomap_v2.encoder import load_encoder, pixel_embeddings
from .settings import work_budget
from .source import series_path
from .spatial import make_grid, validate_raw


# Lo stato include tutti i parametri/buffer, così il controllo frozen non usa la sola loss.
def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# Ogni record ha dimensioni fisse: feature 128x5x5, maschera 1x5x5, pesi 1x5x5.
def encode_record(values):
    blob = np.concatenate([v.ravel() for v in values]).astype('<f4').tobytes()
    return blob, hashlib.sha256(blob).hexdigest()


# La lettura controlla checksum, valori finiti, maschera e copertura prima di usare le feature.
def decode_record(blob, checksum):
    if len(blob) != (3200+25+25)*4 or hashlib.sha256(blob).hexdigest() != checksum:
        raise ValueError('Cache puntuale corrotta: lunghezza/checksum')
    arr = np.frombuffer(blob, dtype='<f4').copy()
    if not np.isfinite(arr).all():
        raise ValueError('Cache puntuale non finita')
    grid, mask, weights = arr[:3200].reshape(128, 5, 5), arr[3200:3225].reshape(1, 5, 5), arr[3225:].reshape(1, 5, 5)
    if not np.isin(mask, [0, 1]).all() or mask.sum() < 1 or (weights <= 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Cache: maschera/pesi non validi')
    return grid, mask, weights


# Il percorso dipende dal checkpoint, protocollo e provenienza, mai dal fold supervisionato.
def cache_path(cfg, audit):
    return Path(cfg['work'])/'cache'/f"points_{audit['fingerprint']['dataset_id']}.sqlite3"


# Solo questo comando calcola gli embedding. Non parte implicitamente dentro train.
def build_cache(cfg, legacy, frame, audit):
    path = cache_path(cfg, audit)
    work_budget(cfg, 1024**2)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    encoder = None
    before = None
    start = time.perf_counter()
    n_new = 0
    complete = False
    sidecar = path.with_suffix('.json')
    previous = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else {}
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS points (cell_id TEXT PRIMARY KEY, series_sha TEXT NOT NULL, blob BLOB NOT NULL, sha TEXT NOT NULL)')
        key = audit['fingerprint']['dataset_id']
        saved = conn.execute("SELECT value FROM meta WHERE key='dataset_id'").fetchone()
        if saved and saved[0] != key:
            raise ValueError('Cache puntuale incompatibile con l’audit')
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('dataset_id', ?)", (key,))
        conn.commit()
        for i, cell in enumerate(frame.itertuples()):
            record = conn.execute('SELECT series_sha,blob,sha FROM points WHERE cell_id=?', (cell.cell_id,)).fetchone()
            if record:
                if record[0] != cell.series_sha256:
                    raise ValueError('Cache serie incompatibile')
                decode_record(record[1], record[2])
            else:
                if encoder is None:
                    enc_cfg = {**legacy, 'model': {**legacy['model'], 'device': cfg['runtime']['device']}}
                    encoder, _, _ = load_encoder(enc_cfg, cfg['checkpoint'])
                    encoder.eval().requires_grad_(False)
                    before = state_hash(encoder)
                source = series_path(cfg, cell.cell_id)
                if digest(source) != cell.series_sha256:
                    raise ValueError(f'Serie cambiata dopo audit: {cell.cell_id}')
                with np.load(source, allow_pickle=False) as z:
                    raw = {k: z[k] for k in z.files}
                validate_raw(raw, cell)
                with torch.no_grad():  # FP32 anche su CUDA: precisione cache fissata prima del training.
                    emb = pixel_embeddings(encoder, raw, cfg['runtime']['pixel_batch'], False).float().cpu().numpy()
                blob, checksum = encode_record(make_grid(emb, raw['points'], cell))
                if n_new % 64 == 0:
                    work_budget(cfg, 64*len(blob)+1024**2)
                conn.execute('INSERT INTO points VALUES (?,?,?,?)', (cell.cell_id, cell.series_sha256, blob, checksum))
                n_new += 1
            if (i+1) % 64 == 0:
                conn.commit()
            if (i+1) % 100 == 0:
                print(f'Cache puntuale: {i+1}/{len(frame)}; nuovi={n_new}', flush=True)
        conn.commit()
        if conn.execute('SELECT count(*) FROM points').fetchone()[0] != len(frame):
            raise ValueError('Numerosità cache diversa dall’indice')
        if encoder is not None and before != state_hash(encoder):
            raise RuntimeError('Encoder modificato durante la generazione frozen')
        complete = True
    finally:
        conn.rollback()  # Conserva solo le transazioni durevoli; non conta l'ultima incompleta.
        history = previous.get('invocations', [])
        history.append({'status': 'complete' if complete else 'failed_or_interrupted',
                        'computed_cells_before_commit': n_new, 'seconds': time.perf_counter()-start})
        save_json(sidecar, {'dataset_id': audit['fingerprint']['dataset_id'], 'n': len(frame),
                  'precision': 'float32', 'checkpoint_sha256': audit['fingerprint']['checkpoint_sha256'],
                  'status': 'complete' if complete else 'incomplete', 'encoder_changed': False if complete else None,
                  'invocations': history, 'total_invocation_seconds': sum(item['seconds'] for item in history)})
        conn.close()  # Le transazioni già completate restano riutilizzabili dopo interruzione.
    return path


class PointDataset(Dataset):
    """Lettura SQLite senza scritture; nessun accesso alle serie durante il training E2."""

    # Controlla anche membership e SHA delle serie: un database completo diverso non è compatibile.
    def __init__(self, cfg, frame, audit):
        self.frame = frame.reset_index(drop=True)
        self.path = cache_path(cfg, audit)
        if not self.path.is_file() or not self.path.with_suffix('.json').is_file():
            raise FileNotFoundError('Cache puntuale incompleta: eseguire prepare --stage cache')
        sidecar = json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
        if sidecar.get('status') != 'complete' or sidecar.get('dataset_id') != audit['fingerprint']['dataset_id']:
            raise ValueError('Cache puntuale non completata per questo audit')
        self.conn = sqlite3.connect(self.path.as_uri()+'?mode=ro&immutable=1', uri=True)
        meta = self.conn.execute("SELECT value FROM meta WHERE key='dataset_id'").fetchone()
        if not meta or meta[0] != audit['fingerprint']['dataset_id']:
            raise ValueError('Identità cache differente dall’audit')
        available = dict(self.conn.execute('SELECT cell_id,series_sha FROM points'))
        if any(available.get(c.cell_id) != c.series_sha256 for c in self.frame.itertuples()):
            raise ValueError('Celle mancanti o serie incompatibili nella cache')

    # I dataset passati al trainer rappresentano solo il fold pertinente, mai Liguria.
    def __len__(self):
        return len(self.frame)

    # Restituisce target separato dalle feature; il target non partecipa alla cache.
    def __getitem__(self, index):
        cell = self.frame.iloc[int(index)]
        blob, sha = self.conn.execute('SELECT blob,sha FROM points WHERE cell_id=?', (cell.cell_id,)).fetchone()
        tensors = [torch.from_numpy(x) for x in decode_record(blob, sha)]
        return (*tensors, torch.tensor(float(cell.target), dtype=torch.float32))

    # Chiusura esplicita per permettere rename/backup del database su Windows.
    def close(self):
        self.conn.close()
