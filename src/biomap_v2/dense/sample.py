# SCOPO: estrarre un piccolo campione Sentinel per verificare l'encoder sul PC ricevente.
# Legge le tile senza decomprimere o scrivere file nella sorgente; il campione è portabile.
# La verifica confronta il Medium ufficiale con gli embedding FP32 già presenti, senza training.
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from biomap_v2.common import WEIGHTS_SHA256, digest
from biomap_v2.encoder import load_encoder
from biomap_v2.reference import encode_pixels
from .v2_support import _doys, discover_tiles


# Il caso gzip carica una sola banda/modo per volta e scrive soltanto il campione finale.
def sample_array(directory, key, indices):
    path = directory/f'{key}.npy'
    if path.is_file():
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        return np.array(arr[:, indices], copy=True)
    with gzip.open(directory/f'{key}.npy.gz', 'rb') as stream:
        arr = np.load(stream, allow_pickle=False)
    return np.array(arr[:, indices], copy=True)


# Selezione geometrica uniforme, indipendente da MSA e predizioni del regressore.
def create_sample(v2_root, destination, pixels=32):
    tiles = discover_tiles(v2_root)
    if not tiles:
        raise ValueError('Nessuna tile completa da campionare')
    tile = tiles[0]
    folder = tile['directory']
    n = int(tile['spec']['pixels'])
    idx = np.unique(np.linspace(0, n-1, min(pixels, n), dtype=np.int64))
    doys = _doys(json.loads((folder/'inputs_manifest.json').read_text(encoding='utf-8')))
    arrays = {k: sample_array(folder, k, idx) for k in ['s2', 'mask', 'asc', 'desc']}
    raw = {k: v.transpose(1, 0, 2).copy() if v.ndim == 3 else v.T.copy() for k, v in arrays.items()}
    raw.update({k+'_doys': v for k, v in doys.items()})
    raw['expected'] = np.array(np.load(folder/'embeddings_fp32.npy', mmap_mode='r')[idx], copy=True)
    raw['pixel_indices'] = idx
    np.savez_compressed(destination, **raw)
    return {'tile': tile['name'], 'pixels': len(idx), 'selection': 'evenly_spaced_pixel_order'}


# Non carica checkpoint adattati; un confronto non finito o oltre tolleranza ferma il run.
def verify_sample(path, checkpoint, device='cpu', pixel_batch=16, tolerance=1e-4):
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError('Indicare il checkpoint Medium già disponibile con --checkpoint')
    if digest(checkpoint) != WEIGHTS_SHA256:
        raise ValueError('SHA256 checkpoint Medium non conforme')
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    cfg = {'indicator': 'msa', 'model': {'device': device, 'checkpoint': str(checkpoint)}}
    model, _, _ = load_encoder(cfg, override=str(checkpoint))
    model.eval().requires_grad_(False)
    with torch.no_grad():
        predicted = encode_pixels(model, raw['s2'], raw['s2_doys'], raw['asc'], raw['asc_doys'],
                                  raw['desc'], raw['desc_doys'], raw['mask'],
                                  batch_pixels=pixel_batch, device=torch.device(device))
    maximum = float(np.max(np.abs(predicted-raw['expected'])))
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f'Equivalenza encoder fallita: max_abs={maximum}, tolleranza={tolerance}')
    return {'status': 'passed', 'pixels': len(predicted), 'max_abs': maximum,
            'tolerance': tolerance, 'checkpoint_sha256': WEIGHTS_SHA256, 'device': str(device)}
