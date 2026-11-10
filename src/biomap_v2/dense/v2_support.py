# Scopo: aggregare a cella gli embedding V2 Medium sugli stessi pixel finiti V1.
# Fasi: scoperta tile -> lettura geometry/embeddings -> media (e std) densa per cella.
# Input: <v2_root>/grid_*/geometry.npz, embeddings_fp32.npy, encoder_progress.json.
# Output: una riga per cella con mean_* (e std_*) sui pixel V1 con scala finita.
# Parametri: include_std=True produce anche std_*; n_total V1 serve per valid_fraction.
# Risorse: CPU; memmap degli embedding, nessuna copia float32 dell'intero tile.
# Ripresa: encoder salva memmap e progresso atomico per tile; i tile completi sono riusati.
from pathlib import Path
import gzip
import json
import shutil

import numpy as np
import pandas as pd
import torch

from .v1_support import DIMS

PROTOCOL = 'medium_mpc_2020_all_dates_exact_v1_pixels_fp32_r1'
CHUNK_PIXELS = 8192


def discover_tiles(v2_root, require_complete=True):
    root = Path(v2_root)
    result = []
    for directory in sorted(root.glob('grid_*')):
        geometry = directory / 'geometry.npz'
        embeddings = directory / 'embeddings_fp32.npy'
        if not geometry.is_file():
            continue
        spec = json.loads((directory / 'geometry.json').read_text(encoding='utf-8')) if (directory / 'geometry.json').is_file() else {}
        complete = False
        progress_path = directory / 'encoder_progress.json'
        if embeddings.is_file() and progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
            complete = int(progress.get('pixels_complete', 0)) == int(progress.get('pixels_total', -1)) == int(spec.get('pixels', -2))
        if require_complete and not complete:
            continue
        result.append({'name': directory.name, 'directory': directory, 'geometry': geometry,
                       'embeddings': embeddings, 'spec': spec, 'complete': complete})
    return result


def aggregate_v2(v2_root, n_total=None, min_valid_fraction=0.8, include_std=False,
                 require_complete=True, dimensions=DIMS):
    """Media (e std) per cella sui pixel con embedding finito."""
    tiles = discover_tiles(v2_root, require_complete=require_complete)
    if not tiles:
        raise FileNotFoundError(f'Nessun tile V2 completo sotto {v2_root}')
    arrays = []
    max_cell = -1
    for tile in tiles:
        with np.load(tile['geometry']) as geometry:
            cells = geometry['cell_ids'].astype(np.int64)
        max_cell = max(max_cell, int(cells.max()) if len(cells) else -1)
        arrays.append((tile, cells))
    n_cells = max_cell + 1
    sums = np.zeros((n_cells, dimensions), dtype=np.float64)
    squares = np.zeros((n_cells, dimensions), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for tile, cells in arrays:
        embeddings = np.load(tile['embeddings'], mmap_mode='r')
        if embeddings.shape != (len(cells), dimensions):
            raise ValueError(f"{tile['name']}: embedding {embeddings.shape} diversi da geometry {len(cells)}")
        # I pixel sono gia' filtrati sui soli supporti V1 con scala finita.
        finite = np.isfinite(embeddings).all(axis=1)
        if not finite.all():
            raise ValueError(f"{tile['name']}: {int((~finite).sum())} embedding non finiti")
        values = np.asarray(embeddings, dtype=np.float64)
        np.add.at(sums, cells, values)
        if include_std:
            np.add.at(squares, cells, np.square(values))
        counts += np.bincount(cells, minlength=n_cells)
    mean = np.divide(sums, np.maximum(counts[:, None], 1),
                     out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    records = {'cell_id': np.arange(n_cells), 'n_valid_v2': counts}
    for band in range(dimensions):
        records[f'mean_{band:03d}'] = mean[:, band]
    if include_std:
        variance = np.divide(squares, np.maximum(counts[:, None], 1),
                             out=np.zeros_like(squares), where=counts[:, None] > 0) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        for band in range(dimensions):
            records[f'std_{band:03d}'] = std[:, band]
    frame = pd.DataFrame(records).set_index('cell_id')
    if n_total is not None:
        denominator = pd.Series(n_total)
        frame['n_total'] = frame.index.map(denominator)
        frame['valid_fraction'] = frame.n_valid_v2 / frame.n_total.replace(0, np.nan)
        frame = frame[frame.n_total.notna() & (frame.valid_fraction >= min_valid_fraction)]
    else:
        frame = frame[frame.n_valid_v2 > 0]
    return frame


def compare_v2(recomputed, reference):
    common = recomputed.index.intersection(reference.index)
    result = {'n_recomputed': int(len(recomputed)), 'n_reference': int(len(reference)),
              'n_common': int(len(common)),
              'missing_in_recomputed': int(len(reference.index.difference(recomputed.index))),
              'extra_in_recomputed': int(len(recomputed.index.difference(reference.index)))}
    if not len(common):
        return result
    left = recomputed.loc[common]
    right = reference.loc[common]
    if 'n_valid' in right:
        result['n_valid_mismatch'] = int((left.n_valid_v2 != right.n_valid).sum())
        result['max_abs_n_valid'] = int(np.max(np.abs(left.n_valid_v2 - right.n_valid)))
    means = [f'mean_{i:03d}' for i in range(DIMS)]
    result['max_abs_mean'] = float(np.max(np.abs(left[means].to_numpy() - right[means].to_numpy())))
    stds = [f'std_{i:03d}' for i in range(DIMS)]
    if all(column in right for column in stds):
        result['max_abs_std'] = float(np.max(np.abs(left[stds].to_numpy() - right[stds].to_numpy())))
    return result


def _load_array(directory, key):
    """Preferisce il memmap npy; se assente decomprime una volta il .npy.gz."""
    raw = directory / f'{key}.npy'
    packed = directory / f'{key}.npy.gz'
    if raw.is_file():
        return np.load(raw, mmap_mode='r')
    if not packed.is_file():
        raise FileNotFoundError(f'{raw} / {packed} assenti')
    temporary = raw.with_suffix('.npy.partial')
    with gzip.open(packed, 'rb') as source, open(temporary, 'wb') as target:
        shutil.copyfileobj(source, target, length=8 * 1024**2)
    temporary.replace(raw)
    return np.load(raw, mmap_mode='r')


def _doys(manifest):
    from datetime import datetime
    return {source: np.array([datetime.fromisoformat(item['date']).timetuple().tm_yday
                              for item in manifest['scene_groups'][source]], dtype=np.int16)
            for source in ('s2', 'asc', 'desc')}


def encode_tiles(v2_root, checkpoint, device='cpu', pixel_batch=128, protocol=PROTOCOL):
    """Completa in modo riprendibile gli embedding mancanti usando reference.encode_pixels."""
    from .. import encoder
    from ..reference import encode_pixels
    from ..common import WEIGHTS_SHA256, digest, save_json

    if digest(checkpoint) != WEIGHTS_SHA256:
        raise ValueError('Checkpoint Medium non verificato (SHA256)')
    cfg = {'indicator': 'msa', 'model': {'device': device, 'checkpoint': str(checkpoint)}}
    model, _, _ = encoder.load_encoder(cfg, override=str(checkpoint))
    tiles = discover_tiles(v2_root, require_complete=False)
    done = []
    for tile in tiles:
        if tile['complete']:
            continue
        directory = tile['directory']
        geometry = np.load(tile['geometry'])
        n = len(geometry['flat_indices'])
        if 'cell_ids' not in geometry:
            raise ValueError(f"{tile['name']}: geometry priva di cell_ids")
        manifest = json.loads((directory / 'inputs_manifest.json').read_text(encoding='utf-8'))
        doys = _doys(manifest)
        signature = {'protocol': protocol, 'checkpoint_sha256': WEIGHTS_SHA256,
                     'inputs_sha256': digest(directory / 'inputs_manifest.json')}
        output_path = directory / 'embeddings_fp32.npy'
        state_path = directory / 'encoder_progress.json'
        start = 0
        if state_path.is_file():
            progress = json.loads(state_path.read_text(encoding='utf-8'))
            if any(progress.get(key) != value for key, value in signature.items()):
                raise ValueError(f"{tile['name']}: embedding parziali con input/checkpoint diversi")
            start = int(progress.get('pixels_complete', 0))
        output = (np.load(output_path, mmap_mode='r+') if output_path.is_file()
                  else np.lib.format.open_memmap(output_path, mode='w+', dtype='float32', shape=(n, DIMS)))
        if output.shape != (n, DIMS):
            raise ValueError(f"{tile['name']}: shape embedding parziale errata")
        inputs = {key: _load_array(directory, key) for key in ('s2', 'mask', 'asc', 'desc')}
        for low in range(start, n, CHUNK_PIXELS):
            high = min(n, low + CHUNK_PIXELS)
            s2 = np.ascontiguousarray(inputs['s2'][:, low:high].transpose(1, 0, 2))
            mask = np.ascontiguousarray(inputs['mask'][:, low:high].T)
            asc = np.ascontiguousarray(inputs['asc'][:, low:high].transpose(1, 0, 2))
            desc = np.ascontiguousarray(inputs['desc'][:, low:high].transpose(1, 0, 2))
            available = mask.any(axis=1) | np.any(asc != 0, axis=(1, 2)) | np.any(desc != 0, axis=(1, 2))
            if not available.all():
                raise ValueError(f"{tile['name']}: {int((~available).sum())} pixel senza osservazioni Sentinel")
            with torch.no_grad():
                embedding = encode_pixels(model, s2, doys['s2'], s1_asc_bands=asc, s1_asc_doys=doys['asc'],
                                          s1_desc_bands=desc, s1_desc_doys=doys['desc'], s2_masks=mask,
                                          batch_pixels=pixel_batch, device=device)
            if not np.isfinite(embedding).all():
                raise ValueError(f"{tile['name']}: embedding non finiti")
            output[low:high] = embedding
            output.flush()
            save_json(state_path, {**signature, 'pixels_complete': high, 'pixels_total': n, 'device': str(device)})
        done.append(tile['name'])
    return done
