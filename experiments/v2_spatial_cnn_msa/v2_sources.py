# SCOPO
#   Sorgente dei VALORI dell'esperimento V2 spatial: embedding TESSERA V2 per pixel salvati da r2-fast
#   (scripts/dense_fast.py stream --keep-pixel-embeddings) in <v2_root>/pixel_embeddings/<aoi>/<tile>.npy.
#   Griglia, centri dei pixel, patch e maschere restano quelli dell'Esperimento B (tile V1): qui si collega
#   ogni pixel del reticolo V1 alla riga corrispondente del file V2.
# COME VIENE RICHIAMATO
#   from v2_sources import v2_spec, inspect_v2, row_lookup
#   patch_builder.build() per ogni tile V1 dell'inventario; dataset.SpatialDataset per leggere le righe.
# LOGICA IN ORDINE
#   1. v2_spec: percorsi del .npy/.json V2 e della geometria della tile nel reference (sola lettura).
#   2. inspect_v2: il JSON deve dichiarare protocollo r2-fast, fp32, pesi student medium e ordine
#      "geometry.npz flat_indices"; lo SHA256 della geometria deve coincidere; griglia (H, W, transform)
#      uguale a quella della tile V1; array (n, 128) float32 con n = len(flat_indices); con deep=True si
#      ricalcola lo SHA256 del .npy e lo si confronta con il JSON.
#   3. row_lookup: vettore lungo H*W con l'indice di riga V2 di ogni pixel V1 (-1 se il pixel non è stato
#      codificato). Le flat_indices sono univoche (verificato su tutte le tile): nessuna scelta arbitraria.
# INPUT E OUTPUT
#   Input: v2_root (work-root r2), reference (geometrie). Output: dizionari di specifica e array in memoria.
# PARAMETRI
#   deep: ricalcolo SHA256 del file V2 (circa 1 minuto per 50 GB su NVMe), usato una volta alla build.
# RISORSE
#   memmap in sola lettura; lookup int64 da 8 byte per pixel della tile (circa 7 MB per tile grande).
# RIPRESA
#   Nessuno stato: le funzioni sono pure rispetto ai file, che non vengono mai modificati.
import json
from pathlib import Path

import numpy as np

from runtime import V2_PIXEL_PROTOCOL, digest, signature


def v2_spec(cfg, aoi, name):
    if not cfg['data'].get('v2_root') or not cfg['data'].get('reference'):
        raise ValueError('Servono --v2-root (embedding V2 per pixel) e --reference (geometrie r1/r2)')
    npy = Path(cfg['data']['v2_root']) / 'pixel_embeddings' / aoi / f'{name}.npy'
    ref = Path(cfg['data']['reference']) / aoi / name
    return {'embedding': str(npy), 'meta': str(npy.with_suffix('.json')),
            'geometry': str(ref / 'geometry.npz'), 'geometry_json': str(ref / 'geometry.json')}


def inspect_v2(spec, v1_tile, deep=False):
    for key in ['embedding', 'meta', 'geometry', 'geometry_json']:
        if not Path(spec[key]).is_file():
            raise FileNotFoundError(f'V2 mancante ({key}): {spec[key]}')
    meta = json.loads(Path(spec['meta']).read_text(encoding='utf-8'))
    ident = meta.get('identity', {})
    from biomap_v2.common import WEIGHTS_SHA256
    if (ident.get('protocol') != V2_PIXEL_PROTOCOL or ident.get('precision') != 'fp32'
            or ident.get('checkpoint') != WEIGHTS_SHA256 or meta.get('order') != 'geometry.npz flat_indices'
            or meta.get('dtype') != 'float32'):
        raise ValueError('Embedding V2 con protocollo/precisione/pesi/ordine non attesi: ' + spec['embedding'])
    geometry_sha = digest(spec['geometry'])
    if meta.get('geometry_sha256') != geometry_sha or ident.get('geometry') != geometry_sha:
        raise ValueError('Geometria V2 diversa da quella del reference: ' + spec['embedding'])
    grid = json.loads(Path(spec['geometry_json']).read_text(encoding='utf-8'))
    if (grid['height'] != v1_tile['height'] or grid['width'] != v1_tile['width']
            or not np.allclose(grid['transform'][:6], v1_tile['transform'][:6], rtol=0, atol=1e-6)):
        raise ValueError('Reticolo V2 diverso dalla tile V1: ' + spec['embedding'])
    with np.load(spec['geometry']) as z:
        n = len(z['flat_indices'])
    emb = np.load(spec['embedding'], mmap_mode='r', allow_pickle=False)
    if emb.dtype != np.float32 or emb.shape != (n, 128):
        raise ValueError(f'Embedding V2 non conformi {emb.shape} {emb.dtype}: ' + spec['embedding'])
    del emb
    if deep and digest(spec['embedding']) != meta['sha256']:
        raise ValueError('SHA256 V2 discordante dal JSON: ' + spec['embedding'])
    stat = Path(spec['embedding']).stat()
    files = {'sha256': meta['sha256'], 'geometry_sha256': geometry_sha, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return {**spec, 'rows': n, 'files': files, 'fingerprint': signature({k: files[k] for k in ['sha256', 'geometry_sha256']}),
            'identity': ident}


def row_lookup(spec, height, width):
    with np.load(spec['geometry']) as z:
        flat = z['flat_indices']
    if len(np.unique(flat)) != len(flat) or flat.min() < 0 or flat.max() >= height * width:
        raise ValueError('flat_indices non univoche o fuori griglia: ' + spec['geometry'])
    lookup = np.full(height * width, -1, dtype=np.int64)
    lookup[flat] = np.arange(len(flat), dtype=np.int64)
    return lookup
