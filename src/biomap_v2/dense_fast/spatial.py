# SCOPO
#   Conservazione opzionale degli embedding V2 PER PIXEL con tutto ciò che serve a ricostruire mappe 2D
#   geograficamente allineate (esperimenti CNN spaziali), senza mai rimodellare vettori medi per cella.
# COME VIENE RICHIAMATO
#   Dalla pipeline con --keep-pixel-embeddings: save_pixel_embeddings(work, aoi, tile, percorso_parziale, ...).
#   Per l'analisi: emb, valid, transform, crs = pixel_map(work_root, aoi, tile, reference_root)
# LOGICA IN ORDINE
#   1. Alla fine della tile il memmap temporaneo (n_pixel, 128) float32, già chiuso da encode_tile, viene spostato
#      con os.replace (stesso volume) in
#      pixel_embeddings/<aoi>/<tile>.npy; l'ordine delle righe è quello di geometry.npz flat_indices.
#   2. Accanto si scrive un JSON con SHA256 del file, identità del protocollo, SHA256 di geometry.npz e
#      la georeferenziazione del reticolo V1 (width, height, transform, crs) letta da geometry.json.
#   3. pixel_map ricostruisce la finestra minima che contiene i pixel della geometria:
#      array (H, W, 128) float32 con NaN fuori geometria e maschera booleana di validità, più la
#      trasformazione affine della finestra (stesso reticolo V1, nessuna riproiezione).
# INPUT E OUTPUT
#   Input: memmap degli embedding, cartella della tile nel reference. Output: .npy + .json per tile.
# PARAMETRI
#   Nessuno oltre ai percorsi.
# RISORSE
#   Disco: 512 byte per pixel (circa 50 GB per l'intera campagna).
# RIPRESA
#   Il file finale compare solo a tile completata; un memmap parziale resta in tmp e si rifà.
import json
import os
from pathlib import Path

import numpy as np
import rasterio

from biomap_v2.common import digest
from .io_safe import save_json


def save_pixel_embeddings(work, aoi, tile_name, partial, reference_tile, identity):
    # partial: .npy già scritto e CHIUSO da encode_tile (un memmap aperto blocca lo spostamento su Windows).
    destination = Path(work) / 'pixel_embeddings' / aoi / f'{tile_name}.npy'
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    spec = json.loads((Path(reference_tile) / 'geometry.json').read_text(encoding='utf-8'))
    save_json(destination.with_suffix('.json'), {
        'sha256': digest(destination), 'identity': identity, 'order': 'geometry.npz flat_indices', 'dtype': 'float32',
        'geometry_sha256': digest(Path(reference_tile) / 'geometry.npz'),
        'grid': {'width': spec['width'], 'height': spec.get('height'), 'transform': spec['transform'], 'crs': spec['crs']}})
    return destination


def pixel_map(work, aoi, tile_name, reference_root):
    tile = Path(reference_root) / aoi / tile_name
    spec = json.loads((tile / 'geometry.json').read_text(encoding='utf-8'))
    with np.load(tile / 'geometry.npz') as z:
        flat = z['flat_indices']
    path = Path(work) / 'pixel_embeddings' / aoi / f'{tile_name}.npy'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if digest(path) != meta['sha256'] or digest(tile / 'geometry.npz') != meta['geometry_sha256']:
        raise ValueError(f'{path}: embedding o geometria non corrispondenti')
    emb = np.load(path, mmap_mode='r')
    rows, cols = flat // spec['width'], flat % spec['width']
    r0, c0 = int(rows.min()), int(cols.min())
    shape = (int(rows.max()) - r0 + 1, int(cols.max()) - c0 + 1)
    out = np.full(shape + (emb.shape[1],), np.nan, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    out[rows - r0, cols - c0] = emb
    valid[rows - r0, cols - c0] = True
    transform = rasterio.Affine(*spec['transform'][:6]) * rasterio.Affine.translation(c0, r0)
    return out, valid, transform, spec['crs']
