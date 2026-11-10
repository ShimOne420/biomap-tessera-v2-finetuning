# SCOPO: verificare che V2 denso usi davvero i medesimi pixel V1, non solo lo stesso conteggio.
# INPUT: tile V1 int8/scale/landmask, geometry e progresso V2, griglia GLOBIO originale.
# CONTROLLI: hash, checkpoint, coordinate nel CRS della tile, indici univoci, celle e copertura.
# OUTPUT: evidenza JSON portabile; nessuna modifica delle tile sorgenti o nuovi download.
import json
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from biomap_v2.common import WEIGHTS_SHA256, digest
from .grid import assign
from .v1_support import discover_tiles as v1_tiles
from .v2_support import discover_tiles as v2_tiles, PROTOCOL


# Ogni punto V2 deve essere un centro pixel V1 valido e appartenere alla stessa cella.
def verify_support(v1_root, v2_root, grid, reference):
    first = {t['stem']: t for t in v1_tiles(v1_root)}
    second = v2_tiles(v2_root, require_complete=False)
    if not second or any(not t['complete'] for t in second):
        raise ValueError('Tile V2 dense mancanti o parziali: completarle prima di creare il pacchetto')
    counts = np.zeros(grid.n_cells, dtype=np.int64)
    evidence = []
    for tile in second:
        if tile['name'] not in first:
            raise ValueError('Tile V2 senza corrispondente V1: '+tile['name'])
        old = first[tile['name']]
        spec = tile['spec']
        folder = tile['directory']
        progress = json.loads((folder/'encoder_progress.json').read_text(encoding='utf-8'))
        inputs = folder/'inputs_manifest.json'
        actual_geometry = digest(tile['geometry'])
        if actual_geometry != spec.get('geometry_sha256'):
            raise ValueError('Hash geometry V2 discordante: '+tile['name'])
        if digest(old['scales']) != spec.get('v1_scales_sha256') or digest(old['landmask']) != spec.get('landmask_sha256'):
            raise ValueError('Scale/landmask V1 diverse dal supporto V2: '+tile['name'])
        if progress.get('checkpoint_sha256') != WEIGHTS_SHA256 or progress.get('protocol') != PROTOCOL:
            raise ValueError('Provenienza encoder V2 incompatibile: '+tile['name'])
        if digest(inputs) != progress.get('inputs_sha256'):
            raise ValueError('Manifest input diverso da quello codificato: '+tile['name'])
        with np.load(tile['geometry'], allow_pickle=False) as z:
            flat, cells = z['flat_indices'].astype(np.int64), z['cell_ids'].astype(np.int64)
        if len(flat) != spec['pixels'] or len(flat) != len(cells) or len(np.unique(flat)) != len(flat):
            raise ValueError('Indici pixel duplicati o conteggi geometry incoerenti')
        height, width = int(spec['height']), int(spec['width'])
        if (flat < 0).any() or (flat >= height*width).any() or (cells < 0).any() or (cells >= grid.n_cells).any():
            raise ValueError('Indici pixel/cella fuori bounds')
        cropped = rasterio.Affine(*spec['transform'])
        with rasterio.open(old['landmask']) as ds:
            original, crs, old_shape = ds.transform, ds.crs, ds.shape
        if str(crs) != spec['crs'] or not np.allclose([cropped.a, cropped.b, cropped.d, cropped.e],
                                                     [original.a, original.b, original.d, original.e], atol=1e-9, rtol=0):
            raise ValueError('CRS o reticolo V1/V2 diverso')
        col0, row0 = (~original)*(cropped.c, cropped.f)
        if not np.allclose([col0, row0], np.rint([col0, row0]), atol=1e-6, rtol=0):
            raise ValueError('Crop V2 non allineato ai pixel V1')
        rr, cc = flat//width+round(row0), flat % width+round(col0)
        if (rr < 0).any() or (cc < 0).any() or (rr >= old_shape[0]).any() or (cc >= old_shape[1]).any():
            raise ValueError('Pixel V2 fuori dalla tile V1')
        scales = np.load(old['scales'], mmap_mode='r')
        if not np.isfinite(scales[rr, cc]).all():
            raise ValueError('V2 include pixel scartati da V1')
        x, y = original*(cc+.5, rr+.5)
        lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
        r, c = assign(grid.transform, lon, lat, 'trunc')
        if (r < 0).any() or (c < 0).any() or (r >= grid.shape[0]).any() or (c >= grid.shape[1]).any():
            raise ValueError('Pixel V2 fuori dalla griglia target')
        if not np.array_equal(grid.cell_index[r, c], cells):
            raise ValueError('Pixel V2 assegnati a celle GLOBIO diverse dal V1 storico')
        counts += np.bincount(cells, minlength=grid.n_cells)
        evidence.append({'tile': tile['name'], 'pixels': len(flat), 'geometry_sha256': actual_geometry,
                         'embeddings_sha256': digest(tile['embeddings']), 'inputs_sha256': digest(inputs),
                         'v1_scales_sha256': spec['v1_scales_sha256'], 'landmask_sha256': spec['landmask_sha256']})
        print(f'Supporto denso verificato: {tile["name"]}, {len(flat):,} pixel', flush=True)
    ids = reference.index.to_numpy(dtype=np.int64)
    if not np.array_equal(counts[ids], reference.n_valid.to_numpy(dtype=np.int64)):
        raise ValueError('Il supporto V2 non contiene tutti i pixel V1 delle celle confrontate')
    return {'status': 'passed', 'method': 'same_V1_pixel_centers_and_finite_scales',
            'edge_op': 'trunc_historical', 'n_cells': len(reference), 'pixels_in_reference_cells': int(counts[ids].sum()),
            'checkpoint_sha256': WEIGHTS_SHA256, 'tiles': evidence}
