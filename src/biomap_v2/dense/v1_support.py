# Scopo: riprodurre l'aggregazione densa V1 (int8 dequantizzato) da tile grezze.
# Fasi: scoperta tile -> centri pixel -> CRS tile a WGS84 -> celle GLOBIO -> media densa.
# Input: global_0.1_degree_representation/2020/grid_*/*.npy e *_scales.npy, landmask tiff.
# Output: una riga per cella con lon, lat, target, copertura, n_valid, n_total, mean_*.
# Parametri: edge_op='trunc' per l'artefatto storico; 'floor' documenta la correzione.
# Risorse: CPU; strisce di 256 righe, accumulatori float64.
# Ripresa: nessuna scrittura intermedia; e' un controllo di equivalenza monolitico.
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

from .grid import assign

STRIP_ROWS = 256
DIMS = 128


def tile_centroids(shape, transform):
    rows, cols = shape
    col_offset = np.arange(cols) + 0.5
    row_offset = np.arange(rows) + 0.5
    x = transform.c + col_offset * transform.a
    y = transform.f + row_offset * transform.e
    return x, y


def dequantise(emb_int8, scales):
    valid = np.isfinite(scales)
    safe = np.where(valid, scales, 0.0)
    f32 = emb_int8.astype(np.float32) * safe[np.newaxis, :, :]
    f32[:, ~valid] = np.nan
    return f32.transpose(1, 2, 0)


def discover_tiles(tiles_root):
    """Tile V1 e relativa landmask a partire dalla radice tessera/2020."""
    root = Path(tiles_root)
    representation = root / 'global_0.1_degree_representation' / '2020'
    candidates = sorted(representation.glob('grid_*/*.npy'))
    embeddings = [p for p in candidates if not p.name.endswith('_scales.npy')]
    if not embeddings:
        raise FileNotFoundError(f'Nessun embedding V1 sotto {representation}')
    result = []
    for path in embeddings:
        stem = path.stem
        scales = path.with_name(stem + '_scales.npy')
        landmask = root / 'global_0.1_degree_tiff_all' / f'{stem}.tiff'
        if not scales.is_file():
            raise FileNotFoundError(f'Scale V1 assenti: {scales}')
        if not landmask.is_file():
            raise FileNotFoundError(f'Landmask V1 assente: {landmask}')
        result.append({'stem': stem, 'embeddings': path, 'scales': scales, 'landmask': landmask})
    return result


def aggregate_v1(tiles_root, target, edge_op='trunc', min_valid_fraction=0.8,
                 dimensions=DIMS, progress=None):
    tiles = discover_tiles(tiles_root)
    n_cells = target.n_cells
    shape = target.shape
    sums = np.zeros((n_cells, dimensions), dtype=np.float64)
    squares = np.zeros((n_cells, dimensions), dtype=np.float64)
    n_valid = np.zeros(n_cells, dtype=np.int64)
    n_total = np.zeros(n_cells, dtype=np.int64)

    for index, tile in enumerate(tiles, 1):
        with rasterio.open(tile['landmask']) as landmask:
            tile_transform = landmask.transform
            tile_crs = landmask.crs
        embedding = np.load(tile['embeddings'], mmap_mode='r')
        scales = np.load(tile['scales'], mmap_mode='r')
        height, width = embedding.shape[:2]
        if embedding.shape[2] != dimensions or scales.shape != (height, width):
            raise ValueError(f"Tile {tile['stem']}: shape incoerente")
        x, y = tile_centroids((height, width), tile_transform)
        transformer = Transformer.from_crs(tile_crs, 'EPSG:4326', always_xy=True)
        lon, lat = transformer.transform(*np.meshgrid(x, y))
        rows, cols = assign(target.transform, lon.ravel(), lat.ravel(), edge_op)
        rows = rows.reshape(height, width)
        cols = cols.reshape(height, width)
        in_bounds = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        row_clip = np.clip(rows, 0, shape[0] - 1)
        col_clip = np.clip(cols, 0, shape[1] - 1)
        inside = in_bounds & target.valid[row_clip, col_clip]
        cell_of_pixel = np.full((height, width), -1, dtype=np.int64)
        cell_of_pixel[inside] = target.cell_index[rows[inside], cols[inside]]

        n_total += np.bincount(cell_of_pixel[cell_of_pixel >= 0], minlength=n_cells)
        scale_valid = np.isfinite(scales)
        for start in range(0, height, STRIP_ROWS):
            stop = min(start + STRIP_ROWS, height)
            strip = dequantise(embedding[start:stop].transpose(2, 0, 1), scales[start:stop])
            cells = cell_of_pixel[start:stop]
            valid = (cells >= 0) & scale_valid[start:stop]
            rows_idx, cols_idx = np.nonzero(valid)
            if not len(rows_idx):
                continue
            cell_ids = cells[valid]
            values = strip[rows_idx, cols_idx]
            np.add.at(sums, cell_ids, values)
            np.add.at(squares, cell_ids, np.square(values))
            n_valid += np.bincount(cell_ids, minlength=n_cells)
        if progress is not None:
            progress(index, len(tiles), tile['stem'])

    mean = np.divide(sums, np.maximum(n_valid[:, None], 1),
                     out=np.full_like(sums, np.nan), where=n_valid[:, None] > 0)
    variance = np.divide(squares, np.maximum(n_valid[:, None], 1),
                         out=np.zeros_like(squares), where=n_valid[:, None] > 0) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    valid_fraction = np.divide(n_valid, np.maximum(n_total, 1),
                               out=np.zeros(n_cells, dtype=np.float64), where=n_total > 0)
    keep = ((n_total > 0) & (n_valid > 0) & (valid_fraction >= min_valid_fraction)
            & np.all(np.isfinite(mean), axis=1) & np.all(np.isfinite(std), axis=1))
    ids = np.flatnonzero(keep)
    records = {
        'cell_id': ids,
        'lon': target.lon[ids],
        'lat': target.lat[ids],
        'msa_overall': target.values[target.rows[ids], target.cols[ids]],
        'valid_fraction': valid_fraction[ids],
        'n_valid': n_valid[ids],
        'n_total': n_total[ids],
    }
    for band in range(dimensions):
        records[f'mean_{band:03d}'] = mean[ids, band]
    for band in range(dimensions):
        records[f'std_{band:03d}'] = std[ids, band]
    return pd.DataFrame(records).set_index('cell_id')


def compare_v1(recomputed, reference):
    common = recomputed.index.intersection(reference.index)
    result = {'n_recomputed': int(len(recomputed)), 'n_reference': int(len(reference)),
              'n_common': int(len(common)),
              'missing_in_recomputed': int(len(reference.index.difference(recomputed.index))),
              'extra_in_recomputed': int(len(recomputed.index.difference(reference.index)))}
    if not len(common):
        return result
    left = recomputed.loc[common]
    right = reference.loc[common]
    result['max_abs_lon'] = float(np.max(np.abs(left.lon - right.lon)))
    result['max_abs_lat'] = float(np.max(np.abs(left.lat - right.lat)))
    if 'msa_overall' in right:
        result['max_abs_target'] = float(np.max(np.abs(left.msa_overall - right.msa_overall)))
    result['n_valid_mismatch'] = int((left.n_valid != right.n_valid).sum())
    if 'n_total' in right:
        result['n_total_mismatch'] = int((left.n_total != right.n_total).sum())
    means = [f'mean_{i:03d}' for i in range(DIMS)]
    result['max_abs_mean'] = float(np.max(np.abs(left[means].to_numpy() - right[means].to_numpy())))
    stds = [f'std_{i:03d}' for i in range(DIMS)]
    if all(column in right for column in stds):
        result['max_abs_std'] = float(np.max(np.abs(left[stds].to_numpy() - right[stds].to_numpy())))
    return result
