# Scopo: leggere la griglia GLOBIO MSA e assegnare coordinate a riga/colonna native.
# Fasi: apertura raster -> maschera valida -> indice locale delle celle -> celle lon/lat.
# Input: raster MSA unitless 10 arc-second. Output: array e trasformazione condivisi.
# Parametri: edge_op='trunc' conserva l'assegnazione storica V1; 'floor' e' la correzione.
# Risorse: CPU; nessuna scrittura sul raster originale.
# Ripresa: nessuno stato persistente, il raster e' la fonte di verita'.
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.transform import rowcol


@dataclass(frozen=True)
class TargetGrid:
    values: np.ndarray
    valid: np.ndarray
    cell_index: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    transform: rasterio.Affine

    @property
    def n_cells(self):
        return len(self.lon)

    @property
    def shape(self):
        return self.values.shape


def load_target(path):
    with rasterio.open(path) as dataset:
        values = dataset.read(1).astype(np.float64)
        transform = dataset.transform
        nodata = dataset.nodata
    valid = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        valid &= values != nodata
    valid &= (values >= 0.0) & (values <= 1.0)
    cell_index = np.full(values.shape, -1, dtype=np.int64)
    cell_index[valid] = np.arange(int(valid.sum()), dtype=np.int64)
    rows, cols = np.nonzero(valid)
    return TargetGrid(values=values, valid=valid, cell_index=cell_index, rows=rows, cols=cols,
                      lon=transform.c + (cols + 0.5) * transform.a,
                      lat=transform.f + (rows + 0.5) * transform.e,
                      transform=transform)


def edge_function(edge_op):
    if edge_op == 'trunc':
        return int
    if edge_op == 'floor':
        return np.floor
    raise ValueError("edge_op: 'trunc' oppure 'floor'")


def assign(transform, lon, lat, edge_op='trunc'):
    """Righe/colonne native con l'assegnazione storica (trunc) o corretta (floor)."""
    rows, cols = rowcol(transform, np.asarray(lon), np.asarray(lat), op=edge_function(edge_op))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def rows_cols_of_cells(grid, cell_ids):
    ids = np.asarray(cell_ids, dtype=np.int64)
    return grid.rows[ids], grid.cols[ids]
