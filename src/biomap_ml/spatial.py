# SCOPO: ricostruire la griglia reale dei 25 punti Sentinel e definire la CNN E2.
# INPUT: coordinate salvate in series.npz, confini della cella, embedding per punto.
# CALCOLI: collocazione nord-in-alto, maschera, aree geodetiche degli strati e pooling.
# OUTPUT: tensori [128,5,5], [1,5,5] e una stima MSA per cella GLOBIO nativa.
# Non si inventano pixel mancanti né un target MSA validato a 10 metri.
# Uso: importato da prepare/cache/train; verificabile con pytest tests/test_e2_spatial.py.
import numpy as np
import torch
from pyproj import Geod
from torch import nn


# Ogni centro deve corrispondere a un unico strato del reticolo regolare 5x5.
def lattice(points, cell):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 25:
        raise ValueError('Punti: attesa matrice N x 2 con 1..25 posizioni')
    if not np.isfinite(points).all():
        raise ValueError('Coordinate dei punti non finite')
    dx, dy = (cell.east-cell.west)/5, (cell.north-cell.south)/5
    col = np.rint((points[:, 0]-cell.west)/dx-.5).astype(int)
    south_row = np.rint((points[:, 1]-cell.south)/dy-.5).astype(int)
    if ((col < 0) | (col > 4) | (south_row < 0) | (south_row > 4)).any():
        raise ValueError('Punto fuori dalla cella')
    expected = np.column_stack([cell.west+(col+.5)*dx, cell.south+(south_row+.5)*dy])
    if not np.allclose(points, expected, atol=1e-9, rtol=0):
        raise ValueError('Punti non corrispondenti alla griglia stratificata 5x5')
    slots = (4-south_row)*5+col  # riga zero a nord, colonne da ovest a est.
    if len(np.unique(slots)) != len(slots):
        raise ValueError('Punti duplicati nella stessa posizione')
    return slots


# L'area di ogni strato è calcolata sull'ellissoide WGS84, non in gradi quadrati.
def stratum_weights(cell):
    geod = Geod(ellps='WGS84')
    xs = np.linspace(cell.west, cell.east, 6)
    ys = np.linspace(cell.north, cell.south, 6)
    areas = np.empty((5, 5), dtype=np.float64)
    for row in range(5):
        for col in range(5):
            area, _ = geod.polygon_area_perimeter(
                [xs[col], xs[col+1], xs[col+1], xs[col]],
                [ys[row], ys[row], ys[row+1], ys[row+1]])
            areas[row, col] = abs(area)
    return (areas/areas.sum()).astype(np.float32)


# Gli zeri sono solo placeholder delle feature: la maschera li distingue dai punti validi.
def make_grid(embeddings, points, cell):
    values = np.asarray(embeddings, dtype=np.float32)
    slots = lattice(points, cell)
    if values.shape != (len(slots), 128) or not np.isfinite(values).all():
        raise ValueError('Embedding per punto: attesi N x 128 finiti')
    grid = np.zeros((25, 128), dtype=np.float32)
    mask = np.zeros(25, dtype=np.float32)
    grid[slots] = values
    mask[slots] = 1
    return grid.reshape(5, 5, 128).transpose(2, 0, 1).copy(), mask.reshape(1, 5, 5), stratum_weights(cell)[None]


# Validazione delle unità e delle dimensioni prima della normalizzazione ufficiale.
def validate_raw(raw, cell):
    lattice(raw['points'], cell)
    n = len(raw['points'])
    for name, channels, dtype in [('s2', 10, np.uint16), ('asc', 2, np.int16), ('desc', 2, np.int16)]:
        arr, days = raw[name], raw[name+'_doys']
        if arr.ndim != 3 or arr.shape[0] != n or arr.shape[2] != channels or (arr.shape[1] and arr.dtype != dtype):
            raise ValueError(f'Serie {name}: shape/dtype non conforme')
        if days.ndim != 1 or len(days) != arr.shape[1] or ((days < 1) | (days > 365)).any():
            raise ValueError(f'DOY {name} non validi')
        # Il contratto ufficiale cap a 365 può assegnare lo stesso DOY a 30/31 dicembre 2020.
        if np.any(np.diff(days.astype(int)) < 0):
            raise ValueError(f'DOY non ordinati in {name}')
    if raw['mask'].shape != raw['s2'].shape[:2] or not np.isin(raw['mask'], [0, 1]).all():
        raise ValueError('Maschera S2 non valida')
    usable = raw['mask'].any(axis=1) | np.any(raw['asc'] != 0, axis=(1, 2)) | np.any(raw['desc'] != 0, axis=(1, 2))
    if not usable.all():
        raise ValueError('Punto privo di input informativo')


class SpatialCNN(nn.Module):
    """Stessa testa minimale prevista per E2 e per i successivi E4/E5."""

    # 129 canali: 128 componenti TESSERA più il canale esplicito di validità.
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(129, 32, 1), nn.Conv2d(32, 32, 3, padding=1),
                                    nn.GroupNorm(8, 32), nn.ReLU(), nn.Conv2d(32, 1, 1), nn.Sigmoid())

    # I pesi degli strati mancanti sono esclusi; l'area osservata viene rinormalizzata.
    def forward(self, features, mask, weights):
        if features.shape[1:] != (128, 5, 5) or mask.shape[1:] != (1, 5, 5):
            raise ValueError('CNN: shape diversa dal protocollo 5x5')
        area = mask*weights
        denominator = area.sum(dim=(1, 2, 3))
        if (denominator <= 0).any():
            raise ValueError('CNN: cella senza punti validi')
        pixel_prediction = self.layers(torch.cat([features*mask, mask], dim=1))
        return (pixel_prediction*area).sum(dim=(1, 2, 3))/denominator
