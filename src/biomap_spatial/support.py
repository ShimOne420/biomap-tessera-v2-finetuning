# Scopo: supporto spaziale autonomo, maschere, finestre non sovrapposte e provenienza patch/cella.
# Fasi: verifica griglia/CRS -> appartenenza del centro -> split delle sole posizioni eleggibili.
# Input: mask HW o poligono nel CRS dichiarato; output: finestre e mask senza resampling/halo.
# Parametri: max_patch_side, policy preservazione contributi legacy. Esempio: windows(mask,64).
# Risorse: CPU e piccole finestre; ripresa senza stato. Le geometrie V2 congelate non vengono rimappate.
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    height: int
    width: int
    transform: tuple
    crs: str

    def __post_init__(self):
        from pyproj import CRS
        if min(self.height, self.width) < 1 or len(self.transform) not in {6, 9}:
            raise ValueError('Griglia non valida')
        a, b, _, d, e, _ = self.transform[:6]
        if not np.isfinite(self.transform).all() or a <= 0 or e >= 0 or b != 0 or d != 0:
            raise ValueError('Servono griglie north-up senza rotazioni; nessun resampling implicito')
        CRS.from_user_input(self.crs)


def cell_mask(grid, polygon, polygon_crs):
    from pyproj import CRS
    from rasterio import Affine
    from rasterio.features import geometry_mask
    if CRS.from_user_input(grid.crs) != CRS.from_user_input(polygon_crs):
        raise ValueError('CRS incompatibili: nessuna riproiezione implicita')
    return geometry_mask([polygon], out_shape=(grid.height, grid.width),
        transform=Affine(*grid.transform[:6]), invert=True, all_touched=False)


def windows(mask, side):
    if mask.ndim != 2 or side < 1:
        raise ValueError('Mask/side non valido')
    rows, cols = np.where(mask)
    if not len(rows):
        raise ValueError('Supporto vuoto')
    for row in range(int(rows.min()), int(rows.max()) + 1, side):
        for col in range(int(cols.min()), int(cols.max()) + 1, side):
            local = mask[row:row + side, col:col + side]
            if local.any():
                yield row, col, local.copy()


def support_keys(grid, row, col, mask):
    from pyproj import CRS
    rr, cc = np.where(mask)
    a, _, x0, _, e, y0 = grid.transform[:6]
    # Exact native centers: no coordinate rounding or silent tolerance-based deduplication.
    crs = CRS.from_user_input(grid.crs).to_string()
    return [(crs, float(x0 + (c + col + .5) * a),
             float(y0 + (r + row + .5) * e)) for r, c in zip(rr, cc)]


def overlap_report(patches):
    seen = set()
    contributions = 0
    grids = set()
    for grid, row, col, mask in patches:
        grids.add((grid.crs, grid.transform[0], grid.transform[4]))
        keys = support_keys(grid, row, col, mask)
        contributions += len(keys)
        seen.update(keys)
    if len(grids) > 1:
        raise ValueError('CRS/risoluzioni incompatibili nella stessa cella')
    return {'contributions': contributions, 'unique_exact_centers': len(seen),
            'duplicate_contributions': contributions - len(seen), 'policy': 'preserve_legacy_contributions'}
