# SCOPO
#   Righe/colonne dei punti nel reticolo di un raster, identiche BIT PER BIT al percorso r1
#   (raster_cache.NativeReader._plan), ma senza il costo Python di rasterio.transform.rowcol.
#   Nella campagna r1 questo calcolo arriva a 600-730 s (sommati sui thread) per tile grande:
#   ~3.700 chiamate rowcol da 256 punti per ogni asset con trasformazione nuova (ogni scena S1 RTC).
# COME VIENE RICHIAMATO
#   from biomap_v2.dense_fast.coords import rowcol_exact, block_groups
# LOGICA IN ORDINE
#   1. rowcol_exact replica AffineTransformer.rowcol di rasterio 1.4.3: matrice 3x3 dell'affine INVERSA,
#      input (3, n) = [x; y; 1], prodotto `M.dot(inp, out=inp)`, poi floor e cast a int32.
#   2. Il prodotto viene fatto sugli STESSI blocchi da 256 punti della r1: BLAS può arrotondare
#      diversamente l'ultimo bit se cambia la forma della matrice, quindi la forma resta identica.
#      Si elimina solo il sovraccarico Python (creazione del transformer, liste, controlli).
#   3. block_groups replica il raggruppamento per blocco nativo della r1 (stesse chiavi, stesso ordine).
# INPUT E OUTPUT
#   Input: affine a 6 coefficienti, x/y nel CRS del raster. Output: array int32 di righe e colonne.
# PARAMETRI
#   chunk=256: NON cambiarlo, fa parte dell'equivalenza bit per bit.
# RISORSE
#   Solo CPU e memoria proporzionale al numero di punti.
# RIPRESA
#   Funzioni pure.
import numpy as np
from rasterio.transform import Affine

CHUNK = 256


def rowcol_exact(transform6, xs, ys, chunk=CHUNK):
    matrix = np.empty((3, 3), dtype='float64')
    matrix.flat[:] = ~Affine(*transform6)
    xs = np.asarray(xs, dtype='float64')
    ys = np.asarray(ys, dtype='float64')
    n = len(xs)
    rows = np.empty(n, dtype='int32')
    cols = np.empty(n, dtype='int32')
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        work = np.empty((3, stop - start))
        work[0] = xs[start:stop]
        work[1] = ys[start:stop]
        work[2] = 1
        matrix.dot(work, out=work)
        rows[start:stop] = np.floor(work[1]).astype('int32')
        cols[start:stop] = np.floor(work[0]).astype('int32')
    return rows, cols


def block_groups(rows, cols, meta):
    groups = {}
    inside = np.flatnonzero((rows >= 0) & (rows < meta['height']) & (cols >= 0) & (cols < meta['width']))
    if len(inside):
        block_rows = rows[inside] // meta['block_height']
        block_cols = cols[inside] // meta['block_width']
        keys = np.stack([block_rows, block_cols], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inverse.ravel(), kind='stable')
        counts = np.bincount(inverse.ravel(), minlength=len(unique))
        for position, indices in zip(unique, np.split(inside[order], np.cumsum(counts)[:-1])):
            groups[(int(position[0]), int(position[1]))] = indices
    return groups
