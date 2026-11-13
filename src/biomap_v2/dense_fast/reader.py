# SCOPO
#   Lettore puntuale (SCL Sentinel-2, VV/VH Sentinel-1) per la pipeline r2-fast. È il lettore della r1
#   (data.Reader + raster_cache.NativeReader) con due sole differenze, entrambe senza effetto sui valori:
#   a) cache LRU dei blocchi nativi in RAM attiva (memory_cache_mib > 0): le tile vicine elaborate
#      insieme rileggono dalla RAM i blocchi già scaricati invece di riscaricarli;
#   b) righe/colonne calcolate con coords.rowcol_exact (stessi blocchi da 256 punti, stesso risultato
#      bit per bit) invece di ~3.700 chiamate rasterio.transform.rowcol per asset.
# COME VIENE RICHIAMATO
#   source = build_reader(profile, work_root, memory_cache_mib=4096)
#   values = source.values(item, 'SCL', points)   # stessa interfaccia della r1
# LOGICA IN ORDINE
#   1. Crea data.Reader con le stesse opzioni di rete della r1 (campaign/streaming.reader) ma con
#      memory_cache_mib configurabile; cache su disco sempre spenta (block_cache_gib = 0).
#   2. Sostituisce il NativeReader con FastNativeReader, che ridefinisce solo _plan.
#   3. _plan: stessa identità di cache, stessa proiezione PROJ per segmento (point_stride), righe e
#      colonne con rowcol_exact per segmento, raggruppamento per blocco con coords.block_groups.
# INPUT E OUTPUT
#   Input: item STAC, banda, punti WGS84 (n, 2). Output: array float64 con NaN o valore esterno come r1.
# PARAMETRI
#   memory_cache_mib: RAM per i blocchi nativi decodificati (default 4096).
# RISORSE
#   Rete e RAM entro il limite indicato.
# RIPRESA
#   Nessuno stato persistente (la cache è solo in memoria).
import hashlib

import numpy as np

from biomap_v2.raster_cache import NativeReader
from .coords import rowcol_exact, block_groups


class FastNativeReader(NativeReader):
    def _plan(self, meta, points):
        identity = (meta['crs'], tuple(meta['transform']), meta['height'], meta['width'], meta['block_height'],
                    meta['block_width'], hashlib.blake2b(points.tobytes(), digest_size=16).digest())
        with self.lock:
            cached = self.coordinates.pop(identity, None)
            if cached is not None:
                self.coordinates[identity] = cached
        if cached is not None:
            self.profile.count('coordinate_cache_hits')
            r, c, groups = cached
        else:
            with self.profile.span('coordinate_transform'):
                stride = self.owner.point_stride or len(points)
                rows, cols = [], []
                for start in range(0, len(points), stride):
                    selected = points[start:start + stride]
                    x, y = self._project(meta['crs'], selected)
                    rr, cc = rowcol_exact(meta['transform'], x, y)
                    rows.append(rr)
                    cols.append(cc)
                r = np.concatenate(rows) if rows else np.zeros(0, dtype='int32')
                c = np.concatenate(cols) if cols else np.zeros(0, dtype='int32')
                groups = block_groups(r, c, meta)
            with self.lock:
                from biomap_v2.raster_cache import MAX_PLANNED_POINTS
                self.coordinates[identity] = (r, c, groups)
                self.coordinate_points += len(points)
                while self.coordinate_points > MAX_PLANNED_POINTS and len(self.coordinates) > 1:
                    _, oldest = self.coordinates.popitem(last=False)
                    self.coordinate_points -= len(oldest[0])
        out = np.full(len(points), np.nan if meta['outside_masked'] else float(meta['outside']), dtype=np.float64)
        return out, r, c, groups


def reader_config(work_root, memory_cache_mib):
    return {'indicator': 'msa', 'data_root': str(work_root), 'dataset_namespace': 'reader',
            'sampling': {'pixels_per_cell': None},
            'acquisition': {'engine': 'optimized', 'memory_cache_mib': memory_cache_mib, 'block_cache_gib': 0,
                            'read_attempts': 6, 'http_retries': 3, 'connect_timeout_seconds': 20,
                            'read_timeout_seconds': 120}}


def build_reader(profile, work_root, memory_cache_mib=4096, reader_class=None):
    from biomap_v2.data import Reader
    source = (reader_class or Reader)(reader_config(work_root, memory_cache_mib), profile=profile)
    source.native = FastNativeReader(source, source.store, source.options, profile)
    return source
