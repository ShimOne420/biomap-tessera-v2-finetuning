# SCOPO
#   Riproiezione bilineare delle bande Sentinel-2 riusando i dataset remoti già aperti, invece di
#   riaprire il COG per ogni banda di ogni scena di ogni tile (campaign/streaming.py: reproject_band).
#   Con un dataset aperto, la cache dei blocchi GDAL (GDAL_CACHEMAX) serve gli stessi blocchi sorgente a
#   tutte le tile vicine elaborate di seguito dallo stesso thread: i byte comuni si scaricano una volta.
# COME VIENE RICHIAMATO
#   pool = HandlePool(source); pool.reproject_band(item, band, transform, shape, crs); pool.close_all()
#   `source` è il lettore (data.Reader) che fornisce signed() e clear_tokens().
# LOGICA IN ORDINE
#   1. Ogni thread ha un proprio dizionario LRU (asset senza token -> dataset aperto, ora di apertura).
#      Un dataset non viene mai usato da un thread diverso da quello che l'ha aperto (rasterio/GDAL
#      non garantiscono la sicurezza tra thread sullo stesso handle).
#   2. Un handle più vecchio di `max_age_s` viene chiuso e riaperto con un token SAS nuovo.
#   3. Riproiezione IDENTICA alla r1: stesso Env GDAL, stesso array float32 di zeri, stessi
#      dst_transform/dst_crs, Resampling.bilinear, nessun nodata dichiarato.
#   4. Errore RasterioError: handle scartato, token azzerati, attesa min(2^tentativo, 60) s e nuovo
#      tentativo, fino a READ_ATTEMPTS (6) come nella r1; poi RuntimeError.
# INPUT E OUTPUT
#   Input: item STAC, nome banda, finestra di destinazione. Output: array float32 (shape della finestra).
# PARAMETRI
#   max_per_thread (default 48) handle aperti per thread; max_age_s (default 600) età massima.
# RISORSE
#   Rete; memoria GDAL limitata da GDAL_CACHEMAX (impostato dalla pipeline).
# RIPRESA
#   Nessuno stato persistente.
import threading
import time
from collections import OrderedDict

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

READ_ATTEMPTS = 6
REPROJECT_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',
                     GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_TIMEOUT='120')


class HandlePool:
    def __init__(self, source, max_per_thread=48, max_age_s=600, profile=None, sleep=time.sleep):
        self.source = source
        self.max_per_thread = max_per_thread
        self.max_age_s = max_age_s
        self.profile = profile
        self.sleep = sleep
        self.local = threading.local()
        self.lock = threading.Lock()
        self.pools = []
        self.stats = {'opens': 0, 'reuses': 0, 'expired': 0, 'dropped': 0}

    def _count(self, name):
        with self.lock:
            self.stats[name] += 1

    def _pool(self):
        pool = getattr(self.local, 'pool', None)
        if pool is None:
            pool = OrderedDict()
            self.local.pool = pool
            with self.lock:
                self.pools.append(pool)
        return pool

    def _get(self, key, href):
        pool = self._pool()
        entry = pool.pop(key, None)
        if entry is not None and time.monotonic() - entry[1] > self.max_age_s:
            entry[0].close()
            entry = None
            self._count('expired')
        if entry is None:
            entry = (rasterio.open(self.source.signed(href)), time.monotonic())
            self._count('opens')
        else:
            self._count('reuses')
        pool[key] = entry
        while len(pool) > self.max_per_thread:
            _, old = pool.popitem(last=False)
            old[0].close()
        return entry[0]

    def _drop(self, key):
        entry = self._pool().pop(key, None)
        if entry is not None:
            try:
                entry[0].close()
            except Exception:
                pass
            self._count('dropped')

    def reproject_band(self, item, band, transform, shape, crs):
        href = item['assets'][band]['href']
        key = href.split('?')[0]
        for attempt in range(READ_ATTEMPTS):
            try:
                with rasterio.Env(**REPROJECT_ENV):
                    ds = self._get(key, href)
                    out = np.zeros(shape, dtype=np.float32)
                    reproject(rasterio.band(ds, 1), out, dst_transform=transform, dst_crs=crs,
                              resampling=Resampling.bilinear)
                return out
            except rasterio.errors.RasterioError:
                self._drop(key)
                self.source.clear_tokens()
                if attempt + 1 == READ_ATTEMPTS:
                    raise RuntimeError(f'Riproiezione fallita: scena {item["id"]}, banda {band}') from None
                self.sleep(min(2 ** attempt, 60))

    # Chiusura a fine run, quando i thread di lavoro sono fermi.
    def close_all(self):
        with self.lock:
            pools, self.pools = self.pools, []
        for pool in pools:
            while pool:
                _, (ds, _) = pool.popitem()
                try:
                    ds.close()
                except Exception:
                    pass
