# SCOPO
#   Acquisizione delle serie annuali Sentinel per un LOTTO di tile vicine, con risultato per tile identico
#   bit per bit a campaign/streaming.acquire_points + compose_day (protocollo storico r1).
#   Differenze rispetto alla r1, solo di organizzazione:
#   - un giorno viene elaborato una volta per tutte le tile del lotto che lo contengono: lo stesso
#     thread legge la stessa scena/banda per tutte le tile di seguito, così blocchi nativi (cache RAM del
#     lettore) e blocchi GDAL (handle riusato) si scaricano una volta sola;
#   - S2, S1 ascendente e S1 discendente vanno nello stesso pool di thread insieme, invece di tre fasi
#     in sequenza con attesa del giorno più lento alla fine di ciascuna.
# COME VIENE RICHIAMATO
#   results = acquire_batch(jobs, source, handles, executor)
#   jobs: lista di TileJob(name, pixels, streams, day_dir, identity); results: dict nome -> raw come r1.
# LOGICA IN ORDINE (per sorgente e giorno)
#   1. Membri = tile del lotto che hanno quel giorno nel PROPRIO catalogo e non lo hanno già in cache
#      (tmp/<aoi>/<tile>/days/<sorgente>_<giorno>.npz con identità e lista scene, come r1).
#   2. Ogni tile tiene la propria lista di scene del giorno, ordinata come la r1:
#      (eo:cloud_cover o 0, id). Le chiavi di tutte le tile si fondono in un unico ordine globale: per
#      ogni tile l'ordine relativo delle sue scene resta identico a quello della r1.
#   3. Per ogni chiave, per ogni tile che ha quella scena, si applica il corpo di compose_day r1:
#      S2: pixel ancora non validi; stop della tile se tutti validi; SCL al pixel (nearest) non valido
#      se in {0,1,2,3,8,9} o non finito; 10 bande riproiettate bilineari sulla finestra dei pixel
#      rimasti; validi se tutte le bande finite; clip 0..65535 -> uint16.
#      S1: VV e VH riempiti separatamente con la prima scena il cui valore convertito
#      (20*log10, +50, *200, int16) è != 0; stop della tile quando entrambe le bande sono piene.
#   4. Ogni giorno nuovo viene salvato per tile in modo atomico (ripresa dal giorno).
#   5. Per tile: stack dei giorni in ordine di data, DOY reali (366 il 31/12/2020), mask S2.
# INPUT E OUTPUT
#   Input: oggetti Pixels (campaign/streaming), cataloghi per tile, lettore e HandlePool.
#   Output: per tile dict {s2, s2_doys, mask, asc, asc_doys, desc, desc_doys} come la r1.
# PARAMETRI
#   executor: ThreadPoolExecutor condiviso; il numero di thread è il limite globale di richieste.
# RISORSE
#   Rete; RAM proporzionale a pixel del lotto x giorni.
# RIPRESA
#   I giorni già salvati con identità uguale vengono riletti; nessun giorno viene rifatto.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from biomap_v2.data import BANDS
from biomap_v2.campaign.streaming import S2_INVALID_SCL, s1_storage, doy
from .io_safe import atomic_npz, cached

SOURCES = ['s2', 'asc', 'desc']


@dataclass
class TileJob:
    name: str
    pixels: object
    streams: dict
    day_dir: Path
    identity: dict


def scene_key(item):
    return (item['properties'].get('eo:cloud_cover', 0) or 0, item['id'])


def _day_tag(job, source_name, day, items):
    return {**job.identity, 'source': source_name, 'day': day, 'scenes': sorted(x['id'] for x in items)}


def _day_path(job, source_name, day):
    return Path(job.day_dir) / f'{source_name}_{day}.npz'


class _Member:
    def __init__(self, pixels, items, is_s2):
        self.pixels = pixels
        self.items = sorted(items, key=scene_key)
        self.position = 0
        self.done = False
        n = len(pixels)
        self.bands = np.zeros((n, 10 if is_s2 else 2), dtype=np.uint16 if is_s2 else np.int16)
        self.valid = np.zeros(n, dtype=bool)


def _s2_step(member, item, source, handles):
    pixels = member.pixels
    need = np.flatnonzero(~member.valid)
    if not len(need):
        member.done = True
        return
    scl = source.values(item, 'SCL', pixels.points[need])
    need = need[np.isfinite(scl) & ~np.isin(scl, S2_INVALID_SCL)]
    if not len(need):
        return
    r, c = pixels.rows[need], pixels.cols[need]
    r0, c0 = int(r.min()), int(c.min())
    shape = (int(r.max()) - r0 + 1, int(c.max()) - c0 + 1)
    window = pixels.transform * rasterio.Affine.translation(c0, r0)
    raw = np.stack([handles.reproject_band(item, b, window, shape, pixels.crs)[r - r0, c - c0] for b in BANDS], axis=1)
    take = np.isfinite(raw).all(axis=1)
    member.bands[need[take]] = np.clip(raw[take], 0, 65535).astype(np.uint16)
    member.valid[need[take]] = True


def _s1_step(member, item, source):
    bands = member.bands
    for b, name in enumerate(['vv', 'vh']):
        need = np.flatnonzero(bands[:, b] == 0)
        if not len(need):
            continue
        stored = s1_storage(source.values(item, name, member.pixels.points[need]))
        fill = stored != 0
        bands[need[fill], b] = stored[fill]
    if (bands != 0).all():
        member.done = True


# Un giorno per più tile: ordine globale per chiave, ordine per tile invariato rispetto alla r1.
def compose_day_multi(source_name, members, source, handles):
    is_s2 = source_name == 's2'
    keys = sorted({scene_key(item) for m in members for item in m.items})
    for key in keys:
        for m in members:
            while not m.done and m.position < len(m.items) and scene_key(m.items[m.position]) == key:
                item = m.items[m.position]
                m.position += 1
                if is_s2:
                    _s2_step(m, item, source, handles)
                else:
                    _s1_step(m, item, source)
    results = []
    for m in members:
        valid = (m.bands != 0).any(axis=1) if not is_s2 else m.valid
        results.append((m.bands, valid))
    return results


def _day_job(source_name, day, jobs, source, handles):
    pending = []
    loaded = {}
    for job in jobs:
        items = job.streams[source_name][day]
        path = _day_path(job, source_name, day)
        tag = _day_tag(job, source_name, day, items)
        if cached(path, tag):
            with np.load(path) as z:
                loaded[job.name] = (z['bands'], z['valid'])
        else:
            pending.append((job, items, path, tag))
    if pending:
        members = [_Member(job.pixels, items, source_name == 's2') for job, items, _, _ in pending]
        for (job, _, path, tag), (bands, valid) in zip(pending, compose_day_multi(source_name, members, source, handles)):
            atomic_npz(path, tag, bands=bands, valid=valid)
            loaded[job.name] = (bands, valid)
    return loaded


def acquire_batch(jobs, source, handles, executor):
    futures = {}
    for source_name in SOURCES:
        days = sorted({day for job in jobs for day in job.streams[source_name]})
        for day in days:
            members = [job for job in jobs if day in job.streams[source_name]]
            futures[(source_name, day)] = executor.submit(_day_job, source_name, day, members, source, handles)
    parts = {(source_name, day): future.result() for (source_name, day), future in futures.items()}
    results = {}
    for job in jobs:
        n = len(job.pixels)
        raw = {}
        for source_name in SOURCES:
            days = sorted(job.streams[source_name].items())
            day_parts = [parts[(source_name, day)][job.name] for day, _ in days]
            width, dtype = (10, np.uint16) if source_name == 's2' else (2, np.int16)
            raw[source_name] = (np.stack([p[0] for p in day_parts], axis=1) if day_parts
                                else np.zeros((n, 0, width), dtype=dtype))
            raw[source_name + '_doys'] = np.array([doy(day) for day, _ in days], dtype=np.int16)
            if source_name == 's2':
                raw['mask'] = (np.stack([p[1] for p in day_parts], axis=1) if day_parts
                               else np.zeros((n, 0), dtype=bool))
        results[job.name] = raw
    return results
