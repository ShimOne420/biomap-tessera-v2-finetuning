# SCOPO
#   Estrazione V2 densa "in streaming" per la campagna 127.909 + Liguria: per ogni tile V1 del
#   reference scarica le serie Sentinel-1/2 2020 dei soli pixel della geometria, le codifica con il
#   TESSERA V2 Medium congelato (SHA256 verificato), salva somme e conteggi per cella e cancella gli
#   input temporanei. Così il disco necessario è quello di UNA tile alla volta, non ~600-700 GB.
#
# COME SI RICHIAMA
#   python scripts\dense_campaign.py stream --reference <ref> --work-root <work> --checkpoint <pt>
#     --device cuda:0 [--pixel-batch 256] [--io-threads 8] [--aoi ...] [--max-tiles N]
#     [--keep-pixel-embeddings]
#   La ripresa è lo stesso comando: le tile complete e i blocchi già codificati vengono riusati.
#
# LOGICA PER TILE (ordine reference.json -> areas -> tiles)
#   1. Se chunks/<aoi>/<tile>/tile.npz ha marker e identità uguali: riuso (sha verificato da cached()).
#   2. Acquisizione con il PROTOCOLLO DENSO V2 STORICO (Mac, medium_mpc_2020_all_dates_exact_v1_pixels),
#      ricostruito e verificato al bit su encoder_sample.npz del pacchetto ligure (legacy_check.py):
#        - catalogo STAC 2020 sul bbox dei BORDI dei pixel della geometria (non della tile intera);
#        - per giorno, scene in ordine (copertura nuvolosa, id); per pixel la prima scena valida;
#        - S2: SCL letto al pixel (nearest), non valido se in {0,1,2,3,8,9}; le 10 bande riproiettate
#          sul reticolo UTM V1 con interpolazione BILINEARE (nodata 0) e troncate a uint16; pixel
#          valido se SCL valido e tutte le bande > 0;
#        - S1 RTC: valore al pixel (nearest), valido se vv,vh > 0, storage (20*log10(x)+50)*200 int16:
#          la convenzione con cui il Medium è stato normalizzato (S1A_BAND_MEAN ~5909/3405);
#        - DOY = giorno dell'anno reale (il 31/12/2020 vale 366, come nello storico).
#      Questo differisce volutamente da inputs.acquire/data.power_to_storage (nearest, 10*log10, DOY<=365):
#      quelle regole NON riproducono gli input storici né la distribuzione dell'encoder (vedi runbook).
#      Le letture puntuali usano raster_cache.NativeReader (blocchi nativi, stesso valore di
#      rasterio.sample); le bande S2 si leggono solo per i pixel ancora da riempire (risultato identico).
#      Ogni giorno è un NPZ atomico in tmp/<aoi>/<tile>/days: un'interruzione riparte dal giorno.
#   3. Codifica a blocchi di --chunk-size pixel (FP32, no_grad/inference_mode, OOM -> batch/2).
#      Ogni blocco: NPZ atomico (cell, sums, counts) in tmp/<aoi>/<tile>/encoded.
#   4. Somma dei blocchi -> tile.npz atomico + marker JSON; il file viene riletto e confrontato con
#      i valori in memoria. Solo dopo si cancella tmp/<aoi>/<tile> (mai catalogo, reference, sorgenti,
#      embedding storici o risultati). Un blocco senza marker non è mai considerato completo.
#   5. Fine campagna: conteggi per cella == n_valid V1 per tutte le 132.900 celle, poi v2.parquet
#      (stesse colonne di extraction.extract) utilizzabile da probe/evaluate-liguria/compare.
#
# REGISTRI NELLA WORK-ROOT
#   stream_progress.json (avanzamento, velocità, ETA lineare), stream_log.jsonl (una riga per tile:
#   tempi, scene, copertura S1/S2), stream_errors.jsonl (eccezioni con traceback), stream_status.json.
import json
import shutil
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from pyproj import Transformer

from biomap_v2.common import ROOT, digest, save_json, WEIGHTS_SHA256
from .state import atomic_npz, cached, MEANS

PROTOCOL = 'dense127909-stream-legacy-exact-r1'
S2_INVALID_SCL = [0, 1, 2, 3, 8, 9]
READ_ATTEMPTS = 6
CODE_FILES = ['src/biomap_v2/campaign/streaming.py', 'src/biomap_v2/raster_cache.py', 'src/biomap_v2/data.py',
              'src/biomap_v2/reference.py', 'src/biomap_v2/encoder.py']


# Interfaccia minima richiesta da NativeReader: tempi e contatori, senza thread o file.
class Timings:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.counters = defaultdict(int)

    @contextmanager
    def span(self, stage, asset=None):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] += time.perf_counter() - started

    def count(self, name, n=1, asset=None):
        self.counters[name] += n

    def asset(self, key, item, band):
        pass

    def stage(self, phase):
        pass

    def reset(self):
        result = {'seconds': dict(self.seconds), 'counters': dict(self.counters)}
        self.seconds.clear()
        self.counters.clear()
        return result


def reader(timings, work):
    from biomap_v2.data import Reader
    cfg = {'indicator': 'msa', 'data_root': str(work), 'dataset_namespace': 'reader', 'sampling': {'pixels_per_cell': None},
           'acquisition': {'engine': 'optimized', 'memory_cache_mib': 0, 'block_cache_gib': 0, 'read_attempts': 6,
                           'http_retries': 3, 'connect_timeout_seconds': 20, 'read_timeout_seconds': 120}}
    return Reader(cfg, profile=timings)


def code_identity():
    return {name: digest(ROOT / name) for name in CODE_FILES}


# Pixel della geometria: centri in WGS84 (letture puntuali) e riga/colonna nel reticolo V1 (riproiezione).
class Pixels:
    def __init__(self, tile, positions=None):
        self.spec = json.loads((Path(tile) / 'geometry.json').read_text(encoding='utf-8'))
        with np.load(Path(tile) / 'geometry.npz') as z:
            flat = z['flat_indices'] if positions is None else z['flat_indices'][positions]
        self.rows, self.cols = flat // self.spec['width'], flat % self.spec['width']
        self.transform = rasterio.Affine(*self.spec['transform'])
        self.crs = self.spec['crs']
        x, y = self.transform * (self.cols + .5, self.rows + .5)
        self.points = np.column_stack(Transformer.from_crs(self.crs, 4326, always_xy=True).transform(x, y))

    def __len__(self):
        return len(self.rows)

    # Bbox WGS84 dei bordi dei pixel (angoli del rettangolo riga/colonna): regola storica del catalogo.
    def edge_bbox(self, all_pixels=None):
        rows, cols = (self.rows, self.cols) if all_pixels is None else all_pixels
        x, y = self.transform * (np.array([cols.min(), cols.max() + 1, cols.min(), cols.max() + 1]),
                                 np.array([rows.min(), rows.min(), rows.max() + 1, rows.max() + 1]))
        lon, lat = Transformer.from_crs(self.crs, 4326, always_xy=True).transform(x, y)
        return [float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat))]


def pixel_points(tile, positions):
    pixels = Pixels(tile, positions)
    return pixels.points, pixels.spec


def catalog(tile, pixels, folder):
    from biomap_v2.data import query
    with np.load(Path(tile) / 'geometry.npz') as z:
        flat = z['flat_indices']
    box = pixels.edge_bbox((flat // pixels.spec['width'], flat % pixels.spec['width']))
    cfg = {'year': 2020, 'data_root': str(folder), 'dataset_namespace': 'catalogs', 'acquisition': {}}
    cell = SimpleNamespace(west=box[0], south=box[1], east=box[2], north=box[3])
    folder.mkdir(parents=True, exist_ok=True)
    streams = {k: {} for k in ['s2', 'asc', 'desc']}
    for collection in ['sentinel-2-l2a', 'sentinel-1-rtc']:
        for item in query(cfg, cell, collection, folder / f'{collection}_edge_bbox.json'):
            props = item['properties']
            source = 's2' if collection == 'sentinel-2-l2a' else ('asc' if props.get('sat:orbit_state') == 'ascending' else 'desc')
            streams[source].setdefault(props['datetime'][:10], []).append(item)
    return streams, box


# Convenzione dB dello storico e della normalizzazione dell'encoder: 20*log10 del valore RTC MPC,
# calcolata in float32 come nello storico (in float64 cambia il troncamento di ~1 caso su 10^3).
def s1_storage(value):
    value = np.asarray(value, dtype=np.float32)
    out = np.zeros(value.shape, dtype=np.int16)
    ok = np.isfinite(value) & (value > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = (np.float32(20) * np.log10(value[ok]) + np.float32(50)) * np.float32(200)
    out[ok] = np.clip(db, 0, 32767).astype(np.int16)
    return out


# Riproiezione bilineare della banda sul sotto-reticolo V1 (stesso valore per pixel del reticolo intero).
# Come nello storico NON si dichiara nodata: i COG S2 MPC non hanno nodata nei metadati e gli zeri di bordo
# scena entrano nell'interpolazione (verificato su 3 casi reali: 247, 435, 530).
def reproject_band(source, item, band, transform, shape, crs):
    from rasterio.warp import reproject, Resampling
    for attempt in range(READ_ATTEMPTS):
        try:
            href = source.signed(item['assets'][band]['href'])
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff',
                              GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_TIMEOUT='120'):
                with rasterio.open(href) as ds:
                    out = np.zeros(shape, dtype=np.float32)
                    reproject(rasterio.band(ds, 1), out, dst_transform=transform, dst_crs=crs,
                              resampling=Resampling.bilinear)
            return out
        except rasterio.errors.RasterioError:
            source.clear_tokens()
            if attempt + 1 == READ_ATTEMPTS:
                raise RuntimeError(f'Riproiezione fallita: scena {item["id"]}, banda {band}') from None
            time.sleep(min(2 ** attempt, 60))


# Composizione di un giorno secondo il protocollo storico; si leggono solo i pixel non ancora validi.
def compose_day(source_name, items, pixels, source):
    from biomap_v2.data import BANDS
    n = len(pixels)
    is_s2 = source_name == 's2'
    bands = np.zeros((n, 10 if is_s2 else 2), dtype=np.uint16 if is_s2 else np.int16)
    valid = np.zeros(n, dtype=bool)
    for item in sorted(items, key=lambda x: (x['properties'].get('eo:cloud_cover', 0) or 0, x['id'])):
        if is_s2:
            need = np.flatnonzero(~valid)
            if not len(need):
                break
            scl = source.values(item, 'SCL', pixels.points[need])
            need = need[np.isfinite(scl) & ~np.isin(scl, S2_INVALID_SCL)]
            if not len(need):
                continue
            r, c = pixels.rows[need], pixels.cols[need]
            r0, c0 = int(r.min()), int(c.min())
            shape = (int(r.max()) - r0 + 1, int(c.max()) - c0 + 1)
            window = pixels.transform * rasterio.Affine.translation(c0, r0)
            raw = np.stack([reproject_band(source, item, b, window, shape, pixels.crs)[r - r0, c - c0] for b in BANDS], axis=1)
            take = np.isfinite(raw).all(axis=1)
            bands[need[take]] = np.clip(raw[take], 0, 65535).astype(np.uint16)
            valid[need[take]] = True
        else:
            # S1: ogni polarizzazione si riempie da sola con la prima scena il cui valore convertito è != 0.
            for b, name in enumerate(['vv', 'vh']):
                need = np.flatnonzero(bands[:, b] == 0)
                if not len(need):
                    continue
                stored = s1_storage(source.values(item, name, pixels.points[need]))
                fill = stored != 0
                bands[need[fill], b] = stored[fill]
            if (bands != 0).all():
                break
    if not is_s2:
        valid = (bands != 0).any(axis=1)
    return bands, valid


def doy(day):
    return datetime.fromisoformat(day).timetuple().tm_yday


# Serie annuali dei pixel; ogni giorno è una transazione atomica riutilizzabile.
def acquire_points(pixels, streams, day_dir, identity, source, threads=1):
    result = {}
    n = len(pixels)
    for name in ['s2', 'asc', 'desc']:
        days = sorted(streams[name].items())

        def one(entry, name=name):
            day, items = entry
            path = day_dir / f'{name}_{day}.npz'
            tag = {**identity, 'source': name, 'day': day, 'scenes': sorted(x['id'] for x in items)}
            if cached(path, tag):
                with np.load(path) as z:
                    return z['bands'], z['valid']
            bands, valid = compose_day(name, items, pixels, source)
            atomic_npz(path, tag, bands=bands, valid=valid)
            return bands, valid
        if threads > 1:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                parts = list(pool.map(one, days))
        else:
            parts = [one(entry) for entry in days]
        width, dtype = (10, np.uint16) if name == 's2' else (2, np.int16)
        result[name] = np.stack([p[0] for p in parts], axis=1) if parts else np.zeros((n, 0, width), dtype=dtype)
        result[name + '_doys'] = np.array([doy(day) for day, _ in days], dtype=np.int16)
        if name == 's2':
            result['mask'] = np.stack([p[1] for p in parts], axis=1) if parts else np.zeros((n, 0), dtype=bool)
    return result


def coverage(raw, n):
    s2 = raw['mask'].sum(axis=1)
    s1 = np.any(raw['asc'] != 0, axis=-1).sum(axis=1) + np.any(raw['desc'] != 0, axis=-1).sum(axis=1)
    return {'pixels': int(n), 's2_days': int(raw['mask'].shape[1]), 's1_asc_days': int(raw['asc'].shape[1]),
            's1_desc_days': int(raw['desc'].shape[1]), 's2_clear_mean': float(s2.mean()) if n else 0.,
            's2_clear_min': int(s2.min()) if n else 0, 's1_valid_mean': float(s1.mean()) if n else 0.,
            's1_valid_min': int(s1.min()) if n else 0, 'pixels_without_s2': int((s2 == 0).sum()),
            'pixels_without_any_input': int(((s2 == 0) & (s1 == 0)).sum())}


def load_model(args):
    from biomap_v2.encoder import load_encoder
    cfg = {'indicator': 'msa', 'model': {'device': args.device, 'checkpoint': str(args.checkpoint)}}
    model, _, _ = load_encoder(cfg, override=args.checkpoint)
    return model.eval().requires_grad_(False)


def append_jsonl(path, record):
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, default=str) + '\n')


# Cancellazione limitata alla cartella temporanea della tile, verificata per percorso.
def remove_temporary(path, work):
    path, base = Path(path).resolve(), (Path(work) / 'tmp').resolve()
    if base not in path.parents:
        raise ValueError(f'Rifiuto di cancellare fuori da {base}: {path}')
    shutil.rmtree(path)


def process_tile(args, aoi, tile_name, ids, state, source, timings):
    from .extraction import infer
    work, tile = Path(args.work_root), Path(args.reference) / aoi / tile_name
    spec = json.loads((tile / 'geometry.json').read_text(encoding='utf-8'))
    n = int(spec['pixels'])
    base = {'protocol': PROTOCOL, 'geometry': digest(tile / 'geometry.npz'), 'year': 2020, 'checkpoint': WEIGHTS_SHA256,
            'precision': 'fp32', 'code': state['code']}
    final = work / 'chunks' / aoi / tile_name / 'tile.npz'
    if cached(final, base):
        with np.load(final) as z:
            return z['cell'], z['sums'], z['counts'], None
    started = time.perf_counter()
    tmp = work / 'tmp' / aoi / tile_name
    pixels = Pixels(tile)
    streams, query_box = catalog(tile, pixels, work / 'api' / aoi / tile_name)
    raw = acquire_points(pixels, streams, tmp / 'days', {**base, 'tile': tile_name}, source, args.io_threads)
    acquired = time.perf_counter()
    stats = coverage(raw, n)
    if stats['pixels_without_any_input']:
        raise ValueError(f'{aoi}/{tile_name}: {stats["pixels_without_any_input"]} pixel senza alcun input S1/S2; '
                         'supporto non ridotto silenziosamente')
    if state['model'] is None:
        state['model'] = load_model(args)
    sums, counts = np.zeros((len(state['frame']), 128)), np.zeros(len(state['frame']), dtype=np.int64)
    keep = np.lib.format.open_memmap(tmp / 'pixels.partial.npy', mode='w+', dtype=np.float32, shape=(n, 128)) \
        if args.keep_pixel_embeddings else None
    for low in range(0, n, args.chunk_size):
        high = min(n, low + args.chunk_size)
        part = tmp / 'encoded' / f'{low}-{high}.npz'
        tag = {**base, 'tile': tile_name, 'low': low, 'high': high, 'pixel_embeddings': bool(args.keep_pixel_embeddings)}
        if cached(part, tag):
            with np.load(part) as z:
                cell, s, c = z['cell'], z['sums'], z['counts']
                emb = z['embeddings'] if 'embeddings' in z.files else None
        else:
            chunk = {k: (v[low:high] if k in {'s2', 'mask', 'asc', 'desc'} else v) for k, v in raw.items()}
            emb = infer(state['model'], chunk, args.device, args.pixel_batch, False)
            if emb.shape != (high - low, 128) or not np.isfinite(emb).all():
                raise ValueError(f'{aoi}/{tile_name}: embedding invalidi nel blocco {low}-{high}')
            selected = ids[low:high]
            if (selected < 0).any():
                raise ValueError('Pixel della geometria fuori dal reference')
            cell, inverse = np.unique(selected, return_inverse=True)
            s = np.zeros((len(cell), 128))
            np.add.at(s, inverse, emb.astype(np.float64))
            c = np.bincount(inverse, minlength=len(cell))
            extra = {'embeddings': emb} if args.keep_pixel_embeddings else {}
            atomic_npz(part, tag, cell=cell, sums=s, counts=c, **extra)
        sums[cell] += s
        counts[cell] += c
        if keep is not None:
            keep[low:high] = emb
        state['pixels_done'] += high - low
        progress(args, state, f'{aoi}/{tile_name}', high, n)
    present = np.flatnonzero(counts)
    atomic_npz(final, base, cell=present, sums=sums[present], counts=counts[present])
    with np.load(final) as z:
        if not (np.array_equal(z['cell'], present) and np.array_equal(z['sums'], sums[present])
                and np.array_equal(z['counts'], counts[present])):
            raise ValueError(f'{final}: rilettura diversa dai valori calcolati; temporanei conservati')
    if keep is not None:
        keep.flush()
        del keep
        destination = work / 'pixel_embeddings' / aoi / f'{tile_name}.npy'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / 'pixels.partial.npy'), destination)
        save_json(destination.with_suffix('.json'), {'sha256': digest(destination), 'geometry': base['geometry'],
                                                     'order': 'geometry.npz flat_indices', 'dtype': 'float32'})
    remove_temporary(tmp, work)
    log = {'aoi_id': aoi, 'tile': tile_name, 'acquire_seconds': acquired - started,
           'encode_seconds': time.perf_counter() - acquired, 'catalog_bbox': query_box, 'scenes': {k: sum(len(v) for v in streams[k].values()) for k in streams},
           'coverage': stats, 'reader': timings.reset(), 'finished_utc': datetime.now(timezone.utc).isoformat()}
    append_jsonl(work / 'stream_log.jsonl', log)
    print(f'Tile completa {aoi}/{tile_name}: {n:,} pixel, acquisizione {log["acquire_seconds"]:.0f}s, '
          f'codifica {log["encode_seconds"]:.0f}s', flush=True)
    return present, sums[present], counts[present], log


def progress(args, state, where, done, total):
    elapsed = time.perf_counter() - state['started']
    new = state['pixels_done'] - state['pixels_at_start']
    rate = new / elapsed if elapsed > 0 else 0.
    save_json(Path(args.work_root) / 'stream_progress.json', {
        'current': where, 'tile_pixels_done': done, 'tile_pixels': total, 'tiles_done': state['tiles_done'],
        'tiles_total': state['tiles_total'], 'pixels_done': state['pixels_done'], 'pixels_total': state['pixels_total'],
        'elapsed_seconds_this_session': elapsed, 'pixels_per_second_this_session': rate,
        'eta_hours_linear': (state['pixels_total'] - state['pixels_done']) / rate / 3600 if rate else None,
        'updated_utc': datetime.now(timezone.utc).isoformat()})


def stream(args, spec, frame):
    work, ref = Path(args.work_root), Path(args.reference)
    if not args.checkpoint or digest(args.checkpoint) != WEIGHTS_SHA256:
        raise ValueError('Checkpoint Medium SHA256 non conforme')
    if args.amp:
        raise ValueError('Lo streaming del confronto V1/V2 è solo FP32')
    timings = Timings()
    plan = [(a['aoi_id'], t) for a in spec['areas'] if not args.aoi or a['aoi_id'] in args.aoi for t in a['tiles']]
    pixels = {key: json.loads((ref / key[0] / key[1] / 'geometry.json').read_text(encoding='utf-8'))['pixels'] for key in plan}
    state = {'frame': frame, 'model': None, 'code': code_identity(), 'started': time.perf_counter(), 'tiles_done': 0,
             'tiles_total': len(plan), 'pixels_total': int(sum(pixels.values())), 'pixels_done': 0}
    for aoi, tile_name in plan:
        if (work / 'chunks' / aoi / tile_name / 'tile.json').exists():
            state['pixels_done'] += pixels[(aoi, tile_name)]
    state['pixels_at_start'] = state['pixels_done']
    sums, counts = np.zeros((len(frame), 128)), np.zeros(len(frame), dtype=np.int64)
    source = reader(timings, work)
    new_tiles = 0
    try:
        for aoi, tile_name in plan:
            if args.max_tiles is not None and new_tiles >= args.max_tiles:
                break
            lookup = np.load(ref / aoi / 'lookup.npy')
            with np.load(ref / aoi / tile_name / 'geometry.npz') as z:
                ids = lookup[z['cell_ids']]
            try:
                cell, s, c, log = process_tile(args, aoi, tile_name, ids, state, source, timings)
            except BaseException as error:
                append_jsonl(work / 'stream_errors.jsonl', {'aoi_id': aoi, 'tile': tile_name, 'error': repr(error),
                             'traceback': traceback.format_exc(), 'utc': datetime.now(timezone.utc).isoformat()})
                raise
            sums[cell] += s
            counts[cell] += c
            state['tiles_done'] += 1
            if log is not None:
                new_tiles += 1
    finally:
        source.close()
    eligible = frame.aoi_id.isin(args.aoi).to_numpy() if args.aoi else np.ones(len(frame), dtype=bool)
    complete = (counts == frame.n_valid.to_numpy()) & (counts > 0)
    status = {'tiles_done': state['tiles_done'], 'tiles_total': state['tiles_total'],
              'cells_complete': int(complete[eligible].sum()), 'cells_eligible': int(eligible.sum()),
              'utc': datetime.now(timezone.utc).isoformat()}
    save_json(work / 'stream_status.json', status)
    if state['tiles_done'] < state['tiles_total'] or not complete[eligible].all():
        print(f'Streaming parziale: {status}', flush=True)
        return status
    result = frame.loc[eligible, ['cell_id', 'aoi_id', 'partition', 'cv_fold', 'lon', 'lat', 'target', 'n_valid',
                                  'n_total', 'valid_fraction']].copy()
    means = sums[eligible] / counts[eligible, None]
    for i, column in enumerate(MEANS):
        result[column] = means[:, i]
    destination = work / ('v2_subset.parquet' if args.aoi else 'v2.parquet')
    result.to_parquet(destination, index=False)
    save_json(work / 'extraction_complete.json', {'sha256': digest(destination), 'amp': False, 'n_cells': len(result),
              'complete_campaign': len(result) == 132900, 'method': PROTOCOL})
    print(f'V2 denso completo: {destination} ({len(result)} celle)', flush=True)
    return status
