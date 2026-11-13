# SCOPO
#   Estrazione V2 densa r2-fast: stessa semantica della campagna r1 (campaign/streaming.stream), con
#   acquisizione e codifica GPU SOVRAPPOSTE e tile vicine acquisite a lotti. Pensata per le campagne
#   future (V2 Dense Mean, BII, mappe spaziali) e per il confronto A/B con la r1.
# COME VIENE RICHIAMATO
#   python scripts\dense_fast.py stream --reference <ref> --work-root <nuova_work_root> --checkpoint <pt>
#     --device cuda:0 [--io-workers 16] [--batch-pixels 1500000] [--prefetch 1] [--aoi ...] [--tiles ...]
#   La work-root NON può essere quella della campagna r1 (identity.refuse_production).
# LOGICA IN ORDINE
#   1. Piano = (AOI, tile) nell'ordine di reference.json, come la r1; tile già complete (tile.npz con
#      identità r2 e SHA256 verificato) vengono saltate.
#   2. Lotti: tile consecutive della stessa AOI fino a --batch-pixels pixel (le tile piccole viaggiano
#      insieme alle vicine; una tile grande forma un lotto da sola).
#   3. Thread PRODUTTORE: per ogni lotto, Pixels e catalogo STAC per tile (stesso bbox dei bordi della r1),
#      acquire_batch (acquisition.py), controllo di copertura come r1; mette il lotto in una coda
#      limitata a --prefetch lotti (limite di RAM).
#   4. Thread PRINCIPALE (GPU): per ogni tile del lotto, codifica a blocchi di --chunk-size pixel con
#      campaign.extraction.infer (FP32, AMP vietata, OOM -> batch/2 come r1), somma per cella con
#      np.add.at in float64 blocco per blocco (stesso ordine della r1), blocco salvato atomicamente,
#      poi tile.npz + marker, rilettura di verifica, cancellazione dei temporanei della tile.
#   5. Avanzamento in stream_progress.json: un errore di scrittura dell'avanzamento NON ferma il run
#      (era la causa dei due arresti della r1 l'11/11/2026).
#   6. Fine: se tutte le tile del piano sono complete e ogni cella ha conteggio = n_valid V1, somma dei
#      tile.npz NELL'ORDINE DEL PIANO (stesso ordine di somma float64 della r1) e scrittura di
#      v2.parquet + extraction_complete.json con method = PROTOCOL r2.
# INPUT E OUTPUT
#   Input: reference (geometry.json/.npz, lookup.npy, reference.json), checkpoint Medium (SHA256 verificato).
#   Output nella work-root: chunks/<aoi>/<tile>/tile.npz(+json), tmp/ (temporaneo), api/ (cataloghi),
#   stream_progress.json, stream_log.jsonl (una riga per tile e per lotto), stream_errors.jsonl,
#   stream_status.json, v2.parquet + extraction_complete.json a campagna completa.
# PARAMETRI
#   Vedere scripts/dense_fast.py (--io-workers, --batch-pixels, --prefetch, --memory-cache-mib,
#   --gdal-cache-mib, --vsi-cache-mib, --chunk-size, --pixel-batch, --keep-pixel-embeddings).
# RISORSE
#   Rete (limite: --io-workers richieste in parallelo), GPU (un solo thread), RAM: circa
#   (prefetch + 1) lotti x pixel x ~3,5 kB + cache (memory-cache + gdal-cache + vsi-cache).
# RIPRESA
#   Stesso comando. Riusa tile complete, blocchi codificati e giorni Sentinel già salvati (identità r2).
import json
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from biomap_v2.common import digest, WEIGHTS_SHA256
from .identity import PROTOCOL, code_identity, refuse_production
from .io_safe import save_json, atomic_npz, cached, append_jsonl


def configure_gdal(gdal_cache_mib=2048, vsi_cache_mib=512):
    # Letti da GDAL alla prima apertura remota: vanno impostati prima di qualunque lettura.
    os.environ.setdefault('GDAL_CACHEMAX', str(int(gdal_cache_mib)))
    os.environ.setdefault('CPL_VSIL_CURL_CACHE_SIZE', str(int(vsi_cache_mib) * 1024 * 1024))
    os.environ.setdefault('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    os.environ.setdefault('GDAL_HTTP_MULTIPLEX', 'YES')


def make_batches(plan, sizes, budget_pixels):
    batches, current, total = [], [], 0
    for key in plan:
        n = sizes[key]
        if current and (key[0] != current[-1][0] or total + n > budget_pixels):
            batches.append(current)
            current, total = [], 0
        current.append(key)
        total += n
    if current:
        batches.append(current)
    return batches


class Progress:
    def __init__(self, work, total_tiles, total_pixels, done_tiles, done_pixels):
        self.path = Path(work) / 'stream_progress.json'
        self.started = time.perf_counter()
        self.state = {'tiles_total': total_tiles, 'pixels_total': total_pixels, 'tiles_done': done_tiles,
                      'pixels_done': done_pixels}
        self.pixels_at_start = done_pixels
        self.failures = 0

    def update(self, where, tile_done, tile_total, add_pixels=0, add_tiles=0):
        s = self.state
        s['pixels_done'] += add_pixels
        s['tiles_done'] += add_tiles
        elapsed = time.perf_counter() - self.started
        rate = (s['pixels_done'] - self.pixels_at_start) / elapsed if elapsed > 0 else 0.
        record = {'protocol': PROTOCOL, 'current': where, 'tile_pixels_done': tile_done, 'tile_pixels': tile_total,
                  **s, 'elapsed_seconds_this_session': elapsed, 'pixels_per_second_this_session': rate,
                  'eta_hours_linear': (s['pixels_total'] - s['pixels_done']) / rate / 3600 if rate else None,
                  'progress_write_failures': self.failures,
                  'updated_utc': datetime.now(timezone.utc).isoformat()}
        try:
            save_json(self.path, record)
        except OSError as error:
            # L'avanzamento è informativo: mai un motivo per fermare l'estrazione.
            self.failures += 1
            print(f'Avviso: avanzamento non scritto ({error!r}); il run continua', flush=True)


def encode_tile(args, state, aoi, tile_name, raw, ids, base, tmp, progress):
    from biomap_v2.campaign.extraction import infer
    n = len(ids)
    sums = np.zeros((len(state['frame']), 128))
    counts = np.zeros(len(state['frame']), dtype=np.int64)
    keep = None
    if args.keep_pixel_embeddings:
        tmp.mkdir(parents=True, exist_ok=True)
        keep = np.lib.format.open_memmap(tmp / 'pixels.partial.npy', mode='w+', dtype=np.float32, shape=(n, 128))
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
            emb = state['infer'](state['model'], chunk, args.device, args.pixel_batch, False)
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
        progress.update(f'{aoi}/{tile_name}', high, n, add_pixels=high - low)
    if keep is None:
        return sums, counts, None
    # Unico riferimento al memmap: flush e chiusura qui. Su Windows un file ancora mappato non si può spostare
    # (WinError 32): a valle si passa solo il percorso del file parziale.
    keep.flush()
    del keep
    return sums, counts, tmp / 'pixels.partial.npy'


def finalize_tile(args, work, aoi, tile_name, base, sums, counts, partial, tmp, reference_tile):
    from biomap_v2.campaign.streaming import remove_temporary
    final = work / 'chunks' / aoi / tile_name / 'tile.npz'
    present = np.flatnonzero(counts)
    atomic_npz(final, base, cell=present, sums=sums[present], counts=counts[present])
    with np.load(final) as z:
        if not (np.array_equal(z['cell'], present) and np.array_equal(z['sums'], sums[present])
                and np.array_equal(z['counts'], counts[present])):
            raise ValueError(f'{final}: rilettura diversa dai valori calcolati; temporanei conservati')
    if partial is not None:
        from .spatial import save_pixel_embeddings
        save_pixel_embeddings(work, aoi, tile_name, partial, reference_tile, base)
    if tmp.exists():
        remove_temporary(tmp, work)


def _producer(args, work, ref, batches, state, out_queue, stop, executor, source, handles, timings):
    from biomap_v2.campaign.streaming import Pixels, catalog, coverage
    from .acquisition import TileJob, acquire_batch
    try:
        for number, batch in enumerate(batches):
            if stop.is_set():
                return
            started = time.perf_counter()
            jobs = []
            for aoi, tile_name in batch:
                tile = ref / aoi / tile_name
                pixels = Pixels(tile)
                streams, box = catalog(tile, pixels, work / 'api' / aoi / tile_name)
                base = dict(state['base'], geometry=digest(tile / 'geometry.npz'))
                jobs.append((aoi, box, base, TileJob(name=tile_name, pixels=pixels, streams=streams,
                                                     day_dir=work / 'tmp' / aoi / tile_name / 'days',
                                                     identity={**base, 'tile': tile_name})))
            raws = acquire_batch([j[3] for j in jobs], source, handles, executor)
            acquired = time.perf_counter() - started
            tiles = []
            for aoi, box, base, job in jobs:
                raw = raws.pop(job.name)
                stats = coverage(raw, len(job.pixels))
                if stats['pixels_without_any_input']:
                    raise ValueError(f'{aoi}/{job.name}: {stats["pixels_without_any_input"]} pixel senza alcun input '
                                     'S1/S2; supporto non ridotto silenziosamente')
                tiles.append({'aoi': aoi, 'tile': job.name, 'raw': raw, 'stats': stats, 'base': base, 'box': box,
                              'scenes': {k: sum(len(v) for v in job.streams[k].values()) for k in job.streams}})
            record = {'batch': number, 'tiles': [f'{a}/{t}' for a, t in batch], 'pixels': sum(len(j[3].pixels) for j in jobs),
                      'acquire_wall_seconds': acquired, 'reader': timings.reset(), 'handles': dict(handles.stats),
                      'finished_utc': datetime.now(timezone.utc).isoformat()}
            append_jsonl(work / 'stream_batches.jsonl', record)
            out_queue.put(('batch', number, tiles, acquired))
        out_queue.put(('end', None, None, None))
    except BaseException as error:
        out_queue.put(('error', error, traceback.format_exc(), None))


def stream_fast(args, spec, frame):
    from biomap_v2.campaign.extraction import infer
    from biomap_v2.campaign.streaming import Timings, load_model
    from .handles import HandlePool
    from .reader import build_reader
    work, ref = refuse_production(args.work_root), Path(args.reference)
    if not args.checkpoint or digest(args.checkpoint) != WEIGHTS_SHA256:
        raise ValueError('Checkpoint Medium SHA256 non conforme')
    configure_gdal(args.gdal_cache_mib, args.vsi_cache_mib)
    plan = [(a['aoi_id'], t) for a in spec['areas'] if not args.aoi or a['aoi_id'] in args.aoi for t in a['tiles']]
    if args.tiles:
        wanted = {tuple(x.split('/', 1)) for x in args.tiles}
        plan = [key for key in plan if key in wanted]
        if len(plan) != len(wanted):
            raise ValueError('Tile richieste non presenti nel reference')
    sizes = {key: json.loads((ref / key[0] / key[1] / 'geometry.json').read_text(encoding='utf-8'))['pixels'] for key in plan}
    base = {'protocol': PROTOCOL, 'year': 2020, 'checkpoint': WEIGHTS_SHA256, 'precision': 'fp32',
            'code': code_identity()}
    state = {'frame': frame, 'model': None, 'infer': infer, 'base': base}

    def complete(key):
        final = work / 'chunks' / key[0] / key[1] / 'tile.npz'
        return cached(final, dict(base, geometry=digest(ref / key[0] / key[1] / 'geometry.npz')))

    done = [key for key in plan if complete(key)]
    todo = [key for key in plan if key not in set(done)]
    if args.max_tiles is not None:
        todo = todo[:args.max_tiles]
    progress = Progress(work, len(plan), int(sum(sizes.values())), len(done), int(sum(sizes[k] for k in done)))
    batches = make_batches(todo, sizes, args.batch_pixels)
    timings = Timings()
    source = build_reader(timings, work, args.memory_cache_mib)
    handles = HandlePool(source)
    executor = ThreadPoolExecutor(max_workers=args.io_workers, thread_name_prefix='acq')
    out_queue, stop = queue.Queue(maxsize=max(1, args.prefetch)), threading.Event()
    producer = threading.Thread(target=_producer, name='producer', daemon=True,
                                args=(args, work, ref, batches, state, out_queue, stop, executor, source, handles, timings))
    session_started = time.perf_counter()
    gpu_busy = 0.
    producer.start()
    try:
        while True:
            waited = time.perf_counter()
            kind, number, payload, acquired = out_queue.get()
            gpu_wait = time.perf_counter() - waited
            if kind == 'end':
                break
            if kind == 'error':
                append_jsonl(work / 'stream_errors.jsonl', {'error': repr(number), 'traceback': payload,
                                                            'utc': datetime.now(timezone.utc).isoformat()})
                raise number
            if state['model'] is None:
                state['model'] = load_model(args)
            for tile in payload:
                aoi, tile_name = tile['aoi'], tile['tile']
                try:
                    started = time.perf_counter()
                    lookup = np.load(ref / aoi / 'lookup.npy')
                    with np.load(ref / aoi / tile_name / 'geometry.npz') as z:
                        ids = lookup[z['cell_ids']]
                    tmp = work / 'tmp' / aoi / tile_name
                    sums, counts, partial = encode_tile(args, state, aoi, tile_name, tile['raw'], ids, tile['base'], tmp, progress)
                    finalize_tile(args, work, aoi, tile_name, tile['base'], sums, counts, partial, tmp, ref / aoi / tile_name)
                    encoded = time.perf_counter() - started
                    gpu_busy += encoded
                    tile['raw'] = None
                    append_jsonl(work / 'stream_log.jsonl', {
                        'aoi_id': aoi, 'tile': tile_name, 'batch': number, 'batch_acquire_wall_seconds': acquired,
                        'encode_seconds': encoded, 'gpu_wait_before_batch_seconds': gpu_wait, 'catalog_bbox': tile['box'],
                        'scenes': tile['scenes'], 'coverage': tile['stats'],
                        'finished_utc': datetime.now(timezone.utc).isoformat()})
                    progress.update(f'{aoi}/{tile_name}', len(ids), len(ids), add_tiles=1)
                    print(f'Tile completa {aoi}/{tile_name}: {len(ids):,} pixel, codifica {encoded:.0f}s '
                          f'(lotto {number}, acquisizione lotto {acquired:.0f}s)', flush=True)
                except BaseException as error:
                    append_jsonl(work / 'stream_errors.jsonl', {'aoi_id': aoi, 'tile': tile_name, 'error': repr(error),
                                                                'traceback': traceback.format_exc(),
                                                                'utc': datetime.now(timezone.utc).isoformat()})
                    raise
    finally:
        stop.set()
        while producer.is_alive():
            try:
                out_queue.get(timeout=0.5)
            except queue.Empty:
                pass
        executor.shutdown(wait=True, cancel_futures=True)
        handles.close_all()
        source.close()
    wall = time.perf_counter() - session_started
    session = {'wall_seconds': wall, 'gpu_busy_seconds': gpu_busy, 'gpu_busy_fraction': gpu_busy / wall if wall else 0.,
               'tiles_this_session': len(todo), 'pixels_this_session': int(sum(sizes[k] for k in todo)),
               'handles': dict(handles.stats), 'utc': datetime.now(timezone.utc).isoformat()}
    append_jsonl(work / 'stream_sessions.jsonl', session)
    return finish(work, ref, plan, frame, args, base, session)


def finish(work, ref, plan, frame, args, base, session):
    sums, counts = np.zeros((len(frame), 128)), np.zeros(len(frame), dtype=np.int64)
    missing = []
    for aoi, tile_name in plan:
        final = work / 'chunks' / aoi / tile_name / 'tile.npz'
        if not cached(final, dict(base, geometry=digest(ref / aoi / tile_name / 'geometry.npz'))):
            missing.append(f'{aoi}/{tile_name}')
            continue
        with np.load(final) as z:
            sums[z['cell']] += z['sums']
            counts[z['cell']] += z['counts']
    eligible = frame.aoi_id.isin(args.aoi).to_numpy() if args.aoi else np.ones(len(frame), dtype=bool)
    complete = (counts == frame.n_valid.to_numpy()) & (counts > 0)
    status = {'protocol': PROTOCOL, 'tiles_planned': len(plan), 'tiles_missing': len(missing),
              'cells_complete': int(complete[eligible].sum()), 'cells_eligible': int(eligible.sum()),
              'session': session, 'utc': datetime.now(timezone.utc).isoformat()}
    save_json(work / 'stream_status.json', status)
    if missing or args.tiles or not complete[eligible].all():
        print(f'Streaming parziale o sottoinsieme: {json.dumps({k: v for k, v in status.items() if k != "session"})}', flush=True)
        return status
    from biomap_v2.campaign.state import MEANS
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
