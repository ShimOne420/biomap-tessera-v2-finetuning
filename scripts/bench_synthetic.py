# SCOPO
#   Benchmark SIMULATO e completamente locale r1 (campaign.streaming) contro r2 (dense_fast): niente
#   internet, niente GPU, niente file di produzione. Un server HTTP locale serve GeoTIFF sintetici con
#   latenza per richiesta e banda limitata (default 100 Mbit/s, come il collegamento Ethernet misurato
#   sul PC il 12/11/2026). L'encoder è finto e deterministico, con un'attesa proporzionale ai pixel che
#   imita la GPU reale (0,327 ms/pixel misurati sulla campagna r1).
#   I numeri NON sono una misura su Planetary Computer: servono a confrontare le due architetture nelle
#   stesse condizioni controllate (tempo reale, richieste, byte scaricati) e a verificarne l'equivalenza.
# COME VIENE RICHIAMATO
#   python scripts\bench_synthetic.py all --root <cartella_vuota_fuori_produzione>
#       [--latency-ms 40] [--mbit 100] [--tiles-x 3] [--tiles-y 2] [--s2-days 16] [--gpu-ms-per-pixel 0.327]
#   I comandi `run` e i parametri --impl/--port sono interni (sottoprocessi lanciati da `all`).
# LOGICA IN ORDINE
#   1. generate: scene S2 (10 bande + SCL, DEFLATE, blocchi 256), S1 asc/desc (VV/VH float32), alcune
#      parziali; reticolo V1 sintetico EPSG:3035 a 30 m diviso in tile vicine con pixel sparsi; cataloghi
#      per tile con bbox diversi (alcune scene visibili solo da alcune tile), come nella campagna reale.
#   2. Server HTTP con Range, latenza fissa per richiesta e banda globale a token bucket; conta richieste
#      e byte inviati per ciascuna esecuzione.
#   3. run --impl legacy: campaign.streaming.stream in un SOTTOPROCESSO (cache GDAL vuote).
#      run --impl fast: dense_fast.pipeline.stream_fast in un altro sottoprocesso.
#   4. Confronto bit per bit dei tile.npz e dei v2.parquet; risultato in <root>\bench_synthetic.json.
# INPUT E OUTPUT
#   Output solo sotto --root: scenes/, reference/, legacy/, fast/, bench_synthetic.json.
# PARAMETRI
#   --latency-ms, --mbit, --tiles-x/--tiles-y, --tile-px (lato tile in pixel 30 m), --s2-days, --s1-days,
#   --gpu-ms-per-pixel, --io-workers (r2) e --legacy-io-threads (r1), --batch-pixels (r2).
# RISORSE
#   CPU e disco locale (qualche centinaio di MB). Nessuna rete esterna, nessuna GPU.
# RIPRESA
#   `all` rigenera tutto in una --root nuova; non riprende.
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

import numpy as np  # noqa: E402

UTM = 'EPSG:32632'
GRID_CRS = 'EPSG:3035'
X0, Y0 = 580000.0, 4882000.0


# ---------------------------------------------------------------- server con latenza e banda limitata
class Throttle:
    def __init__(self, mbit):
        self.rate = mbit * 1e6 / 8
        self.lock = threading.Lock()
        self.next_free = time.perf_counter()
        self.requests = 0
        self.bytes = 0

    def send(self, wfile, data):
        with self.lock:
            self.requests += 1
            self.bytes += len(data)
            start = max(time.perf_counter(), self.next_free)
            self.next_free = start + len(data) / self.rate
            finish = self.next_free
        delay = finish - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        wfile.write(data)


def make_handler(root, throttle, latency):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

        def do_HEAD(self):
            time.sleep(latency)
            path = Path(self.translate_path(self.path.split('?')[0]))
            if not path.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            with throttle.lock:
                throttle.requests += 1

        def do_GET(self):
            time.sleep(latency)
            path = Path(self.translate_path(self.path.split('?')[0]))
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            rng = self.headers.get('Range')
            start, end = 0, size - 1
            if rng:
                spec = rng.split('=', 1)[1]
                if ',' in spec:
                    self.send_error(416, 'multi-range non supportato dal server di benchmark')
                    return
                a, b = spec.split('-')
                start, end = int(a), (int(b) if b else size - 1)
                end = min(end, size - 1)
            with open(path, 'rb') as f:
                f.seek(start)
                data = f.read(end - start + 1)
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            throttle.send(self.wfile, data)
    return Handler


# ---------------------------------------------------------------- dati sintetici
def write_raster(path, array, transform, dtype, nodata=None):
    import rasterio
    profile = {'driver': 'GTiff', 'height': array.shape[0], 'width': array.shape[1], 'count': 1, 'dtype': dtype,
               'crs': UTM, 'transform': transform, 'tiled': True, 'blockxsize': 256, 'blockysize': 256,
               'compress': 'deflate'}
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as ds:
        ds.write(array.astype(dtype), 1)


def generate(args, root):
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import Transformer
    from biomap_v2.data import BANDS
    rng = np.random.default_rng(20261112)
    scenes = root / 'scenes'
    scenes.mkdir(parents=True, exist_ok=True)
    extent = args.tile_px * 30 * max(args.tiles_x, args.tiles_y) + 1200

    def scene(scene_id, kind, offset=(0, 0), size=extent):
        x0, y0 = X0 + offset[0], Y0 - offset[1]
        assets = {}
        smooth = lambda n: (np.cumsum(np.cumsum(rng.normal(0, 1, (n, n)), 0), 1) * 3).astype(np.float64)
        if kind == 's2':
            for band in BANDS:
                res = 20 if band in ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12') else 10
                n = int(size / res)
                a = np.clip(1500 + smooth(n) % 4000, 1, 9000)
                a[:, :4] = 0
                write_raster(scenes / f'{scene_id}_{band}.tif', a, from_origin(x0, y0, res, res), 'uint16')
                assets[band] = {'href': f'http://127.0.0.1:{{port}}/scenes/{scene_id}_{band}.tif'}
            n = int(size / 20)
            scl = np.where((smooth(n) % 10) < 2.5, 9, 4)
            write_raster(scenes / f'{scene_id}_SCL.tif', scl, from_origin(x0, y0, 20, 20), 'uint8')
            assets['SCL'] = {'href': f'http://127.0.0.1:{{port}}/scenes/{scene_id}_SCL.tif'}
        else:
            n = int(size / 10)
            for band in ('vv', 'vh'):
                a = (0.05 + (smooth(n) % 100) / 300).astype('float32')
                a[rng.random((n, n)) < 0.02] = 0
                write_raster(scenes / f'{scene_id}_{band}.tif', a, from_origin(x0, y0, 10, 10), 'float32', nodata=np.nan)
                assets[band] = {'href': f'http://127.0.0.1:{{port}}/scenes/{scene_id}_{band}.tif'}
        return assets

    def it(scene_id, day, assets, cloud=None, orbit=None):
        props = {'datetime': f'{day}T10:00:00Z'}
        if cloud is not None:
            props['eo:cloud_cover'] = cloud
        if orbit:
            props['sat:orbit_state'] = orbit
        return {'id': scene_id, 'collection': 'synthetic', 'properties': props, 'assets': assets}

    days = sorted({f'2020-{m:02d}-{d:02d}' for m, d in zip(rng.integers(1, 13, 400), rng.integers(1, 29, 400))})
    s2_days = days[:args.s2_days]
    s1_days = days[args.s2_days:args.s2_days + 2 * args.s1_days]
    s2 = {d: [it(f'S2_{i}', d, scene(f'S2_{i}', 's2'), cloud=float(rng.integers(0, 60)))] for i, d in enumerate(s2_days)}
    for i, d in enumerate(s2_days[::3]):
        s2[d].append(it(f'S2p_{i}', d, scene(f'S2p_{i}', 's2', offset=(extent / 2, 0), size=extent / 2), cloud=1.0))
    asc = {d: [it(f'S1A_{i}', d, scene(f'S1A_{i}', 's1'), orbit='ascending')] for i, d in enumerate(s1_days[::2])}
    desc = {d: [it(f'S1D_{i}', d, scene(f'S1D_{i}', 's1'), orbit='descending')] for i, d in enumerate(s1_days[1::2])}
    tx = Transformer.from_crs(UTM, GRID_CRS, always_xy=True)
    gx, gy = tx.transform(X0 + 400, Y0 - 400)
    width, height = args.tile_px * args.tiles_x, args.tile_px * args.tiles_y
    grid = from_origin(gx, gy, 30, 30)
    reference = root / 'reference' / 'aoi_s'
    reference.mkdir(parents=True, exist_ok=True)
    tiles, cell_ids, catalogs = [], {}, {}
    for tx_i in range(args.tiles_x):
        for ty_i in range(args.tiles_y):
            name = f'grid_{tx_i}_{ty_i}'
            rr, cc = np.meshgrid(np.arange(ty_i * args.tile_px, (ty_i + 1) * args.tile_px),
                                 np.arange(tx_i * args.tile_px, (tx_i + 1) * args.tile_px), indexing='ij')
            flat = np.sort(rng.choice((rr * width + cc).ravel(), size=int(args.tile_px ** 2 * 0.7), replace=False))
            local = (flat // width) // 33 * 1000 + (flat % width) // 33
            folder = reference / name
            folder.mkdir(exist_ok=True)
            (folder / 'geometry.json').write_text(json.dumps({'width': width, 'height': height, 'pixels': int(len(flat)),
                                                              'transform': list(grid)[:6], 'crs': GRID_CRS}))
            np.savez(folder / 'geometry.npz', flat_indices=flat, cell_ids=local)
            tiles.append(name)
            cell_ids[name] = local
            hide = (tx_i + ty_i) % 2 == 1          # cataloghi diversi tra tile vicine
            catalogs[name] = {'s2': {d: [x for x in v if not (hide and x['id'].startswith('S2p'))] for d, v in s2.items()},
                              'asc': asc, 'desc': desc}
    unique = np.unique(np.concatenate(list(cell_ids.values())))
    lookup = np.full(unique.max() + 1, -1, dtype=np.int64)
    lookup[unique] = np.arange(len(unique))
    np.save(reference / 'lookup.npy', lookup)
    counts = np.zeros(len(unique), dtype=np.int64)
    for local in cell_ids.values():
        np.add.at(counts, lookup[local], 1)
    frame = [{'cell_id': f'c{c}', 'aoi_id': 'aoi_s', 'partition': 'train', 'cv_fold': int(c % 3), 'lon': 10.0, 'lat': 44.0,
              'target': 0.5, 'n_valid': int(counts[i]), 'n_total': 1089, 'valid_fraction': counts[i] / 1089}
             for i, c in enumerate(unique)]
    (root / 'bench_inputs.json').write_text(json.dumps({'spec': {'areas': [{'aoi_id': 'aoi_s', 'tiles': tiles}]},
                                                        'frame': frame, 'catalogs': catalogs}))
    size = sum(p.stat().st_size for p in scenes.glob('*.tif'))
    return {'tiles': len(tiles), 'pixels': int(sum(len(v) for v in cell_ids.values())), 'scene_files': len(list(scenes.glob('*.tif'))),
            'scene_bytes': size, 's2_scenes': sum(len(v) for v in s2.values()), 's1_scenes': len(asc) + len(desc)}


# ---------------------------------------------------------------- esecuzione di una implementazione (sottoprocesso)
def run_impl(args, root):
    import pandas as pd
    from biomap_v2 import data as data_module
    from biomap_v2.common import digest
    from biomap_v2.campaign import streaming as legacy
    from biomap_v2.campaign import extraction
    from biomap_v2.dense_fast import pipeline as fast
    inputs = json.loads((root / 'bench_inputs.json').read_text().replace('{port}', str(args.port)))
    spec, frame, catalogs = inputs['spec'], pd.DataFrame(inputs['frame']), inputs['catalogs']
    checkpoint = root / 'fake.pt'
    checkpoint.write_bytes(b'fake')
    data_module.Reader.signed = lambda self, href: href
    data_module.Reader.clear_tokens = lambda self: None
    legacy.WEIGHTS_SHA256 = fast.WEIGHTS_SHA256 = digest(checkpoint)
    legacy.catalog = lambda tile, pixels, folder: (catalogs[Path(tile).name], pixels.edge_bbox())
    legacy.load_model = lambda a: 'fake-model'
    gpu = args.gpu_ms_per_pixel / 1000

    def fake_infer(model, raw, device, batch, amp=False):
        time.sleep(len(raw['s2']) * gpu)
        s2 = raw['s2'].astype(np.float32)
        mask = raw['mask'].astype(np.float32)
        s1 = np.concatenate([raw['asc'], raw['desc']], axis=1).astype(np.float32)
        feats = np.concatenate([(s2 * mask[..., None]).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1),
                                s1.sum(axis=1), mask.sum(axis=1, keepdims=True)], axis=1)
        return np.sin(np.tile(feats, 10)[:, :128] * 1e-3).astype(np.float32)
    extraction.infer = fake_infer
    common = dict(reference=str(root / 'reference'), checkpoint=str(checkpoint), device='cpu', chunk_size=4096,
                  pixel_batch=256, keep_pixel_embeddings=False, amp=False, aoi=None, max_tiles=None)
    work = root / args.impl
    started = time.perf_counter()
    if args.impl == 'legacy':
        legacy.stream(SimpleNamespace(work_root=str(work), io_threads=args.legacy_io_threads, **common), spec, frame)
    else:
        os.environ['CPL_VSIL_CURL_CACHE_SIZE'] = str(args.vsi_cache_mib * 1024 * 1024)
        fast.stream_fast(SimpleNamespace(work_root=str(work), io_workers=args.io_workers, batch_pixels=args.batch_pixels,
                                         prefetch=1, memory_cache_mib=args.memory_cache_mib, gdal_cache_mib=args.gdal_cache_mib,
                                         vsi_cache_mib=args.vsi_cache_mib, tiles=None, **common), spec, frame)
    wall = time.perf_counter() - started
    (root / f'{args.impl}_wall.json').write_text(json.dumps({'wall_seconds': wall}))


def compare(root):
    import pandas as pd
    rows = []
    for tile in sorted(p.parent.name for p in (root / 'legacy' / 'chunks' / 'aoi_s').glob('*/tile.npz')):
        with np.load(root / 'legacy' / 'chunks' / 'aoi_s' / tile / 'tile.npz') as a, \
                np.load(root / 'fast' / 'chunks' / 'aoi_s' / tile / 'tile.npz') as b:
            rows.append(all(np.array_equal(a[k], b[k]) for k in ('cell', 'sums', 'counts')))
    p1, p2 = pd.read_parquet(root / 'legacy' / 'v2.parquet'), pd.read_parquet(root / 'fast' / 'v2.parquet')
    return {'tiles_compared': len(rows), 'tiles_bitwise_equal': int(sum(rows)),
            'parquet_bitwise_equal': bool(p1.equals(p2))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['all', 'run'])
    p.add_argument('--root', required=True)
    p.add_argument('--impl', choices=['legacy', 'fast'])
    p.add_argument('--port', type=int)
    p.add_argument('--latency-ms', type=float, default=40)
    p.add_argument('--mbit', type=float, default=100)
    p.add_argument('--tiles-x', type=int, default=3)
    p.add_argument('--tiles-y', type=int, default=2)
    p.add_argument('--tile-px', type=int, default=60)
    p.add_argument('--s2-days', type=int, default=16)
    p.add_argument('--s1-days', type=int, default=10)
    p.add_argument('--gpu-ms-per-pixel', type=float, default=0.327)
    p.add_argument('--io-workers', type=int, default=16)
    p.add_argument('--legacy-io-threads', type=int, default=16)
    p.add_argument('--batch-pixels', type=int, default=1_500_000)
    p.add_argument('--memory-cache-mib', type=int, default=1024)
    p.add_argument('--gdal-cache-mib', type=int, default=1024)
    p.add_argument('--vsi-cache-mib', type=int, default=256)
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.command == 'run':
        return run_impl(args, root)
    if root.exists() and any(root.iterdir()):
        sys.exit(f'{root} non è vuota: usare una cartella nuova')
    root.mkdir(parents=True, exist_ok=True)
    info = generate(args, root)
    throttle = Throttle(args.mbit)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(root, throttle, args.latency_ms / 1000))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    results = {'conditions': {'latency_ms': args.latency_ms, 'mbit': args.mbit, 'gpu_ms_per_pixel': args.gpu_ms_per_pixel,
                              'io_workers_r2': args.io_workers, 'io_threads_r1': args.legacy_io_threads}, 'data': info}
    passthrough = ['--latency-ms', str(args.latency_ms), '--gpu-ms-per-pixel', str(args.gpu_ms_per_pixel),
                   '--io-workers', str(args.io_workers), '--legacy-io-threads', str(args.legacy_io_threads),
                   '--batch-pixels', str(args.batch_pixels), '--memory-cache-mib', str(args.memory_cache_mib),
                   '--gdal-cache-mib', str(args.gdal_cache_mib), '--vsi-cache-mib', str(args.vsi_cache_mib)]
    for impl in ['legacy', 'fast']:
        with throttle.lock:
            throttle.requests, throttle.bytes = 0, 0
        started = time.perf_counter()
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), 'run', '--root', str(root), '--impl', impl,
                                '--port', str(port), *passthrough], cwd=REPO)
        if code:
            sys.exit(f'run {impl} fallito ({code})')
        inner = json.loads((root / f'{impl}_wall.json').read_text())['wall_seconds']
        results[impl] = {'wall_seconds': inner, 'wall_seconds_with_process_start': time.perf_counter() - started,
                         'http_requests': throttle.requests, 'bytes_served': throttle.bytes}
    server.shutdown()
    results['speedup_wall'] = results['legacy']['wall_seconds'] / results['fast']['wall_seconds']
    results['bytes_ratio_legacy_over_fast'] = results['legacy']['bytes_served'] / max(1, results['fast']['bytes_served'])
    results['requests_ratio_legacy_over_fast'] = results['legacy']['http_requests'] / max(1, results['fast']['http_requests'])
    results['equivalence'] = compare(root)
    (root / 'bench_synthetic.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
