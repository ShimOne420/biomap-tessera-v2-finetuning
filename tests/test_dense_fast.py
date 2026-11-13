# SCOPO
#   Test OFFLINE di equivalenza e robustezza della pipeline r2-fast contro la campagna r1 (legacy),
#   su raster sintetici locali: nessuna rete, nessuna GPU, nessun file di produzione.
# COME VIENE RICHIAMATO
#   python -m pytest -q tests\test_dense_fast.py      (oppure: python scripts\dense_fast.py verify-synthetic)
# LOGICA IN ORDINE
#   1. coords: rowcol_exact identico a rasterio.transform.rowcol a blocchi da 256 (anche fuori raster).
#   2. reader: FastNativeReader (cache RAM attiva, due passate) identico al NativeReader r1 su GeoTIFF locali.
#   3. acquisition: per due tile vicine con cataloghi diversi, acquire_points r1 (per tile) e
#      acquire_batch r2 (lotto) producono array IDENTICI (S2, mask, S1 asc/desc, DOY).
#   4. end-to-end: campaign.streaming.stream (r1) e pipeline.stream_fast (r2) sullo stesso reference
#      sintetico, con encoder finto deterministico per pixel: tile.npz e v2.parquet identici bit per bit.
#   5. io_safe: os.replace bloccato da un lettore Windows senza condivisione Delete viene ritentato;
#      un salvataggio fallito lascia intatto il file precedente e nessun temporaneo.
#   6. make_batches: tile consecutive della stessa AOI entro il limite di pixel.
# INPUT E OUTPUT
#   Input: raster generati in tmp_path. Output: nessuno oltre a tmp_path.
# PARAMETRI
#   Nessuno.
# RISORSE
#   CPU, qualche decina di MB di RAM, pochi secondi.
# RIPRESA
#   Test indipendenti e ripetibili (seed fissi).
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin, rowcol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from biomap_v2 import data as data_module  # noqa: E402
from biomap_v2.campaign import streaming as legacy  # noqa: E402
from biomap_v2.dense_fast import pipeline as fast_pipeline  # noqa: E402
from biomap_v2.dense_fast.acquisition import TileJob, acquire_batch  # noqa: E402
from biomap_v2.dense_fast.coords import rowcol_exact  # noqa: E402
from biomap_v2.dense_fast.handles import HandlePool  # noqa: E402
from biomap_v2.dense_fast.io_safe import save_json, replace_with_retry  # noqa: E402
from biomap_v2.dense_fast.reader import build_reader  # noqa: E402

UTM = 'EPSG:32632'
GRID_CRS = 'EPSG:3035'
X0, Y0 = 580000.0, 4882000.0       # angolo NW delle scene (UTM 32N, ~10.0E 44.1N)


@pytest.fixture(autouse=True)
def local_hrefs(monkeypatch):
    # Nessun token MPC: gli href dei test sono percorsi locali.
    monkeypatch.setattr(data_module.Reader, 'signed', lambda self, href: href)
    monkeypatch.setattr(data_module.Reader, 'clear_tokens', lambda self: None)


def write_raster(path, array, transform, dtype, crs=UTM, nodata=None):
    profile = {'driver': 'GTiff', 'height': array.shape[0], 'width': array.shape[1], 'count': 1, 'dtype': dtype,
               'crs': crs, 'transform': transform, 'tiled': True, 'blockxsize': 128, 'blockysize': 128}
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as ds:
        ds.write(array.astype(dtype), 1)
    return str(path)


def make_scene(folder, scene_id, kind, rng, offset=(0, 0), size_m=3000):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    x0, y0 = X0 + offset[0], Y0 - offset[1]
    assets = {}
    if kind == 's2':
        for band in data_module.BANDS:
            res = 20 if band in ('B05', 'B06', 'B07', 'B8A', 'B11', 'B12') else 10
            n = int(size_m / res)
            a = rng.integers(1, 9000, (n, n))
            a[:, :3] = 0                          # zeri di bordo scena (nessun nodata dichiarato, come MPC)
            a[rng.random((n, n)) < 0.01] = 0
            assets[band] = {'href': write_raster(folder / f'{scene_id}_{band}.tif', a, from_origin(x0, y0, res, res), 'uint16')}
        n = int(size_m / 20)
        scl = rng.choice([0, 1, 3, 4, 5, 6, 8, 9, 10, 11], size=(n, n), p=[.02, .02, .06, .3, .3, .1, .08, .07, .03, .02])
        assets['SCL'] = {'href': write_raster(folder / f'{scene_id}_SCL.tif', scl, from_origin(x0, y0, 20, 20), 'uint8')}
    else:
        n = int(size_m / 10)
        for band in ('vv', 'vh'):
            a = rng.uniform(0.0005, 0.5, (n, n)).astype('float32')
            a[rng.random((n, n)) < 0.05] = 0
            a[rng.random((n, n)) < 0.02] = np.nan
            assets[band] = {'href': write_raster(folder / f'{scene_id}_{band}.tif', a, from_origin(x0, y0, 10, 10), 'float32', nodata=np.nan)}
    return assets


def item(scene_id, day, assets, cloud=None, orbit=None):
    props = {'datetime': f'{day}T10:00:00Z'}
    if cloud is not None:
        props['eo:cloud_cover'] = cloud
    if orbit:
        props['sat:orbit_state'] = orbit
    return {'id': scene_id, 'collection': 'test', 'properties': props, 'assets': assets}


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic')
    rng = np.random.default_rng(20261112)
    scenes = root / 'scenes'
    s2 = {
        '2020-03-01': [item('S2A_1', '2020-03-01', make_scene(scenes, 'S2A_1', 's2', rng), cloud=12.5),
                       item('S2B_1', '2020-03-01', make_scene(scenes, 'S2B_1', 's2', rng, offset=(900, 600), size_m=2000), cloud=3.0)],
        '2020-07-15': [item('S2A_2', '2020-07-15', make_scene(scenes, 'S2A_2', 's2', rng), cloud=None)],
        '2020-12-31': [item('S2A_3', '2020-12-31', make_scene(scenes, 'S2A_3', 's2', rng), cloud=40.0),
                       item('S2C_3', '2020-12-31', make_scene(scenes, 'S2C_3', 's2', rng, offset=(-300, 200)), cloud=40.0)],
    }
    asc = {'2020-02-02': [item('S1A_1', '2020-02-02', make_scene(scenes, 'S1A_1', 's1', rng), orbit='ascending'),
                          item('S1A_1b', '2020-02-02', make_scene(scenes, 'S1A_1b', 's1', rng, offset=(500, 0)), orbit='ascending')],
           '2020-09-09': [item('S1A_2', '2020-09-09', make_scene(scenes, 'S1A_2', 's1', rng), orbit='ascending')]}
    desc = {'2020-05-05': [item('S1D_1', '2020-05-05', make_scene(scenes, 'S1D_1', 's1', rng), orbit='descending')]}
    # Reticolo V1 sintetico in EPSG:3035, 30 m; due tile vicine con pixel sparsi e cataloghi diversi.
    with rasterio.Env():
        from pyproj import Transformer
        tx = Transformer.from_crs(UTM, GRID_CRS, always_xy=True)
        gx, gy = tx.transform(X0 + 300, Y0 - 300)
    grid_transform = from_origin(gx, gy, 30, 30)
    width, height = 80, 80
    reference = root / 'reference'
    spec = {'areas': [{'aoi_id': 'aoi_x', 'tiles': ['grid_a', 'grid_b']}]}
    cells_per_tile = {}
    frame_rows = []
    all_cell_ids = []
    flat_by_tile = {}
    for t, cols in [('grid_a', range(0, 40)), ('grid_b', range(40, 80))]:
        rr, cc = np.meshgrid(np.arange(height), np.array(list(cols)), indexing='ij')
        flat = (rr * width + cc).ravel()
        flat = np.sort(rng.choice(flat, size=int(len(flat) * 0.6), replace=False))
        flat_by_tile[t] = flat
        local = (flat // width) // 10 * 8 + (flat % width) // 10          # celle 10x10 pixel
        cells_per_tile[t] = local
        all_cell_ids.extend(local.tolist())
    unique = np.unique(all_cell_ids)
    lookup = np.full(unique.max() + 1, -1, dtype=np.int64)
    lookup[unique] = np.arange(len(unique))
    (reference / 'aoi_x').mkdir(parents=True)
    np.save(reference / 'aoi_x' / 'lookup.npy', lookup)
    counts = np.zeros(len(unique), dtype=np.int64)
    for t, flat in flat_by_tile.items():
        folder = reference / 'aoi_x' / t
        folder.mkdir()
        (folder / 'geometry.json').write_text(json.dumps({'width': width, 'height': height, 'pixels': int(len(flat)),
                                                          'transform': list(grid_transform)[:6], 'crs': GRID_CRS}))
        np.savez(folder / 'geometry.npz', flat_indices=flat, cell_ids=cells_per_tile[t])
        np.add.at(counts, lookup[cells_per_tile[t]], 1)
    for i, cell in enumerate(unique):
        frame_rows.append({'cell_id': f'c{cell}', 'aoi_id': 'aoi_x', 'partition': 'train', 'cv_fold': int(cell % 3),
                           'lon': 10.0, 'lat': 44.0, 'target': 0.5, 'n_valid': int(counts[i]), 'n_total': 100,
                           'valid_fraction': counts[i] / 100})
    frame = pd.DataFrame(frame_rows)
    # Cataloghi per tile: grid_b non vede S2B_1 e S1A_1b (bbox diversi come nella campagna reale).
    catalogs = {'grid_a': {'s2': s2, 'asc': asc, 'desc': desc},
                'grid_b': {'s2': {d: [x for x in v if x['id'] != 'S2B_1'] for d, v in s2.items()},
                           'asc': {d: [x for x in v if x['id'] != 'S1A_1b'] for d, v in asc.items()}, 'desc': desc}}
    return SimpleNamespace(root=root, reference=reference, spec=spec, frame=frame, catalogs=catalogs)


def test_rowcol_exact_matches_rasterio_chunks():
    rng = np.random.default_rng(7)
    for t in [(10.0, 0, 600000.0, 0, -10.0, 5100000.0), (20.000037, 0.0, 399960.5, 0.0, -19.99991, 4900020.25)]:
        x = (t[2] + rng.uniform(-500, 130000, 100_003)).tolist()
        y = (t[5] - rng.uniform(-500, 130000, 100_003)).tolist()
        rows, cols = [], []
        for o in range(0, len(x), 256):
            rr, cc = rowcol(rasterio.Affine(*t), x[o:o + 256], y[o:o + 256])
            rows.extend(rr)
            cols.extend(cc)
        r1, c1 = rowcol_exact(t, x, y)
        assert np.asarray(rows).dtype == r1.dtype
        assert np.array_equal(np.asarray(rows), r1) and np.array_equal(np.asarray(cols), c1)


def test_fast_reader_matches_legacy(synthetic, tmp_path):
    pixels = legacy.Pixels(synthetic.reference / 'aoi_x' / 'grid_a')
    old = legacy.reader(legacy.Timings(), tmp_path / 'legacy')
    new = build_reader(legacy.Timings(), tmp_path / 'fast', memory_cache_mib=256)
    try:
        for day_items in synthetic.catalogs['grid_a']['s2'].values():
            for it in day_items:
                a = old.values(it, 'SCL', pixels.points)
                for _ in range(2):                       # seconda passata: blocchi dalla cache RAM
                    b = new.values(it, 'SCL', pixels.points)
                    assert np.array_equal(a, b, equal_nan=True)
        for it in synthetic.catalogs['grid_a']['asc']['2020-02-02']:
            for band in ('vv', 'vh'):
                assert np.array_equal(old.values(it, band, pixels.points), new.values(it, band, pixels.points), equal_nan=True)
    finally:
        old.close()
        new.close()


def test_acquire_batch_matches_legacy(synthetic, tmp_path):
    base = {'protocol': 'test', 'code': {}}
    legacy_results, jobs = {}, []
    old = legacy.reader(legacy.Timings(), tmp_path / 'legacy')
    try:
        for t in ['grid_a', 'grid_b']:
            pixels = legacy.Pixels(synthetic.reference / 'aoi_x' / t)
            legacy_results[t] = legacy.acquire_points(pixels, synthetic.catalogs[t], tmp_path / 'legacy' / t,
                                                      {**base, 'tile': t}, old, threads=1)
            jobs.append(TileJob(t, pixels, synthetic.catalogs[t], tmp_path / 'fast' / t / 'days', {**base, 'tile': t}))
    finally:
        old.close()
    new = build_reader(legacy.Timings(), tmp_path / 'fastreader', memory_cache_mib=256)
    handles = HandlePool(new)
    with ThreadPoolExecutor(4) as pool:
        fast_results = acquire_batch(jobs, new, handles, pool)
        # Seconda chiamata: tutti i giorni dalla cache su disco, stesso risultato.
        again = acquire_batch(jobs, new, handles, pool)
    handles.close_all()
    new.close()
    assert handles.stats['reuses'] > 0
    for t in ['grid_a', 'grid_b']:
        a, b, c = legacy_results[t], fast_results[t], again[t]
        assert set(a) == set(b)
        for key in a:
            assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), (t, key)
            assert np.array_equal(a[key], c[key])
        assert a['mask'].any() and (a['asc'] != 0).any()      # i dati sintetici esercitano davvero i rami


def fake_infer(model, raw, device, batch, amp=False):
    s2 = raw['s2'].astype(np.float32)
    mask = raw['mask'].astype(np.float32)
    s1 = np.concatenate([raw['asc'], raw['desc']], axis=1).astype(np.float32)
    feats = np.concatenate([(s2 * mask[..., None]).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1),
                            s1.sum(axis=1), mask.sum(axis=1, keepdims=True)], axis=1)
    reps = int(np.ceil(128 / feats.shape[1]))
    return np.sin(np.tile(feats, reps)[:, :128] * 1e-3).astype(np.float32)


def test_end_to_end_legacy_vs_fast(synthetic, tmp_path, monkeypatch):
    from biomap_v2.campaign import extraction
    checkpoint = tmp_path / 'fake.pt'
    checkpoint.write_bytes(b'fake checkpoint')
    from biomap_v2.common import digest
    monkeypatch.setattr(legacy, 'WEIGHTS_SHA256', digest(checkpoint))
    monkeypatch.setattr(fast_pipeline, 'WEIGHTS_SHA256', digest(checkpoint))
    monkeypatch.setattr(legacy, 'catalog', lambda tile, pixels, folder: (synthetic.catalogs[Path(tile).name],
                                                                          pixels.edge_bbox()))
    monkeypatch.setattr(legacy, 'load_model', lambda args: 'fake-model')
    monkeypatch.setattr(extraction, 'infer', fake_infer)
    common = dict(reference=str(synthetic.reference), checkpoint=str(checkpoint), device='cpu', chunk_size=97,
                  pixel_batch=16, keep_pixel_embeddings=False, amp=False, aoi=None, max_tiles=None)
    old_args = SimpleNamespace(work_root=str(tmp_path / 'r1'), io_threads=3, **common)
    legacy.stream(old_args, synthetic.spec, synthetic.frame)
    new_args = SimpleNamespace(work_root=str(tmp_path / 'r2'), io_workers=5, batch_pixels=10_000, prefetch=1,
                               memory_cache_mib=64, gdal_cache_mib=64, vsi_cache_mib=16, tiles=None, **common)
    status = fast_pipeline.stream_fast(new_args, synthetic.spec, synthetic.frame)
    assert status['tiles_missing'] == 0
    for t in ['grid_a', 'grid_b']:
        with np.load(tmp_path / 'r1' / 'chunks' / 'aoi_x' / t / 'tile.npz') as a, \
                np.load(tmp_path / 'r2' / 'chunks' / 'aoi_x' / t / 'tile.npz') as b:
            for key in ('cell', 'sums', 'counts'):
                assert np.array_equal(a[key], b[key]), (t, key)
    p1 = pd.read_parquet(tmp_path / 'r1' / 'v2.parquet')
    p2 = pd.read_parquet(tmp_path / 'r2' / 'v2.parquet')
    pd.testing.assert_frame_equal(p1, p2, check_exact=True)
    marker = json.loads((tmp_path / 'r2' / 'extraction_complete.json').read_text())
    assert marker['method'] == fast_pipeline.PROTOCOL and marker['complete_campaign'] is False
    # Ripresa: secondo lancio senza lavoro nuovo, stesso parquet.
    fast_pipeline.stream_fast(new_args, synthetic.spec, synthetic.frame)
    pd.testing.assert_frame_equal(p2, pd.read_parquet(tmp_path / 'r2' / 'v2.parquet'), check_exact=True)


@pytest.mark.skipif(os.name != 'nt', reason='blocco di condivisione specifico di Windows')
def test_replace_retries_while_reader_holds_file(tmp_path):
    import ctypes
    from ctypes import wintypes
    target = tmp_path / 'progress.json'
    save_json(target, {'v': 1})
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    # Lettore come Get-Content: condivisione Read|Write ma NON Delete -> os.replace fallisce con WinError 5.
    handle = kernel32.CreateFileW(str(target), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle != wintypes.HANDLE(-1).value
    timer = threading.Timer(0.4, lambda: kernel32.CloseHandle(handle))
    timer.start()
    started = time.perf_counter()
    save_json(target, {'v': 2})
    timer.join()
    assert time.perf_counter() - started >= 0.3
    assert json.loads(target.read_text()) == {'v': 2}
    assert not list(tmp_path.glob('.*.tmp'))


def test_failed_replace_keeps_previous_file(tmp_path, monkeypatch):
    target = tmp_path / 'tile.json'
    save_json(target, {'v': 1})

    def always_denied(src, dst):
        raise PermissionError(13, 'Accesso negato')
    monkeypatch.setattr(os, 'replace', always_denied)
    with pytest.raises(PermissionError):
        replace_with_retry(tmp_path / 'x', target, attempts=3, max_delay=0.01)
    import biomap_v2.dense_fast.io_safe as io_safe
    monkeypatch.setattr(io_safe, 'replace_with_retry', lambda tmp, path: always_denied(tmp, path))
    with pytest.raises(PermissionError):
        save_json(target, {'v': 2})
    monkeypatch.undo()
    assert json.loads(target.read_text()) == {'v': 1}
    assert not list(tmp_path.glob('.*.tmp'))


def test_make_batches_groups_neighbours_within_aoi():
    plan = [('a', 't1'), ('a', 't2'), ('a', 't3'), ('b', 't1'), ('b', 't2')]
    sizes = {('a', 't1'): 10, ('a', 't2'): 900, ('a', 't3'): 50, ('b', 't1'): 5, ('b', 't2'): 5}
    assert fast_pipeline.make_batches(plan, sizes, 100) == [[('a', 't1')], [('a', 't2')], [('a', 't3')],
                                                            [('b', 't1'), ('b', 't2')]]
    assert fast_pipeline.make_batches(plan, sizes, 1000) == [[('a', 't1'), ('a', 't2'), ('a', 't3')],
                                                             [('b', 't1'), ('b', 't2')]]


def test_end_to_end_keep_pixel_embeddings(synthetic, tmp_path, monkeypatch):
    # Percorso --keep-pixel-embeddings (usato dalla Spatial CNN V2): su Windows il memmap della tile deve essere
    # chiuso prima dello spostamento, altrimenti shutil.move copia il file, fallisce sull'unlink (WinError 32) e
    # il sidecar .json non viene mai scritto; tile.npz è già scritto, quindi la ripresa non lo rigenera.
    from biomap_v2.campaign import extraction
    from biomap_v2.common import digest
    checkpoint = tmp_path / 'fake.pt'
    checkpoint.write_bytes(b'fake checkpoint')
    monkeypatch.setattr(fast_pipeline, 'WEIGHTS_SHA256', digest(checkpoint))
    monkeypatch.setattr(legacy, 'catalog', lambda tile, pixels, folder: (synthetic.catalogs[Path(tile).name],
                                                                          pixels.edge_bbox()))
    monkeypatch.setattr(legacy, 'load_model', lambda args: 'fake-model')
    monkeypatch.setattr(extraction, 'infer', fake_infer)
    common = dict(reference=str(synthetic.reference), checkpoint=str(checkpoint), device='cpu', chunk_size=97,
                  pixel_batch=16, keep_pixel_embeddings=True, amp=False, aoi=None, max_tiles=None)
    args = SimpleNamespace(work_root=str(tmp_path / 'r2'), io_workers=5, batch_pixels=10_000, prefetch=1,
                           memory_cache_mib=64, gdal_cache_mib=64, vsi_cache_mib=16, tiles=None, **common)
    status = fast_pipeline.stream_fast(args, synthetic.spec, synthetic.frame)
    assert status['tiles_missing'] == 0
    work = tmp_path / 'r2'
    for t in ['grid_a', 'grid_b']:
        npy = work / 'pixel_embeddings' / 'aoi_x' / f'{t}.npy'
        meta = json.loads(npy.with_suffix('.json').read_text(encoding='utf-8'))
        assert meta['sha256'] == digest(npy)
        assert meta['geometry_sha256'] == digest(synthetic.reference / 'aoi_x' / t / 'geometry.npz')
        emb = np.load(npy)
        lookup = np.load(synthetic.reference / 'aoi_x' / 'lookup.npy')
        with np.load(synthetic.reference / 'aoi_x' / t / 'geometry.npz') as z:
            ids = lookup[z['cell_ids']]
        with np.load(work / 'chunks' / 'aoi_x' / t / 'tile.npz') as z:
            cell, sums = z['cell'], z['sums']
        assert emb.dtype == np.float32 and emb.shape == (len(ids), 128)
        valid = np.isfinite(emb).all(axis=1)
        recomputed = np.zeros((len(synthetic.frame), 128))
        np.add.at(recomputed, ids[valid], emb[valid].astype(np.float64))
        assert np.allclose(recomputed[cell], sums, rtol=1e-12, atol=0)
    assert not (work / 'tmp' / 'aoi_x').exists() or not any((work / 'tmp' / 'aoi_x').rglob('*.npy'))
    # Ripresa: nessun lavoro nuovo, file per pixel invariati.
    before = {p.name: digest(p) for p in (work / 'pixel_embeddings' / 'aoi_x').glob('*.npy')}
    fast_pipeline.stream_fast(args, synthetic.spec, synthetic.frame)
    assert before == {p.name: digest(p) for p in (work / 'pixel_embeddings' / 'aoi_x').glob('*.npy')}
