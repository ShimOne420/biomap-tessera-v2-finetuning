# SCOPO
#   Test offline (niente rete, niente GPU) del percorso PC-only della campagna densa:
#   inventario tile V1, reference da dati pubblici, protocollo di acquisizione storico, streaming con
#   checkpoint/ripresa/cancellazione sicura dei temporanei, CLI e incertezza appaiata V2-V1.
# COME SI RICHIAMA
#   .venv\Scripts\python.exe -m pytest tests\test_dense_public.py -q
# NOTA
#   Le verifiche su dati reali (hash delle tile V1, equivalenza al bit con il campione storico, smoke test
#   end-to-end) sono comandi separati documentati in docs/ml/DENSE_127909_PC_RUNBOOK.md.
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biomap_v2.campaign import streaming as st
from biomap_v2.campaign.state import MEANS
from biomap_v2.common import ROOT, save_json, digest, WEIGHTS_SHA256
from test_dense_portable import dense_raw  # noqa: F401  (fixture)


def load_download_script():
    spec = importlib.util.spec_from_file_location('download_v1_tiles', ROOT / 'scripts/download_v1_tiles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Quadrato chiuso: riproduce l'inventario storico (Liguria 12 candidate/10 esistenti, Abruzzo 16).
def test_v1_tile_inventory_rule():
    m = load_download_script()
    assert len(m.candidate_tiles((9.35, 44.15, 9.65, 44.35))) == 12
    assert len(m.candidate_tiles((13.45, 42.40, 13.75, 42.60))) == 16
    assert len(m.candidate_tiles((11.1, 46.2, 11.4, 46.4))) == 20
    assert (9.35, 44.25) in m.candidate_tiles((9.35, 44.15, 9.65, 44.35))
    assert m.DATASET_VERSION == '1.0' and m.DATASET_VARIANT == 'vultr'


# La geometria contiene solo i pixel V1 delle celle del manifest e ne riproduce n_valid.
def test_build_area_geometry_matches_v1(dense_raw, tmp_path):
    from biomap_v2.campaign.public_reference import build_area
    from biomap_v2.dense.grid import load_target
    grid = load_target(dense_raw['target'])
    cells = pd.DataFrame({'cell_id': [f'c{i}' for i in range(3)], 'lon': grid.lon[[0, 1, 3]], 'lat': grid.lat[[0, 1, 3]],
                          'aoi_id': 'synthetic', 'partition': 'train', 'cv_fold': 0}, index=[7, 8, 9])
    folder = tmp_path / 'ref' / 'synthetic'
    folder.mkdir(parents=True)
    record, info = build_area('synthetic', cells, dense_raw['v1_tiles'], dense_raw['target'], folder, log=lambda m: None)
    assert info['edge_op'] == 'trunc' and info['pixel_contributions'] == 12
    assert record.reference_index.tolist() == [7, 8, 9] and record.n_valid.tolist() == [4, 4, 4]
    with np.load(folder / 'grid_test' / 'geometry.npz') as z:
        assert len(z['flat_indices']) == 12 and np.all(np.diff(z['flat_indices']) > 0)
    lookup = np.load(folder / 'lookup.npy')
    assert lookup.tolist() == [7, 8, -1, 9]
    spec = json.loads((folder / 'grid_test' / 'geometry.json').read_text())
    assert spec['geometry_sha256'] == digest(folder / 'grid_test' / 'geometry.npz') and spec['pixels'] == 12


def test_s1_storage_historical_convention():
    value = np.array([0.05949767306447029, 0.0, -1.0, np.nan, 1e-9])
    out = st.s1_storage(value)
    assert out[0] == 5098  # caso reale: float64 darebbe 5097, lo storico e il Medium usano float32
    assert out[1:4].tolist() == [0, 0, 0] and out[4] == 0
    assert st.doy('2020-12-31') == 366 and st.doy('2020-01-01') == 1


class FakeSource:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def values(self, item, band, points):
        self.calls.append((item['id'], band, len(points)))
        return np.array([self.table[item['id']][band](p) for p in points], dtype=np.float64)

    def signed(self, href):
        return href

    def clear_tokens(self):
        pass

    def close(self):
        pass


class P:
    def __init__(self, n):
        self.points = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        self.rows, self.cols = np.zeros(n, dtype=int), np.arange(n)
        self.transform, self.crs = st.rasterio.Affine(10, 0, 0, 0, -10, 0), 'EPSG:32632'

    def __len__(self):
        return len(self.rows)


# S2: prima scena valida in ordine (nuvolosità, id); bande lette solo dove lo SCL è valido.
def test_compose_day_s2_order_and_scl(monkeypatch):
    pixels = P(3)
    table = {'A': {'SCL': lambda p: 8. if p[0] == 2 else 4.}, 'B': {'SCL': lambda p: 4.}}
    monkeypatch.setattr(st, 'reproject_band', lambda source, item, band, tf, shape, crs:
                        np.full(shape, 100. if item['id'] == 'A' else 200.5, dtype=np.float32))
    items = [{'id': 'B', 'properties': {'eo:cloud_cover': 10}}, {'id': 'A', 'properties': {'eo:cloud_cover': 5}}]
    bands, valid = st.compose_day('s2', items, pixels, FakeSource(table))
    assert valid.all() and bands[:2, 0].tolist() == [100, 100] and bands[2, 0] == 200  # troncamento uint16


# S1: ogni polarizzazione si riempie con la prima scena il cui valore convertito è diverso da zero.
def test_compose_day_s1_per_band_fill():
    pixels = P(2)
    table = {'x1': {'vv': lambda p: 0.0595, 'vh': lambda p: 1e-9},
             'x2': {'vv': lambda p: 0.0100, 'vh': lambda p: 0.0040}}
    items = [{'id': 'x2', 'properties': {}}, {'id': 'x1', 'properties': {}}]
    bands, valid = st.compose_day('asc', items, pixels, FakeSource(table))
    assert bands[0].tolist() == [int(st.s1_storage(np.array([0.0595]))[0]), int(st.s1_storage(np.array([0.004]))[0])]
    assert valid.all()


def make_reference(tmp_path, n=10):
    ref = tmp_path / 'ref'
    tile = ref / 'aoi' / 'grid_t'
    tile.mkdir(parents=True)
    np.savez(tile / 'geometry.npz', flat_indices=np.arange(n), cell_ids=np.arange(n) % 3)
    save_json(tile / 'geometry.json', {'pixels': n, 'height': 1, 'width': n, 'crs': 'EPSG:4326',
                                       'transform': [1e-4, 0, 9.0, 0, -1e-4, 44.0], 'bbox': [9, 43.9, 9.1, 44]})
    np.save(ref / 'aoi' / 'lookup.npy', np.array([0, 1, 2]))
    frame = pd.DataFrame({'cell_id': ['a', 'b', 'c'], 'aoi_id': 'aoi', 'partition': 'train', 'cv_fold': 0,
                          'lon': [9., 9.1, 9.2], 'lat': 44., 'target': [.1, .2, .3],
                          'n_valid': [4, 3, 3], 'n_total': [4, 3, 3], 'valid_fraction': 1.})
    return ref, frame, {'areas': [{'aoi_id': 'aoi', 'tiles': ['grid_t']}]}


# Checkpoint, crash simulato, ripresa senza ripetere i blocchi completati, pulizia dei temporanei.
def test_stream_resume_after_crash_and_cleanup(tmp_path, monkeypatch):
    ref, frame, spec = make_reference(tmp_path)
    work = tmp_path / 'work'
    ckpt = tmp_path / 'medium.pt'
    ckpt.write_bytes(b'x')
    monkeypatch.setattr(st, 'WEIGHTS_SHA256', digest(ckpt))
    items = {'s2': {'2020-01-01': [{'id': 'S', 'properties': {'eo:cloud_cover': 1}}]},
             'asc': {'2020-01-02': [{'id': 'A', 'properties': {}}]}, 'desc': {'2020-12-31': [{'id': 'D', 'properties': {}}]}}
    monkeypatch.setattr(st, 'catalog', lambda tile, pixels, folder: (folder.mkdir(parents=True, exist_ok=True) or items, [0, 0, 1, 1]))
    table = {'S': {'SCL': lambda p: 4.}, 'A': {'vv': lambda p: .05, 'vh': lambda p: .01}, 'D': {'vv': lambda p: .06, 'vh': lambda p: .02}}
    fake = FakeSource(table)
    monkeypatch.setattr(st, 'reader', lambda timings, work: fake)
    monkeypatch.setattr(st, 'reproject_band', lambda source, item, band, tf, shape, crs:
                        np.arange(shape[1], dtype=np.float32)[None, :].repeat(shape[0], 0) + 500)
    monkeypatch.setattr(st, 'load_model', lambda args: object())
    calls = {'n': 0, 'fail_at': 2}
    import biomap_v2.campaign.extraction as extraction

    def infer(model, raw, device, batch, amp):
        calls['n'] += 1
        if calls['n'] == calls['fail_at']:
            raise RuntimeError('crash simulato')
        assert raw['desc_doys'].tolist() == [366]
        return np.repeat(raw['s2'][:, 0, :1].astype(np.float32), 128, axis=1)
    monkeypatch.setattr(extraction, 'infer', infer)
    args = SimpleNamespace(work_root=str(work), reference=str(ref), checkpoint=str(ckpt), device='cpu', amp=False,
                           aoi=None, max_tiles=None, chunk_size=4, pixel_batch=2, io_threads=1, keep_pixel_embeddings=False)
    with pytest.raises(RuntimeError, match='crash'):
        st.stream(args, spec, frame)
    tmp = work / 'tmp' / 'aoi' / 'grid_t'
    assert (tmp / 'encoded' / '0-4.json').exists() and not (tmp / 'encoded' / '4-8.json').exists()
    assert not (work / 'chunks' / 'aoi' / 'grid_t' / 'tile.npz').exists()
    assert 'crash simulato' in (work / 'stream_errors.jsonl').read_text()
    # Payload senza marker (crash tra i due rename): non deve contare come completo.
    (tmp / 'encoded' / '4-8.npz').write_bytes(b'incompleto')
    calls['fail_at'] = -1
    reads_before = len(fake.calls)
    st.stream(args, spec, frame)
    # 1 riuscita + 1 interrotta dal crash + 2 blocchi mancanti = 4; se il primo blocco fosse rifatto sarebbero 5.
    assert calls['n'] == 4
    assert len(fake.calls) == reads_before  # giorni già acquisiti: nessuna nuova lettura
    assert not tmp.exists() and (work / 'api' / 'aoi' / 'grid_t').exists()
    out = pd.read_parquet(work / 'v2.parquet')
    expected = [np.mean([500 + i for i in range(10) if i % 3 == k]) for k in range(3)]
    assert np.allclose(out[MEANS[0]], expected) and np.allclose(out[MEANS[127]], expected)
    calls['n'] = 0
    st.stream(args, spec, frame)  # terza esecuzione: tutto dalla cache della tile
    assert calls['n'] == 0


def test_remove_temporary_refuses_outside(tmp_path):
    (tmp_path / 'work' / 'tmp' / 'x').mkdir(parents=True)
    (tmp_path / 'other').mkdir()
    with pytest.raises(ValueError):
        st.remove_temporary(tmp_path / 'other', tmp_path / 'work')
    st.remove_temporary(tmp_path / 'work' / 'tmp' / 'x', tmp_path / 'work')
    assert not (tmp_path / 'work' / 'tmp' / 'x').exists()


def test_cli_new_commands():
    from biomap_v2.campaign.cli import parser
    a = parser().parse_args(['stream', '--reference', 'R', '--work-root', 'W', '--io-threads', '4', '--max-tiles', '1'])
    assert a.command == 'stream' and a.io_threads == 4 and a.max_tiles == 1 and not a.keep_pixel_embeddings
    a = parser().parse_args(['build-reference', '--reference', 'R', '--work-root', 'W', '--msa-area', 'A', '--cellarea', 'C'])
    assert a.msa_area == 'A'
    assert parser().parse_args(['verify-legacy-sample', '--reference', 'R', '--work-root', 'W']).command == 'verify-legacy-sample'


# Incertezza: AOI ricampionate per lo sviluppo, blocchi 2 km per la Liguria; segno V2-V1.
def test_paired_uncertainty(tmp_path):
    from biomap_v2.campaign.uncertainty import paired_uncertainty
    rng = np.random.default_rng(0)
    rows = []
    for e, noise in [('v1', .2), ('v2', .1)]:
        for i in range(60):
            y = .5 + .3 * np.sin(i)
            rows.append({'cell_id': str(i), 'aoi_id': f'a{i % 6}', 'cv_fold': i % 3, 'lon': 9 + i * .01, 'lat': 44.,
                         'target': y, 'prediction': y + rng.normal(0, noise), 'model': 'Ridge', 'partition': 'validation', 'encoder': e})
    for e in ['v1', 'v2']:
        d = tmp_path / 'probing' / e
        d.mkdir(parents=True)
        f = pd.DataFrame([r for r in rows if r['encoder'] == e])
        f.to_csv(d / 'oof_predictions.csv', index=False)
        f.assign(partition='liguria_benchmark', aoi_id='liguria_levante').to_csv(d / 'liguria_predictions.csv', index=False)
    result = paired_uncertainty(tmp_path, n_boot=200)
    oof = result[(result.scope == 'development_macro_17aoi')].iloc[0]
    assert oof.rmse_delta < 0 and oof.rmse_ci_high < 0 and oof.n_clusters == 6
    assert set(result.scope) == {'development_macro_17aoi', 'development_micro', 'liguria_cells', 'liguria_blocks_2km'}
