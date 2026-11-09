# Test end-to-end E2 con encoder ridotto, target sintetico e pochi punti reali nella griglia.
# La validazione del manifest storico completo è già nei test legacy; qui viene sostituita
# solo nelle fixture per evitare download, 132.900 file e training scientifici in CI.
import argparse
import builtins
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
from biomap_v2.common import config, digest, save_json
from biomap_v2.data import BANDS, PROTOCOL, sample_points
from biomap_v2.student import PixelStudent
from biomap_ml import source, cache, engine, campaign
from biomap_ml.settings import configuration
from biomap_ml.products import export_rasters, workbook

torch.set_num_threads(1)


@pytest.fixture
def prepared_e2(tmp_path, monkeypatch):
    root = tmp_path/'production'
    src = root/'msa_v1_liguria'
    src.mkdir(parents=True)
    args = argparse.Namespace(config=None, data_root=str(root), set=[])
    legacy = config('msa', args)
    legacy['dataset_namespace'] = 'msa_v1_liguria'
    legacy['sampling']['pixels_per_cell'] = 25
    legacy['target'].update(mode='custom', units='unitless')
    rng = np.random.default_rng(3)
    rows = []
    for i in range(15):
        part = 'train' if i < 12 else 'test'
        fold = i//4 if i < 12 else -1
        west, north, res = 8+i/360, 44., 1/360
        row = {'cell_id': str(i), 'lon': west+res/2, 'lat': north-res/2, 'west': west, 'east': west+res,
               'south': north-res, 'north': north, 'partition': part, 'cv_fold': fold,
               'aoi_id': f'dev_{fold}' if part == 'train' else 'liguria_levante',
               'target': .15+i*.04, 'valid_fraction': 1.}
        c = SimpleNamespace(**row)
        points = sample_points(c, 25)
        raw = {'points': points, 's2': rng.integers(100, 8000, (25, 8, 10)).astype(np.uint16),
               'mask': np.ones((25, 8), dtype=bool), 's2_doys': np.arange(8, dtype=np.int16)*20+1,
               'asc': rng.integers(3000, 8000, (25, 2, 2)).astype(np.int16), 'asc_doys': np.array([20, 200], dtype=np.int16),
               'desc': rng.integers(3000, 8000, (25, 2, 2)).astype(np.int16), 'desc_doys': np.array([30, 210], dtype=np.int16)}
        directory = src/'series'/str(i)
        directory.mkdir(parents=True)
        np.savez_compressed(directory/'series.npz', **raw)
        row['series_sha256'] = digest(directory/'series.npz')
        save_json(directory/'manifest.json', {'sha256': row['series_sha256'], 'protocol': PROTOCOL, 'year': 2020,
                  'sampling': legacy['sampling'], 'bounds': [row[k] for k in ['west', 'south', 'east', 'north']],
                  'valid_fraction': 1., 'band_order': BANDS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    geometry = tmp_path/'geometry.csv'
    frame.drop(columns=['target', 'valid_fraction', 'series_sha256']).to_csv(geometry, index=False)
    legacy['cells'] = str(geometry)
    target = src/'targets/normalized.tif'
    target.parent.mkdir()
    with rasterio.open(target, 'w', driver='GTiff', height=1, width=15, count=1, dtype='float32',
                       crs='EPSG:4326', transform=from_origin(8, 44, 1/360, 1/360), nodata=-9999) as ds:
        ds.write(frame.target.to_numpy(dtype=np.float32)[None], 1)
    legacy['target']['raster'] = str(target)
    frame.to_csv(src/'cells.csv', index=False)
    save_json(src/'cells.json', {'protocol': PROTOCOL, 'year': 2020, 'config_sampling': legacy['sampling'],
              'source_cells_sha256': digest(geometry), 'eligible_sha256': digest(src/'cells.csv'),
              'target_source': {'sha256': digest(target)}})
    model = PixelStudent(latent_dim=8, num_layers=1, nhead=4, dim_feedforward=32, dropout=0)
    checkpoint = root/'fixture.pt'
    torch.save({'model': model.state_dict(), 'args': {'repr_dim': 128, 'latent_dim': 8, 'num_layers': 1,
               'nhead': 4, 'dim_feedforward': 32, 'enable_qk_norm': False}}, checkpoint)
    cfg, _ = configuration(argparse.Namespace(config=str(Path(__file__).resolve().parents[1]/'configs/e2_msa.yaml'),
                          source_data_root=str(src), work_root=str(tmp_path/'work'), checkpoint=str(checkpoint), device='cpu'))
    cfg['runtime'].update(threads=1, pixel_batch=8, mixed_precision=False)
    cfg['training'].update(warmup_epochs=1, epochs=2, patience=8, effective_batch=4, microbatch=2)
    cfg['storage'].update(reserve_gib=0, max_work_gib=1)
    monkeypatch.setattr(source, 'WEIGHTS_SHA256', digest(checkpoint))
    monkeypatch.setattr(source, 'validate_protocol', lambda cfg, f: None)
    return cfg, legacy, frame


def test_audit_cache_train_evaluate_export_and_idempotence(prepared_e2, monkeypatch):
    cfg, legacy, _ = prepared_e2
    src = Path(cfg['source'])
    before = {str(p.relative_to(src)): digest(p) for p in src.rglob('*') if p.is_file()}
    original_open = io.open
    def read_only_guard(file, mode='r', *args, **kwargs):
        if not isinstance(file, int) and any(char in mode for char in 'wax+'):
            resolved = Path(file).resolve()
            if src == resolved or src in resolved.parents:
                raise AssertionError('Tentativo di scrittura nella sorgente del probing')
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(io, 'open', read_only_guard)
    monkeypatch.setattr(builtins, 'open', read_only_guard)
    frame, audit = source.build_index(cfg, legacy)
    point_cache = cache.build_cache(cfg, legacy, frame, audit)
    encoder_hash = digest(cfg['checkpoint'])
    # Un resume cache completo non deve neppure caricare l'encoder.
    monkeypatch.setattr(cache, 'load_encoder', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Unexpected encoding')))
    assert cache.build_cache(cfg, legacy, frame, audit) == point_cache
    assert len(json.loads(point_cache.with_suffix('.json').read_text())['invocations']) == 2
    out = campaign.train_campaign(cfg, 'fixture_E2')
    final = json.loads((out/'final.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    assert final['n_training'] == 12 and selection['liguria_used_for_selection'] is False
    assert len(pd.read_csv(out/'oof_predictions.csv')) == 12
    campaign.evaluate_campaign(cfg, 'fixture_E2')
    test = pd.read_csv(out/'test_predictions.csv')
    assert len(test) == 3 and set(test.cell_id.astype(str)) == {'12', '13', '14'}
    np.testing.assert_allclose(test.dummy_prediction, frame[frame.partition == 'train'].target.mean(), atol=1e-15, rtol=0)
    saved_test_hash = digest(out/'test_predictions.csv')
    # Ripetere training concluso e evaluate non perde il marker test e non reinferisce.
    campaign.train_campaign(cfg, 'fixture_E2', resume=True)
    monkeypatch.setattr(campaign, 'predict_checkpoint', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Second test inference')))
    campaign.evaluate_campaign(cfg, 'fixture_E2')
    assert digest(out/'test_predictions.csv') == saved_test_hash
    from openpyxl import load_workbook
    excel = load_workbook(Path(cfg['work'])/'E2_experiment_registry.xlsx', read_only=True, data_only=True)
    assert excel.sheetnames == ['Summary', 'Risultati', 'R001_OOF', 'R001_Test']
    metric_frame = pd.read_csv(out/'metrics.csv')
    assert excel['Risultati'].max_row == len(metric_frame)+1
    excel.close()
    geo = export_rasters(cfg, out, test)
    with rasterio.open(geo/'liguria_levante_predicted_E2.tif') as ds:
        np.testing.assert_allclose(ds.read(1)[0], test.prediction, rtol=0, atol=1e-7)
        assert ds.crs.to_epsg() == 4326 and ds.nodata == -9999
    assert digest(cfg['checkpoint']) == encoder_hash
    assert {str(p.relative_to(src)): digest(p) for p in src.rglob('*') if p.is_file()} == before


def test_resume_and_oom_preserve_training(prepared_e2, tmp_path, monkeypatch):
    cfg, legacy, _ = prepared_e2
    frame, audit = source.build_index(cfg, legacy)
    cache.build_cache(cfg, legacy, frame, audit)
    train = frame[(frame.partition == 'train') & (frame.cv_fold != 0)]
    val = frame[(frame.partition == 'train') & (frame.cv_fold == 0)]
    full = engine.fit(cfg, train, val, audit, tmp_path/'full', 'E2', 2)
    expected = torch.load(full, weights_only=False)['head']
    original = engine.predict
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('interrupted validation')
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, 'predict', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        engine.fit(cfg, train, val, audit, tmp_path/'broken', 'E2', 2)
    monkeypatch.setattr(engine, 'predict', original)
    resumed = engine.fit(cfg, train, val, audit, tmp_path/'broken', 'E2', 2, resume=True)
    for key, value in torch.load(resumed, weights_only=False)['head'].items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    original_batch = engine.batch
    calls = 0
    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise torch.cuda.OutOfMemoryError('synthetic')
        return original_batch(*args, **kwargs)
    monkeypatch.setattr(engine, 'batch', fail_once)
    oom = engine.fit(cfg, train, val, audit, tmp_path/'oom', 'E2', 2)
    for key, value in torch.load(oom, weights_only=False)['head'].items():
        torch.testing.assert_close(value, expected[key], atol=3e-6, rtol=3e-5)
    assert torch.load(tmp_path/'oom/last.pt', weights_only=False)['microbatch'] == 1
    changed = copy.deepcopy(cfg)
    changed['training']['learning_rate'] = .2
    with pytest.raises(ValueError, match='Resume incompatibile'):
        engine.fit(changed, train, val, audit, tmp_path/'broken', 'E2', 2, resume=True)


def test_corrupted_series_and_open_excel_fail_explicitly(prepared_e2):
    cfg, legacy, frame = prepared_e2
    source.build_index(cfg, legacy)
    path = source.series_path(cfg, frame.cell_id.iloc[0])
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Serie modificata'):
        source.build_index(cfg, legacy)
    marker = Path(cfg['work'])/'~$E2_experiment_registry.xlsx'
    marker.touch()
    with pytest.raises(PermissionError, match='Excel aperto'):
        workbook(cfg)
