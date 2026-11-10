# Fixture CPU: pacchetto denso, supporto fisico, CLI, regressori, report e download verificato.
# Nessun accesso alle serie reali, GPU o rete; le feature sintetiche non sono risultati MSA.
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from biomap_v2.common import digest, save_json, WEIGHTS_SHA256
from biomap_v2.dense import bundle, sample
from biomap_v2.dense.cli import main
from biomap_v2.dense.grid import load_target
from biomap_v2.dense.v1_support import aggregate_v1
from biomap_v2.dense.v2_support import PROTOCOL, aggregate_v2
from biomap_v2.dense.support import verify_support
from biomap_v2.dense.paired import pair_features
from biomap_v2.dense.evaluation import make_estimator


@pytest.fixture
def dense_raw(tmp_path):
    transform = from_origin(9.35, 44.35, 1/360, 1/360)
    tile_transform = from_origin(9.35, 44.35, 1/720, 1/720)
    target = tmp_path/'target.tif'
    def raster(path, arr, tf):
        with rasterio.open(path, 'w', driver='GTiff', width=arr.shape[1], height=arr.shape[0], count=1,
                           dtype=str(arr.dtype), transform=tf, crs='EPSG:4326', nodata=-9999) as ds:
            ds.write(arr, 1)
    raster(target, np.array([[.2, .4], [.6, .8]], dtype='float32'), transform)
    v1root, v2root = tmp_path/'v1', tmp_path/'v2'
    directory = v1root/'global_0.1_degree_representation/2020/grid_test'
    directory.mkdir(parents=True)
    land = v1root/'global_0.1_degree_tiff_all/grid_test.tiff'
    land.parent.mkdir()
    raster(land, np.ones((4, 4), dtype='float32'), tile_transform)
    emb = np.arange(4*4*128).reshape(4, 4, 128).astype('int8')
    np.save(directory/'grid_test.npy', emb)
    scales = directory/'grid_test_scales.npy'
    np.save(scales, np.full((4, 4), .05, dtype='float32'))
    v1 = aggregate_v1(v1root, load_target(target))
    v1parquet = tmp_path/'v1.parquet'
    v1.to_parquet(v1parquet)
    folder = v2root/'grid_test'
    folder.mkdir(parents=True)
    cells = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3], dtype='int32')
    np.savez(folder/'geometry.npz', flat_indices=np.arange(16), cell_ids=cells)
    np.save(folder/'embeddings_fp32.npy', emb.reshape(16, 128).astype('float32')*.07)
    scenes = {k: [{'date': f'2020-01-{i+1:02d}'} for i in range(8)] for k in ['s2', 'asc', 'desc']}
    save_json(folder/'inputs_manifest.json', {'scene_groups': scenes})
    for name, arr in [('s2', np.full((8, 16, 10), 3000, dtype='uint16')),
                      ('mask', np.ones((8, 16), dtype=bool)),
                      ('asc', np.full((8, 16, 2), 4000, dtype='int16')),
                      ('desc', np.full((8, 16, 2), 5000, dtype='int16'))]:
        np.save(folder/f'{name}.npy', arr)
    save_json(folder/'geometry.json', {'pixels': 16, 'height': 4, 'width': 4, 'crs': 'EPSG:4326',
              'transform': list(tile_transform)[:6], 'geometry_sha256': digest(folder/'geometry.npz'),
              'v1_scales_sha256': digest(scales), 'landmask_sha256': digest(land)})
    save_json(folder/'encoder_progress.json', {'pixels_complete': 16, 'pixels_total': 16, 'protocol': PROTOCOL,
              'checkpoint_sha256': WEIGHTS_SHA256, 'inputs_sha256': digest(folder/'inputs_manifest.json')})
    v2parquet = tmp_path/'v2.parquet'
    aggregate_v2(v2root, n_total=v1.n_total).rename(columns={'n_valid_v2': 'n_valid'}).to_parquet(v2parquet)
    return {'v1_tiles': str(v1root), 'v2_tiles': str(v2root), 'target': str(target),
            'v1_parquet': str(v1parquet), 'v2_parquet': str(v2parquet), 'aoi_id': 'synthetic_fixture'}


def test_pack_round_trip_and_checksums(dense_raw, tmp_path):
    directory = tmp_path/'bundle'
    manifest = bundle.pack(dense_raw, directory, pixels=4, folds=2)
    frame, loaded = bundle.load_bundle(directory)
    assert manifest == loaded and len(frame) == 4 and frame.n_valid.sum() == 16
    assert set(frame.predefined_fold) == {0, 1}
    assert (directory.with_suffix('.zip')).is_file()
    assert bundle.verify_bundle(directory)['encoder']['status'].startswith('not_recomputed')
    with pytest.raises(FileExistsError, match='non vuota'):
        bundle.pack(dense_raw, directory)
    (directory/'paired_features.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrotto'):
        bundle.load_bundle(directory)


def test_physical_support_rejects_equal_counts_wrong_pixels(dense_raw):
    grid = load_target(dense_raw['target'])
    reference = pd.read_parquet(dense_raw['v1_parquet'])
    folder = Path(dense_raw['v2_tiles'])/'grid_test'
    with np.load(folder/'geometry.npz') as z:
        cells, flat = z['cell_ids'].copy(), z['flat_indices'].copy()
    cells[0], cells[2] = cells[2], cells[0]  # Conteggi per cella identici, appartenenza errata.
    np.savez(folder/'geometry.npz', cell_ids=cells, flat_indices=flat)
    meta = json.loads((folder/'geometry.json').read_text())
    meta['geometry_sha256'] = digest(folder/'geometry.npz')
    save_json(folder/'geometry.json', meta)
    with pytest.raises(ValueError, match='celle GLOBIO diverse'):
        verify_support(dense_raw['v1_tiles'], dense_raw['v2_tiles'], grid, reference)


def test_pair_rejects_missing_cells_and_different_targets():
    data = {f'mean_{i:03d}': [float(i), float(i+1)] for i in range(128)}
    first = pd.DataFrame({**data, 'lon': [9., 9.1], 'lat': [44., 44.1], 'msa_overall': [.3, .6],
                          'n_valid': [50, 50], 'n_total': [50, 50], 'valid_fraction': [1., 1.]})
    second = first.copy()
    with pytest.raises(ValueError, match='Copertura'):
        pair_features(first, second.iloc[:1])
    second['msa_overall'] = [.9, .6]
    with pytest.raises(ValueError, match='target'):
        pair_features(first, second)


def test_models_keep_user_configuration():
    xgb = make_estimator('XGBoost', {}).get_params()
    assert xgb['n_estimators'] == 400 and xgb['max_depth'] == 2 and xgb['learning_rate'] == .05
    mlp = make_estimator('MLP', {}).steps[-1][1].get_params()
    assert mlp['hidden_layer_sizes'] == (256, 128) and mlp['early_stopping'] is True


def test_historical_model_is_a_distinct_optional_reference(dense_raw, tmp_path):
    import joblib
    from sklearn.dummy import DummyRegressor
    from biomap_v2.dense.historical import evaluate_reference
    first = pd.read_parquet(dense_raw['v1_parquet'])
    second = pd.read_parquet(dense_raw['v2_parquet'])
    frame = pair_features(first, second)
    model = DummyRegressor().fit(np.zeros((3, 128)), [.25, .25, .25])
    path = tmp_path/'historical.joblib'
    joblib.dump(model, path)
    result = evaluate_reference(frame, path)
    assert result['metrics']['n'] == 4 and result['role'].startswith('historical_development')
    assert evaluate_reference(frame) is None


def test_cli_complete_bundle_no_overwrite_and_export(dense_raw, tmp_path):
    directory = tmp_path/'bundle'
    bundle.pack(dense_raw, directory, pixels=4, folds=2)
    hist = tmp_path/'hist.json'
    save_json(hist, {'max_iter': 2, 'min_samples_leaf': 2, 'early_stopping': False})
    xgb, mlp = tmp_path/'xgb.json', tmp_path/'mlp.json'
    save_json(xgb, {'n_estimators': 2})
    save_json(mlp, {'hidden_layer_sizes': [4], 'early_stopping': False, 'max_iter': 4})
    out = tmp_path/'runs/fixture_run'
    args = ['all', '--bundle', str(directory), '--skip-encoder-check', '--folds', '2', '--bootstrap', '5',
            '--histgb-params', str(hist), '--xgb-params', str(xgb), '--mlp-params', str(mlp), '--out', str(out)]
    assert main(args) == 0
    metrics = pd.read_csv(out/'metrics.csv')
    assert set(metrics.model) == {'DummyMean', 'Ridge', 'XGBoost', 'HistGB', 'MLP'} and len(metrics) == 10
    assert len(pd.read_csv(out/'predictions.csv')) == 40
    assert len(list((out/'models').glob('*.joblib'))) == 20
    text = (out/'REPORT_V1_V2_DENSE_MSA.md').read_text()
    assert '4,991' not in text and '2,4e-6' not in text and 'escludono lo zero' not in text
    assert 'XGBoost' in text and 'non eseguita' in text
    from openpyxl import load_workbook
    book = load_workbook(out/'risultati.xlsx', read_only=True)
    assert book['Celle'].max_row == 41
    book.close()
    checksum = digest(out/'predictions.csv')
    with pytest.raises(FileExistsError, match='già un run'):
        main(args)
    assert main(['export', '--out', str(out)]) == 0
    assert digest(out/'predictions.csv') == checksum


def test_gzip_sampling_does_not_write_to_input(dense_raw, tmp_path):
    folder = Path(dense_raw['v2_tiles'])/'grid_test'
    raw = folder/'s2.npy'
    content = raw.read_bytes()
    with gzip.open(folder/'s2.npy.gz', 'wb') as stream:
        stream.write(content)
    raw.unlink()
    result = sample.sample_array(folder, 's2', [0, 15])
    assert result.shape == (8, 2, 10) and not raw.exists()


def test_download_verifies_hash_before_extraction(dense_raw, tmp_path, monkeypatch):
    from biomap_v2.dense.download import fetch_bundle
    directory = tmp_path/'bundle'
    bundle.pack(dense_raw, directory, pixels=4, folds=2)
    archive = directory.with_suffix('.zip')
    spec = tmp_path/'release.json'
    save_json(spec, {'url': 'https://example.invalid/test.zip', 'zip_sha256': digest(archive),
              'zip_bytes': archive.stat().st_size, 'manifest_sha256': digest(directory/'bundle.json'), 'n_cells': 4})
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield archive.read_bytes()
    monkeypatch.setattr('biomap_v2.dense.download.requests.get', lambda *a, **k: Response())
    destination = tmp_path/'downloaded'
    assert fetch_bundle(spec, destination) == destination
    assert fetch_bundle(spec, destination) == destination
    assert (destination/'bundle.json').is_file()
    info = json.loads(spec.read_text())
    info['zip_sha256'] = '0'*64
    save_json(spec, info)
    with pytest.raises(ValueError, match='SHA256'):
        fetch_bundle(spec, tmp_path/'bad_download')
    assert not (tmp_path/'bad_download').exists()
