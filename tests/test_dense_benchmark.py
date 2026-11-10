# Test CPU del benchmark denso V1/V2: assegnazione bordi, aggregazione, appaiamento, metriche.
# Le fixture sono sintetiche e non usano dati reali, GPU o checkpoint.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from biomap_v2.dense.grid import assign, load_target
from biomap_v2.dense.v1_support import aggregate_v1, compare_v1
from biomap_v2.dense.v2_support import aggregate_v2, compare_v2
from biomap_v2.dense.paired import pair_features, features
from biomap_v2.dense.evaluation import (bootstrap_delta, full_metrics, out_of_fold,
                                       spatial_blocks, spatial_folds)

DIMS = 128


def write_raster(path, array, transform, nodata=-999.0):
    with rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype, crs='EPSG:4326', transform=transform,
                       nodata=nodata) as dataset:
        dataset.write(array, 1)


def synthetic_target(tmp_path):
    transform = from_origin(9.35, 44.35, 0.01, 0.01)
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    path = tmp_path / 'target.tif'
    write_raster(path, values, transform)
    return path, transform, values


def synthetic_v1_tiles(tmp_path, transform, values):
    root = tmp_path / 'v1'
    directory = root / 'global_0.1_degree_representation' / '2020' / 'grid_test'
    directory.mkdir(parents=True)
    masks = root / 'global_0.1_degree_tiff_all'
    masks.mkdir(parents=True)
    landmask_transform = from_origin(9.35, 44.35, 0.005, 0.005)
    write_raster(masks / 'grid_test.tiff', np.ones((4, 4), dtype=np.float32), landmask_transform)
    embedding = np.zeros((4, 4, DIMS), dtype=np.int8)
    values_by_cell = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}
    for row in range(4):
        for col in range(4):
            cell = (row // 2, col // 2)
            embedding[row, col, :] = int(values_by_cell[cell] / 0.5)
    scales = np.full((4, 4), 0.5, dtype=np.float32)
    scales[0, 0] = np.nan
    np.save(directory / 'grid_test.npy', embedding)
    np.save(directory / 'grid_test_scales.npy', scales)
    return root


def test_edge_op_keeps_historical_truncation():
    transform = from_origin(9.35, 44.35, 1 / 360, 1 / 360)
    lon = np.array([9.3499999, 9.3500001, 9.6486111, 9.6513889])
    lat = np.array([44.3500001, 44.3499999, 44.1513889, 44.1486111])
    trunc_rows, trunc_cols = assign(transform, lon, lat, 'trunc')
    floor_rows, floor_cols = assign(transform, lon, lat, 'floor')
    assert trunc_rows[0] == 0 and floor_rows[0] == -1
    assert trunc_cols[0] == 0 and floor_cols[0] == -1
    assert np.array_equal(trunc_rows[1:], floor_rows[1:])
    assert np.array_equal(trunc_cols[1:], floor_cols[1:])


def test_v1_aggregation_and_comparison(tmp_path):
    target_path, transform, values = synthetic_target(tmp_path)
    tiles = synthetic_v1_tiles(tmp_path, transform, values)
    target = load_target(target_path)
    frame = aggregate_v1(tiles, target, edge_op='trunc', min_valid_fraction=0.8)
    # La copertura 0,8 esclude la cella con un pixel a scala non finita.
    assert len(frame) == 3
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(frame[[f'mean_{i:03d}' for i in range(DIMS)]].to_numpy(), expected[:, None])
    assert np.allclose(frame[[f'std_{i:03d}' for i in range(DIMS)]].to_numpy(), 0.0)
    assert frame.n_valid.tolist() == [4, 4, 4]
    assert frame.n_total.tolist() == [4, 4, 4]
    assert np.allclose(frame.valid_fraction, [1.0, 1.0, 1.0])
    assert list(frame.index) == [1, 2, 3]
    reference = frame.copy()
    report = compare_v1(frame, reference)
    assert report['n_common'] == 3
    assert report['max_abs_mean'] == 0.0
    assert report['n_valid_mismatch'] == 0


def test_v1_comparison_detects_support_mismatch(tmp_path):
    target_path, transform, values = synthetic_target(tmp_path)
    tiles = synthetic_v1_tiles(tmp_path, transform, values)
    frame = aggregate_v1(tiles, load_target(target_path), edge_op='trunc')
    reference = frame.copy()
    reference.iloc[0, reference.columns.get_loc('n_valid')] += 1
    report = compare_v1(frame, reference)
    assert report['n_valid_mismatch'] == 1


def synthetic_v2(tmp_path, include_std=False):
    root = tmp_path / 'v2'
    directory = root / 'grid_test'
    directory.mkdir(parents=True)
    cells = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    np.savez(directory / 'geometry.npz', flat_indices=np.arange(len(cells)), cell_ids=cells)
    embedding = np.zeros((len(cells), DIMS), dtype=np.float32)
    embedding[:4] = 1.0
    embedding[4:8] = 2.0
    embedding[8:12] = 3.0
    embedding[12:] = 4.0
    np.save(directory / 'embeddings_fp32.npy', embedding)
    shape = embedding.shape
    (directory / 'encoder_progress.json').write_text(
        '{"pixels_complete": %d, "pixels_total": %d}' % (shape[0], shape[0]))
    (directory / 'geometry.json').write_text('{"pixels": %d}' % shape[0])
    return root


def test_v2_aggregation_and_comparison(tmp_path):
    root = synthetic_v2(tmp_path)
    frame = aggregate_v2(root, n_total=pd.Series({0: 4, 1: 4, 2: 4, 3: 4}),
                         min_valid_fraction=0.8, include_std=True)
    names = [f'mean_{i:03d}' for i in range(DIMS)]
    assert np.allclose(frame[names].to_numpy(), np.array([1.0, 2.0, 3.0, 4.0])[:, None])
    stds = [f'std_{i:03d}' for i in range(DIMS)]
    assert np.allclose(frame[stds].to_numpy(), 0.0)
    assert frame.n_valid_v2.tolist() == [4, 4, 4, 4]
    assert compare_v2(frame, frame.rename(columns={'n_valid_v2': 'n_valid'}))['max_abs_mean'] == 0.0


def paired_frames():
    index = [0, 1]
    v1 = pd.DataFrame({'lon': [9.351, 9.352], 'lat': [44.349, 44.348],
                       'msa_overall': [0.2, 0.8], 'valid_fraction': [1.0, 1.0],
                       'n_valid': [4, 4], 'n_total': [4, 4]}, index=index)
    v2 = pd.DataFrame({'n_valid_v2': [4, 4]}, index=index)
    for band in range(DIMS):
        v1[f'mean_{band:03d}'] = [0.1, 0.2]
        v1[f'std_{band:03d}'] = [0.0, 0.0]
        v2[f'mean_{band:03d}'] = [0.3, 0.4]
        v2[f'std_{band:03d}'] = [0.0, 0.0]
    return v1, v2


def test_pair_features_and_feature_matrix():
    v1, v2 = paired_frames()
    paired = pair_features(v1, v2)
    assert len(paired) == 2
    assert features(paired, 'v1').shape == (2, DIMS)
    assert features(paired, 'v2').shape == (2, DIMS)
    assert features(paired, 'v2')[0, 0] == pytest.approx(0.3)
    broken = v2.copy()
    broken['n_valid_v2'] = [4, 3]
    with pytest.raises(ValueError, match='supporto pixel'):
        pair_features(v1, broken)


def test_spatial_folds_are_disjoint_and_complete():
    rng = np.random.default_rng(7)
    lon = 9.35 + rng.random(100) * 0.3
    lat = 44.15 + rng.random(100) * 0.2
    folds = spatial_folds(lon, lat, n_splits=5)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert len(folds) == 100
    assert len(spatial_blocks(lon, lat)) == 100


def test_bootstrap_zero_delta_for_identical_predictions():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    blocks = np.array(['a', 'a', 'b', 'b'])
    result = bootstrap_delta(y, y + 0.01, y + 0.01, blocks, n_boot=50, seed=3)
    assert result['mae_delta'] == 0.0
    assert result['rmse_delta'] == 0.0
    assert result['mae_blocks_ci_low'] == 0.0
    assert result['mae_blocks_ci_high'] == 0.0
    assert result['mae_blocks_n_boot'] == 50


def test_full_metrics_handles_zero_targets():
    y = np.array([0.0, 0.5, 1.0])
    prediction = np.array([0.1, 0.5, 0.9])
    result = full_metrics(y, prediction)
    assert result['n'] == 3
    assert result['mape_n'] == 2
    assert result['mae'] == pytest.approx(np.mean([0.1, 0.0, 0.1]))
    assert result['negative_prediction_fraction'] == 0.0


def test_out_of_fold_uses_all_cells():
    v1, v2 = paired_frames()
    paired = pair_features(v1, v2)
    folds = np.array([0, 1])
    prediction = out_of_fold(paired, 'v1', 'DummyMean', folds,
                             {'max_iter': 5, 'max_leaf_nodes': 3}, 100.0, 1)
    assert np.isfinite(prediction).all()
    assert len(prediction) == 2
