# SCOPO
#   Garantire che l'esperimento V2 spatial cambi SOLO la sorgente dei valori dei pixel rispetto all'Esp. B
#   e che le ottimizzazioni di esecuzione (worker, job paralleli) non cambino i risultati.
# COME VIENE RICHIAMATO
#   pytest experiments/v2_spatial_cnn_msa/tests
# LOGICA IN ORDINE
#   identità dei file condivisi con B -> oracolo B in sottoprocesso con valori V2 = V1 dequantizzati ->
#   stesse patch/maschere/medie/predizioni -> worker 0 vs 2 -> job separati vs run unico -> rifiuti espliciti.
# INPUT E OUTPUT
#   Fixture sintetiche di conftest.py; nessun dato reale.
# PARAMETRI
#   Nessuno.
# RISORSE
#   CPU; il test dell'oracolo dura circa un minuto.
# RIPRESA
#   Non applicabile.
import json
import os
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import EXPERIMENT, B_EXPERIMENT, write_v2
from runtime import Progress


def same_text(a, b):
    return Path(a).read_bytes().replace(b'\r\n', b'\n') == Path(b).read_bytes().replace(b'\r\n', b'\n')


def test_model_and_sources_are_byte_identical_to_experiment_b():
    for name in ['model.py', 'sources.py']:
        assert same_text(EXPERIMENT / name, B_EXPERIMENT / name), name


def test_metric_functions_identical_to_experiment_b():
    import evaluate
    import importlib.util
    spec = importlib.util.spec_from_file_location('b_evaluate_src', B_EXPERIMENT / 'evaluate.py')
    text_b = (B_EXPERIMENT / 'evaluate.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    text_v2 = (EXPERIMENT / 'evaluate.py').read_text(encoding='utf-8')
    for fn in ['def metrics(', 'def macro_rmse(']:
        start_b = text_b.index(fn); start_v2 = text_v2.index(fn)
        end_b = text_b.index('\n\n\n', start_b); end_v2 = text_v2.index('\n\n\n', start_v2)
        assert text_b[start_b:end_b] == text_v2[start_v2:end_v2]
    assert spec is not None and callable(evaluate.metrics)


def run_v2(cfg, frame, run_name='oracle', phase='all', jobs=None):
    from patch_builder import build
    from train import train, predict_external, load_run
    p = Progress(cfg, 'test')
    try:
        path = build(cfg, frame, p)
        run = Path(cfg['work_root']) / 'runs' / run_name
        train(cfg, run, phase, False, p, jobs)
        return path, run
    finally:
        p.close()


def test_v2_equals_experiment_b_when_values_are_equal(synthetic, tmp_path):
    cfg, frame = synthetic
    # Oracolo: codice B originale, stesso input V1, processo separato.
    b_cfg = deepcopy(cfg)
    b_cfg.update(work_root=str(tmp_path / 'b' / 'v1_spatial_msa'), protocol='v1.0-spatial-native-centers-cell-loss-r1')
    for key in ['v2_root', 'encoder', 'v2_protocol']:
        b_cfg['data'].pop(key)
    b_cfg['training'].update(num_workers=0)
    (tmp_path / 'b_cfg.json').write_text(json.dumps(b_cfg))
    frame.to_csv(tmp_path / 'frame.csv', index=False)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '-1'}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('b_oracle.py')), str(tmp_path / 'b_cfg.json'),
                             str(tmp_path / 'frame.csv'), str(tmp_path / 'b_out')], capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    from train import predict_external, load_run
    path, run = run_v2(cfg, frame)
    predict_external(cfg, run, load_run(cfg, run))
    with sqlite3.connect(path) as db:
        cells = pd.read_sql('SELECT cell_id,n_total,n_valid,mean,variance,eligible FROM cells ORDER BY cell_id', db)
        patches = pd.read_sql('SELECT cell_id,tile_id,row0,col0,height,width,mask,n_valid FROM patches ORDER BY patch_id', db)
    cells['mean'] = [np.frombuffer(x, dtype=np.float32).tobytes().hex() for x in cells['mean']]
    cells['variance'] = [np.frombuffer(x, dtype=np.float32).tobytes().hex() for x in cells['variance']]
    patches['mask'] = [bytes(x).hex() for x in patches['mask']]
    b_cells = pd.read_csv(tmp_path / 'b_out' / 'cells.csv', dtype={'cell_id': str})
    b_patches = pd.read_csv(tmp_path / 'b_out' / 'patches.csv', dtype={'cell_id': str})
    pd.testing.assert_frame_equal(cells.reset_index(drop=True), b_cells, check_dtype=False)
    pd.testing.assert_frame_equal(patches.reset_index(drop=True), b_patches, check_dtype=False)
    for name in ['oof', 'liguria']:
        mine = pd.read_csv(run / f'{name}_predictions.csv', dtype={'cell_id': str}).sort_values(['model', 'cell_id']).reset_index(drop=True)
        theirs = pd.read_csv(tmp_path / 'b_out' / f'{name}.csv', dtype={'cell_id': str}).sort_values(['model', 'cell_id']).reset_index(drop=True)
        # Stesso calcolo in FP32 su CPU: uguaglianza esatta, nessuna tolleranza.
        pd.testing.assert_frame_equal(mine, theirs, check_exact=True)
    assert (run / 'selection.json').exists() and (run / 'refit_complete.json').exists()


def test_dataloader_workers_do_not_change_weights(cache):
    from dataset import cell_frame, fit_normalizer
    from train import train_neural, identity
    cfg, frame, path = cache
    cells = cell_frame(path)
    train_part = cells[(cells.partition == 'train') & (cells.cv_fold != 0)].reset_index(drop=True)
    val = cells[(cells.partition == 'train') & (cells.cv_fold == 0)].reset_index(drop=True)
    weights = []
    for workers in [0, 2]:
        local = deepcopy(cfg); local['training'].update(num_workers=workers, epochs=2, patience=5, prefetch_factor=2)
        folder = Path(cfg['work_root']) / f'w{workers}'
        p = Progress(local, 'w')
        try:
            model, _, _ = train_neural(local, path, train_part, val, 'spatial_cnn', folder, 'x', p, 2, False)
        finally:
            p.close()
        weights.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        assert sorted(x.name for x in folder.iterdir()) == ['best.pt', 'epochs.csv', 'last.pt']
    for key in weights[0]:
        assert torch.equal(weights[0][key], weights[1][key]), key


def test_parallel_jobs_equal_single_run(synthetic, tmp_path):
    from train import parse_jobs
    cfg, frame = synthetic
    _, single = run_v2(cfg, frame, 'single')
    for item in ['spatial_cnn:0', 'spatial_cnn:1', 'spatial_cnn:2', 'mean_mlp:0,mean_mlp:1,mean_mlp:2,mean_histgb:0,mean_histgb:1,mean_histgb:2']:
        run_v2(cfg, frame, 'split', 'cv', parse_jobs(item))
    split = Path(cfg['work_root']) / 'runs' / 'split'
    assert not (split / 'selection.json').exists()
    run_v2(cfg, frame, 'split', 'cv', None)  # tutti i fold completi: solo selezione
    for item in ['spatial_cnn:refit', 'mean_mlp:refit', 'mean_histgb:refit']:
        run_v2(cfg, frame, 'split', 'refit', parse_jobs(item))
    assert not (split / 'refit_complete.json').exists()
    run_v2(cfg, frame, 'split', 'refit', None)  # tutti i refit completi: solo verifica e marker finale
    assert (split / 'refit_complete.json').exists()
    assert json.loads((split / 'selection.json').read_text())['selected_model'] == json.loads((single / 'selection.json').read_text())['selected_model']
    a = pd.read_csv(single / 'oof_predictions.csv'); b = pd.read_csv(split / 'oof_predictions.csv')
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    for name in ['spatial_cnn', 'mean_mlp']:
        x = torch.load(single / name / 'refit' / 'best.pt', weights_only=False)['model']
        y = torch.load(split / name / 'refit' / 'best.pt', weights_only=False)['model']
        assert all(torch.equal(x[k], y[k]) for k in x)


def test_missing_v2_pixels_are_rejected(synthetic):
    cfg, frame = synthetic
    ref = Path(cfg['data']['reference']) / 'aoi_03'
    tile = next(ref.iterdir())
    with np.load(tile / 'geometry.npz') as z:
        flat = z['flat_indices']
    np.savez(tile / 'geometry.npz', flat_indices=flat[: len(flat) // 2], cell_ids=np.zeros(len(flat) // 2, dtype=np.int64))
    with pytest.raises(ValueError, match='Geometria V2 diversa|non codificati|non conformi'):
        run_v2(cfg, frame)


def test_wrong_v2_protocol_is_rejected(synthetic, tmp_path):
    cfg, frame = synthetic
    meta = next((Path(cfg['data']['v2_root']) / 'pixel_embeddings' / 'aoi_05').glob('*.json'))
    value = json.loads(meta.read_text()); value['identity']['protocol'] = 'dense127909-stream-legacy-exact-r1'
    meta.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='protocollo'):
        run_v2(cfg, frame)


def test_modified_v2_file_is_rejected_after_build(cache):
    from dataset import verify_sources
    cfg, frame, path = cache
    target = next((Path(cfg['data']['v2_root']) / 'pixel_embeddings' / 'aoi_01').glob('*.npy'))
    data = np.load(target); data[0, 0] += 1; np.save(target, data)
    with pytest.raises(ValueError, match='V2'):
        verify_sources(path)


def test_work_root_name_and_overlap_are_enforced(tmp_path):
    from runtime import common_parser, configuration
    p = common_parser('x')
    with pytest.raises(ValueError, match='v2_spatial_msa'):
        configuration(p.parse_args(['--work-root', str(tmp_path / 'other')]))
    with pytest.raises(ValueError, match='sovrapposti'):
        configuration(p.parse_args(['--work-root', str(tmp_path / 'v2_spatial_msa'), '--v2-root', str(tmp_path / 'v2_spatial_msa' / 'x')]))
