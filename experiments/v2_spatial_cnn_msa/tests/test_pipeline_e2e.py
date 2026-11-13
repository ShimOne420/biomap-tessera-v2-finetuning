# SCOPO
#   Prova end-to-end dell'orchestratore run_pipeline.py e del monitor su dati sintetici: tutte le fasi dopo
#   l'estrazione (verifica r1, build, preflight, profilo, CV parallela, selezione, refit, valutazione) come
#   sottoprocessi reali, con worker del DataLoader attivi, poi ripresa idempotente.
# COME VIENE RICHIAMATO
#   pytest experiments/v2_spatial_cnn_msa/tests/test_pipeline_e2e.py
# LOGICA IN ORDINE
#   fixture sintetica -> membership/target dichiarati via --set -> marker di estrazione r2 e parquet r1/r2
#   identici -> pipeline completa -> output attesi -> secondo lancio che salta tutto -> monitor.
# INPUT E OUTPUT
#   Solo cartelle temporanee.
# PARAMETRI
#   Nessuno.
# RISORSE
#   CPU, circa due minuti.
# RIPRESA
#   Non applicabile.
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import EXPERIMENT


def fake_extraction(root, cells, shift=0.0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({'cell_id': cells})
    for i in range(128):
        frame[f'mean_{i:03d}'] = rng.normal(size=len(cells)) + shift
    frame.to_parquet(root / 'v2.parquet', index=False)
    (root / 'extraction_complete.json').write_text(json.dumps({'method': 'dense127909-stream-legacy-exact-r2-fast', 'complete_campaign': True}))


def test_full_pipeline_runs_resumes_and_monitors(synthetic, tmp_path):
    cfg, frame = synthetic
    frame.to_csv(tmp_path / 'membership.csv', index=False)
    fake_extraction(Path(cfg['data']['v2_root']), frame.cell_id.tolist())
    fake_extraction(tmp_path / 'r1work', frame.cell_id.tolist())
    work = tmp_path / 'v2_spatial_msa'
    sets = [f"data.membership_csv={tmp_path / 'membership.csv'}", f"data.targets_csv={cfg['data']['targets_csv']}",
            'training.device=cpu', 'training.epochs=2', 'training.patience=1', 'training.batch_cells=8', 'training.num_workers=2',
            'training.prefetch_factor=2', 'training.pin_memory=false', 'model.conv_channels=[8,4]', 'model.hidden=8', 'model.dropout=0.0',
            'storage.max_gib=1', 'storage.reserve_gib=0']
    cmd = [sys.executable, str(EXPERIMENT / 'run_pipeline.py'), '--work-root', str(work), '--v2-root', cfg['data']['v2_root'],
           '--v1-root', cfg['data']['v1_root'], '--reference', cfg['data']['reference'], '--r1-work', str(tmp_path / 'r1work'),
           '--checkpoint', str(tmp_path / 'none.pt'), '--run-name', 'e2e']
    for item in sets:
        cmd += ['--set', item]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1500)
    logs = '\n'.join(p.read_text(encoding='utf-8', errors='replace')[-2000:] for p in (work / 'logs').glob('*.log')) if (work / 'logs').exists() else ''
    assert result.returncode == 0, result.stderr[-3000:] + logs
    pipe = json.loads((work / 'pipeline.json').read_text(encoding='utf-8'))
    assert pipe['status'] == 'complete' and all(s['status'] == 'complete' for s in pipe['stages'].values())
    assert json.loads((work / 'verify_r1.json').read_text())['bitwise_equal'] is True
    run = work / 'runs' / 'full' / 'e2e'
    for name in ['selection.json', 'refit_complete.json', 'oof_metrics.csv', 'liguria_metrics.csv', 'liguria_report.md']:
        assert (run / name).exists(), name
    liguria = pd.read_csv(run / 'liguria_predictions.csv')
    assert set(liguria.model) == {'spatial_cnn', 'mean_mlp', 'mean_histgb', 'dummy_mean'}
    # Secondo lancio: tutte le fasi complete vengono saltate.
    again = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    assert again.returncode == 0, again.stderr[-2000:]
    assert len(json.loads((work / 'pipeline.json').read_text(encoding='utf-8'))['history']) == 2
    monitor = subprocess.run([sys.executable, str(EXPERIMENT / 'monitor_v2.py'), '--work-root', str(work), '--v2-root', cfg['data']['v2_root']],
                             capture_output=True, text=True, env=env, timeout=60)
    assert monitor.returncode == 0 and 'RUN COMPLETATO' in monitor.stdout, monitor.stdout + monitor.stderr


def test_pipeline_stops_when_r2_differs_from_r1(synthetic, tmp_path):
    cfg, frame = synthetic
    frame.to_csv(tmp_path / 'membership.csv', index=False)
    fake_extraction(Path(cfg['data']['v2_root']), frame.cell_id.tolist())
    fake_extraction(tmp_path / 'r1work', frame.cell_id.tolist(), shift=1e-3)
    work = tmp_path / 'v2_spatial_msa'
    cmd = [sys.executable, str(EXPERIMENT / 'run_pipeline.py'), '--work-root', str(work), '--v2-root', cfg['data']['v2_root'],
           '--v1-root', cfg['data']['v1_root'], '--reference', cfg['data']['reference'], '--r1-work', str(tmp_path / 'r1work'),
           '--checkpoint', str(tmp_path / 'none.pt'), '--set', f"data.membership_csv={tmp_path / 'membership.csv'}"]
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, timeout=300)
    assert result.returncode != 0
    pipe = json.loads((work / 'pipeline.json').read_text(encoding='utf-8'))
    assert pipe['status'] == 'failed' and pipe['stages']['verify_r1']['status'] == 'failed'
    assert pipe['stages']['build']['status'] == 'pending'
