# Scopo: verificare l'installazione BIOMAP e gli import CPU senza accedere a dati o pesi.
# Fasi: Python/versioni -> sintassi sorgenti -> import installati e B isolato -> forward/backward CPU.
# Input: ambiente con progetto installato; output: JSON a terminale e facoltativamente --json FILE.
# Parametri: --probe-mps prova esplicitamente MPS se disponibile; --json salva il resoconto.
# Esempio: .venv-dev/bin/python scripts/check_dev_environment.py --json .venv-dev/environment.json.
# Risorse: moduli locali e un tensore minuscolo; niente rete, CUDA, dataset o encoder caricato.
# Ripresa: rieseguibile; ogni import/errore di sintassi fallisce il comando, senza skip impliciti.
import argparse
import ast
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT_MODULES = [
    'biomap_v2.cli', 'biomap_v2.encoder', 'biomap_v2.campaign.cli',
    'biomap_v2.campaign.streaming', 'biomap_v2.dense_fast.pipeline',
    'biomap_dense_heads.cli', 'biomap_ml.campaign',
]
DEPENDENCIES = ['numpy', 'pandas', 'pyarrow', 'rasterio', 'pyproj', 'scipy',
                'sklearn', 'xgboost', 'torch', 'yaml', 'requests', 'filelock', 'openpyxl']


def syntax_check():
    count = 0
    for folder in ['src', 'scripts', 'tests', 'experiments']:
        for parent, directories, names in os.walk(ROOT / folder):
            directories[:] = [n for n in directories if not n.startswith('.') and n != '__pycache__']
            for name in names:
                if name.endswith('.py'):
                    path = Path(parent) / name
                    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
                    count += 1
    ast.parse((ROOT / 'conftest.py').read_text(encoding='utf-8'))
    return count + 1


def main(argv=None):
    parser = argparse.ArgumentParser(description='Import e prova CPU BIOMAP, nessun download')
    parser.add_argument('--probe-mps', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error('Questo progetto richiede Python 3.12')
    report = {'python': platform.python_version(), 'executable': sys.executable,
              'platform': platform.platform(), 'architecture': platform.machine(),
              'project_version': importlib.metadata.version('biomap-tessera-v2-finetuning'),
              'syntax_files': syntax_check(), 'imports': []}
    for name in DEPENDENCIES + PROJECT_MODULES:
        module = importlib.import_module(name)
        if name in PROJECT_MODULES and not Path(module.__file__).resolve().is_relative_to(ROOT / 'src'):
            raise RuntimeError(f'{name} proviene da un altro clone: reinstallare questo progetto nella venv privata')
        report['imports'].append(name)
    # B uses bare module imports. Isolate them so they cannot shadow installed project modules.
    experiment = ROOT / 'experiments/v1_spatial_cnn_msa'
    subprocess.run([sys.executable, '-c',
                    'import sys; sys.path.insert(0,sys.argv[1]); import runtime,sources,patch_builder,dataset,model,train,evaluate,preflight,monitor',
                    str(experiment)], cwd=ROOT, check=True)
    import torch
    import numpy as np
    from biomap_v2.campaign.streaming import s1_storage
    value = torch.tensor([1., 2.], requires_grad=True)
    value.square().sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([2., 4.]))
    report.update(torch=torch.__version__, cpu_forward_backward='passed',
                  mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available(),
                  cuda_available=torch.cuda.is_available(), mps_probe='not_requested', encoder_loaded=False)
    # Observe the published boundary case; do not change NumPy dispatch or production arithmetic.
    boundary = np.array([0.05949767306447029, 0., -1., np.nan, 1e-9])
    positive = boundary.astype(np.float32)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    report.update(numpy=np.__version__, numpy_log10_dispatch=np.lib.introspect.opt_func_info(func_name='log10'),
                  r1_s1_boundary={'input': float(boundary[0]), 'storage': int(s1_storage(boundary)[0]),
                                  'before_integer': float(((np.float32(20) * np.log10(positive) + np.float32(50)) * np.float32(200))[0]),
                                  'contract_test': 'tests/test_dense_public.py::test_s1_storage_historical_convention'})
    if args.probe_mps:
        if not report['mps_available']:
            raise RuntimeError('MPS richiesto ma non disponibile; CPU resta il riferimento dei test')
        value = torch.tensor([1., 2.], device='mps', requires_grad=True)
        value.square().sum().backward()
        torch.testing.assert_close(value.grad.cpu(), torch.tensor([2., 4.]))
        report['mps_probe'] = 'passed'
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
