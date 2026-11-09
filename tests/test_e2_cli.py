# Controlli delle interfacce pubbliche, senza sorgente reale, download o GPU.
import os
import subprocess
import sys
import pytest
from biomap_v2.common import ROOT


@pytest.mark.parametrize('module', ['prepare', 'train', 'evaluate', 'predict', 'tools.check_environment', 'tools.profile_training'])
def test_e2_help_from_another_directory(module, tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src')
    result = subprocess.run([sys.executable, '-m', 'biomap_ml.'+module, '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert '--source-data-root' in result.stdout and '--work-root' in result.stdout
