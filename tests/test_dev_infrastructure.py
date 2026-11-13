# Scopo: verificare setup privato e vincoli offline della suite unificata senza installazioni.
# Fasi: piano installazione per piattaforma -> rifiuto percorso produttivo -> blocco I/O esterno.
# Input: setup Python e fixture pytest; output: assert, nessun pacchetto scaricato o stato produttivo.
# Parametri: python -m pytest; test inclusi nello stesso comando delle campagne e degli esperimenti.
# Risorse: CPU e metadati; rete esterna proibita, server loopback sintetici ammessi.
# Ripresa: nessuno stato persistente; i mock vengono ripristinati da pytest.
import importlib.util
from pathlib import Path
import socket
import tomllib

import pytest
import rasterio
import requests

from biomap_v2.common import ROOT


def setup_module_from_file():
    spec = importlib.util.spec_from_file_location('biomap_setup_dev', ROOT / 'scripts/setup_dev.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_preserves_cpu_indices_and_editable_install():
    module = setup_module_from_file()
    for system in ['darwin', 'linux', 'win32']:
        commands = module.install_commands(Path('private_python'), system)
        torch = next(c for c in commands if 'torch==2.10.0' in c)
        assert ('--index-url' in torch) == (system != 'darwin')
        assert any('--no-deps' in c and '--no-build-isolation' in c and '-e' in c for c in commands)
        assert all('cu128' not in ' '.join(c) for c in commands)
    with pytest.raises(ValueError, match='privata'):
        module.destination(ROOT / '.venv')


def test_standard_pytest_includes_spatial_experiment():
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert set(config['tool']['pytest']['ini_options']['testpaths']) == {'tests', 'experiments/v1_spatial_cnn_msa/tests'}


def test_external_requests_and_python_sockets_are_blocked():
    with pytest.raises(RuntimeError, match='offline'):
        requests.get('https://example.org/not-a-real-test-fixture', timeout=1)
    with pytest.raises(RuntimeError, match='offline'):
        socket.create_connection(('example.org', 443), timeout=1)


def test_external_geospatial_reads_are_blocked():
    for url in ['https://example.org/data.tif', '/vsicurl/https://example.org/data.tif', '/vsis3/example/data.tif']:
        with pytest.raises(RuntimeError, match='offline'):
            rasterio.open(url)
