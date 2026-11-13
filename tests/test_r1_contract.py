# Scopo: proteggere il contratto r1 pubblicato, senza modificare i cinque CODE_FILES congelati.
# Fasi: riferimento versionato -> SHA/EOL -> protocollo/default -> cache sintetica/ripresa/rifiuti.
# Input: fixture derivata dal commit pubblico 1cdc762 e checkout locale; output: assert CPU offline.
# Parametri: python -m pytest tests/test_r1_contract.py; nessun dato o marker universitario.
# Risorse: piccoli NPZ sintetici, nessun encoder; firme raw del checkout distinte dalle blob Git.
# Ripresa: fixture sempre nuova in tmp_path; non aggiornare i riferimenti per far passare una regressione.
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biomap_v2.campaign import streaming as r1
from biomap_v2.campaign import cli
from biomap_v2.campaign.state import atomic_npz
from biomap_v2.common import ROOT, WEIGHTS_SHA256, digest

CONTRACT = json.loads((Path(__file__).parent / 'fixtures/r1_contract_1cdc762.json').read_text())


def test_r1_protocol_and_code_identity_are_the_published_contract():
    assert r1.PROTOCOL == CONTRACT['protocol']
    assert r1.CODE_FILES == CONTRACT['code_files']
    assert r1.code_identity() == {name: spec['sha256_worktree'] for name, spec in CONTRACT['files'].items()}
    assert WEIGHTS_SHA256 == CONTRACT['checkpoint_sha256']
    assert r1.S2_INVALID_SCL == CONTRACT['invalid_scl']
    assert r1.READ_ATTEMPTS == CONTRACT['read_attempts']


@pytest.mark.parametrize('name', CONTRACT['code_files'])
def test_r1_raw_bytes_and_line_endings(name):
    spec = CONTRACT['files'][name]
    raw = (ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['sha256_worktree'], f'Byte r1 cambiati: {name}'
    if spec['eol'] == 'lf':
        assert b'\r' not in raw
    else:
        assert raw.count(b'\n') == raw.count(b'\r\n') > 0, f'EOL r1 alterati: {name}'


def test_r1_eol_rules_and_cache_validator_remain_pinned():
    lines = (ROOT / '.gitattributes').read_text().splitlines()
    rules = {line.split()[0]: line.split()[1:] for line in lines if line.strip() and not line.startswith('#')}
    for name, spec in CONTRACT['files'].items():
        assert rules[name] == ['text', 'eol=' + spec['eol']]
    # A broad later rule could override the intended mixed EOL checkout.
    assert '*.py' not in rules and '*' not in rules
    state = (ROOT / 'src/biomap_v2/campaign/state.py').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(state).hexdigest() == CONTRACT['state_source_sha256_lf']


def test_r1_cli_and_production_launcher_defaults():
    args = cli.parser().parse_args(['stream', '--reference', 'synthetic_reference', '--work-root', 'synthetic_work'])
    for name, expected in CONTRACT['cli_stream_defaults'].items():
        assert getattr(args, name) == expected, f'Default r1 cambiato: {name}'
    wrapper = (ROOT / 'scripts/run_v2_dense_stream.ps1').read_text(encoding='utf-8-sig')
    assert hashlib.sha256(wrapper.replace('\r\n', '\n').encode()).hexdigest() == CONTRACT['windows_launcher_sha256_lf']


def test_stream_command_still_dispatches_to_r1_without_extraction(monkeypatch, tmp_path):
    calls = []
    frame = pd.DataFrame({'aoi_id': ['synthetic']})
    monkeypatch.setattr(cli, 'load', lambda path: ({'areas': []}, frame))
    monkeypatch.setattr(r1, 'stream', lambda args, spec, f: calls.append(args))
    monkeypatch.setattr(cli, 'extract', lambda *a, **k: pytest.fail('stream deve usare r1, non extract'))
    work = tmp_path / 'work'
    cli.main(['stream', '--reference', str(tmp_path / 'reference'), '--work-root', str(work)])
    assert len(calls) == 1 and calls[0].amp is False
    assert calls[0].device == CONTRACT['cli_stream_defaults']['device']
    assert json.loads((work / 'last_invocation.json').read_text())['args']['command'] == 'stream'


@pytest.fixture
def cached_r1_tile(tmp_path):
    tile = tmp_path / 'reference/aoi/grid_test'
    tile.mkdir(parents=True)
    np.savez(tile / 'geometry.npz', flat_indices=np.arange(3), cell_ids=np.array([0, 0, 1]))
    (tile / 'geometry.json').write_text(json.dumps({'pixels': 3}))
    identity = {'protocol': CONTRACT['protocol'], 'geometry': digest(tile / 'geometry.npz'),
                'year': CONTRACT['identity_year'], 'checkpoint': CONTRACT['checkpoint_sha256'],
                'precision': CONTRACT['identity_precision'],
                'code': {name: spec['sha256_worktree'] for name, spec in CONTRACT['files'].items()}}
    path = tmp_path / 'work/chunks/aoi/grid_test/tile.npz'
    atomic_npz(path, identity, cell=np.array([0, 1]), sums=np.ones((2, 128)), counts=np.array([2, 1]))
    args = SimpleNamespace(reference=str(tmp_path / 'reference'), work_root=str(tmp_path / 'work'))
    return args, identity, path


def test_published_r1_cache_identity_reuses_tile_without_encoder(cached_r1_tile, monkeypatch):
    args, identity, path = cached_r1_tile
    monkeypatch.setattr(r1, 'load_model', lambda *a: pytest.fail('Cache valida: nessun encoder'))
    monkeypatch.setattr(r1, 'catalog', lambda *a: pytest.fail('Cache valida: nessuna rete'))
    cell, sums, counts, log = r1.process_tile(args, 'aoi', 'grid_test', None, {'code': identity['code']}, None, None)
    assert cell.tolist() == [0, 1] and counts.tolist() == [2, 1] and sums.shape == (2, 128) and log is None
    assert list(identity) == CONTRACT['identity_fields']
    assert not (Path(args.work_root) / 'tmp').exists()


@pytest.mark.parametrize('field', CONTRACT['identity_fields'])
def test_r1_rejects_identity_changes_instead_of_recomputing(cached_r1_tile, field):
    args, identity, path = cached_r1_tile
    marker = path.with_suffix('.json')
    original = path.read_bytes()
    info = json.loads(marker.read_text())
    info['identity'][field] = {'deliberately_corrupted': True}
    marker.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='Cache incompatibile/corrotta'):
        r1.process_tile(args, 'aoi', 'grid_test', None, {'code': identity['code']}, None, None)
    assert path.read_bytes() == original


def test_r1_rejects_payload_checksum_changes(cached_r1_tile):
    args, identity, path = cached_r1_tile
    path.write_bytes(path.read_bytes() + b'deliberate-corruption')
    with pytest.raises(ValueError, match='Cache incompatibile/corrotta'):
        r1.process_tile(args, 'aoi', 'grid_test', None, {'code': identity['code']}, None, None)
