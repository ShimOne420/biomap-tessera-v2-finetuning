# Test CPU senza dati reali: griglia geografica, maschere, pesi e CNN minimale.
# Le fixture non misurano accuratezza MSA né prestazioni GPU universitarie.
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from biomap_v2.data import sample_points
from biomap_ml.spatial import lattice, make_grid, SpatialCNN, stratum_weights
from biomap_ml.settings import separate_roots
from biomap_ml.cache import encode_record, decode_record

torch.set_num_threads(1)


def cell():
    return SimpleNamespace(west=8., east=8+1/360, south=44., north=44+1/360)


def test_north_up_grid_and_missing_points():
    c = cell()
    points = sample_points(c, 25)
    assert lattice(points, c).tolist() == [20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    keep = np.arange(25) != 12
    embeddings = np.broadcast_to(np.arange(25)[keep, None], (24, 128))
    grid, mask, weights = make_grid(embeddings, points[keep], c)
    assert grid[0, 0, 0] == 20 and grid[0, 4, 4] == 4
    assert mask[0, 2, 2] == 0 and mask.sum() == 24
    assert np.all(grid[:, 2, 2] == 0) and np.isclose(weights.sum(), 1)
    assert not np.array_equal(weights[0, 0], weights[0, -1])


def test_lattice_rejects_duplicates_outside_and_arbitrary_points():
    c = cell()
    p = sample_points(c, 25)
    with pytest.raises(ValueError, match='duplicati'):
        lattice(p[[0, 0]], c)
    with pytest.raises(ValueError, match='fuori'):
        lattice(p+1, c)
    with pytest.raises(ValueError, match='griglia'):
        lattice(p+1e-5, c)


def test_masked_weighted_pool_and_gradients():
    c = cell()
    grid, mask, weights = make_grid(np.ones((25, 128)), sample_points(c, 25), c)
    model = SpatialCNN()
    x, m, w = [torch.from_numpy(v[None]) for v in (grid, mask, weights)]
    value = model(x, m, w)
    assert value.shape == (1,) and 0 < value.item() < 1
    value.square().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    torch.testing.assert_close(model(x, m, w), torch.tensor([.5]))
    with pytest.raises(ValueError, match='senza punti'):
        model(x, m*0, w)


def test_cache_corruption_is_detected():
    c = cell()
    values = make_grid(np.ones((25, 128)), sample_points(c, 25), c)
    blob, sha = encode_record(values)
    for expected, actual in zip(values, decode_record(blob, sha)):
        np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ValueError, match='corrotta'):
        decode_record(blob[:-1]+bytes([blob[-1] ^ 1]), sha)


def test_source_and_work_must_be_disjoint_even_with_symlink(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    for work in [source, source/'child', tmp_path]:
        with pytest.raises(ValueError, match='separate'):
            separate_roots(source, work)
    alias = tmp_path/'alias'
    try:
        alias.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip('Creazione symlink non consentita su questo Windows')
    with pytest.raises(ValueError, match='separate'):
        separate_roots(source, alias/'child')
