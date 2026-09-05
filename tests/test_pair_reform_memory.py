import pytest
import torch

from cvsrffi.orbit_teacher import DenseTemporalOrbitMemory, coverage_mixture_weights


def memory(**kwargs):
    return DenseTemporalOrbitMemory(train_physical_ids=torch.tensor([10, 20, 30]), feature_dim=2, **kwargs)


def write(cache, keys, features, quality, step=0, **kwargs):
    cache.update(keys=torch.tensor(keys), features=torch.tensor(features, dtype=torch.float32),
                 reliability=torch.tensor(quality), scenario_bin=torch.zeros(len(keys), dtype=torch.long),
                 receiver_bin=torch.zeros(len(keys), dtype=torch.long), step=step, **kwargs)


def test_reliability_admission_monotonic_and_initial_rejection():
    low, high = memory(), memory()
    for cache in (low, high):
        write(cache, [10, 20], [[1, 0], [0, 1]], [1., .01])
        assert cache.lookup(torch.tensor([20]), step=0)[1].tolist() == [False]
    write(low, [10], [[0, 1]], [.1], step=1)
    write(high, [10], [[0, 1]], [1.], step=1)
    assert low.features[0, 1] < high.features[0, 1]
    before = low.state_dict()
    write(low, [10], [[0, 1]], [0.], step=2)
    assert torch.equal(before['features'], low.features)
    assert torch.equal(before['last_seen'], low.last_seen)


def test_duplicate_merge_is_permutation_invariant():
    a, b = memory(), memory()
    write(a, [10, 10], [[1, 0], [0, 1]], [.25, 1.])
    write(b, [10, 10], [[0, 1], [1, 0]], [1., .25])
    torch.testing.assert_close(a.features, b.features)
    expected = torch.nn.functional.normalize(torch.tensor([.25, 1.]), dim=0)
    torch.testing.assert_close(a.features[0], expected)


def test_missing_stale_and_unknown_role():
    cache = memory(ttl=2)
    write(cache, [10], [[1, 0]], [1.], step=3)
    assert cache.lookup(torch.tensor([10, 20]), step=5)[1].tolist() == [True, False]
    values, found, meta = cache.lookup(torch.tensor([10]), step=6)
    assert not found.any() and not values.any() and meta['last_seen'].item() == -1
    for role in ('V', 'target', 'U'):
        with pytest.raises(ValueError, match='TRAIN'):
            cache.lookup(torch.tensor([10]), step=4, role=role)
    with pytest.raises(ValueError, match='unknown'):
        write(cache, [99], [[1, 0]], [1.])
    with pytest.raises(ValueError, match='unique'):
        DenseTemporalOrbitMemory(train_physical_ids=torch.tensor([10, 10]), feature_dim=2)


def test_restore_validates_map_and_is_atomic():
    cache = memory()
    write(cache, [20], [[0, 1]], [1.], step=4)
    restored = memory()
    restored.load_state_dict(cache.state_dict())
    torch.testing.assert_close(restored.lookup(torch.tensor([20]), step=4)[0], torch.tensor([[0., 1.]]))
    state = cache.state_dict()
    state['keys'][0] = 99
    with pytest.raises(ValueError, match='map'):
        restored.load_state_dict(state)
    state = cache.state_dict()
    state['features'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        restored.load_state_dict(state)
    torch.testing.assert_close(restored.features, cache.features)


def test_invalid_views_receive_no_coverage_mass():
    quality = torch.tensor([[1., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    weights = coverage_mixture_weights(quality, torch.ones_like(quality), prior=torch.ones(3),
                                      coverage_floor=.3, valid_mask=mask)
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))
    torch.testing.assert_close(weights.sum(-1), torch.tensor([1., 1., 0.]))
