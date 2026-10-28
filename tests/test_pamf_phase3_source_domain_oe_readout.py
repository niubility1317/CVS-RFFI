"""Domain-balanced source membership readers; synthetic descriptors only."""
import copy
import importlib
import inspect

import numpy as np
import pytest

from experiments.pamf_phase3_open_world import source_open_readout as core


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_domain_oe_readout')


def toy(known_per_domain=3, oe_per_domain=2):
    rng = np.random.default_rng(14)
    kd = np.repeat(np.arange(15), known_per_domain)
    od = np.repeat(np.arange(15), oe_per_domain)
    known = rng.normal(0, .05, (len(kd), 100)).astype(np.float32)
    oe = rng.normal(0, .05, (len(od), 100)).astype(np.float32)
    known[:, :5] += 2.; oe[:, :5] -= 2.
    known[:, 99] = oe[:, 99] = 7.
    return known, oe, kd, od


def equal_state(left, right):
    if isinstance(left, np.ndarray): np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left: equal_state(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right): equal_state(a, b)
    else: assert left == right


def test_domain_mapping_uses_declared_main_receiver_order():
    rx = np.repeat([1, 3, 4, 6, 8], 3)
    days = np.tile([1, 2, 3], 5)
    result = module().domains_from_metadata(rx, days)
    np.testing.assert_array_equal(result, np.arange(15))
    assert result.dtype == np.int64


@pytest.mark.parametrize('rx,day', [([0], [1]), ([1], [0]), ([1], [4]), ([True], [1]), ([1], [True]), ([1, 3], [1])])
def test_invalid_receiver_day_metadata_is_rejected(rx, day):
    with pytest.raises(ValueError): module().domains_from_metadata(np.asarray(rx), np.asarray(day))


def test_old_uniform_three_to_two_prior_reduces_to_unweighted_normalizer():
    known, oe, kd, od = toy()
    state = module().fit_readouts(known, oe, kd, od)
    combined = np.r_[known, oe].astype(np.float64)
    scale = combined.std(axis=0, ddof=0); scale[scale <= 1e-8] = 1.
    np.testing.assert_allclose(state['mean'], combined.mean(axis=0), atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(state['scale'], scale, atol=2e-14, rtol=2e-14)
    assert state['scale'][99] == 1.
    assert state['training']['regime'] == 'source_role_domain_balanced_v1'
    assert state['training']['normalizer_role_prior'] == {'known': .6, 'oe': .4}
    assert state['training']['loss_role_prior'] == {'known': .5, 'oe': .5}


def test_repeating_one_complete_domain_preserves_role_domain_statistics():
    known, oe, kd, od = toy(2, 2)
    original = module().fit_readouts(known, oe, kd, od)
    selected = kd == 4
    repeated = module().fit_readouts(np.r_[known, np.repeat(known[selected], 9, axis=0)], oe,
                                     np.r_[kd, np.repeat(kd[selected], 9)], od)
    np.testing.assert_allclose(original['mean'], repeated['mean'], atol=3e-14, rtol=2e-14)
    np.testing.assert_allclose(original['scale'], repeated['scale'], atol=3e-14, rtol=2e-14)
    assert repeated['training']['domain_counts']['known'][4] == 20
    assert repeated['training']['loss_weights']['known']['sum'] == pytest.approx(1.)
    assert repeated['training']['loss_weights']['oe']['sum'] == pytest.approx(1.)


def test_row_permutation_cpu_seed_and_core_prediction_compatibility(monkeypatch):
    import torch
    known, oe, kd, od = toy(2, 2)
    before_rng = torch.random.get_rng_state().tolist()
    calls = []
    original = torch.optim.AdamW

    class Optimizer(original):
        def step(self, *args, **kwargs):
            calls.append(self.defaults.copy())
            return super().step(*args, **kwargs)

    def forbidden(*args, **kwargs): raise AssertionError('Unsupported NumPy bridge')
    monkeypatch.setattr(torch.optim, 'AdamW', Optimizer)
    monkeypatch.setattr(torch, 'from_numpy', forbidden)
    monkeypatch.setattr(torch.Tensor, 'numpy', forbidden)
    state = module().fit_readouts(known, oe, kd, od)
    assert len(calls) == 400 and torch.random.get_rng_state().tolist() == before_rng
    assert all(v['lr'] == .001 and v['weight_decay'] == .0001 and v['betas'] == (.9, .999) and v['eps'] == 1e-8 for v in calls)
    assert state['schema'] == core.READOUT_SCHEMA
    assert state['training']['steps'] == 200
    assert state['training']['parameter_counts'] == {'linear_oe': 101, 'mlp16_oe': 1633}
    assert all(len(v) == 200 and np.isfinite(v).all() for v in state['training']['losses'].values())
    assert all(v[-1] < v[0] for v in state['training']['losses'].values())
    core._readout_state(state)
    values = core.score_readouts(np.r_[known, oe], state)
    assert np.all(values[:len(known)] > 0) and np.all(values[len(known):] < 0)
    snapshot = copy.deepcopy(state)
    independent = np.concatenate([core.score_readouts(row[None], state) for row in np.r_[known, oe]])
    np.testing.assert_allclose(values, independent, rtol=2e-6, atol=1e-6)
    equal_state(state, snapshot)
    rng = np.random.default_rng(19); a, b = rng.permutation(len(known)), rng.permutation(len(oe))
    reordered = module().fit_readouts(known[a], oe[b], kd[a], od[b])
    equal_state(state, reordered)
    assert list(inspect.signature(module().fit_readouts).parameters) == [
        'known_descriptors', 'oe_descriptors', 'known_domains', 'oe_domains', 'model_seed']
    assert set(state) == {'schema', 'descriptor_dim', 'method_names', 'mean', 'scale', 'models', 'training'}
    assert not any(key in state['training'] for key in ('features', 'labels', 'domains', 'weights'))


def test_extreme_domain_counts_cannot_create_membership_signal_from_domain_prior():
    # Both roles have exactly the same descriptor conditional on domain. Only
    # their observed domain frequencies differ: this must not be a binary cue.
    kd = np.arange(15)
    od = np.r_[np.zeros(200, dtype=np.int64), np.arange(1, 15)]
    known = np.zeros((15, 100), np.float32); known[:, 0] = kd
    oe = np.zeros((len(od), 100), np.float32); oe[:, 0] = od
    state = module().fit_readouts(known, oe, kd, od)
    assert state['mean'][0] == pytest.approx(7.)
    assert state['scale'][0] == pytest.approx(np.arange(15.).std(ddof=0))
    assert state['training']['loss_weights']['oe']['minimum'] == pytest.approx(1 / (15 * 200))
    assert state['training']['loss_weights']['oe']['maximum'] == pytest.approx(1 / 15)
    assert np.abs(core.score_readouts(known, state)[:, 0]).max() < .02


@pytest.mark.parametrize('fault', ['missing_known_domain', 'missing_oe_domain', 'bool_domain', 'bad_domain',
                                 'domain_shape', 'descriptor_shape', 'nonfinite', 'seed'])
def test_invalid_training_inputs_fail(fault):
    known, oe, kd, od = toy(1, 1)
    seed = 2026092701
    if fault == 'missing_known_domain': kd[-1] = 0
    elif fault == 'missing_oe_domain': od[-1] = 0
    elif fault == 'bool_domain': kd = kd.astype(bool)
    elif fault == 'bad_domain': kd[0] = 15
    elif fault == 'domain_shape': kd = kd[:, None]
    elif fault == 'descriptor_shape': oe = oe[:, :-1]
    elif fault == 'nonfinite': known[0, 0] = np.nan
    else: seed = True
    with pytest.raises(ValueError): module().fit_readouts(known, oe, kd, od, seed)
