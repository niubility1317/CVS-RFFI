"""Source-only OE geometry/readout tests; all training uses small toy arrays."""
import copy
import importlib
import inspect

import numpy as np
import pytest


DIMENSIONS = dict(fused160=160, time160=160, frequency160=160, pa160=160,
                  reference29=29, signature336=336, spectrum128=128, pa_moments48=48)


def core():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_open_readout')


def make_branches(count=18, seed=13):
    rng = np.random.default_rng(seed)
    values = {name: rng.normal(size=(count, dimension)).astype(np.float32)
              for name, dimension in DIMENSIONS.items()}
    values['reference29'][:, 27] = 0
    return values


@pytest.fixture(scope='module')
def geometry():
    branches = make_branches()
    y = np.repeat(np.arange(6, dtype=np.int64), 3)
    return branches, y, core().fit_geometry(branches, y)


def assert_same(left, right):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left: assert_same(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_same(a, b)
    else:
        assert left == right


def test_geometry_uses_only_l_and_stores_aggregates(geometry):
    branches, y, bundle = geometry
    assert list(inspect.signature(core().fit_geometry).parameters) == ['branches', 'y']
    assert bundle['class_count'] == 6
    for name, dimension in DIMENSIONS.items():
        branch = bundle['branches'][name]
        assert set(branch) == {'mean', 'scale', 'prototypes', 'whitener'}
        assert branch['prototypes'].shape == (6, dimension)
        assert branch['whitener'].shape == (dimension, dimension)
        if name == 'fused160':
            np.testing.assert_array_equal(branch['mean'], np.zeros(dimension))
            np.testing.assert_array_equal(branch['scale'], np.ones(dimension))
        else:
            np.testing.assert_allclose(branch['mean'], branches[name].astype(float).mean(axis=0))
    assert bundle['branches']['reference29']['mean'][27] == 0.
    assert bundle['branches']['reference29']['scale'][27] == 1.
    before = copy.deepcopy(bundle)
    core().descriptor(make_branches(3, 19), np.zeros((3, 6), np.float32), bundle)
    assert_same(bundle, before)
    with pytest.raises(TypeError): core().fit_geometry(branches, y, oe=branches)


def test_descriptor_is_class_permutation_invariant_and_row_local(geometry):
    branches, y, bundle = geometry
    inputs = make_branches(5, 21)
    logits = np.random.default_rng(22).normal(size=(5, 6)).astype(np.float32)
    values = core().descriptor(inputs, logits, bundle)
    assert values.shape == (5, 100) and values.dtype == np.float32
    order = np.asarray([4, 2, 5, 1, 0, 3])
    permuted = core().fit_geometry(branches, order[y])
    permuted_logits = np.empty_like(logits); permuted_logits[:, order] = logits
    np.testing.assert_allclose(values, core().descriptor(inputs, permuted_logits, permuted), rtol=2e-6, atol=2e-5)
    single = np.concatenate([core().descriptor({k: v[i:i + 1] for k, v in inputs.items()}, logits[i:i + 1], bundle)
                             for i in range(5)])
    np.testing.assert_allclose(values, single, rtol=2e-6, atol=2e-5)
    for j in range(8):
        assert np.all(np.diff(values[:, j * 12:j * 12 + 6], axis=1) <= 0)
        assert np.all(np.diff(values[:, j * 12 + 6:j * 12 + 12], axis=1) <= 0)


def test_zero_variance_zero_prototypes_and_large_logit_stability():
    branches = {name: np.zeros((12, dimension), np.float32) for name, dimension in DIMENSIONS.items()}
    y = np.tile(np.arange(6), 2)
    bundle = core().fit_geometry(branches, y)
    logits = np.tile(np.arange(1000., 994., -1., dtype=np.float32), (12, 1))
    values = core().descriptor(branches, logits, bundle)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[:, :96], np.zeros((12, 96)))
    np.testing.assert_allclose(values[:, 96], 1 / np.exp(-np.arange(6.)).sum(), rtol=1e-6)
    np.testing.assert_array_equal(values[:, 97:99], np.tile([1000., 1.], (12, 1)))
    np.testing.assert_allclose(values[:, 99], 1000. + np.log(np.exp(-np.arange(6.)).sum()), rtol=1e-6)
    for branch in bundle['branches'].values():
        np.testing.assert_array_equal(branch['prototypes'], np.zeros_like(branch['prototypes']))
        np.testing.assert_array_equal(branch['scale'], np.ones_like(branch['scale']))


@pytest.mark.parametrize('fault', ['missing_branch', 'extra_branch', 'wrong_dim', 'not_fp32', 'nonfinite',
                                 'explicit_cfo', 'missing_class', 'boolean_labels', 'label_count', 'wrong_class'])
def test_invalid_geometry_inputs_fail(fault):
    branches = make_branches(12)
    y = np.tile(np.arange(6), 2)
    if fault == 'missing_branch': branches.pop('pa160')
    elif fault == 'extra_branch': branches['CFO'] = np.zeros((12, 1), np.float32)
    elif fault == 'wrong_dim': branches['fused160'] = branches['fused160'][:, :-1]
    elif fault == 'not_fp32': branches['fused160'] = branches['fused160'].astype(np.float64)
    elif fault == 'nonfinite': branches['pa160'][0, 0] = np.inf
    elif fault == 'explicit_cfo': branches['reference29'][0, 27] = 1.
    elif fault == 'missing_class': y[y == 5] = 4
    elif fault == 'boolean_labels': y = y.astype(bool)
    elif fault == 'label_count': y = y[:-1]
    else: y[0] = 6
    with pytest.raises(ValueError): core().fit_geometry(branches, y)


@pytest.mark.parametrize('fault', ['logits_class', 'logits_nan', 'geometry_class', 'geometry_shape', 'cfo'])
def test_descriptor_contract_is_fixed_six_class(geometry, fault):
    _, _, bundle = geometry
    bundle = copy.deepcopy(bundle); branches = make_branches(2)
    logits = np.zeros((2, 6), np.float32)
    if fault == 'logits_class': logits = np.zeros((2, 4), np.float32)
    elif fault == 'logits_nan': logits[0, 0] = np.nan
    elif fault == 'geometry_class': bundle['class_count'] = 4
    elif fault == 'geometry_shape': bundle['branches']['fused160']['whitener'] = np.eye(4)
    else: branches['reference29'][0, 27] = 1.
    with pytest.raises(ValueError): core().descriptor(branches, logits, bundle)


def toy_descriptors():
    rng = np.random.default_rng(3)
    known = np.zeros((21, 100), np.float32)
    oe = np.zeros((7, 100), np.float32)
    known[:, :5] = 2. + rng.normal(0, .08, (21, 5))
    oe[:, :5] = -2. + rng.normal(0, .08, (7, 5))
    known[:, 99] = oe[:, 99] = 7.
    return known, oe


def test_binary_toy_fixed_budget_balanced_loss_and_numpy_bridge(monkeypatch):
    import torch
    module = core()
    known, oe = toy_descriptors()
    calls, first_bias = [], []
    original = torch.optim.AdamW

    class TrackedAdamW(original):
        def step(self, *args, **kwargs):
            if not calls:
                first_bias.extend(self.param_groups[0]['params'][1].grad.detach().cpu().tolist())
            calls.append(self.defaults.copy())
            return super().step(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Unsupported NumPy/Torch bridge')

    monkeypatch.setattr(torch.optim, 'AdamW', TrackedAdamW)
    monkeypatch.setattr(torch, 'from_numpy', forbidden)
    monkeypatch.setattr(torch.Tensor, 'numpy', forbidden)
    rng_before = torch.random.get_rng_state().tolist()
    state = module.fit_readouts(known, oe)
    assert torch.random.get_rng_state().tolist() == rng_before
    assert len(calls) == 400  # 200 complete updates for each of two models.
    assert first_bias == pytest.approx([0.], abs=1e-7)  # Unequal counts cannot add a membership-prior gradient.
    assert all(row['lr'] == .001 and row['weight_decay'] == .0001 and row['betas'] == (.9, .999)
               and row['eps'] == 1e-8 for row in calls)
    assert state['method_names'] == ['linear_oe', 'mlp16_oe']
    assert state['training']['steps'] == 200
    assert state['training']['parameter_counts'] == {'linear_oe': 101, 'mlp16_oe': 1633}
    assert state['training']['source_membership_supervision'] is True
    for name in state['method_names']:
        losses = state['training']['losses'][name]
        assert len(losses) == 200 and np.isfinite(losses).all() and losses[-1] < losses[0]
        assert sum(value.size for value in state['models'][name].values()) == state['training']['parameter_counts'][name]
    assert state['training']['losses']['linear_oe'][0] == pytest.approx(np.log(2.), rel=1e-6)
    assert state['scale'][99] == 1. and state['mean'][99] == 7.
    values = module.score_readouts(np.r_[known, oe], state)
    assert np.all(values[:len(known)] > 0) and np.all(values[len(known):] < 0)
    before = copy.deepcopy(state)
    single = np.concatenate([module.score_readouts(row[None], state) for row in np.r_[known, oe]])
    np.testing.assert_allclose(values, single, atol=1e-6, rtol=2e-6)
    assert_same(state, before)
    replica = module.fit_readouts(known, oe)
    assert_same(state, replica)
    assert list(inspect.signature(module.fit_readouts).parameters) == ['known_descriptors', 'oe_descriptors', 'model_seed']


@pytest.mark.parametrize('fault', ['empty_known', 'empty_oe', 'bad_columns', 'nonfinite', 'bad_seed'])
def test_readout_training_rejects_invalid_inputs(fault):
    known, oe = toy_descriptors()
    seed = 2026092701
    if fault == 'empty_known': known = known[:0]
    elif fault == 'empty_oe': oe = oe[:0]
    elif fault == 'bad_columns': known = known[:, :-1]
    elif fault == 'nonfinite': oe[0, 0] = np.nan
    else: seed = True
    with pytest.raises(ValueError): core().fit_readouts(known, oe, seed)


def test_numpy_prediction_rejects_invalid_state_without_training():
    state = dict(schema='phase3_source_oe_readouts_v1', descriptor_dim=100,
        method_names=['linear_oe', 'mlp16_oe'], mean=np.zeros(100), scale=np.ones(100),
        models=dict(linear_oe=dict(weight=np.zeros((1, 100), np.float32), bias=np.zeros(1, np.float32)),
                    mlp16_oe=dict(weight1=np.zeros((16, 100), np.float32), bias1=np.zeros(16, np.float32),
                                  weight2=np.zeros((1, 16), np.float32), bias2=np.zeros(1, np.float32))), training={})
    desc = np.zeros((2, 100), np.float32)
    np.testing.assert_array_equal(core().score_readouts(desc, state), np.zeros((2, 2)))
    bad = copy.deepcopy(state); bad['scale'][0] = 0
    with pytest.raises(ValueError): core().score_readouts(desc, bad)
    bad = copy.deepcopy(state); bad['models']['mlp16_oe']['weight1'][0, 0] = np.inf
    with pytest.raises(ValueError): core().score_readouts(desc, bad)
    with pytest.raises(ValueError): core().score_readouts(desc[:, :-1], state)
