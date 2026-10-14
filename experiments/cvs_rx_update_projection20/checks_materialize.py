"""Actual constructor and FP32-addition counterexamples; no source data/CP."""
import copy
import json

import torch
from torch.nn import functional as F

from .materialize import (SOURCE_RX, HEAD, ENCODER, protected_names,
                          fixed_logits, evaluate, actual_displacement)
from experiments.cvs_evidence_recomposition20.capture import forward as capture
from experiments.cvs_sigtime_heldcredit.held_credit import CACHE_ATTRIBUTES
from experiments.cvs_u_schedule50.counterfactual import _rng_snapshot, _equal


def _raises(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError('Malformed or non-source materialization was accepted')


def check():
    from experiments.cvs_crossrx_sharpness20 import design as d, model as api
    torch.set_num_threads(2)
    c = d.config(next(row for row in d.rows() if row['arm'] == 'BASE_SSL'
                     and row['model_seed'] == d.SEEDS[0]))
    model = api.build(c, 'cpu').eval().requires_grad_(True)
    with torch.no_grad():
        model.get_submodule(ENCODER).response_gain.fill_(.1)
    generator = torch.Generator().manual_seed(150319)
    x = torch.randn(10, 2, 256, generator=generator)
    labels = torch.arange(10, dtype=torch.long) % 6
    receivers = torch.tensor(SOURCE_RX, dtype=torch.long).repeat(2)
    bank = (x, labels, receivers)
    components = capture(model, x)
    logits = fixed_logits(model, components)
    assert torch.equal(logits, components['logits'])
    protected = protected_names(model)
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(F.cross_entropy(logits, labels),
                                   [named[n] for n in protected], allow_unused=True)
    assert gradients == (None, None, None, None)
    for field in ('physical', 'reference_raw', 'sequence_raw'):
        wrong = dict(components)
        wrong[field] = components[field] + .2
        _raises(lambda: fixed_logits(model, wrong))
    # Warm real parameter caches; evaluation must not replace their original objects.
    with torch.no_grad():
        model.forward_identity_only(x)
    before = copy.deepcopy(model.state_dict())
    rng = _rng_snapshot()
    flags = [(m, m.training) for m in model.modules()]
    caches = [(m, key, getattr(m, key)) for m in model.modules()
              for key in CACHE_ATTRIBUTES if hasattr(m, key)]
    names = list(named)
    pre = tuple(named[n].detach().clone() for n in names)
    model.get_submodule(HEAD).weight.grad = torch.ones_like(model.get_submodule(HEAD).weight)
    original_grad = model.get_submodule(HEAD).weight.grad
    z, summary = evaluate(model, names, pre, bank, protected)
    assert torch.equal(z, components['joint'].detach())
    head = model.get_submodule(HEAD)
    expected_logits = 30. * F.linear(F.normalize(z.double(), dim=1, eps=1e-4),
                                     F.normalize(head.weight.detach().double(), dim=1, eps=1e-4))
    expected = F.cross_entropy(expected_logits, labels, reduction='none')
    assert summary['mean_CE'] == float(expected.mean())
    assert all(summary['by_RX'][str(r)] == float(expected[receivers == r].mean()) for r in SOURCE_RX)
    assert summary['min_RX_CE'] == min(summary['by_RX'].values())
    assert summary['max_RX_CE'] == max(summary['by_RX'].values())
    protected_only = tuple(value + 4. if name in protected else value.clone()
                           for name, value in zip(names, pre))
    unchanged, fixed = evaluate(model, names, protected_only, bank, protected)
    assert torch.equal(unchanged, z) and fixed['by_RX'] == summary['by_RX']
    assert fixed['mean_CE'] == summary['mean_CE']
    feature_after = list(pre)
    feature_index = names.index(ENCODER + '.sequence_pool.output_norm.bias')
    feature_after[feature_index] = feature_after[feature_index].clone()
    feature_after[feature_index][0] += .25
    feature_z, feature_summary = evaluate(model, names, tuple(feature_after), bank, protected)
    assert not torch.equal(feature_z, z)
    assert feature_summary['protected_pre_step_readout']
    assert _equal(before, model.state_dict()) and _equal(rng, _rng_snapshot())
    assert model.get_submodule(HEAD).weight.grad is original_grad
    assert torch.equal(original_grad, torch.ones_like(original_grad))
    assert all(m.training == flag for m, flag in flags)
    assert all(getattr(m, key) is value for m, key, value in caches)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    model.get_submodule(HEAD).weight.grad = None
    # An original training-mode caller stays training; only the private copy evaluates.
    model.train()
    evaluate(model, names, pre, bank, protected)
    assert model.training and all(m.training for m in model.modules())
    model.eval()
    for role in ('U_s', 'V', 'target', 'query'):
        _raises(lambda: evaluate(model, names, pre, bank, protected, role=role))
    _raises(lambda: evaluate(model, names[:-1], pre, bank, protected))
    _raises(lambda: evaluate(model, names, pre, bank, protected[::-1]))
    _raises(lambda: evaluate(model, ['outside'], (torch.zeros(1),), bank, protected))
    _raises(lambda: evaluate(model, names, pre, (x, labels, receivers * 0), protected))
    wrong = list(pre)
    wrong[0] = torch.full_like(wrong[0], float('nan'))
    _raises(lambda: evaluate(model, names, tuple(wrong), bank, protected))
    # A sub-ULP nonzero ideal update vanishes in the actual FP32 parameter addition.
    theta = (torch.tensor([1., 2.], dtype=torch.float32),)
    ideal = (torch.tensor([2.**-26, -2.**-25], dtype=torch.float64),)
    risks = {r: (torch.tensor([float(i + 1), -float(i + 2)], dtype=torch.float64),)
             for i, r in enumerate(SOURCE_RX)}
    theta_before, ideal_before = theta[0].clone(), ideal[0].clone()
    realized = actual_displacement(theta, ideal, risks)
    assert torch.equal(realized['after_F'][0], theta[0])
    assert torch.count_nonzero(realized['actual_delta_FP64'][0]) == 0
    assert realized['summary']['swallowed_nonzero_ideal_coordinates'] == 2
    for r in SOURCE_RX:
        row = realized['summary']['by_RX'][str(r)]
        g = risks[r][0]
        actual = realized['actual_delta_FP64'][0]
        bound = float((g.abs() * (actual - ideal[0]).abs()).sum())
        assert row['ideal_directional_risk'] == float((g * ideal[0]).sum())
        assert row['actual_directional_risk'] == 0.
        assert row['measured_rounding_error_bound'] == bound
        assert abs(row['rounding_error_dot']) <= bound
    assert torch.equal(theta[0], theta_before) and torch.equal(ideal[0], ideal_before)
    # Feasible ideal direction can become positive actual risk because the two
    # coordinates have different parameter-addition rounding cells.
    different_ulps = (torch.tensor([4., 1.], dtype=torch.float32),)
    feasible = (torch.tensor([-.2 * 2.**-21, .6 * 2.**-23], dtype=torch.float64),)
    ones = {r: (torch.ones(2, dtype=torch.float64),) for r in SOURCE_RX}
    rounded = actual_displacement(different_ulps, feasible, ones)
    assert rounded['summary']['actual_positive_risk_RX'] == list(map(str, SOURCE_RX))
    for row in rounded['summary']['by_RX'].values():
        assert row['ideal_directional_risk'] < 0 < row['actual_directional_risk']
        assert abs(row['rounding_error_dot']) <= row['measured_rounding_error_bound']
        assert abs(row['dot_difference']) <= row['measured_rounding_error_bound']
    null_risk = actual_displacement(theta, ideal, {r: (None,) for r in SOURCE_RX})
    assert all(row['measured_rounding_error_bound'] == 0 for row in null_risk['summary']['by_RX'].values())
    _raises(lambda: actual_displacement(theta, (ideal[0].float(),), risks))
    _raises(lambda: actual_displacement(theta, (ideal[0][:1],), risks))
    _raises(lambda: actual_displacement(theta, ideal, {1: risks[1]}))
    _raises(lambda: actual_displacement(theta, ideal, {**risks, 2: risks[1]}))
    _raises(lambda: actual_displacement(theta, (torch.full_like(ideal[0], float('inf')),), risks))
    _raises(lambda: actual_displacement(theta, ideal, {r: (torch.ones(2),) for r in SOURCE_RX}))
    assert _equal(rng, _rng_snapshot())
    return dict(status='PASS', actual_SIG_TIME_CENTER_constructor=True,
                four_term_SELF_exact_and_protected_auxiliary_gradients_None=True,
                proposed_W_three_gains_restored_to_pre_step=True,
                raw_joint_and_double_per_RX_CE=True, genuine_feature_update_visible=True,
                live_state_flags_gradients_RNG_warm_caches_unchanged=True,
                sub_ULP_actual_addition_swallowed=True,
                feasible_ideal_but_positive_actual_risk_counterexample=True,
                each_RX_actual_dot_and_measured_conversion_bound=True,
                bad_shape_nonfinite_scope_and_role_rejected=True,
                actual_data_or_checkpoint_reads=0, actual_optimizer_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False, indent=2))
