"""Synthetic transfer formulas, labels, graph and exposure; no native model."""
import copy
import json

import torch
from torch.nn import functional as F

from . import objective as api
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _bank():
    cells = [(label, receiver, day, side) for side in (0, 1) for label in range(6)
             for receiver in api.SOURCE_RX for day in api.SOURCE_DAYS]
    return dict(x=torch.zeros(180, 2, 256), y=torch.tensor([cell[0] for cell in cells]),
        rx=torch.tensor([cell[1] for cell in cells]), day=torch.tensor([cell[2] for cell in cells]),
        side=torch.tensor([cell[3] for cell in cells]),
        ids=[(1, receiver, day, 2 + side, 920000 + index)
             for index, (_, receiver, day, side) in enumerate(cells)])


def _summary(summary, mode):
    assert summary['matrix_count'] == 60 and summary['synthetic_views'] == 120
    assert summary['synthetic_decisions'] == summary['fixed_decision_denominator'] == 720
    assert summary['transported_views_this_call'] == 240 and summary['default_class_decisions_this_call'] == 1440
    assert summary['stop'] == .9 and summary['coefficient'] == .1 and summary['scale'] == 30.
    assert summary['label_application_consistent'] and summary['class_weight_gradient_stopped']
    assert summary['actual_three_gains_follow_joint_graph'] and not summary['common_RX_additive_offset_loss_invariance_claimed']
    assert summary['near_eps_count'] == 0 and summary['common_means_RMS'] > 0.
    assert summary['transport_difference_RMS'] > 0.
    assert summary['default_cos_FP64_FP32_max_abs_logit_difference'] >= 0.
    exposure = summary['exposures']
    assert exposure['per_packet_min'] == exposure['per_packet_max'] == 4
    assert exposure['per_packet_classification_min'] == exposure['per_packet_classification_max'] == 4
    assert exposure['anchor_endpoints'] == exposure['donor_endpoints'] == 360
    assert exposure['total_endpoint_appearances'] == exposure['total_synthetic_classification_appearances'] == 720
    assert exposure['by_RX'] == {str(receiver): dict(anchor=72, donor=72, synthetic=144) for receiver in api.SOURCE_RX}
    assert exposure['by_day'] == {str(day): dict(anchor=120, donor=120, synthetic=240) for day in api.SOURCE_DAYS}
    assert exposure['by_TX'] == {str(label): dict(anchor=60, donor=60, synthetic=120) for label in range(6)}
    assert exposure['by_side'] == {'0':dict(endpoint=360, synthetic=360), '1':dict(endpoint=360, synthetic=360)}
    assert all(row['decision_count'] == 144 for row in summary['by_anchor_RX'].values())
    assert sum(row['active_decisions'] for row in summary['by_anchor_RX'].values()) == summary['active_decisions']
    assert sum(row['errors_total'] for row in summary['by_anchor_RX'].values()) == summary['errors_total']
    assert abs(sum(row['mean_CE'] for row in summary['by_anchor_RX'].values()) / 5 - summary['mean_CE']) < 1e-12
    assert summary['unique_matrices'] == (15 if mode == 'SAME_RX' else 60)
    json.dumps(summary, allow_nan=False)


def check():
    torch.set_num_threads(2)
    bank = _bank(); generator = torch.Generator().manual_seed(153020)
    base = torch.randn(180, 160, dtype=torch.float64, generator=generator, requires_grad=True)
    physical = torch.randn(180, 160, dtype=torch.float64, generator=generator)
    reference = torch.randn(180, 160, dtype=torch.float64, generator=generator)
    sequence = torch.randn(180, 160, dtype=torch.float64, generator=generator)
    gains = tuple(torch.nn.Parameter(torch.full((160,), value, dtype=torch.float64)) for value in (.1, .15, .2))
    joint = ((base + gains[0].tanh() * physical) + gains[1].tanh() * reference) + .2 * gains[2].tanh() * sequence
    weight = torch.nn.Parameter(torch.randn(6, 160, dtype=torch.float64, generator=generator))
    before = copy.deepcopy(bank); joint_before = joint.detach().clone(); weight_before = weight.detach().clone(); rng = _rng_snapshot()
    captured = []; original = api._decisions
    def record(views, labels, W):
        captured.append((views.detach().clone(), labels.detach().clone()))
        return original(views, labels, W)
    api._decisions = record
    try:
        values, summaries = api.losses(joint, bank, weight)
    finally:
        api._decisions = original
    assert tuple(values) == api.MODES and tuple(summaries) == api.MODES and len(captured) == 2
    gradients = {}
    for mode in api.MODES:
        _summary(summaries[mode], mode)
        gradient, W_gradient = torch.autograd.grad(values[mode], (base, weight), allow_unused=True, retain_graph=True)
        assert W_gradient is None and torch.isfinite(gradient).all()
        if mode == 'ZERO':
            assert float(values[mode].detach()) == 0. and torch.count_nonzero(gradient) == 0
        else:
            assert float(values[mode].detach()) > 0. and float(gradient.norm()) > 0.
            gradients[mode] = gradient
    gain_gradients = torch.autograd.grad(values['CROSS_RX'], gains, retain_graph=True)
    assert all(torch.isfinite(gradient).all() and float(gradient.norm()) > 0. for gradient in gain_gradients)
    # Verify both view formulas and true labels independently for every pair.
    for control, mode in enumerate(('SAME_RX', 'CROSS_RX')):
        expected_views, expected_labels = [], []
        for pair in summaries[mode]['actual_pairs']:
            def indices(receiver, side):
                return sorted([i for i in range(180) if int(bank['rx'][i]) == receiver
                    and int(bank['day'][i]) == pair['day'] and int(bank['side'][i]) == side], key=lambda i: int(bank['y'][i]))
            ia, ib = indices(pair['anchor_RX'], 0), indices(pair['donor_RX'], 1)
            a, b = joint.detach()[ia], joint.detach()[ib]
            expected_views.extend((b - b.mean(0) + a.mean(0), a - a.mean(0) + b.mean(0)))
            expected_labels.extend((bank['y'][ib], bank['y'][ia]))
            assert pair['label_source_sides'] == [1, 0]
        expected_views, expected_labels = torch.cat(expected_views), torch.cat(expected_labels)
        assert torch.equal(captured[control][0], expected_views) and torch.equal(captured[control][1], expected_labels)
        logits = 30. * F.linear(F.normalize(expected_views, dim=1, eps=1e-4), F.normalize(weight.detach(), dim=1, eps=1e-4))
        log_true = logits.log_softmax(1).gather(1, expected_labels[:, None]).squeeze(1)
        expected_loss = .1 * (torch.tensor(.9, dtype=torch.float64).log() - log_true).clamp_min(0.).sum() / 720
        assert torch.allclose(values[mode].detach(), expected_loss, atol=1e-14, rtol=1e-14)
    # True-class renaming requires the same permutation of original W rows.
    rename = torch.tensor([5, 2, 4, 0, 3, 1]); renamed = copy.deepcopy(bank); renamed['y'] = rename[bank['y']]
    renamed_W = torch.empty_like(weight.detach()); renamed_W[rename] = weight.detach()
    renamed_joint = joint.detach().clone().requires_grad_(True)
    renamed_values, _ = api.losses(renamed_joint, renamed, renamed_W)
    permutation = torch.randperm(180, generator=generator)
    shuffled = {key: value[permutation].clone() if isinstance(value, torch.Tensor)
                else [value[index] for index in permutation.tolist()] for key, value in bank.items()}
    shuffled_joint = joint.detach()[permutation].clone().requires_grad_(True)
    shuffled_values, _ = api.losses(shuffled_joint, shuffled, weight)
    plain_joint = joint.detach().clone().requires_grad_(True)
    plain_values, _ = api.losses(plain_joint, bank, weight)
    for mode in api.MODES:
        assert torch.allclose(values[mode].detach(), renamed_values[mode].detach(), atol=1e-12, rtol=1e-12)
        assert torch.equal(values[mode].detach(), shuffled_values[mode].detach())
        plain_gradient = torch.autograd.grad(plain_values[mode], plain_joint, retain_graph=True)[0]
        renamed_gradient = torch.autograd.grad(renamed_values[mode], renamed_joint, retain_graph=True)[0]
        shuffled_gradient = torch.autograd.grad(shuffled_values[mode], shuffled_joint, retain_graph=True)[0]
        assert torch.allclose(plain_gradient, renamed_gradient, atol=1e-12, rtol=1e-10)
        assert torch.equal(plain_gradient, shuffled_gradient[torch.argsort(permutation)])
    # Global positive scale preserves cosine decisions away from the epsilon floor.
    scaled_values, scaled_summary = api.losses(joint.detach() * 3.25, bank, weight)
    for mode in api.MODES:
        assert torch.allclose(values[mode].detach(), scaled_values[mode], atol=1e-12, rtol=1e-12)
        assert scaled_summary[mode]['near_eps_count'] == 0
    # A common RX offset is deliberately *not* an invariance of this decision task.
    offset = joint.detach().clone(); offset[bank['rx'] == 1] += 2. * F.normalize(weight.detach()[0], dim=0)
    offset_values, _ = api.losses(offset, bank, weight)
    assert abs(float(offset_values['CROSS_RX']) - float(values['CROSS_RX'].detach())) > 1e-6
    # When every true-class probability exceeds .9, both stopped losses/gradients are exact0.
    perfect_W = torch.zeros(6, 160, dtype=torch.float64); perfect_W[:, :6] = torch.eye(6, dtype=torch.float64)
    perfect_joint = perfect_W[bank['y']].clone().requires_grad_(True)
    stopped_values, stopped_summaries = api.losses(perfect_joint, bank, perfect_W)
    for mode in api.MODES:
        assert float(stopped_values[mode].detach()) == 0. and stopped_summaries[mode]['active_decisions'] == 0
        assert torch.count_nonzero(torch.autograd.grad(stopped_values[mode], perfect_joint, retain_graph=True)[0]) == 0
    float_values, _ = api.losses(joint.detach().float().requires_grad_(True), bank, weight.detach().float())
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in float_values.values())
    negatives = 0
    def reject(data=bank, W=weight, role='L_s'):
        nonlocal negatives
        try: api.losses(joint, data, W, source_role=role)
        except ValueError: negatives += 1
        else: raise AssertionError('Invalid source permission or class-W input was accepted')
    for role in ('U_s', 'V', 'target'): reject(data=dict(bank, y=object()), role=role)
    for hint in ({'hide': True}, {'source_role': 'U_s'}, {'tx_label_visible': False}): reject(data=dict(bank, y=object(), **hint))
    reject(W=weight[:5]); reject(W=weight[:, :159]); reject(W=torch.full_like(weight, float('nan')))
    assert _equal(before, bank) and torch.equal(joint_before, joint.detach()) and torch.equal(weight_before, weight.detach())
    assert _equal(rng, _rng_snapshot()) and base.grad is None and weight.grad is None and all(gain.grad is None for gain in gains)
    return dict(status='PASS', source_matrices_per_control=60, SAME_unique15_repeat4=True,
        synthetic_views_per_control=120, true_class_decisions_per_control=720,
        endpoint_and_classification_exposure_per_packet=4, RX_day_TX_side_exposure_matched=True,
        both_transfer_formulas_and_origin_labels_verified=True,
        ZERO_exact_connected_zero=True, W_gradient_None=True, joint_both_sides_and_three_gain_gradients=True,
        label_W_row_rename_and_packet_permutation_invariance=True,
        positive_global_scale_away_from_eps_invariant=True, common_RX_offset_not_loss_invariant=True,
        probability_stop_exact_zero=True, FP64_analysis_FP32_formula_difference_reported=True,
        input_bank_features_weight_RNG_unchanged=True, focused_negative_checks=negatives,
        data_reads=0, checkpoint_reads=0, native_constructors=0, actual_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
