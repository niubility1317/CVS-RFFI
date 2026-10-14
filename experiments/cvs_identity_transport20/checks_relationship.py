"""Synthetic true-L grouping/exposure/gradient invariance; no model or IQ data."""
from collections import Counter
import copy
import json

import torch

from . import relationship as api
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _fixture():
    cells = [(label, receiver, day, side) for side in (0, 1) for label in range(6)
             for receiver in api.SOURCE_RX for day in api.SOURCE_DAYS]
    bank = dict(x=torch.zeros(180, 2, 256), y=torch.tensor([cell[0] for cell in cells]),
                rx=torch.tensor([cell[1] for cell in cells]), day=torch.tensor([cell[2] for cell in cells]),
                side=torch.tensor([cell[3] for cell in cells]),
                ids=[(1, receiver, day, side, 900000 + index)
                     for index, (_, receiver, day, side) in enumerate(cells)])
    generator = torch.Generator().manual_seed(152020)
    joint = torch.randn(180, 160, dtype=torch.float64, generator=generator).requires_grad_(True)
    return joint, bank


def _summary_check(summary, mode):
    assert summary['matrix_count'] == 60 and summary['fixed_matrix_denominator'] == 60
    assert summary['pairwise_cost_entries'] == 2160 and summary['source_group_count'] == 30
    assert summary['packets_per_RX'] == 36
    assert summary['diagonal_row_count'] == 360 and summary['diagonal_column_count'] == 360
    assert 0 <= summary['active_row_count'] <= 360 and 0 <= summary['active_column_count'] <= 360
    assert 0 <= summary['row_argmax_error_count'] <= 360 and 0 <= summary['column_argmax_error_count'] <= 360
    assert summary['active_rows_and_columns'] == summary['active_row_count'] + summary['active_column_count']
    assert summary['matching_errors_total'] == summary['row_argmax_error_count'] + summary['column_argmax_error_count']
    assert abs(summary['mean_correct_matching_mass'] - (summary['correct_row_mass']['mean']
               + summary['correct_column_mass']['mean']) / 2) < 1e-15
    assert summary['max_marginal_residual'] == summary['marginal_residual_max']
    assert summary['approximate_plan_matrix_count'] == summary['approximate_plan_count']
    assert summary['correct_row_mass']['count'] == 360 and summary['correct_column_mass']['count'] == 360
    assert summary['cost_RMS']['count'] == 60 and summary['raw_D_norm']['count'] == 60
    assert summary['matching_evaluations_this_call'] == 180 and summary['all_three_controls_computed']
    assert not summary['W_or_new_head_used'] and summary['IQ_forward_count'] == 0
    exposures = summary['exposures']
    assert exposures['unique_packets'] == 180 and exposures['anchor_endpoints'] == 360
    assert exposures['donor_endpoints'] == 360 and exposures['total_endpoint_appearances'] == 720
    assert exposures['per_packet_min'] == 4 and exposures['per_packet_max'] == 4
    assert exposures['by_RX'] == {str(receiver): {'anchor': 72, 'donor': 72} for receiver in api.SOURCE_RX}
    assert exposures['by_day'] == {str(day): {'anchor': 120, 'donor': 120} for day in api.SOURCE_DAYS}
    assert exposures['by_TX'] == {str(label): {'anchor': 60, 'donor': 60} for label in range(6)}
    assert exposures['by_side'] == {'0': 360, '1': 360}
    pairs = Counter((row['anchor_RX'], row['donor_RX'], row['day']) for row in summary['actual_pairs'])
    if mode == 'SAME_BI':
        assert len(pairs) == 15 and set(pairs.values()) == {4}
        assert summary['unique_matrix_count'] == 15 and summary['ordered_RX_pair_count'] == 5
        assert all(anchor == donor for anchor, donor, _ in pairs)
    else:
        assert len(pairs) == 60 and set(pairs.values()) == {1}
        assert summary['unique_matrix_count'] == 60 and summary['ordered_RX_pair_count'] == 20
        assert all(anchor != donor for anchor, donor, _ in pairs)
    if mode == 'CROSS_ROW':
        assert summary['marginal_residual_max'] is None
    else:
        assert summary['marginal_residual_max'] >= 0.
    assert all('ids' not in row and 'base_index' not in row for row in summary['matrix_diagnostics'])
    json.dumps(summary, allow_nan=False)


def check():
    torch.set_num_threads(2)
    joint, bank = _fixture()
    before = copy.deepcopy(bank); joint_before = joint.detach().clone(); rng = _rng_snapshot()
    calls = []; original_match = api.match
    def record(a, b, mode, labels_a, labels_b):
        calls.append(mode)
        return original_match(a, b, mode, labels_a, labels_b)
    api.match = record
    try:
        values, summaries = api.losses(joint, bank)
    finally:
        api.match = original_match
    assert Counter(calls) == {'BI': 120, 'ROW': 60}, 'SAME repetition did not execute the equal matching budget'
    assert tuple(values) == api.MODES and tuple(summaries) == api.MODES
    assert values['ZERO'].shape == () and float(values['ZERO'].detach()) == 0.
    W = torch.nn.Parameter(torch.zeros(6, 160, dtype=torch.float64))
    zero_gradient, zero_w = torch.autograd.grad(values['ZERO'], (joint, W), allow_unused=True)
    assert torch.count_nonzero(zero_gradient) == 0 and zero_w is None
    gradients = {}
    for mode in api.MODES:
        _summary_check(summaries[mode], mode)
        if mode != 'ZERO':
            assert float(values[mode].detach()) > 0.
            gradient, w_gradient = torch.autograd.grad(values[mode], (joint, W), allow_unused=True, retain_graph=True)
            assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0. and w_gradient is None
            gradients[mode] = gradient
    # Global class renaming changes no matching semantics, loss or physical gradient.
    rename = torch.tensor([4, 0, 5, 2, 1, 3])
    renamed_bank = copy.deepcopy(bank); renamed_bank['y'] = rename[bank['y']]
    renamed_joint = joint.detach().clone().requires_grad_(True)
    renamed_values, renamed_summaries = api.losses(renamed_joint, renamed_bank)
    for mode in api.MODES:
        assert torch.allclose(values[mode].detach(), renamed_values[mode].detach(), atol=1e-11, rtol=1e-11)
        if mode != 'ZERO':
            gradient = torch.autograd.grad(renamed_values[mode], renamed_joint, retain_graph=True)[0]
            assert torch.allclose(gradient, gradients[mode], atol=1e-10, rtol=1e-9)
        assert renamed_summaries[mode]['active_row_count'] == summaries[mode]['active_row_count']
        assert renamed_summaries[mode]['row_argmax_error_count'] == summaries[mode]['row_argmax_error_count']
    # Arbitrary packet ordering preserves canonical matching and gradients.
    generator = torch.Generator().manual_seed(152021)
    permutation = torch.randperm(180, generator=generator)
    permuted_bank = {key: value[permutation].clone() if isinstance(value, torch.Tensor)
                     else [value[index] for index in permutation.tolist()] for key, value in bank.items()}
    permuted_joint = joint.detach()[permutation].clone().requires_grad_(True)
    permuted_values, _ = api.losses(permuted_joint, permuted_bank)
    inverse = torch.argsort(permutation)
    for mode in api.MODES:
        assert torch.equal(values[mode].detach(), permuted_values[mode].detach())
        if mode != 'ZERO':
            gradient = torch.autograd.grad(permuted_values[mode], permuted_joint, retain_graph=True)[0]
            assert torch.allclose(gradient[inverse], gradients[mode], atol=1e-12, rtol=1e-12)
    # Changing/deleting an unrelated W cannot affect the relation inputs or objective.
    with torch.no_grad(): W.fill_(8.)
    # The next native source call supplies FP32 joint; preserve its graph/dtype.
    float_joint = joint.detach().float().requires_grad_(True)
    float_values, _ = api.losses(float_joint, bank)
    for mode in api.MODES:
        assert float_values[mode].dtype == torch.float32 and torch.isfinite(float_values[mode])
        gradient = torch.autograd.grad(float_values[mode], float_joint, retain_graph=True)[0]
        assert gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
        assert (torch.count_nonzero(gradient) == 0) if mode == 'ZERO' else (float(gradient.norm()) > 0.)
    assert _equal(before, bank) and torch.equal(joint.detach(), joint_before)
    assert _equal(rng, _rng_snapshot()) and joint.grad is None and W.grad is None

    negative_checks = 0
    def reject(value=joint, data=bank, role='L_s'):
        nonlocal negative_checks
        try:
            api.losses(value, data, source_role=role)
        except ValueError:
            negative_checks += 1
        else:
            raise AssertionError('Wrong source relationship permission/shape/population was accepted')
    for role in ('U_s', 'V', 'V_proxy', 'target', 'query'):
        reject(role=role)
    # y is deliberately not a tensor: role rejection must precede inspection.
    for flag in ({'hide': True}, {'role': 'U_s'}, {'source_role': 'V'}, {'tx_label_visible': False}):
        reject(data=dict(bank, y=object(), **flag))
    reject(value=joint[:179])
    reject(value=joint[:, :159])
    reject(value=joint.half())
    reject(value=torch.full_like(joint, float('nan')))
    reject(data=dict(bank, x=bank['x'][:179]))
    reject(data=dict(bank, x=torch.full_like(bank['x'], float('inf'))))
    reject(data=dict(bank, y=bank['y'].float()))
    reject(data=dict(bank, y=torch.full_like(bank['y'], -1)))
    reject(data=dict(bank, rx=torch.full_like(bank['rx'], 2)))
    reject(data=dict(bank, day=torch.zeros_like(bank['day'])))
    reject(data=dict(bank, side=torch.zeros_like(bank['side'])))
    wrong = copy.deepcopy(bank); wrong['y'][0] = 1; reject(data=wrong)
    wrong = copy.deepcopy(bank); wrong['ids'][1] = wrong['ids'][0]; reject(data=wrong)
    wrong = copy.deepcopy(bank); wrong['ids'][1] = (*wrong['ids'][1][:4], wrong['ids'][0][4]); reject(data=wrong)
    wrong = copy.deepcopy(bank); identity = wrong['ids'][0]; wrong['ids'][0] = (identity[0], 3, *identity[2:]); reject(data=wrong)
    wrong = copy.deepcopy(bank); identity = wrong['ids'][0]; wrong['ids'][0] = (identity[0], identity[1], 2, *identity[3:]); reject(data=wrong)
    wrong = copy.deepcopy(bank); identity = wrong['ids'][0]; wrong['ids'][0] = (False, *identity[1:]); reject(data=wrong)
    wrong = copy.deepcopy(bank); wrong['ids'] = wrong['ids'][:179]; reject(data=wrong)
    reject(data=dict(bank, tx_i=bank['y']))
    assert _equal(before, bank) and torch.equal(joint.detach(), joint_before) and _equal(rng, _rng_snapshot())
    assert api.match is original_match
    return dict(status='PASS', matrices_per_control=60, SAME_unique_matrices=15, SAME_repeat_count=4,
        CROSS_unique_matrices=60, actual_matching_calls=180,
        source_groups=30, packets_per_RX=36, per_packet_endpoint_exposure=4,
        same_cross_endpoint_RX_day_class_side_balanced=True,
        class_rename_and_sample_permutation_loss_gradient_invariant=True,
        ZERO_exact_connected_zero=True, active_control_feature_gradients_finite_nonzero=True,
        native_FP32_and_FP64_inputs_supported=True,
        W_independent_no_new_head=True, input_bank_joint_RNG_unchanged=True,
        illegal_roles_rejected_before_TX=True, negative_checks=negative_checks,
        data_reads=0, checkpoint_reads=0, model_constructors=0, actual_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
