"""Focused new ROW-only assembly checks; reuse already validated math/bank."""
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
        ids=[(1, receiver, day, 2 + side, 910000 + index)
             for index, (_, receiver, day, side) in enumerate(cells)])
    generator = torch.Generator().manual_seed(152320)
    joint = torch.randn(180, 160, generator=generator).requires_grad_(True)
    return joint, bank


def _summary(summary, mode):
    assert summary['matrix_count'] == 60 and summary['fixed_matrix_denominator'] == 60
    assert summary['source_group_count'] == 30 and summary['packets_per_RX'] == 36
    assert summary['diagonal_row_count'] == 360 and summary['diagonal_column_count'] == 360
    assert summary['matching_evaluations_this_call'] == 120 and summary['ROW_matching_evaluations'] == 120
    assert summary['BI_matching_evaluations'] == 0 and summary['sinkhorn_iterations_executed'] == 0
    assert summary['computed_controls'] == list(api.MODES) and summary['all_two_ROW_controls_computed']
    assert summary['max_marginal_residual'] is None and summary['approximate_plan_matrix_count'] == 0
    assert summary['active_rows_and_columns'] == summary['active_row_count'] + summary['active_column_count']
    assert summary['matching_errors_total'] == summary['row_argmax_error_count'] + summary['column_argmax_error_count']
    assert abs(summary['mean_correct_matching_mass'] - (summary['correct_row_mass']['mean']
        + summary['correct_column_mass']['mean']) / 2) < 1e-15
    exposure = summary['exposures']
    assert exposure['per_packet_min'] == exposure['per_packet_max'] == 4
    assert exposure['anchor_endpoints'] == exposure['donor_endpoints'] == 360
    assert exposure['by_RX'] == {str(receiver): {'anchor': 72, 'donor': 72} for receiver in api.SOURCE_RX}
    assert exposure['by_day'] == {str(day): {'anchor': 120, 'donor': 120} for day in api.SOURCE_DAYS}
    assert exposure['by_TX'] == {str(label): {'anchor': 60, 'donor': 60} for label in range(6)}
    pairs = Counter((row['anchor_RX'], row['donor_RX'], row['day']) for row in summary['actual_pairs'])
    if mode == 'SAME_ROW':
        assert len(pairs) == 15 and set(pairs.values()) == {4} and summary['unique_matrix_count'] == 15
        assert all(anchor == donor for anchor, donor, _ in pairs)
        assert summary['unique_matrix_disclosure'] == 'SAME_ROW has15 unique matrices repeated4'
    else:
        assert len(pairs) == 60 and set(pairs.values()) == {1} and summary['unique_matrix_count'] == 60
        assert all(anchor != donor for anchor, donor, _ in pairs)
    assert all(row['matching']['mode'] == 'ROW' and row['matching']['iterations'] == 0
               for row in summary['matrix_diagnostics'])
    assert 'SAME_BI' not in json.dumps(summary)
    json.dumps(summary, allow_nan=False)


def check():
    torch.set_num_threads(2)
    joint, bank = _fixture(); before = copy.deepcopy(bank); value_before = joint.detach().clone(); rng = _rng_snapshot()
    calls = []; original = api.match
    def row_only(a, b, mode, labels_a, labels_b):
        assert mode == 'ROW', 'Closed BI algorithm was executed by the new ROW-only stage'
        calls.append(mode)
        return original(a, b, mode, labels_a, labels_b)
    api.match = row_only
    try:
        values, summaries = api.losses(joint, bank)
        assert Counter(calls) == {'ROW': 120}
        assert tuple(values) == api.MODES and tuple(summaries) == api.MODES
        gradients = {}
        for mode in api.MODES:
            _summary(summaries[mode], mode)
            gradients[mode] = torch.autograd.grad(values[mode], joint, retain_graph=True)[0]
            assert torch.isfinite(gradients[mode]).all()
            if mode == 'ZERO':
                assert float(values[mode].detach()) == 0. and torch.count_nonzero(gradients[mode]) == 0
            else:
                assert float(values[mode].detach()) > 0. and float(gradients[mode].norm()) > 0.
        renamed = copy.deepcopy(bank); renamed['y'] = torch.tensor([2, 5, 0, 4, 1, 3])[bank['y']]
        renamed_joint = joint.detach().clone().requires_grad_(True)
        renamed_values, _ = api.losses(renamed_joint, renamed)
        generator = torch.Generator().manual_seed(152321); permutation = torch.randperm(180, generator=generator)
        shuffled = {key: value[permutation].clone() if isinstance(value, torch.Tensor)
                    else [value[index] for index in permutation.tolist()] for key, value in bank.items()}
        shuffled_joint = joint.detach()[permutation].clone().requires_grad_(True)
        shuffled_values, _ = api.losses(shuffled_joint, shuffled)
        for mode in api.MODES:
            assert torch.allclose(values[mode].detach(), renamed_values[mode].detach(), atol=1e-7, rtol=1e-6)
            assert torch.equal(values[mode].detach(), shuffled_values[mode].detach())
            renamed_gradient = torch.autograd.grad(renamed_values[mode], renamed_joint, retain_graph=True)[0]
            shuffled_gradient = torch.autograd.grad(shuffled_values[mode], shuffled_joint, retain_graph=True)[0]
            assert torch.allclose(gradients[mode], renamed_gradient, atol=1e-8, rtol=1e-6)
            assert torch.equal(gradients[mode], shuffled_gradient[torch.argsort(permutation)])
        assert len(calls) == 360 and set(calls) == {'ROW'}
        negatives = 0
        for role in ('U_s', 'V', 'target'):
            try: api.losses(joint, dict(bank, y=object()), source_role=role)
            except ValueError: negatives += 1
            else: raise AssertionError('Non-L role inspected hidden TX or entered matching')
        for hint in ({'hide':True}, {'source_role':'U_s'}, {'tx_label_visible':False}):
            try: api.losses(joint, dict(bank, y=object(), **hint))
            except ValueError: negatives += 1
            else: raise AssertionError('Hidden/non-L bank entered correspondence')
        assert len(calls) == 360, 'Rejected role performed matching'
    finally:
        api.match = original
    assert _equal(before, bank) and torch.equal(value_before, joint.detach()) and _equal(rng, _rng_snapshot())
    assert joint.grad is None
    return dict(status='PASS', ROW_matching_calls_per_call=120, BI_matching_calls=0,
        Sinkhorn_iterations=0, SAME_ROW_unique_matrices=15, SAME_ROW_repetition=4, CROSS_ROW_unique_matrices=60,
        endpoint_exposure_per_packet=4, RX_day_TX_side_exposure_matched=True,
        source_permission_rejected_before_TX=True, source_negative_checks=negatives,
        class_rename_sample_permutation_loss_gradient_invariance=True,
        ZERO_exact_connected_zero=True, active_control_gradients_finite_nonzero=True,
        input_bank_joint_RNG_unchanged=True, data_reads=0, checkpoint_reads=0,
        native_model_constructors=0, actual_updates=0)


if __name__ == '__main__':
    print(json.dumps(check(), allow_nan=False))
