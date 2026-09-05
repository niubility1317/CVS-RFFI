import torch
from SSDG.train_ssdg import _pair_diagnostic_step_logs, _aggregate_pair_diagnostic_logs
from cvsrffi.pair_reform_runtime import fixed_head_pair_diagnostics, grouped_pair_weight_diagnostics


def test_fixed_head_geometry_is_detached_and_uses_one_teacher_weight_set():
    clean = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    leo = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    result = fixed_head_pair_diagnostics(clean, leo, clean.detach(), torch.eye(2),
                                         torch.tensor([0, 0]), alpha=.5)
    assert result['fixed_head_clean_margin'].tolist() == [1., 1.]
    assert result['fixed_head_leo_margin'].tolist() == [1., -1.]
    assert result['fixed_head_classification_flip'].tolist() == [0., 1.]
    assert result['safe_radius_inside'].tolist() == [1., 0.]
    assert all(not value.requires_grad for value in result.values())


def test_grouped_weights_skip_unknown_rx_and_aggregate_exact_totals():
    ids = [(4, 0, 0, 0, 1), (4, 0, 0, 1, 2), (8, 0, 0, 0, 3), 'opaque']
    diag = grouped_pair_weight_diagnostics(ids, torch.tensor([0., .25, .99, .5]),
        torch.tensor([0., .25, .99, .5]), torch.tensor([0., .1, .4, .2]))
    assert diag['rx_unknown_count'] == 1
    assert diag['weight_groups']['rx_4/r_phys_bin_1']['feature_weight_sum'].item() == .25
    logs = _pair_diagnostic_step_logs(diag, 'u')
    total = _aggregate_pair_diagnostic_logs([logs, logs])
    assert total['train/pair_u/rx_unknown_count'] == 2
    assert total['train/pair_u/groups/rx_4/r_phys_bin_1/samples_count'] == 2
    assert abs(total['train/pair_u/groups/rx_4/r_phys_bin_1/pseudo_weight_mean'] - .1) < 1e-6


def test_pair_telemetry_weights_unequal_batches_and_sums_work():
    logs = [
        _pair_diagnostic_step_logs({'r_phys': torch.tensor([1., 0.]), 'q_cls': torch.tensor([.8, .2]),
            'feature_weight_sum': 1., 'pseudo_weight_sum': .8, 'physical_quality_unknown_count': 1,
            'teacher_views': 4, 'student_extra_views': 4, 'sampled_count': 1, 'pair_shift': .4}, 'l'),
        _pair_diagnostic_step_logs({'r_phys': torch.tensor([1.]), 'q_cls': torch.tensor([1.]),
            'feature_weight_sum': 1., 'pseudo_weight_sum': 1., 'physical_quality_unknown_count': 0,
            'teacher_views': 2, 'student_extra_views': 0, 'pair_shift': .1}, 'l'),
    ]
    out = _aggregate_pair_diagnostic_logs(logs)
    assert abs(out['train/pair_l/r_phys_mean'] - 2/3) < 1e-6
    assert abs(out['train/pair_l/pair_shift_mean'] - .3) < 1e-6
    assert out['train/pair_l/teacher_forward_samples_count'] == 6
    assert out['train/pair_l/student_extra_forward_samples_count'] == 4
    assert out['train/pair_l/physical_quality_unknown_count'] == 1
    assert out['train/pair_l/teacher_views_per_sample_mean'] == 2
    assert abs(out['train/pair_l/feature_weight_mean'] - 2/3) < 1e-6
    assert 'train/pair_l/pseudo_mean' not in out


def test_inactive_pair_telemetry_is_finite_zero_and_detached():
    inactive = _pair_diagnostic_step_logs({'teacher_views': 0, 'student_extra_views': 0}, 'u')
    assert _aggregate_pair_diagnostic_logs([inactive])['train/pair_u/observed_samples_count'] == 0
    active = _pair_diagnostic_step_logs({'r_phys': torch.ones(2, requires_grad=True)}, 'u')
    assert all(not torch.is_tensor(v) or not v.requires_grad for v in active.values())
