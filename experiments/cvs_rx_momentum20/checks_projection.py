"""Independent synthetic geometry checks for the historical RX projector."""
import json

import torch

from .projection import project_history
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected FP64 finite/shape validation failure')


def check():
    torch.set_num_threads(2)
    history = torch.tensor([3., 4., 5., 6., 7., 8.], dtype=torch.float64)
    direction = torch.tensor([1., 2., 0., 0., 0., 0.], dtype=torch.float64)
    multipliers = torch.tensor([-2., -1., 0., 1., 2.], dtype=torch.float64)
    contrasts = multipliers[:, None] * direction[None, :]
    history_before, contrasts_before = history.clone(), contrasts.clone()
    versions = history._version, contrasts._version
    rng = _rng_snapshot()
    # Independent exact Euclidean projection onto span{(1,2,0,...)}.
    expected = history - ((3. + 2. * 4.) / 5.) * direction
    filtered, rank1 = project_history(history, contrasts)
    torch.testing.assert_close(filtered, expected, rtol=1e-14, atol=1e-14)
    assert rank1['rank'] == 1 and rank1['max_normalized_inner_product_residual'] < 1e-14
    assert filtered.dtype == torch.float64 and filtered.device == history.device
    # The five centered rows span precisely the first four coordinate axes.
    rank4_contrasts = torch.zeros((5, 6), dtype=torch.float64)
    rank4_contrasts[:4, :4] = torch.eye(4, dtype=torch.float64)
    rank4_expected = torch.tensor([0., 0., 0., 0., 7., 8.], dtype=torch.float64)
    rank4_filtered, rank4 = project_history(history, rank4_contrasts)
    torch.testing.assert_close(rank4_filtered, rank4_expected, rtol=1e-14, atol=1e-14)
    assert rank4['rank'] == 4 and rank4['max_normalized_inner_product_residual'] < 1e-14
    # Reordering RX rows and common positive row scaling preserve this span.
    permutation = [4, 0, 3, 1, 2]
    errors = []
    for transformed in (rank4_contrasts[permutation], rank4_contrasts * 1000.,
                        rank4_contrasts * .001,
                        rank4_contrasts + torch.tensor([.7, -.4, .2, .9, .3, -.2], dtype=torch.float64)):
        value, summary = project_history(history, transformed)
        torch.testing.assert_close(value, rank4_filtered, rtol=1e-13, atol=1e-13)
        assert summary['rank'] == 4
        errors.append(float((value - rank4_filtered).abs().max()))
    orthogonal = torch.tensor([0., 0., 1., 2., 3., 4.], dtype=torch.float64)
    unchanged, orthogonal_summary = project_history(orthogonal, contrasts)
    assert torch.equal(unchanged, orthogonal) and orthogonal_summary['removed_norm'] == 0.
    in_span, span_summary = project_history(3. * direction, contrasts)
    torch.testing.assert_close(in_span, torch.zeros_like(in_span), rtol=0., atol=1e-14)
    assert abs(span_summary['removed_ratio'] - 1.) < 1e-14
    for zeros in (torch.zeros_like(contrasts), torch.ones_like(contrasts)):
        value, summary = project_history(history, zeros)
        assert torch.equal(value, history) and summary['rank'] == 0
        assert summary['removed_norm'] == 0. and summary['removed_ratio'] == 0.
    zero_value, zero_summary = project_history(torch.zeros_like(history), contrasts)
    assert torch.equal(zero_value, torch.zeros_like(history)) and zero_summary['removed_ratio'] == 0.
    # Known orthogonal zero-mean RX columns with eigenvalues1 and1e-12:
    # the fixed cutoff1e-10 keeps only the first, without a parameter scan.
    cutoff_contrasts = torch.zeros((5, 6), dtype=torch.float64)
    cutoff_contrasts[:, 0] = torch.tensor([1., -1., 0., 0., 0.], dtype=torch.float64) / (2. ** .5)
    cutoff_contrasts[:, 1] = 1e-6 * torch.tensor([1., 1., -2., 0., 0.], dtype=torch.float64) / (6. ** .5)
    cutoff_value, cutoff_summary = project_history(history, cutoff_contrasts)
    torch.testing.assert_close(cutoff_value, torch.tensor([0., 4., 5., 6., 7., 8.], dtype=torch.float64), rtol=1e-14, atol=1e-14)
    assert cutoff_summary['rank'] == 1 and abs(cutoff_summary['cutoff'] - 1e-10) < 1e-24
    # The absolute floor is part of the fixed rule, so scales that cross it
    # are explicitly not claimed to have rank-invariant output.
    tiny_value, tiny_summary = project_history(history, contrasts * 1e-20)
    assert tiny_summary['rank'] == 0 and tiny_summary['cutoff'] == 1e-30
    assert torch.equal(tiny_value, history)
    attached_history = history.clone().requires_grad_(True)
    attached_contrasts = contrasts.clone().requires_grad_(True)
    detached, _ = project_history(attached_history, attached_contrasts)
    assert not detached.requires_grad and attached_history.grad is None and attached_contrasts.grad is None
    negatives = 0
    for call in (
        lambda: project_history(history.float(), contrasts),
        lambda: project_history(history, contrasts.float()),
        lambda: project_history(history[None, :], contrasts),
        lambda: project_history(history, contrasts[:4]),
        lambda: project_history(history, contrasts[:, :5]),
        lambda: project_history(history, contrasts.reshape(-1)),
        lambda: project_history(history[:0], contrasts[:, :0]),
        lambda: project_history([3., 4.], contrasts),
        lambda: project_history(history, contrasts.to('meta')),
        lambda: project_history(history * float('nan'), contrasts),
        lambda: project_history(history, contrasts + float('inf')),
    ):
        _reject(call)
        negatives += 1
    assert torch.equal(history, history_before) and torch.equal(contrasts, contrasts_before)
    assert versions == (history._version, contrasts._version) and _equal(rng, _rng_snapshot())
    json.dumps(rank1, allow_nan=False)
    json.dumps(rank4, allow_nan=False)
    return dict(status='PASS', device='cpu', torch_version=torch.__version__,
                rank1=rank1['rank'], rank4=rank4['rank'],
                rank1_max_normalized_residual=rank1['max_normalized_inner_product_residual'],
                rank4_max_normalized_residual=rank4['max_normalized_inner_product_residual'],
                permutation_scale_common_offset_max_error=max(errors),
                exact_zero_contrast_noop=True, fixed_cutoff_and_absolute_floor=True,
                negative_checks=negatives, input_versions_and_RNG_unchanged=True,
                no_data_model_or_optimizer_state=True)


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, allow_nan=False, indent=2))
