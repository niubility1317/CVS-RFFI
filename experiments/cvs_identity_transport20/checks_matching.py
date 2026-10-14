"""Focused independent synthetic tests of stopped bilateral identity matching."""
import json
import math

import torch
from torch.nn import functional as F

from .matching import match, _relative_cost
from experiments.cvs_u_schedule50.counterfactual import _equal, _rng_snapshot


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected matching shape/class/finite rejection')


def check():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(41703)
    a = torch.randn(6, 160, dtype=torch.float64, generator=generator).requires_grad_()
    b = torch.randn(6, 160, dtype=torch.float64, generator=generator).requires_grad_()
    labels = list(range(6))
    av, bv = a.detach().clone(), b.detach().clone()
    versions = a._version, b._version
    rng = _rng_snapshot()
    shift_a = torch.randn(160, dtype=torch.float64, generator=generator) * 3.
    shift_b = torch.randn(160, dtype=torch.float64, generator=generator) * 2.
    cost = _relative_cost(a, b)[0]
    shifted_cost = _relative_cost(a + shift_a, b + shift_b)[0]
    scaled_cost = _relative_cost(a * 7., b * 7.)[0]
    torch.testing.assert_close(cost, shifted_cost, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cost, scaled_cost, rtol=1e-12, atol=1e-12)
    invariance, derivatives, summaries = {}, {}, {}
    for mode in ('ROW', 'BI'):
        loss, summary = match(a, b, mode, labels, labels)
        shifted, _ = match(a + shift_a, b + shift_b, mode, labels, labels)
        scaled, _ = match(a * 7., b * 7., mode, labels, labels)
        torch.testing.assert_close(loss, shifted, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(loss, scaled, rtol=1e-12, atol=1e-12)
        ga, gb = torch.autograd.grad(loss, (a, b), retain_graph=True)
        assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
        assert ga.norm() > 0 and gb.norm() > 0
        translation = max(float(ga.sum(0).abs().max()), float(gb.sum(0).abs().max()))
        radial = abs(float(((ga * a).sum() + (gb * b).sum()).detach()))
        assert translation < 1e-12 and radial < 1e-12
        invariance[mode] = max(float((loss - shifted).detach().abs()), float((loss - scaled).detach().abs()))
        derivatives[mode] = dict(translation_max=translation, radial_abs=radial)
        summaries[mode] = summary
        # Independent class renaming and unrelated row orders keep semantics.
        pa, pb = [4, 1, 5, 0, 2, 3], [2, 5, 1, 3, 0, 4]
        permuted, ps = match(a[pa], b[pb], mode, pa, pb)
        renamed, _ = match(a, b, mode, [2, 5, 1, 4, 0, 3], [2, 5, 1, 4, 0, 3])
        torch.testing.assert_close(permuted, loss, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(renamed, loss, rtol=0., atol=0.)
        assert ps['argmaxerrors'] == summary['argmaxerrors']
    assert abs(summaries['BI']['loss'] - summaries['ROW']['loss']) > 1e-4
    assert summaries['BI']['iterations'] == 80 and summaries['ROW']['iterations'] == 0
    assert summaries['BI']['approximate_plan'] == (summaries['BI']['marginal_residual_max'] > 1e-5)
    assert all(0. <= value <= 1. for value in summaries['BI']['diagmass']['row']
               + summaries['BI']['diagmass']['column'])
    # Independent centered-distance closed form for six orthogonal classes:
    # Cdiag=-sqrt(5), Coff=1/sqrt(5), enough to fulfill the fixed .9 mass.
    easy = torch.zeros(6, 160, dtype=torch.float64)
    easy[:, :6] = torch.eye(6, dtype=torch.float64)
    easy.requires_grad_()
    easy_b = easy.detach().clone().requires_grad_()
    expected_cost = torch.full((6, 6), 1. / math.sqrt(5), dtype=torch.float64)
    expected_cost.fill_diagonal_(-math.sqrt(5))
    torch.testing.assert_close(_relative_cost(easy, easy_b)[0], expected_cost, rtol=1e-14, atol=1e-14)
    for mode in ('ROW', 'BI'):
        zero, summary = match(easy, easy_b[[2, 5, 1, 3, 0, 4]], mode,
                              labels, [2, 5, 1, 3, 0, 4])
        grads = torch.autograd.grad(zero, (easy, easy_b))
        assert float(zero.detach()) == 0. and all(torch.count_nonzero(g) == 0 for g in grads)
        assert summary['lowmass'] == {'row': 0, 'column': 0}
        assert summary['argmaxerrors'] == {'row': 0, 'column': 0}
    # Collapsed features give a finite positive loss, correct mass1/6 and zero
    # feature derivative; clipping the squared RMS before sqrt avoids0*inf.
    collapsed_a = torch.zeros(6, 160, dtype=torch.float64, requires_grad=True)
    collapsed_b = torch.zeros_like(collapsed_a, requires_grad=True)
    for mode in ('ROW', 'BI'):
        collapsed_loss, summary = match(collapsed_a, collapsed_b, mode, labels, labels)
        ga, gb = torch.autograd.grad(collapsed_loss, (collapsed_a, collapsed_b))
        assert abs(float(collapsed_loss.detach()) - .1 * math.log(5.4)) < 1e-14
        assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
        assert torch.count_nonzero(ga) == 0 and torch.count_nonzero(gb) == 0
        assert summary['floor_active'] and summary['costRMS'] == 0.
    # Fixed80 is an approximate plan. A finite example must disclose both
    # marginal residuals rather than infer exactness from the last update.
    assert summaries['BI']['row_marginal_residual'] >= 0.
    assert summaries['BI']['column_marginal_residual'] >= 0.
    # One central finite difference tests the complete cost-RMS and unrolled
    # Sinkhorn derivative rather than a mirrored analytical implementation.
    finite_a = av.clone().requires_grad_()
    finite_loss, _ = match(finite_a, bv, 'BI', labels, labels)
    gradient = torch.autograd.grad(finite_loss, finite_a)[0][0, 7]
    plus, minus = av.clone(), av.clone()
    plus[0, 7] += 1e-6
    minus[0, 7] -= 1e-6
    upper, _ = match(plus, bv, 'BI', labels, labels)
    lower, _ = match(minus, bv, 'BI', labels, labels)
    numeric = (upper - lower) / 2e-6
    torch.testing.assert_close(gradient, numeric, rtol=2e-5, atol=1e-8)
    # A default linear cosine head can read all true classes perfectly while
    # source receiver-specific geometry produces the wrong bilateral matches.
    signal_a = torch.zeros(6, 160, dtype=torch.float64)
    signal_b = torch.zeros_like(signal_a)
    signal_a[:, :6] = signal_b[:, :6] = torch.eye(6, dtype=torch.float64)
    signal_a[:, 6:12] = 3. * torch.eye(6, dtype=torch.float64)
    signal_b[:, 6:12] = 3. * torch.eye(6, dtype=torch.float64)[[1, 2, 3, 4, 5, 0]]
    fixed_W = torch.zeros(6, 160, dtype=torch.float64)
    fixed_W[:, :6] = torch.eye(6, dtype=torch.float64)
    logits_a = 30. * F.normalize(signal_a, dim=1, eps=1e-4) @ fixed_W.T
    logits_b = 30. * F.normalize(signal_b, dim=1, eps=1e-4) @ fixed_W.T
    assert torch.equal(logits_a.argmax(1), torch.arange(6)) and torch.equal(logits_b.argmax(1), torch.arange(6))
    signal_a.requires_grad_(); signal_b.requires_grad_()
    signal_loss, signal_summary = match(signal_a, signal_b, 'BI', labels, labels)
    signal_grad = torch.autograd.grad(signal_loss, (signal_a, signal_b))
    assert signal_summary['argmaxerrors'] == {'row': 6, 'column': 6}
    assert float(signal_loss.detach()) > 0 and all(g.norm() > 0 and torch.isfinite(g).all() for g in signal_grad)
    fp32_a, fp32_b = av.float().requires_grad_(), bv.float().requires_grad_()
    fp32_loss, _ = match(fp32_a, fp32_b, 'BI', torch.arange(6), torch.arange(6))
    assert fp32_loss.dtype == torch.float32
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(fp32_loss, (fp32_a, fp32_b)))
    negatives = 0
    for call in (
        lambda: match(a[:5], b, 'BI', labels, labels),
        lambda: match(a[:, :159], b[:, :159], 'BI', labels, labels),
        lambda: match(a, b.float(), 'BI', labels, labels),
        lambda: match(a.float().half(), b.float().half(), 'BI', labels, labels),
        lambda: match(a, b.to('meta'), 'BI', labels, labels),
        lambda: match(a * float('nan'), b, 'BI', labels, labels),
        lambda: match(a, b * float('inf'), 'BI', labels, labels),
        lambda: match(a, b, 'UNKNOWN', labels, labels),
        lambda: match(a, b, 'BI', [0, 1, 2, 3, 4, 4], labels),
        lambda: match(a, b, 'BI', labels, [0, 1, 2, 3, 4, 6]),
        lambda: match(a, b, 'BI', [False, 1, 2, 3, 4, 5], labels),
        lambda: match(a, b, 'BI', torch.arange(6).float(), labels),
        lambda: match(a, b, 'BI', labels, torch.arange(6)[:, None]),
    ):
        _reject(call); negatives += 1
    assert torch.equal(a.detach(), av) and torch.equal(b.detach(), bv)
    assert versions == (a._version, b._version) and a.grad is None and b.grad is None
    assert _equal(rng, _rng_snapshot())
    json.dumps(summaries, allow_nan=False)
    return dict(status='PASS', device='cpu', torch_version=torch.__version__,
                shift_scale_loss_error=invariance, invariance_derivatives=derivatives,
                BI_marginal_residual_max=summaries['BI']['marginal_residual_max'],
                BI_approximate_plan=summaries['BI']['approximate_plan'],
                ROW_loss=summaries['ROW']['loss'], BI_loss=summaries['BI']['loss'],
                finite_difference_abs_error=float((gradient - numeric).abs()),
                default_head_correct_but_BI_match_errors=signal_summary['argmaxerrors'],
                collapse_finite_positive_loss_zero_gradient=True, stopped_exact_zero=True,
                FP32_loss_and_gradient_finite=True, negative_checks=negatives,
                inputs_grad_and_RNG_unchanged=True, no_data_model_head_or_state=True)


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, allow_nan=False, indent=2))
