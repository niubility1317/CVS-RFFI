from __future__ import annotations

import pytest
import torch

from cvsrffi.stage2_binova_d92 import (
    BiNOVAD92Error,
    differentiable_old_d92_logits,
    exact_d92_fit,
)
from cvsrffi.stage2_wiser_p3 import (
    cross_fitted_p3_loss,
    frozen_class_risk,
    stratified_crossfit_indices,
    update_nonnegative_duals,
)


def make_d92_case(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build balanced old-only fit data with independently held-out rows."""

    generator = torch.Generator().manual_seed(713102)
    labels = torch.arange(6, dtype=torch.long).repeat_interleave(10)
    eval_labels = torch.arange(6, dtype=torch.long).repeat_interleave(2)
    fit_identity = 0.04 * torch.randn(60, 160, generator=generator)
    fit_fft = 0.04 * torch.randn(60, 96, generator=generator)
    eval_identity = 0.04 * torch.randn(12, 160, generator=generator)
    eval_fft = 0.04 * torch.randn(12, 96, generator=generator)
    for class_id in range(6):
        fit_mask = labels == class_id
        eval_mask = eval_labels == class_id
        fit_identity[fit_mask, class_id] += 1.0
        fit_fft[fit_mask, class_id] += 0.8
        eval_identity[eval_mask, class_id] += 1.0
        eval_fft[eval_mask, class_id] += 0.8
    if case == "zero_identity":
        fit_identity.zero_()
        eval_identity.zero_()
    elif case == "zero_fft":
        fit_fft.zero_()
        eval_fft.zero_()
    elif case == "tiny":
        fit_identity.mul_(1.0e-7)
        fit_fft.mul_(1.0e-7)
        eval_identity.mul_(1.0e-7)
        eval_fft.mul_(1.0e-7)
    elif case == "ill_conditioned":
        fit_identity[:, 1:] = fit_identity[:, :1]
        fit_fft[:, 1:] = fit_fft[:, :1]
        eval_identity[:, 1:] = eval_identity[:, :1]
        eval_fft[:, 1:] = eval_fft[:, :1]
    elif case != "normal":
        raise ValueError(f"unknown D92 case: {case}")
    return fit_identity, fit_fft, labels, eval_identity, eval_fft


def make_small_k_d92_case(
    k_shot: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact D62 small-K support cases without empty class IDs."""

    generator = torch.Generator().manual_seed(713102 + k_shot)
    labels = torch.arange(6, dtype=torch.long).repeat_interleave(k_shot)
    eval_labels = torch.arange(6, dtype=torch.long)
    fit_identity = 0.04 * torch.randn(6 * k_shot, 160, generator=generator)
    fit_fft = 0.04 * torch.randn(6 * k_shot, 96, generator=generator)
    eval_identity = 0.04 * torch.randn(6, 160, generator=generator)
    eval_fft = 0.04 * torch.randn(6, 96, generator=generator)
    for class_id in range(6):
        fit_identity[labels == class_id, class_id] += 1.0
        fit_fft[labels == class_id, class_id] += 0.8
        eval_identity[eval_labels == class_id, class_id] += 1.0
        eval_fft[eval_labels == class_id, class_id] += 0.8
    return fit_identity, fit_fft, labels, eval_identity, eval_fft


@pytest.mark.parametrize(
    "case", ["normal", "zero_identity", "zero_fft", "tiny", "ill_conditioned"]
)
def test_differentiable_old_d92_matches_exact_logits(case: str) -> None:
    """Catches bridge drift from the locked exact old-only D92 scoring path."""

    fit_id, fit_fft, labels, eval_id, eval_fft = make_d92_case(case)
    exact = exact_d92_fit(
        fit_id.detach().numpy(),
        fit_fft.detach().numpy(),
        labels.numpy(),
        class_ids=range(6),
        old_class_count=6,
        seed=713102,
        device="cpu",
    )
    expected = torch.tensor(exact.score(eval_id.numpy(), eval_fft.numpy()))
    actual = differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft)
    assert torch.max(torch.abs(actual.double() - expected.double())).item() < 1.0e-4


def test_differentiable_d92_rejects_both_modalities_zero() -> None:
    """Catches accepting rows that the exact D92 feature geometry cannot score."""

    labels = torch.arange(6, dtype=torch.long).repeat_interleave(10)
    zero_identity = torch.zeros(60, 160)
    zero_fft = torch.zeros(60, 96)
    with pytest.raises(BiNOVAD92Error, match="both modalities"):
        differentiable_old_d92_logits(
            zero_identity, zero_fft, labels, zero_identity[:2], zero_fft[:2]
        )


@pytest.mark.parametrize("k_shot", [1, 2])
def test_differentiable_old_d92_matches_exact_small_k_fallback(k_shot: int) -> None:
    """Catches omission of the locked D62 K<=2 unit-covariance fallback."""

    fit_id, fit_fft, labels, eval_id, eval_fft = make_small_k_d92_case(k_shot)
    exact = exact_d92_fit(
        fit_id.numpy(), fit_fft.numpy(), labels.numpy(),
        class_ids=range(6), old_class_count=6, seed=713102, device="cpu",
    )
    expected = torch.tensor(exact.score(eval_id.numpy(), eval_fft.numpy()))
    actual = differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft)
    assert torch.isfinite(actual).all()
    assert torch.max(torch.abs(actual.double() - expected.double())).item() < 1.0e-4

    if k_shot == 2:
        fit_id.requires_grad_()
        eval_fft.requires_grad_()
        differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft).square().mean().backward()
        for gradient in (fit_id.grad, eval_fft.grad):
            assert gradient is not None and torch.isfinite(gradient).all()


def test_cross_fit_logits_have_nonzero_identity_and_fft_gradients() -> None:
    """Catches a D92 loss that bypasses either fit or held-out modality."""

    fit_id, fit_fft, labels, eval_id, eval_fft = make_d92_case("normal")
    fit_id.requires_grad_()
    fit_fft.requires_grad_()
    eval_id.requires_grad_()
    eval_fft.requires_grad_()
    logits = differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft)
    logits.square().mean().backward()
    for gradient in (fit_id.grad, fit_fft.grad, eval_id.grad, eval_fft.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0.0


def test_cross_fit_logits_autograd_matches_selected_finite_differences() -> None:
    """Catches a forward-exact bridge whose backward path is a different model."""

    fit_id, fit_fft, labels, eval_id, eval_fft = make_d92_case("normal")
    fit_id.requires_grad_()
    fit_fft.requires_grad_()
    eval_id.requires_grad_()
    eval_fft.requires_grad_()
    score = differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft)[0, 0]
    score.backward()
    epsilon = 1.0e-2

    def finite_difference(argument_index: int, index: tuple[int, int]) -> float:
        source = (fit_id, fit_fft, eval_id, eval_fft)[argument_index]
        plus = source.detach().clone()
        minus = source.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        arguments = [fit_id.detach(), fit_fft.detach(), labels, eval_id.detach(), eval_fft.detach()]
        destination = (0, 1, 3, 4)[argument_index]
        arguments[destination] = plus
        upper = differentiable_old_d92_logits(*arguments)[0, 0]
        arguments[destination] = minus
        lower = differentiable_old_d92_logits(*arguments)[0, 0]
        return float(((upper - lower) / (2.0 * epsilon)).detach())

    selected = ((0, (0, 0)), (1, (1, 0)), (2, (0, 0)), (3, (0, 0)))
    gradients = (fit_id.grad, fit_fft.grad, eval_id.grad, eval_fft.grad)
    for argument_index, index in selected:
        assert float(gradients[argument_index][index]) == pytest.approx(
            finite_difference(argument_index, index), rel=0.05, abs=5.0e-3
        )


def test_five_fold_crossfit_is_eight_fit_two_valid_per_class() -> None:
    """Catches folds that omit support rows or leak a validation row into fitting."""

    labels = torch.arange(6).repeat_interleave(10)
    tokens = tuple(f"opaque-support-{index:02d}" for index in range(len(labels)))
    folds = stratified_crossfit_indices(labels, tokens, fold_count=5, seed=713102)
    assert len(folds) == 5
    seen = torch.zeros(60, dtype=torch.long)
    for fold in folds:
        seen[fold.validation_indices] += 1
        for class_id in range(6):
            assert int((labels[fold.fit_indices] == class_id).sum()) == 8
            assert int((labels[fold.validation_indices] == class_id).sum()) == 2
    assert torch.equal(seen, torch.ones_like(seen))


def test_crossfit_token_membership_is_invariant_to_package_row_order() -> None:
    """Catches a fold assignment that silently changes when a support package is reordered."""

    labels = torch.arange(6).repeat_interleave(10)
    tokens = tuple(f"opaque-support-{index:02d}" for index in range(len(labels)))
    original = stratified_crossfit_indices(labels, tokens, fold_count=5, seed=713102)
    permutation = torch.Generator().manual_seed(713103)
    row_order = torch.randperm(len(labels), generator=permutation)
    permuted_tokens = tuple(tokens[index] for index in row_order.tolist())
    permuted = stratified_crossfit_indices(
        labels[row_order], permuted_tokens, fold_count=5, seed=713102
    )
    original_sets = [
        {tokens[index] for index in fold.validation_indices.tolist()} for fold in original
    ]
    permuted_sets = [
        {permuted_tokens[index] for index in fold.validation_indices.tolist()}
        for fold in permuted
    ]
    assert original_sets == permuted_sets


def test_p3_loss_reports_class_risk_and_penalizes_only_violations() -> None:
    """Catches P3 risk that is not computed from held-out support predictions."""

    identity, fft, labels, _, _ = make_d92_case("normal")
    tokens = tuple(f"opaque-support-{index:02d}" for index in range(len(labels)))
    folds = stratified_crossfit_indices(labels, tokens, fold_count=5, seed=713102)
    result = cross_fitted_p3_loss(
        identity,
        fft,
        labels,
        folds=folds,
        baseline_class_risk=torch.full((6,), 0.5),
        class_duals=torch.ones(6),
        epsilon=torch.zeros(6),
        rho=2.0,
        beta=0.25,
        tau=0.1,
    )
    assert result.class_risk.shape == (6,)
    assert torch.all(result.violation >= 0)
    assert result.total >= result.mean_risk
    assert torch.equal(result.oof_predictions, result.oof_logits.argmax(dim=1))
    assert torch.allclose(result.class_risk, frozen_class_risk(result.oof_logits, labels))


def test_duals_remain_nonnegative() -> None:
    """Catches a dual update that rewards a resolved class-risk violation below zero."""

    updated = update_nonnegative_duals(
        torch.tensor([0.1, 0.0]), torch.tensor([-1.0, 0.4]), rate=0.5
    )
    assert torch.equal(updated, torch.tensor([0.0, 0.2]))


def test_dual_updates_are_detached_between_optimization_steps() -> None:
    """Catches dual ascent retaining a released P3 training graph across steps."""

    identity, fft, labels, _, _ = make_small_k_d92_case(2)
    identity.requires_grad_()
    fft.requires_grad_()
    tokens = tuple(f"opaque-support-{index:02d}" for index in range(len(labels)))
    folds = stratified_crossfit_indices(labels, tokens, fold_count=2, seed=713102)
    duals = torch.zeros(6)
    baseline = torch.full((6,), 0.5, requires_grad=True)
    epsilon = torch.zeros(6, requires_grad=True)
    for _ in range(2):
        result = cross_fitted_p3_loss(
            identity,
            fft,
            labels,
            folds=folds,
            baseline_class_risk=baseline,
            class_duals=duals,
            epsilon=epsilon,
            rho=2.0,
            beta=0.25,
            tau=0.1,
        )
        result.total.backward()
        duals = update_nonnegative_duals(duals, result.violation, rate=0.5)
        assert not duals.requires_grad
        assert duals.grad_fn is None
        identity.grad = None
        fft.grad = None
    assert baseline.grad is None
    assert epsilon.grad is None
