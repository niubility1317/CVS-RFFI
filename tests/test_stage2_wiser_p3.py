from __future__ import annotations

import pytest
import torch

from cvsrffi.stage2_binova_d92 import (
    BiNOVAD92Error,
    differentiable_old_d92_logits,
    exact_d92_fit,
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

    gradients = (eval_id.grad, eval_fft.grad)
    for argument_index, gradient in zip((2, 3), gradients):
        assert float(gradient[0, 0]) == pytest.approx(
            finite_difference(argument_index, (0, 0)), rel=0.05, abs=5.0e-3
        )
