from __future__ import annotations

import numpy as np
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


def test_cross_fit_logits_backpropagate_to_fit_and_held_out_identity() -> None:
    """Catches a detached exact-score bridge that cannot train the P3 loss."""

    fit_id, fit_fft, labels, eval_id, eval_fft = make_d92_case("normal")
    fit_id.requires_grad_()
    eval_id.requires_grad_()
    logits = differentiable_old_d92_logits(fit_id, fit_fft, labels, eval_id, eval_fft)
    logits.square().mean().backward()
    assert fit_id.grad is not None and torch.isfinite(fit_id.grad).all()
    assert eval_id.grad is not None and torch.isfinite(eval_id.grad).all()
