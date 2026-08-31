from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.tailguard import (
    bounded_hard_group_weights,
    margin_group_risks,
    margin_rex_cvar_loss,
)


def test_margin_group_risks_selects_the_worst_tail_within_each_group() -> None:
    margins = torch.tensor([2.0, -1.0, 0.0, -3.0], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1])

    risks = margin_group_risks(margins, groups, tail_fraction=0.5)

    expected = torch.stack(
        [
            torch.nn.functional.softplus(torch.tensor(1.0)),
            torch.nn.functional.softplus(torch.tensor(3.0)),
        ]
    )
    assert torch.allclose(risks, expected)
    risks.mean().backward()
    assert margins.grad is not None
    assert torch.isfinite(margins.grad).all()


def test_margin_rex_cvar_loss_is_finite_and_contains_group_variance_and_cvar() -> None:
    margins = torch.tensor([2.0, -1.0, 0.0, -3.0], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1])

    risks = margin_group_risks(margins, groups, tail_fraction=0.5)
    loss = margin_rex_cvar_loss(
        margins, groups, tail_fraction=0.5, lambda_rex=0.02, lambda_cvar=0.05
    )

    expected = 0.02 * risks.var(unbiased=False) + 0.05 * risks.max()
    assert loss.item() == pytest.approx(expected.item())
    assert torch.isfinite(loss)
    loss.backward()
    assert margins.grad is not None
    assert torch.isfinite(margins.grad).all()


def test_margin_group_risks_supports_class_receiver_view_group_keys() -> None:
    margins = torch.tensor([1.0, -1.0, 2.0, -2.0])
    groups = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 2, 1],
            [1, 2, 1],
        ]
    )

    risks = margin_group_risks(margins, groups, tail_fraction=0.5)

    assert risks.shape == (2,)
    assert torch.isfinite(risks).all()


def test_bounded_hard_group_weights_keep_hard_sampling_mass_at_or_below_30_percent() -> None:
    group_risks = torch.arange(1.0, 11.0)

    weights = bounded_hard_group_weights(group_risks, hard_fraction=0.2)

    hard_indices = torch.topk(group_risks, k=2).indices
    assert weights.shape == group_risks.shape
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[hard_indices].sum().item() <= 0.30 + 1e-7


@pytest.mark.parametrize(
    "bad_margins",
    [torch.tensor([0.0, float("nan")]), torch.tensor([0.0, float("inf")])],
)
def test_tailguard_rejects_nonfinite_margins(bad_margins: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite"):
        margin_rex_cvar_loss(bad_margins, torch.tensor([0, 0]))
