from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.losses import (
    classification_margin,
    paired_satellite_loss,
)


def test_classification_margin_is_true_logit_minus_strongest_other_logit() -> None:
    logits = torch.tensor(
        [[3.0, 1.0, 0.0], [0.0, 2.0, 4.0]], requires_grad=True
    )
    tx = torch.tensor([0, 2])

    margin = classification_margin(logits, tx)

    assert margin.tolist() == [2.0, 2.0]
    margin.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("logits", "tx", "message"),
    [
        (torch.randn(2, 1), torch.tensor([0, 0]), "at least two"),
        (torch.randn(2, 3), torch.tensor([0]), "batch"),
        (torch.randn(2, 3), torch.tensor([-1, 1]), "range"),
        (torch.randn(2, 3), torch.tensor([0, 3]), "range"),
    ],
)
def test_classification_margin_validates_logits_and_tx(
    logits: torch.Tensor, tx: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        classification_margin(logits, tx)


def test_paired_satellite_loss_is_zero_for_identical_finite_pairs_and_differentiable() -> None:
    clean_z_id = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    satellite_z_id = clean_z_id.detach().clone().requires_grad_()
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    satellite_logits = clean_logits.detach().clone().requires_grad_()

    loss = paired_satellite_loss(
        clean_z_id, satellite_z_id, clean_logits, satellite_logits
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(loss)
    loss.backward()
    for tensor in (clean_z_id, satellite_z_id, clean_logits, satellite_logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_paired_satellite_loss_penalizes_identity_or_prediction_drift() -> None:
    clean_z_id = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    satellite_z_id = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    satellite_logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]], requires_grad=True)

    loss = paired_satellite_loss(
        clean_z_id, satellite_z_id, clean_logits, satellite_logits
    )

    assert loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(clean_z_id.grad).all()
    assert torch.isfinite(satellite_z_id.grad).all()
    assert torch.isfinite(clean_logits.grad).all()
    assert torch.isfinite(satellite_logits.grad).all()
