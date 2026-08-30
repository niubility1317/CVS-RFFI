from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.tangent import (
    ReceiverTangentBank,
    factual_tangent,
    one_step_tangent_worst_direction,
)


def test_receiver_tangent_bank_fits_source_receivers_and_rejects_heldout_receiver() -> None:
    source_features = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    source_receivers = torch.tensor([0, 0, 1, 1])
    bank = ReceiverTangentBank(
        feature_dim=3,
        rank=2,
        source_receivers=(0, 1),
    )

    bank.fit(source_features, source_receivers)

    assert bank.receiver_ids == (0, 1)
    assert bank.basis_for(0).shape == (2, 3)
    assert torch.isfinite(bank.basis_for(1)).all()
    with pytest.raises(ValueError, match="source"):
        bank.fit(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), torch.tensor([2, 2]))
    with pytest.raises(ValueError, match="source"):
        factual_tangent(bank, torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([2]))


def test_factual_tangent_uses_the_factual_receiver_basis_once() -> None:
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    receivers = torch.tensor([0, 0, 1, 1])
    bank = ReceiverTangentBank(feature_dim=3, rank=1, source_receivers=(0, 1))
    bank.fit(features, receivers)

    coefficients = torch.tensor([[2.0], [0.0]])
    perturbed = factual_tangent(
        bank,
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([0, 1]),
        coefficients=coefficients,
    )

    assert perturbed.shape == (2, 3)
    assert torch.linalg.vector_norm(perturbed[0]).item() == pytest.approx(2.0)
    assert torch.linalg.vector_norm(perturbed[1]).item() == pytest.approx(0.0)


def test_f2_uses_one_coefficient_gradient_as_the_worst_ascent_direction() -> None:
    coefficients = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = 3.0 * coefficients[0] + coefficients[1].square()

    direction = one_step_tangent_worst_direction(loss, coefficients, radius=2.0)

    expected = torch.tensor([6.0 / (13.0**0.5), 4.0 / (13.0**0.5)])
    assert torch.allclose(direction, expected, atol=1e-6)
    assert not direction.requires_grad
    assert coefficients.grad is None


def test_tangent_bank_fails_closed_on_nonfinite_source_features() -> None:
    bank = ReceiverTangentBank(feature_dim=2, rank=1, source_receivers=(0,))

    with pytest.raises(ValueError, match="finite"):
        bank.fit(torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]), torch.tensor([0, 0]))
