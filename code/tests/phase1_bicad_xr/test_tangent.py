from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.tangent import (
    ReceiverTangentBank,
    factual_tangent,
    one_step_tangent_worst_direction,
)


def test_update_accumulates_source_class_conditioned_centers_and_rejects_heldout() -> None:
    bank = ReceiverTangentBank(feature_dim=2, rank=1, source_receivers=(0, 1))

    bank.update(
        torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 0, 1, 1]),
    )
    bank.update(
        torch.tensor([[5.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 5.0]]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
    )

    assert torch.equal(bank.center_for(0, 0), torch.tensor([3.0, 0.0]))
    assert torch.equal(bank.center_for(0, 1), torch.tensor([2.0, 0.0]))
    assert torch.equal(bank.center_for(1, 0), torch.tensor([0.0, 2.0]))
    assert torch.equal(bank.center_for(1, 1), torch.tensor([0.0, 3.0]))
    before = bank.centers
    with pytest.raises(ValueError, match="source"):
        bank.update(
            torch.tensor([[9.0, 9.0]]),
            torch.tensor([0]),
            torch.tensor([2]),
        )
    assert bank.centers.keys() == before.keys()
    for key in before:
        assert torch.equal(bank.centers[key], before[key])


def _update_bank_with_fixed_centers(bank: ReceiverTangentBank, scatter_axis: str) -> None:
    if scatter_axis == "z":
        z = torch.tensor(
            [
                [3.0, 0.0, 100.0],
                [3.0, 0.0, -100.0],
                [-3.0, 0.0, 80.0],
                [-3.0, 0.0, -80.0],
                [0.0, 1.0, 60.0],
                [0.0, 1.0, -60.0],
                [0.0, -1.0, 40.0],
                [0.0, -1.0, -40.0],
            ]
        )
    else:
        z = torch.tensor(
            [
                [3.0, 50.0, 0.0],
                [3.0, -50.0, 0.0],
                [-3.0, 30.0, 0.0],
                [-3.0, -30.0, 0.0],
                [20.0, 1.0, 0.0],
                [-20.0, 1.0, 0.0],
                [10.0, -1.0, 0.0],
                [-10.0, -1.0, 0.0],
            ]
        )
    bank.update(
        z,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
    )


def test_basis_uses_same_class_cross_receiver_centers_not_within_cell_scatter() -> None:
    bank_z = ReceiverTangentBank(feature_dim=3, rank=1, source_receivers=(0, 1))
    bank_xy = ReceiverTangentBank(feature_dim=3, rank=1, source_receivers=(0, 1))
    _update_bank_with_fixed_centers(bank_z, "z")
    _update_bank_with_fixed_centers(bank_xy, "xy")

    projection_z = bank_z.basis_for(0).transpose(0, 1) @ bank_z.basis_for(0)
    projection_xy = bank_xy.basis_for(0).transpose(0, 1) @ bank_xy.basis_for(0)

    assert torch.allclose(projection_z, projection_xy, atol=1e-6)
    assert torch.allclose(projection_z, torch.diag(torch.tensor([1.0, 0.0, 0.0])), atol=1e-6)


def test_f1_factual_and_f2_worst_direction_run_end_to_end_in_tangent_coefficients() -> None:
    bank = ReceiverTangentBank(feature_dim=3, rank=1, source_receivers=(0, 1))
    _update_bank_with_fixed_centers(bank, "z")
    coefficients = torch.tensor([[0.25]], requires_grad=True)
    base = torch.zeros(1, 3)
    receiver = torch.tensor([0])

    factual = factual_tangent(bank, base, receiver, coefficients=coefficients)
    factual_loss = factual.square().sum()
    worst = one_step_tangent_worst_direction(
        factual_loss,
        coefficients,
        radius=0.5,
    )
    worse = factual_tangent(
        bank,
        base,
        receiver,
        coefficients=coefficients.detach() + worst,
    )

    assert factual.shape == (1, 3)
    assert torch.linalg.vector_norm(factual).item() == pytest.approx(0.25)
    assert torch.equal(worst, torch.tensor([[0.5]]))
    assert worse.square().sum().item() > factual_loss.item()


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
        bank.update(
            torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
        )
