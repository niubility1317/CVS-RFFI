from __future__ import annotations

import inspect

import pytest
import torch

from cvsrffi.phase1_bicad_xr.pair import (
    pair_delta_objectives,
    pair_identity_hinge,
    vicreg_pair_loss,
)


def test_pair_identity_hinge_is_zero_inside_tolerance_and_positive_outside() -> None:
    clean = torch.tensor([[2.0, 0.0], [0.0, 3.0]], requires_grad=True)
    same_direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    orthogonal = torch.tensor([[0.0, 2.0], [3.0, 0.0]], requires_grad=True)

    inside = pair_identity_hinge(clean, same_direction, epsilon=0.05)
    outside = pair_identity_hinge(clean, orthogonal, epsilon=0.05)

    assert inside.item() == pytest.approx(0.0, abs=1e-7)
    assert outside.item() > 0.0
    assert torch.isfinite(inside)
    assert torch.isfinite(outside)
    outside.backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()
    assert orthogonal.grad is not None and torch.isfinite(orthogonal.grad).all()


@pytest.mark.parametrize("batch_size", [0, 1, 4])
def test_vicreg_pair_loss_is_named_finite_and_graph_connected(batch_size: int) -> None:
    clean = torch.randn(batch_size, 4, requires_grad=True)
    satellite = torch.randn(batch_size, 4, requires_grad=True)

    components = vicreg_pair_loss(clean, satellite, gamma=1.0)

    assert set(components) == {"total", "invariance", "variance", "covariance"}
    assert all(torch.isfinite(value) for value in components.values())
    components["total"].backward()
    assert clean.grad is not None and torch.isfinite(clean.grad).all()
    assert satellite.grad is not None and torch.isfinite(satellite.grad).all()


def test_pair_delta_objectives_are_label_free_except_for_channel_and_reach_all_inputs() -> None:
    clean_id = torch.randn(4, 5, requires_grad=True)
    satellite_id = torch.randn(4, 5, requires_grad=True)
    clean_c = torch.randn(4, 3, requires_grad=True)
    satellite_c = torch.randn(4, 3, requires_grad=True)
    channel = torch.tensor([0, 1, 2, 1])

    components = pair_delta_objectives(
        clean_id, satellite_id, clean_c, satellite_c, channel
    )

    assert set(components) == {
        "identity_channel_adversary",
        "channel_prediction",
        "pair_stability",
        "channel_equivariance",
        "delta_norm_hinge",
    }
    assert all(torch.isfinite(value) for value in components.values())
    assert "tx" not in inspect.signature(pair_delta_objectives).parameters

    sum(components.values()).backward()
    for tensor in (clean_id, satellite_id, clean_c, satellite_c):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: pair_identity_hinge(torch.randn(2, 3), torch.randn(2, 4), 0.05),
            "same shape",
        ),
        (
            lambda: vicreg_pair_loss(torch.randn(2, 3), torch.randn(2, 3), gamma=-1.0),
            "gamma",
        ),
        (
            lambda: pair_delta_objectives(
                torch.randn(2, 3),
                torch.randn(2, 3),
                torch.randn(2, 2),
                torch.randn(2, 2),
                torch.tensor([0.0, 1.0]),
            ),
            "channel",
        ),
    ],
)
def test_pair_objectives_reject_invalid_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
