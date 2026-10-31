import math

import pytest
import torch

from experiments.pamf_restoration.phase_equivariant import (
    PhaseEquivariantRestorer, WidelyLinearRestorer,
)


def rotate(iq, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c * iq[:, 0] - s * iq[:, 1],
                        s * iq[:, 0] + c * iq[:, 1]), dim=1)


def test_identity_initialization_and_input_contract():
    model = PhaseEquivariantRestorer(width=4)
    x = torch.randn(2, 2, 256)
    torch.testing.assert_close(model(x), x, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 2, 128))


def test_phase_equivariance_after_nonzero_weights():
    torch.manual_seed(4)
    model = PhaseEquivariantRestorer(width=4)
    torch.nn.init.normal_(model.output.real.weight, std=0.1)
    torch.nn.init.normal_(model.output.imag.weight, std=0.1)
    x = torch.randn(3, 2, 256)
    for angle in (0.41, -1.3, math.pi):
        torch.testing.assert_close(model(rotate(x, angle)), rotate(model(x), angle),
                                   rtol=1e-5, atol=2e-6)


def test_gradients_reach_hidden_layers_after_first_update():
    torch.manual_seed(5)
    model = PhaseEquivariantRestorer(width=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(2, 2, 256)
    target = 0.7 * x
    for _ in range(2):
        optimizer.zero_grad()
        (model(x) - target).square().mean().backward()
        optimizer.step()
    assert model.stem.real.weight.grad is not None
    assert torch.isfinite(model.stem.real.weight.grad).all()
    assert model.stem.real.weight.grad.abs().sum() > 0


def test_image_branch_represents_fixed_iq_imbalance():
    model = WidelyLinearRestorer(width=4)
    y = torch.randn(2, 2, 256)
    torch.testing.assert_close(model(y), y, rtol=0, atol=0)
    # The direct image term acts on conjugate IQ and intentionally has a
    # different phase transformation from the equivariant base.
    with torch.no_grad():
        model.image.real.weight[0, 0, 2] = 0.2
    out = model(y)
    expected = y + 0.2 * torch.stack((y[:, 0], -y[:, 1]), dim=1)
    torch.testing.assert_close(out, expected)
    assert not torch.allclose(model(rotate(y, math.pi / 2)), rotate(out, math.pi / 2))
