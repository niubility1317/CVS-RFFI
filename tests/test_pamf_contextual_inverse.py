import math

import pytest
import torch

from experiments.pamf_dura_direct_train.contextual_model import ContextualSpectralRestorer


def test_starts_as_identity_and_rejects_wrong_shape():
    model = ContextualSpectralRestorer(width=8, context_dim=8)
    x = torch.randn(3, 2, 256)
    torch.testing.assert_close(model(x), x, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(torch.randn(3, 2, 255))


def test_spectral_path_uses_conjugate_mirror_correctly():
    model = ContextualSpectralRestorer(width=8, context_dim=8)
    # A constant real coefficient B implements b * conjugate(y) in time.
    with torch.no_grad():
        model.spectral_head.bias[512:768] = math.atanh(0.1)
    y = torch.randn(2, 2, 256)
    expected = y + 0.2 * torch.stack((y[:, 0], -y[:, 1]), dim=1)
    torch.testing.assert_close(model(y), expected, rtol=1e-5, atol=2e-6)


def test_per_window_independence_and_gradients():
    torch.manual_seed(1)
    model = ContextualSpectralRestorer(width=8, context_dim=8)
    torch.nn.init.normal_(model.spectral_head.weight, std=0.01)
    torch.nn.init.normal_(model.time_output.weight, std=0.01)
    y = torch.randn(3, 2, 256)
    batched = model(y)
    individual = torch.cat([model(y[i:i+1]) for i in range(3)])
    torch.testing.assert_close(batched, individual, rtol=1e-5, atol=2e-6)
    batched.square().mean().backward()
    for name in ('spectral_head.weight', 'time_output.weight', 'context.0.weight', 'time_input.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
