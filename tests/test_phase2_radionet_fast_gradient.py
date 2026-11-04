import math

import pytest
import torch

from experiments.pamf_phase2_practical.vendor_native.comparison_suite import adaptation


def model_with_grads(device):
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(2, device=device))
    model.b = torch.nn.Parameter(torch.zeros(1, device=device))
    model.c = torch.nn.Parameter(torch.zeros(1, device=device))
    model.a.grad = torch.tensor([3.0, 4.0], device=device)
    model.b.grad = torch.tensor([12.0], device=device)
    return model


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_radionet_norm_preserves_per_tensor_reduction_and_rng(device):
    model = model_with_grads(device)
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state() if device == 'cuda' else None
    assert adaptation._radionet_grad_norm(model) == 13.0
    assert torch.equal(cpu_before, torch.get_rng_state())
    if cuda_before is not None:
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())
    assert torch.equal(model.a.grad, torch.tensor([3.0, 4.0], device=device))


def test_radionet_norm_preserves_empty_and_nonfinite_behavior():
    model = model_with_grads('cpu')
    model.a.grad = model.b.grad = None
    assert adaptation._radionet_grad_norm(model) == 0.0
    model.b.grad = torch.tensor([float('inf')])
    assert math.isinf(adaptation._radionet_grad_norm(model))
    model.b.grad = torch.tensor([float('nan')])
    assert math.isnan(adaptation._radionet_grad_norm(model))
