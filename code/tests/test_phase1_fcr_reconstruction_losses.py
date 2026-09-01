import math

import torch

from cvsrffi.phase1_fcr_losses import (
    heteroscedastic_complex_nll,
    mrstft_loss,
    phase_increment_loss,
    physical_feature_loss,
)
from cvsrffi.phase1_fcr_physics import FisherGateOutput, FrozenFingerprintFeatureBank
from cvsrffi.phase1_fcr_types import FCRConfig


def _complex_iq(batch_size: int = 2, length: int = 256) -> torch.Tensor:
    torch.manual_seed(41)
    return torch.randn(batch_size, length, dtype=torch.complex64)


def test_complex_nll_prefers_exact_mean_with_bounded_variance() -> None:
    target = _complex_iq()
    config = FCRConfig(variance_floor=0.02, variance_ceiling=0.20)
    log_variance = torch.full((2, 256), math.log(0.10))

    exact = heteroscedastic_complex_nll(target, target, log_variance, config)
    bad = heteroscedastic_complex_nll(target, target + (1.0 + 1.0j), log_variance, config)

    assert exact < bad


def test_mrstft_identity_uses_noise_floor_and_zero_input_is_finite() -> None:
    target = _complex_iq()
    assert mrstft_loss(target, target) < 1e-6
    assert torch.isfinite(mrstft_loss(torch.zeros_like(target), torch.zeros_like(target)))


def test_phase_increment_wraps_and_is_amplitude_weighted() -> None:
    phase_a = torch.full((1, 32), math.pi - 1e-3)
    phase_b = torch.full((1, 32), -math.pi + 1e-3)
    wrapped_a = torch.polar(torch.ones_like(phase_a), phase_a)
    wrapped_b = torch.polar(torch.ones_like(phase_b), phase_b)

    assert phase_increment_loss(wrapped_a, wrapped_b) < 1e-2
    assert torch.isfinite(phase_increment_loss(torch.zeros_like(wrapped_a), wrapped_b))


def test_physical_loss_is_block_weighted_finite_and_differentiable() -> None:
    target = _complex_iq()
    prediction = (target + 0.2 * (1.0 + 0.3j)).detach().requires_grad_()
    blocks = FrozenFingerprintFeatureBank()(target)
    gate = FisherGateOutput(
        block_weights={name: torch.tensor(1.0) for name in blocks.blocks},
        quality={},
    )

    loss = physical_feature_loss(target, prediction, gate)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad.real).all()
    zero_gate = FisherGateOutput(
        block_weights={name: torch.tensor(0.0) for name in blocks.blocks},
        quality={},
    )
    assert physical_feature_loss(target, prediction.detach(), zero_gate).item() == 0.0
