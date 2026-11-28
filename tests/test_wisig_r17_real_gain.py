import torch

from experiments.pamf_lora_main.dura_real_gain_wisig_spectral import spectral_fusion
from experiments.pamf_lora_main.dura_real_gain_fusion import RealSpectralGain


def test_zero_gain_preserves_r17_and_receives_gradient():
    torch.manual_seed(123)
    observed = torch.randn(2, 256, dtype=torch.complex64)
    prior = .6 * observed
    model = RealSpectralGain(32)
    result = spectral_fusion(model, observed, prior)
    torch.testing.assert_close(result, prior, rtol=0, atol=0)
    result.abs().square().mean().backward()
    assert torch.isfinite(model.net[-1].weight.grad).all()
    assert model.net[-1].weight.grad.norm() > 0
