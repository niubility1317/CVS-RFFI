import torch

from experiments.pamf_lora_main.chirp_band_power import spectrum
from experiments.pamf_lora_main.near_band_gain import (
    GAIN_CAP, adjust, fit_coefficients, increase, inputs, protected_target,
)


def test_near_band_gain_retains_common_and_other_bands():
    restored = torch.randn(3, 6, 256, dtype=torch.complex64)
    features, before, basis, bins = inputs(restored, restored)
    assert features.shape == (3, 5)
    g = torch.full((3,), .2)
    after = spectrum(adjust(restored, before, basis, bins, g), basis)
    torch.testing.assert_close(after[:, 1:, bins], 1.2 * before[:, 1:, bins], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(after[:, 0], before[:, 0], atol=1e-6, rtol=1e-5)
    outside = torch.ones(256, dtype=torch.bool)
    outside[bins] = False
    torch.testing.assert_close(after[..., outside], before[..., outside], atol=1e-6, rtol=1e-5)


def test_training_calibration_recovers_attenuated_signal_with_bounded_gain():
    y = torch.randn(8, 6, 256, dtype=torch.complex64)
    features, after, basis, bins = inputs(y, .9 * y)
    target = protected_target(after, y, basis, bins)
    torch.testing.assert_close(target, torch.full_like(target, 1/.9-1))
    unrelated = protected_target(after, -y, basis, bins)
    assert not unrelated.count_nonzero()
    fit = fit_coefficients(features, target)
    pred = increase(features, fit)
    torch.testing.assert_close(pred, target)
    assert ((pred >= 0) & (pred <= GAIN_CAP)).all()
