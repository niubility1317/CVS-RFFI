import torch

from experiments.pamf_lora_main.variance_bound import (
    VarianceBound, noise_gain, point_gain, spectral_contrast, variance_interval,
)


def test_identical_estimates_collapse_to_point_with_matched_mean():
    outer = torch.full((2, 6, 1), .15)
    weights = torch.ones_like(outer) / 6
    selectors = torch.eye(4)
    contrast = (outer * (weights * (1-weights))).sum(1).expand(2, 4)
    variance, fraction = variance_interval(outer, contrast, weights*(1-weights), selectors)
    torch.testing.assert_close(variance, outer.expand(2, 6, 2))
    torch.testing.assert_close(fraction, torch.zeros_like(fraction), rtol=0, atol=0)
    proposed = torch.full_like(variance,.5)
    torch.testing.assert_close(noise_gain(variance,proposed,6,fraction),
                               point_gain(variance,proposed,6))


def test_uncertainty_reduces_edit_and_respects_each_budget():
    variance = torch.tensor([.01,.1,1.])
    fraction = torch.tensor([.9,.8,.7])
    proposed = torch.tensor([.4,1.,10.])
    point = point_gain(variance, proposed, 6)
    mixed = noise_gain(variance, proposed, 6, fraction)
    assert torch.all(point > mixed)
    assert torch.all(mixed.square()*proposed <= variance)
    zero = noise_gain(torch.zeros_like(variance), proposed, 6, fraction)
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)


def test_contrast_discards_identical_shared_chirps_and_regions_keep_dc():
    wave = torch.randn(2, 1, 256, dtype=torch.complex64).expand(-1,6,-1)
    quality = torch.rand(2, 6, 1)
    contrast, weights = spectral_contrast(wave, quality)
    assert contrast.max() < 1e-12
    assert (weights >= 0).all()
    model = VarianceBound((8,8,8,8),uncertain=True)
    coverage=model.regions.sum(0)
    assert coverage[0]==0 and (coverage[1:]==1).all()
    torch.testing.assert_close(model.selectors.sum(1),torch.ones(8))
