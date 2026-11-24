import torch

from experiments.pamf_lora_main.factorized_chirps import FactorizedChirps
from experiments.pamf_lora_main.fission_risk import (
    band_masks, edit_risk, mix_readouts, optimal_mix, split_observation,
)


def test_gaussian_split_has_near_zero_cross_covariance_at_correct_variance():
    rng = torch.Generator().manual_seed(12)
    noise, probe = torch.randn(2, 150000, generator=rng)
    fit, held = split_observation(noise, probe, tau=.5)
    assert (fit * held).mean().abs() < .02
    assert abs(float(fit.var()) - 1.25) < .03
    assert abs(float(held.var()) - 5) < .05


def test_band_quadratic_selects_interior_and_boundary_weights():
    base = torch.zeros(1, 2, 2, 3, dtype=torch.complex64)
    other = torch.tensor([1j, 2, 3j]).expand_as(base)
    target = other * torch.tensor([.25, -.1, 1.3])
    offset = torch.tensor([.2j, .2, .8j])
    target = target + torch.stack((offset, -offset))[None, :, None]
    weights = torch.tensor([[1., 2.]])[:, :, None]
    linear, quadratic = edit_risk(base, other, target, weights, torch.eye(3))
    torch.testing.assert_close(optimal_mix(linear, quadratic), torch.tensor([[.25, 0., 1.]]))
    torch.testing.assert_close(optimal_mix(torch.zeros(1), torch.zeros(1)), torch.full((1,), .5))


def test_readouts_match_frozen_endpoints_and_packetwise_fixed_probes():
    class Endpoint:
        def __init__(self, gain):
            self.gain = gain

        def __call__(self, wave, quality, rotation):
            dc = wave.mean(-1, keepdim=True)
            refined = dc + self.gain * (wave - dc)
            return refined, refined

    template = FactorizedChirps((16, 20, 28, 40), coupled=True)
    a, b = Endpoint(.75), Endpoint(.25)
    a.bands, a.selectors = template.bands, template.selectors
    rng = torch.Generator().manual_seed(93)
    y = torch.randn(2, 6, 256, generator=rng, dtype=torch.complex64)
    reference = torch.ones_like(y)
    q, rms = torch.ones_like(y[..., :1].real), torch.ones_like(y[..., :1].real)
    edits, doses = mix_readouts(y, reference, q, rms, (a, b))
    for got, model in zip(edits[:2], (a, b), strict=True):
        expected = torch.lerp(y, model(y, q, reference)[-1], .5) - y
        torch.testing.assert_close(got, expected)
    for value in edits:
        assert value.mean(-1).abs().max() < 1e-6
    assert band_masks(y)[:, 0].count_nonzero() == 0
    single, singleton_doses = mix_readouts(y[1:], reference[1:], q[1:], rms[1:], (a, b))
    for expected, actual in zip((*edits, *doses), (*single, *singleton_doses), strict=True):
        torch.testing.assert_close(expected[1:], actual)
