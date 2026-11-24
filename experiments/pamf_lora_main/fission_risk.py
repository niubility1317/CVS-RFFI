"""Use one received LoRa packet to estimate relative risk between two restorers."""

import torch

from .variance_bound import spectral_contrast, variance_interval
from .weak_signal_probe import observed_noise_rms


PROBE_SEED = 2026116637
TAU = .5


def split_observation(observation, perturbation, tau=TAU):
    """Construct a prediction view and its paired evaluation view."""
    return observation + tau * perturbation, observation - perturbation / tau


def edit_risk(base, alternative, validation, weight, masks):
    """Mean quadratic coefficients over antithetic views in physical IQ units."""
    delta = alternative - base
    linear = ((base - validation).conj() * delta).real
    quadratic = delta.abs().square()

    def reduce(coefficient):
        return (coefficient * weight[:, None]).sum(dim=2).mean(dim=1) @ masks.T

    return reduce(linear), reduce(quadratic)


def optimal_mix(linear, quadratic):
    return torch.where(quadratic > 1e-12, (-linear / quadratic.clamp_min(1e-12)).clamp(0, 1), .5)


def band_masks(value):
    band = torch.fft.fftfreq(value.shape[-1], device=value.device).abs()
    return torch.stack(((band > 0) & (band <= 1/32), (band > 1/32) & (band <= 1/8), band > 1/8))


def estimate_noise(spectrum, quality, observed, model):
    contrast, weights = spectral_contrast(spectrum, quality)
    variance, _ = variance_interval(observed_noise_rms(observed).square(), contrast, weights, model.selectors)
    return variance @ model.bands


def axis_noise(spectrum, variance, masks):
    # A fixed independent antithetic pair makes the readout packet-order invariant.
    generator = torch.Generator().manual_seed(PROBE_SEED)
    probe = torch.randn((2, *spectrum.shape[1:]), dtype=spectrum.real.dtype, generator=generator)
    i, q = probe.to(device=spectrum.device).unbind(0)
    return torch.complex(i, q) * (variance * .5).sqrt() * masks.any(0)


def mix_readouts(observed, rotation, quality, rms, controls):
    """Score perturbed probes while applying the selected edit to the original input."""
    base_model = controls[0]
    spectrum = torch.fft.fft(observed, norm='ortho')
    bands = band_masks(spectrum)
    masks = bands.to(spectrum.real.dtype)
    variance = estimate_noise(spectrum, quality, observed, base_model)
    probe = axis_noise(spectrum, variance, bands)
    train, validation = split_observation(spectrum[:, None], torch.stack((probe, -probe), dim=1))
    perturbed = torch.fft.ifft(train, norm='ortho')
    arrived = torch.cat((observed[:, None], perturbed), dim=1)
    probe_q = (perturbed.mean(-1, keepdim=True).abs().square()
               / perturbed.abs().square().mean(-1, keepdim=True).clamp_min(1e-8)).clamp(0, 1)
    all_q = torch.cat((quality[:, None], probe_q), dim=1).flatten(0, 1)
    all_rot = rotation[:, None].expand_as(arrived).flatten(0, 1)
    waves = arrived.flatten(0, 1)

    def recover(model):
        refined = model(waves, all_q, all_rot)[-1]
        return torch.lerp(waves, refined, .5).unflatten(0, arrived.shape[:2])

    base, other = (recover(model) for model in controls)
    a = torch.fft.fft(base[:, 1:], norm='ortho')
    b = torch.fft.fft(other[:, 1:], norm='ortho')
    linear, quadratic = edit_risk(a, b, validation, rms.square(), masks)
    global_dose = optimal_mix(linear.sum(-1, keepdim=True), quadratic.sum(-1, keepdim=True))
    band_dose = optimal_mix(linear, quadratic)
    first, second = base[:, 0] - observed, other[:, 0] - observed
    vector = second - first
    narrow = torch.fft.ifft(torch.fft.fft(vector, norm='ortho') * (band_dose @ masks)[:, None],
                            norm='ortho')
    edits = (first, second, first + .5 * vector, first + global_dose[:, None] * vector, first + narrow)
    return edits, (global_dose, band_dose)
