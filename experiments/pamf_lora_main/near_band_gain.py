"""Calibrate weak amplitude along an existing single-packet LoRa recovery."""

import math

import torch

from .chirp_band_power import frequency_groups, packet_basis, spectrum
from .weak_signal_probe import observed_noise_rms


GAIN_CAP = math.sqrt(2) - 1


def power(value):
    return value.abs().square().mean((-2, -1))


def inputs(observed, restored):
    bins = frequency_groups(observed.shape[-1], observed.device)[0]
    basis = packet_basis(observed)
    raw = spectrum(observed, basis)
    output = spectrum(restored, basis)
    near_raw = power(raw[:, 1:, bins])
    near_output = power(output[:, 1:, bins])
    mean_mode = power(raw[:, :1, bins])
    alias = power(raw[:, 1:, (bins + observed.shape[-1] // 2) % observed.shape[-1]])
    added = observed_noise_rms(observed).square().mean((1, 2))
    coords = torch.stack((near_raw, near_output, mean_mode, added), dim=-1).clamp_min(1e-8).log()
    parity = ((near_raw - alias) / (near_raw + alias).clamp_min(1e-8)).unsqueeze(-1)
    return torch.cat((coords, parity), dim=-1), output, basis, bins


def protected_target(output, clean, basis, bins):
    """Use clean source IQ only to define a bounded, direction-aware amplitude target."""
    recovered = output[:, 1:, bins]
    reference = spectrum(clean, basis)[:, 1:, bins]
    reference_power = power(reference)
    output_power = power(recovered)
    overlap = (recovered.conj() * reference).real.mean((-2, -1)).clamp_min(0.)
    confidence = (overlap.square() / (reference_power * output_power).clamp_min(1e-12)).clamp(0., 1.)
    increase = (reference_power / overlap.clamp_min(1e-8) - 1).clamp(0., GAIN_CAP)
    return confidence * increase


def fit_coefficients(features, target):
    center = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-4)
    normalized = (features-center) / scale
    gram = normalized.T @ normalized / len(features)
    ridge = .10 * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    fit = normalized.T @ (target - target.mean()) / len(features)
    weight = torch.cholesky_solve(fit[:, None], torch.linalg.cholesky(gram + ridge)).squeeze(-1)
    return center, scale, weight, target.mean()


def increase(features, state):
    center, scale, weight, intercept = state
    return (((features - center) / scale) @ weight + intercept).clamp(0., GAIN_CAP)


def adjust(restored, output, basis, bins, factor):
    delta = torch.zeros_like(output)
    delta[:, 1:, bins] = factor[:, None, None] * output[:, 1:, bins]
    return restored + torch.fft.ifft(basis.T.to(output.dtype) @ delta, norm='ortho')
