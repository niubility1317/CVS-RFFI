"""Supervise the signed 9–32 kHz direction of the deployed half edit."""

import torch


WEIGHT = .10


def spectral_direction_risk(edit, target):
    def band(wave):
        spectrum = torch.fft.fft(wave, norm='ortho')
        return torch.cat((spectrum[..., 9:33], spectrum[..., -32:-8]), dim=-1)

    current, desired = band(edit), band(target)
    dims = (-2, -1)
    current_power = current.abs().square().mean(dims)
    desired_power = desired.abs().square().mean(dims)
    numerator = (current * desired.conj()).real.mean(dims)
    cosine = numerator * (current_power + 1e-5).rsqrt() * desired_power.clamp_min(1e-8).rsqrt()
    usable = desired_power > 1e-8
    return ((1 - cosine) * usable).sum() / usable.sum().clamp_min(1)
