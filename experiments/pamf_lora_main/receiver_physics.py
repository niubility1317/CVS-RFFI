"""Generic receiver impairments for source-only LoRa restoration training."""

import math

import torch


def fractional_warp(wave, at):
    """Interpolate near the measured grid without wrapping across chirps."""
    size = wave.shape[-1]
    offsets = torch.arange(-3, 5, device=wave.device)
    index = at.floor().long()[..., None] + offsets
    distance = at[..., None] - index
    weights = torch.sinc(distance) * torch.sinc(distance / 4)
    weights = weights.masked_fill(distance.abs() >= 4, 0)
    weights = weights / weights.sum(-1, keepdim=True)
    period = 2 * (size - 1)
    folded = index.remainder(period)
    reflected = torch.minimum(folded, period - folded)
    samples = wave.gather(-1, reflected.flatten(-2)).unflatten(-1, reflected.shape[-2:])
    return (weights * samples).sum(-1)


def apply_generic_physics(base, shifted, *, generator):
    """Apply one sampled front end to both branches of the IQ difference."""
    batch, chirps, length = base.shape

    def draw(channels, bound):
        uniform = torch.rand((batch, channels, 1), device=base.device, generator=generator)
        return bound * (2 * uniform - 1)

    t = torch.arange(length, device=base.device, dtype=base.real.dtype)
    n = t - (length - 1) / 2
    u = n / length
    order = torch.arange(chirps, device=base.device, dtype=base.real.dtype)[None, :, None]
    delay = draw(1, 1.25) + draw(chirps, .35)
    rate_error = draw(1, .006)
    at = t + delay + rate_error * (n + length * order)
    carrier = draw(1, .45)
    local_carrier = draw(chirps, .18)
    phase = (draw(1, math.pi / 4) + draw(chirps, .2)
             + 2 * math.pi * (carrier * (u + order) + local_carrier * u)
             + draw(1, .30) * (2 * u).square())
    oscillator = torch.polar(torch.ones_like(phase), phase)
    image = torch.polar(draw(1, .06) + .06, draw(1, math.pi))
    active = torch.rand((batch, 1, 1), device=base.device, generator=generator) < .65

    def observe(wave):
        timed = fractional_warp(wave, at) * oscillator
        return torch.where(active, timed + image * timed.conj(), wave)

    return observe(base), observe(shifted)
