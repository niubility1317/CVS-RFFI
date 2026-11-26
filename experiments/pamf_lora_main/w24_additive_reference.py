"""Generic source-only pairs with distinct noiseless channel and source references."""

import math

import torch
from torch.nn.functional import pad

from .w24_probe_transport import phase_probe


TAPS = (2, 6, 13)


def action_pair(raw, probe, rng):
    """Apply one unreported short echo and slow phase drift to both source waves."""
    batch, samples = raw.shape
    dtype, device = raw.real.dtype, raw.device
    draw = torch.rand((batch, 3, 1), device=device, dtype=dtype, generator=rng)
    gains = torch.randn((batch, len(TAPS), 1), device=device, dtype=raw.dtype, generator=rng)
    gains = .12 * gains / gains.abs().clamp_min(1.)
    enable = draw[:, 0] >= .35
    t = torch.linspace(-.5, .5, samples, device=device, dtype=dtype)
    phase = enable * ((draw[:, 1] - .5) * 3. * t + (draw[:, 2] - .5) * t.square())
    rotation = torch.polar(torch.ones_like(phase), phase)

    def apply(wave):
        lags = torch.stack([pad(wave[..., :-delay], (delay, 0)) for delay in TAPS], dim=1)
        return (wave + enable * (lags * gains).sum(1)) * rotation

    return apply(raw), apply(probe)


def additive_noise(signal, rng):
    """Independent generic proper complex disturbances with a broad power range."""
    draw = torch.rand((len(signal), 3, 1), device=signal.device, dtype=signal.real.dtype, generator=rng)
    low, width = math.log(.02), math.log(.95 / .02)
    scale = signal.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    white = torch.randn(signal.shape, device=signal.device, dtype=signal.dtype, generator=rng)
    color = (draw[:, 1] < .45) * (1.2 * draw[:, 2] - .6)
    colored = (white + color * pad(white[..., :-1], (1, 0))) / (1. + color.square()).sqrt()
    return colored * scale * (low + width * draw[:, 0]).exp()


def paired_action_noise(raw, tracker, rng, weak, *, with_identity):
    probe = phase_probe(raw, weak).flatten(-2)
    signal, transported = action_pair(raw, probe, rng)
    first, second = (additive_noise(signal, rng) for _ in range(2))
    if with_identity:
        first[-2:] = second[-2:] = 0
        signal[-2:] = raw[-2:]
        transported[-2:] = probe[-2:]

    def aperture(waves):
        packet = torch.stack(waves, dim=1)
        return tracker.aperture(packet.flatten(0, 1)).unflatten(0, packet.shape[:2])

    observed = aperture((signal + first, signal + second, transported + first))
    return observed, aperture((raw, signal)), aperture((probe, transported))
