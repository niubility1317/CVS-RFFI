"""Place a source-derived phase probe before or after the same generic channel draw."""

import torch

from .data import WINDOW_START


def phase_probe(raw, weak):
    chirps = raw.unflatten(-1, (8, 1024))
    index = torch.arange(chirps.shape[-1], device=raw.device, dtype=raw.real.dtype)
    # Source W24 center spans samples WINDOW_START + k * 125 / 48 at the capture clock.
    position = ((index - WINDOW_START) * (48 / 125)).clamp(0, weak.shape[-1] - 1)
    left = position.long()
    right = (left + 1).clamp_max(weak.shape[-1] - 1)
    phase = weak.imag[..., left].lerp(weak.imag[..., right], position - left)
    taper = (torch.minimum(index, index[-1] - index) / WINDOW_START).clamp_max(1.)
    phase = phase.clamp(-.16, .16) * taper
    return chirps * torch.polar(torch.ones_like(phase), phase)


def probe_views(raw, clean, tracker, condition, rng, weak, *, through_channel, with_identity):
    chirps = raw.unflatten(-1, (8, 1024))
    modified = phase_probe(raw, weak)
    clean_probe = tracker.aperture(modified.flatten(-2))
    if through_channel:
        replay = torch.Generator(device=raw.device)
        replay.set_state(rng.get_state())
    a, b = [tracker.aperture(condition(chirps, rng).flatten(-2)) for _ in range(2)]
    if through_channel:
        probe = tracker.aperture(condition(modified, replay).flatten(-2))
    else:
        probe = a + (clean_probe - clean)
    if with_identity:
        a[-2:] = b[-2:] = clean[-2:]
        probe[-2:] = clean_probe[-2:]
    return torch.stack((a, b, probe), dim=1), clean_probe
