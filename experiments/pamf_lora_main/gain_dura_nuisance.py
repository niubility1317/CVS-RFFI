"""Packet-rate generic training stresses independent of recorded practical views."""
import math

import torch
from torch.nn import functional as F

from experiments.pamf_lora_main.w24_native_dura import capture_scale


def add_packet_nuisance(clean, observed, rng):
    """Add receiver phase drift and partially coherent interference to one source pair."""
    batch, length = observed.shape
    blocks = 8
    width = length // blocks
    options = dict(device=observed.device, dtype=observed.real.dtype, generator=rng)
    draws = torch.rand((batch, 5), **options)
    changed = (observed != clean).any(-1, keepdim=True)
    active = changed & (draws[:, :1] < .75)
    time = torch.linspace(-.5, .5, length, device=observed.device, dtype=observed.real.dtype)
    walk = .45 * torch.randn((batch, 1, blocks), **options).cumsum(-1)
    drift = F.interpolate(walk, size=length, mode='linear', align_corners=True)[:, 0]
    phase = 2 * math.pi * ((draws[:, 1:2] - .5) * 3. * blocks * time
                          + (draws[:, 2:3] - .5) * 6. * time.square()) + drift
    rotated = torch.polar(torch.ones_like(phase), phase) * observed

    interference = torch.randn((batch, 1, width), device=observed.device, dtype=observed.dtype, generator=rng)
    colored = interference + .7 * interference.roll(1, -1) - .25 * interference.roll(3, -1)
    independent = torch.randn((batch, blocks, width), device=observed.device, dtype=observed.dtype, generator=rng)
    tone_phase = torch.rand((batch, blocks, 1), **options) * (2 * math.pi)
    coherence = (.35 + .50 * draws[:, 3:4])[:, None]
    interference = (coherence.sqrt() * colored * torch.polar(torch.ones_like(tone_phase), tone_phase)
                    + (1. - coherence).sqrt() * independent).flatten(1)
    level = 10. ** ((-12. + 24. * draws[:, 4:5]) / 20.)
    disturbed = rotated + level * interference / capture_scale(interference)
    scale = capture_scale(disturbed)
    return torch.where(active, clean / scale, clean), torch.where(active, disturbed / scale, observed)
