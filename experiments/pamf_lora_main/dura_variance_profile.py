"""Compute local time-domain variance multipliers from the two received guard bands."""

from math import log

import torch

from .channel_blind import corruption_score
from .dura_repair import real_iq
from .joint_unroll_source import coordinates
from .lapped_posterior import analysis


WINDOW, HOP = 64, 32


def short_time_bands(y):
    device, dtype = y.device, y.real.dtype
    window = ((torch.arange(WINDOW, device=device, dtype=dtype)+.5)*torch.pi/WINDOW).sin()
    frequency = torch.fft.fftshift(torch.fft.fftfreq(WINDOW, device=device, dtype=dtype))
    positive = (frequency >= .25) & (frequency <= .4375)
    negative = (frequency <= -.25) & (frequency >= -.4375)
    energy = analysis(y, window).abs().square()
    return torch.stack([energy[..., band].mean(-1) for band in (positive, negative)], dim=-1)


def relative_log(power):
    logged = power.clamp_min(1e-10).log()
    return logged-logged.mean(-2, keepdim=True)


def received_profile(y):
    sided = relative_log(short_time_bands(y))
    first, second = sided.unbind(-1)
    agree = (first*second).sum(-1) / (first.square().sum(-1)*second.square().sum(-1)).clamp_min(1e-10).sqrt()
    change = (.75*agree.clamp(0., 1.)[..., None] * sided.mean(-1)).clamp(-log(2.), log(2.))
    fraction = torch.arange(HOP, device=y.device, dtype=y.real.dtype)/HOP
    weights = (change[..., None]*(1.-fraction) + change.roll(-1, dims=-1)[..., None]*fraction)
    return weights.flatten(-2).exp(), agree, change


def restore(iq, tracker, model, threshold, *, shifted=False):
    y, phase, quality, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    count = int(active.sum())
    if not count:
        return iq, count
    y, phase, quality, rms = (field[active] for field in (y, phase, quality, rms))
    profile = received_profile(y)[0]
    if shifted:
        profile = profile.roll(y.shape[-1]//2, dims=-1)
    stages, *_ = model.stages(y, quality, rms, solve_profile=profile)
    output = iq.clone()
    edit = .5*(stages[-1]-y)
    output[active] += real_iq(edit*phase*rms)
    return output, count
