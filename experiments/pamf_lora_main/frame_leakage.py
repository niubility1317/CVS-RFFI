"""Separate gain, rotation offset and rotation slope in LoRa private coordinates."""

from itertools import product

import torch

from .dura_repair import complex_iq
from .joint_unroll_source import coordinates
from .weak_signal_probe import observed_noise_rms


SUBSETS = tuple(''.join(bits) for bits in product('01', repeat=3))
RECEIVED = ('tracked','local_slope','pooled_slope','debiased_rms','pooled_debiased')


def carrier_peak(scores, grid):
    index = scores.argmax(dim=-1, keepdim=True)
    value = scores.gather(-1,index)
    a = scores.gather(-1,(index-1).clamp_min(0))
    b = scores.gather(-1,(index+1).clamp_max(len(grid)-1))
    second = a - 2*value + b
    inner = (index > 0) & (index < len(grid)-1) & (second < -1e-10)
    fraction = (.5*(a-b)/second.clamp_max(-1e-10)).clamp(-.5,.5)
    return grid[index] + (grid[1]-grid[0])*torch.where(inner,fraction,0.)


def slope_refinements(wave, nominal, reference):
    """Use disagreement between interleaved pilot samples to regularize local slopes."""
    points = wave.shape[-1]
    time = torch.linspace(-1,1,points,device=wave.device,dtype=wave.real.dtype)
    grid = torch.linspace(-.8,.8,17,device=wave.device,dtype=wave.real.dtype)
    angle = grid[:,None]*time
    kernel = torch.polar(1-time.square().expand_as(angle),-angle)
    even = wave[...,::2] @ kernel[:,::2].T
    odd = wave[...,1::2] @ kernel[:,1::2].T
    first = carrier_peak(even.abs().square(),grid)
    second = carrier_peak(odd.abs().square(),grid)
    local = carrier_peak((even+odd).abs().square(),grid)
    carrier = nominal*reference.conj()
    initial = (carrier[...,1:]*carrier[...,:-1].conj()).mean(-1,keepdim=True).angle()*(points-1)/2
    absolute = initial+local
    center = absolute.median(1,keepdim=True).values
    # Inflate split error by the packet median to avoid exact-zero uncertainty.
    noise = ((first-second)*.5).square()
    proxy = .5*(noise+noise.median(1,keepdim=True).values)
    between = (absolute-center).square().mean(1,keepdim=True)
    latent = (between-proxy.mean(1,keepdim=True)).clamp_min(0.)
    share = proxy/(proxy+latent).clamp_min(1e-9)
    pooled = (local + share*(center-absolute)).clamp(-.8,.8)
    return nominal_frame(wave,nominal,local,time),nominal_frame(wave,nominal,pooled,time)


def nominal_frame(wave, nominal, slope, time):
    angle = slope*time
    rotation = torch.polar(torch.ones_like(angle),angle)
    tone = (wave*rotation.conj()).mean(-1,keepdim=True)
    phase = torch.where(tone.abs()>1e-8,tone/tone.abs().clamp_min(1e-8),1.)
    return nominal*rotation*phase


def received_frames(tracker, iq):
    wave,rotation,_,rms = coordinates(tracker,iq)
    local,pooled = slope_refinements(wave,rotation,tracker.reference.chirp)
    local_pooled = complex_iq(iq)*pooled.conj()/rms
    noise = observed_noise_rms(wave).square()
    pooled_noise = observed_noise_rms(local_pooled).square()
    adjusted = rms*(1-noise).clamp(.2,1.).sqrt()
    pooled_adjusted = rms*(1-pooled_noise).clamp(.2,1.).sqrt()
    return dict(zip(RECEIVED,((rotation,rms),(local,rms),(pooled,rms),(rotation,adjusted),
                              (pooled,pooled_adjusted)),strict=True))


def frame_factors(actual, reference):
    observed_rotation, observed_rms = actual
    ideal_rotation, ideal_rms = reference
    relative = ideal_rotation*observed_rotation.conj()
    center = relative[...,relative.shape[-1]//2:relative.shape[-1]//2+1]
    return center,relative*center.conj(),ideal_rms/observed_rms


def private_terms(nominal, multiplier):
    """Return private truth, leaked common content and changed private content."""
    def private(value):
        return value-value.mean(dim=1,keepdim=True)
    common = nominal.mean(dim=1,keepdim=True)
    signal = nominal-common
    return signal,private(multiplier*common),private((multiplier-1)*signal)
