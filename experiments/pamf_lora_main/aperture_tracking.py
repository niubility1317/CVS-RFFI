"""Estimate LoRa dechirped frequency from two receive apertures."""

import math

import torch
from torch import nn

from .dura_repair import complex_iq
from .local_phase import rotation


FFT = 2048


def wrap(angle):
    return (angle+math.pi).remainder(2*math.pi)-math.pi


def center(angles):
    origin = angles[:, :1]
    unwrapped = origin+wrap(angles-origin)
    return wrap(unwrapped.median(1,keepdim=True).values)


def carrier_from_frame(frame, chirp):
    wave = frame*chirp.conj()
    return (wave[...,1:]*wave[...,:-1].conj()).mean(-1).angle()


def spectral_peak(profile):
    index = profile.argmax(-1,keepdim=True)
    middle = profile.gather(-1,index)
    previous = profile.gather(-1,(index-1).remainder(FFT))
    following = profile.gather(-1,(index+1).remainder(FFT))
    curvature = previous-2*middle+following
    interpolation = (.5*(previous-following)/curvature.clamp_max(-1e-6)).clamp(-.5,.5)
    return wrap((index+interpolation).squeeze(-1)*(2*math.pi/FFT))


class ApertureTracking(nn.Module):
    """Fuse two local periodograms, and shrink between chirps only when uncertain."""

    def __init__(self):
        super().__init__()
        self.register_buffer('window',torch.hann_window(128,periodic=False),persistent=False)
        self.register_buffer('time',torch.arange(256)-127.5,persistent=False)

    def frequencies(self, iq, chirp, legacy):
        dechirped = complex_iq(iq)*chirp.conj()
        windows = dechirped.unflatten(-1,(2,128))*self.window
        spectrum = torch.fft.fft(windows,n=FFT).abs().square()
        profile = (spectrum/(spectrum.mean(-1,keepdim=True)+1e-6)).clamp_min(1e-4).log()
        halves = spectral_peak(profile)
        candidate = spectral_peak(profile.mean(-2))
        # A whole-chirp peak anchors the aperture estimate in the event of
        # nearly unrelated local peaks. The cap is one dechirped 256-point bin.
        half_disagreement = wrap(halves[...,0]-halves[...,1]).abs()
        bin_width = 2*math.pi/256
        reliability = 1./(1.+(half_disagreement/bin_width).square())
        split = wrap(legacy+reliability*wrap(candidate-legacy).clamp(-bin_width,bin_width))
        packet = center(split)
        uncertainty = (.5*half_disagreement).square()
        spread = (1.4826*wrap(split-packet).abs().median(1,keepdim=True).values).square()
        observed_between = (spread-uncertainty.median(1,keepdim=True).values).clamp_min(0.)
        # A variance floor allows genuine relative motion when aperture
        # slopes agree, including in packets with a tight carrier cluster.
        strength = (uncertainty/(uncertainty+observed_between+(.15*bin_width)**2)).clamp_max(.65)
        adaptive = wrap(split-strength*wrap(split-packet))
        return split,adaptive,uncertainty.sqrt(),strength

    def translate(self, received, frequency, fraction=1.):
        relative = wrap(frequency-center(frequency)).clamp(-2*math.pi/256,2*math.pi/256)
        return received*rotation(-fraction*relative[...,None]*self.time)

    def frame(self, received, frequency, chirp):
        frame = chirp*rotation(frequency[...,None]*self.time)
        observed = received*frame.conj()
        dc = observed.mean(-1,keepdim=True)
        phase = torch.where(dc.abs()>1e-8,dc/dc.abs().clamp_min(1e-8),1.)
        rms = received.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
        return observed*phase.conj()/rms,frame*phase*rms
