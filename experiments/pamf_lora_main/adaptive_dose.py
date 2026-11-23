"""Adapt LoRa denoising dose from the currently observed spectral noise level."""

import torch

from experiments.pamf_unrolled_restore.denoise import Denoise

from .broadband_vector import CAP, partitions
from .complex_vector import fft
from .weak_signal_probe import observed_noise_rms


def dose_limit(noise_rms, adaptive):
    if adaptive:
        return (1.15*noise_rms).clamp(CAP,.98)
    return torch.full_like(noise_rms,CAP)


class AdaptiveDoseDenoise(Denoise):
    def __init__(self, *, adaptive):
        super().__init__()
        self.adaptive = adaptive

    def forward(self, received, coherence, *, packet=True, details=False):
        sigma = observed_noise_rms(received)
        scale = (.40+sigma).clamp(.45,1.25)
        proposal = super().forward(received/scale,coherence,packet=packet)*scale
        frequency = torch.fft.fftfreq(received.shape[-1],device=received.device)
        spectrum = fft(proposal)*(frequency!=0)
        magnitude = spectrum.abs().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
        limit = dose_limit(sigma,self.adaptive)
        noise = torch.fft.ifft(spectrum*(limit/magnitude).clamp_max(1.),norm='ortho')
        return (noise,sigma,limit) if details else noise


def dose_risk(estimate, received, signal, sigma):
    exact, predicted, reference = [fft(z) for z in (received-signal, estimate,signal)]
    mismatch = predicted-exact
    weights = sigma.new_tensor((.35,.35,.15,.15))
    errors, directions = [], []
    for band in partitions(exact.shape[-1],exact.device,True):
        wanted, actual = exact[...,band],predicted[...,band]
        natural_power = reference[...,band].abs().square().mean(-1)
        error = mismatch[...,band].abs().square().mean(-1)
        errors.append((error/(natural_power+.002)).log1p())
        eps = .002
        p = wanted.abs().square().mean(-1,keepdim=True)
        q = actual.abs().square().mean(-1,keepdim=True)
        shape = actual/(q+eps).sqrt()-wanted/(p+eps).sqrt()
        directions.append(shape.abs().square().mean(-1))
    by_chirp = (torch.stack(errors,-1)*weights).sum(-1)
    direction = (torch.stack(directions,-1)*weights).sum(-1)
    severe = ((sigma[...,0]-.40)/.40).clamp(0.,1.)
    focus = 1.+2.*severe
    objective = (focus*by_chirp).sum()/focus.sum() + .07*(severe*direction).mean()
    return torch.stack((by_chirp.mean(),objective))
