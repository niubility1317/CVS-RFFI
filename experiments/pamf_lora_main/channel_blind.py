"""Observation-only two-stage waveform recovery without a channel preset."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core
from experiments.pamf_unrolled_restore.denoise import Denoise

from .weak_signal_probe import observed_noise_rms


def zero_carrier(value):
    return value-value.mean(-1,keepdim=True)


def generic_observation(signal,generator):
    """Independent continuous-scale white and colored perturbations of raw source IQ."""
    shape = signal.shape
    device = signal.device
    real = torch.randn(shape,device=device,generator=generator)
    imag = torch.randn(shape,device=device,generator=generator)
    white = torch.complex(real,imag)
    tilt = 2.*torch.rand((*shape[:-2],1,1),device=device,generator=generator)-1.
    noise = white+tilt*white.roll(1,dims=-1)
    unit = noise.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
    strength = torch.rand((*shape[:-1],1),device=device,generator=generator).square()
    signal_rms = signal.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
    return signal+noise*(strength*signal_rms/unit)


class ChannelBlind(nn.Module):
    def __init__(self, *, widths=None, companion=None):
        super().__init__()
        coarse_config = {} if widths is None else {'widths': widths}
        self.coarse = Denoise(companion=companion, **coarse_config)
        self.prior = Core(incoming=8, widths=(24, 48, 72, 96) if widths is None else widths)

    def propose(self, observed, quality, *, mixed_conv=False):
        variance = observed_noise_rms(observed).square()
        coarse = observed-zero_carrier(self.coarse(observed,quality,packet=True,mixed_conv=mixed_conv))
        innovation = observed-coarse
        fields = torch.stack((observed.real,observed.imag,coarse.real,coarse.imag,
                              innovation.real,innovation.imag,quality.expand_as(observed.real),
                              variance.log1p().expand_as(observed.real)),dim=2)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed_conv):
            change = self.prior(fields.flatten(0,1)).unflatten(0,observed.shape[:2])
        if mixed_conv:
            change = change.float()
        proposed = coarse+zero_carrier(torch.complex(change[:,:,0],change[:,:,1]))
        return coarse,proposed,variance

    def forward(self, observed, quality):
        coarse,proposed,variance = self.propose(observed,quality)
        # Identity observation operator, with packet averaging as the prior scale.
        trust = variance/(variance+1./observed.shape[1])
        refined = observed+trust*(proposed-observed)
        return coarse,refined


def corruption_score(tracked):
    return observed_noise_rms(tracked)[...,0].median(dim=1).values
