"""Refine the common complex spectrum while fixing the measured correction dose."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core

from .complex_vector import fft
from .denoise_components import bands
from .near_spectrum import same_size


def packet_mean(value, rms):
    weight = rms.square()
    return (value*weight).sum(1)/weight.sum(1).clamp_min(1e-8)


def common_inputs(observed, baseline, prior, q, rms):
    wave = fft(observed)
    y = packet_mean(wave, rms)
    b = packet_mean(fft(baseline), rms)
    anchor = packet_mean(prior, rms)
    spread = packet_mean((wave-y[:, None]).abs().square(), rms).clamp_min(1e-8).sqrt()
    reliability = packet_mean(q, rms).expand_as(y.real)
    position = torch.fft.fftfreq(y.shape[-1], device=y.device).expand_as(y.real)
    fields = torch.stack((y.real, y.imag, b.real, b.imag, anchor.real, anchor.imag,
                          spread.log1p(), y.abs().log1p(), anchor.abs().log1p(),
                          reliability, position), dim=1)
    return torch.fft.fftshift(fields, dim=-1), anchor


class SharedSpectrum(nn.Module):
    def __init__(self):
        super().__init__()
        self.refiner = Core(incoming=11, outgoing=2, widths=(24, 48, 72, 96))

    def forward(self, features, anchor):
        prediction = self.refiner(features)
        change = torch.fft.ifftshift(torch.complex(prediction[:, 0], prediction[:, 1]), dim=-1)
        refined = anchor.clone()
        for band in bands(anchor.shape[-1], anchor.device):
            ref = anchor[..., band]
            update = change[..., band]
            rms = ref.abs().square().mean(-1, keepdim=True).sqrt()
            step = .90*update*(1.+update.abs().square().mean(-1, keepdim=True)).rsqrt()
            refined[..., band] = same_size(ref+rms*step, ref)
        return refined


def match_common(anchor, exact):
    result = anchor.clone()
    for band in bands(anchor.shape[-1], anchor.device):
        result[..., band] = same_size(exact[..., band], anchor[..., band])
    return result


def errors(prediction, exact):
    ordinary, fine = [], []
    for band in bands(exact.shape[-1], exact.device):
        goal = exact[..., band]
        power = goal.abs().square()
        mean = power.mean(-1, keepdim=True)
        error = (prediction[..., band]-goal).abs().square()
        weight = (power+.25*mean+.001).reciprocal()
        weight = .30+.70*weight/weight.mean(-1, keepdim=True)
        ordinary.append((error.mean(-1, keepdim=True)/(.003+mean)).log1p().mean())
        fine.append(((weight*error).mean(-1, keepdim=True)/(.003+mean)).log1p().mean())
    return torch.stack((torch.stack(ordinary).mean(), torch.stack(fine).mean()))
