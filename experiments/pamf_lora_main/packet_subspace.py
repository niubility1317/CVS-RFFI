"""Recover shared and chirp-specific complex components of the same LoRa packet."""

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core

from .complex_vector import fft
from .denoise_components import bands
from .packet_data import CHIRPS


CAP = .12


def components(value):
    common = value.mean(dim=1, keepdim=True)
    return common.expand_as(value), value-common


def bounded_near(value):
    near = torch.stack(bands(value.shape[-1], value.device)).any(0)
    masked = value*near
    rms = masked.abs().square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12).sqrt()
    # Use one limit per packet so it preserves the chirp-axis decomposition.
    return masked*(CAP/rms).clamp_max(1.)


class PacketSubspace(nn.Module):
    def __init__(self, *, factorized):
        super().__init__()
        chirps = len(CHIRPS)
        self.spectral = Core(incoming=7*chirps, outgoing=2*chirps)
        self.temporal = Core(incoming=7*chirps, outgoing=2*chirps)
        self.factorized = factorized

    def parts(self, observed, baseline, quality):
        shift = lambda value: torch.fft.fftshift(fft(value), dim=-1)
        y, z = shift(observed), shift(baseline)
        q = quality.expand_as(y.real)
        fields = torch.stack((y.real, y.imag, z.real, z.imag, (y-z).real, (y-z).imag, q), dim=2)
        # The temporal head reads the whole packet in the same independent
        # carrier-tracking coordinates as the spectral head.
        local = torch.stack((observed.real, observed.imag, baseline.real, baseline.imag,
                             (observed-baseline).real, (observed-baseline).imag, q), dim=2)
        a = self.spectral(fields.flatten(1, 2)).unflatten(1, (len(CHIRPS), 2))
        b = self.temporal(local.flatten(1, 2)).unflatten(1, (len(CHIRPS), 2))
        first = torch.fft.ifftshift(torch.complex(a[:, :, 0], a[:, :, 1]), dim=-1)
        second = fft(torch.complex(b[:, :, 0], b[:, :, 1]))
        if self.factorized:
            first = components(first)[0]
            second = components(second)[1]
        return first, second

    def forward(self, observed, baseline, quality):
        first, second = self.parts(observed, baseline, quality)
        return bounded_near(.18*(first+second))


def target_vectors(baseline, signal):
    direction = bounded_near(fft(signal-baseline))
    return {'unrestricted': direction, 'factorized': direction}


def component_risk(estimated, target):
    residual = components(estimated-target)
    expected = components(target)
    losses = []
    for selected in bands(estimated.shape[-1], estimated.device):
        for error, truth in zip(residual, expected, strict=True):
            physical_error = error[..., selected].abs().square().mean(dim=(1, 2))
            energy = truth[..., selected].abs().square().mean(dim=(1, 2))
            losses.append((physical_error/(.002+energy)).log1p().mean())
    return torch.stack(losses).mean()
