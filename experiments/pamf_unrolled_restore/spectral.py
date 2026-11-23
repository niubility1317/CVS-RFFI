"""Add an independently learned complex spectral prior at each R17 step."""

from copy import deepcopy

import torch
from torch import nn

from experiments.pamf_reception_restore.models import Core
from experiments.pamf_restoration.models import complex_iq, pair

from .models import UnrolledRestorer


class SpectralPrior(nn.Module):
    def __init__(self, temporal, reference):
        super().__init__()
        self.temporal = deepcopy(temporal)
        self.spectral = Core(incoming=9, outgoing=4)
        self.register_buffer('reference', None if reference is None else reference.detach().clone())

    def forward(self, value):
        # R17 supplies received, coarse, current and back-projected innovation.
        iq = torch.complex(value[:, :8:2], value[:, 1:8:2])
        framed = iq if self.reference is None else iq * self.reference.conj()
        spectrum = torch.fft.fft(framed, norm='ortho')
        scale = (spectrum.abs().square().mean(-1, keepdim=True) + 1e-6).sqrt()
        features = spectrum / scale
        features = torch.cat((features.real, features.imag, value[:, 8:9]), dim=1)
        weights = .25 * self.spectral(features).tanh()
        observed_weight = complex_iq(weights[:, :2])
        innovation_weight = complex_iq(weights[:, 2:])
        detail = observed_weight * spectrum[:, 0] + innovation_weight * spectrum[:, 3]
        correction = torch.fft.ifft(detail, norm='ortho')
        if self.reference is not None:
            correction = correction * self.reference
        return self.temporal(value) + pair(correction)


class SpectralUnroll(UnrolledRestorer):
    """Keep the frozen R17 action and add three zero-initialized spectral heads."""

    def __init__(self, r17, reference=None):
        if not isinstance(r17, UnrolledRestorer) or r17.candidate != 'unroll_unshared':
            raise ValueError('An unshared three-step R17 parent is required')
        if reference is not None and (reference.shape != (256,) or not reference.is_complex()):
            raise ValueError('The optional frame must be a complex length-256 reference')
        nn.Module.__init__(self)
        self.candidate = r17.candidate
        self.frozen = deepcopy(r17.frozen).requires_grad_(False).eval()
        self.prior = nn.ModuleList(SpectralPrior(stage, reference) for stage in r17.prior)


def paired_loss(stages, clean, reference=None):
    """Balance complex errors in low-energy bands and retain whole-waveform fidelity."""
    estimate = complex_iq(torch.cat(stages)).reshape(3, len(clean), -1)
    target = complex_iq(clean.detach())
    error = estimate - target
    energy = target.abs().square().mean(-1).clamp_min(1e-4)
    iq_loss = error.abs().square().mean(-1) / energy
    if reference is not None:
        frame = reference.conj()
        target = target * frame
        error = error * frame
    truth = torch.fft.fft(target, norm='ortho').abs().square()
    residual = torch.fft.fft(error, norm='ortho').abs().square()
    bins = torch.fft.fftfreq(clean.shape[-1], device=clean.device).abs()
    groups = ((bins > 0) & (bins <= 1 / 32), (bins > 1 / 32) & (bins <= 1 / 8), bins > 1 / 8)
    band_losses = [residual[..., band].mean(-1) / (truth[..., band].mean(-1) + .03 * energy)
                   for band in groups]
    stage_loss = iq_loss + .4 * torch.stack(band_losses).mean(0)
    return .5 * stage_loss[-1] + .5 * stage_loss.mean(0)
