"""Compare posterior means and covariance-preserving LoRa spectral readouts."""

from math import log, sqrt

import torch
from torch import nn


def helmert():
    rows = torch.zeros(6, 6)
    rows[0] = 1. / sqrt(6.)
    for index in range(1, 6):
        rows[index, :index] = 1. / sqrt(index * (index + 1.))
        rows[index, index] = -index / sqrt(index * (index + 1.))
    return rows


def posterior_gain(signal, noise):
    return signal / (signal + noise).clamp_min(1e-12)


def physical_norm(edit, rms):
    return (edit * rms).abs().square().mean((-1, -2), keepdim=True).sqrt()


def match_dose(edit, match, rms):
    wanted = physical_norm(match, rms)
    measured = physical_norm(edit, rms)
    # The matched Wiener control only interpolates toward the raw observation.
    return edit * (wanted / measured.clamp_min(1e-10)).clamp_max(1.)


class VarianceTransport(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('basis', helmert(), persistent=False)
        frequencies = torch.arange(256)
        distance = torch.minimum(frequencies, 256 - frequencies)
        self.register_buffer('weak', (distance > 0) & (distance <= 32), persistent=False)
        self.register_buffer('outer', distance > 64, persistent=False)
        self.register_buffer('center', torch.zeros(6, 64, 2))
        self.register_buffer('eigenvectors', torch.eye(2).repeat(6, 64, 1, 1))
        self.register_buffer('variance', torch.zeros(6, 64, 2))

    def analysis(self, y):
        spectrum = torch.fft.fft(y, norm='ortho')
        return self.basis.to(spectrum) @ spectrum

    def synthesize(self, spectral_edit):
        edit = torch.fft.ifft(spectral_edit, norm='ortho')
        return self.basis.mT.to(edit) @ edit

    @torch.no_grad()
    def calibrate(self, y):
        sample = torch.view_as_real(self.analysis(y)[..., self.weak])
        self.center.copy_(sample.mean(0))
        parts = sample - self.center
        cov = torch.einsum('bmfr,bmfs->mfrs', parts, parts) / len(sample)
        cov[1:] = cov[1:].mean(0)
        isotropic = .5 * cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        cov = .95 * cov + .05 * isotropic[..., None, None] * torch.eye(2, device=y.device)
        eigenvalues, directions = torch.linalg.eigh(cov)
        self.eigenvectors.copy_(directions)
        self.variance.copy_(eigenvalues.clamp_min(1e-8))

    def observation_noise(self, spectrum):
        outside = spectrum[..., self.outer].abs().square()
        common = outside[:, :1].flatten(1).median(-1).values
        private = outside[:, 1:].flatten(1).median(-1).values
        # Complex circular Gaussian power has median ln(2) times its mean.
        component = torch.stack((common, private), dim=1) / (2. * log(2.))
        return torch.cat((component[:, :1], component[:, 1:].expand(-1, 5)), dim=1)

    def changes(self, y, rms):
        spectrum = self.analysis(y)
        noise = self.observation_noise(spectrum)
        gain = posterior_gain(self.variance, noise[:, :, None, None])
        value = torch.view_as_real(spectrum[..., self.weak]) - self.center
        projected = (value.unsqueeze(-2) @ self.eigenvectors).squeeze(-2)
        mean_edit = (gain - 1.) * projected
        transport_edit = (gain.sqrt() - 1.) * projected
        together = torch.stack((mean_edit, transport_edit))
        coordinates = (together.unsqueeze(-2) @ self.eigenvectors.mT).squeeze(-2).contiguous()
        weak_edits = torch.view_as_complex(coordinates)
        edits = torch.zeros_like(spectrum).expand(2, -1, -1, -1).clone()
        edits[..., self.weak] = weak_edits
        posterior, transport = self.synthesize(edits).unbind(0)
        matched, transport = match_dose(posterior, transport, rms), match_dose(transport, posterior, rms)

        # Retain the corresponding Wiener component in the factor controls.
        common = edits[0].clone()
        common[:, :1] = edits[1, :, :1]
        private = edits[0].clone()
        private[:, 1:] = edits[1, :, 1:]
        common, private = self.synthesize(torch.stack((common, private))).unbind(0)
        observed_rms = physical_norm(y, rms)
        edits = torch.stack((posterior, matched, transport, common, private))
        dose = physical_norm(edits, rms)
        cap = (.10 * observed_rms / dose.clamp_min(1e-10)).clamp_max(1.)
        actual = dose * cap / observed_rms.clamp_min(1e-10)
        return edits * cap, noise, actual[..., 0, 0].mT
