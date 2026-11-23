"""Restrict the fine correction along receive-supported mirror directions."""

import torch

from experiments.pamf_unrolled_restore.denoise import spectral_companion

from .pilot_joint import PilotJoint


BANDS = ((1, 8), (9, 32), (33, 64), (65, 127))


class MirrorRefine(PilotJoint):
    def __init__(self, widths):
        super().__init__(equalize=False, widths=widths, companion='mirror')
        frequency = torch.fft.fftshift(torch.fft.fftfreq(256)).abs() * 256
        bands = torch.stack([(frequency >= lo) & (frequency <= hi) for lo, hi in BANDS])
        chirps = torch.arange(6)[:, None]
        self.register_buffer('band_masks', bands, persistent=False)
        self.register_buffer('first', (chirps + torch.tensor([1, 3, 5])) % 6, persistent=False)
        self.register_buffer('second', (chirps + torch.tensor([2, 4])) % 6, persistent=False)

    @torch.no_grad()
    def evidence(self, observed, quality):
        spec = spectrum(observed)
        mirror = spectral_companion(spec, 'mirror')
        weight = quality.square().clamp_min(.005)

        def estimate(indices):
            weights = weight[:, indices]
            return (mirror[:, indices] * weights).sum(2) / weights.sum(2)

        first, second = estimate(self.first), estimate(self.second)
        a = first.unsqueeze(-2) * self.band_masks
        b = second.unsqueeze(-2) * self.band_masks
        va = torch.linalg.vector_norm(a, dim=-1)
        vb = torch.linalg.vector_norm(b, dim=-1)
        agreement = (a.conj() * b).sum(-1).real / (va * vb).clamp_min(1e-12)
        jitter = (2. * self.band_masks.sum(-1)).rsqrt()
        support = (agreement.clamp(max=1) - jitter).clamp_min(0)
        confidence = support / (.2 + support)
        return .5 * (a + b), confidence

    def forward(self, observed, quality, template, *, controls=False):
        coarse, fine = super().forward(observed, quality, template)
        directions, confidence = self.evidence(observed, quality)
        limited = protect(coarse, fine, directions, confidence)
        if controls:
            dose_only = match_dose(observed, fine, limited, self.band_masks)
            return coarse, fine, limited, dose_only, confidence
        return coarse, limited


def spectrum(value):
    return torch.fft.fftshift(torch.fft.fft(value, norm='ortho'), dim=-1)


def wave(value):
    return torch.fft.ifft(torch.fft.ifftshift(value, dim=-1), norm='ortho')


def protect(coarse, fine, directions, confidence):
    increment = spectrum(fine - coarse).unsqueeze(-2)
    numerator = (directions.conj() * increment).sum(-1)
    denominator = directions.abs().square().sum(-1).clamp_min(1e-12)
    projected = directions * (confidence * numerator / denominator).unsqueeze(-1)
    return fine - wave(projected.sum(-2))


def match_dose(observed, ordinary, limited, masks):
    """Retain the ordinary band's edit direction at the protected edit length."""
    original = spectrum(ordinary - observed)
    protected = spectrum(limited - observed)
    ordinary_bands = original.unsqueeze(-2) * masks
    limited_bands = protected.unsqueeze(-2) * masks
    magnitude = torch.linalg.vector_norm(ordinary_bands, dim=-1, keepdim=True)
    target = torch.linalg.vector_norm(limited_bands, dim=-1, keepdim=True)
    scaled = ordinary_bands * (target / magnitude.clamp_min(1e-10))
    matched = torch.where(magnitude > 1e-10, scaled, limited_bands)
    return ordinary + wave((matched - ordinary_bands).sum(-2))
