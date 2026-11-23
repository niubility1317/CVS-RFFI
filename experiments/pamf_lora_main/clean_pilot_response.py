"""Protect repeatable clean preamble responses during LoRa recovery."""

import torch
from torch import nn

from .crossfit_pilot import CrossfitPilot, estimate_shared, regularized_inverse


def relative_response(response):
    band = torch.fft.fftfreq(response.shape[-1], device=response.device).abs() <= .25
    anchor = response[..., band].mean(dim=-1, keepdim=True)
    return response * anchor.conj() / anchor.abs().square().clamp_min(1e-8)


def complex_vector(response):
    return torch.cat((response.real, response.imag), dim=-1)


class CleanPilotResponse(nn.Module):
    """Suppress relative response directions with high clean-source variance."""

    def __init__(self, length=256):
        super().__init__()
        self.register_buffer('mean', torch.zeros(2 * length))
        self.register_buffer('shrink', torch.eye(2 * length))

    @torch.no_grad()
    def fit(self, responses):
        vectors = complex_vector(relative_response(responses).squeeze(1) - 1).cpu()
        mean = vectors.mean(dim=0)
        centered = vectors - mean
        covariance = centered.T @ centered / (len(vectors) - 1)
        values, basis = torch.linalg.eigh(covariance)
        scale = (.04 * values[-1]).clamp_min(1e-6)
        factors = scale / (scale + values.clamp_min(0))
        self.mean.copy_(mean)
        self.shrink.copy_((basis * factors) @ basis.T)

    def forward(self, response, support):
        relative = relative_response(response)
        vector = complex_vector(relative.squeeze(1) - 1) - self.mean
        real, imag = (vector @ self.shrink).chunk(2, dim=-1)
        delta = torch.complex(real, imag)
        limited = delta * (.45 / delta.abs().clamp_min(1e-8)).clamp_max(1.)
        return 1 + torch.where(support[:, None, None] > 0, limited[:, None], 0.)


def preserve_corroborated_edit(observed, coarse, refined, corroboration):
    """Retain an aligned coarse component up to one measured engineering edit."""
    dimensions = (-1, -2)
    energy = corroboration.abs().square().sum(dim=dimensions, keepdim=True)
    conjugate = corroboration.conj()

    def projection(value):
        return ((value - observed) * conjugate).real.sum(dim=dimensions, keepdim=True)

    coarse_projection = projection(coarse)
    floor = coarse_projection.clamp_min(0.).minimum(energy)
    deficit = (floor - projection(refined)).clamp_min(0.)
    coefficient = torch.where(coarse_projection > 0, deficit / energy.clamp_min(1e-8), 0.)
    return refined + coefficient.clamp_max(1.) * corroboration


class ProtectedPilot(CrossfitPilot):
    def __init__(self, widths, *, anchor):
        super().__init__(widths)
        self.calibration = CleanPilotResponse()
        self.anchor = anchor

    def forward(self, observed, quality, template, *, diagnostics=False):
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        coarse, prior, variance = self.propose(observed, quality, mixed_conv=mixed)
        raw, support = estimate_shared(observed, template, quality, variance)
        transfer = self.calibration(raw, support)
        refined = regularized_inverse(observed, prior, template, variance, transfer)
        if self.anchor or diagnostics:
            engineering = regularized_inverse(observed, observed, template, variance, transfer)
        if self.anchor:
            refined = preserve_corroborated_edit(observed, coarse, refined, engineering - observed)
        if diagnostics:
            return coarse, refined, engineering, support
        return coarse, refined
