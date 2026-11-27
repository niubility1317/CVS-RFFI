"""Refine four complex directions at the W24 aperture with matched axial blocks."""

import torch
from torch import nn

from .w24_component_directions import component_angle, matched_component_edit
from .w24_native_center import CENTER
from .w24_packet_dura import CHIRPS
from .w24_peer_spectral_direction import SpectralContext


def rms(value):
    return value.abs().square().mean((-2, -1), keepdim=True).clamp_min(1e-24).sqrt()


class PostPeerRefiner(nn.Module):
    """Change the direction within each component while keeping the parent's four norms."""

    def __init__(self, *, peer_first):
        super().__init__()
        self.peer_first = peer_first
        width = 32
        self.embed = nn.Linear(8, width)
        self.frequency = nn.Sequential(*(SpectralContext(width, dilation) for dilation in (1, 4, 16)))
        self.peer_norm = nn.LayerNorm(width)
        self.peers = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = nn.Linear(width, 2)
        nn.init.normal_(self.output.weight, std=.002)
        nn.init.zeros_(self.output.bias)
        bins = torch.fft.fftfreq(CENTER)
        self.register_buffer('bins', bins, persistent=False)
        self.register_buffer('self_mask', torch.eye(CHIRPS, dtype=torch.bool), persistent=False)

    def across_frequency(self, x):
        batch = len(x)
        x = self.frequency(x.flatten(0, 1).transpose(1, 2))
        return x.transpose(1, 2).reshape(batch, CHIRPS, CENTER, -1)

    def across_chirps(self, x):
        batch, chirps, bins, width = x.shape
        with torch.autocast('cuda', enabled=False):
            tokens = self.peer_norm(x.float()).transpose(1, 2).reshape(batch*bins, chirps, width)
            pooled = self.peers(tokens, tokens, tokens, attn_mask=self.self_mask, need_weights=False)[0]
        return x + pooled.reshape(batch, bins, chirps, width).transpose(1, 2)

    def forward(self, observed, prior, phase, quality):
        if (observed.ndim != 3 or observed.shape[-2:] != (CHIRPS, CENTER) or not observed.is_complex()
                or prior.shape != observed.shape or phase.shape != observed.shape
                or quality.shape != (*observed.shape[:-1], 1)):
            raise ValueError('Expected aligned LoRa W24 center, prior edit, phase, and receive quality')
        frame = phase / phase.abs().clamp_min(1e-12)
        edit_scale = rms(prior)
        y = torch.fft.fft(observed * frame.conj() / rms(observed), norm='ortho')
        edit = torch.fft.fft(prior * frame.conj() / edit_scale, norm='ortho')
        power = y.abs().square()
        noise_floor = power[..., self.bins.abs() >= .35].median(-1, keepdim=True).values
        features = torch.stack((y.real, y.imag, edit.real, edit.imag, power.log1p(),
                                noise_floor.add(1e-8).log().div(6.).expand_as(power),
                                quality.expand_as(power), 2 * self.bins.expand_as(power)), dim=-1)
        mixed = self.training and y.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            tokens = torch.nn.functional.silu(self.embed(features))
            if self.peer_first:
                tokens = self.across_frequency(self.across_chirps(tokens))
            else:
                tokens = self.across_chirps(self.across_frequency(tokens))
            coefficients = self.output(tokens)
        correction = torch.fft.ifft(torch.view_as_complex(coefficients.float().contiguous()), norm='ortho')
        return matched_component_edit(prior, prior + edit_scale * frame * correction, frame)


def endpoint_risk(observed, source, prior, proposal, phase, noisy):
    """Compare actual aperture writeback with legal same-source clean and additive pairs."""
    with torch.no_grad():
        frame = phase / phase.abs().clamp_min(1e-12)
        clean_target = observed + prior
        target = torch.where(noisy[:, None, None], source, clean_target)
        target = target / rms(target)
        reference = torch.fft.fft(target * frame.conj(), norm='ortho')
        weak = (1. + reference.abs().square()).reciprocal()
        observation = observed / rms(observed)
        scale = (target - observation).abs().square().mean((-2, -1)).clamp_min(1e-3)

    restored = observed + proposal
    error = torch.fft.fft((restored / rms(restored) - target) * frame.conj(), norm='ortho')
    per_packet = (weak * error.abs().square()).mean((-2, -1)) / scale
    adjacent = (error[:, 1:] - error[:, :-1]).abs().square().mean((-2, -1)) / scale
    weak_risk = (per_packet * noisy).sum() / noisy.sum().clamp_min(1)
    delta_risk = (adjacent * noisy).sum() / noisy.sum().clamp_min(1)
    identity = (per_packet * ~noisy).sum() / (~noisy).sum().clamp_min(1)
    want = torch.where(noisy[:, None, None], source - observed, 0.)
    angle, _ = component_angle(proposal, want, phase)
    total = weak_risk + .35 * delta_risk + .50 * angle + .50 * identity
    return total, dict(weak_spectrum=weak_risk.detach(), chirp_difference=delta_risk.detach(),
                       component_direction=angle.detach(), identity=identity.detach())
