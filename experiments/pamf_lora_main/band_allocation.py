"""Scale three LoRa non-carrier bands with a common physical edit budget."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from .cross_frequency import CrossFrequencyPrior
from .joint_spectrum import limit, split
from .received_update import ReceivedUpdate


def band_masks(device=None):
    bins = torch.arange(256, device=device)
    offset = torch.minimum(bins, 256 - bins)
    return torch.stack(((offset >= 1) & (offset <= 8),
                        (offset >= 9) & (offset <= 32), offset >= 33))


def fft(wave):
    return torch.fft.fft(wave, norm='ortho')


def ifft(spectrum):
    return torch.fft.ifft(spectrum, norm='ortho')


def band_energy(spectrum, rms, bands):
    return (spectrum * rms).abs().square().sum(-2) @ bands.T


def apply_gain(spectrum, gains, bands):
    return spectrum * (gains @ bands).unsqueeze(-2)


def equal_budget(energy, proposed, active):
    """Return bounded gains by solving a three-coordinate clipped scale."""
    total = energy.sum(-1, keepdim=True).clamp_min(1e-20)
    base = energy / total
    proposal = torch.maximum(proposed / total, 1e-6 * base)
    lower = torch.where(active, .25 * base, base)
    upper = torch.where(active, 2.25 * base, base)
    denominator = proposal.clamp_min(1e-20)
    points = torch.cat((lower / denominator, upper / denominator), -1).sort(-1).values
    # Pick the segment with the target budget. Use the analytic affine
    # inverse within it so active gains keep gradients through the solve.
    totals = (points.unsqueeze(-1) * proposal.unsqueeze(-2)).clamp(
        lower.unsqueeze(-2), upper.unsqueeze(-2)).sum(-1)
    target = base.sum(-1, keepdim=True)
    index = (totals < target).sum(-1, keepdim=True).clamp(1, points.shape[-1] - 1)
    left, right = points.gather(-1, index - 1), points.gather(-1, index)
    middle = .5 * (left + right)
    scaled = middle * proposal
    moving = active & (scaled > lower) & (scaled < upper)
    fixed = torch.where(scaled < lower, lower, upper)
    fixed_sum = torch.where(moving, 0., fixed).sum(-1, keepdim=True)
    slope = torch.where(moving, proposal, 0.).sum(-1, keepdim=True)
    multiplier = torch.where(slope > 0, (target - fixed_sum) / slope.clamp_min(1e-20), middle)
    budget = (multiplier * proposal).clamp(lower, upper)
    return torch.where(active & (base > 0), (budget / base.clamp_min(1e-20)).clamp_min(.25).sqrt(), 1.)


def tracked_anchor(prior, y, rms, noise=None):
    pooled = prior.spectrum(y)
    measured = prior.noise(pooled)
    estimate = measured if noise is None else noise
    return .5 * limit(prior.transport(pooled, estimate), y, rms, y.real.new_tensor(.30)), measured


class BandAllocation(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior = CrossFrequencyPrior()
        self.coarse = nn.Sequential(nn.Linear(14, 48), nn.SiLU(), nn.Linear(48, 48),
                                    nn.SiLU(), nn.Linear(48, 2))
        self.fine = nn.Sequential(nn.Linear(18, 48), nn.SiLU(), nn.Linear(48, 48),
                                  nn.SiLU(), nn.Linear(48, 2))
        self.register_buffer('bands', band_masks().float(), persistent=False)
        self.register_buffer('source_variance', torch.zeros(6, 256))
        self.register_buffer('noise_baseline', torch.full((2,), .05))
        self.register_buffer('margins', torch.zeros(2))
        self.register_buffer('clean_level', torch.zeros(()))

    @torch.no_grad()
    def learn_source_statistics(self, waves):
        self.prior.calibrate(waves)
        variance = (self.prior.joint_axes.square() @ self.prior.joint_variance.unsqueeze(-1))
        power = variance.squeeze(-1).reshape(2, 255, 2).sum(-1)
        groups = torch.cat((power[:1], power[1:].expand(5, -1)), 0)
        self.source_variance.copy_(F.pad(self.prior.basis.T.square() @ groups, (1, 0)))

    def observe(self, y, quality, rms, *, action_enabled=True, adaptive_noise=True):
        substitute = None if adaptive_noise else self.noise_baseline.expand(len(y), -1)
        anchor, measured = tracked_anchor(self.prior, y, rms, substitute)
        noise = measured if adaptive_noise else substitute
        sigma = noise.mean(-1).reshape(-1, 1, 1)
        action, _, mismatch = ReceivedUpdate.estimate(y, sigma)
        if not action_enabled:
            action = torch.ones_like(action)
            mismatch = torch.zeros_like(mismatch)
        direction = fft(anchor)
        energy = band_energy(direction, rms, self.bands)
        observed = fft(y)
        margin = self.prior.center.unflatten(-1, (255, 2)).contiguous()
        center = self.prior.basis.T.to(observed) @ torch.view_as_complex(margin)
        centered = observed - F.pad(center, (1, 0))
        expected = (self.source_variance * rms.square()).sum(-2) @ self.bands.T
        observed_power = band_energy(centered, rms, self.bands)
        excess = (observed_power - expected).clamp_min(0.)
        components = torch.stack((centered.mean(-2, keepdim=True).expand_as(centered),
                                  centered - centered.mean(-2, keepdim=True)), 1)
        component_power = band_energy(components, rms.unsqueeze(1), self.bands)
        mode_noise = torch.cat((noise[:, :1], noise[:, 1:].expand(-1, 5)), 1) * 2.
        white = (mode_noise @ self.prior.basis.square() * rms.squeeze(-1).square()).sum(-1, keepdim=True)
        white = white * self.bands.sum(-1)
        evidence = .5 * (white + excess)
        usable = (energy > 1e-6 * energy.sum(-1, keepdim=True)) & torch.isfinite(evidence) & (evidence > 0)
        shape = energy / energy.sum(-1, keepdim=True).clamp_min(1e-10)
        powers = torch.stack((energy, expected, observed_power, excess, white, *component_power.unbind(1)), -1)
        globals_ = torch.cat((quality.mean((1, 2))[:, None], noise.log1p(),
                              mismatch.mean((1, 2))[:, None]), -1)
        identity = torch.eye(3, device=y.device).expand(len(y), -1, -1)
        features = torch.cat((powers.log1p(), globals_[:, None].expand(-1, 3, -1), identity), -1)
        return direction, energy, evidence, usable, features, action, noise, mismatch, shape

    def step(self, network, features, energy, usable, stage):
        estimate, log_scale = network(features).unbind(-1)
        # A residual around the unchanged edit keeps the initial proposal local.
        estimate = 1. + estimate
        scale = .02 + F.softplus(log_scale)
        # Bound the final gains in the projection. Clipping every proposal to
        # those bounds first would erase relative evidence on strongly hit packets.
        gain = (estimate - self.margins[stage] * scale).clamp(.25, 12.)
        return equal_budget(energy, energy * gain.square(), usable), estimate, scale

    def stages(self, y, rms, state, *, use_residual=True):
        direction, energy, _, usable, features, action, _, _, _ = state
        coarse, first, first_error = self.step(self.coarse, features, energy, usable, 0)
        first_edit = ifft(apply_gain(direction, coarse, self.bands))
        forward = y - action * (y + first_edit)
        forward_spectrum = fft(forward)
        residual = band_energy(forward_spectrum, rms, self.bands)
        alignment = (((forward_spectrum * direction.conj()).real * rms.square()).sum(-2) @ self.bands.T)
        alignment = (alignment / energy.clamp_min(1e-8)).clamp(-20., 20.)
        if not use_residual:
            residual = torch.zeros_like(residual)
            alignment = torch.zeros_like(alignment)
        refine = torch.stack((coarse, first, residual.log1p(), alignment), -1)
        final, mean, error = self.step(self.fine, torch.cat((features, refine), -1), energy, usable, 1)
        second_edit = ifft(apply_gain(direction, final, self.bands))
        risk = ((first, first_error), (mean, error))
        return first_edit, second_edit, final, risk

    def target_projection(self, correction, rms, state):
        direction, energy, _, usable, *_ = state
        h = (((fft(correction) * direction.conj()).real * rms.square()).sum(-2) @ self.bands.T)
        return torch.where(usable, h / energy.clamp_min(1e-8), 0.)

    def wave_risk(self, difference, reference):
        # Keep a quiet band from disappearing under the much wider far band.
        unity = difference.real.new_ones(())
        actual = band_energy(fft(difference), unity, self.bands)
        expected = band_energy(fft(reference), unity, self.bands)
        return (actual / (expected + .002 * self.bands.sum(-1))).log1p().mean()

    def difference_risk(self, measured, expected):
        real, wanted = split(measured), split(expected)
        return self.wave_risk(real[:, :1] - wanted[:, :1], wanted[:, :1]) + self.wave_risk(
            real[:, 1:] - wanted[:, 1:], wanted[:, 1:])
