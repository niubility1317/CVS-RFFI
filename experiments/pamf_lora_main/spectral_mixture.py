"""Condition a LoRa cross-frequency prior on disjoint received chirp groups."""

from math import log, sqrt

import torch

from .cross_frequency import CrossFrequencyPrior
from .variance_transport import match_dose, physical_norm, posterior_gain


COMPONENTS = 8
ROUTING_DIMENSIONS = 12
EM_STEPS = 64


def log_likelihood(samples, center, variance):
    return -.5 * ((samples[:, None] - center).square() / variance + variance.log()).sum(-1)


def gaussian_mixture(samples, *, seed=2026112419):
    """Fit the routing distribution with one full-batch EM schedule."""
    rng = torch.Generator(device=samples.device).manual_seed(seed)
    selected = torch.randperm(len(samples), generator=rng, device=samples.device)[:COMPONENTS]
    covariance = samples.var(0, unbiased=False).clamp_min(1e-7)
    center = samples[selected].clone()
    variance = covariance.expand_as(center).clone()
    weights = samples.new_full((COMPONENTS,), 1. / COMPONENTS)
    floor = .03 * covariance
    second_moment = samples.square()
    trajectory = []
    for step in range(EM_STEPS):
        score = log_likelihood(samples, center, variance) + weights.log()
        responsibilities = score.softmax(dim=-1)
        count = responsibilities.sum(0).clamp_min(1e-5)
        center = (responsibilities.T @ samples) / count[:, None]
        variance = (responsibilities.T @ second_moment / count[:, None] - center.square()).maximum(floor)
        weights = count / count.sum()
        if step in (0, 7, 15, 31, 63):
            trajectory.append(score.logsumexp(dim=1).mean())
    responsibility = (log_likelihood(samples, center, variance) + weights.log()).softmax(dim=-1)
    return center, variance, weights, responsibility, torch.stack(trajectory)


class SpectralMixture(CrossFrequencyPrior):
    def __init__(self):
        super().__init__()
        d = ROUTING_DIMENSIONS
        self.register_buffer('routing_center', torch.zeros(COMPONENTS, d))
        self.register_buffer('routing_variance', torch.zeros(COMPONENTS, d))
        self.register_buffer('mixture_weight', torch.zeros(COMPONENTS))
        self.register_buffer('expert_center', torch.zeros(COMPONENTS, 6, 510))
        self.register_buffer('expert_variance', torch.zeros(COMPONENTS, 2, 510))
        self.register_buffer('route_even', torch.arange(6)[None, :, None] % 2 == 0, persistent=False)
        self.register_buffer('component', torch.tensor((0, 1, 1, 1, 1, 1)), persistent=False)

    def coefficients(self, spectrum):
        value = torch.view_as_real(spectrum).flatten(-2) - self.center
        common = value[:, :1] @ self.joint_axes[0]
        private = value[:, 1:] @ self.joint_axes[1]
        return torch.cat((common, private), dim=1)

    @torch.no_grad()
    def fit(self, frames):
        super().calibrate(frames())
        projections = torch.cat([self.coefficients(self.spectrum(block)) for block in frames()])
        samples = projections[:, 0, -ROUTING_DIMENSIONS:] / sqrt(6.)
        mean, spread, weights, responsibilities, trajectory = gaussian_mixture(samples)
        self.routing_center.copy_(mean)
        self.routing_variance.copy_(spread)
        self.mixture_weight.copy_(weights)
        count = responsibilities.sum(0).clamp_min(1e-5)[:, None, None]
        flat = projections.flatten(1)
        centers = (responsibilities.T @ flat).reshape_as(self.expert_center) / count
        second = (responsibilities.T @ flat.square()).reshape_as(self.expert_center) / count
        residual = (second - centers.square()).clamp_min(0.)
        conditional = torch.stack((residual[:, 0], residual[:, 1:].mean(1)), dim=1)
        shrink = (count / (count + 256.)).clamp(0., .9)
        self.expert_center.copy_(shrink * centers)
        self.expert_variance.copy_(shrink * conditional + (1. - shrink) * self.joint_variance)
        return dict(em_log_likelihood=trajectory.tolist(), effective_packets=count.flatten().tolist(),
                    state_weights=weights.tolist())

    def routing(self, observed):
        spectrum = torch.fft.fft(observed, norm='ortho')[..., 1:]
        average_even, average_odd = spectrum[:, ::2].mean(1), spectrum[:, 1::2].mean(1)
        groups = torch.stack((average_even, average_odd, .5 * (average_even + average_odd)))
        data = torch.view_as_real(groups).flatten(-2) - self.center[0] / sqrt(6.)
        projected = data @ self.joint_axes[0, :, -ROUTING_DIMENSIONS:]
        # Estimate the variance of each mean from that mean alone so a
        # held-out chirp cannot affect its own component probabilities.
        effective = groups[..., self.outer].abs().square().median(-1).values / (2. * log(2.))
        variance = self.routing_variance + effective[..., None, None]
        variance = variance.clamp_min(1e-8)
        weights = (log_likelihood(projected.flatten(0, 1), self.routing_center, variance.flatten(0, 1))
                   .unflatten(0, projected.shape[:2]) + self.mixture_weight.clamp_min(1e-8).log()).softmax(-1)
        even, odd, joint = weights.unbind(0)
        disagreement = .5 * (even - odd).abs().sum(-1)
        evidence = .5 * (joint - self.mixture_weight).abs().sum(-1)
        trust = (1. - disagreement) * (2. * evidence).clamp_max(1.)
        info = torch.stack((joint.max(-1).values, disagreement, evidence, trust), dim=-1)
        return weights, info

    def synthesize(self, coordinates):
        common = coordinates[..., :1, :] @ self.joint_axes[0].T
        private = coordinates[..., 1:, :] @ self.joint_axes[1].T
        packed = torch.cat((common, private), dim=-2).unflatten(-1, (255, 2)).contiguous()
        edit = torch.view_as_complex(packed)
        spectrum = torch.cat((torch.zeros_like(edit[..., :1]), edit), dim=-1)
        return self.basis.mT.to(spectrum) @ torch.fft.ifft(spectrum, norm='ortho')

    def changes(self, observed, rms):
        spectrum = self.spectrum(observed)
        noise = self.noise(spectrum)
        value = self.coefficients(spectrum)
        index = self.component
        received_noise = noise[:, index, None]
        pooled = posterior_gain(self.joint_variance[index], received_noise).sqrt() - 1.
        parent = self.synthesize(value * pooled)
        variance = self.expert_variance[:, index]
        shrink = posterior_gain(variance, received_noise[:, None]).sqrt() - 1.
        experts = shrink * (value[:, None] - self.expert_center)
        weights, info = self.routing(observed)
        mixed = torch.einsum('gmk,mkcd->gmcd', weights, experts)
        even, odd, joint = self.synthesize(mixed)
        cross = torch.where(self.route_even, odd, even)
        prior_weight = self.synthesize((experts * self.mixture_weight[None, :, None, None]).sum(1))
        edits = torch.stack((parent, joint, cross, prior_weight))
        strength = physical_norm(edits, rms)
        bound = .30 * physical_norm(observed, rms)
        parent, joint, cross, prior_weight = (edits * (bound / strength.clamp_min(1e-9)).clamp_max(1.)).unbind(0)
        parent_half, cross_half = .5 * parent, .5 * cross
        guarded = parent_half + info[:, -1, None, None] * (cross_half - parent_half)
        equal = match_dose(cross_half, parent_half, rms)
        control = match_dose(parent_half, cross_half, rms)
        return torch.stack((parent, parent_half, joint, .5 * joint, cross_half, guarded,
                            equal, control, .5 * prior_weight)), noise, info
