"""Conditional diffusion of clean-minus-DURA IQ from one received record.

Fixed finite trajectories are a deterministic inference ensemble; they are not
claimed to be calibrated posterior samples. No classifier or target is an input.
"""
import math
import torch
from torch import nn
from experiments.phase1_joint_particle.particle import ResidualBlock

WIDTH = 192
STEPS = 8
PARTICLES = 4
LATENT_SEED = 2026110820


def alpha(time):
    return torch.cos(time * (math.pi / 2)).square()


def forward_noise(clean_residual, time, noise):
    a = alpha(time)[:, None, None]
    return a.sqrt()*clean_residual + (1-a).clamp_min(0).sqrt()*noise


def reverse_step(noisy, estimate, time, next_time):
    a = alpha(time)[:, None, None]
    following = alpha(next_time)[:, None, None]
    noise = (noisy-a.sqrt()*estimate)/(1-a).clamp_min(1e-12).sqrt()
    return following.sqrt()*estimate+(1-following).clamp_min(0).sqrt()*noise


def scale_of(anchor):
    return anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()


class ResidualDiffusion(nn.Module):
    def __init__(self, width=WIDTH):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('Width must be a positive multiple of eight')
        self.entry = nn.Sequential(nn.Conv1d(10, width, 7, padding=3), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock(width, d) for d in (1, 2, 4, 8, 16, 32)))
        self.output = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        rng = torch.Generator(device='cpu').manual_seed(LATENT_SEED)
        half = torch.randn((PARTICLES//2, 2, 256), generator=rng)
        self.register_buffer('fixed_latent', torch.cat((half, -half), 0))

    def estimate(self, received, anchor, innovation, variance, noisy_residual, time):
        if (received.ndim != 3 or received.shape[1:] != (2, 256) or anchor.shape != received.shape or
                innovation.shape != received.shape or noisy_residual.shape != received.shape or
                variance.shape != (len(received),) or time.shape != (len(received),)):
            raise ValueError('Matching per-record IQ, variance, diffusion time required')
        scale = scale_of(anchor)
        t = time[:, None, None].expand(-1, 1, 256)
        log_noise = torch.log1p(variance[:, None, None]/scale.square()).expand(-1, 1, 256)
        features = torch.cat((received/scale, anchor/scale, innovation/scale, noisy_residual, log_noise, t), 1)
        return self.output(self.blocks(self.entry(features)))

    def forward(self, received, anchor, innovation, variance, *, latent=None):
        if latent is None:
            latent = self.fixed_latent[None].expand(len(received), -1, -1, -1)
        if latent.ndim != 4 or latent.shape[0] != len(received) or latent.shape[1] < 1 or latent.shape[2:] != (2, 256):
            raise ValueError('Latent must have shape [batch, particles, 2, 256]')
        particles = latent.shape[1]
        repeat = lambda x: x.repeat_interleave(particles, 0)
        y, base, direction, v = map(repeat, (received, anchor, innovation, variance))
        state = latent.reshape(-1, 2, 256).to(device=y.device, dtype=y.dtype)
        for index in range(STEPS, 0, -1):
            time = torch.full((len(y),), index/STEPS, dtype=y.dtype, device=y.device)
            following = torch.full_like(time, (index-1)/STEPS)
            estimate = self.estimate(y, base, direction, v, state, time)
            state = reverse_step(state, estimate, time, following)
        result = base + scale_of(base)*state
        return result.reshape(len(received), particles, 2, 256).mean(1)
