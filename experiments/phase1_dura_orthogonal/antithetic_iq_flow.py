"""Antithetic stochastic-start conditional transport with a frozen DURA anchor."""
import torch

from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

LATENT_SEED = 2026110723
TRAIN_LATENT_SEED = 2026110724
LATENT_SCALE = .5
MAX_RELATIVE_STD = .25
FLOW_STEPS = 4


def perturbation_scale(anchor, variance):
    """Bound physical-variance perturbations by a fixed fraction of IQ RMS."""
    if anchor.ndim != 3 or anchor.shape[1:] != (2, 256) or variance.shape != (len(anchor),):
        raise ValueError('Matching anchor IQ and variance required')
    power = anchor.square().mean((1, 2), keepdim=True).clamp_min(1e-12)
    std = variance.clamp_min(0.).sqrt()[:, None, None]
    return torch.minimum(LATENT_SCALE*std, MAX_RELATIVE_STD*power.sqrt())


class AntitheticIQFlow(ConditionalIQFlow):
    """Average opposite fixed-latent flow corrections, gated back to DURA."""
    def __init__(self, width=32, steps=FLOW_STEPS):
        super().__init__(width=width, steps=steps)
        generator = torch.Generator(device='cpu').manual_seed(LATENT_SEED)
        self.register_buffer('fixed_latent', torch.randn((1, 2, 256), generator=generator))

    def forward(self, received, anchor, innovation, variance):
        perturbation = perturbation_scale(anchor, variance)*self.fixed_latent
        initial_plus, initial_minus = anchor+perturbation, anchor-perturbation
        plus, minus = initial_plus, initial_minus
        for index in range(self.steps):
            t = torch.full((len(anchor),), index/self.steps, device=anchor.device, dtype=anchor.dtype)
            plus = plus + self.velocity(plus, received, anchor, innovation, variance, t)/self.steps
            minus = minus + self.velocity(minus, received, anchor, innovation, variance, t)/self.steps
        delta = .5*((plus-initial_plus)+(minus-initial_minus))
        return anchor + noise_gate(anchor, variance)*delta


def stochastic_flow_loss(model, received, anchor, innovation, variance, clean, time, latent):
    """Match linear transport from a perturbed anchor to its paired clean IQ."""
    if clean.shape != anchor.shape or latent.shape != anchor.shape or time.shape != (len(anchor),):
        raise ValueError('Matching source IQ, time and latent samples required')
    if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError('Finite flow time in [0, 1] required')
    start = anchor + perturbation_scale(anchor, variance)*latent
    alpha = time[:, None, None]
    state = (1-alpha)*start + alpha*clean
    target = clean-start
    predicted = model.velocity(state, received, anchor, innovation, variance, time)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return ((predicted-target).square().mean((1, 2))/energy).mean()
