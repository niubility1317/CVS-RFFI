"""Frozen two-path readout; no trained classifier or posterior calibration claim."""
import torch

from experiments.phase1_dura_orthogonal.antithetic_iq_flow import AntitheticIQFlow, perturbation_scale
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate

MODE = 'antithetic_probability_mean'


def corrected_paths(model, received, anchor, innovation, variance):
    """Retain each fixed-latent correction instead of averaging its IQ first."""
    if not isinstance(model, AntitheticIQFlow):
        raise ValueError('Antithetic source model required for two-path readout')
    perturbation = perturbation_scale(anchor, variance)*model.fixed_latent
    initial_plus, initial_minus = anchor+perturbation, anchor-perturbation
    plus, minus = initial_plus, initial_minus
    for index in range(model.steps):
        time = torch.full((len(anchor),), index/model.steps, device=anchor.device, dtype=anchor.dtype)
        plus = plus + model.velocity(plus, received, anchor, innovation, variance, time)/model.steps
        minus = minus + model.velocity(minus, received, anchor, innovation, variance, time)/model.steps
    gate = noise_gate(anchor, variance)
    return anchor+gate*(plus-initial_plus), anchor+gate*(minus-initial_minus)


def probability_mean(logits):
    if len(logits) != 2 or any(value.ndim != 2 or value.shape[1] != 6 for value in logits):
        raise ValueError('Two six-class frozen reader outputs required')
    if logits[0].shape != logits[1].shape or any(not torch.isfinite(value).all() for value in logits):
        raise ValueError('Matching finite logits required')
    return .5*(logits[0].softmax(1)+logits[1].softmax(1))


@torch.no_grad()
def path_probabilities(model, reader, received, anchor, innovation, variance):
    paths = corrected_paths(model, received, anchor, innovation, variance)
    if any(not torch.isfinite(path).all() for path in paths):
        raise ValueError('Finite antithetic corrections required')
    return probability_mean([reader.classify_features(reader.features(path)) for path in paths])
