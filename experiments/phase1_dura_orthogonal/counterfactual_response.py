"""Source-only synthetic response constraint; never used at inference."""
import torch

from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import (
    DIRECTIONS, EPSILON, perturbations,
)

WEIGHT = .01


def choose_perturbation(clean, start):
    directions = torch.stack(perturbations(clean), dim=1)
    indices = (torch.arange(len(clean), device=clean.device) + start) % len(DIRECTIONS)
    return directions[torch.arange(len(clean), device=clean.device), indices]


def response_loss(original, changed, delta):
    if original.shape != changed.shape or original.shape != delta.shape or original.ndim != 3 or original.shape[1:] != (2, 256):
        raise ValueError('Matching paired source IQ responses required')
    error = (changed - original - delta).square().sum((1, 2))
    energy = delta.square().sum((1, 2)).clamp_min(1e-20)
    return (error / energy).mean()


@torch.no_grad()
def prepare_counterfactual(dura, data, extract_context, batch_size):
    """Precompute a single fixed direction per source view, with DURA re-estimated."""
    x_all, _, _, _, y_all = data[:5]
    parts = []
    for start in range(0, len(x_all), batch_size):
        x, y = x_all[start:start + batch_size], y_all[start:start + batch_size]
        delta = choose_perturbation(y, start)
        action = dura.estimate_context(x)[1]
        received_delta = action(delta)
        changed = x + torch.stack((received_delta.real, received_delta.imag), dim=1)
        anchor, direction, variance = extract_context(dura, changed)
        delta_iq = torch.stack((delta.real, delta.imag), dim=1)
        parts.append((changed, anchor, direction, variance, delta_iq))
        if (start + len(x)) % 4096 == 0 or start + len(x) == len(x_all):
            print('SOURCE_COUNTERFACTUAL_PREP', start + len(x), len(x_all), flush=True)
    return tuple(torch.cat(values, dim=0) for values in zip(*parts))
