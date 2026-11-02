"""Source training weighted by input-only estimated relative noise."""
import torch
from experiments.phase1_volterra_aligned.objective import aligned_iq_error, source_loss
from experiments.phase1_dura_focus.objective import group_weighted_mean

WEIGHT_COUNTS = [0, 0.0, 0]


@torch.no_grad()
def noise_weights(received, log_variance):
    if received.ndim != 3 or received.shape[1:] != (2, 256) or log_variance.shape != (len(received),):
        raise ValueError('Matching IQ and one estimated noise value are required')
    if not torch.isfinite(received).all() or not torch.isfinite(log_variance).all():
        raise FloatingPointError('Finite input and noise estimates are required')
    power = received.square().mean((1, 2)).clamp_min(1e-8)
    relative = log_variance - power.log()
    return 1.0 + 7.0 * torch.sigmoid(relative + 3.0)


def noise_weighted_objective(reader, output, anchor, clean, clean_features, groups, received, log_variance):
    _, terms = source_loss(reader, output, clean, clean_features, groups)
    with torch.no_grad():
        weights = noise_weights(received, log_variance)
        baseline_error = aligned_iq_error(anchor.detach(), clean)
        WEIGHT_COUNTS[0] += len(weights)
        WEIGHT_COUNTS[1] += float(weights.sum())
        WEIGHT_COUNTS[2] += int((weights >= 4.5).sum())
    degradation = (terms['aligned'] - baseline_error).clamp_min(0)
    per = (terms['aligned'] + 0.05 * terms['power'] + 0.2 * terms['feature']
           + 0.2 * terms['distill'] + 0.05 * terms['margin'] + 5.0 * degradation)
    return group_weighted_mean(per, weights, groups), dict(terms, degradation=degradation, weights=weights)
