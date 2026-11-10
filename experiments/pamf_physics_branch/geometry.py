"""Bounded received-IQ echo and branch-wise intervention-aware distances.

The echo is a post-reception proxy, not an identified transmitter/channel model.
The finite differences are detached when used to construct the training metric.
"""
import torch
from torch.nn import functional as F


def bounded_echo(iq, coefficients, delays, max_magnitude=.1):
    """Apply y[n] = x[n] + c*x[n-d] with zero history for each IQ record."""
    if (iq.ndim != 3 or iq.shape[1] != 2 or not iq.is_floating_point()
            or coefficients.shape != (len(iq), 2) or coefficients.device != iq.device
            or coefficients.dtype != iq.dtype or delays.shape != (len(iq),)
            or delays.device != iq.device or delays.dtype != torch.long
            or max_magnitude <= 0 or max_magnitude >= 1
            or not bool(torch.isfinite(iq).all()) or not bool(torch.isfinite(coefficients).all())
            or bool((delays < 1).any()) or bool((delays > 3).any())
            or bool((delays >= iq.shape[2]).any())
            or bool((coefficients.square().sum(1) > (max_magnitude + 1e-7) ** 2).any())):
        raise ValueError('Finite aligned bounded causal echo required')
    positions = torch.arange(iq.shape[2], device=iq.device)[None, :] - delays[:, None]
    valid = positions >= 0
    positions = positions.clamp_min(0).unsqueeze(1).expand(-1, 2, -1)
    shifted = iq.gather(2, positions) * valid[:, None, :]
    real = coefficients[:, 0, None] * shifted[:, 0] - coefficients[:, 1, None] * shifted[:, 1]
    imag = coefficients[:, 0, None] * shifted[:, 1] + coefficients[:, 1, None] * shifted[:, 0]
    return iq + torch.stack((real, imag), dim=1)


def tangent(raw_eval, perturbed_eval, eps=1e-5):
    """Detached actual branch direction; zero when the finite difference is tiny."""
    if (raw_eval.ndim != 2 or raw_eval.shape != perturbed_eval.shape or eps <= 0
            or not bool(torch.isfinite(raw_eval).all()) or not bool(torch.isfinite(perturbed_eval).all())):
        raise ValueError('Finite aligned branch vectors required')
    raw = F.normalize(raw_eval.detach(), dim=1, eps=1e-8)
    changed = F.normalize(perturbed_eval.detach(), dim=1, eps=1e-8)
    difference = changed - raw
    norm = difference.norm(dim=1, keepdim=True)
    unit = torch.where(norm >= eps, difference / norm.clamp_min(eps), torch.zeros_like(difference))
    return unit, norm.squeeze(1)


def soft_distances(features, directions, beta):
    """Pairwise soft directional quotients, symmetric but not a global metric."""
    if (features.ndim != 2 or directions.shape != features.shape or not 0 <= beta <= 1
            or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(directions).all())):
        raise ValueError('Finite aligned features, directions and beta in [0,1] required')
    z = F.normalize(features, dim=1, eps=1e-8)
    u = directions.detach()
    if bool((u.norm(dim=1) > 1.0001).any()):
        raise ValueError('Directions must be unit vectors or zero')
    norms = (z * z).sum(1)
    ordinary = (norms[:, None] + norms[None, :] - 2 * (z @ z.T)).clamp_min(0.)
    diagonal = (z * u).sum(1)
    left = diagonal[:, None] - u @ z.T
    right = z @ u.T - diagonal[None, :]
    corrected = (ordinary - .5 * beta * (left.square() + right.square())).clamp_min(0.)
    return ordinary, corrected


def physical_branch_loss(features, directions, labels, receivers, days, record_ids, accepted,
                         *, beta, single_receiver, temperature=.2, reference_scale=1.):
    """Same-day source identity relation, using true or already accepted pseudo labels."""
    if (not features or features.keys() != directions.keys() or labels.ndim != 1
            or any(v.shape != labels.shape for v in (receivers, days, record_ids, accepted))
            or accepted.dtype != torch.bool or labels.dtype != torch.long
            or temperature <= 0 or not 0 <= beta <= 1 or not 0 <= reference_scale <= 1):
        raise ValueError('Aligned branch features and legal source relation required')
    if bool((labels[accepted] < 0).any()):
        raise ValueError('Accepted identities must be nonnegative')
    if single_receiver and receivers[accepted].unique().numel() > 1:
        raise ValueError('Single-receiver fallback cannot mix receivers')
    same_label = labels[:, None] == labels[None, :]
    same_receiver = receivers[:, None] == receivers[None, :]
    same_day = days[:, None] == days[None, :]
    distinct = record_ids[:, None] != record_ids[None, :]
    eligible = accepted[:, None] & accepted[None, :] & distinct & same_day
    positive = eligible & same_label & (same_receiver if single_receiver else ~same_receiver)
    negative = eligible & ~same_label & same_receiver
    valid = positive.any(1) & negative.any(1)
    losses, correction = [], []
    for name, value in features.items():
        if value.shape[0] != len(labels):
            raise ValueError('Actual branch batch size differs')
        original, adjusted = soft_distances(value, directions[name], beta)
        if not bool(valid.any()):
            losses.append(value.sum() * 0.)
            correction.append(original.sum().detach() * 0.)
            continue
        score = -adjusted[valid] / (2 * temperature)
        pos = positive[valid]
        permitted = (positive | negative)[valid]
        denominator = torch.logsumexp(score.masked_fill(~permitted, -torch.inf), dim=1)
        positive_mean = score.masked_fill(~pos, 0.).sum(1) / pos.sum(1)
        scale = reference_scale if name == 'reference' else 1.
        losses.append(scale * (denominator - positive_mean).mean())
        correction.append((original - adjusted)[valid].mean().detach())
    return torch.stack(losses).mean(), dict(valid_anchors=int(valid.sum()),
           nonzero_directions={k: int((v.detach().norm(dim=1) > 0).sum()) for k, v in directions.items()},
           mean_distance_correction=torch.stack(correction).mean())
