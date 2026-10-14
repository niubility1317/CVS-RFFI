"""Conservative teacher evidence sets and student set-mass loss primitives.

Use the existing default cosine6x160 head.  No head, parameter, optimizer state,
truth label, confidence gate or probability-vector distillation is created.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _scalar(value, name, *, minimum=None, maximum=None, minimum_open=False, maximum_open=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite fixed scalar")
    if minimum is not None and (value <= minimum if minimum_open else value < minimum):
        raise ValueError(f"{name} is below its allowed range")
    if maximum is not None and (value >= maximum if maximum_open else value > maximum):
        raise ValueError(f"{name} is above its allowed range")
    return float(value)


def _features(value, name, reference=None):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.ndim != 2 or value.shape[1] != 160 or not value.is_floating_point():
        raise ValueError(f"{name} must be a real dense N x 160 floating tensor")
    if reference is not None and (value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f"{name} must match the feature shape, dtype and device")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains nonfinite features")


def _weight(weight, reference):
    _features(weight, "weight")
    if weight.shape != (6, 160) or weight.dtype != reference.dtype or weight.device != reference.device:
        raise ValueError("weight must be the matching existing default [6,160] classifier weight")


def _histogram(sizes):
    return {str(size): int((sizes == size).sum().item()) for size in range(1, 7)}


@torch.no_grad()
def candidate_set(a, b, c, weight, *, beta=0.5, tolerance=1e-6, normalize_eps=1e-4):
    """Return detached bool candidates and pairwise evidence maxima.

    At u,v in [0,beta], z=a+(1-u)b+(1-v)c.  For each k,j the
    maximum score difference is d0 + beta*relu(-db) + beta*relu(-dc).
    Class k remains iff every pairwise maximum is >= -tolerance.  Each j can
    attain its maximum at a different rectangle corner: this is conservative
    pairwise possibility, not the exact union of feasible winners or a promise
    to retain the unknown truth class.

    The existing W normalizes in its own dtype with the default epsilon.  A
    common positive scale per packet makes teacher score arithmetic stable;
    tolerance divides by that same scale, preserving its original raw-score
    units.  Per-class score differences retain the nominal maximum even at
    beta=0/tolerance=0.  Inputs and W are detached, with no RNG consumption.
    """
    _features(a, "a"); _features(b, "b", a); _features(c, "c", a); _weight(weight, a)
    beta = _scalar(beta, "beta", minimum=0, maximum=1, maximum_open=True)
    tolerance = _scalar(tolerance, "tolerance", minimum=0)
    normalize_eps = _scalar(normalize_eps, "normalize_eps", minimum=0, minimum_open=True)
    normalized_weight = F.normalize(weight.detach(), p=2, dim=1, eps=normalize_eps).to(torch.float64)
    detached = [value.detach().to(torch.float64) for value in (a, b, c)]
    score_scale = torch.stack([value.abs().amax(dim=1) for value in detached], dim=1).amax(dim=1).clamp_min(1.0)
    scaled_a, scaled_b, scaled_c = [value / score_scale[:, None] for value in detached]
    nominal_scores = F.linear(scaled_a + scaled_b + scaled_c, normalized_weight)
    scores_b = F.linear(scaled_b, normalized_weight)
    scores_c = F.linear(scaled_c, normalized_weight)
    d0 = nominal_scores[:, :, None] - nominal_scores[:, None, :]
    db = scores_b[:, :, None] - scores_b[:, None, :]
    dc = scores_c[:, :, None] - scores_c[:, None, :]
    maxima_scaled = d0 + beta * F.relu(-db) + beta * F.relu(-dc)
    candidates = (maxima_scaled >= -(tolerance / score_scale)[:, None, None]).all(dim=2)
    if not bool(candidates.any(dim=1).all().item()):
        raise RuntimeError("finite evidence geometry unexpectedly produced an empty candidate set")
    sizes = candidates.sum(dim=1)
    raw_maxima = maxima_scaled * score_scale[:, None, None]
    raw_representable = bool(torch.isfinite(raw_maxima).all().item())
    return {
        "candidates": candidates,
        "summary": {
            "input_count": a.shape[0], "beta": beta, "tolerance": tolerance,
            "normalize_eps": normalize_eps, "size_histogram": _histogram(sizes),
            "singleton_count": int((sizes == 1).sum().item()),
            "nontrivial_count": int((sizes < 6).sum().item()),
            "full_set_count": int((sizes == 6).sum().item()),
            "empty_count": 0, "teacher_inputs_detached": True, "weight_detached": True,
            "tolerance_units": "raw joint dot-product differences with default-normalized W",
            "score_scale_returned": True, "raw_pairwise_maxima_representable": raw_representable,
            "set_semantics": "conservative_pairwise_possible_not_exact_feasible_winner_union",
            "truth_retention_guaranteed": False, "global_RNG_consumed": False,
        },
        "tensors": {
            "pairwise_maxima": raw_maxima if raw_representable else None,
            "pairwise_maxima_scaled": maxima_scaled, "score_scale": score_scale,
            "nominal_scores_scaled": nominal_scores,
        },
    }


def set_loss(z, weight, candidates, *, scale=30.0, probability=0.9,
             coefficient=0.16, normalize_eps=1e-4, eligible_mask=None):
    """Penalize insufficient student probability mass in a nonempty class set.

    loss_i=relu(log(probability)-log P(C_i)); weighted loss is coefficient times
    sum(eligible_i*loss_i)/N for the FULL input U batch.  Eligibility is supplied
    by the caller (e.g. complement of original hard acceptance), with no student
    agreement test here.  Internal ratios among candidate probabilities are not
    fixed.  Full sets and fulfilled mass requirements stop the derivative.

    W detaches, z retains its existing graph, and no labels/model/optimizer are
    accepted.  The caller must construct z with only fusion gains detached;
    this primitive does not implement gain handling or alter normal main CE.
    """
    _features(z, "z"); _weight(weight, z)
    scale = _scalar(scale, "scale", minimum=0, minimum_open=True)
    probability = _scalar(probability, "probability", minimum=0, maximum=1, minimum_open=True)
    coefficient = _scalar(coefficient, "coefficient", minimum=0)
    normalize_eps = _scalar(normalize_eps, "normalize_eps", minimum=0, minimum_open=True)
    count = z.shape[0]
    if not isinstance(candidates, torch.Tensor) or candidates.dtype != torch.bool or candidates.shape != (count, 6):
        raise ValueError("candidates must be a bool [N,6] mask")
    candidate_mask = candidates.to(z.device)
    if not bool(candidate_mask.any(dim=1).all().item()):
        raise ValueError("every candidate set must be nonempty")
    if eligible_mask is None:
        eligible = torch.ones(count, dtype=torch.bool, device=z.device)
    else:
        if not isinstance(eligible_mask, torch.Tensor) or eligible_mask.dtype != torch.bool or eligible_mask.shape != (count,):
            raise ValueError("eligible_mask must be a bool [N] mask")
        eligible = eligible_mask.to(z.device)
    sizes = candidate_mask.sum(dim=1)
    if count:
        normalized_weight = F.normalize(weight.detach(), p=2, dim=1, eps=normalize_eps)
        logits = F.linear(F.normalize(z, p=2, dim=1, eps=normalize_eps), normalized_weight) * scale
        log_probability = F.log_softmax(logits, dim=1)
        log_mass = torch.logsumexp(log_probability.masked_fill(~candidate_mask, -float("inf")), dim=1)
        # Exact full-set mass avoids roundoff triggering a loss at probability=1.
        log_mass = torch.where(sizes == 6, torch.zeros_like(log_mass), log_mass)
        packet_loss = F.relu(math.log(probability) - log_mass)
        masked_loss = torch.where(eligible, packet_loss, torch.zeros_like(packet_loss))
        mean_loss = masked_loss.sum() / count
        weighted_loss = coefficient * mean_loss
        if not bool(torch.isfinite(packet_loss).all().item()) or not bool(torch.isfinite(weighted_loss).item()):
            raise ValueError("set-mass loss produced a nonfinite result")
    else:
        log_mass = z.new_empty((0,))
        packet_loss = z.new_empty((0,))
        mean_loss = weighted_loss = (z * 0.0).sum()
    active = eligible & (packet_loss > 0)
    return {
        "weighted_loss": weighted_loss,
        "summary": {
            "input_count": count, "scale": scale, "probability": probability,
            "coefficient": coefficient, "normalize_eps": normalize_eps,
            "eligible_count": int(eligible.sum().item()), "active_count": int(active.sum().item()),
            "nontrivial_count": int((sizes < 6).sum().item()),
            "eligible_nontrivial_count": int((eligible & (sizes < 6)).sum().item()),
            "eligible_coverage": float(eligible.sum().item()) / count if count else 0.0,
            "active_coverage": float(active.sum().item()) / count if count else 0.0,
            "size_histogram": _histogram(sizes), "mean_loss": float(mean_loss.item()),
            "unweighted_loss": float(mean_loss.item()), "weighted_loss": float(weighted_loss.item()),
            "denominator": "full_input_batch_N", "within_set_probability_ratios_fixed": False,
            "student_endpoints_detached": False, "weight_detached": True,
            "new_parameters": False, "model_state": False, "optimizer_state": False,
            "global_RNG_consumed": False,
        },
        "tensors": {"packet_loss": packet_loss, "log_candidate_mass": log_mass},
    }
