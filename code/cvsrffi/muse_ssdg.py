"""Stateless MUSE-SSDG training schedule primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


_PROBABILITY_EPS = 1e-8
_DEFAULT_FUSION_WEIGHTS = (0.50, 0.25, 0.25)
_DEFAULT_RATIO_CLIP = (0.5, 2.0)
_DEFAULT_RELIABILITY_WEIGHTS = (0.25, 0.20, 0.20, 0.20, 0.15)
_DEFAULT_UNLABELED_PROTOTYPE_WEIGHT = 0.075
_MIN_UNLABELED_PROTOTYPE_WEIGHT = 0.05
_MAX_UNLABELED_PROTOTYPE_WEIGHT = 0.10
_TEMPORAL_STABILITY_STEPS = 3


@dataclass(frozen=True)
class MUSEConfig:
    """Immutable defaults for the 200-epoch MUSE training schedule."""

    s2a_start: int = 17
    s2b_start: int = 41
    s3a_start: int = 69
    s3b_start: int = 161
    s3c_start: int = 181
    final_epoch: int = 200
    lambda_u_full: float = 0.60
    lambda_u_consolidate: float = 0.25
    p_sat_s2a_end: float = 0.25
    p_sat_full: float = 0.50
    grl_min: float = 0.02
    grl_max: float = 0.10


@dataclass(frozen=True)
class MUSEScheduleState:
    """The complete stateless schedule decision for one training epoch."""

    stage: str
    ema_decay: float
    lambda_u: float
    p_sat: float
    grl_lambda: float
    proto_momentum: float
    pseudo_enabled: bool
    candidate_enabled: bool
    freeze_statistics: bool


def _linear_ramp(start: float, end: float, epoch: int, start_epoch: int, end_epoch: int) -> float:
    """Return an endpoint-inclusive linear ramp clipped to its interval."""

    if end_epoch <= start_epoch:
        return float(end)
    if epoch <= start_epoch:
        return float(start)
    if epoch >= end_epoch:
        return float(end)
    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(start + (end - start) * progress)


def _probability(value: float) -> float:
    """Keep schedule probabilities inside the closed unit interval."""

    return float(max(0.0, min(1.0, float(value))))


def muse_schedule_for_epoch(epoch: int, config: MUSEConfig) -> MUSEScheduleState:
    """Build the immutable MUSE schedule state for ``epoch``.

    The schedule has six named stages: S1 (1-16), S2A (17-40), S2B
    (41-68), S3A (69-160), S3B (161-180), and final consolidation S3C
    (181-200). S2A and S2B ramps include both segment endpoints.
    """

    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError("epoch must be an int")
    if epoch < 1 or epoch > config.final_epoch:
        raise ValueError(f"epoch must be in [1, {config.final_epoch}], got {epoch}")

    # Keep the documented decimal segment endpoints stable despite binary
    # floating-point representation (for example, 0.60 / 3 == 0.199999...).
    lambda_u_s2a_end = round(float(config.lambda_u_full) / 3.0, 12)
    lambda_u_s2b_end = round(float(config.lambda_u_full) * 5.0 / 6.0, 12)

    if epoch < config.s2a_start:
        stage = "S1"
        ema_decay = 0.99
        lambda_u = 0.0
        p_sat = 0.0
        proto_momentum = 0.95
    elif epoch < config.s2b_start:
        stage = "S2A"
        ema_decay = 0.995
        lambda_u = _linear_ramp(
            0.0,
            lambda_u_s2a_end,
            epoch,
            config.s2a_start,
            config.s2b_start - 1,
        )
        p_sat = _linear_ramp(
            0.0,
            float(config.p_sat_s2a_end),
            epoch,
            config.s2a_start,
            config.s2b_start - 1,
        )
        proto_momentum = 0.95
    elif epoch < config.s3a_start:
        stage = "S2B"
        ema_decay = 0.995
        lambda_u = _linear_ramp(
            lambda_u_s2a_end,
            lambda_u_s2b_end,
            epoch,
            config.s2b_start,
            config.s3a_start - 1,
        )
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.95
    elif epoch < config.s3b_start:
        stage = "S3A"
        ema_decay = 0.999
        lambda_u = float(config.lambda_u_full)
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.95
    elif epoch < config.s3c_start:
        stage = "S3B"
        ema_decay = 0.999
        lambda_u = float(config.lambda_u_full)
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.99
    else:
        stage = "S3C"
        ema_decay = 0.999
        lambda_u = float(config.lambda_u_consolidate)
        p_sat = float(config.p_sat_full)
        proto_momentum = 0.99

    grl_lambda = _linear_ramp(
        float(config.grl_min),
        float(config.grl_max),
        epoch,
        1,
        config.final_epoch,
    )
    return MUSEScheduleState(
        stage=stage,
        ema_decay=float(ema_decay),
        lambda_u=float(lambda_u),
        p_sat=_probability(p_sat),
        grl_lambda=_probability(grl_lambda),
        proto_momentum=float(proto_momentum),
        pseudo_enabled=stage != "S1",
        candidate_enabled=stage in {"S2B", "S3A", "S3B", "S3C"},
        freeze_statistics=stage == "S3C",
    )


@dataclass(frozen=True)
class MUSERoute:
    """Mutually exclusive high/mid/low reliability masks."""

    high: torch.Tensor
    mid: torch.Tensor
    low: torch.Tensor


def _probability_heads(probabilities: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Convert probability heads to a common floating dtype and validate shape."""

    heads = list(probabilities)
    if not heads:
        raise ValueError("at least one probability head is required")

    tensors = [value if torch.is_tensor(value) else torch.as_tensor(value) for value in heads]
    reference = tensors[0]
    if reference.ndim < 1 or reference.shape[-1] == 0:
        raise ValueError("probability heads must have a non-empty class dimension")
    if reference.is_complex():
        raise TypeError("probability heads must be real-valued")
    dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
    for tensor in tensors[1:]:
        if tensor.is_complex():
            raise TypeError("probability heads must be real-valued")
        if tensor.device != reference.device:
            raise ValueError("probability heads must be on the same device")
        if tensor.ndim < 1 or tensor.shape != reference.shape:
            raise ValueError("probability heads must have matching shapes")
        if tensor.is_floating_point():
            dtype = torch.promote_types(dtype, tensor.dtype)
    if not reference.is_floating_point():
        dtype = torch.get_default_dtype()
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return [tensor.to(device=reference.device, dtype=dtype) for tensor in tensors]


def _sanitize_probability(probability: torch.Tensor) -> torch.Tensor:
    """Replace non-finite probability values and keep every log input positive."""

    return torch.nan_to_num(
        probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp_min(_PROBABILITY_EPS)


def _normalize_probability(probability: torch.Tensor) -> torch.Tensor:
    """Normalize a probability tensor in log space without summation overflow."""

    safe = _sanitize_probability(probability)
    log_probability = safe.log()
    return torch.exp(log_probability - torch.logsumexp(log_probability, dim=-1, keepdim=True))


def geometric_fuse_probabilities(
    probabilities: Sequence[torch.Tensor],
    weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse class probabilities by a normalized, weighted geometric mean.

    Inputs are sanitized before taking logarithms.  A zero weight is allowed,
    but the total weight must be finite and strictly positive.
    """

    heads = _probability_heads(probabilities)
    if weights is None:
        weights = _DEFAULT_FUSION_WEIGHTS if len(heads) == 3 else [1.0] * len(heads)
    weight_tensor = torch.as_tensor(weights, dtype=heads[0].dtype, device=heads[0].device).reshape(-1)
    if weight_tensor.numel() != len(heads):
        raise ValueError("weights must contain one value per probability head")
    if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
        raise ValueError("weights must be finite and non-negative")
    weight_sum = weight_tensor.sum()
    if not torch.isfinite(weight_sum) or weight_sum.item() <= 0.0:
        raise ValueError("weight sum must be greater than zero")

    weighted_log = torch.zeros_like(heads[0])
    for probability, weight in zip(heads, weight_tensor):
        weighted_log = weighted_log + weight * _sanitize_probability(probability).log()
    weighted_log = weighted_log / weight_sum
    return torch.exp(weighted_log - torch.logsumexp(weighted_log, dim=-1, keepdim=True))


def _ratio_bounds(ratio_clip: Sequence[float] | torch.Tensor | float | None) -> tuple[float, float]:
    """Parse a ratio clip while retaining the design range ``[0.5, 2.0]``."""

    if ratio_clip is None:
        lower, upper = _DEFAULT_RATIO_CLIP
    elif isinstance(ratio_clip, Real):
        value = float(ratio_clip)
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise ValueError("ratio_clip must be positive")
        lower, upper = (value, 1.0 / value) if value < 1.0 else (1.0 / value, value)
    else:
        values = torch.as_tensor(ratio_clip, dtype=torch.get_default_dtype()).reshape(-1)
        if values.numel() != 2 or not torch.isfinite(values).all():
            raise ValueError("ratio_clip must contain two finite values")
        lower, upper = (float(values[0]), float(values[1]))
    lower, upper = max(0.5, lower), min(2.0, upper)
    if lower <= 0.0 or upper <= 0.0 or lower > upper:
        raise ValueError("ratio_clip must define a non-empty positive interval")
    return lower, upper


def align_source_domain_prior(
    prob: torch.Tensor,
    domain_prior: torch.Tensor,
    global_prior: torch.Tensor,
    gamma: float = 0.35,
    ratio_clip: Sequence[float] | torch.Tensor | float = _DEFAULT_RATIO_CLIP,
) -> torch.Tensor:
    """Apply source-domain to global class-prior correction and renormalize."""

    heads = _probability_heads([prob])
    probability = _normalize_probability(heads[0])
    dtype, device = probability.dtype, probability.device
    try:
        domain = torch.as_tensor(domain_prior, dtype=dtype, device=device)
        global_ = torch.as_tensor(global_prior, dtype=dtype, device=device)
        probability, domain, global_ = torch.broadcast_tensors(probability, domain, global_)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("prob, domain_prior, and global_prior must be broadcastable") from exc
    if not torch.isfinite(torch.as_tensor(gamma)).item():
        raise ValueError("gamma must be finite")

    domain = torch.nan_to_num(domain, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(_PROBABILITY_EPS)
    global_ = torch.nan_to_num(global_, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(_PROBABILITY_EPS)
    lower, upper = _ratio_bounds(ratio_clip)
    ratio = (global_ / domain).clamp(min=lower, max=upper)
    corrected = probability * ratio.pow(float(gamma))
    return _normalize_probability(corrected)


def js_head_disagreement(probabilities: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the mean head-to-mean KL divergence for each sample."""

    heads = [_normalize_probability(head) for head in _probability_heads(probabilities)]
    mean_probability = torch.stack(heads, dim=0).mean(dim=0)
    mean_probability = _normalize_probability(mean_probability)
    log_mean = mean_probability.log()
    divergences = [
        (probability * (probability.log() - log_mean)).sum(dim=-1)
        for probability in heads
    ]
    return torch.stack(divergences, dim=0).mean(dim=0).clamp_min(0.0)


def _evidence_tensors(values: Sequence[object]) -> list[torch.Tensor]:
    """Broadcast reliability evidence values to one floating dtype and device."""

    reference = next((value for value in values if torch.is_tensor(value)), None)
    if reference is None:
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()
    else:
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        if reference.is_complex():
            raise TypeError("reliability evidence must be real-valued")
        for value in values:
            if torch.is_tensor(value):
                if value.device != device:
                    raise ValueError("reliability evidence must be on the same device")
                if value.is_complex():
                    raise TypeError("reliability evidence must be real-valued")
                if value.is_floating_point():
                    dtype = torch.promote_types(dtype, value.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    tensors = [torch.as_tensor(value, dtype=dtype, device=device) for value in values]
    try:
        return list(torch.broadcast_tensors(*tensors))
    except RuntimeError as exc:
        raise ValueError("reliability evidence must be broadcastable") from exc


def compute_muse_reliability(
    confidence: torch.Tensor,
    margin: torch.Tensor,
    js: torch.Tensor,
    proto_distance: torch.Tensor,
    stability: torch.Tensor,
    weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine confidence, margin, disagreement, distance, and stability in ``[0, 1]``."""

    confidence, margin, js, proto_distance, stability = _evidence_tensors(
        [confidence, margin, js, proto_distance, stability]
    )
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    margin = torch.nan_to_num(margin, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    js = torch.nan_to_num(js, nan=1.0, posinf=1.0e6, neginf=0.0).clamp_min(0.0)
    proto_distance = torch.nan_to_num(
        proto_distance,
        nan=1.0e6,
        posinf=1.0e6,
        neginf=0.0,
    ).clamp_min(0.0)
    stability = torch.nan_to_num(stability, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    evidence = torch.stack(
        [confidence, margin, torch.exp(-js), torch.exp(-proto_distance), stability],
        dim=0,
    )
    if weights is None:
        weights = _DEFAULT_RELIABILITY_WEIGHTS
    weight_tensor = torch.as_tensor(weights, dtype=evidence.dtype, device=evidence.device).reshape(-1)
    if weight_tensor.numel() != 5:
        raise ValueError("reliability weights must contain five values")
    if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
        raise ValueError("reliability weights must be finite and non-negative")
    weight_sum = weight_tensor.sum()
    if not torch.isfinite(weight_sum) or weight_sum.item() <= 0.0:
        raise ValueError("reliability weight sum must be greater than zero")
    view_shape = (5,) + (1,) * (evidence.ndim - 1)
    reliability = (evidence * weight_tensor.view(view_shape)).sum(dim=0) / weight_sum
    return reliability.clamp(0.0, 1.0)


def route_muse_reliability(
    reliability: torch.Tensor,
    high_threshold: float,
    low_threshold: float,
) -> MUSERoute:
    """Partition reliability into high (inclusive), mid, and low (exclusive) masks."""

    high_threshold = float(high_threshold)
    low_threshold = float(low_threshold)
    if not all(torch.isfinite(torch.tensor(value)).item() for value in (high_threshold, low_threshold)):
        raise ValueError("reliability thresholds must be finite")
    if not 0.0 <= low_threshold <= high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low <= high <= 1")
    value = reliability if torch.is_tensor(reliability) else torch.as_tensor(reliability)
    if value.is_complex():
        raise TypeError("reliability must be real-valued")
    if not value.is_floating_point():
        value = value.to(dtype=torch.get_default_dtype())
    value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    high = value >= high_threshold
    low = value < low_threshold
    mid = ~(high | low)
    return MUSERoute(high=high, mid=mid, low=low)


def _loss_logits(logits: torch.Tensor, name: str = "logits") -> tuple[torch.Tensor, tuple[int, ...]]:
    """Validate and flatten logits while retaining a gradient path to the input."""

    value = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if value.is_complex():
        raise TypeError(f"{name} must be real-valued")
    if not value.is_floating_point():
        value = value.to(dtype=torch.get_default_dtype())
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    shape = tuple(value.shape[:-1])
    return value.reshape(-1, value.shape[-1]), shape


def _sample_mask(mask: object, sample_count: int, device: torch.device, name: str) -> torch.Tensor:
    """Convert a sample mask to a flat boolean tensor."""

    value = torch.as_tensor(mask, device=device)
    if value.is_complex():
        raise TypeError(f"{name} must be real-valued")
    if value.numel() == 1 and sample_count != 1:
        value = value.reshape(1).expand(sample_count)
    elif value.numel() != sample_count:
        raise ValueError(f"{name} must contain one value per sample")
    return value.reshape(-1).to(dtype=torch.bool)


def _sample_weights(
    weights: object | None,
    sample_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert per-sample weights and reject values that cannot define a mean."""

    if weights is None:
        value = torch.ones(sample_count, device=device, dtype=dtype)
    else:
        value = torch.as_tensor(weights, device=device, dtype=dtype)
        if value.numel() == 1 and sample_count != 1:
            value = value.reshape(1).expand(sample_count)
        elif value.numel() != sample_count:
            raise ValueError("weights must contain one value per sample")
        else:
            value = value.reshape(-1)
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return value.reshape(-1)


def _weighted_sample_mean(
    values: torch.Tensor,
    weights: object | None,
    mask: object,
    graph: torch.Tensor,
) -> torch.Tensor:
    """Reduce sample losses by the selected weight mass, preserving an empty graph."""

    flat_values = values.reshape(-1)
    selected = _sample_mask(mask, flat_values.numel(), flat_values.device, "mask")
    if not bool(selected.any().item()):
        return graph.sum() * 0.0
    flat_weights = _sample_weights(weights, flat_values.numel(), flat_values.device, flat_values.dtype)
    selected_weights = flat_weights[selected]
    denominator = selected_weights.sum().clamp_min(_PROBABILITY_EPS)
    return (flat_values[selected] * selected_weights).sum() / denominator


def weighted_soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    weights: object | None,
    mask: object,
) -> torch.Tensor:
    """Compute a masked, per-sample weighted soft-label cross entropy.

    The denominator is the selected weight mass.  Teacher probabilities are
    detached targets; only the student logits receive identity gradients.
    """

    logits, sample_shape = _loss_logits(student_logits, "student_logits")
    teacher = teacher_prob if torch.is_tensor(teacher_prob) else torch.as_tensor(teacher_prob)
    if teacher.shape != tuple(sample_shape) + (logits.shape[-1],):
        raise ValueError("teacher_prob must have the same shape as student_logits")
    if teacher.is_complex():
        raise TypeError("teacher_prob must be real-valued")
    teacher = teacher.to(device=logits.device, dtype=logits.dtype).reshape_as(logits).detach()
    teacher = _normalize_probability(teacher)
    per_sample = -(teacher * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    return _weighted_sample_mean(per_sample, weights, mask, logits)


def candidate_set_mask(
    prob: torch.Tensor,
    mass: float = 0.75,
    max_classes: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the smallest top-probability candidate set up to ``max_classes``.

    ``active`` is false when the requested mass cannot be reached by the cap.
    The returned mask still exposes the capped top classes, allowing callers to
    retain diagnostics while passing ``active`` as the loss sample mask.
    """

    value = prob if torch.is_tensor(prob) else torch.as_tensor(prob)
    if value.ndim < 1:
        raise ValueError("prob must have a class dimension")
    if value.is_complex():
        raise TypeError("prob must be real-valued")
    if not value.is_floating_point():
        value = value.to(dtype=torch.get_default_dtype())
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    mass = float(mass)
    if not torch.isfinite(torch.tensor(mass)).item() or not 0.75 <= mass <= 1.0:
        raise ValueError("mass must be finite and in [0.75, 1]")
    if (
        not isinstance(max_classes, int)
        or isinstance(max_classes, bool)
        or not 1 <= max_classes <= 3
    ):
        raise ValueError("max_classes must be an integer in [1, 3]")

    class_count = value.shape[-1]
    if class_count == 0:
        raise ValueError("prob must have a non-empty class dimension")
    normalized = _normalize_probability(value)
    flat = normalized.reshape(-1, class_count)
    k = min(max_classes, class_count)
    top_values, top_indices = torch.topk(flat, k=k, dim=-1)
    cumulative = top_values.cumsum(dim=-1)
    reached = cumulative >= mass
    active = reached[:, -1]
    first_reached = reached.to(dtype=torch.long).argmax(dim=-1) + 1
    selected_count = torch.where(active, first_reached, torch.full_like(first_reached, k))
    rank = torch.arange(k, device=value.device).reshape(1, -1)
    selected = rank < selected_count.reshape(-1, 1)
    flat_mask = torch.zeros_like(flat, dtype=torch.bool)
    flat_mask.scatter_(1, top_indices, selected)
    return flat_mask.reshape(value.shape), active.reshape(value.shape[:-1])


def candidate_set_cross_entropy(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: object | None,
    sample_mask: object,
) -> torch.Tensor:
    """Train against the total probability assigned to a candidate set.

    No uniform target is imposed inside the set.  Rows without candidates are
    excluded from the reduction, so an unreachable set cannot create an
    identity gradient.
    """

    flat_logits, sample_shape = _loss_logits(logits)
    candidate_active = None
    if isinstance(candidate_mask, tuple):
        if len(candidate_mask) != 2:
            raise ValueError("candidate_mask tuple must contain mask and active")
        candidates, candidate_active = candidate_mask
    else:
        candidates = candidate_mask
    candidates = candidates if torch.is_tensor(candidates) else torch.as_tensor(candidates)
    expected_shape = tuple(sample_shape) + (flat_logits.shape[-1],)
    if candidates.shape != expected_shape:
        raise ValueError("candidate_mask must have the same shape as logits")
    if candidates.device != flat_logits.device:
        candidates = candidates.to(device=flat_logits.device)
    candidates = candidates.reshape_as(flat_logits).to(dtype=torch.bool)
    sample = _sample_mask(sample_mask, flat_logits.shape[0], flat_logits.device, "sample_mask")
    valid = sample & candidates.any(dim=-1)
    if candidate_active is not None:
        active = _sample_mask(
            candidate_active,
            flat_logits.shape[0],
            flat_logits.device,
            "candidate_active",
        )
        valid = valid & active
    if not bool(valid.any().item()):
        return flat_logits.sum() * 0.0
    probability = torch.softmax(flat_logits, dim=-1)
    candidate_mass = (probability * candidates.to(dtype=probability.dtype)).sum(dim=-1)
    per_sample = -candidate_mass.clamp_min(_PROBABILITY_EPS).log()
    return _weighted_sample_mean(per_sample, weights, valid, flat_logits)


def _python_key_atom(value: Any) -> Any:
    """Convert scalar tensor/sequence key members into hashable Python values."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("memory key members must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, list):
        value = tuple(_python_key_atom(item) for item in value)
    elif isinstance(value, tuple):
        value = tuple(_python_key_atom(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("memory key members must be hashable scalars") from exc
    return value


def _temporal_keys(keys: object) -> list[tuple[Any, ...]]:
    """Normalize a batch of five-member stable keys."""

    if torch.is_tensor(keys):
        if keys.ndim == 1:
            if keys.numel() != 5:
                raise ValueError("each memory key must contain five members")
            raw_keys = [keys.detach().cpu().tolist()]
        elif keys.ndim == 2 and keys.shape[1] == 5:
            raw_keys = keys.detach().cpu().tolist()
        else:
            raise ValueError("keys must have shape [N, 5]")
    elif isinstance(keys, tuple) and len(keys) == 5 and not any(
        isinstance(item, (tuple, list)) for item in keys
    ):
        raw_keys = [keys]
    elif isinstance(keys, list) and len(keys) == 5 and not any(
        isinstance(item, (tuple, list)) for item in keys
    ):
        raw_keys = [keys]
    else:
        raw_keys = list(keys)  # type: ignore[arg-type]
    normalized: list[tuple[Any, ...]] = []
    for key in raw_keys:
        if len(key) != 5:
            raise ValueError("each memory key must contain five members")
        normalized.append(tuple(_python_key_atom(item) for item in key))
    return normalized


def _observation_vector(
    values: object,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Normalize one-dimensional observation values, allowing scalar broadcast."""

    result = torch.as_tensor(values, device=device)
    if result.numel() == 1 and count != 1:
        result = result.reshape(1).expand(count)
    elif result.numel() != count:
        raise ValueError(f"{name} must contain one value per key")
    return result.reshape(-1)


def stable_sample_keys(extra: Mapping[str, object]) -> list[tuple[int, int, int, int, int]]:
    """Return batch-order-independent MUSE sample identities from metadata."""

    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a metadata mapping")
    names = ("rx_i", "day_i", "eq_i", "sig_i", "base_index")
    columns: list[list[object]] = []
    for name in names:
        if name not in extra:
            raise ValueError(f"extra is missing required MUSE key {name!r}")
        value = extra[name]
        if torch.is_tensor(value):
            values = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (str, bytes)):
            values = [value]
        else:
            try:
                values = list(value)  # type: ignore[arg-type]
            except TypeError:
                values = [value]
        columns.append(values)
    count = len(columns[0])
    if count == 0 or any(len(column) != count for column in columns[1:]):
        raise ValueError("MUSE metadata keys must be non-empty and have equal lengths")
    try:
        return [tuple(int(column[index]) for column in columns) for index in range(count)]
    except (TypeError, ValueError) as exc:
        raise ValueError("MUSE metadata keys must contain integer identities") from exc


def select_satellite_student_mask(
    keys: object,
    epoch: int,
    probability: float,
    seed: int,
) -> torch.Tensor:
    """Select satellite students by stable identity, independent of batch order."""

    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError("epoch must be an int")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    probability = _probability(probability)
    normalized_keys = _temporal_keys(keys)
    threshold = int(probability * (1 << 64))
    selected = []
    for rx, day, eq, sig, base_index in normalized_keys:
        payload = f"{seed}|{epoch}|{rx}|{day}|{eq}|{sig}|{base_index}".encode("utf-8")
        draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
        selected.append(draw < threshold)
    return torch.tensor(selected, dtype=torch.bool)


class MUSETemporalMemory:
    """Track per-key pseudo-label runs and expose stable observations after 3 hits."""

    def __init__(self, stability_steps: int = _TEMPORAL_STABILITY_STEPS) -> None:
        if (
            not isinstance(stability_steps, int)
            or isinstance(stability_steps, bool)
            or stability_steps < _TEMPORAL_STABILITY_STEPS
        ):
            raise ValueError("stability_steps must be at least three")
        self.stability_steps = int(stability_steps)
        self._entries: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._frozen = False

    def observe(
        self,
        keys: object,
        predictions: object,
        confidence: object,
        epoch: int,
    ) -> torch.Tensor:
        """Observe one batch and return the stable mask for its keys."""

        normalized_keys = _temporal_keys(keys)
        prediction_tensor = predictions if torch.is_tensor(predictions) else torch.as_tensor(predictions)
        if prediction_tensor.is_complex():
            raise TypeError("predictions must be real-valued")
        prediction_tensor = prediction_tensor.reshape(-1)
        prediction_tensor = _observation_vector(
            prediction_tensor, len(normalized_keys), prediction_tensor.device, "predictions"
        )
        confidence_tensor = _observation_vector(
            confidence, len(normalized_keys), prediction_tensor.device, "confidence"
        )
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be an int")

        stable: list[bool] = []
        for index, key in enumerate(normalized_keys):
            prediction = prediction_tensor[index].detach().cpu().item()
            confidence_value = confidence_tensor[index].detach().cpu().item()
            existing = self._entries.get(key)
            if existing is not None and existing["prediction"] == prediction:
                streak = min(int(existing["streak"]) + 1, self.stability_steps)
            else:
                streak = 1
            is_stable = streak >= self.stability_steps
            stable.append(is_stable)
            if not self._frozen:
                self._entries[key] = {
                    "prediction": prediction,
                    "confidence": float(confidence_value),
                    "epoch": int(epoch),
                    "streak": int(streak),
                }
            elif existing is None:
                is_stable = False
                stable[-1] = False
        return torch.tensor(stable, dtype=torch.bool, device=prediction_tensor.device)

    def freeze(self) -> None:
        self._frozen = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "stability_steps": self.stability_steps,
            "frozen": self._frozen,
            "entries": {
                key: dict(value) for key, value in self._entries.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._frozen:
            raise RuntimeError("cannot load state into a frozen temporal memory")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        steps = state.get("stability_steps", _TEMPORAL_STABILITY_STEPS)
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < _TEMPORAL_STABILITY_STEPS
        ):
            raise ValueError("state stability_steps must be at least three")
        entries = state.get("entries", {})
        if not isinstance(entries, Mapping):
            raise ValueError("state entries must be a mapping")
        restored: dict[tuple[Any, ...], dict[str, Any]] = {}
        for raw_key, raw_entry in entries.items():
            key = _temporal_keys([raw_key])[0]
            if not isinstance(raw_entry, Mapping):
                raise ValueError("each memory entry must be a mapping")
            restored[key] = {
                "prediction": _python_key_atom(raw_entry["prediction"]),
                "confidence": float(raw_entry["confidence"]),
                "epoch": int(raw_entry["epoch"]),
                "streak": int(raw_entry["streak"]),
            }
        self.stability_steps = int(steps)
        self._entries = restored
        self._frozen = bool(state.get("frozen", False))


def _prototype_weight(value: object) -> float:
    """Validate the bounded contribution weight for unlabeled prototypes."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unlabeled_weight must be in [0.05, 0.10]") from exc
    if not torch.isfinite(torch.tensor(result)).item() or not (
        _MIN_UNLABELED_PROTOTYPE_WEIGHT <= result <= _MAX_UNLABELED_PROTOTYPE_WEIGHT
    ):
        raise ValueError("unlabeled_weight must be in [0.05, 0.10]")
    return result


class MUSEClassificationPrototypeBank:
    """EMA class prototypes fed only by high-confidence stable pseudo-labels."""

    def __init__(
        self,
        feature_dim: int | None = None,
        unlabeled_weight: float = _DEFAULT_UNLABELED_PROTOTYPE_WEIGHT,
    ) -> None:
        if feature_dim is not None and (
            not isinstance(feature_dim, int)
            or isinstance(feature_dim, bool)
            or feature_dim < 1
        ):
            raise ValueError("feature_dim must be a positive integer")
        self._feature_dim = feature_dim
        self._unlabeled_weight = _prototype_weight(unlabeled_weight)
        self._prototypes: dict[int, torch.Tensor] = {}
        self._counts: dict[int, float] = {}
        self._domain_counts: dict[tuple[int, Any], float] = {}
        self._frozen = False

    @property
    def feature_dim(self) -> int | None:
        return self._feature_dim

    @property
    def unlabeled_weight(self) -> float:
        return self._unlabeled_weight

    @property
    def prototypes(self) -> dict[int, torch.Tensor]:
        return {class_id: value.clone() for class_id, value in self._prototypes.items()}

    def observe(
        self,
        features: torch.Tensor,
        pseudo: object,
        domains: object,
        high_mask: object,
        stable_mask: object,
        unlabeled_weight: float | None = None,
    ) -> None:
        """Update class prototypes using only ``high_mask & stable_mask`` rows."""

        update_weight = self._unlabeled_weight if unlabeled_weight is None else _prototype_weight(unlabeled_weight)
        value = features if torch.is_tensor(features) else torch.as_tensor(features)
        if value.ndim != 2:
            raise ValueError("features must have shape [N, D]")
        if value.is_complex():
            raise TypeError("features must be real-valued")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        if self._feature_dim is None:
            if self._frozen:
                return
            self._feature_dim = int(value.shape[1])
        if value.shape[1] != self._feature_dim:
            raise ValueError("feature dimension does not match the prototype bank")
        sample_count = value.shape[0]
        pseudo_tensor = _observation_vector(pseudo, sample_count, value.device, "pseudo")
        if torch.is_tensor(domains):
            domain_values = domains.reshape(-1).detach().cpu().tolist()
        elif isinstance(domains, (str, bytes)):
            domain_values = [domains]
        else:
            try:
                domain_values = list(domains)  # type: ignore[arg-type]
            except TypeError:
                domain_values = [domains]
        if len(domain_values) == 1 and sample_count != 1:
            domain_values = domain_values * sample_count
        if len(domain_values) != sample_count:
            raise ValueError("domains must contain one value per feature")
        high = _sample_mask(high_mask, sample_count, value.device, "high_mask")
        stable = _sample_mask(stable_mask, sample_count, value.device, "stable_mask")
        accepted = high & stable
        if self._frozen or not bool(accepted.any().item()):
            return

        detached = value.detach().float()
        labels = pseudo_tensor.detach().cpu().tolist()
        accepted_indices = accepted.detach().cpu().tolist()
        accepted_by_class: dict[int, list[int]] = {}
        for index, (label, is_accepted) in enumerate(zip(labels, accepted_indices)):
            if not is_accepted:
                continue
            class_id = int(label)
            accepted_by_class.setdefault(class_id, []).append(index)
            domain = _python_key_atom(domain_values[index])
            domain_key = (class_id, domain)
            self._domain_counts[domain_key] = self._domain_counts.get(domain_key, 0.0) + 1.0

        for class_id, indices in accepted_by_class.items():
            batch_mean = detached[indices].mean(dim=0)
            old = self._prototypes.get(class_id)
            if old is None:
                self._prototypes[class_id] = batch_mean.clone()
            else:
                old = old.to(device=batch_mean.device, dtype=batch_mean.dtype)
                self._prototypes[class_id] = (1.0 - update_weight) * old + update_weight * batch_mean
            self._counts[class_id] = self._counts.get(class_id, 0.0) + float(len(indices))

    def freeze(self) -> None:
        self._frozen = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self._feature_dim,
            "unlabeled_weight": self._unlabeled_weight,
            "frozen": self._frozen,
            "prototypes": {
                int(class_id): value.clone() for class_id, value in self._prototypes.items()
            },
            "counts": dict(self._counts),
            "domain_counts": dict(self._domain_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._frozen:
            raise RuntimeError("cannot load state into a frozen prototype bank")
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        feature_dim = state.get("feature_dim")
        if feature_dim is not None:
            feature_dim = int(feature_dim)
        if self._feature_dim is not None and feature_dim is not None and self._feature_dim != feature_dim:
            raise ValueError("state feature dimension does not match the prototype bank")
        prototypes = state.get("prototypes", {})
        if not isinstance(prototypes, Mapping):
            raise ValueError("state prototypes must be a mapping")
        restored: dict[int, torch.Tensor] = {}
        inferred_dim = feature_dim
        for raw_class_id, raw_value in prototypes.items():
            class_id = int(raw_class_id)
            tensor = torch.as_tensor(raw_value).detach().clone()
            if tensor.ndim != 1:
                raise ValueError("each prototype must be one-dimensional")
            if inferred_dim is None:
                inferred_dim = int(tensor.numel())
            if tensor.numel() != inferred_dim:
                raise ValueError("state prototypes have inconsistent feature dimensions")
            restored[class_id] = tensor
        self._feature_dim = inferred_dim
        self._unlabeled_weight = _prototype_weight(
            state.get("unlabeled_weight", self._unlabeled_weight)
        )
        self._prototypes = restored
        counts = state.get("counts", {})
        domain_counts = state.get("domain_counts", {})
        self._counts = {int(class_id): float(value) for class_id, value in dict(counts).items()}
        self._domain_counts = {
            (int(key[0]), _python_key_atom(key[1])): float(value)
            for key, value in dict(domain_counts).items()
        }
        self._frozen = bool(state.get("frozen", False))


class MUSETrainingHeads(nn.Module):
    """Training-only local, self-supervised, and nuisance heads for MUSE.

    The local classifier keeps one shared projection and base classifier.  A
    domain selects only a small factorized logit delta, so source-domain
    conditioning does not replicate a full ``z_id -> class`` classifier.
    None of these parameters are part of the deployment state.
    """

    _LOCAL_RANK = 32
    _LOCAL_DELTA_RANK = 4
    _PROJECTION_DIM = 128
    _PREDICTION_HIDDEN_DIM = 64
    _NUISANCE_HIDDEN_DIM = 32

    def __init__(
        self,
        z_id_dim: int,
        z_dom_dim: int,
        num_classes: int,
        num_domains: int,
        nuisance_dim: int,
    ) -> None:
        super().__init__()
        dimensions = {
            "z_id_dim": z_id_dim,
            "z_dom_dim": z_dom_dim,
            "num_classes": num_classes,
            "num_domains": num_domains,
            "nuisance_dim": nuisance_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.z_id_dim = z_id_dim
        self.z_dom_dim = z_dom_dim
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.nuisance_dim = nuisance_dim
        self.local_rank = min(self._LOCAL_RANK, z_id_dim)
        self.local_delta_rank = min(self._LOCAL_DELTA_RANK, self.local_rank, num_classes)
        projection_dim = min(self._PROJECTION_DIM, max(16, z_id_dim))

        self.shared_projection = nn.Linear(z_id_dim, self.local_rank, bias=False)
        self.shared_classifier = nn.Linear(self.local_rank, num_classes)
        self.domain_delta_left = nn.Parameter(
            torch.empty(num_domains, self.local_rank, self.local_delta_rank)
        )
        self.domain_delta_right = nn.Parameter(
            torch.empty(num_domains, self.local_delta_rank, num_classes)
        )

        self.self_projection = nn.Sequential(
            nn.Linear(z_id_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.self_prediction = nn.Sequential(
            nn.Linear(projection_dim, self._PREDICTION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self._PREDICTION_HIDDEN_DIM, projection_dim),
        )
        self.nuisance_head = nn.Sequential(
            nn.Linear(z_dom_dim, self._NUISANCE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self._NUISANCE_HIDDEN_DIM, nuisance_dim),
        )

        nn.init.normal_(self.domain_delta_left, mean=0.0, std=0.02)
        nn.init.normal_(self.domain_delta_right, mean=0.0, std=0.02)

    @staticmethod
    def _feature_matrix(value: torch.Tensor, width: int, name: str, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [N, {width}]")
        if value.is_complex():
            raise TypeError(f"{name} must be real-valued")
        return torch.nan_to_num(value.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)

    def _module_dtype(self) -> torch.dtype:
        return self.shared_projection.weight.dtype

    def _domain_indices(self, domains: object, sample_count: int, device: torch.device) -> torch.Tensor:
        values = domains if torch.is_tensor(domains) else torch.as_tensor(domains)
        if values.is_complex() or values.dtype == torch.bool:
            raise TypeError("domains must be integer-valued")
        values = values.reshape(-1)
        if values.numel() == 1 and sample_count != 1:
            values = values.expand(sample_count)
        if values.numel() != sample_count:
            raise ValueError("domains must contain one value per feature")
        if values.is_floating_point():
            if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
                raise ValueError("domains must contain finite integer values")
        values = values.to(device=device, dtype=torch.long)
        if bool((values < 0).any().item()) or bool((values >= self.num_domains).any().item()):
            raise ValueError(f"domains must be in [0, {self.num_domains})")
        return values

    def local_prob(self, z_id: torch.Tensor, domains: object) -> torch.Tensor:
        """Return source-domain-conditioned class probabilities."""

        features = self._feature_matrix(z_id, self.z_id_dim, "z_id", self._module_dtype())
        domain_indices = self._domain_indices(domains, features.shape[0], features.device)
        projected = self.shared_projection(features)
        logits = self.shared_classifier(projected)
        left = self.domain_delta_left[domain_indices]
        right = self.domain_delta_right[domain_indices]
        delta = torch.bmm(torch.bmm(projected.unsqueeze(1), left), right).squeeze(1)
        logits = torch.nan_to_num(logits + delta, nan=0.0, posinf=0.0, neginf=0.0)
        return F.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)

    def self_supervised_loss(self, z_id_a: torch.Tensor, z_id_b: torch.Tensor) -> torch.Tensor:
        """Compute symmetric stop-gradient negative cosine consistency loss."""

        first = self._feature_matrix(z_id_a, self.z_id_dim, "z_id_a", self._module_dtype())
        second = self._feature_matrix(z_id_b, self.z_id_dim, "z_id_b", self._module_dtype())
        if first.shape != second.shape:
            raise ValueError("z_id_a and z_id_b must have matching shapes")

        projected_a = self.self_projection(first)
        projected_b = self.self_projection(second)
        predicted_a = self.self_prediction(projected_a)
        predicted_b = self.self_prediction(projected_b)
        target_a = F.normalize(projected_a.detach(), dim=-1, eps=1e-8)
        target_b = F.normalize(projected_b.detach(), dim=-1, eps=1e-8)
        predicted_a = F.normalize(predicted_a, dim=-1, eps=1e-8)
        predicted_b = F.normalize(predicted_b, dim=-1, eps=1e-8)
        loss_a = -(predicted_a * target_b).sum(dim=-1).mean()
        loss_b = -(predicted_b * target_a).sum(dim=-1).mean()
        return torch.nan_to_num(0.5 * (loss_a + loss_b), nan=0.0, posinf=0.0, neginf=0.0)

    def _nuisance_prediction(self, z_dom: torch.Tensor) -> torch.Tensor:
        features = self._feature_matrix(z_dom, self.z_dom_dim, "z_dom", self._module_dtype())
        return torch.nan_to_num(
            self.nuisance_head(features),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def nuisance_loss(
        self,
        z_dom: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: object,
    ) -> torch.Tensor:
        """Return masked smooth-L1 regression loss for six nuisance targets."""

        features = self._feature_matrix(z_dom, self.z_dom_dim, "z_dom", self._module_dtype())
        target_tensor = targets if torch.is_tensor(targets) else torch.as_tensor(targets)
        if target_tensor.ndim != 2 or target_tensor.shape != (features.shape[0], self.nuisance_dim):
            raise ValueError(
                f"targets must have shape [{features.shape[0]}, {self.nuisance_dim}]"
            )
        mask = valid_mask if torch.is_tensor(valid_mask) else torch.as_tensor(valid_mask)
        mask = mask.reshape(-1)
        if mask.numel() != features.shape[0]:
            raise ValueError("valid_mask must contain one value per sample")
        mask = mask.to(device=features.device, dtype=torch.bool)
        prediction = self._nuisance_prediction(features)
        if not bool(mask.any().item()):
            return prediction.sum() * 0.0

        target_tensor = target_tensor.to(device=prediction.device, dtype=prediction.dtype)
        return F.smooth_l1_loss(prediction[mask], target_tensor[mask], reduction="mean")

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached checkpoint copy of training-only parameters."""

        return {key: value.detach().clone() for key, value in self.state_dict().items()}

    def deployment_state_dict(self) -> dict[str, torch.Tensor]:
        """Training-only parameters must never enter the Phase2 bundle."""

        return {}


__all__ = [
    "MUSEConfig",
    "MUSEScheduleState",
    "MUSERoute",
    "muse_schedule_for_epoch",
    "geometric_fuse_probabilities",
    "align_source_domain_prior",
    "js_head_disagreement",
    "compute_muse_reliability",
    "route_muse_reliability",
    "weighted_soft_cross_entropy",
    "candidate_set_mask",
    "candidate_set_cross_entropy",
    "stable_sample_keys",
    "select_satellite_student_mask",
    "MUSETemporalMemory",
    "MUSEClassificationPrototypeBank",
    "MUSETrainingHeads",
]
