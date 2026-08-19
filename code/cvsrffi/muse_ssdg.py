"""Stateless MUSE-SSDG training schedule primitives."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import torch


_PROBABILITY_EPS = 1e-8
_DEFAULT_FUSION_WEIGHTS = (0.50, 0.25, 0.25)
_DEFAULT_RATIO_CLIP = (0.5, 2.0)
_DEFAULT_RELIABILITY_WEIGHTS = (0.25, 0.20, 0.20, 0.20, 0.15)


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
]
