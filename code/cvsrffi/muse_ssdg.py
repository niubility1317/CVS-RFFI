"""Stateless MUSE-SSDG training schedule primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
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


def adv3b02_core90_u_satellite_policy(epoch: int) -> tuple[float, tuple[str, ...]]:
    """Return the exact ADV3B02 CORE90 LEO weak schedule for a U_s view."""

    if not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= 200:
        raise ValueError("ADV3B02 CORE90 U satellite epoch must be in [1,200]")
    if epoch <= 40:
        return 0.30, ("leo_clear_weak",)
    if epoch <= 90:
        return 0.60, ("leo_low_elev_weak", "leo_rain_weak")
    return 0.80, (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )


def rc4_tail_transition_scale(
    epoch: int,
    *,
    start_epoch: int = 91,
    ramp_epochs: int = 20,
    floor: float = 0.25,
) -> float:
    """Ramp RC4 all-U alignment back in after the Core90 E91 stage change."""

    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("RC4 tail epoch must be a positive integer")
    if not isinstance(start_epoch, int) or isinstance(start_epoch, bool) or start_epoch < 1:
        raise ValueError("RC4 tail start_epoch must be a positive integer")
    if not isinstance(ramp_epochs, int) or isinstance(ramp_epochs, bool) or ramp_epochs < 1:
        raise ValueError("RC4 tail ramp_epochs must be a positive integer")
    floor = float(floor)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("RC4 tail floor must be finite and in [0,1]")
    if epoch < start_epoch or epoch >= start_epoch + ramp_epochs:
        return 1.0
    progress = (epoch - start_epoch) / float(ramp_epochs)
    return float(floor + (1.0 - floor) * progress)


def select_adv3b02_u_satellite_scenario(epoch: int, batch_index: int, seed: int) -> str:
    """Select one scheduled scenario deterministically without reading TX truth."""

    _, scenarios = adv3b02_core90_u_satellite_policy(int(epoch))
    payload = f"{int(seed)}:{int(epoch)}:{int(batch_index)}".encode("utf-8")
    position = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(scenarios)
    return scenarios[position]


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


@dataclass(frozen=True)
class FastTrustRoute:
    """Mutually exclusive identity routes plus hard-gate evidence."""

    hard: torch.Tensor
    soft: torch.Tensor
    candidate: torch.Tensor
    no_identity: torch.Tensor
    agreement: torch.Tensor
    class_cap: torch.Tensor


@dataclass(frozen=True)
class SATAnchorThresholds:
    """Per-class confidence and margin thresholds calibrated on ``V_cal``."""

    confidence: torch.Tensor
    margin: torch.Tensor


@dataclass(frozen=True)
class SATAnchorRoute:
    """Truth-free routing decision for strict satellite identity supervision."""

    pseudo: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    agreement: torch.Tensor
    strict: torch.Tensor
    trusted: torch.Tensor
    filled: torch.Tensor
    no_identity: torch.Tensor
    class_cap: torch.Tensor
    receiver_cap: torch.Tensor


@dataclass(frozen=True)
class RC4Calibration:
    """Stage-frozen source ``V_cal`` package for risk-calibrated pseudo labels."""

    temperature: float
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    correctness_weight: torch.Tensor
    partial_safety_weight: torch.Tensor
    exclusion_safety_weight: torch.Tensor
    aps_global: float
    aps_by_class: torch.Tensor
    aps_by_domain: torch.Tensor
    hard_risk_threshold: float
    partial_safety_threshold: float
    hard_ready: bool
    partial_ready: bool
    negative_ready: bool
    hard_precision: float
    hard_coverage: float
    partial_coverage: float
    partial_precision: float
    partial_selected_coverage: float
    partial_mean_size: float
    negative_false_exclusion: float
    calibration_rows: int
    crossfit_folds: int
    num_classes: int
    num_domains: int


@dataclass(frozen=True)
class RC4Route:
    """Mutually exclusive H/P/N/R routing without an identity fill quota."""

    pseudo: torch.Tensor
    fused_probability: torch.Tensor
    risk: torch.Tensor
    p_correct: torch.Tensor
    p_set_safe: torch.Tensor
    p_exclusion_safe: torch.Tensor
    partial_threshold: torch.Tensor
    candidate_mask: torch.Tensor
    excluded_mask: torch.Tensor
    hard: torch.Tensor
    partial: torch.Tensor
    negative: torch.Tensor
    representation: torch.Tensor
    agreement: torch.Tensor
    disagreement: torch.Tensor
    weights: torch.Tensor
    class_receiver_cap: torch.Tensor


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


def route_fasttrust(
    reliability: torch.Tensor,
    stable: torch.Tensor,
    evidence_probabilities: Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    high_threshold: float,
    low_threshold: float,
    hard_max_fraction: float = 0.25,
    identity_max_fraction: float = 0.50,
    class_balanced_cap: bool = True,
) -> FastTrustRoute:
    """Route identity supervision with strict hard evidence and deterministic caps."""

    value = reliability if torch.is_tensor(reliability) else torch.as_tensor(reliability)
    if value.ndim != 1:
        raise ValueError("FastTrust reliability must be one-dimensional")
    if value.is_complex():
        raise TypeError("FastTrust reliability must be real-valued")
    if not value.is_floating_point():
        value = value.float()
    value = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    batch_size = int(value.numel())
    stable_mask = torch.as_tensor(stable, device=value.device, dtype=torch.bool).reshape(-1)
    if stable_mask.numel() != batch_size:
        raise ValueError("FastTrust stable must contain one value per sample")
    heads = (
        list(evidence_probabilities.values())
        if isinstance(evidence_probabilities, Mapping)
        else list(evidence_probabilities)
    )
    if len(heads) != 3:
        raise ValueError("FastTrust requires exactly three identity evidence heads")
    normalized_heads = _probability_heads(heads)
    if any(head.ndim != 2 or int(head.shape[0]) != batch_size for head in normalized_heads):
        raise ValueError("FastTrust evidence heads must have shape [batch, classes]")
    predictions = torch.stack(
        [_normalize_probability(head).argmax(dim=-1) for head in normalized_heads],
        dim=0,
    )
    agreement = (predictions == predictions[0].unsqueeze(0)).all(dim=0)
    pseudo = predictions[0]

    hard_fraction = float(hard_max_fraction)
    identity_fraction = float(identity_max_fraction)
    if not (
        math.isfinite(hard_fraction)
        and math.isfinite(identity_fraction)
        and 0.0 <= hard_fraction <= identity_fraction <= 1.0
    ):
        raise ValueError(
            "FastTrust fractions must satisfy 0 <= hard_max_fraction <= identity_max_fraction <= 1"
        )
    base = route_muse_reliability(value, high_threshold, low_threshold)
    hard_limit = int(math.floor(batch_size * hard_fraction + 1e-12))
    identity_limit = int(math.floor(batch_size * identity_fraction + 1e-12))
    hard = torch.zeros(batch_size, device=value.device, dtype=torch.bool)
    class_cap = torch.zeros_like(hard)

    def ranked_indices(mask: torch.Tensor) -> list[int]:
        indices = mask.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
        scores = value.detach().cpu().tolist()
        return sorted(indices, key=lambda index: (-float(scores[index]), int(index)))

    hard_eligible = base.high & stable_mask & agreement
    active_classes = sorted(
        int(item)
        for item in torch.unique(pseudo[hard_eligible]).detach().cpu().tolist()
    )
    if hard_limit > 0 and active_classes:
        if bool(class_balanced_cap):
            per_class_limit = int(math.ceil(hard_limit / float(len(active_classes))))
            selected: list[int] = []
            for class_id in active_classes:
                class_mask = hard_eligible & (pseudo == class_id)
                selected.extend(ranked_indices(class_mask)[:per_class_limit])
            selected = sorted(
                selected, key=lambda index: (-float(value[index].item()), int(index))
            )[:hard_limit]
        else:
            selected = ranked_indices(hard_eligible)[:hard_limit]
        if selected:
            selected_tensor = torch.as_tensor(selected, device=value.device, dtype=torch.long)
            hard[selected_tensor] = True
            class_cap[selected_tensor] = True

    remaining = max(0, identity_limit - int(hard.sum().item()))
    soft = torch.zeros_like(hard)
    soft_pool = base.mid | (base.high & ~hard)
    soft_selected = ranked_indices(soft_pool)[:remaining]
    if soft_selected:
        soft[torch.as_tensor(soft_selected, device=value.device, dtype=torch.long)] = True
    remaining -= len(soft_selected)
    candidate = torch.zeros_like(hard)
    candidate_selected = ranked_indices(base.low)[: max(0, remaining)]
    if candidate_selected:
        candidate[
            torch.as_tensor(candidate_selected, device=value.device, dtype=torch.long)
        ] = True
    no_identity = ~(hard | soft | candidate)
    return FastTrustRoute(
        hard=hard,
        soft=soft,
        candidate=candidate,
        no_identity=no_identity,
        agreement=agreement,
        class_cap=class_cap,
    )


def fuse_anchor_ema_probabilities(
    anchor_probabilities: torch.Tensor,
    ema_probabilities: torch.Tensor,
    *,
    beta: float = 0.5,
) -> torch.Tensor:
    """Fuse frozen-anchor and EMA probabilities with a normalized geometric mean."""

    anchor, ema = _probability_heads([anchor_probabilities, ema_probabilities])
    if anchor.shape != ema.shape or anchor.ndim != 2:
        raise ValueError("anchor and EMA probabilities must have matching [batch, classes] shape")
    weight = float(beta)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("beta must be finite and in [0, 1]")
    anchor = _normalize_probability(anchor).clamp_min(_PROBABILITY_EPS)
    ema = _normalize_probability(ema).clamp_min(_PROBABILITY_EPS)
    log_fused = weight * anchor.log() + (1.0 - weight) * ema.log()
    return F.softmax(log_fused, dim=-1)


def _rc4_fused_probability(
    anchor_logits: torch.Tensor,
    ema_logits_1: torch.Tensor,
    ema_logits_2: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three calibrated teacher distributions and their log-opinion pool."""

    anchor, ema_1, ema_2 = _probability_heads(
        [anchor_logits.float(), ema_logits_1.float(), ema_logits_2.float()]
    )
    if anchor.ndim != 2:
        raise ValueError("RC4 teacher logits must have shape [batch, classes]")
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("RC4 temperature must be finite and positive")
    anchor_p = F.softmax(anchor / scale, dim=-1)
    ema_1_p = F.softmax(ema_1 / scale, dim=-1)
    ema_2_p = F.softmax(ema_2 / scale, dim=-1)
    fused = geometric_fuse_probabilities(
        [anchor_p, ema_1_p, ema_2_p], weights=(0.5, 0.25, 0.25)
    )
    return anchor_p, ema_1_p, ema_2_p, fused


def _rc4_features(
    anchor_p: torch.Tensor,
    ema_1_p: torch.Tensor,
    ema_2_p: torch.Tensor,
    fused: torch.Tensor,
    z_norm: torch.Tensor,
) -> torch.Tensor:
    """Build the seven truth-free correctness features defined by RC4."""

    top = fused.topk(min(2, fused.shape[1]), dim=-1).values
    confidence = top[:, 0]
    margin = top[:, 0] - (top[:, 1] if top.shape[1] == 2 else 0.0)
    entropy = -(fused * fused.clamp_min(_PROBABILITY_EPS).log()).sum(dim=-1)
    entropy = entropy / max(math.log(max(2, int(fused.shape[1]))), _PROBABILITY_EPS)
    ema_mean = _normalize_probability(0.5 * (ema_1_p + ema_2_p))
    anchor_ema_js = js_head_disagreement([anchor_p, ema_mean])
    ema_view_js = js_head_disagreement([ema_1_p, ema_2_p])
    norm = torch.as_tensor(z_norm, device=fused.device, dtype=fused.dtype).reshape(-1)
    if norm.numel() != fused.shape[0]:
        raise ValueError("z_norm must contain one value per RC4 row")
    norm = torch.log1p(torch.nan_to_num(norm, nan=0.0, posinf=1.0e6, neginf=0.0).clamp_min(0.0))
    predictions = torch.stack(
        [anchor_p.argmax(dim=-1), ema_1_p.argmax(dim=-1), ema_2_p.argmax(dim=-1)],
        dim=0,
    )
    agreement = (predictions == predictions[0:1]).all(dim=0).to(fused.dtype)
    return torch.stack(
        [confidence, margin, entropy, anchor_ema_js, ema_view_js, norm, agreement],
        dim=-1,
    )


def _rc4_design_matrix(
    features: torch.Tensor,
    predicted: torch.Tensor,
    domains: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    num_classes: int,
    num_domains: int,
) -> torch.Tensor:
    standardized = (features - feature_mean) / feature_scale.clamp_min(1e-6)
    class_one_hot = F.one_hot(predicted.long(), num_classes=int(num_classes)).to(standardized.dtype)
    domain_one_hot = F.one_hot(domains.long(), num_classes=int(num_domains)).to(standardized.dtype)
    return torch.cat(
        [
            torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype),
            standardized,
            class_one_hot,
            domain_one_hot,
        ],
        dim=1,
    )


def _fit_rc4_logistic(design: torch.Tensor, target: torch.Tensor, l2: float) -> torch.Tensor:
    """Fit a tiny regularized logistic model by bounded Newton iterations."""

    x = design.detach().double().cpu()
    y = target.detach().double().cpu().reshape(-1)
    if x.shape[0] != y.numel() or x.shape[0] == 0:
        raise ValueError("RC4 logistic inputs must be non-empty and aligned")
    beta = torch.zeros(x.shape[1], dtype=torch.float64)
    penalty = torch.eye(x.shape[1], dtype=torch.float64) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(12):
        probability = torch.sigmoid(x @ beta).clamp(1e-6, 1.0 - 1e-6)
        gradient = x.t().mv(probability - y) / float(x.shape[0]) + penalty.mv(beta)
        curvature = probability * (1.0 - probability)
        hessian = (x.t() @ (x * curvature.unsqueeze(1))) / float(x.shape[0]) + penalty
        hessian = hessian + torch.eye(x.shape[1], dtype=x.dtype) * 1e-6
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            step = torch.linalg.pinv(hessian).mv(gradient)
        beta = (beta - step).clamp(-8.0, 8.0)
        if float(step.abs().max().item()) < 1e-6:
            break
    return beta.float()


def _rc4_calibrated_probability(design: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return a finite, non-saturated calibrated event probability."""

    logit = design @ weight.to(design.device, design.dtype)
    return torch.sigmoid(logit.clamp(-12.0, 12.0)).clamp(1e-4, 1.0 - 1e-4)


def _rc4_quantile(values: torch.Tensor, coverage: float) -> float:
    ordered = values.detach().float().reshape(-1).sort().values
    if ordered.numel() == 0:
        return float("nan")
    rank = min(
        int(ordered.numel()) - 1,
        max(0, int(math.ceil((ordered.numel() + 1) * float(coverage))) - 1),
    )
    return float(ordered[rank].item())


def _rc4_aps_mask(probability: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Return APS-style variable-size sets, always including the crossing class."""

    values, indices = probability.sort(dim=-1, descending=True)
    before = values.cumsum(dim=-1) - values
    selected = before < thresholds.reshape(-1, 1)
    mask = torch.zeros_like(selected)
    mask.scatter_(1, indices, selected)
    return mask


def build_rc4_calibration(
    anchor_logits: torch.Tensor,
    ema_logits_1: torch.Tensor,
    ema_logits_2: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    z_norm: torch.Tensor,
    *,
    num_classes: int,
    num_domains: int,
    folds: int = 5,
    l2: float = 0.01,
    min_stratum_samples: int = 16,
    hard_precision_target: float = 0.98,
    hard_min_coverage: float = 0.01,
    partial_coverage_target: float = 0.95,
    partial_precision_target: float = 0.98,
    partial_min_coverage: float = 0.01,
    partial_candidate_max_classes: int = 3,
    negative_false_exclusion_target: float = 0.01,
) -> RC4Calibration:
    """Fit temperature, cross-fitted correctness risk, and APS thresholds on ``V_cal``."""

    labels = torch.as_tensor(labels, device=anchor_logits.device).long().reshape(-1)
    domains = torch.as_tensor(domains, device=anchor_logits.device).long().reshape(-1)
    rows = int(labels.numel())
    if rows == 0 or domains.numel() != rows:
        raise ValueError("RC4 V_cal labels/domains must be non-empty and aligned")
    if bool((domains < 0).any()) or bool((domains >= int(num_domains)).any()):
        raise ValueError("RC4 V_cal domains are outside the declared source-domain range")
    mean_logits = (anchor_logits.float() + ema_logits_1.float() + ema_logits_2.float()) / 3.0
    temperatures = torch.linspace(0.25, 4.0, steps=76, device=mean_logits.device)
    losses = torch.stack(
        [F.cross_entropy(mean_logits / value, labels, reduction="mean") for value in temperatures]
    )
    temperature = float(temperatures[int(losses.argmin().item())].item())
    anchor_p, ema_1_p, ema_2_p, fused = _rc4_fused_probability(
        anchor_logits, ema_logits_1, ema_logits_2, temperature
    )
    features = _rc4_features(anchor_p, ema_1_p, ema_2_p, fused, z_norm)
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-3)
    predicted = fused.argmax(dim=-1)
    correct = predicted.eq(labels).float()
    design = _rc4_design_matrix(
        features,
        predicted,
        domains,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        num_classes=int(num_classes),
        num_domains=int(num_domains),
    )
    unique_domains = torch.unique(domains, sorted=True)
    fold_count = max(1, min(int(folds), int(unique_domains.numel())))
    oof_risk = torch.empty(rows, device=design.device, dtype=design.dtype)
    if fold_count > 1:
        for fold in range(fold_count):
            held_domains = unique_domains[
                torch.arange(unique_domains.numel(), device=domains.device).remainder(fold_count).eq(fold)
            ]
            held = (domains.reshape(-1, 1) == held_domains.reshape(1, -1)).any(dim=1)
            train = ~held
            beta_fold = _fit_rc4_logistic(design[train], correct[train], float(l2)).to(design.device)
            oof_risk[held] = _rc4_calibrated_probability(design[held], beta_fold)
    else:
        beta_fold = _fit_rc4_logistic(design, correct, float(l2)).to(design.device)
        oof_risk = _rc4_calibrated_probability(design, beta_fold)
    final_weight = _fit_rc4_logistic(design, correct, float(l2))

    order = torch.argsort(oof_risk, descending=True, stable=True)
    ordered_correct = correct[order]
    precision = ordered_correct.cumsum(0) / torch.arange(
        1, rows + 1, device=correct.device, dtype=correct.dtype
    )
    coverage = torch.arange(1, rows + 1, device=correct.device, dtype=correct.dtype) / float(rows)
    valid_hard = (precision >= float(hard_precision_target)) & (coverage >= float(hard_min_coverage))
    if bool(valid_hard.any()):
        hard_end = int(valid_hard.nonzero(as_tuple=False)[-1].item())
        hard_threshold = float(oof_risk[order[hard_end]].item())
        hard_precision = float(precision[hard_end].item())
        hard_coverage = float(coverage[hard_end].item())
        hard_ready = True
    else:
        hard_threshold = 1.0
        hard_precision = 0.0
        hard_coverage = 0.0
        hard_ready = False

    sorted_probability, sorted_index = fused.sort(dim=-1, descending=True)
    cumulative = sorted_probability.cumsum(dim=-1)
    inverse_rank = torch.empty_like(sorted_index)
    rank = torch.arange(int(num_classes), device=fused.device).expand_as(sorted_index)
    inverse_rank.scatter_(1, sorted_index, rank)
    true_rank = inverse_rank.gather(1, labels.reshape(-1, 1)).squeeze(1)
    true_score = cumulative.gather(1, true_rank.reshape(-1, 1)).squeeze(1)
    aps_target = max(
        float(partial_coverage_target),
        1.0 - float(negative_false_exclusion_target),
    )
    aps_global = _rc4_quantile(true_score, aps_target)
    aps_class = fused.new_full((int(num_classes),), float("nan"))
    aps_domain = fused.new_full((int(num_domains),), float("nan"))
    for class_id in range(int(num_classes)):
        selected = labels.eq(class_id)
        if int(selected.sum().item()) >= int(min_stratum_samples):
            aps_class[class_id] = _rc4_quantile(true_score[selected], aps_target)
    for domain_id in range(int(num_domains)):
        selected = domains.eq(domain_id)
        if int(selected.sum().item()) >= int(min_stratum_samples):
            aps_domain[domain_id] = _rc4_quantile(true_score[selected], aps_target)
    row_threshold = aps_class[predicted]
    row_threshold = torch.where(torch.isfinite(row_threshold), row_threshold, aps_domain[domains])
    row_threshold = torch.where(
        torch.isfinite(row_threshold), row_threshold, fused.new_full((rows,), aps_global)
    )
    candidate = _rc4_aps_mask(fused, row_threshold)
    contains_truth = candidate.gather(1, labels.reshape(-1, 1)).squeeze(1)
    containment_target = contains_truth.float()
    oof_partial_safety = torch.empty(rows, device=design.device, dtype=design.dtype)
    if fold_count > 1:
        for fold in range(fold_count):
            held_domains = unique_domains[
                torch.arange(unique_domains.numel(), device=domains.device).remainder(fold_count).eq(fold)
            ]
            held = (domains.reshape(-1, 1) == held_domains.reshape(1, -1)).any(dim=1)
            train = ~held
            beta_fold = _fit_rc4_logistic(
                design[train], containment_target[train], float(l2)
            ).to(design.device)
            oof_partial_safety[held] = _rc4_calibrated_probability(
                design[held], beta_fold
            )
    else:
        beta_fold = _fit_rc4_logistic(design, containment_target, float(l2)).to(
            design.device
        )
        oof_partial_safety = _rc4_calibrated_probability(design, beta_fold)
    partial_safety_weight = _fit_rc4_logistic(design, containment_target, float(l2))
    exclusion_safety_weight = _fit_rc4_logistic(design, containment_target, float(l2))
    partial_coverage = float(contains_truth.float().mean().item())
    candidate_size = candidate.sum(dim=1)
    partial_mean_size = float(candidate_size.float().mean().item())
    partial_eligible = candidate_size.ge(2) & candidate_size.le(
        int(partial_candidate_max_classes)
    )
    eligible_order = torch.argsort(
        oof_partial_safety[partial_eligible], descending=True, stable=True
    )
    eligible_ids = partial_eligible.nonzero(as_tuple=False).reshape(-1)[eligible_order]
    if eligible_ids.numel() > 0:
        ordered_safe = containment_target[eligible_ids]
        partial_precision_curve = ordered_safe.cumsum(0) / torch.arange(
            1,
            eligible_ids.numel() + 1,
            device=ordered_safe.device,
            dtype=ordered_safe.dtype,
        )
        partial_coverage_curve = torch.arange(
            1,
            eligible_ids.numel() + 1,
            device=ordered_safe.device,
            dtype=ordered_safe.dtype,
        ) / float(rows)
        valid_partial = (
            partial_precision_curve.ge(float(partial_precision_target))
            & partial_coverage_curve.ge(float(partial_min_coverage))
        )
    else:
        partial_precision_curve = fused.new_empty(0)
        partial_coverage_curve = fused.new_empty(0)
        valid_partial = torch.zeros(0, dtype=torch.bool, device=fused.device)
    if bool(valid_partial.any()):
        partial_end = int(valid_partial.nonzero(as_tuple=False)[-1].item())
        partial_threshold = float(oof_partial_safety[eligible_ids[partial_end]].item())
        partial_precision = float(partial_precision_curve[partial_end].item())
        partial_selected_coverage = float(partial_coverage_curve[partial_end].item())
        partial_ready = partial_mean_size <= 2.5
    else:
        partial_threshold = 1.0
        partial_precision = 0.0
        partial_selected_coverage = 0.0
        partial_ready = False
    false_exclusion = 1.0 - partial_coverage
    return RC4Calibration(
        temperature=temperature,
        feature_mean=feature_mean.detach().cpu(),
        feature_scale=feature_scale.detach().cpu(),
        correctness_weight=final_weight.detach().cpu(),
        partial_safety_weight=partial_safety_weight.detach().cpu(),
        exclusion_safety_weight=exclusion_safety_weight.detach().cpu(),
        aps_global=float(aps_global),
        aps_by_class=aps_class.detach().cpu(),
        aps_by_domain=aps_domain.detach().cpu(),
        hard_risk_threshold=hard_threshold,
        partial_safety_threshold=partial_threshold,
        hard_ready=bool(hard_ready),
        partial_ready=bool(partial_ready),
        negative_ready=bool(false_exclusion <= float(negative_false_exclusion_target)),
        hard_precision=hard_precision,
        hard_coverage=hard_coverage,
        partial_coverage=partial_coverage,
        partial_precision=partial_precision,
        partial_selected_coverage=partial_selected_coverage,
        partial_mean_size=partial_mean_size,
        negative_false_exclusion=float(false_exclusion),
        calibration_rows=rows,
        crossfit_folds=fold_count,
        num_classes=int(num_classes),
        num_domains=int(num_domains),
    )


def apply_rc4_quality_budget(
    hard: torch.Tensor,
    partial: torch.Tensor,
    weights: torch.Tensor,
    *,
    total_budget: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one exact H-first effective budget and let P fill only the residual."""

    hard = torch.as_tensor(hard, device=weights.device, dtype=torch.bool).reshape(-1)
    partial = torch.as_tensor(partial, device=weights.device, dtype=torch.bool).reshape(-1)
    adjusted = torch.as_tensor(weights).clone().reshape(-1)
    if hard.numel() != adjusted.numel() or partial.numel() != adjusted.numel():
        raise ValueError("RC4 quality-budget masks and weights must be aligned")
    if bool((hard & partial).any()):
        raise ValueError("RC4 quality-budget H and P masks must be disjoint")
    fraction = float(total_budget)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("RC4 total effective identity budget must be finite and in [0,1]")
    adjusted = torch.nan_to_num(adjusted, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    kept_hard = torch.zeros_like(hard)
    kept_partial = torch.zeros_like(partial)
    remaining = float(adjusted.numel()) * fraction

    def consume(mask: torch.Tensor, kept: torch.Tensor) -> None:
        nonlocal remaining
        ordered = sorted(
            mask.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist(),
            key=lambda index: (-float(adjusted[index].detach().cpu().item()), int(index)),
        )
        for index in ordered:
            value = float(adjusted[index].detach().cpu().item())
            if value <= 0.0 or remaining <= 1e-12:
                adjusted[index] = 0.0
                continue
            used = min(value, remaining)
            adjusted[index] = used
            kept[index] = True
            remaining -= used

    consume(hard, kept_hard)
    consume(partial, kept_partial)
    adjusted = torch.where(kept_hard | kept_partial, adjusted, torch.zeros_like(adjusted))
    return kept_hard, kept_partial, adjusted


def route_fasttrust_rc4(
    anchor_logits: torch.Tensor,
    ema_logits_1: torch.Tensor,
    ema_logits_2: torch.Tensor,
    *,
    domains: torch.Tensor,
    receivers: torch.Tensor,
    z_norm: torch.Tensor,
    calibration: RC4Calibration,
    hard_max_fraction: float = 0.25,
    candidate_max_classes: int = 3,
    partial_min_risk: float = 0.50,
    partial_effective_budget: float = 0.10,
    negative_effective_budget: float = 0.10,
    total_identity_effective_budget: float = 0.0,
    use_calibrated_partial_threshold: bool = False,
    enable_hard: bool = True,
    enable_partial: bool = True,
    enable_negative: bool = True,
    class_receiver_cap: bool = True,
    use_calibrated_risk: bool = True,
) -> RC4Route:
    """Apply stage-frozen source risk rules to U rows without reading TX truth."""

    anchor_p, ema_1_p, ema_2_p, fused = _rc4_fused_probability(
        anchor_logits, ema_logits_1, ema_logits_2, calibration.temperature
    )
    rows, classes = fused.shape
    if classes != calibration.num_classes:
        raise ValueError("RC4 calibration class count does not match teacher logits")
    domains = torch.as_tensor(domains, device=fused.device).long().reshape(-1)
    receivers = torch.as_tensor(receivers, device=fused.device).long().reshape(-1)
    if domains.numel() != rows or receivers.numel() != rows:
        raise ValueError("RC4 domains/receivers must contain one value per U row")
    features = _rc4_features(anchor_p, ema_1_p, ema_2_p, fused, z_norm)
    predicted = fused.argmax(dim=-1)
    design = _rc4_design_matrix(
        features,
        predicted,
        domains,
        feature_mean=calibration.feature_mean.to(fused.device, fused.dtype),
        feature_scale=calibration.feature_scale.to(fused.device, fused.dtype),
        num_classes=calibration.num_classes,
        num_domains=calibration.num_domains,
    )
    p_correct = (
        _rc4_calibrated_probability(design, calibration.correctness_weight)
        if bool(use_calibrated_risk)
        else fused.max(dim=-1).values
    )
    p_set_safe = (
        _rc4_calibrated_probability(design, calibration.partial_safety_weight)
        if bool(use_calibrated_risk)
        else fused.max(dim=-1).values
    )
    p_exclusion_safe = (
        _rc4_calibrated_probability(design, calibration.exclusion_safety_weight)
        if bool(use_calibrated_risk)
        else p_set_safe
    )
    risk = p_correct
    threshold = calibration.aps_by_class.to(fused.device, fused.dtype)[predicted]
    domain_threshold = calibration.aps_by_domain.to(fused.device, fused.dtype)[domains]
    threshold = torch.where(torch.isfinite(threshold), threshold, domain_threshold)
    threshold = torch.where(
        torch.isfinite(threshold), threshold, fused.new_full((rows,), calibration.aps_global)
    )
    candidate = _rc4_aps_mask(fused, threshold)
    set_size = candidate.sum(dim=-1)
    predictions = torch.stack(
        [anchor_p.argmax(dim=-1), ema_1_p.argmax(dim=-1), ema_2_p.argmax(dim=-1)], dim=0
    )
    agreement = (predictions == predictions[0:1]).all(dim=0)
    disagreement = js_head_disagreement([anchor_p, ema_1_p, ema_2_p])
    hard_threshold = (
        float(calibration.hard_risk_threshold) if bool(use_calibrated_risk) else 0.95
    )
    hard_ready = bool(calibration.hard_ready) if bool(use_calibrated_risk) else True
    hard_eligible = (
        bool(enable_hard)
        & hard_ready
        & set_size.eq(1)
        & risk.ge(hard_threshold)
        & agreement
    )
    limit = int(math.floor(rows * float(hard_max_fraction) + 1e-12))
    hard = torch.zeros(rows, dtype=torch.bool, device=fused.device)
    cap_mask = torch.zeros_like(hard)
    score = risk.detach().cpu().tolist()

    def ranked(mask: torch.Tensor) -> list[int]:
        ids = mask.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
        return sorted(ids, key=lambda index: (-float(score[index]), int(index)))

    if limit > 0:
        selected: list[int] = []
        active_classes = sorted(int(value) for value in torch.unique(predicted[hard_eligible]).cpu().tolist())
        per_class = int(math.ceil(limit / float(max(1, len(active_classes)))))
        for class_id in active_classes:
            class_mask = hard_eligible & predicted.eq(class_id)
            if class_receiver_cap:
                active_receivers = sorted(
                    int(value) for value in torch.unique(receivers[class_mask]).cpu().tolist()
                )
                per_cell = int(math.ceil(per_class / float(max(1, len(active_receivers)))))
                class_selected: list[int] = []
                for receiver_id in active_receivers:
                    class_selected.extend(ranked(class_mask & receivers.eq(receiver_id))[:per_cell])
                selected.extend(sorted(class_selected, key=lambda index: (-score[index], index))[:per_class])
            else:
                selected.extend(ranked(class_mask)[:per_class])
        selected = sorted(selected, key=lambda index: (-score[index], index))[:limit]
        if selected:
            tensor = torch.as_tensor(selected, device=fused.device, dtype=torch.long)
            hard[tensor] = True
            cap_mask[tensor] = True
    partial_threshold = (
        float(calibration.partial_safety_threshold)
        if bool(use_calibrated_risk) and bool(use_calibrated_partial_threshold)
        else float(partial_min_risk)
    )
    partial = (
        ~hard
        & bool(enable_partial)
        & bool(calibration.partial_ready)
        & set_size.ge(2)
        & set_size.le(int(candidate_max_classes))
        & p_set_safe.ge(partial_threshold)
    )
    negative = (
        ~(hard | partial)
        & bool(enable_negative)
        & bool(calibration.negative_ready)
        & set_size.gt(0)
        & set_size.lt(classes)
        & p_exclusion_safe.ge(float(partial_min_risk))
    )
    route_probability = torch.where(hard, p_correct, torch.where(partial, p_set_safe, p_exclusion_safe))
    risk_floor = torch.where(
        hard,
        risk.new_full((rows,), hard_threshold),
        torch.where(
            partial,
            risk.new_full((rows,), partial_threshold),
            torch.where(
                negative,
                risk.new_full((rows,), float(partial_min_risk)),
                torch.zeros_like(risk),
            ),
        ),
    )
    risk_weight = ((route_probability - risk_floor) / (1.0 - risk_floor).clamp_min(1e-6)).clamp(0.0, 1.0).square()
    agree_weight = torch.exp(-2.0 * disagreement)
    set_weight = set_size.clamp_min(1).to(fused.dtype).reciprocal()
    balance = torch.ones_like(risk)
    routed = hard | partial | negative
    if bool(routed.any()):
        cells = predicted * (int(receivers.max().item()) + 1) + receivers
        for state in (hard, partial, negative):
            if not bool(state.any()):
                continue
            state_cells = cells[state]
            counts = torch.bincount(state_cells, minlength=int(cells.max().item()) + 1).float()
            mean_count = counts[counts > 0].mean().clamp_min(1.0)
            balance[state] = torch.sqrt(
                mean_count / counts[state_cells].clamp_min(1.0)
            ).clamp(max=4.0).to(balance.dtype)
    weights = (risk_weight * agree_weight * set_weight * balance).clamp(0.0, 4.0)
    weights = torch.where(routed, weights, torch.zeros_like(weights))

    def apply_effective_budget(mask: torch.Tensor, budget_fraction: float) -> torch.Tensor:
        budget_fraction = float(budget_fraction)
        if not math.isfinite(budget_fraction) or not 0.0 <= budget_fraction <= 1.0:
            raise ValueError("RC4 effective budget must be finite and in [0,1]")
        selected = torch.zeros_like(mask)
        budget = float(rows) * budget_fraction
        if budget <= 0.0 or not bool(mask.any()):
            return selected
        ordered = sorted(
            mask.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist(),
            key=lambda index: (-float(weights[index].detach().cpu().item()), int(index)),
        )
        used = 0.0
        for index in ordered:
            value = float(weights[index].detach().cpu().item())
            if value <= 0.0:
                continue
            if used + value <= budget + 1e-12:
                selected[index] = True
                used += value
        if not bool(selected.any()) and ordered and budget > 0.0:
            selected[ordered[0]] = True
            weights[ordered[0]] = min(float(weights[ordered[0]].item()), budget)
        return selected

    if float(total_identity_effective_budget) > 0.0:
        if bool(enable_negative):
            raise ValueError("RC4 total identity quality budget requires negative routing disabled")
        hard, partial, weights = apply_rc4_quality_budget(
            hard,
            partial,
            weights,
            total_budget=float(total_identity_effective_budget),
        )
        negative = torch.zeros_like(negative)
        cap_mask = cap_mask & hard
    else:
        kept_partial = apply_effective_budget(partial, partial_effective_budget)
        kept_negative = apply_effective_budget(negative, negative_effective_budget)
        partial = kept_partial
        negative = kept_negative
    routed = hard | partial | negative
    weights = torch.where(routed, weights, torch.zeros_like(weights))
    representation = ~routed
    return RC4Route(
        pseudo=predicted,
        fused_probability=fused,
        risk=risk,
        p_correct=p_correct,
        p_set_safe=p_set_safe,
        p_exclusion_safe=p_exclusion_safe,
        partial_threshold=fused.new_tensor(partial_threshold),
        candidate_mask=candidate,
        excluded_mask=~candidate,
        hard=hard,
        partial=partial,
        negative=negative,
        representation=representation,
        agreement=agreement,
        disagreement=disagreement,
        weights=weights,
        class_receiver_cap=cap_mask,
    )


def rc4_identity_losses(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    *,
    pseudo: torch.Tensor,
    candidate_mask: torch.Tensor,
    hard_mask: torch.Tensor,
    partial_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    weights: torch.Tensor,
    full_unlabeled_batch_size: int,
    enable_partial_set: bool = True,
    enable_partial_conditional: bool = True,
    enable_negative_set: bool = True,
) -> dict[str, torch.Tensor]:
    """Return H/P/N losses normalized by the complete physical U batch."""

    logits, _ = _loss_logits(student_logits, "student_logits")
    rows, classes = logits.shape
    denominator = int(full_unlabeled_batch_size)
    if denominator <= 0 or rows != denominator:
        raise ValueError("RC4 student rows must equal full_unlabeled_batch_size")
    teacher = torch.as_tensor(teacher_probability, device=logits.device, dtype=logits.dtype)
    if teacher.shape != logits.shape:
        raise ValueError("RC4 teacher_probability must match student_logits")
    teacher = _normalize_probability(teacher.detach())
    candidates = torch.as_tensor(candidate_mask, device=logits.device, dtype=torch.bool)
    if candidates.shape != logits.shape:
        raise ValueError("RC4 candidate_mask must match student_logits")
    hard = _sample_mask(hard_mask, rows, logits.device, "hard_mask")
    partial = _sample_mask(partial_mask, rows, logits.device, "partial_mask")
    negative = _sample_mask(negative_mask, rows, logits.device, "negative_mask")
    selected_set_rows = partial | negative
    if bool(selected_set_rows.any()) and bool((~candidates[selected_set_rows].any(dim=-1)).any()):
        raise ValueError("RC4 selected P/N row must have a non-empty allowed set")
    weight = _sample_weights(weights, rows, logits.device, logits.dtype)
    pseudo = torch.as_tensor(pseudo, device=logits.device, dtype=torch.long).reshape(-1)
    if pseudo.numel() != rows:
        raise ValueError("RC4 pseudo must contain one label per student row")
    zero = logits.sum() * 0.0
    hard_loss = (
        (F.cross_entropy(logits[hard], pseudo[hard], reduction="none") * weight[hard]).sum()
        / float(denominator)
        if bool(hard.any())
        else zero
    )
    work_logits = logits.float()
    all_lse = torch.logsumexp(work_logits, dim=-1)
    allowed_lse = torch.logsumexp(work_logits.masked_fill(~candidates, float("-inf")), dim=-1)
    per_set_mass = all_lse - allowed_lse
    candidate_float = candidates.to(work_logits.dtype)
    restricted = teacher.float() * candidate_float
    restricted = restricted / restricted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_restricted = restricted.clamp_min(1e-12).log()
    log_student_conditional = torch.where(
        candidates, work_logits - allowed_lse.unsqueeze(1), torch.zeros_like(work_logits)
    )
    per_partial_conditional = torch.where(
        candidates, restricted * (log_restricted - log_student_conditional), torch.zeros_like(work_logits)
    ).sum(dim=-1)
    partial_set = (
        (per_set_mass[partial] * weight[partial].float()).sum() / float(denominator)
        if bool(partial.any())
        else zero
    )
    partial_conditional = (
        (per_partial_conditional[partial] * weight[partial].float()).sum() / float(denominator)
        if bool(partial.any())
        else zero
    )
    negative_set = (
        (per_set_mass[negative] * weight[negative].float()).sum() / float(denominator)
        if bool(negative.any())
        else zero
    )
    partial_loss = (partial_set if bool(enable_partial_set) else zero) + (
        partial_conditional if bool(enable_partial_conditional) else zero
    )
    negative_loss = negative_set if bool(enable_negative_set) else zero
    return {
        "hard": hard_loss,
        "partial_set": partial_set,
        "partial_conditional": partial_conditional,
        "partial_positive": partial_set,
        "partial_negative": partial_conditional,
        "partial": partial_loss,
        "negative_set": negative_set,
        "negative": negative_loss,
        "total": hard_loss + partial_loss + negative_loss,
    }


def calibrate_sat_anchor_thresholds(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    epsilon: float = 0.02,
) -> SATAnchorThresholds:
    """Choose class-complete ``V_cal`` thresholds under a selected-error bound."""

    probability = _normalize_probability(probabilities)
    if probability.ndim != 2 or int(probability.shape[1]) != int(num_classes):
        raise ValueError("probabilities must have shape [samples, num_classes]")
    truth = torch.as_tensor(labels, device=probability.device).reshape(-1).long()
    if truth.numel() != probability.shape[0]:
        raise ValueError("labels must contain one value per probability row")
    risk_limit = float(epsilon)
    if not math.isfinite(risk_limit) or not 0.0 <= risk_limit <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")

    confidence, predicted = probability.max(dim=-1)
    top2 = probability.topk(min(2, probability.shape[1]), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] == 2 else 0.0)
    confidence_thresholds = probability.new_full((int(num_classes),), float("inf"))
    margin_thresholds = probability.new_full((int(num_classes),), float("inf"))

    for class_id in range(int(num_classes)):
        class_rows = (predicted == class_id).nonzero(as_tuple=False).reshape(-1)
        if class_rows.numel() == 0:
            continue
        class_confidence = confidence[class_rows]
        class_margin = margin[class_rows]
        order = torch.argsort(class_confidence, descending=True, stable=True)
        ordered_rows = class_rows[order]
        ordered_confidence = class_confidence[order]
        ordered_margin = class_margin[order]
        errors = (predicted[ordered_rows] != truth[ordered_rows]).to(torch.float32)
        counts = torch.arange(
            1,
            int(class_rows.numel()) + 1,
            device=probability.device,
            dtype=torch.float32,
        )
        safe = errors.cumsum(dim=0) / counts <= risk_limit + 1e-12
        confidence_boundary = torch.ones_like(safe)
        if confidence_boundary.numel() > 1:
            confidence_boundary[:-1] = (
                ordered_confidence[:-1] > ordered_confidence[1:]
            )
        safe_prefixes = (safe & confidence_boundary).nonzero(as_tuple=False).reshape(-1)
        if safe_prefixes.numel() > 0:
            prefix_end = int(safe_prefixes[-1].item())
            confidence_thresholds[class_id] = ordered_confidence[prefix_end]
            margin_thresholds[class_id] = ordered_margin[: prefix_end + 1].min()
    return SATAnchorThresholds(
        confidence=confidence_thresholds,
        margin=margin_thresholds,
    )


def route_sat_anchor_trusted(
    anchor_probabilities: torch.Tensor,
    ema_probabilities: torch.Tensor,
    *,
    confidence_thresholds: torch.Tensor,
    margin_thresholds: torch.Tensor,
    receivers: torch.Tensor | None = None,
    beta: float = 0.5,
    hard_max_fraction: float = 0.25,
    fill_to_fraction: float = 0.0,
    class_balanced_cap: bool = True,
    receiver_balanced_cap: bool = False,
) -> SATAnchorRoute:
    """Route only teacher-agreed, calibrated samples unless a control enables fill."""

    anchor, ema = _probability_heads([anchor_probabilities, ema_probabilities])
    if anchor.shape != ema.shape or anchor.ndim != 2:
        raise ValueError("anchor and EMA probabilities must have matching [batch, classes] shape")
    batch_size, num_classes = anchor.shape
    confidence_thresholds = torch.as_tensor(
        confidence_thresholds, device=anchor.device, dtype=anchor.dtype
    ).reshape(-1)
    margin_thresholds = torch.as_tensor(
        margin_thresholds, device=anchor.device, dtype=anchor.dtype
    ).reshape(-1)
    if confidence_thresholds.numel() != num_classes or margin_thresholds.numel() != num_classes:
        raise ValueError("threshold tensors must contain one value per class")
    hard_fraction = float(hard_max_fraction)
    fill_fraction = float(fill_to_fraction)
    if not (math.isfinite(hard_fraction) and math.isfinite(fill_fraction)):
        raise ValueError("routing fractions must be finite")
    if not 0.0 <= hard_fraction <= 1.0 or not 0.0 <= fill_fraction <= 1.0:
        raise ValueError("routing fractions must be in [0, 1]")

    fused = fuse_anchor_ema_probabilities(anchor, ema, beta=beta)
    confidence, pseudo = fused.max(dim=-1)
    top2 = fused.topk(min(2, num_classes), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] == 2 else 0.0)
    agreement = anchor.argmax(dim=-1) == ema.argmax(dim=-1)
    strict = agreement & (confidence >= confidence_thresholds[pseudo]) & (
        margin >= margin_thresholds[pseudo]
    )

    score = confidence.detach().cpu().tolist()
    def ranked(mask: torch.Tensor) -> list[int]:
        indices = mask.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()
        return sorted(indices, key=lambda index: (-float(score[index]), int(index)))

    limit = int(math.floor(int(batch_size) * hard_fraction + 1e-12))
    trusted = torch.zeros(int(batch_size), dtype=torch.bool, device=anchor.device)
    class_cap = torch.zeros_like(trusted)
    receiver_cap = torch.zeros_like(trusted)
    selected: list[int] = []
    active_classes = sorted(int(value) for value in torch.unique(pseudo[strict]).cpu().tolist())
    if limit > 0 and active_classes:
        class_limit = (
            int(math.ceil(limit / float(len(active_classes))))
            if class_balanced_cap
            else limit
        )
        receiver_values = None
        if receivers is not None:
            receiver_values = torch.as_tensor(receivers, device=anchor.device).reshape(-1)
            if receiver_values.numel() != batch_size:
                raise ValueError("receivers must contain one value per sample")
        for class_id in active_classes:
            class_mask = strict & (pseudo == class_id)
            class_selected: list[int] = []
            if receiver_balanced_cap and receiver_values is not None:
                active_receivers = sorted(
                    int(value) for value in torch.unique(receiver_values[class_mask]).cpu().tolist()
                )
                cell_limit = int(math.ceil(class_limit / float(max(1, len(active_receivers)))))
                for receiver_id in active_receivers:
                    cell_mask = class_mask & (receiver_values == receiver_id)
                    class_selected.extend(ranked(cell_mask)[:cell_limit])
                class_selected = sorted(
                    class_selected, key=lambda index: (-float(score[index]), int(index))
                )[:class_limit]
            else:
                class_selected = ranked(class_mask)[:class_limit]
            selected.extend(class_selected)
        selected = sorted(selected, key=lambda index: (-float(score[index]), int(index)))[:limit]
    if selected:
        selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=anchor.device)
        trusted[selected_tensor] = True
        class_cap[selected_tensor] = True
        receiver_cap[selected_tensor] = True

    filled = torch.zeros_like(trusted)
    fill_limit = int(math.floor(int(batch_size) * fill_fraction + 1e-12))
    remaining = max(0, fill_limit - int(trusted.sum().item()))
    if remaining:
        fill_candidates = ~trusted
        fill_indices = ranked(fill_candidates)[:remaining]
        if fill_indices:
            fill_tensor = torch.as_tensor(fill_indices, dtype=torch.long, device=anchor.device)
            trusted[fill_tensor] = True
            filled[fill_tensor] = True
    return SATAnchorRoute(
        pseudo=pseudo,
        confidence=confidence,
        margin=margin,
        agreement=agreement,
        strict=strict,
        trusted=trusted,
        filled=filled,
        no_identity=~trusted,
        class_cap=class_cap,
        receiver_cap=receiver_cap,
    )


def sat_anchor_clean_kl(
    student_logits: torch.Tensor,
    frozen_teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """One-way clean-view KL whose frozen-teacher argument never receives gradients."""

    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature must be finite and positive")
    teacher_probability = F.softmax(frozen_teacher_logits.detach().float() / scale, dim=-1)
    student_log_probability = F.log_softmax(student_logits.float() / scale, dim=-1)
    return F.kl_div(student_log_probability, teacher_probability, reduction="batchmean") * (scale**2)


def trusted_satellite_cross_entropy(
    satellite_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    trusted_mask: torch.Tensor,
    *,
    full_unlabeled_batch_size: int,
) -> torch.Tensor:
    """Satellite CE normalized by the full U batch, never the selected count."""

    mask = torch.as_tensor(trusted_mask, device=satellite_logits.device, dtype=torch.bool).reshape(-1)
    labels = torch.as_tensor(pseudo_labels, device=satellite_logits.device, dtype=torch.long).reshape(-1)
    if satellite_logits.shape[0] != mask.numel() or labels.numel() != mask.numel():
        raise ValueError("logits, pseudo labels and trusted mask must share a batch dimension")
    denominator = int(full_unlabeled_batch_size)
    if denominator <= 0:
        raise ValueError("full_unlabeled_batch_size must be positive")
    if not bool(mask.any().item()):
        return satellite_logits.sum() * 0.0
    return F.cross_entropy(satellite_logits[mask], labels[mask], reduction="sum") / float(denominator)


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
    """Validate the bounded contribution weight or explicit disabled sentinel."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "unlabeled_weight must be 0 or in [0.05, 0.10]"
        ) from exc
    if not torch.isfinite(torch.tensor(result)).item() or not (
        result == 0.0
        or _MIN_UNLABELED_PROTOTYPE_WEIGHT
        <= result
        <= _MAX_UNLABELED_PROTOTYPE_WEIGHT
    ):
        raise ValueError("unlabeled_weight must be 0 or in [0.05, 0.10]")
    return result


def _prototype_momentum(value: object) -> float:
    """Validate the EMA momentum independently of U_s contribution weight."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("momentum must be finite and in [0, 1)") from exc
    if not torch.isfinite(torch.tensor(result)).item() or not 0.0 <= result < 1.0:
        raise ValueError("momentum must be finite and in [0, 1)")
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

    def class_probabilities(
        self,
        features: torch.Tensor,
        *,
        num_classes: int,
        temperature: float = 0.10,
    ) -> torch.Tensor:
        """Score ``z_id`` against registered classification prototypes.

        Classes without a prototype receive exactly zero probability.  When
        the bank is empty, the only non-fabricated fallback is a uniform
        distribution over the declared classification classes.
        """

        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes < 1
        ):
            raise ValueError("num_classes must be a positive integer")
        try:
            temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be finite and positive") from exc
        if not torch.isfinite(torch.tensor(temperature)).item() or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

        value = features if torch.is_tensor(features) else torch.as_tensor(features)
        if value.ndim != 2:
            raise ValueError("features must have shape [N, D]")
        if value.is_complex():
            raise TypeError("features must be real-valued")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        if self._feature_dim is not None and int(value.shape[1]) != self._feature_dim:
            raise ValueError("feature dimension does not match the prototype bank")
        if not self._prototypes:
            return value.new_full(
                (int(value.shape[0]), num_classes),
                1.0 / float(num_classes),
            )

        present = sorted(int(class_id) for class_id in self._prototypes)
        if present[0] < 0 or present[-1] >= num_classes:
            raise ValueError("prototype class id falls outside num_classes")
        normalized = F.normalize(value.float(), dim=-1, eps=_PROBABILITY_EPS)
        prototype_matrix = torch.stack(
            [
                self._prototypes[class_id].to(
                    device=normalized.device,
                    dtype=normalized.dtype,
                )
                for class_id in present
            ],
            dim=0,
        )
        prototype_matrix = F.normalize(
            prototype_matrix,
            dim=-1,
            eps=_PROBABILITY_EPS,
        )
        present_probability = F.softmax(
            (normalized @ prototype_matrix.transpose(0, 1)) / temperature,
            dim=-1,
        )
        probability = normalized.new_zeros((normalized.shape[0], num_classes))
        probability[:, present] = present_probability
        return probability.to(dtype=value.dtype)

    def _observe_rows(
        self,
        features: torch.Tensor,
        labels: object,
        domains: object,
        accepted_mask: object,
        *,
        momentum: float,
        contribution: float,
        allow_new_classes: bool,
    ) -> None:
        """Apply one validated EMA update without mixing control semantics."""

        momentum = _prototype_momentum(momentum)
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
        label_tensor = _observation_vector(labels, sample_count, value.device, "labels")
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
        accepted = _sample_mask(
            accepted_mask, sample_count, value.device, "accepted_mask"
        )
        if self._frozen or not bool(accepted.any().item()):
            return

        detached = value.detach().float()
        label_values = label_tensor.detach().cpu().tolist()
        accepted_indices = accepted.detach().cpu().tolist()
        accepted_by_class: dict[int, list[int]] = {}
        for index, (label, is_accepted) in enumerate(
            zip(label_values, accepted_indices)
        ):
            if not is_accepted:
                continue
            class_id = int(label)
            if class_id < 0:
                raise ValueError("classification prototype labels must be non-negative")
            if not allow_new_classes and class_id not in self._prototypes:
                continue
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
                alpha = (1.0 - momentum) * float(contribution)
                self._prototypes[class_id] = (1.0 - alpha) * old + alpha * batch_mean
            self._counts[class_id] = self._counts.get(class_id, 0.0) + float(len(indices))

    def observe_labeled(
        self,
        features: torch.Tensor,
        labels: object,
        domains: object,
        *,
        momentum: float,
    ) -> None:
        """Initialize/update prototypes from legal L_s labels with full EMA mass."""

        value = features if torch.is_tensor(features) else torch.as_tensor(features)
        sample_count = int(value.shape[0]) if value.ndim >= 1 else 0
        self._observe_rows(
            value,
            labels,
            domains,
            torch.ones(sample_count, dtype=torch.bool, device=value.device),
            momentum=momentum,
            contribution=1.0,
            allow_new_classes=True,
        )

    def observe(
        self,
        features: torch.Tensor,
        pseudo: object,
        domains: object,
        high_mask: object,
        stable_mask: object,
        unlabeled_weight: float | None = None,
        *,
        momentum: float = 0.95,
    ) -> None:
        """Update L_s-seeded classes using only stable high-confidence U_s rows."""

        update_weight = (
            self._unlabeled_weight
            if unlabeled_weight is None
            else _prototype_weight(unlabeled_weight)
        )
        if update_weight == 0.0:
            return
        value = features if torch.is_tensor(features) else torch.as_tensor(features)
        sample_count = int(value.shape[0]) if value.ndim >= 1 else 0
        high = _sample_mask(high_mask, sample_count, value.device, "high_mask")
        stable = _sample_mask(stable_mask, sample_count, value.device, "stable_mask")
        self._observe_rows(
            value,
            pseudo,
            domains,
            high & stable,
            momentum=momentum,
            contribution=update_weight,
            allow_new_classes=False,
        )

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
        self.register_buffer(
            "_local_teacher_frozen_state",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

        nn.init.normal_(self.domain_delta_left, mean=0.0, std=0.02)
        nn.init.normal_(self.domain_delta_right, mean=0.0, std=0.02)

    @property
    def local_teacher_frozen(self) -> bool:
        return bool(self._local_teacher_frozen_state.item())

    def _local_teacher_parameters(self):
        yield from self.shared_projection.parameters()
        yield from self.shared_classifier.parameters()
        yield self.domain_delta_left
        yield self.domain_delta_right

    def freeze_local_teacher(self) -> None:
        """Permanently freeze the source-domain local teacher for S3C."""

        self._local_teacher_frozen_state.fill_(True)
        for parameter in self._local_teacher_parameters():
            parameter.requires_grad_(False)
        self.shared_projection.eval()
        self.shared_classifier.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.local_teacher_frozen:
            self.shared_projection.eval()
            self.shared_classifier.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.local_teacher_frozen:
            self.freeze_local_teacher()
        return result

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
        # Keep probabilities in float32 under AMP.  Casting them back to
        # float16 can underflow non-target classes to exact zero; the later
        # log-probability supervision then has a finite forward value for the
        # target row but produces 0/0 NaN gradients for zero-probability rows.
        return F.softmax(logits.float(), dim=-1)

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

    def sat_anchor_pair_loss(self, clean_z_id: torch.Tensor, satellite_z_id: torch.Tensor) -> torch.Tensor:
        """Symmetric clean-satellite SimSiam objective used by SAT-Anchor-SSL."""

        return self.self_supervised_loss(clean_z_id, satellite_z_id)

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
