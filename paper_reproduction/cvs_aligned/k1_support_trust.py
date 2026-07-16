"""Support-only trust-region utilities for the lightweight JG-R8 adapter.

The module never accepts query rows.  A caller must materialize every
pre-registered support validation view for each candidate delta scale before
calling :func:`select_largest_safe_support_scale`.  This keeps the selection
rule independent of target-query truth, roles, quotas, ordering, and batch
statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


EPS = 1.0e-8
DEFAULT_ALPHA_GRID = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def _normalize(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), EPS)


def _validate_support_feature_groups(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("support features must have shape [view_group,class,dim]")
    if values.shape[0] < 3 or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("support features require at least 3 groups, 2 classes, 2 dims")
    if not np.isfinite(values).all():
        raise FloatingPointError("support features contain non-finite values")
    return values


def leave_one_group_margins(features: np.ndarray) -> np.ndarray:
    """Return one class margin for every support validation group and class.

    For group ``g``, each class prototype is formed from every other group.
    The returned margin is the true-class cosine score minus the largest
    other-class score.  Every class follows exactly the same rule.
    """

    values = _normalize(_validate_support_feature_groups(features))
    group_count, class_count, _ = values.shape
    prototype_sums = values.sum(axis=0, keepdims=True) - values
    prototypes = _normalize(prototype_sums / float(group_count - 1))
    scores = np.einsum("gcd,gkd->gck", values, prototypes, optimize=True)
    class_index = np.arange(class_count)
    true_scores = scores[:, class_index, class_index]
    negative_scores = scores.copy()
    negative_scores[:, class_index, class_index] = -np.inf
    return (true_scores - negative_scores.max(axis=-1)).astype(np.float32)


def mean_cosine_drift(reference: np.ndarray, candidate: np.ndarray) -> float:
    base = _normalize(_validate_support_feature_groups(reference))
    adapted = _normalize(_validate_support_feature_groups(candidate))
    if base.shape != adapted.shape:
        raise ValueError("reference and candidate support feature shapes differ")
    return float(np.mean(1.0 - np.sum(base * adapted, axis=-1)))


@dataclass(frozen=True)
class SupportTrustDecision:
    selected_alpha: float
    status: str
    alpha_grid: tuple[float, ...]
    rows: tuple[dict[str, Any], ...]
    mean_margin_tolerance: float
    worst_group_tolerance: float
    mean_cosine_drift_cap: float
    support_view_group_count: int
    registered_class_count: int
    query_rows_used: int = 0
    query_truth_used: bool = False
    role_labels_used: bool = False
    class_quota_used: bool = False
    dense_query_graph_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_largest_safe_support_scale(
    base_support_features: np.ndarray,
    support_features_by_alpha: Mapping[float, np.ndarray],
    *,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    mean_margin_tolerance: float = 0.0,
    worst_group_tolerance: float = 0.002,
    mean_cosine_drift_cap: float = 0.02,
) -> SupportTrustDecision:
    """Select the largest support-safe adapter scale, otherwise return zero.

    ``support_features_by_alpha`` must contain exact model features evaluated
    after applying every requested scale.  Feature interpolation is deliberately
    not performed here because a scaled weight delta need not induce a linear
    feature path.
    """

    base = _validate_support_feature_groups(base_support_features)
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or tuple(sorted(set(grid))) != grid:
        raise ValueError("alpha_grid must be strictly increasing and unique")
    if grid[0] != 0.0 or any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("alpha_grid must start at 0 and remain inside [0,1]")
    if float(mean_margin_tolerance) < 0.0:
        raise ValueError("mean_margin_tolerance must be nonnegative")
    if float(worst_group_tolerance) < 0.0:
        raise ValueError("worst_group_tolerance must be nonnegative")
    if float(mean_cosine_drift_cap) < 0.0:
        raise ValueError("mean_cosine_drift_cap must be nonnegative")
    keyed = {float(key): value for key, value in support_features_by_alpha.items()}
    if set(keyed) != set(grid):
        raise ValueError("support_features_by_alpha must exactly match alpha_grid")
    zero = _validate_support_feature_groups(keyed[0.0])
    if zero.shape != base.shape or not np.allclose(zero, base, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("alpha=0 support features must reproduce the P4 identity baseline")

    base_margins = leave_one_group_margins(base)
    base_negative_count = int(np.count_nonzero(base_margins < 0.0))
    rows: list[dict[str, Any]] = []
    safe_nonzero: list[float] = []
    for alpha in grid:
        candidate = _validate_support_feature_groups(keyed[alpha])
        if candidate.shape != base.shape:
            raise ValueError(f"support feature shape drift at alpha={alpha}")
        margins = leave_one_group_margins(candidate)
        margin_delta = margins - base_margins
        mean_delta = float(margin_delta.mean())
        worst_delta = float(margin_delta.min())
        negative_count = int(np.count_nonzero(margins < 0.0))
        drift = mean_cosine_drift(base, candidate)
        mean_safe = mean_delta >= -float(mean_margin_tolerance) - 1.0e-8
        worst_safe = worst_delta >= -float(worst_group_tolerance) - 1.0e-8
        errors_safe = negative_count <= base_negative_count
        drift_safe = drift <= float(mean_cosine_drift_cap) + 1.0e-8
        safe = bool(mean_safe and worst_safe and errors_safe and drift_safe)
        if alpha > 0.0 and safe:
            safe_nonzero.append(float(alpha))
        rows.append(
            {
                "alpha": float(alpha),
                "mean_margin": float(margins.mean()),
                "worst_group_margin": float(margins.min()),
                "mean_margin_delta": mean_delta,
                "worst_group_margin_delta": worst_delta,
                "negative_margin_count": negative_count,
                "base_negative_margin_count": base_negative_count,
                "mean_cosine_drift": drift,
                "mean_margin_safe": bool(mean_safe),
                "worst_group_safe": bool(worst_safe),
                "negative_count_safe": bool(errors_safe),
                "drift_safe": bool(drift_safe),
                "safe": safe,
            }
        )
    selected = max(safe_nonzero) if safe_nonzero else 0.0
    return SupportTrustDecision(
        selected_alpha=float(selected),
        status=(
            "support_safe_nonzero_delta"
            if selected > 0.0
            else "fallback_p4_identity_alpha_zero"
        ),
        alpha_grid=grid,
        rows=tuple(rows),
        mean_margin_tolerance=float(mean_margin_tolerance),
        worst_group_tolerance=float(worst_group_tolerance),
        mean_cosine_drift_cap=float(mean_cosine_drift_cap),
        support_view_group_count=int(base.shape[0]),
        registered_class_count=int(base.shape[1]),
    )


def scale_lora_trainable_state(
    state: Mapping[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    """Scale a LoRA residual by multiplying only each ``lora_b`` tensor.

    Multiplying both low-rank factors would scale the composed residual by
    ``alpha**2``.  The identity-initialized trainer persists only ``lora_a``
    and ``lora_b`` trainable tensors, so any other key is rejected.
    """

    scale = float(alpha)
    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("alpha must be finite and inside [0,1]")
    if not state:
        raise ValueError("LoRA state cannot be empty")
    output: dict[str, torch.Tensor] = {}
    b_count = 0
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"LoRA state value is not a tensor: {name}")
        if name.endswith(".lora_a.weight"):
            output[str(name)] = value.detach().clone()
        elif name.endswith(".lora_b.weight"):
            output[str(name)] = (value.detach() * scale).to(dtype=value.dtype)
            b_count += 1
        else:
            raise ValueError(f"unexpected non-LoRA trainable state key: {name}")
    if b_count <= 0:
        raise ValueError("LoRA state contains no lora_b residual tensor")
    return output


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "SupportTrustDecision",
    "leave_one_group_margins",
    "mean_cosine_drift",
    "scale_lora_trainable_state",
    "select_largest_safe_support_scale",
]
