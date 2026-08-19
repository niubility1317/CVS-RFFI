"""ESS-safe support reliability for ERBT-IDR M2.4."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


_EPS = 1.0e-12


class M24QualityError(ValueError):
    pass


def effective_sample_size(weights: Any) -> float:
    value = np.asarray(weights, dtype=np.float64)
    if value.ndim != 1 or len(value) == 0 or np.any(value < 0.0) or not np.isfinite(value).all():
        raise M24QualityError("weights must be a finite nonnegative vector")
    total = float(np.sum(value))
    if total <= _EPS:
        raise M24QualityError("weights must have positive mass")
    normalized = value / total
    return float(1.0 / np.sum(np.square(normalized)))


def ess_safe_weights(raw_quality: Any, *, k_shot: int) -> tuple[np.ndarray, dict[str, float]]:
    quality = np.asarray(raw_quality, dtype=np.float64)
    if quality.ndim != 1 or len(quality) != int(k_shot) or k_shot <= 0:
        raise M24QualityError("quality length must equal positive k_shot")
    if np.any(quality <= 0.0) or not np.isfinite(quality).all():
        raise M24QualityError("quality must be finite and positive")
    target = min(float(k_shot), max(3.0, float(k_shot) / 2.0))
    uniform = np.full(k_shot, 1.0 / k_shot, dtype=np.float64)
    if k_shot == 1:
        return uniform, {"required_ess": 1.0, "raw_ess": 1.0, "final_ess": 1.0, "uniform_blend": 1.0}
    raw = quality / np.sum(quality)
    raw_ess = effective_sample_size(raw)
    if raw_ess >= target:
        blend = 0.0
        result = raw
    else:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            candidate = (1.0 - middle) * raw + middle * uniform
            if effective_sample_size(candidate) >= target:
                high = middle
            else:
                low = middle
        blend = high
        result = (1.0 - blend) * raw + blend * uniform
    return result, {
        "required_ess": target,
        "raw_ess": raw_ess,
        "final_ess": effective_sample_size(result),
        "uniform_blend": float(blend),
    }


def if_residual_reliability(rows: Any, labels: Any, classes: Sequence[str]) -> np.ndarray:
    features = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if features.ndim != 2 or len(features) != len(target) or not np.isfinite(features).all():
        raise M24QualityError("IF rows/labels are invalid")
    result = np.empty(len(features), dtype=np.float64)
    for item in registry:
        mask = target == item
        if not np.any(mask):
            raise M24QualityError("class has no support")
        local = features[mask]
        center = np.mean(local, axis=0)
        residual = np.sum(np.square(local - center[None, :]), axis=1)
        positive = residual[residual > _EPS]
        scale = max(float(np.median(positive)) if len(positive) else 1.0, 1.0e-9)
        result[mask] = 1.0 / (1.0 + residual / scale)
    return result


__all__ = ["M24QualityError", "effective_sample_size", "ess_safe_weights", "if_residual_reliability"]
