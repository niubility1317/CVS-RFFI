"""Normalized and capped optional center-uncertainty penalty."""

from __future__ import annotations

from typing import Any


import numpy as np


def normalized_capped_penalty(uncertainty: Any, precision: Any, *, cap: float = 0.2) -> np.ndarray:
    variance = np.asarray(uncertainty, dtype=np.float64)
    inverse = np.asarray(precision, dtype=np.float64)
    if variance.ndim != 2 or inverse.shape != (variance.shape[1], variance.shape[1]) or not 0.0 <= cap <= 1.0:
        raise ValueError("uncertainty/precision geometry is invalid")
    raw = 0.5 * np.sum(variance * np.diag(inverse)[None, :], axis=1)
    scale = max(float(np.median(raw[raw > 0.0])) if np.any(raw > 0.0) else 1.0, 1.0e-12)
    return np.clip(raw / scale, 0.0, cap)


__all__ = ["normalized_capped_penalty"]
