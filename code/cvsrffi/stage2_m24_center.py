"""Separate support, decision, and covariance centers for M2.4."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


_EPS = 1.0e-12


class M24CenterError(ValueError):
    pass


def _weighted(local: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / np.sum(weights)
    return np.sum(local * normalized[:, None], axis=0)


def _unit(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, _EPS)


def estimate_centers(
    rows: Any,
    labels: Any,
    classes: Sequence[str],
    *,
    center_weights: Any | None = None,
    covariance_weights: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels).astype(str)
    registry = tuple(str(item) for item in classes)
    if features.ndim != 2 or len(features) != len(target) or not np.isfinite(features).all():
        raise M24CenterError("support rows/labels are invalid")
    center_w = np.ones(len(features)) if center_weights is None else np.asarray(center_weights, dtype=np.float64)
    covariance_w = np.ones(len(features)) if covariance_weights is None else np.asarray(covariance_weights, dtype=np.float64)
    if center_w.shape != (len(features),) or covariance_w.shape != (len(features),):
        raise M24CenterError("center weights have invalid shape")
    support, decision, covariance = [], [], []
    for item in registry:
        mask = target == item
        if not np.any(mask):
            raise M24CenterError("class has no support")
        local = features[mask]
        support.append(np.mean(local, axis=0))
        decision.append(_weighted(local, center_w[mask]))
        covariance.append(_weighted(local, covariance_w[mask]))
    return np.stack(support), _unit(np.stack(decision)), np.stack(covariance)


__all__ = ["M24CenterError", "estimate_centers"]
