"""Support-LOO conservative old-class prior transport for M2.4."""

from __future__ import annotations

from typing import Any

import numpy as np


_EPS = 1.0e-12


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), _EPS)


def gated_old_prior(
    support_centers: Any,
    prior_centers: Any,
    *,
    k_shot: int,
    support_rows: Any,
    support_targets: Any,
    maximum_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    support = np.asarray(support_centers, dtype=np.float64)
    prior = np.asarray(prior_centers, dtype=np.float64)
    rows = np.asarray(support_rows, dtype=np.float64)
    targets = np.asarray(support_targets, dtype=np.int64)
    if support.shape != prior.shape or support.ndim != 2 or maximum_weight < 0.0 or maximum_weight > 0.1:
        raise ValueError("prior geometry or maximum_weight is invalid")
    if rows.ndim != 2 or rows.shape[1] != support.shape[1] or len(rows) != len(targets):
        raise ValueError("prior LOO support geometry is invalid")
    if k_shot <= 1:
        return np.array(support, copy=True), np.zeros(len(support)), {"mode": "forced_off_k1", "fallback_count": len(support)}
    alignment = np.sum(_unit(support) * _unit(prior), axis=1)
    helps = np.zeros(len(support), dtype=np.int64)
    harms = np.zeros(len(support), dtype=np.int64)
    for row_index, row in enumerate(rows):
        centres = []
        for class_index in range(len(support)):
            mask = targets == class_index
            if targets[row_index] == class_index:
                mask = np.array(mask, copy=True)
                mask[row_index] = False
            centres.append(np.mean(rows[mask], axis=0) if np.any(mask) else support[class_index])
        base = _unit(np.stack(centres))
        candidate = _unit((1.0 - maximum_weight) * base + maximum_weight * prior)
        base_prediction = int(np.argmax(base @ row))
        candidate_prediction = int(np.argmax(candidate @ row))
        target = int(targets[row_index])
        helps[target] += int(base_prediction != target and candidate_prediction == target)
        harms[target] += int(base_prediction == target and candidate_prediction != target)
    gate = np.where((helps >= 2) & (harms == 0) & (alignment > 0.0), maximum_weight * alignment, 0.0)
    fused = _unit((1.0 - gate[:, None]) * support + gate[:, None] * prior)
    return fused, gate, {
        "mode": "support_loo_no_harm",
        "fallback_count": int(np.sum(gate == 0.0)),
        "alignment": alignment.tolist(),
        "help_by_class": helps.tolist(),
        "harm_by_class": harms.tolist(),
    }


__all__ = ["gated_old_prior"]
