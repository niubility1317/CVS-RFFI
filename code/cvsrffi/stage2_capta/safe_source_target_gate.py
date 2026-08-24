"""Support-only safe source/target path selection for CAPTA-P0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


PREREGISTERED_SOURCE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


class CaptaGateError(ValueError):
    """Raised when support-only gate evidence is malformed."""


@dataclass(frozen=True)
class SourceTargetGateResult:
    source_weight: float
    audit: dict[str, Any]


def _balanced_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predictions = np.argmax(scores, axis=1)
    class_accuracy = np.asarray(
        [float(np.mean(predictions[labels == index] == index)) for index in np.unique(labels)],
        dtype=np.float64,
    )
    return float(np.mean(class_accuracy)), float(np.min(class_accuracy))


def select_source_weight(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    support_class_indices: np.ndarray,
    *,
    candidate_weights: Sequence[float] = PREREGISTERED_SOURCE_WEIGHTS,
) -> SourceTargetGateResult:
    """Choose a frozen global source weight from support-only scores."""

    source = np.asarray(source_scores, dtype=np.float32)
    target = np.asarray(target_scores, dtype=np.float32)
    labels = np.asarray(support_class_indices)
    weights = tuple(float(value) for value in candidate_weights)
    if (
        source.ndim != 2
        or source.shape != target.shape
        or min(source.shape) < 1
        or labels.shape != (len(source),)
        or labels.dtype.kind not in "iu"
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or bool(np.any(labels < 0))
        or bool(np.any(labels >= source.shape[1]))
        or not np.array_equal(np.unique(labels), np.arange(source.shape[1]))
    ):
        raise CaptaGateError("support gate scores or labels are invalid")
    if weights != PREREGISTERED_SOURCE_WEIGHTS:
        raise CaptaGateError("source gate weight grid is not preregistered")

    candidates: list[dict[str, float]] = []
    best_objective: tuple[float, float, float] | None = None
    selected = 1.0
    for source_weight in weights:
        mixed = source_weight * source + (1.0 - source_weight) * target
        macro, floor = _balanced_metrics(mixed, labels.astype(np.int64))
        objective = (macro, floor, source_weight)
        candidates.append(
            {
                "source_weight": source_weight,
                "support_macro_accuracy": macro,
                "support_floor_accuracy": floor,
            }
        )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            selected = source_weight
    return SourceTargetGateResult(
        source_weight=selected,
        audit={
            "schema": "cvs.phase2.capta_p0.safe_source_target_gate.v1",
            "support_only": True,
            "query_rows_used": 0,
            "candidate_weights": list(weights),
            "tie_break": "prefer_higher_source_weight",
            "selected_source_weight": selected,
            "candidates": candidates,
        },
    )
