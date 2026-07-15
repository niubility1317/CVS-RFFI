"""Per-sample 1->3->5 receive-view gating for deployable rx_light5.

The gate consumes only the current sample's class scores.  Labels are accepted
only by the offline calibration helper, which is restricted by protocol to a
source validation split or registered support.  No old/new role, class quota,
query ordering, or cross-query graph is represented in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


RX_LIGHT5_ORDER = (
    "rx_base",
    "rx_shift_m2",
    "rx_shift_p2",
    "rx_cfo_m1e4",
    "rx_cfo_p1e4",
)


@dataclass(frozen=True)
class AdaptiveTTAThresholds:
    base_stop_margin: float
    shift3_stop_margin: float
    shift3_max_disagreement: float


def _validate_scores(view_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(view_scores, dtype=np.float32)
    if scores.ndim != 3 or scores.shape[1] != 5 or scores.shape[2] < 2:
        raise ValueError("view_scores must have shape [N,5,C] with C>=2")
    if not np.isfinite(scores).all():
        raise FloatingPointError("view_scores contain non-finite values")
    return scores


def _top2_margin(scores: np.ndarray) -> np.ndarray:
    top2 = np.partition(scores, kth=scores.shape[1] - 2, axis=1)[:, -2:]
    return np.max(top2, axis=1) - np.min(top2, axis=1)


def apply_adaptive_rxlight_tta(
    view_scores: np.ndarray,
    thresholds: AdaptiveTTAThresholds,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """Apply a strictly per-sample sequential 1/3/5-view decision."""

    scores = _validate_scores(view_scores)
    base = scores[:, 0]
    shift3 = scores[:, :3].mean(axis=1)
    full5 = scores.mean(axis=1)
    base_margin = _top2_margin(base)
    shift_margin = _top2_margin(shift3)
    shift_predictions = np.argmax(scores[:, :3], axis=2)
    shift_consensus = np.argmax(shift3, axis=1)
    shift_disagreement = np.mean(
        shift_predictions != shift_consensus[:, None], axis=1
    ).astype(np.float32)

    budgets = np.full(scores.shape[0], 5, dtype=np.int64)
    stop_at_one = base_margin >= float(thresholds.base_stop_margin)
    budgets[stop_at_one] = 1
    stop_at_three = (
        ~stop_at_one
        & (shift_margin >= float(thresholds.shift3_stop_margin))
        & (
            shift_disagreement
            <= float(thresholds.shift3_max_disagreement)
        )
    )
    budgets[stop_at_three] = 3
    selected = full5.copy()
    selected[stop_at_three] = shift3[stop_at_three]
    selected[stop_at_one] = base[stop_at_one]
    predictions = np.argmax(selected, axis=1).astype(np.int64)
    trigger_rates = {
        "view1_rate": float(np.mean(budgets == 1)),
        "view3_rate": float(np.mean(budgets == 3)),
        "view5_rate": float(np.mean(budgets == 5)),
    }
    return {
        "scores": selected,
        "predictions": predictions,
        "view_budgets": budgets,
        "base_margin": base_margin,
        "shift3_margin": shift_margin,
        "shift3_disagreement": shift_disagreement,
        "mean_backbone_forwards": float(np.mean(budgets)),
        "p95_backbone_forwards": float(np.percentile(budgets, 95)),
        "trigger_rates": trigger_rates,
    }


def calibrate_adaptive_rxlight_tta(
    view_scores: np.ndarray,
    labels: np.ndarray,
    *,
    base_margin_grid: Sequence[float],
    shift3_margin_grid: Sequence[float],
    disagreement_grid: Sequence[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0),
    max_accuracy_drop_pp: float = 1.0,
) -> dict[str, object]:
    """Select the cheapest source/support-calibrated gate within a 5-view loss cap."""

    scores = _validate_scores(view_scores)
    truth = np.asarray(labels, dtype=np.int64)
    if truth.shape != (scores.shape[0],):
        raise ValueError("labels must have shape [N]")
    if truth.size == 0 or truth.min() < 0 or truth.max() >= scores.shape[2]:
        raise ValueError("labels are empty or outside the class-score width")
    full5_predictions = np.argmax(scores.mean(axis=1), axis=1)
    full5_accuracy = float(np.mean(full5_predictions == truth))
    floor = full5_accuracy - float(max_accuracy_drop_pp) / 100.0
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, float, float, float], dict[str, object]] | None = None
    for base_margin in base_margin_grid:
        for shift_margin in shift3_margin_grid:
            if float(base_margin) < float(shift_margin):
                continue
            for disagreement in disagreement_grid:
                thresholds = AdaptiveTTAThresholds(
                    base_stop_margin=float(base_margin),
                    shift3_stop_margin=float(shift_margin),
                    shift3_max_disagreement=float(disagreement),
                )
                result = apply_adaptive_rxlight_tta(scores, thresholds)
                accuracy = float(np.mean(result["predictions"] == truth))
                row: dict[str, object] = {
                    "thresholds": thresholds,
                    "accuracy": accuracy,
                    "accuracy_drop_pp_vs_full5": float(
                        (full5_accuracy - accuracy) * 100.0
                    ),
                    "mean_backbone_forwards": result["mean_backbone_forwards"],
                    "p95_backbone_forwards": result["p95_backbone_forwards"],
                    "trigger_rates": result["trigger_rates"],
                    "passes_accuracy_cap": bool(accuracy + 1.0e-12 >= floor),
                }
                candidates.append(row)
                if not row["passes_accuracy_cap"]:
                    continue
                key = (
                    float(row["mean_backbone_forwards"]),
                    float(row["p95_backbone_forwards"]),
                    -accuracy,
                    float(base_margin) + float(shift_margin),
                )
                if best is None or key < best[0]:
                    best = (key, row)
    if best is None:
        raise ValueError("no adaptive TTA threshold passes the accuracy-drop cap")
    return {
        "calibration_scope_required": "source_validation_or_registered_support_only",
        "uses_query_labels": False,
        "uses_old_new_role": False,
        "uses_class_quota": False,
        "full5_accuracy": full5_accuracy,
        "max_accuracy_drop_pp": float(max_accuracy_drop_pp),
        "selected": best[1],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
