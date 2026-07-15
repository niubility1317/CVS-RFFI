"""Per-sample 1->3->5 receive-view gating for deployable rx_light5.

The gate consumes only the current sample's class scores.  Labels are accepted
only by the offline calibration helper, which is restricted by protocol to a
source validation split or registered support.  No old/new role, class quota,
query ordering, or cross-query graph is represented in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

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
    base_stop_min_score: float = -1.0e9
    shift3_stop_min_score: float = -1.0e9
    fusion_std_penalty: float = 0.0


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


def _top1_score(scores: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(scores, dtype=np.float32), axis=1).astype(np.float32)


def _fuse_view_scores(view_scores: np.ndarray, *, std_penalty: float) -> np.ndarray:
    """Fuse views with a lower-confidence bound on class-score stability."""

    values = np.asarray(view_scores, dtype=np.float32)
    penalty = float(std_penalty)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("view-score fusion requires finite [N,V,C] scores")
    if penalty < 0.0 or not np.isfinite(penalty):
        raise ValueError("view-score standard-deviation penalty must be nonnegative")
    return (values.mean(axis=1) - penalty * values.std(axis=1)).astype(np.float32)


def apply_adaptive_rxlight_tta(
    view_scores: np.ndarray,
    thresholds: AdaptiveTTAThresholds,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """Apply a strictly per-sample sequential 1/3/5-view decision."""

    scores = _validate_scores(view_scores)
    base = scores[:, 0]
    shift3 = _fuse_view_scores(
        scores[:, :3], std_penalty=thresholds.fusion_std_penalty
    )
    full5 = _fuse_view_scores(
        scores, std_penalty=thresholds.fusion_std_penalty
    )
    base_margin = _top2_margin(base)
    base_top1_score = _top1_score(base)
    shift_margin = _top2_margin(shift3)
    shift_top1_score = _top1_score(shift3)
    shift_predictions = np.argmax(scores[:, :3], axis=2)
    shift_consensus = np.argmax(shift3, axis=1)
    shift_disagreement = np.mean(
        shift_predictions != shift_consensus[:, None], axis=1
    ).astype(np.float32)

    budgets = np.full(scores.shape[0], 5, dtype=np.int64)
    stop_at_one = (
        (base_margin >= float(thresholds.base_stop_margin))
        & (base_top1_score >= float(thresholds.base_stop_min_score))
    )
    budgets[stop_at_one] = 1
    stop_at_three = (
        ~stop_at_one
        & (shift_margin >= float(thresholds.shift3_stop_margin))
        & (shift_top1_score >= float(thresholds.shift3_stop_min_score))
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
        "base_top1_score": base_top1_score,
        "shift3_margin": shift_margin,
        "shift3_top1_score": shift_top1_score,
        "shift3_disagreement": shift_disagreement,
        "mean_backbone_forwards": float(np.mean(budgets)),
        "p95_backbone_forwards": float(np.percentile(budgets, 95)),
        "trigger_rates": trigger_rates,
    }


def apply_adaptive_rxlight_tta_lazy(
    base_scores: np.ndarray,
    shift_score_provider: Callable[[np.ndarray], np.ndarray],
    cfo_score_provider: Callable[[np.ndarray], np.ndarray],
    thresholds: AdaptiveTTAThresholds,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """Execute 1->3->5 gating while requesting extra views only when needed.

    Providers receive the row indices that failed the preceding confidence gate
    and must return exactly two additional score views shaped ``[M,2,C]``.
    This is the deployable counterpart of :func:`apply_adaptive_rxlight_tta`;
    unlike the array-only helper, it never requests shift/CFO views for rows
    that have already exited.
    """

    base = np.asarray(base_scores, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] < 2:
        raise ValueError("base_scores must have shape [N,C] with C>=2")
    if not np.isfinite(base).all():
        raise FloatingPointError("base_scores contain non-finite values")
    row_count, class_count = base.shape
    selected = base.copy()
    budgets = np.ones(row_count, dtype=np.int64)
    base_margin = _top2_margin(base)
    base_top1_score = _top1_score(base)
    shift_margin = np.full(row_count, np.nan, dtype=np.float32)
    shift_top1_score = np.full(row_count, np.nan, dtype=np.float32)
    shift_disagreement = np.full(row_count, np.nan, dtype=np.float32)

    shift_indices = np.flatnonzero(
        (base_margin < float(thresholds.base_stop_margin))
        | (base_top1_score < float(thresholds.base_stop_min_score))
    ).astype(np.int64)
    shift_scores_by_row: dict[int, np.ndarray] = {}
    if len(shift_indices):
        shift_extra = np.asarray(shift_score_provider(shift_indices), dtype=np.float32)
        expected = (len(shift_indices), 2, class_count)
        if shift_extra.shape != expected or not np.isfinite(shift_extra).all():
            raise ValueError(
                f"shift score provider returned invalid shape/values: "
                f"{shift_extra.shape} != {expected}"
            )
        shift_all = np.concatenate(
            [base[shift_indices, None, :], shift_extra], axis=1
        )
        shift_mean = _fuse_view_scores(
            shift_all, std_penalty=thresholds.fusion_std_penalty
        )
        shift_predictions = np.argmax(shift_all, axis=2)
        shift_consensus = np.argmax(shift_mean, axis=1)
        local_margin = _top2_margin(shift_mean)
        local_top1_score = _top1_score(shift_mean)
        local_disagreement = np.mean(
            shift_predictions != shift_consensus[:, None], axis=1
        ).astype(np.float32)
        shift_margin[shift_indices] = local_margin
        shift_top1_score[shift_indices] = local_top1_score
        shift_disagreement[shift_indices] = local_disagreement
        selected[shift_indices] = shift_mean
        budgets[shift_indices] = 3
        needs_cfo = (
            (local_margin < float(thresholds.shift3_stop_margin))
            | (local_top1_score < float(thresholds.shift3_stop_min_score))
            | (local_disagreement > float(thresholds.shift3_max_disagreement))
        )
        cfo_indices = shift_indices[needs_cfo]
        for local_index, row_index in enumerate(shift_indices.tolist()):
            if needs_cfo[local_index]:
                shift_scores_by_row[int(row_index)] = shift_all[local_index]
    else:
        cfo_indices = np.empty(0, dtype=np.int64)

    if len(cfo_indices):
        cfo_extra = np.asarray(cfo_score_provider(cfo_indices), dtype=np.float32)
        expected = (len(cfo_indices), 2, class_count)
        if cfo_extra.shape != expected or not np.isfinite(cfo_extra).all():
            raise ValueError(
                f"CFO score provider returned invalid shape/values: "
                f"{cfo_extra.shape} != {expected}"
            )
        for local_index, row_index in enumerate(cfo_indices.tolist()):
            full_scores = np.concatenate(
                [shift_scores_by_row[int(row_index)], cfo_extra[local_index]],
                axis=0,
            )
            selected[int(row_index)] = _fuse_view_scores(
                full_scores[None, :, :],
                std_penalty=thresholds.fusion_std_penalty,
            )[0]
        budgets[cfo_indices] = 5

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
        "base_top1_score": base_top1_score,
        "shift3_margin": shift_margin,
        "shift3_top1_score": shift_top1_score,
        "shift3_disagreement": shift_disagreement,
        "mean_backbone_forwards": float(np.mean(budgets)),
        "p95_backbone_forwards": float(np.percentile(budgets, 95)),
        "trigger_rates": trigger_rates,
        "shift_rows_requested": int(len(shift_indices)),
        "cfo_rows_requested": int(len(cfo_indices)),
    }


def calibrate_adaptive_rxlight_tta(
    view_scores: np.ndarray,
    labels: np.ndarray,
    *,
    base_margin_grid: Sequence[float],
    shift3_margin_grid: Sequence[float],
    disagreement_grid: Sequence[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0),
    base_min_score_grid: Sequence[float] | None = None,
    shift3_min_score_grid: Sequence[float] | None = None,
    fusion_std_penalty_grid: Sequence[float] = (0.0, 0.1, 0.25),
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
    if base_min_score_grid is None:
        base_top1 = _top1_score(scores[:, 0])
        base_min_score_grid = (
            -1.0e9,
            float(np.quantile(base_top1, 0.10)),
            float(np.quantile(base_top1, 0.25)),
        )
    if shift3_min_score_grid is None:
        shift_top1 = _top1_score(scores[:, :3].mean(axis=1))
        shift3_min_score_grid = (
            -1.0e9,
            float(np.quantile(shift_top1, 0.10)),
        )
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, float, float, float], dict[str, object]] | None = None
    for fusion_penalty in fusion_std_penalty_grid:
        for base_min_score in base_min_score_grid:
            for shift_min_score in shift3_min_score_grid:
                for base_margin in base_margin_grid:
                    for shift_margin in shift3_margin_grid:
                        if float(base_margin) < float(shift_margin):
                            continue
                        for disagreement in disagreement_grid:
                            thresholds = AdaptiveTTAThresholds(
                                base_stop_margin=float(base_margin),
                                shift3_stop_margin=float(shift_margin),
                                shift3_max_disagreement=float(disagreement),
                                base_stop_min_score=float(base_min_score),
                                shift3_stop_min_score=float(shift_min_score),
                                fusion_std_penalty=float(fusion_penalty),
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
                                -accuracy,
                                float(row["mean_backbone_forwards"]),
                                float(row["p95_backbone_forwards"]),
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
        "selection_policy": "accuracy_first_then_minimum_forwards",
        "selected": best[1],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
