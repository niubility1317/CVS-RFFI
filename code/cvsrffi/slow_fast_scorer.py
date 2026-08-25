"""Truth-last scoring and candidate aggregation for the slow/fast diag9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .stage2_meta_adapter_matrix import _write_json_exclusive
from .stage2_meta_adapter_scorer import PairedStage2BScore, score_meta_adapter_pair
from .stage2_slow_fast_matrix import _validate_matrix


def _aggregate_state(scores: Sequence[PairedStage2BScore], state: str) -> dict[str, Any]:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for score in scores:
        value = score.da0 if state == "DA0_REG0" else score.da1
        for class_id, count in value.per_class_correct.items():
            correct[class_id] = correct.get(class_id, 0) + int(count)
            total[class_id] = total.get(class_id, 0) + int(value.per_class_total[class_id])
    if not correct or set(correct) != set(total):
        raise ValueError("candidate rows do not expose aligned per-class counts")
    accuracy = {class_id: correct[class_id] / total[class_id] for class_id in sorted(correct)}
    return {
        "mean_old_acc": float(sum(accuracy.values()) / len(accuracy)),
        "floor_accuracy": float(min(accuracy.values())),
        "per_class_accuracy": accuracy,
        "per_class_correct": correct,
        "per_class_total": total,
        "new_class_accuracy": None,
        "old_new_harmonic_mean": None,
    }


def summarize_candidate_scores(scores: Sequence[PairedStage2BScore]) -> dict[str, Any]:
    if len(scores) != 3:
        raise ValueError("one slow-fast candidate requires exactly three scene scores")
    candidate_ids = {score.candidate_id for score in scores}
    scenarios = {str(score.row["scenario"]) for score in scores}
    if len(candidate_ids) != 1 or scenarios != {
        "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"
    }:
        raise ValueError("candidate score rows must cover the three frozen scenes")
    da0 = _aggregate_state(scores, "DA0_REG0")
    da1 = _aggregate_state(scores, "DA1_REG0")
    mean_delta = (da1["mean_old_acc"] - da0["mean_old_acc"]) * 100.0
    floor_delta = (da1["floor_accuracy"] - da0["floor_accuracy"]) * 100.0
    class_deltas = {
        class_id: (da1["per_class_accuracy"][class_id] - accuracy) * 100.0
        for class_id, accuracy in da0["per_class_accuracy"].items()
    }
    max_class_drop = min(class_deltas.values())
    promote = mean_delta >= 1.0 - 1.0e-9 and floor_delta >= 0.5 - 1.0e-9 and max_class_drop >= -5.0 - 1.0e-9
    return {
        "candidate_id": next(iter(candidate_ids)),
        "DA0_REG0": da0,
        "DA1_REG0": da1,
        "mean_delta_pp": float(mean_delta),
        "floor_delta_pp": float(floor_delta),
        "per_class_delta_pp": class_deltas,
        "max_class_drop_pp": float(max_class_drop),
        "verdict": "PROMOTE_TO_TARGET25" if promote else "SCIENTIFIC_FAILURE_NO_PROMOTION",
    }


def score_slow_fast_matrix(
    matrix_config: Mapping[str, Any],
    prediction_root: str | Path,
    truth_by_scenario: Mapping[str, str],
    *,
    class_binding_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = _validate_matrix(matrix_config)
    required_scenes = {row["config"]["scenario"] for row in rows}
    if set(truth_by_scenario) != required_scenes:
        raise ValueError("truth map must exactly cover the three prediction scenes")
    root = Path(prediction_root)
    if not root.is_dir():
        raise FileNotFoundError(f"prediction root is missing: {root}")
    grouped: dict[str, list[PairedStage2BScore]] = {}
    row_summaries: list[dict[str, Any]] = []
    for row in rows:
        row_root = root / row["row_id"]
        score = score_meta_adapter_pair(
            row_root / "predictions_DA0_REG0.npz",
            row_root / "predictions_DA1_REG0.npz",
            truth_by_scenario[row["config"]["scenario"]],
            receipt_path=row_root / "receipt.json",
            class_binding_path=class_binding_path,
        )
        _write_json_exclusive(row_root / "score.json", score.to_dict())
        with np.load(row_root / "predictions_DA0_REG0.npz", allow_pickle=False) as da0:
            pred0 = np.asarray(da0["predicted_class_ids"])
        with np.load(row_root / "predictions_DA1_REG0.npz", allow_pickle=False) as da1:
            pred1 = np.asarray(da1["predicted_class_ids"])
        decision_changes = int(np.count_nonzero(pred0 != pred1))
        grouped.setdefault(score.candidate_id, []).append(score)
        row_summaries.append(
            {
                "row_id": row["row_id"],
                "candidate_id": score.candidate_id,
                "scenario": row["config"]["scenario"],
                "mean_delta_pp": score.mean_delta_pp,
                "floor_delta_pp": score.floor_delta_pp,
                "decision_changes": decision_changes,
            }
        )
    candidates = {
        candidate: summarize_candidate_scores(candidate_scores)
        for candidate, candidate_scores in sorted(grouped.items())
    }
    summary = {
        "schema": "cvs.stage2.slow_fast.diag9_score.v1",
        "status": "ANALYZED",
        "states": ["DA0_REG0", "DA1_REG0"],
        "truth_opened_after_predictions_complete": True,
        "row_count": len(row_summaries),
        "rows": row_summaries,
        "candidates": candidates,
    }
    _write_json_exclusive(root / "diag9_score_summary.json", summary)
    return summary


__all__ = ["score_slow_fast_matrix", "summarize_candidate_scores"]
