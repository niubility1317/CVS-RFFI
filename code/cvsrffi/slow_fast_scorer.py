"""Truth-last scoring and candidate aggregation for the slow/fast diag9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .slow_fast_diagnostics import build_shadow_response_surface
from .stage2_meta_adapter_matrix import _write_json_exclusive
from .stage2_meta_adapter_scorer import (
    MetaAdapterScoringError,
    PairedStage2BScore,
    _load_and_validate_prediction,
    _load_class_binding,
    _load_truth_json,
    _make_row_id,
    _metrics,
    _registered_ids,
    _validate_row,
    score_meta_adapter_pair,
)
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


def _aggregate_shadow_state(rows: Sequence[Mapping[str, Any]], state: str) -> dict[str, Any]:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for row in rows:
        payload = row["states"][state]
        metrics = payload.get("old_class_metrics", payload)
        for class_id, count in metrics["per_class_correct"].items():
            correct[class_id] = correct.get(class_id, 0) + int(count)
            total[class_id] = total.get(class_id, 0) + int(metrics["per_class_total"][class_id])
    if not correct or set(correct) != set(total):
        raise ValueError("shadow rows do not expose aligned per-class counts")
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


def summarize_shadow_candidate_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate arbitrary preregistered REG0 states without feeding truth back."""

    if len(rows) != 3:
        raise ValueError("one shadow candidate requires exactly three scene scores")
    candidate_ids = {str(row["candidate_id"]) for row in rows}
    scenarios = {str(row["scenario"]) for row in rows}
    if len(candidate_ids) != 1 or scenarios != {
        "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"
    }:
        raise ValueError("shadow score rows must cover the three frozen scenes")
    state_sets = [set(row["states"]) for row in rows]
    if not state_sets or any(value != state_sets[0] for value in state_sets[1:]):
        raise ValueError("shadow score rows must expose identical states")
    if "DA0_REG0" not in state_sets[0]:
        raise ValueError("shadow score rows lack DA0_REG0")
    baseline = _aggregate_shadow_state(rows, "DA0_REG0")
    states: dict[str, Any] = {"DA0_REG0": {**baseline, "mean_delta_pp": 0.0, "floor_delta_pp": 0.0, "max_class_drop_pp": 0.0, "verdict": "BASELINE"}}
    for state in sorted(state_sets[0] - {"DA0_REG0"}):
        aggregate = _aggregate_shadow_state(rows, state)
        mean_delta = (aggregate["mean_old_acc"] - baseline["mean_old_acc"]) * 100.0
        floor_delta = (aggregate["floor_accuracy"] - baseline["floor_accuracy"]) * 100.0
        class_deltas = {
            class_id: (aggregate["per_class_accuracy"][class_id] - accuracy) * 100.0
            for class_id, accuracy in baseline["per_class_accuracy"].items()
        }
        max_drop = min(class_deltas.values())
        promote = mean_delta >= 1.0 - 1.0e-9 and floor_delta >= 0.5 - 1.0e-9 and max_drop >= -5.0 - 1.0e-9
        states[state] = {
            **aggregate,
            "mean_delta_pp": float(mean_delta),
            "floor_delta_pp": float(floor_delta),
            "per_class_delta_pp": class_deltas,
            "max_class_drop_pp": float(max_drop),
            "verdict": "PROMOTE_TO_TARGET25" if promote else "SCIENTIFIC_FAILURE_NO_PROMOTION",
        }
    best = max(
        (state for state in states if state != "DA0_REG0"),
        key=lambda state: (states[state]["mean_old_acc"], states[state]["floor_accuracy"], state),
    )
    return {
        "candidate_id": next(iter(candidate_ids)),
        "states": states,
        "best_truth_last_shadow_state": best,
        "truth_last_selection_reused_for_adaptation": False,
    }


def _score_transition_diagnostics(
    *,
    registered_class_ids: Sequence[int],
    baseline_predictions: np.ndarray,
    adapted_predictions: np.ndarray,
    baseline_scores: np.ndarray,
    adapted_scores: np.ndarray,
    old_positions: np.ndarray,
    old_true_class_ids: np.ndarray,
    new_positions: np.ndarray,
) -> dict[str, Any]:
    """Compute truth-last decision and raw-cosine changes for one REG0 state."""

    registered = tuple(int(value) for value in registered_class_ids)
    if len(registered) < 2:
        raise MetaAdapterScoringError("transition diagnostics require at least two registered classes")
    if baseline_scores.shape != adapted_scores.shape or baseline_scores.ndim != 2:
        raise MetaAdapterScoringError("transition diagnostic score matrices must align")
    if baseline_scores.shape[1] != len(registered):
        raise MetaAdapterScoringError("transition diagnostic score columns do not match classes")
    if baseline_predictions.shape != adapted_predictions.shape or baseline_predictions.ndim != 1:
        raise MetaAdapterScoringError("transition diagnostic prediction vectors must align")
    old_positions = np.asarray(old_positions, dtype=np.int64)
    new_positions = np.asarray(new_positions, dtype=np.int64)
    true_ids = np.asarray(old_true_class_ids, dtype=np.int64)
    if old_positions.size != true_ids.size or old_positions.size == 0:
        raise MetaAdapterScoringError("transition diagnostics require aligned old-class truth")
    column_by_class = {class_id: index for index, class_id in enumerate(registered)}
    try:
        true_columns = np.asarray([column_by_class[int(value)] for value in true_ids], dtype=np.int64)
    except KeyError as error:
        raise MetaAdapterScoringError("old-class truth is outside registered score columns") from error

    baseline_old_pred = baseline_predictions[old_positions]
    adapted_old_pred = adapted_predictions[old_positions]
    baseline_correct = baseline_old_pred == true_ids
    adapted_correct = adapted_old_pred == true_ids
    baseline_old_scores = baseline_scores[old_positions]
    adapted_old_scores = adapted_scores[old_positions]
    row_indices = np.arange(old_positions.size, dtype=np.int64)
    baseline_true = baseline_old_scores[row_indices, true_columns]
    adapted_true = adapted_old_scores[row_indices, true_columns]

    baseline_competitors = baseline_old_scores.copy()
    adapted_competitors = adapted_old_scores.copy()
    baseline_competitors[row_indices, true_columns] = -np.inf
    adapted_competitors[row_indices, true_columns] = -np.inf
    baseline_true_margin = baseline_true - np.max(baseline_competitors, axis=1)
    adapted_true_margin = adapted_true - np.max(adapted_competitors, axis=1)
    true_margin_delta = adapted_true_margin - baseline_true_margin

    baseline_top2 = np.partition(baseline_old_scores, -2, axis=1)[:, -2:]
    adapted_top2 = np.partition(adapted_old_scores, -2, axis=1)[:, -2:]
    baseline_top_margin = np.max(baseline_top2, axis=1) - np.min(baseline_top2, axis=1)
    adapted_top_margin = np.max(adapted_top2, axis=1) - np.min(adapted_top2, axis=1)
    per_class_margin_delta = {
        str(class_id): float(np.mean(true_margin_delta[true_ids == class_id]))
        for class_id in sorted(set(int(value) for value in true_ids.tolist()))
    }
    if new_positions.size:
        new_intrusion = float(
            np.mean(
                np.max(adapted_scores[new_positions], axis=1)
                - np.max(baseline_scores[new_positions], axis=1)
            )
        )
        new_changes = int(
            np.count_nonzero(
                adapted_predictions[new_positions] != baseline_predictions[new_positions]
            )
        )
    else:
        new_intrusion = None
        new_changes = 0
    return {
        "old_query_count": int(old_positions.size),
        "new_query_count": int(new_positions.size),
        "old_decision_change_count": int(np.count_nonzero(adapted_old_pred != baseline_old_pred)),
        "old_positive_flip_count": int(np.count_nonzero(~baseline_correct & adapted_correct)),
        "old_negative_flip_count": int(np.count_nonzero(baseline_correct & ~adapted_correct)),
        "new_decision_change_count": new_changes,
        "old_true_class_raw_cosine_delta_mean": float(np.mean(adapted_true - baseline_true)),
        "old_top1_top2_margin_delta_mean": float(np.mean(adapted_top_margin - baseline_top_margin)),
        "old_score_l2_change_mean": float(
            np.mean(np.linalg.vector_norm(adapted_old_scores - baseline_old_scores, axis=1))
        ),
        "per_class_true_margin_delta_mean": per_class_margin_delta,
        "new_intrusion_delta_mean": new_intrusion,
    }


def _score_shadow_row(
    receipt_path: Path,
    truth_path: str | Path,
    *,
    class_binding_path: str | Path | None,
) -> dict[str, Any]:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetaAdapterScoringError(f"shadow receipt cannot be loaded: {receipt_path}") from exc
    if not isinstance(receipt, Mapping) or receipt.get("status") != "PREDICTIONS_COMPLETE":
        raise MetaAdapterScoringError("shadow receipt must close predictions before truth")
    if receipt.get("states_same_row") is not True:
        raise MetaAdapterScoringError("shadow states_same_row must be true")
    for key in ("query_truth_opened", "query_role_opened", "source_opened"):
        if receipt.get(key) is not False:
            raise MetaAdapterScoringError(f"shadow receipt {key} must be false")
    if int(receipt.get("query_state_update_count", -1)) != 0:
        raise MetaAdapterScoringError("shadow query_state_update_count must be zero")
    states = receipt.get("states")
    paths = receipt.get("prediction_paths")
    if not isinstance(states, list) or not states or states[0] != "DA0_REG0":
        raise MetaAdapterScoringError("shadow receipt states must start with DA0_REG0")
    if len(states) != len(set(states)) or not isinstance(paths, Mapping) or set(paths) != set(states):
        raise MetaAdapterScoringError("shadow receipt states and prediction paths mismatch")
    registered = _registered_ids(receipt.get("registered_class_ids"))
    row = _validate_row(receipt)
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    reference_ids: np.ndarray | None = None
    for state in states:
        path = Path(str(paths[state]))
        expected = receipt_path.parent / f"predictions_{state}.npz"
        if path.resolve() != expected.resolve():
            raise MetaAdapterScoringError(f"shadow receipt {state} path mismatch")
        query_ids, predicted, scores = _load_and_validate_prediction(path, registered)
        if reference_ids is None:
            reference_ids = query_ids
        elif not np.array_equal(reference_ids, query_ids):
            raise MetaAdapterScoringError("shadow predictions must share ordered query IDs")
        predictions[state] = (query_ids, predicted, scores)
    if reference_ids is None:
        raise MetaAdapterScoringError("shadow predictions are empty")
    class_binding = _load_class_binding(class_binding_path) if class_binding_path else None
    if class_binding is not None and set(class_binding.values()) != set(registered):
        raise MetaAdapterScoringError("class binding does not match registered classes")

    # This is intentionally the first access to the truth artifact.
    truth = _load_truth_json(
        truth_path,
        scenario=str(row["scenario"]),
        class_handle_to_id=class_binding,
    )
    opaque_ids = tuple(str(value) for value in reference_ids.tolist())
    if truth.all_query_ids != frozenset(opaque_ids):
        raise MetaAdapterScoringError("exact opaque-ID join failed for shadow predictions")
    old_positions = [index for index, value in enumerate(opaque_ids) if value in truth.old_class_by_query_id]
    position_array = np.asarray(old_positions, dtype=np.int64)
    new_position_array = np.asarray(
        [index for index, value in enumerate(opaque_ids) if value not in truth.old_class_by_query_id],
        dtype=np.int64,
    )
    old_ids = reference_ids[position_array]
    true_ids = np.asarray(
        [truth.old_class_by_query_id[opaque_ids[index]] for index in old_positions],
        dtype=np.int64,
    )
    if not np.isin(true_ids, np.asarray(registered, dtype=np.int64)).all():
        raise MetaAdapterScoringError("shadow truth contains an unregistered class")
    scored_states = {
        state: _metrics(state, old_ids, true_ids, predicted[position_array]).to_dict()
        for state, (_, predicted, _) in predictions.items()
    }
    baseline_pred = predictions["DA0_REG0"][1]
    baseline_scores = predictions["DA0_REG0"][2]
    transitions = {
        state: _score_transition_diagnostics(
            registered_class_ids=registered,
            baseline_predictions=baseline_pred,
            adapted_predictions=predicted,
            baseline_scores=baseline_scores,
            adapted_scores=scores,
            old_positions=position_array,
            old_true_class_ids=true_ids,
            new_positions=new_position_array,
        )
        for state, (_, predicted, scores) in predictions.items()
    }
    return {
        "schema": "cvs.stage2.slow_fast.shadow_row_score.v3",
        "status": "ANALYZED",
        "row_id": _make_row_id(row),
        "candidate_id": str(row["candidate_id"]),
        "bundle_id": str(row["bundle_id"]),
        "scenario": str(row["scenario"]),
        "states": scored_states,
        "decision_changes_vs_DA0": {
            state: int(np.count_nonzero(predicted != baseline_pred))
            for state, (_, predicted, _) in predictions.items()
        },
        "transition_diagnostics_vs_DA0": transitions,
        "truth_opened_after_all_predictions_validated": True,
        "truth_last_selection_reused_for_adaptation": False,
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
    first_receipt = json.loads((root / rows[0]["row_id"] / "receipt.json").read_text(encoding="utf-8"))
    if first_receipt.get("states") != ["DA0_REG0", "DA1_REG0"]:
        shadow_grouped: dict[str, list[dict[str, Any]]] = {}
        shadow_rows: list[dict[str, Any]] = []
        shadow_scored_rows: list[dict[str, Any]] = []
        shadow_receipts: list[dict[str, Any]] = []
        for row in rows:
            row_root = root / row["row_id"]
            receipt_payload = json.loads(
                (row_root / "receipt.json").read_text(encoding="utf-8")
            )
            scored = _score_shadow_row(
                row_root / "receipt.json",
                truth_by_scenario[row["config"]["scenario"]],
                class_binding_path=class_binding_path,
            )
            _write_json_exclusive(row_root / "score.json", scored)
            shadow_grouped.setdefault(scored["candidate_id"], []).append(scored)
            shadow_scored_rows.append(scored)
            shadow_receipts.append(receipt_payload)
            shadow_rows.append(
                {
                    "row_id": scored["row_id"],
                    "candidate_id": scored["candidate_id"],
                    "scenario": scored["scenario"],
                    "decision_changes_vs_DA0": scored["decision_changes_vs_DA0"],
                    "transition_diagnostics_vs_DA0": scored["transition_diagnostics_vs_DA0"],
                }
            )
        candidates = {
            candidate: summarize_shadow_candidate_scores(candidate_rows)
            for candidate, candidate_rows in sorted(shadow_grouped.items())
        }
        summary = {
            "schema": "cvs.stage2.slow_fast.shadow_diag9_score.v2",
            "status": "ANALYZED",
            "states": first_receipt["states"],
            "truth_opened_after_predictions_complete": True,
            "truth_last_selection_reused_for_adaptation": False,
            "row_count": len(shadow_rows),
            "rows": shadow_rows,
            "candidates": candidates,
            "response_surface": (
                build_shadow_response_surface(shadow_scored_rows, shadow_receipts)
                if all(
                    isinstance(receipt.get("shadow_support_diagnostics"), Mapping)
                    for receipt in shadow_receipts
                )
                else {
                    "status": "UNAVAILABLE",
                    "reason": "LEGACY_RECEIPTS_LACK_SHADOW_SUPPORT_DIAGNOSTICS",
                }
            ),
        }
        _write_json_exclusive(root / "diag9_score_summary.json", summary)
        return summary
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


__all__ = [
    "score_slow_fast_matrix",
    "summarize_candidate_scores",
    "summarize_shadow_candidate_scores",
]
