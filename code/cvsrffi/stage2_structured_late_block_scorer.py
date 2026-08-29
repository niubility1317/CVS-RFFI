"""Independent truth-side scorer for structured late-block Stage2-B rows.

The runner prediction artifact is completely validated before this module
opens the truth sidecar.  Scoring joins only opaque query IDs and never imports
or calls the predictor, checkpoint, adaptation model, or runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PREDICTION_SCHEMA = frozenset(
    {"query_ids", "predicted_class_ids", "scores"}
)
SCORE_SCHEMA = "cvs.stage2.structured_lateblock.score.v1"
STAGE2C_SCORE_SCHEMA = "cvs.stage2.structured_lateblock.score.stage2c.v1"


class StructuredLateBlockScoringError(ValueError):
    """Raised when prediction validation or truth-side scoring cannot close."""


def _load_and_validate_predictions(
    prediction_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(prediction_path)
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise StructuredLateBlockScoringError(
            f"prediction artifact is missing or invalid: {path}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = frozenset(str(name) for name in archive.files)
            if names != PREDICTION_SCHEMA:
                raise StructuredLateBlockScoringError(
                    "prediction artifact allowlist mismatch: "
                    f"missing={sorted(PREDICTION_SCHEMA - names)}, "
                    f"extra={sorted(names - PREDICTION_SCHEMA)}"
                )
            query_ids = np.asarray(archive["query_ids"]).copy()
            predicted = np.asarray(archive["predicted_class_ids"]).copy()
            scores = np.asarray(archive["scores"]).copy()
    except StructuredLateBlockScoringError:
        raise
    except (OSError, ValueError) as exc:
        raise StructuredLateBlockScoringError(
            f"prediction artifact cannot be loaded: {path}"
        ) from exc

    if (
        query_ids.ndim != 1
        or query_ids.shape[0] < 1
        or query_ids.dtype.kind not in {"U", "S"}
    ):
        raise StructuredLateBlockScoringError(
            "prediction query_ids must be a nonempty opaque string vector"
        )
    query_ids = query_ids.astype(str)
    if any(not value for value in query_ids.tolist()):
        raise StructuredLateBlockScoringError(
            "prediction query_ids contain an empty ID"
        )
    if len(set(query_ids.tolist())) != query_ids.shape[0]:
        raise StructuredLateBlockScoringError(
            "prediction query_ids must be unique"
        )
    if (
        predicted.ndim != 1
        or not np.issubdtype(predicted.dtype, np.integer)
    ):
        raise StructuredLateBlockScoringError(
            "prediction predicted_class_ids must be an integer vector"
        )
    predicted = np.ascontiguousarray(predicted, dtype=np.int64)
    if np.any(predicted < 0):
        raise StructuredLateBlockScoringError(
            "prediction class IDs must be nonnegative"
        )
    if (
        scores.ndim != 2
        or scores.shape[1] < 1
        or not np.issubdtype(scores.dtype, np.number)
    ):
        raise StructuredLateBlockScoringError(
            "prediction scores must be a numeric [queries,classes] matrix"
        )
    if not np.isfinite(scores).all():
        raise StructuredLateBlockScoringError(
            "prediction scores contain non-finite values"
        )
    row_count = int(query_ids.shape[0])
    if predicted.shape[0] != row_count or scores.shape[0] != row_count:
        raise StructuredLateBlockScoringError(
            "prediction array lengths do not align"
        )
    return query_ids, predicted, np.ascontiguousarray(scores, dtype=np.float32)


def _load_truth_json(truth_path: str | Path) -> Mapping[str, Any]:
    path = Path(truth_path)
    if not path.is_file():
        raise StructuredLateBlockScoringError(f"truth sidecar is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StructuredLateBlockScoringError(
            f"truth sidecar cannot be loaded: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StructuredLateBlockScoringError("truth sidecar must be a mapping")
    return payload


def _validate_truth_rows(
    truth: Mapping[str, Any],
) -> dict[str, int]:
    rows = truth.get("rows")
    if not isinstance(rows, list) or not rows:
        raise StructuredLateBlockScoringError(
            "truth sidecar rows must be a nonempty list"
        )
    truth_by_id: dict[str, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StructuredLateBlockScoringError(
                f"truth row {position} must be a mapping"
            )
        if "query_token" not in row or "true_class_index" not in row:
            raise StructuredLateBlockScoringError(
                f"truth row {position} lacks query_token/true_class_index"
            )
        query_id = row["query_token"]
        true_class = row["true_class_index"]
        if not isinstance(query_id, str) or not query_id:
            raise StructuredLateBlockScoringError(
                f"truth row {position} has an invalid opaque query_token"
            )
        if (
            isinstance(true_class, bool)
            or not isinstance(true_class, int)
            or true_class < 0
        ):
            raise StructuredLateBlockScoringError(
                f"truth row {position} has an invalid true_class_index"
            )
        if query_id in truth_by_id:
            raise StructuredLateBlockScoringError(
                "truth sidecar contains duplicate query_token"
            )
        truth_by_id[query_id] = int(true_class)
    return truth_by_id


def _old_class_metrics(
    true_class_ids: np.ndarray,
    predicted_class_ids: np.ndarray,
) -> dict[str, Any]:
    correct_mask = predicted_class_ids == true_class_ids
    per_class_accuracy: dict[str, float] = {}
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}
    for class_id in sorted(int(value) for value in np.unique(true_class_ids).tolist()):
        class_mask = true_class_ids == class_id
        total = int(class_mask.sum())
        correct = int(correct_mask[class_mask].sum())
        key = str(class_id)
        per_class_total[key] = total
        per_class_correct[key] = correct
        per_class_accuracy[key] = float(correct / total)
    class_accuracies = list(per_class_accuracy.values())
    return {
        "micro_accuracy": float(correct_mask.mean()),
        "macro_accuracy": float(np.mean(class_accuracies)),
        "per_class_accuracy": per_class_accuracy,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
        "floor_accuracy": float(min(class_accuracies)),
    }


def score_stage2b_predictions(
    prediction_path: str | Path,
    truth_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Score one DA1_REG0 row after closing prediction validation first."""

    destination = Path(output_path)
    if destination.exists():
        raise StructuredLateBlockScoringError(
            f"scoring output already exists: {destination}"
        )

    # This function returns only after exact schema, shapes, unique IDs, dtypes
    # and finite scores have all been checked.  Truth is unopened until then.
    query_ids, predicted_class_ids, _scores = _load_and_validate_predictions(
        prediction_path
    )
    prediction_row_count = int(query_ids.shape[0])

    truth = _load_truth_json(truth_path)
    truth_by_id = _validate_truth_rows(truth)
    prediction_id_set = set(query_ids.tolist())
    truth_id_set = set(truth_by_id)
    if prediction_id_set != truth_id_set:
        raise StructuredLateBlockScoringError(
            "exact opaque-ID join failed: "
            f"prediction_only={sorted(prediction_id_set - truth_id_set)}, "
            f"truth_only={sorted(truth_id_set - prediction_id_set)}"
        )
    true_class_ids = np.asarray(
        [truth_by_id[query_id] for query_id in query_ids.tolist()],
        dtype=np.int64,
    )
    metrics = _old_class_metrics(true_class_ids, predicted_class_ids)
    result: dict[str, Any] = {
        "schema": SCORE_SCHEMA,
        "status": "ANALYZED",
        "state": "DA1_REG0",
        "registration_state": "REG0",
        "join_policy": "exact_opaque_query_id",
        "prediction_rows_verified_before_truth_open": prediction_row_count,
        "truth_rows_joined": int(true_class_ids.shape[0]),
        "old_class_metrics": metrics,
        "new_class_metrics": {
            "new_class_accuracy": "N/A",
            "seen_new_accuracy": "N/A",
            "old_new_harmonic_mean": "N/A",
            "reason": "REG0 has no registered new classes",
        },
        "mrior_comparison": {
            "status": "UNKNOWN",
            "baseline": "source_free_MRIOR_same_row",
            "mean_delta_pp": "UNKNOWN",
            "floor_delta_pp": "UNKNOWN",
            "promotion_verdict": "UNKNOWN",
            "reason": "no compliant same-row source-free MRIOR baseline was supplied",
        },
        "scorer_output_must_not_feed_predictor": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def score_stage2c_predictions(
    prediction_path: str | Path,
    truth_path: str | Path,
    *,
    output_path: str | Path,
    old_class_ids: list[int] | tuple[int, ...],
    class_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Score one frozen DA1_REG1 row after prediction closure."""

    destination = Path(output_path)
    if destination.exists():
        raise StructuredLateBlockScoringError(
            f"scoring output already exists: {destination}"
        )
    names = [str(value) for value in class_names]
    if not names or any(not value for value in names) or len(set(names)) != len(names):
        raise StructuredLateBlockScoringError(
            "class registry names must be nonempty and unique"
        )
    old_ids = sorted({int(value) for value in old_class_ids})
    registry_ids = set(range(len(names)))
    if not old_ids or not set(old_ids) < registry_ids:
        raise StructuredLateBlockScoringError(
            "old class registry must be a nonempty strict subset"
        )

    # Exact prediction schema, shape, IDs, dtypes and finite values are closed
    # before the scorer is permitted to open the truth sidecar.
    query_ids, predicted_class_ids, scores = _load_and_validate_predictions(
        prediction_path
    )
    if scores.shape[1] != len(names) or np.any(predicted_class_ids >= len(names)):
        raise StructuredLateBlockScoringError(
            "prediction class registry does not match scores/class_names"
        )
    prediction_row_count = int(query_ids.shape[0])

    truth = _load_truth_json(truth_path)
    truth_by_id = _validate_truth_rows(truth)
    prediction_id_set = set(query_ids.tolist())
    truth_id_set = set(truth_by_id)
    if prediction_id_set != truth_id_set:
        raise StructuredLateBlockScoringError(
            "exact opaque-ID join failed: "
            f"prediction_only={sorted(prediction_id_set - truth_id_set)}, "
            f"truth_only={sorted(truth_id_set - prediction_id_set)}"
        )
    true_class_ids = np.asarray(
        [truth_by_id[query_id] for query_id in query_ids.tolist()], dtype=np.int64
    )
    if np.any(true_class_ids >= len(names)):
        raise StructuredLateBlockScoringError(
            "truth class registry is outside class_names"
        )

    correct = predicted_class_ids == true_class_ids
    old_mask = np.isin(true_class_ids, np.asarray(old_ids, dtype=np.int64))
    new_mask = ~old_mask
    if not old_mask.any() or not new_mask.any():
        raise StructuredLateBlockScoringError(
            "REG1 scoring requires both old-class and new-class query rows"
        )
    per_class_accuracy: dict[str, float] = {}
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}
    for class_id, name in enumerate(names):
        class_mask = true_class_ids == class_id
        total = int(class_mask.sum())
        if total < 1:
            raise StructuredLateBlockScoringError(
                f"truth has no query row for registered class {name}"
            )
        class_correct = int(correct[class_mask].sum())
        per_class_total[name] = total
        per_class_correct[name] = class_correct
        per_class_accuracy[name] = float(class_correct / total)

    result: dict[str, Any] = {
        "schema": STAGE2C_SCORE_SCHEMA,
        "status": "ANALYZED",
        "state": "DA1_REG1",
        "registration_state": "REG1",
        "join_policy": "exact_opaque_query_id",
        "prediction_rows_verified_before_truth_open": prediction_row_count,
        "truth_rows_joined": int(true_class_ids.shape[0]),
        "overall_accuracy": float(correct.mean()),
        "old_class_accuracy": float(correct[old_mask].mean()),
        "new_class_accuracy": float(correct[new_mask].mean()),
        "macro_accuracy": float(np.mean(list(per_class_accuracy.values()))),
        "floor_accuracy": float(min(per_class_accuracy.values())),
        "per_class_accuracy": per_class_accuracy,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
        "old_class_ids": old_ids,
        "class_names": names,
        "scorer_output_must_not_feed_predictor": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


__all__ = [
    "PREDICTION_SCHEMA",
    "SCORE_SCHEMA",
    "STAGE2C_SCORE_SCHEMA",
    "StructuredLateBlockScoringError",
    "score_stage2b_predictions",
    "score_stage2c_predictions",
]
