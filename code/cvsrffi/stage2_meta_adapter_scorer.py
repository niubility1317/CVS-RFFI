"""Truth-last, same-row scorer for the tri-R4 meta-adapter Stage2-B row.

The scorer has no predictor or model dependency.  It validates both frozen
prediction artifacts, including their ordered opaque query IDs, before it
opens the independent truth sidecar.  REG0 therefore reports only old-class
metrics; new-class metrics are represented as ``None`` rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PREDICTION_SCHEMA = frozenset(
    {"query_ids", "predicted_class_ids", "scores"}
)
SCORE_SCHEMA = "cvs.stage2.meta_adapter.score.v1"
REG0_STATES = ("DA0_REG0", "DA1_REG0")


class MetaAdapterScoringError(ValueError):
    """Raised when a prediction/truth pair cannot be scored safely."""


@dataclass(frozen=True)
class StateScore:
    """Truth-side metrics for one frozen REG0 state."""

    state: str
    registration_state: str
    query_ids: tuple[str, ...]
    mean_old_acc: float
    old_class_floor: float
    per_class_accuracy: Mapping[str, float]
    per_class_correct: Mapping[str, int]
    per_class_total: Mapping[str, int]
    micro_old_acc: float
    seen_new_acc: None = None
    h_old_new: None = None

    @property
    def old_acc(self) -> float:
        """Compatibility alias for the equal-weight old-class mean."""

        return self.mean_old_acc

    @property
    def macro_old_acc(self) -> float:
        return self.mean_old_acc

    @property
    def floor_old_acc(self) -> float:
        return self.old_class_floor

    @property
    def new_class_accuracy(self) -> None:
        return None

    @property
    def old_new_harmonic_mean(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "registration_state": self.registration_state,
            "query_ids": list(self.query_ids),
            "old_class_metrics": {
                "micro_accuracy": self.micro_old_acc,
                "macro_accuracy": self.mean_old_acc,
                "mean_old_acc": self.mean_old_acc,
                "floor_accuracy": self.old_class_floor,
                "per_class_accuracy": dict(self.per_class_accuracy),
                "per_class_correct": dict(self.per_class_correct),
                "per_class_total": dict(self.per_class_total),
            },
            "new_class_metrics": {
                "new_class_accuracy": None,
                "seen_new_acc": None,
                "seen_new_accuracy": None,
                "h_old_new": None,
                "old_new_harmonic_mean": None,
                "reason": "REG0 has no registered new classes",
            },
        }


@dataclass(frozen=True)
class PairedStage2BScore:
    """Paired DA0/DA1 score for one immutable row."""

    da0: StateScore
    da1: StateScore
    mean_delta_pp: float
    floor_delta_pp: float
    same_row_ids: bool = True
    schema: str = SCORE_SCHEMA
    status: str = "ANALYZED"

    @property
    def delta_mean_old_acc_pp(self) -> float:
        return self.mean_delta_pp

    @property
    def delta_floor_old_acc_pp(self) -> float:
        return self.floor_delta_pp

    @property
    def promote(self) -> bool:
        return self.mean_delta_pp >= 1.0 and self.floor_delta_pp >= 0.5

    @property
    def verdict(self) -> str:
        return "PROMOTE_TO_TARGET25" if self.promote else "SCIENTIFIC_FAILURE_NO_PROMOTION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "states": list(REG0_STATES),
            "join_policy": "exact_ordered_opaque_query_id",
            "same_row_ids": bool(self.same_row_ids),
            "prediction_rows_verified_before_truth_open": len(self.da0.query_ids),
            "truth_rows_joined": len(self.da0.query_ids),
            "DA0_REG0": self.da0.to_dict(),
            "DA1_REG0": self.da1.to_dict(),
            "da0": self.da0.to_dict(),
            "da1": self.da1.to_dict(),
            "mean_delta_pp": self.mean_delta_pp,
            "floor_delta_pp": self.floor_delta_pp,
            "promotion_verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedStage2BScore":
        if not isinstance(payload, Mapping):
            raise MetaAdapterScoringError("score payload must be a mapping")
        da0_payload = payload.get("da0", payload.get("DA0_REG0"))
        da1_payload = payload.get("da1", payload.get("DA1_REG0"))
        if not isinstance(da0_payload, Mapping) or not isinstance(da1_payload, Mapping):
            raise MetaAdapterScoringError("score payload lacks DA0/DA1 states")
        return cls(
            da0=_state_from_dict(da0_payload, expected_state="DA0_REG0"),
            da1=_state_from_dict(da1_payload, expected_state="DA1_REG0"),
            mean_delta_pp=_finite_number(payload.get("mean_delta_pp"), "mean_delta_pp"),
            floor_delta_pp=_finite_number(payload.get("floor_delta_pp"), "floor_delta_pp"),
            same_row_ids=bool(payload.get("same_row_ids", True)),
            schema=str(payload.get("schema", SCORE_SCHEMA)),
            status=str(payload.get("status", "ANALYZED")),
        )


@dataclass(frozen=True)
class MatrixDecision:
    """Target5/Target25 decision from same-row score objects."""

    mean_delta_pp: float
    floor_delta_pp: float
    promote: bool
    verdict: str
    row_count: int = 1
    mean_threshold_pp: float = 1.0
    floor_threshold_pp: float = 0.5

    @property
    def promotion_verdict(self) -> str:
        return self.verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cvs.stage2.meta_adapter.matrix_decision.v1",
            "status": "ANALYZED",
            "row_count": self.row_count,
            "mean_delta_pp": self.mean_delta_pp,
            "floor_delta_pp": self.floor_delta_pp,
            "thresholds_pp": {
                "mean": self.mean_threshold_pp,
                "floor": self.floor_threshold_pp,
            },
            "promote": self.promote,
            "verdict": self.verdict,
        }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetaAdapterScoringError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MetaAdapterScoringError(f"{name} must be a finite number")
    return result


def _load_and_validate_prediction(
    prediction_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(prediction_path)
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise MetaAdapterScoringError(
            f"prediction artifact is missing or invalid: {path}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = frozenset(str(name) for name in archive.files)
            if names != PREDICTION_SCHEMA:
                raise MetaAdapterScoringError(
                    "prediction artifact schema mismatch: "
                    f"missing={sorted(PREDICTION_SCHEMA - names)} "
                    f"extra={sorted(names - PREDICTION_SCHEMA)}"
                )
            query_ids = np.asarray(archive["query_ids"]).copy()
            predicted = np.asarray(archive["predicted_class_ids"]).copy()
            scores = np.asarray(archive["scores"]).copy()
    except MetaAdapterScoringError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise MetaAdapterScoringError(
            f"prediction artifact cannot be loaded: {path}"
        ) from exc

    if (
        query_ids.ndim != 1
        or query_ids.shape[0] < 1
        or query_ids.dtype.kind not in {"U", "S"}
    ):
        raise MetaAdapterScoringError(
            "prediction artifact query_ids must be a nonempty opaque string vector"
        )
    query_ids = query_ids.astype(str)
    if any(not value for value in query_ids.tolist()):
        raise MetaAdapterScoringError("prediction artifact query_ids contain an empty ID")
    if len(set(query_ids.tolist())) != int(query_ids.shape[0]):
        raise MetaAdapterScoringError("prediction artifact query_ids must be unique")

    if predicted.ndim != 1 or not np.issubdtype(predicted.dtype, np.integer):
        raise MetaAdapterScoringError(
            "prediction artifact predicted_class_ids must be an integer vector"
        )
    predicted = np.ascontiguousarray(predicted, dtype=np.int64)
    if np.any(predicted < 0):
        raise MetaAdapterScoringError(
            "prediction artifact class IDs must be nonnegative"
        )

    if (
        scores.ndim != 2
        or scores.shape[1] < 1
        or not np.issubdtype(scores.dtype, np.number)
    ):
        raise MetaAdapterScoringError(
            "prediction artifact scores must be a numeric [queries,classes] matrix"
        )
    if not np.isfinite(scores).all():
        raise MetaAdapterScoringError(
            "prediction artifact scores contain non-finite values"
        )
    rows = int(query_ids.shape[0])
    if predicted.shape[0] != rows or scores.shape[0] != rows:
        raise MetaAdapterScoringError(
            "prediction artifact arrays do not have the same row count"
        )
    return query_ids, predicted, np.ascontiguousarray(scores, dtype=np.float32)


def _truth_rows_to_mapping(rows: Any) -> dict[str, int]:
    if not isinstance(rows, list) or not rows:
        raise MetaAdapterScoringError("truth rows must be a nonempty list")
    result: dict[str, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MetaAdapterScoringError(f"truth row {position} must be a mapping")
        query_id = row.get("query_token", row.get("query_id"))
        true_class = row.get("true_class_index", row.get("true_class_id"))
        if not isinstance(query_id, str) or not query_id:
            raise MetaAdapterScoringError(
                f"truth row {position} has an invalid opaque query ID"
            )
        if query_id in result:
            raise MetaAdapterScoringError("truth sidecar contains duplicate query ID")
        if (
            isinstance(true_class, bool)
            or not isinstance(true_class, int)
            or true_class < 0
        ):
            raise MetaAdapterScoringError(
                f"truth row {position} has an invalid true class ID"
            )
        result[query_id] = int(true_class)
    return result


def _load_truth(truth_path: str | Path) -> dict[str, int]:
    """Open and validate truth; callers must invoke this only after predictions."""

    path = Path(truth_path)
    if not path.is_file():
        raise MetaAdapterScoringError(f"truth sidecar is missing: {path}")
    if path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                names = frozenset(str(name) for name in archive.files)
                if names != {"query_ids", "true_class_ids"}:
                    raise MetaAdapterScoringError(
                        "truth artifact schema mismatch: "
                        f"missing={sorted({'query_ids', 'true_class_ids'} - names)} "
                        f"extra={sorted(names - {'query_ids', 'true_class_ids'})}"
                    )
                query_ids = np.asarray(archive["query_ids"]).copy()
                true_ids = np.asarray(archive["true_class_ids"]).copy()
        except MetaAdapterScoringError:
            raise
        except (OSError, ValueError, EOFError) as exc:
            raise MetaAdapterScoringError(f"truth artifact cannot be loaded: {path}") from exc
        if query_ids.ndim != 1 or query_ids.dtype.kind not in {"U", "S"}:
            raise MetaAdapterScoringError("truth query_ids must be an opaque string vector")
        query_ids = query_ids.astype(str)
        if (
            true_ids.ndim != 1
            or true_ids.shape[0] != query_ids.shape[0]
            or not np.issubdtype(true_ids.dtype, np.integer)
        ):
            raise MetaAdapterScoringError("truth true_class_ids must align with query_ids")
        if np.any(true_ids < 0):
            raise MetaAdapterScoringError("truth class IDs must be nonnegative")
        return _truth_rows_to_mapping(
            [
                {"query_token": str(query_id), "true_class_index": int(true_id)}
                for query_id, true_id in zip(query_ids.tolist(), true_ids.tolist())
            ]
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetaAdapterScoringError(f"truth sidecar cannot be loaded: {path}") from exc
    if not isinstance(payload, Mapping):
        raise MetaAdapterScoringError("truth sidecar must be a mapping")
    if "rows" in payload:
        return _truth_rows_to_mapping(payload["rows"])
    if "query_ids" in payload and "true_class_ids" in payload:
        query_ids = payload["query_ids"]
        true_ids = payload["true_class_ids"]
        if not isinstance(query_ids, list) or not isinstance(true_ids, list):
            raise MetaAdapterScoringError("truth query_ids/true_class_ids must be lists")
        if len(query_ids) != len(true_ids):
            raise MetaAdapterScoringError("truth query_ids/true_class_ids lengths do not align")
        return _truth_rows_to_mapping(
            [
                {"query_token": query_id, "true_class_index": true_id}
                for query_id, true_id in zip(query_ids, true_ids)
            ]
        )
    raise MetaAdapterScoringError("truth sidecar lacks rows")


def _load_truth_json(truth_path: str | Path) -> dict[str, int]:
    """Named compatibility hook for tests and JSON truth-sidecar callers."""

    return _load_truth(truth_path)


def _metrics(
    state: str,
    query_ids: np.ndarray,
    true_class_ids: np.ndarray,
    predicted_class_ids: np.ndarray,
) -> StateScore:
    per_class_accuracy: dict[str, float] = {}
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}
    correct = predicted_class_ids == true_class_ids
    for class_id in sorted(int(value) for value in np.unique(true_class_ids).tolist()):
        mask = true_class_ids == class_id
        total = int(mask.sum())
        right = int(correct[mask].sum())
        key = str(class_id)
        per_class_total[key] = total
        per_class_correct[key] = right
        per_class_accuracy[key] = float(right / total)
    class_values = tuple(per_class_accuracy.values())
    return StateScore(
        state=state,
        registration_state="REG0",
        query_ids=tuple(str(value) for value in query_ids.tolist()),
        mean_old_acc=float(np.mean(class_values)),
        old_class_floor=float(min(class_values)),
        per_class_accuracy=per_class_accuracy,
        per_class_correct=per_class_correct,
        per_class_total=per_class_total,
        micro_old_acc=float(correct.mean()),
    )


def score_meta_adapter_pair(
    da0_path: str | Path,
    da1_path: str | Path,
    truth_path: str | Path,
) -> PairedStage2BScore:
    """Score one DA0/DA1 pair, opening truth only after prediction closure."""

    da0_ids, da0_pred, _da0_scores = _load_and_validate_prediction(da0_path)
    da1_ids, da1_pred, _da1_scores = _load_and_validate_prediction(da1_path)
    if not np.array_equal(da0_ids, da1_ids):
        raise MetaAdapterScoringError(
            "DA0/DA1 prediction artifacts must use the same ordered query IDs"
        )

    # This is intentionally the first access to truth_path.
    truth_by_id = _load_truth_json(truth_path)
    prediction_ids = tuple(str(value) for value in da0_ids.tolist())
    truth_ids = set(truth_by_id)
    prediction_id_set = set(prediction_ids)
    if truth_ids != prediction_id_set:
        raise MetaAdapterScoringError(
            "exact opaque-ID join failed: "
            f"prediction_only={sorted(prediction_id_set - truth_ids)} "
            f"truth_only={sorted(truth_ids - prediction_id_set)}"
        )
    truth_labels = np.asarray(
        [truth_by_id[query_id] for query_id in prediction_ids],
        dtype=np.int64,
    )
    da0 = _metrics("DA0_REG0", da0_ids, truth_labels, da0_pred)
    da1 = _metrics("DA1_REG0", da1_ids, truth_labels, da1_pred)
    return PairedStage2BScore(
        da0=da0,
        da1=da1,
        mean_delta_pp=float((da1.mean_old_acc - da0.mean_old_acc) * 100.0),
        floor_delta_pp=float((da1.old_class_floor - da0.old_class_floor) * 100.0),
    )


def summarize_rows(*, mean_delta_pp: float, floor_delta_pp: float) -> MatrixDecision:
    """Apply the two independent Target25 thresholds to one aggregate."""

    mean_delta = _finite_number(mean_delta_pp, "mean_delta_pp")
    floor_delta = _finite_number(floor_delta_pp, "floor_delta_pp")
    promote = mean_delta >= 1.0 and floor_delta >= 0.5
    return MatrixDecision(
        mean_delta_pp=mean_delta,
        floor_delta_pp=floor_delta,
        promote=promote,
        verdict="PROMOTE_TO_TARGET25" if promote else "SCIENTIFIC_FAILURE_NO_PROMOTION",
    )


def _state_from_dict(payload: Mapping[str, Any], *, expected_state: str) -> StateScore:
    state = str(payload.get("state", expected_state))
    if state != expected_state:
        raise MetaAdapterScoringError(
            f"matrix score state mismatch: expected {expected_state}, got {state}"
        )
    metrics = payload.get("old_class_metrics", payload)
    if not isinstance(metrics, Mapping):
        raise MetaAdapterScoringError("matrix state lacks old_class_metrics")
    per_class = metrics.get("per_class_accuracy", {})
    correct = metrics.get("per_class_correct", {})
    total = metrics.get("per_class_total", {})
    if not isinstance(per_class, Mapping) or not isinstance(correct, Mapping) or not isinstance(total, Mapping):
        raise MetaAdapterScoringError("matrix state class metrics must be mappings")
    query_ids = payload.get("query_ids", [])
    if not isinstance(query_ids, list):
        query_ids = list(query_ids) if isinstance(query_ids, tuple) else []
    return StateScore(
        state=state,
        registration_state="REG0",
        query_ids=tuple(str(value) for value in query_ids),
        mean_old_acc=_finite_number(
            metrics.get("mean_old_acc", metrics.get("macro_accuracy")),
            "mean_old_acc",
        ),
        old_class_floor=_finite_number(
            metrics.get("floor_accuracy"), "floor_accuracy"
        ),
        per_class_accuracy={str(key): float(value) for key, value in per_class.items()},
        per_class_correct={str(key): int(value) for key, value in correct.items()},
        per_class_total={str(key): int(value) for key, value in total.items()},
        micro_old_acc=_finite_number(
            metrics.get("micro_accuracy", metrics.get("mean_old_acc", metrics.get("macro_accuracy"))),
            "micro_accuracy",
        ),
    )


def summarize_meta_adapter_matrix(
    scores: Iterable[PairedStage2BScore | Mapping[str, Any]],
) -> MatrixDecision:
    """Aggregate same-row scores with equal row means and global floors."""

    parsed = [
        item if isinstance(item, PairedStage2BScore) else PairedStage2BScore.from_dict(item)
        for item in scores
    ]
    if not parsed:
        raise MetaAdapterScoringError("matrix scores must be nonempty")
    if any(not item.same_row_ids for item in parsed):
        raise MetaAdapterScoringError("matrix contains a non-same-row score")
    da0_mean = float(np.mean([item.da0.mean_old_acc for item in parsed]))
    da1_mean = float(np.mean([item.da1.mean_old_acc for item in parsed]))
    da0_floor = float(min(item.da0.old_class_floor for item in parsed))
    da1_floor = float(min(item.da1.old_class_floor for item in parsed))
    decision = summarize_rows(
        mean_delta_pp=(da1_mean - da0_mean) * 100.0,
        floor_delta_pp=(da1_floor - da0_floor) * 100.0,
    )
    return MatrixDecision(
        mean_delta_pp=decision.mean_delta_pp,
        floor_delta_pp=decision.floor_delta_pp,
        promote=decision.promote,
        verdict=decision.verdict,
        row_count=len(parsed),
    )


def write_score_json(score: PairedStage2BScore, output_path: str | Path) -> None:
    """Persist a score once; an existing output is never replaced."""

    if not isinstance(score, PairedStage2BScore):
        raise TypeError("score must be a PairedStage2BScore")
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"scoring output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(score.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "MatrixDecision",
    "MetaAdapterScoringError",
    "PREDICTION_SCHEMA",
    "PairedStage2BScore",
    "SCORE_SCHEMA",
    "StateScore",
    "score_meta_adapter_pair",
    "summarize_meta_adapter_matrix",
    "summarize_rows",
    "write_score_json",
]
