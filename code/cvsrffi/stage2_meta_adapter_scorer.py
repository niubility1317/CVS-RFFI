"""Truth-last, receipt-bound scorer for tri-R4 Stage2-B rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PREDICTION_SCHEMA = frozenset({"query_ids", "predicted_class_ids", "scores"})
SCORE_SCHEMA = "cvs.stage2.meta_adapter.score.v1"
MATRIX_SCHEMA = "cvs.stage2.meta_adapter.matrix_decision.v1"
REG0_STATES = ("DA0_REG0", "DA1_REG0")
_PROMOTION_EPSILON_PP = 1.0e-9
_ROW_FIELDS = (
    "candidate_id", "bundle_id", "protocol_schema", "phase2_data_status",
    "capsule_id", "split_id", "receiver", "scenario", "operating_point",
    "seed", "k_shot",
)


class MetaAdapterScoringError(ValueError):
    """Raised when a prediction/truth pair cannot be scored safely."""


@dataclass(frozen=True)
class StateScore:
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
                "new_class_accuracy": None, "seen_new_acc": None,
                "seen_new_accuracy": None, "h_old_new": None,
                "old_new_harmonic_mean": None,
                "reason": "REG0 has no registered new classes",
            },
        }


@dataclass(frozen=True)
class PairedStage2BScore:
    da0: StateScore
    da1: StateScore
    mean_delta_pp: float
    floor_delta_pp: float
    candidate_id: str
    bundle_id: str
    row_id: str
    row: Mapping[str, Any]
    registered_class_ids: tuple[int, ...]
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
        return _meets_threshold(self.mean_delta_pp, 1.0) and _meets_threshold(
            self.floor_delta_pp, 0.5
        )

    @property
    def verdict(self) -> str:
        return "PROMOTE_TO_TARGET25" if self.promote else "SCIENTIFIC_FAILURE_NO_PROMOTION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "candidate_id": self.candidate_id,
            "bundle_id": self.bundle_id,
            "row_id": self.row_id,
            "row": dict(self.row),
            "registered_class_ids": list(self.registered_class_ids),
            "states": list(REG0_STATES),
            "join_policy": "exact_ordered_opaque_query_id",
            "same_row_ids": self.same_row_ids,
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
        if payload.get("schema") != SCORE_SCHEMA:
            raise MetaAdapterScoringError("score schema mismatch")
        if payload.get("status") != "ANALYZED":
            raise MetaAdapterScoringError("score status must be ANALYZED")
        if payload.get("states") != list(REG0_STATES):
            raise MetaAdapterScoringError("score states mismatch")
        if "same_row_ids" not in payload:
            raise MetaAdapterScoringError("score payload lacks same_row_ids")
        if payload["same_row_ids"] is not True:
            raise MetaAdapterScoringError("score is not a same-row result")
        candidate_id = _required_string(payload, "candidate_id", "score")
        bundle_id = _required_string(payload, "bundle_id", "score")
        row_id = _required_string(payload, "row_id", "score")
        row = payload.get("row")
        if not isinstance(row, Mapping):
            raise MetaAdapterScoringError("score row must be a mapping")
        normalized_row = _validate_row(row)
        if normalized_row["candidate_id"] != candidate_id:
            raise MetaAdapterScoringError("score candidate_id disagrees with row")
        if normalized_row["bundle_id"] != bundle_id:
            raise MetaAdapterScoringError("score bundle_id disagrees with row")
        if _make_row_id(normalized_row) != row_id:
            raise MetaAdapterScoringError("score row_id does not bind row")
        registered = _registered_ids(payload.get("registered_class_ids"))
        da0_payload = payload.get("DA0_REG0")
        da1_payload = payload.get("DA1_REG0")
        if not isinstance(da0_payload, Mapping) or not isinstance(da1_payload, Mapping):
            raise MetaAdapterScoringError("score payload lacks DA0_REG0/DA1_REG0")
        da0 = _state_from_dict(da0_payload, expected_state="DA0_REG0")
        da1 = _state_from_dict(da1_payload, expected_state="DA1_REG0")
        if da0.query_ids != da1.query_ids:
            raise MetaAdapterScoringError("score state query IDs are not same-row")
        return cls(
            da0=da0, da1=da1,
            mean_delta_pp=_finite_number(payload.get("mean_delta_pp"), "mean_delta_pp"),
            floor_delta_pp=_finite_number(payload.get("floor_delta_pp"), "floor_delta_pp"),
            candidate_id=candidate_id, bundle_id=bundle_id, row_id=row_id,
            row=normalized_row, registered_class_ids=registered,
        )


@dataclass(frozen=True)
class MatrixDecision:
    mean_delta_pp: float
    floor_delta_pp: float
    promote: bool
    verdict: str
    row_count: int = 1
    mean_threshold_pp: float = 1.0
    floor_threshold_pp: float = 0.5
    candidate_id: str | None = None
    bundle_id: str | None = None
    target: str | None = None
    row_ids: tuple[str, ...] = ()

    @property
    def promotion_verdict(self) -> str:
        return self.verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MATRIX_SCHEMA, "status": "ANALYZED",
            "candidate_id": self.candidate_id, "bundle_id": self.bundle_id,
            "target": self.target, "row_ids": list(self.row_ids),
            "row_count": self.row_count, "mean_delta_pp": self.mean_delta_pp,
            "floor_delta_pp": self.floor_delta_pp,
            "thresholds_pp": {"mean": self.mean_threshold_pp, "floor": self.floor_threshold_pp},
            "promote": self.promote, "verdict": self.verdict,
        }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetaAdapterScoringError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MetaAdapterScoringError(f"{name} must be a finite number")
    return result


def _required_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MetaAdapterScoringError(f"{label} {key} must be a nonempty string")
    return value


def _registered_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MetaAdapterScoringError("registered_class_ids must be a nonempty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise MetaAdapterScoringError("registered_class_ids must contain nonnegative integers")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise MetaAdapterScoringError("registered_class_ids must be unique")
    return result


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _ROW_FIELDS if field not in row]
    if missing:
        raise MetaAdapterScoringError(f"receipt/score row lacks {missing[0]}")
    normalized = {field: row[field] for field in _ROW_FIELDS}
    for field in _ROW_FIELDS[:-2]:
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise MetaAdapterScoringError(f"row {field} must be a nonempty string")
    if normalized["protocol_schema"] != "p2_min_v1":
        raise MetaAdapterScoringError("row protocol_schema must be p2_min_v1")
    if normalized["phase2_data_status"] != "VALIDATED_ONCE":
        raise MetaAdapterScoringError("row phase2_data_status must be VALIDATED_ONCE")
    for field in ("seed", "k_shot"):
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MetaAdapterScoringError(f"row {field} must be a nonnegative integer")
    if normalized["k_shot"] < 1:
        raise MetaAdapterScoringError("row k_shot must be positive")
    return normalized


def _make_row_id(row: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _load_receipt(receipt_path: str | Path, da0_path: str | Path, da1_path: str | Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    path = Path(receipt_path)
    if not path.is_file():
        raise MetaAdapterScoringError(f"Task10 receipt is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetaAdapterScoringError(f"Task10 receipt cannot be loaded: {path}") from exc
    if not isinstance(payload, Mapping):
        raise MetaAdapterScoringError("Task10 receipt must be a mapping")
    if payload.get("status") != "PREDICTIONS_COMPLETE":
        raise MetaAdapterScoringError("Task10 receipt status must be PREDICTIONS_COMPLETE")
    if payload.get("states") != list(REG0_STATES):
        raise MetaAdapterScoringError("Task10 receipt states mismatch")
    if payload.get("states_same_row") is not True:
        raise MetaAdapterScoringError("Task10 receipt states_same_row must be true")
    row = _validate_row(payload)
    registered = _registered_ids(payload.get("registered_class_ids"))
    prediction_paths = payload.get("prediction_paths")
    if not isinstance(prediction_paths, Mapping):
        raise MetaAdapterScoringError("Task10 receipt lacks prediction_paths")
    expected = {"DA0_REG0": Path(da0_path).resolve(), "DA1_REG0": Path(da1_path).resolve()}
    for state, actual in expected.items():
        raw = prediction_paths.get(state)
        if not isinstance(raw, str) or not raw:
            raise MetaAdapterScoringError(f"Task10 receipt lacks {state} state path")
        bound = Path(raw)
        if not bound.is_absolute():
            bound = path.parent / bound
        if bound.resolve() != actual:
            raise MetaAdapterScoringError(f"Task10 receipt {state} state path mismatch")
    return row, registered, _make_row_id(row)


def _load_and_validate_prediction(prediction_path: str | Path, registered_class_ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(prediction_path)
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise MetaAdapterScoringError(f"prediction artifact is missing or invalid: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = frozenset(str(name) for name in archive.files)
            if names != PREDICTION_SCHEMA:
                raise MetaAdapterScoringError(
                    "prediction artifact schema mismatch: "
                    f"missing={sorted(PREDICTION_SCHEMA - names)} extra={sorted(names - PREDICTION_SCHEMA)}"
                )
            query_ids = np.asarray(archive["query_ids"]).copy()
            predicted = np.asarray(archive["predicted_class_ids"]).copy()
            scores = np.asarray(archive["scores"]).copy()
    except MetaAdapterScoringError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise MetaAdapterScoringError(f"prediction artifact cannot be loaded: {path}") from exc
    if query_ids.ndim != 1 or query_ids.dtype.kind not in {"U", "S"}:
        raise MetaAdapterScoringError("prediction artifact query_ids must be a string vector")
    query_ids = query_ids.astype(str)
    if not query_ids.size or any(not item for item in query_ids.tolist()):
        raise MetaAdapterScoringError("prediction artifact query IDs must be nonempty")
    if len(set(query_ids.tolist())) != query_ids.size:
        raise MetaAdapterScoringError("prediction artifact contains duplicate query IDs")
    if predicted.ndim != 1 or not np.issubdtype(predicted.dtype, np.integer):
        raise MetaAdapterScoringError("prediction artifact class IDs must be an integer vector")
    predicted = np.ascontiguousarray(predicted, dtype=np.int64)
    if scores.ndim != 2 or not np.issubdtype(scores.dtype, np.number):
        raise MetaAdapterScoringError("prediction artifact scores must be a numeric matrix")
    if scores.shape[1] != len(registered_class_ids):
        raise MetaAdapterScoringError("prediction score columns must match registered_class_ids")
    if not np.isfinite(scores).all():
        raise MetaAdapterScoringError("prediction artifact scores contain non-finite values")
    if predicted.shape[0] != query_ids.shape[0] or scores.shape[0] != query_ids.shape[0]:
        raise MetaAdapterScoringError("prediction artifact arrays do not have the same row count")
    registered = np.asarray(registered_class_ids, dtype=np.int64)
    if not np.isin(predicted, registered).all():
        raise MetaAdapterScoringError("prediction contains an ID outside the registered class set")
    if not np.array_equal(predicted, registered[np.argmax(scores, axis=1)]):
        raise MetaAdapterScoringError("predicted_class_ids do not match score argmax")
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
            raise MetaAdapterScoringError(f"truth row {position} has an invalid opaque query ID")
        if query_id in result:
            raise MetaAdapterScoringError("truth sidecar contains duplicate query ID")
        if isinstance(true_class, bool) or not isinstance(true_class, int) or true_class < 0:
            raise MetaAdapterScoringError(f"truth row {position} has an invalid true class ID")
        result[query_id] = int(true_class)
    return result


def _load_truth(truth_path: str | Path) -> dict[str, int]:
    path = Path(truth_path)
    if not path.is_file():
        raise MetaAdapterScoringError(f"truth sidecar is missing: {path}")
    if path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as archive:
                names = frozenset(str(name) for name in archive.files)
                if names != {"query_ids", "true_class_ids"}:
                    raise MetaAdapterScoringError("truth artifact schema mismatch")
                query_ids = np.asarray(archive["query_ids"]).copy()
                true_ids = np.asarray(archive["true_class_ids"]).copy()
        except MetaAdapterScoringError:
            raise
        except (OSError, ValueError, EOFError) as exc:
            raise MetaAdapterScoringError(f"truth artifact cannot be loaded: {path}") from exc
        if query_ids.ndim != 1 or query_ids.dtype.kind not in {"U", "S"}:
            raise MetaAdapterScoringError("truth query_ids must be an opaque string vector")
        if true_ids.ndim != 1 or true_ids.shape[0] != query_ids.shape[0] or not np.issubdtype(true_ids.dtype, np.integer):
            raise MetaAdapterScoringError("truth true_class_ids must align with query_ids")
        return _truth_rows_to_mapping([
            {"query_token": str(query_id), "true_class_index": int(true_id)}
            for query_id, true_id in zip(query_ids.astype(str).tolist(), true_ids.tolist())
        ])
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetaAdapterScoringError(f"truth sidecar cannot be loaded: {path}") from exc
    if not isinstance(payload, Mapping):
        raise MetaAdapterScoringError("truth sidecar must be a mapping")
    if "rows" in payload:
        return _truth_rows_to_mapping(payload["rows"])
    if isinstance(payload.get("query_ids"), list) and isinstance(payload.get("true_class_ids"), list):
        if len(payload["query_ids"]) != len(payload["true_class_ids"]):
            raise MetaAdapterScoringError("truth query_ids/true_class_ids lengths do not align")
        return _truth_rows_to_mapping([
            {"query_token": query_id, "true_class_index": true_id}
            for query_id, true_id in zip(payload["query_ids"], payload["true_class_ids"])
        ])
    raise MetaAdapterScoringError("truth sidecar lacks rows")


def _load_truth_json(truth_path: str | Path) -> dict[str, int]:
    return _load_truth(truth_path)


def _metrics(state: str, query_ids: np.ndarray, true_ids: np.ndarray, predicted: np.ndarray) -> StateScore:
    correct = predicted == true_ids
    per_accuracy: dict[str, float] = {}
    per_correct: dict[str, int] = {}
    per_total: dict[str, int] = {}
    for class_id in sorted(int(value) for value in np.unique(true_ids).tolist()):
        mask = true_ids == class_id
        total = int(mask.sum())
        right = int(correct[mask].sum())
        key = str(class_id)
        per_total[key], per_correct[key], per_accuracy[key] = total, right, float(right / total)
    values = tuple(per_accuracy.values())
    return StateScore(
        state=state, registration_state="REG0",
        query_ids=tuple(str(value) for value in query_ids.tolist()),
        mean_old_acc=float(np.mean(values)), old_class_floor=float(min(values)),
        per_class_accuracy=per_accuracy, per_class_correct=per_correct,
        per_class_total=per_total, micro_old_acc=float(correct.mean()),
    )


def score_meta_adapter_pair(da0_path: str | Path, da1_path: str | Path, truth_path: str | Path, *, receipt_path: str | Path | None = None) -> PairedStage2BScore:
    """Score one pair only after receipt and both predictions close."""
    da0, da1 = Path(da0_path), Path(da1_path)
    if da0.parent.resolve() != da1.parent.resolve():
        raise MetaAdapterScoringError("DA0/DA1 artifacts must share one receipt directory")
    receipt = Path(receipt_path) if receipt_path is not None else da0.parent / "receipt.json"
    row, registered, row_id = _load_receipt(receipt, da0, da1)
    da0_ids, da0_pred, _ = _load_and_validate_prediction(da0, registered)
    da1_ids, da1_pred, _ = _load_and_validate_prediction(da1, registered)
    if not np.array_equal(da0_ids, da1_ids):
        raise MetaAdapterScoringError("DA0/DA1 prediction artifacts must use the same ordered query IDs")
    # This is intentionally the first access to truth_path.
    truth_by_id = _load_truth_json(truth_path)
    prediction_ids = tuple(str(value) for value in da0_ids.tolist())
    if set(truth_by_id) != set(prediction_ids):
        raise MetaAdapterScoringError("exact opaque-ID join failed between prediction and truth")
    truth_labels = np.asarray([truth_by_id[item] for item in prediction_ids], dtype=np.int64)
    if not np.isin(truth_labels, np.asarray(registered, dtype=np.int64)).all():
        raise MetaAdapterScoringError("truth contains an ID outside the registered class set")
    da0_score = _metrics("DA0_REG0", da0_ids, truth_labels, da0_pred)
    da1_score = _metrics("DA1_REG0", da1_ids, truth_labels, da1_pred)
    return PairedStage2BScore(
        da0=da0_score, da1=da1_score,
        mean_delta_pp=float((da1_score.mean_old_acc - da0_score.mean_old_acc) * 100.0),
        floor_delta_pp=float((da1_score.old_class_floor - da0_score.old_class_floor) * 100.0),
        candidate_id=str(row["candidate_id"]), bundle_id=str(row["bundle_id"]),
        row_id=row_id, row=row, registered_class_ids=registered,
    )


def _meets_threshold(value: float, threshold: float) -> bool:
    return value + _PROMOTION_EPSILON_PP >= threshold


def summarize_rows(*, mean_delta_pp: float, floor_delta_pp: float) -> MatrixDecision:
    mean_delta = _finite_number(mean_delta_pp, "mean_delta_pp")
    floor_delta = _finite_number(floor_delta_pp, "floor_delta_pp")
    promote = _meets_threshold(mean_delta, 1.0) and _meets_threshold(floor_delta, 0.5)
    return MatrixDecision(
        mean_delta_pp=mean_delta, floor_delta_pp=floor_delta, promote=promote,
        verdict="PROMOTE_TO_TARGET25" if promote else "SCIENTIFIC_FAILURE_NO_PROMOTION",
    )


def _state_from_dict(payload: Mapping[str, Any], *, expected_state: str) -> StateScore:
    if payload.get("state") != expected_state or payload.get("registration_state") != "REG0":
        raise MetaAdapterScoringError(f"matrix score state mismatch for {expected_state}")
    metrics = payload.get("old_class_metrics")
    if not isinstance(metrics, Mapping):
        raise MetaAdapterScoringError("matrix state lacks old_class_metrics")
    query_ids = payload.get("query_ids")
    if not isinstance(query_ids, list) or not query_ids:
        raise MetaAdapterScoringError("matrix state query_ids must be nonempty")
    per_class, correct, total = metrics.get("per_class_accuracy"), metrics.get("per_class_correct"), metrics.get("per_class_total")
    if not all(isinstance(item, Mapping) for item in (per_class, correct, total)):
        raise MetaAdapterScoringError("matrix state class metrics must be mappings")
    return StateScore(
        state=expected_state, registration_state="REG0",
        query_ids=tuple(str(value) for value in query_ids),
        mean_old_acc=_finite_number(metrics.get("mean_old_acc"), "mean_old_acc"),
        old_class_floor=_finite_number(metrics.get("floor_accuracy"), "floor_accuracy"),
        per_class_accuracy={str(key): _finite_number(value, "per_class_accuracy") for key, value in per_class.items()},
        per_class_correct={str(key): int(value) for key, value in correct.items()},
        per_class_total={str(key): int(value) for key, value in total.items()},
        micro_old_acc=_finite_number(metrics.get("micro_accuracy"), "micro_accuracy"),
    )


def summarize_meta_adapter_matrix(scores: Iterable[PairedStage2BScore | Mapping[str, Any]], *, expected_target: str | None = None) -> MatrixDecision:
    raw_scores = list(scores)
    raw_candidate_values = [
        item.candidate_id if isinstance(item, PairedStage2BScore) else item.get("candidate_id")
        for item in raw_scores
        if isinstance(item, (PairedStage2BScore, Mapping))
    ]
    if all(isinstance(value, str) for value in raw_candidate_values) and len(set(raw_candidate_values)) > 1:
        raise MetaAdapterScoringError("matrix mixes candidate_id values")
    raw_bundle_values = [
        item.bundle_id if isinstance(item, PairedStage2BScore) else item.get("bundle_id")
        for item in raw_scores
        if isinstance(item, (PairedStage2BScore, Mapping))
    ]
    if all(isinstance(value, str) for value in raw_bundle_values) and len(set(raw_bundle_values)) > 1:
        raise MetaAdapterScoringError("matrix mixes bundle_id values")
    parsed = [
        PairedStage2BScore.from_dict(item.to_dict())
        if isinstance(item, PairedStage2BScore)
        else PairedStage2BScore.from_dict(item)
        for item in raw_scores
    ]
    if not parsed:
        suffix = f" for {expected_target}" if expected_target else ""
        raise MetaAdapterScoringError(f"matrix scores must be nonempty{suffix}")
    if expected_target not in {None, "Target5", "Target25"}:
        raise MetaAdapterScoringError("expected_target must be Target5 or Target25")
    if expected_target is not None:
        expected_count = 5 if expected_target == "Target5" else 25
        if len(parsed) != expected_count:
            raise MetaAdapterScoringError(f"{expected_target} requires {expected_count} rows")
    if len({item.candidate_id for item in parsed}) != 1:
        raise MetaAdapterScoringError("matrix mixes candidate_id values")
    if len({item.bundle_id for item in parsed}) != 1:
        raise MetaAdapterScoringError("matrix mixes bundle_id values")
    row_ids = tuple(item.row_id for item in parsed)
    if len(set(row_ids)) != len(row_ids):
        raise MetaAdapterScoringError("matrix contains duplicate row_id")
    if len({item.registered_class_ids for item in parsed}) != 1:
        raise MetaAdapterScoringError("matrix mixes registered_class_ids")
    da0_mean = float(np.mean([item.da0.mean_old_acc for item in parsed]))
    da1_mean = float(np.mean([item.da1.mean_old_acc for item in parsed]))
    da0_floor = float(min(item.da0.old_class_floor for item in parsed))
    da1_floor = float(min(item.da1.old_class_floor for item in parsed))
    decision = summarize_rows(
        mean_delta_pp=(da1_mean - da0_mean) * 100.0,
        floor_delta_pp=(da1_floor - da0_floor) * 100.0,
    )
    return MatrixDecision(
        mean_delta_pp=decision.mean_delta_pp, floor_delta_pp=decision.floor_delta_pp,
        promote=decision.promote, verdict=decision.verdict, row_count=len(parsed),
        candidate_id=parsed[0].candidate_id, bundle_id=parsed[0].bundle_id,
        target=expected_target, row_ids=row_ids,
    )


def write_score_json(score: PairedStage2BScore, output_path: str | Path) -> None:
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
    "MATRIX_SCHEMA", "MatrixDecision", "MetaAdapterScoringError",
    "PREDICTION_SCHEMA", "PairedStage2BScore", "SCORE_SCHEMA", "StateScore",
    "score_meta_adapter_pair", "summarize_meta_adapter_matrix", "summarize_rows",
    "write_score_json",
]
