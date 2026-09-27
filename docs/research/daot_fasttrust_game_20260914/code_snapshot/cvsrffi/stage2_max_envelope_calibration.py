"""Support-only max-new-envelope calibration for Stage2 registration.

For every row, old score columns are frozen.  A support-fitted bias is added
to the contiguous new-class suffix and then translated back so that the
maximum new-class score is exactly the same as before calibration.  Hence the
old-versus-new envelope is invariant while the identity selected inside the
new-class suffix may change.

The fitter consumes registered support scores, labels, and within-class shot
ranks only.  It exposes no query-fitting, role-oracle, quota, or batch-level
assignment surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from typing import Any, Sequence

import numpy as np


SCHEMA = "cvs.phase2.d30_max_envelope_calibration.v1"
OOF_FOLD_COUNT = 5
OBJECTIVES = ("floor_first", "balance_first", "overall_first")
MAX_COORDINATE_PASSES = 2
WORST_CLASS_FRACTION = 0.20
MAX_PREDICTOR_STATE_BYTES = 256 * 1024
_TOL = 1.0e-12


class MaxEnvelopeCalibrationError(ValueError):
    """Raised when support, calibration state, or inference drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_scores(
    value: np.ndarray, *, class_count: int | None = None, name: str
) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float32)
    if (
        scores.ndim != 2
        or len(scores) < 1
        or scores.shape[1] < 4
        or not np.isfinite(scores).all()
        or (class_count is not None and scores.shape[1] != class_count)
    ):
        raise MaxEnvelopeCalibrationError(f"{name} must be finite [N,C]")
    return np.ascontiguousarray(scores, dtype=np.float32)


@dataclass(frozen=True)
class MaxEnvelopeCalibrationConfig:
    objective: str = "floor_first"
    coordinate_passes: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", str(self.objective))
        self.validate()

    def validate(self) -> None:
        if self.objective not in OBJECTIVES:
            raise MaxEnvelopeCalibrationError(
                "max-envelope objective is not method-locked"
            )
        if (
            not isinstance(self.coordinate_passes, Integral)
            or isinstance(self.coordinate_passes, (bool, np.bool_))
            or not 1 <= int(self.coordinate_passes) <= MAX_COORDINATE_PASSES
        ):
            raise MaxEnvelopeCalibrationError(
                "max-envelope calibration uses one or two coordinate passes"
            )


@dataclass(frozen=True)
class MaxEnvelopeCalibrationState:
    schema: str
    registered_classes: tuple[str, ...]
    old_class_count: int
    k_shot: int
    enabled: bool
    biases: np.ndarray
    audit_json: str
    config: MaxEnvelopeCalibrationConfig

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.registered_classes)
        if (
            not isinstance(self.old_class_count, Integral)
            or isinstance(self.old_class_count, (bool, np.bool_))
            or not isinstance(self.k_shot, Integral)
            or isinstance(self.k_shot, (bool, np.bool_))
        ):
            raise MaxEnvelopeCalibrationError(
                "max-envelope state counts must be integers"
            )
        old_count = int(self.old_class_count)
        k_shot = int(self.k_shot)
        new_count = len(classes) - old_count
        biases = np.asarray(self.biases)
        if (
            not isinstance(self.enabled, (bool, np.bool_))
            or self.schema != SCHEMA
            or len(classes) < 4
            or len(set(classes)) != len(classes)
            or not 2 <= old_count < len(classes)
            or new_count < 2
            or k_shot < 1
            or biases.dtype != np.float32
            or biases.shape != (new_count,)
            or not np.isfinite(biases).all()
        ):
            raise MaxEnvelopeCalibrationError("max-envelope state drift")
        if not self.enabled and bool(np.any(biases != 0.0)):
            raise MaxEnvelopeCalibrationError(
                "disabled max-envelope state must be exact passthrough"
            )
        try:
            audit = json.loads(str(self.audit_json))
        except json.JSONDecodeError as exc:
            raise MaxEnvelopeCalibrationError(
                "max-envelope audit is invalid JSON"
            ) from exc
        if not isinstance(audit, dict):
            raise MaxEnvelopeCalibrationError(
                "max-envelope audit must be one object"
            )
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "k_shot", k_shot)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "biases", _readonly(biases, np.float32))
        object.__setattr__(self, "audit_json", _canonical_json(audit))
        if self.deployable_predictor_state_bytes > MAX_PREDICTOR_STATE_BYTES:
            raise MaxEnvelopeCalibrationError(
                "max-envelope predictor state exceeds 256KB"
            )

    @property
    def deployable_predictor_state_bytes(self) -> int:
        # The 32-byte fixed allowance covers enable/count fields and a compact
        # registry-binding digest; detailed fit evidence is external.
        return int(self.biases.nbytes + 32)

    @property
    def persistent_evidence_object_bytes(self) -> int:
        return int(
            self.deployable_predictor_state_bytes
            + len(self.audit_json.encode("utf-8"))
            + len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.registered_classes)
        )

    def resource_audit(self) -> dict[str, Any]:
        new_count = len(self.registered_classes) - self.old_class_count
        scalar_ops = 5 * new_count - 2 if self.enabled else 0
        return {
            "schema": "cvs.phase2.d30_max_envelope_calibration_resource.v1",
            "enabled": self.enabled,
            "k_shot": self.k_shot,
            "new_class_count": new_count,
            "fitted_parameter_count": new_count if self.enabled else 0,
            "gradient_trainable_parameter_count": 0,
            "bias_scalar_count": new_count if self.enabled else 0,
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "coordinate_pass_limit": int(self.config.coordinate_passes),
            "deployable_predictor_state_bytes": self.deployable_predictor_state_bytes,
            "external_evidence_audit_bytes": len(self.audit_json.encode("utf-8")),
            "audit_metadata_excluded_from_deployment_state": True,
            "persistent_state_cap_bytes": MAX_PREDICTOR_STATE_BYTES,
            "persistent_state_cap_pass": (
                self.deployable_predictor_state_bytes
                <= MAX_PREDICTOR_STATE_BYTES
            ),
            "estimated_extra_macs_per_query": 0,
            "estimated_scalar_ops_per_query": scalar_ops,
            "estimated_add_sub_ops_per_query": (
                3 * new_count if self.enabled else 0
            ),
            "estimated_max_compare_ops_per_query": (
                2 * (new_count - 1) if self.enabled else 0
            ),
            "scratch_bytes_per_query": 4 * new_count if self.enabled else 0,
            "old_score_writes_per_query": 0,
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_features_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "query_batch_statistics_used": False,
            "row_local_inference": True,
            "per_sample_all_registered_classes": True,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "source_sample_access": False,
            "source_derived_signal_access": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_source_sample_access": False,
            "phase2_source_cache_access": False,
            "phase2_source_label_access": False,
            "phase2_unapproved_source_derived_signal_access": False,
        }


def _validate_registry(
    registered_classes: Sequence[str], old_class_count: int
) -> tuple[tuple[str, ...], int]:
    classes = tuple(str(value) for value in registered_classes)
    if (
        not isinstance(old_class_count, Integral)
        or isinstance(old_class_count, (bool, np.bool_))
    ):
        raise MaxEnvelopeCalibrationError(
            "max-envelope old class count must be an integer"
        )
    old_count = int(old_class_count)
    if (
        len(classes) < 4
        or len(set(classes)) != len(classes)
        or not 2 <= old_count <= len(classes) - 2
    ):
        raise MaxEnvelopeCalibrationError(
            "max-envelope registered class order drift"
        )
    return classes, old_count


def _validate_support(
    support_scores: np.ndarray,
    support_labels: Sequence[str],
    support_shot_ranks: Sequence[int],
    registered_classes: Sequence[str],
    old_class_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    int,
    int,
]:
    classes, old_count = _validate_registry(
        registered_classes, old_class_count
    )
    scores = _validate_scores(
        support_scores,
        class_count=len(classes),
        name="max-envelope support scores",
    )
    labels = np.asarray(support_labels, dtype=str)
    ranks_raw = np.asarray(support_shot_ranks)
    if (
        labels.ndim != 1
        or ranks_raw.ndim != 1
        or len(labels) != len(scores)
        or len(ranks_raw) != len(scores)
        or not np.issubdtype(ranks_raw.dtype, np.integer)
        or isinstance(support_shot_ranks, (str, bytes))
    ):
        raise MaxEnvelopeCalibrationError(
            "max-envelope support labels/ranks drift"
        )
    ranks = np.asarray(ranks_raw, dtype=np.int64)
    if bool(np.any(ranks < 0)) or set(labels.tolist()) != set(classes):
        raise MaxEnvelopeCalibrationError(
            "max-envelope support registry drift"
        )
    counts = {name: int(np.sum(labels == name)) for name in classes}
    if len(set(counts.values())) != 1:
        raise MaxEnvelopeCalibrationError(
            "max-envelope support must be class-symmetric K-shot"
        )
    k_shot = next(iter(counts.values()))
    expected_ranks = list(range(k_shot))
    for name in classes:
        observed = sorted(ranks[labels == name].tolist())
        if observed != expected_ranks:
            raise MaxEnvelopeCalibrationError(
                "max-envelope each class must expose unique shot ranks 0..K-1"
            )
    return scores, labels, ranks, classes, old_count, k_shot


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    total = old_accuracy + new_accuracy
    return 0.0 if total <= 0.0 else 2.0 * old_accuracy * new_accuracy / total


def _metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_count: int,
) -> dict[str, Any]:
    predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    old_classes = classes[:old_count]
    new_classes = classes[old_count:]
    old_mask = np.isin(labels, np.asarray(old_classes))
    new_mask = ~old_mask
    per_old = {
        name: float(np.mean(predictions[labels == name] == name))
        for name in old_classes
    }
    per_new = {
        name: float(np.mean(predictions[labels == name] == name))
        for name in new_classes
    }
    ordered_new = sorted(per_new.values())
    worst_count = max(1, int(math.ceil(WORST_CLASS_FRACTION * len(new_classes))))
    old_acc = float(np.mean(predictions[old_mask] == labels[old_mask]))
    new_acc = float(np.mean(predictions[new_mask] == labels[new_mask]))
    return {
        "old_overall_accuracy": old_acc,
        "new_overall_accuracy": new_acc,
        "new_class_floor_accuracy": min(per_new.values()),
        "new_worst_20pct_accuracy": float(
            np.mean(np.asarray(ordered_new[:worst_count], dtype=np.float64))
        ),
        "H_old_new": _harmonic(old_acc, new_acc),
        "per_old_class_accuracy": per_old,
        "per_new_class_accuracy": per_new,
    }


def audit_envelope_confusions(
    scores: np.ndarray,
    labels: Sequence[str],
    registered_classes: Sequence[str],
    old_class_count: int,
) -> dict[str, Any]:
    """Audit the row-role ceiling using support labels only.

    ``old_win`` rows of a true new class cannot be repaired by a
    max-new-envelope-preserving calibration.  ``new_wrong`` rows form the
    potentially repairable pool; a single static bias need not repair all of
    them.
    """

    classes, old_count = _validate_registry(
        registered_classes, old_class_count
    )
    values = _validate_scores(
        scores, class_count=len(classes), name="max-envelope audit scores"
    )
    label_values = np.asarray(labels, dtype=str)
    if (
        label_values.ndim != 1
        or len(label_values) != len(values)
        or not set(label_values.tolist()).issubset(set(classes))
    ):
        raise MaxEnvelopeCalibrationError(
            "max-envelope audit labels drift"
        )
    old_max = np.max(values[:, :old_count], axis=1)
    new_scores = values[:, old_count:]
    new_max = np.max(new_scores, axis=1)
    new_winner = np.argmax(new_scores, axis=1)
    # The registry is old-prefix and np.argmax is first-winner.  Therefore an
    # exact old/new envelope tie belongs to the old prefix.
    new_group_wins = new_max > old_max
    new_classes = classes[old_count:]
    per_new: dict[str, Any] = {}
    aggregate = {
        "sample_count": 0,
        "old_win": 0,
        "new_correct": 0,
        "new_wrong": 0,
    }
    for new_index, name in enumerate(new_classes):
        mask = label_values == name
        count = int(np.sum(mask))
        if count == 0:
            continue
        old_win = int(np.sum(mask & ~new_group_wins))
        new_correct = int(np.sum(mask & new_group_wins & (new_winner == new_index)))
        new_wrong = int(np.sum(mask & new_group_wins & (new_winner != new_index)))
        record = {
            "sample_count": count,
            "old_win": old_win,
            "new_correct": new_correct,
            "new_wrong": new_wrong,
            "current_accuracy": new_correct / count,
            "reachable_ceiling_accuracy": (new_correct + new_wrong) / count,
            "repairable_gain_ceiling": new_wrong / count,
        }
        per_new[name] = record
        for key in aggregate:
            aggregate[key] += int(record[key])
    total_new = aggregate["sample_count"]
    aggregate.update(
        {
            "current_accuracy": (
                aggregate["new_correct"] / total_new if total_new else 0.0
            ),
            "reachable_ceiling_accuracy": (
                (aggregate["new_correct"] + aggregate["new_wrong"])
                / total_new
                if total_new
                else 0.0
            ),
            "repairable_gain_ceiling": (
                aggregate["new_wrong"] / total_new if total_new else 0.0
            ),
        }
    )
    old_mask = np.isin(label_values, np.asarray(classes[:old_count]))
    old_count_rows = int(np.sum(old_mask))
    old_to_new = int(np.sum(old_mask & new_group_wins))
    return {
        "schema": "cvs.phase2.d30_max_envelope_confusion_audit.v1",
        "tie_policy": "old_prefix_first_argmax",
        "per_new_class": per_new,
        "new_aggregate": aggregate,
        "old_aggregate": {
            "sample_count": old_count_rows,
            "old_to_new": old_to_new,
            "old_group_win": old_count_rows - old_to_new,
        },
    }


def _apply_biases(
    scores: np.ndarray, old_count: int, biases: np.ndarray
) -> np.ndarray:
    values = np.ascontiguousarray(scores, dtype=np.float32)
    if bool(np.all(np.asarray(biases, dtype=np.float32) == 0.0)):
        return values.copy()
    new_raw = values[:, old_count:]
    raw_max = np.max(new_raw, axis=1)
    shifted = new_raw.astype(np.float64) + np.asarray(biases, dtype=np.float64)
    shifted_max = np.max(shifted, axis=1)
    shifted_winner = np.argmax(shifted, axis=1)
    calibrated64 = (shifted - shifted_max[:, None]) + raw_max.astype(
        np.float64
    )[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        calibrated = calibrated64.astype(np.float32)
    if not np.isfinite(calibrated).all():
        raise MaxEnvelopeCalibrationError(
            "max-envelope inference produced non-finite scores"
        )
    row_ids = np.arange(len(values), dtype=np.int64)
    # Anchor the selected biased-new winner exactly to the original float32
    # envelope.  Guard against a cast-induced tie stealing the intended
    # winner by moving only such non-winners one float below the envelope.
    for row, winner in zip(row_ids.tolist(), shifted_winner.tolist()):
        calibrated[row, winner] = raw_max[row]
        collision = calibrated[row] >= raw_max[row]
        collision[winner] = False
        if bool(np.any(collision)):
            calibrated[row, collision] = np.nextafter(
                raw_max[row], np.float32(-np.inf), dtype=np.float32
            )
    adjusted = values.copy()
    adjusted[:, old_count:] = calibrated
    if not np.array_equal(adjusted[:, :old_count], values[:, :old_count]):
        raise MaxEnvelopeCalibrationError(
            "max-envelope calibration mutated old score columns"
        )
    if not np.array_equal(np.max(calibrated, axis=1), raw_max):
        raise MaxEnvelopeCalibrationError(
            "max-envelope calibration changed the new-score envelope"
        )
    raw_prediction = np.argmax(values, axis=1)
    adjusted_prediction = np.argmax(adjusted, axis=1)
    raw_old = raw_prediction < old_count
    adjusted_old = adjusted_prediction < old_count
    if not np.array_equal(raw_old, adjusted_old):
        raise MaxEnvelopeCalibrationError(
            "max-envelope calibration changed the old/new group winner"
        )
    if not np.array_equal(
        raw_prediction[raw_old], adjusted_prediction[adjusted_old]
    ):
        raise MaxEnvelopeCalibrationError(
            "max-envelope calibration changed an old-prefix prediction"
        )
    return adjusted


def _rank_metrics(
    metrics: dict[str, Any],
    objective: str,
    biases: np.ndarray,
) -> tuple[float, ...]:
    floor = float(metrics["new_class_floor_accuracy"])
    worst = float(metrics["new_worst_20pct_accuracy"])
    overall = float(metrics["new_overall_accuracy"])
    if objective == "floor_first":
        primary = (floor, worst, overall)
    elif objective == "balance_first":
        primary = (worst, floor, overall)
    else:
        primary = (overall, worst, floor)
    vector = np.asarray(biases, dtype=np.float64)
    return primary + (
        -float(np.max(np.abs(vector))),
        -float(np.dot(vector, vector)),
    )


def _center_biases(value: np.ndarray) -> np.ndarray:
    vector64 = np.asarray(value, dtype=np.float64)
    centered = vector64 - float(np.mean(vector64))
    result = np.asarray(centered, dtype=np.float32)
    if not np.isfinite(result).all():
        raise MaxEnvelopeCalibrationError(
            "max-envelope coordinate search produced non-finite bias"
        )
    return result


def _candidate_coordinate_values(
    new_scores: np.ndarray, biases: np.ndarray, coordinate: int
) -> list[np.float32]:
    shifted = new_scores.astype(np.float64) + biases.astype(np.float64)
    other = np.delete(shifted, coordinate, axis=1)
    thresholds = np.max(other, axis=1) - new_scores[:, coordinate].astype(
        np.float64
    )
    values: set[float] = {float(biases[coordinate])}
    for threshold in np.unique(thresholds):
        base = np.float32(threshold)
        values.add(float(base))
        values.add(
            float(np.nextafter(base, np.float32(-np.inf), dtype=np.float32))
        )
        values.add(
            float(np.nextafter(base, np.float32(np.inf), dtype=np.float32))
        )
    return [np.float32(value) for value in sorted(values)]


def _fit_coordinate_biases(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_count: int,
    config: MaxEnvelopeCalibrationConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    new_classes = classes[old_count:]
    new_mask = np.isin(labels, np.asarray(new_classes))
    breakpoint_scores = scores[new_mask]
    biases = np.zeros(len(new_classes), dtype=np.float32)
    trace: list[dict[str, Any]] = []
    for pass_index in range(int(config.coordinate_passes)):
        pass_changed = False
        for coordinate, class_name in enumerate(new_classes):
            current_adjusted = _apply_biases(scores, old_count, biases)
            current_metrics = _metrics(
                current_adjusted, labels, classes, old_count
            )
            best_biases = biases.copy()
            best_metrics = current_metrics
            best_rank = _rank_metrics(current_metrics, config.objective, biases)
            trials = 0
            for candidate in _candidate_coordinate_values(
                breakpoint_scores[:, old_count:], biases, coordinate
            ):
                trial = biases.copy()
                trial[coordinate] = candidate
                trial = _center_biases(trial)
                adjusted = _apply_biases(scores, old_count, trial)
                metrics = _metrics(adjusted, labels, classes, old_count)
                rank = _rank_metrics(metrics, config.objective, trial)
                trials += 1
                if rank > best_rank:
                    best_biases = trial
                    best_metrics = metrics
                    best_rank = rank
            changed = not np.array_equal(best_biases, biases)
            biases = best_biases
            pass_changed = pass_changed or changed
            trace.append(
                {
                    "pass": pass_index,
                    "new_class": class_name,
                    "candidate_count": trials,
                    "changed": changed,
                    "selected_bias": float(biases[coordinate]),
                    "selected_rank": [float(value) for value in best_rank],
                    "selected_metrics": best_metrics,
                }
            )
        if not pass_changed:
            break
    return _center_biases(biases), trace


def _evidence(
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_count: int,
) -> dict[str, Any]:
    raw = _metrics(raw_scores, labels, classes, old_count)
    adjusted = _metrics(adjusted_scores, labels, classes, old_count)
    raw_predictions = np.argmax(raw_scores, axis=1)
    adjusted_predictions = np.argmax(adjusted_scores, axis=1)
    raw_old = raw_predictions < old_count
    adjusted_old = adjusted_predictions < old_count
    old_prefix_unchanged = np.array_equal(
        raw_scores[:, :old_count], adjusted_scores[:, :old_count]
    )
    envelope_unchanged = np.array_equal(
        np.max(raw_scores[:, old_count:], axis=1),
        np.max(adjusted_scores[:, old_count:], axis=1),
    )
    group_winner_unchanged = np.array_equal(raw_old, adjusted_old)
    old_identity_unchanged = np.array_equal(
        raw_predictions[raw_old], adjusted_predictions[adjusted_old]
    )
    new_nondegrade = bool(
        adjusted["new_class_floor_accuracy"] + _TOL
        >= raw["new_class_floor_accuracy"]
        and adjusted["new_worst_20pct_accuracy"] + _TOL
        >= raw["new_worst_20pct_accuracy"]
        and adjusted["new_overall_accuracy"] + _TOL
        >= raw["new_overall_accuracy"]
    )
    per_new_class_nondegrade = bool(
        all(
            adjusted["per_new_class_accuracy"][class_name] + _TOL
            >= raw_accuracy
            for class_name, raw_accuracy in raw[
                "per_new_class_accuracy"
            ].items()
        )
    )
    strict_gain = bool(
        adjusted["new_class_floor_accuracy"]
        > raw["new_class_floor_accuracy"] + _TOL
        or adjusted["new_worst_20pct_accuracy"]
        > raw["new_worst_20pct_accuracy"] + _TOL
        or adjusted["new_overall_accuracy"]
        > raw["new_overall_accuracy"] + _TOL
    )
    safe = bool(
        old_prefix_unchanged
        and envelope_unchanged
        and group_winner_unchanged
        and old_identity_unchanged
        and new_nondegrade
        and per_new_class_nondegrade
    )
    return {
        "raw": raw,
        "adjusted": adjusted,
        "old_prefix_bitwise_unchanged": old_prefix_unchanged,
        "max_new_envelope_bitwise_unchanged": envelope_unchanged,
        "old_new_group_winner_unchanged": group_winner_unchanged,
        "old_prefix_prediction_identity_unchanged": old_identity_unchanged,
        "new_floor_worst20_and_overall_non_degradation": new_nondegrade,
        "per_new_class_accuracy_non_degradation": per_new_class_nondegrade,
        "strict_new_improvement": strict_gain,
        "safe": safe,
        "raw_confusion": audit_envelope_confusions(
            raw_scores, labels, classes, old_count
        ),
        "adjusted_confusion": audit_envelope_confusions(
            adjusted_scores, labels, classes, old_count
        ),
    }


def _disabled_state(
    *,
    classes: tuple[str, ...],
    old_count: int,
    k_shot: int,
    config: MaxEnvelopeCalibrationConfig,
    audit: dict[str, Any],
) -> MaxEnvelopeCalibrationState:
    return MaxEnvelopeCalibrationState(
        schema=SCHEMA,
        registered_classes=classes,
        old_class_count=old_count,
        k_shot=k_shot,
        enabled=False,
        biases=np.zeros(len(classes) - old_count, dtype=np.float32),
        audit_json=_canonical_json(audit),
        config=config,
    )


def fit_max_envelope_calibration(
    support_scores: np.ndarray,
    support_labels: Sequence[str],
    support_shot_ranks: Sequence[int],
    registered_classes: Sequence[str],
    old_class_count: int,
    *,
    config: MaxEnvelopeCalibrationConfig | None = None,
) -> MaxEnvelopeCalibrationState:
    """Fit a support-only five-fold max-envelope calibration."""

    actual_config = config or MaxEnvelopeCalibrationConfig()
    actual_config.validate()
    scores, labels, ranks, classes, old_count, k_shot = _validate_support(
        support_scores,
        support_labels,
        support_shot_ranks,
        registered_classes,
        old_class_count,
    )
    raw_confusion = audit_envelope_confusions(
        scores, labels, classes, old_count
    )
    if k_shot == 1:
        audit = {
            "schema": "cvs.phase2.d30_max_envelope_calibration_fit_audit.v1",
            "k_shot": 1,
            "enabled": False,
            "selection_policy": "k1_disabled_exact_passthrough",
            "oof_fold_count": 0,
            "raw_confusion": raw_confusion,
            "query_rows_used": 0,
            "query_batch_statistics_used": False,
        }
        return _disabled_state(
            classes=classes,
            old_count=old_count,
            k_shot=k_shot,
            config=actual_config,
            audit=audit,
        )
    if 2 <= k_shot < OOF_FOLD_COUNT:
        raise MaxEnvelopeCalibrationError(
            "max-envelope calibration requires K>=5; only K1 may bypass"
        )

    oof_adjusted = scores.copy()
    fold_evidence: list[dict[str, Any]] = []
    for fold_index in range(OOF_FOLD_COUNT):
        held = ranks % OOF_FOLD_COUNT == fold_index
        train = ~held
        if not bool(np.any(held)) or not bool(np.any(train)):
            raise MaxEnvelopeCalibrationError(
                "max-envelope five-fold shot-rank partition drift"
            )
        train_biases, coordinate_trace = _fit_coordinate_biases(
            scores[train], labels[train], classes, old_count, actual_config
        )
        held_adjusted = _apply_biases(scores[held], old_count, train_biases)
        oof_adjusted[held] = held_adjusted
        fold_evidence.append(
            {
                "fold": fold_index,
                "held_shot_ranks": sorted(set(ranks[held].tolist())),
                "train_sample_count": int(np.sum(train)),
                "held_sample_count": int(np.sum(held)),
                "biases": [float(value) for value in train_biases],
                "coordinate_trace": coordinate_trace,
                "held_evidence": _evidence(
                    scores[held],
                    held_adjusted,
                    labels[held],
                    classes,
                    old_count,
                ),
            }
        )
    oof_evidence = _evidence(
        scores, oof_adjusted, labels, classes, old_count
    )
    oof_pass = bool(
        oof_evidence["safe"] and oof_evidence["strict_new_improvement"]
    )
    if not oof_pass:
        audit = {
            "schema": "cvs.phase2.d30_max_envelope_calibration_fit_audit.v1",
            "k_shot": k_shot,
            "enabled": False,
            "selection_policy": "atomic_passthrough_oof_gate_failed",
            "fallback_reason": (
                "oof_safety_failed"
                if not oof_evidence["safe"]
                else "oof_no_strict_new_gain"
            ),
            "objective": actual_config.objective,
            "coordinate_passes": actual_config.coordinate_passes,
            "oof_fold_count": OOF_FOLD_COUNT,
            "fold_fit_evidence": fold_evidence,
            "oof_evidence": oof_evidence,
            "raw_confusion": raw_confusion,
            "query_rows_used": 0,
            "query_batch_statistics_used": False,
        }
        return _disabled_state(
            classes=classes,
            old_count=old_count,
            k_shot=k_shot,
            config=actual_config,
            audit=audit,
        )

    full_biases, full_trace = _fit_coordinate_biases(
        scores, labels, classes, old_count, actual_config
    )
    full_adjusted = _apply_biases(scores, old_count, full_biases)
    full_evidence = _evidence(
        scores, full_adjusted, labels, classes, old_count
    )
    full_pass = bool(
        full_evidence["safe"] and full_evidence["strict_new_improvement"]
    )
    if not full_pass:
        audit = {
            "schema": "cvs.phase2.d30_max_envelope_calibration_fit_audit.v1",
            "k_shot": k_shot,
            "enabled": False,
            "selection_policy": "atomic_passthrough_full_refit_gate_failed",
            "fallback_reason": (
                "full_refit_safety_failed"
                if not full_evidence["safe"]
                else "full_refit_no_strict_new_gain"
            ),
            "objective": actual_config.objective,
            "coordinate_passes": actual_config.coordinate_passes,
            "oof_fold_count": OOF_FOLD_COUNT,
            "fold_fit_evidence": fold_evidence,
            "oof_evidence": oof_evidence,
            "full_coordinate_trace": full_trace,
            "full_support_evidence": full_evidence,
            "raw_confusion": raw_confusion,
            "query_rows_used": 0,
            "query_batch_statistics_used": False,
        }
        return _disabled_state(
            classes=classes,
            old_count=old_count,
            k_shot=k_shot,
            config=actual_config,
            audit=audit,
        )

    audit = {
        "schema": "cvs.phase2.d30_max_envelope_calibration_fit_audit.v1",
        "k_shot": k_shot,
        "enabled": True,
        "selection_policy": "fivefold_oof_safe_strict_gain_then_full_refit",
        "fallback_reason": None,
        "objective": actual_config.objective,
        "coordinate_passes": actual_config.coordinate_passes,
        "oof_fold_count": OOF_FOLD_COUNT,
        "fold_fit_evidence": fold_evidence,
        "oof_evidence": oof_evidence,
        "full_coordinate_trace": full_trace,
        "full_support_evidence": full_evidence,
        "raw_confusion": raw_confusion,
        "query_rows_used": 0,
        "query_batch_statistics_used": False,
    }
    return MaxEnvelopeCalibrationState(
        schema=SCHEMA,
        registered_classes=classes,
        old_class_count=old_count,
        k_shot=k_shot,
        enabled=True,
        biases=full_biases,
        audit_json=_canonical_json(audit),
        config=actual_config,
    )


def apply_max_envelope_calibration(
    state: MaxEnvelopeCalibrationState, raw_scores: np.ndarray
) -> np.ndarray:
    """Apply one row-local calibration and preserve the old/new envelope."""

    scores = _validate_scores(
        raw_scores,
        class_count=len(state.registered_classes),
        name="max-envelope inference scores",
    )
    if not state.enabled:
        return _readonly(scores, np.float32)
    adjusted = _apply_biases(scores, state.old_class_count, state.biases)
    return _readonly(adjusted, np.float32)


def predict_with_max_envelope_calibration(
    state: MaxEnvelopeCalibrationState, raw_scores: np.ndarray
) -> np.ndarray:
    adjusted = apply_max_envelope_calibration(state, raw_scores)
    return np.asarray(state.registered_classes)[np.argmax(adjusted, axis=1)]


__all__ = [
    "MaxEnvelopeCalibrationConfig",
    "MaxEnvelopeCalibrationError",
    "MaxEnvelopeCalibrationState",
    "OBJECTIVES",
    "SCHEMA",
    "apply_max_envelope_calibration",
    "audit_envelope_confusions",
    "fit_max_envelope_calibration",
    "predict_with_max_envelope_calibration",
]
