"""D9 floor-focused sparse fusion over fixed received-IQ operators.

Every physical support/query row has exactly one sealed LEO_weak received IQ.
The three D7a operators are deterministic post-reception computations from
that same IQ.  D9 selects, from physical-support deletion evidence only, at
most two operators and fixed convex weights per class.  Query inference
extracts each deduplicated used operator once and scores every registered
class independently.
"""

from __future__ import annotations

import inspect
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_class_conditional_iq_head import (
    BASE,
    OPERATORS,
    OperatorCalibration,
    extract_operator_features,
)


EPS = 1.0e-8
MAX_COMPONENTS_PER_CLASS = 2
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
MAX_TRAINABLE_PARAMETERS = 50_000
DEFAULT_FLOOR_PRIORITY_CLASSES = ("20-19", "1-18")


class FloorSparseOperatorFusionError(ValueError):
    """Raised when D9 support, registration, or inference invariants drift."""


@dataclass(frozen=True)
class SamplewiseSealedFeatureExtractor:
    """Vectorized extractor with an immutable per-sample independence seal."""

    callback: Callable[[np.ndarray], np.ndarray]
    extractor_id: str
    samplewise_contract_sha256: str
    validation_rows_sha256: str
    validation_max_abs_error: float
    batch_independent: bool = True
    query_updates: int = 0

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.callback(rows)


def seal_samplewise_feature_extractor(
    callback: Callable[[np.ndarray], np.ndarray],
    *,
    extractor_id: str,
    validation_rows: np.ndarray,
) -> SamplewiseSealedFeatureExtractor:
    """Create the only accepted query extractor interface."""

    identifier = str(extractor_id)
    if not identifier:
        raise FloorSparseOperatorFusionError(
            "samplewise extractor ID is required"
        )
    rows = np.asarray(validation_rows)
    if rows.ndim < 2 or len(rows) < 2 or not np.isfinite(rows).all():
        raise FloorSparseOperatorFusionError(
            "samplewise seal requires at least two finite validation rows"
        )
    batch = np.asarray(callback(rows), dtype=np.float32)
    singles = np.concatenate(
        [
            np.asarray(callback(rows[index : index + 1]), dtype=np.float32)
            for index in range(len(rows))
        ],
        axis=0,
    )
    if (
        batch.shape != singles.shape
        or not np.isfinite(batch).all()
        or not np.isfinite(singles).all()
    ):
        raise FloorSparseOperatorFusionError(
            "samplewise extractor validation output drift"
        )
    max_error = float(np.max(np.abs(batch - singles)))
    if max_error > 1.0e-6:
        raise FloorSparseOperatorFusionError(
            "batch-dependent query feature extractor is forbidden"
        )
    validation_sha = hashlib.sha256(
        np.ascontiguousarray(rows).tobytes()
    ).hexdigest()
    digest = _samplewise_seal_digest(identifier, validation_sha)
    return SamplewiseSealedFeatureExtractor(
        callback=callback,
        extractor_id=identifier,
        samplewise_contract_sha256=digest,
        validation_rows_sha256=validation_sha,
        validation_max_abs_error=max_error,
    )


def _samplewise_seal_digest(
    extractor_id: str, validation_rows_sha256: str
) -> str:
    return hashlib.sha256(
        (
            "cvs.phase2.samplewise_feature_extractor.v1|"
            + str(extractor_id)
            + "|validation_rows_sha256="
            + str(validation_rows_sha256)
            + "|batch_independent=true|query_updates=0"
        ).encode("utf-8")
    ).hexdigest()


def build_operator_feature_provenance(
    parent_received_iq_sha256: Sequence[str],
    *,
    view_seed: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Bind every support feature row to its parent IQ/operator/view seed."""

    hashes = tuple(
        str(value).lower() for value in parent_received_iq_sha256
    )
    return {
        operator: tuple(
            {
                "parent_received_iq_sha256": digest,
                "operator_id": operator,
                "view_seed": int(view_seed),
            }
            for digest in hashes
        )
        for operator in OPERATORS
    }


@dataclass(frozen=True)
class FusionCandidate:
    candidate_id: str
    operator_indices: tuple[int, int]
    weights: tuple[float, float]

    @property
    def active_count(self) -> int:
        return sum(value > 0.0 for value in self.weights)


def _fixed_candidates() -> tuple[FusionCandidate, ...]:
    candidates = [
        FusionCandidate(
            candidate_id=f"single_{operator}",
            operator_indices=(index, -1),
            weights=(1.0, 0.0),
        )
        for index, operator in enumerate(OPERATORS)
    ]
    for left in range(len(OPERATORS)):
        for right in range(left + 1, len(OPERATORS)):
            for left_weight in (0.75, 0.50, 0.25):
                candidates.append(
                    FusionCandidate(
                        candidate_id=(
                            f"pair_{OPERATORS[left]}_{left_weight:.2f}_"
                            f"{OPERATORS[right]}_{1.0-left_weight:.2f}"
                        ),
                        operator_indices=(left, right),
                        weights=(left_weight, 1.0 - left_weight),
                    )
                )
    return tuple(candidates)


FIXED_CANDIDATES = _fixed_candidates()
BASE_CANDIDATE = FIXED_CANDIDATES[0]


@dataclass(frozen=True)
class FloorSparseOperatorFusionState:
    schema: str
    classes: tuple[str, ...]
    operator_indices: np.ndarray
    weights: np.ndarray
    prototypes: np.ndarray
    calibrations: tuple[OperatorCalibration, ...]
    feature_dim: int
    used_operators: tuple[str, ...]
    old_class_count: int
    registration_generation: int
    current_k: int
    selection_lock_k: int
    selection_lock_sha256: str
    support_lineage: tuple[tuple[str, str, str], ...]
    base_persistent_state_bytes: int
    base_head_macs_per_query: int
    support_audit: Mapping[str, Any]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def persistent_state_bytes(self) -> int:
        incremental = int(
            self.operator_indices.nbytes
            + self.weights.nbytes
            + self.prototypes.nbytes
            + 8 * len(self.calibrations)
            + sum(len(value.encode("utf-8")) for value in self.classes)
            + sum(
                len(label.encode("utf-8"))
                + len(physical_id.encode("utf-8"))
                + len(parent_hash.encode("utf-8"))
                for label, physical_id, parent_hash in self.support_lineage
            )
        )
        return self.base_persistent_state_bytes + incremental

    @property
    def incremental_state_bytes(self) -> int:
        return self.persistent_state_bytes - self.base_persistent_state_bytes

    @property
    def active_component_count(self) -> int:
        return int(np.sum(self.weights > 0.0))

    @property
    def estimated_head_macs_per_query(self) -> int:
        # Each active component has one D-dimensional dot product plus
        # calibration, weight, and accumulation.
        return self.base_head_macs_per_query + int(
            self.active_component_count * (self.feature_dim + 4)
        )

    @property
    def incremental_head_macs_per_query(self) -> int:
        return (
            self.estimated_head_macs_per_query
            - self.base_head_macs_per_query
        )

    def calibration_for(self, operator_index: int) -> OperatorCalibration:
        operator = OPERATORS[int(operator_index)]
        for calibration in self.calibrations:
            if calibration.operator_id == operator:
                return calibration
        raise FloorSparseOperatorFusionError("D9 calibration missing")

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d9_resource.v1",
            "candidate": "d9_floor_sparse_operator_fusion",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_SUPPORT_SELECTION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "trainable_parameter_limit": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_pass": True,
            "persistent_state_bytes": self.persistent_state_bytes,
            "base_persistent_state_bytes": (
                self.base_persistent_state_bytes
            ),
            "d9_incremental_state_bytes": self.incremental_state_bytes,
            "combined_persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "estimated_head_macs_per_query": (
                self.estimated_head_macs_per_query
            ),
            "base_head_macs_per_query": self.base_head_macs_per_query,
            "d9_incremental_head_macs_per_query": (
                self.incremental_head_macs_per_query
            ),
            "combined_head_macs_per_query": (
                self.estimated_head_macs_per_query
            ),
            "backbone_forwards_per_query": len(self.used_operators),
            "used_operator_count": len(self.used_operators),
            "used_operators": list(self.used_operators),
            "maximum_fixed_received_iq_views": len(OPERATORS),
            "maximum_components_per_class": MAX_COMPONENTS_PER_CLASS,
            "views_count_as_additional_k": False,
            "additional_physical_samples_from_views": 0,
            "additional_leo_channel_states_generated": 0,
            "fixed_received_iq_only": True,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "source_sample_access": False,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "query_decision_policy": "per_sample_all_registered_classes",
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "old_state_locked_after_registration": (
                self.registration_generation > 0
            ),
            "current_k": self.current_k,
            "selection_lock_k": self.selection_lock_k,
            "selection_lock_sha256": self.selection_lock_sha256,
            "samplewise_query_extractor_required": True,
        }


@dataclass(frozen=True)
class FloorSparseFusionPrediction:
    labels: tuple[str, ...]
    scores: np.ndarray
    operators_computed: tuple[str, ...]


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), EPS
    )


def _prototype(rows: np.ndarray) -> np.ndarray:
    return _normalize(_normalize(rows).mean(axis=0, keepdims=True))[0]


def _operator_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    return np.stack(
        [_prototype(features[labels == label]) for label in classes],
        axis=0,
    ).astype(np.float32)


def _calibration(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    prototypes: np.ndarray,
    operator: str,
) -> OperatorCalibration:
    scores = _normalize(features) @ prototypes.T
    class_index = {label: index for index, label in enumerate(classes)}
    off = np.asarray(
        [
            score
            for row, label in zip(scores, labels.tolist())
            for index, score in enumerate(row.tolist())
            if index != class_index[label]
        ],
        dtype=np.float64,
    )
    return OperatorCalibration(
        operator_id=operator,
        center=float(np.mean(off)),
        scale=float(max(np.std(off), 0.05)),
    )


def _calibrated_scores(
    features: np.ndarray,
    prototypes: np.ndarray,
    calibration: OperatorCalibration,
) -> np.ndarray:
    return (
        _normalize(features) @ prototypes.T - calibration.center
    ) / calibration.scale


def _validate_support(
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    int,
]:
    if set(features_by_operator) != set(OPERATORS):
        raise FloorSparseOperatorFusionError("D9 exact operator set drift")
    if set(operator_feature_provenance) != set(OPERATORS):
        raise FloorSparseOperatorFusionError(
            "D9 exact operator provenance set drift"
        )
    labels = np.asarray(tuple(str(value) for value in support_labels))
    ids = tuple(str(value) for value in physical_sample_ids)
    hashes = tuple(
        str(value).lower() for value in parent_received_iq_sha256
    )
    row_count = len(labels)
    prepared: dict[str, np.ndarray] = {}
    feature_dim: int | None = None
    for operator in OPERATORS:
        rows = np.asarray(features_by_operator[operator], dtype=np.float32)
        provenance = tuple(operator_feature_provenance[operator])
        if (
            rows.ndim != 2
            or len(rows) != row_count
            or len(provenance) != row_count
            or not np.isfinite(rows).all()
        ):
            raise FloorSparseOperatorFusionError(
                "D9 support feature drift"
            )
        if feature_dim is None:
            feature_dim = rows.shape[1]
        elif rows.shape[1] != feature_dim:
            raise FloorSparseOperatorFusionError(
                "D9 feature dimension drift"
            )
        prepared[operator] = np.ascontiguousarray(rows)
        for row_index, record in enumerate(provenance):
            if (
                set(record)
                != {
                    "parent_received_iq_sha256",
                    "operator_id",
                    "view_seed",
                }
                or str(record["parent_received_iq_sha256"]).lower()
                != hashes[row_index]
                or str(record["operator_id"]) != operator
                or not isinstance(record["view_seed"], (int, np.integer))
            ):
                raise FloorSparseOperatorFusionError(
                    "D9 per-sample operator provenance drift"
                )
    classes, counts = np.unique(labels, return_counts=True)
    if (
        row_count < 1
        or len(classes) < 2
        or np.any(counts < 1)
        or len(ids) != row_count
        or len(hashes) != row_count
        or len(set(ids)) != row_count
        or len(set(hashes)) != row_count
        or any(not value for value in ids)
        or any(
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        )
    ):
        raise FloorSparseOperatorFusionError(
            "D9 single-observation support lineage drift"
        )
    if len(set(counts.tolist())) != 1:
        raise FloorSparseOperatorFusionError(
            "D9 requires one uniform physical K across classes"
        )
    return prepared, labels, ids, hashes, int(counts[0])


def _lineage(
    labels: np.ndarray,
    ids: Sequence[str],
    hashes: Sequence[str],
    classes: Sequence[str],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (label, ids[index], hashes[index])
        for label in classes
        for index in np.flatnonzero(labels == label).tolist()
    )


def _selection_lock_sha256(
    classes: Sequence[str],
    operator_indices: np.ndarray,
    weights: np.ndarray,
    calibrations: Sequence[OperatorCalibration],
) -> str:
    payload = "|".join(
        [
            ",".join(classes),
            operator_indices.tobytes().hex(),
            weights.tobytes().hex(),
            ";".join(
                f"{value.operator_id}:{value.center:.17g}:{value.scale:.17g}"
                for value in calibrations
            ),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_base_resources(
    base_resource_audit: Mapping[str, Any],
) -> tuple[int, int]:
    required = {"persistent_state_bytes", "estimated_head_macs_per_query"}
    if not required.issubset(base_resource_audit):
        raise FloorSparseOperatorFusionError(
            "D9 base resource audit incomplete"
        )
    persistent = int(base_resource_audit["persistent_state_bytes"])
    macs = int(base_resource_audit["estimated_head_macs_per_query"])
    if persistent < 0 or macs < 0:
        raise FloorSparseOperatorFusionError(
            "D9 base resource audit invalid"
        )
    return persistent, macs


def _folds(
    labels: np.ndarray,
    classes: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    by_class = [
        np.flatnonzero(labels == label).tolist() for label in classes
    ]
    width = 2 if min(map(len, by_class)) >= 4 else 1
    if min(map(len, by_class)) <= width:
        raise FloorSparseOperatorFusionError(
            "D9 support deletion must retain each class"
        )
    folds = []
    for offset in range(0, max(map(len, by_class)), width):
        held = [
            index
            for values in by_class
            for index in values[offset : offset + width]
        ]
        if held:
            folds.append(tuple(sorted(held)))
    covered = sorted(index for fold in folds for index in fold)
    expected = sorted(index for values in by_class for index in values)
    if covered != expected:
        raise FloorSparseOperatorFusionError(
            "D9 physical support deletion coverage drift"
        )
    return tuple(folds)


def _candidate_column(
    operator_scores: Mapping[str, np.ndarray],
    candidate: FusionCandidate,
    class_index: int,
) -> np.ndarray:
    column = np.zeros(
        len(next(iter(operator_scores.values()))), dtype=np.float32
    )
    for operator_index, weight in zip(
        candidate.operator_indices, candidate.weights
    ):
        if operator_index >= 0 and weight > 0.0:
            column += float(weight) * operator_scores[
                OPERATORS[operator_index]
            ][:, class_index]
    return column


def _metrics(
    scores: np.ndarray,
    truth: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    prediction = np.argmax(scores, axis=1)
    correct = prediction == truth
    per_class = {
        label: float(np.mean(correct[truth == index]))
        for index, label in enumerate(classes)
        if np.any(truth == index)
    }
    margins = np.asarray(
        [
            scores[row, target]
            - np.max(np.delete(scores[row], target))
            for row, target in enumerate(truth.tolist())
        ],
        dtype=np.float64,
    )
    return {
        "overall_accuracy": float(np.mean(correct)),
        "min_class_accuracy": min(per_class.values()),
        "per_class_accuracy": per_class,
        "mean_true_margin": float(np.mean(margins)),
        "worst_true_margin": float(np.min(margins)),
    }


def _before_records(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    classes: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold in _folds(labels, classes):
        held = np.asarray(fold, dtype=np.int64)
        keep = np.ones(len(labels), dtype=bool)
        keep[held] = False
        operator_scores: dict[str, np.ndarray] = {}
        for operator in OPERATORS:
            prototypes = _operator_prototypes(
                features[operator][keep], labels[keep], classes
            )
            calibration = _calibration(
                features[operator][keep],
                labels[keep],
                classes,
                prototypes,
                operator,
            )
            operator_scores[operator] = _calibrated_scores(
                features[operator][held], prototypes, calibration
            )
        records.append(
            {
                "operator_scores": operator_scores,
                "truth": np.asarray(
                    [classes.index(value) for value in labels[held]],
                    dtype=np.int64,
                ),
            }
        )
    return records


def _combined_base(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(
            [record["operator_scores"][BASE] for record in records],
            axis=0,
        ),
        np.concatenate([record["truth"] for record in records], axis=0),
    )


def _candidate_evidence(
    records: Sequence[Mapping[str, Any]],
    classes: tuple[str, ...],
    class_index: int,
    candidate: FusionCandidate,
) -> dict[str, Any]:
    scores_parts = []
    truth_parts = []
    fold_accuracies = []
    target_margins = []
    for record in records:
        mixed = record["operator_scores"][BASE].copy()
        mixed[:, class_index] = _candidate_column(
            record["operator_scores"], candidate, class_index
        )
        truth = record["truth"]
        scores_parts.append(mixed)
        truth_parts.append(truth)
        target_mask = truth == class_index
        if np.any(target_mask):
            fold_accuracies.append(
                float(
                    np.mean(
                        np.argmax(mixed[target_mask], axis=1)
                        == class_index
                    )
                )
            )
            for row in mixed[target_mask]:
                target_margins.append(
                    float(
                        row[class_index]
                        - np.max(np.delete(row, class_index))
                    )
                )
    scores = np.concatenate(scores_parts, axis=0)
    truth = np.concatenate(truth_parts, axis=0)
    metrics = _metrics(scores, truth, classes)
    return {
        "candidate_id": candidate.candidate_id,
        "operator_handles": [
            OPERATORS[index]
            for index, weight in zip(
                candidate.operator_indices, candidate.weights
            )
            if index >= 0 and weight > 0.0
        ],
        "weights": [
            weight for weight in candidate.weights if weight > 0.0
        ],
        "active_operator_count": candidate.active_count,
        "target_class_accuracy": metrics["per_class_accuracy"][
            classes[class_index]
        ],
        "worst_fold_target_accuracy": min(fold_accuracies),
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_target_margin": float(np.mean(target_margins)),
        "worst_target_margin": min(target_margins),
    }


def _select_before(
    records: Sequence[Mapping[str, Any]],
    classes: tuple[str, ...],
    floor_priority_classes: Sequence[str],
) -> tuple[list[FusionCandidate], dict[str, Any]]:
    base_scores, truth = _combined_base(records)
    baseline = _metrics(base_scores, truth, classes)
    selected: list[FusionCandidate] = []
    per_class_trace = []
    for class_index, label in enumerate(classes):
        evidence = [
            _candidate_evidence(
                records, classes, class_index, candidate
            )
            for candidate in FIXED_CANDIDATES
        ]
        for row in evidence:
            row["target_non_degradation_pass"] = bool(
                row["target_class_accuracy"]
                >= baseline["per_class_accuracy"][label]
            )
            row["overall_non_degradation_pass"] = bool(
                row["overall_accuracy"] >= baseline["overall_accuracy"]
            )
            row["eligible"] = bool(
                row["target_non_degradation_pass"]
                and row["overall_non_degradation_pass"]
            )
        eligible = [
            (index, row)
            for index, row in enumerate(evidence)
            if row["eligible"]
        ]
        chosen_index, chosen_row = max(
            eligible,
            key=lambda item: (
                item[1]["target_class_accuracy"],
                item[1]["worst_fold_target_accuracy"],
                item[1]["overall_accuracy"],
                item[1]["worst_target_margin"],
                -item[1]["active_operator_count"],
                -item[0],
            ),
        )
        selected.append(FIXED_CANDIDATES[chosen_index])
        per_class_trace.append(
            {
                "class_handle": label,
                "floor_priority": label in set(floor_priority_classes),
                "selected_candidate_id": chosen_row["candidate_id"],
                "candidate_evidence": evidence,
            }
        )
    combined = base_scores.copy()
    offset = 0
    for record in records:
        rows = len(record["truth"])
        for class_index, candidate in enumerate(selected):
            combined[offset : offset + rows, class_index] = (
                _candidate_column(
                    record["operator_scores"], candidate, class_index
                )
            )
        offset += rows
    combined_metrics = _metrics(combined, truth, classes)
    global_pass = bool(
        combined_metrics["overall_accuracy"]
        >= baseline["overall_accuracy"]
        and combined_metrics["min_class_accuracy"]
        >= baseline["min_class_accuracy"]
        and all(
            combined_metrics["per_class_accuracy"][label]
            >= baseline["per_class_accuracy"][label]
            for label in classes
        )
    )
    assignment_before_fallback = [
        candidate.candidate_id for candidate in selected
    ]
    if not global_pass:
        selected = [BASE_CANDIDATE] * len(classes)
        combined_metrics = baseline
    return selected, {
        "schema": "cvs.phase2.d9_support_selection.v1",
        "selection_scope": "registered_physical_support_deletion_only",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "fixed_candidate_ids": [
            candidate.candidate_id for candidate in FIXED_CANDIDATES
        ],
        "floor_priority_class_handles": list(floor_priority_classes),
        "baseline": baseline,
        "per_class_selection": per_class_trace,
        "assignment_before_fallback": assignment_before_fallback,
        "combined_before_fallback": _metrics(combined, truth, classes),
        "global_overall_and_floor_non_degradation_pass": global_pass,
        "global_every_class_non_degradation_pass": global_pass,
        "fallback_to_all_base": not global_pass,
        "combined_final": combined_metrics,
    }


def _state_tensors(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    classes: tuple[str, ...],
    selected: Sequence[FusionCandidate],
    calibrations: Sequence[OperatorCalibration],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    operator_prototypes = {
        operator: _operator_prototypes(
            features[operator], labels, classes
        )
        for operator in OPERATORS
    }
    operator_indices = np.full(
        (len(classes), MAX_COMPONENTS_PER_CLASS), -1, dtype=np.int8
    )
    weights = np.zeros(
        (len(classes), MAX_COMPONENTS_PER_CLASS), dtype=np.float32
    )
    prototypes = np.zeros(
        (
            len(classes),
            MAX_COMPONENTS_PER_CLASS,
            next(iter(features.values())).shape[1],
        ),
        dtype=np.float32,
    )
    for class_index, candidate in enumerate(selected):
        for slot, (operator_index, weight) in enumerate(
            zip(candidate.operator_indices, candidate.weights)
        ):
            if operator_index >= 0 and weight > 0.0:
                operator_indices[class_index, slot] = operator_index
                weights[class_index, slot] = weight
                prototypes[class_index, slot] = operator_prototypes[
                    OPERATORS[operator_index]
                ][class_index]
    return operator_indices, weights, prototypes


def _used_operators(
    operator_indices: np.ndarray, weights: np.ndarray
) -> tuple[str, ...]:
    active = {
        int(operator_indices[row, slot])
        for row in range(len(operator_indices))
        for slot in range(MAX_COMPONENTS_PER_CLASS)
        if weights[row, slot] > 0.0
    }
    return tuple(
        operator
        for index, operator in enumerate(OPERATORS)
        if index in active
    )


def fit_floor_sparse_operator_fusion(
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    base_resource_audit: Mapping[str, Any],
    floor_priority_classes: Sequence[str] = (
        DEFAULT_FLOOR_PRIORITY_CLASSES
    ),
) -> FloorSparseOperatorFusionState:
    """Fit a before-registration D9 state from physical support only."""

    features, labels, ids, hashes, current_k = _validate_support(
        features_by_operator,
        operator_feature_provenance,
        support_labels,
        physical_sample_ids,
        parent_received_iq_sha256,
    )
    if current_k != 10:
        raise FloorSparseOperatorFusionError(
            "D9 operator/weight selection is locked to uniform K10"
        )
    base_state_bytes, base_head_macs = _validate_base_resources(
        base_resource_audit
    )
    classes = tuple(sorted(set(labels.tolist())))
    records = _before_records(features, labels, classes)
    selected, selection = _select_before(
        records, classes, floor_priority_classes
    )
    all_operator_prototypes = {
        operator: _operator_prototypes(
            features[operator], labels, classes
        )
        for operator in OPERATORS
    }
    calibrations = tuple(
        _calibration(
            features[operator],
            labels,
            classes,
            all_operator_prototypes[operator],
            operator,
        )
        for operator in OPERATORS
    )
    operator_indices, weights, prototypes = _state_tensors(
        features, labels, classes, selected, calibrations
    )
    selection_lock = _selection_lock_sha256(
        classes, operator_indices, weights, calibrations
    )
    support_lineage = _lineage(labels, ids, hashes, classes)
    audit = {
        "schema": "cvs.phase2.d9_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "physical_support_sample_count": len(labels),
        "physical_support_ids_unique": True,
        "parent_received_iq_hashes_unique": True,
        "fixed_received_iq_operator_set": list(OPERATORS),
        "computation_views_count_as_additional_physical_samples": False,
        "additional_leo_channel_state_generation": False,
        "clean_sample_access": False,
        "source_sample_access": False,
        "query_rows_used": 0,
        "old_state_bitwise_locked": False,
        "selection": selection,
    }
    state = FloorSparseOperatorFusionState(
        schema="cvs.phase2.d9_floor_sparse_operator_fusion.v1",
        classes=classes,
        operator_indices=_readonly(operator_indices, np.int8),
        weights=_readonly(weights, np.float32),
        prototypes=_readonly(prototypes, np.float32),
        calibrations=calibrations,
        feature_dim=prototypes.shape[2],
        used_operators=_used_operators(operator_indices, weights),
        old_class_count=len(classes),
        registration_generation=0,
        current_k=current_k,
        selection_lock_k=10,
        selection_lock_sha256=selection_lock,
        support_lineage=support_lineage,
        base_persistent_state_bytes=base_state_bytes,
        base_head_macs_per_query=base_head_macs,
        support_audit=audit,
    )
    if state.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise FloorSparseOperatorFusionError(
            "D9 persistent state cap exceeded"
        )
    return state


def _score_from_features(
    state: FloorSparseOperatorFusionState,
    features_by_operator: Mapping[str, np.ndarray],
) -> np.ndarray:
    if set(features_by_operator) != set(state.used_operators):
        raise FloorSparseOperatorFusionError(
            "D9 query feature operator set drift"
        )
    row_count = len(next(iter(features_by_operator.values())))
    normalized: dict[str, np.ndarray] = {}
    for operator in state.used_operators:
        rows = np.asarray(features_by_operator[operator], dtype=np.float32)
        if (
            rows.ndim != 2
            or len(rows) != row_count
            or rows.shape[1] != state.feature_dim
            or not np.isfinite(rows).all()
        ):
            raise FloorSparseOperatorFusionError(
                "D9 query feature drift"
            )
        normalized[operator] = _normalize(rows)
    scores = np.zeros(
        (row_count, state.class_count), dtype=np.float32
    )
    for class_index in range(state.class_count):
        for slot in range(MAX_COMPONENTS_PER_CLASS):
            weight = float(state.weights[class_index, slot])
            operator_index = int(
                state.operator_indices[class_index, slot]
            )
            if weight <= 0.0:
                continue
            calibration = state.calibration_for(operator_index)
            component = (
                normalized[OPERATORS[operator_index]]
                @ state.prototypes[class_index, slot]
                - calibration.center
            ) / calibration.scale
            scores[:, class_index] += weight * component
    return scores


def _score_old_rows_with_new_candidates(
    parent: FloorSparseOperatorFusionState,
    old_features: Mapping[str, np.ndarray],
    new_operator_prototypes: Mapping[str, np.ndarray],
    new_candidates: Sequence[FusionCandidate],
    new_classes: Sequence[str],
) -> np.ndarray:
    parent_features = {
        operator: old_features[operator]
        for operator in parent.used_operators
    }
    old_scores = _score_from_features(parent, parent_features)
    scores = np.zeros(
        (len(old_scores), parent.class_count + len(new_classes)),
        dtype=np.float32,
    )
    scores[:, : parent.class_count] = old_scores
    normalized = {
        operator: _normalize(old_features[operator])
        for operator in OPERATORS
    }
    for new_index, candidate in enumerate(new_candidates):
        column = np.zeros(len(scores), dtype=np.float32)
        for operator_index, weight in zip(
            candidate.operator_indices, candidate.weights
        ):
            if operator_index < 0 or weight <= 0.0:
                continue
            calibration = parent.calibration_for(operator_index)
            column += float(weight) * (
                normalized[OPERATORS[operator_index]]
                @ new_operator_prototypes[OPERATORS[operator_index]][
                    new_index
                ]
                - calibration.center
            ) / calibration.scale
        scores[:, parent.class_count + new_index] = column
    return scores


def _extension_records(
    parent: FloorSparseOperatorFusionState,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    new_classes: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold in _folds(labels, new_classes):
        held = np.asarray(fold, dtype=np.int64)
        keep = np.ones(len(labels), dtype=bool)
        keep[held] = False
        parent_features = {
            operator: features[operator][held]
            for operator in parent.used_operators
        }
        old_scores = _score_from_features(parent, parent_features)
        operator_scores: dict[str, np.ndarray] = {}
        for operator_index, operator in enumerate(OPERATORS):
            new_prototypes = np.stack(
                [
                    _prototype(
                        features[operator][
                            keep & (labels == label)
                        ]
                    )
                    for label in new_classes
                ],
                axis=0,
            )
            calibration = parent.calibration_for(operator_index)
            operator_scores[operator] = _calibrated_scores(
                features[operator][held],
                new_prototypes,
                calibration,
            )
        records.append(
            {
                "old_scores": old_scores,
                "operator_scores": operator_scores,
                "truth": np.asarray(
                    [
                        parent.class_count
                        + new_classes.index(value)
                        for value in labels[held]
                    ],
                    dtype=np.int64,
                ),
            }
        )
    return records


def _extension_scores(
    records: Sequence[Mapping[str, Any]],
    parent: FloorSparseOperatorFusionState,
    new_candidates: Sequence[FusionCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    truths = []
    for record in records:
        scores = np.zeros(
            (
                len(record["truth"]),
                parent.class_count + len(new_candidates),
            ),
            dtype=np.float32,
        )
        scores[:, : parent.class_count] = record["old_scores"]
        for new_index, candidate in enumerate(new_candidates):
            scores[:, parent.class_count + new_index] = _candidate_column(
                record["operator_scores"], candidate, new_index
            )
        parts.append(scores)
        truths.append(record["truth"])
    return np.concatenate(parts, axis=0), np.concatenate(truths, axis=0)


def _select_extension(
    parent: FloorSparseOperatorFusionState,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    new_classes: tuple[str, ...],
    floor_priority_classes: Sequence[str],
) -> tuple[list[FusionCandidate], dict[str, Any]]:
    records = _extension_records(parent, features, labels, new_classes)
    base_candidates = [BASE_CANDIDATE] * len(new_classes)
    baseline_new_scores, new_truth = _extension_scores(
        records, parent, base_candidates
    )
    all_classes = parent.classes + new_classes
    baseline_new = _metrics(
        baseline_new_scores, new_truth, all_classes
    )
    new_operator_prototypes = {
        operator: np.stack(
            [
                _prototype(features[operator][labels == label])
                for label in new_classes
            ],
            axis=0,
        )
        for operator in OPERATORS
    }
    old_mask = np.isin(labels, np.asarray(parent.classes))
    old_features = {
        operator: rows[old_mask] for operator, rows in features.items()
    }
    old_truth = np.asarray(
        [parent.classes.index(value) for value in labels[old_mask]],
        dtype=np.int64,
    )
    baseline_old_scores = _score_old_rows_with_new_candidates(
        parent,
        old_features,
        new_operator_prototypes,
        base_candidates,
        new_classes,
    )
    baseline_old = _metrics(
        baseline_old_scores, old_truth, all_classes
    )
    selected = list(base_candidates)
    per_class_trace = []
    for new_index, label in enumerate(new_classes):
        evidence = []
        for candidate_index, candidate in enumerate(FIXED_CANDIDATES):
            trial = list(base_candidates)
            trial[new_index] = candidate
            new_scores, _truth = _extension_scores(
                records, parent, trial
            )
            new_metrics = _metrics(new_scores, new_truth, all_classes)
            old_scores = _score_old_rows_with_new_candidates(
                parent,
                old_features,
                new_operator_prototypes,
                trial,
                new_classes,
            )
            old_metrics = _metrics(
                old_scores, old_truth, all_classes
            )
            target_accuracy = new_metrics["per_class_accuracy"][label]
            eligible = bool(
                target_accuracy
                >= baseline_new["per_class_accuracy"][label]
                and new_metrics["overall_accuracy"]
                >= baseline_new["overall_accuracy"]
                and old_metrics["overall_accuracy"]
                >= baseline_old["overall_accuracy"]
                and all(
                    old_metrics["per_class_accuracy"][old_label]
                    >= baseline_old["per_class_accuracy"][old_label]
                    for old_label in parent.classes
                )
            )
            evidence.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": candidate.candidate_id,
                    "active_operator_count": candidate.active_count,
                    "target_class_accuracy": target_accuracy,
                    "new_support": new_metrics,
                    "old_support_intrusion_guard": old_metrics,
                    "eligible": eligible,
                }
            )
        eligible_rows = [row for row in evidence if row["eligible"]]
        chosen = max(
            eligible_rows,
            key=lambda row: (
                row["target_class_accuracy"],
                row["new_support"]["overall_accuracy"],
                row["old_support_intrusion_guard"][
                    "overall_accuracy"
                ],
                row["new_support"]["worst_true_margin"],
                -row["active_operator_count"],
                -row["candidate_index"],
            ),
        )
        selected[new_index] = FIXED_CANDIDATES[
            chosen["candidate_index"]
        ]
        per_class_trace.append(
            {
                "class_handle": label,
                "floor_priority": label in set(floor_priority_classes),
                "selected_candidate_id": chosen["candidate_id"],
                "candidate_evidence": evidence,
            }
        )
    combined_new_scores, _truth = _extension_scores(
        records, parent, selected
    )
    combined_new = _metrics(
        combined_new_scores, new_truth, all_classes
    )
    combined_old_scores = _score_old_rows_with_new_candidates(
        parent,
        old_features,
        new_operator_prototypes,
        selected,
        new_classes,
    )
    combined_old = _metrics(
        combined_old_scores, old_truth, all_classes
    )
    global_pass = bool(
        combined_new["overall_accuracy"]
        >= baseline_new["overall_accuracy"]
        and combined_new["min_class_accuracy"]
        >= baseline_new["min_class_accuracy"]
        and all(
            combined_new["per_class_accuracy"][label]
            >= baseline_new["per_class_accuracy"][label]
            for label in new_classes
        )
        and combined_old["overall_accuracy"]
        >= baseline_old["overall_accuracy"]
        and all(
            combined_old["per_class_accuracy"][label]
            >= baseline_old["per_class_accuracy"][label]
            for label in parent.classes
        )
    )
    before_fallback = [
        candidate.candidate_id for candidate in selected
    ]
    if not global_pass:
        selected = base_candidates
        combined_new = baseline_new
        combined_old = baseline_old
    return selected, {
        "schema": "cvs.phase2.d9_extension_selection.v1",
        "selection_scope": "registered_physical_support_only",
        "old_state_during_selection": "bitwise_locked",
        "new_support_protocol": "physical_support_deletion",
        "old_support_protocol": "fixed_full_support_intrusion_guard",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "baseline_new": baseline_new,
        "baseline_old": baseline_old,
        "per_new_class_selection": per_class_trace,
        "assignment_before_fallback": before_fallback,
        "global_overall_floor_and_old_class_non_degradation_pass": (
            global_pass
        ),
        "fallback_new_classes_to_base": not global_pass,
        "combined_final_new": combined_new,
        "combined_final_old": combined_old,
    }


def extend_floor_sparse_operator_fusion(
    parent: FloorSparseOperatorFusionState,
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    floor_priority_classes: Sequence[str] = (
        DEFAULT_FLOOR_PRIORITY_CLASSES
    ),
) -> FloorSparseOperatorFusionState:
    """Append absent classes while bitwise locking all old D9 state."""

    if not isinstance(parent, FloorSparseOperatorFusionState):
        raise FloorSparseOperatorFusionError("parent D9 state is required")
    features, labels, ids, hashes, current_k = _validate_support(
        features_by_operator,
        operator_feature_provenance,
        support_labels,
        physical_sample_ids,
        parent_received_iq_sha256,
    )
    if current_k != parent.selection_lock_k:
        raise FloorSparseOperatorFusionError(
            "D9 registration must use the locked uniform K10 workpoint"
        )
    support_classes = set(labels.tolist())
    if not set(parent.classes).issubset(support_classes):
        raise FloorSparseOperatorFusionError(
            "D9 registration support must retain all old classes"
        )
    new_classes = tuple(sorted(support_classes - set(parent.classes)))
    if not new_classes:
        raise FloorSparseOperatorFusionError(
            "D9 registration received no absent classes"
        )
    incoming_old_lineage = _lineage(
        labels, ids, hashes, parent.classes
    )
    if incoming_old_lineage != parent.support_lineage:
        raise FloorSparseOperatorFusionError(
            "D9 old support lineage changed during registration"
        )
    selected, selection = _select_extension(
        parent,
        features,
        labels,
        new_classes,
        floor_priority_classes,
    )
    new_mask = np.isin(labels, np.asarray(new_classes))
    new_features = {
        operator: rows[new_mask] for operator, rows in features.items()
    }
    new_labels = labels[new_mask]
    new_indices, new_weights, new_prototypes = _state_tensors(
        new_features,
        new_labels,
        new_classes,
        selected,
        parent.calibrations,
    )
    operator_indices = np.concatenate(
        [parent.operator_indices.copy(), new_indices], axis=0
    )
    weights = np.concatenate(
        [parent.weights.copy(), new_weights], axis=0
    )
    prototypes = np.concatenate(
        [parent.prototypes.copy(), new_prototypes], axis=0
    )
    new_lineage = _lineage(
        new_labels,
        tuple(ids[index] for index in np.flatnonzero(new_mask)),
        tuple(hashes[index] for index in np.flatnonzero(new_mask)),
        new_classes,
    )
    support_lineage = parent.support_lineage + new_lineage
    selection_lock = _selection_lock_sha256(
        parent.classes + new_classes,
        operator_indices,
        weights,
        parent.calibrations,
    )
    audit = {
        "schema": "cvs.phase2.d9_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "physical_support_sample_count": len(labels),
        "physical_support_ids_unique": True,
        "parent_received_iq_hashes_unique": True,
        "clean_sample_access": False,
        "source_sample_access": False,
        "query_rows_used": 0,
        "old_state_bitwise_locked": True,
        "old_support_lineage_verified": True,
        "old_state_update_count": 0,
        "new_class_handles": list(new_classes),
        "selection": selection,
    }
    state = FloorSparseOperatorFusionState(
        schema=parent.schema,
        classes=parent.classes + new_classes,
        operator_indices=_readonly(operator_indices, np.int8),
        weights=_readonly(weights, np.float32),
        prototypes=_readonly(prototypes, np.float32),
        calibrations=parent.calibrations,
        feature_dim=parent.feature_dim,
        used_operators=_used_operators(operator_indices, weights),
        old_class_count=parent.old_class_count,
        registration_generation=parent.registration_generation + 1,
        current_k=current_k,
        selection_lock_k=parent.selection_lock_k,
        selection_lock_sha256=selection_lock,
        support_lineage=support_lineage,
        base_persistent_state_bytes=parent.base_persistent_state_bytes,
        base_head_macs_per_query=parent.base_head_macs_per_query,
        support_audit=audit,
    )
    old_count = parent.class_count
    if (
        not np.array_equal(
            state.operator_indices[:old_count],
            parent.operator_indices,
        )
        or not np.array_equal(
            state.weights[:old_count], parent.weights
        )
        or not np.array_equal(
            state.prototypes[:old_count], parent.prototypes
        )
        or state.calibrations != parent.calibrations
        or state.classes[:old_count] != parent.classes
        or state.support_lineage[: len(parent.support_lineage)]
        != parent.support_lineage
    ):
        raise FloorSparseOperatorFusionError(
            "D9 parent old state mutation"
        )
    if state.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise FloorSparseOperatorFusionError(
            "D9 persistent state cap exceeded"
        )
    return state


def rebuild_locked_floor_sparse_prototypes(
    locked_k10_state: FloorSparseOperatorFusionState,
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
) -> FloorSparseOperatorFusionState:
    """Rebuild only prototypes for nested K1/K5 under a K10 selection lock."""

    if (
        not isinstance(
            locked_k10_state, FloorSparseOperatorFusionState
        )
        or locked_k10_state.selection_lock_k != 10
    ):
        raise FloorSparseOperatorFusionError(
            "D9 nested rebuild requires a K10 selection lock"
        )
    features, labels, ids, hashes, current_k = _validate_support(
        features_by_operator,
        operator_feature_provenance,
        support_labels,
        physical_sample_ids,
        parent_received_iq_sha256,
    )
    if current_k not in {1, 5}:
        raise FloorSparseOperatorFusionError(
            "D9 locked prototype rebuild only permits K1 or K5"
        )
    if set(labels.tolist()) != set(locked_k10_state.classes):
        raise FloorSparseOperatorFusionError(
            "D9 nested rebuild registered-class set drift"
        )
    expected_lineage = tuple(
        row
        for label in locked_k10_state.classes
        for row in [
            value
            for value in locked_k10_state.support_lineage
            if value[0] == label
        ][:current_k]
    )
    support_lineage = _lineage(
        labels, ids, hashes, locked_k10_state.classes
    )
    if support_lineage != expected_lineage:
        raise FloorSparseOperatorFusionError(
            "D9 K1/K5 support is not the locked K10 lineage prefix"
        )
    prototypes = np.zeros_like(locked_k10_state.prototypes)
    for class_index, label in enumerate(locked_k10_state.classes):
        mask = labels == label
        for slot in range(MAX_COMPONENTS_PER_CLASS):
            weight = float(
                locked_k10_state.weights[class_index, slot]
            )
            operator_index = int(
                locked_k10_state.operator_indices[class_index, slot]
            )
            if weight > 0.0:
                prototypes[class_index, slot] = _prototype(
                    features[OPERATORS[operator_index]][mask]
                )
    state = FloorSparseOperatorFusionState(
        schema=locked_k10_state.schema,
        classes=locked_k10_state.classes,
        operator_indices=locked_k10_state.operator_indices,
        weights=locked_k10_state.weights,
        prototypes=_readonly(prototypes, np.float32),
        calibrations=locked_k10_state.calibrations,
        feature_dim=locked_k10_state.feature_dim,
        used_operators=locked_k10_state.used_operators,
        old_class_count=locked_k10_state.old_class_count,
        registration_generation=(
            locked_k10_state.registration_generation
        ),
        current_k=current_k,
        selection_lock_k=locked_k10_state.selection_lock_k,
        selection_lock_sha256=(
            locked_k10_state.selection_lock_sha256
        ),
        support_lineage=support_lineage,
        base_persistent_state_bytes=(
            locked_k10_state.base_persistent_state_bytes
        ),
        base_head_macs_per_query=(
            locked_k10_state.base_head_macs_per_query
        ),
        support_audit={
            "schema": "cvs.phase2.d9_nested_k_rebuild_audit.v1",
            "fit_scope": "registered_physical_support_only",
            "current_k": current_k,
            "selection_lock_k": 10,
            "selection_reused_without_reselection": True,
            "operator_indices_bitwise_locked": True,
            "weights_bitwise_locked": True,
            "calibrations_locked": True,
            "only_prototypes_rebuilt": True,
            "k10_lineage_prefix_verified": True,
            "query_rows_used": 0,
            "query_roles_used": False,
            "query_quota_used": False,
        },
    )
    if (
        state.operator_indices is not locked_k10_state.operator_indices
        or state.weights is not locked_k10_state.weights
        or state.calibrations != locked_k10_state.calibrations
        or state.selection_lock_sha256
        != locked_k10_state.selection_lock_sha256
    ):
        raise FloorSparseOperatorFusionError(
            "D9 K10 operator/weight/calibration lock drift"
        )
    if state.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise FloorSparseOperatorFusionError(
            "D9 combined persistent state cap exceeded"
        )
    return state


def predict_floor_sparse_operator_fusion(
    state: FloorSparseOperatorFusionState,
    received_iq: np.ndarray,
    *,
    feature_extractor: SamplewiseSealedFeatureExtractor,
) -> FloorSparseFusionPrediction:
    """Score each query independently over all registered classes."""

    if not isinstance(
        feature_extractor, SamplewiseSealedFeatureExtractor
    ):
        raise FloorSparseOperatorFusionError(
            "D9 query extractor must be samplewise sealed"
        )
    expected_seal = _samplewise_seal_digest(
        feature_extractor.extractor_id,
        feature_extractor.validation_rows_sha256,
    )
    if (
        feature_extractor.samplewise_contract_sha256 != expected_seal
        or len(feature_extractor.validation_rows_sha256) != 64
        or feature_extractor.validation_max_abs_error > 1.0e-6
        or not feature_extractor.batch_independent
        or feature_extractor.query_updates != 0
    ):
        raise FloorSparseOperatorFusionError(
            "D9 samplewise extractor contract drift"
        )
    features = extract_operator_features(
        received_iq,
        feature_extractor=feature_extractor,
        operator_ids=state.used_operators,
    )
    scores = _readonly(_score_from_features(state, features), np.float32)
    prediction = np.argmax(scores, axis=1)
    return FloorSparseFusionPrediction(
        labels=tuple(state.classes[int(index)] for index in prediction),
        scores=scores,
        operators_computed=state.used_operators,
    )


def public_query_interface_is_oracle_free() -> bool:
    forbidden = {"label", "truth", "role", "quota", "assignment", "graph"}
    return not any(
        token in parameter.lower()
        for parameter in inspect.signature(
            predict_floor_sparse_operator_fusion
        ).parameters
        for token in forbidden
    )


__all__ = [
    "BASE_CANDIDATE",
    "DEFAULT_FLOOR_PRIORITY_CLASSES",
    "FIXED_CANDIDATES",
    "FloorSparseFusionPrediction",
    "FloorSparseOperatorFusionError",
    "FloorSparseOperatorFusionState",
    "FusionCandidate",
    "SamplewiseSealedFeatureExtractor",
    "build_operator_feature_provenance",
    "extend_floor_sparse_operator_fusion",
    "fit_floor_sparse_operator_fusion",
    "predict_floor_sparse_operator_fusion",
    "public_query_interface_is_oracle_free",
    "rebuild_locked_floor_sparse_prototypes",
    "seal_samplewise_feature_extractor",
]
