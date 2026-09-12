"""D7a class-conditional representation head from one fixed received IQ.

Only three pre-registered post-reception operators are available.  Each class
selects one operator from physical-support leave-two-out evidence.  Query
inference computes only the operators used by the immutable registry and
scores every registered class independently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EPS = 1.0e-8
BASE = "base"
DC_RMS = "dc_rms"
DC_RMS_SPEC15 = "dc_rms_spec15"
OPERATORS = (BASE, DC_RMS, DC_RMS_SPEC15)
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
MAX_TRAINABLE_PARAMETERS = 80_000
DEFAULT_OVERALL_DROP_TOLERANCE = 0.01
VALIDATED_FEATURE_SCHEMA = "cvs.phase2.validated_operator_features.v1"
DETERMINISTIC_VIEW_SEED = 0
_FEATURE_ARTIFACT_AUTHORITY = object()


class ClassConditionalIQHeadError(ValueError):
    """Raised when D7a input, selection, or inference drifts."""


@dataclass(frozen=True)
class OperatorCalibration:
    operator_id: str
    center: float
    scale: float


@dataclass(frozen=True)
class ClassConditionalIQHeadState:
    schema: str
    classes: tuple[str, ...]
    class_operators: tuple[str, ...]
    prototypes: np.ndarray
    calibrations: tuple[OperatorCalibration, ...]
    feature_dim: int
    selection_trace: tuple[dict[str, Any], ...]
    used_operators: tuple[str, ...]
    registration_generation: int
    persistent_state_bytes: int
    trainable_parameters: int = 0
    query_rows_used_for_fit: int = 0
    query_updates: int = 0

    def calibration_for(self, operator_id: str) -> OperatorCalibration:
        for calibration in self.calibrations:
            if calibration.operator_id == operator_id:
                return calibration
        raise ClassConditionalIQHeadError("D7a operator calibration missing")

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d7a_resource.v1",
            "candidate": "d7a_class_conditional_fixed_received_iq_head",
            "support_only": True,
            "adaptation_epochs": 0,
            "trainable_parameters": self.trainable_parameters,
            "trainable_parameter_limit_pass": (
                self.trainable_parameters <= MAX_TRAINABLE_PARAMETERS
            ),
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_updates": self.query_updates,
            "query_decision_policy": "per_sample_all_registered_classes",
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "used_operator_count": len(self.used_operators),
            "maximum_query_operator_views": len(OPERATORS),
            "query_operator_views": list(self.used_operators),
            "additional_physical_samples_from_views": 0,
            "additional_leo_channel_states_generated": 0,
            "fixed_received_iq_only": True,
            "formal_query_feature_extractor_batch_size": 1,
            "query_query_feature_interaction_possible": False,
            "old_state_locked_after_registration": (
                self.registration_generation > 0
            ),
        }


@dataclass(frozen=True)
class ClassConditionalPrediction:
    labels: tuple[str, ...]
    scores: np.ndarray
    operators_computed: tuple[str, ...]


@dataclass(frozen=True)
class OperatorFeatureBinding:
    """One deterministic post-reception view bound to one received IQ row."""

    sample_index: int
    physical_sample_id: str
    parent_received_iq_sha256: str
    operator_id: str
    view_seed: int
    feature_sha256: str
    post_reception_view_used: bool = True


@dataclass(frozen=True)
class ValidatedOperatorFeatureArtifact:
    """Runtime-authorized operator features derived from fixed received IQ."""

    schema: str
    operator_ids: tuple[str, ...]
    physical_sample_ids: tuple[str, ...]
    parent_received_iq_sha256: tuple[str, ...]
    feature_dim: int
    bindings: tuple[OperatorFeatureBinding, ...]
    seal_sha256: str
    _operator_features: tuple[tuple[str, np.ndarray], ...]
    _authority: object

    @property
    def sample_count(self) -> int:
        return len(self.physical_sample_ids)

    def feature_map(self) -> dict[str, np.ndarray]:
        return {operator: rows for operator, rows in self._operator_features}


def _received_iq_sha256(rows: np.ndarray) -> tuple[str, ...]:
    iq = np.asarray(rows)
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or not np.isfinite(iq).all()
    ):
        raise ClassConditionalIQHeadError(
            "received IQ must be finite float32 [N,2,L]"
        )
    return tuple(
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    )


def _feature_sha256(row: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(row, dtype=np.float32).tobytes()
    ).hexdigest()


def _artifact_seal(
    *,
    operator_ids: Sequence[str],
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    feature_dim: int,
    features: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(VALIDATED_FEATURE_SCHEMA.encode())
    digest.update(str(int(feature_dim)).encode())
    for value in operator_ids:
        digest.update(b"\0op\0")
        digest.update(str(value).encode())
    for value in physical_sample_ids:
        digest.update(b"\0id\0")
        digest.update(str(value).encode())
    for value in parent_received_iq_sha256:
        digest.update(b"\0iq\0")
        digest.update(str(value).encode())
    for operator in operator_ids:
        digest.update(b"\0feature\0")
        digest.update(str(operator).encode())
        digest.update(
            np.ascontiguousarray(
                features[str(operator)], dtype=np.float32
            ).tobytes()
        )
    return digest.hexdigest()


def build_validated_operator_feature_artifact(
    received_iq: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    operator_ids: Sequence[str] = OPERATORS,
) -> ValidatedOperatorFeatureArtifact:
    """Extract each operator one physical row at a time and bind provenance."""

    iq = np.asarray(received_iq)
    actual_hashes = _received_iq_sha256(iq)
    ids = tuple(str(value) for value in physical_sample_ids)
    hashes = tuple(
        str(value).lower() for value in parent_received_iq_sha256
    )
    operators = tuple(dict.fromkeys(str(value) for value in operator_ids))
    if (
        not operators
        or any(operator not in OPERATORS for operator in operators)
        or len(iq) < 1
        or len(ids) != len(iq)
        or len(hashes) != len(iq)
        or len(set(ids)) != len(ids)
        or len(set(hashes)) != len(hashes)
        or any(not value for value in ids)
        or hashes != actual_hashes
    ):
        raise ClassConditionalIQHeadError(
            "validated operator feature lineage drift"
        )
    feature_rows: list[tuple[str, np.ndarray]] = []
    bindings: list[OperatorFeatureBinding] = []
    feature_dim: int | None = None
    for operator in operators:
        rows = []
        for sample_index in range(len(iq)):
            view = apply_received_iq_operator(
                iq[sample_index : sample_index + 1], operator
            )
            feature = np.asarray(
                feature_extractor(view), dtype=np.float32
            )
            if (
                feature.ndim != 2
                or feature.shape[0] != 1
                or feature.shape[1] < 1
                or not np.isfinite(feature).all()
            ):
                raise ClassConditionalIQHeadError(
                    "samplewise feature extractor output drift"
                )
            if feature_dim is None:
                feature_dim = int(feature.shape[1])
            elif feature.shape[1] != feature_dim:
                raise ClassConditionalIQHeadError(
                    "samplewise operator feature dimension drift"
                )
            row = np.ascontiguousarray(feature[0], dtype=np.float32)
            rows.append(row)
            bindings.append(
                OperatorFeatureBinding(
                    sample_index=sample_index,
                    physical_sample_id=ids[sample_index],
                    parent_received_iq_sha256=hashes[sample_index],
                    operator_id=operator,
                    view_seed=DETERMINISTIC_VIEW_SEED,
                    feature_sha256=_feature_sha256(row),
                )
            )
        array = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        array.setflags(write=False)
        feature_rows.append((operator, array))
    if feature_dim is None:
        raise ClassConditionalIQHeadError(
            "validated operator feature artifact is empty"
        )
    feature_map = dict(feature_rows)
    return ValidatedOperatorFeatureArtifact(
        schema=VALIDATED_FEATURE_SCHEMA,
        operator_ids=operators,
        physical_sample_ids=ids,
        parent_received_iq_sha256=hashes,
        feature_dim=feature_dim,
        bindings=tuple(bindings),
        seal_sha256=_artifact_seal(
            operator_ids=operators,
            physical_sample_ids=ids,
            parent_received_iq_sha256=hashes,
            feature_dim=feature_dim,
            features=feature_map,
        ),
        _operator_features=tuple(feature_rows),
        _authority=_FEATURE_ARTIFACT_AUTHORITY,
    )


def validate_operator_feature_artifact(
    artifact: ValidatedOperatorFeatureArtifact,
    *,
    expected_operator_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Fail closed unless a runtime-built artifact and all bindings match."""

    expected = tuple(str(value) for value in expected_operator_ids)
    if (
        not isinstance(artifact, ValidatedOperatorFeatureArtifact)
        or artifact._authority is not _FEATURE_ARTIFACT_AUTHORITY
        or artifact.schema != VALIDATED_FEATURE_SCHEMA
        or artifact.operator_ids != expected
        or artifact.feature_dim < 1
        or len(set(artifact.physical_sample_ids)) != artifact.sample_count
        or len(set(artifact.parent_received_iq_sha256))
        != artifact.sample_count
    ):
        raise ClassConditionalIQHeadError(
            "validated operator feature artifact authority drift"
        )
    features = artifact.feature_map()
    if set(features) != set(expected):
        raise ClassConditionalIQHeadError(
            "validated operator feature set drift"
        )
    for operator in expected:
        rows = np.asarray(features[operator])
        if (
            rows.dtype != np.float32
            or rows.ndim != 2
            or rows.shape
            != (artifact.sample_count, artifact.feature_dim)
            or rows.flags.writeable
            or not np.isfinite(rows).all()
        ):
            raise ClassConditionalIQHeadError(
                "validated operator feature payload drift"
            )
    expected_bindings = []
    for operator in expected:
        for sample_index in range(artifact.sample_count):
            expected_bindings.append(
                OperatorFeatureBinding(
                    sample_index=sample_index,
                    physical_sample_id=artifact.physical_sample_ids[
                        sample_index
                    ],
                    parent_received_iq_sha256=(
                        artifact.parent_received_iq_sha256[sample_index]
                    ),
                    operator_id=operator,
                    view_seed=DETERMINISTIC_VIEW_SEED,
                    feature_sha256=_feature_sha256(
                        features[operator][sample_index]
                    ),
                )
            )
    if tuple(expected_bindings) != artifact.bindings:
        raise ClassConditionalIQHeadError(
            "validated operator feature binding drift"
        )
    seal = _artifact_seal(
        operator_ids=expected,
        physical_sample_ids=artifact.physical_sample_ids,
        parent_received_iq_sha256=artifact.parent_received_iq_sha256,
        feature_dim=artifact.feature_dim,
        features=features,
    )
    if seal != artifact.seal_sha256:
        raise ClassConditionalIQHeadError(
            "validated operator feature seal drift"
        )
    return features


def apply_received_iq_operator(
    received_iq: np.ndarray, operator_id: str
) -> np.ndarray:
    """Apply one deterministic operator to finite float32 [N,2,L] IQ."""

    iq = np.asarray(received_iq)
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or not np.isfinite(iq).all()
    ):
        raise ClassConditionalIQHeadError(
            "received IQ must be finite float32 [N,2,L]"
        )
    operator = str(operator_id)
    if operator not in OPERATORS:
        raise ClassConditionalIQHeadError("unsupported D7a operator")
    if operator == BASE:
        return np.ascontiguousarray(iq)
    centered = iq - iq.mean(axis=2, keepdims=True, dtype=np.float32)
    rms = np.sqrt(
        np.mean(np.square(centered), axis=(1, 2), keepdims=True) + EPS
    )
    normalized = centered / rms
    if operator == DC_RMS:
        return np.ascontiguousarray(normalized, dtype=np.float32)
    complex_rows = normalized[:, 0] + 1j * normalized[:, 1]
    spectrum = np.fft.fft(complex_rows, axis=1)
    magnitude = np.abs(spectrum)
    phase = spectrum / np.maximum(magnitude, EPS)
    median = np.median(magnitude, axis=1, keepdims=True)
    shrunk = 0.85 * magnitude + 0.15 * median
    output = np.fft.ifft(shrunk * phase, axis=1)
    return np.ascontiguousarray(
        np.stack([output.real, output.imag], axis=1), dtype=np.float32
    )


def extract_operator_features(
    received_iq: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    operator_ids: Sequence[str] = OPERATORS,
) -> dict[str, np.ndarray]:
    """Extract actual operator-specific features from the fixed IQ payload."""

    result: dict[str, np.ndarray] = {}
    row_count = len(received_iq)
    dimension: int | None = None
    for operator in tuple(dict.fromkeys(str(value) for value in operator_ids)):
        view = apply_received_iq_operator(received_iq, operator)
        features = np.asarray(feature_extractor(view), dtype=np.float32)
        if (
            features.ndim != 2
            or len(features) != row_count
            or not np.isfinite(features).all()
        ):
            raise ClassConditionalIQHeadError(
                "operator feature extractor output drift"
            )
        if dimension is None:
            dimension = features.shape[1]
        elif features.shape[1] != dimension:
            raise ClassConditionalIQHeadError(
                "operator feature dimension drift"
            )
        result[operator] = np.ascontiguousarray(features)
    return result


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), EPS
    )


def _prototype(rows: np.ndarray) -> np.ndarray:
    return _normalize(_normalize(rows).mean(axis=0, keepdims=True))[0]


def class_conditional_state_persistent_bytes(
    state: ClassConditionalIQHeadState,
) -> int:
    """Recompute deployable D7a state bytes instead of trusting metadata."""

    if not isinstance(state, ClassConditionalIQHeadState):
        raise ClassConditionalIQHeadError("D7a state is required")
    return int(
        np.asarray(state.prototypes).nbytes
        + sum(
            len(value.encode())
            for value in state.classes + state.class_operators
        )
        + len(state.calibrations) * 8
    )


def _prototypes(
    features: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]
) -> np.ndarray:
    return np.stack(
        [_prototype(features[labels == label]) for label in classes], axis=0
    ).astype(np.float32)


def _calibration(
    features: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    operator_id: str,
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
        operator_id=operator_id,
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


def _folds(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    classes = tuple(sorted(set(labels.tolist())))
    by_class = [np.flatnonzero(labels == label).tolist() for label in classes]
    if min(len(values) for values in by_class) < 3:
        raise ClassConditionalIQHeadError(
            "D7a leave-two-out requires at least three support samples per class"
        )
    return tuple(
        tuple(
            sorted(
                index
                for values in by_class
                for index in values[offset : offset + 2]
            )
        )
        for offset in range(0, max(map(len, by_class)), 2)
        if any(values[offset : offset + 2] for values in by_class)
    )


def _validate_inputs(
    artifact: ValidatedOperatorFeatureArtifact,
    labels: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features_by_operator = validate_operator_feature_artifact(
        artifact, expected_operator_ids=OPERATORS
    )
    label_array = np.asarray(tuple(str(value) for value in labels))
    row_count = len(label_array)
    prepared: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for operator in OPERATORS:
        rows = np.asarray(features_by_operator[operator], dtype=np.float32)
        if (
            rows.ndim != 2
            or len(rows) != row_count
            or not np.isfinite(rows).all()
        ):
            raise ClassConditionalIQHeadError("D7a support feature drift")
        if dimension is None:
            dimension = rows.shape[1]
        elif rows.shape[1] != dimension:
            raise ClassConditionalIQHeadError("D7a feature dimension drift")
        prepared[operator] = np.ascontiguousarray(rows)
    if (
        row_count < 1
        or artifact.sample_count != row_count
    ):
        raise ClassConditionalIQHeadError(
            "D7a single-observation lineage drift"
        )
    return prepared, label_array


def _deletion_evidence(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    calibration_override: Mapping[str, OperatorCalibration] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    classes = tuple(sorted(set(labels.tolist())))
    class_results = {
        label: {
            operator: {"correct": 0, "total": 0, "margins": [], "overall_correct": 0}
            for operator in OPERATORS
        }
        for label in classes
    }
    baseline_correct = 0
    baseline_by_class = {label: [0, 0] for label in classes}
    combined_records: list[dict[str, Any]] = []
    for held_tuple in _folds(labels):
        held = np.asarray(held_tuple, dtype=np.int64)
        keep = np.ones(len(labels), dtype=bool)
        keep[held] = False
        prototypes: dict[str, np.ndarray] = {}
        calibrations: dict[str, OperatorCalibration] = {}
        scores: dict[str, np.ndarray] = {}
        for operator in OPERATORS:
            prototypes[operator] = _prototypes(
                features[operator][keep], labels[keep], classes
            )
            calibrations[operator] = (
                calibration_override[operator]
                if calibration_override is not None
                else _calibration(
                    features[operator][keep],
                    labels[keep],
                    classes,
                    prototypes[operator],
                    operator,
                )
            )
            scores[operator] = _calibrated_scores(
                features[operator][held],
                prototypes[operator],
                calibrations[operator],
            )
        base_pred = np.argmax(scores[BASE], axis=1)
        for row_index, true_label in enumerate(labels[held].tolist()):
            true_index = classes.index(true_label)
            baseline_correct += int(base_pred[row_index] == true_index)
            baseline_by_class[true_label][0] += int(
                base_pred[row_index] == true_index
            )
            baseline_by_class[true_label][1] += 1
        for target_index, target_label in enumerate(classes):
            for operator in OPERATORS:
                mixed = scores[BASE].copy()
                mixed[:, target_index] = scores[operator][:, target_index]
                prediction = np.argmax(mixed, axis=1)
                order = np.sort(mixed, axis=1)
                margins = order[:, -1] - order[:, -2]
                result = class_results[target_label][operator]
                result["overall_correct"] += int(
                    np.sum(prediction == np.asarray(
                        [classes.index(value) for value in labels[held]]
                    ))
                )
                for row_index, true_label in enumerate(labels[held].tolist()):
                    if true_label == target_label:
                        result["correct"] += int(
                            prediction[row_index] == target_index
                        )
                        result["total"] += 1
                        result["margins"].append(float(margins[row_index]))
        combined_records.append(
            {
                "held": held,
                "scores": scores,
                "truth": np.asarray(
                    [classes.index(value) for value in labels[held]]
                ),
            }
        )
    baseline = {
        "overall_accuracy": baseline_correct / len(labels),
        "per_class_accuracy": {
            label: correct / total
            for label, (correct, total) in baseline_by_class.items()
        },
    }
    for label in classes:
        base_class = baseline["per_class_accuracy"][label]
        for operator in OPERATORS:
            result = class_results[label][operator]
            result["class_accuracy"] = result["correct"] / result["total"]
            result["overall_accuracy"] = (
                result["overall_correct"] / len(labels)
            )
            result["mean_margin"] = float(np.mean(result["margins"]))
            result["class_non_degradation_pass"] = (
                result["class_accuracy"] >= base_class
            )
            result["overall_tolerance_pass"] = (
                result["overall_accuracy"]
                + DEFAULT_OVERALL_DROP_TOLERANCE
                >= baseline["overall_accuracy"]
            )
    return class_results, {
        "classes": classes,
        "baseline": baseline,
        "fold_records": combined_records,
    }


def fit_class_conditional_head(
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
) -> ClassConditionalIQHeadState:
    """Select one operator per class using support leave-two-out only."""

    features, labels = _validate_inputs(
        artifact,
        support_labels,
    )
    results, common = _deletion_evidence(features, labels)
    classes = common["classes"]
    complexity = {BASE: 0, DC_RMS: 1, DC_RMS_SPEC15: 2}
    selected: list[str] = []
    trace: list[dict[str, Any]] = []
    for label in classes:
        eligible = [
            (operator, row)
            for operator, row in results[label].items()
            if row["class_non_degradation_pass"]
            and row["overall_tolerance_pass"]
        ]
        chosen, _row = max(
            eligible,
            key=lambda item: (
                item[1]["class_accuracy"],
                item[1]["overall_accuracy"],
                item[1]["mean_margin"],
                -complexity[item[0]],
            ),
        )
        selected.append(chosen)
        trace.append(
            {
                "class_handle": label,
                "identity_baseline_accuracy": common["baseline"][
                    "per_class_accuracy"
                ][label],
                "selected_operator": chosen,
                "operators": results[label],
            }
        )

    combined_correct = 0
    combined_by_class = {label: [0, 0] for label in classes}
    for record in common["fold_records"]:
        rows = len(record["truth"])
        mixed = np.empty((rows, len(classes)), dtype=np.float32)
        for class_index, operator in enumerate(selected):
            mixed[:, class_index] = record["scores"][operator][:, class_index]
        prediction = np.argmax(mixed, axis=1)
        combined_correct += int(np.sum(prediction == record["truth"]))
        for predicted, truth_index in zip(prediction, record["truth"]):
            label = classes[int(truth_index)]
            combined_by_class[label][0] += int(predicted == truth_index)
            combined_by_class[label][1] += 1
    combined_overall = combined_correct / len(labels)
    combined_floor = min(
        correct / total for correct, total in combined_by_class.values()
    )
    baseline_floor = min(common["baseline"]["per_class_accuracy"].values())
    global_pass = (
        combined_overall + DEFAULT_OVERALL_DROP_TOLERANCE
        >= common["baseline"]["overall_accuracy"]
        and combined_floor >= baseline_floor
    )
    assignment_before_fallback = tuple(selected)
    if not global_pass:
        selected = [BASE] * len(classes)
    trace.append(
        {
            "combined_operator_assignment_before_fallback": {
                label: operator
                for label, operator in zip(
                    classes, assignment_before_fallback
                )
            },
            "combined_overall_accuracy": combined_overall,
            "combined_min_class_accuracy": combined_floor,
            "baseline_overall_accuracy": common["baseline"]["overall_accuracy"],
            "baseline_min_class_accuracy": baseline_floor,
            "global_non_degradation_pass": global_pass,
            "fallback_to_all_base": not global_pass,
        }
    )
    calibrations: list[OperatorCalibration] = []
    all_prototypes: dict[str, np.ndarray] = {}
    for operator in OPERATORS:
        all_prototypes[operator] = _prototypes(
            features[operator], labels, classes
        )
        calibrations.append(
            _calibration(
                features[operator],
                labels,
                classes,
                all_prototypes[operator],
                operator,
            )
        )
    prototypes = np.stack(
        [
            all_prototypes[operator][class_index]
            for class_index, operator in enumerate(selected)
        ],
        axis=0,
    ).astype(np.float32)
    used = tuple(operator for operator in OPERATORS if operator in selected)
    state_bytes = int(
        prototypes.nbytes
        + sum(len(value.encode()) for value in classes + tuple(selected))
        + len(calibrations) * 8
    )
    if state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise ClassConditionalIQHeadError("D7a persistent state cap exceeded")
    prototypes.setflags(write=False)
    return ClassConditionalIQHeadState(
        schema="cvs.phase2.d7a_class_conditional_iq_head.v1",
        classes=classes,
        class_operators=tuple(selected),
        prototypes=prototypes,
        calibrations=tuple(calibrations),
        feature_dim=prototypes.shape[1],
        selection_trace=tuple(trace),
        used_operators=used,
        registration_generation=0,
        persistent_state_bytes=state_bytes,
    )


def register_absent_classes(
    base_state: ClassConditionalIQHeadState,
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
) -> ClassConditionalIQHeadState:
    """Freeze the old state and append only absent support-registered classes."""

    features, labels = _validate_inputs(
        artifact,
        support_labels,
    )
    absent = tuple(sorted(set(labels.tolist()) - set(base_state.classes)))
    if not absent:
        raise ClassConditionalIQHeadError(
            "D7a registration received no absent classes"
        )
    mask = np.isin(labels, np.asarray(absent))
    new_features = {
        operator: rows[mask] for operator, rows in features.items()
    }
    new_labels = labels[mask]
    calibration_map = {
        value.operator_id: value for value in base_state.calibrations
    }
    results, common = _deletion_evidence(
        new_features,
        new_labels,
        calibration_override=calibration_map,
    )
    new_classes = common["classes"]
    complexity = {BASE: 0, DC_RMS: 1, DC_RMS_SPEC15: 2}
    selected: list[str] = []
    new_trace: list[dict[str, Any]] = []
    for label in new_classes:
        eligible = [
            (operator, row)
            for operator, row in results[label].items()
            if row["class_non_degradation_pass"]
            and row["overall_tolerance_pass"]
        ]
        chosen, _row = max(
            eligible,
            key=lambda item: (
                item[1]["class_accuracy"],
                item[1]["overall_accuracy"],
                item[1]["mean_margin"],
                -complexity[item[0]],
            ),
        )
        selected.append(chosen)
        new_trace.append(
            {
                "class_handle": label,
                "identity_baseline_accuracy": common["baseline"][
                    "per_class_accuracy"
                ][label],
                "selected_operator": chosen,
                "operators": results[label],
                "selection_calibration_policy": (
                    "frozen_before_operator_calibration"
                ),
            }
        )
    all_new_prototypes = {
        operator: _prototypes(new_features[operator], new_labels, new_classes)
        for operator in OPERATORS
    }
    new_prototypes = [
        all_new_prototypes[operator][index]
        for index, operator in enumerate(selected)
    ]
    classes = base_state.classes + new_classes
    operators = base_state.class_operators + tuple(selected)
    prototypes = np.concatenate(
        [base_state.prototypes, np.stack(new_prototypes)], axis=0
    ).astype(np.float32)
    prototypes.setflags(write=False)
    used = tuple(operator for operator in OPERATORS if operator in operators)
    state_bytes = int(
        prototypes.nbytes
        + sum(len(value.encode()) for value in classes + operators)
        + len(base_state.calibrations) * 8
    )
    if state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise ClassConditionalIQHeadError(
            "D7a registered persistent state cap exceeded"
        )
    return replace(
        base_state,
        classes=classes,
        class_operators=operators,
        prototypes=prototypes,
        used_operators=used,
        registration_generation=base_state.registration_generation + 1,
        persistent_state_bytes=state_bytes,
        selection_trace=base_state.selection_trace
        + (
            {
                "registration": "absent_classes_only",
                "new_class_selection_trace": tuple(new_trace),
                "old_calibration_locked": True,
                "old_prototypes_locked": True,
            },
        ),
    )


def rebuild_prototypes_from_locked_policy(
    locked_state: ClassConditionalIQHeadState,
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
    *,
    expected_k: int,
    locked_from_k: int = 10,
) -> ClassConditionalIQHeadState:
    """Rebuild only prototypes for K1/K5 using a K10-locked D7a policy."""

    if (
        not isinstance(locked_state, ClassConditionalIQHeadState)
        or int(locked_from_k) != 10
        or int(expected_k) not in {1, 5, 10, 20}
        or locked_state.query_rows_used_for_fit != 0
        or locked_state.query_updates != 0
        or locked_state.trainable_parameters > MAX_TRAINABLE_PARAMETERS
        or locked_state.persistent_state_bytes
        != class_conditional_state_persistent_bytes(locked_state)
    ):
        raise ClassConditionalIQHeadError(
            "locked D7a strategy or resource contract drift"
        )
    features, labels = _validate_inputs(artifact, support_labels)
    if tuple(sorted(set(labels.tolist()))) != tuple(
        sorted(locked_state.classes)
    ):
        raise ClassConditionalIQHeadError(
            "locked D7a strategy class set drift"
        )
    counts = {
        label: int(np.sum(labels == label)) for label in locked_state.classes
    }
    if set(counts.values()) != {int(expected_k)}:
        raise ClassConditionalIQHeadError(
            "locked D7a prototype rebuild K-shot drift"
        )
    prototypes = np.stack(
        [
            _prototype(
                features[operator][labels == locked_state.classes[index]]
            )
            for index, operator in enumerate(locked_state.class_operators)
        ]
    ).astype(np.float32)
    prototypes.setflags(write=False)
    rebuilt = replace(
        locked_state,
        prototypes=prototypes,
        selection_trace=locked_state.selection_trace
        + (
            {
                "prototype_rebuild_only": True,
                "locked_from_k": int(locked_from_k),
                "prototype_rebuild_k": int(expected_k),
                "operator_reselected": False,
                "calibration_reselected": False,
            },
        ),
    )
    rebuilt = replace(
        rebuilt,
        persistent_state_bytes=class_conditional_state_persistent_bytes(
            rebuilt
        ),
    )
    if rebuilt.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise ClassConditionalIQHeadError(
            "rebuilt D7a persistent state cap exceeded"
        )
    return rebuilt


def predict_all_registered(
    state: ClassConditionalIQHeadState,
    received_iq: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
) -> ClassConditionalPrediction:
    """Compute samplewise fixed-IQ views and score all registered classes."""

    artifact = build_validated_operator_feature_artifact(
        received_iq,
        feature_extractor=feature_extractor,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
        operator_ids=state.used_operators,
    )
    features = validate_operator_feature_artifact(
        artifact, expected_operator_ids=state.used_operators
    )
    scores = np.empty(
        (len(received_iq), len(state.classes)), dtype=np.float32
    )
    for class_index, operator in enumerate(state.class_operators):
        calibration = state.calibration_for(operator)
        scores[:, class_index] = (
            _normalize(features[operator]) @ state.prototypes[class_index]
            - calibration.center
        ) / calibration.scale
    scores.setflags(write=False)
    indices = np.argmax(scores, axis=1)
    return ClassConditionalPrediction(
        labels=tuple(state.classes[int(index)] for index in indices),
        scores=scores,
        operators_computed=state.used_operators,
    )


__all__ = [
    "BASE",
    "DC_RMS",
    "DC_RMS_SPEC15",
    "OPERATORS",
    "ClassConditionalIQHeadError",
    "ClassConditionalIQHeadState",
    "ClassConditionalPrediction",
    "OperatorFeatureBinding",
    "ValidatedOperatorFeatureArtifact",
    "apply_received_iq_operator",
    "build_validated_operator_feature_artifact",
    "class_conditional_state_persistent_bytes",
    "extract_operator_features",
    "fit_class_conditional_head",
    "predict_all_registered",
    "rebuild_prototypes_from_locked_policy",
    "register_absent_classes",
    "validate_operator_feature_artifact",
]
