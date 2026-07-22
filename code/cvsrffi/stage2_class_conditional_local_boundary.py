"""D7c class-conditional calibrated-confusion local boundary.

This is not a strict D7b prototype-Gram composition because D7a classes may
use different fixed received-IQ operator spaces. Rivals are selected from
support-deletion D7a calibrated confusion. Fit inputs must be runtime-built
validated operator-feature artifacts from one fixed received IQ per sample.
"""

from __future__ import annotations

import inspect
import math
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_class_conditional_iq_head import (
    OPERATORS,
    ClassConditionalIQHeadState,
    ValidatedOperatorFeatureArtifact,
    build_validated_operator_feature_artifact,
    class_conditional_state_persistent_bytes,
    rebuild_prototypes_from_locked_policy,
    validate_operator_feature_artifact,
)


EPS = 1.0e-8
DEFAULT_BETA_CANDIDATES = (0.0, 0.05, 0.10, 0.20)
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
MAX_TRAINABLE_PARAMETERS = 50_000


class ClassConditionalLocalBoundaryError(ValueError):
    """Raised when D7c support, registration, or query invariants drift."""


@dataclass(frozen=True)
class ClassConditionalLocalBoundaryState:
    """Immutable D7a state plus one local rival/beta per registered class."""

    schema: str
    base_state: ClassConditionalIQHeadState
    rival_indices: np.ndarray
    beta: np.ndarray
    old_class_count: int
    support_labels: tuple[str, ...]
    support_physical_sample_ids: tuple[str, ...]
    support_parent_received_iq_sha256: tuple[str, ...]
    support_binding_fingerprints: tuple[str, ...]
    strategy_locked_k: int | None
    support_audit: Mapping[str, Any]

    @property
    def classes(self) -> tuple[str, ...]:
        return self.base_state.classes

    @property
    def class_count(self) -> int:
        return len(self.base_state.classes)

    @property
    def persistent_state_bytes(self) -> int:
        lineage_bytes = sum(
            len(value.encode())
            for value in (
                self.support_labels
                + self.support_physical_sample_ids
                + self.support_parent_received_iq_sha256
                + self.support_binding_fingerprints
            )
        )
        return int(
            class_conditional_state_persistent_bytes(self.base_state)
            + self.beta.nbytes
            + self.rival_indices.nbytes
            + lineage_bytes
        )

    @property
    def estimated_head_macs_per_query(self) -> int:
        # D7a class-prototype dot products plus subtract/scale and D7c
        # rival subtract/scale/add. Operator/backbone MACs are intentionally
        # not fabricated here.
        return int(
            self.class_count * self.base_state.feature_dim
            + 5 * self.class_count
        )

    def resource_audit(self) -> dict[str, Any]:
        _assert_local_state_contract(self)
        trainable = int(self.base_state.trainable_parameters)
        query_rows = int(self.base_state.query_rows_used_for_fit)
        query_updates = int(self.base_state.query_updates)
        return {
            "schema": "cvs.phase2.d7c_resource.v1",
            "candidate": "d7c_calibrated_confusion_local_boundary",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_SUPPORT_SELECTION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": trainable,
            "trainable_parameter_limit": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_pass": (
                trainable <= MAX_TRAINABLE_PARAMETERS
            ),
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "estimated_head_macs_per_query": (
                self.estimated_head_macs_per_query
            ),
            "head_macs_scope": (
                "prototype_dot_calibration_and_local_margin_only"
            ),
            "end_to_end_macs_per_query": None,
            "end_to_end_latency_ms": None,
            "peak_vram_bytes": None,
            "operator_transform_profile_required": True,
            "backbone_profile_required": True,
            "backbone_forwards_per_query": len(
                self.base_state.used_operators
            ),
            "average_backbone_forwards_per_query": len(
                self.base_state.used_operators
            ),
            "p95_backbone_forwards_per_query": len(
                self.base_state.used_operators
            ),
            "used_operator_count": len(self.base_state.used_operators),
            "used_operators": list(self.base_state.used_operators),
            "maximum_fixed_received_iq_views": len(OPERATORS),
            "views_count_as_additional_k": False,
            "additional_physical_samples_from_views": 0,
            "additional_leo_channel_states_generated": 0,
            "fixed_received_iq_only": True,
            "query_rows_used_for_fit": query_rows,
            "query_updates": query_updates,
            "query_feature_extractor_batch_size": 1,
            "query_query_feature_interaction_possible": False,
            "query_decision_policy": "per_sample_all_registered_classes",
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "dense_query_graph_bytes": 0,
            "strategy_locked_k": self.strategy_locked_k,
            "old_state_locked_after_registration": (
                self.base_state.registration_generation > 0
            ),
        }


@dataclass(frozen=True)
class ClassConditionalLocalPrediction:
    labels: tuple[str, ...]
    scores: np.ndarray
    base_scores: np.ndarray
    operators_computed: tuple[str, ...]


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _assert_base_resource_contract(
    state: ClassConditionalIQHeadState,
) -> None:
    if (
        not isinstance(state, ClassConditionalIQHeadState)
        or state.trainable_parameters < 0
        or state.trainable_parameters > MAX_TRAINABLE_PARAMETERS
        or state.query_rows_used_for_fit != 0
        or state.query_updates != 0
        or state.persistent_state_bytes
        != class_conditional_state_persistent_bytes(state)
    ):
        raise ClassConditionalLocalBoundaryError(
            "D7c inherited D7a resource or query-fit contract drift"
        )


def _binding_fingerprints(
    artifact: ValidatedOperatorFeatureArtifact,
    labels: Sequence[str],
    *,
    include_classes: set[str] | None = None,
) -> tuple[str, ...]:
    label_values = tuple(str(value) for value in labels)
    if len(label_values) != artifact.sample_count:
        raise ClassConditionalLocalBoundaryError(
            "D7c binding label alignment drift"
        )
    values = []
    for binding in artifact.bindings:
        label = label_values[binding.sample_index]
        if include_classes is not None and label not in include_classes:
            continue
        digest = hashlib.sha256()
        for value in (
            label,
            binding.physical_sample_id,
            binding.parent_received_iq_sha256,
            binding.operator_id,
            str(binding.view_seed),
            binding.feature_sha256,
        ):
            digest.update(value.encode())
            digest.update(b"\0")
        values.append(digest.hexdigest())
    return tuple(sorted(values))


def _state_lineage(
    artifact: ValidatedOperatorFeatureArtifact,
    labels: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    return {
        "support_labels": tuple(str(value) for value in labels),
        "support_physical_sample_ids": artifact.physical_sample_ids,
        "support_parent_received_iq_sha256": (
            artifact.parent_received_iq_sha256
        ),
        "support_binding_fingerprints": _binding_fingerprints(
            artifact, labels
        ),
    }


def _assert_local_state_contract(
    state: ClassConditionalLocalBoundaryState,
) -> None:
    if not isinstance(state, ClassConditionalLocalBoundaryState):
        raise ClassConditionalLocalBoundaryError("D7c state is required")
    _assert_base_resource_contract(state.base_state)
    count = state.class_count
    sample_count = len(state.support_labels)
    if (
        state.rival_indices.shape != (count,)
        or state.beta.shape != (count,)
        or np.any(state.rival_indices < 0)
        or np.any(state.rival_indices >= count)
        or np.any(state.rival_indices == np.arange(count))
        or len(state.support_physical_sample_ids) != sample_count
        or len(state.support_parent_received_iq_sha256) != sample_count
        or len(state.support_binding_fingerprints)
        != sample_count * len(OPERATORS)
        or state.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES
    ):
        raise ClassConditionalLocalBoundaryError(
            "D7c local state or persistent resource drift"
        )


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), EPS
    )


def _prototype(rows: np.ndarray) -> np.ndarray:
    return _normalize(_normalize(rows).mean(axis=0, keepdims=True))[0]


def _validate_beta_candidates(
    beta_candidates: Sequence[float],
) -> tuple[float, ...]:
    values = tuple(float(value) for value in beta_candidates)
    if (
        not values
        or values[0] != 0.0
        or len(set(values)) != len(values)
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 0.50
            for value in values
        )
    ):
        raise ClassConditionalLocalBoundaryError(
            "beta candidates must be unique, begin at zero, and stay in [0,0.5]"
        )
    return values


def _validate_support(
    base_state: ClassConditionalIQHeadState,
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
    *,
    expected_classes: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not isinstance(base_state, ClassConditionalIQHeadState):
        raise ClassConditionalLocalBoundaryError("D7a base state is required")
    _assert_base_resource_contract(base_state)
    try:
        features_by_operator = validate_operator_feature_artifact(
            artifact, expected_operator_ids=OPERATORS
        )
    except Exception as exc:
        raise ClassConditionalLocalBoundaryError(
            "D7c validated operator feature artifact drift"
        ) from exc
    labels = np.asarray(tuple(str(value) for value in support_labels))
    row_count = len(labels)
    prepared: dict[str, np.ndarray] = {}
    for operator in OPERATORS:
        rows = np.asarray(features_by_operator[operator], dtype=np.float32)
        if (
            rows.ndim != 2
            or len(rows) != row_count
            or rows.shape[1] != base_state.feature_dim
            or not np.isfinite(rows).all()
        ):
            raise ClassConditionalLocalBoundaryError(
                "D7c support feature drift"
            )
        prepared[operator] = np.ascontiguousarray(rows)
    if (
        row_count < 1
        or artifact.sample_count != row_count
    ):
        raise ClassConditionalLocalBoundaryError(
            "D7c single-observation lineage drift"
        )
    if tuple(sorted(set(labels.tolist()))) != tuple(sorted(expected_classes)):
        raise ClassConditionalLocalBoundaryError(
            "D7c support registered-class set drift"
        )
    if len(tuple(expected_classes)) < 2:
        raise ClassConditionalLocalBoundaryError(
            "D7c requires at least two registered classes"
        )
    return prepared, labels


def _folds(
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    width_per_class: int,
) -> tuple[tuple[int, ...], ...]:
    width = int(width_per_class)
    by_class = [
        np.flatnonzero(labels == label).tolist() for label in classes
    ]
    if width not in {1, 2} or min(map(len, by_class)) <= width:
        raise ClassConditionalLocalBoundaryError(
            "support deletion must retain each selected class"
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
        raise ClassConditionalLocalBoundaryError(
            "physical support deletion coverage drift"
        )
    return tuple(folds)


def _scores_from_features_and_prototypes(
    state: ClassConditionalIQHeadState,
    features_by_operator: Mapping[str, np.ndarray],
    prototypes: np.ndarray,
) -> np.ndarray:
    row_count = len(next(iter(features_by_operator.values())))
    scores = np.empty((row_count, len(state.classes)), dtype=np.float32)
    normalized = {
        operator: _normalize(features_by_operator[operator])
        for operator in state.used_operators
    }
    for class_index, operator in enumerate(state.class_operators):
        calibration = state.calibration_for(operator)
        scores[:, class_index] = (
            normalized[operator] @ prototypes[class_index]
            - calibration.center
        ) / calibration.scale
    return scores


def _score_class_conditional_features(
    state: ClassConditionalIQHeadState,
    features_by_operator: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Expose D7a calibrated base scores for already extracted fixed views."""

    if set(features_by_operator) != set(state.used_operators):
        raise ClassConditionalLocalBoundaryError(
            "query feature operator set must equal locked used operators"
        )
    dimensions = {
        np.asarray(rows).shape[1]
        for rows in features_by_operator.values()
        if np.asarray(rows).ndim == 2
    }
    lengths = {len(np.asarray(rows)) for rows in features_by_operator.values()}
    if (
        dimensions != {state.feature_dim}
        or len(lengths) != 1
        or any(
            not np.isfinite(np.asarray(rows, dtype=np.float32)).all()
            for rows in features_by_operator.values()
        )
    ):
        raise ClassConditionalLocalBoundaryError(
            "query feature shape or value drift"
        )
    scores = _scores_from_features_and_prototypes(
        state, features_by_operator, state.prototypes
    )
    return _readonly(scores, np.float32)


def _apply_local_margin(
    base_scores: np.ndarray,
    rival_indices: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    rivals = base_scores[:, rival_indices]
    return (
        base_scores
        + beta[None, :] * (base_scores - rivals)
    ).astype(np.float32)


def _heldout_records_before(
    state: ClassConditionalIQHeadState,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    classes = state.classes
    width = 2 if min(
        int(np.sum(labels == label)) for label in classes
    ) >= 4 else 1
    records: list[dict[str, np.ndarray]] = []
    for fold in _folds(labels, classes, width_per_class=width):
        held = np.asarray(fold, dtype=np.int64)
        keep = np.ones(len(labels), dtype=bool)
        keep[held] = False
        prototypes = np.stack(
            [
                _prototype(
                    features[operator][
                        keep & (labels == state.classes[class_index])
                    ]
                )
                for class_index, operator in enumerate(state.class_operators)
            ]
        ).astype(np.float32)
        held_features = {
            operator: rows[held] for operator, rows in features.items()
        }
        records.append(
            {
                "scores": _scores_from_features_and_prototypes(
                    state, held_features, prototypes
                ),
                "truth": np.asarray(
                    [classes.index(value) for value in labels[held]],
                    dtype=np.int64,
                ),
            }
        )
    return records


def _select_rivals(
    records: Sequence[Mapping[str, np.ndarray]],
    class_count: int,
    target_indices: Sequence[int],
) -> np.ndarray:
    rivals = np.zeros(class_count, dtype=np.int64)
    for target in target_indices:
        relevant = [
            record["scores"][record["truth"] == target]
            for record in records
            if np.any(record["truth"] == target)
        ]
        mean_scores = np.concatenate(relevant, axis=0).mean(axis=0)
        mean_scores[target] = -np.inf
        rivals[target] = int(np.argmax(mean_scores))
    return rivals


def _metrics(
    scores: np.ndarray,
    truth: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    predicted = np.argmax(scores, axis=1)
    correct = predicted == truth
    per_class = {
        label: float(np.mean(correct[truth == class_index]))
        for class_index, label in enumerate(classes)
        if np.any(truth == class_index)
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


def _combined_record_arrays(
    records: Sequence[Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([record["scores"] for record in records], axis=0),
        np.concatenate([record["truth"] for record in records], axis=0),
    )


def _select_betas_before(
    records: Sequence[Mapping[str, np.ndarray]],
    rivals: np.ndarray,
    classes: Sequence[str],
    candidates: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    base_scores, truth = _combined_record_arrays(records)
    baseline = _metrics(base_scores, truth, classes)
    selected = np.zeros(len(classes), dtype=np.float32)
    class_trace: list[dict[str, Any]] = []
    for target, label in enumerate(classes):
        evidence = []
        for candidate in candidates:
            beta = np.zeros(len(classes), dtype=np.float32)
            beta[target] = candidate
            metrics = _metrics(
                _apply_local_margin(base_scores, rivals, beta),
                truth,
                classes,
            )
            metrics.update(
                {
                    "beta": candidate,
                    "target_class_accuracy": metrics[
                        "per_class_accuracy"
                    ][label],
                    "eligible": bool(
                        metrics["per_class_accuracy"][label]
                        >= baseline["per_class_accuracy"][label]
                        and metrics["overall_accuracy"]
                        >= baseline["overall_accuracy"]
                    ),
                }
            )
            evidence.append(metrics)
        eligible = [row for row in evidence if row["eligible"]]
        chosen = max(
            eligible,
            key=lambda row: (
                row["target_class_accuracy"],
                row["overall_accuracy"],
                row["mean_true_margin"],
                row["worst_true_margin"],
                -row["beta"],
            ),
        )
        selected[target] = chosen["beta"]
        class_trace.append(
            {
                "class_handle": label,
                "rival_class_handle": classes[int(rivals[target])],
                "selected_beta": chosen["beta"],
                "candidate_evidence": evidence,
            }
        )
    combined = _metrics(
        _apply_local_margin(base_scores, rivals, selected), truth, classes
    )
    global_pass = bool(
        combined["overall_accuracy"] >= baseline["overall_accuracy"]
        and combined["min_class_accuracy"] >= baseline["min_class_accuracy"]
        and all(
            combined["per_class_accuracy"][label]
            >= baseline["per_class_accuracy"][label]
            for label in classes
        )
    )
    before_fallback = selected.copy()
    if not global_pass:
        selected[:] = 0.0
        combined = baseline
    return selected, {
        "schema": "cvs.phase2.d7c_support_selection.v1",
        "selection_scope": "registered_physical_support_deletion_only",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "rival_source": "heldout_d7a_calibrated_class_confusion",
        "beta_candidates": list(candidates),
        "baseline": baseline,
        "per_class_selection": class_trace,
        "combined_before_fallback": _metrics(
            _apply_local_margin(
                base_scores, rivals, before_fallback
            ),
            truth,
            classes,
        ),
        "global_non_degradation_pass": global_pass,
        "fallback_to_beta_zero": not global_pass,
        "combined_final": combined,
    }


def fit_class_conditional_local_boundary(
    base_state: ClassConditionalIQHeadState,
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
    *,
    beta_candidates: Sequence[float] = DEFAULT_BETA_CANDIDATES,
) -> ClassConditionalLocalBoundaryState:
    """Fit D7c before registration from the same physical support as D7a."""

    candidates = _validate_beta_candidates(beta_candidates)
    features, labels = _validate_support(
        base_state,
        artifact,
        support_labels,
        expected_classes=base_state.classes,
    )
    records = _heldout_records_before(base_state, features, labels)
    rivals = _select_rivals(
        records, len(base_state.classes), range(len(base_state.classes))
    )
    beta, selection = _select_betas_before(
        records, rivals, base_state.classes, candidates
    )
    audit = {
        "schema": "cvs.phase2.d7c_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "physical_support_sample_count": len(labels),
        "physical_support_ids_unique": True,
        "parent_received_iq_hashes_unique": True,
        "operator_feature_binding_verified": True,
        "operator_feature_view_seed": 0,
        "d7a_operator_prototype_calibration_reused": True,
        "computation_views_count_as_additional_physical_samples": False,
        "additional_leo_channel_state_generation": False,
        "query_rows_used": 0,
        "query_roles_used": False,
        "query_quota_used": False,
        "old_state_bitwise_locked": False,
        "selection": selection,
    }
    state = ClassConditionalLocalBoundaryState(
        schema="cvs.phase2.d7c_class_conditional_local_boundary.v1",
        base_state=base_state,
        rival_indices=_readonly(rivals, np.int64),
        beta=_readonly(beta, np.float32),
        old_class_count=len(base_state.classes),
        **_state_lineage(artifact, labels),
        strategy_locked_k=None,
        support_audit=audit,
    )
    if state.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise ClassConditionalLocalBoundaryError(
            "D7c persistent state cap exceeded"
        )
    _assert_local_state_contract(state)
    return state


def _assert_d7a_extension_locked(
    parent: ClassConditionalIQHeadState,
    extended: ClassConditionalIQHeadState,
) -> None:
    old_count = len(parent.classes)
    _assert_base_resource_contract(parent)
    if (
        extended.schema != parent.schema
        or extended.registration_generation
        != parent.registration_generation + 1
        or extended.classes[:old_count] != parent.classes
        or extended.class_operators[:old_count] != parent.class_operators
        or not np.array_equal(
            extended.prototypes[:old_count], parent.prototypes
        )
        or extended.calibrations != parent.calibrations
        or extended.feature_dim != parent.feature_dim
        or extended.trainable_parameters != parent.trainable_parameters
        or extended.query_rows_used_for_fit
        != parent.query_rows_used_for_fit
        or extended.query_updates != parent.query_updates
    ):
        raise ClassConditionalLocalBoundaryError(
            "D7a old state is not bitwise locked during registration"
        )
    _assert_base_resource_contract(extended)


def _heldout_records_extension(
    extended: ClassConditionalIQHeadState,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    new_classes: Sequence[str],
) -> list[dict[str, np.ndarray]]:
    width = 2 if min(
        int(np.sum(labels == label)) for label in new_classes
    ) >= 4 else 1
    records: list[dict[str, np.ndarray]] = []
    for fold in _folds(labels, new_classes, width_per_class=width):
        held = np.asarray(fold, dtype=np.int64)
        keep = np.ones(len(labels), dtype=bool)
        keep[held] = False
        prototypes = extended.prototypes.copy()
        for label in new_classes:
            class_index = extended.classes.index(label)
            operator = extended.class_operators[class_index]
            prototypes[class_index] = _prototype(
                features[operator][keep & (labels == label)]
            )
        held_features = {
            operator: rows[held] for operator, rows in features.items()
        }
        records.append(
            {
                "scores": _scores_from_features_and_prototypes(
                    extended, held_features, prototypes
                ),
                "truth": np.asarray(
                    [extended.classes.index(value) for value in labels[held]],
                    dtype=np.int64,
                ),
            }
        )
    return records


def _select_betas_extension(
    parent: ClassConditionalLocalBoundaryState,
    extended: ClassConditionalIQHeadState,
    records: Sequence[Mapping[str, np.ndarray]],
    old_support_scores: np.ndarray,
    old_support_truth: np.ndarray,
    rivals: np.ndarray,
    new_indices: Sequence[int],
    candidates: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    new_scores, new_truth = _combined_record_arrays(records)
    base_beta = np.concatenate(
        [
            parent.beta,
            np.zeros(len(extended.classes) - len(parent.classes)),
        ]
    ).astype(np.float32)
    baseline_new = _metrics(
        _apply_local_margin(new_scores, rivals, base_beta),
        new_truth,
        extended.classes,
    )
    baseline_old = _metrics(
        _apply_local_margin(old_support_scores, rivals, base_beta),
        old_support_truth,
        extended.classes,
    )
    selected = base_beta.copy()
    class_trace: list[dict[str, Any]] = []
    for target in new_indices:
        label = extended.classes[target]
        evidence = []
        for candidate in candidates:
            beta = base_beta.copy()
            beta[target] = candidate
            new_metrics = _metrics(
                _apply_local_margin(new_scores, rivals, beta),
                new_truth,
                extended.classes,
            )
            old_metrics = _metrics(
                _apply_local_margin(old_support_scores, rivals, beta),
                old_support_truth,
                extended.classes,
            )
            eligible = bool(
                new_metrics["per_class_accuracy"][label]
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
                    "beta": candidate,
                    "new_support": new_metrics,
                    "old_support_intrusion_guard": old_metrics,
                    "eligible": eligible,
                }
            )
        eligible_rows = [row for row in evidence if row["eligible"]]
        chosen = max(
            eligible_rows,
            key=lambda row: (
                row["new_support"]["per_class_accuracy"][label],
                row["new_support"]["overall_accuracy"],
                row["old_support_intrusion_guard"]["overall_accuracy"],
                row["new_support"]["mean_true_margin"],
                -row["beta"],
            ),
        )
        selected[target] = chosen["beta"]
        class_trace.append(
            {
                "class_handle": label,
                "rival_class_handle": extended.classes[
                    int(rivals[target])
                ],
                "selected_beta": chosen["beta"],
                "candidate_evidence": evidence,
            }
        )
    combined_new = _metrics(
        _apply_local_margin(new_scores, rivals, selected),
        new_truth,
        extended.classes,
    )
    combined_old = _metrics(
        _apply_local_margin(old_support_scores, rivals, selected),
        old_support_truth,
        extended.classes,
    )
    global_pass = bool(
        combined_new["overall_accuracy"] >= baseline_new["overall_accuracy"]
        and combined_new["min_class_accuracy"]
        >= baseline_new["min_class_accuracy"]
        and all(
            combined_new["per_class_accuracy"][label]
            >= baseline_new["per_class_accuracy"][label]
            for label in extended.classes
            if label in combined_new["per_class_accuracy"]
        )
        and combined_old["overall_accuracy"] >= baseline_old["overall_accuracy"]
        and all(
            combined_old["per_class_accuracy"][label]
            >= baseline_old["per_class_accuracy"][label]
            for label in parent.classes
        )
    )
    before_fallback = selected.copy()
    if not global_pass:
        selected = base_beta
        combined_new = baseline_new
        combined_old = baseline_old
    return selected, {
        "schema": "cvs.phase2.d7c_extension_selection.v1",
        "selection_scope": "registered_physical_support_only",
        "old_d7a_and_d7c_state_during_selection": "bitwise_locked",
        "new_support_protocol": "physical_support_deletion",
        "old_support_protocol": "fixed_full_support_intrusion_guard",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "beta_candidates": list(candidates),
        "baseline_new": baseline_new,
        "baseline_old": baseline_old,
        "per_new_class_selection": class_trace,
        "combined_before_fallback_beta": before_fallback.tolist(),
        "global_non_degradation_pass": global_pass,
        "fallback_new_beta_to_zero": not global_pass,
        "combined_final_new": combined_new,
        "combined_final_old": combined_old,
    }


def extend_class_conditional_local_boundary(
    parent: ClassConditionalLocalBoundaryState,
    extended_base_state: ClassConditionalIQHeadState,
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
    *,
    beta_candidates: Sequence[float] = DEFAULT_BETA_CANDIDATES,
) -> ClassConditionalLocalBoundaryState:
    """Append new D7c rivals/betas while bitwise locking every old state."""

    if not isinstance(parent, ClassConditionalLocalBoundaryState):
        raise ClassConditionalLocalBoundaryError("parent D7c state is required")
    _assert_local_state_contract(parent)
    _assert_d7a_extension_locked(parent.base_state, extended_base_state)
    candidates = _validate_beta_candidates(beta_candidates)
    features, labels = _validate_support(
        extended_base_state,
        artifact,
        support_labels,
        expected_classes=extended_base_state.classes,
    )
    old_classes = parent.classes
    new_classes = extended_base_state.classes[len(old_classes) :]
    if not new_classes:
        raise ClassConditionalLocalBoundaryError(
            "D7c registration received no absent classes"
        )
    if not set(old_classes).issubset(set(labels.tolist())):
        raise ClassConditionalLocalBoundaryError(
            "registration support must retain every locked old class"
        )
    parent_old_rows = sorted(
        zip(
            parent.support_labels,
            parent.support_physical_sample_ids,
            parent.support_parent_received_iq_sha256,
        )
    )
    incoming_old_rows = sorted(
        (
            str(label),
            artifact.physical_sample_ids[index],
            artifact.parent_received_iq_sha256[index],
        )
        for index, label in enumerate(labels.tolist())
        if label in set(old_classes)
    )
    incoming_old_fingerprints = _binding_fingerprints(
        artifact, labels, include_classes=set(old_classes)
    )
    if (
        incoming_old_rows != parent_old_rows
        or incoming_old_fingerprints
        != parent.support_binding_fingerprints
    ):
        raise ClassConditionalLocalBoundaryError(
            "registration old support lineage or feature binding drift"
        )
    records = _heldout_records_extension(
        extended_base_state, features, labels, new_classes
    )
    rivals = np.concatenate(
        [
            parent.rival_indices.copy(),
            np.zeros(len(new_classes), dtype=np.int64),
        ]
    )
    new_indices = tuple(
        range(len(old_classes), len(extended_base_state.classes))
    )
    selected_new_rivals = _select_rivals(
        records, len(extended_base_state.classes), new_indices
    )
    rivals[list(new_indices)] = selected_new_rivals[list(new_indices)]
    old_mask = np.isin(labels, np.asarray(old_classes))
    old_features = {
        operator: rows[old_mask] for operator, rows in features.items()
    }
    old_scores = _scores_from_features_and_prototypes(
        extended_base_state,
        old_features,
        extended_base_state.prototypes,
    )
    old_truth = np.asarray(
        [
            extended_base_state.classes.index(value)
            for value in labels[old_mask]
        ],
        dtype=np.int64,
    )
    beta, selection = _select_betas_extension(
        parent,
        extended_base_state,
        records,
        old_scores,
        old_truth,
        rivals,
        new_indices,
        candidates,
    )
    audit = {
        "schema": "cvs.phase2.d7c_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "physical_support_sample_count": len(labels),
        "physical_support_ids_unique": True,
        "parent_received_iq_hashes_unique": True,
        "operator_feature_binding_verified": True,
        "parent_old_support_lineage_verified": True,
        "old_state_bitwise_locked": True,
        "old_state_update_count": 0,
        "old_rivals_can_reference_new_classes": False,
        "new_class_handles": list(new_classes),
        "new_rival_class_handles": [
            extended_base_state.classes[int(rivals[index])]
            for index in new_indices
        ],
        "query_rows_used": 0,
        "query_roles_used": False,
        "query_quota_used": False,
        "selection": selection,
    }
    state = ClassConditionalLocalBoundaryState(
        schema=parent.schema,
        base_state=extended_base_state,
        rival_indices=_readonly(rivals, np.int64),
        beta=_readonly(beta, np.float32),
        old_class_count=parent.old_class_count,
        **_state_lineage(artifact, labels),
        strategy_locked_k=None,
        support_audit=audit,
    )
    old_count = len(old_classes)
    if (
        not np.array_equal(
            state.rival_indices[:old_count], parent.rival_indices
        )
        or not np.array_equal(state.beta[:old_count], parent.beta)
        or np.any(state.rival_indices[:old_count] >= old_count)
    ):
        raise ClassConditionalLocalBoundaryError(
            "parent old D7c state mutation"
        )
    if state.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise ClassConditionalLocalBoundaryError(
            "D7c persistent state cap exceeded"
        )
    _assert_local_state_contract(state)
    return state


def lock_k10_class_conditional_local_boundary_strategy(
    state: ClassConditionalLocalBoundaryState,
) -> ClassConditionalLocalBoundaryState:
    """Lock operator/rival/beta policy from exactly K10 physical support."""

    if not isinstance(state, ClassConditionalLocalBoundaryState):
        raise ClassConditionalLocalBoundaryError("D7c state is required")
    _assert_local_state_contract(state)
    counts = {
        label: state.support_labels.count(label) for label in state.classes
    }
    if set(counts.values()) != {10}:
        raise ClassConditionalLocalBoundaryError(
            "D7c strategy lock requires exactly K10 per class"
        )
    audit = dict(state.support_audit)
    audit.update(
        {
            "strategy_locked_from_k": 10,
            "operator_reselected_for_lower_k": False,
            "rival_reselected_for_lower_k": False,
            "beta_reselected_for_lower_k": False,
        }
    )
    return replace(state, strategy_locked_k=10, support_audit=audit)


def rebuild_from_locked_k10_strategy(
    locked_state: ClassConditionalLocalBoundaryState,
    artifact: ValidatedOperatorFeatureArtifact,
    support_labels: Sequence[str],
    *,
    expected_k: int,
) -> ClassConditionalLocalBoundaryState:
    """For K1/K5/K10/K20 rebuild prototypes only; never reselect policy."""

    if (
        not isinstance(locked_state, ClassConditionalLocalBoundaryState)
        or locked_state.strategy_locked_k != 10
        or int(expected_k) not in {1, 5, 10, 20}
    ):
        raise ClassConditionalLocalBoundaryError(
            "D7c locked K10 strategy is required"
        )
    _assert_local_state_contract(locked_state)
    labels = tuple(str(value) for value in support_labels)
    try:
        validate_operator_feature_artifact(
            artifact, expected_operator_ids=OPERATORS
        )
        rebuilt_base = rebuild_prototypes_from_locked_policy(
            locked_state.base_state,
            artifact,
            labels,
            expected_k=int(expected_k),
            locked_from_k=10,
        )
    except Exception as exc:
        raise ClassConditionalLocalBoundaryError(
            "D7c locked-policy prototype rebuild drift"
        ) from exc
    audit = {
        "schema": "cvs.phase2.d7c_locked_policy_rebuild.v1",
        "fit_scope": "registered_physical_support_prototype_rebuild_only",
        "strategy_locked_from_k": 10,
        "prototype_rebuild_k": int(expected_k),
        "operator_reselected": False,
        "rival_reselected": False,
        "beta_reselected": False,
        "calibration_reselected": False,
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "operator_feature_binding_verified": True,
        "additional_leo_channel_state_generation": False,
    }
    rebuilt = ClassConditionalLocalBoundaryState(
        schema=locked_state.schema,
        base_state=rebuilt_base,
        rival_indices=_readonly(
            locked_state.rival_indices.copy(), np.int64
        ),
        beta=_readonly(locked_state.beta.copy(), np.float32),
        old_class_count=locked_state.old_class_count,
        **_state_lineage(artifact, labels),
        strategy_locked_k=10,
        support_audit=audit,
    )
    if (
        rebuilt.base_state.class_operators
        != locked_state.base_state.class_operators
        or rebuilt.base_state.calibrations
        != locked_state.base_state.calibrations
        or not np.array_equal(
            rebuilt.rival_indices, locked_state.rival_indices
        )
        or not np.array_equal(rebuilt.beta, locked_state.beta)
        or rebuilt.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES
    ):
        raise ClassConditionalLocalBoundaryError(
            "D7c locked strategy mutated during prototype rebuild"
        )
    _assert_local_state_contract(rebuilt)
    return rebuilt


def predict_class_conditional_local_boundary(
    state: ClassConditionalLocalBoundaryState,
    received_iq: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
) -> ClassConditionalLocalPrediction:
    """Score samplewise queries; callback never receives another query row."""

    _assert_local_state_contract(state)
    artifact = build_validated_operator_feature_artifact(
        received_iq,
        feature_extractor=feature_extractor,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
        operator_ids=state.base_state.used_operators,
    )
    features = validate_operator_feature_artifact(
        artifact,
        expected_operator_ids=state.base_state.used_operators,
    )
    base_scores = _score_class_conditional_features(
        state.base_state, features
    )
    scores = _readonly(
        _apply_local_margin(
            base_scores, state.rival_indices, state.beta
        ),
        np.float32,
    )
    indices = np.argmax(scores, axis=1)
    return ClassConditionalLocalPrediction(
        labels=tuple(state.classes[int(index)] for index in indices),
        scores=scores,
        base_scores=base_scores,
        operators_computed=state.base_state.used_operators,
    )


def public_query_interface_is_oracle_free() -> bool:
    forbidden = {"label", "truth", "role", "quota", "assignment", "graph"}
    return not any(
        token in parameter.lower()
        for parameter in inspect.signature(
            predict_class_conditional_local_boundary
        ).parameters
        for token in forbidden
    )


__all__ = [
    "DEFAULT_BETA_CANDIDATES",
    "ClassConditionalLocalBoundaryError",
    "ClassConditionalLocalBoundaryState",
    "ClassConditionalLocalPrediction",
    "extend_class_conditional_local_boundary",
    "fit_class_conditional_local_boundary",
    "lock_k10_class_conditional_local_boundary_strategy",
    "predict_class_conditional_local_boundary",
    "public_query_interface_is_oracle_free",
    "rebuild_from_locked_k10_strategy",
]
