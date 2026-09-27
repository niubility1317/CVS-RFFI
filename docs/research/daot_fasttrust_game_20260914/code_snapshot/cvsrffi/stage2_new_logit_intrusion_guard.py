"""D13 support-only class-conditional new-logit intrusion guard.

The guard preserves every old-class cosine score exactly.  Registration only
adds new prototypes and subtracts one support-calibrated scalar penalty from
each new-class score.  The module has no optimizer, query fitting, role/quota
input, source/clean input, batch assignment, or dense query graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_joint_residual_logit_head import (
    RuntimeAuthorizedFeatureArtifact,
)


EPS = 1.0e-8
SCHEMA = "cvs.phase2.new_logit_intrusion_guard.v1"
MAX_STATE_BYTES = 256 * 1024


class NewLogitIntrusionGuardError(ValueError):
    """Raised when the D13 support-only contract fails closed."""


@dataclass(frozen=True)
class IntrusionGuardHyperparameters:
    candidate_id: str
    mode: str = "constant"
    old_risk_quantile: float = 0.90
    new_room_quantile: float = 0.25
    safety: float = 0.01
    cap: float = 0.20
    new_floor_margin: float = 0.01
    hinge_strength: float = 0.0
    force_zero: bool = False


@dataclass(frozen=True)
class NewLogitIntrusionGuardState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    new_logit_penalties: np.ndarray
    hinge_thresholds: np.ndarray
    hinge_strengths: np.ndarray
    hyperparameters: IntrusionGuardHyperparameters
    feature_dim: int
    k_shot: int
    old_class_count: int
    registration_generation: int
    resource: Mapping[str, Any]
    support_feature_artifact_sha256: str
    support_selection_sha256: str
    sealed_runtime_sha256: str
    feature_code_sha256: str
    sealed_phase1_checkpoint_sha256: str
    operator_id: str
    view_seed: int
    calibration_diagnostics: tuple[Mapping[str, Any], ...]
    state_content_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "prototypes",
            "new_logit_penalties",
            "hinge_thresholds",
            "hinge_strengths",
        ):
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            immutable = np.frombuffer(source.tobytes(), dtype=np.float32).reshape(
                source.shape
            )
            object.__setattr__(self, name, immutable)
        computed = _state_content_sha256(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise NewLogitIntrusionGuardError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterGuardFitResult:
    before_state: NewLogitIntrusionGuardState
    after_state: NewLogitIntrusionGuardState
    trace: tuple[dict[str, Any], ...]


def _validate_hyperparameters(value: IntrusionGuardHyperparameters) -> None:
    if (
        not value.candidate_id
        or value.mode not in {"constant", "hinge_margin"}
        or not (0.0 <= value.old_risk_quantile <= 1.0)
        or not (0.0 <= value.new_room_quantile <= 1.0)
        or not np.isfinite(value.safety)
        or value.safety < 0.0
        or not np.isfinite(value.cap)
        or value.cap < 0.0
        or not np.isfinite(value.new_floor_margin)
        or value.new_floor_margin < 0.0
        or not np.isfinite(value.hinge_strength)
        or value.hinge_strength < 0.0
        or (value.mode == "constant" and value.hinge_strength != 0.0)
        or (
            value.mode == "hinge_margin"
            and not value.force_zero
            and not (0.0 < value.hinge_strength <= 1.0)
        )
        or (
            value.force_zero
            and any(
                (
                    value.safety,
                    value.cap,
                    value.new_floor_margin,
                    value.hinge_strength,
                )
            )
        )
    ):
        raise NewLogitIntrusionGuardError("hyperparameter drift")


def _artifact_rows(value: RuntimeAuthorizedFeatureArtifact) -> np.ndarray:
    if not isinstance(value, RuntimeAuthorizedFeatureArtifact):
        raise NewLogitIntrusionGuardError(
            "ordinary feature mapping/array forbidden; authorized artifact required"
        )
    return value.features


def _validate_support(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = _artifact_rows(artifact)
    label_rows = np.asarray(labels).astype(str)
    rank_rows = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(label_rows) or len(rows) != len(rank_rows):
        raise NewLogitIntrusionGuardError("support alignment drift")
    classes, counts = np.unique(label_rows, return_counts=True)
    if (
        int(k_shot) < 1
        or len(classes) < 2
        or set(counts.tolist()) != {int(k_shot)}
        or any(
            set(rank_rows[label_rows == label].tolist()) != set(range(int(k_shot)))
            for label in classes
        )
    ):
        raise NewLogitIntrusionGuardError("strict physical K-shot support drift")
    return rows, label_rows, rank_rows, tuple(sorted(classes.tolist()))


def _validate_old_lineage_exact_reuse(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: Sequence[str],
) -> None:
    def keyed(
        artifact: RuntimeAuthorizedFeatureArtifact,
        labels: np.ndarray,
        ranks: np.ndarray,
        allowed: set[str],
    ) -> dict[tuple[str, int], tuple[str, str, str]]:
        return {
            (str(labels[index]), int(ranks[index])): (
                artifact.physical_sample_ids[index],
                artifact.parent_received_iq_sha256[index],
                artifact.per_row_feature_sha256[index],
            )
            for index in range(len(labels))
            if str(labels[index]) in allowed
        }

    allowed = set(old_classes)
    before = keyed(before_artifact, before_labels, before_ranks, allowed)
    after = keyed(after_artifact, after_labels, after_ranks, allowed)
    if before != after:
        raise NewLogitIntrusionGuardError(
            "after old support lineage exact-reuse lock failed"
        )


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def _prototypes(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    normalized = _normalize(rows)
    values = np.stack(
        [np.mean(normalized[labels == label], axis=0) for label in classes]
    )
    return _normalize(values).astype(np.float32)


def _support_selection_sha256(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    selection: np.ndarray | None = None,
) -> str:
    selected = (
        np.ones(len(labels), dtype=bool)
        if selection is None
        else np.asarray(selection, dtype=bool)
    )
    if len(selected) != len(labels) or len(labels) != len(artifact.features):
        raise NewLogitIntrusionGuardError("support selection alignment drift")
    rows = [
        {
            "label": str(labels[index]),
            "rank": int(ranks[index]),
            "physical_sample_id": artifact.physical_sample_ids[index],
            "parent_received_iq_sha256": artifact.parent_received_iq_sha256[index],
            "feature_sha256": artifact.per_row_feature_sha256[index],
        }
        for index in np.flatnonzero(selected)
    ]
    return hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _quantile(values: np.ndarray, q: float) -> float:
    if not len(values) or not np.isfinite(values).all():
        raise NewLogitIntrusionGuardError("non-finite or empty calibration margin")
    return float(np.quantile(values, q, method="linear"))


def _calibrate_new_penalties(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    *,
    old_class_count: int,
    hyperparameters: IntrusionGuardHyperparameters,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[dict[str, Any], ...],
]:
    _validate_hyperparameters(hyperparameters)
    penalties = np.zeros(len(classes), dtype=np.float32)
    thresholds = np.zeros(len(classes), dtype=np.float32)
    strengths = np.zeros(len(classes), dtype=np.float32)
    scores = _normalize(rows) @ prototypes.T
    lookup = {label: index for index, label in enumerate(classes)}
    old_mask = np.isin(labels, classes[:old_class_count])
    old_targets = np.asarray([lookup[label] for label in labels[old_mask]])
    old_scores = scores[old_mask, :old_class_count]
    old_max_indices = np.argmax(old_scores, axis=1)
    before_correct = old_max_indices == old_targets
    if not np.any(before_correct):
        raise NewLogitIntrusionGuardError(
            "no Before-correct old support available for intrusion calibration"
        )
    old_max_scores = np.max(old_scores[before_correct], axis=1)
    diagnostics = []
    for class_index in range(old_class_count, len(classes)):
        label = classes[class_index]
        old_intrusion_margin = (
            scores[old_mask, class_index][before_correct] - old_max_scores
        )
        new_mask = labels == label
        other = np.ones(len(classes), dtype=bool)
        other[class_index] = False
        new_positive_margin = (
            scores[new_mask, class_index] - np.max(scores[new_mask][:, other], axis=1)
        )
        old_risk = _quantile(
            old_intrusion_margin, hyperparameters.old_risk_quantile
        )
        new_room = _quantile(
            new_positive_margin, hyperparameters.new_room_quantile
        )
        requested = max(0.0, old_risk + hyperparameters.safety)
        room_bound = max(0.0, new_room - hyperparameters.new_floor_margin)
        bounded = (
            0.0
            if hyperparameters.force_zero
            else min(requested, hyperparameters.cap, room_bound)
        )
        protection_shortfall = max(0.0, requested - bounded)
        feasible = protection_shortfall <= 1.0e-12
        if hyperparameters.mode == "constant":
            penalties[class_index] = np.float32(bounded)
        else:
            thresholds[class_index] = np.float32(bounded)
            strengths[class_index] = np.float32(
                0.0 if bounded <= 0.0 else hyperparameters.hinge_strength
            )
        diagnostics.append(
            {
                "class_handle": label,
                "old_risk_quantile_value": old_risk,
                "new_room_quantile_value": new_room,
                "quantile_method": "linear",
                "requested_delta": requested,
                "room_bound": room_bound,
                "cap": hyperparameters.cap,
                "protection_feasible": bool(feasible),
                "protection_shortfall": protection_shortfall,
                "mode": hyperparameters.mode,
                "delta": float(penalties[class_index]),
                "hinge_threshold": float(thresholds[class_index]),
                "hinge_strength": float(strengths[class_index]),
                "old_margin_count": int(len(old_intrusion_margin)),
                "old_intrusion_count": int(
                    np.sum(old_intrusion_margin > 0.0)
                ),
                "before_correct_old_margin_only": True,
                "new_margin_count": int(len(new_positive_margin)),
            }
        )
    return penalties, thresholds, strengths, tuple(diagnostics)


def _score_numpy(
    rows: np.ndarray,
    state: NewLogitIntrusionGuardState,
) -> np.ndarray:
    _validate_state(state)
    base = _normalize(rows) @ state.prototypes.T
    guarded = np.array(base, dtype=np.float32, copy=True)
    new_slice = slice(state.old_class_count, None)
    guarded[:, new_slice] -= state.new_logit_penalties[new_slice][None, :]
    if np.any(state.hinge_strengths[new_slice] > 0.0):
        old_max = np.max(base[:, : state.old_class_count], axis=1, keepdims=True)
        new_margin = base[:, new_slice] - old_max
        correction = state.hinge_strengths[new_slice][None, :] * np.maximum(
            state.hinge_thresholds[new_slice][None, :] - new_margin,
            0.0,
        )
        correction = np.minimum(correction, state.hyperparameters.cap)
        guarded[:, new_slice] -= correction
    if not np.array_equal(
        guarded[:, : state.old_class_count],
        base[:, : state.old_class_count],
    ):
        raise NewLogitIntrusionGuardError("old score bitwise freeze violated")
    return guarded


def _state_content_sha256(state: NewLogitIntrusionGuardState) -> str:
    digest = hashlib.sha256()
    digest.update(state.schema.encode("utf-8"))
    digest.update(state.candidate_id.encode("utf-8"))
    digest.update(json.dumps(state.classes, separators=(",", ":")).encode("utf-8"))
    for value in (
        state.prototypes,
        state.new_logit_penalties,
        state.hinge_thresholds,
        state.hinge_strengths,
    ):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        json.dumps(
            {
                "feature_dim": state.feature_dim,
                "k_shot": state.k_shot,
                "old_class_count": state.old_class_count,
                "registration_generation": state.registration_generation,
                "support_feature_artifact_sha256": (
                    state.support_feature_artifact_sha256
                ),
                "support_selection_sha256": state.support_selection_sha256,
                "sealed_runtime_sha256": state.sealed_runtime_sha256,
                "feature_code_sha256": state.feature_code_sha256,
                "sealed_phase1_checkpoint_sha256": (
                    state.sealed_phase1_checkpoint_sha256
                ),
                "operator_id": state.operator_id,
                "view_seed": state.view_seed,
                "hyperparameters": {
                    "candidate_id": state.hyperparameters.candidate_id,
                    "mode": state.hyperparameters.mode,
                    "old_risk_quantile": state.hyperparameters.old_risk_quantile,
                    "new_room_quantile": state.hyperparameters.new_room_quantile,
                    "safety": state.hyperparameters.safety,
                    "cap": state.hyperparameters.cap,
                    "new_floor_margin": state.hyperparameters.new_floor_margin,
                    "hinge_strength": state.hyperparameters.hinge_strength,
                    "force_zero": state.hyperparameters.force_zero,
                },
                "resource": dict(state.resource),
                "calibration_diagnostics": [
                    dict(value) for value in state.calibration_diagnostics
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _validate_state(state: NewLogitIntrusionGuardState) -> None:
    total_bytes = int(
        state.prototypes.nbytes
        + state.new_logit_penalties.nbytes
        + state.hinge_thresholds.nbytes
        + state.hinge_strengths.nbytes
    )
    incremental_bytes = int(
        state.new_logit_penalties[state.old_class_count :].nbytes
        + state.hinge_thresholds[state.old_class_count :].nbytes
        + state.hinge_strengths[state.old_class_count :].nbytes
    )
    if (
        state.schema != SCHEMA
        or state.prototypes.shape != (len(state.classes), state.feature_dim)
        or state.new_logit_penalties.shape != (len(state.classes),)
        or state.hinge_thresholds.shape != (len(state.classes),)
        or state.hinge_strengths.shape != (len(state.classes),)
        or np.any(state.new_logit_penalties[: state.old_class_count] != 0.0)
        or np.any(state.hinge_thresholds[: state.old_class_count] != 0.0)
        or np.any(state.hinge_strengths[: state.old_class_count] != 0.0)
        or np.any(state.new_logit_penalties < 0.0)
        or np.any(state.hinge_thresholds < 0.0)
        or np.any(state.hinge_strengths < 0.0)
        or total_bytes > MAX_STATE_BYTES
        or int(state.resource.get("trainable_parameters", -1)) != 0
        or int(state.resource.get("adapt_epochs", -1)) != 0
        or int(state.resource.get("persistent_state_bytes", -1)) != total_bytes
        or int(state.resource.get("incremental_guard_state_bytes", -1))
        != incremental_bytes
        or state.state_content_sha256 != _state_content_sha256(state)
        or any(
            len(value) != 64
            for value in (
                state.support_feature_artifact_sha256,
                state.support_selection_sha256,
                state.sealed_runtime_sha256,
                state.feature_code_sha256,
                state.sealed_phase1_checkpoint_sha256,
            )
        )
        or not state.operator_id
    ):
        raise NewLogitIntrusionGuardError("state content/resource/binding drift")


def _make_state(
    artifact: RuntimeAuthorizedFeatureArtifact,
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    penalties: np.ndarray,
    thresholds: np.ndarray,
    strengths: np.ndarray,
    diagnostics: tuple[Mapping[str, Any], ...],
    *,
    k_shot: int,
    old_class_count: int,
    registration_generation: int,
    hyperparameters: IntrusionGuardHyperparameters,
    support_selection_sha256: str,
) -> NewLogitIntrusionGuardState:
    total_bytes = int(
        prototypes.nbytes + penalties.nbytes + thresholds.nbytes + strengths.nbytes
    )
    incremental_bytes = int(
        penalties[old_class_count:].nbytes
        + thresholds[old_class_count:].nbytes
        + strengths[old_class_count:].nbytes
    )
    return NewLogitIntrusionGuardState(
        schema=SCHEMA,
        candidate_id=hyperparameters.candidate_id,
        classes=classes,
        prototypes=prototypes,
        new_logit_penalties=penalties,
        hinge_thresholds=thresholds,
        hinge_strengths=strengths,
        hyperparameters=hyperparameters,
        feature_dim=int(rows.shape[1]),
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
        registration_generation=int(registration_generation),
        resource={
            "trainable_parameters": 0,
            "adapt_epochs": 0,
            "closed_form_solve_count": 1 if registration_generation else 0,
            "persistent_state_bytes": total_bytes,
            "incremental_guard_state_bytes": incremental_bytes,
            "prototype_cosine_mac_per_sample": int(len(classes) * rows.shape[1]),
            "guard_subtractions_per_sample": int(len(classes) - old_class_count),
            "guard_relu_per_sample": (
                int(len(classes) - old_class_count)
                if hyperparameters.mode == "hinge_margin"
                else 0
            ),
            "backbone_forwards_per_physical_sample": 1,
            "dense_query_graph": False,
        },
        support_feature_artifact_sha256=artifact.artifact_sha256,
        support_selection_sha256=support_selection_sha256,
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=artifact.sealed_phase1_checkpoint_sha256,
        operator_id=artifact.operator_id,
        view_seed=artifact.view_seed,
        calibration_diagnostics=diagnostics,
    )


def _fit_states(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    *,
    k_shot: int,
    hyperparameters: IntrusionGuardHyperparameters,
) -> BeforeAfterGuardFitResult:
    before_rows, old_labels, old_rank_rows, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=k_shot
    )
    after_rows, joint_labels, joint_rank_rows, joint_classes_found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=k_shot
    )
    if (
        not set(old_classes) < set(joint_classes_found)
        or before_artifact.sealed_runtime_sha256
        != after_artifact.sealed_runtime_sha256
        or before_artifact.feature_code_sha256 != after_artifact.feature_code_sha256
        or before_artifact.sealed_phase1_checkpoint_sha256
        != after_artifact.sealed_phase1_checkpoint_sha256
        or before_artifact.operator_id != after_artifact.operator_id
        or before_artifact.view_seed != after_artifact.view_seed
    ):
        raise NewLogitIntrusionGuardError("before/after class or runtime binding drift")
    _validate_old_lineage_exact_reuse(
        before_artifact,
        old_labels,
        old_rank_rows,
        after_artifact,
        joint_labels,
        joint_rank_rows,
        old_classes,
    )
    joint_classes = old_classes + tuple(
        sorted(set(joint_classes_found) - set(old_classes))
    )
    before_prototypes = _prototypes(before_rows, old_labels, old_classes)
    after_prototypes = _prototypes(after_rows, joint_labels, joint_classes)
    if not np.array_equal(
        before_prototypes, after_prototypes[: len(old_classes)]
    ):
        raise NewLogitIntrusionGuardError("old prototype bitwise freeze violated")
    before_penalties = np.zeros(len(old_classes), dtype=np.float32)
    before_thresholds = np.zeros(len(old_classes), dtype=np.float32)
    before_strengths = np.zeros(len(old_classes), dtype=np.float32)
    before_state = _make_state(
        before_artifact,
        before_rows,
        old_labels,
        old_classes,
        before_prototypes,
        before_penalties,
        before_thresholds,
        before_strengths,
        (),
        k_shot=k_shot,
        old_class_count=len(old_classes),
        registration_generation=0,
        hyperparameters=hyperparameters,
        support_selection_sha256=_support_selection_sha256(
            before_artifact, old_labels, old_rank_rows
        ),
    )
    penalties, thresholds, strengths, diagnostics = _calibrate_new_penalties(
        after_rows,
        joint_labels,
        joint_classes,
        after_prototypes,
        old_class_count=len(old_classes),
        hyperparameters=hyperparameters,
    )
    after_state = _make_state(
        after_artifact,
        after_rows,
        joint_labels,
        joint_classes,
        after_prototypes,
        penalties,
        thresholds,
        strengths,
        diagnostics,
        k_shot=k_shot,
        old_class_count=len(old_classes),
        registration_generation=1,
        hyperparameters=hyperparameters,
        support_selection_sha256=_support_selection_sha256(
            after_artifact, joint_labels, joint_rank_rows
        ),
    )
    trace = (
        {
            "phase": "before_closed_form_fit",
            "candidate_id": hyperparameters.candidate_id,
            "k_shot": int(k_shot),
            "class_count": len(old_classes),
            "trainable_parameters": 0,
            "adapt_epochs": 0,
            "penalties": [0.0] * len(old_classes),
        },
        {
            "phase": "after_closed_form_fit",
            "candidate_id": hyperparameters.candidate_id,
            "k_shot": int(k_shot),
            "class_count": len(joint_classes),
            "trainable_parameters": 0,
            "adapt_epochs": 0,
            "new_class_penalties": penalties[len(old_classes) :].tolist(),
            "new_class_hinge_thresholds": thresholds[len(old_classes) :].tolist(),
            "new_class_hinge_strengths": strengths[len(old_classes) :].tolist(),
            "calibration_diagnostics": [dict(value) for value in diagnostics],
        },
    )
    return BeforeAfterGuardFitResult(
        before_state=before_state,
        after_state=after_state,
        trace=trace,
    )


def fit_before_after_locked(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
    hyperparameters: IntrusionGuardHyperparameters,
) -> BeforeAfterGuardFitResult:
    """Fit prototype states and support-only new-class penalties."""

    return _fit_states(
        before_artifact,
        np.asarray(before_labels).astype(str),
        np.asarray(before_ranks, dtype=np.int64),
        after_artifact,
        np.asarray(after_labels).astype(str),
        np.asarray(after_ranks, dtype=np.int64),
        k_shot=k_shot,
        hyperparameters=hyperparameters,
    )


def _leave_two_out_masks(labels: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, ...]:
    if set(np.unique(ranks).tolist()) != set(range(10)):
        raise NewLogitIntrusionGuardError("joint L2O requires strict K10 ranks")
    masks = []
    for first in range(0, 10, 2):
        held = np.isin(ranks, (first, first + 1))
        if any(np.sum(held & (labels == label)) != 2 for label in np.unique(labels)):
            raise NewLogitIntrusionGuardError("leave-two-out physical fold drift")
        masks.append(held)
    return tuple(masks)


def _prediction_metrics(
    truth: np.ndarray, predictions: np.ndarray, classes: Sequence[str]
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean(predictions[truth == label] == label))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean(predictions == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _aggregate_metrics(
    folds: Sequence[Mapping[str, Any]], classes: Sequence[str], key: str
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean([row[key]["per_class_accuracy"][label] for row in folds]))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean([row[key]["overall_accuracy"] for row in folds])),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _harmonic(old: float, new: float) -> float:
    return 0.0 if old + new <= 0.0 else 2.0 * old * new / (old + new)


def _fit_fold_states(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    old_rows: np.ndarray,
    old_labels: np.ndarray,
    old_ranks: np.ndarray,
    old_classes: tuple[str, ...],
    train_old: np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    joint_rows: np.ndarray,
    joint_labels: np.ndarray,
    joint_ranks: np.ndarray,
    joint_classes: tuple[str, ...],
    train_joint: np.ndarray,
    *,
    hyperparameters: IntrusionGuardHyperparameters,
) -> BeforeAfterGuardFitResult:
    if any(
        np.sum(train_old & (old_labels == label)) != 8 for label in old_classes
    ) or any(
        np.sum(train_joint & (joint_labels == label)) != 8
        for label in joint_classes
    ):
        raise NewLogitIntrusionGuardError("fold-train K8 selection drift")
    old_selected_rows = old_rows[train_old]
    old_selected_labels = old_labels[train_old]
    joint_selected_rows = joint_rows[train_joint]
    joint_selected_labels = joint_labels[train_joint]
    before_prototypes = _prototypes(
        old_selected_rows, old_selected_labels, old_classes
    )
    after_prototypes = _prototypes(
        joint_selected_rows, joint_selected_labels, joint_classes
    )
    if not np.array_equal(
        before_prototypes, after_prototypes[: len(old_classes)]
    ):
        raise NewLogitIntrusionGuardError(
            "fold old prototype bitwise freeze violated"
        )
    before_zero = np.zeros(len(old_classes), dtype=np.float32)
    before_state = _make_state(
        before_artifact,
        old_selected_rows,
        old_selected_labels,
        old_classes,
        before_prototypes,
        before_zero,
        before_zero,
        before_zero,
        (),
        k_shot=8,
        old_class_count=len(old_classes),
        registration_generation=0,
        hyperparameters=hyperparameters,
        support_selection_sha256=_support_selection_sha256(
            before_artifact, old_labels, old_ranks, train_old
        ),
    )
    penalties, thresholds, strengths, diagnostics = _calibrate_new_penalties(
        joint_selected_rows,
        joint_selected_labels,
        joint_classes,
        after_prototypes,
        old_class_count=len(old_classes),
        hyperparameters=hyperparameters,
    )
    after_state = _make_state(
        after_artifact,
        joint_selected_rows,
        joint_selected_labels,
        joint_classes,
        after_prototypes,
        penalties,
        thresholds,
        strengths,
        diagnostics,
        k_shot=8,
        old_class_count=len(old_classes),
        registration_generation=1,
        hyperparameters=hyperparameters,
        support_selection_sha256=_support_selection_sha256(
            after_artifact, joint_labels, joint_ranks, train_joint
        ),
    )
    return BeforeAfterGuardFitResult(
        before_state=before_state,
        after_state=after_state,
        trace=(
            {
                "phase": "joint_l2o_before_closed_form",
                "candidate_id": hyperparameters.candidate_id,
                "support_selection_sha256": (
                    before_state.support_selection_sha256
                ),
            },
            {
                "phase": "joint_l2o_after_closed_form",
                "candidate_id": hyperparameters.candidate_id,
                "support_selection_sha256": (
                    after_state.support_selection_sha256
                ),
                "new_class_penalties": penalties[len(old_classes) :].tolist(),
                "new_class_hinge_thresholds": thresholds[
                    len(old_classes) :
                ].tolist(),
                "new_class_hinge_strengths": strengths[
                    len(old_classes) :
                ].tolist(),
            },
        ),
    )


def evaluate_joint_leave_two_out(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    hyperparameters: IntrusionGuardHyperparameters,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Joint strict-K10 L2O with old/new each held2 over all registered classes."""

    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=10
    )
    joint_rows, joint_labels, joint_ranks, joint_classes_found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=10
    )
    if not set(old_classes) < set(joint_classes_found):
        raise NewLogitIntrusionGuardError("joint L2O class registration drift")
    _validate_old_lineage_exact_reuse(
        before_artifact,
        old_labels,
        old_ranks,
        after_artifact,
        joint_labels,
        joint_ranks,
        old_classes,
    )
    joint_classes = old_classes + tuple(
        sorted(set(joint_classes_found) - set(old_classes))
    )
    new_classes = joint_classes[len(old_classes) :]
    old_masks = _leave_two_out_masks(old_labels, old_ranks)
    joint_masks = _leave_two_out_masks(joint_labels, joint_ranks)
    old_in_joint = np.isin(joint_labels, old_classes)
    folds = []
    trace = []
    zero_hp = IntrusionGuardHyperparameters(
        candidate_id="d13_delta0_base",
        mode=hyperparameters.mode,
        old_risk_quantile=hyperparameters.old_risk_quantile,
        new_room_quantile=hyperparameters.new_room_quantile,
        safety=0.0,
        cap=0.0,
        new_floor_margin=0.0,
        force_zero=True,
    )
    for fold, (held_old, held_joint) in enumerate(zip(old_masks, joint_masks)):
        train_old = ~held_old
        train_joint = ~held_joint
        fitted = _fit_fold_states(
            before_artifact,
            old_rows,
            old_labels,
            old_ranks,
            old_classes,
            train_old,
            after_artifact,
            joint_rows,
            joint_labels,
            joint_ranks,
            joint_classes,
            train_joint,
            hyperparameters=hyperparameters,
        )
        base_fitted = _fit_fold_states(
            before_artifact,
            old_rows,
            old_labels,
            old_ranks,
            old_classes,
            train_old,
            after_artifact,
            joint_rows,
            joint_labels,
            joint_ranks,
            joint_classes,
            train_joint,
            hyperparameters=zero_hp,
        )
        held_old_joint = held_joint & old_in_joint
        held_new_joint = held_joint & ~old_in_joint
        before_scores = _score_numpy(old_rows[held_old], fitted.before_state)
        after_old_scores = _score_numpy(
            joint_rows[held_old_joint], fitted.after_state
        )
        after_new_scores = _score_numpy(
            joint_rows[held_new_joint], fitted.after_state
        )
        base_old_scores = _score_numpy(
            joint_rows[held_old_joint], base_fitted.after_state
        )
        base_new_scores = _score_numpy(
            joint_rows[held_new_joint], base_fitted.after_state
        )
        before_predictions = np.asarray(old_classes)[
            np.argmax(before_scores, axis=1)
        ]
        after_old_predictions = np.asarray(joint_classes)[
            np.argmax(after_old_scores, axis=1)
        ]
        after_new_predictions = np.asarray(joint_classes)[
            np.argmax(after_new_scores, axis=1)
        ]
        base_old_predictions = np.asarray(joint_classes)[
            np.argmax(base_old_scores, axis=1)
        ]
        base_new_predictions = np.asarray(joint_classes)[
            np.argmax(base_new_scores, axis=1)
        ]
        before_metric = _prediction_metrics(
            old_labels[held_old], before_predictions, old_classes
        )
        old_metric = _prediction_metrics(
            joint_labels[held_old_joint], after_old_predictions, old_classes
        )
        new_metric = _prediction_metrics(
            joint_labels[held_new_joint], after_new_predictions, new_classes
        )
        base_old_metric = _prediction_metrics(
            joint_labels[held_old_joint], base_old_predictions, old_classes
        )
        base_new_metric = _prediction_metrics(
            joint_labels[held_new_joint], base_new_predictions, new_classes
        )
        joint_truth = np.concatenate(
            [joint_labels[held_old_joint], joint_labels[held_new_joint]]
        )
        joint_predictions = np.concatenate(
            [after_old_predictions, after_new_predictions]
        )
        base_joint_predictions = np.concatenate(
            [base_old_predictions, base_new_predictions]
        )
        row = {
            "fold": fold,
            "candidate_id": hyperparameters.candidate_id,
            "mode": hyperparameters.mode,
            "before_old": before_metric,
            "after_old": old_metric,
            "after_new": new_metric,
            "base_after_old": base_old_metric,
            "base_after_new": base_new_metric,
            "joint_accuracy": float(np.mean(joint_predictions == joint_truth)),
            "base_joint_accuracy": float(
                np.mean(base_joint_predictions == joint_truth)
            ),
            "h_old_new": _harmonic(
                old_metric["overall_accuracy"], new_metric["overall_accuracy"]
            ),
            "base_h_old_new": _harmonic(
                base_old_metric["overall_accuracy"],
                base_new_metric["overall_accuracy"],
            ),
            "old_forgetting": (
                before_metric["overall_accuracy"] - old_metric["overall_accuracy"]
            ),
            "old_score_columns_bitwise_unchanged": bool(
                np.array_equal(
                    after_old_scores[:, : len(old_classes)],
                    base_old_scores[:, : len(old_classes)],
                )
                and np.array_equal(
                    after_new_scores[:, : len(old_classes)],
                    base_new_scores[:, : len(old_classes)],
                )
            ),
            "new_class_penalties": fitted.after_state.new_logit_penalties[
                len(old_classes) :
            ].tolist(),
            "calibration_diagnostics": [
                dict(value) for value in fitted.after_state.calibration_diagnostics
            ],
            "all_new_class_calibration_feasible": all(
                bool(value["protection_feasible"])
                for value in fitted.after_state.calibration_diagnostics
            ),
            "old_train_rows_per_class": 8,
            "new_train_rows_per_class": 8,
            "old_held_rows_per_class": 2,
            "new_held_rows_per_class": 2,
        }
        folds.append(row)
        trace.extend({"fold": fold, **value} for value in fitted.trace)
        trace.append({"phase": "joint_l2o_fold_summary", **row})
    before_old = _aggregate_metrics(folds, old_classes, "before_old")
    after_old = _aggregate_metrics(folds, old_classes, "after_old")
    after_new = _aggregate_metrics(folds, new_classes, "after_new")
    base_after_old = _aggregate_metrics(folds, old_classes, "base_after_old")
    base_after_new = _aggregate_metrics(folds, new_classes, "base_after_new")
    joint_accuracy = float(np.mean([row["joint_accuracy"] for row in folds]))
    base_joint_accuracy = float(
        np.mean([row["base_joint_accuracy"] for row in folds])
    )
    h_value = _harmonic(
        after_old["overall_accuracy"], after_new["overall_accuracy"]
    )
    base_h = _harmonic(
        base_after_old["overall_accuracy"], base_after_new["overall_accuracy"]
    )
    return (
        {
            "selection_policy": (
                "joint_physical_leave_two_out_old_new_each_held2_all_registered"
            ),
            "before_old": before_old,
            "after_old": after_old,
            "after_new": after_new,
            "base_after_old": base_after_old,
            "base_after_new": base_after_new,
            "joint_accuracy": joint_accuracy,
            "base_joint_accuracy": base_joint_accuracy,
            "h_old_new": h_value,
            "base_h_old_new": base_h,
            "delta_vs_base_joint_accuracy": joint_accuracy - base_joint_accuracy,
            "delta_vs_base_h_old_new": h_value - base_h,
            "old_forgetting": (
                before_old["overall_accuracy"] - after_old["overall_accuracy"]
            ),
            "old_per_class_non_degraded_vs_before": all(
                after_old["per_class_accuracy"][label] + 1.0e-12
                >= before_old["per_class_accuracy"][label]
                for label in old_classes
            ),
            "old_per_class_non_degraded_vs_base_cosine": all(
                after_old["per_class_accuracy"][label] + 1.0e-12
                >= base_after_old["per_class_accuracy"][label]
                for label in old_classes
            ),
            "new_per_class_non_degraded_vs_base_cosine": all(
                after_new["per_class_accuracy"][label] + 1.0e-12
                >= base_after_new["per_class_accuracy"][label]
                for label in new_classes
            ),
            "new_no_class_collapsed_to_zero": all(
                after_new["per_class_accuracy"][label] > 0.0
                for label in new_classes
            ),
            "all_new_class_calibration_feasible": all(
                row["all_new_class_calibration_feasible"] for row in folds
            ),
            "old_score_columns_bitwise_unchanged": all(
                row["old_score_columns_bitwise_unchanged"] for row in folds
            ),
            "folds": folds,
        },
        tuple(trace),
    )
def predict_all_registered(
    state: NewLogitIntrusionGuardState,
    query_artifact: RuntimeAuthorizedFeatureArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict exactly one physical query over all registered classes."""

    _validate_state(state)
    rows = _artifact_rows(query_artifact)
    if len(rows) != 1:
        raise NewLogitIntrusionGuardError(
            "formal prediction requires exactly one physical query"
        )
    if (
        query_artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or query_artifact.feature_code_sha256 != state.feature_code_sha256
        or query_artifact.sealed_phase1_checkpoint_sha256
        != state.sealed_phase1_checkpoint_sha256
        or query_artifact.operator_id != state.operator_id
        or query_artifact.view_seed != state.view_seed
    ):
        raise NewLogitIntrusionGuardError(
            "query runtime/code/checkpoint/operator binding mismatch"
        )
    scores = _score_numpy(rows, state)
    predictions = np.asarray(state.classes)[np.argmax(scores, axis=1)]
    return predictions, scores
