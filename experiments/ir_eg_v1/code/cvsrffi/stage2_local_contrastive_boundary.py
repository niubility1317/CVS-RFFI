"""D7b support-only local contrastive boundary head.

The head operates on already extracted representations from one fixed received
LEO_weak IQ per physical sample. Rival graphs are class-prototype local and
fit only from registered support. Query scoring is per sample over all
registered classes and has no query-role, truth, quota, ordering, or graph API.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-8
DEFAULT_BETA_CANDIDATES = (0.0, 0.05, 0.10, 0.20)
DEFAULT_RIVAL_COUNT = 1
DEFAULT_MAX_OVERALL_DROP = 0.02
DEFAULT_MAX_CLASS_DROP = 0.0
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024


class LocalContrastiveBoundaryError(ValueError):
    """Raised when D7b support-only or immutable-state invariants drift."""


@dataclass(frozen=True)
class LocalBoundaryHead:
    """Immutable scenario-local registered-class boundary state."""

    schema: str
    classes: np.ndarray
    prototypes: np.ndarray
    rival_indices: np.ndarray
    beta: np.ndarray
    old_class_count: int
    support_audit: Mapping[str, Any]

    @property
    def class_count(self) -> int:
        return int(len(self.classes))

    @property
    def feature_dim(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def rival_count(self) -> int:
        return int(self.rival_indices.shape[1])

    @property
    def trainable_parameters(self) -> int:
        return 0

    @property
    def persistent_state_bytes(self) -> int:
        # FP16 prototypes/beta, uint16 rival indices.
        return int(
            2 * self.prototypes.size
            + 2 * self.beta.size
            + 2 * self.rival_indices.size
        )

    @property
    def estimated_macs_per_query(self) -> int:
        # Base registered-class cosine plus local margin/subtract/scale/add.
        return int(
            self.class_count * self.feature_dim
            + self.class_count * self.rival_count
            + 3 * self.class_count
        )

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d7b_resource.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "trainable_parameter_limit": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_pass": True,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "estimated_macs_per_query": self.estimated_macs_per_query,
            "registered_class_count": self.class_count,
            "rival_count_per_class": self.rival_count,
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "per_sample_all_registered_classes": True,
        }


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _readonly_strings(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value).astype(str))
    result.setflags(write=False)
    return result


def _normalize(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    return value / np.maximum(
        np.linalg.norm(value, axis=-1, keepdims=True), EPS
    )


def _validate_support(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    physical_sample_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(tuple(str(value) for value in support_labels))
    ids = np.asarray(tuple(str(value) for value in physical_sample_ids))
    if (
        rows.ndim != 2
        or rows.shape[0] < 2
        or rows.shape[1] < 2
        or len(rows) != len(labels)
        or len(rows) != len(ids)
        or not np.isfinite(rows).all()
        or any(not value for value in labels.tolist())
        or any(not value for value in ids.tolist())
        or len(set(ids.tolist())) != len(ids)
    ):
        raise LocalContrastiveBoundaryError(
            "support must be finite [N,D] with unique physical IDs"
        )
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 1):
        raise LocalContrastiveBoundaryError(
            "D7b requires at least two registered classes"
        )
    return np.ascontiguousarray(rows), labels, ids


def _classes_and_prototypes(
    rows: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(set(labels.tolist())))
    prototypes = _normalize(
        np.stack(
            [np.mean(rows[labels == label], axis=0) for label in classes]
        )
    )
    return classes, prototypes


def _rivals_from_gram(
    prototypes: np.ndarray,
    *,
    rival_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(rival_count)
    class_count = len(prototypes)
    if count < 1 or count >= class_count:
        raise LocalContrastiveBoundaryError(
            "rival count must be in [1,class_count-1]"
        )
    gram = _normalize(prototypes) @ _normalize(prototypes).T
    np.fill_diagonal(gram, -np.inf)
    rivals = np.argsort(-gram, axis=1, kind="stable")[:, :count]
    if np.any(rivals == np.arange(class_count)[:, None]):
        raise LocalContrastiveBoundaryError("self rival selected")
    return rivals.astype(np.int64), gram.astype(np.float32)


def _score_with_state(
    rows: np.ndarray,
    prototypes: np.ndarray,
    rival_indices: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    base = _normalize(rows) @ _normalize(prototypes).T
    local_rival = np.max(
        np.take_along_axis(
            base[:, None, :],
            np.broadcast_to(
                rival_indices[None, :, :],
                (len(base),) + rival_indices.shape,
            ),
            axis=2,
        ),
        axis=2,
    )
    return (
        base + beta[None, :] * (base - local_rival)
    ).astype(np.float32)


def _deletion_folds(
    labels: np.ndarray,
    *,
    width_per_class: int,
) -> tuple[tuple[int, ...], ...]:
    width = int(width_per_class)
    if width not in {1, 2}:
        raise LocalContrastiveBoundaryError("fold width must be one or two")
    classes = tuple(sorted(set(labels.tolist())))
    by_class = [np.flatnonzero(labels == label).tolist() for label in classes]
    if min(len(values) for values in by_class) <= width:
        raise LocalContrastiveBoundaryError(
            "each class must retain support after deletion"
        )
    folds = []
    for offset in range(0, max(len(values) for values in by_class), width):
        held = []
        for values in by_class:
            held.extend(values[offset : offset + width])
        if held:
            folds.append(tuple(sorted(held)))
    if sorted(index for fold in folds for index in fold) != list(
        range(len(labels))
    ):
        raise LocalContrastiveBoundaryError(
            "physical support deletion coverage drift"
        )
    return tuple(folds)


def _evaluate_beta_candidate(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    beta_value: float,
    rival_count: int,
    width_per_class: int,
) -> dict[str, Any]:
    predictions: list[str | None] = [None] * len(rows)
    margins: list[float | None] = [None] * len(rows)
    for fold in _deletion_folds(labels, width_per_class=width_per_class):
        held = np.asarray(fold, dtype=np.int64)
        keep = np.ones(len(rows), dtype=bool)
        keep[held] = False
        classes, prototypes = _classes_and_prototypes(rows[keep], labels[keep])
        rivals, _gram = _rivals_from_gram(
            prototypes, rival_count=rival_count
        )
        beta = np.full(len(classes), float(beta_value), dtype=np.float32)
        score = _score_with_state(rows[held], prototypes, rivals, beta)
        for local, physical_index in enumerate(held.tolist()):
            prediction = int(np.argmax(score[local]))
            truth_index = int(
                np.flatnonzero(classes == labels[physical_index])[0]
            )
            predictions[physical_index] = str(classes[prediction])
            margins[physical_index] = float(
                score[local, truth_index]
                - np.max(np.delete(score[local], truth_index))
            )
    if any(value is None for value in predictions + margins):
        raise LocalContrastiveBoundaryError("support holdout evidence incomplete")
    predicted = np.asarray(predictions)
    correct = predicted == labels
    margin_values = np.asarray(margins, dtype=np.float64)
    per_class = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        per_class[label] = {
            "accuracy": float(np.mean(correct[mask])),
            "mean_margin": float(np.mean(margin_values[mask])),
            "worst_margin": float(np.min(margin_values[mask])),
        }
    return {
        "beta": float(beta_value),
        "protocol": (
            "support_leave_two_physical_samples_out"
            if width_per_class == 2
            else "support_leave_one_physical_sample_out"
        ),
        "overall_accuracy": float(np.mean(correct)),
        "min_class_accuracy": min(
            value["accuracy"] for value in per_class.values()
        ),
        "mean_class_accuracy": float(
            np.mean([value["accuracy"] for value in per_class.values()])
        ),
        "worst_margin": float(np.min(margin_values)),
        "per_class": per_class,
    }


def _select_beta(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    beta_candidates: Sequence[float],
    rival_count: int,
    max_overall_drop: float,
    max_class_drop: float,
) -> tuple[float, dict[str, Any]]:
    values = tuple(float(value) for value in beta_candidates)
    if (
        not values
        or values[0] != 0.0
        or len(set(values)) != len(values)
        or any(not math.isfinite(value) or not 0.0 <= value <= 0.50 for value in values)
    ):
        raise LocalContrastiveBoundaryError(
            "beta candidates must be unique, begin at zero, and stay in [0,0.5]"
        )
    width = 2 if min(
        int(np.sum(labels == label)) for label in set(labels.tolist())
    ) >= 4 else 1
    evidence = [
        _evaluate_beta_candidate(
            rows,
            labels,
            beta_value=value,
            rival_count=rival_count,
            width_per_class=width,
        )
        for value in values
    ]
    baseline = evidence[0]
    baseline_class = baseline["per_class"]
    eligible = []
    for row in evidence:
        row["overall_noninferiority_pass"] = bool(
            row["overall_accuracy"] + float(max_overall_drop)
            >= baseline["overall_accuracy"]
        )
        row["per_class_non_degradation_pass"] = bool(
            all(
                row["per_class"][label]["accuracy"] + float(max_class_drop)
                >= baseline_class[label]["accuracy"]
                for label in baseline_class
            )
        )
        row["floor_non_degradation_pass"] = bool(
            row["min_class_accuracy"] + float(max_class_drop)
            >= baseline["min_class_accuracy"]
        )
        row["eligible"] = all(
            row[field]
            for field in (
                "overall_noninferiority_pass",
                "per_class_non_degradation_pass",
                "floor_non_degradation_pass",
            )
        )
        row["selection_key"] = [
            row["min_class_accuracy"],
            row["mean_class_accuracy"],
            row["overall_accuracy"],
            row["worst_margin"],
            -row["beta"],
        ]
        if row["eligible"]:
            eligible.append(row)
    if not eligible:
        raise LocalContrastiveBoundaryError("no support-safe beta remains")
    selected = max(
        eligible, key=lambda row: tuple(float(value) for value in row["selection_key"])
    )
    return float(selected["beta"]), {
        "schema": "cvs.phase2.d7b_support_beta_selection.v1",
        "selection_scope": "registered_physical_support_only",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "rival_count": int(rival_count),
        "beta_candidates": list(values),
        "selected_beta": float(selected["beta"]),
        "max_overall_drop": float(max_overall_drop),
        "max_class_drop": float(max_class_drop),
        "candidate_evidence": evidence,
    }


def _extension_tensors(
    parent: LocalBoundaryHead,
    rows: np.ndarray,
    labels: np.ndarray,
    new_classes: Sequence[str],
    *,
    beta_value: float,
    rival_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_classes = parent.classes.astype(str).tolist()
    new_class_list = [str(value) for value in new_classes]
    new_prototypes = _normalize(
        np.stack(
            [np.mean(rows[labels == label], axis=0) for label in new_class_list]
        )
    )
    all_classes = np.asarray(old_classes + new_class_list)
    all_prototypes = np.concatenate(
        [parent.prototypes.copy(), new_prototypes], axis=0
    )
    gram = _normalize(all_prototypes) @ _normalize(all_prototypes).T
    new_rivals = []
    for class_index in range(len(old_classes), len(all_classes)):
        row = gram[class_index].copy()
        row[class_index] = -np.inf
        new_rivals.append(
            np.argsort(-row, kind="stable")[: int(rival_count)]
        )
    rival_indices = np.concatenate(
        [parent.rival_indices.copy(), np.asarray(new_rivals, dtype=np.int64)],
        axis=0,
    )
    beta = np.concatenate(
        [
            parent.beta.copy(),
            np.full(len(new_class_list), float(beta_value), dtype=np.float32),
        ]
    )
    return all_classes, all_prototypes, rival_indices, beta


def _evaluate_extension_beta_candidate(
    parent: LocalBoundaryHead,
    rows: np.ndarray,
    labels: np.ndarray,
    new_classes: Sequence[str],
    *,
    beta_value: float,
    rival_count: int,
    width_per_class: int,
) -> dict[str, Any]:
    old_classes = parent.classes.astype(str).tolist()
    new_class_list = [str(value) for value in new_classes]
    new_mask = np.isin(labels, np.asarray(new_class_list))
    new_rows = rows[new_mask]
    new_labels = labels[new_mask]
    new_source_indices = np.flatnonzero(new_mask)
    predictions: dict[int, str] = {}
    margins: dict[int, float] = {}
    for fold in _deletion_folds(
        new_labels, width_per_class=width_per_class
    ):
        held_local = np.asarray(fold, dtype=np.int64)
        keep = np.ones(len(rows), dtype=bool)
        keep[new_source_indices[held_local]] = False
        classes, prototypes, rivals, beta = _extension_tensors(
            parent,
            rows[keep],
            labels[keep],
            new_class_list,
            beta_value=beta_value,
            rival_count=rival_count,
        )
        score = _score_with_state(
            new_rows[held_local], prototypes, rivals, beta
        )
        for local_index, new_index in enumerate(held_local.tolist()):
            source_index = int(new_source_indices[new_index])
            truth_index = int(
                np.flatnonzero(classes == labels[source_index])[0]
            )
            predictions[source_index] = str(
                classes[int(np.argmax(score[local_index]))]
            )
            margins[source_index] = float(
                score[local_index, truth_index]
                - np.max(np.delete(score[local_index], truth_index))
            )

    classes, prototypes, rivals, beta = _extension_tensors(
        parent,
        rows,
        labels,
        new_class_list,
        beta_value=beta_value,
        rival_count=rival_count,
    )
    old_mask = np.isin(labels, np.asarray(old_classes))
    old_source_indices = np.flatnonzero(old_mask)
    old_score = _score_with_state(
        rows[old_mask], prototypes, rivals, beta
    )
    for local_index, source_index in enumerate(old_source_indices.tolist()):
        truth_index = int(
            np.flatnonzero(classes == labels[source_index])[0]
        )
        predictions[source_index] = str(
            classes[int(np.argmax(old_score[local_index]))]
        )
        margins[source_index] = float(
            old_score[local_index, truth_index]
            - np.max(np.delete(old_score[local_index], truth_index))
        )
    if len(predictions) != len(rows) or len(margins) != len(rows):
        raise LocalContrastiveBoundaryError(
            "extension support evidence incomplete"
        )
    predicted = np.asarray([predictions[index] for index in range(len(rows))])
    margin_values = np.asarray(
        [margins[index] for index in range(len(rows))], dtype=np.float64
    )
    correct = predicted == labels
    per_class = {}
    for label in old_classes + new_class_list:
        mask = labels == label
        per_class[label] = {
            "lifecycle": "old_locked" if label in old_classes else "new",
            "accuracy": float(np.mean(correct[mask])),
            "mean_margin": float(np.mean(margin_values[mask])),
            "worst_margin": float(np.min(margin_values[mask])),
        }
    new_values = [per_class[label] for label in new_class_list]
    old_values = [per_class[label] for label in old_classes]
    return {
        "beta": float(beta_value),
        "protocol": (
            "new_support_leave_two_out_old_state_locked"
            if width_per_class == 2
            else "new_support_leave_one_out_old_state_locked"
        ),
        "old_support_evaluation": "fixed_full_support_intrusion_guard",
        "overall_accuracy": float(np.mean(correct)),
        "min_class_accuracy": min(
            value["accuracy"] for value in per_class.values()
        ),
        "new_min_class_accuracy": min(
            value["accuracy"] for value in new_values
        ),
        "old_min_class_accuracy": min(
            value["accuracy"] for value in old_values
        ),
        "mean_class_accuracy": float(
            np.mean([value["accuracy"] for value in per_class.values()])
        ),
        "worst_margin": float(np.min(margin_values)),
        "per_class": per_class,
    }


def _select_extension_beta(
    parent: LocalBoundaryHead,
    rows: np.ndarray,
    labels: np.ndarray,
    new_classes: Sequence[str],
    *,
    beta_candidates: Sequence[float],
    rival_count: int,
    max_overall_drop: float,
    max_class_drop: float,
) -> tuple[float, dict[str, Any]]:
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
        raise LocalContrastiveBoundaryError(
            "beta candidates must be unique, begin at zero, and stay in [0,0.5]"
        )
    new_class_list = [str(value) for value in new_classes]
    width = 2 if min(
        int(np.sum(labels == label)) for label in new_class_list
    ) >= 4 else 1
    evidence = [
        _evaluate_extension_beta_candidate(
            parent,
            rows,
            labels,
            new_class_list,
            beta_value=value,
            rival_count=rival_count,
            width_per_class=width,
        )
        for value in values
    ]
    baseline = evidence[0]
    eligible = []
    for row in evidence:
        row["overall_noninferiority_pass"] = bool(
            row["overall_accuracy"] + float(max_overall_drop)
            >= baseline["overall_accuracy"]
        )
        row["per_class_non_degradation_pass"] = bool(
            all(
                row["per_class"][label]["accuracy"] + float(max_class_drop)
                >= baseline["per_class"][label]["accuracy"]
                for label in baseline["per_class"]
            )
        )
        row["new_floor_non_degradation_pass"] = bool(
            row["new_min_class_accuracy"] + float(max_class_drop)
            >= baseline["new_min_class_accuracy"]
        )
        row["old_floor_non_degradation_pass"] = bool(
            row["old_min_class_accuracy"] + float(max_class_drop)
            >= baseline["old_min_class_accuracy"]
        )
        row["eligible"] = all(
            row[field]
            for field in (
                "overall_noninferiority_pass",
                "per_class_non_degradation_pass",
                "new_floor_non_degradation_pass",
                "old_floor_non_degradation_pass",
            )
        )
        row["selection_key"] = [
            row["new_min_class_accuracy"],
            row["old_min_class_accuracy"],
            row["mean_class_accuracy"],
            row["overall_accuracy"],
            row["worst_margin"],
            -row["beta"],
        ]
        if row["eligible"]:
            eligible.append(row)
    if not eligible:
        raise LocalContrastiveBoundaryError(
            "no support-safe extension beta remains"
        )
    selected = max(
        eligible,
        key=lambda row: tuple(
            float(value) for value in row["selection_key"]
        ),
    )
    return float(selected["beta"]), {
        "schema": "cvs.phase2.d7b_extension_beta_selection.v1",
        "selection_scope": "registered_physical_support_only",
        "old_state_during_selection": "bitwise_locked",
        "new_support_deletion_width_per_class": width,
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "rival_count": int(rival_count),
        "beta_candidates": list(values),
        "selected_beta": float(selected["beta"]),
        "max_overall_drop": float(max_overall_drop),
        "max_class_drop": float(max_class_drop),
        "candidate_evidence": evidence,
    }


def fit_local_contrastive_boundary(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    beta_candidates: Sequence[float] = DEFAULT_BETA_CANDIDATES,
    rival_count: int = DEFAULT_RIVAL_COUNT,
    max_overall_drop: float = DEFAULT_MAX_OVERALL_DROP,
    max_class_drop: float = DEFAULT_MAX_CLASS_DROP,
) -> LocalBoundaryHead:
    """Fit a before-registration D7b head from physical support only."""

    rows, labels, ids = _validate_support(
        support_features, support_labels, physical_sample_ids
    )
    beta_value, beta_audit = _select_beta(
        rows,
        labels,
        beta_candidates=beta_candidates,
        rival_count=int(rival_count),
        max_overall_drop=float(max_overall_drop),
        max_class_drop=float(max_class_drop),
    )
    classes, prototypes = _classes_and_prototypes(rows, labels)
    rivals, gram = _rivals_from_gram(
        prototypes, rival_count=int(rival_count)
    )
    beta = np.full(len(classes), beta_value, dtype=np.float32)
    audit = {
        "schema": "cvs.phase2.d7b_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "physical_support_sample_count": len(ids),
        "physical_support_ids_unique": True,
        "computation_views_count_as_additional_physical_samples": False,
        "additional_leo_channel_state_generation": False,
        "query_rows_used": 0,
        "query_roles_used": False,
        "query_quota_used": False,
        "prototype_gram": gram.tolist(),
        "rival_class_handles": [
            classes[row].astype(str).tolist() for row in rivals
        ],
        "beta_selection": beta_audit,
        "old_state_bitwise_locked": False,
    }
    head = LocalBoundaryHead(
        schema="cvs.phase2.d7b_local_contrastive_boundary.v1",
        classes=_readonly_strings(classes),
        prototypes=_readonly(prototypes, np.float32),
        rival_indices=_readonly(rivals, np.int64),
        beta=_readonly(beta, np.float32),
        old_class_count=len(classes),
        support_audit=audit,
    )
    if head.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise LocalContrastiveBoundaryError("D7b state exceeds 256KB")
    return head


def extend_local_contrastive_boundary(
    parent: LocalBoundaryHead,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    beta_candidates: Sequence[float] = DEFAULT_BETA_CANDIDATES,
    rival_count: int = DEFAULT_RIVAL_COUNT,
    max_overall_drop: float = DEFAULT_MAX_OVERALL_DROP,
    max_class_drop: float = DEFAULT_MAX_CLASS_DROP,
) -> LocalBoundaryHead:
    """Register absent classes while bitwise locking every old D7b tensor."""

    if not isinstance(parent, LocalBoundaryHead):
        raise LocalContrastiveBoundaryError("parent D7b head is required")
    rows, labels, ids = _validate_support(
        support_features, support_labels, physical_sample_ids
    )
    old_classes = parent.classes.astype(str).tolist()
    new_classes = sorted(set(labels.tolist()) - set(old_classes))
    if not new_classes:
        raise LocalContrastiveBoundaryError("no absent class to register")
    if any(np.sum(labels == label) < 2 for label in new_classes):
        raise LocalContrastiveBoundaryError(
            "new-class beta selection requires at least two physical samples"
        )
    if not set(old_classes).issubset(set(labels.tolist())):
        raise LocalContrastiveBoundaryError(
            "registration support must retain every locked old class"
        )
    beta_value, beta_audit = _select_extension_beta(
        parent,
        rows,
        labels,
        new_classes,
        beta_candidates=beta_candidates,
        rival_count=int(rival_count),
        max_overall_drop=float(max_overall_drop),
        max_class_drop=float(max_class_drop),
    )
    all_classes, all_prototypes, rival_indices, beta = _extension_tensors(
        parent,
        rows,
        labels,
        new_classes,
        beta_value=beta_value,
        rival_count=int(rival_count),
    )
    new_rivals = rival_indices[len(old_classes) :]
    audit = {
        "schema": "cvs.phase2.d7b_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "physical_support_sample_count": len(ids),
        "physical_support_ids_unique": True,
        "query_rows_used": 0,
        "query_roles_used": False,
        "query_quota_used": False,
        "old_state_bitwise_locked": True,
        "old_state_update_count": 0,
        "old_rivals_can_reference_new_classes": False,
        "new_class_handles": new_classes,
        "new_rival_class_handles": [
            all_classes[row].astype(str).tolist() for row in new_rivals
        ],
        "beta_selection_for_new_classes": beta_audit,
    }
    head = LocalBoundaryHead(
        schema=parent.schema,
        classes=_readonly_strings(all_classes),
        prototypes=_readonly(all_prototypes, np.float32),
        rival_indices=_readonly(rival_indices, np.int64),
        beta=_readonly(beta, np.float32),
        old_class_count=parent.old_class_count,
        support_audit=audit,
    )
    old_count = len(old_classes)
    if (
        not np.array_equal(head.classes[:old_count], parent.classes)
        or not np.array_equal(head.prototypes[:old_count], parent.prototypes)
        or not np.array_equal(
            head.rival_indices[:old_count], parent.rival_indices
        )
        or not np.array_equal(head.beta[:old_count], parent.beta)
    ):
        raise LocalContrastiveBoundaryError("parent old D7b state mutation")
    if head.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
        raise LocalContrastiveBoundaryError("D7b state exceeds 256KB")
    return head


def score_local_contrastive_boundary(
    query_features: np.ndarray,
    head: LocalBoundaryHead,
) -> np.ndarray:
    """Score each query independently over every registered class."""

    rows = np.asarray(query_features, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != head.feature_dim
        or not np.isfinite(rows).all()
    ):
        raise LocalContrastiveBoundaryError(
            "query representations must be finite [N,D]"
        )
    return _score_with_state(
        rows,
        head.prototypes,
        head.rival_indices,
        head.beta,
    )


def predict_local_contrastive_boundary(
    query_features: np.ndarray,
    head: LocalBoundaryHead,
) -> np.ndarray:
    score = score_local_contrastive_boundary(query_features, head)
    return head.classes[np.argmax(score, axis=1)].astype(str)


def public_interface_is_query_oracle_free() -> bool:
    forbidden = {"label", "truth", "role", "quota", "assignment", "graph"}
    query_functions = (
        score_local_contrastive_boundary,
        predict_local_contrastive_boundary,
    )
    return all(
        not any(
            token in parameter.lower()
            for parameter in inspect.signature(function).parameters
            for token in forbidden
        )
        for function in query_functions
    )


__all__ = [
    "DEFAULT_BETA_CANDIDATES",
    "LocalBoundaryHead",
    "LocalContrastiveBoundaryError",
    "extend_local_contrastive_boundary",
    "fit_local_contrastive_boundary",
    "predict_local_contrastive_boundary",
    "public_interface_is_query_oracle_free",
    "score_local_contrastive_boundary",
]
