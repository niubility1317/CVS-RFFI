"""D5 collision-aware compressed multi-prototype head for Phase2.

The module is deliberately dataset- and scorer-blind.  It fits only from
registered labeled support features, scores every query independently over all
registered classes, and exposes no query-label, query-role, quota, ordering, or
query-query graph interface.

Prototype multiplicity is selected conservatively from physical-sample
leave-one-out evidence.  Computation views are not accepted by this API, so a
view ablation cannot masquerade as an additional physical support observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-8
SCHEMA = "cvs.phase2.d5_collision_aware_multiproto.v1"
MAX_PROTOTYPES_PER_CLASS = 3
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024


class CollisionAwareMultiPrototypeError(ValueError):
    """Raised when the support-only D5 head fails closed."""


@dataclass(frozen=True)
class D5Config:
    residual_shrinkage: float = 0.65
    scale_min: float = 0.65
    scale_max: float = 1.55
    centroid_mix: float = 0.30
    collision_penalty_weight: float = 0.10
    loo_gain_required: float = 0.025
    margin_gain_required: float = 0.015
    compactness_gain_required: float = 0.010
    max_collision_worsening: float = 0.020
    complexity_penalty: float = 0.008
    deconfusion_steps: tuple[float, ...] = (0.0, 0.03, 0.06, 0.09)
    max_floor_penalty: float = 0.35
    new_floor_safety_margin: float = 0.0


@dataclass(frozen=True)
class CollisionAwareMultiPrototypeHead:
    """Immutable scenario-local deployment state."""

    schema: str
    classes: np.ndarray
    residual_scale: np.ndarray
    prototypes: np.ndarray
    prototype_mask: np.ndarray
    centroids: np.ndarray
    class_penalty: np.ndarray
    prototype_count_by_class: np.ndarray
    centroid_mix: float
    old_class_count: int
    support_audit: Mapping[str, Any]

    @property
    def class_count(self) -> int:
        return int(len(self.classes))

    @property
    def feature_dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def active_prototype_count(self) -> int:
        return int(np.sum(self.prototype_mask))

    @property
    def trainable_parameters(self) -> int:
        return 0

    @property
    def persistent_state_bytes_fp16(self) -> int:
        # Flight tensors: FP16 scale/prototypes/centroids/class penalty,
        # uint8 mask/counts. Class handles and JSON audit are metadata.
        return int(
            2
            * (
                self.residual_scale.size
                + self.prototypes.size
                + self.centroids.size
                + self.class_penalty.size
                + 1
            )
            + self.prototype_mask.nbytes
            + self.prototype_count_by_class.astype(np.uint8).nbytes
        )

    @property
    def extra_macs_per_query(self) -> int:
        d = self.feature_dim
        c = self.class_count
        p = self.active_prototype_count
        # Diagonal scale + normalization + prototype/centroid dot products
        # plus max/mix/penalty pooling.
        return int(3 * d + p * d + c * d + 2 * p + 4 * c)

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d5_resource.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "trainable_parameter_limit": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_pass": True,
            "persistent_state_bytes_fp16": self.persistent_state_bytes_fp16,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": (
                self.persistent_state_bytes_fp16
                <= MAX_PERSISTENT_STATE_BYTES
            ),
            "estimated_extra_macs_per_query": self.extra_macs_per_query,
            "active_prototype_count": self.active_prototype_count,
            "max_prototypes_per_class": MAX_PROTOTYPES_PER_CLASS,
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "per_sample_all_registered_classes": True,
            "scenario_atomic_fit_required": True,
        }


@dataclass(frozen=True)
class SupportOnlyMarginSelection:
    """D6c margin lock selected without any query input."""

    selected_margin: float
    selected_config: D5Config
    selection_evidence: Mapping[str, Any]
    canonical_sha256: str


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), EPS
    )


def _validate_config(config: D5Config) -> None:
    if not 0.0 <= config.residual_shrinkage <= 1.0:
        raise CollisionAwareMultiPrototypeError(
            "residual_shrinkage must be in [0,1]"
        )
    if not 0.0 < config.scale_min <= 1.0 <= config.scale_max:
        raise CollisionAwareMultiPrototypeError(
            "scale bounds must contain one"
        )
    if not 0.0 <= config.centroid_mix <= 1.0:
        raise CollisionAwareMultiPrototypeError(
            "centroid_mix must be in [0,1]"
        )
    if (
        config.collision_penalty_weight < 0.0
        or config.loo_gain_required < 0.0
        or config.margin_gain_required < 0.0
        or config.compactness_gain_required < 0.0
        or config.max_collision_worsening < 0.0
        or config.complexity_penalty < 0.0
    ):
        raise CollisionAwareMultiPrototypeError(
            "D5 stability thresholds must be nonnegative"
        )
    if (
        not config.deconfusion_steps
        or config.deconfusion_steps[0] != 0.0
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 0.15
            for value in config.deconfusion_steps
        )
    ):
        raise CollisionAwareMultiPrototypeError(
            "deconfusion steps must begin at zero and stay in [0,0.15]"
        )
    if not 0.0 <= config.max_floor_penalty <= 0.5:
        raise CollisionAwareMultiPrototypeError(
            "max_floor_penalty must be in [0,0.5]"
        )
    if not 0.0 <= config.new_floor_safety_margin <= 0.20:
        raise CollisionAwareMultiPrototypeError(
            "new_floor_safety_margin must be in [0,0.20]"
        )


def _validate_support(
    features: np.ndarray,
    labels: Sequence[str],
    physical_sample_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float32)
    targets = np.asarray(tuple(str(value) for value in labels))
    sample_ids = np.asarray(tuple(str(value) for value in physical_sample_ids))
    if (
        rows.ndim != 2
        or rows.shape[0] < 2
        or rows.shape[1] < 2
        or len(targets) != len(rows)
        or len(sample_ids) != len(rows)
        or not np.isfinite(rows).all()
    ):
        raise CollisionAwareMultiPrototypeError(
            "support must be finite [N,D] with aligned labels and physical IDs"
        )
    if any(not value for value in targets.tolist()) or any(
        not value for value in sample_ids.tolist()
    ):
        raise CollisionAwareMultiPrototypeError(
            "support labels and physical IDs must be non-empty"
        )
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise CollisionAwareMultiPrototypeError(
            "each support row must be one unique physical sample"
        )
    classes, counts = np.unique(targets, return_counts=True)
    if len(classes) < 2 or np.any(counts < 1):
        raise CollisionAwareMultiPrototypeError(
            "D5 requires at least two registered support classes"
        )
    return np.ascontiguousarray(rows), targets, sample_ids


def _fit_residual_scale(
    rows: np.ndarray,
    labels: np.ndarray,
    config: D5Config,
) -> np.ndarray:
    class_variances = []
    for class_handle in sorted(set(labels.tolist())):
        selected = rows[labels == class_handle]
        residual = selected - selected.mean(axis=0, keepdims=True)
        class_variances.append(np.mean(np.square(residual), axis=0))
    within = np.mean(np.stack(class_variances), axis=0)
    global_var = float(np.mean(within))
    shrunk = (
        (1.0 - config.residual_shrinkage) * within
        + config.residual_shrinkage * global_var
    )
    inverse = 1.0 / np.sqrt(np.maximum(shrunk, EPS))
    inverse /= max(float(np.median(inverse)), EPS)
    return np.clip(
        inverse, config.scale_min, config.scale_max
    ).astype(np.float32)


def _farthest_first(rows: np.ndarray, count: int) -> np.ndarray:
    values = _normalize(rows)
    target = min(max(1, int(count)), len(values))
    center = _normalize(values.mean(axis=0, keepdims=True))[0]
    chosen = [int(np.argmax(values @ center))]
    while len(chosen) < target:
        similarity = values @ values[np.asarray(chosen)].T
        nearest = np.max(similarity, axis=1)
        nearest[np.asarray(chosen)] = np.inf
        chosen.append(int(np.argmin(nearest)))
    return np.asarray(chosen, dtype=np.int64)


def _spherical_prototypes(rows: np.ndarray, count: int) -> np.ndarray:
    values = _normalize(rows)
    target = min(max(1, int(count)), len(values))
    if target == 1:
        return _normalize(values.mean(axis=0, keepdims=True))
    centers = values[_farthest_first(values, target)].copy()
    for _ in range(8):
        assignment = np.argmax(values @ centers.T, axis=1)
        updated = []
        for index in range(target):
            selected = values[assignment == index]
            if len(selected) == 0:
                nearest = np.max(values @ centers.T, axis=1)
                selected = values[[int(np.argmin(nearest))]]
            updated.append(_normalize(selected.mean(axis=0, keepdims=True))[0])
        next_centers = np.stack(updated)
        if np.allclose(next_centers, centers, rtol=0.0, atol=1.0e-7):
            centers = next_centers
            break
        centers = next_centers
    order = np.lexsort(np.flipud(centers.T))
    return centers[order].astype(np.float32)


def _class_score(
    rows: np.ndarray,
    prototypes: np.ndarray,
    centroid: np.ndarray,
    *,
    centroid_mix: float,
    penalty: float,
) -> np.ndarray:
    values = _normalize(rows)
    maximum = np.max(values @ _normalize(prototypes).T, axis=1)
    center = values @ _normalize(centroid[None, :]).T
    return (
        (1.0 - centroid_mix) * maximum
        + centroid_mix * center[:, 0]
        - float(penalty)
    ).astype(np.float32)


def _local_1nn_statistics(
    rows: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    similarity = _normalize(rows) @ _normalize(rows).T
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argmax(similarity, axis=1)
    agreement = labels[nearest] == labels
    per_class = {
        class_handle: float(np.mean(agreement[labels == class_handle]))
        for class_handle in sorted(set(labels.tolist()))
    }
    return {
        "overall_accuracy": float(np.mean(agreement)),
        "per_class_accuracy": per_class,
    }


def _full_candidate_geometry(
    class_rows: np.ndarray,
    count: int,
    other_centroids: np.ndarray,
) -> dict[str, float | np.ndarray]:
    prototypes = _spherical_prototypes(class_rows, count)
    centroid = _normalize(class_rows.mean(axis=0, keepdims=True))[0]
    own_similarity = _normalize(class_rows) @ prototypes.T
    compactness = float(np.mean(np.max(own_similarity, axis=1)))
    collision = (
        float(np.max(prototypes @ _normalize(other_centroids).T))
        if len(other_centroids)
        else -1.0
    )
    return {
        "prototypes": prototypes,
        "centroid": centroid,
        "compactness": compactness,
        "collision": collision,
    }


def _physical_loo_candidate(
    rows: np.ndarray,
    labels: np.ndarray,
    class_handle: str,
    count: int,
    *,
    centroids_by_class: Mapping[str, np.ndarray],
    centroid_mix: float,
) -> dict[str, float]:
    selected_indices = np.flatnonzero(labels == class_handle)
    correct: list[bool] = []
    margins: list[float] = []
    for held_index in selected_indices:
        remaining = selected_indices[selected_indices != held_index]
        if len(remaining) == 0:
            # K1 cannot provide physical-sample LOO evidence; force the
            # conservative one-prototype fallback.
            return {
                "loo_accuracy": 0.0,
                "mean_margin": -1.0,
                "worst_margin": -1.0,
                "physical_loo_available": 0.0,
            }
        own_prototypes = _spherical_prototypes(
            rows[remaining], min(count, len(remaining))
        )
        held = rows[held_index : held_index + 1]
        class_order = sorted(centroids_by_class)
        scores = []
        for candidate in class_order:
            if candidate == class_handle:
                scores.append(
                    _class_score(
                        held,
                        own_prototypes,
                        centroids_by_class[candidate],
                        centroid_mix=centroid_mix,
                        penalty=0.0,
                    )[0]
                )
            else:
                scores.append(
                    _class_score(
                        held,
                        centroids_by_class[candidate][None, :],
                        centroids_by_class[candidate],
                        centroid_mix=centroid_mix,
                        penalty=0.0,
                    )[0]
                )
        score_array = np.asarray(scores, dtype=np.float32)
        truth_index = class_order.index(class_handle)
        other = np.delete(score_array, truth_index)
        correct.append(int(np.argmax(score_array)) == truth_index)
        margins.append(float(score_array[truth_index] - np.max(other)))
    return {
        "loo_accuracy": float(np.mean(correct)),
        "mean_margin": float(np.mean(margins)),
        "worst_margin": float(np.min(margins)),
        "physical_loo_available": 1.0,
    }


def _select_count(
    candidates: Sequence[Mapping[str, Any]],
    config: D5Config,
) -> int:
    selected = 0
    for index in range(1, len(candidates)):
        base = candidates[selected]
        candidate = candidates[index]
        if candidate["physical_loo_available"] == 0.0:
            break
        loo_gain = float(candidate["loo_accuracy"] - base["loo_accuracy"])
        margin_gain = float(candidate["worst_margin"] - base["worst_margin"])
        compactness_gain = float(
            candidate["compactness"] - base["compactness"]
        )
        collision_worsening = float(
            candidate["collision"] - base["collision"]
        )
        stability_score_gain = float(
            loo_gain
            + 0.20 * margin_gain
            + 0.10 * compactness_gain
            - config.complexity_penalty
        )
        stable_gain = (
            loo_gain >= config.loo_gain_required
            or (
                loo_gain >= 0.0
                and margin_gain >= config.margin_gain_required
                and compactness_gain >= config.compactness_gain_required
            )
        )
        if (
            stable_gain
            and collision_worsening <= config.max_collision_worsening
            and stability_score_gain > 0.0
        ):
            selected = index
        else:
            break
    return selected + 1


def _build_class_models(
    transformed: np.ndarray,
    labels: np.ndarray,
    config: D5Config,
    *,
    allowed_classes: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    all_classes = tuple(sorted(set(labels.tolist())))
    classes = (
        tuple(str(value) for value in allowed_classes)
        if allowed_classes is not None
        else all_classes
    )
    centroids_by_class = {
        class_handle: _normalize(
            transformed[labels == class_handle].mean(axis=0, keepdims=True)
        )[0]
        for class_handle in all_classes
    }
    local = _local_1nn_statistics(transformed, labels)
    models: dict[str, dict[str, Any]] = {}
    per_class_audit: dict[str, Any] = {}
    for class_handle in classes:
        class_rows = transformed[labels == class_handle]
        other_centroids = np.stack(
            [
                centroids_by_class[value]
                for value in all_classes
                if value != class_handle
            ]
        )
        candidates = []
        for count in range(
            1, min(MAX_PROTOTYPES_PER_CLASS, len(class_rows)) + 1
        ):
            geometry = _full_candidate_geometry(
                class_rows, count, other_centroids
            )
            loo = _physical_loo_candidate(
                transformed,
                labels,
                class_handle,
                count,
                centroids_by_class=centroids_by_class,
                centroid_mix=config.centroid_mix,
            )
            candidates.append(
                {
                    **geometry,
                    **loo,
                    "prototype_count": count,
                    "local_1nn_accuracy": local["per_class_accuracy"][
                        class_handle
                    ],
                }
            )
        selected_count = _select_count(candidates, config)
        selected = candidates[selected_count - 1]
        collision_excess = max(0.0, float(selected["collision"]))
        penalty = config.collision_penalty_weight * collision_excess
        models[class_handle] = {
            "prototypes": np.asarray(selected["prototypes"], dtype=np.float32),
            "centroid": np.asarray(selected["centroid"], dtype=np.float32),
            "penalty": float(penalty),
        }
        per_class_audit[class_handle] = {
            "physical_support_count": int(len(class_rows)),
            "selected_prototype_count": selected_count,
            "local_1nn_accuracy": float(
                local["per_class_accuracy"][class_handle]
            ),
            "candidate_diagnostics": [
                {
                    key: (
                        float(value)
                        if isinstance(value, (float, np.floating))
                        else int(value)
                    )
                    for key, value in candidate.items()
                    if key
                    not in {
                        "prototypes",
                        "centroid",
                    }
                }
                for candidate in candidates
            ],
            "class_penalty": float(penalty),
        }
    return models, {
        "class_gram": (
            np.stack([centroids_by_class[value] for value in all_classes])
            @ np.stack([centroids_by_class[value] for value in all_classes]).T
        ).astype(np.float32),
        "local_1nn": local,
        "per_class": per_class_audit,
    }


def _assemble_head(
    *,
    classes: Sequence[str],
    residual_scale: np.ndarray,
    models: Mapping[str, Mapping[str, Any]],
    centroid_mix: float,
    old_class_count: int,
    support_audit: Mapping[str, Any],
) -> CollisionAwareMultiPrototypeHead:
    class_order = tuple(str(value) for value in classes)
    feature_dim = int(len(residual_scale))
    prototypes = np.zeros(
        (len(class_order), MAX_PROTOTYPES_PER_CLASS, feature_dim),
        dtype=np.float32,
    )
    mask = np.zeros(
        (len(class_order), MAX_PROTOTYPES_PER_CLASS), dtype=np.uint8
    )
    centroids = np.zeros((len(class_order), feature_dim), dtype=np.float32)
    penalties = np.zeros(len(class_order), dtype=np.float32)
    counts = np.zeros(len(class_order), dtype=np.uint8)
    for index, class_handle in enumerate(class_order):
        model = models[class_handle]
        class_prototypes = np.asarray(model["prototypes"], dtype=np.float32)
        count = len(class_prototypes)
        if not 1 <= count <= MAX_PROTOTYPES_PER_CLASS:
            raise CollisionAwareMultiPrototypeError(
                "prototype count must be in [1,3]"
            )
        prototypes[index, :count] = class_prototypes
        mask[index, :count] = 1
        counts[index] = count
        centroids[index] = np.asarray(model["centroid"], dtype=np.float32)
        penalties[index] = float(model["penalty"])
    head = CollisionAwareMultiPrototypeHead(
        schema=SCHEMA,
        classes=_readonly(np.asarray(class_order), np.dtype("<U128")),
        residual_scale=_readonly(residual_scale, np.float32),
        prototypes=_readonly(prototypes, np.float32),
        prototype_mask=_readonly(mask, np.uint8),
        centroids=_readonly(centroids, np.float32),
        class_penalty=_readonly(penalties, np.float32),
        prototype_count_by_class=_readonly(counts, np.uint8),
        centroid_mix=float(centroid_mix),
        old_class_count=int(old_class_count),
        support_audit=dict(support_audit),
    )
    if head.trainable_parameters > MAX_TRAINABLE_PARAMETERS:
        raise CollisionAwareMultiPrototypeError(
            "D5 trainable parameter cap exceeded"
        )
    if head.persistent_state_bytes_fp16 > MAX_PERSISTENT_STATE_BYTES:
        raise CollisionAwareMultiPrototypeError(
            "D5 persistent state cap exceeded"
        )
    return head


def fit_collision_aware_multiproto(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    config: D5Config = D5Config(),
) -> CollisionAwareMultiPrototypeHead:
    """Fit one scenario-local D5 head from registered physical support only."""

    _validate_config(config)
    rows, labels, sample_ids = _validate_support(
        support_features, support_labels, physical_sample_ids
    )
    scale = _fit_residual_scale(_normalize(rows), labels, config)
    transformed = _normalize(rows * scale[None, :])
    models, evidence = _build_class_models(transformed, labels, config)
    classes = tuple(sorted(models))
    audit = {
        "schema": "cvs.phase2.d5_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_true_batch_class_count_used": False,
        "query_class_quota_used": False,
        "query_global_assignment_used": False,
        "dense_query_graph_used": False,
        "role_symmetric_scoring_rule": True,
        "registration_lifecycle_used_only_for_old_state_lock": False,
        "physical_support_sample_count": int(len(sample_ids)),
        "physical_support_ids_unique": True,
        "computation_views_used_as_physical_loo": False,
        "class_count": len(classes),
        "class_gram": np.asarray(evidence["class_gram"]).tolist(),
        "local_1nn": evidence["local_1nn"],
        "per_class": evidence["per_class"],
        "config": json.loads(json.dumps(config.__dict__)),
    }
    return _assemble_head(
        classes=classes,
        residual_scale=scale,
        models=models,
        centroid_mix=config.centroid_mix,
        old_class_count=len(classes),
        support_audit=audit,
    )


def _models_from_head(
    head: CollisionAwareMultiPrototypeHead,
) -> dict[str, dict[str, Any]]:
    result = {}
    for index, class_handle in enumerate(head.classes.astype(str).tolist()):
        mask = head.prototype_mask[index].astype(bool)
        result[class_handle] = {
            "prototypes": head.prototypes[index, mask].copy(),
            "centroid": head.centroids[index].copy(),
            "penalty": float(head.class_penalty[index]),
        }
    return result


def _score_models(
    rows: np.ndarray,
    class_order: Sequence[str],
    models: Mapping[str, Mapping[str, Any]],
    *,
    centroid_mix: float,
) -> np.ndarray:
    columns = [
        _class_score(
            rows,
            np.asarray(models[class_handle]["prototypes"]),
            np.asarray(models[class_handle]["centroid"]),
            centroid_mix=centroid_mix,
            penalty=float(models[class_handle]["penalty"]),
        )
        for class_handle in class_order
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def _bounded_new_deconfusion(
    *,
    parent: CollisionAwareMultiPrototypeHead,
    transformed: np.ndarray,
    labels: np.ndarray,
    new_classes: Sequence[str],
    new_models: Mapping[str, Mapping[str, Any]],
    config: D5Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    old_classes = parent.classes.astype(str).tolist()
    old_models = _models_from_head(parent)
    old_mask = np.isin(labels, old_classes)
    new_mask = np.isin(labels, list(new_classes))
    old_rows = transformed[old_mask]
    old_labels = labels[old_mask]
    new_rows = transformed[new_mask]
    new_labels = labels[new_mask]
    old_scores = _score_models(
        old_rows,
        old_classes,
        old_models,
        centroid_mix=parent.centroid_mix,
    )
    old_truth = np.asarray(
        [old_classes.index(value) for value in old_labels.tolist()],
        dtype=np.int64,
    )
    before_prediction = np.argmax(old_scores, axis=1)
    before_correct = before_prediction == old_truth
    reference_centroids = np.stack(
        [
            old_models[value]["centroid"]
            for value in old_classes
        ]
        + [
            new_models[value]["centroid"]
            for value in new_classes
        ]
    )
    class_order = old_classes + list(new_classes)
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for alpha in config.deconfusion_steps:
        adjusted: dict[str, dict[str, Any]] = {}
        for class_handle in new_classes:
            model = new_models[class_handle]
            center = np.asarray(model["centroid"], dtype=np.float32)
            other = reference_centroids[
                [
                    index
                    for index, value in enumerate(class_order)
                    if value != class_handle
                ]
            ]
            adjusted_prototypes = []
            for prototype in np.asarray(model["prototypes"], dtype=np.float32):
                nearest = other[int(np.argmax(other @ prototype))]
                shifted = _normalize(
                    (
                        prototype
                        + float(alpha) * (prototype - nearest)
                    )[None, :]
                )[0]
                adjusted_prototypes.append(shifted)
            adjusted[class_handle] = {
                "prototypes": np.stack(adjusted_prototypes),
                "centroid": center.copy(),
                "penalty": float(model["penalty"]),
            }
        combined = {**old_models, **adjusted}
        # Support-floor penalty is class-local and derived only from old
        # registered support. Old state remains bitwise unchanged.
        new_score = _score_models(
            old_rows,
            list(new_classes),
            adjusted,
            centroid_mix=parent.centroid_mix,
        )
        floor_penalties: dict[str, float] = {}
        correct_indices = np.flatnonzero(before_correct)
        for new_index, class_handle in enumerate(new_classes):
            required = 0.0
            if len(correct_indices):
                required = max(
                    0.0,
                    float(
                        np.max(
                            new_score[correct_indices, new_index]
                            - old_scores[
                                correct_indices,
                                old_truth[correct_indices],
                            ]
                            + 1.0e-6
                        )
                    ),
                )
            base_penalty = float(adjusted[class_handle]["penalty"])
            final_penalty = min(
                config.max_floor_penalty,
                max(
                    base_penalty,
                    base_penalty
                    + required
                    + config.new_floor_safety_margin,
                ),
            )
            adjusted[class_handle]["penalty"] = final_penalty
            floor_penalties[class_handle] = final_penalty
        combined = {**old_models, **adjusted}
        combined_old_scores = _score_models(
            old_rows,
            class_order,
            combined,
            centroid_mix=parent.centroid_mix,
        )
        old_after = np.argmax(combined_old_scores, axis=1)
        old_intrusions = int(
            np.sum(before_correct & (old_after != old_truth))
        )
        combined_new_scores = _score_models(
            new_rows,
            class_order,
            combined,
            centroid_mix=parent.centroid_mix,
        )
        new_truth = np.asarray(
            [class_order.index(value) for value in new_labels.tolist()],
            dtype=np.int64,
        )
        new_correct = np.argmax(combined_new_scores, axis=1) == new_truth
        per_new_accuracy = {
            class_handle: float(
                np.mean(new_correct[new_labels == class_handle])
            )
            for class_handle in new_classes
        }
        new_margins = []
        for row_index, truth_index in enumerate(new_truth):
            row = combined_new_scores[row_index]
            new_margins.append(
                float(row[truth_index] - np.max(np.delete(row, truth_index)))
            )
        mean_penalty = float(np.mean(list(floor_penalties.values())))
        evidence = {
            "deconfusion_alpha": float(alpha),
            "old_before_correct_after_intruded": old_intrusions,
            "old_support_accuracy_before": float(
                np.mean(before_prediction == old_truth)
            ),
            "old_support_accuracy_after": float(
                np.mean(old_after == old_truth)
            ),
            "min_new_support_accuracy": min(per_new_accuracy.values()),
            "overall_new_support_accuracy": float(np.mean(new_correct)),
            "worst_new_support_margin": float(np.min(new_margins)),
            "mean_new_class_penalty": mean_penalty,
            "new_floor_safety_margin": float(
                config.new_floor_safety_margin
            ),
            "new_class_penalty": floor_penalties,
            "old_state_mutation_count": 0,
        }
        ranking = (
            1.0 if old_intrusions == 0 else 0.0,
            evidence["min_new_support_accuracy"],
            evidence["overall_new_support_accuracy"],
            evidence["worst_new_support_margin"],
            -mean_penalty,
            -float(alpha),
        )
        candidates.append(
            (ranking, {"models": adjusted, "evidence": evidence})
        )
    selected = max(candidates, key=lambda item: item[0])[1]
    return dict(selected["models"]), {
        "schema": "cvs.phase2.d5_bounded_deconfusion.v1",
        "support_only": True,
        "query_rows_used": 0,
        "old_head_bitwise_locked": True,
        "selected": selected["evidence"],
        "candidate_diagnostics": [
            candidate[1]["evidence"] for candidate in candidates
        ],
    }


def extend_collision_aware_multiproto(
    parent: CollisionAwareMultiPrototypeHead,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    config: D5Config = D5Config(),
) -> CollisionAwareMultiPrototypeHead:
    """Append new lifecycle classes while locking the parent old head bits."""

    _validate_config(config)
    rows, labels, sample_ids = _validate_support(
        support_features, support_labels, physical_sample_ids
    )
    old_classes = parent.classes.astype(str).tolist()
    all_classes = sorted(set(labels.tolist()))
    if not set(old_classes).issubset(all_classes):
        raise CollisionAwareMultiPrototypeError(
            "after support must retain every parent registered class"
        )
    new_classes = sorted(set(all_classes) - set(old_classes))
    if not new_classes:
        raise CollisionAwareMultiPrototypeError(
            "after registry must add at least one class"
        )
    if rows.shape[1] != parent.feature_dim:
        raise CollisionAwareMultiPrototypeError(
            "after support feature dimension differs from parent"
        )
    transformed = _normalize(rows * parent.residual_scale[None, :])
    new_models, evidence = _build_class_models(
        transformed,
        labels,
        config,
        allowed_classes=new_classes,
    )
    adjusted_new, deconfusion = _bounded_new_deconfusion(
        parent=parent,
        transformed=transformed,
        labels=labels,
        new_classes=new_classes,
        new_models=new_models,
        config=config,
    )
    old_models = _models_from_head(parent)
    models = {**old_models, **adjusted_new}
    class_order = old_classes + new_classes
    audit = {
        "schema": "cvs.phase2.d5_support_audit.v1",
        "fit_scope": "registered_physical_support_only",
        "query_rows_used": 0,
        "query_labels_used": False,
        "query_roles_used": False,
        "query_true_batch_class_count_used": False,
        "query_class_quota_used": False,
        "query_global_assignment_used": False,
        "dense_query_graph_used": False,
        "role_symmetric_scoring_rule": True,
        "registration_lifecycle_used_only_for_old_state_lock": True,
        "old_head_bitwise_locked": True,
        "old_head_update_count": 0,
        "physical_support_sample_count": int(len(sample_ids)),
        "physical_support_ids_unique": True,
        "computation_views_used_as_physical_loo": False,
        "old_class_count": len(old_classes),
        "new_class_count": len(new_classes),
        "new_class_gram": np.asarray(evidence["class_gram"]).tolist(),
        "local_1nn": evidence["local_1nn"],
        "per_new_class": evidence["per_class"],
        "bounded_deconfusion": deconfusion,
        "config": json.loads(json.dumps(config.__dict__)),
    }
    head = _assemble_head(
        classes=class_order,
        residual_scale=parent.residual_scale.copy(),
        models=models,
        centroid_mix=parent.centroid_mix,
        old_class_count=len(old_classes),
        support_audit=audit,
    )
    if (
        not np.array_equal(
            head.prototypes[: len(old_classes)], parent.prototypes
        )
        or not np.array_equal(
            head.prototype_mask[: len(old_classes)], parent.prototype_mask
        )
        or not np.array_equal(
            head.centroids[: len(old_classes)], parent.centroids
        )
        or not np.array_equal(
            head.class_penalty[: len(old_classes)], parent.class_penalty
        )
        or not np.array_equal(head.residual_scale, parent.residual_scale)
    ):
        raise CollisionAwareMultiPrototypeError(
            "parent old head mutation detected"
        )
    return head


def score_collision_aware_multiproto(
    query_features: np.ndarray,
    head: CollisionAwareMultiPrototypeHead,
) -> np.ndarray:
    """Score each query independently over all registered classes."""

    rows = np.asarray(query_features, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != head.feature_dim
        or not np.isfinite(rows).all()
    ):
        raise CollisionAwareMultiPrototypeError(
            "query features must be finite [N,D] matching the head"
        )
    transformed = _normalize(rows * head.residual_scale[None, :])
    columns = []
    for index in range(head.class_count):
        mask = head.prototype_mask[index].astype(bool)
        columns.append(
            _class_score(
                transformed,
                head.prototypes[index, mask],
                head.centroids[index],
                centroid_mix=head.centroid_mix,
                penalty=float(head.class_penalty[index]),
            )
        )
    return np.stack(columns, axis=1).astype(np.float32)


def predict_collision_aware_multiproto(
    query_features: np.ndarray,
    head: CollisionAwareMultiPrototypeHead,
) -> np.ndarray:
    scores = score_collision_aware_multiproto(query_features, head)
    return head.classes[np.argmax(scores, axis=1)].astype(str)


def _wilson_lower(correct: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = float(correct) / float(total)
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt(
        (p * (1.0 - p) + z * z / (4.0 * total)) / total
    )
    return float((center - spread) / denominator)


def _support_predictions(
    head: CollisionAwareMultiPrototypeHead,
    rows: np.ndarray,
) -> np.ndarray:
    return predict_collision_aware_multiproto(rows, head)


def _old_support_floor_evidence(
    *,
    parent: CollisionAwareMultiPrototypeHead,
    child: CollisionAwareMultiPrototypeHead,
    rows: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    old_classes = parent.classes.astype(str).tolist()
    mask = np.isin(labels, old_classes)
    old_rows = rows[mask]
    old_labels = labels[mask]
    before = _support_predictions(parent, old_rows)
    after = _support_predictions(child, old_rows)
    before_correct = before == old_labels
    intrusions = int(np.sum(before_correct & (after != old_labels)))
    per_class = {}
    for class_handle in old_classes:
        selected = old_labels == class_handle
        correct = int(np.sum(after[selected] == old_labels[selected]))
        total = int(np.sum(selected))
        per_class[class_handle] = {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total),
            "wilson95_lower": _wilson_lower(correct, total),
        }
    return {
        "before_correct_after_intruded": intrusions,
        "per_class": per_class,
        "min_class_accuracy": min(
            value["accuracy"] for value in per_class.values()
        ),
        "min_class_wilson95_lower": min(
            value["wilson95_lower"] for value in per_class.values()
        ),
    }


def _new_physical_holdout_evidence(
    *,
    parent: CollisionAwareMultiPrototypeHead,
    rows: np.ndarray,
    labels: np.ndarray,
    physical_sample_ids: np.ndarray,
    config: D5Config,
) -> dict[str, Any]:
    old_classes = set(parent.classes.astype(str).tolist())
    new_classes = sorted(set(labels.tolist()) - old_classes)
    records: dict[str, list[tuple[bool, float, int]]] = {
        value: [] for value in new_classes
    }
    # Each class contributes physical leave-one-out folds and deterministic
    # adjacent leave-two-out folds when at least four physical samples exist.
    for class_handle in new_classes:
        indices = np.flatnonzero(labels == class_handle)
        folds = [(int(index),) for index in indices]
        if len(indices) >= 4:
            folds.extend(
                [
                    (int(indices[index]), int(indices[(index + 1) % len(indices)]))
                    for index in range(0, len(indices), 2)
                ]
            )
        for held in folds:
            keep = np.ones(len(rows), dtype=bool)
            keep[np.asarray(held, dtype=np.int64)] = False
            if not np.any(labels[keep] == class_handle):
                continue
            child = extend_collision_aware_multiproto(
                parent,
                rows[keep],
                labels[keep],
                physical_sample_ids=physical_sample_ids[keep],
                config=config,
            )
            held_rows = rows[np.asarray(held, dtype=np.int64)]
            scores = score_collision_aware_multiproto(held_rows, child)
            predictions = child.classes[np.argmax(scores, axis=1)].astype(str)
            truth_index = int(
                np.flatnonzero(
                    child.classes.astype(str) == class_handle
                )[0]
            )
            for local_index, prediction in enumerate(predictions.tolist()):
                row_score = scores[local_index]
                margin = float(
                    row_score[truth_index]
                    - np.max(np.delete(row_score, truth_index))
                )
                records[class_handle].append(
                    (prediction == class_handle, margin, len(held))
                )
    per_class = {}
    all_records: list[tuple[bool, float, int]] = []
    for class_handle, values in records.items():
        if not values:
            raise CollisionAwareMultiPrototypeError(
                "D6c new-class physical holdout evidence is empty"
            )
        all_records.extend(values)
        correct = int(sum(int(value[0]) for value in values))
        margins = np.asarray([value[1] for value in values], dtype=np.float64)
        per_class[class_handle] = {
            "evaluation_rows": len(values),
            "leave_one_rows": int(sum(value[2] == 1 for value in values)),
            "leave_two_rows": int(sum(value[2] == 2 for value in values)),
            "accuracy": float(correct / len(values)),
            "wilson95_lower": _wilson_lower(correct, len(values)),
            "mean_margin": float(np.mean(margins)),
            "worst_margin": float(np.min(margins)),
        }
    return {
        "per_class": per_class,
        "overall_accuracy": float(
            np.mean([value[0] for value in all_records])
        ),
        "min_class_accuracy": min(
            value["accuracy"] for value in per_class.values()
        ),
        "min_class_wilson95_lower": min(
            value["wilson95_lower"] for value in per_class.values()
        ),
        "worst_class_margin": min(
            value["worst_margin"] for value in per_class.values()
        ),
    }


def select_support_only_margin(
    parent: CollisionAwareMultiPrototypeHead,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    margin_candidates: Sequence[float] = (0.0, 0.004, 0.006, 0.008, 0.02),
    base_config: D5Config = D5Config(),
) -> SupportOnlyMarginSelection:
    """Select one D6c margin using registered support before query opening."""

    rows, labels, sample_ids = _validate_support(
        support_features, support_labels, physical_sample_ids
    )
    candidates = tuple(float(value) for value in margin_candidates)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(not 0.0 <= value <= 0.20 for value in candidates)
    ):
        raise CollisionAwareMultiPrototypeError(
            "D6c margin candidates must be unique values in [0,0.20]"
        )
    evidence_rows = []
    for margin in candidates:
        config = D5Config(
            **{
                **base_config.__dict__,
                "new_floor_safety_margin": float(margin),
            }
        )
        child = extend_collision_aware_multiproto(
            parent,
            rows,
            labels,
            physical_sample_ids=sample_ids,
            config=config,
        )
        old_evidence = _old_support_floor_evidence(
            parent=parent,
            child=child,
            rows=rows,
            labels=labels,
        )
        new_evidence = _new_physical_holdout_evidence(
            parent=parent,
            rows=rows,
            labels=labels,
            physical_sample_ids=sample_ids,
            config=config,
        )
        complexity = {
            "active_prototype_count": child.active_prototype_count,
            "persistent_state_bytes_fp16": child.persistent_state_bytes_fp16,
            "extra_macs_per_query": child.extra_macs_per_query,
            "margin_magnitude": float(margin),
        }
        ranking = (
            -int(old_evidence["before_correct_after_intruded"]),
            float(old_evidence["min_class_wilson95_lower"]),
            float(new_evidence["min_class_wilson95_lower"]),
            float(new_evidence["min_class_accuracy"]),
            float(new_evidence["worst_class_margin"]),
            float(new_evidence["overall_accuracy"]),
            -int(complexity["extra_macs_per_query"]),
            -float(margin),
        )
        evidence_rows.append(
            {
                "margin": float(margin),
                "ranking": list(ranking),
                "old_support": old_evidence,
                "new_physical_leave_one_two_out": new_evidence,
                "complexity": complexity,
            }
        )
    selected = max(
        evidence_rows, key=lambda value: tuple(value["ranking"])
    )
    payload = {
        "schema": "cvs.phase2.d6c_support_only_margin_selection.v1",
        "selection_scope": "registered_physical_support_only",
        "query_package_opened": False,
        "query_rows_used": 0,
        "query_truth_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "selected_margin": selected["margin"],
        "margin_candidates": list(candidates),
        "ranking_policy": [
            "minimize_before_correct_old_intrusion",
            "maximize_min_old_class_wilson95_lower",
            "maximize_min_new_class_holdout_wilson95_lower",
            "maximize_min_new_class_holdout_accuracy",
            "maximize_worst_new_class_holdout_margin",
            "maximize_overall_new_holdout_accuracy",
            "minimize_query_macs",
            "minimize_margin",
        ],
        "candidate_evidence": evidence_rows,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    selected_config = D5Config(
        **{
            **base_config.__dict__,
            "new_floor_safety_margin": float(selected["margin"]),
        }
    )
    return SupportOnlyMarginSelection(
        selected_margin=float(selected["margin"]),
        selected_config=selected_config,
        selection_evidence=payload,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def write_support_only_margin_commit(
    selection: SupportOnlyMarginSelection,
    output_root: str | Path,
) -> dict[str, Any]:
    """Write an immutable D6c lock and COMMIT before any query is opened."""

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    selection_path = root / "support_only_margin_selection.json"
    raw = (
        json.dumps(
            selection.selection_evidence,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    selection_path.write_bytes(raw)
    os.chmod(selection_path, stat.S_IREAD)
    selection_sha = hashlib.sha256(raw).hexdigest()
    commit = {
        "schema": "cvs.phase2.d6c_support_only_margin_commit.v1",
        "status": "SUPPORT_ONLY_MARGIN_LOCKED_BEFORE_QUERY_OPEN",
        "query_package_opened": False,
        "query_truth_used": False,
        "selected_margin": selection.selected_margin,
        "selection_artifact": selection_path.name,
        "selection_artifact_sha256": selection_sha,
        "selection_canonical_sha256": selection.canonical_sha256,
    }
    commit_raw = (
        json.dumps(
            commit,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    commit_path = root / "COMMIT.json"
    commit_path.write_bytes(commit_raw)
    os.chmod(commit_path, stat.S_IREAD)
    return {
        **commit,
        "commit_sha256": hashlib.sha256(commit_raw).hexdigest(),
        "commit_path": str(commit_path),
    }


def select_support_only_margin_scenario_atomic(
    parents: Mapping[str, CollisionAwareMultiPrototypeHead],
    support_by_scenario: Mapping[
        str, tuple[np.ndarray, Sequence[str], Sequence[str]]
    ],
    *,
    margin_candidates: Sequence[float] = (0.0, 0.004, 0.006, 0.008, 0.02),
    base_config: D5Config = D5Config(),
) -> SupportOnlyMarginSelection:
    """Lock one shared D6c margin from all scenario supports, without query."""

    if set(parents) != set(support_by_scenario) or not parents:
        raise CollisionAwareMultiPrototypeError(
            "D6c scenario parent/support keys must match"
        )
    per_scenario = {
        scenario: select_support_only_margin(
            parents[scenario],
            values[0],
            values[1],
            physical_sample_ids=values[2],
            margin_candidates=margin_candidates,
            base_config=base_config,
        )
        for scenario, values in support_by_scenario.items()
    }
    candidates = tuple(float(value) for value in margin_candidates)
    aggregate_rows = []
    for candidate_index, margin in enumerate(candidates):
        rows = [
            per_scenario[scenario].selection_evidence[
                "candidate_evidence"
            ][candidate_index]
            for scenario in sorted(per_scenario)
        ]
        ranking = (
            -sum(
                int(row["old_support"]["before_correct_after_intruded"])
                for row in rows
            ),
            min(
                float(row["old_support"]["min_class_wilson95_lower"])
                for row in rows
            ),
            min(
                float(
                    row["new_physical_leave_one_two_out"][
                        "min_class_wilson95_lower"
                    ]
                )
                for row in rows
            ),
            min(
                float(
                    row["new_physical_leave_one_two_out"][
                        "min_class_accuracy"
                    ]
                )
                for row in rows
            ),
            min(
                float(
                    row["new_physical_leave_one_two_out"][
                        "worst_class_margin"
                    ]
                )
                for row in rows
            ),
            float(
                np.mean(
                    [
                        row["new_physical_leave_one_two_out"][
                            "overall_accuracy"
                        ]
                        for row in rows
                    ]
                )
            ),
            -sum(
                int(row["complexity"]["extra_macs_per_query"])
                for row in rows
            ),
            -float(margin),
        )
        aggregate_rows.append(
            {
                "margin": margin,
                "ranking": list(ranking),
                "by_scenario": {
                    scenario: per_scenario[
                        scenario
                    ].selection_evidence["candidate_evidence"][
                        candidate_index
                    ]
                    for scenario in sorted(per_scenario)
                },
            }
        )
    selected = max(
        aggregate_rows, key=lambda value: tuple(value["ranking"])
    )
    payload = {
        "schema": (
            "cvs.phase2.d6c_scenario_atomic_support_only_margin_selection.v1"
        ),
        "selection_scope": "registered_physical_support_only",
        "scenario_atomic_fit": True,
        "cross_scenario_support_concat": False,
        "query_package_opened": False,
        "query_rows_used": 0,
        "query_truth_used": False,
        "query_roles_used": False,
        "query_quota_used": False,
        "selected_margin": selected["margin"],
        "margin_candidates": list(candidates),
        "scenario_count": len(per_scenario),
        "candidate_evidence": aggregate_rows,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return SupportOnlyMarginSelection(
        selected_margin=float(selected["margin"]),
        selected_config=D5Config(
            **{
                **base_config.__dict__,
                "new_floor_safety_margin": float(selected["margin"]),
            }
        ),
        selection_evidence=payload,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "CollisionAwareMultiPrototypeError",
    "CollisionAwareMultiPrototypeHead",
    "D5Config",
    "MAX_PROTOTYPES_PER_CLASS",
    "extend_collision_aware_multiproto",
    "fit_collision_aware_multiproto",
    "predict_collision_aware_multiproto",
    "score_collision_aware_multiproto",
    "select_support_only_margin",
    "select_support_only_margin_scenario_atomic",
    "write_support_only_margin_commit",
]
