"""D21 support-only single-prototype lifecycle for Stage2-B/C.

The module owns only target state built from already received LEO_weak support.
Stage2-B commits an immutable old-class snapshot.  Stage2-C can only append
new-class state; old prototypes, radii, and their scoring path remain bitwise
stable.  Query APIs are sample-local and expose no fitting, role, quota, or
batch-assignment surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_DIM = 160
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
SUPPORTED_K = (1, 5, 10, 20)
SCHEMA = "cvs.phase2.d21.prototype_lifecycle.v1"
EPS = 1.0e-8


class LifecycleError(ValueError):
    """Fail-closed lifecycle validation error."""


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise LifecycleError(f"expected one finite {FEATURE_DIM}-D vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPS:
        raise LifecycleError("zero or non-finite feature norm")
    return np.ascontiguousarray(vector / np.float32(norm), dtype=np.float32)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
        raise LifecycleError(f"support must have shape [N,{FEATURE_DIM}]")
    if not np.isfinite(matrix).all():
        raise LifecycleError("support contains non-finite features")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if bool(np.any(norms <= EPS)) or not np.isfinite(norms).all():
        raise LifecycleError("support contains zero-norm features")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _support_content_sha256(
    normalized_features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
) -> str:
    rows = sorted(
        (
            str(label),
            hashlib.sha256(np.ascontiguousarray(feature).tobytes()).hexdigest(),
        )
        for feature, label in zip(normalized_features, labels)
    )
    envelope = {
        "schema": "cvs.phase2.d21.normalized_support_content.v1",
        "classes": tuple(str(value) for value in classes),
        "rows": rows,
    }
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class LifecycleConfig:
    """Pre-registered D21 single-prototype and uncertainty controls."""

    radius_enabled: bool = True
    boundary_enabled: bool = True
    radius_prior: float = 0.20
    radius_min: float = 0.0
    radius_max: float = 2.0
    radius_shrink_offset: float = 4.0
    radius_penalty_weight: float = 0.10
    radius_penalty_clip: float = 0.05
    robust_trim_fraction: float = 0.20
    boundary_topk: int = 8
    boundary_min_collision_cosine: float = 0.60
    boundary_weight: float = 0.10
    boundary_penalty_clip: float = 0.05

    def validate(self) -> None:
        numeric = (
            self.radius_prior,
            self.radius_min,
            self.radius_max,
            self.radius_shrink_offset,
            self.radius_penalty_weight,
            self.radius_penalty_clip,
            self.robust_trim_fraction,
            self.boundary_min_collision_cosine,
            self.boundary_weight,
            self.boundary_penalty_clip,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise LifecycleError("lifecycle configuration must be finite")
        if (
            not isinstance(self.radius_enabled, (bool, np.bool_))
            or not isinstance(self.boundary_enabled, (bool, np.bool_))
            or not 0.0 <= self.radius_min <= self.radius_prior <= self.radius_max <= 2.0
            or not 0.0 < self.radius_shrink_offset <= 100.0
            or not 0.0 <= self.radius_penalty_weight <= 1.0
            or not 0.0 <= self.radius_penalty_clip <= 1.0
            or not 0.0 <= self.robust_trim_fraction < 0.50
            or isinstance(self.boundary_topk, (bool, np.bool_))
            or not isinstance(self.boundary_topk, (int, np.integer))
            or not 1 <= int(self.boundary_topk) <= FEATURE_DIM
            or not -1.0 < self.boundary_min_collision_cosine < 1.0
            or not 0.0 <= self.boundary_weight <= 1.0
            or not 0.0 <= self.boundary_penalty_clip <= 1.0
        ):
            raise LifecycleError("lifecycle configuration is out of range")


@dataclass(frozen=True)
class SparseCollisionBoundary:
    """At most one support-only sparse rival boundary for one new class."""

    new_class_index: int
    rival_class_index: int
    feature_indices: np.ndarray
    direction_values: np.ndarray
    midpoint_projection: float
    safe_threshold: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.feature_indices)
        values = np.asarray(self.direction_values)
        scalars = (self.midpoint_projection, self.safe_threshold)
        if (
            isinstance(self.new_class_index, (bool, np.bool_))
            or isinstance(self.rival_class_index, (bool, np.bool_))
            or int(self.new_class_index) < 1
            or int(self.rival_class_index) < 0
            or int(self.rival_class_index) == int(self.new_class_index)
            or indices.ndim != 1
            or values.shape != indices.shape
            or not 1 <= len(indices) <= FEATURE_DIM
            or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != len(indices)
            or bool(np.any(indices < 0))
            or bool(np.any(indices >= FEATURE_DIM))
            or not np.isfinite(values).all()
            or float(np.linalg.norm(values)) <= EPS
            or not all(math.isfinite(float(value)) for value in scalars)
            or not -2.0 <= float(self.safe_threshold) <= 2.0
        ):
            raise LifecycleError("sparse collision boundary drift")
        object.__setattr__(self, "new_class_index", int(self.new_class_index))
        object.__setattr__(self, "rival_class_index", int(self.rival_class_index))
        object.__setattr__(self, "feature_indices", _readonly(indices, np.int16))
        object.__setattr__(self, "direction_values", _readonly(values, np.float32))

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.feature_indices.nbytes
            + self.direction_values.nbytes
            + 2 * np.dtype(np.float32).itemsize
            + 2 * np.dtype(np.int16).itemsize
        )


@dataclass(frozen=True)
class PrototypeLifecycleState:
    schema: str
    stage: str
    classes: tuple[str, ...]
    old_class_count: int
    k_shot: int
    old_support_capsule_root_sha256: str
    old_support_content_sha256: str
    old_support_receipt_sha256: str
    current_support_capsule_root_sha256: str
    current_support_receipt_sha256: str
    prototypes: np.ndarray
    radii: np.ndarray
    radius_active: np.ndarray
    support_count_by_class: np.ndarray
    old_prototype_snapshot: np.ndarray
    old_radius_snapshot: np.ndarray
    old_radius_active_snapshot: np.ndarray
    center_policy: str
    radius_policy: str
    boundaries: tuple[SparseCollisionBoundary, ...]
    config: LifecycleConfig
    support_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        prototypes = np.asarray(self.prototypes)
        radii = np.asarray(self.radii)
        radius_active = np.asarray(self.radius_active)
        counts = np.asarray(self.support_count_by_class)
        old_prototypes = np.asarray(self.old_prototype_snapshot)
        old_radii = np.asarray(self.old_radius_snapshot)
        old_radius_active = np.asarray(self.old_radius_active_snapshot)
        old_count = int(self.old_class_count)
        boundaries = tuple(self.boundaries)
        boundary_new_indices = [item.new_class_index for item in boundaries]
        if (
            self.schema != SCHEMA
            or self.stage not in ("stage2b_old_snapshot", "stage2c_append_only")
            or not classes
            or len(set(classes)) != len(classes)
            or any(not value for value in classes)
            or old_count < 2
            or old_count > len(classes)
            or int(self.k_shot) not in SUPPORTED_K
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value.lower())
                for value in (
                    self.old_support_capsule_root_sha256,
                    self.old_support_content_sha256,
                    self.old_support_receipt_sha256,
                    self.current_support_capsule_root_sha256,
                    self.current_support_receipt_sha256,
                )
            )
            or prototypes.shape != (len(classes), FEATURE_DIM)
            or radii.shape != (len(classes),)
            or radius_active.shape != (len(classes),)
            or radius_active.dtype != np.bool_
            or counts.shape != (len(classes),)
            or old_prototypes.shape != (old_count, FEATURE_DIM)
            or old_radii.shape != (old_count,)
            or old_radius_active.shape != (old_count,)
            or old_radius_active.dtype != np.bool_
            or not np.array_equal(prototypes[:old_count], old_prototypes)
            or not np.array_equal(radii[:old_count], old_radii)
            or not np.array_equal(radius_active[:old_count], old_radius_active)
            or bool(np.any(counts != int(self.k_shot)))
            or bool(np.any(radii < self.config.radius_min))
            or bool(np.any(radii > self.config.radius_max))
            or not all(
                np.isfinite(value).all()
                for value in (prototypes, radii, old_prototypes, old_radii)
            )
            or not np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1.0e-5)
            or len(boundary_new_indices) != len(set(boundary_new_indices))
            or any(index < old_count or index >= len(classes) for index in boundary_new_indices)
            or any(
                item.rival_class_index >= len(classes)
                or item.rival_class_index == item.new_class_index
                for item in boundaries
            )
            or not isinstance(self.support_audit, Mapping)
        ):
            raise LifecycleError("prototype lifecycle state drift")
        if self.stage == "stage2b_old_snapshot" and len(classes) != old_count:
            raise LifecycleError("Stage2-B state cannot contain registered new classes")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "k_shot", int(self.k_shot))
        object.__setattr__(self, "prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(self, "radii", _readonly(radii, np.float32))
        object.__setattr__(self, "radius_active", _readonly(radius_active, np.bool_))
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.int16))
        object.__setattr__(
            self, "old_prototype_snapshot", _readonly(old_prototypes, np.float32)
        )
        object.__setattr__(self, "old_radius_snapshot", _readonly(old_radii, np.float32))
        object.__setattr__(
            self,
            "old_radius_active_snapshot",
            _readonly(old_radius_active, np.bool_),
        )
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "support_audit", MappingProxyType(dict(self.support_audit)))
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise LifecycleError("prototype lifecycle state exceeds 256KiB")

    @property
    def persistent_numeric_state_bytes(self) -> int:
        array_bytes = sum(
            value.nbytes
            for value in (
                self.prototypes,
                self.radii,
                self.radius_active,
                self.support_count_by_class,
                self.old_prototype_snapshot,
                self.old_radius_snapshot,
                self.old_radius_active_snapshot,
            )
        )
        boundary_bytes = sum(item.persistent_state_bytes for item in self.boundaries)
        return int(array_bytes + boundary_bytes)

    @property
    def serialized_metadata_estimate_bytes(self) -> int:
        metadata = {
            "schema": self.schema,
            "stage": self.stage,
            "classes": self.classes,
            "old_class_count": self.old_class_count,
            "k_shot": self.k_shot,
            "old_support_capsule_root_sha256": self.old_support_capsule_root_sha256,
            "old_support_content_sha256": self.old_support_content_sha256,
            "old_support_receipt_sha256": self.old_support_receipt_sha256,
            "current_support_capsule_root_sha256": self.current_support_capsule_root_sha256,
            "current_support_receipt_sha256": self.current_support_receipt_sha256,
            "center_policy": self.center_policy,
            "radius_policy": self.radius_policy,
            "config": vars(self.config),
        }
        return len(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )

    @property
    def support_audit_artifact_bytes(self) -> int:
        def plain(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        return len(
            json.dumps(
                plain(self.support_audit), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )

    @property
    def persistent_state_bytes(self) -> int:
        metadata = self.serialized_metadata_estimate_bytes
        return int(self.persistent_numeric_state_bytes + metadata)

    def resource_audit(self) -> dict[str, Any]:
        class_count = len(self.classes)
        boundary_macs = sum(2 * len(item.feature_indices) for item in self.boundaries)
        active_radius_count = int(np.sum(self.radius_active))
        scalar_ops = 5 * active_radius_count
        return {
            "schema": SCHEMA,
            "trainable_parameters": 0,
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
            "query_batch_interaction": False,
            "dense_query_graph_bytes": 0,
            "prototype_count_per_class_max": 1,
            "sparse_collision_boundary_count": len(self.boundaries),
            "sparse_collision_boundaries_per_new_class_max": 1,
            "radius_config_enabled": bool(self.config.radius_enabled),
            "radius_active_class_count": active_radius_count,
            "radius_active_mask": tuple(bool(value) for value in self.radius_active),
            "boundary_config_enabled": bool(self.config.boundary_enabled),
            "boundary_effective_enabled": bool(
                self.config.boundary_enabled and self.k_shot != 1
            ),
            "boundary_k1_forced_off": self.k_shot == 1,
            "old_prototype_radius_score_path_bitwise_locked": True,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_numeric_state_bytes": self.persistent_numeric_state_bytes,
            "serialized_metadata_estimate_bytes": self.serialized_metadata_estimate_bytes,
            "support_audit_artifact_bytes": self.support_audit_artifact_bytes,
            "persistent_state_excludes_support_audit_artifact": True,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "estimated_head_macs_per_query": int(
                FEATURE_DIM * class_count + boundary_macs + scalar_ops
            ),
            "dense_query_graph": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "source_sample_access": False,
            "clean_sample_access": False,
        }


def _validated_support(
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
    *,
    expected_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    features = _normalize_rows(support_z_id)
    labels = np.asarray([str(value) for value in support_labels], dtype=np.str_)
    class_tuple = tuple(str(value) for value in classes)
    if (
        labels.shape != (len(features),)
        or not class_tuple
        or len(set(class_tuple)) != len(class_tuple)
        or any(not value for value in class_tuple)
        or set(labels.tolist()) != set(class_tuple)
    ):
        raise LifecycleError("support registry or labels drift")
    counts = [int(np.count_nonzero(labels == value)) for value in class_tuple]
    if len(set(counts)) != 1 or counts[0] not in SUPPORTED_K:
        raise LifecycleError("support must expose exact K in {1,5,10,20} per class")
    k_shot = counts[0]
    if expected_k is not None and k_shot != int(expected_k):
        raise LifecycleError("registered support K does not match old snapshot K")
    return features, labels, class_tuple, k_shot


def _center(rows: np.ndarray, policy: str, trim_fraction: float) -> np.ndarray:
    if policy == "mean":
        return _normalize_vector(np.mean(rows, axis=0, dtype=np.float32))
    similarity = rows @ rows.T
    medoid_index = int(np.argmax(np.sum(similarity, axis=1, dtype=np.float32)))
    if policy == "medoid":
        return np.array(rows[medoid_index], dtype=np.float32, copy=True)
    if policy == "robust_trim":
        keep = max(2, int(math.ceil((1.0 - trim_fraction) * len(rows))))
        order = np.argsort(similarity[medoid_index], kind="stable")[-keep:]
        return _normalize_vector(np.mean(rows[order], axis=0, dtype=np.float32))
    raise LifecycleError("unknown center policy")


def _candidate_policies(k_shot: int) -> tuple[str, ...]:
    if k_shot == 1:
        return ("mean",)
    if k_shot == 5:
        return ("mean", "medoid")
    return ("mean", "medoid", "robust_trim")


def _held_rank_folds(k_shot: int) -> tuple[tuple[int, ...], ...]:
    if k_shot == 1:
        return ()
    if k_shot == 5:
        return tuple((index,) for index in range(k_shot))
    return tuple((index, index + 1) for index in range(0, k_shot, 2))


def _rows_by_class(
    features: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> tuple[np.ndarray, ...]:
    return tuple(np.ascontiguousarray(features[labels == value]) for value in classes)


def _select_center_policy(
    rows_by_class: tuple[np.ndarray, ...],
    k_shot: int,
    config: LifecycleConfig,
) -> tuple[str, Mapping[str, Any]]:
    policies = _candidate_policies(k_shot)
    if k_shot == 1:
        return "mean", MappingProxyType(
            {"candidate_metrics": {"mean": {"support_cv": "not_estimable_k1"}}}
        )
    folds = _held_rank_folds(k_shot)
    metrics: dict[str, dict[str, float]] = {}
    best_policy = policies[0]
    best_key = (-1.0, -1.0, -float("inf"), 0)
    for policy_index, policy in enumerate(policies):
        correct_by_class = np.zeros(len(rows_by_class), dtype=np.int64)
        total_by_class = np.zeros(len(rows_by_class), dtype=np.int64)
        margins: list[float] = []
        for held in folds:
            train_prototypes = []
            held_rows = []
            held_targets = []
            for class_index, rows in enumerate(rows_by_class):
                keep_mask = np.ones(k_shot, dtype=bool)
                keep_mask[list(held)] = False
                train_prototypes.append(
                    _center(rows[keep_mask], policy, config.robust_trim_fraction)
                )
                for rank in held:
                    held_rows.append(rows[rank])
                    held_targets.append(class_index)
            prototype_matrix = np.stack(train_prototypes)
            score = np.stack(held_rows) @ prototype_matrix.T
            predictions = np.argmax(score, axis=1)
            for row_index, target in enumerate(held_targets):
                total_by_class[target] += 1
                correct_by_class[target] += int(predictions[row_index] == target)
                rivals = np.delete(score[row_index], target)
                if rivals.size:
                    margins.append(float(score[row_index, target] - np.max(rivals)))
                else:
                    # A one-class append still selects among the pre-registered
                    # center candidates, using fold-held support compactness.
                    margins.append(float(score[row_index, target]))
        class_acc = correct_by_class / np.maximum(total_by_class, 1)
        minimum = float(np.min(class_acc))
        overall = float(np.sum(correct_by_class) / np.sum(total_by_class))
        mean_margin = float(np.mean(margins))
        metrics[policy] = {
            "min_class_accuracy": minimum,
            "overall_accuracy": overall,
            "mean_true_margin": mean_margin,
        }
        key = (minimum, overall, mean_margin, -policy_index)
        if key > best_key:
            best_key = key
            best_policy = policy
    return best_policy, MappingProxyType({"candidate_metrics": metrics})


def _estimate_radii(
    rows_by_class: tuple[np.ndarray, ...],
    policy: str,
    k_shot: int,
    config: LifecycleConfig,
) -> tuple[np.ndarray, str, float]:
    if k_shot == 1:
        value = np.full(len(rows_by_class), config.radius_prior, dtype=np.float32)
        return value, "fixed_preregistered_prior_k1", 0.0
    folds = _held_rank_folds(k_shot)
    empirical = []
    for rows in rows_by_class:
        distances = []
        for held in folds:
            keep_mask = np.ones(k_shot, dtype=bool)
            keep_mask[list(held)] = False
            center = _center(rows[keep_mask], policy, config.robust_trim_fraction)
            distances.extend((1.0 - rows[list(held)] @ center).tolist())
        empirical.append(float(np.quantile(np.asarray(distances, dtype=np.float32), 0.80)))
    effective = k_shot - (1 if k_shot == 5 else 2)
    shrink = float(effective) / (float(effective) + config.radius_shrink_offset)
    empirical_value = np.asarray(empirical, dtype=np.float32)
    value = np.sqrt(
        np.float32(shrink) * np.square(empirical_value)
        + np.float32(1.0 - shrink) * np.float32(config.radius_prior**2)
    )
    value = np.clip(value, config.radius_min, config.radius_max).astype(np.float32)
    policy_name = "loo_q80_shrunk" if k_shot == 5 else "lto_q80_shrunk"
    return value, policy_name, shrink


def _fit_support_state(
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
    *,
    config: LifecycleConfig,
    expected_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, Mapping[str, Any], int]:
    config.validate()
    features, labels, class_tuple, k_shot = _validated_support(
        support_z_id, support_labels, classes, expected_k=expected_k
    )
    rows_by_class = _rows_by_class(features, labels, class_tuple)
    center_policy, selection_audit = _select_center_policy(rows_by_class, k_shot, config)
    prototypes = np.stack(
        [_center(rows, center_policy, config.robust_trim_fraction) for rows in rows_by_class]
    ).astype(np.float32)
    radii, radius_policy, radius_shrink = _estimate_radii(
        rows_by_class, center_policy, k_shot, config
    )
    audit = {
        "selection_data": "registered_leo_weak_support_only",
        "center_policy": center_policy,
        "center_candidates": _candidate_policies(k_shot),
        "radius_policy": radius_policy,
        "radius_shrink": radius_shrink,
        "radius_shrink_space": "squared_radius_rms",
        "query_rows_used_for_fit": 0,
        "query_role_oracle_access": False,
        "query_class_quota_access": False,
        "candidate_metrics": selection_audit["candidate_metrics"],
    }
    return features, labels, prototypes, center_policy, radius_policy, audit, k_shot


def _radius_adjusted_scores(
    cosine: np.ndarray,
    radii: np.ndarray,
    radius_active: np.ndarray,
    config: LifecycleConfig,
) -> np.ndarray:
    value = np.ascontiguousarray(cosine, dtype=np.float32)
    active = np.asarray(radius_active, dtype=np.bool_)
    if not config.radius_enabled or not bool(np.any(active)):
        return value
    distance = np.float32(1.0) - value
    overflow = np.maximum(distance - radii, np.float32(0.0))
    penalty = np.minimum(
        np.float32(config.radius_penalty_weight) * overflow,
        np.float32(config.radius_penalty_clip),
    ).astype(np.float32)
    penalty = penalty * active.astype(np.float32)
    return np.ascontiguousarray(value - penalty, dtype=np.float32)


def _stage2b_radius_guard(
    rows_by_class: tuple[np.ndarray, ...],
    classes: Sequence[str],
    center_policy: str,
    radii: np.ndarray,
    k_shot: int,
    config: LifecycleConfig,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    inactive = np.zeros(len(classes), dtype=np.bool_)
    if not config.radius_enabled:
        return inactive, MappingProxyType(
            {
                "status": "OFF_BY_CONFIG",
                "self_excluded_support": False,
                "radius_active": False,
            }
        )
    if k_shot == 1:
        return inactive, MappingProxyType(
            {
                "status": "OFF_K1_SELF_EXCLUSION_NOT_ESTIMABLE",
                "self_excluded_support": False,
                "radius_active": False,
            }
        )
    held_rows = []
    held_labels = []
    baseline_rows = []
    for held in _held_rank_folds(k_shot):
        prototypes = []
        for rows in rows_by_class:
            keep = np.ones(k_shot, dtype=bool)
            keep[list(held)] = False
            prototypes.append(
                _center(rows[keep], center_policy, config.robust_trim_fraction)
            )
        prototype_matrix = np.stack(prototypes)
        for class_index, rows in enumerate(rows_by_class):
            for rank in held:
                held_rows.append(rows[rank])
                held_labels.append(str(classes[class_index]))
                baseline_rows.append(rows[rank] @ prototype_matrix.T)
    support = np.stack(held_rows).astype(np.float32)
    labels = np.asarray(held_labels, dtype=np.str_)
    baseline_scores = np.stack(baseline_rows).astype(np.float32)
    candidate_scores = _radius_adjusted_scores(
        baseline_scores,
        radii,
        np.ones(len(classes), dtype=np.bool_),
        config,
    )
    baseline_metrics = _support_guard_metrics(baseline_scores, labels, classes)
    candidate_metrics = _support_guard_metrics(candidate_scores, labels, classes)
    passed = _passes_support_guard(candidate_metrics, baseline_metrics)
    baseline_accuracy, baseline_margin = baseline_metrics
    candidate_accuracy, candidate_margin = candidate_metrics
    return np.full(len(classes), passed, dtype=np.bool_), MappingProxyType(
        {
            "status": "PASS_ACTIVE" if passed else "NONDEGRADATION_FAIL_OFF",
            "self_excluded_support": True,
            "support_rows": len(support),
            "baseline": "pure_cosine",
            "candidate": "radius_residual",
            "baseline_per_class_accuracy": tuple(
                float(value) for value in baseline_accuracy
            ),
            "candidate_per_class_accuracy": tuple(
                float(value) for value in candidate_accuracy
            ),
            "baseline_worst_true_margin": baseline_margin,
            "candidate_worst_true_margin": candidate_margin,
            "radius_active": passed,
            "criteria": (
                "every_class_accuracy_non_decreasing",
                "worst_true_margin_non_decreasing",
            ),
        }
    )


def fit_old_snapshot(
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    old_classes: Sequence[str],
    *,
    old_support_capsule_root_sha256: str,
    old_support_receipt_sha256: str,
    config: LifecycleConfig | None = None,
) -> PrototypeLifecycleState:
    """Commit the immutable Stage2-B old-class target snapshot."""

    chosen = config or LifecycleConfig()
    _, _, prototypes, center_policy, radius_policy, audit, k_shot = _fit_support_state(
        support_z_id, support_labels, old_classes, config=chosen
    )
    # Recompute radii through the common path and keep one canonical copy.
    features, labels, classes, _ = _validated_support(
        support_z_id, support_labels, old_classes, expected_k=k_shot
    )
    support_content_sha256 = _support_content_sha256(features, labels, classes)
    radii, _, _ = _estimate_radii(
        _rows_by_class(features, labels, classes), center_policy, k_shot, chosen
    )
    radius_active, radius_guard = _stage2b_radius_guard(
        _rows_by_class(features, labels, classes),
        classes,
        center_policy,
        radii,
        k_shot,
        chosen,
    )
    audit = dict(audit)
    audit.update(
        {
            "old_support_capsule_root_sha256": old_support_capsule_root_sha256,
            "old_support_content_sha256": support_content_sha256,
            "old_support_receipt_sha256": old_support_receipt_sha256,
            "stage2b_radius_support_guard": radius_guard,
            "boundary_policy": (
                "forced_off_k1_self_exclusion_not_estimable"
                if k_shot == 1
                else (
                    "enabled_subject_to_stage2c_support_guard"
                    if chosen.boundary_enabled
                    else "off_by_config"
                )
            ),
        }
    )
    counts = np.full(len(classes), k_shot, dtype=np.int16)
    return PrototypeLifecycleState(
        schema=SCHEMA,
        stage="stage2b_old_snapshot",
        classes=classes,
        old_class_count=len(classes),
        k_shot=k_shot,
        old_support_capsule_root_sha256=old_support_capsule_root_sha256,
        old_support_content_sha256=support_content_sha256,
        old_support_receipt_sha256=old_support_receipt_sha256,
        current_support_capsule_root_sha256=old_support_capsule_root_sha256,
        current_support_receipt_sha256=old_support_receipt_sha256,
        prototypes=prototypes,
        radii=radii,
        radius_active=radius_active,
        support_count_by_class=counts,
        old_prototype_snapshot=prototypes,
        old_radius_snapshot=radii,
        old_radius_active_snapshot=radius_active,
        center_policy=center_policy,
        radius_policy=radius_policy,
        boundaries=(),
        config=chosen,
        support_audit=audit,
    )


def _collision_boundary(
    registry_prototypes: np.ndarray,
    registry_classes: Sequence[str],
    config: LifecycleConfig,
    new_class_index: int,
    new_prototype: np.ndarray,
    new_rows: np.ndarray,
) -> SparseCollisionBoundary | None:
    if (
        not config.boundary_enabled
        or registry_prototypes.shape[0] < 2
        or registry_prototypes.shape[0] != len(registry_classes)
        or not 0 <= new_class_index < registry_prototypes.shape[0]
    ):
        return None
    similarity = registry_prototypes @ new_prototype
    rival_indices = [
        index for index in range(len(registry_classes)) if index != new_class_index
    ]
    # Similarity wins; class-name tie breaking makes the semantic decision
    # invariant to the input order of a multi-class registration call.
    rival_index = min(
        rival_indices,
        key=lambda index: (-float(similarity[index]), str(registry_classes[index])),
    )
    if float(similarity[rival_index]) < config.boundary_min_collision_cosine:
        return None
    dense_direction = np.asarray(
        new_prototype - registry_prototypes[rival_index], dtype=np.float32
    )
    topk = min(int(config.boundary_topk), FEATURE_DIM)
    indices = np.argpartition(np.abs(dense_direction), -topk)[-topk:]
    indices = np.sort(indices).astype(np.int16)
    values = dense_direction[indices]
    norm = float(np.linalg.norm(values))
    if norm <= EPS:
        return None
    values = np.ascontiguousarray(values / np.float32(norm), dtype=np.float32)
    midpoint = np.float32(0.5) * (
        new_prototype[indices] + registry_prototypes[rival_index, indices]
    )
    midpoint_projection = float(midpoint @ values)
    support_projection = new_rows[:, indices] @ values - np.float32(midpoint_projection)
    safe_threshold = min(0.0, float(np.quantile(support_projection, 0.10)))
    safe_threshold = float(np.clip(safe_threshold, -2.0, 2.0))
    return SparseCollisionBoundary(
        new_class_index=new_class_index,
        rival_class_index=rival_index,
        feature_indices=indices,
        direction_values=values,
        midpoint_projection=midpoint_projection,
        safe_threshold=safe_threshold,
    )


def _apply_boundaries_to_support_scores(
    scores: np.ndarray,
    normalized_support: np.ndarray,
    boundaries: Sequence[SparseCollisionBoundary],
    config: LifecycleConfig,
) -> np.ndarray:
    adjusted = np.array(scores, dtype=np.float32, copy=True, order="C")
    for boundary in boundaries:
        projection = (
            normalized_support[:, boundary.feature_indices]
            @ boundary.direction_values
            - np.float32(boundary.midpoint_projection)
        )
        shortfall = np.maximum(
            np.float32(boundary.safe_threshold) - projection, np.float32(0.0)
        )
        penalty = np.minimum(
            np.float32(config.boundary_weight) * shortfall,
            np.float32(config.boundary_penalty_clip),
        )
        adjusted[:, boundary.new_class_index] -= penalty.astype(np.float32)
    return np.ascontiguousarray(adjusted, dtype=np.float32)


def _support_guard_metrics(
    scores: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> tuple[np.ndarray, float]:
    index_by_class = {value: index for index, value in enumerate(classes)}
    targets = np.asarray(
        [index_by_class[str(value)] for value in support_labels], dtype=np.int64
    )
    predictions = np.argmax(scores, axis=1)
    per_class_accuracy = np.asarray(
        [
            np.mean(predictions[targets == class_index] == class_index)
            for class_index in range(len(classes))
        ],
        dtype=np.float64,
    )
    rival_scores = np.array(scores, dtype=np.float32, copy=True)
    rival_scores[np.arange(len(scores)), targets] = -np.inf
    true_scores = scores[np.arange(len(scores)), targets]
    worst_true_margin = float(np.min(true_scores - np.max(rival_scores, axis=1)))
    return per_class_accuracy, worst_true_margin


def _passes_support_guard(
    candidate_metrics: tuple[np.ndarray, float],
    baseline_metrics: tuple[np.ndarray, float],
) -> bool:
    candidate_accuracy, candidate_margin = candidate_metrics
    baseline_accuracy, baseline_margin = baseline_metrics
    tolerance = 1.0e-7
    return bool(
        np.all(candidate_accuracy + tolerance >= baseline_accuracy)
        and candidate_margin + tolerance >= baseline_margin
    )


def _guard_boundary_candidates(
    base_state: PrototypeLifecycleState,
    registered_support: np.ndarray,
    registered_labels: np.ndarray,
    candidates: Sequence[SparseCollisionBoundary],
) -> tuple[tuple[SparseCollisionBoundary, ...], Mapping[str, Any]]:
    base_scores = np.asarray(score_batch(base_state, registered_support))
    baseline_metrics = _support_guard_metrics(
        base_scores, registered_labels, base_state.classes
    )
    baseline_accuracy, baseline_margin = baseline_metrics
    accepted: list[SparseCollisionBoundary] = []
    decisions: dict[str, Mapping[str, Any]] = {}
    ordered = sorted(
        candidates, key=lambda item: str(base_state.classes[item.new_class_index])
    )
    for candidate in ordered:
        candidate_scores = _apply_boundaries_to_support_scores(
            base_scores, registered_support, (candidate,), base_state.config
        )
        candidate_metrics = _support_guard_metrics(
            candidate_scores, registered_labels, base_state.classes
        )
        candidate_accuracy, candidate_margin = candidate_metrics
        passed = _passes_support_guard(candidate_metrics, baseline_metrics)
        new_name = base_state.classes[candidate.new_class_index]
        decisions[new_name] = MappingProxyType(
            {
                "rival_class": base_state.classes[candidate.rival_class_index],
                "accepted": passed,
                "per_class_accuracy": tuple(float(x) for x in candidate_accuracy),
                "worst_true_margin": candidate_margin,
            }
        )
        if passed:
            accepted.append(candidate)

    # Independently safe candidates can interact.  The final combined state is
    # therefore guarded once more; conservative all-off is deterministic and
    # preserves the exact safe fallback when the combination does not pass.
    combined_passed = True
    final_metrics = baseline_metrics
    if accepted:
        combined_scores = _apply_boundaries_to_support_scores(
            base_scores, registered_support, accepted, base_state.config
        )
        combined_metrics = _support_guard_metrics(
            combined_scores, registered_labels, base_state.classes
        )
        combined_passed = _passes_support_guard(combined_metrics, baseline_metrics)
        if combined_passed:
            final_metrics = combined_metrics
        else:
            accepted = []
    final_accuracy, final_margin = final_metrics
    audit = MappingProxyType(
        {
            "data": "all_registered_leo_weak_support_only",
            "guard_directions": ("old_to_new", "new_to_old"),
            "support_rows": int(len(registered_support)),
            "class_count": len(base_state.classes),
            "baseline_per_class_accuracy": tuple(
                float(x) for x in baseline_accuracy
            ),
            "baseline_worst_true_margin": baseline_margin,
            "final_per_class_accuracy": tuple(float(x) for x in final_accuracy),
            "final_worst_true_margin": final_margin,
            "candidate_decisions": MappingProxyType(decisions),
            "combined_guard_passed": combined_passed,
            "accepted_boundary_count": len(accepted),
            "criteria": (
                "every_class_accuracy_non_decreasing",
                "worst_true_margin_non_decreasing",
            ),
        }
    )
    return tuple(accepted), audit


def _guard_new_radius_candidates(
    base_state: PrototypeLifecycleState,
    registered_support: np.ndarray,
    registered_labels: np.ndarray,
    new_class_indices: Sequence[int],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    new_active = np.zeros(len(new_class_indices), dtype=np.bool_)
    base_scores = np.asarray(score_batch(base_state, registered_support))
    baseline_metrics = _support_guard_metrics(
        base_scores, registered_labels, base_state.classes
    )
    baseline_accuracy, baseline_margin = baseline_metrics
    decisions: dict[str, Mapping[str, Any]] = {}
    if base_state.k_shot == 1:
        return new_active, MappingProxyType(
            {
                "status": "OFF_K1_ALL_CLASSES_CONSISTENT",
                "data": "all_registered_leo_weak_support_only",
                "accepted_radius_count": 0,
                "candidate_decisions": MappingProxyType({}),
            }
        )
    if not base_state.config.radius_enabled:
        return new_active, MappingProxyType(
            {
                "status": "OFF_BY_CONFIG",
                "data": "all_registered_leo_weak_support_only",
                "accepted_radius_count": 0,
                "candidate_decisions": MappingProxyType({}),
            }
        )
    normalized = _normalize_rows(registered_support)
    raw_cosine = normalized @ base_state.prototypes.T
    accepted_indices = []
    ordered = sorted(
        new_class_indices, key=lambda index: str(base_state.classes[index])
    )
    for class_index in ordered:
        candidate_scores = np.array(base_scores, copy=True)
        candidate_scores[:, class_index] = _radius_adjusted_scores(
            raw_cosine[:, class_index : class_index + 1],
            base_state.radii[class_index : class_index + 1],
            np.ones(1, dtype=np.bool_),
            base_state.config,
        )[:, 0]
        candidate_metrics = _support_guard_metrics(
            candidate_scores, registered_labels, base_state.classes
        )
        candidate_accuracy, candidate_margin = candidate_metrics
        passed = _passes_support_guard(candidate_metrics, baseline_metrics)
        decisions[base_state.classes[class_index]] = MappingProxyType(
            {
                "accepted": passed,
                "per_class_accuracy": tuple(
                    float(value) for value in candidate_accuracy
                ),
                "worst_true_margin": candidate_margin,
            }
        )
        if passed:
            accepted_indices.append(class_index)
    combined_passed = True
    final_metrics = baseline_metrics
    if accepted_indices:
        combined_scores = np.array(base_scores, copy=True)
        for class_index in accepted_indices:
            combined_scores[:, class_index] = _radius_adjusted_scores(
                raw_cosine[:, class_index : class_index + 1],
                base_state.radii[class_index : class_index + 1],
                np.ones(1, dtype=np.bool_),
                base_state.config,
            )[:, 0]
        combined_metrics = _support_guard_metrics(
            combined_scores, registered_labels, base_state.classes
        )
        combined_passed = _passes_support_guard(combined_metrics, baseline_metrics)
        if combined_passed:
            final_metrics = combined_metrics
        else:
            accepted_indices = []
    for class_index in accepted_indices:
        new_active[class_index - new_class_indices[0]] = True
    final_accuracy, final_margin = final_metrics
    return new_active, MappingProxyType(
        {
            "status": "PASS_WITH_PER_CLASS_MASK",
            "data": "all_registered_leo_weak_support_only",
            "baseline": "existing_old_path_plus_new_pure_cosine",
            "candidate": "new_class_radius_residual",
            "support_rows": len(registered_support),
            "baseline_per_class_accuracy": tuple(
                float(value) for value in baseline_accuracy
            ),
            "baseline_worst_true_margin": baseline_margin,
            "final_per_class_accuracy": tuple(float(value) for value in final_accuracy),
            "final_worst_true_margin": final_margin,
            "candidate_decisions": MappingProxyType(decisions),
            "combined_guard_passed": combined_passed,
            "accepted_radius_count": int(np.sum(new_active)),
            "criteria": (
                "every_class_accuracy_non_decreasing",
                "worst_true_margin_non_decreasing",
            ),
        }
    )


def _old_support_metrics(
    scores: np.ndarray,
    old_labels: np.ndarray,
    old_classes: Sequence[str],
) -> tuple[np.ndarray, float]:
    index = {str(value): offset for offset, value in enumerate(old_classes)}
    targets = np.asarray([index[str(value)] for value in old_labels], dtype=np.int64)
    predictions = np.argmax(scores, axis=1)
    per_class_accuracy = np.asarray(
        [
            np.mean(predictions[targets == class_index] == class_index)
            for class_index in range(len(old_classes))
        ],
        dtype=np.float64,
    )
    rival = np.array(scores, copy=True)
    rival[np.arange(len(scores)), targets] = -np.inf
    margin = scores[np.arange(len(scores)), targets] - np.max(rival, axis=1)
    return per_class_accuracy, float(np.min(margin))


def _guard_new_prototype_append(
    state: PrototypeLifecycleState,
    old_support: np.ndarray,
    old_labels: np.ndarray,
    new_support: np.ndarray,
    new_labels: np.ndarray,
    new_classes: Sequence[str],
    new_prototypes: np.ndarray,
) -> Mapping[str, Any]:
    original_old = state.classes[: state.old_class_count]
    old_mask = np.isin(old_labels, np.asarray(original_old))
    guarded_old_support = old_support[old_mask]
    guarded_old_labels = old_labels[old_mask]
    baseline_scores = np.ascontiguousarray(
        guarded_old_support @ state.old_prototype_snapshot.T, dtype=np.float32
    )
    baseline_metrics = _old_support_metrics(
        baseline_scores, guarded_old_labels, original_old
    )
    baseline_accuracy, baseline_margin = baseline_metrics
    decisions: dict[str, Mapping[str, Any]] = {}
    for class_index, class_name in enumerate(new_classes):
        intrusion = guarded_old_support @ new_prototypes[class_index : class_index + 1].T
        candidate_scores = np.concatenate([baseline_scores, intrusion], axis=1)
        candidate_metrics = _old_support_metrics(
            candidate_scores, guarded_old_labels, original_old
        )
        candidate_accuracy, candidate_margin = candidate_metrics
        passed = _passes_support_guard(candidate_metrics, baseline_metrics)
        decisions[str(class_name)] = MappingProxyType(
            {
                "accepted": passed,
                "old_per_class_accuracy": tuple(
                    float(value) for value in candidate_accuracy
                ),
                "old_worst_true_margin": candidate_margin,
            }
        )
        if not passed:
            raise LifecycleError(
                f"new prototype intrusion guard failed for {class_name}"
            )
    combined_intrusion = guarded_old_support @ new_prototypes.T
    combined_scores = np.concatenate([baseline_scores, combined_intrusion], axis=1)
    combined_metrics = _old_support_metrics(
        combined_scores, guarded_old_labels, original_old
    )
    combined_accuracy, combined_margin = combined_metrics
    combined_passed = _passes_support_guard(combined_metrics, baseline_metrics)
    if not combined_passed:
        raise LifecycleError("combined new prototype intrusion guard failed")

    new_true = np.empty(len(new_support), dtype=np.float32)
    old_rival = new_support @ state.old_prototype_snapshot.T
    new_rival = new_support @ new_prototypes.T
    new_target = {str(value): index for index, value in enumerate(new_classes)}
    new_to_new_margin = []
    for row_index, label in enumerate(new_labels):
        target = new_target[str(label)]
        new_true[row_index] = new_rival[row_index, target]
        other = np.delete(new_rival[row_index], target)
        if other.size:
            new_to_new_margin.append(float(new_true[row_index] - np.max(other)))
    return MappingProxyType(
        {
            "status": "PASS",
            "baseline": "stage2b_old_snapshot_pure_cosine_prototypes",
            "candidate": "pure_cosine_new_prototype_append",
            "evaluated_directions": ("old_to_new", "new_to_old", "new_to_new"),
            "old_to_new": MappingProxyType(
                {
                    "baseline_per_class_accuracy": tuple(
                        float(value) for value in baseline_accuracy
                    ),
                    "baseline_worst_true_margin": baseline_margin,
                    "combined_per_class_accuracy": tuple(
                        float(value) for value in combined_accuracy
                    ),
                    "combined_worst_true_margin": combined_margin,
                    "per_new_class_decisions": MappingProxyType(decisions),
                    "combined_passed": combined_passed,
                }
            ),
            "new_to_old": MappingProxyType(
                {
                    "worst_true_margin_vs_old": float(
                        np.min(new_true - np.max(old_rival, axis=1))
                    )
                }
            ),
            "new_to_new": MappingProxyType(
                {
                    "worst_true_margin_vs_other_new": (
                        min(new_to_new_margin) if new_to_new_margin else None
                    )
                }
            ),
            "criteria": (
                "each_new_prototype_old_class_accuracy_non_decreasing",
                "each_new_prototype_old_worst_margin_non_decreasing",
                "combined_new_prototypes_old_class_accuracy_non_decreasing",
                "combined_new_prototypes_old_worst_margin_non_decreasing",
            ),
        }
    )


def register_new_classes(
    state: PrototypeLifecycleState,
    old_support_z_id: np.ndarray,
    old_support_labels: Sequence[str],
    new_support_z_id: np.ndarray,
    new_support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    old_support_capsule_root_sha256: str,
    old_support_receipt_sha256: str,
    after_registration_capsule_root_sha256: str,
    after_registration_receipt_sha256: str,
) -> PrototypeLifecycleState:
    """Append Stage2-C classes with an all-registered-support safety guard."""

    if not isinstance(state, PrototypeLifecycleState):
        raise LifecycleError("valid old lifecycle state required")
    if old_support_capsule_root_sha256 != state.old_support_capsule_root_sha256:
        raise LifecycleError("old support capsule root SHA256 mismatch")
    if old_support_receipt_sha256 != state.old_support_receipt_sha256:
        raise LifecycleError("old support receipt SHA256 mismatch")
    new_classes = tuple(str(value) for value in registered_classes)
    if (
        not new_classes
        or len(set(new_classes)) != len(new_classes)
        or any(not value for value in new_classes)
        or bool(set(new_classes) & set(state.classes))
    ):
        raise LifecycleError("new registration must be non-empty, unique, and append-only")
    old_features, old_labels, old_registry, _ = _validated_support(
        old_support_z_id,
        old_support_labels,
        state.classes,
        expected_k=state.k_shot,
    )
    if old_registry != state.classes:
        raise LifecycleError("old support registry does not match lifecycle state")
    recomputed_old_content_sha256 = _support_content_sha256(
        old_features, old_labels, old_registry
    )
    if recomputed_old_content_sha256 != state.old_support_content_sha256:
        raise LifecycleError("old support content SHA256 mismatch")
    features, labels, new_prototypes, center_policy, radius_policy, audit, _ = (
        _fit_support_state(
            new_support_z_id,
            new_support_labels,
            new_classes,
            config=state.config,
            expected_k=state.k_shot,
        )
    )
    rows_by_class = _rows_by_class(features, labels, new_classes)
    new_radii, _, _ = _estimate_radii(
        rows_by_class, center_policy, state.k_shot, state.config
    )
    combined_prototypes = np.concatenate([state.prototypes, new_prototypes], axis=0)
    combined_classes = state.classes + new_classes
    candidate_boundaries = []
    # All new prototypes exist before rivalry selection.  Each new class sees
    # every other old/new class, so registration input order cannot hide a rival.
    if state.k_shot != 1:
        for offset, (prototype, rows) in enumerate(zip(new_prototypes, rows_by_class)):
            absolute_index = len(state.classes) + offset
            boundary = _collision_boundary(
                combined_prototypes,
                combined_classes,
                state.config,
                absolute_index,
                prototype,
                rows,
            )
            if boundary is not None:
                candidate_boundaries.append(boundary)
    combined_radii = np.concatenate([state.radii, new_radii])
    combined_counts = np.concatenate(
        [
            state.support_count_by_class,
            np.full(len(new_classes), state.k_shot, dtype=np.int16),
        ]
    )
    all_support = np.concatenate([old_features, features], axis=0)
    all_labels = np.concatenate([old_labels, labels], axis=0)
    prototype_intrusion_audit = _guard_new_prototype_append(
        state,
        old_features,
        old_labels,
        features,
        labels,
        new_classes,
        new_prototypes,
    )
    initial_radius_active = np.concatenate(
        [state.radius_active, np.zeros(len(new_classes), dtype=np.bool_)]
    )
    pre_guard_audit = dict(state.support_audit)
    pre_guard_audit.update(
        {
            "new_registration_rule": "append_only_single_support_prototype",
            "old_support_registry_validated": True,
            "old_support_k_validated": state.k_shot,
            "old_support_capsule_root_sha256_matched": True,
            "old_support_receipt_sha256_matched": True,
            "old_support_content_sha256_recomputed_matched": True,
            "new_prototype_intrusion_guard": prototype_intrusion_audit,
        }
    )
    base_radius_state = PrototypeLifecycleState(
        schema=SCHEMA,
        stage="stage2c_append_only",
        classes=combined_classes,
        old_class_count=state.old_class_count,
        k_shot=state.k_shot,
        old_support_capsule_root_sha256=state.old_support_capsule_root_sha256,
        old_support_content_sha256=state.old_support_content_sha256,
        old_support_receipt_sha256=state.old_support_receipt_sha256,
        current_support_capsule_root_sha256=after_registration_capsule_root_sha256,
        current_support_receipt_sha256=after_registration_receipt_sha256,
        prototypes=combined_prototypes,
        radii=combined_radii,
        radius_active=initial_radius_active,
        support_count_by_class=combined_counts,
        old_prototype_snapshot=state.old_prototype_snapshot,
        old_radius_snapshot=state.old_radius_snapshot,
        old_radius_active_snapshot=state.old_radius_active_snapshot,
        center_policy=state.center_policy,
        radius_policy=state.radius_policy,
        boundaries=state.boundaries,
        config=state.config,
        support_audit=pre_guard_audit,
    )
    new_indices = tuple(range(len(state.classes), len(combined_classes)))
    new_radius_active, new_radius_guard_audit = _guard_new_radius_candidates(
        base_radius_state, all_support, all_labels, new_indices
    )
    combined_radius_active = np.concatenate(
        [state.radius_active, new_radius_active]
    )
    radius_guard_state = PrototypeLifecycleState(
        schema=SCHEMA,
        stage="stage2c_append_only",
        classes=combined_classes,
        old_class_count=state.old_class_count,
        k_shot=state.k_shot,
        old_support_capsule_root_sha256=state.old_support_capsule_root_sha256,
        old_support_content_sha256=state.old_support_content_sha256,
        old_support_receipt_sha256=state.old_support_receipt_sha256,
        current_support_capsule_root_sha256=after_registration_capsule_root_sha256,
        current_support_receipt_sha256=after_registration_receipt_sha256,
        prototypes=combined_prototypes,
        radii=combined_radii,
        radius_active=combined_radius_active,
        support_count_by_class=combined_counts,
        old_prototype_snapshot=state.old_prototype_snapshot,
        old_radius_snapshot=state.old_radius_snapshot,
        old_radius_active_snapshot=state.old_radius_active_snapshot,
        center_policy=state.center_policy,
        radius_policy=state.radius_policy,
        boundaries=state.boundaries,
        config=state.config,
        support_audit=pre_guard_audit,
    )
    accepted_boundaries, boundary_guard_audit = _guard_boundary_candidates(
        radius_guard_state, all_support, all_labels, candidate_boundaries
    )
    boundaries = state.boundaries + accepted_boundaries
    combined_audit = dict(state.support_audit)
    combined_audit.update(
        {
            "new_registration_rule": "append_only_single_support_prototype",
            "old_prototype_radius_state_bitwise_unchanged": True,
            "old_score_path_bitwise_unchanged": True,
            "new_center_policy": center_policy,
            "new_radius_policy": radius_policy,
            "new_support_audit": audit,
            "old_support_registry_validated": True,
            "old_support_k_validated": state.k_shot,
            "old_support_capsule_root_sha256_matched": True,
            "old_support_receipt_sha256_matched": True,
            "old_support_content_sha256_recomputed_matched": True,
            "after_registration_capsule_root_sha256": (
                after_registration_capsule_root_sha256
            ),
            "after_registration_receipt_sha256": (
                after_registration_receipt_sha256
            ),
            "new_prototype_intrusion_guard": prototype_intrusion_audit,
            "new_radius_support_guard": new_radius_guard_audit,
            "boundary_support_guard": boundary_guard_audit,
            "sparse_collision_boundary_policy": (
                "forced_off_k1_self_exclusion_not_estimable"
                if state.k_shot == 1
                else "at_most_one_support_only_boundary_per_new_class"
                if state.config.boundary_enabled
                else "off_safe_fallback"
            ),
        }
    )
    return PrototypeLifecycleState(
        schema=SCHEMA,
        stage="stage2c_append_only",
        classes=combined_classes,
        old_class_count=state.old_class_count,
        k_shot=state.k_shot,
        old_support_capsule_root_sha256=state.old_support_capsule_root_sha256,
        old_support_content_sha256=state.old_support_content_sha256,
        old_support_receipt_sha256=state.old_support_receipt_sha256,
        current_support_capsule_root_sha256=after_registration_capsule_root_sha256,
        current_support_receipt_sha256=after_registration_receipt_sha256,
        prototypes=combined_prototypes,
        radii=combined_radii,
        radius_active=combined_radius_active,
        support_count_by_class=combined_counts,
        old_prototype_snapshot=state.old_prototype_snapshot,
        old_radius_snapshot=state.old_radius_snapshot,
        old_radius_active_snapshot=state.old_radius_active_snapshot,
        center_policy=state.center_policy,
        radius_policy=state.radius_policy,
        boundaries=boundaries,
        config=state.config,
        support_audit=combined_audit,
    )


def _old_scores(state: PrototypeLifecycleState, value: np.ndarray) -> np.ndarray:
    cosine = np.ascontiguousarray(value @ state.old_prototype_snapshot.T, dtype=np.float32)
    return _radius_adjusted_scores(
        cosine,
        state.old_radius_snapshot,
        state.old_radius_active_snapshot,
        state.config,
    )


def _new_scores(state: PrototypeLifecycleState, value: np.ndarray) -> np.ndarray:
    if len(state.classes) == state.old_class_count:
        return np.empty(0, dtype=np.float32)
    cosine = np.ascontiguousarray(
        value @ state.prototypes[state.old_class_count :].T, dtype=np.float32
    )
    scores = _radius_adjusted_scores(
        cosine,
        state.radii[state.old_class_count :],
        state.radius_active[state.old_class_count :],
        state.config,
    )
    if not state.config.boundary_enabled or state.k_shot == 1:
        return scores
    scores = np.array(scores, dtype=np.float32, copy=True)
    for boundary in state.boundaries:
        local_index = boundary.new_class_index - state.old_class_count
        projection = float(
            value[boundary.feature_indices] @ boundary.direction_values
            - boundary.midpoint_projection
        )
        shortfall = max(0.0, boundary.safe_threshold - projection)
        penalty = min(
            state.config.boundary_weight * shortfall,
            state.config.boundary_penalty_clip,
        )
        scores[local_index] = np.float32(scores[local_index] - np.float32(penalty))
    return np.ascontiguousarray(scores, dtype=np.float32)


def score_one(state: PrototypeLifecycleState, z_id: np.ndarray) -> np.ndarray:
    """Score one sample against every registered class without state updates."""

    if not isinstance(state, PrototypeLifecycleState):
        raise LifecycleError("valid lifecycle state required")
    value = _normalize_vector(z_id)
    old = _old_scores(state, value)
    new = _new_scores(state, value)
    scores = old if not len(new) else np.concatenate([old, new]).astype(np.float32)
    return _readonly(scores, np.float32)


def score_batch(state: PrototypeLifecycleState, z_id: np.ndarray) -> np.ndarray:
    """Vectorization wrapper with strict per-sample locality."""

    matrix = np.asarray(z_id, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM or not np.isfinite(matrix).all():
        raise LifecycleError(f"batch must have shape [N,{FEATURE_DIM}] and be finite")
    return _readonly(np.stack([score_one(state, row) for row in matrix]), np.float32)


def predict_one(state: PrototypeLifecycleState, z_id: np.ndarray) -> tuple[str, np.ndarray]:
    scores = score_one(state, z_id)
    return state.classes[int(np.argmax(scores))], scores


__all__ = [
    "FEATURE_DIM",
    "LifecycleConfig",
    "LifecycleError",
    "PrototypeLifecycleState",
    "SparseCollisionBoundary",
    "fit_old_snapshot",
    "predict_one",
    "register_new_classes",
    "score_batch",
    "score_one",
]
