"""D19 compatible int8-anchor fusion (CIAF) for Phase2.

CIAF is a zero-training, support-only prototype head.  Its only Phase1 input
is an immutable domain-by-class int8 centroid component.  It never accepts a
source sample, a sample-level source feature, or any query-side fitting
surface.  Enrollment has two explicit stages:

* old-class target support is fused with compatible compressed ground anchors;
* new classes are appended as one target-support prototype per class.

Every inference call scores one sample against every registered class.  The
API intentionally has no query label, role, quota, order, or batch-assignment
argument.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-8
FEATURE_DIM = 160
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
SCHEMA = "cvs.phase2.d19_ciaf.v1"


class CiafError(ValueError):
    """Raised when CIAF input, protocol, or state validation fails closed."""


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise CiafError(f"z_id must be finite [N,{FEATURE_DIM}]")
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norm <= EPS)):
        raise CiafError("z_id rows must have non-zero norm")
    return np.ascontiguousarray(rows / norm, dtype=np.float32)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    row = _normalize_rows(np.asarray(value, dtype=np.float32).reshape(1, -1))
    return row[0]


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = np.asarray(value, dtype=np.float64) - float(np.max(value))
    weight = np.exp(shifted)
    return np.ascontiguousarray(weight / np.sum(weight), dtype=np.float32)


@dataclass(frozen=True)
class Int8DomainClassComponent:
    """Immutable, aggregate-only Phase1 model-knowledge component."""

    domain_class_q: np.ndarray
    domain_class_scale: np.ndarray
    domain_class_mask: np.ndarray
    class_registry: tuple[str, ...]
    feature_schema: str = "ADV3B02:z_id:unit_l2:160:v1"

    def __post_init__(self) -> None:
        q = np.asarray(self.domain_class_q)
        scale = np.asarray(self.domain_class_scale)
        mask = np.asarray(self.domain_class_mask)
        classes = tuple(str(value) for value in self.class_registry)
        if q.dtype != np.int8 or q.ndim != 3 or q.shape[2] != FEATURE_DIM:
            raise CiafError(f"domain_class_q must be int8[D,C,{FEATURE_DIM}]")
        if scale.shape != q.shape[:2] or scale.dtype != np.float16:
            raise CiafError("domain_class_scale must be float16[D,C]")
        if mask.shape != q.shape[:2] or mask.dtype not in (np.uint8, np.bool_):
            raise CiafError("domain_class_mask must be uint8/bool[D,C]")
        if (
            q.shape[0] < 1
            or q.shape[1] < 2
            or len(classes) != q.shape[1]
            or len(set(classes)) != len(classes)
            or any(not value for value in classes)
            or str(self.feature_schema) != "ADV3B02:z_id:unit_l2:160:v1"
            or not np.isfinite(scale).all()
            or bool(np.any(scale <= 0.0))
            or bool(np.any((mask != 0) & (mask != 1)))
            or bool(np.any(q == -128))
            or bool(np.any(q[mask == 0] != 0))
            or bool(np.any(np.sum(mask != 0, axis=0) == 0))
        ):
            raise CiafError("int8 domain-class component drift")
        object.__setattr__(self, "domain_class_q", _readonly(q, np.int8))
        object.__setattr__(self, "domain_class_scale", _readonly(scale, np.float16))
        object.__setattr__(self, "domain_class_mask", _readonly(mask, np.uint8))
        object.__setattr__(self, "class_registry", classes)

    @property
    def state_bytes(self) -> int:
        return int(
            self.domain_class_q.nbytes
            + self.domain_class_scale.nbytes
            + self.domain_class_mask.nbytes
        )

    def dequantized_class_anchors(self, class_index: int) -> np.ndarray:
        """Return transient normalized anchors for one class; never persisted."""

        index = int(class_index)
        if index < 0 or index >= len(self.class_registry):
            raise CiafError("class index is out of range")
        active = self.domain_class_mask[:, index].astype(bool)
        value = self.domain_class_q[active, index].astype(np.float32)
        value *= self.domain_class_scale[active, index, None].astype(np.float32)
        return _normalize_rows(value)


@dataclass(frozen=True)
class CiafConfig:
    """One unified, class-symmetric CIAF rule set."""

    anchor_temperature: float = 0.20
    robustness_temperature: float = 0.20
    support_alpha_min: float = 0.25
    support_alpha_max: float = 0.95
    collision_threshold: float = 0.55
    collision_bias_weight: float = 0.12

    def validate(self) -> None:
        values = (
            self.anchor_temperature,
            self.robustness_temperature,
            self.support_alpha_min,
            self.support_alpha_max,
            self.collision_threshold,
            self.collision_bias_weight,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise CiafError("CIAF configuration must be finite")
        if (
            self.anchor_temperature <= 0.0
            or self.robustness_temperature <= 0.0
            or not 0.0 <= self.support_alpha_min <= self.support_alpha_max <= 1.0
            or not -1.0 < self.collision_threshold < 1.0
            or not 0.0 <= self.collision_bias_weight <= 1.0
        ):
            raise CiafError("CIAF configuration is out of range")


@dataclass(frozen=True)
class CiafState:
    """Immutable deployment state after old fit or new-class registration."""

    schema: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    score_bias: np.ndarray
    old_support_prototypes: np.ndarray
    old_support_alpha: np.ndarray
    old_support_robustness: np.ndarray
    old_domain_weights: np.ndarray
    support_count_by_class: np.ndarray
    old_class_count: int
    k_shot: int
    int8_component_state_bytes: int
    config: CiafConfig
    support_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        prototypes = np.asarray(self.prototypes)
        bias = np.asarray(self.score_bias)
        old_support = np.asarray(self.old_support_prototypes)
        alpha = np.asarray(self.old_support_alpha)
        robustness = np.asarray(self.old_support_robustness)
        domain_weights = np.asarray(self.old_domain_weights)
        counts = np.asarray(self.support_count_by_class)
        old_count = int(self.old_class_count)
        if (
            self.schema != SCHEMA
            or len(classes) < old_count
            or old_count < 2
            or len(set(classes)) != len(classes)
            or prototypes.shape != (len(classes), FEATURE_DIM)
            or bias.shape != (len(classes),)
            or old_support.shape != (old_count, FEATURE_DIM)
            or alpha.shape != (old_count,)
            or robustness.shape != (old_count,)
            or domain_weights.ndim != 2
            or domain_weights.shape[0] != old_count
            or counts.shape != (len(classes),)
            or int(self.k_shot) < 1
            or int(self.int8_component_state_bytes) < 1
            or not all(
                np.isfinite(value).all()
                for value in (
                    prototypes,
                    bias,
                    old_support,
                    alpha,
                    robustness,
                    domain_weights,
                )
            )
            or bool(np.any(alpha < 0.0))
            or bool(np.any(alpha > 1.0))
            or bool(np.any(robustness < 0.0))
            or bool(np.any(robustness > 1.0))
            or bool(np.any(bias < 0.0))
            or bool(np.any(counts != int(self.k_shot)))
            or not np.allclose(np.sum(domain_weights, axis=1), 1.0, atol=1e-6)
        ):
            raise CiafError("CIAF state drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(self, "score_bias", _readonly(bias, np.float32))
        object.__setattr__(self, "old_support_prototypes", _readonly(old_support, np.float32))
        object.__setattr__(self, "old_support_alpha", _readonly(alpha, np.float32))
        object.__setattr__(self, "old_support_robustness", _readonly(robustness, np.float32))
        object.__setattr__(self, "old_domain_weights", _readonly(domain_weights, np.float32))
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.int16))
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise CiafError("CIAF persistent state exceeds 256KiB")

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def persistent_state_bytes(self) -> int:
        # Count the arrays in their actual persistent dtype.  Ground anchors are
        # deliberately absent: dequantization is transient inside fit_old_ciaf.
        array_bytes = sum(
            value.nbytes
            for value in (
                self.prototypes,
                self.score_bias,
                self.old_support_prototypes,
                self.old_support_alpha,
                self.old_support_robustness,
                self.old_domain_weights,
                self.support_count_by_class,
            )
        )
        metadata_bytes = len(self.schema.encode("utf-8")) + sum(
            len(value.encode("utf-8")) for value in self.classes
        )
        return int(self.int8_component_state_bytes + array_bytes + metadata_bytes)

    def resource_audit(self) -> dict[str, Any]:
        return {
            "schema": "cvs.phase2.d19_ciaf.resource.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES,
            "int8_component_state_bytes": int(self.int8_component_state_bytes),
            "estimated_macs_per_query": int(self.class_count * FEATURE_DIM + 3 * FEATURE_DIM + self.class_count),
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "per_sample_all_registered_classes": True,
            "source_sample_access": False,
            "sample_level_source_feature_access": False,
            "int8_component_update_access": False,
            "persistent_full_precision_ground_anchor_count": 0,
            "ground_anchor_dequantization_policy": "transient_fit_only_not_persisted",
        }


def _support_matrix(
    z_id: np.ndarray,
    labels: Sequence[str],
    *,
    expected_classes: Sequence[str] | None = None,
    expected_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    features = _normalize_rows(np.asarray(z_id))
    label_array = np.asarray(tuple(str(value) for value in labels))
    if label_array.ndim != 1 or len(label_array) != len(features) or any(not value for value in label_array):
        raise CiafError("support labels and z_id rows must align")
    classes, counts = np.unique(label_array, return_counts=True)
    class_tuple = tuple(sorted(classes.tolist()))
    if len(class_tuple) < 1 or len(set(counts.tolist())) != 1:
        raise CiafError("support must use one class-symmetric physical K-shot")
    k_shot = int(counts[0])
    if expected_classes is not None and set(class_tuple) != set(expected_classes):
        raise CiafError("support class registry drift")
    if expected_k is not None and k_shot != int(expected_k):
        raise CiafError("support K-shot drift")
    return features, label_array, class_tuple, k_shot


def _support_prototypes(features: np.ndarray, labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    return np.stack([
        _normalize_vector(np.mean(features[labels == label], axis=0)) for label in classes
    ]).astype(np.float32)


def _class_robustness(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    support_prototypes: np.ndarray,
    class_index: int,
    temperature: float,
) -> tuple[float, float]:
    label = classes[class_index]
    rows = features[labels == label]
    if len(rows) == 1:
        return 0.0, float("-inf")
    competitors = np.delete(support_prototypes, class_index, axis=0)
    margins: list[float] = []
    for row_index, row in enumerate(rows):
        own = _normalize_vector(np.mean(np.delete(rows, row_index, axis=0), axis=0))
        own_score = float(row @ own)
        rival_score = float(np.max(competitors @ row))
        margins.append(own_score - rival_score)
    robust_margin = float(np.quantile(np.asarray(margins), 0.25))
    robustness = 1.0 / (1.0 + math.exp(-robust_margin / float(temperature)))
    return float(robustness), robust_margin


def fit_old_ciaf(
    component: Int8DomainClassComponent,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    *,
    config: CiafConfig = CiafConfig(),
) -> CiafState:
    """Fit old target-support prototypes and compatible int8 anchors."""

    if not isinstance(component, Int8DomainClassComponent):
        raise CiafError("immutable int8 domain-class component required")
    config.validate()
    features, labels, _, k_shot = _support_matrix(
        support_z_id,
        support_labels,
        expected_classes=component.class_registry,
    )
    classes = component.class_registry
    support_proto = _support_prototypes(features, labels, classes)
    anchors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    robustness: list[float] = []
    margins: list[float] = []
    alpha: list[float] = []
    for class_index, _ in enumerate(classes):
        domain_anchor = component.dequantized_class_anchors(class_index)
        similarity = domain_anchor @ support_proto[class_index]
        active_weight = _softmax(similarity / float(config.anchor_temperature))
        fused_anchor = _normalize_vector(active_weight @ domain_anchor)
        full_weight = np.zeros(component.domain_class_q.shape[0], dtype=np.float32)
        full_weight[component.domain_class_mask[:, class_index].astype(bool)] = active_weight
        score, margin = _class_robustness(
            features,
            labels,
            classes,
            support_proto,
            class_index,
            config.robustness_temperature,
        )
        class_alpha = config.support_alpha_min + (
            config.support_alpha_max - config.support_alpha_min
        ) * score
        anchors.append(fused_anchor)
        weights.append(full_weight)
        robustness.append(score)
        margins.append(margin)
        alpha.append(class_alpha)
    anchor_array = np.stack(anchors).astype(np.float32)
    alpha_array = np.asarray(alpha, dtype=np.float32)
    fused = _normalize_rows(
        alpha_array[:, None] * support_proto
        + (1.0 - alpha_array[:, None]) * anchor_array
    )
    audit = {
        "selection_data": "registered_support_only",
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "class_symmetric_rule": True,
        "support_robustness_rule": "class_q25_leave_one_out_cosine_margin_sigmoid",
        "k1_robustness_rule": "fixed_zero_then_alpha_min",
        "anchor_compatibility_rule": "softmax_cosine_over_active_domains",
        "old_support_robust_margin_by_class": {
            label: (None if not math.isfinite(margins[index]) else margins[index])
            for index, label in enumerate(classes)
        },
    }
    return CiafState(
        schema=SCHEMA,
        classes=classes,
        prototypes=fused,
        score_bias=np.zeros(len(classes), dtype=np.float32),
        old_support_prototypes=support_proto,
        old_support_alpha=alpha_array,
        old_support_robustness=np.asarray(robustness, dtype=np.float32),
        old_domain_weights=np.stack(weights),
        support_count_by_class=np.full(len(classes), k_shot, dtype=np.int16),
        old_class_count=len(classes),
        k_shot=k_shot,
        int8_component_state_bytes=component.state_bytes,
        config=config,
        support_audit=audit,
    )


def register_new_classes(
    state: CiafState,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    *,
    registered_classes: Sequence[str] | None = None,
) -> CiafState:
    """Append one prototype per new class without changing old score columns."""

    if not isinstance(state, CiafState):
        raise CiafError("valid CIAF state required")
    features, labels, discovered_classes, _ = _support_matrix(
        support_z_id,
        support_labels,
        expected_classes=registered_classes,
        expected_k=state.k_shot,
    )
    new_classes = (
        discovered_classes
        if registered_classes is None
        else tuple(str(value) for value in registered_classes)
    )
    if len(new_classes) != len(set(new_classes)) or set(new_classes) != set(discovered_classes):
        raise CiafError("registered new-class order drift")
    if set(new_classes) & set(state.classes):
        raise CiafError("new-class labels overlap the registered class set")
    new_proto = _support_prototypes(features, labels, new_classes)
    collision = np.max(new_proto @ state.old_support_prototypes.T, axis=1)
    normalized_collision = np.clip(
        (collision - state.config.collision_threshold)
        / (1.0 - state.config.collision_threshold),
        0.0,
        1.0,
    )
    new_bias = state.config.collision_bias_weight * normalized_collision
    audit = dict(state.support_audit)
    audit.update(
        {
            "new_registration_rule": "single_normalized_support_prototype_per_class",
            "collision_bias_source": "old_registered_support_prototypes_only",
            "collision_bias_rule": "uniform_thresholded_max_old_support_cosine",
            "new_collision_cosine_by_class": {
                label: float(collision[index]) for index, label in enumerate(new_classes)
            },
        }
    )
    return CiafState(
        schema=SCHEMA,
        classes=state.classes + new_classes,
        prototypes=np.concatenate([state.prototypes, new_proto], axis=0),
        score_bias=np.concatenate([state.score_bias, new_bias.astype(np.float32)]),
        old_support_prototypes=state.old_support_prototypes,
        old_support_alpha=state.old_support_alpha,
        old_support_robustness=state.old_support_robustness,
        old_domain_weights=state.old_domain_weights,
        support_count_by_class=np.concatenate([
            state.support_count_by_class,
            np.full(len(new_classes), state.k_shot, dtype=np.int16),
        ]),
        old_class_count=state.old_class_count,
        k_shot=state.k_shot,
        int8_component_state_bytes=state.int8_component_state_bytes,
        config=state.config,
        support_audit=audit,
    )


def score_one(state: CiafState, z_id: np.ndarray) -> np.ndarray:
    """Score one sample independently over all registered classes."""

    if not isinstance(state, CiafState):
        raise CiafError("valid CIAF state required")
    value = np.asarray(z_id, dtype=np.float32)
    if value.shape == (1, FEATURE_DIM):
        value = value[0]
    if value.shape != (FEATURE_DIM,):
        raise CiafError(f"score_one requires exactly one {FEATURE_DIM}-D z_id")
    normalized = _normalize_vector(value)
    scores = normalized @ state.prototypes.T - state.score_bias
    return _readonly(scores, np.float32)


def predict_one(state: CiafState, z_id: np.ndarray) -> tuple[str, np.ndarray]:
    """Return one label and its complete all-registered-class score vector."""

    scores = score_one(state, z_id)
    return state.classes[int(np.argmax(scores))], scores


__all__ = [
    "CiafConfig",
    "CiafError",
    "CiafState",
    "FEATURE_DIM",
    "Int8DomainClassComponent",
    "fit_old_ciaf",
    "predict_one",
    "register_new_classes",
    "score_one",
]
