"""D24 uncertainty-fused old classes and target-only new registration.

The deployment head consumes one immutable Phase1 int8 domain-by-class
component and registered LEO_weak support.  Ground anchors are dequantized
only while fitting old classes; no full-precision ground anchor is retained.
Target-old and target-new prototypes are stored in FP32 in the same normalized
160-D ADV3B02 identity space.  Public fitting surfaces intentionally expose no
query, truth-role, quota, global-assignment, clean, or source-sample input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Sequence

import numpy as np

from .stage2_ciaf import FEATURE_DIM, Int8DomainClassComponent


EPS = 1.0e-8
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
SCHEMA = "cvs.phase2.d24_uncertainty_proto_fusion.v1"


class UncertaintyProtoFusionError(ValueError):
    """Raised when D24 input, configuration, or immutable state drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise UncertaintyProtoFusionError(
            f"identity features must be finite [N,{FEATURE_DIM}]"
        )
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norms <= EPS)):
        raise UncertaintyProtoFusionError("identity features must have non-zero norm")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    return _normalize_rows(np.asarray(value, dtype=np.float32).reshape(1, -1))[0]


def _cosine_distance_rows(rows: np.ndarray, center: np.ndarray) -> np.ndarray:
    # Clipping removes only floating-point excursions outside the mathematical
    # cosine-distance range; it is not a learned or query-derived calibration.
    return np.clip(1.0 - rows.astype(np.float64) @ center.astype(np.float64), 0.0, 2.0)


def _q90_radius(rows: np.ndarray, center: np.ndarray, *, k1_r0: float | None) -> float:
    if len(rows) == 1 and k1_r0 is not None:
        return float(k1_r0)
    return float(np.quantile(_cosine_distance_rows(rows, center), 0.90))


def _component_sha256(component: Int8DomainClassComponent) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d24.int8_component.v1\0")
    for value in (
        component.domain_class_q,
        component.domain_class_scale,
        component.domain_class_mask,
    ):
        array = np.ascontiguousarray(value)
        dtype = array.dtype.str.encode("ascii")
        digest.update(struct.pack("<B", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack("<B", array.ndim))
        for dimension in array.shape:
            digest.update(struct.pack("<I", int(dimension)))
        digest.update(array.tobytes())
    for label in component.class_registry:
        encoded = label.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _old_prefix_sha256(
    *,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    radius: np.ndarray,
    count: np.ndarray,
    old_target_prototypes: np.ndarray,
    old_ground_radius: np.ndarray,
    old_target_radius: np.ndarray,
    old_target_weight: np.ndarray,
    int8_component_sha256: str,
) -> str:
    old_count = len(old_target_weight)
    digest = hashlib.sha256(b"cvs.phase2.d24.old_prefix.v1\0")
    digest.update(bytes.fromhex(int8_component_sha256))
    for label in classes[:old_count]:
        encoded = label.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded)))
        digest.update(encoded)
    for array in (
        prototypes[:old_count],
        radius[:old_count],
        count[:old_count],
        old_target_prototypes,
        old_ground_radius,
        old_target_radius,
        old_target_weight,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(struct.pack("<B", contiguous.ndim))
        for dimension in contiguous.shape:
            digest.update(struct.pack("<I", int(dimension)))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class UncertaintyFusionConfig:
    """Method-locked, class-symmetric D24 constants."""

    r0: float = 0.05
    r_min: float = 1.0e-3
    separation_margin: float = 0.0

    def validate(self) -> None:
        values = (self.r0, self.r_min, self.separation_margin)
        if not all(math.isfinite(float(value)) for value in values):
            raise UncertaintyProtoFusionError("D24 configuration must be finite")
        if not 0.0 <= self.r0 <= 2.0 or not 0.0 < self.r_min <= 2.0:
            raise UncertaintyProtoFusionError("D24 radius constants are out of range")
        if not 0.0 <= self.separation_margin < 2.0:
            raise UncertaintyProtoFusionError("D24 separation margin is out of range")


@dataclass(frozen=True)
class UncertaintyFusionState:
    """Immutable D24 state; Stage2-C may append only a new-class suffix."""

    schema: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    radius: np.ndarray
    support_count_by_class: np.ndarray
    old_class_count: int
    old_target_prototypes: np.ndarray
    old_ground_radius: np.ndarray
    old_target_radius: np.ndarray
    old_target_weight: np.ndarray
    int8_component_sha256: str
    int8_component_state_bytes: int
    old_prefix_sha256: str
    config: UncertaintyFusionConfig

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        prototypes = np.asarray(self.prototypes)
        radius = np.asarray(self.radius)
        counts = np.asarray(self.support_count_by_class)
        target = np.asarray(self.old_target_prototypes)
        ground_radius = np.asarray(self.old_ground_radius)
        target_radius = np.asarray(self.old_target_radius)
        target_weight = np.asarray(self.old_target_weight)
        old_count = int(self.old_class_count)
        if (
            self.schema != SCHEMA
            or old_count < 2
            or old_count > len(classes)
            or len(set(classes)) != len(classes)
            or any(not value for value in classes)
            or prototypes.dtype != np.float32
            or prototypes.shape != (len(classes), FEATURE_DIM)
            or radius.dtype != np.float32
            or radius.shape != (len(classes),)
            or counts.dtype != np.uint16
            or counts.shape != (len(classes),)
            or target.dtype != np.float32
            or target.shape != (old_count, FEATURE_DIM)
            or ground_radius.dtype != np.float32
            or ground_radius.shape != (old_count,)
            or target_radius.dtype != np.float32
            or target_radius.shape != (old_count,)
            or target_weight.dtype != np.float32
            or target_weight.shape != (old_count,)
            or not all(
                np.isfinite(value).all()
                for value in (prototypes, radius, target, ground_radius, target_radius, target_weight)
            )
            or bool(np.any(radius < 0.0))
            or bool(np.any(radius > 2.0))
            or bool(np.any(ground_radius < 0.0))
            or bool(np.any(target_radius < 0.0))
            or bool(np.any(target_weight < 0.0))
            or bool(np.any(target_weight > 1.0))
            or bool(np.any(counts < 1))
            or len(set(int(value) for value in counts.tolist())) != 1
            or not np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1.0e-5)
            or not np.allclose(np.linalg.norm(target, axis=1), 1.0, atol=1.0e-5)
            or len(str(self.int8_component_sha256)) != 64
            or int(self.int8_component_state_bytes) < 1
        ):
            raise UncertaintyProtoFusionError("D24 state drift")
        expected_prefix = _old_prefix_sha256(
            classes=classes,
            prototypes=prototypes,
            radius=radius,
            count=counts,
            old_target_prototypes=target,
            old_ground_radius=ground_radius,
            old_target_radius=target_radius,
            old_target_weight=target_weight,
            int8_component_sha256=str(self.int8_component_sha256),
        )
        if str(self.old_prefix_sha256) != expected_prefix:
            raise UncertaintyProtoFusionError("D24 old prefix SHA256 drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(self, "radius", _readonly(radius, np.float32))
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.uint16))
        object.__setattr__(self, "old_target_prototypes", _readonly(target, np.float32))
        object.__setattr__(self, "old_ground_radius", _readonly(ground_radius, np.float32))
        object.__setattr__(self, "old_target_radius", _readonly(target_radius, np.float32))
        object.__setattr__(self, "old_target_weight", _readonly(target_weight, np.float32))
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "old_prefix_sha256", expected_prefix)
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise UncertaintyProtoFusionError("D24 persistent state exceeds 256KiB")

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def k_shot(self) -> int:
        return int(self.support_count_by_class[0])

    @property
    def target_fp32_state_bytes(self) -> int:
        return int(
            self.prototypes.nbytes
            + self.radius.nbytes
            + self.old_target_prototypes.nbytes
            + self.old_target_radius.nbytes
        )

    @property
    def fusion_metadata_state_bytes(self) -> int:
        return int(
            self.old_ground_radius.nbytes
            + self.old_target_weight.nbytes
            + self.support_count_by_class.nbytes
        )

    @property
    def persistent_state_bytes(self) -> int:
        metadata = len(self.schema.encode("utf-8")) + sum(
            len(value.encode("utf-8")) for value in self.classes
        ) + 64
        return int(
            self.int8_component_state_bytes
            + self.target_fp32_state_bytes
            + self.fusion_metadata_state_bytes
            + metadata
        )

    def geometry_audit(self) -> dict[str, Any]:
        pairs: list[dict[str, Any]] = []
        collisions: list[dict[str, Any]] = []
        for left in range(self.class_count):
            for right in range(left + 1, self.class_count):
                distance = float(
                    np.clip(
                        1.0
                        - np.dot(
                            self.prototypes[left].astype(np.float64),
                            self.prototypes[right].astype(np.float64),
                        ),
                        0.0,
                        2.0,
                    )
                )
                radius_sum = float(self.radius[left]) + float(self.radius[right])
                required = radius_sum + float(self.config.separation_margin)
                record = {
                    "left_index": left,
                    "right_index": right,
                    "left_class": self.classes[left],
                    "right_class": self.classes[right],
                    "left_role": "old" if left < self.old_class_count else "new",
                    "right_role": "old" if right < self.old_class_count else "new",
                    "center_cosine_distance": distance,
                    "radius_sum": radius_sum,
                    "separation_margin": float(self.config.separation_margin),
                    "required_distance": required,
                    "gap": distance - required,
                    "pass": distance > required,
                }
                pairs.append(record)
                if not record["pass"]:
                    collisions.append(record)
        return {
            "schema": "cvs.phase2.d24.geometry_audit.v1",
            "support_derived_only": True,
            "query_rows_used": 0,
            "strict_rule": "cosine_distance_gt_radius_sum_plus_margin",
            "pair_count": len(pairs),
            "collision_count": len(collisions),
            "pass": not collisions,
            "pairs": pairs,
            "collision_pairs": collisions,
        }

    def resource_audit(self) -> dict[str, Any]:
        # Dequantization scratch is explicitly counted as max active-domain
        # anchors for one old class plus three 160-D FP32 work vectors.
        max_ground_anchor_scratch = self.old_class_count * FEATURE_DIM * 4
        return {
            "schema": "cvs.phase2.d24.resource_audit.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "int8_ground_component_state_bytes": int(self.int8_component_state_bytes),
            "target_fp32_state_bytes": self.target_fp32_state_bytes,
            "fusion_metadata_state_bytes": self.fusion_metadata_state_bytes,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES,
            "estimated_macs_per_query": int(self.class_count * FEATURE_DIM),
            "estimated_fit_scratch_bytes_upper_bound": int(max_ground_anchor_scratch + 3 * FEATURE_DIM * 4),
            "persistent_full_precision_ground_anchor_count": 0,
            "ground_anchor_dequantization_policy": "transient_fit_only_not_persisted",
            "int8_component_update_access": False,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "dense_query_graph_bytes": 0,
            "per_sample_all_registered_classes": True,
        }


def _support_matrix(
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    *,
    expected_classes: Sequence[str] | None = None,
    expected_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    features = _normalize_rows(np.asarray(support_z_id))
    labels = np.asarray(tuple(str(value) for value in support_labels))
    if labels.ndim != 1 or len(labels) != len(features) or any(not value for value in labels):
        raise UncertaintyProtoFusionError("support labels and features must align")
    classes, counts = np.unique(labels, return_counts=True)
    discovered = tuple(sorted(classes.tolist()))
    if not discovered or len(set(int(value) for value in counts.tolist())) != 1:
        raise UncertaintyProtoFusionError(
            "support must use one class-symmetric physical K-shot"
        )
    k_shot = int(counts[0])
    if expected_classes is not None and set(discovered) != set(expected_classes):
        raise UncertaintyProtoFusionError("support class registry drift")
    if expected_k is not None and k_shot != int(expected_k):
        raise UncertaintyProtoFusionError("support K-shot drift")
    return features, labels, discovered, k_shot


def _target_prototypes_radius(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    r0: float,
) -> tuple[np.ndarray, np.ndarray]:
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    for label in classes:
        rows = features[labels == label]
        center = _normalize_vector(np.mean(rows, axis=0, dtype=np.float64))
        prototypes.append(center)
        radii.append(_q90_radius(rows, center, k1_r0=r0))
    return (
        np.ascontiguousarray(np.stack(prototypes), dtype=np.float32),
        np.ascontiguousarray(radii, dtype=np.float32),
    )


def fit_old(
    component: Int8DomainClassComponent,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    *,
    config: UncertaintyFusionConfig = UncertaintyFusionConfig(),
) -> UncertaintyFusionState:
    """Fit target-old FP32 prototypes and inverse-uncertainty fusion."""

    if not isinstance(component, Int8DomainClassComponent):
        raise UncertaintyProtoFusionError("immutable int8 domain-class component required")
    config.validate()
    component_hash_before = _component_sha256(component)
    features, labels, _, k_shot = _support_matrix(
        support_z_id,
        support_labels,
        expected_classes=component.class_registry,
    )
    classes = component.class_registry
    target_prototypes, target_radius = _target_prototypes_radius(
        features, labels, classes, r0=config.r0
    )
    fused: list[np.ndarray] = []
    fused_radius: list[float] = []
    ground_radius: list[float] = []
    target_weight: list[float] = []
    for class_index in range(len(classes)):
        anchors = component.dequantized_class_anchors(class_index)
        ground_center = _normalize_vector(np.mean(anchors, axis=0, dtype=np.float64))
        radius_ground = _q90_radius(anchors, ground_center, k1_r0=None)
        radius_target = float(target_radius[class_index])
        precision_ground = 1.0 / max(radius_ground, config.r_min) ** 2
        precision_target = k_shot / max(radius_target, config.r_min) ** 2
        weight_target = precision_target / (precision_ground + precision_target)
        center = _normalize_vector(
            (1.0 - weight_target) * ground_center
            + weight_target * target_prototypes[class_index]
        )
        center_shift = float(
            _cosine_distance_rows(ground_center[None, :], target_prototypes[class_index])[0]
        )
        radius_fused = math.sqrt(1.0 / (precision_ground + precision_target))
        radius_fused += weight_target * (1.0 - weight_target) * center_shift
        fused.append(center)
        fused_radius.append(radius_fused)
        ground_radius.append(radius_ground)
        target_weight.append(weight_target)
    if _component_sha256(component) != component_hash_before:
        raise UncertaintyProtoFusionError("int8 component changed during old fit")
    prototypes = np.ascontiguousarray(np.stack(fused), dtype=np.float32)
    radius = np.ascontiguousarray(fused_radius, dtype=np.float32)
    count = np.full(len(classes), k_shot, dtype=np.uint16)
    ground_radius_array = np.ascontiguousarray(ground_radius, dtype=np.float32)
    target_weight_array = np.ascontiguousarray(target_weight, dtype=np.float32)
    prefix = _old_prefix_sha256(
        classes=classes,
        prototypes=prototypes,
        radius=radius,
        count=count,
        old_target_prototypes=target_prototypes,
        old_ground_radius=ground_radius_array,
        old_target_radius=target_radius,
        old_target_weight=target_weight_array,
        int8_component_sha256=component_hash_before,
    )
    return UncertaintyFusionState(
        schema=SCHEMA,
        classes=classes,
        prototypes=prototypes,
        radius=radius,
        support_count_by_class=count,
        old_class_count=len(classes),
        old_target_prototypes=target_prototypes,
        old_ground_radius=ground_radius_array,
        old_target_radius=target_radius,
        old_target_weight=target_weight_array,
        int8_component_sha256=component_hash_before,
        int8_component_state_bytes=component.state_bytes,
        old_prefix_sha256=prefix,
        config=config,
    )


def append_new_classes(
    state: UncertaintyFusionState,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    *,
    registered_classes: Sequence[str] | None = None,
) -> UncertaintyFusionState:
    """Append pure-target FP32 new classes while byte-freezing old state."""

    if not isinstance(state, UncertaintyFusionState):
        raise UncertaintyProtoFusionError("valid D24 state required")
    features, labels, discovered, _ = _support_matrix(
        support_z_id,
        support_labels,
        expected_classes=registered_classes,
        expected_k=state.k_shot,
    )
    new_classes = discovered if registered_classes is None else tuple(str(value) for value in registered_classes)
    if len(new_classes) != len(set(new_classes)) or set(new_classes) != set(discovered):
        raise UncertaintyProtoFusionError("registered new-class order drift")
    if set(new_classes) & set(state.classes):
        raise UncertaintyProtoFusionError("new-class labels overlap registered classes")
    new_prototypes, new_radius = _target_prototypes_radius(
        features, labels, new_classes, r0=state.config.r0
    )
    result = UncertaintyFusionState(
        schema=SCHEMA,
        classes=state.classes + new_classes,
        prototypes=np.concatenate([state.prototypes, new_prototypes], axis=0).astype(np.float32),
        radius=np.concatenate([state.radius, new_radius]).astype(np.float32),
        support_count_by_class=np.concatenate(
            [state.support_count_by_class, np.full(len(new_classes), state.k_shot, dtype=np.uint16)]
        ),
        old_class_count=state.old_class_count,
        old_target_prototypes=state.old_target_prototypes,
        old_ground_radius=state.old_ground_radius,
        old_target_radius=state.old_target_radius,
        old_target_weight=state.old_target_weight,
        int8_component_sha256=state.int8_component_sha256,
        int8_component_state_bytes=state.int8_component_state_bytes,
        old_prefix_sha256=state.old_prefix_sha256,
        config=state.config,
    )
    if result.old_prefix_sha256 != state.old_prefix_sha256:
        raise UncertaintyProtoFusionError("old prefix changed during new registration")
    return result


def score_one(state: UncertaintyFusionState, z_id: np.ndarray) -> np.ndarray:
    """Score exactly one sample independently over all registered classes."""

    if not isinstance(state, UncertaintyFusionState):
        raise UncertaintyProtoFusionError("valid D24 state required")
    value = np.asarray(z_id, dtype=np.float32)
    if value.shape == (1, FEATURE_DIM):
        value = value[0]
    if value.shape != (FEATURE_DIM,):
        raise UncertaintyProtoFusionError(
            f"score_one requires exactly one {FEATURE_DIM}-D identity feature"
        )
    scores = _normalize_vector(value) @ state.prototypes.T
    return _readonly(scores, np.float32)


def predict_one(state: UncertaintyFusionState, z_id: np.ndarray) -> tuple[str, np.ndarray]:
    """Return the highest-scoring handle and immutable complete score vector."""

    scores = score_one(state, z_id)
    return state.classes[int(np.argmax(scores))], scores


__all__ = [
    "FEATURE_DIM",
    "MAX_PERSISTENT_STATE_BYTES",
    "SCHEMA",
    "UncertaintyFusionConfig",
    "UncertaintyFusionState",
    "UncertaintyProtoFusionError",
    "append_new_classes",
    "fit_old",
    "predict_one",
    "score_one",
]
