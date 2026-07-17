"""D25 block-balanced 288-D registration from one received LEO_weak IQ.

The three feature blocks are deterministic mathematical descriptions of the
same already-overlaid IQ row.  They are concatenated into one feature row and
never treated as additional support samples or additional channel views.

Phase1 int8 ground knowledge is defined only in the 160-D identity block.
Target-old FFT96/RF32 prototypes and every target-new block are support-only.
The module intentionally exposes no query fitting, truth-role, class-quota,
global-assignment, clean-data, source-sample, or scorer interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Sequence

import numpy as np

from .stage2_ciaf import Int8DomainClassComponent
from .stage2_uncertainty_proto_fusion import (
    UncertaintyFusionConfig,
    fit_old as fit_old_identity_fusion,
)


Z_DIM = 160
FFT_DIM = 96
RF_DIM = 32
FEATURE_DIM = Z_DIM + FFT_DIM + RF_DIM
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
SCHEMA = "cvs.phase2.d25_multimodal_concat_fusion.v1"
SCORE_COSINE = "weighted_block_cosine"
SCORE_RADIUS = "weighted_block_radius_likelihood"


class MultimodalConcatFusionError(ValueError):
    """Raised when D25 support, configuration, or immutable state drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    immutable.setflags(write=False)
    return immutable


def _normalize_rows(value: np.ndarray, dimension: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != int(dimension):
        raise MultimodalConcatFusionError(
            f"{name} must have shape [N,{int(dimension)}]"
        )
    if not np.isfinite(rows).all():
        raise MultimodalConcatFusionError(f"{name} contains non-finite values")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if bool(np.any(norms <= 1.0e-8)):
        raise MultimodalConcatFusionError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _normalize_vector(value: np.ndarray, dimension: int, name: str) -> np.ndarray:
    return _normalize_rows(
        np.asarray(value, dtype=np.float32).reshape(1, -1), dimension, name
    )[0]


def _cosine_distance(rows: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.clip(
        1.0 - rows.astype(np.float64) @ center.astype(np.float64), 0.0, 2.0
    )


def _q90_radius(rows: np.ndarray, center: np.ndarray, r0: float) -> float:
    if len(rows) == 1:
        return float(r0)
    return float(np.quantile(_cosine_distance(rows, center), 0.90))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class MultimodalConcatConfig:
    """Method-locked D25 block geometry and support-only scoring constants."""

    block_energy: tuple[float, float, float] = (5.0 / 9.0, 1.0 / 3.0, 1.0 / 9.0)
    r0_by_block: tuple[float, float, float] = (0.05, 0.05, 0.05)
    r_min: float = 1.0e-3
    separation_margin: float = 0.0
    score_mode: str = SCORE_COSINE
    use_ground_identity_fusion: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_energy", tuple(float(value) for value in self.block_energy)
        )
        object.__setattr__(
            self, "r0_by_block", tuple(float(value) for value in self.r0_by_block)
        )
        self.validate()

    def validate(self) -> None:
        energy = tuple(float(value) for value in self.block_energy)
        radii = tuple(float(value) for value in self.r0_by_block)
        if len(energy) != 3 or len(radii) != 3:
            raise MultimodalConcatFusionError("D25 requires three block constants")
        values = energy + radii + (float(self.r_min), float(self.separation_margin))
        if not all(math.isfinite(value) for value in values):
            raise MultimodalConcatFusionError("D25 configuration must be finite")
        if any(value <= 0.0 for value in energy) or not math.isclose(
            sum(energy), 1.0, rel_tol=0.0, abs_tol=1.0e-7
        ):
            raise MultimodalConcatFusionError(
                "D25 block energy must be positive and sum to one"
            )
        if energy[0] < 0.5:
            raise MultimodalConcatFusionError(
                "D25 identity block must retain at least half of squared energy"
            )
        if any(not 0.0 <= value <= 2.0 for value in radii):
            raise MultimodalConcatFusionError("D25 K1 radii are out of range")
        if not 0.0 < float(self.r_min) <= 2.0:
            raise MultimodalConcatFusionError("D25 r_min is out of range")
        if not 0.0 <= float(self.separation_margin) < 2.0:
            raise MultimodalConcatFusionError("D25 separation margin is out of range")
        if self.score_mode not in (SCORE_COSINE, SCORE_RADIUS):
            raise MultimodalConcatFusionError("unsupported D25 score mode")
        if not isinstance(self.use_ground_identity_fusion, bool):
            raise MultimodalConcatFusionError(
                "D25 ground identity fusion flag must be boolean"
            )


def build_concat288(
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    block_energy: tuple[float, float, float] = (5.0 / 9.0, 1.0 / 3.0, 1.0 / 9.0),
) -> np.ndarray:
    """Build one 288-D row per physical IQ after independent block normalization."""

    config = MultimodalConcatConfig(block_energy=block_energy)
    config.validate()
    z_rows = _normalize_rows(z_id160, Z_DIM, "z_id160")
    fft_rows = _normalize_rows(fft96, FFT_DIM, "fft96")
    rf_rows = _normalize_rows(rf32, RF_DIM, "rf32")
    if not len(z_rows) == len(fft_rows) == len(rf_rows):
        raise MultimodalConcatFusionError(
            "D25 feature blocks must have one aligned row per physical IQ"
        )
    scales = np.sqrt(np.asarray(block_energy, dtype=np.float64)).astype(np.float32)
    result = np.concatenate(
        [z_rows * scales[0], fft_rows * scales[1], rf_rows * scales[2]], axis=1
    )
    if not np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1.0e-5):
        raise MultimodalConcatFusionError("D25 concatenated feature norm drift")
    return _readonly(result, np.float32)


def _block_support(
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    support_labels: Sequence[str],
    *,
    expected_classes: Sequence[str] | None = None,
    expected_k: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, tuple[str, ...], int]:
    z_rows = _normalize_rows(z_id160, Z_DIM, "support z_id160")
    fft_rows = _normalize_rows(fft96, FFT_DIM, "support fft96")
    rf_rows = _normalize_rows(rf32, RF_DIM, "support rf32")
    labels = np.asarray(tuple(str(value) for value in support_labels))
    if not len(z_rows) == len(fft_rows) == len(rf_rows) == len(labels):
        raise MultimodalConcatFusionError("D25 support blocks and labels must align")
    if labels.ndim != 1 or any(not value for value in labels):
        raise MultimodalConcatFusionError("D25 support labels are invalid")
    unique, counts = np.unique(labels, return_counts=True)
    discovered = tuple(sorted(unique.tolist()))
    if not discovered or len(set(int(value) for value in counts.tolist())) != 1:
        raise MultimodalConcatFusionError(
            "D25 support must use one class-symmetric physical K-shot"
        )
    k_shot = int(counts[0])
    if expected_classes is not None and set(discovered) != set(expected_classes):
        raise MultimodalConcatFusionError("D25 support class registry drift")
    if expected_k is not None and k_shot != int(expected_k):
        raise MultimodalConcatFusionError("D25 support K-shot drift")
    return (z_rows, fft_rows, rf_rows), labels, discovered, k_shot


def _target_block_prototypes(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    dimension: int,
    r0: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    radii: list[float] = []
    for label in classes:
        selected = rows[labels == str(label)]
        if not len(selected):
            raise MultimodalConcatFusionError(f"D25 {name} class missing")
        center = _normalize_vector(
            np.mean(selected, axis=0, dtype=np.float64), dimension, name
        )
        centers.append(center)
        radii.append(_q90_radius(selected, center, r0))
    return (
        np.ascontiguousarray(np.stack(centers), dtype=np.float32),
        np.ascontiguousarray(radii, dtype=np.float32),
    )


def _old_prefix_sha256(
    *,
    classes: tuple[str, ...],
    old_count: int,
    arrays: Sequence[np.ndarray],
    config: MultimodalConcatConfig,
    component_sha256: str,
) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d25.old_prefix.v1\0")
    digest.update(_canonical_json_bytes(classes[:old_count]))
    digest.update(
        _canonical_json_bytes(
            {
                "block_energy": list(config.block_energy),
                "r0_by_block": list(config.r0_by_block),
                "r_min": config.r_min,
                "separation_margin": config.separation_margin,
                "score_mode": config.score_mode,
                "use_ground_identity_fusion": config.use_ground_identity_fusion,
                "component_sha256": component_sha256,
            }
        )
    )
    for value in arrays:
        array = np.ascontiguousarray(value[:old_count])
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(struct.pack("<B", array.ndim))
        for dimension in array.shape:
            digest.update(struct.pack("<I", int(dimension)))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MultimodalConcatFusionState:
    """Immutable D25 block state; Stage2-C may append only a new suffix."""

    schema: str
    classes: tuple[str, ...]
    prototype_z: np.ndarray
    prototype_fft: np.ndarray
    prototype_rf: np.ndarray
    radius_z: np.ndarray
    radius_fft: np.ndarray
    radius_rf: np.ndarray
    support_count_by_class: np.ndarray
    old_class_count: int
    old_target_z: np.ndarray
    old_ground_radius_z: np.ndarray
    old_target_radius_z: np.ndarray
    old_target_weight_z: np.ndarray
    int8_component_sha256: str
    int8_component_state_bytes: int
    max_active_ground_domain_count: int
    old_prefix_sha256: str
    config: MultimodalConcatConfig

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        old_count = int(self.old_class_count)
        arrays = (
            np.asarray(self.prototype_z),
            np.asarray(self.prototype_fft),
            np.asarray(self.prototype_rf),
        )
        radii = (
            np.asarray(self.radius_z),
            np.asarray(self.radius_fft),
            np.asarray(self.radius_rf),
        )
        counts = np.asarray(self.support_count_by_class)
        old_target_z = np.asarray(self.old_target_z)
        ground_radius = np.asarray(self.old_ground_radius_z)
        target_radius = np.asarray(self.old_target_radius_z)
        target_weight = np.asarray(self.old_target_weight_z)
        if (
            self.schema != SCHEMA
            or old_count < 2
            or old_count > len(classes)
            or len(set(classes)) != len(classes)
            or any(not value for value in classes)
            or arrays[0].dtype != np.float32
            or arrays[0].shape != (len(classes), Z_DIM)
            or arrays[1].dtype != np.float32
            or arrays[1].shape != (len(classes), FFT_DIM)
            or arrays[2].dtype != np.float32
            or arrays[2].shape != (len(classes), RF_DIM)
            or any(value.dtype != np.float32 or value.shape != (len(classes),) for value in radii)
            or counts.dtype != np.uint16
            or counts.shape != (len(classes),)
            or old_target_z.dtype != np.float32
            or old_target_z.shape != (old_count, Z_DIM)
            or any(
                value.dtype != np.float32 or value.shape != (old_count,)
                for value in (ground_radius, target_radius, target_weight)
            )
            or not all(np.isfinite(value).all() for value in arrays + radii)
            or not all(
                np.isfinite(value).all()
                for value in (old_target_z, ground_radius, target_radius, target_weight)
            )
            or bool(np.any(counts < 1))
            or len(set(int(value) for value in counts.tolist())) != 1
            or any(bool(np.any(value < 0.0)) or bool(np.any(value > 2.0)) for value in radii)
            or bool(np.any(target_weight < 0.0))
            or bool(np.any(target_weight > 1.0))
            or any(
                not np.allclose(np.linalg.norm(value, axis=1), 1.0, atol=1.0e-5)
                for value in arrays
            )
            or not np.allclose(np.linalg.norm(old_target_z, axis=1), 1.0, atol=1.0e-5)
            or len(str(self.int8_component_sha256)) != 64
            or int(self.int8_component_state_bytes) < 0
            or int(self.max_active_ground_domain_count) < 0
        ):
            raise MultimodalConcatFusionError("D25 state drift")
        prefix_arrays = arrays + radii + (
            counts,
            old_target_z,
            ground_radius,
            target_radius,
            target_weight,
        )
        expected_prefix = _old_prefix_sha256(
            classes=classes,
            old_count=old_count,
            arrays=prefix_arrays,
            config=self.config,
            component_sha256=str(self.int8_component_sha256),
        )
        if str(self.old_prefix_sha256) != expected_prefix:
            raise MultimodalConcatFusionError("D25 old prefix SHA256 drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "old_class_count", old_count)
        for field, value, dtype in (
            ("prototype_z", arrays[0], np.float32),
            ("prototype_fft", arrays[1], np.float32),
            ("prototype_rf", arrays[2], np.float32),
            ("radius_z", radii[0], np.float32),
            ("radius_fft", radii[1], np.float32),
            ("radius_rf", radii[2], np.float32),
            ("support_count_by_class", counts, np.uint16),
            ("old_target_z", old_target_z, np.float32),
            ("old_ground_radius_z", ground_radius, np.float32),
            ("old_target_radius_z", target_radius, np.float32),
            ("old_target_weight_z", target_weight, np.float32),
        ):
            object.__setattr__(self, field, _readonly(value, dtype))
        object.__setattr__(self, "old_prefix_sha256", expected_prefix)
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise MultimodalConcatFusionError("D25 persistent state exceeds 256KiB")

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def k_shot(self) -> int:
        return int(self.support_count_by_class[0])

    @property
    def target_fp32_state_bytes(self) -> int:
        return int(
            self.prototype_z.nbytes
            + self.prototype_fft.nbytes
            + self.prototype_rf.nbytes
            + self.radius_z.nbytes
            + self.radius_fft.nbytes
            + self.radius_rf.nbytes
            + self.old_target_z.nbytes
            + self.old_target_radius_z.nbytes
        )

    @property
    def fusion_metadata_state_bytes(self) -> int:
        return int(
            self.old_ground_radius_z.nbytes
            + self.old_target_weight_z.nbytes
            + self.support_count_by_class.nbytes
            + 6 * 4
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

    def concatenated_prototypes(self) -> np.ndarray:
        return build_concat288(
            self.prototype_z,
            self.prototype_fft,
            self.prototype_rf,
            block_energy=self.config.block_energy,
        )

    def geometry_audit(self) -> dict[str, Any]:
        combined = self.concatenated_prototypes()
        energy = np.asarray(self.config.block_energy, dtype=np.float64)
        aggregate_radius = (
            energy[0] * self.radius_z.astype(np.float64)
            + energy[1] * self.radius_fft.astype(np.float64)
            + energy[2] * self.radius_rf.astype(np.float64)
        )
        pairs: list[dict[str, Any]] = []
        collisions: list[dict[str, Any]] = []
        for left in range(self.class_count):
            for right in range(left + 1, self.class_count):
                distance = float(
                    np.clip(
                        1.0
                        - np.dot(
                            combined[left].astype(np.float64),
                            combined[right].astype(np.float64),
                        ),
                        0.0,
                        2.0,
                    )
                )
                radius_sum = float(aggregate_radius[left] + aggregate_radius[right])
                required = radius_sum + float(self.config.separation_margin)
                record = {
                    "left_index": left,
                    "right_index": right,
                    "left_class": self.classes[left],
                    "right_class": self.classes[right],
                    "left_role": "old" if left < self.old_class_count else "new",
                    "right_role": "old" if right < self.old_class_count else "new",
                    "center_cosine_distance": distance,
                    "aggregate_radius_sum": radius_sum,
                    "separation_margin": float(self.config.separation_margin),
                    "gap": distance - required,
                    "pass": distance > required,
                }
                pairs.append(record)
                if not record["pass"]:
                    collisions.append(record)
        return {
            "schema": "cvs.phase2.d25.geometry_audit.v1",
            "support_derived_only": True,
            "query_rows_used": 0,
            "pair_count": len(pairs),
            "collision_count": len(collisions),
            "pass": not collisions,
            "pairs": pairs,
            "collision_pairs": collisions,
        }

    def resource_audit(self) -> dict[str, Any]:
        radius_ops = 6 * self.class_count if self.config.score_mode == SCORE_RADIUS else 0
        head_macs = FEATURE_DIM + self.class_count * FEATURE_DIM + radius_ops
        scratch = (
            self.max_active_ground_domain_count * Z_DIM * 4
            + (3 * Z_DIM + 2 * FFT_DIM + 2 * RF_DIM) * 4
        )
        return {
            "schema": "cvs.phase2.d25.resource_audit.v1",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "trainable_parameters": 0,
            "feature_dimension": FEATURE_DIM,
            "block_dimensions": [Z_DIM, FFT_DIM, RF_DIM],
            "block_energy": [float(value) for value in self.config.block_energy],
            "int8_ground_component_state_bytes": int(self.int8_component_state_bytes),
            "target_fp32_state_bytes": self.target_fp32_state_bytes,
            "fusion_metadata_state_bytes": self.fusion_metadata_state_bytes,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES,
            "concat_scale_multiplications_per_query": FEATURE_DIM,
            "registered_prototype_dot_macs_per_query": self.class_count * FEATURE_DIM,
            "radius_score_ops_per_query": radius_ops,
            "estimated_head_macs_per_query": head_macs,
            "fft96_calls_per_physical_sample": 1,
            "fft96_complexity_report_separately": "O(T log T)",
            "rf32_calls_per_physical_sample": 1,
            "rf32_complexity_report_separately": "O(T) plus quantile selection/sort",
            "fft96_rf32_included_in_head_macs": False,
            "backbone_forwards_per_physical_sample": 1,
            "estimated_fit_scratch_bytes_upper_bound": int(scratch),
            "max_active_ground_domain_count": int(self.max_active_ground_domain_count),
            "persistent_full_precision_ground_anchor_count": 0,
            "ground_anchor_dequantization_policy": "transient_z_block_fit_only_not_persisted",
            "support_view_count": 1,
            "support_row_multiplicity": 1,
            "derived_support_rows": 0,
            "additional_physical_sample_count": 0,
            "additional_leo_overlay_count": 0,
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "dense_query_graph_bytes": 0,
            "per_sample_all_registered_classes": True,
        }


def fit_old_concat(
    component: Int8DomainClassComponent | None,
    support_z_id160: np.ndarray,
    support_fft96: np.ndarray,
    support_rf32: np.ndarray,
    support_labels: Sequence[str],
    *,
    config: MultimodalConcatConfig = MultimodalConcatConfig(),
) -> MultimodalConcatFusionState:
    """Fit old classes from aligned LEO_weak support; ground affects z only."""

    config.validate()
    blocks, labels, discovered, k_shot = _block_support(
        support_z_id160, support_fft96, support_rf32, support_labels
    )
    z_rows, fft_rows, rf_rows = blocks
    if config.use_ground_identity_fusion:
        if not isinstance(component, Int8DomainClassComponent):
            raise MultimodalConcatFusionError(
                "D25 ground identity fusion requires the immutable int8 component"
            )
        if set(discovered) != set(component.class_registry):
            raise MultimodalConcatFusionError("D25 old class registry drift")
        classes = component.class_registry
    else:
        classes = discovered
    target_z, target_radius_z = _target_block_prototypes(
        z_rows,
        labels,
        classes,
        dimension=Z_DIM,
        r0=config.r0_by_block[0],
        name="z_id160",
    )
    target_fft, radius_fft = _target_block_prototypes(
        fft_rows,
        labels,
        classes,
        dimension=FFT_DIM,
        r0=config.r0_by_block[1],
        name="fft96",
    )
    target_rf, radius_rf = _target_block_prototypes(
        rf_rows,
        labels,
        classes,
        dimension=RF_DIM,
        r0=config.r0_by_block[2],
        name="rf32",
    )
    if config.use_ground_identity_fusion:
        identity_state = fit_old_identity_fusion(
            component,
            z_rows,
            labels,
            config=UncertaintyFusionConfig(
                r0=config.r0_by_block[0],
                r_min=config.r_min,
                separation_margin=config.separation_margin,
            ),
        )
        prototype_z = identity_state.prototypes
        radius_z = identity_state.radius
        ground_radius_z = identity_state.old_ground_radius
        target_weight_z = identity_state.old_target_weight
        component_sha256 = identity_state.int8_component_sha256
        component_bytes = identity_state.int8_component_state_bytes
        max_domains = max(
            int(np.sum(component.domain_class_mask[:, index].astype(bool)))
            for index in range(len(classes))
        )
    else:
        prototype_z = target_z
        radius_z = target_radius_z
        ground_radius_z = np.zeros(len(classes), dtype=np.float32)
        target_weight_z = np.ones(len(classes), dtype=np.float32)
        component_sha256 = "0" * 64
        component_bytes = 0
        max_domains = 0
    counts = np.full(len(classes), k_shot, dtype=np.uint16)
    prefix_arrays = (
        prototype_z,
        target_fft,
        target_rf,
        radius_z,
        radius_fft,
        radius_rf,
        counts,
        target_z,
        ground_radius_z,
        target_radius_z,
        target_weight_z,
    )
    prefix = _old_prefix_sha256(
        classes=classes,
        old_count=len(classes),
        arrays=prefix_arrays,
        config=config,
        component_sha256=component_sha256,
    )
    return MultimodalConcatFusionState(
        schema=SCHEMA,
        classes=classes,
        prototype_z=np.asarray(prototype_z, dtype=np.float32),
        prototype_fft=target_fft,
        prototype_rf=target_rf,
        radius_z=np.asarray(radius_z, dtype=np.float32),
        radius_fft=radius_fft,
        radius_rf=radius_rf,
        support_count_by_class=counts,
        old_class_count=len(classes),
        old_target_z=target_z,
        old_ground_radius_z=np.asarray(ground_radius_z, dtype=np.float32),
        old_target_radius_z=target_radius_z,
        old_target_weight_z=np.asarray(target_weight_z, dtype=np.float32),
        int8_component_sha256=component_sha256,
        int8_component_state_bytes=component_bytes,
        max_active_ground_domain_count=max_domains,
        old_prefix_sha256=prefix,
        config=config,
    )


def append_new_classes_concat(
    state: MultimodalConcatFusionState,
    support_z_id160: np.ndarray,
    support_fft96: np.ndarray,
    support_rf32: np.ndarray,
    support_labels: Sequence[str],
    *,
    registered_classes: Sequence[str] | None = None,
) -> MultimodalConcatFusionState:
    """Append pure-target new classes while byte-freezing the old prefix."""

    if not isinstance(state, MultimodalConcatFusionState):
        raise MultimodalConcatFusionError("valid D25 state required")
    blocks, labels, discovered, _ = _block_support(
        support_z_id160,
        support_fft96,
        support_rf32,
        support_labels,
        expected_classes=registered_classes,
        expected_k=state.k_shot,
    )
    new_classes = (
        discovered
        if registered_classes is None
        else tuple(str(value) for value in registered_classes)
    )
    if len(new_classes) != len(set(new_classes)) or set(new_classes) != set(discovered):
        raise MultimodalConcatFusionError("D25 registered new-class order drift")
    if set(new_classes) & set(state.classes):
        raise MultimodalConcatFusionError("D25 new classes overlap registered classes")
    z_rows, fft_rows, rf_rows = blocks
    new_z, new_radius_z = _target_block_prototypes(
        z_rows,
        labels,
        new_classes,
        dimension=Z_DIM,
        r0=state.config.r0_by_block[0],
        name="new z_id160",
    )
    new_fft, new_radius_fft = _target_block_prototypes(
        fft_rows,
        labels,
        new_classes,
        dimension=FFT_DIM,
        r0=state.config.r0_by_block[1],
        name="new fft96",
    )
    new_rf, new_radius_rf = _target_block_prototypes(
        rf_rows,
        labels,
        new_classes,
        dimension=RF_DIM,
        r0=state.config.r0_by_block[2],
        name="new rf32",
    )
    result = MultimodalConcatFusionState(
        schema=SCHEMA,
        classes=state.classes + new_classes,
        prototype_z=np.concatenate([state.prototype_z, new_z], axis=0).astype(np.float32),
        prototype_fft=np.concatenate([state.prototype_fft, new_fft], axis=0).astype(np.float32),
        prototype_rf=np.concatenate([state.prototype_rf, new_rf], axis=0).astype(np.float32),
        radius_z=np.concatenate([state.radius_z, new_radius_z]).astype(np.float32),
        radius_fft=np.concatenate([state.radius_fft, new_radius_fft]).astype(np.float32),
        radius_rf=np.concatenate([state.radius_rf, new_radius_rf]).astype(np.float32),
        support_count_by_class=np.concatenate(
            [
                state.support_count_by_class,
                np.full(len(new_classes), state.k_shot, dtype=np.uint16),
            ]
        ),
        old_class_count=state.old_class_count,
        old_target_z=state.old_target_z,
        old_ground_radius_z=state.old_ground_radius_z,
        old_target_radius_z=state.old_target_radius_z,
        old_target_weight_z=state.old_target_weight_z,
        int8_component_sha256=state.int8_component_sha256,
        int8_component_state_bytes=state.int8_component_state_bytes,
        max_active_ground_domain_count=state.max_active_ground_domain_count,
        old_prefix_sha256=state.old_prefix_sha256,
        config=state.config,
    )
    if result.old_prefix_sha256 != state.old_prefix_sha256:
        raise MultimodalConcatFusionError("D25 old prefix changed during append")
    return result


def score_one(
    state: MultimodalConcatFusionState, feature_288: np.ndarray
) -> np.ndarray:
    """Score exactly one sample independently over all registered classes."""

    if not isinstance(state, MultimodalConcatFusionState):
        raise MultimodalConcatFusionError("valid D25 state required")
    value = np.asarray(feature_288, dtype=np.float32)
    if value.shape == (1, FEATURE_DIM):
        value = value[0]
    if value.shape != (FEATURE_DIM,) or not np.isfinite(value).all():
        raise MultimodalConcatFusionError(
            "D25 score_one requires exactly one finite 288-D feature"
        )
    split_z = value[:Z_DIM]
    split_fft = value[Z_DIM : Z_DIM + FFT_DIM]
    split_rf = value[Z_DIM + FFT_DIM :]
    expected = np.sqrt(np.asarray(state.config.block_energy, dtype=np.float64))
    actual = np.asarray(
        [np.linalg.norm(split_z), np.linalg.norm(split_fft), np.linalg.norm(split_rf)],
        dtype=np.float64,
    )
    if not np.allclose(actual, expected, atol=2.0e-5):
        raise MultimodalConcatFusionError(
            "D25 score feature must be produced by the locked block-balanced concat"
        )
    blocks = (
        split_z / np.float32(expected[0]),
        split_fft / np.float32(expected[1]),
        split_rf / np.float32(expected[2]),
    )
    similarities = (
        blocks[0] @ state.prototype_z.T,
        blocks[1] @ state.prototype_fft.T,
        blocks[2] @ state.prototype_rf.T,
    )
    energy = np.asarray(state.config.block_energy, dtype=np.float32)
    if state.config.score_mode == SCORE_COSINE:
        scores = (
            energy[0] * similarities[0]
            + energy[1] * similarities[1]
            + energy[2] * similarities[2]
        )
    else:
        block_scores = []
        for similarity, radius in zip(
            similarities, (state.radius_z, state.radius_fft, state.radius_rf)
        ):
            scale = np.maximum(radius, np.float32(state.config.r_min))
            block_scores.append(-(1.0 - similarity) / scale - np.log(scale))
        scores = (
            energy[0] * block_scores[0]
            + energy[1] * block_scores[1]
            + energy[2] * block_scores[2]
        )
    return _readonly(scores, np.float32)


def predict_one(
    state: MultimodalConcatFusionState, feature_288: np.ndarray
) -> tuple[str, np.ndarray]:
    scores = score_one(state, feature_288)
    return state.classes[int(np.argmax(scores))], scores


__all__ = [
    "FEATURE_DIM",
    "FFT_DIM",
    "MAX_PERSISTENT_STATE_BYTES",
    "MultimodalConcatConfig",
    "MultimodalConcatFusionError",
    "MultimodalConcatFusionState",
    "RF_DIM",
    "SCHEMA",
    "SCORE_COSINE",
    "SCORE_RADIUS",
    "Z_DIM",
    "append_new_classes_concat",
    "build_concat288",
    "fit_old_concat",
    "predict_one",
    "score_one",
]
