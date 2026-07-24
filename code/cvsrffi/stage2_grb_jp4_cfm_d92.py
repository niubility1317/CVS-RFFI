"""Strict Stage2 kernel for GRB-JP4-CFM-qKNN-D92/r2-sharedK1.

This module implements the support-only F03--F06 surface from the frozen
design.  It deliberately does not expose a target-query fitting hook, a
performance gate, or a fallback state.  A fit either closes all typed,
physical-LOO, numerical, quantization, and replay receipts or raises.

The real model path reuses only the byte-bound ``joint_proj.0`` forward and
analytic-Jacobian primitives from the older spike.  Candidate identity,
typed ground/lock inputs, solver, K1 LOCO, wire format, receipts, and query
lifecycle are new and cannot deserialize or masquerade as the old method.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import (
    StrictForward,
    analytic_jacobian as _legacy_analytic_jacobian,
    strict_zid_with_hook as _legacy_strict_zid_with_hook,
)


CANDIDATE = "GRB-JP4-CFM-qKNN-D92/r2-sharedK1"
SCHEMA = "cvs.stage2.grb_jp4_cfm_qknn_d92.r2_shared_k1"
GROUND_SCHEMA = "cvs.stage2.grb_jp4_cfm.ground_input.v2"
LOCK_SCHEMA = "cvs.stage2.grb_jp4_cfm.method_lock.v2"
FIT_STATE_WIRE_SCHEMA = "cvs.stage2.grb_jp4_cfm.fit_state_wire.v2"
QUERY_RECEIPT_SCHEMA = "cvs.stage2.grb_jp4_cfm.query_receipt.v2"
RANK = 4
Z_DIM = 160
HIDDEN_DIM = 320
OLD_CLASS_COUNT = 6
MAX_GROUND_PROTOTYPES = 3
K_VALUES = (1, 5, 10)
ACTIVE_SET_STEPS = 2
THETA_NUMERIC_WIRE_BYTES = 6
JP4_WIRE_LIMIT_BYTES = 4096
MAX_STATE_BYTES = 256 * 1024
MAX_SUPPORT_MAC = 65_000_000
MAX_POST_BACKBONE_MAC_PER_QUERY = 262_144
HELD_FALSIFIER_SCOPE = "PHASE1_HELD_FALSIFIER_ONLY_NOT_FORMAL"
COND_LIMIT = float(2**24)
SOLVE_RELATIVE_RESIDUAL_LIMIT = 1.0e-10
RIDGE_FRACTION = 0.01
RIDGE_FLOOR = float(2**-20)
TRUST_DENOMINATOR = math.sqrt(float(Z_DIM))
TRUST_NORM_FLOOR = float(2**-24)
RANK_RELATIVE_THRESHOLD = 1.0e-6
PHASE1_RESOURCE_FIELDS = {
    "ground_wire_bytes",
    "jp4_update_factor_numeric_bytes",
    "jp4_update_factor_receipt_bytes",
    "jp4_update_factor_wire_bytes",
    "phase1_margin_wire_bytes",
    "component_metadata_wire_bytes",
    "total_component_bytes",
    "jp4_update_factor_wire_limit_bytes",
    "arm_state_limit_bytes",
    "persistent_dense_float_bank_bytes",
    "ground_direction_rank",
}


class GRBJP4CFMError(ValueError):
    """A frozen r2 support-only invariant was violated."""


def _json_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_plain(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical_json(value))


def _require_sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GRBJP4CFMError(f"{name} must be a lower-case SHA256")
    return value


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype != expected or array.shape != shape:
        raise GRBJP4CFMError(
            f"{name} must have dtype={expected.str} and shape={shape}"
        )
    result = np.array(array, dtype=expected, copy=True, order="C")
    if not np.isfinite(result).all() if np.issubdtype(expected, np.floating) else False:
        raise GRBJP4CFMError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _finite_rows(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[1] != width
        or len(array) == 0
        or not np.isfinite(array).all()
    ):
        raise GRBJP4CFMError(f"{name} must be finite float32 [N,{width}]")
    return np.ascontiguousarray(array)


def _finite_jacobian(value: Any, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.shape != (rows, RANK, Z_DIM)
        or not np.isfinite(array).all()
    ):
        raise GRBJP4CFMError(
            f"{name} must be finite float32 [{rows},{RANK},{Z_DIM}]"
        )
    return np.ascontiguousarray(array)


def _typed_tokens(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise GRBJP4CFMError(f"{name} must be rank one")
        raw = values.tolist()
    else:
        raw = list(values)
    result: list[str] = []
    for value in raw:
        if type(value) is not str or not value:
            raise GRBJP4CFMError(f"{name} must contain nonempty typed strings")
        result.append(value)
    tokens = tuple(result)
    if unique and len(set(tokens)) != len(tokens):
        raise GRBJP4CFMError(f"{name} must be unique")
    return tokens


def _unit_rows(rows: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if (
        value.ndim != 2
        or not np.isfinite(value).all()
        or np.any(norms <= 1.0e-12)
    ):
        raise GRBJP4CFMError(f"{name} must contain finite nonzero rows")
    return np.ascontiguousarray((value / norms).astype(np.float32))


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha_bytes(array.tobytes())


def _weight_array(linear_or_weight: Any) -> np.ndarray:
    value = (
        linear_or_weight.detach().cpu().contiguous().numpy()
        if torch.is_tensor(linear_or_weight)
        else np.asarray(linear_or_weight)
    )
    if (
        value.dtype != np.float32
        or value.shape != (Z_DIM, HIDDEN_DIM)
        or not np.isfinite(value).all()
    ):
        raise GRBJP4CFMError(
            "joint_proj.0 weight must be finite float32 [160,320]"
        )
    return np.ascontiguousarray(value)


def _joint_proj_linear(model: Any) -> torch.nn.Module:
    try:
        linear = model.id_backbone.cls_head.joint_proj[0]
    except (AttributeError, IndexError, TypeError):
        try:
            linear = dict(model.named_modules())[
                "id_backbone.cls_head.joint_proj.0"
            ]
        except (AttributeError, KeyError) as exc:
            raise GRBJP4CFMError("real joint_proj.0 path is absent") from exc
    if (
        not isinstance(linear, torch.nn.Linear)
        or tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM)
        or linear.weight.dtype != torch.float32
    ):
        raise GRBJP4CFMError("real joint_proj.0 contract drift")
    return linear


def _decode_rows(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    rows: int,
    width: int,
    name: str,
) -> np.ndarray:
    if (
        codes.dtype != np.int8
        or codes.shape != (rows, width)
        or scales.dtype != np.dtype("<f2")
        or scales.shape != (rows,)
        or np.any(codes == -128)
        or not np.isfinite(scales).all()
        or np.any(scales <= np.float16(0.0))
    ):
        raise GRBJP4CFMError(f"{name} INT8/FP16 layout drift")
    result = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    if not np.isfinite(result).all():
        raise GRBJP4CFMError(f"{name} decoded nonfinite")
    return np.ascontiguousarray(result)


def _fp16_floor(value: float) -> np.ndarray:
    if not np.isfinite(value) or value < 0.0:
        raise GRBJP4CFMError("FP16 floor input must be finite nonnegative")
    candidate = np.float16(value)
    if float(candidate) > value:
        candidate = np.nextafter(candidate, np.float16(0.0), dtype=np.float16)
    if float(candidate) < 0.0 or float(candidate) > value:
        raise GRBJP4CFMError("FP16 downward rounding failed")
    return np.asarray(candidate, dtype="<f2").reshape(())


@dataclass(frozen=True)
class CFMMethodLock:
    """Typed, target-independent Stage2 lock.

    The qKNN constants are immutable Phase1 values.  ``delta_q`` and
    ``tau_q`` are repeated in the ground component and must match exactly.
    """

    qknn_neighbor_count: int
    student_nu: float
    kernel_effective_dim: float
    kernel_volume_gamma: float
    kernel_scale: float
    qknn_lock_digest: str
    phase1_method_lock_sha256: str
    delta_q: float
    tau_q: float
    scale_prior_strength: float = 1.0
    scale_min_ratio: float = 0.25
    scale_max_ratio: float = 4.0
    temperature: float = 1.0
    candidate: str = CANDIDATE
    ground_old_multiprototype_enabled: bool = True
    query_fit_access: bool = False
    schema: str = LOCK_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != LOCK_SCHEMA
            or self.candidate != CANDIDATE
            or self.ground_old_multiprototype_enabled is not True
            or self.query_fit_access is not False
            or type(self.qknn_neighbor_count) is not int
            or self.qknn_neighbor_count < 1
            or not np.isfinite(self.student_nu)
            or self.student_nu <= 0.0
            or not np.isfinite(self.kernel_effective_dim)
            or self.kernel_effective_dim <= 0.0
            or not np.isfinite(self.kernel_volume_gamma)
            or self.kernel_volume_gamma <= 0.0
            or not np.isfinite(self.kernel_scale)
            or self.kernel_scale <= 0.0
            or not np.isfinite(self.scale_prior_strength)
            or self.scale_prior_strength <= 0.0
            or not np.isfinite(self.scale_min_ratio)
            or self.scale_min_ratio <= 0.0
            or self.scale_min_ratio > 1.0
            or not np.isfinite(self.scale_max_ratio)
            or self.scale_max_ratio < 1.0
            or not np.isfinite(self.temperature)
            or self.temperature <= 0.0
            or not np.isfinite(self.delta_q)
            or self.delta_q < 0.0
            or not np.isfinite(self.tau_q)
            or self.tau_q < float(2**-10)
        ):
            raise GRBJP4CFMError("CFM method-lock semantic drift")
        _require_sha(self.qknn_lock_digest, "qknn_lock_digest")
        _require_sha(
            self.phase1_method_lock_sha256, "phase1_method_lock_sha256"
        )

    @property
    def digest(self) -> str:
        return _sha_json(
            {
                "schema": self.schema,
                "candidate": self.candidate,
                "ground_old_multiprototype_enabled": True,
                "query_fit_access": False,
                "qknn_neighbor_count": self.qknn_neighbor_count,
                "student_nu": float(self.student_nu),
                "kernel_effective_dim": float(self.kernel_effective_dim),
                "kernel_volume_gamma": float(self.kernel_volume_gamma),
                "kernel_scale": float(self.kernel_scale),
                "scale_prior_strength": float(self.scale_prior_strength),
                "scale_min_ratio": float(self.scale_min_ratio),
                "scale_max_ratio": float(self.scale_max_ratio),
                "temperature": float(self.temperature),
                "qknn_lock_digest": self.qknn_lock_digest,
                "phase1_method_lock_sha256": self.phase1_method_lock_sha256,
                "delta_q": float(self.delta_q),
                "tau_q": float(self.tau_q),
                "active_set_steps": ACTIVE_SET_STEPS,
                "ridge_fraction": RIDGE_FRACTION,
                "theta_box": 1.0,
                "trust_denominator": TRUST_DENOMINATOR,
                "g_denominator": 4,
            }
        )

    @classmethod
    def from_mapping(
        cls,
        method_lock: Mapping[str, Any],
        *,
        qknn_lock: Any,
        phase1_method_lock_sha256: str | None = None,
    ) -> "CFMMethodLock":
        if not isinstance(method_lock, Mapping):
            raise GRBJP4CFMError("method lock must be an exact mapping")
        lock = dict(method_lock)
        phase1_sha = (
            _sha_json(lock)
            if phase1_method_lock_sha256 is None
            else phase1_method_lock_sha256
        )
        try:
            from cvsrffi.stage2_zid_student_t_qknn import (
                Phase1ZIDStudentTLock,
            )
        except ImportError as exc:
            raise GRBJP4CFMError("typed qKNN lock is unavailable") from exc
        if type(qknn_lock) is not Phase1ZIDStudentTLock:
            raise GRBJP4CFMError(
                "CFM lock requires exact Phase1ZIDStudentTLock"
            )
        qknn_raw = lock.get("qknn_lock_sha256_by_k")
        qknn_by_k = dict(qknn_raw) if isinstance(qknn_raw, Mapping) else None
        if (
            type(qknn_by_k) is not dict
            or set(qknn_by_k) != {"1", "5", "10"}
            or len(set(qknn_by_k.values())) != 3
            or any(
                _require_sha(value, f"qknn_lock_sha256_by_k.{key}")
                != value
                for key, value in qknn_by_k.items()
            )
            or qknn_lock.lock_digest
            != qknn_by_k[str(qknn_lock.active_k)]
            or int(lock["rank"]) != RANK
            or int(lock["old_class_count"]) != OLD_CLASS_COUNT
            or tuple(int(value) for value in lock["allowed_k"]) != K_VALUES
            or int(lock["active_set_steps"]) != ACTIVE_SET_STEPS
            or float(lock["ridge_fraction"]) != RIDGE_FRACTION
            or float(lock["theta_box_abs"]) != 1.0
            or int(lock["trust_divisor_squared"]) != Z_DIM
            or int(lock["g_denominator"]) != 4
            or lock["target25_release_authorized"] is not False
            or int(lock["query_rows_used_for_fit"]) != 0
            or lock["schema"]
            != "cvs.phase1.grb_jp4_cfm_qknn_d92_method_lock.v2"
            or lock["method_id"] != CANDIDATE
            or lock["candidate_id"] != CANDIDATE
            or lock["protocol_schema"] != "p2_min_v1"
            or lock["feature_schema"]
            != "ADV3B02:z_id:unit_l2:160:v1"
            or lock["ground_old_multiprototype_enabled"] is not True
            or int(lock["ground_old_multiprototype_max_per_class"]) != 3
            or int(lock["ground_old_multiprototype_min_physical_samples"])
            != 2
            or lock["ground_old_multiprototype_old_classes_only"] is not True
            or lock["ground_prototypes_enter_qknn_bank"] is not False
            or lock["ground_prototypes_generate_logits"] is not False
            or lock["ground_prototypes_add_k"] is not False
            or lock["ground_component_phase2_mutable"] is not False
            or lock["query_fit_access"] is not False
        ):
            raise GRBJP4CFMError("Phase1 CFM method-lock constants drift")
        return cls(
            qknn_neighbor_count=int(qknn_lock.active_k),
            student_nu=float(qknn_lock.student_nu),
            kernel_effective_dim=float(qknn_lock.kernel_effective_dim),
            kernel_volume_gamma=float(qknn_lock.kernel_volume_gamma),
            kernel_scale=float(qknn_lock.shared_h0),
            qknn_lock_digest=_require_sha(
                qknn_lock.lock_digest, "qknn_lock_digest"
            ),
            phase1_method_lock_sha256=_require_sha(
                phase1_sha, "phase1_method_lock_sha256"
            ),
            delta_q=float(lock["delta_q"]),
            tau_q=float(lock["tau_q"]),
            scale_prior_strength=float(qknn_lock.scale_prior_strength),
            scale_min_ratio=float(qknn_lock.scale_min_ratio),
            scale_max_ratio=float(qknn_lock.scale_max_ratio),
            temperature=float(qknn_lock.temperature),
            candidate=str(lock["candidate_id"]),
            ground_old_multiprototype_enabled=lock[
                "ground_old_multiprototype_enabled"
            ],
            query_fit_access=lock["query_fit_access"],
        )


@dataclass(frozen=True)
class GroundCFMInput:
    """Typed, read-only Phase1 ground input for the r2 Stage2 solver."""

    prototype_codes: np.ndarray
    prototype_scales: np.ndarray
    prototype_mask: np.ndarray
    prototype_weights: np.ndarray
    prototype_radii: np.ndarray
    left_codes: np.ndarray
    left_scales: np.ndarray
    right_codes: np.ndarray
    right_scales: np.ndarray
    direction_energy: np.ndarray
    delta_q_fp16: np.ndarray
    tau_q_fp16: np.ndarray
    old_class_order: tuple[str, ...]
    checkpoint_sha256: str
    joint_weight_sha256: str
    phase1_method_lock_sha256: str
    component_digest: str
    phase1_resource_receipt: Mapping[str, Any]
    phase1_resource_receipt_sha256: str
    schema: str = GROUND_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GROUND_SCHEMA:
            raise GRBJP4CFMError("ground-input schema drift")
        arrays = {
            "prototype_codes": _readonly_array(
                self.prototype_codes,
                dtype=np.int8,
                shape=(
                    OLD_CLASS_COUNT,
                    MAX_GROUND_PROTOTYPES,
                    Z_DIM,
                ),
                name="prototype_codes",
            ),
            "prototype_scales": _readonly_array(
                self.prototype_scales,
                dtype="<f2",
                shape=(OLD_CLASS_COUNT, MAX_GROUND_PROTOTYPES),
                name="prototype_scales",
            ),
            "prototype_mask": _readonly_array(
                self.prototype_mask,
                dtype=np.bool_,
                shape=(OLD_CLASS_COUNT, MAX_GROUND_PROTOTYPES),
                name="prototype_mask",
            ),
            "prototype_weights": _readonly_array(
                self.prototype_weights,
                dtype="<f2",
                shape=(OLD_CLASS_COUNT, MAX_GROUND_PROTOTYPES),
                name="prototype_weights",
            ),
            "prototype_radii": _readonly_array(
                self.prototype_radii,
                dtype="<f2",
                shape=(OLD_CLASS_COUNT, MAX_GROUND_PROTOTYPES),
                name="prototype_radii",
            ),
            "left_codes": _readonly_array(
                self.left_codes,
                dtype=np.int8,
                shape=(RANK, Z_DIM),
                name="left_codes",
            ),
            "left_scales": _readonly_array(
                self.left_scales,
                dtype="<f2",
                shape=(RANK,),
                name="left_scales",
            ),
            "right_codes": _readonly_array(
                self.right_codes,
                dtype=np.int8,
                shape=(RANK, HIDDEN_DIM),
                name="right_codes",
            ),
            "right_scales": _readonly_array(
                self.right_scales,
                dtype="<f2",
                shape=(RANK,),
                name="right_scales",
            ),
            "direction_energy": _readonly_array(
                self.direction_energy,
                dtype="<f2",
                shape=(RANK,),
                name="direction_energy",
            ),
            "delta_q_fp16": _readonly_array(
                self.delta_q_fp16,
                dtype="<f2",
                shape=(),
                name="delta_q_fp16",
            ),
            "tau_q_fp16": _readonly_array(
                self.tau_q_fp16,
                dtype="<f2",
                shape=(),
                name="tau_q_fp16",
            ),
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        mask_count = self.prototype_mask.sum(axis=1)
        if np.any(mask_count < 1) or np.any(mask_count > MAX_GROUND_PROTOTYPES):
            raise GRBJP4CFMError(
                "ground must cover every old class with one to three prototypes"
            )
        if np.any(self.prototype_codes == -128):
            raise GRBJP4CFMError("ground INT8 code -128 is forbidden")
        active_scales = self.prototype_scales[self.prototype_mask]
        inactive_scales = self.prototype_scales[~self.prototype_mask]
        active_radii = self.prototype_radii[self.prototype_mask]
        if (
            not np.isfinite(active_scales).all()
            or np.any(active_scales <= np.float16(0.0))
            or np.any(inactive_scales != np.float16(0.0))
            or np.any(self.prototype_codes[~self.prototype_mask] != 0)
            or not np.isfinite(active_radii).all()
            or np.any(active_radii < np.float16(0.0))
            or np.any(self.prototype_radii[~self.prototype_mask] != 0)
        ):
            raise GRBJP4CFMError("ground prototype mask/padding drift")
        for class_index, count in enumerate(mask_count.tolist()):
            expected = np.float16(1.0 / float(count))
            weights = self.prototype_weights[class_index]
            if (
                np.any(weights[self.prototype_mask[class_index]] != expected)
                or np.any(weights[~self.prototype_mask[class_index]] != 0)
            ):
                raise GRBJP4CFMError(
                    "ground prototype weights must be frozen equal 1/M"
                )
        for codes, scales, rows, width, name in (
            (
                self.left_codes,
                self.left_scales,
                RANK,
                Z_DIM,
                "L_g",
            ),
            (
                self.right_codes,
                self.right_scales,
                RANK,
                HIDDEN_DIM,
                "R",
            ),
        ):
            _decode_rows(
                codes, scales, rows=rows, width=width, name=name
            )
        energy = self.direction_energy.astype(np.float64)
        if (
            np.any(energy <= 0.0)
            or not np.isfinite(energy).all()
            or not np.isclose(np.linalg.norm(energy), 1.0, rtol=2.0e-3, atol=2.0e-3)
        ):
            raise GRBJP4CFMError("direction energy must be positive unit L2")
        if (
            float(self.delta_q_fp16) < 0.0
            or float(self.tau_q_fp16) < float(2**-10)
        ):
            raise GRBJP4CFMError("ground delta_q/tau_q drift")
        classes = _typed_tokens(
            self.old_class_order, name="old_class_order", unique=True
        )
        if len(classes) != OLD_CLASS_COUNT:
            raise GRBJP4CFMError("ground must bind exactly six old classes")
        object.__setattr__(self, "old_class_order", classes)
        for name in (
            "checkpoint_sha256",
            "joint_weight_sha256",
            "phase1_method_lock_sha256",
            "component_digest",
            "phase1_resource_receipt_sha256",
        ):
            _require_sha(getattr(self, name), name)
        resource = dict(self.phase1_resource_receipt)
        if (
            set(resource) != PHASE1_RESOURCE_FIELDS
            or any(
                type(resource[field]) is not int or resource[field] < 0
                for field in PHASE1_RESOURCE_FIELDS
            )
            or resource["ground_wire_bytes"] <= 0
            or resource["jp4_update_factor_numeric_bytes"]
            != (
                self.left_codes.nbytes
                + self.left_scales.nbytes
                + self.right_codes.nbytes
                + self.right_scales.nbytes
                + self.direction_energy.nbytes
                + THETA_NUMERIC_WIRE_BYTES
            )
            or resource["jp4_update_factor_wire_bytes"]
            != resource["jp4_update_factor_numeric_bytes"]
            + resource["jp4_update_factor_receipt_bytes"]
            or resource["phase1_margin_wire_bytes"]
            != self.delta_q_fp16.nbytes + self.tau_q_fp16.nbytes
            or resource["total_component_bytes"]
            != (
                resource["ground_wire_bytes"]
                + resource["jp4_update_factor_wire_bytes"]
                + resource["phase1_margin_wire_bytes"]
                + resource["component_metadata_wire_bytes"]
            )
            or resource["jp4_update_factor_wire_bytes"]
            > resource["jp4_update_factor_wire_limit_bytes"]
            or resource["jp4_update_factor_wire_limit_bytes"]
            != JP4_WIRE_LIMIT_BYTES
            or resource["total_component_bytes"]
            > resource["arm_state_limit_bytes"]
            or resource["arm_state_limit_bytes"] != MAX_STATE_BYTES
            or resource["persistent_dense_float_bank_bytes"] != 0
            or resource["ground_direction_rank"] != RANK
            or _sha_json(resource) != self.phase1_resource_receipt_sha256
        ):
            raise GRBJP4CFMError("Phase1 component resource receipt drift")
        object.__setattr__(
            self, "phase1_resource_receipt", MappingProxyType(resource)
        )

    @classmethod
    def from_phase1_component(
        cls,
        component: Any,
        *,
        checkpoint_weight: torch.Tensor | np.ndarray,
    ) -> "GroundCFMInput":
        try:
            from cvsrffi.phase1_grb_jp4_cfm_bundle import (
                COMPONENT_PROFILE,
                GRBJP4CFMPhase1Component,
                METHOD_LOCK_SCHEMA,
                PROTOCOL_SCHEMA,
                SCHEMA as PHASE1_COMPONENT_SCHEMA,
                canonical_array_sha256,
                class_handle_binding_sha256,
            )
        except ImportError as exc:
            raise GRBJP4CFMError(
                "the typed Phase1 GRB-JP4-CFM component is unavailable"
            ) from exc
        if type(component) is not GRBJP4CFMPhase1Component:
            raise GRBJP4CFMError(
                "formal ground input requires exact typed Phase1 component"
            )
        manifest = dict(component.manifest)
        method = dict(component.method_lock)
        weight = _weight_array(checkpoint_weight)
        joint_sha = _array_sha(weight)
        checkpoint_sha = _require_sha(
            manifest["checkpoint_sha256"], "checkpoint_sha256"
        )
        arrays = {
            "p_g_q": np.asarray(component.p_g_q),
            "p_g_scale": np.asarray(component.p_g_scale),
            "p_g_mask": np.asarray(component.p_g_mask),
            "p_g_weight": np.asarray(component.p_g_weight),
            "p_g_radius": np.asarray(component.p_g_radius),
            "p_g_physical_counts": np.asarray(
                component.p_g_physical_counts
            ),
            "p_g_receipt_sha256": np.asarray(
                component.p_g_receipt_sha256
            ),
            "p_g_source_prototype_sha256": np.asarray(
                component.p_g_source_prototype_sha256
            ),
            "p_g_quantization_max_abs_error": np.asarray(
                component.p_g_quantization_max_abs_error
            ),
            "p_g_quantization_certificate_sha256": np.asarray(
                component.p_g_quantization_certificate_sha256
            ),
            "l_g_q": np.asarray(component.l_g_q),
            "l_g_scale": np.asarray(component.l_g_scale),
            "r_q": np.asarray(component.r_q),
            "r_scale": np.asarray(component.r_scale),
            "direction_energy_a": np.asarray(
                component.direction_energy_a
            ),
            "delta_q": np.asarray(np.float16(component.delta_q), dtype="<f2"),
            "tau_q": np.asarray(np.float16(component.tau_q), dtype="<f2"),
            "class_registry": np.asarray(component.class_registry),
            "feature_schema": np.asarray(manifest.get("feature_schema")),
            "protocol_schema": np.asarray(manifest.get("protocol_schema")),
        }
        array_hashes = dict(manifest.get("array_sha256", {}))
        counts = arrays["p_g_physical_counts"]
        mask = arrays["p_g_mask"]
        resource = dict(manifest.get("resource_audit", {}))
        expected_ground_wire = int(
            sum(
                arrays[name].nbytes
                for name in (
                    "p_g_q",
                    "p_g_scale",
                    "p_g_weight",
                    "p_g_radius",
                    "p_g_mask",
                    "p_g_physical_counts",
                    "p_g_receipt_sha256",
                    "p_g_source_prototype_sha256",
                    "p_g_quantization_max_abs_error",
                    "p_g_quantization_certificate_sha256",
                )
            )
        )
        expected_factor_numeric = int(
            arrays["l_g_q"].nbytes
            + arrays["l_g_scale"].nbytes
            + arrays["r_q"].nbytes
            + arrays["r_scale"].nbytes
            + arrays["direction_energy_a"].nbytes
            + THETA_NUMERIC_WIRE_BYTES
        )
        expected_factor_receipt = 4 * 64
        expected_margin_wire = int(
            arrays["delta_q"].nbytes + arrays["tau_q"].nbytes
        )
        expected_metadata_wire = int(
            arrays["class_registry"].nbytes
            + arrays["feature_schema"].nbytes
            + arrays["protocol_schema"].nbytes
        )
        expected_resource = {
            "ground_wire_bytes": expected_ground_wire,
            "jp4_update_factor_numeric_bytes": expected_factor_numeric,
            "jp4_update_factor_receipt_bytes": expected_factor_receipt,
            "jp4_update_factor_wire_bytes": (
                expected_factor_numeric + expected_factor_receipt
            ),
            "phase1_margin_wire_bytes": expected_margin_wire,
            "component_metadata_wire_bytes": expected_metadata_wire,
            "total_component_bytes": (
                expected_ground_wire
                + expected_factor_numeric
                + expected_factor_receipt
                + expected_margin_wire
                + expected_metadata_wire
            ),
            "jp4_update_factor_wire_limit_bytes": JP4_WIRE_LIMIT_BYTES,
            "arm_state_limit_bytes": MAX_STATE_BYTES,
            "persistent_dense_float_bank_bytes": 0,
            "ground_direction_rank": RANK,
        }
        if (
            manifest.get("schema") != PHASE1_COMPONENT_SCHEMA
            or manifest.get("component_profile") != COMPONENT_PROFILE
            or manifest.get("method_lock_schema") != METHOD_LOCK_SCHEMA
            or manifest.get("protocol_schema") != PROTOCOL_SCHEMA
            or manifest.get("method_lock") != method
            or manifest.get("method_lock_sha256") != _sha_json(method)
            or manifest.get("class_handle_binding_sha256")
            != class_handle_binding_sha256(component.class_registry)
            or set(array_hashes) != set(arrays)
            or any(
                array_hashes[name] != canonical_array_sha256(value)
                for name, value in arrays.items()
            )
            or resource != expected_resource
            or counts.dtype != np.int16
            or counts.shape
            != (OLD_CLASS_COUNT, MAX_GROUND_PROTOTYPES)
            or np.any(counts[mask] < 2)
            or np.any(counts[~mask] != 0)
            or manifest.get("ground_old_multiprototype_enabled") is not True
            or manifest.get("phase2_phase1_component_immutable") is not True
            or manifest.get("phase2_phase1_component_update_access") is not False
            or manifest.get("ground_prototypes_enter_qknn_bank") is not False
            or manifest.get("ground_prototypes_generate_logits") is not False
            or manifest.get("ground_prototypes_add_k") is not False
        ):
            raise GRBJP4CFMError("loaded Phase1 component lifecycle drift")
        return cls(
            np.asarray(component.p_g_q),
            np.asarray(component.p_g_scale),
            np.asarray(component.p_g_mask),
            np.asarray(component.p_g_weight),
            np.asarray(component.p_g_radius),
            np.asarray(component.l_g_q),
            np.asarray(component.l_g_scale),
            np.asarray(component.r_q),
            np.asarray(component.r_scale),
            np.asarray(component.direction_energy_a),
            np.asarray(np.float16(component.delta_q), dtype="<f2"),
            np.asarray(np.float16(component.tau_q), dtype="<f2"),
            tuple(component.class_registry),
            checkpoint_sha,
            joint_sha,
            _require_sha(
                manifest["method_lock_sha256"], "method_lock_sha256"
            ),
            _require_sha(
                manifest["pre_sign_content_root_sha256"],
                "pre_sign_content_root_sha256",
            ),
            resource,
            _sha_json(resource),
        )

    def left(self) -> np.ndarray:
        return _decode_rows(
            self.left_codes,
            self.left_scales,
            rows=RANK,
            width=Z_DIM,
            name="L_g",
        )

    def right(self) -> np.ndarray:
        return _decode_rows(
            self.right_codes,
            self.right_scales,
            rows=RANK,
            width=HIDDEN_DIM,
            name="R",
        )

    def weighted_left(self) -> np.ndarray:
        return np.ascontiguousarray(
            self.left() * self.direction_energy.astype(np.float32)[:, None]
        )

    def barycenters(self) -> np.ndarray:
        result = np.empty((OLD_CLASS_COUNT, Z_DIM), dtype=np.float32)
        for class_index in range(OLD_CLASS_COUNT):
            mask = self.prototype_mask[class_index]
            decoded = (
                self.prototype_codes[class_index, mask].astype(np.float32)
                * self.prototype_scales[class_index, mask]
                .astype(np.float32)[:, None]
            )
            if (
                not np.isfinite(decoded).all()
                or np.any(
                    np.linalg.norm(decoded.astype(np.float64), axis=1)
                    <= 1.0e-12
                )
            ):
                raise GRBJP4CFMError(
                    "decoded ground multiprototype became zero/nonfinite"
                )
            weights = self.prototype_weights[class_index, mask].astype(
                np.float32
            )
            result[class_index] = np.einsum(
                "m,md->d", weights, decoded
            ).astype(np.float32)
        return _unit_rows(result, "ground barycenters")

    @property
    def digest(self) -> str:
        header = {
            "schema": self.schema,
            "old_class_order": list(self.old_class_order),
            "checkpoint_sha256": self.checkpoint_sha256,
            "joint_weight_sha256": self.joint_weight_sha256,
            "phase1_method_lock_sha256": self.phase1_method_lock_sha256,
            "component_digest": self.component_digest,
            "phase1_resource_receipt_sha256": (
                self.phase1_resource_receipt_sha256
            ),
            "delta_q_hex": self.delta_q_fp16.tobytes().hex(),
            "tau_q_hex": self.tau_q_fp16.tobytes().hex(),
        }
        payload = b"".join(
            (
                _canonical_json(header),
                self.prototype_codes.tobytes(),
                self.prototype_scales.tobytes(),
                self.prototype_mask.tobytes(),
                self.prototype_weights.tobytes(),
                self.prototype_radii.tobytes(),
                self.left_codes.tobytes(),
                self.left_scales.tobytes(),
                self.right_codes.tobytes(),
                self.right_scales.tobytes(),
                self.direction_energy.tobytes(),
            )
        )
        return _sha_bytes(payload)

    @property
    def numeric_wire_bytes(self) -> int:
        return int(
            self.prototype_codes.nbytes
            + self.prototype_scales.nbytes
            + self.prototype_mask.nbytes
            + self.prototype_weights.nbytes
            + self.prototype_radii.nbytes
            + self.left_codes.nbytes
            + self.left_scales.nbytes
            + self.right_codes.nbytes
            + self.right_scales.nbytes
            + self.direction_energy.nbytes
            + self.delta_q_fp16.nbytes
            + self.tau_q_fp16.nbytes
        )

    @property
    def ground_multiprototype_numeric_bytes(self) -> int:
        return int(
            self.prototype_codes.nbytes
            + self.prototype_scales.nbytes
            + self.prototype_mask.nbytes
            + self.prototype_weights.nbytes
            + self.prototype_radii.nbytes
        )

    @property
    def jp4_factor_numeric_bytes(self) -> int:
        return int(
            self.left_codes.nbytes
            + self.left_scales.nbytes
            + self.right_codes.nbytes
            + self.right_scales.nbytes
            + self.direction_energy.nbytes
            + self.delta_q_fp16.nbytes
            + self.tau_q_fp16.nbytes
        )


@dataclass(frozen=True)
class FoldClosure:
    held_physical_token: str
    fit_physical_tokens: tuple[str, ...]
    qknn_bank_tokens: tuple[str, ...]
    d92_stat_tokens: tuple[str, ...]
    score_calibration_tokens: tuple[str, ...]
    normal_equation_tokens: tuple[str, ...]


def validate_fold_closure(closure: FoldClosure) -> dict[str, Any]:
    if type(closure) is not FoldClosure:
        raise GRBJP4CFMError("fold closure must be exact typed input")
    held = closure.held_physical_token
    _typed_tokens((held,), name="held physical token")
    fields = (
        closure.fit_physical_tokens,
        closure.qknn_bank_tokens,
        closure.d92_stat_tokens,
        closure.score_calibration_tokens,
        closure.normal_equation_tokens,
    )
    for field_value in fields:
        values = _typed_tokens(
            field_value, name="fold physical tokens", unique=True
        )
        if held in values:
            raise GRBJP4CFMError(
                "held physical sample contaminated a fold statistic"
            )
    if not all(set(value) == set(fields[0]) for value in fields[1:]):
        raise GRBJP4CFMError(
            "fold qKNN/D92/calibration/normal token sets diverged"
        )
    root = _sha_json(sorted(fields[0]))
    return {
        "held_token_sha256": _sha_json(held),
        "fit_token_root_sha256": root,
        "held_fully_excluded": True,
    }


def strict_physical_loo_fold(
    snapshot: "SupportSnapshot",
    *,
    held_physical_token: str,
) -> tuple["SupportSnapshot", dict[str, Any]]:
    """Materialize the one fold shared by qKNN, D92, calibration and solve."""
    if type(snapshot) is not SupportSnapshot:
        raise GRBJP4CFMError("LOO fold requires exact SupportSnapshot")
    _typed_tokens(
        (held_physical_token,), name="held physical token", unique=True
    )
    matches = [
        index
        for index, token in enumerate(snapshot.physical_tokens)
        if token == held_physical_token
    ]
    if len(matches) != 1:
        raise GRBJP4CFMError(
            "held physical token must identify exactly one support row"
        )
    positions = np.asarray(
        [
            index
            for index, token in enumerate(snapshot.physical_tokens)
            if token != held_physical_token
        ],
        dtype=np.intp,
    )
    tokens = tuple(snapshot.physical_tokens[index] for index in positions)
    closure = FoldClosure(
        held_physical_token,
        tokens,
        tokens,
        tokens,
        tokens,
        tokens,
    )
    receipt = validate_fold_closure(closure)
    receipt.update(
        {
            "materialized_fold_row_count": int(len(positions)),
            "shared_fold_for_qknn_d92_calibration_and_normal": True,
        }
    )
    return (
        SupportSnapshot(
            np.ascontiguousarray(snapshot.z_id[positions]),
            np.ascontiguousarray(snapshot.jacobian[positions]),
            tuple(snapshot.labels[index] for index in positions),
            tokens,
        ),
        receipt,
    )


@dataclass(frozen=True)
class SupportSnapshot:
    z_id: np.ndarray
    jacobian: np.ndarray
    labels: tuple[str, ...]
    physical_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        z = _finite_rows(self.z_id, Z_DIM, "support z_id")
        jac = _finite_jacobian(
            self.jacobian, len(z), "support Jacobian"
        )
        labels = _typed_tokens(self.labels, name="support labels")
        tokens = _typed_tokens(
            self.physical_tokens,
            name="support physical tokens",
            unique=True,
        )
        if len(labels) != len(z) or len(tokens) != len(z):
            raise GRBJP4CFMError("support snapshot row binding drift")
        object.__setattr__(self, "z_id", z)
        object.__setattr__(self, "jacobian", jac)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "physical_tokens", tokens)


class SupportBackend(Protocol):
    def snapshot(self, theta: np.ndarray) -> SupportSnapshot:
        """Return support features/Jacobians at the exact full theta."""


@dataclass
class AffineSupportBackend:
    """Deterministic algebra backend used by focused solver tests."""

    base_z_id: np.ndarray
    jacobian: np.ndarray
    labels: Sequence[str]
    physical_tokens: Sequence[str]

    def snapshot(self, theta: np.ndarray) -> SupportSnapshot:
        base = _finite_rows(self.base_z_id, Z_DIM, "base support z_id")
        jac = _finite_jacobian(
            self.jacobian, len(base), "base support Jacobian"
        )
        value = np.asarray(theta, dtype=np.float64)
        if value.shape != (RANK,) or not np.isfinite(value).all():
            raise GRBJP4CFMError("backend theta drift")
        moved = base.astype(np.float64) + np.einsum(
            "nrd,r->nd", jac.astype(np.float64), value
        )
        return SupportSnapshot(
            _unit_rows(moved.astype(np.float32), "affine moved support"),
            jac,
            tuple(self.labels),
            tuple(self.physical_tokens),
        )


@dataclass(frozen=True)
class _ExcludedPhysicalBackend:
    """Backend view that physically removes one support row at every theta."""

    backend: SupportBackend
    held_physical_token: str

    def snapshot(self, theta: np.ndarray) -> SupportSnapshot:
        full = self.backend.snapshot(theta)
        fold, receipt = strict_physical_loo_fold(
            full, held_physical_token=self.held_physical_token
        )
        if receipt["held_fully_excluded"] is not True:
            raise GRBJP4CFMError("physical exclusion backend closure drift")
        return fold


@dataclass
class _TorchSupportBackend:
    base_forward: StrictForward
    direction_response: np.ndarray
    labels: tuple[str, ...]
    physical_tokens: tuple[str, ...]

    def snapshot(self, theta: np.ndarray) -> SupportSnapshot:
        value = np.asarray(theta, dtype=np.float64)
        response = np.asarray(self.direction_response, dtype=np.float64)
        pre_base = np.asarray(self.base_forward.pre_relu, dtype=np.float64)
        if (
            value.shape != (RANK,)
            or response.shape != (len(pre_base), RANK, Z_DIM)
            or not np.isfinite(value).all()
            or not np.isfinite(response).all()
        ):
            raise GRBJP4CFMError("cached strict support backend drift")
        pre = pre_base + np.einsum("nrd,r->nd", response, value)
        raw = np.maximum(pre, 0.0)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        if np.any(norms <= 1.0e-12) or not np.isfinite(norms).all():
            raise GRBJP4CFMError(
                "theta produced zero/nonfinite post-ReLU support"
            )
        unit = raw / norms
        active_response = response * (pre[:, None, :] > 0.0)
        projected = active_response - np.einsum(
            "nd,nrd->nr", unit, active_response
        )[:, :, None] * unit[:, None, :]
        jacobian = projected / norms[:, None, :]
        return SupportSnapshot(
            np.ascontiguousarray(unit.astype(np.float32)),
            np.ascontiguousarray(jacobian.astype(np.float32)),
            self.labels,
            self.physical_tokens,
        )


def _directions(ground: GroundCFMInput) -> np.ndarray:
    return np.ascontiguousarray(
        np.einsum(
            "ri,rj->rij",
            ground.weighted_left(),
            ground.right(),
        ).astype(np.float32)
    )


def _delta_weight(theta: np.ndarray, ground: GroundCFMInput) -> np.ndarray:
    value = np.asarray(theta, dtype=np.float64)
    if value.shape != (RANK,) or not np.isfinite(value).all():
        raise GRBJP4CFMError("theta must be finite rank four")
    return np.ascontiguousarray(
        np.einsum(
            "r,ri,rj->ij",
            value,
            ground.weighted_left().astype(np.float64),
            ground.right().astype(np.float64),
        ).astype(np.float32)
    )


def _project_theta(
    theta: np.ndarray,
    *,
    ground: GroundCFMInput,
    base_weight: np.ndarray,
) -> np.ndarray:
    value = np.asarray(theta, dtype=np.float64)
    if value.shape != (RANK,) or not np.isfinite(value).all():
        raise GRBJP4CFMError("trust projection theta drift")
    boxed = np.clip(value, -1.0, 1.0)
    r_weight = float(np.linalg.norm(base_weight.astype(np.float64))) / TRUST_DENOMINATOR
    delta_norm = float(
        np.linalg.norm(_delta_weight(boxed, ground).astype(np.float64))
    )
    factor = min(1.0, r_weight / max(delta_norm, TRUST_NORM_FLOOR))
    projected = boxed * factor
    if np.any(np.abs(projected) > 1.0 + 1.0e-12):
        raise GRBJP4CFMError("box projection failed")
    if (
        np.linalg.norm(_delta_weight(projected, ground).astype(np.float64))
        > r_weight + 32.0 * np.finfo(np.float32).eps * max(1.0, r_weight)
    ):
        raise GRBJP4CFMError("Frobenius trust projection failed")
    return np.ascontiguousarray(projected.astype(np.float64))


def _validate_support_layout(
    snapshot: SupportSnapshot,
    ground: GroundCFMInput,
    registered_old_classes: tuple[str, ...],
    registered_new_classes: tuple[str, ...],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    labels = snapshot.labels
    old = _typed_tokens(
        registered_old_classes,
        name="registered old classes",
        unique=True,
    )
    new = _typed_tokens(
        registered_new_classes,
        name="registered new classes",
        unique=True,
    )
    if (
        not old
        or not new
        or set(old) & set(new)
        or not set(old).issubset(set(ground.old_class_order))
    ):
        raise GRBJP4CFMError("explicit old/new registry partition drift")
    registered = (*old, *new)
    if set(labels) != set(registered):
        raise GRBJP4CFMError(
            "support labels must exactly match explicit old/new registries"
        )
    counts = {label: labels.count(label) for label in registered}
    if len(set(counts.values())) != 1:
        raise GRBJP4CFMError("support must be balanced K-shot by class")
    k_shot = next(iter(counts.values()))
    if k_shot not in K_VALUES:
        raise GRBJP4CFMError("support K must be one of 1/5/10")
    return k_shot, registered, new


def _ground_rows(
    snapshot: SupportSnapshot,
    ground: GroundCFMInput,
    *,
    active_old_classes: tuple[str, ...],
    excluded_token: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], tuple[str, ...]]:
    prototypes = ground.barycenters().astype(np.float64)
    ground_index = {
        label: index for index, label in enumerate(ground.old_class_order)
    }
    old = _typed_tokens(
        active_old_classes, name="active old classes", unique=True
    )
    if not old or not set(old).issubset(set(ground.old_class_order)):
        raise GRBJP4CFMError("active old registry lacks ground coverage")
    old_positions = [
        index
        for index, (label, token) in enumerate(
            zip(snapshot.labels, snapshot.physical_tokens)
        )
        if label in set(old) and token != excluded_token
    ]
    if not old_positions:
        raise GRBJP4CFMError("ground equation has no old support rows")
    class_counts = {
        label: sum(
            1
            for index in old_positions
            if snapshot.labels[index] == label
        )
        for label in old
    }
    if any(count == 0 for count in class_counts.values()):
        raise GRBJP4CFMError("ground exclusion removed an entire required class")
    G = np.zeros((RANK, RANK), dtype=np.float64)
    b = np.zeros((RANK,), dtype=np.float64)
    contributions: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.zeros((RANK, RANK), dtype=np.float64),
            np.zeros((RANK,), dtype=np.float64),
        )
        for _ in snapshot.labels
    ]
    tokens: list[str] = []
    for index in old_positions:
        label = snapshot.labels[index]
        jac = snapshot.jacobian[index].astype(np.float64)
        residual = prototypes[ground_index[label]] - snapshot.z_id[index].astype(
            np.float64
        )
        weight = 1.0 / (
            len(old) * class_counts[label] * float(Z_DIM)
        )
        Gi = weight * np.einsum("rd,sd->rs", jac, jac)
        bi = weight * np.einsum("rd,d->r", jac, residual)
        G += Gi
        b += bi
        contributions[index] = (Gi, bi)
        tokens.append(snapshot.physical_tokens[index])
    return G, b, contributions, tuple(tokens)


def _logsumexp(values: np.ndarray) -> float:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise GRBJP4CFMError("logsumexp input drift")
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


@dataclass(frozen=True)
class _OOFTuple:
    sample_index: int
    label: str
    token: str
    margin: float
    gradient: np.ndarray
    deficit: float
    active: bool
    closure_receipt: Mapping[str, Any]


def _qknn_oof_tuple(
    snapshot: SupportSnapshot,
    held_index: int,
    *,
    lock: CFMMethodLock,
    registered_classes: tuple[str, ...],
) -> _OOFTuple:
    token = snapshot.physical_tokens[held_index]
    fold, closure_receipt = strict_physical_loo_fold(
        snapshot, held_physical_token=token
    )
    original_by_token = {
        item: index for index, item in enumerate(snapshot.physical_tokens)
    }
    bank_positions = [
        original_by_token[item] for item in fold.physical_tokens
    ]
    query = snapshot.z_id[held_index].astype(np.float64)
    query_jac = snapshot.jacobian[held_index].astype(np.float64)
    scores = np.empty((len(registered_classes),), dtype=np.float64)
    gradients = np.empty((len(registered_classes), RANK), dtype=np.float64)
    coefficient = -0.5 * (
        float(lock.student_nu) + float(lock.kernel_effective_dim)
    )
    for class_position, class_label in enumerate(registered_classes):
        candidates = [
            index
            for index in bank_positions
            if snapshot.labels[index] == class_label
        ]
        if not candidates:
            raise GRBJP4CFMError(
                "strict LOO left a registered class without a neighbor"
            )
        ordered = sorted(
            candidates,
            key=lambda index: (
                float(
                    np.sum(
                        (
                            query
                            - snapshot.z_id[index].astype(np.float64)
                        )
                        ** 2
                    )
                ),
                snapshot.physical_tokens[index].encode("utf-8"),
            ),
        )
        chosen = ordered[: min(lock.qknn_neighbor_count, len(ordered))]
        local_support = snapshot.z_id[chosen].astype(np.float64)
        if len(local_support) <= 1:
            class_scale = float(lock.kernel_scale)
        else:
            cosine = np.clip(
                local_support @ local_support.T, -1.0, 1.0
            )
            distances = np.maximum(2.0 * (1.0 - cosine), 0.0)
            upper = distances[np.triu_indices(len(local_support), 1)]
            empirical = float(np.mean(upper))
            shrunk = (
                empirical
                + float(lock.scale_prior_strength)
                * float(lock.kernel_scale) ** 2
            ) / (1.0 + float(lock.scale_prior_strength))
            class_scale = float(
                np.clip(
                    math.sqrt(max(shrunk, np.finfo(np.float64).eps)),
                    float(lock.kernel_scale)
                    * float(lock.scale_min_ratio),
                    float(lock.kernel_scale)
                    * float(lock.scale_max_ratio),
                )
            )
        denominator = float(lock.student_nu) * class_scale**2
        volume_term = (
            -float(lock.kernel_volume_gamma)
            * float(lock.kernel_effective_dim)
            * math.log(class_scale)
        )
        log_kernel = np.empty((len(chosen),), dtype=np.float64)
        kernel_gradient = np.empty((len(chosen), RANK), dtype=np.float64)
        for position, bank_index in enumerate(chosen):
            difference = query - snapshot.z_id[bank_index].astype(np.float64)
            jac_difference = (
                query_jac
                - snapshot.jacobian[bank_index].astype(np.float64)
            )
            distance = float(np.dot(difference, difference))
            distance_gradient = 2.0 * np.einsum(
                "d,rd->r", difference, jac_difference
            )
            ratio = distance / denominator
            log_kernel[position] = volume_term + coefficient * math.log1p(
                ratio
            )
            kernel_gradient[position] = (
                coefficient
                * distance_gradient
                / (denominator * (1.0 + ratio))
            )
        class_lse = _logsumexp(log_kernel)
        posterior = np.exp(log_kernel - class_lse)
        scores[class_position] = class_lse - math.log(float(len(chosen)))
        gradients[class_position] = np.einsum(
            "n,nr->r", posterior, kernel_gradient
        )
    true_position = registered_classes.index(snapshot.labels[held_index])
    other_positions = np.asarray(
        [index for index in range(len(scores)) if index != true_position],
        dtype=np.intp,
    )
    other_lse = _logsumexp(scores[other_positions])
    other_weights = np.exp(scores[other_positions] - other_lse)
    margin = float(scores[true_position] - other_lse)
    margin_gradient = gradients[true_position] - np.einsum(
        "n,nr->r", other_weights, gradients[other_positions]
    )
    scaled_gradient = margin_gradient / float(lock.tau_q)
    deficit = float(lock.delta_q - margin / float(lock.tau_q))
    return _OOFTuple(
        held_index,
        snapshot.labels[held_index],
        token,
        margin,
        np.ascontiguousarray(scaled_gradient),
        deficit,
        deficit > 0.0,
        closure_receipt,
    )


def _cfm_system(
    tuples: Sequence[_OOFTuple],
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray]],
    np.ndarray,
    np.ndarray,
]:
    C = np.zeros((RANK, RANK), dtype=np.float64)
    b = np.zeros((RANK,), dtype=np.float64)
    contributions: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.zeros((RANK, RANK), dtype=np.float64),
            np.zeros((RANK,), dtype=np.float64),
        )
        for _ in tuples
    ]
    design_rows: list[np.ndarray] = []
    residual_rows: list[float] = []
    by_label = {
        label: [item for item in tuples if item.label == label]
        for label in (*old_classes, *new_classes)
    }
    if any(not by_label[label] for label in by_label):
        raise GRBJP4CFMError("CFM task/class balance has an empty class")
    for tuple_index, item in enumerate(tuples):
        group = old_classes if item.label in set(old_classes) else new_classes
        weight = (
            0.5
            / float(len(group))
            / float(len(by_label[item.label]))
        )
        if not item.active:
            continue
        gradient = item.gradient.astype(np.float64)
        Ci = weight * np.outer(gradient, gradient)
        bi = weight * gradient * item.deficit
        C += Ci
        b += bi
        contributions[tuple_index] = (Ci, bi)
        design_rows.append(math.sqrt(weight) * gradient)
        residual_rows.append(math.sqrt(weight) * item.deficit)
    A = (
        np.vstack(design_rows).astype(np.float64)
        if design_rows
        else np.empty((0, RANK), dtype=np.float64)
    )
    residual = np.asarray(residual_rows, dtype=np.float64)
    return C, b, contributions, A, residual


def _ground_design(
    snapshot: SupportSnapshot,
    ground: GroundCFMInput,
    *,
    active_old_classes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    prototypes = ground.barycenters().astype(np.float64)
    ground_index = {
        label: index for index, label in enumerate(ground.old_class_order)
    }
    old = _typed_tokens(
        active_old_classes, name="active old classes", unique=True
    )
    class_counts = {
        label: snapshot.labels.count(label) for label in old
    }
    rows: list[np.ndarray] = []
    residuals: list[float] = []
    for index, label in enumerate(snapshot.labels):
        if label not in set(old):
            continue
        weight = 1.0 / (
            len(old) * class_counts[label] * float(Z_DIM)
        )
        root = math.sqrt(weight)
        jac = snapshot.jacobian[index].astype(np.float64)
        residual = prototypes[ground_index[label]] - snapshot.z_id[index].astype(
            np.float64
        )
        rows.extend(root * jac[:, dimension] for dimension in range(Z_DIM))
        residuals.extend((root * residual).tolist())
    return np.vstack(rows), np.asarray(residuals, dtype=np.float64)


def _solve_increment(
    H: np.ndarray,
    b: np.ndarray,
    *,
    round_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(H, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    if (
        matrix.shape != (RANK, RANK)
        or rhs.shape != (RANK,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(rhs).all()
    ):
        raise GRBJP4CFMError("normal equation contains nonfinite values")
    trace = float(np.trace(matrix))
    if trace <= 64.0 * np.finfo(np.float64).eps:
        raise GRBJP4CFMError("normal equation trace is degenerate")
    ridge = max(RIDGE_FLOOR, RIDGE_FRACTION * trace / float(RANK))
    regularized = matrix + ridge * np.eye(RANK, dtype=np.float64)
    condition = float(np.linalg.cond(regularized))
    if not np.isfinite(condition) or condition > COND_LIMIT:
        raise GRBJP4CFMError("regularized normal equation is ill-conditioned")
    try:
        increment = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError as exc:
        raise GRBJP4CFMError("normal equation solve failed") from exc
    residual = regularized @ increment - rhs
    relative = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    )
    if not np.isfinite(relative) or relative > SOLVE_RELATIVE_RESIDUAL_LIMIT:
        raise GRBJP4CFMError("normal equation solve residual exceeded lock")
    return np.ascontiguousarray(increment), {
        "round": round_index,
        "status": "solved",
        "trace": trace,
        "ridge": ridge,
        "condition": condition,
        "solve_relative_residual": relative,
        "normal_matrix_sha256": _array_sha(regularized),
        "rhs_sha256": _array_sha(rhs),
        "increment_sha256": _array_sha(increment),
    }


def _solve_increment_with_ground_off_zero_information(
    H: np.ndarray,
    b: np.ndarray,
    *,
    round_index: int,
    ground_equation_enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the held-only exact-zero ground-off falsifier contract.

    No tolerance is used: both arrays must be finite, correctly shaped, and
    bytewise/numerically exact zero.  Every formal or nonzero system follows
    the ordinary fail-closed solver.
    """
    matrix = np.asarray(H, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    exact_zero = (
        matrix.shape == (RANK, RANK)
        and rhs.shape == (RANK,)
        and np.isfinite(matrix).all()
        and np.isfinite(rhs).all()
        and np.array_equal(matrix, np.zeros((RANK, RANK), dtype=np.float64))
        and np.array_equal(rhs, np.zeros((RANK,), dtype=np.float64))
    )
    if ground_equation_enabled is False and exact_zero:
        increment = np.zeros((RANK,), dtype=np.float64)
        return increment, {
            "round": round_index,
            "status": "ground_off_zero_information_identity",
            "trace": 0.0,
            "ridge": None,
            "condition": None,
            "solve_relative_residual": 0.0,
            "normal_matrix_sha256": _array_sha(matrix),
            "rhs_sha256": _array_sha(rhs),
            "increment_sha256": _array_sha(increment),
            "exact_zero_H": True,
            "exact_zero_b": True,
            "held_falsifier_only": True,
        }
    return _solve_increment(matrix, rhs, round_index=round_index)


def _fit_kgt1_reduced_support_two_rounds(
    *,
    backend: SupportBackend,
    excluded_physical_token: str,
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    registered_classes: tuple[str, ...],
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    base_weight: np.ndarray,
    k_shot: int,
    ground_equation_enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit one K5/K10 fold after the outer held row is already absent.

    The fold is not derived by subtracting cached full-support contributions.
    Both active-set rounds rebuild ground rows, every remaining OOF tuple,
    task/class balance, and the normal equation from this reduced backend.
    """
    _typed_tokens(
        (excluded_physical_token,),
        name="excluded physical token",
        unique=True,
    )
    if (
        k_shot not in (5, 10)
        or type(ground_equation_enabled) is not bool
    ):
        raise GRBJP4CFMError("strict reduced fold requires K5/K10")
    registered = _typed_tokens(
        registered_classes, name="fold registered classes", unique=True
    )
    old = _typed_tokens(old_classes, name="fold old classes", unique=True)
    new = _typed_tokens(new_classes, name="fold new classes", unique=True)
    if (
        not old
        or not new
        or set(old) & set(new)
        or set(registered) != set((*old, *new))
    ):
        raise GRBJP4CFMError("fold old/new registry partition drift")

    theta = np.zeros((RANK,), dtype=np.float64)
    factor = float(k_shot - 1) / float(k_shot)
    g_k = float(k_shot) / float(k_shot + 4)
    support_root: str | None = None
    support_count: int | None = None
    round_receipts: list[dict[str, Any]] = []
    for round_index in range(ACTIVE_SET_STEPS):
        snapshot = backend.snapshot(theta)
        if excluded_physical_token in set(snapshot.physical_tokens):
            raise GRBJP4CFMError(
                "outer held physical sample entered reduced fold backend"
            )
        if set(snapshot.labels) != set(registered):
            raise GRBJP4CFMError(
                "reduced fold lost a registered class"
            )
        current_root = _sha_json(sorted(snapshot.physical_tokens))
        if support_root is None:
            support_root = current_root
            support_count = len(snapshot.physical_tokens)
        elif (
            current_root != support_root
            or len(snapshot.physical_tokens) != support_count
        ):
            raise GRBJP4CFMError(
                "reduced fold support identity changed between rounds"
            )

        if ground_equation_enabled:
            G, bg, _ground_parts, ground_tokens = _ground_rows(
                snapshot, ground, active_old_classes=old
            )
        else:
            G = np.zeros((RANK, RANK), dtype=np.float64)
            bg = np.zeros((RANK,), dtype=np.float64)
            ground_tokens = ()
        tuples = [
            _qknn_oof_tuple(
                snapshot,
                index,
                lock=lock,
                registered_classes=registered,
            )
            for index in range(len(snapshot.z_id))
        ]
        if (
            excluded_physical_token in set(ground_tokens)
            or any(
                item.token == excluded_physical_token for item in tuples
            )
        ):
            raise GRBJP4CFMError(
                "outer held physical sample entered fold ground/OOF rows"
            )
        bank_roots: list[str] = []
        for item in tuples:
            expected_bank_root = _sha_json(
                sorted(
                    token
                    for token in snapshot.physical_tokens
                    if token != item.token
                )
            )
            actual_bank_root = str(
                item.closure_receipt["fit_token_root_sha256"]
            )
            if actual_bank_root != expected_bank_root:
                raise GRBJP4CFMError(
                    "reduced fold OOF bank closure drift"
                )
            bank_roots.append(actual_bank_root)
        C, bc, _cfm_parts, _A, _r = _cfm_system(
            tuples, old_classes=old, new_classes=new
        )
        H = G + factor * C
        b = bg + factor * bc
        increment, solve = (
            _solve_increment_with_ground_off_zero_information(
                H,
                b,
                round_index=round_index,
                ground_equation_enabled=ground_equation_enabled,
            )
        )
        theta_before = theta.copy()
        theta = _project_theta(
            theta + g_k * increment,
            ground=ground,
            base_weight=base_weight,
        )
        oof_token_root = _sha_json(sorted(item.token for item in tuples))
        if oof_token_root != support_root:
            raise GRBJP4CFMError(
                "reduced fold OOF tuple universe drift"
            )
        solve.update(
            {
                "theta_base_sha256": _array_sha(theta_before),
                "theta_full_sha256": _array_sha(theta),
                "increment_accumulated_not_overwritten": True,
                "fold_support_token_root_sha256": support_root,
                "fold_support_row_count": support_count,
                "ground_token_root_sha256": _sha_json(
                    sorted(ground_tokens)
                ),
                "ground_row_count": len(ground_tokens),
                "oof_tuple_token_root_sha256": oof_token_root,
                "oof_tuple_count": len(tuples),
                "oof_bank_closure_root_sha256": _sha_json(
                    sorted(bank_roots)
                ),
                "oof_bank_count": len(bank_roots),
                "outer_held_absent_from_support_ground_and_oof": True,
                "normal_equation_rebuilt_from_fold_only": True,
                "class_task_weights_recomputed_after_outer_holdout": True,
            }
        )
        round_receipts.append(solve)

    if support_root is None or support_count is None:
        raise GRBJP4CFMError("reduced fold emitted no support closure")
    return np.ascontiguousarray(theta), {
        "schema": SCHEMA,
        "mode": "physical_delete_then_two_round_refit",
        "excluded_token_sha256": _sha_json(excluded_physical_token),
        "fold_support_token_root_sha256": support_root,
        "fold_support_row_count": support_count,
        "two_active_set_rounds_rebuilt_from_fold_only": True,
        "subtractive_normal_equation_approximation_used": False,
        "outer_held_absent_from_every_bank_ground_oof": True,
        "d92_statistics_materialized_in_solver": False,
        "d92_fold_token_root_sha256": support_root,
        "d92_fold_contract": (
            "head_must_rebuild_D92_from_exact_reduced_support_tokens"
        ),
        "rounds": round_receipts,
    }


def _coverage_receipt(
    ground_A: np.ndarray,
    ground_r: np.ndarray,
    cfm_A: np.ndarray,
    cfm_r: np.ndarray,
) -> dict[str, Any]:
    A = np.vstack((ground_A, cfm_A))
    residual = np.concatenate((ground_r, cfm_r))
    denominator = float(np.dot(residual, residual))
    if denominator == 0.0:
        coverage: float | None = None
    else:
        projection = A @ np.linalg.lstsq(A, residual, rcond=None)[0]
        coverage = float(np.dot(projection, projection) / denominator)
    return {
        "definition": "squared_projection_on_final_weighted_design",
        "rho_J": coverage,
        "row_count": int(len(A)),
        "design_sha256": _array_sha(np.ascontiguousarray(A)),
        "residual_sha256": _array_sha(np.ascontiguousarray(residual)),
        "used_for_selection": False,
    }


def _quantize_theta(
    theta: np.ndarray,
    *,
    ground: GroundCFMInput,
    base_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    projected = _project_theta(theta, ground=ground, base_weight=base_weight)
    peak = float(np.max(np.abs(projected)))
    r_weight = float(np.linalg.norm(base_weight.astype(np.float64))) / TRUST_DENOMINATOR
    if peak == 0.0:
        codes = np.zeros((RANK,), dtype=np.int8)
        scale = np.asarray(np.float16(0.0), dtype="<f2").reshape(())
    else:
        s0 = peak / 127.0
        codes = np.clip(np.rint(projected / s0), -127, 127).astype(np.int8)
        unit_delta_norm = float(
            np.linalg.norm(
                _delta_weight(codes.astype(np.float64), ground).astype(
                    np.float64
                )
            )
        )
        trust_scale = r_weight / max(unit_delta_norm, TRUST_NORM_FLOOR)
        scale = _fp16_floor(min(s0, trust_scale))
    theta_q = codes.astype(np.float32) * np.float32(scale)
    delta = _delta_weight(theta_q, ground)
    delta_norm = float(np.linalg.norm(delta.astype(np.float64)))
    if delta_norm > r_weight + 32.0 * np.finfo(np.float32).eps * max(
        1.0, r_weight
    ):
        raise GRBJP4CFMError("quantized theta violates Frobenius trust")
    singular = np.linalg.svd(delta.astype(np.float64), compute_uv=False)
    effective_rank = (
        0
        if len(singular) == 0 or singular[0] == 0.0
        else int(np.count_nonzero(singular > RANK_RELATIVE_THRESHOLD * singular[0]))
    )
    return np.ascontiguousarray(codes), scale, {
        "codec": "int8_rne_single_fp16_downward_scale_v2",
        "theta_code_sha256": _array_sha(codes),
        "theta_scale_fp16_hex": scale.tobytes().hex(),
        "theta_numeric_wire_bytes": THETA_NUMERIC_WIRE_BYTES,
        "trust_radius_fro": r_weight,
        "quantized_delta_fro": delta_norm,
        "trust_verified": True,
        "quantized_rank": effective_rank,
        "rank_threshold_relative": RANK_RELATIVE_THRESHOLD,
        "rank_used_for_fallback_or_selection": False,
    }


def _decode_theta(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if (
        codes.dtype != np.int8
        or codes.shape != (RANK,)
        or np.any(codes == -128)
        or scale.dtype != np.dtype("<f2")
        or scale.shape != ()
        or not np.isfinite(scale)
        or float(scale) < 0.0
        or (float(scale) == 0.0 and np.any(codes != 0))
    ):
        raise GRBJP4CFMError("theta INT8/FP16 state drift")
    return np.ascontiguousarray(
        codes.astype(np.float32) * np.float32(scale)
    )


def _semantic_weight_sha(
    base_weight: np.ndarray,
    theta_q: np.ndarray,
    ground: GroundCFMInput,
) -> str:
    return _array_sha(
        np.ascontiguousarray(base_weight + _delta_weight(theta_q, ground))
    )


def _loco_metrics(
    backend: SupportBackend,
    *,
    ground: GroundCFMInput,
    active_old_classes: tuple[str, ...],
    base_weight: np.ndarray,
    full_theta: np.ndarray,
) -> dict[str, Any]:
    full_delta = _delta_weight(full_theta, ground).astype(np.float64)
    full_delta_norm = float(np.linalg.norm(full_delta))
    full_theta_norm = float(np.linalg.norm(full_theta))
    folds: list[dict[str, Any]] = []
    old = _typed_tokens(
        active_old_classes, name="LOCO old classes", unique=True
    )
    for held_class in old:
        theta = np.zeros((RANK,), dtype=np.float64)
        round_receipts: list[dict[str, Any]] = []
        for round_index in range(ACTIVE_SET_STEPS):
            snapshot = backend.snapshot(theta)
            held_tokens = [
                token
                for label, token in zip(
                    snapshot.labels, snapshot.physical_tokens
                )
                if label == held_class
            ]
            if len(held_tokens) != 1:
                raise GRBJP4CFMError(
                    "K1 LOCO must remove exactly one old support pair"
                )
            fold_old = tuple(
                label for label in old if label != held_class
            )
            G, b, _parts, tokens = _ground_rows(
                snapshot,
                ground,
                active_old_classes=fold_old,
                excluded_token=held_tokens[0],
            )
            if any(
                label not in old
                for label, token in zip(
                    snapshot.labels, snapshot.physical_tokens
                )
                if token in tokens
            ):
                raise GRBJP4CFMError("K1 new support entered LOCO ground fit")
            increment, solve = _solve_increment(
                G, b, round_index=round_index
            )
            theta = _project_theta(
                theta + 0.2 * increment,
                ground=ground,
                base_weight=base_weight,
            )
            round_receipts.append(solve)
        theta_norm = float(np.linalg.norm(theta))
        cosine = (
            1.0
            if theta_norm == 0.0 and full_theta_norm == 0.0
            else float(
                np.dot(theta, full_theta)
                / max(theta_norm * full_theta_norm, TRUST_NORM_FLOOR)
            )
        )
        delta = _delta_weight(theta, ground).astype(np.float64)
        folds.append(
            {
                "held_old_class_sha256": _sha_json(held_class),
                "finite": bool(np.isfinite(theta).all()),
                "cosine_to_full": cosine,
                "theta_norm_ratio": theta_norm
                / max(full_theta_norm, TRUST_NORM_FLOOR),
                "relative_delta_weight_difference": float(
                    np.linalg.norm(delta - full_delta)
                    / max(full_delta_norm, TRUST_NORM_FLOOR)
                ),
                "round_solve_receipts": round_receipts,
                "new_support_rows_used": 0,
            }
        )
    cosines = np.asarray(
        [float(item["cosine_to_full"]) for item in folds], dtype=np.float64
    )
    differences = np.asarray(
        [
            float(item["relative_delta_weight_difference"])
            for item in folds
        ],
        dtype=np.float64,
    )
    return {
        "schema": SCHEMA,
        "fold_count": len(old),
        "folds": folds,
        "all_finite": bool(all(item["finite"] for item in folds)),
        "median_cosine": float(np.median(cosines)),
        "minimum_cosine": float(np.min(cosines)),
        "maximum_relative_delta_weight_difference": float(
            np.max(differences)
        ),
        "theta_full_nonzero": bool(full_theta_norm > 0.0),
        "stability_gate_pass": bool(
            full_theta_norm > 0.0
            and np.median(cosines) >= 0.75
            and np.min(cosines) > 0.0
            and np.max(differences) <= 1.0
            and all(item["finite"] for item in folds)
        ),
        "query_agreement_used_for_fit_or_selection": False,
        "new_support_rows_used": 0,
    }


@dataclass(frozen=True)
class CFMFitState:
    """Persisted state: only INT8 theta plus a single FP16 scale."""

    theta_codes: np.ndarray
    theta_scale: np.ndarray
    k_shot: int
    ground_digest: str
    lock_digest: str
    checkpoint_sha256: str
    joint_weight_sha256_before: str
    joint_weight_semantic_sha256: str
    fit_receipt: Mapping[str, Any]
    receipt_sha256: str = field(default="")
    schema: str = field(default=SCHEMA)

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise GRBJP4CFMError("fit-state schema drift")
        codes = _readonly_array(
            self.theta_codes,
            dtype=np.int8,
            shape=(RANK,),
            name="theta_codes",
        )
        scale = _readonly_array(
            self.theta_scale,
            dtype="<f2",
            shape=(),
            name="theta_scale",
        )
        object.__setattr__(self, "theta_codes", codes)
        object.__setattr__(self, "theta_scale", scale)
        _decode_theta(codes, scale)
        if type(self.k_shot) is not int or self.k_shot not in K_VALUES:
            raise GRBJP4CFMError("fit-state K drift")
        for name in (
            "ground_digest",
            "lock_digest",
            "checkpoint_sha256",
            "joint_weight_sha256_before",
            "joint_weight_semantic_sha256",
        ):
            _require_sha(getattr(self, name), name)
        receipt = dict(self.fit_receipt)
        required = {
            "schema",
            "candidate",
            "status",
            "k_shot",
            "query_rows_used_for_fit",
            "claim_scope",
            "ground_equation_enabled",
            "active_set_steps",
            "update_semantics",
            "ground_digest",
            "lock_digest",
            "checkpoint_sha256",
            "joint_weight_sha256_before",
            "joint_weight_semantic_sha256",
            "support_token_root_sha256",
            "registry_receipt",
            "rounds",
            "fold_receipt",
            "coverage_receipt",
            "rank_receipt",
            "loco_receipt",
            "wire_receipt",
            "semantic_receipt",
            "resource_receipt",
        }
        if (
            set(receipt) != required
            or receipt["schema"] != SCHEMA
            or receipt["candidate"] != CANDIDATE
            or receipt["status"]
            not in {
                "support_only_cfm_solved",
                "ground_off_cfm_solved",
                "ground_off_zero_information_identity",
            }
            or receipt["k_shot"] != self.k_shot
            or receipt["query_rows_used_for_fit"] != 0
            or type(receipt["ground_equation_enabled"]) is not bool
            or receipt["claim_scope"]
            != (
                "LOCAL_SUPPORT_ONLY_NOT_RELEASED"
                if receipt["ground_equation_enabled"]
                else HELD_FALSIFIER_SCOPE
            )
            or (
                receipt["ground_equation_enabled"]
                and receipt["status"] != "support_only_cfm_solved"
            )
            or (
                receipt["status"]
                == "ground_off_zero_information_identity"
                and (
                    receipt["ground_equation_enabled"] is not False
                    or np.any(codes != 0)
                    or float(scale) != 0.0
                )
            )
            or receipt["active_set_steps"] != ACTIVE_SET_STEPS
            or receipt["update_semantics"]
            != "theta_next=Pi(theta_base+g_K*u_increment)"
            or receipt["ground_digest"] != self.ground_digest
            or receipt["lock_digest"] != self.lock_digest
            or receipt["checkpoint_sha256"] != self.checkpoint_sha256
            or receipt["joint_weight_sha256_before"]
            != self.joint_weight_sha256_before
            or receipt["joint_weight_semantic_sha256"]
            != self.joint_weight_semantic_sha256
            or len(receipt["rounds"]) != ACTIVE_SET_STEPS
        ):
            raise GRBJP4CFMError("fit-state support-only receipt drift")
        payload = self._receipt_payload()
        expected = _sha_json(payload)
        if self.receipt_sha256:
            if self.receipt_sha256 != expected:
                raise GRBJP4CFMError("fit-state receipt SHA drift")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    def _receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "theta_codes_hex": self.theta_codes.tobytes().hex(),
            "theta_scale_hex": self.theta_scale.tobytes().hex(),
            "k_shot": self.k_shot,
            "ground_digest": self.ground_digest,
            "lock_digest": self.lock_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "joint_weight_sha256_before": self.joint_weight_sha256_before,
            "joint_weight_semantic_sha256": self.joint_weight_semantic_sha256,
            "fit_receipt": dict(self.fit_receipt),
        }

    def theta(self) -> np.ndarray:
        return _decode_theta(self.theta_codes, self.theta_scale)


def _fit_from_backend(
    *,
    backend: SupportBackend,
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    registered_old_classes: Sequence[str],
    registered_new_classes: Sequence[str],
    checkpoint_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
    ground_equation_enabled: bool = True,
) -> CFMFitState:
    if (
        type(ground) is not GroundCFMInput
        or type(lock) is not CFMMethodLock
        or type(ground_equation_enabled) is not bool
    ):
        raise GRBJP4CFMError("fit requires exact typed ground and method lock")
    ground.__post_init__()
    lock.__post_init__()
    if (
        lock.phase1_method_lock_sha256
        != ground.phase1_method_lock_sha256
        or np.float16(lock.delta_q).tobytes()
        != ground.delta_q_fp16.tobytes()
        or np.float16(lock.tau_q).tobytes()
        != ground.tau_q_fp16.tobytes()
    ):
        raise GRBJP4CFMError("ground/method-lock binding drift")
    base_weight = _weight_array(checkpoint_weight)
    before_sha = _array_sha(base_weight)
    if (
        _require_sha(checkpoint_sha256, "checkpoint_sha256")
        != ground.checkpoint_sha256
        or before_sha != ground.joint_weight_sha256
    ):
        raise GRBJP4CFMError("checkpoint/ground joint-weight binding drift")
    initial = backend.snapshot(np.zeros((RANK,), dtype=np.float64))
    active_old = _typed_tokens(
        registered_old_classes,
        name="registered old classes",
        unique=True,
    )
    active_new = _typed_tokens(
        registered_new_classes,
        name="registered new classes",
        unique=True,
    )
    k_shot, registered, new_classes = _validate_support_layout(
        initial, ground, active_old, active_new
    )
    if lock.qknn_neighbor_count != k_shot:
        raise GRBJP4CFMError("qKNN Phase1 lock active K/support K drift")
    canonical_order = tuple(
        sorted(
            range(len(initial.physical_tokens)),
            key=lambda index: initial.physical_tokens[index].encode("utf-8"),
        )
    )
    if canonical_order != tuple(range(len(initial.physical_tokens))):
        raise GRBJP4CFMError(
            "support must be pre-sorted by physical_sample_id bytes"
        )
    g_k = float(k_shot) / float(k_shot + 4)
    theta = np.zeros((RANK,), dtype=np.float64)
    round_receipts: list[dict[str, Any]] = []
    fold_receipts: list[dict[str, Any]] = []
    final_ground_A = np.empty((0, RANK), dtype=np.float64)
    final_ground_r = np.empty((0,), dtype=np.float64)
    final_cfm_A = np.empty((0, RANK), dtype=np.float64)
    final_cfm_r = np.empty((0,), dtype=np.float64)
    if k_shot == 1 and not ground_equation_enabled:
        for round_index in range(ACTIVE_SET_STEPS):
            round_receipts.append(
                {
                    "round": round_index,
                    "status": "ground_off_zero_information_identity",
                    "ground_off_k1_fixed_zero": True,
                    "active_oof_tuple_count": 0,
                    "all_oof_tuple_count": 0,
                    "theta_base_sha256": _array_sha(theta),
                    "theta_full_sha256": _array_sha(theta),
                    "increment_sha256": _array_sha(theta),
                    "increment_accumulated_not_overwritten": True,
                    "new_support_rows_used": 0,
                }
            )
        loco = {
            "schema": SCHEMA,
            "not_applicable": "ground_off_K1_fixed_theta_zero",
            "new_support_rows_used": 0,
            "query_agreement_used_for_fit_or_selection": False,
        }
        fold_summary = {
            "mode": "K1_ground_off_fixed_theta_zero",
            "fold_count": 0,
            "new_support_rows_used": 0,
            "all_held_fully_excluded": True,
        }
    elif k_shot == 1:
        for round_index in range(ACTIVE_SET_STEPS):
            snapshot = backend.snapshot(theta)
            G, b, _parts, ground_tokens = _ground_rows(
                snapshot, ground, active_old_classes=active_old
            )
            if any(
                label not in active_old
                for label, token in zip(
                    snapshot.labels, snapshot.physical_tokens
                )
                if token in set(ground_tokens)
            ):
                raise GRBJP4CFMError("K1 new support entered shared fit")
            increment, solve = _solve_increment(
                G, b, round_index=round_index
            )
            theta_before = theta.copy()
            theta = _project_theta(
                theta + g_k * increment,
                ground=ground,
                base_weight=base_weight,
            )
            solve.update(
                {
                    "active_oof_tuple_count": 0,
                    "all_oof_tuple_count": 0,
                    "theta_base_sha256": _array_sha(theta_before),
                    "theta_full_sha256": _array_sha(theta),
                    "increment_accumulated_not_overwritten": True,
                    "new_support_rows_used": 0,
                }
            )
            round_receipts.append(solve)
            final_ground_A, final_ground_r = _ground_design(
                snapshot, ground, active_old_classes=active_old
            )
        loco = _loco_metrics(
            backend,
            ground=ground,
            active_old_classes=active_old,
            base_weight=base_weight,
            full_theta=theta,
        )
        fold_summary: Mapping[str, Any] = {
            "mode": "K1_old_only_six_fold_LOCO",
            "fold_count": len(active_old),
            "new_support_rows_used": 0,
            "all_held_fully_excluded": True,
        }
    else:
        # Round zero: theta0 is support-independent.  Build strict OOF tuples
        # from banks where the held physical sample is absent everywhere.
        snapshot0 = backend.snapshot(theta)
        if ground_equation_enabled:
            G0, bg0, _ground_parts0, _ground_tokens0 = _ground_rows(
                snapshot0, ground, active_old_classes=active_old
            )
        else:
            G0 = np.zeros((RANK, RANK), dtype=np.float64)
            bg0 = np.zeros((RANK,), dtype=np.float64)
        tuples0 = [
            _qknn_oof_tuple(
                snapshot0,
                index,
                lock=lock,
                registered_classes=registered,
            )
            for index in range(len(snapshot0.z_id))
        ]
        C0, bc0, _cfm_parts0, _A0, _r0 = _cfm_system(
            tuples0,
            old_classes=active_old,
            new_classes=new_classes,
        )
        factor = float(k_shot - 1) / float(k_shot)
        H0 = G0 + factor * C0
        b0 = bg0 + factor * bc0
        increment1, solve0 = (
            _solve_increment_with_ground_off_zero_information(
                H0,
                b0,
                round_index=0,
                ground_equation_enabled=ground_equation_enabled,
            )
        )
        theta0 = theta.copy()
        theta1 = _project_theta(
            theta0 + g_k * increment1,
            ground=ground,
            base_weight=base_weight,
        )
        solve0.update(
            {
                "active_oof_tuple_count": int(
                    sum(item.active for item in tuples0)
                ),
                "all_oof_tuple_count": len(tuples0),
                "theta_base_sha256": _array_sha(theta0),
                "theta_full_sha256": _array_sha(theta1),
                "increment_accumulated_not_overwritten": True,
            }
        )
        round_receipts.append(solve0)
        fold_tuples1: list[_OOFTuple] = []
        for item0 in tuples0:
            held_index = item0.sample_index
            reduced_backend = _ExcludedPhysicalBackend(
                backend, item0.token
            )
            theta_fold, fold_refit = (
                _fit_kgt1_reduced_support_two_rounds(
                    backend=reduced_backend,
                    excluded_physical_token=item0.token,
                    ground=ground,
                    lock=lock,
                    registered_classes=registered,
                    old_classes=active_old,
                    new_classes=new_classes,
                    base_weight=base_weight,
                    k_shot=k_shot,
                    ground_equation_enabled=ground_equation_enabled,
                )
            )
            fold_snapshot = backend.snapshot(theta_fold)
            if (
                fold_snapshot.labels != snapshot0.labels
                or fold_snapshot.physical_tokens
                != snapshot0.physical_tokens
            ):
                raise GRBJP4CFMError(
                    "support backend changed row identity across folds"
                )
            item1 = _qknn_oof_tuple(
                fold_snapshot,
                held_index,
                lock=lock,
                registered_classes=registered,
            )
            if (
                item1.closure_receipt["fit_token_root_sha256"]
                != fold_refit["fold_support_token_root_sha256"]
            ):
                raise GRBJP4CFMError(
                    "held evaluation bank/refit support closure drift"
                )
            fold_tuples1.append(item1)
            fold_receipts.append(
                {
                    "held_token_sha256": _sha_json(item1.token),
                    "fold_theta_sha256": _array_sha(theta_fold),
                    "fold_refit": fold_refit,
                    "closure": dict(item1.closure_receipt),
                    "held_ground_contribution_removed": bool(
                        item1.label in set(active_old)
                    ),
                    "held_oof_tuple_removed": True,
                    "held_removed_before_all_other_oof_tuples": True,
                    "subtractive_normal_equation_approximation_used": False,
                    "d92_statistics_materialized_in_solver": False,
                    "d92_fold_token_root_sha256": fold_refit[
                        "d92_fold_token_root_sha256"
                    ],
                }
            )
        snapshot1 = backend.snapshot(theta1)
        if ground_equation_enabled:
            G1, bg1, _ground_parts1, _ground_tokens1 = _ground_rows(
                snapshot1, ground, active_old_classes=active_old
            )
        else:
            G1 = np.zeros((RANK, RANK), dtype=np.float64)
            bg1 = np.zeros((RANK,), dtype=np.float64)
        C1, bc1, _cfm_parts1, final_cfm_A, final_cfm_r = _cfm_system(
            fold_tuples1,
            old_classes=active_old,
            new_classes=new_classes,
        )
        H1 = G1 + factor * C1
        b1 = bg1 + factor * bc1
        increment2, solve1 = (
            _solve_increment_with_ground_off_zero_information(
                H1,
                b1,
                round_index=1,
                ground_equation_enabled=ground_equation_enabled,
            )
        )
        theta2 = _project_theta(
            theta1 + g_k * increment2,
            ground=ground,
            base_weight=base_weight,
        )
        if np.array_equal(theta2, _project_theta(
            g_k * increment2, ground=ground, base_weight=base_weight
        )) and np.linalg.norm(theta1) > 1.0e-12:
            # This is diagnostic only: equality can occur after clipping, so
            # the authoritative evidence is the explicit base/increment hashes.
            overwrite_collision = True
        else:
            overwrite_collision = False
        solve1.update(
            {
                "active_oof_tuple_count": int(
                    sum(item.active for item in fold_tuples1)
                ),
                "all_oof_tuple_count": len(fold_tuples1),
                "theta_base_sha256": _array_sha(theta1),
                "theta_full_sha256": _array_sha(theta2),
                "increment_accumulated_not_overwritten": True,
                "overwrite_value_collision": overwrite_collision,
            }
        )
        round_receipts.append(solve1)
        theta = theta2
        if ground_equation_enabled:
            final_ground_A, final_ground_r = _ground_design(
                snapshot1, ground, active_old_classes=active_old
            )
        final_cfm_A = math.sqrt(factor) * final_cfm_A
        final_cfm_r = math.sqrt(factor) * final_cfm_r
        loco = None
        compact_fold_receipts = [
            {
                "held_token_sha256": item["held_token_sha256"],
                "fold_theta_sha256": item["fold_theta_sha256"],
                "fold_support_token_root_sha256": item["fold_refit"][
                    "fold_support_token_root_sha256"
                ],
                "fold_support_row_count": item["fold_refit"][
                    "fold_support_row_count"
                ],
                "two_active_set_rounds_rebuilt_from_fold_only": item[
                    "fold_refit"
                ]["two_active_set_rounds_rebuilt_from_fold_only"],
                "outer_held_absent_from_every_bank_ground_oof": item[
                    "fold_refit"
                ]["outer_held_absent_from_every_bank_ground_oof"],
                "subtractive_normal_equation_approximation_used": item[
                    "subtractive_normal_equation_approximation_used"
                ],
                "held_evaluation_bank_token_root_sha256": item["closure"][
                    "fit_token_root_sha256"
                ],
                "d92_fold_token_root_sha256": item[
                    "d92_fold_token_root_sha256"
                ],
                "d92_statistics_materialized_in_solver": False,
                "fold_refit_receipt_sha256": _sha_json(
                    item["fold_refit"]
                ),
            }
            for item in fold_receipts
        ]
        fold_summary = {
            "mode": "K5_K10_strict_physical_LOO",
            "fold_count": len(fold_receipts),
            "all_held_fully_excluded": bool(
                all(
                    item["closure"]["held_fully_excluded"]
                    for item in fold_receipts
                )
            ),
            "folds": compact_fold_receipts,
            "fold_receipt_root_sha256": _sha_json(
                compact_fold_receipts
            ),
            "qknn_d92_token_roots_identical": True,
            "all_folds_delete_then_two_round_refit": bool(
                all(
                    item["fold_refit"][
                        "two_active_set_rounds_rebuilt_from_fold_only"
                    ]
                    and not item[
                        "subtractive_normal_equation_approximation_used"
                    ]
                    for item in fold_receipts
                )
            ),
            "d92_statistics_materialized_in_solver": False,
            "d92_fold_contract": (
                "head_must_rebuild_D92_from_exact_reduced_support_tokens"
            ),
        }
    codes, scale, rank_receipt = _quantize_theta(
        theta, ground=ground, base_weight=base_weight
    )
    theta_q = _decode_theta(codes, scale)
    semantic_sha = _semantic_weight_sha(base_weight, theta_q, ground)
    coverage = _coverage_receipt(
        final_ground_A, final_ground_r, final_cfm_A, final_cfm_r
    )
    support_root = _sha_json(sorted(initial.physical_tokens))
    jp4_numeric = (
        ground.jp4_factor_numeric_bytes + THETA_NUMERIC_WIRE_BYTES
    )
    phase1_resource = dict(ground.phase1_resource_receipt)
    update_factor_wire_bytes = phase1_resource[
        "jp4_update_factor_wire_bytes"
    ]
    if (
        jp4_numeric > JP4_WIRE_LIMIT_BYTES
        or update_factor_wire_bytes > JP4_WIRE_LIMIT_BYTES
    ):
        raise GRBJP4CFMError("JP4 numeric wire exceeds 4096 bytes")
    support_rows = len(initial.z_id)
    jacobian_mac = (
        support_rows
        * RANK
        * (Z_DIM * HIDDEN_DIM + 3 * Z_DIM)
    )
    pair_mac = (
        support_rows * (support_rows - 1) * Z_DIM
        if k_shot > 1
        else 0
    )
    support_mac = jacobian_mac + pair_mac + ACTIVE_SET_STEPS * RANK**3
    if support_mac >= MAX_SUPPORT_MAC:
        raise GRBJP4CFMError("support fit exceeds frozen 65M MAC budget")
    resource = {
        "ground_multiprototype_numeric_bytes": (
            ground.ground_multiprototype_numeric_bytes
        ),
        "ground_wire_bytes": phase1_resource["ground_wire_bytes"],
        "jp4_factor_numeric_bytes": ground.jp4_factor_numeric_bytes,
        "update_factor_wire_bytes": update_factor_wire_bytes,
        "jp4_update_factor_wire_bytes": update_factor_wire_bytes,
        "total_component_numeric_bytes": ground.numeric_wire_bytes,
        "total_component_bytes": phase1_resource["total_component_bytes"],
        "arm_state_base_component_bytes": phase1_resource[
            "total_component_bytes"
        ],
        "phase1_resource_receipt_sha256": (
            ground.phase1_resource_receipt_sha256
        ),
        "theta_numeric_wire_bytes": THETA_NUMERIC_WIRE_BYTES,
        "jp4_numeric_wire_bytes": jp4_numeric,
        "jp4_wire_limit_bytes": JP4_WIRE_LIMIT_BYTES,
        "support_rows": support_rows,
        "analytic_jacobian_mac_upper_bound": jacobian_mac,
        "loo_pair_distance_mac_upper_bound": pair_mac,
        "solve_mac_upper_bound": ACTIVE_SET_STEPS * RANK**3,
        "support_fit_mac_upper_bound": support_mac,
        "support_fit_mac_limit": MAX_SUPPORT_MAC,
        "adapter_mac_per_query_after_merge": 0,
        "post_backbone_mac_limit_per_query": MAX_POST_BACKBONE_MAC_PER_QUERY,
        "persistent_fp32_theta_or_delta_weight_sidecar": False,
    }
    if (
        not ground_equation_enabled
        and all(
            item.get("status")
            == "ground_off_zero_information_identity"
            for item in round_receipts
        )
    ):
        fit_status = "ground_off_zero_information_identity"
    elif ground_equation_enabled:
        fit_status = "support_only_cfm_solved"
    else:
        fit_status = "ground_off_cfm_solved"
    receipt = {
        "schema": SCHEMA,
        "candidate": CANDIDATE,
        "status": fit_status,
        "k_shot": k_shot,
        "query_rows_used_for_fit": 0,
        "claim_scope": (
            "LOCAL_SUPPORT_ONLY_NOT_RELEASED"
            if ground_equation_enabled
            else HELD_FALSIFIER_SCOPE
        ),
        "ground_equation_enabled": ground_equation_enabled,
        "active_set_steps": ACTIVE_SET_STEPS,
        "update_semantics": "theta_next=Pi(theta_base+g_K*u_increment)",
        "ground_digest": ground.digest,
        "lock_digest": lock.digest,
        "checkpoint_sha256": checkpoint_sha256,
        "joint_weight_sha256_before": before_sha,
        "joint_weight_semantic_sha256": semantic_sha,
        "support_token_root_sha256": support_root,
        "registry_receipt": {
            "old_count": len(active_old),
            "new_count": len(active_new),
            "old_registry_sha256": _sha_json(list(active_old)),
            "new_registry_sha256": _sha_json(list(active_new)),
            "explicit_old_new_partition": True,
            "held_pseudo_new_ground_rows_used": 0,
        },
        "rounds": round_receipts,
        "fold_receipt": fold_summary,
        "coverage_receipt": coverage,
        "rank_receipt": rank_receipt,
        "loco_receipt": loco,
        "wire_receipt": {
            "fit_state_schema": FIT_STATE_WIRE_SCHEMA,
            "numeric_wire_bytes": jp4_numeric,
            "update_factor_wire_bytes": update_factor_wire_bytes,
            "ground_wire_bytes": phase1_resource["ground_wire_bytes"],
            "total_component_bytes": phase1_resource[
                "total_component_bytes"
            ],
            "arm_state_base_component_bytes": phase1_resource[
                "total_component_bytes"
            ],
            "phase1_resource_receipt_sha256": (
                ground.phase1_resource_receipt_sha256
            ),
            "limit_bytes": JP4_WIRE_LIMIT_BYTES,
        },
        "semantic_receipt": {
            "theta_code_sha256": _array_sha(codes),
            "theta_scale_fp16_hex": scale.tobytes().hex(),
            "joint_weight_sha256_before": before_sha,
            "joint_weight_semantic_sha256": semantic_sha,
            "same_theta_bytes_for_M_DA_and_M_DA92": True,
        },
        "resource_receipt": resource,
    }
    return CFMFitState(
        codes,
        scale,
        k_shot,
        ground.digest,
        lock.digest,
        checkpoint_sha256,
        before_sha,
        semantic_sha,
        receipt,
    )


def fit_cfm_from_precomputed(
    *,
    base_support_zid: np.ndarray,
    support_jacobian: np.ndarray,
    support_labels: Sequence[str],
    support_physical_tokens: Sequence[str],
    registered_old_classes: Sequence[str],
    registered_new_classes: Sequence[str],
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    checkpoint_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
) -> CFMFitState:
    """Deterministic algebra entry for focused local verification."""
    backend = AffineSupportBackend(
        base_support_zid,
        support_jacobian,
        support_labels,
        support_physical_tokens,
    )
    return _fit_from_backend(
        backend=backend,
        ground=ground,
        lock=lock,
        registered_old_classes=registered_old_classes,
        registered_new_classes=registered_new_classes,
        checkpoint_weight=checkpoint_weight,
        checkpoint_sha256=checkpoint_sha256,
    )


def fit_cfm_from_support_iq(
    *,
    model: Any,
    support_iq: torch.Tensor,
    support_labels: Sequence[str],
    support_physical_tokens: Sequence[str],
    registered_old_classes: Sequence[str],
    registered_new_classes: Sequence[str],
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    checkpoint_sha256: str,
) -> CFMFitState:
    """Formal local fit entry; query IQ is not accepted by this API."""
    if (
        not torch.is_tensor(support_iq)
        or support_iq.dtype != torch.float32
        or support_iq.ndim != 3
        or support_iq.shape[1] != 2
        or not torch.isfinite(support_iq).all()
    ):
        raise GRBJP4CFMError(
            "support IQ must be finite float32 [N,2,T]"
        )
    labels = _typed_tokens(support_labels, name="support labels")
    tokens = _typed_tokens(
        support_physical_tokens,
        name="support physical tokens",
        unique=True,
    )
    if len(labels) != len(support_iq) or len(tokens) != len(support_iq):
        raise GRBJP4CFMError("support IQ row binding drift")
    linear = _joint_proj_linear(model)
    base_weight = _weight_array(linear.weight)
    try:
        base_forward = _legacy_strict_zid_with_hook(model, support_iq)
    except Exception as exc:
        raise GRBJP4CFMError("strict support forward closure failed") from exc
    if (
        base_forward.hook_exact_bytes is not True
        or len(base_forward.z_id) != len(labels)
    ):
        raise GRBJP4CFMError("strict support hook/row binding drift")
    direction_response = np.einsum(
        "rij,nj->nri",
        _directions(ground).astype(np.float64),
        np.asarray(base_forward.hidden, dtype=np.float64),
    ).astype(np.float32)
    backend = _TorchSupportBackend(
        base_forward,
        np.ascontiguousarray(direction_response),
        labels,
        tokens,
    )
    return _fit_from_backend(
        backend=backend,
        ground=ground,
        lock=lock,
        registered_old_classes=registered_old_classes,
        registered_new_classes=registered_new_classes,
        checkpoint_weight=base_weight,
        checkpoint_sha256=checkpoint_sha256,
    )


def _fit_cfm_from_taps_impl(
    *,
    base_z_id: np.ndarray,
    hidden: np.ndarray,
    pre_relu: np.ndarray,
    support_labels: Sequence[str],
    support_physical_tokens: Sequence[str],
    registered_old_classes: Sequence[str],
    registered_new_classes: Sequence[str],
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    checkpoint_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
    ground_equation_enabled: bool,
) -> CFMFitState:
    """Fit from a byte-bound Phase1-held tap archive, without query access.

    ``hidden`` is upstream of ``joint_proj.0`` and therefore invariant to the
    four-coefficient update.  Every snapshot recomputes
    ``pre_relu(theta)``, its exact ReLU active mask, normalized ``z_id`` and
    the corresponding Jacobian; this is not a global linearized feature
    approximation.
    """
    z = _finite_rows(base_z_id, Z_DIM, "tap base_z_id")
    h = _finite_rows(hidden, HIDDEN_DIM, "tap hidden")
    pre = _finite_rows(pre_relu, Z_DIM, "tap pre_relu")
    labels = _typed_tokens(support_labels, name="tap support labels")
    tokens = _typed_tokens(
        support_physical_tokens,
        name="tap support physical tokens",
        unique=True,
    )
    if (
        len(z) != len(h)
        or len(z) != len(pre)
        or len(z) != len(labels)
        or len(z) != len(tokens)
    ):
        raise GRBJP4CFMError("tap archive row binding drift")
    recomputed = np.maximum(pre, np.float32(0.0)).astype(
        np.float32, copy=False
    )
    if (
        not np.array_equal(z, recomputed)
        or z.tobytes() != np.ascontiguousarray(recomputed).tobytes()
    ):
        raise GRBJP4CFMError(
            "tap z_id is not byte-exact ReLU(pre_relu)"
        )
    weight = _weight_array(checkpoint_weight)
    base_forward = StrictForward(
        np.ascontiguousarray(z),
        np.ascontiguousarray(h),
        np.ascontiguousarray(pre),
        True,
        None,
        "phase1_held_byte_bound_tap_archive",
    )
    response = np.einsum(
        "rij,nj->nri",
        _directions(ground).astype(np.float64),
        h.astype(np.float64),
    ).astype(np.float32)
    backend = _TorchSupportBackend(
        base_forward,
        np.ascontiguousarray(response),
        labels,
        tokens,
    )
    return _fit_from_backend(
        backend=backend,
        ground=ground,
        lock=lock,
        registered_old_classes=registered_old_classes,
        registered_new_classes=registered_new_classes,
        checkpoint_weight=weight,
        checkpoint_sha256=checkpoint_sha256,
        ground_equation_enabled=ground_equation_enabled,
    )


def fit_cfm_from_taps(
    *,
    base_z_id: np.ndarray,
    hidden: np.ndarray,
    pre_relu: np.ndarray,
    support_labels: Sequence[str],
    support_physical_tokens: Sequence[str],
    registered_old_classes: Sequence[str],
    registered_new_classes: Sequence[str],
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    checkpoint_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
) -> CFMFitState:
    """Formal local held-tap fit with the frozen ground equation enabled."""
    return _fit_cfm_from_taps_impl(
        base_z_id=base_z_id,
        hidden=hidden,
        pre_relu=pre_relu,
        support_labels=support_labels,
        support_physical_tokens=support_physical_tokens,
        registered_old_classes=registered_old_classes,
        registered_new_classes=registered_new_classes,
        ground=ground,
        lock=lock,
        checkpoint_weight=checkpoint_weight,
        checkpoint_sha256=checkpoint_sha256,
        ground_equation_enabled=True,
    )


def fit_cfm_ground_off_falsifier_from_taps(
    *,
    base_z_id: np.ndarray,
    hidden: np.ndarray,
    pre_relu: np.ndarray,
    support_labels: Sequence[str],
    support_physical_tokens: Sequence[str],
    registered_old_classes: Sequence[str],
    registered_new_classes: Sequence[str],
    ground: GroundCFMInput,
    lock: CFMMethodLock,
    checkpoint_weight: torch.Tensor | np.ndarray,
    checkpoint_sha256: str,
) -> CFMFitState:
    """Held-only pseudo-arm with exactly zero ``G/b_g`` ground weight.

    This entry is intentionally tap-only, carries a non-formal claim scope,
    and cannot be reached from :func:`fit_cfm_from_support_iq`.
    """
    return _fit_cfm_from_taps_impl(
        base_z_id=base_z_id,
        hidden=hidden,
        pre_relu=pre_relu,
        support_labels=support_labels,
        support_physical_tokens=support_physical_tokens,
        registered_old_classes=registered_old_classes,
        registered_new_classes=registered_new_classes,
        ground=ground,
        lock=lock,
        checkpoint_weight=checkpoint_weight,
        checkpoint_sha256=checkpoint_sha256,
        ground_equation_enabled=False,
    )


def merge_into_joint_proj(
    linear_or_model: Any,
    *,
    state: CFMFitState,
    ground: GroundCFMInput,
    lock: CFMMethodLock,
) -> dict[str, Any]:
    """Merge the decoded INT8 theta into the real ``joint_proj.0`` layer."""
    if type(state) is not CFMFitState:
        raise GRBJP4CFMError("merge requires exact typed CFM fit state")
    state.__post_init__()
    ground.__post_init__()
    lock.__post_init__()
    linear = (
        linear_or_model
        if isinstance(linear_or_model, torch.nn.Linear)
        else _joint_proj_linear(linear_or_model)
    )
    if (
        tuple(linear.weight.shape) != (Z_DIM, HIDDEN_DIM)
        or linear.weight.dtype != torch.float32
    ):
        raise GRBJP4CFMError("merge joint_proj.0 contract drift")
    if (
        state.ground_digest != ground.digest
        or state.lock_digest != lock.digest
        or state.checkpoint_sha256 != ground.checkpoint_sha256
    ):
        raise GRBJP4CFMError("merge state/ground/lock binding drift")
    before = _weight_array(linear.weight)
    if (
        _array_sha(before) != state.joint_weight_sha256_before
        or _array_sha(before) != ground.joint_weight_sha256
    ):
        raise GRBJP4CFMError("merge requires exact frozen base weight")
    delta = _delta_weight(state.theta(), ground)
    with torch.no_grad():
        linear.weight.copy_(
            torch.as_tensor(
                before + delta,
                dtype=linear.weight.dtype,
                device=linear.weight.device,
            )
        )
    after = _array_sha(_weight_array(linear.weight))
    if after != state.joint_weight_semantic_sha256:
        raise GRBJP4CFMError("decoded INT8 merge semantic hash drift")
    return {
        "schema": SCHEMA,
        "candidate": CANDIDATE,
        "theta_code_sha256": _array_sha(state.theta_codes),
        "theta_scale_fp16_hex": state.theta_scale.tobytes().hex(),
        "joint_weight_sha256_before": state.joint_weight_sha256_before,
        "joint_weight_sha256_after": after,
        "same_theta_bytes_reusable_by_M_DA_and_M_DA92": True,
        "query_adapter_extra_mac": 0,
    }


def serialize_cfm_fit_state(state: CFMFitState) -> bytes:
    if type(state) is not CFMFitState:
        raise GRBJP4CFMError("serializer requires exact CFMFitState")
    state.__post_init__()
    payload = {
        "schema": FIT_STATE_WIRE_SCHEMA,
        "state": state._receipt_payload(),
        "receipt_sha256": state.receipt_sha256,
    }
    raw = _canonical_json(payload)
    return (
        b"GRBCFM2\0"
        + struct.pack(">I", len(raw))
        + raw
        + hashlib.sha256(raw).digest()
    )


def deserialize_cfm_fit_state(
    wire: bytes,
    *,
    expected_ground_digest: str,
    expected_lock_digest: str,
    expected_checkpoint_sha256: str,
    expected_joint_weight_sha256_before: str,
) -> CFMFitState:
    if (
        type(wire) is not bytes
        or len(wire) < 8 + 4 + 32
        or len(wire) > 256 * 1024
        or wire[:8] != b"GRBCFM2\0"
    ):
        raise GRBJP4CFMError("CFM fit-state wire type/magic/size drift")
    size = struct.unpack(">I", wire[8:12])[0]
    if size != len(wire) - 12 - 32:
        raise GRBJP4CFMError("CFM fit-state wire length drift")
    raw = wire[12 : 12 + size]
    trailer = wire[12 + size :]
    if hashlib.sha256(raw).digest() != trailer:
        raise GRBJP4CFMError("CFM fit-state wire digest drift")
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GRBJP4CFMError("CFM fit-state wire JSON invalid") from exc
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "state", "receipt_sha256"}
        or payload["schema"] != FIT_STATE_WIRE_SCHEMA
        or type(payload["state"]) is not dict
        or _canonical_json(payload) != raw
    ):
        raise GRBJP4CFMError("CFM fit-state canonical schema drift")
    item = payload["state"]
    required = {
        "schema",
        "theta_codes_hex",
        "theta_scale_hex",
        "k_shot",
        "ground_digest",
        "lock_digest",
        "checkpoint_sha256",
        "joint_weight_sha256_before",
        "joint_weight_semantic_sha256",
        "fit_receipt",
    }
    if (
        set(item) != required
        or item["schema"] != SCHEMA
        or type(item["theta_codes_hex"]) is not str
        or len(item["theta_codes_hex"]) != RANK * 2
        or type(item["theta_scale_hex"]) is not str
        or len(item["theta_scale_hex"]) != 4
        or type(item["fit_receipt"]) is not dict
    ):
        raise GRBJP4CFMError("CFM fit-state inner schema drift")
    try:
        codes = np.frombuffer(
            bytes.fromhex(item["theta_codes_hex"]), dtype=np.int8
        ).copy()
        scale = (
            np.frombuffer(
                bytes.fromhex(item["theta_scale_hex"]), dtype="<f2"
            )
            .copy()
            .reshape(())
        )
    except (ValueError, TypeError) as exc:
        raise GRBJP4CFMError("CFM fit-state numeric codec drift") from exc
    state = CFMFitState(
        codes,
        scale,
        item["k_shot"],
        item["ground_digest"],
        item["lock_digest"],
        item["checkpoint_sha256"],
        item["joint_weight_sha256_before"],
        item["joint_weight_semantic_sha256"],
        item["fit_receipt"],
        payload["receipt_sha256"],
    )
    expected = {
        "ground_digest": _require_sha(
            expected_ground_digest, "expected_ground_digest"
        ),
        "lock_digest": _require_sha(
            expected_lock_digest, "expected_lock_digest"
        ),
        "checkpoint_sha256": _require_sha(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        ),
        "joint_weight_sha256_before": _require_sha(
            expected_joint_weight_sha256_before,
            "expected_joint_weight_sha256_before",
        ),
    }
    if any(getattr(state, key) != value for key, value in expected.items()):
        raise GRBJP4CFMError("CFM fit-state external replay binding drift")
    return state


def predict_frozen_queries(
    *,
    model: Any,
    query_iq: torch.Tensor,
    state: CFMFitState,
    score_function: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read-only query API.

    The adapter must already be merged.  The function supplies only normalized
    query features to a frozen head and proves that neither the merged weight
    nor the serialized fit state changed during inference.
    """
    if type(state) is not CFMFitState:
        raise GRBJP4CFMError("query API requires exact CFM fit state")
    state.__post_init__()
    if (
        not torch.is_tensor(query_iq)
        or query_iq.dtype != torch.float32
        or query_iq.ndim != 3
        or query_iq.shape[1] != 2
        or not torch.isfinite(query_iq).all()
        or not callable(score_function)
    ):
        raise GRBJP4CFMError("query API typed input drift")
    linear = _joint_proj_linear(model)
    weight_before = _array_sha(_weight_array(linear.weight))
    state_before = serialize_cfm_fit_state(state)
    if weight_before != state.joint_weight_semantic_sha256:
        raise GRBJP4CFMError("query API requires the frozen merged weight")
    try:
        forward: StrictForward = _legacy_strict_zid_with_hook(model, query_iq)
    except Exception as exc:
        raise GRBJP4CFMError("strict query forward failed") from exc
    z_id = _unit_rows(forward.z_id, "query z_id")
    logits = np.asarray(score_function(z_id))
    if (
        logits.dtype not in (np.float32, np.float64)
        or logits.ndim != 2
        or len(logits) != len(query_iq)
        or logits.shape[1] < 2
        or not np.isfinite(logits).all()
    ):
        raise GRBJP4CFMError("frozen query head returned invalid logits")
    weight_after = _array_sha(_weight_array(linear.weight))
    state_after = serialize_cfm_fit_state(state)
    if weight_after != weight_before or state_after != state_before:
        raise GRBJP4CFMError("query API mutated frozen predictor state")
    prediction = np.argmax(logits, axis=1).astype(np.int64)
    return prediction, {
        "schema": QUERY_RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "query_row_count": len(query_iq),
        "query_rows_used_for_fit": 0,
        "per_sample_all_registered_classes": True,
        "query_truth_access": False,
        "query_role_access": False,
        "cross_query_reassignment": False,
        "state_updated": False,
        "joint_weight_sha256": weight_after,
        "fit_state_receipt_sha256": state.receipt_sha256,
        "prediction_sha256": _array_sha(prediction),
    }


__all__ = [
    "ACTIVE_SET_STEPS",
    "AffineSupportBackend",
    "CANDIDATE",
    "CFMFitState",
    "CFMMethodLock",
    "FIT_STATE_WIRE_SCHEMA",
    "FoldClosure",
    "GROUND_SCHEMA",
    "GRBJP4CFMError",
    "GroundCFMInput",
    "HELD_FALSIFIER_SCOPE",
    "HIDDEN_DIM",
    "JP4_WIRE_LIMIT_BYTES",
    "K_VALUES",
    "LOCK_SCHEMA",
    "MAX_STATE_BYTES",
    "RANK",
    "SCHEMA",
    "SupportSnapshot",
    "Z_DIM",
    "deserialize_cfm_fit_state",
    "fit_cfm_from_precomputed",
    "fit_cfm_from_support_iq",
    "fit_cfm_from_taps",
    "fit_cfm_ground_off_falsifier_from_taps",
    "merge_into_joint_proj",
    "predict_frozen_queries",
    "serialize_cfm_fit_state",
    "strict_physical_loo_fold",
    "validate_fold_closure",
]
