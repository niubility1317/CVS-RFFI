"""Typed support-fitted SCXMAP receiver-domain correction.

The Phase1 lock contains only quantized, class-shared cross-map arrays and
quantized old-class ground anchors.  Stage2 fits one non-negative scalar from
target-old support.  A query contributes only its own fixed ``z_id`` and
``z_dom`` rows and cannot update the fitted state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


Z_DIM = 160
CONTEXT_DIM = 4
SCHEMA = "cvs.phase2.scxmap_transform.v1"
ALLOWED_K = (1, 5, 10)
WIRE_MAGIC = b"CVS-SCXMAP-RUNTIME\0"


class SCXMapError(ValueError):
    """Raised when an SCXMAP lock, fit, or transform drifts."""


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise SCXMapError(f"{name} must be a lowercase SHA256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise SCXMapError(f"{name} must be hexadecimal") from exc
    return value


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _finite_rows(value: np.ndarray, width: int, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != width
        or not np.isfinite(rows).all()
    ):
        raise SCXMapError(f"{name} must be finite float32 [N,{width}]")
    return rows


def _normalize(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise SCXMapError("SCXMAP normalization requires finite nonzero rows")
    return np.asarray(rows / norms, dtype=np.float32)


def _decode(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scales.astype(np.float32).reshape(
        (-1,) + (1,) * (codes.ndim - 1)
    )


def _quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim not in (1, 2) or not np.isfinite(rows).all():
        raise SCXMapError("SCXMAP quantization input drift")
    if rows.ndim == 1:
        scales = np.maximum(
            np.abs(rows) / 127.0, np.finfo(np.float16).tiny
        ).astype(np.float16)
        codes = np.sign(rows).astype(np.int8) * np.int8(127)
        replay = codes.astype(np.float32) * scales.astype(np.float32)
        return codes, scales, float(np.max(np.abs(replay - rows)))
    matrix = rows
    peak = np.max(np.abs(matrix), axis=1)
    scales = np.maximum(peak / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    codes = np.rint(matrix / scales.astype(np.float32)[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    replay = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    error = float(np.max(np.abs(replay - matrix)))
    return codes, scales, error


@dataclass(frozen=True, slots=True)
class Phase1SCXMapLock:
    ground_classes: tuple[str, ...]
    zdom_center_qint8: np.ndarray
    zdom_center_scales_fp16: np.ndarray
    zdom_scale_qint8: np.ndarray
    zdom_scale_scales_fp16: np.ndarray
    receiver_projection_qint8: np.ndarray
    receiver_projection_scales_fp16: np.ndarray
    context_to_shift_qint8: np.ndarray
    context_to_shift_scales_fp16: np.ndarray
    zid_basis_qint8: np.ndarray
    zid_basis_scales_fp16: np.ndarray
    ground_anchor_qint8: np.ndarray
    ground_anchor_scales_fp16: np.ndarray
    ridge_per_row: float
    shrink_tau: float
    beta_max: float
    source_receipt_sha256: str
    schema: str = SCHEMA
    lock_digest: str = ""

    def __post_init__(self) -> None:
        classes = tuple(self.ground_classes)
        arrays = (
            ("zdom_center_qint8", self.zdom_center_qint8, np.int8, (Z_DIM,)),
            (
                "zdom_center_scales_fp16",
                self.zdom_center_scales_fp16,
                np.float16,
                (Z_DIM,),
            ),
            ("zdom_scale_qint8", self.zdom_scale_qint8, np.int8, (Z_DIM,)),
            (
                "zdom_scale_scales_fp16",
                self.zdom_scale_scales_fp16,
                np.float16,
                (Z_DIM,),
            ),
            (
                "receiver_projection_qint8",
                self.receiver_projection_qint8,
                np.int8,
                (CONTEXT_DIM, Z_DIM),
            ),
            (
                "receiver_projection_scales_fp16",
                self.receiver_projection_scales_fp16,
                np.float16,
                (CONTEXT_DIM,),
            ),
            (
                "context_to_shift_qint8",
                self.context_to_shift_qint8,
                np.int8,
                (CONTEXT_DIM, CONTEXT_DIM),
            ),
            (
                "context_to_shift_scales_fp16",
                self.context_to_shift_scales_fp16,
                np.float16,
                (CONTEXT_DIM,),
            ),
            ("zid_basis_qint8", self.zid_basis_qint8, np.int8, (CONTEXT_DIM, Z_DIM)),
            (
                "zid_basis_scales_fp16",
                self.zid_basis_scales_fp16,
                np.float16,
                (CONTEXT_DIM,),
            ),
            (
                "ground_anchor_qint8",
                self.ground_anchor_qint8,
                np.int8,
                (len(classes), Z_DIM),
            ),
            (
                "ground_anchor_scales_fp16",
                self.ground_anchor_scales_fp16,
                np.float16,
                (len(classes),),
            ),
        )
        if (
            self.schema != SCHEMA
            or not classes
            or classes != tuple(sorted(classes))
            or len(set(classes)) != len(classes)
            or any(type(item) is not str or not item for item in classes)
        ):
            raise SCXMapError("SCXMAP Phase1 class registry drift")
        for name, value, dtype, shape in arrays:
            if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
                raise SCXMapError(f"SCXMAP lock {name} dtype/shape/finite drift")
            if dtype == np.int8 and np.any(value == np.int8(-128)):
                raise SCXMapError(f"SCXMAP lock {name} code range drift")
            if dtype == np.float16 and np.any(value <= 0.0):
                raise SCXMapError(f"SCXMAP lock {name} scale drift")
            object.__setattr__(self, name, _readonly(value, dtype))
        for value, name, allow_zero in (
            (self.ridge_per_row, "ridge_per_row", True),
            (self.shrink_tau, "shrink_tau", True),
            (self.beta_max, "beta_max", True),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise SCXMapError(f"SCXMAP {name} drift")
            if not allow_zero and value == 0.0:
                raise SCXMapError(f"SCXMAP {name} must be positive")
        _require_sha(self.source_receipt_sha256, "SCXMAP source receipt")
        expected = _lock_digest(self)
        if self.lock_digest != expected:
            raise SCXMapError("SCXMAP lock digest drift")


def _lock_payload(lock: Phase1SCXMapLock) -> dict[str, Any]:
    return {
        "schema": lock.schema,
        "ground_classes": list(lock.ground_classes),
        "arrays": {
            name: _array_receipt(getattr(lock, name))
            for name in (
                "zdom_center_qint8",
                "zdom_center_scales_fp16",
                "zdom_scale_qint8",
                "zdom_scale_scales_fp16",
                "receiver_projection_qint8",
                "receiver_projection_scales_fp16",
                "context_to_shift_qint8",
                "context_to_shift_scales_fp16",
                "zid_basis_qint8",
                "zid_basis_scales_fp16",
                "ground_anchor_qint8",
                "ground_anchor_scales_fp16",
            )
        },
        "ridge_per_row": lock.ridge_per_row,
        "shrink_tau": lock.shrink_tau,
        "beta_max": lock.beta_max,
        "source_receipt_sha256": lock.source_receipt_sha256,
    }


def _lock_digest(lock: Phase1SCXMapLock) -> str:
    return _canonical_sha(_lock_payload(lock))


def build_phase1_scxmap_lock(
    *,
    ground_classes: Sequence[str],
    zdom_center: np.ndarray,
    zdom_scale: np.ndarray,
    receiver_projection: np.ndarray,
    context_to_shift: np.ndarray,
    zid_basis: np.ndarray,
    ground_anchors: np.ndarray,
    ridge_per_row: float,
    shrink_tau: float,
    beta_max: float,
    source_receipt_sha256: str,
) -> tuple[Phase1SCXMapLock, dict[str, Any]]:
    """Quantize one precomputed Phase1 lock; this function does not fit it."""

    classes = tuple(str(value) for value in ground_classes)
    center = np.asarray(zdom_center, dtype=np.float32)
    scale = np.asarray(zdom_scale, dtype=np.float32)
    projection = np.asarray(receiver_projection, dtype=np.float32)
    cross = np.asarray(context_to_shift, dtype=np.float32)
    basis = np.asarray(zid_basis, dtype=np.float32)
    anchors = _normalize(np.asarray(ground_anchors, dtype=np.float32))
    if (
        center.shape != (Z_DIM,)
        or scale.shape != (Z_DIM,)
        or np.any(scale <= 0.0)
        or projection.shape != (CONTEXT_DIM, Z_DIM)
        or cross.shape != (CONTEXT_DIM, CONTEXT_DIM)
        or basis.shape != (CONTEXT_DIM, Z_DIM)
        or anchors.shape != (len(classes), Z_DIM)
        or not all(
            np.isfinite(value).all()
            for value in (center, scale, projection, cross, basis, anchors)
        )
    ):
        raise SCXMapError("SCXMAP Phase1 float lock input drift")
    encoded: dict[str, np.ndarray] = {}
    errors: dict[str, float] = {}
    for name, value in (
        ("zdom_center", center),
        ("zdom_scale", scale),
        ("receiver_projection", projection),
        ("context_to_shift", cross),
        ("zid_basis", basis),
        ("ground_anchor", anchors),
    ):
        codes, scales, error = _quantize_rows(value)
        encoded[name + "_qint8"] = _readonly(codes, np.int8)
        encoded[name + "_scales_fp16"] = _readonly(scales, np.float16)
        errors[name] = error
    scalar_values = {
        "ridge_per_row": float(ridge_per_row),
        "shrink_tau": float(shrink_tau),
        "beta_max": float(beta_max),
        "source_receipt_sha256": str(source_receipt_sha256),
    }
    digest = _canonical_sha(
        {
            "schema": SCHEMA,
            "ground_classes": list(classes),
            "arrays": {
                name: _array_receipt(value) for name, value in encoded.items()
            },
            **scalar_values,
        }
    )
    lock = Phase1SCXMapLock(
        ground_classes=classes,
        **encoded,
        **scalar_values,
        lock_digest=digest,
    )
    return lock, {
        "lock_digest": digest,
        "quantization_max_abs_error": errors,
        "persistent_array_bytes": int(
            sum(getattr(lock, name).nbytes for name in encoded)
        ),
    }


def _correction(lock: Phase1SCXMapLock, zdom: np.ndarray) -> np.ndarray:
    rows = _finite_rows(zdom, Z_DIM, "SCXMAP z_dom")
    center = _decode(lock.zdom_center_qint8, lock.zdom_center_scales_fp16)
    scale = _decode(lock.zdom_scale_qint8, lock.zdom_scale_scales_fp16)
    projection = _decode(
        lock.receiver_projection_qint8, lock.receiver_projection_scales_fp16
    )
    cross = _decode(
        lock.context_to_shift_qint8, lock.context_to_shift_scales_fp16
    )
    basis = _decode(lock.zid_basis_qint8, lock.zid_basis_scales_fp16)
    context = ((rows - center[None, :]) / scale[None, :]) @ projection.T
    return np.asarray((context @ cross) @ basis, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class FittedSCXMapState:
    lock_digest: str
    support_receipt_sha256: str
    beta_fp32: float
    beta_raw_fp64: float
    shrinkage_fp64: float
    k_shot: int
    old_support_rows: int
    old_class_count: int
    numerator_fp64: float
    denominator_fp64: float
    state_receipt_sha256: str
    schema: str = SCHEMA + ".state"

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA + ".state"
            or type(self.k_shot) is not int
            or self.k_shot not in ALLOWED_K
            or type(self.old_support_rows) is not int
            or self.old_support_rows < 1
            or type(self.old_class_count) is not int
            or self.old_class_count < 2
        ):
            raise SCXMapError("SCXMAP fitted state count/schema drift")
        for value in (
            self.beta_fp32,
            self.beta_raw_fp64,
            self.shrinkage_fp64,
            self.numerator_fp64,
            self.denominator_fp64,
        ):
            if not math.isfinite(float(value)):
                raise SCXMapError("SCXMAP fitted state nonfinite scalar")
        _require_sha(self.lock_digest, "SCXMAP state lock digest")
        _require_sha(self.support_receipt_sha256, "SCXMAP support receipt")
        expected = _canonical_sha(
            {
                "schema": self.schema,
                "lock_digest": self.lock_digest,
                "support_receipt_sha256": self.support_receipt_sha256,
                "beta_fp32": self.beta_fp32,
                "beta_raw_fp64": self.beta_raw_fp64,
                "shrinkage_fp64": self.shrinkage_fp64,
                "k_shot": self.k_shot,
                "old_support_rows": self.old_support_rows,
                "old_class_count": self.old_class_count,
                "numerator_fp64": self.numerator_fp64,
                "denominator_fp64": self.denominator_fp64,
            }
        )
        if self.state_receipt_sha256 != expected:
            raise SCXMapError("SCXMAP fitted state receipt drift")


def _verify_state_lock(
    lock: Phase1SCXMapLock, state: FittedSCXMapState
) -> None:
    if state.lock_digest != lock.lock_digest:
        raise SCXMapError("SCXMAP transform lock/state drift")
    expected_shrinkage = state.old_support_rows / (
        state.old_support_rows + lock.shrink_tau
    )
    if state.denominator_fp64 <= 0.0:
        raise SCXMapError("SCXMAP state denominator must be positive")
    expected_raw = max(0.0, state.numerator_fp64 / state.denominator_fp64)
    expected_beta = float(
        np.float32(min(lock.beta_max, expected_shrinkage * expected_raw))
    )
    if (
        not math.isclose(
            state.shrinkage_fp64, expected_shrinkage, rel_tol=0.0, abs_tol=1e-15
        )
        or not math.isclose(
            state.beta_raw_fp64, expected_raw, rel_tol=0.0, abs_tol=1e-15
        )
        or state.beta_fp32 != expected_beta
        or state.beta_fp32 < 0.0
        or state.beta_fp32 > lock.beta_max
        or state.old_class_count != len(lock.ground_classes)
        or state.old_support_rows != state.old_class_count * state.k_shot
    ):
        raise SCXMapError("SCXMAP fitted scalar closure drift")


def fit_scxmap_state(
    lock: Phase1SCXMapLock,
    support_zid: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    *,
    support_receipt_sha256: str,
) -> FittedSCXMapState:
    """Fit one frozen scalar using target-old support and ground anchors only."""

    lock.__post_init__()
    zid = _finite_rows(support_zid, Z_DIM, "SCXMAP support z_id")
    zdom = _finite_rows(support_zdom, Z_DIM, "SCXMAP support z_dom")
    if any(type(value) is not str or not value for value in support_labels):
        raise SCXMapError("SCXMAP support labels must be nonempty strings")
    labels = tuple(support_labels)
    if (
        len(zid) != len(zdom)
        or len(labels) != len(zid)
    ):
        raise SCXMapError("SCXMAP support alignment/receipt drift")
    _require_sha(support_receipt_sha256, "SCXMAP support receipt")
    ground_index = {value: index for index, value in enumerate(lock.ground_classes)}
    if any(value not in ground_index for value in labels):
        raise SCXMapError("SCXMAP DA fit accepts target-old support only")
    counts = {value: labels.count(value) for value in lock.ground_classes}
    if (
        set(labels) != set(lock.ground_classes)
        or len(set(counts.values())) != 1
        or next(iter(counts.values())) not in ALLOWED_K
    ):
        raise SCXMapError(
            "SCXMAP DA fit requires every ground-old class with the same legal K"
        )
    k_shot = next(iter(counts.values()))
    anchors = _decode(lock.ground_anchor_qint8, lock.ground_anchor_scales_fp16)
    anchors = _normalize(anchors)
    residual = _normalize(zid) - np.asarray(
        [anchors[ground_index[value]] for value in labels], dtype=np.float32
    )
    correction = _correction(lock, zdom)
    numerator = float(np.sum(residual.astype(np.float64) * correction.astype(np.float64)))
    correction_energy = float(
        np.sum(correction.astype(np.float64) * correction.astype(np.float64))
    )
    denominator = correction_energy + lock.ridge_per_row * len(zid)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise SCXMapError("SCXMAP support correction/ridge denominator degeneracy")
    raw = max(0.0, numerator / denominator)
    shrinkage = len(zid) / (len(zid) + lock.shrink_tau)
    beta = float(np.float32(min(lock.beta_max, shrinkage * raw)))
    payload = {
        "schema": SCHEMA + ".state",
        "lock_digest": lock.lock_digest,
        "support_receipt_sha256": support_receipt_sha256,
        "beta_fp32": beta,
        "beta_raw_fp64": raw,
        "shrinkage_fp64": shrinkage,
        "k_shot": k_shot,
        "old_support_rows": len(zid),
        "old_class_count": len(set(labels)),
        "numerator_fp64": numerator,
        "denominator_fp64": denominator,
    }
    return FittedSCXMapState(**payload, state_receipt_sha256=_canonical_sha(payload))


def transform_scxmap_rows(
    lock: Phase1SCXMapLock,
    state: FittedSCXMapState,
    zid: np.ndarray,
    zdom: np.ndarray,
    *,
    enabled: bool = True,
) -> np.ndarray:
    """Transform independent rows without reading or updating query state."""

    lock.__post_init__()
    state.__post_init__()
    identity = _finite_rows(zid, Z_DIM, "SCXMAP z_id")
    domain = _finite_rows(zdom, Z_DIM, "SCXMAP z_dom")
    if len(identity) != len(domain) or type(enabled) is not bool:
        raise SCXMapError("SCXMAP transform row alignment/config drift")
    _verify_state_lock(lock, state)
    if not enabled or state.beta_fp32 == 0.0:
        return np.array(identity, dtype=np.float32, copy=True)
    delta = float(state.beta_fp32) * _correction(lock, domain).astype(np.float64)
    active = np.linalg.norm(delta, axis=1) > 0.0
    result = np.array(identity, dtype=np.float32, copy=True)
    if np.any(active):
        corrected = identity[active].astype(np.float64) - delta[active]
        result[active] = _normalize(np.asarray(corrected, dtype=np.float32))
    return result


def serialize_scxmap_runtime_state(
    lock: Phase1SCXMapLock, state: FittedSCXMapState
) -> bytes:
    """Serialize the exact persistent SCXMAP lock/state for resource closure."""

    lock.__post_init__()
    state.__post_init__()
    _verify_state_lock(lock, state)
    array_names = (
        "zdom_center_qint8",
        "zdom_center_scales_fp16",
        "zdom_scale_qint8",
        "zdom_scale_scales_fp16",
        "receiver_projection_qint8",
        "receiver_projection_scales_fp16",
        "context_to_shift_qint8",
        "context_to_shift_scales_fp16",
        "zid_basis_qint8",
        "zid_basis_scales_fp16",
        "ground_anchor_qint8",
        "ground_anchor_scales_fp16",
    )
    records = [
        {
            "name": name,
            "dtype": getattr(lock, name).dtype.str,
            "shape": list(getattr(lock, name).shape),
            "nbytes": int(getattr(lock, name).nbytes),
        }
        for name in array_names
    ]
    header = {
        "schema": SCHEMA + ".wire.v1",
        "ground_classes": list(lock.ground_classes),
        "ridge_per_row": lock.ridge_per_row,
        "shrink_tau": lock.shrink_tau,
        "beta_max": lock.beta_max,
        "source_receipt_sha256": lock.source_receipt_sha256,
        "lock_digest": lock.lock_digest,
        "state": dataclass_to_dict(state),
        "records": records,
    }
    header_bytes = json.dumps(
        header,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return (
        WIRE_MAGIC
        + len(header_bytes).to_bytes(8, "little", signed=False)
        + header_bytes
        + b"".join(np.ascontiguousarray(getattr(lock, name)).tobytes() for name in array_names)
    )


def dataclass_to_dict(state: FittedSCXMapState) -> dict[str, Any]:
    return {
        name: getattr(state, name)
        for name in (
            "lock_digest",
            "support_receipt_sha256",
            "beta_fp32",
            "beta_raw_fp64",
            "shrinkage_fp64",
            "k_shot",
            "old_support_rows",
            "old_class_count",
            "numerator_fp64",
            "denominator_fp64",
            "state_receipt_sha256",
            "schema",
        )
    }


def audit_scxmap_resources(
    lock: Phase1SCXMapLock, state: FittedSCXMapState
) -> dict[str, Any]:
    lock.__post_init__()
    state.__post_init__()
    _verify_state_lock(lock, state)
    n = state.old_support_rows
    cross = _decode(
        lock.context_to_shift_qint8, lock.context_to_shift_scales_fp16
    ).astype(np.float64)
    effective_rank = int(np.linalg.matrix_rank(cross))
    wire_bytes = len(serialize_scxmap_runtime_state(lock, state))
    correction_mac = Z_DIM * CONTEXT_DIM + CONTEXT_DIM**2 + CONTEXT_DIM * Z_DIM
    query_elementwise = 7 * Z_DIM
    fit_elementwise = n * (query_elementwise + 4 * Z_DIM)
    return {
        "persistent_state_wire_bytes": wire_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "fit_matmul_mac": int(n * correction_mac),
        "fit_elementwise_ops": int(fit_elementwise),
        "query_matmul_mac_per_row": int(correction_mac),
        "query_elementwise_ops_per_row": int(query_elementwise),
        "optimizer_steps": 0,
        "query_rows_used_for_fit": 0,
        "effective_rank": effective_rank,
        "beta_fp32": state.beta_fp32,
        "accounting_scope": (
            "exact persistent wire bytes; matrix multiply MAC and explicit "
            "elementwise arithmetic counted separately; excludes backbone, "
            "hashing, JSON/base64, comparisons, and serialization latency"
        ),
    }


__all__ = [
    "CONTEXT_DIM",
    "FittedSCXMapState",
    "Phase1SCXMapLock",
    "SCHEMA",
    "SCXMapError",
    "Z_DIM",
    "audit_scxmap_resources",
    "build_phase1_scxmap_lock",
    "fit_scxmap_state",
    "serialize_scxmap_runtime_state",
    "transform_scxmap_rows",
]
