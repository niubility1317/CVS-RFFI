"""NEXT-R4 FA-RDCE3: a compact, support-only Fisher-anchored RDCE state.

The module deliberately has no Phase1 row, LOO, query, truth, role, or
accuracy input.  Phase1 may call :func:`build_fa_rdce3_phase1_asset` only
after it has already reduced its data to class aggregates.  The resulting
wire contains quantized aggregate-only statistics and a fixed RDCE basis.

Phase2 has one fitting entry point, :func:`fit_fa_rdce3_reg0`.  It accepts
balanced old-class ``REG0`` support and estimates one shared three-dimensional
offset.  ``REG1`` has no fitting entry point: it must reuse the exact state
returned for ``REG0`` through :func:`reuse_fa_rdce3_state_for_reg1`.

The representation path is fixed:

``R0 canonical unit z -> z - B.T a -> one fixed S_kappa -> signed-unit R1``.

There is intentionally no ReLU or second normalisation after the final signed
unit output.  The persistent Phase2 numerical state is exactly three binary16
numbers (six bytes).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import struct

import numpy as np


CANDIDATE_ID = "NEXT-R4-FA-RDCE3-R1"
Z_DIM = 160
RANK = 3
ALLOWED_K = (1, 5)
INT8_MAX = 127

ASSET_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.phase1_asset.v1"
ASSET_WIRE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.phase1_asset_wire.v1"
RUNTIME_BINDING_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.runtime_binding.v1"
RUNTIME_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.runtime.v1"
RUNTIME_WIRE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.runtime_wire.v1"
RESOURCE_RECEIPT_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.resource_receipt.v1"
REUSE_RECEIPT_SCHEMA = "cvs.stage2.next_r4.fa_rdce3.reg1_reuse_receipt.v1"

R0_REPRESENTATION_RULE = "d106_canonical_normalized_relu_zid160"
R1_REPRESENTATION_RULE = "fa_rdce3_once_rdce_signed_unit_zid160"
FIT_MODE_FISHER_CLOSED_FORM = "FISHER_CLOSED_FORM"
FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE = "POSTERIOR_ZERO_FIXED_RDCE"
RUNTIME_WIRE_MAGIC = b"CVSNR4FA\x00\x01"


class NextR4FARDCE3Error(ValueError):
    """Raised when the compact FA-RDCE3 contract drifts."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return dict(_array_receipt(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise NextR4FARDCE3Error(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise NextR4FARDCE3Error(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise NextR4FARDCE3Error(f"{name} must be a non-empty exact string")
    return value


def _registry(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_require_text(item, name=name) for item in value)
    if len(result) < 2 or len(set(result)) != len(result):
        raise NextR4FARDCE3Error(f"{name} must be a unique registry of at least two classes")
    return result


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return MappingProxyType(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_mapping(item)
            if isinstance(item, Mapping)
            else tuple(item)
            if isinstance(item, list | tuple)
            else item
            for key, item in value.items()
        }
    )


def _strict_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    name: str,
    finite: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR4FARDCE3Error(f"{name} must be a numpy array")
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise NextR4FARDCE3Error(f"{name} dtype/shape drift")
    if finite and not np.isfinite(array).all():
        raise NextR4FARDCE3Error(f"{name} must be finite")
    return _readonly(array, dtype)


def _decoded_basis(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    raw = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    if raw.shape != (RANK, Z_DIM) or not np.isfinite(raw).all():
        raise NextR4FARDCE3Error("decoded RDCE basis shape/finite drift")
    gram = raw @ raw.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if (
        not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= 1.0e-9
    ):
        raise NextR4FARDCE3Error("quantized RDCE basis lost rank")
    closed = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T @ raw
    if not np.allclose(closed @ closed.T, np.eye(RANK), rtol=0.0, atol=2.0e-10):
        raise NextR4FARDCE3Error("quantized RDCE basis orthogonal closure drift")
    return np.ascontiguousarray(closed, dtype=np.float64)


def _decode_centers(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    value = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    if not np.isfinite(value).all():
        raise NextR4FARDCE3Error("decoded class centers are non-finite")
    return np.ascontiguousarray(value, dtype=np.float64)


def _decode_positive(codes: np.ndarray, scales: np.ndarray, *, name: str) -> np.ndarray:
    value = codes.astype(np.float64) * scales.astype(np.float64)
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise NextR4FARDCE3Error(f"decoded {name} must be finite and positive")
    return np.ascontiguousarray(value, dtype=np.float64)


def _decode_kappa(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    value = codes.astype(np.float64) * scales.astype(np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value >= 1.0):
        raise NextR4FARDCE3Error("decoded fixed RDCE kappa must lie in [0,1)")
    return np.ascontiguousarray(value, dtype=np.float64)


def _encode_array(value: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(value).tobytes(order="C")).decode("ascii")


def _decode_array(
    value: Any, *, dtype: np.dtype[Any], shape: tuple[int, ...], name: str
) -> np.ndarray:
    if type(value) is not str:
        raise NextR4FARDCE3Error(f"{name} wire field must be base64 text")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise NextR4FARDCE3Error(f"{name} wire base64 is invalid") from error
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise NextR4FARDCE3Error(f"{name} wire byte length drift")
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _quantize_signed_rows(value: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise NextR4FARDCE3Error(f"{name} must be a finite matrix")
    codes = np.zeros(rows.shape, dtype=np.int8)
    scales = np.zeros(rows.shape[0], dtype=np.dtype("<f2"))
    for index, row in enumerate(rows):
        maximum = float(np.max(np.abs(row)))
        if maximum == 0.0:
            scales[index] = np.float16(1.0)
            continue
        scale = np.float16(maximum / float(INT8_MAX))
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise NextR4FARDCE3Error(f"{name} quantization scale is not representable")
        code = np.clip(np.rint(row / float(scale)), -INT8_MAX, INT8_MAX).astype(np.int8)
        if np.any(code == np.int8(-128)):
            raise NextR4FARDCE3Error(f"{name} quantization code range drift")
        codes[index] = code
        scales[index] = scale
    return codes, scales


def _quantize_positive(value: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise NextR4FARDCE3Error(f"{name} must be finite and positive")
    codes = np.full(values.shape, INT8_MAX, dtype=np.int8)
    scales = np.asarray(values / float(INT8_MAX), dtype=np.dtype("<f2"))
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise NextR4FARDCE3Error(f"{name} quantization scale is not representable")
    return codes, scales


def _quantize_kappa(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (RANK,) or not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values >= 1.0):
        raise NextR4FARDCE3Error("fixed RDCE kappa must lie in [0,1)")
    maximum = float(np.max(values))
    if maximum == 0.0:
        return np.zeros(RANK, dtype=np.int8), np.ones(RANK, dtype=np.dtype("<f2"))
    scale = np.float16(maximum / float(INT8_MAX))
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise NextR4FARDCE3Error("fixed RDCE kappa quantization scale is not representable")
    codes = np.clip(np.rint(values / float(scale)), 0, INT8_MAX).astype(np.int8)
    scales = np.full(RANK, scale, dtype=np.dtype("<f2"))
    decoded = codes.astype(np.float64) * scales.astype(np.float64)
    if np.any(decoded >= 1.0):
        raise NextR4FARDCE3Error("fixed RDCE kappa cannot be represented below one")
    return codes, scales


def _raw_float_matrix(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR4FARDCE3Error(f"{name} must be a numpy float32 array")
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise NextR4FARDCE3Error(f"{name} must be finite float32 with the frozen shape")
    return np.ascontiguousarray(array, dtype=np.float64)


def _raw_float_vector(value: Any, *, shape: tuple[int, ...], name: str, positive: bool) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR4FARDCE3Error(f"{name} must be a numpy float32 array")
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise NextR4FARDCE3Error(f"{name} must be finite float32 with the frozen shape")
    converted = np.ascontiguousarray(array, dtype=np.float64)
    if positive and np.any(converted <= 0.0):
        raise NextR4FARDCE3Error(f"{name} must be positive")
    return converted


def _asset_payload(asset: "FARDCE3Phase1Asset") -> dict[str, Any]:
    return {
        "schema": ASSET_WIRE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "asset_schema": ASSET_SCHEMA,
        "old_classes": list(asset.old_classes),
        "aggregate_samples_per_class": list(asset.aggregate_samples_per_class),
        "lineage": {
            "checkpoint_sha256": asset.checkpoint_sha256,
            "phase1_bundle_sha256": asset.phase1_bundle_sha256,
            "phase1_aggregate_receipt_sha256": asset.phase1_aggregate_receipt_sha256,
            "method_lock_sha256": asset.method_lock_sha256,
        },
        "aggregate_only": True,
        "phase1_source_rows_retained": False,
        "phase1_per_row_features_retained": False,
        "phase1_loo_required": False,
        "arrays": {
            "centers_codes_qint8": _encode_array(asset.centers_codes_qint8),
            "centers_scales_fp16": _encode_array(asset.centers_scales_fp16),
            "fisher_codes_qint8": _encode_array(asset.fisher_codes_qint8),
            "fisher_scales_fp16": _encode_array(asset.fisher_scales_fp16),
            "residual_variance_codes_qint8": _encode_array(asset.residual_variance_codes_qint8),
            "residual_variance_scales_fp16": _encode_array(asset.residual_variance_scales_fp16),
            "rho_codes_qint8": _encode_array(asset.rho_codes_qint8),
            "rho_scales_fp16": _encode_array(asset.rho_scales_fp16),
            "kappa_codes_qint8": _encode_array(asset.kappa_codes_qint8),
            "kappa_scales_fp16": _encode_array(asset.kappa_scales_fp16),
            "basis_codes_qint8": _encode_array(asset.basis_codes_qint8),
            "basis_scales_fp16": _encode_array(asset.basis_scales_fp16),
        },
    }


@dataclass(frozen=True, slots=True)
class FARDCE3Phase1Asset:
    """Aggregate-only Phase1 asset jointly bound to one checkpoint.

    The asset intentionally carries per-class *counts* but never physical IDs,
    source rows, source features, LOO folds, or a reversible source sidecar.
    """

    old_classes: tuple[str, ...]
    aggregate_samples_per_class: tuple[int, ...]
    centers_codes_qint8: np.ndarray
    centers_scales_fp16: np.ndarray
    fisher_codes_qint8: np.ndarray
    fisher_scales_fp16: np.ndarray
    residual_variance_codes_qint8: np.ndarray
    residual_variance_scales_fp16: np.ndarray
    rho_codes_qint8: np.ndarray
    rho_scales_fp16: np.ndarray
    kappa_codes_qint8: np.ndarray
    kappa_scales_fp16: np.ndarray
    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    checkpoint_sha256: str
    phase1_bundle_sha256: str
    phase1_aggregate_receipt_sha256: str
    method_lock_sha256: str
    schema: str = ASSET_SCHEMA
    asset_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != ASSET_SCHEMA:
            raise NextR4FARDCE3Error("FA-RDCE3 Phase1 asset schema drift")
        classes = _registry(self.old_classes, name="old_classes")
        counts = tuple(self.aggregate_samples_per_class)
        if (
            len(counts) != len(classes)
            or any(type(count) is not int or count < 2 for count in counts)
        ):
            raise NextR4FARDCE3Error("aggregate_samples_per_class must prove multi-sample aggregation")
        object.__setattr__(self, "old_classes", classes)
        object.__setattr__(self, "aggregate_samples_per_class", counts)
        for field_name in (
            "checkpoint_sha256",
            "phase1_bundle_sha256",
            "phase1_aggregate_receipt_sha256",
            "method_lock_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), name=field_name),
            )
        arrays = {
            "centers_codes_qint8": _strict_array(
                self.centers_codes_qint8,
                dtype=np.dtype(np.int8),
                shape=(len(classes), RANK),
                name="centers_codes_qint8",
            ),
            "centers_scales_fp16": _strict_array(
                self.centers_scales_fp16,
                dtype=np.dtype("<f2"),
                shape=(len(classes),),
                name="centers_scales_fp16",
                finite=True,
            ),
            "fisher_codes_qint8": _strict_array(
                self.fisher_codes_qint8,
                dtype=np.dtype(np.int8),
                shape=(RANK,),
                name="fisher_codes_qint8",
            ),
            "fisher_scales_fp16": _strict_array(
                self.fisher_scales_fp16,
                dtype=np.dtype("<f2"),
                shape=(RANK,),
                name="fisher_scales_fp16",
                finite=True,
            ),
            "residual_variance_codes_qint8": _strict_array(
                self.residual_variance_codes_qint8,
                dtype=np.dtype(np.int8),
                shape=(RANK,),
                name="residual_variance_codes_qint8",
            ),
            "residual_variance_scales_fp16": _strict_array(
                self.residual_variance_scales_fp16,
                dtype=np.dtype("<f2"),
                shape=(RANK,),
                name="residual_variance_scales_fp16",
                finite=True,
            ),
            "rho_codes_qint8": _strict_array(
                self.rho_codes_qint8,
                dtype=np.dtype(np.int8),
                shape=(1,),
                name="rho_codes_qint8",
            ),
            "rho_scales_fp16": _strict_array(
                self.rho_scales_fp16,
                dtype=np.dtype("<f2"),
                shape=(1,),
                name="rho_scales_fp16",
                finite=True,
            ),
            "kappa_codes_qint8": _strict_array(
                self.kappa_codes_qint8,
                dtype=np.dtype(np.int8),
                shape=(RANK,),
                name="kappa_codes_qint8",
            ),
            "kappa_scales_fp16": _strict_array(
                self.kappa_scales_fp16,
                dtype=np.dtype("<f2"),
                shape=(RANK,),
                name="kappa_scales_fp16",
                finite=True,
            ),
            "basis_codes_qint8": _strict_array(
                self.basis_codes_qint8,
                dtype=np.dtype(np.int8),
                shape=(RANK, Z_DIM),
                name="basis_codes_qint8",
            ),
            "basis_scales_fp16": _strict_array(
                self.basis_scales_fp16,
                dtype=np.dtype("<f2"),
                shape=(RANK,),
                name="basis_scales_fp16",
                finite=True,
            ),
        }
        for field_name, array in arrays.items():
            if field_name.endswith("codes_qint8") and np.any(array == np.int8(-128)):
                raise NextR4FARDCE3Error(f"{field_name} cannot use the -128 code")
            if field_name.endswith("scales_fp16") and np.any(array <= 0.0):
                raise NextR4FARDCE3Error(f"{field_name} must be positive")
            object.__setattr__(self, field_name, array)
        for field_name in (
            "fisher_codes_qint8",
            "residual_variance_codes_qint8",
            "rho_codes_qint8",
        ):
            if np.any(getattr(self, field_name) <= 0):
                raise NextR4FARDCE3Error(f"{field_name} must encode a positive aggregate")
        if np.any(self.kappa_codes_qint8 < 0):
            raise NextR4FARDCE3Error("kappa_codes_qint8 cannot be negative")
        _decoded_basis(self.basis_codes_qint8, self.basis_scales_fp16)
        _decode_centers(self.centers_codes_qint8, self.centers_scales_fp16)
        _decode_positive(self.fisher_codes_qint8, self.fisher_scales_fp16, name="Fisher precision")
        _decode_positive(
            self.residual_variance_codes_qint8,
            self.residual_variance_scales_fp16,
            name="residual variance",
        )
        _decode_positive(self.rho_codes_qint8, self.rho_scales_fp16, name="Fisher radius")
        _decode_kappa(self.kappa_codes_qint8, self.kappa_scales_fp16)
        object.__setattr__(self, "asset_sha256", _sha256_bytes(_canonical_bytes(_asset_payload(self))))

    @property
    def binding_sha256(self) -> str:
        """The immutable Phase1/checkpoint binding digest."""

        return self.asset_sha256

    @property
    def numeric_payload_bytes(self) -> int:
        return int(
            sum(
                getattr(self, field_name).nbytes
                for field_name in (
                    "centers_codes_qint8",
                    "centers_scales_fp16",
                    "fisher_codes_qint8",
                    "fisher_scales_fp16",
                    "residual_variance_codes_qint8",
                    "residual_variance_scales_fp16",
                    "rho_codes_qint8",
                    "rho_scales_fp16",
                    "kappa_codes_qint8",
                    "kappa_scales_fp16",
                    "basis_codes_qint8",
                    "basis_scales_fp16",
                )
            )
        )

    @property
    def wire_mapping(self) -> Mapping[str, Any]:
        return _freeze_mapping(_asset_payload(self))


def build_fa_rdce3_phase1_asset(
    *,
    old_classes: Sequence[str],
    aggregate_samples_per_class: Sequence[int],
    class_centers_3d: np.ndarray,
    fisher_precision_3d: np.ndarray,
    residual_variance_3d: np.ndarray,
    fisher_radius: np.ndarray,
    rdce_kappa_3d: np.ndarray,
    basis_3x160: np.ndarray,
    checkpoint_sha256: str,
    phase1_bundle_sha256: str,
    phase1_aggregate_receipt_sha256: str,
    method_lock_sha256: str,
) -> FARDCE3Phase1Asset:
    """Build a sealed asset from pre-aggregated Phase1 values only.

    This function purposefully has no API for Phase1 physical rows, IDs,
    labels, source caches, or leave-one-out folds.  Callers must aggregate
    those inputs before entering this narrow construction boundary.
    """

    classes = _registry(old_classes, name="old_classes")
    centers = _raw_float_matrix(
        class_centers_3d,
        shape=(len(classes), RANK),
        name="class_centers_3d",
    )
    fisher = _raw_float_vector(
        fisher_precision_3d,
        shape=(RANK,),
        name="fisher_precision_3d",
        positive=True,
    )
    variance = _raw_float_vector(
        residual_variance_3d,
        shape=(RANK,),
        name="residual_variance_3d",
        positive=True,
    )
    radius = _raw_float_vector(
        fisher_radius,
        shape=(1,),
        name="fisher_radius",
        positive=True,
    )
    kappa = _raw_float_vector(
        rdce_kappa_3d,
        shape=(RANK,),
        name="rdce_kappa_3d",
        positive=False,
    )
    if np.any(kappa < 0.0) or np.any(kappa >= 1.0):
        raise NextR4FARDCE3Error("rdce_kappa_3d must lie in [0,1)")
    basis = _raw_float_matrix(
        basis_3x160,
        shape=(RANK, Z_DIM),
        name="basis_3x160",
    )
    center_codes, center_scales = _quantize_signed_rows(centers, name="class centers")
    fisher_codes, fisher_scales = _quantize_positive(fisher, name="Fisher precision")
    variance_codes, variance_scales = _quantize_positive(variance, name="residual variance")
    radius_codes, radius_scales = _quantize_positive(radius, name="Fisher radius")
    kappa_codes, kappa_scales = _quantize_kappa(kappa)
    basis_codes, basis_scales = _quantize_signed_rows(basis, name="RDCE basis")
    return FARDCE3Phase1Asset(
        old_classes=classes,
        aggregate_samples_per_class=tuple(aggregate_samples_per_class),
        centers_codes_qint8=center_codes,
        centers_scales_fp16=center_scales,
        fisher_codes_qint8=fisher_codes,
        fisher_scales_fp16=fisher_scales,
        residual_variance_codes_qint8=variance_codes,
        residual_variance_scales_fp16=variance_scales,
        rho_codes_qint8=radius_codes,
        rho_scales_fp16=radius_scales,
        kappa_codes_qint8=kappa_codes,
        kappa_scales_fp16=kappa_scales,
        basis_codes_qint8=basis_codes,
        basis_scales_fp16=basis_scales,
        checkpoint_sha256=checkpoint_sha256,
        phase1_bundle_sha256=phase1_bundle_sha256,
        phase1_aggregate_receipt_sha256=phase1_aggregate_receipt_sha256,
        method_lock_sha256=method_lock_sha256,
    )


def decode_fa_rdce3_basis(asset: FARDCE3Phase1Asset) -> np.ndarray:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("basis decode requires an exact FA-RDCE3 asset")
    return _readonly(_decoded_basis(asset.basis_codes_qint8, asset.basis_scales_fp16), np.dtype(np.float64))


def decode_fa_rdce3_centers(asset: FARDCE3Phase1Asset) -> np.ndarray:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("center decode requires an exact FA-RDCE3 asset")
    return _readonly(_decode_centers(asset.centers_codes_qint8, asset.centers_scales_fp16), np.dtype(np.float64))


def decode_fa_rdce3_fisher_precision(asset: FARDCE3Phase1Asset) -> np.ndarray:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("Fisher decode requires an exact FA-RDCE3 asset")
    return _readonly(
        _decode_positive(asset.fisher_codes_qint8, asset.fisher_scales_fp16, name="Fisher precision"),
        np.dtype(np.float64),
    )


def decode_fa_rdce3_residual_variance(asset: FARDCE3Phase1Asset) -> np.ndarray:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("variance decode requires an exact FA-RDCE3 asset")
    return _readonly(
        _decode_positive(
            asset.residual_variance_codes_qint8,
            asset.residual_variance_scales_fp16,
            name="residual variance",
        ),
        np.dtype(np.float64),
    )


def decode_fa_rdce3_radius(asset: FARDCE3Phase1Asset) -> float:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("radius decode requires an exact FA-RDCE3 asset")
    return float(
        _decode_positive(asset.rho_codes_qint8, asset.rho_scales_fp16, name="Fisher radius")[0]
    )


def decode_fa_rdce3_kappa(asset: FARDCE3Phase1Asset) -> np.ndarray:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("kappa decode requires an exact FA-RDCE3 asset")
    return _readonly(_decode_kappa(asset.kappa_codes_qint8, asset.kappa_scales_fp16), np.dtype(np.float64))


def serialize_fa_rdce3_phase1_asset(asset: FARDCE3Phase1Asset) -> bytes:
    if type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("asset serialization requires an exact FA-RDCE3 asset")
    return _canonical_bytes(_asset_payload(asset))


def deserialize_fa_rdce3_phase1_asset(value: bytes) -> FARDCE3Phase1Asset:
    if not isinstance(value, bytes):
        raise NextR4FARDCE3Error("FA-RDCE3 asset wire must be bytes")
    try:
        payload = json.loads(value.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR4FARDCE3Error("FA-RDCE3 asset wire is not canonical ASCII JSON") from error
    if _canonical_bytes(payload) != value:
        raise NextR4FARDCE3Error("FA-RDCE3 asset wire is not canonical")
    required = {
        "schema",
        "candidate_id",
        "asset_schema",
        "old_classes",
        "aggregate_samples_per_class",
        "lineage",
        "aggregate_only",
        "phase1_source_rows_retained",
        "phase1_per_row_features_retained",
        "phase1_loo_required",
        "arrays",
    }
    if type(payload) is not dict or set(payload) != required:
        raise NextR4FARDCE3Error("FA-RDCE3 asset wire schema fields drift")
    if (
        payload["schema"] != ASSET_WIRE_SCHEMA
        or payload["candidate_id"] != CANDIDATE_ID
        or payload["asset_schema"] != ASSET_SCHEMA
        or payload["aggregate_only"] is not True
        or payload["phase1_source_rows_retained"] is not False
        or payload["phase1_per_row_features_retained"] is not False
        or payload["phase1_loo_required"] is not False
    ):
        raise NextR4FARDCE3Error("FA-RDCE3 asset wire contract drift")
    if type(payload["lineage"]) is not dict or set(payload["lineage"]) != {
        "checkpoint_sha256",
        "phase1_bundle_sha256",
        "phase1_aggregate_receipt_sha256",
        "method_lock_sha256",
    }:
        raise NextR4FARDCE3Error("FA-RDCE3 asset lineage fields drift")
    arrays = payload["arrays"]
    specs = {
        "centers_codes_qint8": (np.dtype(np.int8), (len(payload["old_classes"]), RANK)),
        "centers_scales_fp16": (np.dtype("<f2"), (len(payload["old_classes"]),)),
        "fisher_codes_qint8": (np.dtype(np.int8), (RANK,)),
        "fisher_scales_fp16": (np.dtype("<f2"), (RANK,)),
        "residual_variance_codes_qint8": (np.dtype(np.int8), (RANK,)),
        "residual_variance_scales_fp16": (np.dtype("<f2"), (RANK,)),
        "rho_codes_qint8": (np.dtype(np.int8), (1,)),
        "rho_scales_fp16": (np.dtype("<f2"), (1,)),
        "kappa_codes_qint8": (np.dtype(np.int8), (RANK,)),
        "kappa_scales_fp16": (np.dtype("<f2"), (RANK,)),
        "basis_codes_qint8": (np.dtype(np.int8), (RANK, Z_DIM)),
        "basis_scales_fp16": (np.dtype("<f2"), (RANK,)),
    }
    if type(arrays) is not dict or set(arrays) != set(specs):
        raise NextR4FARDCE3Error("FA-RDCE3 asset array fields drift")
    decoded = {
        name: _decode_array(arrays[name], dtype=dtype, shape=shape, name=name)
        for name, (dtype, shape) in specs.items()
    }
    asset = FARDCE3Phase1Asset(
        old_classes=tuple(payload["old_classes"]),
        aggregate_samples_per_class=tuple(payload["aggregate_samples_per_class"]),
        checkpoint_sha256=payload["lineage"]["checkpoint_sha256"],
        phase1_bundle_sha256=payload["lineage"]["phase1_bundle_sha256"],
        phase1_aggregate_receipt_sha256=payload["lineage"]["phase1_aggregate_receipt_sha256"],
        method_lock_sha256=payload["lineage"]["method_lock_sha256"],
        **decoded,
    )
    if serialize_fa_rdce3_phase1_asset(asset) != value:
        raise NextR4FARDCE3Error("FA-RDCE3 asset wire roundtrip drift")
    return asset


def roundtrip_fa_rdce3_phase1_asset(asset: FARDCE3Phase1Asset) -> FARDCE3Phase1Asset:
    recovered = deserialize_fa_rdce3_phase1_asset(serialize_fa_rdce3_phase1_asset(asset))
    if recovered.asset_sha256 != asset.asset_sha256:
        raise NextR4FARDCE3Error("FA-RDCE3 asset binding drift after roundtrip")
    return recovered


@dataclass(frozen=True, slots=True)
class FARDCE3RuntimeBinding:
    """Validated-once REG0 support binding; it deliberately has no query field."""

    checkpoint_sha256: str
    capsule_id: str
    split_id: str
    row_id: str
    seed: int
    active_k: int
    old_classes: tuple[str, ...]
    support_physical_root_sha256: str
    support_authority_sha256: str
    protocol_schema: str = "p2_min_v1"
    phase2_data_status: str = "VALIDATED_ONCE"
    schema: str = RUNTIME_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RUNTIME_BINDING_SCHEMA
            or self.protocol_schema != "p2_min_v1"
            or self.phase2_data_status != "VALIDATED_ONCE"
            or type(self.seed) is not int
            or self.seed < 0
            or self.active_k not in ALLOWED_K
        ):
            raise NextR4FARDCE3Error("FA-RDCE3 runtime binding lifecycle drift")
        for field_name in (
            "checkpoint_sha256",
            "capsule_id",
            "split_id",
            "support_physical_root_sha256",
            "support_authority_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(self, "row_id", _require_text(self.row_id, name="row_id"))
        object.__setattr__(self, "old_classes", _registry(self.old_classes, name="old_classes"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "row_id": self.row_id,
            "seed": self.seed,
            "active_k": self.active_k,
            "old_classes": list(self.old_classes),
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "support_authority_sha256": self.support_authority_sha256,
            "protocol_schema": self.protocol_schema,
            "phase2_data_status": self.phase2_data_status,
        }

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.as_dict()))

    @classmethod
    def from_mapping(cls, value: Any) -> "FARDCE3RuntimeBinding":
        required = {
            "schema",
            "checkpoint_sha256",
            "capsule_id",
            "split_id",
            "row_id",
            "seed",
            "active_k",
            "old_classes",
            "support_physical_root_sha256",
            "support_authority_sha256",
            "protocol_schema",
            "phase2_data_status",
        }
        if type(value) is not dict or set(value) != required:
            raise NextR4FARDCE3Error("FA-RDCE3 runtime binding fields drift")
        return cls(
            checkpoint_sha256=value["checkpoint_sha256"],
            capsule_id=value["capsule_id"],
            split_id=value["split_id"],
            row_id=value["row_id"],
            seed=value["seed"],
            active_k=value["active_k"],
            old_classes=tuple(value["old_classes"]),
            support_physical_root_sha256=value["support_physical_root_sha256"],
            support_authority_sha256=value["support_authority_sha256"],
            protocol_schema=value["protocol_schema"],
            phase2_data_status=value["phase2_data_status"],
            schema=value["schema"],
        )


def _runtime_payload(
    *,
    asset: FARDCE3Phase1Asset,
    binding: FARDCE3RuntimeBinding,
    a_fp16: np.ndarray,
    fit_mode: str,
    support_content_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "asset_sha256": asset.asset_sha256,
        "runtime_binding": binding.as_dict(),
        "runtime_binding_sha256": binding.binding_sha256,
        "a_fp16": dict(_array_receipt(a_fp16)),
        "fit_mode": fit_mode,
        "support_content_sha256": support_content_sha256,
        "fit_input": "REG0_old_class_support_only",
        "phase1_source_rows_used": 0,
        "phase1_loo_folds_used": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_gradient_calls": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_batch_dependency": False,
        "target_optimizer_steps": 0,
        "dynamic_numeric_bytes": RANK * np.dtype("<f2").itemsize,
    }


@dataclass(frozen=True, slots=True)
class FARDCE3RuntimeState:
    """Immutable six-byte shared shift fitted once from REG0 support."""

    asset: FARDCE3Phase1Asset
    binding: FARDCE3RuntimeBinding
    a_fp16: np.ndarray
    fit_mode: str
    support_content_sha256: str
    runtime_receipt_sha256: str
    schema: str = RUNTIME_SCHEMA

    def __post_init__(self) -> None:
        if type(self.asset) is not FARDCE3Phase1Asset or type(self.binding) is not FARDCE3RuntimeBinding:
            raise NextR4FARDCE3Error("FA-RDCE3 state requires exact asset and binding types")
        if self.schema != RUNTIME_SCHEMA or self.binding.checkpoint_sha256 != self.asset.checkpoint_sha256:
            raise NextR4FARDCE3Error("FA-RDCE3 state/checkpoint binding drift")
        if self.binding.old_classes != self.asset.old_classes:
            raise NextR4FARDCE3Error("FA-RDCE3 state must use exactly the REG0 old-class registry")
        if self.fit_mode not in {FIT_MODE_FISHER_CLOSED_FORM, FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE}:
            raise NextR4FARDCE3Error("FA-RDCE3 fit mode drift")
        a = _strict_array(
            self.a_fp16,
            dtype=np.dtype("<f2"),
            shape=(RANK,),
            name="a_fp16",
            finite=True,
        )
        a64 = a.astype(np.float64)
        if self.fit_mode == FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE and np.any(a64 != 0.0):
            raise NextR4FARDCE3Error("posterior-zero mode must retain an exact zero shift")
        if self.fit_mode == FIT_MODE_FISHER_CLOSED_FORM and not np.any(a64 != 0.0):
            raise NextR4FARDCE3Error("closed-form mode cannot hide a zero posterior")
        fisher = decode_fa_rdce3_fisher_precision(self.asset).astype(np.float64)
        radius = decode_fa_rdce3_radius(self.asset)
        fisher_norm = math.sqrt(float(np.sum(fisher * np.square(a64))))
        if not math.isfinite(fisher_norm) or fisher_norm > radius + 1.0e-7 * max(1.0, radius):
            raise NextR4FARDCE3Error("FA-RDCE3 FP16 shift escapes its Fisher radius")
        support_sha = _require_sha256(self.support_content_sha256, name="support_content_sha256")
        receipt = _require_sha256(self.runtime_receipt_sha256, name="runtime_receipt_sha256")
        payload = _runtime_payload(
            asset=self.asset,
            binding=self.binding,
            a_fp16=a,
            fit_mode=self.fit_mode,
            support_content_sha256=support_sha,
        )
        if _sha256_bytes(_canonical_bytes(payload)) != receipt:
            raise NextR4FARDCE3Error("FA-RDCE3 runtime receipt drift")
        object.__setattr__(self, "a_fp16", a)
        object.__setattr__(self, "support_content_sha256", support_sha)
        object.__setattr__(self, "runtime_receipt_sha256", receipt)

    @property
    def a(self) -> np.ndarray:
        return _readonly(self.a_fp16.astype(np.float64), np.dtype(np.float64))

    @property
    def dynamic_numeric_bytes(self) -> int:
        return int(self.a_fp16.nbytes)

    @property
    def runtime_payload(self) -> Mapping[str, Any]:
        return _freeze_mapping(
            _runtime_payload(
                asset=self.asset,
                binding=self.binding,
                a_fp16=self.a_fp16,
                fit_mode=self.fit_mode,
                support_content_sha256=self.support_content_sha256,
            )
        )


def _r0_rows(value: Any, *, name: str, count: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR4FARDCE3Error(f"{name} must be a numpy float32 array")
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or rows.shape[0] < 1
        or (count is not None and rows.shape[0] != count)
        or not np.isfinite(rows).all()
    ):
        raise NextR4FARDCE3Error(f"{name} must be finite float32 [N,{Z_DIM}]")
    if np.any(rows < 0.0):
        raise NextR4FARDCE3Error(f"{name} must be canonical non-negative R0 rows")
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR4FARDCE3Error(f"{name} must already be canonical unit R0 rows")
    return np.ascontiguousarray(rows, dtype=np.float64)


def _support_content_sha(asset: FARDCE3Phase1Asset, support: Mapping[str, np.ndarray]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": "cvs.stage2.next_r4.fa_rdce3.reg0_support_content.v1",
                "old_classes": list(asset.old_classes),
                "arrays": {
                    class_handle: dict(_array_receipt(support[class_handle]))
                    for class_handle in asset.old_classes
                },
            }
        )
    )


def _quantize_shift_toward_zero(value: np.ndarray) -> np.ndarray:
    """Seal a projected float64 shift in six bytes without radius overshoot."""

    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != (RANK,) or not np.isfinite(raw).all():
        raise NextR4FARDCE3Error("FA-RDCE3 posterior shift is non-finite")
    encoded = np.asarray(raw, dtype=np.dtype("<f2"))
    if not np.isfinite(encoded).all():
        raise NextR4FARDCE3Error("FA-RDCE3 posterior shift cannot fit the FP16 wire")
    for index, item in enumerate(encoded):
        if abs(float(item)) > abs(float(raw[index])):
            encoded[index] = np.nextafter(item, np.float16(0.0), dtype=np.float16)
    if np.any(raw != 0.0) and not np.any(encoded != 0.0):
        raise NextR4FARDCE3Error("non-zero Fisher posterior underflowed the six-byte state")
    return _readonly(encoded, np.dtype("<f2"))


def _make_runtime_state(
    *,
    asset: FARDCE3Phase1Asset,
    binding: FARDCE3RuntimeBinding,
    a_fp16: np.ndarray,
    fit_mode: str,
    support_content_sha256: str,
) -> FARDCE3RuntimeState:
    payload = _runtime_payload(
        asset=asset,
        binding=binding,
        a_fp16=a_fp16,
        fit_mode=fit_mode,
        support_content_sha256=support_content_sha256,
    )
    return FARDCE3RuntimeState(
        asset=asset,
        binding=binding,
        a_fp16=a_fp16,
        fit_mode=fit_mode,
        support_content_sha256=support_content_sha256,
        runtime_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
    )


def fit_fa_rdce3_reg0(
    asset: FARDCE3Phase1Asset,
    support_by_class: Mapping[str, np.ndarray],
    *,
    binding: FARDCE3RuntimeBinding,
) -> FARDCE3RuntimeState:
    """Fit the one shared Fisher-anchored shift using only REG0 old support.

    All old classes must supply exactly ``K`` canonical R0 rows.  The formula
    is class-equally weighted because every class has the same required K:

    ``a=(D_F + C K D_v^-1)^-1 D_v^-1 sum_c,sum_i(B z_ci-m_c)``.

    No query, support correctness, role, top-k, or optimiser input is accepted.
    """

    if type(asset) is not FARDCE3Phase1Asset or type(binding) is not FARDCE3RuntimeBinding:
        raise NextR4FARDCE3Error("FA-RDCE3 REG0 fit requires exact asset and binding types")
    if binding.checkpoint_sha256 != asset.checkpoint_sha256 or binding.old_classes != asset.old_classes:
        raise NextR4FARDCE3Error("FA-RDCE3 REG0 fit asset/binding drift")
    if not isinstance(support_by_class, Mapping) or set(support_by_class) != set(asset.old_classes):
        raise NextR4FARDCE3Error("FA-RDCE3 REG0 fit requires exactly the old-class support registry")
    support: dict[str, np.ndarray] = {
        class_handle: _r0_rows(
            support_by_class[class_handle],
            name=f"support_by_class[{class_handle!r}]",
            count=binding.active_k,
        )
        for class_handle in asset.old_classes
    }
    basis = decode_fa_rdce3_basis(asset).astype(np.float64)
    centers = decode_fa_rdce3_centers(asset).astype(np.float64)
    fisher = decode_fa_rdce3_fisher_precision(asset).astype(np.float64)
    variance = decode_fa_rdce3_residual_variance(asset).astype(np.float64)
    residual_sum = np.zeros(RANK, dtype=np.float64)
    for class_index, class_handle in enumerate(asset.old_classes):
        projected = support[class_handle] @ basis.T
        residual_sum += np.sum(projected - centers[class_index][None, :], axis=0)
    c_count = len(asset.old_classes)
    precision = fisher + float(c_count * binding.active_k) / variance
    raw_a = (residual_sum / variance) / precision
    if not np.isfinite(raw_a).all() or np.any(precision <= 0.0):
        raise NextR4FARDCE3Error("FA-RDCE3 Fisher posterior is non-finite or undefined")
    fisher_norm_sq = float(np.sum(fisher * np.square(raw_a)))
    if not math.isfinite(fisher_norm_sq) or fisher_norm_sq < 0.0:
        raise NextR4FARDCE3Error("FA-RDCE3 Fisher direction is undefined")
    # There is no K=1 heuristic.  The only deterministic degeneracy is the
    # exact zero posterior / undefined zero Fisher direction, which leaves the
    # fixed RDCE transform intact.  This branch uses no correctness signal.
    if fisher_norm_sq == 0.0:
        a_fp16 = _readonly(np.zeros(RANK, dtype=np.dtype("<f2")), np.dtype("<f2"))
        fit_mode = FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE
    else:
        raw_norm = math.sqrt(fisher_norm_sq)
        radius = decode_fa_rdce3_radius(asset)
        if raw_norm > radius:
            raw_a = raw_a * (radius / raw_norm)
        a_fp16 = _quantize_shift_toward_zero(raw_a)
        fit_mode = FIT_MODE_FISHER_CLOSED_FORM
    return _make_runtime_state(
        asset=asset,
        binding=binding,
        a_fp16=a_fp16,
        fit_mode=fit_mode,
        support_content_sha256=_support_content_sha(asset, support),
    )


def reuse_fa_rdce3_state_for_reg1(
    state: FARDCE3RuntimeState, *, registered_classes: Sequence[str]
) -> FARDCE3RuntimeState:
    """Return the exact REG0 state object for REG1; no refit is possible."""

    if type(state) is not FARDCE3RuntimeState:
        raise NextR4FARDCE3Error("REG1 reuse requires an exact FA-RDCE3 state")
    registry = _registry(registered_classes, name="registered_classes")
    if not set(state.asset.old_classes).issubset(set(registry)) or len(registry) <= len(state.asset.old_classes):
        raise NextR4FARDCE3Error("REG1 must retain every REG0 old class and append new classes")
    return state


def fa_rdce3_reg1_reuse_receipt(
    state: FARDCE3RuntimeState, *, registered_classes: Sequence[str]
) -> Mapping[str, Any]:
    """Audit receipt proving that REG1 receives the very same six-byte state."""

    reused = reuse_fa_rdce3_state_for_reg1(state, registered_classes=registered_classes)
    payload = {
        "schema": REUSE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "da1_reg0_state_sha256": state.runtime_receipt_sha256,
        "da1_reg1_state_sha256": reused.runtime_receipt_sha256,
        "same_state_object": reused is state,
        "a_fp16_sha256": _array_receipt(state.a_fp16)["sha256"],
        "a_fp16_bytes": state.dynamic_numeric_bytes,
        "reg0_fit_calls": 1,
        "reg1_fit_calls": 0,
        "new_class_support_rows_used_for_da": 0,
        "bitwise_state_reuse": reused.a_fp16.tobytes(order="C") == state.a_fp16.tobytes(order="C"),
    }
    payload["reuse_receipt_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return _freeze_mapping(payload)


def transform_fa_rdce3_r1(state: FARDCE3RuntimeState, r0_zid160: np.ndarray) -> np.ndarray:
    """Apply ``z-B.T a`` followed by one fixed RDCE transform and one unit map.

    The low-rank expression is algebraically identical to the stated sequence
    while using two rank-three projections (``2*3*160`` MAC per row):

    ``R1 = unit(z - B.T [a + (1-sqrt(1-kappa))*(Bz-a)])``.
    """

    if type(state) is not FARDCE3RuntimeState:
        raise NextR4FARDCE3Error("R1 transform requires an exact FA-RDCE3 state")
    r0 = _r0_rows(r0_zid160, name="r0_zid160")
    basis = decode_fa_rdce3_basis(state.asset).astype(np.float64)
    kappa = decode_fa_rdce3_kappa(state.asset).astype(np.float64)
    a = state.a_fp16.astype(np.float64)
    projected = r0 @ basis.T
    fixed_rdce_coeff = 1.0 - np.sqrt(1.0 - kappa)
    # This is the one prescribed R0->subtract(B.T a)->S_kappa route.  No ReLU
    # is applied, and the only normalisation is the final signed-unit output.
    transformed = r0 - (a[None, :] + fixed_rdce_coeff[None, :] * (projected - a[None, :])) @ basis
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise NextR4FARDCE3Error("FA-RDCE3 R1 transform produced an undefined signed direction")
    result = transformed / norms
    if not np.isfinite(result).all():
        raise NextR4FARDCE3Error("FA-RDCE3 R1 transform produced non-finite rows")
    return _readonly(result, np.dtype(np.float32))


def fa_rdce3_resource_receipt(state: FARDCE3RuntimeState) -> Mapping[str, Any]:
    """Return the fixed resource accounting for this exact support state."""

    if type(state) is not FARDCE3RuntimeState:
        raise NextR4FARDCE3Error("resource receipt requires an exact FA-RDCE3 state")
    class_count = len(state.asset.old_classes)
    fit_mac = class_count * state.binding.active_k * RANK * Z_DIM
    payload: dict[str, Any] = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "runtime_state_sha256": state.runtime_receipt_sha256,
        "asset_sha256": state.asset.asset_sha256,
        "active_k": state.binding.active_k,
        "registered_old_class_count": class_count,
        "dynamic_state_storage": "fp16_shared_a[3]",
        "dynamic_numeric_bytes": state.dynamic_numeric_bytes,
        "fit_mac_formula": "C*K*3*160",
        "fit_mac": fit_mac,
        "fixed_rdce_query_mac_formula": "2*3*160",
        "fixed_rdce_query_mac": 2 * RANK * Z_DIM,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_gradient_calls": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "phase1_source_rows_used": 0,
        "phase1_loo_folds_used": 0,
        "target_optimizer_steps": 0,
    }
    payload["resource_receipt_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return _freeze_mapping(payload)


def serialize_fa_rdce3_runtime_state(state: FARDCE3RuntimeState) -> bytes:
    """Serialize only the state header and its six dynamic numerical bytes."""

    if type(state) is not FARDCE3RuntimeState:
        raise NextR4FARDCE3Error("runtime serialization requires an exact FA-RDCE3 state")
    header = {
        "schema": RUNTIME_WIRE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "asset_sha256": state.asset.asset_sha256,
        "runtime": dict(state.runtime_payload),
        "runtime_receipt_sha256": state.runtime_receipt_sha256,
    }
    encoded_header = _canonical_bytes(header)
    return (
        RUNTIME_WIRE_MAGIC
        + struct.pack(">I", len(encoded_header))
        + encoded_header
        + np.ascontiguousarray(state.a_fp16).tobytes(order="C")
    )


def deserialize_fa_rdce3_runtime_state(
    value: bytes, *, asset: FARDCE3Phase1Asset
) -> FARDCE3RuntimeState:
    """Recover a state against its separately sealed Phase1 asset."""

    if not isinstance(value, bytes) or type(asset) is not FARDCE3Phase1Asset:
        raise NextR4FARDCE3Error("runtime deserialization requires bytes and an exact asset")
    minimum = len(RUNTIME_WIRE_MAGIC) + 4 + RANK * np.dtype("<f2").itemsize
    if len(value) < minimum or not value.startswith(RUNTIME_WIRE_MAGIC):
        raise NextR4FARDCE3Error("FA-RDCE3 runtime wire magic/length drift")
    offset = len(RUNTIME_WIRE_MAGIC)
    header_length = struct.unpack(">I", value[offset : offset + 4])[0]
    offset += 4
    if header_length <= 0 or offset + header_length + RANK * np.dtype("<f2").itemsize != len(value):
        raise NextR4FARDCE3Error("FA-RDCE3 runtime wire length/trailing drift")
    raw_header = value[offset : offset + header_length]
    try:
        header = json.loads(raw_header.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR4FARDCE3Error("FA-RDCE3 runtime header is not canonical ASCII JSON") from error
    if _canonical_bytes(header) != raw_header:
        raise NextR4FARDCE3Error("FA-RDCE3 runtime header is not canonical")
    if type(header) is not dict or set(header) != {
        "schema",
        "candidate_id",
        "asset_sha256",
        "runtime",
        "runtime_receipt_sha256",
    }:
        raise NextR4FARDCE3Error("FA-RDCE3 runtime header fields drift")
    if (
        header["schema"] != RUNTIME_WIRE_SCHEMA
        or header["candidate_id"] != CANDIDATE_ID
        or header["asset_sha256"] != asset.asset_sha256
        or type(header["runtime"]) is not dict
    ):
        raise NextR4FARDCE3Error("FA-RDCE3 runtime header binding drift")
    a = np.frombuffer(value[offset + header_length :], dtype=np.dtype("<f2")).copy()
    runtime = header["runtime"]
    required = {
        "schema",
        "candidate_id",
        "asset_sha256",
        "runtime_binding",
        "runtime_binding_sha256",
        "a_fp16",
        "fit_mode",
        "support_content_sha256",
        "fit_input",
        "phase1_source_rows_used",
        "phase1_loo_folds_used",
        "query_rows_used_for_fit",
        "query_state_updates",
        "query_selection_count",
        "query_gradient_calls",
        "query_truth_access",
        "query_role_access",
        "query_batch_dependency",
        "target_optimizer_steps",
        "dynamic_numeric_bytes",
    }
    if set(runtime) != required:
        raise NextR4FARDCE3Error("FA-RDCE3 runtime payload fields drift")
    binding = FARDCE3RuntimeBinding.from_mapping(runtime["runtime_binding"])
    if runtime["runtime_binding_sha256"] != binding.binding_sha256:
        raise NextR4FARDCE3Error("FA-RDCE3 runtime binding digest drift")
    if runtime["a_fp16"] != dict(_array_receipt(a)):
        raise NextR4FARDCE3Error("FA-RDCE3 runtime six-byte receipt drift")
    state = FARDCE3RuntimeState(
        asset=asset,
        binding=binding,
        a_fp16=a,
        fit_mode=runtime["fit_mode"],
        support_content_sha256=runtime["support_content_sha256"],
        runtime_receipt_sha256=header["runtime_receipt_sha256"],
    )
    if (
        _canonical_bytes(dict(state.runtime_payload)) != _canonical_bytes(runtime)
        or serialize_fa_rdce3_runtime_state(state) != value
    ):
        raise NextR4FARDCE3Error("FA-RDCE3 runtime wire roundtrip drift")
    return state


def roundtrip_fa_rdce3_runtime_state(state: FARDCE3RuntimeState) -> FARDCE3RuntimeState:
    recovered = deserialize_fa_rdce3_runtime_state(
        serialize_fa_rdce3_runtime_state(state), asset=state.asset
    )
    if recovered.runtime_receipt_sha256 != state.runtime_receipt_sha256:
        raise NextR4FARDCE3Error("FA-RDCE3 runtime state digest drift after roundtrip")
    return recovered


__all__ = [
    "ALLOWED_K",
    "ASSET_SCHEMA",
    "CANDIDATE_ID",
    "FARDCE3Phase1Asset",
    "FARDCE3RuntimeBinding",
    "FARDCE3RuntimeState",
    "FIT_MODE_FISHER_CLOSED_FORM",
    "FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE",
    "NextR4FARDCE3Error",
    "R0_REPRESENTATION_RULE",
    "R1_REPRESENTATION_RULE",
    "RANK",
    "Z_DIM",
    "build_fa_rdce3_phase1_asset",
    "decode_fa_rdce3_basis",
    "decode_fa_rdce3_centers",
    "decode_fa_rdce3_fisher_precision",
    "decode_fa_rdce3_kappa",
    "decode_fa_rdce3_radius",
    "decode_fa_rdce3_residual_variance",
    "deserialize_fa_rdce3_phase1_asset",
    "deserialize_fa_rdce3_runtime_state",
    "fa_rdce3_reg1_reuse_receipt",
    "fa_rdce3_resource_receipt",
    "fit_fa_rdce3_reg0",
    "reuse_fa_rdce3_state_for_reg1",
    "roundtrip_fa_rdce3_phase1_asset",
    "roundtrip_fa_rdce3_runtime_state",
    "serialize_fa_rdce3_phase1_asset",
    "serialize_fa_rdce3_runtime_state",
    "transform_fa_rdce3_r1",
]
