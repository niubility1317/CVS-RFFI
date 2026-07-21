"""Pure-z_id INT8 Student-t single-qKNN for causal Phase2 DA ablations.

The support bank and classification formula are independent of the metric.
The A-id arm supplies an exact rank-zero metric.  A future C-id adapter may
provide one immutable, class-shared, strictly positive-definite low-rank metric
without changing the bank, class scales, temperature, or Student-t formula.

This module has no ground/source, receiver, TX, old/new-role, query-fit, graph,
quota, D81, RDA, or fusion input surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


Z_DIM = 160
INT8_MAX = 127.0
EPSILON = 1.0e-12
ALLOWED_K = (1, 5, 10, 20)
MAX_METRIC_RANK = 8

LOCK_SCHEMA = "cvs.phase1.zid_student_t_qknn.lock.v1"
BANK_SCHEMA = "cvs.phase2.zid_student_t_qknn.bank.v1"
METRIC_SCHEMA = "cvs.phase2.zid_student_t_qknn.shared_psd_metric.v1"
METRIC_PROVENANCE_SCHEMA = "cvs.phase2.zid_student_t_qknn.metric_provenance.v1"
WIRE_SCHEMA = "cvs.phase2.zid_student_t_qknn.runtime_wire.v1"
WIRE_MAGIC = b"CVSZIDQKNN\x00\x01"
ALLOWED_METRIC_FIT_SCOPES = ("phase1_lodo", "target_support_only")
MAX_WIRE_BYTES = 16 * 1024 * 1024
MAX_WIRE_HEADER_BYTES = 1024 * 1024
WIRE_ARRAY_SPECS = (
    ("support_codes_qint8", np.dtype("int8"), 2),
    ("support_scales_fp16", np.dtype("<f2"), 1),
    ("class_indices_int16", np.dtype("<i2"), 1),
    ("class_scales_fp16", np.dtype("<f2"), 1),
    ("metric_basis_codes_qint8", np.dtype("int8"), 2),
    ("metric_basis_scales_fp16", np.dtype("<f2"), 1),
    ("metric_attenuation_fp16", np.dtype("<f2"), 1),
)


class ZIDStudentTQKNNError(ValueError):
    """Raised when the pure-z_id bank, metric, or lifecycle drifts."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ZIDStudentTQKNNError(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype).copy()
    contiguous.setflags(write=False)
    return contiguous


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ZIDStudentTQKNNError(f"{name} must be positive and finite")
    return result


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) < 2 or len(set(result)) != len(result) or any(not value for value in result):
        raise ZIDStudentTQKNNError("registered class registry must contain unique non-empty values")
    return result


def _finite_zid_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise ZIDStudentTQKNNError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(rows)


def normalize_zid_rows(value: np.ndarray) -> np.ndarray:
    """L2-normalize finite float32 z_id160 rows without changing their order."""

    rows = _finite_zid_rows(value, "z_id rows").astype(np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPSILON) or not np.isfinite(norms).all():
        raise ZIDStudentTQKNNError("z_id rows contain a zero-norm vector")
    return _readonly(rows / norms, np.float32)


@dataclass(frozen=True, slots=True)
class Phase1ZIDStudentTLock:
    """Exact K-specific head parameters sealed before target/query access."""

    active_k: int
    student_nu: float
    kernel_effective_dim: int
    kernel_volume_gamma: float
    shared_h0: float
    scale_prior_strength: float
    scale_min_ratio: float
    scale_max_ratio: float
    temperature: float
    phase1_lodo_receipt_sha256: str
    quantization_margin_audit_sha256: str
    schema: str = LOCK_SCHEMA

    def __post_init__(self) -> None:
        k_shot = int(self.active_k)
        effective_dim = int(self.kernel_effective_dim)
        if (
            self.schema != LOCK_SCHEMA
            or type(self.active_k) is not int
            or type(self.kernel_effective_dim) is not int
            or k_shot not in ALLOWED_K
            or k_shot != self.active_k
            or effective_dim < 1
            or effective_dim != self.kernel_effective_dim
        ):
            raise ZIDStudentTQKNNError("Phase1 lock active K/effective dimension drift")
        for value, name in (
            (self.student_nu, "student_nu"),
            (self.kernel_volume_gamma, "kernel_volume_gamma"),
            (self.shared_h0, "shared_h0"),
            (self.scale_prior_strength, "scale_prior_strength"),
            (self.scale_min_ratio, "scale_min_ratio"),
            (self.scale_max_ratio, "scale_max_ratio"),
            (self.temperature, "temperature"),
        ):
            _finite_positive(value, name)
        if float(self.scale_min_ratio) > 1.0 or float(self.scale_max_ratio) < 1.0:
            raise ZIDStudentTQKNNError("Phase1 lock scale ratio interval must contain one")
        _require_sha256(self.phase1_lodo_receipt_sha256, "phase1_lodo_receipt_sha256")
        _require_sha256(
            self.quantization_margin_audit_sha256,
            "quantization_margin_audit_sha256",
        )

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256({"schema": LOCK_SCHEMA, **asdict(self)})


@dataclass(frozen=True, slots=True)
class TypedMetricProvenanceReceipt:
    """Typed proof boundary for one externally precomputed shared metric."""

    fit_scope: str
    source_receipt_sha256: str
    query_rows_used_for_fit: int
    schema: str = METRIC_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != METRIC_PROVENANCE_SCHEMA:
            raise ZIDStudentTQKNNError("metric provenance schema drift")
        if str(self.fit_scope) not in ALLOWED_METRIC_FIT_SCOPES:
            raise ZIDStudentTQKNNError(
                "metric provenance fit scope must be Phase1 LODO or target support only"
            )
        if (
            type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
        ):
            raise ZIDStudentTQKNNError(
                "metric provenance query rows used for fit must equal zero"
            )
        _require_sha256(
            self.source_receipt_sha256,
            "metric provenance source_receipt_sha256",
        )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


def _verify_typed_metric_provenance(
    provenance: TypedMetricProvenanceReceipt,
) -> None:
    if type(provenance) is not TypedMetricProvenanceReceipt:
        raise ZIDStudentTQKNNError("metric provenance must use the exact typed receipt")
    if (
        provenance.schema != METRIC_PROVENANCE_SCHEMA
        or provenance.fit_scope not in ALLOWED_METRIC_FIT_SCOPES
    ):
        raise ZIDStudentTQKNNError("metric provenance fit scope/schema drift")
    if (
        type(provenance.query_rows_used_for_fit) is not int
        or provenance.query_rows_used_for_fit != 0
    ):
        raise ZIDStudentTQKNNError(
            "metric provenance query rows used for fit must equal zero"
        )
    _require_sha256(
        provenance.source_receipt_sha256,
        "metric provenance source_receipt_sha256",
    )


def _quantize_rows(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(normalized, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != Z_DIM or not np.isfinite(rows).all():
        raise ZIDStudentTQKNNError("normalized support must be finite [N,160]")
    codes = np.zeros(rows.shape, dtype=np.int8)
    scales = np.zeros(len(rows), dtype=np.float16)
    decoded = np.zeros(rows.shape, dtype=np.float32)
    minimum_scale = float(np.finfo(np.float16).tiny)
    for index, row in enumerate(rows):
        scale16 = np.float16(max(float(np.max(np.abs(row))) / INT8_MAX, minimum_scale))
        if not np.isfinite(scale16) or scale16 <= 0.0:
            raise ZIDStudentTQKNNError("support quantization scale overflow")
        code = np.clip(np.rint(row / float(scale16)), -127, 127).astype(np.int8)
        codes[index] = code
        scales[index] = scale16
        decoded[index] = code.astype(np.float32) * np.float32(scale16)
    return codes, scales, normalize_zid_rows(decoded)


def _quantize_basis_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(value)
    if (
        basis.dtype != np.float32
        or basis.ndim != 2
        or basis.shape[1] != Z_DIM
        or not np.isfinite(basis).all()
    ):
        raise ZIDStudentTQKNNError("metric basis must be finite float32 [rank,160]")
    if len(basis) == 0:
        return (
            np.empty((0, Z_DIM), dtype=np.int8),
            np.empty(0, dtype=np.float16),
            np.empty((0, Z_DIM), dtype=np.float32),
        )
    normalized = normalize_zid_rows(basis)
    return _quantize_rows(normalized)


def _decode_basis(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float64) * scales.astype(np.float64)[:, None]


def _metric_numeric_properties(
    codes: np.ndarray, scales: np.ndarray, attenuation: np.ndarray
) -> tuple[float, float, float]:
    rank = len(attenuation)
    if rank == 0:
        return 1.0, 1.0, 0.0
    basis = _decode_basis(codes, scales)
    weighted = np.sqrt(attenuation.astype(np.float64))[:, None] * basis
    penalty_eigenvalues = np.linalg.eigvalsh(weighted @ weighted.T)
    maximum = float(np.max(penalty_eigenvalues))
    minimum = 1.0 - maximum
    if not math.isfinite(minimum) or minimum <= 1.0e-6:
        raise ZIDStudentTQKNNError("shared low-rank metric is not strictly positive definite")
    condition = 1.0 / minimum
    update_norm = float(
        np.sqrt(np.sum(np.square(1.0 - np.sqrt(1.0 - penalty_eigenvalues))))
    )
    return minimum, condition, update_norm


def _metric_payload(
    *,
    codes: np.ndarray,
    scales: np.ndarray,
    attenuation: np.ndarray,
    source: str,
    config_lock_digest: str,
    minimum_eigenvalue: float,
    condition_number: float,
    update_norm: float,
    builder_no_fit: bool,
    provenance: TypedMetricProvenanceReceipt | None,
) -> dict[str, Any]:
    provenance_payload = None
    if provenance is not None:
        provenance_payload = {
            **asdict(provenance),
            "receipt_sha256": provenance.receipt_sha256,
        }
    return {
        "schema": METRIC_SCHEMA,
        "basis_codes_qint8": _array_receipt(codes),
        "basis_scales_fp16": _array_receipt(scales),
        "attenuation_fp16": _array_receipt(attenuation),
        "effective_rank": int(len(attenuation)),
        "source": str(source),
        "class_shared": True,
        "config_lock_digest": str(config_lock_digest),
        "minimum_eigenvalue": float(minimum_eigenvalue),
        "condition_number": float(condition_number),
        "sqrt_metric_update_frobenius_norm": float(update_norm),
        "builder_no_fit": bool(builder_no_fit),
        "typed_provenance_receipt": provenance_payload,
    }


@dataclass(frozen=True, slots=True)
class TypedSharedPSDMetric:
    """One immutable class-shared M=I-B.T diag(a) B metric."""

    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    attenuation_fp16: np.ndarray
    effective_rank: int
    source: str
    config_lock_digest: str
    minimum_eigenvalue: float
    condition_number: float
    sqrt_metric_update_frobenius_norm: float
    metric_receipt_sha256: str
    builder_no_fit: bool
    provenance: TypedMetricProvenanceReceipt | None
    class_shared: bool = True
    schema: str = METRIC_SCHEMA

    def __post_init__(self) -> None:
        rank = int(self.effective_rank)
        codes = np.asarray(self.basis_codes_qint8)
        scales = np.asarray(self.basis_scales_fp16)
        attenuation = np.asarray(self.attenuation_fp16)
        if self.provenance is not None:
            _verify_typed_metric_provenance(self.provenance)
        identity_contract = (
            rank == 0
            and self.builder_no_fit is True
            and self.provenance is None
            and self.source == "identity_rank0"
        )
        adapted_contract = (
            rank > 0
            and self.builder_no_fit is False
            and type(self.provenance) is TypedMetricProvenanceReceipt
        )
        if (
            self.schema != METRIC_SCHEMA
            or rank < 0
            or rank > MAX_METRIC_RANK
            or type(self.effective_rank) is not int
            or rank != self.effective_rank
            or codes.dtype != np.int8
            or codes.shape != (rank, Z_DIM)
            or scales.dtype != np.float16
            or scales.shape != (rank,)
            or attenuation.dtype != np.float16
            or attenuation.shape != (rank,)
            or not self.class_shared
            or type(self.class_shared) is not bool
            or not str(self.source)
            or not np.isfinite(scales).all()
            or not np.isfinite(attenuation).all()
            or (rank > 0 and np.any(attenuation <= 0.0))
            or np.any(attenuation >= 1.0)
            or (rank and np.any(scales <= 0.0))
            or np.any(codes == np.int8(-128))
            or type(self.builder_no_fit) is not bool
            or not (identity_contract or adapted_contract)
        ):
            raise ZIDStudentTQKNNError(
                "shared low-rank metric basis code range/attenuation invariant drift"
            )
        _require_sha256(self.config_lock_digest, "metric config lock digest")
        _require_sha256(self.metric_receipt_sha256, "metric receipt")
        minimum, condition, update_norm = _metric_numeric_properties(
            codes, scales, attenuation
        )
        if (
            not math.isclose(float(self.minimum_eigenvalue), minimum, rel_tol=0.0, abs_tol=1e-10)
            or not math.isclose(float(self.condition_number), condition, rel_tol=0.0, abs_tol=1e-8)
            or not math.isclose(
                float(self.sqrt_metric_update_frobenius_norm),
                update_norm,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise ZIDStudentTQKNNError("shared metric numeric audit drift")
        payload = _metric_payload(
            codes=codes,
            scales=scales,
            attenuation=attenuation,
            source=self.source,
            config_lock_digest=self.config_lock_digest,
            minimum_eigenvalue=minimum,
            condition_number=condition,
            update_norm=update_norm,
            builder_no_fit=self.builder_no_fit,
            provenance=self.provenance,
        )
        if _canonical_sha256(payload) != self.metric_receipt_sha256:
            raise ZIDStudentTQKNNError("shared metric receipt verification failed")
        object.__setattr__(self, "basis_codes_qint8", _readonly(codes, np.int8))
        object.__setattr__(self, "basis_scales_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "attenuation_fp16", _readonly(attenuation, np.float16))

    @property
    def exact_identity(self) -> bool:
        return self.effective_rank == 0


def build_typed_shared_psd_metric(
    basis_rows: np.ndarray,
    attenuation: np.ndarray,
    *,
    config: Phase1ZIDStudentTLock,
    source: str,
    provenance: TypedMetricProvenanceReceipt | None,
) -> TypedSharedPSDMetric:
    """Quantize and close a precomputed shared metric; this function does not fit it."""

    if type(config) is not Phase1ZIDStudentTLock:
        raise ZIDStudentTQKNNError("metric requires an exact Phase1 lock")
    raw_attenuation = np.asarray(attenuation)
    if (
        raw_attenuation.dtype != np.float32
        or raw_attenuation.ndim != 1
        or not np.isfinite(raw_attenuation).all()
        or (len(raw_attenuation) > 0 and np.any(raw_attenuation <= 0.0))
        or np.any(raw_attenuation >= 1.0)
    ):
        raise ZIDStudentTQKNNError("metric attenuation must be finite float32 in (0,1)")
    codes, scales, _ = _quantize_basis_rows(np.asarray(basis_rows))
    if len(codes) > MAX_METRIC_RANK:
        raise ZIDStudentTQKNNError(
            f"metric must remain low-rank with rank at most {MAX_METRIC_RANK}"
        )
    if len(codes) != len(raw_attenuation):
        raise ZIDStudentTQKNNError("metric basis/attenuation rank drift")
    if len(codes) == 0:
        if source != "identity_rank0" or provenance is not None:
            raise ZIDStudentTQKNNError(
                "rank-zero metric is reserved for the identity builder-no-fit path"
            )
        builder_no_fit = True
    else:
        if type(provenance) is not TypedMetricProvenanceReceipt:
            raise ZIDStudentTQKNNError(
                "nonidentity metric requires an exact typed provenance receipt"
            )
        _verify_typed_metric_provenance(provenance)
        builder_no_fit = False
    attenuation16 = np.asarray(raw_attenuation, dtype=np.float16)
    if (
        (len(attenuation16) > 0 and np.any(attenuation16 <= 0.0))
        or np.any(attenuation16 >= 1.0)
        or not np.isfinite(attenuation16).all()
    ):
        raise ZIDStudentTQKNNError("metric attenuation FP16 closure drift")
    minimum, condition, update_norm = _metric_numeric_properties(
        codes, scales, attenuation16
    )
    payload = _metric_payload(
        codes=codes,
        scales=scales,
        attenuation=attenuation16,
        source=source,
        config_lock_digest=config.lock_digest,
        minimum_eigenvalue=minimum,
        condition_number=condition,
        update_norm=update_norm,
        builder_no_fit=builder_no_fit,
        provenance=provenance,
    )
    return TypedSharedPSDMetric(
        basis_codes_qint8=_readonly(codes, np.int8),
        basis_scales_fp16=_readonly(scales, np.float16),
        attenuation_fp16=_readonly(attenuation16, np.float16),
        effective_rank=len(attenuation16),
        source=str(source),
        config_lock_digest=config.lock_digest,
        minimum_eigenvalue=minimum,
        condition_number=condition,
        sqrt_metric_update_frobenius_norm=update_norm,
        metric_receipt_sha256=_canonical_sha256(payload),
        builder_no_fit=builder_no_fit,
        provenance=provenance,
    )


def identity_shared_psd_metric(
    *, config: Phase1ZIDStudentTLock
) -> TypedSharedPSDMetric:
    return build_typed_shared_psd_metric(
        np.empty((0, Z_DIM), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        config=config,
        source="identity_rank0",
        provenance=None,
    )


def _bank_payload(
    *,
    classes: tuple[str, ...],
    counts: tuple[int, ...],
    codes: np.ndarray,
    scales: np.ndarray,
    class_indices: np.ndarray,
    class_scales: np.ndarray,
    config: Phase1ZIDStudentTLock,
    quantization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": BANK_SCHEMA,
        "classes": list(classes),
        "support_counts": list(counts),
        "codes_qint8": _array_receipt(codes),
        "scales_fp16": _array_receipt(scales),
        "class_indices_int16": _array_receipt(class_indices),
        "class_scales_fp16": _array_receipt(class_scales),
        "active_k": int(config.active_k),
        "config_lock_digest": config.lock_digest,
        "quantization_audit": _json_value(quantization),
        "same_formula_all_registered_classes": True,
        "query_rows_used_for_fit": 0,
    }


@dataclass(frozen=True, slots=True)
class TypedINT8ZIDSupportBank:
    classes: tuple[str, ...]
    support_counts: tuple[int, ...]
    codes_qint8: np.ndarray
    scales_fp16: np.ndarray
    class_indices_int16: np.ndarray
    class_scales_fp16: np.ndarray
    active_k: int
    config_lock_digest: str
    config: Phase1ZIDStudentTLock
    quantization_audit: Mapping[str, Any]
    bank_receipt_sha256: str
    schema: str = BANK_SCHEMA

    def __post_init__(self) -> None:
        if type(self.config) is not Phase1ZIDStudentTLock:
            raise ZIDStudentTQKNNError("support bank requires an exact Phase1 lock")
        classes = _registry(self.classes)
        counts = tuple(int(value) for value in self.support_counts)
        rows = len(classes) * int(self.active_k)
        codes = np.asarray(self.codes_qint8)
        scales = np.asarray(self.scales_fp16)
        indices = np.asarray(self.class_indices_int16)
        class_scales = np.asarray(self.class_scales_fp16)
        if (
            self.schema != BANK_SCHEMA
            or self.active_k not in ALLOWED_K
            or self.active_k != self.config.active_k
            or self.config_lock_digest != self.config.lock_digest
        ):
            raise ZIDStudentTQKNNError("support bank active K/config lock drift")
        if (
            codes.dtype != np.int8
            or codes.shape != (rows, Z_DIM)
            or scales.dtype != np.float16
            or scales.shape != (rows,)
            or indices.dtype != np.int16
            or indices.shape != (rows,)
            or class_scales.dtype != np.float16
            or class_scales.shape != (len(classes),)
            or not np.isfinite(scales).all()
            or not np.isfinite(class_scales).all()
            or np.any(scales <= 0.0)
            or np.any(class_scales <= 0.0)
            or np.any(indices < 0)
            or np.any(indices >= len(classes))
        ):
            raise ZIDStudentTQKNNError("support bank class index/array invariant drift")
        if np.any(codes == np.int8(-128)):
            raise ZIDStudentTQKNNError("support bank qint8 code range must be [-127,127]")
        actual_counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
        if (
            counts != actual_counts
            or any(value != self.active_k for value in counts)
            or len(counts) != len(classes)
        ):
            raise ZIDStudentTQKNNError("support bank balanced K/class index closure drift")
        _require_sha256(self.config_lock_digest, "support bank config lock digest")
        _require_sha256(self.bank_receipt_sha256, "support bank receipt")
        quantization = dict(self.quantization_audit)
        payload = _bank_payload(
            classes=classes,
            counts=counts,
            codes=codes,
            scales=scales,
            class_indices=indices,
            class_scales=class_scales,
            config=self.config,
            quantization=quantization,
        )
        if _canonical_sha256(payload) != self.bank_receipt_sha256:
            raise ZIDStudentTQKNNError("support bank receipt verification failed")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "support_counts", counts)
        object.__setattr__(self, "codes_qint8", _readonly(codes, np.int8))
        object.__setattr__(self, "scales_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "class_indices_int16", _readonly(indices, np.int16))
        object.__setattr__(self, "class_scales_fp16", _readonly(class_scales, np.float16))
        object.__setattr__(self, "quantization_audit", MappingProxyType(quantization))

    @property
    def support_row_count(self) -> int:
        return len(self.codes_qint8)


def _canonical_order(
    codes: np.ndarray, scales: np.ndarray, class_indices: np.ndarray
) -> np.ndarray:
    keys = [
        (
            int(class_indices[index]),
            np.ascontiguousarray(codes[index]).tobytes(),
            np.ascontiguousarray(scales[index]).tobytes(),
            index,
        )
        for index in range(len(codes))
    ]
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def _identity_class_scales(
    decoded: np.ndarray,
    class_indices: np.ndarray,
    class_count: int,
    config: Phase1ZIDStudentTLock,
) -> np.ndarray:
    if config.active_k == 1:
        return np.full(class_count, config.shared_h0, dtype=np.float64)
    values = []
    for class_index in range(class_count):
        local = decoded[class_indices == class_index].astype(np.float64)
        cosine = np.clip(local @ local.T, -1.0, 1.0)
        distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
        upper = distance[np.triu_indices(config.active_k, 1)]
        empirical = float(np.mean(upper))
        shrunk = (
            empirical + config.scale_prior_strength * config.shared_h0**2
        ) / (1.0 + config.scale_prior_strength)
        values.append(
            np.clip(
                math.sqrt(max(shrunk, EPSILON)),
                config.shared_h0 * config.scale_min_ratio,
                config.shared_h0 * config.scale_max_ratio,
            )
        )
    return np.asarray(values, dtype=np.float64)


def build_typed_zid_support_bank(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: Phase1ZIDStudentTLock,
) -> TypedINT8ZIDSupportBank:
    """Compile one balanced target support bank; no query argument exists."""

    if type(config) is not Phase1ZIDStudentTLock:
        raise ZIDStudentTQKNNError("support bank requires an exact Phase1 lock")
    normalized = normalize_zid_rows(support_zid)
    labels = tuple(str(value) for value in support_labels)
    classes = _registry(registered_classes)
    if len(labels) != len(normalized) or any(label not in classes for label in labels):
        raise ZIDStudentTQKNNError("support labels must map to a registered class")
    class_map = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
    if any(value < 1 for value in counts):
        raise ZIDStudentTQKNNError("every registered class requires target support")
    if len(set(counts)) != 1:
        raise ZIDStudentTQKNNError("formal target support must be balanced K-shot")
    if counts[0] != config.active_k:
        raise ZIDStudentTQKNNError("support count does not match the Phase1 active K lock")
    codes, scales, decoded = _quantize_rows(normalized)
    order = _canonical_order(codes, scales, indices)
    codes, scales, decoded, indices = (
        codes[order],
        scales[order],
        decoded[order],
        indices[order],
    )
    ordered = np.asarray(normalized, dtype=np.float32)[order]
    class_scales = _identity_class_scales(decoded, indices, len(classes), config)
    class_scales16 = np.asarray(class_scales, dtype=np.float16)
    reconstruction_error = np.abs(decoded.astype(np.float64) - ordered.astype(np.float64))
    reconstruction_cosine = np.sum(
        decoded.astype(np.float64) * ordered.astype(np.float64), axis=1
    )
    quantization = {
        "schema": "cvs.phase2.zid_student_t_qknn.quantization_audit.v1",
        "feature_space": "z_id160_only",
        "support_only": True,
        "single_received_observation": True,
        "support_rows": int(len(codes)),
        "class_count": int(len(classes)),
        "support_counts": list(counts),
        "per_vector_scale": True,
        "quantization_error_mean": float(np.mean(reconstruction_error)),
        "quantization_error_max": float(np.max(reconstruction_error)),
        "reconstruction_cosine_mean": float(np.mean(reconstruction_cosine)),
        "reconstruction_cosine_min": float(np.min(reconstruction_cosine)),
        "class_scale_source": (
            "phase1_locked_shared_h0"
            if config.active_k == 1
            else "identity_metric_support_only_uniform_class_formula"
        ),
        "class_count_normalization": "logsumexp_minus_log_Kc",
        "same_formula_all_registered_classes": True,
        "old_new_role_specific_scoring": False,
        "query_rows_used_for_fit": 0,
        "config_lock_digest": config.lock_digest,
    }
    payload = _bank_payload(
        classes=classes,
        counts=counts,
        codes=codes,
        scales=scales,
        class_indices=indices,
        class_scales=class_scales16,
        config=config,
        quantization=quantization,
    )
    return TypedINT8ZIDSupportBank(
        classes=classes,
        support_counts=counts,
        codes_qint8=_readonly(codes, np.int8),
        scales_fp16=_readonly(scales, np.float16),
        class_indices_int16=_readonly(indices, np.int16),
        class_scales_fp16=_readonly(class_scales16, np.float16),
        active_k=config.active_k,
        config_lock_digest=config.lock_digest,
        config=config,
        quantization_audit=quantization,
        bank_receipt_sha256=_canonical_sha256(payload),
    )


def decode_zid_support_bank(bank: TypedINT8ZIDSupportBank) -> np.ndarray:
    if type(bank) is not TypedINT8ZIDSupportBank:
        raise ZIDStudentTQKNNError("decode requires an exact typed support bank")
    raw = (
        bank.codes_qint8.astype(np.float32)
        * bank.scales_fp16.astype(np.float32)[:, None]
    )
    return normalize_zid_rows(raw)


def _verify_bank(bank: TypedINT8ZIDSupportBank) -> None:
    payload = _bank_payload(
        classes=bank.classes,
        counts=bank.support_counts,
        codes=bank.codes_qint8,
        scales=bank.scales_fp16,
        class_indices=bank.class_indices_int16,
        class_scales=bank.class_scales_fp16,
        config=bank.config,
        quantization=bank.quantization_audit,
    )
    if _canonical_sha256(payload) != bank.bank_receipt_sha256:
        raise ZIDStudentTQKNNError("support bank receipt verification failed")


def _verify_metric(metric: TypedSharedPSDMetric) -> None:
    rank = int(metric.effective_rank)
    if metric.provenance is not None:
        _verify_typed_metric_provenance(metric.provenance)
    identity_contract = (
        rank == 0
        and metric.builder_no_fit is True
        and metric.provenance is None
        and metric.source == "identity_rank0"
    )
    adapted_contract = (
        rank > 0
        and metric.builder_no_fit is False
        and type(metric.provenance) is TypedMetricProvenanceReceipt
    )
    if (
        metric.schema != METRIC_SCHEMA
        or type(metric.effective_rank) is not int
        or rank < 0
        or rank > MAX_METRIC_RANK
        or metric.basis_codes_qint8.dtype != np.int8
        or metric.basis_codes_qint8.shape != (rank, Z_DIM)
        or metric.basis_scales_fp16.dtype != np.float16
        or metric.basis_scales_fp16.shape != (rank,)
        or metric.attenuation_fp16.dtype != np.float16
        or metric.attenuation_fp16.shape != (rank,)
        or not metric.class_shared
        or type(metric.class_shared) is not bool
        or type(metric.builder_no_fit) is not bool
        or not (identity_contract or adapted_contract)
    ):
        raise ZIDStudentTQKNNError("shared low-rank metric rank/array invariant drift")
    if np.any(metric.basis_codes_qint8 == np.int8(-128)):
        raise ZIDStudentTQKNNError("shared metric qint8 code range must be [-127,127]")
    if (
        not np.isfinite(metric.basis_scales_fp16).all()
        or not np.isfinite(metric.attenuation_fp16).all()
        or (rank > 0 and np.any(metric.attenuation_fp16 <= 0.0))
        or np.any(metric.attenuation_fp16 >= 1.0)
        or (rank > 0 and np.any(metric.basis_scales_fp16 <= 0.0))
    ):
        raise ZIDStudentTQKNNError("shared metric attenuation invariant drift")
    minimum, condition, update_norm = _metric_numeric_properties(
        metric.basis_codes_qint8,
        metric.basis_scales_fp16,
        metric.attenuation_fp16,
    )
    if (
        not math.isclose(
            float(metric.minimum_eigenvalue), minimum, rel_tol=0.0, abs_tol=1e-10
        )
        or not math.isclose(
            float(metric.condition_number), condition, rel_tol=0.0, abs_tol=1e-8
        )
        or not math.isclose(
            float(metric.sqrt_metric_update_frobenius_norm),
            update_norm,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        raise ZIDStudentTQKNNError("shared metric numeric audit drift")
    payload = _metric_payload(
        codes=metric.basis_codes_qint8,
        scales=metric.basis_scales_fp16,
        attenuation=metric.attenuation_fp16,
        source=metric.source,
        config_lock_digest=metric.config_lock_digest,
        minimum_eigenvalue=minimum,
        condition_number=condition,
        update_norm=update_norm,
        builder_no_fit=metric.builder_no_fit,
        provenance=metric.provenance,
    )
    if _canonical_sha256(payload) != metric.metric_receipt_sha256:
        raise ZIDStudentTQKNNError("shared metric receipt verification failed")


def _precision_cosine(
    left: np.ndarray, right: np.ndarray, metric: TypedSharedPSDMetric
) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if metric.effective_rank == 0:
        return np.clip(left64 @ right64.T, -1.0, 1.0)
    basis = _decode_basis(metric.basis_codes_qint8, metric.basis_scales_fp16)
    attenuation = metric.attenuation_fp16.astype(np.float64)
    left_projection = left64 @ basis.T
    right_projection = right64 @ basis.T
    numerator = (
        left64 @ right64.T
        - (left_projection * attenuation) @ right_projection.T
    )
    left_quadratic = np.sum(np.square(left64), axis=1) - np.sum(
        np.square(left_projection) * attenuation, axis=1
    )
    right_quadratic = np.sum(np.square(right64), axis=1) - np.sum(
        np.square(right_projection) * attenuation, axis=1
    )
    left_norm = np.sqrt(np.maximum(left_quadratic, EPSILON))
    right_norm = np.sqrt(np.maximum(right_quadratic, EPSILON))
    return np.clip(numerator / (left_norm[:, None] * right_norm[None, :]), -1.0, 1.0)


def _score_with_support(
    *,
    support: np.ndarray,
    class_indices: np.ndarray,
    support_counts: tuple[int, ...],
    class_scales: np.ndarray,
    query: np.ndarray,
    config: Phase1ZIDStudentTLock,
    metric: TypedSharedPSDMetric,
) -> np.ndarray:
    cosine = _precision_cosine(query, support, metric)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    columns = []
    for class_index, expected in enumerate(support_counts):
        local = distance[:, class_indices == class_index]
        if local.shape[1] != expected:
            raise ZIDStudentTQKNNError("class support count drift during scoring")
        h = float(class_scales[class_index])
        kernel = (
            -config.kernel_volume_gamma
            * config.kernel_effective_dim
            * math.log(h)
            - 0.5
            * (config.student_nu + config.kernel_effective_dim)
            * np.log1p(local / (config.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        column = (
            maximum[:, 0]
            + np.log(np.sum(np.exp(kernel - maximum), axis=1))
            - math.log(expected)
        )
        columns.append(column)
    logits = np.stack(columns, axis=1)
    if not np.isfinite(logits).all():
        raise ZIDStudentTQKNNError("Student-t qKNN logits became non-finite")
    return _readonly(logits, np.float32)


def score_zid_student_t_logits(
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
    *,
    metric: TypedSharedPSDMetric,
) -> np.ndarray:
    """Score independent queries over every registered class with one formula."""

    if type(bank) is not TypedINT8ZIDSupportBank or type(metric) is not TypedSharedPSDMetric:
        raise ZIDStudentTQKNNError("scoring requires exact typed bank and metric states")
    _verify_bank(bank)
    _verify_metric(metric)
    if metric.config_lock_digest != bank.config_lock_digest:
        raise ZIDStudentTQKNNError("support bank/metric Phase1 lock drift")
    query = normalize_zid_rows(query_zid).astype(np.float64)
    support = decode_zid_support_bank(bank).astype(np.float64)
    return _score_with_support(
        support=support,
        class_indices=bank.class_indices_int16,
        support_counts=bank.support_counts,
        class_scales=bank.class_scales_fp16,
        query=query,
        config=bank.config,
        metric=metric,
    )


def softmax_probabilities(
    logits: np.ndarray, *, config: Phase1ZIDStudentTLock
) -> np.ndarray:
    scores = np.asarray(logits)
    if type(config) is not Phase1ZIDStudentTLock:
        raise ZIDStudentTQKNNError("softmax requires an exact Phase1 lock")
    temp = float(config.temperature)
    if (
        scores.dtype != np.float32
        or scores.ndim != 2
        or scores.shape[0] < 1
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
        or not math.isfinite(temp)
        or temp <= 0.0
    ):
        raise ZIDStudentTQKNNError("softmax logits/temperature drift")
    scaled = scores.astype(np.float64) / temp
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return _readonly(exp / np.sum(exp, axis=1, keepdims=True), np.float32)


def audit_int8_margin(
    bank: TypedINT8ZIDSupportBank,
    full_precision_support_zid: np.ndarray,
    support_labels: Sequence[str],
    validation_zid: np.ndarray,
    *,
    metric: TypedSharedPSDMetric,
) -> dict[str, Any]:
    """Compare FP32-support and INT8-support logits on caller-owned audit rows."""

    if metric.config_lock_digest != bank.config_lock_digest:
        raise ZIDStudentTQKNNError("margin audit bank/metric lock drift")
    support = normalize_zid_rows(full_precision_support_zid).astype(np.float64)
    validation = normalize_zid_rows(validation_zid).astype(np.float64)
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(support) or any(label not in bank.classes for label in labels):
        raise ZIDStudentTQKNNError("margin audit support labels/classes drift")
    class_map = {label: index for index, label in enumerate(bank.classes)}
    indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(bank.classes)))
    if counts != bank.support_counts:
        raise ZIDStudentTQKNNError("margin audit support count drift")
    teacher_class_scales = _identity_class_scales(
        support,
        indices,
        len(bank.classes),
        bank.config,
    )
    fp_logits = _score_with_support(
        support=support,
        class_indices=indices,
        support_counts=counts,
        class_scales=teacher_class_scales,
        query=validation,
        config=bank.config,
        metric=metric,
    ).astype(np.float64)
    int8_logits = score_zid_student_t_logits(
        bank, np.asarray(validation_zid), metric=metric
    ).astype(np.float64)
    order = np.argsort(fp_logits, axis=1, kind="stable")
    winner = order[:, -1]
    runner_up = order[:, -2]
    row = np.arange(len(fp_logits))
    fp_margin = fp_logits[row, winner] - fp_logits[row, runner_up]
    int8_teacher_margin = int8_logits[row, winner] - int8_logits[row, runner_up]
    flip = int8_teacher_margin <= 0.0
    return {
        "schema": "cvs.phase2.zid_student_t_qknn.margin_audit.v1",
        "validation_row_count": int(len(validation)),
        "logit_abs_error_mean": float(np.mean(np.abs(fp_logits - int8_logits))),
        "logit_abs_error_max": float(np.max(np.abs(fp_logits - int8_logits))),
        "top1_agreement": float(
            np.mean(np.argmax(fp_logits, axis=1) == np.argmax(int8_logits, axis=1))
        ),
        "teacher_margin_mean": float(np.mean(fp_margin)),
        "quantized_teacher_margin_mean": float(np.mean(int8_teacher_margin)),
        "margin_sign_flip_count": int(np.sum(flip)),
        "margin_sign_flip_rate": float(np.mean(flip)),
        "fp32_teacher_bandwidth_source": (
            "full_precision_support_same_class_symmetric_formula"
        ),
        "fp32_teacher_class_scales": [
            float(value) for value in teacher_class_scales
        ],
        "int8_bank_class_scales": [
            float(value) for value in bank.class_scales_fp16.astype(np.float64)
        ],
        "teacher_bank_bandwidth_abs_delta_max": float(
            np.max(
                np.abs(
                    teacher_class_scales
                    - bank.class_scales_fp16.astype(np.float64)
                )
            )
        ),
        "query_rows_used_for_fit": 0,
        "state_updates": 0,
    }


def _wire_array(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    name_bytes = name.encode("ascii")
    dtype_bytes = array.dtype.str.encode("ascii")
    payload = array.tobytes(order="C")
    return b"".join(
        [
            struct.pack("<H", len(name_bytes)),
            name_bytes,
            struct.pack("<H", len(dtype_bytes)),
            dtype_bytes,
            struct.pack("<H", array.ndim),
            struct.pack("<" + "Q" * array.ndim, *array.shape),
            struct.pack("<Q", len(payload)),
            payload,
        ]
    )


def serialize_typed_zid_runtime_state(
    bank: TypedINT8ZIDSupportBank, metric: TypedSharedPSDMetric
) -> bytes:
    if type(bank) is not TypedINT8ZIDSupportBank or type(metric) is not TypedSharedPSDMetric:
        raise ZIDStudentTQKNNError("serialization requires exact typed bank and metric")
    _verify_bank(bank)
    _verify_metric(metric)
    if metric.config_lock_digest != bank.config_lock_digest:
        raise ZIDStudentTQKNNError("serialized bank/metric lock drift")
    provenance_payload = None
    if metric.provenance is not None:
        provenance_payload = {
            **asdict(metric.provenance),
            "receipt_sha256": metric.provenance.receipt_sha256,
        }
    header = {
        "schema": WIRE_SCHEMA,
        "bank_schema": bank.schema,
        "metric_schema": metric.schema,
        "classes": list(bank.classes),
        "support_counts": list(bank.support_counts),
        "active_k": bank.active_k,
        "config": {"schema": LOCK_SCHEMA, **asdict(bank.config)},
        "config_lock_digest": bank.config_lock_digest,
        "bank_receipt_sha256": bank.bank_receipt_sha256,
        "bank_quantization_audit": _json_value(bank.quantization_audit),
        "metric_receipt_sha256": metric.metric_receipt_sha256,
        "metric_source": metric.source,
        "metric_class_shared": metric.class_shared,
        "metric_effective_rank": metric.effective_rank,
        "metric_builder_no_fit": metric.builder_no_fit,
        "metric_typed_provenance_receipt": provenance_payload,
        "metric_minimum_eigenvalue": metric.minimum_eigenvalue,
        "metric_condition_number": metric.condition_number,
        "metric_sqrt_update_frobenius_norm": (
            metric.sqrt_metric_update_frobenius_norm
        ),
        "query_state_updates": 0,
        "query_batch_dependency": False,
        "query_file_io": False,
    }
    header_bytes = _canonical_bytes(header)
    if len(header_bytes) > MAX_WIRE_HEADER_BYTES:
        raise ZIDStudentTQKNNError("serialized wire header exceeds the fixed size limit")
    arrays = (
        ("support_codes_qint8", bank.codes_qint8),
        ("support_scales_fp16", bank.scales_fp16),
        ("class_indices_int16", bank.class_indices_int16),
        ("class_scales_fp16", bank.class_scales_fp16),
        ("metric_basis_codes_qint8", metric.basis_codes_qint8),
        ("metric_basis_scales_fp16", metric.basis_scales_fp16),
        ("metric_attenuation_fp16", metric.attenuation_fp16),
    )
    wire = b"".join(
        [
            WIRE_MAGIC,
            struct.pack("<Q", len(header_bytes)),
            header_bytes,
            struct.pack("<H", len(arrays)),
            *[_wire_array(name, value) for name, value in arrays],
        ]
    )
    if len(wire) > MAX_WIRE_BYTES:
        raise ZIDStudentTQKNNError("serialized wire exceeds the fixed size limit")
    return wire


def _wire_take(data: bytes, position: int, size: int, context: str) -> tuple[bytes, int]:
    if size < 0 or position < 0 or position + size > len(data):
        raise ZIDStudentTQKNNError(f"wire is truncated while reading {context}")
    return data[position : position + size], position + size


def _wire_uint(data: bytes, position: int, fmt: str, context: str) -> tuple[int, int]:
    size = struct.calcsize(fmt)
    raw, position = _wire_take(data, position, size, context)
    return int(struct.unpack(fmt, raw)[0]), position


def deserialize_typed_zid_runtime_state(
    wire: bytes,
) -> tuple[TypedINT8ZIDSupportBank, TypedSharedPSDMetric]:
    """Strictly reconstruct one bank/metric pair from the canonical wire."""

    if type(wire) is not bytes:
        raise ZIDStudentTQKNNError("wire must be exact bytes")
    if len(wire) > MAX_WIRE_BYTES:
        raise ZIDStudentTQKNNError("wire exceeds the fixed size limit")
    if not wire.startswith(WIRE_MAGIC):
        raise ZIDStudentTQKNNError("wire magic mismatch or truncated prefix")
    position = len(WIRE_MAGIC)
    header_length, position = _wire_uint(wire, position, "<Q", "header length")
    if header_length < 2 or header_length > MAX_WIRE_HEADER_BYTES:
        raise ZIDStudentTQKNNError("wire header length is invalid")
    header_bytes, position = _wire_take(
        wire, position, header_length, "canonical header"
    )
    try:
        header = json.loads(header_bytes.decode("utf-8"))
        canonical_header = _canonical_bytes(header)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ZIDStudentTQKNNError("wire header is not canonical JSON") from exc
    if type(header) is not dict or canonical_header != header_bytes:
        raise ZIDStudentTQKNNError("wire header canonical field order drift")
    required_header = {
        "schema",
        "bank_schema",
        "metric_schema",
        "classes",
        "support_counts",
        "active_k",
        "config",
        "config_lock_digest",
        "bank_receipt_sha256",
        "bank_quantization_audit",
        "metric_receipt_sha256",
        "metric_source",
        "metric_class_shared",
        "metric_effective_rank",
        "metric_builder_no_fit",
        "metric_typed_provenance_receipt",
        "metric_minimum_eigenvalue",
        "metric_condition_number",
        "metric_sqrt_update_frobenius_norm",
        "query_state_updates",
        "query_batch_dependency",
        "query_file_io",
    }
    if set(header) != required_header:
        raise ZIDStudentTQKNNError("wire header exact schema mismatch")
    if (
        header["schema"] != WIRE_SCHEMA
        or header["bank_schema"] != BANK_SCHEMA
        or header["metric_schema"] != METRIC_SCHEMA
        or header["query_state_updates"] != 0
        or header["query_batch_dependency"] is not False
        or header["query_file_io"] is not False
    ):
        raise ZIDStudentTQKNNError("wire lifecycle/schema drift")

    config_payload = header["config"]
    config_fields = set(Phase1ZIDStudentTLock.__dataclass_fields__)
    if type(config_payload) is not dict or set(config_payload) != config_fields:
        raise ZIDStudentTQKNNError("wire Phase1 config exact schema mismatch")
    try:
        config = Phase1ZIDStudentTLock(**config_payload)
    except (TypeError, ValueError) as exc:
        raise ZIDStudentTQKNNError("wire Phase1 config is invalid") from exc
    if (
        type(header["active_k"]) is not int
        or header["active_k"] != config.active_k
        or header["config_lock_digest"] != config.lock_digest
    ):
        raise ZIDStudentTQKNNError("wire active K/config lock drift")

    raw_classes = header["classes"]
    raw_counts = header["support_counts"]
    if (
        type(raw_classes) is not list
        or not all(type(value) is str for value in raw_classes)
        or type(raw_counts) is not list
        or not all(type(value) is int for value in raw_counts)
    ):
        raise ZIDStudentTQKNNError("wire class registry/support counts type drift")
    classes = _registry(raw_classes)
    counts = tuple(raw_counts)
    if (
        len(counts) != len(classes)
        or any(value != config.active_k for value in counts)
    ):
        raise ZIDStudentTQKNNError("wire support counts are not balanced active K")
    support_rows = sum(counts)

    rank = header["metric_effective_rank"]
    if type(rank) is not int or not 0 <= rank <= MAX_METRIC_RANK:
        raise ZIDStudentTQKNNError("wire metric rank drift")
    if (
        type(header["metric_source"]) is not str
        or type(header["metric_class_shared"]) is not bool
        or type(header["metric_builder_no_fit"]) is not bool
        or any(
            type(header[field]) not in (int, float)
            for field in (
                "metric_minimum_eigenvalue",
                "metric_condition_number",
                "metric_sqrt_update_frobenius_norm",
            )
        )
    ):
        raise ZIDStudentTQKNNError("wire metric header type drift")
    raw_provenance = header["metric_typed_provenance_receipt"]
    provenance = None
    if raw_provenance is not None:
        provenance_fields = set(TypedMetricProvenanceReceipt.__dataclass_fields__)
        if (
            type(raw_provenance) is not dict
            or set(raw_provenance) != provenance_fields | {"receipt_sha256"}
        ):
            raise ZIDStudentTQKNNError("wire metric provenance exact schema mismatch")
        receipt_sha256 = raw_provenance["receipt_sha256"]
        provenance_values = {
            key: value
            for key, value in raw_provenance.items()
            if key != "receipt_sha256"
        }
        try:
            provenance = TypedMetricProvenanceReceipt(**provenance_values)
        except (TypeError, ValueError) as exc:
            raise ZIDStudentTQKNNError("wire metric provenance is invalid") from exc
        if receipt_sha256 != provenance.receipt_sha256:
            raise ZIDStudentTQKNNError("wire metric provenance receipt mismatch")

    array_count, position = _wire_uint(wire, position, "<H", "array count")
    if array_count != len(WIRE_ARRAY_SPECS):
        raise ZIDStudentTQKNNError("wire array count drift")
    expected_shapes = (
        (support_rows, Z_DIM),
        (support_rows,),
        (support_rows,),
        (len(classes),),
        (rank, Z_DIM),
        (rank,),
        (rank,),
    )
    arrays: dict[str, np.ndarray] = {}
    for (expected_name, expected_dtype, expected_ndim), expected_shape in zip(
        WIRE_ARRAY_SPECS, expected_shapes
    ):
        name_length, position = _wire_uint(wire, position, "<H", "array name length")
        if name_length < 1 or name_length > 128:
            raise ZIDStudentTQKNNError("wire array name length drift")
        name_bytes, position = _wire_take(wire, position, name_length, "array name")
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ZIDStudentTQKNNError("wire array name is not ASCII") from exc
        if name != expected_name:
            raise ZIDStudentTQKNNError("wire array field order drift")

        dtype_length, position = _wire_uint(wire, position, "<H", "dtype length")
        dtype_bytes, position = _wire_take(wire, position, dtype_length, "dtype")
        try:
            dtype_text = dtype_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ZIDStudentTQKNNError("wire array dtype is not ASCII") from exc
        if dtype_text != expected_dtype.str:
            raise ZIDStudentTQKNNError(
                f"wire array dtype drift for {expected_name}"
            )
        ndim, position = _wire_uint(wire, position, "<H", "array ndim")
        if ndim != expected_ndim:
            raise ZIDStudentTQKNNError(f"wire array shape drift for {expected_name}")
        shape = []
        for axis in range(ndim):
            value, position = _wire_uint(
                wire, position, "<Q", f"array shape axis {axis}"
            )
            shape.append(value)
        if tuple(shape) != expected_shape:
            raise ZIDStudentTQKNNError(f"wire array shape drift for {expected_name}")
        payload_length, position = _wire_uint(
            wire, position, "<Q", "array payload length"
        )
        expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * int(
            expected_dtype.itemsize
        )
        if payload_length != expected_bytes:
            raise ZIDStudentTQKNNError(
                f"wire array payload/shape drift for {expected_name}"
            )
        payload, position = _wire_take(
            wire, position, payload_length, f"array payload {expected_name}"
        )
        arrays[expected_name] = np.frombuffer(
            payload, dtype=expected_dtype
        ).copy().reshape(expected_shape)
    if position != len(wire):
        raise ZIDStudentTQKNNError("wire has trailing bytes")

    quantization = header["bank_quantization_audit"]
    if not isinstance(quantization, Mapping):
        raise ZIDStudentTQKNNError("wire bank quantization audit must be a mapping")
    bank = TypedINT8ZIDSupportBank(
        classes=classes,
        support_counts=counts,
        codes_qint8=arrays["support_codes_qint8"],
        scales_fp16=arrays["support_scales_fp16"],
        class_indices_int16=arrays["class_indices_int16"],
        class_scales_fp16=arrays["class_scales_fp16"],
        active_k=config.active_k,
        config_lock_digest=header["config_lock_digest"],
        config=config,
        quantization_audit=quantization,
        bank_receipt_sha256=header["bank_receipt_sha256"],
    )
    metric = TypedSharedPSDMetric(
        basis_codes_qint8=arrays["metric_basis_codes_qint8"],
        basis_scales_fp16=arrays["metric_basis_scales_fp16"],
        attenuation_fp16=arrays["metric_attenuation_fp16"],
        effective_rank=rank,
        source=header["metric_source"],
        config_lock_digest=header["config_lock_digest"],
        minimum_eigenvalue=float(header["metric_minimum_eigenvalue"]),
        condition_number=float(header["metric_condition_number"]),
        sqrt_metric_update_frobenius_norm=float(
            header["metric_sqrt_update_frobenius_norm"]
        ),
        metric_receipt_sha256=header["metric_receipt_sha256"],
        builder_no_fit=header["metric_builder_no_fit"],
        provenance=provenance,
        class_shared=header["metric_class_shared"],
    )
    return bank, metric


def audit_runtime_state(
    bank: TypedINT8ZIDSupportBank, metric: TypedSharedPSDMetric
) -> dict[str, Any]:
    wire = serialize_typed_zid_runtime_state(bank, metric)
    arrays = (
        bank.codes_qint8,
        bank.scales_fp16,
        bank.class_indices_int16,
        bank.class_scales_fp16,
        metric.basis_codes_qint8,
        metric.basis_scales_fp16,
        metric.attenuation_fp16,
    )
    rank = metric.effective_rank
    support_rows = bank.support_row_count
    return {
        "schema": "cvs.phase2.zid_student_t_qknn.resource_audit.v1",
        "feature_dim": Z_DIM,
        "class_count": len(bank.classes),
        "active_k": bank.active_k,
        "support_rows": support_rows,
        "metric_rank": rank,
        "trainable_parameters": 0,
        "parameter_equivalent_metric_state": rank * (Z_DIM + 1),
        "epochs": 0,
        "optimizer_steps": 0,
        "numeric_array_state_bytes": int(sum(array.nbytes for array in arrays)),
        "actual_serialized_state_bytes": len(wire),
        "serialized_state_sha256": _sha256_bytes(wire),
        "score_call_fixed_matmul_mac": support_rows * rank * Z_DIM,
        "support_dot_matmul_mac_per_query": support_rows * Z_DIM,
        "metric_query_projection_matmul_mac_per_query": rank * Z_DIM,
        "metric_projected_pair_matmul_mac_per_query": support_rows * rank,
        "score_query_variable_matmul_mac_per_query": (
            support_rows * Z_DIM + rank * Z_DIM + support_rows * rank
        ),
        "matmul_mac_formula": "S*r*d + Q*(S*d + r*d + S*r)",
        "matmul_mac_scope": (
            "matrix multiplications in precision-cosine only; excludes "
            "receipt hashing, INT8 decode, normalization, reductions, "
            "elementwise operations, exp, log, and serialization"
        ),
        "non_matmul_work_included_in_mac": False,
        "latency_measurement_included": False,
        "persistent_decoded_cache_bytes": 0,
        "persistent_metric_projection_cache_bytes": 0,
        "query_state_updates": 0,
        "query_batch_dependency": False,
        "query_file_io": False,
        "dense_query_graph": False,
        "query_dependent_batch_optimization": False,
    }


__all__ = [
    "ALLOWED_K",
    "ALLOWED_METRIC_FIT_SCOPES",
    "BANK_SCHEMA",
    "LOCK_SCHEMA",
    "MAX_METRIC_RANK",
    "METRIC_SCHEMA",
    "METRIC_PROVENANCE_SCHEMA",
    "Phase1ZIDStudentTLock",
    "TypedINT8ZIDSupportBank",
    "TypedMetricProvenanceReceipt",
    "TypedSharedPSDMetric",
    "WIRE_MAGIC",
    "ZIDStudentTQKNNError",
    "Z_DIM",
    "audit_int8_margin",
    "audit_runtime_state",
    "build_typed_shared_psd_metric",
    "build_typed_zid_support_bank",
    "decode_zid_support_bank",
    "deserialize_typed_zid_runtime_state",
    "identity_shared_psd_metric",
    "normalize_zid_rows",
    "score_zid_student_t_logits",
    "serialize_typed_zid_runtime_state",
    "softmax_probabilities",
]
