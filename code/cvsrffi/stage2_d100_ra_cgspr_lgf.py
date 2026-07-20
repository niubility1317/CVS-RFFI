"""D100 redundancy-aware coverage-gated simplex-prototype ridge core.

This module adds one discriminative, registry-global head to the local D99
metric-kernel core.  It consumes only the immutable INT8 target support bank
compiled by D99.  Ground aggregates can affect the shared generalized cosine
metric, but never contribute a class logit, a target class mean, or an old/new
specific score.  Every registered class is mapped to the same centered simplex
target and every query is evaluated independently.

The repository does not yet contain the external Phase1 LODO rescue authority
or the external formal target-state producer.  Consequently all artifacts from
this module are local research artifacts and formal prediction fails closed.
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

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99


SCHEMA = "cvs.phase2.d100.ra_cgspr_lgf.local_core.v1"
LOCK_SCHEMA = "cvs.phase1.d100.ra_cgspr_lgf_lock.v1"
STATE_SCHEMA = "cvs.phase2.d100.int8_simplex_ridge_state.v1"
RESOURCE_SCHEMA = "cvs.phase2.d100.resource_closure.v1"
QUANTIZATION_SCHEMA = "cvs.phase2.d100.int8_margin_audit.v1"
COMPLEMENTARITY_SCHEMA = "cvs.phase1.d100.complementarity_audit.v1"
COMBINED_RESOURCE_SCHEMA = "cvs.phase2.d100.combined_wire_budget.v1"
D81_LOGIT_BATCH_SCHEMA = "cvs.phase2.d100.typed_d81_logit_batch.v1"
CANONICAL_FUSION_SCHEMA = "cvs.phase2.d100.canonical_d81_d99_d100_fusion.v1"
WIRE_MAGIC = b"D100-RA-CGSPR-LGF\0"
FEATURE_DIM = 288
Z_DIM = 160
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
ALLOWED_K = (1, 5, 10, 20)
INT8_MAX = 127.0
EPSILON = 1.0e-12
STATE_LIMIT_BYTES = 256 * 1024

# These must be provisioned by independently reviewed Phase1/external producer
# work.  Caller-provided digests cannot grant authority while either is None.
TRUSTED_PHASE1_LODO_RESCUE_RECEIPT_SHA256: str | None = None
TRUSTED_EXTERNAL_PHASE2_AUTHORITY_SHA256: str | None = None


class D100RACGSPRError(ValueError):
    """Raised when D100 geometry, lifecycle, wire, or resource closure drifts."""


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
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D100RACGSPRError(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _classes(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) < 2 or len(set(result)) != len(result) or any(not value for value in result):
        raise D100RACGSPRError("D100 requires at least two unique registered classes")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise D100RACGSPRError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class Phase1D100Lock:
    """All D100 tunables; each K value must be locked by Phase1 evidence."""

    lambda_k1: float
    lambda_k5: float
    lambda_k10: float
    lambda_k20: float
    temperature_k1: float
    temperature_k5: float
    temperature_k10: float
    temperature_k20: float
    d99_temperature_k1: float
    d99_temperature_k5: float
    d99_temperature_k10: float
    d99_temperature_k20: float
    alpha_k1: float
    alpha_k5: float
    alpha_k10: float
    alpha_k20: float
    d99_phase1_lock_digest: str
    phase1_lodo_rescue_receipt_sha256: str
    external_phase2_authority_sha256: str
    quantization_margin_audit_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "lambda_k1",
            "lambda_k5",
            "lambda_k10",
            "lambda_k20",
            "temperature_k1",
            "temperature_k5",
            "temperature_k10",
            "temperature_k20",
            "d99_temperature_k1",
            "d99_temperature_k5",
            "d99_temperature_k10",
            "d99_temperature_k20",
        ):
            _positive_finite(getattr(self, name), name)
        for name in ("alpha_k1", "alpha_k5", "alpha_k10", "alpha_k20"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise D100RACGSPRError(f"{name} must be finite and in [0,1]")
        for name in (
            "d99_phase1_lock_digest",
            "phase1_lodo_rescue_receipt_sha256",
            "external_phase2_authority_sha256",
            "quantization_margin_audit_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256({"schema": LOCK_SCHEMA, **asdict(self)})

    def values_for_k(self, k_shot: int) -> tuple[float, float, float, float]:
        k = int(k_shot)
        if k not in ALLOWED_K:
            raise D100RACGSPRError(f"D100 supports only K in {ALLOWED_K}")
        return (
            float(getattr(self, f"lambda_k{k}")),
            float(getattr(self, f"temperature_k{k}")),
            float(getattr(self, f"d99_temperature_k{k}")),
            float(getattr(self, f"alpha_k{k}")),
        )

    @property
    def formal_authorities_ready(self) -> bool:
        return bool(
            TRUSTED_PHASE1_LODO_RESCUE_RECEIPT_SHA256 is not None
            and TRUSTED_EXTERNAL_PHASE2_AUTHORITY_SHA256 is not None
            and self.phase1_lodo_rescue_receipt_sha256
            == TRUSTED_PHASE1_LODO_RESCUE_RECEIPT_SHA256
            and self.external_phase2_authority_sha256
            == TRUSTED_EXTERNAL_PHASE2_AUTHORITY_SHA256
        )


def _normalize_rows(value: np.ndarray, dimension: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != dimension or not np.isfinite(rows).all():
        raise D100RACGSPRError(f"{name} must be finite [N,{dimension}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPSILON):
        raise D100RACGSPRError(f"{name} contains a zero-norm row")
    return rows / norms


def generalized_precision_sqrt_transform(
    rows: np.ndarray,
    metric_basis: np.ndarray,
    precision_attenuation: np.ndarray,
) -> np.ndarray:
    """Apply the analytic PSD square root of D99's low-rank precision.

    If P=I-B diag(a) B.T, then sqrt(P)=I-B diag(1-sqrt(1-a)) B.T.
    The returned rows are normalized so their ordinary dot product equals the
    generalized cosine induced by P.
    """

    values = _normalize_rows(rows, Z_DIM, "D100 z160 rows")
    basis = np.asarray(metric_basis, dtype=np.float64)
    attenuation = np.asarray(precision_attenuation, dtype=np.float64)
    rank = basis.shape[1] if basis.ndim == 2 else -1
    if (
        basis.shape != (Z_DIM, rank)
        or attenuation.shape != (rank,)
        or not np.isfinite(basis).all()
        or not np.isfinite(attenuation).all()
        or np.any(attenuation < 0.0)
        or np.any(attenuation >= 1.0)
    ):
        raise D100RACGSPRError("D100 precision basis/attenuation drift")
    if rank and not np.allclose(basis.T @ basis, np.eye(rank), atol=3e-5):
        raise D100RACGSPRError("D100 precision basis must be orthonormal")
    if rank:
        square_root_attenuation = 1.0 - np.sqrt(1.0 - attenuation)
        values = values - ((values @ basis) * square_root_attenuation) @ basis.T
    return _normalize_rows(values, Z_DIM, "D100 precision-sqrt z160 rows")


def _feature_map(
    features: np.ndarray,
    metric_basis: np.ndarray,
    precision_attenuation: np.ndarray,
    block_weights: tuple[float, float, float],
) -> np.ndarray:
    rows = np.asarray(features, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D100RACGSPRError("D100 features must be finite [N,288]")
    weights = np.asarray(block_weights, dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(float(np.sum(weights)), 1.0, atol=1e-12)
    ):
        raise D100RACGSPRError("D100 block weights must be positive and sum to one")
    z = generalized_precision_sqrt_transform(
        rows[:, :Z_DIM], metric_basis, precision_attenuation
    )
    fft = _normalize_rows(rows[:, 160:256], 96, "D100 FFT96 rows")
    rf = _normalize_rows(rows[:, 256:288], 32, "D100 RF32 rows")
    mapped = np.concatenate(
        [
            math.sqrt(weights[0]) * z,
            math.sqrt(weights[1]) * fft,
            math.sqrt(weights[2]) * rf,
        ],
        axis=1,
    )
    if not np.allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=2e-6):
        raise D100RACGSPRError("D100 mapped feature norm drift")
    return mapped


def _quantize_weight_rows(weights_c_by_d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(weights_c_by_d, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != FEATURE_DIM or not np.isfinite(weights).all():
        raise D100RACGSPRError("D100 ridge weights must be finite [C,288]")
    codes = np.empty(weights.shape, dtype=np.int8)
    scales = np.empty((len(weights), 3), dtype=np.float16)
    decoded = np.empty(weights.shape, dtype=np.float64)
    for row in range(len(weights)):
        for block_index, block in enumerate(BLOCK_SLICES):
            part = weights[row, block]
            maximum = float(np.max(np.abs(part)))
            scale = max(maximum / INT8_MAX, float(np.finfo(np.float16).tiny))
            scale16 = np.float16(scale)
            if not np.isfinite(scale16) or scale16 <= 0.0:
                raise D100RACGSPRError("D100 INT8 scale became invalid")
            code = np.clip(np.rint(part / float(scale16)), -127, 127).astype(np.int8)
            codes[row, block] = code
            scales[row, block_index] = scale16
            decoded[row, block] = code.astype(np.float64) * float(scale16)
    return codes, scales, decoded


def _decode_weight_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    code = np.asarray(codes)
    scale = np.asarray(scales)
    if code.dtype != np.int8 or code.ndim != 2 or code.shape[1] != FEATURE_DIM:
        raise D100RACGSPRError("D100 INT8 weight code drift")
    if scale.dtype != np.float16 or scale.shape != (len(code), 3):
        raise D100RACGSPRError("D100 FP16 weight scale drift")
    result = np.empty(code.shape, dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        result[:, block] = (
            code[:, block].astype(np.float32)
            * scale[:, block_index].astype(np.float32)[:, None]
        )
    return result


def _wire_arrays(
    codes: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray,
    basis: np.ndarray,
    attenuation: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("weight_codes_qint8", np.asarray(codes)),
        ("weight_scales_fp16", np.asarray(scales)),
        ("bias_fp16", np.asarray(bias)),
        ("metric_basis_fp32", np.asarray(basis)),
        ("precision_attenuation_fp32", np.asarray(attenuation)),
    )


def _serialize_wire(header: Mapping[str, Any], arrays: Sequence[tuple[str, np.ndarray]]) -> bytes:
    header_bytes = _canonical_bytes(header)
    parts = [WIRE_MAGIC, struct.pack("<Q", len(header_bytes)), header_bytes]
    for name, value in arrays:
        name_bytes = str(name).encode("ascii")
        payload = np.ascontiguousarray(value).tobytes(order="C")
        parts.extend(
            [
                struct.pack("<H", len(name_bytes)),
                name_bytes,
                struct.pack("<Q", len(payload)),
                payload,
            ]
        )
    return b"".join(parts)


def _resource_from_dimensions(
    *,
    class_count: int,
    k_shot: int,
    rank: int,
    alpha: float,
    numeric_bytes: int,
    wire_bytes: int,
) -> dict[str, Any]:
    classes, shots, retained_rank = int(class_count), int(k_shot), int(rank)
    if classes < 2 or shots not in ALLOWED_K or not 0 <= retained_rank <= 8:
        raise D100RACGSPRError("D100 resource dimensions drift")
    support_rows = classes * shots
    feature_map_macs = support_rows * (
        FEATURE_DIM + 2 * Z_DIM * max(1, retained_rank)
    )
    class_mean_macs = support_rows * FEATURE_DIM
    gram_macs = classes * classes * FEATURE_DIM
    compile_macs = FEATURE_DIM * classes * classes
    solve_flops = 3 * classes**3
    # D99's canonical public scorer decodes/normalizes the full support bank on
    # each prediction call and then evaluates every query/support pair.  Mirror
    # its dimension formula instead of trusting a caller-provided resource map.
    d99_support_decode_normalize_macs = support_rows * FEATURE_DIM * 5
    d99_query_precision_norm_macs = Z_DIM
    d99_query_kernel_pair_macs = support_rows * (
        Z_DIM * (2 * retained_rank + 3) + 96 + 32
    )
    d99_query_normalize_macs = FEATURE_DIM * 5
    d99_query_macs = int(
        d99_support_decode_normalize_macs
        + d99_query_precision_norm_macs
        + d99_query_kernel_pair_macs
        + d99_query_normalize_macs
    )
    # D100 currently calls the D99 scorer and its own scorer independently, so
    # normalization/projection cannot be counted as shared work.  INT8 weight
    # decode is also performed on every local ridge score call.
    d100_weight_decode_macs = classes * FEATURE_DIM
    d100_query_normalize_macs = FEATURE_DIM * 5
    d100_metric_projection_macs = 2 * Z_DIM * retained_rank + retained_rank
    d100_block_scale_macs = FEATURE_DIM
    d100_ridge_matvec_macs = classes * FEATURE_DIM
    d100_bias_macs = classes
    d100_incremental_macs = int(
        d100_weight_decode_macs
        + d100_query_normalize_macs
        + d100_metric_projection_macs
        + d100_block_scale_macs
        + d100_ridge_matvec_macs
        + d100_bias_macs
    )
    d99_softmax_scalar_ops = classes * 8
    d100_softmax_scalar_ops = classes * 8
    fusion_scalar_ops = classes * 3
    alpha_zero = float(alpha) == 0.0
    combined_query_macs = d99_query_macs + (0 if alpha_zero else d100_incremental_macs)
    combined_scalar_ops = d99_softmax_scalar_ops + (
        0 if alpha_zero else d100_softmax_scalar_ops + fusion_scalar_ops
    )
    return {
        "schema": RESOURCE_SCHEMA,
        "class_count": classes,
        "k_shot": shots,
        "metric_rank": retained_rank,
        "trainable_parameter_equivalent": classes * FEATURE_DIM + classes,
        "optimizer_steps": 0,
        "epochs": 0,
        "support_feature_map_mac_upper_bound": int(feature_map_macs),
        "class_mean_mac_upper_bound": int(class_mean_macs),
        "gram_mac_upper_bound": int(gram_macs),
        "ridge_compile_mac_upper_bound": int(compile_macs),
        "cholesky_solve_flop_upper_bound": int(solve_flops),
        "d99_support_decode_normalize_mac_per_prediction_call": int(
            d99_support_decode_normalize_macs
        ),
        "d99_query_kernel_pair_mac_upper_bound": int(d99_query_kernel_pair_macs),
        "d99_query_normalize_mac_upper_bound": int(d99_query_normalize_macs),
        "d99_query_mac_upper_bound_per_sample": int(d99_query_macs),
        "d100_weight_decode_mac_upper_bound": int(d100_weight_decode_macs),
        "d100_query_normalize_mac_upper_bound": int(d100_query_normalize_macs),
        "d100_metric_projection_mac_upper_bound": int(d100_metric_projection_macs),
        "d100_block_scale_mac_upper_bound": int(d100_block_scale_macs),
        "d100_ridge_matvec_mac_upper_bound": int(d100_ridge_matvec_macs),
        "d100_bias_mac_upper_bound": int(d100_bias_macs),
        "d100_incremental_query_mac_upper_bound_per_sample": int(
            d100_incremental_macs
        ),
        "combined_query_mac_upper_bound_per_sample": int(combined_query_macs),
        "combined_query_scalar_operation_upper_bound_per_sample": int(
            combined_scalar_ops
        ),
        "alpha_zero_skips_d100_query_branch": alpha_zero,
        "numeric_logical_state_bytes": int(numeric_bytes),
        "actual_serialized_state_bytes": int(wire_bytes),
        "query_state_updates": 0,
        "query_file_io": False,
        "query_batch_dependency": False,
        "formal_resource_claim": False,
    }


def _state_core(
    *,
    classes: tuple[str, ...],
    k_shot: int,
    codes: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray,
    basis: np.ndarray,
    attenuation: np.ndarray,
    block_weights: tuple[float, float, float],
    ridge_lambda: float,
    temperature: float,
    d99_temperature: float,
    alpha: float,
    d99_bank_receipt_sha256: str,
    d99_lock_digest: str,
    d100_lock_digest: str,
    source_bank_deployment_status: str,
    formal_phase2_eligible: bool,
    quantization_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "classes": list(classes),
        "k_shot": int(k_shot),
        "arrays": {
            name: _array_receipt(value)
            for name, value in _wire_arrays(codes, scales, bias, basis, attenuation)
        },
        "block_weights": list(block_weights),
        "ridge_lambda": float(ridge_lambda),
        "temperature": float(temperature),
        "d99_temperature": float(d99_temperature),
        "alpha": float(alpha),
        "d99_bank_receipt_sha256": d99_bank_receipt_sha256,
        "d99_lock_digest": d99_lock_digest,
        "d100_lock_digest": d100_lock_digest,
        "source_bank_deployment_status": str(source_bank_deployment_status),
        "formal_phase2_eligible": bool(formal_phase2_eligible),
        "authority_status": (
            "FORMAL_AUTHORITIES_BOUND"
            if formal_phase2_eligible
            else "BLOCKED_MISSING_EXTERNAL_AUTHORITY_OR_PHASE1_LODO_RESCUE_RECEIPT"
        ),
        "query_rows_used_for_fit": 0,
        "same_formula_all_registered_classes": True,
        "old_new_role_specific_scoring": False,
        "quantization_audit": _json_value(quantization_audit),
    }


def _closed_state_artifact(
    *,
    classes: tuple[str, ...],
    k_shot: int,
    codes: np.ndarray,
    scales: np.ndarray,
    bias: np.ndarray,
    basis: np.ndarray,
    attenuation: np.ndarray,
    block_weights: tuple[float, float, float],
    ridge_lambda: float,
    temperature: float,
    d99_temperature: float,
    alpha: float,
    d99_bank_receipt_sha256: str,
    d99_lock_digest: str,
    d100_lock_digest: str,
    source_bank_deployment_status: str,
    formal_phase2_eligible: bool,
    quantization_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bytes]:
    arrays = _wire_arrays(codes, scales, bias, basis, attenuation)
    numeric_bytes = int(sum(value.nbytes for _name, value in arrays))
    core = _state_core(
        classes=classes,
        k_shot=k_shot,
        codes=codes,
        scales=scales,
        bias=bias,
        basis=basis,
        attenuation=attenuation,
        block_weights=block_weights,
        ridge_lambda=ridge_lambda,
        temperature=temperature,
        d99_temperature=d99_temperature,
        alpha=alpha,
        d99_bank_receipt_sha256=d99_bank_receipt_sha256,
        d99_lock_digest=d99_lock_digest,
        d100_lock_digest=d100_lock_digest,
        source_bank_deployment_status=source_bank_deployment_status,
        formal_phase2_eligible=formal_phase2_eligible,
        quantization_audit=quantization_audit,
    )
    wire_size = 0
    for _ in range(8):
        resource = _resource_from_dimensions(
            class_count=len(classes),
            k_shot=k_shot,
            rank=basis.shape[1],
            alpha=alpha,
            numeric_bytes=numeric_bytes,
            wire_bytes=wire_size,
        )
        receipt = _canonical_sha256(
            {"schema": STATE_SCHEMA, "core": core, "resource_audit": resource}
        )
        header = {
            **core,
            "resource_audit": resource,
            "state_receipt_sha256": receipt,
        }
        wire = _serialize_wire(header, arrays)
        if len(wire) == wire_size:
            return resource, receipt, wire
        wire_size = len(wire)
    raise D100RACGSPRError("D100 wire-size fixed point did not converge")


@dataclass(frozen=True, slots=True)
class D100SimplexRidgeState:
    classes: tuple[str, ...]
    k_shot: int
    weight_codes_qint8: np.ndarray
    weight_scales_fp16: np.ndarray
    bias_fp16: np.ndarray
    metric_basis_fp32: np.ndarray
    precision_attenuation_fp32: np.ndarray
    block_weights: tuple[float, float, float]
    ridge_lambda: float
    temperature: float
    d99_temperature: float
    alpha: float
    d99_bank_receipt_sha256: str
    d99_lock_digest: str
    d100_lock_digest: str
    source_bank_deployment_status: str
    formal_phase2_eligible: bool
    quantization_audit: Mapping[str, Any]
    resource_audit: Mapping[str, Any]
    state_receipt_sha256: str
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        classes = _classes(self.classes)
        rank = self.metric_basis_fp32.shape[1] if self.metric_basis_fp32.ndim == 2 else -1
        block_weights = np.asarray(self.block_weights, dtype=np.float64)
        if (
            self.schema != STATE_SCHEMA
            or int(self.k_shot) not in ALLOWED_K
            or self.weight_codes_qint8.dtype != np.int8
            or self.weight_codes_qint8.shape != (len(classes), FEATURE_DIM)
            or self.weight_scales_fp16.dtype != np.float16
            or self.weight_scales_fp16.shape != (len(classes), 3)
            or self.bias_fp16.dtype != np.float16
            or self.bias_fp16.shape != (len(classes),)
            or self.metric_basis_fp32.dtype != np.float32
            or self.metric_basis_fp32.shape != (Z_DIM, rank)
            or self.precision_attenuation_fp32.dtype != np.float32
            or self.precision_attenuation_fp32.shape != (rank,)
            or not np.isfinite(self.weight_scales_fp16).all()
            or not np.isfinite(self.bias_fp16).all()
            or not np.isfinite(self.metric_basis_fp32).all()
            or not np.isfinite(self.precision_attenuation_fp32).all()
            or np.any(self.weight_scales_fp16 <= 0.0)
            or np.any(self.precision_attenuation_fp32 < 0.0)
            or np.any(self.precision_attenuation_fp32 >= 1.0)
            or block_weights.shape != (3,)
            or not np.isfinite(block_weights).all()
            or np.any(block_weights <= 0.0)
            or not np.isclose(float(np.sum(block_weights)), 1.0, atol=1e-12)
            or (
                rank
                and not np.allclose(
                    self.metric_basis_fp32.T @ self.metric_basis_fp32,
                    np.eye(rank),
                    atol=3e-5,
                )
            )
            or not math.isfinite(float(self.ridge_lambda))
            or float(self.ridge_lambda) <= 0.0
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) <= 0.0
            or not math.isfinite(float(self.d99_temperature))
            or float(self.d99_temperature) <= 0.0
            or not math.isfinite(float(self.alpha))
            or not 0.0 <= float(self.alpha) <= 1.0
            or self.formal_phase2_eligible is not False
        ):
            raise D100RACGSPRError("D100 state invariant drift")
        for name in (
            "d99_bank_receipt_sha256",
            "d99_lock_digest",
            "d100_lock_digest",
            "state_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name, dtype in (
            ("weight_codes_qint8", np.int8),
            ("weight_scales_fp16", np.float16),
            ("bias_fp16", np.float16),
            ("metric_basis_fp32", np.float32),
            ("precision_attenuation_fp32", np.float32),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "block_weights", tuple(float(v) for v in self.block_weights))
        object.__setattr__(self, "quantization_audit", MappingProxyType(dict(self.quantization_audit)))
        object.__setattr__(self, "resource_audit", MappingProxyType(dict(self.resource_audit)))
        resource, receipt, _wire = _closed_state_artifact(
            classes=classes,
            k_shot=self.k_shot,
            codes=self.weight_codes_qint8,
            scales=self.weight_scales_fp16,
            bias=self.bias_fp16,
            basis=self.metric_basis_fp32,
            attenuation=self.precision_attenuation_fp32,
            block_weights=self.block_weights,
            ridge_lambda=self.ridge_lambda,
            temperature=self.temperature,
            d99_temperature=self.d99_temperature,
            alpha=self.alpha,
            d99_bank_receipt_sha256=self.d99_bank_receipt_sha256,
            d99_lock_digest=self.d99_lock_digest,
            d100_lock_digest=self.d100_lock_digest,
            source_bank_deployment_status=self.source_bank_deployment_status,
            formal_phase2_eligible=self.formal_phase2_eligible,
            quantization_audit=self.quantization_audit,
        )
        if _json_value(self.resource_audit) != resource or self.state_receipt_sha256 != receipt:
            raise D100RACGSPRError("D100 state resource/receipt closure drift")


def build_simplex_ridge_state(
    bank: d99.TypedINT8MetricKernelBank,
    *,
    config: Phase1D100Lock,
) -> D100SimplexRidgeState:
    """Build a class-symmetric ridge head from the exact typed D99 support bank."""

    if type(bank) is not d99.TypedINT8MetricKernelBank or type(config) is not Phase1D100Lock:
        raise D100RACGSPRError("D100 requires exact typed D99 bank and lock types")
    if bank.config.lock_digest != config.d99_phase1_lock_digest:
        raise D100RACGSPRError("D100/D99 Phase1 lock digest drift")
    classes = _classes(bank.classes)
    k_shot = int(bank.metric.k_shot)
    ridge_scale, temperature, d99_temperature, alpha = config.values_for_k(k_shot)
    if tuple(bank.support_counts) != tuple(k_shot for _ in classes):
        raise D100RACGSPRError("D100 requires balanced K-shot D99 support bank")
    decoded = d99.decode_support_bank(bank)
    basis = np.asarray(bank.metric.metric_basis_fp32, dtype=np.float32)
    attenuation = np.asarray(bank.metric.precision_attenuation_fp32, dtype=np.float32)
    block_weights = (
        float(bank.config.z_weight),
        float(bank.config.fft_weight),
        float(bank.config.rf_weight),
    )
    mapped = _feature_map(decoded, basis, attenuation, block_weights)
    indices = np.asarray(bank.class_indices_int16, dtype=np.int64)
    prototypes = np.stack(
        [np.mean(mapped[indices == index], axis=0) for index in range(len(classes))]
    )
    prototypes = _normalize_rows(prototypes, FEATURE_DIM, "D100 target class prototypes")
    mean = np.mean(prototypes, axis=0)
    centered = prototypes - mean[None, :]
    target = np.eye(len(classes), dtype=np.float64) - np.full(
        (len(classes), len(classes)), 1.0 / len(classes), dtype=np.float64
    )
    gram = centered @ centered.T
    scale = float(np.trace(gram)) / max(len(classes) - 1, 1)
    ridge_lambda = float(EPSILON + ridge_scale * max(scale, EPSILON))
    system = 0.5 * (gram + gram.T) + ridge_lambda * np.eye(len(classes))
    eigenvalue_min = float(np.min(np.linalg.eigvalsh(system)))
    if not math.isfinite(eigenvalue_min) or eigenvalue_min <= 0.0:
        raise D100RACGSPRError("D100 ridge system is not positive definite")
    try:
        dual = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as exc:
        raise D100RACGSPRError("D100 ridge solve failed") from exc
    teacher_weight = centered.T @ dual
    teacher_bias = -mean @ teacher_weight
    codes, scales, decoded_weight_rows = _quantize_weight_rows(teacher_weight.T)
    decoded_weight = decoded_weight_rows.T
    deployed_bias = np.asarray(-mean @ decoded_weight, dtype=np.float16)
    teacher_logits = mapped @ teacher_weight + teacher_bias[None, :]
    deployed_logits = mapped @ decoded_weight + deployed_bias.astype(np.float64)[None, :]
    order = np.argsort(teacher_logits, axis=1, kind="stable")
    rows = np.arange(len(mapped))
    winner = order[:, -1]
    runner = order[:, -2]
    teacher_margin = teacher_logits[rows, winner] - teacher_logits[rows, runner]
    deployed_margin = deployed_logits[rows, winner] - deployed_logits[rows, runner]
    flips = deployed_margin <= 0.0
    quantization = {
        "schema": QUANTIZATION_SCHEMA,
        "teacher_deployed_logit_max_abs_error": float(
            np.max(np.abs(teacher_logits - deployed_logits))
        ),
        "support_top1_agreement": float(
            np.mean(np.argmax(teacher_logits, axis=1) == np.argmax(deployed_logits, axis=1))
        ),
        "teacher_margin_sign_flip_count": int(np.sum(flips)),
        "teacher_margin_min": float(np.min(teacher_margin)),
        "deployed_teacher_winner_margin_min": float(np.min(deployed_margin)),
        "weight_codes_qint8": _array_receipt(codes),
        "weight_scales_fp16": _array_receipt(scales),
        "bias_fp16": _array_receipt(deployed_bias),
        "target_class_prototype_or_weight_fp32_sidecar_present": False,
        "shared_target_metric_fp32_persisted": True,
        "shared_target_metric_fp32_bytes": int(basis.nbytes + attenuation.nbytes),
        "same_quantization_formula_all_registered_classes": True,
        "query_rows_used": 0,
    }
    # Formal target execution remains blocked until both independent roots are
    # provisioned and the upstream D99 bank is itself formally authorized.
    formal = bool(
        config.formal_authorities_ready
        and str(bank.deployment_status).startswith("FORMAL_")
    )
    if formal:
        raise D100RACGSPRError(
            "D100 formal enablement requires a separately reviewed formal-state type"
        )
    resource, receipt, _wire = _closed_state_artifact(
        classes=classes,
        k_shot=k_shot,
        codes=codes,
        scales=scales,
        bias=deployed_bias,
        basis=basis,
        attenuation=attenuation,
        block_weights=block_weights,
        ridge_lambda=ridge_lambda,
        temperature=temperature,
        d99_temperature=d99_temperature,
        alpha=alpha,
        d99_bank_receipt_sha256=bank.bank_receipt_sha256,
        d99_lock_digest=bank.config.lock_digest,
        d100_lock_digest=config.lock_digest,
        source_bank_deployment_status=bank.deployment_status,
        formal_phase2_eligible=False,
        quantization_audit=quantization,
    )
    return D100SimplexRidgeState(
        classes=classes,
        k_shot=k_shot,
        weight_codes_qint8=codes,
        weight_scales_fp16=scales,
        bias_fp16=deployed_bias,
        metric_basis_fp32=basis,
        precision_attenuation_fp32=attenuation,
        block_weights=block_weights,
        ridge_lambda=ridge_lambda,
        temperature=temperature,
        d99_temperature=d99_temperature,
        alpha=alpha,
        d99_bank_receipt_sha256=bank.bank_receipt_sha256,
        d99_lock_digest=bank.config.lock_digest,
        d100_lock_digest=config.lock_digest,
        source_bank_deployment_status=bank.deployment_status,
        formal_phase2_eligible=False,
        quantization_audit=quantization,
        resource_audit=resource,
        state_receipt_sha256=receipt,
    )


def _verify_state(state: D100SimplexRidgeState) -> bool:
    if type(state) is not D100SimplexRidgeState:
        return False
    try:
        resource, receipt, _wire = _closed_state_artifact(
            classes=state.classes,
            k_shot=state.k_shot,
            codes=state.weight_codes_qint8,
            scales=state.weight_scales_fp16,
            bias=state.bias_fp16,
            basis=state.metric_basis_fp32,
            attenuation=state.precision_attenuation_fp32,
            block_weights=state.block_weights,
            ridge_lambda=state.ridge_lambda,
            temperature=state.temperature,
            d99_temperature=state.d99_temperature,
            alpha=state.alpha,
            d99_bank_receipt_sha256=state.d99_bank_receipt_sha256,
            d99_lock_digest=state.d99_lock_digest,
            d100_lock_digest=state.d100_lock_digest,
            source_bank_deployment_status=state.source_bank_deployment_status,
            formal_phase2_eligible=state.formal_phase2_eligible,
            quantization_audit=state.quantization_audit,
        )
        return resource == _json_value(state.resource_audit) and receipt == state.state_receipt_sha256
    except (D100RACGSPRError, ValueError, TypeError):
        return False


def _query_ready_state(state: D100SimplexRidgeState) -> bool:
    """Cheap in-memory query gate; full wire hashing is load/serialize-time work."""

    return bool(
        type(state) is D100SimplexRidgeState
        and state.formal_phase2_eligible is False
        and state.schema == STATE_SCHEMA
        and state.k_shot in ALLOWED_K
        and state.weight_codes_qint8.shape == (len(state.classes), FEATURE_DIM)
        and state.weight_scales_fp16.shape == (len(state.classes), 3)
        and state.bias_fp16.shape == (len(state.classes),)
        and state.metric_basis_fp32.shape[0] == Z_DIM
        and state.precision_attenuation_fp32.shape
        == (state.metric_basis_fp32.shape[1],)
        and not state.weight_codes_qint8.flags.writeable
        and not state.weight_scales_fp16.flags.writeable
        and not state.bias_fp16.flags.writeable
        and not state.metric_basis_fp32.flags.writeable
        and not state.precision_attenuation_fp32.flags.writeable
    )


def serialize_simplex_ridge_state(state: D100SimplexRidgeState) -> bytes:
    if not _verify_state(state):
        raise D100RACGSPRError("D100 serialization state verification failed")
    resource, receipt, wire = _closed_state_artifact(
        classes=state.classes,
        k_shot=state.k_shot,
        codes=state.weight_codes_qint8,
        scales=state.weight_scales_fp16,
        bias=state.bias_fp16,
        basis=state.metric_basis_fp32,
        attenuation=state.precision_attenuation_fp32,
        block_weights=state.block_weights,
        ridge_lambda=state.ridge_lambda,
        temperature=state.temperature,
        d99_temperature=state.d99_temperature,
        alpha=state.alpha,
        d99_bank_receipt_sha256=state.d99_bank_receipt_sha256,
        d99_lock_digest=state.d99_lock_digest,
        d100_lock_digest=state.d100_lock_digest,
        source_bank_deployment_status=state.source_bank_deployment_status,
        formal_phase2_eligible=state.formal_phase2_eligible,
        quantization_audit=state.quantization_audit,
    )
    if resource != _json_value(state.resource_audit) or receipt != state.state_receipt_sha256:
        raise D100RACGSPRError("D100 serialization resource/receipt drift")
    return wire


def score_simplex_ridge_logits(
    state: D100SimplexRidgeState,
    query_features: np.ndarray,
) -> np.ndarray:
    """Score independent query rows without mutating or consulting batch state."""

    if not _query_ready_state(state):
        raise D100RACGSPRError("D100 score state verification failed")
    mapped = _feature_map(
        query_features,
        state.metric_basis_fp32,
        state.precision_attenuation_fp32,
        state.block_weights,
    )
    weights = _decode_weight_rows(
        state.weight_codes_qint8, state.weight_scales_fp16
    ).astype(np.float64)
    logits = mapped @ weights.T + state.bias_fp16.astype(np.float64)[None, :]
    if not np.isfinite(logits).all():
        raise D100RACGSPRError("D100 ridge logits became non-finite")
    return _readonly(logits, np.float32)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / _positive_finite(
        temperature, "D100 temperature"
    )
    values -= np.max(values, axis=1, keepdims=True)
    exponential = np.exp(values)
    probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
    return _readonly(probabilities, np.float32)


def _typed_d81_batch_payload(
    *,
    classes: tuple[str, ...],
    k_shot: int,
    logits: np.ndarray,
    query_feature_receipt: Mapping[str, Any],
    source_schema: str,
    source_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": D81_LOGIT_BATCH_SCHEMA,
        "classes": list(classes),
        "k_shot": int(k_shot),
        "logits_fp32": _array_receipt(logits),
        "query_feature_receipt": dict(query_feature_receipt),
        "source_schema": str(source_schema),
        "source_receipt_sha256": source_receipt_sha256,
        "query_labels_input": False,
        "query_state_updates": 0,
    }


@dataclass(frozen=True, slots=True)
class TypedD81LogitBatch:
    """Receipt-bound D81 logits for the exact query feature rows."""

    classes: tuple[str, ...]
    k_shot: int
    logits_fp32: np.ndarray
    query_feature_receipt: Mapping[str, Any]
    source_schema: str
    source_receipt_sha256: str
    batch_receipt_sha256: str
    schema: str = D81_LOGIT_BATCH_SCHEMA

    def __post_init__(self) -> None:
        classes = _classes(self.classes)
        logits = np.asarray(self.logits_fp32)
        query_receipt = dict(self.query_feature_receipt)
        source_schema = str(self.source_schema)
        source_receipt = _require_sha256(
            self.source_receipt_sha256, "typed D81 source receipt"
        )
        if (
            self.schema != D81_LOGIT_BATCH_SCHEMA
            or int(self.k_shot) not in ALLOWED_K
            or logits.dtype != np.float32
            or logits.ndim != 2
            or logits.shape[1] != len(classes)
            or len(logits) < 1
            or not np.isfinite(logits).all()
            or not source_schema
            or set(query_receipt) != {"dtype", "shape", "nbytes", "sha256"}
            or query_receipt.get("dtype") != np.dtype(np.float32).str
            or query_receipt.get("shape") != [len(logits), FEATURE_DIM]
            or query_receipt.get("nbytes") != len(logits) * FEATURE_DIM * 4
        ):
            raise D100RACGSPRError("typed D81 logit batch invariant drift")
        _require_sha256(query_receipt.get("sha256", ""), "typed D81 query receipt")
        expected = _canonical_sha256(
            _typed_d81_batch_payload(
                classes=classes,
                k_shot=int(self.k_shot),
                logits=logits,
                query_feature_receipt=query_receipt,
                source_schema=source_schema,
                source_receipt_sha256=source_receipt,
            )
        )
        if self.batch_receipt_sha256 != expected:
            raise D100RACGSPRError("typed D81 logit batch receipt drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "k_shot", int(self.k_shot))
        object.__setattr__(self, "logits_fp32", _readonly(logits, np.float32))
        object.__setattr__(self, "query_feature_receipt", MappingProxyType(query_receipt))
        object.__setattr__(self, "source_schema", source_schema)
        object.__setattr__(self, "source_receipt_sha256", source_receipt)


def bind_typed_d81_logits(
    logits: np.ndarray,
    query_features: np.ndarray,
    classes: Sequence[str],
    k_shot: int,
    *,
    source_schema: str,
    source_receipt_sha256: str,
) -> TypedD81LogitBatch:
    """Bind D81 outputs to exact raw288 query rows before canonical fusion."""

    class_registry = _classes(classes)
    query = np.asarray(query_features)
    values = np.asarray(logits)
    if (
        query.dtype != np.float32
        or query.ndim != 2
        or query.shape[1] != FEATURE_DIM
        or not np.isfinite(query).all()
        or values.dtype != np.float32
        or values.shape != (len(query), len(class_registry))
        or not np.isfinite(values).all()
    ):
        raise D100RACGSPRError("typed D81 binding requires float32 logits/query rows")
    query_receipt = _array_receipt(query)
    source_receipt = _require_sha256(source_receipt_sha256, "typed D81 source receipt")
    payload = _typed_d81_batch_payload(
        classes=class_registry,
        k_shot=int(k_shot),
        logits=values,
        query_feature_receipt=query_receipt,
        source_schema=str(source_schema),
        source_receipt_sha256=source_receipt,
    )
    return TypedD81LogitBatch(
        classes=class_registry,
        k_shot=int(k_shot),
        logits_fp32=values,
        query_feature_receipt=query_receipt,
        source_schema=str(source_schema),
        source_receipt_sha256=source_receipt,
        batch_receipt_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class D100CanonicalFusionResult:
    d81_probability_fp32: np.ndarray
    student_t_probability_fp32: np.ndarray
    d99_probability_fp32: np.ndarray
    ridge_probability_fp32: np.ndarray | None
    fused_probability_fp32: np.ndarray
    prediction: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        arrays = (
            self.d81_probability_fp32,
            self.student_t_probability_fp32,
            self.d99_probability_fp32,
            self.fused_probability_fp32,
        )
        ridge = self.ridge_probability_fp32
        if (
            any(np.asarray(value).dtype != np.float32 for value in arrays)
            or any(np.asarray(value).shape != np.asarray(arrays[0]).shape for value in arrays)
            or self.prediction.shape != (len(arrays[0]),)
            or (
                ridge is not None
                and np.asarray(ridge).shape != np.asarray(arrays[0]).shape
            )
            or any(not np.isfinite(np.asarray(value)).all() for value in arrays)
            or any(np.any(np.asarray(value) < 0.0) for value in arrays)
            or any(
                not np.allclose(np.sum(np.asarray(value), axis=1), 1.0, atol=2e-6)
                for value in arrays
            )
            or (
                ridge is not None
                and (
                    np.asarray(ridge).dtype != np.float32
                    or not np.isfinite(np.asarray(ridge)).all()
                    or np.any(np.asarray(ridge) < 0.0)
                    or not np.allclose(
                        np.sum(np.asarray(ridge), axis=1), 1.0, atol=2e-6
                    )
                )
            )
        ):
            raise D100RACGSPRError("canonical fusion result shape drift")
        for name in (
            "d81_probability_fp32",
            "student_t_probability_fp32",
            "d99_probability_fp32",
            "fused_probability_fp32",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float32))
        if self.ridge_probability_fp32 is not None:
            object.__setattr__(
                self,
                "ridge_probability_fp32",
                _readonly(self.ridge_probability_fp32, np.float32),
            )
        prediction = np.asarray(self.prediction, dtype=str).copy()
        prediction.setflags(write=False)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def canonical_fuse_typed_d81_d99_d100(
    state: D100SimplexRidgeState,
    bank: d99.TypedINT8MetricKernelBank,
    typed_d81: TypedD81LogitBatch,
    query_features: np.ndarray,
    *,
    evaluate_complementarity_branch: bool = False,
) -> D100CanonicalFusionResult:
    """Apply the sole D81→D99→D100 probability formula.

    ``evaluate_complementarity_branch`` is a Phase1 diagnostic flag.  It may
    evaluate the ridge branch when alpha is zero so LODO can audit rescue, but
    never changes the returned fused probability.
    """

    if not _query_ready_state(state):
        raise D100RACGSPRError("canonical fusion state verification failed")
    if type(bank) is not d99.TypedINT8MetricKernelBank:
        raise D100RACGSPRError("canonical fusion requires an exact typed D99 bank")
    if type(typed_d81) is not TypedD81LogitBatch:
        raise D100RACGSPRError("canonical fusion requires an exact typed D81 batch")
    if (
        bank.bank_receipt_sha256 != state.d99_bank_receipt_sha256
        or bank.classes != state.classes
        or bank.metric.k_shot != state.k_shot
        or bank.config.lock_digest != state.d99_lock_digest
        or bank.deployment_status != state.source_bank_deployment_status
        or typed_d81.classes != state.classes
        or typed_d81.k_shot != state.k_shot
    ):
        raise D100RACGSPRError("canonical D81/D99/D100 binding drift")
    query = np.asarray(query_features)
    if (
        query.dtype != np.float32
        or query.ndim != 2
        or query.shape != (len(typed_d81.logits_fp32), FEATURE_DIM)
        or not np.isfinite(query).all()
        or _array_receipt(query) != dict(typed_d81.query_feature_receipt)
    ):
        raise D100RACGSPRError("canonical query/typed D81 receipt drift")
    d81_probability = _softmax(typed_d81.logits_fp32, 1.0)
    student_t_probability = _softmax(
        d99.score_metric_kernel_raw_logits(bank, query), state.d99_temperature
    )
    eta = float(bank.eta_phase1_locked)
    if eta == 0.0:
        probability99 = _readonly(d81_probability, np.float32)
    elif eta == 1.0:
        probability99 = _readonly(student_t_probability, np.float32)
    else:
        probability99 = _readonly(
            (1.0 - eta) * d81_probability.astype(np.float64)
            + eta * student_t_probability.astype(np.float64),
            np.float32,
        )
    evaluate_ridge = bool(state.alpha > 0.0 or evaluate_complementarity_branch)
    ridge_probability = (
        _softmax(score_simplex_ridge_logits(state, query), state.temperature)
        if evaluate_ridge
        else None
    )
    if state.alpha == 0.0:
        fused = _readonly(probability99, np.float32)
    elif state.alpha == 1.0:
        if ridge_probability is None:
            raise AssertionError("alpha=1 requires ridge probability")
        fused = _readonly(ridge_probability, np.float32)
    else:
        if ridge_probability is None:
            raise AssertionError("nonzero alpha requires ridge probability")
        fused = _readonly(
            (1.0 - state.alpha) * probability99.astype(np.float64)
            + state.alpha * ridge_probability.astype(np.float64),
            np.float32,
        )
    prediction = np.asarray(state.classes, dtype=str)[np.argmax(fused, axis=1)]
    audit = {
        "schema": CANONICAL_FUSION_SCHEMA,
        "typed_d81_batch_receipt_sha256": typed_d81.batch_receipt_sha256,
        "typed_d81_source_schema": typed_d81.source_schema,
        "typed_d81_source_receipt_sha256": typed_d81.source_receipt_sha256,
        "d99_bank_receipt_sha256": bank.bank_receipt_sha256,
        "d100_state_receipt_sha256": state.state_receipt_sha256,
        "eta_phase1_locked": eta,
        "d99_temperature_phase1_locked": float(state.d99_temperature),
        "alpha_phase1_locked": float(state.alpha),
        "ridge_temperature_phase1_locked": float(state.temperature),
        "formula": (
            "p99=(1-eta_K)*softmax(D81)+eta_K*softmax(StudentT/T99);"
            "p100=(1-alpha_K)*p99+alpha_K*softmax(ridge/TR)"
        ),
        "student_t_branch_evaluated": True,
        "ridge_branch_evaluated": evaluate_ridge,
        "complementarity_diagnostic_requested": bool(evaluate_complementarity_branch),
        "query_batch_dependency": False,
        "query_state_updates": 0,
        "formal_phase2_eligible": False,
    }
    return D100CanonicalFusionResult(
        d81_probability_fp32=d81_probability,
        student_t_probability_fp32=student_t_probability,
        d99_probability_fp32=probability99,
        ridge_probability_fp32=ridge_probability,
        fused_probability_fp32=fused,
        prediction=prediction,
        audit=audit,
    )


def fuse_with_typed_d99_bank(
    state: D100SimplexRidgeState,
    bank: d99.TypedINT8MetricKernelBank,
    query_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Internally score an exact typed D99 bank and apply the locked mixture."""

    if not _query_ready_state(state):
        raise D100RACGSPRError("D100 fusion state verification failed")
    if type(bank) is not d99.TypedINT8MetricKernelBank:
        raise D100RACGSPRError("D100 fusion requires an exact typed D99 bank")
    if (
        bank.bank_receipt_sha256 != state.d99_bank_receipt_sha256
        or bank.classes != state.classes
        or bank.metric.k_shot != state.k_shot
        or bank.config.lock_digest != state.d99_lock_digest
        or bank.deployment_status != state.source_bank_deployment_status
    ):
        raise D100RACGSPRError("D100 fusion typed D99 bank/state binding drift")
    query = np.asarray(query_features)
    if (
        query.ndim != 2
        or query.shape[1] != FEATURE_DIM
        or not np.isfinite(query).all()
    ):
        raise D100RACGSPRError("D100 query must be finite [N,288]")
    # This canonical D99 entry verifies the exact bank type, numeric resource,
    # receipt, and deployment state before computing its typed raw logits.
    d99_logits = d99.score_metric_kernel_raw_logits(bank, query)
    base = _softmax(d99_logits, state.d99_temperature)
    if state.alpha == 0.0:
        fused = _readonly(base, np.float32)
        ridge_used = False
    else:
        ridge = _softmax(score_simplex_ridge_logits(state, query), state.temperature)
        fused = _readonly(
            (1.0 - state.alpha) * base.astype(np.float64)
            + state.alpha * ridge.astype(np.float64),
            np.float32,
        )
        ridge_used = True
    prediction = np.asarray(state.classes, dtype=object)[np.argmax(fused, axis=1)]
    audit = MappingProxyType(
        {
            "schema": SCHEMA,
            "alpha_phase1_locked": float(state.alpha),
            "ridge_temperature_phase1_locked": float(state.temperature),
            "d99_temperature_phase1_locked": float(state.d99_temperature),
            "d99_bank_receipt_sha256": bank.bank_receipt_sha256,
            "d99_bank_exact_type_verified": True,
            "d99_canonical_typed_scorer_used": True,
            "ridge_branch_evaluated": ridge_used,
            "same_formula_all_registered_classes": True,
            "query_batch_dependency": False,
            "query_state_updates": 0,
            "formal_phase2_eligible": False,
            "authority_status": "BLOCKED_MISSING_EXTERNAL_AUTHORITY_OR_PHASE1_LODO_RESCUE_RECEIPT",
        }
    )
    return fused, prediction.astype(str), audit


def complementarity_audit(
    d99_probabilities: np.ndarray,
    ridge_probabilities: np.ndarray,
    true_class_indices: Sequence[int],
) -> Mapping[str, Any]:
    """Compute local rescue/error overlap; this never grants Phase1 authority."""

    base = np.asarray(d99_probabilities, dtype=np.float64)
    ridge = np.asarray(ridge_probabilities, dtype=np.float64)
    truth = np.asarray(true_class_indices, dtype=np.int64)
    if (
        base.ndim != 2
        or ridge.shape != base.shape
        or truth.shape != (len(base),)
        or base.shape[1] < 2
        or not np.isfinite(base).all()
        or not np.isfinite(ridge).all()
        or np.any(base < 0.0)
        or np.any(ridge < 0.0)
        or not np.allclose(np.sum(base, axis=1), 1.0, atol=2e-6)
        or not np.allclose(np.sum(ridge, axis=1), 1.0, atol=2e-6)
        or np.any(truth < 0)
        or np.any(truth >= base.shape[1])
    ):
        raise D100RACGSPRError("D100 complementarity inputs drift")
    base_correct = np.argmax(base, axis=1) == truth
    ridge_correct = np.argmax(ridge, axis=1) == truth
    base_wrong = ~base_correct
    ridge_wrong = ~ridge_correct
    rescue_ridge = int(np.sum(ridge_correct & base_wrong))
    rescue_base = int(np.sum(base_correct & ridge_wrong))
    disagreement = int(np.sum(np.argmax(base, axis=1) != np.argmax(ridge, axis=1)))
    union = base_correct | ridge_correct
    return MappingProxyType(
        {
            "schema": COMPLEMENTARITY_SCHEMA,
            "row_count": int(len(base)),
            "disagreement_count": disagreement,
            "ridge_correct_when_d99_wrong_count": rescue_ridge,
            "d99_correct_when_ridge_wrong_count": rescue_base,
            "bidirectional_rescue_nonzero": bool(rescue_ridge > 0 and rescue_base > 0),
            "d99_accuracy": float(np.mean(base_correct)),
            "ridge_accuracy": float(np.mean(ridge_correct)),
            "oracle_union_accuracy": float(np.mean(union)),
            "formal_phase1_rescue_receipt": False,
            "target_or_query_selection_authorized": False,
        }
    )


def audit_combined_wire_budget(
    *,
    d100_state_wire: bytes,
    d99_bank_wire: bytes,
    typed_d81_wire: bytes,
    ground_bundle_wire: bytes,
) -> Mapping[str, Any]:
    """Count exact supplied bytes; no caller-supplied size field is trusted."""

    values = {
        "d100_state": bytes(d100_state_wire),
        "d99_bank": bytes(d99_bank_wire),
        "typed_d81": bytes(typed_d81_wire),
        "ground_bundle": bytes(ground_bundle_wire),
    }
    if any(len(value) == 0 for value in values.values()):
        raise D100RACGSPRError("D100 combined budget requires four non-empty wires")
    sizes = {f"{name}_bytes": len(value) for name, value in values.items()}
    roots = {f"{name}_sha256": _sha256_bytes(value) for name, value in values.items()}
    total = int(sum(sizes.values()))
    return MappingProxyType(
        {
            "schema": COMBINED_RESOURCE_SCHEMA,
            **sizes,
            **roots,
            "known_component_wire_bytes": total,
            "state_limit_bytes": STATE_LIMIT_BYTES,
            "known_components_below_256kib": bool(total <= STATE_LIMIT_BYTES),
            "all_component_sizes_computed_from_bytes": True,
            "complete_combined_state_upper_bound_available": False,
            "under_256kib_formal_claim": False,
            "formal_combined_resource_claim": False,
            "authority_status": (
                "DEVELOPMENT_KNOWN_COMPONENT_BYTES_ONLY_"
                "MISSING_COMPLETE_AUTHORITY_AND_GROUND_RECEIPTS"
            ),
        }
    )


def predict_formal(*_args: Any, **_kwargs: Any) -> None:
    """Formal inference is unavailable until independently provisioned roots exist."""

    raise D100RACGSPRError(
        "D100 formal prediction is blocked: external authority and Phase1 LODO rescue receipt are absent"
    )


__all__ = [
    "ALLOWED_K",
    "D100RACGSPRError",
    "D100CanonicalFusionResult",
    "D100SimplexRidgeState",
    "Phase1D100Lock",
    "TypedD81LogitBatch",
    "STATE_LIMIT_BYTES",
    "audit_combined_wire_budget",
    "build_simplex_ridge_state",
    "bind_typed_d81_logits",
    "canonical_fuse_typed_d81_d99_d100",
    "complementarity_audit",
    "fuse_with_typed_d99_bank",
    "generalized_precision_sqrt_transform",
    "predict_formal",
    "score_simplex_ridge_logits",
    "serialize_simplex_ridge_state",
]
