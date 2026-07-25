"""D104 support-only angular-grid INT8 quantization for typed Student-t qKNN.

The public builders deliberately have no query argument. They accept only one
balanced target-support set, its opaque labels/registry, and the immutable
Phase1 qKNN lock. The default D103 quantizer remains unchanged.
"""

from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .stage2_zid_student_t_qknn import (
    INT8_MAX,
    MAX_WIRE_BYTES,
    Z_DIM,
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    ZIDStudentTQKNNError,
    _array_receipt,
    _bank_payload,
    _canonical_order,
    _canonical_sha256,
    _identity_class_scales,
    _readonly,
    _registry,
    audit_runtime_state,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    serialize_typed_zid_runtime_state,
)


CANDIDATE_ID = "D104-R1-ANGQ-RXID-MB4"
ANGQ_SCHEMA = "cvs.phase2.d104_r1.angq.quantization_audit.v1"
RESOURCE_SCHEMA = "cvs.phase2.d104_r1.angq.resource_receipt.v1"
FACTOR_START = 0.75
FACTOR_STEP = 0.005
FACTOR_COUNT = 101
FACTORS = np.asarray(
    [FACTOR_START + FACTOR_STEP * index for index in range(FACTOR_COUNT)],
    dtype=np.float64,
)
FACTOR_ONE_INDEX = 50
MAC_PER_CANDIDATE = 2 * Z_DIM
VECTOR_ELEMENTWISE_OPS_PER_CANDIDATE = 4 * Z_DIM
ANGQ_MAC_PER_SUPPORT = FACTOR_COUNT * MAC_PER_CANDIDATE
ANGQ_VECTOR_ELEMENTWISE_OPS_PER_SUPPORT = (
    FACTOR_COUNT * VECTOR_ELEMENTWISE_OPS_PER_CANDIDATE
)
PEAK_TEMPORARY_BYTES_BOUND = 16 * 1024

_FACTORS_SHA256 = hashlib.sha256(
    np.ascontiguousarray(FACTORS).tobytes(order="C")
).hexdigest()


class D104ANGQError(ZIDStudentTQKNNError):
    """Raised when the frozen D104 quantization or resource contract drifts."""


def _normalized_float32_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D104ANGQError(
            f"ANGQ normalized support must be finite float32 [N,{Z_DIM}]"
        )
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    if (
        not np.isfinite(norms).all()
        or np.any(norms <= 0.0)
        or np.any(np.abs(norms - 1.0) > 2.0e-6)
    ):
        raise D104ANGQError("ANGQ input must already be normalized exactly once")
    return np.ascontiguousarray(rows)


def _decode_candidate(
    normalized_row: np.ndarray,
    factor: float,
) -> tuple[np.float16, np.ndarray, np.ndarray, float]:
    """Quantize one already-normalized row without normalizing it again."""

    row = np.asarray(normalized_row)
    if (
        row.dtype != np.float32
        or row.ndim != 1
        or row.shape != (Z_DIM,)
        or not np.isfinite(row).all()
        or not math.isfinite(float(factor))
        or float(factor) <= 0.0
    ):
        raise D104ANGQError("finite normalized float32 row and factor required")
    base_scale = float(np.max(np.abs(row))) / INT8_MAX
    scale16 = np.float16(
        max(
            base_scale * float(factor),
            float(np.finfo(np.float16).tiny),
        )
    )
    if not np.isfinite(scale16) or scale16 <= 0.0:
        raise D104ANGQError("ANGQ scale underflow or non-finite value")
    code = np.clip(
        np.rint(row / np.float32(scale16)),
        -127,
        127,
    ).astype(np.int8)
    if np.any(code == np.int8(-128)):
        raise D104ANGQError("ANGQ emitted forbidden INT8 code -128")
    raw = code.astype(np.float32) * np.float32(scale16)
    raw64 = raw.astype(np.float64)
    norm = float(np.linalg.norm(raw64))
    if not np.isfinite(raw).all() or not math.isfinite(norm) or norm <= 0.0:
        raise D104ANGQError("ANGQ reconstruction is non-finite or zero norm")
    decoded = np.asarray(raw64 / norm, dtype=np.float32)
    if not np.isfinite(decoded).all():
        raise D104ANGQError("ANGQ normalized reconstruction is non-finite")
    cosine = float(
        np.dot(row.astype(np.float64), decoded.astype(np.float64))
    )
    if not math.isfinite(cosine):
        raise D104ANGQError("ANGQ reconstruction cosine is non-finite")
    return scale16, code, decoded, cosine


def quantize_d104_angq_normalized_rows(
    normalized_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen 101-point stable-first ANGQ search.

    The caller must supply float32 unit rows. This function never normalizes
    the input; it normalizes only candidate reconstructions.
    """

    rows = _normalized_float32_rows(normalized_support)
    codes = np.empty(rows.shape, dtype=np.int8)
    scales = np.empty(len(rows), dtype=np.float16)
    decoded = np.empty(rows.shape, dtype=np.float32)
    selected_factors = np.empty(len(rows), dtype=np.float64)
    selected_cosines = np.empty(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows):
        best_cosine = -math.inf
        best_scale: np.float16 | None = None
        best_code: np.ndarray | None = None
        best_decoded: np.ndarray | None = None
        best_factor: float | None = None
        c_one: tuple[np.float16, np.ndarray, np.ndarray, float] | None = None
        for factor_index, factor_value in enumerate(FACTORS):
            candidate = _decode_candidate(row, float(factor_value))
            if factor_index == FACTOR_ONE_INDEX:
                c_one = candidate
            candidate_cosine = candidate[3]
            if candidate_cosine > best_cosine:
                best_cosine = candidate_cosine
                best_scale = candidate[0]
                best_code = candidate[1]
                best_decoded = candidate[2]
                best_factor = float(factor_value)
        if (
            best_scale is None
            or best_code is None
            or best_decoded is None
            or best_factor is None
            or c_one is None
        ):
            raise D104ANGQError("ANGQ fixed grid produced no complete row")
        if best_cosine < c_one[3]:
            raise D104ANGQError("ANGQ selected result regressed below c=1")
        codes[row_index] = best_code
        scales[row_index] = best_scale
        decoded[row_index] = best_decoded
        selected_factors[row_index] = best_factor
        selected_cosines[row_index] = best_cosine
    return (
        np.ascontiguousarray(codes),
        np.ascontiguousarray(scales),
        np.ascontiguousarray(decoded),
        np.ascontiguousarray(selected_factors),
        np.ascontiguousarray(selected_cosines),
    )


def _validate_balanced_support(
    normalized: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    config: Phase1ZIDStudentTLock,
) -> tuple[tuple[str, ...], tuple[int, ...], np.ndarray]:
    if type(config) is not Phase1ZIDStudentTLock:
        raise D104ANGQError("ANGQ bank requires an exact Phase1 lock")
    labels = tuple(str(value) for value in support_labels)
    classes = _registry(registered_classes)
    if len(labels) != len(normalized) or any(label not in classes for label in labels):
        raise D104ANGQError("support labels must map to the opaque registry")
    class_map = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
    if any(value < 1 for value in counts):
        raise D104ANGQError("every registered class requires target support")
    if len(set(counts)) != 1 or counts[0] != config.active_k:
        raise D104ANGQError("ANGQ support must match the balanced active K lock")
    return classes, counts, indices


def build_d104_angq_support_bank(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: Phase1ZIDStudentTLock,
) -> TypedINT8ZIDSupportBank:
    """Compile one D104 typed support bank; no query surface exists."""

    normalized = normalize_zid_rows(support_zid)
    classes, counts, indices = _validate_balanced_support(
        normalized,
        support_labels,
        registered_classes,
        config,
    )
    codes, scales, decoded, factors, cosines = (
        quantize_d104_angq_normalized_rows(normalized)
    )
    order = _canonical_order(codes, scales, indices)
    codes = codes[order]
    scales = scales[order]
    decoded = decoded[order]
    indices = indices[order]
    factors = factors[order]
    cosines = cosines[order]
    ordered = np.asarray(normalized, dtype=np.float32)[order]
    class_scales = _identity_class_scales(
        decoded,
        indices,
        len(classes),
        config,
    )
    class_scales16 = np.asarray(class_scales, dtype=np.float16)
    if (
        not np.isfinite(class_scales16).all()
        or np.any(class_scales16 <= 0.0)
    ):
        raise D104ANGQError("ANGQ class bandwidth FP16 closure failed")
    reconstruction_error = np.abs(
        decoded.astype(np.float64) - ordered.astype(np.float64)
    )
    factor_receipt = _array_receipt(factors)
    quantization: dict[str, Any] = {
        "schema": ANGQ_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "feature_space": "z_id160_only",
        "support_only": True,
        "single_received_observation": True,
        "support_rows": int(len(codes)),
        "class_count": int(len(classes)),
        "support_counts": list(counts),
        "per_vector_scale": True,
        "input_normalization_count": 1,
        "candidate_input_renormalization_count": 0,
        "candidate_decoded_normalization": "float64_l2_to_float32",
        "factor_start": FACTOR_START,
        "factor_step": FACTOR_STEP,
        "factor_count": FACTOR_COUNT,
        "factor_grid_sha256": _FACTORS_SHA256,
        "selected_factor_receipt": factor_receipt,
        "selected_factor_min": float(np.min(factors)),
        "selected_factor_max": float(np.max(factors)),
        "selected_factor_mean": float(np.mean(factors)),
        "scale_dtype": np.dtype(np.float16).str,
        "code_dtype": np.dtype(np.int8).str,
        "rounding": "numpy.rint_ties_to_even",
        "factor_tie_break": "ascending_factor_stable_first_strict_greater",
        "quantization_error_mean": float(np.mean(reconstruction_error)),
        "quantization_error_max": float(np.max(reconstruction_error)),
        "reconstruction_cosine_mean": float(np.mean(cosines)),
        "reconstruction_cosine_min": float(np.min(cosines)),
        "class_scale_source": (
            "phase1_locked_shared_h0"
            if config.active_k == 1
            else "angq_decoded_support_only_uniform_class_formula"
        ),
        "class_bandwidth_dtype": np.dtype(np.float16).str,
        "class_count_normalization": "logsumexp_minus_log_Kc",
        "same_formula_all_registered_classes": True,
        "old_new_role_specific_scoring": False,
        "receiver_specific_grid": False,
        "k_specific_grid": False,
        "scene_specific_grid": False,
        "class_specific_grid": False,
        "query_features_used_for_scale": 0,
        "query_rows_used_for_fit": 0,
        "query_truth_read": False,
        "query_state_updates": 0,
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


def _numeric_bank_array_bytes(bank: TypedINT8ZIDSupportBank) -> int:
    return int(
        bank.codes_qint8.nbytes
        + bank.scales_fp16.nbytes
        + bank.class_indices_int16.nbytes
        + bank.class_scales_fp16.nbytes
    )


def audit_d104_angq_resource_delta(
    legacy_bank: TypedINT8ZIDSupportBank,
    angq_bank: TypedINT8ZIDSupportBank,
) -> Mapping[str, Any]:
    """Compare matched legacy/ANGQ typed banks without reading a query."""

    if (
        type(legacy_bank) is not TypedINT8ZIDSupportBank
        or type(angq_bank) is not TypedINT8ZIDSupportBank
        or legacy_bank.classes != angq_bank.classes
        or legacy_bank.support_counts != angq_bank.support_counts
        or legacy_bank.active_k != angq_bank.active_k
        or legacy_bank.config_lock_digest != angq_bank.config_lock_digest
    ):
        raise D104ANGQError("resource audit requires matched typed support banks")
    if dict(angq_bank.quantization_audit).get("schema") != ANGQ_SCHEMA:
        raise D104ANGQError("resource audit ANGQ bank schema drift")
    metric = identity_shared_psd_metric(config=legacy_bank.config)
    legacy_wire = serialize_typed_zid_runtime_state(legacy_bank, metric)
    angq_wire = serialize_typed_zid_runtime_state(angq_bank, metric)
    legacy_runtime = audit_runtime_state(legacy_bank, metric)
    angq_runtime = audit_runtime_state(angq_bank, metric)
    legacy_numeric = _numeric_bank_array_bytes(legacy_bank)
    angq_numeric = _numeric_bank_array_bytes(angq_bank)
    legacy_metadata = len(legacy_wire) - int(
        legacy_runtime["numeric_array_state_bytes"]
    )
    angq_metadata = len(angq_wire) - int(
        angq_runtime["numeric_array_state_bytes"]
    )
    class_count = len(angq_bank.classes)
    active_k = int(angq_bank.active_k)
    support_count = class_count * active_k
    query_mac_before = int(
        legacy_runtime["score_query_variable_matmul_mac_per_query"]
    )
    query_mac_after = int(
        angq_runtime["score_query_variable_matmul_mac_per_query"]
    )
    receipt: dict[str, Any] = {
        "schema": RESOURCE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "registered_class_count": class_count,
        "active_k": active_k,
        "support_count": support_count,
        "factor_count": FACTOR_COUNT,
        "adaptation_mac_per_support": ANGQ_MAC_PER_SUPPORT,
        "adaptation_mac_total": ANGQ_MAC_PER_SUPPORT * support_count,
        "adaptation_mac_formula": (
            "32320*registered_class_count*active_k"
        ),
        "adaptation_vector_elementwise_ops_per_support": (
            ANGQ_VECTOR_ELEMENTWISE_OPS_PER_SUPPORT
        ),
        "adaptation_vector_elementwise_ops_total": (
            ANGQ_VECTOR_ELEMENTWISE_OPS_PER_SUPPORT * support_count
        ),
        "adaptation_vector_elementwise_ops_formula": (
            "64640*registered_class_count*active_k"
        ),
        "scalar_reduction_ops": {
            "maxabs_reduction_comparisons_total": support_count * (Z_DIM - 1),
            "base_scale_divisions_total": support_count,
            "factor_multiplications_total": support_count * FACTOR_COUNT,
            "fp16_scale_casts_total": support_count * FACTOR_COUNT,
            "scale_floor_comparisons_total": support_count * FACTOR_COUNT,
            "decoded_norm_sqrt_total": support_count * FACTOR_COUNT,
            "finite_zero_checks_total": support_count * FACTOR_COUNT,
            "best_candidate_comparisons_total": support_count * FACTOR_COUNT,
        },
        "shared_input_normalization_excluded_from_angq_delta": True,
        "peak_temporary_bytes_upper_bound": 4096,
        "peak_temporary_bytes_gate": PEAK_TEMPORARY_BYTES_BOUND,
        "passes_peak_temporary_bytes_gate": 4096 <= PEAK_TEMPORARY_BYTES_BOUND,
        "numeric_bank_array_bytes_before": legacy_numeric,
        "numeric_bank_array_bytes_after": angq_numeric,
        "numeric_bank_array_bytes_delta": angq_numeric - legacy_numeric,
        "actual_serialized_state_bytes_before": len(legacy_wire),
        "actual_serialized_state_bytes_after": len(angq_wire),
        "actual_serialized_state_bytes_delta": len(angq_wire) - len(legacy_wire),
        "metadata_framing_bytes_before": legacy_metadata,
        "metadata_framing_bytes_after": angq_metadata,
        "metadata_framing_bytes_delta": angq_metadata - legacy_metadata,
        "wire_bytes_gate": MAX_WIRE_BYTES,
        "passes_wire_bytes_gate": len(angq_wire) <= MAX_WIRE_BYTES,
        "query_mac_before": query_mac_before,
        "query_mac_after": query_mac_after,
        "query_mac_delta": query_mac_after - query_mac_before,
        "query_features_used_for_scale": 0,
        "query_truth_read": False,
        "query_state_updates": 0,
    }
    receipt["passes_d104_resource_gate"] = bool(
        receipt["numeric_bank_array_bytes_delta"] == 0
        and receipt["query_mac_delta"] == 0
        and receipt["passes_peak_temporary_bytes_gate"]
        and receipt["passes_wire_bytes_gate"]
    )
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    if not receipt["passes_d104_resource_gate"]:
        raise D104ANGQError("D104 resource gate failed")
    return MappingProxyType(receipt)


def build_matched_legacy_and_d104_banks(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: Phase1ZIDStudentTLock,
) -> tuple[
    TypedINT8ZIDSupportBank,
    TypedINT8ZIDSupportBank,
    Mapping[str, Any],
]:
    """Build the matched head pair and its immutable resource comparison."""

    legacy = build_typed_zid_support_bank(
        support_zid,
        support_labels,
        registered_classes,
        config=config,
    )
    angq = build_d104_angq_support_bank(
        support_zid,
        support_labels,
        registered_classes,
        config=config,
    )
    return legacy, angq, audit_d104_angq_resource_delta(legacy, angq)


__all__ = [
    "ANGQ_MAC_PER_SUPPORT",
    "ANGQ_SCHEMA",
    "ANGQ_VECTOR_ELEMENTWISE_OPS_PER_SUPPORT",
    "CANDIDATE_ID",
    "D104ANGQError",
    "FACTORS",
    "FACTOR_COUNT",
    "RESOURCE_SCHEMA",
    "audit_d104_angq_resource_delta",
    "build_d104_angq_support_bank",
    "build_matched_legacy_and_d104_banks",
    "quantize_d104_angq_normalized_rows",
]
