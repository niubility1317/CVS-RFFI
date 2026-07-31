"""D105 LPO-RC support-only calibration for typed Student-t qKNN.

This module deliberately leaves the typed INT8 bank, its class scales, and
the base Student-t log-kernel untouched.  It learns one finite, zero-centred
class bias from physical leave-one-out target supports when ``K >= 2``.  No
ground/source, lifecycle role, query truth, query batch, quota, or global
assignment surface is present.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .stage2_zid_student_t_qknn import (
    EPSILON,
    Z_DIM,
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    ZIDStudentTQKNNError,
    _array_receipt,
    _canonical_bytes,
    _canonical_order,
    _canonical_sha256,
    _precision_cosine,
    _quantize_rows,
    _readonly,
    _registry,
    _require_sha256,
    _score_with_support,
    _sha256_bytes,
    _verify_bank,
    _verify_metric,
    decode_zid_support_bank,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


LPO_RC_SCHEMA = "cvs.phase2.d105.lpo_rc_qknn.v1"
LPO_RC_RECEIPT_SCHEMA = "cvs.phase2.d105.lpo_rc_qknn.receipt.v1"
P2_SPLIT_HANDLE_SCHEMA = "cvs.phase2.d105.validated_once_split_handle.v1"
P2_PROTOCOL_SCHEMA = "p2_min_v1"
P2_DATA_STATUS = "VALIDATED_ONCE"
LPO_RC_WIRE_MAGIC = b"CVSLPORC\x00\x01"
LPO_RC_NUMERIC_WORKSPACE_BUDGET_BYTES = 16 * 1024 * 1024


class LPORCQKNNError(ZIDStudentTQKNNError):
    """Raised when the D105 pure support-only HEAD contract drifts."""


class LPORCProtocolHandleError(LPORCQKNNError):
    """Raised when the already-validated p2_min_v1 split handle drifts."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TypedValidatedOnceP2SplitHandle:
    """Opaque, already-validated split authority consumed without revalidation."""

    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    support_physical_root_sha256: str
    query_physical_root_sha256: str
    support_query_disjoint: bool
    schema: str = P2_SPLIT_HANDLE_SCHEMA
    protocol_schema: str = P2_PROTOCOL_SCHEMA
    phase2_data_status: str = P2_DATA_STATUS

    def __post_init__(self) -> None:
        if (
            self.schema != P2_SPLIT_HANDLE_SCHEMA
            or self.protocol_schema != P2_PROTOCOL_SCHEMA
            or self.phase2_data_status != P2_DATA_STATUS
            or self.support_query_disjoint is not True
        ):
            raise LPORCProtocolHandleError(
                "split handle must be VALIDATED_ONCE p2_min_v1 and disjoint"
            )
        for name in (
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "support_physical_root_sha256",
            "query_physical_root_sha256",
        ):
            try:
                _require_sha256(getattr(self, name), f"split handle {name}")
            except ZIDStudentTQKNNError as exc:
                raise LPORCProtocolHandleError(str(exc)) from exc

    @property
    def handle_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": self.schema,
                "protocol_schema": self.protocol_schema,
                "phase2_data_status": self.phase2_data_status,
                "capsule_id": self.capsule_id,
                "split_id": self.split_id,
                "validator_receipt_sha256": self.validator_receipt_sha256,
                "support_physical_root_sha256": self.support_physical_root_sha256,
                "query_physical_root_sha256": self.query_physical_root_sha256,
                "support_query_disjoint": self.support_query_disjoint,
            }
        )


def _verify_split_handle(handle: TypedValidatedOnceP2SplitHandle) -> None:
    if type(handle) is not TypedValidatedOnceP2SplitHandle:
        raise LPORCProtocolHandleError(
            "exact typed VALIDATED_ONCE p2_min_v1 split handle required"
        )
    if (
        handle.schema != P2_SPLIT_HANDLE_SCHEMA
        or handle.protocol_schema != P2_PROTOCOL_SCHEMA
        or handle.phase2_data_status != P2_DATA_STATUS
        or handle.support_query_disjoint is not True
    ):
        raise LPORCProtocolHandleError("validated split handle lifecycle drift")
    for name in (
        "capsule_id",
        "split_id",
        "validator_receipt_sha256",
        "support_physical_root_sha256",
        "query_physical_root_sha256",
    ):
        try:
            _require_sha256(getattr(handle, name), f"split handle {name}")
        except ZIDStudentTQKNNError as exc:
            raise LPORCProtocolHandleError(str(exc)) from exc


def _physical_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise LPORCQKNNError(f"{name} must contain unique non-empty physical IDs")
    return result


def validate_lpo_rc_physical_id_disjointness(
    support_physical_ids: Sequence[str],
    query_physical_ids: Sequence[str],
) -> None:
    """Fail closed when any physical record appears in both support and query.

    This audit helper is intentionally separate from scoring: prediction has no
    physical-ID, truth, or role argument and cannot update a fitted state.
    """

    support = _physical_ids(support_physical_ids, "support")
    query = _physical_ids(query_physical_ids, "query")
    overlap = sorted(set(support).intersection(query))
    if overlap:
        raise LPORCQKNNError("support/query physical IDs must be disjoint")


def _logmeanexp(values: np.ndarray) -> float:
    row = np.asarray(values, dtype=np.float64)
    if row.ndim != 1 or len(row) < 1 or not np.isfinite(row).all():
        raise LPORCQKNNError("logmeanexp requires a finite non-empty vector")
    maximum = float(np.max(row))
    return maximum + math.log(float(np.mean(np.exp(row - maximum))))


def _loo_scales_for_one_class(
    decoded_local: np.ndarray,
    config: Phase1ZIDStudentTLock,
) -> np.ndarray:
    """Recompute temporary K-1 scales with the base support-only formula."""

    local = np.asarray(decoded_local, dtype=np.float64)
    k_shot = len(local)
    if local.shape != (k_shot, Z_DIM) or k_shot < 2:
        raise LPORCQKNNError("physical leave-one-out requires at least two supports")
    if k_shot - 1 == 1:
        return np.full(k_shot, config.shared_h0, dtype=np.float64)

    cosine = np.clip(local @ local.T, -1.0, 1.0)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    total_upper = float(np.sum(distance[np.triu_indices(k_shot, 1)]))
    pair_count = (k_shot - 1) * (k_shot - 2) // 2
    if pair_count < 1:
        raise LPORCQKNNError("leave-one-out pair count became invalid")

    values = np.empty(k_shot, dtype=np.float64)
    for index in range(k_shot):
        removed = float(np.sum(distance[index]))
        empirical = (total_upper - removed) / float(pair_count)
        shrunk = (
            empirical + config.scale_prior_strength * config.shared_h0**2
        ) / (1.0 + config.scale_prior_strength)
        values[index] = np.clip(
            math.sqrt(max(shrunk, EPSILON)),
            config.shared_h0 * config.scale_min_ratio,
            config.shared_h0 * config.scale_max_ratio,
        )
    # The deployed bank closes class scales to FP16; do the same for a fair LOO
    # estimate, while never writing the temporary values into that bank.
    return np.asarray(values, dtype=np.float16).astype(np.float64)


def _verify_and_align_support(
    bank: TypedINT8ZIDSupportBank,
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    support_physical_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Bind transient FP32 supports/IDs exactly to the bank's canonical rows."""

    normalized = normalize_zid_rows(support_zid)
    labels = tuple(str(value) for value in support_labels)
    classes = _registry(registered_classes)
    physical_ids = _physical_ids(support_physical_ids, "support")
    if (
        classes != bank.classes
        or len(labels) != len(normalized)
        or len(physical_ids) != len(normalized)
        or any(label not in classes for label in labels)
    ):
        raise LPORCQKNNError("support registry/labels/physical IDs do not bind the bank")

    class_map = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
    if counts != bank.support_counts:
        raise LPORCQKNNError("support counts do not bind the typed bank")

    # D105 uses the unmodified base typed bank.  Reconstruct the exact canonical
    # row order to bind physical IDs before raw FP32 supports are discarded.
    codes, scales, _ = _quantize_rows(normalized)
    order = _canonical_order(codes, scales, indices)
    if (
        not np.array_equal(codes[order], bank.codes_qint8)
        or not np.array_equal(scales[order], bank.scales_fp16)
        or not np.array_equal(indices[order], bank.class_indices_int16)
    ):
        raise LPORCQKNNError(
            "support rows do not reproduce the exact base INT8 bank canonical order"
        )
    return (
        np.ascontiguousarray(normalized[order], dtype=np.float64),
        tuple(physical_ids[index] for index in order),
    )


def _metric_reconstruction_error_by_class(
    original: np.ndarray,
    decoded: np.ndarray,
    class_indices: np.ndarray,
    class_count: int,
    metric: TypedSharedPSDMetric,
) -> np.ndarray:
    """Compute e_c in the same squared metric distance used for scoring."""

    if original.shape != decoded.shape:
        raise LPORCQKNNError("original/decoded support shapes drift")
    values = np.empty(len(original), dtype=np.float64)
    for index in range(len(original)):
        cosine = float(
            _precision_cosine(
                original[index : index + 1], decoded[index : index + 1], metric
            )[0, 0]
        )
        values[index] = max(2.0 * (1.0 - cosine), 0.0)
    result = np.empty(class_count, dtype=np.float64)
    for class_index in range(class_count):
        local = values[class_indices == class_index]
        if len(local) < 1 or not np.isfinite(local).all():
            raise LPORCQKNNError("class reconstruction error became invalid")
        result[class_index] = float(np.mean(local))
    return result


def _physical_loo_statistics(
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    original: np.ndarray,
    decoded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return e, u, delta and within-class LOO margin dispersion."""

    class_count = len(bank.classes)
    k_shot = bank.active_k
    if k_shot < 2:
        raise LPORCQKNNError("physical LOO is unavailable for K=1")
    class_indices = bank.class_indices_int16
    e_values = _metric_reconstruction_error_by_class(
        original,
        decoded,
        class_indices,
        class_count,
        metric,
    )
    h_values = bank.class_scales_fp16.astype(np.float64)
    u_values = (np.square(h_values) + e_values) / float(k_shot - 1)
    delta_values = np.empty(class_count, dtype=np.float64)
    spread_values = np.empty(class_count, dtype=np.float64)

    for class_index in range(class_count):
        local_rows = np.flatnonzero(class_indices == class_index)
        if len(local_rows) != k_shot:
            raise LPORCQKNNError("balanced K drift during physical LOO")
        loo_scales = _loo_scales_for_one_class(decoded[local_rows], bank.config)
        margins = np.empty(k_shot, dtype=np.float64)
        for local_index, global_index in enumerate(local_rows):
            keep = np.ones(len(decoded), dtype=bool)
            keep[global_index] = False
            counts = list(bank.support_counts)
            counts[class_index] -= 1
            if counts[class_index] < 1:
                raise LPORCQKNNError("physical LOO removed an entire class")
            scales = h_values.copy()
            scales[class_index] = loo_scales[local_index]
            logits = _score_with_support(
                support=decoded[keep],
                class_indices=class_indices[keep],
                support_counts=tuple(counts),
                class_scales=scales,
                query=original[global_index : global_index + 1],
                config=bank.config,
                metric=metric,
            )[0].astype(np.float64)
            competitor = np.delete(logits, class_index)
            margins[local_index] = float(logits[class_index]) - _logmeanexp(
                competitor
            )
        delta_values[class_index] = float(np.mean(margins))
        spread_values[class_index] = float(
            math.sqrt(float(np.mean(np.square(margins - delta_values[class_index]))))
        )
    if not (
        np.isfinite(e_values).all()
        and np.isfinite(u_values).all()
        and np.isfinite(delta_values).all()
        and np.isfinite(spread_values).all()
    ):
        raise LPORCQKNNError("physical LOO statistics became non-finite")
    return e_values, u_values, delta_values, spread_values


def _lpo_rc_bias(
    active_k: int,
    uncertainty: np.ndarray,
    delta: np.ndarray,
    spread: np.ndarray,
) -> np.ndarray:
    if active_k == 1:
        return np.zeros(len(delta), dtype=np.float64)
    if active_k < 2:
        raise LPORCQKNNError("active K must be positive")
    s_bar = float(np.median(spread))
    if not math.isfinite(s_bar) or s_bar <= EPSILON:
        return np.zeros(len(delta), dtype=np.float64)
    u_bar = float(np.mean(uncertainty))
    if not math.isfinite(u_bar) or u_bar <= 0.0:
        raise LPORCQKNNError("LOO uncertainty mean became invalid")
    delta_bar = float(np.mean(delta))
    reliability = (
        u_bar / (u_bar + uncertainty + EPSILON)
    ) * np.tanh((delta_bar - delta) / (s_bar + EPSILON))
    bias = ((active_k - 1.0) / float(active_k)) * s_bar * (
        reliability - float(np.mean(reliability))
    )
    if not np.isfinite(bias).all():
        raise LPORCQKNNError("LPO-RC bias became non-finite")
    return np.asarray(bias, dtype=np.float64)


def _metric_fit_matmul_mac(
    *,
    query_rows: int,
    support_rows: int,
    metric_rank: int,
) -> int:
    """Exact matrix-product count for the current _precision_cosine routine."""

    if metric_rank == 0:
        return int(query_rows * support_rows * Z_DIM)
    return int(
        query_rows * support_rows * Z_DIM
        + query_rows * metric_rank * Z_DIM
        + support_rows * metric_rank * Z_DIM
        + query_rows * support_rows * metric_rank
    )


def _base_kernel_scalar_logic_counts(
    *, class_count: int, active_k: int
) -> dict[str, int]:
    """Logical scalar counts for one query in current _score_with_support."""

    support_rows = class_count * active_k
    return {
        "squared_distance_subtract_ops": support_rows,
        "squared_distance_multiply_ops": support_rows,
        "squared_distance_clamp_ops": support_rows,
        "volume_log_ops": class_count,
        "volume_multiply_ops": 2 * class_count,
        "radial_h_square_multiply_ops": class_count,
        "radial_nu_scale_multiply_ops": class_count,
        "radial_divide_ops": support_rows,
        "radial_log1p_ops": support_rows,
        "radial_coefficient_add_ops": class_count,
        "radial_coefficient_scalar_multiply_ops": class_count,
        "radial_elementwise_multiply_ops": support_rows,
        "kernel_add_subtract_ops": support_rows,
        "logsumexp_max_comparisons": class_count * (active_k - 1),
        "logsumexp_center_subtract_ops": support_rows,
        "logsumexp_exp_ops": support_rows,
        "logsumexp_sum_add_ops": class_count * (active_k - 1),
        "logsumexp_log_ops": class_count,
        "logsumexp_output_add_ops": class_count,
        "log_k_ops": class_count,
        "log_k_subtract_ops": class_count,
    }


def _state_wire_bytes(
    *,
    classes: tuple[str, ...],
    active_k: int,
    class_scales: np.ndarray,
    bias: np.ndarray,
    bank_receipt_sha256: str,
    metric_receipt_sha256: str,
    config_lock_digest: str,
    split_handle_digest: str,
) -> bytes:
    header = {
        "schema": LPO_RC_SCHEMA,
        "classes": list(classes),
        "active_k": int(active_k),
        "bank_receipt_sha256": bank_receipt_sha256,
        "metric_receipt_sha256": metric_receipt_sha256,
        "config_lock_digest": config_lock_digest,
        "split_handle_digest": split_handle_digest,
        "class_scales_fp16": _array_receipt(class_scales),
        "bias_fp16": _array_receipt(bias),
    }
    header_bytes = _canonical_bytes(header)
    return b"".join(
        (
            LPO_RC_WIRE_MAGIC,
            struct.pack("<Q", len(header_bytes)),
            header_bytes,
            np.ascontiguousarray(class_scales, dtype=np.float16).tobytes(order="C"),
            np.ascontiguousarray(bias, dtype=np.float16).tobytes(order="C"),
        )
    )


def _normalization_named_allocations(row_count: int) -> dict[str, int]:
    """Named numeric arrays in one current normalize_zid_rows invocation."""

    row_fp32 = row_count * Z_DIM * 4
    row_fp64 = row_count * Z_DIM * 8
    scalar_fp64 = row_count * 8
    return {
        "phase_input_fp32": row_fp32,
        "finite_contiguous_fp32": row_fp32,
        "rows_cast_fp64": row_fp64,
        "norm_square_temp_fp64": row_fp64,
        "norm_reduction_temp_fp64": scalar_fp64,
        "norm_output_fp64": scalar_fp64,
        "division_output_fp64": row_fp64,
        "readonly_ascontiguous_fp32": row_fp32,
        "readonly_copy_fp32": row_fp32,
    }


def _numeric_workspace_receipt(
    *, class_count: int, active_k: int, metric_rank: int
) -> dict[str, Any]:
    """Conservative named-allocation bound for the current NumPy implementation.

    This is deliberately not a measured process-RSS or allocator peak.  Each
    phase sums the input/output arrays and every named NumPy temporary that can
    occur along its maximum-sized current path.  Potential aliases and buffers
    that may be released earlier are still counted, so the result cannot be
    reduced by implementation-dependent aliasing or allocator reuse.
    """

    support_rows = class_count * active_k
    right_rows = max(support_rows - 1, 0)
    rank = metric_rank

    normalize_support = _normalization_named_allocations(support_rows)

    decoded_normalize = _normalization_named_allocations(support_rows)
    decoded_normalize.pop("phase_input_fp32")
    quantize_binding = {
        "outer_support_input_fp32": support_rows * Z_DIM * 4,
        "normalized_support_input_fp32": support_rows * Z_DIM * 4,
        "bank_codes_input_int8": support_rows * Z_DIM,
        "bank_scales_input_fp16": support_rows * 2,
        "bank_class_indices_input_int16": support_rows * 2,
        "bank_class_scales_input_fp16": class_count * 2,
        "quantize_codes_int8": support_rows * Z_DIM,
        "quantize_scales_fp16": support_rows * 2,
        "quantize_decoded_fp32": support_rows * Z_DIM * 4,
        "row_abs_temp_fp32": Z_DIM * 4,
        "row_divide_temp_fp32": Z_DIM * 4,
        "row_rint_temp_fp32": Z_DIM * 4,
        "row_clip_temp_fp32": Z_DIM * 4,
        "row_code_temp_int8": Z_DIM,
        "row_decode_cast_temp_fp32": Z_DIM * 4,
        "row_decode_multiply_temp_fp32": Z_DIM * 4,
        **{
            f"decoded_normalize_{name}": size
            for name, size in decoded_normalize.items()
        },
        "support_class_indices_int16": support_rows * 2,
        "canonical_order_int64": support_rows * 8,
        "ordered_codes_compare_int8": support_rows * Z_DIM,
        "ordered_scales_compare_fp16": support_rows * 2,
        "ordered_indices_compare_int16": support_rows * 2,
        "ordered_normalized_index_fp32": support_rows * Z_DIM * 4,
        "ordered_original_output_fp64": support_rows * Z_DIM * 8,
    }

    metric_inputs = {
        "metric_basis_codes_input_int8": rank * Z_DIM,
        "metric_basis_scales_input_fp16": rank * 2,
        "metric_attenuation_input_fp16": rank * 2,
    }
    if active_k == 1:
        loo = {
            "ordered_original_fp64": support_rows * Z_DIM * 8,
            "decoded_support_fp64": support_rows * Z_DIM * 8,
            "bank_class_indices_input_int16": support_rows * 2,
            "bank_class_scales_input_fp16": class_count * 2,
            **metric_inputs,
            "reconstruction_error_values_fp64": support_rows * 8,
            "class_statistics_fp64": class_count * 5 * 8,
            "bias_fp64": class_count * 8,
            "bias_readonly_ascontiguous_fp16": class_count * 2,
            "bias_readonly_copy_fp16": class_count * 2,
            "class_scales_readonly_ascontiguous_fp16": class_count * 2,
            "class_scales_readonly_copy_fp16": class_count * 2,
        }
    else:
        pair_count = active_k * (active_k - 1) // 2
        precision = {
            "precision_left_input_fp64": Z_DIM * 8,
            "precision_right_input_fp64": right_rows * Z_DIM * 8,
            "precision_raw_dot_fp64": right_rows * 8,
            "precision_raw_dot_clipped_fp64": right_rows * 8,
        }
        if rank:
            precision.update(
                {
                    "metric_basis_decode_fp64": rank * Z_DIM * 8,
                    "metric_attenuation_cast_fp64": rank * 8,
                    "metric_left_projection_fp64": rank * 8,
                    "metric_right_projection_fp64": right_rows * rank * 8,
                    "metric_projection_product_fp64": right_rows * rank * 8,
                    "metric_projection_weighted_fp64": right_rows * rank * 8,
                    "metric_projection_correction_fp64": right_rows * 8,
                    "metric_numerator_fp64": right_rows * 8,
                    "left_square_fp64": Z_DIM * 8,
                    "left_projection_square_fp64": rank * 8,
                    "left_projection_weighted_fp64": rank * 8,
                    "right_square_fp64": right_rows * Z_DIM * 8,
                    "right_projection_square_fp64": right_rows * rank * 8,
                    "right_projection_weighted_fp64": right_rows * rank * 8,
                    "left_quadratic_fp64": 8,
                    "right_quadratic_fp64": right_rows * 8,
                    "left_norm_fp64": 8,
                    "right_norm_fp64": right_rows * 8,
                    "norm_outer_product_fp64": right_rows * 8,
                    "precision_division_fp64": right_rows * 8,
                    "precision_result_clipped_fp64": right_rows * 8,
                }
            )
        loo = {
            "ordered_original_fp64": support_rows * Z_DIM * 8,
            "decoded_support_fp64": support_rows * Z_DIM * 8,
            "bank_codes_input_int8": support_rows * Z_DIM,
            "bank_scales_input_fp16": support_rows * 2,
            "bank_class_indices_input_int16": support_rows * 2,
            "bank_class_scales_input_fp16": class_count * 2,
            **metric_inputs,
            "reconstruction_error_values_fp64": support_rows * 8,
            "reconstruction_error_class_output_fp64": class_count * 8,
            "class_scale_cast_fp64": class_count * 8,
            "class_scale_square_fp64": class_count * 8,
            "uncertainty_sum_fp64": class_count * 8,
            "uncertainty_output_fp64": class_count * 8,
            "delta_output_fp64": class_count * 8,
            "spread_output_fp64": class_count * 8,
            "local_rows_int64": active_k * 8,
            "local_decoded_index_fp64": active_k * Z_DIM * 8,
            "class_pairwise_cosine_fp64": active_k * active_k * 8,
            "class_pairwise_cosine_clipped_fp64": active_k * active_k * 8,
            "class_pairwise_distance_subtract_fp64": active_k * active_k * 8,
            "class_pairwise_distance_scaled_fp64": active_k * active_k * 8,
            "class_pairwise_distance_fp64": active_k * active_k * 8,
            "class_upper_triangle_row_indices_int64": pair_count * 8,
            "class_upper_triangle_col_indices_int64": pair_count * 8,
            "class_upper_triangle_values_fp64": pair_count * 8,
            "class_loo_scales_fp64": active_k * 8,
            "class_loo_scales_cast_fp16": active_k * 2,
            "class_loo_scales_recast_fp64": active_k * 8,
            "class_loo_margins_fp64": active_k * 8,
            "loo_keep_bool": support_rows,
            "loo_support_subset_fp64": right_rows * Z_DIM * 8,
            "loo_class_indices_int16": right_rows * 2,
            "loo_class_scales_fp64": class_count * 8,
            **precision,
            "score_distance_subtract_fp64": right_rows * 8,
            "score_distance_scaled_fp64": right_rows * 8,
            "score_distance_fp64": right_rows * 8,
            "largest_class_local_distance_fp64": active_k * 8,
            "largest_class_kernel_division_fp64": active_k * 8,
            "largest_class_kernel_log1p_fp64": active_k * 8,
            "largest_class_kernel_fp64": active_k * 8,
            "largest_class_kernel_centered_fp64": active_k * 8,
            "largest_class_kernel_exp_fp64": active_k * 8,
            "score_columns_retained_fp64": class_count * 8,
            "score_logits_stack_fp64": class_count * 8,
            "score_readonly_ascontiguous_fp32": class_count * 4,
            "score_readonly_copy_fp32": class_count * 4,
            "loo_logits_cast_fp64": class_count * 8,
            "loo_competitor_delete_fp64": (class_count - 1) * 8,
            "logmeanexp_centered_fp64": (class_count - 1) * 8,
            "logmeanexp_exp_fp64": (class_count - 1) * 8,
            "margin_centered_fp64": active_k * 8,
            "margin_square_fp64": active_k * 8,
            "reliability_denominator_fp64": class_count * 8,
            "reliability_ratio_fp64": class_count * 8,
            "reliability_tanh_input_fp64": class_count * 8,
            "reliability_tanh_fp64": class_count * 8,
            "reliability_fp64": class_count * 8,
            "bias_centered_fp64": class_count * 8,
            "bias_fp64": class_count * 8,
            "bias_readonly_ascontiguous_fp16": class_count * 2,
            "bias_readonly_copy_fp16": class_count * 2,
            "class_scales_readonly_ascontiguous_fp16": class_count * 2,
            "class_scales_readonly_copy_fp16": class_count * 2,
        }

    phase_components = {
        "normalize_support": normalize_support,
        "quantize_and_canonical_bind": quantize_binding,
        "physical_loo_max_iteration": loo,
    }
    phase_bytes = {
        name: int(sum(components.values()))
        for name, components in phase_components.items()
    }
    upper_bound = max(phase_bytes.values())
    passes_gate = upper_bound <= LPO_RC_NUMERIC_WORKSPACE_BUDGET_BYTES
    return {
        "semantics": (
            "deterministic_named_numeric_allocation_conservative_upper_bound;"
            "not_measured_allocator_or_process_peak"
        ),
        "rule": (
            "sum input/output and named NumPy numeric allocations on each "
            "maximum-sized current phase path, including possible aliases and "
            "buffers that may be released earlier; take the maximum phase sum"
        ),
        "scope": (
            "current LPO-RC Python/NumPy arrays and directly called qKNN "
            "normalization, quantization, precision-cosine and scoring helpers"
        ),
        "excluded": (
            "Python object headers, JSON/receipt bytes, BLAS-internal scratch, "
            "interpreter and allocator bookkeeping"
        ),
        "phase_components": phase_components,
        "phase_bytes": phase_bytes,
        "formal_workspace_upper_bound_bytes": upper_bound,
        "formal_workspace_budget_bytes": LPO_RC_NUMERIC_WORKSPACE_BUDGET_BYTES,
        "passes_formal_workspace_gate": passes_gate,
        # Compatibility aliases remain receipt-bound for existing audits.
        "support_binding_components": quantize_binding,
        "support_binding_bytes": phase_bytes["quantize_and_canonical_bind"],
        "loo_components": loo,
        "loo_bytes": phase_bytes["physical_loo_max_iteration"],
        "peak_bytes": upper_bound,
    }


def _receipt_payload(
    *,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    class_scales: np.ndarray,
    bias: np.ndarray,
    canonical_physical_ids: tuple[str, ...],
    e_values: np.ndarray,
    u_values: np.ndarray,
    delta_values: np.ndarray,
    spread_values: np.ndarray,
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> dict[str, Any]:
    class_count = len(bank.classes)
    support_rows = bank.support_row_count
    active_k = bank.active_k
    rank = metric.effective_rank
    loo_enabled = active_k >= 2
    if loo_enabled:
        loo_score_kernel_evaluations = support_rows * (support_rows - 1)
        loo_metric_mac = support_rows * _metric_fit_matmul_mac(
            query_rows=1,
            support_rows=support_rows - 1,
            metric_rank=rank,
        ) + support_rows * _metric_fit_matmul_mac(
            query_rows=1,
            support_rows=1,
            metric_rank=rank,
        )
        loo_scale_mac = class_count * active_k * active_k * Z_DIM
    else:
        loo_score_kernel_evaluations = 0
        loo_metric_mac = 0
        loo_scale_mac = 0
    state_wire = _state_wire_bytes(
        classes=bank.classes,
        active_k=active_k,
        class_scales=class_scales,
        bias=bias,
        bank_receipt_sha256=bank.bank_receipt_sha256,
        metric_receipt_sha256=metric.metric_receipt_sha256,
        config_lock_digest=bank.config_lock_digest,
        split_handle_digest=split_handle.handle_digest,
    )
    workspace = _numeric_workspace_receipt(
        class_count=class_count,
        active_k=active_k,
        metric_rank=rank,
    )
    if not workspace["passes_formal_workspace_gate"]:
        raise LPORCQKNNError(
            "LPO-RC named numeric workspace upper bound exceeds the formal budget"
        )
    return {
        "schema": LPO_RC_RECEIPT_SCHEMA,
        "head_schema": LPO_RC_SCHEMA,
        "bank_receipt_sha256": bank.bank_receipt_sha256,
        "metric_receipt_sha256": metric.metric_receipt_sha256,
        "config_lock_digest": bank.config_lock_digest,
        "split_handle_digest": split_handle.handle_digest,
        "capsule_id": split_handle.capsule_id,
        "split_id": split_handle.split_id,
        "validator_receipt_sha256": split_handle.validator_receipt_sha256,
        "support_physical_root_sha256": (
            split_handle.support_physical_root_sha256
        ),
        "query_physical_root_sha256": split_handle.query_physical_root_sha256,
        "support_query_disjoint": True,
        "classes": list(bank.classes),
        "active_k": active_k,
        "class_scales_fp16": _array_receipt(class_scales),
        "bias_fp16": _array_receipt(bias),
        "physical_support_id_count": len(canonical_physical_ids),
        "physical_support_id_canonical_sha256": _canonical_sha256(
            list(canonical_physical_ids)
        ),
        "physical_loo_enabled": loo_enabled,
        "physical_loo_self_exclusion": loo_enabled,
        "quantization_squared_distance_by_class": [
            float(value) for value in e_values
        ],
        "uncertainty_by_class": [float(value) for value in u_values],
        "loo_margin_by_class": [float(value) for value in delta_values],
        "loo_margin_spread_by_class": [float(value) for value in spread_values],
        "bias_sum_fp64": float(np.sum(bias.astype(np.float64))),
        "int8_support_vectors_retained": True,
        "raw_iq_retained": False,
        "fp32_support_vector_retained": False,
        "ground_input_read": False,
        "source_input_read": False,
        "registry_role_input_read": False,
        "query_truth_read": False,
        "query_role_read": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_update_count": 0,
        "query_batch_dependency": False,
        "query_file_io": False,
        "support_bank_int8_vector_bytes": int(bank.codes_qint8.nbytes),
        "support_bank_scale_bytes": int(bank.scales_fp16.nbytes),
        "head_deployment_state_bytes": int(
            class_scales.nbytes + bias.nbytes
        ),
        "head_state_wire_serialized_bytes": len(state_wire),
        "head_state_wire_sha256": _sha256_bytes(state_wire),
        "fit_numeric_workspace": workspace,
        "loo_fit_score_kernel_evaluations": int(loo_score_kernel_evaluations),
        "loo_fit_metric_matmul_mac": int(loo_metric_mac),
        "loo_fit_identity_scale_matmul_mac": int(loo_scale_mac),
        "loo_fit_total_matmul_mac": int(loo_metric_mac + loo_scale_mac),
        "base_kernel_evaluations_per_query": int(support_rows),
        "base_kernel_scalar_logic_counts_per_query": (
            _base_kernel_scalar_logic_counts(
                class_count=class_count,
                active_k=active_k,
            )
        ),
        "base_kernel_scalar_logic_count_scope": (
            "one query in current _score_with_support; excludes precision-cosine "
            "matrix products and ndarray allocation"
        ),
        "query_extra_dot_product_MAC": 0,
        "query_bias_add_ops": int(class_count),
        "operation_counter_source": "deterministic_exact_current_algorithm",
    }


@dataclass(frozen=True, slots=True)
class TypedLPORCQKNNState:
    """Immutable deployment state: bank-matched FP16 h and LPO-RC bias only."""

    classes: tuple[str, ...]
    active_k: int
    class_scales_fp16: np.ndarray
    bias_fp16: np.ndarray
    bank_receipt_sha256: str
    metric_receipt_sha256: str
    config_lock_digest: str
    split_handle_digest: str
    receipt: Mapping[str, Any]
    receipt_sha256: str
    schema: str = LPO_RC_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        class_scales = np.asarray(self.class_scales_fp16)
        bias = np.asarray(self.bias_fp16)
        if (
            self.schema != LPO_RC_SCHEMA
            or type(self.active_k) is not int
            or self.active_k < 1
            or class_scales.dtype != np.float16
            or bias.dtype != np.float16
            or class_scales.shape != (len(classes),)
            or bias.shape != (len(classes),)
            or not np.isfinite(class_scales).all()
            or not np.isfinite(bias).all()
            or np.any(class_scales <= 0.0)
        ):
            raise LPORCQKNNError("typed LPO-RC state array/schema drift")
        _require_sha256(self.bank_receipt_sha256, "LPO-RC bank receipt")
        _require_sha256(self.metric_receipt_sha256, "LPO-RC metric receipt")
        _require_sha256(self.config_lock_digest, "LPO-RC config lock")
        _require_sha256(self.split_handle_digest, "LPO-RC split handle")
        _require_sha256(self.receipt_sha256, "LPO-RC receipt")
        receipt = _thaw(self.receipt)
        if _canonical_sha256(receipt) != self.receipt_sha256:
            raise LPORCQKNNError("LPO-RC receipt verification failed")
        required = {
            "schema",
            "head_schema",
            "bank_receipt_sha256",
            "metric_receipt_sha256",
            "config_lock_digest",
            "split_handle_digest",
            "capsule_id",
            "split_id",
            "validator_receipt_sha256",
            "support_physical_root_sha256",
            "query_physical_root_sha256",
            "support_query_disjoint",
            "classes",
            "active_k",
            "class_scales_fp16",
            "bias_fp16",
            "physical_support_id_count",
            "physical_support_id_canonical_sha256",
            "physical_loo_enabled",
            "physical_loo_self_exclusion",
            "quantization_squared_distance_by_class",
            "uncertainty_by_class",
            "loo_margin_by_class",
            "loo_margin_spread_by_class",
            "bias_sum_fp64",
            "int8_support_vectors_retained",
            "raw_iq_retained",
            "fp32_support_vector_retained",
            "ground_input_read",
            "source_input_read",
            "registry_role_input_read",
            "query_truth_read",
            "query_role_read",
            "query_rows_used_for_fit",
            "query_state_updates",
            "query_update_count",
            "query_batch_dependency",
            "query_file_io",
            "support_bank_int8_vector_bytes",
            "support_bank_scale_bytes",
            "head_deployment_state_bytes",
            "head_state_wire_serialized_bytes",
            "head_state_wire_sha256",
            "fit_numeric_workspace",
            "loo_fit_score_kernel_evaluations",
            "loo_fit_metric_matmul_mac",
            "loo_fit_identity_scale_matmul_mac",
            "loo_fit_total_matmul_mac",
            "base_kernel_evaluations_per_query",
            "base_kernel_scalar_logic_counts_per_query",
            "base_kernel_scalar_logic_count_scope",
            "query_extra_dot_product_MAC",
            "query_bias_add_ops",
            "operation_counter_source",
        }
        if set(receipt) != required:
            raise LPORCQKNNError("LPO-RC receipt exact schema drift")
        if (
            receipt["schema"] != LPO_RC_RECEIPT_SCHEMA
            or receipt["head_schema"] != LPO_RC_SCHEMA
            or receipt["bank_receipt_sha256"] != self.bank_receipt_sha256
            or receipt["metric_receipt_sha256"] != self.metric_receipt_sha256
            or receipt["config_lock_digest"] != self.config_lock_digest
            or receipt["split_handle_digest"] != self.split_handle_digest
            or tuple(receipt["classes"]) != classes
            or receipt["active_k"] != self.active_k
            or receipt["class_scales_fp16"] != _array_receipt(class_scales)
            or receipt["bias_fp16"] != _array_receipt(bias)
            or receipt["int8_support_vectors_retained"] is not True
            or receipt["raw_iq_retained"] is not False
            or receipt["fp32_support_vector_retained"] is not False
            or receipt["ground_input_read"] is not False
            or receipt["source_input_read"] is not False
            or receipt["registry_role_input_read"] is not False
            or receipt["query_truth_read"] is not False
            or receipt["query_role_read"] is not False
            or receipt["query_rows_used_for_fit"] != 0
            or receipt["query_state_updates"] != 0
            or receipt["query_update_count"] != 0
            or receipt["query_batch_dependency"] is not False
            or receipt["query_file_io"] is not False
            or receipt["query_extra_dot_product_MAC"] != 0
            or receipt["query_bias_add_ops"] != len(classes)
        ):
            raise LPORCQKNNError("LPO-RC receipt lifecycle/invariant drift")
        if self.active_k == 1 and (
            receipt["physical_loo_enabled"] is not False
            or receipt["physical_loo_self_exclusion"] is not False
            or np.any(bias != np.float16(0.0))
        ):
            raise LPORCQKNNError("K1 LPO-RC state must be exact identity")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "class_scales_fp16", _readonly(class_scales, np.float16))
        object.__setattr__(self, "bias_fp16", _readonly(bias, np.float16))
        wire = _state_wire_bytes(
            classes=classes,
            active_k=self.active_k,
            class_scales=class_scales,
            bias=bias,
            bank_receipt_sha256=self.bank_receipt_sha256,
            metric_receipt_sha256=self.metric_receipt_sha256,
            config_lock_digest=self.config_lock_digest,
            split_handle_digest=self.split_handle_digest,
        )
        if (
            receipt["head_state_wire_serialized_bytes"] != len(wire)
            or receipt["head_state_wire_sha256"] != _sha256_bytes(wire)
        ):
            raise LPORCQKNNError("LPO-RC state wire receipt drift")
        object.__setattr__(self, "receipt", _freeze(receipt))


def build_lpo_rc_qknn_state(
    bank: TypedINT8ZIDSupportBank,
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    metric: TypedSharedPSDMetric,
    support_physical_ids: Sequence[str],
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> TypedLPORCQKNNState:
    """Fit immutable LPO-RC state from one legal target support row.

    ``support_zid`` is transient and is used only to bind the canonical INT8
    bank and calculate support-side reconstruction/LOO statistics.  It is not
    retained in the returned state or receipt.
    """

    if type(bank) is not TypedINT8ZIDSupportBank or type(metric) is not TypedSharedPSDMetric:
        raise LPORCQKNNError("LPO-RC requires exact typed bank and metric states")
    _verify_split_handle(split_handle)
    _verify_bank(bank)
    _verify_metric(metric)
    if metric.config_lock_digest != bank.config_lock_digest:
        raise LPORCQKNNError("bank/metric config lock drift")
    original, canonical_physical_ids = _verify_and_align_support(
        bank,
        support_zid,
        support_labels,
        registered_classes,
        support_physical_ids,
    )
    observed_support_root = _canonical_sha256(
        sorted(str(value) for value in support_physical_ids)
    )
    if observed_support_root != split_handle.support_physical_root_sha256:
        raise LPORCProtocolHandleError(
            "validated split handle support physical root mismatch"
        )
    decoded = decode_zid_support_bank(bank).astype(np.float64)
    if bank.active_k == 1:
        class_count = len(bank.classes)
        e_values = np.zeros(class_count, dtype=np.float64)
        u_values = np.zeros(class_count, dtype=np.float64)
        delta_values = np.zeros(class_count, dtype=np.float64)
        spread_values = np.zeros(class_count, dtype=np.float64)
        bias = np.zeros(class_count, dtype=np.float64)
    else:
        e_values, u_values, delta_values, spread_values = (
            _physical_loo_statistics(
                bank,
                metric,
                original,
                decoded,
            )
        )
        bias = _lpo_rc_bias(
            bank.active_k,
            u_values,
            delta_values,
            spread_values,
        )
    bias16 = np.asarray(bias, dtype=np.float16)
    if not np.isfinite(bias16).all():
        raise LPORCQKNNError("LPO-RC FP16 bias closure failed")
    class_scales = np.asarray(bank.class_scales_fp16, dtype=np.float16).copy()
    receipt = _receipt_payload(
        bank=bank,
        metric=metric,
        class_scales=class_scales,
        bias=bias16,
        canonical_physical_ids=canonical_physical_ids,
        e_values=e_values,
        u_values=u_values,
        delta_values=delta_values,
        spread_values=spread_values,
        split_handle=split_handle,
    )
    return TypedLPORCQKNNState(
        classes=bank.classes,
        active_k=bank.active_k,
        class_scales_fp16=class_scales,
        bias_fp16=bias16,
        bank_receipt_sha256=bank.bank_receipt_sha256,
        metric_receipt_sha256=metric.metric_receipt_sha256,
        config_lock_digest=bank.config_lock_digest,
        split_handle_digest=split_handle.handle_digest,
        receipt=receipt,
        receipt_sha256=_canonical_sha256(receipt),
    )


def _verify_state_binding(
    state: TypedLPORCQKNNState,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> None:
    if (
        type(state) is not TypedLPORCQKNNState
        or type(bank) is not TypedINT8ZIDSupportBank
        or type(metric) is not TypedSharedPSDMetric
    ):
        raise LPORCQKNNError("LPO-RC scoring requires exact typed states")
    _verify_split_handle(split_handle)
    _verify_bank(bank)
    _verify_metric(metric)
    receipt = _thaw(state.receipt)
    if _canonical_sha256(receipt) != state.receipt_sha256:
        raise LPORCQKNNError("LPO-RC receipt hash drift before use")
    if (
        not np.isfinite(state.class_scales_fp16).all()
        or not np.isfinite(state.bias_fp16).all()
        or receipt.get("class_scales_fp16")
        != _array_receipt(state.class_scales_fp16)
        or receipt.get("bias_fp16") != _array_receipt(state.bias_fp16)
    ):
        raise LPORCQKNNError("LPO-RC state array/receipt tamper detected")
    wire = _state_wire_bytes(
        classes=state.classes,
        active_k=state.active_k,
        class_scales=state.class_scales_fp16,
        bias=state.bias_fp16,
        bank_receipt_sha256=state.bank_receipt_sha256,
        metric_receipt_sha256=state.metric_receipt_sha256,
        config_lock_digest=state.config_lock_digest,
        split_handle_digest=state.split_handle_digest,
    )
    if (
        receipt.get("head_state_wire_serialized_bytes") != len(wire)
        or receipt.get("head_state_wire_sha256") != _sha256_bytes(wire)
    ):
        raise LPORCQKNNError("LPO-RC state wire tamper detected")
    if (
        state.classes != bank.classes
        or state.active_k != bank.active_k
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.metric_receipt_sha256 != metric.metric_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
        or metric.config_lock_digest != bank.config_lock_digest
        or not np.array_equal(state.class_scales_fp16, bank.class_scales_fp16)
    ):
        raise LPORCQKNNError("LPO-RC state/bank/metric binding drift")
    if state.split_handle_digest != split_handle.handle_digest:
        raise LPORCProtocolHandleError("LPO-RC validated split handle digest drift")


def score_lpo_rc_qknn_logits(
    state: TypedLPORCQKNNState,
    query_zid: np.ndarray,
    *,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> np.ndarray:
    """Score independent queries without state mutation or batch coupling."""

    _verify_state_binding(state, bank, metric, split_handle)
    base_logits = score_zid_student_t_logits(bank, query_zid, metric=metric)
    # Execute the fixed C bias additions for the resource contract.  K1 returns
    # the original array afterwards to retain an exact base-logit byte boundary.
    biased = np.asarray(base_logits, dtype=np.float32) + state.bias_fp16.astype(
        np.float32
    )[None, :]
    if state.active_k == 1:
        return base_logits
    if not np.isfinite(biased).all():
        raise LPORCQKNNError("LPO-RC query logits became non-finite")
    return _readonly(biased, np.float32)


def audit_lpo_rc_resource(
    state: TypedLPORCQKNNState,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> dict[str, Any]:
    """Return the exact state-bound resource/lifecycle receipt."""

    _verify_state_binding(state, bank, metric, split_handle)
    return _thaw(state.receipt)


def serialize_lpo_rc_qknn_state(
    state: TypedLPORCQKNNState,
    *,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    split_handle: TypedValidatedOnceP2SplitHandle,
) -> bytes:
    """Serialize the exact deployed FP16 h/b state after all bindings verify."""

    _verify_state_binding(state, bank, metric, split_handle)
    return _state_wire_bytes(
        classes=state.classes,
        active_k=state.active_k,
        class_scales=state.class_scales_fp16,
        bias=state.bias_fp16,
        bank_receipt_sha256=state.bank_receipt_sha256,
        metric_receipt_sha256=state.metric_receipt_sha256,
        config_lock_digest=state.config_lock_digest,
        split_handle_digest=state.split_handle_digest,
    )


__all__ = [
    "LPO_RC_RECEIPT_SCHEMA",
    "LPO_RC_SCHEMA",
    "LPO_RC_NUMERIC_WORKSPACE_BUDGET_BYTES",
    "LPO_RC_WIRE_MAGIC",
    "LPORCQKNNError",
    "LPORCProtocolHandleError",
    "P2_DATA_STATUS",
    "P2_PROTOCOL_SCHEMA",
    "P2_SPLIT_HANDLE_SCHEMA",
    "TypedLPORCQKNNState",
    "TypedValidatedOnceP2SplitHandle",
    "audit_lpo_rc_resource",
    "build_lpo_rc_qknn_state",
    "score_lpo_rc_qknn_logits",
    "serialize_lpo_rc_qknn_state",
    "validate_lpo_rc_physical_id_disjointness",
]
