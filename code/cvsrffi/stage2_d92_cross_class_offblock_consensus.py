"""D92 CCOC support-only cross-class off-block consensus covariance.

CCOC reuses the fixed D92 old/new registration covariance endpoints.  It uses
only canonicalized target support rows to decide how much of each endpoint's
cross-block structure is retained, then compiles one equal-prior FULL affine
head.  Query rows and query-derived state are intentionally absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .stage2_d92_registration_balanced_covariance import (
    OLD_CLASS_COUNT,
    RegistrationBalancedStatistics,
    build_registration_balanced_statistics,
)


_BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
_UPPER_BLOCK_PAIRS = (
    (_BLOCK_SLICES[0], _BLOCK_SLICES[1]),
    (_BLOCK_SLICES[0], _BLOCK_SLICES[2]),
    (_BLOCK_SLICES[1], _BLOCK_SLICES[2]),
)
_CANONICALIZATION = "lexicographic_float32_row_bytes_then_float64_reduce"
_CANONICAL_TIE_POLICY = (
    "float32_row_bytes_then_float64_row_bytes_"
    "duplicate_class_handle_fail_closed"
)
_FEATURE_DIMENSION = 288
_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_COVARIANCE_BUFFER_BYTES = int(
    _FEATURE_DIMENSION * _FEATURE_DIMENSION * _FLOAT64_BYTES
)
_MAX_BLOCK_DIMENSION = max(block.stop - block.start for block in _BLOCK_SLICES)
_MAX_DIAGONAL_BLOCK_WORKSPACE_BYTES = int(
    _MAX_BLOCK_DIMENSION * _MAX_BLOCK_DIMENSION * _FLOAT64_BYTES
)
_MAX_BLOCK_ROW_WORKSPACE_BYTES = int(_MAX_BLOCK_DIMENSION * _FLOAT64_BYTES)
_UPPER_ACCUMULATOR_BYTES = int(
    sum(
        (left.stop - left.start) * (right.stop - right.start) * _FLOAT64_BYTES
        for left, right in _UPPER_BLOCK_PAIRS
    )
)
_CROSS_BLOCK_WORKSPACE_BYTES = int(160 * 96 * _FLOAT64_BYTES)
_K10_RESIDUAL_WORKSPACE_BYTES = int(10 * _FEATURE_DIMENSION * _FLOAT64_BYTES)
_K10_NUMERIC_WORKSPACE_BYTES = int(
    _UPPER_ACCUMULATOR_BYTES
    + _CROSS_BLOCK_WORKSPACE_BYTES
    + _K10_RESIDUAL_WORKSPACE_BYTES
)


class D92CCOCError(RuntimeError):
    """Raised when the frozen D92 CCOC support-only contract drifts."""


class D92CCOCNumericalError(D92CCOCError):
    """Raised for a finite-support CCOC numerical degeneration."""


def _require_exact_integer_count(value: Any, *, name: str) -> int:
    """Reject public count coercions before they can alter a CCOC route."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise D92CCOCError(f"ccoc_{name}_not_exact_integer")
    return int(value)


@dataclass(frozen=True)
class CrossClassOffblockConsensusStatistics:
    """One CCOC covariance and the reused D92 registration statistics."""

    base: RegistrationBalancedStatistics
    covariance: np.ndarray
    old_rho: float
    new_rho: float
    audit: dict[str, Any]


def _float32_row_bytes(row: np.ndarray) -> bytes:
    """Build the canonical sort key without changing the FP64 reduction row."""

    with np.errstate(over="ignore", invalid="ignore"):
        row32 = np.ascontiguousarray(np.asarray(row, dtype=np.float32))
    if not np.isfinite(row32).all():
        raise D92CCOCNumericalError("ccoc_q_nonfinite")
    return row32.tobytes(order="C")


def _float64_row_bytes(row: np.ndarray) -> bytes:
    """Build the lossless tie key from the original FP64 reduction row."""

    row64 = np.ascontiguousarray(np.asarray(row, dtype=np.float64))
    if not np.isfinite(row64).all():
        raise D92CCOCNumericalError("ccoc_q_nonfinite")
    return row64.tobytes(order="C")


def _canonical_class_row_order(
    rows: np.ndarray,
    labels: np.ndarray,
    class_index: int,
    k_shot: int,
) -> tuple[tuple[tuple[bytes, bytes], ...], tuple[int, ...]]:
    """Return a lossless canonical class handle without copying class rows."""

    indices = np.flatnonzero(labels == int(class_index))
    if len(indices) != int(k_shot):
        raise D92CCOCError("ccoc_unbalanced_group_registry")
    keyed = [
        (
            (
                _float32_row_bytes(rows[int(index)]),
                _float64_row_bytes(rows[int(index)]),
            ),
            int(index),
        )
        for index in indices
    ]
    keyed.sort(key=lambda item: item[0])
    return tuple(item[0] for item in keyed), tuple(item[1] for item in keyed)


def _canonical_group_class_orders(
    rows: np.ndarray,
    labels: np.ndarray,
    class_indices: Iterable[int],
    k_shot: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Order class contributions by canonical support content, not class ID."""

    keyed: list[
        tuple[tuple[tuple[bytes, bytes], ...], int, tuple[int, ...]]
    ] = []
    for raw_index in class_indices:
        class_index = int(raw_index)
        class_key, row_order = _canonical_class_row_order(
            rows, labels, class_index, k_shot
        )
        keyed.append((class_key, class_index, row_order))
    keyed.sort(key=lambda item: item[0])
    for previous, current in zip(keyed, keyed[1:]):
        if previous[0] == current[0]:
            # A complete handle proves equal FP64 rows and therefore equal
            # reductions.  Reject rather than let a stable sort use class IDs
            # as an implicit ordering rule for indistinguishable classes.
            raise D92CCOCError("ccoc_identical_class_handle")
    return tuple((item[1], item[2]) for item in keyed)


def _cross_block_into(
    residual: np.ndarray,
    left: slice,
    right: slice,
    denominator: float,
    workspace: np.ndarray,
) -> np.ndarray:
    """Overwrite the one reusable max-cross-block workspace and return its view."""

    rows = int(left.stop - left.start)
    columns = int(right.stop - right.start)
    block = workspace[: rows * columns].reshape(rows, columns)
    np.matmul(residual[:, left].T, residual[:, right], out=block)
    np.multiply(block, 1.0 / denominator, out=block)
    if not np.isfinite(block).all():
        raise D92CCOCNumericalError("ccoc_q_nonfinite")
    return block


def _workspace_receipt(k_shot: int) -> dict[str, int]:
    """Report the exact live numerical buffers used by the streaming core."""

    shots = int(k_shot)
    residual_bytes = int(shots * _FEATURE_DIMENSION * _FLOAT64_BYTES)
    numeric_bytes = int(
        _UPPER_ACCUMULATOR_BYTES
        + _CROSS_BLOCK_WORKSPACE_BYTES
        + residual_bytes
    )
    return {
        "upper_accumulators_bytes": _UPPER_ACCUMULATOR_BYTES,
        "cross_block_buffer_bytes": _CROSS_BLOCK_WORKSPACE_BYTES,
        "residual_buffer_bytes": residual_bytes,
        "numeric_bytes_upper_bound": numeric_bytes,
        "frozen_k10_numeric_bytes_upper_bound": _K10_NUMERIC_WORKSPACE_BYTES,
    }


def _stream_group_consensus(
    transformed: np.ndarray,
    targets: np.ndarray,
    class_indices: Iterable[int],
    k_shot: int,
) -> tuple[float, dict[str, Any]]:
    """Stream the average pairwise cosine of a registration group's Q blocks.

    A class first accumulates the joint Frobenius norm of its three upper
    cross-blocks.  It then recomputes each block and adds the normalized block
    into its one group accumulator.  This keeps no class stack of Q or unit
    directions while preserving the exact pairwise-cosine identity:
    ``(||sum u_c||^2-C)/(C*(C-1))``.
    """

    rows = np.asarray(transformed)
    labels = np.asarray(targets)
    classes = tuple(int(index) for index in class_indices)
    shots = int(k_shot)
    if (
        rows.ndim != 2
        or rows.shape[1] != _FEATURE_DIMENSION
        or labels.shape != (len(rows),)
        or not np.issubdtype(labels.dtype, np.integer)
        or len(classes) < 2
        or shots <= 2
    ):
        raise D92CCOCError("ccoc_invalid_group_registry")
    ordered_classes = _canonical_group_class_orders(
        rows, labels, classes, shots
    )
    accumulators = [
        np.zeros((left.stop - left.start, right.stop - right.start), dtype=np.float64)
        for left, right in _UPPER_BLOCK_PAIRS
    ]
    cross_block_workspace = np.empty(
        _CROSS_BLOCK_WORKSPACE_BYTES // _FLOAT64_BYTES, dtype=np.float64
    )
    residual = np.empty((shots, _FEATURE_DIMENSION), dtype=np.float64)
    norm_min = math.inf
    norm_max = 0.0
    denominator = float(shots - 1)

    for _, row_order in ordered_classes:
        for local_index, row_index in enumerate(row_order):
            source_row = rows[row_index]
            if not np.isfinite(source_row).all():
                raise D92CCOCNumericalError("ccoc_q_nonfinite")
            residual[local_index] = source_row
        mean_workspace = cross_block_workspace[:_FEATURE_DIMENSION]
        mean_workspace.fill(0.0)
        for local_index in range(shots):
            np.add(mean_workspace, residual[local_index], out=mean_workspace)
        np.multiply(mean_workspace, 1.0 / float(shots), out=mean_workspace)
        np.subtract(residual, mean_workspace, out=residual)
        if not np.isfinite(mean_workspace).all() or not np.isfinite(residual).all():
            raise D92CCOCNumericalError("ccoc_q_nonfinite")

        norm_squared = 0.0
        for left, right in _UPPER_BLOCK_PAIRS:
            q_block = _cross_block_into(
                residual, left, right, denominator, cross_block_workspace
            )
            block_norm_squared = float(np.vdot(q_block, q_block).real)
            if not math.isfinite(block_norm_squared) or block_norm_squared < 0.0:
                raise D92CCOCNumericalError("ccoc_q_nonfinite")
            norm_squared += block_norm_squared
        if not math.isfinite(norm_squared):
            raise D92CCOCNumericalError("ccoc_q_nonfinite")
        class_norm = math.sqrt(norm_squared)
        if class_norm <= 0.0:
            raise D92CCOCNumericalError("ccoc_q_zero_frobenius_norm")

        for accumulator, (left, right) in zip(accumulators, _UPPER_BLOCK_PAIRS):
            q_block = _cross_block_into(
                residual, left, right, denominator, cross_block_workspace
            )
            np.multiply(q_block, 1.0 / class_norm, out=q_block)
            np.add(accumulator, q_block, out=accumulator)
        norm_min = min(norm_min, class_norm)
        norm_max = max(norm_max, class_norm)

    summed_unit_norm_squared = 0.0
    for accumulator in accumulators:
        if not np.isfinite(accumulator).all():
            raise D92CCOCNumericalError("ccoc_rho_nonfinite")
        summed_unit_norm_squared += float(np.vdot(accumulator, accumulator).real)
    class_total = len(ordered_classes)
    rho_raw = float(
        (summed_unit_norm_squared - float(class_total))
        / float(class_total * (class_total - 1))
    )
    if not math.isfinite(rho_raw):
        raise D92CCOCNumericalError("ccoc_rho_nonfinite")
    rho = float(np.clip(rho_raw, 0.0, 1.0))
    if not math.isfinite(rho):
        raise D92CCOCNumericalError("ccoc_rho_nonfinite")
    return rho, {
        "class_count": class_total,
        "offblock_norm_min": float(norm_min),
        "offblock_norm_max": float(norm_max),
        "pairwise_cosine_raw": rho_raw,
        "pairwise_cosine_clipped": rho,
        "crossblock_passes_per_class": 2,
        "upper_block_count": len(_UPPER_BLOCK_PAIRS),
        "canonicalization": _CANONICALIZATION,
        "canonicalization_tie_policy": _CANONICAL_TIE_POLICY,
    }


def _blockdiag(covariance: np.ndarray) -> np.ndarray:
    """Keep exactly the three D42 diagonal blocks of one 288d covariance."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (288, 288):
        raise D92CCOCError("ccoc_endpoint_shape_drift")
    result = np.zeros_like(matrix)
    for block in _BLOCK_SLICES:
        result[block, block] = matrix[block, block]
    return result


def _covariance_mix_workspace_receipt() -> dict[str, int]:
    """Report the explicit live buffers for the low-peak covariance assembly.

    The two frozen D92 endpoints are caller-owned read-only inputs.  CCOC adds
    one output covariance, one reusable largest diagonal-block workspace, and
    one row scratch for preserving the original diagonal multiplication order.
    The streaming rho workspace has returned before this assembly starts, so
    its separately reported bound does not overlap this live set.
    """

    live_bytes = int(
        _COVARIANCE_BUFFER_BYTES
        + _MAX_DIAGONAL_BLOCK_WORKSPACE_BYTES
        + _MAX_BLOCK_ROW_WORKSPACE_BYTES
    )
    return {
        "candidate_covariance_result_bytes": _COVARIANCE_BUFFER_BYTES,
        "candidate_covariance_block_workspace_bytes": (
            _MAX_DIAGONAL_BLOCK_WORKSPACE_BYTES
        ),
        "candidate_covariance_row_workspace_bytes": _MAX_BLOCK_ROW_WORKSPACE_BYTES,
        "candidate_covariance_full_buffer_count_upper_bound": 1,
        "candidate_covariance_mix_live_bytes_upper_bound": live_bytes,
    }


def _validate_endpoint_and_rho(
    covariance: np.ndarray, rho: float
) -> tuple[np.ndarray, float]:
    """Validate one frozen endpoint without allocating another full matrix."""

    value = float(rho)
    if not math.isfinite(value):
        raise D92CCOCNumericalError("ccoc_rho_nonfinite")
    if value < 0.0 or value > 1.0:
        raise D92CCOCNumericalError("ccoc_rho_out_of_range")
    full = np.asarray(covariance, dtype=np.float64)
    if full.shape != (_FEATURE_DIMENSION, _FEATURE_DIMENSION):
        raise D92CCOCError("ccoc_endpoint_shape_drift")
    if not np.isfinite(full).all():
        raise D92CCOCNumericalError("ccoc_endpoint_nonfinite")
    return full, value


def _mix_endpoint_into(
    result: np.ndarray,
    endpoint: np.ndarray,
    rho: float,
    *,
    block_workspace: np.ndarray,
) -> None:
    """Overwrite ``result`` with one literal frozen full/block endpoint mix.

    This retains the old elementwise multiplication/addition order, including
    diagonal rounding, while avoiding a full block-diagonal matrix.  The three
    upper off-diagonal blocks determine their transposes exactly because each
    frozen D92 endpoint is symmetric.
    """

    other = 1.0 - float(rho)
    for left, right in _UPPER_BLOCK_PAIRS:
        upper = result[left, right]
        np.multiply(upper, rho, out=upper)
        # The frozen literal includes ``+(1-rho)*0`` on off-diagonal blocks.
        # Retain that signed-zero rounding without creating a second matrix.
        np.add(upper, 0.0, out=upper)
        np.copyto(result[right, left], upper.T)
    for block in _BLOCK_SLICES:
        rows = int(block.stop - block.start)
        target = result[block, block]
        source = endpoint[block, block]
        scratch = block_workspace[:rows, :rows]
        np.multiply(target, rho, out=target)
        np.multiply(source, other, out=scratch)
        np.add(target, scratch, out=target)


def _add_half_endpoint_mix_into(
    result: np.ndarray,
    endpoint: np.ndarray,
    rho: float,
    *,
    block_workspace: np.ndarray,
    row_workspace: np.ndarray,
) -> None:
    """Add half a frozen endpoint mix using one reusable block workspace."""

    other = 1.0 - float(rho)
    for left, right in _UPPER_BLOCK_PAIRS:
        rows = int(left.stop - left.start)
        columns = int(right.stop - right.start)
        source = endpoint[left, right]
        scratch = block_workspace[:rows, :columns]
        np.multiply(source, rho, out=scratch)
        np.add(scratch, 0.0, out=scratch)
        np.multiply(scratch, 0.5, out=scratch)
        upper = result[left, right]
        np.add(upper, scratch, out=upper)
        np.copyto(result[right, left], upper.T)
    for block in _BLOCK_SLICES:
        rows = int(block.stop - block.start)
        target = result[block, block]
        source = endpoint[block, block]
        scratch = block_workspace[:rows, :rows]
        np.multiply(source, rho, out=scratch)
        for row_index in range(rows):
            row = row_workspace[:rows]
            np.multiply(source[row_index], other, out=row)
            np.add(scratch[row_index], row, out=scratch[row_index])
        np.multiply(scratch, 0.5, out=scratch)
        np.add(target, scratch, out=target)


def _stream_task_covariance_mix(
    old_covariance: np.ndarray,
    new_covariance: np.ndarray,
    old_rho: float,
    new_rho: float,
) -> np.ndarray:
    """Assemble the frozen task-balanced covariance without full temporaries."""

    old, old_value = _validate_endpoint_and_rho(old_covariance, old_rho)
    new, new_value = _validate_endpoint_and_rho(new_covariance, new_rho)
    if old.shape != new.shape:
        raise D92CCOCError("ccoc_task_covariance_shape_drift")
    result = old.copy()
    block_workspace = np.empty(
        (_MAX_BLOCK_DIMENSION, _MAX_BLOCK_DIMENSION), dtype=np.float64
    )
    row_workspace = np.empty(_MAX_BLOCK_DIMENSION, dtype=np.float64)
    _mix_endpoint_into(
        result, old, old_value, block_workspace=block_workspace
    )
    np.multiply(result, 0.5, out=result)
    _add_half_endpoint_mix_into(
        result,
        new,
        new_value,
        block_workspace=block_workspace,
        row_workspace=row_workspace,
    )
    return result


def _mix_full_and_blockdiag(covariance: np.ndarray, rho: float) -> np.ndarray:
    """Interpolate one full endpoint and its block-diagonal endpoint."""

    full, value = _validate_endpoint_and_rho(covariance, rho)
    result = full.copy()
    workspace = np.empty(
        (_MAX_BLOCK_DIMENSION, _MAX_BLOCK_DIMENSION), dtype=np.float64
    )
    _mix_endpoint_into(result, full, value, block_workspace=workspace)
    return result


def _combine_task_covariances(
    old_covariance: np.ndarray, new_covariance: np.ndarray
) -> np.ndarray:
    """Apply the locked equal old/new registration-task mixture exactly once."""

    old = np.asarray(old_covariance, dtype=np.float64)
    new = np.asarray(new_covariance, dtype=np.float64)
    if old.shape != new.shape or old.shape != (288, 288):
        raise D92CCOCError("ccoc_task_covariance_shape_drift")
    result = old.copy()
    workspace = np.empty(
        (_MAX_BLOCK_DIMENSION, _MAX_BLOCK_DIMENSION), dtype=np.float64
    )
    np.multiply(result, 0.5, out=result)
    for row_block in _BLOCK_SLICES:
        rows = int(row_block.stop - row_block.start)
        for column_block in _BLOCK_SLICES:
            columns = int(column_block.stop - column_block.start)
            scratch = workspace[:rows, :columns]
            np.multiply(new[row_block, column_block], 0.5, out=scratch)
            np.add(result[row_block, column_block], scratch, out=result[row_block, column_block])
    return result


def _require_symmetric_positive_definite(
    covariance: np.ndarray, *, name: str
) -> float:
    """Reject nonfinite, asymmetric, or non-SPD endpoint/covariance math."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (288, 288):
        raise D92CCOCNumericalError(f"ccoc_{name}_shape")
    if not np.isfinite(matrix).all():
        raise D92CCOCNumericalError(f"ccoc_{name}_nonfinite")
    if not np.array_equal(matrix, matrix.T):
        raise D92CCOCNumericalError(f"ccoc_{name}_not_symmetric")
    try:
        cholesky = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise D92CCOCNumericalError(
            f"ccoc_{name}_not_positive_definite"
        ) from error
    diagonal_min = float(np.min(np.diag(cholesky)))
    if not math.isfinite(diagonal_min) or diagonal_min <= 0.0:
        raise D92CCOCNumericalError(f"ccoc_{name}_not_positive_definite")
    return diagonal_min


def _support_macs_upper_bound(class_count: int, k_shot: int) -> int:
    """Count the two streaming upper-block passes without a class Q stack."""

    cross_coordinates = sum(
        (left.stop - left.start) * (right.stop - right.start)
        for left, right in _UPPER_BLOCK_PAIRS
    )
    return int(2 * int(class_count) * int(k_shot) * cross_coordinates)


def _ccoc_statistics_audit(
    base: RegistrationBalancedStatistics,
    old_audit: dict[str, Any],
    new_audit: dict[str, Any],
    covariance: np.ndarray,
    *,
    old_cholesky_min: float,
    new_cholesky_min: float,
    final_cholesky_min: float,
) -> dict[str, Any]:
    """Emit a compact, support-only receipt for the active CCOC core."""

    audit = dict(base.covariance_audit)
    support_macs = _support_macs_upper_bound(base.class_count, base.k_shot)
    workspace = _workspace_receipt(base.k_shot)
    covariance_mix_workspace = _covariance_mix_workspace_receipt()
    audit.update(
        {
            "d92_ccoc_active": True,
            "d92_ccoc_fallback_active": False,
            "d92_ccoc_fallback_reason": None,
            "d92_ccoc_formula_revision": "pairwise_cosine_v1",
            "d92_ccoc_formula": (
                "Sigma=0.5*mix(Sigma_old,rho_old)+0.5*mix(Sigma_new,rho_new)"
            ),
            "d92_ccoc_old_rho": float(old_audit["pairwise_cosine_clipped"]),
            "d92_ccoc_new_rho": float(new_audit["pairwise_cosine_clipped"]),
            "d92_ccoc_old_group_class_count": int(old_audit["class_count"]),
            "d92_ccoc_new_group_class_count": int(new_audit["class_count"]),
            "d92_ccoc_old_offblock_norm_min": float(old_audit["offblock_norm_min"]),
            "d92_ccoc_old_offblock_norm_max": float(old_audit["offblock_norm_max"]),
            "d92_ccoc_new_offblock_norm_min": float(new_audit["offblock_norm_min"]),
            "d92_ccoc_new_offblock_norm_max": float(new_audit["offblock_norm_max"]),
            "d92_ccoc_old_pairwise_cosine_raw": float(old_audit["pairwise_cosine_raw"]),
            "d92_ccoc_new_pairwise_cosine_raw": float(new_audit["pairwise_cosine_raw"]),
            "d92_ccoc_canonicalization": _CANONICALIZATION,
            "d92_ccoc_canonicalization_tie_policy": _CANONICAL_TIE_POLICY,
            "d92_ccoc_crossblock_passes_per_class": 2,
            "d92_ccoc_upper_block_count": len(_UPPER_BLOCK_PAIRS),
            "d92_ccoc_covariance_symmetric": bool(np.array_equal(covariance, covariance.T)),
            "d92_ccoc_full_endpoint_reused": True,
            "d92_ccoc_full_endpoint_reuse": True,
            "d92_ccoc_additional_fit_count": 0,
            "d92_ccoc_additional_full_fit_count": 0,
            "d92_ccoc_additional_block_fit_count": 0,
            "d92_ccoc_additional_loo_fit_count": 0,
            "d92_ccoc_additional_fisher_fit_count": 0,
            "d92_ccoc_additional_scan_count": 0,
            "d92_ccoc_block_fit_count": 0,
            "d92_ccoc_loo_fit_count": 0,
            "d92_ccoc_fisher_fit_count": 0,
            "d92_ccoc_scan_count": 0,
            "d92_ccoc_hyperparameter_scan_count": 0,
            "d92_ccoc_weight_scan_count": 0,
            "d92_ccoc_dense_solve_count": 0,
            "d92_ccoc_cholesky_check_count": 3,
            "d92_ccoc_cholesky_endpoint_check_count": 2,
            "d92_ccoc_cholesky_final_check_count": 1,
            "d92_ccoc_cholesky_pass": True,
            "d92_ccoc_old_endpoint_cholesky_min_diagonal": old_cholesky_min,
            "d92_ccoc_new_endpoint_cholesky_min_diagonal": new_cholesky_min,
            "d92_ccoc_final_cholesky_min_diagonal": final_cholesky_min,
            "d92_ccoc_support_macs_upper_bound": support_macs,
            "d92_ccoc_workspace_upper_accumulators_bytes": workspace[
                "upper_accumulators_bytes"
            ],
            "d92_ccoc_workspace_cross_block_buffer_bytes": workspace[
                "cross_block_buffer_bytes"
            ],
            "d92_ccoc_workspace_residual_buffer_bytes": workspace[
                "residual_buffer_bytes"
            ],
            "d92_ccoc_workspace_numeric_bytes_upper_bound": workspace[
                "numeric_bytes_upper_bound"
            ],
            "d92_ccoc_workspace_frozen_k10_numeric_bytes_upper_bound": workspace[
                "frozen_k10_numeric_bytes_upper_bound"
            ],
            "d92_ccoc_workspace_candidate_covariance_result_bytes": (
                covariance_mix_workspace["candidate_covariance_result_bytes"]
            ),
            "d92_ccoc_workspace_candidate_covariance_block_workspace_bytes": (
                covariance_mix_workspace[
                    "candidate_covariance_block_workspace_bytes"
                ]
            ),
            "d92_ccoc_workspace_candidate_covariance_row_workspace_bytes": (
                covariance_mix_workspace[
                    "candidate_covariance_row_workspace_bytes"
                ]
            ),
            "d92_ccoc_workspace_candidate_covariance_full_buffer_count_upper_bound": (
                covariance_mix_workspace[
                    "candidate_covariance_full_buffer_count_upper_bound"
                ]
            ),
            "d92_ccoc_workspace_candidate_covariance_mix_live_bytes_upper_bound": (
                covariance_mix_workspace[
                    "candidate_covariance_mix_live_bytes_upper_bound"
                ]
            ),
            "d92_ccoc_support_transient_bytes_upper_bound": workspace[
                "numeric_bytes_upper_bound"
            ],
            "d92_ccoc_persistent_state_bytes_delta": 0,
            "d92_ccoc_persistent_bytes_delta": 0,
            "d92_ccoc_query_state_bytes_delta": 0,
            "d92_ccoc_query_bytes_delta": 0,
            "d92_ccoc_query_macs_delta": 0,
            "d92_ccoc_query_macs": 0,
            "d92_ccoc_query_rows_used": 0,
            "d92_ccoc_query_fit_access": False,
            "d92_ccoc_query_update_access": False,
            "d92_ccoc_query_selection_access": False,
            "d92_ccoc_query_truth_access": False,
            "d92_ccoc_query_role_oracle_access": False,
            "d92_ccoc_query_class_quota_access": False,
            "d92_ccoc_query_global_reassignment": False,
            "support_macs_upper_bound": support_macs,
            "support_transient_bytes_upper_bound": workspace[
                "numeric_bytes_upper_bound"
            ],
            "persistent_state_bytes_delta": 0,
            "query_state_bytes_delta": 0,
            "query_macs_delta": 0,
        }
    )
    return audit


def build_cross_class_offblock_consensus_statistics(
    d42: Any,
    transformed: np.ndarray,
    targets: np.ndarray,
    *,
    class_count: int,
    k_shot: int,
) -> CrossClassOffblockConsensusStatistics:
    """Build one CCOC covariance from a locked D92 registered support registry."""

    base = build_registration_balanced_statistics(
        d42,
        transformed,
        targets,
        class_count=class_count,
        k_shot=k_shot,
    )
    if int(d42.FEATURE_DIM) != 288 or tuple(d42.BLOCK_SLICES) != _BLOCK_SLICES:
        raise D92CCOCError("ccoc_d42_block_layout_drift")
    if base.class_count != int(class_count) or base.k_shot != int(k_shot):
        raise D92CCOCError("ccoc_base_registry_drift")

    old_cholesky_min = _require_symmetric_positive_definite(
        base.old_covariance, name="old_endpoint"
    )
    new_cholesky_min = _require_symmetric_positive_definite(
        base.new_covariance, name="new_endpoint"
    )
    old_rho, old_audit = _stream_group_consensus(
        transformed, targets, range(OLD_CLASS_COUNT), k_shot
    )
    new_rho, new_audit = _stream_group_consensus(
        transformed, targets, range(OLD_CLASS_COUNT, int(class_count)), k_shot
    )
    covariance = _stream_task_covariance_mix(
        base.old_covariance,
        base.new_covariance,
        old_rho,
        new_rho,
    )
    final_cholesky_min = _require_symmetric_positive_definite(
        covariance, name="final_covariance"
    )
    covariance.setflags(write=False)
    return CrossClassOffblockConsensusStatistics(
        base=base,
        covariance=covariance,
        old_rho=old_rho,
        new_rho=new_rho,
        audit=_ccoc_statistics_audit(
            base,
            old_audit,
            new_audit,
            covariance,
            old_cholesky_min=old_cholesky_min,
            new_cholesky_min=new_cholesky_min,
            final_cholesky_min=final_cholesky_min,
        ),
    )


def compile_cross_class_offblock_consensus_affine(
    d42: Any,
    statistics: CrossClassOffblockConsensusStatistics,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compile the one equal-prior CCOC FULL LDA affine head exactly once."""

    if not isinstance(statistics, CrossClassOffblockConsensusStatistics):
        raise D92CCOCError("ccoc_statistics_type_drift")
    if int(d42.FEATURE_DIM) != 288:
        raise D92CCOCError("ccoc_feature_dimension_drift")
    classes = int(statistics.base.class_count)
    means = np.asarray(statistics.base.means, dtype=np.float64)
    covariance = np.asarray(statistics.covariance, dtype=np.float64)
    if (
        means.shape != (classes, 288)
        or covariance.shape != (288, 288)
        or not np.isfinite(means).all()
        or not np.isfinite(covariance).all()
    ):
        raise D92CCOCNumericalError("ccoc_compile_statistics_nonfinite_or_shape_drift")
    try:
        coefficients = np.linalg.solve(covariance, means.T).T
    except np.linalg.LinAlgError as error:
        raise D92CCOCNumericalError("ccoc_dense_solve_failure") from error
    intercept = -0.5 * np.sum(means * coefficients, axis=1)
    intercept -= np.log(float(classes))
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise D92CCOCNumericalError("ccoc_compile_affine_nonfinite")
    audit = dict(statistics.audit)
    audit.update(
        {
            "solver": "lsqr_equivalent_explicit_full_solve",
            "shrinkage": "ccoc_pairwise_cosine_full_block_endpoint_mix",
            "prior_policy": "equal_1_over_registered_class_count",
            "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
            "unit_covariance_fallback": False,
            "d92_ccoc_dense_solve_count": 1,
            "d92_ccoc_compile_solve_count": 1,
            "d92_ccoc_full_solve_count": 1,
            "d92_ccoc_full_dense_288_solve_count": 1,
            "d92_ccoc_compiled_cholesky_check_count": 0,
            "d92_ccoc_covariance_equation_residual_max": float(
                np.max(np.abs(covariance @ coefficients.T - means.T))
            ),
            "covariance_equation_residual_max": float(
                np.max(np.abs(covariance @ coefficients.T - means.T))
            ),
        }
    )
    return coefficients.astype(np.float32), intercept.astype(np.float32), audit


def ccoc_inactive_receipt(
    class_count: int,
    k_shot: int,
    *,
    old_class_count: int = OLD_CLASS_COUNT,
) -> dict[str, Any]:
    """Return the no-fit receipt for pre-registration or K1/K2 CCOC states."""

    classes = _require_exact_integer_count(class_count, name="class_count")
    shots = _require_exact_integer_count(k_shot, name="k_shot")
    old_count = _require_exact_integer_count(
        old_class_count, name="old_class_count"
    )
    if old_count != OLD_CLASS_COUNT:
        raise D92CCOCError("ccoc_old_class_count_override_drift")
    if classes < old_count or shots < 1:
        raise D92CCOCError("ccoc_invalid_inactive_registry")
    if classes == old_count:
        status = "before_exact_d81"
    elif shots <= 2:
        status = "k1_k2_exact_d81_fallback"
    else:
        raise D92CCOCError("ccoc_inactive_receipt_active_registration")
    return {
        "d92_ccoc_active": False,
        "d92_ccoc_fallback_active": False,
        "d92_ccoc_fallback_reason": status,
        "d92_ccoc_formula_revision": "pairwise_cosine_v1",
        "d92_ccoc_status": status,
        "d92_ccoc_old_rho": None,
        "d92_ccoc_new_rho": None,
        "d92_ccoc_old_group_class_count": old_count,
        "d92_ccoc_new_group_class_count": max(0, classes - old_count),
        "d92_ccoc_canonicalization": _CANONICALIZATION,
        "d92_ccoc_canonicalization_tie_policy": _CANONICAL_TIE_POLICY,
        "d92_ccoc_full_endpoint_reused": False,
        "d92_ccoc_full_endpoint_reuse": False,
        "d92_ccoc_additional_fit_count": 0,
        "d92_ccoc_additional_full_fit_count": 0,
        "d92_ccoc_additional_block_fit_count": 0,
        "d92_ccoc_additional_loo_fit_count": 0,
        "d92_ccoc_additional_fisher_fit_count": 0,
        "d92_ccoc_additional_scan_count": 0,
        "d92_ccoc_hyperparameter_scan_count": 0,
        "d92_ccoc_weight_scan_count": 0,
        "d92_ccoc_dense_solve_count": 0,
        "d92_ccoc_cholesky_check_count": 0,
        "d92_ccoc_cholesky_pass": False,
        "d92_ccoc_support_macs_upper_bound": 0,
        "d92_ccoc_support_transient_bytes_upper_bound": 0,
        "d92_ccoc_persistent_state_bytes_delta": 0,
        "d92_ccoc_persistent_bytes_delta": 0,
        "d92_ccoc_query_state_bytes_delta": 0,
        "d92_ccoc_query_bytes_delta": 0,
        "d92_ccoc_query_macs_delta": 0,
        "d92_ccoc_query_macs": 0,
        "d92_ccoc_query_rows_used": 0,
        "d92_ccoc_query_fit_access": False,
        "d92_ccoc_query_update_access": False,
        "d92_ccoc_query_selection_access": False,
        "d92_ccoc_query_truth_access": False,
        "d92_ccoc_query_role_oracle_access": False,
        "d92_ccoc_query_class_quota_access": False,
        "d92_ccoc_query_global_reassignment": False,
    }


__all__ = [
    "CrossClassOffblockConsensusStatistics",
    "D92CCOCError",
    "D92CCOCNumericalError",
    "build_cross_class_offblock_consensus_statistics",
    "ccoc_inactive_receipt",
    "compile_cross_class_offblock_consensus_affine",
]
