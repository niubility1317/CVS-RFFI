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
_CANONICALIZATION = (
    "per_class_lexicographic_float32_row_bytes_then_float64_mean_residual_scatter"
)
_SUPPORT_TRANSIENT_BYTES_UPPER_BOUND = 334336


class D92CCOCError(RuntimeError):
    """Raised when the frozen D92 CCOC support-only contract drifts."""


class D92CCOCNumericalError(D92CCOCError):
    """Raised for a finite-support CCOC numerical degeneration."""


@dataclass(frozen=True)
class CrossClassOffblockConsensusStatistics:
    """One CCOC covariance and the reused D92 registration statistics."""

    base: RegistrationBalancedStatistics
    covariance: np.ndarray
    old_rho: float
    new_rho: float
    audit: dict[str, Any]


def _canonical_float32_class_rows(rows: np.ndarray) -> np.ndarray:
    """Canonicalize one class by its float32 row bytes before FP64 math."""

    source = np.asarray(rows)
    if source.ndim != 2 or len(source) == 0:
        raise D92CCOCError("ccoc_invalid_class_rows")
    with np.errstate(over="ignore", invalid="ignore"):
        rows32 = np.ascontiguousarray(np.asarray(source, dtype=np.float32))
    if not np.isfinite(rows32).all():
        raise D92CCOCNumericalError("ccoc_q_nonfinite")
    order = sorted(
        range(len(rows32)),
        key=lambda index: rows32[index].tobytes(order="C"),
    )
    return np.asarray(rows32[np.asarray(order, dtype=np.int64)], dtype=np.float64)


def _canonical_group_class_indices(
    rows: np.ndarray, labels: np.ndarray, class_indices: Iterable[int]
) -> tuple[int, ...]:
    """Order class contributions by their canonical support content, not IDs."""

    keyed: list[tuple[tuple[bytes, ...], int]] = []
    for raw_index in class_indices:
        class_index = int(raw_index)
        class_rows = _canonical_float32_class_rows(rows[labels == class_index])
        class_key = tuple(
            np.ascontiguousarray(row, dtype=np.float32).tobytes(order="C")
            for row in class_rows
        )
        keyed.append((class_key, class_index))
    keyed.sort(key=lambda item: item[0])
    return tuple(item[1] for item in keyed)


def _cross_block(
    residual: np.ndarray, left: slice, right: slice, denominator: float
) -> np.ndarray:
    """Return one deterministic upper covariance block without retaining it."""

    block = np.matmul(residual[:, left].T, residual[:, right]) / denominator
    if not np.isfinite(block).all():
        raise D92CCOCNumericalError("ccoc_q_nonfinite")
    return block


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
    labels = np.asarray(targets, dtype=np.int64)
    classes = tuple(int(index) for index in class_indices)
    shots = int(k_shot)
    if (
        rows.ndim != 2
        or rows.shape[1] != 288
        or labels.shape != (len(rows),)
        or len(classes) < 2
        or shots <= 2
    ):
        raise D92CCOCError("ccoc_invalid_group_registry")
    if any(int(np.sum(labels == class_index)) != shots for class_index in classes):
        raise D92CCOCError("ccoc_unbalanced_group_registry")

    ordered_classes = _canonical_group_class_indices(rows, labels, classes)
    accumulators = [
        np.zeros((left.stop - left.start, right.stop - right.start), dtype=np.float64)
        for left, right in _UPPER_BLOCK_PAIRS
    ]
    norm_min = math.inf
    norm_max = 0.0
    denominator = float(shots - 1)

    for class_index in ordered_classes:
        class_rows = _canonical_float32_class_rows(rows[labels == class_index])
        mean = np.sum(class_rows, axis=0, dtype=np.float64) / float(shots)
        residual = class_rows - mean
        if not np.isfinite(mean).all() or not np.isfinite(residual).all():
            raise D92CCOCNumericalError("ccoc_q_nonfinite")

        norm_squared = 0.0
        for left, right in _UPPER_BLOCK_PAIRS:
            q_block = _cross_block(residual, left, right, denominator)
            block_norm = float(np.linalg.norm(q_block, ord="fro"))
            if not math.isfinite(block_norm):
                raise D92CCOCNumericalError("ccoc_q_nonfinite")
            norm_squared += block_norm * block_norm
        if not math.isfinite(norm_squared):
            raise D92CCOCNumericalError("ccoc_q_nonfinite")
        class_norm = math.sqrt(norm_squared)
        if class_norm <= 0.0:
            raise D92CCOCNumericalError("ccoc_q_zero_frobenius_norm")

        for accumulator, (left, right) in zip(accumulators, _UPPER_BLOCK_PAIRS):
            q_block = _cross_block(residual, left, right, denominator)
            accumulator += q_block / class_norm
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
    endpoint_tolerance = 64.0 * np.finfo(np.float64).eps
    if abs(rho_raw) <= endpoint_tolerance:
        rho = 0.0
    elif abs(rho_raw - 1.0) <= endpoint_tolerance:
        rho = 1.0
    else:
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


def _mix_full_and_blockdiag(covariance: np.ndarray, rho: float) -> np.ndarray:
    """Interpolate one full endpoint and its block-diagonal endpoint."""

    value = float(rho)
    if not math.isfinite(value):
        raise D92CCOCNumericalError("ccoc_rho_nonfinite")
    if value < 0.0 or value > 1.0:
        raise D92CCOCNumericalError("ccoc_rho_out_of_range")
    full = np.asarray(covariance, dtype=np.float64)
    if not np.isfinite(full).all():
        raise D92CCOCNumericalError("ccoc_endpoint_nonfinite")
    return value * full + (1.0 - value) * _blockdiag(full)


def _combine_task_covariances(
    old_covariance: np.ndarray, new_covariance: np.ndarray
) -> np.ndarray:
    """Apply the locked equal old/new registration-task mixture exactly once."""

    old = np.asarray(old_covariance, dtype=np.float64)
    new = np.asarray(new_covariance, dtype=np.float64)
    if old.shape != new.shape or old.shape != (288, 288):
        raise D92CCOCError("ccoc_task_covariance_shape_drift")
    return 0.5 * old + 0.5 * new


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
            "d92_ccoc_support_transient_bytes_upper_bound": (
                _SUPPORT_TRANSIENT_BYTES_UPPER_BOUND
            ),
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
            "support_transient_bytes_upper_bound": _SUPPORT_TRANSIENT_BYTES_UPPER_BOUND,
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
    old_covariance = _mix_full_and_blockdiag(base.old_covariance, old_rho)
    new_covariance = _mix_full_and_blockdiag(base.new_covariance, new_rho)
    covariance = _combine_task_covariances(old_covariance, new_covariance)
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

    classes, shots, old_count = int(class_count), int(k_shot), int(old_class_count)
    if classes < old_count or shots < 1:
        raise D92CCOCError("ccoc_invalid_inactive_registry")
    status = "before_exact_d81" if classes == old_count else "k1_k2_exact_d81_fallback"
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
