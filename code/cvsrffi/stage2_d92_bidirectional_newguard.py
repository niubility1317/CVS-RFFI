"""Support-only affine NewGuard over the single D92 E0 FULL head.

The module deliberately has no query or scorer inputs.  Its only mutable
candidate is an old-class affine residual constrained to the orthogonal
complement of the registered-new augmented support row span plus one shared,
non-positive old-envelope intercept shift.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from cvsrffi.stage2_d42_unified_shrinkage_lda import D42UnifiedShrinkageLDAError
from cvsrffi.stage2_d92_registration_balanced_covariance import OLD_CLASS_COUNT


TAIL_FRACTION = 0.20
TAIL_QUANTILE_METHOD = "lower"
# The trust-region fraction is the single pre-registered strength used by the
# one max-min solve.  Deployment validation never searches another strength.
TRUST_REGION_FRACTION = 1.0e-4
_CLOSURE_EPS_MULTIPLIER = 128.0
_PROTECTION_EPS_MULTIPLIER = 1024.0
_D42_CODEC_MACS_PER_COEFFICIENT = 8
_SOLVER_NAME = "scipy_highs_linear_program_v1"
_MODE = "newguard_maxmin"
_CANDIDATE_ID = "d92_e0_full_bidirectional_newguard_maxmin"


class D92NewGuardError(RuntimeError):
    """Raised for invalid NewGuard input or protection-contract drift."""


class D92NewGuardNumericalError(D92NewGuardError):
    """A numerical/infeasible condition eligible for exact E0 fallback."""


def _as_validated_inputs(
    *,
    full_rows: np.ndarray,
    full_labels: np.ndarray,
    full_coefficient: np.ndarray,
    full_intercept: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Validate registry/head closure before any fallback-eligible arithmetic."""

    classes, shots = int(class_count), int(k_shot)
    rows = np.asarray(full_rows)
    labels = np.asarray(full_labels)
    coefficient = np.asarray(full_coefficient)
    intercept = np.asarray(full_intercept)
    if (
        classes <= OLD_CLASS_COUNT
        or shots <= 0
        or rows.ndim != 2
        or rows.shape[0] != classes * shots
        or labels.shape != (classes * shots,)
        or coefficient.dtype != np.float32
        or intercept.dtype != np.float32
        or coefficient.shape != (classes, rows.shape[1])
        or intercept.shape != (classes,)
        or not np.issubdtype(labels.dtype, np.integer)
    ):
        raise D92NewGuardError("D92 NewGuard registry/head shape drift")
    if not np.issubdtype(rows.dtype, np.number) or not np.isfinite(rows).all():
        raise D92NewGuardError("D92 NewGuard support rows are non-finite")
    if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise D92NewGuardError("D92 NewGuard FULL head is non-finite")
    labels64 = np.asarray(labels, dtype=np.int64)
    if (
        not np.array_equal(np.unique(labels64), np.arange(classes, dtype=np.int64))
        or any(int(np.sum(labels64 == index)) != shots for index in range(classes))
    ):
        raise D92NewGuardError("D92 NewGuard support registry is incomplete")
    return (
        np.asarray(rows, dtype=np.float64),
        labels64,
        np.asarray(coefficient, dtype=np.float32),
        np.asarray(intercept, dtype=np.float32),
        classes,
        shots,
        int(rows.shape[1]),
    )


def _validate_roundtrip(
    roundtrip: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    coefficient: np.ndarray,
    intercept: np.ndarray,
    *,
    classes: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Call the supplied deployed D42 codec without hiding contract drift."""

    try:
        result = roundtrip(
            np.asarray(coefficient, dtype=np.float32),
            np.asarray(intercept, dtype=np.float32),
        )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, D42UnifiedShrinkageLDAError) as error:
        raise D92NewGuardNumericalError("deployment_quantize_decode_numeric_failure") from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise D92NewGuardError("D92 NewGuard deployment callback contract drift")
    deployed_coefficient = np.asarray(result[0])
    deployed_intercept = np.asarray(result[1])
    if (
        deployed_coefficient.shape != (classes, dimension)
        or deployed_intercept.shape != (classes,)
    ):
        raise D92NewGuardError("D92 NewGuard deployment callback shape drift")
    if not (
        np.issubdtype(deployed_coefficient.dtype, np.number)
        and np.issubdtype(deployed_intercept.dtype, np.number)
        and np.isfinite(deployed_coefficient).all()
        and np.isfinite(deployed_intercept).all()
    ):
        raise D92NewGuardNumericalError("deployment_quantize_decode_nonfinite")
    return (
        np.asarray(deployed_coefficient, dtype=np.float32),
        np.asarray(deployed_intercept, dtype=np.float32),
    )


def _margin(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return true-versus-all-competitor margins for an already sealed support set."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise D92NewGuardNumericalError("support_margin_shape_drift")
    row_indices = np.arange(values.shape[0], dtype=np.int64)
    target = values[row_indices, targets]
    competitors = values.copy()
    competitors[row_indices, targets] = -np.inf
    result = target - np.max(competitors, axis=1)
    if not np.isfinite(result).all():
        raise D92NewGuardNumericalError("support_margin_nonfinite")
    return result


def _fixed_tail_indices(
    margins: np.ndarray, labels: np.ndarray, *, shots: int
) -> tuple[list[np.ndarray], list[float]]:
    """Freeze one lower-Q20 tail per old class from the baseline FULL head."""

    arrays: list[np.ndarray] = []
    thresholds: list[float] = []
    for old_class in range(OLD_CLASS_COUNT):
        indices = np.flatnonzero(labels == old_class)
        if indices.shape != (shots,):
            raise D92NewGuardError("D92 NewGuard old support registry drift")
        current = np.asarray(margins[indices], dtype=np.float64)
        threshold = float(
            np.quantile(current, TAIL_FRACTION, method=TAIL_QUANTILE_METHOD)
        )
        selected = indices[current <= threshold]
        if selected.size <= 0 or not np.isfinite(threshold):
            raise D92NewGuardNumericalError("fixed_tail_selection_degenerate")
        arrays.append(np.asarray(selected, dtype=np.int64))
        thresholds.append(threshold)
    return arrays, thresholds


def _compact_nullspace_operator(
    new_augmented_rows: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], int, float, int]:
    """Return projection through the compact SVD row basis, never a dense P."""

    matrix = np.asarray(new_augmented_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or not np.isfinite(matrix).all():
        raise D92NewGuardNumericalError("new_support_rowspace_input_invalid")
    try:
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise D92NewGuardNumericalError("new_support_rowspace_svd_failed") from error
    if singular_values.size == 0 or not np.isfinite(singular_values).all():
        raise D92NewGuardNumericalError("new_support_rowspace_singular_values_invalid")
    largest = float(singular_values[0])
    rank_threshold = float(
        np.finfo(np.float64).eps * max(matrix.shape) * largest
    )
    rank = int(np.sum(singular_values > rank_threshold))
    nullspace_rank = int(matrix.shape[1] - rank)
    if rank <= 0 or nullspace_rank <= 0:
        raise D92NewGuardNumericalError("new_support_nullspace_insufficient")
    row_basis = np.asarray(vt[:rank], dtype=np.float64)

    def project(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (matrix.shape[1],):
            raise D92NewGuardError("D92 NewGuard compact projection shape drift")
        result = value - row_basis.T @ (row_basis @ value)
        if not np.isfinite(result).all():
            raise D92NewGuardNumericalError("new_support_compact_projection_nonfinite")
        return result

    return project, rank, rank_threshold, nullspace_rank


def _support_scale(
    augmented_rows: np.ndarray, baseline_scores: np.ndarray
) -> tuple[float, float, float]:
    """Use one support-derived scale to bound all internal residuals and tau."""

    feature_rms = float(
        np.sqrt(np.mean(np.sum(np.asarray(augmented_rows, dtype=np.float64) ** 2, axis=1)))
    )
    score_rms = float(np.sqrt(np.mean(np.asarray(baseline_scores, dtype=np.float64) ** 2)))
    if (
        not np.isfinite(feature_rms)
        or not np.isfinite(score_rms)
        or feature_rms <= 0.0
        or score_rms <= 0.0
    ):
        raise D92NewGuardNumericalError("support_scale_degenerate")
    radius = float(TRUST_REGION_FRACTION * score_rms / feature_rms)
    if not np.isfinite(radius) or radius <= 0.0:
        raise D92NewGuardNumericalError("support_trust_region_degenerate")
    return feature_rms, score_rms, radius


def _solve_small_maxmin(
    *,
    augmented_rows: np.ndarray,
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    tail_indices: list[np.ndarray],
    directions: np.ndarray,
    trust_radius: float,
) -> tuple[np.ndarray, float, float]:
    """Solve exactly one deterministic linear max-min problem over 6+1 variables."""

    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise D92NewGuardNumericalError("small_maxmin_solver_unavailable") from error
    old_count, width = directions.shape
    if old_count != OLD_CLASS_COUNT or width != augmented_rows.shape[1]:
        raise D92NewGuardError("D92 NewGuard direction shape drift")
    # Variable a_l contributes q_l to class l and -q_l/6 to every old row.
    contribution = np.empty((old_count, old_count, width), dtype=np.float64)
    for variable in range(old_count):
        for old_class in range(old_count):
            contribution[variable, old_class] = -directions[variable] / old_count
        contribution[variable, variable] += directions[variable]
    rows: list[np.ndarray] = []
    for old_class, indices in enumerate(tail_indices):
        for row_index in indices:
            x = augmented_rows[int(row_index)]
            for competitor in range(baseline_scores.shape[1]):
                if competitor == old_class:
                    continue
                linear = np.empty(old_count + 2, dtype=np.float64)
                for variable in range(old_count):
                    target_change = float(x @ contribution[variable, old_class])
                    competitor_change = (
                        float(x @ contribution[variable, competitor])
                        if competitor < old_count
                        else 0.0
                    )
                    linear[variable] = -(target_change - competitor_change)
                linear[old_count] = -1.0 if competitor >= old_count else 0.0
                linear[old_count + 1] = 1.0
                rows.append(linear)
    new_indices = np.flatnonzero(labels >= old_count)
    for row_index in new_indices:
        x = augmented_rows[int(row_index)]
        for old_class in range(old_count):
            linear = np.empty(old_count + 2, dtype=np.float64)
            for variable in range(old_count):
                old_change = float(x @ contribution[variable, old_class])
                linear[variable] = old_change
            linear[old_count] = 1.0
            linear[old_count + 1] = 1.0
            rows.append(linear)
    if not rows:
        raise D92NewGuardNumericalError("small_maxmin_constraints_empty")
    constraints = np.stack(rows, axis=0)
    if not np.isfinite(constraints).all():
        raise D92NewGuardNumericalError("small_maxmin_constraints_nonfinite")
    variable_radius = float(trust_radius / 2.0)
    bounds = [(-variable_radius, variable_radius)] * old_count
    bounds.extend([(-trust_radius, 0.0), (0.0, None)])
    objective = np.zeros(old_count + 2, dtype=np.float64)
    objective[-1] = -1.0
    try:
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=np.zeros(constraints.shape[0], dtype=np.float64),
            bounds=bounds,
            method="highs",
        )
    except (FloatingPointError, OverflowError, ValueError) as error:
        raise D92NewGuardNumericalError("small_maxmin_solver_numeric_failure") from error
    if not bool(result.success) or result.x is None:
        raise D92NewGuardNumericalError("small_maxmin_infeasible")
    solution = np.asarray(result.x, dtype=np.float64)
    if solution.shape != (old_count + 2,) or not np.isfinite(solution).all():
        raise D92NewGuardNumericalError("small_maxmin_solution_nonfinite")
    strengths, tau, objective_value = solution[:old_count], float(solution[-2]), float(solution[-1])
    if tau > 0.0 or objective_value < 0.0:
        raise D92NewGuardNumericalError("small_maxmin_solution_constraint_drift")
    return strengths, tau, objective_value


def _apply_internal_directions(
    *, directions: np.ndarray, strengths: np.ndarray
) -> np.ndarray:
    """Map six class strengths to a six-row residual with exact FP64 zero sum."""

    scaled = np.asarray(strengths, dtype=np.float64)[:, None] * np.asarray(
        directions, dtype=np.float64
    )
    internal = scaled - scaled.mean(axis=0, keepdims=True)
    if not np.isfinite(internal).all():
        raise D92NewGuardNumericalError("internal_residual_nonfinite")
    return internal


def _head_scores(
    rows: np.ndarray, coefficient: np.ndarray, intercept: np.ndarray
) -> np.ndarray:
    scores = (
        np.asarray(rows, dtype=np.float64) @ np.asarray(coefficient, dtype=np.float64).T
        + np.asarray(intercept, dtype=np.float64)[None, :]
    )
    if not np.isfinite(scores).all():
        raise D92NewGuardNumericalError("affine_scores_nonfinite")
    return scores


def _protection_tolerances(
    *, augmented_rows: np.ndarray, coefficient: np.ndarray, intercept: np.ndarray
) -> tuple[float, float]:
    """Keep equality closure relative, but freeze inequality score tolerance."""

    closure_scale = max(
        1.0,
        float(np.max(np.abs(augmented_rows))),
        float(np.max(np.abs(coefficient))),
        float(np.max(np.abs(intercept))),
    )
    closure = float(
        _CLOSURE_EPS_MULTIPLIER * np.finfo(np.float32).eps * closure_scale
    )
    protection = float(_PROTECTION_EPS_MULTIPLIER * np.finfo(np.float32).eps)
    return closure, protection


def _protection_receipt(
    *,
    rows: np.ndarray,
    labels: np.ndarray,
    baseline_coefficient: np.ndarray,
    baseline_intercept: np.ndarray,
    candidate_coefficient: np.ndarray,
    candidate_intercept: np.ndarray,
    internal: np.ndarray,
    tau: float,
    tail_indices: list[np.ndarray],
    quantize_decode: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    baseline_deployed_coefficient: np.ndarray,
    baseline_deployed_intercept: np.ndarray,
    classes: int,
    dimension: int,
) -> dict[str, Any]:
    """Recheck raw and deployed support protection against the same frozen tail."""

    augmented = np.concatenate(
        [rows, np.ones((rows.shape[0], 1), dtype=np.float64)], axis=1
    )
    new_indices = np.flatnonzero(labels >= OLD_CLASS_COUNT)
    x_new = augmented[new_indices]
    closure_tolerance, protection_tolerance = _protection_tolerances(
        augmented_rows=augmented,
        coefficient=baseline_coefficient,
        intercept=baseline_intercept,
    )
    raw_internal = np.concatenate(
        [
            candidate_coefficient[:OLD_CLASS_COUNT].astype(np.float64)
            - baseline_coefficient[:OLD_CLASS_COUNT].astype(np.float64),
            (
                candidate_intercept[:OLD_CLASS_COUNT].astype(np.float64)
                - baseline_intercept[:OLD_CLASS_COUNT].astype(np.float64)
                - float(tau)
            )[:, None],
        ],
        axis=1,
    )
    raw_new_residual = x_new @ raw_internal.T
    raw_group_sum = np.sum(raw_internal, axis=0)
    max_raw_new_residual = float(np.max(np.abs(raw_new_residual)))
    max_raw_group_sum = float(np.max(np.abs(raw_group_sum)))
    raw_baseline_scores = _head_scores(rows, baseline_coefficient, baseline_intercept)
    raw_candidate_scores = _head_scores(rows, candidate_coefficient, candidate_intercept)
    raw_margin_delta = _margin(raw_candidate_scores, labels) - _margin(
        raw_baseline_scores, labels
    )
    raw_tail_delta = [
        float(np.min(raw_margin_delta[indices])) for indices in tail_indices
    ]
    raw_old_before = raw_baseline_scores[new_indices, :OLD_CLASS_COUNT]
    raw_old_after = raw_candidate_scores[new_indices, :OLD_CLASS_COUNT]
    raw_envelope_delta = np.max(raw_old_after, axis=1) - np.max(raw_old_before, axis=1)
    raw_new_margin_delta = _margin(raw_candidate_scores, labels)[new_indices] - _margin(
        raw_baseline_scores, labels
    )[new_indices]
    raw_envelope_error = float(np.max(np.abs(raw_envelope_delta - float(tau))))
    if (
        max_raw_new_residual > closure_tolerance
        or max_raw_group_sum > closure_tolerance
    ):
        raise D92NewGuardNumericalError("raw_protection_closure_failed")
    if (
        min(raw_tail_delta) < -protection_tolerance
        or float(np.min(raw_new_margin_delta)) < -protection_tolerance
        or raw_envelope_error > closure_tolerance
        or float(tau) > 0.0
    ):
        raise D92NewGuardNumericalError("raw_protection_failed")
    candidate_deployed_coefficient, candidate_deployed_intercept = _validate_roundtrip(
        quantize_decode,
        candidate_coefficient,
        candidate_intercept,
        classes=classes,
        dimension=dimension,
    )
    deployment_new_rows_byte_exact = bool(
        candidate_deployed_coefficient[OLD_CLASS_COUNT:].tobytes()
        == baseline_deployed_coefficient[OLD_CLASS_COUNT:].tobytes()
        and candidate_deployed_intercept[OLD_CLASS_COUNT:].tobytes()
        == baseline_deployed_intercept[OLD_CLASS_COUNT:].tobytes()
    )
    deployment_full_head_byte_exact = bool(
        candidate_deployed_coefficient.tobytes()
        == baseline_deployed_coefficient.tobytes()
        and candidate_deployed_intercept.tobytes()
        == baseline_deployed_intercept.tobytes()
    )
    deployed_internal = np.concatenate(
        [
            candidate_deployed_coefficient[:OLD_CLASS_COUNT].astype(np.float64)
            - baseline_deployed_coefficient[:OLD_CLASS_COUNT].astype(np.float64),
            (
                candidate_deployed_intercept[:OLD_CLASS_COUNT].astype(np.float64)
                - baseline_deployed_intercept[:OLD_CLASS_COUNT].astype(np.float64)
                - float(tau)
            )[:, None],
        ],
        axis=1,
    )
    max_deployed_new_residual = float(np.max(np.abs(x_new @ deployed_internal.T)))
    max_deployed_group_sum = float(np.max(np.abs(np.sum(deployed_internal, axis=0))))
    deployed_baseline_scores = _head_scores(
        rows, baseline_deployed_coefficient, baseline_deployed_intercept
    )
    deployed_candidate_scores = _head_scores(
        rows, candidate_deployed_coefficient, candidate_deployed_intercept
    )
    deployed_margin_delta = _margin(deployed_candidate_scores, labels) - _margin(
        deployed_baseline_scores, labels
    )
    deployed_tail_delta = [
        float(np.min(deployed_margin_delta[indices])) for indices in tail_indices
    ]
    deployed_old_before = deployed_baseline_scores[new_indices, :OLD_CLASS_COUNT]
    deployed_old_after = deployed_candidate_scores[new_indices, :OLD_CLASS_COUNT]
    deployed_envelope_delta = np.max(deployed_old_after, axis=1) - np.max(
        deployed_old_before, axis=1
    )
    deployed_new_margin_delta = _margin(
        deployed_candidate_scores, labels
    )[new_indices] - _margin(deployed_baseline_scores, labels)[new_indices]
    deployed_envelope_error = float(
        np.max(np.abs(deployed_envelope_delta - float(tau)))
    )
    deployment_pass = bool(
        deployment_new_rows_byte_exact
        and max_deployed_new_residual <= closure_tolerance
        and max_deployed_group_sum <= closure_tolerance
        and min(deployed_tail_delta) >= -protection_tolerance
        and float(np.min(deployed_new_margin_delta)) >= -protection_tolerance
        and deployed_envelope_error <= closure_tolerance
        and float(tau) <= 0.0
    )
    return {
        "d92_newguard_closure_tolerance": closure_tolerance,
        "d92_newguard_protection_tolerance": protection_tolerance,
        "d92_newguard_max_abs_Xnew_internal_residual": max_raw_new_residual,
        "d92_newguard_old_group_zero_sum_residual_max_abs": max_raw_group_sum,
        "d92_newguard_new_support_old_envelope_change_max_abs_error": raw_envelope_error,
        "d92_newguard_new_support_old_envelope_change_max": float(
            np.max(raw_envelope_delta)
        ),
        "d92_newguard_new_support_min_margin_change": float(
            np.min(raw_new_margin_delta)
        ),
        "d92_newguard_tail_margin_change_by_old_class": raw_tail_delta,
        "d92_newguard_deployment_max_abs_Xnew_internal_residual": max_deployed_new_residual,
        "d92_newguard_deployment_old_group_zero_sum_residual_max_abs": max_deployed_group_sum,
        "d92_newguard_deployment_new_support_old_envelope_change_max_abs_error": deployed_envelope_error,
        "d92_newguard_deployment_new_support_old_envelope_change_max": float(
            np.max(deployed_envelope_delta)
        ),
        "d92_newguard_deployment_new_support_min_margin_change": float(
            np.min(deployed_new_margin_delta)
        ),
        "d92_newguard_deployment_tail_margin_change_by_old_class": deployed_tail_delta,
        "d92_newguard_deployment_new_rows_byte_exact": deployment_new_rows_byte_exact,
        "d92_newguard_deployment_full_head_byte_exact": deployment_full_head_byte_exact,
        "d92_newguard_deployment_protection_pass": deployment_pass,
    }


def _resource_upper_bounds(
    *,
    classes: int,
    shots: int,
    dimension: int,
    new_row_count: int,
    row_rank: int,
    tail_row_count: int,
) -> tuple[int, int]:
    """Conservatively bound only NewGuard's transient support-side workspace."""

    width = int(dimension + 1)
    svd_macs = int(4 * new_row_count * width * min(new_row_count, width))
    projection_macs = int(2 * OLD_CLASS_COUNT * row_rank * width)
    constraint_count = int(
        tail_row_count * (classes - 1) + new_row_count * OLD_CLASS_COUNT
    )
    solve_macs = int(constraint_count * (OLD_CLASS_COUNT + 2) * 8)
    macs = int(svd_macs + projection_macs + solve_macs)
    transient = int(
        new_row_count * width * 8
        + min(new_row_count, width) * width * 8
        + OLD_CLASS_COUNT * width * 8
        + constraint_count * (OLD_CLASS_COUNT + 2) * 8
        + 4 * classes * shots * dimension * 8
    )
    return macs, transient


def _deployment_codec_macs_upper_bound(
    *, roundtrip_count: int, classes: int, dimension: int
) -> int:
    """Conservatively charge each real D42 two-level codec traversal."""

    return int(
        int(roundtrip_count)
        * _D42_CODEC_MACS_PER_COEFFICIENT
        * int(classes)
        * int(dimension)
    )


def _fallback_audit(
    *,
    reason: str,
    classes: int,
    shots: int,
    dimension: int,
    full_head_byte_exact: bool = True,
) -> dict[str, Any]:
    """Stable exact-E0 receipt for a numerical or explicit low-K fallback."""

    return {
        "d92_newguard_candidate_id": _CANDIDATE_ID,
        "d92_newguard_mode": _MODE,
        "d92_newguard_active": False,
        "d92_newguard_nullspace_operator": "compact_rowspace_svd",
        "d92_newguard_nullspace_rank": None,
        "d92_newguard_rank_threshold": None,
        "d92_newguard_explicit_projector_bytes": 0,
        "d92_newguard_tail_fraction": TAIL_FRACTION,
        "d92_newguard_tail_quantile_method": TAIL_QUANTILE_METHOD,
        "d92_newguard_tau_old_envelope_shift": 0.0,
        "d92_newguard_new_rows_byte_exact": True,
        "d92_newguard_deployment_new_rows_byte_exact": None,
        "d92_newguard_deployment_full_head_byte_exact": True,
        "d92_newguard_deployment_strength_scale": None,
        "d92_newguard_deployment_candidate_count": 0,
        "d92_newguard_deployment_codec_roundtrip_count": 0,
        "d92_newguard_deployment_codec_macs_upper_bound": 0,
        "d92_newguard_max_abs_Xnew_internal_residual": None,
        "d92_newguard_old_group_zero_sum_residual_max_abs": None,
        "d92_newguard_new_support_old_envelope_change_max_abs_error": None,
        "d92_newguard_new_support_old_envelope_change_max": None,
        "d92_newguard_new_support_min_margin_change": None,
        "d92_newguard_tail_margin_change_by_old_class": None,
        "d92_newguard_deployment_max_abs_Xnew_internal_residual": None,
        "d92_newguard_deployment_old_group_zero_sum_residual_max_abs": None,
        "d92_newguard_deployment_new_support_old_envelope_change_max_abs_error": None,
        "d92_newguard_deployment_new_support_old_envelope_change_max": None,
        "d92_newguard_deployment_new_support_min_margin_change": None,
        "d92_newguard_deployment_tail_margin_change_by_old_class": None,
        "d92_newguard_residual_l2_by_old_class": None,
        "d92_newguard_maxmin_objective": None,
        "d92_newguard_maxmin_solver": _SOLVER_NAME,
        "d92_newguard_maxmin_solve_count": 0,
        "d92_newguard_support_feature_rms": None,
        "d92_newguard_support_score_rms": None,
        "d92_newguard_trust_region_radius": None,
        "d92_newguard_trust_region_utilization": 0.0,
        "d92_newguard_closure_tolerance": None,
        "d92_newguard_protection_tolerance": None,
        "d92_newguard_deployment_protection_pass": False,
        "d92_newguard_fallback_active": True,
        "d92_newguard_fallback_reason": str(reason),
        "d92_newguard_full_head_byte_exact": bool(full_head_byte_exact),
        "d92_newguard_support_optimization_macs_upper_bound": 0,
        "d92_newguard_support_transient_bytes_upper_bound": 0,
        "d92_newguard_persistent_state_bytes_delta": 0,
        "d92_newguard_query_rows_used": 0,
        "d92_newguard_query_macs": int(classes * dimension),
        "d92_newguard_query_fit_access": False,
        "d92_newguard_query_update_access": False,
        "d92_newguard_query_selection_access": False,
        "d92_newguard_query_truth_access": False,
        "d92_newguard_query_role_oracle_access": False,
        "d92_newguard_query_class_quota_access": False,
        "d92_newguard_query_global_reassignment": False,
        "d92_newguard_full_component_fit_count": 1,
        "d92_newguard_support_rows": int(classes * shots),
        "d92_newguard_feature_dimension": int(dimension),
    }


def newguard_inactive_receipt(
    *, reason: str, class_count: int, k_shot: int, feature_dimension: int
) -> dict[str, Any]:
    """Return the stable no-residual receipt for before/K1/K2 lifecycle states."""

    receipt = _fallback_audit(
        reason=str(reason),
        classes=int(class_count),
        shots=int(k_shot),
        dimension=int(feature_dimension),
    )
    receipt["d92_newguard_fallback_active"] = False
    receipt["d92_newguard_full_head_byte_exact"] = True
    return receipt


def build_bidirectional_newguard_affine_state(
    *,
    full_rows: np.ndarray,
    full_labels: np.ndarray,
    full_coefficient: np.ndarray,
    full_intercept: np.ndarray,
    class_count: int,
    k_shot: int,
    quantize_decode: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one support-only NewGuard head or byte-exactly fall back to E0.

    The caller is responsible for supplying the *single* FULL component's
    already metric/center-transformed support rows.  Registry/head contract
    drift remains an exception.  Only numerical/infeasible construction and
    deployed-head protection failures return an exact E0 fallback receipt.
    """

    (
        rows,
        labels,
        baseline_coefficient,
        baseline_intercept,
        classes,
        shots,
        dimension,
    ) = _as_validated_inputs(
        full_rows=full_rows,
        full_labels=full_labels,
        full_coefficient=full_coefficient,
        full_intercept=full_intercept,
        class_count=class_count,
        k_shot=k_shot,
    )
    baseline_coefficient_exact = baseline_coefficient.copy()
    baseline_intercept_exact = baseline_intercept.copy()
    if shots <= 2:
        return (
            baseline_coefficient_exact,
            baseline_intercept_exact,
            _fallback_audit(
                reason="K1_K2_EXACT_D92_FULL_ALIAS",
                classes=classes,
                shots=shots,
                dimension=dimension,
            ),
        )
    try:
        augmented = np.concatenate(
            [rows, np.ones((rows.shape[0], 1), dtype=np.float64)], axis=1
        )
        baseline_scores = _head_scores(rows, baseline_coefficient, baseline_intercept)
        baseline_margins = _margin(baseline_scores, labels)
        tail_indices, tail_thresholds = _fixed_tail_indices(
            baseline_margins, labels, shots=shots
        )
        new_indices = np.flatnonzero(labels >= OLD_CLASS_COUNT)
        project, row_rank, rank_threshold, nullspace_rank = _compact_nullspace_operator(
            augmented[new_indices]
        )
        directions = np.empty((OLD_CLASS_COUNT, dimension + 1), dtype=np.float64)
        for old_class, indices in enumerate(tail_indices):
            projected = project(np.mean(augmented[indices], axis=0))
            norm = float(np.linalg.norm(projected))
            if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
                raise D92NewGuardNumericalError("old_tail_direction_degenerate")
            directions[old_class] = projected / norm
        feature_rms, score_rms, trust_radius = _support_scale(
            augmented, baseline_scores
        )
        strengths, tau_solution, objective_value = _solve_small_maxmin(
            augmented_rows=augmented,
            labels=labels,
            baseline_scores=baseline_scores,
            tail_indices=tail_indices,
            directions=directions,
            trust_radius=trust_radius,
        )
        tau = min(0.0, float(np.float32(tau_solution)))
        internal = _apply_internal_directions(
            directions=directions, strengths=strengths
        )
        residual_norms = np.linalg.norm(internal, axis=1)
        if (
            not np.isfinite(residual_norms).all()
            or float(np.max(residual_norms)) > trust_radius + np.finfo(np.float64).eps
        ):
            raise D92NewGuardNumericalError("internal_trust_region_exceeded")
        candidate_coefficient64 = baseline_coefficient.astype(np.float64).copy()
        candidate_intercept64 = baseline_intercept.astype(np.float64).copy()
        candidate_coefficient64[:OLD_CLASS_COUNT] += internal[:, :dimension]
        candidate_intercept64[:OLD_CLASS_COUNT] += internal[:, dimension] + tau
        candidate_coefficient = np.asarray(candidate_coefficient64, dtype=np.float32)
        candidate_intercept = np.asarray(candidate_intercept64, dtype=np.float32)
        if not np.isfinite(candidate_coefficient).all() or not np.isfinite(
            candidate_intercept
        ).all():
            raise D92NewGuardNumericalError("candidate_affine_nonfinite")
        new_rows_byte_exact = bool(
            candidate_coefficient[OLD_CLASS_COUNT:].tobytes()
            == baseline_coefficient_exact[OLD_CLASS_COUNT:].tobytes()
            and candidate_intercept[OLD_CLASS_COUNT:].tobytes()
            == baseline_intercept_exact[OLD_CLASS_COUNT:].tobytes()
        )
        if not new_rows_byte_exact:
            raise D92NewGuardNumericalError("new_rows_byte_drift")
        support_macs, support_transient = _resource_upper_bounds(
            classes=classes,
            shots=shots,
            dimension=dimension,
            new_row_count=int(new_indices.size),
            row_rank=row_rank,
            tail_row_count=int(sum(len(indices) for indices in tail_indices)),
        )
        baseline_deployed_coefficient, baseline_deployed_intercept = _validate_roundtrip(
            quantize_decode,
            baseline_coefficient_exact,
            baseline_intercept_exact,
            classes=classes,
            dimension=dimension,
        )
        protection = _protection_receipt(
            rows=rows,
            labels=labels,
            baseline_coefficient=baseline_coefficient_exact,
            baseline_intercept=baseline_intercept_exact,
            candidate_coefficient=candidate_coefficient,
            candidate_intercept=candidate_intercept,
            internal=internal,
            tau=tau,
            tail_indices=tail_indices,
            quantize_decode=quantize_decode,
            baseline_deployed_coefficient=baseline_deployed_coefficient,
            baseline_deployed_intercept=baseline_deployed_intercept,
            classes=classes,
            dimension=dimension,
        )
        if (
            not protection["d92_newguard_deployment_protection_pass"]
            or protection["d92_newguard_deployment_full_head_byte_exact"]
        ):
            fallback = _fallback_audit(
                reason="deployment_protection_failed",
                classes=classes,
                shots=shots,
                dimension=dimension,
            )
            if protection is not None:
                fallback.update(protection)
            fallback.update(
                {
                    "d92_newguard_deployment_strength_scale": None,
                    "d92_newguard_deployment_candidate_count": 1,
                    "d92_newguard_deployment_full_head_byte_exact": True,
                }
            )
            codec_roundtrips = 2
            codec_macs = _deployment_codec_macs_upper_bound(
                roundtrip_count=codec_roundtrips,
                classes=classes,
                dimension=dimension,
            )
            fallback.update(
                {
                    "d92_newguard_deployment_codec_roundtrip_count": codec_roundtrips,
                    "d92_newguard_deployment_codec_macs_upper_bound": codec_macs,
                    "d92_newguard_support_optimization_macs_upper_bound": (
                        support_macs + codec_macs
                    ),
                    "d92_newguard_support_transient_bytes_upper_bound": (
                        support_transient
                    ),
                }
            )
            return baseline_coefficient_exact, baseline_intercept_exact, fallback
        codec_roundtrips = 2
        codec_macs = _deployment_codec_macs_upper_bound(
            roundtrip_count=codec_roundtrips,
            classes=classes,
            dimension=dimension,
        )
        audit = {
            "d92_newguard_candidate_id": _CANDIDATE_ID,
            "d92_newguard_mode": _MODE,
            "d92_newguard_active": True,
            "d92_newguard_nullspace_operator": "compact_rowspace_svd",
            "d92_newguard_nullspace_rank": nullspace_rank,
            "d92_newguard_rowspace_rank": row_rank,
            "d92_newguard_rank_threshold": rank_threshold,
            "d92_newguard_explicit_projector_bytes": 0,
            "d92_newguard_tail_fraction": TAIL_FRACTION,
            "d92_newguard_tail_quantile_method": TAIL_QUANTILE_METHOD,
            "d92_newguard_tail_threshold_by_old_class": tail_thresholds,
            "d92_newguard_tail_count_by_old_class": [
                int(len(indices)) for indices in tail_indices
            ],
            "d92_newguard_tau_old_envelope_shift": tau,
            "d92_newguard_deployment_strength_scale": 1.0,
            "d92_newguard_deployment_candidate_count": 1,
            "d92_newguard_deployment_codec_roundtrip_count": codec_roundtrips,
            "d92_newguard_deployment_codec_macs_upper_bound": codec_macs,
            "d92_newguard_new_rows_byte_exact": new_rows_byte_exact,
            "d92_newguard_residual_l2_by_old_class": [
                float(value) for value in residual_norms
            ],
            "d92_newguard_maxmin_objective": objective_value,
            "d92_newguard_maxmin_solver": _SOLVER_NAME,
            "d92_newguard_maxmin_solve_count": 1,
            "d92_newguard_support_feature_rms": feature_rms,
            "d92_newguard_support_score_rms": score_rms,
            "d92_newguard_trust_region_radius": trust_radius,
            "d92_newguard_trust_region_utilization": float(
                max(
                    float(np.max(residual_norms)) / trust_radius,
                    abs(tau) / trust_radius,
                )
            ),
            "d92_newguard_fallback_active": False,
            "d92_newguard_fallback_reason": None,
            "d92_newguard_full_head_byte_exact": bool(
                candidate_coefficient.tobytes() == baseline_coefficient_exact.tobytes()
                and candidate_intercept.tobytes() == baseline_intercept_exact.tobytes()
            ),
            "d92_newguard_support_optimization_macs_upper_bound": (
                support_macs + codec_macs
            ),
            "d92_newguard_support_transient_bytes_upper_bound": support_transient,
            "d92_newguard_persistent_state_bytes_delta": 0,
            "d92_newguard_query_rows_used": 0,
            "d92_newguard_query_macs": int(classes * dimension),
            "d92_newguard_query_fit_access": False,
            "d92_newguard_query_update_access": False,
            "d92_newguard_query_selection_access": False,
            "d92_newguard_query_truth_access": False,
            "d92_newguard_query_role_oracle_access": False,
            "d92_newguard_query_class_quota_access": False,
            "d92_newguard_query_global_reassignment": False,
            "d92_newguard_full_component_fit_count": 1,
            "d92_newguard_support_rows": int(classes * shots),
            "d92_newguard_feature_dimension": dimension,
            **protection,
        }
        return candidate_coefficient, candidate_intercept, audit
    except D92NewGuardNumericalError as error:
        return (
            baseline_coefficient_exact,
            baseline_intercept_exact,
            _fallback_audit(
                reason=str(error),
                classes=classes,
                shots=shots,
                dimension=dimension,
            ),
        )


__all__ = [
    "D92NewGuardError",
    "D92NewGuardNumericalError",
    "TAIL_FRACTION",
    "TAIL_QUANTILE_METHOD",
    "TRUST_REGION_FRACTION",
    "build_bidirectional_newguard_affine_state",
    "newguard_inactive_receipt",
]
