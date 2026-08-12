"""Support-only Cauchy-scatter OAS covariance for frozen D92 E0 FULL.

The module consumes the already transformed D81 support rows and the matching
D81 Cauchy weights.  It deliberately keeps one reusable class-scatter buffer:
there is no class-by-feature-by-feature covariance stack and no query input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class D92CauchyScatterOASError(RuntimeError):
    """Raised for a support-side CSOAS degeneration eligible for E0 fallback."""


class D92CauchyScatterOASNumericalError(D92CauchyScatterOASError):
    """A finite, structurally valid CSOAS numeric failure eligible for E0 fallback."""


@dataclass(frozen=True)
class CauchyScatterOASStatistics:
    """Sufficient statistics for one registered equal-prior FULL LDA solve."""

    classification_means: np.ndarray
    covariance: np.ndarray
    class_count: int
    k_shot: int
    old_class_count: int
    audit: dict[str, Any]


def _require_balanced_registry(
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    k_shot: int,
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Validate the frozen registered support registry before any statistic."""

    x = np.asarray(rows, dtype=np.float64)
    raw_labels = np.asarray(labels)
    classes, shots, old_count = int(class_count), int(k_shot), int(old_class_count)
    if (
        x.ndim != 2
        or x.shape[0] != classes * shots
        or raw_labels.shape != (len(x),)
        or not np.issubdtype(raw_labels.dtype, np.integer)
        or not np.isfinite(x).all()
        or classes <= old_count
        or old_count <= 0
        or shots <= 2
    ):
        raise D92CauchyScatterOASError("invalid_registered_support_registry")
    y = np.asarray(raw_labels, dtype=np.int64)
    if (
        not np.array_equal(np.unique(y), np.arange(classes, dtype=np.int64))
        or any(int(np.sum(y == index)) != shots for index in range(classes))
    ):
        raise D92CauchyScatterOASError("unbalanced_registered_support_registry")
    return x, y, classes, shots, old_count, int(x.shape[1])


def _validated_weights(
    normalized_cauchy_weights: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_count: int,
    k_shot: int,
) -> np.ndarray:
    """Use D81's normalized weights exactly; never renormalize or rescan them."""

    weights = np.asarray(normalized_cauchy_weights, dtype=np.float64)
    if (
        weights.shape != (class_count, k_shot)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.allclose(
            np.sum(weights, axis=1),
            np.ones(class_count, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise D92CauchyScatterOASError("invalid_normalized_cauchy_weights")
    return weights


def build_cauchy_scatter_oas_statistics(
    transformed_rows: np.ndarray,
    labels: np.ndarray,
    normalized_cauchy_weights: Sequence[Sequence[float]] | np.ndarray,
    *,
    class_count: int,
    k_shot: int,
    old_class_count: int = 6,
) -> CauchyScatterOASStatistics:
    """Build fixed 0.5/0.5 old/new CSOAS covariance from D81 support only.

    Classification means are the exact nonweighted means of transformed support.
    The independent weighted centers are used only for the per-class scatter.
    """

    rows, targets, classes, shots, old_count, dimension = _require_balanced_registry(
        transformed_rows,
        labels,
        class_count=class_count,
        k_shot=k_shot,
        old_class_count=old_class_count,
    )
    weights = _validated_weights(
        normalized_cauchy_weights,
        class_count=classes,
        k_shot=shots,
    )
    new_count = classes - old_count
    classification_means = np.empty((classes, dimension), dtype=np.float64)
    # `covariance` is the only group/final matrix.  `scatter` is reused for
    # every class, avoiding an O(C*p*p) materialization.
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    scatter = np.zeros((dimension, dimension), dtype=np.float64)
    weighted_residual = np.empty((shots, dimension), dtype=np.float64)
    sqrt_weight = np.empty(shots, dtype=np.float64)
    weighted_centers: list[list[float]] = []
    effective_sizes: list[float] = []
    scatter_traces: list[float] = []
    oas_taus: list[float] = []
    oas_alphas: list[float] = []
    oas_denominators: list[float] = []
    oas_rhos: list[float] = []
    shrunk_traces: list[float] = []
    old_trace_sum = 0.0
    new_trace_sum = 0.0

    for class_index in range(classes):
        indices = np.flatnonzero(targets == class_index)
        class_rows = rows[indices]
        class_weight = weights[class_index]
        classification_means[class_index] = np.mean(class_rows, axis=0)
        weighted_center = np.sum(class_weight[:, None] * class_rows, axis=0)
        squared_weight_sum = float(np.vdot(class_weight, class_weight).real)
        scatter_denominator = float(1.0 - squared_weight_sum)
        if (
            not np.isfinite(weighted_center).all()
            or not np.isfinite(squared_weight_sum)
            or scatter_denominator <= np.finfo(np.float64).eps
        ):
            raise D92CauchyScatterOASNumericalError(
                "cauchy_weight_effective_dof_degenerate"
            )
        effective_size = float(1.0 / squared_weight_sum)
        if not np.isfinite(effective_size) or effective_size <= 1.0:
            raise D92CauchyScatterOASNumericalError(
                "cauchy_effective_sample_size_degenerate"
            )

        # Exact weighted scatter through one reusable p-by-p buffer.  The
        # k-by-p residual workspace is bounded by K and does not materialize a
        # class stack of covariance matrices.
        np.sqrt(class_weight, out=sqrt_weight)
        np.subtract(class_rows, weighted_center, out=weighted_residual)
        weighted_residual *= sqrt_weight[:, None]
        np.matmul(weighted_residual.T, weighted_residual, out=scatter)
        scatter /= scatter_denominator
        if not np.isfinite(scatter).all():
            raise D92CauchyScatterOASNumericalError("weighted_scatter_nonfinite")
        scatter_trace = float(np.trace(scatter))
        tau = float(scatter_trace / dimension)
        alpha = float(np.vdot(scatter, scatter).real / (dimension * dimension))
        oas_denominator = float(
            (effective_size + 1.0) * (alpha - (tau * tau) / dimension)
        )
        if not np.isfinite(scatter_trace) or not np.isfinite(tau) or not np.isfinite(alpha):
            raise D92CauchyScatterOASNumericalError("oas_statistic_nonfinite")
        if oas_denominator <= 0.0:
            rho = 1.0
        else:
            rho_raw = float((alpha + tau * tau) / oas_denominator)
            if not np.isfinite(rho_raw):
                raise D92CauchyScatterOASNumericalError("oas_rho_nonfinite")
            rho = float(np.clip(rho_raw, 0.0, 1.0))
        scatter *= 1.0 - rho
        diagonal = np.diag_indices(dimension)
        scatter[diagonal] += rho * tau
        shrunk_trace = float(np.trace(scatter))
        trace_tolerance = max(1.0e-12, abs(scatter_trace) * 1.0e-12)
        if (
            not np.isfinite(shrunk_trace)
            or abs(shrunk_trace - scatter_trace) > trace_tolerance
        ):
            raise D92CauchyScatterOASNumericalError(
                "oas_trace_preservation_failure"
            )

        group_scale = 0.5 / (old_count if class_index < old_count else new_count)
        covariance += group_scale * scatter
        if class_index < old_count:
            old_trace_sum += shrunk_trace
        else:
            new_trace_sum += shrunk_trace
        weighted_centers.append(np.asarray(weighted_center, dtype=np.float64).tolist())
        effective_sizes.append(effective_size)
        scatter_traces.append(scatter_trace)
        oas_taus.append(tau)
        oas_alphas.append(alpha)
        oas_denominators.append(oas_denominator)
        oas_rhos.append(rho)
        shrunk_traces.append(shrunk_trace)

    # X.T @ W @ X is symmetric by construction.  Only the accumulated group
    # covariance receives one vectorized round-off symmetrization before SPD.
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or not np.isfinite(eigenvalues).all():
        raise D92CauchyScatterOASNumericalError("csoas_final_covariance_nonfinite")
    if float(np.min(eigenvalues)) <= 0.0:
        raise D92CauchyScatterOASNumericalError("csoas_final_covariance_not_spd")
    old_group_trace = float(old_trace_sum / old_count)
    new_group_trace = float(new_trace_sum / new_count)
    final_trace = float(np.trace(covariance))
    expected_trace = float(0.5 * old_group_trace + 0.5 * new_group_trace)
    if abs(final_trace - expected_trace) > max(1.0e-12, abs(expected_trace) * 1.0e-12):
        raise D92CauchyScatterOASNumericalError(
            "csoas_group_trace_balance_failure"
        )

    audit: dict[str, Any] = {
        "schema": "cvs.phase2.d92.cauchy_scatter_oas.v1",
        "d92_csoas_active": True,
        "d92_csoas_fallback_active": False,
        "d92_csoas_fallback_reason": None,
        "d92_csoas_classification_mean_policy": "nonweighted_d81_transformed_support_mean",
        "d92_csoas_scatter_center_policy": "independent_normalized_d81_cauchy_weighted_center",
        "d92_csoas_scatter_denominator_policy": "one_minus_sum_normalized_cauchy_weight_squared",
        "d92_csoas_oas_policy": "per_class_effective_dof_closed_form_preserve_trace",
        "d92_csoas_group_policy": "class_mean_then_fixed_0.5_old_0.5_new",
        "d92_csoas_support_rows": int(len(rows)),
        "d92_csoas_class_count": classes,
        "d92_csoas_k_shot": shots,
        "d92_csoas_old_class_count": old_count,
        "d92_csoas_new_class_count": new_count,
        "d92_csoas_normalized_cauchy_weight_by_class": weights.tolist(),
        "d92_csoas_weighted_center_by_class": weighted_centers,
        "d92_csoas_effective_sample_size_by_class": effective_sizes,
        "d92_csoas_scatter_trace_by_class": scatter_traces,
        "d92_csoas_oas_tau_by_class": oas_taus,
        "d92_csoas_oas_alpha_by_class": oas_alphas,
        "d92_csoas_oas_denominator_by_class": oas_denominators,
        "d92_csoas_oas_rho_by_class": oas_rhos,
        "d92_csoas_shrunk_trace_by_class": shrunk_traces,
        "d92_csoas_old_group_trace": old_group_trace,
        "d92_csoas_new_group_trace": new_group_trace,
        "d92_csoas_final_trace": final_trace,
        "d92_csoas_final_eigenvalue_min": float(np.min(eigenvalues)),
        "d92_csoas_final_eigenvalue_max": float(np.max(eigenvalues)),
        "d92_csoas_spd_pass": True,
        "d92_csoas_live_class_scatter_buffers": 1,
        "d92_csoas_class_matrix_stack": False,
        "d92_csoas_single_scatter_buffer_bytes": int(
            dimension * dimension * np.dtype(np.float64).itemsize
        ),
        "d92_csoas_final_covariance_buffer_bytes": int(
            dimension * dimension * np.dtype(np.float64).itemsize
        ),
        "d92_csoas_support_scatter_macs_upper_bound": int(
            classes
            * (
                2 * shots * dimension * dimension
                + 5 * shots * dimension
                + dimension * dimension
            )
        ),
        "d92_csoas_support_transient_bytes_upper_bound": int(
            2 * dimension * dimension * np.dtype(np.float64).itemsize
            + 2 * shots * dimension * np.dtype(np.float64).itemsize
            + 2 * shots * np.dtype(np.float64).itemsize
        ),
        "d92_csoas_query_rows_used": 0,
        "d92_csoas_query_fit_access": False,
        "d92_csoas_query_update_access": False,
        "d92_csoas_query_selection_access": False,
        "d92_csoas_query_truth_access": False,
        "d92_csoas_query_role_oracle_access": False,
        "d92_csoas_query_class_quota_access": False,
        "d92_csoas_query_global_reassignment": False,
    }
    return CauchyScatterOASStatistics(
        classification_means=classification_means,
        covariance=covariance,
        class_count=classes,
        k_shot=shots,
        old_class_count=old_count,
        audit=audit,
    )


def compile_cauchy_scatter_oas_affine(
    statistics: CauchyScatterOASStatistics,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compile the one equal-prior FULL LDA head from CSOAS statistics once."""

    if not isinstance(statistics, CauchyScatterOASStatistics):
        raise D92CauchyScatterOASError("csoas_statistics_type_drift")
    classes, dimension = statistics.classification_means.shape
    if (
        classes != int(statistics.class_count)
        or statistics.covariance.shape != (dimension, dimension)
        or not np.isfinite(statistics.classification_means).all()
        or not np.isfinite(statistics.covariance).all()
    ):
        raise D92CauchyScatterOASError("csoas_statistics_shape_drift")
    covariance = np.asarray(statistics.covariance, dtype=np.float64)
    try:
        coefficient = np.linalg.solve(covariance, statistics.classification_means.T).T
    except np.linalg.LinAlgError as error:
        raise D92CauchyScatterOASNumericalError("csoas_full_solve_failure") from error
    priors = np.full(classes, 1.0 / classes, dtype=np.float64)
    intercept = -0.5 * np.diag(statistics.classification_means @ coefficient.T)
    intercept += np.log(priors)
    equation_residual = float(
        np.max(
            np.abs(
                covariance @ coefficient.T - statistics.classification_means.T
            )
        )
    )
    coefficient -= coefficient.mean(axis=0, keepdims=True)
    intercept -= intercept.mean()
    if (
        not np.isfinite(coefficient).all()
        or not np.isfinite(intercept).all()
        or not np.isfinite(equation_residual)
    ):
        raise D92CauchyScatterOASNumericalError("csoas_full_affine_nonfinite")
    audit = dict(statistics.audit)
    audit.update(
        {
            "solver": "lsqr_equivalent_explicit_full_solve",
            "shrinkage": "per_class_effective_dof_oas",
            "prior_policy": "equal_1_over_registered_class_count",
            # D42's immutable state schema admits this locked equal-prior LDA
            # policy identifier; CSOAS provenance remains in the prefixed fields.
            "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
            "unit_covariance_fallback": False,
            "d92_csoas_covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
            "d92_csoas_full_solve_count": 1,
            "d92_csoas_full_dense_288_solve_count": 1,
            "d92_csoas_covariance_equation_residual_max": equation_residual,
            "covariance_equation_residual_max": equation_residual,
            "d92_csoas_class_common_affine_omitted_before_fp32": True,
            "d92_csoas_centered_coefficient_mean_max_abs": float(
                np.max(np.abs(coefficient.mean(axis=0)))
            ),
            "d92_csoas_centered_intercept_mean_abs": float(
                abs(intercept.mean())
            ),
        }
    )
    return (
        np.asarray(coefficient, dtype=np.float32),
        np.asarray(intercept, dtype=np.float32),
        audit,
    )


__all__ = [
    "CauchyScatterOASStatistics",
    "D92CauchyScatterOASError",
    "D92CauchyScatterOASNumericalError",
    "build_cauchy_scatter_oas_statistics",
    "compile_cauchy_scatter_oas_affine",
]
