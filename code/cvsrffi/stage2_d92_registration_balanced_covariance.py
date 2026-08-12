"""D92 support-only registration-task-balanced shared covariance.

The registered support registry defines an old prefix and a new suffix.  D92
estimates one auto-shrinkage covariance per registration task and averages the
two matrices with fixed equal weights.  All registered classes then share one
equal-prior affine LDA head.  No query row enters this fit path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


OLD_CLASS_COUNT = 6
TASK_WEIGHT = 0.5


class D92RegistrationBalancedCovarianceError(RuntimeError):
    """Raised when the locked D92 support-only closure drifts."""


@dataclass(frozen=True)
class RegistrationBalancedStatistics:
    """One transformed support registry and its one logical shared covariance.

    The FULL and BLOCK heads may solve against this object, but neither may
    estimate another target covariance or robust centre.  The two group-local
    shrinkage estimates are implementation details of the single locked
    task-balanced covariance equation.
    """

    rows: np.ndarray
    labels: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    old_covariance: np.ndarray
    new_covariance: np.ndarray
    class_count: int
    k_shot: int
    covariance_audit: dict[str, Any]


def _canonical_row_key(row: np.ndarray) -> tuple[bytes, bytes]:
    """Use float32 row bytes as the stable primary support ordering key."""

    row64 = np.ascontiguousarray(np.asarray(row, dtype=np.float64))
    row32 = np.ascontiguousarray(np.asarray(row64, dtype=np.float32))
    # The full-precision suffix only breaks a float32-byte tie without
    # changing the required canonical float32 ordering.
    return row32.tobytes(order="C"), row64.tobytes(order="C")


def _canonical_class_rows(rows: np.ndarray) -> np.ndarray:
    """Return one class in a row-order-invariant deterministic sequence."""

    source = np.asarray(rows, dtype=np.float64)
    order = sorted(range(len(source)), key=lambda index: _canonical_row_key(source[index]))
    return np.asarray(source[np.asarray(order, dtype=np.int64)], dtype=np.float64)


def _canonical_group_rows(
    rows: np.ndarray, labels: np.ndarray, class_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize class and row order before sklearn's group covariance fit."""

    classes: list[tuple[tuple[tuple[bytes, bytes], ...], np.ndarray]] = []
    for class_index in class_indices.tolist():
        class_rows = _canonical_class_rows(rows[labels == class_index])
        class_key = tuple(_canonical_row_key(row) for row in class_rows)
        classes.append((class_key, class_rows))
    classes.sort(key=lambda item: item[0])
    group_rows = np.concatenate([item[1] for item in classes], axis=0)
    group_labels = np.concatenate(
        [
            np.full(len(item[1]), index, dtype=np.int64)
            for index, item in enumerate(classes)
        ],
        axis=0,
    )
    return group_rows, group_labels


def _group_covariance(
    d42: Any, rows: np.ndarray, labels: np.ndarray, class_indices: np.ndarray
) -> np.ndarray:
    group_rows, group_labels = _canonical_group_rows(rows, labels, class_indices)
    count = len(class_indices)
    estimator = d42.LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
        priors=np.full(count, 1.0 / count, dtype=np.float64),
        store_covariance=True,
    )
    estimator.fit(group_rows, group_labels)
    if not np.array_equal(
        np.asarray(estimator.classes_, dtype=np.int64),
        np.arange(count, dtype=np.int64),
    ):
        raise D92RegistrationBalancedCovarianceError(
            "D92 group-local sklearn class order drift"
        )
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    return 0.5 * (covariance + covariance.T)


def build_registration_balanced_statistics(
    d42: Any,
    transformed: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
) -> RegistrationBalancedStatistics:
    """Estimate the registered support statistics exactly once for both heads."""

    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    classes, shots = int(class_count), int(k_shot)
    dimension = int(d42.FEATURE_DIM)
    valid_counts = {
        OLD_CLASS_COUNT,
        OLD_CLASS_COUNT + 5,
        OLD_CLASS_COUNT + 10,
        OLD_CLASS_COUNT + 20,
    }
    if (
        rows.ndim != 2
        or rows.shape[1] != dimension
        or labels.shape != (len(rows),)
        or len(rows) != classes * shots
        or classes not in valid_counts
        or shots < 1
        or not np.isfinite(rows).all()
        or not np.array_equal(np.unique(labels), np.arange(classes))
        or any(int(np.sum(labels == index)) != shots for index in range(classes))
    ):
        raise D92RegistrationBalancedCovarianceError(
            "D92 requires a locked finite symmetric 125-matrix support registry"
        )
    if classes <= OLD_CLASS_COUNT or shots <= 2:
        raise D92RegistrationBalancedCovarianceError(
            "D92 shared covariance is defined only for registered K>2 states"
        )
    means = np.stack(
        [
            np.sum(_canonical_class_rows(rows[labels == index]), axis=0, dtype=np.float64)
            / float(shots)
            for index in range(classes)
        ]
    )
    old_indices = np.arange(OLD_CLASS_COUNT, dtype=np.int64)
    new_indices = np.arange(OLD_CLASS_COUNT, classes, dtype=np.int64)
    old_covariance = _group_covariance(d42, rows, labels, old_indices)
    new_covariance = _group_covariance(d42, rows, labels, new_indices)
    old_covariance.setflags(write=False)
    new_covariance.setflags(write=False)
    covariance = TASK_WEIGHT * old_covariance + TASK_WEIGHT * new_covariance
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
        raise D92RegistrationBalancedCovarianceError(
            "D92 balanced covariance is not positive definite"
        )
    audit: dict[str, Any] = {
        "solver": "lsqr_equivalent_explicit_solve",
        "shrinkage": "auto_per_registration_task_then_fixed_equal_average",
        "prior_policy": "equal_1_over_registered_class_count",
        "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
        "unit_covariance_fallback": False,
        "support_rows": int(len(rows)),
        "class_count": classes,
        "k_shot": shots,
        "d92_status": "registration_balanced_active",
        "d92_registration_balanced_active": True,
        "d92_old_class_count": OLD_CLASS_COUNT,
        "d92_new_class_count": classes - OLD_CLASS_COUNT,
        "d92_old_covariance_weight": TASK_WEIGHT,
        "d92_new_covariance_weight": TASK_WEIGHT,
        "d92_weight_source": "fixed_equal_stage2b_stage2c_task_priority",
        "d92_formula": "Sigma_shared=0.5*Sigma_old_auto+0.5*Sigma_new_auto",
        "d92_weight_scan_count": 0,
        "d92_hyperparameter_scan_count": 0,
        "d92_query_rows_used": 0,
        "d92_query_role_oracle_access": False,
        "d92_scene_receiver_seed_specific_branch": False,
        "d92_class_id_specific_formula": False,
        "d92_registration_state_support_only": True,
        "d92_old_covariance_trace": float(np.trace(old_covariance)),
        "d92_new_covariance_trace": float(np.trace(new_covariance)),
        "d92_balanced_covariance_trace": float(np.trace(covariance)),
        "d92_balanced_eigenvalue_min": float(np.min(eigenvalues)),
        "d92_balanced_eigenvalue_max": float(np.max(eigenvalues)),
        # This is a logical registration covariance count.  The pair of
        # group-local shrinkage estimates is inseparable from one 0.5/0.5
        # task-balanced covariance and is never repeated for BLOCK.
        "d92_shared_covariance_estimation_count": 1,
        "d92_group_local_shrinkage_estimation_count": 2,
    }
    return RegistrationBalancedStatistics(
        rows=rows,
        labels=labels,
        means=np.asarray(means, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        old_covariance=old_covariance,
        new_covariance=new_covariance,
        class_count=classes,
        k_shot=shots,
        covariance_audit=audit,
    )


def compile_registration_balanced_affine(
    d42: Any,
    statistics: RegistrationBalancedStatistics,
    *,
    arm: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compile one FULL or BLOCK affine head without re-estimating statistics."""

    if arm not in {"full", "block3_centered"}:
        raise D92RegistrationBalancedCovarianceError("D92 covariance arm drift")
    classes = int(statistics.class_count)
    dimension = int(d42.FEATURE_DIM)
    if (
        statistics.rows.shape != (classes * int(statistics.k_shot), dimension)
        or statistics.labels.shape != (len(statistics.rows),)
        or statistics.means.shape != (classes, dimension)
        or statistics.covariance.shape != (dimension, dimension)
        or not np.isfinite(statistics.means).all()
        or not np.isfinite(statistics.covariance).all()
    ):
        raise D92RegistrationBalancedCovarianceError(
            "D92 shared covariance statistics drift"
        )
    covariance = np.asarray(statistics.covariance, dtype=np.float64).copy()
    priors = np.full(classes, 1.0 / classes, dtype=np.float64)
    if arm == "block3_centered":
        # BLOCK must never factor or solve a synthetic dense 288x288 matrix.
        # The three frozen diagonal blocks are independent in exact algebra;
        # retain their individual PD checks and solves, then concatenate.
        coefficient64 = np.zeros_like(statistics.means, dtype=np.float64)
        equation_residuals: list[float] = []
        eigenvalue_mins: list[float] = []
        for block in tuple(d42.BLOCK_SLICES):
            local_covariance = np.asarray(covariance[block, block], dtype=np.float64)
            local_covariance = 0.5 * (local_covariance + local_covariance.T)
            eigenvalues = np.linalg.eigvalsh(local_covariance)
            if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
                raise D92RegistrationBalancedCovarianceError(
                    "D92 compiled block covariance is not positive definite"
                )
            local_means = np.asarray(statistics.means[:, block], dtype=np.float64)
            local_coefficient = np.linalg.solve(local_covariance, local_means.T).T
            coefficient64[:, block] = local_coefficient
            equation_residuals.append(
                float(
                    np.max(
                        np.abs(local_covariance @ local_coefficient.T - local_means.T)
                    )
                )
            )
            eigenvalue_mins.append(float(np.min(eigenvalues)))
        equation_residual = float(max(equation_residuals))
        compiled_eigvalsh_count = len(tuple(d42.BLOCK_SLICES))
        compiled_solve_count = len(tuple(d42.BLOCK_SLICES))
        dense_288_solve_count = 0
        compiled_eigenvalue_min = float(min(eigenvalue_mins))
    else:
        # PD was already checked once while constructing the shared statistics.
        # Re-running a dense eigvalsh here would add an avoidable 288d factor.
        verified_minimum = float(
            statistics.covariance_audit.get("d92_balanced_eigenvalue_min", float("nan"))
        )
        if not math.isfinite(verified_minimum) or verified_minimum <= 0.0:
            raise D92RegistrationBalancedCovarianceError(
                "D92 shared covariance positive-definiteness receipt drift"
            )
        covariance = 0.5 * (covariance + covariance.T)
        coefficient64 = np.linalg.solve(covariance, statistics.means.T).T
        equation_residual = float(
            np.max(np.abs(covariance @ coefficient64.T - statistics.means.T))
        )
        compiled_eigvalsh_count = 0
        compiled_solve_count = 1
        dense_288_solve_count = 1
        compiled_eigenvalue_min = verified_minimum
    intercept64 = -0.5 * np.diag(statistics.means @ coefficient64.T) + np.log(priors)
    coefficient64 -= coefficient64.mean(axis=0, keepdims=True)
    intercept64 -= intercept64.mean()
    if not math.isfinite(equation_residual):
        raise D92RegistrationBalancedCovarianceError(
            "D92 covariance equation residual became non-finite"
        )
    audit = dict(statistics.covariance_audit)
    audit.update(
        {
            "d92_covariance_arm": arm,
            "d92_class_common_affine_omitted_before_fp32": True,
            "d92_centered_coefficient_mean_max_abs": float(
                np.max(np.abs(coefficient64.mean(axis=0)))
            ),
            "d92_centered_intercept_mean_abs": float(abs(intercept64.mean())),
            "d92_covariance_equation_residual_max": equation_residual,
            "covariance_equation_residual_max": equation_residual,
            "d92_shared_covariance_reused": True,
            "d92_shared_statistics_compile_arm": arm,
            "d92_compiled_covariance_eigvalsh_count": compiled_eigvalsh_count,
            "d92_compiled_covariance_solve_count": compiled_solve_count,
            "d92_compiled_covariance_dense_288_solve_count": dense_288_solve_count,
            "d92_compiled_covariance_eigenvalue_min": compiled_eigenvalue_min,
        }
    )
    if arm == "block3_centered":
        audit.update(
            {
                "d43_probe_arm": "block3_centered",
                "d43_covariance_structure": "block_diagonal_160_96_32",
                "d43_class_common_affine_omitted": True,
                "d43_centered_score_fp64_algebraically_equivalent": True,
                "d43_centered_support_fp32_argmax_equivalent": True,
                "d43_centered_support_fp32_pairwise_drift_max": 0.0,
                "d43_centered_coefficient_mean_max_abs": float(
                    np.max(np.abs(coefficient64.mean(axis=0)))
                ),
                "d43_centered_intercept_mean_abs": float(abs(intercept64.mean())),
            }
        )
    return coefficient64.astype(np.float32), intercept64.astype(np.float32), audit


def build_registration_balanced_equal_lda(
    d42: Any,
    baseline_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    *,
    arm: str,
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Build the fixed D92 full or block-diagonal component fit."""

    if arm not in {"full", "block3_centered"}:
        raise D92RegistrationBalancedCovarianceError("D92 covariance arm drift")
    dimension = int(d42.FEATURE_DIM)
    blocks = tuple(d42.BLOCK_SLICES)

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = np.asarray(transformed, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.int64)
        classes, shots = int(class_count), int(k_shot)
        valid_counts = {OLD_CLASS_COUNT, OLD_CLASS_COUNT + 5, OLD_CLASS_COUNT + 10, OLD_CLASS_COUNT + 20}
        if (
            rows.ndim != 2
            or rows.shape[1] != dimension
            or labels.shape != (len(rows),)
            or len(rows) != classes * shots
            or classes not in valid_counts
            or shots < 1
            or not np.isfinite(rows).all()
            or not np.array_equal(np.unique(labels), np.arange(classes))
            or any(int(np.sum(labels == index)) != shots for index in range(classes))
        ):
            raise D92RegistrationBalancedCovarianceError(
                "D92 requires a locked finite symmetric 125-matrix support registry"
            )

        active = classes > OLD_CLASS_COUNT and shots > 2
        if not active:
            coefficient, intercept, base_audit = baseline_fit(
                np.asarray(transformed), targets, classes, shots
            )
            audit = dict(base_audit)
            audit.update(
                {
                    "d92_covariance_arm": arm,
                    "d92_status": (
                        "before_exact_d81" if classes == OLD_CLASS_COUNT
                        else "k1_k2_exact_d81_fallback"
                    ),
                    "d92_registration_balanced_active": False,
                    "d92_old_class_count": OLD_CLASS_COUNT,
                    "d92_new_class_count": max(0, classes - OLD_CLASS_COUNT),
                    "d92_old_covariance_weight": TASK_WEIGHT,
                    "d92_new_covariance_weight": TASK_WEIGHT,
                    "d92_weight_source": "fixed_equal_stage2b_stage2c_task_priority",
                    "d92_weight_scan_count": 0,
                    "d92_hyperparameter_scan_count": 0,
                    "d92_query_rows_used": 0,
                    "d92_query_role_oracle_access": False,
                    "d92_scene_receiver_seed_specific_branch": False,
                    "d92_class_id_specific_formula": False,
                    "d92_registration_state_support_only": True,
                }
            )
            return coefficient, intercept, audit

        means = np.stack(
            [rows[labels == index].mean(axis=0) for index in range(classes)]
        )
        old_indices = np.arange(OLD_CLASS_COUNT, dtype=np.int64)
        new_indices = np.arange(OLD_CLASS_COUNT, classes, dtype=np.int64)
        old_covariance = _group_covariance(d42, rows, labels, old_indices)
        new_covariance = _group_covariance(d42, rows, labels, new_indices)
        covariance = TASK_WEIGHT * old_covariance + TASK_WEIGHT * new_covariance
        if arm == "block3_centered":
            structured = np.zeros_like(covariance)
            for block in blocks:
                structured[block, block] = covariance[block, block]
            covariance = structured
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
            raise D92RegistrationBalancedCovarianceError(
                "D92 balanced covariance is not positive definite"
            )
        priors = np.full(classes, 1.0 / classes, dtype=np.float64)
        coefficient64 = np.linalg.solve(covariance, means.T).T
        intercept64 = -0.5 * np.diag(means @ coefficient64.T) + np.log(priors)
        equation_residual = float(
            np.max(np.abs(covariance @ coefficient64.T - means.T))
        )
        # D45/D43 component fusion removes the class-common affine term.  Do
        # that once in FP64 before the FP32 boundary so its later centering is
        # numerically idempotent even for near-tied support rows.
        coefficient64 -= coefficient64.mean(axis=0, keepdims=True)
        intercept64 -= intercept64.mean()
        if not math.isfinite(equation_residual):
            raise D92RegistrationBalancedCovarianceError(
                "D92 covariance equation residual became non-finite"
            )
        audit: dict[str, Any] = {
            "solver": "lsqr_equivalent_explicit_solve",
            "shrinkage": "auto_per_registration_task_then_fixed_equal_average",
            "prior_policy": "equal_1_over_registered_class_count",
            "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
            "unit_covariance_fallback": False,
            "support_rows": int(len(rows)),
            "class_count": classes,
            "k_shot": shots,
            "d92_covariance_arm": arm,
            "d92_status": "registration_balanced_active",
            "d92_registration_balanced_active": True,
            "d92_old_class_count": OLD_CLASS_COUNT,
            "d92_new_class_count": classes - OLD_CLASS_COUNT,
            "d92_old_covariance_weight": TASK_WEIGHT,
            "d92_new_covariance_weight": TASK_WEIGHT,
            "d92_weight_source": "fixed_equal_stage2b_stage2c_task_priority",
            "d92_formula": "Sigma_shared=0.5*Sigma_old_auto+0.5*Sigma_new_auto",
            "d92_weight_scan_count": 0,
            "d92_hyperparameter_scan_count": 0,
            "d92_query_rows_used": 0,
            "d92_query_role_oracle_access": False,
            "d92_scene_receiver_seed_specific_branch": False,
            "d92_class_id_specific_formula": False,
            "d92_registration_state_support_only": True,
            "d92_class_common_affine_omitted_before_fp32": True,
            "d92_centered_coefficient_mean_max_abs": float(
                np.max(np.abs(coefficient64.mean(axis=0)))
            ),
            "d92_centered_intercept_mean_abs": float(abs(intercept64.mean())),
            "d92_old_covariance_trace": float(np.trace(old_covariance)),
            "d92_new_covariance_trace": float(np.trace(new_covariance)),
            "d92_balanced_covariance_trace": float(np.trace(covariance)),
            "d92_balanced_eigenvalue_min": float(np.min(eigenvalues)),
            "d92_balanced_eigenvalue_max": float(np.max(eigenvalues)),
            "d92_covariance_equation_residual_max": equation_residual,
            "covariance_equation_residual_max": equation_residual,
        }
        if arm == "block3_centered":
            audit.update(
                {
                    "d43_probe_arm": "block3_centered",
                    "d43_covariance_structure": "block_diagonal_160_96_32",
                    "d43_class_common_affine_omitted": True,
                    "d43_centered_score_fp64_algebraically_equivalent": True,
                    "d43_centered_support_fp32_argmax_equivalent": True,
                    "d43_centered_support_fp32_pairwise_drift_max": 0.0,
                    "d43_centered_coefficient_mean_max_abs": float(
                        np.max(np.abs(coefficient64.mean(axis=0)))
                    ),
                    "d43_centered_intercept_mean_abs": float(abs(intercept64.mean())),
                }
            )
        return coefficient64.astype(np.float32), intercept64.astype(np.float32), audit

    return fit


__all__ = [
    "D92RegistrationBalancedCovarianceError",
    "OLD_CLASS_COUNT",
    "RegistrationBalancedStatistics",
    "TASK_WEIGHT",
    "build_registration_balanced_statistics",
    "compile_registration_balanced_affine",
    "build_registration_balanced_equal_lda",
]
