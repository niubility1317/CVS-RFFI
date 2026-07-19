"""D83 support-only ground-nuisance precision loading for equal-prior LDA."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


Z_DIM = 160


class D83GroundPrecisionError(RuntimeError):
    """Raised when the fixed ground precision-loading closure drifts."""


def build_ground_precision_loaded_equal_lda(
    d42: Any,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    *,
    arm: str,
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Add mean-estimation-scale nuisance covariance before inversion."""

    if arm not in {"full", "block3_centered"}:
        raise D83GroundPrecisionError("D83 covariance arm drift")
    u = np.asarray(basis, dtype=np.float64)
    pi = np.asarray(spectral_weights, dtype=np.float64)
    dimension = int(d42.FEATURE_DIM)
    blocks = tuple(d42.BLOCK_SLICES)
    if (
        u.ndim != 2
        or u.shape[0] != Z_DIM
        or pi.shape != (u.shape[1],)
        or not np.isfinite(u).all()
        or not np.isfinite(pi).all()
        or np.any(pi <= 0.0)
        or not np.isclose(np.sum(pi), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise D83GroundPrecisionError("D83 ground basis drift")
    ground_shape = (u * pi[None, :]) @ u.T
    ground_shape = 0.5 * (ground_shape + ground_shape.T)
    rank = int(u.shape[1])
    baseline_fit = d42._fit_equal_prior_lda

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = np.asarray(transformed, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.int64)
        classes, shots = int(class_count), int(k_shot)
        if (
            rows.ndim != 2
            or rows.shape[1] != dimension
            or labels.shape != (len(rows),)
            or len(rows) != classes * shots
            or classes < 2
            or shots < 1
            or not np.isfinite(rows).all()
            or not np.array_equal(np.unique(labels), np.arange(classes))
            or any(int(np.sum(labels == index)) != shots for index in range(classes))
        ):
            raise D83GroundPrecisionError("D83 requires finite symmetric support")
        if shots <= 2:
            coefficient, intercept, base_audit = baseline_fit(
                rows, labels, classes, shots
            )
            audit = dict(base_audit)
            audit.update(
                {
                    "d83_covariance_arm": arm,
                    "d83_ground_retained_rank": rank,
                    "d83_loading_formula": "target_z_mean_variance_times_rank_over_k_times_normalized_ground_spectrum",
                    "d83_loading_scale": 0.0,
                    "d83_loading_trace": 0.0,
                    "d83_loading_mean_retained_direction": 0.0,
                    "d83_target_z_mean_variance": 0.0,
                    "d83_loading_to_target_mean_variance_ratio": 0.0,
                    "d83_k1_k2_exact_no_loading": True,
                    "d83_hyperparameter_count": 0,
                    "d83_loading_scan_count": 0,
                    "d83_class_id_specific_formula": False,
                    "d83_old_new_role_specific_branch": False,
                    "d83_query_rows_used": 0,
                }
            )
            return coefficient, intercept, audit
        priors = np.full(classes, 1.0 / classes, dtype=np.float64)
        estimator = d42.LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=priors, store_covariance=True
        )
        estimator.fit(rows, labels)
        if not np.array_equal(estimator.classes_, np.arange(classes)):
            raise D83GroundPrecisionError("D83 sklearn class order drift")
        means = np.asarray(estimator.means_, dtype=np.float64)
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        if arm == "block3_centered":
            structured = np.zeros_like(covariance)
            for block in blocks:
                structured[block, block] = covariance[block, block]
            covariance = structured
        covariance = 0.5 * (covariance + covariance.T)
        target_z_mean_variance = float(np.trace(covariance[:Z_DIM, :Z_DIM]) / Z_DIM)
        identity = shots <= 2
        loading_scale = 0.0 if identity else target_z_mean_variance * rank / shots
        loading = loading_scale * ground_shape
        posterior = covariance.copy()
        posterior[:Z_DIM, :Z_DIM] += loading
        posterior = 0.5 * (posterior + posterior.T)
        eigenvalues = np.linalg.eigvalsh(posterior)
        if (
            not np.isfinite(eigenvalues).all()
            or float(np.min(eigenvalues)) <= 0.0
            or target_z_mean_variance <= 0.0
        ):
            raise D83GroundPrecisionError("D83 posterior covariance is not SPD")
        coefficient64 = np.linalg.solve(posterior, means.T).T
        intercept64 = -0.5 * np.diag(means @ coefficient64.T) + np.log(priors)
        equation_residual = float(
            np.max(np.abs(posterior @ coefficient64.T - means.T))
        )
        raw_prediction = np.argmax(
            rows @ coefficient64.T + intercept64[None, :], axis=1
        )
        if arm == "block3_centered":
            coefficient64 -= coefficient64.mean(axis=0, keepdims=True)
            intercept64 -= intercept64.mean()
            if not np.array_equal(
                raw_prediction,
                np.argmax(rows @ coefficient64.T + intercept64[None, :], axis=1),
            ):
                raise D83GroundPrecisionError("D83 score centering changed argmax")
        audit = {
            "solver": "lsqr",
            "shrinkage": "auto_plus_ground_nuisance_mean_uncertainty_loading",
            "prior_policy": "equal_1_over_registered_class_count",
            "covariance_policy": "sklearn_lsqr_auto_plus_rank14_ground_loading",
            "unit_covariance_fallback": False,
            "support_rows": int(len(rows)),
            "class_count": classes,
            "k_shot": shots,
            "d83_covariance_arm": arm,
            "d83_ground_retained_rank": rank,
            "d83_loading_formula": "target_z_mean_variance_times_rank_over_k_times_normalized_ground_spectrum",
            "d83_loading_scale": loading_scale,
            "d83_loading_trace": float(np.trace(loading)),
            "d83_loading_mean_retained_direction": (
                0.0 if identity else target_z_mean_variance / shots
            ),
            "d83_target_z_mean_variance": target_z_mean_variance,
            "d83_loading_to_target_mean_variance_ratio": (
                0.0 if identity else 1.0 / shots
            ),
            "d83_k1_k2_exact_no_loading": identity,
            "d83_hyperparameter_count": 0,
            "d83_loading_scan_count": 0,
            "d83_class_id_specific_formula": False,
            "d83_old_new_role_specific_branch": False,
            "d83_query_rows_used": 0,
            "d83_posterior_eigenvalue_min": float(np.min(eigenvalues)),
            "d83_posterior_eigenvalue_max": float(np.max(eigenvalues)),
            "d83_covariance_equation_residual_max": equation_residual,
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
    "D83GroundPrecisionError",
    "Z_DIM",
    "build_ground_precision_loaded_equal_lda",
]
