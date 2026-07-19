"""Ground nuisance empirical-Bayes Mahalanobis covariance for D80."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import numpy as np


Z_DIM = 160
QUANTIZATION_NOISE_DIVISOR = 12.0


class D80GroundCovarianceError(RuntimeError):
    """Raised when D80 ground or target covariance evidence is malformed."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ground_classcentered_covariance(
    domain_class_prototypes: np.ndarray,
    domain_class_scales: np.ndarray,
    domain_class_mask: np.ndarray,
    *,
    z_dim: int = Z_DIM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a shared cross-domain centroid-drift covariance, never class scores."""

    prototypes = np.asarray(domain_class_prototypes, dtype=np.float64)
    scales = np.asarray(domain_class_scales, dtype=np.float64)
    mask = np.asarray(domain_class_mask, dtype=bool)
    if (
        prototypes.ndim != 3
        or prototypes.shape[:2] != mask.shape
        or prototypes.shape[2] != int(z_dim)
        or scales.shape != mask.shape
        or prototypes.shape[0] < 2
        or prototypes.shape[1] < 2
        or not np.isfinite(prototypes).all()
        or not np.isfinite(scales).all()
    ):
        raise D80GroundCovarianceError("D80 requires a finite ground grid")
    registry_domains, classes = int(mask.shape[0]), int(mask.shape[1])
    full_domains = np.all(mask, axis=1)
    if np.any(np.any(mask, axis=1) != full_domains) or int(np.sum(full_domains)) < 2:
        raise D80GroundCovarianceError("D80 ground mask must contain whole class grids")
    active = prototypes[full_domains]
    active_scales = scales[full_domains]
    domains = int(active.shape[0])
    class_means = np.mean(active, axis=0, keepdims=True)
    residual = active - class_means
    if not np.allclose(
        np.mean(residual, axis=0), 0.0, rtol=0.0, atol=1e-12
    ):
        raise D80GroundCovarianceError("D80 class-centered residual drift")
    ground_degrees = classes * (domains - 1)
    raw_covariance = np.einsum("dcz,dcw->zw", residual, residual) / float(
        ground_degrees
    )
    quantization_noise = float(
        np.mean(active_scales * active_scales) / QUANTIZATION_NOISE_DIVISOR
    )
    covariance = raw_covariance + quantization_noise * np.eye(int(z_dim))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    raw_singular = np.linalg.svd(
        residual.reshape(domains * classes, int(z_dim)), compute_uv=False
    )
    tolerance = float(
        max(domains * classes, int(z_dim))
        * np.finfo(np.float64).eps
        * float(raw_singular[0])
    )
    raw_rank = int(np.sum(raw_singular > tolerance))
    trace = float(np.trace(raw_covariance))
    squared_trace = float(np.sum(raw_covariance * raw_covariance))
    effective_rank = trace * trace / squared_trace
    if (
        not np.isfinite(covariance).all()
        or float(np.min(eigenvalues)) <= 0.0
        or quantization_noise <= 0.0
        or raw_rank < 1
    ):
        raise D80GroundCovarianceError("D80 ground covariance is not positive definite")
    audit = {
        "schema": "cvs.phase2.d80.ground_classcentered_covariance_audit.v1",
        "ground_registry_domain_count": registry_domains,
        "ground_domain_count": domains,
        "ground_class_count": classes,
        "ground_component_input_count": int(np.sum(mask)),
        "z_dimension": int(z_dim),
        "ground_covariance_degrees_of_freedom": ground_degrees,
        "ground_independent_domain_degrees_of_freedom": domains - 1,
        "ground_residual_numerical_rank": raw_rank,
        "ground_residual_effective_rank": effective_rank,
        "ground_raw_covariance_trace": trace,
        "quantization_noise_model": "mean_fp16_scale_squared_over_12",
        "quantization_noise_floor": quantization_noise,
        "covariance_eigenvalue_min": float(np.min(eigenvalues)),
        "covariance_eigenvalue_max": float(np.max(eigenvalues)),
        "covariance_condition_number": float(
            np.max(eigenvalues) / np.min(eigenvalues)
        ),
        "covariance_sha256": _sha256(covariance.astype(np.float64)),
        "ground_class_centers_discarded_after_residualization": True,
        "ground_class_score_access": False,
        "ground_class_registry_prediction_branch": False,
        "ground_component_update_access": False,
    }
    return _readonly(covariance, np.float64), audit


def _block_diagonal(matrix: np.ndarray, blocks: tuple[slice, ...]) -> np.ndarray:
    result = np.zeros_like(matrix)
    for block in blocks:
        result[block, block] = matrix[block, block]
    return result


def build_ground_prior_equal_lda(
    d42: Any,
    ground_covariance_z: np.ndarray,
    ground_audit: dict[str, Any],
    *,
    arm: str,
) -> Callable[[np.ndarray, np.ndarray, int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Return a D42-compatible full or block fit using one fixed EB prior."""

    if arm not in {"full", "block3_centered"}:
        raise D80GroundCovarianceError("D80 covariance arm drift")
    ground = np.asarray(ground_covariance_z, dtype=np.float64)
    dimension = int(d42.FEATURE_DIM)
    blocks = tuple(d42.BLOCK_SLICES)
    if (
        ground.shape != (Z_DIM, Z_DIM)
        or dimension < Z_DIM
        or not np.isfinite(ground).all()
        or float(np.min(np.linalg.eigvalsh(ground))) <= 0.0
    ):
        raise D80GroundCovarianceError("D80 ground covariance shape drift")
    ground_trace = float(np.trace(ground))
    domain_degrees = int(ground_audit["ground_independent_domain_degrees_of_freedom"])

    def fit(
        transformed: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = np.asarray(transformed, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.int64)
        classes = int(class_count)
        shots = int(k_shot)
        if (
            rows.ndim != 2
            or rows.shape[1] != dimension
            or labels.shape != (len(rows),)
            or classes < 2
            or shots < 1
            or len(rows) != classes * shots
            or not np.array_equal(np.unique(labels), np.arange(classes))
            or any(int(np.sum(labels == index)) != shots for index in range(classes))
            or not np.isfinite(rows).all()
        ):
            raise D80GroundCovarianceError("D80 requires finite symmetric target support")
        means = np.stack(
            [rows[labels == index].mean(axis=0) for index in range(classes)]
        )
        residuals = rows - means[labels]
        residual_energy = float(np.sum(residuals * residuals))
        residual_rank = int(np.linalg.matrix_rank(residuals))
        target_degrees = classes * (shots - 1)
        priors = np.full(classes, 1.0 / classes, dtype=np.float64)
        target_fallback = bool(
            shots == 1
            or residual_rank == 0
            or not math.isfinite(residual_energy)
            or residual_energy <= float(d42.ENERGY_EPSILON)
        )
        sklearn_equivalent: bool | None = None
        if target_fallback:
            target_covariance = np.eye(dimension, dtype=np.float64)
            fitted_means = means
        else:
            estimator = d42.LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
                priors=priors,
                store_covariance=True,
            )
            estimator.fit(rows, labels)
            if not np.array_equal(
                np.asarray(estimator.classes_, dtype=np.int64),
                np.arange(classes, dtype=np.int64),
            ):
                raise D80GroundCovarianceError("D80 sklearn class order drift")
            target_covariance = np.asarray(estimator.covariance_, dtype=np.float64)
            fitted_means = np.asarray(estimator.means_, dtype=np.float64)
        if arm == "block3_centered":
            target_covariance = _block_diagonal(target_covariance, blocks)
        target_covariance = 0.5 * (target_covariance + target_covariance.T)
        target_z_trace = float(np.trace(target_covariance[:Z_DIM, :Z_DIM]))
        if target_z_trace <= 0.0 or not math.isfinite(target_z_trace):
            raise D80GroundCovarianceError("D80 target z covariance trace drift")
        trace_match_scale = target_z_trace / ground_trace
        prior = np.zeros_like(target_covariance)
        prior[:Z_DIM, :Z_DIM] = ground * trace_match_scale
        for block in blocks[1:]:
            prior[block, block] = target_covariance[block, block]
        posterior = (
            target_degrees * target_covariance + domain_degrees * prior
        ) / float(target_degrees + domain_degrees)
        posterior = 0.5 * (posterior + posterior.T)
        eigenvalues = np.linalg.eigvalsh(posterior)
        if float(np.min(eigenvalues)) <= 0.0 or not np.isfinite(eigenvalues).all():
            raise D80GroundCovarianceError("D80 posterior covariance is not positive definite")
        coefficients64 = np.linalg.solve(posterior, fitted_means.T).T
        intercept64 = -0.5 * np.diag(fitted_means @ coefficients64.T) + np.log(priors)
        uncentered_prediction = np.argmax(
            rows @ coefficients64.T + intercept64[None, :], axis=1
        )
        if arm == "block3_centered":
            coefficients64 -= np.mean(coefficients64, axis=0, keepdims=True)
            intercept64 -= np.mean(intercept64)
            if not np.array_equal(
                uncentered_prediction,
                np.argmax(rows @ coefficients64.T + intercept64[None, :], axis=1),
            ):
                raise D80GroundCovarianceError("D80 affine centering changed argmax")
        coefficients = np.asarray(coefficients64, dtype=np.float32)
        intercept = np.asarray(intercept64, dtype=np.float32)
        equation_residual = float(
            np.max(np.abs(posterior @ coefficients64.T - fitted_means.T))
        )
        shrink_weight = domain_degrees / float(target_degrees + domain_degrees)
        audit = {
            "solver": "lsqr",
            "shrinkage": "auto_plus_fixed_ground_empirical_bayes",
            "prior_policy": "equal_1_over_registered_class_count",
            "covariance_policy": "sklearn_lsqr_auto_shrinkage_equal_prior",
            "unit_covariance_fallback": False,
            "within_class_residual_rank": residual_rank,
            "within_class_residual_energy": residual_energy,
            "support_rows": int(len(rows)),
            "class_count": classes,
            "k_shot": shots,
            "coefficient_source": "d80_ground_empirical_bayes_sigma_inverse_mu",
            "covariance_equation_residual_max": equation_residual,
            "sklearn_prediction_equivalent": sklearn_equivalent,
            "d80_covariance_arm": arm,
            "d80_target_covariance_fallback": target_fallback,
            "d80_target_degrees_of_freedom": target_degrees,
            "d80_ground_independent_domain_degrees_of_freedom": domain_degrees,
            "d80_ground_shrinkage_weight": shrink_weight,
            "d80_ground_shrinkage_rule": "nu_ground/(nu_ground+class_count*(k_shot-1))",
            "d80_ground_z_trace_match_scale": trace_match_scale,
            "d80_target_z_covariance_trace": target_z_trace,
            "d80_ground_covariance_trace_before_match": ground_trace,
            "d80_posterior_eigenvalue_min": float(np.min(eigenvalues)),
            "d80_posterior_eigenvalue_max": float(np.max(eigenvalues)),
            "d80_posterior_condition_number": float(
                np.max(eigenvalues) / np.min(eigenvalues)
            ),
            "d80_ground_covariance_sha256": ground_audit["covariance_sha256"],
            "d80_ground_class_centers_used_for_prediction": False,
            "d80_ground_class_score_access": False,
            "d80_class_id_specific_formula": False,
            "d80_old_new_role_specific_branch": False,
            "d80_query_rows_used": 0,
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
                        np.max(np.abs(coefficients64.mean(axis=0)))
                    ),
                    "d43_centered_intercept_mean_abs": float(abs(intercept64.mean())),
                    "d43_covariance_eigenvalue_min": float(np.min(eigenvalues)),
                    "d43_covariance_eigenvalue_max": float(np.max(eigenvalues)),
                    "d43_covariance_condition_number": float(
                        np.max(eigenvalues) / np.min(eigenvalues)
                    ),
                }
            )
        return coefficients, intercept, audit

    return fit


__all__ = [
    "D80GroundCovarianceError",
    "QUANTIZATION_NOISE_DIVISOR",
    "Z_DIM",
    "build_ground_prior_equal_lda",
    "ground_classcentered_covariance",
]
