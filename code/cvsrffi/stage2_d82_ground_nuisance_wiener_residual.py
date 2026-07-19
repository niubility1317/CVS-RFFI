"""Ground-spectrum robust centers plus parameter-free nuisance shrinkage.

The immutable ground bundle contributes only a class-agnostic nuisance spectrum.
Every target class uses the same support-only Cauchy center and closed-form Wiener
residual transform; query rows never enter the transform or gain an extra path.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


Z_DIM = 160
ENERGY_EPSILON = 1.0e-24


class D82GroundWienerError(RuntimeError):
    """Raised when the ground spectrum or support-Wiener closure drifts."""


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def ground_nuisance_basis(
    covariance: np.ndarray,
    quantization_noise_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive one fixed low-rank nuisance spectrum without a rank scan."""

    matrix = np.asarray(covariance, dtype=np.float64)
    floor = float(quantization_noise_floor)
    if (
        matrix.shape != (Z_DIM, Z_DIM)
        or not np.isfinite(matrix).all()
        or not np.isfinite(floor)
        or floor < 0.0
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-14)
    ):
        raise D82GroundWienerError("D82 covariance input drift")
    signal = 0.5 * (matrix + matrix.T) - floor * np.eye(Z_DIM)
    eigenvalues, eigenvectors = np.linalg.eigh(signal)
    tolerance = max(float(np.max(np.abs(eigenvalues))) * Z_DIM * np.finfo(float).eps, 0.0)
    positive = eigenvalues > tolerance
    positive_values = eigenvalues[positive]
    if len(positive_values) == 0:
        raise D82GroundWienerError("D82 ground spectrum has no positive direction")
    effective_rank = float(
        np.square(np.sum(positive_values)) / np.sum(np.square(positive_values))
    )
    retained_rank = int(np.ceil(effective_rank))
    retained_rank = min(retained_rank, len(positive_values))
    order = np.argsort(eigenvalues, kind="stable")[-retained_rank:][::-1]
    values = np.asarray(eigenvalues[order], dtype=np.float64)
    basis = np.asarray(eigenvectors[:, order], dtype=np.float64)
    weights = values / np.sum(values)
    if (
        basis.shape != (Z_DIM, retained_rank)
        or weights.shape != (retained_rank,)
        or not np.isfinite(basis).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise D82GroundWienerError("D82 retained spectrum drift")
    isotropic_signal_scale = 1.0 / retained_rank
    retention = isotropic_signal_scale / (weights + isotropic_signal_scale)
    nuisance_shrinkage = 1.0 - retention
    basis.setflags(write=False)
    weights.setflags(write=False)
    audit = {
        "schema": "cvs.phase2.d82.ground_nuisance_basis.v1",
        "z_dimension": Z_DIM,
        "positive_numerical_rank": int(len(positive_values)),
        "participation_ratio_effective_rank": effective_rank,
        "retained_rank": retained_rank,
        "rank_policy": "ceil_participation_ratio_effective_rank",
        "rank_scan_count": 0,
        "quantization_noise_floor_removed_before_spectrum": floor,
        "retained_signal_fraction": float(
            np.sum(values) / np.sum(positive_values)
        ),
        "retained_eigenvalue_min": float(np.min(values)),
        "retained_eigenvalue_max": float(np.max(values)),
        "basis_sha256": _sha256_array(basis),
        "spectral_weight_sha256": _sha256_array(weights),
        "wiener_signal_scale": isotropic_signal_scale,
        "wiener_retention_by_direction": retention.tolist(),
        "wiener_nuisance_shrinkage_by_direction": nuisance_shrinkage.tolist(),
        "wiener_retention_min": float(np.min(retention)),
        "wiener_retention_max": float(np.max(retention)),
        "wiener_formula": "mean_retained_ground_eigenvalue_over_direction_plus_mean",
        "wiener_hyperparameter_count": 0,
        "ground_class_score_access": False,
        "ground_anchor_access": False,
        "ground_radius_access": False,
        "ground_count_access": False,
    }
    return basis, weights, audit


def transform_support_with_ground_wiener(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust-center and shrink support residuals along ground nuisance axes."""

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    u = np.asarray(basis, dtype=np.float64)
    pi = np.asarray(spectral_weights, dtype=np.float64)
    classes, shots = int(class_count), int(k_shot)
    if (
        x.ndim != 2
        or x.shape[1] < Z_DIM
        or y.shape != (len(x),)
        or len(x) != classes * shots
        or u.ndim != 2
        or u.shape[0] != Z_DIM
        or pi.shape != (u.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(u).all()
        or not np.isfinite(pi).all()
        or classes <= 1
        or shots <= 0
        or any(int(np.sum(y == index)) != shots for index in range(classes))
    ):
        raise D82GroundWienerError("D82 requires finite symmetric support")
    if (
        np.any(pi <= 0.0)
        or not np.isclose(np.sum(pi), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise D82GroundWienerError("D82 spectral weight closure drift")
    signal_scale = 1.0 / u.shape[1]
    retention = signal_scale / (pi + signal_scale)
    nuisance_shrinkage = 1.0 - retention
    transformed = x.copy()
    shifts: list[np.ndarray] = []
    raw_weights_by_class: list[list[float]] = []
    normalized_weights_by_class: list[list[float]] = []
    energies_by_class: list[list[float]] = []
    effective_samples: list[float] = []
    nuisance_energy_before_by_class: list[float] = []
    nuisance_energy_after_by_class: list[float] = []
    identity = shots <= 2
    for class_index in range(classes):
        indices = np.flatnonzero(y == class_index)
        z = x[indices, :Z_DIM]
        mean = np.mean(z, axis=0)
        residual = z - mean
        if identity:
            energy = np.zeros(shots, dtype=np.float64)
            raw_weight = np.ones(shots, dtype=np.float64)
            weight = np.full(shots, 1.0 / shots, dtype=np.float64)
            shift = np.zeros(Z_DIM, dtype=np.float64)
            shrunk_residual = residual
        else:
            projected = residual @ u
            energy = np.sum(np.square(projected) * pi[None, :], axis=1)
            scale = float(np.mean(energy))
            if not np.isfinite(scale) or scale <= ENERGY_EPSILON:
                raw_weight = np.ones(shots, dtype=np.float64)
            else:
                raw_weight = 1.0 / (1.0 + energy / scale)
            weight = raw_weight / np.sum(raw_weight)
            robust_mean = np.sum(weight[:, None] * z, axis=0)
            shift = robust_mean - mean
            shrunk_residual = residual - (
                projected * nuisance_shrinkage[None, :]
            ) @ u.T
            transformed[indices, :Z_DIM] = robust_mean + shrunk_residual
        shifts.append(shift)
        energies_by_class.append(energy.tolist())
        raw_weights_by_class.append(raw_weight.tolist())
        normalized_weights_by_class.append(weight.tolist())
        effective_samples.append(float(1.0 / np.sum(np.square(weight))))
        before_projected = residual @ u
        after_projected = shrunk_residual @ u
        nuisance_energy_before_by_class.append(float(np.sum(np.square(before_projected))))
        nuisance_energy_after_by_class.append(float(np.sum(np.square(after_projected))))
    before_means = np.stack(
        [np.mean(x[y == index], axis=0) for index in range(classes)]
    )
    after_means = np.stack(
        [np.mean(transformed[y == index], axis=0) for index in range(classes)]
    )
    before_residual = x - before_means[y]
    after_residual = transformed - after_means[y]
    expected_residual = before_residual.copy()
    if not identity:
        expected_residual[:, :Z_DIM] -= (
            (before_residual[:, :Z_DIM] @ u) * nuisance_shrinkage[None, :]
        ) @ u.T
    residual_transform_error = float(np.max(np.abs(expected_residual - after_residual)))
    residual_change = float(np.max(np.abs(before_residual - after_residual)))
    center_error = float(
        np.max(np.abs(after_means[:, :Z_DIM] - (before_means[:, :Z_DIM] + np.stack(shifts))))
    )
    fft_rf_error = float(np.max(np.abs(transformed[:, Z_DIM:] - x[:, Z_DIM:])))
    if (
        not np.isfinite(transformed).all()
        or residual_transform_error > 2.0e-12
        or center_error > 2.0e-12
        or fft_rf_error != 0.0
    ):
        raise D82GroundWienerError("D82 support-Wiener invariant drift")
    shift_array = np.stack(shifts)
    all_weights = np.asarray(normalized_weights_by_class, dtype=np.float64)
    audit = {
        "schema": "cvs.phase2.d82.support_center_wiener_transform.v1",
        "support_rows": int(len(x)),
        "class_count": classes,
        "k_shot": shots,
        "retained_rank": int(u.shape[1]),
        "center_formula": "one_step_classwise_cauchy_ground_nuisance_energy",
        "energy_scale_policy": "per_class_mean_ground_spectral_energy",
        "transform_scope": "z160_class_common_center_plus_within_class_ground_wiener",
        "k1_k2_exact_identity": identity,
        "wiener_signal_scale": signal_scale,
        "wiener_retention_by_direction": retention.tolist(),
        "wiener_nuisance_shrinkage_by_direction": nuisance_shrinkage.tolist(),
        "wiener_retention_min": float(np.min(retention)),
        "wiener_retention_max": float(np.max(retention)),
        "wiener_formula": "mean_retained_ground_eigenvalue_over_direction_plus_mean",
        "class_id_specific_formula": False,
        "old_new_role_specific_branch": False,
        "scene_receiver_handle_specific_branch": False,
        "uses_outer_held_or_query": False,
        "query_rows_used": 0,
        "hyperparameter_count": 0,
        "weight_scan_count": 0,
        "within_class_residual_changed": not identity,
        "within_class_residual_max_abs_change": residual_change,
        "wiener_residual_formula_max_abs_error": residual_transform_error,
        "robust_center_formula_max_abs_error": center_error,
        "fft96_rf32_max_abs_error": fft_rf_error,
        "center_shift_l2_by_class": np.linalg.norm(shift_array, axis=1).tolist(),
        "center_shift_l2_max": float(np.max(np.linalg.norm(shift_array, axis=1))),
        "normalized_weight_min": float(np.min(all_weights)),
        "normalized_weight_max": float(np.max(all_weights)),
        "effective_sample_size_by_class": effective_samples,
        "nuisance_energy_before_by_class": nuisance_energy_before_by_class,
        "nuisance_energy_after_by_class": nuisance_energy_after_by_class,
        "nuisance_energy_retention_ratio_by_class": [
            after / before if before > ENERGY_EPSILON else 1.0
            for before, after in zip(
                nuisance_energy_before_by_class, nuisance_energy_after_by_class
            )
        ],
        "nuisance_energy_by_class": energies_by_class,
        "raw_cauchy_weight_by_class": raw_weights_by_class,
        "normalized_cauchy_weight_by_class": normalized_weights_by_class,
    }
    return transformed, audit


def build_wiener_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    basis_audit: dict[str, Any],
    component_arm: str,
    collector: list[dict[str, Any]],
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Wrap every full/block OOF fit with its own support-only transform."""

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        transformed, transform_audit = transform_support_with_ground_wiener(
            rows,
            labels,
            class_count,
            k_shot,
            basis,
            spectral_weights,
        )
        coefficient, intercept, base_audit = component_fit(
            transformed, labels, class_count, k_shot
        )
        record = {
            "component_arm": component_arm,
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "center_shift_l2_max": transform_audit["center_shift_l2_max"],
            "normalized_weight_min": transform_audit["normalized_weight_min"],
            "effective_sample_size_min": min(
                transform_audit["effective_sample_size_by_class"]
            ),
            "wiener_retention_min": transform_audit["wiener_retention_min"],
            "nuisance_energy_retention_max": max(
                transform_audit["nuisance_energy_retention_ratio_by_class"]
            ),
        }
        collector.append(record)
        audit = dict(base_audit)
        audit.update(
            {
                "d82_component_arm": component_arm,
                "d82_ground_basis_sha256": basis_audit["basis_sha256"],
                "d82_ground_spectral_weight_sha256": basis_audit[
                    "spectral_weight_sha256"
                ],
                "d82_ground_effective_rank": basis_audit[
                    "participation_ratio_effective_rank"
                ],
                "d82_ground_retained_rank": basis_audit["retained_rank"],
                "d82_ground_rank_policy": basis_audit["rank_policy"],
                "d82_transform_audit": transform_audit,
            }
        )
        return coefficient, intercept, audit

    return fit


__all__ = [
    "D82GroundWienerError",
    "Z_DIM",
    "build_wiener_component_fit",
    "ground_nuisance_basis",
    "transform_support_with_ground_wiener",
]



