"""D90 directionwise Cauchy support centers on the D89 compressed spectrum."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


_D89_PATH = Path(__file__).with_name("stage2_d89_v2_radius_cauchy_center.py")
_SPEC = importlib.util.spec_from_file_location("d90_d89_spectrum", _D89_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("D90 could not load the locked D89 spectrum")
d89 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = d89
_SPEC.loader.exec_module(d89)

Z_DIM = d89.Z_DIM
ENERGY_EPSILON = 1.0e-24
TRANSLATION_EXTRA_MACS_PER_ROW_RANK = 7
radius_reliability_ground_spectrum = d89.radius_reliability_ground_spectrum


class D90DirectionwiseCenterError(RuntimeError):
    """Raised when the D90 support-center invariants drift."""


def translate_to_directionwise_cauchy_centers(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace the D81 radial center only inside each retained ground direction."""

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
        raise D90DirectionwiseCenterError("D90 requires finite symmetric support")
    transformed = x.copy()
    identity = shots <= 2
    shifts: list[np.ndarray] = []
    radial_weights_by_class: list[list[float]] = []
    energies_by_class: list[list[float]] = []
    axis_weights_by_class: list[list[list[float]]] = []
    axis_scales_by_class: list[list[float]] = []
    axis_centers_by_class: list[list[float]] = []
    effective_samples: list[float] = []
    axis_effective_samples: list[list[float]] = []
    radial_subspace_replacement: list[float] = []
    for class_index in range(classes):
        indices = np.flatnonzero(y == class_index)
        z = x[indices, :Z_DIM]
        mean = np.mean(z, axis=0)
        residual = z - mean
        projected = residual @ u
        if identity:
            energy = np.zeros(shots, dtype=np.float64)
            radial_weight = np.full(shots, 1.0 / shots, dtype=np.float64)
            axis_scale = np.zeros(u.shape[1], dtype=np.float64)
            axis_weight = np.ones((shots, u.shape[1]), dtype=np.float64)
            axis_center = np.zeros(u.shape[1], dtype=np.float64)
            radial_shift = np.zeros(Z_DIM, dtype=np.float64)
            shift = radial_shift
        else:
            energy = np.sum(np.square(projected) * pi[None, :], axis=1)
            energy_scale = float(np.mean(energy))
            if not np.isfinite(energy_scale) or energy_scale <= ENERGY_EPSILON:
                radial_raw = np.ones(shots, dtype=np.float64)
            else:
                radial_raw = 1.0 / (1.0 + energy / energy_scale)
            radial_weight = radial_raw / np.sum(radial_raw)
            radial_center = np.sum(radial_weight[:, None] * z, axis=0)
            radial_shift = radial_center - mean

            axis_scale = np.mean(np.square(projected), axis=0)
            axis_weight = np.ones_like(projected)
            active = axis_scale > ENERGY_EPSILON
            axis_weight[:, active] = 1.0 / (
                1.0
                + np.square(projected[:, active]) / axis_scale[None, active]
            )
            axis_center = np.sum(axis_weight * projected, axis=0) / np.sum(
                axis_weight, axis=0
            )
            radial_subspace = radial_shift @ u
            replacement = axis_center - radial_subspace
            shift = radial_shift + replacement @ u.T
            transformed[indices, :Z_DIM] += shift[None, :]
        shifts.append(shift)
        energies_by_class.append(energy.tolist())
        radial_weights_by_class.append(radial_weight.tolist())
        axis_weights_by_class.append(axis_weight.tolist())
        axis_scales_by_class.append(axis_scale.tolist())
        axis_centers_by_class.append(axis_center.tolist())
        effective_samples.append(float(1.0 / np.sum(np.square(radial_weight))))
        normalized_axis = axis_weight / np.sum(axis_weight, axis=0, keepdims=True)
        axis_effective_samples.append(
            (1.0 / np.sum(np.square(normalized_axis), axis=0)).tolist()
        )
        radial_subspace_replacement.append(
            float(np.linalg.norm(axis_center - radial_shift @ u))
        )

    before_means = np.stack(
        [np.mean(x[y == index], axis=0) for index in range(classes)]
    )
    after_means = np.stack(
        [np.mean(transformed[y == index], axis=0) for index in range(classes)]
    )
    before_residual = x - before_means[y]
    after_residual = transformed - after_means[y]
    residual_error = float(np.max(np.abs(before_residual - after_residual)))
    fft_rf_error = float(np.max(np.abs(transformed[:, Z_DIM:] - x[:, Z_DIM:])))
    if (
        not np.isfinite(transformed).all()
        or residual_error > 2.0e-12
        or fft_rf_error != 0.0
    ):
        raise D90DirectionwiseCenterError("D90 translation invariant drift")
    shift_array = np.stack(shifts)
    radial_array = np.asarray(radial_weights_by_class, dtype=np.float64)
    axis_array = np.asarray(axis_weights_by_class, dtype=np.float64)
    axis_ess = np.asarray(axis_effective_samples, dtype=np.float64)
    audit = {
        "schema": "cvs.phase2.d90.directionwise_cauchy_translation.v1",
        "support_rows": int(len(x)),
        "class_count": classes,
        "k_shot": shots,
        "retained_rank": int(u.shape[1]),
        "center_formula": "d81_radial_orthogonal_plus_directionwise_cauchy_subspace",
        "energy_scale_policy": "per_class_mean_ground_spectral_energy",
        "axis_scale_policy": "per_class_per_direction_mean_squared_projection",
        "translation_scope": "z160_class_common_only",
        "k1_k2_exact_identity": identity,
        "class_id_specific_formula": False,
        "old_new_role_specific_branch": False,
        "scene_receiver_handle_specific_branch": False,
        "uses_outer_held_or_query": False,
        "query_rows_used": 0,
        "hyperparameter_count": 0,
        "weight_scan_count": 0,
        "within_class_residual_max_abs_error": residual_error,
        "fft96_rf32_max_abs_error": fft_rf_error,
        "center_shift_l2_by_class": np.linalg.norm(shift_array, axis=1).tolist(),
        "center_shift_l2_max": float(np.max(np.linalg.norm(shift_array, axis=1))),
        "normalized_weight_min": float(np.min(radial_array)),
        "normalized_weight_max": float(np.max(radial_array)),
        "effective_sample_size_by_class": effective_samples,
        "nuisance_energy_by_class": energies_by_class,
        "raw_cauchy_weight_by_class": radial_weights_by_class,
        "normalized_cauchy_weight_by_class": radial_weights_by_class,
        "axis_cauchy_weight_by_class": axis_weights_by_class,
        "axis_scale_by_class": axis_scales_by_class,
        "axis_center_by_class": axis_centers_by_class,
        "axis_weight_min": float(np.min(axis_array)),
        "axis_weight_max": float(np.max(axis_array)),
        "axis_effective_sample_size_min": float(np.min(axis_ess)),
        "axis_effective_sample_size_max": float(np.max(axis_ess)),
        "radial_subspace_replacement_l2_by_class": radial_subspace_replacement,
        "radial_subspace_replacement_l2_max": float(
            np.max(radial_subspace_replacement)
        ),
        "d81_orthogonal_center_preserved": True,
        "directionwise_subspace_center_replaced": True,
    }
    return transformed, audit


translate_to_robust_centers = translate_to_directionwise_cauchy_centers


def build_robust_center_component_fit(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    basis: np.ndarray,
    spectral_weights: np.ndarray,
    basis_audit: dict[str, Any],
    component_arm: str,
    collector: list[dict[str, Any]],
) -> Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Wrap each full/block fit with the D90 directionwise center."""

    def fit(
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        transformed, transform_audit = translate_to_directionwise_cauchy_centers(
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
        collector.append({
            "component_arm": component_arm,
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "center_shift_l2_max": transform_audit["center_shift_l2_max"],
            "normalized_weight_min": transform_audit["normalized_weight_min"],
            "effective_sample_size_min": min(
                transform_audit["effective_sample_size_by_class"]
            ),
            "axis_effective_sample_size_min": transform_audit[
                "axis_effective_sample_size_min"
            ],
            "radial_subspace_replacement_l2_max": transform_audit[
                "radial_subspace_replacement_l2_max"
            ],
        })
        audit = dict(base_audit)
        audit.update({
            "d81_component_arm": component_arm,
            "d81_ground_basis_sha256": basis_audit["basis_sha256"],
            "d81_ground_spectral_weight_sha256": basis_audit[
                "spectral_weight_sha256"
            ],
            "d81_ground_effective_rank": basis_audit[
                "participation_ratio_effective_rank"
            ],
            "d81_ground_retained_rank": basis_audit["retained_rank"],
            "d81_ground_rank_policy": basis_audit["rank_policy"],
            "d81_transform_audit": transform_audit,
        })
        return coefficient, intercept, audit

    return fit


__all__ = [
    "D90DirectionwiseCenterError",
    "Z_DIM",
    "TRANSLATION_EXTRA_MACS_PER_ROW_RANK",
    "build_robust_center_component_fit",
    "radius_reliability_ground_spectrum",
    "translate_to_directionwise_cauchy_centers",
    "translate_to_robust_centers",
]
