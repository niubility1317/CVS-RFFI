"""Class-centroid-orthogonal rank-one nuisance removal for D74."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class D74ProjectionError(RuntimeError):
    """Raised when D74 support or projection evidence drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _validate_support(
    rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or x.shape[1] < 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(
            int(np.sum(y == index)) != int(k_shot)
            for index in range(int(class_count))
        )
    ):
        raise D74ProjectionError("D74 requires finite exact symmetric support")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _machine_rank(singular_values: np.ndarray, rows: int, columns: int) -> int:
    values = np.asarray(singular_values, dtype=np.float64)
    if len(values) == 0 or float(values[0]) == 0.0:
        return 0
    tolerance = np.finfo(np.float64).eps * max(int(rows), int(columns)) * float(
        values[0]
    )
    return int(np.sum(values > tolerance))


def _pairwise_squared(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    difference = x[:, None, :] - x[None, :, :]
    return np.einsum("ijd,ijd->ij", difference, difference, optimize=True)


def fit_orthogonal_nuisance_direction(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return nuisance direction, projected support, and a closed audit."""

    x, y = _validate_support(rows, labels, class_count, k_shot)
    dimension = int(x.shape[1])
    if int(k_shot) == 1:
        direction = np.zeros(dimension, dtype=np.float64)
        audit = {
            "schema": "cvs.phase2.d74.orthogonal_nuisance_projection_audit.v1",
            "status": "k1_exact_d62_fallback",
            "class_count": int(class_count),
            "k_shot": 1,
            "dimension": dimension,
            "projection_active": False,
            "projection_removed_rank": 0,
            "projection_rank": dimension,
            "centroid_span_rank": 0,
            "orthogonal_residual_rank": 0,
            "direction_l2": 0.0,
            "centroid_direction_max_abs": 0.0,
            "centroid_pairwise_squared_distance_drift_max": 0.0,
            "removed_residual_energy_fraction": 0.0,
            "within_residual_energy_before": 0.0,
            "within_residual_energy_after": 0.0,
            "direction_sha256": hashlib.sha256(
                np.ascontiguousarray(direction, dtype=np.float32).tobytes()
            ).hexdigest(),
            "ground_component_input_count": 0,
            "query_rows_used": 0,
        }
        return (
            _readonly(direction, np.float32),
            _readonly(x, np.float32),
            audit,
        )

    means = np.stack(
        [np.mean(x[y == index], axis=0) for index in range(int(class_count))]
    )
    centered_means = means - np.mean(means, axis=0, keepdims=True)
    _, center_singular, center_vh = np.linalg.svd(
        centered_means, full_matrices=False
    )
    center_rank = _machine_rank(
        center_singular, centered_means.shape[0], centered_means.shape[1]
    )
    center_basis = center_vh[:center_rank].T
    residual = x - means[y]
    if center_rank:
        orthogonal_residual = residual - (
            (residual @ center_basis) @ center_basis.T
        )
    else:
        orthogonal_residual = residual.copy()
    _, residual_singular, residual_vh = np.linalg.svd(
        orthogonal_residual, full_matrices=False
    )
    residual_rank = _machine_rank(
        residual_singular, orthogonal_residual.shape[0], dimension
    )
    if residual_rank == 0 or float(residual_singular[0]) <= 0.0:
        raise D74ProjectionError("D74 orthogonal residual is degenerate")
    direction = residual_vh[0].copy()
    if center_rank:
        direction -= center_basis @ (center_basis.T @ direction)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= np.finfo(np.float64).eps:
        raise D74ProjectionError("D74 nuisance direction vanished after projection")
    direction /= direction_norm
    pivot = int(np.argmax(np.abs(direction)))
    if float(direction[pivot]) < 0.0:
        direction *= -1.0
    projected = x - np.outer(x @ direction, direction)
    projected_means = np.stack(
        [np.mean(projected[y == index], axis=0) for index in range(class_count)]
    )
    residual_after = projected - projected_means[y]
    energy_before = float(np.sum(residual * residual))
    energy_after = float(np.sum(residual_after * residual_after))
    orthogonal_energy = float(np.sum(orthogonal_residual * orthogonal_residual))
    removed_energy = float(residual_singular[0] ** 2)
    centroid_pair_drift = float(
        np.max(
            np.abs(
                _pairwise_squared(projected_means)
                - _pairwise_squared(means)
            )
        )
    )
    centroid_direction = centered_means @ direction
    projector = np.eye(dimension, dtype=np.float64) - np.outer(
        direction, direction
    )
    audit = {
        "schema": "cvs.phase2.d74.orthogonal_nuisance_projection_audit.v1",
        "status": "rank1_orthogonal_nuisance_removal_active",
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "dimension": dimension,
        "projection_active": True,
        "projection_removed_rank": 1,
        "projection_rank": dimension - 1,
        "centroid_span_rank": center_rank,
        "orthogonal_residual_rank": residual_rank,
        "direction_l2": float(np.linalg.norm(direction)),
        "direction_pivot_index": pivot,
        "direction_pivot_value": float(direction[pivot]),
        "centroid_direction_max_abs": float(np.max(np.abs(centroid_direction))),
        "centroid_pairwise_squared_distance_drift_max": centroid_pair_drift,
        "removed_residual_energy_fraction": (
            removed_energy / orthogonal_energy if orthogonal_energy > 0.0 else 0.0
        ),
        "within_residual_energy_before": energy_before,
        "within_residual_energy_after": energy_after,
        "within_residual_energy_removed_fraction": (
            (energy_before - energy_after) / energy_before
            if energy_before > 0.0
            else 0.0
        ),
        "projector_symmetry_max_abs_error": float(
            np.max(np.abs(projector - projector.T))
        ),
        "projector_idempotence_max_abs_error": float(
            np.max(np.abs(projector @ projector - projector))
        ),
        "projector_annihilation_l2": float(np.linalg.norm(projector @ direction)),
        "direction_sha256": hashlib.sha256(
            np.ascontiguousarray(direction, dtype=np.float32).tobytes()
        ).hexdigest(),
        "class_id_specific_formula": False,
        "class_permutation_equivariant": True,
        "scene_receiver_handle_specific_branch": False,
        "ground_component_input_count": 0,
        "query_rows_used": 0,
    }
    if (
        audit["centroid_direction_max_abs"] > 1e-9
        or audit["projector_idempotence_max_abs_error"] > 1e-10
        or audit["projector_annihilation_l2"] > 1e-10
        or energy_after > energy_before + 1e-10
    ):
        raise D74ProjectionError("D74 projection invariant drift")
    return (
        _readonly(direction, np.float32),
        _readonly(projected, np.float32),
        audit,
    )
