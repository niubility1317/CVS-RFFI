"""Nested support-held margin safety gate for the D75 projection."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


class D75MarginSafetyError(RuntimeError):
    """Raised when D75 support-held evidence is malformed."""


DirectionFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]
LDAFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _validate(
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
        or any(np.sum(y == index) != int(k_shot) for index in range(class_count))
    ):
        raise D75MarginSafetyError("D75 requires finite exact symmetric support")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _within_class_ranks(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.zeros(int(class_count), dtype=np.int64)
    ranks = np.empty(len(labels), dtype=np.int64)
    for row_index, class_index in enumerate(labels):
        ranks[row_index] = counts[int(class_index)]
        counts[int(class_index)] += 1
    return ranks


def _margins(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(y):
        raise D75MarginSafetyError("D75 score/target shape drift")
    competing = values.copy()
    competing[np.arange(len(y)), y] = -np.inf
    return values[np.arange(len(y)), y] - np.max(competing, axis=1)


def fit_crossfitted_margin_safe_projection(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    direction_fit: DirectionFit,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Gate the full D74 proposal with nested support-held score margins."""

    x, y = _validate(rows, labels, class_count, k_shot)
    dimension = int(x.shape[1])
    full_direction, full_projected, full_audit = direction_fit(
        x, y, int(class_count), int(k_shot)
    )
    proposal = dict(full_audit)
    if int(k_shot) == 1 or not bool(proposal.get("projection_active", False)):
        audit = {
            **proposal,
            "schema": "cvs.phase2.d75.crossfitted_margin_safe_projection_audit.v1",
            "status": "k1_exact_d62_fallback",
            "crossfit_gate_pass": False,
            "crossfit_fold_count": 0,
            "crossfit_lda_fit_count": 0,
            "crossfit_held_row_count": 0,
            "crossfit_margin_delta_mean": 0.0,
            "crossfit_margin_delta_min_class_mean": 0.0,
            "crossfit_margin_delta_max_class_mean": 0.0,
            "crossfit_base_correct_count": 0,
            "crossfit_projected_correct_count": 0,
            "crossfit_correct_delta": 0,
            "crossfit_numeric_tolerance": 0.0,
            "crossfit_per_class_margin_delta_mean": [
                0.0 for _ in range(int(class_count))
            ],
            "crossfit_unique_direction_count": 0,
            "proposed_projection_removed_rank": int(
                proposal.get("projection_removed_rank", 0)
            ),
            "ground_component_input_count": 0,
            "query_rows_used": 0,
        }
        return full_direction, full_projected, audit

    ranks = _within_class_ranks(y, int(class_count))
    deltas: list[np.ndarray] = []
    base_correct = projected_correct = 0
    loo_direction_hashes: set[str] = set()
    absolute_margins: list[np.ndarray] = []
    for held_rank in range(int(k_shot)):
        held = ranks == held_rank
        train = ~held
        if int(np.sum(held)) != int(class_count):
            raise D75MarginSafetyError("D75 physical-rank holdout drift")
        train_x, train_y = x[train], y[train]
        held_x, held_y = x[held], y[held]
        loo_direction, _, loo_audit = direction_fit(
            train_x, train_y, int(class_count), int(k_shot) - 1
        )
        if not bool(loo_audit.get("projection_active", False)):
            raise D75MarginSafetyError("D75 LOO direction unexpectedly inactive")
        coefficient, intercept, _ = lda_fit(
            train_x, train_y, int(class_count), int(k_shot) - 1
        )
        w = np.asarray(coefficient, dtype=np.float64)
        b = np.asarray(intercept, dtype=np.float64)
        u = np.asarray(loo_direction, dtype=np.float64)
        if w.shape != (int(class_count), dimension) or b.shape != (
            int(class_count),
        ):
            raise D75MarginSafetyError("D75 LOO LDA shape drift")
        projected_w = w - np.outer(w @ u, u)
        base_scores = held_x @ w.T + b[None, :]
        projected_scores = held_x @ projected_w.T + b[None, :]
        base_margin = _margins(base_scores, held_y)
        projected_margin = _margins(projected_scores, held_y)
        deltas.append(projected_margin - base_margin)
        absolute_margins.extend((np.abs(base_margin), np.abs(projected_margin)))
        base_correct += int(np.sum(np.argmax(base_scores, axis=1) == held_y))
        projected_correct += int(
            np.sum(np.argmax(projected_scores, axis=1) == held_y)
        )
        loo_direction_hashes.add(str(loo_audit["direction_sha256"]))

    delta_by_rank = np.stack(deltas, axis=0)
    if delta_by_rank.shape != (int(k_shot), int(class_count)):
        raise D75MarginSafetyError("D75 crossfit margin matrix drift")
    per_class_delta = np.mean(delta_by_rank, axis=0)
    mean_delta = float(np.mean(delta_by_rank))
    scale = max(
        1.0,
        max(float(np.max(value)) for value in absolute_margins),
    )
    tolerance = float(64.0 * np.finfo(np.float64).eps * scale)
    gate_pass = bool(
        float(np.min(per_class_delta)) >= -tolerance
        and mean_delta >= -tolerance
        and projected_correct >= base_correct
    )
    common = {
        "schema": "cvs.phase2.d75.crossfitted_margin_safe_projection_audit.v1",
        "crossfit_gate_pass": gate_pass,
        "crossfit_fold_count": int(k_shot),
        "crossfit_lda_fit_count": int(k_shot),
        "crossfit_held_row_count": int(class_count) * int(k_shot),
        "crossfit_margin_delta_mean": mean_delta,
        "crossfit_margin_delta_min_class_mean": float(np.min(per_class_delta)),
        "crossfit_margin_delta_max_class_mean": float(np.max(per_class_delta)),
        "crossfit_base_correct_count": int(base_correct),
        "crossfit_projected_correct_count": int(projected_correct),
        "crossfit_correct_delta": int(projected_correct - base_correct),
        "crossfit_numeric_tolerance": tolerance,
        "crossfit_per_class_margin_delta_mean": [
            float(value) for value in per_class_delta
        ],
        "crossfit_unique_direction_count": len(loo_direction_hashes),
        "proposed_direction_sha256": str(proposal["direction_sha256"]),
        "proposed_projection_removed_rank": int(
            proposal["projection_removed_rank"]
        ),
        "proposed_removed_residual_energy_fraction": float(
            proposal["removed_residual_energy_fraction"]
        ),
        "proposed_within_residual_energy_removed_fraction": float(
            proposal["within_residual_energy_removed_fraction"]
        ),
        "class_id_specific_formula": False,
        "class_permutation_equivariant": True,
        "scene_receiver_handle_specific_branch": False,
        "ground_component_input_count": 0,
        "query_rows_used": 0,
    }
    if gate_pass:
        audit = {
            **proposal,
            **common,
            "status": "crossfitted_margin_safe_projection_active",
        }
        return full_direction, full_projected, audit

    zero = np.zeros(dimension, dtype=np.float32)
    audit = {
        **proposal,
        **common,
        "status": "crossfitted_margin_rejected_exact_d62_fallback",
        "projection_active": False,
        "projection_removed_rank": 0,
        "projection_rank": dimension,
        "direction_l2": 0.0,
        "direction_sha256": hashlib.sha256(zero.tobytes()).hexdigest(),
        "centroid_direction_max_abs": 0.0,
        "centroid_pairwise_squared_distance_drift_max": 0.0,
        "removed_residual_energy_fraction": 0.0,
        "within_residual_energy_after": float(
            proposal["within_residual_energy_before"]
        ),
        "within_residual_energy_removed_fraction": 0.0,
        "projector_symmetry_max_abs_error": 0.0,
        "projector_idempotence_max_abs_error": 0.0,
        "projector_annihilation_l2": 0.0,
    }
    return _readonly(zero, np.float32), _readonly(x, np.float32), audit


__all__ = [
    "D75MarginSafetyError",
    "fit_crossfitted_margin_safe_projection",
]
