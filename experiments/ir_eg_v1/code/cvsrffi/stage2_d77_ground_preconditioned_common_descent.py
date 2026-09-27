"""Ground-metric all-class support-held common descent for D77."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


FW_ITERATIONS = 20
Z_DIM = 160


class D77CommonDescentError(RuntimeError):
    """Raised when D77 support or numerical evidence is malformed."""


LDAFit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ground_reliability_preconditioner(
    d66_shared_scale: np.ndarray,
    *,
    z_dim: int = Z_DIM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert the validated D66 reliability scale into a determinant-neutral metric."""

    scale = np.asarray(d66_shared_scale, dtype=np.float64)
    if (
        scale.ndim != 1
        or int(z_dim) < 2
        or len(scale) < int(z_dim)
        or not np.isfinite(scale).all()
        or np.any(scale[:z_dim] <= 1.0)
        or np.any(scale[:z_dim] >= np.sqrt(2.0) + 1e-12)
        or not np.array_equal(scale[z_dim:], np.ones(len(scale) - int(z_dim)))
    ):
        raise D77CommonDescentError("D77 ground reliability scale drift")
    reliability = scale[:z_dim] * scale[:z_dim] - 1.0
    epsilon = float(np.finfo(np.float32).eps)
    log_reliability = np.log(reliability + epsilon)
    centered = 0.5 * (log_reliability - float(np.mean(log_reliability)))
    metric = np.ones(len(scale), dtype=np.float64)
    metric[:z_dim] = np.exp(centered)
    if (
        not np.isfinite(metric).all()
        or np.any(metric <= 0.0)
        or abs(float(np.exp(np.mean(np.log(metric[:z_dim])))) - 1.0) > 1e-12
    ):
        raise D77CommonDescentError("D77 preconditioner numerical drift")
    frozen = _readonly(metric, np.float64)
    audit = {
        "schema": "cvs.phase2.d77.ground_preconditioner_audit.v1",
        "z_dimension": int(z_dim),
        "feature_dimension": len(scale),
        "reliability_min": float(np.min(reliability)),
        "reliability_mean": float(np.mean(reliability)),
        "reliability_max": float(np.max(reliability)),
        "preconditioner_min": float(np.min(metric)),
        "preconditioner_mean": float(np.mean(metric)),
        "preconditioner_max": float(np.max(metric)),
        "preconditioner_z_geometric_mean": float(
            np.exp(np.mean(np.log(metric[:z_dim])))
        ),
        "preconditioner_condition_number": float(
            np.max(metric) / np.min(metric)
        ),
        "preconditioner_sha256": _sha256(metric.astype(np.float64)),
        "ground_class_score_access": False,
        "ground_class_registry_prediction_branch": False,
        "ground_component_update_access": False,
    }
    return frozen, audit


def _validate(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    base_coefficient: np.ndarray,
    preconditioner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(base_coefficient, dtype=np.float64)
    metric = np.asarray(preconditioner, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] < 2
        or y.shape != (len(x),)
        or w.shape != (int(class_count), x.shape[1])
        or metric.shape != (x.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(w).all()
        or not np.isfinite(metric).all()
        or np.any(metric <= 0.0)
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(np.sum(y == index) != int(k_shot) for index in range(class_count))
    ):
        raise D77CommonDescentError("D77 requires finite exact symmetric support")
    return (
        np.ascontiguousarray(x),
        np.ascontiguousarray(y),
        np.ascontiguousarray(w),
        np.ascontiguousarray(metric),
    )


def _within_class_ranks(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.zeros(int(class_count), dtype=np.int64)
    ranks = np.empty(len(labels), dtype=np.int64)
    for row_index, class_index in enumerate(labels):
        ranks[row_index] = counts[int(class_index)]
        counts[int(class_index)] += 1
    return ranks


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _cross_entropy(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    return log_normalizer - shifted[np.arange(len(targets)), targets]


def _frank_wolfe_minimum_norm(
    gradients: np.ndarray,
    metric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    classes = int(gradients.shape[0])
    alpha = np.full(classes, 1.0 / float(classes), dtype=np.float64)
    trace: list[dict[str, Any]] = []
    for iteration in range(1, FW_ITERATIONS + 1):
        combined = np.tensordot(alpha, gradients, axes=(0, 0))
        metric_combined = combined * metric[None, :]
        objective_before = float(np.sum(combined * metric_combined))
        derivatives = 2.0 * np.einsum(
            "cij,ij->c", gradients, metric_combined, optimize=True
        )
        scale = max(1.0, float(np.max(np.abs(derivatives))))
        tie_tolerance = float(64.0 * np.finfo(np.float64).eps * scale)
        minimum = float(np.min(derivatives))
        tied = np.flatnonzero(derivatives <= minimum + tie_tolerance)
        vertex = np.zeros(classes, dtype=np.float64)
        vertex[tied] = 1.0 / float(len(tied))
        vertex_gradient = np.tensordot(vertex, gradients, axes=(0, 0))
        direction = vertex_gradient - combined
        metric_direction = direction * metric[None, :]
        denominator = float(np.sum(direction * metric_direction))
        numerator = float(np.sum(combined * metric_direction))
        gamma = 0.0 if denominator <= 0.0 else float(
            np.clip(-numerator / denominator, 0.0, 1.0)
        )
        alpha = (1.0 - gamma) * alpha + gamma * vertex
        updated = np.tensordot(alpha, gradients, axes=(0, 0))
        objective_after = float(np.sum(updated * updated * metric[None, :]))
        trace.append(
            {
                "iteration": iteration,
                "objective_before": objective_before,
                "objective_after": objective_after,
                "line_search_gamma": gamma,
                "tied_vertex_count": int(len(tied)),
                "minimum_derivative": minimum,
            }
        )
    combined = np.tensordot(alpha, gradients, axes=(0, 0))
    return alpha, combined, trace


def fit_ground_preconditioned_common_descent(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    *,
    base_coefficient: np.ndarray,
    preconditioner: np.ndarray,
    lda_fit: LDAFit,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a D62-row residual derived from all-class support-held CE gradients."""

    x, y, base_w, metric = _validate(
        rows, labels, class_count, k_shot, base_coefficient, preconditioner
    )
    classes, dimension = int(class_count), int(x.shape[1])
    zero = np.zeros((classes, dimension), dtype=np.float32)
    if int(k_shot) == 1:
        return _readonly(zero, np.float32), {
            "schema": "cvs.phase2.d77.ground_preconditioned_common_descent_audit.v1",
            "status": "k1_exact_d62_fallback",
            "class_count": classes,
            "k_shot": 1,
            "dimension": dimension,
            "crossfit_fold_count": 0,
            "crossfit_lda_fit_count": 0,
            "crossfit_held_row_count": 0,
            "frank_wolfe_iterations": 0,
            "optimizer_objective_trace": [],
            "residual_active": False,
            "query_rows_used": 0,
            "ground_class_score_access": False,
        }

    ranks = _within_class_ranks(y, classes)
    gradient_sums = np.zeros((classes, classes, dimension), dtype=np.float64)
    held_score_rows: list[np.ndarray] = []
    held_feature_rows: list[np.ndarray] = []
    held_target_rows: list[np.ndarray] = []
    fold_coefficient_hashes: set[str] = set()
    for held_rank in range(int(k_shot)):
        held = ranks == held_rank
        train = ~held
        if int(np.sum(held)) != classes:
            raise D77CommonDescentError("D77 physical-rank holdout drift")
        train_x, train_y = x[train], y[train]
        held_x, held_y = x[held], y[held]
        coefficient, intercept, _ = lda_fit(
            train_x, train_y, classes, int(k_shot) - 1
        )
        w = np.asarray(coefficient, dtype=np.float64)
        b = np.asarray(intercept, dtype=np.float64)
        if (
            w.shape != (classes, dimension)
            or b.shape != (classes,)
            or not np.isfinite(w).all()
            or not np.isfinite(b).all()
        ):
            raise D77CommonDescentError("D77 crossfit LDA shape drift")
        scores = held_x @ w.T + b[None, :]
        probabilities = _softmax(scores)
        errors = probabilities.copy()
        errors[np.arange(classes), held_y] -= 1.0
        sample_gradients = errors[:, :, None] * held_x[:, None, :]
        for held_index, true_class in enumerate(held_y):
            gradient_sums[int(true_class)] += sample_gradients[held_index]
        held_score_rows.append(scores)
        held_feature_rows.append(held_x)
        held_target_rows.append(held_y)
        fold_coefficient_hashes.add(_sha256(w.astype(np.float64)))

    gradients = gradient_sums / float(k_shot)
    alpha, combined, objective_trace = _frank_wolfe_minimum_norm(
        gradients, metric
    )
    preconditioned = combined * metric[None, :]
    common_inner = np.einsum(
        "cij,ij->c", gradients, preconditioned, optimize=True
    )
    direction_norm_sq = float(np.sum(preconditioned * preconditioned))
    held_features = np.stack(held_feature_rows, axis=0)
    held_scores = np.stack(held_score_rows, axis=0)
    held_targets = np.stack(held_target_rows, axis=0)
    lipschitz = np.zeros(classes, dtype=np.float64)
    for class_index in range(classes):
        selected = held_features[held_targets == class_index]
        lipschitz[class_index] = 0.5 * float(
            np.max(np.sum(selected * selected, axis=1))
        )
    numeric_scale = max(
        1.0,
        float(np.max(np.abs(common_inner))),
        float(np.max(np.abs(gradients))),
    )
    tolerance = float(256.0 * np.finfo(np.float64).eps * numeric_scale)
    active = bool(
        direction_norm_sq > tolerance
        and float(np.min(common_inner)) > tolerance
        and np.all(lipschitz > 0.0)
    )
    raw_step = 0.0
    trust_cap = float(np.linalg.norm(base_w) / np.sqrt(classes * dimension))
    trust_scale = 0.0
    delta64 = np.zeros((classes, dimension), dtype=np.float64)
    if active:
        raw_step = float(
            np.min(common_inner / (lipschitz * direction_norm_sq))
        )
        delta64 = -raw_step * preconditioned
        delta_norm = float(np.linalg.norm(delta64))
        trust_scale = 1.0 if delta_norm <= trust_cap else trust_cap / delta_norm
        delta64 *= trust_scale
    flat_scores = held_scores.reshape(-1, classes)
    flat_features = held_features.reshape(-1, dimension)
    flat_targets = held_targets.reshape(-1)
    updated_scores = flat_scores + flat_features @ delta64.T
    before_ce_rows = _cross_entropy(flat_scores, flat_targets)
    after_ce_rows = _cross_entropy(updated_scores, flat_targets)
    before_ce = np.asarray(
        [np.mean(before_ce_rows[flat_targets == index]) for index in range(classes)]
    )
    after_ce = np.asarray(
        [np.mean(after_ce_rows[flat_targets == index]) for index in range(classes)]
    )
    ce_scale = max(1.0, float(np.max(np.abs(before_ce))))
    ce_tolerance = float(1024.0 * np.finfo(np.float64).eps * ce_scale)
    if active and (
        np.any(after_ce > before_ce + ce_tolerance)
        or not np.any(after_ce < before_ce - ce_tolerance)
    ):
        raise D77CommonDescentError("D77 analytic common-descent CE drift")
    base_prediction = np.argmax(flat_scores, axis=1)
    updated_prediction = np.argmax(updated_scores, axis=1)
    delta = _readonly(delta64, np.float32)
    audit = {
        "schema": "cvs.phase2.d77.ground_preconditioned_common_descent_audit.v1",
        "status": (
            "ground_preconditioned_allclass_common_descent_active"
            if active
            else "degenerate_minimum_norm_exact_d62_fallback"
        ),
        "class_count": classes,
        "k_shot": int(k_shot),
        "dimension": dimension,
        "crossfit_fold_count": int(k_shot),
        "crossfit_lda_fit_count": int(k_shot),
        "crossfit_held_row_count": classes * int(k_shot),
        "crossfit_unique_lda_coefficient_count": len(fold_coefficient_hashes),
        "class_gradient_count": classes,
        "class_gradient_sha256": _sha256(gradients.astype(np.float64)),
        "frank_wolfe_iterations": FW_ITERATIONS,
        "optimizer_objective_trace": objective_trace,
        "simplex_weights": [float(value) for value in alpha],
        "simplex_weight_sum": float(np.sum(alpha)),
        "simplex_weight_min": float(np.min(alpha)),
        "simplex_weight_max": float(np.max(alpha)),
        "minimum_common_descent_inner_product": float(np.min(common_inner)),
        "maximum_common_descent_inner_product": float(np.max(common_inner)),
        "preconditioned_direction_norm_sq": direction_norm_sq,
        "lipschitz_min": float(np.min(lipschitz)),
        "lipschitz_max": float(np.max(lipschitz)),
        "analytic_raw_step": raw_step,
        "trust_cap_frobenius": trust_cap,
        "trust_scale": trust_scale,
        "residual_active": active,
        "residual_frobenius": float(np.linalg.norm(delta64)),
        "residual_sha256": _sha256(delta.astype(np.float32)),
        "oof_ce_before_mean": float(np.mean(before_ce)),
        "oof_ce_after_mean": float(np.mean(after_ce)),
        "oof_ce_delta_mean": float(np.mean(after_ce - before_ce)),
        "oof_ce_delta_max_class": float(np.max(after_ce - before_ce)),
        "oof_ce_delta_min_class": float(np.min(after_ce - before_ce)),
        "oof_per_class_ce_before": [float(value) for value in before_ce],
        "oof_per_class_ce_after": [float(value) for value in after_ce],
        "oof_per_class_ce_delta": [
            float(value) for value in (after_ce - before_ce)
        ],
        "oof_base_correct_count": int(np.sum(base_prediction == flat_targets)),
        "oof_updated_correct_count": int(
            np.sum(updated_prediction == flat_targets)
        ),
        "oof_correct_delta": int(
            np.sum(updated_prediction == flat_targets)
            - np.sum(base_prediction == flat_targets)
        ),
        "numeric_tolerance": tolerance,
        "ce_numeric_tolerance": ce_tolerance,
        "class_id_specific_formula": False,
        "class_permutation_equivariant": True,
        "old_new_role_specific_branch": False,
        "scene_receiver_handle_specific_branch": False,
        "query_rows_used": 0,
        "ground_class_score_access": False,
    }
    return delta, audit


__all__ = [
    "D77CommonDescentError",
    "FW_ITERATIONS",
    "ground_reliability_preconditioner",
    "fit_ground_preconditioned_common_descent",
]
