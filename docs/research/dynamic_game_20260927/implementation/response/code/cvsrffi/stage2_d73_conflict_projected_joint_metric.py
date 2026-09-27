"""One-step old/new conflict-projected diagonal metric update for D73."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


class D73MetricError(RuntimeError):
    """Raised when D73 support or first-order evidence drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _validate_support(
    rows: np.ndarray,
    labels: np.ndarray,
    old_class_count: int,
    class_count: int,
    k_shot: int,
    base_log_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    log_diag = np.asarray(base_log_diag, dtype=np.float64)
    if (
        x.ndim != 2
        or x.shape[1] < 2
        or y.shape != (len(x),)
        or log_diag.shape != (x.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(log_diag).all()
        or int(old_class_count) < 2
        or int(class_count) <= int(old_class_count)
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(
            int(np.sum(y == index)) != int(k_shot)
            for index in range(int(class_count))
        )
    ):
        raise D73MetricError("D73 requires finite exact symmetric registered support")
    return np.ascontiguousarray(x), np.ascontiguousarray(y), log_diag


def _leave_one_differences(
    rows: np.ndarray, labels: np.ndarray, class_count: int
) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=int(class_count)).astype(np.float64)
    if np.min(counts) <= 1.0:
        raise D73MetricError("D73 leave-one objective requires at least K2")
    sums = np.stack([np.sum(x[y == index], axis=0) for index in range(class_count)])
    centroids = sums / counts[:, None]
    sample_centroids = np.broadcast_to(
        centroids[None, :, :], (len(x), int(class_count), x.shape[1])
    ).copy()
    true_centroids = (sums[y] - x) / (counts[y, None] - 1.0)
    sample_centroids[np.arange(len(x)), y] = true_centroids
    return x[:, None, :] - sample_centroids


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _task_loss_gradient(
    squared_difference: np.ndarray,
    labels: np.ndarray,
    task_mask: np.ndarray,
    squared_scale: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray, float]:
    distances = np.einsum(
        "ncd,d->nc", squared_difference, squared_scale, optimize=True
    )
    logits = -distances / float(temperature)
    probabilities = _softmax(logits)
    indices = np.flatnonzero(task_mask)
    if len(indices) == 0:
        raise D73MetricError("D73 task partition is empty")
    chosen = np.clip(
        probabilities[indices, labels[indices]], np.finfo(np.float64).tiny, 1.0
    )
    loss = float(-np.mean(np.log(chosen)))
    residual = probabilities[indices].copy()
    residual[np.arange(len(indices)), labels[indices]] -= 1.0
    gradient = (-2.0 / float(temperature)) * squared_scale * np.mean(
        np.einsum(
            "nc,ncd->nd",
            residual,
            squared_difference[indices],
            optimize=True,
        ),
        axis=0,
    )
    accuracy = float(
        np.mean(np.argmax(logits[indices], axis=1) == labels[indices])
    )
    return loss, gradient, accuracy


def _loss_accuracy(
    squared_difference: np.ndarray,
    labels: np.ndarray,
    task_mask: np.ndarray,
    squared_scale: np.ndarray,
    temperature: float,
) -> tuple[float, float]:
    distances = np.einsum(
        "ncd,d->nc", squared_difference, squared_scale, optimize=True
    )
    probabilities = _softmax(-distances / float(temperature))
    indices = np.flatnonzero(task_mask)
    chosen = np.clip(
        probabilities[indices, labels[indices]], np.finfo(np.float64).tiny, 1.0
    )
    return (
        float(-np.mean(np.log(chosen))),
        float(np.mean(np.argmax(probabilities[indices], axis=1) == labels[indices])),
    )


def fit_conflict_projected_log_diag(
    rows: np.ndarray,
    labels: np.ndarray,
    old_class_count: int,
    class_count: int,
    k_shot: int,
    base_log_diag: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one deterministic support-only D73 log-diagonal update and audit."""

    x, y, base = _validate_support(
        rows, labels, old_class_count, class_count, k_shot, base_log_diag
    )
    dimension = int(x.shape[1])
    if int(k_shot) == 1:
        audit = {
            "schema": "cvs.phase2.d73.conflict_projected_joint_metric_audit.v1",
            "status": "k1_exact_d62_fallback",
            "class_count": int(class_count),
            "old_class_count": int(old_class_count),
            "new_class_count": int(class_count - old_class_count),
            "k_shot": 1,
            "dimension": dimension,
            "stage2c_step_count": 0,
            "temperature": None,
            "old_loss_before": None,
            "new_loss_before": None,
            "old_loss_after": None,
            "new_loss_after": None,
            "old_gradient_l2": 0.0,
            "new_gradient_l2": 0.0,
            "task_gradient_cosine": None,
            "conflict_projection_active": False,
            "delta_l2": 0.0,
            "delta_rms": 0.0,
            "delta_max_abs": 0.0,
            "delta_mean": 0.0,
            "old_first_order_change": 0.0,
            "new_first_order_change": 0.0,
            "ground_component_input_count": 0,
            "query_rows_used": 0,
        }
        return _readonly(base, np.float32), audit

    difference = _leave_one_differences(x, y, class_count)
    squared_difference = difference * difference
    squared_scale = np.exp(2.0 * base)
    true_distances = np.einsum(
        "nd,d->n",
        squared_difference[np.arange(len(y)), y],
        squared_scale,
        optimize=True,
    )
    positive = true_distances[true_distances > np.finfo(np.float64).eps]
    temperature = float(np.median(positive)) if len(positive) else 1.0
    temperature = max(temperature, np.finfo(np.float64).eps)
    old_mask = y < int(old_class_count)
    new_mask = ~old_mask
    old_loss, old_gradient, old_accuracy = _task_loss_gradient(
        squared_difference, y, old_mask, squared_scale, temperature
    )
    new_loss, new_gradient, new_accuracy = _task_loss_gradient(
        squared_difference, y, new_mask, squared_scale, temperature
    )
    old_norm = float(np.linalg.norm(old_gradient))
    new_norm = float(np.linalg.norm(new_gradient))
    epsilon = np.finfo(np.float64).eps
    if old_norm <= epsilon or new_norm <= epsilon:
        raise D73MetricError("D73 task gradient is degenerate")
    old_unit = old_gradient / old_norm
    new_unit = new_gradient / new_norm
    cosine = float(np.clip(np.dot(old_unit, new_unit), -1.0, 1.0))
    if cosine < 0.0:
        old_projected = old_unit - cosine * new_unit
        new_projected = new_unit - cosine * old_unit
        projection_active = True
    else:
        old_projected = old_unit
        new_projected = new_unit
        projection_active = False
    joint = old_projected + new_projected
    joint -= float(np.mean(joint))
    joint_norm = float(np.linalg.norm(joint))
    if joint_norm <= epsilon:
        raise D73MetricError("D73 centered joint direction is degenerate")
    trust_radius = float(np.sqrt(float(k_shot) / (float(k_shot) + dimension)))
    delta = -trust_radius * joint / joint_norm
    updated = base + delta
    updated_squared_scale = np.exp(2.0 * updated)
    old_loss_after, old_accuracy_after = _loss_accuracy(
        squared_difference, y, old_mask, updated_squared_scale, temperature
    )
    new_loss_after, new_accuracy_after = _loss_accuracy(
        squared_difference, y, new_mask, updated_squared_scale, temperature
    )
    old_first_order = float(np.dot(old_gradient, delta))
    new_first_order = float(np.dot(new_gradient, delta))
    if old_first_order > 1e-10 or new_first_order > 1e-10:
        raise D73MetricError("D73 conflict projection failed first-order nonincrease")
    direction_payload = {
        "old_gradient": np.round(old_gradient, 12).tolist(),
        "new_gradient": np.round(new_gradient, 12).tolist(),
        "delta": np.round(delta, 12).tolist(),
    }
    audit = {
        "schema": "cvs.phase2.d73.conflict_projected_joint_metric_audit.v1",
        "status": "one_step_conflict_projected_joint_metric_active",
        "class_count": int(class_count),
        "old_class_count": int(old_class_count),
        "new_class_count": int(class_count - old_class_count),
        "k_shot": int(k_shot),
        "dimension": dimension,
        "stage2c_step_count": 1,
        "temperature": temperature,
        "temperature_rule": "median_base_leave_one_true_squared_distance",
        "trust_radius": trust_radius,
        "trust_radius_rule": "sqrt(K/(K+D))",
        "old_loss_before": old_loss,
        "new_loss_before": new_loss,
        "old_loss_after": old_loss_after,
        "new_loss_after": new_loss_after,
        "old_support_accuracy_before": old_accuracy,
        "new_support_accuracy_before": new_accuracy,
        "old_support_accuracy_after": old_accuracy_after,
        "new_support_accuracy_after": new_accuracy_after,
        "old_gradient_l2": old_norm,
        "new_gradient_l2": new_norm,
        "task_gradient_cosine": cosine,
        "conflict_projection_active": projection_active,
        "delta_l2": float(np.linalg.norm(delta)),
        "delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "delta_max_abs": float(np.max(np.abs(delta))),
        "delta_mean": float(np.mean(delta)),
        "old_first_order_change": old_first_order,
        "new_first_order_change": new_first_order,
        "first_order_both_nonincreasing": True,
        "direction_sha256": hashlib.sha256(
            json.dumps(
                direction_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "class_id_specific_formula": False,
        "within_role_class_permutation_equivariant": True,
        "scene_receiver_handle_specific_branch": False,
        "ground_component_input_count": 0,
        "query_rows_used": 0,
    }
    return _readonly(updated, np.float32), audit
