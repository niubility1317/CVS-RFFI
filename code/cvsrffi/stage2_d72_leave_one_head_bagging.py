"""Physical-rank leave-one affine-head bagging for D72."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np


class D72BaggingError(RuntimeError):
    """Raised when D72 support, partitions, or affine evidence drift."""


Fit = Callable[
    [np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _validate_support(
    rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(
            int(np.sum(y == index)) != int(k_shot)
            for index in range(int(class_count))
        )
    ):
        raise D72BaggingError("D72 requires finite exact symmetric support")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def physical_rank_leave_one_partitions(
    labels: np.ndarray, class_count: int, k_shot: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(labels, dtype=np.int64)
    by_class = [np.flatnonzero(y == index) for index in range(int(class_count))]
    if any(len(indices) != int(k_shot) for indices in by_class):
        raise D72BaggingError("D72 partition requires symmetric support")
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for rank in range(int(k_shot)):
        held = np.asarray(
            [indices[rank] for indices in by_class], dtype=np.int64
        )
        train = np.setdiff1d(
            np.arange(len(y), dtype=np.int64), held, assume_unique=True
        )
        partitions.append((train, held))
    held_flat = (
        np.concatenate([held for _, held in partitions])
        if partitions
        else np.empty(0, dtype=np.int64)
    )
    if (
        len(held_flat) != len(y)
        or len(np.unique(held_flat)) != len(y)
        or not np.array_equal(np.sort(held_flat), np.arange(len(y)))
        or any(
            len(np.intersect1d(train, held)) != 0
            for train, held in partitions
        )
    ):
        raise D72BaggingError("D72 physical-rank partition exact-once drift")
    return partitions


def _validate_affine(
    coefficient: np.ndarray,
    intercept: np.ndarray,
    class_count: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    if (
        weights.shape != (int(class_count), int(dimension))
        or bias.shape != (int(class_count),)
        or not np.isfinite(weights).all()
        or not np.isfinite(bias).all()
    ):
        raise D72BaggingError("D72 affine head drift")
    return weights, bias


def _center_affine(
    coefficient: np.ndarray, intercept: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    return (
        weights - np.mean(weights, axis=0, keepdims=True),
        bias - float(np.mean(bias)),
    )

def fit_leave_one_bagged_affine(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    fit: Fit,
    fallback_coefficient: np.ndarray,
    fallback_intercept: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, y = _validate_support(rows, labels, class_count, k_shot)
    fallback_w, fallback_b = _validate_affine(
        fallback_coefficient,
        fallback_intercept,
        class_count,
        x.shape[1],
    )
    if int(k_shot) <= 2:
        return (
            _readonly(fallback_w, np.float32),
            _readonly(fallback_b, np.float32),
            {
                "schema": "cvs.phase2.d72.leave_one_head_bagging_audit.v1",
                "status": "k1_k2_exact_d62_fallback",
                "class_count": int(class_count),
                "k_shot": int(k_shot),
                "inner_k_shot": None,
                "leave_one_fit_count": 0,
                "partition_exact_once": True,
                "partition_audit": [],
                "support_prediction_change_count": 0,
                "support_accuracy_base": float(
                    np.mean(np.argmax(x @ fallback_w.T + fallback_b, axis=1) == y)
                ),
                "support_accuracy_bagged": float(
                    np.mean(np.argmax(x @ fallback_w.T + fallback_b, axis=1) == y)
                ),
                "coefficient_dispersion_rms": 0.0,
                "coefficient_dispersion_max_row_l2": 0.0,
                "intercept_dispersion_rms": 0.0,
                "inner_audit_sha256": hashlib.sha256(b"[]").hexdigest(),
            },
        )

    partitions = physical_rank_leave_one_partitions(y, class_count, k_shot)
    coefficients: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []
    audit_summaries: list[dict[str, Any]] = []
    partition_audit: list[dict[str, Any]] = []
    for rank, (train, held) in enumerate(partitions):
        coefficient, intercept, inner = fit(
            x[train], y[train], int(class_count), int(k_shot) - 1
        )
        weights, bias = _validate_affine(
            coefficient, intercept, class_count, x.shape[1]
        )
        coefficients.append(weights)
        intercepts.append(bias)
        accept = inner.get("d62_final_accept_mask", [])
        audit_summaries.append(
            {
                "rank": int(rank),
                "d62_boundary_status": str(
                    inner.get("d62_boundary_status", "missing")
                ),
                "d62_accept_count": int(np.sum(np.asarray(accept, dtype=bool))),
                "coefficient_l2": float(np.linalg.norm(weights)),
                "intercept_l2": float(np.linalg.norm(bias)),
            }
        )
        partition_audit.append(
            {
                "rank": int(rank),
                "train_count": int(len(train)),
                "held_count": int(len(held)),
                "train_held_overlap_count": int(
                    len(np.intersect1d(train, held))
                ),
                "held_indices": held.tolist(),
                "held_class_histogram": np.bincount(
                    y[held], minlength=int(class_count)
                ).tolist(),
            }
        )

    stack_w = np.stack(coefficients, axis=0)
    stack_b = np.stack(intercepts, axis=0)
    mean_w, mean_b = _center_affine(
        np.mean(stack_w, axis=0), np.mean(stack_b, axis=0)
    )
    base_scores = x.astype(np.float64) @ fallback_w.T + fallback_b
    bagged_scores = x.astype(np.float64) @ mean_w.T + mean_b
    base_prediction = np.argmax(base_scores, axis=1)
    bagged_prediction = np.argmax(bagged_scores, axis=1)
    coefficient_delta = stack_w - np.mean(stack_w, axis=0, keepdims=True)
    intercept_delta = stack_b - np.mean(stack_b, axis=0, keepdims=True)
    audit_json = json.dumps(
        audit_summaries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    audit = {
        "schema": "cvs.phase2.d72.leave_one_head_bagging_audit.v1",
        "status": "physical_rank_leave_one_head_bagging_active",
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "inner_k_shot": int(k_shot) - 1,
        "leave_one_fit_count": int(len(partitions)),
        "partition_exact_once": True,
        "partition_audit": partition_audit,
        "support_prediction_change_count": int(
            np.sum(base_prediction != bagged_prediction)
        ),
        "support_accuracy_base": float(np.mean(base_prediction == y)),
        "support_accuracy_bagged": float(np.mean(bagged_prediction == y)),
        "coefficient_dispersion_rms": float(
            np.sqrt(np.mean(coefficient_delta**2))
        ),
        "coefficient_dispersion_max_row_l2": float(
            np.max(np.linalg.norm(coefficient_delta, axis=2))
        ),
        "intercept_dispersion_rms": float(
            np.sqrt(np.mean(intercept_delta**2))
        ),
        "inner_audit_sha256": hashlib.sha256(audit_json).hexdigest(),
        "inner_audit_summary": audit_summaries,
        "class_common_coefficient_center_max_abs": float(
            np.max(np.abs(np.mean(mean_w, axis=0)))
        ),
        "class_common_intercept_center_abs": float(abs(np.mean(mean_b))),
    }
    return (
        _readonly(mean_w, np.float32),
        _readonly(mean_b, np.float32),
        audit,
    )
