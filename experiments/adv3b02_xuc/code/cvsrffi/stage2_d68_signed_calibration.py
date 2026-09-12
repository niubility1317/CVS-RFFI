"""Class-agnostic signed affine calibration for the D68 probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class D68SignedCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AffineStandardization:
    coefficient: np.ndarray
    intercept: np.ndarray
    positive_mean: np.ndarray
    negative_mean: np.ndarray
    within_scale: np.ndarray
    gap_scale: np.ndarray
    scale: np.ndarray


def validate_symmetric_support(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    count, k = int(class_count), int(k_shot)
    if (
        x.ndim != 2
        or y.ndim != 1
        or len(x) != len(y)
        or count < 2
        or k < 1
        or len(x) != count * k
        or not np.isfinite(x).all()
    ):
        raise D68SignedCalibrationError("D68 symmetric support shape drift")
    if not np.array_equal(np.unique(y), np.arange(count, dtype=np.int64)):
        raise D68SignedCalibrationError("D68 support registry is not contiguous")
    if any(int(np.sum(y == index)) != k for index in range(count)):
        raise D68SignedCalibrationError("D68 support is not class symmetric")
    return x, y


def leave_one_rank_partitions(
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    y = np.asarray(labels, dtype=np.int64)
    count, k = int(class_count), int(k_shot)
    validate_symmetric_support(
        np.zeros((len(y), 1), dtype=np.float32), y, count, k
    )
    if k < 2:
        raise D68SignedCalibrationError("D68 leave-one-rank-out requires K>=2")
    by_class = [np.flatnonzero(y == index) for index in range(count)]
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    held_once = np.zeros(len(y), dtype=np.int64)
    universe = np.arange(len(y), dtype=np.int64)
    for rank in range(k):
        held = np.asarray([indices[rank] for indices in by_class], dtype=np.int64)
        train = np.setdiff1d(universe, held, assume_unique=True)
        if len(held) != count or len(train) != count * (k - 1):
            raise D68SignedCalibrationError("D68 partition cardinality drift")
        if len(np.intersect1d(train, held)) != 0:
            raise D68SignedCalibrationError("D68 held/train partition overlap")
        held_once[held] += 1
        partitions.append((train, held))
    if not np.array_equal(held_once, np.ones(len(y), dtype=np.int64)):
        raise D68SignedCalibrationError("D68 held rows are not exact-once")
    return tuple(partitions)


def standardize_affine_rows(
    coefficient: np.ndarray,
    intercept: np.ndarray,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    class_count: int,
) -> AffineStandardization:
    coef = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    x = np.asarray(support_rows, dtype=np.float64)
    y = np.asarray(support_labels, dtype=np.int64)
    count = int(class_count)
    if (
        coef.ndim != 2
        or bias.shape != (count,)
        or coef.shape[0] != count
        or x.ndim != 2
        or x.shape[1] != coef.shape[1]
        or len(x) != len(y)
        or not np.isfinite(coef).all()
        or not np.isfinite(bias).all()
        or not np.isfinite(x).all()
    ):
        raise D68SignedCalibrationError("D68 affine standardization shape drift")
    scores = x @ coef.T + bias[None, :]
    positive_mean = np.empty(count, dtype=np.float64)
    negative_mean = np.empty(count, dtype=np.float64)
    within_scale = np.empty(count, dtype=np.float64)
    for index in range(count):
        positive = scores[y == index, index]
        negative = scores[y != index, index]
        if len(positive) == 0 or len(negative) == 0:
            raise D68SignedCalibrationError("D68 one-vs-rest support is empty")
        positive_mean[index] = float(np.mean(positive))
        negative_mean[index] = float(np.mean(negative))
        within_scale[index] = float(
            np.sqrt(0.5 * (np.var(positive) + np.var(negative)))
        )
    gap_scale = 0.5 * np.abs(positive_mean - negative_mean)
    scale = np.maximum.reduce(
        (
            within_scale,
            gap_scale,
            np.full(count, np.finfo(np.float32).eps, dtype=np.float64),
        )
    )
    center = 0.5 * (positive_mean + negative_mean)
    standardized_coefficient = coef / scale[:, None]
    standardized_intercept = (bias - center) / scale
    if (
        not np.isfinite(standardized_coefficient).all()
        or not np.isfinite(standardized_intercept).all()
    ):
        raise D68SignedCalibrationError("D68 standardized affine became non-finite")
    return AffineStandardization(
        coefficient=standardized_coefficient,
        intercept=standardized_intercept,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        within_scale=within_scale,
        gap_scale=gap_scale,
        scale=scale,
    )


def standardized_scores(
    rows: np.ndarray, state: AffineStandardization
) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    scores = x @ state.coefficient.T + state.intercept[None, :]
    if not np.isfinite(scores).all():
        raise D68SignedCalibrationError("D68 standardized score became non-finite")
    return scores


def class_balanced_squared_risk(
    scores: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    count = int(class_count)
    if values.ndim != 2 or values.shape != (len(y), count):
        raise D68SignedCalibrationError("D68 risk score shape drift")
    risk = np.empty(count, dtype=np.float64)
    for index in range(count):
        positive = values[y == index, index]
        negative = values[y != index, index]
        risk[index] = 0.5 * float(np.mean((positive - 1.0) ** 2)) + 0.5 * float(
            np.mean((negative + 1.0) ** 2)
        )
    return risk


def solve_orientations(
    held_scores: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(held_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    count = int(class_count)
    if scores.shape != (len(y), count) or not np.isfinite(scores).all():
        raise D68SignedCalibrationError("D68 held score shape drift")
    positive_mean = np.empty(count, dtype=np.float64)
    negative_mean = np.empty(count, dtype=np.float64)
    for index in range(count):
        positive_mean[index] = float(np.mean(scores[y == index, index]))
        negative_mean[index] = float(np.mean(scores[y != index, index]))
    delta = positive_mean - negative_mean
    orientation = np.where(delta >= 0.0, 1.0, -1.0)
    signed = scores * orientation[None, :]
    risk_raw = class_balanced_squared_risk(scores, y, count)
    risk_signed = class_balanced_squared_risk(signed, y, count)
    return orientation, {
        "crossfit_positive_mean": positive_mean,
        "crossfit_negative_mean": negative_mean,
        "crossfit_delta": delta,
        "risk_raw": risk_raw,
        "risk_signed": risk_signed,
    }


def compile_signed_affine(
    state: AffineStandardization,
    orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sign = np.asarray(orientation, dtype=np.float64)
    count = state.coefficient.shape[0]
    if sign.shape != (count,) or not np.all(np.isin(sign, (-1.0, 1.0))):
        raise D68SignedCalibrationError("D68 orientation closure drift")
    coefficient = sign[:, None] * state.coefficient
    intercept = sign * state.intercept
    coefficient -= np.mean(coefficient, axis=0, keepdims=True)
    intercept -= float(np.mean(intercept))
    result_coefficient = coefficient.astype(np.float32)
    result_intercept = intercept.astype(np.float32)
    if (
        not np.isfinite(result_coefficient).all()
        or not np.isfinite(result_intercept).all()
    ):
        raise D68SignedCalibrationError("D68 compiled affine became non-finite")
    return result_coefficient, result_intercept
