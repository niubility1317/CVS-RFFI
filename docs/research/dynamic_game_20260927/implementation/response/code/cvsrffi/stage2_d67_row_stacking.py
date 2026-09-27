"""Pure support-only math for D67 registry-consistent affine row stacking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = float(np.finfo(np.float32).eps)


class D67RowStackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RowStandardization:
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
    x = np.asarray(rows, dtype=np.float64)
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
            int(np.sum(y == class_index)) != int(k_shot)
            for class_index in range(int(class_count))
        )
    ):
        raise D67RowStackingError("D67 requires finite exact symmetric support")
    return x, y


def four_rank_partitions(
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    y = np.asarray(labels, dtype=np.int64)
    if int(k_shot) < 8 or int(k_shot) % 4 != 0:
        raise D67RowStackingError("D67 four-rank cross-fit requires K>=8 divisible by four")
    by_class = [np.flatnonzero(y == index) for index in range(int(class_count))]
    if any(len(indices) != int(k_shot) for indices in by_class):
        raise D67RowStackingError("D67 partition support is not symmetric")
    all_indices = np.arange(len(y), dtype=np.int64)
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    held_union: list[int] = []
    for fold_index in range(4):
        held = np.sort(
            np.concatenate(
                [indices[fold_index::4] for indices in by_class]
            )
        ).astype(np.int64)
        train = np.setdiff1d(all_indices, held, assume_unique=True)
        if (
            len(held) != int(class_count) * (int(k_shot) // 4)
            or len(train) != int(class_count) * (int(k_shot) * 3 // 4)
        ):
            raise D67RowStackingError("D67 partition cardinality drift")
        held_union.extend(int(value) for value in held)
        partitions.append((train, held))
    if sorted(held_union) != all_indices.tolist():
        raise D67RowStackingError("D67 held ranks are not exact-once")
    return tuple(partitions)


def standardize_affine_rows(
    coefficient: np.ndarray,
    intercept: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> RowStandardization:
    coef = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        coef.shape != (int(class_count), x.shape[1])
        or bias.shape != (int(class_count),)
        or y.shape != (len(x),)
        or not np.isfinite(coef).all()
        or not np.isfinite(bias).all()
    ):
        raise D67RowStackingError("D67 affine standardization shape drift")
    scores = x @ coef.T + bias[None, :]
    positive_mean = np.empty(int(class_count), dtype=np.float64)
    negative_mean = np.empty(int(class_count), dtype=np.float64)
    within_scale = np.empty(int(class_count), dtype=np.float64)
    gap_scale = np.empty(int(class_count), dtype=np.float64)
    for class_index in range(int(class_count)):
        positive = scores[y == class_index, class_index]
        negative = scores[y != class_index, class_index]
        if len(positive) == 0 or len(negative) == 0:
            raise D67RowStackingError("D67 one-vs-rest support is empty")
        positive_mean[class_index] = float(np.mean(positive))
        negative_mean[class_index] = float(np.mean(negative))
        within_scale[class_index] = float(
            np.sqrt(0.5 * (np.var(positive) + np.var(negative)))
        )
        gap_scale[class_index] = float(
            0.5 * abs(positive_mean[class_index] - negative_mean[class_index])
        )
    scale = np.maximum(np.maximum(within_scale, gap_scale), EPSILON)
    center = 0.5 * (positive_mean + negative_mean)
    standardized_coefficient = coef / scale[:, None]
    standardized_intercept = (bias - center) / scale
    if (
        not np.isfinite(standardized_coefficient).all()
        or not np.isfinite(standardized_intercept).all()
    ):
        raise D67RowStackingError("D67 standardized affine became non-finite")
    return RowStandardization(
        coefficient=standardized_coefficient,
        intercept=standardized_intercept,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        within_scale=within_scale,
        gap_scale=gap_scale,
        scale=scale,
    )


def standardized_scores(rows: np.ndarray, state: RowStandardization) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    scores = x @ state.coefficient.T + state.intercept[None, :]
    if not np.isfinite(scores).all():
        raise D67RowStackingError("D67 standardized score became non-finite")
    return scores


def solve_class_balanced_convex_weights(
    d62_scores: np.ndarray,
    d65_scores: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    first = np.asarray(d62_scores, dtype=np.float64)
    second = np.asarray(d65_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        first.shape != second.shape
        or first.shape != (len(y), int(class_count))
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise D67RowStackingError("D67 stacking score shape drift")
    alpha = np.zeros(int(class_count), dtype=np.float64)
    numerator = np.zeros_like(alpha)
    denominator = np.zeros_like(alpha)
    risk_d62 = np.zeros_like(alpha)
    risk_d65 = np.zeros_like(alpha)
    risk_stacked = np.zeros_like(alpha)
    for class_index in range(int(class_count)):
        positive = y == class_index
        negative = ~positive
        if not np.any(positive) or not np.any(negative):
            raise D67RowStackingError("D67 stacking one-vs-rest support is empty")
        weights = np.empty(len(y), dtype=np.float64)
        weights[positive] = 0.5 / float(np.sum(positive))
        weights[negative] = 0.5 / float(np.sum(negative))
        target = np.where(positive, 1.0, -1.0)
        base = first[:, class_index]
        delta = second[:, class_index] - base
        numerator[class_index] = float(np.sum(weights * delta * (target - base)))
        denominator[class_index] = float(np.sum(weights * delta * delta))
        if denominator[class_index] > EPSILON:
            alpha[class_index] = float(
                np.clip(numerator[class_index] / denominator[class_index], 0.0, 1.0)
            )
        stacked = base + alpha[class_index] * delta
        risk_d62[class_index] = float(np.sum(weights * (target - base) ** 2))
        risk_d65[class_index] = float(
            np.sum(weights * (target - second[:, class_index]) ** 2)
        )
        risk_stacked[class_index] = float(np.sum(weights * (target - stacked) ** 2))
        if risk_stacked[class_index] > risk_d62[class_index] + 1.0e-12:
            raise D67RowStackingError("D67 analytic stacking risk exceeded D62")
    return alpha, {
        "numerator": numerator,
        "denominator": denominator,
        "risk_d62": risk_d62,
        "risk_d65": risk_d65,
        "risk_stacked": risk_stacked,
    }


def compile_stacked_affine(
    d62_state: RowStandardization,
    d65_state: RowStandardization,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    weight = np.asarray(alpha, dtype=np.float64)
    if (
        d62_state.coefficient.shape != d65_state.coefficient.shape
        or d62_state.intercept.shape != d65_state.intercept.shape
        or weight.shape != d62_state.intercept.shape
        or np.any(weight < 0.0)
        or np.any(weight > 1.0)
    ):
        raise D67RowStackingError("D67 compiled stacking shape or weight drift")
    normalized_coefficient = (
        (1.0 - weight[:, None]) * d62_state.coefficient
        + weight[:, None] * d65_state.coefficient
    )
    normalized_intercept = (
        (1.0 - weight) * d62_state.intercept
        + weight * d65_state.intercept
    )
    d62_center = 0.5 * (d62_state.positive_mean + d62_state.negative_mean)
    coefficient = d62_state.scale[:, None] * normalized_coefficient
    intercept = d62_state.scale * normalized_intercept + d62_center
    coefficient -= np.mean(coefficient, axis=0, keepdims=True)
    intercept -= float(np.mean(intercept))
    coef32 = coefficient.astype(np.float32)
    intercept32 = intercept.astype(np.float32)
    error = float(
        max(
            np.max(np.abs(coefficient - coef32.astype(np.float64))),
            np.max(np.abs(intercept - intercept32.astype(np.float64))),
        )
    )
    if not np.isfinite(coef32).all() or not np.isfinite(intercept32).all():
        raise D67RowStackingError("D67 compiled affine became non-finite")
    return coef32, intercept32, error
