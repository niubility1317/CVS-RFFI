"""Cross-fitted atomic-safe top-2 centroid reranker for D71."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np


class D71RerankerError(RuntimeError):
    """Raised when D71 support, gate, or state evidence is invalid."""


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
        raise D71RerankerError("D71 requires finite exact symmetric support")
    return np.ascontiguousarray(x), y


def twofold_rank_partitions(
    labels: np.ndarray, class_count: int, k_shot: int
) -> list[np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    by_class = [np.flatnonzero(y == index) for index in range(int(class_count))]
    if any(len(indices) != int(k_shot) for indices in by_class):
        raise D71RerankerError("D71 partition requires symmetric support")
    held = [
        np.concatenate([indices[offset::2] for indices in by_class]).astype(np.int64)
        for offset in range(min(2, int(k_shot)))
    ]
    flat = np.concatenate(held) if held else np.empty(0, dtype=np.int64)
    if (
        len(flat) != len(y)
        or len(np.unique(flat)) != len(flat)
        or not np.array_equal(np.sort(flat), np.arange(len(y)))
    ):
        raise D71RerankerError("D71 held partition exact-once drift")
    return held


def _pairs(class_count: int) -> np.ndarray:
    return np.asarray(list(combinations(range(int(class_count)), 2)), dtype=np.int64)


def centroid_pair_affine(
    rows: np.ndarray, labels: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise D71RerankerError("D71 centroid input shape drift")
    means = np.stack([x[y == index].mean(axis=0) for index in range(int(class_count))])
    pair_index = _pairs(int(class_count))
    directions = means[pair_index[:, 0]] - means[pair_index[:, 1]]
    biases = -0.5 * (
        np.sum(means[pair_index[:, 0]] ** 2, axis=1)
        - np.sum(means[pair_index[:, 1]] ** 2, axis=1)
    )
    if (
        not np.isfinite(directions).all()
        or not np.isfinite(biases).all()
        or bool(np.any(np.linalg.norm(directions, axis=1) <= 1.0e-12))
    ):
        raise D71RerankerError("D71 centroid pair became degenerate")
    return pair_index, directions.astype(np.float32), biases.astype(np.float32)


def _counts(scores: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    predicted = np.argmax(values, axis=1)
    positive = np.asarray(
        [np.sum(predicted[labels == index] == index) for index in range(values.shape[1])],
        dtype=np.int64,
    )
    false_positive = np.asarray(
        [np.sum(predicted[labels != index] == index) for index in range(values.shape[1])],
        dtype=np.int64,
    )
    return positive, false_positive


def _stable_top2(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :2]


def rerank_top2_fp32(
    base_scores: np.ndarray,
    transformed: np.ndarray,
    pair_index: np.ndarray,
    directions: np.ndarray,
    biases: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, int]:
    scores = np.asarray(base_scores, dtype=np.float32).copy()
    x = np.asarray(transformed, dtype=np.float32)
    pairs = np.asarray(pair_index, dtype=np.int64)
    mask = np.asarray(accepted, dtype=bool)
    if scores.ndim != 2 or x.ndim != 2 or len(scores) != len(x):
        raise D71RerankerError("D71 rerank input shape drift")
    lookup = {
        (int(pair[0]), int(pair[1])): index
        for index, pair in enumerate(pairs)
        if bool(mask[index])
    }
    changed = 0
    for row_index, top in enumerate(_stable_top2(scores)):
        first, second = int(top[0]), int(top[1])
        key = (min(first, second), max(first, second))
        pair_position = lookup.get(key)
        if pair_position is None:
            continue
        value = float(
            x[row_index] @ np.asarray(directions[pair_position], dtype=np.float32)
            + float(biases[pair_position])
        )
        predicted = key[0] if value >= 0.0 else key[1]
        if predicted != first:
            scores[row_index, first], scores[row_index, second] = (
                scores[row_index, second],
                scores[row_index, first],
            )
            changed += 1
    return scores, changed


@dataclass(frozen=True)
class PairRerankerState:
    precision: str
    class_count: int
    dimension: int
    pair_index: np.ndarray
    direction_qint8: np.ndarray
    scale_fp16: np.ndarray
    bias_fp16: np.ndarray
    direction_fp32: np.ndarray
    bias_fp32: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.pair_index)
        int8_mode = self.precision == "int8"
        fp32_mode = self.precision == "fp32"
        valid = (
            (int8_mode or fp32_mode)
            and int(self.class_count) >= 2
            and int(self.dimension) >= 1
            and self.pair_index.dtype == np.int64
            and self.pair_index.shape == (count, 2)
            and all(int(a) < int(b) for a, b in self.pair_index)
        )
        if int8_mode:
            valid = valid and (
                self.direction_qint8.dtype == np.int8
                and self.direction_qint8.shape == (count, int(self.dimension))
                and self.scale_fp16.dtype == np.float16
                and self.scale_fp16.shape == (count,)
                and self.bias_fp16.dtype == np.float16
                and self.bias_fp16.shape == (count,)
                and self.direction_fp32.shape == (0, int(self.dimension))
                and self.bias_fp32.shape == (0,)
                and np.isfinite(self.scale_fp16).all()
                and np.all(self.scale_fp16 > 0)
                and np.isfinite(self.bias_fp16).all()
            )
        else:
            valid = valid and (
                self.direction_qint8.shape == (0, int(self.dimension))
                and self.scale_fp16.shape == (0,)
                and self.bias_fp16.shape == (0,)
                and self.direction_fp32.dtype == np.float32
                and self.direction_fp32.shape == (count, int(self.dimension))
                and self.bias_fp32.dtype == np.float32
                and self.bias_fp32.shape == (count,)
                and np.isfinite(self.direction_fp32).all()
                and np.isfinite(self.bias_fp32).all()
            )
        if not valid:
            raise D71RerankerError("D71 pair state drift")
        for name, dtype in (
            ("pair_index", np.int64),
            ("direction_qint8", np.int8),
            ("scale_fp16", np.float16),
            ("bias_fp16", np.float16),
            ("direction_fp32", np.float32),
            ("bias_fp32", np.float32),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for value in (
                    self.pair_index,
                    self.direction_qint8,
                    self.scale_fp16,
                    self.bias_fp16,
                    self.direction_fp32,
                    self.bias_fp32,
                )
            )
        )


def compile_pair_states(
    pairs: np.ndarray, directions: np.ndarray, biases: np.ndarray, accepted: np.ndarray
) -> tuple[PairRerankerState, PairRerankerState, dict[str, Any]]:
    selected = np.flatnonzero(np.asarray(accepted, dtype=bool))
    active_pairs = np.asarray(pairs[selected], dtype=np.int64)
    active_directions = np.asarray(directions[selected], dtype=np.float32)
    active_biases = np.asarray(biases[selected], dtype=np.float32)
    dimension = int(directions.shape[1])
    if len(selected):
        max_abs = np.max(np.abs(active_directions), axis=1)
        scales = np.asarray(max_abs / 127.0, dtype=np.float16)
        if bool(np.any(scales <= 0)) or not np.isfinite(scales).all():
            raise D71RerankerError("D71 pair quantization scale drift")
        codes = np.clip(
            np.rint(active_directions / scales.astype(np.float32)[:, None]),
            -127,
            127,
        ).astype(np.int8)
        decoded = codes.astype(np.float32) * scales.astype(np.float32)[:, None]
    else:
        scales = np.empty(0, dtype=np.float16)
        codes = np.empty((0, dimension), dtype=np.int8)
        decoded = np.empty((0, dimension), dtype=np.float32)
    int8 = PairRerankerState(
        precision="int8",
        class_count=int(np.max(pairs) + 1),
        dimension=dimension,
        pair_index=active_pairs,
        direction_qint8=codes,
        scale_fp16=scales,
        bias_fp16=active_biases.astype(np.float16),
        direction_fp32=np.empty((0, dimension), dtype=np.float32),
        bias_fp32=np.empty(0, dtype=np.float32),
    )
    fp32 = PairRerankerState(
        precision="fp32",
        class_count=int(np.max(pairs) + 1),
        dimension=dimension,
        pair_index=active_pairs,
        direction_qint8=np.empty((0, dimension), dtype=np.int8),
        scale_fp16=np.empty(0, dtype=np.float16),
        bias_fp16=np.empty(0, dtype=np.float16),
        direction_fp32=active_directions,
        bias_fp32=active_biases,
    )
    error = float(np.max(np.abs(decoded - active_directions))) if len(selected) else 0.0
    bias_error = (
        float(np.max(np.abs(active_biases.astype(np.float16).astype(np.float32) - active_biases)))
        if len(selected)
        else 0.0
    )
    return int8, fp32, {
        "active_pair_count": int(len(selected)),
        "direction_quantization_error_max": error,
        "bias_quantization_error_max": bias_error,
        "int8_pair_state_bytes": int(int8.persistent_state_bytes),
        "fp32_pair_state_bytes": int(fp32.persistent_state_bytes),
    }


def decode_pair_state(state: PairRerankerState) -> tuple[np.ndarray, np.ndarray]:
    if state.precision == "int8":
        directions = (
            state.direction_qint8.astype(np.float32)
            * state.scale_fp16.astype(np.float32)[:, None]
        )
        biases = state.bias_fp16.astype(np.float32)
    else:
        directions = state.direction_fp32
        biases = state.bias_fp32
    return directions, biases


def score_with_pair_state(
    base_scores: np.ndarray, transformed: np.ndarray, state: PairRerankerState
) -> tuple[np.ndarray, int]:
    directions, biases = decode_pair_state(state)
    accepted = np.ones(len(state.pair_index), dtype=bool)
    return rerank_top2_fp32(
        base_scores,
        transformed,
        state.pair_index,
        directions,
        biases,
        accepted,
    )


def fit_crossfitted_pair_reranker(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    base_fit: Fit,
) -> tuple[PairRerankerState, PairRerankerState, dict[str, Any]]:
    x, y = _validate_support(rows, labels, class_count, k_shot)
    pairs, full_directions, full_biases = centroid_pair_affine(x, y, class_count)
    pair_count = len(pairs)
    if int(k_shot) == 1:
        accepted = np.zeros(pair_count, dtype=bool)
        int8, fp32, quant = compile_pair_states(
            pairs, full_directions, full_biases, accepted
        )
        return int8, fp32, {
            "gate_status": "k1_exact_d62_fallback",
            "partition_audit": [],
            "initial_accept_mask": accepted.tolist(),
            "final_accept_mask": accepted.tolist(),
            "atomic_safe": True,
            "base_positive": [0] * int(class_count),
            "base_false_positive": [0] * int(class_count),
            "joint_positive": [0] * int(class_count),
            "joint_false_positive": [0] * int(class_count),
            "pair_base_correct": [[0, 0] for _ in range(pair_count)],
            "pair_candidate_correct": [[0, 0] for _ in range(pair_count)],
            "inner_base_fit_count": 0,
            **quant,
        }

    pair_base_correct = np.zeros((pair_count, 2), dtype=np.int64)
    pair_candidate_correct = np.zeros((pair_count, 2), dtype=np.int64)
    fold_records: list[dict[str, Any]] = []
    fold_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_index, held in enumerate(twofold_rank_partitions(y, class_count, k_shot)):
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[held] = False
        train_x, train_y = x[train_mask], y[train_mask]
        train_k = int(len(train_x) // int(class_count))
        coefficient, intercept, _ = base_fit(
            train_x, train_y, int(class_count), train_k
        )
        base_scores = (
            x[held].astype(np.float32) @ np.asarray(coefficient, dtype=np.float32).T
            + np.asarray(intercept, dtype=np.float32)[None, :]
        )
        fold_pairs, directions, biases = centroid_pair_affine(
            train_x, train_y, class_count
        )
        if not np.array_equal(fold_pairs, pairs):
            raise D71RerankerError("D71 pair registry drift")
        truth = y[held]
        for pair_position, (left, right) in enumerate(pairs):
            relevant = (truth == int(left)) | (truth == int(right))
            pair_truth = truth[relevant]
            base_pair = np.where(
                base_scores[relevant, int(left)]
                >= base_scores[relevant, int(right)],
                int(left),
                int(right),
            )
            pair_values = (
                x[held][relevant] @ directions[pair_position]
                + biases[pair_position]
            )
            candidate_pair = np.where(pair_values >= 0.0, int(left), int(right))
            for side, class_index in enumerate((int(left), int(right))):
                class_mask = pair_truth == class_index
                pair_base_correct[pair_position, side] += int(
                    np.sum(base_pair[class_mask] == class_index)
                )
                pair_candidate_correct[pair_position, side] += int(
                    np.sum(candidate_pair[class_mask] == class_index)
                )
        fold_payloads.append((x[held], truth, base_scores, directions, biases))
        fold_records.append(
            {
                "fold_index": fold_index,
                "held_indices": held.tolist(),
                "train_indices": np.flatnonzero(train_mask).tolist(),
                "train_held_overlap_count": int(
                    np.intersect1d(np.flatnonzero(train_mask), held).size
                ),
                "train_k": train_k,
            }
        )

    initial = np.all(pair_candidate_correct >= pair_base_correct, axis=1) & np.any(
        pair_candidate_correct > pair_base_correct, axis=1
    )
    base_all: list[np.ndarray] = []
    reranked_all: list[np.ndarray] = []
    truth_all: list[np.ndarray] = []
    for held_x, truth, base_scores, directions, biases in fold_payloads:
        reranked, _ = rerank_top2_fp32(
            base_scores, held_x, pairs, directions, biases, initial
        )
        base_all.append(base_scores)
        reranked_all.append(reranked)
        truth_all.append(truth)
    base_scores_all = np.concatenate(base_all)
    reranked_scores_all = np.concatenate(reranked_all)
    truth_values = np.concatenate(truth_all)
    base_positive, base_fp = _counts(base_scores_all, truth_values)
    joint_positive, joint_fp = _counts(reranked_scores_all, truth_values)
    atomic_safe = bool(
        np.all(joint_positive >= base_positive) and np.all(joint_fp <= base_fp)
    )
    accepted = initial if atomic_safe else np.zeros(pair_count, dtype=bool)
    if np.any(accepted):
        status = "crossfitted_top2_centroid_pairs_active"
    elif np.any(initial):
        status = "joint_atomic_failure_exact_d62_fallback"
    else:
        status = "no_pair_accepted_exact_d62_fallback"
    int8, fp32, quant = compile_pair_states(
        pairs, full_directions, full_biases, accepted
    )
    return int8, fp32, {
        "gate_status": status,
        "partition_audit": fold_records,
        "initial_accept_mask": initial.tolist(),
        "final_accept_mask": accepted.tolist(),
        "accepted_pairs": pairs[accepted].tolist(),
        "atomic_safe": atomic_safe,
        "base_positive": base_positive.tolist(),
        "base_false_positive": base_fp.tolist(),
        "joint_positive": joint_positive.tolist(),
        "joint_false_positive": joint_fp.tolist(),
        "pair_base_correct": pair_base_correct.tolist(),
        "pair_candidate_correct": pair_candidate_correct.tolist(),
        "inner_base_fit_count": len(fold_records),
        "partition_exact_once": sorted(
            index for record in fold_records for index in record["held_indices"]
        )
        == list(range(len(x))),
        **quant,
    }


__all__ = [
    "D71RerankerError",
    "PairRerankerState",
    "centroid_pair_affine",
    "compile_pair_states",
    "decode_pair_state",
    "fit_crossfitted_pair_reranker",
    "rerank_top2_fp32",
    "score_with_pair_state",
    "twofold_rank_partitions",
]
