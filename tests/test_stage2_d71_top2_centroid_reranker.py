from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_d71_top2_centroid_reranker import (
    D71RerankerError,
    centroid_pair_affine,
    compile_pair_states,
    decode_pair_state,
    fit_crossfitted_pair_reranker,
    rerank_top2_fp32,
    score_with_pair_state,
    twofold_rank_partitions,
)


def _support(class_count: int = 3, k: int = 4, dimension: int = 5):
    rows = []
    labels = []
    for class_index in range(class_count):
        for rank in range(k):
            row = np.zeros(dimension, dtype=np.float32)
            row[class_index] = 1.0
            row[-1] = 0.01 * rank
            rows.append(row)
            labels.append(class_index)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _bad_base_fit(rows, labels, class_count, k_shot):
    del rows, labels, k_shot
    dimension = 5
    coefficient = np.zeros((class_count, dimension), dtype=np.float32)
    intercept = np.arange(class_count, dtype=np.float32) * 1.0e-3
    return coefficient, intercept, {"synthetic": True}


def test_twofold_partition_is_exact_once_and_class_balanced() -> None:
    _, labels = _support()
    held = twofold_rank_partitions(labels, 3, 4)
    assert len(held) == 2
    assert sorted(np.concatenate(held).tolist()) == list(range(12))
    assert all(np.bincount(labels[index], minlength=3).tolist() == [2, 2, 2] for index in held)


def test_centroid_pair_affine_has_lexicographic_pair_registry() -> None:
    rows, labels = _support()
    pairs, directions, biases = centroid_pair_affine(rows, labels, 3)
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]
    assert directions.shape == (3, 5)
    assert biases.shape == (3,)
    assert float(directions[0, 0]) > 0
    assert float(directions[0, 1]) < 0


def test_top2_rerank_can_only_swap_first_two_scores() -> None:
    base = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    rows = np.asarray([[0.0, 1.0]], dtype=np.float32)
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    directions = np.asarray([[1.0, -1.0]], dtype=np.float32)
    biases = np.asarray([0.0], dtype=np.float32)
    reranked, changed = rerank_top2_fp32(
        base, rows, pairs, directions, biases, np.asarray([True])
    )
    assert changed == 1
    assert reranked.tolist() == [[2.0, 3.0, 1.0]]
    assert reranked[0, 2] == base[0, 2]


def test_unaccepted_top2_pair_is_exact_base() -> None:
    base = np.asarray([[3.0, 2.0, 1.0]], dtype=np.float32)
    reranked, changed = rerank_top2_fp32(
        base,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([[1.0, -1.0]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
    )
    assert changed == 0
    assert np.array_equal(reranked, base)


def test_pair_state_int8_and_fp32_have_same_registry_and_small_error() -> None:
    rows, labels = _support()
    pairs, directions, biases = centroid_pair_affine(rows, labels, 3)
    accepted = np.asarray([True, False, True])
    int8, fp32, audit = compile_pair_states(pairs, directions, biases, accepted)
    assert int8.pair_index.tolist() == fp32.pair_index.tolist() == [[0, 1], [1, 2]]
    decoded, decoded_bias = decode_pair_state(int8)
    exact, exact_bias = decode_pair_state(fp32)
    assert np.max(np.abs(decoded - exact)) <= audit["direction_quantization_error_max"] + 1e-7
    assert np.max(np.abs(decoded_bias - exact_bias)) <= audit["bias_quantization_error_max"] + 1e-7
    assert int8.persistent_state_bytes < fp32.persistent_state_bytes


def test_crossfitted_gate_accepts_centroid_when_base_is_constant() -> None:
    rows, labels = _support(class_count=2)
    int8, fp32, audit = fit_crossfitted_pair_reranker(
        rows, labels, 2, 4, _bad_base_fit
    )
    assert audit["gate_status"] == "crossfitted_top2_centroid_pairs_active"
    assert audit["partition_exact_once"] is True
    assert audit["accepted_pairs"] == [[0, 1]]
    base = np.column_stack([np.ones(len(rows)), np.zeros(len(rows))]).astype(np.float32)
    scores, changed = score_with_pair_state(base, rows, fp32)
    assert changed == 4
    assert np.mean(np.argmax(scores, axis=1) == labels) == 1.0
    assert int8.class_count == fp32.class_count == 2


def test_k1_is_exact_empty_pair_fallback() -> None:
    rows, labels = _support(class_count=2, k=1)
    int8, fp32, audit = fit_crossfitted_pair_reranker(
        rows, labels, 2, 1, _bad_base_fit
    )
    assert audit["gate_status"] == "k1_exact_d62_fallback"
    assert int8.pair_index.shape == fp32.pair_index.shape == (0, 2)
    base = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    scores, changed = score_with_pair_state(base, rows, int8)
    assert changed == 0
    assert np.array_equal(scores, base)


def test_invalid_asymmetric_support_fails_closed() -> None:
    rows, labels = _support(class_count=2)
    with pytest.raises(D71RerankerError):
        fit_crossfitted_pair_reranker(rows[:-1], labels[:-1], 2, 4, _bad_base_fit)
