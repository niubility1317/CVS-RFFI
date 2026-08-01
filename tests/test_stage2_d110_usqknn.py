"""Mechanical regression tests for the D110 US-qKNN four-arm core."""

from __future__ import annotations

import inspect
import math

import numpy as np

from cvsrffi import stage2_d110_usqknn as usqknn
from cvsrffi import stage2_zid_student_t_qknn as qknn


CLASSES = ("c0", "c1", "c2")


def _lock(k_shot: int) -> qknn.Phase1ZIDStudentTLock:
    return qknn.Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _basis() -> np.ndarray:
    return np.eye(3, 160, dtype=np.float64)


def _rows(k_shot: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(CLASSES), 160)).astype(np.float32)
    centers[:, :3] += np.asarray(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    support = []
    labels: list[str] = []
    for index, class_id in enumerate(CLASSES):
        local = centers[index] + 0.06 * rng.normal(size=(k_shot, 160))
        support.append(local.astype(np.float32))
        labels.extend([class_id] * k_shot)
    query = (centers + 0.07 * rng.normal(size=(len(CLASSES), 160))).astype(np.float32)
    return np.concatenate(support), np.asarray(labels, dtype=str), query


def _states(k_shot: int, *, seed: int = 110):
    support, labels, query = _rows(k_shot, seed=seed)
    states = usqknn.fit_d110_usqknn_four_arms(
        support,
        labels,
        CLASSES,
        config=_lock(k_shot),
        closed_u=_basis(),
        prior_variances=np.asarray([0.25, 4.0, 12.0, 0.8], dtype=np.float64),
    )
    return states, support, labels, query


def _manual_relative_distance(
    left: np.ndarray, right: np.ndarray, relative: np.ndarray
) -> np.ndarray:
    basis = _basis()
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    projected = (
        (left64 @ basis.T)[:, None, :] - (right64 @ basis.T)[None, :, :]
    )
    parallel = np.sum(np.square(projected), axis=2)
    total = (
        np.sum(np.square(left64), axis=1)[:, None]
        + np.sum(np.square(right64), axis=1)[None, :]
        - 2.0 * (left64 @ right64.T)
    )
    return (
        np.sum(np.square(projected) / relative[None, None, :-1], axis=2)
        + np.maximum(total - parallel, 0.0) / relative[-1]
    )


def _manual_student_t(
    distances: np.ndarray,
    bank: qknn.TypedINT8ZIDSupportBank,
    scales: np.ndarray,
) -> np.ndarray:
    config = bank.config
    columns = []
    for class_index, count in enumerate(bank.support_counts):
        local = distances[:, bank.class_indices_int16 == class_index]
        h = float(scales[class_index])
        kernel = (
            -config.kernel_volume_gamma * config.kernel_effective_dim * math.log(h)
            - 0.5
            * (config.student_nu + config.kernel_effective_dim)
            * np.log1p(local / (config.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(
            maximum[:, 0]
            + np.log(np.sum(np.exp(kernel - maximum), axis=1))
            - math.log(count)
        )
    return np.stack(columns, axis=1).astype(np.float32)


def test_m0_bit_matches_existing_qknn_and_k1_head_effect_is_identity() -> None:
    states, support, labels, query = _states(1)
    lock = _lock(1)
    bank = qknn.build_typed_zid_support_bank(support, labels, CLASSES, config=lock)
    expected = qknn.score_zid_student_t_logits(
        bank, query, metric=qknn.identity_shared_psd_metric(config=lock)
    )

    actual = usqknn.score_d110_usqknn_logits(states["M0"], query)
    head = usqknn.score_d110_usqknn_logits(states["M_HEAD"], query)
    da = usqknn.score_d110_usqknn_logits(states["M_DA"], query)
    joint = usqknn.score_d110_usqknn_logits(states["M_JOINT"], query)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(head, actual)
    np.testing.assert_array_equal(joint, da)
    np.testing.assert_array_equal(
        states["M0"].class_scales_fp16, states["M_HEAD"].class_scales_fp16
    )
    np.testing.assert_array_equal(
        states["M_DA"].class_scales_fp16,
        states["M_JOINT"].class_scales_fp16,
    )


def test_da_pair_metric_uses_safe_relative_variances_not_predictive_factor() -> None:
    states, _, _, query = _states(5)
    state = states["M_DA"]
    assert state.scpm_state is not None
    support = qknn.decode_zid_support_bank(state.bank).astype(np.float64)
    normalized_query = qknn.normalize_zid_rows(query).astype(np.float64)
    relative = state.scpm_state.safe_relative_variances
    predictive = state.scpm_state.predictive_variances
    assert not np.array_equal(relative, predictive)

    expected_distance = _manual_relative_distance(
        normalized_query, support, relative
    )
    expected = _manual_student_t(
        expected_distance, state.bank, state.class_scales_fp16
    )
    actual = usqknn.score_d110_usqknn_logits(state, query)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-6)
    predictive_distance = _manual_relative_distance(
        normalized_query, support, predictive
    )
    predictive_logits = _manual_student_t(
        predictive_distance, state.bank, state.class_scales_fp16
    )
    assert not np.allclose(actual, predictive_logits, rtol=0.0, atol=1.0e-7)


def test_shared_head_is_equal_class_mean_then_the_same_locked_shrink_clip() -> None:
    states, _, _, _ = _states(5)
    state = states["M_HEAD"]
    support = qknn.decode_zid_support_bank(state.bank).astype(np.float64)
    distances = np.maximum(2.0 * (1.0 - support @ support.T), 0.0)
    energies = []
    for class_index, k_shot in enumerate(state.bank.support_counts):
        local = distances[
            np.ix_(
                state.bank.class_indices_int16 == class_index,
                state.bank.class_indices_int16 == class_index,
            )
        ]
        energies.append(float(np.mean(local[np.triu_indices(k_shot, 1)])))
    config = state.bank.config
    shared = float(np.mean(energies))
    expected = np.clip(
        math.sqrt(
            (shared + config.scale_prior_strength * config.shared_h0**2)
            / (1.0 + config.scale_prior_strength)
        ),
        config.shared_h0 * config.scale_min_ratio,
        config.shared_h0 * config.scale_max_ratio,
    )
    np.testing.assert_array_equal(
        state.class_scales_fp16,
        np.full(len(CLASSES), expected, dtype=np.float16),
    )


def test_query_scoring_is_stateless_all_class_and_permutation_equivariant() -> None:
    states, support, labels, query = _states(5)
    original = usqknn.score_d110_usqknn_logits(states["M_JOINT"], query)
    before = (
        states["M_JOINT"].bank.codes_qint8.copy(),
        states["M_JOINT"].class_scales_fp16.copy(),
        states["M_JOINT"].scpm_state.variances.copy(),  # type: ignore[union-attr]
    )
    chunked = np.concatenate(
        (
            usqknn.score_d110_usqknn_logits(states["M_JOINT"], query[:1]),
            usqknn.score_d110_usqknn_logits(states["M_JOINT"], query[1:]),
        ),
        axis=0,
    )
    np.testing.assert_allclose(chunked, original, rtol=0.0, atol=2.0e-6)
    np.testing.assert_array_equal(states["M_JOINT"].bank.codes_qint8, before[0])
    np.testing.assert_array_equal(states["M_JOINT"].class_scales_fp16, before[1])
    np.testing.assert_array_equal(
        states["M_JOINT"].scpm_state.variances, before[2]  # type: ignore[union-attr]
    )

    registry = ("c2", "c0", "c1")
    permutation = np.asarray([7, 0, 9, 2, 11, 4, 13, 1, 5, 6, 8, 10, 3, 12, 14])
    reordered = usqknn.fit_d110_usqknn_four_arms(
        support[permutation],
        labels[permutation],
        registry,
        config=_lock(5),
        closed_u=_basis(),
        prior_variances=np.asarray([0.25, 4.0, 12.0, 0.8], dtype=np.float64),
    )
    permuted_logits = usqknn.score_d110_usqknn_logits(reordered["M_JOINT"], query)
    columns = [CLASSES.index(class_id) for class_id in registry]
    np.testing.assert_allclose(permuted_logits, original[:, columns], rtol=0.0, atol=2.0e-6)


def test_interface_is_truth_free_and_audit_is_state_only() -> None:
    states, _, _, query = _states(10)
    signature = inspect.signature(usqknn.fit_d110_usqknn_four_arms)
    forbidden = {"query", "truth", "role", "quota", "receiver"}
    assert not forbidden.intersection(signature.parameters)
    predictions = usqknn.predict_d110_usqknn(states["M0"], query)
    assert len(predictions) == len(query)
    assert set(predictions).issubset(set(CLASSES))
    audit = usqknn.audit_d110_usqknn_state(states["M_JOINT"])
    assert audit["active_k"] == 10
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_state_updates"] == 0
    assert audit["predictive_variance_used"] is False
    assert audit["parameter_scan_count"] == 0
    assert audit["support_bank_numeric_bytes"] > 0
    assert audit["metric_numeric_bytes"] > 0
