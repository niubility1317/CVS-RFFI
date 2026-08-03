from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d127_d92_lite as d127


def _support(
    classes: int, shots: int, *, seed: int = 127
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    registry = tuple(f"tx_{index:02d}" for index in range(classes))
    labels = tuple(label for label in registry for _ in range(shots))
    centers = rng.normal(size=(classes, d127.Z_DIM)).astype(np.float32)
    rows = np.vstack(
        [
            centers[index]
            + np.float32(0.07) * rng.normal(size=(shots, d127.Z_DIM)).astype(np.float32)
            for index in range(classes)
        ]
    ).astype(np.float32)
    return rows, labels, registry


@pytest.mark.parametrize("class_count", [11, 16, 26])
def test_active_state_is_exact_int8_fp16_readonly_164c_layout(class_count: int):
    rows, labels, registry = _support(class_count, 5, seed=class_count)
    fitted = d127.fit_d92_lite(rows, labels, registry)
    state = fitted.state

    assert type(state) is d127.D92LiteQuantizedLDAState
    assert state.q_int8.shape == (class_count, 160)
    assert state.q_int8.dtype == np.int8
    assert state.scale_fp16.shape == (class_count,)
    assert state.scale_fp16.dtype == np.float16
    assert state.intercept_fp16.shape == (class_count,)
    assert state.intercept_fp16.dtype == np.float16
    assert not state.q_int8.flags.writeable
    assert not state.scale_fp16.flags.writeable
    assert not state.intercept_fp16.flags.writeable
    assert np.all(state.q_int8 >= -127)
    assert np.all(state.q_int8 <= 127)
    assert np.all(state.scale_fp16 > 0)
    assert state.numeric_state_bytes == 164 * class_count
    assert fitted.resource_receipt["deployed_numeric_state_bytes"] == 164 * class_count
    assert fitted.resource_receipt["d92_lite_incremental_state_formula"] == "160C+2C+2C=164C_B"
    ndarray_fields = [
        value
        for value in (state.q_int8, state.scale_fp16, state.intercept_fp16)
        if isinstance(value, np.ndarray)
    ]
    assert [value.dtype for value in ndarray_fields] == [
        np.dtype(np.int8),
        np.dtype(np.float16),
        np.dtype(np.float16),
    ]


def test_k1_is_strict_qknn_alias_without_diagonal_fit(monkeypatch: pytest.MonkeyPatch):
    rows, labels, registry = _support(11, 1)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("K1 must not compute D92-Lite diagonal statistics")

    monkeypatch.setattr(d127, "_compile_diagonal_oas_state", _must_not_run)
    fitted = d127.fit_d92_lite(rows, labels, registry)
    assert type(fitted.state) is d127.D92LiteQKNNAlias
    assert fitted.fit_receipt["fit_mode"] == "exact_qknn_alias"
    assert fitted.fit_receipt["diagonal_statistics_computed"] is False
    assert fitted.state.numeric_state_bytes == 0
    resource = fitted.resource_receipt
    assert resource["d92_lite_incremental_deployed_numeric_state_bytes"] == 0
    assert resource["d92_lite_incremental_query_macs_per_sample"] == 0
    assert resource["d92_lite_incremental_query_state_bytes"] == 0
    assert resource["underlying_qknn_resource_required"] is True
    assert resource["underlying_qknn_resource_included"] is False
    assert (
        resource["underlying_qknn_resource_receipt_binding"]
        == "caller_formal_receipt_required"
    )
    assert "deployed_numeric_state_bytes" not in resource
    assert "query_head_macs_per_sample" not in resource
    assert "query_state_bytes" not in resource

    query = np.arange(3 * d127.Z_DIM, dtype=np.float32).reshape(3, d127.Z_DIM)
    query_ids = ("query_a", "query_b", "query_c")
    qknn_logits = np.arange(33, dtype=np.float32).reshape(3, 11)
    alias_receipt = d127.make_qknn_alias_receipt(
        classes=registry,
        query_zid=query,
        opaque_query_ids=query_ids,
        qknn_logits=qknn_logits,
    )
    logits = d127.score_d92_lite_logits(
        fitted.state,
        query,
        qknn_logits=qknn_logits,
        qknn_alias_receipt=alias_receipt,
    )
    assert logits is qknn_logits
    score = d127.score_d92_lite(
        fitted.state,
        query,
        qknn_logits=qknn_logits,
        qknn_alias_receipt=alias_receipt,
    )
    assert score.logits is qknn_logits
    assert score.score_receipt["query_state_updates"] == 0

    wrong_classes = d127.make_qknn_alias_receipt(
        classes=tuple(reversed(registry)),
        query_zid=query,
        opaque_query_ids=query_ids,
        qknn_logits=qknn_logits,
    )
    with pytest.raises(d127.D92LiteError, match="class-column order"):
        d127.score_d92_lite_logits(
            fitted.state,
            query,
            qknn_logits=qknn_logits,
            qknn_alias_receipt=wrong_classes,
        )
    with pytest.raises(d127.D92LiteError, match="identity/order binding"):
        d127.score_d92_lite_logits(
            fitted.state,
            query[[2, 0, 1]],
            qknn_logits=qknn_logits,
            qknn_alias_receipt=alias_receipt,
        )
    with pytest.raises(d127.D92LiteError, match="logits binding"):
        d127.score_d92_lite_logits(
            fitted.state,
            query,
            qknn_logits=qknn_logits[[2, 0, 1]],
            qknn_alias_receipt=alias_receipt,
        )


def test_k5_is_an_active_head_not_an_alias():
    rows, labels, registry = _support(11, 5)
    fitted = d127.fit_d92_lite(rows, labels, registry)
    query = rows[:2].copy()

    assert type(fitted.state) is d127.D92LiteQuantizedLDAState
    assert fitted.fit_receipt["fit_mode"] == "diagonal_oas_form"
    logits = d127.score_d92_lite_logits(fitted.state, query)
    assert logits.shape == (2, 11)
    with pytest.raises(d127.D92LiteError, match="cannot accept qKNN alias"):
        d127.score_d92_lite_logits(
            fitted.state,
            query,
            qknn_logits=np.zeros((2, 11), dtype=np.float32),
        )


def test_diagonal_oas_matches_an_independent_n_eff_reference():
    rows, labels, registry = _support(3, 5, seed=610)
    fitted = d127.fit_d92_lite(rows, labels, registry)
    state = fitted.state
    assert type(state) is d127.D92LiteQuantizedLDAState

    # Deliberately spell out the frozen scalar formula here rather than use a
    # module-private helper: n_eff is pooled class-within residual freedom.
    raw = rows.astype(np.float64)
    normalized = raw / np.sqrt(np.sum(raw * raw, axis=1, keepdims=True))
    grouped = normalized.reshape(3, 5, d127.Z_DIM)
    means = grouped.mean(axis=1)
    residuals = grouped - means[:, None, :]
    n_eff = 3 * (5 - 1)
    scatter = np.sum(residuals * residuals, axis=(0, 1)) / float(n_eff)
    total = float(np.sum(scatter))
    second_moment = float(np.sum(scatter * scatter))
    tau = total / d127.Z_DIM
    delta = second_moment - total * total / d127.Z_DIM
    if delta <= 0.0:
        shrinkage = 1.0
    else:
        shrinkage = min(
            1.0,
            ((1.0 - 2.0 / d127.Z_DIM) * second_moment + total * total)
            / ((n_eff + 1.0 - 2.0 / d127.Z_DIM) * delta),
        )
    variance_floor = max(
        float(np.finfo(np.float64).tiny),
        float(np.finfo(np.float64).eps) * max(1.0, tau),
    )
    variance = np.maximum(
        (1.0 - shrinkage) * scatter + shrinkage * tau,
        variance_floor,
    )
    expected_weights = means / variance[None, :]
    expected_intercepts = -0.5 * np.sum(
        means * means / variance[None, :], axis=1
    )
    expected_weights -= expected_weights.mean(axis=0, keepdims=True)
    expected_intercepts -= expected_intercepts.mean()
    expected_scales = np.maximum(
        np.max(np.abs(expected_weights), axis=1) / 127.0,
        float(np.finfo(np.float16).tiny),
    ).astype(np.float16)
    expected_codes = np.clip(
        np.rint(expected_weights / expected_scales.astype(np.float64)[:, None]),
        -127.0,
        127.0,
    ).astype(np.int8)

    assert fitted.fit_receipt["residual_degrees_of_freedom"] == n_eff
    assert fitted.fit_receipt["shrinkage"] == pytest.approx(shrinkage)
    assert fitted.fit_receipt["variance_floor"] == variance_floor
    np.testing.assert_array_equal(state.q_int8, expected_codes)
    np.testing.assert_array_equal(state.scale_fp16, expected_scales)
    np.testing.assert_array_equal(
        state.intercept_fp16, expected_intercepts.astype(np.float16)
    )


def test_class_label_permutation_is_equivariant():
    rows, labels, registry = _support(16, 10, seed=116)
    query = rows[:7].copy()
    fitted = d127.fit_d92_lite(rows, labels, registry)
    permutation = np.asarray([4, 9, 2, 1, 5, 11, 6, 14, 0, 7, 12, 3, 15, 8, 10, 13])
    permuted_registry = tuple(registry[index] for index in permutation)
    permuted = d127.fit_d92_lite(rows, labels, permuted_registry)

    state = fitted.state
    state_permuted = permuted.state
    assert type(state) is d127.D92LiteQuantizedLDAState
    assert type(state_permuted) is d127.D92LiteQuantizedLDAState
    np.testing.assert_array_equal(state_permuted.q_int8, state.q_int8[permutation])
    np.testing.assert_array_equal(state_permuted.scale_fp16, state.scale_fp16[permutation])
    np.testing.assert_array_equal(
        state_permuted.intercept_fp16, state.intercept_fp16[permutation]
    )
    logits = d127.score_d92_lite_logits(state, query)
    permuted_logits = d127.score_d92_lite_logits(state_permuted, query)
    np.testing.assert_array_equal(permuted_logits, logits[:, permutation])


def test_query_scoring_has_zero_state_change():
    rows, labels, registry = _support(11, 5, seed=901)
    fitted = d127.fit_d92_lite(rows, labels, registry)
    state = fitted.state
    assert type(state) is d127.D92LiteQuantizedLDAState
    before = (
        state.state_receipt_sha256,
        state.q_int8.tobytes(),
        state.scale_fp16.tobytes(),
        state.intercept_fp16.tobytes(),
    )
    score = d127.score_d92_lite(state, rows[:4])
    after = (
        state.state_receipt_sha256,
        state.q_int8.tobytes(),
        state.scale_fp16.tobytes(),
        state.intercept_fp16.tobytes(),
    )
    assert score.logits.shape == (4, 11)
    assert before == after
    assert score.score_receipt["query_state_updates"] == 0
    assert score.score_receipt["query_selection_count"] == 0
    assert score.score_receipt["query_batch_dependency"] is False


def test_active_queries_are_consistent_per_row_batch_chunk_and_reorder():
    rows, labels, registry = _support(11, 5, seed=905)
    state = d127.fit_d92_lite(rows, labels, registry).state
    assert type(state) is d127.D92LiteQuantizedLDAState
    query = rows[:7].copy()
    batch = d127.score_d92_lite_logits(state, query)
    per_row = np.vstack(
        [d127.score_d92_lite_logits(state, query[index : index + 1]) for index in range(7)]
    )
    chunked = np.vstack(
        [
            d127.score_d92_lite_logits(state, query[:2]),
            d127.score_d92_lite_logits(state, query[2:5]),
            d127.score_d92_lite_logits(state, query[5:]),
        ]
    )
    permutation = np.asarray([5, 1, 6, 0, 4, 2, 3])
    reordered = d127.score_d92_lite_logits(state, query[permutation])
    np.testing.assert_array_equal(per_row, batch)
    np.testing.assert_array_equal(chunked, batch)
    np.testing.assert_array_equal(reordered, batch[permutation])


def test_zero_total_scatter_is_p0_error_not_a_unit_fallback():
    registry = ("tx_a", "tx_b")
    labels = tuple(label for label in registry for _ in range(5))
    rows = np.vstack(
        [
            np.tile(np.eye(1, d127.Z_DIM, 0, dtype=np.float32), (5, 1)),
            np.tile(np.eye(1, d127.Z_DIM, 1, dtype=np.float32), (5, 1)),
        ]
    )
    with pytest.raises(d127.D92LiteError, match="t<=0"):
        d127.fit_d92_lite(rows, labels, registry)


def test_protocol_and_no_dense_linear_algebra_paths_are_auditable():
    fit_signature = inspect.signature(d127.fit_d92_lite)
    score_signature = inspect.signature(d127.score_d92_lite_logits)
    forbidden = {"query_truth", "truth", "quota", "old", "new", "role"}
    assert not forbidden.intersection(fit_signature.parameters)
    assert not forbidden.intersection(score_signature.parameters)

    resource = d127.d92_lite_resource_receipt(
        d127.fit_d92_lite(*_support(11, 5)).state
    )
    assert resource["dense_matrix_elements_constructed"] == 0
    assert resource["spectral_factorization_count"] == 0
    assert resource["linear_system_solve_count"] == 0

    source = inspect.getsource(d127)
    assert "np.linalg.eig" not in source
    assert "np.linalg.solve" not in source
    assert "np.cov(" not in source
