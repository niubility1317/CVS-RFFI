from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import cvsrffi.stage2_d42_unified_shrinkage_lda as d42

from cvsrffi.stage2_d38_strong_b3_quantized import (
    D38StrongB3Config,
    fit_d38_strong_b3_quantized,
)
from cvsrffi.stage2_d42_unified_shrinkage_lda import (
    D42UnifiedShrinkageLDAConfig,
    D42UnifiedShrinkageLDAError,
    decode_d42_coefficients,
    fit_d42_unified_shrinkage_lda,
    pairwise_support_diagnostics_d42,
    predict_d42_unified_shrinkage_lda,
    score_d42_unified_shrinkage_lda,
)


def _support(
    classes: tuple[str, ...], *, k: int, seed: int, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, label in enumerate(classes):
        center = rng.normal(size=288).astype(np.float32)
        center[(17 * index + int(offset)) % 288] += np.float32(5.0 + offset)
        for _ in range(k):
            row = center + np.float32(0.20) * rng.normal(size=288).astype(
                np.float32
            )
            row /= np.linalg.norm(row)
            rows.append(row.astype(np.float32))
            labels.append(label)
    return np.stack(rows).astype(np.float32), np.asarray(labels)


def _fit(
    *, k: int = 5, old_count: int = 2, new_count: int = 2, seed: int = 19
):
    old_classes = tuple(f"old_{index}" for index in range(old_count))
    new_classes = tuple(f"new_{index}" for index in range(new_count))
    old_x, old_y = _support(old_classes, k=k, seed=seed, offset=1.0)
    new_x, new_y = _support(new_classes, k=k, seed=seed + 1, offset=41.0)
    result = fit_d42_unified_shrinkage_lda(
        old_x,
        old_y,
        old_classes,
        new_x,
        new_y,
        new_classes,
        seed=seed + 10,
        config=D42UnifiedShrinkageLDAConfig(),
    )
    return result, old_x, old_y, new_x, new_y


def _state_bytes(state) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(value).tobytes()
        for value in (
            state.log_diag_fp32,
            state.coef1_qint8,
            state.coef2_qint8,
            state.scale1_fp16,
            state.scale2_fp16,
            state.intercept_fp16,
            state.coef_fp32,
            state.intercept_fp32,
        )
    )


def test_metric_is_exact_d38_fullbatch_b20_before_trajectory() -> None:
    result, old_x, old_y, new_x, new_y = _fit(k=5)
    direct = fit_d38_strong_b3_quantized(
        old_x,
        old_y,
        ("old_0", "old_1"),
        new_x,
        new_y,
        ("new_0", "new_1"),
        seed=29,
        config=D38StrongB3Config(arm="A"),
    )

    assert np.array_equal(
        result.before_state.log_diag_fp32, direct.before_state.log_diag_fp32
    )
    assert result.training_trace == direct.training_trace
    assert len(result.training_trace) == 20
    assert [row["optimizer_step"] for row in result.training_trace] == list(
        range(1, 21)
    )
    assert result.resource_audit["metric_optimizer_steps"] == 20
    assert result.resource_audit["lda_optimizer_steps"] == 0
    assert result.resource_audit["optimizer_steps"] == 20


def test_old_only_metric_helper_has_no_new_surface_and_runs_before_new_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = tuple(inspect.signature(d42._fit_old_only_b3_metric).parameters)
    assert parameters == (
        "old_rows",
        "old_targets",
        "old_class_count",
        "seed",
        "device",
    )

    class PoisonNew:
        def __array__(self, *args, **kwargs):
            raise AssertionError("new support was read before old-only helper")

    def stop_at_old_helper(*args, **kwargs):
        raise RuntimeError("old-only helper reached first")

    monkeypatch.setattr(d42, "_fit_old_only_b3_metric", stop_at_old_helper)
    old_x, old_y = _support(("a", "b"), k=1, seed=301, offset=1.0)
    with pytest.raises(RuntimeError, match="old-only helper reached first"):
        d42.fit_d42_unified_shrinkage_lda(
            old_x,
            old_y,
            ("a", "b"),
            PoisonNew(),
            ("c", "d"),
            ("c", "d"),
            seed=1,
        )


def test_changing_new_support_cannot_change_before_artifact() -> None:
    old_x, old_y = _support(("a", "b"), k=5, seed=311, offset=1.0)
    first_new_x, first_new_y = _support(("c", "d"), k=5, seed=312, offset=21.0)
    second_new_x, second_new_y = _support(("c", "d"), k=5, seed=313, offset=61.0)
    first = fit_d42_unified_shrinkage_lda(
        old_x,
        old_y,
        ("a", "b"),
        first_new_x,
        first_new_y,
        ("c", "d"),
        seed=17,
    )
    second = fit_d42_unified_shrinkage_lda(
        old_x,
        old_y,
        ("a", "b"),
        second_new_x,
        second_new_y,
        ("c", "d"),
        seed=17,
    )

    assert first.before_state.classes == second.before_state.classes
    assert _state_bytes(first.before_state) == _state_bytes(second.before_state)
    assert _state_bytes(first.matched_fp32_before_state) == _state_bytes(
        second.matched_fp32_before_state
    )


def test_stage2b_old_only_then_stage2c_all_registry_with_frozen_metric() -> None:
    result, _, _, _, _ = _fit(k=5)

    assert result.before_state.classes == ("old_0", "old_1")
    assert result.state.classes == ("old_0", "old_1", "new_0", "new_1")
    assert np.array_equal(
        result.before_state.log_diag_fp32, result.state.log_diag_fp32
    )
    assert result.geometry_audit["metric_frozen_during_stage2c"] is True
    assert result.geometry_audit["stage2b_classifier"].startswith("old_only")
    assert result.geometry_audit["stage2c_classifier"].startswith("all_registry")
    assert result.geometry_audit["sklearn_runtime_version"] == "1.7.2"
    assert result.resource_audit["sklearn_runtime_version_lock_pass"] is True


def test_sklearn_runtime_version_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked = D42UnifiedShrinkageLDAConfig()
    old_x, old_y = _support(("a", "b"), k=1, seed=401, offset=1.0)
    new_x, new_y = _support(("c", "d"), k=1, seed=402, offset=21.0)
    monkeypatch.setattr(d42.sklearn, "__version__", "1.7.3")

    with pytest.raises(D42UnifiedShrinkageLDAError, match="runtime version drift"):
        D42UnifiedShrinkageLDAConfig()
    with pytest.raises(D42UnifiedShrinkageLDAError, match="runtime version drift"):
        fit_d42_unified_shrinkage_lda(
            old_x,
            old_y,
            ("a", "b"),
            new_x,
            new_y,
            ("c", "d"),
            seed=1,
            config=locked,
        )


@pytest.mark.parametrize("class_count", [2, 4])
def test_all_class_counts_use_explicit_sigma_inverse_mu_formula(
    class_count: int,
) -> None:
    classes = tuple(f"class_{index}" for index in range(class_count))
    rows, labels = _support(classes, k=5, seed=410 + class_count, offset=3.0)
    targets = np.asarray([classes.index(value) for value in labels], dtype=np.int64)
    coefficients, intercept, audit = d42._fit_equal_prior_lda(
        rows, targets, class_count, 5
    )
    priors = np.full(class_count, 1.0 / class_count, dtype=np.float64)
    estimator = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", priors=priors, store_covariance=True
    ).fit(rows.astype(np.float64), targets)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    means = np.asarray(estimator.means_, dtype=np.float64)
    expected_coefficients = np.linalg.lstsq(
        covariance, means.T, rcond=None
    )[0].T
    expected_intercept = (
        -0.5 * np.diag(means @ expected_coefficients.T) + np.log(priors)
    )
    deployed_predictions = np.argmax(
        rows.astype(np.float64) @ coefficients.astype(np.float64).T
        + intercept.astype(np.float64)[None, :],
        axis=1,
    )

    assert coefficients.shape == (class_count, 288)
    assert np.allclose(
        coefficients, expected_coefficients.astype(np.float32), rtol=0.0, atol=0.0
    )
    assert np.allclose(
        intercept, expected_intercept.astype(np.float32), rtol=0.0, atol=0.0
    )
    assert np.allclose(
        covariance @ coefficients.astype(np.float64).T,
        means.T,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert np.array_equal(
        deployed_predictions,
        np.asarray(estimator.predict(rows.astype(np.float64)), dtype=np.int64),
    )
    assert (
        audit["coefficient_source"]
        == "locked_sklearn_covariance_means_explicit_lstsq_sigma_inverse_mu"
    )
    assert audit["sklearn_prediction_equivalent"] is True


def test_before_formal_and_fp32_states_are_immutable() -> None:
    result, old_x, _, _, _ = _fit(k=5)
    formal = _state_bytes(result.before_state)
    matched = _state_bytes(result.matched_fp32_before_state)

    _ = score_d42_unified_shrinkage_lda(result.state, old_x)
    assert _state_bytes(result.before_state) == formal
    assert _state_bytes(result.matched_fp32_before_state) == matched
    assert result.geometry_audit["before_state_immutable_during_stage2c"] is True
    assert result.geometry_audit["old_only_metric_helper_called_once"] is True
    assert result.geometry_audit["old_only_metric_new_support_argument_count"] == 0
    assert result.geometry_audit["before_materialized_pre_stage2c"] is True
    assert result.geometry_audit["before_materialization_optimizer_step"] == 20
    assert result.geometry_audit["before_covariance_audit"]["support_rows"] == len(
        old_x
    )


def test_formal_storage_is_residual_int8_and_fp16_intercept_only() -> None:
    result, _, _, _, _ = _fit(k=5)
    state = result.state

    assert state.is_int8
    assert state.coef1_qint8.dtype == np.int8
    assert state.coef2_qint8.dtype == np.int8
    assert state.coef1_qint8.shape == (4, 288)
    assert state.scale1_fp16.dtype == np.float16
    assert state.scale1_fp16.shape == (4, 3)
    assert state.scale2_fp16.dtype == np.float16
    assert state.intercept_fp16.dtype == np.float16
    assert np.isfinite(state.intercept_fp16).all()
    assert state.coef_fp32.shape == (0, 288)
    assert state.intercept_fp32.shape == (0,)
    assert all(not value.flags.writeable for value in (
        state.log_diag_fp32,
        state.coef1_qint8,
        state.coef2_qint8,
        state.scale1_fp16,
        state.scale2_fp16,
        state.intercept_fp16,
    ))
    assert (
        result.resource_audit["formal_target_vectors_int8_no_fp32_sidecar"]
        is True
    )
    assert result.resource_audit["resident_fp32_target_coefficient_count"] == 0
    assert result.geometry_audit["formal_old_target_vectors_residual_int8"] is True
    assert result.geometry_audit["formal_new_target_vectors_residual_int8"] is True
    assert result.geometry_audit["class_means_persisted_in_formal_state"] is False
    assert (
        result.geometry_audit["shared_covariance_persisted_in_formal_state"]
        is False
    )


def test_matched_fp32_ablation_and_quantization_audit() -> None:
    result, old_x, _, new_x, _ = _fit(k=5)
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    int8_scores = score_d42_unified_shrinkage_lda(result.state, rows)
    fp32_scores = score_d42_unified_shrinkage_lda(
        result.matched_fp32_state, rows
    )

    assert not result.matched_fp32_state.is_int8
    assert result.matched_fp32_state.coef_fp32.shape == (4, 288)
    assert decode_d42_coefficients(result.state).shape == (4, 288)
    assert np.isfinite(int8_scores).all() and np.isfinite(fp32_scores).all()
    assert result.geometry_audit["final_support_score_max_abs_error"] >= 0.0
    assert (
        result.geometry_audit[
            "int8_vs_fp32_final_support_argmax_change_count"
        ]
        >= 0
    )


def test_k1_uses_unit_covariance_without_synthetic_samples() -> None:
    result, old_x, _, new_x, _ = _fit(k=1)
    rows = np.concatenate([old_x, new_x]).astype(np.float32)

    assert result.geometry_audit["k1_unit_covariance_fallback"] is True
    assert (
        result.geometry_audit["before_covariance_audit"][
            "covariance_policy"
        ]
        == "unit_covariance_equal_prior_nearest_centroid"
    )
    assert result.geometry_audit["before_covariance_audit"]["support_rows"] == 2
    assert result.geometry_audit["final_covariance_audit"]["support_rows"] == 4
    assert score_d42_unified_shrinkage_lda(result.state, rows).shape == (4, 4)


@pytest.mark.parametrize("k_shot", [1, 5, 10, 20])
def test_k_shot_closure(k_shot: int) -> None:
    result, _, _, _, _ = _fit(k=k_shot)

    assert result.resource_audit["old_k_shot"] == k_shot
    assert result.resource_audit["new_k_shot"] == k_shot
    assert result.resource_audit["optimizer_steps"] == 20
    assert len(result.training_trace) == 20


@pytest.mark.parametrize("new_count", [2, 5, 10, 20])
def test_new_class_count_and_resource_closure(new_count: int) -> None:
    result, _, _, _, _ = _fit(k=1, new_count=new_count)

    assert len(result.state.classes) == 2 + new_count
    assert result.state.persistent_state_bytes <= 256 * 1024
    assert result.resource_audit["persistent_state_cap_pass"] is True
    assert result.resource_audit["trainable_parameter_cap_pass"] is True


def test_actual_registry_permutation_and_label_renaming_are_equivariant() -> None:
    old_x, old_y = _support(("old_a", "old_b"), k=5, seed=101, offset=1.0)
    new_x, new_y = _support(("new_a", "new_b"), k=5, seed=102, offset=31.0)
    first = fit_d42_unified_shrinkage_lda(
        old_x,
        old_y,
        ("old_a", "old_b"),
        new_x,
        new_y,
        ("new_a", "new_b"),
        seed=111,
    )
    mapping = {"old_a": "x", "old_b": "y", "new_a": "u", "new_b": "v"}
    second = fit_d42_unified_shrinkage_lda(
        old_x,
        np.asarray([mapping[value] for value in old_y]),
        ("y", "x"),
        new_x,
        np.asarray([mapping[value] for value in new_y]),
        ("v", "u"),
        seed=111,
    )
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    columns = [
        second.state.classes.index(mapping[value]) for value in first.state.classes
    ]

    assert second.state.classes == ("y", "x", "v", "u")
    assert np.allclose(
        score_d42_unified_shrinkage_lda(first.state, rows),
        score_d42_unified_shrinkage_lda(second.state, rows)[:, columns],
        rtol=1.0e-6,
        atol=2.0e-4,
    )
    assert np.array_equal(
        predict_d42_unified_shrinkage_lda(second.state, rows),
        np.asarray(
            [
                mapping[value]
                for value in predict_d42_unified_shrinkage_lda(first.state, rows)
            ]
        ),
    )


def test_score_is_row_local_batch_order_independent_and_all_registry() -> None:
    result, old_x, _, new_x, _ = _fit(k=5)
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    whole = score_d42_unified_shrinkage_lda(result.state, rows)
    split = np.concatenate(
        [score_d42_unified_shrinkage_lda(result.state, row[None, :]) for row in rows]
    )
    permutation = np.random.default_rng(7).permutation(len(rows))

    assert whole.shape == (len(rows), len(result.state.classes))
    assert np.array_equal(whole, split)
    assert np.array_equal(
        score_d42_unified_shrinkage_lda(result.state, rows[permutation]),
        whole[permutation],
    )
    assert result.resource_audit["query_rows_used_for_fit"] == 0
    assert result.resource_audit["query_role_oracle_access"] is False
    assert result.resource_audit["query_class_quota_access"] is False
    assert result.resource_audit["query_batch_global_assignment"] is False


def test_pairwise_diagnostics_separate_three_margin_error_types() -> None:
    result, old_x, old_y, new_x, new_y = _fit(k=5)
    rows = np.concatenate([old_x, new_x]).astype(np.float32)
    labels = np.concatenate([old_y, new_y])
    diagnostics = pairwise_support_diagnostics_d42(
        result.state,
        rows,
        labels,
        [f"physical-{index}" for index in range(len(rows))],
        scenario="leo_clear_weak",
        outer_fold=0,
        physical_ranks=list(range(len(rows))),
    )

    assert len(diagnostics) == len(rows)
    assert all(row["query_rows_used"] == 0 for row in diagnostics)
    old_rows = [row for row in diagnostics if row["true_role"] == "old"]
    new_rows = [row for row in diagnostics if row["true_role"] == "new"]
    assert all(row["old_to_new_margin"] is not None for row in old_rows)
    assert all(row["new_to_old_margin"] is None for row in old_rows)
    assert all(row["new_to_old_margin"] is not None for row in new_rows)
    assert all(row["new_new_margin"] is not None for row in new_rows)


def test_ground_query_and_forbidden_access_audits_are_closed() -> None:
    result, _, _, _, _ = _fit(k=1)
    geometry = result.geometry_audit
    resource = result.resource_audit

    assert geometry["ground_int8_component_input_count"] == 0
    assert geometry["ground_int8_update_access"] is False
    assert resource["ground_int8_component_input_count"] == 0
    assert resource["ground_int8_update_access"] is False
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False
    assert resource["dense_query_graph_bytes"] == 0


def test_invalid_config_nonfinite_and_asymmetric_k_fail_closed() -> None:
    with pytest.raises(D42UnifiedShrinkageLDAError):
        D42UnifiedShrinkageLDAConfig(shrinkage="manual")
    old_x, old_y = _support(("a", "b"), k=1, seed=201, offset=1.0)
    new_x, new_y = _support(("c", "d"), k=1, seed=202, offset=21.0)
    damaged = old_x.copy()
    damaged[0, 0] = np.nan
    with pytest.raises(D42UnifiedShrinkageLDAError):
        fit_d42_unified_shrinkage_lda(
            damaged, old_y, ("a", "b"), new_x, new_y, ("c", "d"), seed=1
        )
    with pytest.raises(D42UnifiedShrinkageLDAError, match="symmetric K-shot"):
        fit_d42_unified_shrinkage_lda(
            old_x,
            old_y,
            ("a", "b"),
            np.concatenate([new_x, new_x[:1]]).astype(np.float32),
            np.concatenate([new_y, new_y[:1]]),
            ("c", "d"),
            seed=1,
        )
