from __future__ import annotations

import numpy as np
import pytest

from paper_reproduction.cvs_aligned.extreme_light_adapter import (
    concatenate_registered_features,
    fit_predict_extreme_light_diag_cosine,
    fit_predict_extreme_light_low_rank_cosine,
    fit_predict_support_ridge,
    predict_support_prototype_cosine,
)


def _separable(seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = np.eye(4, 16, dtype=np.float32)
    support = np.vstack([centers[index] + 0.02 * rng.normal(size=(8, 16)) for index in range(4)])
    query = np.vstack([centers[index] + 0.02 * rng.normal(size=(5, 16)) for index in range(4)])
    support_y = np.repeat(["old-a", "old-b", "new-a", "new-b"], 8)
    query_y = np.repeat(["old-a", "old-b", "new-a", "new-b"], 5)
    return support.astype(np.float32), support_y, query.astype(np.float32), query_y


def test_extreme_light_adapter_is_support_only_and_resource_bounded():
    support, support_y, query, query_y = _separable()
    predicted, info, trace = fit_predict_extreme_light_diag_cosine(
        support,
        support_y,
        query,
        seed=17,
        epochs=5,
        device="cpu",
    )
    assert np.mean(predicted.astype(str) == query_y.astype(str)) == 1.0
    assert info["query_labels_used_for_adaptation"] is False
    assert info["query_features_used_for_adaptation"] is False
    assert info["query_query_graph_used"] is False
    assert info["role_oracle_used"] is False
    assert info["equal_class_quota_used"] is False
    assert info["trainable_parameters"] <= 50_000
    assert info["persistent_state_bytes"] <= 128 * 1024
    assert len(trace) == 5
    assert {"total_loss", "ce_loss", "source_anchor_loss", "learning_rate", "gradient_norm", "support_accuracy"} <= set(trace[0])


def test_query_batch_extension_does_not_change_existing_predictions():
    support, support_y, query, _ = _separable()
    first, _, _ = fit_predict_extreme_light_diag_cosine(
        support, support_y, query, seed=23, epochs=3, device="cpu"
    )
    extended_query = np.vstack([query, np.full((11, query.shape[1]), 50.0, dtype=np.float32)])
    extended, _, _ = fit_predict_extreme_light_diag_cosine(
        support, support_y, extended_query, seed=23, epochs=3, device="cpu"
    )
    assert np.array_equal(first, extended[: len(first)])


def test_closed_form_prototype_head_is_zero_train_and_query_independent():
    support, support_y, query, query_y = _separable()
    predicted, info, trace = predict_support_prototype_cosine(support, support_y, query)
    extended, _, _ = predict_support_prototype_cosine(
        support,
        support_y,
        np.vstack([query, np.full((7, query.shape[1]), 9.0, dtype=np.float32)]),
    )
    assert np.mean(predicted.astype(str) == query_y.astype(str)) == 1.0
    assert np.array_equal(predicted, extended[: len(predicted)])
    assert info["trainable_parameters"] == 0
    assert info["feature_adapter_gradient_updates"] == 0
    assert info["query_features_used_for_adaptation"] is False
    assert len(trace) == 1


def test_closed_form_ridge_head_is_zero_train_and_query_independent():
    support, support_y, query, query_y = _separable()
    predicted, info, trace = fit_predict_support_ridge(
        support, support_y, query, ridge_lambda=0.1
    )
    extended, _, _ = fit_predict_support_ridge(
        support,
        support_y,
        np.vstack([query, np.full((7, query.shape[1]), 9.0, dtype=np.float32)]),
        ridge_lambda=0.1,
    )
    assert np.mean(predicted.astype(str) == query_y.astype(str)) == 1.0
    assert np.array_equal(predicted, extended[: len(predicted)])
    assert info["trainable_parameters"] == 0
    assert info["feature_adapter_gradient_updates"] == 0
    assert info["query_features_used_for_adaptation"] is False
    assert info["persistent_state_bytes"] <= 128 * 1024
    assert len(trace) == 1


def test_low_rank_cosine_head_is_bounded_and_query_independent():
    support, support_y, query, query_y = _separable()
    predicted, info, trace = fit_predict_extreme_light_low_rank_cosine(
        support,
        support_y,
        query,
        seed=17,
        rank=8,
        cosine_margin=0.1,
        epochs=5,
        device="cpu",
    )
    extended, _, _ = fit_predict_extreme_light_low_rank_cosine(
        support,
        support_y,
        np.vstack([query, np.full((7, query.shape[1]), 9.0, dtype=np.float32)]),
        seed=17,
        rank=8,
        cosine_margin=0.1,
        epochs=5,
        device="cpu",
    )
    assert np.mean(predicted.astype(str) == query_y.astype(str)) == 1.0
    assert np.array_equal(predicted, extended[: len(predicted)])
    assert info["trainable_parameters"] <= 50_000
    assert info["persistent_state_bytes"] <= 128 * 1024
    assert info["query_features_used_for_adaptation"] is False
    assert len(trace) == 5


def test_feature_concatenation_is_per_sample_and_validates_alignment():
    primary = np.eye(3, dtype=np.float32)
    auxiliary = np.flip(primary, axis=1).copy()
    combined = concatenate_registered_features(primary, auxiliary, auxiliary_weight=2.0)
    assert combined.shape == (3, 6)
    assert np.allclose(np.linalg.norm(combined, axis=1), 1.0)
    primary_only = concatenate_registered_features(primary, auxiliary, auxiliary_weight=0.0)
    assert primary_only.shape == (3, 3)
    assert np.allclose(primary_only, primary)
    with pytest.raises(ValueError, match="align"):
        concatenate_registered_features(primary, auxiliary[:2], auxiliary_weight=1.0)


def test_resource_and_epoch_caps_fail_closed():
    support, support_y, query, _ = _separable()
    with pytest.raises(ValueError, match="epochs"):
        fit_predict_extreme_light_diag_cosine(
            support, support_y, query, seed=1, epochs=21, device="cpu"
        )
    large_support = np.pad(support, ((0, 0), (0, 4096 - support.shape[1])))
    large_query = np.pad(query, ((0, 0), (0, 4096 - query.shape[1])))
    with pytest.raises(ValueError, match="parameter cap"):
        fit_predict_extreme_light_diag_cosine(
            large_support,
            support_y,
            large_query,
            seed=1,
            epochs=1,
            max_trainable_parameters=10_000,
            device="cpu",
        )
