from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi.stage2_predictor_runtime import (
    ADAPTER_SCHEMA,
    HEAD_SCHEMA,
    TTA_SCHEMA,
    Stage2PredictorRuntimeError,
    apply_feature_adapter,
    predict_all_streams,
    select_nested_support_prefix,
    spectral_logmag_sketch,
)


class TinyRuntime(torch.nn.Module):
    def forward(self, rows: torch.Tensor):
        features = rows.mean(dim=2)
        logits = torch.stack((features[:, 0], features[:, 1]), dim=1)
        return {"features": features, "logits": logits}


def _adapter(fft_dim: int = 0):
    return {
        "schema": ADAPTER_SCHEMA,
        "mode": "identity",
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "persistent_state_bytes": 0,
        "fft_dim": fft_dim,
        "fft_weight": 1.0,
    }


def _head():
    return {"schema": HEAD_SCHEMA, "metric": "cosine", "temperature": 10.0}


def _qknn_fft96_adapter():
    return {
        "schema": ADAPTER_SCHEMA,
        "mode": "qknnv42_fft96",
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "persistent_state_bytes": 256 * 1024,
        "fft_dim": 96,
        "fft_weight": 0.34,
        "feature_adapter_mode": "support_diag_whiten_fisher",
        "quantization_bits": 8,
        "support_score_weight": 0.55,
        "prototype_score_weight": 0.45,
        "support_prototype_residual_weight": 0.025,
        "labelprop_mode": "support_prototype",
        "old_anchor_bias": 0.0,
        "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
    }


def _tta(mode: str = "base_only"):
    if mode == "base_only":
        return {"schema": TTA_SCHEMA, "mode": mode, "base_views": 1, "max_views": 1}
    return {
        "schema": TTA_SCHEMA, "mode": "adaptive_1_3_5", "base_views": 1,
        "max_views": 5, "base_stop_margin": 0.0, "shift3_stop_margin": 0.0,
        "shift3_max_disagreement": 0.0,
        "calibration_scope": "registered_support",
        "uses_query_labels": False, "uses_query_role": False, "uses_class_quota": False,
    }


def _arrays():
    support = np.asarray([
        [[2, 2, 2, 2], [0, 0, 0, 0]],
        [[2, 2, 2, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [2, 2, 2, 2]],
        [[0, 0, 0, 0], [2, 2, 2, 2]],
    ], dtype=np.float32)
    support_arrays = {
        "support_pool_leo_weak_iq": support,
        "support_pool_class_indices": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "support_pool_rank_within_class": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "support_pool_tokens": np.asarray([f"sid_{i:064x}" for i in range(4)]),
    }
    query_arrays = {
        "query_leo_weak_iq": np.asarray([
            [[1, 1, 1, 1], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 1, 1, 1]],
        ], dtype=np.float32),
        "query_tokens": np.asarray(["qid_" + "1" * 64, "qid_" + "2" * 64]),
    }
    return support_arrays, query_arrays


def test_nested_k_prefix_is_per_class() -> None:
    support, _query = _arrays()
    _iq, labels, tokens = select_nested_support_prefix(support, k_shot=1, class_count=2)
    assert labels.tolist() == [0, 1]
    assert tokens.tolist() == [f"sid_{0:064x}", f"sid_{2:064x}"]


def test_predictor_emits_five_role_blind_streams_with_shared_budget() -> None:
    support, query = _arrays()
    predictions, resources = predict_all_streams(
        TinyRuntime(), support, query, k_shot=1, registered_class_count=2,
        new_class_count=1, adapter_config=_adapter(), head_config=_head(),
        tta_config=_tta(), device=torch.device("cpu"), batch_size=2,
    )
    assert set(predictions) == {
        "candidate_after", "candidate_before", "identity_after",
        "identity_before", "direct", "shared_view_counts",
    }
    assert predictions["candidate_after"].tolist() == [0, 1]
    assert predictions["candidate_before"].tolist() == [0, 0]
    assert predictions["shared_view_counts"].tolist() == [1, 1]
    assert resources["shared_view_budget_for_all_streams"] is True


def test_fft96_descriptor_is_deterministic_and_finite() -> None:
    support, _query = _arrays()
    rows = support["support_pool_leo_weak_iq"]
    first = spectral_logmag_sketch(rows, dim=96)
    second = spectral_logmag_sketch(rows, dim=96)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 96)
    assert np.isfinite(first).all()


def test_qknnv42_fft96_is_role_blind_single_view_and_reports_closed_form_state() -> None:
    support, query = _arrays()
    predictions, resources = predict_all_streams(
        TinyRuntime(),
        support,
        query,
        k_shot=2,
        registered_class_count=2,
        new_class_count=1,
        adapter_config=_qknn_fft96_adapter(),
        head_config=_head(),
        tta_config=_tta(),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert predictions["candidate_after"].tolist() == [0, 1]
    assert predictions["shared_view_counts"].tolist() == [1, 1]
    runtime = resources["candidate_runtime"]["after_registration"]
    assert runtime["adaptation_mode"] == "EVAL_ONLY_CLOSED_FORM_ADAPTATION"
    assert runtime["query_query_graph_used"] is False
    assert runtime["fft_dim"] == 96
    assert runtime["fft_weight"] == pytest.approx(0.34)
    assert runtime["persistent_state_bytes"] <= 256 * 1024
    assert resources["persistent_state_bytes"] == runtime["persistent_state_bytes"]
    assert resources["persistent_state_cap_bytes"] == 256 * 1024


def test_qknnv42_fft96_rejects_adaptive_multiview() -> None:
    support, query = _arrays()
    with pytest.raises(Stage2PredictorRuntimeError, match="locked to one"):
        predict_all_streams(
            TinyRuntime(),
            support,
            query,
            k_shot=1,
            registered_class_count=2,
            new_class_count=1,
            adapter_config=_qknn_fft96_adapter(),
            head_config=_head(),
            tta_config=_tta("adaptive"),
            device=torch.device("cpu"),
            batch_size=2,
        )


def test_adapter_resource_cap_fails_closed() -> None:
    support, _query = _arrays()
    config = _adapter()
    config["trainable_parameters"] = 50_001
    with pytest.raises(Stage2PredictorRuntimeError, match="resource bound"):
        apply_feature_adapter(
            support["support_pool_leo_weak_iq"].mean(axis=2),
            support["support_pool_leo_weak_iq"],
            config,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("adapt_epochs", 21), ("persistent_state_bytes", 256 * 1024 + 1)),
)
def test_adapter_epoch_and_state_caps_fail_closed(field: str, value: int) -> None:
    support, _query = _arrays()
    config = _adapter()
    config[field] = value
    with pytest.raises(Stage2PredictorRuntimeError, match=f"resource bound invalid: {field}"):
        apply_feature_adapter(
            support["support_pool_leo_weak_iq"].mean(axis=2),
            support["support_pool_leo_weak_iq"],
            config,
        )
