from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi.stage2_predictor_runtime import (
    ADAPTER_SCHEMA,
    HEAD_SCHEMA,
    SYMMETRIC_HEAD_SCHEMA_V2,
    TTA_SCHEMA,
    Stage2PredictorRuntimeError,
    _float32_tensor_without_numpy_bridge,
    apply_feature_adapter,
    build_formal_support_state,
    predict_formal_scenario_streams,
    predict_all_streams,
    select_nested_support_prefix,
    spectral_logmag_sketch,
)


class TinyRuntime(torch.nn.Module):
    def forward(self, rows: torch.Tensor):
        features = rows.mean(dim=2)
        logits = torch.stack((features[:, 0], features[:, 1]), dim=1)
        return {"features": features, "logits": logits}


class WideRuntime(torch.nn.Module):
    def forward(self, rows: torch.Tensor):
        narrow = rows.mean(dim=2)
        features = narrow.repeat(1, 128)
        logits = torch.stack((narrow[:, 0], narrow[:, 1]), dim=1)
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


def test_float32_tensor_avoids_numpy_bridge_and_owns_storage(monkeypatch) -> None:
    rows = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    expected = rows.copy()

    def forbidden_from_numpy(*_args, **_kwargs):
        raise AssertionError("strict runtime must not use torch.from_numpy")

    monkeypatch.setattr(torch, "from_numpy", forbidden_from_numpy)
    tensor = _float32_tensor_without_numpy_bridge(rows, device=torch.device("cpu"))
    rows.fill(-1.0)

    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == expected.shape
    assert tensor.tolist() == expected.tolist()


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


def test_effective8_formal_runtime_uses_three_scenario_head_and_distinct_base() -> None:
    support, query = _arrays()
    support_by_scenario = {
        name: support
        for name in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    }
    adapter = _adapter()
    adapter.update(
        {
            "trainable_parameters": 44_048,
            "adapt_epochs": 12,
            "persistent_state_bytes": 100_000,
        }
    )
    head = {
        "schema": "cvs.phase2.symmetric_locked_head.v1",
        "mode": "three_leo_support_symmetric_locked",
        "selected": {
            "use_alignment": False,
            "prototype_rule": "mean",
            "ridge": None,
            "gram_mix": 0.0,
            "uncertainty_penalty": 0.0,
        },
        "source_feature_mean": [0.0, 0.0],
        "source_feature_std": [1.0, 1.0],
        "variance_floor": 0.05,
        "storage_dtype": "fp16",
    }
    tta = {
        "schema": TTA_SCHEMA,
        "mode": "adaptive_1_3_5",
        "base_views": 1,
        "max_views": 5,
        "base_stop_margin": 0.0,
        "shift3_stop_margin": 0.0,
        "shift3_max_disagreement": 1.0,
        "base_stop_min_score": -1.0e9,
        "shift3_stop_min_score": -1.0e9,
        "fusion_std_penalty": 0.0,
        "calibration_scope": "source_validation",
        "uses_query_labels": False,
        "uses_query_role": False,
        "uses_class_quota": False,
    }
    candidate = TinyRuntime()
    base = TinyRuntime()
    state = build_formal_support_state(
        candidate,
        base,
        support_by_scenario,
        scenarios=tuple(support_by_scenario),
        k_shot=1,
        registered_class_count=2,
        new_class_count=1,
        adapter_config=adapter,
        head_config=head,
        device=torch.device("cpu"),
        batch_size=2,
    )
    predictions, resources = predict_formal_scenario_streams(
        candidate,
        base,
        query,
        state,
        scenario="leo_clear_weak",
        old_class_count=1,
        adapter_config=adapter,
        tta_config=tta,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert predictions["candidate_after"].tolist() == [0, 1]
    assert predictions["candidate_before"].tolist() == [0, 0]
    assert predictions["identity_after"].tolist() == [0, 1]
    assert predictions["direct"].tolist() == [0, 0]
    assert predictions["shared_view_counts"].tolist() == [1, 1]
    assert resources["candidate_and_base_runtimes_distinct"] is True
    assert resources["candidate_head"] == "three_leo_support_symmetric_locked_fp16"
    assert "persistent_state_bytes_total" not in state
    assert "candidate_head_deployment_state_bytes_fp16" not in resources


def test_effective8_formal_runtime_reaches_symmetric_evidence_head_v2() -> None:
    support, query = _arrays()
    support_by_scenario = {
        name: support
        for name in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    }
    adapter = _adapter()
    head = {
        "schema": SYMMETRIC_HEAD_SCHEMA_V2,
        "mode": "three_leo_support_symmetric_evidence_locked",
        "selected": {
            "use_alignment": False,
            "prototype_rule": "mean",
            "ridge": None,
            "gram_mix": 0.0,
            "uncertainty_penalty": 0.0,
            "evidence_calibration": {
                "mode": "robust_lopo_class_symmetric",
                "negative_quantile": 0.95,
                "prior_physical_shots": 8.0,
                "scale_floor": 0.05,
                "inverse_scale_cap": 10.0,
            },
        },
        "source_feature_mean": [0.0, 0.0],
        "source_feature_std": [1.0, 1.0],
        "variance_floor": 0.05,
        "storage_dtype": "fp16",
    }
    tta = {
        "schema": TTA_SCHEMA,
        "mode": "adaptive_1_3_5",
        "base_views": 1,
        "max_views": 5,
        "base_stop_margin": 0.0,
        "shift3_stop_margin": 0.0,
        "shift3_max_disagreement": 1.0,
        "base_stop_min_score": -1.0e9,
        "shift3_stop_min_score": -1.0e9,
        "fusion_std_penalty": 0.0,
        "calibration_scope": "registered_support",
        "uses_query_labels": False,
        "uses_query_role": False,
        "uses_class_quota": False,
    }
    state = build_formal_support_state(
        TinyRuntime(),
        TinyRuntime(),
        support_by_scenario,
        scenarios=tuple(support_by_scenario),
        k_shot=1,
        registered_class_count=2,
        new_class_count=1,
        adapter_config=adapter,
        head_config=head,
        device=torch.device("cpu"),
        batch_size=2,
    )
    predictions, resources = predict_formal_scenario_streams(
        TinyRuntime(),
        TinyRuntime(),
        query,
        state,
        scenario="leo_clear_weak",
        old_class_count=1,
        adapter_config=adapter,
        tta_config=tta,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert predictions["candidate_after"].tolist() == [0, 1]
    assert predictions["shared_view_counts"].tolist() == [1, 1]
    assert resources["candidate_head"] == (
        "three_leo_support_symmetric_evidence_locked_fp16"
    )
    diagnostics = resources["candidate_head_closed_form_diagnostics"]
    assert diagnostics["fold_mode"] == "leave_one_view_out"
    assert diagnostics["state_bytes_fp16"] == 8
    assert state["candidate_head_evidence_deployment_state_bytes_fp16"] == 8
    assert state[
        "candidate_head_evidence_evaluation_comparator_state_bytes_fp16"
    ] == 4
    assert state["candidate_head_evidence_formal_dual_stream_state_bytes_fp16"] == 12
    assert state["candidate_head_deployment_state_bytes_fp16"] == 28
    assert state["candidate_head_evaluation_comparator_state_bytes_fp16"] == 12
    assert state["candidate_head_formal_dual_stream_state_bytes_fp16"] == 40
    assert state["candidate_head_deployment_live_array_bytes"] == 56
    assert state["candidate_head_evaluation_comparator_live_array_bytes"] == 24
    assert state["candidate_head_formal_dual_stream_live_array_bytes"] == 80
    assert state["candidate_head_live_array_bytes"] == 80
    assert state["persistent_state_bytes_total"] == 28
    assert state["formal_dual_stream_persistent_state_bytes_total"] == 40
    assert resources["candidate_head_evidence_deployment_state_bytes_fp16"] == 8


def _wide_support(class_count: int) -> dict[str, np.ndarray]:
    iq = np.zeros((class_count, 2, 4), dtype=np.float32)
    for class_index in range(class_count):
        iq[class_index, class_index % 2] = 1.0 + class_index / class_count
    return {
        "support_pool_leo_weak_iq": iq,
        "support_pool_class_indices": np.arange(class_count, dtype=np.int64),
        "support_pool_rank_within_class": np.zeros(class_count, dtype=np.int64),
        "support_pool_tokens": np.asarray(
            [f"sid_{class_index:064x}" for class_index in range(class_count)]
        ),
    }


def test_evidence_resource_accounting_matches_256d_26_and_6_class_heads() -> None:
    support = _wide_support(26)
    support_by_scenario = {
        name: support
        for name in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    }
    head = {
        "schema": SYMMETRIC_HEAD_SCHEMA_V2,
        "mode": "three_leo_support_symmetric_evidence_locked",
        "selected": {
            "use_alignment": False,
            "prototype_rule": "mean",
            "ridge": None,
            "gram_mix": 0.0,
            "uncertainty_penalty": 0.0,
            "evidence_calibration": {
                "mode": "robust_lopo_class_symmetric",
                "negative_quantile": 0.95,
                "prior_physical_shots": 8.0,
                "scale_floor": 0.05,
                "inverse_scale_cap": 10.0,
            },
        },
        "source_feature_mean": [0.0] * 256,
        "source_feature_std": [1.0] * 256,
        "variance_floor": 0.05,
        "storage_dtype": "fp16",
    }
    state = build_formal_support_state(
        WideRuntime(),
        WideRuntime(),
        support_by_scenario,
        scenarios=tuple(support_by_scenario),
        k_shot=1,
        registered_class_count=26,
        new_class_count=20,
        adapter_config=_adapter(),
        head_config=head,
        device=torch.device("cpu"),
        batch_size=26,
    )

    assert state["candidate_head_deployment_state_bytes_fp16"] == 14_820
    assert state["candidate_head_evaluation_comparator_state_bytes_fp16"] == 3_180
    assert state["candidate_head_formal_dual_stream_state_bytes_fp16"] == 18_000
    assert state["candidate_head_deployment_live_array_bytes"] == 29_640
    assert state["candidate_head_evaluation_comparator_live_array_bytes"] == 6_360
    assert state["candidate_head_formal_dual_stream_live_array_bytes"] == 36_000
    assert state["candidate_head_evidence_deployment_state_bytes_fp16"] == 104
    assert state[
        "candidate_head_evidence_evaluation_comparator_state_bytes_fp16"
    ] == 24
    assert state["candidate_head_evidence_formal_dual_stream_state_bytes_fp16"] == 128


def test_evidence_deployment_head_pushes_adapter_over_state_cap() -> None:
    support = _wide_support(26)
    support_by_scenario = {
        name: support
        for name in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    }
    adapter = _adapter()
    adapter["persistent_state_bytes"] = 256 * 1024
    head = {
        "schema": SYMMETRIC_HEAD_SCHEMA_V2,
        "mode": "three_leo_support_symmetric_evidence_locked",
        "selected": {
            "use_alignment": False,
            "prototype_rule": "mean",
            "ridge": None,
            "gram_mix": 0.0,
            "uncertainty_penalty": 0.0,
            "evidence_calibration": {
                "mode": "robust_lopo_class_symmetric",
                "negative_quantile": 0.95,
                "prior_physical_shots": 8.0,
                "scale_floor": 0.05,
                "inverse_scale_cap": 10.0,
            },
        },
        "source_feature_mean": [0.0] * 256,
        "source_feature_std": [1.0] * 256,
        "variance_floor": 0.05,
        "storage_dtype": "fp16",
    }

    with pytest.raises(Stage2PredictorRuntimeError, match="deployment head exceeds"):
        build_formal_support_state(
            WideRuntime(),
            WideRuntime(),
            support_by_scenario,
            scenarios=tuple(support_by_scenario),
            k_shot=1,
            registered_class_count=26,
            new_class_count=20,
            adapter_config=adapter,
            head_config=head,
            device=torch.device("cpu"),
            batch_size=26,
        )
