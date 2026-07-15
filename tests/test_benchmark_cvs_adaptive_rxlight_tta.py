from __future__ import annotations

import numpy as np
import pytest
import torch

import paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta as benchmark

from paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta import (
    _requested_rxlight_views,
    _reference_parity,
    apply_fp16_checkpoint_delta,
    apply_fp16_lora_state,
    audit_adapter_manifest,
    build_view_prototypes,
    build_single_view_prototypes,
    leave_one_out_support_scores,
    load_trusted_class_id_to_tx,
    order_k_support_observations,
    order_k1_support_views,
    predict_direct_adv3b02_base_view,
    score_symmetric_named_views,
    score_views,
    validate_formal_phase2_config,
)
from cvsrffi.leo_weak_cache import PHASE2_SAMPLE_VIEW_POLICY
from paper_reproduction.cvs_aligned.k1_symmetric_head import fit_symmetric_k1_head


def test_matching_view_prototypes_preserve_shape_and_class_order() -> None:
    rng = np.random.default_rng(7)
    support = rng.normal(size=(5, 6, 4)).astype(np.float32)
    labels = np.asarray(["b", "a", "b", "a", "b", "a"])
    classes = ["a", "b"]
    prototypes = build_view_prototypes(support, labels, classes)
    scores = score_views(support[:, :2], prototypes)
    assert prototypes.shape == (5, 2, 4)
    assert prototypes.dtype == np.float16
    assert scores.shape == (2, 5, 2)
    assert np.isfinite(scores).all()


def test_identity_single_qknn_supports_k1_without_role_rules() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prototypes = build_single_view_prototypes(
        features, np.asarray(["old", "new"]), ["new", "old"]
    )
    assert prototypes.shape == (2, 2)
    np.testing.assert_array_equal(np.argmax(features @ prototypes.T, axis=1), [1, 0])


def test_leave_one_out_scores_do_not_use_the_sample_in_its_class_mean() -> None:
    support = np.zeros((5, 4, 2), dtype=np.float32)
    support[:, 0] = [1.0, 0.0]
    support[:, 1] = [0.0, 1.0]
    support[:, 2] = [-1.0, 0.0]
    support[:, 3] = [0.0, -1.0]
    labels = np.asarray(["a", "a", "b", "b"])
    scores = leave_one_out_support_scores(support, labels, ["a", "b"])
    # Row 0's class-a LOO prototype is row 1, hence orthogonal rather than self-aligned.
    assert scores.shape == (4, 5, 2)
    assert scores[0, 0, 0] == pytest.approx(0.0, abs=1e-6)


def test_k1_support_order_and_symmetric_scoring() -> None:
    rng = np.random.default_rng(17)
    support = rng.normal(size=(5, 3, 6)).astype(np.float32)
    labels = np.asarray(["b", "c", "a"])
    ordered = order_k1_support_views(support, labels, ["a", "b", "c"])
    np.testing.assert_array_equal(ordered[:, 0], support[:, 2])
    all_views = np.concatenate([ordered, ordered + 0.01, ordered - 0.01], axis=0)
    head = fit_symmetric_k1_head(
        all_views,
        prototype_rules=("mean",),
        ridges=(None,),
        allow_alignment=False,
    )
    scores = score_symmetric_named_views(support, head)
    assert scores.shape == (3, 5, 3)


def test_k1_support_order_rejects_more_than_one_physical_shot() -> None:
    support = np.ones((5, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly one physical support"):
        order_k1_support_views(support, np.asarray(["a", "a", "b"]), ["a", "b"])


def test_k5_support_observations_preserve_every_physical_shot() -> None:
    features = np.arange(2 * 10 * 4, dtype=np.float32).reshape(2, 10, 4)
    labels = np.asarray(["a"] * 5 + ["b"] * 5)
    ordered = order_k_support_observations(
        features, labels, ["b", "a"], k_shot=5
    )
    assert ordered.shape == (10, 2, 4)
    np.testing.assert_array_equal(ordered[:5, 0], features[0, 5:10])
    np.testing.assert_array_equal(ordered[5:, 1], features[1, 0:5])


def test_direct_adv3b02_baseline_uses_one_base_view_without_fft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.asarray(
        [
            [[2.0, 0.0], [0.0, 0.0]],
            [[-2.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    def fake_views(rows: torch.Tensor, _policy: str):
        return tuple((name, rows) for name in benchmark.RX_LIGHT5_ORDER)

    def fake_feature(_model: torch.nn.Module, rows: torch.Tensor):
        logits = torch.stack([rows[:, 0, 0], -rows[:, 0, 0]], dim=1)
        return rows.flatten(1), logits

    monkeypatch.setattr(benchmark, "_satellite_tta_views", fake_views)
    monkeypatch.setattr(benchmark, "_feature_forward", fake_feature)
    labels, audit = predict_direct_adv3b02_base_view(
        torch.nn.Identity(),
        raw,
        class_labels=["a", "b"],
        batch_size=2,
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(labels, np.asarray(["a", "b"]))
    assert audit["support_rows_used"] == 0
    assert audit["fft_used"] is False
    assert audit["tta_view_count"] == 1


def test_base_view_does_not_materialize_the_other_four_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = torch.zeros((2, 2, 16), dtype=torch.float32)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full/cfo view builder must not run for base-only inference")

    monkeypatch.setattr(benchmark, "_satellite_tta_views", forbidden)
    generated = _requested_rxlight_views(rows, ("rx_base",))
    assert [name for name, _value in generated] == ["rx_base"]
    assert generated[0][1] is rows


def test_formal_config_rejects_legacy_raw_or_feature_paths() -> None:
    config = {
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "old_new_role_oracle_used": False,
        "class_quota_used": False,
        "query_fit_used": False,
        "target_channel_scenarios": list(benchmark.FORMAL_LEO_WEAK_SCENARIOS),
        "leo_weak_cache_set_manifest": "sealed.json",
        "leo_weak_iq_input_len": 256,
        "raw_iq_input_len": 256,
    }
    with pytest.raises(ValueError, match="legacy feature_npz/raw_iq"):
        validate_formal_phase2_config(config)


def test_trusted_direct_mapping_is_explicit_and_ordered(tmp_path) -> None:
    path = tmp_path / "split_manifest.json"
    path.write_text(
        '{"class_id_to_tx":["tx-b","tx-a"]}\n', encoding="utf-8"
    )
    assert load_trusted_class_id_to_tx(path) == ["tx-b", "tx-a"]


def test_split_fails_when_support_pool_is_smaller_than_k() -> None:
    arrays = {"tx_ids": np.asarray(["a", "a"])}
    config = {
        "target_receiver_labels": ["r"],
        "seed": 1,
        "k_shot": 20,
        "support_pool_max_k": 10,
        "query_per_tx": 1,
        "target_old_tx_labels": ["a"],
        "target_new_tx_labels": ["b"],
    }
    with pytest.raises(ValueError, match="support_pool_max_k must cover K"):
        benchmark._split_indices(arrays, config, "leo_clear_weak")


def test_split_fails_closed_on_support_query_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = {"tx_ids": np.asarray(["a", "a", "b", "b"])}
    calls = iter([([0], [0]), ([2], [3])])
    monkeypatch.setattr(benchmark, "_select_split", lambda *_args, **_kwargs: next(calls))
    config = {
        "target_receiver_labels": ["r"],
        "seed": 1,
        "k_shot": 1,
        "support_pool_max_k": 1,
        "query_per_tx": 1,
        "target_old_tx_labels": ["a"],
        "target_new_tx_labels": ["b"],
    }
    with pytest.raises(ValueError, match="support/query overlap"):
        benchmark._split_indices(arrays, config, "leo_clear_weak")


class _TinyLateKey(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.t_proj = torch.nn.Linear(1, 1)
        self.id_backbone.f_proj = torch.nn.Linear(1, 1)
        self.id_backbone.pa_proj = torch.nn.Sequential(torch.nn.Linear(1, 1))


def test_fp16_delta_rejects_wrong_element_budget() -> None:
    model = _TinyLateKey()
    state = {
        name: torch.zeros_like(parameter, dtype=torch.float16)
        for name, parameter in model.named_parameters()
    }
    with pytest.raises(ValueError, match="element budget drift"):
        apply_fp16_checkpoint_delta(model, state)


def test_reference_parity_accepts_feature_cache_without_raw_iq(tmp_path) -> None:
    path = tmp_path / "features_only.npz"
    arrays = {
        "dataset_role": np.asarray(["target_old"]),
        "tx_ids": np.asarray(["a"]),
        "rx_ids": np.asarray(["r"]),
        "day_ids": np.asarray(["d"]),
        "eq_ids": np.asarray(["1"]),
        "sig_ids": np.asarray(["0"]),
    }
    primary = np.asarray([[1.0, 0.0]], dtype=np.float32)
    fft = np.asarray([[0.0, 1.0]], dtype=np.float32)
    np.savez(path, **arrays, features=primary, fft_logmag_features=fft)
    generated = np.asarray([[1.0, 0.0, 0.0, 2.0]], dtype=np.float32)
    audit = _reference_parity(path, arrays, [0], generated)
    assert audit["checked"] is True
    assert audit["min_cosine"] == pytest.approx(1.0)


def test_adapter_manifest_requires_support_only_pair_provenance(tmp_path) -> None:
    state = tmp_path / "state.pt"
    state.write_bytes(b"state")
    import hashlib

    digest = hashlib.sha256(b"state").hexdigest()
    manifest = {
        "method": "support_only_late_key_ft_source_init_rx_shift_pair_v1",
        "support_only": True,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "epochs": 5,
        "adapter_state_format": "fp16_delta_from_strict_checkpoint",
        "adapter_state_sha256": digest,
        "support_view_policy": "rx_shift_pair_cycle",
        "runtime": {"optimizer_steps": 20},
        "resources": {
            "trainable_parameters": 31_200,
            "adapter_state_bytes_fp16": 62_400,
            "deployment_added_macs_per_query_after_merge": 0,
        },
    }
    audit = audit_adapter_manifest(manifest, adapter_state=state)
    assert audit["method"].endswith("rx_shift_pair_v1")
    manifest["class_quota_used_at_inference"] = True
    with pytest.raises(ValueError, match="no_class_quota"):
        audit_adapter_manifest(manifest, adapter_state=state)


class _TinyFullHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_proj = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.pa_proj = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.id_gate = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.joint_proj = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.imp_merge = torch.nn.Sequential(torch.nn.Linear(2, 2))


class _TinyFullModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.t_proj = torch.nn.Linear(2, 2)
        self.id_backbone.f_proj = torch.nn.Linear(2, 2)
        self.id_backbone.pa_proj = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.id_backbone.fuse = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.id_backbone.con_proj = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.id_backbone.cls_head = _TinyFullHead()


def test_fp16_full_feature_lora_state_loads_strictly() -> None:
    source = _TinyFullModel()
    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
        inject_feat_joint_lora,
    )

    resources = inject_feat_joint_lora(
        source, rank=2, alpha=2.0, scope="full_feature"
    )
    state = {
        name: parameter.detach().half()
        for name, parameter in source.named_parameters()
        if parameter.requires_grad
    }
    target = _TinyFullModel()
    audit = apply_fp16_lora_state(
        target, state, scope="full_feature", rank=2, alpha=2.0
    )
    assert audit["element_count"] == resources["trainable_parameters"]
    assert audit["tensor_bytes_fp16"] == 2 * resources["trainable_parameters"]
    assert audit["mergeable_into_base_linear_weights"] is True


def test_relaxed_lora_manifest_keeps_permission_gates(tmp_path) -> None:
    state = tmp_path / "lora.pt"
    state.write_bytes(b"lora")
    import hashlib

    digest = hashlib.sha256(b"lora").hexdigest()
    manifest = {
        "method": "support_only_full_feature_lora_v1",
        "resource_tier": "performance_relaxed",
        "support_only": True,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "epochs": 10,
        "adapter_state_format": "fp16_trainable_state",
        "adapter_state_sha256": digest,
        "support_view_policy": "rx_shift_pair_cycle",
        "hyperparameters": {"scope": "full_feature", "rank": 24, "alpha": 24.0},
        "runtime": {"optimizer_steps": 100},
        "resources": {
            "trainable_parameters": 81_432,
            "adapter_state_bytes_fp16": 162_864,
            "combined_persistent_state_within_cap": True,
        },
    }
    audit = audit_adapter_manifest(manifest, adapter_state=state)
    assert audit["method"] == "support_only_full_feature_lora_v1"
    manifest["query_labels_used_for_training"] = True
    with pytest.raises(ValueError, match="no_query_labels"):
        audit_adapter_manifest(manifest, adapter_state=state)


def test_ground_source_lora_requires_passed_source_validation(tmp_path) -> None:
    state = tmp_path / "ground_lora.pt"
    state.write_bytes(b"ground-lora")
    validation = tmp_path / "source_validation.json"
    stats = tmp_path / "source_stats.npz"
    np.savez(stats, mean=np.zeros(256), std=np.ones(256))
    import hashlib
    import json

    validation_payload = {
        "schema": "cvs_ground_source_lora_multiview_validation_v1",
        "source_validation_pass": True,
        "adapter_state_sha256": hashlib.sha256(b"ground-lora").hexdigest(),
        "checkpoint_sha256": "checkpoint-hash",
        "training_manifest_sha256": "training-manifest-hash",
        "failed_gates": [],
        "gates": {"all_source_checks": True},
        "receiver_holdout": {"disjoint": True, "overlap": []},
        "symmetric_head_lock": {
            "selection_source": "disjoint_source_receiver_holdout_k1_episodes",
            "support_view_policy": "three_leo_weak_scenario_base_views",
            "support_receive_views_per_physical_sample": 3,
            "target_support_used_for_selection": False,
            "target_query_features_used": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
        },
        "source_feature_statistics": {
            "path": str(stats),
            "sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "feature_kind": "normalized_z_id_plus_fft96_weight2",
            "feature_dim": 256,
            "fft_dim": 96,
            "fft_weight": 2.0,
            "target_rows_used": False,
        },
        "permissions": {
            "target_support_used": False,
            "target_query_features_used": False,
            "target_query_labels_used": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
        },
    }
    validation.write_text(json.dumps(validation_payload), encoding="utf-8")

    manifest = {
        "method": "ground_source_full_feature_lora_v1",
        "resource_tier": "preferred",
        "source_only": True,
        "support_only": False,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "target_receiver_data_used_for_training": False,
        "source_validation_pass": True,
        "checkpoint_sha256": "checkpoint-hash",
        "training_manifest_sha256": "training-manifest-hash",
        "source_validation_manifest": str(validation),
        "source_validation_manifest_sha256": hashlib.sha256(
            validation.read_bytes()
        ).hexdigest(),
        "source_validation_permissions": {
            "target_support_used": False,
            "target_query_features_used": False,
            "target_query_labels_used": False,
        },
        "epochs": 20,
        "adapter_state_format": "fp16_trainable_state",
        "adapter_state_sha256": hashlib.sha256(b"ground-lora").hexdigest(),
        "hyperparameters": {"scope": "full_feature", "rank": 12, "alpha": 12.0},
        "resources": {
            "trainable_parameters": 40_716,
            "adapter_state_bytes_fp16": 81_432,
            "combined_persistent_state_within_cap": True,
        },
    }
    audit = audit_adapter_manifest(manifest, adapter_state=state)
    assert audit["method"] == "ground_source_full_feature_lora_v1"
    assert audit["source_validation_pass"] is True
    manifest["source_validation_pass"] = False
    with pytest.raises(ValueError, match="source_validation_pass"):
        audit_adapter_manifest(manifest, adapter_state=state)


def test_effective_ground_lora_audit_branch_accepts_leo_only_preferred_state(
    tmp_path,
) -> None:
    state = tmp_path / "ground_lora.pt"
    state.write_bytes(b"ground-lora")
    stats = tmp_path / "source_stats.npz"
    np.savez(stats, mean=np.zeros(256), std=np.ones(256))
    import hashlib
    import json

    validation = tmp_path / "source_validation.json"
    payload = {
        "schema": "cvs_ground_source_lora_multiview_validation_v1",
        "source_validation_pass": True,
        "clean_samples_used_for_validation": False,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "validation_input_stage": "phase1_offline_prechannel_export",
        "source_leo_weak_cache_set_manifest_sha256": "a" * 64,
        "adapter_state_sha256": hashlib.sha256(b"ground-lora").hexdigest(),
        "checkpoint_sha256": "checkpoint-hash",
        "training_manifest_sha256": "training-manifest-hash",
        "failed_gates": [],
        "gates": {"leo_only": True},
        "receiver_holdout": {"disjoint": True, "overlap": []},
        "permissions": {
            "target_support_used": False,
            "target_query_features_used": False,
            "target_query_labels_used": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
        },
        "symmetric_head_lock": {
            "selection_source": "disjoint_source_receiver_holdout_k1_episodes",
            "support_view_policy": "three_leo_weak_scenario_base_views",
            "support_receive_views_per_physical_sample": 3,
            "target_support_used_for_selection": False,
            "target_query_features_used": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
        },
        "source_feature_statistics": {
            "path": str(stats),
            "sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
            "feature_kind": "normalized_z_id_plus_fft96_weight2",
            "feature_dim": 256,
            "fft_dim": 96,
            "fft_weight": 2.0,
            "target_rows_used": False,
        },
    }
    validation.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(b"ground-lora").hexdigest()
    manifest = {
        "method": "ground_source_effective_feature_lora_v1",
        "resource_tier": "preferred",
        "source_only": True,
        "support_only": False,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "target_receiver_data_used_for_training": False,
        "clean_samples_used_for_training": False,
        "formal_training_view": "leo_weak_only",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "training_input_stage": "phase1_offline_prechannel_export",
        "source_leo_weak_cache_set_manifest_sha256": "b" * 64,
        "proxy_data_used_for_training": False,
        "proxy_training_rows": 0,
        "source_validation_pass": True,
        "source_validation_manifest": str(validation),
        "source_validation_manifest_sha256": hashlib.sha256(
            validation.read_bytes()
        ).hexdigest(),
        "source_validation_permissions": payload["permissions"],
        "checkpoint_sha256": "checkpoint-hash",
        "training_manifest_sha256": "training-manifest-hash",
        "epochs": 12,
        "adapter_state_format": "fp16_trainable_state",
        "adapter_state_sha256": digest,
        "hyperparameters": {
            "scope": "effective_feature",
            "rank": 16,
            "alpha": 16.0,
        },
        "resources": {
            "trainable_parameters": 44_048,
            "adapter_state_bytes_fp16": 88_096,
            "combined_persistent_state_within_cap": True,
        },
    }
    audit = audit_adapter_manifest(manifest, adapter_state=state)
    assert audit["method"] == "ground_source_effective_feature_lora_v1"
