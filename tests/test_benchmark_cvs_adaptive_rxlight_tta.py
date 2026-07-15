from __future__ import annotations

import numpy as np
import pytest
import torch

from paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta import (
    _reference_parity,
    apply_fp16_checkpoint_delta,
    apply_fp16_lora_state,
    audit_adapter_manifest,
    build_view_prototypes,
    leave_one_out_support_scores,
    score_views,
)


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
