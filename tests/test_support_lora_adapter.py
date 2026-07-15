from __future__ import annotations

import torch
import pytest

from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    LATE_FILM_TARGETS,
    LORA_TARGETS,
    ChannelAffineLinear,
    LoRALinear,
    _prototype_banks_from_matched_views,
    _validate_deployment_controls,
    inject_feat_joint_lora,
    inject_late_channel_film,
    train_support_only_lora,
)


class _Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_proj = torch.nn.Sequential(torch.nn.Linear(160, 160))
        self.pa_proj = torch.nn.Sequential(torch.nn.Linear(320, 160))
        self.id_gate = torch.nn.Sequential(torch.nn.Linear(160, 160))
        self.joint_proj = torch.nn.Sequential(torch.nn.Linear(320, 160))
        self.imp_merge = torch.nn.Sequential(torch.nn.Linear(160, 160))
        self.head = torch.nn.Linear(160, 6, bias=False)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t_proj = torch.nn.Linear(96, 160)
        self.f_proj = torch.nn.Linear(32, 160)
        self.pa_proj = torch.nn.Sequential(torch.nn.Linear(64, 160))
        self.fuse = torch.nn.Sequential(torch.nn.Linear(321, 160))
        self.cls_head = _Head()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _Backbone()


def test_lora_linear_is_exact_identity_at_initialization() -> None:
    base = torch.nn.Linear(7, 5)
    rows = torch.randn(4, 7)
    expected = base(rows).detach()
    adapter = LoRALinear(base, rank=2, alpha=2.0)
    torch.testing.assert_close(adapter(rows), expected, rtol=0.0, atol=0.0)
    assert adapter.trainable_parameter_count == 24
    assert not any(parameter.requires_grad for parameter in adapter.base.parameters())


def test_channel_affine_linear_is_exact_identity_at_initialization() -> None:
    base = torch.nn.Linear(7, 5)
    rows = torch.randn(4, 7)
    expected = base(rows).detach()
    adapter = ChannelAffineLinear(base)
    torch.testing.assert_close(adapter(rows), expected, rtol=0.0, atol=0.0)
    assert adapter.trainable_parameter_count == 10
    assert adapter.added_macs_per_sample == 10
    assert not any(parameter.requires_grad for parameter in adapter.base.parameters())


def test_feat_joint_lora_is_under_caps_and_freezes_original_checkpoint() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(model, rank=8, alpha=8.0)
    assert audit["trainable_parameters"] == 12_800
    assert audit["adapter_state_bytes_fp16"] == 25_600
    assert audit["adapter_macs_per_query"] == 12_800
    assert len(audit["target_modules"]) == len(LORA_TARGETS) == 4
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".lora_" in name for name in trainable)
    assert not model.id_backbone.cls_head.head.weight.requires_grad
    assert not model.id_backbone.cls_head.imp_merge[0].weight.requires_grad


def test_late_feat_joint_rank4_compresses_full_late_scope() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=4, alpha=4.0, scope="late_feat_joint"
    )
    assert audit["trainable_parameters"] == 11_012
    assert audit["adapter_state_bytes_fp16"] == 22_024
    assert audit["adapter_macs_per_query"] == 11_012
    assert audit["scope"] == "late_feat_joint"
    assert len(audit["target_modules"]) == 8


def test_late_channel_film_is_1280_params_and_freezes_checkpoint() -> None:
    model = _Model()
    audit = inject_late_channel_film(model)
    assert audit["trainable_parameters"] == 1_280
    assert audit["adapter_state_bytes_fp16"] == 2_560
    assert audit["adapter_macs_per_query"] == 1_280
    assert audit["scope"] == "late_pooled_projection"
    assert len(audit["target_modules"]) == len(LATE_FILM_TARGETS) == 4
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(
        name.endswith(".film_scale") or name.endswith(".film_bias")
        for name in trainable
    )
    assert not model.id_backbone.cls_head.head.weight.requires_grad


def test_rotating_single_view_film_uses_cached_teacher_and_step_cap(
    monkeypatch,
) -> None:
    import numpy as np
    import paper_reproduction.scripts.train_export_cvs_micro_iq_adapter as micro_trainer
    import paper_reproduction.scripts.train_export_cvs_support_lora_adapter as trainer

    model = _Model()
    inject_late_channel_film(model)

    def fake_feature_forward(current_model, rows):
        signal = rows.mean(dim=(1, 2), keepdim=False)

        def expand(width):
            return signal.unsqueeze(1).expand(-1, width)

        backbone = current_model.id_backbone
        features = (
            backbone.t_proj(expand(96))
            + backbone.f_proj(expand(32))
            + backbone.pa_proj[0](expand(64))
            + backbone.fuse[0](expand(321))
        )
        return features, features[:, :2]

    monkeypatch.setattr(micro_trainer, "_feature_forward", fake_feature_forward)
    monkeypatch.setattr(trainer, "_feature_forward", fake_feature_forward)
    physical = np.asarray(
        [
            np.full((2, 4), -1.0, dtype=np.float32),
            np.full((2, 4), -0.8, dtype=np.float32),
            np.full((2, 4), 0.8, dtype=np.float32),
            np.full((2, 4), 1.0, dtype=np.float32),
        ]
    )
    support_rows = np.concatenate(
        [physical, physical * 0.9, physical * 1.1], axis=0
    )
    support_labels = np.tile(np.asarray([0, 0, 1, 1]), 3)
    trace, runtime = train_support_only_lora(
        model,
        support_rows,
        support_labels,
        epochs=3,
        learning_rate=3.0e-3,
        weight_decay=1.0e-4,
        temperature=8.0,
        feature_anchor_weight=0.05,
        view_consistency_weight=0.0,
        cross_view_prototype_weight=0.0,
        cosine_margin=0.0,
        class_dro_temperature=0.0,
        support_view_count=3,
        batch_size=4,
        optimizer_name="sgd",
        max_optimizer_steps=2,
        grad_clip=1.0,
        view_sampling_mode="rotating_single",
        matched_view_teacher_weight=0.25,
        seed=713101,
        device=torch.device("cpu"),
    )
    assert [row["train_view_index"] for row in trace] == [0, 1]
    assert runtime["optimizer"] == "sgd"
    assert runtime["optimizer_steps"] == 2
    assert runtime["optimizer_training_state_bytes_estimate"] == 0
    assert runtime["teacher_precompute_view_count"] == 3
    assert runtime["train_views_per_physical_sample_per_epoch"] == 1
    assert runtime["support_forward_sample_equivalents"] == 28
    assert runtime["terminated_by_step_cap"]


def test_leave_one_view_out_prototypes_exclude_current_view() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.4, 0.6],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    all_view, leave_one_out = _prototype_banks_from_matched_views(
        features, labels, class_count=2, view_count=3
    )
    assert all_view.shape == (2, 2)
    assert leave_one_out.shape == (3, 2, 2)
    expected_class0_without_view0 = torch.nn.functional.normalize(
        torch.nn.functional.normalize(features[[2, 4]], dim=1).mean(dim=0), dim=0
    )
    torch.testing.assert_close(
        leave_one_out[0, 0], expected_class0_without_view0
    )
    assert torch.allclose(torch.linalg.norm(leave_one_out, dim=-1), torch.ones(3, 2))


def test_cli_accepts_class_symmetric_dro_controls() -> None:
    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import parse_args

    args = parse_args(
        [
            "--config",
            "config.json",
            "--ckpt",
            "model.pth",
            "--out_root",
            "out",
            "--receiver",
            "8-8",
            "--new_count",
            "20",
            "--seed",
            "713101",
            "--cosine_margin",
            "0.1",
            "--class_dro_temperature",
            "5",
            "--cross_view_prototype_weight",
            "1",
        ]
    )
    assert args.cosine_margin == 0.1
    assert args.class_dro_temperature == 5.0
    assert args.cross_view_prototype_weight == 1.0


def test_cli_accepts_late_film_fast_adaptation_controls() -> None:
    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import parse_args

    args = parse_args(
        [
            "--config",
            "config.json",
            "--ckpt",
            "model.pth",
            "--out_root",
            "out",
            "--receiver",
            "8-8",
            "--new_count",
            "20",
            "--seed",
            "713101",
            "--adapter_type",
            "late_film",
            "--epochs",
            "5",
            "--optimizer",
            "sgd",
            "--max_optimizer_steps",
            "50",
            "--grad_clip",
            "1",
            "--view_sampling_mode",
            "rotating_single",
            "--matched_view_teacher_weight",
            "0.25",
        ]
    )
    assert args.adapter_type == "late_film"
    assert args.epochs == 5
    assert args.optimizer == "sgd"
    assert args.max_optimizer_steps == 50
    assert args.grad_clip == 1.0
    assert args.view_sampling_mode == "rotating_single"
    assert args.matched_view_teacher_weight == 0.25
    _validate_deployment_controls(args)

    args.optimizer = "adamw"
    with pytest.raises(ValueError, match="sgd_without_moment_state"):
        _validate_deployment_controls(args)
