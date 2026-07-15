from __future__ import annotations

import torch
import pytest

from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    LATE_FILM_TARGETS,
    FULL_FEATURE_LORA_TARGETS,
    LORA_TARGETS,
    ChannelAffineLinear,
    LATE_KEY_FT_TARGETS,
    LoRALinear,
    _prototype_banks_from_matched_views,
    _validate_deployment_controls,
    build_rx_shift_pair_cycle,
    inject_feat_joint_lora,
    inject_late_channel_film,
    enable_late_key_layer_finetune,
    load_trainable_adapter_state,
    train_support_only_lora,
    view_score_distillation_loss,
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
        self.con_proj = torch.nn.Sequential(torch.nn.Linear(160, 160))
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


def test_full_feature_rank24_uses_relaxed_tier_but_stays_under_100k() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=24, alpha=24.0, scope="full_feature"
    )
    assert audit["trainable_parameters"] == 81_432
    assert audit["adapter_state_bytes_fp16"] == 162_864
    assert audit["adapter_macs_per_query"] == 81_432
    assert audit["scope"] == "full_feature"
    assert audit["resource_tier"] == "performance_relaxed"
    assert audit["trainable_parameter_cap"] == 100_000
    assert len(audit["target_modules"]) == len(FULL_FEATURE_LORA_TARGETS) == 10


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


def test_late_key_layer_finetune_is_exact_31200_parameter_whitelist() -> None:
    model = _Model()
    audit = enable_late_key_layer_finetune(model)
    assert audit["trainable_parameters"] == 31_200
    assert audit["delta_patch_state_bytes_fp16"] == 62_400
    assert audit["adapter_macs_per_query"] == 0
    assert audit["deployment_added_macs_per_query_after_merge"] == 0
    assert audit["checkpoint_update_target_modules"] == list(LATE_KEY_FT_TARGETS)
    assert audit["original_checkpoint_trainable_parameters"] == 31_200
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "id_backbone.t_proj.weight",
        "id_backbone.t_proj.bias",
        "id_backbone.f_proj.weight",
        "id_backbone.f_proj.bias",
        "id_backbone.pa_proj.0.weight",
        "id_backbone.pa_proj.0.bias",
    }
    assert not model.id_backbone.fuse[0].weight.requires_grad
    assert not model.id_backbone.cls_head.head.weight.requires_grad


def test_ground_adapter_state_load_is_strict_and_finite(tmp_path) -> None:
    model = _Model()
    inject_late_channel_film(model)
    expected = {
        name: torch.full_like(parameter, 0.125).half()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state_path = tmp_path / "ground_film.pt"
    torch.save(expected, state_path)
    audit = load_trainable_adapter_state(model, state_path)
    assert audit["mode"] == "ground_source_pretrained"
    assert audit["element_count"] == 1_280
    assert audit["tensor_count"] == 8
    assert audit["strict_key_match"]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, expected[name].float())

    torch.save({**expected, "unexpected": torch.zeros(1)}, state_path)
    with pytest.raises(ValueError, match="key mismatch"):
        load_trainable_adapter_state(model, state_path)


def test_ground_late_key_state_load_is_strict_and_within_patch_budget(tmp_path) -> None:
    model = _Model()
    audit = enable_late_key_layer_finetune(model)
    expected = {
        name: torch.full_like(parameter, 0.0625).half()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state_path = tmp_path / "ground_late_key.pt"
    torch.save(expected, state_path)
    load_audit = load_trainable_adapter_state(model, state_path)
    assert load_audit["element_count"] == 31_200
    assert load_audit["tensor_count"] == 6
    assert audit["delta_patch_state_bytes_fp16"] == 62_400
    assert audit["deployment_added_macs_per_query_after_merge"] == 0


def test_rx_shift_pair_cycle_uses_two_views_per_epoch_and_three_unique_views() -> None:
    import numpy as np

    physical = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    rows = np.concatenate([physical, physical + 100, physical + 200], axis=0)
    labels = np.tile(np.asarray([0, 0, 1, 1]), 3)
    expanded, expanded_labels, audit = build_rx_shift_pair_cycle(
        rows, labels, input_view_count=3
    )
    assert expanded.shape == (5 * 2 * 4, 2, 8)
    assert expanded_labels.shape == (5 * 2 * 4,)
    assert audit["receive_views_per_physical_sample_per_epoch"] == 2
    assert audit["unique_receive_view_names"] == [
        "rx_base",
        "rx_shift_m2",
        "rx_shift_p2",
    ]
    assert [row["scenario"] for row in audit["schedule"]] == [
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
        "leo_clear_weak",
        "leo_low_elev_weak",
    ]
    assert [row["receive_views"][1] for row in audit["schedule"]] == [
        "rx_shift_m2",
        "rx_shift_p2",
        "rx_shift_m2",
        "rx_shift_p2",
        "rx_shift_m2",
    ]


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
        view_score_distill_weight=0.0,
        view_score_distill_temperature=2.0,
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


def test_view_score_distillation_preserves_full_view_ensemble() -> None:
    identical = torch.tensor(
        [[3.0, 1.0], [2.0, 0.0], [3.0, 1.0], [2.0, 0.0]],
        requires_grad=True,
    )
    zero = view_score_distillation_loss(
        identical, view_count=2, temperature=2.0
    )
    assert float(zero.detach()) == pytest.approx(0.0, abs=1.0e-7)

    divergent = torch.tensor(
        [[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]],
        requires_grad=True,
    )
    loss = view_score_distillation_loss(
        divergent, view_count=2, temperature=2.0
    )
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert divergent.grad is not None
    assert bool(torch.isfinite(divergent.grad).all())


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
            "--init_adapter_state",
            "ground_film.pt",
        ]
    )
    assert args.adapter_type == "late_film"
    assert args.epochs == 5
    assert args.optimizer == "sgd"
    assert args.max_optimizer_steps == 50
    assert args.grad_clip == 1.0
    assert args.view_sampling_mode == "rotating_single"
    assert args.matched_view_teacher_weight == 0.25
    assert args.init_adapter_state.name == "ground_film.pt"
    _validate_deployment_controls(args)

    args.optimizer = "adamw"
    with pytest.raises(ValueError, match="sgd_without_moment_state"):
        _validate_deployment_controls(args)


def test_cli_accepts_sparse_key_layer_fast_adaptation_controls() -> None:
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
            "late_key_ft",
            "--epochs",
            "5",
            "--optimizer",
            "sgd",
            "--max_optimizer_steps",
            "50",
            "--view_sampling_mode",
            "rotating_single",
            "--matched_view_teacher_weight",
            "0.25",
            "--init_adapter_state",
            "ground_late_key.pt",
            "--support_view_policy",
            "rx_shift_pair_cycle",
        ]
    )
    _validate_deployment_controls(args)
    assert args.adapter_type == "late_key_ft"
    assert args.init_adapter_state.name == "ground_late_key.pt"
    assert args.support_view_policy == "rx_shift_pair_cycle"


def test_trainer_source_persists_adapter_state_hash_contract() -> None:
    from pathlib import Path

    source = Path(
        "paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py"
    ).read_text(encoding="utf-8")
    assert '"adapter_state_sha256": _sha256_file(adapter_state_path)' in source
