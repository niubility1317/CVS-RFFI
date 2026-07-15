from __future__ import annotations

import torch
import pytest

from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    EFFECTIVE_FEATURE_LORA_TARGETS,
    LATE_FILM_TARGETS,
    FULL_FEATURE_LORA_TARGETS,
    JOINT_GATE_LORA_TARGETS,
    JOINT_PROJECTION_LORA_TARGETS,
    PROJECTION_LORA_TARGETS,
    LORA_TARGETS,
    ChannelAffineLinear,
    LATE_KEY_FT_TARGETS,
    LoRALinear,
    _prototype_banks_from_matched_views,
    _validate_deployment_controls,
    bp_jg_episode_loss,
    build_support_run_id,
    build_shot_index_episode_positions,
    build_rx_shift_pair_cycle,
    inject_feat_joint_lora,
    merge_feat_joint_lora,
    inject_late_channel_film,
    enable_late_key_layer_finetune,
    load_trainable_adapter_state,
    load_and_merge_ground_lora,
    roundtrip_fp16_target_lora_and_merge,
    train_support_only_bp_jg,
    train_support_only_lora,
    validate_bp_jg_qknn_config,
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


def test_lora_linear_inherits_base_device_and_dtype() -> None:
    base = torch.nn.Linear(7, 5).to(dtype=torch.float64)
    rows = torch.randn(4, 7, dtype=torch.float64)
    expected = base(rows).detach()
    adapter = LoRALinear(base, rank=2, alpha=2.0)
    assert adapter.lora_a.weight.device == base.weight.device
    assert adapter.lora_b.weight.device == base.weight.device
    assert adapter.lora_a.weight.dtype == base.weight.dtype
    assert adapter.lora_b.weight.dtype == base.weight.dtype
    torch.testing.assert_close(adapter(rows), expected, rtol=0.0, atol=0.0)


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


def test_effective_feature_rank16_removes_dead_branches_and_stays_preferred() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=16, alpha=16.0, scope="effective_feature"
    )
    assert audit["trainable_parameters"] == 44_048
    assert audit["adapter_state_bytes_fp16"] == 88_096
    assert audit["adapter_macs_per_query"] == 44_048
    assert audit["scope"] == "effective_feature"
    assert audit["resource_tier"] == "preferred"
    assert audit["trainable_parameter_cap"] == 50_000
    modules = [row["module"] for row in audit["target_modules"]]
    assert modules == list(EFFECTIVE_FEATURE_LORA_TARGETS)
    assert "id_backbone.con_proj.0" not in modules
    assert "id_backbone.cls_head.imp_merge.0" not in modules


def test_projection_feature_rank16_updates_only_four_pooled_projection_layers() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=16, alpha=16.0, scope="projection_feature"
    )
    assert audit["trainable_parameters"] == 18_448
    assert audit["adapter_state_bytes_fp16"] == 36_896
    assert audit["adapter_macs_per_query"] == 18_448
    assert audit["scope"] == "projection_feature"
    assert [row["module"] for row in audit["target_modules"]] == list(
        PROJECTION_LORA_TARGETS
    )


def test_target_joint_gate_rank8_is_exactly_6400_parameters() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=8, alpha=8.0, scope="joint_gate"
    )
    assert audit["trainable_parameters"] == 6_400
    assert audit["adapter_state_bytes_fp16"] == 12_800
    assert audit["adapter_macs_per_query"] == 6_400
    assert [row["module"] for row in audit["target_modules"]] == list(
        JOINT_GATE_LORA_TARGETS
    )


def test_target_joint_projection_rank8_is_exactly_3840_parameters() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=8, alpha=8.0, scope="joint_projection"
    )
    assert audit["trainable_parameters"] == 3_840
    assert audit["adapter_state_bytes_fp16"] == 7_680
    assert [row["module"] for row in audit["target_modules"]] == list(
        JOINT_PROJECTION_LORA_TARGETS
    )


def test_ground_p4_is_loaded_merged_then_target_jg_is_injected(tmp_path) -> None:
    source = _Model()
    inject_feat_joint_lora(
        source, rank=16, alpha=16.0, scope="projection_feature"
    )
    state = {
        name: torch.full_like(parameter, 0.01).half()
        for name, parameter in source.named_parameters()
        if parameter.requires_grad
    }
    state_path = tmp_path / "ground_p4.pt"
    torch.save(state, state_path)

    model = _Model()
    ground = load_and_merge_ground_lora(
        model,
        state_path,
        scope="projection_feature",
        rank=16,
        alpha=16.0,
    )
    assert ground["resources"]["trainable_parameters"] == 18_448
    assert ground["merge"]["merged_module_count"] == 4
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    target = inject_feat_joint_lora(
        model, rank=8, alpha=8.0, scope="joint_gate"
    )
    assert target["trainable_parameters"] == 6_400
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(
        ("cls_head.id_gate.0" in name or "cls_head.joint_proj.0" in name)
        and ".lora_" in name
        for name in trainable
    )


def test_lora_merge_removes_wrappers_and_preserves_outputs() -> None:
    model = _Model()
    inject_feat_joint_lora(model, rank=4, alpha=4.0, scope="effective_feature")
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_b.weight.normal_(0.0, 0.01)
    audit = merge_feat_joint_lora(model)
    assert audit["merged_module_count"] == 8
    assert audit["remaining_lora_wrappers"] == []
    assert audit["algebraic_probe_parity_pass"] is True
    assert not any(isinstance(module, LoRALinear) for module in model.modules())


def test_lora_merge_rejects_nan_delta() -> None:
    model = _Model()
    inject_feat_joint_lora(model, rank=8, alpha=8.0, scope="joint_projection")
    with torch.no_grad():
        model.id_backbone.cls_head.joint_proj[0].lora_b.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="parity failed"):
        merge_feat_joint_lora(model)


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


def test_ground_lora_sha_is_checked_before_load(tmp_path) -> None:
    import hashlib

    model = _Model()
    inject_feat_joint_lora(model, rank=16, alpha=16.0, scope="projection_feature")
    state = {
        name: parameter.detach().half()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state_path = tmp_path / "ground_p4.pt"
    torch.save(state, state_path)
    expected_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()

    audit = load_and_merge_ground_lora(
        _Model(),
        state_path,
        scope="projection_feature",
        rank=16,
        alpha=16.0,
        expected_sha256=expected_sha256,
    )
    assert audit["sha256_preload_match"] is True
    assert audit["observed_sha256_before_load"] == expected_sha256

    with pytest.raises(ValueError, match="mismatch before load"):
        load_and_merge_ground_lora(
            _Model(),
            state_path,
            scope="projection_feature",
            rank=16,
            alpha=16.0,
            expected_sha256="0" * 64,
        )


def test_target_fp16_artifact_is_reloaded_before_merge(tmp_path) -> None:
    import copy

    torch.manual_seed(713101)
    model = _Model()
    inject_feat_joint_lora(model, rank=8, alpha=8.0, scope="joint_gate")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(torch.randn_like(parameter) * 0.017)
    state = {
        name: parameter.detach().cpu().half()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state_path = tmp_path / "target_fp16.pt"
    torch.save(state, state_path)

    expected = copy.deepcopy(model)
    load_trainable_adapter_state(expected, state_path)
    merge_feat_joint_lora(expected)

    audit = roundtrip_fp16_target_lora_and_merge(model, state_path)
    assert audit["mode"] == "fp16_artifact_roundtrip_then_merge"
    assert audit["state_roundtrip"]["strict_key_match"] is True
    assert audit["merge"]["remaining_lora_wrappers"] == []
    assert audit["merge"]["post_merge_trainable_parameters"] == 0
    assert not any(parameter.requires_grad for parameter in model.parameters())
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, expected.state_dict()[name])


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


def test_maxk20_split_is_nested_and_uses_one_disjoint_query_set() -> None:
    import numpy as np
    from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
        assemble_support_views,
    )
    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
        SCENARIOS,
    )

    roles: list[str] = []
    labels: list[str] = []
    sample_numbers: list[int] = []
    for role, label in (("target_old", "old0"), ("target_new", "new0")):
        for sample_number in range(45):
            roles.append(role)
            labels.append(label)
            sample_numbers.append(sample_number)
    count = len(roles)
    caches = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        caches[scenario] = {
            "raw_iq": np.full(
                (count, 2, 8), float(scenario_index), dtype=np.float32
            ),
            "dataset_role": np.asarray(roles),
            "tx_ids": np.asarray(labels),
            "rx_ids": np.asarray(["rx0"] * count),
            "day_ids": np.asarray([f"d{value}" for value in sample_numbers]),
            "eq_ids": np.asarray([f"e{value}" for value in sample_numbers]),
            "sig_ids": np.asarray([f"s{value}" for value in sample_numbers]),
            "sat_scenarios": np.asarray([scenario] * count),
        }

    manifests = {}
    for k_shot in (1, 5, 10, 20):
        _, _, manifests[k_shot] = assemble_support_views(
            caches,
            receiver="rx0",
            old_labels=["old0"],
            new_labels=["new0"],
            seed=713101,
            k_shot=k_shot,
            support_pool_max_k=20,
            query_per_tx=20,
        )
    support_sets = {
        k_shot: set(manifest["physical_support_ids"])
        for k_shot, manifest in manifests.items()
    }
    assert support_sets[1] < support_sets[5] < support_sets[10] < support_sets[20]
    query_ids = [manifests[k_shot]["physical_query_ids"] for k_shot in (1, 5, 10, 20)]
    assert query_ids[1:] == [query_ids[0], query_ids[0], query_ids[0]]
    assert not support_sets[20].intersection(query_ids[0])


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


def test_shot_index_episodes_cover_k20_in_ten_balanced_steps() -> None:
    physical_labels = torch.arange(3).repeat_interleave(20)
    labels = physical_labels.repeat(3)
    episodes = build_shot_index_episode_positions(
        labels, view_count=3, max_episodes_per_epoch=10
    )
    assert len(episodes) == 10
    assert all(int(episode.numel()) == 3 * 3 * 2 for episode in episodes)
    first_view = torch.cat([episode[:6] for episode in episodes])
    torch.testing.assert_close(
        torch.sort(first_view).values, torch.arange(60), rtol=0.0, atol=0.0
    )
    for episode in episodes:
        per_view_labels = labels[episode].reshape(3, -1)
        assert torch.equal(per_view_labels[0], per_view_labels[1])
        assert torch.equal(per_view_labels[0], per_view_labels[2])
        assert torch.bincount(per_view_labels[0], minlength=3).tolist() == [2, 2, 2]


def test_bp_jg_episode_loss_is_finite_symmetric_and_differentiable() -> None:
    torch.manual_seed(7)
    physical_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    labels = physical_labels.repeat(3)
    base = torch.randn(18, 12)
    features = (base + 0.03 * torch.randn_like(base)).requires_grad_(True)
    losses = bp_jg_episode_loss(
        features,
        base,
        labels,
        view_count=3,
        temperature=18.0,
    )
    expected = {
        "loss",
        "xview_prototype_ce",
        "boundary_margin_loss",
        "feature_anchor_loss",
        "prototype_gram_loss",
        "prototype_separation_loss",
        "view_consistency_loss",
        "mean_margin",
        "mean_base_margin",
        "correct",
        "sample_count",
    }
    assert set(losses) == expected
    assert all(bool(torch.isfinite(value).all()) for value in losses.values())
    assert int(losses["sample_count"]) == 18
    class_permutation = torch.tensor([2, 0, 1])
    permuted = bp_jg_episode_loss(
        features,
        base,
        class_permutation[labels],
        view_count=3,
        temperature=18.0,
    )
    for key in expected:
        torch.testing.assert_close(losses[key], permuted[key])
    losses["loss"].backward()
    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())


def test_bp_jg_k1_uses_five_steps_and_zero_optimizer_state(monkeypatch) -> None:
    import numpy as np
    import paper_reproduction.scripts.train_export_cvs_micro_iq_adapter as micro_trainer
    import paper_reproduction.scripts.train_export_cvs_support_lora_adapter as trainer

    model = _Model()
    inject_feat_joint_lora(model, rank=8, alpha=8.0, scope="joint_gate")

    def fake_feature_forward(current_model, rows):
        signal = rows.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        identity = signal.expand(-1, 160)
        gate = current_model.id_backbone.cls_head.id_gate[0](identity)
        joint = current_model.id_backbone.cls_head.joint_proj[0](
            torch.cat([identity, gate], dim=1)
        )
        return joint, joint[:, :2]

    monkeypatch.setattr(micro_trainer, "_feature_forward", fake_feature_forward)
    monkeypatch.setattr(trainer, "_feature_forward", fake_feature_forward)
    physical = np.asarray(
        [
            np.full((2, 4), -1.0, dtype=np.float32),
            np.full((2, 4), 1.0, dtype=np.float32),
        ]
    )
    support_rows = np.concatenate(
        [physical, physical * 0.9, physical * 1.1], axis=0
    )
    support_labels = np.tile(np.asarray([0, 1]), 3)
    trace, runtime = train_support_only_bp_jg(
        model,
        support_rows,
        support_labels,
        physical_support_ids=["c0-shot0", "c1-shot0"],
        support_row_physical_ids=["c0-shot0", "c1-shot0"] * 3,
        epochs=5,
        learning_rate=5.0e-3,
        weight_decay=1.0e-4,
        temperature=18.0,
        support_view_count=3,
        batch_size=32,
        max_optimizer_steps=50,
        grad_clip=1.0,
        seed=713101,
        device=torch.device("cpu"),
    )
    assert len(trace) == 5
    assert runtime["optimizer_steps"] == 5
    assert runtime["episodes_per_epoch"] == 1
    assert runtime["k_shot_inferred"] == 1
    assert runtime["optimizer_training_state_bytes_estimate"] == 0
    assert runtime["support_forward_sample_equivalents"] == 36
    assert runtime["query_rows_used_for_training"] == 0
    assert runtime["old_new_role_used_by_optimizer"] is False
    assert runtime["class_quota_used_by_optimizer"] is False
    assert runtime["dense_query_graph_used"] is False
    assert runtime["physical_support_ids_unique"] is True
    assert len(runtime["physical_support_ids_sha256"]) == 64
    assert runtime["train_receive_views_per_physical_sample_per_epoch"] == 3

    with pytest.raises(ValueError, match="non-empty and unique"):
        train_support_only_bp_jg(
            model,
            support_rows,
            support_labels,
            physical_support_ids=["duplicate", "duplicate"],
            support_row_physical_ids=["duplicate", "duplicate"] * 3,
            epochs=5,
            learning_rate=5.0e-3,
            weight_decay=1.0e-4,
            temperature=18.0,
            support_view_count=3,
            batch_size=32,
            max_optimizer_steps=50,
            grad_clip=1.0,
            seed=713101,
            device=torch.device("cpu"),
        )

    with pytest.raises(ValueError, match="alignment drift"):
        train_support_only_bp_jg(
            model,
            support_rows,
            support_labels,
            physical_support_ids=["c0-shot0", "c1-shot0"],
            support_row_physical_ids=[
                "c0-shot0",
                "c1-shot0",
                "c1-shot0",
                "c0-shot0",
                "c0-shot0",
                "c1-shot0",
            ],
            epochs=5,
            learning_rate=5.0e-3,
            weight_decay=1.0e-4,
            temperature=18.0,
            support_view_count=3,
            batch_size=32,
            max_optimizer_steps=50,
            grad_clip=1.0,
            seed=713101,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("k_shot", "expected_steps", "expected_episodes"),
    [(5, 25, 5), (10, 50, 10), (20, 50, 10)],
)
def test_bp_jg_k_grid_executes_locked_step_budget(
    monkeypatch, k_shot: int, expected_steps: int, expected_episodes: int
) -> None:
    import numpy as np
    import paper_reproduction.scripts.train_export_cvs_micro_iq_adapter as micro_trainer
    import paper_reproduction.scripts.train_export_cvs_support_lora_adapter as trainer

    model = _Model()
    inject_feat_joint_lora(model, rank=8, alpha=8.0, scope="joint_gate")

    def fake_feature_forward(current_model, rows):
        signal = rows.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        identity = signal.expand(-1, 160)
        gate = current_model.id_backbone.cls_head.id_gate[0](identity)
        joint = current_model.id_backbone.cls_head.joint_proj[0](
            torch.cat([identity, gate], dim=1)
        )
        return joint, joint[:, :2]

    monkeypatch.setattr(micro_trainer, "_feature_forward", fake_feature_forward)
    monkeypatch.setattr(trainer, "_feature_forward", fake_feature_forward)
    physical_labels = np.repeat(np.asarray([0, 1]), k_shot)
    signal = np.where(physical_labels == 0, -1.0, 1.0).astype(np.float32)
    physical = np.repeat(signal[:, None, None], 2 * 4, axis=1).reshape(-1, 2, 4)
    support_rows = np.concatenate([physical, physical * 0.9, physical * 1.1])
    support_labels = np.tile(physical_labels, 3)
    physical_ids = [
        f"c{class_index}-shot{shot_index}"
        for class_index in (0, 1)
        for shot_index in range(k_shot)
    ]
    _, runtime = train_support_only_bp_jg(
        model,
        support_rows,
        support_labels,
        physical_support_ids=physical_ids,
        support_row_physical_ids=physical_ids * 3,
        epochs=5,
        learning_rate=5.0e-3,
        weight_decay=1.0e-4,
        temperature=18.0,
        support_view_count=3,
        batch_size=256,
        max_optimizer_steps=50,
        grad_clip=1.0,
        seed=713101,
        device=torch.device("cpu"),
    )
    assert runtime["optimizer_steps"] == expected_steps
    assert runtime["episodes_per_epoch"] == expected_episodes
    assert runtime["support_forward_sample_equivalents"] == 36 * k_shot


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


def test_cli_locks_bp_jg_p4_and_five_epoch_controls() -> None:
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
            "--k_shot",
            "1",
            "--adapt_objective",
            "bp_jg",
            "--scope",
            "joint_gate",
            "--rank",
            "8",
            "--alpha",
            "8",
            "--epochs",
            "5",
            "--optimizer",
            "sgd",
            "--max_optimizer_steps",
            "50",
            "--learning_rate",
            "0.005",
            "--weight_decay",
            "0.0001",
            "--temperature",
            "18",
            "--grad_clip",
            "1",
            "--view_sampling_mode",
            "shot_index",
            "--ground_adapter_state",
            "ground_p4.pt",
            "--ground_adapter_sha256",
            "0" * 64,
        ]
    )
    _validate_deployment_controls(args)
    assert args.k_shot == 1
    assert args.scope == "joint_gate"
    assert args.ground_adapter_scope == "projection_feature"
    assert args.ground_adapter_rank == 16
    run_id = build_support_run_id(args)
    assert "joint_gate_r8" in run_id
    assert "000000000000" in run_id
    import copy

    rank16 = copy.copy(args)
    rank16.rank = 16
    rank16.alpha = 16.0
    assert build_support_run_id(rank16) != run_id
    other_ground = copy.copy(args)
    other_ground.ground_adapter_sha256 = "1" * 64
    assert build_support_run_id(other_ground) != run_id
    args.view_sampling_mode = "stacked"
    with pytest.raises(ValueError, match="bp_jg_shot_index_episodes"):
        _validate_deployment_controls(args)


def test_bp_jg_qknn_config_forbids_legacy_head_and_old_bias() -> None:
    config = {
        "support_pool_max_k": 20,
        "qknnv42_head_mode": "qknn",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_old_anchor_bias": 0.0,
    }
    validate_bp_jg_qknn_config(config)
    for key, value in (
        ("qknnv42_head_mode", "extreme_light_diag_cosine"),
        ("qknnv42_old_anchor_bias", 0.001),
        ("qknnv42_feature_adapter_mode", "support_diag_whiten"),
    ):
        invalid = dict(config)
        invalid[key] = value
        with pytest.raises(ValueError, match="class-symmetric qKNN"):
            validate_bp_jg_qknn_config(invalid)
    legacy = dict(config, primary_k_shot=10, sensitivity_k_shot=5)
    with pytest.raises(ValueError, match="legacy K10/K5"):
        validate_bp_jg_qknn_config(legacy)


def test_p4_identity_control_is_zero_update_and_collision_safe() -> None:
    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
        parse_args,
    )

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
            "--k_shot",
            "10",
            "--adapt_objective",
            "p4_identity",
            "--epochs",
            "0",
            "--optimizer",
            "sgd",
            "--max_optimizer_steps",
            "0",
            "--ground_adapter_state",
            "ground_p4.pt",
            "--ground_adapter_sha256",
            "a" * 64,
        ]
    )
    _validate_deployment_controls(args)
    assert args.adapt_objective == "p4_identity"
    run_id = build_support_run_id(args)
    assert "p4_aaaaaaaaaaaa_identity" in run_id
    assert "rx_8-8_new_20_seed_713101_k_10" in run_id
    args.epochs = 1
    with pytest.raises(ValueError, match="p4_identity_zero_epochs"):
        _validate_deployment_controls(args)


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
