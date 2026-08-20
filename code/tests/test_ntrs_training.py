from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from SSDG.train_ssdg import (
    build_arg_parser,
    build_optimizer_with_crra_groups,
)
from cvsrffi.losses import (
    ntrs_class_attraction_loss,
    ntrs_class_conditional_alignment_loss,
    ntrs_conditional_decorrelation_loss,
    ntrs_correctability_loss,
    ntrs_margin_preservation_loss,
    ntrs_relation_distillation_loss,
    ntrs_score_stability_loss,
    ntrs_shared_receiver_offset_loss,
)
from cvsrffi.ntrs_training import (
    compute_ntrs_loss_bundle,
    ntrs_source_update_mask,
    ntrs_stage_code,
    ntrs_relative_correction_loss,
    ntrs_training_stage,
    set_ntrs_optimizer_learning_rates,
    validate_ntrs_phase1_config,
    validate_ntrs_phase1_scenarios,
)
from model_dual_cvsincnet import build_dual_model


def test_ntrs_defaults_match_the_frozen_first_version_report():
    args = build_arg_parser().parse_args(["--output_dir", "x"])

    assert args.use_ntrs is False
    assert args.ntrs_rank == 8
    assert args.ntrs_alpha_max == pytest.approx(0.20)
    assert args.ntrs_slow_ema_decay == pytest.approx(0.95)
    assert args.ntrs_support_tau == pytest.approx(1.0)
    assert args.ntrs_energy_threshold == pytest.approx(0.10)
    assert args.ntrs_unknown_rescue is False
    assert args.ntrs_target_adapter is False
    assert args.lambda_ntrs_sat_kl == pytest.approx(0.01)
    assert args.lambda_ntrs_margin == pytest.approx(0.03)
    assert args.lambda_ntrs_relation == pytest.approx(0.02)
    assert args.lambda_ntrs_cond_decorr == pytest.approx(0.01)
    assert args.lambda_ntrs_min_correction == pytest.approx(0.001)
    assert args.lambda_ntrs_subspace == pytest.approx(0.02)
    assert args.lambda_ntrs_correctability == pytest.approx(0.02)


def test_ntrs_stage_contract_and_optimizer_ratio_are_explicit():
    assert ntrs_training_stage(1).name == "S1"
    assert ntrs_training_stage(16).ntrs_lr_scale == 0.0
    assert ntrs_training_stage(17).name == "S2-a"
    assert ntrs_training_stage(40).geometry_scale == 0.0
    assert ntrs_training_stage(41).name == "S2-b"
    assert ntrs_training_stage(68).geometry_scale == 1.0
    assert ntrs_training_stage(69).name == "S3"
    assert ntrs_training_stage(200).safety_scale == 1.0

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = nn.Linear(2, 2)
            self.ntrs_context = nn.Linear(2, 2)

    optimizer = build_optimizer_with_crra_groups(
        TinyModel(),
        base_lr=2e-4,
        weight_decay=1e-4,
        use_crra=False,
        use_ntrs=True,
    )
    rates = set_ntrs_optimizer_learning_rates(optimizer, epoch=69, base_lr=2e-4)
    assert rates == {"core": pytest.approx(2e-5), "ntrs": pytest.approx(1e-4)}
    assert rates["ntrs"] / rates["core"] == pytest.approx(5.0)


def test_ntrs_v2_schedule_keeps_core_fair_and_delays_residual_until_epoch_91():
    assert ntrs_training_stage(1, variant="v2_min").name == "V2-S0"
    assert ntrs_training_stage(90, variant="v2_min").ntrs_lr_scale == 0.0
    assert ntrs_training_stage(91, variant="v2_min").name == "V2-RAMP"
    assert ntrs_training_stage(91, variant="v2_min").ntrs_lr_scale == pytest.approx(1.0)
    assert 0.0 < ntrs_training_stage(110, variant="v2_min").geometry_scale < 1.0
    assert ntrs_training_stage(130, variant="v2_min").ntrs_lr_scale == pytest.approx(1.0)
    assert ntrs_training_stage(200, variant="v2_min").core_lr_scale == pytest.approx(1.0)
    assert [ntrs_stage_code(ntrs_training_stage(epoch, variant="v2_min")) for epoch in (1, 91, 131)] == [5, 6, 7]


def test_ntrs_core_lr_mode_separates_fair_and_historical_controls():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = nn.Linear(2, 2)
            self.ntrs_context = nn.Linear(2, 2)

    optimizer = build_optimizer_with_crra_groups(
        TinyModel(),
        base_lr=2e-4,
        weight_decay=1e-4,
        use_crra=False,
        use_ntrs=True,
    )
    fair = set_ntrs_optimizer_learning_rates(
        optimizer,
        epoch=69,
        base_lr=2e-4,
        variant="v1",
        core_lr_mode="baseline",
    )
    assert fair == {"core": pytest.approx(2e-4), "ntrs": pytest.approx(1e-4)}

    historical = set_ntrs_optimizer_learning_rates(
        optimizer,
        epoch=69,
        base_lr=2e-4,
        variant="v1",
        core_lr_mode="v1",
    )
    assert historical == {"core": pytest.approx(2e-5), "ntrs": pytest.approx(1e-4)}


def test_source_update_mask_uses_only_clean_source_rows_for_concat_pairs():
    mask = ntrs_source_update_mask(batch_size=8, clean_count=4, concat_expanded=True)
    assert mask.tolist() == [True, True, True, True, False, False, False, False]
    assert ntrs_source_update_mask(batch_size=4, clean_count=0, concat_expanded=False).all()


def test_margin_and_relation_losses_preserve_geometry_without_pointwise_collapse():
    clean = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    same = clean.clone().detach().requires_grad_(True)
    collapsed = torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    prototypes = torch.eye(2)

    margin_same, _ = ntrs_margin_preservation_loss(clean, same, labels, prototypes, epsilon=0.01)
    margin_bad, _ = ntrs_margin_preservation_loss(clean, collapsed, labels, prototypes, epsilon=0.01)
    relation_same, _ = ntrs_relation_distillation_loss(clean, same)
    relation_bad, _ = ntrs_relation_distillation_loss(clean, collapsed)

    assert float(margin_same.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(margin_bad.detach()) > 0.0
    assert float(relation_same.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(relation_bad.detach()) > 0.0
    (margin_bad + relation_bad).backward()
    assert collapsed.grad is not None


def test_v2_direct_residual_penalty_is_relative_to_anchor_norm():
    anchor = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    candidate = torch.tensor([[0.3, 0.4], [0.0, 0.2]], requires_grad=True)
    loss = ntrs_relative_correction_loss(anchor, candidate)
    assert float(loss.detach()) == pytest.approx(0.01, rel=1e-5)
    loss.backward()
    assert candidate.grad is not None


def test_class_conditional_alignment_is_zero_for_identical_views_and_connected():
    clean = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    satellite = clean.detach().clone().requires_grad_(True)

    loss, info = ntrs_class_conditional_alignment_loss(clean, satellite, labels)

    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6)
    assert info["active_classes"] == 2.0
    loss.backward()
    assert satellite.grad is not None


def test_conditional_decorrelation_and_shared_receiver_offset_are_finite():
    z_id = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]],
        requires_grad=True,
    )
    z_dom = torch.tensor(
        [[0.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, -1.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    receiver = torch.tensor([0, 1, 0, 1])

    cond, cond_info = ntrs_conditional_decorrelation_loss(z_id, z_dom, labels)
    shared, shared_info = ntrs_shared_receiver_offset_loss(z_id, labels, receiver)

    assert torch.isfinite(cond)
    assert torch.isfinite(shared)
    assert cond_info["active_classes"] == 2.0
    assert shared_info["active_receivers"] == 2.0
    (cond + shared).backward()
    assert z_id.grad is not None
    assert z_dom.grad is not None


def test_correctability_target_is_derived_from_per_sample_source_improvement():
    raw_logits = torch.tensor([[0.2, 2.0], [2.0, 0.2]])
    robust_logits = torch.tensor([[3.0, 0.1], [0.2, 2.0]])
    labels = torch.tensor([0, 0])
    predicted = torch.tensor([0.8, 0.2], requires_grad=True)

    loss, target, info = ntrs_correctability_loss(
        raw_logits,
        robust_logits,
        labels,
        predicted,
        improvement_epsilon=0.01,
    )

    assert target.tolist() == [1.0, 0.0]
    assert info["positive_rate"] == pytest.approx(0.5)
    loss.backward()
    assert predicted.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression requires a CUDA device")
def test_correctability_probability_loss_is_safe_under_cuda_amp_autocast():
    device = torch.device("cuda")
    raw_logits = torch.tensor([[0.2, 2.0], [2.0, 0.2]], device=device)
    robust_logits = torch.tensor([[3.0, 0.1], [0.2, 2.0]], device=device)
    labels = torch.tensor([0, 0], device=device)
    predicted = torch.tensor([0.8, 0.2], device=device, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, target, _info = ntrs_correctability_loss(
            raw_logits,
            robust_logits,
            labels,
            predicted,
            improvement_epsilon=0.01,
        )

    assert target.tolist() == [1.0, 0.0]
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_open_set_safety_penalizes_unsafe_knownness_gain_and_class_attraction():
    raw = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    robust = torch.tensor([[5.0, 0.0], [2.5, 0.0]], requires_grad=True)
    energy = torch.tensor([0.2, 0.01])
    stability, info = ntrs_score_stability_loss(
        raw,
        robust,
        correction_energy=energy,
        energy_threshold=0.10,
    )
    anchor = torch.tensor([[0.0, 0.0]])
    correction = torch.tensor([[-1.0, 0.0]], requires_grad=True)
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    attraction, attraction_info = ntrs_class_attraction_loss(
        anchor,
        correction,
        raw_logits=torch.tensor([[2.0, 0.0]]),
        prototypes=prototypes,
        max_cosine=0.5,
    )

    assert float(stability.detach()) > 0.0
    assert info["unsafe_count"] == 1.0
    assert float(attraction.detach()) > 0.0
    assert attraction_info["active_count"] == 1.0
    (stability + attraction).backward()
    assert robust.grad is not None
    assert correction.grad is not None


def test_phase1_ntrs_configuration_accepts_only_the_safe_leo_weak_first_version():
    args = SimpleNamespace(
        use_ntrs=True,
        use_crra=False,
        ntrs_target_adapter=False,
        ntrs_unknown_rescue=False,
        ntrs_rank=8,
        ntrs_alpha_max=0.20,
        ntrs_support_tau=1.0,
        ntrs_energy_threshold=0.10,
        ntrs_slow_ema_decay=0.95,
        lambda_ntrs_margin=0.03,
    )
    validate_ntrs_phase1_config(args)
    validate_ntrs_phase1_scenarios(
        ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
    )


def test_ntrs_loss_bundle_connects_all_four_report_groups():
    def _out(offset):
        anchor = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
            requires_grad=True,
        )
        robust = (anchor + offset).requires_grad_(True)
        raw_logits = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [2.5, 0.2], [0.2, 2.5]],
            requires_grad=True,
        )
        robust_logits = (raw_logits + torch.tensor([[0.2, 0.0]])).requires_grad_(True)
        return {
            "tx_logits": robust_logits,
            "z_id": robust,
            "z_dom": torch.flip(anchor, dims=[1]),
            "ntrs_raw_logits": raw_logits,
            "ntrs_robust_logits": robust_logits,
            "ntrs_z_anchor": anchor,
            "ntrs_z_rob": robust,
            "ntrs_receiver_logits": torch.randn(4, 3, requires_grad=True),
            "ntrs_day_logits": torch.randn(4, 3, requires_grad=True),
            "ntrs_channel_logits": torch.randn(4, 2, requires_grad=True),
            "ntrs_context_tx_adv_logits": torch.randn(4, 2, requires_grad=True),
            "ntrs_alpha": torch.full((4,), 0.1, requires_grad=True),
            "ntrs_correctability": torch.full((4,), 0.5, requires_grad=True),
            "ntrs_correction": torch.full((4, 2), 0.01, requires_grad=True),
            "ntrs_correction_energy": torch.full((4,), 0.01, requires_grad=True),
            "ntrs_subspace_residual": torch.zeros(4, requires_grad=True),
        }

    labels = torch.tensor([0, 1, 0, 1])
    receiver = torch.tensor([0, 0, 1, 1])
    day = torch.tensor([0, 1, 0, 1])
    clean = _out(0.0)
    satellite = _out(0.05)
    bundle = compute_ntrs_loss_bundle(
        clean,
        satellite,
        clean_labels=labels,
        satellite_labels=labels,
        clean_receivers=receiver,
        satellite_receivers=receiver,
        clean_days=day,
        satellite_days=day,
        clean_channels=torch.zeros(4, dtype=torch.long),
        satellite_channels=torch.ones(4, dtype=torch.long),
        prototypes=torch.eye(2),
        margin_epsilon=0.05,
        correctability_epsilon=0.01,
        energy_threshold=0.10,
        class_attraction_max_cosine=0.50,
    )

    expected = {
        "robust_ce",
        "sat_kl",
        "margin",
        "relation",
        "class_conditional",
        "receiver",
        "day",
        "channel",
        "context_tx_adv",
        "conditional_decorrelation",
        "shared_receiver",
        "minimum_correction",
        "alpha",
        "subspace",
        "correctability",
        "score_stability",
        "class_attraction",
    }
    assert expected <= set(bundle["losses"])
    assert float(bundle["losses"]["minimum_correction"].detach()) == pytest.approx(0.0025)
    total = torch.stack([bundle["losses"][name] for name in sorted(expected)]).sum()
    assert torch.isfinite(total)
    total.backward()
    assert clean["ntrs_z_anchor"].grad is not None


def test_real_ntrs_model_pair_supports_backward_then_tangent_update():
    model = build_dual_model(
        num_classes=2,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        use_ntrs=True,
        ntrs_rank=8,
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
        ntrs_slow_dim=8,
        ntrs_metadata_dim=3,
    )
    model.train()
    labels = torch.tensor([0, 1, 0, 1])
    domains = torch.tensor([0, 1, 0, 1])
    clean = torch.randn(4, 2, 64)
    satellite = clean + 0.05 * torch.randn_like(clean)
    clean_out = model(
        clean,
        y_tx=labels,
        return_aux=True,
        domain_labels=domains,
        ntrs_epoch=68,
        update_ntrs_source=True,
        ntrs_source_mask=torch.ones(4, dtype=torch.bool),
    )
    satellite_out = model(
        satellite,
        y_tx=labels,
        return_aux=True,
        domain_labels=domains,
        ntrs_epoch=68,
        ntrs_metadata=torch.randn(4, 3),
        ntrs_metadata_valid=torch.ones(4, dtype=torch.bool),
    )
    bundle = compute_ntrs_loss_bundle(
        clean_out,
        satellite_out,
        clean_labels=labels,
        satellite_labels=labels,
        clean_receivers=domains,
        satellite_receivers=domains,
        clean_days=domains,
        satellite_days=domains,
        clean_channels=torch.zeros(4, dtype=torch.long),
        satellite_channels=torch.ones(4, dtype=torch.long),
        prototypes=model.id_backbone.cls_head.head.weight,
        margin_epsilon=0.05,
        correctability_epsilon=0.01,
        energy_threshold=0.10,
        class_attraction_max_cosine=0.50,
    )
    total = sum(bundle["losses"].values()) + torch.nn.functional.cross_entropy(
        satellite_out["tx_logits"], labels
    )
    total.backward()
    assert [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ] == []
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if "ntrs_" in name
    )
    model.update_ntrs_tangent(
        clean_out["ntrs_z_anchor"].detach(),
        satellite_out["ntrs_z_anchor"].detach(),
    )
    assert int(model.ntrs_robustifier.tangent.update_count.item()) == 1


def test_s1_zero_weighted_ntrs_losses_do_not_poison_core_gradients():
    """S1 freezes NTRS, so its zero-weight terms must not create NaN gradients."""

    torch.manual_seed(392034)
    model = build_dual_model(
        num_classes=2,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        use_ntrs=True,
        ntrs_rank=8,
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
        ntrs_slow_dim=8,
        ntrs_metadata_dim=3,
    )
    model.train()
    labels = torch.tensor([0, 1, 0, 1])
    domains = torch.tensor([0, 1, 0, 1])
    output = model(
        torch.randn(4, 2, 64),
        y_tx=labels,
        return_aux=True,
        domain_labels=domains,
        ntrs_epoch=1,
        update_ntrs_source=False,
        ntrs_source_mask=torch.ones(4, dtype=torch.bool),
    )
    bundle = compute_ntrs_loss_bundle(
        output,
        None,
        clean_labels=labels,
        satellite_labels=None,
        clean_receivers=domains,
        satellite_receivers=None,
        clean_days=domains,
        satellite_days=None,
        clean_channels=torch.zeros(4, dtype=torch.long),
        satellite_channels=None,
        prototypes=model.id_backbone.cls_head.head.weight,
        margin_epsilon=0.05,
        correctability_epsilon=0.01,
        energy_threshold=0.10,
        class_attraction_max_cosine=0.50,
    )
    loss = torch.nn.functional.cross_entropy(output["tx_logits"], labels)
    loss = loss + sum(0.0 * term for term in bundle["losses"].values())
    loss.backward()

    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert nonfinite == []
