from __future__ import annotations

import copy

import torch

from SSDG.train_ssdg import build_arg_parser
from cvsrffi.nmfdu_training import (
    NMFDUStageController,
    apply_nmfdu_optimizer_lr,
    branch_auxiliary_loss,
    fused_pair_loss,
    nmfdu_labeled_objective,
    nmfdu_optimizer_groups,
    oracle_margin_distribution,
    quality_weighted_mean,
    reliable_branch_pair_loss,
    route_kl_loss,
)
from model import CVSincNet


def _model() -> CVSincNet:
    return CVSincNet(
        num_classes=3,
        input_len=64,
        sinc_out=8,
        sinc_kernel=15,
        time_bottleneck=8,
        emb_dim=16,
        drop=0.0,
        freq_bands=16,
        pa_memory_depth=1,
        pa_orders=(1, 3, 5),
        time_ch1=8,
        time_ch2=8,
        time_ch3=8,
        dac_ch=8,
        freq_ch1=8,
        freq_ch2=8,
        freq_ch3=8,
        pa_ch1=8,
        pa_ch2=8,
        pa_ch3=8,
        physical_gate_variant="nmfdu_v1",
    )


def test_three_stage_boundaries_freeze_exact_parameter_families() -> None:
    model = _model()
    controller = NMFDUStageController(boundaries=(80, 120, 200))
    assert controller.stage_for_epoch(1) == 1
    assert controller.stage_for_epoch(80) == 1
    assert controller.stage_for_epoch(81) == 2
    assert controller.stage_for_epoch(120) == 2
    assert controller.stage_for_epoch(121) == 3
    assert controller.stage_for_epoch(200) == 3

    controller.apply(model, epoch=20)
    assert model.nmfdu_gate.training_stage.item() == 1
    assert not any(p.requires_grad for p in model.nmfdu_gate.sample_gate.parameters())
    assert any(p.requires_grad for p in model.nmfdu_gate.branch_bank.parameters())
    assert any(p.requires_grad for p in model.nmfdu_gate.branch_heads.parameters())

    controller.apply(model, epoch=100)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith("nmfdu_gate.sample_gate.") for name in trainable)
    assert model.nmfdu_gate.evidence_state.discriminability_frozen.item()

    controller.apply(model, epoch=150)
    assert all(p.requires_grad for p in model.parameters())
    assert model.nmfdu_gate.evidence_state.discriminability_frozen.item()


def test_stage_one_bypasses_competition_with_equal_non_null_weights() -> None:
    torch.manual_seed(81)
    model = _model().eval()
    model.set_nmfdu_stage(1)
    with torch.no_grad():
        diag = model(
            torch.randn(2, 2, 64),
            return_aux=True,
            return_physical_gate_diag=True,
        )["physical_gate_diag"]["per_sample"]
    torch.testing.assert_close(diag["weights"], torch.full((2, 5), 0.2))
    assert torch.equal(diag["null_weight"], torch.zeros(2))
    assert torch.equal(diag["q_sample"], torch.ones(2))


def test_stage_and_discriminability_state_survive_checkpoint_round_trip() -> None:
    model = _model()
    controller = NMFDUStageController()
    controller.apply(model, epoch=100)
    model.nmfdu_gate.evidence_state.discriminability_ema.fill_(0.4)
    restored = _model()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert restored.nmfdu_gate.training_stage.item() == 2
    assert restored.nmfdu_gate.evidence_state.discriminability_frozen.item()
    torch.testing.assert_close(
        restored.nmfdu_gate.evidence_state.discriminability_ema,
        model.nmfdu_gate.evidence_state.discriminability_ema,
    )


def test_optimizer_roles_preserve_state_and_apply_stage_specific_learning_rates() -> None:
    model = _model()
    groups = nmfdu_optimizer_groups(model, model.parameters(), base_lr=1e-3)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    roles = {group["nmfdu_role"] for group in optimizer.param_groups}
    assert roles == {"nmfdu_backbone", "nmfdu_branch", "nmfdu_sample_gate"}

    apply_nmfdu_optimizer_lr(optimizer, stage=1, base_lr=1e-3)
    stage_one = {group["nmfdu_role"]: group["lr"] for group in optimizer.param_groups}
    assert stage_one == {
        "nmfdu_backbone": 1e-3,
        "nmfdu_branch": 1e-3,
        "nmfdu_sample_gate": 0.0,
    }
    apply_nmfdu_optimizer_lr(optimizer, stage=2, base_lr=1e-3)
    stage_two = {group["nmfdu_role"]: group["lr"] for group in optimizer.param_groups}
    assert stage_two == {
        "nmfdu_backbone": 0.0,
        "nmfdu_branch": 0.0,
        "nmfdu_sample_gate": 5e-4,
    }
    apply_nmfdu_optimizer_lr(optimizer, stage=3, base_lr=1e-3)
    stage_three = {group["nmfdu_role"]: group["lr"] for group in optimizer.param_groups}
    assert stage_three == {
        "nmfdu_backbone": 1e-4,
        "nmfdu_branch": 5e-4,
        "nmfdu_sample_gate": 5e-4,
    }


def test_training_cli_exposes_frozen_nmfdu_schedule_and_loss_controls() -> None:
    args = build_arg_parser().parse_args(["--output_dir", "unused"])
    assert (
        args.nmfdu_stage1_end,
        args.nmfdu_stage2_end,
        args.nmfdu_stage3_end,
    ) == (80, 120, 200)
    assert args.nmfdu_gate_lr_scale == 0.5
    assert args.nmfdu_joint_backbone_lr_scale == 0.1
    assert args.lambda_nmfdu_branch_aux > 0.0
    assert args.lambda_nmfdu_route > 0.0
    assert args.lambda_nmfdu_phys > 0.0


def test_oracle_route_auxiliary_and_quality_losses_follow_report_semantics() -> None:
    labels = torch.tensor([0, 1])
    branch_logits = {
        "raw": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        "hom": torch.tensor([[0.0, 4.0], [4.0, 0.0]]),
    }
    oracle = oracle_margin_distribution(
        branch_logits, labels, uncertainty=torch.zeros(2, 2), temperature=0.5
    )
    assert oracle.shape == (2, 2)
    assert torch.all(oracle[:, 0] > oracle[:, 1])
    torch.testing.assert_close(oracle.sum(dim=-1), torch.ones(2))
    assert branch_auxiliary_loss(branch_logits, labels).item() > 0.0
    gate_weights = torch.tensor([[0.8, 0.1], [0.7, 0.2]])
    assert route_kl_loss(gate_weights, oracle).item() >= 0.0

    per_sample = torch.tensor([2.0, 6.0])
    assert quality_weighted_mean(per_sample, torch.tensor([1.0, 0.0])).item() == 2.0


def test_pair_losses_allow_gate_change_and_ignore_unreliable_branch_alignment() -> None:
    clean_fused = torch.tensor([[1.0, 0.0]])
    leo_fused = torch.tensor([[0.0, 1.0]])
    pair = fused_pair_loss(
        clean_fused,
        leo_fused,
        clean_quality=torch.tensor([1.0]),
        leo_quality=torch.tensor([0.5]),
    )
    assert pair.item() == 1.0

    clean = {
        "raw": torch.tensor([[1.0, 0.0]]),
        "pa": torch.tensor([[1.0, 0.0]]),
    }
    leo = {
        "raw": torch.tensor([[1.0, 0.0]]),
        "pa": torch.tensor([[-1.0, 0.0]]),
    }
    reliability_clean = torch.tensor([[1.0, 1.0]])
    reliability_leo = torch.tensor([[1.0, 0.0]])
    loss = reliable_branch_pair_loss(
        clean,
        leo,
        branch_names=("raw", "pa"),
        clean_reliability=reliability_clean,
        leo_reliability=reliability_leo,
    )
    assert loss.item() == 0.0


def test_high_level_labeled_objective_is_finite_for_paired_stage_three_batch() -> None:
    torch.manual_seed(82)
    model = _model().train()
    controller = NMFDUStageController()
    controller.apply(model, epoch=150)
    labels = torch.tensor([0, 1, 0, 1])
    output = model(
        torch.randn(4, 2, 64),
        y=labels,
        return_aux=True,
        return_physical_gate_diag=True,
    )
    losses = nmfdu_labeled_objective(
        output,
        labels,
        stage=3,
        clean_count=2,
        lambda_branch_aux=0.2,
        lambda_route=0.1,
        lambda_phys=0.1,
        lambda_fused_pair=0.2,
        lambda_branch_pair=0.1,
        lambda_null_cal=0.05,
        lambda_balance=0.01,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(
        parameter.grad is not None
        for parameter in model.nmfdu_gate.sample_gate.parameters()
    )
