from __future__ import annotations

import pytest
import torch
from torch import nn

from cvsrffi.phase1_bicad_xr.adversarial_game import (
    AdversarialGamePlan,
    DynamicGRLDoseController,
    DualRatioController,
    build_adversarial_optimizers,
)


def test_game_plan_reuses_one_forward_with_explicit_detached_and_encoder_phases() -> None:
    plan = AdversarialGamePlan()
    features = torch.randn(4, 6, requires_grad=True)

    discriminator_features = plan.discriminator_features(features)
    encoder_features = plan.encoder_features(features)

    assert plan.one_backbone_forward is True
    assert plan.phase_order == ("discriminator", "encoder")
    assert discriminator_features is not features
    assert torch.equal(discriminator_features, features)
    assert discriminator_features.requires_grad is False
    assert encoder_features is features
    assert encoder_features.requires_grad is True


def test_freezing_discriminator_preserves_input_gradients_and_restores_flags() -> None:
    plan = AdversarialGamePlan()
    discriminator = nn.Linear(3, 2)
    original_flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    features = torch.randn(4, 3, requires_grad=True)

    with plan.freeze_discriminator(discriminator):
        assert not any(parameter.requires_grad for parameter in discriminator.parameters())
        discriminator(features).sum().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    assert [parameter.requires_grad for parameter in discriminator.parameters()] == original_flags


def test_optimizer_builder_keeps_encoder_and_discriminator_parameter_groups_disjoint() -> None:
    encoder = nn.Linear(4, 3)
    discriminator = nn.Linear(3, 2)

    encoder_optimizer, discriminator_optimizer = build_adversarial_optimizers(
        encoder, discriminator, encoder_lr=0.002
    )

    encoder_ids = {
        id(parameter)
        for group in encoder_optimizer.param_groups
        for parameter in group["params"]
    }
    discriminator_ids = {
        id(parameter)
        for group in discriminator_optimizer.param_groups
        for parameter in group["params"]
    }

    assert encoder_ids.isdisjoint(discriminator_ids)
    assert discriminator_optimizer.param_groups[0]["lr"] == pytest.approx(0.003)


def test_optimizer_builder_rejects_a_parameter_shared_by_both_games() -> None:
    shared = nn.Parameter(torch.ones(2))

    with pytest.raises(ValueError, match="disjoint"):
        build_adversarial_optimizers([shared], [shared])


def test_dual_ratio_controller_keeps_conditional_and_zdom_tx_ratios_independent() -> None:
    controller = DualRatioController(ema_decay=0.0)

    first = controller.update(
        reference_gradient=torch.tensor([2.0]),
        conditional_gradient=torch.tensor([1.0]),
        zdom_tx_gradient=torch.tensor([4.0]),
    )
    second = controller.update(
        reference_gradient=torch.tensor([2.0]),
        conditional_gradient=torch.tensor([2.0]),
        zdom_tx_gradient=torch.tensor([4.0]),
    )

    assert first["conditional"] == pytest.approx(0.30)
    assert first["zdom_tx"] == pytest.approx(0.0275)
    assert second["conditional"] == pytest.approx(0.15)
    assert second["zdom_tx"] == pytest.approx(first["zdom_tx"])
    assert controller.conditional_ratio == pytest.approx(0.15)
    assert controller.zdom_tx_ratio == pytest.approx(0.0275)


def test_game_plan_validates_the_two_adversarial_ratio_windows() -> None:
    plan = AdversarialGamePlan()

    assert plan.discriminator_lr_ratio == pytest.approx(1.5)
    assert plan.conditional_ratio_bounds == pytest.approx((0.10, 0.20))
    assert plan.zdom_tx_ratio_bounds == pytest.approx((0.03, 0.08))


def test_dynamic_grl_doses_consume_all_feedback_and_remain_bounded() -> None:
    controller = DynamicGRLDoseController(ema_decay=0.0)

    permissive = controller.update(
        discriminator_accuracy=0.95,
        tx_margin=1.0,
        adversarial_gradient_ratio=0.01,
        conflict_signal=0.0,
    )
    fragile = controller.update(
        discriminator_accuracy=0.50,
        tx_margin=-1.0,
        adversarial_gradient_ratio=0.50,
        conflict_signal=1.0,
    )

    assert permissive["identity"] > fragile["identity"]
    assert permissive["zdom"] > fragile["zdom"]
    assert controller.identity_dose_bounds[0] <= fragile["identity"] <= controller.identity_dose_bounds[1]
    assert controller.zdom_dose_bounds[0] <= fragile["zdom"] <= controller.zdom_dose_bounds[1]
    assert controller.last_feedback is not None
    assert controller.last_feedback["discriminator_accuracy"] == pytest.approx(0.50)
    assert controller.conditional_ratio_bounds == pytest.approx((0.10, 0.20))
    assert controller.zdom_tx_ratio_bounds == pytest.approx((0.03, 0.08))
