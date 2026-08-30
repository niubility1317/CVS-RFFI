from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from cvsrffi.phase1_hcfdg.model import (
    CommonSpecificLowRankHead,
    CounterfactualTransport,
    HCFDGModel,
)


class _CountingIdentityBackbone(nn.Module):
    def __init__(self, input_dim: int = 12, feature_dim: int = 160, num_classes: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Mapping[str, torch.Tensor]:
        self.forward_calls += 1
        feature = self.projection(x)
        output = {"logits": self.classifier(feature), "feat_joint": feature}
        if return_aux:
            return output
        return output


class _OpaqueIdentityBackbone(nn.Module):
    def forward(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return {"logits": x[:, :6], "feat_joint": x}


def _make_model() -> tuple[HCFDGModel, _CountingIdentityBackbone, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(11)
    backbone = _CountingIdentityBackbone()
    model = HCFDGModel(
        backbone,
        num_classes=6,
        num_receivers=4,
        num_days=3,
        num_channels=5,
    )
    x = torch.randn(96, 12, requires_grad=True)
    tx_labels = torch.arange(96) % 6
    env_meta = {
        "receiver": torch.arange(96) % 4,
        "day": torch.arange(96) % 3,
        "channel": torch.arange(96) % 5,
    }
    return model, backbone, x, tx_labels, env_meta


def test_learned_identity_projection_is_explicit_and_checkpoint_reconstructible() -> None:
    backbone = _CountingIdentityBackbone(feature_dim=96)
    model = HCFDGModel(
        backbone,
        num_classes=6,
        num_receivers=4,
        num_days=3,
        num_channels=5,
        backbone_feature_dim=96,
    )

    assert isinstance(model.p_id, nn.Linear)
    assert (model.p_id.in_features, model.p_id.out_features) == (96, 160)
    assert model.state_dict()["p_id.weight"].shape == (160, 96)

    restored = HCFDGModel(
        _CountingIdentityBackbone(feature_dim=96),
        num_classes=6,
        num_receivers=4,
        num_days=3,
        num_channels=5,
        backbone_feature_dim=96,
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.p_id.weight, model.p_id.weight)


def test_backbone_feature_dimension_must_be_explicit_or_safely_inferred() -> None:
    with pytest.raises(ValueError, match="backbone_feature_dim"):
        HCFDGModel(
            _OpaqueIdentityBackbone(),
            num_classes=6,
            num_receivers=4,
            num_days=3,
            num_channels=5,
        )


def test_environment_uses_detached_pre_projection_fusion_feature() -> None:
    torch.manual_seed(13)
    backbone = _CountingIdentityBackbone(feature_dim=96)
    model = HCFDGModel(
        backbone,
        num_classes=6,
        num_receivers=4,
        num_days=3,
        num_channels=5,
        backbone_feature_dim=96,
    )
    x = torch.randn(12, 12, requires_grad=True)
    tx_labels = torch.arange(12) % 6
    env_meta = {
        "receiver": torch.arange(12) % 4,
        "day": torch.arange(12) % 3,
        "channel": torch.arange(12) % 5,
    }

    out = model(x, tx_labels=tx_labels, env_meta=env_meta, training_aux=True)

    assert backbone.forward_calls == 1
    assert out.fused_feature.shape == (12, 96)
    assert out.z_id.shape == (12, 160)
    assert model.environment_encoder.input_dim == 96
    assert model.counterfactual_transport.feature_dim == 96
    env_to_fusion = torch.autograd.grad(
        out.z_env.sum(), out.fused_feature, retain_graph=True, allow_unused=True
    )[0]
    common_to_fusion = torch.autograd.grad(out.common_logits.sum(), out.fused_feature)[0]
    assert env_to_fusion is None
    assert torch.isfinite(common_to_fusion).all()


def test_hcfdg_output_has_single_backbone_and_48d_environment() -> None:
    model, backbone, x, tx_labels, env_meta = _make_model()

    out = model(x, tx_labels=tx_labels, env_meta=env_meta, training_aux=True)

    assert backbone.forward_calls == 1
    assert out.z_id.shape == (96, 160)
    assert out.z_env.shape == (96, 48)
    assert out.specific_logits is not None
    assert out.specific_logits.shape == out.common_logits.shape == (96, 6)
    assert out.receiver_logits is not None
    assert out.day_logits is not None
    assert out.channel_logits is not None
    assert out.tx_from_env_logits is not None
    assert out.conditional_receiver_logits is not None


def test_environment_encoder_consumes_all_five_satellite_physical_factors() -> None:
    model, _, x, tx_labels, env_meta = _make_model()
    factors = torch.randn(x.shape[0], 5)
    env_meta = dict(env_meta, channel_factors=factors)

    out = model(x, tx_labels=tx_labels, env_meta=env_meta, training_aux=True)

    assert model.environment_encoder.q_phys_dim == 5
    assert model.environment_encoder.shared[0].in_features == model.backbone_feature_dim + 5
    assert torch.isfinite(out.z_env).all()


def test_environment_branch_stops_backbone_gradient_but_identity_branch_keeps_it() -> None:
    model, _, x, tx_labels, env_meta = _make_model()
    out = model(x, tx_labels=tx_labels, env_meta=env_meta, training_aux=True)

    out.z_env.sum().backward(retain_graph=True)
    assert x.grad is None

    out.common_logits.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_conditional_receiver_adversary_reads_zid_and_tx_onehot() -> None:
    model, _, x, tx_labels, env_meta = _make_model()
    model.eval()
    assert model.conditional_receiver_head.in_features == 160 + 6
    tx_weights = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.2, 0.0, 0.4, 0.1, 0.5, 0.3],
            [0.3, 0.5, 0.1, 0.4, 0.0, 0.2],
        ]
    )
    with torch.no_grad():
        model.conditional_receiver_head.weight.zero_()
        model.conditional_receiver_head.bias.zero_()
        model.conditional_receiver_head.weight[:, 160:].copy_(tx_weights)

    out = model(
        x,
        tx_labels=tx_labels,
        env_meta=env_meta,
        training_aux=True,
        grl_strength=0.05,
    )
    expected = torch.nn.functional.one_hot(tx_labels, num_classes=6).float() @ tx_weights.T

    assert out.conditional_receiver_logits is not None
    torch.testing.assert_close(out.conditional_receiver_logits, expected)


def test_bounded_grl_reverses_both_adversaries_and_keeps_direct_heads() -> None:
    model, _, x, tx_labels, env_meta = _make_model()
    model.eval()
    assert model.environment_encoder.tx_from_env_head is not None
    with torch.no_grad():
        model.conditional_receiver_head.weight.fill_(1.0)
        model.conditional_receiver_head.bias.zero_()
        model.environment_encoder.tx_from_env_head.weight.fill_(1.0)
        model.environment_encoder.tx_from_env_head.bias.zero_()
        model.environment_encoder.receiver_head.weight.fill_(1.0)
        model.environment_encoder.receiver_head.bias.zero_()

    out = model(
        x,
        tx_labels=tx_labels,
        env_meta=env_meta,
        training_aux=True,
        grl_strength=0.05,
    )
    assert out.conditional_receiver_logits is not None
    assert out.tx_from_env_logits is not None
    assert out.receiver_logits is not None

    conditional_grad = torch.autograd.grad(
        out.conditional_receiver_logits.sum(), out.z_id, retain_graph=True, allow_unused=True
    )[0]
    tx_adversary_grad = torch.autograd.grad(
        out.tx_from_env_logits.sum(), out.z_env, retain_graph=True, allow_unused=True
    )[0]
    receiver_direct_grad = torch.autograd.grad(out.receiver_logits.sum(), out.z_rx)[0]

    assert conditional_grad is not None
    assert tx_adversary_grad is not None
    torch.testing.assert_close(conditional_grad, torch.full_like(out.z_id, -0.20))
    torch.testing.assert_close(tx_adversary_grad, torch.full_like(out.z_env, -0.30))
    torch.testing.assert_close(receiver_direct_grad, torch.full_like(out.z_rx, 4.0))


def test_grl_strength_above_point_zero_five_is_rejected_consistently() -> None:
    model, _, x, tx_labels, env_meta = _make_model()
    with pytest.raises(ValueError, match="0.05"):
        model(
            x,
            tx_labels=tx_labels,
            env_meta=env_meta,
            training_aux=True,
            grl_strength=0.050001,
        )
    with pytest.raises(ValueError, match="0.05"):
        model.environment_encoder(
            torch.randn(4, model.backbone_feature_dim),
            env_meta={
                "receiver": torch.arange(4) % 4,
                "day": torch.arange(4) % 3,
                "channel": torch.arange(4) % 5,
            },
            grl_strength=0.050001,
        )


def test_low_rank_head_uses_rank_four_and_specific_dropout() -> None:
    head = CommonSpecificLowRankHead(
        feature_dim=160,
        num_classes=6,
        rank=4,
        specific=True,
        dropout=0.5,
    )

    assert head.rank == 4
    assert head.dropout.p == pytest.approx(0.5)
    assert tuple(head.U.shape) == (6, 4)
    assert tuple(head.V.shape) == (160, 4)

    features = torch.randn(8, 160)
    z_rx = torch.randn(8, 16)
    z_day = torch.randn(8, 16)
    z_channel = torch.randn(8, 16)
    logits = head(features, z_rx=z_rx, z_day=z_day, z_channel=z_channel)
    assert logits.shape == (8, 6)
    assert torch.isfinite(logits).all()


def test_inference_uses_only_common_head() -> None:
    model, _, x, _, _ = _make_model()
    model.eval()

    def _unexpected(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("inference must not execute training-only modules")

    model.environment_encoder.forward = _unexpected  # type: ignore[method-assign]
    model.specific_head.forward = _unexpected  # type: ignore[method-assign]
    model.counterfactual_transport.forward = _unexpected  # type: ignore[method-assign]

    with torch.no_grad():
        logits = model.inference_logits(x)
        expected = model.common_head(model.identity_features(x))

    assert torch.equal(logits, expected)


def test_counterfactual_transport_pairs_only_same_tx_and_returns_target_labels() -> None:
    torch.manual_seed(3)
    transport = CounterfactualTransport(feature_dim=8, env_dim=48, gamma_cap=0.2, beta_cap=0.3)
    h = torch.randn(6, 8)
    env = torch.randn(6, 48)
    tx_labels = torch.tensor([0, 0, 0, 1, 1, 2])
    receiver_labels = torch.tensor([0, 1, 2, 0, 1, 0])
    target_env_labels = {"receiver": receiver_labels, "day": torch.arange(6) % 2}

    pair = transport.receiver_swap(
        h,
        env,
        tx_labels=tx_labels,
        receiver_labels=receiver_labels,
        target_env_labels=target_env_labels,
    )

    assert pair.source_indices.numel() == 5
    assert torch.equal(tx_labels[pair.source_indices], tx_labels[pair.target_indices])
    assert torch.all(receiver_labels[pair.source_indices] != receiver_labels[pair.target_indices])
    assert pair.target_env_labels is target_env_labels
    assert pair.features.shape == (5, 8)
    assert torch.isfinite(pair.features).all()


def test_counterfactual_transport_accepts_explicit_delta_environment() -> None:
    torch.manual_seed(5)
    transport = CounterfactualTransport(feature_dim=8, env_dim=48, gamma_cap=0.2, beta_cap=0.3)
    h = torch.randn(4, 8)
    delta_env = torch.randn(4, 48)

    result = transport(h, delta_env)
    gamma = transport.gamma_head(delta_env).clamp(-0.2, 0.2)
    beta = transport.beta_head(delta_env).clamp(-0.3, 0.3)
    expected = (1.0 + gamma) * torch.nn.functional.layer_norm(h, h.shape[1:]) + beta

    torch.testing.assert_close(result, expected)


def test_model_training_path_has_finite_gradients() -> None:
    model, _, x, tx_labels, env_meta = _make_model()
    out = model(x, tx_labels=tx_labels, env_meta=env_meta, training_aux=True)
    loss = out.common_logits.square().mean() + out.specific_logits.square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
