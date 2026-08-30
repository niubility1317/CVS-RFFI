from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from cvsrffi.stage2_wiser_rf import (
    configure_progressive_identity_update,
    leave_one_out_prototype_logits,
    normalized_l2sp_penalty,
    wiser_dual_supervision_loss,
)


class _IdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sinc = nn.Linear(4, 4)
        self.time_fuse = nn.Linear(4, 4)
        self.t1 = nn.Linear(4, 4)
        self.t2 = nn.Linear(4, 4)
        self.t3 = nn.Linear(4, 4)
        self.f1 = nn.Linear(4, 4)
        self.f2 = nn.Linear(4, 4)
        self.f3 = nn.Linear(4, 4)
        self.t_proj = nn.Linear(4, 4)
        self.f_proj = nn.Linear(4, 4)
        self.freq_gate = nn.Linear(4, 4)
        self.fuse = nn.Linear(8, 4)
        self.dac_b1 = nn.Linear(4, 4)
        self.pa_b1 = nn.Linear(4, 4)
        self.cls_head = nn.Module()
        self.cls_head.id_proj = nn.Linear(4, 4)
        self.cls_head.id_gate = nn.Linear(8, 4)
        self.cls_head.joint_proj = nn.Linear(12, 4)
        self.cls_head.head = nn.Linear(4, 2, bias=False)
        self.cls_head.dac_head = nn.Linear(4, 1)
        self.cls_head.pa_head = nn.Linear(4, 1)


class _DualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _IdentityBackbone()
        self.dom_backbone = nn.Linear(4, 4)
        self.dom_head = nn.Linear(4, 2)
        self.adv_head = nn.Linear(4, 2)


def _trainable_names(model: nn.Module) -> set[str]:
    return {name for name, value in model.named_parameters() if value.requires_grad}


def test_stage1_updates_only_late_primary_identity_path() -> None:
    model = _DualModel()

    audit = configure_progressive_identity_update(model, stage=1)
    names = _trainable_names(model)

    assert names
    assert all(
        name.startswith(
            (
                "id_backbone.t3.",
                "id_backbone.f3.",
                "id_backbone.t_proj.",
                "id_backbone.f_proj.",
                "id_backbone.fuse.",
                "id_backbone.cls_head.id_proj.",
                "id_backbone.cls_head.id_gate.",
                "id_backbone.cls_head.joint_proj.",
            )
        )
        for name in names
    )
    assert not model.id_backbone.cls_head.head.weight.requires_grad
    assert not model.id_backbone.sinc.weight.requires_grad
    assert not model.id_backbone.dac_b1.weight.requires_grad
    assert not model.dom_backbone.weight.requires_grad
    assert model.training is False
    assert set(audit.trainable_parameter_names) == names


def test_stage3_adds_early_primary_path_but_keeps_sinc_and_aux_frozen() -> None:
    model = _DualModel()

    configure_progressive_identity_update(model, stage=3)
    names = _trainable_names(model)

    assert any(name.startswith("id_backbone.t1.") for name in names)
    assert any(name.startswith("id_backbone.f1.") for name in names)
    assert any(name.startswith("id_backbone.time_fuse.") for name in names)
    assert any(name.startswith("id_backbone.freq_gate.") for name in names)
    assert not any(name.startswith("id_backbone.sinc.") for name in names)
    assert not any(name.startswith("id_backbone.cls_head.head.") for name in names)
    assert not any(name.startswith("id_backbone.cls_head.dac_head.") for name in names)


def test_leave_one_out_prototypes_exclude_each_support_example() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    logits = leave_one_out_prototype_logits(features, labels, scale=1.0)

    expected_first_positive = F.cosine_similarity(
        features[0:1], features[1:2], dim=1
    ).item()
    assert torch.allclose(logits[0, 0], torch.tensor(expected_first_positive))
    assert logits.shape == (4, 2)


def test_dual_loss_reaches_features_without_updating_frozen_source_head() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [0.8, 0.2, -0.1],
            [0.0, 1.0, 0.1],
            [0.2, 0.8, -0.1],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    source_weight = nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        requires_grad=False,
    )
    source_logits = 10.0 * F.linear(F.normalize(features, dim=1), source_weight)

    losses = wiser_dual_supervision_loss(
        source_logits,
        features,
        labels,
        lambda_proto=0.5,
        prototype_scale=10.0,
    )
    losses.total.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert source_weight.grad is None
    assert torch.allclose(losses.total, losses.source_head + 0.5 * losses.target_proto)


def test_normalized_l2sp_is_zero_at_anchor_and_positive_after_change() -> None:
    model = nn.Linear(3, 2)
    anchors = {name: value.detach().clone() for name, value in model.named_parameters()}

    zero = normalized_l2sp_penalty(model.named_parameters(), anchors)
    with torch.no_grad():
        model.weight.add_(0.25)
    moved = normalized_l2sp_penalty(model.named_parameters(), anchors)

    assert zero.item() == 0.0
    assert moved.item() > 0.0
