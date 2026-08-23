from __future__ import annotations

import importlib
import inspect

_SUT = importlib.import_module("cvsrffi.stage2_structured_late_block_adaptation")

import pytest
import torch
import torch.nn as nn


def _subject():
    return _SUT


class _DepthwisePointwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.pw(self.dw(rows))))


class _FrozenClassifierHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.id_gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.head = nn.Linear(channels, 2, bias=False)


class _ToyIdentityBackbone(nn.Module):
    def __init__(self, *, reserve: int = 500) -> None:
        super().__init__()
        self.freq_stem = nn.Conv1d(2, 4, kernel_size=1, bias=False)
        self.f3 = _DepthwisePointwiseBlock(4)
        self.f_pool = nn.AdaptiveAvgPool1d(1)
        self.f_proj = nn.Linear(4, 4)
        self.time_stem = nn.Conv1d(2, 4, kernel_size=1, bias=False)
        self.t3 = _DepthwisePointwiseBlock(4)
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        self.t_proj = nn.Linear(4, 4)
        self.cls_head = _FrozenClassifierHead(4)
        self.reserve = nn.Parameter(torch.zeros(reserve), requires_grad=False)

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        freq = self.f_proj(self.f_pool(self.f3(self.freq_stem(rows))).squeeze(-1))
        time = self.t_proj(self.t_pool(self.t3(self.time_stem(rows))).squeeze(-1))
        return 0.5 * (freq + time)


class _ToyADV3B02(nn.Module):
    def __init__(self, *, reserve: int = 500) -> None:
        super().__init__()
        self.id_backbone = _ToyIdentityBackbone(reserve=reserve)
        self.dom_backbone = nn.Linear(4, 3)

    def forward(self, rows: torch.Tensor, **_kwargs):
        features = self.id_backbone.encode(rows)
        logits = self.id_backbone.cls_head.head(features)
        return {"z_id": features, "tx_logits": logits}


def _support() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.tensor(
        [
            [[2.0, 1.5, 1.0], [0.1, 0.2, 0.0]],
            [[1.7, 1.2, 0.9], [0.2, 0.1, 0.1]],
            [[0.0, 0.2, 0.1], [1.8, 1.4, 1.0]],
            [[0.1, 0.0, 0.2], [1.6, 1.3, 0.8]],
        ],
        dtype=torch.float32,
    )
    return rows, torch.tensor([10, 10, 20, 20], dtype=torch.long)


def _prototypes() -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    return prototypes, torch.tensor([10, 20], dtype=torch.long)


def _context() -> dict[str, str]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed-received-iq",
        "split_id": "split-support-query-disjoint",
    }


def _changed_parameters(
    model: nn.Module, before: dict[str, torch.Tensor]
) -> set[str]:
    return {
        name
        for name, value in model.named_parameters()
        if not torch.equal(value.detach(), before[name])
    }


def test_freq_candidate_updates_complete_contiguous_block_only() -> None:
    subject = _subject()
    torch.manual_seed(7)
    model = _ToyADV3B02()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes()
    prototype_before = prototypes.clone()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    audit = subject.adapt_on_target_support_with_frozen_prototypes(
        model,
        support_iq,
        support_labels,
        frozen_prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
        context=_context(),
        config=subject.StructuredLateBlockConfig(
            candidate="freq_f3_proj",
            steps=3,
            learning_rate=0.02,
        ),
    )

    changed = _changed_parameters(model, before)
    assert changed == set(audit.trainable_parameter_names)
    assert changed
    assert all(
        name.startswith(("id_backbone.f3.", "id_backbone.f_proj."))
        for name in changed
    )
    assert any(name.endswith("f3.dw.weight") for name in changed)
    assert any(name.endswith("f3.pw.weight") for name in changed)
    assert any(name.endswith("f_proj.weight") for name in changed)
    assert not any("gate" in name or "cls_head" in name for name in changed)
    assert 0.05 <= audit.trainable_fraction <= 0.15
    assert audit.structural_trainable_parameters > 0
    assert audit.gradient_updates == 3
    assert audit.classifier_parameters_changed == 0
    assert audit.prototypes_changed is False
    assert torch.equal(prototypes, prototype_before)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.training is False


def test_time_candidate_updates_the_full_t3_block_and_is_distinct() -> None:
    subject = _subject()
    torch.manual_seed(11)
    model = _ToyADV3B02()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    audit = subject.adapt_on_target_support_with_frozen_prototypes(
        model,
        support_iq,
        support_labels,
        frozen_prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
        context=_context(),
        config=subject.StructuredLateBlockConfig(
            candidate="time_t3",
            steps=2,
            learning_rate=0.02,
        ),
    )

    changed = _changed_parameters(model, before)
    assert changed == set(audit.trainable_parameter_names)
    assert changed
    assert all(name.startswith("id_backbone.t3.") for name in changed)
    assert any(name.endswith("t3.dw.weight") for name in changed)
    assert any(name.endswith("t3.pw.weight") for name in changed)
    assert any(name.endswith("t3.norm.weight") for name in changed)
    assert 0.05 <= audit.trainable_fraction <= 0.15
    assert audit.gradient_updates == 2


def test_exhaustive_phase2_allowlist_and_resource_caps_fail_closed() -> None:
    subject = _subject()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes()

    for extra_key in ("source_cache_path", "query_truth", "receiver"):
        context = _context()
        context[extra_key] = "forbidden-extra-input"
        with pytest.raises(subject.StructuredLateBlockError, match="allowlist"):
            subject.adapt_on_target_support_with_frozen_prototypes(
                _ToyADV3B02(),
                support_iq,
                support_labels,
                frozen_prototypes=prototypes,
                prototype_class_ids=prototype_class_ids,
                context=context,
                config=subject.StructuredLateBlockConfig(steps=1),
            )

    with pytest.raises(subject.StructuredLateBlockError, match="40"):
        subject.adapt_on_target_support_with_frozen_prototypes(
            _ToyADV3B02(),
            support_iq,
            support_labels,
            frozen_prototypes=prototypes,
            prototype_class_ids=prototype_class_ids,
            context=_context(),
            config=subject.StructuredLateBlockConfig(steps=41),
        )

    with pytest.raises(subject.StructuredLateBlockError, match="fraction"):
        subject.adapt_on_target_support_with_frozen_prototypes(
            _ToyADV3B02(reserve=0),
            support_iq,
            support_labels,
            frozen_prototypes=prototypes,
            prototype_class_ids=prototype_class_ids,
            context=_context(),
            config=subject.StructuredLateBlockConfig(steps=1),
        )

    trainable_prototypes = prototypes.clone().requires_grad_(True)
    with pytest.raises(subject.StructuredLateBlockError, match="immutable"):
        subject.adapt_on_target_support_with_frozen_prototypes(
            _ToyADV3B02(),
            support_iq,
            support_labels,
            frozen_prototypes=trainable_prototypes,
            prototype_class_ids=prototype_class_ids,
            context=_context(),
            config=subject.StructuredLateBlockConfig(steps=1),
        )

    adapt_parameters = set(
        inspect.signature(
            subject.adapt_on_target_support_with_frozen_prototypes
        ).parameters
    )
    assert not {"query_iq", "query_labels", "query_truth", "query_role"} & adapt_parameters


def test_query_prediction_is_per_sample_and_state_read_only() -> None:
    subject = _subject()
    torch.manual_seed(13)
    model = _ToyADV3B02()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes()
    subject.adapt_on_target_support_with_frozen_prototypes(
        model,
        support_iq,
        support_labels,
        frozen_prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
        context=_context(),
        config=subject.StructuredLateBlockConfig(steps=2, learning_rate=0.02),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    predictions, scores = subject.predict_query_with_frozen_prototypes(
        model,
        support_iq[:2],
        frozen_prototypes=prototypes,
        prototype_class_ids=prototype_class_ids,
    )
    reversed_predictions, reversed_scores = (
        subject.predict_query_with_frozen_prototypes(
            model,
            support_iq[:2].flip(0),
            frozen_prototypes=prototypes,
            prototype_class_ids=prototype_class_ids,
        )
    )

    assert predictions.shape == (2,)
    assert scores.shape == (2, 2)
    assert torch.equal(reversed_predictions, predictions.flip(0))
    assert torch.allclose(reversed_scores, scores.flip(0))
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in before.items()
    )
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    query_parameters = set(
        inspect.signature(subject.predict_query_with_frozen_prototypes).parameters
    )
    assert not {"query_labels", "query_truth", "query_role"} & query_parameters
