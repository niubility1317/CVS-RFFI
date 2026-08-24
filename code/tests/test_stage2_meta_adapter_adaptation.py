from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import ResidualMetaAdapter, iter_inner_adapter_parameters  # noqa: E402
from cvsrffi.stage2_meta_adapter_adaptation import (  # noqa: E402
    MetaAdapterPhase2Config,
    adapt_meta_adapter_on_support,
    predict_with_frozen_meta_adapter,
)


class _ToyPhase2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.meta_adapter_time = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_freq = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_fusion = ResidualMetaAdapter(dim=4, rank=2)
        self.fixed_backbone_parameter = nn.Parameter(torch.zeros(10_000))
        self.register_buffer("query_counter", torch.zeros((), dtype=torch.long))

    def forward(self, x, y=None, return_aux=False):
        del y, return_aux
        z = self.meta_adapter_time(x)
        z = self.meta_adapter_freq(z)
        z = self.meta_adapter_fusion(z)
        self.query_counter.add_(1)
        return {"feat_cls": z}


class _ToyPartialPhase2Model(_ToyPhase2Model):
    def forward(self, x, y=None, return_aux=False):
        del y, return_aux
        return {"feat_cls": self.meta_adapter_time(x)}


def _context() -> dict[str, str]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test-01",
        "split_id": "split-test-01",
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    torch.manual_seed(23)
    support_iq = torch.randn(6, 4)
    support_labels = torch.tensor([10, 20, 30, 10, 20, 30], dtype=torch.long)
    prototypes = torch.eye(4, dtype=torch.float32)[:3]
    class_ids = [10, 20, 30]
    return support_iq, support_labels, prototypes, class_ids


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_phase2_updates_only_adapter_and_exactly_three_steps():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    before = _state(model)
    learned_steps_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.endswith("log_step_size")
    }

    audit = adapt_meta_adapter_on_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        class_ids,
        context=_context(),
        config=MetaAdapterPhase2Config(steps=3),
    )

    assert audit.backward_count == 3
    assert audit.steps == 3
    assert audit.trainable_fraction <= 0.01
    assert audit.updated_parameter_names
    assert all("meta_adapter" in name for name in audit.updated_parameter_names)
    for name, value in model.named_parameters():
        if name.endswith("log_step_size"):
            assert torch.equal(value.detach(), learned_steps_before[name])
        if not name.startswith("meta_adapter_"):
            assert torch.equal(value.detach(), before[name])
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_phase2_rejects_source_or_query_keys():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    for key in ("source_samples", "clean_iq", "cache", "query_role", "query_truth", "role"):
        bad = {**_context(), key: "forbidden"}
        with pytest.raises(ValueError, match="context allowlist"):
            adapt_meta_adapter_on_support(
                model,
                support_iq,
                support_labels,
                prototypes,
                class_ids,
                context=bad,
                config=MetaAdapterPhase2Config(steps=3),
            )


def test_phase2_accepts_diagnostic_zero_to_five_steps_but_rejects_six():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    zero = adapt_meta_adapter_on_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        class_ids,
        context=_context(),
        config=MetaAdapterPhase2Config(steps=0),
    )
    assert zero.backward_count == 0
    assert zero.updated_parameter_names == ()
    with pytest.raises(ValueError, match="5"):
        adapt_meta_adapter_on_support(
            _ToyPhase2Model(),
            support_iq,
            support_labels,
            prototypes,
            class_ids,
            context=_context(),
            config=MetaAdapterPhase2Config(steps=6),
        )


def test_phase2_query_prediction_is_frozen_and_returns_registered_class_ids():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    adapt_meta_adapter_on_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        class_ids,
        context=_context(),
        config=MetaAdapterPhase2Config(steps=3),
    )
    query_iq = support_iq[:3].clone()
    before = _state(model)
    prediction = predict_with_frozen_meta_adapter(model, query_iq, prototypes, class_ids)
    assert prediction.shape == (3,)
    assert prediction.dtype == torch.long
    assert set(prediction.tolist()).issubset(set(class_ids))
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_phase2_query_prediction_restores_even_a_mutating_buffer():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    adapt_meta_adapter_on_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        class_ids,
        context=_context(),
        config=MetaAdapterPhase2Config(steps=3),
    )
    before = _state(model)
    predict_with_frozen_meta_adapter(model, support_iq[:2], prototypes, class_ids)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_phase2_rejects_invalid_support_contract():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    with pytest.raises(ValueError, match="support_labels"):
        adapt_meta_adapter_on_support(
            model,
            support_iq,
            support_labels[:-1],
            prototypes,
            class_ids,
            context=_context(),
            config=MetaAdapterPhase2Config(steps=3),
        )


def test_phase2_keeps_unreachable_adapter_frozen_without_rejecting_support_update():
    model = _ToyPartialPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    before = _state(model)
    audit = adapt_meta_adapter_on_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        class_ids,
        context=_context(),
        config=MetaAdapterPhase2Config(steps=3),
    )
    assert audit.updated_parameter_names
    assert all(name.startswith("meta_adapter_time.") for name in audit.updated_parameter_names)
    for name, value in model.state_dict().items():
        if name.startswith(("meta_adapter_freq.", "meta_adapter_fusion.")):
            assert torch.equal(value, before[name]), name
    with pytest.raises(ValueError, match="class_ids"):
        adapt_meta_adapter_on_support(
            model,
            support_iq,
            support_labels,
            prototypes,
            [10, 10, 30],
            context=_context(),
            config=MetaAdapterPhase2Config(steps=3),
        )
