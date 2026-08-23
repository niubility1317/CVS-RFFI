from __future__ import annotations

import inspect

import numpy as np
import cvsrffi.stage2_structured_late_block_adaptation as adaptation_module
from cvsrffi.stage2_structured_late_block_adaptation import (
    Phase2Context,
    StructuredLateBlockConfig,
    StructuredLateBlockError,
    adapt_on_target_support,
    predict_query_read_only,
)
import pytest
import torch
import torch.nn as nn


def test_numpy_inputs_cross_to_torch_by_values_not_array_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_array_identity_bridge(*args, **kwargs):
        raise AssertionError("torch.as_tensor must not bridge NumPy in this runtime")

    monkeypatch.setattr(torch, "as_tensor", reject_array_identity_bridge)
    rows = adaptation_module._validate_received_iq(
        np.ones((2, 2, 8), dtype=np.float32), name="support"
    )
    prototypes, class_ids = adaptation_module._validate_prototypes(
        np.eye(2, dtype=np.float32), ("a", "b")
    )
    assert tuple(rows.shape) == (2, 2, 8)
    assert tuple(prototypes.shape) == (2, 2)
    assert class_ids == ("a", "b")


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reserve = nn.Parameter(torch.zeros(400))
        self.t3 = nn.Linear(2, 4)
        self.t_proj = nn.Linear(4, 4)
        self.fuse = nn.Linear(4, 4)
        self.cls_head = _ToyPhysicalHead()

    def forward(
        self,
        x: torch.Tensor,
        *,
        y: torch.Tensor | None = None,
        return_aux: bool = True,
        **_: object,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        del y
        rows = x.mean(dim=-1)
        rows = torch.tanh(self.t3(rows))
        rows = torch.tanh(self.t_proj(rows))
        z_id = self.fuse(rows)
        logits = self.cls_head.head(z_id)
        if not return_aux:
            return logits
        return {"z_id": z_id, "logits": logits}


class _ToyDualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _ToyBackbone()

    @staticmethod
    def _pick_z_id(aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return aux["z_id"]


class _ToyCosFaceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4) * 0.1)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return 30.0 * torch.nn.functional.linear(
            torch.nn.functional.normalize(rows, dim=1, eps=1.0e-4),
            torch.nn.functional.normalize(self.weight, dim=1, eps=1.0e-4),
        )


class _ToyPhysicalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = _ToyCosFaceHead()


def _context() -> Phase2Context:
    return Phase2Context(
        protocol_schema="p2_min_v1",
        phase2_data_status="VALIDATED_ONCE",
        capsule_id="capsule-target5-fixed",
        split_id="split-target5-fixed",
    )


def _support() -> tuple[torch.Tensor, tuple[str, ...]]:
    centers = torch.tensor(
        [
            [1.0, 0.1],
            [0.1, 1.0],
            [-0.8, -0.6],
        ],
        dtype=torch.float32,
    )
    rows = []
    labels: list[str] = []
    for class_index, label in enumerate(("old-a", "old-b", "old-c")):
        for shot in range(2):
            value = centers[class_index].view(2, 1).repeat(1, 8)
            value = value + (shot + 1) * 0.01
            rows.append(value)
            labels.append(label)
    return torch.stack(rows), tuple(labels)


def _prototypes(model: _ToyDualModel) -> tuple[torch.Tensor, tuple[str, ...]]:
    rows = model.id_backbone.cls_head.head.weight.detach().clone()
    return rows, ("old-a", "old-b", "old-c")


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_time_fusion_updates_real_structural_weights_within_budget() -> None:
    torch.manual_seed(7)
    model = _ToyDualModel()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes(model)
    before = _state(model)
    prototypes_before = prototypes.clone()

    audit = adapt_on_target_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        prototype_class_ids,
        context=_context(),
        config=StructuredLateBlockConfig(
            candidate_id="TIME_FUSION_V1",
            steps=3,
            learning_rate=2.0e-3,
        ),
        device="cpu",
    )

    allowed_prefixes = (
        "id_backbone.t3.",
        "id_backbone.t_proj.",
        "id_backbone.fuse.",
    )
    assert 0.05 <= audit.trainable_parameter_fraction <= 0.15
    assert audit.trainable_parameter_count == 52
    assert audit.base_parameter_count == 464
    assert audit.structural_parameter_count > 0
    assert audit.changed_parameter_names
    assert any(name.endswith("weight") for name in audit.changed_parameter_names)
    assert all(name.startswith(allowed_prefixes) for name in audit.changed_parameter_names)
    assert audit.non_selected_changed_parameter_names == ()
    assert audit.steps_completed == 3
    assert audit.prototypes_unchanged is True
    assert torch.equal(prototypes, prototypes_before)
    assert torch.equal(
        model.id_backbone.cls_head.head.weight,
        before["id_backbone.cls_head.head.weight"],
    )
    assert torch.equal(model.id_backbone.reserve, before["id_backbone.reserve"])
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_schema", "p2_full_v0"),
        ("phase2_data_status", "UNVALIDATED"),
        ("capsule_id", ""),
        ("split_id", ""),
    ],
)
def test_phase2_context_fails_closed(field: str, value: str) -> None:
    values = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
    }
    values[field] = value
    context = Phase2Context(**values)
    model = _ToyDualModel()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes(model)

    with pytest.raises(StructuredLateBlockError):
        adapt_on_target_support(
            model,
            support_iq,
            support_labels,
            prototypes,
            prototype_class_ids,
            context=context,
            config=StructuredLateBlockConfig(steps=1),
        )


def test_protocol_surface_and_step_bound_reject_forbidden_inputs() -> None:
    names = set(inspect.signature(adapt_on_target_support).parameters)
    forbidden = {
        "source",
        "clean",
        "source_loader",
        "source_cache",
        "source_features",
        "source_statistics",
        "query",
        "query_labels",
        "query_truth",
        "query_roles",
        "class_quota",
    }
    assert names.isdisjoint(forbidden)
    prediction_names = set(inspect.signature(predict_query_read_only).parameters)
    assert prediction_names.isdisjoint(
        {"query_labels", "query_truth", "query_roles", "class_quota"}
    )

    with pytest.raises(StructuredLateBlockError):
        StructuredLateBlockConfig(steps=41).validate()

    with pytest.raises(TypeError):
        Phase2Context(
            protocol_schema="p2_min_v1",
            phase2_data_status="VALIDATED_ONCE",
            capsule_id="capsule-fixed",
            split_id="split-fixed",
            source_loader=object(),
        )

    model = _ToyDualModel()
    support_iq, support_labels = _support()
    wrong_prototypes, prototype_class_ids = _prototypes(model)
    wrong_prototypes[0, 0] += 0.5
    with pytest.raises(
        StructuredLateBlockError,
        match="not bound to the frozen checkpoint decision head",
    ):
        adapt_on_target_support(
            model,
            support_iq,
            support_labels,
            wrong_prototypes,
            prototype_class_ids,
            context=_context(),
            config=StructuredLateBlockConfig(steps=1),
        )


def test_query_prediction_is_read_only_and_order_independent() -> None:
    torch.manual_seed(11)
    model = _ToyDualModel()
    support_iq, support_labels = _support()
    prototypes, prototype_class_ids = _prototypes(model)
    adapt_on_target_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        prototype_class_ids,
        context=_context(),
        config=StructuredLateBlockConfig(steps=2, learning_rate=1.0e-3),
    )
    before = _state(model)
    prototypes_before = prototypes.clone()
    query = support_iq[[0, 2, 4]]
    with torch.no_grad():
        expected_scores = torch.stack(
            [
                model.id_backbone(row.unsqueeze(0), return_aux=True)["logits"]
                .squeeze(0)
                .cpu()
                for row in query
            ]
        )

    first = predict_query_read_only(
        model,
        query,
        prototypes,
        prototype_class_ids,
        context=_context(),
    )
    order = torch.tensor([2, 0, 1])
    permuted = predict_query_read_only(
        model,
        query[order],
        prototypes,
        prototype_class_ids,
        context=_context(),
    )

    assert tuple(permuted.predicted_class_ids[index] for index in (1, 2, 0)) == (
        first.predicted_class_ids
    )
    assert torch.allclose(permuted.scores[[1, 2, 0]], first.scores)
    assert torch.allclose(first.scores, expected_scores)
    assert torch.equal(prototypes, prototypes_before)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
