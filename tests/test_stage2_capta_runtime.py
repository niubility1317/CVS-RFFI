from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.stage2_capta.prototype_transport import A3_R4_SUPPORT_SHIFT
from cvsrffi.stage2_capta.runtime import (
    CaptaConfig,
    CaptaPhase2Context,
    CaptaRuntimeError,
    adapt_on_target_support,
    predict_query_read_only,
)
from cvsrffi.stage2_capta.safe_source_target_gate import select_source_weight


class _ToyCosFaceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s = 30.0
        self.weight = nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.s * F.linear(
            F.normalize(rows, dim=1), F.normalize(self.weight, dim=1)
        )


class _ToyPhysicalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = _ToyCosFaceHead()


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reserve = nn.Parameter(torch.zeros(4))
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
        z_id = x.mean(dim=-1)
        logits = self.cls_head.head(z_id)
        return {"z_id": z_id, "logits": logits} if return_aux else logits


class _ToyDualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _ToyBackbone()

    @staticmethod
    def _pick_z_id(aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return aux["z_id"]


def _context() -> CaptaPhase2Context:
    return CaptaPhase2Context(
        protocol_schema="p2_min_v1",
        phase2_data_status="VALIDATED_ONCE",
        capsule_id="capsule-fixed",
        split_id="split-fixed",
    )


def _support() -> tuple[torch.Tensor, tuple[str, ...]]:
    values = [
        (1.0, 0.0, "old-a"),
        (0.9, 0.1, "old-a"),
        (0.0, 1.0, "old-b"),
        (0.1, 0.9, "old-b"),
    ]
    rows = [torch.tensor([a, b]).view(2, 1).repeat(1, 8) for a, b, _ in values]
    return torch.stack(rows), tuple(label for _, _, label in values)


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_safe_gate_tie_prefers_the_immutable_source_path() -> None:
    scores = np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)

    result = select_source_weight(
        scores,
        scores.copy(),
        labels,
        candidate_weights=(0.0, 0.25, 0.5, 0.75, 1.0),
    )

    assert result.source_weight == 1.0
    assert result.audit["tie_break"] == "prefer_higher_source_weight"


def test_support_adaptation_has_zero_model_update_and_zero_backward() -> None:
    model = _ToyDualModel()
    support_iq, support_labels = _support()
    prototypes = model.id_backbone.cls_head.head.weight.detach().clone()
    before = _model_state(model)

    state = adapt_on_target_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        ("old-a", "old-b"),
        context=_context(),
        config=CaptaConfig(
            candidate_id=A3_R4_SUPPORT_SHIFT,
            rank=2,
            prior_strength=2.0,
        ),
    )

    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert state.audit["trainable_parameter_count"] == 0
    assert state.audit["backward_count"] == 0
    assert state.audit["query_rows_used_for_fit"] == 0
    assert state.prototype_state.target_prototypes.flags.writeable is False


def test_query_prediction_is_state_read_only_and_order_independent() -> None:
    model = _ToyDualModel()
    support_iq, support_labels = _support()
    prototypes = model.id_backbone.cls_head.head.weight.detach().clone()
    state = adapt_on_target_support(
        model,
        support_iq,
        support_labels,
        prototypes,
        ("old-a", "old-b"),
        context=_context(),
        config=CaptaConfig(
            candidate_id=A3_R4_SUPPORT_SHIFT,
            rank=2,
            prior_strength=2.0,
        ),
    )
    model_before = _model_state(model)
    prototypes_before = state.prototype_state.target_prototypes.copy()
    query = torch.stack(
        [
            torch.tensor([1.0, 0.0]).view(2, 1).repeat(1, 8),
            torch.tensor([0.0, 1.0]).view(2, 1).repeat(1, 8),
        ]
    )

    forward = predict_query_read_only(model, query, state, context=_context())
    reverse = predict_query_read_only(
        model, query.flip(0), state, context=_context()
    )

    assert forward.predicted_class_ids == tuple(reversed(reverse.predicted_class_ids))
    np.testing.assert_allclose(
        forward.mixed_scores.numpy(), reverse.mixed_scores.flip(0).numpy()
    )
    np.testing.assert_array_equal(
        state.prototype_state.target_prototypes, prototypes_before
    )
    assert all(
        torch.equal(model_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert forward.query_batch_state_updated is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_schema", "p2_hybrid_v1"),
        ("phase2_data_status", "UNVALIDATED"),
        ("capsule_id", ""),
        ("split_id", ""),
    ],
)
def test_context_rejects_non_minimal_or_unbound_rows(field: str, value: str) -> None:
    values = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
    }
    values[field] = value

    with pytest.raises(CaptaRuntimeError):
        CaptaPhase2Context(**values).validate()


def test_adaptation_api_has_no_query_or_source_surface() -> None:
    names = tuple(inspect.signature(adapt_on_target_support).parameters)
    assert names == (
        "model",
        "support_received_iq",
        "support_labels",
        "frozen_prototypes",
        "prototype_class_ids",
        "context",
        "config",
        "device",
    )
    assert all(
        token not in name.lower()
        for name in names
        for token in ("query", "source", "clean", "truth", "role", "quota")
    )
