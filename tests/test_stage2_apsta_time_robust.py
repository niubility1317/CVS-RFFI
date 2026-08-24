from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.stage2_apsta_time_robust import (
    ApstaConfig,
    ApstaPhase2Context,
    CheckpointEvidence,
    adapt_on_target_support,
    anchored_loo_logits,
    predict_query_read_only,
    prototype_topology_drift,
    robust_class_risk,
    select_safe_checkpoint,
)


class _ToyCosFaceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s = 10.0
        self.weight = nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.s * F.linear(
            F.normalize(rows, dim=1), F.normalize(self.weight, dim=1)
        )


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reserve = nn.Parameter(torch.zeros(32))
        self.t3 = nn.Linear(2, 4)
        self.t_proj = nn.Linear(4, 4)
        self.fuse = nn.Linear(4, 2)
        self.cls_head = nn.Module()
        self.cls_head.head = _ToyCosFaceHead()

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
        return {"z_id": z_id, "logits": logits} if return_aux else logits


class _ToyDualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _ToyBackbone()

    @staticmethod
    def _pick_z_id(aux: dict[str, torch.Tensor]) -> torch.Tensor:
        return aux["z_id"]


def _support() -> tuple[torch.Tensor, tuple[str, ...]]:
    values = [
        (1.0, 0.0, "old-a"),
        (0.8, 0.2, "old-a"),
        (0.0, 1.0, "old-b"),
        (0.2, 0.8, "old-b"),
    ]
    rows = [torch.tensor([a, b]).view(2, 1).repeat(1, 8) for a, b, _ in values]
    return torch.stack(rows), tuple(label for _, _, label in values)


def _context() -> ApstaPhase2Context:
    return ApstaPhase2Context(
        protocol_schema="p2_min_v1",
        phase2_data_status="VALIDATED_ONCE",
        capsule_id="capsule-fixed",
        split_id="split-fixed",
    )


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_anchored_loo_excludes_the_query_sample_from_its_class_prototype() -> None:
    features = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    logits = anchored_loo_logits(
        features,
        targets,
        anchors,
        scale=1.0,
        anchor_strength=1.0,
    )

    assert logits.shape == (4, 2)
    assert logits[0, 0].item() == 0.0
    assert logits[0, 1].item() == 1.0


def test_tail_risk_increases_for_one_weak_class_and_is_label_permutation_invariant() -> None:
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    uniform = torch.tensor([1.0, 1.0, 1.0, 1.0])
    weak_tail = torch.tensor([1.0, 1.0, 1.0, 5.0])

    uniform_risk = robust_class_risk(
        uniform, targets, class_count=2, temperature=0.5
    )
    weak_risk = robust_class_risk(
        weak_tail, targets, class_count=2, temperature=0.5
    )
    permuted = robust_class_risk(
        weak_tail,
        1 - targets,
        class_count=2,
        temperature=0.5,
    )

    assert weak_risk.tail_risk.item() > uniform_risk.tail_risk.item()
    assert torch.equal(weak_risk.per_class_loss, permuted.per_class_loss.flip(0))
    assert torch.allclose(weak_risk.mean_risk, permuted.mean_risk)
    assert torch.allclose(weak_risk.tail_risk, permuted.tail_risk)


def test_topology_drift_is_zero_only_when_class_geometry_matches_anchors() -> None:
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    matching = prototype_topology_drift(anchors, anchors)
    collapsed = prototype_topology_drift(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]), anchors
    )

    assert matching.item() == 0.0
    assert collapsed.item() > 0.0


def test_safe_checkpoint_requires_both_robust_risk_and_worst_margin() -> None:
    baseline = CheckpointEvidence(
        step=0,
        robust_risk=1.0,
        worst_class_margin=0.20,
        topology_drift=0.0,
        parameter_drift=0.0,
    )
    risk_only = CheckpointEvidence(
        step=10,
        robust_risk=0.8,
        worst_class_margin=0.19,
        topology_drift=0.01,
        parameter_drift=0.01,
    )
    pareto_safe = CheckpointEvidence(
        step=30,
        robust_risk=0.9,
        worst_class_margin=0.21,
        topology_drift=0.02,
        parameter_drift=0.02,
    )

    assert select_safe_checkpoint((baseline, risk_only)).step == 0
    assert select_safe_checkpoint((baseline, risk_only, pareto_safe)).step == 30


def test_partial_adaptation_backpropagates_only_time_fusion_and_freezes_query() -> None:
    torch.manual_seed(7)
    student = _ToyDualModel()
    teacher = _ToyDualModel()
    teacher.load_state_dict(student.state_dict())
    support_iq, support_labels = _support()
    prototypes = student.id_backbone.cls_head.head.weight.detach().clone()
    initial = _state(student)

    audit = adapt_on_target_support(
        student,
        support_iq,
        support_labels,
        prototypes,
        ("old-a", "old-b"),
        context=_context(),
        config=ApstaConfig(checkpoints=(0, 1, 3), learning_rate=1.0e-3),
    )

    assert audit.backward_count == 3
    assert audit.steps_completed == 3
    assert tuple(item.step for item in audit.checkpoint_evidence) == (0, 1, 3)
    assert audit.selected_step in (0, 1, 3)
    assert audit.optimization_changed_parameter_names
    assert all(
        name.startswith(
            ("id_backbone.t3.", "id_backbone.t_proj.", "id_backbone.fuse.")
        )
        for name in audit.optimization_changed_parameter_names
    )
    assert audit.non_selected_changed_parameter_names == ()
    assert audit.changed_buffer_names == ()
    assert torch.equal(student.id_backbone.cls_head.head.weight, prototypes)
    assert torch.equal(teacher.id_backbone.cls_head.head.weight, prototypes)
    assert all(not parameter.requires_grad for parameter in student.parameters())

    before_query = _state(student)
    prediction = predict_query_read_only(
        student,
        teacher,
        support_iq[:2],
        prototypes,
        ("old-a", "old-b"),
        context=_context(),
    )

    assert prediction.student_scores.shape == (2, 2)
    assert prediction.teacher_scores.shape == (2, 2)
    assert prediction.query_state_updated is False
    assert all(
        torch.equal(before_query[name], value)
        for name, value in student.state_dict().items()
    )
    assert all(torch.equal(initial[name], teacher.state_dict()[name]) for name in initial)


def test_adaptation_api_has_no_source_query_or_trainable_head_surface() -> None:
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
        for token in ("source", "query", "clean", "truth", "role", "head")
    )


def test_formal_300_step_schedule_is_valid_without_parameter_fraction_gate() -> None:
    config = ApstaConfig()

    config.validate()
    assert config.checkpoints == (0, 10, 30, 100, 300)
