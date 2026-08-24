"""Protocol-safe APSTA-P1 support objectives and partial adaptation runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity_only_forward import identity_only_feature_forward


_CANDIDATE_ID = "APSTA_P1_TIME_FUSION_ROBUST"
_TRAINABLE_PREFIXES = (
    "id_backbone.t3.",
    "id_backbone.t_proj.",
    "id_backbone.fuse.",
)


class ApstaError(ValueError):
    """Raised when an APSTA support objective or runtime bound is invalid."""


@dataclass(frozen=True)
class RobustClassRisk:
    per_class_loss: torch.Tensor
    mean_risk: torch.Tensor
    tail_risk: torch.Tensor


@dataclass(frozen=True)
class ApstaPhase2Context:
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str

    def validate(self) -> None:
        if self.protocol_schema != "p2_min_v1":
            raise ApstaError("protocol_schema must be p2_min_v1")
        if self.phase2_data_status != "VALIDATED_ONCE":
            raise ApstaError("phase2_data_status must be VALIDATED_ONCE")
        if not str(self.capsule_id).strip() or not str(self.split_id).strip():
            raise ApstaError("capsule_id and split_id must bind the validated row")


@dataclass(frozen=True)
class ApstaConfig:
    candidate_id: str = _CANDIDATE_ID
    checkpoints: tuple[int, ...] = (0, 10, 30, 100, 300)
    learning_rate: float = 2.0e-4
    anchor_strength: float = 3.0
    head_ce_weight: float = 0.25
    loo_mean_weight: float = 1.0
    tail_weight: float = 0.50
    tail_temperature: float = 0.50
    topology_weight: float = 0.25
    l2sp_weight: float = 1.0e-3
    margin_epsilon: float = 0.0

    def validate(self) -> None:
        if self.candidate_id != _CANDIDATE_ID:
            raise ApstaError("candidate_id is not preregistered")
        checkpoints = tuple(int(value) for value in self.checkpoints)
        if (
            not checkpoints
            or checkpoints[0] != 0
            or checkpoints != tuple(sorted(set(checkpoints)))
            or checkpoints[-1] < 1
            or checkpoints[-1] > 300
        ):
            raise ApstaError("checkpoints must be ordered from step 0 through at most 300")
        positive = (
            self.learning_rate,
            self.anchor_strength,
            self.tail_temperature,
        )
        nonnegative = (
            self.head_ce_weight,
            self.loo_mean_weight,
            self.tail_weight,
            self.topology_weight,
            self.l2sp_weight,
            self.margin_epsilon,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ApstaError("positive APSTA hyperparameters must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in nonnegative
        ):
            raise ApstaError("nonnegative APSTA hyperparameters must be finite")


@dataclass(frozen=True)
class CheckpointEvidence:
    step: int
    robust_risk: float
    worst_class_margin: float
    topology_drift: float
    parameter_drift: float


@dataclass(frozen=True)
class ApstaAudit:
    candidate_id: str
    base_parameter_count: int
    trainable_parameter_count: int
    trainable_parameter_fraction: float
    structural_parameter_count: int
    selected_parameter_names: tuple[str, ...]
    optimization_changed_parameter_names: tuple[str, ...]
    final_changed_parameter_names: tuple[str, ...]
    non_selected_changed_parameter_names: tuple[str, ...]
    changed_buffer_names: tuple[str, ...]
    backward_count: int
    steps_completed: int
    selected_step: int
    fallback_to_teacher: bool
    checkpoint_evidence: tuple[CheckpointEvidence, ...]
    loss_trace: tuple[dict[str, float | int], ...]
    prototypes_unchanged: bool
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str


@dataclass(frozen=True)
class ApstaPredictionResult:
    predicted_class_ids: tuple[str, ...]
    student_scores: torch.Tensor
    teacher_scores: torch.Tensor
    query_state_updated: bool


def _finite_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 1
        or not torch.isfinite(value).all().item()
    ):
        raise ApstaError(f"{name} must be a finite non-empty matrix")
    return value.float()


def anchored_loo_logits(
    features: torch.Tensor,
    targets: torch.Tensor,
    frozen_prototypes: torch.Tensor,
    *,
    scale: float,
    anchor_strength: float,
) -> torch.Tensor:
    """Return differentiable class logits with each row excluded from its class sum."""

    rows = _finite_matrix(features, name="features")
    anchors = _finite_matrix(frozen_prototypes, name="frozen_prototypes")
    if (
        not torch.is_tensor(targets)
        or targets.ndim != 1
        or len(targets) != len(rows)
        or anchors.shape[1] != rows.shape[1]
        or targets.dtype != torch.long
        or torch.any(targets < 0).item()
        or torch.any(targets >= len(anchors)).item()
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or not math.isfinite(float(anchor_strength))
        or float(anchor_strength) < 0.0
    ):
        raise ApstaError("LOO feature, target, anchor, or scale binding is invalid")
    normalized_rows = F.normalize(rows, dim=1, eps=1.0e-4)
    normalized_anchors = F.normalize(anchors, dim=1, eps=1.0e-4)
    one_hot = F.one_hot(targets, num_classes=len(anchors)).to(rows.dtype)
    class_sums = one_hot.transpose(0, 1) @ normalized_rows
    anchored_sums = float(anchor_strength) * normalized_anchors + class_sums
    loo_sums = (
        anchored_sums.unsqueeze(0)
        - one_hot.unsqueeze(2) * normalized_rows.unsqueeze(1)
    )
    loo_prototypes = F.normalize(loo_sums, dim=2, eps=1.0e-4)
    return float(scale) * torch.einsum(
        "nd,nkd->nk", normalized_rows, loo_prototypes
    )


def robust_class_risk(
    per_sample_loss: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_count: int,
    temperature: float,
) -> RobustClassRisk:
    """Aggregate support loss without allowing easy classes to hide a weak class."""

    if (
        not torch.is_tensor(per_sample_loss)
        or per_sample_loss.ndim != 1
        or not torch.isfinite(per_sample_loss).all().item()
        or not torch.is_tensor(targets)
        or targets.ndim != 1
        or len(targets) != len(per_sample_loss)
        or targets.dtype != torch.long
        or int(class_count) < 2
        or torch.any(targets < 0).item()
        or torch.any(targets >= int(class_count)).item()
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ApstaError("class-robust risk inputs are invalid")
    sums = torch.zeros(
        int(class_count), dtype=per_sample_loss.dtype, device=per_sample_loss.device
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, targets, per_sample_loss)
    counts.scatter_add_(0, targets, torch.ones_like(per_sample_loss))
    if torch.any(counts <= 0).item():
        raise ApstaError("every registered class needs legal target support")
    per_class = sums / counts
    mean_risk = per_class.mean()
    tail_risk = float(temperature) * (
        torch.logsumexp(per_class / float(temperature), dim=0)
        - math.log(int(class_count))
    )
    return RobustClassRisk(
        per_class_loss=per_class,
        mean_risk=mean_risk,
        tail_risk=tail_risk,
    )


def prototype_topology_drift(
    target_prototypes: torch.Tensor,
    frozen_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Measure class-angle drift using normalized prototype Gram matrices."""

    target = _finite_matrix(target_prototypes, name="target_prototypes")
    frozen = _finite_matrix(frozen_prototypes, name="frozen_prototypes")
    if target.shape != frozen.shape:
        raise ApstaError("target and frozen prototype geometry must align")
    target = F.normalize(target, dim=1, eps=1.0e-4)
    frozen = F.normalize(frozen, dim=1, eps=1.0e-4)
    return F.mse_loss(target @ target.transpose(0, 1), frozen @ frozen.transpose(0, 1))


def select_safe_checkpoint(
    evidence: Sequence[CheckpointEvidence],
    *,
    margin_epsilon: float = 0.0,
) -> CheckpointEvidence:
    """Choose an adapted checkpoint only when risk and worst-class margin are safe."""

    items = tuple(evidence)
    if not items or items[0].step != 0 or any(
        not math.isfinite(float(value))
        for item in items
        for value in (
            item.robust_risk,
            item.worst_class_margin,
            item.topology_drift,
            item.parameter_drift,
        )
    ):
        raise ApstaError("checkpoint evidence must start with a finite step-0 baseline")
    if not math.isfinite(float(margin_epsilon)) or float(margin_epsilon) < 0.0:
        raise ApstaError("margin_epsilon must be finite and nonnegative")
    baseline = items[0]
    eligible = [
        item
        for item in items[1:]
        if item.robust_risk <= baseline.robust_risk + 1.0e-12
        and item.worst_class_margin
        >= baseline.worst_class_margin - float(margin_epsilon) - 1.0e-12
    ]
    if not eligible:
        return baseline
    return min(
        eligible,
        key=lambda item: (
            item.robust_risk,
            -item.worst_class_margin,
            item.topology_drift,
            item.parameter_drift,
            item.step,
        ),
    )


def _float_tensor_from_values(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    plain = value.tolist() if hasattr(value, "tolist") else value
    return torch.tensor(plain, dtype=torch.float32)


def _validate_received_iq(value: Any, *, name: str) -> torch.Tensor:
    rows = _float_tensor_from_values(value)
    if (
        rows.ndim != 3
        or rows.shape[0] < 1
        or rows.shape[1] != 2
        or not torch.isfinite(rows).all().item()
    ):
        raise ApstaError(f"{name} must be finite received IQ with shape [N, 2, L]")
    return rows


def _validate_prototypes(
    value: Any,
    prototype_class_ids: Sequence[str],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    prototypes = _float_tensor_from_values(value)
    class_ids = tuple(str(item) for item in prototype_class_ids)
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] < 2
        or prototypes.shape[0] != len(class_ids)
        or len(set(class_ids)) != len(class_ids)
        or any(not item for item in class_ids)
        or not torch.isfinite(prototypes).all().item()
        or torch.any(torch.linalg.vector_norm(prototypes, dim=1) <= 0.0).item()
    ):
        raise ApstaError("frozen prototypes and class mapping are invalid")
    return prototypes.detach().clone(), class_ids


def _identity_forward(
    model: nn.Module,
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = identity_only_feature_forward(model, rows, "z_id")
    if output is None:
        raise ApstaError("model does not expose the ADV3B02 identity-only forward")
    features, logits = output
    if (
        features.ndim != 2
        or logits.ndim != 2
        or len(features) != len(logits)
        or not torch.isfinite(features).all().item()
        or not torch.isfinite(logits).all().item()
    ):
        raise ApstaError("identity output is invalid")
    return features, logits


def _validate_checkpoint_prototype_binding(
    model: nn.Module,
    prototypes: torch.Tensor,
) -> float:
    try:
        head = model.id_backbone.cls_head.head
        checkpoint_weight = head.weight
        scale = float(head.s)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ApstaError("checkpoint does not expose frozen CosFace anchors") from exc
    if (
        not torch.is_tensor(checkpoint_weight)
        or checkpoint_weight.ndim != 2
        or checkpoint_weight.shape != prototypes.shape
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ApstaError("prototype shape or frozen CosFace scale is invalid")
    expected = F.normalize(checkpoint_weight.detach().float().cpu(), dim=1, eps=1.0e-4)
    observed = F.normalize(prototypes.detach().float().cpu(), dim=1, eps=1.0e-4)
    if not torch.allclose(observed, expected, rtol=1.0e-5, atol=1.0e-6):
        raise ApstaError("prototypes are not bound to the frozen checkpoint head")
    return scale


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _select_trainable_parameters(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], int, int, float, int]:
    named = list(model.named_parameters())
    selected = [
        (name, parameter)
        for name, parameter in named
        if name.startswith(_TRAINABLE_PREFIXES)
    ]
    missing = [
        prefix
        for prefix in _TRAINABLE_PREFIXES
        if not any(name.startswith(prefix) for name, _ in selected)
    ]
    base_count = int(sum(parameter.numel() for _, parameter in named))
    if base_count <= 0 or not selected or missing:
        raise ApstaError("the preregistered time-fusion blocks are missing")
    selected_count = int(sum(parameter.numel() for _, parameter in selected))
    structural_count = int(
        sum(
            parameter.numel()
            for name, parameter in selected
            if parameter.ndim >= 2
            and "norm" not in name.lower()
            and "gate" not in name.lower()
            and not name.lower().endswith("bias")
        )
    )
    if structural_count <= 0:
        raise ApstaError("candidate degenerates to Norm/Bias/Gate-only adaptation")
    return (
        selected,
        base_count,
        selected_count,
        float(selected_count / base_count),
        structural_count,
    )


def _anchored_prototypes(
    features: torch.Tensor,
    targets: torch.Tensor,
    frozen_prototypes: torch.Tensor,
    anchor_strength: float,
) -> torch.Tensor:
    normalized = F.normalize(features, dim=1, eps=1.0e-4)
    anchors = F.normalize(frozen_prototypes, dim=1, eps=1.0e-4)
    one_hot = F.one_hot(targets, num_classes=len(anchors)).to(normalized.dtype)
    sums = one_hot.transpose(0, 1) @ normalized
    return F.normalize(float(anchor_strength) * anchors + sums, dim=1, eps=1.0e-4)


def _class_mean(values: torch.Tensor, targets: torch.Tensor, count: int) -> torch.Tensor:
    sums = torch.zeros(count, dtype=values.dtype, device=values.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, targets, values)
    counts.scatter_add_(0, targets, torch.ones_like(values))
    if torch.any(counts <= 0).item():
        raise ApstaError("every registered class needs legal target support")
    return sums / counts


def _support_objective(
    model: nn.Module,
    rows: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    scale: float,
    selected: Sequence[tuple[str, nn.Parameter]],
    initial_selected: dict[str, torch.Tensor],
    config: ApstaConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features, head_logits = _identity_forward(model, rows)
    if features.shape[1] != prototypes.shape[1] or head_logits.shape[1] != len(prototypes):
        raise ApstaError("feature, head, and prototype dimensions do not align")
    loo_logits = anchored_loo_logits(
        features,
        targets,
        prototypes,
        scale=scale,
        anchor_strength=config.anchor_strength,
    )
    per_sample_loo = F.cross_entropy(loo_logits, targets, reduction="none")
    risk = robust_class_risk(
        per_sample_loo,
        targets,
        class_count=len(prototypes),
        temperature=config.tail_temperature,
    )
    target_prototypes = _anchored_prototypes(
        features, targets, prototypes, config.anchor_strength
    )
    topology = prototype_topology_drift(target_prototypes, prototypes)
    parameter_drift = torch.stack(
        [
            (parameter - initial_selected[name]).pow(2).mean()
            for name, parameter in selected
        ]
    ).mean()
    head_ce = F.cross_entropy(head_logits, targets)
    total = (
        float(config.head_ce_weight) * head_ce
        + float(config.loo_mean_weight) * risk.mean_risk
        + float(config.tail_weight) * risk.tail_risk
        + float(config.topology_weight) * topology
        + float(config.l2sp_weight) * parameter_drift
    )
    return total, {
        "head_ce": head_ce,
        "loo_mean": risk.mean_risk,
        "loo_tail": risk.tail_risk,
        "topology": topology,
        "parameter_drift": parameter_drift,
        "loo_logits": loo_logits,
    }


def _checkpoint_evidence(
    *,
    step: int,
    model: nn.Module,
    rows: torch.Tensor,
    targets: torch.Tensor,
    prototypes: torch.Tensor,
    scale: float,
    selected: Sequence[tuple[str, nn.Parameter]],
    initial_selected: dict[str, torch.Tensor],
    config: ApstaConfig,
) -> CheckpointEvidence:
    with torch.no_grad():
        _, parts = _support_objective(
            model,
            rows,
            targets,
            prototypes,
            scale,
            selected,
            initial_selected,
            config,
        )
        logits = parts["loo_logits"]
        correct = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        other = logits.masked_fill(
            F.one_hot(targets, num_classes=logits.shape[1]).bool(),
            -torch.inf,
        ).max(dim=1).values
        class_margins = _class_mean(correct - other, targets, logits.shape[1])
        robust_risk_value = (
            float(config.loo_mean_weight) * parts["loo_mean"]
            + float(config.tail_weight) * parts["loo_tail"]
        )
    return CheckpointEvidence(
        step=int(step),
        robust_risk=float(robust_risk_value.cpu()),
        worst_class_margin=float(class_margins.min().cpu()),
        topology_drift=float(parts["topology"].cpu()),
        parameter_drift=float(parts["parameter_drift"].cpu()),
    )


def adapt_on_target_support(
    model: nn.Module,
    support_received_iq: Any,
    support_labels: Sequence[str],
    frozen_prototypes: Any,
    prototype_class_ids: Sequence[str],
    context: ApstaPhase2Context,
    config: ApstaConfig,
    device: torch.device | str = "cpu",
) -> ApstaAudit:
    """Run support-only APSTA; no source, query, or trainable-head surface exists."""

    context.validate()
    config.validate()
    if not isinstance(model, nn.Module):
        raise ApstaError("a frozen ADV3B02 checkpoint model is required")
    rows = _validate_received_iq(support_received_iq, name="support_received_iq")
    labels = tuple(str(value) for value in support_labels)
    if not labels or len(labels) != len(rows):
        raise ApstaError("support label alignment is invalid")
    prototypes_cpu, class_ids = _validate_prototypes(
        frozen_prototypes, prototype_class_ids
    )
    lookup = {class_id: index for index, class_id in enumerate(class_ids)}
    if any(label not in lookup for label in labels):
        raise ApstaError("support label is absent from the frozen class mapping")

    target_device = torch.device(device)
    model.to(target_device)
    _freeze(model)
    scale = _validate_checkpoint_prototype_binding(model, prototypes_cpu)
    selected, base_count, selected_count, fraction, structural_count = (
        _select_trainable_parameters(model)
    )
    selected_names = tuple(name for name, _ in selected)
    selected_name_set = set(selected_names)
    before_state = _copy_state(model)
    initial_selected = {
        name: parameter.detach().clone() for name, parameter in selected
    }
    prototypes_before = prototypes_cpu.detach().clone()
    rows = rows.to(target_device)
    prototypes = prototypes_cpu.to(target_device)
    targets = torch.tensor(
        [lookup[label] for label in labels],
        dtype=torch.long,
        device=target_device,
    )
    if set(targets.detach().cpu().tolist()) != set(range(len(class_ids))):
        raise ApstaError("every frozen class needs legal target support")

    for _, parameter in selected:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=float(config.learning_rate),
        weight_decay=0.0,
    )
    evidence: list[CheckpointEvidence] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    trace: list[dict[str, float | int]] = []
    backward_count = 0

    def capture(step: int) -> None:
        evidence.append(
            _checkpoint_evidence(
                step=step,
                model=model,
                rows=rows,
                targets=targets,
                prototypes=prototypes,
                scale=scale,
                selected=selected,
                initial_selected=initial_selected,
                config=config,
            )
        )
        snapshots[step] = {
            name: parameter.detach().clone() for name, parameter in selected
        }

    try:
        capture(0)
        checkpoint_set = set(config.checkpoints[1:])
        for step in range(1, int(config.checkpoints[-1]) + 1):
            optimizer.zero_grad(set_to_none=True)
            total, parts = _support_objective(
                model,
                rows,
                targets,
                prototypes,
                scale,
                selected,
                initial_selected,
                config,
            )
            if not torch.isfinite(total).item():
                raise ApstaError("support-only APSTA loss is non-finite")
            total.backward()
            backward_count += 1
            optimizer.step()
            trace.append(
                {
                    "step": step,
                    "loss": float(total.detach().cpu()),
                    "head_ce": float(parts["head_ce"].detach().cpu()),
                    "loo_mean": float(parts["loo_mean"].detach().cpu()),
                    "loo_tail": float(parts["loo_tail"].detach().cpu()),
                    "topology": float(parts["topology"].detach().cpu()),
                    "parameter_drift": float(parts["parameter_drift"].detach().cpu()),
                }
            )
            if step in checkpoint_set:
                capture(step)

        optimized_state = _copy_state(model)
        selected_evidence = select_safe_checkpoint(
            evidence, margin_epsilon=config.margin_epsilon
        )
        with torch.no_grad():
            for name, parameter in selected:
                parameter.copy_(snapshots[selected_evidence.step][name])
    finally:
        _freeze(model)

    after_state = model.state_dict()
    parameter_names = {name for name, _ in model.named_parameters()}
    optimization_changed = tuple(
        name
        for name in selected_names
        if not torch.equal(before_state[name], optimized_state[name])
    )
    final_changed = tuple(
        name
        for name in selected_names
        if not torch.equal(before_state[name], after_state[name])
    )
    non_selected_changed = tuple(
        sorted(
            name
            for name in parameter_names - selected_name_set
            if not torch.equal(before_state[name], after_state[name])
        )
    )
    changed_buffers = tuple(
        sorted(
            name
            for name in after_state
            if name not in parameter_names
            and not torch.equal(before_state[name], after_state[name])
        )
    )
    prototypes_unchanged = torch.equal(prototypes_before, prototypes_cpu)
    if not optimization_changed:
        raise ApstaError("support loss produced no real selected-block update")
    if non_selected_changed or changed_buffers:
        raise ApstaError("adaptation changed frozen parameters or buffers")
    if not prototypes_unchanged:
        raise ApstaError("frozen prototypes were modified")

    return ApstaAudit(
        candidate_id=config.candidate_id,
        base_parameter_count=base_count,
        trainable_parameter_count=selected_count,
        trainable_parameter_fraction=fraction,
        structural_parameter_count=structural_count,
        selected_parameter_names=selected_names,
        optimization_changed_parameter_names=optimization_changed,
        final_changed_parameter_names=final_changed,
        non_selected_changed_parameter_names=non_selected_changed,
        changed_buffer_names=changed_buffers,
        backward_count=backward_count,
        steps_completed=backward_count,
        selected_step=selected_evidence.step,
        fallback_to_teacher=selected_evidence.step == 0,
        checkpoint_evidence=tuple(evidence),
        loss_trace=tuple(trace),
        prototypes_unchanged=prototypes_unchanged,
        protocol_schema=context.protocol_schema,
        phase2_data_status=context.phase2_data_status,
        capsule_id=context.capsule_id,
        split_id=context.split_id,
    )


def predict_query_read_only(
    student_model: nn.Module,
    teacher_model: nn.Module,
    query_received_iq: Any,
    frozen_prototypes: Any,
    prototype_class_ids: Sequence[str],
    context: ApstaPhase2Context,
) -> ApstaPredictionResult:
    """Score each query independently with frozen student and teacher checkpoints."""

    context.validate()
    if not isinstance(student_model, nn.Module) or not isinstance(teacher_model, nn.Module):
        raise ApstaError("student and teacher checkpoint models are required")
    _freeze(student_model)
    _freeze(teacher_model)
    rows = _validate_received_iq(query_received_iq, name="query_received_iq")
    prototypes_cpu, class_ids = _validate_prototypes(
        frozen_prototypes, prototype_class_ids
    )
    _validate_checkpoint_prototype_binding(student_model, prototypes_cpu)
    _validate_checkpoint_prototype_binding(teacher_model, prototypes_cpu)
    student_before = _copy_state(student_model)
    teacher_before = _copy_state(teacher_model)
    try:
        student_device = next(student_model.parameters()).device
        teacher_device = next(teacher_model.parameters()).device
    except StopIteration as exc:
        raise ApstaError("checkpoint model has no parameters") from exc
    student_scores: list[torch.Tensor] = []
    teacher_scores: list[torch.Tensor] = []
    with torch.no_grad():
        for row in rows:
            student_features, student_logits = _identity_forward(
                student_model, row.unsqueeze(0).to(student_device)
            )
            teacher_features, teacher_logits = _identity_forward(
                teacher_model, row.unsqueeze(0).to(teacher_device)
            )
            if (
                student_features.shape[1] != prototypes_cpu.shape[1]
                or teacher_features.shape[1] != prototypes_cpu.shape[1]
                or student_logits.shape[1] != len(class_ids)
                or teacher_logits.shape[1] != len(class_ids)
            ):
                raise ApstaError("query feature and decision dimensions do not align")
            student_scores.append(student_logits.squeeze(0).cpu())
            teacher_scores.append(teacher_logits.squeeze(0).cpu())
    changed = any(
        not torch.equal(student_before[name], value)
        for name, value in student_model.state_dict().items()
    ) or any(
        not torch.equal(teacher_before[name], value)
        for name, value in teacher_model.state_dict().items()
    )
    if changed:
        raise ApstaError("query inference modified checkpoint state")
    student_tensor = torch.stack(student_scores)
    teacher_tensor = torch.stack(teacher_scores)
    predicted = tuple(class_ids[int(index)] for index in student_tensor.argmax(dim=1))
    return ApstaPredictionResult(
        predicted_class_ids=predicted,
        student_scores=student_tensor,
        teacher_scores=teacher_tensor,
        query_state_updated=False,
    )
