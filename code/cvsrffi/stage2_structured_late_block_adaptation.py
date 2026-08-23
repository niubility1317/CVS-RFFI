"""Support-only Stage2-B adaptation of late non-classifier encoder blocks.

The public surface is intentionally limited to fixed target received IQ,
legal support labels, immutable class prototypes/class mapping, the frozen
checkpoint model, and preregistered configuration.  There is no source,
clean, replay, query-label, query-role, quota, or trainable-head interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity_only_forward import identity_only_feature_forward


MAX_ADAPTATION_STEPS = 40
TARGET_MIN_PARAMETER_FRACTION = 0.05
TARGET_MAX_PARAMETER_FRACTION = 0.15
HARD_MAX_PARAMETER_FRACTION = 0.20

_CANDIDATE_PREFIXES: dict[str, tuple[str, ...]] = {
    "TIME_FUSION_V1": (
        "id_backbone.t3.",
        "id_backbone.t_proj.",
        "id_backbone.fuse.",
    ),
    "FREQ_FUSION_V1": (
        "id_backbone.f3.",
        "id_backbone.f_proj.",
        "id_backbone.fuse.",
    ),
}


class StructuredLateBlockError(ValueError):
    """Raised when the Phase2 whitelist or adaptation budget fails closed."""


@dataclass(frozen=True)
class Phase2Context:
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str

    def validate(self) -> None:
        if self.protocol_schema != "p2_min_v1":
            raise StructuredLateBlockError("protocol_schema must be p2_min_v1")
        if self.phase2_data_status != "VALIDATED_ONCE":
            raise StructuredLateBlockError(
                "phase2_data_status must be VALIDATED_ONCE"
            )
        if not str(self.capsule_id).strip() or not str(self.split_id).strip():
            raise StructuredLateBlockError(
                "capsule_id and split_id must bind the validated Phase2 row"
            )


@dataclass(frozen=True)
class StructuredLateBlockConfig:
    candidate_id: str = "TIME_FUSION_V1"
    steps: int = 24
    learning_rate: float = 2.0e-4
    prototype_anchor_weight: float = 0.50
    feature_drift_weight: float = 1.00
    parameter_drift_weight: float = 1.0e-3

    def validate(self) -> None:
        if self.candidate_id not in _CANDIDATE_PREFIXES:
            raise StructuredLateBlockError("candidate_id is not preregistered")
        if not 1 <= int(self.steps) <= MAX_ADAPTATION_STEPS:
            raise StructuredLateBlockError("adaptation steps exceed the hard bound")
        positive = (self.learning_rate,)
        nonnegative = (
            self.prototype_anchor_weight,
            self.feature_drift_weight,
            self.parameter_drift_weight,
        )
        if not all(torch.isfinite(torch.tensor(value)).item() for value in positive):
            raise StructuredLateBlockError("positive hyperparameters must be finite")
        if not all(value > 0.0 for value in positive):
            raise StructuredLateBlockError("learning rate must be positive")
        if not all(
            torch.isfinite(torch.tensor(value)).item() and value >= 0.0
            for value in nonnegative
        ):
            raise StructuredLateBlockError(
                "loss weights must be finite and nonnegative"
            )


@dataclass(frozen=True)
class AdaptationAudit:
    candidate_id: str
    base_parameter_count: int
    trainable_parameter_count: int
    trainable_parameter_fraction: float
    structural_parameter_count: int
    selected_parameter_names: tuple[str, ...]
    changed_parameter_names: tuple[str, ...]
    non_selected_changed_parameter_names: tuple[str, ...]
    changed_buffer_names: tuple[str, ...]
    steps_completed: int
    loss_trace: tuple[dict[str, float | int], ...]
    prototypes_unchanged: bool
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str


@dataclass(frozen=True)
class PredictionResult:
    predicted_class_ids: tuple[str, ...]
    scores: torch.Tensor


def _validate_received_iq(value: Any, *, name: str) -> torch.Tensor:
    rows = torch.as_tensor(value, dtype=torch.float32)
    if (
        rows.ndim != 3
        or rows.shape[0] < 1
        or rows.shape[1] != 2
        or not torch.isfinite(rows).all().item()
    ):
        raise StructuredLateBlockError(
            f"{name} must be finite received IQ with shape [N, 2, L]"
        )
    return rows


def _validate_prototypes(
    frozen_prototypes: Any,
    prototype_class_ids: Sequence[str],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    prototypes = torch.as_tensor(frozen_prototypes, dtype=torch.float32)
    class_ids = tuple(str(value) for value in prototype_class_ids)
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] < 2
        or prototypes.shape[0] != len(class_ids)
        or len(set(class_ids)) != len(class_ids)
        or any(not value for value in class_ids)
        or not torch.isfinite(prototypes).all().item()
        or torch.any(torch.linalg.vector_norm(prototypes, dim=1) <= 0.0).item()
    ):
        raise StructuredLateBlockError(
            "frozen prototypes and immutable class mapping are invalid"
        )
    return prototypes.detach().clone(), class_ids


def _identity_forward(
    model: nn.Module, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = identity_only_feature_forward(model, rows, "z_id")
    if output is None:
        raise StructuredLateBlockError(
            "model does not expose the exact ADV3B02 identity-only forward"
        )
    features, logits = output
    if (
        features.ndim != 2
        or logits.ndim != 2
        or len(features) != len(logits)
        or not torch.isfinite(features).all().item()
        or not torch.isfinite(logits).all().item()
    ):
        raise StructuredLateBlockError("identity feature output is invalid")
    return features, logits


def _validate_checkpoint_prototype_binding(
    model: nn.Module, prototypes: torch.Tensor
) -> None:
    try:
        checkpoint_weight = model.id_backbone.cls_head.head.weight
    except AttributeError as exc:
        raise StructuredLateBlockError(
            "checkpoint does not expose the frozen CosFace class anchors"
        ) from exc
    if (
        not torch.is_tensor(checkpoint_weight)
        or checkpoint_weight.ndim != 2
        or checkpoint_weight.shape != prototypes.shape
    ):
        raise StructuredLateBlockError(
            "prototype shape does not match the frozen CosFace class anchors"
        )
    expected = F.normalize(checkpoint_weight.detach().float().cpu(), dim=1, eps=1.0e-4)
    observed = F.normalize(prototypes.detach().float().cpu(), dim=1, eps=1.0e-4)
    if not torch.allclose(observed, expected, rtol=1.0e-5, atol=1.0e-6):
        raise StructuredLateBlockError(
            "prototype rows are not bound to the frozen checkpoint decision head"
        )


def _select_trainable_parameters(
    model: nn.Module,
    candidate_id: str,
) -> tuple[
    list[tuple[str, nn.Parameter]],
    int,
    int,
    float,
    int,
]:
    prefixes = _CANDIDATE_PREFIXES[candidate_id]
    named = list(model.named_parameters())
    base_count = int(sum(parameter.numel() for _, parameter in named))
    selected = [
        (name, parameter)
        for name, parameter in named
        if name.startswith(prefixes)
    ]
    missing = [
        prefix
        for prefix in prefixes
        if not any(name.startswith(prefix) for name, _ in selected)
    ]
    if base_count <= 0 or not selected or missing:
        raise StructuredLateBlockError(
            "preregistered late feature blocks are missing from the checkpoint model"
        )
    selected_count = int(sum(parameter.numel() for _, parameter in selected))
    fraction = float(selected_count / base_count)
    if fraction > HARD_MAX_PARAMETER_FRACTION:
        raise StructuredLateBlockError("trainable parameter fraction exceeds 20%")
    if not (
        TARGET_MIN_PARAMETER_FRACTION
        <= fraction
        <= TARGET_MAX_PARAMETER_FRACTION
    ):
        raise StructuredLateBlockError(
            "trainable parameter fraction is outside the preregistered 5%-15% target"
        )
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
        raise StructuredLateBlockError(
            "candidate degenerates to Norm/Bias/Gate-only adaptation"
        )
    return selected, base_count, selected_count, fraction, structural_count


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def _freeze_for_inference(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def adapt_on_target_support(
    model: nn.Module,
    support_received_iq: Any,
    support_labels: Sequence[str],
    frozen_prototypes: Any,
    prototype_class_ids: Sequence[str],
    *,
    context: Phase2Context,
    config: StructuredLateBlockConfig,
    device: torch.device | str = "cpu",
) -> AdaptationAudit:
    """Backpropagate through preregistered late blocks using target support only."""

    context.validate()
    config.validate()
    if not isinstance(model, nn.Module):
        raise StructuredLateBlockError("frozen checkpoint model is required")
    rows = _validate_received_iq(support_received_iq, name="support_received_iq")
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(rows) or not labels:
        raise StructuredLateBlockError("support label alignment is invalid")
    prototypes_cpu, class_ids = _validate_prototypes(
        frozen_prototypes, prototype_class_ids
    )
    class_lookup = {label: index for index, label in enumerate(class_ids)}
    if any(label not in class_lookup for label in labels):
        raise StructuredLateBlockError(
            "support label is absent from the frozen prototype mapping"
        )

    target_device = torch.device(device)
    model.to(target_device)
    _freeze_for_inference(model)
    _validate_checkpoint_prototype_binding(model, prototypes_cpu)
    selected, base_count, selected_count, fraction, structural_count = (
        _select_trainable_parameters(model, config.candidate_id)
    )
    before_state = _copy_state(model)
    selected_names = tuple(name for name, _ in selected)
    selected_name_set = set(selected_names)
    initial_selected = {
        name: parameter.detach().clone() for name, parameter in selected
    }
    rows = rows.to(target_device)
    prototypes = prototypes_cpu.to(target_device)
    targets = torch.tensor(
        [class_lookup[label] for label in labels],
        dtype=torch.long,
        device=target_device,
    )

    with torch.no_grad():
        frozen_features, _ = _identity_forward(model, rows)
        frozen_features = frozen_features.detach()
    if frozen_features.shape[1] != prototypes.shape[1]:
        raise StructuredLateBlockError(
            "frozen prototype dimension does not match checkpoint identity features"
        )

    for _, parameter in selected:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=float(config.learning_rate),
        weight_decay=0.0,
    )
    trace: list[dict[str, float | int]] = []
    try:
        normalized_prototypes = F.normalize(prototypes, dim=1)
        frozen_normalized = F.normalize(frozen_features, dim=1)
        for step in range(int(config.steps)):
            optimizer.zero_grad(set_to_none=True)
            features, logits = _identity_forward(model, rows)
            if logits.shape[1] != len(class_ids):
                raise StructuredLateBlockError(
                    "frozen decision head class count does not match prototype mapping"
                )
            normalized = F.normalize(features, dim=1)
            supervised_loss = F.cross_entropy(logits, targets)
            prototype_anchor_loss = (
                1.0
                - torch.sum(
                    normalized * normalized_prototypes[targets], dim=1
                )
            ).mean()
            feature_drift_loss = F.mse_loss(normalized, frozen_normalized)
            parameter_drift_loss = torch.stack(
                [
                    (parameter - initial_selected[name]).pow(2).mean()
                    for name, parameter in selected
                ]
            ).mean()
            loss = (
                supervised_loss
                + float(config.prototype_anchor_weight) * prototype_anchor_loss
                + float(config.feature_drift_weight) * feature_drift_loss
                + float(config.parameter_drift_weight) * parameter_drift_loss
            )
            if not torch.isfinite(loss).item():
                raise StructuredLateBlockError("support-only adaptation loss is non-finite")
            loss.backward()
            optimizer.step()
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "supervised_loss": float(supervised_loss.detach().cpu()),
                    "prototype_anchor_loss": float(
                        prototype_anchor_loss.detach().cpu()
                    ),
                    "feature_drift_loss": float(
                        feature_drift_loss.detach().cpu()
                    ),
                    "parameter_drift_loss": float(
                        parameter_drift_loss.detach().cpu()
                    ),
                }
            )
    finally:
        _freeze_for_inference(model)

    after_state = model.state_dict()
    parameter_names = {name for name, _ in model.named_parameters()}
    changed_parameters = tuple(
        name
        for name in parameter_names
        if not torch.equal(before_state[name], after_state[name])
    )
    changed_parameter_set = set(changed_parameters)
    changed_selected = tuple(
        name for name in selected_names if name in changed_parameter_set
    )
    changed_non_selected = tuple(
        sorted(changed_parameter_set - selected_name_set)
    )
    changed_buffers = tuple(
        sorted(
            name
            for name in after_state
            if name not in parameter_names
            and not torch.equal(before_state[name], after_state[name])
        )
    )
    prototypes_unchanged = torch.equal(
        prototypes_cpu, torch.as_tensor(frozen_prototypes, dtype=torch.float32)
    )
    if not changed_selected:
        raise StructuredLateBlockError(
            "support loss produced no real update in the selected structural blocks"
        )
    if changed_non_selected or changed_buffers:
        raise StructuredLateBlockError(
            "adaptation changed frozen parameters or checkpoint buffers"
        )
    if not prototypes_unchanged:
        raise StructuredLateBlockError("frozen class prototypes were modified")

    return AdaptationAudit(
        candidate_id=config.candidate_id,
        base_parameter_count=base_count,
        trainable_parameter_count=selected_count,
        trainable_parameter_fraction=fraction,
        structural_parameter_count=structural_count,
        selected_parameter_names=selected_names,
        changed_parameter_names=changed_selected,
        non_selected_changed_parameter_names=changed_non_selected,
        changed_buffer_names=changed_buffers,
        steps_completed=len(trace),
        loss_trace=tuple(trace),
        prototypes_unchanged=prototypes_unchanged,
        protocol_schema=context.protocol_schema,
        phase2_data_status=context.phase2_data_status,
        capsule_id=context.capsule_id,
        split_id=context.split_id,
    )


def predict_query_read_only(
    model: nn.Module,
    query_received_iq: Any,
    frozen_prototypes: Any,
    prototype_class_ids: Sequence[str],
    *,
    context: Phase2Context,
) -> PredictionResult:
    """Infer each query independently after adaptation has fully frozen."""

    context.validate()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise StructuredLateBlockError(
            "query inference requires a fully frozen adapted checkpoint"
        )
    rows = _validate_received_iq(query_received_iq, name="query_received_iq")
    prototypes_cpu, class_ids = _validate_prototypes(
        frozen_prototypes, prototype_class_ids
    )
    try:
        target_device = next(model.parameters()).device
    except StopIteration as exc:
        raise StructuredLateBlockError("checkpoint model has no parameters") from exc
    before_state = _copy_state(model)
    model.eval()
    _validate_checkpoint_prototype_binding(model, prototypes_cpu)
    score_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for row in rows:
            features, logits = _identity_forward(
                model, row.unsqueeze(0).to(target_device)
            )
            if (
                features.shape[1] != prototypes_cpu.shape[1]
                or logits.shape[1] != len(class_ids)
            ):
                raise StructuredLateBlockError(
                    "frozen prototype/decision dimension does not match query features"
                )
            score_rows.append(logits.squeeze(0).cpu())
    after_state = model.state_dict()
    changed_state = tuple(
        name
        for name in after_state
        if not torch.equal(before_state[name], after_state[name])
    )
    if changed_state:
        raise StructuredLateBlockError("query inference modified checkpoint state")
    scores = torch.stack(score_rows)
    predicted = tuple(class_ids[int(index)] for index in scores.argmax(dim=1))
    return PredictionResult(predicted_class_ids=predicted, scores=scores)
