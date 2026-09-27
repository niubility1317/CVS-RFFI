"""Support-only structured late-block adaptation for frozen ADV3B02.

Phase2 adaptation has an intentionally small input surface: fixed target
received IQ, its legal support labels, immutable class prototypes and mapping,
the frozen checkpoint model, and a preregistered configuration.  Query rows
are accepted only by the separate read-only prediction function after every
model parameter has been frozen again.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_ADAPTATION_STEPS = 40
TARGET_MIN_TRAINABLE_FRACTION = 0.05
TARGET_MAX_TRAINABLE_FRACTION = 0.15
HARD_MAX_TRAINABLE_FRACTION = 0.20

_PHASE2_CONTEXT_ALLOWLIST = frozenset(
    {
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
    }
)
_CANDIDATE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "freq_f3_proj": ("id_backbone.f3.", "id_backbone.f_proj."),
    "time_t3": ("id_backbone.t3.",),
}
_CLASSIFIER_TOKENS = (
    ".cls_head.",
    ".classifier.",
    ".classification_head.",
    ".tx_head.",
    ".dom_head.",
    ".domain_head.",
)


class StructuredLateBlockError(ValueError):
    """Raised when the structured late-block Phase2 contract is violated."""


@dataclass(frozen=True)
class StructuredLateBlockConfig:
    """Frozen, preregistered configuration for one support adaptation row."""

    candidate: str = "freq_f3_proj"
    steps: int = 20
    learning_rate: float = 5.0e-4
    prototype_anchor_weight: float = 0.25
    feature_drift_weight: float = 0.05
    parameter_drift_weight: float = 1.0e-4
    gradient_clip: float = 1.0
    logit_scale: float = 8.0
    max_trainable_fraction: float = TARGET_MAX_TRAINABLE_FRACTION


@dataclass(frozen=True)
class StructuredLateBlockAudit:
    method_id: str
    candidate: str
    gradient_updates: int
    support_samples: int
    support_class_count: int
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    trainable_parameter_names: tuple[str, ...]
    structural_trainable_parameters: int
    classifier_parameters_changed: int
    prototypes_changed: bool
    loss_trace: tuple[Mapping[str, float], ...]


def _canonical_parameter_name(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def _is_classifier_parameter(name: str) -> bool:
    lowered = f".{_canonical_parameter_name(name).lower()}"
    return any(token in lowered for token in _CLASSIFIER_TOKENS)


def _validate_context(context: Mapping[str, Any]) -> None:
    if not isinstance(context, Mapping):
        raise StructuredLateBlockError("Phase2 context must be an allowlist mapping")
    if any(not isinstance(key, str) for key in context):
        raise StructuredLateBlockError(
            "Phase2 context allowlist accepts string keys only"
        )
    actual = frozenset(context)
    if actual != _PHASE2_CONTEXT_ALLOWLIST:
        missing = sorted(_PHASE2_CONTEXT_ALLOWLIST - actual)
        extra = sorted(actual - _PHASE2_CONTEXT_ALLOWLIST)
        raise StructuredLateBlockError(
            "Phase2 context allowlist mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if str(context["protocol_schema"]) != "p2_min_v1":
        raise StructuredLateBlockError("protocol_schema must be p2_min_v1")
    if str(context["phase2_data_status"]) != "VALIDATED_ONCE":
        raise StructuredLateBlockError(
            "phase2_data_status must be VALIDATED_ONCE"
        )
    for key in ("capsule_id", "split_id"):
        if not str(context[key]).strip():
            raise StructuredLateBlockError(f"{key} must be nonempty")


def _validate_config(config: StructuredLateBlockConfig) -> None:
    if str(config.candidate) not in _CANDIDATE_PREFIXES:
        raise StructuredLateBlockError(
            f"unknown structured late-block candidate: {config.candidate!r}"
        )
    steps = int(config.steps)
    if steps < 1 or steps > MAX_ADAPTATION_STEPS:
        raise StructuredLateBlockError(
            f"adaptation steps must be in [1, {MAX_ADAPTATION_STEPS}]"
        )
    positive = {
        "learning_rate": config.learning_rate,
        "gradient_clip": config.gradient_clip,
        "logit_scale": config.logit_scale,
    }
    for name, raw_value in positive.items():
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise StructuredLateBlockError(f"{name} must be finite and positive")
    nonnegative = {
        "prototype_anchor_weight": config.prototype_anchor_weight,
        "feature_drift_weight": config.feature_drift_weight,
        "parameter_drift_weight": config.parameter_drift_weight,
    }
    for name, raw_value in nonnegative.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise StructuredLateBlockError(f"{name} must be finite and nonnegative")
    configured_cap = float(config.max_trainable_fraction)
    if (
        not math.isfinite(configured_cap)
        or configured_cap < TARGET_MIN_TRAINABLE_FRACTION
        or configured_cap > HARD_MAX_TRAINABLE_FRACTION
    ):
        raise StructuredLateBlockError(
            "max_trainable_fraction must be within the 5%-20% fraction bounds"
        )


def _validate_received_iq(rows: torch.Tensor, *, label: str) -> None:
    if not torch.is_tensor(rows) or rows.ndim != 3:
        raise StructuredLateBlockError(f"{label} must be a [N,2,L] tensor")
    if rows.shape[0] < 1 or rows.shape[1] != 2 or rows.shape[2] < 1:
        raise StructuredLateBlockError(f"{label} must contain nonempty [N,2,L] rows")
    if not torch.isfinite(rows).all():
        raise StructuredLateBlockError(f"{label} contains non-finite values")


def _validate_integer_vector(value: torch.Tensor, *, label: str) -> None:
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if (
        not torch.is_tensor(value)
        or value.ndim != 1
        or value.numel() < 1
        or value.dtype not in integer_dtypes
    ):
        raise StructuredLateBlockError(f"{label} must be a nonempty integer vector")


def _validate_prototypes(
    frozen_prototypes: torch.Tensor,
    prototype_class_ids: torch.Tensor,
) -> None:
    if not torch.is_tensor(frozen_prototypes) or frozen_prototypes.ndim != 2:
        raise StructuredLateBlockError(
            "frozen_prototypes must be a [classes,features] tensor"
        )
    if frozen_prototypes.shape[0] < 1 or frozen_prototypes.shape[1] < 1:
        raise StructuredLateBlockError("frozen_prototypes must be nonempty")
    if frozen_prototypes.requires_grad:
        raise StructuredLateBlockError("frozen prototypes must be immutable")
    if not torch.isfinite(frozen_prototypes).all():
        raise StructuredLateBlockError("frozen_prototypes contain non-finite values")
    _validate_integer_vector(prototype_class_ids, label="prototype_class_ids")
    if prototype_class_ids.shape[0] != frozen_prototypes.shape[0]:
        raise StructuredLateBlockError(
            "prototype_class_ids and frozen_prototypes must align"
        )
    ids = prototype_class_ids.detach().cpu().tolist()
    if len(set(int(value) for value in ids)) != len(ids):
        raise StructuredLateBlockError("prototype_class_ids must be unique")


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise StructuredLateBlockError("checkpoint model has no parameters") from exc


def _identity_features(model: nn.Module, rows: torch.Tensor) -> torch.Tensor:
    identity_only = None
    if hasattr(model, "id_backbone") and callable(getattr(model, "_pick_z_id", None)):
        from cvsrffi.identity_only_forward import identity_only_feature_forward

        identity_only = identity_only_feature_forward(model, rows, "z_id")
    if identity_only is not None:
        features, _frozen_logits = identity_only
        return features.float()
    output = model(rows)
    if not isinstance(output, Mapping):
        raise StructuredLateBlockError("model output must be a mapping")
    features = output.get("z_id")
    if not torch.is_tensor(features):
        raise StructuredLateBlockError("model output must contain tensor z_id")
    if features.ndim != 2 or features.shape[0] != rows.shape[0]:
        raise StructuredLateBlockError("z_id must be a row-aligned feature matrix")
    return features.float()


def _is_identity_feature_parameter(name: str) -> bool:
    canonical = _canonical_parameter_name(name).lower()
    return canonical.startswith("id_backbone.") and not _is_classifier_parameter(name)


def _candidate_allows(candidate: str, name: str) -> bool:
    canonical = _canonical_parameter_name(name).lower()
    if _is_classifier_parameter(name) or "gate" in canonical:
        return False
    return any(
        canonical.startswith(prefix)
        for prefix in _CANDIDATE_PREFIXES[str(candidate)]
    )


def _is_structural_parameter(name: str) -> bool:
    canonical = _canonical_parameter_name(name).lower()
    return (
        canonical.endswith(".weight")
        and ".norm." not in canonical
        and "gate" not in canonical
    )


def _select_parameters(
    model: nn.Module,
    config: StructuredLateBlockConfig,
) -> tuple[list[tuple[str, nn.Parameter]], int, float, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _candidate_allows(str(config.candidate), name)
    ]
    if not selected:
        raise StructuredLateBlockError(
            f"candidate {config.candidate!r} selected no encoder parameters"
        )
    canonical_names = tuple(_canonical_parameter_name(name) for name, _ in selected)
    for prefix in _CANDIDATE_PREFIXES[str(config.candidate)]:
        if not any(name.lower().startswith(prefix) for name in canonical_names):
            raise StructuredLateBlockError(
                f"candidate {config.candidate!r} is missing complete block {prefix}"
            )
    total = int(
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if _is_identity_feature_parameter(name)
        )
    )
    if total < 1:
        raise StructuredLateBlockError("identity feature base has no parameters")
    trainable = int(sum(parameter.numel() for _, parameter in selected))
    fraction = float(trainable / total)
    if (
        fraction < TARGET_MIN_TRAINABLE_FRACTION
        or fraction > float(config.max_trainable_fraction)
        or fraction > HARD_MAX_TRAINABLE_FRACTION
    ):
        raise StructuredLateBlockError(
            "selected trainable fraction is outside the preregistered bounds: "
            f"fraction={fraction:.6f}, target=5%-15%, hard_cap=20%"
        )
    structural = int(
        sum(
            parameter.numel()
            for name, parameter in selected
            if _is_structural_parameter(name)
        )
    )
    if structural < 1:
        raise StructuredLateBlockError(
            "candidate cannot be Norm/Bias/Gate-only; structural weights are required"
        )
    for _, parameter in selected:
        parameter.requires_grad_(True)
    return selected, total, fraction, structural


def _prototype_targets(
    support_labels: torch.Tensor,
    prototype_class_ids: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    mapping = {
        int(class_id): index
        for index, class_id in enumerate(prototype_class_ids.detach().cpu().tolist())
    }
    target_indices: list[int] = []
    for raw_label in support_labels.detach().cpu().tolist():
        label = int(raw_label)
        if label not in mapping:
            raise StructuredLateBlockError(
                f"support label {label} has no frozen prototype mapping"
            )
        target_indices.append(mapping[label])
    return torch.tensor(target_indices, dtype=torch.long, device=device)


def _prototype_scores(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    if features.ndim != 2 or prototypes.ndim != 2:
        raise StructuredLateBlockError("features and prototypes must be matrices")
    if features.shape[1] != prototypes.shape[1]:
        raise StructuredLateBlockError(
            "frozen prototype width does not match checkpoint feature width"
        )
    return float(logit_scale) * (
        F.normalize(features.float(), dim=1)
        @ F.normalize(prototypes.float(), dim=1).transpose(0, 1)
    )


def _restore_parameters(
    model: nn.Module,
    parameter_before: Mapping[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(parameter_before[name])


def _freeze_for_query(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def adapt_on_target_support_with_frozen_prototypes(
    model: nn.Module,
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    frozen_prototypes: torch.Tensor,
    prototype_class_ids: torch.Tensor,
    context: Mapping[str, Any],
    config: StructuredLateBlockConfig = StructuredLateBlockConfig(),
) -> StructuredLateBlockAudit:
    """Adapt one continuous late encoder block from legal target support only.

    The signature deliberately has no query or source input.  Classifier heads
    and frozen prototypes never enter the optimizer and all model parameters
    are frozen again before this function returns.
    """

    _validate_context(context)
    _validate_config(config)
    _validate_received_iq(support_iq, label="support_iq")
    _validate_integer_vector(support_labels, label="support_labels")
    if support_labels.shape[0] != support_iq.shape[0]:
        raise StructuredLateBlockError("support IQ and labels must align")
    _validate_prototypes(frozen_prototypes, prototype_class_ids)

    device = _model_device(model)
    rows = support_iq.detach().to(device=device, dtype=torch.float32)
    labels = support_labels.detach().to(device=device, dtype=torch.long)
    prototypes_input_before = frozen_prototypes.detach().clone()
    prototypes = frozen_prototypes.detach().clone().to(
        device=device, dtype=torch.float32
    )
    target_indices = _prototype_targets(
        labels, prototype_class_ids, device=device
    )

    _freeze_for_query(model)
    with torch.enable_grad():
        reference_features = _identity_features(model, rows).detach()
    if reference_features.shape[1] != prototypes.shape[1]:
        raise StructuredLateBlockError(
            "frozen prototype width does not match checkpoint feature width"
        )

    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    classifier_names = tuple(
        name for name, _ in model.named_parameters() if _is_classifier_parameter(name)
    )
    selected, total_parameters, trainable_fraction, structural_parameters = (
        _select_parameters(model, config)
    )
    selected_initial = {
        name: parameter.detach().clone() for name, parameter in selected
    }
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in selected],
        lr=float(config.learning_rate),
    )
    trace: list[Mapping[str, float]] = []
    completed_updates = 0
    try:
        for step in range(int(config.steps)):
            optimizer.zero_grad(set_to_none=True)
            features = _identity_features(model, rows)
            scores = _prototype_scores(
                features, prototypes, logit_scale=float(config.logit_scale)
            )
            support_loss = F.cross_entropy(scores, target_indices)
            matched_prototypes = prototypes.index_select(0, target_indices)
            prototype_anchor = (
                1.0
                - F.cosine_similarity(
                    features.float(), matched_prototypes.float(), dim=1
                )
            ).mean()
            feature_drift = (
                1.0
                - F.cosine_similarity(
                    features.float(), reference_features.float(), dim=1
                )
            ).mean()
            parameter_drift = torch.stack(
                [
                    (parameter - selected_initial[name]).float().square().mean()
                    for name, parameter in selected
                ]
            ).mean()
            loss = (
                support_loss
                + float(config.prototype_anchor_weight) * prototype_anchor
                + float(config.feature_drift_weight) * feature_drift
                + float(config.parameter_drift_weight) * parameter_drift
            )
            if not torch.isfinite(loss):
                raise StructuredLateBlockError("non-finite support adaptation loss")
            loss.backward()
            gradients = [
                parameter.grad
                for _, parameter in selected
                if parameter.grad is not None
            ]
            if len(gradients) != len(selected):
                raise StructuredLateBlockError(
                    "selected late block is not fully connected to z_id"
                )
            if any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise StructuredLateBlockError(
                    "support adaptation produced non-finite gradients"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in selected],
                max_norm=float(config.gradient_clip),
            )
            optimizer.step()
            completed_updates += 1
            trace.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "support_supervised_loss": float(
                        support_loss.detach().cpu().item()
                    ),
                    "prototype_anchor_loss": float(
                        prototype_anchor.detach().cpu().item()
                    ),
                    "feature_drift_loss": float(feature_drift.detach().cpu().item()),
                    "parameter_drift_loss": float(
                        parameter_drift.detach().cpu().item()
                    ),
                    "gradient_norm": float(
                        torch.as_tensor(gradient_norm).detach().cpu().item()
                    ),
                }
            )
    except Exception:
        _restore_parameters(model, parameter_before)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        _freeze_for_query(model)

    selected_names = {name for name, _ in selected}
    unexpected_changes = [
        name
        for name, parameter in model.named_parameters()
        if name not in selected_names
        and not torch.equal(parameter.detach(), parameter_before[name])
    ]
    if unexpected_changes:
        _restore_parameters(model, parameter_before)
        _freeze_for_query(model)
        raise StructuredLateBlockError(
            f"non-allowlisted parameters changed: {unexpected_changes}"
        )
    named_parameters = dict(model.named_parameters())
    classifier_changes = sum(
        not torch.equal(named_parameters[name].detach(), parameter_before[name])
        for name in classifier_names
    )
    prototypes_changed = not torch.equal(
        frozen_prototypes.detach(), prototypes_input_before
    )
    if classifier_changes or prototypes_changed:
        _restore_parameters(model, parameter_before)
        _freeze_for_query(model)
        raise StructuredLateBlockError(
            "frozen classifier or immutable prototypes changed during adaptation"
        )

    return StructuredLateBlockAudit(
        method_id="SCLBA_V1",
        candidate=str(config.candidate),
        gradient_updates=completed_updates,
        support_samples=int(rows.shape[0]),
        support_class_count=int(torch.unique(labels).numel()),
        trainable_parameters=int(sum(parameter.numel() for _, parameter in selected)),
        total_parameters=total_parameters,
        trainable_fraction=trainable_fraction,
        trainable_parameter_names=tuple(name for name, _ in selected),
        structural_trainable_parameters=structural_parameters,
        classifier_parameters_changed=int(classifier_changes),
        prototypes_changed=bool(prototypes_changed),
        loss_trace=tuple(trace),
    )


def predict_query_with_frozen_prototypes(
    model: nn.Module,
    received_iq: torch.Tensor,
    *,
    frozen_prototypes: torch.Tensor,
    prototype_class_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frozen-prototype prediction independently for every query row."""

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise StructuredLateBlockError(
            "query inference requires a fully frozen adaptation state"
        )
    _validate_received_iq(received_iq, label="received_iq")
    _validate_prototypes(frozen_prototypes, prototype_class_ids)
    device = _model_device(model)
    prototypes = frozen_prototypes.detach().clone().to(
        device=device, dtype=torch.float32
    )
    class_ids = prototype_class_ids.detach().clone().to(
        device=device, dtype=torch.long
    )
    rows = received_iq.detach().to(device=device, dtype=torch.float32)

    per_row_scores: list[torch.Tensor] = []
    # A one-row call prevents batch composition, class quota, or global query
    # counts from influencing another query's decision.
    with torch.enable_grad():
        for row in rows.split(1, dim=0):
            features = _identity_features(model, row)
            per_row_scores.append(
                _prototype_scores(features, prototypes, logit_scale=8.0).detach()
            )
    scores = torch.cat(per_row_scores, dim=0)
    predictions = class_ids.index_select(0, scores.argmax(dim=1))
    return predictions, scores


__all__ = [
    "HARD_MAX_TRAINABLE_FRACTION",
    "MAX_ADAPTATION_STEPS",
    "TARGET_MAX_TRAINABLE_FRACTION",
    "TARGET_MIN_TRAINABLE_FRACTION",
    "StructuredLateBlockAudit",
    "StructuredLateBlockConfig",
    "StructuredLateBlockError",
    "adapt_on_target_support_with_frozen_prototypes",
    "predict_query_with_frozen_prototypes",
]
