"""Protocol-bound support adaptation and read-only query inference for CAPTA."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.identity_only_forward import identity_only_feature_forward

from .prototype_transport import (
    A1_SUPPORT_SHRINK,
    A2_SHARED_SHIFT,
    A3_R4_SUPPORT_SHIFT,
    CANDIDATE_IDS,
    CaptaPrototypeState,
    fit_capta_prototypes,
)
from .safe_source_target_gate import (
    PREREGISTERED_SOURCE_WEIGHTS,
    SourceTargetGateResult,
    select_source_weight,
)


class CaptaRuntimeError(ValueError):
    """Raised when the CAPTA Phase2 runtime contract fails closed."""


@dataclass(frozen=True)
class CaptaPhase2Context:
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str

    def validate(self) -> None:
        if self.protocol_schema != "p2_min_v1":
            raise CaptaRuntimeError("protocol_schema must be p2_min_v1")
        if self.phase2_data_status != "VALIDATED_ONCE":
            raise CaptaRuntimeError("phase2_data_status must be VALIDATED_ONCE")
        if not str(self.capsule_id).strip() or not str(self.split_id).strip():
            raise CaptaRuntimeError("capsule_id and split_id must bind one row")


@dataclass(frozen=True)
class CaptaConfig:
    candidate_id: str = A3_R4_SUPPORT_SHIFT
    rank: int = 4
    prior_strength: float = 3.0
    source_weight_grid: tuple[float, ...] = PREREGISTERED_SOURCE_WEIGHTS

    def validate(self) -> None:
        if self.candidate_id not in CANDIDATE_IDS:
            raise CaptaRuntimeError("candidate_id is not preregistered")
        if isinstance(self.rank, bool) or not 1 <= int(self.rank) <= 4:
            raise CaptaRuntimeError("rank must be an integer in [1,4]")
        if not math.isfinite(float(self.prior_strength)) or self.prior_strength <= 0.0:
            raise CaptaRuntimeError("prior_strength must be finite and positive")
        if tuple(float(value) for value in self.source_weight_grid) != PREREGISTERED_SOURCE_WEIGHTS:
            raise CaptaRuntimeError("source_weight_grid is not preregistered")


@dataclass(frozen=True)
class CaptaAdaptationState:
    prototype_state: CaptaPrototypeState
    prototype_class_ids: tuple[str, ...]
    source_weight: float
    decision_scale: float
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class CaptaPredictionResult:
    predicted_class_ids: tuple[str, ...]
    source_scores: torch.Tensor
    target_scores: torch.Tensor
    mixed_scores: torch.Tensor
    query_batch_state_updated: bool


def _float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    plain = value.tolist() if hasattr(value, "tolist") else value
    return torch.tensor(plain, dtype=torch.float32)


def _received_iq(value: Any, *, field: str) -> torch.Tensor:
    rows = _float_tensor(value)
    if (
        rows.ndim != 3
        or rows.shape[0] < 1
        or rows.shape[1] != 2
        or not torch.isfinite(rows).all().item()
    ):
        raise CaptaRuntimeError(f"{field} must be finite received IQ [N,2,L]")
    return rows


def _prototypes(
    value: Any, class_ids: Sequence[str]
) -> tuple[torch.Tensor, tuple[str, ...]]:
    prototypes = _float_tensor(value)
    classes = tuple(str(item) for item in class_ids)
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] < 2
        or prototypes.shape[0] != len(classes)
        or len(set(classes)) != len(classes)
        or any(not item for item in classes)
        or not torch.isfinite(prototypes).all().item()
        or torch.any(torch.linalg.vector_norm(prototypes, dim=1) <= 0.0).item()
    ):
        raise CaptaRuntimeError("frozen prototypes or class mapping are invalid")
    return prototypes.detach().clone(), classes


def _identity_forward(
    model: nn.Module, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = identity_only_feature_forward(model, rows, "z_id")
    if output is None:
        raise CaptaRuntimeError("exact ADV3B02 identity-only forward is unavailable")
    features, logits = output
    if (
        features.ndim != 2
        or logits.ndim != 2
        or len(features) != len(logits)
        or not torch.isfinite(features).all().item()
        or not torch.isfinite(logits).all().item()
    ):
        raise CaptaRuntimeError("identity features or logits are invalid")
    return features, logits


def _freeze(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _validate_binding(model: nn.Module, prototypes: torch.Tensor) -> float:
    try:
        head = model.id_backbone.cls_head.head
        weight = head.weight
        scale = float(head.s)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CaptaRuntimeError("checkpoint CosFace anchors are unavailable") from exc
    if (
        not torch.is_tensor(weight)
        or weight.shape != prototypes.shape
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise CaptaRuntimeError("checkpoint CosFace shape or scale is invalid")
    expected = F.normalize(weight.detach().float().cpu(), dim=1, eps=1.0e-4)
    observed = F.normalize(prototypes.detach().float().cpu(), dim=1, eps=1.0e-4)
    if not torch.allclose(observed, expected, rtol=1.0e-5, atol=1.0e-6):
        raise CaptaRuntimeError("prototypes are not bound to the checkpoint head")
    return scale


def _target_scores(
    features: torch.Tensor, prototypes: np.ndarray, scale: float
) -> torch.Tensor:
    target = _float_tensor(np.asarray(prototypes).copy()).to(features.device)
    return float(scale) * F.linear(
        F.normalize(features.float(), dim=1, eps=1.0e-4),
        F.normalize(target, dim=1, eps=1.0e-4),
    )


def _support_gate(
    features: np.ndarray,
    source_scores: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    config: CaptaConfig,
    scale: float,
) -> SourceTargetGateResult:
    counts = [int(np.sum(labels == index)) for index in range(len(prototypes))]
    if min(counts) < 2:
        return SourceTargetGateResult(
            source_weight=1.0,
            audit={
                "schema": "cvs.phase2.capta_p0.safe_source_target_gate.v1",
                "support_only": True,
                "query_rows_used": 0,
                "selected_source_weight": 1.0,
                "fallback": "insufficient_leave_one_out_shots",
            },
        )
    loo_target = np.empty_like(source_scores, dtype=np.float32)
    for row_index in range(len(features)):
        keep = np.arange(len(features)) != row_index
        state = fit_capta_prototypes(
            prototypes,
            features[keep],
            labels[keep],
            candidate_id=config.candidate_id,
            rank=config.rank,
            prior_strength=config.prior_strength,
        )
        feature = _float_tensor(features[row_index : row_index + 1])
        loo_target[row_index] = _target_scores(
            feature, state.target_prototypes, scale
        ).squeeze(0).numpy()
    return select_source_weight(
        source_scores,
        loo_target,
        labels,
        candidate_weights=config.source_weight_grid,
    )


def adapt_on_target_support(
    model: nn.Module,
    support_received_iq: Any,
    support_labels: Sequence[str],
    frozen_prototypes: Any,
    prototype_class_ids: Sequence[str],
    *,
    context: CaptaPhase2Context,
    config: CaptaConfig,
    device: torch.device | str = "cpu",
) -> CaptaAdaptationState:
    """Freeze a CAPTA state using target support; never invoke backward."""

    context.validate()
    config.validate()
    if not isinstance(model, nn.Module):
        raise CaptaRuntimeError("frozen checkpoint model is required")
    rows = _received_iq(support_received_iq, field="support_received_iq")
    labels_text = tuple(str(item) for item in support_labels)
    prototypes_cpu, classes = _prototypes(frozen_prototypes, prototype_class_ids)
    if len(labels_text) != len(rows) or not labels_text:
        raise CaptaRuntimeError("support label alignment is invalid")
    lookup = {label: index for index, label in enumerate(classes)}
    if any(label not in lookup for label in labels_text):
        raise CaptaRuntimeError("support label is absent from frozen mapping")
    labels = np.asarray([lookup[label] for label in labels_text], dtype=np.int64)

    target_device = torch.device(device)
    model.to(target_device)
    _freeze(model)
    before = _model_state(model)
    scale = _validate_binding(model, prototypes_cpu)
    with torch.no_grad():
        features_tensor, source_logits = _identity_forward(model, rows.to(target_device))
    features = F.normalize(features_tensor.float(), dim=1, eps=1.0e-4).cpu().numpy()
    prototype_state = fit_capta_prototypes(
        prototypes_cpu.numpy(),
        features,
        labels,
        candidate_id=config.candidate_id,
        rank=config.rank,
        prior_strength=config.prior_strength,
    )
    gate = _support_gate(
        features,
        source_logits.detach().cpu().numpy(),
        labels,
        prototypes_cpu.numpy(),
        config,
        scale,
    )
    changed = tuple(
        name
        for name, value in model.state_dict().items()
        if not torch.equal(before[name], value)
    )
    if changed or any(parameter.requires_grad for parameter in model.parameters()):
        raise CaptaRuntimeError("support adaptation changed or unfroze checkpoint state")
    audit = dict(prototype_state.audit)
    audit.update(
        {
            "schema": "cvs.phase2.capta_p0.adaptation_state.v1",
            "protocol_schema": context.protocol_schema,
            "phase2_data_status": context.phase2_data_status,
            "capsule_id": context.capsule_id,
            "split_id": context.split_id,
            "support_input_count": len(rows),
            "query_rows_used_for_fit": 0,
            "source_input_count": 0,
            "model_state_changed": False,
            "selected_source_weight": gate.source_weight,
            "gate": gate.audit,
        }
    )
    return CaptaAdaptationState(
        prototype_state=prototype_state,
        prototype_class_ids=classes,
        source_weight=float(gate.source_weight),
        decision_scale=scale,
        protocol_schema=context.protocol_schema,
        phase2_data_status=context.phase2_data_status,
        capsule_id=context.capsule_id,
        split_id=context.split_id,
        audit=audit,
    )


def predict_query_read_only(
    model: nn.Module,
    query_received_iq: Any,
    state: CaptaAdaptationState,
    *,
    context: CaptaPhase2Context,
) -> CaptaPredictionResult:
    """Predict one row at a time without modifying model or CAPTA state."""

    context.validate()
    if (
        context.protocol_schema != state.protocol_schema
        or context.phase2_data_status != state.phase2_data_status
        or context.capsule_id != state.capsule_id
        or context.split_id != state.split_id
    ):
        raise CaptaRuntimeError("query context is not bound to adapted support row")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise CaptaRuntimeError("query inference requires a fully frozen checkpoint")
    rows = _received_iq(query_received_iq, field="query_received_iq")
    before_model = _model_state(model)
    before_target = state.prototype_state.target_prototypes.copy()
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise CaptaRuntimeError("checkpoint model has no parameters") from exc
    _validate_binding(
        model,
        _float_tensor(state.prototype_state.source_prototypes.copy()),
    )
    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for row in rows:
            features, source_logits = _identity_forward(model, row.unsqueeze(0).to(device))
            source_rows.append(source_logits.squeeze(0).cpu())
            target_rows.append(
                _target_scores(
                    features,
                    state.prototype_state.target_prototypes,
                    state.decision_scale,
                ).squeeze(0).cpu()
            )
    source_scores = torch.stack(source_rows)
    target_scores = torch.stack(target_rows)
    mixed_scores = (
        float(state.source_weight) * source_scores
        + (1.0 - float(state.source_weight)) * target_scores
    )
    changed_model = tuple(
        name
        for name, value in model.state_dict().items()
        if not torch.equal(before_model[name], value)
    )
    state_changed = not np.array_equal(
        state.prototype_state.target_prototypes, before_target
    )
    if changed_model or state_changed:
        raise CaptaRuntimeError("query inference modified model or CAPTA state")
    predicted = tuple(
        state.prototype_class_ids[int(index)]
        for index in mixed_scores.argmax(dim=1).tolist()
    )
    return CaptaPredictionResult(
        predicted_class_ids=predicted,
        source_scores=source_scores,
        target_scores=target_scores,
        mixed_scores=mixed_scores,
        query_batch_state_updated=False,
    )


__all__ = [
    "A1_SUPPORT_SHRINK",
    "A2_SHARED_SHIFT",
    "A3_R4_SUPPORT_SHIFT",
    "CaptaAdaptationState",
    "CaptaConfig",
    "CaptaPhase2Context",
    "CaptaPredictionResult",
    "CaptaRuntimeError",
    "adapt_on_target_support",
    "predict_query_read_only",
]
