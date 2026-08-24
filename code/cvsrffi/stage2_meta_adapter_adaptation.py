"""Phase2 support-only adaptation for the V1 meta adapter.

The public boundary is deliberately typed.  A validated received-IQ support
carrier is adapted with the formal three-step plan (or through the explicitly
named diagnostic API), and the result is returned as a frozen handle.  Query
prediction accepts only that handle and runs on a disposable deep copy.
"""

from __future__ import annotations

import copy
import inspect
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .meta_adapter import adapter_parameter_budget, iter_inner_adapter_parameters
from .meta_inner_loop import FastAdapterState, first_order_adapt


_PHASE2_CONTEXT_ALLOWLIST = frozenset(
    {"protocol_schema", "phase2_data_status", "capsule_id", "split_id"}
)
_FORMAL_PHASE2_STEPS = 3
_MAX_PHASE2_STEPS = 5
_HARD_PHASE2_STEP_LIMIT = 40
_PROTOTYPE_EPS = 1.0e-8
_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


class Phase2SupportProvenance(str, Enum):
    """The sole support carrier provenance accepted by this core."""

    VALIDATED_RECEIVED_IQ = "p2_min_v1_received_iq"


# Explicit aliases make the typed boundary discoverable without introducing
# multiple accepted provenance values or alternate runtime schemas.
ReceivedIQProvenance = Phase2SupportProvenance
SupportCarrierProvenance = Phase2SupportProvenance


def _validate_context(context: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(context, Mapping):
        raise ValueError(
            "Phase2 context allowlist requires exactly protocol_schema, "
            "phase2_data_status, capsule_id, split_id"
        )
    keys = set(context.keys())
    if any(not isinstance(key, str) for key in keys) or keys != set(_PHASE2_CONTEXT_ALLOWLIST):
        missing = sorted(
            (key for key in _PHASE2_CONTEXT_ALLOWLIST if key not in keys),
            key=str,
        )
        unexpected = sorted(
            (repr(key) for key in keys if key not in _PHASE2_CONTEXT_ALLOWLIST),
            key=str,
        )
        raise ValueError(
            "Phase2 context allowlist mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    values: dict[str, str] = {}
    for key in _PHASE2_CONTEXT_ALLOWLIST:
        value = context[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Phase2 context allowlist value {key!r} must be a non-empty string")
        values[key] = value
    if values["protocol_schema"] != "p2_min_v1":
        raise ValueError("Phase2 context protocol_schema must be p2_min_v1")
    if values["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("Phase2 context phase2_data_status must be VALIDATED_ONCE")
    return values


def _validate_received_iq(value: Tensor, *, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim < 2 or value.size(0) <= 0:
        raise ValueError(f"{name} must have a non-empty batch dimension")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating-point received IQ")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_labels(value: Tensor, *, batch_size: int, name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1 or value.numel() != batch_size:
        raise ValueError(f"{name} must be one-dimensional and match received_iq")
    if value.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must use an integer dtype")


def _validate_physical_ids(value: Sequence[str], *, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("support_physical_ids must be a sequence of non-empty strings")
    ids = tuple(value)
    if len(ids) != batch_size:
        raise ValueError("support_physical_ids count must match received_iq batch size")
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("support_physical_ids must contain non-empty physical IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("support_physical_ids must be non-empty and unique")
    return ids


def _validate_receiver_id(value: object) -> int | str:
    if isinstance(value, bool):
        raise ValueError("receiver_id must be a non-empty integer or string")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("receiver_id must be a non-negative integer")
        return int(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("receiver_id must be a non-empty integer or string")


@dataclass(frozen=True)
class ValidatedTargetSupportBatch:
    """Immutable, received-IQ-only support carrier for Phase2 adaptation."""

    received_iq: Tensor
    labels: Tensor
    support_physical_ids: tuple[str, ...]
    receiver_id: int | str
    context: Mapping[str, str]
    provenance: Phase2SupportProvenance = Phase2SupportProvenance.VALIDATED_RECEIVED_IQ

    def __post_init__(self) -> None:
        _validate_received_iq(self.received_iq, name="received_iq")
        _validate_labels(self.labels, batch_size=self.received_iq.size(0), name="labels")
        ids = _validate_physical_ids(
            self.support_physical_ids,
            batch_size=self.received_iq.size(0),
        )
        receiver_id = _validate_receiver_id(self.receiver_id)
        context = _validate_context(self.context)
        if self.provenance is not Phase2SupportProvenance.VALIDATED_RECEIVED_IQ:
            raise ValueError(
                "provenance must be Phase2SupportProvenance.VALIDATED_RECEIVED_IQ"
            )
        # Clone the carrier tensors so later caller mutation cannot alter the
        # fixed support record consumed by the adaptation core.
        object.__setattr__(self, "received_iq", self.received_iq.detach().clone())
        object.__setattr__(self, "labels", self.labels.detach().clone())
        object.__setattr__(self, "support_physical_ids", ids)
        object.__setattr__(self, "receiver_id", receiver_id)
        object.__setattr__(self, "context", MappingProxyType(context))


@dataclass(frozen=True)
class MetaAdapterPhase2Config:
    """Formal V1 configuration; the step count is intentionally not a field."""

    expected_capsule_id: str
    expected_split_id: str
    hard_step_limit: int = _HARD_PHASE2_STEP_LIMIT

    def __post_init__(self) -> None:
        for name in ("expected_capsule_id", "expected_split_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty frozen handle")
        if isinstance(self.hard_step_limit, bool) or not isinstance(self.hard_step_limit, int):
            raise ValueError("hard_step_limit must be an integer in [3, 40]")
        if self.hard_step_limit < _FORMAL_PHASE2_STEPS or self.hard_step_limit > _HARD_PHASE2_STEP_LIMIT:
            raise ValueError("hard_step_limit must be in [3, 40]")


@dataclass(frozen=True)
class MetaAdapterPhase2DiagnosticConfig:
    """Explicit diagnostic-only step configuration bounded to five steps."""

    steps: int
    expected_capsule_id: str
    expected_split_id: str
    hard_step_limit: int = _HARD_PHASE2_STEP_LIMIT

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise ValueError("diagnostic steps must be an integer in [0, 5]")
        if self.steps < 0 or self.steps > _MAX_PHASE2_STEPS:
            raise ValueError("diagnostic steps must be in [0, 5]")
        for name in ("expected_capsule_id", "expected_split_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty frozen handle")
        if isinstance(self.hard_step_limit, bool) or not isinstance(self.hard_step_limit, int):
            raise ValueError("hard_step_limit must be an integer in [1, 40]")
        if self.hard_step_limit < max(1, self.steps) or self.hard_step_limit > _HARD_PHASE2_STEP_LIMIT:
            raise ValueError("hard_step_limit must contain diagnostic steps and be <= 40")


# Short alias for callers that prefer the unambiguous diagnostic name.
MetaAdapterDiagnosticConfig = MetaAdapterPhase2DiagnosticConfig


@dataclass(frozen=True)
class MetaAdapterAdaptAudit:
    """Immutable evidence for one formal or diagnostic support adaptation."""

    steps: int
    diagnostic: bool
    support_loss_evaluations: int
    gradient_updates: int
    updated_parameter_names: tuple[str, ...]
    trainable_names: tuple[str, ...]
    trainable_count: int
    total_parameters: int
    trainable_fraction: float
    support_losses: tuple[float, ...]
    protocol_schema: str
    phase2_data_status: str
    capsule_id: str
    split_id: str
    model_eval: bool
    log_step_size_names: tuple[str, ...]

    @property
    def support_loss_history(self) -> tuple[float, ...]:
        return self.support_losses

    @property
    def optimizer_created(self) -> bool:
        return False

    @property
    def query_consumed(self) -> bool:
        return False


@dataclass(frozen=True)
class AdaptedMetaAdapterState:
    """Frozen model handle carrying the post-support fast state and audit."""

    model: nn.Module
    fast_state: FastAdapterState
    context: Mapping[str, str]
    audit: MetaAdapterAdaptAudit

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("AdaptedMetaAdapterState.model must be a torch.nn.Module")
        if not isinstance(self.fast_state, FastAdapterState):
            raise TypeError("AdaptedMetaAdapterState.fast_state must be a FastAdapterState")
        if not isinstance(self.audit, MetaAdapterAdaptAudit):
            raise TypeError("AdaptedMetaAdapterState.audit must be a MetaAdapterAdaptAudit")
        context = _validate_context(self.context)
        if self.audit.diagnostic and not self.audit.steps <= _MAX_PHASE2_STEPS:
            raise ValueError("diagnostic handle exceeds the Phase2 step bound")
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("AdaptedMetaAdapterState.model must be eval and fully frozen")
        object.__setattr__(self, "context", MappingProxyType(context))


# The brief names this returned object as either state or handle; both names
# intentionally refer to one immutable type, not two accepted runtime types.
AdaptedMetaAdapterHandle = AdaptedMetaAdapterState


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    for buffer in model.buffers():
        if buffer is not None and buffer.is_floating_point():
            return buffer.device, buffer.dtype
    raise ValueError("model must expose at least one floating-point parameter or buffer")


def _class_ids_length(class_ids: Sequence[int] | Tensor) -> int:
    if torch.is_tensor(class_ids):
        if class_ids.ndim != 1:
            raise ValueError("class_ids must be one-dimensional")
        return int(class_ids.numel())
    if isinstance(class_ids, Sequence) and not isinstance(class_ids, (str, bytes)):
        return len(class_ids)
    raise TypeError("class_ids must be an integer tensor or sequence")


def _normalize_class_ids(
    class_ids: Sequence[int] | Tensor,
    *,
    expected: int,
    device: torch.device,
) -> Tensor:
    if torch.is_tensor(class_ids):
        if class_ids.ndim != 1 or class_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("class_ids must be a one-dimensional integer tensor")
        values = class_ids.detach().to(device=device, dtype=torch.long)
    elif isinstance(class_ids, Sequence) and not isinstance(class_ids, (str, bytes)):
        parsed: list[int] = []
        for value in class_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("class_ids must contain integer identifiers")
            parsed.append(int(value))
        values = torch.tensor(parsed, dtype=torch.long, device=device)
    else:
        raise TypeError("class_ids must be an integer tensor or sequence")
    if values.numel() != expected:
        raise ValueError("class_ids length must match frozen_prototypes classes")
    if values.numel() == 0 or torch.unique(values).numel() != values.numel():
        raise ValueError("class_ids must be non-empty and unique")
    return values


def _validate_prototypes(frozen_prototypes: Tensor, *, class_count: int) -> Tensor:
    if not torch.is_tensor(frozen_prototypes):
        raise TypeError("frozen_prototypes must be a tensor")
    if frozen_prototypes.ndim != 2 or frozen_prototypes.size(0) != class_count:
        raise ValueError("frozen_prototypes must have shape [len(class_ids), dimension]")
    if frozen_prototypes.size(1) <= 0 or not frozen_prototypes.is_floating_point():
        raise ValueError("frozen_prototypes must be a non-empty floating-point matrix")
    if not bool(torch.isfinite(frozen_prototypes).all()):
        raise ValueError("frozen_prototypes must contain only finite values")
    return frozen_prototypes.detach()


def _map_labels_to_prototypes(labels: Tensor, class_ids: Tensor) -> Tensor:
    labels = labels.detach().to(device=class_ids.device, dtype=torch.long)
    matches = labels[:, None] == class_ids[None, :]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("support labels contain an identifier outside class_ids")
    return matches.to(dtype=torch.long).argmax(dim=1)


def _forward_kwargs(model: nn.Module) -> dict[str, object]:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot inspect model.forward signature") from exc
    kwargs: dict[str, object] = {}
    if "return_aux" in parameters:
        kwargs["return_aux"] = True
    labels = [name for name in ("y", "y_tx") if name in parameters]
    if len(labels) > 1:
        raise ValueError("model.forward exposes ambiguous label arguments")
    if labels:
        kwargs[labels[0]] = None
    return kwargs


def _extract_embedding(outputs: Any, *, batch_size: int) -> Tensor:
    if torch.is_tensor(outputs):
        embedding = outputs
    elif isinstance(outputs, Mapping):
        if "z_id" in outputs:
            preferred = ["z_id"]
            z_id_key = outputs.get("z_id_key")
            if isinstance(z_id_key, str) and f"id_{z_id_key}" in outputs:
                preferred.insert(0, f"id_{z_id_key}")
        else:
            preferred = ["feat_cls"]
        preferred.extend(
            key
            for key in ("id_feat_joint", "embedding", "features", "feature", "feat_joint", "base")
            if key not in preferred
        )
        embedding = None
        for key in preferred:
            if key in outputs:
                value = outputs[key]
                if not torch.is_tensor(value):
                    raise ValueError(f"model output embedding key {key!r} must be a tensor")
                embedding = value
                break
        if embedding is None:
            raise ValueError("model output must provide feat_cls, z_id or a supported embedding key")
    else:
        raise ValueError("model(return_aux=True) must return a tensor or mapping")
    if embedding.ndim != 2 or embedding.size(0) != batch_size:
        raise ValueError("model embedding must have shape [batch, dimension]")
    if not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all()):
        raise ValueError("model embedding must be a finite floating-point tensor")
    return embedding


def _cosine_logits(embedding: Tensor, prototypes: Tensor) -> Tensor:
    if embedding.size(1) != prototypes.size(1):
        raise ValueError("model embedding dimension must match frozen_prototypes dimension")
    anchors = prototypes.detach().to(device=embedding.device, dtype=embedding.dtype)
    return F.normalize(embedding, dim=1, eps=_PROTOTYPE_EPS) @ F.normalize(
        anchors, dim=1, eps=_PROTOTYPE_EPS
    ).transpose(0, 1)


def _state_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_non_adapter_state_unchanged(
    model: nn.Module,
    before: Mapping[str, Tensor],
    inner_names: set[str],
) -> None:
    after = model.state_dict()
    changed = []
    for name, old_value in before.items():
        if name in inner_names:
            continue
        current = after.get(name)
        if current is None or not torch.equal(current, old_value):
            changed.append(name)
    if changed:
        raise RuntimeError(f"Phase2 support adaptation changed non-adapter state: {sorted(changed)!r}")


def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _validate_config_context(
    context: Mapping[str, str],
    *,
    expected_capsule_id: str,
    expected_split_id: str,
) -> dict[str, str]:
    values = _validate_context(context)
    if values["capsule_id"] != expected_capsule_id:
        raise ValueError(
            f"capsule_id mismatch: expected {expected_capsule_id!r}, got {values['capsule_id']!r}"
        )
    if values["split_id"] != expected_split_id:
        raise ValueError(
            f"split_id mismatch: expected {expected_split_id!r}, got {values['split_id']!r}"
        )
    return values


def _adapt_impl(
    model: nn.Module,
    support_batch: ValidatedTargetSupportBatch,
    frozen_prototypes: Tensor,
    class_ids: Sequence[int] | Tensor,
    *,
    steps: int,
    expected_capsule_id: str,
    expected_split_id: str,
    hard_step_limit: int,
    diagnostic: bool,
) -> AdaptedMetaAdapterState:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(support_batch, ValidatedTargetSupportBatch):
        raise TypeError("support_batch must be a ValidatedTargetSupportBatch")
    if steps < 0 or steps > _MAX_PHASE2_STEPS or steps > hard_step_limit:
        raise ValueError("Phase2 adaptation steps must be in [0, 5] and within hard_step_limit")
    context_values = _validate_config_context(
        support_batch.context,
        expected_capsule_id=expected_capsule_id,
        expected_split_id=expected_split_id,
    )
    if support_batch.provenance is not Phase2SupportProvenance.VALIDATED_RECEIVED_IQ:
        raise ValueError("support carrier provenance is not validated received IQ")
    support_iq = support_batch.received_iq
    support_labels = support_batch.labels
    _validate_received_iq(support_iq, name="support_batch.received_iq")
    _validate_labels(support_labels, batch_size=support_iq.size(0), name="support_batch.labels")
    device, dtype = _model_device_dtype(model)
    inner_items = tuple(iter_inner_adapter_parameters(model))
    if not inner_items:
        raise ValueError("Phase2 adaptation requires enabled meta_adapter parameters")
    budget = adapter_parameter_budget(model)
    trainable_fraction = float(budget["inner_ratio"])
    if not math.isfinite(trainable_fraction) or trainable_fraction > 0.01:
        raise ValueError(
            f"Phase2 adapter trainable parameter fraction exceeds 1%: {trainable_fraction:.8f}"
        )
    inner_names = {name for name, _ in inner_items}
    step_names = tuple(name for name, _ in model.named_parameters() if name.endswith("log_step_size"))
    class_count = _class_ids_length(class_ids)
    prototypes = _validate_prototypes(frozen_prototypes, class_count=class_count)
    normalized_class_ids = _normalize_class_ids(
        class_ids,
        expected=prototypes.size(0),
        device=device,
    )
    mapped_labels = _map_labels_to_prototypes(support_labels, normalized_class_ids)
    support_x = support_iq.detach().to(device=device, dtype=dtype)
    support_y = support_labels.detach().to(device=device, dtype=torch.long)
    before = _state_snapshot(model)
    step_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.endswith("log_step_size")
    }
    requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    training_before = tuple((module, module.training) for module in model.modules())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in inner_items:
        parameter.requires_grad_(True)

    support_loss_evaluations = 0

    def support_loss(outputs: Mapping[str, Any], _labels: Tensor, fast: Mapping[str, Tensor]) -> Tensor:
        nonlocal support_loss_evaluations
        support_loss_evaluations += 1
        embedding = _extract_embedding(outputs, batch_size=support_x.size(0))
        logits = _cosine_logits(embedding, prototypes)
        loss = F.cross_entropy(logits, mapped_labels)
        # Task6 requires every fast leaf to be present in the gradient call.
        # An unused dual domain branch is touched by an exact zero and remains
        # bitwise unchanged instead of being silently dropped.
        zero_touch = loss.new_zeros(())
        for value in fast.values():
            zero_touch = zero_touch + value.sum() * 0.0
        return loss + zero_touch

    try:
        fast_state = first_order_adapt(
            model,
            support_x,
            support_y,
            support_loss,
            steps=steps,
        )
        if support_loss_evaluations != steps:
            raise RuntimeError(
                "support loss evaluation count did not match the fixed adaptation plan"
            )
        if steps:
            named_parameters = dict(model.named_parameters())
            for name in inner_names:
                if name not in named_parameters:
                    raise RuntimeError(f"inner adapter parameter disappeared: {name!r}")
                with torch.no_grad():
                    named_parameters[name].copy_(fast_state.parameters[name].detach())
        _assert_non_adapter_state_unchanged(model, before, inner_names)
        for name, old_value in step_before.items():
            current = dict(model.named_parameters())[name].detach()
            if not torch.equal(current, old_value):
                raise RuntimeError(f"Phase2 changed frozen module step size {name!r}")
    except Exception:
        with torch.no_grad():
            current_state = model.state_dict()
            for name, old_value in before.items():
                current = current_state.get(name)
                if current is not None:
                    current.copy_(old_value)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad_before[name])
        for module, was_training in training_before:
            module.training = was_training
        raise

    _freeze_model(model)
    before_adapter = {name: before[name] for name in inner_names}
    after_state = model.state_dict()
    updated_names = tuple(
        name
        for name, _ in inner_items
        if steps and not torch.equal(after_state[name], before_adapter[name])
    )
    audit = MetaAdapterAdaptAudit(
        steps=steps,
        diagnostic=diagnostic,
        support_loss_evaluations=support_loss_evaluations,
        gradient_updates=steps,
        updated_parameter_names=updated_names,
        trainable_names=tuple(name for name, _ in inner_items),
        trainable_count=int(budget["inner_parameters"]),
        total_parameters=int(budget["total_parameters"]),
        trainable_fraction=trainable_fraction,
        support_losses=tuple(float(value) for value in fast_state.support_losses),
        protocol_schema=context_values["protocol_schema"],
        phase2_data_status=context_values["phase2_data_status"],
        capsule_id=context_values["capsule_id"],
        split_id=context_values["split_id"],
        model_eval=not model.training,
        log_step_size_names=step_names,
    )
    frozen_fast_state = FastAdapterState(
        OrderedDict(
            (name, value.detach().clone())
            for name, value in fast_state.parameters.items()
        ),
        fast_state.steps,
        fast_state.support_losses,
    )
    return AdaptedMetaAdapterState(
        model=model,
        fast_state=frozen_fast_state,
        context=context_values,
        audit=audit,
    )


def adapt_meta_adapter_on_support(
    model: nn.Module,
    support_batch: ValidatedTargetSupportBatch,
    frozen_prototypes: Tensor,
    class_ids: Sequence[int] | Tensor,
    config: MetaAdapterPhase2Config,
) -> AdaptedMetaAdapterState:
    """Formal V1 adaptation: exactly three support-only gradient updates."""

    if not isinstance(config, MetaAdapterPhase2Config):
        raise TypeError("formal Phase2 API requires MetaAdapterPhase2Config; diagnostic config is separate")
    return _adapt_impl(
        model,
        support_batch,
        frozen_prototypes,
        class_ids,
        steps=_FORMAL_PHASE2_STEPS,
        expected_capsule_id=config.expected_capsule_id,
        expected_split_id=config.expected_split_id,
        hard_step_limit=config.hard_step_limit,
        diagnostic=False,
    )


def adapt_meta_adapter_diagnostic_on_support(
    model: nn.Module,
    support_batch: ValidatedTargetSupportBatch,
    frozen_prototypes: Tensor,
    class_ids: Sequence[int] | Tensor,
    config: MetaAdapterPhase2DiagnosticConfig,
) -> AdaptedMetaAdapterState:
    """Diagnostic-only support adaptation bounded to zero through five steps."""

    if not isinstance(config, MetaAdapterPhase2DiagnosticConfig):
        raise TypeError("diagnostic API requires MetaAdapterPhase2DiagnosticConfig")
    return _adapt_impl(
        model,
        support_batch,
        frozen_prototypes,
        class_ids,
        steps=config.steps,
        expected_capsule_id=config.expected_capsule_id,
        expected_split_id=config.expected_split_id,
        hard_step_limit=config.hard_step_limit,
        diagnostic=True,
    )


@torch.no_grad()
def predict_with_frozen_meta_adapter(
    handle: AdaptedMetaAdapterHandle,
    query_iq: Tensor,
    frozen_prototypes: Tensor,
    class_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Predict through a formal frozen handle on a disposable model copy."""

    if not isinstance(handle, AdaptedMetaAdapterHandle):
        raise TypeError("predict_with_frozen_meta_adapter requires an AdaptedMetaAdapterHandle")
    if handle.audit.diagnostic:
        raise ValueError("diagnostic handles are not eligible for the formal query prediction API")
    if handle.model.training or any(parameter.requires_grad for parameter in handle.model.parameters()):
        raise ValueError("formal adapted handle must remain eval and fully frozen")
    if not torch.is_tensor(query_iq):
        raise TypeError("query_iq must be a tensor")
    _validate_received_iq(query_iq, name="query_iq")
    query_model = copy.deepcopy(handle.model)
    _freeze_model(query_model)
    device, dtype = _model_device_dtype(query_model)
    class_count = _class_ids_length(class_ids)
    prototypes = _validate_prototypes(frozen_prototypes, class_count=class_count)
    normalized_class_ids = _normalize_class_ids(
        class_ids,
        expected=prototypes.size(0),
        device=device,
    )
    query_x = query_iq.detach().to(device=device, dtype=dtype)
    outputs = query_model(query_x, **_forward_kwargs(query_model))
    embedding = _extract_embedding(outputs, batch_size=query_x.size(0))
    logits = _cosine_logits(embedding, prototypes)
    return normalized_class_ids[logits.argmax(dim=1)].detach().clone()


__all__ = [
    "AdaptedMetaAdapterHandle",
    "AdaptedMetaAdapterState",
    "MetaAdapterAdaptAudit",
    "MetaAdapterDiagnosticConfig",
    "MetaAdapterPhase2Config",
    "MetaAdapterPhase2DiagnosticConfig",
    "Phase2SupportProvenance",
    "ReceivedIQProvenance",
    "SupportCarrierProvenance",
    "ValidatedTargetSupportBatch",
    "adapt_meta_adapter_diagnostic_on_support",
    "adapt_meta_adapter_on_support",
    "predict_with_frozen_meta_adapter",
]
