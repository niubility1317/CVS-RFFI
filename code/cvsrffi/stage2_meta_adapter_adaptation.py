"""Phase2 support-only adaptation for the V1 meta adapter.

This module is intentionally a narrow runtime boundary.  It validates the
four immutable Phase2 handles, maps support labels to the frozen prototype
rows, performs the configured adapter-only first-order updates, and freezes
the resulting model before returning.  Prediction is a separate no-gradient
operation and never writes model state.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .meta_adapter import adapter_parameter_budget, iter_inner_adapter_parameters
from .meta_inner_loop import first_order_adapt


_PHASE2_CONTEXT_ALLOWLIST = frozenset(
    {"protocol_schema", "phase2_data_status", "capsule_id", "split_id"}
)
_MAX_PHASE2_STEPS = 5
_HARD_PHASE2_STEP_LIMIT = 40
_PROTOTYPE_EPS = 1.0e-8


@dataclass(frozen=True)
class MetaAdapterPhase2Config:
    """Runtime limits for support-only V1 adaptation.

    ``steps=3`` is the formal V1 setting.  Diagnostic callers may select
    ``0..5``; the deployment wrapper never permits a larger value, even
    though the project-wide hard protection remains 40 steps.
    """

    steps: int = 3
    max_steps: int = 5
    hard_step_limit: int = 40

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise ValueError("Phase2 adaptation steps must be an integer in [0, 5]")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("Phase2 max_steps must be an integer in [1, 5]")
        if isinstance(self.hard_step_limit, bool) or not isinstance(self.hard_step_limit, int):
            raise ValueError("Phase2 hard_step_limit must be an integer in [1, 40]")
        if self.max_steps < 1 or self.max_steps > _MAX_PHASE2_STEPS:
            raise ValueError("Phase2 max_steps must be in [1, 5]")
        if self.hard_step_limit < self.max_steps or self.hard_step_limit > _HARD_PHASE2_STEP_LIMIT:
            raise ValueError("Phase2 hard_step_limit must contain max_steps and be <= 40")
        if self.steps < 0 or self.steps > self.max_steps or self.steps > _MAX_PHASE2_STEPS:
            raise ValueError("Phase2 adaptation steps must be in [0, 5]")
        if self.steps > self.hard_step_limit:
            raise ValueError("Phase2 adaptation steps exceed hard_step_limit")


@dataclass(frozen=True)
class MetaAdapterAdaptAudit:
    """Immutable evidence for one support-only adaptation call."""

    steps: int
    backward_count: int
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
        """Compatibility alias for callers that name the history explicitly."""

        return self.support_losses

    @property
    def optimizer_created(self) -> bool:
        """The Phase2 core has no optimizer by construction."""

        return False

    @property
    def query_consumed(self) -> bool:
        """Adaptation accepts support tensors only."""

        return False


def _validate_context(context: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(context, Mapping):
        raise ValueError(
            "Phase2 context allowlist requires exactly protocol_schema, "
            "phase2_data_status, capsule_id, split_id"
        )
    keys = set(context.keys())
    if keys != set(_PHASE2_CONTEXT_ALLOWLIST) or any(not isinstance(key, str) for key in keys):
        missing = sorted(_PHASE2_CONTEXT_ALLOWLIST.difference(keys))
        unexpected = sorted(keys.difference(_PHASE2_CONTEXT_ALLOWLIST))
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


def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    for buffer in model.buffers():
        if buffer is not None and buffer.is_floating_point():
            return buffer.device, buffer.dtype
    raise ValueError("model must expose at least one floating-point parameter or buffer")


def _validate_support_iq(support_iq: Tensor) -> None:
    if not torch.is_tensor(support_iq):
        raise TypeError("support_iq must be a tensor")
    if support_iq.ndim < 2 or support_iq.size(0) <= 0:
        raise ValueError("support_iq must have a non-empty batch dimension")
    if not support_iq.is_floating_point():
        raise ValueError("support_iq must be floating-point received IQ")
    if not bool(torch.isfinite(support_iq).all()):
        raise ValueError("support_iq must contain only finite values")


def _normalize_class_ids(class_ids: Sequence[int] | Tensor, *, expected: int, device: torch.device) -> Tensor:
    if torch.is_tensor(class_ids):
        if class_ids.ndim != 1:
            raise ValueError("class_ids must be one-dimensional")
        if class_ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("class_ids must use an integer dtype")
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


def _validate_labels(support_labels: Tensor, *, batch_size: int, class_ids: Tensor) -> Tensor:
    if not torch.is_tensor(support_labels):
        raise TypeError("support_labels must be a tensor")
    if support_labels.ndim != 1 or support_labels.numel() != batch_size:
        raise ValueError("support_labels must be one-dimensional and match support_iq")
    if support_labels.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("support_labels must use an integer dtype")
    labels = support_labels.detach().to(device=class_ids.device, dtype=torch.long)
    matches = labels[:, None] == class_ids[None, :]
    if not bool(matches.any(dim=1).all()):
        raise ValueError("support_labels contain an identifier outside class_ids")
    return matches.to(dtype=torch.long).argmax(dim=1)


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
        raise ValueError("model.forward exposes ambiguous support-label arguments")
    if labels:
        kwargs[labels[0]] = None
    return kwargs


def _extract_embedding(outputs: Any, *, batch_size: int) -> Tensor:
    if torch.is_tensor(outputs):
        embedding = outputs
    elif isinstance(outputs, Mapping):
        # The first two entries are the explicit Task5 canonical choices for
        # single and dual ADV3B02 outputs.  The remaining names are only
        # narrow mapping carriers used by small test models.
        preferred = (
            "feat_cls",
            "z_id",
            "id_feat_joint",
            "embedding",
            "features",
            "feature",
            "feat_joint",
            "base",
        )
        selected: Tensor | None = None
        selected_key: str | None = None
        z_id_key = outputs.get("z_id_key")
        if "z_id" in outputs and isinstance(z_id_key, str):
            alias = f"id_{z_id_key}"
            if alias in outputs:
                preferred = (alias,) + tuple(key for key in preferred if key != alias)
        for key in preferred:
            if key in outputs:
                value = outputs[key]
                if not torch.is_tensor(value):
                    raise ValueError(f"model output embedding key {key!r} must be a tensor")
                selected = value
                selected_key = key
                break
        if selected is None:
            raise ValueError(
                "model output must provide an embedding via feat_cls, z_id or a supported feature key"
            )
        embedding = selected
        del selected_key
    else:
        raise ValueError("model(return_aux=True) must return a tensor or mapping")
    if embedding.ndim != 2 or embedding.size(0) != batch_size:
        raise ValueError("model embedding must have shape [support_batch, dimension]")
    if not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all()):
        raise ValueError("model embedding must be a finite floating-point tensor")
    return embedding


def _cosine_logits(embedding: Tensor, prototypes: Tensor) -> Tensor:
    if embedding.size(1) != prototypes.size(1):
        raise ValueError(
            "model embedding dimension must match frozen_prototypes dimension"
        )
    anchors = prototypes.detach().to(device=embedding.device, dtype=embedding.dtype)
    if not bool(torch.isfinite(anchors).all()):
        raise ValueError("frozen_prototypes must remain finite")
    return F.normalize(embedding, dim=1, eps=_PROTOTYPE_EPS) @ F.normalize(
        anchors, dim=1, eps=_PROTOTYPE_EPS
    ).transpose(0, 1)


def _model_parameter_snapshot(model: nn.Module) -> dict[str, Tensor]:
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
        raise RuntimeError(
            "Phase2 support adaptation changed non-adapter state: "
            f"{sorted(changed)!r}"
        )


def _freeze_after_adaptation(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def adapt_meta_adapter_on_support(
    model: nn.Module,
    support_iq: Tensor,
    support_labels: Tensor,
    frozen_prototypes: Tensor,
    class_ids: Sequence[int] | Tensor,
    context: Mapping[str, Any],
    config: MetaAdapterPhase2Config,
) -> MetaAdapterAdaptAudit:
    """Apply fixed-step adapter updates from legal support IQ only."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(config, MetaAdapterPhase2Config):
        raise TypeError("config must be a MetaAdapterPhase2Config")
    context_values = _validate_context(context)
    _validate_support_iq(support_iq)
    device, dtype = _model_device_dtype(model)
    inner_items = tuple(iter_inner_adapter_parameters(model))
    if not inner_items:
        raise ValueError("Phase2 adaptation requires enabled meta_adapter parameters")
    budget = adapter_parameter_budget(model)
    trainable_fraction = float(budget["inner_ratio"])
    if not math.isfinite(trainable_fraction) or trainable_fraction > 0.01:
        raise ValueError(
            "Phase2 adapter trainable parameter fraction exceeds 1%: "
            f"{trainable_fraction:.8f}"
        )
    inner_names = {name for name, _ in inner_items}
    step_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if name.endswith("log_step_size")
    )
    prototypes = _validate_prototypes(frozen_prototypes, class_count=len(class_ids))
    normalized_class_ids = _normalize_class_ids(
        class_ids,
        expected=prototypes.size(0),
        device=device,
    )
    mapped_labels = _validate_labels(
        support_labels,
        batch_size=support_iq.size(0),
        class_ids=normalized_class_ids,
    )
    support_x = support_iq.detach().to(device=device, dtype=dtype)
    support_y = support_labels.detach().to(device=device, dtype=torch.long)
    before = _model_parameter_snapshot(model)
    step_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.endswith("log_step_size")
    }
    requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    training_before = tuple((module, module.training) for module in model.modules())

    # Task4 bundles may arrive with every parameter frozen.  Enable only the
    # Task3 inner leaves for the temporary functional gradient calculation.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in inner_items:
        parameter.requires_grad_(True)

    def support_loss(outputs: Mapping[str, Any], _labels: Tensor, _fast: Mapping[str, Tensor]) -> Tensor:
        embedding = _extract_embedding(outputs, batch_size=support_x.size(0))
        logits = _cosine_logits(embedding, prototypes)
        loss = F.cross_entropy(logits, mapped_labels)
        # Task6 intentionally rejects missing gradients.  A dual ADV3B02
        # model exposes both identity and domain adapter sites, while the
        # fixed identity prototypes consume only the identity branch.  Touch
        # every fast leaf with an exact zero so that the shared first-order
        # contract remains explicit; unreachable domain leaves therefore stay
        # bitwise frozen rather than being silently dropped or updated.
        zero_touch = loss.new_zeros(())
        for value in _fast.values():
            zero_touch = zero_touch + value.sum() * 0.0
        return loss + zero_touch

    try:
        fast_state = first_order_adapt(
            model,
            support_x,
            support_y,
            support_loss,
            steps=config.steps,
        )
        if config.steps:
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
            for name, old_value in before.items():
                current = model.state_dict().get(name)
                if current is not None:
                    current.copy_(old_value)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad_before[name])
        for module, was_training in training_before:
            module.training = was_training
        raise

    _freeze_after_adaptation(model)
    before_adapter = {name: before[name] for name in inner_names}
    after_state = model.state_dict()
    updated_names = tuple(
        name
        for name, _ in inner_items
        if fast_state.steps and not torch.equal(after_state[name], before_adapter[name])
    )
    return MetaAdapterAdaptAudit(
        steps=fast_state.steps,
        backward_count=fast_state.steps,
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


@torch.no_grad()
def predict_with_frozen_meta_adapter(
    model: nn.Module,
    query_iq: Tensor,
    frozen_prototypes: Tensor,
    class_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Return one fixed-prototype class-ID prediction per query IQ row."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not torch.is_tensor(query_iq):
        raise TypeError("query_iq must be a tensor")
    if query_iq.ndim < 2 or query_iq.size(0) <= 0:
        raise ValueError("query_iq must have a non-empty batch dimension")
    if not query_iq.is_floating_point() or not bool(torch.isfinite(query_iq).all()):
        raise ValueError("query_iq must be finite floating-point received IQ")
    device, dtype = _model_device_dtype(model)
    prototypes = _validate_prototypes(frozen_prototypes, class_count=len(class_ids))
    normalized_class_ids = _normalize_class_ids(
        class_ids,
        expected=prototypes.size(0),
        device=device,
    )
    query_x = query_iq.detach().to(device=device, dtype=dtype)
    state_before = _model_parameter_snapshot(model)
    training_before = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        outputs = model(query_x, **_forward_kwargs(model))
        embedding = _extract_embedding(outputs, batch_size=query_x.size(0))
        logits = _cosine_logits(embedding, prototypes)
        indices = logits.argmax(dim=1)
        return normalized_class_ids[indices].detach().clone()
    finally:
        # A compliant model should be read-only in eval mode, but restoring
        # the complete state also protects the query boundary from custom
        # counters or other mutable buffers in wrappers.
        with torch.no_grad():
            current_state = model.state_dict()
            for name, old_value in state_before.items():
                current = current_state.get(name)
                if current is not None:
                    current.copy_(old_value)
        for module, was_training in training_before:
            module.training = was_training


__all__ = [
    "MetaAdapterAdaptAudit",
    "MetaAdapterPhase2Config",
    "adapt_meta_adapter_on_support",
    "predict_with_frozen_meta_adapter",
]
