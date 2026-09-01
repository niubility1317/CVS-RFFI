"""Source-only Phase1 meta training for the MARC-OT weight-delta bank."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .marc_ot_support_features import build_marc_ot_support_features
from .meta_bank_inner_loop import BankFastState, first_order_bank_adapt
from .meta_support_set_encoder import SupportDomainState, SupportSetEncoder
from .meta_trainer import (
    MetaEpisodeBatch,
    _SOURCE_ROLES,
    _validate_episode_batch_integrity,
    _validate_episode_roles,
)
from .meta_weight_bank import (
    WEIGHT_DELTA_BANK_SCHEMA,
    WeightDeltaBank,
    compose_weight_delta,
    parameter_block_key,
)


class MetaBankTrainerError(RuntimeError):
    """Raised when a Phase1 bank meta-step violates its gradient contract."""


@dataclass(frozen=True)
class MetaBankTrainerConfig:
    """Finite source-only objective and inner-loop settings."""

    source_receiver_ids: tuple[int, ...]
    inner_steps: int = 3
    receiver_cvar_fraction: float = 0.5
    receiver_cvar_weight: float = 1.0
    worst_class_guard_weight: float = 1.0

    def __post_init__(self) -> None:
        receiver_ids = tuple(self.source_receiver_ids)
        if (
            not receiver_ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in receiver_ids)
            or len(set(receiver_ids)) != len(receiver_ids)
        ):
            raise ValueError("source_receiver_ids must contain unique integer IDs")
        if (
            isinstance(self.inner_steps, bool)
            or not isinstance(self.inner_steps, int)
            or not 1 <= self.inner_steps <= 10
        ):
            raise ValueError("inner_steps must be an integer in [1, 10]")
        fraction = float(self.receiver_cvar_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("receiver_cvar_fraction must be in (0, 1]")
        for name in ("receiver_cvar_weight", "worst_class_guard_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "source_receiver_ids", receiver_ids)
        object.__setattr__(self, "receiver_cvar_fraction", fraction)


@dataclass(frozen=True)
class MetaBankStepResult:
    """Detached outer metrics plus the differentiable final fast state."""

    loss: Tensor
    query_adapt_mean: Tensor
    receiver_cvar: Tensor
    worst_class_guard: Tensor
    fast_state: BankFastState


def _finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _extract_logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        candidates = [output[key] for key in ("logits", "tx_logits") if key in output]
        if not candidates:
            raise MetaBankTrainerError("functional model output is missing logits")
        logits = candidates[0]
        if any(not isinstance(value, Tensor) or not torch.equal(value, logits) for value in candidates[1:]):
            raise MetaBankTrainerError("functional model logit aliases conflict")
    else:
        raise MetaBankTrainerError("functional model output must be logits or a mapping")
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 2
        or not logits.is_floating_point()
        or not _finite(logits)
    ):
        raise MetaBankTrainerError("functional logits must be finite floating [batch, classes]")
    return logits


def _validate_bank_and_compose_initial(
    base_state: Mapping[str, Tensor],
    base_checkpoint_id: str,
    bank: WeightDeltaBank,
    support_state: SupportDomainState,
) -> tuple[OrderedDict[str, Tensor], dict[str, Tensor]]:
    if not isinstance(base_checkpoint_id, str) or not base_checkpoint_id:
        raise ValueError("base_checkpoint_id must be a non-empty string")
    if not isinstance(bank, WeightDeltaBank) or bank.schema != WEIGHT_DELTA_BANK_SCHEMA:
        raise ValueError("weight delta bank schema mismatch")
    if bank.base_checkpoint_id != base_checkpoint_id:
        raise ValueError("base checkpoint identity does not match the weight bank")
    if not bank.entries:
        raise ValueError("weight delta bank must contain at least one block")
    expected_q = sum(entry.effective_rank for entry in bank.entries)
    if (
        support_state.q.shape != (expected_q,)
        or support_state.block_gates.shape != (len(bank.entries),)
        or support_state.block_lrs.shape != (len(bank.entries),)
        or support_state.uncertainty.numel() != 1
    ):
        raise ValueError("support encoder output geometry does not match the bank")
    controls = (
        support_state.q,
        support_state.uncertainty,
        support_state.block_gates,
        support_state.block_lrs,
    )
    if any(not _finite(value) for value in controls):
        raise ValueError("support encoder output contains non-finite values")

    initial: OrderedDict[str, Tensor] = OrderedDict()
    block_lrs: dict[str, Tensor] = {}
    q_offset = 0
    seen_blocks: set[str] = set()
    for block_index, entry in enumerate(bank.entries):
        spec = entry.spec
        if spec.name in seen_blocks:
            raise ValueError("weight delta bank contains duplicate block names")
        seen_blocks.add(spec.name)
        if not isinstance(entry.basis, Tensor) or not entry.basis.requires_grad:
            raise ValueError(f"trainable bank basis for block {spec.name!r} must require gradients")
        q_next = q_offset + entry.effective_rank
        composed = compose_weight_delta(entry, support_state.q[q_offset:q_next])
        q_offset = q_next
        scale = (1.0 - support_state.uncertainty.reshape(())) * support_state.block_gates[block_index]
        if not _finite(scale):
            raise ValueError("support-conditioned bank scale is non-finite")
        for name, delta in composed.items():
            if name in initial:
                raise ValueError("weight delta bank repeats a parameter")
            if parameter_block_key(name) != spec.name:
                raise ValueError("weight delta bank parameter/block routing mismatch")
            base_value = base_state.get(name)
            if not isinstance(base_value, Tensor) or not base_value.is_floating_point():
                raise ValueError(f"base state is missing floating bank parameter {name!r}")
            if tuple(base_value.shape) != tuple(delta.shape) or base_value.dtype != delta.dtype:
                raise ValueError(f"base/bank geometry mismatch for {name!r}")
            if not _finite(base_value):
                raise ValueError(f"base parameter {name!r} is non-finite")
            combined = base_value + scale.to(base_value) * delta.to(base_value)
            if not _finite(combined):
                raise ValueError(f"initial bank parameter {name!r} is non-finite")
            initial[name] = combined
        block_lrs[spec.name] = support_state.block_lrs[block_index]
    return initial, block_lrs


def _outer_components(
    logits: Tensor,
    batch: MetaEpisodeBatch,
    receiver_cvar_fraction: float,
) -> tuple[Tensor, Tensor, Tensor]:
    labels = batch.query_y.to(device=logits.device, dtype=torch.long)
    if logits.shape[0] != labels.numel():
        raise MetaBankTrainerError("query logits and labels have different lengths")
    row_losses = F.cross_entropy(logits, labels, reduction="none")
    adapt_mask = batch.adapt_mask.to(device=logits.device)
    guard_mask = batch.guard_mask.to(device=logits.device)
    if not bool(adapt_mask.any()) or not bool(guard_mask.any()):
        raise MetaBankTrainerError("outer objective requires query_adapt and query_guard rows")
    query_adapt_mean = row_losses[adapt_mask].mean()

    query_rows = batch.episode.query_adapt + batch.episode.query_guard
    receiver_losses: dict[int, list[Tensor]] = defaultdict(list)
    class_losses: dict[int, list[Tensor]] = defaultdict(list)
    for index, row in enumerate(query_rows):
        if bool(batch.adapt_mask[index]):
            receiver_losses[int(row.rx_i)].append(row_losses[index])
        if bool(batch.guard_mask[index]):
            class_losses[int(row.tx_i)].append(row_losses[index])
    receiver_means = torch.stack(
        [torch.stack(receiver_losses[key]).mean() for key in sorted(receiver_losses)]
    )
    tail_count = max(1, math.ceil(receiver_cvar_fraction * receiver_means.numel()))
    receiver_cvar = torch.topk(receiver_means, k=tail_count, largest=True).values.mean()
    guard_class_means = torch.stack(
        [torch.stack(class_losses[key]).mean() for key in sorted(class_losses)]
    )
    worst_class_guard = guard_class_means.max()
    return query_adapt_mean, receiver_cvar, worst_class_guard


def _required_outer_parameters(
    support_encoder: SupportSetEncoder, bank: WeightDeltaBank
) -> tuple[tuple[str, Tensor], ...]:
    required: list[tuple[str, Tensor]] = []
    for name, parameter in support_encoder.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"required support encoder parameter {name!r} is frozen")
        required.append((f"support_encoder.{name}", parameter))
    for entry in bank.entries:
        if not isinstance(entry.basis, Tensor) or not entry.basis.requires_grad:
            raise ValueError(
                f"required bank basis {entry.spec.name!r} must require gradients"
            )
        required.append((f"bank_basis.{entry.spec.name}", entry.basis))
    identities = [id(parameter) for _, parameter in required]
    if len(identities) != len(set(identities)):
        raise ValueError("required outer parameters contain duplicate tensor identities")
    return tuple(required)


def _clear_required_gradients(required: tuple[tuple[str, Tensor], ...]) -> None:
    for _, parameter in required:
        parameter.grad = None


def _validate_optimizer_scope(
    optimizer: torch.optim.Optimizer,
    required: tuple[tuple[str, Tensor], ...],
) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    actual = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    actual_ids = tuple(id(parameter) for parameter in actual)
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("optimizer contains duplicate parameter identities")
    expected_by_id = {id(parameter): name for name, parameter in required}
    actual_id_set = set(actual_ids)
    expected_id_set = set(expected_by_id)
    if actual_id_set != expected_id_set:
        missing = tuple(
            expected_by_id[identity] for identity in expected_id_set - actual_id_set
        )
        extra = len(actual_id_set - expected_id_set)
        raise ValueError(
            f"optimizer parameter scope mismatch: missing={sorted(missing)!r}, extra_count={extra}"
        )


def _require_outer_gradients(
    support_encoder: SupportSetEncoder,
    support_state: SupportDomainState,
    bank: WeightDeltaBank,
) -> None:
    for name, value in (
        ("coefficient query", support_state.q),
        ("uncertainty", support_state.uncertainty),
        ("block gate", support_state.block_gates),
        ("block learning rate", support_state.block_lrs),
    ):
        if value.grad is None or not _finite(value.grad) or not bool(torch.count_nonzero(value.grad)):
            raise MetaBankTrainerError(f"{name} has no finite nonzero outer gradient")
    for name, parameter in support_encoder.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None
            or not _finite(parameter.grad)
            or not bool(torch.count_nonzero(parameter.grad))
        ):
            raise MetaBankTrainerError(f"support encoder parameter {name!r} has no finite nonzero outer gradient")
    for entry in bank.entries:
        gradient = entry.basis.grad
        if gradient is None or not _finite(gradient) or not bool(torch.count_nonzero(gradient)):
            raise MetaBankTrainerError(
                f"trainable bank basis {entry.spec.name!r} has no finite nonzero outer gradient"
            )


def run_meta_bank_step(
    functional_forward: Callable[[Mapping[str, Tensor], Tensor], object],
    *,
    base_state: Mapping[str, Tensor],
    base_checkpoint_id: str,
    bank: WeightDeltaBank,
    support_encoder: SupportSetEncoder,
    support_feature_model: nn.Module,
    batch: MetaEpisodeBatch,
    config: MetaBankTrainerConfig,
    optimizer: torch.optim.Optimizer,
) -> MetaBankStepResult:
    """Backpropagate one source-only bank meta episode.

    Only ``(support_x, support_y)`` is handed to the inner-loop closure.  Query
    tensors are opened exactly once after the fast state is frozen for the
    outer objective.
    """

    if not callable(functional_forward):
        raise TypeError("functional_forward must be callable")
    if not isinstance(config, MetaBankTrainerConfig):
        raise TypeError("config must be a MetaBankTrainerConfig")
    if not isinstance(support_encoder, SupportSetEncoder):
        raise TypeError("support_encoder must be a SupportSetEncoder")
    if not isinstance(support_feature_model, nn.Module):
        raise TypeError("support_feature_model must be a torch.nn.Module")
    _validate_episode_batch_integrity(batch)
    _validate_episode_roles(batch, _SOURCE_ROLES, config.source_receiver_ids)

    required_parameters = _required_outer_parameters(support_encoder, bank)
    _clear_required_gradients(required_parameters)
    _validate_optimizer_scope(optimizer, required_parameters)
    support_labels = batch.support_y.to(device=batch.support_x.device, dtype=torch.long)
    physical_tokens = tuple(row.physical_sample_id for row in batch.episode.support)
    support_batch = build_marc_ot_support_features(
        support_feature_model,
        batch.support_x,
        support_labels,
        physical_tokens,
        nominal_k=int(batch.episode.k_shot),
        effective_mask=torch.ones(
            len(support_labels), device=batch.support_x.device, dtype=batch.support_x.dtype
        ),
    )
    support_state = support_encoder(
        support_batch.rows,
        support_batch.labels,
        support_batch.physical_tokens,
        support_batch.effective_mask,
    )
    for value in (
        support_state.q,
        support_state.uncertainty,
        support_state.block_gates,
        support_state.block_lrs,
    ):
        value.retain_grad()
    initial, block_lrs = _validate_bank_and_compose_initial(
        base_state, base_checkpoint_id, bank, support_state
    )

    def support_loss(fast: Mapping[str, Tensor], support: object) -> Tensor:
        support_x, support_y = support
        logits = _extract_logits(functional_forward(fast, support_x))
        return F.cross_entropy(logits, support_y.to(device=logits.device, dtype=torch.long))

    fast_state = first_order_bank_adapt(
        support_loss,
        initial,
        block_lrs,
        (batch.support_x, batch.support_y),
        config.inner_steps,
    )
    query_logits = _extract_logits(functional_forward(fast_state.parameters, batch.query_x))
    query_adapt_mean, receiver_cvar, worst_class_guard = _outer_components(
        query_logits, batch, config.receiver_cvar_fraction
    )
    total_loss = (
        query_adapt_mean
        + float(config.receiver_cvar_weight) * receiver_cvar
        + float(config.worst_class_guard_weight) * worst_class_guard
    )
    if not total_loss.requires_grad or not _finite(total_loss):
        raise MetaBankTrainerError("meta-bank outer loss must be finite and differentiable")
    total_loss.backward()
    _require_outer_gradients(support_encoder, support_state, bank)
    optimizer.step()
    return MetaBankStepResult(
        loss=total_loss.detach(),
        query_adapt_mean=query_adapt_mean.detach(),
        receiver_cvar=receiver_cvar.detach(),
        worst_class_guard=worst_class_guard.detach(),
        fast_state=fast_state,
    )


__all__ = [
    "MetaBankStepResult",
    "MetaBankTrainerConfig",
    "MetaBankTrainerError",
    "run_meta_bank_step",
]
