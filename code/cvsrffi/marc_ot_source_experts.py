"""Source-only MARC-OT domain-expert construction over the shared weight bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .meta_weight_bank import (
    DeltaBankEntry,
    DeltaTaskKey,
    WeightDeltaBank,
    extract_block_delta,
    fit_weight_delta_bank,
    parameter_block_key,
)


@dataclass(frozen=True)
class MARCOTSourceExpertConfig:
    trainable_prefixes: tuple[str, ...]
    base_checkpoint_id: str
    steps: int
    lr: float
    max_rank: int


@dataclass
class SourceExpertBankResult:
    bank: WeightDeltaBank
    task_losses: dict[DeltaTaskKey, float]
    updated_parameter_names: tuple[str, ...]
    task_deltas: dict[DeltaTaskKey, dict[str, Tensor]]
    adapted_states: dict[DeltaTaskKey, dict[str, Tensor]]


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _resolve_updated_parameters(
    model: nn.Module, config: MARCOTSourceExpertConfig
) -> list[tuple[str, nn.Parameter]]:
    if not config.trainable_prefixes or any(not isinstance(prefix, str) or not prefix for prefix in config.trainable_prefixes):
        raise ValueError("at least one explicit allowlisted training prefix is required")
    if len(set(config.trainable_prefixes)) != len(config.trainable_prefixes):
        raise ValueError("allowlisted training prefixes must not contain duplicates")
    if not isinstance(config.base_checkpoint_id, str) or not config.base_checkpoint_id:
        raise ValueError("base_checkpoint_id must be a non-empty string")
    if config.steps <= 0 or config.lr <= 0.0 or config.max_rank <= 0:
        raise ValueError("steps, lr, and max_rank must be positive")
    updated = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if any(name.startswith(prefix) for prefix in config.trainable_prefixes)
        and parameter_block_key(name) is not None
        and parameter.is_floating_point()
    ]
    if not updated:
        raise ValueError("no allowlisted floating canonical parameters were found")
    return updated


def _unpack_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if isinstance(batch, Mapping):
        iq = batch.get("iq", batch.get("x"))
        labels = batch.get("labels", batch.get("label", batch.get("y")))
    elif isinstance(batch, Sequence) and len(batch) == 2:
        iq, labels = batch
    else:
        raise ValueError("each source task batch must contain IQ tensor and labels")
    if not torch.is_tensor(iq) or not torch.is_tensor(labels):
        raise ValueError("each source task batch must contain IQ tensor and labels")
    if iq.size(0) != labels.reshape(-1).size(0):
        raise ValueError("IQ and labels must have the same batch dimension")
    return iq, labels.reshape(-1).long()


def _extract_logits(output: Any) -> Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, Mapping):
        names = [name for name in ("logits", "tx_logits") if name in output]
        if not names:
            raise ValueError("model output mapping must provide logits or tx_logits")
        logits = output[names[0]]
        if len(names) == 2 and (not torch.is_tensor(output[names[1]]) or not torch.equal(logits, output[names[1]])):
            raise ValueError("conflicting logits and tx_logits aliases")
    else:
        raise ValueError("model output must be a logits tensor or mapping")
    if not torch.is_tensor(logits) or not logits.is_floating_point():
        raise ValueError("logits must be a floating tensor")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite logits")
    return logits


def _make_trainable_bank(bank: WeightDeltaBank) -> WeightDeltaBank:
    entries = tuple(
        DeltaBankEntry(
            spec=entry.spec,
            basis=entry.basis.detach().clone().requires_grad_(True),
            task_coefficients=entry.task_coefficients.detach().clone().requires_grad_(False),
            effective_rank=entry.effective_rank,
            relative_error=entry.relative_error,
        )
        for entry in bank.entries
    )
    return WeightDeltaBank(
        schema=bank.schema,
        base_checkpoint_id=bank.base_checkpoint_id,
        task_keys=bank.task_keys,
        entries=entries,
    )


def build_source_expert_bank(
    model: nn.Module,
    task_batches: Mapping[DeltaTaskKey, Any],
    config: MARCOTSourceExpertConfig,
) -> SourceExpertBankResult:
    """Train each source expert from one base state and restore the model afterwards."""
    if len(task_batches) < 2:
        raise ValueError("at least two unique source tasks are required")
    updated = _resolve_updated_parameters(model, config)
    updated_names = tuple(name for name, _ in updated)
    base_state = _clone_state_dict(model)
    original_training = model.training
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    task_losses: dict[DeltaTaskKey, float] = {}
    task_deltas: dict[DeltaTaskKey, dict[str, Tensor]] = {}
    adapted_states: dict[DeltaTaskKey, dict[str, Tensor]] = {}
    try:
        for task_key, batch in task_batches.items():
            iq, labels = _unpack_batch(batch)
            model.load_state_dict(base_state, strict=True)
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name in updated_names)
            optimizer = torch.optim.SGD([parameter for _, parameter in updated], lr=float(config.lr))
            final_loss = None
            for _ in range(int(config.steps)):
                optimizer.zero_grad(set_to_none=True)
                logits = _extract_logits(model(iq))
                if logits.size(0) != labels.size(0):
                    raise ValueError("logits and labels must have the same batch dimension")
                loss = F.cross_entropy(logits, labels.to(device=logits.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite supervised cross-entropy")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for _, parameter in updated):
                    raise FloatingPointError("non-finite or missing allowlisted gradient")
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for _, parameter in updated):
                    raise FloatingPointError("non-finite allowlisted parameter")
                final_loss = float(loss.detach().item())
            adapted = _clone_state_dict(model)
            task_deltas[task_key] = extract_block_delta(
                base_state, adapted, prefixes=config.trainable_prefixes
            )
            adapted_states[task_key] = adapted
            task_losses[task_key] = float(final_loss)
        rank = min(int(config.max_rank), len(task_deltas) - 1)
        bank = fit_weight_delta_bank(
            config.base_checkpoint_id,
            task_deltas,
            max_rank=rank,
        )
        return SourceExpertBankResult(
            bank=_make_trainable_bank(bank),
            task_losses=task_losses,
            updated_parameter_names=updated_names,
            task_deltas=task_deltas,
            adapted_states=adapted_states,
        )
    finally:
        model.load_state_dict(base_state, strict=True)
        model.train(original_training)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
