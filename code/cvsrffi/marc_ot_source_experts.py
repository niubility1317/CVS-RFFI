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
    paired_consistency_weight: float = 0.0
    expert_mode: str = "legacy_final_step"


@dataclass
class SourceExpertBankResult:
    bank: WeightDeltaBank
    task_losses: dict[DeltaTaskKey, float]
    updated_parameter_names: tuple[str, ...]
    task_deltas: dict[DeltaTaskKey, dict[str, Tensor]]
    adapted_states: dict[DeltaTaskKey, dict[str, Tensor]]
    selected_steps: dict[DeltaTaskKey, int]
    select_losses: dict[DeltaTaskKey, float]


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
    if not 0.0 <= float(config.paired_consistency_weight) <= 1.0:
        raise ValueError("paired_consistency_weight must be in [0, 1]")
    if config.expert_mode not in {"legacy_final_step", "stratified_select"}:
        raise ValueError("expert_mode must be legacy_final_step or stratified_select")
    if config.expert_mode == "legacy_final_step" and float(config.paired_consistency_weight) != 0.0:
        raise ValueError("legacy expert mode cannot enable paired consistency")
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


def _unpack_task_splits(batch: Any) -> tuple[Any, Any]:
    if not isinstance(batch, Mapping) or not any(
        name in batch for name in ("expert_fit", "expert_select")
    ):
        return batch, batch
    if not all(name in batch for name in ("expert_fit", "expert_select")):
        raise ValueError("source task must contain expert_fit and expert_select")
    fit, select = batch["expert_fit"], batch["expert_select"]
    if not isinstance(fit, Mapping) or not isinstance(select, Mapping):
        raise ValueError("expert_fit and expert_select must be mappings")
    fit_ids = tuple(str(value) for value in fit.get("physical_ids", ()))
    select_ids = tuple(str(value) for value in select.get("physical_ids", ()))
    if not fit_ids or not select_ids or set(fit_ids).intersection(select_ids):
        raise ValueError("expert_fit and expert_select physical IDs must be non-empty and disjoint")
    _, fit_labels = _unpack_batch(fit)
    _, select_labels = _unpack_batch(select)
    expected = set(torch.cat((fit_labels, select_labels)).tolist())
    if set(fit_labels.tolist()) != expected or set(select_labels.tolist()) != expected:
        raise ValueError("expert_fit and expert_select must each cover all old classes")
    return fit, select


def _paired_iq(fit: Any) -> tuple[Tensor, Tensor] | None:
    if not isinstance(fit, Mapping):
        return None
    clean, leo = fit.get("clean_iq"), fit.get("leo_iq")
    if clean is None and leo is None:
        return None
    if not torch.is_tensor(clean) or not torch.is_tensor(leo) or clean.shape != leo.shape:
        raise ValueError("paired clean_iq and leo_iq must have identical tensor geometry")
    return clean, leo


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
    original_training_modes = {name: module.training for name, module in model.named_modules()}
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    task_losses: dict[DeltaTaskKey, float] = {}
    task_deltas: dict[DeltaTaskKey, dict[str, Tensor]] = {}
    adapted_states: dict[DeltaTaskKey, dict[str, Tensor]] = {}
    selected_steps: dict[DeltaTaskKey, int] = {}
    select_losses: dict[DeltaTaskKey, float] = {}
    try:
        for task_key, batch in task_batches.items():
            strict_split = isinstance(batch, Mapping) and all(
                name in batch for name in ("expert_fit", "expert_select")
            )
            if config.expert_mode == "legacy_final_step":
                if strict_split:
                    raise ValueError("legacy expert mode requires one single batch per task")
                fit_batch, select_batch = batch, batch
            else:
                fit_batch, select_batch = _unpack_task_splits(batch)
                if not strict_split:
                    raise ValueError("stratified_select expert mode requires fit/select splits")
            iq, labels = _unpack_batch(fit_batch)
            select_iq, select_labels = _unpack_batch(select_batch)
            paired = _paired_iq(fit_batch)
            model.load_state_dict(base_state, strict=True)
            model.eval()
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name in updated_names)
            optimizer = torch.optim.SGD([parameter for _, parameter in updated], lr=float(config.lr))
            if config.expert_mode == "legacy_final_step":
                final_loss = None
                for _step in range(1, int(config.steps) + 1):
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
                best_step = int(config.steps)
                best_select_loss = float(final_loss)
                best_state = _clone_state_dict(model)
            else:
                with torch.no_grad():
                    select_logits = _extract_logits(model(select_iq))
                    expected_classes = set(range(int(select_logits.shape[-1])))
                    if (
                        set(labels.tolist()) != expected_classes
                        or set(select_labels.tolist()) != expected_classes
                    ):
                        raise ValueError(
                            "expert_fit and expert_select must each cover all old classes"
                        )
                    best_select_loss = float(
                        F.cross_entropy(select_logits, select_labels.to(select_logits.device)).item()
                    )
                best_step = 0
                best_state = _clone_state_dict(model)
                final_loss = best_select_loss
                for step in range(1, int(config.steps) + 1):
                    optimizer.zero_grad(set_to_none=True)
                    logits = _extract_logits(model(iq))
                    if logits.size(0) != labels.size(0):
                        raise ValueError("logits and labels must have the same batch dimension")
                    loss = F.cross_entropy(logits, labels.to(device=logits.device))
                    if paired is not None and float(config.paired_consistency_weight) > 0.0:
                        clean_logits = _extract_logits(model(paired[0]))
                        leo_logits = _extract_logits(model(paired[1]))
                        loss = loss + float(config.paired_consistency_weight) * F.mse_loss(
                            clean_logits, leo_logits
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite supervised cross-entropy")
                    loss.backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for _, parameter in updated):
                        raise FloatingPointError("non-finite or missing allowlisted gradient")
                    optimizer.step()
                    if any(not torch.isfinite(parameter).all() for _, parameter in updated):
                        raise FloatingPointError("non-finite allowlisted parameter")
                    final_loss = float(loss.detach().item())
                    with torch.no_grad():
                        select_logits = _extract_logits(model(select_iq))
                        candidate_loss = float(
                            F.cross_entropy(
                                select_logits, select_labels.to(select_logits.device)
                            ).item()
                        )
                    if candidate_loss < best_select_loss:
                        best_select_loss = candidate_loss
                        best_step = step
                        best_state = _clone_state_dict(model)
            model.load_state_dict(best_state, strict=True)
            adapted = _clone_state_dict(model)
            task_deltas[task_key] = extract_block_delta(
                base_state, adapted, prefixes=config.trainable_prefixes
            )
            adapted_states[task_key] = adapted
            task_losses[task_key] = float(final_loss)
            selected_steps[task_key] = int(best_step)
            select_losses[task_key] = float(best_select_loss)
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
            selected_steps=selected_steps,
            select_losses=select_losses,
        )
    finally:
        model.load_state_dict(base_state, strict=True)
        for name, module in model.named_modules():
            module.train(original_training_modes[name])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
