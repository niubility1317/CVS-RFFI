from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from paper_reproduction.receiver_agnostic_twostage_uda.losses import dann_loss, stage2_lmmd_objective
from paper_reproduction.receiver_agnostic_twostage_uda.sampling import (
    balanced_source_replay_indices,
    balanced_target_selection,
    fine_tune_budget_from_unlabeled,
    rank_uncertain_samples,
)


def _tensor_batch(batch: dict[str, Any], key: str, *, device: torch.device | str | None = None) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if device is not None:
        value = value.to(device)
    return value


def dann_stage1_train_step(
    model: torch.nn.Module,
    source_batch: dict[str, Any],
    target_batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    domain_weight: float = 1.0,
    grl_lambda: float = 1.0,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """One paper Stage1 DANN update; target labels are intentionally unused."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    source_iq = _tensor_batch(source_batch, "iq", device=device).float()
    source_labels = _tensor_batch(source_batch, "label", device=device).long()
    target_iq = _tensor_batch(target_batch, "iq", device=device).float()
    source_outputs = model(source_iq, grl_lambda=grl_lambda)
    target_outputs = model(target_iq, grl_lambda=grl_lambda)
    losses = dann_loss(source_outputs, target_outputs, source_labels, domain_weight=domain_weight)
    losses["loss"].backward()
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def lmmd_stage2_train_step(
    model: torch.nn.Module,
    source_batch: dict[str, Any],
    target_batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    num_classes: int,
    lmmd_lambda: float = 1.0,
    lmmd_layers: str = "activations",
    target_temperature: float = 1.0,
    target_confidence_threshold: float = 0.0,
    target_pseudo_quota_per_class: int = 0,
    detach_target_probs: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """One paper Stage2 LMMD update; target labels remain hidden for UDA."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    source_iq = _tensor_batch(source_batch, "iq", device=device).float()
    source_labels = _tensor_batch(source_batch, "label", device=device).long()
    target_iq = _tensor_batch(target_batch, "iq", device=device).float()
    source_outputs = model(source_iq, return_activations=True)
    target_outputs = model(target_iq, return_activations=True)
    losses = stage2_lmmd_objective(
        source_outputs,
        target_outputs,
        source_labels,
        num_classes=num_classes,
        lmmd_lambda=lmmd_lambda,
        lmmd_layers=lmmd_layers,
        target_temperature=target_temperature,
        target_confidence_threshold=target_confidence_threshold,
        target_pseudo_quota_per_class=target_pseudo_quota_per_class,
        detach_target_probs=detach_target_probs,
    )
    losses["loss"].backward()
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def select_fig8_labeled_target_indices(
    logits: torch.Tensor,
    *,
    strategy: str,
    denominator: int = 50,
    seed: int | None = None,
    labels: torch.Tensor | None = None,
    receivers: list[str] | None = None,
    balance_mode: str = "none",
) -> dict[str, Any]:
    budget = fine_tune_budget_from_unlabeled(int(logits.shape[0]), denominator=denominator)
    ranked = rank_uncertain_samples(logits, strategy=strategy, k=None, seed=seed)
    selected = balanced_target_selection(
        ranked,
        k=budget,
        labels=labels,
        receivers=receivers,
        balance_mode=balance_mode,
    )
    return {
        "selected": selected,
        "budget": int(budget),
        "strategy": strategy,
        "denominator": int(denominator),
        "balance_mode": str(balance_mode),
        "synthetic_smoke": False,
        "result_claim": False,
    }


def compose_fig8_finetune_batch(
    target_iq: torch.Tensor,
    target_labels: torch.Tensor,
    target_indices: torch.Tensor,
    source_iq: torch.Tensor,
    source_labels: torch.Tensor,
    *,
    source_replay_per_class: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    replay_indices = balanced_source_replay_indices(source_labels, per_class=source_replay_per_class, seed=seed)
    iq = torch.cat([target_iq[target_indices], source_iq[replay_indices]], dim=0)
    labels = torch.cat([target_labels[target_indices], source_labels[replay_indices]], dim=0).long()
    role = torch.cat(
        [
            torch.ones(target_indices.numel(), dtype=torch.long, device=target_iq.device),
            torch.zeros(replay_indices.numel(), dtype=torch.long, device=source_iq.device),
        ],
        dim=0,
    )
    return {
        "iq": iq,
        "label": labels,
        "role": role,
        "target_indices": target_indices,
        "source_replay_indices": replay_indices,
    }


def fig8_finetune_train_step(
    model: torch.nn.Module,
    finetune_batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """One optional Fig.8 fine-tuning update over selected target labels plus source replay."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iq = _tensor_batch(finetune_batch, "iq", device=device).float()
    labels = _tensor_batch(finetune_batch, "label", device=device).long()
    outputs = model(iq)
    loss = F.cross_entropy(outputs["tx_logits"], labels)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "result_claim": 0.0}
