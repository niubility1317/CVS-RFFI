from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CSILLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    knowledge_distillation: torch.Tensor
    ewc: torch.Tensor


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def compute_ewc_penalty(
    *,
    params: Mapping[str, torch.Tensor],
    previous_params: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    penalty = _zero_like(reference)
    for name, current in params.items():
        if name not in previous_params or name not in fisher:
            continue
        prior = previous_params[name].to(device=current.device, dtype=current.dtype)
        importance = fisher[name].to(device=current.device, dtype=current.dtype)
        if importance.shape != prior.shape:
            raise ValueError(f"fisher[{name}] must match previous_params[{name}] shape")
        if current.shape != prior.shape:
            if current.ndim != prior.ndim or any(cur < old for cur, old in zip(current.shape, prior.shape)):
                raise ValueError(
                    f"current parameter {name} shape {tuple(current.shape)} cannot be sliced "
                    f"to previous shape {tuple(prior.shape)}"
                )
            old_block = tuple(slice(0, size) for size in prior.shape)
            current = current[old_block]
        penalty = penalty + 0.5 * torch.sum(importance * torch.square(current - prior))
    return penalty


def compute_csil_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    current_old_response: torch.Tensor | None = None,
    previous_old_response: torch.Tensor | None = None,
    params: Mapping[str, torch.Tensor] | None = None,
    previous_params: Mapping[str, torch.Tensor] | None = None,
    fisher: Mapping[str, torch.Tensor] | None = None,
    kd_weight: float = 1.0,
    ewc_weight: float = 1.0,
) -> CSILLoss:
    cross_entropy = F.cross_entropy(logits, labels.long())
    if current_old_response is None or previous_old_response is None:
        kd = _zero_like(cross_entropy)
    else:
        if current_old_response.shape != previous_old_response.shape:
            raise ValueError("KD responses must have the same shape")
        previous_response = previous_old_response.detach().to(
            device=current_old_response.device, dtype=current_old_response.dtype
        )
        kd = F.mse_loss(current_old_response, previous_response)
    if params is None or previous_params is None or fisher is None:
        ewc = _zero_like(cross_entropy)
    else:
        ewc = compute_ewc_penalty(params=params, previous_params=previous_params, fisher=fisher, reference=cross_entropy)
    total = cross_entropy + float(kd_weight) * kd + float(ewc_weight) * ewc
    return CSILLoss(total=total, cross_entropy=cross_entropy, knowledge_distillation=kd, ewc=ewc)
