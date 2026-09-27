"""Numerically bounded domain confusion with explicit gradient ownership."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
import torch.nn.functional as F


def bounded_domain_confusion_loss(
    logits: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return ``KL(p(domain|z)||Uniform)`` in float32.

    The per-row objective is bounded by ``[0, log(num_domains)]`` and remains
    finite for saturated float16/bfloat16 logits.
    """

    if logits.ndim != 2 or int(logits.shape[1]) < 2:
        raise ValueError("domain logits must have shape [batch, domains>=2]")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        work = logits.float()
        if not bool(torch.isfinite(work).all()):
            raise FloatingPointError("domain confusion logits contain non-finite values")
        log_probability = F.log_softmax(work, dim=-1)
        probability = log_probability.exp()
        per_row = (probability * log_probability).sum(dim=-1) + math.log(
            int(work.shape[1])
        )
        per_row = per_row.clamp(min=0.0, max=math.log(int(work.shape[1])))
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    return per_row.mean()


@contextmanager
def _frozen_parameters(module: torch.nn.Module) -> Iterator[None]:
    states = [bool(parameter.requires_grad) for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(module.parameters(), states):
            parameter.requires_grad_(state)


def bounded_domain_objectives(
    domain_head: torch.nn.Module,
    z_id: torch.Tensor,
    domains: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return isolated discriminator and identity-confusion objectives.

    The discriminator path sees detached features. The confusion path sees
    trainable features while the discriminator parameters are frozen for that
    forward graph.
    """

    if z_id.ndim != 2:
        raise ValueError("z_id must have shape [batch, features]")
    target = torch.as_tensor(domains, device=z_id.device).long().reshape(-1)
    if int(target.numel()) != int(z_id.shape[0]):
        raise ValueError("domains must contain one label per z_id row")
    if not bool(torch.isfinite(z_id.detach().float()).all()):
        raise FloatingPointError("z_id contains non-finite values before domain objectives")
    with torch.autocast(device_type=z_id.device.type, enabled=False):
        discriminator_logits = domain_head(z_id.detach().float())
    if not bool(torch.isfinite(discriminator_logits).all()):
        raise FloatingPointError("domain discriminator logits contain non-finite values")
    with torch.autocast(device_type=z_id.device.type, enabled=False):
        discriminator = F.cross_entropy(discriminator_logits.float(), target)
    with _frozen_parameters(domain_head):
        with torch.autocast(device_type=z_id.device.type, enabled=False):
            confusion_logits = domain_head(z_id.float())
    confusion = bounded_domain_confusion_loss(confusion_logits)
    return {
        "discriminator": discriminator,
        "confusion": confusion,
        "discriminator_logits": discriminator_logits,
        "confusion_logits": confusion_logits,
    }
