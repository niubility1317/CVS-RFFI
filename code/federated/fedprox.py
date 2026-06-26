from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn as nn


def _candidate_keys(name: str):
    if name.startswith("module."):
        yield name
        yield name[len("module.") :]
    else:
        yield name
        yield f"module.{name}"


def compute_fedprox_loss(
    model: nn.Module,
    global_params: Mapping[str, torch.Tensor],
    mu: float,
    exclude_keys: Iterable[str] | None = None,
) -> torch.Tensor:
    ref = next(model.parameters(), None)
    if ref is None:
        return torch.tensor(0.0)
    if float(mu) <= 0.0:
        return ref.new_tensor(0.0)
    excluded = set(exclude_keys or set())
    prox = ref.new_tensor(0.0)
    for name, param in model.named_parameters():
        key = None
        for cand in _candidate_keys(name):
            if cand in global_params:
                key = cand
                break
        if key is None or key in excluded:
            continue
        target = global_params[key].to(device=param.device, dtype=param.dtype)
        prox = prox + torch.sum((param - target) ** 2)
    return 0.5 * float(mu) * prox
