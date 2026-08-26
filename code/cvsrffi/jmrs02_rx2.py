"""Focused numerical repair for the JMRS02 receiver waveform corrector.

RX2 is RX1 with a finite zero-point correction norm and a mandatory real-Core
backward probe. RX0 learns the same global correction and gate without reading
the IQ receiver condition. No gradient element is ever sanitized or silently
discarded.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from cvsrffi.jmrs02_j1 import IdentityInitRCX, J1Config


RX2_ROWS = ("B0", "RX0", "RX2")


def build_rx2_module(row: str, cfg: J1Config) -> IdentityInitRCX:
    normalized = str(row).strip().upper()
    if normalized not in ("RX0", "RX2"):
        raise ValueError(f"unsupported RX2 row: {row}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed + (707 if normalized == "RX0" else 808))
        return IdentityInitRCX(cfg, conditioning_enabled=normalized == "RX2")


def require_finite_gradients(model: nn.Module) -> dict[str, float | int]:
    nonfinite = 0
    total_sq = 0.0
    estimator_sq = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        invalid = ~torch.isfinite(gradient)
        nonfinite += int(invalid.sum().item())
        if bool(invalid.any()):
            raise FloatingPointError(
                f"non-finite RX2 gradient in {name}: {int(invalid.sum().item())} elements"
            )
        norm_sq = float(gradient.float().square().sum().item())
        total_sq += norm_sq
        if name.startswith("estimator."):
            estimator_sq += norm_sq
    return {
        "nonfinite_elements": nonfinite,
        "gradient_norm": math.sqrt(total_sq),
        "estimator_grad_norm": math.sqrt(estimator_sq),
    }


def real_core_backward_probe(
    model: nn.Module,
    candidate_logits: Tensor,
    labels: Tensor,
    correction_norm: Tensor,
) -> dict[str, float | int | bool]:
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(candidate_logits, labels) + 0.01 * correction_norm.square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite RX2 real-Core smoke loss")
    loss.backward()
    health = require_finite_gradients(model)
    if float(health["estimator_grad_norm"]) <= 0.0:
        raise RuntimeError("RX2 real-Core smoke has zero estimator gradient")
    model.zero_grad(set_to_none=True)
    return {"pass": True, "loss": float(loss.detach().item()), **health}


__all__ = ["RX2_ROWS", "build_rx2_module", "real_core_backward_probe", "require_finite_gradients"]
