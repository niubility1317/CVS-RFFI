"""Explicit auxiliary gradient routing; never route response prediction through GRL."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping

import torch


@dataclass
class SourceResponseGate:
    min_blocks: int = 16
    error_ratio: float = .95
    stable_checks: int = 3
    consecutive: int = 0
    checks: int = 0
    opened: bool = False
    last_ratio: float | None = None

    def observe(self, *, response_error: float, constant_error: float,
                blocks: int, source_role: str) -> dict:
        if source_role != "source_validation":
            raise ValueError("response gradient gate only accepts source_validation")
        self.checks += 1
        ratio = response_error / constant_error if constant_error > 1e-12 else math.inf
        self.last_ratio = ratio if math.isfinite(ratio) else None
        passed = (blocks >= self.min_blocks and math.isfinite(response_error)
                  and response_error >= 0 and math.isfinite(constant_error)
                  and constant_error > 1e-12 and ratio <= self.error_ratio)
        self.consecutive = self.consecutive + 1 if passed else 0
        self.opened = self.opened or self.consecutive >= self.stable_checks
        return {"gate_open": float(self.opened), "gate_pass": float(passed),
                "gate_checks": self.checks, "gate_consecutive": self.consecutive,
                "gate_ratio": self.last_ratio, "gate_blocks": blocks}

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        for key in ("min_blocks", "error_ratio", "stable_checks"):
            if state[key] != getattr(self, key):
                raise ValueError(f"response gate configuration changed: {key}")
        for key in ("consecutive", "checks", "opened", "last_ratio"):
            setattr(self, key, state[key])


def parameter_roles(model, auxiliary, tail_prefixes):
    """Classify by object identity, so aliased shared stem is never counted twice."""
    id_parameters = {id(p) for p in model.id_backbone.parameters()}
    dom_parameters = {id(p) for p in model.dom_backbone.parameters()}
    shared = id_parameters & dom_parameters
    roles = {"shared": [], "identity_tail": [], "identity_front": [],
             "domain": [], "other": [], "auxiliary": []}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in shared:
            role = "shared"
        elif id(p) in id_parameters:
            role = "identity_tail" if any(name.startswith(prefix) for prefix in tail_prefixes) else "identity_front"
        elif id(p) in dom_parameters or name.startswith("dom_enhancer."):
            role = "domain"
        else:
            role = "other"
        roles[role].append((name, p))
    if auxiliary is not None:
        for name, p in auxiliary.named_parameters():
            if p.requires_grad:
                if id(p) in seen:
                    raise ValueError("auxiliary parameters must not alias model parameters")
                roles["auxiliary"].append((name, p))
                seen.add(id(p))
    if not roles["identity_tail"]:
        raise ValueError("gradient_tail_prefixes matched no identity parameters")
    return roles


def _grad(loss, params, retain_graph=True, scale=1.):
    if not params or not loss.requires_grad:
        return tuple(None for _ in params)
    gradients = torch.autograd.grad(loss * scale, params, allow_unused=True, retain_graph=retain_graph)
    return tuple(g / scale if g is not None else None for g in gradients)


def _norm(grads):
    terms = [g.detach().float().square().sum() for g in grads if g is not None]
    return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0.)


def response_backward(*, baseline_loss, identity_loss, response_loss, decision_loss,
                      cross_loss, roles, scaler, lambda_resp, lambda_dec, lambda_cross,
                      joint_open, gradient_cap, head_only=False):
    """One backward transaction; caller performs one finite-checked optimizer step.

    Loss scalars are unscaled for routing/cap diagnostics. All assigned gradients
    use the same GradScaler scale before the caller unscales the optimizer.
    Shared/front/GRL classifier parameters receive only the baseline/decision
    contribution, never the response loss. No permanent requires_grad mutation.
    """
    eligible = list(roles["auxiliary"])
    if not head_only:
        eligible += roles["domain"]
        if joint_open:
            eligible += roles["identity_tail"]
    params = [p for _, p in eligible]
    scale = float(scaler.get_scale())
    weighted_resp = float(lambda_resp) * response_loss
    # Scale before autograd traverses fp16 activations, then unscale the returned
    # fp32 parameter gradients for routing/caps. Scaling only at assignment is
    # too late to prevent intermediate-gradient underflow.
    resp_grad = _grad(weighted_resp, params, scale=scale) if lambda_resp else tuple(None for _ in params)
    tail_ids = {id(p) for _, p in roles["identity_tail"]}
    tail_resp = [g for (_, p), g in zip(eligible, resp_grad) if id(p) in tail_ids]
    tail_params = [p for _, p in roles["identity_tail"]]
    reference = _grad(identity_loss, tail_params, scale=scale) if joint_open and not head_only else []
    response_norm = _norm(tail_resp)
    reference_norm = _norm(reference).to(response_norm.device)
    limit = float(gradient_cap) * reference_norm
    coefficient = min(1., float(limit / response_norm.clamp_min(1e-12)))
    if float(reference_norm) <= 1e-12:
        coefficient = 0.
    aux_loss = baseline_loss + float(lambda_dec) * decision_loss + float(lambda_cross) * cross_loss
    scaler.scale(aux_loss).backward()
    by_role = {key: {id(p) for _, p in value} for key, value in roles.items()}
    actual = {key: [] for key in roles}
    for (_, p), g in zip(eligible, resp_grad):
        if g is None:
            continue
        contribution = g * (coefficient if id(p) in tail_ids else 1.)
        for role, ids in by_role.items():
            if id(p) in ids:
                actual[role].append(contribution)
        if p.grad is None:
            p.grad = contribution.detach() * scale
        else:
            p.grad.add_(contribution.detach(), alpha=scale)
    dot = sum((a.detach().float() * b.detach().float()).sum()
              for a, b in zip(tail_resp, reference) if a is not None and b is not None) if len(tail_resp) == len(reference) else 0.
    denominator = float(response_norm * reference_norm)
    logs = {"response_identity_reference_norm": float(reference_norm),
            "response_identity_raw_norm": float(response_norm),
            "response_identity_cap_scale": coefficient,
            "response_identity_clipped": float(coefficient < 1 and float(response_norm) > 0),
            "response_identity_cosine": float(dot) / denominator if denominator > 1e-12 else None,
            "response_joint_open": float(joint_open)}
    logs.update({f"response_grad_{key}": float(_norm(value)) for key, value in actual.items()})
    return logs
