"""Explicit auxiliary gradient routing; never route response prediction through GRL."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping

import torch


class IndependentAuxiliaryTransaction:
    """A head-only optimizer whose overflow/clip/scale cannot affect the model."""
    def __init__(self, parameters, *, lr, weight_decay, amp=False, max_grad_norm=0.):
        self.parameters = list(parameters)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay)
        # N607 uses PyTorch 2.1, before the unified torch.amp scaler API.
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.max_grad_norm = float(max_grad_norm)

    def step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        scale = float(self.scaler.get_scale())
        gradients = torch.autograd.grad(self.scaler.scale(loss), self.parameters,
                                        allow_unused=True, retain_graph=True)
        present = [g for g in gradients if g is not None]
        for p, g in zip(self.parameters, gradients):
            p.grad = g.detach() if g is not None else None
        finite = bool(present) and bool(torch.isfinite(loss.detach())) and all(
            bool(torch.isfinite(g).all()) for g in present)
        norm = float(_norm([g / scale for g in present]))
        if present:
            self.scaler.unscale_(self.optimizer)
            if finite:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return {"auxiliary_step_applied": float(finite), "auxiliary_scale": scale,
                "auxiliary_nonfinite": float(not finite),
                "response_grad_auxiliary": norm if math.isfinite(norm) else None}

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict()}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])


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


def routed_response_gradients(*, identity_loss, response_loss, roles, scale,
                              lambda_resp, joint_open, gradient_cap, head_only=False,
                              response_group_losses=None):
    """Compute the exact routed/capped contributions without changing .grad.

    Used by both the main update and disposable actual-loss counterfactuals.
    Returns unscaled contributions; assignment must use add_(..., alpha=scale).
    """
    eligible = list(roles["auxiliary"])
    if not head_only:
        eligible += roles["domain"]
        if joint_open:
            eligible += roles["identity_tail"]
    params = [p for _, p in eligible]
    scale = float(scale)
    weighted_resp = float(lambda_resp) * response_loss
    # Scale before autograd traverses fp16 activations, then unscale the returned
    # fp32 parameter gradients for routing/caps. Scaling only at assignment is
    # too late to prevent intermediate-gradient underflow.
    if response_group_losses is not None and lambda_resp:
        if set(response_group_losses) != {"predictor_full", "identity", "domain"}:
            raise ValueError("decomposed routing requires exactly three independent objectives")
        gradients_by_id = {}
        for role, objective in (("auxiliary", "predictor_full"), ("domain", "domain"), ("identity_tail", "identity")):
            selected = [(name, p) for name, p in roles[role] if any(p is q for _, q in eligible)]
            gradients = _grad(float(lambda_resp)*response_group_losses[objective], [p for _, p in selected], scale=scale)
            gradients_by_id.update({id(p): g for (_,p),g in zip(selected, gradients)})
        resp_grad = tuple(gradients_by_id.get(id(p)) for p in params)
    else:
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
    by_role = {key: {id(p) for _, p in value} for key, value in roles.items()}
    actual = {key: [] for key in roles}
    contributions = []
    for (_, p), g in zip(eligible, resp_grad):
        if g is None:
            continue
        contribution = g * (coefficient if id(p) in tail_ids else 1.)
        for role, ids in by_role.items():
            if id(p) in ids:
                actual[role].append(contribution)
        contributions.append((p, contribution.detach()))
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
    return contributions, logs


def response_backward(*, baseline_loss, identity_loss, response_loss, decision_loss,
                      cross_loss, roles, scaler, lambda_resp, lambda_dec, lambda_cross,
                      joint_open, gradient_cap, head_only=False, response_group_losses=None):
    """One backward transaction followed by one caller-owned optimizer step."""
    scale = float(scaler.get_scale())
    contributions, logs = routed_response_gradients(identity_loss=identity_loss,
        response_loss=response_loss, roles=roles, scale=scale, lambda_resp=lambda_resp,
        joint_open=joint_open, gradient_cap=gradient_cap, head_only=head_only,
        response_group_losses=response_group_losses)
    aux_loss = baseline_loss + float(lambda_dec) * decision_loss + float(lambda_cross) * cross_loss
    scaler.scale(aux_loss).backward()
    for p, contribution in contributions:
        if p.grad is None:
            p.grad = contribution * scale
        else:
            p.grad.add_(contribution, alpha=scale)
    return logs
