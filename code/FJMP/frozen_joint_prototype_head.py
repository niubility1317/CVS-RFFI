"""Frozen Joint Multi-Prototype Calibration Head.

This module implements the FJ-MP head described in
``frozen_joint_multi_prototype_head_design_formula_fixed.md`` as a standalone
PyTorch component.  It intentionally keeps the frozen backbone boundary explicit:
``FrozenJointPrototypeClassifier`` detaches ``z_id_raw`` and ``z_dom`` before the
projector so prototype losses cannot update the baseline backbone.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(_as_float_tensor(x), dim=dim, eps=float(eps))


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def apply_zdom_mode(
    z_dom: torch.Tensor,
    mode: str = "normal",
    *,
    zdom_mean: Optional[torch.Tensor] = None,
    drop_prob: float = 0.0,
    training: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the required z_dom shortcut ablations.

    Supported modes match the design doc: ``normal``, ``shuffled``, ``zero``,
    ``mean``, ``random_source`` and ``dropout``.  ``random_source`` intentionally
    has the same tensor contract as a batch-level source replacement helper; a
    full dataset-level replacement can pass an already sampled source-domain
    tensor as ``z_dom`` before calling the classifier.
    """

    mode = str(mode or "normal").lower().strip()
    if mode in ("normal", "none"):
        return z_dom
    if mode == "zero":
        return torch.zeros_like(z_dom)
    if mode == "mean":
        if zdom_mean is None:
            mean = z_dom.mean(dim=0, keepdim=True)
        else:
            mean = zdom_mean.to(device=z_dom.device, dtype=z_dom.dtype).view(1, -1)
        return mean.expand_as(z_dom)
    if mode in ("shuffled", "shuffle", "random_source"):
        if z_dom.size(0) <= 1:
            return z_dom
        perm = torch.randperm(z_dom.size(0), device=z_dom.device, generator=generator)
        return z_dom[perm]
    if mode == "dropout":
        if (not training) or float(drop_prob) <= 0.0:
            return z_dom
        keep = torch.rand((z_dom.size(0), 1), device=z_dom.device, generator=generator) >= float(drop_prob)
        return z_dom * keep.to(dtype=z_dom.dtype)
    raise ValueError(f"Unsupported z_dom ablation mode: {mode}")


def freeze_frozen_backbone(model: nn.Module) -> nn.Module:
    """Put a trained baseline backbone into the required frozen/eval state."""

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_frozen_features(
    outputs: Mapping[str, object],
    *,
    id_key: str = "z_id_raw",
    dom_key: str = "z_dom",
    base_logits_key: str = "base_logits",
    strict_raw: bool = True,
) -> TensorDict:
    """Extract the feature contract required by FJ-MP training.

    The default is intentionally strict: the main experiment should consume
    ``z_id_raw`` rather than a gated/post-domain-modulated feature.  Set
    ``strict_raw=False`` only for explicit compatibility or ablation runs.
    """

    def pick_tensor(keys: Iterable[str], label: str) -> torch.Tensor:
        for key in keys:
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
        raise KeyError(f"Missing {label}; tried keys={list(keys)}; available={list(outputs.keys())}")

    id_keys = [id_key]
    if not strict_raw:
        id_keys.extend(["z_id", "id_feat_joint", "feat_joint"])
    z_id_raw = pick_tensor(id_keys, "z_id_raw")
    z_dom = pick_tensor([dom_key, "z_dom_raw"] if dom_key == "z_dom" else [dom_key], "z_dom")
    base_logits = pick_tensor([base_logits_key, "tx_logits", "logits"], "base_logits")
    out = {
        "z_id_raw": z_id_raw,
        "z_dom": z_dom,
        "base_logits": base_logits,
    }
    z_id_gated = outputs.get("z_id_gated")
    if torch.is_tensor(z_id_gated):
        out["z_id_gated"] = z_id_gated
    domain_logits = outputs.get("domain_logits", outputs.get("dom_logits"))
    if torch.is_tensor(domain_logits):
        out["domain_logits"] = domain_logits
    return out


@torch.no_grad()
def forward_frozen_backbone(
    frozen_model: nn.Module,
    x: torch.Tensor,
    *,
    id_key: str = "z_id_raw",
    dom_key: str = "z_dom",
    base_logits_key: str = "base_logits",
    strict_raw: bool = True,
    **forward_kwargs,
) -> TensorDict:
    """Run the frozen model under no_grad and return detached FJ-MP features."""

    frozen_model.eval()
    try:
        outputs = frozen_model(x, return_aux=True, return_raw_features=True, **forward_kwargs)
    except TypeError:
        outputs = frozen_model(x, return_aux=True, **forward_kwargs)
    if not isinstance(outputs, Mapping):
        raise TypeError("frozen_model must return a mapping when return_aux=True.")
    features = extract_frozen_features(
        outputs,
        id_key=id_key,
        dom_key=dom_key,
        base_logits_key=base_logits_key,
        strict_raw=strict_raw,
    )
    return {key: value.detach() for key, value in features.items()}


class ResidualDominantJointProjector(nn.Module):
    """z_id-dominant, z_dom-restricted residual joint projector."""

    def __init__(
        self,
        id_dim: int,
        dom_dim: int,
        proto_dim: int,
        hidden_dim: Optional[int] = None,
        drop: float = 0.1,
        dom_drop_prob: float = 0.3,
        init_res_scale: float = 0.1,
        max_res_scale: float = 0.5,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if id_dim <= 0 or dom_dim <= 0 or proto_dim <= 0:
            raise ValueError("id_dim, dom_dim and proto_dim must be positive.")
        hidden_dim = int(hidden_dim or max(proto_dim, (id_dim + dom_dim) // 2))
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.id_dim = int(id_dim)
        self.dom_dim = int(dom_dim)
        self.proto_dim = int(proto_dim)
        self.normalize = bool(normalize)
        self.dom_drop_prob = float(dom_drop_prob)
        self.max_res_scale = float(max_res_scale)
        if self.max_res_scale <= 0.0:
            raise ValueError("max_res_scale must be positive.")

        self.id_proj = nn.Sequential(
            nn.Linear(self.id_dim, self.proto_dim),
            nn.LayerNorm(self.proto_dim),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(self.id_dim + self.dom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(drop)),
            nn.Linear(hidden_dim, self.proto_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(self.dom_dim, self.proto_dim),
            nn.Sigmoid(),
        )

        init_ratio = min(max(float(init_res_scale) / self.max_res_scale, 1e-4), 1.0 - 1e-4)
        self.res_scale_logit = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32)
        )

    def residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.res_scale_logit) * self.max_res_scale

    def _maybe_drop_dom(self, z_dom: torch.Tensor) -> torch.Tensor:
        if self.training and self.dom_drop_prob > 0.0:
            keep = torch.rand((z_dom.size(0), 1), device=z_dom.device) >= self.dom_drop_prob
            return z_dom * keep.to(dtype=z_dom.dtype)
        return z_dom

    def forward(self, z_id_raw: torch.Tensor, z_dom: torch.Tensor) -> tuple[torch.Tensor, TensorDict]:
        if z_id_raw.dim() != 2 or z_dom.dim() != 2:
            raise ValueError("z_id_raw and z_dom must be [B, F] tensors.")
        if z_id_raw.size(0) != z_dom.size(0):
            raise ValueError("z_id_raw and z_dom must have the same batch size.")
        if z_id_raw.size(1) != self.id_dim or z_dom.size(1) != self.dom_dim:
            raise ValueError(
                f"Expected z_id_raw [B,{self.id_dim}] and z_dom [B,{self.dom_dim}], "
                f"got {tuple(z_id_raw.shape)} and {tuple(z_dom.shape)}."
            )

        z_id = _as_float_tensor(z_id_raw)
        z_dom_used = self._maybe_drop_dom(_as_float_tensor(z_dom))
        z_base = self.id_proj(z_id)
        delta = self.delta_net(torch.cat([z_id, z_dom_used], dim=1))
        gate = self.gate_net(z_dom_used)
        scale = self.residual_scale()
        residual = scale * gate * delta
        z_joint_pre = z_base + residual
        z_joint = _safe_normalize(z_joint_pre, dim=1) if self.normalize else z_joint_pre

        aux = {
            "z_base": z_base,
            "delta": delta,
            "gate": gate,
            "residual": residual,
            "residual_scale": scale.detach(),
            "delta_norm": delta.norm(dim=1).detach(),
            "z_base_norm": z_base.norm(dim=1).detach(),
            "relative_residual_norm": (
                residual.norm(dim=1) / z_base.norm(dim=1).clamp_min(1e-6)
            ).detach(),
        }
        return z_joint, aux


class MultiPrototypeHead(nn.Module):
    """Cosine C x K prototype head with learnable logit scale."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        num_prototypes: int = 14,
        init_scale: float = 8.0,
        min_scale: float = 1.0,
        max_scale: float = 30.0,
        aggregation: str = "logsumexp",
    ) -> None:
        super().__init__()
        if num_classes <= 0 or feat_dim <= 0 or num_prototypes <= 0:
            raise ValueError("num_classes, feat_dim and num_prototypes must be positive.")
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.num_prototypes = int(num_prototypes)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.aggregation = str(aggregation or "logsumexp").lower().strip()
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.num_prototypes, self.feat_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        init_scale = min(max(float(init_scale), self.min_scale), self.max_scale)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))

    def current_scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(self.min_scale, self.max_scale)

    def forward(self, z_joint: torch.Tensor) -> TensorDict:
        if z_joint.dim() != 2 or z_joint.size(1) != self.feat_dim:
            raise ValueError(f"Expected z_joint shaped [B,{self.feat_dim}], got {tuple(z_joint.shape)}.")
        z = _safe_normalize(z_joint, dim=1)
        p = _safe_normalize(self.prototypes, dim=-1)
        scale = self.current_scale()

        proto_sim = torch.einsum("bf,ckf->bck", z, p).clamp(-1.0, 1.0)
        proto_scores = proto_sim * scale
        if self.aggregation == "top2_mean":
            topk = min(2, self.num_prototypes)
            class_logits = proto_scores.topk(k=topk, dim=2).values.mean(dim=2)
        elif self.aggregation in {"trimmed_lse", "trimmed_logsumexp"} and self.num_prototypes > 2:
            sorted_scores = proto_scores.sort(dim=2).values[:, :, 1:-1]
            class_logits = torch.logsumexp(sorted_scores - math.log(sorted_scores.size(2)), dim=2)
        elif self.aggregation == "max":
            class_logits = proto_scores.max(dim=2).values
        elif self.aggregation == "mean":
            class_logits = proto_scores.mean(dim=2)
        else:
            class_logits = torch.logsumexp(proto_scores - math.log(self.num_prototypes), dim=2)
        class_prob = F.softmax(class_logits, dim=1)
        pred = class_prob.argmax(dim=1)

        flat_sim = proto_sim.reshape(proto_sim.size(0), -1)
        nearest_flat = flat_sim.argmax(dim=1)
        nearest_class_idx = nearest_flat // self.num_prototypes
        nearest_proto_idx = nearest_flat % self.num_prototypes
        nearest_proto_score = flat_sim.gather(1, nearest_flat[:, None]).squeeze(1)
        proto_assign_prob = F.softmax(proto_scores, dim=2)
        proto_assignment_entropy = -(proto_assign_prob * proto_assign_prob.clamp_min(1e-12).log()).sum(dim=2)

        return {
            "logits": class_logits,
            "class_logits": class_logits,
            "head_logits": class_logits,
            "class_prob": class_prob,
            "proto_sim": proto_sim,
            "proto_scores": proto_scores,
            "proto_assign": proto_assign_prob,
            "proto_assign_prob": proto_assign_prob,
            "proto_assignment_entropy": proto_assignment_entropy,
            "pred": pred,
            "nearest_class_idx": nearest_class_idx,
            "nearest_proto_idx": nearest_proto_idx,
            "nearest_proto_score": nearest_proto_score,
            "logit_scale": scale.detach(),
        }


class FrozenJointPrototypeClassifier(nn.Module):
    """Full FJ-MP classifier: residual projector plus C x K prototype head."""

    def __init__(
        self,
        id_dim: int,
        dom_dim: int,
        num_classes: int,
        num_prototypes: int = 14,
        proto_dim: int = 128,
        hidden_dim: Optional[int] = None,
        init_scale: float = 8.0,
        drop: float = 0.1,
        dom_drop_prob: float = 0.3,
        init_res_scale: float = 0.1,
        max_res_scale: float = 0.5,
        detach_inputs: bool = True,
        aggregation: str = "logsumexp",
    ) -> None:
        super().__init__()
        self.detach_inputs = bool(detach_inputs)
        self.projector = ResidualDominantJointProjector(
            id_dim=id_dim,
            dom_dim=dom_dim,
            proto_dim=proto_dim,
            hidden_dim=hidden_dim,
            drop=drop,
            dom_drop_prob=dom_drop_prob,
            init_res_scale=init_res_scale,
            max_res_scale=max_res_scale,
            normalize=True,
        )
        self.head = MultiPrototypeHead(
            num_classes=num_classes,
            feat_dim=proto_dim,
            num_prototypes=num_prototypes,
            init_scale=init_scale,
            aggregation=aggregation,
        )

    def forward(
        self,
        z_id_raw: torch.Tensor,
        z_dom: torch.Tensor,
        *,
        zdom_mode: str = "normal",
        zdom_mean: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        z_id_in = z_id_raw.detach() if self.detach_inputs else z_id_raw
        z_dom_in = z_dom.detach() if self.detach_inputs else z_dom
        z_dom_in = apply_zdom_mode(
            z_dom_in,
            zdom_mode,
            zdom_mean=zdom_mean,
            drop_prob=self.projector.dom_drop_prob,
            training=self.training,
        )
        z_joint, proj_aux = self.projector(z_id_in, z_dom_in)
        out = self.head(z_joint)
        out["z_joint"] = z_joint
        out["proj_aux"] = proj_aux
        return out


class CalibratedFusion(nn.Module):
    """Fusion module for proto-only, base-only, calibrated logits and conservative residual fusion."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        mode: str = "calibrated_logit",
        *,
        learnable: bool = False,
        probability_weight: float = 0.5,
        base_temperature: float = 1.0,
        proto_temperature: float = 1.0,
        eta: float = 0.05,
        eta_max: float = 0.10,
        center_proto: bool = True,
    ) -> None:
        super().__init__()
        self.mode = str(mode or "calibrated_logit").lower().strip()
        self.learnable = bool(learnable)
        self.probability_weight = float(probability_weight)
        self.base_temperature = float(base_temperature)
        self.proto_temperature = float(proto_temperature)
        self.eta_max = max(float(eta_max), 1e-8)
        self.center_proto = bool(center_proto)
        if self.learnable:
            self.raw_alpha = nn.Parameter(torch.tensor(_inverse_softplus(alpha), dtype=torch.float32))
            self.raw_beta = nn.Parameter(torch.tensor(_inverse_softplus(beta), dtype=torch.float32))
            eta_ratio = min(max(float(eta) / self.eta_max, 1e-6), 1.0 - 1e-6)
            self.raw_eta = nn.Parameter(torch.tensor(math.log(eta_ratio / (1.0 - eta_ratio)), dtype=torch.float32))
        else:
            self.register_buffer("raw_alpha", torch.tensor(float(alpha), dtype=torch.float32))
            self.register_buffer("raw_beta", torch.tensor(float(beta), dtype=torch.float32))
            self.register_buffer("raw_eta", torch.tensor(float(eta), dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) if self.learnable else self.raw_alpha.clamp_min(0.0)

    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta) if self.learnable else self.raw_beta.clamp_min(0.0)

    def eta(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self.raw_eta) * self.eta_max
        return self.raw_eta.clamp(0.0, self.eta_max)

    def forward(
        self,
        *,
        base_logits: Optional[torch.Tensor],
        proto_logits: torch.Tensor,
        accept_proto: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        mode = self.mode
        proto_logits = _as_float_tensor(proto_logits)
        if mode == "proto_only":
            logits = proto_logits
        elif mode == "base_only":
            if base_logits is None:
                raise ValueError("base_logits is required for base_only fusion.")
            logits = _as_float_tensor(base_logits)
        elif mode == "simple_logit":
            if base_logits is None:
                raise ValueError("base_logits is required for simple_logit fusion.")
            logits = _as_float_tensor(base_logits) + self.alpha() * proto_logits
        elif mode == "calibrated_logit":
            if base_logits is None:
                raise ValueError("base_logits is required for calibrated_logit fusion.")
            logits = self.beta() * _as_float_tensor(base_logits) + self.alpha() * proto_logits
        elif mode == "probability_ensemble":
            if base_logits is None:
                raise ValueError("base_logits is required for probability_ensemble fusion.")
            w = min(max(self.probability_weight, 0.0), 1.0)
            base_prob = F.softmax(_as_float_tensor(base_logits) / max(self.base_temperature, 1e-6), dim=1)
            proto_prob = F.softmax(proto_logits / max(self.proto_temperature, 1e-6), dim=1)
            logits = ((1.0 - w) * base_prob + w * proto_prob).clamp_min(1e-12).log()
        elif mode == "residual":
            if base_logits is None:
                raise ValueError("base_logits is required for residual fusion.")
            residual = proto_logits
            if self.center_proto:
                residual = residual - residual.mean(dim=1, keepdim=True)
            fused = _as_float_tensor(base_logits) + self.eta() * residual
            if accept_proto is not None:
                mask = accept_proto.to(device=proto_logits.device, dtype=torch.bool).view(-1, 1)
                logits = torch.where(mask, fused, _as_float_tensor(base_logits))
            else:
                logits = fused
        elif mode == "confidence_gated":
            if base_logits is None or accept_proto is None:
                raise ValueError("base_logits and accept_proto are required for confidence_gated fusion.")
            fused = self.beta() * _as_float_tensor(base_logits) + self.alpha() * proto_logits
            mask = accept_proto.to(device=proto_logits.device, dtype=torch.bool).view(-1, 1)
            logits = torch.where(mask, fused, _as_float_tensor(base_logits))
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

        return {
            "logits": logits,
            "alpha": self.alpha().detach(),
            "beta": self.beta().detach(),
            "eta": self.eta().detach(),
        }


class ConfidenceGate(nn.Module):
    """OOD-aware gate for deciding whether prototype logits are accepted."""

    def __init__(
        self,
        threshold_score: float = 0.0,
        threshold_margin: float = 0.0,
        threshold_entropy: float = 1.5,
        threshold_ood: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.threshold_score = float(threshold_score)
        self.threshold_margin = float(threshold_margin)
        self.threshold_entropy = float(threshold_entropy)
        self.threshold_ood = None if threshold_ood is None else float(threshold_ood)

    def forward(
        self,
        *,
        proto_logits: torch.Tensor,
        proto_scores: torch.Tensor,
        nearest_proto_score: torch.Tensor,
        ood_distance: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        logits = _as_float_tensor(proto_logits)
        if logits.size(1) < 2:
            class_margin = torch.full((logits.size(0),), float("inf"), device=logits.device)
        else:
            top2 = logits.topk(k=2, dim=1).values
            class_margin = top2[:, 0] - top2[:, 1]
        pred = logits.argmax(dim=1)

        scores = _as_float_tensor(proto_scores)
        idx = torch.arange(scores.size(0), device=scores.device)
        assign = F.softmax(scores[idx, pred, :], dim=1)
        entropy = -(assign * assign.clamp_min(1e-12).log()).sum(dim=1)

        accept = (
            (_as_float_tensor(nearest_proto_score).view(-1) > self.threshold_score)
            & (class_margin > self.threshold_margin)
            & (entropy < self.threshold_entropy)
        )
        if self.threshold_ood is not None and ood_distance is not None:
            accept = accept & (_as_float_tensor(ood_distance).view(-1) < self.threshold_ood)

        return {
            "accept_proto": accept,
            "class_margin": class_margin.detach(),
            "assignment_entropy": entropy.detach(),
            "nearest_proto_score": nearest_proto_score.detach(),
        }


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    student = _as_float_tensor(student_logits).clamp(-30.0, 30.0)
    teacher = _as_float_tensor(teacher_logits).detach().clamp(-30.0, 30.0)
    log_p = F.log_softmax(student / t, dim=1)
    q = F.softmax(teacher / t, dim=1).clamp_min(1e-8)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


def do_no_harm_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize fused logits when they increase per-sample CE over the frozen baseline."""

    ce_fused = F.cross_entropy(_as_float_tensor(fused_logits), y.long(), reduction="none")
    ce_base = F.cross_entropy(_as_float_tensor(base_logits).detach(), y.long(), reduction="none")
    return F.relu(ce_fused - ce_base + float(margin)).mean()


def _true_class_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = _as_float_tensor(logits)
    y = y.to(device=logits.device).long().view(-1)
    true = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, num_classes=logits.size(1)).bool(), float("-inf"))
    other = masked.max(dim=1).values
    return true - other


def margin_preservation_loss(
    fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    y: torch.Tensor,
    delta: float = 0.0,
) -> torch.Tensor:
    """Protect the frozen baseline's true-class margin from prototype fusion erosion."""

    base_margin = _true_class_margin(base_logits, y).detach()
    fused_margin = _true_class_margin(fused_logits, y)
    return F.relu(base_margin - fused_margin + float(delta)).mean()


@torch.no_grad()
def compute_relative_harm_metrics(
    base_logits: torch.Tensor,
    fused_logits: torch.Tensor,
    y: torch.Tensor,
    *,
    proto_logits: Optional[torch.Tensor] = None,
    accept_proto: Optional[torch.Tensor] = None,
    ood_reject: Optional[torch.Tensor] = None,
) -> TensorDict:
    """Compute baseline-relative FJMP rescue/harm diagnostics for one split."""

    y = y.to(device=fused_logits.device).long().view(-1)
    base_logits = _as_float_tensor(base_logits).to(device=fused_logits.device)
    fused_logits = _as_float_tensor(fused_logits)
    pred_base = base_logits.argmax(dim=1)
    pred_fused = fused_logits.argmax(dim=1)
    base_ok = pred_base.eq(y)
    fused_ok = pred_fused.eq(y)
    changed = pred_fused.ne(pred_base)
    rescue = (~base_ok) & fused_ok
    harm = base_ok & (~fused_ok)
    fused_conf = F.softmax(fused_logits, dim=1).max(dim=1).values
    base_conf = F.softmax(base_logits, dim=1).max(dim=1).values

    def rate(mask: torch.Tensor) -> torch.Tensor:
        return mask.float().mean() if mask.numel() > 0 else fused_logits.new_tensor(0.0)

    harm_conf_mean = fused_conf[harm].mean() if bool(harm.any()) else fused_logits.new_tensor(0.0)
    out = {
        "changed_pred_rate": rate(changed),
        "rescue_rate": rate(rescue),
        "harm_rate": rate(harm),
        "net_gain_rate": rate(rescue) - rate(harm),
        "harm_conf_mean": harm_conf_mean,
        "base_acc": rate(base_ok),
        "fused_acc": rate(fused_ok),
        "base_conf_mean": base_conf.mean(),
        "fused_conf_mean": fused_conf.mean(),
    }
    if proto_logits is not None:
        proto_pred = _as_float_tensor(proto_logits).argmax(dim=1)
        out["proto_acc"] = rate(proto_pred.eq(y))
    if accept_proto is not None:
        out["proto_accept_rate"] = accept_proto.to(device=fused_logits.device).bool().float().mean()
    if ood_reject is not None:
        out["ood_reject_rate"] = ood_reject.to(device=fused_logits.device).bool().float().mean()
    return out


def inter_class_prototype_sep_loss(prototypes: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    p = _safe_normalize(prototypes, dim=-1)
    c, k, fdim = p.shape
    flat = p.reshape(c * k, fdim)
    sim = flat @ flat.t()
    labels = torch.arange(c, device=p.device).repeat_interleave(k)
    mask = labels[:, None] != labels[None, :]
    if not bool(mask.any()):
        return prototypes.new_tensor(0.0)
    return F.relu(sim[mask] - float(margin)).mean()


def intra_class_anti_collapse_loss(prototypes: torch.Tensor, margin: float = 0.9) -> torch.Tensor:
    p = _safe_normalize(prototypes, dim=-1)
    c, k, _ = p.shape
    if k <= 1:
        return prototypes.new_tensor(0.0)
    sim = torch.einsum("ckf,clf->ckl", p, p)
    mask = ~torch.eye(k, device=p.device, dtype=torch.bool).view(1, k, k)
    return F.relu(sim[mask.expand(c, k, k)] - float(margin)).mean()


def coverage_loss(
    proto_sim: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    if proto_sim.dim() != 3:
        raise ValueError("proto_sim must be [B, C, K].")
    y = y.long().view(-1)
    idx = torch.arange(proto_sim.size(0), device=proto_sim.device)
    true_best = proto_sim[idx, y, :].max(dim=1).values
    return F.relu(float(margin) - true_best).mean()


def true_class_usage_loss(
    proto_scores: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    min_usage: float = 0.005,
    max_usage: float = 0.70,
) -> torch.Tensor:
    if proto_scores.dim() != 3:
        raise ValueError("proto_scores must be [B, C, K].")
    bsz, classes, k = proto_scores.shape
    if int(num_classes) != classes:
        raise ValueError(f"num_classes={num_classes} does not match proto_scores C={classes}.")
    y = y.long().view(-1)
    valid = (y >= 0) & (y < classes)
    if not bool(valid.any()):
        return proto_scores.new_tensor(0.0)
    idx = torch.arange(bsz, device=proto_scores.device)[valid]
    yy = y[valid]
    true_scores = proto_scores[idx, yy, :].float()
    assign = F.softmax(true_scores, dim=1)

    usage_sum = torch.zeros(classes, k, device=proto_scores.device, dtype=torch.float32)
    count = torch.zeros(classes, device=proto_scores.device, dtype=torch.float32)
    usage_sum.index_add_(0, yy, assign)
    count.index_add_(0, yy, torch.ones_like(yy, dtype=torch.float32))
    usage = usage_sum / count.clamp_min(1.0).unsqueeze(1)

    usage_valid = usage[count > 0]
    if usage_valid.numel() == 0:
        return proto_scores.new_tensor(0.0)
    dead_penalty = F.relu(float(min_usage) - usage_valid).mean()
    dominance_penalty = F.relu(usage_valid.max(dim=1).values - float(max_usage)).mean()
    return dead_penalty + dominance_penalty


def delta_regularization_loss(proj_aux: Mapping[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    delta = proj_aux["delta"]
    z_base = proj_aux["z_base"]
    gate = proj_aux["gate"]
    scale = proj_aux["residual_scale"].to(device=delta.device, dtype=delta.dtype)
    residual = scale * gate * delta
    ratio = residual.norm(dim=1) / z_base.norm(dim=1).clamp_min(float(eps))
    return (ratio * ratio).mean()


def compute_fjmp_loss(
    out: Mapping[str, torch.Tensor],
    y: torch.Tensor,
    *,
    base_logits: Optional[torch.Tensor] = None,
    ce_on: str = "fused",
    ce_proto_weight: float = 1.0,
    ce_fused_weight: float = 1.0,
    kd_on: str = "fused",
    lambda_kd: float = 0.3,
    kd_temperature: float = 4.0,
    lambda_dnh: float = 0.0,
    dnh_margin: float = 0.0,
    lambda_margin_preserve: float = 0.0,
    margin_preserve_delta: float = 0.0,
    lambda_sep: float = 0.01,
    lambda_div: float = 0.003,
    lambda_usage: float = 0.003,
    lambda_delta: float = 0.0005,
    lambda_cov: float = 0.0,
    sep_margin: float = 0.3,
    intra_margin: float = 0.9,
    coverage_margin: float = 0.5,
    prototypes: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
) -> TensorDict:
    """Compute CE/KD/structure losses with explicit proto-vs-fused supervision targets."""

    logits = out["logits"]
    fused_logits = out.get("fused_logits", logits)
    proto_logits = out.get("proto_logits", logits)
    proto_scores = out["proto_scores"]
    if prototypes is None:
        prototypes = out.get("prototypes", None)
    if prototypes is None:
        raise ValueError("prototypes must be provided for separation/diversity losses.")
    classes = int(num_classes or logits.size(1))

    ce_on = str(ce_on or "fused").lower().strip()
    if ce_on == "none":
        loss_ce_proto = logits.new_tensor(0.0)
        loss_ce_fused = logits.new_tensor(0.0)
        loss_ce = logits.new_tensor(0.0)
    elif ce_on == "proto":
        loss_ce_proto = F.cross_entropy(proto_logits, y.long())
        loss_ce_fused = logits.new_tensor(0.0)
        loss_ce = float(ce_proto_weight) * loss_ce_proto
    elif ce_on == "both":
        loss_ce_proto = F.cross_entropy(proto_logits, y.long())
        loss_ce_fused = F.cross_entropy(fused_logits, y.long())
        loss_ce = float(ce_proto_weight) * loss_ce_proto + float(ce_fused_weight) * loss_ce_fused
    else:
        loss_ce_proto = logits.new_tensor(0.0)
        loss_ce_fused = F.cross_entropy(fused_logits, y.long())
        loss_ce = float(ce_fused_weight) * loss_ce_fused

    kd_on = str(kd_on or "fused").lower().strip()
    loss_kd_proto = logits.new_tensor(0.0)
    loss_kd_fused = logits.new_tensor(0.0)
    if base_logits is not None and float(lambda_kd) > 0.0:
        if kd_on in {"proto", "both"}:
            loss_kd_proto = kd_loss(proto_logits, base_logits, kd_temperature)
        if kd_on in {"fused", "both"}:
            loss_kd_fused = kd_loss(fused_logits, base_logits, kd_temperature)
    loss_kd = loss_kd_proto + loss_kd_fused
    loss_dnh = (
        do_no_harm_loss(fused_logits, base_logits, y, margin=dnh_margin)
        if base_logits is not None and float(lambda_dnh) > 0.0
        else logits.new_tensor(0.0)
    )
    loss_margin_preserve = (
        margin_preservation_loss(fused_logits, base_logits, y, delta=margin_preserve_delta)
        if base_logits is not None and float(lambda_margin_preserve) > 0.0
        else logits.new_tensor(0.0)
    )
    loss_sep = inter_class_prototype_sep_loss(prototypes, sep_margin)
    loss_div = intra_class_anti_collapse_loss(prototypes, intra_margin)
    loss_usage = true_class_usage_loss(proto_scores, y, classes)
    loss_delta = delta_regularization_loss(out["proj_aux"]) if "proj_aux" in out else logits.new_tensor(0.0)
    loss_cov = coverage_loss(out["proto_sim"], y, coverage_margin) if lambda_cov > 0.0 else logits.new_tensor(0.0)
    total = (
        loss_ce
        + float(lambda_kd) * loss_kd
        + float(lambda_dnh) * loss_dnh
        + float(lambda_margin_preserve) * loss_margin_preserve
        + float(lambda_sep) * loss_sep
        + float(lambda_div) * loss_div
        + float(lambda_usage) * loss_usage
        + float(lambda_delta) * loss_delta
        + float(lambda_cov) * loss_cov
    )
    return {
        "loss": total,
        "loss_ce": loss_ce,
        "loss_ce_proto": loss_ce_proto,
        "loss_ce_fused": loss_ce_fused,
        "loss_kd": loss_kd,
        "loss_kd_proto": loss_kd_proto,
        "loss_kd_fused": loss_kd_fused,
        "loss_dnh": loss_dnh,
        "loss_margin_preserve": loss_margin_preserve,
        "loss_sep": loss_sep,
        "loss_div": loss_div,
        "loss_usage": loss_usage,
        "loss_delta": loss_delta,
        "loss_cov": loss_cov,
    }


@torch.no_grad()
def init_prototypes_by_class_domain(
    head: MultiPrototypeHead,
    z_joint: torch.Tensor,
    y: torch.Tensor,
    domain: torch.Tensor,
    num_domains: int,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Initialize prototypes from TX x source-domain centers.

    When K equals ``num_domains`` centers are copied directly.  For K != D this
    function uses evenly sampled domain centers as a dependency-free fallback;
    external KMeans can be used before copying for the full class-domain-kmeans
    variant described in the document.
    """

    device = head.prototypes.device
    dtype = head.prototypes.dtype
    z = _safe_normalize(z_joint.to(device=device, dtype=dtype), dim=1)
    y = y.to(device=device).long().view(-1)
    domain = domain.to(device=device).long().view(-1)
    classes, k, fdim = head.prototypes.shape
    if z.size(1) != fdim:
        raise ValueError(f"z_joint dim {z.size(1)} does not match prototype dim {fdim}.")

    sums = torch.zeros(classes, int(num_domains), fdim, device=device, dtype=dtype)
    counts = torch.zeros(classes, int(num_domains), device=device, dtype=dtype)
    valid = (y >= 0) & (y < classes) & (domain >= 0) & (domain < int(num_domains))
    if bool(valid.any()):
        flat_index = y[valid] * int(num_domains) + domain[valid]
        sums_flat = sums.view(classes * int(num_domains), fdim)
        counts_flat = counts.view(classes * int(num_domains))
        sums_flat.index_add_(0, flat_index, z[valid])
        counts_flat.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=dtype))

    class_sums = torch.zeros(classes, fdim, device=device, dtype=dtype)
    class_counts = torch.zeros(classes, device=device, dtype=dtype)
    valid_y = (y >= 0) & (y < classes)
    if bool(valid_y.any()):
        class_sums.index_add_(0, y[valid_y], z[valid_y])
        class_counts.index_add_(0, y[valid_y], torch.ones_like(y[valid_y], dtype=dtype))
    class_centers = class_sums / class_counts.clamp_min(1.0).unsqueeze(1)
    class_centers = _safe_normalize(class_centers, dim=1)

    centers = sums / counts.clamp_min(1.0).unsqueeze(-1)
    missing = counts <= 0
    if bool(missing.any()):
        centers[missing] = class_centers[:, None, :].expand_as(centers)[missing]
        if noise_std > 0.0:
            centers[missing] = centers[missing] + float(noise_std) * torch.randn_like(centers[missing])
    centers = _safe_normalize(centers, dim=-1)

    if k == int(num_domains):
        init = centers
    else:
        idx = torch.linspace(0, int(num_domains) - 1, steps=k, device=device).round().long()
        init = centers[:, idx, :]
    head.prototypes.copy_(_safe_normalize(init, dim=-1))
    return head.prototypes


@torch.no_grad()
def init_prototypes_from_feature_loader(
    proto_model: FrozenJointPrototypeClassifier,
    feature_loader: Iterable[Mapping[str, torch.Tensor]],
    num_domains: int,
) -> torch.Tensor:
    """Feature-loader variant of TX x source-domain initialization."""

    device = next(proto_model.parameters()).device
    proto_model.eval()
    z_list = []
    y_list = []
    d_list = []
    for batch in feature_loader:
        z_id_raw = batch["z_id_raw"].to(device)
        z_dom = batch["z_dom"].to(device)
        y = batch["y"].to(device)
        d = batch["domain"].to(device)
        z_joint, _ = proto_model.projector(z_id_raw, z_dom)
        z_list.append(z_joint)
        y_list.append(y)
        d_list.append(d)
    if not z_list:
        raise ValueError("feature_loader produced no batches.")
    return init_prototypes_by_class_domain(
        proto_model.head,
        torch.cat(z_list, dim=0),
        torch.cat(y_list, dim=0),
        torch.cat(d_list, dim=0),
        num_domains=num_domains,
    )


@torch.no_grad()
def compute_prototype_usage(proto_scores: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return prototype usage. With y, usage is true-class-only [C,K]."""

    if proto_scores.dim() != 3:
        raise ValueError("proto_scores must be [B, C, K].")
    bsz, classes, k = proto_scores.shape
    if y is None:
        return F.softmax(proto_scores, dim=2).mean(dim=0)
    y = y.to(device=proto_scores.device).long().view(-1)
    valid = (y >= 0) & (y < classes)
    usage_sum = torch.zeros(classes, k, device=proto_scores.device, dtype=proto_scores.dtype)
    count = torch.zeros(classes, device=proto_scores.device, dtype=proto_scores.dtype)
    if bool(valid.any()):
        idx = torch.arange(bsz, device=proto_scores.device)[valid]
        yy = y[valid]
        assign = F.softmax(proto_scores[idx, yy, :], dim=1)
        usage_sum.index_add_(0, yy, assign)
        count.index_add_(0, yy, torch.ones_like(yy, dtype=proto_scores.dtype))
    return usage_sum / count.clamp_min(1.0).unsqueeze(1)


@torch.no_grad()
def compute_dead_prototype_ratio(usage: torch.Tensor, min_usage: float = 0.005) -> torch.Tensor:
    return (usage < float(min_usage)).float().mean()


@torch.no_grad()
def compute_confidence_metrics(logits: torch.Tensor, y: torch.Tensor) -> TensorDict:
    prob = F.softmax(_as_float_tensor(logits), dim=1)
    conf, pred = prob.max(dim=1)
    correct = pred.eq(y.to(device=logits.device).long())
    return {
        "accuracy": correct.float().mean(),
        "confidence_mean": conf.mean(),
        "incorrect_high_confidence_ratio": ((~correct) & (conf >= 0.90)).float().mean(),
    }


@torch.no_grad()
def compute_ece(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    prob = F.softmax(_as_float_tensor(logits), dim=1)
    conf, pred = prob.max(dim=1)
    correct = pred.eq(y.to(device=logits.device).long()).float()
    ece = logits.new_tensor(0.0, dtype=torch.float32)
    edges = torch.linspace(0.0, 1.0, steps=int(n_bins) + 1, device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return ece


@torch.no_grad()
def compute_brier_score(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(_as_float_tensor(logits), dim=1)
    target = F.one_hot(y.to(device=logits.device).long(), num_classes=prob.size(1)).to(dtype=prob.dtype)
    return ((prob - target) ** 2).sum(dim=1).mean()


@torch.no_grad()
def compute_ood_distance(z_joint: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Distance to the closest source prototype manifold point: 1 - max cosine."""

    z = _safe_normalize(z_joint, dim=1)
    p = _safe_normalize(prototypes, dim=-1).reshape(-1, prototypes.size(-1))
    sim = z @ p.t()
    return 1.0 - sim.max(dim=1).values


@torch.no_grad()
def build_fjmp_checkpoint_payload(
    proto_model: FrozenJointPrototypeClassifier,
    *,
    args: Optional[Mapping[str, object]] = None,
    baseline_checkpoint: Optional[str] = None,
    baseline_config: Optional[Mapping[str, object]] = None,
    best_stats: Optional[Mapping[str, object]] = None,
    calibration_params: Optional[Mapping[str, object]] = None,
    diagnostics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Create the checkpoint payload recommended in the design doc."""

    return {
        "proto_model": proto_model.state_dict(),
        "args": dict(args or {}),
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_config": dict(baseline_config or {}),
        "num_classes": proto_model.head.num_classes,
        "id_dim": proto_model.projector.id_dim,
        "dom_dim": proto_model.projector.dom_dim,
        "num_prototypes": proto_model.head.num_prototypes,
        "proto_dim": proto_model.head.feat_dim,
        "init_scale": float(proto_model.head.current_scale().detach().cpu()),
        "dom_drop_prob": proto_model.projector.dom_drop_prob,
        "max_res_scale": proto_model.projector.max_res_scale,
        "feature_source": "z_id_raw + z_dom",
        "prototype_init": "class_domain_center_supported",
        "best_stats": dict(best_stats or {}),
        "calibration_params": dict(calibration_params or {}),
        "diagnostics": dict(diagnostics or {}),
    }


__all__ = [
    "ResidualDominantJointProjector",
    "MultiPrototypeHead",
    "FrozenJointPrototypeClassifier",
    "CalibratedFusion",
    "ConfidenceGate",
    "apply_zdom_mode",
    "freeze_frozen_backbone",
    "extract_frozen_features",
    "forward_frozen_backbone",
    "kd_loss",
    "inter_class_prototype_sep_loss",
    "intra_class_anti_collapse_loss",
    "coverage_loss",
    "true_class_usage_loss",
    "delta_regularization_loss",
    "do_no_harm_loss",
    "margin_preservation_loss",
    "compute_relative_harm_metrics",
    "compute_fjmp_loss",
    "init_prototypes_by_class_domain",
    "init_prototypes_from_feature_loader",
    "compute_prototype_usage",
    "compute_dead_prototype_ratio",
    "compute_confidence_metrics",
    "compute_ece",
    "compute_brier_score",
    "compute_ood_distance",
    "build_fjmp_checkpoint_payload",
]
