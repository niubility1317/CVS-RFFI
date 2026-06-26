from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.eval import accuracy_from_logits
from cvsrffi.tensors import get_nested_tensor, safe_cosine_similarity, safe_l2_normalize


def covariance_orth_loss(z_id: torch.Tensor, z_dom: torch.Tensor) -> torch.Tensor:
    z_id = torch.nan_to_num(z_id.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z_dom = torch.nan_to_num(z_dom.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z_id = z_id - z_id.mean(dim=0, keepdim=True)
    z_dom = z_dom - z_dom.mean(dim=0, keepdim=True)
    n = z_id.size(0)
    if n <= 1:
        return z_id.new_tensor(0.0)
    cov = (z_id.t() @ z_dom) / float(n - 1)
    return torch.mean(cov * cov)


def feature_norm_guard_loss(
    z: torch.Tensor,
    *,
    mode: str = "l2",
    target: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """Feature-norm guard for low-shot identity embeddings.

    RIEI-style few-shot runs benefit from preventing the identity feature from
    storing nuisance details through unbounded norm growth. The default `l2`
    mode penalizes mean squared feature norm; target modes are opt-in for
    bounded-norm sweeps.
    """
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, float("nan")
    zf = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norms = torch.linalg.vector_norm(zf, ord=2, dim=1)
    mode_l = str(mode or "l2").lower().strip()
    tgt = float(target)
    if mode_l == "mean_norm":
        loss = norms.mean()
    elif mode_l == "hinge":
        loss = torch.relu(norms - max(0.0, tgt)).pow(2).mean()
    elif mode_l == "target":
        loss = (norms - max(0.0, tgt)).pow(2).mean()
    else:
        loss = zf.pow(2).sum(dim=1).mean()
    return loss, float(norms.detach().mean().item())


def same_tx_cross_domain_consistency(z_id: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    if d is None:
        return z_id.new_tensor(0.0), float("nan")
    z = safe_l2_normalize(z_id, dim=1)
    y = y.view(-1)
    d = d.view(-1)
    losses = []
    sims = []
    for cls in torch.unique(y):
        m_cls = (y == cls)
        doms = torch.unique(d[m_cls])
        if doms.numel() < 2:
            continue
        cents = []
        for dom in doms:
            m = m_cls & (d == dom)
            if m.sum() == 0:
                continue
            cents.append(safe_l2_normalize(z[m].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        if len(cents) < 2:
            continue
        C = torch.stack(cents, dim=0)
        sim = C @ C.t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        pair_sim = sim[iu[0], iu[1]]
        losses.append((1.0 - pair_sim).mean())
        sims.append(pair_sim.mean().item())
    if len(losses) == 0:
        return z_id.new_tensor(0.0), float("nan")
    return torch.stack(losses).mean(), float(np.mean(sims))


def hard_domain_ce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    *,
    label_smoothing: float = 0.0,
    top_frac: float = 0.35,
    min_domains: int = 2,
) -> Tuple[torch.Tensor, float]:
    """Top-domain CE regularizer for receiver-robust training.

    It optimizes the hardest train domains in the current batch instead of only
    the batch average. For WiSig rx_day this pushes the model to avoid solving
    only the easy receiver/day combinations, which is useful when unseen RX
    groups such as rx7/rx8 lag behind.
    """
    ref = logits
    if d is None:
        return ref.new_tensor(0.0), float("nan")
    d = d.view(-1).long()
    y = y.view(-1).long()
    valid = d >= 0
    if not bool(valid.any()):
        return ref.new_tensor(0.0), float("nan")

    losses = []
    for dom in torch.unique(d[valid]):
        m = valid & (d == dom)
        if int(m.sum().item()) <= 0:
            continue
        losses.append(
            F.cross_entropy(
                logits[m].float(),
                y[m],
                reduction="mean",
                label_smoothing=float(label_smoothing),
            )
        )
    if len(losses) < max(1, int(min_domains)):
        return ref.new_tensor(0.0), float("nan")
    vals = torch.stack(losses)
    k = max(1, int(math.ceil(vals.numel() * max(0.0, min(1.0, float(top_frac))))))
    hard = torch.topk(vals, k=k, largest=True).values
    return hard.mean(), float(hard.detach().mean().item())


class SmoothGroupDROState:
    """EMA state for smooth worst-domain reweighting."""

    def __init__(self, momentum: float = 0.95):
        self.momentum = float(momentum)
        self.loss_ema: Dict[int, float] = {}

    def update(self, group_id: int, loss_value: float) -> None:
        gid = int(group_id)
        lv = float(loss_value)
        if not math.isfinite(lv):
            return
        if gid not in self.loss_ema:
            self.loss_ema[gid] = lv
        else:
            m = max(0.0, min(0.9999, self.momentum))
            self.loss_ema[gid] = m * self.loss_ema[gid] + (1.0 - m) * lv

    def value(self, group_id: int, fallback: float) -> float:
        return float(self.loss_ema.get(int(group_id), float(fallback)))


def smooth_groupdro_ce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    group_ids: Optional[torch.Tensor],
    state: Optional[SmoothGroupDROState],
    *,
    label_smoothing: float = 0.0,
    tau: float = 0.5,
    cap: float = 0.65,
    min_groups: int = 2,
    key_offset: int = 0,
    capped: bool = False,
) -> Tuple[torch.Tensor, float]:
    ref = logits
    if group_ids is None:
        return ref.new_tensor(0.0), float("nan")
    g = group_ids.view(-1).long()
    y = y.view(-1).long()
    valid = g >= 0
    if not bool(valid.any()):
        return ref.new_tensor(0.0), float("nan")

    losses = []
    keys = []
    for gid in torch.unique(g[valid]):
        gid_int = int(gid.item()) + int(key_offset)
        m = valid & (g == gid)
        if int(m.sum().item()) <= 0:
            continue
        loss_g = F.cross_entropy(
            logits[m].float(),
            y[m],
            reduction="mean",
            label_smoothing=float(label_smoothing),
        )
        losses.append(loss_g)
        keys.append(gid_int)
    if len(losses) < max(1, int(min_groups)):
        return ref.new_tensor(0.0), float("nan")

    vals = torch.stack(losses)
    with torch.no_grad():
        if state is not None:
            for gid_int, loss_g in zip(keys, vals.detach()):
                state.update(gid_int, float(loss_g.item()))
            ema_vals = torch.as_tensor(
                [state.value(gid_int, float(vals.detach()[i].item())) for i, gid_int in enumerate(keys)],
                device=vals.device,
                dtype=vals.dtype,
            )
        else:
            ema_vals = vals.detach()
        weights = torch.softmax(ema_vals / max(1e-4, float(tau)), dim=0)
        if capped:
            weights = torch.clamp(weights, max=max(1e-4, float(cap)))
            weights = weights / weights.sum().clamp_min(1e-12)
    return (weights.detach() * vals).sum(), float(vals.detach().max().item())


def groupdro_or_hard_domain_ce_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    state: Optional[SmoothGroupDROState],
    *,
    mode: str = "hard",
    label_smoothing: float = 0.0,
    top_frac: float = 0.35,
    min_domains: int = 2,
    tau: float = 0.5,
    cap: float = 0.65,
    rx_day_num_days: int = 4,
) -> Tuple[torch.Tensor, float]:
    mode = str(mode or "hard").lower().strip()
    if mode in ("hard", "top", "topk"):
        return hard_domain_ce_loss(
            logits,
            y,
            d,
            label_smoothing=float(label_smoothing),
            top_frac=float(top_frac),
            min_domains=int(min_domains),
        )
    if mode in ("smooth_dro", "smooth", "ema"):
        return smooth_groupdro_ce_loss(
            logits, y, d, state,
            label_smoothing=float(label_smoothing),
            tau=float(tau),
            min_groups=int(min_domains),
            capped=False,
        )
    if mode in ("smooth_dro_capped", "capped"):
        return smooth_groupdro_ce_loss(
            logits, y, d, state,
            label_smoothing=float(label_smoothing),
            tau=float(tau),
            cap=float(cap),
            min_groups=int(min_domains),
            capped=True,
        )
    if mode in ("dual_worst", "rx_day_dual"):
        if d is None:
            return logits.new_tensor(0.0), float("nan")
        dd = d.view(-1).long()
        valid = dd >= 0
        if not bool(valid.any()):
            return logits.new_tensor(0.0), float("nan")
        nday = max(1, int(rx_day_num_days))
        rx = torch.where(valid, dd // nday, dd)
        day = torch.where(valid, dd % nday, dd)
        loss_dom, hard_dom = smooth_groupdro_ce_loss(
            logits, y, dd, state,
            label_smoothing=float(label_smoothing),
            tau=float(tau),
            cap=float(cap),
            min_groups=int(min_domains),
            key_offset=0,
            capped=True,
        )
        loss_rx, hard_rx = smooth_groupdro_ce_loss(
            logits, y, rx, state,
            label_smoothing=float(label_smoothing),
            tau=float(tau),
            cap=float(cap),
            min_groups=2,
            key_offset=10000,
            capped=True,
        )
        loss_day, hard_day = smooth_groupdro_ce_loss(
            logits, y, day, state,
            label_smoothing=float(label_smoothing),
            tau=float(tau),
            cap=float(cap),
            min_groups=2,
            key_offset=20000,
            capped=True,
        )
        parts = [loss_dom, loss_rx, loss_day]
        finite_parts = [p for p in parts if torch.is_tensor(p) and torch.isfinite(p.detach()).all()]
        if not finite_parts:
            return logits.new_tensor(0.0), float("nan")
        hard_vals = [v for v in [hard_dom, hard_rx, hard_day] if math.isfinite(float(v))]
        return torch.stack(finite_parts).mean(), float(max(hard_vals) if hard_vals else float("nan"))
    raise ValueError(f"Unknown group_ce_mode={mode}")


def domain_aware_supcon_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    *,
    temperature: float = 0.12,
) -> torch.Tensor:
    """Supervised contrastive loss with positives restricted to same-TX cross-domain pairs."""
    if z is None or z.size(0) <= 1:
        return z.new_tensor(0.0)
    z = safe_l2_normalize(z, dim=1)
    y = y.view(-1).long()
    if d is None:
        return z.new_tensor(0.0)
    d = d.view(-1).long()
    valid = d >= 0
    if int(valid.sum().item()) <= 1:
        return z.new_tensor(0.0)
    logits = (z @ z.t()) / max(1e-4, float(temperature))
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    same_tx = y.view(-1, 1).eq(y.view(1, -1))
    cross_domain = d.view(-1, 1).ne(d.view(1, -1))
    valid_pair = valid.view(-1, 1) & valid.view(1, -1) & (~eye)
    pos = same_tx & cross_domain & valid_pair
    denom_mask = valid_pair
    has_pos = pos.sum(dim=1) > 0
    if not bool(has_pos.any()):
        return z.new_tensor(0.0)
    exp_logits = torch.exp(logits) * denom_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_log_prob = (log_prob * pos.float()).sum(dim=1) / pos.float().sum(dim=1).clamp_min(1.0)
    return -pos_log_prob[has_pos].mean()


def fishr_logit_gradient_variance_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    *,
    min_domains: int = 2,
) -> torch.Tensor:
    """Cheap Fishr-style proxy: match domain-level variance of classifier logit gradients."""
    if d is None or logits.size(0) <= 1:
        return logits.new_tensor(0.0)
    d = d.view(-1).long()
    valid = d >= 0
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    prob = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(y.view(-1).long(), num_classes=logits.size(1)).to(prob.dtype)
    grad_proxy = prob - one_hot
    vars_by_domain = []
    for dom in torch.unique(d[valid]):
        m = valid & (d == dom)
        if int(m.sum().item()) <= 1:
            continue
        vars_by_domain.append(grad_proxy[m].var(dim=0, unbiased=False))
    if len(vars_by_domain) < max(2, int(min_domains)):
        return logits.new_tensor(0.0)
    V = torch.stack(vars_by_domain, dim=0)
    target = V.mean(dim=0, keepdim=True).detach()
    return ((V - target) ** 2).mean()


class PrototypeMemoryBank:
    """Momentum TX/domain prototypes for cross-epoch identity consistency."""

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        *,
        momentum: float = 0.95,
        margin: float = 0.15,
        domain_align_weight: float = 0.5,
        push_weight: float = 0.1,
        min_count: int = 2,
    ):
        self.num_classes = int(num_classes)
        self.num_domains = int(max(1, num_domains))
        self.momentum = float(momentum)
        self.margin = float(margin)
        self.domain_align_weight = float(domain_align_weight)
        self.push_weight = float(push_weight)
        self.min_count = int(min_count)
        self.class_proto: Optional[torch.Tensor] = None
        self.domain_proto: Optional[torch.Tensor] = None
        self.class_count: Optional[torch.Tensor] = None
        self.domain_count: Optional[torch.Tensor] = None

    def _lazy_init(self, feat_dim: int, device, dtype) -> None:
        if self.class_proto is not None and self.class_proto.size(1) == int(feat_dim):
            return
        self.class_proto = torch.zeros(self.num_classes, int(feat_dim), device=device, dtype=dtype)
        self.domain_proto = torch.zeros(self.num_classes, self.num_domains, int(feat_dim), device=device, dtype=dtype)
        self.class_count = torch.zeros(self.num_classes, device=device, dtype=torch.long)
        self.domain_count = torch.zeros(self.num_classes, self.num_domains, device=device, dtype=torch.long)

    def loss(self, z: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._lazy_init(z.size(1), z.device, z.dtype)
        assert self.class_proto is not None and self.domain_proto is not None
        assert self.class_count is not None and self.domain_count is not None
        z_norm = safe_l2_normalize(z, dim=1)
        y = y.view(-1).long()
        valid_y = (y >= 0) & (y < self.num_classes)
        active_class = self.class_count[y.clamp(0, self.num_classes - 1)] >= int(self.min_count)
        pull_mask = valid_y & active_class
        loss_pull = z.new_tensor(0.0)
        pull_cos = float("nan")
        if bool(pull_mask.any()):
            proto = safe_l2_normalize(self.class_proto[y[pull_mask]], dim=1)
            cos = (z_norm[pull_mask] * proto.detach()).sum(dim=1).clamp(-1.0, 1.0)
            loss_pull = (1.0 - cos).mean()
            pull_cos = float(cos.detach().mean().item())

        loss_domain = z.new_tensor(0.0)
        if d is not None:
            d = d.view(-1).long()
            domain_losses = []
            for cls in torch.unique(y[valid_y]):
                cls_int = int(cls.item())
                for dom in torch.unique(d[(y == cls) & (d >= 0)]):
                    dom_int = int(dom.item())
                    if dom_int < 0 or dom_int >= self.num_domains:
                        continue
                    if int(self.domain_count[cls_int, dom_int].item()) < int(self.min_count):
                        continue
                    domain_p = safe_l2_normalize(self.domain_proto[cls_int, dom_int].view(1, -1), dim=1)
                    class_p = safe_l2_normalize(self.class_proto[cls_int].view(1, -1), dim=1)
                    domain_losses.append(1.0 - (domain_p * class_p.detach()).sum())
            if domain_losses:
                loss_domain = torch.stack(domain_losses).mean()

        loss_push = z.new_tensor(0.0)
        active = self.class_count >= int(self.min_count)
        if int(active.sum().item()) > 1:
            P = safe_l2_normalize(self.class_proto[active], dim=1)
            sim = P @ P.t()
            eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            loss_push = F.relu(sim[~eye] - float(self.margin)).pow(2).mean()

        loss = loss_pull + float(self.domain_align_weight) * loss_domain + float(self.push_weight) * loss_push
        return loss, {
            "proto_pull_cos": pull_cos,
            "proto_push": float(loss_push.detach().item()) if torch.is_tensor(loss_push) else float("nan"),
        }

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor]) -> None:
        self._lazy_init(z.size(1), z.device, z.dtype)
        assert self.class_proto is not None and self.domain_proto is not None
        assert self.class_count is not None and self.domain_count is not None
        z_norm = safe_l2_normalize(z.detach(), dim=1)
        y = y.view(-1).long()
        d = d.view(-1).long() if d is not None else None
        m = max(0.0, min(0.9999, float(self.momentum)))
        for cls in torch.unique(y[(y >= 0) & (y < self.num_classes)]):
            cls_int = int(cls.item())
            mask = y == cls_int
            if not bool(mask.any()):
                continue
            mean = safe_l2_normalize(z_norm[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            if int(self.class_count[cls_int].item()) <= 0:
                self.class_proto[cls_int].copy_(mean)
            else:
                self.class_proto[cls_int].mul_(m).add_(mean, alpha=1.0 - m)
                self.class_proto[cls_int].copy_(safe_l2_normalize(self.class_proto[cls_int].view(1, -1), dim=1).squeeze(0))
            self.class_count[cls_int] += int(mask.sum().item())
            if d is None:
                continue
            for dom in torch.unique(d[mask & (d >= 0)]):
                dom_int = int(dom.item())
                if dom_int < 0 or dom_int >= self.num_domains:
                    continue
                dm = mask & (d == dom_int)
                if not bool(dm.any()):
                    continue
                dmean = safe_l2_normalize(z_norm[dm].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                if int(self.domain_count[cls_int, dom_int].item()) <= 0:
                    self.domain_proto[cls_int, dom_int].copy_(dmean)
                else:
                    self.domain_proto[cls_int, dom_int].mul_(m).add_(dmean, alpha=1.0 - m)
                    self.domain_proto[cls_int, dom_int].copy_(
                        safe_l2_normalize(self.domain_proto[cls_int, dom_int].view(1, -1), dim=1).squeeze(0)
                    )
                self.domain_count[cls_int, dom_int] += int(dm.sum().item())


def zero_like_with_grad(ref: torch.Tensor) -> torch.Tensor:
    """Return scalar zero while preserving a computation graph when possible.

    Why this matters:
      If a loss term becomes NaN/Inf and we replace it with ref.new_tensor(0.0),
      the resulting scalar is detached. If all active terms in a rare batch are
      sanitized this way, the final loss has no grad_fn and backward() crashes.
      ref.float().sum() * 0.0 is numerically zero but still attached to the graph.
    """
    if torch.is_tensor(ref) and ref.requires_grad:
        # Preserve a graph edge without letting NaN/Inf in ref leak through
        # as NaN * 0. This was a direct source of non-finite "zeroed" losses.
        return torch.nan_to_num(ref.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    device = ref.device if torch.is_tensor(ref) else torch.device("cpu")
    dtype = ref.dtype if torch.is_tensor(ref) and ref.dtype.is_floating_point else torch.float32
    return torch.zeros((), device=device, dtype=dtype)


def finite_or_zero(t: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    """Sanitize a scalar loss term without detaching the final graph."""
    if t is None:
        return zero_like_with_grad(ref)
    if not torch.is_tensor(t):
        try:
            t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)
        except Exception:
            return zero_like_with_grad(ref)
    if not torch.isfinite(t.detach()).all():
        return zero_like_with_grad(ref)
    return t


def sanitize_loss(
    name: str,
    t: Optional[torch.Tensor],
    ref: torch.Tensor,
    warn_counts: Optional[Dict[str, int]] = None,
    max_warn: int = 3,
) -> torch.Tensor:
    """Return a finite scalar loss, recording local sanitization events."""
    bad = t is None or (torch.is_tensor(t) and not torch.isfinite(t.detach()).all())
    out = finite_or_zero(t, ref)
    if bad and warn_counts is not None:
        warn_counts[name] = warn_counts.get(name, 0) + 1
        if warn_counts[name] <= int(max_warn):
            print(f"[WARN][LOSS] {name} non-finite; term set to zero (count={warn_counts[name]})", flush=True)
    return out


def cosine_distance_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (1.0 - safe_cosine_similarity(a, b, dim=1, eps=max(float(eps), 1e-6))).clamp(0.0, 2.0)


def cosine_consistency_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, float]:
    dist = cosine_distance_per_sample(a, b, eps=eps)
    cos = (1.0 - dist).mean().item()
    return dist.mean(), float(cos)


def one_way_kl_from_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    T = float(max(1e-6, temperature))
    student = torch.nan_to_num(student_logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    teacher = torch.nan_to_num(teacher_logits.float().detach(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    log_p_s = F.log_softmax(student / T, dim=1)
    p_t = F.softmax(teacher / T, dim=1).clamp_min(1e-8)
    p_t = p_t / p_t.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def masked_pseudo_label_ce_loss(
    logits: torch.Tensor,
    pseudo_y: torch.Tensor,
    mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, float]:
    """CE over accepted pseudo labels only; empty masks return graph-safe zero."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    y = pseudo_y.view(-1).long().to(logits.device)
    m = mask.view(-1).bool().to(logits.device)
    if y.numel() != logits.size(0) or m.numel() != logits.size(0):
        raise ValueError("pseudo_y and mask must have one value per sample")
    if not bool(m.any()):
        return zero_like_with_grad(logits), 0.0
    loss = F.cross_entropy(
        logits[m].float(),
        y[m],
        reduction="mean",
        label_smoothing=float(label_smoothing),
    )
    return loss, float(m.float().mean().detach().item())


def prototype_agreement_pull_loss(
    features: torch.Tensor,
    pseudo_y: torch.Tensor,
    class_prototypes: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Pull accepted source-unlabeled features toward class prototypes."""
    if features.ndim != 2 or class_prototypes.ndim != 2:
        raise ValueError("features and class_prototypes must both be rank-2")
    if features.size(1) != class_prototypes.size(1):
        raise ValueError("features and class_prototypes must have matching feature dimensions")
    y = pseudo_y.view(-1).long().to(features.device)
    m = mask.view(-1).bool().to(features.device)
    valid = m & (y >= 0) & (y < class_prototypes.size(0))
    if not bool(valid.any()):
        return zero_like_with_grad(features), float("nan")
    feat = safe_l2_normalize(features[valid].float(), dim=1)
    proto = safe_l2_normalize(class_prototypes[y[valid]].float(), dim=1)
    cos = (feat * proto.detach()).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean(), float(cos.detach().mean().item())


def smooth_strength_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = torch.nan_to_num(pred.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    target_f = torch.nan_to_num(target.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return F.smooth_l1_loss(pred_f, target_f)


def compute_core_losses(
    out_main: Dict[str, Any],
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    domain_stats: Dict[str, Any],
    domain_gates: Dict[str, bool],
    ce_tx,
    ce_dom,
    label_smoothing: float = 0.0,
    group_top_frac: float = 0.35,
    group_min_domains: int = 2,
    group_ce_mode: str = "hard",
    groupdro_state: Optional[SmoothGroupDROState] = None,
    groupdro_tau: float = 0.5,
    groupdro_cap: float = 0.65,
    groupdro_num_days: int = 4,
) -> Dict[str, Any]:
    tx_logits = out_main["tx_logits"]
    dom_logits = out_main["dom_logits"]
    adv_dom_logits = out_main["adv_dom_logits"]
    z_id = out_main["z_id"]
    z_dom = out_main["z_dom"]

    loss_cls = ce_tx(tx_logits.float(), y)
    loss_dom = z_id.new_tensor(0.0)
    loss_adv = z_id.new_tensor(0.0)
    loss_cons = z_id.new_tensor(0.0)
    loss_group_ce = z_id.new_tensor(0.0)
    cons_cos = float("nan")
    group_ce_hard = float("nan")
    dom_acc = float("nan")

    valid = domain_stats.get("valid", None)
    if d is not None and valid is not None and bool(valid.any()):
        d_valid = d[valid].long()
        if domain_gates.get("dom", False):
            loss_dom = ce_dom(dom_logits[valid].float(), d_valid)
            dom_acc = accuracy_from_logits(dom_logits[valid], d_valid)
        if domain_gates.get("adv", False):
            loss_adv = ce_dom(adv_dom_logits[valid].float(), d_valid)
        if domain_gates.get("cons", False):
            loss_cons, cons_cos = same_tx_cross_domain_consistency(z_id[valid], y[valid], d_valid)
        if domain_gates.get("group_ce", False):
            loss_group_ce, group_ce_hard = groupdro_or_hard_domain_ce_loss(
                tx_logits,
                y,
                d,
                groupdro_state,
                mode=str(group_ce_mode),
                label_smoothing=float(label_smoothing),
                top_frac=float(group_top_frac),
                min_domains=int(group_min_domains),
                tau=float(groupdro_tau),
                cap=float(groupdro_cap),
                rx_day_num_days=int(groupdro_num_days),
            )

    return {
        "loss_cls": loss_cls,
        "loss_dom": loss_dom,
        "loss_adv": loss_adv,
        "loss_cons": loss_cons,
        "loss_group_ce": loss_group_ce,
        "loss_orth": covariance_orth_loss(z_id, z_dom),
        "cons_cos": cons_cos,
        "group_ce_hard": group_ce_hard,
        "dom_acc": dom_acc,
    }


def compute_aux_losses(
    out_dac: Optional[Dict[str, Any]],
    out_pa: Optional[Dict[str, Any]],
    anchor: Dict[str, Any],
    y: torch.Tensor,
    s_dac: torch.Tensor,
    s_pa: torch.Tensor,
    need_dac_aux: bool,
    need_pa_aux: bool,
    cur_w: Dict[str, float],
    args,
    ce_tx,
    ref: torch.Tensor,
) -> Dict[str, Any]:
    clean_joint = get_nested_tensor(anchor, "id_feat_joint", "aux_id", "feat_joint")
    clean_dac = get_nested_tensor(anchor, "id_feat_dac", "aux_id", "feat_dac")
    clean_pa = get_nested_tensor(anchor, "id_feat_pa", "aux_id", "feat_pa")
    clean_logits = anchor["tx_logits"]

    zeros_b = torch.zeros(y.size(0), device=ref.device, dtype=ref.dtype)
    out = {
        "loss_cls_pa": ref.new_tensor(0.0),
        "loss_cls_dac": ref.new_tensor(0.0),
        "loss_pa_joint_inv": ref.new_tensor(0.0),
        "loss_pa_kl": ref.new_tensor(0.0),
        "loss_dac_reg": ref.new_tensor(0.0),
        "loss_pa_reg": ref.new_tensor(0.0),
        "shift_dac_on_dac": zeros_b,
        "shift_dac_on_pa": zeros_b,
        "shift_pa_on_pa": zeros_b,
        "shift_pa_on_dac": zeros_b,
        "cos_joint_pa": float("nan"),
        "cos_imp_pa": float("nan"),
    }

    pa_dac = clean_dac
    if need_pa_aux and out_pa is not None:
        pa_joint = get_nested_tensor(out_pa, "id_feat_joint", "aux_id", "feat_joint")
        pa_dac = get_nested_tensor(out_pa, "id_feat_dac", "aux_id", "feat_dac")
        pa_pa = get_nested_tensor(out_pa, "id_feat_pa", "aux_id", "feat_pa")
        pa_pred_pa = get_nested_tensor(out_pa, "id_pa_pred", "aux_id", "pa_pred")
        out["loss_cls_pa"] = ce_tx(out_pa["tx_logits"].float(), y)
        out["loss_pa_joint_inv"], out["cos_joint_pa"] = cosine_consistency_loss(pa_joint, clean_joint)
        out["loss_pa_kl"] = one_way_kl_from_teacher(out_pa["tx_logits"], clean_logits, temperature=float(args.robust_temp))
        out["shift_pa_on_pa"] = cosine_distance_per_sample(clean_pa, pa_pa)
        out["loss_pa_reg"] = smooth_strength_loss(pa_pred_pa, s_pa)

    if need_dac_aux and out_dac is not None:
        dac_dac = get_nested_tensor(out_dac, "id_feat_dac", "aux_id", "feat_dac")
        dac_pred_dac = get_nested_tensor(out_dac, "id_dac_pred", "aux_id", "dac_pred")
        out["loss_cls_dac"] = ce_tx(out_dac["tx_logits"].float(), y)
        out["shift_dac_on_dac"] = cosine_distance_per_sample(clean_dac, dac_dac)
        out["loss_dac_reg"] = smooth_strength_loss(dac_pred_dac, s_dac)

    if need_pa_aux and need_dac_aux and out_pa is not None and out_dac is not None:
        dac_pa = get_nested_tensor(out_dac, "id_feat_pa", "aux_id", "feat_pa")
        out["shift_pa_on_dac"] = cosine_distance_per_sample(clean_pa, dac_pa)
        out["shift_dac_on_pa"] = cosine_distance_per_sample(clean_dac, pa_dac)

    return out

