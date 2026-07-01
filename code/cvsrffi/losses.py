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


def _safe_angle_from_cos(cos: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6))


def _scalar_metric(value: torch.Tensor) -> float:
    if not torch.is_tensor(value):
        return float(value)
    if value.numel() == 0:
        return float("nan")
    detached = value.detach()
    if not torch.isfinite(detached).all():
        return float("nan")
    return float(detached.float().mean().item())


def _robust_three_sigma_radius_from_angles(angles: torch.Tensor, *, fallback_rad: float) -> torch.Tensor:
    """Detached angular radius used as a source-only class-tail boundary."""
    if angles.numel() <= 1:
        return angles.new_tensor(max(0.0, float(fallback_rad)))
    angles_det = angles.detach()
    median = torch.quantile(angles_det, 0.50)
    mad = torch.quantile((angles_det - median).abs(), 0.50)
    q25 = torch.quantile(angles_det, 0.25)
    q75 = torch.quantile(angles_det, 0.75)
    iqr = q75 - q25
    robust_sigma = 1.4826 * mad
    if float(robust_sigma.item()) <= 0.0:
        robust_sigma = 0.7413 * iqr
    if float(robust_sigma.item()) <= 0.0 and angles_det.numel() > 1:
        robust_sigma = angles_det.std(unbiased=False)
    if float(robust_sigma.item()) <= 0.0:
        return torch.maximum(angles_det.max(), angles.new_tensor(max(0.0, float(fallback_rad))))
    return torch.minimum(angles_det.max(), median + 3.0 * robust_sigma)


def open_world_feature_space_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    *,
    radius_rad: float = math.radians(12.0),
    inter_margin_rad: float = math.radians(55.0),
    sample_margin_rad: float = math.radians(5.0),
    domain_align_weight: float = 0.0,
    min_classes: int = 2,
    min_samples_per_class: int = 1,
    tail_mode: str = "none",
    tail_weight: float = 0.0,
    cvar_alpha: float = 0.95,
    vacuum_weight: float = 0.0,
    vacuum_width_rad: float = math.radians(4.0),
    vacuum_hard_k: int = 2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Batch-level angular geometry loss for open-world identity embeddings.

    The loss is intentionally stateless: it optimizes the selected identity
    feature used by the existing prototype/SupCon path without creating a
    second training-time memory bank.
    """
    default_metrics = {
        "compact": 0.0,
        "inter": 0.0,
        "sample_margin": 0.0,
        "domain_align": 0.0,
        "active_classes": 0.0,
        "pos_angle_deg": float("nan"),
        "min_inter_angle_deg": float("nan"),
        "pos_angle_p50_deg": float("nan"),
        "pos_angle_p95_deg": float("nan"),
        "pos_angle_p99_deg": float("nan"),
        "pos_angle_max_deg": float("nan"),
        "tail_loss": 0.0,
        "tail_cvar_deg": float("nan"),
        "tail_frac_gt_3sigma": 0.0,
        "tail_radius_3sigma_deg": float("nan"),
        "vacuum_loss": 0.0,
        "vacuum_violation_rate": 0.0,
        "vacuum_min_neg_angle_deg": float("nan"),
        "vacuum_margin_deg": float("nan"),
        "vacuum_boundary_deg": float("nan"),
    }
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, default_metrics
    if z.dim() != 2:
        raise ValueError(f"open_world_feature_space_loss expects 2D features, got shape={tuple(z.shape)}")
    if y is None or not torch.is_tensor(y):
        raise ValueError("open_world_feature_space_loss expects tensor labels")
    labels = y.view(-1).long()
    if labels.numel() != z.size(0):
        raise ValueError(f"label count {labels.numel()} does not match feature batch {z.size(0)}")

    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    valid_y = labels >= 0
    if not bool(valid_y.any()):
        return zero_like_with_grad(z), default_metrics

    centers = []
    class_ids = []
    sample_mask = torch.zeros(labels.shape, device=labels.device, dtype=torch.bool)
    min_count = max(1, int(min_samples_per_class))
    for cls in torch.unique(labels[valid_y]):
        cls_mask = valid_y & labels.eq(cls)
        if int(cls_mask.sum().item()) < min_count:
            continue
        center = safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centers.append(center)
        class_ids.append(cls)
        sample_mask = sample_mask | cls_mask

    active_classes = len(centers)
    default_metrics["active_classes"] = float(active_classes)
    if active_classes < max(1, int(min_classes)):
        return zero_like_with_grad(z), default_metrics

    proto = torch.stack(centers, dim=0)
    center_labels = torch.stack(class_ids, dim=0).to(device=labels.device)
    sample_z = z_norm[sample_mask]
    sample_labels = labels[sample_mask]
    sample_cos = (sample_z @ proto.t()).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    own_center = sample_labels.view(-1, 1).eq(center_labels.view(1, -1))
    pos_cos = sample_cos[own_center]

    cos_radius = math.cos(max(0.0, float(radius_rad)))
    compact = F.relu(float(cos_radius) - pos_cos).pow(2).mean()
    sample_margin = z.new_tensor(0.0)
    if active_classes > 1:
        neg_cos = sample_cos.masked_fill(own_center, -float("inf")).max(dim=1).values
        cos_gap = 1.0 - math.cos(max(0.0, float(sample_margin_rad)))
        sample_margin = F.relu(neg_cos + float(cos_gap) - pos_cos).pow(2).mean()

    inter = z.new_tensor(0.0)
    min_inter_angle = z.new_tensor(float("nan"))
    if active_classes > 1:
        proto_cos = (proto @ proto.t()).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        tri = torch.triu_indices(active_classes, active_classes, offset=1, device=proto.device)
        pair_cos = proto_cos[tri[0], tri[1]]
        cos_inter_margin = math.cos(max(0.0, float(inter_margin_rad)))
        inter = F.relu(pair_cos - float(cos_inter_margin)).pow(2).mean()
        min_inter_angle = _safe_angle_from_cos(pair_cos.detach()).min()

    domain_align = z.new_tensor(0.0)
    domain_align_angle = z.new_tensor(0.0)
    if d is not None and torch.is_tensor(d):
        domains = d.view(-1).long()
        if domains.numel() != z.size(0):
            raise ValueError(f"domain count {domains.numel()} does not match feature batch {z.size(0)}")
        domain_terms = []
        domain_angle_terms = []
        for cls, center in zip(center_labels, proto):
            cls_mask = labels.eq(cls) & (domains >= 0)
            for dom in torch.unique(domains[cls_mask]):
                dom_mask = cls_mask & domains.eq(dom)
                if not bool(dom_mask.any()):
                    continue
                dom_center = safe_l2_normalize(z_norm[dom_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                dom_cos = (dom_center * center.detach()).sum().clamp(-1.0 + 1e-4, 1.0 - 1e-4)
                domain_terms.append(1.0 - dom_cos)
                domain_angle_terms.append(_safe_angle_from_cos(dom_cos.detach()))
        if domain_terms:
            domain_align = torch.stack(domain_terms).mean()
            domain_align_angle = torch.stack(domain_angle_terms).mean()

    pos_angles = _safe_angle_from_cos(pos_cos.detach())
    pos_angles_train = _safe_angle_from_cos(pos_cos)
    tail_mode_l = str(tail_mode or "none").lower().strip()
    tail_loss = z.new_tensor(0.0)
    tail_cvar = z.new_tensor(float("nan"))
    tail_frac = 0.0
    tail_radius_mean = z.new_tensor(float("nan"))
    if tail_mode_l not in ("none", "off", "false", "0") and float(tail_weight) > 0.0:
        class_tail_losses = []
        class_cvar = []
        class_tail_frac = []
        class_radius = []
        alpha = max(0.0, min(0.999, float(cvar_alpha)))
        for cls in center_labels:
            cls_mask = sample_labels.eq(cls)
            if int(cls_mask.sum().item()) <= 1:
                continue
            angles_c = pos_angles_train[cls_mask]
            angles_c_det = angles_c.detach()
            median = torch.quantile(angles_c_det, 0.50)
            mad = torch.quantile((angles_c_det - median).abs(), 0.50)
            q25 = torch.quantile(angles_c_det, 0.25)
            q75 = torch.quantile(angles_c_det, 0.75)
            iqr = q75 - q25
            robust_sigma = 1.4826 * mad
            if float(robust_sigma.item()) <= 0.0:
                robust_sigma = 0.7413 * iqr
            if float(robust_sigma.item()) <= 0.0 and angles_c_det.numel() > 1:
                robust_sigma = angles_c_det.std(unbiased=False)
            radius_3s = torch.minimum(angles_c_det.max(), median + 3.0 * robust_sigma)
            overflow = torch.relu(angles_c - radius_3s)
            class_tail_losses.append(overflow.pow(2).mean())
            k = max(1, int(math.ceil(float(angles_c.numel()) * (1.0 - alpha))))
            class_cvar.append(torch.topk(angles_c, k=k, largest=True).values.mean())
            class_tail_frac.append(float((angles_c_det > radius_3s).float().mean().item()))
            class_radius.append(radius_3s)
        if class_tail_losses:
            tail_loss = torch.stack(class_tail_losses).mean()
            tail_cvar = torch.stack(class_cvar).mean()
            tail_frac = float(np.mean(class_tail_frac))
            tail_radius_mean = torch.stack(class_radius).mean()

    vacuum_loss = z.new_tensor(0.0)
    vacuum_violation_rate = 0.0
    vacuum_min_neg_angle = z.new_tensor(float("nan"))
    vacuum_margin_mean = z.new_tensor(float("nan"))
    vacuum_boundary_mean = z.new_tensor(float("nan"))
    if active_classes > 1 and float(vacuum_weight) > 0.0:
        class_radii = []
        for cls in center_labels:
            cls_mask = sample_labels.eq(cls)
            angles_c = pos_angles_train[cls_mask]
            class_radii.append(_robust_three_sigma_radius_from_angles(angles_c, fallback_rad=radius_rad))
        radius_vec = torch.stack(class_radii, dim=0).to(device=sample_cos.device, dtype=sample_cos.dtype)
        all_angles = _safe_angle_from_cos(sample_cos)
        boundary = (radius_vec + max(0.0, float(vacuum_width_rad))).view(1, -1)
        foreign_mask = ~own_center
        violation = F.relu(boundary - all_angles).pow(2).masked_fill(~foreign_mask, 0.0)
        hard_k = max(1, min(int(vacuum_hard_k), active_classes - 1))
        vacuum_loss = violation.topk(k=hard_k, dim=1, largest=True).values.mean()
        with torch.no_grad():
            margins = (all_angles.detach() - boundary.detach())[foreign_mask]
            foreign_angles = all_angles.detach()[foreign_mask]
            foreign_boundary = boundary.detach().expand_as(all_angles)[foreign_mask]
            if margins.numel():
                vacuum_violation_rate = float((margins < 0.0).float().mean().item())
                vacuum_min_neg_angle = foreign_angles.min()
                vacuum_margin_mean = margins.mean()
                vacuum_boundary_mean = foreign_boundary.mean()

    loss = (
        compact
        + inter
        + sample_margin
        + max(0.0, float(domain_align_weight)) * domain_align
        + max(0.0, float(tail_weight)) * tail_loss
        + max(0.0, float(vacuum_weight)) * vacuum_loss
    )
    q50 = torch.quantile(pos_angles, 0.50) if pos_angles.numel() else z.new_tensor(float("nan"))
    q95 = torch.quantile(pos_angles, 0.95) if pos_angles.numel() else z.new_tensor(float("nan"))
    q99 = torch.quantile(pos_angles, 0.99) if pos_angles.numel() else z.new_tensor(float("nan"))
    qmax = torch.quantile(pos_angles, 1.0) if pos_angles.numel() else z.new_tensor(float("nan"))
    metrics = {
        "compact": _scalar_metric(compact),
        "inter": _scalar_metric(inter),
        "sample_margin": _scalar_metric(sample_margin),
        "domain_align": _scalar_metric(domain_align_angle),
        "active_classes": float(active_classes),
        "pos_angle_deg": math.degrees(_scalar_metric(pos_angles)),
        "min_inter_angle_deg": math.degrees(_scalar_metric(min_inter_angle)),
        "pos_angle_p50_deg": math.degrees(_scalar_metric(q50)),
        "pos_angle_p95_deg": math.degrees(_scalar_metric(q95)),
        "pos_angle_p99_deg": math.degrees(_scalar_metric(q99)),
        "pos_angle_max_deg": math.degrees(_scalar_metric(qmax)),
        "tail_loss": _scalar_metric(tail_loss),
        "tail_cvar_deg": math.degrees(_scalar_metric(tail_cvar)),
        "tail_frac_gt_3sigma": float(tail_frac),
        "tail_radius_3sigma_deg": math.degrees(_scalar_metric(tail_radius_mean)),
        "vacuum_loss": _scalar_metric(vacuum_loss),
        "vacuum_violation_rate": float(vacuum_violation_rate),
        "vacuum_min_neg_angle_deg": math.degrees(_scalar_metric(vacuum_min_neg_angle)),
        "vacuum_margin_deg": math.degrees(_scalar_metric(vacuum_margin_mean)),
        "vacuum_boundary_deg": math.degrees(_scalar_metric(vacuum_boundary_mean)),
    }
    return loss, metrics


def zid_compactness_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    *,
    radius_rad: float = math.radians(40.0),
    cvar_alpha: float = 0.90,
    supcon_weight: float = 0.35,
    radius_weight: float = 0.35,
    cvar_weight: float = 0.30,
    temperature: float = 0.12,
    min_classes: int = 2,
    min_samples_per_class: int = 1,
    domain_aware: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Unified z_id compactness objective: SupCon + angular radius + tail CVaR."""
    default_metrics = {
        "supcon": 0.0,
        "radius": 0.0,
        "tail_cvar": 0.0,
        "active_classes": 0.0,
        "pos_angle_p50_deg": float("nan"),
        "pos_angle_p95_deg": float("nan"),
        "pos_angle_p99_deg": float("nan"),
        "tail_cvar_deg": float("nan"),
    }
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, default_metrics
    if z.dim() != 2:
        raise ValueError(f"zid_compactness_loss expects 2D features, got shape={tuple(z.shape)}")
    labels = y.view(-1).long()
    if labels.numel() != z.size(0):
        raise ValueError(f"label count {labels.numel()} does not match feature batch {z.size(0)}")

    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    valid = labels >= 0
    centers = []
    class_ids = []
    sample_mask = torch.zeros(labels.shape, device=labels.device, dtype=torch.bool)
    min_count = max(1, int(min_samples_per_class))
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls)
        if int(cls_mask.sum().item()) < min_count:
            continue
        centers.append(safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        class_ids.append(cls)
        sample_mask = sample_mask | cls_mask

    active_classes = len(centers)
    default_metrics["active_classes"] = float(active_classes)
    if active_classes < max(1, int(min_classes)):
        return zero_like_with_grad(z), default_metrics

    proto = torch.stack(centers, dim=0)
    center_labels = torch.stack(class_ids, dim=0).to(device=labels.device)
    sample_z = z_norm[sample_mask]
    sample_labels = labels[sample_mask]
    sample_cos = (sample_z @ proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    own_center = sample_labels.view(-1, 1).eq(center_labels.view(1, -1))
    pos_cos = sample_cos[own_center]
    pos_angles = _safe_angle_from_cos(pos_cos)
    pos_angles_det = pos_angles.detach()

    radius_loss = F.relu(pos_angles - max(0.0, float(radius_rad))).pow(2).mean()
    alpha = max(0.0, min(0.999, float(cvar_alpha)))
    class_cvars = []
    for cls in center_labels:
        angles_c = pos_angles[sample_labels.eq(cls)]
        if angles_c.numel() == 0:
            continue
        k = max(1, int(math.ceil(float(angles_c.numel()) * (1.0 - alpha))))
        class_cvars.append(torch.topk(angles_c, k=k, largest=True).values.mean())
    tail_cvar = torch.stack(class_cvars).mean() if class_cvars else z.new_tensor(0.0)

    if float(supcon_weight) > 0.0:
        supcon = domain_aware_supcon_loss(z_norm, labels, d if bool(domain_aware) else None, temperature=float(temperature))
        if (not bool(domain_aware)) and float(supcon.detach().abs().item()) == 0.0:
            supcon = _plain_supcon_loss(z_norm, labels, temperature=float(temperature))
    else:
        supcon = z.new_tensor(0.0)

    loss = (
        max(0.0, float(supcon_weight)) * supcon
        + max(0.0, float(radius_weight)) * radius_loss
        + max(0.0, float(cvar_weight)) * tail_cvar
    )
    q50 = torch.quantile(pos_angles_det, 0.50) if pos_angles_det.numel() else z.new_tensor(float("nan"))
    q95 = torch.quantile(pos_angles_det, 0.95) if pos_angles_det.numel() else z.new_tensor(float("nan"))
    q99 = torch.quantile(pos_angles_det, 0.99) if pos_angles_det.numel() else z.new_tensor(float("nan"))
    metrics = {
        "supcon": _scalar_metric(supcon),
        "radius": _scalar_metric(radius_loss),
        "tail_cvar": _scalar_metric(tail_cvar),
        "active_classes": float(active_classes),
        "pos_angle_p50_deg": math.degrees(_scalar_metric(q50)),
        "pos_angle_p95_deg": math.degrees(_scalar_metric(q95)),
        "pos_angle_p99_deg": math.degrees(_scalar_metric(q99)),
        "tail_cvar_deg": math.degrees(_scalar_metric(tail_cvar.detach())),
    }
    return loss, metrics


def _plain_supcon_loss(z: torch.Tensor, y: torch.Tensor, *, temperature: float = 0.12) -> torch.Tensor:
    if z is None or z.size(0) <= 1:
        return z.new_tensor(0.0)
    labels = y.view(-1).long()
    valid = labels >= 0
    if int(valid.sum().item()) <= 1:
        return z.new_tensor(0.0)
    z = safe_l2_normalize(z, dim=1)
    logits = (z @ z.t()) / max(1e-4, float(temperature))
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    valid_pair = valid.view(-1, 1) & valid.view(1, -1) & (~eye)
    pos = labels.view(-1, 1).eq(labels.view(1, -1)) & valid_pair
    has_pos = pos.sum(dim=1) > 0
    if not bool(has_pos.any()):
        return z.new_tensor(0.0)
    exp_logits = torch.exp(logits) * valid_pair.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_log_prob = (log_prob * pos.float()).sum(dim=1) / pos.float().sum(dim=1).clamp_min(1.0)
    return -pos_log_prob[has_pos].mean()


def proxy_unknown_energy_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    holdout_label: Optional[int] = None,
    virtual_count: int = 16,
    energy_margin: float = 1.0,
    placeholder_weight: float = 0.5,
    virtual_detach: bool = True,
    vacuum_weight: float = 0.0,
    vacuum_width_rad: float = math.radians(4.0),
    vacuum_hard_k: int = 2,
    vacuum_radius_rad: float = math.radians(40.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Source-only proxy unknown loss using leave-one-TX-out and feature outliers."""
    default_metrics = {
        "active": 0.0,
        "known_count": 0.0,
        "proxy_unknown_count": 0.0,
        "virtual_count": 0.0,
        "energy_known": float("nan"),
        "energy_proxy": float("nan"),
        "energy_virtual": float("nan"),
        "energy_margin": float("nan"),
        "proxy_unknown_auc": float("nan"),
        "virtual_accept_rate": float("nan"),
        "vacuum_loss": 0.0,
        "vacuum_violation_rate": 0.0,
        "vacuum_margin_deg": float("nan"),
        "vacuum_min_angle_deg": float("nan"),
    }
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, default_metrics
    if z.dim() != 2:
        raise ValueError(f"proxy_unknown_energy_loss expects 2D features, got shape={tuple(z.shape)}")
    labels = y.view(-1).long()
    if labels.numel() != z.size(0):
        raise ValueError(f"label count {labels.numel()} does not match feature batch {z.size(0)}")
    valid_labels = torch.unique(labels[labels >= 0])
    if valid_labels.numel() < 2:
        return zero_like_with_grad(z), default_metrics
    if holdout_label is None:
        holdout = int(valid_labels[-1].detach().item())
    else:
        holdout = int(holdout_label)
    proxy_mask = labels.eq(holdout)
    known_mask = (labels >= 0) & (~proxy_mask)
    if not bool(proxy_mask.any()) or not bool(known_mask.any()):
        return zero_like_with_grad(z), default_metrics

    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    known_labels = torch.unique(labels[known_mask])
    centers = []
    class_radii = []
    for cls in known_labels:
        cls_mask = known_mask & labels.eq(cls)
        if bool(cls_mask.any()):
            center = safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            centers.append(center)
            own_cos = (z_norm[cls_mask] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
            class_radii.append(
                _robust_three_sigma_radius_from_angles(
                    _safe_angle_from_cos(own_cos),
                    fallback_rad=vacuum_radius_rad,
                )
            )
    if len(centers) <= 0:
        return zero_like_with_grad(z), default_metrics
    proto = torch.stack(centers, dim=0)
    radius_vec = torch.stack(class_radii, dim=0).to(device=proto.device, dtype=proto.dtype)

    def energy(feat: torch.Tensor) -> torch.Tensor:
        scores = safe_l2_normalize(feat, dim=1) @ proto.t()
        return -torch.logsumexp(scores, dim=1)

    e_known = energy(z_norm[known_mask])
    e_proxy = energy(z_norm[proxy_mask])
    virtual = _make_virtual_outliers(z_norm[known_mask], proto, count=max(0, int(virtual_count)))
    if bool(virtual_detach):
        virtual = virtual.detach()
    e_virtual = energy(virtual) if virtual.numel() else z.new_zeros((0,))

    proxy_target = torch.cat([e_proxy, e_virtual], dim=0)
    margin_loss = F.relu(float(energy_margin) - (proxy_target.mean() - e_known.mean())).pow(2)
    logits_known = -e_known
    logits_unknown = torch.cat([e_proxy, e_virtual], dim=0)
    placeholder = (
        F.softplus(-logits_known).mean()
        + F.softplus(-logits_unknown).mean()
    ) if logits_unknown.numel() else z.new_tensor(0.0)

    vacuum_loss = z.new_tensor(0.0)
    vacuum_violation_rate = 0.0
    vacuum_margin_mean = z.new_tensor(float("nan"))
    vacuum_min_angle = z.new_tensor(float("nan"))
    if float(vacuum_weight) > 0.0:
        unknown_feat = torch.cat([z_norm[proxy_mask], virtual], dim=0)
        if unknown_feat.numel():
            unknown_cos = (safe_l2_normalize(unknown_feat, dim=1) @ proto.t()).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
            unknown_angles = _safe_angle_from_cos(unknown_cos)
            boundary = (radius_vec + max(0.0, float(vacuum_width_rad))).view(1, -1)
            violation = F.relu(boundary - unknown_angles).pow(2)
            hard_k = max(1, min(int(vacuum_hard_k), proto.size(0)))
            vacuum_loss = violation.topk(k=hard_k, dim=1, largest=True).values.mean()
            with torch.no_grad():
                margins = unknown_angles.detach() - boundary.detach()
                if margins.numel():
                    nearest_margin = margins.min(dim=1).values
                    vacuum_violation_rate = float((nearest_margin < 0.0).float().mean().item())
                    vacuum_margin_mean = nearest_margin.mean()
                    vacuum_min_angle = unknown_angles.detach().min()

    loss = (
        margin_loss
        + max(0.0, float(placeholder_weight)) * placeholder
        + max(0.0, float(vacuum_weight)) * vacuum_loss
    )

    with torch.no_grad():
        known_scores = e_known.detach()
        unknown_scores = torch.cat([e_proxy.detach(), e_virtual.detach()], dim=0)
        auc = _binary_auc(known_scores, unknown_scores)
        known_accept_threshold = torch.quantile(known_scores, 0.95) if known_scores.numel() else z.new_tensor(float("nan"))
        virtual_accept = (
            float((e_virtual.detach() <= known_accept_threshold).float().mean().item())
            if e_virtual.numel() and torch.isfinite(known_accept_threshold)
            else float("nan")
        )
    metrics = {
        "active": 1.0,
        "known_count": float(int(known_mask.sum().item())),
        "proxy_unknown_count": float(int(proxy_mask.sum().item())),
        "virtual_count": float(int(e_virtual.numel())),
        "energy_known": _scalar_metric(e_known),
        "energy_proxy": _scalar_metric(e_proxy),
        "energy_virtual": _scalar_metric(e_virtual) if e_virtual.numel() else float("nan"),
        "energy_margin": _scalar_metric(e_proxy.mean() - e_known.mean()),
        "proxy_unknown_auc": float(auc),
        "virtual_accept_rate": float(virtual_accept),
        "vacuum_loss": _scalar_metric(vacuum_loss),
        "vacuum_violation_rate": float(vacuum_violation_rate),
        "vacuum_margin_deg": math.degrees(_scalar_metric(vacuum_margin_mean)),
        "vacuum_min_angle_deg": math.degrees(_scalar_metric(vacuum_min_angle)),
    }
    return loss, metrics


def _make_virtual_outliers(z_known: torch.Tensor, proto: torch.Tensor, *, count: int) -> torch.Tensor:
    if count <= 0 or z_known.numel() == 0:
        return z_known.new_zeros((0, z_known.size(1)))
    base = z_known[:count]
    if base.size(0) < count:
        reps = int(math.ceil(count / max(1, base.size(0))))
        base = base.repeat(reps, 1)[:count]
    nearest = proto[(base @ proto.t()).argmax(dim=1)]
    out = safe_l2_normalize(base + 0.75 * (base - nearest), dim=1)
    if proto.size(0) >= 2:
        idx = torch.arange(count, device=proto.device)
        p1 = proto[idx % proto.size(0)]
        p2 = proto[(idx + 1) % proto.size(0)]
        interp = safe_l2_normalize(0.5 * (p1 + p2), dim=1)
        out = safe_l2_normalize(0.5 * out + 0.5 * interp, dim=1)
    return out


def _binary_auc(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    if known_scores.numel() == 0 or unknown_scores.numel() == 0:
        return float("nan")
    return float((unknown_scores.view(-1, 1) > known_scores.view(1, -1)).float().mean().item())


def source_episode_three_sigma_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    *,
    min_domains: int = 2,
    min_samples_per_class_domain: int = 1,
    radius_cap_rad: float = math.radians(30.0),
    min_sigma_rad: float = math.radians(3.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Source-only leave-domain angular shell objective.

    This is a Phase1 proxy objective: all support/query roles are drawn from
    source domains in the current batch. It must not be interpreted as Stage2
    target-new or unknown rejection evidence.
    """
    default_metrics = {
        "source_episode_loss": 0.0,
        "source_episode_overflow_rate": 0.0,
        "source_episode_radius_3sigma_deg": float("nan"),
        "source_episode_val_angle_deg": float("nan"),
        "source_episode_classes": 0.0,
        "source_episode_domains": 0.0,
    }
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, default_metrics
    if d is None or not torch.is_tensor(d):
        return zero_like_with_grad(z), default_metrics
    labels = y.view(-1).long()
    domains = d.view(-1).long()
    if labels.numel() != z.size(0) or domains.numel() != z.size(0):
        raise ValueError("source_episode_three_sigma_loss expects one label/domain per feature")

    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    valid = (labels >= 0) & (domains >= 0)
    losses = []
    overflow_rates = []
    radii = []
    val_angles_all = []
    used_classes = set()
    used_domains = set()
    min_cell = max(1, int(min_samples_per_class_domain))
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls)
        doms = [dom for dom in torch.unique(domains[cls_mask]) if int((cls_mask & domains.eq(dom)).sum().item()) >= min_cell]
        if len(doms) < max(2, int(min_domains)):
            continue
        for val_dom in doms:
            train_mask = cls_mask & domains.ne(val_dom)
            val_mask = cls_mask & domains.eq(val_dom)
            if int(train_mask.sum().item()) < min_cell or int(val_mask.sum().item()) < min_cell:
                continue
            center = safe_l2_normalize(z_norm[train_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            train_cos = (z_norm[train_mask] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            train_angles = _safe_angle_from_cos(train_cos.detach())
            median = torch.quantile(train_angles, 0.50)
            mad = torch.quantile((train_angles - median).abs(), 0.50)
            q25 = torch.quantile(train_angles, 0.25)
            q75 = torch.quantile(train_angles, 0.75)
            robust_sigma = 1.4826 * mad
            if float(robust_sigma.item()) <= 0.0:
                robust_sigma = 0.7413 * (q75 - q25)
            if float(robust_sigma.item()) <= 0.0 and train_angles.numel() > 1:
                robust_sigma = train_angles.std(unbiased=False)
            robust_sigma = torch.maximum(robust_sigma, train_angles.new_tensor(max(0.0, float(min_sigma_rad))))
            radius = torch.minimum(
                train_angles.new_tensor(float(radius_cap_rad)),
                torch.minimum(
                    train_angles.max() + 3.0 * robust_sigma,
                    median + 3.0 * robust_sigma,
                ),
            )
            val_cos = (z_norm[val_mask] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            val_angles = _safe_angle_from_cos(val_cos)
            overflow = torch.relu(val_angles - radius)
            losses.append(overflow.pow(2).mean())
            overflow_rates.append(float((val_angles.detach() > radius).float().mean().item()))
            radii.append(radius.detach())
            val_angles_all.append(val_angles.detach().mean())
            used_classes.add(int(cls.item()))
            used_domains.add(int(val_dom.item()))
    if not losses:
        return zero_like_with_grad(z), default_metrics
    loss = torch.stack(losses).mean()
    metrics = {
        "source_episode_loss": _scalar_metric(loss),
        "source_episode_overflow_rate": float(np.mean(overflow_rates)) if overflow_rates else 0.0,
        "source_episode_radius_3sigma_deg": math.degrees(_scalar_metric(torch.stack(radii))),
        "source_episode_val_angle_deg": math.degrees(_scalar_metric(torch.stack(val_angles_all))),
        "source_episode_classes": float(len(used_classes)),
        "source_episode_domains": float(len(used_domains)),
    }
    return loss, metrics


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


def energy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return -float(temperature) * torch.logsumexp(logits.float() / float(temperature), dim=-1)


def energy_in_out_loss(
    known_core_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    m_in: float = -8.0,
    m_out: float = -2.0,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ref = known_core_logits if torch.is_tensor(known_core_logits) else negative_logits
    terms = []
    metrics: Dict[str, float] = {
        "known_count": 0.0,
        "negative_count": 0.0,
        "energy_known": float("nan"),
        "energy_negative": float("nan"),
    }
    if torch.is_tensor(known_core_logits) and known_core_logits.numel() > 0:
        e_in = energy_from_logits(known_core_logits, temperature=temperature)
        terms.append(torch.relu(e_in - float(m_in)).mean())
        metrics["known_count"] = float(known_core_logits.size(0))
        metrics["energy_known"] = float(e_in.detach().mean().item())
    if torch.is_tensor(negative_logits) and negative_logits.numel() > 0:
        e_out = energy_from_logits(negative_logits, temperature=temperature)
        terms.append(torch.relu(float(m_out) - e_out).mean())
        metrics["negative_count"] = float(negative_logits.size(0))
        metrics["energy_negative"] = float(e_out.detach().mean().item())
    if not terms:
        return zero_like_with_grad(ref), metrics
    return finite_or_zero(torch.stack(terms).sum(), ref), metrics


def reject_negative_loss(
    negative_logits: torch.Tensor,
    reject_class_index: int = -1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if negative_logits is None or negative_logits.numel() == 0:
        ref = negative_logits if torch.is_tensor(negative_logits) else torch.tensor(0.0)
        return zero_like_with_grad(ref), {"negative_count": 0.0, "reject_acc": float("nan")}
    logits = negative_logits.float()
    idx = int(reject_class_index)
    if idx < 0:
        idx = logits.size(1) + idx
    target = torch.full((logits.size(0),), idx, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, target)
    pred = logits.argmax(dim=1)
    return loss, {"negative_count": float(logits.size(0)), "reject_acc": float((pred == target).float().mean().item())}


def negative_entropy_or_margin_loss(
    negative_logits: torch.Tensor,
    max_known_prob: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if negative_logits is None or negative_logits.numel() == 0:
        ref = negative_logits if torch.is_tensor(negative_logits) else torch.tensor(0.0)
        return zero_like_with_grad(ref), {"negative_count": 0.0, "neg_max_known_prob": float("nan")}
    probs = negative_logits.float().softmax(dim=-1)
    max_prob = probs.max(dim=-1).values
    loss = torch.relu(max_prob - float(max_known_prob)).mean()
    return loss, {"negative_count": float(negative_logits.size(0)), "neg_max_known_prob": float(max_prob.detach().mean().item())}


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

