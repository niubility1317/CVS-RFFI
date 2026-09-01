from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.eval import accuracy_from_logits
from cvsrffi.tensors import get_nested_tensor, safe_cosine_similarity, safe_l2_normalize


def crra_nuisance_huber_loss(
    prediction: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    *,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Regress only nuisance values carried by the same satellite view.

    Clean duplicates, missing metadata and invalid rows contribute a connected
    zero rather than a fabricated target. The helper is intentionally generic
    so it can consume the normalized `mixed_orbit` metadata tensor from the
    view augmenter without reading receiver truth.
    """

    if not torch.is_tensor(prediction):
        zero = torch.tensor(0.0)
        return zero, {"valid_count": 0.0, "field_count": 0.0}
    zero = prediction.sum() * 0.0
    if not torch.is_tensor(target) or not torch.is_tensor(valid_mask):
        return zero, {"valid_count": 0.0, "field_count": 0.0}
    if prediction.dim() != 2 or target.dim() != 2 or prediction.size(0) != target.size(0):
        raise ValueError("CRRA nuisance prediction and target must be aligned rank-2 tensors")
    valid = valid_mask.to(device=prediction.device).view(-1).bool()
    if valid.numel() != prediction.size(0) or not bool(valid.any()):
        return zero, {"valid_count": 0.0, "field_count": 0.0}
    if int(prediction.size(1)) != int(target.size(1)):
        raise ValueError(
            "CRRA nuisance prediction and target must have the same fixed field dimension"
        )
    field_count = int(target.size(1))
    if field_count <= 0:
        return zero, {"valid_count": 0.0, "field_count": 0.0}
    pred = torch.nan_to_num(prediction[valid].float(), nan=0.0, posinf=0.0, neginf=0.0)
    truth = torch.nan_to_num(
        target.to(device=prediction.device, dtype=torch.float32)[valid],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss = F.huber_loss(pred, truth, reduction="mean", delta=float(delta))
    return loss, {
        "valid_count": float(valid.sum().detach().item()),
        "field_count": float(field_count),
    }


def crra_satellite_shell_loss(
    clean_z: Optional[torch.Tensor],
    clean_y: Optional[torch.Tensor],
    sat_z: Optional[torch.Tensor],
    sat_y: Optional[torch.Tensor],
    *,
    shell_width_rad: float = math.radians(12.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Keep satellite identity features inside a detached clean-class shell."""

    if not torch.is_tensor(sat_z):
        ref = clean_z if torch.is_tensor(clean_z) else torch.tensor(0.0)
        return ref.sum() * 0.0, {"valid_count": 0.0, "active_classes": 0.0, "mean_excess_rad": 0.0}
    zero = sat_z.sum() * 0.0
    if not all(torch.is_tensor(value) for value in (clean_z, clean_y, sat_y)):
        return zero, {"valid_count": 0.0, "active_classes": 0.0, "mean_excess_rad": 0.0}
    if clean_z.dim() != 2 or sat_z.dim() != 2 or clean_z.size(1) != sat_z.size(1):
        raise ValueError("CRRA shell clean and satellite features must be aligned rank-2 tensors")
    clean_y = clean_y.to(device=clean_z.device).view(-1).long()
    sat_y = sat_y.to(device=sat_z.device).view(-1).long()
    if clean_z.size(0) != clean_y.numel() or sat_z.size(0) != sat_y.numel():
        raise ValueError("CRRA shell features and labels must have aligned batch dimensions")
    clean_unit = safe_l2_normalize(
        torch.nan_to_num(clean_z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1
    ).detach()
    sat_unit = safe_l2_normalize(
        torch.nan_to_num(sat_z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1
    )
    width = max(0.0, float(shell_width_rad))
    losses = []
    excess_values = []
    active_classes = 0
    for cls in torch.unique(sat_y):
        clean_mask = clean_y == cls.to(device=clean_y.device)
        sat_mask = sat_y == cls.to(device=sat_y.device)
        if not bool(clean_mask.any()) or not bool(sat_mask.any()):
            continue
        center = safe_l2_normalize(
            clean_unit[clean_mask].mean(dim=0, keepdim=True), dim=1
        ).squeeze(0).detach()
        clean_angles = torch.acos(torch.clamp(clean_unit[clean_mask] @ center, -1.0 + 1e-6, 1.0 - 1e-6))
        clean_radius = clean_angles.max().detach()
        sat_angles = torch.acos(torch.clamp(sat_unit[sat_mask] @ center, -1.0 + 1e-6, 1.0 - 1e-6))
        excess = torch.relu(sat_angles - (clean_radius + width))
        losses.append(excess.mean())
        excess_values.append(excess.detach())
        active_classes += 1
    if not losses:
        return zero, {"valid_count": 0.0, "active_classes": 0.0, "mean_excess_rad": 0.0}
    excess_all = torch.cat(excess_values)
    return torch.stack(losses).mean(), {
        "valid_count": float(sat_y.numel()),
        "active_classes": float(active_classes),
        "mean_excess_rad": float(excess_all.mean().item()),
    }


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


def _safe_angle_from_cos(cos: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    eps_f = max(1e-7, min(1e-2, float(eps)))
    cos_f = torch.nan_to_num(cos.float(), nan=0.0, posinf=1.0 - eps_f, neginf=-1.0 + eps_f)
    angle = torch.acos(torch.clamp(cos_f, -1.0 + eps_f, 1.0 - eps_f))
    return torch.nan_to_num(angle, nan=math.pi / 2.0, posinf=math.pi - eps_f, neginf=eps_f)


def _bounded_softplus(value: torch.Tensor, *, clip: float = 20.0) -> torch.Tensor:
    value_f = torch.nan_to_num(value.float(), nan=0.0, posinf=float(clip), neginf=-float(clip))
    clip_f = float(clip)
    clamped = torch.clamp(value_f, -clip_f, clip_f)
    # Preserve the linear positive-tail gradient. A hard upper clamp makes
    # badly violated geometry targets report a large loss while providing no
    # corrective gradient, exactly the failure mode this helper must avoid.
    return F.softplus(clamped) + F.relu(value_f - clip_f)


def _bounded_positive_scalar_loss(value: torch.Tensor, *, cap: float) -> torch.Tensor:
    """Bound a non-negative scalar objective while preserving its local gradient."""

    cap_f = max(1e-6, float(cap))
    value_f = torch.nan_to_num(value.float(), nan=0.0, posinf=cap_f * 20.0, neginf=0.0).clamp_min(0.0)
    return value_f.new_tensor(cap_f) * torch.tanh(value_f / cap_f)


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
) -> Tuple[torch.Tensor, Dict[str, Any]]:
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


@dataclass(frozen=True)
class SoftUnknownMixupBatch:
    """Synthetic low-density samples mixed from multiple TX classes."""

    features: torch.Tensor
    source_indices: torch.Tensor
    source_labels: torch.Tensor
    weights: torch.Tensor
    class_ids: torch.Tensor


def make_soft_unknown_mixup(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    mixup_count: int = 16,
    count: Optional[int] = None,
    mixup_order: int = 3,
    alpha: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> SoftUnknownMixupBatch:
    """Build virtual unknown samples by mixing samples from distinct TX classes."""

    device = z.device
    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    labels = labels.view(-1).long()
    if labels.numel() != z_norm.size(0):
        raise ValueError(f"label count {labels.numel()} does not match feature batch {z_norm.size(0)}")
    if count is not None:
        mixup_count = int(count)
    mixup_order = int(max(2, mixup_order))
    mixup_count = int(max(0, mixup_count))
    valid = labels >= 0
    unique_labels = torch.unique(labels[valid])

    if mixup_count == 0 or unique_labels.numel() < mixup_order:
        empty_idx = torch.empty((0, mixup_order), device=device, dtype=torch.long)
        return SoftUnknownMixupBatch(
            features=z_norm.new_zeros((0, z_norm.shape[1])),
            source_indices=empty_idx,
            source_labels=empty_idx.clone(),
            weights=z_norm.new_zeros((0, mixup_order)),
            class_ids=unique_labels,
        )

    by_label: Dict[int, torch.Tensor] = {
        int(label.item()): torch.nonzero(labels == label, as_tuple=False).flatten()
        for label in unique_labels
    }
    alpha = float(max(alpha, 1e-4))
    inv_alpha = 1.0 / alpha
    mixed_features = []
    source_indices = []
    source_labels = []
    source_weights = []

    for _ in range(mixup_count):
        label_perm = torch.randperm(unique_labels.numel(), device=device, generator=generator)[:mixup_order]
        chosen_labels = unique_labels[label_perm].long()
        chosen_indices = []
        for label in chosen_labels:
            candidates = by_label[int(label.item())]
            pick = torch.randint(candidates.numel(), (1,), device=device, generator=generator)
            chosen_indices.append(candidates[pick].reshape(()))
        idx = torch.stack(chosen_indices)
        weights = torch.rand((mixup_order,), device=device, dtype=z_norm.dtype, generator=generator).clamp_min(1e-6)
        weights = weights.pow(inv_alpha)
        weights = weights / weights.sum().clamp_min(1e-6)
        mixed = (z_norm[idx] * weights.unsqueeze(1)).sum(dim=0)
        mixed_features.append(mixed)
        source_indices.append(idx)
        source_labels.append(chosen_labels)
        source_weights.append(weights)

    features = safe_l2_normalize(torch.stack(mixed_features, dim=0), dim=1)
    return SoftUnknownMixupBatch(
        features=features,
        source_indices=torch.stack(source_indices, dim=0),
        source_labels=torch.stack(source_labels, dim=0),
        weights=torch.stack(source_weights, dim=0),
        class_ids=unique_labels,
    )


def _soft_targets_from_mixup(
    mixup: SoftUnknownMixupBatch,
    num_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    targets = torch.zeros((mixup.features.shape[0], num_classes), device=device, dtype=dtype)
    if mixup.features.numel() == 0 or num_classes <= 0:
        return targets
    labels = mixup.source_labels.clamp(min=0, max=num_classes - 1).long()
    targets.scatter_add_(1, labels, mixup.weights.to(device=device, dtype=dtype))
    return targets


def _energy_to_centers(features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    scores = safe_l2_normalize(features, dim=1) @ centers.t()
    return -torch.logsumexp(scores, dim=1)


def _top_cvar_mean(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the largest alpha fraction, preserving gradients."""
    if values.numel() == 0:
        return values.new_tensor(0.0)
    flat = values.reshape(-1)
    frac = max(1e-6, min(1.0, float(alpha)))
    k = max(1, int(math.ceil(float(flat.numel()) * frac)))
    return torch.topk(flat, k=k, largest=True).values.mean()


def soft_unknown_mixup_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    logits: Optional[torch.Tensor] = None,
    mixup: Optional[SoftUnknownMixupBatch] = None,
    mixup_count: int = 16,
    mixup_order: int = 3,
    alpha: float = 0.5,
    energy_margin: float = 1.0,
    ce_weight: float = 1.0,
    energy_weight: float = 1.0,
    vacuum_weight: float = 0.0,
    vacuum_width_rad: float = math.radians(6.0),
    vacuum_hard_k: int = 2,
    detach_mixup: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Proxy-unknown loss using soft-label multi-TX mixup samples."""

    metrics: Dict[str, float] = {
        "soft_unknown_mixup_count": 0.0,
        "soft_unknown_mixup_order": float(max(2, mixup_order)),
        "soft_unknown_mixup_ce": 0.0,
        "soft_unknown_mixup_energy": 0.0,
        "soft_unknown_mixup_vacuum": 0.0,
        "soft_unknown_mixup_virtual_accept_rate": 0.0,
        "soft_unknown_mixup_vacuum_violation": 0.0,
    }
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, metrics
    labels = labels.view(-1).long()
    if labels.numel() != z.size(0):
        raise ValueError(f"label count {labels.numel()} does not match feature batch {z.size(0)}")
    if mixup is None:
        mixup = make_soft_unknown_mixup(
            z,
            labels,
            mixup_count=mixup_count,
            mixup_order=mixup_order,
            alpha=alpha,
            generator=generator,
        )
    if mixup.features.numel() == 0:
        return zero_like_with_grad(z), metrics

    mix_features = mixup.features.detach() if bool(detach_mixup) else mixup.features
    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    valid = labels >= 0
    classes = []
    centers = []
    radii = []
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls)
        if int(cls_mask.sum().item()) < 2:
            continue
        center = safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
        own_cos = (z_norm[cls_mask] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        centers.append(center)
        radii.append(
            _robust_three_sigma_radius_from_angles(
                _safe_angle_from_cos(own_cos.detach()),
                fallback_rad=math.radians(40.0),
            )
        )
        classes.append(cls)
    if len(centers) < 2:
        return zero_like_with_grad(z), metrics

    proto = torch.stack(centers, dim=0)
    radius_vec = torch.stack(radii, dim=0).to(device=proto.device, dtype=proto.dtype)
    metrics["soft_unknown_mixup_count"] = float(mix_features.shape[0])
    metrics["soft_unknown_mixup_order"] = float(mixup.source_labels.shape[1])

    energy_known = _energy_to_centers(z_norm[valid], proto)
    energy_mix = _energy_to_centers(mix_features, proto)
    threshold = torch.quantile(energy_known.detach(), 0.95) if energy_known.numel() > 1 else energy_known.detach().mean()
    energy_loss = F.relu(float(energy_margin) - (energy_mix.mean() - energy_known.mean())).pow(2)
    metrics["soft_unknown_mixup_energy"] = _scalar_metric(energy_loss)
    metrics["soft_unknown_mixup_virtual_accept_rate"] = float(
        (energy_mix.detach() <= threshold).float().mean().item()
    )

    ce_loss = z.new_tensor(0.0)
    if logits is not None and torch.is_tensor(logits) and logits.numel() > 0 and float(ce_weight) > 0.0:
        idx = mixup.source_indices.clamp(min=0, max=logits.shape[0] - 1).long()
        mix_logits = (logits[idx] * mixup.weights.to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)).sum(dim=1)
        targets = _soft_targets_from_mixup(mixup, logits.shape[1], device=logits.device, dtype=logits.dtype)
        ce_loss = -(targets * F.log_softmax(mix_logits, dim=1)).sum(dim=1).mean()
        metrics["soft_unknown_mixup_ce"] = _scalar_metric(ce_loss)

    vacuum_loss = z.new_tensor(0.0)
    if float(vacuum_weight) > 0.0:
        angles = _safe_angle_from_cos((mix_features @ proto.t()).clamp(-1.0 + 1e-4, 1.0 - 1e-4))
        boundary = radius_vec.view(1, -1) + max(0.0, float(vacuum_width_rad))
        violation = F.relu(boundary - angles).pow(2)
        hard_k = max(1, min(int(vacuum_hard_k), proto.size(0)))
        vacuum_loss = violation.topk(k=hard_k, dim=1, largest=True).values.mean()
        metrics["soft_unknown_mixup_vacuum"] = _scalar_metric(vacuum_loss)
        metrics["soft_unknown_mixup_vacuum_violation"] = float((violation.detach() > 0.0).float().mean().item())

    total = (
        max(0.0, float(ce_weight)) * ce_loss
        + max(0.0, float(energy_weight)) * energy_loss
        + max(0.0, float(vacuum_weight)) * vacuum_loss
    )
    return total, metrics


def proxy_unknown_energy_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    holdout_label: Optional[int] = None,
    virtual_count: int = 16,
    virtual_mode: str = "legacy",
    energy_margin: float = 1.0,
    energy_temperature: float = 1.0,
    placeholder_weight: float = 0.5,
    virtual_detach: bool = True,
    vacuum_weight: float = 0.0,
    vacuum_width_rad: float = math.radians(4.0),
    vacuum_hard_k: int = 2,
    vacuum_radius_rad: float = math.radians(40.0),
    core_quantile: float = 0.90,
    accept_quantile: float = 0.95,
    tail_quantile: float = 0.95,
    overflow_quantile: float = 0.99,
    component_radius_mode: str = "core_quantile",
    component_radius_quantile: float = 0.80,
    vaccept_weight: float = 0.0,
    core_accept_weight: float = 0.0,
    component_gate_weight: float = 0.0,
    tail_quarantine_weight: float = 0.0,
    source_safe_weight: float = 0.0,
    bridge_accept_weight: float = 0.0,
    shell_outward_accept_weight: float = 0.0,
    low_density_accept_weight: float = 0.0,
    energy_margin_quantile_weight: float = 0.0,
    radius_budget_weight: float = 0.0,
    radius_inter_ratio_weight: float = 0.0,
    vaccept_cvar_alpha: float = 0.25,
    unknown_margin: float = 0.08,
    known_margin: float = 0.05,
    energy_softplus_temperature: float = 0.04,
    accept_softplus_temperature: float = 0.04,
    bridge_accept_target: float = 0.20,
    shell_outward_accept_target: float = 0.25,
    tail_accept_target: float = 0.45,
    overflow_accept_target: float = 0.25,
    energy_margin_q: float = 0.10,
    energy_margin_target: float = 0.08,
    radius_budget_rad: float = math.radians(10.0),
    radius_max_budget_rad: float = math.radians(15.0),
    radius_inter_ratio_target: float = 0.25,
    density_temperature_rad: float = math.radians(3.0),
    component_temperature_rad: float = math.radians(3.0),
    component_margin_rad: float = math.radians(4.0),
    component_margin_temperature_rad: float = math.radians(3.0),
    shell_width_rad: float = math.radians(4.0),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Source-only proxy unknown loss using leave-one-TX-out and feature outliers."""
    default_metrics = {
        "active": 0.0,
        "known_count": 0.0,
        "proxy_unknown_count": 0.0,
        "virtual_count": 0.0,
        "core_count": 0.0,
        "tail_count": 0.0,
        "overflow_count": 0.0,
        "energy_known": float("nan"),
        "energy_proxy": float("nan"),
        "energy_virtual": float("nan"),
        "energy_margin": float("nan"),
        "accept_energy_threshold": float("nan"),
        "core_energy_threshold": float("nan"),
        "vaccept_surrogate": 0.0,
        "vaccept_surrogate_CVaR": 0.0,
        "core_accept_loss": 0.0,
        "component_gate_unknown": 0.0,
        "component_gate_accept_prob": float("nan"),
        "component_gate_accept_prob_max": float("nan"),
        "tail_quarantine_loss": 0.0,
        "source_safe_loss": 0.0,
        "bridge_governance_loss": 0.0,
        "shell_outward_accept_loss": 0.0,
        "low_density_accept_loss": 0.0,
        "energy_margin_quantile_loss": 0.0,
        "radius_budget_loss": 0.0,
        "radius_inter_ratio_loss": 0.0,
        "tail_accept_loss": 0.0,
        "overflow_accept_loss": 0.0,
        "energy_margin_q05": float("nan"),
        "energy_margin_q10": float("nan"),
        "component_radius_p95_deg": float("nan"),
        "component_radius_max_deg": float("nan"),
        "component_radius_mode_code": 1.0,
        "component_gate_radius_p95_deg": float("nan"),
        "component_gate_radius_max_deg": float("nan"),
        "radius_inter_ratio": float("nan"),
        "radius_to_inter_ratio": float("nan"),
        "low_density_accept_prob": float("nan"),
        "low_density_accept_rate": float("nan"),
        "proxy_unknown_auc": float("nan"),
        "virtual_accept_rate": float("nan"),
        "proxy_vaccept": float("nan"),
        "proxy_vaccept_proxy_only": float("nan"),
        "proxy_reject_claim_allowed": 0.0,
        "virtual_accept_rate_core": float("nan"),
        "proxy_accept_rate": float("nan"),
        "hard_proxy_accept_rate": float("nan"),
        "shell_accept_rate": float("nan"),
        "bridge_accept_rate": float("nan"),
        "outward_accept_rate": float("nan"),
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
    radius_mode = str(component_radius_mode or "core_quantile").lower().strip()
    radius_mode_code = {
        "three_sigma": 0.0,
        "legacy_three_sigma": 0.0,
        "core": 1.0,
        "core_quantile": 1.0,
        "strict_core": 1.0,
        "min_three_sigma_core": 2.0,
        "min_core_three_sigma": 2.0,
        "core_safe": 2.0,
        "accept": 3.0,
        "accept_quantile": 3.0,
        "min_three_sigma_quantile": 4.0,
        "min_quantile_three_sigma": 4.0,
    }.get(radius_mode, 1.0)
    radius_q = max(0.0, min(1.0, float(component_radius_quantile)))
    for cls in known_labels:
        cls_mask = known_mask & labels.eq(cls)
        if bool(cls_mask.any()):
            center = safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            centers.append(center)
            own_cos = (z_norm[cls_mask] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
            own_angles = _safe_angle_from_cos(own_cos)
            robust_radius = _robust_three_sigma_radius_from_angles(
                own_angles,
                fallback_rad=vacuum_radius_rad,
            )
            if own_angles.numel() > 1:
                quant_radius = torch.quantile(own_angles.detach(), radius_q)
                core_radius = torch.quantile(own_angles.detach(), max(0.0, min(1.0, float(core_quantile))))
                accept_radius = torch.quantile(own_angles.detach(), max(0.0, min(1.0, float(accept_quantile))))
            else:
                quant_radius = own_angles.detach().max()
                core_radius = own_angles.detach().max()
                accept_radius = own_angles.detach().max()
            if radius_mode in {"core", "core_quantile", "strict_core"}:
                gate_radius = quant_radius
            elif radius_mode in {"accept", "accept_quantile"}:
                gate_radius = accept_radius
            elif radius_mode in {"min_three_sigma_core", "min_core_three_sigma", "core_safe"}:
                gate_radius = torch.minimum(robust_radius, core_radius)
            elif radius_mode in {"min_three_sigma_quantile", "min_quantile_three_sigma"}:
                gate_radius = torch.minimum(robust_radius, quant_radius)
            elif radius_mode in {"three_sigma", "legacy_three_sigma"}:
                gate_radius = robust_radius
            else:
                gate_radius = quant_radius
            class_radii.append(
                torch.clamp(gate_radius, min=0.0)
            )
    if len(centers) <= 0:
        return zero_like_with_grad(z), default_metrics
    proto = torch.stack(centers, dim=0)
    radius_vec = torch.stack(class_radii, dim=0).to(device=proto.device, dtype=proto.dtype)
    z_known = z_norm[known_mask]
    y_known = labels[known_mask]

    def energy(feat: torch.Tensor) -> torch.Tensor:
        scores = (safe_l2_normalize(feat, dim=1) @ proto.t()) / max(1e-4, float(energy_temperature))
        return -torch.logsumexp(scores, dim=1)

    e_known = energy(z_known)
    e_proxy = energy(z_norm[proxy_mask])
    virtual_parts = _make_proxy_virtual_unknown_pool(
        z_known,
        y_known,
        known_labels,
        proto,
        radius_vec,
        count=max(0, int(virtual_count)),
        mode=str(virtual_mode or "legacy"),
        shell_width_rad=float(shell_width_rad),
    )
    virtual_tensors = [part for part in virtual_parts.values() if part.numel() > 0]
    virtual = torch.cat(virtual_tensors, dim=0) if virtual_tensors else z.new_zeros((0, z.size(1)))
    if bool(virtual_detach):
        virtual = virtual.detach()
        virtual_parts = {name: part.detach() for name, part in virtual_parts.items()}
    e_virtual = energy(virtual) if virtual.numel() else z.new_zeros((0,))

    proxy_target = torch.cat([e_proxy, e_virtual], dim=0)
    margin_loss = F.relu(float(energy_margin) - (proxy_target.mean() - e_known.mean())).pow(2)
    logits_known = -e_known
    logits_unknown = torch.cat([e_proxy, e_virtual], dim=0)
    placeholder = (
        F.softplus(-logits_known).mean()
        + F.softplus(-logits_unknown).mean()
    ) if logits_unknown.numel() else z.new_tensor(0.0)

    core_q = max(0.0, min(1.0, float(core_quantile)))
    tail_q = max(core_q, min(1.0, float(tail_quantile)))
    overflow_q = max(tail_q, min(1.0, float(overflow_quantile)))
    known_angles = z_known.new_zeros((z_known.size(0),))
    known_proto_indices = torch.zeros((z_known.size(0),), device=z_known.device, dtype=torch.long)
    core_mask_local = torch.zeros((z_known.size(0),), device=z_known.device, dtype=torch.bool)
    tail_mask_local = torch.zeros_like(core_mask_local)
    overflow_mask_local = torch.zeros_like(core_mask_local)
    for proto_idx, cls in enumerate(known_labels):
        local_mask = y_known.eq(cls)
        if not bool(local_mask.any()):
            continue
        cls_cos = (z_known[local_mask] * proto[proto_idx].view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        cls_angles = _safe_angle_from_cos(cls_cos)
        known_angles[local_mask] = cls_angles
        known_proto_indices[local_mask] = int(proto_idx)
        cls_det = cls_angles.detach()
        core_radius = torch.quantile(cls_det, core_q) if cls_det.numel() > 1 else cls_det.max()
        tail_radius = torch.quantile(cls_det, tail_q) if cls_det.numel() > 1 else cls_det.max()
        overflow_radius = torch.quantile(cls_det, overflow_q) if cls_det.numel() > 1 else cls_det.max()
        core_mask_local |= local_mask & (known_angles <= core_radius)
        tail_mask_local |= local_mask & (known_angles > core_radius) & (known_angles <= tail_radius)
        overflow_mask_local |= local_mask & (known_angles > overflow_radius)
    if not bool(core_mask_local.any()):
        core_mask_local = torch.ones_like(core_mask_local)
    e_core = e_known[core_mask_local]
    accept_q = max(0.0, min(1.0, float(accept_quantile)))
    t_core = torch.quantile(e_core.detach(), accept_q) if e_core.numel() > 1 else e_core.detach().mean()
    tau_e = max(1e-4, float(energy_softplus_temperature))
    tau_a = max(1e-4, float(accept_softplus_temperature))
    tau_density = max(1e-4, float(density_temperature_rad))
    tau_ratio = max(1e-4, float(component_margin_temperature_rad))
    alpha = max(1e-6, min(1.0, float(vaccept_cvar_alpha)))
    vaccept_terms = F.softplus((t_core + float(unknown_margin) - proxy_target) / tau_e) if proxy_target.numel() else z.new_zeros((0,))
    vaccept_loss = _top_cvar_mean(vaccept_terms, alpha) if vaccept_terms.numel() else z.new_tensor(0.0)
    core_accept_loss = F.softplus((e_core - (t_core - float(known_margin))) / tau_e).mean() if e_core.numel() else z.new_tensor(0.0)

    z_core_density = z_known[core_mask_local]
    if z_core_density.numel() and bool(core_mask_local.any()):
        density_radius = torch.quantile(known_angles[core_mask_local].detach(), min(0.95, max(0.50, core_q)))
    else:
        density_radius = radius_vec.detach().median() if radius_vec.numel() else z.new_tensor(float(vacuum_radius_rad))

    def _soft_accept_prob(feat: torch.Tensor, feat_energy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if feat is None or feat.numel() == 0:
            return z.new_zeros((0,)), z.new_zeros((0,))
        feat_norm = safe_l2_normalize(feat, dim=1)
        feat_cos = (feat_norm @ proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        feat_angles = _safe_angle_from_cos(feat_cos)
        if feat_energy is None or feat_energy.numel() != feat_norm.size(0):
            feat_energy = energy(feat_norm)
        radius_gate = torch.sigmoid(
            (radius_vec.detach().view(1, -1) - feat_angles) / max(1e-4, float(component_temperature_rad))
        )
        energy_gate = torch.sigmoid((t_core - feat_energy) / tau_e).view(-1, 1)
        if feat_angles.size(1) > 1:
            sorted_angles = torch.sort(feat_angles.detach(), dim=1).values
            class_gap = sorted_angles[:, 1] - sorted_angles[:, 0]
            margin_gate = torch.sigmoid(
                (class_gap - float(component_margin_rad)) / max(1e-4, float(component_margin_temperature_rad))
            ).view(-1, 1)
        else:
            margin_gate = torch.ones((feat_angles.size(0), 1), device=feat_angles.device, dtype=feat_angles.dtype)
        if z_core_density.numel():
            core_cos = (feat_norm @ z_core_density.detach().t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            nearest_core_angle = _safe_angle_from_cos(core_cos.max(dim=1).values)
            density_gate = torch.sigmoid((density_radius - nearest_core_angle) / tau_density).view(-1, 1)
            low_density_prob = torch.sigmoid((nearest_core_angle - density_radius) / tau_density)
        else:
            density_gate = torch.ones((feat_angles.size(0), 1), device=feat_angles.device, dtype=feat_angles.dtype)
            low_density_prob = torch.zeros((feat_angles.size(0),), device=feat_angles.device, dtype=feat_angles.dtype)
        accept_prob = (radius_gate * energy_gate * margin_gate * density_gate).max(dim=1).values
        return accept_prob, low_density_prob

    def _accept_cvar_loss(prob: torch.Tensor, target: float) -> torch.Tensor:
        if prob.numel() == 0:
            return z.new_tensor(0.0)
        terms = F.softplus((prob - float(target)) / tau_a)
        return _top_cvar_mean(terms, alpha)

    e_tail = e_known[tail_mask_local]
    tail_energy_terms = F.softplus((t_core + float(unknown_margin) - e_tail) / tau_e) if e_tail.numel() else z.new_zeros((0,))
    tail_energy_loss = _top_cvar_mean(tail_energy_terms, alpha) if tail_energy_terms.numel() else z.new_tensor(0.0)
    tail_accept_prob, _tail_low_density = _soft_accept_prob(z_known[tail_mask_local], e_tail)
    tail_accept_loss = _accept_cvar_loss(tail_accept_prob, float(tail_accept_target))
    tail_quarantine_loss = tail_energy_loss + tail_accept_loss
    e_overflow = e_known[overflow_mask_local]
    source_safe_energy_terms = F.softplus((t_core + float(unknown_margin) - e_overflow) / tau_e) if e_overflow.numel() else z.new_zeros((0,))
    source_safe_energy_loss = _top_cvar_mean(source_safe_energy_terms, alpha) if source_safe_energy_terms.numel() else z.new_tensor(0.0)
    overflow_accept_prob, _overflow_low_density = _soft_accept_prob(z_known[overflow_mask_local], e_overflow)
    overflow_accept_loss = _accept_cvar_loss(overflow_accept_prob, float(overflow_accept_target))
    source_safe_loss = source_safe_energy_loss + overflow_accept_loss

    component_gate_loss = z.new_tensor(0.0)
    component_accept_prob = z.new_tensor(float("nan"))
    component_accept_prob_max = z.new_tensor(float("nan"))
    unknown_feat_all = torch.cat([z_norm[proxy_mask], virtual], dim=0)
    unknown_accept_prob = z.new_zeros((0,))
    unknown_low_density_prob = z.new_zeros((0,))
    if unknown_feat_all.numel():
        unknown_accept_prob, unknown_low_density_prob = _soft_accept_prob(unknown_feat_all, proxy_target)
        component_gate_loss = _top_cvar_mean(unknown_accept_prob, alpha)
        component_accept_prob = unknown_accept_prob.detach().mean()
        component_accept_prob_max = unknown_accept_prob.detach().max()

    def _part_accept_loss(name: str, target: float) -> torch.Tensor:
        part = virtual_parts.get(name)
        if part is None or part.numel() == 0:
            return z.new_tensor(0.0)
        part_energy = energy(part)
        part_prob, _ = _soft_accept_prob(part, part_energy)
        accept_loss = _accept_cvar_loss(part_prob, target)
        delta = part_energy - t_core
        energy_terms = F.softplus((float(energy_margin_target) - delta) / tau_e)
        return accept_loss + _top_cvar_mean(energy_terms, alpha)

    bridge_governance_loss = _part_accept_loss("bridge", float(bridge_accept_target))
    shell_outward_losses = [
        _part_accept_loss("shell", float(shell_outward_accept_target)),
        _part_accept_loss("outward", float(shell_outward_accept_target)),
    ]
    shell_outward_accept_loss = torch.stack(shell_outward_losses).mean() if shell_outward_losses else z.new_tensor(0.0)

    density_feats = [unknown_feat_all]
    density_energy = [proxy_target]
    if e_tail.numel():
        density_feats.append(z_known[tail_mask_local])
        density_energy.append(e_tail)
    if e_overflow.numel():
        density_feats.append(z_known[overflow_mask_local])
        density_energy.append(e_overflow)
    if density_feats and sum(int(t.numel()) for t in density_feats) > 0:
        density_feat_all = torch.cat([t for t in density_feats if t.numel() > 0], dim=0)
        density_energy_all = torch.cat([t for t in density_energy if t.numel() > 0], dim=0)
        density_accept_prob, density_low_prob = _soft_accept_prob(density_feat_all, density_energy_all)
        low_density_terms = density_accept_prob * density_low_prob
        low_density_accept_loss = _top_cvar_mean(low_density_terms, alpha) if low_density_terms.numel() else z.new_tensor(0.0)
        low_density_accept_prob = low_density_terms.detach().mean() if low_density_terms.numel() else z.new_tensor(float("nan"))
    else:
        low_density_accept_loss = z.new_tensor(0.0)
        low_density_accept_prob = z.new_tensor(float("nan"))

    energy_margin_loss = z.new_tensor(0.0)
    energy_margin_q05 = z.new_tensor(float("nan"))
    energy_margin_q10 = z.new_tensor(float("nan"))
    if proxy_target.numel():
        delta_e = proxy_target - t_core
        energy_q_alpha = max(1e-6, min(1.0, float(energy_margin_q)))
        energy_margin_terms = F.softplus((float(energy_margin_target) - delta_e) / tau_e)
        energy_margin_loss = _top_cvar_mean(energy_margin_terms, energy_q_alpha)
        delta_det = delta_e.detach()
        energy_margin_q05 = torch.quantile(delta_det, 0.05) if delta_det.numel() > 1 else delta_det.mean()
        energy_margin_q10 = torch.quantile(delta_det, 0.10) if delta_det.numel() > 1 else delta_det.mean()

    component_radius_p95 = torch.quantile(known_angles.detach(), 0.95) if known_angles.numel() > 1 else known_angles.detach().mean()
    component_radius_max = known_angles.detach().max() if known_angles.numel() else z.new_tensor(float("nan"))
    component_gate_radius_p95 = torch.quantile(radius_vec.detach(), 0.95) if radius_vec.numel() > 1 else radius_vec.detach().mean()
    component_gate_radius_max = radius_vec.detach().max() if radius_vec.numel() else z.new_tensor(float("nan"))
    radius_budget_terms = F.softplus((known_angles - float(radius_budget_rad)) / max(1e-4, float(component_temperature_rad)))
    radius_budget_loss = _top_cvar_mean(radius_budget_terms, alpha) if radius_budget_terms.numel() else z.new_tensor(0.0)
    if known_angles.numel():
        radius_max_terms = F.softplus((known_angles - float(radius_max_budget_rad)) / max(1e-4, float(component_temperature_rad)))
        radius_budget_loss = radius_budget_loss + _top_cvar_mean(radius_max_terms, alpha)

    radius_inter_ratio_loss = z.new_tensor(0.0)
    radius_inter_ratio_metric = z.new_tensor(float("nan"))
    if proto.size(0) > 1 and known_angles.numel():
        inter_angles = _safe_angle_from_cos((proto @ proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        inf_diag = torch.eye(inter_angles.size(0), device=inter_angles.device, dtype=torch.bool)
        inter_angles = inter_angles.masked_fill(inf_diag, float("inf"))
        nearest_inter = inter_angles.min(dim=1).values.detach().clamp_min(1e-4)
        sample_inter = nearest_inter[known_proto_indices]
        ratio = known_angles / sample_inter
        radius_inter_ratio_metric = ratio.detach().max() if ratio.numel() else z.new_tensor(float("nan"))
        ratio_terms = F.softplus((ratio - float(radius_inter_ratio_target)) / tau_ratio)
        radius_inter_ratio_loss = _top_cvar_mean(ratio_terms, alpha)

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
        + max(0.0, float(vaccept_weight)) * vaccept_loss
        + max(0.0, float(core_accept_weight)) * core_accept_loss
        + max(0.0, float(component_gate_weight)) * component_gate_loss
        + max(0.0, float(tail_quarantine_weight)) * tail_quarantine_loss
        + max(0.0, float(source_safe_weight)) * source_safe_loss
        + max(0.0, float(bridge_accept_weight)) * bridge_governance_loss
        + max(0.0, float(shell_outward_accept_weight)) * shell_outward_accept_loss
        + max(0.0, float(low_density_accept_weight)) * low_density_accept_loss
        + max(0.0, float(energy_margin_quantile_weight)) * energy_margin_loss
        + max(0.0, float(radius_budget_weight)) * radius_budget_loss
        + max(0.0, float(radius_inter_ratio_weight)) * radius_inter_ratio_loss
    )

    with torch.no_grad():
        known_scores = e_known.detach()
        unknown_scores = torch.cat([e_proxy.detach(), e_virtual.detach()], dim=0)
        auc = _binary_auc(known_scores, unknown_scores)
        known_accept_threshold = torch.quantile(known_scores, 0.95) if known_scores.numel() else z.new_tensor(float("nan"))
        core_accept_threshold = t_core.detach()
        virtual_accept = (
            float((e_virtual.detach() <= known_accept_threshold).float().mean().item())
            if e_virtual.numel() and torch.isfinite(known_accept_threshold)
            else float("nan")
        )
        virtual_accept_core = (
            float((e_virtual.detach() <= core_accept_threshold).float().mean().item())
            if e_virtual.numel() and torch.isfinite(core_accept_threshold)
            else float("nan")
        )
        proxy_accept = (
            float((e_proxy.detach() <= core_accept_threshold).float().mean().item())
            if e_proxy.numel() and torch.isfinite(core_accept_threshold)
            else float("nan")
        )
        hard_proxy_accept = (
            float((unknown_scores <= core_accept_threshold).float().mean().item())
            if unknown_scores.numel() and torch.isfinite(core_accept_threshold)
            else float("nan")
        )

        def _part_accept_rate(name: str) -> float:
            part = virtual_parts.get(name)
            if part is None or part.numel() == 0 or not torch.isfinite(core_accept_threshold):
                return float("nan")
            part_energy = energy(part).detach()
            return float((part_energy <= core_accept_threshold).float().mean().item())

    metrics = {
        "active": 1.0,
        "known_count": float(int(known_mask.sum().item())),
        "proxy_unknown_count": float(int(proxy_mask.sum().item())),
        "virtual_count": float(int(e_virtual.numel())),
        "core_count": float(int(core_mask_local.sum().item())),
        "tail_count": float(int(tail_mask_local.sum().item())),
        "overflow_count": float(int(overflow_mask_local.sum().item())),
        "energy_known": _scalar_metric(e_known),
        "energy_proxy": _scalar_metric(e_proxy),
        "energy_virtual": _scalar_metric(e_virtual) if e_virtual.numel() else float("nan"),
        "energy_margin": _scalar_metric(e_proxy.mean() - e_known.mean()),
        "accept_energy_threshold": _scalar_metric(known_accept_threshold),
        "core_energy_threshold": _scalar_metric(t_core),
        "vaccept_surrogate": _scalar_metric(vaccept_loss),
        "vaccept_surrogate_CVaR": _scalar_metric(vaccept_loss),
        "core_accept_loss": _scalar_metric(core_accept_loss),
        "component_gate_unknown": _scalar_metric(component_gate_loss),
        "component_gate_accept_prob": _scalar_metric(component_accept_prob),
        "component_gate_accept_prob_max": _scalar_metric(component_accept_prob_max),
        "tail_quarantine_loss": _scalar_metric(tail_quarantine_loss),
        "source_safe_loss": _scalar_metric(source_safe_loss),
        "bridge_governance_loss": _scalar_metric(bridge_governance_loss),
        "shell_outward_accept_loss": _scalar_metric(shell_outward_accept_loss),
        "low_density_accept_loss": _scalar_metric(low_density_accept_loss),
        "energy_margin_quantile_loss": _scalar_metric(energy_margin_loss),
        "radius_budget_loss": _scalar_metric(radius_budget_loss),
        "radius_inter_ratio_loss": _scalar_metric(radius_inter_ratio_loss),
        "tail_accept_loss": _scalar_metric(tail_accept_loss),
        "overflow_accept_loss": _scalar_metric(overflow_accept_loss),
        "energy_margin_q05": _scalar_metric(energy_margin_q05),
        "energy_margin_q10": _scalar_metric(energy_margin_q10),
        "component_radius_p95_deg": math.degrees(_scalar_metric(component_radius_p95)),
        "component_radius_max_deg": math.degrees(_scalar_metric(component_radius_max)),
        "component_radius_mode_code": float(radius_mode_code),
        "component_gate_radius_p95_deg": math.degrees(_scalar_metric(component_gate_radius_p95)),
        "component_gate_radius_max_deg": math.degrees(_scalar_metric(component_gate_radius_max)),
        "radius_inter_ratio": _scalar_metric(radius_inter_ratio_metric),
        "radius_to_inter_ratio": _scalar_metric(radius_inter_ratio_metric),
        "low_density_accept_prob": _scalar_metric(low_density_accept_prob),
        "low_density_accept_rate": _scalar_metric(low_density_accept_prob),
        "proxy_unknown_auc": float(auc),
        "virtual_accept_rate": float(virtual_accept),
        "proxy_vaccept": float(virtual_accept),
        "proxy_vaccept_proxy_only": float(virtual_accept),
        "proxy_reject_claim_allowed": 0.0,
        "virtual_accept_rate_core": float(virtual_accept_core),
        "proxy_accept_rate": float(proxy_accept),
        "hard_proxy_accept_rate": float(hard_proxy_accept),
        "shell_accept_rate": _part_accept_rate("shell"),
        "bridge_accept_rate": _part_accept_rate("bridge"),
        "outward_accept_rate": _part_accept_rate("outward"),
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


def _make_component_shell_outliers(
    z_known: torch.Tensor,
    y_known: torch.Tensor,
    class_ids: torch.Tensor,
    proto: torch.Tensor,
    radius_vec: torch.Tensor,
    *,
    count: int,
    shell_width_rad: float,
) -> torch.Tensor:
    if count <= 0 or z_known.numel() == 0 or proto.numel() == 0:
        return z_known.new_zeros((0, z_known.size(1)))
    out = []
    n_proto = proto.size(0)
    for i in range(int(count)):
        proto_idx = i % n_proto
        cls = class_ids[proto_idx]
        cls_idx = torch.nonzero(y_known.eq(cls), as_tuple=False).flatten()
        if cls_idx.numel() == 0:
            base = proto[proto_idx]
        else:
            base = z_known[cls_idx[i % cls_idx.numel()]]
        center = proto[proto_idx]
        tangent = base - (base * center).sum() * center
        if float(torch.linalg.vector_norm(tangent).detach().item()) <= 1e-6 and n_proto > 1:
            tangent = proto[(proto_idx + 1) % n_proto] - (proto[(proto_idx + 1) % n_proto] * center).sum() * center
        tangent = safe_l2_normalize(tangent.view(1, -1), dim=1).squeeze(0)
        theta = torch.clamp(radius_vec[proto_idx].detach() + max(0.0, float(shell_width_rad)), 1e-4, math.pi - 1e-4)
        shell = torch.cos(theta) * center + torch.sin(theta) * tangent
        out.append(shell)
    return safe_l2_normalize(torch.stack(out, dim=0), dim=1)


def _make_interclass_bridge_outliers(
    proto: torch.Tensor,
    *,
    count: int,
    class_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if count <= 0 or proto.numel() == 0 or proto.size(0) < 2:
        feat_dim = proto.size(1) if proto.dim() == 2 else 0
        return proto.new_zeros((0, feat_dim))
    out = []
    n_proto = proto.size(0)
    for i in range(int(count)):
        first = i % n_proto
        second = (i + 1 + (i // n_proto)) % n_proto
        if class_ids is not None and class_ids.numel() == n_proto:
            for offset in range(1, n_proto + 1):
                candidate = (first + offset + (i // n_proto)) % n_proto
                if int(class_ids[candidate].item()) != int(class_ids[first].item()):
                    second = candidate
                    break
        p1 = proto[first]
        p2 = proto[second]
        out.append(p1 + p2)
    return safe_l2_normalize(torch.stack(out, dim=0), dim=1)


def _make_query_interclass_bridge_outliers(
    z_known: torch.Tensor,
    y_known: torch.Tensor,
    class_ids: torch.Tensor,
    proto: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    """Create differentiable bridges from current queries to another class."""

    if count <= 0 or z_known.numel() == 0 or proto.size(0) < 2:
        feat_dim = z_known.size(1) if z_known.dim() == 2 else proto.size(1)
        return z_known.new_zeros((0, feat_dim))
    out = []
    for i in range(int(count)):
        query_idx = i % int(z_known.size(0))
        query = z_known[query_idx]
        other = class_ids.ne(y_known[query_idx])
        if not bool(other.any()):
            continue
        other_proto = proto[other]
        nearest_other = other_proto[(query.detach() @ other_proto.detach().t()).argmax()]
        out.append(query + nearest_other.detach())
    if not out:
        return z_known.new_zeros((0, z_known.size(1)))
    return safe_l2_normalize(torch.stack(out, dim=0), dim=1)


def _make_tail_outward_outliers(z_known: torch.Tensor, proto: torch.Tensor, *, count: int) -> torch.Tensor:
    if count <= 0 or z_known.numel() == 0 or proto.numel() == 0:
        return z_known.new_zeros((0, z_known.size(1)))
    sim = z_known @ proto.t()
    nearest_idx = sim.argmax(dim=1)
    nearest_sim = sim.max(dim=1).values
    order = torch.argsort(nearest_sim, descending=False)
    base = z_known[order[: min(int(count), order.numel())]]
    if base.size(0) < int(count):
        reps = int(math.ceil(int(count) / max(1, base.size(0))))
        base = base.repeat(reps, 1)[: int(count)]
        nearest_idx = nearest_idx[order[: min(order.numel(), int(count))]].repeat(reps)[: int(count)]
    else:
        nearest_idx = nearest_idx[order[: int(count)]]
    nearest = proto[nearest_idx]
    return safe_l2_normalize(base + 1.25 * (base - nearest), dim=1)


def _make_proxy_virtual_unknown_pool(
    z_known: torch.Tensor,
    y_known: torch.Tensor,
    class_ids: torch.Tensor,
    proto: torch.Tensor,
    radius_vec: torch.Tensor,
    *,
    count: int,
    mode: str,
    shell_width_rad: float,
) -> Dict[str, torch.Tensor]:
    count = max(0, int(count))
    feat_dim = z_known.size(1) if z_known.dim() == 2 else proto.size(1)
    empty = z_known.new_zeros((0, feat_dim))
    if count <= 0:
        return {"legacy": empty}
    mode_l = str(mode or "legacy").lower().strip()
    if mode_l in {"legacy", "old"}:
        return {"legacy": _make_virtual_outliers(z_known, proto, count=count)}

    shell_n = int(math.ceil(count / 3.0))
    bridge_n = int(math.ceil(count / 3.0))
    outward_n = max(0, count - shell_n - bridge_n)
    parts = {
        "shell": _make_component_shell_outliers(
            z_known,
            y_known,
            class_ids,
            proto,
            radius_vec,
            count=shell_n,
            shell_width_rad=shell_width_rad,
        ),
        "bridge": _make_interclass_bridge_outliers(proto, count=bridge_n, class_ids=class_ids),
        "outward": _make_tail_outward_outliers(z_known, proto, count=outward_n),
    }
    if mode_l in {"mixed", "hybrid", "legacy_hard"}:
        parts["legacy"] = _make_virtual_outliers(z_known, proto, count=count)
    return parts


def tx_conditional_domain_invariance_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    receiver_labels: Optional[torch.Tensor] = None,
    day_labels: Optional[torch.Tensor] = None,
    channel_labels: Optional[torch.Tensor] = None,
    receiver_weight: float = 1.0,
    day_weight: float = 1.0,
    channel_weight: float = 1.0,
    channel_pair_weight: float = 1.0,
    paired_view_count: int = 0,
    min_groups: int = 2,
    min_samples_per_group: int = 2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Directly align nuisance-group centers inside each known TX class.

    Conditioning every alignment term on TX prevents the invariance objective
    from collapsing identity separation while still removing receiver, day and
    clean/satellite channel shortcuts from ``z_id``.
    """

    if z.dim() != 2:
        raise ValueError(f"tx_conditional_domain_invariance_loss expects [B,D], got {tuple(z.shape)}")
    labels = y.view(-1).long()
    if labels.numel() != z.size(0):
        raise ValueError("TX labels must align with z_id features")
    z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)

    def _field_loss(field: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = zero_like_with_grad(z_norm)
        if field is None or not torch.is_tensor(field) or field.numel() != labels.numel():
            return zero, {"active": 0.0, "tx_count": 0.0, "group_count": 0.0, "mean_center_angle_deg": float("nan")}
        nuisance = field.view(-1).long().to(device=labels.device)
        terms = []
        angles = []
        active_tx = 0
        active_groups = 0
        for cls in torch.unique(labels[labels >= 0]):
            cls_mask = labels.eq(cls) & nuisance.ge(0)
            valid_groups = []
            for group in torch.unique(nuisance[cls_mask]):
                group_mask = cls_mask & nuisance.eq(group)
                if int(group_mask.sum().item()) >= max(1, int(min_samples_per_group)):
                    valid_groups.append(group_mask)
            if len(valid_groups) < max(2, int(min_groups)):
                continue
            active_tx += 1
            active_groups += len(valid_groups)
            tx_center = safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            for group_mask in valid_groups:
                group_center = safe_l2_normalize(z_norm[group_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                cos = (group_center * tx_center).sum().clamp(-1.0 + 1e-4, 1.0 - 1e-4)
                terms.append(1.0 - cos)
                angles.append(_safe_angle_from_cos(cos.view(1), eps=1e-4).squeeze(0))
        if not terms:
            return zero, {"active": 0.0, "tx_count": 0.0, "group_count": 0.0, "mean_center_angle_deg": float("nan")}
        loss = torch.stack(terms).mean()
        angle = torch.stack(angles).detach().mean()
        return loss, {
            "active": 1.0,
            "tx_count": float(active_tx),
            "group_count": float(active_groups),
            "mean_center_angle_deg": math.degrees(_scalar_metric(angle)),
        }

    receiver_loss, receiver_info = _field_loss(receiver_labels)
    day_loss, day_info = _field_loss(day_labels)
    channel_loss, channel_info = _field_loss(channel_labels)
    channel_pair_loss = zero_like_with_grad(z_norm)
    channel_pair_angle = z_norm.new_tensor(float("nan"))
    pair_n = max(0, int(paired_view_count))
    if pair_n > 0 and z_norm.size(0) >= 2 * pair_n:
        clean = z_norm[:pair_n]
        satellite = z_norm[pair_n : 2 * pair_n]
        pair_labels_match = labels[:pair_n].eq(labels[pair_n : 2 * pair_n])
        if bool(pair_labels_match.any()):
            pair_cos = (clean[pair_labels_match] * satellite[pair_labels_match]).sum(dim=1).clamp(
                -1.0 + 1e-4, 1.0 - 1e-4
            )
            channel_pair_loss = (1.0 - pair_cos).mean()
            channel_pair_angle = _safe_angle_from_cos(pair_cos.detach(), eps=1e-4).mean()
    total = (
        max(0.0, float(receiver_weight)) * receiver_loss
        + max(0.0, float(day_weight)) * day_loss
        + max(0.0, float(channel_weight))
        * (channel_loss + max(0.0, float(channel_pair_weight)) * channel_pair_loss)
    )
    metrics: Dict[str, float] = {
        "active": max(receiver_info["active"], day_info["active"], channel_info["active"]),
        "receiver_loss": _scalar_metric(receiver_loss.detach()),
        "day_loss": _scalar_metric(day_loss.detach()),
        "channel_loss": _scalar_metric(channel_loss.detach()),
        "channel_pair_loss": _scalar_metric(channel_pair_loss.detach()),
        "channel_pair_count": float(pair_n),
        "channel_pair_angle_deg": math.degrees(_scalar_metric(channel_pair_angle)),
    }
    for prefix, info in (("receiver", receiver_info), ("day", day_info), ("channel", channel_info)):
        for key, value in info.items():
            metrics[f"{prefix}_{key}"] = value
    return total, metrics


def direct_metric_acceptance_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    *,
    paired_view_count: int = 0,
    virtual_count: int = 32,
    virtual_mode: str = "hard",
    core_quantile: float = 0.70,
    accept_quantile: float = 0.80,
    tail_quantile: float = 0.90,
    overflow_quantile: float = 0.97,
    zid_p50_target_rad: float = math.radians(28.0),
    zid_p95_target_rad: float = math.radians(54.0),
    zid_p99_target_rad: float = math.radians(70.0),
    zid_tail_cvar_target_rad: float = math.radians(56.0),
    source_overflow_target: float = 0.45,
    proxy_vaccept_target: float = 0.35,
    bridge_accept_target: float = 0.25,
    low_density_accept_target: float = 0.12,
    tail_accept_target: float = 0.35,
    overflow_accept_target: float = 0.20,
    radius_inter_ratio_target: float = 0.85,
    core_accept_target: float = 0.82,
    core_tpr_target: float = 0.85,
    known_accept_target: float = 0.65,
    known_tpr_target: float = 0.85,
    sat_pair_target_rad: float = math.radians(10.0),
    zid_quantile_weight: float = 1.0,
    source_overflow_weight: float = 1.0,
    proxy_vaccept_weight: float = 1.0,
    bridge_accept_weight: float = 1.0,
    low_density_accept_weight: float = 1.0,
    tail_accept_weight: float = 1.0,
    overflow_accept_weight: float = 1.0,
    radius_inter_ratio_weight: float = 1.0,
    global_quantile_weight: float = 0.0,
    component_inter_margin_weight: float = 0.0,
    component_overlap_weight: float = 0.0,
    core_accept_weight: float = 0.25,
    core_tpr_weight: float = 0.0,
    known_coverage_weight: float = 0.0,
    sat_pair_weight: float = 0.0,
    quantile_temperature_rad: float = math.radians(3.0),
    accept_temperature: float = 0.04,
    component_temperature_rad: float = math.radians(3.0),
    density_temperature_rad: float = math.radians(3.0),
    component_margin_rad: float = math.radians(4.0),
    component_inter_margin_rad: float = math.radians(55.0),
    component_overlap_margin_rad: float = math.radians(4.0),
    source_margin_rad: float = math.radians(2.0),
    source_radius_cap_rad: float = 0.0,
    shell_width_rad: float = math.radians(4.0),
    accept_cvar_alpha: float = 0.25,
    positive_first: bool = False,
    negative_start_tpr: float = 0.75,
    negative_full_tpr: float = 0.85,
    require_effective_negative_grad: bool = False,
    virtual_detach: bool = True,
    gate_reference_detach: bool = True,
    min_classes: int = 2,
    min_samples_per_class: int = 2,
    use_domain_local_components: bool = False,
    require_domain_local_components: bool = False,
    min_samples_per_component: int = 2,
    hierarchical_class_gate: bool = False,
    hierarchical_gate_combine: str = "product",
    reference_z: Optional[torch.Tensor] = None,
    reference_y: Optional[torch.Tensor] = None,
    reference_d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Directly optimize Phase1 source-only open-set proxy geometry metrics.

    This loss does not use target receivers or true unknown transmitters. It
    builds only source-class prototypes, source-domain holdout episodes, and
    virtual negatives synthesized from the current source features.
    """
    default_metrics = {
        "active": 0.0,
        "active_classes": 0.0,
        "zid_p50_deg": float("nan"),
        "zid_p95_deg": float("nan"),
        "zid_p99_deg": float("nan"),
        "zid_tail_cvar_deg": float("nan"),
        "source_overflow": float("nan"),
        "source_overflow_hard": float("nan"),
        "source_overflow_loss": 0.0,
        "proxy_vaccept": float("nan"),
        "proxy_vaccept_loss": 0.0,
        "bridge_accept_rate": float("nan"),
        "bridge_accept_loss": 0.0,
        "shell_accept_rate": float("nan"),
        "outward_accept_rate": float("nan"),
        "low_density_accept_rate": float("nan"),
        "low_density_accept_loss": 0.0,
        "tail_accept_rate": float("nan"),
        "tail_accept_loss": 0.0,
        "overflow_accept_rate": float("nan"),
        "overflow_accept_loss": 0.0,
        "radius_to_inter_ratio": float("nan"),
        "radius_inter_ratio_loss": 0.0,
        "component_inter_margin_loss": 0.0,
        "component_overlap_loss": 0.0,
        "component_min_inter_deg": float("nan"),
        "hierarchical_class_gate": 0.0,
        "global_zid_quantile_loss": 0.0,
        "core_accept_rate": float("nan"),
        "core_accept_loss": 0.0,
        "core_hard_tpr": float("nan"),
        "core_soft_tpr": float("nan"),
        "core_tpr_loss": 0.0,
        "known_accept_rate": float("nan"),
        "known_hard_tpr": float("nan"),
        "known_soft_tpr": float("nan"),
        "known_coverage_loss": 0.0,
        "known_probability_loss": 0.0,
        "known_radius_loss": 0.0,
        "known_margin_loss": 0.0,
        "known_density_loss": 0.0,
        "known_tpr_loss": 0.0,
        "negative_risk_scale": 1.0,
        "proxy_gradient_active": 0.0,
        "query_inter_margin_loss": 0.0,
        "query_overlap_loss": 0.0,
        "sat_pair_angle_p95_deg": float("nan"),
        "sat_pair_loss": 0.0,
        "zid_quantile_loss": 0.0,
        "virtual_count": 0.0,
        "geometry_stabilized": 1.0,
        "geometry_reference_detached": 1.0,
        "virtual_negative_detached": 1.0,
        "gate_reference_detached": 1.0,
        "angle_clamp_eps": 1e-4,
        "softplus_clip": 20.0,
        "domain_local_component_gate": 0.0,
        "global_ball_accept": 1.0,
        "local_component_count": 0.0,
        "local_component_class_coverage": 0.0,
        "local_zid_p50_deg": float("nan"),
        "local_zid_p95_deg": float("nan"),
        "local_zid_p99_deg": float("nan"),
        "local_zid_tail_cvar_deg": float("nan"),
        "reference_anchor_count": 0.0,
        "query_count": 0.0,
    }
    if z is None or not torch.is_tensor(z) or z.numel() == 0:
        ref = torch.tensor(0.0)
        return ref, default_metrics
    if z.dim() != 2:
        raise ValueError(f"direct_metric_acceptance_loss expects 2D features, got shape={tuple(z.shape)}")
    query_labels = y.view(-1).long()
    if query_labels.numel() != z.size(0):
        raise ValueError(f"label count {query_labels.numel()} does not match feature batch {z.size(0)}")

    query_z_norm = safe_l2_normalize(torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    reference_count = 0
    if reference_z is not None:
        if reference_y is None or reference_z.dim() != 2 or reference_z.size(1) != z.size(1):
            raise ValueError("direct metric reference features require aligned rank-2 reference_z/reference_y")
        reference_labels = reference_y.view(-1).long().to(device=query_labels.device)
        if reference_labels.numel() != reference_z.size(0):
            raise ValueError("direct metric reference label count mismatch")
        reference_z_norm = safe_l2_normalize(
            torch.nan_to_num(reference_z.detach().float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1
        )
        reference_count = int(reference_z_norm.size(0))
        z_norm = torch.cat([reference_z_norm, query_z_norm], dim=0)
        labels = torch.cat([reference_labels, query_labels], dim=0)
        if reference_d is not None and d is not None:
            domains = torch.cat(
                [reference_d.view(-1).long().to(device=query_labels.device), d.view(-1).long()], dim=0
            )
        else:
            domains = None
    else:
        z_norm = query_z_norm
        labels = query_labels
        domains = d.view(-1).long() if d is not None and torch.is_tensor(d) else None
    query_full_mask = torch.arange(labels.numel(), device=labels.device) >= int(reference_count)
    fit_full_mask = ~query_full_mask if reference_count > 0 else torch.ones_like(query_full_mask)
    valid = labels >= 0
    centers = []
    class_ids = []
    sample_mask = torch.zeros_like(valid, dtype=torch.bool)
    min_count = max(1, int(min_samples_per_class))
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls) & fit_full_mask
        if int(cls_mask.sum().item()) < min_count:
            continue
        centers.append(safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        class_ids.append(cls)
        sample_mask |= valid & labels.eq(cls)
    active_classes = len(centers)
    default_metrics["active_classes"] = float(active_classes)
    if active_classes < max(1, int(min_classes)):
        return zero_like_with_grad(z), default_metrics

    proto = torch.stack(centers, dim=0)
    angle_eps = 1e-4
    softplus_clip = 20.0
    proto_ref = proto.detach()
    center_labels = torch.stack(class_ids, dim=0).to(device=labels.device)
    sample_z = z_norm[sample_mask]
    sample_labels = labels[sample_mask]
    sample_query_mask = query_full_mask[sample_mask]
    sample_fit_mask = fit_full_mask[sample_mask]
    sample_domains = domains[sample_mask] if domains is not None and domains.numel() == labels.numel() else None
    sample_cos = (sample_z @ proto_ref.t()).clamp(-1.0 + angle_eps, 1.0 - angle_eps)
    own_center = sample_labels.view(-1, 1).eq(center_labels.view(1, -1))
    pos_angles = _safe_angle_from_cos(sample_cos[own_center], eps=angle_eps)
    proto_index_per_sample = own_center.float().argmax(dim=1).long()
    if pos_angles.numel() == 0:
        return zero_like_with_grad(z), default_metrics

    core_q = max(0.0, min(1.0, float(core_quantile)))
    accept_q = max(core_q, min(1.0, float(accept_quantile)))
    tail_q = max(accept_q, min(1.0, float(tail_quantile)))
    overflow_q = max(tail_q, min(1.0, float(overflow_quantile)))
    radius_vec = []
    accept_radius_vec = []
    tail_radius_vec = []
    overflow_radius_vec = []
    core_mask = torch.zeros((sample_z.size(0),), device=sample_z.device, dtype=torch.bool)
    tail_mask = torch.zeros_like(core_mask)
    overflow_mask = torch.zeros_like(core_mask)
    for proto_idx, cls in enumerate(center_labels):
        cls_mask = sample_labels.eq(cls)
        fit_cls_mask = cls_mask & sample_fit_mask
        cls_angles = pos_angles[fit_cls_mask]
        if cls_angles.numel() == 0:
            radius_vec.append(sample_z.new_tensor(math.radians(40.0)))
            accept_radius_vec.append(sample_z.new_tensor(math.radians(40.0)))
            tail_radius_vec.append(sample_z.new_tensor(math.radians(40.0)))
            overflow_radius_vec.append(sample_z.new_tensor(math.radians(40.0)))
            continue
        det = cls_angles.detach()
        core_radius = torch.quantile(det, core_q) if det.numel() > 1 else det.max()
        accept_radius = torch.quantile(det, accept_q) if det.numel() > 1 else det.max()
        tail_radius = torch.quantile(det, tail_q) if det.numel() > 1 else det.max()
        overflow_radius = torch.quantile(det, overflow_q) if det.numel() > 1 else det.max()
        radius_vec.append(core_radius.clamp_min(1e-4))
        accept_radius_vec.append(accept_radius.clamp_min(1e-4))
        tail_radius_vec.append(tail_radius.clamp_min(1e-4))
        overflow_radius_vec.append(overflow_radius.clamp_min(1e-4))
        core_mask |= cls_mask & sample_query_mask & (pos_angles <= core_radius)
        tail_mask |= cls_mask & sample_query_mask & (pos_angles > core_radius) & (pos_angles <= tail_radius)
        overflow_mask |= cls_mask & sample_query_mask & (pos_angles > overflow_radius)
    radius_vec_t = torch.stack(radius_vec, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
    accept_radius_vec_t = torch.stack(accept_radius_vec, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
    tail_radius_vec_t = torch.stack(tail_radius_vec, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
    overflow_radius_vec_t = torch.stack(overflow_radius_vec, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
    if not bool(core_mask.any()):
        core_mask = sample_query_mask.clone()

    gate_proto = proto
    gate_labels = center_labels
    gate_radius = accept_radius_vec_t
    gate_shell_radius = radius_vec_t
    gate_tail_radius = tail_radius_vec_t
    gate_overflow_radius = overflow_radius_vec_t
    gate_core_masks: List[torch.Tensor] = []
    gate_density_radius: List[torch.Tensor] = []
    for proto_idx, cls in enumerate(center_labels):
        cls_mask = sample_labels.eq(cls) & sample_fit_mask
        gate_core_masks.append(cls_mask & (pos_angles <= radius_vec_t[proto_idx].detach()))
        gate_density_radius.append(radius_vec_t[proto_idx].detach())
    class_gate_proto = proto
    class_gate_labels = center_labels
    class_gate_radius = accept_radius_vec_t
    class_gate_core_masks = list(gate_core_masks)
    class_gate_density_radius = list(gate_density_radius)
    local_component_active = False
    local_pos_angles = sample_z.new_zeros((0,))
    if bool(use_domain_local_components) and sample_domains is not None:
        local_centers = []
        local_labels = []
        local_accept_radii = []
        local_shell_radii = []
        local_tail_radii = []
        local_overflow_radii = []
        local_core_masks = []
        local_density_radii = []
        local_angle_parts = []
        component_min = max(1, int(min_samples_per_component))
        for cls in center_labels:
            cls_mask = sample_labels.eq(cls) & sample_fit_mask
            for domain_id in torch.unique(sample_domains[cls_mask]):
                component_mask = cls_mask & sample_domains.eq(domain_id)
                if int(component_mask.sum().item()) < component_min:
                    continue
                component_center = safe_l2_normalize(
                    sample_z[component_mask].mean(dim=0, keepdim=True), dim=1
                ).squeeze(0)
                component_angles = _safe_angle_from_cos(
                    (sample_z[component_mask] * component_center.detach().view(1, -1))
                    .sum(dim=1)
                    .clamp(-1.0 + angle_eps, 1.0 - angle_eps),
                    eps=angle_eps,
                )
                det = component_angles.detach()
                component_core_radius = torch.quantile(det, core_q) if det.numel() > 1 else det.max()
                component_accept_radius = torch.quantile(det, accept_q) if det.numel() > 1 else det.max()
                component_tail_radius = torch.quantile(det, tail_q) if det.numel() > 1 else det.max()
                component_overflow_radius = torch.quantile(det, overflow_q) if det.numel() > 1 else det.max()
                local_centers.append(component_center)
                local_labels.append(cls)
                local_shell_radii.append(component_core_radius.clamp_min(1e-4))
                local_accept_radii.append(component_accept_radius.clamp_min(1e-4))
                local_tail_radii.append(component_tail_radius.clamp_min(1e-4))
                local_overflow_radii.append(component_overflow_radius.clamp_min(1e-4))
                component_core_mask = component_mask.clone()
                component_core_mask[component_mask] = component_angles <= component_core_radius
                local_core_masks.append(component_core_mask)
                local_density_radii.append(component_core_radius.clamp_min(1e-4))
                local_angle_parts.append(component_angles)
        local_class_coverage = len({int(v.item()) for v in local_labels})
        if local_centers and local_class_coverage == active_classes:
            gate_proto = torch.stack(local_centers, dim=0)
            gate_labels = torch.stack(local_labels, dim=0).to(device=labels.device)
            gate_radius = torch.stack(local_accept_radii, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
            gate_shell_radius = torch.stack(local_shell_radii, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
            gate_tail_radius = torch.stack(local_tail_radii, dim=0).to(
                device=sample_z.device, dtype=sample_z.dtype
            )
            gate_overflow_radius = torch.stack(local_overflow_radii, dim=0).to(
                device=sample_z.device, dtype=sample_z.dtype
            )
            gate_core_masks = local_core_masks
            gate_density_radius = local_density_radii
            local_pos_angles = torch.cat(local_angle_parts, dim=0)
            local_component_active = True
        elif bool(require_domain_local_components):
            default_metrics.update(
                {
                    "global_ball_accept": 0.0,
                    "local_component_count": float(len(local_centers)),
                    "local_component_class_coverage": float(local_class_coverage),
                }
            )
            return zero_like_with_grad(z), default_metrics

    gate_proto_ref = gate_proto.detach() if bool(gate_reference_detach) else gate_proto
    class_gate_proto_ref = (
        class_gate_proto.detach() if bool(gate_reference_detach) else class_gate_proto
    )
    if bool(require_domain_local_components) and not local_component_active:
        default_metrics["global_ball_accept"] = 0.0
        return zero_like_with_grad(z), default_metrics

    global_pos_angles = pos_angles[sample_query_mask]
    global_q50 = torch.quantile(global_pos_angles.detach(), 0.50) if global_pos_angles.numel() > 1 else global_pos_angles.detach().mean()
    global_q95 = torch.quantile(global_pos_angles.detach(), 0.95) if global_pos_angles.numel() > 1 else global_pos_angles.detach().mean()
    global_q99 = torch.quantile(global_pos_angles.detach(), 0.99) if global_pos_angles.numel() > 1 else global_pos_angles.detach().mean()
    global_tail_cvar = _top_cvar_mean(global_pos_angles.detach(), 0.05)
    optimization_pos_angles = global_pos_angles
    if local_component_active:
        sample_gate_angles = _safe_angle_from_cos(
            (sample_z @ gate_proto_ref.t()).clamp(-1.0 + angle_eps, 1.0 - angle_eps), eps=angle_eps
        )
        own_component = sample_labels.view(-1, 1).eq(gate_labels.view(1, -1))
        own_local_angles = sample_gate_angles.masked_fill(~own_component, float("inf"))
        local_pos_angles, local_component_index = own_local_angles.min(dim=1)
        optimization_pos_angles = local_pos_angles[sample_query_mask]
        query_local_angles = local_pos_angles
        core_mask = sample_query_mask & (query_local_angles <= gate_shell_radius.detach()[local_component_index])
        tail_mask = sample_query_mask & (query_local_angles > gate_shell_radius.detach()[local_component_index]) & (
            query_local_angles <= gate_tail_radius.detach()[local_component_index]
        )
        overflow_mask = sample_query_mask & (query_local_angles > gate_overflow_radius.detach()[local_component_index])
        if not bool(core_mask.any()):
            core_mask = sample_query_mask.clone()

    tau_q = max(1e-4, float(quantile_temperature_rad))
    tau_a = max(1e-4, float(accept_temperature))
    cvar_frac = max(1e-6, min(1.0, float(accept_cvar_alpha)))

    def _angle_target_loss(angles: torch.Tensor, target_rad: float, frac: float) -> torch.Tensor:
        terms = _bounded_softplus((angles - float(target_rad)) / tau_q, clip=softplus_clip)
        return _top_cvar_mean(terms, frac)

    local_zid_quantile_loss = (
        _angle_target_loss(optimization_pos_angles, float(zid_p50_target_rad), 0.50)
        + _angle_target_loss(optimization_pos_angles, float(zid_p95_target_rad), 0.05)
        + _angle_target_loss(optimization_pos_angles, float(zid_p99_target_rad), 0.01)
        + _angle_target_loss(optimization_pos_angles, float(zid_tail_cvar_target_rad), cvar_frac)
    )
    global_zid_quantile_loss = (
        _angle_target_loss(global_pos_angles, float(zid_p50_target_rad), 0.50)
        + _angle_target_loss(global_pos_angles, float(zid_p95_target_rad), 0.05)
        + _angle_target_loss(global_pos_angles, float(zid_p99_target_rad), 0.01)
        + _angle_target_loss(global_pos_angles, float(zid_tail_cvar_target_rad), cvar_frac)
    )
    zid_quantile_loss = local_zid_quantile_loss + max(
        0.0, float(global_quantile_weight)
    ) * global_zid_quantile_loss

    def _top_angle_mean(frac: float) -> torch.Tensor:
        return _top_cvar_mean(optimization_pos_angles, max(1e-6, min(1.0, float(frac))))

    with torch.no_grad():
        pos_det = optimization_pos_angles.detach()
        q50 = torch.quantile(pos_det, 0.50) if pos_det.numel() > 1 else pos_det.mean()
        q95 = torch.quantile(pos_det, 0.95) if pos_det.numel() > 1 else pos_det.mean()
        q99 = torch.quantile(pos_det, 0.99) if pos_det.numel() > 1 else pos_det.mean()
        tail_cvar_metric = _top_angle_mean(0.05).detach()

    def _geometry_accept_prob(
        feat: torch.Tensor,
        *,
        expected_labels: Optional[torch.Tensor] = None,
        core_radius: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if feat is None or feat.numel() == 0:
            return sample_z.new_zeros((0,)), sample_z.new_zeros((0,))
        feat_norm = safe_l2_normalize(feat, dim=1)
        feat_angles = _safe_angle_from_cos(
            (feat_norm @ gate_proto_ref.t()).clamp(-1.0 + angle_eps, 1.0 - angle_eps),
            eps=angle_eps,
        )
        radius_source = gate_shell_radius if bool(core_radius) else gate_radius
        radius_gate = torch.sigmoid(
            (radius_source.detach().view(1, -1) - feat_angles) / max(1e-4, float(component_temperature_rad))
        )
        margin_columns = []
        density_columns = []
        for component_idx in range(gate_proto_ref.size(0)):
            other_class = gate_labels.ne(gate_labels[component_idx])
            if bool(other_class.any()):
                other_nearest = feat_angles[:, other_class].min(dim=1).values
                class_gap = other_nearest - feat_angles[:, component_idx]
                margin_columns.append(
                    torch.sigmoid(
                        (class_gap - float(component_margin_rad))
                        / max(1e-4, float(component_temperature_rad))
                    )
                )
            else:
                margin_columns.append(torch.ones_like(feat_angles[:, component_idx]))
            component_core = sample_z[gate_core_masks[component_idx]]
            if component_core.numel():
                nearest_core = _safe_angle_from_cos(
                    (feat_norm @ component_core.detach().t())
                    .clamp(-1.0 + angle_eps, 1.0 - angle_eps)
                    .max(dim=1)
                    .values,
                    eps=angle_eps,
                )
                density_columns.append(
                    torch.sigmoid(
                        (gate_density_radius[component_idx].detach() - nearest_core)
                        / max(1e-4, float(density_temperature_rad))
                    )
                )
            else:
                density_columns.append(torch.zeros_like(feat_angles[:, component_idx]))
        margin_gate = torch.stack(margin_columns, dim=1)
        density_gate = torch.stack(density_columns, dim=1)
        component_accept = radius_gate * margin_gate * density_gate
        effective_density = density_gate
        if bool(hierarchical_class_gate) and local_component_active:
            class_angles = _safe_angle_from_cos(
                (feat_norm @ class_gate_proto_ref.t()).clamp(
                    -1.0 + angle_eps, 1.0 - angle_eps
                ),
                eps=angle_eps,
            )
            class_radius_gate = torch.sigmoid(
                (
                    class_gate_radius.detach().view(1, -1)
                    - class_angles
                )
                / max(1e-4, float(component_temperature_rad))
            )
            class_margin_columns = []
            class_density_columns = []
            for class_idx in range(class_gate_proto_ref.size(0)):
                other_class = class_gate_labels.ne(class_gate_labels[class_idx])
                if bool(other_class.any()):
                    other_nearest = class_angles[:, other_class].min(dim=1).values
                    class_gap = other_nearest - class_angles[:, class_idx]
                    class_margin_columns.append(
                        torch.sigmoid(
                            (class_gap - float(component_margin_rad))
                            / max(1e-4, float(component_temperature_rad))
                        )
                    )
                else:
                    class_margin_columns.append(torch.ones_like(class_angles[:, class_idx]))
                class_core = sample_z[class_gate_core_masks[class_idx]]
                if class_core.numel():
                    nearest_class_core = _safe_angle_from_cos(
                        (feat_norm @ class_core.detach().t())
                        .clamp(-1.0 + angle_eps, 1.0 - angle_eps)
                        .max(dim=1)
                        .values,
                        eps=angle_eps,
                    )
                    class_density_columns.append(
                        torch.sigmoid(
                            (
                                class_gate_density_radius[class_idx].detach()
                                - nearest_class_core
                            )
                            / max(1e-4, float(density_temperature_rad))
                        )
                    )
                else:
                    class_density_columns.append(torch.zeros_like(class_angles[:, class_idx]))
            class_margin_gate = torch.stack(class_margin_columns, dim=1)
            class_density_gate = torch.stack(class_density_columns, dim=1)
            class_accept = class_radius_gate * class_margin_gate * class_density_gate
            class_index = torch.stack(
                [
                    torch.where(class_gate_labels.eq(label))[0][0]
                    for label in gate_labels
                ]
            ).to(device=gate_labels.device)
            class_component_accept = class_accept[:, class_index]
            class_component_density = class_density_gate[:, class_index]
            if str(hierarchical_gate_combine).strip().lower() == "smooth_min":
                # A calibrated smooth minimum keeps the deployment-like AND
                # semantics without the product gate's reject-all shortcut.
                smooth_eps = 1e-6
                component_accept = 0.5 * (
                    component_accept
                    + class_component_accept
                    - torch.sqrt(
                        (component_accept - class_component_accept).pow(2)
                        + smooth_eps
                    )
                ).clamp(0.0, 1.0)
                effective_density = 0.5 * (
                    effective_density
                    + class_component_density
                    - torch.sqrt(
                        (effective_density - class_component_density).pow(2)
                        + smooth_eps
                    )
                ).clamp(0.0, 1.0)
            else:
                component_accept = component_accept * class_component_accept
                effective_density = effective_density * class_component_density
        if expected_labels is not None:
            expected = expected_labels.view(-1).long().to(device=gate_labels.device)
            if expected.numel() != component_accept.size(0):
                raise ValueError("expected_labels must align with direct metric query features")
            expected_component = expected.view(-1, 1).eq(gate_labels.view(1, -1))
            component_accept = component_accept.masked_fill(~expected_component, 0.0)
            effective_density = effective_density.masked_fill(~expected_component, 0.0)
        low_density_prob = 1.0 - effective_density.max(dim=1).values
        return component_accept.max(dim=1).values, low_density_prob

    def _accept_loss(prob: torch.Tensor, target: float) -> torch.Tensor:
        if prob.numel() == 0:
            return sample_z.new_tensor(0.0)
        return _top_cvar_mean(_bounded_softplus((prob - float(target)) / tau_a, clip=softplus_clip), cvar_frac)

    core_prob, _ = _geometry_accept_prob(
        sample_z[core_mask],
        expected_labels=sample_labels[core_mask],
        core_radius=True,
    )
    core_accept_loss = (
        _top_cvar_mean(_bounded_softplus((float(core_accept_target) - core_prob) / tau_a, clip=softplus_clip), cvar_frac)
        if core_prob.numel()
        else sample_z.new_tensor(0.0)
    )
    core_soft_tpr_tensor = (
        torch.sigmoid((core_prob - 0.5) / tau_a).mean()
        if core_prob.numel()
        else sample_z.new_tensor(0.0)
    )
    core_tpr_loss = F.relu(float(core_tpr_target) - core_soft_tpr_tensor).pow(2)
    known_prob, _ = _geometry_accept_prob(
        sample_z[sample_query_mask],
        expected_labels=sample_labels[sample_query_mask],
        core_radius=True,
    )
    known_probability_loss = (
        _top_cvar_mean(
            _bounded_softplus(
                (float(known_accept_target) - known_prob) / tau_a,
                clip=softplus_clip,
            ),
            cvar_frac,
        )
        if known_prob.numel()
        else sample_z.new_tensor(0.0)
    )
    known_radius_loss = sample_z.new_tensor(0.0)
    known_margin_loss = sample_z.new_tensor(0.0)
    known_density_loss = sample_z.new_tensor(0.0)
    known_query_z = sample_z[sample_query_mask]
    known_query_labels = sample_labels[sample_query_mask]
    if known_query_z.numel():
        known_angles = _safe_angle_from_cos(
            (known_query_z @ gate_proto_ref.t()).clamp(
                -1.0 + angle_eps, 1.0 - angle_eps
            ),
            eps=angle_eps,
        )
        known_own_component = known_query_labels.view(-1, 1).eq(
            gate_labels.view(1, -1)
        )
        known_own_angles = known_angles.masked_fill(
            ~known_own_component, float("inf")
        )
        known_own_angle, known_own_index = known_own_angles.min(dim=1)
        known_radius_loss = _top_cvar_mean(
            _bounded_softplus(
                (
                    known_own_angle
                    - gate_shell_radius.detach()[known_own_index]
                )
                / tau_q,
                clip=softplus_clip,
            ),
            cvar_frac,
        )
        known_other_angle = known_angles.masked_fill(
            known_own_component, float("inf")
        ).min(dim=1).values
        known_margin_loss = _top_cvar_mean(
            _bounded_softplus(
                (
                    float(component_margin_rad)
                    - (known_other_angle - known_own_angle)
                )
                / tau_q,
                clip=softplus_clip,
            ),
            cvar_frac,
        )
        known_density_angles = []
        known_density_radii = []
        for row_idx, component_idx in enumerate(known_own_index):
            component_i = int(component_idx.detach().item())
            component_core = sample_z[gate_core_masks[component_i]]
            if component_core.numel():
                nearest_density_angle = _safe_angle_from_cos(
                    (
                        known_query_z[row_idx : row_idx + 1]
                        @ component_core.detach().t()
                    )
                    .clamp(-1.0 + angle_eps, 1.0 - angle_eps)
                    .max(dim=1)
                    .values,
                    eps=angle_eps,
                ).squeeze(0)
                known_density_angles.append(nearest_density_angle)
                known_density_radii.append(
                    gate_density_radius[component_i].detach()
                )
        if known_density_angles:
            known_density_angle_t = torch.stack(known_density_angles)
            known_density_radius_t = torch.stack(known_density_radii)
            known_density_loss = _top_cvar_mean(
                _bounded_softplus(
                    (known_density_angle_t - known_density_radius_t) / tau_q,
                    clip=softplus_clip,
                ),
                cvar_frac,
            )
    known_coverage_loss = (
        known_probability_loss
        + known_radius_loss
        + known_margin_loss
        + known_density_loss
    )
    known_soft_tpr_tensor = (
        torch.sigmoid((known_prob - 0.5) / tau_a).mean()
        if known_prob.numel()
        else sample_z.new_tensor(0.0)
    )
    known_hard_tpr_tensor = (
        (known_prob.detach() >= 0.5).float().mean()
        if known_prob.numel()
        else sample_z.new_tensor(0.0)
    )
    known_tpr_loss = F.relu(float(known_tpr_target) - known_soft_tpr_tensor).pow(2)
    negative_risk_scale = sample_z.new_tensor(1.0)
    if bool(positive_first):
        start_tpr = min(float(negative_start_tpr), float(negative_full_tpr) - 1e-4)
        full_tpr = max(float(negative_full_tpr), start_tpr + 1e-4)
        coverage_control_tpr = torch.minimum(
            known_soft_tpr_tensor.detach(), known_hard_tpr_tensor.detach()
        )
        negative_risk_scale = (
            (coverage_control_tpr - start_tpr) / (full_tpr - start_tpr)
        ).clamp(0.0, 1.0)
    tail_prob, tail_low_density_prob = _geometry_accept_prob(sample_z[tail_mask])
    overflow_prob, overflow_low_density_prob = _geometry_accept_prob(sample_z[overflow_mask])
    tail_accept_loss = _accept_loss(tail_prob, float(tail_accept_target))
    overflow_accept_loss = _accept_loss(overflow_prob, float(overflow_accept_target))

    virtual_geometry_basis = gate_proto.detach() if bool(virtual_detach) else gate_proto
    virtual_source_mask = sample_fit_mask
    if not bool(virtual_detach) and bool(sample_query_mask.any()):
        virtual_source_mask = sample_query_mask
    virtual_parts = _make_proxy_virtual_unknown_pool(
        sample_z[virtual_source_mask],
        sample_labels[virtual_source_mask],
        gate_labels,
        virtual_geometry_basis,
        gate_shell_radius,
        count=max(0, int(virtual_count)),
        mode=str(virtual_mode or "hard"),
        shell_width_rad=float(shell_width_rad),
    )
    if not bool(virtual_detach) and "bridge" in virtual_parts:
        virtual_parts["bridge"] = _make_query_interclass_bridge_outliers(
            sample_z[virtual_source_mask],
            sample_labels[virtual_source_mask],
            gate_labels,
            gate_proto_ref,
            count=int(virtual_parts["bridge"].size(0)),
        )
    if bool(virtual_detach):
        virtual_parts = {name: part.detach() for name, part in virtual_parts.items()}
    virtual_tensors = [part for part in virtual_parts.values() if part.numel() > 0]
    virtual_all = torch.cat(virtual_tensors, dim=0) if virtual_tensors else sample_z.new_zeros((0, sample_z.size(1)))
    virtual_prob, virtual_low_density_prob = _geometry_accept_prob(virtual_all)
    proxy_vaccept_loss = _accept_loss(virtual_prob, float(proxy_vaccept_target))

    def _part_prob(name: str) -> torch.Tensor:
        part = virtual_parts.get(name)
        if part is None or part.numel() == 0:
            return sample_z.new_zeros((0,))
        prob, _ = _geometry_accept_prob(part)
        return prob

    bridge_prob = _part_prob("bridge")
    shell_prob = _part_prob("shell")
    outward_prob = _part_prob("outward")
    bridge_accept_loss = _accept_loss(bridge_prob, float(bridge_accept_target))
    density_terms = []
    if virtual_prob.numel():
        density_terms.append(virtual_prob * virtual_low_density_prob)
    if tail_prob.numel():
        density_terms.append(tail_prob * tail_low_density_prob)
    if overflow_prob.numel():
        density_terms.append(overflow_prob * overflow_low_density_prob)
    low_density_terms = torch.cat(density_terms, dim=0) if density_terms else sample_z.new_zeros((0,))
    low_density_accept_loss = _accept_loss(low_density_terms, float(low_density_accept_target))

    radius_inter_ratio_loss = sample_z.new_tensor(0.0)
    component_inter_margin_loss = sample_z.new_tensor(0.0)
    component_overlap_loss = sample_z.new_tensor(0.0)
    query_inter_margin_loss = sample_z.new_tensor(0.0)
    query_overlap_loss = sample_z.new_tensor(0.0)
    radius_inter_ratio_metric = sample_z.new_tensor(float("nan"))
    component_min_inter = sample_z.new_tensor(float("nan"))
    if gate_proto.size(0) > 1:
        inter_angles = _safe_angle_from_cos(
            (gate_proto_ref @ gate_proto_ref.t()).clamp(-1.0 + angle_eps, 1.0 - angle_eps),
            eps=angle_eps,
        )
        other_class = gate_labels.view(-1, 1).ne(gate_labels.view(1, -1))
        nearest_inter = inter_angles.masked_fill(~other_class, float("inf")).min(dim=1).values.clamp_min(1e-4)
        component_min_inter = nearest_inter.detach().min()
        query_sample_z = sample_z[sample_query_mask]
        query_sample_labels = sample_labels[sample_query_mask]
        sample_gate_angles = _safe_angle_from_cos(
            (query_sample_z @ gate_proto_ref.t()).clamp(-1.0 + angle_eps, 1.0 - angle_eps), eps=angle_eps
        )
        own_component = query_sample_labels.view(-1, 1).eq(gate_labels.view(1, -1))
        own_angles = sample_gate_angles.masked_fill(~own_component, float("inf"))
        own_index = own_angles.argmin(dim=1)
        nearest_own_angle = own_angles.gather(1, own_index.view(-1, 1)).squeeze(1)
        other_angles = sample_gate_angles.masked_fill(own_component, float("inf"))
        nearest_other_angle = other_angles.min(dim=1).values
        query_class_gap = nearest_other_angle - nearest_own_angle
        query_inter_margin_loss = _top_cvar_mean(
            F.relu(float(component_inter_margin_rad) - query_class_gap).pow(2),
            cvar_frac,
        )
        sample_inter = nearest_inter[own_index]
        ratio = nearest_own_angle / sample_inter
        sample_ratio_loss = _top_cvar_mean(
            _bounded_softplus((ratio - float(radius_inter_ratio_target)) / tau_a, clip=softplus_clip),
            cvar_frac,
        )
        component_ratio = gate_radius.detach() / nearest_inter
        radius_inter_ratio_metric = (
            component_ratio.detach().max()
            if component_ratio.numel()
            else sample_z.new_tensor(float("nan"))
        )
        component_ratio_loss = _top_cvar_mean(
            _bounded_softplus(
                (component_ratio - float(radius_inter_ratio_target)) / tau_a,
                clip=softplus_clip,
            ),
            cvar_frac,
        )
        radius_inter_ratio_loss = sample_ratio_loss + component_ratio_loss
        reference_inter_margin_loss = _top_cvar_mean(
            F.relu(float(component_inter_margin_rad) - nearest_inter).pow(2),
            cvar_frac,
        )
        component_inter_margin_loss = reference_inter_margin_loss + query_inter_margin_loss
        pair_radius = gate_radius.detach().view(-1, 1) + gate_radius.detach().view(1, -1)
        overlap = F.relu(
            pair_radius + float(component_overlap_margin_rad) - inter_angles
        ).masked_select(other_class)
        reference_overlap_loss = (
            _top_cvar_mean(overlap.pow(2), cvar_frac)
            if overlap.numel()
            else sample_z.new_tensor(0.0)
        )
        query_overlap = F.relu(
            nearest_own_angle
            + gate_radius.detach()[own_index]
            + float(component_overlap_margin_rad)
            - nearest_other_angle
        )
        query_overlap_loss = _top_cvar_mean(query_overlap.pow(2), cvar_frac)
        component_overlap_loss = reference_overlap_loss + query_overlap_loss

    source_probs = []
    source_hard_overflow = []
    if sample_domains is not None:
        for cls in center_labels:
            cls_fit = sample_labels.eq(cls) & sample_fit_mask
            cls_query = sample_labels.eq(cls) & sample_query_mask
            class_domains = torch.unique(sample_domains[cls_fit])
            if class_domains.numel() < 2 or not bool(cls_query.any()):
                continue
            for dom in class_domains:
                support = cls_fit & sample_domains.eq(dom)
                query = cls_query & (~sample_domains.eq(dom))
                if int(support.sum().item()) < min_count or not bool(query.any()):
                    continue
                support_center = safe_l2_normalize(sample_z[support].mean(dim=0, keepdim=True), dim=1).squeeze(0).detach()
                support_angles = _safe_angle_from_cos(
                    (sample_z[support] * support_center.view(1, -1)).sum(dim=1).clamp(-1.0 + angle_eps, 1.0 - angle_eps),
                    eps=angle_eps,
                )
                support_radius = (
                    torch.quantile(support_angles.detach(), core_q)
                    if support_angles.numel() > 1
                    else support_angles.detach().max()
                ) + max(0.0, float(source_margin_rad))
                if float(source_radius_cap_rad) > 0.0:
                    support_radius = support_radius.clamp_max(float(source_radius_cap_rad))
                query_angles = _safe_angle_from_cos(
                    (sample_z[query] * support_center.view(1, -1)).sum(dim=1).clamp(-1.0 + angle_eps, 1.0 - angle_eps),
                    eps=angle_eps,
                )
                source_probs.append(torch.sigmoid((query_angles - support_radius) / tau_q))
                source_hard_overflow.append(query_angles.detach() > support_radius.detach())
    source_overflow_prob = torch.cat(source_probs, dim=0) if source_probs else sample_z.new_zeros((0,))
    source_overflow_loss = _accept_loss(source_overflow_prob, float(source_overflow_target))

    sat_pair_loss = sample_z.new_tensor(0.0)
    sat_pair_p95 = sample_z.new_tensor(float("nan"))
    pair_n = max(0, int(paired_view_count))
    if pair_n > 0 and z_norm.size(0) >= 2 * pair_n:
        clean = z_norm[:pair_n]
        sat = z_norm[pair_n : 2 * pair_n]
        pair_angles = _safe_angle_from_cos((clean * sat).sum(dim=1).clamp(-1.0 + angle_eps, 1.0 - angle_eps), eps=angle_eps)
        sat_pair_loss = _top_cvar_mean(
            _bounded_softplus((pair_angles - float(sat_pair_target_rad)) / tau_q, clip=softplus_clip),
            cvar_frac,
        )
        sat_pair_p95 = torch.quantile(pair_angles.detach(), 0.95) if pair_angles.numel() > 1 else pair_angles.detach().mean()

    if (
        bool(require_effective_negative_grad)
        and float(negative_risk_scale.detach().item()) > 0.0
        and virtual_all.numel() > 0
        and (float(proxy_vaccept_weight) > 0.0 or float(bridge_accept_weight) > 0.0)
        and not bool(proxy_vaccept_loss.requires_grad or bridge_accept_loss.requires_grad)
    ):
        raise RuntimeError(
            "direct metric proxy/bridge risk has no gradient; set "
            "--direct_metric_virtual_detach false or enable a trainable gate reference"
        )

    negative_risk_loss = negative_risk_scale * (
        max(0.0, float(proxy_vaccept_weight)) * proxy_vaccept_loss
        + max(0.0, float(bridge_accept_weight)) * bridge_accept_loss
        + max(0.0, float(low_density_accept_weight)) * low_density_accept_loss
        + max(0.0, float(tail_accept_weight)) * tail_accept_loss
        + max(0.0, float(overflow_accept_weight)) * overflow_accept_loss
    )
    loss = (
        max(0.0, float(zid_quantile_weight)) * zid_quantile_loss
        + max(0.0, float(source_overflow_weight)) * source_overflow_loss
        + negative_risk_loss
        + max(0.0, float(radius_inter_ratio_weight)) * radius_inter_ratio_loss
        + max(0.0, float(component_inter_margin_weight)) * component_inter_margin_loss
        + max(0.0, float(component_overlap_weight)) * component_overlap_loss
        + max(0.0, float(core_accept_weight)) * core_accept_loss
        + max(0.0, float(core_tpr_weight)) * core_tpr_loss
        + max(0.0, float(known_coverage_weight))
        * (known_coverage_loss + known_tpr_loss)
        + max(0.0, float(sat_pair_weight)) * sat_pair_loss
    )

    def _mean_prob(prob: torch.Tensor) -> float:
        return _scalar_metric(prob.detach().mean()) if prob.numel() else float("nan")

    metrics = {
        "active": 1.0,
        "active_classes": float(active_classes),
        "zid_p50_deg": math.degrees(_scalar_metric(q50)),
        "zid_p95_deg": math.degrees(_scalar_metric(q95)),
        "zid_p99_deg": math.degrees(_scalar_metric(q99)),
        "zid_tail_cvar_deg": math.degrees(_scalar_metric(tail_cvar_metric)),
        "source_overflow": _mean_prob(source_overflow_prob),
        "source_overflow_loss": _scalar_metric(source_overflow_loss),
        "proxy_vaccept": _mean_prob(virtual_prob),
        "proxy_vaccept_loss": _scalar_metric(proxy_vaccept_loss),
        "bridge_accept_rate": _mean_prob(bridge_prob),
        "bridge_accept_loss": _scalar_metric(bridge_accept_loss),
        "shell_accept_rate": _mean_prob(shell_prob),
        "outward_accept_rate": _mean_prob(outward_prob),
        "low_density_accept_rate": _mean_prob(low_density_terms),
        "low_density_accept_loss": _scalar_metric(low_density_accept_loss),
        "tail_accept_rate": _mean_prob(tail_prob),
        "tail_accept_loss": _scalar_metric(tail_accept_loss),
        "overflow_accept_rate": _mean_prob(overflow_prob),
        "overflow_accept_loss": _scalar_metric(overflow_accept_loss),
        "radius_to_inter_ratio": _scalar_metric(radius_inter_ratio_metric),
        "radius_inter_ratio_loss": _scalar_metric(radius_inter_ratio_loss),
        "component_inter_margin_loss": _scalar_metric(component_inter_margin_loss),
        "component_overlap_loss": _scalar_metric(component_overlap_loss),
        "query_inter_margin_loss": _scalar_metric(query_inter_margin_loss),
        "query_overlap_loss": _scalar_metric(query_overlap_loss),
        "component_min_inter_deg": math.degrees(_scalar_metric(component_min_inter)),
        "hierarchical_class_gate": 1.0 if bool(hierarchical_class_gate) else 0.0,
        "global_zid_quantile_loss": _scalar_metric(global_zid_quantile_loss),
        "core_accept_rate": _mean_prob(core_prob),
        "core_accept_loss": _scalar_metric(core_accept_loss),
        "core_hard_tpr": (
            _scalar_metric((core_prob.detach() >= 0.5).float().mean())
            if core_prob.numel()
            else float("nan")
        ),
        "core_soft_tpr": _scalar_metric(core_soft_tpr_tensor.detach()),
        "core_tpr_loss": _scalar_metric(core_tpr_loss),
        "known_accept_rate": _mean_prob(known_prob),
        "known_hard_tpr": (
            _scalar_metric(known_hard_tpr_tensor)
            if known_prob.numel()
            else float("nan")
        ),
        "known_soft_tpr": _scalar_metric(known_soft_tpr_tensor.detach()),
        "known_coverage_loss": _scalar_metric(known_coverage_loss),
        "known_probability_loss": _scalar_metric(known_probability_loss),
        "known_radius_loss": _scalar_metric(known_radius_loss),
        "known_margin_loss": _scalar_metric(known_margin_loss),
        "known_density_loss": _scalar_metric(known_density_loss),
        "known_tpr_loss": _scalar_metric(known_tpr_loss),
        "negative_risk_scale": _scalar_metric(negative_risk_scale),
        "proxy_gradient_active": 1.0
        if bool(proxy_vaccept_loss.requires_grad or bridge_accept_loss.requires_grad)
        else 0.0,
        "source_overflow_hard": (
            _scalar_metric(torch.cat(source_hard_overflow).float().mean())
            if source_hard_overflow
            else float("nan")
        ),
        "sat_pair_angle_p95_deg": math.degrees(_scalar_metric(sat_pair_p95)),
        "sat_pair_loss": _scalar_metric(sat_pair_loss),
        "zid_quantile_loss": _scalar_metric(zid_quantile_loss),
        "virtual_count": float(int(virtual_all.size(0))),
        "geometry_stabilized": 1.0,
        # Retain the legacy field semantics while exposing the two independent
        # gradient controls explicitly.
        "geometry_reference_detached": 1.0 if bool(virtual_detach) else 0.0,
        "virtual_negative_detached": 1.0 if bool(virtual_detach) else 0.0,
        "gate_reference_detached": 1.0 if bool(gate_reference_detach) else 0.0,
        "angle_clamp_eps": float(angle_eps),
        "softplus_clip": float(softplus_clip),
        "domain_local_component_gate": 1.0 if local_component_active else 0.0,
        "global_ball_accept": 0.0 if local_component_active else 1.0,
        "local_component_count": float(int(gate_proto.size(0))) if local_component_active else 0.0,
        "local_component_class_coverage": float(active_classes) if local_component_active else 0.0,
        "local_zid_p50_deg": math.degrees(_scalar_metric(torch.quantile(optimization_pos_angles.detach(), 0.50))) if optimization_pos_angles.numel() else float("nan"),
        "local_zid_p95_deg": math.degrees(_scalar_metric(torch.quantile(optimization_pos_angles.detach(), 0.95))) if optimization_pos_angles.numel() else float("nan"),
        "local_zid_p99_deg": math.degrees(_scalar_metric(torch.quantile(optimization_pos_angles.detach(), 0.99))) if optimization_pos_angles.numel() else float("nan"),
        "local_zid_tail_cvar_deg": math.degrees(_scalar_metric(_top_cvar_mean(optimization_pos_angles.detach(), 0.05))) if optimization_pos_angles.numel() else float("nan"),
        "reference_anchor_count": float(reference_count),
        "query_count": float(int(sample_query_mask.sum().item())),
        "global_diag_zid_p50_deg": math.degrees(_scalar_metric(global_q50)),
        "global_diag_zid_p95_deg": math.degrees(_scalar_metric(global_q95)),
        "global_diag_zid_p99_deg": math.degrees(_scalar_metric(global_q99)),
        "global_diag_zid_tail_cvar_deg": math.degrees(_scalar_metric(global_tail_cvar)),
        "quantile_optimization_scope_local": 1.0 if local_component_active else 0.0,
    }
    return loss, metrics


def multiview_direct_metric_acceptance_loss(
    clean_z: torch.Tensor,
    sat_z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    *,
    clean_weight: float = 1.0,
    sat_weight: float = 1.0,
    pair_weight: float = 1.0,
    sat_pair_target_rad: float = math.radians(10.0),
    quantile_temperature_rad: float = math.radians(3.0),
    accept_cvar_alpha: float = 0.25,
    virtual_detach: bool = True,
    gate_reference_detach: bool = True,
    clean_reference_z: Optional[torch.Tensor] = None,
    sat_reference_z: Optional[torch.Tensor] = None,
    reference_y: Optional[torch.Tensor] = None,
    reference_d: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Optimize clean and satellite known geometry without one pooled class ball."""

    if clean_z.dim() != 2 or sat_z.dim() != 2 or clean_z.shape != sat_z.shape:
        raise ValueError(
            "multiview_direct_metric_acceptance_loss expects aligned clean/sat [B,D] tensors, "
            f"got clean={tuple(clean_z.shape)} sat={tuple(sat_z.shape)}"
        )
    if y.view(-1).numel() != clean_z.size(0):
        raise ValueError("multiview labels must have one value per clean/satellite pair")

    base_kwargs = dict(kwargs)
    base_kwargs.pop("paired_view_count", None)
    base_kwargs.pop("sat_pair_weight", None)
    base_kwargs.pop("sat_pair_target_rad", None)
    base_kwargs["quantile_temperature_rad"] = float(quantile_temperature_rad)
    base_kwargs["accept_cvar_alpha"] = float(accept_cvar_alpha)
    base_kwargs["virtual_detach"] = bool(virtual_detach)
    base_kwargs["gate_reference_detach"] = bool(gate_reference_detach)
    clean_loss, clean_metrics = direct_metric_acceptance_loss(
        clean_z,
        y,
        d,
        paired_view_count=0,
        sat_pair_weight=0.0,
        reference_z=clean_reference_z,
        reference_y=reference_y,
        reference_d=reference_d,
        **base_kwargs,
    )
    sat_loss, sat_metrics = direct_metric_acceptance_loss(
        sat_z,
        y,
        d,
        paired_view_count=0,
        sat_pair_weight=0.0,
        reference_z=sat_reference_z,
        reference_y=reference_y,
        reference_d=reference_d,
        **base_kwargs,
    )

    clean_norm = safe_l2_normalize(torch.nan_to_num(clean_z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    sat_norm = safe_l2_normalize(torch.nan_to_num(sat_z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    pair_angles = _safe_angle_from_cos(
        (clean_norm * sat_norm).sum(dim=1).clamp(-1.0 + 1e-4, 1.0 - 1e-4),
        eps=1e-4,
    )
    tau = max(1e-4, float(quantile_temperature_rad))
    pair_loss = _top_cvar_mean(
        _bounded_softplus((pair_angles - float(sat_pair_target_rad)) / tau),
        max(1e-6, min(1.0, float(accept_cvar_alpha))),
    )
    pair_p95 = torch.quantile(pair_angles.detach(), 0.95) if pair_angles.numel() > 1 else pair_angles.detach().mean()
    loss = (
        max(0.0, float(clean_weight)) * clean_loss
        + max(0.0, float(sat_weight)) * sat_loss
        + max(0.0, float(pair_weight)) * pair_loss
    )

    def _worst(key: str) -> float:
        values = []
        for obj in (clean_metrics, sat_metrics):
            try:
                value = float(obj.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
        return max(values) if values else float("nan")

    def _weighted_view_sum(key: str) -> float:
        total = 0.0
        found = False
        for weight, obj in ((clean_weight, clean_metrics), (sat_weight, sat_metrics)):
            try:
                value = float(obj.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if math.isfinite(value):
                total += max(0.0, float(weight)) * value
                found = True
        return total if found else float("nan")

    metrics: Dict[str, float] = {
        "active": min(float(clean_metrics.get("active", 0.0)), float(sat_metrics.get("active", 0.0))),
        "active_classes": min(
            float(clean_metrics.get("active_classes", 0.0)),
            float(sat_metrics.get("active_classes", 0.0)),
        ),
        "sat_pair_angle_p95_deg": math.degrees(_scalar_metric(pair_p95)),
        "sat_pair_loss": _scalar_metric(pair_loss),
        "multiview_separate_geometry": 1.0,
        "multiview_telemetry_aggregated": 1.0,
        "geometry_stabilized": min(
            float(clean_metrics.get("geometry_stabilized", 0.0)),
            float(sat_metrics.get("geometry_stabilized", 0.0)),
        ),
        "geometry_reference_detached": min(
            float(clean_metrics.get("geometry_reference_detached", 0.0)),
            float(sat_metrics.get("geometry_reference_detached", 0.0)),
        ),
        "virtual_negative_detached": min(
            float(clean_metrics.get("virtual_negative_detached", 0.0)),
            float(sat_metrics.get("virtual_negative_detached", 0.0)),
        ),
        "gate_reference_detached": min(
            float(clean_metrics.get("gate_reference_detached", 0.0)),
            float(sat_metrics.get("gate_reference_detached", 0.0)),
        ),
        "domain_local_component_gate": min(
            float(clean_metrics.get("domain_local_component_gate", 0.0)),
            float(sat_metrics.get("domain_local_component_gate", 0.0)),
        ),
        "global_ball_accept": max(
            float(clean_metrics.get("global_ball_accept", 1.0)),
            float(sat_metrics.get("global_ball_accept", 1.0)),
        ),
        "local_component_count": float(clean_metrics.get("local_component_count", 0.0))
        + float(sat_metrics.get("local_component_count", 0.0)),
        "local_component_class_coverage": min(
            float(clean_metrics.get("local_component_class_coverage", 0.0)),
            float(sat_metrics.get("local_component_class_coverage", 0.0)),
        ),
        "component_min_inter_deg": min(
            float(clean_metrics.get("component_min_inter_deg", float("inf"))),
            float(sat_metrics.get("component_min_inter_deg", float("inf"))),
        ),
        "hierarchical_class_gate": min(
            float(clean_metrics.get("hierarchical_class_gate", 0.0)),
            float(sat_metrics.get("hierarchical_class_gate", 0.0)),
        ),
        "reference_anchor_count": min(
            float(clean_metrics.get("reference_anchor_count", 0.0)),
            float(sat_metrics.get("reference_anchor_count", 0.0)),
        ),
        "query_count": min(
            float(clean_metrics.get("query_count", 0.0)),
            float(sat_metrics.get("query_count", 0.0)),
        ),
        "known_accept_rate": min(
            float(clean_metrics.get("known_accept_rate", float("nan"))),
            float(sat_metrics.get("known_accept_rate", float("nan"))),
        ),
        "known_hard_tpr": min(
            float(clean_metrics.get("known_hard_tpr", float("nan"))),
            float(sat_metrics.get("known_hard_tpr", float("nan"))),
        ),
        "known_soft_tpr": min(
            float(clean_metrics.get("known_soft_tpr", float("nan"))),
            float(sat_metrics.get("known_soft_tpr", float("nan"))),
        ),
        "negative_risk_scale": min(
            float(clean_metrics.get("negative_risk_scale", 1.0)),
            float(sat_metrics.get("negative_risk_scale", 1.0)),
        ),
        "proxy_gradient_active": min(
            float(clean_metrics.get("proxy_gradient_active", 0.0)),
            float(sat_metrics.get("proxy_gradient_active", 0.0)),
        ),
    }
    conservative_keys = (
        "zid_p50_deg",
        "zid_p95_deg",
        "zid_p99_deg",
        "zid_tail_cvar_deg",
        "source_overflow",
        "proxy_vaccept",
        "bridge_accept_rate",
        "low_density_accept_rate",
        "tail_accept_rate",
        "overflow_accept_rate",
        "radius_to_inter_ratio",
        "local_zid_p50_deg",
        "local_zid_p95_deg",
        "local_zid_p99_deg",
        "local_zid_tail_cvar_deg",
        "global_diag_zid_p50_deg",
        "global_diag_zid_p95_deg",
        "global_diag_zid_p99_deg",
        "global_diag_zid_tail_cvar_deg",
    )
    for key in conservative_keys:
        metrics[key] = _worst(key)
    metrics["core_accept_rate"] = min(
        float(clean_metrics.get("core_accept_rate", float("nan"))),
        float(sat_metrics.get("core_accept_rate", float("nan"))),
    )
    weighted_loss_keys = (
        "zid_quantile_loss",
        "source_overflow_loss",
        "proxy_vaccept_loss",
        "bridge_accept_loss",
        "low_density_accept_loss",
        "tail_accept_loss",
        "overflow_accept_loss",
        "radius_inter_ratio_loss",
        "component_inter_margin_loss",
        "component_overlap_loss",
        "global_zid_quantile_loss",
        "core_accept_loss",
        "known_coverage_loss",
        "known_probability_loss",
        "known_radius_loss",
        "known_margin_loss",
        "known_density_loss",
        "known_tpr_loss",
        "query_inter_margin_loss",
        "query_overlap_loss",
    )
    for key in weighted_loss_keys:
        metrics[key] = _weighted_view_sum(key)
    metrics["virtual_count"] = float(clean_metrics.get("virtual_count", 0.0)) + float(
        sat_metrics.get("virtual_count", 0.0)
    )
    for prefix, obj in (("clean", clean_metrics), ("sat", sat_metrics)):
        for key, value in obj.items():
            metrics[f"{prefix}_{key}"] = value
    metrics["quantile_optimization_scope_local"] = min(
        float(clean_metrics.get("quantile_optimization_scope_local", 0.0)),
        float(sat_metrics.get("quantile_optimization_scope_local", 0.0)),
    )
    return loss, metrics


def unlabeled_known_acceptance_quarantine_loss(
    anchor_z: torch.Tensor,
    anchor_y: torch.Tensor,
    query_z: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    *,
    anchor_d: Optional[torch.Tensor] = None,
    query_y: Optional[torch.Tensor] = None,
    query_d: Optional[torch.Tensor] = None,
    paired_view_count: int = 0,
    return_state_masks: bool = False,
    core_quantile: float = 0.70,
    accept_quantile: float = 0.80,
    accept_target: float = 0.20,
    core_accept_target: float = 0.80,
    cvar_alpha: float = 0.25,
    accept_temperature: float = 0.04,
    component_temperature_rad: float = math.radians(3.0),
    density_temperature_rad: float = math.radians(3.0),
    component_margin_rad: float = math.radians(4.0),
    min_classes: int = 2,
    min_samples_per_class: int = 2,
    require_domain_local_components: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Route source-unlabeled known samples without treating them as unknown negatives.

    ``outside_reject`` rejects a pseudo-label route, not known-class membership.
    Only geometry-trusted core queries receive a known-core pull. Ambiguous and
    outside queries are quarantined from class geometry and remain available to
    domain and paired clean/satellite invariance losses.
    """
    default_metrics = {
        "active": 0.0,
        "anchor_count": 0.0,
        "query_count": 0.0,
        "active_classes": 0.0,
        "local_component_count": 0.0,
        "accept_rate": float("nan"),
        "accept_loss": 0.0,
        "core_keep_loss": 0.0,
        "tail_quarantine_loss": 0.0,
        "outside_reject_loss": 0.0,
        "low_density_accept_rate": float("nan"),
        "nearest_angle_p50_deg": float("nan"),
        "nearest_angle_p95_deg": float("nan"),
        "nearest_angle_p99_deg": float("nan"),
        "radius_to_inter_ratio": float("nan"),
        "tri_trusted_core_count": 0.0,
        "tri_ambiguous_tail_count": 0.0,
        "tri_outside_reject_count": 0.0,
        "tri_trusted_core_rate": float("nan"),
        "tri_ambiguous_tail_rate": float("nan"),
        "tri_outside_reject_rate": float("nan"),
        "tri_class_coverage": 0.0,
        "tri_domain_coverage": 0.0,
        "tri_pair_disagreement_rate": float("nan"),
        "tri_pseudo_component_agreement_rate": float("nan"),
        "outside_known_negative_disabled": 1.0,
        "global_component_fallback": 0.0,
        "domain_local_components_required": 1.0 if bool(require_domain_local_components) else 0.0,
    }
    if (
        anchor_z is None
        or query_z is None
        or (not torch.is_tensor(anchor_z))
        or (not torch.is_tensor(query_z))
        or anchor_z.numel() == 0
        or query_z.numel() == 0
    ):
        ref = anchor_z if torch.is_tensor(anchor_z) else query_z if torch.is_tensor(query_z) else torch.tensor(0.0)
        return zero_like_with_grad(ref), default_metrics
    if anchor_z.dim() != 2 or query_z.dim() != 2:
        raise ValueError(
            "unlabeled_known_acceptance_quarantine_loss expects 2D anchor/query features, "
            f"got anchor={tuple(anchor_z.shape)} query={tuple(query_z.shape)}"
        )
    labels = anchor_y.view(-1).long()
    if labels.numel() != anchor_z.size(0):
        raise ValueError(f"anchor label count {labels.numel()} does not match feature batch {anchor_z.size(0)}")
    if query_mask is not None:
        if not torch.is_tensor(query_mask) or query_mask.numel() != query_z.size(0):
            raise ValueError("query_mask must be a tensor with one value per query feature")
        query_z = query_z[query_mask.view(-1).bool()]
        if query_y is not None:
            query_y = query_y.view(-1)[query_mask.view(-1).bool()]
        if query_d is not None:
            query_d = query_d.view(-1)[query_mask.view(-1).bool()]
    if query_z.numel() == 0:
        return zero_like_with_grad(anchor_z), default_metrics
    if query_y is None:
        raise ValueError("query_y pseudo labels are required for fail-closed U_s geometry routing")
    if int(query_y.numel()) != int(query_z.size(0)):
        raise ValueError("query_y must have one pseudo label per query feature")
    if query_d is not None and int(query_d.numel()) != int(query_z.size(0)):
        raise ValueError("query_d must have one source-domain label per query feature")

    anchor_norm = safe_l2_normalize(torch.nan_to_num(anchor_z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    query_norm = safe_l2_normalize(torch.nan_to_num(query_z.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=1)
    valid = labels >= 0
    domains = None
    if anchor_d is not None and torch.is_tensor(anchor_d):
        domains = anchor_d.view(-1).long()
        if domains.numel() != labels.numel():
            raise ValueError("anchor_d must have one source-domain value per anchor")
    centers = []
    class_ids = []
    sample_mask = torch.zeros_like(valid, dtype=torch.bool)
    min_count = max(1, int(min_samples_per_class))
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls)
        class_component_count = 0
        if domains is not None:
            for dom in torch.unique(domains[cls_mask & (domains >= 0)]):
                cell = cls_mask & domains.eq(dom)
                if int(cell.sum().item()) < min_count:
                    continue
                centers.append(safe_l2_normalize(anchor_norm[cell].mean(dim=0, keepdim=True), dim=1).squeeze(0))
                class_ids.append(cls)
                sample_mask |= cell
                class_component_count += 1
        if (
            class_component_count == 0
            and not bool(require_domain_local_components)
            and int(cls_mask.sum().item()) >= min_count
        ):
            centers.append(safe_l2_normalize(anchor_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0))
            class_ids.append(cls)
            sample_mask |= cls_mask

    active_classes = len({int(cls.item()) for cls in class_ids})
    default_metrics["anchor_count"] = float(int(sample_mask.sum().detach().item()))
    default_metrics["query_count"] = float(int(query_norm.size(0)))
    default_metrics["active_classes"] = float(active_classes)
    default_metrics["local_component_count"] = float(len(centers))
    if active_classes < max(1, int(min_classes)):
        return zero_like_with_grad(anchor_z), default_metrics

    proto = torch.stack(centers, dim=0)
    center_labels = torch.stack(class_ids, dim=0).to(device=labels.device)
    sample_z = anchor_norm[sample_mask]
    sample_labels = labels[sample_mask]
    sample_cos = (sample_z @ proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    sample_angles = _safe_angle_from_cos(sample_cos)
    own_center = sample_labels.view(-1, 1).eq(center_labels.view(1, -1))
    own_angles = sample_angles.masked_fill(~own_center, float("inf"))
    pos_angles, own_component_idx = own_angles.min(dim=1)
    if pos_angles.numel() == 0:
        return zero_like_with_grad(anchor_z), default_metrics

    core_q = max(0.0, min(1.0, float(core_quantile)))
    accept_q = max(core_q, min(1.0, float(accept_quantile)))
    accept_radius_vec = []
    core_radius_vec = []
    core_mask = torch.zeros((sample_z.size(0),), device=sample_z.device, dtype=torch.bool)
    for component_idx in range(proto.size(0)):
        cls_mask = own_component_idx.eq(component_idx)
        cls_angles = pos_angles[cls_mask]
        if cls_angles.numel() == 0:
            fallback_radius = sample_z.new_tensor(math.radians(40.0))
            core_radius_vec.append(fallback_radius)
            accept_radius_vec.append(fallback_radius)
            continue
        det = cls_angles.detach()
        core_radius = torch.quantile(det, core_q) if det.numel() > 1 else det.max()
        accept_radius = torch.quantile(det, accept_q) if det.numel() > 1 else det.max()
        core_radius_vec.append(core_radius.clamp_min(1e-4))
        accept_radius_vec.append(accept_radius.clamp_min(1e-4))
        core_mask |= cls_mask & (pos_angles <= core_radius)
    accept_radius_vec_t = torch.stack(accept_radius_vec, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
    core_radius_vec_t = torch.stack(core_radius_vec, dim=0).to(device=sample_z.device, dtype=sample_z.dtype)
    if not bool(core_mask.any()):
        core_mask = torch.ones_like(core_mask)
    z_core_density = sample_z[core_mask]
    density_radius = (
        torch.quantile(pos_angles[core_mask].detach(), min(0.95, max(0.50, core_q)))
        if bool(core_mask.any())
        else accept_radius_vec_t.detach().median()
    )

    query_angles = _safe_angle_from_cos((query_norm @ proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    radius_gate = torch.sigmoid(
        (accept_radius_vec_t.detach().view(1, -1) - query_angles) / max(1e-4, float(component_temperature_rad))
    )
    unique_classes = torch.unique(center_labels)
    class_angle_columns = torch.stack(
        [query_angles[:, center_labels.eq(cls)].min(dim=1).values for cls in unique_classes],
        dim=1,
    )
    if class_angle_columns.size(1) > 1:
        sorted_angles = torch.sort(class_angle_columns.detach(), dim=1).values
        class_gap = sorted_angles[:, 1] - sorted_angles[:, 0]
        margin_gate = torch.sigmoid(
            (class_gap - float(component_margin_rad)) / max(1e-4, float(component_temperature_rad))
        ).view(-1, 1)
    else:
        margin_gate = torch.ones((query_angles.size(0), 1), device=query_angles.device, dtype=query_angles.dtype)
    core_cos = (query_norm @ z_core_density.detach().t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    nearest_core_angle = _safe_angle_from_cos(core_cos.max(dim=1).values)
    density_gate = torch.sigmoid((density_radius - nearest_core_angle) / max(1e-4, float(density_temperature_rad))).view(-1, 1)
    low_density_prob = torch.sigmoid((nearest_core_angle - density_radius) / max(1e-4, float(density_temperature_rad)))
    accept_prob = (radius_gate * margin_gate * density_gate).max(dim=1).values
    with torch.no_grad():
        nearest = query_angles.min(dim=1)
        nearest_angle = nearest.values.detach()
        nearest_idx = nearest.indices.detach()
        nearest_class = center_labels.detach()[nearest_idx]
        tri_label_match = nearest_class.eq(query_y.view(-1).long().to(device=nearest_class.device))
        nearest_core_radius = core_radius_vec_t.detach()[nearest_idx]
        nearest_accept_radius = accept_radius_vec_t.detach()[nearest_idx]
        tri_class_gap = class_gap.detach().view(-1)
        tri_margin_safe = tri_class_gap >= float(component_margin_rad)
        tri_margin_reject = tri_class_gap < 0.5 * float(component_margin_rad)
        tri_density_safe = nearest_core_angle.detach() <= density_radius.detach()
        tri_density_reject = nearest_core_angle.detach() > (
            density_radius.detach() + max(1e-4, float(density_temperature_rad))
        )
        tri_trusted_core = (
            (nearest_angle <= nearest_core_radius)
            & tri_margin_safe
            & tri_density_safe
            & tri_label_match
        )
        tri_outside_reject = (
            (nearest_angle > nearest_accept_radius)
            | tri_margin_reject
            | tri_density_reject
            | (~tri_label_match)
        )
        tri_ambiguous_tail = (~tri_trusted_core) & (~tri_outside_reject)
        q50 = torch.quantile(nearest_angle, 0.50) if nearest_angle.numel() > 1 else nearest_angle.mean()
        q95 = torch.quantile(nearest_angle, 0.95) if nearest_angle.numel() > 1 else nearest_angle.mean()
        q99 = torch.quantile(nearest_angle, 0.99) if nearest_angle.numel() > 1 else nearest_angle.mean()
        radius_inter_ratio_metric = sample_z.new_tensor(float("nan"))
        if proto.size(0) > 1:
            inter_angles = _safe_angle_from_cos((proto @ proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            different_class = center_labels.view(-1, 1).ne(center_labels.view(1, -1))
            nearest_inter = inter_angles.masked_fill(~different_class, float("inf")).min(dim=1).values.detach().clamp_min(1e-4)
            radius_inter_ratio_metric = (accept_radius_vec_t.detach() / nearest_inter).max()
        tri_query_count = max(1, int(query_norm.size(0)))
        tri_trusted_core_count = float(int(tri_trusted_core.sum().item()))
        tri_ambiguous_tail_count = float(int(tri_ambiguous_tail.sum().item()))
        tri_outside_reject_count = float(int(tri_outside_reject.sum().item()))
        tri_class_coverage = float(int(torch.unique(query_y.view(-1).long()).numel())) if query_y is not None else 0.0
        tri_domain_coverage = float(int(torch.unique(query_d.view(-1).long()).numel())) if query_d is not None else 0.0
        pair_count = max(0, int(paired_view_count))
        tri_pair_disagreement_rate = float("nan")
        if pair_count > 0:
            if int(query_norm.size(0)) != 2 * pair_count:
                raise ValueError("paired_view_count requires clean and satellite query blocks of equal size")
            tri_state = torch.full_like(nearest_idx, 1)
            tri_state[tri_trusted_core] = 0
            tri_state[tri_outside_reject] = 2
            tri_pair_disagreement_rate = float(
                (tri_state[:pair_count] != tri_state[pair_count:]).float().mean().item()
            )
        tri_pseudo_component_agreement_rate = float(tri_label_match.float().mean().item())

    cvar_frac = max(1e-6, min(1.0, float(cvar_alpha)))
    tau = max(1e-4, float(accept_temperature))

    def _masked_cvar(values: torch.Tensor) -> torch.Tensor:
        return _top_cvar_mean(values, cvar_frac) if values.numel() else accept_prob.new_tensor(0.0)

    core_keep_loss = _masked_cvar(
        _bounded_softplus((float(core_accept_target) - accept_prob[tri_trusted_core]) / tau)
    )
    # U_s is sampled from known source TXs. Ambiguous/outside states block
    # pseudo-label geometry; they must never be optimized as unknown negatives.
    tail_quarantine_loss = accept_prob[tri_ambiguous_tail].sum() * 0.0
    outside_reject_loss = accept_prob[tri_outside_reject].sum() * 0.0
    accept_loss = core_keep_loss + tail_quarantine_loss + outside_reject_loss

    metrics = {
        "active": 1.0,
        "anchor_count": float(int(sample_mask.sum().detach().item())),
        "query_count": float(int(query_norm.size(0))),
        "active_classes": float(active_classes),
        "local_component_count": float(proto.size(0)),
        "accept_rate": _scalar_metric(accept_prob.detach().mean()),
        "accept_loss": _scalar_metric(accept_loss),
        "core_keep_loss": _scalar_metric(core_keep_loss),
        "tail_quarantine_loss": _scalar_metric(tail_quarantine_loss),
        "outside_reject_loss": _scalar_metric(outside_reject_loss),
        "low_density_accept_rate": _scalar_metric((accept_prob.detach() * low_density_prob.detach()).mean()),
        "nearest_angle_p50_deg": math.degrees(_scalar_metric(q50)),
        "nearest_angle_p95_deg": math.degrees(_scalar_metric(q95)),
        "nearest_angle_p99_deg": math.degrees(_scalar_metric(q99)),
        "radius_to_inter_ratio": _scalar_metric(radius_inter_ratio_metric),
        "tri_trusted_core_count": tri_trusted_core_count,
        "tri_ambiguous_tail_count": tri_ambiguous_tail_count,
        "tri_outside_reject_count": tri_outside_reject_count,
        "tri_trusted_core_rate": tri_trusted_core_count / float(tri_query_count),
        "tri_ambiguous_tail_rate": tri_ambiguous_tail_count / float(tri_query_count),
        "tri_outside_reject_rate": tri_outside_reject_count / float(tri_query_count),
        "tri_class_coverage": tri_class_coverage,
        "tri_domain_coverage": tri_domain_coverage,
        "tri_pair_disagreement_rate": tri_pair_disagreement_rate,
        "tri_pseudo_component_agreement_rate": tri_pseudo_component_agreement_rate,
        "outside_known_negative_disabled": 1.0,
        "global_component_fallback": 0.0 if domains is not None else 1.0,
        "domain_local_components_required": 1.0 if bool(require_domain_local_components) else 0.0,
    }
    if return_state_masks:
        metrics["_tri_trusted_core_mask"] = tri_trusted_core.detach()
        metrics["_tri_ambiguous_tail_mask"] = tri_ambiguous_tail.detach()
        metrics["_tri_outside_reject_mask"] = tri_outside_reject.detach()
        metrics["_tri_accept_prob"] = accept_prob.detach()
        metrics["_tri_label_match_mask"] = tri_label_match.detach()
    return accept_loss, metrics


def _binary_auc(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    if known_scores.numel() == 0 or unknown_scores.numel() == 0:
        return float("nan")
    return float((unknown_scores.view(-1, 1) > known_scores.view(1, -1)).float().mean().item())


def multiview_source_episode_three_sigma_loss(
    clean_z: torch.Tensor,
    sat_z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    *,
    clean_weight: float = 1.0,
    sat_weight: float = 1.0,
    normalize_active_weights: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Keep clean/satellite receiver-local components separate and expose worst-view risk."""

    if clean_z.shape != sat_z.shape or clean_z.dim() != 2:
        raise ValueError("multiview source episode expects aligned clean/satellite [B,D] features")
    clean_loss, clean_info = source_episode_three_sigma_loss(clean_z, y, d, **kwargs)
    sat_loss, sat_info = source_episode_three_sigma_loss(sat_z, y, d, **kwargs)
    clean_weight_f = max(0.0, float(clean_weight))
    sat_weight_f = max(0.0, float(sat_weight))
    active_weight = clean_weight_f + sat_weight_f
    weighted_sum = clean_weight_f * clean_loss + sat_weight_f * sat_loss
    loss = weighted_sum / max(active_weight, 1e-8) if bool(normalize_active_weights) else weighted_sum
    metrics: Dict[str, float] = {
        "source_episode_multiview_separate_geometry": 1.0,
        "source_episode_multiview_active_weight": float(active_weight),
        "source_episode_multiview_normalized": 1.0 if bool(normalize_active_weights) else 0.0,
        "source_episode_multiview_weighted_sum_loss": _scalar_metric(weighted_sum),
    }
    for prefix, info in (("clean", clean_info), ("sat", sat_info)):
        for key, value in info.items():
            metrics[f"{prefix}_{key}"] = value
    risk_keys = (
        "source_episode_overflow_rate",
        "source_overflow",
        "source_episode_zid_p50_deg",
        "source_episode_zid_p95_deg",
        "source_episode_zid_p99_deg",
        "source_episode_zid_tail_cvar_deg",
        "source_episode_tail_query_rate",
        "source_episode_local_component_radius_p95_deg",
        "source_episode_local_component_center_spread_deg",
        "source_episode_local_component_min_inter_deg",
        "source_episode_local_component_max_radius_inter_ratio",
    )
    for key in risk_keys:
        values = []
        for info in (clean_info, sat_info):
            try:
                value = float(info.get(key, float("nan")))
            except Exception:
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
        metrics[key] = max(values) if values else float("nan")
    metrics["source_episode_loss"] = _scalar_metric(loss)
    loss_keys = (
        "source_episode_leave_domain_loss",
        "source_episode_mixup_loss",
        "source_episode_local_component_compact_loss",
        "source_episode_local_component_invariant_loss",
        "source_episode_local_component_inter_loss",
        "source_episode_local_component_overlap_loss",
        "source_episode_leave_domain_target_loss",
        "source_episode_local_component_accept_loss",
        "source_episode_local_component_density_loss",
        "source_episode_local_component_accept_raw_loss",
        "source_episode_local_component_density_raw_loss",
    )
    for key in loss_keys:
        clean_value = float(clean_info.get(key, 0.0))
        sat_value = float(sat_info.get(key, 0.0))
        weighted_value = clean_weight_f * clean_value + sat_weight_f * sat_value
        metrics[key] = weighted_value / max(active_weight, 1e-8) if bool(normalize_active_weights) else weighted_value
    upper_bounds = [
        float(info.get("source_episode_loss_upper_bound", float("nan")))
        for info in (clean_info, sat_info)
    ]
    finite_upper_bounds = [value for value in upper_bounds if math.isfinite(value)]
    metrics["source_episode_loss_upper_bound"] = max(finite_upper_bounds) if finite_upper_bounds else float("nan")
    for key in (
        "source_episode_receiver_local_component_count",
        "source_episode_local_component_coverage",
        "source_episode_local_component_structural_active",
        "source_episode_core_tail_outside_ready",
        "source_episode_density_gate_active",
    ):
        values = [float(info.get(key, 0.0)) for info in (clean_info, sat_info)]
        metrics[key] = min(values)
    return loss, metrics


def source_episode_three_sigma_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    d: Optional[torch.Tensor],
    *,
    min_domains: int = 2,
    min_samples_per_class_domain: int = 1,
    radius_cap_rad: float = math.radians(30.0),
    min_sigma_rad: float = math.radians(3.0),
    radius_mode: str = "min_three_sigma_core",
    core_quantile: float = 0.80,
    mixup_features: Optional[torch.Tensor] = None,
    mixup_weight: float = 0.0,
    mixup_order: int = 3,
    mixup_hard_k: int = 2,
    local_component_compact_weight: float = 0.0,
    local_component_invariant_weight: float = 0.0,
    local_component_inter_weight: float = 0.0,
    local_component_inter_margin_rad: float = math.radians(20.0),
    local_component_center_target_rad: float = 0.0,
    local_component_overlap_weight: float = 0.0,
    local_component_overlap_margin_rad: float = math.radians(4.0),
    local_component_accept_weight: float = 0.0,
    local_component_density_weight: float = 0.0,
    local_component_min_samples: int = 2,
    local_component_radius_floor_rad: float = math.radians(3.0),
    local_component_density_beta: float = 0.20,
    local_component_density_cap: float = 2.0,
    local_component_term_cap: float = 4.0,
    leave_domain_target_rad: float = math.radians(40.0),
    leave_domain_target_weight: float = 0.0,
    structural_cvar_alpha: float = 0.20,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Source-only leave-domain angular shell objective.

    This is a Phase1 proxy objective: all support/query roles are drawn from
    source domains in the current batch. It must not be interpreted as Stage2
    target-new or unknown rejection evidence.
    """
    default_metrics = {
        "source_episode_loss": 0.0,
        "source_episode_leave_domain_loss": 0.0,
        "source_episode_overflow_rate": 0.0,
        "source_overflow": 0.0,
        "source_episode_radius_3sigma_deg": float("nan"),
        "source_episode_radius_core_deg": float("nan"),
        "source_episode_radius_safe_deg": float("nan"),
        "source_episode_radius_mode_code": 2.0,
        "source_episode_val_angle_deg": float("nan"),
        "source_episode_tail_query_rate": 0.0,
        "source_episode_classes": 0.0,
        "source_episode_domains": 0.0,
        "source_episode_mixup_count": 0.0,
        "source_episode_mixup_order": float(max(2, int(mixup_order))),
        "source_episode_mixup_loss": 0.0,
        "source_episode_mixup_overflow_rate": 0.0,
        "source_episode_mixup_margin_deg": float("nan"),
        "source_episode_receiver_local_component_count": 0.0,
        "source_episode_local_component_coverage": 0.0,
        "source_episode_local_component_compact_loss": 0.0,
        "source_episode_local_component_invariant_loss": 0.0,
        "source_episode_local_component_inter_loss": 0.0,
        "source_episode_local_component_overlap_loss": 0.0,
        "source_episode_leave_domain_target_loss": 0.0,
        "source_episode_local_component_min_inter_deg": float("nan"),
        "source_episode_local_component_max_radius_inter_ratio": float("nan"),
        "source_episode_local_component_accept_loss": 0.0,
        "source_episode_local_component_density_loss": 0.0,
        "source_episode_local_component_accept_raw_loss": 0.0,
        "source_episode_local_component_density_raw_loss": 0.0,
        "source_episode_local_component_radius_p95_deg": float("nan"),
        "source_episode_local_component_radius_min_deg": float("nan"),
        "source_episode_local_component_loss_radius_floor_deg": math.degrees(
            max(float(min_sigma_rad), float(local_component_radius_floor_rad))
        ),
        "source_episode_local_component_radius_floor_rate": 0.0,
        "source_episode_local_component_center_spread_deg": float("nan"),
        "source_episode_local_component_structural_active": 0.0,
        "source_episode_loss_upper_bound": float("nan"),
        "source_episode_core_count": 0.0,
        "source_episode_tail_count": 0.0,
        "source_episode_outside_count": 0.0,
        "source_episode_leave_domain_core_count": 0.0,
        "source_episode_leave_domain_tail_count": 0.0,
        "source_episode_leave_domain_outside_count": 0.0,
        "source_episode_core_rate": 0.0,
        "source_episode_tail_rate": 0.0,
        "source_episode_outside_rate": 0.0,
        "source_episode_zid_p50_deg": float("nan"),
        "source_episode_zid_p95_deg": float("nan"),
        "source_episode_zid_p99_deg": float("nan"),
        "source_episode_zid_tail_cvar_deg": float("nan"),
        "source_episode_core_tail_outside_ready": 0.0,
        "source_episode_density_gate_active": 0.0,
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
    core_radii = []
    safe_radii = []
    val_angles_all = []
    val_angles_flat = []
    local_component_keys = set()
    local_component_centers = []
    local_component_labels = []
    local_component_radii = []
    local_component_spreads = []
    local_compact_terms = []
    local_invariant_terms = []
    leave_domain_target_terms = []
    local_possible_components = 0.0
    min_cell = max(1, int(min_samples_per_class_domain))
    local_min_cell = max(min_cell, int(local_component_min_samples), 2)
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls)
        cls_centers = []
        for dom in torch.unique(domains[cls_mask]):
            cell = cls_mask & domains.eq(dom)
            if int(cell.sum().item()) < local_min_cell:
                continue
            local_possible_components += 1.0
            center = safe_l2_normalize(z_norm[cell].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            angles = _safe_angle_from_cos(
                (z_norm[cell] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            )
            local_compact_terms.append(angles.pow(2).mean())
            radius = torch.quantile(angles.detach(), 0.95) if angles.numel() > 1 else angles.detach().max()
            local_component_radii.append(radius)
            local_component_centers.append(center)
            local_component_labels.append(cls)
            cls_centers.append(center)
            local_component_keys.add((int(cls.item()), int(dom.item())))
        if len(cls_centers) >= max(2, int(min_domains)):
            stacked = torch.stack(cls_centers, dim=0)
            invariant_core = safe_l2_normalize(stacked.mean(dim=0, keepdim=True), dim=1).squeeze(0).detach()
            spread = _safe_angle_from_cos(
                (stacked * invariant_core.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            )
            local_component_spreads.append(spread.detach().mean())
            local_invariant_terms.append(
                F.relu(spread - float(local_component_center_target_rad)).pow(2)
            )

    local_compact_raw_loss = (
        torch.stack(local_compact_terms).mean() if local_compact_terms else z_norm.sum() * 0.0
    )
    local_invariant_raw_loss = (
        _top_cvar_mean(
            torch.cat([term.reshape(-1) for term in local_invariant_terms], dim=0),
            max(1e-6, min(1.0, float(structural_cvar_alpha))),
        )
        if local_invariant_terms
        else z_norm.sum() * 0.0
    )
    local_inter_raw_loss = z_norm.sum() * 0.0
    local_accept_raw_loss = z_norm.sum() * 0.0
    local_density_raw_loss = z_norm.sum() * 0.0
    local_overlap_raw_loss = z_norm.sum() * 0.0
    local_compact_loss = _bounded_positive_scalar_loss(local_compact_raw_loss, cap=local_component_term_cap)
    local_invariant_loss = _bounded_positive_scalar_loss(local_invariant_raw_loss, cap=local_component_term_cap)
    local_inter_loss = z_norm.sum() * 0.0
    local_accept_loss = z_norm.sum() * 0.0
    local_density_loss = z_norm.sum() * 0.0
    local_overlap_loss = z_norm.sum() * 0.0
    local_min_inter = z_norm.new_tensor(float("nan"))
    local_max_radius_inter_ratio = z_norm.new_tensor(float("nan"))
    local_core_count = 0.0
    local_tail_count = 0.0
    local_outside_count = 0.0
    local_radius_floor_count = 0.0
    local_radius_count = 0.0
    if len(local_component_centers) > 1:
        component_tensor = torch.stack(local_component_centers, dim=0)
        component_labels = torch.stack(local_component_labels, dim=0).to(device=labels.device)
        component_angles = _safe_angle_from_cos(
            (component_tensor @ component_tensor.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        )
        different_class = component_labels.view(-1, 1).ne(component_labels.view(1, -1))
        if bool(different_class.any()):
            nearest_other_component = component_angles.masked_fill(
                ~different_class, float("inf")
            ).min(dim=1).values
            finite_nearest = nearest_other_component[torch.isfinite(nearest_other_component)]
            local_inter_raw_loss = _top_cvar_mean(
                F.relu(
                    float(local_component_inter_margin_rad) - finite_nearest
                ).pow(2),
                max(1e-6, min(1.0, float(structural_cvar_alpha))),
            )
            local_inter_loss = _bounded_positive_scalar_loss(
                local_inter_raw_loss,
                cap=local_component_term_cap,
            )
            local_min_inter = finite_nearest.detach().min()
            component_radius_tensor_for_overlap = torch.stack(local_component_radii).to(
                device=z_norm.device, dtype=z_norm.dtype
            ).detach()
            pair_radius = (
                component_radius_tensor_for_overlap.view(-1, 1)
                + component_radius_tensor_for_overlap.view(1, -1)
            )
            overlap = F.relu(
                pair_radius
                + float(local_component_overlap_margin_rad)
                - component_angles
            ).masked_select(different_class)
            local_overlap_raw_loss = _top_cvar_mean(
                overlap.pow(2),
                max(1e-6, min(1.0, float(structural_cvar_alpha))),
            )
            local_overlap_loss = _bounded_positive_scalar_loss(
                local_overlap_raw_loss,
                cap=local_component_term_cap,
            )
            local_max_radius_inter_ratio = (
                component_radius_tensor_for_overlap
                / finite_nearest.detach().clamp_min(1e-4)
            ).max()
        sample_local_angles = _safe_angle_from_cos(
            (z_norm[valid] @ component_tensor.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        )
        sample_labels_valid = labels[valid]
        own_component = sample_labels_valid.view(-1, 1).eq(component_labels.view(1, -1))
        own_angles = sample_local_angles.masked_fill(~own_component, float("inf"))
        nearest_own, nearest_own_idx = own_angles.min(dim=1)
        nearest_other = sample_local_angles.masked_fill(own_component, float("inf")).min(dim=1).values
        component_radius_tensor = torch.stack(local_component_radii).to(
            device=z_norm.device, dtype=z_norm.dtype
        ).detach().clamp_min(1e-4)
        loss_radius_floor = max(
            1e-4,
            float(min_sigma_rad),
            float(local_component_radius_floor_rad),
        )
        component_loss_radius_tensor = component_radius_tensor.clamp_min(loss_radius_floor)
        own_radius = component_radius_tensor[nearest_own_idx]
        own_loss_radius = component_loss_radius_tensor[nearest_own_idx]
        finite_local = torch.isfinite(nearest_own) & torch.isfinite(nearest_other)
        if bool(finite_local.any()):
            own_valid = nearest_own[finite_local]
            other_valid = nearest_other[finite_local]
            radius_valid = own_radius[finite_local]
            loss_radius_valid = own_loss_radius[finite_local]
            local_overflow = F.relu(own_valid - radius_valid).pow(2).mean()
            local_bridge = F.relu(
                float(local_component_inter_margin_rad) - (other_valid - own_valid)
            ).pow(2).mean()
            local_accept_raw_loss = local_overflow + local_bridge
            local_accept_loss = _bounded_positive_scalar_loss(
                local_accept_raw_loss,
                cap=local_component_term_cap,
            )
            normalized_radius = own_valid / loss_radius_valid
            density_excess = F.relu(normalized_radius - 0.80)
            density_terms = F.smooth_l1_loss(
                density_excess,
                torch.zeros_like(density_excess),
                reduction="none",
                beta=max(1e-4, float(local_component_density_beta)),
            )
            local_density_raw_loss = _top_cvar_mean(density_terms, 0.20)
            local_density_loss = _bounded_positive_scalar_loss(
                local_density_raw_loss,
                cap=local_component_density_cap,
            )
            with torch.no_grad():
                local_radius_floor_count = float((radius_valid < loss_radius_floor).float().sum().item())
                local_radius_count = float(radius_valid.numel())
                local_core_count = float((normalized_radius <= 0.80).float().sum().item())
                local_outside_count = float((normalized_radius > 1.0).float().sum().item())
                local_tail_count = float(normalized_radius.numel()) - local_core_count - local_outside_count
    tail_query_rates = []
    episode_centers = []
    episode_radii = []
    component_keys = set()
    used_classes = set()
    used_domains = set()
    core_count = 0.0
    tail_count = 0.0
    outside_count = 0.0
    possible_components = 0.0
    mode_code = 2.0
    for cls in torch.unique(labels[valid]):
        cls_mask = valid & labels.eq(cls)
        doms = [dom for dom in torch.unique(domains[cls_mask]) if int((cls_mask & domains.eq(dom)).sum().item()) >= min_cell]
        if len(doms) < max(2, int(min_domains)):
            continue
        possible_components += float(len(doms))
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
            radius_3sigma = torch.minimum(
                train_angles.new_tensor(float(radius_cap_rad)),
                torch.minimum(
                    train_angles.max() + 3.0 * robust_sigma,
                    median + 3.0 * robust_sigma,
                ),
            )
            core_q = max(0.0, min(1.0, float(core_quantile)))
            radius_core = torch.quantile(train_angles, core_q) if train_angles.numel() > 1 else train_angles.max()
            radius_core = torch.minimum(train_angles.new_tensor(float(radius_cap_rad)), radius_core)
            mode_l = str(radius_mode or "min_three_sigma_core").lower().strip()
            mode_code = {
                "three_sigma": 0.0,
                "legacy_three_sigma": 0.0,
                "core": 1.0,
                "core_quantile": 1.0,
                "strict_core": 1.0,
                "min_three_sigma_core": 2.0,
                "min_core_three_sigma": 2.0,
                "core_safe": 2.0,
            }.get(mode_l, 2.0)
            if mode_l in {"core", "core_quantile", "strict_core"}:
                radius = radius_core
            elif mode_l in {"min_three_sigma_core", "min_core_three_sigma", "core_safe"}:
                radius = torch.minimum(radius_3sigma, radius_core)
            elif mode_l in {"three_sigma", "legacy_three_sigma"}:
                radius = radius_3sigma
            else:
                radius = torch.minimum(radius_3sigma, radius_core)
            val_cos = (z_norm[val_mask] * center.view(1, -1)).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            val_angles = _safe_angle_from_cos(val_cos)
            overflow = torch.relu(val_angles - radius)
            losses.append(overflow.pow(2).mean())
            leave_domain_target_terms.append(
                F.relu(val_angles - float(leave_domain_target_rad)).pow(2)
            )
            overflow_rates.append(float((val_angles.detach() > radius).float().mean().item()))
            radii.append(radius_3sigma.detach())
            core_radii.append(radius_core.detach())
            safe_radii.append(radius.detach())
            val_det = val_angles.detach()
            radius_core_det = radius_core.detach()
            radius_det = radius.detach()
            val_angles_all.append(val_det.mean())
            val_angles_flat.append(val_det)
            is_core = val_det <= radius_core_det
            is_outside = val_det > radius_det
            is_tail = (~is_core) & (~is_outside)
            core_count += float(is_core.float().sum().item())
            tail_count += float(is_tail.float().sum().item())
            outside_count += float(is_outside.float().sum().item())
            tail_query_rates.append(float((val_det > radius_core_det).float().mean().item()))
            episode_centers.append(center)
            episode_radii.append(radius.detach())
            component_keys.add((int(cls.item()), int(val_dom.item())))
            used_classes.add(int(cls.item()))
            used_domains.add(int(val_dom.item()))
    source_loss = torch.stack(losses).mean() if losses else z_norm.sum() * 0.0
    leave_domain_target_loss = (
        _top_cvar_mean(
            torch.cat([term.reshape(-1) for term in leave_domain_target_terms], dim=0),
            max(1e-6, min(1.0, float(structural_cvar_alpha))),
        )
        if leave_domain_target_terms
        else z_norm.sum() * 0.0
    )
    mixup_loss = z.new_tensor(0.0)
    mixup_overflow_rate = 0.0
    mixup_margin = z.new_tensor(float("nan"))
    mixup_count = 0.0
    if (
        mixup_features is not None
        and torch.is_tensor(mixup_features)
        and mixup_features.numel() > 0
        and float(mixup_weight) > 0.0
        and episode_centers
    ):
        mix_norm = safe_l2_normalize(
            torch.nan_to_num(mixup_features.float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=1,
        )
        ep_proto = torch.stack(episode_centers, dim=0)
        ep_radii = torch.stack(episode_radii, dim=0).to(device=ep_proto.device, dtype=ep_proto.dtype)
        mix_cos = (mix_norm @ ep_proto.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        mix_angles = _safe_angle_from_cos(mix_cos)
        inside = F.relu(ep_radii.view(1, -1) - mix_angles)
        hard_k = max(1, min(int(mixup_hard_k), inside.size(1)))
        mixup_loss = inside.pow(2).topk(k=hard_k, dim=1, largest=True).values.mean()
        with torch.no_grad():
            margins = mix_angles.detach() - ep_radii.detach().view(1, -1)
            nearest_margin = margins.min(dim=1).values
            mixup_overflow_rate = float((nearest_margin < 0.0).float().mean().item())
            mixup_margin = nearest_margin.mean()
            mixup_count = float(mix_norm.size(0))
    loss = (
        source_loss
        + max(0.0, float(mixup_weight)) * mixup_loss
        + max(0.0, float(local_component_compact_weight)) * local_compact_loss
        + max(0.0, float(local_component_invariant_weight)) * local_invariant_loss
        + max(0.0, float(local_component_inter_weight)) * local_inter_loss
        + max(0.0, float(local_component_overlap_weight)) * local_overlap_loss
        + max(0.0, float(local_component_accept_weight)) * local_accept_loss
        + max(0.0, float(local_component_density_weight)) * local_density_loss
        + max(0.0, float(leave_domain_target_weight)) * leave_domain_target_loss
    )
    loss_upper_bound = (
        math.pi * math.pi
        + max(0.0, float(mixup_weight)) * math.pi * math.pi
        + max(0.0, float(local_component_compact_weight)) * max(1e-6, float(local_component_term_cap))
        + max(0.0, float(local_component_invariant_weight)) * max(1e-6, float(local_component_term_cap))
        + max(0.0, float(local_component_inter_weight)) * max(1e-6, float(local_component_term_cap))
        + max(0.0, float(local_component_overlap_weight)) * max(1e-6, float(local_component_term_cap))
        + max(0.0, float(local_component_accept_weight)) * max(1e-6, float(local_component_term_cap))
        + max(0.0, float(local_component_density_weight)) * max(1e-6, float(local_component_density_cap))
        + max(0.0, float(leave_domain_target_weight)) * math.pi * math.pi
    )
    all_val_angles = torch.cat(val_angles_flat, dim=0) if val_angles_flat else z.new_zeros(0)
    if all_val_angles.numel() > 0:
        zid_p50 = torch.quantile(all_val_angles, 0.50)
        zid_p95 = torch.quantile(all_val_angles, 0.95)
        zid_p99 = torch.quantile(all_val_angles, 0.99)
        tail_values = all_val_angles[all_val_angles >= zid_p95]
        zid_tail_cvar = tail_values.mean() if tail_values.numel() > 0 else zid_p95
    else:
        zid_p50 = zid_p95 = zid_p99 = zid_tail_cvar = z.new_tensor(float("nan"))
    component_count = float(len(local_component_keys))
    component_coverage = component_count / local_possible_components if local_possible_components > 0.0 else 0.0
    leave_domain_core_count = float(core_count)
    leave_domain_tail_count = float(tail_count)
    leave_domain_outside_count = float(outside_count)
    if component_count > 0.0 and local_radius_count > 0.0:
        core_count = float(local_core_count)
        tail_count = float(local_tail_count)
        outside_count = float(local_outside_count)
    tri_total = max(1.0, core_count + tail_count + outside_count)
    tri_ready = component_count > 0.0 and (core_count + tail_count + outside_count) > 0.0
    density_active = (
        component_count > 0.0
        and local_radius_count > 0.0
        and torch.isfinite(local_density_raw_loss.detach()).item()
    )
    metrics = {
        "source_episode_loss": _scalar_metric(loss),
        "source_episode_leave_domain_loss": _scalar_metric(source_loss),
        "source_episode_overflow_rate": float(np.mean(overflow_rates)) if overflow_rates else 0.0,
        "source_overflow": float(np.mean(overflow_rates)) if overflow_rates else 0.0,
        "source_episode_radius_3sigma_deg": math.degrees(_scalar_metric(torch.stack(radii))) if radii else float("nan"),
        "source_episode_radius_core_deg": math.degrees(_scalar_metric(torch.stack(core_radii))) if core_radii else float("nan"),
        "source_episode_radius_safe_deg": math.degrees(_scalar_metric(torch.stack(safe_radii))) if safe_radii else float("nan"),
        "source_episode_radius_mode_code": float(mode_code),
        "source_episode_val_angle_deg": math.degrees(_scalar_metric(torch.stack(val_angles_all))) if val_angles_all else float("nan"),
        "source_episode_tail_query_rate": float(np.mean(tail_query_rates)) if tail_query_rates else 0.0,
        "source_episode_classes": float(len(used_classes)),
        "source_episode_domains": float(len(used_domains)),
        "source_episode_mixup_count": mixup_count,
        "source_episode_mixup_order": float(max(2, int(mixup_order))),
        "source_episode_mixup_loss": _scalar_metric(mixup_loss),
        "source_episode_mixup_overflow_rate": float(mixup_overflow_rate),
        "source_episode_mixup_margin_deg": math.degrees(_scalar_metric(mixup_margin)),
        "source_episode_receiver_local_component_count": component_count,
        "source_episode_local_component_coverage": float(component_coverage),
        "source_episode_local_component_compact_loss": _scalar_metric(local_compact_loss),
        "source_episode_local_component_invariant_loss": _scalar_metric(local_invariant_loss),
        "source_episode_local_component_inter_loss": _scalar_metric(local_inter_loss),
        "source_episode_local_component_overlap_loss": _scalar_metric(local_overlap_loss),
        "source_episode_leave_domain_target_loss": _scalar_metric(leave_domain_target_loss),
        "source_episode_local_component_min_inter_deg": math.degrees(
            _scalar_metric(local_min_inter)
        ),
        "source_episode_local_component_max_radius_inter_ratio": _scalar_metric(
            local_max_radius_inter_ratio
        ),
        "source_episode_local_component_accept_loss": _scalar_metric(local_accept_loss),
        "source_episode_local_component_density_loss": _scalar_metric(local_density_loss),
        "source_episode_local_component_accept_raw_loss": _scalar_metric(local_accept_raw_loss),
        "source_episode_local_component_density_raw_loss": _scalar_metric(local_density_raw_loss),
        "source_episode_local_component_radius_p95_deg": (
            math.degrees(_scalar_metric(torch.quantile(torch.stack(local_component_radii), 0.95)))
            if local_component_radii
            else float("nan")
        ),
        "source_episode_local_component_radius_min_deg": (
            math.degrees(_scalar_metric(torch.stack(local_component_radii).min()))
            if local_component_radii
            else float("nan")
        ),
        "source_episode_local_component_loss_radius_floor_deg": math.degrees(
            max(float(min_sigma_rad), float(local_component_radius_floor_rad))
        ),
        "source_episode_local_component_radius_floor_rate": (
            float(local_radius_floor_count / local_radius_count) if local_radius_count > 0.0 else 0.0
        ),
        "source_episode_local_component_center_spread_deg": (
            math.degrees(_scalar_metric(torch.stack(local_component_spreads).mean()))
            if local_component_spreads
            else float("nan")
        ),
        "source_episode_local_component_structural_active": 1.0 if (
            component_count > 0.0
            and (
                float(local_component_compact_weight) > 0.0
                or float(local_component_invariant_weight) > 0.0
                or float(local_component_inter_weight) > 0.0
                or float(local_component_accept_weight) > 0.0
                or float(local_component_density_weight) > 0.0
            )
        ) else 0.0,
        "source_episode_core_count": float(core_count),
        "source_episode_tail_count": float(tail_count),
        "source_episode_outside_count": float(outside_count),
        "source_episode_leave_domain_core_count": leave_domain_core_count,
        "source_episode_leave_domain_tail_count": leave_domain_tail_count,
        "source_episode_leave_domain_outside_count": leave_domain_outside_count,
        "source_episode_core_rate": float(core_count / tri_total),
        "source_episode_tail_rate": float(tail_count / tri_total),
        "source_episode_outside_rate": float(outside_count / tri_total),
        "source_episode_zid_p50_deg": math.degrees(_scalar_metric(zid_p50)),
        "source_episode_zid_p95_deg": math.degrees(_scalar_metric(zid_p95)),
        "source_episode_zid_p99_deg": math.degrees(_scalar_metric(zid_p99)),
        "source_episode_zid_tail_cvar_deg": math.degrees(_scalar_metric(zid_tail_cvar)),
        "source_episode_core_tail_outside_ready": 1.0 if tri_ready else 0.0,
        "source_episode_density_gate_active": 1.0 if density_active else 0.0,
        "source_episode_loss_upper_bound": float(loss_upper_bound),
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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "class_proto": self.class_proto.detach().clone() if self.class_proto is not None else None,
            "domain_proto": self.domain_proto.detach().clone() if self.domain_proto is not None else None,
            "class_count": self.class_count.detach().clone() if self.class_count is not None else None,
            "domain_count": self.domain_count.detach().clone() if self.domain_count is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in ("class_proto", "domain_proto", "class_count", "domain_count"):
            value = state.get(name)
            setattr(self, name, value.detach().clone() if torch.is_tensor(value) else None)

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
                cls_mask = valid_y & y.eq(cls)
                if int(self.class_count[cls_int].item()) >= int(self.min_count):
                    class_target = safe_l2_normalize(
                        self.class_proto[cls_int].detach().view(1, -1), dim=1
                    ).squeeze(0)
                elif int(cls_mask.sum().item()) >= int(self.min_count):
                    class_target = safe_l2_normalize(
                        z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1
                    ).squeeze(0).detach()
                else:
                    continue
                for dom in torch.unique(d[(y == cls) & (d >= 0)]):
                    dom_int = int(dom.item())
                    if dom_int < 0 or dom_int >= self.num_domains:
                        continue
                    domain_mask = cls_mask & d.eq(dom)
                    if int(domain_mask.sum().item()) < int(self.min_count):
                        continue
                    current_domain = safe_l2_normalize(
                        z_norm[domain_mask].mean(dim=0, keepdim=True), dim=1
                    ).squeeze(0)
                    domain_losses.append(1.0 - (current_domain * class_target).sum())
            if domain_losses:
                loss_domain = torch.stack(domain_losses).mean()

        loss_push = z.new_tensor(0.0)
        current_class_centers = []
        for cls in torch.unique(y[valid_y]):
            cls_mask = valid_y & y.eq(cls)
            if int(cls_mask.sum().item()) >= int(self.min_count):
                current_class_centers.append(
                    safe_l2_normalize(z_norm[cls_mask].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                )
        if len(current_class_centers) > 1:
            P = torch.stack(current_class_centers, dim=0)
            sim = P @ P.t()
            eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            loss_push = F.relu(sim[~eye] - float(self.margin)).pow(2).mean()

        loss = loss_pull + float(self.domain_align_weight) * loss_domain + float(self.push_weight) * loss_push
        return loss, {
            "proto_pull_cos": pull_cos,
            "proto_domain_align": float(loss_domain.detach().item()),
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


def stratified_response_split(
    amplitude: torch.Tensor,
    *,
    bins: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic 50/50 split with alternating ranks inside amplitude strata."""
    if amplitude.ndim != 2:
        raise ValueError("amplitude must have shape [batch, samples]")
    batch, samples = amplitude.shape
    if int(samples) < 4:
        raise ValueError("split-fit requires at least four samples")
    mask_a = torch.zeros_like(amplitude, dtype=torch.bool)
    for row in range(int(batch)):
        order = torch.argsort(amplitude[row].detach(), stable=True)
        chunks = torch.tensor_split(order, max(1, int(bins)))
        for chunk in chunks:
            mask_a[row, chunk[::2]] = True
    return mask_a, ~mask_a


def _fit_response_coef(
    design: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    sqrt_weight = weight.float().clamp_min(0.05).sqrt().to(torch.complex64)
    weighted_design = design.to(torch.complex64) * sqrt_weight.unsqueeze(-1)
    weighted_target = target.to(torch.complex64) * sqrt_weight
    regularizer = torch.eye(
        int(design.size(-1)), device=design.device, dtype=torch.complex64
    ) * math.sqrt(max(0.0, float(ridge)))
    augmented_design = torch.cat([weighted_design, regularizer], dim=0)
    augmented_target = torch.cat(
        [
            weighted_target,
            torch.zeros(int(design.size(-1)), device=design.device, dtype=torch.complex64),
        ]
    )
    return torch.linalg.lstsq(augmented_design, augmented_target).solution


def _weighted_complex_nmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    weight = weight.float().clamp_min(0.05)
    return (weight * (prediction - target).abs().square()).sum() / (
        (weight * target.abs().square()).sum().clamp_min(float(eps))
    )


def response_split_fit_loss(
    design: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    amplitude: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Fit A/predict B and fit B/predict A without reading TX labels."""
    mask_a, mask_b = stratified_response_split(amplitude)
    losses = []
    for row in range(int(design.size(0))):
        for fit_mask, eval_mask in ((mask_a[row], mask_b[row]), (mask_b[row], mask_a[row])):
            coef = _fit_response_coef(
                design[row, fit_mask], target[row, fit_mask], weight[row, fit_mask], ridge
            )
            prediction = design[row, eval_mask].to(torch.complex64) @ coef
            losses.append(
                _weighted_complex_nmse(
                    prediction, target[row, eval_mask], weight[row, eval_mask]
                )
            )
    return torch.stack(losses).mean() if losses else target.real.sum() * 0.0


def response_pair_cross_prediction_loss(
    resp_coef: torch.Tensor,
    design: torch.Tensor,
    target: torch.Tensor,
    pair_ids: List[str],
    clean_mask: torch.Tensor,
    leo_mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply clean coefficients to LEO content and LEO coefficients to clean content."""
    losses = []
    unique_pairs = list(dict.fromkeys(str(value) for value in pair_ids))
    for pair_id in unique_pairs:
        indices = [index for index, value in enumerate(pair_ids) if str(value) == pair_id]
        clean = [index for index in indices if bool(clean_mask[index])]
        leo = [index for index in indices if bool(leo_mask[index])]
        if len(clean) != 1 or len(leo) != 1:
            continue
        clean_index, leo_index = clean[0], leo[0]
        for source, destination in ((clean_index, leo_index), (leo_index, clean_index)):
            prediction = design[destination] @ resp_coef[source]
            losses.append(
                _weighted_complex_nmse(
                    prediction, target[destination], weight[destination]
                )
            )
    return torch.stack(losses).mean() if losses else target.real.sum() * 0.0


def response_surface_distance(
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
    variance_a: torch.Tensor,
    variance_b: torch.Tensor,
    *,
    tau: float = 1e-3,
) -> torch.Tensor:
    reliability = (variance_a.float() + variance_b.float() + float(tau)).reciprocal()
    return (reliability * (anchor_a - anchor_b).abs().square()).sum() / reliability.sum().clamp_min(1e-6)


def same_tx_cross_response_loss(
    resp_coef: torch.Tensor,
    design: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    receiver_id: torch.Tensor,
    day_id: torch.Tensor,
    label_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for left in range(int(design.size(0))):
        if not bool(label_mask[left]):
            continue
        for right in range(left + 1, int(design.size(0))):
            if not bool(label_mask[right]) or int(labels[left]) != int(labels[right]):
                continue
            if int(receiver_id[left]) == int(receiver_id[right]) and int(day_id[left]) == int(day_id[right]):
                continue
            for source, destination in ((left, right), (right, left)):
                prediction = design[destination] @ resp_coef[source]
                losses.append(
                    _weighted_complex_nmse(
                        prediction, target[destination], weight[destination]
                    )
                )
    return torch.stack(losses).mean() if losses else target.real.sum() * 0.0


def different_tx_cross_response_error(
    resp_coef: torch.Tensor,
    design: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    receiver_id: torch.Tensor,
    day_id: torch.Tensor,
    view_type: List[str],
    label_mask: torch.Tensor,
) -> torch.Tensor:
    """Diagnostic cross-prediction error for different TX under matched RX/day/view."""
    losses = []
    count = int(design.size(0))
    for left in range(count):
        if not bool(label_mask[left]):
            continue
        for right in range(left + 1, count):
            if not bool(label_mask[right]) or int(labels[left]) == int(labels[right]):
                continue
            if int(receiver_id[left]) != int(receiver_id[right]) or int(day_id[left]) != int(day_id[right]):
                continue
            if str(view_type[left]) != str(view_type[right]):
                continue
            for source, destination in ((left, right), (right, left)):
                prediction = design[destination] @ resp_coef[source]
                losses.append(
                    _weighted_complex_nmse(
                        prediction, target[destination], weight[destination]
                    )
                )
    if losses:
        return torch.stack(losses).mean()
    return target.real.new_full((), float("nan"))


def different_tx_response_ranking_loss(
    anchors: torch.Tensor,
    variance: torch.Tensor,
    labels: torch.Tensor,
    receiver_id: torch.Tensor,
    day_id: torch.Tensor,
    view_type: List[str],
    label_mask: torch.Tensor,
    *,
    margin: float = 0.5,
) -> torch.Tensor:
    losses = []
    count = int(anchors.size(0))
    for anchor_index in range(count):
        if not bool(label_mask[anchor_index]):
            continue
        positives = [
            index
            for index in range(count)
            if index != anchor_index
            and bool(label_mask[index])
            and int(labels[index]) == int(labels[anchor_index])
            and (
                int(receiver_id[index]) != int(receiver_id[anchor_index])
                or int(day_id[index]) != int(day_id[anchor_index])
            )
        ]
        negatives = [
            index
            for index in range(count)
            if bool(label_mask[index])
            and int(labels[index]) != int(labels[anchor_index])
            and int(receiver_id[index]) == int(receiver_id[anchor_index])
            and int(day_id[index]) == int(day_id[anchor_index])
            and str(view_type[index]) == str(view_type[anchor_index])
        ]
        if not positives or not negatives:
            continue
        positive = positives[0]
        negative = negatives[0]
        d_positive = response_surface_distance(
            anchors[anchor_index], anchors[positive], variance[anchor_index], variance[positive]
        )
        d_negative = response_surface_distance(
            anchors[anchor_index], anchors[negative], variance[anchor_index], variance[negative]
        )
        losses.append(F.relu(float(margin) + d_positive - d_negative))
    return torch.stack(losses).mean() if losses else anchors.real.sum() * 0.0


def basis_gauge_loss(
    design: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    losses = []
    identity = torch.eye(int(design.size(-1)), device=design.device, dtype=torch.complex64)
    for row in range(int(design.size(0))):
        w = weight[row].float().clamp_min(0.05)
        gram = design[row].conj().transpose(0, 1) @ (w.unsqueeze(-1) * design[row])
        diagonal = gram.diagonal().real.clamp_min(float(eps)).sqrt()
        correlation = gram / (diagonal[:, None] * diagonal[None, :])
        losses.append((correlation - identity).abs().square().mean())
    return torch.stack(losses).mean()


def response_gate_calibration_loss(
    rho: torch.Tensor,
    raw_correct: torch.Tensor,
    fused_correct: torch.Tensor,
    *,
    rho_max: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    rescue = (~raw_correct.bool()) & fused_correct.bool()
    harm = raw_correct.bool() & (~fused_correct.bool())
    active = rescue | harm
    probability = (rho.float() / float(rho_max)).clamp(1e-5, 1.0 - 1e-5)
    target = rescue.float()
    loss = (
        F.binary_cross_entropy(probability[active], target[active])
        if bool(active.any())
        else probability.sum() * 0.0
    )
    return loss, {
        "rescue": int(rescue.sum().item()),
        "harm": int(harm.sum().item()),
        "unchanged": int((~active).sum().item()),
    }

