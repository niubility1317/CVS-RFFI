import math
import time
import argparse
import json
import random
from copy import deepcopy
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
from dataset_wisig import load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx
try:
    from model_dual_cvsincnet import build_dual_model
except Exception:
    from model_dual_cvsincnet_stagewise_v2 import build_dual_model
from DataAugmentation import build_augmentor, apply_receiver_dg
from training_controls import (
    collapse_guard_decision,
    compute_mixstyle_epoch_state,
    parse_sat_scenarios,
    sat_channel_config_for_scenario,
)
try:
    from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
except Exception:
    SatSimConfig = None
    apply_sat_gnd_channel_batch = None
try:
    from sgc_losses import residual_regularization
except Exception:
    residual_regularization = None


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        try:
            fv = float(v)
        except Exception:
            return
        if not math.isfinite(fv):
            return
        self.sum += fv * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return (self.sum / self.count) if self.count > 0 else float("nan")


class NanMeter:
    def __init__(self):
        self.values = []

    def update(self, v):
        if v is None:
            return
        try:
            fv = float(v)
        except Exception:
            return
        if math.isnan(fv):
            return
        self.values.append(fv)

    @property
    def avg(self):
        return float(np.mean(self.values)) if len(self.values) > 0 else float("nan")

    @property
    def count(self):
        return len(self.values)


def ecc_tau_for_epoch(
    epoch: int,
    tau_start: float,
    tau_end: float,
    ecc_epochs: int,
    start_epoch: int = 1,
    schedule: str = "cosine",
) -> float:
    """Confidence cap threshold schedule used by Early Confidence Cap loss."""
    start_epoch = int(start_epoch)
    ecc_epochs = max(1, int(ecc_epochs))
    if int(epoch) <= start_epoch:
        return float(tau_start)
    if int(epoch) >= start_epoch + ecc_epochs:
        return float(tau_end)
    t = (float(epoch) - float(start_epoch)) / float(ecc_epochs)
    t = max(0.0, min(1.0, t))
    if str(schedule).lower().strip() == "linear":
        a = t
    else:
        a = 0.5 - 0.5 * math.cos(math.pi * t)
    return float(tau_start) + (float(tau_end) - float(tau_start)) * a


def ecc_weight_for_epoch(
    epoch: int,
    lambda_ecc: float,
    ecc_epochs: int,
    start_epoch: int = 1,
    schedule: str = "cosine",
) -> float:
    """Decay ECC strength so it regularizes early learning and turns off later."""
    base = float(lambda_ecc)
    if base <= 0.0:
        return 0.0
    start_epoch = int(start_epoch)
    ecc_epochs = max(1, int(ecc_epochs))
    if int(epoch) < start_epoch or int(epoch) >= start_epoch + ecc_epochs:
        return 0.0
    t = (float(epoch) - float(start_epoch)) / float(ecc_epochs)
    t = max(0.0, min(1.0, t))
    if str(schedule).lower().strip() == "linear":
        return base * (1.0 - t)
    return base * (0.5 + 0.5 * math.cos(math.pi * t))


def compute_ecc_loss(
    logits: torch.Tensor,
    tau: float,
    gate: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Penalize over-confident predictions above a scheduled probability cap."""
    probs = torch.softmax(logits.float(), dim=1)
    max_prob = probs.max(dim=1).values
    penalty = torch.relu(max_prob - float(tau)).pow(2)
    if gate is not None:
        gate_f = gate.to(device=penalty.device, dtype=penalty.dtype).view(-1)
        if gate_f.numel() != penalty.numel():
            raise ValueError(f"ECC gate shape mismatch: gate={tuple(gate_f.shape)} penalty={tuple(penalty.shape)}")
        denom = gate_f.sum().clamp_min(1.0)
        loss = (penalty * gate_f).sum() / denom
    else:
        loss = penalty.mean()
    return loss, float(max_prob.detach().mean().item())


def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra


def extract_domain_from_extra(extra, device) -> Optional[torch.Tensor]:
    if extra is None or len(extra) == 0:
        return None
    d0 = extra[0]
    if torch.is_tensor(d0):
        return d0.to(device, non_blocking=True).view(-1)
    try:
        return torch.as_tensor(d0, device=device).view(-1)
    except Exception:
        return None


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


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


class SSDGPseudoLabelMemory:
    """Track stable pseudo-label predictions for unlabeled SSDG samples."""

    def __init__(self, num_classes: int, momentum: float = 0.9):
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.probs: Dict[int, torch.Tensor] = {}
        self.pred: Dict[int, int] = {}
        self.streak: Dict[int, int] = {}

    @torch.no_grad()
    def update(self, sample_ids: torch.Tensor, probs: torch.Tensor, confidence_threshold: float = 0.9):
        ids = sample_ids.detach().view(-1).long().cpu()
        p = torch.nan_to_num(probs.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
        p = p[:, : self.num_classes]
        if int(p.numel()) == 0:
            empty_long = torch.empty(0, dtype=torch.long)
            return empty_long, torch.empty(0, dtype=torch.float32), empty_long
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        preds = []
        confs = []
        streaks = []
        for i, sid_t in enumerate(ids):
            sid = int(sid_t.item())
            cur = p[i]
            if sid in self.probs:
                cur = self.momentum * self.probs[sid] + (1.0 - self.momentum) * cur
                cur = cur / cur.sum().clamp_min(1e-12)
            pred = int(cur.argmax().item())
            conf = float(cur[pred].item())
            if conf >= float(confidence_threshold) and self.pred.get(sid) == pred:
                stable = self.streak.get(sid, 0) + 1
            elif conf >= float(confidence_threshold):
                stable = 1
            else:
                stable = 0
            self.probs[sid] = cur
            self.pred[sid] = pred
            self.streak[sid] = stable
            preds.append(pred)
            confs.append(conf)
            streaks.append(stable)
        return (
            torch.tensor(preds, dtype=torch.long),
            torch.tensor(confs, dtype=torch.float32),
            torch.tensor(streaks, dtype=torch.long),
        )


class AveragedModelState:
    """EMA/SWA/SWAD-style online weight averaging."""

    def __init__(self, mode: str, decay: float = 0.999):
        self.mode = str(mode)
        self.decay = float(decay)
        self.n = 0
        self.avg: Dict[str, torch.Tensor] = {}
        self.non_float: Dict[str, torch.Tensor] = {}
        self.epochs: List[int] = []

    def update(self, model, epoch: int, *, ema: bool = False) -> None:
        state = getattr(model, "_orig_mod", model).state_dict()
        with torch.no_grad():
            for k, v in state.items():
                vv = v.detach()
                if torch.is_floating_point(vv):
                    vf = vv.float().clone()
                    if k not in self.avg:
                        self.avg[k] = vf
                    elif ema:
                        self.avg[k].mul_(float(self.decay)).add_(vf, alpha=1.0 - float(self.decay))
                    else:
                        self.avg[k].mul_(float(self.n) / float(self.n + 1)).add_(vf, alpha=1.0 / float(self.n + 1))
                else:
                    self.non_float[k] = vv.clone()
        self.n += 1
        self.epochs.append(int(epoch))

    def has_state(self) -> bool:
        return self.n > 0 and len(self.avg) > 0

    def averaged_state_dict(self, model) -> Dict[str, torch.Tensor]:
        ref_state = getattr(model, "_orig_mod", model).state_dict()
        out = {}
        for k, v in ref_state.items():
            if k in self.avg:
                out[k] = self.avg[k].to(device=v.device, dtype=v.dtype)
            elif k in self.non_float:
                out[k] = self.non_float[k].to(device=v.device, dtype=v.dtype)
            else:
                out[k] = v.detach().clone()
        return out

    def cpu_state_dict(self, model) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.averaged_state_dict(model).items()}


def save_checkpoint(path: str, *, model, optimizer, scheduler, scaler, epoch: int, args, split_info, stats: dict):
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    state_model = getattr(model, "_orig_mod", model)
    payload = {
        "model": state_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
        "split_info": split_info,
        "stats": stats,
    }
    torch.save(payload, path)


def parse_csv_indices(s: str):
    s = str(s).strip()
    if s == "":
        return None
    out = []
    for item in s.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            out.append(int(item))
        except Exception:
            out.append(item)
    return out if len(out) > 0 else None


def parse_float_csv(s: str, default: Optional[List[float]] = None) -> List[float]:
    ss = str(s).strip()
    if ss == "":
        return list(default or [])
    out = []
    for item in ss.split(","):
        item = item.strip()
        if item == "":
            continue
        out.append(float(item))
    return out if len(out) > 0 else list(default or [])


def sample_strength_from_tiers(batch_size: int, tiers: List[float], device, dtype=torch.float32) -> torch.Tensor:
    if tiers is None or len(tiers) == 0:
        return torch.rand((batch_size,), device=device, dtype=dtype)
    vals = torch.as_tensor(tiers, device=device, dtype=dtype).clamp(0.0, 1.0)
    idx = torch.randint(low=0, high=int(vals.numel()), size=(batch_size,), device=device)
    return vals[idx]


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_true: str, help_false: str):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name.replace('-', '_'), action="store_true", help=help_true)
    group.add_argument(f"--no_{name}", dest=name.replace('-', '_'), action="store_false", help=help_false)
    parser.set_defaults(**{name.replace('-', '_'): default})


def parse_json_dict(value, name: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "{}").strip()
    if text == "":
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--{name} must be valid JSON object text: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"--{name} must decode to a JSON object")
    return parsed


def safe_nan(v: float) -> str:
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"


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


def safe_l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(float(eps))
    return x / n


def safe_cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return (safe_l2_normalize(a, dim=dim, eps=eps) * safe_l2_normalize(b, dim=dim, eps=eps)).sum(dim=dim).clamp(-1.0, 1.0)


def safe_batch_var(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size(dim) <= 1:
        return torch.zeros_like(x.mean(dim=dim))
    return x.var(dim=dim, unbiased=False).clamp_min(float(eps))


def safe_batch_std(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    return safe_batch_var(x, dim=dim, eps=eps).sqrt()


def safe_iq_tensor(x: torch.Tensor, clamp: float = 8.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(-float(clamp), float(clamp))


def batch_domain_stats(d: Optional[torch.Tensor], y: torch.Tensor, num_domains: int) -> Dict[str, Any]:
    valid = None
    if d is not None:
        valid = d.view(-1).long() >= 0
    if d is None or valid is None or not bool(valid.any()):
        return {"valid": valid, "num_valid": 0, "num_domains": 0, "domain_frac": 0.0, "has_cross_pairs": False}
    d_valid = d.view(-1).long()[valid]
    y_valid = y.view(-1).long()[valid]
    uniq_d = torch.unique(d_valid)
    has_cross_pairs = False
    for cls in torch.unique(y_valid):
        if torch.unique(d_valid[y_valid == cls]).numel() >= 2:
            has_cross_pairs = True
            break
    return {
        "valid": valid,
        "num_valid": int(d_valid.numel()),
        "num_domains": int(uniq_d.numel()),
        "domain_frac": float(uniq_d.numel()) / float(max(1, num_domains)),
        "has_cross_pairs": bool(has_cross_pairs),
    }


def domain_loss_gates(domain_stats: Dict[str, Any], args, num_domains: int) -> Dict[str, bool]:
    min_domains = int(getattr(args, "min_batch_domains_for_domain_loss", 2))
    min_frac = float(getattr(args, "min_batch_domain_frac", 0.15))
    enough_domains = (
        num_domains > 1
        and int(domain_stats.get("num_domains", 0)) >= max(2, min_domains)
        and float(domain_stats.get("domain_frac", 0.0)) >= min_frac
    )
    has_pairs = enough_domains and bool(domain_stats.get("has_cross_pairs", False))
    return {
        "dom": enough_domains,
        "adv": enough_domains,
        "cons": has_pairs,
        "group_ce": enough_domains,
    }


def metric_or_neg_inf(stats: Dict[str, Any], key: str = "tx_acc") -> float:
    """Read a metric safely for best-checkpoint comparisons."""
    try:
        v = float(stats.get(key, float("-inf")))
    except Exception:
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def derive_checkpoint_path(base_path: str, suffix: str) -> str:
    """Derive a checkpoint path when user does not provide one explicitly.

    Example:
      best_model.pth + test_overall -> best_model_test_overall.pth
    """
    base_path = str(base_path).strip() or "best_model.pth"
    root, ext = os.path.splitext(base_path)
    if ext == "":
        ext = ".pth"
    return f"{root}_{suffix}{ext}"


def compute_primary_ood_score(test_overall: float, unseen_day_unseen_rx: float, udu_weight: float) -> float:
    if not math.isfinite(float(test_overall)):
        test_overall = float("-inf")
    if not math.isfinite(float(unseen_day_unseen_rx)):
        unseen_day_unseen_rx = float("-inf")
    w = max(0.0, min(1.0, float(udu_weight)))
    if not math.isfinite(test_overall) or not math.isfinite(unseen_day_unseen_rx):
        return max(float(test_overall), float(unseen_day_unseen_rx))
    return (1.0 - w) * float(test_overall) + w * float(unseen_day_unseen_rx)


def compute_worst_unseen_rx_score(named_test_stats: Dict[str, Dict[str, Any]]) -> Tuple[float, str]:
    rx_scores: List[Tuple[str, float]] = []
    for name, stats in named_test_stats.items():
        if not str(name).startswith("test_rx_"):
            continue
        score = metric_or_neg_inf(stats, "tx_acc")
        if math.isfinite(score):
            rx_scores.append((str(name), float(score)))
    if not rx_scores:
        return float("-inf"), ""
    worst_name, worst_score = min(rx_scores, key=lambda x: x[1])
    return float(worst_score), worst_name


def default_is_path(p: str, default_name: str) -> bool:
    return str(p).strip() == default_name


def set_pa_weights(args, *, cls_pa: float, joint_inv: float, imp_inv: float, pa_kl: float, pa_reg: float, pa_select: float, pa_mono: float):
    args.lambda_cls_pa = float(cls_pa)
    args.lambda_pa_joint_inv = float(joint_inv)
    args.lambda_pa_imp_inv = 0.0
    args.lambda_pa_kl = float(pa_kl)
    args.lambda_pa_reg = float(pa_reg)
    args.lambda_pa_select = 0.0
    args.lambda_pa_mono = 0.0


def zero_pa_path(args):
    args.enable_pa_aux = False
    args.aug_enable_pa_normal = False
    args.aug_p_pa = 0.0
    set_pa_weights(args, cls_pa=0.0, joint_inv=0.0, imp_inv=0.0, pa_kl=0.0, pa_reg=0.0, pa_select=0.0, pa_mono=0.0)


def zero_dac_path(args):
    args.enable_dac_aux = False
    args.aug_p_dac = 0.0
    args.lambda_cls_dac = 0.0
    args.lambda_dac_reg = 0.0
    args.lambda_dac_select = 0.0
    args.lambda_dac_mono = 0.0


def set_dac_weights(args, *, cls_dac: float, dac_reg: float, dac_select: float, dac_mono: float):
    args.lambda_cls_dac = float(cls_dac)
    args.lambda_dac_reg = float(dac_reg)
    args.lambda_dac_select = 0.0
    args.lambda_dac_mono = 0.0


def parse_branch_ablation_flags(branch_ablation: str) -> frozenset[str]:
    raw = str(branch_ablation or "none").lower().replace(";", ",").replace("+", ",")
    aliases = {
        "none": "",
        "base": "",
        "off": "",
        "no_time_branch": "no_time",
        "no_dac_branch": "no_dac",
        "no_pa_branch": "no_pa",
        "no_freq_branch": "no_freq",
        "no_spectral": "no_freq",
        "no_spec": "no_freq",
        "no_stat": "no_stats",
        "no_spectral_stats": "no_stats",
        "no_dac_pa": "no_dac,no_pa",
        "no_physical": "no_dac,no_pa",
        "time_only": "no_dac,no_pa,no_freq,no_stats",
        "freq_only": "no_time,no_dac,no_pa,no_stats",
        "no_defect_branches": "no_dac,no_pa",
    }
    expanded = []
    for item in raw.split(","):
        item = item.strip()
        if item == "":
            continue
        item = aliases.get(item, item)
        expanded.extend([z.strip() for z in item.split(",") if z.strip()])
    return frozenset(expanded)


def apply_experiment_preset(args):
    g_raw = str(args.exp_group).strip()
    alias_map = {
        # new stagewise names
        "s1_core_only": "s1_core_only",
        "s2_pure_aux_no_select": "s2_pure_aux_no_select",
        "s3_stagewise_pa_focus": "s3_stagewise_pa_focus",
        "s4_stagewise_full_dual": "s4_stagewise_full_dual",
        "s3_stable_no_dac": "s3_stable_no_dac",
        "s4_late_stable_full": "s4_late_stable_full",
        "s3_rxrobust_no_dac": "s3_rxrobust_no_dac",
        "s4_rxrobust_full": "s4_rxrobust_full",
        # backward-compatible aliases from older versions
        "g1_true_no_pa": "s1_core_only",
        "g2_pa_aux_only": "s3_stagewise_pa_focus",
        "g3_pa_main_only": "s3_stagewise_pa_focus",
        "g4_pa_main_plus_aux": "s3_stagewise_pa_focus",
        "g5_full_dual_puredefect": "s4_stagewise_full_dual",
    }
    if g_raw not in alias_map:
        valid = ", ".join(sorted(alias_map.keys()))
        raise ValueError(f"Unknown exp_group={g_raw}. Valid values: {valid}")
    g = alias_map[g_raw]
    args.exp_group = g

    # Shared defaults
    args.use_aug = True
    args.aug_enable_class_signature = False
    args.aug_scale_min = 0.10
    args.aug_scale_max = 0.35
    args.aug_warmup_epochs = 3
    args.aug_ramp_epochs = 15
    args.aug_ramp_curve = 1.25
    args.aux_warmup_epochs = 3
    args.aux_ramp_epochs = 15
    args.robust_temp = 1.0
    args.select_margin = 0.03
    args.mono_margin = 0.00
    args.aug_pa_mp_sigma = 0.04
    args.aug_pa_mem_sigma = 0.03
    args.aug_pa_ampm_max = 0.15
    args.aug_pa_iq_img_max = 0.010

    # Stagewise defaults: pure defect-only views, mixed DAC+PA view may still include channel.
    args.aug_dac_only_apply_anti_shortcut = False
    args.aug_dac_only_apply_channel = False
    args.aug_pa_only_apply_anti_shortcut = False
    args.aug_pa_only_apply_channel = False
    args.aug_dac_pa_apply_anti_shortcut = True
    args.aug_dac_pa_apply_channel = True
    args.aug_defect_strength_mode = "tiered"
    args.aug_dac_only_tiers = "0.15,0.35,0.55"
    args.aug_pa_only_tiers = "0.15,0.35,0.60"

    # Clean slate before per-group setup.
    args.lambda_cross_zero = 0.0
    args.stage1_epochs = int(getattr(args, "stage1_epochs", 15))
    args.stage2_epochs = int(getattr(args, "stage2_epochs", 45))
    args.stage3_ramp_epochs = int(getattr(args, "stage3_ramp_epochs", 20))
    args.late_stable_start = int(getattr(args, "late_stable_start", 0))
    args.late_stable_ramp_epochs = int(getattr(args, "late_stable_ramp_epochs", 12))
    args.late_adv_min_scale = float(getattr(args, "late_adv_min_scale", 0.75))
    args.late_cons_min_scale = float(getattr(args, "late_cons_min_scale", 0.55))
    args.late_cls_aux_min_scale = float(getattr(args, "late_cls_aux_min_scale", 0.35))
    args.late_reg_aux_min_scale = float(getattr(args, "late_reg_aux_min_scale", 0.35))
    args.late_joint_inv_min_scale = float(getattr(args, "late_joint_inv_min_scale", 0.12))
    args.late_kl_min_scale = float(getattr(args, "late_kl_min_scale", 0.25))
    args.late_group_ce_min_scale = float(getattr(args, "late_group_ce_min_scale", 0.75))
    args.late_aug_min_scale = float(getattr(args, "late_aug_min_scale", -1.0))

    # Base defaults for DAC/PA paths.
    args.enable_dac_aux = False
    args.enable_pa_aux = False
    args.aug_enable_pa_normal = False
    args.aug_p_dac = 0.0
    args.aug_p_pa = 0.0
    args.lambda_cls_dac = 0.0
    args.lambda_dac_reg = 0.0
    args.lambda_dac_select = 0.0
    args.lambda_dac_mono = 0.0
    set_pa_weights(args, cls_pa=0.0, joint_inv=0.0, imp_inv=0.0, pa_kl=0.0, pa_reg=0.0, pa_select=0.0, pa_mono=0.0)

    if g == "s1_core_only":
        args.exp_desc = "S1 仅主任务：cls + dom/adv/orth/cons，不启用 DAC/PA 辅助"
        args.stage1_epochs = max(args.stage1_epochs, 999999)
        args.stage2_epochs = max(args.stage2_epochs, 999999)
    elif g == "s2_pure_aux_no_select":
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        args.aug_p_dac = 0.22
        args.lambda_cls_dac = 0.10
        args.lambda_dac_reg = 0.25
        args.lambda_dac_select = 0.0
        args.lambda_dac_mono = 0.0
        set_pa_weights(args, cls_pa=0.30, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.0, pa_mono=0.0)
        args.exp_desc = "S2 纯辅助无选择性：纯 DAC/PA-only 视图 + 轻 reg/joint_inv/kl，不启用 select/mono/cross_zero"
    elif g == "s3_stagewise_pa_focus":
        args.enable_dac_aux = False
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        set_pa_weights(args, cls_pa=0.32, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.06, pa_mono=0.04)
        args.exp_desc = "S3 PA 重点阶段式：主视图温和 PA + 纯 PA-only 辅助，仅保留 joint/kl/reg"
    elif g == "s4_stagewise_full_dual":
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        args.aug_p_dac = 0.22
        args.lambda_cls_dac = 0.10
        args.lambda_dac_reg = 0.25
        args.lambda_dac_select = 0.0
        args.lambda_dac_mono = 0.0
        set_pa_weights(args, cls_pa=0.30, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.08, pa_mono=0.05)
        args.lambda_cross_zero = 0.0
        args.exp_desc = "S4 双缺陷阶段式：joint 特征去域 + 纯 DAC/PA-only 辅助，移除 select/mono/cross_zero"
    elif g == "s3_stable_no_dac":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = False
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.16
        args.stage1_epochs = max(12, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 24, int(round(total_epochs * 0.38)))
        args.stage3_ramp_epochs = max(8, int(round(total_epochs * 0.08)))
        args.late_stable_start = max(52, int(round(total_epochs * 0.65)))
        args.late_stable_ramp_epochs = max(12, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.50
        args.late_cls_aux_min_scale = 0.30
        args.late_reg_aux_min_scale = 0.30
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.18
        args.late_group_ce_min_scale = 0.75
        args.late_aug_min_scale = 0.22
        set_pa_weights(args, cls_pa=0.24, joint_inv=0.08, imp_inv=0.00, pa_kl=0.03, pa_reg=0.12, pa_select=0.0, pa_mono=0.0)
        args.exp_desc = "S3 stable no-DAC: PA-only auxiliary path with late loss decay for OOD stability"
    elif g == "s4_late_stable_full":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.14
        args.aug_p_dac = 0.12
        args.lambda_cls_dac = 0.05
        args.lambda_dac_reg = 0.10
        args.stage1_epochs = max(12, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 24, int(round(total_epochs * 0.36)))
        args.stage3_ramp_epochs = max(8, int(round(total_epochs * 0.08)))
        args.late_stable_start = max(50, int(round(total_epochs * 0.65)))
        args.late_stable_ramp_epochs = max(12, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.45
        args.late_cls_aux_min_scale = 0.25
        args.late_reg_aux_min_scale = 0.25
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.16
        args.late_group_ce_min_scale = 0.75
        args.late_aug_min_scale = 0.20
        set_pa_weights(args, cls_pa=0.22, joint_inv=0.08, imp_inv=0.00, pa_kl=0.03, pa_reg=0.12, pa_select=0.0, pa_mono=0.0)
        args.lambda_cross_zero = 0.0
        args.exp_desc = "S4 late-stable full: reduced DAC/PA auxiliary pressure and late loss decay"
    elif g == "s3_rxrobust_no_dac":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = False
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.14
        args.lambda_adv = 0.45
        args.lambda_cons = 0.08
        args.lambda_group_ce = 0.10
        args.group_ce_top_frac = 0.35
        args.group_ce_min_domains = 4
        args.stage1_epochs = max(16, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 40, int(round(total_epochs * 0.40)))
        args.stage3_ramp_epochs = max(16, int(round(total_epochs * 0.10)))
        args.late_stable_start = max(120, int(round(total_epochs * 0.68)))
        args.late_stable_ramp_epochs = max(30, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.45
        args.late_cls_aux_min_scale = 0.25
        args.late_reg_aux_min_scale = 0.25
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.16
        args.late_group_ce_min_scale = 0.80
        args.late_aug_min_scale = 0.20
        set_pa_weights(args, cls_pa=0.20, joint_inv=0.06, imp_inv=0.00, pa_kl=0.02, pa_reg=0.10, pa_select=0.0, pa_mono=0.0)
        args.exp_desc = "S3 RX-robust no-DAC: PA-only aux + hard-domain CE for weak unseen receivers"
    elif g == "s4_rxrobust_full":
        total_epochs = int(max(1, getattr(args, "epochs", 100)))
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.12
        args.aug_p_dac = 0.08
        args.lambda_adv = 0.45
        args.lambda_cons = 0.08
        args.lambda_group_ce = 0.08
        args.group_ce_top_frac = 0.35
        args.group_ce_min_domains = 4
        args.lambda_cls_dac = 0.03
        args.lambda_dac_reg = 0.06
        args.stage1_epochs = max(16, int(round(total_epochs * 0.08)))
        args.stage2_epochs = max(args.stage1_epochs + 40, int(round(total_epochs * 0.40)))
        args.stage3_ramp_epochs = max(16, int(round(total_epochs * 0.10)))
        args.late_stable_start = max(120, int(round(total_epochs * 0.68)))
        args.late_stable_ramp_epochs = max(30, int(round(total_epochs * 0.18)))
        args.late_adv_min_scale = 0.70
        args.late_cons_min_scale = 0.45
        args.late_cls_aux_min_scale = 0.25
        args.late_reg_aux_min_scale = 0.25
        args.late_joint_inv_min_scale = 0.08
        args.late_kl_min_scale = 0.16
        args.late_group_ce_min_scale = 0.80
        args.late_aug_min_scale = 0.20
        set_pa_weights(args, cls_pa=0.18, joint_inv=0.06, imp_inv=0.00, pa_kl=0.02, pa_reg=0.10, pa_select=0.0, pa_mono=0.0)
        args.lambda_cross_zero = 0.0
        args.exp_desc = "S4 RX-robust full/no-stats friendly: weak DAC/PA aux + hard-domain CE for low RX groups"
    else:
        raise ValueError(f"Internal exp_group dispatch failure: {g}")

    if not args.enable_pa_aux or not args.enable_dac_aux:
        args.lambda_cross_zero = 0.0

    return args


def apply_slim_ablation_preset(args):
    g = str(getattr(args, "slim_group", "none") or "none").lower().strip()
    sgc_full_kwargs = {
        "use_amp_norm": True,
        "use_freq_comp": True,
        "use_spectral_suppressor": True,
        "use_residual_comp": True,
        "freq_hidden_dim": 32,
        "max_norm_freq_offset": 0.05,
        "spectral_hidden_dim": 32,
        "spectral_residual_alpha": 0.5,
        "residual_channels": 32,
        "residual_blocks": 2,
        "residual_kernel_size": 5,
        "residual_init_gamma": 0.0,
    }
    table = {
        "none": {
            "desc": "不额外覆盖结构预设，完全使用手动配置。",
        },
        "balanced": {
            "model_variant": "lite_c",
            "branch_ablation": "none",
            "exp_group": "s4_stagewise_full_dual",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "默认推荐：Lite-C 全分支，兼顾精度、参数量和推理延迟。",
        },
        "no_dac": {
            "model_variant": "lite_c",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stagewise_pa_focus",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "优先减时延：去掉 DAC 分支，保留 PA 分支与频域摘要。",
        },
        "no_stats": {
            "model_variant": "lite_c",
            "branch_ablation": "no_stats",
            "exp_group": "s4_stagewise_full_dual",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "轻量裁剪：保留频域卷积，移除频谱统计投影，主要测试小幅时延收益。",
        },
        "lite_b": {
            "model_variant": "lite_b",
            "branch_ablation": "none",
            "exp_group": "s4_stagewise_full_dual",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "参数优先：Lite-B 全分支，适合先测结构压缩对精度的影响。",
        },
        "lite_b_no_dac": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stagewise_pa_focus",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "推荐第二梯队：Lite-B + 去 DAC，进一步压缩参数并降低推理时延。",
        },
        "lite_d_no_dac": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stagewise_pa_focus",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "mixstyle_p": 0.15,
            "desc": "更激进的小模型：Lite-D + 去 DAC，适合主力瘦身实验。",
        },
        "lite_e_time_only": {
            "model_variant": "lite_e",
            "branch_ablation": "time_only",
            "exp_group": "s1_core_only",
            "use_mixstyle": False,
            "desc": "极限小模型：Lite-E + 仅时间分支，用来测最小参数/最低时延边界。",
        },
    }
    table.update({
        "balanced_stable": {
            "model_variant": "lite_c",
            "branch_ablation": "none",
            "exp_group": "s4_late_stable_full",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "Lite-C full branches with reduced auxiliary pressure and late-stage loss decay.",
        },
        "lite_b_no_dac_stable": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stable_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "desc": "Recommended stable deployment candidate: Lite-B + structural no-DAC.",
        },
        "lite_d_no_dac_stable": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "exp_group": "s3_stable_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "crossdomain",
            "mixstyle_p": 0.2,
            "desc": "Compact stable candidate: Lite-D + structural no-DAC.",
        },
        "rxrobust_balanced": {
            "model_variant": "lite_c",
            "branch_ablation": "none",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s4_rxrobust_full",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.10,
            "mixstyle_late_min_strength": 0.45,
            "desc": "Lite-C full branches with hard-domain CE for weak receiver groups.",
        },
        "rxrobust_no_stats": {
            "model_variant": "lite_c",
            "branch_ablation": "no_stats",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s4_rxrobust_full",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 90,
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": 0.05,
            "mixstyle_late_min_strength": 0.30,
            "desc": "Best low-RX direction from 4.26 logs: no handcrafted stats + hard-domain CE.",
        },
        "rxrobust_lite_b_no_dac": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.08,
            "mixstyle_late_min_strength": 0.40,
            "desc": "Lite-B no-DAC with hard-domain CE and delayed late stabilization.",
        },
        "rxrobust_lite_d_no_dac": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.20,
            "mixstyle_late_start": 110,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.06,
            "mixstyle_late_min_strength": 0.35,
            "desc": "Lite-D no-DAC compact hard-domain CE candidate.",
        },
        "rxrobust_no_dac_no_stats": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac,no_stats",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 90,
            "mixstyle_late_ramp_epochs": 30,
            "mixstyle_late_min_p": 0.04,
            "mixstyle_late_min_strength": 0.30,
            "desc": "Tests whether removing DAC and stats together helps weak RX while staying compact.",
        },
        "rxrobust_lite_b_no_dac_refined": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.08,
            "mixstyle_late_min_strength": 0.40,
            "desc": "R05 refined default: best 4.27 deployment route with late MixStyle annealing.",
        },
        "rxrobust_lite_b_no_dac_mix015": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.65,
            "mixstyle_p": 0.15,
            "mixstyle_late_start": 110,
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": 0.05,
            "mixstyle_late_min_strength": 0.35,
            "desc": "R05 conservative MixStyle: lower p/strength for no-stats-sensitive domains.",
        },
        "rxrobust_lite_b_no_dac_domain020": {
            "model_variant": "lite_b",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "domain_enhancer_strength": 0.20,
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.75,
            "mixstyle_p": 0.25,
            "mixstyle_late_start": 120,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.08,
            "mixstyle_late_min_strength": 0.40,
            "desc": "R05 with weaker RCN enhancer injection; tests over-domainization.",
        },
        "rxrobust_lite_d_no_dac_refined": {
            "model_variant": "lite_d",
            "branch_ablation": "no_dac",
            "domain_branch_ablation": "no_stats",
            "domain_enhancer": "rcn_stats",
            "exp_group": "s3_rxrobust_no_dac",
            "use_mixstyle": True,
            "mixstyle_layers": "time_down,t1",
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_strength": 0.70,
            "mixstyle_p": 0.18,
            "mixstyle_late_start": 110,
            "mixstyle_late_ramp_epochs": 40,
            "mixstyle_late_min_p": 0.05,
            "mixstyle_late_min_strength": 0.32,
            "desc": "R06 refined compact route with gentler MixStyle for lower latency models.",
        },
    })
    for name, group_ce, desc in [
        ("rxrobust_lite_b_no_dac_gce006", 0.06, "R05 refined with weaker hard-domain CE."),
        ("rxrobust_lite_b_no_dac_gce014", 0.14, "R05 refined with stronger hard-domain CE."),
    ]:
        cfg = dict(table["rxrobust_lite_b_no_dac_refined"])
        cfg["lambda_group_ce"] = float(group_ce)
        cfg["desc"] = desc
        table[name] = cfg
    sgc_base = {
        "model_variant": "lite_b",
        "branch_ablation": "no_dac",
        "domain_branch_ablation": "no_stats",
        "domain_enhancer": "rcn_stats",
        "exp_group": "s3_rxrobust_no_dac",
        "sgc_adapter": True,
        "sgc_adapter_kwargs": sgc_full_kwargs,
        "use_mixstyle": True,
        "mixstyle_layers": "time_down,t1",
        "mixstyle_mix": "same_tx_crossdomain",
        "mixstyle_fallback": "skip",
        "mixstyle_strength": 0.65,
        "mixstyle_p": 0.15,
        "mixstyle_late_start": 110,
        "mixstyle_late_ramp_epochs": 35,
        "mixstyle_late_min_p": 0.05,
        "mixstyle_late_min_strength": 0.35,
    }
    table["sgc_lite_b_no_dac"] = {
        **sgc_base,
        "desc": "SGC-Adapter full module on the R19 Lite-B no-DAC baseline.",
    }
    for suffix, key, desc in [
        ("no_amp", "use_amp_norm", "SGC ablation without RMS amplitude normalization."),
        ("no_freq", "use_freq_comp", "SGC ablation without CFO/Doppler compensation."),
        ("no_spec", "use_spectral_suppressor", "SGC ablation without FFT-domain interference suppression."),
        ("no_res", "use_residual_comp", "SGC ablation without residual channel compensation."),
    ]:
        cfg = dict(sgc_base)
        kwargs = dict(sgc_full_kwargs)
        kwargs[key] = False
        cfg["sgc_adapter_kwargs"] = kwargs
        cfg["desc"] = desc
        table[f"sgc_lite_b_no_dac_{suffix}"] = cfg
    for name, kwargs_update, desc in [
        (
            "sgc_lite_b_no_dac_no_amp_freq",
            {"use_amp_norm": False, "use_freq_comp": False},
            "SGC combined ablation without amplitude normalization and CFO/Doppler compensation.",
        ),
        (
            "sgc_lite_b_no_dac_residual_only",
            {"use_amp_norm": False, "use_freq_comp": False, "use_spectral_suppressor": False, "use_residual_comp": True},
            "SGC residual-only control for checking whether learned residual compensation alone is enough.",
        ),
        (
            "sgc_lite_b_no_dac_light",
            {"freq_hidden_dim": 16, "spectral_hidden_dim": 16, "spectral_residual_alpha": 0.35, "residual_channels": 16, "residual_blocks": 1},
            "SGC light adapter with a smaller parameter budget for satellite-side deployment.",
        ),
    ]:
        cfg = dict(sgc_base)
        kwargs = dict(sgc_full_kwargs)
        kwargs.update(kwargs_update)
        cfg["sgc_adapter_kwargs"] = kwargs
        cfg["desc"] = desc
        table[name] = cfg
    lite_d_cfg = dict(sgc_base)
    lite_d_cfg["model_variant"] = "lite_d"
    lite_d_cfg["mixstyle_strength"] = 0.70
    lite_d_cfg["mixstyle_p"] = 0.18
    lite_d_cfg["desc"] = "SGC full adapter on the compact Lite-D no-DAC backbone."
    table["sgc_lite_d_no_dac"] = lite_d_cfg
    lite_d_light = dict(lite_d_cfg)
    lite_d_light_kwargs = dict(sgc_full_kwargs)
    lite_d_light_kwargs.update({
        "freq_hidden_dim": 16,
        "spectral_hidden_dim": 16,
        "spectral_residual_alpha": 0.35,
        "residual_channels": 16,
        "residual_blocks": 1,
    })
    lite_d_light["sgc_adapter_kwargs"] = lite_d_light_kwargs
    lite_d_light["desc"] = "SGC light adapter on Lite-D for the most compact deployable candidate."
    table["sgc_lite_d_no_dac_light"] = lite_d_light
    cfg = dict(sgc_base)
    cfg["sgc_adapter"] = False
    cfg["sgc_adapter_kwargs"] = {}
    cfg["desc"] = "R19 Lite-B no-DAC baseline without SGC-Adapter."
    table["sgc_baseline_no_adapter"] = cfg
    if g not in table:
        valid = ", ".join(sorted(table.keys()))
        raise ValueError(f"Unknown slim_group={g}. Valid values: {valid}")
    cfg = table[g]
    for key, value in cfg.items():
        if key == "desc":
            continue
        setattr(args, key, value)
    args.slim_group = g
    args.slim_desc = cfg.get("desc", "")
    return args


def apply_slim_post_preset_overrides(args):
    """Reapply slim-group values that intentionally override exp_group defaults."""
    g = str(getattr(args, "slim_group", "none") or "none").lower().strip()
    if g == "rxrobust_lite_b_no_dac_gce006":
        args.lambda_group_ce = 0.06
    elif g == "rxrobust_lite_b_no_dac_gce014":
        args.lambda_group_ce = 0.14
    return args


def apply_model_variant_training_defaults(args):
    variant = str(getattr(args, "model_variant", "base") or "base").lower().strip()
    args.lambda_probe = 0.0
    args.lambda_pa_imp_inv = 0.0
    args.lambda_cross_zero = 0.0
    args.lambda_dac_select = 0.0
    args.lambda_pa_select = 0.0
    args.lambda_dac_mono = 0.0
    args.lambda_pa_mono = 0.0
    if variant == "lite_c":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-C streamlined trainer"
    elif variant == "lite_d":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-D compact trunk"
    elif variant == "lite_e":
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | Lite-E tiny trunk"
    return args


def align_training_with_branch_ablation(args):
    ablated = parse_branch_ablation_flags(getattr(args, "branch_ablation", "none"))
    notes = []
    if "no_dac" in ablated:
        zero_dac_path(args)
        notes.append("no_dac->disable_dac_aux")
    if "no_pa" in ablated:
        zero_pa_path(args)
        notes.append("no_pa->disable_pa_aux")
    if "no_time" in ablated:
        args.use_mixstyle = False
        notes.append("no_time->disable_mixstyle")
    if notes:
        args.exp_desc = str(getattr(args, "exp_desc", "")) + " | " + ",".join(notes)
    return args


def ramp_value(epoch: int, epochs: int, warmup_epochs: int, ramp_epochs: int, min_scale: float, max_scale: float, curve: float = 1.0) -> float:
    if max_scale <= min_scale:
        return float(max_scale)
    if epoch <= warmup_epochs:
        return float(min_scale)
    if ramp_epochs <= 0:
        return float(max_scale)
    t = (epoch - warmup_epochs) / float(ramp_epochs)
    t = max(0.0, min(1.0, t))
    t = t ** max(1e-6, float(curve))
    return float(min_scale + (max_scale - min_scale) * t)


def apply_late_stability(epoch: int, args, stage_state: Dict[str, float]) -> Dict[str, float]:
    start = int(getattr(args, "late_stable_start", 0))
    if start <= 0 or epoch < start:
        return stage_state

    ramp_epochs = int(max(1, getattr(args, "late_stable_ramp_epochs", 12)))
    t = ramp_value(epoch, args.epochs, start, ramp_epochs, 0.0, 1.0, 1.25)

    decay_targets = {
        "adv_scale": float(getattr(args, "late_adv_min_scale", stage_state["adv_scale"])),
        "cons_scale": float(getattr(args, "late_cons_min_scale", stage_state["cons_scale"])),
        "cls_aux_scale": float(getattr(args, "late_cls_aux_min_scale", stage_state["cls_aux_scale"])),
        "reg_aux_scale": float(getattr(args, "late_reg_aux_min_scale", stage_state["reg_aux_scale"])),
        "joint_inv_scale": float(getattr(args, "late_joint_inv_min_scale", stage_state["joint_inv_scale"])),
        "kl_scale": float(getattr(args, "late_kl_min_scale", stage_state["kl_scale"])),
        "group_ce_scale": float(getattr(args, "late_group_ce_min_scale", stage_state["group_ce_scale"])),
    }
    out = dict(stage_state)
    for key, target in decay_targets.items():
        target = max(0.0, min(float(out[key]), target))
        out[key] = float(out[key]) * (1.0 - t) + target * t
    out["phase"] = str(out["phase"]) + "_late_stable"
    return out


def build_stage_state(epoch: int, args) -> Dict[str, float]:
    e1 = int(max(0, args.stage1_epochs))
    e2 = int(max(e1, args.stage2_epochs))
    r3 = int(max(1, args.stage3_ramp_epochs))

    if epoch <= e1:
        return apply_late_stability(epoch, args, {
            "phase": "S1_core",
            "use_aux_views": 0.0,
            "dom_scale": 1.0,
            "adv_scale": 0.70,
            "orth_scale": 0.50,
            "cons_scale": 0.00,
            "cls_aux_scale": 0.0,
            "reg_aux_scale": 0.0,
            "joint_inv_scale": 0.0,
            "kl_scale": 0.0,
            "group_ce_scale": 0.50,
        })

    if epoch <= e2:
        t = ramp_value(epoch, args.epochs, e1, max(1, e2 - e1), 0.0, 1.0, 1.75)
        return apply_late_stability(epoch, args, {
            "phase": "S2_stabilize_aux",
            "use_aux_views": 1.0,
            "dom_scale": 1.0,
            "adv_scale": 0.70 + 0.30 * t,
            "orth_scale": 1.0,
            "cons_scale": 0.20 + 0.55 * t,
            "cls_aux_scale": 0.15 + 0.55 * t,
            "reg_aux_scale": 0.35 + 0.45 * t,
            "joint_inv_scale": 0.15 + 0.20 * t,
            "kl_scale": 0.15 + 0.35 * t,
            "group_ce_scale": 0.70 + 0.30 * t,
        })

    late = ramp_value(epoch, args.epochs, e2, r3, 0.0, 1.0, 1.75)
    return apply_late_stability(epoch, args, {
        "phase": "S3_refine_aux",
        "use_aux_views": 1.0,
        "dom_scale": 1.0,
        "adv_scale": 1.0,
        "orth_scale": 1.0,
        "cons_scale": 0.85 + 0.15 * late,
        "cls_aux_scale": 0.80 + 0.20 * late,
        "reg_aux_scale": 0.85 + 0.15 * late,
        "joint_inv_scale": 0.25 + 0.05 * late,
        "kl_scale": 0.50 + 0.10 * late,
        "group_ce_scale": 1.0,
    })


def current_weight_dict(args, stage_state: Dict[str, float]) -> Dict[str, float]:
    return {
        "dom": float(args.lambda_dom) * float(stage_state["dom_scale"]),
        "adv": float(args.lambda_adv) * float(stage_state["adv_scale"]),
        "orth": float(args.lambda_orth) * float(stage_state["orth_scale"]),
        "cons": float(args.lambda_cons) * float(stage_state["cons_scale"]),
        "cls_pa": float(args.lambda_cls_pa) * float(stage_state["cls_aux_scale"]),
        "cls_dac": float(args.lambda_cls_dac) * float(stage_state["cls_aux_scale"]),
        "pa_joint_inv": float(args.lambda_pa_joint_inv) * float(stage_state["joint_inv_scale"]),
        "pa_kl": float(args.lambda_pa_kl) * float(stage_state["kl_scale"]),
        "dac_reg": float(args.lambda_dac_reg) * float(stage_state["reg_aux_scale"]),
        "pa_reg": float(args.lambda_pa_reg) * float(stage_state["reg_aux_scale"]),
        "group_ce": float(getattr(args, "lambda_group_ce", 0.0)) * float(stage_state["group_ce_scale"]),
    }


def format_stage_state(stage_state: Dict[str, float]) -> str:
    return (
        f"phase={stage_state['phase']} | use_aux={stage_state['use_aux_views']:.1f} "
        f"cons={stage_state['cons_scale']:.2f} cls_aux={stage_state['cls_aux_scale']:.2f} "
        f"reg={stage_state['reg_aux_scale']:.2f} joint_inv={stage_state['joint_inv_scale']:.2f} "
        f"kl={stage_state['kl_scale']:.2f} group_ce={stage_state['group_ce_scale']:.2f}"
    )


def build_aug_base_cfg(args) -> Dict[str, Any]:
    return {
        "p_dac": float(args.aug_p_dac),
        "p_pa": float(args.aug_p_pa),
        "enable_class_signature": bool(args.aug_enable_class_signature),
        "class_sig_mix": float(args.aug_class_sig_mix),
        "seed": int(args.seed),
        "p_time_shift": float(args.aug_p_time_shift),
        "max_time_shift": int(args.aug_max_time_shift),
        "p_amp_scale": float(args.aug_p_amp_scale),
        "amp_min": float(args.aug_amp_min),
        "amp_max": float(args.aug_amp_max),
        "p_phase_rot": float(args.aug_p_phase_rot),
        "p_cfo": float(args.aug_p_cfo),
        "cfo_max": float(args.aug_cfo_max),
        "p_phase_noise": float(args.aug_p_phase_noise),
        "phase_noise_sigma_max": float(args.aug_phase_noise_sigma_max),
        "p_awgn": float(args.aug_p_awgn),
        "snr_min_db": float(args.aug_snr_min_db),
        "snr_max_db": float(args.aug_snr_max_db),
        "p_multipath": float(args.aug_p_multipath),
        "mp_taps_min": int(args.aug_mp_taps_min),
        "mp_taps_max": int(args.aug_mp_taps_max),
        "mp_delay_max": int(args.aug_mp_delay_max),
        "p_dc_offset": float(args.aug_p_dc_offset),
        "dc_offset_max": float(args.aug_dc_offset_max),
        "p_bandedge_taper": float(args.aug_p_bandedge_taper),
        "taper_alpha_min": float(args.aug_taper_alpha_min),
        "taper_alpha_max": float(args.aug_taper_alpha_max),
        "defect_apply_channel": False,
        "dac_only_apply_anti_shortcut": bool(args.aug_dac_only_apply_anti_shortcut),
        "dac_only_apply_channel": bool(args.aug_dac_only_apply_channel),
        "pa_only_apply_anti_shortcut": bool(args.aug_pa_only_apply_anti_shortcut),
        "pa_only_apply_channel": bool(args.aug_pa_only_apply_channel),
        "dac_pa_apply_anti_shortcut": bool(args.aug_dac_pa_apply_anti_shortcut),
        "dac_pa_apply_channel": bool(args.aug_dac_pa_apply_channel),
    }


def make_augmentor(base_cfg: Dict[str, Any]):
    return build_augmentor(**deepcopy(base_cfg))


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device, drop_last: bool, prefetch_factor: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
        "drop_last": drop_last,
        "persistent_workers": (int(num_workers) > 0),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(dataset, **kwargs)


def make_torch_generator(device, seed: int):
    try:
        gen = torch.Generator(device=device)
    except Exception:
        gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def make_sat_config(scenario: str, args):
    if SatSimConfig is None:
        raise ImportError("sat_channel.py is required for satellite channel evaluation/training.")
    kwargs = sat_channel_config_for_scenario(scenario)
    kwargs["fs_hz"] = float(getattr(args, "sat_fs_hz", 25e6))
    kwargs["fc_hz"] = float(getattr(args, "sat_fc_hz", 2.462e9))
    return SatSimConfig(**kwargs)


def apply_sat_channel_for_scenario(
    x: torch.Tensor,
    scenario: str,
    args,
    *,
    gen=None,
    return_meta: bool = False,
):
    if apply_sat_gnd_channel_batch is None:
        raise ImportError("sat_channel.py is required for satellite channel evaluation/training.")
    cfg = make_sat_config(scenario, args)
    y, meta, _ = apply_sat_gnd_channel_batch(safe_iq_tensor(x), cfg, gen=gen, return_meta=return_meta)
    return y.to(device=x.device, dtype=x.dtype), meta


def resolve_sat_eval_loader_names(named_loaders: Dict[str, DataLoader], spec: str) -> List[str]:
    raw = str(spec or "all").strip().lower()
    if raw in ("all", "all_named", "*"):
        return list(named_loaders.keys())
    if raw in ("main", "main_ood", "ood", "target", "targets", "target_ood"):
        wanted = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
        return [k for k in wanted if k in named_loaders]
    if raw in ("strict", "target_strict", "strict_target", "udu", "unseen_day_unseen_rx"):
        return ["test_unseen_day_unseen_rx"] if "test_unseen_day_unseen_rx" in named_loaders else []
    names = []
    for item in raw.replace(";", ",").replace("+", ",").split(","):
        name = item.strip()
        if name and name in named_loaders and name not in names:
            names.append(name)
    if not names:
        names = list(named_loaders.keys())
    return names


def count_parameters(model) -> Tuple[int, int]:
    raw_model = getattr(model, "_orig_mod", model)
    total = sum(int(p.numel()) for p in raw_model.parameters())
    trainable = sum(int(p.numel()) for p in raw_model.parameters() if p.requires_grad)
    return total, trainable


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def load_checkpoint_model_state(ckpt_path: str, device) -> Dict[str, torch.Tensor]:
    ckpt_path = str(ckpt_path).strip()
    if not ckpt_path:
        return None
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"--source_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            state = ckpt.get(key)
            if isinstance(state, dict):
                return state
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"Cannot find model state dict in checkpoint: {ckpt_path}")


def configure_sgc_trainable_params(model, args):
    raw_model = unwrap_model(model)
    adapter = getattr(raw_model, "sgc_adapter", None)
    if str(getattr(args, "stage", "source")).lower() != "sgc_adapt":
        return None
    if adapter is None:
        raise ValueError("--stage sgc_adapt requires --sgc_adapter or an SGC preset")
    for p in raw_model.parameters():
        p.requires_grad = False
    for p in adapter.parameters():
        p.requires_grad = True
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("SGC adaptation has no trainable adapter parameters")
    return trainable


def sgc_residual_loss_from_output(out: Dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
    if residual_regularization is None or not isinstance(out, dict):
        return ref.new_tensor(0.0)
    loss = residual_regularization(out.get("sgc_aux", {}))
    if torch.is_tensor(loss):
        return loss.to(device=ref.device, dtype=ref.dtype)
    return ref.new_tensor(float(loss))


def configure_augmentor_for_epoch(augmentor, base_cfg: Dict[str, Any], epoch: int, args):
    scale = ramp_value(
        epoch=epoch,
        epochs=args.epochs,
        warmup_epochs=int(args.aug_warmup_epochs),
        ramp_epochs=int(args.aug_ramp_epochs),
        min_scale=float(args.aug_scale_min),
        max_scale=float(args.aug_scale_max),
        curve=float(args.aug_ramp_curve),
    )
    late_aug_min = float(getattr(args, "late_aug_min_scale", -1.0))
    late_start = int(getattr(args, "late_stable_start", 0))
    if late_aug_min >= 0.0 and late_start > 0 and epoch >= late_start:
        t_late = ramp_value(epoch, args.epochs, late_start, int(getattr(args, "late_stable_ramp_epochs", 12)), 0.0, 1.0, 1.25)
        scale = scale * (1.0 - t_late) + min(scale, late_aug_min) * t_late

    prob_keys = [
        "p_dac", "p_pa", "p_time_shift", "p_amp_scale", "p_phase_rot", "p_cfo",
        "p_phase_noise", "p_awgn", "p_multipath", "p_dc_offset", "p_bandedge_taper",
    ]
    for k in prob_keys:
        setattr(augmentor, k, min(1.0, max(0.0, base_cfg[k] * scale)))

    augmentor.max_time_shift = max(0, int(round(base_cfg["max_time_shift"] * scale)))
    augmentor.cfo_max = float(base_cfg["cfo_max"] * scale)
    augmentor.phase_noise_sigma_max = float(base_cfg["phase_noise_sigma_max"] * scale)
    augmentor.dc_offset_max = float(base_cfg["dc_offset_max"] * scale)
    augmentor.taper_alpha_min = float(base_cfg["taper_alpha_min"] * scale)
    augmentor.taper_alpha_max = float(base_cfg["taper_alpha_max"] * scale)
    augmentor.mp_delay_max = max(0, int(round(base_cfg["mp_delay_max"] * scale)))

    if hasattr(augmentor, "dac"):
        augmentor.dac.jitter_max = float(args.aug_dac_jitter_max * scale)
        augmentor.dac.poly_a3 = float(args.aug_dac_poly_a3 * scale)
        augmentor.dac.poly_a5 = float(args.aug_dac_poly_a5 * scale)
        augmentor.dac.iq_img_max = float(args.aug_dac_iq_img_max * scale)
        augmentor.dac.inter_gain_max = float(args.aug_dac_inter_gain_max * scale)
        augmentor.dac.inter_off_max = float(args.aug_dac_inter_off_max * scale)
        augmentor.dac.inter_skew_max = float(args.aug_dac_inter_skew_max * scale)
        augmentor.dac.dither = float(args.aug_dac_dither * scale)
        augmentor.dac.inl_warp = float(args.aug_dac_inl_warp * scale)
        augmentor.dac.spur_amp_max = float(args.aug_dac_spur_amp_max * scale)
        augmentor.dac.slew_max = float(args.aug_dac_slew_max * scale)

    if hasattr(augmentor, "pa"):
        augmentor.pa.mp_sigma = float(args.aug_pa_mp_sigma * scale)
        augmentor.pa.mem_sigma = float(args.aug_pa_mem_sigma * scale)
        augmentor.pa.ampm_max = float(args.aug_pa_ampm_max * scale)
        augmentor.pa.iq_img_max = float(args.aug_pa_iq_img_max * scale)

    # Defect-only view purity controls are not ramped.
    augmentor.dac_only_apply_anti_shortcut = bool(base_cfg["dac_only_apply_anti_shortcut"])
    augmentor.dac_only_apply_channel = bool(base_cfg["dac_only_apply_channel"])
    augmentor.pa_only_apply_anti_shortcut = bool(base_cfg["pa_only_apply_anti_shortcut"])
    augmentor.pa_only_apply_channel = bool(base_cfg["pa_only_apply_channel"])
    augmentor.dac_pa_apply_anti_shortcut = bool(base_cfg["dac_pa_apply_anti_shortcut"])
    augmentor.dac_pa_apply_channel = bool(base_cfg["dac_pa_apply_channel"])

    return {
        "scale": scale,
        "p_dac": augmentor.p_dac,
        "p_pa": augmentor.p_pa,
        "p_time_shift": augmentor.p_time_shift,
        "p_cfo": augmentor.p_cfo,
        "p_awgn": augmentor.p_awgn,
        "p_multipath": augmentor.p_multipath,
        "max_time_shift": augmentor.max_time_shift,
        "cfo_max": augmentor.cfo_max,
        "phase_noise_sigma_max": augmentor.phase_noise_sigma_max,
    }


def configure_mixstyle_for_epoch(model, args, epoch: int) -> Dict[str, Any]:
    """Anneal MixStyle late in training without coupling to torch internals."""
    raw_model = getattr(model, "_orig_mod", model)
    id_backbone = getattr(raw_model, "id_backbone", raw_model)
    mix = getattr(id_backbone, "mixstyle", None)
    if mix is None:
        return {"enabled": False, "p": 0.0, "strength": 0.0, "phase": "missing", "anneal_t": 0.0}
    if not bool(getattr(args, "use_mixstyle", False)):
        setattr(id_backbone, "mixstyle_on", False)
        return {"enabled": False, "p": 0.0, "strength": 0.0, "phase": "disabled", "anneal_t": 0.0}

    late_start = int(getattr(args, "mixstyle_late_start", 0))
    if late_start <= 0:
        late_start = int(getattr(args, "late_stable_start", 0))
    ramp_epochs = int(getattr(args, "mixstyle_late_ramp_epochs", 0))
    if ramp_epochs <= 0:
        ramp_epochs = int(getattr(args, "late_stable_ramp_epochs", 1))

    state = compute_mixstyle_epoch_state(
        epoch=int(epoch),
        base_p=float(getattr(args, "mixstyle_p", getattr(mix, "p", 0.0))),
        base_strength=float(getattr(args, "mixstyle_strength", getattr(mix, "strength", 0.0))),
        late_start=late_start,
        ramp_epochs=ramp_epochs,
        min_p=float(getattr(args, "mixstyle_late_min_p", -1.0)),
        min_strength=float(getattr(args, "mixstyle_late_min_strength", -1.0)),
        stop_epoch=int(getattr(args, "mixstyle_stop_epoch", 0)),
    )
    setattr(mix, "p", float(state["p"]))
    setattr(mix, "strength", float(state["strength"]))
    setattr(id_backbone, "mixstyle_on", bool(state["enabled"]))
    return state


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


def smooth_strength_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = torch.nan_to_num(pred.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    target_f = torch.nan_to_num(target.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return F.smooth_l1_loss(pred_f, target_f)


def get_nested_tensor(out: Dict[str, Any], top_key: str, nested_group: str, nested_key: str) -> torch.Tensor:
    v = out.get(top_key, None)
    if torch.is_tensor(v):
        return v
    aux = out.get(nested_group, {})
    v = aux.get(nested_key, None) if isinstance(aux, dict) else None
    if not torch.is_tensor(v):
        raise KeyError(f"Cannot find tensor {top_key} / {nested_group}.{nested_key}")
    return v


def select_generalization_feature(out: Dict[str, Any], feature_name: str) -> torch.Tensor:
    name = str(feature_name or "z_id").lower().strip()
    if name == "z_id":
        return out["z_id"]
    if name in ("id_feat_joint", "feat_joint", "joint"):
        return get_nested_tensor(out, "id_feat_joint", "aux_id", "feat_joint")
    if name in ("id_feat_pa", "feat_pa", "pa"):
        return get_nested_tensor(out, "id_feat_pa", "aux_id", "feat_pa")
    if name in ("id_feat_dac", "feat_dac", "dac"):
        return get_nested_tensor(out, "id_feat_dac", "aux_id", "feat_dac")
    raise ValueError(f"Unknown generalization feature: {feature_name}")


@torch.no_grad()
def forward_anchor_eval(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    grl_lambda: float = 1.0,
    domain_labels: Optional[torch.Tensor] = None,
):
    was_training = model.training
    model.eval()
    out = model(x, y_tx=y, grl_lambda=float(grl_lambda), return_aux=True, domain_labels=domain_labels)
    if was_training:
        model.train()
    return out


def forward_main(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    grl_lambda: float,
    domain_labels: Optional[torch.Tensor] = None,
):
    return model(x, y_tx=y, grl_lambda=float(grl_lambda), return_aux=True, domain_labels=domain_labels)


def forward_aux(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    grl_lambda: float,
    enabled: bool,
    domain_labels: Optional[torch.Tensor] = None,
):
    if not enabled:
        return None
    return model(x, y_tx=y, grl_lambda=float(grl_lambda), return_aux=True, domain_labels=domain_labels)


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


def training_stage_controller(epoch: int, args, domain_stats: Dict[str, Any], num_domains: int) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
    stage_state = build_stage_state(epoch, args)
    cur_w = current_weight_dict(args, stage_state)
    gates = domain_loss_gates(domain_stats, args, num_domains)
    if not gates["dom"]:
        cur_w["dom"] = 0.0
        cur_w["adv"] = 0.0
    if not gates["cons"]:
        cur_w["cons"] = 0.0
    if not gates.get("group_ce", False):
        cur_w["group_ce"] = 0.0
    return stage_state, cur_w, gates


def grad_norm_for_params(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            return float("inf")
        total += float(g.float().norm(2).item()) ** 2
    return math.sqrt(total)


def module_params(module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def safe_backward_step(model, optimizer, scaler, loss: torch.Tensor, args, use_amp: bool) -> Tuple[bool, Dict[str, float]]:
    if (not torch.is_tensor(loss)) or (not torch.isfinite(loss.detach()).all()) or (not loss.requires_grad):
        optimizer.zero_grad(set_to_none=True)
        return False, {"grad_total": float("nan"), "grad_backbone": float("nan"), "grad_aux": float("nan"), "grad_domain": float("nan")}

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    raw_model = getattr(model, "_orig_mod", model)
    id_params = module_params(getattr(raw_model, "id_backbone", raw_model))
    dom_backbone = getattr(raw_model, "dom_backbone", None)
    dom_backbone_params = module_params(dom_backbone) if dom_backbone is not None else []
    domain_head_params = []
    for name in ("dom_head", "adv_head"):
        head = getattr(raw_model, name, None)
        if head is not None:
            domain_head_params.extend(module_params(head))

    max_backbone = float(getattr(args, "clip_grad_backbone", 1.0))
    max_aux = float(getattr(args, "clip_grad_aux", 0.75))
    max_domain = float(getattr(args, "clip_grad_domain", 0.5))
    if id_params:
        torch.nn.utils.clip_grad_norm_(id_params, max_backbone, error_if_nonfinite=False)
    if dom_backbone_params:
        torch.nn.utils.clip_grad_norm_(dom_backbone_params, max_aux, error_if_nonfinite=False)
    if domain_head_params:
        torch.nn.utils.clip_grad_norm_(domain_head_params, max_domain, error_if_nonfinite=False)

    all_params = module_params(raw_model)
    grad_total = grad_norm_for_params(all_params)
    stats = {
        "grad_total": grad_total,
        "grad_backbone": grad_norm_for_params(id_params),
        "grad_aux": grad_norm_for_params(dom_backbone_params),
        "grad_domain": grad_norm_for_params(domain_head_params),
    }
    if not math.isfinite(grad_total):
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        return False, stats

    scaler.step(optimizer)
    scaler.update()
    return True, stats


def unwrap_wisig_dataset(dataset):
    """Return the underlying WiSig dataset-like object when DataLoader wraps Subset/Concat.

    The training split usually is WiSigSubsetDataset, which already exposes index,
    _domain_lut, rx_list and day_list. This helper makes the domain utilities robust
    to future torch.utils.data.Subset wrappers.
    """
    cur = dataset
    visited = set()
    while True:
        oid = id(cur)
        if oid in visited:
            break
        visited.add(oid)
        if hasattr(cur, "index") and hasattr(cur, "_domain_lut"):
            return cur
        if hasattr(cur, "dataset"):
            cur = cur.dataset
            continue
        break
    return dataset


def get_wisig_domain_mode(dataset, default: str = "unknown") -> str:
    obj = unwrap_wisig_dataset(dataset)
    mode = getattr(obj, "domain", None)
    if mode is None and hasattr(obj, "base"):
        mode = getattr(obj.base, "domain", None)
    return str(mode or default).lower()


def build_domain_label_map(dataset) -> Dict[int, int]:
    """Build raw-domain -> compact-domain mapping from the TRAIN split only.

    Raw labels are defined by dataset_wisig.WiSigCompactDataset._build_domain_lut:
      day    : raw domain = day_i
      rx     : raw domain = rx_i
      rx_day : raw domain = unique (rx_i, day_i) pair

    The map is intentionally built from the train split. During evaluation, samples
    from unseen raw domains are mapped to -1 and ignored for domain accuracy.
    That prevents the domain classifier from being judged on classes it was never
    trained to predict.
    """
    obj = unwrap_wisig_dataset(dataset)
    if not (hasattr(obj, "index") and hasattr(obj, "_domain_lut")):
        return {}
    raw_labels = sorted({int(obj._domain_lut[(it.rx_i, it.day_i)]) for it in obj.index})
    return {raw: idx for idx, raw in enumerate(raw_labels)}


def decode_wisig_domain_label(dataset, raw: int) -> str:
    """Human-readable label for a raw WiSig domain id."""
    obj = unwrap_wisig_dataset(dataset)
    mode = get_wisig_domain_mode(obj)
    day_list = list(getattr(obj, "day_list", getattr(getattr(obj, "base", None), "day_list", [])) or [])
    rx_list = list(getattr(obj, "rx_list", getattr(getattr(obj, "base", None), "rx_list", [])) or [])
    n_day = max(1, len(day_list))

    raw = int(raw)
    if mode == "day":
        name = day_list[raw] if 0 <= raw < len(day_list) else raw
        return f"day[{raw}]={name}"
    if mode == "rx":
        name = rx_list[raw] if 0 <= raw < len(rx_list) else raw
        return f"rx[{raw}]={name}"
    if mode == "rx_day":
        # Must match dataset_wisig: for rx_i in all_rx: for day_i in all_day: did += 1
        rx_i = raw // n_day
        day_i = raw % n_day
        rx_name = rx_list[rx_i] if 0 <= rx_i < len(rx_list) else rx_i
        day_name = day_list[day_i] if 0 <= day_i < len(day_list) else day_i
        return f"rx_day[{raw}]=rx[{rx_i}]={rx_name} × day[{day_i}]={day_name}"
    return f"domain[{raw}]"


def summarize_wisig_rx_counts(dataset) -> Optional[List[str]]:
    obj = unwrap_wisig_dataset(dataset)
    if not hasattr(obj, "index"):
        return None
    rx_list = list(getattr(obj, "rx_list", getattr(getattr(obj, "base", None), "rx_list", [])) or [])
    counts: Dict[int, int] = {}
    for it in getattr(obj, "index", []):
        rx_i = int(getattr(it, "rx_i", -1))
        counts[rx_i] = counts.get(rx_i, 0) + 1
    if not counts:
        return None
    out = []
    for rx_i in sorted(counts.keys()):
        rx_name = rx_list[rx_i] if 0 <= rx_i < len(rx_list) else rx_i
        out.append(f"rx[{rx_i}]={rx_name}:{counts[rx_i]}")
    return out


def print_dataset_sample_summary(args, train_ds, val_ds):
    print(f"[SAMPLES] train={len(train_ds)} | val={len(val_ds)}")
    if str(args.dataset).lower() != "wisig":
        return
    train_rx = summarize_wisig_rx_counts(train_ds)
    val_rx = summarize_wisig_rx_counts(val_ds)
    if train_rx:
        print(f"[SAMPLES-RX][TRAIN] {' | '.join(train_rx)}")
    if val_rx:
        print(f"[SAMPLES-RX][VAL]   {' | '.join(val_rx)}")


def domain_mode_description(mode: str) -> Dict[str, str]:
    mode = str(mode).lower()
    table = {
        "day": {
            "target": "DATE / capture-day domain",
            "cn": "日期/采集天域",
            "dom": "让 z_dom 显式分类不同采集日期，捕捉温漂、环境、时间批次、信道统计变化。",
            "adv": "通过 GRL 让 z_id 尽量去除日期/采集天信息。",
            "cons": "约束同一发射机在不同日期下的 ID 特征中心更接近。",
            "risk": "只抑制日期域；如果跨接收机下降，day 模式本身不直接解决 RX/ADC/AGC 偏置。",
        },
        "rx": {
            "target": "RECEIVER / RX domain",
            "cn": "接收机域",
            "dom": "让 z_dom 显式分类不同接收机，捕捉 LNA/Mixer/滤波器/AGC/ADC/采样链路差异。",
            "adv": "通过 GRL 让 z_id 尽量去除接收机伪特征。",
            "cons": "约束同一发射机在不同接收机下的 ID 特征中心更接近。",
            "risk": "只抑制 RX 域；如果跨日期下降，rx 模式本身不直接建模日期/环境漂移。",
        },
        "rx_day": {
            "target": "JOINT RECEIVER × DATE domain",
            "cn": "接收机×日期联合域",
            "dom": "让 z_dom 分类每一个 RX×day 组合域，显式捕捉接收机与日期的耦合偏置。",
            "adv": "通过 GRL 让 z_id 同时去除 RX 与 day 的联合域信息。",
            "cons": "约束同一发射机跨 RX×day 组合域的 ID 特征中心更接近。",
            "risk": "域数最多，域分类更难；batch 内域覆盖不足时 cons/adv 可能不稳定。",
        },
    }
    return table.get(mode, {
        "target": "UNKNOWN domain",
        "cn": "未知域",
        "dom": "未知域设置。",
        "adv": "未知域设置。",
        "cons": "未知域设置。",
        "risk": "请检查 --wisig_domain。",
    })


def print_domain_configuration(args, train_ds, split_info, domain_label_map: Dict[int, int]):
    """Print an explicit, audit-friendly description of what domain losses target."""
    mode = str(getattr(args, "wisig_domain", get_wisig_domain_mode(train_ds))).lower()
    desc = domain_mode_description(mode)
    n_domains = max(1, len(domain_label_map))

    print("=" * 120)
    print(f"[DOMAIN-MODE] wisig_domain={mode} | target={desc['target']} | 中文={desc['cn']}")
    print(f"[DOMAIN-MODE] num_train_domains={n_domains}")
    print(f"[DOMAIN-LOSS] loss_dom  : {desc['dom']}")
    print(f"[DOMAIN-LOSS] loss_adv  : {desc['adv']}")
    print(f"[DOMAIN-LOSS] loss_cons : {desc['cons']}")
    print(f"[DOMAIN-RISK] {desc['risk']}")

    if split_info is not None:
        print(
            f"[DOMAIN-SPLIT] train_days={split_info.get('train_days_label', [])} | "
            f"train_rxs_idx={split_info.get('train_rxs_idx', [])} | "
            f"test_days={split_info.get('test_days_label', [])} | "
            f"test_rxs_idx={split_info.get('test_rxs_idx', [])}"
        )

    if len(domain_label_map) > 0:
        print("[DOMAIN-LABELS] raw_domain -> mapped_domain -> readable_label")
        for raw, mapped in domain_label_map.items():
            print(f"  raw={int(raw):>4} -> mapped={int(mapped):>3} -> {decode_wisig_domain_label(train_ds, int(raw))}")
    else:
        print("[DOMAIN-LABELS] no explicit WiSig domain labels found; fallback num_domains=1")

    # Sanity warnings: these prevent false claims in ablation analysis.
    if mode == "day" and split_info is not None and len(split_info.get("train_days_idx", [])) < 2:
        print("[DOMAIN-WARN] wisig_domain=day 但训练日期少于 2 个；dom/adv/cons 几乎不能证明跨日期去域。")
    if mode == "rx" and split_info is not None and len(split_info.get("train_rxs_idx", [])) < 2:
        print("[DOMAIN-WARN] wisig_domain=rx 但训练接收机少于 2 个；dom/adv/cons 几乎不能证明跨接收机去域。")
    if mode == "rx_day" and split_info is not None:
        nd = len(split_info.get("train_days_idx", []))
        nr = len(split_info.get("train_rxs_idx", []))
        if nd < 2 or nr < 2:
            print("[DOMAIN-WARN] wisig_domain=rx_day 但 train day 或 train rx 数不足；联合域去偏证据会很弱。")
    print("=" * 120)


def remap_domain_tensor(d: Optional[torch.Tensor], domain_label_map: Dict[int, int], device) -> Optional[torch.Tensor]:
    """Map raw WiSig domain labels to compact train-domain labels.

    Returns -1 for raw labels not present in training domains. Those samples are
    ignored by domain accuracy in evaluation and should not enter domain CE.
    """
    if d is None:
        return None
    out = torch.full_like(d.view(-1).long(), fill_value=-1, device=device)
    for raw, mapped in domain_label_map.items():
        out[d.view(-1).long() == int(raw)] = int(mapped)
    return out


@torch.no_grad()
def evaluate_loader(model, loader, device, domain_label_map: Dict[int, int], max_batches: int = 0):
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d_raw = extract_domain_from_extra(extra, device)
        d = remap_domain_tensor(d_raw, domain_label_map, device) if d_raw is not None else None

        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        tx_logits = out["tx_logits"]
        tx_pred = tx_logits.argmax(dim=1)
        tx_correct += int((tx_pred == y).sum().item())
        tx_total += int(y.numel())

        if d is not None:
            valid = d >= 0
            if valid.any():
                dom_y = d[valid]
                dom_correct += int((out["dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                dom_total += int(dom_y.numel())

        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    return {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": float("nan"),
        "tx_correct": int(tx_correct),
        "tx_total": int(tx_total),
    }


@torch.no_grad()
def evaluate_named_loaders(model, named_loaders: Dict[str, DataLoader], device, domain_label_map: Dict[int, int], max_batches: int = 0):
    out = {}
    for name, loader in named_loaders.items():
        out[name] = evaluate_loader(model, loader, device, domain_label_map=domain_label_map, max_batches=max_batches)
    return out


@torch.no_grad()
def evaluate_loader_sat_channel(
    model,
    loader,
    device,
    domain_label_map: Dict[int, int],
    scenario: str,
    args,
    max_batches: int = 0,
    seed: int = 0,
):
    model.eval()
    tx_correct = tx_total = 0
    dom_correct = dom_total = 0
    gen = make_torch_generator(device, int(seed))
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_sat, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
        d_raw = extract_domain_from_extra(extra, device)
        d = remap_domain_tensor(d_raw, domain_label_map, device) if d_raw is not None else None

        out = model(x_sat, y_tx=None, grl_lambda=1.0, return_aux=True)
        tx_logits = out["tx_logits"]
        tx_pred = tx_logits.argmax(dim=1)
        tx_correct += int((tx_pred == y).sum().item())
        tx_total += int(y.numel())

        if d is not None:
            valid = d >= 0
            if valid.any():
                dom_y = d[valid]
                dom_correct += int((out["dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                dom_total += int(dom_y.numel())

        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    return {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": float("nan"),
        "tx_correct": int(tx_correct),
        "tx_total": int(tx_total),
    }


@torch.no_grad()
def evaluate_sat_scenarios(
    model,
    named_loaders: Dict[str, DataLoader],
    device,
    domain_label_map: Dict[int, int],
    scenario_names: List[str],
    args,
    max_batches: int = 0,
):
    selected_names = resolve_sat_eval_loader_names(named_loaders, getattr(args, "eval_sat_on", "all"))
    out = {}
    for si, scenario in enumerate(scenario_names):
        named_stats = {}
        for li, name in enumerate(selected_names):
            named_stats[name] = evaluate_loader_sat_channel(
                model,
                named_loaders[name],
                device,
                domain_label_map=domain_label_map,
                scenario=scenario,
                args=args,
                max_batches=max_batches,
                seed=int(getattr(args, "sat_seed", 2027)) + si * 1009 + li * 97,
            )
        main_keys = [k for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_stats]
        if not main_keys:
            main_keys = list(named_stats.keys())
        aggregate = aggregate_named_stats(named_stats, main_keys)
        all_named_aggregate = aggregate_named_stats(named_stats, list(named_stats.keys()))
        strict = named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan"))
        out[scenario] = {
            "aggregate": aggregate,
            "all_named_aggregate": all_named_aggregate,
            "strict_udu": strict,
            "named": named_stats,
            "selected_names": list(selected_names),
        }
    return out


def format_sat_test_lines(
    sat_stats: Dict[str, Dict[str, Any]],
    named_test_meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    lines = []
    named_test_meta = named_test_meta or {}
    for scenario, stats in sat_stats.items():
        agg = stats.get("aggregate", {})
        all_agg = stats.get("all_named_aggregate", {})
        strict = stats.get("strict_udu", float("nan"))
        selected = ",".join(stats.get("selected_names", []))
        lines.append(
            f"[SAT-TEST] scenario={scenario} selected={selected} "
            f"overall_tx={agg.get('tx_acc', float('nan')):.2f}% "
            f"all_named_tx={all_agg.get('tx_acc', float('nan')):.2f}% "
            f"strict_udu={safe_nan(strict)}% "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})"
        )
        named = stats.get("named", {})
        if named:
            ordered_names = list(named.keys())
            priority = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
            ordered_names = [k for k in priority if k in named] + [k for k in ordered_names if k not in priority]
            for name in ordered_names:
                cur = named[name]
                label = make_test_subset_label(name, named_test_meta.get(name, {}))
                lines.append(
                    f"[SAT-TEST-SPLIT] scenario={scenario} {label}: "
                    f"tx={cur.get('tx_acc', float('nan')):.2f}% "
                    f"({int(cur.get('tx_correct', 0))}/{int(cur.get('tx_total', 0))})"
                )
    return lines


def aggregate_named_stats(named_stats: Dict[str, Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    total_correct = 0
    total_count = 0
    for k in keys:
        if k not in named_stats:
            continue
        total_correct += int(named_stats[k].get("tx_correct", 0))
        total_count += int(named_stats[k].get("tx_total", 0))
    return {
        "tx_acc": 100.0 * total_correct / max(1, total_count),
        "tx_correct": int(total_correct),
        "tx_total": int(total_count),
    }


def make_test_subset_label(name: str, meta: Dict[str, Any]) -> str:
    if name == "test_unseen_day_seen_rx":
        return f"unseen_day_seen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name == "test_seen_day_unseen_rx":
        return f"seen_day_unseen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name == "test_unseen_day_unseen_rx":
        return f"unseen_day_unseen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name.startswith("test_day_"):
        return f"day={meta.get('days_label', ['?'])[0]} on seen_rxs={meta.get('rxs_idx', [])}"
    if name.startswith("test_rx_"):
        return f"rx={meta.get('rxs_idx', ['?'])[0]} on seen_days={meta.get('days_label', [])}"
    return name


def format_named_test_lines(named_test_stats: Dict[str, Dict[str, float]], named_test_meta: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = []
    ordered_names = list(named_test_stats.keys())
    priority = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    ordered_names = [k for k in priority if k in named_test_stats] + [k for k in ordered_names if k not in priority]
    for name in ordered_names:
        stats = named_test_stats[name]
        meta = named_test_meta.get(name, {})
        label = make_test_subset_label(name, meta)
        lines.append(f"          {label}: tx={stats['tx_acc']:.2f}% ({stats['tx_correct']}/{stats['tx_total']})")
    return lines


def format_epoch_block(
    epoch: int,
    epochs: int,
    lr: float,
    epoch_time_s: float,
    meters: Dict[str, AverageMeter],
    m_domacc: NanMeter,
    cons_cos_epoch: float,
    val_stats: Dict[str, float],
    test_stats: Dict[str, float],
    named_test_stats: Dict[str, Dict[str, float]],
    named_test_meta: Dict[str, Dict[str, Any]],
    best_joint_val_tx: float,
    best_joint_test_tx: float,
    best_epoch: int,
    latest_path: str,
    best_path: str,
    is_best: bool,
    aug_state: Optional[Dict[str, Any]],
    aux_scale: float,
    stage_state: Optional[Dict[str, float]] = None,
    mixstyle_state: Optional[Dict[str, Any]] = None,
    collapse_guard: Optional[Dict[str, Any]] = None,
    latest_saved: bool = True,
):
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(f"[Epoch {epoch:03d}/{epochs:03d}] time={epoch_time_s:.1f}s | lr={lr:.2e} | aux_scale={aux_scale:.3f}")
    if stage_state is not None:
        lines.append(f"[STAGE] {format_stage_state(stage_state)}")
    if mixstyle_state is not None:
        lines.append(
            "[MIXSTYLE-EPOCH] "
            f"phase={mixstyle_state.get('phase', 'unknown')} enabled={int(bool(mixstyle_state.get('enabled', False)))} "
            f"p={float(mixstyle_state.get('p', 0.0)):.3f} "
            f"strength={float(mixstyle_state.get('strength', 0.0)):.3f} "
            f"anneal_t={float(mixstyle_state.get('anneal_t', 0.0)):.3f}"
        )
    if aug_state is not None:
        lines.append(
            "[AUG] "
            f"scale={aug_state['scale']:.3f} | p_dac={aug_state['p_dac']:.3f} p_pa={aug_state['p_pa']:.3f} "
            f"p_shift={aug_state['p_time_shift']:.3f} p_cfo={aug_state['p_cfo']:.3f} "
            f"p_awgn={aug_state['p_awgn']:.3f} p_mp={aug_state['p_multipath']:.3f} | "
            f"max_shift={aug_state['max_time_shift']} cfo_max={aug_state['cfo_max']:.4g} "
            f"pn_max={aug_state['phase_noise_sigma_max']:.4g}"
        )
    else:
        lines.append("[AUG] disabled")
    lines.append(minor)
    lines.append(
        "[LOSS-CORE] "
        f"total={meters['loss'].avg:.4f} cls={meters['cls'].avg:.4f} dom={meters['dom'].avg:.4f} "
        f"adv={meters['adv'].avg:.4f} orth={meters['orth'].avg:.4f} cons={meters['cons'].avg:.4f} "
        f"group_ce={meters['group_ce'].avg:.4f}"
    )
    lines.append(
        "[LOSS-AUX]  "
        f"cls_pa={meters['cls_pa'].avg:.4f} cls_dac={meters['cls_dac'].avg:.4f} "
        f"pa_joint_inv={meters['pa_joint_inv'].avg:.4f} pa_kl={meters['pa_kl'].avg:.4f} "
        f"dac_reg={meters['dac_reg'].avg:.4f} pa_reg={meters['pa_reg'].avg:.4f} "
        f"gap_dac={meters['gap_dac'].avg:.4f} gap_pa={meters['gap_pa'].avg:.4f} "
        f"cos_joint_pa={meters['cos_joint_pa'].avg:.4f} cos_imp_pa={meters['cos_imp_pa'].avg:.4f}"
    )
    lines.append(
        "[LOSS-SAT]  "
        f"cls_sat={meters['sat_cls'].avg:.4f} sat_cons={meters['sat_cons'].avg:.4f} "
        f"sat_cos={meters['sat_cos'].avg:.4f}"
    )
    lines.append(
        "[LOSS-DG]   "
        f"proto={meters['proto'].avg:.4f} proto_cos={meters['proto_pull_cos'].avg:.4f} "
        f"supcon={meters['supcon'].avg:.4f} fishr={meters['fishr'].avg:.4f} "
        f"sgc_res={meters['sgc_res'].avg:.4f}"
    )
    lines.append(
        "[LOSS-CAL]  "
        f"ecc={meters['ecc'].avg:.4f} ecc_w={meters['ecc_w'].avg:.4f} "
        f"ecc_tau={meters['ecc_tau'].avg:.4f} ecc_maxp={meters['ecc_maxp'].avg:.4f}"
    )
    lines.append(
        "[TRAIN] "
        f"tx={meters['txacc'].avg:.2f}% dom={safe_nan(m_domacc.avg)}% cons_cos={cons_cos_epoch:.4f}"
    )
    lines.append(
        "[GRAD]  "
        f"total={meters['grad_total'].avg:.3f} backbone={meters['grad_backbone'].avg:.3f} "
        f"aux={meters['grad_aux'].avg:.3f} domain={meters['grad_domain'].avg:.3f}"
    )
    lines.append(
        "[VAL]   "
        f"tx={val_stats['tx_acc']:.2f}% dom={safe_nan(val_stats['dom_acc'])}%"
    )
    lines.append(f"[TEST]  overall_tx={test_stats['tx_acc']:.2f}% ({test_stats['tx_correct']}/{test_stats['tx_total']})")
    lines.append("[TEST-SPLIT]")
    lines.extend(format_named_test_lines(named_test_stats, named_test_meta))
    lines.append(f"[BEST-JOINT]  val_tx={best_joint_val_tx:.2f}% & test_tx={best_joint_test_tx:.2f}% @ E{best_epoch:03d}")
    latest_note = "saved" if latest_saved else f"protected: {collapse_guard.get('reason', 'unknown') if collapse_guard else 'unknown'}"
    lines.append(f"[CKPT]  latest -> {latest_path} ({latest_note}) | best -> {best_path}{' (updated: val improved)' if is_best else ''}")
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wisig", choices=["wisig", "oralce"])
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_equalized", type=str, default="1")
    parser.add_argument("--wisig_domain", type=str, default="rx_day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_train_ratio", type=float, default=0.2,
                        help="WiSig train/val contiguous split ratio inside train_days x train_rxs. Must be in (0, 1).")
    parser.add_argument("--wisig_val_ratio", type=float, default=-1.0,
                        help="Optional convenience override. If >0, train_ratio is set to 1 - wisig_val_ratio.")
    parser.add_argument("--wisig_guard_gap", type=int, default=8)
    parser.add_argument("--wisig_train_days", type=str, default="0,1")
    parser.add_argument("--wisig_test_days", type=str, default="2,3")
    parser.add_argument("--wisig_train_rxs", type=str, default="0,1,2,3,4,5,6")
    parser.add_argument("--wisig_test_rxs", type=str, default="7,8,9,10,11")
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)

    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--model_variant", type=str, default="lite_c", choices=["base", "lite_a", "lite_b", "lite_c", "lite_d", "lite_e"],
                        help="Lightweight model variant. lite_c is the streamlined default.")
    parser.add_argument(
        "--slim_group",
        type=str,
        default="none",
        choices=[
            "none", "balanced", "balanced_stable", "no_dac", "no_stats",
            "lite_b", "lite_b_no_dac", "lite_b_no_dac_stable",
            "lite_d_no_dac", "lite_d_no_dac_stable", "lite_e_time_only",
            "rxrobust_balanced", "rxrobust_no_stats",
            "rxrobust_lite_b_no_dac", "rxrobust_lite_d_no_dac", "rxrobust_no_dac_no_stats",
            "rxrobust_lite_b_no_dac_refined", "rxrobust_lite_b_no_dac_mix015",
            "rxrobust_lite_b_no_dac_domain020", "rxrobust_lite_d_no_dac_refined",
            "rxrobust_lite_b_no_dac_gce006", "rxrobust_lite_b_no_dac_gce014",
            "sgc_lite_b_no_dac", "sgc_lite_b_no_dac_no_amp",
            "sgc_lite_b_no_dac_no_freq", "sgc_lite_b_no_dac_no_spec",
            "sgc_lite_b_no_dac_no_res", "sgc_baseline_no_adapter",
            "sgc_lite_b_no_dac_no_amp_freq", "sgc_lite_b_no_dac_residual_only",
            "sgc_lite_b_no_dac_light", "sgc_lite_d_no_dac", "sgc_lite_d_no_dac_light",
        ],
        help="瘦身/时延消融预设组，会联合覆盖 model_variant、branch_ablation、exp_group 和 MixStyle。",
    )
    parser.add_argument("--preset", type=str, default="",
                        help="Alias for --slim_group, kept for SGC launch scripts.")
    parser.add_argument(
        "--branch_ablation",
        type=str,
        default="none",
        help=(
            "Comma-separated model branch ablations. "
            "Valid: none,no_time,no_dac,no_pa,no_freq,no_stats. "
            "Aliases: time_only,freq_only,no_dac_pa,no_physical,no_defect_branches."
        ),
    )
    parser.add_argument(
        "--domain_branch_ablation",
        type=str,
        default="same",
        help="Branch ablation used by the second/domain backbone. Use 'same' to mirror --branch_ablation.",
    )
    parser.add_argument("--domain_enhancer", type=str, default="rcn_stats", choices=["off", "rcn_stats"],
                        help="Second-backbone RCN enhancement module for receiver/channel/noise domain cues.")
    parser.add_argument("--domain_enhancer_strength", type=float, default=0.35)
    parser.add_argument("--force_domain_branch_ablation", type=str, default="",
                        help="Late override for domain_branch_ablation after slim/experiment presets.")
    parser.add_argument("--force_domain_enhancer", type=str, default="", choices=["", "off", "rcn_stats"],
                        help="Late override for domain_enhancer after slim/experiment presets.")
    parser.add_argument("--force_domain_enhancer_strength", type=float, default=-1.0,
                        help="Late override for domain_enhancer_strength when >=0.")
    add_bool_arg(parser, "fast_infer_when_no_aux", True,
                 "Skip the second/domain backbone when model(..., return_aux=False)",
                 "Always run both backbones even when return_aux=False")
    parser.add_argument("--stage", type=str, default="source",
                        choices=["source", "sgc_augment", "sgc_adapt"],
                        help="SGC training stage.")
    add_bool_arg(parser, "sgc_adapter", False,
                 "Enable SGC-Adapter before the dual backbones",
                 "Disable SGC-Adapter")
    parser.add_argument("--sgc_adapter_kwargs", type=str, default="{}",
                        help="JSON object passed to SGCAdapter.")
    parser.add_argument("--source_ckpt", type=str, default="",
                        help="Checkpoint used to initialize sgc_augment or sgc_adapt stages.")
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.85,
                        help="Reserved threshold for target-domain pseudo labels.")
    parser.add_argument("--lambda_feat", type=float, default=1.0,
                        help="Convenience alias for satellite feature consistency in sgc_augment.")
    parser.add_argument("--lambda_ent", type=float, default=0.01,
                        help="Reserved entropy weight for target-domain adaptation.")
    parser.add_argument("--lambda_res", type=float, default=0.01,
                        help="Residual regularization weight for SGC-Adapter auxiliary output.")
    parser.add_argument("--adapt_lr", type=float, default=1e-4,
                        help="Optimizer learning rate for --stage sgc_adapt.")
    parser.add_argument("--adapt_epochs", type=int, default=50,
                        help="Epoch count override for --stage sgc_adapt.")
    add_bool_arg(parser, "use_mixstyle", False, "Enable MixStyle1D on the ID backbone time branch", "Disable MixStyle1D")
    parser.add_argument("--mixstyle_p", type=float, default=0.3)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)
    parser.add_argument("--mixstyle_eps", type=float, default=1e-6)
    parser.add_argument("--mixstyle_layers", type=str, default="time_down,t1")
    add_bool_arg(parser, "mixstyle_use_domain_label", True, "Use domain labels for cross-domain MixStyle pairing", "Do not use domain labels for MixStyle pairing")
    parser.add_argument("--mixstyle_mix", type=str, default="crossdomain",
                        choices=["crossdomain", "random", "same_tx", "same_tx_crossdomain"])
    parser.add_argument("--mixstyle_strength", type=float, default=1.0)
    parser.add_argument("--mixstyle_fallback", type=str, default="random", choices=["random", "skip"])
    parser.add_argument("--mixstyle_late_start", type=int, default=0,
                        help="Epoch to start annealing MixStyle p/strength. 0 reuses late_stable_start.")
    parser.add_argument("--mixstyle_late_ramp_epochs", type=int, default=0,
                        help="MixStyle annealing ramp length. 0 reuses late_stable_ramp_epochs.")
    parser.add_argument("--mixstyle_late_min_p", type=float, default=-1.0,
                        help="Late MixStyle probability target. <0 disables probability annealing.")
    parser.add_argument("--mixstyle_late_min_strength", type=float, default=-1.0,
                        help="Late MixStyle strength target. <0 disables strength annealing.")
    parser.add_argument("--mixstyle_stop_epoch", type=int, default=0,
                        help="If >0, disables MixStyle after this epoch.")

    parser.add_argument("--lambda_dom", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.5)
    parser.add_argument("--lambda_orth", type=float, default=0.05)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--force_lambda_adv", type=float, default=None,
                        help="Late override for lambda_adv after experiment presets.")
    parser.add_argument("--force_lambda_orth", type=float, default=None,
                        help="Late override for lambda_orth after experiment presets.")
    parser.add_argument("--force_lambda_cons", type=float, default=None,
                        help="Late override for lambda_cons after experiment presets.")
    parser.add_argument("--lambda_group_ce", type=float, default=0.0,
                        help="Hard-domain CE weight. Optimizes high-loss train rx/day groups for receiver robustness.")
    parser.add_argument("--group_ce_top_frac", type=float, default=0.35,
                        help="Fraction of hardest domains used by hard-domain CE.")
    parser.add_argument("--group_ce_min_domains", type=int, default=2,
                        help="Minimum valid domains in a batch before hard-domain CE is enabled.")
    parser.add_argument("--group_ce_mode", type=str, default="hard",
                        choices=["hard", "smooth_dro", "smooth_dro_capped", "dual_worst"],
                        help="Group loss mode: hard top-domain CE, Smooth GroupDRO, capped Smooth GroupDRO, or rx/day dual weighting.")
    parser.add_argument("--groupdro_momentum", type=float, default=0.95)
    parser.add_argument("--groupdro_tau", type=float, default=0.5)
    parser.add_argument("--groupdro_cap", type=float, default=0.65)
    parser.add_argument("--groupdro_num_days", type=int, default=4)
    parser.add_argument("--generalization_feature", type=str, default="z_id",
                        choices=["z_id", "id_feat_joint", "feat_joint", "id_feat_pa", "id_feat_dac"],
                        help="Feature used by prototype memory and domain-aware SupCon.")
    add_bool_arg(parser, "use_proto_memory", False,
                 "Enable class-conditional prototype memory bank",
                 "Disable class-conditional prototype memory bank")
    parser.add_argument("--lambda_proto", type=float, default=0.0)
    parser.add_argument("--proto_momentum", type=float, default=0.95)
    parser.add_argument("--proto_margin", type=float, default=0.15)
    parser.add_argument("--proto_domain_align_weight", type=float, default=0.5)
    parser.add_argument("--proto_push_weight", type=float, default=0.1)
    parser.add_argument("--proto_min_count", type=int, default=2)
    parser.add_argument("--lambda_supcon_id", type=float, default=0.0)
    parser.add_argument("--supcon_temp", type=float, default=0.12)
    parser.add_argument("--lambda_fishr", type=float, default=0.0)
    parser.add_argument("--fishr_min_domains", type=int, default=2)
    parser.add_argument("--lambda_probe", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")
    parser.add_argument("--grl_lambda", type=float, default=1.0)

    parser.add_argument("--aux_warmup_epochs", type=int, default=3)
    parser.add_argument("--aux_ramp_epochs", type=int, default=25)
    parser.add_argument("--robust_temp", type=float, default=1.0)
    parser.add_argument("--select_margin", type=float, default=0.03)
    parser.add_argument("--mono_margin", type=float, default=0.00)

    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=45)
    parser.add_argument("--stage3_ramp_epochs", type=int, default=20)
    parser.add_argument("--late_stable_start", type=int, default=0,
                        help="Epoch to start late-stage loss stabilization. 0 disables it.")
    parser.add_argument("--late_stable_ramp_epochs", type=int, default=12)
    parser.add_argument("--late_adv_min_scale", type=float, default=0.75)
    parser.add_argument("--late_cons_min_scale", type=float, default=0.55)
    parser.add_argument("--late_cls_aux_min_scale", type=float, default=0.35)
    parser.add_argument("--late_reg_aux_min_scale", type=float, default=0.35)
    parser.add_argument("--late_joint_inv_min_scale", type=float, default=0.12)
    parser.add_argument("--late_kl_min_scale", type=float, default=0.25)
    parser.add_argument("--late_group_ce_min_scale", type=float, default=0.75)
    parser.add_argument("--late_aug_min_scale", type=float, default=-1.0,
                        help="Optional late-stage augmentation scale floor/target. <0 disables augmentation decay.")
    add_bool_arg(parser, "collapse_guard", True,
                 "Protect latest checkpoint from random-level late collapse",
                 "Always overwrite latest checkpoint")
    parser.add_argument("--collapse_guard_min_epoch", type=int, default=40)
    parser.add_argument("--collapse_guard_random_margin", type=float, default=3.0)
    parser.add_argument("--collapse_guard_best_margin", type=float, default=25.0)
    parser.add_argument("--collapse_guard_max_skipped_delta", type=int, default=3)

    parser.add_argument("--lambda_cls_pa", type=float, default=0.60)
    parser.add_argument("--lambda_cls_dac", type=float, default=0.15)
    parser.add_argument("--lambda_pa_joint_inv", type=float, default=0.25)
    parser.add_argument("--lambda_pa_imp_inv", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")
    parser.add_argument("--lambda_pa_kl", type=float, default=0.12)
    parser.add_argument("--lambda_dac_reg", type=float, default=0.35)
    parser.add_argument("--lambda_pa_reg", type=float, default=0.35)
    parser.add_argument("--lambda_cross_zero", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")
    parser.add_argument("--lambda_dac_select", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")
    parser.add_argument("--lambda_pa_select", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")
    parser.add_argument("--lambda_dac_mono", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")
    parser.add_argument("--lambda_pa_mono", type=float, default=0.0,
                        help="Deprecated and ignored by the streamlined trainer.")

    parser.add_argument(
        "--exp_group",
        type=str,
        default="s4_stagewise_full_dual",
        choices=[
            "s1_core_only", "s2_pure_aux_no_select", "s3_stagewise_pa_focus", "s4_stagewise_full_dual",
            "s3_stable_no_dac", "s4_late_stable_full", "s3_rxrobust_no_dac", "s4_rxrobust_full",
            "g1_true_no_pa", "g2_pa_aux_only", "g3_pa_main_only", "g4_pa_main_plus_aux", "g5_full_dual_puredefect",
        ],
        help="分阶段多目标训练预设。推荐使用 s4_stagewise_full_dual。",
    )
    add_bool_arg(parser, "enable_pa_aux", True, "启用PA-only辅助分支与PA辅助损失", "关闭PA-only辅助分支与PA辅助损失")
    add_bool_arg(parser, "enable_dac_aux", True, "启用DAC-only辅助分支与DAC辅助损失", "关闭DAC-only辅助分支与DAC辅助损失")

    add_bool_arg(parser, "use_aug", True, "启用训练时数据增强", "关闭训练时数据增强")
    add_bool_arg(parser, "aug_enable_class_signature", False, "增强中启用类签名偏置", "增强中关闭类签名偏置")
    add_bool_arg(parser, "aug_enable_pa_normal", True, "正常训练视图也启用 PA 增强", "正常训练视图不启用 PA 增强")
    add_bool_arg(parser, "aug_dac_only_apply_anti_shortcut", False, "DAC-only视图叠加anti-shortcut", "DAC-only视图不叠加anti-shortcut")
    add_bool_arg(parser, "aug_dac_only_apply_channel", False, "DAC-only视图叠加通道扰动", "DAC-only视图不叠加通道扰动")
    add_bool_arg(parser, "aug_pa_only_apply_anti_shortcut", False, "PA-only视图叠加anti-shortcut", "PA-only视图不叠加anti-shortcut")
    add_bool_arg(parser, "aug_pa_only_apply_channel", False, "PA-only视图叠加通道扰动", "PA-only视图不叠加通道扰动")
    add_bool_arg(parser, "aug_dac_pa_apply_anti_shortcut", True, "DAC+PA视图叠加anti-shortcut", "DAC+PA视图不叠加anti-shortcut")
    add_bool_arg(parser, "aug_dac_pa_apply_channel", True, "DAC+PA视图叠加通道扰动", "DAC+PA视图不叠加通道扰动")

    parser.add_argument("--aug_scale_min", type=float, default=0.10)
    parser.add_argument("--aug_scale_max", type=float, default=0.80)
    parser.add_argument("--aug_warmup_epochs", type=int, default=3)
    parser.add_argument("--aug_ramp_epochs", type=int, default=20)
    parser.add_argument("--aug_ramp_curve", type=float, default=1.5)

    parser.add_argument("--aug_p_dac", type=float, default=0.35)
    parser.add_argument("--aug_p_pa", type=float, default=0.5)
    parser.add_argument("--aug_class_sig_mix", type=float, default=0.1)
    parser.add_argument("--aug_p_rx_chain", type=float, default=0.0,
                        help="Probability of receiver-chain domain randomization on the normal training view.")
    parser.add_argument("--aug_rx_chain_envs", type=int, default=4)
    parser.add_argument("--aug_rx_chain_fs_hz", type=float, default=25e6)
    parser.add_argument("--aug_rx_chain_p_lowpass", type=float, default=0.7)
    parser.add_argument("--aug_rx_chain_p_multipath", type=float, default=0.7)
    parser.add_argument("--aug_defect_strength_mode", type=str, default="tiered", choices=["random", "tiered"])
    parser.add_argument("--aug_dac_only_tiers", type=str, default="0.15,0.35,0.55")
    parser.add_argument("--aug_pa_only_tiers", type=str, default="0.15,0.35,0.60")

    parser.add_argument("--aug_p_time_shift", type=float, default=0.35)
    parser.add_argument("--aug_max_time_shift", type=int, default=32)
    parser.add_argument("--aug_p_amp_scale", type=float, default=0.45)
    parser.add_argument("--aug_amp_min", type=float, default=0.90)
    parser.add_argument("--aug_amp_max", type=float, default=1.10)
    parser.add_argument("--aug_p_phase_rot", type=float, default=0.45)
    parser.add_argument("--aug_p_cfo", type=float, default=0.35)
    parser.add_argument("--aug_cfo_max", type=float, default=4e-4)
    parser.add_argument("--aug_p_phase_noise", type=float, default=0.30)
    parser.add_argument("--aug_phase_noise_sigma_max", type=float, default=0.006)
    parser.add_argument("--aug_p_awgn", type=float, default=0.40)
    parser.add_argument("--aug_snr_min_db", type=float, default=20.0)
    parser.add_argument("--aug_snr_max_db", type=float, default=36.0)
    parser.add_argument("--aug_p_multipath", type=float, default=0.18)
    parser.add_argument("--aug_mp_taps_min", type=int, default=2)
    parser.add_argument("--aug_mp_taps_max", type=int, default=4)
    parser.add_argument("--aug_mp_delay_max", type=int, default=4)
    parser.add_argument("--aug_p_dc_offset", type=float, default=0.30)
    parser.add_argument("--aug_dc_offset_max", type=float, default=0.02)
    parser.add_argument("--aug_p_bandedge_taper", type=float, default=0.25)
    parser.add_argument("--aug_taper_alpha_min", type=float, default=0.02)
    parser.add_argument("--aug_taper_alpha_max", type=float, default=0.10)

    parser.add_argument("--aug_dac_jitter_max", type=float, default=0.002)
    parser.add_argument("--aug_dac_poly_a3", type=float, default=0.12)
    parser.add_argument("--aug_dac_poly_a5", type=float, default=0.03)
    parser.add_argument("--aug_dac_iq_img_max", type=float, default=0.04)
    parser.add_argument("--aug_dac_inter_gain_max", type=float, default=0.03)
    parser.add_argument("--aug_dac_inter_off_max", type=float, default=0.008)
    parser.add_argument("--aug_dac_inter_skew_max", type=float, default=0.05)
    parser.add_argument("--aug_dac_dither", type=float, default=0.002)
    parser.add_argument("--aug_dac_inl_warp", type=float, default=0.03)
    parser.add_argument("--aug_dac_spur_amp_max", type=float, default=0.012)
    parser.add_argument("--aug_dac_slew_max", type=float, default=0.18)

    parser.add_argument("--aug_pa_mp_sigma", type=float, default=0.05)
    parser.add_argument("--aug_pa_mem_sigma", type=float, default=0.04)
    parser.add_argument("--aug_pa_ampm_max", type=float, default=0.20)
    parser.add_argument("--aug_pa_iq_img_max", type=float, default=0.02)

    add_bool_arg(parser, "eval_sat_channel", False,
                 "Enable satellite-channel OOD evaluation after each epoch",
                 "Disable satellite-channel OOD evaluation")
    parser.add_argument("--eval_sat_scenarios", type=str,
                        default="clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit",
                        help="Satellite scenarios to evaluate. Built-ins: clear_leo,low_elev_leo,rain_leo,storm_mp,geo_clear,mixed_orbit.")
    parser.add_argument("--eval_sat_on", type=str, default="all",
                        help="Named test loaders for satellite evaluation: all, main, strict/test_unseen_day_unseen_rx, or comma-separated names.")
    parser.add_argument("--sat_eval_max_batches", type=int, default=-1,
                        help="Max batches for satellite evaluation. <0 reuses --eval_max_batches.")
    parser.add_argument("--sat_seed", type=int, default=2027)
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    add_bool_arg(parser, "use_sat_consistency", False,
                 "Enable clean-to-satellite consistency training",
                 "Disable clean-to-satellite consistency training")
    add_bool_arg(parser, "train_sat_channel", False,
                 "Alias for enabling SGC satellite-channel augmentation training",
                 "Disable SGC satellite-channel augmentation training")
    parser.add_argument("--sat_train_scenario", type=str, default="clear_leo")
    parser.add_argument("--train_sat_scenario", type=str, default="",
                        help="Alias for --sat_train_scenario used by SGC scripts.")
    parser.add_argument("--sat_view_source", type=str, default="main", choices=["clean", "main"],
                        help="Input used to build the satellite-ground training view: clean raw IQ or the main augmented view.")
    parser.add_argument("--lambda_sat_cons", type=float, default=0.0,
                        help="Cosine-distance consistency weight between clean and satellite z_id features.")
    parser.add_argument("--lambda_sat_cls", type=float, default=0.0,
                        help="Classification CE weight on satellite-channel augmented samples.")
    parser.add_argument("--sat_cons_start_epoch", type=int, default=1)
    parser.add_argument("--lambda_ecc", type=float, default=0.0,
                        help="Early Confidence Cap loss weight. >0 penalizes early over-confident TX predictions.")
    parser.add_argument("--ecc_start_epoch", type=int, default=1,
                        help="First epoch where Early Confidence Cap can be active.")
    parser.add_argument("--ecc_epochs", type=int, default=60,
                        help="Number of epochs used for ECC tau ramp and weight decay.")
    parser.add_argument("--ecc_tau_start", type=float, default=0.65,
                        help="Initial max-probability cap for ECC.")
    parser.add_argument("--ecc_tau_end", type=float, default=0.95,
                        help="Final max-probability cap for ECC.")
    parser.add_argument("--ecc_schedule", type=str, default="cosine", choices=["cosine", "linear"],
                        help="Schedule shape used by ECC tau ramp and weight decay.")
    parser.add_argument("--ecc_apply_to", type=str, default="sat",
                        choices=["sat", "main", "sat_main"],
                        help="Logits regularized by ECC: SAT view, main augmented view, or both.")

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--min_batch_domains_for_domain_loss", type=int, default=2)
    parser.add_argument("--min_batch_domain_frac", type=float, default=0.15)
    parser.add_argument("--clip_grad_backbone", type=float, default=1.0)
    parser.add_argument("--clip_grad_aux", type=float, default=0.75)
    parser.add_argument("--clip_grad_domain", type=float, default=0.5)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--best_save_path", type=str, default="best_model.pth",
                        help="按 VAL tx_acc 最优保存的权重路径。")
    parser.add_argument("--latest_save_path", type=str, default="latest_model.pth",
                        help="每个 epoch 覆盖保存的最新权重路径。")
    parser.add_argument("--best_test_save_path", type=str, default="",
                        help="按 overall TEST tx_acc 最优保存的权重路径。为空时由 best_save_path 自动派生。")
    parser.add_argument("--best_primary_save_path", type=str, default="",
                        help="Best checkpoint by primary OOD score: (1-w)*overall + w*unseen_day_unseen_rx.")
    parser.add_argument("--primary_udu_weight", type=float, default=0.5,
                        help="Weight of unseen_day_unseen_rx in the primary OOD checkpoint score.")
    parser.add_argument("--best_unseen_day_unseen_rx_save_path", type=str, default="best_test_model.pth",
                        help="按 test_unseen_day_unseen_rx 最优保存的权重路径；这是最严格跨日期+跨接收机指标。")
    parser.add_argument("--best_unseen_day_seen_rx_save_path", type=str, default="",
                        help="按 test_unseen_day_seen_rx 最优保存的权重路径。")
    parser.add_argument("--best_seen_day_unseen_rx_save_path", type=str, default="",
                        help="按 test_seen_day_unseen_rx 最优保存的权重路径。")
    parser.add_argument("--best_worst_rx_save_path", type=str, default="",
                        help="Best checkpoint by the minimum tx_acc among test_rx_* receiver groups.")
    add_bool_arg(parser, "use_ema_ckpt", False,
                 "Track and evaluate EMA checkpoint averaging",
                 "Disable EMA checkpoint averaging")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_epoch", type=int, default=1)
    parser.add_argument("--ema_save_path", type=str, default="")
    add_bool_arg(parser, "use_swa_ckpt", False,
                 "Track and evaluate SWA checkpoint averaging",
                 "Disable SWA checkpoint averaging")
    parser.add_argument("--swa_start_epoch", type=int, default=120)
    parser.add_argument("--swa_interval", type=int, default=5)
    parser.add_argument("--swa_save_path", type=str, default="")
    add_bool_arg(parser, "use_swad_ckpt", False,
                 "Track and evaluate SWAD-style dense checkpoint averaging",
                 "Disable SWAD-style dense checkpoint averaging")
    parser.add_argument("--swad_start_epoch", type=int, default=80)
    parser.add_argument("--swad_interval", type=int, default=1)
    parser.add_argument("--swad_tolerance", type=float, default=2.0,
                        help="Collect epochs whose primary OOD score is within this margin of the best-so-far score.")
    parser.add_argument("--swad_save_path", type=str, default="")
    args = parser.parse_args()
    if str(getattr(args, "preset", "") or "").strip():
        args.slim_group = str(args.preset).strip()
    args.stage = str(getattr(args, "stage", "source") or "source").lower().strip()
    if args.stage == "sgc_adapt":
        args.sgc_adapter = True
        args.epochs = int(args.adapt_epochs)
        args.lr = float(args.adapt_lr)
    if str(getattr(args, "train_sat_scenario", "") or "").strip():
        args.sat_train_scenario = str(args.train_sat_scenario).strip()
    if args.stage == "sgc_augment":
        args.sgc_adapter = True
        args.train_sat_channel = True
        if float(args.lambda_sat_cons) <= 0.0:
            args.lambda_sat_cons = float(args.lambda_feat)
        if float(args.lambda_sat_cls) <= 0.0:
            args.lambda_sat_cls = 1.0
    if bool(getattr(args, "train_sat_channel", False)):
        args.use_sat_consistency = True
    args = apply_slim_ablation_preset(args)
    args = apply_experiment_preset(args)
    args = apply_slim_post_preset_overrides(args)
    args = apply_model_variant_training_defaults(args)
    args = align_training_with_branch_ablation(args)
    if str(getattr(args, "force_domain_branch_ablation", "") or "").strip():
        args.domain_branch_ablation = str(args.force_domain_branch_ablation).strip()
    if str(getattr(args, "force_domain_enhancer", "") or "").strip():
        args.domain_enhancer = str(args.force_domain_enhancer).strip()
    if float(getattr(args, "force_domain_enhancer_strength", -1.0)) >= 0.0:
        args.domain_enhancer_strength = float(args.force_domain_enhancer_strength)
    if getattr(args, "force_lambda_adv", None) is not None:
        args.lambda_adv = float(args.force_lambda_adv)
    if getattr(args, "force_lambda_orth", None) is not None:
        args.lambda_orth = float(args.force_lambda_orth)
    if getattr(args, "force_lambda_cons", None) is not None:
        args.lambda_cons = float(args.force_lambda_cons)
    args.sgc_adapter_kwargs = parse_json_dict(args.sgc_adapter_kwargs, "sgc_adapter_kwargs")
    explicit_no_sgc = str(getattr(args, "slim_group", "")).lower().strip() == "sgc_baseline_no_adapter"
    if args.stage in ("sgc_augment", "sgc_adapt") and not explicit_no_sgc:
        args.sgc_adapter = True
    if args.stage == "sgc_adapt":
        args.epochs = int(args.adapt_epochs)
        args.lr = float(args.adapt_lr)
    if args.stage == "sgc_augment" and float(args.lambda_sat_cons) <= 0.0:
        args.lambda_sat_cons = float(args.lambda_feat)
    if bool(getattr(args, "train_sat_channel", False)):
        args.use_sat_consistency = True

    # Auto-derive extra checkpoint paths after preset application.
    if str(args.best_test_save_path).strip() == "":
        args.best_test_save_path = derive_checkpoint_path(args.best_save_path, "test_overall")
    if str(args.best_primary_save_path).strip() == "":
        args.best_primary_save_path = derive_checkpoint_path(args.best_save_path, "primary_ood")
    if str(args.best_unseen_day_unseen_rx_save_path).strip() == "":
        args.best_unseen_day_unseen_rx_save_path = derive_checkpoint_path(args.best_save_path, "test_unseen_day_unseen_rx")
    if str(args.best_unseen_day_seen_rx_save_path).strip() == "":
        args.best_unseen_day_seen_rx_save_path = derive_checkpoint_path(args.best_save_path, "test_unseen_day_seen_rx")
    if str(args.best_seen_day_unseen_rx_save_path).strip() == "":
        args.best_seen_day_unseen_rx_save_path = derive_checkpoint_path(args.best_save_path, "test_seen_day_unseen_rx")
    if str(args.best_worst_rx_save_path).strip() == "":
        args.best_worst_rx_save_path = derive_checkpoint_path(args.best_save_path, "test_worst_rx")
    if str(args.ema_save_path).strip() == "":
        args.ema_save_path = derive_checkpoint_path(args.best_save_path, "ema")
    if str(args.swa_save_path).strip() == "":
        args.swa_save_path = derive_checkpoint_path(args.best_save_path, "swa")
    if str(args.swad_save_path).strip() == "":
        args.swad_save_path = derive_checkpoint_path(args.best_save_path, "swad")

    args.eval_sat_scenario_list = parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []
    args.sat_train_scenario = str(args.sat_train_scenario or "clear_leo").strip().lower().replace("-", "_")
    if (bool(args.eval_sat_channel) or bool(args.use_sat_consistency)) and SatSimConfig is None:
        raise ImportError("sat_channel.py is required when --eval_sat_channel or --use_sat_consistency is enabled.")
    if bool(args.eval_sat_channel):
        for scenario in args.eval_sat_scenario_list:
            sat_channel_config_for_scenario(scenario)
    if bool(args.use_sat_consistency):
        sat_channel_config_for_scenario(args.sat_train_scenario)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Starting Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} | AMP: {use_amp} | use_aug={args.use_aug}")
    print(f"[EXP] {args.exp_group} | {getattr(args, 'exp_desc', 'custom')}")
    print(f"[SLIM] group={args.slim_group} | {getattr(args, 'slim_desc', 'manual override')}")
    if bool(args.eval_sat_channel):
        print(
            f"[SAT-EVAL] enabled scenarios={','.join(args.eval_sat_scenario_list)} "
            f"on={args.eval_sat_on} max_batches={args.sat_eval_max_batches}",
            flush=True,
        )
    if bool(args.use_sat_consistency):
        print(
            f"[SAT-TRAIN] scenario={args.sat_train_scenario} "
            f"view_source={args.sat_view_source} "
            f"lambda_cons={args.lambda_sat_cons:.4f} lambda_cls={args.lambda_sat_cls:.4f} "
            f"start_epoch={args.sat_cons_start_epoch}",
            flush=True,
        )
    if float(args.lambda_ecc) > 0.0:
        print(
            f"[ECC] lambda={args.lambda_ecc:.4f} apply_to={args.ecc_apply_to} "
            f"tau={args.ecc_tau_start:.3f}->{args.ecc_tau_end:.3f} "
            f"start_epoch={args.ecc_start_epoch} epochs={args.ecc_epochs} schedule={args.ecc_schedule}",
            flush=True,
        )
    print(f"[EXP] pa_main={args.aug_enable_pa_normal} pa_aux={args.enable_pa_aux} dac_aux={args.enable_dac_aux} | aug_p_pa={args.aug_p_pa:.3f} aug_p_dac={args.aug_p_dac:.3f}")
    print(f"[EXP] pure_views: dac_only(channel={args.aug_dac_only_apply_channel}, anti={args.aug_dac_only_apply_anti_shortcut}) | pa_only(channel={args.aug_pa_only_apply_channel}, anti={args.aug_pa_only_apply_anti_shortcut})")
    print(f"[EXP] stage schedule: stage1<=E{args.stage1_epochs}, stage2<=E{args.stage2_epochs}, stage3 ramp={args.stage3_ramp_epochs}")
    print(f"[EXP] pure_views: dac_only(channel={args.aug_dac_only_apply_channel}, anti={args.aug_dac_only_apply_anti_shortcut}) "
          f"pa_only(channel={args.aug_pa_only_apply_channel}, anti={args.aug_pa_only_apply_anti_shortcut}) "
          f"| defect_strength_mode={args.aug_defect_strength_mode}")

    if float(args.sample_rate_hz) <= 0.0:
        args.sample_rate_hz = 25e6 if args.dataset == "wisig" else 5e6

    if args.dataset == "wisig":
        if float(args.wisig_val_ratio) > 0.0:
            args.wisig_train_ratio = 1.0 - float(args.wisig_val_ratio)
        if not (0.01 <= float(args.wisig_train_ratio) <= 0.99):
            raise ValueError(
                f"--wisig_train_ratio must be in [0.01, 0.99] after optional --wisig_val_ratio override, "
                f"got {args.wisig_train_ratio}"
            )

    split_info = None
    input_len = 1024
    val_ds = None
    test_ds = None
    named_tests = {}
    named_test_meta = {}

    if args.dataset == "wisig":
        ds_w = load_wisig_compact_pkl(args.wisig_pkl)
        infer_nc = len(ds_w.get("tx_list", []))
        if infer_nc > 0 and args.num_classes != infer_nc:
            print(f"[WISIG] overriding num_classes {args.num_classes} -> {infer_nc}")
            args.num_classes = infer_nc

        eq2 = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
        max_day123 = None if int(args.wisig_max_day123_per_combo) <= 0 else int(args.wisig_max_day123_per_combo)
        max_tr = None if int(args.wisig_max_train_per_combo) <= 0 else int(args.wisig_max_train_per_combo)
        max_va = None if int(args.wisig_max_val_per_combo) <= 0 else int(args.wisig_max_val_per_combo)
        max_te = None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo)

        train_ds, val_ds, test_ds, named_tests, named_test_meta, split_info = make_wisig_trainval_test_by_day_rx(
            ds_w,
            equalized=eq2,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform_train=None,
            transform_eval=None,
            train_ratio=float(args.wisig_train_ratio),
            guard_gap=int(args.wisig_guard_gap),
            train_days=parse_csv_indices(args.wisig_train_days),
            test_days=parse_csv_indices(args.wisig_test_days),
            train_rxs=parse_csv_indices(args.wisig_train_rxs),
            test_rxs=parse_csv_indices(args.wisig_test_rxs),
            max_samples_per_combo_day123=max_day123,
            max_samples_per_combo_test=max_te,
            max_samples_per_combo_train=max_tr,
            max_samples_per_combo_val=max_va,
            seed=int(args.seed),
        )
        input_len = int(args.wisig_out_len)
        print(f"[WISIG] pkl={args.wisig_pkl} eq={eq2} out_len={input_len} domain={args.wisig_domain}")
        print(f"[WISIG] TRAIN DAYS: {split_info['train_days_label']} | TRAIN RXS: {split_info['train_rxs_idx']}")
        print(f"[WISIG] TEST  DAYS: {split_info['test_days_label']} | TEST  RXS: {split_info['test_rxs_idx']}")
        print(f"[WISIG] VAL   source: same train_days x train_rxs tail (guard_gap={split_info['guard_gap']})")
        print(
            f"[WISIG-SPLIT] train_ratio={split_info['train_ratio']:.3f} "
            f"requested_val_ratio={split_info.get('requested_val_ratio', 1.0 - split_info['train_ratio']):.3f} "
            f"effective_train={split_info.get('effective_train_ratio', 0.0):.3f} "
            f"effective_val={split_info.get('effective_val_ratio', 0.0):.3f}"
        )
        print(f"[WISIG] named_test_sizes={split_info['named_test_sizes']}")
        print(f"[WISIG] split_info={split_info}")
    else:
        train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train", run_name=args.run_name)
        test_ds = WiFiRFFIDataset(args.dataset_dir, mode="test", run_name=args.run_name)
        val_ds = test_ds
        named_tests = {"test_default": test_ds}
        named_test_meta = {"test_default": {"size": len(test_ds)}}
        try:
            x0, _ = train_ds[0]
            input_len = int(x0.shape[-1])
        except Exception:
            input_len = 1024
        print(f"[ORALCE] dir={args.dataset_dir} run={args.run_name} input_len={input_len}")
        print("[WARN] ORALCE currently has no separate val set in this script; val=test only for compatibility.")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device, True, args.prefetch_factor)
    val_loader = make_loader(val_ds, args.eval_batch_size, False, args.num_workers, device, False, args.prefetch_factor)
    named_test_loaders = {
        k: make_loader(ds, args.eval_batch_size, False, args.num_workers, device, False, args.prefetch_factor)
        for k, ds in named_tests.items()
    }

    print_dataset_sample_summary(args, train_ds, val_ds)
    domain_label_map = build_domain_label_map(train_ds)
    num_domains = max(1, len(domain_label_map))
    print_domain_configuration(args, train_ds, split_info, domain_label_map)

    model = build_dual_model(args.num_classes, num_domains, model_size=args.model_size, dataset=args.dataset,
                             input_len=input_len, sample_rate_hz=float(args.sample_rate_hz),
                             id_feature_key="feat_joint", dom_feature_key="feat_imp",
                             model_variant=str(args.model_variant),
                             branch_ablation=str(args.branch_ablation),
                             mixstyle_on=bool(args.use_mixstyle),
                             mixstyle_p=float(args.mixstyle_p),
                             mixstyle_alpha=float(args.mixstyle_alpha),
                             mixstyle_eps=float(args.mixstyle_eps),
                             mixstyle_layers=str(args.mixstyle_layers),
                             mixstyle_use_domain_label=bool(args.mixstyle_use_domain_label),
                             mixstyle_mix=str(args.mixstyle_mix),
                             mixstyle_strength=float(args.mixstyle_strength),
                             mixstyle_fallback=str(args.mixstyle_fallback),
                             domain_branch_ablation=str(args.domain_branch_ablation),
                             domain_enhancer=str(args.domain_enhancer),
                             domain_enhancer_strength=float(args.domain_enhancer_strength),
                             fast_infer_when_no_aux=bool(args.fast_infer_when_no_aux),
                             sgc_adapter=bool(args.sgc_adapter),
                             sgc_adapter_kwargs=args.sgc_adapter_kwargs).to(device)
    if str(args.source_ckpt).strip():
        state = load_checkpoint_model_state(args.source_ckpt, device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[SGC-CKPT] loaded source_ckpt={args.source_ckpt} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    sgc_train_params = configure_sgc_trainable_params(model, args)
    model_emb_dim = getattr(model, "emb_dim", "unknown")
    n_total, n_trainable = count_parameters(model)
    if bool(args.compile_model):
        try:
            model = torch.compile(model)
            print("[MODEL] torch.compile enabled")
        except Exception as exc:
            print(f"[MODEL-WARN] torch.compile failed, fallback to eager: {exc}")
    print(f"[MODEL] DualCVSincNetDisentangle variant={args.model_variant} branch_ablation={args.branch_ablation} emb_dim={model_emb_dim} num_domains={num_domains} params={n_total:,} trainable={n_trainable:,}")
    raw_model = unwrap_model(model)
    if hasattr(raw_model, "sgc_adapter") and raw_model.sgc_adapter is not None:
        sgc_params = sum(p.numel() for p in raw_model.sgc_adapter.parameters())
        print(
            f"[SGC-ADAPTER] enabled=True stage={args.stage} params={sgc_params:,} "
            f"submodules={raw_model.sgc_adapter.submodule_status}",
            flush=True,
        )
    else:
        print(f"[SGC-ADAPTER] enabled=False stage={args.stage}", flush=True)
    print(
        "[MIXSTYLE] "
        f"on={int(args.use_mixstyle)} p={args.mixstyle_p:.3f} alpha={args.mixstyle_alpha:.3f} "
        f"eps={args.mixstyle_eps:.1e} layers={args.mixstyle_layers} "
        f"use_domain={int(args.mixstyle_use_domain_label)} mix={args.mixstyle_mix} "
        f"strength={args.mixstyle_strength:.2f} fallback={args.mixstyle_fallback}"
    )
    print(
        "[MIXSTYLE-SCHEDULE] "
        f"late_start={args.mixstyle_late_start or args.late_stable_start} "
        f"ramp={args.mixstyle_late_ramp_epochs or args.late_stable_ramp_epochs} "
        f"min_p={args.mixstyle_late_min_p:.3f} min_strength={args.mixstyle_late_min_strength:.2f} "
        f"stop_epoch={args.mixstyle_stop_epoch}"
    )
    print(
        "[DOMAIN-BACKBONE] "
        f"branch_ablation={args.domain_branch_ablation} enhancer={args.domain_enhancer} "
        f"enhancer_strength={args.domain_enhancer_strength:.2f} "
        f"fast_infer_no_aux={int(args.fast_infer_when_no_aux)}"
    )

    aug_base_cfg = build_aug_base_cfg(args) if args.use_aug else None
    augmentor = make_augmentor(aug_base_cfg) if args.use_aug else None
    if args.use_aug:
        print(
            "[AUG-INIT] enabled | "
            f"scale:[{args.aug_scale_min:.2f}->{args.aug_scale_max:.2f}] warmup={args.aug_warmup_epochs} ramp={args.aug_ramp_epochs} "
            f"curve={args.aug_ramp_curve:.2f} | base_p_dac={args.aug_p_dac:.2f} base_p_pa={args.aug_p_pa:.2f}"
        )

    optimizer_params = sgc_train_params if sgc_train_params is not None else model.parameters()
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_tx = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    ce_dom = nn.CrossEntropyLoss()

    best_joint_val_tx = -1.0
    best_joint_test_tx = -1.0
    best_epoch = -1

    # Additional checkpoint criteria for cross-domain research.
    # best_test_epoch: best aggregated test over the main named test buckets.
    # best_unseen_day_unseen_rx_epoch: strictest held-out domain criterion.
    best_test_tx = -1.0
    best_test_epoch = -1
    best_primary_score = -1.0
    best_primary_epoch = -1
    best_primary_test_tx = -1.0
    best_primary_unseen_day_unseen_rx_tx = -1.0
    best_unseen_day_unseen_rx_tx = -1.0
    best_unseen_day_unseen_rx_epoch = -1
    best_unseen_day_seen_rx_tx = -1.0
    best_unseen_day_seen_rx_epoch = -1
    best_seen_day_unseen_rx_tx = -1.0
    best_seen_day_unseen_rx_epoch = -1
    best_worst_rx_tx = -1.0
    best_worst_rx_name = ""
    best_worst_rx_epoch = -1

    print("[CKPT-PATHS]", flush=True)
    print(f"  latest                       -> {args.latest_save_path}", flush=True)
    print(f"  best_by_val                  -> {args.best_save_path}", flush=True)
    print(f"  best_by_test_overall         -> {args.best_test_save_path}", flush=True)
    print(f"  best_by_primary_ood         -> {args.best_primary_save_path} (udu_weight={args.primary_udu_weight:.2f})", flush=True)
    print(f"  best_by_unseen_day_unseen_rx -> {args.best_unseen_day_unseen_rx_save_path}", flush=True)
    print(f"  best_by_unseen_day_seen_rx   -> {args.best_unseen_day_seen_rx_save_path}", flush=True)
    print(f"  best_by_seen_day_unseen_rx   -> {args.best_seen_day_unseen_rx_save_path}", flush=True)
    print(f"  best_by_worst_rx             -> {args.best_worst_rx_save_path}", flush=True)
    if bool(args.use_ema_ckpt):
        print(f"  ema_average                  -> {args.ema_save_path}", flush=True)
    if bool(args.use_swa_ckpt):
        print(f"  swa_average                  -> {args.swa_save_path}", flush=True)
    if bool(args.use_swad_ckpt):
        print(f"  swad_average                 -> {args.swad_save_path}", flush=True)

    skipped_backward_batches = 0
    loss_warn_counts = {}
    groupdro_state = SmoothGroupDROState(momentum=float(args.groupdro_momentum))
    proto_bank = PrototypeMemoryBank(
        int(args.num_classes),
        int(num_domains),
        momentum=float(args.proto_momentum),
        margin=float(args.proto_margin),
        domain_align_weight=float(args.proto_domain_align_weight),
        push_weight=float(args.proto_push_weight),
        min_count=int(args.proto_min_count),
    ) if bool(args.use_proto_memory) or float(args.lambda_proto) > 0.0 else None
    ema_avg = AveragedModelState("ema", decay=float(args.ema_decay)) if bool(args.use_ema_ckpt) else None
    swa_avg = AveragedModelState("swa") if bool(args.use_swa_ckpt) else None
    swad_avg = AveragedModelState("swad") if bool(args.use_swad_ckpt) else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        skipped_before_epoch = int(skipped_backward_batches)
        t0 = time.time()
        meters = {k: AverageMeter() for k in [
            "loss", "cls", "dom", "adv", "orth", "cons", "group_ce", "txacc",
            "cls_pa", "cls_dac", "pa_joint_inv", "pa_kl",
            "dac_reg", "pa_reg",
            "gap_dac", "gap_pa", "cos_joint_pa", "cos_imp_pa",
            "sat_cls", "sat_cons", "sat_cos",
            "proto", "proto_pull_cos", "supcon", "fishr", "sgc_res",
            "ecc", "ecc_w", "ecc_tau", "ecc_maxp",
            "grad_total", "grad_backbone", "grad_aux", "grad_domain",
        ]}
        m_domacc = NanMeter()
        cons_cos_vals = []
        mixstyle_state = configure_mixstyle_for_epoch(model, args, epoch)
        aug_state = configure_augmentor_for_epoch(augmentor, aug_base_cfg, epoch, args) if augmentor is not None else None
        aux_scale = ramp_value(epoch, args.epochs, int(args.aux_warmup_epochs), int(args.aux_ramp_epochs), 0.0, 1.0, 1.0)
        stage_state = build_stage_state(epoch, args)
        cur_w = current_weight_dict(args, stage_state)

        for batch in train_loader:
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d_raw = extract_domain_from_extra(extra, device)
            d = remap_domain_tensor(d_raw, domain_label_map, device) if d_raw is not None else None
            domain_stats = batch_domain_stats(d, y, num_domains)
            stage_state, cur_w, domain_gates = training_stage_controller(epoch, args, domain_stats, num_domains)

            need_dac_aux = bool(args.enable_dac_aux and stage_state["use_aux_views"] > 0.0 and (
                cur_w["cls_dac"] > 0.0 or cur_w["dac_reg"] > 0.0
            ))
            need_pa_aux = bool(args.enable_pa_aux and stage_state["use_aux_views"] > 0.0 and (
                cur_w["cls_pa"] > 0.0 or cur_w["pa_joint_inv"] > 0.0 or cur_w["pa_kl"] > 0.0 or cur_w["pa_reg"] > 0.0
            ))

            if augmentor is not None:
                with torch.no_grad():
                    x_main = safe_iq_tensor(augmentor(x, labels=y, no_pa=(not args.aug_enable_pa_normal)))
                    if float(args.aug_p_rx_chain) > 0.0 and torch.rand((), device=x_main.device) < float(args.aug_p_rx_chain):
                        env_id = torch.randint(
                            low=0,
                            high=max(1, int(args.aug_rx_chain_envs)),
                            size=(x_main.size(0),),
                            device=x_main.device,
                        )
                        fs_rx = float(args.aug_rx_chain_fs_hz) if float(args.aug_rx_chain_fs_hz) > 0 else float(args.sample_rate_hz or 25e6)
                        x_main = safe_iq_tensor(apply_receiver_dg(
                            x_main,
                            fs=fs_rx,
                            env_id=env_id,
                            p_lowpass=float(args.aug_rx_chain_p_lowpass),
                            p_multipath=float(args.aug_rx_chain_p_multipath),
                        ))
                    if need_dac_aux:
                        if str(args.aug_defect_strength_mode).lower() == "tiered":
                            s_dac_in = sample_strength_from_tiers(x.size(0), parse_float_csv(args.aug_dac_only_tiers, [0.15, 0.35, 0.55]), x.device, x.dtype)
                        else:
                            s_dac_in = None
                        x_dac, s_dac = augmentor(x, labels=y, dac_only=True, return_dac_strength=True, dac_strength=s_dac_in)
                        x_dac = safe_iq_tensor(x_dac)
                    else:
                        x_dac = x_main
                        s_dac = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                    if need_pa_aux:
                        if str(args.aug_defect_strength_mode).lower() == "tiered":
                            s_pa_in = sample_strength_from_tiers(x.size(0), parse_float_csv(args.aug_pa_only_tiers, [0.15, 0.35, 0.60]), x.device, x.dtype)
                        else:
                            s_pa_in = None
                        x_pa, s_pa = augmentor(x, labels=y, pa_only=True, return_pa_strength=True, pa_strength=s_pa_in)
                        x_pa = safe_iq_tensor(x_pa)
                    else:
                        x_pa = x_main
                        s_pa = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            else:
                x_main = safe_iq_tensor(x)
                if float(args.aug_p_rx_chain) > 0.0 and torch.rand((), device=x_main.device) < float(args.aug_p_rx_chain):
                    env_id = torch.randint(
                        low=0,
                        high=max(1, int(args.aug_rx_chain_envs)),
                        size=(x_main.size(0),),
                        device=x_main.device,
                    )
                    fs_rx = float(args.aug_rx_chain_fs_hz) if float(args.aug_rx_chain_fs_hz) > 0 else float(args.sample_rate_hz or 25e6)
                    x_main = safe_iq_tensor(apply_receiver_dg(
                        x_main,
                        fs=fs_rx,
                        env_id=env_id,
                        p_lowpass=float(args.aug_rx_chain_p_lowpass),
                        p_multipath=float(args.aug_rx_chain_p_multipath),
                    ))
                x_dac = x_main
                x_pa = x_main
                s_dac = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                s_pa = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

            if need_dac_aux or need_pa_aux:
                with torch.no_grad():
                    anchor = forward_anchor_eval(model, x, y, grl_lambda=float(args.grl_lambda), domain_labels=d_raw)
            else:
                anchor = None

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_main = forward_main(model, x_main, y, float(args.grl_lambda), domain_labels=d_raw)
                out_dac = forward_aux(model, x_dac, y, float(args.grl_lambda), need_dac_aux, domain_labels=d_raw)
                out_pa = forward_aux(model, x_pa, y, float(args.grl_lambda), need_pa_aux, domain_labels=d_raw)

                tx_logits = out_main["tx_logits"]
                z_id = out_main["z_id"]
                core = compute_core_losses(
                    out_main,
                    y,
                    d,
                    domain_stats,
                    domain_gates,
                    ce_tx,
                    ce_dom,
                    label_smoothing=float(args.label_smoothing),
                    group_top_frac=float(args.group_ce_top_frac),
                    group_min_domains=int(args.group_ce_min_domains),
                    group_ce_mode=str(args.group_ce_mode),
                    groupdro_state=groupdro_state,
                    groupdro_tau=float(args.groupdro_tau),
                    groupdro_cap=float(args.groupdro_cap),
                    groupdro_num_days=int(args.groupdro_num_days),
                )
                if not math.isnan(core["cons_cos"]):
                    cons_cos_vals.append(core["cons_cos"])
                m_domacc.update(core["dom_acc"])
                if anchor is None:
                    aux = {
                        "loss_cls_pa": z_id.new_tensor(0.0), "loss_cls_dac": z_id.new_tensor(0.0),
                        "loss_pa_joint_inv": z_id.new_tensor(0.0),
                        "loss_pa_kl": z_id.new_tensor(0.0), "loss_dac_reg": z_id.new_tensor(0.0),
                        "loss_pa_reg": z_id.new_tensor(0.0),
                        "shift_dac_on_dac": torch.zeros(y.size(0), device=z_id.device, dtype=z_id.dtype),
                        "shift_dac_on_pa": torch.zeros(y.size(0), device=z_id.device, dtype=z_id.dtype),
                        "shift_pa_on_pa": torch.zeros(y.size(0), device=z_id.device, dtype=z_id.dtype),
                        "shift_pa_on_dac": torch.zeros(y.size(0), device=z_id.device, dtype=z_id.dtype),
                        "cos_joint_pa": float("nan"), "cos_imp_pa": float("nan"),
                    }
                else:
                    aux = compute_aux_losses(out_dac, out_pa, anchor, y, s_dac, s_pa, need_dac_aux, need_pa_aux, cur_w, args, ce_tx, z_id)

                loss_sat_cls = z_id.new_tensor(0.0)
                loss_sat_cons = z_id.new_tensor(0.0)
                loss_ecc = z_id.new_tensor(0.0)
                sat_cos = float("nan")
                ecc_maxp = float("nan")
                ecc_tau = ecc_tau_for_epoch(
                    epoch,
                    float(args.ecc_tau_start),
                    float(args.ecc_tau_end),
                    int(args.ecc_epochs),
                    int(args.ecc_start_epoch),
                    str(args.ecc_schedule),
                )
                ecc_w = ecc_weight_for_epoch(
                    epoch,
                    float(args.lambda_ecc),
                    int(args.ecc_epochs),
                    int(args.ecc_start_epoch),
                    str(args.ecc_schedule),
                )
                out_sat = None
                use_sat_train = bool(args.use_sat_consistency) and epoch >= int(args.sat_cons_start_epoch) and (
                    float(args.lambda_sat_cons) > 0.0 or float(args.lambda_sat_cls) > 0.0
                )
                if use_sat_train:
                    with torch.no_grad():
                        sat_view_base = x_main if str(args.sat_view_source).lower() == "main" else x
                        x_sat_train, _ = apply_sat_channel_for_scenario(
                            sat_view_base,
                            args.sat_train_scenario,
                            args,
                            gen=None,
                            return_meta=False,
                        )
                    out_sat = forward_main(model, x_sat_train, y, float(args.grl_lambda), domain_labels=d_raw)
                    loss_sat_cls = ce_tx(out_sat["tx_logits"].float(), y)
                    loss_sat_cons, sat_cos = cosine_consistency_loss(out_sat["z_id"], z_id.detach())
                if ecc_w > 0.0:
                    ecc_losses = []
                    ecc_maxps = []
                    ecc_apply = str(args.ecc_apply_to).lower().strip()
                    if ecc_apply in ("main", "sat_main"):
                        cur_loss_ecc, cur_maxp = compute_ecc_loss(tx_logits, tau=float(ecc_tau))
                        ecc_losses.append(cur_loss_ecc)
                        ecc_maxps.append(cur_maxp)
                    if ecc_apply in ("sat", "sat_main") and out_sat is not None:
                        cur_loss_ecc, cur_maxp = compute_ecc_loss(out_sat["tx_logits"], tau=float(ecc_tau))
                        ecc_losses.append(cur_loss_ecc)
                        ecc_maxps.append(cur_maxp)
                    if ecc_losses:
                        loss_ecc = torch.stack(ecc_losses).mean()
                        ecc_maxp = float(sum(ecc_maxps) / len(ecc_maxps))

                dg_feat = select_generalization_feature(out_main, str(args.generalization_feature))
                loss_proto = z_id.new_tensor(0.0)
                proto_info = {"proto_pull_cos": float("nan")}
                if proto_bank is not None and float(args.lambda_proto) > 0.0:
                    loss_proto, proto_info = proto_bank.loss(dg_feat, y, d)
                loss_supcon = z_id.new_tensor(0.0)
                if float(args.lambda_supcon_id) > 0.0:
                    loss_supcon = domain_aware_supcon_loss(
                        dg_feat,
                        y,
                        d,
                        temperature=float(args.supcon_temp),
                    )
                loss_fishr = z_id.new_tensor(0.0)
                if float(args.lambda_fishr) > 0.0:
                    loss_fishr = fishr_logit_gradient_variance_loss(
                        tx_logits,
                        y,
                        d,
                        min_domains=int(args.fishr_min_domains),
                    )
                loss_sgc_res = sgc_residual_loss_from_output(out_main, z_id)

                loss_cls = core["loss_cls"]
                loss_dom = core["loss_dom"]
                loss_adv = core["loss_adv"]
                loss_orth = core["loss_orth"]
                loss_cons = core["loss_cons"]
                loss_group_ce = core["loss_group_ce"]
                loss_cls_pa = aux["loss_cls_pa"]
                loss_cls_dac = aux["loss_cls_dac"]
                loss_pa_joint_inv = aux["loss_pa_joint_inv"]
                loss_pa_kl = aux["loss_pa_kl"]
                loss_dac_reg = aux["loss_dac_reg"]
                loss_pa_reg = aux["loss_pa_reg"]
                shift_dac_on_dac = aux["shift_dac_on_dac"]
                shift_dac_on_pa = aux["shift_dac_on_pa"]
                shift_pa_on_pa = aux["shift_pa_on_pa"]
                shift_pa_on_dac = aux["shift_pa_on_dac"]
                cos_joint_pa = aux["cos_joint_pa"]
                cos_imp_pa = aux["cos_imp_pa"]

                aux_terms = [
                    cur_w["cls_pa"] * sanitize_loss("cls_pa", loss_cls_pa, z_id, loss_warn_counts),
                    cur_w["cls_dac"] * sanitize_loss("cls_dac", loss_cls_dac, z_id, loss_warn_counts),
                    cur_w["pa_joint_inv"] * sanitize_loss("pa_joint_inv", loss_pa_joint_inv, z_id, loss_warn_counts),
                    cur_w["pa_kl"] * sanitize_loss("pa_kl", loss_pa_kl, z_id, loss_warn_counts),
                    cur_w["dac_reg"] * sanitize_loss("dac_reg", loss_dac_reg, z_id, loss_warn_counts),
                    cur_w["pa_reg"] * sanitize_loss("pa_reg", loss_pa_reg, z_id, loss_warn_counts),
                ]

                loss = (
                    sanitize_loss("cls", loss_cls, z_id, loss_warn_counts)
                    + cur_w["dom"] * sanitize_loss("dom", loss_dom, z_id, loss_warn_counts)
                    + cur_w["adv"] * sanitize_loss("adv", loss_adv, z_id, loss_warn_counts)
                    + cur_w["orth"] * sanitize_loss("orth", loss_orth, z_id, loss_warn_counts)
                    + cur_w["cons"] * sanitize_loss("cons", loss_cons, z_id, loss_warn_counts)
                    + cur_w["group_ce"] * sanitize_loss("group_ce", loss_group_ce, z_id, loss_warn_counts)
                    + aux_scale * sum(aux_terms)
                    + float(args.lambda_sat_cls) * sanitize_loss("sat_cls", loss_sat_cls, z_id, loss_warn_counts)
                    + float(args.lambda_sat_cons) * sanitize_loss("sat_cons", loss_sat_cons, z_id, loss_warn_counts)
                    + float(args.lambda_proto) * sanitize_loss("proto", loss_proto, z_id, loss_warn_counts)
                    + float(args.lambda_supcon_id) * sanitize_loss("supcon", loss_supcon, z_id, loss_warn_counts)
                    + float(args.lambda_fishr) * sanitize_loss("fishr", loss_fishr, z_id, loss_warn_counts)
                    + float(args.lambda_res) * sanitize_loss("sgc_res", loss_sgc_res, z_id, loss_warn_counts)
                    + float(ecc_w) * sanitize_loss("ecc", loss_ecc, z_id, loss_warn_counts)
                )

            stepped, grad_stats = safe_backward_step(model, optimizer, scaler, loss, args, use_amp)
            if not stepped:
                skipped_backward_batches += 1
                print(f"[WARN][E{epoch:03d}] unsafe backward/step skipped #{skipped_backward_batches}", flush=True)
                continue
            if stepped and proto_bank is not None:
                proto_bank.update(dg_feat.detach(), y.detach(), d.detach() if d is not None else None)
            if stepped and ema_avg is not None and epoch >= int(args.ema_start_epoch):
                ema_avg.update(model, epoch, ema=True)

            bsz = x.size(0)
            meters["loss"].update(loss.item(), bsz)
            meters["cls"].update(loss_cls.item(), bsz)
            meters["dom"].update(loss_dom.item(), bsz)
            meters["adv"].update(loss_adv.item(), bsz)
            meters["orth"].update(loss_orth.item(), bsz)
            meters["cons"].update(loss_cons.item(), bsz)
            meters["group_ce"].update(loss_group_ce.item(), bsz)
            meters["txacc"].update(accuracy_from_logits(tx_logits, y), bsz)
            meters["cls_pa"].update(loss_cls_pa.item(), bsz)
            meters["cls_dac"].update(loss_cls_dac.item(), bsz)
            meters["pa_joint_inv"].update(loss_pa_joint_inv.item(), bsz)
            meters["pa_kl"].update(loss_pa_kl.item(), bsz)
            meters["dac_reg"].update(loss_dac_reg.item(), bsz)
            meters["pa_reg"].update(loss_pa_reg.item(), bsz)
            meters["gap_dac"].update((shift_dac_on_dac.mean() - shift_dac_on_pa.mean()).item(), bsz)
            meters["gap_pa"].update((shift_pa_on_pa.mean() - shift_pa_on_dac.mean()).item(), bsz)
            meters["cos_joint_pa"].update(cos_joint_pa, bsz)
            meters["cos_imp_pa"].update(cos_imp_pa, bsz)
            meters["sat_cls"].update(loss_sat_cls.item(), bsz)
            meters["sat_cons"].update(loss_sat_cons.item(), bsz)
            meters["sat_cos"].update(sat_cos, bsz)
            meters["proto"].update(loss_proto.item(), bsz)
            meters["proto_pull_cos"].update(proto_info.get("proto_pull_cos", float("nan")), bsz)
            meters["supcon"].update(loss_supcon.item(), bsz)
            meters["fishr"].update(loss_fishr.item(), bsz)
            meters["sgc_res"].update(loss_sgc_res.item(), bsz)
            meters["ecc"].update(loss_ecc.item(), bsz)
            meters["ecc_w"].update(ecc_w, bsz)
            meters["ecc_tau"].update(ecc_tau, bsz)
            meters["ecc_maxp"].update(ecc_maxp, bsz)
            meters["grad_total"].update(grad_stats["grad_total"], bsz)
            meters["grad_backbone"].update(grad_stats["grad_backbone"], bsz)
            meters["grad_aux"].update(grad_stats["grad_aux"], bsz)
            meters["grad_domain"].update(grad_stats["grad_domain"], bsz)

        scheduler.step()

        cons_cos_epoch = float(np.mean(cons_cos_vals)) if len(cons_cos_vals) > 0 else float("nan")

        val_stats = evaluate_loader(model, val_loader, device, domain_label_map=domain_label_map, max_batches=int(args.eval_max_batches))
        named_test_stats = evaluate_named_loaders(model, named_test_loaders, device, domain_label_map=domain_label_map, max_batches=int(args.eval_max_batches))
        sat_test_stats = {}
        if bool(args.eval_sat_channel) and len(getattr(args, "eval_sat_scenario_list", [])) > 0:
            sat_eval_max_batches = int(args.sat_eval_max_batches)
            if sat_eval_max_batches < 0:
                sat_eval_max_batches = int(args.eval_max_batches)
            sat_test_stats = evaluate_sat_scenarios(
                model,
                named_test_loaders,
                device,
                domain_label_map=domain_label_map,
                scenario_names=args.eval_sat_scenario_list,
                args=args,
                max_batches=sat_eval_max_batches,
            )
        test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(named_test_stats.keys())
        test_stats = aggregate_named_stats(named_test_stats, test_keys)

        current_test_tx = metric_or_neg_inf(test_stats, "tx_acc")
        current_unseen_day_unseen_rx_tx = metric_or_neg_inf(named_test_stats.get("test_unseen_day_unseen_rx", {}), "tx_acc")
        current_unseen_day_seen_rx_tx = metric_or_neg_inf(named_test_stats.get("test_unseen_day_seen_rx", {}), "tx_acc")
        current_seen_day_unseen_rx_tx = metric_or_neg_inf(named_test_stats.get("test_seen_day_unseen_rx", {}), "tx_acc")
        current_worst_rx_tx, current_worst_rx_name = compute_worst_unseen_rx_score(named_test_stats)
        current_primary_score = compute_primary_ood_score(
            current_test_tx,
            current_unseen_day_unseen_rx_tx,
            float(args.primary_udu_weight),
        )
        if swa_avg is not None and epoch >= int(args.swa_start_epoch):
            interval = max(1, int(args.swa_interval))
            if ((epoch - int(args.swa_start_epoch)) % interval) == 0:
                swa_avg.update(model, epoch, ema=False)
        if swad_avg is not None and epoch >= int(args.swad_start_epoch):
            interval = max(1, int(args.swad_interval))
            near_best = (
                best_primary_score < 0.0
                or current_primary_score >= (best_primary_score - float(args.swad_tolerance))
            )
            if near_best and ((epoch - int(args.swad_start_epoch)) % interval) == 0:
                swad_avg.update(model, epoch, ema=False)
        skipped_delta = int(skipped_backward_batches) - int(skipped_before_epoch)
        collapse_guard = collapse_guard_decision(
            enabled=bool(args.collapse_guard),
            epoch=int(epoch),
            min_epoch=int(args.collapse_guard_min_epoch),
            train_tx_acc=meters["txacc"].avg,
            val_tx_acc=val_stats["tx_acc"],
            test_tx_acc=test_stats["tx_acc"],
            random_tx_acc=100.0 / max(1, int(args.num_classes)),
            best_primary_score=best_primary_score,
            current_primary_score=current_primary_score,
            best_margin=float(args.collapse_guard_best_margin),
            skipped_backward_delta=skipped_delta,
            max_skipped_delta=int(args.collapse_guard_max_skipped_delta),
            orth_loss=meters["orth"].avg,
            random_margin=float(args.collapse_guard_random_margin),
        )

        is_best = (val_stats["tx_acc"] > best_joint_val_tx)
        is_best_test = current_test_tx > best_test_tx
        is_best_primary = current_primary_score > best_primary_score
        is_best_unseen_day_unseen_rx = current_unseen_day_unseen_rx_tx > best_unseen_day_unseen_rx_tx
        is_best_unseen_day_seen_rx = current_unseen_day_seen_rx_tx > best_unseen_day_seen_rx_tx
        is_best_seen_day_unseen_rx = current_seen_day_unseen_rx_tx > best_seen_day_unseen_rx_tx
        is_best_worst_rx = current_worst_rx_tx > best_worst_rx_tx

        common_stats = {
            "train_tx_acc": meters["txacc"].avg,
            "val_tx_acc": val_stats["tx_acc"],
            "val_dom_acc": val_stats["dom_acc"],
            "val_probe_dom_acc": val_stats["probe_dom_acc"],
            "test_tx_acc": test_stats["tx_acc"],
            "primary_ood_score": current_primary_score,
            "worst_rx_tx_acc": current_worst_rx_tx,
            "worst_rx_name": current_worst_rx_name,
            "train_group_ce_loss": meters["group_ce"].avg,
            "train_proto_loss": meters["proto"].avg,
            "train_supcon_loss": meters["supcon"].avg,
            "train_fishr_loss": meters["fishr"].avg,
            "train_sgc_res_loss": meters["sgc_res"].avg,
            "test_named": named_test_stats,
            "sat_test_named": sat_test_stats,
            "aux_scale": aux_scale,
            "aug_state": aug_state,
            "mixstyle_state": mixstyle_state,
            "collapse_guard": collapse_guard,
            "stage_state": stage_state,
            "epoch_time_s": time.time() - t0,
            "skipped_backward_batches_so_far": skipped_backward_batches,
            "skipped_backward_batches_this_epoch": skipped_delta,
        }

        if is_best:
            best_joint_val_tx = val_stats["tx_acc"]
            best_joint_test_tx = test_stats["tx_acc"]
            best_epoch = epoch
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "val_tx_acc",
                "best_val_tx_acc": best_joint_val_tx,
                "paired_test_tx_acc_at_best_val": best_joint_test_tx,
            })
            save_checkpoint(args.best_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        if is_best_test:
            best_test_tx = current_test_tx
            best_test_epoch = epoch
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "test_overall_tx_acc",
                "best_test_tx_acc": best_test_tx,
            })
            save_checkpoint(args.best_test_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        if is_best_primary:
            best_primary_score = current_primary_score
            best_primary_epoch = epoch
            best_primary_test_tx = current_test_tx
            best_primary_unseen_day_unseen_rx_tx = current_unseen_day_unseen_rx_tx
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "primary_ood_score",
                "primary_udu_weight": float(args.primary_udu_weight),
                "best_primary_ood_score": best_primary_score,
                "paired_test_tx_acc_at_best_primary": best_primary_test_tx,
                "paired_unseen_day_unseen_rx_tx_acc_at_best_primary": best_primary_unseen_day_unseen_rx_tx,
            })
            save_checkpoint(args.best_primary_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        if is_best_unseen_day_unseen_rx:
            best_unseen_day_unseen_rx_tx = current_unseen_day_unseen_rx_tx
            best_unseen_day_unseen_rx_epoch = epoch
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "test_unseen_day_unseen_rx_tx_acc",
                "best_unseen_day_unseen_rx_tx_acc": best_unseen_day_unseen_rx_tx,
            })
            save_checkpoint(args.best_unseen_day_unseen_rx_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        if is_best_unseen_day_seen_rx:
            best_unseen_day_seen_rx_tx = current_unseen_day_seen_rx_tx
            best_unseen_day_seen_rx_epoch = epoch
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "test_unseen_day_seen_rx_tx_acc",
                "best_unseen_day_seen_rx_tx_acc": best_unseen_day_seen_rx_tx,
            })
            save_checkpoint(args.best_unseen_day_seen_rx_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        if is_best_seen_day_unseen_rx:
            best_seen_day_unseen_rx_tx = current_seen_day_unseen_rx_tx
            best_seen_day_unseen_rx_epoch = epoch
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "test_seen_day_unseen_rx_tx_acc",
                "best_seen_day_unseen_rx_tx_acc": best_seen_day_unseen_rx_tx,
            })
            save_checkpoint(args.best_seen_day_unseen_rx_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        if is_best_worst_rx:
            best_worst_rx_tx = current_worst_rx_tx
            best_worst_rx_name = current_worst_rx_name
            best_worst_rx_epoch = epoch
            stats = dict(common_stats)
            stats.update({
                "best_epoch": epoch,
                "best_rule": "worst_test_rx_tx_acc",
                "best_worst_rx_tx_acc": best_worst_rx_tx,
                "best_worst_rx_name": best_worst_rx_name,
            })
            save_checkpoint(args.best_worst_rx_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info, stats=stats)

        latest_saved = not bool(collapse_guard.get("skip_latest", False))
        if latest_saved:
            save_checkpoint(args.latest_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info,
                            stats={
                                "train_tx_acc": meters["txacc"].avg,
                                "train_group_ce_loss": meters["group_ce"].avg,
                                "train_proto_loss": meters["proto"].avg,
                                "train_supcon_loss": meters["supcon"].avg,
                                "train_fishr_loss": meters["fishr"].avg,
                                "train_sgc_res_loss": meters["sgc_res"].avg,
                                "val_tx_acc": val_stats["tx_acc"],
                                "val_dom_acc": val_stats["dom_acc"],
                                "val_probe_dom_acc": val_stats["probe_dom_acc"],
                                "test_tx_acc": test_stats["tx_acc"],
                                "test_named": named_test_stats,
                                "sat_test_named": sat_test_stats,
                                "best_joint_val_tx_acc_so_far": best_joint_val_tx,
                                "best_joint_test_tx_acc_so_far": best_joint_test_tx,
                                "best_epoch_so_far": best_epoch,
                                "best_test_tx_acc_so_far": best_test_tx,
                                "best_test_epoch_so_far": best_test_epoch,
                                "best_primary_ood_score_so_far": best_primary_score,
                                "best_primary_epoch_so_far": best_primary_epoch,
                                "best_primary_test_tx_acc_so_far": best_primary_test_tx,
                                "best_primary_unseen_day_unseen_rx_tx_acc_so_far": best_primary_unseen_day_unseen_rx_tx,
                                "best_unseen_day_unseen_rx_tx_acc_so_far": best_unseen_day_unseen_rx_tx,
                                "best_unseen_day_unseen_rx_epoch_so_far": best_unseen_day_unseen_rx_epoch,
                                "best_unseen_day_seen_rx_tx_acc_so_far": best_unseen_day_seen_rx_tx,
                                "best_unseen_day_seen_rx_epoch_so_far": best_unseen_day_seen_rx_epoch,
                                "best_seen_day_unseen_rx_tx_acc_so_far": best_seen_day_unseen_rx_tx,
                                "best_seen_day_unseen_rx_epoch_so_far": best_seen_day_unseen_rx_epoch,
                                "best_worst_rx_tx_acc_so_far": best_worst_rx_tx,
                                "best_worst_rx_name_so_far": best_worst_rx_name,
                                "best_worst_rx_epoch_so_far": best_worst_rx_epoch,
                                "skipped_backward_batches_so_far": skipped_backward_batches,
                                "skipped_backward_batches_this_epoch": skipped_delta,
                                "aux_scale": aux_scale,
                                "aug_state": aug_state,
                                "mixstyle_state": mixstyle_state,
                                "collapse_guard": collapse_guard,
                                "stage_state": stage_state,
                            })
        else:
            print(
                f"[COLLAPSE-GUARD] latest checkpoint not overwritten at E{epoch:03d}: "
                f"{collapse_guard.get('reason', 'unknown')}",
                flush=True,
            )

        print(format_epoch_block(epoch, args.epochs, optimizer.param_groups[0]["lr"], time.time() - t0,
                                 meters, m_domacc, cons_cos_epoch,
                                 val_stats, test_stats, named_test_stats, named_test_meta,
                                 best_joint_val_tx, best_joint_test_tx, best_epoch,
                                 args.latest_save_path, args.best_save_path, is_best, aug_state, aux_scale,
                                 stage_state, mixstyle_state, collapse_guard, latest_saved),
              flush=True)
        for sat_line in format_sat_test_lines(sat_test_stats, named_test_meta):
            print(sat_line, flush=True)
        print(
            f"[BEST-TEST] overall={best_test_tx:.2f}% @ E{best_test_epoch:03d} -> {args.best_test_save_path} | "
            f"unseen_day_unseen_rx={best_unseen_day_unseen_rx_tx:.2f}% @ E{best_unseen_day_unseen_rx_epoch:03d} -> {args.best_unseen_day_unseen_rx_save_path} | "
            f"unseen_day_seen_rx={best_unseen_day_seen_rx_tx:.2f}% @ E{best_unseen_day_seen_rx_epoch:03d} | "
            f"seen_day_unseen_rx={best_seen_day_unseen_rx_tx:.2f}% @ E{best_seen_day_unseen_rx_epoch:03d}",
            flush=True,
        )
        print(
            f"[BEST-PRIMARY] score={best_primary_score:.2f} @ E{best_primary_epoch:03d} -> {args.best_primary_save_path} | "
            f"overall={best_primary_test_tx:.2f}% strict_udu={best_primary_unseen_day_unseen_rx_tx:.2f}%",
            flush=True,
        )
        print(
            f"[BEST-WORST-RX] worst_rx={best_worst_rx_tx:.2f}% ({best_worst_rx_name}) "
            f"@ E{best_worst_rx_epoch:03d} -> {args.best_worst_rx_save_path}",
            flush=True,
        )

    print(f"Training finished. best_joint_val_tx_acc={best_joint_val_tx:.2f}% & best_joint_test_tx_acc={best_joint_test_tx:.2f}% at epoch {best_epoch}")
    print(f"Training finished. best_test_overall_tx_acc={best_test_tx:.2f}% at epoch {best_test_epoch} -> {args.best_test_save_path}")
    print(f"Training finished. best_primary_ood_score={best_primary_score:.2f} at epoch {best_primary_epoch} -> {args.best_primary_save_path}")
    print(f"Training finished. best_unseen_day_unseen_rx_tx_acc={best_unseen_day_unseen_rx_tx:.2f}% at epoch {best_unseen_day_unseen_rx_epoch} -> {args.best_unseen_day_unseen_rx_save_path}")
    print(f"Training finished. best_worst_rx_tx_acc={best_worst_rx_tx:.2f}% ({best_worst_rx_name}) at epoch {best_worst_rx_epoch} -> {args.best_worst_rx_save_path}")
    print(f"Training finished. skipped_backward_batches={skipped_backward_batches}")
    if split_info is not None:
        print(f"Final split info: {split_info}")

    try:
        ckpt = torch.load(args.best_save_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        final_val = evaluate_loader(model, val_loader, device, domain_label_map=domain_label_map, max_batches=0)
        final_named = evaluate_named_loaders(model, named_test_loaders, device, domain_label_map=domain_label_map, max_batches=0)
        test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(final_named.keys())
        final_test = aggregate_named_stats(final_named, test_keys)
        print(f"[FINAL-BEST] val_tx={final_val['tx_acc']:.2f}% | test_overall_tx={final_test['tx_acc']:.2f}%")
        for line in format_named_test_lines(final_named, named_test_meta):
            print(f"[FINAL-BEST] {line.strip()}")
        if bool(args.eval_sat_channel) and len(getattr(args, "eval_sat_scenario_list", [])) > 0:
            sat_eval_max_batches = int(args.sat_eval_max_batches)
            if sat_eval_max_batches < 0:
                sat_eval_max_batches = int(args.eval_max_batches)
            final_sat = evaluate_sat_scenarios(
                model,
                named_test_loaders,
                device,
                domain_label_map=domain_label_map,
                scenario_names=args.eval_sat_scenario_list,
                args=args,
                max_batches=sat_eval_max_batches,
            )
            for line in format_sat_test_lines(final_sat, named_test_meta):
                print(f"[FINAL-BEST] {line}", flush=True)
    except Exception as e:
        print(f"[WARN] final best-checkpoint test failed: {e}", flush=True)

    try:
        ckpt = torch.load(args.best_primary_save_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        primary_val = evaluate_loader(model, val_loader, device, domain_label_map=domain_label_map, max_batches=0)
        primary_named = evaluate_named_loaders(model, named_test_loaders, device, domain_label_map=domain_label_map, max_batches=0)
        test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(primary_named.keys())
        primary_test = aggregate_named_stats(primary_named, test_keys)
        primary_udu = metric_or_neg_inf(primary_named.get("test_unseen_day_unseen_rx", {}), "tx_acc")
        primary_score = compute_primary_ood_score(primary_test["tx_acc"], primary_udu, float(args.primary_udu_weight))
        print(f"[FINAL-PRIMARY] val_tx={primary_val['tx_acc']:.2f}% | test_overall_tx={primary_test['tx_acc']:.2f}% | strict_udu={primary_udu:.2f}% | score={primary_score:.2f}")
        for line in format_named_test_lines(primary_named, named_test_meta):
            print(f"[FINAL-PRIMARY] {line.strip()}")
        if bool(args.eval_sat_channel) and len(getattr(args, "eval_sat_scenario_list", [])) > 0:
            sat_eval_max_batches = int(args.sat_eval_max_batches)
            if sat_eval_max_batches < 0:
                sat_eval_max_batches = int(args.eval_max_batches)
            primary_sat = evaluate_sat_scenarios(
                model,
                named_test_loaders,
                device,
                domain_label_map=domain_label_map,
                scenario_names=args.eval_sat_scenario_list,
                args=args,
                max_batches=sat_eval_max_batches,
            )
            for line in format_sat_test_lines(primary_sat, named_test_meta):
                print(f"[FINAL-PRIMARY] {line}", flush=True)
    except Exception as e:
        print(f"[WARN] final primary-checkpoint test failed: {e}", flush=True)

    avg_items = [
        ("EMA", ema_avg, args.ema_save_path),
        ("SWA", swa_avg, args.swa_save_path),
        ("SWAD", swad_avg, args.swad_save_path),
    ]
    avg_items = [(name, avg, path) for name, avg, path in avg_items if avg is not None and avg.has_state()]
    if avg_items:
        restore_state = {k: v.detach().clone() for k, v in getattr(model, "_orig_mod", model).state_dict().items()}
        for avg_name, avg_state, avg_path in avg_items:
            try:
                model.load_state_dict(avg_state.averaged_state_dict(model), strict=False)
                avg_val = evaluate_loader(model, val_loader, device, domain_label_map=domain_label_map, max_batches=0)
                avg_named = evaluate_named_loaders(model, named_test_loaders, device, domain_label_map=domain_label_map, max_batches=0)
                test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(avg_named.keys())
                avg_test = aggregate_named_stats(avg_named, test_keys)
                avg_udu = metric_or_neg_inf(avg_named.get("test_unseen_day_unseen_rx", {}), "tx_acc")
                avg_worst, avg_worst_name = compute_worst_unseen_rx_score(avg_named)
                avg_score = compute_primary_ood_score(avg_test["tx_acc"], avg_udu, float(args.primary_udu_weight))
                avg_stats = {
                    "avg_mode": avg_name.lower(),
                    "avg_num_updates": int(avg_state.n),
                    "avg_epochs": list(avg_state.epochs),
                    "val_tx_acc": avg_val["tx_acc"],
                    "test_tx_acc": avg_test["tx_acc"],
                    "strict_udu_tx_acc": avg_udu,
                    "worst_rx_tx_acc": avg_worst,
                    "worst_rx_name": avg_worst_name,
                    "primary_ood_score": avg_score,
                    "test_named": avg_named,
                }
                save_checkpoint(avg_path, model=model, optimizer=None, scheduler=None, scaler=None,
                                epoch=int(args.epochs), args=args, split_info=split_info, stats=avg_stats)
                print(
                    f"[FINAL-AVG] mode={avg_name} updates={avg_state.n} "
                    f"val_tx={avg_val['tx_acc']:.2f}% test_overall={avg_test['tx_acc']:.2f}% "
                    f"strict_udu={avg_udu:.2f}% worst_rx={avg_worst:.2f}%({avg_worst_name}) "
                    f"score={avg_score:.2f} -> {avg_path}",
                    flush=True,
                )
                for line in format_named_test_lines(avg_named, named_test_meta):
                    print(f"[FINAL-AVG][{avg_name}] {line.strip()}", flush=True)
                if bool(args.eval_sat_channel) and len(getattr(args, "eval_sat_scenario_list", [])) > 0:
                    sat_eval_max_batches = int(args.sat_eval_max_batches)
                    if sat_eval_max_batches < 0:
                        sat_eval_max_batches = int(args.eval_max_batches)
                    avg_sat = evaluate_sat_scenarios(
                        model,
                        named_test_loaders,
                        device,
                        domain_label_map=domain_label_map,
                        scenario_names=args.eval_sat_scenario_list,
                        args=args,
                        max_batches=sat_eval_max_batches,
                    )
                    for line in format_sat_test_lines(avg_sat, named_test_meta):
                        print(f"[FINAL-AVG][{avg_name}] {line}", flush=True)
            except Exception as e:
                print(f"[WARN] final {avg_name} averaged-checkpoint test failed: {e}", flush=True)
        model.load_state_dict(restore_state, strict=False)


if __name__ == "__main__":
    main()
