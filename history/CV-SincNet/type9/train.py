# train.py
import os
import math
import time
import copy
import inspect
import argparse
from dataclasses import dataclass
from datetime import datetime
import random

import builtins
from functools import partial
print = partial(builtins.print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset

# Optional: satellite channel simulator (torch batch)
try:
    from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
except Exception:
    SatSimConfig = None
    apply_sat_gnd_channel_batch = None

# Optional: receiver-chain domain randomization (from original DataAugmentation_v2)
try:
    from DataAugmentation_v2 import apply_receiver_dg as _apply_receiver_dg
except Exception:
    _apply_receiver_dg = None


# Optional: WiSig (UCLA) loader
try:
    from dataset_wisig import load_wisig_compact_pkl, make_leave_one_day_out_split
except Exception:
    load_wisig_compact_pkl = None
    make_leave_one_day_out_split = None


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 1337):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits: [B, C]
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item() * 100.0


def safe_to_float(x: torch.Tensor) -> torch.Tensor:
    # avoid fp16 overflow inside CE / log-softmax
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def unpack_batch(batch):
    """Unpack dataloader batch.

    Supports datasets that return:
      - (x, y)
      - (x, y, d)
      - (x, y, d, meta)
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(f"Batch has <2 elements: {len(batch)}")
        x = batch[0]
        y = batch[1]
        extra = batch[2:] if len(batch) > 2 else ()
        return x, y, extra
    raise TypeError(f"Unsupported batch type: {type(batch)}")




def _call_model(model, x, return_aux: bool = False):
    # Try to call model with a return_aux keyword; if the model does not support it, fall back.
    try:
        return model(x, return_aux=return_aux)
    except TypeError:
        return model(x)


def _unpack_model_output(out):
    # Unpack common model output conventions.
    # Returns: logits, dac_pred, pa_pred, feat_cls, feat_con, cos_logits
    logits = dac_pred = pa_pred = feat_cls = feat_con = cos_logits = None

    if isinstance(out, dict):
        for k in ("logits", "pred", "y_pred", "out"):
            if k in out:
                logits = out[k]
                break
        dac_pred = out.get("dac_pred", out.get("dac", None))
        pa_pred = out.get("pa_pred", out.get("pa", None))
        feat_cls = out.get("feat_cls", out.get("feat", None))
        feat_con = out.get("feat_con", out.get("proj", None))
        cos_logits = out.get("cos_logits", out.get("cos", None))

    elif isinstance(out, (tuple, list)):
        if len(out) >= 1:
            logits = out[0]
        if len(out) >= 2:
            dac_pred = out[1]
        if len(out) >= 3:
            pa_pred = out[2]
        if len(out) >= 4:
            feat_cls = out[3]
        if len(out) >= 5:
            feat_con = out[4]
        # Some models may return (logits, feat) only.
        if feat_con is None and feat_cls is None and pa_pred is None and len(out) == 2:
            feat_con = out[1]

    else:
        # torch.Tensor logits only
        try:
            import torch
            if torch.is_tensor(out):
                logits = out
            else:
                raise TypeError
        except Exception as e:
            raise RuntimeError(f"Unsupported model output type: {type(out)}") from e

    return logits, dac_pred, pa_pred, feat_cls, feat_con, cos_logits
def linear_warmup(epoch_idx: int, warmup_epochs: int) -> float:
    """Return a 0..1 warmup factor for 1-based epoch index."""
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch_idx / max(1, warmup_epochs))))


def smooth_gate(epoch_idx: int, start_epoch: int, transition_epochs: int) -> float:
    """Gate turns on at start_epoch with optional linear transition."""
    if epoch_idx < start_epoch:
        return 0.0
    if transition_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch_idx - start_epoch + 1) / max(1, transition_epochs)))


def compute_stage_gates(
    epoch_idx: int,
    stage1_end: int,
    stage2_end: int,
    stage3_end: int,
    transition_epochs: int = 0,
):
    """
    Stage 1: CE only
    Stage 2: CE + Con
    Stage 3: CE + Con + Proto
    Stage 4: CE + Con + Proto + DAC/PA/DA
    """
    g_con = smooth_gate(epoch_idx, stage1_end + 1, transition_epochs)
    g_proto = smooth_gate(epoch_idx, stage2_end + 1, transition_epochs)
    g_dac = smooth_gate(epoch_idx, stage3_end + 1, transition_epochs)
    if epoch_idx <= stage1_end:
        stage_id = 1
    elif epoch_idx <= stage2_end:
        stage_id = 2
    elif epoch_idx <= stage3_end:
        stage_id = 3
    else:
        stage_id = 4
    return stage_id, g_con, g_proto, g_dac


def compute_tau(
    epoch_idx: int,
    tau_start: float,
    tau_end: float,
    decay_start: int,
    decay_end: int,
) -> float:
    """Cosine temperature decay from tau_start -> tau_end."""
    t0 = max(1, int(decay_start))
    t1 = max(t0, int(decay_end))
    if epoch_idx <= t0:
        return float(tau_start)
    if epoch_idx >= t1:
        return float(tau_end)
    p = (epoch_idx - t0) / max(1, (t1 - t0))
    p = min(1.0, max(0.0, p))
    # cosine interpolation: 1 -> 0
    c = 0.5 * (1.0 + math.cos(math.pi * p))
    return float(tau_end + (tau_start - tau_end) * c)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    eta_min: float = 1e-6,
    warmup_start_factor: float = 0.1,
):
    warmup_epochs = int(max(0, warmup_epochs))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(1e-4, min(1.0, float(warmup_start_factor))),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs) - warmup_epochs),
            eta_min=float(eta_min),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(total_epochs)),
        eta_min=float(eta_min),
    )


def shrink_scheduler_base_lrs(scheduler, factor: float, lr_min: float = 0.0):
    """Recursively shrink scheduler base_lrs after rollback."""
    if scheduler is None:
        return
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [max(float(lr_min), float(v) * float(factor)) for v in scheduler.base_lrs]
    subs = getattr(scheduler, "_schedulers", None)
    if isinstance(subs, (list, tuple)):
        for s in subs:
            shrink_scheduler_base_lrs(s, factor=factor, lr_min=lr_min)


# -----------------------
# Prototype bank (no dependency on contrastive_loss.py)
# -----------------------
class PrototypeBank:
    """
    momentum prototype bank: prototypes[c] = m*proto + (1-m)*mean(feats_of_class_c)
    also maintains counts to mark valid classes
    """
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float = 0.9,
        warmup_momentum: float = 0.99,
        warmup_updates: int = 20,
        min_batch_samples: int = 2,
        min_count_valid: int = 8,
        device="cuda",
    ):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = float(momentum)
        self.warmup_momentum = float(warmup_momentum)
        self.warmup_updates = int(max(0, warmup_updates))
        self.min_batch_samples = int(max(1, min_batch_samples))
        self.min_count_valid = int(max(1, min_count_valid))
        self.device = device

        self.prototypes = torch.zeros(num_classes, feat_dim, device=device, dtype=torch.float32)
        self.counts = torch.zeros(num_classes, device=device, dtype=torch.long)
        self.update_steps = torch.zeros(num_classes, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: [B, D] (already normalized or not)
        labels: [B]
        """
        feats = feats.detach().float()
        labels = labels.detach()
        for c in labels.unique():
            c = int(c.item())
            idx = (labels == c)
            n = int(idx.sum().item())
            if n < self.min_batch_samples:
                continue
            mean_feat = feats[idx].mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=0)
            if self.counts[c] == 0:
                self.prototypes[c] = mean_feat
            else:
                m_eff = self.momentum
                if self.update_steps[c] < self.warmup_updates:
                    m_eff = self.warmup_momentum
                self.prototypes[c] = F.normalize(
                    m_eff * self.prototypes[c] + (1.0 - m_eff) * mean_feat,
                    dim=0
                )
            self.counts[c] += n
            self.update_steps[c] += 1

    def valid_mask(self, min_count: int = None) -> torch.Tensor:
        mc = self.min_count_valid if min_count is None else int(max(1, min_count))
        return self.counts >= mc


def proto_ce_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    bank: PrototypeBank,
    tau: float = 0.07,
    min_count: int = 8,
) -> torch.Tensor:
    """
    feats: [B, D] normalized
    prototypes: [C, D] normalized
    use CE over prototypes (safe fp16: compute in fp32, mask invalid with -1e4)
    """
    feats = F.normalize(feats.float(), dim=1)
    prot = F.normalize(bank.prototypes.float(), dim=1)

    logits = (feats @ prot.t()) / max(1e-6, tau)  # [B, C] float32
    valid = bank.valid_mask(min_count=min_count)  # [C]
    if valid.sum().item() == 0:
        return logits.new_tensor(0.0)

    # mask invalid prototypes with fp16-safe negative
    neg = -1e4
    logits[:, ~valid] = neg

    # if a sample's class prototype invalid, ignore it
    tgt = labels.clone()
    invalid_tgt = ~valid[tgt]
    if invalid_tgt.any():
        tgt[invalid_tgt] = -100  # ignore_index

    loss = F.cross_entropy(logits, tgt, ignore_index=-100)
    return loss


# -----------------------
# SupCon (safe fp16)
# -----------------------
def supcon_loss(feats: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    feats: [B, V, D]  (float/half ok)
    labels: [B]
    """
    B, V, D = feats.shape
    x = F.normalize(feats.float(), dim=2)  # [B,V,D] float32

    # flatten views as anchors
    x = x.reshape(B * V, D)  # [BV, D]
    y = labels.repeat_interleave(V)  # [BV]

    # similarity
    logits = (x @ x.t()) / max(1e-6, tau)  # [BV, BV]
    # mask self
    diag = torch.eye(B * V, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag, -1e4)

    # positives mask: same class, not self
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~diag)  # [BV, BV]

    # log-softmax
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # [BV,BV]

    # mean over positives
    pos_cnt = pos.sum(dim=1)  # [BV]
    # avoid div0
    loss = -(log_prob * pos.float()).sum(dim=1) / pos_cnt.clamp_min(1.0)
    # only anchors with >=1 positive contribute
    loss = loss[pos_cnt > 0].mean() if (pos_cnt > 0).any() else logits.new_tensor(0.0)
    return loss



# -----------------------
# Domain adaptation (CORAL / MMD / DANN)
# -----------------------
def _covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    n = x.size(0)
    if n <= 1:
        return x.new_zeros((x.size(1), x.size(1)))
    return (x.t() @ x) / (n - 1)

def coral_multi_domain(feat: torch.Tensor, dom: torch.Tensor) -> torch.Tensor:
    """Multi-domain CORAL: align each domain's mean/cov to global mean/cov within the batch."""
    feat = feat.float()
    dom = dom.long().view(-1)
    uniq = torch.unique(dom)
    if uniq.numel() < 2:
        return feat.new_tensor(0.0)
    mu_g = feat.mean(dim=0)
    cov_g = _covariance(feat)
    loss = feat.new_tensor(0.0)
    for d in uniq:
        m = (dom == d)
        if m.sum() < 2:
            continue
        f = feat[m]
        mu = f.mean(dim=0)
        cov = _covariance(f)
        loss = loss + F.mse_loss(mu, mu_g) + F.mse_loss(cov, cov_g)
    return loss / uniq.numel()

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    dist2 = x2 + y2 - 2.0 * (x @ y.t())
    return torch.exp(-gamma * dist2.clamp_min(0.0))

def mmd_rbf_multi_domain(feat: torch.Tensor, dom: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Align each domain to global using RBF-MMD (batch-based)."""
    feat = feat.float()
    dom = dom.long().view(-1)
    uniq = torch.unique(dom)
    if uniq.numel() < 2:
        return feat.new_tensor(0.0)
    f_all = feat
    K_aa = _rbf_kernel(f_all, f_all, gamma)
    mmd_sum = feat.new_tensor(0.0)
    for d in uniq:
        m = (dom == d)
        if m.sum() < 2:
            continue
        f_d = feat[m]
        K_dd = _rbf_kernel(f_d, f_d, gamma)
        K_da = _rbf_kernel(f_d, f_all, gamma)
        mmd = K_dd.mean() + K_aa.mean() - 2.0 * K_da.mean()
        mmd_sum = mmd_sum + mmd
    return mmd_sum / uniq.numel()

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_domains: int):
        super().__init__()
        hid = max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hid, num_domains),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------
# Domain-balanced batch sampler (for WiSig day/rx domains)
# -----------------------
class DomainBalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, domain_ids, batch_size: int, drop_last: bool = True, seed: int = 1337):
        self.domain_ids = [int(d) for d in domain_ids]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(seed)

        self.dom2idx = {}
        for i, d in enumerate(self.domain_ids):
            self.dom2idx.setdefault(d, []).append(i)
        self.domains = sorted(self.dom2idx.keys())
        self.k = len(self.domains)
        if self.k <= 0:
            raise ValueError("No domains found for DomainBalancedBatchSampler.")

        self._p = {d: 0 for d in self.domains}
        self._lists = {}
        for d in self.domains:
            lst = self.dom2idx[d][:]
            self.rng.shuffle(lst)
            self._lists[d] = lst

        self.num_samples = len(self.domain_ids)
        self.num_batches = self.num_samples // self.batch_size if self.drop_last else math.ceil(self.num_samples / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        bs = self.batch_size
        k = self.k
        base = bs // k
        rem = bs - base * k
        for _ in range(self.num_batches):
            batch = []
            for di, d in enumerate(self.domains):
                take = base + (1 if di < rem else 0)
                if take <= 0:
                    continue
                lst = self._lists[d]
                p = self._p[d]
                if p + take > len(lst):
                    self.rng.shuffle(lst)
                    p = 0
                batch.extend(lst[p:p+take])
                p += take
                self._p[d] = p
            while len(batch) < bs:
                d = self.domains[self.rng.randrange(k)]
                lst = self._lists[d]
                batch.append(lst[self.rng.randrange(len(lst))])
            self.rng.shuffle(batch)
            yield batch


# -----------------------
# Augmentor loader (auto-detect)
# -----------------------
def build_augmentor_safely(device: str):
    try:
        import importlib
        try:
            mod = importlib.import_module("DataAugmentation_v2_sat")
            _aug_src = "DataAugmentation_v2_sat"
        except Exception:
            mod = importlib.import_module("DataAugmentation_v2")
            _aug_src = "DataAugmentation_v2"

        # 1) build_augmentor() function
        if hasattr(mod, "build_augmentor") and callable(mod.build_augmentor):
            aug = mod.build_augmentor()
            return aug, f"[OK] build_augmentor() ({_aug_src})"

        # 2) common class names
        preferred = ["DataAugmentationV2", "DataAugmentation", "Augmentor", "RFFIAugmentor"]
        for name in preferred:
            if hasattr(mod, name) and isinstance(getattr(mod, name), type):
                cls = getattr(mod, name)
                try:
                    aug = cls()
                except TypeError:
                    # try passing device if accepted
                    sig = inspect.signature(cls.__init__)
                    kwargs = {}
                    if "device" in sig.parameters:
                        kwargs["device"] = device
                    aug = cls(**kwargs)
                return aug, f"[OK] class {name} ({_aug_src})"

        # 3) fallback: find any class with __call__
        candidates = []
        for k, v in vars(mod).items():
            if isinstance(v, type) and hasattr(v, "__call__"):
                candidates.append(k)
        if len(candidates) == 0:
            raise RuntimeError("No augmentor class found in DataAugmentation_v2.py")

        # pick the most likely one (name contains 'Aug' first)
        candidates.sort(key=lambda s: (("aug" not in s.lower()), len(s)))
        cls = getattr(mod, candidates[0])
        try:
            aug = cls()
        except TypeError:
            sig = inspect.signature(cls.__init__)
            kwargs = {}
            if "device" in sig.parameters:
                kwargs["device"] = device
            aug = cls(**kwargs)
        return aug, f"[OK] fallback class {candidates[0]} ({_aug_src}) (candidates={candidates})"

    except Exception as e:
        return None, f"[WARN] Failed to import/construct DataAugmentation_v2 augmentor. Aug disabled. Reason: {e}"


def call_augmentor(
    augmentor,
    x_iq: torch.Tensor,
    labels=None,
    defect_apply_channel=None,
    return_dac_strength: bool = False,
    return_pa_strength: bool = False,
    return_defect_strengths: bool = False,
    dac_only: bool = False,
    pa_only: bool = False,
    dac_pa: bool = False,
    strength: float = 1.0,
    mix_mode: str = "prob",
):
    """
    Robust wrapper for different DataAugmentation_v2 implementations.

    - For NORMAL views: supports soft 'strength' knob (prob or blend).
    - For DEFECT-only views (dac_only/pa_only/dac_pa): always apply FULL augmentation
      (do NOT dilute by strength).
    - Compatible with augmentors that:
        * are callable, or expose methods like augment/apply/forward/transform
        * expose simulate_dac/simulate_pa/simulate_dac_pa
        * may or may not accept kwargs like return_*_strength
        * may return:
            - x_aug (Tensor)
            - (x_aug, s)
            - (x_aug, sd, sp)
            - dict with tensor fields
    """
    B = int(x_iq.size(0))

    def _zeros():
        return torch.zeros((B,), device=x_iq.device, dtype=torch.float32)

    # No augmentor: identity
    if augmentor is None:
        if return_defect_strengths:
            return x_iq, _zeros(), _zeros()
        if return_dac_strength or return_pa_strength:
            return x_iq, _zeros()
        return x_iq

    # defect views should NEVER be diluted
    is_defect_view = bool(dac_only or pa_only or dac_pa)

    # Clamp strength
    try:
        s = float(strength)
    except Exception:
        s = 1.0
    s = min(1.0, max(0.0, s))
    if is_defect_view:
        s = 1.0  # force full strength

    def _safe_call(fn, x, **kw):
        """Call fn(x, **kw) but filter unsupported kwargs; fallback to fn(x)."""
        try:
            return fn(x, **kw)
        except TypeError:
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                if not has_varkw:
                    kw2 = {k: v for k, v in kw.items() if k in params}
                else:
                    kw2 = kw
                return fn(x, **kw2)
            except Exception:
                try:
                    return fn(x)
                except Exception:
                    return None
        except Exception:
            return None

    def _resolve_aug_fn():
        """Resolve an augment function from augmentor object."""
        # callable object
        if callable(augmentor):
            return augmentor

        obj = augmentor
        # defect-specific methods
        if dac_only and hasattr(obj, "simulate_dac") and callable(getattr(obj, "simulate_dac")):
            return getattr(obj, "simulate_dac")
        if pa_only and hasattr(obj, "simulate_pa") and callable(getattr(obj, "simulate_pa")):
            return getattr(obj, "simulate_pa")
        if dac_pa and hasattr(obj, "simulate_dac_pa") and callable(getattr(obj, "simulate_dac_pa")):
            return getattr(obj, "simulate_dac_pa")

        # generic names
        for name in ("augment", "apply", "forward", "transform", "run", "process"):
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn

        # last resort: any callable attribute
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                fn = getattr(obj, name)
            except Exception:
                continue
            if callable(fn):
                return fn
        return None

    aug_fn = _resolve_aug_fn()
    if aug_fn is None:
        # cannot augment -> identity
        if return_defect_strengths:
            return x_iq, _zeros(), _zeros()
        if return_dac_strength or return_pa_strength:
            return x_iq, _zeros()
        return x_iq

    # ---- call augmentor ----
    kw = {
        "strength": s,
        "mix_mode": mix_mode,
        "labels": labels,
        "dac_only": dac_only,
        "pa_only": pa_only,
        "dac_pa": dac_pa,
    }
    if defect_apply_channel is not None:
        kw["defect_apply_channel"] = bool(defect_apply_channel)
    # Request strengths if supported (safe_call will drop if not)
    if return_defect_strengths:
        kw["return_defect_strengths"] = True
    if return_dac_strength:
        kw["return_dac_strength"] = True
    if return_pa_strength:
        kw["return_pa_strength"] = True

    out = _safe_call(aug_fn, x_iq, **kw)

    # ---- parse outputs ----
    def _extract_tensor(out_any):
        if torch.is_tensor(out_any):
            return out_any
        if isinstance(out_any, dict):
            for k in ("x_aug", "x", "signal", "out"):
                v = out_any.get(k, None)
                if torch.is_tensor(v):
                    return v
            for v in out_any.values():
                if torch.is_tensor(v):
                    return v
            return None
        if isinstance(out_any, (tuple, list)):
            return out_any[0] if len(out_any) > 0 and torch.is_tensor(out_any[0]) else None
        return None

    def _extract_strengths(out_any):
        # returns (sd, sp, s1)
        sd = sp = s1 = None
        if isinstance(out_any, dict):
            sd = out_any.get("dac_strength", None)
            sp = out_any.get("pa_strength", None)
            s1 = out_any.get("strength", None)
        elif isinstance(out_any, (tuple, list)):
            if len(out_any) == 3:
                sd, sp = out_any[1], out_any[2]
            elif len(out_any) == 2:
                s1 = out_any[1]
        return sd, sp, s1

    x_aug = _extract_tensor(out)
    sd, sp, s1 = _extract_strengths(out)

    # if no augmentation produced
    if x_aug is None:
        x_aug = x_iq
        sd = sp = None
        s1 = None

    # strength synthesis if requested but missing
    if return_defect_strengths:
        # prefer provided sd/sp, else fall back to s1, else use s
        if not torch.is_tensor(sd):
            sd = s1 if torch.is_tensor(s1) else torch.full((B,), float(s), device=x_iq.device, dtype=torch.float32)
        if not torch.is_tensor(sp):
            sp = s1 if torch.is_tensor(s1) else torch.full((B,), float(s), device=x_iq.device, dtype=torch.float32)
    elif return_dac_strength:
        if not torch.is_tensor(sd):
            sd = s1 if torch.is_tensor(s1) else torch.full((B,), float(s), device=x_iq.device, dtype=torch.float32)
    elif return_pa_strength:
        if not torch.is_tensor(sp):
            sp = s1 if torch.is_tensor(s1) else torch.full((B,), float(s), device=x_iq.device, dtype=torch.float32)

    # ---- mix for normal views only ----
    if (not is_defect_view) and s < 1.0:
        if mix_mode == "blend":
            x_out = x_iq * (1.0 - s) + x_aug * s
        else:
            mask = (torch.rand((B, 1, 1), device=x_iq.device) < s)
            x_out = torch.where(mask, x_aug, x_iq)
    else:
        x_out = x_aug

    if return_defect_strengths:
        return x_out, sd, sp
    if return_dac_strength:
        return x_out, sd
    if return_pa_strength:
        return x_out, sp
    return x_out


def build_model_safely(num_classes: int, device: str, model_size: str = "M", dataset: str = "oralce", input_len: int = 1024, sample_rate_hz: float = 5e6):
    import importlib
    mod = importlib.import_module("model")

    # common factory (prefer this when model.py provides build_model)
    if hasattr(mod, "build_model") and callable(mod.build_model):
        fn = mod.build_model
        sig = inspect.signature(fn)
        kw = {}
        if "num_classes" in sig.parameters:
            kw["num_classes"] = num_classes
        if "model_size" in sig.parameters:
            kw["model_size"] = model_size
        if "dataset" in sig.parameters:
            kw["dataset"] = dataset
        if "input_len" in sig.parameters:
            kw["input_len"] = input_len
        if "sample_rate_hz" in sig.parameters:
            kw["sample_rate_hz"] = sample_rate_hz
        if "sample_rate" in sig.parameters and "sample_rate_hz" not in kw:
            kw["sample_rate"] = sample_rate_hz
        m = fn(**kw)
        return m, f"[OK] model.build_model({kw})"

    # preferred class names
    preferred = [
        "DACSpecializedCVSincNet",
        "DAC_CV_SincNet",
        "CVSincNet",
        "SincNet",
        "Model",
        "Net",
    ]
    for name in preferred:
        if hasattr(mod, name) and isinstance(getattr(mod, name), type):
            cls = getattr(mod, name)
            if not issubclass(cls, nn.Module):
                continue
            sig = inspect.signature(cls.__init__)
            kwargs = {}
            if "num_classes" in sig.parameters:
                kwargs["num_classes"] = num_classes
            elif "n_classes" in sig.parameters:
                kwargs["n_classes"] = num_classes
            if "model_size" in sig.parameters:
                kwargs["model_size"] = model_size
            m = cls(**kwargs)
            return m, f"[OK] class {name}"

    # fallback: pick first nn.Module subclass with 'Net' or 'Model' in name
    cands = []
    for k, v in vars(mod).items():
        if isinstance(v, type) and issubclass(v, nn.Module):
            cands.append(k)
    if len(cands) == 0:
        raise RuntimeError("No nn.Module model class found in model.py")

    def score(n):
        n2 = n.lower()
        s = 0
        if "net" in n2: s -= 5
        if "model" in n2: s -= 4
        if "sinc" in n2: s -= 3
        if "dac" in n2: s -= 2
        return (s, len(n))

    cands.sort(key=score)
    cls = getattr(mod, cands[0])
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes
    elif "n_classes" in sig.parameters:
        kwargs["n_classes"] = num_classes
    if "model_size" in sig.parameters:
        kwargs["model_size"] = model_size
    m = cls(**kwargs)
    return m, f"[OK] fallback class {cands[0]} (candidates={cands})"


# -----------------------
# Checkpoint
# -----------------------
@dataclass
class BestState:
    best_acc: float = -1.0
    best_epoch: int = -1
    best_blob: dict = None  # model/optim/scaler etc


def pack_state(model, optimizer, scaler, epoch, lr_base, best_acc, best_epoch, global_step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "lr_base": lr_base,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "global_step": global_step,
    }


def unpack_state(blob, model, optimizer, scaler):
    model.load_state_dict(blob["model"], strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    if scaler is not None and blob.get("scaler", None) is not None:
        scaler.load_state_dict(blob["scaler"])
    return blob.get("epoch", 0), blob.get("lr_base", None), blob.get("global_step", 0)


# -----------------------
# Main
# -----------------------
class AdaptiveAugController:
    """Reduce augmentation strength if validation/test performance drops repeatedly."""

    def __init__(
        self,
        init_strength: float = 1.0,
        min_strength: float = 0.25,
        decay: float = 0.7,
        drop_patience: int = 3,
        eps: float = 1e-4,
    ):
        self.init_strength = float(init_strength)
        self.min_strength = float(min_strength)
        self.decay = float(decay)
        self.drop_patience = int(drop_patience)
        self.eps = float(eps)

        self.strength = float(init_strength)
        self._prev_metric = None
        self._drop_streak = 0

    def step(self, metric: float) -> float:
        """Update controller with the latest metric (higher is better)."""
        try:
            m = float(metric)
        except Exception:
            m = None

        if m is None:
            return self.strength

        if self._prev_metric is not None and (m < self._prev_metric - self.eps):
            self._drop_streak += 1
        else:
            self._drop_streak = 0

        if self._drop_streak >= self.drop_patience:
            self.strength = max(self.min_strength, self.strength * self.decay)
            self._drop_streak = 0

        self._prev_metric = m
        return self.strength


def sample_aug_strength(base_strength: float, jitter: float = 0.0) -> float:
    """Sample a per-view strength around a base value."""
    try:
        b = float(base_strength)
    except Exception:
        b = 1.0
    j = max(0.0, float(jitter))
    if j <= 0:
        return b
    # uniform in [1-j, 1+j]
    r = (2.0 * random.random() - 1.0) * j
    return max(0.0, min(1.0, b * (1.0 + r)))


def get_worst_classes(per_class_acc: torch.Tensor, per_class_total: torch.Tensor, k: int):
    """Return indices of worst-k classes (lowest accuracy), excluding empty classes."""
    if k <= 0:
        return []
    total = per_class_total.clone()
    acc = per_class_acc.clone()

    valid = total > 0
    if valid.sum().item() == 0:
        return []
    acc_valid = acc[valid]
    idx_valid = torch.nonzero(valid, as_tuple=False).view(-1)
    # sort ascending
    order = torch.argsort(acc_valid)
    worst = idx_valid[order[: min(k, order.numel())]].tolist()
    return worst


def update_ce_weights(
    ce_weights: torch.Tensor,
    worst_classes: list,
    num_classes: int,
    hard_weight: float = 1.5,
    momentum: float = 0.9,
) -> torch.Tensor:
    """EMA update of CE class weights to emphasize worst classes."""
    if ce_weights is None or ce_weights.numel() != num_classes:
        ce_weights = torch.ones(num_classes, dtype=torch.float32)

    target = torch.ones(num_classes, dtype=torch.float32)
    hw = float(hard_weight)
    for c in worst_classes:
        if 0 <= int(c) < num_classes:
            target[int(c)] = hw

    m = float(momentum)
    m = min(0.99, max(0.0, m))
    ce_weights = ce_weights.float() * m + target * (1.0 - m)
    # keep weights within a reasonable range
    ce_weights = torch.clamp(ce_weights, 0.25, 10.0)
    return ce_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")

    # dataset selector
    parser.add_argument("--dataset", type=str, default="wisig", choices=["oralce", "wisig"],
                        help="Which dataset loader to use: oralce (npy) or wisig (ManySig pkl).")
    parser.add_argument("--run_name", type=str, default="run1",
                        help="For oralce dataset: which run folder to use (run1/run2).")

    # WiSig (UCLA) options (only used when --dataset wisig)
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl",
                        help="Path to WiSig compact PKL (e.g., ManySig.pkl).")
    parser.add_argument("--wisig_equalized", type=str, default="1",
                        help="Equalization selector: '1' (recommended), '0', or 'both'.")
    parser.add_argument("--wisig_heldout_day", type=str, default="last",
                        help="Leave-One-Day-Out heldout day: 'last' (default), an int index, or a day value string.")
    parser.add_argument("--wisig_domain", type=str, default="day", choices=["day", "rx", "rx_day"],
                        help="Domain id definition (returned by dataset): day / rx / rx_day.")
    parser.add_argument("--wisig_out_len", type=int, default=256,
                        help="Output IQ length after pad/crop for WiSig samples.")
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0,
                        help="Speed knob: max samples per (tx,rx,day,eq) combo for training (0=all).")
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0,
                        help="Speed knob: max samples per (tx,rx,day,eq) combo for test (0=all).")

    # Sample rate (affects SincConv filterbank spacing etc.). If 0, auto-pick by dataset.
    parser.add_argument("--sample_rate_hz", type=float, default=0.0,
                        help="Sample rate in Hz. 0 => auto (oralce=5e6, wisig=25e6).")
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr_base_init", type=float, default=0.0002)
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum LR for cosine annealing.")
    parser.add_argument("--lr_warmup_epochs", type=int, default=8,
                        help="Warmup epochs for LR scheduler (LinearLR).")
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1,
                        help="LinearLR start_factor.")
    parser.add_argument("--wd", type=float, default=0.0)
    # legacy knobs kept for backward CLI compatibility
    parser.add_argument("--warmup_epochs", type=int, default=8)
    parser.add_argument("--ramp_epochs", type=int, default=100)

    # staged optimization (prevents early embedding collapse)
    parser.add_argument("--stage1_end", type=int, default=8,
                        help="Epoch end for Stage 1 (CE only).")
    parser.add_argument("--stage2_end", type=int, default=50,
                        help="Epoch end for Stage 2 (CE+Con).")
    parser.add_argument("--stage3_end", type=int, default=200,
                        help="Epoch end for Stage 3 (CE+Con+Proto).")
    parser.add_argument("--stage_transition_epochs", type=int, default=5,
                        help="Linear transition epochs when enabling a new loss.")

    parser.add_argument("--n_views", type=int, default=4, help="Sat-DG recommended 4: weak/receiver/sat/defect.")

    # SatSim / ReceiverDG views
    parser.add_argument("--enable_receiver_view", type=int, default=1, help="Enable receiver_dg view (apply_receiver_dg).")
    parser.add_argument("--enable_sat_view", type=int, default=1, help="Enable sat_channel view (sat_channel.py).")
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9, help="Carrier frequency for sat_channel (Hz). WiFi ch13=2.462e9.")
    parser.add_argument("--sat_strength", type=float, default=0.5, help="Base mix strength for sat view (0..1).")
    parser.add_argument("--receiver_env_max", type=int, default=3, help="Max receiver_dg env id (0..3). Strength controls within this range.")

    # Domain adaptation (default ON)
    parser.add_argument("--da_method", type=str, default="coral", choices=["coral", "mmd", "dann", "none"], help="Domain adaptation loss. Default coral (ON).")
    parser.add_argument("--da_lambda", type=float, default=0.3, help="Weight for DA loss (enabled in Stage 4 by default).")
    parser.add_argument("--da_grl_lambda", type=float, default=1.0, help="GRL strength for DANN.")
    parser.add_argument("--mmd_gamma", type=float, default=1.0, help="RBF gamma for MMD.")
    parser.add_argument("--domain_balanced", type=int, default=1, help="Use domain-balanced batch sampling when domain ids are available (WiSig).")

    # PA head loss
    parser.add_argument("--pa_lambda", type=float, default=3.0, help="Weight for PA regression loss (on defect view).")
    parser.add_argument("--pa_zero_weight", type=float, default=0.2, help="Down-weight PA loss when pa_strength=0.")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--tau_proto", type=float, default=0.07)
    parser.add_argument("--tau_start", type=float, default=0.30,
                        help="Initial SupCon temperature.")
    parser.add_argument("--tau_end", type=float, default=0.07,
                        help="Final SupCon temperature.")
    parser.add_argument("--tau_decay_start", type=int, default=21,
                        help="Epoch to start tau decay.")
    parser.add_argument("--tau_decay_end", type=int, default=220,
                        help="Epoch to end tau decay.")
    parser.add_argument("--lambda_con", type=float, default=0.05)
    parser.add_argument("--lambda_proto", type=float, default=0.2)

    parser.add_argument("--dac_lambda", type=float, default=3.0)
    parser.add_argument("--dac_zero_weight", type=float, default=0.2)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Cosine-margin branch metrics (optional). Default ON for backward-compat with older logs.
    mg = parser.add_mutually_exclusive_group()
    mg.add_argument("--use_cosine_margin", dest="use_cosine_margin", action="store_true",
                    help="Enable cosine-margin branch metrics (default: on).")
    mg.add_argument("--no_cosine_margin", dest="use_cosine_margin", action="store_false",
                    help="Disable cosine-margin branch metrics.")
    parser.set_defaults(use_cosine_margin=True)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=4)
    # AMP mixed precision (default: ON). Use --no_amp to disable.
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable AMP mixed precision.")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable AMP mixed precision.")
    parser.set_defaults(amp=True)
    parser.add_argument("--amp_init_scale", type=float, default=2.0**6, help="GradScaler init_scale for FP16 AMP.")
    parser.add_argument("--device", type=str, default="cuda:0")

    # rollback
    parser.add_argument("--rollback_enable", action="store_true", default=True)
    parser.add_argument("--rollback_drop_abs", type=float, default=12.0)
    parser.add_argument("--rollback_shrink", type=float, default=0.5)
    parser.add_argument("--rollback_freeze_aux_epochs", type=int, default=5,
                        help="After rollback, freeze ramped auxiliary losses for this many epochs.")

    parser.add_argument("--save_path", type=str, default="best_model_v9.pth")
    parser.add_argument("--model_size", type=str, default="M")

    # --- stability / regularization ---
    parser.add_argument("--label_smoothing", type=float, default=0.01,
                        help="CE label smoothing (0 disables).")

    # --- adaptive augmentation (reduce strength after repeated performance drops) ---
    parser.add_argument("--aug_strength_init", type=float, default=1.0,
                        help="Initial augmentation strength (0~1).")
    parser.add_argument("--aug_strength_min", type=float, default=0.25,
                        help="Minimum augmentation strength (0~1).")
    parser.add_argument("--aug_strength_decay", type=float, default=0.7,
                        help="Multiply strength by this when drops happen repeatedly.")
    parser.add_argument("--aug_drop_patience", type=int, default=3,
                        help="How many consecutive metric drops before reducing aug strength.")
    parser.add_argument("--aug_mix", type=str, default="prob", choices=["prob", "blend"],
                        help="How to realize partial strength: prob (default) or blend.")
    parser.add_argument("--aug_jitter", type=float, default=0.10,
                        help="Per-view strength jitter around base strength (0 disables).")
    parser.add_argument("--dg_warmup_epochs", type=int, default=80,
                        help="Warmup epochs for receiver/sat domain-generalization augmentations.")

    # prototype bank stability
    parser.add_argument("--proto_momentum", type=float, default=0.95,
                        help="Base EMA momentum for prototype updates.")
    parser.add_argument("--proto_warmup_momentum", type=float, default=0.995,
                        help="EMA momentum in early prototype updates.")
    parser.add_argument("--proto_warmup_updates", type=int, default=10,
                        help="Use proto_warmup_momentum for the first N updates per class.")
    parser.add_argument("--proto_min_batch_samples", type=int, default=2,
                        help="Skip prototype update when class samples in batch are fewer than this.")
    parser.add_argument("--proto_min_count", type=int, default=8,
                        help="A class prototype is valid only after accumulating at least this many samples.")

    # --- hard class mining (emphasize worst classes next epoch via CE weights) ---
    parser.add_argument("--hard_k", type=int, default=2,
                        help="After each epoch, find worst-k classes on test set and upweight them in CE (0 disables).")
    parser.add_argument("--hard_weight", type=float, default=1.5,
                        help="CE weight factor for worst classes (only if hard_k>0).")
    parser.add_argument("--hard_momentum", type=float, default=0.9,
                        help="EMA momentum for CE class weights (only if hard_k>0).")

    # --- evaluation speed knobs ---
    parser.add_argument("--eval_every", type=int, default=2,
                        help="Run full evaluation every N epochs (1 = every epoch).")
    parser.add_argument("--eval_max_batches", type=int, default=0,
                        help="If >0, evaluate on at most this many test batches each eval (0 = all).")
    args = parser.parse_args()
    if not (int(args.stage1_end) < int(args.stage2_end) < int(args.stage3_end)):
        raise ValueError("Require stage1_end < stage2_end < stage3_end.")
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # speed: allow TF32 on Ampere+ and enable cudnn autotune
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Starting Training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} | AMP: {args.amp}")
    print(f"DATASET_DIR={args.dataset_dir}")
    print(f"DATASET={args.dataset} | sample_rate_hz={(args.sample_rate_hz if args.sample_rate_hz else 0)}")
    print(f"NUM_CLASSES={args.num_classes}, BATCH_SIZE={args.batch_size}, EPOCHS={args.epochs}")
    print(f"LR_BASE_INIT={args.lr_base_init}, LR_MIN={args.lr_min}, WD={args.wd}")
    print(f"LR schedule: warmup={args.lr_warmup_epochs} (start_factor={args.lr_warmup_start_factor}) + cosine")
    print(f"Stages: S1<=E{args.stage1_end}, S2<=E{args.stage2_end}, S3<=E{args.stage3_end}, S4>E{args.stage3_end}")
    print(f"N_VIEWS={args.n_views}, TAU(start/end)=({args.tau_start},{args.tau_end}), TAU_PROTO={args.tau_proto}")
    print(f"LAMBDA_CON={args.lambda_con}, LAMBDA_PROTO={args.lambda_proto}")
    print(f"DAC_LAMBDA={args.dac_lambda}, DAC_ZERO_WEIGHT={args.dac_zero_weight}")
    print(f"PA_LAMBDA={getattr(args,'pa_lambda',0.0)}, PA_ZERO_WEIGHT={getattr(args,'pa_zero_weight',0.2)}")
    print(f"DG_WARMUP_EPOCHS={args.dg_warmup_epochs}")
    print(f"PROTO: m={args.proto_momentum}, m_warm={args.proto_warmup_momentum}, warm_updates={args.proto_warmup_updates}, min_batch={args.proto_min_batch_samples}, min_count={args.proto_min_count}")
    print(f"LEGACY_WARMUP_EPOCHS={args.warmup_epochs}, LEGACY_RAMP_EPOCHS={args.ramp_epochs}")
    print(f"GRAD_CLIP={args.grad_clip}")
    print(f"ROLLBACK: enable={args.rollback_enable}, drop_abs={args.rollback_drop_abs}, shrink={args.rollback_shrink}")

    # -------- dataset --------
    if args.sample_rate_hz is None:
        args.sample_rate_hz = 0.0
    if float(args.sample_rate_hz) <= 0.0:
        # auto pick
        args.sample_rate_hz = 25e6 if args.dataset.lower() == "wisig" else 5e6

    if args.dataset.lower() == "wisig":
        if load_wisig_compact_pkl is None or make_leave_one_day_out_split is None:
            raise RuntimeError("dataset_wisig.py not available or failed to import. Please place dataset_wisig.py in the same folder.")

        if not args.wisig_pkl:
            raise ValueError("For --dataset wisig, you must set --wisig_pkl /path/to/ManySig.pkl")

        ds_w = load_wisig_compact_pkl(args.wisig_pkl)

        # infer heldout day
        held = args.wisig_heldout_day
        heldout_day = None
        if held is None:
            heldout_day = None
        else:
            hs = str(held).strip()
            if hs.lower() in ("", "none", "null", "last"):
                heldout_day = None
            else:
                try:
                    heldout_day = int(hs)
                except Exception:
                    heldout_day = hs

        # equalization
        eq = args.wisig_equalized
        eq2 = "both" if str(eq).lower() == "both" else int(eq)

        # max samples per combo
        max_tr = None if int(args.wisig_max_train_per_combo) <= 0 else int(args.wisig_max_train_per_combo)
        max_te = None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo)

        train_ds, test_ds = make_leave_one_day_out_split(
            ds_w,
            heldout_day=heldout_day,
            equalized=eq2,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform=None,  # keep deterministic here; augmentations happen in training loop
            max_samples_per_combo_train=max_tr,
            max_samples_per_combo_test=max_te,
            seed=int(args.seed),
        )

        # num_classes auto check
        infer_nc = len(ds_w.get("tx_list", []))
        if args.num_classes == 16 and infer_nc not in (0, 16):
            print(f"[WISIG] overriding --num_classes 16 -> inferred {infer_nc} from tx_list")
            args.num_classes = infer_nc
        elif infer_nc not in (0, args.num_classes):
            print(f"[WISIG] note: inferred num_classes={infer_nc} from tx_list, but you set --num_classes={args.num_classes}")

        args.input_len = int(args.wisig_out_len)
        print(f"[WISIG] pkl={args.wisig_pkl} heldout_day={heldout_day} eq={eq2} out_len={args.input_len} domain={args.wisig_domain}")
    else:
        # ORALCE / npy dataset
        train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train", run_name=args.run_name)
        test_ds = WiFiRFFIDataset(args.dataset_dir, mode="test", run_name=args.run_name)
        # try to infer input length from one sample
        try:
            x0, _y0 = train_ds[0]
            args.input_len = int(x0.shape[-1])
        except Exception:
            args.input_len = 1024
        print(f"[ORALCE] dir={args.dataset_dir} run={args.run_name} input_len={args.input_len}")

    # ---- train loader (optionally domain-balanced) ----
    train_domain_ids = None
    if int(getattr(args, "domain_balanced", 0)) == 1:
        # Best effort: use WiSigCompactDataset internal index for cheap domain extraction
        if hasattr(train_ds, "index") and hasattr(train_ds, "_domain_lut"):
            try:
                train_domain_ids = [int(train_ds._domain_lut[(it.rx_i, it.day_i)]) for it in train_ds.index]
            except Exception:
                train_domain_ids = None

    if train_domain_ids is not None:
        uniq = sorted(set(train_domain_ids))
        args.num_domains = len(uniq)
        if args.num_domains >= 2:
            batch_sampler = DomainBalancedBatchSampler(train_domain_ids, batch_size=args.batch_size, drop_last=True, seed=int(args.seed))
            train_loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            print(f"[DG] Domain-balanced sampling ON: num_domains={args.num_domains} ({uniq})")
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=(args.num_workers > 0),
            )
            print("[DG] Domain-balanced sampling skipped: only 1 domain detected.")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
        print("[DG] Domain-balanced sampling OFF.")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # model
    model, model_msg = build_model_safely(args.num_classes, str(device), model_size=args.model_size, dataset=args.dataset, input_len=int(getattr(args,'input_len',1024)), sample_rate_hz=float(args.sample_rate_hz))
    model.to(device)
    print(model_msg)

    # infer feat_dim for prototype bank: do a tiny forward
    feat_dim = None
    model.eval()
    # Infer feat_dim for prototype bank (prefer projection head feat_con)
    feat_dim = None
    with torch.no_grad():
        x0 = torch.zeros(2, 2, int(getattr(args,'input_len',1024)), device=device)
        try:
            out0 = _call_model(model, x0, return_aux=True)
            logits0, dac0, pa0, feat_cls0, feat_con0, cos0 = _unpack_model_output(out0)
            feat0 = feat_con0 if feat_con0 is not None else feat_cls0
            if feat0 is not None and torch.is_tensor(feat0):
                feat_dim = int(feat0.shape[-1])
        except Exception as e:
            feat_dim = None
    if feat_dim is None:
        feat_dim = 256
        print(f"[WARN] Cannot infer feat_dim from model output. Fallback feat_dim={feat_dim}")

    proto_bank = PrototypeBank(
        args.num_classes,
        feat_dim,
        momentum=float(getattr(args, "proto_momentum", 0.95)),
        warmup_momentum=float(getattr(args, "proto_warmup_momentum", 0.995)),
        warmup_updates=int(getattr(args, "proto_warmup_updates", 20)),
        min_batch_samples=int(getattr(args, "proto_min_batch_samples", 2)),
        min_count_valid=int(getattr(args, "proto_min_count", 8)),
        device=device,
    )

    # domain discriminator for DANN (optional)
    domain_clf = None
    if getattr(args, "da_method", "coral") == "dann":
        num_dom = int(getattr(args, "num_domains", 0))
        if num_dom <= 1:
            num_dom = 2
        domain_clf = DomainDiscriminator(in_dim=int(feat_dim), num_domains=num_dom).to(device)
        print(f"[DA] DANN enabled: num_domains={num_dom}, da_grl_lambda={args.da_grl_lambda}")

    # optimizer
    lr_base = args.lr_base_init
    params = list(model.parameters())
    if domain_clf is not None:
        params += list(domain_clf.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr_base, weight_decay=args.wd)
    lr_warmup_epochs = int(getattr(args, "lr_warmup_epochs", 0))
    if lr_warmup_epochs <= 0:
        # backward-compatible fallback
        lr_warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    scheduler = build_lr_scheduler(
        optimizer,
        total_epochs=int(args.epochs),
        warmup_epochs=lr_warmup_epochs,
        eta_min=float(getattr(args, "lr_min", 1e-6)),
        warmup_start_factor=float(getattr(args, "lr_warmup_start_factor", 0.1)),
    )

    # AMP scaler
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=float(getattr(args, "amp_init_scale", 2.0**6)))

    # augmentor
    augmentor, aug_msg = build_augmentor_safely(str(device))
    print(aug_msg)

    best = BestState(best_acc=-1.0, best_epoch=-1, best_blob=None)
    # --- adaptive augmentation controller ---
    aug_ctrl = AdaptiveAugController(
        init_strength=args.aug_strength_init,
        min_strength=args.aug_strength_min,
        decay=args.aug_strength_decay,
        drop_patience=args.aug_drop_patience,
    )

    # --- CE class weights (for hard-mining) ---
    ce_weights = torch.ones(args.num_classes, dtype=torch.float32)

    # --- cached eval stats (used when eval_every>1) ---
    last_test_acc = -1.0
    last_per_class_total = torch.zeros(args.num_classes, dtype=torch.long)
    last_per_class_correct = torch.zeros(args.num_classes, dtype=torch.long)
    last_per_class_acc = torch.zeros(args.num_classes, dtype=torch.float32)

    aux_freeze_until_epoch = 0
    # -----------------------
    # train loop
    # -----------------------
    for epoch in range(1, args.epochs + 1):
        base_aug_strength = aug_ctrl.strength
        # (optional) print current augmentation strength for transparency
        # print(f"[AUG] strength={base_aug_strength:.3f}")
        t0 = time.time()
        stage_id, g_con, g_proto, g_dac = compute_stage_gates(
            epoch,
            stage1_end=int(getattr(args, "stage1_end", 30)),
            stage2_end=int(getattr(args, "stage2_end", 120)),
            stage3_end=int(getattr(args, "stage3_end", 200)),
            transition_epochs=int(getattr(args, "stage_transition_epochs", 0)),
        )
        # rollback safety: temporarily freeze auxiliary objectives
        if epoch <= aux_freeze_until_epoch:
            g_con = 0.0
            g_proto = 0.0
            g_dac = 0.0

        # DA is also delayed to Stage 4 to avoid early representation over-constraint.
        g_da = g_dac
        lr_now = float(optimizer.param_groups[0]["lr"])

        tau_now = compute_tau(
            epoch,
            tau_start=float(getattr(args, "tau_start", args.tau)),
            tau_end=float(getattr(args, "tau_end", args.tau)),
            decay_start=int(getattr(args, "tau_decay_start", int(getattr(args, "stage1_end", 30)) + 1)),
            decay_end=int(getattr(args, "tau_decay_end", int(getattr(args, "stage3_end", 200)))),
        )
        dg_warm = linear_warmup(epoch, int(getattr(args, "dg_warmup_epochs", 0)))

        model.train()

        loss_sum = 0.0
        ce_sum = 0.0
        con_sum = 0.0
        proto_sum = 0.0
        dac_sum = 0.0
        pa_sum = 0.0
        da_sum = 0.0

        correct = 0
        total = 0

        # optional margin stats if model returns cos logits
        correct_margin = 0
        margin_viol = 0
        margin_total = 0

        skipped_steps = 0
        gnorm_avg = 0.0
        gnorm_max = 0.0
        gnorm_cnt = 0

        for batch in train_loader:
            x, y, _extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True)  # [B,2,1024]
            y = y.to(device, non_blocking=True)

            # domain ids (if provided by dataset, e.g., WiSig day/rx)
            dom = None
            if _extra is not None and len(_extra) >= 1:
                d0 = _extra[0]
                try:
                    if torch.is_tensor(d0):
                        dom = d0.to(device, non_blocking=True).view(-1)
                    else:
                        dom = torch.as_tensor(d0, device=device).view(-1)
                except Exception:
                    dom = None

            B = x.size(0)
            # ---- build views (stage-aware) ----
            V_cfg = int(getattr(args, "n_views", 4))
            V_cfg = max(1, min(4, V_cfg))

            w_con = float(getattr(args, "lambda_con", 0.0)) * g_con
            w_proto = float(getattr(args, "lambda_proto", 0.0)) * g_proto
            w_dac = float(getattr(args, "dac_lambda", 0.0)) * g_dac
            w_pa = float(getattr(args, "pa_lambda", 0.0)) * g_dac
            w_da = float(getattr(args, "da_lambda", 0.0)) * g_da

            need_defect_view = (w_dac > 0.0) or (w_pa > 0.0)

            # Stage 1 should be CE-only: avoid extra views entirely.
            if (w_con <= 0.0) and (w_proto <= 0.0) and (w_dac <= 0.0) and (w_pa <= 0.0) and (w_da <= 0.0):
                V_target = 1
            else:
                if need_defect_view:
                    # with defect view: [v0, v1, v2, v_def] style
                    V_target = max(2, V_cfg)
                else:
                    # no defect view: use up to 3 non-defect views for contrastive learning
                    V_target = max(2, min(3, V_cfg))

            views = []
            dac_strength = torch.zeros(B, device=device)
            pa_strength = torch.zeros(B, device=device)

            if augmentor is not None:
                # view0: weak/near-clean (used for CE)
                s0_base = min(0.3, float(base_aug_strength) * 0.3)
                s0 = sample_aug_strength(s0_base, args.aug_jitter)
                v0 = call_augmentor(
                    augmentor, x,
                    labels=y,
                    strength=s0,
                    mix_mode=args.aug_mix,
                )

                # warmup domain-generalization views (receiver/sat)
                dg_base = float(base_aug_strength) * float(dg_warm)

                v_def = None
                sd = torch.zeros(B, device=device)
                sp = torch.zeros(B, device=device)
                if need_defect_view:
                    # defect view is added only in Stage 4 (via g_dac)
                    v_def, sd, sp = call_augmentor(
                        augmentor, x,
                        labels=y,
                        return_defect_strengths=True,
                        dac_pa=True,
                        strength=1.0,
                        mix_mode=args.aug_mix,
                        defect_apply_channel=False,
                    )

                # view1: receiver-chain domain randomization (SRO/LPF/AGC/softclip/multipath)
                v1 = None
                if (need_defect_view and V_target >= 3) or ((not need_defect_view) and V_target >= 2):
                    if int(getattr(args, "enable_receiver_view", 1)) == 1 and (_apply_receiver_dg is not None):
                        max_env = int(getattr(args, "receiver_env_max", 3))
                        max_env = max(0, min(3, max_env))
                        sev = int(round(float(dg_base) * max_env))
                        sev = max(0, min(max_env, sev))
                        env_id = torch.randint(low=0, high=sev + 1, size=(B,), device=device)
                        v1 = _apply_receiver_dg(x, fs=float(args.sample_rate_hz), env_id=env_id)
                        s_rx = sample_aug_strength(float(dg_base), args.aug_jitter)
                        if s_rx < 1.0:
                            if args.aug_mix == "blend":
                                v1 = x * (1.0 - s_rx) + v1 * s_rx
                            else:
                                mask = (torch.rand((B, 1, 1), device=device) < s_rx)
                                v1 = torch.where(mask, v1, x)
                    else:
                        # fallback: normal aug view
                        s1 = sample_aug_strength(float(dg_base), args.aug_jitter)
                        v1 = call_augmentor(
                            augmentor, x,
                            labels=y,
                            strength=s1,
                            mix_mode=args.aug_mix,
                        )

                # view2: satellite channel view (independent)
                v2 = None
                if (need_defect_view and V_target >= 4) or ((not need_defect_view) and V_target >= 3):
                    if int(getattr(args, "enable_sat_view", 1)) == 1 and (SatSimConfig is not None) and (apply_sat_gnd_channel_batch is not None):
                        cfg = SatSimConfig(
                            fs_hz=float(args.sample_rate_hz),
                            fc_hz=float(getattr(args, "sat_fc_hz", 2.462e9)),
                            scenario="urban",
                            weather="clear",
                            loo_level="mid",
                            enable_multipath=False,
                        )
                        y_sat, _meta, _st = apply_sat_gnd_channel_batch(x, cfg, return_meta=False)
                        s_sat = sample_aug_strength(float(dg_base) * float(getattr(args, "sat_strength", 1.0)), args.aug_jitter)
                        s_sat = max(0.0, min(1.0, s_sat))
                        if s_sat < 1.0:
                            if args.aug_mix == "blend":
                                v2 = x * (1.0 - s_sat) + y_sat * s_sat
                            else:
                                mask = (torch.rand((B, 1, 1), device=device) < s_sat)
                                v2 = torch.where(mask, y_sat, x)
                        else:
                            v2 = y_sat
                    else:
                        # fallback: normal aug view
                        s2 = sample_aug_strength(float(dg_base), args.aug_jitter)
                        v2 = call_augmentor(
                            augmentor, x,
                            labels=y,
                            strength=s2,
                            mix_mode=args.aug_mix,
                        )

                # choose views (defect view stays LAST when enabled)
                if need_defect_view:
                    if V_target <= 2:
                        views = [v0, v_def]
                    elif V_target == 3:
                        views = [v0, v1, v_def]
                    else:
                        views = [v0, v1, v2, v_def]
                else:
                    if V_target <= 1:
                        views = [v0]
                    elif V_target == 2:
                        views = [v0, v1]
                    else:
                        views = [v0, v1, v2]

                dac_strength = sd
                pa_strength = sp

            else:
                # No augmentor -> do the simplest thing to avoid misleading defect supervision
                views = [x]
                V_target = 1

            V = len(views)

            # concat forward once (cheaper & consistent)
            x_cat = torch.cat(views, dim=0)  # [B*V,2,1024]

            optimizer.zero_grad(set_to_none=True)

            # repeat labels for all views so CosFace margin can be applied (if model supports y=...)
            y_cat = y.repeat(V)
            with torch.cuda.amp.autocast(enabled=use_amp):
                try:
                    out = model(x_cat, y=y_cat, return_aux=True)
                except TypeError:
                    # fallback for older models without y=
                    out = _call_model(model, x_cat, return_aux=True)

            logits_all, dac_pred_all, pa_pred_all, feat_cls_all, feat_con_all, cos_logits_all = _unpack_model_output(out)
            if logits_all is None:
                raise RuntimeError("Model output has no logits. Please ensure model returns logits or dict['logits'].")

            # Use cosine logits if the model provides it; otherwise fall back to logits
            cos_all = None
            if args.use_cosine_margin:
                cos_src = cos_logits_all if cos_logits_all is not None else logits_all
                if torch.isfinite(cos_src).all():
                    cos_all = cos_src.float()
                else:
                    cos_all = None

            # ---- losses in float32 to avoid fp16 overflow (especially for exp/logsumexp) ----
            if not torch.isfinite(logits_all).all():
                skipped_steps += 1
                continue
            logits_all_f = logits_all.float()
            logits0_f = logits_all_f[:B]
            w_ce = ce_weights.to(logits0_f.device, dtype=logits0_f.dtype)
            ce_loss = F.cross_entropy(logits0_f, y, weight=w_ce, label_smoothing=args.label_smoothing)

            # acc
            with torch.no_grad():
                correct += (logits0_f.argmax(dim=1) == y).sum().item()
                total += B

            # ---- stage-aware auxiliary losses ----
            con_loss = logits0_f.new_tensor(0.0)
            pr_loss = logits0_f.new_tensor(0.0)
            d_loss = logits0_f.new_tensor(0.0)
            p_loss = logits0_f.new_tensor(0.0)
            da_loss = logits0_f.new_tensor(0.0)

            if (w_con > 0.0 or w_proto > 0.0 or w_da > 0.0) and (feat_con_all is not None):
                if not torch.isfinite(feat_con_all).all():
                    skipped_steps += 1
                    continue
                # explicit L2 normalization before contrastive/prototype objectives
                feat_con_all_f = F.normalize(feat_con_all.float(), dim=1)
                # split feat_con into [B,V,D]
                feat_con = feat_con_all_f.view(V, B, -1).permute(1, 0, 2).contiguous()

                # Domain adaptation loss (default: CORAL) on view0 feat_con
                if (w_da > 0.0) and (getattr(args, 'da_method', 'coral') != 'none') and (dom is not None):
                    dom_b = dom[:B].long()
                    if torch.unique(dom_b).numel() >= 2:
                        mth = getattr(args, 'da_method', 'coral')
                        if mth == 'coral':
                            da_loss = coral_multi_domain(feat_con[:, 0, :], dom_b)
                        elif mth == 'mmd':
                            da_loss = mmd_rbf_multi_domain(feat_con[:, 0, :], dom_b, gamma=float(getattr(args, 'mmd_gamma', 1.0)))
                        elif mth == 'dann' and (domain_clf is not None):
                            dom_uniq = torch.unique(dom_b)
                            mapping = {int(v.item()): i for i, v in enumerate(dom_uniq)}
                            dom_y = torch.tensor([mapping[int(v.item())] for v in dom_b], device=feat_con.device, dtype=torch.long)
                            feat_grl = grad_reverse(feat_con[:, 0, :], lambd=float(getattr(args, 'da_grl_lambda', 1.0)))
                            dom_logits = domain_clf(feat_grl)
                            da_loss = F.cross_entropy(dom_logits, dom_y)


                # SupCon over feat_con
                if w_con > 0.0:
                    con_loss = supcon_loss(feat_con, y, tau=tau_now)

                # Prototype loss: use view0 features
                if w_proto > 0.0:
                    feat0 = feat_con[:, 0, :]
                    proto_bank.update(feat0, y)
                    pr_loss = proto_ce_loss(
                        feat0,
                        y,
                        proto_bank,
                        tau=args.tau_proto,
                        min_count=int(getattr(args, "proto_min_count", 8)),
                    )

            # DAC loss: use last view dac_pred if available (defect view is last by construction)
            if (w_dac > 0.0) and (dac_pred_all is not None) and need_defect_view:
                dac_pred = dac_pred_all[B * (V - 1): B * V]
                if not torch.isfinite(dac_pred).all():
                    skipped_steps += 1
                    continue
                dac_pred = dac_pred.float()
                dac_pred = dac_pred.view(B, -1).squeeze(-1)
                target = dac_strength.view(B, -1).squeeze(-1).float()

                w = torch.ones_like(target)
                w = torch.where(target.abs() < 1e-12, w * args.dac_zero_weight, w)
                # huber is more stable than mse under strong DAC
                d_loss = (F.smooth_l1_loss(dac_pred, target, reduction="none") * w).mean()

            # PA loss: use last view pa_pred if available
            if (w_pa > 0.0) and (pa_pred_all is not None) and need_defect_view:
                pa_pred = pa_pred_all[B * (V - 1): B * V]
                if not torch.isfinite(pa_pred).all():
                    skipped_steps += 1
                    continue
                pa_pred = pa_pred.float()
                pa_pred = pa_pred.view(B, -1).squeeze(-1)
                target_p = pa_strength.view(B, -1).squeeze(-1).float()
                wpa = torch.ones_like(target_p)
                wpa = torch.where(target_p.abs() < 1e-12, wpa * float(getattr(args,'pa_zero_weight',0.2)), wpa)
                p_loss = (F.smooth_l1_loss(pa_pred, target_p, reduction="none") * wpa).mean()

            loss = (
                ce_loss
                + w_con * con_loss
                + w_proto * pr_loss
                + w_dac * d_loss
                + w_pa * p_loss
                + w_da * da_loss
            )

            # ---- backward ----
            if not torch.isfinite(loss.detach()):
                skipped_steps += 1
                continue

            if use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()

                # grad clip (must unscale first)
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                else:
                    scaler.unscale_(optimizer)
                    gn = torch.norm(
                        torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]),
                        p=2
                    ) if any(p.grad is not None for p in model.parameters()) else torch.tensor(0.0, device=device)

                # step
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                # if scale dropped, step was effectively skipped due to inf/nan
                if scale_after < scale_before:
                    skipped_steps += 1

            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            # grad norm stats
            if torch.is_tensor(gn):
                g = float(gn.detach().item())
                if math.isfinite(g):
                    gnorm_avg += g
                    gnorm_max = max(gnorm_max, g)
                    gnorm_cnt += 1
                else:
                    skipped_steps += 1

            # meters
            loss_sum += float(loss.detach().item())
            ce_sum += float(ce_loss.detach().item())
            con_sum += float(con_loss.detach().item())
            proto_sum += float(pr_loss.detach().item())
            dac_sum += float(d_loss.detach().item())
            pa_sum += float(p_loss.detach().item())
            da_sum += float(da_loss.detach().item())

            # optional margin stats if cos logits exist
            if cos_all is not None:
                cos0 = cos_all[:B]
                with torch.no_grad():
                    correct_margin += (cos0.argmax(dim=1) == y).sum().item()
                    margin_total += B
                    # rough "violation": true cos <= max other cos  (you can refine using your margin m)
                    tmp = cos0.clone()
                    tmp[torch.arange(B, device=device), y] = -1e4
                    other_max = tmp.max(dim=1).values
                    true_cos = cos0[torch.arange(B, device=device), y]
                    margin_viol += (true_cos <= other_max).sum().item()

        # epoch stats
        train_acc = (correct / max(1, total)) * 100.0
        train_acc_margin = (correct_margin / max(1, margin_total)) * 100.0 if margin_total > 0 else 0.0
        viol_pct = (margin_viol / max(1, margin_total)) * 100.0 if margin_total > 0 else 0.0

        denom = max(1, len(train_loader))
        loss_avg = loss_sum / denom
        ce_avg = ce_sum / denom
        con_avg = con_sum / denom
        proto_avg = proto_sum / denom
        dac_avg = dac_sum / denom
        pa_avg = pa_sum / denom
        da_avg = da_sum / denom

        gavg = (gnorm_avg / max(1, gnorm_cnt)) if gnorm_cnt > 0 else 0.0

        # ---- eval ----
        eval_every = int(getattr(args, "eval_every", 1))
        eval_every = max(1, eval_every)
        do_eval = (eval_every <= 1) or (epoch % eval_every == 0) or (epoch == 1) or (epoch == args.epochs)

        if do_eval:
            model.eval()
            test_correct = 0
            test_total = 0
            per_class_total = torch.zeros(args.num_classes, dtype=torch.long)
            per_class_correct = torch.zeros(args.num_classes, dtype=torch.long)

            max_eval_batches = int(getattr(args, "eval_max_batches", 0))
            max_eval_batches = max(0, max_eval_batches)
            seen_batches = 0

            with torch.no_grad():
                for batch in test_loader:
                    xb, yb, _extra = unpack_batch(batch)
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    out = _call_model(model, xb)
                    logits, _, _, _, _, _ = _unpack_model_output(out)
                    pred = logits.argmax(dim=1)
                    test_correct += (pred == yb).sum().item()
                    test_total += yb.numel()

                    # per-class statistics (on CPU for compatibility)
                    y_cpu = yb.detach().to('cpu')
                    p_cpu = pred.detach().to('cpu')
                    per_class_total += torch.bincount(y_cpu, minlength=args.num_classes)
                    per_class_correct += torch.bincount(y_cpu[p_cpu == y_cpu], minlength=args.num_classes)

                    seen_batches += 1
                    if max_eval_batches > 0 and seen_batches >= max_eval_batches:
                        break

            test_acc = (test_correct / max(1, test_total)) * 100.0
            per_class_acc = per_class_correct.float() / per_class_total.clamp(min=1).float() * 100.0

            # cache
            last_test_acc = float(test_acc)
            last_per_class_total = per_class_total.clone()
            last_per_class_correct = per_class_correct.clone()
            last_per_class_acc = per_class_acc.clone()

        else:
            # reuse last cached eval stats (fast path)
            test_acc = float(last_test_acc) if last_test_acc >= 0 else 0.0
            per_class_total = last_per_class_total.clone()
            per_class_correct = last_per_class_correct.clone()
            per_class_acc = last_per_class_acc.clone()

        # Identify worst classes and (optionally) upweight them in CE next epoch
        worst_classes = get_worst_classes(per_class_acc, per_class_total, args.hard_k)
        if args.hard_k > 0 and len(worst_classes) > 0:
            ce_weights = update_ce_weights(
                ce_weights, worst_classes, args.num_classes,
                hard_weight=args.hard_weight, momentum=args.hard_momentum
            )
            # Print a short diagnostic
            worst_str = ", ".join([f"{c}:{per_class_acc[c]:.1f}%" for c in worst_classes])
            print(f"  Worst classes (acc): {worst_str}")

        # Adaptive augmentation: if test_acc drops repeatedly, reduce augmentation strength
        prev_strength = aug_ctrl.strength
        new_strength = aug_ctrl.step(test_acc)
        if new_strength < prev_strength - 1e-6:
            print(f"  [AUG] performance dropped repeatedly -> strength {prev_strength:.3f} -> {new_strength:.3f}")

        epoch_time = time.time() - t0

        # ---- auto AMP fail-safe: if almost all steps skipped, disable AMP next epoch ----
        if use_amp and skipped_steps >= int(0.9 * len(train_loader)):
            print(f"  !!! AMP unstable: skipped_steps={skipped_steps}/{len(train_loader)}. Disabling AMP & resetting scaler.")
            use_amp = False
            scaler = None  # disable

        # print
        msg = (
            f"Epoch [{epoch}/{args.epochs}] Time: {epoch_time:.1f}s | "
            f"Loss: {loss_avg:.4f} (CE: {ce_avg:.4f}, Con: {con_avg:.4f}, Proto: {proto_avg:.4f}, DAC: {dac_avg:.4f}, PA: {pa_avg:.4f}, DA: {da_avg:.4f}) | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"stage=S{stage_id} | g(con/proto/dac/da)=({g_con:.2f}/{g_proto:.2f}/{g_dac:.2f}/{g_da:.2f}) | "
            f"tau={tau_now:.4f} | lr={lr_now:.6g} | aug={base_aug_strength:.3f}->{aug_ctrl.strength:.3f} | dg_warm={dg_warm:.2f} | "
            f"Train Acc (margin): {train_acc_margin:.2f}% | Margin Viol: {viol_pct:.2f}% | "
            f"gnorm(avg/max): {gavg:.2f}/{gnorm_max:.2f}"
        )
        if skipped_steps > 0:
            msg += f" | skipped_steps: {skipped_steps}"
        print(msg)

        # ---- save best ----
        if do_eval and (test_acc > best.best_acc):
            best.best_acc = test_acc
            best.best_epoch = epoch
            blob = pack_state(model, optimizer, scaler, epoch, lr_base, best.best_acc, best.best_epoch, global_step=0)
            best.best_blob = copy.deepcopy(blob)
            torch.save(blob, args.save_path)
            print(f"  >>> Best Model Saved! (Acc: {test_acc:.2f}%)")

        # ---- rollback on collapse ----
        if args.rollback_enable and best.best_acc >= 0 and (best.best_blob is not None):
            if test_acc < best.best_acc - args.rollback_drop_abs:
                print(
                    f"  !!! Collapse detected: test_acc={test_acc:.2f}% < best_acc={best.best_acc:.2f}% - {args.rollback_drop_abs}%. "
                    f"Rolling back to best and shrinking LR."
                )
                aux_freeze_until_epoch = epoch + args.rollback_freeze_aux_epochs
                unpack_state(best.best_blob, model, optimizer, scaler)
                lr_shrink = float(args.rollback_shrink)
                lr_min = float(getattr(args, "lr_min", 0.0))
                for pg in optimizer.param_groups:
                    pg["lr"] = max(lr_min, float(pg["lr"]) * lr_shrink)
                shrink_scheduler_base_lrs(scheduler, factor=lr_shrink, lr_min=lr_min)
                # reset scaler to avoid repeated AMP skipped-steps after rollback
                if device.type == "cuda":
                    use_amp = bool(args.amp)
                    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=float(getattr(args, "amp_init_scale", 2.0**6)))
                else:
                    use_amp = False
                    scaler = None
                aux_freeze_until_epoch = max(aux_freeze_until_epoch, epoch + int(getattr(args, "rollback_freeze_aux_epochs", 5)))
                lr_now = float(optimizer.param_groups[0]["lr"])
                print(f"  >>> Rollback done. lr_now={lr_now:.6g}, freeze_aux_until={aux_freeze_until_epoch}")

        # epoch-wise scheduler update (warmup + cosine annealing)
        scheduler.step()

    print(f"Training done. Best Acc={best.best_acc:.2f}% at epoch {best.best_epoch}. Saved to {args.save_path}")


if __name__ == "__main__":
    main()

