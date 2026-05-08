import math
import time
import argparse
import inspect
import dataclasses
import random
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from dataset import WiFiRFFIDataset
except Exception:
    WiFiRFFIDataset = None

try:
    from dataset_wisig_modified import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test
except Exception:
    from dataset_wisig import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test

from model_dual_cvsincnet import build_dual_model

try:
    from DataAugmentation_v2 import build_augmentor
except Exception:
    build_augmentor = None

try:
    from sat_channel import SatSimConfig, apply_sat_gnd_channel_batch
except Exception:
    SatSimConfig = None
    apply_sat_gnd_channel_batch = None


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
        self.sum += float(v) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.count)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr_scale: float = 0.05):
        self.optimizer = optimizer
        self.epochs = max(1, int(epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_scale = float(min_lr_scale)
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def get_scale(self, epoch: int) -> float:
        epoch = int(epoch)
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            return max(1e-6, float(epoch) / float(self.warmup_epochs))
        if self.epochs <= self.warmup_epochs:
            return 1.0
        t = float(epoch - self.warmup_epochs) / float(max(1, self.epochs - self.warmup_epochs))
        t = min(1.0, max(0.0, t))
        cosv = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosv

    def step(self, epoch: int):
        scale = self.get_scale(epoch)
        for base_lr, pg in zip(self.base_lrs, self.optimizer.param_groups):
            pg["lr"] = base_lr * scale


def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0


def fp32_zero(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=ref.device, dtype=torch.float32)


def sanitize_batch_tensor(x: torch.Tensor, abs_clip: float = 0.0) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if abs_clip > 0:
        x = x.clamp(min=-abs_clip, max=abs_clip)
    return x


def renorm_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim != 3 or x.size(1) < 2:
        return x
    p = (x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(p + eps)


def tensor_is_finite(x: Optional[torch.Tensor]) -> bool:
    return torch.is_tensor(x) and bool(torch.isfinite(x).all().item())


def find_nonfinite_tensors(named_tensors: Dict[str, Optional[torch.Tensor]]) -> Tuple[str, ...]:
    bad = []
    for name, value in named_tensors.items():
        if torch.is_tensor(value) and (not tensor_is_finite(value)):
            bad.append(name)
    return tuple(bad)


def sanitize_gradients(model: nn.Module, value_clip: float = 5.0, limit: int = 8) -> Tuple[str, ...]:
    repaired = []
    do_clip = float(value_clip) > 0
    clip_val = float(value_clip)
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        had_bad = not torch.isfinite(grad).all()
        if had_bad:
            if len(repaired) < limit:
                repaired.append(name)
            param.grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad = param.grad
        if do_clip:
            grad.clamp_(min=-clip_val, max=clip_val)
    return tuple(repaired)


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


def extract_meta_from_extra(extra) -> Optional[Dict[str, Any]]:
    if extra is None or len(extra) < 2:
        return None
    meta = extra[1]
    return meta if isinstance(meta, dict) else None


def meta_get_tensor(meta: Optional[Dict[str, Any]], key: str, device) -> Optional[torch.Tensor]:
    if meta is None or key not in meta:
        return None
    value = meta[key]
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True).view(-1)
    try:
        return torch.as_tensor(value, device=device).view(-1)
    except Exception:
        return None


def get_nuisance_target(extra, device, target: str = "auto") -> Optional[torch.Tensor]:
    target = str(target).lower().strip()
    d = extract_domain_from_extra(extra, device)
    meta = extract_meta_from_extra(extra)
    rx = meta_get_tensor(meta, "rx_i", device)
    day = meta_get_tensor(meta, "day_i", device)
    if target == "rx":
        return rx if rx is not None else d
    if target == "day":
        return day if day is not None else d
    if target == "domain":
        return d
    if target == "auto":
        return rx if rx is not None else d
    return d


def infer_nuisance_classes(train_ds, target: str = "auto") -> int:
    target = str(target).lower().strip()
    index = getattr(train_ds, "index", None)
    if index is not None and len(index) > 0:
        if target == "rx":
            return len(sorted({int(it.rx_i) for it in index}))
        if target == "day":
            return len(sorted({int(it.day_i) for it in index}))
        if target == "auto":
            vals = sorted({int(it.rx_i) for it in index})
            if len(vals) > 0:
                return len(vals)
        if hasattr(train_ds, "_domain_lut"):
            return len(sorted({int(train_ds._domain_lut[(it.rx_i, it.day_i)]) for it in index}))
    if hasattr(train_ds, "rx_list") and target in ("rx", "auto"):
        return max(1, len(getattr(train_ds, "rx_list")))
    if hasattr(train_ds, "day_list") and target == "day":
        return max(1, len(getattr(train_ds, "day_list")))
    return 1


def covariance_decorrelation_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean(dim=0, keepdim=True)
    b = b.float() - b.float().mean(dim=0, keepdim=True)
    n = a.size(0)
    if n <= 1:
        return fp32_zero(a)
    cov = (a.t() @ b) / float(n - 1)
    return torch.mean(cov * cov)


def same_tx_diff_nuisance_consistency(z: torch.Tensor, y: torch.Tensor, nlab: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    if nlab is None:
        return fp32_zero(z), float("nan")
    z = F.normalize(z.float(), dim=1, eps=1e-4)
    y = y.view(-1)
    nlab = nlab.view(-1)
    losses: List[torch.Tensor] = []
    sims = []
    for cls in torch.unique(y):
        m_cls = y == cls
        nuis = torch.unique(nlab[m_cls])
        if nuis.numel() < 2:
            continue
        cents = []
        for nv in nuis:
            m = m_cls & (nlab == nv)
            if int(m.sum()) == 0:
                continue
            cents.append(F.normalize(z[m].mean(dim=0, keepdim=True), dim=1, eps=1e-4).squeeze(0))
        if len(cents) < 2:
            continue
        c = torch.stack(cents, dim=0)
        sim = c @ c.t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        pair_sim = sim[iu[0], iu[1]]
        losses.append((1.0 - pair_sim).mean())
        sims.append(pair_sim.mean().item())
    if len(losses) == 0:
        return fp32_zero(z), float("nan")
    return torch.stack(losses).mean(), float(np.mean(sims))


def same_nuisance_diff_tx_consistency(z: torch.Tensor, nlab: Optional[torch.Tensor], y: torch.Tensor) -> Tuple[torch.Tensor, float]:
    if nlab is None:
        return fp32_zero(z), float("nan")
    z = F.normalize(z.float(), dim=1, eps=1e-4)
    y = y.view(-1)
    nlab = nlab.view(-1)
    losses: List[torch.Tensor] = []
    sims = []
    for nv in torch.unique(nlab):
        m_nv = nlab == nv
        txs = torch.unique(y[m_nv])
        if txs.numel() < 2:
            continue
        cents = []
        for tx in txs:
            m = m_nv & (y == tx)
            if int(m.sum()) == 0:
                continue
            cents.append(F.normalize(z[m].mean(dim=0, keepdim=True), dim=1, eps=1e-4).squeeze(0))
        if len(cents) < 2:
            continue
        c = torch.stack(cents, dim=0)
        sim = c @ c.t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        pair_sim = sim[iu[0], iu[1]]
        losses.append((1.0 - pair_sim).mean())
        sims.append(pair_sim.mean().item())
    if len(losses) == 0:
        return fp32_zero(z), float("nan")
    return torch.stack(losses).mean(), float(np.mean(sims))


def center_compact_loss(z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
    if labels is None:
        return fp32_zero(z)
    z = F.normalize(z.float(), dim=1, eps=1e-4)
    labels = labels.view(-1)
    uniq = torch.unique(labels)
    losses = []
    for cls in uniq:
        m = labels == cls
        if int(m.sum()) < 2:
            continue
        c = F.normalize(z[m].mean(dim=0, keepdim=True), dim=1, eps=1e-4)
        losses.append(1.0 - (z[m] * c).sum(dim=1).mean())
    if len(losses) == 0:
        return fp32_zero(z)
    return torch.stack(losses).mean()


def batch_corrcoef(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return float("nan")
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    n = min(x.numel(), y.numel())
    if n < 2:
        return float("nan")
    x = x[:n] - x[:n].mean()
    y = y[:n] - y[:n].mean()
    den = x.norm().item() * y.norm().item()
    if den <= 1e-12:
        return float("nan")
    return float((x * y).sum().item() / den)


def paired_cosine_consistency(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, float]:
    a = F.normalize(a.float(), dim=1, eps=1e-4)
    b = F.normalize(b.float(), dim=1, eps=1e-4)
    cos = (a * b).sum(dim=1)
    return (1.0 - cos).mean(), float(cos.mean().item())


def smoothstep01(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def ramp_between(r: float, start: float, end: float, low: float = 0.0, high: float = 1.0) -> float:
    if r <= start:
        return float(low)
    if r >= end:
        return float(high)
    t = (float(r) - float(start)) / max(1e-8, float(end) - float(start))
    return float(low) + (float(high) - float(low)) * smoothstep01(t)


def decay_between(r: float, start: float, end: float, high: float = 1.0, low: float = 0.0) -> float:
    return ramp_between(r, start, end, low=high, high=low)


def stage_scales(epoch: int, epochs: int) -> Dict[str, float]:
    r = float(epoch - 1) / max(1.0, float(epochs - 1))
    return {
        "rx": ramp_between(r, 0.03, 0.28, low=0.35, high=1.00),
        "adv_rx": ramp_between(r, 0.18, 0.68, low=0.00, high=1.00),
        "adv_tx": ramp_between(r, 0.24, 0.84, low=0.00, high=1.00),
        "grl_rx": ramp_between(r, 0.20, 0.72, low=0.00, high=1.00),
        "grl_tx": ramp_between(r, 0.26, 0.88, low=0.00, high=1.10),
        "sep": ramp_between(r, 0.10, 0.68, low=0.00, high=1.00),
        "center": ramp_between(r, 0.04, 0.40, low=0.10, high=1.00),
        "cons_tx": ramp_between(r, 0.12, 0.55, low=0.00, high=1.00),
        "cons_rx": ramp_between(r, 0.18, 0.72, low=0.00, high=1.00),
        "probe_rx": ramp_between(r, 0.10, 0.52, low=0.00, high=1.00),
        "probe_tx": ramp_between(r, 0.18, 0.78, low=0.00, high=1.00),
        "proxy": decay_between(r, 0.10, 0.88, high=0.70, low=0.20),
        "sat_cls": ramp_between(r, 0.04, 0.40, low=0.05, high=0.70),
        "sat_cons": ramp_between(r, 0.08, 0.62, low=0.00, high=1.00),
        "sat_mix": ramp_between(r, 0.00, 0.22, low=0.20, high=1.00),
    }


def set_sinc_trainable(model: nn.Module, trainable: bool = True):
    changed = 0
    for name, param in model.named_parameters():
        if "id_backbone.sinc." in name:
            param.requires_grad = bool(trainable)
            changed += 1
    return changed


@torch.no_grad()
def evaluate(model, loader, device, nuisance_target: str = "auto", num_nuisance_train: int = 1, max_batches: int = 0):
    model.eval()
    meters = {k: AverageMeter() for k in [
        "tx_acc", "rx_acc", "probe_rx_acc", "probe_tx_acc"
    ]}
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)
        nlab = get_nuisance_target(extra, device, target=nuisance_target)
        out = model(x, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True)
        meters["tx_acc"].update(accuracy_from_logits(out["tx_logits"], y), x.size(0))
        if nlab is not None and int(num_nuisance_train) > 1:
            meters["rx_acc"].update(accuracy_from_logits(out["rx_logits"], nlab), x.size(0))
            meters["probe_rx_acc"].update(accuracy_from_logits(out["probe_rx_logits"], nlab), x.size(0))
        meters["probe_tx_acc"].update(accuracy_from_logits(out["probe_tx_logits"], y), x.size(0))
    return {k: v.avg for k, v in meters.items()}



def make_sat_cfg(args) -> Optional[SatSimConfig]:
    if SatSimConfig is None or apply_sat_gnd_channel_batch is None:
        return None

    def _map_loo_level(v: str) -> str:
        s = str(v).lower().strip()
        lut = {
            "low": "light",
            "light": "light",
            "mid": "mid",
            "medium": "mid",
            "high": "severe",
            "severe": "severe",
        }
        return lut.get(s, s)

    base_kwargs = {
        "fc_hz": float(args.sat_fc_hz),
        "scenario": str(args.sat_scenario),
        "weather": str(args.sat_weather),
        "loo_level": _map_loo_level(args.sat_loo_level),
        "theta_deg": (float(args.sat_theta_min), float(args.sat_theta_max)),
        "snr_db": (float(args.sat_snr_min), float(args.sat_snr_max)),
        "cfo_std_hz": float(args.sat_cfo_std_hz),
        "phase_noise_inc_std": (float(args.sat_pn_min), float(args.sat_pn_max)),
        "iq_amp_db": (float(args.sat_iq_amp_min_db), float(args.sat_iq_amp_max_db)),
        "iq_phase_deg": (float(args.sat_iq_phase_min_deg), float(args.sat_iq_phase_max_deg)),
        "agc_resid_db": (float(args.sat_agc_min_db), float(args.sat_agc_max_db)),
        "K_db_range": (float(args.sat_K_min_db), float(args.sat_K_max_db)),
        "enable_multipath": bool(args.sat_enable_multipath),
        "num_taps": (int(args.sat_num_taps_min), int(args.sat_num_taps_max)),
        "max_delay_samp": int(args.sat_max_delay_samp),
        "pwr_decay": float(args.sat_pwr_decay),
        "markov_alpha": float(args.sat_markov_alpha),
    }

    # Compatibility with multiple sat_channel.py variants:
    # older/newer versions may name sampling rate as fs_hz or sample_rate_hz.
    try:
        if dataclasses.is_dataclass(SatSimConfig):
            field_names = {f.name for f in dataclasses.fields(SatSimConfig)}
        else:
            field_names = set(inspect.signature(SatSimConfig).parameters.keys())
    except Exception:
        field_names = set()

    if "fs_hz" in field_names:
        base_kwargs["fs_hz"] = float(args.sample_rate_hz)
    elif "sample_rate_hz" in field_names:
        base_kwargs["sample_rate_hz"] = float(args.sample_rate_hz)
    else:
        # last resort: try common aliases in order
        for k in ("fs_hz", "sample_rate_hz"):
            try:
                return SatSimConfig(**{k: float(args.sample_rate_hz), **base_kwargs})
            except TypeError:
                pass
        return SatSimConfig(**base_kwargs)

    return SatSimConfig(**base_kwargs)


def maybe_build_augmentor(sample_rate_hz: float):
    if build_augmentor is None:
        return None
    try:
        return build_augmentor(sample_rate_hz=sample_rate_hz)
    except TypeError:
        return build_augmentor()


def blend_sat_view(x: torch.Tensor, x_sat: torch.Tensor, alpha: float, abs_clip: float = 0.0) -> torch.Tensor:
    alpha = min(1.0, max(0.0, float(alpha)))
    if alpha <= 0.0:
        return sanitize_batch_tensor(x, abs_clip=abs_clip)
    y = (1.0 - alpha) * x.float() + alpha * x_sat.float()
    y = renorm_rms(y)
    return sanitize_batch_tensor(y, abs_clip=abs_clip)


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    groups = {
        "id_rest": [],
        "id_sinc": [],
        "rx_backbone": [],
        "rx_head": [],
        "adv_probe": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "id_backbone.sinc." in name:
            groups["id_sinc"].append(p)
        elif name.startswith("id_backbone.") or name.startswith("tx_adapter."):
            groups["id_rest"].append(p)
        elif name.startswith("rx_backbone.") or name.startswith("rx_adapter."):
            groups["rx_backbone"].append(p)
        elif name.startswith("rx_head."):
            groups["rx_head"].append(p)
        else:
            groups["adv_probe"].append(p)

    param_groups = []
    if groups["id_rest"]:
        param_groups.append({"params": groups["id_rest"], "lr": args.lr, "weight_decay": args.weight_decay})
    if groups["id_sinc"]:
        param_groups.append({"params": groups["id_sinc"], "lr": args.lr * args.sinc_lr_scale, "weight_decay": args.weight_decay})
    if groups["rx_backbone"]:
        param_groups.append({"params": groups["rx_backbone"], "lr": args.lr * args.rx_lr_scale, "weight_decay": args.weight_decay})
    if groups["rx_head"]:
        param_groups.append({"params": groups["rx_head"], "lr": args.lr * args.head_lr_scale, "weight_decay": args.weight_decay})
    if groups["adv_probe"]:
        param_groups.append({"params": groups["adv_probe"], "lr": args.lr * args.adv_head_lr_scale, "weight_decay": args.weight_decay})

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wisig", choices=["oralce", "wisig"])
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_eq", type=str, default="1")
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_domain", type=str, default="day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_train_days", type=str, default="0,1,2")
    parser.add_argument("--wisig_full_test_days", type=str, default="3")
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--sample_rate_hz", type=float, default=25e6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sinc_lr_scale", type=float, default=0.10)
    parser.add_argument("--rx_lr_scale", type=float, default=0.75)
    parser.add_argument("--head_lr_scale", type=float, default=0.70)
    parser.add_argument("--adv_head_lr_scale", type=float, default=0.50)
    parser.add_argument("--freeze_sinc_epochs", type=int, default=8)
    parser.add_argument("--rx_dim", type=int, default=128)
    parser.add_argument("--rx_branch_dim", type=int, default=64)

    parser.add_argument("--lambda_rx", type=float, default=1.0)
    parser.add_argument("--lambda_adv_rx", type=float, default=0.10)
    parser.add_argument("--lambda_adv_tx", type=float, default=0.16)
    parser.add_argument("--lambda_sep", type=float, default=0.025)
    parser.add_argument("--lambda_center_rx", type=float, default=0.05)
    parser.add_argument("--lambda_cons_tx", type=float, default=0.08)
    parser.add_argument("--lambda_cons_rx", type=float, default=0.08)
    parser.add_argument("--lambda_probe_rx", type=float, default=0.015)
    parser.add_argument("--lambda_probe_tx", type=float, default=0.040)
    parser.add_argument("--lambda_pa_proxy", type=float, default=0.18)
    parser.add_argument("--lambda_dac_proxy", type=float, default=0.15)

    parser.add_argument("--nuisance_target", type=str, default="auto", choices=["auto", "domain", "rx", "day"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="best_model_dual_rxadc_v3_satstable.pth")

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--amp_warmup_epochs", type=int, default=10)
    parser.add_argument("--grad_value_clip", type=float, default=1.5)
    parser.add_argument("--grad_norm_clip", type=float, default=0.8)
    parser.add_argument("--input_abs_clip", type=float, default=3.5)
    parser.add_argument("--grl_lambda", type=float, default=1.0)
    parser.add_argument("--max_loss_value", type=float, default=100.0)
    parser.add_argument("--skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_true")
    parser.add_argument("--no_skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_false")
    parser.set_defaults(skip_nonfinite_batches=True)
    parser.add_argument("--nonfinite_log_limit", type=int, default=8)

    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)

    parser.add_argument("--tx_label_smoothing", type=float, default=0.02)
    parser.add_argument("--rx_label_smoothing", type=float, default=0.02)

    parser.add_argument("--sat_start_epoch", type=int, default=15)
    parser.add_argument("--sat_tx_weight", type=float, default=0.30)
    parser.add_argument("--lambda_sat_cons_tx", type=float, default=0.12)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--sat_scenario", type=str, default="urban", choices=["urban", "suburban"])
    parser.add_argument("--sat_weather", type=str, default="clear", choices=["clear", "cloudy", "rain", "storm"])
    parser.add_argument("--sat_loo_level", type=str, default="mid", choices=["low", "light", "mid", "high", "severe"])
    parser.add_argument("--sat_theta_min", type=float, default=15.0)
    parser.add_argument("--sat_theta_max", type=float, default=85.0)
    parser.add_argument("--sat_snr_min", type=float, default=12.0)
    parser.add_argument("--sat_snr_max", type=float, default=28.0)
    parser.add_argument("--sat_cfo_std_hz", type=float, default=150.0)
    parser.add_argument("--sat_pn_min", type=float, default=0.0)
    parser.add_argument("--sat_pn_max", type=float, default=1.5e-3)
    parser.add_argument("--sat_iq_amp_min_db", type=float, default=-0.25)
    parser.add_argument("--sat_iq_amp_max_db", type=float, default=0.25)
    parser.add_argument("--sat_iq_phase_min_deg", type=float, default=-2.0)
    parser.add_argument("--sat_iq_phase_max_deg", type=float, default=2.0)
    parser.add_argument("--sat_agc_min_db", type=float, default=-0.75)
    parser.add_argument("--sat_agc_max_db", type=float, default=0.75)
    parser.add_argument("--sat_K_min_db", type=float, default=2.0)
    parser.add_argument("--sat_K_max_db", type=float, default=16.0)
    parser.add_argument("--sat_enable_multipath", dest="sat_enable_multipath", action="store_true")
    parser.add_argument("--no_sat_enable_multipath", dest="sat_enable_multipath", action="store_false")
    parser.set_defaults(sat_enable_multipath=False)
    parser.add_argument("--sat_num_taps_min", type=int, default=2)
    parser.add_argument("--sat_num_taps_max", type=int, default=4)
    parser.add_argument("--sat_max_delay_samp", type=int, default=4)
    parser.add_argument("--sat_pwr_decay", type=float, default=0.85)
    parser.add_argument("--sat_markov_alpha", type=float, default=0.0)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(time.strftime("Starting Training at %Y-%m-%d %H:%M:%S"))
    print(f"Device: {device} | AMP: {bool(args.amp)}")

    split_info = None
    if args.dataset.lower() == "wisig":
        tx_days = [int(s) for s in args.wisig_train_days.split(",") if str(s).strip() != ""]
        test_days = [int(s) for s in args.wisig_full_test_days.split(",") if str(s).strip() != ""]
        ds = load_wisig_compact_pkl(args.wisig_pkl)
        train_ds, test_ds, split_info = make_day123_randomsplit_plus_day4_test(
            ds,
            out_len=args.wisig_out_len,
            crop_mode="center",
            normalize=True,
            equalized=args.wisig_eq,
            domain=args.wisig_domain,
            train_days=tx_days,
            full_test_days=test_days,
            train_ratio=0.8,
            seed=args.seed,
        )
        num_classes = len(getattr(train_ds, "tx_list", [])) if hasattr(train_ds, "tx_list") else 6
        print(f"[WISIG] overriding num_classes 16 -> {num_classes}")
        print(f"[WISIG] pkl={args.wisig_pkl} eq={args.wisig_eq} out_len={args.wisig_out_len} domain={args.wisig_domain}")
        if split_info is not None:
            print(f"[WISIG] TRAIN SOURCES: {split_info.get('train_days_label')} -> random 80%")
            print(f"[WISIG] TEST SOURCES : same days remaining 20% + full {split_info.get('full_test_days_label')}")
            print(f"[WISIG] split_info={split_info}")
    else:
        if WiFiRFFIDataset is None:
            raise RuntimeError("dataset.WiFiRFFIDataset not available, cannot run ORALCE mode.")
        train_ds = WiFiRFFIDataset(args.dataset_dir, split="train")
        test_ds = WiFiRFFIDataset(args.dataset_dir, split="test")
        num_classes = int(getattr(train_ds, "num_classes", 16))

    num_nuisance = infer_nuisance_classes(train_ds, target=args.nuisance_target)
    print(f"[NUISANCE] target={args.nuisance_target} classes={num_nuisance}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = build_dual_model(
        num_classes=num_classes,
        num_domains=num_nuisance,
        model_size=args.model_size,
        dataset=args.dataset,
        input_len=args.wisig_out_len if args.dataset.lower() == "wisig" else 1024,
        sample_rate_hz=args.sample_rate_hz,
        rx_dim=args.rx_dim,
        rx_branch_dim=args.rx_branch_dim,
    ).to(device)
    print(f"[MODEL] DualRxADCDisentangle tx_dim={model.tx_dim} rx_dim={model.rx_dim}")

    optimizer = make_optimizer(model, args)
    scheduler = WarmupCosineScheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr_scale=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    ce_tx = nn.CrossEntropyLoss(label_smoothing=float(args.tx_label_smoothing))
    ce_rx = nn.CrossEntropyLoss(label_smoothing=float(args.rx_label_smoothing))
    augmentor_main = maybe_build_augmentor(sample_rate_hz=args.sample_rate_hz) if args.augment else None
    augmentor_proxy = maybe_build_augmentor(sample_rate_hz=args.sample_rate_hz) if args.augment else None
    print("[AUG] enabled." if augmentor_main is not None and augmentor_proxy is not None else "[AUG] disabled.")

    sat_cfg = make_sat_cfg(args)
    if sat_cfg is None:
        raise RuntimeError("Satellite view is required now, but sat_channel.py or its API is unavailable.")
    print(
        f"[SAT] always enabled. scenario={args.sat_scenario} weather={args.sat_weather} "
        f"theta=({args.sat_theta_min:.1f},{args.sat_theta_max:.1f}) snr=({args.sat_snr_min:.1f},{args.sat_snr_max:.1f}) "
        f"start_epoch={args.sat_start_epoch}"
    )

    best_tx = -1.0
    best_epoch = -1
    nonfinite_logs = 0

    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch)
        if epoch <= int(args.freeze_sinc_epochs):
            set_sinc_trainable(model, trainable=False)
        elif epoch == int(args.freeze_sinc_epochs) + 1:
            set_sinc_trainable(model, trainable=True)

        model.train()
        t0 = time.time()
        scale = stage_scales(epoch, args.epochs)
        use_amp_epoch = bool(args.amp) and (epoch > int(args.amp_warmup_epochs))
        meters = {k: AverageMeter() for k in [
            "loss", "tx", "rx", "adv_rx", "adv_tx", "sep", "center", "cons_tx", "cons_rx",
            "probe_rx", "probe_tx", "pa_proxy", "dac_proxy", "sat_tx", "sat_cons",
            "tx_acc", "rx_acc", "probe_rx_acc", "probe_tx_acc", "sat_tx_acc", "corr_pa", "corr_dac", "gnorm"
        ]}
        cons_tx_cos_vals = []
        cons_rx_cos_vals = []
        sat_cos_vals = []
        nonfinite_batches = 0

        for batch in train_loader:
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long().view(-1)
            nlab = get_nuisance_target(extra, device, target=args.nuisance_target)

            clip_val = float(args.input_abs_clip)
            if augmentor_main is not None and augmentor_proxy is not None:
                x_main = augmentor_main(x, labels=y, no_dac=True, no_pa=True)
                x_pa, s_p = augmentor_proxy(x, labels=y, pa_only=True, return_pa_strength=True)
                x_dac, s_d = augmentor_proxy(x, labels=y, dac_only=True, return_dac_strength=True)
            else:
                x_main = x
                x_pa = x
                x_dac = x
                s_p = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
                s_d = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)

            use_sat_batch = epoch >= int(args.sat_start_epoch)
            if use_sat_batch:
                x_sat_raw, _, _ = apply_sat_gnd_channel_batch(x, sat_cfg, return_meta=False)
                x_sat = blend_sat_view(x, x_sat_raw, alpha=scale["sat_mix"], abs_clip=clip_val)
            else:
                x_sat = sanitize_batch_tensor(x, abs_clip=clip_val)

            x_main = renorm_rms(sanitize_batch_tensor(x_main, abs_clip=clip_val))
            x_pa = renorm_rms(sanitize_batch_tensor(x_pa, abs_clip=clip_val))
            x_dac = renorm_rms(sanitize_batch_tensor(x_dac, abs_clip=clip_val))
            x_sat = renorm_rms(x_sat)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp_epoch):
                grl_rx = float(args.grl_lambda) * float(scale["grl_rx"])
                grl_tx = float(args.grl_lambda) * float(scale["grl_tx"])
                out_main = model(x_main, y_tx=y, grl_rx_on_tx=grl_rx, grl_tx_on_rx=grl_tx, return_aux=True)
                out_pa = model(x_pa, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True)
                out_dac = model(x_dac, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True)
                out_sat = model(x_sat, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0, return_aux=True) if use_sat_batch else None

                tx_logits = out_main["tx_logits"]
                rx_logits = out_main["rx_logits"]
                adv_rx_logits = out_main["adv_rx_logits"]
                adv_tx_logits = out_main["adv_tx_logits"]
                probe_rx_logits = out_main["probe_rx_logits"]
                probe_tx_logits = out_main["probe_tx_logits"]
                z_tx = out_main["z_tx"]
                z_rx = out_main["z_rx"]

                pa_pred_pa = out_pa["aux_id"].get("pa_pred", None)
                dac_pred_dac = out_dac["aux_id"].get("dac_pred", None)
                sat_tx_logits = out_sat["tx_logits"] if out_sat is not None else None
                z_tx_sat = out_sat["z_tx"] if out_sat is not None else None

                loss_ref = tx_logits.float().mean()
                loss_tx = ce_tx(tx_logits.float(), y)

                if nlab is not None and int(num_nuisance) > 1:
                    loss_rx = ce_rx(rx_logits.float(), nlab)
                    loss_adv_rx = ce_rx(adv_rx_logits.float(), nlab)
                    loss_probe_rx = ce_rx(probe_rx_logits.float(), nlab)
                    loss_center = center_compact_loss(z_rx, nlab)
                    loss_cons_tx, cons_tx_cos = same_tx_diff_nuisance_consistency(z_tx, y, nlab)
                    loss_cons_rx, cons_rx_cos = same_nuisance_diff_tx_consistency(z_rx, nlab, y)
                    if not math.isnan(cons_tx_cos):
                        cons_tx_cos_vals.append(cons_tx_cos)
                    if not math.isnan(cons_rx_cos):
                        cons_rx_cos_vals.append(cons_rx_cos)
                    meters["rx_acc"].update(accuracy_from_logits(rx_logits, nlab), x.size(0))
                    meters["probe_rx_acc"].update(accuracy_from_logits(probe_rx_logits, nlab), x.size(0))
                else:
                    loss_rx = fp32_zero(loss_ref)
                    loss_adv_rx = fp32_zero(loss_ref)
                    loss_probe_rx = fp32_zero(loss_ref)
                    loss_center = fp32_zero(loss_ref)
                    loss_cons_tx = fp32_zero(loss_ref)
                    loss_cons_rx = fp32_zero(loss_ref)
                    cons_tx_cos = float("nan")
                    cons_rx_cos = float("nan")

                loss_adv_tx = ce_tx(adv_tx_logits.float(), y)
                loss_probe_tx = ce_tx(probe_tx_logits.float(), y)
                loss_sep = covariance_decorrelation_loss(z_tx, z_rx)

                if torch.is_tensor(pa_pred_pa):
                    loss_pa_proxy = F.mse_loss(pa_pred_pa.float().view(-1), s_p.float().view(-1))
                else:
                    loss_pa_proxy = fp32_zero(loss_ref)
                if torch.is_tensor(dac_pred_dac):
                    loss_dac_proxy = F.mse_loss(dac_pred_dac.float().view(-1), s_d.float().view(-1))
                else:
                    loss_dac_proxy = fp32_zero(loss_ref)

                if out_sat is not None and torch.is_tensor(sat_tx_logits) and torch.is_tensor(z_tx_sat):
                    loss_sat_tx = ce_tx(sat_tx_logits.float(), y)
                    loss_sat_cons, sat_cos = paired_cosine_consistency(z_tx, z_tx_sat)
                    sat_cos_vals.append(sat_cos)
                    meters["sat_tx_acc"].update(accuracy_from_logits(sat_tx_logits, y), x.size(0))
                else:
                    loss_sat_tx = fp32_zero(loss_ref)
                    loss_sat_cons = fp32_zero(loss_ref)
                    sat_cos = float("nan")

                total = (
                    loss_tx
                    + float(args.lambda_rx) * scale["rx"] * loss_rx
                    + float(args.lambda_adv_rx) * scale["adv_rx"] * loss_adv_rx
                    + float(args.lambda_adv_tx) * scale["adv_tx"] * loss_adv_tx
                    + float(args.lambda_sep) * scale["sep"] * loss_sep
                    + float(args.lambda_center_rx) * scale["center"] * loss_center
                    + float(args.lambda_cons_tx) * scale["cons_tx"] * loss_cons_tx
                    + float(args.lambda_cons_rx) * scale["cons_rx"] * loss_cons_rx
                    + float(args.lambda_probe_rx) * scale["probe_rx"] * loss_probe_rx
                    + float(args.lambda_probe_tx) * scale["probe_tx"] * loss_probe_tx
                    + float(args.lambda_pa_proxy) * scale["proxy"] * loss_pa_proxy
                    + float(args.lambda_dac_proxy) * scale["proxy"] * loss_dac_proxy
                    + float(args.sat_tx_weight) * scale["sat_cls"] * loss_sat_tx
                    + float(args.lambda_sat_cons_tx) * scale["sat_cons"] * loss_sat_cons
                )
                total = total.float()
                bad = find_nonfinite_tensors({
                    "loss": total,
                    "tx_logits": tx_logits,
                    "rx_logits": rx_logits,
                    "adv_rx_logits": adv_rx_logits,
                    "adv_tx_logits": adv_tx_logits,
                    "z_tx": z_tx,
                    "z_rx": z_rx,
                    "sat_tx_logits": sat_tx_logits if out_sat is not None else None,
                    "z_tx_sat": z_tx_sat if out_sat is not None else None,
                })
                loss = total if len(bad) == 0 and float(total.detach().abs().item()) <= float(args.max_loss_value) else loss_ref.new_tensor(float("nan"))

            if not tensor_is_finite(loss):
                nonfinite_batches += 1
                if nonfinite_logs < int(args.nonfinite_log_limit):
                    print(f"[WARN][E{epoch:03d}] skip batch due to non-finite or exploding loss", flush=True)
                    nonfinite_logs += 1
                if bool(args.skip_nonfinite_batches):
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            repaired = sanitize_gradients(model, value_clip=float(args.grad_value_clip))
            if len(repaired) > 0 and nonfinite_logs < int(args.nonfinite_log_limit):
                print(f"[WARN][E{epoch:03d}] repaired non-finite grads: {', '.join(repaired)}", flush=True)
                nonfinite_logs += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_norm_clip))
            if torch.is_tensor(grad_norm):
                grad_norm = float(grad_norm.item())
            if not math.isfinite(float(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                nonfinite_batches += 1
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

            bsz = x.size(0)
            meters["loss"].update(loss.item(), bsz)
            meters["tx"].update(loss_tx.item(), bsz)
            meters["rx"].update(loss_rx.item(), bsz)
            meters["adv_rx"].update(loss_adv_rx.item(), bsz)
            meters["adv_tx"].update(loss_adv_tx.item(), bsz)
            meters["sep"].update(loss_sep.item(), bsz)
            meters["center"].update(loss_center.item(), bsz)
            meters["cons_tx"].update(loss_cons_tx.item(), bsz)
            meters["cons_rx"].update(loss_cons_rx.item(), bsz)
            meters["probe_rx"].update(loss_probe_rx.item(), bsz)
            meters["probe_tx"].update(loss_probe_tx.item(), bsz)
            meters["pa_proxy"].update(loss_pa_proxy.item(), bsz)
            meters["dac_proxy"].update(loss_dac_proxy.item(), bsz)
            meters["sat_tx"].update(loss_sat_tx.item(), bsz)
            meters["sat_cons"].update(loss_sat_cons.item(), bsz)
            meters["tx_acc"].update(accuracy_from_logits(tx_logits, y), bsz)
            meters["probe_tx_acc"].update(accuracy_from_logits(probe_tx_logits, y), bsz)
            meters["gnorm"].update(grad_norm, 1)
            corr_pa = batch_corrcoef(pa_pred_pa, s_p)
            corr_dac = batch_corrcoef(dac_pred_dac, s_d)
            if not math.isnan(corr_pa):
                meters["corr_pa"].update(corr_pa, bsz)
            if not math.isnan(corr_dac):
                meters["corr_dac"].update(corr_dac, bsz)

        cons_tx_cos_epoch = float(np.mean(cons_tx_cos_vals)) if len(cons_tx_cos_vals) > 0 else float("nan")
        cons_rx_cos_epoch = float(np.mean(cons_rx_cos_vals)) if len(cons_rx_cos_vals) > 0 else float("nan")
        sat_cos_epoch = float(np.mean(sat_cos_vals)) if len(sat_cos_vals) > 0 else float("nan")

        eval_stats = None
        if epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            eval_stats = evaluate(
                model,
                test_loader,
                device,
                nuisance_target=args.nuisance_target,
                num_nuisance_train=num_nuisance,
                max_batches=int(args.eval_max_batches),
            )
            if eval_stats["tx_acc"] > best_tx:
                best_tx = eval_stats["tx_acc"]
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_tx_acc": best_tx,
                        "args": vars(args),
                        "split_info": split_info,
                        "num_nuisance": num_nuisance,
                    },
                    args.save_path,
                )

        msg = (
            f"[E{epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e} loss={meters['loss'].avg:.4f} "
            f"tx={meters['tx'].avg:.4f} rx={meters['rx'].avg:.4f} adv_rx={meters['adv_rx'].avg:.4f} adv_tx={meters['adv_tx'].avg:.4f} "
            f"sep={meters['sep'].avg:.4f} center={meters['center'].avg:.4f} cons_tx={meters['cons_tx'].avg:.4f} cons_rx={meters['cons_rx'].avg:.4f} "
            f"probe_rx={meters['probe_rx'].avg:.4f} probe_tx={meters['probe_tx'].avg:.4f} "
            f"pa_proxy={meters['pa_proxy'].avg:.4f} dac_proxy={meters['dac_proxy'].avg:.4f} "
            f"sat_tx={meters['sat_tx'].avg:.4f} sat_cons={meters['sat_cons'].avg:.4f} | "
            f"tx_acc={meters['tx_acc'].avg:.2f}% rx_acc={meters['rx_acc'].avg if meters['rx_acc'].count else float('nan'):.2f}% "
            f"probe_rx_acc={meters['probe_rx_acc'].avg if meters['probe_rx_acc'].count else float('nan'):.2f}% "
            f"probe_tx_acc={meters['probe_tx_acc'].avg:.2f}% sat_tx_acc={meters['sat_tx_acc'].avg if meters['sat_tx_acc'].count else float('nan'):.2f}% "
            f"cons_tx_cos={cons_tx_cos_epoch:.4f} cons_rx_cos={cons_rx_cos_epoch:.4f} sat_cos={sat_cos_epoch:.4f} "
            f"corr_pa={meters['corr_pa'].avg if meters['corr_pa'].count else float('nan'):.4f} "
            f"corr_dac={meters['corr_dac'].avg if meters['corr_dac'].count else float('nan'):.4f} gnorm={meters['gnorm'].avg:.3f} "
            f"stage=rx{scale['rx']:.2f}/arx{scale['adv_rx']:.2f}/atx{scale['adv_tx']:.2f}/grx{scale['grl_rx']:.2f}/gtx{scale['grl_tx']:.2f}/"
            f"sep{scale['sep']:.2f}/ctr{scale['center']:.2f}/ctx{scale['cons_tx']:.2f}/crx{scale['cons_rx']:.2f}/"
            f"prx{scale['probe_rx']:.2f}/ptx{scale['probe_tx']:.2f}/px{scale['proxy']:.2f}/sm{scale['sat_mix']:.2f}/sc{scale['sat_cls']:.2f}/ss{scale['sat_cons']:.2f} "
            f"nf_skip={nonfinite_batches} | time={time.time()-t0:.1f}s"
        )
        if eval_stats is not None:
            msg += (
                f" | val_tx={eval_stats['tx_acc']:.2f}% val_rx={eval_stats['rx_acc']:.2f}% "
                f"val_probe_rx={eval_stats['probe_rx_acc']:.2f}% val_probe_tx={eval_stats['probe_tx_acc']:.2f}% "
                f"best={best_tx:.2f}%@E{best_epoch:03d}"
            )
        print(msg, flush=True)

    print(f"Training finished. best_tx_acc={best_tx:.2f}% at epoch {best_epoch}")
    if split_info is not None:
        print(f"Final split info: {split_info}")


if __name__ == "__main__":
    main()
