import math
import time
import argparse
import random
import inspect
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
    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr_scale: float = 0.08):
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


def infer_input_len(dataset, default: int) -> int:
    try:
        x0 = dataset[0][0]
        if torch.is_tensor(x0) and x0.ndim >= 2:
            return int(x0.shape[-1])
    except Exception:
        pass
    return int(default)


def paired_cosine_consistency(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, float]:
    a = F.normalize(a.float(), dim=1, eps=1e-4)
    b = F.normalize(b.float(), dim=1, eps=1e-4)
    cos = (a * b).sum(dim=1)
    return (1.0 - cos).mean(), float(cos.mean().item())


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


def orthogonality_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(1) != b.size(1):
        raise ValueError(f"orthogonality_loss expects same feature dim, got {a.size(1)} vs {b.size(1)}")
    a = F.normalize(a.float(), dim=1, eps=1e-4)
    b = F.normalize(b.float(), dim=1, eps=1e-4)
    dot = (a * b).sum(dim=1)
    return (dot * dot).mean()


def cosine_repulsion_sq(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, float]:
    a = F.normalize(a.float(), dim=1, eps=1e-4)
    b = F.normalize(b.float(), dim=1, eps=1e-4)
    cos = (a * b).sum(dim=1)
    return (cos * cos).mean(), float(cos.abs().mean().item())


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


def stage_phase(epoch: int, epochs: int) -> str:
    r = float(epoch - 1) / max(1.0, float(epochs - 1))
    if r < 0.12:
        return "A"
    if r < 0.30:
        return "B"
    if r < 0.60:
        return "C"
    return "D"


def stage_scales(epoch: int, epochs: int) -> Dict[str, float]:
    r = float(epoch - 1) / max(1.0, float(epochs - 1))
    phase = stage_phase(epoch, epochs)
    out = dict(
        tx_aux=0.0,
        pa_proxy=0.0,
        dac_proxy=0.0,
        pa_keep=0.0,
        rx=0.0,
        adv_rx=0.0,
        adv_tx=0.0,
        clean_cons=0.0,
        pa_repel=0.0,
        orth=0.0,
        beta=0.0,
        dac_scale=0.0,
        probe_as_trigger=0.0,
        phase=phase,
    )
    if phase == "A":
        out.update(tx_aux=1.0, pa_proxy=1.0, dac_proxy=1.0, pa_keep=0.35)
    elif phase == "B":
        t = (r - 0.12) / 0.18
        out.update(
            tx_aux=1.0 - 0.35 * t,
            pa_proxy=1.0 - 0.25 * t,
            dac_proxy=1.0 - 0.25 * t,
            pa_keep=0.35 + 0.35 * t,
            rx=0.25 + 0.45 * t,
            adv_rx=0.10 * t,
        )
    elif phase == "C":
        t = (r - 0.30) / 0.30
        out.update(
            tx_aux=max(0.0, 0.65 - 0.65 * t),
            pa_proxy=max(0.0, 0.75 - 0.55 * t),
            dac_proxy=max(0.0, 0.75 - 0.55 * t),
            pa_keep=0.70 + 0.30 * t,
            rx=0.70 - 0.10 * t,
            adv_rx=0.10 + 0.15 * t,
            clean_cons=0.20 + 0.80 * t,
            pa_repel=0.10 + 0.40 * t,
            orth=0.10 + 0.40 * t,
            beta=0.15 + 0.85 * t,
            dac_scale=0.20 + 0.80 * t,
            probe_as_trigger=1.0,
        )
    else:
        t = (r - 0.60) / 0.40
        out.update(
            pa_keep=1.0,
            rx=max(0.20, 0.60 - 0.35 * t),
            adv_rx=0.25 + 0.15 * t,
            adv_tx=0.15 + 0.20 * t,
            clean_cons=1.0,
            pa_repel=0.50 + 0.50 * t,
            orth=0.50 + 0.50 * t,
            beta=1.0,
            dac_scale=1.0,
            probe_as_trigger=1.0,
        )
    return out


def set_sinc_trainable(model: nn.Module, trainable: bool = True):
    changed = 0
    for name, param in model.named_parameters():
        if "id_backbone.sinc." in name:
            param.requires_grad = bool(trainable)
            changed += 1
    return changed


def make_sat_cfg(args) -> Optional[object]:
    if SatSimConfig is None or apply_sat_gnd_channel_batch is None:
        return None
    loo = {"low": "light", "high": "severe"}.get(str(args.sat_loo_level), str(args.sat_loo_level))
    kwargs = dict(
        fc_hz=float(args.sat_fc_hz),
        scenario=str(args.sat_scenario),
        weather=str(args.sat_weather),
        loo_level=loo,
        theta_deg=(float(args.sat_theta_min), float(args.sat_theta_max)),
        snr_db=(float(args.sat_snr_min), float(args.sat_snr_max)),
        cfo_std_hz=float(args.sat_cfo_std_hz),
        phase_noise_inc_std=(float(args.sat_pn_min), float(args.sat_pn_max)),
        iq_amp_db=(float(args.sat_iq_amp_min_db), float(args.sat_iq_amp_max_db)),
        iq_phase_deg=(float(args.sat_iq_phase_min_deg), float(args.sat_iq_phase_max_deg)),
        agc_resid_db=(float(args.sat_agc_min_db), float(args.sat_agc_max_db)),
        K_db_range=(float(args.sat_K_min_db), float(args.sat_K_max_db)),
        enable_multipath=bool(args.sat_enable_multipath),
        num_taps=(int(args.sat_num_taps_min), int(args.sat_num_taps_max)),
        max_delay_samp=int(args.sat_max_delay_samp),
        pwr_decay=float(args.sat_pwr_decay),
        markov_alpha=float(args.sat_markov_alpha),
    )
    sig = inspect.signature(SatSimConfig)
    if "fs_hz" in sig.parameters:
        kwargs["fs_hz"] = float(args.sample_rate_hz)
    elif "sample_rate_hz" in sig.parameters:
        kwargs["sample_rate_hz"] = float(args.sample_rate_hz)
    else:
        return None
    return SatSimConfig(**kwargs)


def blend_sat_view(x: torch.Tensor, x_sat: torch.Tensor, alpha: float, abs_clip: float = 0.0) -> torch.Tensor:
    alpha = float(min(1.0, max(0.0, alpha)))
    y = (1.0 - alpha) * x + alpha * x_sat
    y = renorm_rms(y)
    return sanitize_batch_tensor(y, abs_clip=abs_clip)


def create_optimizer(model: nn.Module, args):
    sinc_params, backbone_params, rx_params, head_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "id_backbone.sinc." in name:
            sinc_params.append(param)
        elif name.startswith("id_backbone."):
            backbone_params.append(param)
        elif any(name.startswith(p) for p in ("rx_backbone.", "overlap.", "pa_fusion.", "pa_to_rx.")):
            rx_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if backbone_params:
        groups.append(dict(params=backbone_params, lr=float(args.lr), weight_decay=float(args.weight_decay)))
    if sinc_params:
        groups.append(dict(params=sinc_params, lr=float(args.lr) * float(args.sinc_lr_scale), weight_decay=float(args.weight_decay)))
    if rx_params:
        groups.append(dict(params=rx_params, lr=float(args.lr) * float(args.rx_lr_scale), weight_decay=float(args.weight_decay)))
    if head_params:
        groups.append(dict(params=head_params, lr=float(args.lr) * float(args.head_lr_scale), weight_decay=float(args.weight_decay)))
    return torch.optim.AdamW(groups, betas=(0.9, 0.99))


def _gather_named_trainable_params(model: nn.Module):
    named = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            named.append((name, p))
    return named


def _zero_like_named(named_params):
    return [torch.zeros_like(p, memory_format=torch.preserve_format) for _, p in named_params]


def _grad_list(loss: torch.Tensor, named_params, retain_graph: bool):
    params = [p for _, p in named_params]
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    out = []
    for g, (_, p) in zip(grads, named_params):
        if g is None:
            out.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            out.append(g.detach())
    return out


def _dot(gs1, gs2):
    s = None
    for g1, g2 in zip(gs1, gs2):
        v = (g1 * g2).sum()
        s = v if s is None else (s + v)
    return s if s is not None else torch.tensor(0.0)


def _norm(gs):
    s = None
    for g in gs:
        v = (g * g).sum()
        s = v if s is None else (s + v)
    if s is None:
        return torch.tensor(0.0)
    return torch.sqrt(s.clamp_min(1e-12))


def _scale_grad_list(gs, scale: float):
    return [g * float(scale) for g in gs]


def _add_grad_lists(a, b):
    return [ga + gb for ga, gb in zip(a, b)]


def _project_conflict(aux, main):
    dot = _dot(aux, main)
    n2 = _norm(main) ** 2
    cos = float((dot / (_norm(aux) * _norm(main) + 1e-12)).item()) if float(_norm(aux).item()) > 0 and float(_norm(main).item()) > 0 else float("nan")
    if float(dot.item()) < 0.0 and float(n2.item()) > 1e-12:
        coef = dot / n2
        aux = [ga - coef * gm for ga, gm in zip(aux, main)]
    return aux, cos


def _cap_grad_norm(aux, main, max_ratio: float):
    n_aux = float(_norm(aux).item())
    n_main = float(_norm(main).item())
    if n_aux <= 0 or n_main <= 0:
        return aux
    limit = float(max_ratio) * n_main
    if n_aux > limit > 0:
        scale = limit / max(n_aux, 1e-12)
        aux = _scale_grad_list(aux, scale)
    return aux


def _assign_named_grads(named_params, grads):
    for (_, p), g in zip(named_params, grads):
        p.grad = g.clone()


def maybe_enable_adv_tx(scale: Dict[str, float], probe_tx_acc_epoch: float, num_classes: int, force_off: bool) -> bool:
    if force_off:
        return False
    chance = 100.0 / max(1, int(num_classes))
    trigger = max(45.0, 3.0 * chance)
    if float(scale.get("adv_tx", 0.0)) <= 0:
        return False
    return float(probe_tx_acc_epoch) >= trigger


@torch.no_grad()
def evaluate(model, loader, device, nuisance_target: str = "auto", num_nuisance_train: int = 1, max_batches: int = 0,
             dac_resid_scale: float = 0.10, overlap_beta: float = 0.18):
    model.eval()
    meters = {k: AverageMeter() for k in [
        "tx_acc", "raw_tx_acc", "rx_acc", "probe_rx_acc", "probe_tx_acc", "clean_ratio", "common_norm"
    ]}
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long().view(-1)
        nlab = get_nuisance_target(extra, device, target=nuisance_target)
        out = model(
            x, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0,
            dac_resid_scale=dac_resid_scale, overlap_beta=overlap_beta, return_aux=True
        )
        meters["tx_acc"].update(accuracy_from_logits(out["tx_logits"], y), x.size(0))
        meters["raw_tx_acc"].update(accuracy_from_logits(out["raw_tx_logits"], y), x.size(0))
        if nlab is not None and int(num_nuisance_train) > 1:
            meters["rx_acc"].update(accuracy_from_logits(out["rx_logits"], nlab), x.size(0))
            meters["probe_rx_acc"].update(accuracy_from_logits(out["probe_rx_logits"], nlab), x.size(0))
        meters["probe_tx_acc"].update(accuracy_from_logits(out["probe_tx_logits"], y), x.size(0))
        if torch.is_tensor(out.get("clean_ratio", None)):
            meters["clean_ratio"].update(float(out["clean_ratio"].float().mean().item()), x.size(0))
        if torch.is_tensor(out.get("common_norm", None)):
            meters["common_norm"].update(float(out["common_norm"].float().mean().item()), x.size(0))
    return {k: v.avg for k, v in meters.items()}


def is_better_result(cur: Dict[str, float], best: Dict[str, float]) -> bool:
    if cur["tx_acc"] > best["tx_acc"] + 1e-6:
        return True
    if abs(cur["tx_acc"] - best["tx_acc"]) <= 0.05:
        return cur.get("probe_tx_acc", 1e9) < best.get("probe_tx_acc", 1e9) - 1e-6
    return False


def make_wisig_split_compat(ds, args):
    tx_days = [int(s) for s in args.wisig_train_days.split(",") if str(s).strip() != ""]
    test_days = [int(s) for s in args.wisig_full_test_days.split(",") if str(s).strip() != ""]
    sig = inspect.signature(make_day123_randomsplit_plus_day4_test)
    params = sig.parameters

    kwargs = dict(
        out_len=args.wisig_out_len,
        crop_mode="center",
        normalize=True,
        domain=args.wisig_domain,
        train_ratio=0.8,
        seed=args.seed,
    )

    if "train_days_idx" in params:
        kwargs["train_days_idx"] = tx_days
    elif "train_days" in params:
        kwargs["train_days"] = tx_days

    if "full_test_days_idx" in params:
        kwargs["full_test_days_idx"] = test_days
    elif "full_test_days" in params:
        kwargs["full_test_days"] = test_days

    if "eq" in params:
        kwargs["eq"] = str(args.wisig_eq)
    elif "equalized" in params:
        kwargs["equalized"] = args.wisig_eq

    return make_day123_randomsplit_plus_day4_test(ds, **kwargs)


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
    parser.add_argument("--warmup_epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sinc_lr_scale", type=float, default=0.12)
    parser.add_argument("--rx_lr_scale", type=float, default=0.85)
    parser.add_argument("--head_lr_scale", type=float, default=1.00)
    parser.add_argument("--freeze_sinc_epochs", type=int, default=6)
    parser.add_argument("--rx_dim", type=int, default=64)
    parser.add_argument("--rx_branch_dim", type=int, default=16)
    parser.add_argument("--overlap_beta", type=float, default=0.18)
    parser.add_argument("--alpha_id", type=float, default=0.25)
    parser.add_argument("--dac_resid_scale_max", type=float, default=0.08)
    parser.add_argument("--overlap_max_clean_ratio", type=float, default=0.20)

    parser.add_argument("--lambda_tx_aux", type=float, default=0.03)
    parser.add_argument("--lambda_rx", type=float, default=0.12)
    parser.add_argument("--lambda_adv_rx", type=float, default=0.02)
    parser.add_argument("--lambda_adv_tx", type=float, default=0.005)
    parser.add_argument("--lambda_orth", type=float, default=0.005)
    parser.add_argument("--lambda_clean_cons", type=float, default=0.12)
    parser.add_argument("--lambda_pa_repel", type=float, default=0.01)
    parser.add_argument("--lambda_pa_keep", type=float, default=0.10)
    parser.add_argument("--lambda_pa_proxy", type=float, default=0.05)
    parser.add_argument("--lambda_dac_proxy", type=float, default=0.03)

    parser.add_argument("--budget_domain", type=float, default=0.35)
    parser.add_argument("--budget_struct", type=float, default=0.15)
    parser.add_argument("--budget_proxy", type=float, default=0.20)
    parser.add_argument("--disable_adv_tx", dest="disable_adv_tx", action="store_true")
    parser.add_argument("--no_disable_adv_tx", dest="disable_adv_tx", action="store_false")
    parser.set_defaults(disable_adv_tx=False)

    parser.add_argument("--nuisance_target", type=str, default="auto", choices=["auto", "domain", "rx", "day"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="best_model_v15.pth")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--grad_value_clip", type=float, default=2.0)
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--input_abs_clip", type=float, default=4.0)
    parser.add_argument("--grl_lambda", type=float, default=1.0)
    parser.add_argument("--skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_true")
    parser.add_argument("--no_skip_nonfinite_batches", dest="skip_nonfinite_batches", action="store_false")
    parser.set_defaults(skip_nonfinite_batches=True)
    parser.add_argument("--nonfinite_log_limit", type=int, default=8)
    parser.add_argument("--max_loss_value", type=float, default=1e4)
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)

    # kept for compatibility; satellite branch remains optional and off by default
    parser.add_argument("--sat_start_epoch", type=int, default=1000)
    parser.add_argument("--sat_tx_weight", type=float, default=0.0)
    parser.add_argument("--lambda_sat_cons_tx", type=float, default=0.0)
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
    parser.add_argument("--sat_iq_phase_min_deg", type=float, default=-2.5)
    parser.add_argument("--sat_iq_phase_max_deg", type=float, default=2.5)
    parser.add_argument("--sat_agc_min_db", type=float, default=-0.25)
    parser.add_argument("--sat_agc_max_db", type=float, default=0.25)
    parser.add_argument("--sat_K_min_db", type=float, default=8.0)
    parser.add_argument("--sat_K_max_db", type=float, default=16.0)
    parser.add_argument("--sat_enable_multipath", dest="sat_enable_multipath", action="store_true")
    parser.add_argument("--no_sat_enable_multipath", dest="sat_enable_multipath", action="store_false")
    parser.set_defaults(sat_enable_multipath=True)
    parser.add_argument("--sat_num_taps_min", type=int, default=2)
    parser.add_argument("--sat_num_taps_max", type=int, default=5)
    parser.add_argument("--sat_max_delay_samp", type=int, default=6)
    parser.add_argument("--sat_pwr_decay", type=float, default=0.72)
    parser.add_argument("--sat_markov_alpha", type=float, default=0.75)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if args.amp:
        print("[AMP] grouped gradient coordination runs in FP32; AMP request is ignored for stability.")

    split_info = None
    if args.dataset.lower() == "wisig":
        ds = load_wisig_compact_pkl(args.wisig_pkl)
        train_ds, test_ds, split_info = make_wisig_split_compat(ds, args)
        num_classes = len(ds.get("tx_list", [])) or (len(getattr(train_ds, "tx_list", [])) if hasattr(train_ds, "tx_list") else 6)
        print(f"[WISIG] overriding num_classes 16 -> {num_classes}")
        print(f"[WISIG] pkl={args.wisig_pkl} eq={args.wisig_eq} out_len={args.wisig_out_len} domain={args.wisig_domain}")
        if isinstance(split_info, dict):
            print(f"[WISIG] TRAIN SOURCES: {split_info.get('train_days_label', [])} -> random 80%")
            print(f"[WISIG] TEST SOURCES : same days remaining 20% + full {split_info.get('full_test_days_label', [])}")
            print(f"[WISIG] split_info={split_info}")
    else:
        if WiFiRFFIDataset is None:
            raise RuntimeError("dataset.py / WiFiRFFIDataset not available.")
        train_ds = WiFiRFFIDataset(args.dataset_dir, mode="train")
        test_ds = WiFiRFFIDataset(args.dataset_dir, mode="test")
        num_classes = 16

    input_len = infer_input_len(train_ds, args.wisig_out_len if args.dataset.lower() == "wisig" else 1024)
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(args.num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    num_nuisance = infer_nuisance_classes(train_ds, target=args.nuisance_target)
    print(f"[NUISANCE] target={args.nuisance_target} classes={num_nuisance}")

    model = build_dual_model(
        num_classes=num_classes,
        num_domains=num_nuisance,
        model_size=args.model_size,
        dataset=args.dataset,
        input_len=input_len,
        sample_rate_hz=args.sample_rate_hz,
        rx_dim=args.rx_dim,
        rx_branch_dim=args.rx_branch_dim,
        overlap_beta=args.overlap_beta,
        alpha_id=args.alpha_id,
        default_eta_dac=args.dac_resid_scale_max,
        overlap_max_clean_ratio=args.overlap_max_clean_ratio,
    ).to(device)
    print(f"[MODEL] DualRxDACOverlapPAAware tx_dim={model.tx_dim} rx_dim={model.rx_dim}")

    optimizer = create_optimizer(model, args)
    scheduler = WarmupCosineScheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    ce_tx = nn.CrossEntropyLoss(label_smoothing=0.03)
    ce_rx = nn.CrossEntropyLoss(label_smoothing=0.02)

    augmentor_main = None
    augmentor_proxy = None
    if bool(args.augment) and build_augmentor is not None:
        try:
            augmentor_main = build_augmentor(sample_rate_hz=float(args.sample_rate_hz), p_dac=0.20, p_pa=0.25)
            augmentor_proxy = build_augmentor(sample_rate_hz=float(args.sample_rate_hz), p_dac=0.0, p_pa=0.0)
        except TypeError:
            augmentor_main = build_augmentor(sample_rate_hz=float(args.sample_rate_hz))
            augmentor_proxy = build_augmentor(sample_rate_hz=float(args.sample_rate_hz))
        print("[AUG] enabled.")
    else:
        print("[AUG] disabled or unavailable.")

    sat_cfg = make_sat_cfg(args)
    if sat_cfg is None or args.sat_start_epoch >= args.epochs or (args.sat_tx_weight == 0 and args.lambda_sat_cons_tx == 0):
        print("[SAT] disabled in optimized training flow.")
    else:
        print(f"[SAT] enabled. scenario={args.sat_scenario} weather={args.sat_weather} start_epoch={args.sat_start_epoch}")

    best = {"tx_acc": -1.0, "probe_tx_acc": 1e9, "epoch": 0}
    nonfinite_logs = 0
    probe_tx_epoch_prev = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if epoch <= int(args.freeze_sinc_epochs):
            set_sinc_trainable(model, False)
        elif epoch == int(args.freeze_sinc_epochs) + 1:
            set_sinc_trainable(model, True)
        scheduler.step(epoch)
        scale = stage_scales(epoch, args.epochs)
        phase = scale["phase"]

        meters = {k: AverageMeter() for k in [
            "loss", "tx", "tx_aux", "rx", "adv_rx", "adv_tx", "orth", "clean_cons",
            "pa_repel", "pa_keep", "pa_proxy", "dac_proxy", "tx_acc", "raw_tx_acc",
            "rx_acc", "probe_rx_acc", "probe_tx_acc", "corr_pa", "corr_dac", "common_norm",
            "clean_ratio", "gcos_dom", "gcos_struct", "gcos_proxy"
        ]}
        clean_cos_vals: List[float] = []
        pa_rep_vals: List[float] = []
        nonfinite_batches = 0

        for batch in train_loader:
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long().view(-1)
            x = sanitize_batch_tensor(x, abs_clip=float(args.input_abs_clip))
            nlab = get_nuisance_target(extra, device, target=args.nuisance_target)

            if augmentor_main is not None:
                x_main = augmentor_main(x, labels=y)
            else:
                x_main = x
            if augmentor_proxy is not None:
                try:
                    x_pa, s_p = augmentor_proxy(x, labels=y, pa_only=True, return_pa_strength=True)
                    x_dac, s_d = augmentor_proxy(x, labels=y, dac_only=True, return_dac_strength=True)
                except TypeError:
                    x_pa = x
                    x_dac = x
                    s_p = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
                    s_d = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
            else:
                x_pa = x
                x_dac = x
                s_p = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
                s_d = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)

            x_main = sanitize_batch_tensor(x_main, abs_clip=float(args.input_abs_clip))
            x_pa = sanitize_batch_tensor(x_pa, abs_clip=float(args.input_abs_clip))
            x_dac = sanitize_batch_tensor(x_dac, abs_clip=float(args.input_abs_clip))

            x_sat_mix = None
            use_sat_batch = sat_cfg is not None and int(epoch) >= int(args.sat_start_epoch) and (
                float(args.sat_tx_weight) > 0 or float(args.lambda_sat_cons_tx) > 0
            )
            if use_sat_batch:
                with torch.no_grad():
                    x_sat = apply_sat_gnd_channel_batch(renorm_rms(x_main), sat_cfg)
                    x_sat = renorm_rms(torch.nan_to_num(x_sat, nan=0.0, posinf=0.0, neginf=0.0))
                    x_sat_mix = blend_sat_view(x_main, x_sat, alpha=0.25, abs_clip=float(args.input_abs_clip))

            optimizer.zero_grad(set_to_none=True)

            dac_scale = float(args.dac_resid_scale_max) * float(scale["dac_scale"])
            overlap_beta_eff = float(args.overlap_beta) * float(scale["beta"])
            grl_rx = float(args.grl_lambda) * float(scale["adv_rx"])
            grl_tx = float(args.grl_lambda) * float(scale["adv_tx"])

            out_main = model(
                x_main, y_tx=y, grl_rx_on_tx=grl_rx, grl_tx_on_rx=grl_tx,
                dac_resid_scale=dac_scale, overlap_beta=overlap_beta_eff, return_aux=True
            )
            out_pa = model(x_pa, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0,
                           dac_resid_scale=dac_scale, overlap_beta=overlap_beta_eff, return_aux=True)
            out_dac = model(x_dac, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0,
                            dac_resid_scale=dac_scale, overlap_beta=overlap_beta_eff, return_aux=True)

            tx_logits = out_main["tx_logits"]
            raw_tx_logits = out_main["raw_tx_logits"]
            rx_logits = out_main["rx_logits"]
            adv_rx_logits = out_main["adv_rx_logits"]
            adv_tx_logits = out_main["adv_tx_logits"]
            probe_rx_logits = out_main["probe_rx_logits"]
            probe_tx_logits = out_main["probe_tx_logits"]
            z_main = out_main["z_main"]
            z_txc_clean = out_main["z_txc_clean"]
            r_overlap = out_main["r_overlap"]
            pa_ref_tx = out_main["pa_ref_tx"]
            common_dac = out_main["common_dac"]
            feat_pa_main = out_main["feat_pa"]
            common_norm = out_main.get("common_norm", None)
            clean_ratio = out_main.get("clean_ratio", None)

            pa_pred_pa = out_pa["aux_id"].get("pa_pred", None)
            dac_pred_dac = out_dac["aux_id"].get("dac_pred", None)

            loss_ref = tx_logits.float().mean()
            loss_tx = ce_tx(tx_logits.float(), y)
            loss_tx_aux = ce_tx(raw_tx_logits.float(), y)

            if nlab is not None and int(num_nuisance) > 1:
                loss_rx = ce_rx(rx_logits.float(), nlab)
                loss_adv_rx = ce_rx(adv_rx_logits.float(), nlab)
                loss_clean_cons, clean_cos = same_tx_diff_nuisance_consistency(z_txc_clean, y, nlab)
                if not math.isnan(clean_cos):
                    clean_cos_vals.append(clean_cos)
                meters["rx_acc"].update(accuracy_from_logits(rx_logits, nlab), x.size(0))
                meters["probe_rx_acc"].update(accuracy_from_logits(probe_rx_logits, nlab), x.size(0))
            else:
                loss_rx = fp32_zero(loss_ref)
                loss_adv_rx = fp32_zero(loss_ref)
                loss_clean_cons = fp32_zero(loss_ref)

            enable_adv_tx = maybe_enable_adv_tx(scale, probe_tx_epoch_prev, num_classes, force_off=bool(args.disable_adv_tx))
            loss_adv_tx = ce_tx(adv_tx_logits.float(), y) if enable_adv_tx else fp32_zero(loss_ref)

            loss_orth = orthogonality_loss(z_txc_clean, common_dac.detach())
            loss_pa_repel_a, pa_rep_a = cosine_repulsion_sq(common_dac, feat_pa_main.detach())
            loss_pa_repel_b, pa_rep_b = cosine_repulsion_sq(r_overlap, pa_ref_tx.detach())
            loss_pa_repel = loss_pa_repel_a + 0.5 * loss_pa_repel_b
            pa_rep_mean = 0.5 * (pa_rep_a + pa_rep_b)
            if not math.isnan(pa_rep_mean):
                pa_rep_vals.append(pa_rep_mean)

            # only enable PA repel when cleaner is actually active and non-trivial
            if torch.is_tensor(common_norm):
                common_norm_mean = float(common_norm.float().mean().item())
                if common_norm_mean < 0.02:
                    loss_pa_repel = fp32_zero(loss_ref)
                    loss_orth = fp32_zero(loss_ref)

            loss_pa_keep, _ = paired_cosine_consistency(z_main, feat_pa_main.detach())
            if torch.is_tensor(pa_pred_pa):
                loss_pa_proxy = F.mse_loss(pa_pred_pa.float().view(-1), s_p.float().view(-1))
            else:
                loss_pa_proxy = fp32_zero(loss_ref)
            if torch.is_tensor(dac_pred_dac):
                loss_dac_proxy = F.mse_loss(dac_pred_dac.float().view(-1), s_d.float().view(-1))
            else:
                loss_dac_proxy = fp32_zero(loss_ref)

            if x_sat_mix is not None:
                out_sat = model(
                    x_sat_mix, y_tx=y, grl_rx_on_tx=0.0, grl_tx_on_rx=0.0,
                    dac_resid_scale=dac_scale, overlap_beta=overlap_beta_eff, return_aux=True
                )
                loss_sat_tx = ce_tx(out_sat["tx_logits"].float(), y)
                loss_sat_cons, _ = paired_cosine_consistency(out_main["z_main"], out_sat["z_main"])
            else:
                loss_sat_tx = fp32_zero(loss_ref)
                loss_sat_cons = fp32_zero(loss_ref)

            group_main = (
                loss_tx
                + float(args.lambda_pa_keep) * float(scale["pa_keep"]) * loss_pa_keep
                + float(args.lambda_clean_cons) * float(scale["clean_cons"]) * loss_clean_cons
            )
            group_domain = (
                float(args.lambda_rx) * float(scale["rx"]) * loss_rx
                + float(args.lambda_adv_rx) * float(scale["adv_rx"]) * loss_adv_rx
                + float(args.lambda_adv_tx) * float(scale["adv_tx"]) * loss_adv_tx
            )
            group_struct = (
                float(args.lambda_pa_repel) * float(scale["pa_repel"]) * loss_pa_repel
                + float(args.lambda_orth) * float(scale["orth"]) * loss_orth
            )
            group_proxy = (
                float(args.lambda_tx_aux) * float(scale["tx_aux"]) * loss_tx_aux
                + float(args.lambda_pa_proxy) * float(scale["pa_proxy"]) * loss_pa_proxy
                + float(args.lambda_dac_proxy) * float(scale["dac_proxy"]) * loss_dac_proxy
            )
            group_sat = (
                float(args.sat_tx_weight) * loss_sat_tx + float(args.lambda_sat_cons_tx) * loss_sat_cons
            )
            total_for_log = group_main + group_domain + group_struct + group_proxy + group_sat

            bad = []
            for name, value in {
                "loss": total_for_log,
                "tx_logits": tx_logits,
                "rx_logits": rx_logits,
                "adv_rx_logits": adv_rx_logits,
                "adv_tx_logits": adv_tx_logits,
                "z_main": z_main,
                "z_txc_clean": z_txc_clean,
                "common_dac": common_dac,
            }.items():
                if torch.is_tensor(value) and (not tensor_is_finite(value)):
                    bad.append(name)
            if total_for_log.detach().float().abs().item() > float(args.max_loss_value):
                bad.append("loss_too_large")
            if len(bad) > 0:
                nonfinite_batches += 1
                if nonfinite_logs < int(args.nonfinite_log_limit):
                    print(f"[WARN][E{epoch:03d}] skip batch due to non-finite values: {', '.join(bad)}", flush=True)
                    nonfinite_logs += 1
                if bool(args.skip_nonfinite_batches):
                    continue

            named_params = _gather_named_trainable_params(model)
            grad_main = _grad_list(group_main, named_params, retain_graph=True)
            grad_domain = _grad_list(group_domain, named_params, retain_graph=True) if float(group_domain.detach().abs().item()) > 0 else _zero_like_named(named_params)
            grad_struct = _grad_list(group_struct, named_params, retain_graph=True) if float(group_struct.detach().abs().item()) > 0 else _zero_like_named(named_params)
            grad_proxy = _grad_list(group_proxy, named_params, retain_graph=True) if float(group_proxy.detach().abs().item()) > 0 else _zero_like_named(named_params)
            grad_sat = _grad_list(group_sat, named_params, retain_graph=False) if float(group_sat.detach().abs().item()) > 0 else _zero_like_named(named_params)

            grad_domain, cos_dom = _project_conflict(grad_domain, grad_main)
            grad_struct, cos_struct = _project_conflict(grad_struct, grad_main)
            grad_proxy, cos_proxy = _project_conflict(grad_proxy, grad_main)
            grad_sat, _ = _project_conflict(grad_sat, grad_main)

            grad_domain = _cap_grad_norm(grad_domain, grad_main, max_ratio=float(args.budget_domain))
            grad_struct = _cap_grad_norm(grad_struct, grad_main, max_ratio=float(args.budget_struct))
            grad_proxy = _cap_grad_norm(grad_proxy, grad_main, max_ratio=float(args.budget_proxy))
            grad_sat = _cap_grad_norm(grad_sat, grad_main, max_ratio=0.15)

            total_grads = _add_grad_lists(grad_main, _add_grad_lists(grad_domain, _add_grad_lists(grad_struct, _add_grad_lists(grad_proxy, grad_sat))))
            _assign_named_grads(named_params, total_grads)

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
                continue
            optimizer.step()

            bsz = x.size(0)
            meters["loss"].update(total_for_log.item(), bsz)
            meters["tx"].update(loss_tx.item(), bsz)
            meters["tx_aux"].update(loss_tx_aux.item(), bsz)
            meters["rx"].update(loss_rx.item(), bsz)
            meters["adv_rx"].update(loss_adv_rx.item(), bsz)
            meters["adv_tx"].update(loss_adv_tx.item(), bsz)
            meters["orth"].update(loss_orth.item(), bsz)
            meters["clean_cons"].update(loss_clean_cons.item(), bsz)
            meters["pa_repel"].update(loss_pa_repel.item(), bsz)
            meters["pa_keep"].update(loss_pa_keep.item(), bsz)
            meters["pa_proxy"].update(loss_pa_proxy.item(), bsz)
            meters["dac_proxy"].update(loss_dac_proxy.item(), bsz)
            meters["tx_acc"].update(accuracy_from_logits(tx_logits, y), bsz)
            meters["raw_tx_acc"].update(accuracy_from_logits(raw_tx_logits, y), bsz)
            meters["probe_tx_acc"].update(accuracy_from_logits(probe_tx_logits, y), bsz)
            if not math.isnan(cos_dom):
                meters["gcos_dom"].update(cos_dom, bsz)
            if not math.isnan(cos_struct):
                meters["gcos_struct"].update(cos_struct, bsz)
            if not math.isnan(cos_proxy):
                meters["gcos_proxy"].update(cos_proxy, bsz)
            if torch.is_tensor(common_norm):
                meters["common_norm"].update(float(common_norm.float().mean().item()), bsz)
            if torch.is_tensor(clean_ratio):
                meters["clean_ratio"].update(float(clean_ratio.float().mean().item()), bsz)

            corr_pa = batch_corrcoef(pa_pred_pa, s_p)
            corr_dac = batch_corrcoef(dac_pred_dac, s_d)
            if not math.isnan(corr_pa):
                meters["corr_pa"].update(corr_pa, bsz)
            if not math.isnan(corr_dac):
                meters["corr_dac"].update(corr_dac, bsz)

        probe_tx_epoch_prev = meters["probe_tx_acc"].avg

        eval_stats = None
        if epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            eval_stats = evaluate(
                model,
                test_loader,
                device,
                nuisance_target=args.nuisance_target,
                num_nuisance_train=num_nuisance,
                max_batches=int(args.eval_max_batches),
                dac_resid_scale=dac_scale,
                overlap_beta=overlap_beta_eff,
            )
            candidate = {"tx_acc": eval_stats["tx_acc"], "probe_tx_acc": eval_stats["probe_tx_acc"], "epoch": epoch}
            if is_better_result(candidate, best):
                best = candidate
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_tx_acc": best["tx_acc"],
                        "best_probe_tx_acc": best["probe_tx_acc"],
                        "args": vars(args),
                        "split_info": split_info,
                        "num_nuisance": num_nuisance,
                    },
                    args.save_path,
                )

        msg = (
            f"[E{epoch:03d}] phase={phase} lr={optimizer.param_groups[0]['lr']:.2e} loss={meters['loss'].avg:.4f} "
            f"tx={meters['tx'].avg:.4f} tx_aux={meters['tx_aux'].avg:.4f} rx={meters['rx'].avg:.4f} "
            f"adv_rx={meters['adv_rx'].avg:.4f} adv_tx={meters['adv_tx'].avg:.4f} orth={meters['orth'].avg:.4f} "
            f"clean_cons={meters['clean_cons'].avg:.4f} pa_repel={meters['pa_repel'].avg:.4f} "
            f"pa_keep={meters['pa_keep'].avg:.4f} pa_proxy={meters['pa_proxy'].avg:.4f} dac_proxy={meters['dac_proxy'].avg:.4f} | "
            f"tx_acc={meters['tx_acc'].avg:.2f}% raw_tx_acc={meters['raw_tx_acc'].avg:.2f}% "
            f"rx_acc={meters['rx_acc'].avg if meters['rx_acc'].count else float('nan'):.2f}% "
            f"probe_rx_acc={meters['probe_rx_acc'].avg if meters['probe_rx_acc'].count else float('nan'):.2f}% "
            f"probe_tx_acc={meters['probe_tx_acc'].avg:.2f}% "
            f"clean_cos={np.mean(clean_cos_vals) if len(clean_cos_vals) else float('nan'):.4f} "
            f"pa_rep={np.mean(pa_rep_vals) if len(pa_rep_vals) else float('nan'):.4f} "
            f"corr_pa={meters['corr_pa'].avg if meters['corr_pa'].count else float('nan'):.4f} "
            f"corr_dac={meters['corr_dac'].avg if meters['corr_dac'].count else float('nan'):.4f} "
            f"common_norm={meters['common_norm'].avg if meters['common_norm'].count else float('nan'):.4f} "
            f"clean_ratio={meters['clean_ratio'].avg if meters['clean_ratio'].count else float('nan'):.4f} "
            f"gcos_dom={meters['gcos_dom'].avg if meters['gcos_dom'].count else float('nan'):.4f} "
            f"gcos_struct={meters['gcos_struct'].avg if meters['gcos_struct'].count else float('nan'):.4f} "
            f"gcos_proxy={meters['gcos_proxy'].avg if meters['gcos_proxy'].count else float('nan'):.4f} "
            f"nf_skip={nonfinite_batches} | time={time.time() - t0:.1f}s"
        )
        if eval_stats is not None:
            msg += (
                f" | val_tx={eval_stats['tx_acc']:.2f}% val_raw_tx={eval_stats['raw_tx_acc']:.2f}% "
                f"val_rx={eval_stats['rx_acc'] if 'rx_acc' in eval_stats else float('nan'):.2f}% "
                f"val_probe_rx={eval_stats['probe_rx_acc'] if 'probe_rx_acc' in eval_stats else float('nan'):.2f}% "
                f"val_probe_tx={eval_stats['probe_tx_acc']:.2f}% "
                f"val_common_norm={eval_stats['common_norm'] if 'common_norm' in eval_stats else float('nan'):.4f} "
                f"val_clean_ratio={eval_stats['clean_ratio'] if 'clean_ratio' in eval_stats else float('nan'):.4f} "
                f"best={best['tx_acc']:.2f}%@E{best['epoch']:03d}"
            )
        print(msg, flush=True)


if __name__ == "__main__":
    main()
