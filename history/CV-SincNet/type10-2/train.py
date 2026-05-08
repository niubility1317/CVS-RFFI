import math
import time
import argparse
import random
from copy import deepcopy
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
from dataset_wisig import load_wisig_compact_pkl, make_wisig_trainval_test_by_day_rx
try:
    from model import build_model
except Exception:
    from model_pa_dac_coop_optimized import build_model
from DataAugmentation import build_augmentor


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


class CosFaceAuxHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x_f = F.normalize(x.float(), dim=1, eps=1e-4)
            w_f = F.normalize(self.weight.float(), dim=1, eps=1e-4)
            cos = F.linear(x_f, w_f)
            if labels is None:
                return cos * self.s
            labels = labels.view(-1).long()
            one_hot = torch.zeros_like(cos)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            return (cos - one_hot * self.m) * self.s


class BranchAuxHeads(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.pa_head = CosFaceAuxHead(emb_dim, num_classes, s=s, m=m)
        self.dac_head = CosFaceAuxHead(emb_dim, num_classes, s=s, m=m)



def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra



def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item() * 100.0



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



def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_true: str, help_false: str):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name.replace('-', '_'), action="store_true", help=help_true)
    group.add_argument(f"--no_{name}", dest=name.replace('-', '_'), action="store_false", help=help_false)
    parser.set_defaults(**{name.replace('-', '_'): default})



def default_is_path(p: str, default_name: str) -> bool:
    return str(p).strip() == default_name



def safe_nan(v: float) -> str:
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"



def save_checkpoint(path: str, *, model, aux_heads, optimizer, scheduler, scaler, epoch: int, args, split_info, stats: dict):
    payload = {
        "model": model.state_dict(),
        "aux_heads": aux_heads.state_dict() if aux_heads is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "args": vars(args),
        "split_info": split_info,
        "stats": stats,
    }
    torch.save(payload, path)



def ramp_value(epoch: int, warmup_epochs: int, ramp_epochs: int, min_scale: float, max_scale: float, curve: float = 1.0) -> float:
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



def cosine_distance_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a.float(), dim=1, eps=eps)
    b = F.normalize(b.float(), dim=1, eps=eps)
    return (1.0 - torch.sum(a * b, dim=1)).clamp_min(0.0)



def cosine_consistency_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, float]:
    dist = cosine_distance_per_sample(a, b, eps=eps)
    cos = (1.0 - dist).mean().item()
    return dist.mean(), float(cos)



def smooth_strength_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred.float().view(-1), target.float().view(-1))



def selective_shift_margin_loss(main_shift: torch.Tensor, other_shift: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    return F.relu(float(margin) + other_shift.float() - main_shift.float()).mean()



def pairwise_monotonic_loss(strength: torch.Tensor, response: torch.Tensor, margin: float = 0.0, min_delta: float = 1e-4) -> torch.Tensor:
    s = strength.float().view(-1)
    r = response.float().view(-1)
    if s.numel() <= 1:
        return response.new_tensor(0.0)
    ds = s[:, None] - s[None, :]
    dr = r[:, None] - r[None, :]
    mask = ds > float(min_delta)
    if not mask.any():
        return response.new_tensor(0.0)
    viol = float(margin) - dr
    return F.relu(viol[mask]).mean()



def one_way_kl_from_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    T = float(max(1e-6, temperature))
    log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
    p_t = F.softmax(teacher_logits.float().detach() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)



def covariance_orth_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = z_a.float() - z_a.float().mean(dim=0, keepdim=True)
    z_b = z_b.float() - z_b.float().mean(dim=0, keepdim=True)
    n = z_a.size(0)
    if n <= 1:
        return z_a.new_tensor(0.0)
    cov = (z_a.t() @ z_b) / float(n - 1)
    return torch.mean(cov * cov)



def evaluate_loader(model, aux_heads, loader, device, max_batches: int = 0):
    model.eval()
    aux_heads.eval()
    main_correct = pa_correct = dac_correct = total = 0
    for bi, batch in enumerate(loader):
        x, y, _ = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, y=y, return_aux=True)
        pa_logits = aux_heads.pa_head(out["feat_pa"], labels=None)
        dac_logits = aux_heads.dac_head(out["feat_dac"], labels=None)
        main_correct += int((out["logits"].argmax(dim=1) == y).sum().item())
        pa_correct += int((pa_logits.argmax(dim=1) == y).sum().item())
        dac_correct += int((dac_logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
        if max_batches > 0 and (bi + 1) >= max_batches:
            break
    return {
        "main_acc": 100.0 * main_correct / max(1, total),
        "pa_acc": 100.0 * pa_correct / max(1, total),
        "dac_acc": 100.0 * dac_correct / max(1, total),
        "main_correct": int(main_correct),
        "pa_correct": int(pa_correct),
        "dac_correct": int(dac_correct),
        "total": int(total),
    }



def evaluate_named_loaders(model, aux_heads, named_loaders: Dict[str, DataLoader], device, max_batches: int = 0):
    out = {}
    for name, loader in named_loaders.items():
        out[name] = evaluate_loader(model, aux_heads, loader, device=device, max_batches=max_batches)
    return out



def aggregate_named_stats(named_stats: Dict[str, Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    total = main_c = pa_c = dac_c = 0
    for k in keys:
        if k not in named_stats:
            continue
        total += int(named_stats[k].get("total", 0))
        main_c += int(named_stats[k].get("main_correct", 0))
        pa_c += int(named_stats[k].get("pa_correct", 0))
        dac_c += int(named_stats[k].get("dac_correct", 0))
    return {
        "main_acc": 100.0 * main_c / max(1, total),
        "pa_acc": 100.0 * pa_c / max(1, total),
        "dac_acc": 100.0 * dac_c / max(1, total),
        "main_correct": int(main_c),
        "pa_correct": int(pa_c),
        "dac_correct": int(dac_c),
        "total": int(total),
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
        lines.append(
            f"          {label}: main={stats['main_acc']:.2f}% pa={stats['pa_acc']:.2f}% dac={stats['dac_acc']:.2f}% ({stats['main_correct']}/{stats['total']})"
        )
    return lines



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
        "defect_apply_channel": bool(args.aug_defect_apply_channel),
    }



def make_augmentor(base_cfg: Dict[str, Any]):
    return build_augmentor(**deepcopy(base_cfg))



def configure_augmentor_for_epoch(augmentor, base_cfg: Dict[str, Any], epoch: int, args):
    scale = ramp_value(
        epoch=epoch,
        warmup_epochs=int(args.aug_warmup_epochs),
        ramp_epochs=int(args.aug_ramp_epochs),
        min_scale=float(args.aug_scale_min),
        max_scale=float(args.aug_scale_max),
        curve=float(args.aug_ramp_curve),
    )

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



def set_loss_pack(args, *, main_ce=1.0, pa_cls=0.25, dac_cls=0.20, pa_kl=0.06, dac_kl=0.04,
                  pa_reg=0.18, pa_select=0.08, pa_mono=0.05,
                  dac_reg=0.10, dac_select=0.04, dac_mono=0.02,
                  pa_dac_orth=0.02):
    args.lambda_main_ce = float(main_ce)
    args.lambda_pa_cls = float(pa_cls)
    args.lambda_dac_cls = float(dac_cls)
    args.lambda_pa_kl = float(pa_kl)
    args.lambda_dac_kl = float(dac_kl)
    args.lambda_pa_reg = float(pa_reg)
    args.lambda_pa_select = float(pa_select)
    args.lambda_pa_mono = float(pa_mono)
    args.lambda_dac_reg = float(dac_reg)
    args.lambda_dac_select = float(dac_select)
    args.lambda_dac_mono = float(dac_mono)
    args.lambda_pa_dac_orth = float(pa_dac_orth)



def apply_experiment_preset(args):
    g = str(args.exp_group)

    # Common default: single-day/single-rx train, cross-domain test via held-out day/rx.
    args.use_aug = True
    args.enable_clean_branch_cls = True
    args.enable_pa_aux = True
    args.enable_dac_aux = True
    args.aug_enable_class_signature = False
    args.aug_defect_apply_channel = True
    args.aug_enable_pa_normal = False
    args.aug_scale_min = 0.10
    args.aug_scale_max = 0.35
    args.aug_warmup_epochs = 3
    args.aug_ramp_epochs = 15
    args.aug_ramp_curve = 1.25
    args.branch_warmup_epochs = 1
    args.branch_ramp_epochs = 8
    args.aux_warmup_epochs = 3
    args.aux_ramp_epochs = 15
    args.mono_warmup_epochs = 8
    args.mono_ramp_epochs = 12
    args.robust_temp = 1.0
    args.select_margin = 0.03
    args.mono_margin = 0.00
    args.aug_p_pa = 0.0
    args.aug_p_dac = 0.25
    args.aug_pa_mp_sigma = 0.04
    args.aug_pa_mem_sigma = 0.03
    args.aug_pa_ampm_max = 0.15
    args.aug_pa_iq_img_max = 0.015
    set_loss_pack(args)

    if g == "g0_baseline_joint":
        args.enable_clean_branch_cls = False
        args.enable_pa_aux = False
        args.enable_dac_aux = False
        set_loss_pack(args, pa_cls=0.0, dac_cls=0.0, pa_kl=0.0, dac_kl=0.0,
                      pa_reg=0.0, pa_select=0.0, pa_mono=0.0,
                      dac_reg=0.0, dac_select=0.0, dac_mono=0.0, pa_dac_orth=0.0)
        args.exp_desc = "纯主头基线：无分支分类、无PA/DAC辅助"

    elif g == "g1_clean_coop":
        args.enable_pa_aux = False
        args.enable_dac_aux = False
        set_loss_pack(args, pa_cls=0.22, dac_cls=0.18, pa_kl=0.05, dac_kl=0.04,
                      pa_reg=0.0, pa_select=0.0, pa_mono=0.0,
                      dac_reg=0.0, dac_select=0.0, dac_mono=0.0, pa_dac_orth=0.02)
        args.exp_desc = "仅分类头协同：clean上训PA/DAC分支分类与协同KL"

    elif g == "g2_pa_aux":
        args.enable_dac_aux = False
        set_loss_pack(args, pa_cls=0.28, dac_cls=0.15, pa_kl=0.06, dac_kl=0.0,
                      pa_reg=0.20, pa_select=0.10, pa_mono=0.06,
                      dac_reg=0.0, dac_select=0.0, dac_mono=0.0, pa_dac_orth=0.02)
        args.exp_desc = "PA主导辅助：clean协同 + PA-only 分支损失"

    elif g == "g3_dac_aux":
        args.enable_pa_aux = False
        set_loss_pack(args, pa_cls=0.22, dac_cls=0.18, pa_kl=0.0, dac_kl=0.04,
                      pa_reg=0.0, pa_select=0.0, pa_mono=0.0,
                      dac_reg=0.12, dac_select=0.05, dac_mono=0.03, pa_dac_orth=0.02)
        args.exp_desc = "DAC辅助对照：clean协同 + DAC-only 分支损失"

    elif g == "g4_full_balanced":
        set_loss_pack(args, pa_cls=0.28, dac_cls=0.18, pa_kl=0.06, dac_kl=0.04,
                      pa_reg=0.20, pa_select=0.10, pa_mono=0.06,
                      dac_reg=0.10, dac_select=0.04, dac_mono=0.02, pa_dac_orth=0.02)
        args.exp_desc = "全量平衡：分类头协同 + PA/DAC 双辅助"

    elif g == "g5_pa_main_plus_aux":
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.20
        set_loss_pack(args, pa_cls=0.30, dac_cls=0.16, pa_kl=0.06, dac_kl=0.03,
                      pa_reg=0.22, pa_select=0.10, pa_mono=0.06,
                      dac_reg=0.08, dac_select=0.03, dac_mono=0.02, pa_dac_orth=0.02)
        args.exp_desc = "主视图轻PA + 双辅助：检验 mild PA 是否增强主表征"

    elif g == "g6_pa_dominant":
        set_loss_pack(args, pa_cls=0.38, dac_cls=0.10, pa_kl=0.08, dac_kl=0.02,
                      pa_reg=0.28, pa_select=0.14, pa_mono=0.08,
                      dac_reg=0.06, dac_select=0.02, dac_mono=0.01, pa_dac_orth=0.025)
        args.exp_desc = "PA主导：提高PA分支分类/回归/选择性权重，压低DAC"

    elif g == "g7_dac_dominant_check":
        set_loss_pack(args, pa_cls=0.16, dac_cls=0.28, pa_kl=0.03, dac_kl=0.06,
                      pa_reg=0.12, pa_select=0.05, pa_mono=0.03,
                      dac_reg=0.16, dac_select=0.07, dac_mono=0.04, pa_dac_orth=0.02)
        args.exp_desc = "DAC主导反证：故意提高DAC权重做对照"

    elif g == "g8_no_dac_aux":
        args.enable_dac_aux = False
        set_loss_pack(args, pa_cls=0.32, dac_cls=0.12, pa_kl=0.06, dac_kl=0.0,
                      pa_reg=0.24, pa_select=0.12, pa_mono=0.07,
                      dac_reg=0.0, dac_select=0.0, dac_mono=0.0, pa_dac_orth=0.02)
        args.exp_desc = "去DAC辅助：检验PA能否独立承担主提升"

    elif g == "g9_no_pa_aux":
        args.enable_pa_aux = False
        set_loss_pack(args, pa_cls=0.20, dac_cls=0.18, pa_kl=0.0, dac_kl=0.04,
                      pa_reg=0.0, pa_select=0.0, pa_mono=0.0,
                      dac_reg=0.12, dac_select=0.05, dac_mono=0.03, pa_dac_orth=0.02)
        args.exp_desc = "去PA辅助：验证PA损失是否真带来收益"

    elif g == "g10_pa_aug_strong":
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.30
        args.aug_scale_max = 0.45
        args.aug_pa_mp_sigma = 0.05
        args.aug_pa_mem_sigma = 0.04
        args.aug_pa_ampm_max = 0.18
        set_loss_pack(args, pa_cls=0.30, dac_cls=0.16, pa_kl=0.06, dac_kl=0.03,
                      pa_reg=0.24, pa_select=0.12, pa_mono=0.07,
                      dac_reg=0.08, dac_select=0.03, dac_mono=0.02, pa_dac_orth=0.02)
        args.exp_desc = "强PA增强：检验主视图PA扰动强度对跨域的影响"

    elif g == "g11_slow_aux_ramp":
        args.aux_warmup_epochs = 5
        args.aux_ramp_epochs = 25
        args.mono_warmup_epochs = 12
        args.mono_ramp_epochs = 18
        set_loss_pack(args, pa_cls=0.28, dac_cls=0.18, pa_kl=0.06, dac_kl=0.04,
                      pa_reg=0.20, pa_select=0.10, pa_mono=0.06,
                      dac_reg=0.10, dac_select=0.04, dac_mono=0.02, pa_dac_orth=0.02)
        args.exp_desc = "慢启动：辅助/单调性更晚介入，减少早期过约束"

    elif g == "g12_fast_aux_ramp":
        args.aux_warmup_epochs = 1
        args.aux_ramp_epochs = 8
        args.mono_warmup_epochs = 4
        args.mono_ramp_epochs = 8
        set_loss_pack(args, pa_cls=0.28, dac_cls=0.18, pa_kl=0.06, dac_kl=0.04,
                      pa_reg=0.20, pa_select=0.10, pa_mono=0.06,
                      dac_reg=0.10, dac_select=0.04, dac_mono=0.02, pa_dac_orth=0.02)
        args.exp_desc = "快启动：辅助更早介入，检验训练稳定性与收敛速度"

    elif g == "g13_pa_select_mono_high":
        set_loss_pack(args, pa_cls=0.28, dac_cls=0.16, pa_kl=0.06, dac_kl=0.03,
                      pa_reg=0.20, pa_select=0.16, pa_mono=0.10,
                      dac_reg=0.08, dac_select=0.03, dac_mono=0.02, pa_dac_orth=0.025)
        args.exp_desc = "高选择性/高单调：重点测试PA选择性与强度排序约束"
    else:
        raise ValueError(f"Unknown exp_group={g}")

    if default_is_path(args.best_save_path, "best_model_coop.pth"):
        args.best_save_path = f"best_{g}.pth"
    if default_is_path(args.latest_save_path, "latest_model_coop.pth"):
        args.latest_save_path = f"latest_{g}.pth"
    return args



def format_epoch_block(
    epoch: int,
    epochs: int,
    lr: float,
    epoch_time_s: float,
    meters: Dict[str, AverageMeter],
    val_stats: Dict[str, float],
    test_stats: Dict[str, float],
    named_test_stats: Dict[str, Dict[str, float]],
    named_test_meta: Dict[str, Dict[str, Any]],
    best_val_main: float,
    best_test_main: float,
    best_epoch: int,
    latest_path: str,
    best_path: str,
    is_best: bool,
    aug_state: Optional[Dict[str, Any]],
    branch_scale: float,
    aux_scale: float,
    mono_scale: float,
):
    sep = "=" * 136
    minor = "-" * 136
    lines = [sep]
    lines.append(
        f"[Epoch {epoch:03d}/{epochs:03d}] time={epoch_time_s:.1f}s | lr={lr:.2e} | branch_scale={branch_scale:.3f} aux_scale={aux_scale:.3f} mono_scale={mono_scale:.3f}"
    )
    if aug_state is not None:
        lines.append(
            "[AUG] "
            f"scale={aug_state['scale']:.3f} | p_dac={aug_state['p_dac']:.3f} p_pa={aug_state['p_pa']:.3f} "
            f"p_shift={aug_state['p_time_shift']:.3f} p_cfo={aug_state['p_cfo']:.3f} "
            f"p_awgn={aug_state['p_awgn']:.3f} p_mp={aug_state['p_multipath']:.3f} | "
            f"max_shift={aug_state['max_time_shift']} cfo_max={aug_state['cfo_max']:.4g} pn_max={aug_state['phase_noise_sigma_max']:.4g}"
        )
    else:
        lines.append("[AUG] disabled")
    lines.append(minor)
    lines.append(
        "[LOSS-CORE] "
        f"total={meters['loss'].avg:.4f} main={meters['main_ce'].avg:.4f} pa_cls={meters['pa_cls'].avg:.4f} dac_cls={meters['dac_cls'].avg:.4f} "
        f"pa_kl={meters['pa_kl'].avg:.4f} dac_kl={meters['dac_kl'].avg:.4f} orth={meters['orth'].avg:.4f}"
    )
    lines.append(
        "[LOSS-AUX]  "
        f"pa_reg={meters['pa_reg'].avg:.4f} pa_sel={meters['pa_sel'].avg:.4f} pa_mono={meters['pa_mono'].avg:.4f} | "
        f"dac_reg={meters['dac_reg'].avg:.4f} dac_sel={meters['dac_sel'].avg:.4f} dac_mono={meters['dac_mono'].avg:.4f}"
    )
    lines.append(
        "[SHIFT/COS] "
        f"gap_pa={meters['gap_pa'].avg:.4f} gap_dac={meters['gap_dac'].avg:.4f} "
        f"cos_pa={meters['cos_pa'].avg:.4f} cos_dac={meters['cos_dac'].avg:.4f}"
    )
    lines.append(
        "[TRAIN] "
        f"main={meters['main_acc'].avg:.2f}% pa={meters['pa_acc'].avg:.2f}% dac={meters['dac_acc'].avg:.2f}%"
    )
    lines.append(
        "[VAL]   "
        f"main={val_stats['main_acc']:.2f}% pa={val_stats['pa_acc']:.2f}% dac={val_stats['dac_acc']:.2f}%"
    )
    lines.append(
        "[TEST]  "
        f"main={test_stats['main_acc']:.2f}% pa={test_stats['pa_acc']:.2f}% dac={test_stats['dac_acc']:.2f}% ({test_stats['main_correct']}/{test_stats['total']})"
    )
    lines.append("[TEST-SPLIT]")
    lines.extend(format_named_test_lines(named_test_stats, named_test_meta))
    lines.append(f"[BEST]  val_main={best_val_main:.2f}% & test_main={best_test_main:.2f}% @ E{best_epoch:03d}")
    lines.append(f"[CKPT] latest -> {latest_path} | best -> {best_path}{' (updated)' if is_best else ''}")
    lines.append(sep)
    return "\n".join(lines)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wisig", choices=["wisig", "oralce"])
    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_equalized", type=str, default="1")
    parser.add_argument("--wisig_domain", type=str, default="day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_train_ratio", type=float, default=0.8)
    parser.add_argument("--wisig_guard_gap", type=int, default=8)
    parser.add_argument("--wisig_train_days", type=str, default="0")
    parser.add_argument("--wisig_test_days", type=str, default="2,3")
    parser.add_argument("--wisig_train_rxs", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--wisig_test_rxs", type=str, default="6,7,8,9,10,11")
    parser.add_argument("--wisig_max_day123_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_val_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)

    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--aux_head_lr_mult", type=float, default=1.0)

    parser.add_argument("--branch_warmup_epochs", type=int, default=1)
    parser.add_argument("--branch_ramp_epochs", type=int, default=8)
    parser.add_argument("--aux_warmup_epochs", type=int, default=3)
    parser.add_argument("--aux_ramp_epochs", type=int, default=15)
    parser.add_argument("--mono_warmup_epochs", type=int, default=8)
    parser.add_argument("--mono_ramp_epochs", type=int, default=12)
    parser.add_argument("--robust_temp", type=float, default=1.0)
    parser.add_argument("--select_margin", type=float, default=0.03)
    parser.add_argument("--mono_margin", type=float, default=0.0)

    parser.add_argument("--lambda_main_ce", type=float, default=1.0)
    parser.add_argument("--lambda_pa_cls", type=float, default=0.25)
    parser.add_argument("--lambda_dac_cls", type=float, default=0.20)
    parser.add_argument("--lambda_pa_kl", type=float, default=0.06)
    parser.add_argument("--lambda_dac_kl", type=float, default=0.04)
    parser.add_argument("--lambda_pa_reg", type=float, default=0.18)
    parser.add_argument("--lambda_pa_select", type=float, default=0.08)
    parser.add_argument("--lambda_pa_mono", type=float, default=0.05)
    parser.add_argument("--lambda_dac_reg", type=float, default=0.10)
    parser.add_argument("--lambda_dac_select", type=float, default=0.04)
    parser.add_argument("--lambda_dac_mono", type=float, default=0.02)
    parser.add_argument("--lambda_pa_dac_orth", type=float, default=0.02)

    parser.add_argument(
        "--exp_group",
        type=str,
        default="g4_full_balanced",
        choices=[
            "g0_baseline_joint",
            "g1_clean_coop",
            "g2_pa_aux",
            "g3_dac_aux",
            "g4_full_balanced",
            "g5_pa_main_plus_aux",
            "g6_pa_dominant",
            "g7_dac_dominant_check",
            "g8_no_dac_aux",
            "g9_no_pa_aux",
            "g10_pa_aug_strong",
            "g11_slow_aux_ramp",
            "g12_fast_aux_ramp",
            "g13_pa_select_mono_high",
        ],
        help="预设实验组，覆盖基线、协同头、PA/DAC消融、权重倾斜、增强强度与调度。",
    )
    add_bool_arg(parser, "enable_clean_branch_cls", True, "启用clean视图的PA/DAC分支分类损失", "关闭clean视图的PA/DAC分支分类损失")
    add_bool_arg(parser, "enable_pa_aux", True, "启用PA-only辅助视图与PA辅助损失", "关闭PA-only辅助视图与PA辅助损失")
    add_bool_arg(parser, "enable_dac_aux", True, "启用DAC-only辅助视图与DAC辅助损失", "关闭DAC-only辅助视图与DAC辅助损失")

    add_bool_arg(parser, "use_aug", True, "启用训练时数据增强", "关闭训练时数据增强")
    add_bool_arg(parser, "aug_enable_class_signature", False, "增强中启用类签名偏置", "增强中关闭类签名偏置")
    add_bool_arg(parser, "aug_enable_pa_normal", False, "正常训练主视图也启用 PA 增强", "正常训练主视图不启用 PA 增强")
    add_bool_arg(parser, "aug_defect_apply_channel", True, "缺陷视图同时叠加通道与 anti-shortcut", "缺陷视图不叠加通道与 anti-shortcut")

    parser.add_argument("--aug_scale_min", type=float, default=0.10)
    parser.add_argument("--aug_scale_max", type=float, default=0.35)
    parser.add_argument("--aug_warmup_epochs", type=int, default=3)
    parser.add_argument("--aug_ramp_epochs", type=int, default=15)
    parser.add_argument("--aug_ramp_curve", type=float, default=1.25)

    parser.add_argument("--aug_p_dac", type=float, default=0.25)
    parser.add_argument("--aug_p_pa", type=float, default=0.3)
    parser.add_argument("--aug_class_sig_mix", type=float, default=0.1)

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

    parser.add_argument("--aug_pa_mp_sigma", type=float, default=0.04)
    parser.add_argument("--aug_pa_mem_sigma", type=float, default=0.03)
    parser.add_argument("--aug_pa_ampm_max", type=float, default=0.15)
    parser.add_argument("--aug_pa_iq_img_max", type=float, default=0.015)

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--best_save_path", type=str, default="best_model_g4_full_balanced_v2.pth")
    parser.add_argument("--latest_save_path", type=str, default="latest_model_g4_full_balanced_v2.pth")
    args = parser.parse_args()
    args = apply_experiment_preset(args)

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
    print(
        f"[EXP] clean_branch_cls={args.enable_clean_branch_cls} pa_aux={args.enable_pa_aux} dac_aux={args.enable_dac_aux} | "
        f"pa_main={args.aug_enable_pa_normal} aug_p_pa={args.aug_p_pa:.3f} aug_p_dac={args.aug_p_dac:.3f}"
    )

    if float(args.sample_rate_hz) <= 0.0:
        args.sample_rate_hz = 25e6 if args.dataset == "wisig" else 5e6

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=(args.num_workers > 0))
    named_test_loaders = {
        k: DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                      pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=(args.num_workers > 0))
        for k, ds in named_tests.items()
    }

    model = build_model(
        num_classes=args.num_classes,
        model_size=args.model_size,
        dataset=args.dataset,
        input_len=input_len,
        sample_rate_hz=float(args.sample_rate_hz),
    ).to(device)
    emb_dim = int(getattr(model, "f_proj").out_features)
    aux_heads = BranchAuxHeads(emb_dim=emb_dim, num_classes=args.num_classes).to(device)
    print(f"[MODEL] CVSincNet emb_dim={emb_dim} num_classes={args.num_classes}")

    aug_base_cfg = build_aug_base_cfg(args) if args.use_aug else None
    augmentor = make_augmentor(aug_base_cfg) if args.use_aug else None
    if args.use_aug:
        print(
            "[AUG-INIT] enabled | "
            f"scale:[{args.aug_scale_min:.2f}->{args.aug_scale_max:.2f}] warmup={args.aug_warmup_epochs} ramp={args.aug_ramp_epochs} "
            f"curve={args.aug_ramp_curve:.2f} | base_p_dac={args.aug_p_dac:.2f} base_p_pa={args.aug_p_pa:.2f}"
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": aux_heads.parameters(), "lr": args.lr * float(args.aux_head_lr_mult)},
        ],
        weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_tx = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    best_val_main = -1.0
    best_test_main = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        aux_heads.train()
        t0 = time.time()
        meters = {k: AverageMeter() for k in [
            "loss", "main_ce", "pa_cls", "dac_cls", "pa_kl", "dac_kl", "orth",
            "pa_reg", "pa_sel", "pa_mono", "dac_reg", "dac_sel", "dac_mono",
            "gap_pa", "gap_dac", "cos_pa", "cos_dac", "main_acc", "pa_acc", "dac_acc",
        ]}
        aug_state = configure_augmentor_for_epoch(augmentor, aug_base_cfg, epoch, args) if augmentor is not None else None
        branch_scale = ramp_value(epoch, int(args.branch_warmup_epochs), int(args.branch_ramp_epochs), 0.0, 1.0, 1.0)
        aux_scale = ramp_value(epoch, int(args.aux_warmup_epochs), int(args.aux_ramp_epochs), 0.0, 1.0, 1.0)
        mono_scale = ramp_value(epoch, int(args.mono_warmup_epochs), int(args.mono_ramp_epochs), 0.0, 1.0, 1.0)

        for batch in train_loader:
            x, y, _ = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if augmentor is not None:
                with torch.no_grad():
                    x_main = augmentor(x, labels=y, no_pa=(not args.aug_enable_pa_normal))
                    if args.enable_dac_aux:
                        x_dac, s_dac = augmentor(x, labels=y, dac_only=True, return_dac_strength=True)
                    else:
                        x_dac = x
                        s_dac = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                    if args.enable_pa_aux:
                        x_pa, s_pa = augmentor(x, labels=y, pa_only=True, return_pa_strength=True)
                    else:
                        x_pa = x
                        s_pa = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            else:
                x_main = x
                x_dac = x
                x_pa = x
                s_dac = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                s_pa = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

            with torch.no_grad():
                anchor = model(x, y=y, return_aux=True)
                feat_pa_clean = anchor["feat_pa"]
                feat_dac_clean = anchor["feat_dac"]
                main_logits_clean = anchor["logits"]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_main = model(x_main, y=y, return_aux=True)
                out_dac = model(x_dac, y=y, return_aux=True) if args.enable_dac_aux else None
                out_pa = model(x_pa, y=y, return_aux=True) if args.enable_pa_aux else None

                main_logits = out_main["logits"]
                feat_pa_main = out_main["feat_pa"]
                feat_dac_main = out_main["feat_dac"]

                pa_logits_main = aux_heads.pa_head(feat_pa_main, labels=y)
                dac_logits_main = aux_heads.dac_head(feat_dac_main, labels=y)

                loss_main_ce = ce_tx(main_logits.float(), y)

                if args.enable_clean_branch_cls:
                    loss_pa_cls = ce_tx(pa_logits_main.float(), y)
                    loss_dac_cls = ce_tx(dac_logits_main.float(), y)
                    loss_pa_kl = one_way_kl_from_teacher(pa_logits_main, main_logits_clean, temperature=float(args.robust_temp))
                    loss_dac_kl = one_way_kl_from_teacher(dac_logits_main, main_logits_clean, temperature=float(args.robust_temp))
                else:
                    loss_pa_cls = main_logits.new_tensor(0.0)
                    loss_dac_cls = main_logits.new_tensor(0.0)
                    loss_pa_kl = main_logits.new_tensor(0.0)
                    loss_dac_kl = main_logits.new_tensor(0.0)

                if args.enable_pa_aux:
                    feat_pa_pa = out_pa["feat_pa"]
                    pa_pred = out_pa["pa_pred"]
                    pa_logits_aux = aux_heads.pa_head(feat_pa_pa, labels=y)
                    loss_pa_cls = 0.5 * loss_pa_cls + 0.5 * ce_tx(pa_logits_aux.float(), y) if args.enable_clean_branch_cls else ce_tx(pa_logits_aux.float(), y)
                    shift_pa_on_pa = cosine_distance_per_sample(feat_pa_clean, feat_pa_pa)
                    if args.enable_dac_aux:
                        shift_pa_on_dac = cosine_distance_per_sample(feat_pa_clean, out_dac["feat_pa"])
                    else:
                        shift_pa_on_dac = torch.zeros_like(shift_pa_on_pa)
                    loss_pa_reg = smooth_strength_loss(pa_pred, s_pa)
                    loss_pa_sel = selective_shift_margin_loss(shift_pa_on_pa, shift_pa_on_dac, margin=float(args.select_margin))
                    loss_pa_mono = pairwise_monotonic_loss(s_pa, shift_pa_on_pa, margin=float(args.mono_margin))
                    _, cos_pa = cosine_consistency_loss(feat_pa_clean, feat_pa_pa)
                else:
                    shift_pa_on_pa = torch.zeros(y.size(0), device=main_logits.device, dtype=main_logits.dtype)
                    shift_pa_on_dac = torch.zeros_like(shift_pa_on_pa)
                    loss_pa_reg = main_logits.new_tensor(0.0)
                    loss_pa_sel = main_logits.new_tensor(0.0)
                    loss_pa_mono = main_logits.new_tensor(0.0)
                    cos_pa = float("nan")

                if args.enable_dac_aux:
                    feat_dac_dac = out_dac["feat_dac"]
                    dac_pred = out_dac["dac_pred"]
                    dac_logits_aux = aux_heads.dac_head(feat_dac_dac, labels=y)
                    loss_dac_cls = 0.5 * loss_dac_cls + 0.5 * ce_tx(dac_logits_aux.float(), y) if args.enable_clean_branch_cls else ce_tx(dac_logits_aux.float(), y)
                    shift_dac_on_dac = cosine_distance_per_sample(feat_dac_clean, feat_dac_dac)
                    if args.enable_pa_aux:
                        shift_dac_on_pa = cosine_distance_per_sample(feat_dac_clean, out_pa["feat_dac"])
                    else:
                        shift_dac_on_pa = torch.zeros_like(shift_dac_on_dac)
                    loss_dac_reg = smooth_strength_loss(dac_pred, s_dac)
                    loss_dac_sel = selective_shift_margin_loss(shift_dac_on_dac, shift_dac_on_pa, margin=float(args.select_margin))
                    loss_dac_mono = pairwise_monotonic_loss(s_dac, shift_dac_on_dac, margin=float(args.mono_margin))
                    _, cos_dac = cosine_consistency_loss(feat_dac_clean, feat_dac_dac)
                else:
                    shift_dac_on_dac = torch.zeros(y.size(0), device=main_logits.device, dtype=main_logits.dtype)
                    shift_dac_on_pa = torch.zeros_like(shift_dac_on_dac)
                    loss_dac_reg = main_logits.new_tensor(0.0)
                    loss_dac_sel = main_logits.new_tensor(0.0)
                    loss_dac_mono = main_logits.new_tensor(0.0)
                    cos_dac = float("nan")

                loss_orth = covariance_orth_loss(feat_pa_main, feat_dac_main)

                loss = (
                    float(args.lambda_main_ce) * loss_main_ce
                    + branch_scale * (
                        float(args.lambda_pa_cls) * loss_pa_cls
                        + float(args.lambda_dac_cls) * loss_dac_cls
                        + float(args.lambda_pa_kl) * loss_pa_kl
                        + float(args.lambda_dac_kl) * loss_dac_kl
                        + float(args.lambda_pa_dac_orth) * loss_orth
                    )
                    + aux_scale * (
                        float(args.lambda_pa_reg) * loss_pa_reg
                        + float(args.lambda_pa_select) * loss_pa_sel
                        + float(args.lambda_dac_reg) * loss_dac_reg
                        + float(args.lambda_dac_select) * loss_dac_sel
                    )
                    + mono_scale * (
                        float(args.lambda_pa_mono) * loss_pa_mono
                        + float(args.lambda_dac_mono) * loss_dac_mono
                    )
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux_heads.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bsz = x.size(0)
            meters["loss"].update(loss.item(), bsz)
            meters["main_ce"].update(loss_main_ce.item(), bsz)
            meters["pa_cls"].update(loss_pa_cls.item(), bsz)
            meters["dac_cls"].update(loss_dac_cls.item(), bsz)
            meters["pa_kl"].update(loss_pa_kl.item(), bsz)
            meters["dac_kl"].update(loss_dac_kl.item(), bsz)
            meters["orth"].update(loss_orth.item(), bsz)
            meters["pa_reg"].update(loss_pa_reg.item(), bsz)
            meters["pa_sel"].update(loss_pa_sel.item(), bsz)
            meters["pa_mono"].update(loss_pa_mono.item(), bsz)
            meters["dac_reg"].update(loss_dac_reg.item(), bsz)
            meters["dac_sel"].update(loss_dac_sel.item(), bsz)
            meters["dac_mono"].update(loss_dac_mono.item(), bsz)
            meters["gap_pa"].update((shift_pa_on_pa.mean() - shift_pa_on_dac.mean()).item(), bsz)
            meters["gap_dac"].update((shift_dac_on_dac.mean() - shift_dac_on_pa.mean()).item(), bsz)
            meters["cos_pa"].update(cos_pa, bsz)
            meters["cos_dac"].update(cos_dac, bsz)
            meters["main_acc"].update(accuracy_from_logits(main_logits, y), bsz)
            meters["pa_acc"].update(accuracy_from_logits(pa_logits_main, y), bsz)
            meters["dac_acc"].update(accuracy_from_logits(dac_logits_main, y), bsz)

        scheduler.step()

        val_stats = evaluate_loader(model, aux_heads, val_loader, device=device, max_batches=int(args.eval_max_batches))
        named_test_stats = evaluate_named_loaders(model, aux_heads, named_test_loaders, device=device, max_batches=int(args.eval_max_batches))
        test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(named_test_stats.keys())
        test_stats = aggregate_named_stats(named_test_stats, test_keys)

        is_best = (val_stats["main_acc"] > best_val_main)
        if is_best:
            best_val_main = val_stats["main_acc"]
            best_test_main = test_stats["main_acc"]
            best_epoch = epoch
            save_checkpoint(args.best_save_path, model=model, aux_heads=aux_heads, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info,
                            stats={
                                "train_main_acc": meters["main_acc"].avg,
                                "train_pa_acc": meters["pa_acc"].avg,
                                "train_dac_acc": meters["dac_acc"].avg,
                                "val": val_stats,
                                "test": test_stats,
                                "test_named": named_test_stats,
                                "best_epoch": epoch,
                                "branch_scale": branch_scale,
                                "aux_scale": aux_scale,
                                "mono_scale": mono_scale,
                                "aug_state": aug_state,
                            })

        save_checkpoint(args.latest_save_path, model=model, aux_heads=aux_heads, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch, args=args, split_info=split_info,
                        stats={
                            "train_main_acc": meters["main_acc"].avg,
                            "train_pa_acc": meters["pa_acc"].avg,
                            "train_dac_acc": meters["dac_acc"].avg,
                            "val": val_stats,
                            "test": test_stats,
                            "test_named": named_test_stats,
                            "best_val_main_so_far": best_val_main,
                            "best_test_main_so_far": best_test_main,
                            "best_epoch_so_far": best_epoch,
                            "branch_scale": branch_scale,
                            "aux_scale": aux_scale,
                            "mono_scale": mono_scale,
                            "aug_state": aug_state,
                        })

        print(format_epoch_block(epoch, args.epochs, optimizer.param_groups[0]["lr"], time.time() - t0,
                                 meters, val_stats, test_stats, named_test_stats, named_test_meta,
                                 best_val_main, best_test_main, best_epoch,
                                 args.latest_save_path, args.best_save_path, is_best,
                                 aug_state, branch_scale, aux_scale, mono_scale),
              flush=True)

    print(f"Training finished. best_val_main_acc={best_val_main:.2f}% & best_test_main_acc={best_test_main:.2f}% at epoch {best_epoch}")
    if split_info is not None:
        print(f"Final split info: {split_info}")

    try:
        ckpt = torch.load(args.best_save_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("aux_heads", None) is not None:
            aux_heads.load_state_dict(ckpt["aux_heads"], strict=False)
        final_val = evaluate_loader(model, aux_heads, val_loader, device=device, max_batches=0)
        final_named = evaluate_named_loaders(model, aux_heads, named_test_loaders, device=device, max_batches=0)
        test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(final_named.keys())
        final_test = aggregate_named_stats(final_named, test_keys)
        print(
            f"[FINAL-BEST] val(main/pa/dac)={final_val['main_acc']:.2f}/{final_val['pa_acc']:.2f}/{final_val['dac_acc']:.2f}% | "
            f"test(main/pa/dac)={final_test['main_acc']:.2f}/{final_test['pa_acc']:.2f}/{final_test['dac_acc']:.2f}%"
        )
        for line in format_named_test_lines(final_named, named_test_meta):
            print(f"[FINAL-BEST] {line.strip()}")
    except Exception as e:
        print(f"[WARN] final best-checkpoint test failed: {e}", flush=True)


if __name__ == "__main__":
    main()
