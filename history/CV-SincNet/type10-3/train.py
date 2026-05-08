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
    from model_dual_cvsincnet import build_dual_model
except Exception:
    from model_dual_cvsincnet_stagewise_v2 import build_dual_model
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


def normalized_accuracy(acc_pct: float, n_cls: int) -> float:
    if n_cls <= 1:
        return 0.0
    chance = 100.0 / float(n_cls)
    return max(0.0, (float(acc_pct) - chance) / max(1e-6, 100.0 - chance))


def covariance_orth_loss(z_id: torch.Tensor, z_dom: torch.Tensor) -> torch.Tensor:
    z_id = z_id.float() - z_id.float().mean(dim=0, keepdim=True)
    z_dom = z_dom.float() - z_dom.float().mean(dim=0, keepdim=True)
    n = z_id.size(0)
    if n <= 1:
        return z_id.new_tensor(0.0)
    cov = (z_id.t() @ z_dom) / float(n - 1)
    return torch.mean(cov * cov)


def same_tx_cross_domain_consistency(z_id: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    if d is None:
        return z_id.new_tensor(0.0), float("nan")
    z = F.normalize(z_id.float(), dim=1)
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
            cents.append(F.normalize(z[m].mean(dim=0, keepdim=True), dim=1).squeeze(0))
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


def should_warn_domain_bias(dbi_hist, patience: int = 3, threshold: float = 0.05) -> bool:
    if len(dbi_hist) < patience:
        return False
    tail = dbi_hist[-patience:]
    return all(v > threshold for v in tail) and all(tail[i] >= tail[i - 1] - 1e-6 for i in range(1, len(tail)))


def save_checkpoint(path: str, *, model, optimizer, scheduler, scaler, epoch: int, args, split_info, stats: dict):
    payload = {
        "model": model.state_dict(),
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


def safe_nan(v: float) -> str:
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"


def finite_or_zero(t: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if t is None:
        return ref.new_tensor(0.0)
    if not torch.is_tensor(t):
        try:
            t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)
        except Exception:
            return ref.new_tensor(0.0)
    if not torch.isfinite(t).all():
        return ref.new_tensor(0.0)
    return t


def default_is_path(p: str, default_name: str) -> bool:
    return str(p).strip() == default_name


def set_pa_weights(args, *, cls_pa: float, joint_inv: float, imp_inv: float, pa_kl: float, pa_reg: float, pa_select: float, pa_mono: float):
    args.lambda_cls_pa = float(cls_pa)
    args.lambda_pa_joint_inv = float(joint_inv)
    args.lambda_pa_imp_inv = float(imp_inv)
    args.lambda_pa_kl = float(pa_kl)
    args.lambda_pa_reg = float(pa_reg)
    args.lambda_pa_select = float(pa_select)
    args.lambda_pa_mono = float(pa_mono)


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
    args.lambda_dac_select = float(dac_select)
    args.lambda_dac_mono = float(dac_mono)


def apply_experiment_preset(args):
    g_raw = str(args.exp_group).strip()
    alias_map = {
        # new stagewise names
        "s1_core_only": "s1_core_only",
        "s2_pure_aux_no_select": "s2_pure_aux_no_select",
        "s3_stagewise_pa_focus": "s3_stagewise_pa_focus",
        "s4_stagewise_full_dual": "s4_stagewise_full_dual",
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
        args.exp_desc = "S1 仅主任务：cls + dom/adv/orth/cons/probe，不启用 DAC/PA 辅助"
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
        args.exp_desc = "S3 PA 重点阶段式：主视图温和 PA + 纯 PA-only 辅助，select/mono 仅在后期启用"
    elif g == "s4_stagewise_full_dual":
        args.enable_dac_aux = True
        args.enable_pa_aux = True
        args.aug_enable_pa_normal = True
        args.aug_p_pa = 0.18
        args.aug_p_dac = 0.22
        args.lambda_cls_dac = 0.10
        args.lambda_dac_reg = 0.25
        args.lambda_dac_select = 0.10
        args.lambda_dac_mono = 0.05
        set_pa_weights(args, cls_pa=0.30, joint_inv=0.10, imp_inv=0.00, pa_kl=0.04, pa_reg=0.18, pa_select=0.08, pa_mono=0.05)
        args.lambda_cross_zero = 0.05
        args.exp_desc = "S4 双缺陷阶段式：joint 特征去域 + 纯 DAC/PA-only 辅助 + select/mono/cross_zero 后期启用"
    else:
        raise ValueError(f"Internal exp_group dispatch failure: {g}")

    if not args.enable_pa_aux or not args.enable_dac_aux:
        args.lambda_cross_zero = 0.0

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


def build_stage_state(epoch: int, args) -> Dict[str, float]:
    e1 = int(max(0, args.stage1_epochs))
    e2 = int(max(e1, args.stage2_epochs))
    r3 = int(max(1, args.stage3_ramp_epochs))

    if epoch <= e1:
        return {
            "phase": "S1_core",
            "use_aux_views": 0.0,
            "dom_scale": 1.0,
            "adv_scale": 0.70,
            "orth_scale": 0.50,
            "cons_scale": 0.00,
            "probe_scale": 1.0,
            "cls_aux_scale": 0.0,
            "reg_aux_scale": 0.0,
            "joint_inv_scale": 0.0,
            "imp_inv_scale": 0.0,
            "kl_scale": 0.0,
            "select_scale": 0.0,
            "mono_scale": 0.0,
            "cross_scale": 0.0,
        }

    if epoch <= e2:
        t = ramp_value(epoch, args.epochs, e1, max(1, e2 - e1), 0.0, 1.0, 1.0)
        return {
            "phase": "S2_stabilize_aux",
            "use_aux_views": 1.0,
            "dom_scale": 1.0,
            "adv_scale": 1.0,
            "orth_scale": 1.0,
            "cons_scale": 0.60 + 0.40 * t,
            "probe_scale": 1.0,
            "cls_aux_scale": 0.45 + 0.35 * t,
            "reg_aux_scale": 0.75 + 0.25 * t,
            "joint_inv_scale": 0.35,
            "imp_inv_scale": 0.0,
            "kl_scale": 0.50,
            "select_scale": 0.0,
            "mono_scale": 0.0,
            "cross_scale": 0.0,
        }

    late = ramp_value(epoch, args.epochs, e2, r3, 0.0, 1.0, 1.0)
    return {
        "phase": "S3_selective_late",
        "use_aux_views": 1.0,
        "dom_scale": 1.0,
        "adv_scale": 1.0,
        "orth_scale": 1.0,
        "cons_scale": 1.0,
        "probe_scale": 1.0,
        "cls_aux_scale": 1.0,
        "reg_aux_scale": 1.0,
        "joint_inv_scale": 0.30,
        "imp_inv_scale": 0.0,
        "kl_scale": 0.60,
        "select_scale": late,
        "mono_scale": late,
        "cross_scale": 0.50 * late,
    }


def current_weight_dict(args, stage_state: Dict[str, float]) -> Dict[str, float]:
    return {
        "dom": float(args.lambda_dom) * float(stage_state["dom_scale"]),
        "adv": float(args.lambda_adv) * float(stage_state["adv_scale"]),
        "orth": float(args.lambda_orth) * float(stage_state["orth_scale"]),
        "cons": float(args.lambda_cons) * float(stage_state["cons_scale"]),
        "probe": float(args.lambda_probe) * float(stage_state["probe_scale"]),
        "cls_pa": float(args.lambda_cls_pa) * float(stage_state["cls_aux_scale"]),
        "cls_dac": float(args.lambda_cls_dac) * float(stage_state["cls_aux_scale"]),
        "pa_joint_inv": float(args.lambda_pa_joint_inv) * float(stage_state["joint_inv_scale"]),
        "pa_imp_inv": float(args.lambda_pa_imp_inv) * float(stage_state["imp_inv_scale"]),
        "pa_kl": float(args.lambda_pa_kl) * float(stage_state["kl_scale"]),
        "dac_reg": float(args.lambda_dac_reg) * float(stage_state["reg_aux_scale"]),
        "pa_reg": float(args.lambda_pa_reg) * float(stage_state["reg_aux_scale"]),
        "cross_zero": float(args.lambda_cross_zero) * float(stage_state["cross_scale"]),
        "dac_select": float(args.lambda_dac_select) * float(stage_state["select_scale"]),
        "pa_select": float(args.lambda_pa_select) * float(stage_state["select_scale"]),
        "dac_mono": float(args.lambda_dac_mono) * float(stage_state["mono_scale"]),
        "pa_mono": float(args.lambda_pa_mono) * float(stage_state["mono_scale"]),
    }


def format_stage_state(stage_state: Dict[str, float]) -> str:
    return (
        f"phase={stage_state['phase']} | use_aux={stage_state['use_aux_views']:.1f} "
        f"cons={stage_state['cons_scale']:.2f} cls_aux={stage_state['cls_aux_scale']:.2f} "
        f"reg={stage_state['reg_aux_scale']:.2f} sel={stage_state['select_scale']:.2f} "
        f"mono={stage_state['mono_scale']:.2f} cross={stage_state['cross_scale']:.2f}"
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


def cosine_distance_per_sample(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a.float(), dim=1, eps=eps)
    b = F.normalize(b.float(), dim=1, eps=eps)
    return (1.0 - torch.sum(a * b, dim=1)).clamp_min(0.0)


def cosine_consistency_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, float]:
    dist = cosine_distance_per_sample(a, b, eps=eps)
    cos = (1.0 - dist).mean().item()
    return dist.mean(), float(cos)


def one_way_kl_from_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    T = float(max(1e-6, temperature))
    log_p_s = F.log_softmax(student_logits.float() / T, dim=1)
    p_t = F.softmax(teacher_logits.float().detach() / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


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


def get_nested_tensor(out: Dict[str, Any], top_key: str, nested_group: str, nested_key: str) -> torch.Tensor:
    v = out.get(top_key, None)
    if torch.is_tensor(v):
        return v
    aux = out.get(nested_group, {})
    v = aux.get(nested_key, None) if isinstance(aux, dict) else None
    if not torch.is_tensor(v):
        raise KeyError(f"Cannot find tensor {top_key} / {nested_group}.{nested_key}")
    return v


@torch.no_grad()
def forward_anchor_eval(model, x: torch.Tensor, y: torch.Tensor, grl_lambda: float = 1.0):
    was_training = model.training
    model.eval()
    out = model(x, y_tx=y, grl_lambda=float(grl_lambda), return_aux=True)
    if was_training:
        model.train()
    return out


def build_domain_label_map(dataset) -> Dict[int, int]:
    if not (hasattr(dataset, "index") and hasattr(dataset, "_domain_lut")):
        return {}
    raw_labels = sorted({int(dataset._domain_lut[(it.rx_i, it.day_i)]) for it in dataset.index})
    return {raw: idx for idx, raw in enumerate(raw_labels)}


def remap_domain_tensor(d: Optional[torch.Tensor], domain_label_map: Dict[int, int], device) -> Optional[torch.Tensor]:
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
    probe_correct = probe_total = 0
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
                probe_correct += int((out["probe_dom_logits"][valid].argmax(dim=1) == dom_y).sum().item())
                probe_total += int(dom_y.numel())

        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    return {
        "tx_acc": 100.0 * tx_correct / max(1, tx_total),
        "dom_acc": 100.0 * dom_correct / max(1, dom_total) if dom_total > 0 else float("nan"),
        "probe_dom_acc": 100.0 * probe_correct / max(1, probe_total) if probe_total > 0 else float("nan"),
        "tx_correct": int(tx_correct),
        "tx_total": int(tx_total),
    }


@torch.no_grad()
def evaluate_named_loaders(model, named_loaders: Dict[str, DataLoader], device, domain_label_map: Dict[int, int], max_batches: int = 0):
    out = {}
    for name, loader in named_loaders.items():
        out[name] = evaluate_loader(model, loader, device, domain_label_map=domain_label_map, max_batches=max_batches)
    return out


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
    m_probeacc: NanMeter,
    cons_cos_epoch: float,
    train_dbi: float,
    bias_flag: bool,
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
):
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(f"[Epoch {epoch:03d}/{epochs:03d}] time={epoch_time_s:.1f}s | lr={lr:.2e} | aux_scale={aux_scale:.3f}")
    if stage_state is not None:
        lines.append(f"[STAGE] {format_stage_state(stage_state)}")
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
        f"adv={meters['adv'].avg:.4f} orth={meters['orth'].avg:.4f} cons={meters['cons'].avg:.4f} probe={meters['probe'].avg:.4f}"
    )
    lines.append(
        "[LOSS-AUX]  "
        f"cls_pa={meters['cls_pa'].avg:.4f} cls_dac={meters['cls_dac'].avg:.4f} "
        f"pa_joint_inv={meters['pa_joint_inv'].avg:.4f} pa_imp_inv={meters['pa_imp_inv'].avg:.4f} pa_kl={meters['pa_kl'].avg:.4f} "
        f"dac_reg={meters['dac_reg'].avg:.4f} pa_reg={meters['pa_reg'].avg:.4f} cross0={meters['cross_zero'].avg:.4f}"
    )
    lines.append(
        "[SEL/MONO]  "
        f"dac_sel={meters['dac_sel'].avg:.4f} pa_sel={meters['pa_sel'].avg:.4f} "
        f"dac_mono={meters['dac_mono'].avg:.4f} pa_mono={meters['pa_mono'].avg:.4f} "
        f"gap_dac={meters['gap_dac'].avg:.4f} gap_pa={meters['gap_pa'].avg:.4f} "
        f"cos_joint_pa={meters['cos_joint_pa'].avg:.4f} cos_imp_pa={meters['cos_imp_pa'].avg:.4f}"
    )
    lines.append(
        "[TRAIN] "
        f"tx={meters['txacc'].avg:.2f}% dom={safe_nan(m_domacc.avg)}% probe={safe_nan(m_probeacc.avg)}% "
        f"cons_cos={cons_cos_epoch:.4f} DBI={train_dbi:.4f} bias_flag={int(bias_flag)}"
    )
    lines.append(
        "[VAL]   "
        f"tx={val_stats['tx_acc']:.2f}% dom={safe_nan(val_stats['dom_acc'])}% "
        f"probe={safe_nan(val_stats['probe_dom_acc'])}%"
    )
    lines.append(f"[TEST]  overall_tx={test_stats['tx_acc']:.2f}% ({test_stats['tx_correct']}/{test_stats['tx_total']})")
    lines.append("[TEST-SPLIT]")
    lines.extend(format_named_test_lines(named_test_stats, named_test_meta))
    lines.append(f"[BEST-JOINT]  val_tx={best_joint_val_tx:.2f}% & test_tx={best_joint_test_tx:.2f}% @ E{best_epoch:03d}")
    lines.append(f"[CKPT]  latest -> {latest_path} | best -> {best_path}{' (updated: val improved)' if is_best else ''}")
    if bias_flag:
        lines.append("[BIAS-WARN] DBI连续升高且超过阈值，模型可能正在由学指纹转向学域偏置。")
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
    parser.add_argument("--wisig_train_days", type=str, default="0,1,2")
    parser.add_argument("--wisig_test_days", type=str, default="3")
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--model_size", type=str, default="M")

    parser.add_argument("--lambda_dom", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.5)
    parser.add_argument("--lambda_orth", type=float, default=0.05)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--lambda_probe", type=float, default=1.0)
    parser.add_argument("--grl_lambda", type=float, default=1.0)

    parser.add_argument("--aux_warmup_epochs", type=int, default=3)
    parser.add_argument("--aux_ramp_epochs", type=int, default=25)
    parser.add_argument("--robust_temp", type=float, default=1.0)
    parser.add_argument("--select_margin", type=float, default=0.03)
    parser.add_argument("--mono_margin", type=float, default=0.00)

    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=45)
    parser.add_argument("--stage3_ramp_epochs", type=int, default=20)

    parser.add_argument("--lambda_cls_pa", type=float, default=0.60)
    parser.add_argument("--lambda_cls_dac", type=float, default=0.15)
    parser.add_argument("--lambda_pa_joint_inv", type=float, default=0.25)
    parser.add_argument("--lambda_pa_imp_inv", type=float, default=0.08)
    parser.add_argument("--lambda_pa_kl", type=float, default=0.12)
    parser.add_argument("--lambda_dac_reg", type=float, default=0.35)
    parser.add_argument("--lambda_pa_reg", type=float, default=0.35)
    parser.add_argument("--lambda_cross_zero", type=float, default=0.10)
    parser.add_argument("--lambda_dac_select", type=float, default=0.15)
    parser.add_argument("--lambda_pa_select", type=float, default=0.15)
    parser.add_argument("--lambda_dac_mono", type=float, default=0.08)
    parser.add_argument("--lambda_pa_mono", type=float, default=0.08)

    parser.add_argument(
        "--exp_group",
        type=str,
        default="s4_stagewise_full_dual",
        choices=["s1_core_only", "s2_pure_aux_no_select", "s3_stagewise_pa_focus", "s4_stagewise_full_dual", "g1_true_no_pa", "g2_pa_aux_only", "g3_pa_main_only", "g4_pa_main_plus_aux", "g5_full_dual_puredefect"],
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

    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--best_save_path", type=str, default="best_model_manyday.pth")
    parser.add_argument("--latest_save_path", type=str, default="latest_model_manyday..pth")
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
    print(f"[EXP] pa_main={args.aug_enable_pa_normal} pa_aux={args.enable_pa_aux} dac_aux={args.enable_dac_aux} | aug_p_pa={args.aug_p_pa:.3f} aug_p_dac={args.aug_p_dac:.3f}")
    print(f"[EXP] pure_views: dac_only(channel={args.aug_dac_only_apply_channel}, anti={args.aug_dac_only_apply_anti_shortcut}) | pa_only(channel={args.aug_pa_only_apply_channel}, anti={args.aug_pa_only_apply_anti_shortcut})")
    print(f"[EXP] stage schedule: stage1<=E{args.stage1_epochs}, stage2<=E{args.stage2_epochs}, stage3 ramp={args.stage3_ramp_epochs}")
    print(f"[EXP] pure_views: dac_only(channel={args.aug_dac_only_apply_channel}, anti={args.aug_dac_only_apply_anti_shortcut}) "
          f"pa_only(channel={args.aug_pa_only_apply_channel}, anti={args.aug_pa_only_apply_anti_shortcut}) "
          f"| defect_strength_mode={args.aug_defect_strength_mode}")

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

    domain_label_map = build_domain_label_map(train_ds)
    num_domains = max(1, len(domain_label_map))
    if len(domain_label_map) > 0:
        print(f"[DOMAIN] train raw->mapped domains={domain_label_map} (used for domain heads / DBI monitor)")
    else:
        print("[DOMAIN] no explicit domain map found; fallback num_domains=1")

    model = build_dual_model(args.num_classes, num_domains, model_size=args.model_size, dataset=args.dataset,
                             input_len=input_len, sample_rate_hz=float(args.sample_rate_hz),
                             id_feature_key="feat_joint", dom_feature_key="feat_imp").to(device)
    print(f"[MODEL] DualCVSincNetDisentangle emb_dim={model.emb_dim} num_domains={num_domains}")

    aug_base_cfg = build_aug_base_cfg(args) if args.use_aug else None
    augmentor = make_augmentor(aug_base_cfg) if args.use_aug else None
    if args.use_aug:
        print(
            "[AUG-INIT] enabled | "
            f"scale:[{args.aug_scale_min:.2f}->{args.aug_scale_max:.2f}] warmup={args.aug_warmup_epochs} ramp={args.aug_ramp_epochs} "
            f"curve={args.aug_ramp_curve:.2f} | base_p_dac={args.aug_p_dac:.2f} base_p_pa={args.aug_p_pa:.2f}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ce_tx = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    ce_dom = nn.CrossEntropyLoss()

    best_joint_val_tx = -1.0
    best_joint_test_tx = -1.0
    best_epoch = -1
    dbi_hist = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        meters = {k: AverageMeter() for k in [
            "loss", "cls", "dom", "adv", "orth", "cons", "probe", "txacc",
            "cls_pa", "cls_dac", "pa_joint_inv", "pa_imp_inv", "pa_kl",
            "dac_reg", "pa_reg", "cross_zero", "dac_sel", "pa_sel", "dac_mono", "pa_mono",
            "gap_dac", "gap_pa", "cos_joint_pa", "cos_imp_pa",
        ]}
        m_domacc, m_probeacc = NanMeter(), NanMeter()
        cons_cos_vals = []
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

            need_dac_aux = bool(args.enable_dac_aux and stage_state["use_aux_views"] > 0.0 and (
                cur_w["cls_dac"] > 0.0 or cur_w["dac_reg"] > 0.0 or cur_w["dac_select"] > 0.0 or cur_w["dac_mono"] > 0.0 or cur_w["cross_zero"] > 0.0
            ))
            need_pa_aux = bool(args.enable_pa_aux and stage_state["use_aux_views"] > 0.0 and (
                cur_w["cls_pa"] > 0.0 or cur_w["pa_joint_inv"] > 0.0 or cur_w["pa_imp_inv"] > 0.0 or cur_w["pa_kl"] > 0.0 or cur_w["pa_reg"] > 0.0 or cur_w["pa_select"] > 0.0 or cur_w["pa_mono"] > 0.0 or cur_w["cross_zero"] > 0.0
            ))

            if augmentor is not None:
                with torch.no_grad():
                    x_main = augmentor(x, labels=y, no_pa=(not args.aug_enable_pa_normal))
                    if need_dac_aux:
                        if str(args.aug_defect_strength_mode).lower() == "tiered":
                            s_dac_in = sample_strength_from_tiers(x.size(0), parse_float_csv(args.aug_dac_only_tiers, [0.15, 0.35, 0.55]), x.device, x.dtype)
                        else:
                            s_dac_in = None
                        x_dac, s_dac = augmentor(x, labels=y, dac_only=True, return_dac_strength=True, dac_strength=s_dac_in)
                    else:
                        x_dac = x
                        s_dac = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                    if need_pa_aux:
                        if str(args.aug_defect_strength_mode).lower() == "tiered":
                            s_pa_in = sample_strength_from_tiers(x.size(0), parse_float_csv(args.aug_pa_only_tiers, [0.15, 0.35, 0.60]), x.device, x.dtype)
                        else:
                            s_pa_in = None
                        x_pa, s_pa = augmentor(x, labels=y, pa_only=True, return_pa_strength=True, pa_strength=s_pa_in)
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
                anchor = forward_anchor_eval(model, x, y, grl_lambda=float(args.grl_lambda))

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_main = model(x_main, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True)
                out_dac = model(x_dac, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True) if need_dac_aux else None
                out_pa = model(x_pa, y_tx=y, grl_lambda=float(args.grl_lambda), return_aux=True) if need_pa_aux else None

                tx_logits = out_main["tx_logits"]
                dom_logits = out_main["dom_logits"]
                adv_dom_logits = out_main["adv_dom_logits"]
                probe_dom_logits = out_main["probe_dom_logits"]
                z_id = out_main["z_id"]
                z_dom = out_main["z_dom"]

                loss_cls = ce_tx(tx_logits.float(), y)
                if d is not None and num_domains > 1:
                    valid = d >= 0
                    if valid.any():
                        d_valid = d[valid]
                        loss_dom = ce_dom(dom_logits[valid].float(), d_valid)
                        loss_adv = ce_dom(adv_dom_logits[valid].float(), d_valid)
                        loss_probe = ce_dom(probe_dom_logits[valid].float(), d_valid)
                        loss_cons, cons_cos = same_tx_cross_domain_consistency(z_id[valid], y[valid], d_valid)
                        if not math.isnan(cons_cos):
                            cons_cos_vals.append(cons_cos)
                        m_domacc.update(accuracy_from_logits(dom_logits[valid], d_valid))
                        m_probeacc.update(accuracy_from_logits(probe_dom_logits[valid], d_valid))
                    else:
                        loss_dom = z_id.new_tensor(0.0)
                        loss_adv = z_id.new_tensor(0.0)
                        loss_probe = z_id.new_tensor(0.0)
                        loss_cons = z_id.new_tensor(0.0)
                else:
                    loss_dom = z_id.new_tensor(0.0)
                    loss_adv = z_id.new_tensor(0.0)
                    loss_probe = z_id.new_tensor(0.0)
                    loss_cons = z_id.new_tensor(0.0)
                loss_orth = covariance_orth_loss(z_id, z_dom)

                clean_joint = get_nested_tensor(anchor, "id_feat_joint", "aux_id", "feat_joint")
                clean_imp = get_nested_tensor(anchor, "id_feat_imp", "aux_id", "feat_imp")
                clean_dac = get_nested_tensor(anchor, "id_feat_dac", "aux_id", "feat_dac")
                clean_pa = get_nested_tensor(anchor, "id_feat_pa", "aux_id", "feat_pa")
                clean_logits = anchor["tx_logits"]

                if need_pa_aux:
                    pa_joint = get_nested_tensor(out_pa, "id_feat_joint", "aux_id", "feat_joint")
                    pa_imp = get_nested_tensor(out_pa, "id_feat_imp", "aux_id", "feat_imp")
                    pa_dac = get_nested_tensor(out_pa, "id_feat_dac", "aux_id", "feat_dac")
                    pa_pa = get_nested_tensor(out_pa, "id_feat_pa", "aux_id", "feat_pa")
                    pa_pred_pa = get_nested_tensor(out_pa, "id_pa_pred", "aux_id", "pa_pred")
                    loss_cls_pa = ce_tx(out_pa["tx_logits"].float(), y)
                    loss_pa_joint_inv, cos_joint_pa = cosine_consistency_loss(pa_joint, clean_joint)
                    loss_pa_imp_inv, cos_imp_pa = cosine_consistency_loss(pa_imp, clean_imp)
                    loss_pa_kl = one_way_kl_from_teacher(out_pa["tx_logits"], clean_logits, temperature=float(args.robust_temp))
                    shift_pa_on_pa = cosine_distance_per_sample(clean_pa, pa_pa)
                    if need_dac_aux:
                        dac_pa = get_nested_tensor(out_dac, "id_feat_pa", "aux_id", "feat_pa")
                        shift_pa_on_dac = cosine_distance_per_sample(clean_pa, dac_pa)
                    else:
                        shift_pa_on_dac = torch.zeros_like(shift_pa_on_pa)
                    loss_pa_select = selective_shift_margin_loss(shift_pa_on_pa, shift_pa_on_dac, margin=float(args.select_margin))
                    loss_pa_mono = pairwise_monotonic_loss(s_pa, shift_pa_on_pa, margin=float(args.mono_margin))
                    loss_pa_reg = smooth_strength_loss(pa_pred_pa, s_pa)
                else:
                    pa_dac = clean_dac
                    loss_cls_pa = z_id.new_tensor(0.0)
                    loss_pa_joint_inv = z_id.new_tensor(0.0)
                    loss_pa_imp_inv = z_id.new_tensor(0.0)
                    loss_pa_kl = z_id.new_tensor(0.0)
                    loss_pa_select = z_id.new_tensor(0.0)
                    loss_pa_mono = z_id.new_tensor(0.0)
                    loss_pa_reg = z_id.new_tensor(0.0)
                    shift_pa_on_pa = torch.zeros(y.size(0), device=z_id.device, dtype=z_id.dtype)
                    shift_pa_on_dac = torch.zeros_like(shift_pa_on_pa)
                    cos_joint_pa = float("nan")
                    cos_imp_pa = float("nan")

                if need_dac_aux:
                    dac_dac = get_nested_tensor(out_dac, "id_feat_dac", "aux_id", "feat_dac")
                    dac_pred_dac = get_nested_tensor(out_dac, "id_dac_pred", "aux_id", "dac_pred")
                    loss_cls_dac = ce_tx(out_dac["tx_logits"].float(), y)
                    shift_dac_on_dac = cosine_distance_per_sample(clean_dac, dac_dac)
                    shift_dac_on_pa = cosine_distance_per_sample(clean_dac, pa_dac) if need_pa_aux else torch.zeros_like(shift_dac_on_dac)
                    loss_dac_select = selective_shift_margin_loss(shift_dac_on_dac, shift_dac_on_pa, margin=float(args.select_margin))
                    loss_dac_mono = pairwise_monotonic_loss(s_dac, shift_dac_on_dac, margin=float(args.mono_margin))
                    loss_dac_reg = smooth_strength_loss(dac_pred_dac, s_dac)
                else:
                    loss_cls_dac = z_id.new_tensor(0.0)
                    loss_dac_select = z_id.new_tensor(0.0)
                    loss_dac_mono = z_id.new_tensor(0.0)
                    loss_dac_reg = z_id.new_tensor(0.0)
                    shift_dac_on_dac = torch.zeros(y.size(0), device=z_id.device, dtype=z_id.dtype)
                    shift_dac_on_pa = torch.zeros_like(shift_dac_on_dac)

                if need_pa_aux and need_dac_aux and cur_w["cross_zero"] > 0.0:
                    dac_pred_pa = get_nested_tensor(out_pa, "id_dac_pred", "aux_id", "dac_pred")
                    pa_pred_dac = get_nested_tensor(out_dac, "id_pa_pred", "aux_id", "pa_pred")
                    zeros = torch.zeros_like(s_dac)
                    loss_cross_zero = 0.5 * (smooth_strength_loss(dac_pred_pa, zeros) + smooth_strength_loss(pa_pred_dac, zeros))
                else:
                    loss_cross_zero = z_id.new_tensor(0.0)

                aux_terms = [
                    cur_w["cls_pa"] * finite_or_zero(loss_cls_pa, z_id),
                    cur_w["cls_dac"] * finite_or_zero(loss_cls_dac, z_id),
                    cur_w["pa_joint_inv"] * finite_or_zero(loss_pa_joint_inv, z_id),
                    cur_w["pa_imp_inv"] * finite_or_zero(loss_pa_imp_inv, z_id),
                    cur_w["pa_kl"] * finite_or_zero(loss_pa_kl, z_id),
                    cur_w["dac_reg"] * finite_or_zero(loss_dac_reg, z_id),
                    cur_w["pa_reg"] * finite_or_zero(loss_pa_reg, z_id),
                    cur_w["cross_zero"] * finite_or_zero(loss_cross_zero, z_id),
                    cur_w["dac_select"] * finite_or_zero(loss_dac_select, z_id),
                    cur_w["pa_select"] * finite_or_zero(loss_pa_select, z_id),
                    cur_w["dac_mono"] * finite_or_zero(loss_dac_mono, z_id),
                    cur_w["pa_mono"] * finite_or_zero(loss_pa_mono, z_id),
                ]

                loss = (
                    finite_or_zero(loss_cls, z_id)
                    + cur_w["dom"] * finite_or_zero(loss_dom, z_id)
                    + cur_w["adv"] * finite_or_zero(loss_adv, z_id)
                    + cur_w["orth"] * finite_or_zero(loss_orth, z_id)
                    + cur_w["cons"] * finite_or_zero(loss_cons, z_id)
                    + cur_w["probe"] * finite_or_zero(loss_probe, z_id)
                    + aux_scale * sum(aux_terms)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bsz = x.size(0)
            meters["loss"].update(loss.item(), bsz)
            meters["cls"].update(loss_cls.item(), bsz)
            meters["dom"].update(loss_dom.item(), bsz)
            meters["adv"].update(loss_adv.item(), bsz)
            meters["orth"].update(loss_orth.item(), bsz)
            meters["cons"].update(loss_cons.item(), bsz)
            meters["probe"].update(loss_probe.item(), bsz)
            meters["txacc"].update(accuracy_from_logits(tx_logits, y), bsz)
            meters["cls_pa"].update(loss_cls_pa.item(), bsz)
            meters["cls_dac"].update(loss_cls_dac.item(), bsz)
            meters["pa_joint_inv"].update(loss_pa_joint_inv.item(), bsz)
            meters["pa_imp_inv"].update(loss_pa_imp_inv.item(), bsz)
            meters["pa_kl"].update(loss_pa_kl.item(), bsz)
            meters["dac_reg"].update(loss_dac_reg.item(), bsz)
            meters["pa_reg"].update(loss_pa_reg.item(), bsz)
            meters["cross_zero"].update(loss_cross_zero.item(), bsz)
            meters["dac_sel"].update(loss_dac_select.item(), bsz)
            meters["pa_sel"].update(loss_pa_select.item(), bsz)
            meters["dac_mono"].update(loss_dac_mono.item(), bsz)
            meters["pa_mono"].update(loss_pa_mono.item(), bsz)
            meters["gap_dac"].update((shift_dac_on_dac.mean() - shift_dac_on_pa.mean()).item(), bsz)
            meters["gap_pa"].update((shift_pa_on_pa.mean() - shift_pa_on_dac.mean()).item(), bsz)
            meters["cos_joint_pa"].update(cos_joint_pa, bsz)
            meters["cos_imp_pa"].update(cos_imp_pa, bsz)

        scheduler.step()

        train_dbi = normalized_accuracy(m_probeacc.avg if m_probeacc.count else 0.0, num_domains) - normalized_accuracy(meters["txacc"].avg, args.num_classes)
        dbi_hist.append(train_dbi)
        bias_flag = should_warn_domain_bias(dbi_hist, patience=3, threshold=0.05)
        cons_cos_epoch = float(np.mean(cons_cos_vals)) if len(cons_cos_vals) > 0 else float("nan")

        val_stats = evaluate_loader(model, val_loader, device, domain_label_map=domain_label_map, max_batches=int(args.eval_max_batches))
        named_test_stats = evaluate_named_loaders(model, named_test_loaders, device, domain_label_map=domain_label_map, max_batches=int(args.eval_max_batches))
        test_keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if args.dataset == "wisig" else list(named_test_stats.keys())
        test_stats = aggregate_named_stats(named_test_stats, test_keys)

        is_best = (val_stats["tx_acc"] > best_joint_val_tx)
        if is_best:
            best_joint_val_tx = val_stats["tx_acc"]
            best_joint_test_tx = test_stats["tx_acc"]
            best_epoch = epoch
            save_checkpoint(args.best_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            epoch=epoch, args=args, split_info=split_info,
                            stats={
                                "train_tx_acc": meters["txacc"].avg,
                                "val_tx_acc": val_stats["tx_acc"],
                                "val_dom_acc": val_stats["dom_acc"],
                                "val_probe_dom_acc": val_stats["probe_dom_acc"],
                                "test_tx_acc": test_stats["tx_acc"],
                                "test_named": named_test_stats,
                                "best_epoch": epoch,
                                "aux_scale": aux_scale,
                                "aug_state": aug_state,
                                "stage_state": stage_state,
                                "best_rule": "val_only",
                            })

        save_checkpoint(args.latest_save_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        epoch=epoch, args=args, split_info=split_info,
                        stats={
                            "train_tx_acc": meters["txacc"].avg,
                            "val_tx_acc": val_stats["tx_acc"],
                            "val_dom_acc": val_stats["dom_acc"],
                            "val_probe_dom_acc": val_stats["probe_dom_acc"],
                            "test_tx_acc": test_stats["tx_acc"],
                            "test_named": named_test_stats,
                            "best_joint_val_tx_acc_so_far": best_joint_val_tx,
                            "best_joint_test_tx_acc_so_far": best_joint_test_tx,
                            "best_epoch_so_far": best_epoch,
                            "aux_scale": aux_scale,
                            "aug_state": aug_state,
                            "stage_state": stage_state,
                        })

        print(format_epoch_block(epoch, args.epochs, optimizer.param_groups[0]["lr"], time.time() - t0,
                                 meters, m_domacc, m_probeacc, cons_cos_epoch, train_dbi, bias_flag,
                                 val_stats, test_stats, named_test_stats, named_test_meta,
                                 best_joint_val_tx, best_joint_test_tx, best_epoch,
                                 args.latest_save_path, args.best_save_path, is_best, aug_state, aux_scale, stage_state),
              flush=True)

    print(f"Training finished. best_joint_val_tx_acc={best_joint_val_tx:.2f}% & best_joint_test_tx_acc={best_joint_test_tx:.2f}% at epoch {best_epoch}")
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
    except Exception as e:
        print(f"[WARN] final best-checkpoint test failed: {e}", flush=True)


if __name__ == "__main__":
    main()
