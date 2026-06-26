from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Dict, Tuple

from DataAugmentation import build_augmentor
from training_controls import compute_mixstyle_epoch_state


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_true: str, help_false: str):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name.replace('-', '_'), action="store_true", help=help_true)
    group.add_argument(f"--no_{name}", dest=name.replace('-', '_'), action="store_false", help=help_false)
    parser.set_defaults(**{name.replace('-', '_'): default})


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

