from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training_test_eval import format_named_test_lines
from cvsrffi.schedule import format_stage_state


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


def safe_nan(v: float) -> str:
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"


def fmt_float(value: float, digits: int = 4) -> str:
    try:
        fv = float(value)
    except Exception:
        return "nan"
    return f"{fv:.{digits}f}" if math.isfinite(fv) else "nan"


def meter_avg(meters: Dict[str, "AverageMeter"], key: str) -> float:
    meter = meters.get(key)
    if meter is None:
        return float("nan")
    return float(meter.avg)


def format_weighted_loss_top(meters: Dict[str, "AverageMeter"], keys: List[str], *, limit: int = 8) -> str:
    values = []
    for key in keys:
        value = meter_avg(meters, key)
        if math.isfinite(value) and abs(value) > 0.0:
            values.append((abs(value), key, value))
    values.sort(reverse=True)
    if not values:
        return "[LOSS-TOP] none"
    parts = [f"{key[2:] if key.startswith('w_') else key}={value:.4f}" for _, key, value in values[: int(limit)]]
    return "[LOSS-TOP] " + " | ".join(parts)


def count_parameters(model) -> Tuple[int, int]:
    raw_model = getattr(model, "_orig_mod", model)
    total = sum(int(p.numel()) for p in raw_model.parameters())
    trainable = sum(int(p.numel()) for p in raw_model.parameters() if p.requires_grad)
    return total, trainable


def print_backbone_config_block(args, *, device, use_amp: bool, input_len: int, num_domains: int, model_params: Tuple[int, int], split_info):
    n_total, n_trainable = model_params
    sep = "=" * 132
    print(sep, flush=True)
    print("[CONFIG-BEGIN] backbone_train.py resolved experiment configuration", flush=True)
    print(
        f"[CONFIG-RUN] time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} run_name={args.run_name} "
        f"seed={args.seed} device={device} amp={int(bool(use_amp))} compile={int(bool(args.compile_model))}",
        flush=True,
    )
    print(
        f"[CONFIG-DATA] dataset={args.dataset} wisig_domain={args.wisig_domain} pkl={args.wisig_pkl} "
        f"train_ratio={args.wisig_train_ratio:.3f} val_ratio={args.wisig_val_ratio:.3f} "
        f"batch={args.batch_size} eval_batch={args.eval_batch_size} workers={args.num_workers} "
        f"input_len={input_len} num_classes={args.num_classes} num_domains={num_domains}",
        flush=True,
    )
    if split_info is not None:
        print(
            f"[CONFIG-SPLIT] train_days={split_info.get('train_days_label', [])} "
            f"train_rxs={split_info.get('train_rxs_idx', [])} "
            f"test_days={split_info.get('test_days_label', [])} "
            f"test_rxs={split_info.get('test_rxs_idx', [])} guard_gap={split_info.get('guard_gap', '-')}",
            flush=True,
        )
    print(
        f"[CONFIG-MODEL] variant={args.model_variant} slim_group={args.slim_group} exp_group={args.exp_group} "
        f"id_branch={args.branch_ablation} domain_branch={args.domain_branch_ablation} "
        f"domain_enhancer={args.domain_enhancer} enhancer_strength={args.domain_enhancer_strength:.3f} "
        f"params_total={n_total:,} params_trainable={n_trainable:,}",
        flush=True,
    )
    print(
        f"[CONFIG-BACKBONE-STABILITY] id_time={getattr(args, 'id_time_stability_mode', 'off')} "
        f"id_freq={getattr(args, 'id_freq_stability_mode', 'off')} "
        f"domain_time={getattr(args, 'domain_time_stability_mode', 'off')} "
        f"domain_freq={getattr(args, 'domain_freq_stability_mode', 'off')} "
        f"time_ch={int(getattr(args, 'time_stability_channels', 8))} "
        f"freq_ch={int(getattr(args, 'freq_stability_channels', 4))}",
        flush=True,
    )
    print(
        f"[CONFIG-MIXSTYLE] enabled={int(bool(args.use_mixstyle))} p={args.mixstyle_p:.3f} "
        f"strength={args.mixstyle_strength:.3f} alpha={args.mixstyle_alpha:.3f} layers={args.mixstyle_layers} "
        f"mix={args.mixstyle_mix} late_start={args.mixstyle_late_start or args.late_stable_start} "
        f"late_ramp={args.mixstyle_late_ramp_epochs or args.late_stable_ramp_epochs} "
        f"late_min_p={args.mixstyle_late_min_p:.3f} late_min_strength={args.mixstyle_late_min_strength:.3f}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] optimizer=AdamW lr={args.lr:.3e} lr_min={args.lr_min:.3e} wd={args.wd:.3e} "
        f"epochs={args.epochs} label_smoothing={args.label_smoothing:.4f} "
        f"clip_backbone={args.clip_grad_backbone:.3f} clip_aux={args.clip_grad_aux:.3f} clip_domain={args.clip_grad_domain:.3f}",
        flush=True,
    )
    print(
        f"[CONFIG-LOSS] lambda_dom={args.lambda_dom:.4f} lambda_adv={args.lambda_adv:.4f} "
        f"lambda_orth={args.lambda_orth:.4f} lambda_cons={args.lambda_cons:.4f} "
        f"lambda_group_ce={args.lambda_group_ce:.4f} group_mode={args.group_ce_mode} "
        f"group_ce_min_domains={int(getattr(args, 'group_ce_min_domains', 0))} "
        f"lambda_proto={args.lambda_proto:.4f} lambda_supcon_id={args.lambda_supcon_id:.4f} "
        f"lambda_fishr={args.lambda_fishr:.4f} fishr_min_domains={int(getattr(args, 'fishr_min_domains', 0))} "
        f"lambda_feature_norm_guard={float(getattr(args, 'lambda_feature_norm_guard', 0.0)):.6f} "
        f"feature_norm_guard_mode={getattr(args, 'feature_norm_guard_mode', 'l2')} "
        f"feature_norm_guard_target={float(getattr(args, 'feature_norm_guard_target', 0.0)):.3f}",
        flush=True,
    )
    print(
        f"[CONFIG-META-SSL] enabled={int(bool(getattr(args, 'use_meta_ssl_cvs', False)))} "
        f"check_only={int(bool(getattr(args, 'meta_ssl_protocol_check_only', False)))} "
        f"split={float(getattr(args, 'ssl_labeled_ratio', 0.1)):.2f}L/"
        f"{float(getattr(args, 'ssl_unlabeled_ratio', 0.7)):.2f}U/"
        f"{float(getattr(args, 'ssl_val_ratio', 0.2)):.2f}Val "
        f"teacher_ema={float(getattr(args, 'ssl_teacher_ema', 0.999)):.4f} "
        f"lambda_ssl_tx={float(getattr(args, 'lambda_ssl_tx', 0.0)):.4f} "
        f"lambda_ssl_proto={float(getattr(args, 'lambda_ssl_proto', 0.0)):.4f} "
        f"lambda_meta_ssl={float(getattr(args, 'lambda_meta_ssl', 0.0)):.4f} "
        f"gate={getattr(args, 'ssl_gate_mode', 'freematch_ups_proto')} "
        f"min_conf={float(getattr(args, 'ssl_min_conf', 0.85)):.3f} "
        f"min_margin={float(getattr(args, 'ssl_min_margin', 0.05)):.3f} "
        f"max_uncertainty={float(getattr(args, 'ssl_max_uncertainty', 0.08)):.3f}",
        flush=True,
    )
    print(
        f"[CONFIG-AUX] enable_pa={int(bool(args.enable_pa_aux))} enable_dac={int(bool(args.enable_dac_aux))} "
        f"pa_main={int(bool(args.aug_enable_pa_normal))} lambda_cls_pa={args.lambda_cls_pa:.4f} "
        f"lambda_cls_dac={args.lambda_cls_dac:.4f} lambda_pa_joint_inv={args.lambda_pa_joint_inv:.4f} "
        f"lambda_pa_kl={args.lambda_pa_kl:.4f} lambda_dac_reg={args.lambda_dac_reg:.4f} "
        f"lambda_pa_reg={args.lambda_pa_reg:.4f} aux_warmup={args.aux_warmup_epochs} aux_ramp={args.aux_ramp_epochs}",
        flush=True,
    )
    print(
        f"[CONFIG-SAT] train_enabled={int(bool(args.use_sat_consistency))} "
        f"train_scenario={args.sat_train_scenario} "
        f"train_cycle={','.join(getattr(args, 'sat_train_scenario_list', [args.sat_train_scenario]))} "
        f"lambda_sat_cls={args.lambda_sat_cls:.4f} lambda_sat_cons={args.lambda_sat_cons:.4f} "
        f"start_epoch={args.sat_cons_start_epoch} eval_enabled={int(bool(args.eval_sat_channel))} "
        f"eval_scenarios={','.join(getattr(args, 'eval_sat_scenario_list', []))} eval_on={args.eval_sat_on} "
        f"eval_max_batches={args.sat_eval_max_batches}",
        flush=True,
    )
    print(
        f"[CONFIG-CONCAT-SAT] enabled={int(bool(getattr(args, 'use_concat_sat_channel_aug', False)))} "
        f"name=拼接星地信道增强 start_epoch={getattr(args, 'concat_sat_start_epoch', 1)} "
        f"view_prob={getattr(args, 'sat_view_prob', 1.0):.3f} seed={getattr(args, 'sat_view_seed', 2027)}",
        flush=True,
    )
    print(
        f"[CONFIG-AUG] use_aug={int(bool(args.use_aug))} scale_min={args.aug_scale_min:.3f} "
        f"scale_max={args.aug_scale_max:.3f} p_dac={args.aug_p_dac:.3f} p_pa={args.aug_p_pa:.3f} "
        f"p_rx_chain={args.aug_p_rx_chain:.3f} rx_envs={args.aug_rx_chain_envs}",
        flush=True,
    )
    print(
        f"[CONFIG-CKPT] latest={args.latest_save_path} best_val={args.best_save_path} "
        f"best_primary={args.best_primary_save_path} best_strict_udu={args.best_unseen_day_unseen_rx_save_path} "
        f"primary_udu_weight={args.primary_udu_weight:.3f}",
        flush=True,
    )
    print("[CONFIG-END]", flush=True)
    print(sep, flush=True)


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
    test_ran: bool = True,
    time_stats: Optional[Dict[str, float]] = None,
):
    sep = "=" * 132
    minor = "-" * 132
    lines = [sep]
    lines.append(f"[EPOCH-BEGIN] E{epoch:03d}/{epochs:03d} | time={epoch_time_s:.1f}s | lr={lr:.2e} | aux_scale={aux_scale:.3f}")
    if time_stats:
        lines.append(
            "[TIME] "
            f"train={float(time_stats.get('train_time_s', float('nan'))):.1f}s "
            f"val={float(time_stats.get('val_time_s', float('nan'))):.1f}s "
            f"test={float(time_stats.get('test_time_s', float('nan'))):.1f}s "
            f"sat_test={float(time_stats.get('sat_test_time_s', float('nan'))):.1f}s "
            f"eval={float(time_stats.get('eval_time_s', float('nan'))):.1f}s"
        )
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
        "[LOSS-CORE-RAW] "
        f"total={meters['loss'].avg:.4f} cls={meters['cls'].avg:.4f} dom={meters['dom'].avg:.4f} "
        f"adv={meters['adv'].avg:.4f} orth={meters['orth'].avg:.4f} cons={meters['cons'].avg:.4f} "
        f"group_ce={meters['group_ce'].avg:.4f}"
    )
    lines.append(
        "[LOSS-CORE-W]   "
        f"cls={meter_avg(meters, 'w_cls'):.4f} dom={meter_avg(meters, 'w_dom'):.4f} "
        f"adv={meter_avg(meters, 'w_adv'):.4f} orth={meter_avg(meters, 'w_orth'):.4f} "
        f"cons={meter_avg(meters, 'w_cons'):.4f} group_ce={meter_avg(meters, 'w_group_ce'):.4f}"
    )
    lines.append(
        "[LOSS-AUX-RAW]  "
        f"cls_pa={meters['cls_pa'].avg:.4f} cls_dac={meters['cls_dac'].avg:.4f} "
        f"pa_joint_inv={meters['pa_joint_inv'].avg:.4f} pa_kl={meters['pa_kl'].avg:.4f} "
        f"dac_reg={meters['dac_reg'].avg:.4f} pa_reg={meters['pa_reg'].avg:.4f} "
        f"gap_dac={meters['gap_dac'].avg:.4f} gap_pa={meters['gap_pa'].avg:.4f} "
        f"cos_joint_pa={meters['cos_joint_pa'].avg:.4f} cos_imp_pa={meters['cos_imp_pa'].avg:.4f}"
    )
    lines.append(
        "[LOSS-AUX-W]    "
        f"cls_pa={meter_avg(meters, 'w_cls_pa'):.4f} cls_dac={meter_avg(meters, 'w_cls_dac'):.4f} "
        f"pa_joint_inv={meter_avg(meters, 'w_pa_joint_inv'):.4f} pa_kl={meter_avg(meters, 'w_pa_kl'):.4f} "
        f"dac_reg={meter_avg(meters, 'w_dac_reg'):.4f} pa_reg={meter_avg(meters, 'w_pa_reg'):.4f}"
    )
    lines.append(
        "[LOSS-SAT-RAW]  "
        f"cls_sat={meters['sat_cls'].avg:.4f} sat_cons={meters['sat_cons'].avg:.4f} "
        f"sat_cos={meters['sat_cos'].avg:.4f}"
    )
    lines.append(
        "[LOSS-SAT-W]    "
        f"cls_sat={meter_avg(meters, 'w_sat_cls'):.4f} sat_cons={meter_avg(meters, 'w_sat_cons'):.4f}"
    )
    lines.append(
        "[LOSS-DG-RAW]   "
        f"proto={meters['proto'].avg:.4f} proto_cos={meters['proto_pull_cos'].avg:.4f} "
        f"supcon={meters['supcon'].avg:.4f} fishr={meters['fishr'].avg:.4f} "
        f"feature_norm={meter_avg(meters, 'feature_norm'):.4f} z_id_norm={meter_avg(meters, 'zid_norm'):.4f}"
    )
    lines.append(
        "[LOSS-DG-W]     "
        f"proto={meter_avg(meters, 'w_proto'):.4f} supcon={meter_avg(meters, 'w_supcon'):.4f} "
        f"fishr={meter_avg(meters, 'w_fishr'):.6e} feature_norm={meter_avg(meters, 'w_feature_norm'):.6e}"
    )
    lines.append(
        "[LOSS-META-SSL] "
        f"tx={meter_avg(meters, 'meta_ssl_tx'):.4f} proto={meter_avg(meters, 'meta_ssl_proto'):.4f} "
        f"dom={meter_avg(meters, 'meta_ssl_dom'):.4f} adv={meter_avg(meters, 'meta_ssl_adv'):.4f} "
        f"coverage={meter_avg(meters, 'meta_ssl_coverage'):.4f} accepted={meter_avg(meters, 'meta_ssl_accept'):.2f} "
        f"proto_agree={meter_avg(meters, 'meta_ssl_proto_agree'):.4f} "
        f"teacher_conf={meter_avg(meters, 'meta_ssl_teacher_conf'):.4f} "
        f"active_proto={meter_avg(meters, 'meta_ssl_proto_active'):.1f}"
    )
    lines.append(
        "[LOSS-META-SSL-W] "
        f"tx={meter_avg(meters, 'w_meta_ssl_tx'):.4f} proto={meter_avg(meters, 'w_meta_ssl_proto'):.4f} "
        f"dom={meter_avg(meters, 'w_meta_ssl_dom'):.4f} adv={meter_avg(meters, 'w_meta_ssl_adv'):.4f}"
    )
    if stage_state is not None:
        lines.append(
            "[LOSS-WEIGHT] "
            f"dom={stage_state.get('dom_scale', float('nan')):.3f} adv={stage_state.get('adv_scale', float('nan')):.3f} "
            f"orth={stage_state.get('orth_scale', float('nan')):.3f} cons={stage_state.get('cons_scale', float('nan')):.3f} "
            f"group_ce={stage_state.get('group_ce_scale', float('nan')):.3f} aux_scale={aux_scale:.3f}"
        )
    lines.append(format_weighted_loss_top(meters, [
        "w_cls", "w_dom", "w_adv", "w_orth", "w_cons", "w_group_ce",
        "w_cls_pa", "w_cls_dac", "w_pa_joint_inv", "w_pa_kl", "w_dac_reg", "w_pa_reg",
        "w_sat_cls", "w_sat_cons", "w_proto", "w_supcon", "w_fishr", "w_feature_norm",
        "w_meta_ssl_tx", "w_meta_ssl_proto", "w_meta_ssl_dom", "w_meta_ssl_adv",
    ], limit=10))
    lines.append(minor)
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
    if test_ran:
        lines.append(f"[TEST]  overall_tx={test_stats['tx_acc']:.2f}% ({test_stats['tx_correct']}/{test_stats['tx_total']})")
        lines.append("[TEST-SPLIT]")
        lines.extend(format_named_test_lines(named_test_stats, named_test_meta))
    else:
        lines.append("[TEST]  skipped by training-time test gate")
    lines.append(f"[BEST-JOINT]  val_tx={best_joint_val_tx:.2f}% & test_tx={best_joint_test_tx:.2f}% @ E{best_epoch:03d}")
    latest_note = "saved" if latest_saved else f"protected: {collapse_guard.get('reason', 'unknown') if collapse_guard else 'unknown'}"
    lines.append(f"[CKPT]  latest -> {latest_path} ({latest_note}) | best -> {best_path}{' (updated: val improved)' if is_best else ''}")
    lines.append(f"[EPOCH-END] E{epoch:03d}/{epochs:03d}")
    lines.append(sep)
    return "\n".join(lines)
