#!/usr/bin/env python3
"""Generate the second pure few-shot CVS/CEN51 search matrix.

Contract:
* shots < 100 use the pure per-class few-shot path and shot-aware training.
* shots >= 100 restore the original CEN51 R04 ratio path, not the few-shot path.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_RUN_ID = "cen51_pure_fewshot_step_sampler_20260610_140100"
SAT_SCENARIOS = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
FEWSHOT_BOUNDARY = 100


def fmt_value(value: object) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0.0"
        if abs(value) < 0.01:
            return f"{value:.4g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def arg_pairs(params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            args.append(flag if value else f"--no_{key}")
        elif value is None:
            continue
        else:
            args.extend([flag, fmt_value(value)])
    return args


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def merge_params(*parts: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for part in parts:
        out.update(part)
    return out


def batch_for_shot(shots: int) -> int:
    if shots <= 10:
        return 16
    if shots <= 30:
        return 32
    return 64


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    run_name: str
    gpu: int
    regime: str
    shots_per_class: int | None
    sampler: str
    strategy: str
    rationale: str
    success_gate: str
    params: Dict[str, object] = field(default_factory=dict)

    @property
    def is_pure_fewshot(self) -> bool:
        return self.shots_per_class is not None and self.shots_per_class < FEWSHOT_BOUNDARY

    def args(self) -> List[str]:
        return arg_pairs(self.params)


LOW_BASE: Dict[str, object] = {
    "train_mode": "centralized",
    "eval_batch_size": 256,
    "dataset": "wisig",
    "wisig_protocol": "cvs_day_rx",
    "wisig_domain": "rx_day",
    "wisig_train_ratio": 0.5,
    "wisig_split_strategy": "random",
    "wisig_cap_strategy": "random",
    "wisig_max_train_per_combo": 0,
    "test_eval_policy": "interval_final",
    "test_eval_start_epoch": 1,
    "test_eval_interval": 10,
    "eval_sat_channel": True,
    "eval_sat_on": "test_unseen_day_unseen_rx",
    "eval_sat_scenarios": SAT_SCENARIOS,
    "sat_eval_max_batches": -1,
    "arch_family": "cvsincnet",
    "slim_group": "none",
    "branch_ablation": "no_dac",
    "domain_branch_ablation": "no_stats",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    "exp_group": "s3_rxrobust_no_dac",
    "model_variant": "lite_d",
    "seed": 2028,
    "collapse_guard": True,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 12.0,
    "collapse_guard_max_skipped_delta": 2,
    "use_ema_ckpt": True,
    "ema_decay": 0.999,
    "use_swad_ckpt": True,
    "swad_interval": 1,
    "primary_udu_weight": 0.84,
    "train_drop_last": False,
}


ORIGINAL_CEN51_BASE: Dict[str, object] = {
    "train_mode": "centralized",
    "batch_size": 256,
    "eval_batch_size": 256,
    "dataset": "wisig",
    "wisig_domain": "rx_day",
    "wisig_train_ratio": 0.1,
    "wisig_split_strategy": "random",
    "wisig_cap_strategy": "random",
    "epochs": 200,
    "test_eval_policy": "interval_final",
    "test_eval_start_epoch": 1,
    "test_eval_interval": 10,
    "eval_sat_channel": True,
    "eval_sat_on": "test_unseen_day_unseen_rx",
    "eval_sat_scenarios": SAT_SCENARIOS,
    "sat_eval_max_batches": -1,
    "arch_family": "cvsincnet",
    "slim_group": "none",
    "branch_ablation": "no_dac",
    "domain_branch_ablation": "no_stats",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    "exp_group": "s3_rxrobust_no_dac",
    "model_variant": "lite_d",
    "use_aug": True,
    "use_concat_sat_channel_aug": True,
    "concat_sat_start_epoch": 1,
    "lambda_sat_cls": 0.0,
    "lambda_sat_cons": 0.0,
    "seed": 1337,
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
    "sat_train_scenarios": SAT_SCENARIOS,
    "sat_view_prob": 1.0,
    "domain_freq_stability_mode": "dsq",
    "freq_stability_channels": 2,
    "lambda_group_ce": 0.06,
    "group_ce_mode": "smooth_dro_capped",
    "group_ce_top_frac": 0.35,
    "groupdro_tau": 0.50,
    "groupdro_cap": 0.65,
    "use_proto_memory": True,
    "lambda_proto": 0.015,
    "proto_momentum": 0.95,
    "lambda_supcon_id": 0.02,
    "supcon_temp": 0.12,
    "lambda_fishr": 0.005,
    "fishr_min_domains": 4,
    "generalization_feature": "z_id",
    "collapse_guard": True,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 12.0,
    "collapse_guard_max_skipped_delta": 2,
    "use_ema_ckpt": True,
    "ema_decay": 0.999,
    "use_swad_ckpt": True,
    "swad_start_epoch": 90,
    "swad_tolerance": 0.8,
}


ORIGINAL_R04: Dict[str, object] = {
    "primary_udu_weight": 0.84,
    "concat_sat_ce_weight": 1.19,
    "pa_orders": "1,3,5",
    "lambda_group_ce": 0.088,
    "group_ce_min_domains": 4,
    "group_ce_top_frac": 0.20,
    "groupdro_tau": 0.37,
    "groupdro_cap": 0.48,
    "lambda_proto": 0.016,
    "proto_momentum": 0.970,
    "lambda_supcon_id": 0.022,
    "lambda_fishr": 0.002,
    "fishr_min_domains": 4,
    "use_sat_consistency": True,
    "lambda_sat_cons": 0.006,
    "sat_cons_start_epoch": 118,
    "swad_start_epoch": 70,
    "swad_tolerance": 0.34,
    "sat_view_schedule": "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit",
}


def low_rxgrl(
    *,
    shots: int,
    steps: int,
    adv: float,
    norm: float,
    sat_weight: float,
    lr: float = 3e-4,
    wd: float = 5e-5,
) -> Dict[str, object]:
    params: Dict[str, object] = {
        "batch_size": batch_for_shot(shots),
        "epochs": 170 if steps <= 64 else 140,
        "train_steps_per_epoch": steps,
        "lr": lr,
        "lr_min": 1e-6,
        "wd": wd,
        "wisig_train_shots_per_class": shots,
        "wisig_train_shot_strategy": "rx_day_balanced",
        "swad_start_epoch": 55 if shots <= 5 else 65,
        "swad_tolerance": 0.85,
        "use_aug": False,
        "use_mixstyle": False,
        "enable_pa_aux": False,
        "enable_dac_aux": False,
        "aug_enable_pa_normal": False,
        "aug_p_pa": 0.0,
        "aug_p_dac": 0.0,
        "lambda_cls_pa": 0.0,
        "lambda_pa_joint_inv": 0.0,
        "lambda_pa_kl": 0.0,
        "lambda_pa_reg": 0.0,
        "lambda_dom": 0.32 if shots <= 5 else 0.36,
        "lambda_adv": adv,
        "grl_lambda": 1.0,
        "lambda_orth": 0.01,
        "lambda_cons": 0.0,
        "lambda_group_ce": 0.0,
        "lambda_proto": 0.0,
        "lambda_supcon_id": 0.0,
        "lambda_fishr": 0.0,
        "lambda_feature_norm_guard": norm,
        "feature_norm_guard_mode": "hinge",
        "feature_norm_guard_target": 8.0,
        "min_batch_domains_for_domain_loss": 2,
        "min_batch_domain_frac": 0.10,
        "domain_freq_stability_mode": "off",
        "pa_orders": "1,3,5",
        "use_sat_consistency": False,
        "lambda_sat_cons": 0.0,
        "lambda_sat_cls": 0.0,
        "sat_cons_start_epoch": 999,
        "use_concat_sat_channel_aug": False,
        "concat_sat_ce_weight": 0.0,
        "sat_view_prob": 0.0,
    }
    if sat_weight > 0.0:
        params.update(
            {
                "use_concat_sat_channel_aug": True,
                "concat_sat_ce_weight": sat_weight,
                "sat_view_prob": sat_weight,
                "sat_train_scenarios": "clear_leo,mixed_orbit",
                "sat_view_schedule": f"1@{sat_weight:.2f}:clear_leo,mixed_orbit",
            }
        )
    return params


def strong_riei_constraint(
    *,
    shots: int,
    steps: int,
    adv: float,
    dom: float,
    norm: float,
    norm_target: float,
    sat_weight: float,
) -> Dict[str, object]:
    params = low_rxgrl(
        shots=shots,
        steps=steps,
        adv=adv,
        norm=norm,
        sat_weight=sat_weight,
        lr=2.5e-4,
        wd=8e-5,
    )
    params.update(
        {
            "lambda_dom": dom,
            "lambda_adv": adv,
            "lambda_orth": 0.04,
            "lambda_cons": 0.0,
            "feature_norm_guard_mode": "hinge",
            "feature_norm_guard_target": norm_target,
            "stage1_epochs": 20 if shots <= 5 else 18,
            "stage2_epochs": 55 if shots <= 5 else 50,
            "stage3_ramp_epochs": 20,
            "late_stable_start": 115 if shots <= 5 else 110,
            "late_stable_ramp_epochs": 25,
            "late_adv_min_scale": 0.85,
            "late_cons_min_scale": 0.0,
            "late_group_ce_min_scale": 0.0,
        }
    )
    return params


def shotaware(
    *,
    shots: int,
    steps: int,
    adv: float,
    norm: float,
    group_ce: float,
    proto: float,
    supcon: float,
    fishr: float,
    mix_p: float,
    mix_strength: float,
    sat_weight: float,
    lr: float = 2.5e-4,
    wd: float = 7e-5,
) -> Dict[str, object]:
    params: Dict[str, object] = {
        "batch_size": batch_for_shot(shots),
        "epochs": 180 if shots <= 30 else 190,
        "train_steps_per_epoch": steps,
        "lr": lr,
        "lr_min": 1e-6,
        "wd": wd,
        "wisig_train_shots_per_class": shots,
        "wisig_train_shot_strategy": "rx_day_balanced",
        "swad_start_epoch": 75 if shots <= 20 else (85 if shots <= 30 else 95),
        "swad_tolerance": 0.70,
        "use_aug": True,
        "aug_scale_min": 0.03,
        "aug_scale_max": 0.18 if shots <= 20 else (0.22 if shots <= 30 else 0.26),
        "late_aug_min_scale": 0.14 if shots <= 20 else (0.16 if shots <= 30 else 0.18),
        "use_mixstyle": True,
        "mixstyle_layers": "time_down,t1",
        "mixstyle_mix": "same_tx_crossdomain",
        "mixstyle_fallback": "skip",
        "mixstyle_p": mix_p,
        "mixstyle_strength": mix_strength,
        "mixstyle_late_start": 120 if shots <= 20 else (130 if shots <= 30 else 140),
        "mixstyle_late_ramp_epochs": 40,
        "mixstyle_late_min_p": max(0.02, mix_p * 0.45),
        "mixstyle_late_min_strength": max(0.12, mix_strength * 0.55),
        "enable_pa_aux": False,
        "enable_dac_aux": False,
        "aug_enable_pa_normal": False,
        "lambda_cls_pa": 0.0,
        "lambda_pa_joint_inv": 0.0,
        "lambda_pa_kl": 0.0,
        "lambda_pa_reg": 0.0,
        "lambda_dom": 0.42 if shots <= 20 else (0.48 if shots <= 30 else 0.54),
        "lambda_adv": adv,
        "grl_lambda": 1.0,
        "lambda_orth": 0.015,
        "lambda_cons": 0.008 if shots <= 20 else (0.012 if shots <= 30 else 0.016),
        "lambda_group_ce": group_ce,
        "group_ce_mode": "smooth_dro_capped",
        "group_ce_min_domains": 2,
        "group_ce_top_frac": 0.18 if shots <= 20 else (0.22 if shots <= 30 else 0.25),
        "groupdro_tau": 0.35 if shots <= 20 else (0.42 if shots <= 30 else 0.50),
        "groupdro_cap": 0.45 if shots <= 20 else (0.55 if shots <= 30 else 0.62),
        "use_proto_memory": proto > 0.0,
        "lambda_proto": proto,
        "proto_momentum": 0.95,
        "lambda_supcon_id": supcon,
        "supcon_temp": 0.12,
        "lambda_fishr": fishr,
        "fishr_min_domains": 2,
        "lambda_feature_norm_guard": norm,
        "feature_norm_guard_mode": "l2",
        "feature_norm_guard_target": 0.0,
        "domain_freq_stability_mode": "off",
        "pa_orders": "1,3,5",
        "use_sat_consistency": False,
        "lambda_sat_cons": 0.0,
        "lambda_sat_cls": 0.0,
        "sat_cons_start_epoch": 999,
        "use_concat_sat_channel_aug": False,
        "concat_sat_ce_weight": 0.0,
        "sat_view_prob": 0.0,
    }
    if sat_weight > 0.0:
        params.update(
            {
                "use_concat_sat_channel_aug": True,
                "concat_sat_ce_weight": sat_weight,
                "sat_view_prob": sat_weight,
                "sat_train_scenarios": "clear_leo,mixed_orbit",
                "sat_view_schedule": f"1@{sat_weight:.2f}:clear_leo,mixed_orbit;145@{max(0.05, sat_weight * 0.60):.2f}:clear_leo,mixed_orbit",
            }
        )
    return params


def make_low_candidate(
    idx: int,
    label: str,
    shots: int,
    suffix: str,
    sampler: str,
    params: Dict[str, object],
    rationale: str,
    success_gate: str,
) -> Candidate:
    merged = merge_params(LOW_BASE, params)
    merged["wisig_train_shot_strategy"] = sampler
    return Candidate(
        candidate_id=f"PFS{idx:02d}_{label}_{suffix}",
        run_name=f"CEN51_PFS2_{label}_{suffix}_seed2028",
        gpu=(idx - 1) % 8,
        regime="pure_fewshot_lt100",
        shots_per_class=shots,
        sampler=sampler,
        strategy=suffix,
        rationale=rationale,
        success_gate=success_gate,
        params=merged,
    )


def make_original_candidate(idx: int) -> Candidate:
    params = merge_params(ORIGINAL_CEN51_BASE, ORIGINAL_R04)
    return Candidate(
        candidate_id=f"PFS{idx:02d}_ORIG100_CEN51_R04_R010",
        run_name="CEN51_ORIG100_R04_sat_joint_guard_no_overdrive_r010",
        gpu=(idx - 1) % 8,
        regime="restore_original_cen51_ge100",
        shots_per_class=None,
        sampler="original_ratio_random",
        strategy="original_cen51_r04_ratio_0p1",
        rationale="Boundary control: at >=100 shots restore original CEN51_R04 ratio path with no pure-shot cap, no rx_day_balanced sampler, and no repeated few-shot steps.",
        success_gate="Startup must show ratio-based original split, train_steps_per_epoch=0/default, and no wisig_train_shots_per_class cap.",
        params=params,
    )


def make_candidates() -> List[Candidate]:
    rows: List[tuple[str, int, str, str, Dict[str, object], str, str]] = [
        ("FS005", 5, "FIT64_WEAKSAT", "rx_day_balanced", low_rxgrl(shots=5, steps=64, adv=0.06, norm=0.00045, sat_weight=0.06), "K5 fit-budget test with weaker GRL/norm so identity CE can catch up.", "Train acc should rise clearly above the previous 48 percent while strict UDU stays above R04."),
        ("FS005", 5, "FIT96_WEAKSAT", "rx_day_balanced", low_rxgrl(shots=5, steps=96, adv=0.06, norm=0.00045, sat_weight=0.06), "K5 same recipe with larger update budget to test underfitting versus overfit.", "Best val improves without strict UDU collapse versus FIT64."),
        ("FS005", 5, "RIEI_STRONG64", "rx_day_balanced", strong_riei_constraint(shots=5, steps=64, adv=0.22, dom=0.55, norm=0.0015, norm_target=6.5, sat_weight=0.04), "K5 RIEI-style strong targeted constraint: high RX/day GRL, stronger orthogonality, tighter feature-norm hinge, minimal satellite.", "If CVS was under-constrained, strict UDU and receiver floor should beat the weaker FIT branch without train-acc collapse."),
        ("FS005", 5, "FIT64_RANDOM", "random", low_rxgrl(shots=5, steps=64, adv=0.06, norm=0.00045, sat_weight=0.06), "Sampler ablation for K5: random pure shots versus receiver/day balanced shots.", "rx_day_balanced counterpart should win or tie with lower receiver-floor variance."),
        ("FS010", 10, "FIT48_WEAKSAT", "rx_day_balanced", low_rxgrl(shots=10, steps=48, adv=0.08, norm=0.00035, sat_weight=0.08), "K10 lower-pressure RIEI-like bottleneck plus enough optimizer steps.", "Target best val >=90 while strict UDU stays above previous 72.52."),
        ("FS010", 10, "FIT80_WEAKSAT", "rx_day_balanced", low_rxgrl(shots=10, steps=80, adv=0.08, norm=0.00035, sat_weight=0.08), "K10 update-budget sweep to separate underfit from overfit.", "Should improve train/val without strict UDU rollback versus FIT48."),
        ("FS010", 10, "RIEI_STRONG64", "rx_day_balanced", strong_riei_constraint(shots=10, steps=64, adv=0.28, dom=0.60, norm=0.0010, norm_target=7.0, sat_weight=0.06), "K10 RIEI-style strong targeted constraint with repeated steps and minimal satellite.", "Target val near 90 while strict UDU and receiver floor improve over weaker FIT branches."),
        ("FS010", 10, "FIT48_RANDOM", "random", low_rxgrl(shots=10, steps=48, adv=0.08, norm=0.00035, sat_weight=0.08), "Sampler ablation for K10.", "rx_day_balanced counterpart should reduce seed sensitivity."),
        ("FS020", 20, "CLEAN64_RXDAY", "rx_day_balanced", shotaware(shots=20, steps=64, adv=0.16, norm=0.00008, group_ce=0.008, proto=0.001, supcon=0.001, fishr=0.0, mix_p=0.03, mix_strength=0.16, sat_weight=0.0), "K20 clean-priority relaxed DG with repeated steps.", "Best val/strict UDU should exceed previous K20 clean 77.73/70.67."),
        ("FS020", 20, "SAT64_RXDAY", "rx_day_balanced", shotaware(shots=20, steps=64, adv=0.15, norm=0.00006, group_ce=0.008, proto=0.001, supcon=0.001, fishr=0.0, mix_p=0.03, mix_strength=0.16, sat_weight=0.10), "K20 weak full-DG satellite re-entry.", "Satellite metrics improve without lowering clean strict below CLEAN64."),
        ("FS020", 20, "SAT96_RXDAY", "rx_day_balanced", shotaware(shots=20, steps=96, adv=0.15, norm=0.00006, group_ce=0.008, proto=0.001, supcon=0.001, fishr=0.0, mix_p=0.03, mix_strength=0.16, sat_weight=0.10), "K20 larger update-budget check.", "Should improve train fit without satellite-induced clean rollback."),
        ("FS030", 30, "CLEAN56_RXDAY", "rx_day_balanced", shotaware(shots=30, steps=56, adv=0.20, norm=0.00006, group_ce=0.014, proto=0.002, supcon=0.002, fishr=0.0002, mix_p=0.05, mix_strength=0.22, sat_weight=0.0), "K30 clean-priority relaxation.", "Should beat previous K30 clean 80.72/73.73."),
        ("FS030", 30, "SAT56_RXDAY", "rx_day_balanced", shotaware(shots=30, steps=56, adv=0.19, norm=0.00005, group_ce=0.014, proto=0.002, supcon=0.002, fishr=0.0002, mix_p=0.05, mix_strength=0.22, sat_weight=0.14), "K30 satellite gate with full DG path.", "Should beat previous K30 satgate 83.81/74.29."),
        ("FS050", 50, "SAT48_RXDAY", "rx_day_balanced", shotaware(shots=50, steps=48, adv=0.23, norm=0.00004, group_ce=0.022, proto=0.003, supcon=0.003, fishr=0.0004, mix_p=0.08, mix_strength=0.28, sat_weight=0.18), "K50 near-boundary few-shot schedule.", "Should recover part of the original CEN51 advantage while staying under pure-shot protocol."),
        ("FS080", 80, "BRIDGE32_RXDAY", "rx_day_balanced", shotaware(shots=80, steps=32, adv=0.25, norm=0.00003, group_ce=0.030, proto=0.004, supcon=0.004, fishr=0.0006, mix_p=0.10, mix_strength=0.32, sat_weight=0.22), "K80 bridge test: still <100 so use few-shot path, but close to original CEN51 regularization.", "Performance should approach ORIG100 without showing a drop from K50."),
    ]
    candidates = [
        make_low_candidate(i + 1, label, shots, suffix, sampler, params, rationale, gate)
        for i, (label, shots, suffix, sampler, params, rationale, gate) in enumerate(rows)
    ]
    candidates.append(make_original_candidate(len(candidates) + 1))
    return candidates


def render_launcher(run_id: str, candidates: Sequence[Candidate]) -> str:
    lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"',
        'PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"',
        'TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"',
        f'RUN_ID="${{RUN_ID:-{run_id}}}"',
        'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
        'RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"',
        'MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"',
        'DRY_RUN="${DRY_RUN:-0}"',
        'ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"',
        "",
        'for arg in "$@"; do',
        '  case "${arg}" in',
        "    --dry-run) DRY_RUN=1 ;;",
        '    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;',
        '    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;',
        "  esac",
        "done",
        "",
        "gpu_process_count() {",
        '  local gpu="$1"',
        '  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \\',
        "    | sed '/^$/d' | wc -l | tr -d ' '",
        "}",
        "",
        "print_cmd() { printf '%q ' \"$@\"; printf '\\n'; }",
        "",
        "should_skip() {",
        '  local candidate_id="$1" run_name="$2"',
        '  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]',
        "}",
        "",
        "declare -A LAUNCHED_BY_GPU=()",
        "",
        "run_candidate() {",
        '  local candidate_id="$1" run_name="$2" regime="$3" gpu="$4"',
        "  shift 4",
        '  local run_dir="${RUNS_ROOT}/${run_name}"',
        '  local log_path="${LOG_ROOT}/${run_name}.out"',
        '  local cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "$@"',
        '    --run_name "${run_name}"',
        '    --latest_save_path "${run_dir}/latest_model.pth"',
        '    --best_save_path "${run_dir}/best_val_model.pth"',
        '    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"',
        '    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"',
        '    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"',
        '    --ema_save_path "${run_dir}/ema_model.pth"',
        '    --swa_save_path "${run_dir}/swa_model.pth"',
        '    --swad_save_path "${run_dir}/swad_model.pth")',
        "",
        '  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi',
        '  echo "[CEN51-PFS2] candidate=${candidate_id} run=${run_name} regime=${regime} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        "  printf '[CEN51-PFS2-CMD]'",
        '  print_cmd "${cmd[@]}"',
        '  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi',
        '  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then',
        '    mkdir -p "${LOG_ROOT}"',
        '    printf "%s\\t%s\\t%s\\t%s\\t%s\\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"',
        "    return 0",
        "  fi",
        '  local current_count local_count',
        '  current_count="$(gpu_process_count "${gpu}")"',
        '  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"',
        '  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then',
        '    mkdir -p "${LOG_ROOT}"',
        '    printf "%s\\t%s\\t%s\\tgpu=%s active_count=%s local_count=%s max=%s\\n" "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"',
        "    return 0",
        "  fi",
        '  mkdir -p "${LOG_ROOT}" "${run_dir}"',
        '  nohup "${cmd[@]}" > "${log_path}" 2>&1 &',
        '  local pid="$!"',
        '  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))',
        '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
        "}",
        "",
        'if [[ "${DRY_RUN}" != "1" ]]; then',
        '  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }',
        "fi",
        'cd "${ROOT}"',
        'echo "[CEN51-PFS2] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        "",
    ]
    for cand in candidates:
        lines.append(
            "run_candidate "
            + shell_join([cand.candidate_id, cand.run_name, cand.regime, str(cand.gpu)])
            + " \\"
        )
        args = cand.args()
        for idx, arg in enumerate(args):
            cont = " \\" if idx < len(args) - 1 else ""
            lines.append(f"  {shlex.quote(arg)}{cont}")
        lines.append("")
    lines.append('echo "[CEN51-PFS2] launch submissions complete"')
    return "\n".join(lines) + "\n"


def candidate_rows(candidates: Sequence[Candidate]) -> List[Dict[str, object]]:
    return [
        {
            "candidate_id": c.candidate_id,
            "run_name": c.run_name,
            "gpu": c.gpu,
            "regime": c.regime,
            "shots_per_class": "" if c.shots_per_class is None else c.shots_per_class,
            "pure_fewshot": int(c.is_pure_fewshot),
            "sampler": c.sampler,
            "strategy": c.strategy,
            "batch_size": c.params.get("batch_size", ""),
            "train_steps_per_epoch": c.params.get("train_steps_per_epoch", 0),
            "wisig_train_ratio": c.params.get("wisig_train_ratio", ""),
            "lr": c.params.get("lr", ""),
            "wd": c.params.get("wd", ""),
            "rationale": c.rationale,
            "success_gate": c.success_gate,
        }
        for c in candidates
    ]


def write_outputs(run_id: str, out_dir: Path) -> None:
    candidates = make_candidates()
    launcher = Path("code/scripts") / f"launch_{run_id}.sh"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")

    report_dir = Path("automation_reports/CV-SincNet") / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = candidate_rows(candidates)
    for path in [out_dir / "matrix.csv", report_dir / "matrix.csv"]:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Improve pure <100-shot CVS/CEN51 performance while restoring original CEN51 at >=100 shots.",
        "boundary_contract": {
            "fewshot_active_when": "shots_per_class < 100",
            "fewshot_args": ["--wisig_train_shots_per_class", "--wisig_train_shot_strategy", "--train_steps_per_epoch", "--no_train_drop_last"],
            "restore_original_when": "shots_per_class >= 100",
            "restore_control": "PFS16_ORIG100_CEN51_R04_R010",
            "restore_args": "ratio-based original CEN51_R04, no pure-shot cap, no repeated steps, default train_drop_last",
        },
        "hypotheses": [
            "The completed PFS batch was update-budget limited: low-shot train accuracy was low even when strict UDU improved.",
            "rx_day_balanced sample selection should reduce receiver/day shortcut variance compared with random selection.",
            "Weak full-DG satellite views help only after identity fit is sufficient; therefore K5/K10 use very low satellite weights.",
            "At K80, shot-aware regularization should approach the ORIG100 CEN51 behavior without crossing the boundary.",
        ],
        "metrics_to_watch": [
            "split_info.train_size == 6 * K for PFS candidates",
            "PFS16 does not show max_samples_per_class_train or train_steps_per_epoch override",
            "train_tx, best_val_tx, latest_val_tx",
            "test_unseen_day_unseen_rx strict UDU",
            "worst receiver and satellite scenario floors",
            "LOSS-TOP balance, domain accuracy, feature norm guard",
        ],
        "candidates": rows,
        "local_files": {
            "launcher": str(launcher),
            "report": str(report_dir / "report.md"),
            "matrix_csv": str(report_dir / "matrix.csv"),
        },
    }
    for path in [out_dir / "manifest.json", report_dir / "manifest.json"]:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# CVS/CEN51 Pure Few-Shot Step/Sampler Search - {run_id}",
        "",
        "## Objective",
        "Optimize strict pure few-shot CVS/CEN51 only below 100 shots, then restore the original CEN51_R04 setting at the 100-shot boundary.",
        "",
        "## Boundary Contract",
        "- `<100 shots`: use `--wisig_train_shots_per_class K`, `rx_day_balanced` or random sampler ablation, `--no_train_drop_last`, and explicit `--train_steps_per_epoch`.",
        "- `>=100 shots`: do not use pure-shot cap, do not use `rx_day_balanced`, do not use repeated few-shot steps; run the original CEN51_R04 ratio=0.1 recipe.",
        "- `PFS16_ORIG100_CEN51_R04_R010` is a restore control, not a low-shot optimization candidate.",
        "",
        "## Why This Batch Exists",
        "The previous pure few-shot batch showed the best K5/K10 candidates were still underfit: train accuracy stayed low while strict UDU improved. This batch tests whether repeated optimizer steps plus better receiver/day shot selection can raise validation accuracy without losing the shortcut-suppression benefit.",
        "",
        "## Candidates",
        "| candidate | regime | shots/class | gpu | sampler | steps/epoch | lr | rationale |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        shots = row["shots_per_class"] if row["shots_per_class"] != "" else "ratio0.1"
        lines.append(
            f"| `{row['candidate_id']}` | {row['regime']} | {shots} | {row['gpu']} | {row['sampler']} | {row['train_steps_per_epoch']} | {row['lr']} | {row['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Success Criteria",
            "- K10 target: best validation reaches or approaches 90% and strict UDU improves over the previous PFS K10 weak-sat result.",
            "- K5 target: train accuracy rises materially above the previous underfit run while strict UDU remains above the R04 controls.",
            "- K20/K30/K50/K80 target: more shots should not reduce strict UDU, and validation should trend upward.",
            "- ORIG100 target: startup confirms original CEN51_R04 ratio path and serves as the return-to-default boundary.",
            "",
            "## Local Verification",
            "- `conda activate ssr-gpu; $env:PYTHONPATH='E:\\type10-7\\code'; python -m pytest tests\\test_train_steps_per_epoch.py tests\\test_wisig_random_split.py -q`",
            "- `conda activate ssr-gpu; $env:PYTHONPATH='E:\\type10-7\\code'; python -m py_compile code\\train.py code\\dataset_wisig.py baselines\\common\\cvs_data.py tools\\cen51_pure_fewshot_step_sampler_matrix.py`",
            "- `conda activate ssr-gpu; python tools\\cen51_pure_fewshot_step_sampler_matrix.py --run-id " + run_id + "`",
            "",
            "## Launch Plan",
            f"- Local launcher: `{launcher}`",
            "- Remote command after sync: `bash code/scripts/launch_" + run_id + ".sh`",
            "- Default capacity: at most 2 training processes per GPU.",
            "",
            "## Post-Run Checklist",
            "- First check startup split lines for all PFS candidates and the ORIG100 restore control.",
            "- Rank by same-shot best validation, strict UDU, worst receiver, and satellite floors.",
            "- Promote only configurations that improve same-shot validation without strict UDU regression.",
            "- If K5/K10 still underfit, the next axis is CE/domain loss schedule rather than simply extending epochs.",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    run_id = str(args.run_id)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else Path("analysis_tmp") / run_id
    write_outputs(run_id, out_dir)
    print(f"[CEN51-PFS2] run_id={run_id}")
    print(f"[CEN51-PFS2] launcher=code/scripts/launch_{run_id}.sh")
    print(f"[CEN51-PFS2] report=automation_reports/CV-SincNet/{run_id}/report.md")
    print(f"[CEN51-PFS2] candidates={len(make_candidates())}")


if __name__ == "__main__":
    main()
