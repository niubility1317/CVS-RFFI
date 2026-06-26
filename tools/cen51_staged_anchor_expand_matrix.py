#!/usr/bin/env python
"""Generate staged CEN51 anchor-expansion experiments.

Each staged candidate runs two commands on the same GPU:

1. Stage-1: few-shot clean/invariant anchor.
2. Stage-2: 0.1-ratio expansion initialized from the Stage-1 checkpoint.

Direct controls skip Stage-1 and train the Stage-2 configuration from scratch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
DEFAULT_RUN_ID = "cen51_staged_anchor_expand_20260609_201500"
ALL_SAT = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LIGHT_SAT = "clear_leo,mixed_orbit"


@dataclass(frozen=True)
class Candidate:
    cid: str
    run_name: str
    gpu: int
    seed: int
    anchor_shots: int
    stage2_shots: int
    cap_strategy: str
    direct: bool
    stage1_epochs: int
    stage2_epochs: int
    stage2_kind: str
    hypothesis: str


def gpu_plan() -> list[int]:
    return [0, 1, 2, 3, 4, 5, 6, 7] * 3 + [0, 1, 2, 3, 4, 5, 6]


def make_candidates() -> list[Candidate]:
    rows = [
        ("S5_RND_CLEAN_2028", 2028, 5, "random", False, "clean_expand", "5-shot random clean anchor, then clean 0.1 expansion."),
        ("S5_RND_LATESAT_2028", 2028, 5, "random", False, "late_sat", "5-shot random anchor, then delayed full-DG satellite expansion."),
        ("S5_RND_RXGUARD_2028", 2028, 5, "random", False, "rxguard", "5-shot random anchor, then stronger receiver/day shortcut guard."),
        ("S5_RND_STRONGSAT_NEG_2028", 2028, 5, "random", False, "strong_sat_neg", "Negative control: 5-shot anchor followed by early strong satellite pressure."),
        ("S5_FRONT_CLEAN_NEG_2028", 2028, 5, "front", False, "clean_expand", "Sample-selection negative control: front cap anchor, clean expansion."),
        ("S5_RND_CLEAN_1337", 1337, 5, "random", False, "clean_expand", "Seed replicate for 5-shot clean staged anchor."),
        ("S5_RND_LATESAT_1337", 1337, 5, "random", False, "late_sat", "Seed replicate for 5-shot delayed satellite staged anchor."),
        ("S5_RND_RXGUARD_1337", 1337, 5, "random", False, "rxguard", "Seed replicate for 5-shot rxguard staged anchor."),
        ("S5_RND_CLEAN_2030", 2030, 5, "random", False, "clean_expand", "Second split replicate for 5-shot clean staged anchor."),
        ("S5_RND_LATESAT_2030", 2030, 5, "random", False, "late_sat", "Second split replicate for 5-shot delayed satellite staged anchor."),
        ("S10_RND_CLEAN_2028", 2028, 10, "random", False, "clean_expand", "10-shot random clean anchor, then clean 0.1 expansion."),
        ("S10_RND_LATESAT_2028", 2028, 10, "random", False, "late_sat", "10-shot random anchor, then delayed full-DG satellite expansion."),
        ("S10_RND_RXGUARD_2028", 2028, 10, "random", False, "rxguard", "10-shot random anchor, then stronger receiver/day shortcut guard."),
        ("S10_RND_PROTO_2028", 2028, 10, "random", False, "proto_balanced", "10-shot anchor, then prototype/SupCon-balanced expansion."),
        ("S10_RND_STRONGSAT_NEG_2028", 2028, 10, "random", False, "strong_sat_neg", "Negative control: 10-shot anchor followed by early strong satellite pressure."),
        ("S10_FRONT_CLEAN_NEG_2028", 2028, 10, "front", False, "clean_expand", "Sample-selection negative control: front cap 10-shot anchor."),
        ("S10_RND_CLEAN_1337", 1337, 10, "random", False, "clean_expand", "Seed replicate for 10-shot clean staged anchor."),
        ("S10_RND_LATESAT_1337", 1337, 10, "random", False, "late_sat", "Seed replicate for 10-shot delayed satellite staged anchor."),
        ("S10_RND_LATESAT_2030", 2030, 10, "random", False, "late_sat", "Second split replicate for 10-shot delayed satellite staged anchor."),
        ("S20_RND_CLEAN_2028", 2028, 20, "random", False, "clean_expand", "20-shot anchor: tests whether more anchor samples improve clean expansion."),
        ("S20_RND_LATESAT_2028", 2028, 20, "random", False, "late_sat", "20-shot anchor with delayed satellite expansion."),
        ("S20_RND_RXGUARD_2028", 2028, 20, "random", False, "rxguard", "20-shot anchor with stronger receiver/day guard."),
        ("S30_RND_BAL_2028", 2028, 30, "random", False, "balanced", "30-shot staged balanced DG expansion."),
        ("S30_RND_SATFLOOR_2028", 2028, 30, "random", False, "sat_floor", "30-shot staged satellite-floor expansion."),
        ("S50_RND_BAL_2028", 2028, 50, "random", False, "balanced", "50-shot staged balanced expansion for monotonic scaling."),
        ("S50_RND_CLEAN_2028", 2028, 50, "random", False, "clean_expand", "50-shot staged clean-priority expansion."),
        ("D0P1_CLEAN_2028", 2028, 0, "random", True, "clean_expand", "Direct 0.1 clean expansion control without few-shot anchor."),
        ("D0P1_LATESAT_2028", 2028, 0, "random", True, "late_sat", "Direct 0.1 delayed satellite control without few-shot anchor."),
        ("D0P1_RXGUARD_2028", 2028, 0, "random", True, "rxguard", "Direct 0.1 rxguard control without few-shot anchor."),
        ("D0P1_STRONGSAT_NEG_2028", 2028, 0, "random", True, "strong_sat_neg", "Direct 0.1 early strong satellite negative control."),
        ("D0P1_BALANCED_1337", 1337, 0, "random", True, "balanced", "Direct 0.1 balanced seed control."),
    ]
    gpus = gpu_plan()
    candidates: list[Candidate] = []
    for idx, row in enumerate(rows):
        cid, seed, anchor_shots, cap, direct, kind, hyp = row
        stage1_epochs = 0 if direct else (70 if anchor_shots <= 5 else 80 if anchor_shots <= 10 else 90)
        stage2_epochs = 150 if anchor_shots <= 10 and not direct else 170
        if anchor_shots >= 30:
            stage2_epochs = 180
        run_name = f"CEN51_STAGED_{cid}_r010"
        candidates.append(
            Candidate(
                cid=cid,
                run_name=run_name,
                gpu=gpus[idx],
                seed=seed,
                anchor_shots=anchor_shots,
                stage2_shots=100,
                cap_strategy=cap,
                direct=direct,
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                stage2_kind=kind,
                hypothesis=hyp,
            )
        )
    return candidates


def q(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def bash_items(items: Iterable[str], indent: str = "  ") -> str:
    return "".join(f"{indent}{q(str(item))}\n" for item in items)


def common_args() -> list[str]:
    return [
        "--train_mode", "centralized",
        "--dataset", "wisig",
        "--wisig_protocol", "cvs_day_rx",
        "--wisig_domain", "rx_day",
        "--wisig_equalized", "1",
        "--wisig_train_ratio", "0.1",
        "--wisig_val_ratio", "-1.0",
        "--wisig_split_strategy", "random",
        "--wisig_train_days", "0,1",
        "--wisig_test_days", "2,3",
        "--wisig_train_rxs", "0,1,2,3,4,5,6",
        "--wisig_test_rxs", "7,8,9,10,11",
        "--eval_batch_size", "192",
        "--batch_size", "96",
        "--num_workers", "1",
        "--cpu_threads", "2",
        "--cpu_interop_threads", "1",
        "--prefetch_factor", "2",
        "--test_eval_policy", "interval_final",
        "--test_eval_interval", "25",
        "--arch_family", "cvsincnet",
        "--slim_group", "none",
        "--model_variant", "lite_d",
        "--branch_ablation", "no_dac",
        "--domain_branch_ablation", "no_stats",
        "--domain_enhancer", "rcn_stats",
        "--domain_enhancer_strength", "0.35",
        "--id_time_stability_mode", "off",
        "--id_freq_stability_mode", "off",
        "--domain_time_stability_mode", "off",
        "--domain_freq_stability_mode", "off",
        "--exp_group", "s3_rxrobust_no_dac",
        "--pa_orders", "1,3,5",
        "--collapse_guard",
        "--collapse_guard_min_epoch", "25",
        "--collapse_guard_best_margin", "10.0",
        "--collapse_guard_max_skipped_delta", "2",
        "--use_ema_ckpt",
        "--ema_decay", "0.999",
        "--use_swad_ckpt",
        "--swad_interval", "1",
        "--swad_tolerance", "0.9",
        "--primary_udu_weight", "0.80",
        "--label_smoothing", "0.0",
        "--no_enable_pa_aux",
        "--no_enable_dac_aux",
        "--no_aug_enable_pa_normal",
        "--aug_p_pa", "0.0",
        "--aug_p_dac", "0.0",
        "--lambda_cls_pa", "0.0",
        "--lambda_pa_joint_inv", "0.0",
        "--lambda_pa_kl", "0.0",
        "--lambda_pa_reg", "0.0",
    ]


def stage1_args(c: Candidate) -> list[str]:
    fn = 2.2e-4 if c.anchor_shots <= 5 else 1.6e-4 if c.anchor_shots <= 10 else 1.0e-4
    dom = 0.32 if c.anchor_shots <= 5 else 0.38 if c.anchor_shots <= 10 else 0.42
    adv = 0.12 if c.anchor_shots <= 5 else 0.16 if c.anchor_shots <= 10 else 0.20
    return [
        "--run_name", f"{c.run_name}_S1_anchor",
        "--epochs", str(c.stage1_epochs),
        "--seed", str(c.seed),
        "--wisig_cap_strategy", c.cap_strategy,
        "--wisig_max_train_per_combo", str(c.anchor_shots),
        "--test_eval_start_epoch", str(max(1, c.stage1_epochs - 24)),
        "--no_eval_sat_channel",
        "--no_use_aug",
        "--no_use_mixstyle",
        "--no_use_concat_sat_channel_aug",
        "--no_use_sat_consistency",
        "--lambda_sat_cls", "0.0",
        "--lambda_sat_cons", "0.0",
        "--concat_sat_ce_weight", "0.0",
        "--sat_cons_start_epoch", "999",
        "--lambda_dom", f"{dom:.4g}",
        "--lambda_adv", f"{adv:.4g}",
        "--grl_lambda", "1.0",
        "--lambda_orth", "0.01",
        "--lambda_cons", "0.0",
        "--lambda_group_ce", "0.0",
        "--lambda_proto", "0.0",
        "--lambda_supcon_id", "0.0",
        "--lambda_fishr", "0.0",
        "--lambda_feature_norm_guard", f"{fn:.4g}",
        "--feature_norm_guard_mode", "l2",
        "--feature_norm_guard_target", "0.0",
        "--no_use_proto_memory",
        "--swad_start_epoch", str(max(20, c.stage1_epochs // 2)),
    ]


def stage2_profile(kind: str, anchor_shots: int) -> dict[str, object]:
    low = anchor_shots <= 10
    if kind == "clean_expand":
        return dict(sat=False, aug=True, mix=True, sat_p=0.0, sat_start=999, scenarios=LIGHT_SAT,
                    dom=0.42 if low else 0.50, adv=0.20 if low else 0.28, gce=0.018 if low else 0.036,
                    proto=0.003 if low else 0.007, sup=0.003 if low else 0.007, fishr=0.0 if low else 0.0005,
                    fn=9e-5 if low else 5e-5, mix_p=0.04 if low else 0.08, mix_s=0.16 if low else 0.22,
                    aug_min=0.02 if low else 0.04, aug_max=0.14 if low else 0.22)
    if kind == "late_sat":
        return dict(sat=True, aug=True, mix=True, sat_p=0.16 if low else 0.28, sat_start=45 if low else 35, scenarios=ALL_SAT,
                    dom=0.45 if low else 0.52, adv=0.22 if low else 0.30, gce=0.022 if low else 0.040,
                    proto=0.004 if low else 0.008, sup=0.004 if low else 0.008, fishr=0.0 if low else 0.0008,
                    fn=8e-5 if low else 4.5e-5, mix_p=0.05 if low else 0.10, mix_s=0.18 if low else 0.24,
                    aug_min=0.02 if low else 0.05, aug_max=0.15 if low else 0.24)
    if kind == "rxguard":
        return dict(sat=True, aug=True, mix=True, sat_p=0.12 if low else 0.22, sat_start=35 if low else 25, scenarios=LIGHT_SAT,
                    dom=0.58 if low else 0.62, adv=0.24 if low else 0.34, gce=0.032 if low else 0.052,
                    proto=0.005 if low else 0.010, sup=0.005 if low else 0.010, fishr=0.0003 if low else 0.001,
                    fn=9e-5 if low else 4.5e-5, mix_p=0.04 if low else 0.09, mix_s=0.16 if low else 0.23,
                    aug_min=0.02 if low else 0.05, aug_max=0.14 if low else 0.24)
    if kind == "proto_balanced":
        return dict(sat=True, aug=True, mix=True, sat_p=0.18, sat_start=40, scenarios=ALL_SAT,
                    dom=0.48, adv=0.24, gce=0.036, proto=0.008, sup=0.010, fishr=0.0004,
                    fn=7e-5, mix_p=0.06, mix_s=0.18, aug_min=0.03, aug_max=0.18)
    if kind == "sat_floor":
        return dict(sat=True, aug=True, mix=True, sat_p=0.38, sat_start=25, scenarios=ALL_SAT,
                    dom=0.56, adv=0.30, gce=0.044, proto=0.008, sup=0.008, fishr=0.001,
                    fn=4e-5, mix_p=0.12, mix_s=0.25, aug_min=0.06, aug_max=0.26)
    if kind == "balanced":
        return dict(sat=True, aug=True, mix=True, sat_p=0.30 if low else 0.34, sat_start=30, scenarios=ALL_SAT,
                    dom=0.52 if low else 0.56, adv=0.28 if low else 0.32, gce=0.036 if low else 0.048,
                    proto=0.007 if low else 0.010, sup=0.007 if low else 0.010, fishr=0.0005 if low else 0.001,
                    fn=5e-5, mix_p=0.08 if low else 0.12, mix_s=0.22 if low else 0.25,
                    aug_min=0.04 if low else 0.06, aug_max=0.22 if low else 0.26)
    if kind == "strong_sat_neg":
        return dict(sat=True, aug=True, mix=True, sat_p=0.55 if low else 0.60, sat_start=1, scenarios=ALL_SAT,
                    dom=0.62, adv=0.36, gce=0.060, proto=0.012, sup=0.012, fishr=0.0015,
                    fn=3e-5, mix_p=0.18, mix_s=0.30, aug_min=0.08, aug_max=0.30)
    raise ValueError(f"unknown stage2 kind: {kind}")


def stage2_args(c: Candidate) -> list[str]:
    p = stage2_profile(c.stage2_kind, c.anchor_shots)
    args = [
        "--run_name", c.run_name,
        "--epochs", str(c.stage2_epochs),
        "--seed", str(c.seed),
        "--sat_view_seed", str(c.seed + 7919),
        "--wisig_cap_strategy", "random",
        "--wisig_max_train_per_combo", str(c.stage2_shots),
        "--test_eval_start_epoch", str(max(1, c.stage2_epochs - 74)),
        "--eval_sat_channel",
        "--eval_sat_on", "test_unseen_day_unseen_rx",
        "--eval_sat_scenarios", ALL_SAT,
        "--sat_eval_max_batches", "-1",
        "--lambda_dom", f"{p['dom']:.4g}",
        "--lambda_adv", f"{p['adv']:.4g}",
        "--grl_lambda", "1.0",
        "--lambda_orth", "0.02",
        "--lambda_cons", "0.006",
        "--lambda_group_ce", f"{p['gce']:.4g}",
        "--group_ce_mode", "smooth_dro_capped",
        "--group_ce_min_domains", "2",
        "--group_ce_top_frac", "0.20",
        "--groupdro_tau", "0.38",
        "--groupdro_cap", "0.50",
        "--lambda_proto", f"{p['proto']:.4g}",
        "--lambda_supcon_id", f"{p['sup']:.4g}",
        "--lambda_fishr", f"{p['fishr']:.4g}",
        "--fishr_min_domains", "2",
        "--lambda_feature_norm_guard", f"{p['fn']:.4g}",
        "--feature_norm_guard_mode", "l2",
        "--feature_norm_guard_target", "0.0",
        "--aug_scale_min", f"{p['aug_min']:.3f}",
        "--aug_scale_max", f"{p['aug_max']:.3f}",
        "--late_aug_min_scale", f"{p['aug_min']:.3f}",
        "--swad_start_epoch", str(max(45, c.stage2_epochs // 3)),
    ]
    args.append("--use_aug" if p["aug"] else "--no_use_aug")
    if p["mix"]:
        args.extend([
            "--use_mixstyle",
            "--mixstyle_p", f"{p['mix_p']:.3f}",
            "--mixstyle_strength", f"{p['mix_s']:.3f}",
            "--mixstyle_mix", "same_tx_crossdomain",
            "--mixstyle_fallback", "skip",
            "--mixstyle_late_start", str(max(60, c.stage2_epochs // 2)),
            "--mixstyle_late_ramp_epochs", "35",
            "--mixstyle_late_min_p", "0.020",
            "--mixstyle_late_min_strength", "0.120",
        ])
    else:
        args.append("--no_use_mixstyle")
    if p["sat"]:
        args.extend([
            "--use_concat_sat_channel_aug",
            "--no_use_sat_consistency",
            "--lambda_sat_cls", "0.0",
            "--lambda_sat_cons", "0.0",
            "--concat_sat_ce_weight", "0.0",
            "--sat_cons_start_epoch", "999",
            "--sat_view_prob", f"{p['sat_p']:.3f}",
            "--sat_train_scenarios", str(p["scenarios"]),
            "--concat_sat_start_epoch", str(p["sat_start"]),
            "--sat_view_schedule", f"1@{p['sat_p']:.3f}:{p['scenarios']}",
        ])
    else:
        args.extend([
            "--no_use_concat_sat_channel_aug",
            "--no_use_sat_consistency",
            "--lambda_sat_cls", "0.0",
            "--lambda_sat_cons", "0.0",
            "--concat_sat_ce_weight", "0.0",
            "--sat_cons_start_epoch", "999",
        ])
    args.append("--use_proto_memory" if float(p["proto"]) > 0.0 else "--no_use_proto_memory")
    return args


def render_candidate_block(c: Candidate) -> str:
    stage1 = stage1_args(c) if not c.direct else []
    stage2 = stage2_args(c)
    direct = "1" if c.direct else "0"
    return f"""
launch_{c.cid}() {{
  local candidate_id={q(c.cid)}
  local run_name={q(c.run_name)}
  local gpu={c.gpu}
  local direct={direct}
  if should_skip "${{candidate_id}}" "${{run_name}}"; then return 0; fi
  if ! reserve_gpu "${{candidate_id}}" "${{run_name}}" "${{gpu}}"; then return 0; fi
  if [[ "${{DRY_RUN}}" == "1" ]]; then return 0; fi
  local driver_log="${{LOG_ROOT}}/${{run_name}}.driver.out"
  (
    set -euo pipefail
    cd "${{ROOT}}"
    local init_ckpt=""
    if [[ "${{direct}}" == "0" ]]; then
      local s1_dir="${{RUNS_ROOT}}/${{run_name}}_S1_anchor"
      local s1_log="${{LOG_ROOT}}/${{run_name}}_S1_anchor.out"
      if [[ -e "${{s1_dir}}" || -e "${{s1_log}}" ]]; then echo "[BLOCKED] stage1 path collision ${{candidate_id}}"; exit 3; fi
      mkdir -p "${{s1_dir}}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${{gpu}}" "PYTHONPATH=${{ROOT}}/code:${{ROOT}}:${{PYTHONPATH:-}}" "${{PYTHON}}" -u "${{TRAIN_SCRIPT}}" "${{COMMON_ARGS[@]}}"
{bash_items(stage1, "        ").rstrip()}
        --latest_save_path "${{s1_dir}}/latest_model.pth"
        --best_save_path "${{s1_dir}}/best_val_model.pth"
        --best_primary_save_path "${{s1_dir}}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${{s1_dir}}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${{s1_dir}}/best_worst_rx_model.pth"
        --ema_save_path "${{s1_dir}}/ema_model.pth"
        --swa_save_path "${{s1_dir}}/swa_model.pth"
        --swad_save_path "${{s1_dir}}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${{s1_cmd[@]}}"
      "${{s1_cmd[@]}}" > "${{s1_log}}" 2>&1
      init_ckpt="$(choose_init_ckpt "${{s1_dir}}")"
      echo "[STAGE1-DONE] candidate=${{candidate_id}} init_ckpt=${{init_ckpt}}"
    fi
    local s2_dir="${{RUNS_ROOT}}/${{run_name}}"
    local s2_log="${{LOG_ROOT}}/${{run_name}}.out"
    if [[ -e "${{s2_dir}}" || -e "${{s2_log}}" ]]; then echo "[BLOCKED] stage2 path collision ${{candidate_id}}"; exit 3; fi
    mkdir -p "${{s2_dir}}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${{gpu}}" "PYTHONPATH=${{ROOT}}/code:${{ROOT}}:${{PYTHONPATH:-}}" "${{PYTHON}}" -u "${{TRAIN_SCRIPT}}" "${{COMMON_ARGS[@]}}"
{bash_items(stage2, "        ").rstrip()}
        --latest_save_path "${{s2_dir}}/latest_model.pth"
        --best_save_path "${{s2_dir}}/best_val_model.pth"
        --best_primary_save_path "${{s2_dir}}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${{s2_dir}}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${{s2_dir}}/best_worst_rx_model.pth"
        --ema_save_path "${{s2_dir}}/ema_model.pth"
        --swa_save_path "${{s2_dir}}/swa_model.pth"
        --swad_save_path "${{s2_dir}}/swad_model.pth")
    if [[ -n "${{init_ckpt}}" ]]; then s2_cmd+=(--init_checkpoint "${{init_ckpt}}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${{s2_cmd[@]}}"
    "${{s2_cmd[@]}}" > "${{s2_log}}" 2>&1
    echo "[CANDIDATE-DONE] ${{candidate_id}}"
  ) > "${{driver_log}}" 2>&1 &
  local driver_pid="$!"
  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${{candidate_id}}" "${{run_name}}" "{c.anchor_shots}" "${{gpu}}" "${{driver_pid}}" "${{driver_log}}" | tee -a "${{LOG_ROOT}}/launch_pids.tsv"
}}
"""


def render_launcher(run_id: str, candidates: list[Candidate]) -> str:
    blocks = "\n".join(render_candidate_block(c) for c in candidates)
    calls = "\n".join(f"launch_{c.cid}" for c in candidates)
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="${{ROOT:-{REMOTE_ROOT}}}"
PYTHON="${{PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}}"
TRAIN_SCRIPT="${{TRAIN_SCRIPT:-${{ROOT}}/code/train.py}}"
RUN_ID="${{RUN_ID:-{run_id}}}"
LOG_ROOT="${{LOG_ROOT:-${{ROOT}}/logs/${{RUN_ID}}}}"
RUNS_ROOT="${{RUNS_ROOT:-${{ROOT}}/runs/${{RUN_ID}}}}"
MAX_TRAIN_PER_GPU="${{MAX_TRAIN_PER_GPU:-4}}"
DRY_RUN="${{DRY_RUN:-0}}"
ONLY_CANDIDATE="${{ONLY_CANDIDATE:-}}"

for arg in "$@"; do
  case "${{arg}}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${{arg#--only=}}" ;;
    *) echo "[ERROR] unknown argument: ${{arg}}" >&2; exit 2 ;;
  esac
done

gpu_process_count() {{
  local gpu="$1"
  nvidia-smi --id="${{gpu}}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}}

print_cmd() {{ printf '%q ' "$@"; printf '\\n'; }}

should_skip() {{
  local candidate_id="$1"
  local run_name="$2"
  [[ -n "${{ONLY_CANDIDATE}}" && "${{candidate_id}}" != "${{ONLY_CANDIDATE}}" && "${{run_name}}" != "${{ONLY_CANDIDATE}}" ]]
}}

choose_init_ckpt() {{
  local run_dir="$1"
  for name in best_primary_ood_model.pth best_val_model.pth best_strict_udu_model.pth latest_model.pth; do
    if [[ -f "${{run_dir}}/${{name}}" ]]; then echo "${{run_dir}}/${{name}}"; return 0; fi
  done
  echo "[ERROR] no Stage-1 checkpoint in ${{run_dir}}" >&2
  return 4
}}

declare -A INITIAL_BY_GPU=()
declare -A LAUNCHED_BY_GPU=()

snapshot_capacity() {{
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    INITIAL_BY_GPU[${{gpu}}]="$(gpu_process_count "${{gpu}}")"
    LAUNCHED_BY_GPU[${{gpu}}]=0
  done
}}

reserve_gpu() {{
  local candidate_id="$1" run_name="$2" gpu="$3"
  if [[ "${{DRY_RUN}}" == "1" ]]; then
    echo "[DRY-RUN] reserve candidate=${{candidate_id}} gpu=${{gpu}}"
    return 0
  fi
  local initial_count="${{INITIAL_BY_GPU[${{gpu}}]:-0}}"
  local local_count="${{LAUNCHED_BY_GPU[${{gpu}}]:-0}}"
  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then
    printf "%s\\t%s\\t%s\\tgpu=%s initial_count=%s local_count=%s max=%s\\n" "${{candidate_id}}" "${{run_name}}" "BLOCKED_CAPACITY" "${{gpu}}" "${{initial_count}}" "${{local_count}}" "${{MAX_TRAIN_PER_GPU}}" | tee -a "${{LOG_ROOT}}/blocked.tsv"
    return 1
  fi
  LAUNCHED_BY_GPU[${{gpu}}]=$(( local_count + 1 ))
  return 0
}}

COMMON_ARGS=(
{bash_items(common_args()).rstrip()}
)

cd "${{ROOT}}"
mkdir -p "${{LOG_ROOT}}" "${{RUNS_ROOT}}"
snapshot_capacity
echo "[CEN51-STAGED] run_id=${{RUN_ID}} dry_run=${{DRY_RUN}} max_train_per_gpu=${{MAX_TRAIN_PER_GPU}} initial_gpu_counts=${{INITIAL_BY_GPU[*]}}"
if [[ "${{DRY_RUN}}" != "1" ]]; then
  [[ -f "${{TRAIN_SCRIPT}}" ]] || {{ echo "[ERROR] missing train script: ${{TRAIN_SCRIPT}}" >&2; exit 2; }}
fi

{blocks}

{calls}
echo "[CEN51-STAGED] launch submissions complete"
"""


def write_report(run_id: str, candidates: list[Candidate], script_path: Path, matrix_path: Path, report_path: Path) -> None:
    gpu_counts: dict[int, int] = {}
    for c in candidates:
        gpu_counts[c.gpu] = gpu_counts.get(c.gpu, 0) + 1
    rows = [
        "| ID | GPU | seed | anchor | selector | Stage-2 | direct | hypothesis |",
        "|---|---:|---:|---:|---|---|---:|---|",
    ]
    for c in candidates:
        rows.append(
            f"| `{c.cid}` | {c.gpu} | {c.seed} | {c.anchor_shots or '0.1'} | "
            f"{c.cap_strategy} | {c.stage2_kind} | {int(c.direct)} | {c.hypothesis} |"
        )
    report = f"""# {run_id}

## Objective

验证“极少样本 clean anchor 先压制 receiver/day/channel shortcut，再用 0.1 样本扩展鲁棒训练”的可行性。每个 staged 实验在同一 GPU 上顺序执行：

1. Stage-1：K-shot clean invariant anchor，关闭 satellite/MixStyle/强增广，只保留轻量 domain adversarial 与 feature-norm guard。
2. Stage-2：`wisig_max_train_per_combo=100` 的 0.1 扩展训练，从 Stage-1 checkpoint 初始化，并根据专家类型打开 clean / late-sat / rxguard / proto / sat-floor 等约束。

Direct controls 跳过 Stage-1，用相同 Stage-2 配置从随机初始化直接训练。

## Capacity

- GPU0-6：每卡 4 个 driver 实验。
- GPU7：3 个 driver 实验。
- 每个 driver 内部 Stage-1 与 Stage-2 串行，因此每个实验槽同一时刻只有一个 Python 训练进程。
- GPU counts: `{json.dumps(gpu_counts, sort_keys=True)}`

## Experimental Questions

- 极少样本 anchor 是否能让后续 0.1 扩展比 direct 0.1 更少学习 receiver/day/channel shortcut？
- random cap 是否稳定强于 front cap，证明样本选择/随机划分会影响低 shot 结论？
- late full-DG satellite 是否优于 early strong satellite，证明 satellite view 需要 gate？
- clean / late-sat / rxguard / sat-floor 专家是否有互补错误，为后续 LightMoE 提供依据？

## Local Artifacts

- Launcher: `{script_path.as_posix()}`
- Matrix JSON: `{matrix_path.as_posix()}`
- Report: `{report_path.as_posix()}`

## Remote Paths

- Project root: `{REMOTE_ROOT}`
- Launcher destination: `{REMOTE_ROOT}/code/scripts/{script_path.name}`
- Logs: `{REMOTE_ROOT}/logs/{run_id}/`
- Runs/checkpoints: `{REMOTE_ROOT}/runs/{run_id}/`

## Candidate Matrix

{chr(10).join(rows)}

## Success Criteria

- Staged K5/K10 的 best primary OOD 或 strict UDU 高于对应 direct 0.1 control，且 validation TX 不显著下降。
- random selector replicate 的方差小于 front negative control，并且 front control 不应成为稳定最优。
- `late_sat` 优于 `strong_sat_neg`，尤其是 clean strict UDU 与 receiver floor 不被 satellite 早期强扰动破坏。
- K20/K30/K50 staged 不能伤害 CVS 在 0.1 和更大样本下的优势；如果 staged 只提升 5/10 但压低 50/direct，即判为不可用。

## Verification Plan

- `conda activate ssr-gpu; python -m py_compile code/train.py tools/cen51_staged_anchor_expand_matrix.py`
- `conda activate ssr-gpu; python tools/cen51_staged_anchor_expand_matrix.py --run-id {run_id}`
- `bash -n code/scripts/launch_{run_id}.sh`
- remote dry-run: `MAX_TRAIN_PER_GPU=4 bash code/scripts/launch_{run_id}.sh --dry-run`
- startup health: inspect driver logs plus Stage-1/Stage-2 logs for `[INIT-CKPT]`, `[EPOCH-BEGIN]`, `Traceback`, `unrecognized`, OOM, and full-DG `ce_only=0` satellite logs.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    run_id = str(args.run_id)
    candidates = make_candidates()
    if len(candidates) != 31:
        raise SystemExit(f"expected 31 candidates, got {len(candidates)}")

    script_path = REPO_ROOT / "code" / "scripts" / f"launch_{run_id}.sh"
    report_dir = REPO_ROOT / "automation_reports" / "CV-SincNet" / run_id
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    script_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(run_id, candidates, script_path, matrix_path, report_path)
    print(json.dumps({"run_id": run_id, "candidates": len(candidates), "launcher": str(script_path), "matrix": str(matrix_path), "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
