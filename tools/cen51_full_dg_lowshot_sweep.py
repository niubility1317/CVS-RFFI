#!/usr/bin/env python
"""Generate the CEN51 full-DG satellite low-shot sweep launcher and report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
DEFAULT_RUN_ID = "cen51_full_dg_lowshot_sweep_20260609_162000"
ALL_SAT = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LIGHT_SAT = "clear_leo,mixed_orbit"


@dataclass(frozen=True)
class Candidate:
    cid: str
    run_name: str
    shots: int
    gpu: int
    seed: int
    epochs: int
    swad_start: int
    batch_size: int
    use_aug: bool
    use_mixstyle: bool
    use_sat: bool
    sat_prob: float
    sat_start: int
    sat_scenarios: str
    lambda_dom: float
    lambda_adv: float
    lambda_orth: float
    lambda_cons: float
    lambda_group_ce: float
    lambda_proto: float
    lambda_supcon: float
    lambda_fishr: float
    lambda_feature_norm: float
    group_top_frac: float
    group_tau: float
    group_cap: float
    mixstyle_p: float
    mixstyle_strength: float
    aug_scale_min: float
    aug_scale_max: float
    hypothesis: str


def make_candidates() -> list[Candidate]:
    rows = [
        # cid, shots, seed, epochs, swad, aug, mix, sat, p, start, scenarios,
        # dom, adv, orth, cons, group, proto, supcon, fishr, norm, top, tau, cap,
        # mix_p, mix_strength, aug_min, aug_max, hypothesis
        ("FS005_CLEAN_FN", 5, 2028, 190, 55, False, False, False, 0.00, 999, LIGHT_SAT, 0.32, 0.12, 0.00, 0.000, 0.000, 0.0000, 0.0000, 0.0000, 2.0e-4, 0.18, 0.35, 0.45, 0.00, 0.00, 0.00, 0.00, "5-shot clean receiver/day guard; tests whether sat-DG is still diluting identity."),
        ("FS005_SATMIN", 5, 2028, 190, 55, False, False, True, 0.08, 1, LIGHT_SAT, 0.35, 0.16, 0.01, 0.004, 0.004, 0.0005, 0.0005, 0.0000, 1.8e-4, 0.16, 0.35, 0.42, 0.00, 0.00, 0.00, 0.00, "Minimal full-DG satellite view with RIEI-like norm guard."),
        ("FS005_SATLATE", 5, 2028, 190, 60, False, False, True, 0.12, 80, LIGHT_SAT, 0.38, 0.18, 0.01, 0.006, 0.006, 0.0008, 0.0008, 0.0000, 1.6e-4, 0.16, 0.35, 0.42, 0.00, 0.00, 0.00, 0.00, "Late full-DG satellite so early epochs learn TX identity before channel invariance."),
        ("FS005_RXGUARD", 5, 2028, 190, 55, False, False, True, 0.10, 1, LIGHT_SAT, 0.50, 0.16, 0.02, 0.006, 0.006, 0.0005, 0.0005, 0.0000, 1.5e-4, 0.14, 0.32, 0.40, 0.00, 0.00, 0.00, 0.00, "Stronger domain pressure to suppress receiver/day shortcuts at 5-shot."),
        ("FS005_GROUPLIGHT", 5, 2028, 190, 60, False, True, True, 0.10, 50, LIGHT_SAT, 0.40, 0.18, 0.02, 0.008, 0.010, 0.0010, 0.0010, 0.0000, 1.2e-4, 0.18, 0.35, 0.45, 0.02, 0.12, 0.02, 0.12, "Light GroupCE/proto/SupCon after identity signal becomes stable."),
        ("FS005_SEED1337", 5, 1337, 190, 55, False, False, True, 0.08, 1, LIGHT_SAT, 0.35, 0.16, 0.01, 0.004, 0.004, 0.0005, 0.0005, 0.0000, 1.8e-4, 0.16, 0.35, 0.42, 0.00, 0.00, 0.00, 0.00, "Seed replicate of minimal full-DG satellite for split stability."),
        ("FS010_CLEAN_FN", 10, 2028, 195, 60, False, False, False, 0.00, 999, LIGHT_SAT, 0.38, 0.16, 0.01, 0.004, 0.006, 0.0010, 0.0010, 0.0000, 1.6e-4, 0.16, 0.35, 0.45, 0.00, 0.00, 0.00, 0.00, "10-shot clean guard control; separates data split from satellite effect."),
        ("FS010_SATMIN", 10, 2028, 195, 60, False, False, True, 0.12, 1, LIGHT_SAT, 0.40, 0.20, 0.02, 0.006, 0.010, 0.0015, 0.0015, 0.0000, 1.4e-4, 0.18, 0.35, 0.45, 0.00, 0.00, 0.00, 0.00, "10-shot weak full-DG satellite with moderate GroupCE/proto."),
        ("FS010_SATLATE_ALL", 10, 2028, 195, 65, False, False, True, 0.18, 60, ALL_SAT, 0.45, 0.22, 0.02, 0.008, 0.012, 0.0020, 0.0020, 0.0000, 1.2e-4, 0.18, 0.35, 0.45, 0.00, 0.00, 0.00, 0.00, "Late all-scenario satellite once 10-shot identity has separated."),
        ("FS010_GROUP_PROTO", 10, 2028, 195, 65, True, True, True, 0.16, 40, ALL_SAT, 0.42, 0.24, 0.02, 0.010, 0.018, 0.0030, 0.0030, 0.0000, 1.0e-4, 0.20, 0.40, 0.50, 0.03, 0.14, 0.03, 0.16, "Checks whether light augmentation plus prototype constraints are now beneficial."),
        ("FS010_RXGUARD", 10, 2028, 195, 60, False, False, True, 0.12, 1, LIGHT_SAT, 0.52, 0.18, 0.03, 0.006, 0.010, 0.0010, 0.0010, 0.0000, 1.4e-4, 0.14, 0.32, 0.40, 0.00, 0.00, 0.00, 0.00, "Higher receiver/day shortcut suppression while keeping sat weak."),
        ("FS010_SEED1337", 10, 1337, 195, 60, False, False, True, 0.12, 1, LIGHT_SAT, 0.40, 0.20, 0.02, 0.006, 0.010, 0.0015, 0.0015, 0.0000, 1.4e-4, 0.18, 0.35, 0.45, 0.00, 0.00, 0.00, 0.00, "Seed replicate of 10-shot satmin for stability."),
        ("FS020_BAL_FULLDG", 20, 2028, 200, 70, True, True, True, 0.18, 1, ALL_SAT, 0.45, 0.26, 0.03, 0.012, 0.020, 0.0030, 0.0030, 0.0002, 8.0e-5, 0.22, 0.42, 0.55, 0.04, 0.18, 0.04, 0.20, "20-shot balanced full-DG setting."),
        ("FS020_LATE_FULLDG", 20, 2028, 200, 70, True, True, True, 0.24, 50, ALL_SAT, 0.48, 0.26, 0.03, 0.012, 0.024, 0.0040, 0.0040, 0.0002, 7.0e-5, 0.22, 0.42, 0.55, 0.04, 0.18, 0.04, 0.20, "More satellite, but delayed to protect identity formation."),
        ("FS020_STRONG_DG", 20, 2028, 200, 75, True, True, True, 0.20, 1, ALL_SAT, 0.52, 0.32, 0.03, 0.014, 0.030, 0.0050, 0.0050, 0.0005, 6.0e-5, 0.24, 0.45, 0.58, 0.06, 0.20, 0.05, 0.22, "Stronger shortcut suppression once sample count can support it."),
        ("FS020_CLEANCTRL", 20, 2028, 200, 70, True, True, False, 0.00, 999, LIGHT_SAT, 0.45, 0.24, 0.02, 0.010, 0.020, 0.0030, 0.0030, 0.0002, 8.0e-5, 0.22, 0.42, 0.55, 0.04, 0.18, 0.04, 0.20, "20-shot no-satellite control for monotonic clean performance."),
        ("FS030_BAL_FULLDG", 30, 2028, 200, 75, True, True, True, 0.24, 1, ALL_SAT, 0.50, 0.30, 0.03, 0.014, 0.034, 0.0060, 0.0060, 0.0007, 5.0e-5, 0.26, 0.45, 0.60, 0.08, 0.22, 0.05, 0.24, "30-shot balanced setting with stronger DG."),
        ("FS030_SATFLOOR", 30, 2028, 200, 75, True, True, True, 0.32, 1, ALL_SAT, 0.55, 0.28, 0.03, 0.014, 0.030, 0.0050, 0.0050, 0.0005, 5.0e-5, 0.24, 0.42, 0.58, 0.08, 0.22, 0.05, 0.24, "Higher satellite floor for strict satellite robustness."),
        ("FS030_LATE_SAT", 30, 2028, 200, 75, True, True, True, 0.36, 40, ALL_SAT, 0.52, 0.30, 0.03, 0.016, 0.038, 0.0070, 0.0070, 0.0007, 4.5e-5, 0.26, 0.45, 0.60, 0.08, 0.22, 0.05, 0.24, "Tests delayed high satellite pressure at 30-shot."),
        ("FS030_CLEANCTRL", 30, 2028, 200, 75, True, True, False, 0.00, 999, LIGHT_SAT, 0.50, 0.28, 0.02, 0.012, 0.034, 0.0060, 0.0060, 0.0005, 5.0e-5, 0.26, 0.45, 0.60, 0.08, 0.22, 0.05, 0.24, "30-shot no-satellite control for clean ceiling."),
        ("FS050_BAL_FULLDG", 50, 2028, 210, 80, True, True, True, 0.35, 1, ALL_SAT, 0.55, 0.32, 0.03, 0.016, 0.045, 0.0080, 0.0080, 0.0010, 4.0e-5, 0.28, 0.48, 0.62, 0.12, 0.25, 0.06, 0.26, "50-shot balanced high-capacity DG."),
        ("FS050_SATSTRONG", 50, 2028, 210, 80, True, True, True, 0.45, 1, ALL_SAT, 0.60, 0.30, 0.03, 0.016, 0.040, 0.0070, 0.0070, 0.0015, 4.0e-5, 0.28, 0.48, 0.62, 0.12, 0.25, 0.06, 0.26, "High satellite pressure to optimize strict satellite while guarding clean."),
        ("FS050_CLEAN_CEIL", 50, 2028, 210, 80, True, True, True, 0.28, 30, LIGHT_SAT, 0.55, 0.32, 0.03, 0.018, 0.052, 0.0100, 0.0100, 0.0005, 3.5e-5, 0.30, 0.50, 0.65, 0.14, 0.25, 0.06, 0.26, "Clean-priority 50-shot ceiling with only light satellite."),
    ]
    gpus = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6]
    candidates: list[Candidate] = []
    for idx, row in enumerate(rows):
        cid = row[0]
        shots = row[1]
        run_name = f"CEN51_FULLDG_{cid}_seed{row[2]}"
        candidates.append(Candidate(cid, run_name, shots, gpus[idx], row[2], row[3], row[4], 128, *row[5:]))
    return candidates


def q(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def bash_array_items(items: Iterable[str], indent: str = "  ") -> str:
    return "".join(f"{indent}{q(str(item))}\n" for item in items)


def base_args() -> list[str]:
    return [
        "--train_mode", "centralized",
        "--eval_batch_size", "256",
        "--dataset", "wisig",
        "--wisig_protocol", "cvs_day_rx",
        "--wisig_domain", "rx_day",
        "--wisig_equalized", "1",
        "--wisig_train_ratio", "0.1",
        "--wisig_val_ratio", "-1.0",
        "--wisig_split_strategy", "random",
        "--wisig_cap_strategy", "random",
        "--wisig_train_days", "0,1",
        "--wisig_test_days", "2,3",
        "--wisig_train_rxs", "0,1,2,3,4,5,6",
        "--wisig_test_rxs", "7,8,9,10,11",
        "--test_eval_policy", "interval_final",
        "--test_eval_start_epoch", "31",
        "--test_eval_interval", "10",
        "--eval_sat_channel",
        "--eval_sat_on", "test_unseen_day_unseen_rx",
        "--eval_sat_scenarios", ALL_SAT,
        "--sat_eval_max_batches", "-1",
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
        "--collapse_guard_min_epoch", "35",
        "--collapse_guard_best_margin", "10.0",
        "--collapse_guard_max_skipped_delta", "2",
        "--use_ema_ckpt",
        "--ema_decay", "0.999",
        "--use_swad_ckpt",
        "--swad_interval", "1",
        "--swad_tolerance", "1.0",
        "--primary_udu_weight", "0.78",
        "--label_smoothing", "0.0",
    ]


def candidate_args(c: Candidate) -> list[str]:
    args = [
        "--batch_size", str(c.batch_size),
        "--epochs", str(c.epochs),
        "--swad_start_epoch", str(c.swad_start),
        "--seed", str(c.seed),
        "--sat_view_seed", str(c.seed + 7919),
        "--no_enable_pa_aux",
        "--no_enable_dac_aux",
        "--no_aug_enable_pa_normal",
        "--aug_p_pa", "0.0",
        "--aug_p_dac", "0.0",
        "--lambda_cls_pa", "0.0",
        "--lambda_pa_joint_inv", "0.0",
        "--lambda_pa_kl", "0.0",
        "--lambda_pa_reg", "0.0",
        "--no_use_sat_consistency",
        "--lambda_sat_cls", "0.0",
        "--lambda_sat_cons", "0.0",
        "--concat_sat_ce_weight", "0.0",
        "--sat_cons_start_epoch", "999",
        "--sat_view_prob", f"{c.sat_prob:.3f}",
        "--sat_train_scenarios", c.sat_scenarios,
        "--concat_sat_start_epoch", str(c.sat_start),
        "--lambda_dom", f"{c.lambda_dom:.4g}",
        "--lambda_adv", f"{c.lambda_adv:.4g}",
        "--grl_lambda", "1.0",
        "--lambda_orth", f"{c.lambda_orth:.4g}",
        "--lambda_cons", f"{c.lambda_cons:.4g}",
        "--lambda_group_ce", f"{c.lambda_group_ce:.4g}",
        "--lambda_proto", f"{c.lambda_proto:.4g}",
        "--lambda_supcon_id", f"{c.lambda_supcon:.4g}",
        "--lambda_fishr", f"{c.lambda_fishr:.4g}",
        "--lambda_feature_norm_guard", f"{c.lambda_feature_norm:.4g}",
        "--feature_norm_guard_mode", "l2",
        "--feature_norm_guard_target", "0.0",
        "--group_ce_mode", "smooth_dro_capped",
        "--group_ce_min_domains", "2",
        "--group_ce_top_frac", f"{c.group_top_frac:.3f}",
        "--groupdro_tau", f"{c.group_tau:.3f}",
        "--groupdro_cap", f"{c.group_cap:.3f}",
        "--fishr_min_domains", "2",
        "--aug_scale_min", f"{c.aug_scale_min:.3f}",
        "--aug_scale_max", f"{c.aug_scale_max:.3f}",
        "--late_aug_min_scale", f"{max(c.aug_scale_min, min(c.aug_scale_max, 0.16)):.3f}",
    ]
    args.append("--use_aug" if c.use_aug else "--no_use_aug")
    args.append("--use_mixstyle" if c.use_mixstyle else "--no_use_mixstyle")
    if c.use_mixstyle:
        args.extend([
            "--mixstyle_p", f"{c.mixstyle_p:.3f}",
            "--mixstyle_strength", f"{c.mixstyle_strength:.3f}",
            "--mixstyle_mix", "same_tx_crossdomain",
            "--mixstyle_fallback", "skip",
            "--mixstyle_late_start", str(max(60, c.swad_start + 20)),
            "--mixstyle_late_ramp_epochs", "35",
            "--mixstyle_late_min_p", f"{max(0.02, c.mixstyle_p * 0.5):.3f}",
            "--mixstyle_late_min_strength", f"{max(0.10, c.mixstyle_strength * 0.65):.3f}",
        ])
    if c.use_sat:
        args.append("--use_concat_sat_channel_aug")
        args.extend(["--sat_view_schedule", f"1@{c.sat_prob:.3f}:{c.sat_scenarios}"])
    else:
        args.append("--no_use_concat_sat_channel_aug")
    if c.lambda_proto > 0.0:
        args.append("--use_proto_memory")
    else:
        args.append("--no_use_proto_memory")
    return args


def render_launcher(run_id: str, candidates: list[Candidate]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"ROOT=\"${{ROOT:-{REMOTE_ROOT}}}\"",
        "PYTHON=\"${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}\"",
        "TRAIN_SCRIPT=\"${TRAIN_SCRIPT:-${ROOT}/code/train.py}\"",
        f"RUN_ID=\"${{RUN_ID:-{run_id}}}\"",
        "LOG_ROOT=\"${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}\"",
        "RUNS_ROOT=\"${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}\"",
        "MAX_TRAIN_PER_GPU=\"${MAX_TRAIN_PER_GPU:-3}\"",
        "DRY_RUN=\"${DRY_RUN:-0}\"",
        "ONLY_CANDIDATE=\"${ONLY_CANDIDATE:-}\"",
        "",
        "for arg in \"$@\"; do",
        "  case \"${arg}\" in",
        "    --dry-run) DRY_RUN=1 ;;",
        "    --only=*) ONLY_CANDIDATE=\"${arg#--only=}\" ;;",
        "    *) echo \"[ERROR] unknown argument: ${arg}\" >&2; exit 2 ;;",
        "  esac",
        "done",
        "",
        "gpu_process_count() {",
        "  local gpu=\"$1\"",
        "  nvidia-smi --id=\"${gpu}\" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \\",
        "    | sed '/^$/d' | wc -l | tr -d ' '",
        "}",
        "",
        "print_cmd() { printf '%q ' \"$@\"; printf '\\n'; }",
        "",
        "should_skip() {",
        "  local candidate_id=\"$1\"",
        "  local run_name=\"$2\"",
        "  [[ -n \"${ONLY_CANDIDATE}\" && \"${candidate_id}\" != \"${ONLY_CANDIDATE}\" && \"${run_name}\" != \"${ONLY_CANDIDATE}\" ]]",
        "}",
        "",
        "declare -A INITIAL_BY_GPU=()",
        "declare -A LAUNCHED_BY_GPU=()",
        "snapshot_capacity() {",
        "  local gpu",
        "  for gpu in 0 1 2 3 4 5 6 7; do",
        "    INITIAL_BY_GPU[${gpu}]=\"$(gpu_process_count \"${gpu}\")\"",
        "    LAUNCHED_BY_GPU[${gpu}]=0",
        "  done",
        "}",
        "",
        "run_candidate() {",
        "  local candidate_id=\"$1\" run_name=\"$2\" shots=\"$3\" gpu=\"$4\"",
        "  shift 4",
        "  local run_dir=\"${RUNS_ROOT}/${run_name}\"",
        "  local log_path=\"${LOG_ROOT}/${run_name}.out\"",
        "  local cmd=(env \"CUDA_VISIBLE_DEVICES=${gpu}\" \"PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}\" \"${PYTHON}\" -u \"${TRAIN_SCRIPT}\" \"${BASE_ARGS[@]}\" \"$@\"",
        "    --run_name \"${run_name}\"",
        "    --wisig_max_train_per_combo \"${shots}\"",
        "    --latest_save_path \"${run_dir}/latest_model.pth\"",
        "    --best_save_path \"${run_dir}/best_val_model.pth\"",
        "    --best_primary_save_path \"${run_dir}/best_primary_ood_model.pth\"",
        "    --best_unseen_day_unseen_rx_save_path \"${run_dir}/best_strict_udu_model.pth\"",
        "    --best_worst_rx_save_path \"${run_dir}/best_worst_rx_model.pth\"",
        "    --ema_save_path \"${run_dir}/ema_model.pth\"",
        "    --swa_save_path \"${run_dir}/swa_model.pth\"",
        "    --swad_save_path \"${run_dir}/swad_model.pth\")",
        "",
        "  if should_skip \"${candidate_id}\" \"${run_name}\"; then return 0; fi",
        "  echo \"[CEN51-FULLDG] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}\"",
        "  printf '[CEN51-FULLDG-CMD]'",
        "  print_cmd \"${cmd[@]}\"",
        "  if [[ \"${DRY_RUN}\" == \"1\" ]]; then return 0; fi",
        "  if [[ -e \"${run_dir}\" || -e \"${log_path}\" ]]; then",
        "    mkdir -p \"${LOG_ROOT}\"",
        "    printf \"%s\\t%s\\t%s\\t%s\\t%s\\n\" \"${candidate_id}\" \"${run_name}\" \"BLOCKED_PATH_COLLISION\" \"${log_path}\" \"${run_dir}\" | tee -a \"${LOG_ROOT}/blocked.tsv\"",
        "    return 0",
        "  fi",
        "  local initial_count local_count",
        "  initial_count=\"${INITIAL_BY_GPU[${gpu}]:-0}\"",
        "  local_count=\"${LAUNCHED_BY_GPU[${gpu}]:-0}\"",
        "  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then",
        "    mkdir -p \"${LOG_ROOT}\"",
        "    printf \"%s\\t%s\\t%s\\tgpu=%s initial_count=%s local_count=%s max=%s\\n\" \"${candidate_id}\" \"${run_name}\" \"BLOCKED_CAPACITY\" \"${gpu}\" \"${initial_count}\" \"${local_count}\" \"${MAX_TRAIN_PER_GPU}\" | tee -a \"${LOG_ROOT}/blocked.tsv\"",
        "    return 0",
        "  fi",
        "  mkdir -p \"${LOG_ROOT}\" \"${run_dir}\"",
        "  nohup \"${cmd[@]}\" > \"${log_path}\" 2>&1 &",
        "  local pid=\"$!\"",
        "  LAUNCHED_BY_GPU[${gpu}]=$(( local_count + 1 ))",
        "  printf \"%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n\" \"${candidate_id}\" \"${run_name}\" \"${shots}\" \"${gpu}\" \"${pid}\" \"${log_path}\" \"${run_dir}\" | tee -a \"${LOG_ROOT}/launch_pids.tsv\"",
        "}",
        "",
        "BASE_ARGS=(",
        bash_array_items(base_args()).rstrip(),
        ")",
        "",
        "if [[ \"${DRY_RUN}\" != \"1\" ]]; then",
        "  [[ -f \"${TRAIN_SCRIPT}\" ]] || { echo \"[ERROR] missing train script: ${TRAIN_SCRIPT}\" >&2; exit 2; }",
        "fi",
        "cd \"${ROOT}\"",
        "snapshot_capacity",
        "echo \"[CEN51-FULLDG] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}\"",
        "echo \"[CEN51-FULLDG] initial_gpu_counts: ${INITIAL_BY_GPU[*]}\"",
        "",
    ]
    for c in candidates:
        lines.append(
            f"run_candidate {c.cid} {c.run_name} {c.shots} {c.gpu} \\"
        )
        args = candidate_args(c)
        for i, item in enumerate(args):
            suffix = " \\" if i < len(args) - 1 else ""
            lines.append(f"  {q(item)}{suffix}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(run_id: str, candidates: list[Candidate], report_path: Path, script_path: Path, matrix_path: Path) -> None:
    counts: dict[int, int] = {}
    shot_counts: dict[int, int] = {}
    for c in candidates:
        counts[c.gpu] = counts.get(c.gpu, 0) + 1
        shot_counts[c.shots] = shot_counts.get(c.shots, 0) + 1
    rows = [
        "| ID | shots | GPU | seed | sat | p/start | losses | hypothesis |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for c in candidates:
        sat = "full-DG" if c.use_sat else "clean-only"
        losses = (
            f"dom={c.lambda_dom}, adv={c.lambda_adv}, gce={c.lambda_group_ce}, "
            f"proto={c.lambda_proto}, sup={c.lambda_supcon}, fishr={c.lambda_fishr}, "
            f"fn={c.lambda_feature_norm}"
        )
        rows.append(
            f"| {c.cid} | {c.shots} | {c.gpu} | {c.seed} | {sat} | "
            f"{c.sat_prob}/{c.sat_start} | {losses} | {c.hypothesis} |"
        )
    report = f"""# {run_id}

## Objective

继续优化 CVS/CEN51 低 shots 设置，目标不是只抬 5/10 shots，而是找到一个随样本增多可以单调变强的配置族。默认 satellite view 使用 full-DG：卫星增强样本进入主 TX 分类、domain/GRL、GroupCE、prototype、SupCon、Fishr、feature-norm guard 等训练路径；不使用 `concat_sat_ce_only`。

## Hypothesis

RIEI 的核心启示是极少样本下泛化瓶颈主要来自 receiver/day/channel 捷径，而不是继续堆容量。本 sweep 因此沿三条轴搜索：

- 5/10 shots：强 feature-norm guard、弱或 late full-DG satellite、低 MixStyle/augmentation，优先保护 TX identity。
- 20/30 shots：逐步增加 GroupCE/proto/SupCon/Fishr 和 full-DG satellite 概率，验证样本增多后 DG 约束能转化为严格 UDU 和 satellite robustness。
- 50 shots：提高 satellite floor 和 DG 强度，同时保留 clean-priority 候选，避免只优化星地而牺牲 clean strict。

## Capacity Plan

- GPU0-6: each 3 experiments.
- GPU7: 2 experiments.
- Launcher capacity guard uses frozen initial GPU occupancy plus the number launched by this script, so third local launch on a GPU is not double-counted.

GPU counts: {json.dumps(counts, sort_keys=True)}
Shot counts: {json.dumps(shot_counts, sort_keys=True)}

## Local Artifacts

- Launcher: `{script_path.as_posix()}`
- Matrix JSON: `{matrix_path.as_posix()}`
- Report: `{report_path.as_posix()}`

## Remote Paths

- Project root: `{REMOTE_ROOT}`
- Launcher destination: `{REMOTE_ROOT}/code/scripts/{script_path.name}`
- Logs: `{REMOTE_ROOT}/logs/{run_id}/`
- Checkpoints: `{REMOTE_ROOT}/runs/{run_id}/`

## Verification Before Launch

- `python -m py_compile tools/cen51_full_dg_lowshot_sweep.py`
- `conda activate ssr-gpu; python tools/cen51_full_dg_lowshot_sweep.py --run-id {run_id}`
- `bash -n code/scripts/{script_path.name}`
- Remote dry run: `bash code/scripts/{script_path.name} --dry-run`

## Candidate Matrix

{chr(10).join(rows)}

## Metrics To Watch

- validation TX accuracy, but not as the sole selector;
- clean strict UDU, satellite strict UDU, primary OOD score;
- domain accuracy/leakage, feature norm, prototype dispersion, loss balance;
- whether 20/30/50 shots improve over 5/10 rather than showing inverse scaling.

## Risks

- Full-DG satellite doubles the effective batch path when enabled; low shots may still over-regularize if p is too high.
- Validation can remain lower than source training accuracy if the split is genuinely random and receiver/day shortcuts are suppressed.
- 23 concurrent jobs may make validation slower; startup health should inspect logs for argument or OOM failures.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    run_id = args.run_id
    candidates = make_candidates()
    script_path = REPO_ROOT / "code" / "scripts" / f"launch_{run_id}.sh"
    report_dir = REPO_ROOT / "automation_reports" / "CV-SincNet" / run_id
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"

    report_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    matrix_path.write_text(
        json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(run_id, candidates, report_path, script_path, matrix_path)
    print(f"[OK] launcher={script_path}")
    print(f"[OK] matrix={matrix_path}")
    print(f"[OK] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
