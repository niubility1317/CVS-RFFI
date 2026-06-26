#!/usr/bin/env python3
"""Generate a RIEI-style shot-aware CEN51 CVS guard sweep.

The matrix targets 5/10-shot recovery and 20/30/50-shot monotonic improvement.
It keeps the CEN51 backbone, but tests whether explicit nuisance suppression
at very low shots and relaxed identity-preserving regularization at higher
shots work better than the heavier R04 and LAC low-shot recipes:

* CE + domain-style branch + GRL + feature-norm guard.
* CE + feature-norm guard only, to expose whether GRL hurts source-val.
* CE + GRL + feature-norm guard only, to isolate the adversary path.
* Shot-aware clean/satellite-balanced variants whose regularization is relaxed
  as samples increase.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


SAT_SCENARIOS = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"


def arg_pairs(params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            args.append(flag if value else f"--no_{key}")
        elif value is None:
            continue
        else:
            args.extend([flag, str(value)])
    return args


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


@dataclass
class Candidate:
    candidate_id: str
    run_name: str
    shots: int
    gpu: int
    strategy: str
    rationale: str
    success_gate: str
    params: Dict[str, object] = field(default_factory=dict)

    def args(self) -> List[str]:
        return arg_pairs(self.params)


BASE_PARAMS: Dict[str, object] = {
    "train_mode": "centralized",
    "eval_batch_size": 256,
    "dataset": "wisig",
    "wisig_protocol": "cvs_day_rx",
    "wisig_domain": "rx_day",
    "wisig_train_ratio": 0.1,
    "wisig_split_strategy": "random",
    "wisig_cap_strategy": "random",
    "test_eval_policy": "interval_final",
    "test_eval_start_epoch": 31,
    "test_eval_interval": 10,
    "eval_sat_channel": True,
    "eval_sat_on": "test_unseen_day_unseen_rx",
    "eval_sat_scenarios": SAT_SCENARIOS,
    "sat_eval_max_batches": -1,
    "arch_family": "cvsincnet",
    "slim_group": "none",
    "model_variant": "lite_d",
    "branch_ablation": "no_dac",
    "domain_branch_ablation": "no_stats",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    "id_time_stability_mode": "off",
    "id_freq_stability_mode": "off",
    "domain_time_stability_mode": "off",
    "domain_freq_stability_mode": "off",
    "exp_group": "s3_rxrobust_no_dac",
    "pa_orders": "1,3,5",
    "collapse_guard": True,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 10.0,
    "collapse_guard_max_skipped_delta": 2,
    "use_ema_ckpt": True,
    "ema_decay": 0.999,
    "use_swad_ckpt": True,
    "swad_interval": 1,
    "swad_tolerance": 1.0,
    "primary_udu_weight": 0.78,
    "label_smoothing": 0.0,
    "seed": 2028,
}


def clean_guard_params(
    *,
    shots: int,
    adv: float,
    norm: float,
    dom: float = 0.35,
    norm_mode: str = "l2",
    norm_target: float = 0.0,
) -> Dict[str, object]:
    epochs = 150 if shots <= 5 else 160
    return {
        "batch_size": 128,
        "epochs": epochs,
        "swad_start_epoch": 45 if shots <= 5 else 55,
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
        "use_sat_consistency": False,
        "use_concat_sat_channel_aug": False,
        "lambda_sat_cls": 0.0,
        "lambda_sat_cons": 0.0,
        "concat_sat_ce_weight": 0.0,
        "sat_view_prob": 0.0,
        "sat_cons_start_epoch": 999,
        "lambda_dom": dom,
        "lambda_adv": adv,
        "grl_lambda": 1.0,
        "lambda_orth": 0.01 if dom > 0.0 and adv > 0.0 else 0.0,
        "lambda_cons": 0.0,
        "lambda_group_ce": 0.0,
        "lambda_proto": 0.0,
        "lambda_supcon_id": 0.0,
        "lambda_fishr": 0.0,
        "lambda_feature_norm_guard": norm,
        "feature_norm_guard_mode": norm_mode,
        "feature_norm_guard_target": norm_target,
    }


def force_grl_params(*, shots: int, adv: float, norm: float) -> Dict[str, object]:
    params = clean_guard_params(shots=shots, adv=adv, norm=norm, dom=0.0)
    params["force_ce_grl_only"] = True
    return params


def gated_sat_params(*, shots: int, adv: float, norm: float, sat_weight: float) -> Dict[str, object]:
    params = clean_guard_params(shots=shots, adv=adv, norm=norm, dom=0.35)
    params.update(
        {
            "use_concat_sat_channel_aug": True,
            "concat_sat_ce_weight": sat_weight,
            "sat_view_prob": sat_weight,
            "sat_view_schedule": f"1@{sat_weight:.2f}:clear_leo,mixed_orbit",
            "sat_train_scenarios": "clear_leo,mixed_orbit",
        }
    )
    return params


def shotaware_params(
    *,
    shots: int,
    adv: float,
    dom: float,
    norm: float,
    group_ce: float,
    proto: float,
    supcon: float,
    fishr: float,
    mix_p: float,
    mix_strength: float,
    sat_weight: float = 0.0,
) -> Dict[str, object]:
    epochs_by_shot = {20: 175, 30: 185, 50: 195}
    epochs = int(epochs_by_shot.get(shots, 180))
    params = clean_guard_params(shots=shots, adv=adv, norm=norm, dom=dom)
    params.update(
        {
            "epochs": epochs,
            "swad_start_epoch": 65 if shots <= 20 else (75 if shots <= 30 else 85),
            "use_aug": True,
            "aug_scale_min": 0.04,
            "aug_scale_max": 0.20 if shots <= 20 else (0.24 if shots <= 30 else 0.28),
            "late_aug_min_scale": 0.16 if shots <= 20 else (0.18 if shots <= 30 else 0.20),
            "use_mixstyle": True,
            "mixstyle_p": mix_p,
            "mixstyle_strength": mix_strength,
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_late_start": 110 if shots <= 20 else (120 if shots <= 30 else 130),
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": max(0.02, mix_p * 0.45),
            "mixstyle_late_min_strength": max(0.12, mix_strength * 0.55),
            "lambda_orth": 0.02,
            "lambda_cons": 0.01 if shots <= 20 else (0.015 if shots <= 30 else 0.02),
            "lambda_group_ce": group_ce,
            "group_ce_mode": "smooth_dro_capped",
            "group_ce_min_domains": 2,
            "group_ce_top_frac": 0.18 if shots <= 20 else (0.22 if shots <= 30 else 0.25),
            "groupdro_tau": 0.35 if shots <= 20 else (0.42 if shots <= 30 else 0.50),
            "groupdro_cap": 0.45 if shots <= 20 else (0.55 if shots <= 30 else 0.62),
            "use_proto_memory": proto > 0.0,
            "lambda_proto": proto,
            "lambda_supcon_id": supcon,
            "lambda_fishr": fishr,
            "fishr_min_domains": 2,
            "feature_norm_guard_mode": "l2",
            "feature_norm_guard_target": 0.0,
        }
    )
    if sat_weight > 0.0:
        params.update(
            {
                "use_concat_sat_channel_aug": True,
                "concat_sat_ce_weight": sat_weight,
                "sat_view_prob": sat_weight,
                "sat_view_schedule": f"1@{sat_weight:.2f}:clear_leo,mixed_orbit",
                "sat_train_scenarios": "clear_leo,mixed_orbit",
                "use_sat_consistency": False,
                "lambda_sat_cons": 0.0,
                "sat_cons_start_epoch": 999,
            }
        )
    return params


def make_candidates() -> List[Candidate]:
    specs = [
        (
            "FS005",
            5,
            [
                ("CE_FN_L2", "CE + mild feature-norm guard only; source-val upper-bound check", clean_guard_params(shots=5, adv=0.0, norm=0.0002, dom=0.0)),
                (
                    "FORCE_GRLLOW_HINGE",
                    "CE + low GRL + hinge feature-norm guard to avoid over-compressing TX separability",
                    {
                        **force_grl_params(shots=5, adv=0.12, norm=0.0010),
                        "feature_norm_guard_mode": "hinge",
                        "feature_norm_guard_target": 8.0,
                    },
                ),
                ("DOMSIDE_GRLLOW_L2", "CE + supervised nuisance side branch + low GRL + mild L2 guard", clean_guard_params(shots=5, adv=0.12, norm=0.0002, dom=0.35)),
            ],
        ),
        (
            "FS010",
            10,
            [
                ("CE_FN_L2", "CE + mild feature-norm guard only; source-val upper-bound check", clean_guard_params(shots=10, adv=0.0, norm=0.00015, dom=0.0)),
                ("FORCE_GRLMID_L2", "CE + moderate GRL + mild L2 feature-norm guard with every other loss disabled", force_grl_params(shots=10, adv=0.18, norm=0.00015)),
                ("DOMSIDE_GRLMID_L2", "CE + supervised nuisance side branch + moderate GRL + mild L2 guard", clean_guard_params(shots=10, adv=0.18, norm=0.00015, dom=0.35)),
            ],
        ),
        (
            "FS020",
            20,
            [
                ("SHOTAWARE_CLEANBAL", "Relaxed norm/GRL with weak GroupCE/proto/SupCon/MixStyle for 20-shot clean growth", shotaware_params(shots=20, adv=0.20, dom=0.45, norm=0.00010, group_ce=0.015, proto=0.0015, supcon=0.0015, fishr=0.0, mix_p=0.04, mix_strength=0.18)),
                ("SHOTAWARE_SATGATE", "20-shot clean-balanced candidate with a very weak clean/mixed satellite view", shotaware_params(shots=20, adv=0.18, dom=0.45, norm=0.00008, group_ce=0.012, proto=0.0015, supcon=0.0015, fishr=0.0, mix_p=0.04, mix_strength=0.18, sat_weight=0.14)),
            ],
        ),
        (
            "FS030",
            30,
            [
                ("SHOTAWARE_CLEANBAL", "Relaxed norm/GRL with moderate GroupCE/proto/SupCon/MixStyle for 30-shot clean growth", shotaware_params(shots=30, adv=0.24, dom=0.50, norm=0.00008, group_ce=0.022, proto=0.0025, supcon=0.0025, fishr=0.0003, mix_p=0.06, mix_strength=0.24)),
                ("SHOTAWARE_SATGATE", "30-shot clean-balanced candidate with a weak clean/mixed satellite view", shotaware_params(shots=30, adv=0.22, dom=0.50, norm=0.00006, group_ce=0.020, proto=0.0025, supcon=0.0025, fishr=0.0003, mix_p=0.06, mix_strength=0.24, sat_weight=0.18)),
            ],
        ),
        (
            "FS050",
            50,
            [
                ("SHOTAWARE_CLEANBAL", "Further relaxed norm with stronger identity-preserving DG for 50-shot growth", shotaware_params(shots=50, adv=0.28, dom=0.55, norm=0.00005, group_ce=0.032, proto=0.0040, supcon=0.0040, fishr=0.0005, mix_p=0.10, mix_strength=0.32)),
                ("SHOTAWARE_SATGATE", "50-shot clean-balanced candidate with gated satellite view for strict satellite robustness", shotaware_params(shots=50, adv=0.26, dom=0.55, norm=0.00004, group_ce=0.030, proto=0.0040, supcon=0.0040, fishr=0.0005, mix_p=0.10, mix_strength=0.32, sat_weight=0.24)),
            ],
        ),
    ]

    candidates: List[Candidate] = []
    gpu = 0
    for shot_label, shots, rows in specs:
        for suffix, rationale, params in rows:
            candidate_id = f"{shot_label}_{suffix}"
            run_name = f"CEN51_RIEIFD_{shot_label}_{suffix}_seed2028"
            if "SATGATE" in suffix or "GATEDSAT" in suffix:
                gate = "best val within 1.5pp of clean-balanced and sat strict mean improves by >=3pp"
                strategy = "ratio-aware satellite gate"
            elif "SHOTAWARE" in suffix:
                gate = "best val and strict UDU improve same-shot LAC/R04, with no higher-shot regression in the promoted chain"
                strategy = "shot-aware relaxed regularization"
            else:
                gate = "best val >= 90 or best strict UDU improves LAC without rollback >3pp"
                strategy = "explicit nuisance deshortcut"
            if suffix.startswith("CE_FN"):
                gate = "best val >= LAC + 2pp and strict UDU not below LAC by >5pp"
                strategy = "source-val upper-bound guard"
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    run_name=run_name,
                    shots=shots,
                    gpu=gpu % 8,
                    strategy=strategy,
                    rationale=rationale,
                    success_gate=gate,
                    params=params,
                )
            )
            gpu += 1
    return candidates


def render_launcher(run_id: str, candidates: Sequence[Candidate]) -> str:
    base_args = arg_pairs(BASE_PARAMS)
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
        'MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-6}"',
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
        '  local candidate_id="$1"',
        '  local run_name="$2"',
        '  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]',
        "}",
        "",
        "declare -A LAUNCHED_BY_GPU=()",
        "",
        "run_candidate() {",
        '  local candidate_id="$1" run_name="$2" shots="$3" gpu="$4"',
        "  shift 4",
        '  local run_dir="${RUNS_ROOT}/${run_name}"',
        '  local log_path="${LOG_ROOT}/${run_name}.out"',
        '  local cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${BASE_ARGS[@]}" "$@"',
        '    --run_name "${run_name}"',
        '    --wisig_max_train_per_combo "${shots}"',
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
        '  echo "[CEN51-RIEI-FD] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        "  printf '[CEN51-RIEI-FD-CMD]'",
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
        '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${candidate_id}" "${run_name}" "${shots}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
        "}",
        "",
        "BASE_ARGS=(",
    ]
    lines.extend(f"  {shell_join([arg])}" for arg in base_args)
    lines.extend(
        [
            ")",
            "",
            'if [[ "${DRY_RUN}" != "1" ]]; then',
            '  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }',
            "fi",
            'cd "${ROOT}"',
            'echo "[CEN51-RIEI-FD] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
            "",
        ]
    )
    for cand in candidates:
        lines.append(
            "run_candidate "
            + shell_join([cand.candidate_id, cand.run_name, str(cand.shots), str(cand.gpu)])
            + " \\"
        )
        args = cand.args()
        for idx, arg in enumerate(args):
            cont = " \\" if idx < len(args) - 1 else ""
            lines.append(f"  {shlex.quote(arg)}{cont}")
        lines.append("")
    return "\n".join(lines) + "\n"


def payload(run_id: str, candidates: Sequence[Candidate]) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "objective": "RIEI-style shot-aware CEN51 CVS feature-disentanglement search for 5/10/20/30/50 shots.",
        "hypothesis": (
            "In 5/10-shot CVS, validation and strict UDU are limited by receiver/day/channel shortcut leakage, so "
            "CE plus explicit domain side branch, GRL, and feature-norm guard should improve low-shot generalization. "
            "As shots increase, the guard should relax and lightweight GroupCE/prototype/SupCon/MixStyle/satellite gates "
            "should be restored so that more samples yield higher clean and strict performance."
        ),
        "baselines": {
            "FS005_LAC": {"best_val": 77.77, "best_strict_udu": 70.35},
            "FS010_LAC": {"best_val": 88.87, "best_strict_udu": 74.63},
            "FS020_LAC": {"best_val": 93.37, "best_strict_udu": 76.75},
            "FS030_LAC": {"best_val": 96.13, "best_strict_udu": 79.79},
            "FS005_R04": {"best_val": 66.30, "best_strict_udu": 56.06},
            "FS010_R04": {"best_val": 77.93, "best_strict_udu": 64.96},
            "FS020_R04": {"best_val": 79.34, "best_strict_udu": 67.44},
            "FS030_R04": {"best_val": 82.56, "best_strict_udu": 71.39},
            "FS050_R04": {"best_val": 96.34, "best_strict_udu": 78.92},
        },
        "selection_score": {
            "formula": "0.40*best_val + 0.25*best_strict_udu + 0.15*receiver_floor + 0.10*sat_strict_mean + 0.10*shot_monotonic_bonus - rollback_penalty",
            "rollback": "max(0, best_val-final_val) + max(0, best_strict_udu-final_strict_udu)",
            "shot_monotonic_bonus": "promoted chain should be non-decreasing from 5->10->20->30->50 in best_val and should not drop strict UDU by >1pp at higher shots",
            "promotion_rule": "Promote only if the same-shot baseline is improved or matched while preserving the monotonic chain; 5/10-shot still require rollback <=3pp.",
        },
        "candidates": [
            {
                "candidate_id": cand.candidate_id,
                "run_name": cand.run_name,
                "shots": cand.shots,
                "gpu": cand.gpu,
                "strategy": cand.strategy,
                "rationale": cand.rationale,
                "success_gate": cand.success_gate,
                "params": cand.params,
                "args": cand.args(),
            }
            for cand in candidates
        ],
    }


def render_report(data: Dict[str, object], launcher_path: Path, matrix_path: Path) -> str:
    lines = [
        f"# CEN51 RIEI-style low-shot guard sweep - {data['run_id']}",
        "",
        "## Objective",
        "",
        str(data["objective"]),
        "",
        "## Hypothesis",
        "",
        str(data["hypothesis"]),
        "",
        "## Baseline evidence",
        "",
        "| shot | baseline | best val | best strict UDU |",
        "|---:|---|---:|---:|",
    ]
    baselines = data["baselines"]
    for name, metrics in baselines.items():
        shot = name.split("_", 1)[0].replace("FS", "")
        baseline = name.split("_", 1)[1]
        lines.append(f"| {int(shot)} | {baseline} | {metrics['best_val']:.2f} | {metrics['best_strict_udu']:.2f} |")
    lines.extend(
        [
            "",
            "## Candidate matrix",
            "",
            "| candidate | shots | gpu | strategy | success gate |",
            "|---|---:|---:|---|---|",
        ]
    )
    for cand in data["candidates"]:
        lines.append(
            f"| `{cand['candidate_id']}` | {cand['shots']} | {cand['gpu']} | {cand['strategy']} | {cand['success_gate']} |"
        )
    score = data["selection_score"]
    lines.extend(
        [
            "",
            "## Selection score",
            "",
            f"- formula: `{score['formula']}`",
            f"- rollback: `{score['rollback']}`",
            f"- promotion rule: {score['promotion_rule']}",
            "",
            "## Local artifacts",
            "",
            f"- matrix: `{matrix_path}`",
            f"- launcher: `{launcher_path}`",
            "",
            "## Verification plan",
            "",
            "- local: `conda activate ssr-gpu; python -m py_compile code/train.py code/cvsrffi/losses.py code/cvsrffi/logging.py tools/cen51_riei_lowshot_guard_matrix.py tools/cen51_lowshot_config_search.py`",
            "- local: `conda activate ssr-gpu; python -m pytest tests/test_feature_norm_guard.py tests/test_cen51_lowshot_cli_overrides.py -q`",
            "- launcher: `bash -n <launcher>` and `bash <launcher> --dry-run`",
            "",
            "## N607 launch plan",
            "",
            "- Run AGENTS preflight first.",
            "- Sync local-first changes to `/home/szu2070436088/2510044040/CV-SincNet`.",
            "- Remote-verify imports and dry-run before launch.",
            "- Launcher default `MAX_TRAIN_PER_GPU=5`; capacity gate skips candidates if an occupied GPU is already at or above that count.",
            "- Startup-health after 4-5 minutes must inspect `[CONFIG-LOSS]`, `[CONFIG-SPLIT]`, `[LOSS-DG-RAW]`, process table, `launch_pids.tsv`, and traceback/OOM markers.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=Path("automation_reports") / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=Path("code") / "scripts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_riei_fd_guard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidates = make_candidates()

    report_dir = args.output_root / run_id
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = artifact_dir / "cen51_riei_fd_guard_matrix.json"
    launcher_path = args.scripts_dir / f"launch_{run_id}.sh"
    report_path = report_dir / "report.md"

    data = payload(run_id, candidates)
    matrix_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    launcher_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    report_path.write_text(render_report(data, launcher_path, matrix_path), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": len(candidates),
                "matrix": str(matrix_path),
                "launcher": str(launcher_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
