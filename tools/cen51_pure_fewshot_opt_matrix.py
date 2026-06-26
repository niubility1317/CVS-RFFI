#!/usr/bin/env python3
"""Generate pure per-class few-shot CEN51 optimization experiments.

This matrix is intentionally different from earlier ``max_train_per_combo``
low-shot sweeps.  Each experiment uses ``--wisig_train_shots_per_class K`` so
the final train set is K samples per transmitter class in total, across all
source receiver/day domains.
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


DEFAULT_RUN_ID = "cen51_pure_fewshot_opt_20260610_000000"
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


def batch_for_shot(shots: int) -> int:
    if int(shots) <= 10:
        return 16
    if int(shots) <= 30:
        return 32
    return 64


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    run_name: str
    shots: int
    gpu: int
    strategy: str
    sampler: str
    rationale: str
    success_gate: str
    params: Dict[str, object] = field(default_factory=dict)

    def args(self) -> List[str]:
        merged = dict(self.params)
        merged["batch_size"] = batch_for_shot(self.shots)
        merged["wisig_train_shot_strategy"] = self.sampler
        return arg_pairs(merged)


BASE_PARAMS: Dict[str, object] = {
    "train_mode": "centralized",
    "eval_batch_size": 256,
    "train_drop_last": False,
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
}


def r04_params(shots: int) -> Dict[str, object]:
    return {
        "epochs": 200,
        "use_aug": True,
        "use_concat_sat_channel_aug": True,
        "concat_sat_start_epoch": 1,
        "lambda_sat_cls": 0.0,
        "lambda_sat_cons": 0.006,
        "sat_cons_start_epoch": 118,
        "use_sat_consistency": True,
        "sat_train_scenarios": SAT_SCENARIOS,
        "sat_view_prob": 1.0,
        "concat_sat_ce_weight": 1.19,
        "sat_view_schedule": "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit",
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
        "domain_freq_stability_mode": "dsq",
        "freq_stability_channels": 2,
        "lambda_group_ce": 0.088,
        "group_ce_mode": "smooth_dro_capped",
        "group_ce_min_domains": 4,
        "group_ce_top_frac": 0.20,
        "groupdro_tau": 0.37,
        "groupdro_cap": 0.48,
        "use_proto_memory": True,
        "lambda_proto": 0.016,
        "proto_momentum": 0.970,
        "lambda_supcon_id": 0.022,
        "supcon_temp": 0.12,
        "lambda_fishr": 0.002,
        "fishr_min_domains": 4,
        "generalization_feature": "z_id",
        "swad_start_epoch": 70,
        "swad_tolerance": 0.34,
        "pa_orders": "1,3,5",
    }


def lowshot_rxgrl_params(*, shots: int, adv: float, norm: float, sat_weight: float = 0.0) -> Dict[str, object]:
    params: Dict[str, object] = {
        "epochs": 220 if shots <= 5 else 210,
        "swad_start_epoch": 60 if shots <= 5 else 70,
        "swad_tolerance": 0.80,
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
        "lambda_dom": 0.35,
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


def shotaware_params(
    *,
    shots: int,
    adv: float,
    norm: float,
    group_ce: float,
    proto: float,
    supcon: float,
    fishr: float,
    mix_p: float,
    mix_strength: float,
    sat_weight: float = 0.0,
) -> Dict[str, object]:
    epochs = {20: 210, 30: 220, 50: 230}.get(int(shots), 220)
    params: Dict[str, object] = {
        "epochs": epochs,
        "swad_start_epoch": {20: 80, 30: 90, 50: 100}.get(int(shots), 90),
        "swad_tolerance": 0.65,
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
        "mixstyle_late_start": 125 if shots <= 20 else (135 if shots <= 30 else 145),
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
        "lambda_dom": 0.45 if shots <= 20 else (0.50 if shots <= 30 else 0.55),
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
                "sat_view_schedule": f"1@{sat_weight:.2f}:clear_leo,mixed_orbit;150@{max(0.05, sat_weight * 0.60):.2f}:clear_leo,mixed_orbit",
            }
        )
    return params


def make_candidates() -> List[Candidate]:
    specs = [
        (
            "FS005",
            5,
            [
                ("R04_RANDOMSEL", "random", "pure R04 control with random per-class K", r04_params(5)),
                ("R04_DOMBAL", "domain_balanced", "pure R04 control with domain-balanced per-class K", r04_params(5)),
                ("RXGRL_HINGE", "domain_balanced", "RIEI-style receiver/day bottleneck, no satellite", lowshot_rxgrl_params(shots=5, adv=0.12, norm=0.0010)),
                ("RXGRL_WEAKSAT", "domain_balanced", "RIEI-style bottleneck with weak clean/mixed satellite view", lowshot_rxgrl_params(shots=5, adv=0.10, norm=0.0008, sat_weight=0.10)),
            ],
        ),
        (
            "FS010",
            10,
            [
                ("R04_RANDOMSEL", "random", "pure R04 control with random per-class K", r04_params(10)),
                ("R04_DOMBAL", "domain_balanced", "pure R04 control with domain-balanced per-class K", r04_params(10)),
                ("RXGRL_HINGE", "domain_balanced", "RIEI-style receiver/day bottleneck, no satellite", lowshot_rxgrl_params(shots=10, adv=0.18, norm=0.0007)),
                ("RXGRL_WEAKSAT", "domain_balanced", "RIEI-style bottleneck with weak clean/mixed satellite view", lowshot_rxgrl_params(shots=10, adv=0.16, norm=0.0006, sat_weight=0.12)),
            ],
        ),
        (
            "FS020",
            20,
            [
                ("R04_DOMBAL", "domain_balanced", "pure R04 domain-balanced control", r04_params(20)),
                ("SHOTAWARE_CLEAN", "domain_balanced", "relaxed clean DG as K grows", shotaware_params(shots=20, adv=0.20, norm=0.00010, group_ce=0.012, proto=0.0015, supcon=0.0015, fishr=0.0, mix_p=0.04, mix_strength=0.18)),
                ("SHOTAWARE_SATGATE", "domain_balanced", "same as clean plus weak gated satellite", shotaware_params(shots=20, adv=0.18, norm=0.00008, group_ce=0.010, proto=0.0015, supcon=0.0015, fishr=0.0, mix_p=0.04, mix_strength=0.18, sat_weight=0.14)),
            ],
        ),
        (
            "FS030",
            30,
            [
                ("R04_DOMBAL", "domain_balanced", "pure R04 domain-balanced control", r04_params(30)),
                ("SHOTAWARE_CLEAN", "domain_balanced", "relaxed clean DG as K grows", shotaware_params(shots=30, adv=0.24, norm=0.00008, group_ce=0.020, proto=0.0025, supcon=0.0025, fishr=0.0003, mix_p=0.06, mix_strength=0.24)),
                ("SHOTAWARE_SATGATE", "domain_balanced", "same as clean plus weak gated satellite", shotaware_params(shots=30, adv=0.22, norm=0.00006, group_ce=0.018, proto=0.0025, supcon=0.0025, fishr=0.0003, mix_p=0.06, mix_strength=0.24, sat_weight=0.18)),
            ],
        ),
        (
            "FS050",
            50,
            [
                ("R04_DOMBAL", "domain_balanced", "pure R04 domain-balanced control", r04_params(50)),
                ("SHOTAWARE_SATGATE", "domain_balanced", "higher-shot relaxed DG plus gated satellite", shotaware_params(shots=50, adv=0.26, norm=0.00004, group_ce=0.028, proto=0.0040, supcon=0.0040, fishr=0.0005, mix_p=0.10, mix_strength=0.32, sat_weight=0.24)),
            ],
        ),
    ]

    candidates: List[Candidate] = []
    gpu = 0
    for shot_label, shots, rows in specs:
        for suffix, sampler, rationale, params in rows:
            candidate_id = f"{shot_label}_{suffix}"
            run_name = f"CEN51_PFS_{shot_label}_{suffix}_seed2028"
            if suffix.endswith("RANDOMSEL"):
                strategy = "sampler_ablation_random"
                success_gate = "domain-balanced R04 should exceed this on val/strict UDU or be statistically equivalent"
            elif suffix.startswith("R04"):
                strategy = "pure_r04_control"
                success_gate = "optimized same-shot candidates must exceed this without sacrificing strict UDU"
            elif "RXGRL" in suffix:
                strategy = "riei_bottleneck_lowshot"
                success_gate = "target low-shot val >=90 for K10 and largest strict UDU gain for K5/K10"
            else:
                strategy = "shotaware_relaxed_dg"
                success_gate = "more samples should improve val and not reduce strict UDU versus lower-shot promoted config"
            candidates.append(
                Candidate(
                    candidate_id=candidate_id,
                    run_name=run_name,
                    shots=int(shots),
                    gpu=gpu % 8,
                    strategy=strategy,
                    sampler=sampler,
                    rationale=rationale,
                    success_gate=success_gate,
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
        '    --wisig_train_shots_per_class "${shots}"',
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
        '  echo "[CEN51-PURE-FS] candidate=${candidate_id} run=${run_name} shots_per_class=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        "  printf '[CEN51-PURE-FS-CMD]'",
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
            'echo "[CEN51-PURE-FS] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
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


def write_outputs(run_id: str, out_dir: Path) -> None:
    candidates = make_candidates()
    script_dir = Path("code/scripts")
    script_dir.mkdir(parents=True, exist_ok=True)
    launcher = script_dir / f"launch_{run_id}.sh"
    launcher.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")

    report_dir = Path("automation_reports/CV-SincNet") / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "candidate_id": c.candidate_id,
            "run_name": c.run_name,
            "shots_per_class": c.shots,
            "gpu": c.gpu,
            "strategy": c.strategy,
            "sampler": c.sampler,
            "batch_size": batch_for_shot(c.shots),
            "rationale": c.rationale,
            "success_gate": c.success_gate,
        }
        for c in candidates
    ]
    for path in [out_dir / "matrix.csv", report_dir / "matrix.csv"]:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Optimize CVS/CEN51 under strict pure few-shot training with K total samples per TX class.",
        "pure_fewshot_contract": {
            "train_cap": "--wisig_train_shots_per_class K",
            "not_used_as_shot_definition": "--wisig_max_train_per_combo",
            "candidate_pool_ratio": BASE_PARAMS["wisig_train_ratio"],
            "train_drop_last": False,
            "primary_sampler": "domain_balanced",
            "control_sampler": "random for FS005/FS010 R04_RANDOMSEL",
        },
        "hypotheses": [
            "If domain-balanced per-class sampling beats random per-class sampling, receiver/day coverage is a first-order few-shot variable.",
            "If RXGRL_HINGE beats R04_DOMBAL at FS005/FS010, low-shot CVS needs a RIEI-like bottleneck rather than full R04 satellite/DG pressure.",
            "If SHOTAWARE candidates beat R04_DOMBAL at 20/30/50 shots, regularization must relax as samples increase instead of keeping one fixed R04 recipe.",
            "Weak satellite gates should help satellite strict metrics only when they do not depress clean validation and strict UDU.",
        ],
        "metrics_to_watch": [
            "best_val_tx",
            "best_test_unseen_day_unseen_rx",
            "best_primary_ood",
            "best_worst_rx",
            "satellite strict mean and storm_mp floor",
            "split_info.train_size equals num_tx * shots_per_class",
            "LOSS-TOP balance and feature_norm_guard trend",
        ],
        "candidates": rows,
        "local_files": {
            "launcher": str(launcher),
            "matrix_csv": str(report_dir / "matrix.csv"),
            "report": str(report_dir / "report.md"),
        },
        "verification_commands": [
            "conda activate ssr-gpu; $env:PYTHONPATH='E:\\\\type10-7\\\\code'; python -m pytest tests/test_wisig_random_split.py -q",
            "conda activate ssr-gpu; $env:PYTHONPATH='E:\\\\type10-7\\\\code'; python -m py_compile code/dataset_wisig.py code/train.py baselines/common/cvs_data.py tools/cen51_pure_fewshot_opt_matrix.py",
            f"conda activate ssr-gpu; python tools/cen51_pure_fewshot_opt_matrix.py --run-id {run_id}",
        ],
    }
    for path in [out_dir / "manifest.json", report_dir / "manifest.json"]:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# CVS/CEN51 Pure Few-Shot Optimization - {run_id}",
        "",
        "## Objective",
        "Use strict pure few-shot training for CVS: K total train samples per TX class, not K per TX x receiver x day combo.",
        "",
        "## Why This Batch Exists",
        "Earlier low-shot launchers used `--wisig_max_train_per_combo`, so FS005 was still 5 samples for every TX/receiver/day combo. This batch fixes that protocol mismatch with `--wisig_train_shots_per_class` and disables train `drop_last` so every selected low-shot sample participates in training.",
        "",
        "## Experiment Logic",
        "- `R04_RANDOMSEL` vs `R04_DOMBAL` isolates sample selection under the same pure R04 recipe.",
        "- `RXGRL_HINGE` and `RXGRL_WEAKSAT` test RIEI-style receiver/day/channel shortcut suppression for 5/10 shots.",
        "- `SHOTAWARE_CLEAN` and `SHOTAWARE_SATGATE` test whether regularization can relax as shots increase while preserving satellite robustness.",
        "",
        "## Pure Few-Shot Contract",
        f"- `--wisig_train_ratio {BASE_PARAMS['wisig_train_ratio']}` only defines the train/validation candidate split; final training size is controlled by `--wisig_train_shots_per_class K`.",
        "- `--wisig_max_train_per_combo 0` is kept explicit so old per-combo shot semantics are not used.",
        "- `--no_train_drop_last` is used because K5 has only 5 samples per TX class.",
        "- Expected startup log check: `split_info.train_size == num_tx * K` and `split_info.max_samples_per_class_train == K`.",
        "",
        "## Candidates",
        "| candidate | shots/class | gpu | sampler | strategy | batch | rationale |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for c in candidates:
        lines.append(
            f"| `{c.candidate_id}` | {c.shots} | {c.gpu} | {c.sampler} | {c.strategy} | {batch_for_shot(c.shots)} | {c.rationale} |"
        )
    lines.extend(
        [
            "",
            "## Local Verification",
            "- `conda activate ssr-gpu; $env:PYTHONPATH='E:\\type10-7\\code'; python -m pytest tests/test_wisig_random_split.py -q`",
            "- `conda activate ssr-gpu; $env:PYTHONPATH='E:\\type10-7\\code'; python -m py_compile code/dataset_wisig.py code/train.py baselines/common/cvs_data.py tools/cen51_pure_fewshot_opt_matrix.py`",
            "",
            "## Launch",
            f"- Local launcher: `{launcher}`",
            "- Remote command after sync: `bash code/scripts/launch_" + run_id + ".sh`",
            "- Default launcher capacity: at most 2 training processes per GPU.",
            "",
            "## Completion Analysis Checklist",
            "- Confirm each log reports pure few-shot split sizes.",
            "- Rank by same-shot `best_val`, strict UDU, worst receiver, and satellite strict/storm_mp floor.",
            "- Promote only configs that beat pure R04 controls under the same sampler/shot protocol.",
            "- If `R04_DOMBAL` beats `R04_RANDOMSEL`, repeat winning low-shot configs across seeds before claiming stability.",
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
    print(f"[PURE-FEWSHOT] run_id={run_id}")
    print(f"[PURE-FEWSHOT] launcher=code/scripts/launch_{run_id}.sh")
    print(f"[PURE-FEWSHOT] report=automation_reports/CV-SincNet/{run_id}/report.md")
    print(f"[PURE-FEWSHOT] candidates={len(make_candidates())}")


if __name__ == "__main__":
    main()
