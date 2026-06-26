#!/usr/bin/env python
"""Generate CEN51 low-shot adaptive search candidates.

The generator is intentionally evidence driven:

* It reads the completed CEN51_R04 few-shot summary.
* It classifies each shot level by rollback, plateau, and receiver floor.
* It renders a bounded screening matrix that keeps the CEN51 architecture but
  weakens the full-ratio regularization stack for low-shot regimes.

The output is a JSON matrix, a runnable bash launcher, and a short design report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_SUMMARY = (
    Path("automation_reports")
    / "CV-SincNet"
    / "fewshot_cen51_r04_20260608_194541"
    / "full_log_analysis"
    / "fewshot_curve_summary.csv"
)

SAT_SCENARIOS = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"


def as_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def fmt_value(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean values should be rendered as standalone flags")
    if isinstance(value, float):
        if value == 0:
            return "0.0"
        if abs(value) < 0.01:
            return f"{value:.4g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def arg_pairs(params: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, fmt_value(value)])
    return args


def shell_join(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in args)


@dataclass
class ShotEvidence:
    shots: int
    train_size: int
    val_size: int
    best_val: float
    best_val_epoch: int
    final_val: float
    val_drop: float
    best_udu: float
    best_udu_epoch: int
    final_udu: float
    receiver_floor: float
    receiver_floor_name: str

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "ShotEvidence":
        return cls(
            shots=as_int(row.get("shots_per_combo")),
            train_size=as_int(row.get("train_size")),
            val_size=as_int(row.get("val_size")),
            best_val=as_float(row.get("best_val_tx_by_curve")),
            best_val_epoch=as_int(row.get("best_val_epoch_by_curve")),
            final_val=as_float(row.get("final_val_tx")),
            val_drop=as_float(row.get("best_val_drop_to_final")),
            best_udu=as_float(row.get("best_udu_by_curve")),
            best_udu_epoch=as_int(row.get("best_udu_epoch_by_curve")),
            final_udu=as_float(row.get("final_test_unseen_day_unseen_rx")),
            receiver_floor=as_float(row.get("final_best_unseen_rx_min")),
            receiver_floor_name=row.get("final_best_unseen_rx_min_rx", ""),
        )

    @property
    def regime(self) -> str:
        if self.shots <= 10 and self.val_drop >= 5:
            return "rollback_lowshot"
        if self.shots <= 20 and self.receiver_floor < 55:
            return "receiver_floor_limited"
        if self.val_drop <= 0.5:
            return "plateau"
        return "mixed"


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
    "wisig_domain": "rx_day",
    "wisig_train_ratio": 0.1,
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
    "seed": 1337,
    "use_mixstyle": True,
    "mixstyle_layers": "time_down,t1",
    "mixstyle_mix": "same_tx_crossdomain",
    "mixstyle_fallback": "skip",
    "mixstyle_late_ramp_epochs": 40,
    "domain_freq_stability_mode": "dsq",
    "freq_stability_channels": 2,
    "group_ce_mode": "smooth_dro_capped",
    "group_ce_min_domains": 4,
    "use_proto_memory": True,
    "proto_momentum": 0.97,
    "supcon_temp": 0.12,
    "fishr_min_domains": 4,
    "generalization_feature": "z_id",
    "pa_orders": "1,3,5",
    "collapse_guard": True,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 10.0,
    "collapse_guard_max_skipped_delta": 2,
    "use_ema_ckpt": True,
    "ema_decay": 0.999,
    "use_swad_ckpt": True,
}


def clean_lite_params(shots: int) -> Dict[str, object]:
    if shots <= 5:
        return {
            "batch_size": 128,
            "epochs": 170,
            "concat_sat_ce_weight": 0.35,
            "sat_view_prob": 0.35,
            "sat_view_schedule": f"1@0.35:{SAT_SCENARIOS}",
            "lambda_sat_cons": 0.0,
            "sat_cons_start_epoch": 999,
            "lambda_adv": 0.16,
            "lambda_cons": 0.03,
            "lambda_group_ce": 0.025,
            "group_ce_top_frac": 0.20,
            "groupdro_tau": 0.30,
            "groupdro_cap": 0.42,
            "lambda_proto": 0.004,
            "lambda_supcon_id": 0.004,
            "lambda_fishr": 0.0,
            "mixstyle_p": 0.10,
            "mixstyle_strength": 0.38,
            "mixstyle_late_start": 80,
            "mixstyle_late_min_p": 0.02,
            "mixstyle_late_min_strength": 0.18,
            "late_stable_start": 80,
            "late_stable_ramp_epochs": 25,
            "late_adv_min_scale": 0.35,
            "late_cons_min_scale": 0.20,
            "late_group_ce_min_scale": 0.35,
            "late_aug_min_scale": 0.45,
            "primary_udu_weight": 0.84,
            "swad_start_epoch": 50,
            "swad_tolerance": 0.8,
        }
    if shots <= 10:
        return {
            "batch_size": 128,
            "epochs": 175,
            "concat_sat_ce_weight": 0.45,
            "sat_view_prob": 0.45,
            "sat_view_schedule": f"1@0.45:{SAT_SCENARIOS}",
            "lambda_sat_cons": 0.0,
            "sat_cons_start_epoch": 999,
            "lambda_adv": 0.18,
            "lambda_cons": 0.04,
            "lambda_group_ce": 0.035,
            "group_ce_top_frac": 0.20,
            "groupdro_tau": 0.32,
            "groupdro_cap": 0.45,
            "lambda_proto": 0.006,
            "lambda_supcon_id": 0.006,
            "lambda_fishr": 0.0,
            "mixstyle_p": 0.12,
            "mixstyle_strength": 0.45,
            "mixstyle_late_start": 85,
            "mixstyle_late_min_p": 0.025,
            "mixstyle_late_min_strength": 0.20,
            "late_stable_start": 85,
            "late_stable_ramp_epochs": 25,
            "late_adv_min_scale": 0.38,
            "late_cons_min_scale": 0.22,
            "late_group_ce_min_scale": 0.38,
            "late_aug_min_scale": 0.48,
            "primary_udu_weight": 0.84,
            "swad_start_epoch": 60,
            "swad_tolerance": 0.85,
        }
    return {
        "batch_size": 256,
        "epochs": 200,
        "concat_sat_ce_weight": 0.65,
        "sat_view_prob": 0.60,
        "sat_view_schedule": f"1@0.55:{SAT_SCENARIOS};150@0.65:{SAT_SCENARIOS}",
        "lambda_sat_cons": 0.0,
        "sat_cons_start_epoch": 999,
        "lambda_adv": 0.25,
        "lambda_cons": 0.055,
        "lambda_group_ce": 0.050,
        "group_ce_top_frac": 0.22,
        "groupdro_tau": 0.35,
        "groupdro_cap": 0.48,
        "lambda_proto": 0.010,
        "lambda_supcon_id": 0.012,
        "lambda_fishr": 0.001,
        "mixstyle_p": 0.15,
        "mixstyle_strength": 0.55,
        "mixstyle_late_start": 100,
        "mixstyle_late_min_p": 0.035,
        "mixstyle_late_min_strength": 0.25,
        "late_stable_start": 100,
        "late_stable_ramp_epochs": 25,
        "late_adv_min_scale": 0.45,
        "late_cons_min_scale": 0.28,
        "late_group_ce_min_scale": 0.45,
        "late_aug_min_scale": 0.55,
        "primary_udu_weight": 0.84,
        "swad_start_epoch": 70,
        "swad_tolerance": 0.8,
    }


def delayed_sat_params(shots: int) -> Dict[str, object]:
    if shots <= 5:
        base = clean_lite_params(shots)
        base.update(
            {
                "epochs": 190,
                "concat_sat_ce_weight": 0.55,
                "sat_view_prob": 0.45,
                "sat_view_schedule": f"1@0.30:clear_leo,mixed_orbit;150@0.55:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0015,
                "sat_cons_start_epoch": 160,
                "lambda_adv": 0.20,
                "lambda_group_ce": 0.035,
                "lambda_proto": 0.006,
                "lambda_supcon_id": 0.006,
                "lambda_fishr": 0.0005,
                "mixstyle_p": 0.12,
                "mixstyle_strength": 0.42,
                "swad_start_epoch": 70,
            }
        )
        return base
    if shots <= 10:
        base = clean_lite_params(shots)
        base.update(
            {
                "epochs": 190,
                "concat_sat_ce_weight": 0.65,
                "sat_view_prob": 0.55,
                "sat_view_schedule": f"1@0.35:clear_leo,mixed_orbit;150@0.65:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.002,
                "sat_cons_start_epoch": 160,
                "lambda_adv": 0.22,
                "lambda_group_ce": 0.045,
                "lambda_proto": 0.008,
                "lambda_supcon_id": 0.008,
                "lambda_fishr": 0.0008,
                "mixstyle_p": 0.14,
                "mixstyle_strength": 0.50,
                "swad_start_epoch": 70,
            }
        )
        return base
    base = clean_lite_params(shots)
    base.update(
        {
            "concat_sat_ce_weight": 0.85,
            "sat_view_prob": 0.70,
            "sat_view_schedule": f"1@0.50:clear_leo,mixed_orbit;140@0.75:{SAT_SCENARIOS}",
            "use_sat_consistency": True,
            "lambda_sat_cons": 0.003,
            "sat_cons_start_epoch": 150,
            "lambda_adv": 0.30,
            "lambda_group_ce": 0.065,
            "lambda_proto": 0.012,
            "lambda_supcon_id": 0.015,
            "lambda_fishr": 0.0015,
            "mixstyle_p": 0.16,
            "mixstyle_strength": 0.60,
        }
    )
    return base


def rx_floor_params(shots: int) -> Dict[str, object]:
    base = clean_lite_params(shots)
    base.update(
        {
            "epochs": 185 if shots <= 10 else 200,
            "concat_sat_ce_weight": 0.50 if shots <= 10 else 0.75,
            "sat_view_prob": 0.40 if shots <= 10 else 0.60,
            "sat_view_schedule": (
                f"1@0.35:clear_leo,mixed_orbit;145@0.50:{SAT_SCENARIOS}"
                if shots <= 10
                else f"1@0.50:clear_leo,mixed_orbit;145@0.65:{SAT_SCENARIOS}"
            ),
            "lambda_sat_cons": 0.0,
            "sat_cons_start_epoch": 999,
            "lambda_adv": 0.24 if shots <= 10 else 0.30,
            "lambda_group_ce": 0.060 if shots <= 10 else 0.075,
            "group_ce_top_frac": 0.20,
            "groupdro_tau": 0.35,
            "groupdro_cap": 0.48 if shots <= 10 else 0.52,
            "lambda_proto": 0.010 if shots <= 10 else 0.014,
            "lambda_supcon_id": 0.012 if shots <= 10 else 0.016,
            "lambda_fishr": 0.001 if shots <= 10 else 0.002,
            "mixstyle_p": 0.13 if shots <= 10 else 0.14,
            "mixstyle_strength": 0.50 if shots <= 10 else 0.55,
            "primary_udu_weight": 0.90,
            "swad_start_epoch": 65 if shots <= 10 else 75,
        }
    )
    return base


def make_candidates(evidence: Dict[int, ShotEvidence], shots: Sequence[int]) -> List[Candidate]:
    candidates: List[Candidate] = []
    gpu = 0

    def add(shots_value: int, suffix: str, strategy: str, params: Dict[str, object], rationale: str, gate: str) -> None:
        nonlocal gpu
        candidate_id = f"CEN51_LS{len(candidates) + 1:02d}_FS{shots_value:03d}_{suffix}"
        run_name = f"{candidate_id}_r010"
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                run_name=run_name,
                shots=shots_value,
                gpu=gpu,
                strategy=strategy,
                rationale=rationale,
                success_gate=gate,
                params=params,
            )
        )
        gpu += 1

    if 5 in shots:
        ev = evidence.get(5)
        add(
            5,
            "clean_lite",
            "rollback_cut_sat_domain",
            clean_lite_params(5),
            f"FS005 rollback was {ev.val_drop:.2f} points; cut late satellite consistency and strong domain stats.",
            "beat FS005 primary strict UDU 56.06 and reduce best-to-final rollback below 3 points",
        )
        add(
            5,
            "delayed_sat",
            "late_sat_reentry",
            delayed_sat_params(5),
            "Keep a small delayed satellite path to test whether late re-entry preserves OOD without the E118 collapse.",
            "match clean-lite val while improving SAT strict UDU and not losing strict UDU",
        )

    if 10 in shots:
        ev = evidence.get(10)
        add(
            10,
            "clean_lite",
            "rollback_cut_sat_domain",
            clean_lite_params(10),
            f"FS010 peaked at E{ev.best_val_epoch} then dropped {ev.val_drop:.2f}; use lower satellite/domain pressure.",
            "beat FS010 primary strict UDU 64.96 and reduce final strict UDU drop",
        )
        add(
            10,
            "delayed_sat",
            "late_sat_reentry",
            delayed_sat_params(10),
            "Restore a weak late satellite consistency path only after the clean decision boundary stabilizes.",
            "preserve strict UDU >= 65 while improving SAT clear/rain floor",
        )
        add(
            10,
            "rx_floor",
            "receiver_floor_repair",
            rx_floor_params(10),
            "FS010 receiver floor was rx8=32.61; keep moderate group/proto pressure but no sat consistency.",
            "raise unseen receiver floor by at least 8 points without val rollback above 4",
        )

    if 20 in shots:
        ev = evidence.get(20)
        add(
            20,
            "clean_lite",
            "plateau_clean_stabilize",
            clean_lite_params(20),
            f"FS020 val was stable but strict UDU peaked early at E{ev.best_udu_epoch}; reduce late interference.",
            "beat FS020 primary strict UDU 67.44 and keep val near 79",
        )
        add(
            20,
            "delayed_sat",
            "plateau_late_sat",
            delayed_sat_params(20),
            "Use more satellite than clean-lite but still delay consistency until after E150.",
            "improve SAT floor without strict UDU regression",
        )
        add(
            20,
            "rx_floor",
            "receiver_floor_repair",
            rx_floor_params(20),
            "FS020 receiver floor was still rx8=51.81; test a receiver-floor-focused middle setting.",
            "raise receiver floor and strict UDU while keeping clean val within 1 point",
        )

    return candidates


def load_evidence(path: Path) -> Dict[int, ShotEvidence]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    evidence = {ShotEvidence.from_row(row).shots: ShotEvidence.from_row(row) for row in rows}
    missing = sorted({5, 10, 20} - set(evidence))
    if missing:
        raise SystemExit(f"missing shot evidence in {path}: {missing}")
    return evidence


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
        '  echo "[CEN51-LOWSHOT-SEARCH] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        "  printf '[CEN51-LOWSHOT-CMD]'",
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
            'echo "[CEN51-LOWSHOT-SEARCH] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
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
    return "\n".join(lines)


def candidate_payload(candidates: Sequence[Candidate], evidence: Dict[int, ShotEvidence]) -> Dict[str, object]:
    return {
        "objective": "Find CEN51 low-shot configurations that preserve clean validation while improving strict UDU and receiver floor.",
        "selection_score": {
            "formula": "0.45*best_primary_ood + 0.25*best_val + 0.20*best_receiver_floor + 0.10*sat_strict_mean - rollback_penalty",
            "rollback_penalty": "max(0, best_val-final_val-3)*1.5 + max(0, best_udu-final_udu-3)*1.5",
            "promotion_rule": "Promote top 2 candidates per shot to 3-seed verification only if rollback <= 3 and strict UDU improves the R04 baseline.",
        },
        "baseline_evidence": {
            str(k): {
                "train_size": v.train_size,
                "val_size": v.val_size,
                "best_val": v.best_val,
                "best_val_epoch": v.best_val_epoch,
                "final_val": v.final_val,
                "val_drop": v.val_drop,
                "best_udu": v.best_udu,
                "best_udu_epoch": v.best_udu_epoch,
                "final_udu": v.final_udu,
                "receiver_floor": v.receiver_floor,
                "receiver_floor_name": v.receiver_floor_name,
                "regime": v.regime,
            }
            for k, v in sorted(evidence.items())
        },
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "run_name": c.run_name,
                "shots": c.shots,
                "gpu": c.gpu,
                "strategy": c.strategy,
                "rationale": c.rationale,
                "success_gate": c.success_gate,
                "args": c.args(),
                "params": c.params,
            }
            for c in candidates
        ],
    }


def render_report(run_id: str, payload: Dict[str, object], launcher_path: Path, matrix_path: Path) -> str:
    lines = [
        f"# CEN51 low-shot adaptive search - {run_id}",
        "",
        "## Objective",
        "",
        str(payload["objective"]),
        "",
        "## Evidence rule",
        "",
        "- FS005/FS010 showed large best-to-final rollback, so the first search axis cuts satellite consistency, concat-sat CE, MixStyle, and domain-stat losses.",
        "- FS020 was mostly plateaued but strict UDU peaked early, so its search is a mild stabilization and receiver-floor repair pass.",
        "- FS030/FS050 are excluded from the first screen because they were already stable enough to serve as higher-shot references.",
        "",
        "## Selection score",
        "",
    ]
    score = payload["selection_score"]
    lines.extend(
        [
            f"- formula: `{score['formula']}`",
            f"- rollback penalty: `{score['rollback_penalty']}`",
            f"- promotion rule: {score['promotion_rule']}",
            "",
            "## Local artifacts",
            "",
            f"- matrix: `{matrix_path}`",
            f"- launcher: `{launcher_path}`",
            "",
            "## Candidate matrix",
            "",
            "| candidate | shots | gpu | strategy | success gate |",
            "|---|---:|---:|---|---|",
        ]
    )
    for cand in payload["candidates"]:
        lines.append(
            f"| `{cand['candidate_id']}` | {cand['shots']} | {cand['gpu']} | {cand['strategy']} | {cand['success_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- run `bash -n <launcher>`",
            "- on this Windows host, run `bash -lc 'ROOT=/mnt/e/type10-7 bash /mnt/e/type10-7/<launcher> --dry-run'`",
            "- run `conda activate ssr-gpu; python -m py_compile tools/cen51_lowshot_config_search.py code/train.py`",
            "",
            "## Launch note",
            "",
            "This report only defines the search algorithm and launcher. Sync/launch still requires the N607 preflight, local verification, SCP, remote verification, and startup-health pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--shots", default="5,10,20")
    parser.add_argument("--output-root", type=Path, default=Path("automation_reports") / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=Path("code") / "scripts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_lowshot_adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shots = [int(part.strip()) for part in args.shots.split(",") if part.strip()]
    evidence = load_evidence(args.summary_csv)
    candidates = make_candidates(evidence, shots)
    if not candidates:
        raise SystemExit("no candidates generated")

    report_dir = args.output_root / run_id
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = artifact_dir / "cen51_lowshot_matrix.json"
    launcher_path = args.scripts_dir / f"launch_{run_id}.sh"
    report_path = report_dir / "report.md"

    payload = candidate_payload(candidates, evidence)
    matrix_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    launcher_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    report_path.write_text(render_report(run_id, payload, launcher_path, matrix_path), encoding="utf-8")

    print(json.dumps({"run_id": run_id, "candidates": len(candidates), "matrix": str(matrix_path), "launcher": str(launcher_path), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
