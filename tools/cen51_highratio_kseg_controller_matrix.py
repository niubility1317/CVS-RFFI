#!/usr/bin/env python3
"""Generate high-ratio large-K validation for the K-segmented controller.

The experiment tests whether domain-metric control should segment by the
per-combo effective sample count K_eff, by total effective samples N_eff, or by
a ratio/global rule once WiSig train_ratio is 0.2/0.3+.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
REMOTE_PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
RUN_PREFIX = "CEN51_HRKSEG"
SAT_SCENARIOS = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LIGHT_SAT = "clear_leo,mixed_orbit"
TRAIN_DOMAIN_COMBOS = 84


@dataclass(frozen=True)
class GridPoint:
    ratio: float
    k_cap: int
    cap_fraction: float

    @property
    def ratio_tag(self) -> str:
        return f"r{int(round(self.ratio * 100)):03d}"

    @property
    def k_tag(self) -> str:
        return f"k{self.k_cap:03d}"

    @property
    def n_eff_nominal(self) -> int:
        return self.k_cap * TRAIN_DOMAIN_COMBOS


@dataclass(frozen=True)
class Strategy:
    name: str
    axis: str
    action: str
    hypothesis: str
    success_gate: str


@dataclass(frozen=True)
class Candidate:
    cid: str
    run_name: str
    ratio: float
    ratio_tag: str
    k_cap: int
    n_eff_nominal: int
    cap_fraction: float
    strategy: str
    axis: str
    action: str
    gpu: int
    seed: int
    hypothesis: str
    success_gate: str
    params: dict[str, object]


GRID: Sequence[GridPoint] = (
    GridPoint(0.2, 100, 0.50),
    GridPoint(0.2, 150, 0.75),
    GridPoint(0.2, 200, 1.00),
    GridPoint(0.3, 150, 0.50),
    GridPoint(0.3, 225, 0.75),
    GridPoint(0.3, 300, 1.00),
    GridPoint(0.5, 250, 0.50),
    GridPoint(0.5, 375, 0.75),
    GridPoint(0.5, 500, 1.00),
)

STRATEGIES: Sequence[Strategy] = (
    Strategy(
        "KONLY_B03",
        "k_only_replay",
        "k100_b03_guard_replay",
        "Replay the successful K100 segmented guard regardless of train_ratio and larger K_eff.",
        "Valid if strict/primary remains competitive across ratios; fail if higher ratio requires re-scaling.",
    ),
    Strategy(
        "NEFF_RXSAT",
        "neff_segmented",
        "effective_sample_scaled_rx_sat",
        "Scale receiver/group and satellite guard with K_eff while preserving strict as the primary constraint.",
        "Should beat KONLY when K_eff grows and should avoid the over-global strict loss.",
    ),
    Strategy(
        "RATIO_STRICT",
        "ratio_segmented",
        "high_ratio_strict_guard",
        "Use extra ratio mainly to protect identity and lower rollback rather than increasing all invariance losses.",
        "Valid if strict/overall improve while floors stay acceptable.",
    ),
    Strategy(
        "TOTAL_OVER",
        "total_sample_control",
        "n_total_overdrive_negative",
        "Negative control: if the controller only sees N_eff, it will over-increase domain/sat pressure.",
        "Reject when floors rise but strict/primary or rollback gets worse.",
    ),
)


def q(value: object) -> str:
    return shlex.quote(str(value))


def fmt_value(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean flags are rendered separately")
    if isinstance(value, float):
        if value == 0:
            return "0.0"
        if abs(value) < 0.01:
            return f"{value:.4g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def arg_pairs(params: dict[str, object]) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, fmt_value(value)])
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    parser.add_argument("--max-active-per-gpu", type=int, default=2)
    parser.add_argument("--scheduler-hours", type=float, default=18.0)
    return parser.parse_args()


def base_params(point: GridPoint, seed: int) -> dict[str, object]:
    return {
        "train_mode": "centralized",
        "eval_batch_size": 256,
        "dataset": "wisig",
        "wisig_protocol": "cvs_day_rx",
        "wisig_domain": "rx_day",
        "wisig_equalized": 1,
        "wisig_train_ratio": point.ratio,
        "wisig_val_ratio": -1.0,
        "wisig_split_strategy": "random",
        "wisig_cap_strategy": "random",
        "wisig_train_days": "1,2",
        "wisig_test_days": "0,3",
        "wisig_train_rxs": "2,3,4,5,8,9,10",
        "wisig_test_rxs": "0,1,6,7,11",
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
        "domain_freq_stability_mode": "dsq",
        "freq_stability_channels": 2,
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
        "swad_tolerance": 0.65,
        "primary_udu_weight": 0.86,
        "label_smoothing": 0.0,
        "batch_size": 256,
        "epochs": 180,
        "seed": seed,
        "sat_view_seed": 9000 + seed % 1000 + int(point.ratio * 100) + point.k_cap,
        "no_enable_pa_aux": True,
        "no_enable_dac_aux": True,
        "no_aug_enable_pa_normal": True,
        "aug_p_pa": 0.0,
        "aug_p_dac": 0.0,
        "lambda_cls_pa": 0.0,
        "lambda_pa_joint_inv": 0.0,
        "lambda_pa_kl": 0.0,
        "lambda_pa_reg": 0.0,
        "use_aug": True,
        "use_mixstyle": True,
        "mixstyle_layers": "time_down,t1",
        "mixstyle_mix": "same_tx_crossdomain",
        "mixstyle_fallback": "skip",
        "mixstyle_late_ramp_epochs": 40,
        "group_ce_mode": "smooth_dro_capped",
        "group_ce_min_domains": 4,
        "use_proto_memory": True,
        "proto_momentum": 0.97,
        "supcon_temp": 0.12,
        "fishr_min_domains": 4,
        "generalization_feature": "z_id",
        "use_concat_sat_channel_aug": True,
        "concat_sat_start_epoch": 1,
        "lambda_sat_cls": 0.0,
        "use_sat_consistency": True,
        "lambda_sat_cons": 0.004,
        "sat_cons_start_epoch": 125,
        "late_adv_min_scale": 0.70,
        "late_cons_min_scale": 0.45,
        "late_group_ce_min_scale": 0.80,
        "late_aug_min_scale": 0.35,
        "aug_scale_min": 0.050,
        "aug_scale_max": 0.240,
    }


def strategy_updates(point: GridPoint, strategy: Strategy) -> dict[str, object]:
    ratio_boost = {0.2: 0.00, 0.3: 0.02, 0.5: 0.04}[point.ratio]
    cap_boost = (point.cap_fraction - 0.50) * 0.06

    if strategy.name == "KONLY_B03":
        return {
            "lambda_adv": 0.30,
            "lambda_cons": 0.055,
            "lambda_group_ce": 0.060,
            "group_ce_top_frac": 0.28,
            "groupdro_tau": 0.50,
            "groupdro_cap": 0.58,
            "lambda_proto": 0.009,
            "lambda_supcon_id": 0.011,
            "lambda_fishr": 0.0010,
            "sat_view_prob": 0.58,
            "concat_sat_ce_weight": 0.72,
            "sat_view_schedule": f"1@0.42:{LIGHT_SAT};120@0.60:{SAT_SCENARIOS}",
            "mixstyle_p": 0.16,
            "mixstyle_strength": 0.58,
            "mixstyle_late_start": 105,
            "mixstyle_late_min_p": 0.045,
            "mixstyle_late_min_strength": 0.30,
            "late_stable_start": 105,
            "late_stable_ramp_epochs": 25,
            "swad_start_epoch": 75,
        }

    if strategy.name == "NEFF_RXSAT":
        adv = 0.32 + ratio_boost + cap_boost
        group = 0.070 + ratio_boost * 0.40 + cap_boost * 0.60
        cap = min(0.66, 0.60 + cap_boost + ratio_boost * 0.60)
        sat_p = max(0.50, 0.57 - ratio_boost * 0.8 + cap_boost * 0.5)
        sat_ce = 0.70 + cap_boost * 1.2 + ratio_boost
        return {
            "lambda_adv": round(adv, 4),
            "lambda_cons": 0.060,
            "lambda_group_ce": round(group, 4),
            "group_ce_top_frac": min(0.34, 0.28 + cap_boost + ratio_boost),
            "groupdro_tau": 0.50,
            "groupdro_cap": round(cap, 4),
            "lambda_proto": 0.010,
            "lambda_supcon_id": 0.012,
            "lambda_fishr": 0.0012,
            "sat_view_prob": round(sat_p, 4),
            "concat_sat_ce_weight": round(sat_ce, 4),
            "sat_view_schedule": f"1@0.38:{LIGHT_SAT};110@{round(min(0.72, sat_p + 0.08), 3)}:{SAT_SCENARIOS}",
            "mixstyle_p": 0.18,
            "mixstyle_strength": 0.62,
            "mixstyle_late_start": 100,
            "mixstyle_late_min_p": 0.050,
            "mixstyle_late_min_strength": 0.32,
            "late_stable_start": 100,
            "late_stable_ramp_epochs": 25,
            "swad_start_epoch": 75,
        }

    if strategy.name == "RATIO_STRICT":
        return {
            "lambda_adv": 0.28 + ratio_boost * 0.5,
            "lambda_cons": 0.050,
            "lambda_group_ce": 0.052 + ratio_boost * 0.35,
            "group_ce_top_frac": 0.26 + ratio_boost,
            "groupdro_tau": 0.48,
            "groupdro_cap": 0.56 + ratio_boost,
            "lambda_proto": 0.008,
            "lambda_supcon_id": 0.010,
            "lambda_fishr": 0.0008,
            "sat_view_prob": 0.44 + ratio_boost,
            "concat_sat_ce_weight": 0.58 + ratio_boost * 1.5,
            "sat_view_schedule": f"1@0.34:{LIGHT_SAT};130@{round(0.50 + ratio_boost, 3)}:{SAT_SCENARIOS}",
            "mixstyle_p": 0.14,
            "mixstyle_strength": 0.50,
            "mixstyle_late_start": 110,
            "mixstyle_late_min_p": 0.040,
            "mixstyle_late_min_strength": 0.28,
            "late_stable_start": 110,
            "late_stable_ramp_epochs": 25,
            "swad_start_epoch": 70,
        }

    if strategy.name == "TOTAL_OVER":
        return {
            "lambda_adv": 0.43 + ratio_boost,
            "lambda_cons": 0.080,
            "lambda_group_ce": 0.120 + cap_boost,
            "group_ce_top_frac": 0.35,
            "groupdro_tau": 0.55,
            "groupdro_cap": 0.68,
            "lambda_proto": 0.016,
            "lambda_supcon_id": 0.020,
            "lambda_fishr": 0.0020,
            "sat_view_prob": 0.88,
            "concat_sat_ce_weight": 1.05,
            "sat_view_schedule": f"1@0.70:{LIGHT_SAT};90@0.90:{SAT_SCENARIOS}",
            "mixstyle_p": 0.22,
            "mixstyle_strength": 0.72,
            "mixstyle_late_start": 90,
            "mixstyle_late_min_p": 0.060,
            "mixstyle_late_min_strength": 0.38,
            "late_stable_start": 90,
            "late_stable_ramp_epochs": 25,
            "swad_start_epoch": 80,
        }

    raise KeyError(strategy.name)


def make_candidates() -> list[Candidate]:
    rows: list[Candidate] = []
    for point_index, point in enumerate(GRID):
        for strat_index, strategy in enumerate(STRATEGIES):
            seed = 1337
            params = base_params(point, seed)
            params.update(strategy_updates(point, strategy))
            gpu = (point_index * len(STRATEGIES) + strat_index) % 8
            cid = f"{point.ratio_tag.upper()}_{point.k_tag.upper()}_{strategy.name}"
            run_name = f"{RUN_PREFIX}_{cid}"
            rows.append(
                Candidate(
                    cid=cid,
                    run_name=run_name,
                    ratio=point.ratio,
                    ratio_tag=point.ratio_tag,
                    k_cap=point.k_cap,
                    n_eff_nominal=point.n_eff_nominal,
                    cap_fraction=point.cap_fraction,
                    strategy=strategy.name,
                    axis=strategy.axis,
                    action=strategy.action,
                    gpu=gpu,
                    seed=seed,
                    hypothesis=strategy.hypothesis,
                    success_gate=strategy.success_gate,
                    params=params,
                )
            )
    if len(rows) != 36:
        raise AssertionError(f"expected 36 candidates, got {len(rows)}")
    if len({row.cid for row in rows}) != len(rows):
        raise AssertionError("duplicate candidate ids")
    return rows


def command_for(run_id: str, candidate: Candidate) -> str:
    run_dir = f"{REMOTE_ROOT}/runs/{run_id}/{candidate.run_name}"
    args = [
        "env",
        f"CUDA_VISIBLE_DEVICES={candidate.gpu}",
        f"PYTHONPATH={REMOTE_ROOT}/code:{REMOTE_ROOT}/tools:{REMOTE_ROOT}",
        REMOTE_PYTHON,
        "-u",
        f"{REMOTE_ROOT}/code/train.py",
        *arg_pairs(candidate.params),
        "--run_name",
        candidate.run_name,
        "--wisig_max_train_per_combo",
        str(candidate.k_cap),
        "--latest_save_path",
        f"{run_dir}/latest_model.pth",
        "--best_save_path",
        f"{run_dir}/best_val_model.pth",
        "--best_primary_save_path",
        f"{run_dir}/best_primary_ood_model.pth",
        "--best_unseen_day_unseen_rx_save_path",
        f"{run_dir}/best_strict_udu_model.pth",
        "--best_worst_rx_save_path",
        f"{run_dir}/best_worst_rx_model.pth",
        "--ema_save_path",
        f"{run_dir}/ema_model.pth",
        "--swa_save_path",
        f"{run_dir}/swa_model.pth",
        "--swad_save_path",
        f"{run_dir}/swad_model.pth",
    ]
    return " ".join(q(item) for item in args)


def render_launcher(run_id: str, rows: Sequence[Candidate], max_active_per_gpu: int, scheduler_hours: float) -> str:
    max_seconds = int(scheduler_hours * 3600)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="${{ROOT:-{REMOTE_ROOT}}}"',
        f'RUN_ID="${{RUN_ID:-{run_id}}}"',
        'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
        'RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"',
        f'MAX_ACTIVE_PER_GPU="${{MAX_ACTIVE_PER_GPU:-{max_active_per_gpu}}}"',
        f'MAX_SCHEDULER_SECONDS="${{MAX_SCHEDULER_SECONDS:-{max_seconds}}}"',
        'POLL_SECONDS="${POLL_SECONDS:-45}"',
        'DRY_RUN="${DRY_RUN:-0}"',
        'ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"',
        "",
        'for arg in "$@"; do',
        '  case "${arg}" in',
        '    --dry-run) DRY_RUN=1 ;;',
        '    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;',
        '    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;',
        "  esac",
        "done",
        "",
        "gpu_process_count() {",
        '  local gpu="$1"',
        '  if [[ "${DRY_RUN}" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then echo 0; return 0; fi',
        "  local count",
        '  count="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed \'/^$/d\' | wc -l | tr -d \' \' || true)"',
        '  [[ "${count}" =~ ^[0-9]+$ ]] || count=0',
        '  echo "${count}"',
        "}",
        "",
        "CAND_ID=(); CAND_RUN=(); CAND_RATIO=(); CAND_K=(); CAND_GPU=(); CAND_AXIS=(); CAND_ACTION=(); CAND_STRATEGY=(); CAND_NOMINAL_N=(); CAND_CMD=()",
        "STATUS=(); PID=(); LOG_PATH=()",
        "",
        "add_candidate() {",
        '  CAND_ID+=("$1"); CAND_RUN+=("$2"); CAND_RATIO+=("$3"); CAND_K+=("$4"); CAND_GPU+=("$5"); CAND_AXIS+=("$6"); CAND_ACTION+=("$7"); CAND_STRATEGY+=("$8"); CAND_NOMINAL_N+=("$9"); CAND_CMD+=("${10}")',
        '  STATUS+=("queued"); PID+=(""); LOG_PATH+=("${LOG_ROOT}/$2.out")',
        "}",
        "",
    ]
    for row in rows:
        lines.append(
            "add_candidate "
            + " ".join(
                [
                    q(row.cid),
                    q(row.run_name),
                    q(f"{row.ratio:.1f}"),
                    q(row.k_cap),
                    q(row.gpu),
                    q(row.axis),
                    q(row.action),
                    q(row.strategy),
                    q(row.n_eff_nominal),
                    q(command_for(run_id, row)),
                ]
            )
        )
    lines.extend(
        [
            "",
            "should_skip() {",
            '  local candidate_id="$1" run_name="$2"',
            '  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]',
            "}",
            "",
            "defer_queued_after_window() {",
            "  local i",
            '  for i in "${!STATUS[@]}"; do',
            '    if [[ "${STATUS[$i]}" == "queued" ]]; then',
            '      STATUS[$i]="deferred_window"',
            '      printf "%s\\t%s\\t%s\\t%s\\t%s\\tDEFERRED_WINDOW\\n" "$(date -Is)" "${CAND_ID[$i]}" "${CAND_RUN[$i]}" "${CAND_GPU[$i]}" "${CAND_K[$i]}" | tee -a "${LOG_ROOT}/deferred.tsv"',
            "    fi",
            "  done",
            "}",
            "",
            "launch_idx() {",
            '  local i="$1" cid="${CAND_ID[$1]}" run="${CAND_RUN[$1]}" gpu="${CAND_GPU[$1]}" log_path="${LOG_PATH[$1]}" run_dir="${RUNS_ROOT}/${CAND_RUN[$1]}"',
            '  if should_skip "${cid}" "${run}"; then STATUS[$i]="skipped_only"; return 0; fi',
            '  if [[ "${DRY_RUN}" == "1" ]]; then',
            '    mkdir -p "${LOG_ROOT}"',
            '    printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tDRY_RUN\\t%s\\n" "$(date -Is)" "${cid}" "${run}" "${gpu}" "${CAND_RATIO[$i]}" "${CAND_K[$i]}" "${CAND_STRATEGY[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            '    echo "[DRY-RUN] ${CAND_CMD[$i]}"; STATUS[$i]="dry_run"; return 0',
            "  fi",
            '  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then',
            '    STATUS[$i]="blocked_path"; printf "%s\\t%s\\tBLOCKED_PATH\\t%s\\t%s\\n" "${cid}" "${run}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"; return 0',
            "  fi",
            '  mkdir -p "${LOG_ROOT}" "${run_dir}"',
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tSTART\\t%s\\n" "$(date -Is)" "${cid}" "${run}" "${gpu}" "${CAND_RATIO[$i]}" "${CAND_K[$i]}" "${CAND_STRATEGY[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            '  bash -lc "${CAND_CMD[$i]}" > "${log_path}" 2>&1 &',
            '  PID[$i]="$!"; STATUS[$i]="running"',
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${cid}" "${run}" "${CAND_RATIO[$i]}" "${CAND_K[$i]}" "${CAND_NOMINAL_N[$i]}" "${gpu}" "${PID[$i]}" "${CAND_AXIS[$i]}" "${CAND_ACTION[$i]}" "${CAND_STRATEGY[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
            "}",
            "",
            "reap_finished() {",
            "  local i rc",
            '  for i in "${!STATUS[@]}"; do',
            '    if [[ "${STATUS[$i]}" == "running" ]]; then',
            '      if ! kill -0 "${PID[$i]}" 2>/dev/null; then',
            '        if wait "${PID[$i]}"; then rc=0; else rc="$?"; fi',
            '        STATUS[$i]="done_${rc}"',
            '        printf "%s\\t%s\\t%s\\t%s\\t%s\\tDONE\\trc=%s\\n" "$(date -Is)" "${CAND_ID[$i]}" "${CAND_RUN[$i]}" "${CAND_GPU[$i]}" "${CAND_K[$i]}" "${rc}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "      fi",
            "    fi",
            "  done",
            "}",
            "",
            'queued_left() { local i n=0; for i in "${!STATUS[@]}"; do [[ "${STATUS[$i]}" == "queued" ]] && n=$((n + 1)); done; echo "${n}"; }',
            'running_left() { local i n=0; for i in "${!STATUS[@]}"; do [[ "${STATUS[$i]}" == "running" ]] && n=$((n + 1)); done; echo "${n}"; }',
            "",
            "launch_available() {",
            "  local gpu i current capacity launched",
            "  for gpu in 0 1 2 3 4 5 6 7; do",
            '    current="$(gpu_process_count "${gpu}")"',
            '    [[ "${current}" =~ ^[0-9]+$ ]] || current=0',
            '    if [[ "${DRY_RUN}" == "1" ]]; then capacity=999; else capacity=$((MAX_ACTIVE_PER_GPU - current)); fi',
            "    launched=0",
            "    if (( capacity <= 0 )); then continue; fi",
            '    for i in "${!STATUS[@]}"; do',
            '      if [[ "${STATUS[$i]}" == "queued" && "${CAND_GPU[$i]}" == "${gpu}" ]]; then',
            '        launch_idx "${i}"',
            "        launched=$((launched + 1))",
            "        if (( launched >= capacity )); then break; fi",
            "      fi",
            "    done",
            "  done",
            "}",
            "",
            'mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"',
            'cd "${ROOT}"',
            'echo "[CEN51-HRKSEG] run_id=${RUN_ID} dry_run=${DRY_RUN} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} max_seconds=${MAX_SCHEDULER_SECONDS}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            'for gpu in 0 1 2 3 4 5 6 7; do echo "[CEN51-HRKSEG] initial gpu=${gpu} count=$(gpu_process_count "${gpu}")" | tee -a "${LOG_ROOT}/scheduler_events.tsv"; done',
            'START_TS="$(date +%s)"',
            "WINDOW_EXPIRED=0",
            "while true; do",
            "  reap_finished",
            '  NOW_TS="$(date +%s)"',
            "  if (( NOW_TS - START_TS < MAX_SCHEDULER_SECONDS )); then",
            "    launch_available",
            '  elif [[ "${WINDOW_EXPIRED}" == "0" ]]; then',
            "    WINDOW_EXPIRED=1",
            "    defer_queued_after_window",
            "  fi",
            '  q_left="$(queued_left)"; r_left="$(running_left)"',
            '  echo "[CEN51-HRKSEG] heartbeat=$(date -Is) queued=${q_left} running=${r_left} window_expired=${WINDOW_EXPIRED}" | tee -a "${LOG_ROOT}/scheduler_heartbeat.log"',
            '  if [[ "${q_left}" == "0" && "${r_left}" == "0" ]]; then break; fi',
            '  if [[ "${DRY_RUN}" == "1" ]]; then break; fi',
            '  sleep "${POLL_SECONDS}"',
            "done",
            'echo "[CEN51-HRKSEG] scheduler_complete $(date -Is)" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(path: Path, rows: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cid",
            "run_name",
            "ratio",
            "ratio_tag",
            "k_cap",
            "n_eff_nominal",
            "cap_fraction",
            "gpu",
            "seed",
            "strategy",
            "axis",
            "action",
            "hypothesis",
            "success_gate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def render_report(run_id: str, rows: Sequence[Candidate], script_path: Path, matrix_path: Path, manifest_path: Path, scheduler_hours: float) -> str:
    by_ratio = Counter(f"{row.ratio:.1f}" for row in rows)
    by_strategy = Counter(row.strategy for row in rows)
    by_gpu = Counter(row.gpu for row in rows)
    table = [
        "| ratio | K caps | nominal train sizes | candidates |",
        "|---:|---|---|---:|",
    ]
    for ratio in sorted({row.ratio for row in rows}):
        subset = [row for row in rows if row.ratio == ratio]
        caps = sorted({row.k_cap for row in subset})
        sizes = [cap * TRAIN_DOMAIN_COMBOS for cap in caps]
        table.append(f"| {ratio:.1f} | `{','.join(map(str, caps))}` | `{','.join(map(str, sizes))}` | {len(subset)} |")

    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## 目标",
            "",
            "验证高训练比例和大 K 下，域指标控制器应按 `K_eff`（每 TX/domain combo 的有效样本）分段，还是按 `N_eff=K_eff×domain_combos` 总样本数分段。",
            "",
            "## 核心判断",
            "",
            "- 控制器不能只按 `K×域组合数`。总样本数增加会降低平均梯度方差，但每 combo 的样本数决定 TX 身份估计稳定性，域组合数决定 receiver/day floor 的覆盖度。",
            "- 当前 WiSig 子集固定 84 个 train combo，因此在本批中 `K_eff` 与 `N_eff` 单调相关；为了区分二者，矩阵加入 `TOTAL_OVER` 负控，专门模拟只看总样本而过度增强 domain/sat 压力。",
            "- 对 0.2/0.3/0.5 ratio，控制器应该从少样本的保守 K 分段，过渡到 `K_eff` 驱动的 RX/group-DRO 与 moderate satellite guard，而不是所有 loss 同时增大。",
            "",
            "## 实验矩阵",
            "",
            *table,
            "",
            f"- candidates: {len(rows)}",
            f"- ratio counts: `{dict(sorted(by_ratio.items()))}`",
            f"- strategy counts: `{dict(sorted(by_strategy.items()))}`",
            f"- GPU counts: `{dict(sorted(by_gpu.items()))}`",
            f"- scheduler window: {scheduler_hours:.1f} hours",
            "- max active per GPU: 2",
            "",
            "## 四类控制策略",
            "",
            "- `KONLY_B03`：复用 K100 的 `B03_SAT_FLOOR_GUARD`，检验只按旧 K 段是否足够。",
            "- `NEFF_RXSAT`：按 `K_eff`/cap fraction 提升 RX/group-DRO，并保持 moderate satellite guard，是本批主假设。",
            "- `RATIO_STRICT`：高 ratio 下先保护 strict/overall，降低 satellite 过压，检验是否应把新数据用于 identity 稳定。",
            "- `TOTAL_OVER`：按总样本数过度增强 domain/sat/proto/supcon/fishr 的负控；若 floor 高但 strict 掉，说明不能按总样本盲控。",
            "",
            "## 成功判据",
            "",
            "- 每个 `(ratio,K)` 内，`NEFF_RXSAT` 或 `RATIO_STRICT` 至少一个应在 strict/primary 上优于 `KONLY_B03`，否则旧 K100 控制可迁移。",
            "- `TOTAL_OVER` 若只提升 satellite/receiver floor 但损害 strict、primary 或 rollback，应被控制器拒绝。",
            "- 解析时必须记录实际 `split_info.train_size`，用实际 `K_eff`/`N_eff` 回归，而不是只看名义 K。",
            "- 对 ratio 0.3/0.5，如果 `RATIO_STRICT` 胜出，说明高比例阶段应偏 identity/rollback；如果 `NEFF_RXSAT` 胜出，说明 K_eff 仍是主分段变量。",
            "",
            "## 路径与验证",
            "",
            f"- local launcher: `{script_path}`",
            f"- local matrix: `{matrix_path}`",
            f"- local manifest: `{manifest_path}`",
            f"- remote logs: `{REMOTE_ROOT}/logs/{run_id}`",
            f"- remote runs: `{REMOTE_ROOT}/runs/{run_id}`",
            "",
            "## 完成后分析命令",
            "",
            "```powershell",
            f"conda activate ssr-gpu",
            f"python tools\\cen51_domain_metric_full_log_analysis.py --log-dir analysis_tmp\\{run_id}\\remote_logs_full --matrix-json {matrix_path} --out-dir analysis_tmp\\{run_id}\\full_log_analysis",
            "```",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_highratio_kseg_controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows = make_candidates()
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    manifest_path = report_dir / "manifest.tsv"
    report_path = report_dir / "report.md"

    script_path.write_text(render_launcher(run_id, rows, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_manifest(manifest_path, rows)
    report_path.write_text(render_report(run_id, rows, script_path, matrix_path, manifest_path, args.scheduler_hours), encoding="utf-8", newline="\n")

    print(f"[CEN51-HRKSEG] run_id={run_id}")
    print(f"[CEN51-HRKSEG] launcher={script_path}")
    print(f"[CEN51-HRKSEG] report={report_path}")
    print(f"[CEN51-HRKSEG] candidates={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
