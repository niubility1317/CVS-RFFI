#!/usr/bin/env python
"""Generate the CEN51 full-DG shot-aware LACSR-FD per-combo matrix."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
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
    feature_norm_mode: str
    feature_norm_target: float
    group_top_frac: float
    group_tau: float
    group_cap: float
    mixstyle_p: float
    mixstyle_strength: float
    aug_scale_min: float
    aug_scale_max: float
    rationale: str
    success_gate: str


def q(value: object) -> str:
    return shlex.quote(str(value))


def bash_items(items: Iterable[object], indent: str = "  ") -> str:
    return "".join(f"{indent}{q(item)}\n" for item in items)


def base_args() -> list[str]:
    return [
        "--train_mode",
        "centralized",
        "--eval_batch_size",
        "256",
        "--dataset",
        "wisig",
        "--wisig_protocol",
        "cvs_day_rx",
        "--wisig_domain",
        "rx_day",
        "--wisig_equalized",
        "1",
        "--wisig_train_ratio",
        "0.1",
        "--wisig_val_ratio",
        "-1.0",
        "--wisig_split_strategy",
        "random",
        "--wisig_cap_strategy",
        "random",
        "--wisig_train_days",
        "0,1",
        "--wisig_test_days",
        "2,3",
        "--wisig_train_rxs",
        "0,1,2,3,4,5,6",
        "--wisig_test_rxs",
        "7,8,9,10,11",
        "--test_eval_policy",
        "interval_final",
        "--test_eval_start_epoch",
        "31",
        "--test_eval_interval",
        "10",
        "--eval_sat_channel",
        "--eval_sat_on",
        "test_unseen_day_unseen_rx",
        "--eval_sat_scenarios",
        ALL_SAT,
        "--sat_eval_max_batches",
        "-1",
        "--arch_family",
        "cvsincnet",
        "--slim_group",
        "none",
        "--model_variant",
        "lite_d",
        "--branch_ablation",
        "no_dac",
        "--domain_branch_ablation",
        "no_stats",
        "--domain_enhancer",
        "rcn_stats",
        "--domain_enhancer_strength",
        "0.35",
        "--id_time_stability_mode",
        "off",
        "--id_freq_stability_mode",
        "off",
        "--domain_time_stability_mode",
        "off",
        "--domain_freq_stability_mode",
        "off",
        "--exp_group",
        "s3_rxrobust_no_dac",
        "--pa_orders",
        "1,3,5",
        "--collapse_guard",
        "--collapse_guard_min_epoch",
        "35",
        "--collapse_guard_best_margin",
        "10.0",
        "--collapse_guard_max_skipped_delta",
        "2",
        "--use_ema_ckpt",
        "--ema_decay",
        "0.999",
        "--use_swad_ckpt",
        "--swad_interval",
        "1",
        "--swad_tolerance",
        "0.85",
        "--primary_udu_weight",
        "0.82",
        "--label_smoothing",
        "0.0",
    ]


def make_candidates() -> list[Candidate]:
    rows = [
        # cid, shots, seed, epochs, swad, aug, mix, sat_p, sat_start, scenarios,
        # dom, adv, orth, cons, gce, proto, sup, fishr, fn, fn_mode, fn_target,
        # top, tau, cap, mix_p, mix_s, aug_min, aug_max, rationale, gate
        (
            "FS005_SATMIN_HINGE_1337",
            5,
            1337,
            195,
            55,
            False,
            False,
            0.08,
            1,
            LIGHT_SAT,
            0.36,
            0.14,
            0.010,
            0.004,
            0.004,
            0.0005,
            0.0005,
            0.0,
            7.5e-4,
            "hinge",
            8.0,
            0.16,
            0.35,
            0.42,
            0.0,
            0.0,
            0.0,
            0.0,
            "Replicate the best full-DG K5 family but replace zero-norm compression with a hinge norm guard.",
            "val_tx >= 90.0, strict UDU >= 74.0, rollback <= 2.5.",
        ),
        (
            "FS005_SATMIN_HINGE_2029",
            5,
            2029,
            195,
            55,
            False,
            False,
            0.08,
            1,
            LIGHT_SAT,
            0.36,
            0.14,
            0.010,
            0.004,
            0.004,
            0.0005,
            0.0005,
            0.0,
            7.5e-4,
            "hinge",
            8.0,
            0.16,
            0.35,
            0.42,
            0.0,
            0.0,
            0.0,
            0.0,
            "Seed replicate for the K5 hinge full-DG hypothesis.",
            "Seed gap versus 1337 <= 3pp strict and <= 2pp val.",
        ),
        (
            "FS005_RXGUARD_LATE_2028",
            5,
            2028,
            195,
            60,
            False,
            False,
            0.10,
            55,
            LIGHT_SAT,
            0.50,
            0.16,
            0.020,
            0.006,
            0.006,
            0.0005,
            0.0005,
            0.0,
            1.2e-4,
            "l2",
            0.0,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "Protect the first identity boundary, then add weak full-DG satellite pressure.",
            "strict UDU >= current K5 full-DG seed2028 branch and val >= 88.",
        ),
        (
            "FS005_SATMIN_HINGE_2028",
            5,
            2028,
            195,
            55,
            False,
            False,
            0.08,
            1,
            LIGHT_SAT,
            0.36,
            0.14,
            0.010,
            0.004,
            0.004,
            0.0005,
            0.0005,
            0.0,
            7.5e-4,
            "hinge",
            8.0,
            0.16,
            0.35,
            0.42,
            0.0,
            0.0,
            0.0,
            0.0,
            "Seed2028 variant of the K5 SATMIN hinge branch, matching the previous K5 val leader split.",
            "Retain the previous K5 val leader while lifting strict toward the seed1337 strict leader.",
        ),
        (
            "FS010_RXGUARD_HINGE_2028",
            10,
            2028,
            200,
            60,
            False,
            False,
            0.12,
            1,
            LIGHT_SAT,
            0.52,
            0.16,
            0.025,
            0.006,
            0.010,
            0.0010,
            0.0010,
            0.0,
            5.0e-4,
            "hinge",
            8.5,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "Repair K10 validation/strict trade-off by keeping RX guard but avoiding zero-norm compression.",
            "strict UDU >= 76.5 and val_tx >= 93.0.",
        ),
        (
            "FS010_RXGUARD_HINGE_2029",
            10,
            2029,
            200,
            60,
            False,
            False,
            0.12,
            1,
            LIGHT_SAT,
            0.52,
            0.16,
            0.025,
            0.006,
            0.010,
            0.0010,
            0.0010,
            0.0,
            5.0e-4,
            "hinge",
            8.5,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "Seed replicate for the K10 RX-guard hinge setting.",
            "Seed gap versus 2028 <= 2.5pp strict and <= 1.5pp val.",
        ),
        (
            "FS010_LACSRFD_BAL_1337",
            10,
            1337,
            200,
            65,
            False,
            True,
            0.14,
            35,
            LIGHT_SAT,
            0.44,
            0.18,
            0.020,
            0.008,
            0.014,
            0.0015,
            0.0015,
            0.0,
            1.1e-4,
            "l2",
            0.0,
            0.18,
            0.36,
            0.45,
            0.025,
            0.12,
            0.02,
            0.12,
            "LACSR-style light grouping and prototype pressure, but satellite remains full-DG rather than CE-only.",
            "Match RXGUARD strict while improving satellite floor.",
        ),
        (
            "FS010_IDFIRST_LATE_2030",
            10,
            2030,
            195,
            65,
            False,
            False,
            0.10,
            70,
            LIGHT_SAT,
            0.42,
            0.12,
            0.015,
            0.004,
            0.006,
            0.0008,
            0.0008,
            0.0,
            4.5e-4,
            "hinge",
            8.5,
            0.16,
            0.34,
            0.42,
            0.0,
            0.0,
            0.0,
            0.0,
            "Late full-DG satellite tests whether K10 still benefits from an identity-first phase.",
            "val_tx >= 93.0 with final rollback <= 2pp.",
        ),
        (
            "FS020_LACSRFD_RELAX_2028",
            20,
            2028,
            205,
            70,
            True,
            True,
            0.16,
            35,
            ALL_SAT,
            0.45,
            0.20,
            0.020,
            0.010,
            0.014,
            0.0020,
            0.0020,
            0.0,
            8.0e-5,
            "l2",
            0.0,
            0.18,
            0.36,
            0.48,
            0.040,
            0.16,
            0.03,
            0.18,
            "K20 repair: relax full-DG pressure because prior full-DG dipped below LACSR/RIEI-FD strict.",
            "strict UDU >= 77.6 and val_tx >= 95.0.",
        ),
        (
            "FS020_RIEIFD_FDGATE_2029",
            20,
            2029,
            205,
            70,
            True,
            True,
            0.14,
            1,
            LIGHT_SAT,
            0.46,
            0.18,
            0.020,
            0.008,
            0.012,
            0.0015,
            0.0015,
            0.0,
            7.0e-5,
            "l2",
            0.0,
            0.18,
            0.35,
            0.46,
            0.035,
            0.14,
            0.03,
            0.16,
            "RIEI-FD-like clean-balanced branch with weak full-DG satellite.",
            "Keep RIEI-FD validation advantage without K20 strict regression.",
        ),
        (
            "FS020_LATE_REPAIR_1337",
            20,
            1337,
            205,
            70,
            True,
            True,
            0.20,
            75,
            ALL_SAT,
            0.48,
            0.22,
            0.025,
            0.010,
            0.018,
            0.0025,
            0.0025,
            0.0002,
            6.0e-5,
            "l2",
            0.0,
            0.20,
            0.38,
            0.50,
            0.045,
            0.16,
            0.03,
            0.18,
            "Delayed all-scenario full-DG satellite for K20 monotonic repair.",
            "Strict improves over full-DG K20 while satellite floor does not regress.",
        ),
        (
            "FS030_LACSRFD_BAL_2028",
            30,
            2028,
            210,
            75,
            True,
            True,
            0.22,
            35,
            ALL_SAT,
            0.50,
            0.24,
            0.025,
            0.012,
            0.024,
            0.0035,
            0.0035,
            0.0003,
            5.0e-5,
            "l2",
            0.0,
            0.22,
            0.40,
            0.55,
            0.060,
            0.20,
            0.04,
            0.22,
            "K30 balanced growth point: stronger than K20 but below the old strong-DG setting.",
            "strict UDU >= 80.0 and val_tx >= 97.0.",
        ),
        (
            "FS030_RXFLOOR_2029",
            30,
            2029,
            210,
            75,
            True,
            True,
            0.20,
            1,
            ALL_SAT,
            0.56,
            0.22,
            0.030,
            0.010,
            0.022,
            0.0030,
            0.0030,
            0.0003,
            5.0e-5,
            "l2",
            0.0,
            0.18,
            0.36,
            0.48,
            0.050,
            0.18,
            0.04,
            0.20,
            "Receiver-floor branch for K30, aimed at rx8/rx11 rather than only average strict.",
            "Worst-rx floor improves without strict falling below 79.",
        ),
        (
            "FS050_LACSRFD_MONO_2028",
            50,
            2028,
            215,
            80,
            True,
            True,
            0.26,
            45,
            ALL_SAT,
            0.54,
            0.28,
            0.030,
            0.014,
            0.032,
            0.0050,
            0.0050,
            0.0005,
            3.5e-5,
            "l2",
            0.0,
            0.24,
            0.44,
            0.58,
            0.080,
            0.22,
            0.05,
            0.24,
            "K50 monotonic branch: keep LACSR high-shot strength while using full-DG satellite.",
            "strict UDU >= 84.1, val_tx >= 98.0, no K30->K50 regression.",
        ),
        (
            "FS050_LACSRFD_MONO_2029",
            50,
            2029,
            215,
            80,
            True,
            True,
            0.26,
            45,
            ALL_SAT,
            0.54,
            0.28,
            0.030,
            0.014,
            0.032,
            0.0050,
            0.0050,
            0.0005,
            3.5e-5,
            "l2",
            0.0,
            0.24,
            0.44,
            0.58,
            0.080,
            0.22,
            0.05,
            0.24,
            "Seed replicate for the K50 monotonic branch.",
            "Seed gap versus 2028 <= 2pp strict and <= 1pp val.",
        ),
        (
            "FS050_CAP_RELAX_1337",
            50,
            1337,
            215,
            80,
            True,
            True,
            0.22,
            25,
            ALL_SAT,
            0.52,
            0.24,
            0.025,
            0.012,
            0.026,
            0.0040,
            0.0040,
            0.0003,
            3.0e-5,
            "l2",
            0.0,
            0.22,
            0.40,
            0.55,
            0.070,
            0.20,
            0.05,
            0.22,
            "Higher-shot relaxed branch to test whether the previous K50 full-DG loss stack was too strong.",
            "Beat full-DG K50 strict 81.45 and approach LACSR K50 84.11.",
        ),
    ]
    gpus = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    candidates: list[Candidate] = []
    for idx, row in enumerate(rows):
        cid = row[0]
        shots = int(row[1])
        seed = int(row[2])
        run_name = f"CEN51_FDLACSRFD_{cid}"
        batch_size = 128 if shots <= 10 else 256
        candidates.append(Candidate(cid, run_name, shots, int(gpus[idx]), seed, row[3], row[4], batch_size, *row[5:]))
    return candidates


def candidate_args(c: Candidate) -> list[str]:
    args = [
        "--batch_size",
        str(c.batch_size),
        "--epochs",
        str(c.epochs),
        "--swad_start_epoch",
        str(c.swad_start),
        "--seed",
        str(c.seed),
        "--sat_view_seed",
        str(c.seed + 7919),
        "--no_enable_pa_aux",
        "--no_enable_dac_aux",
        "--no_aug_enable_pa_normal",
        "--aug_p_pa",
        "0.0",
        "--aug_p_dac",
        "0.0",
        "--lambda_cls_pa",
        "0.0",
        "--lambda_pa_joint_inv",
        "0.0",
        "--lambda_pa_kl",
        "0.0",
        "--lambda_pa_reg",
        "0.0",
        "--use_concat_sat_channel_aug",
        "--no_use_sat_consistency",
        "--lambda_sat_cls",
        "0.0",
        "--lambda_sat_cons",
        "0.0",
        "--concat_sat_ce_weight",
        "0.0",
        "--sat_cons_start_epoch",
        "999",
        "--sat_view_prob",
        f"{c.sat_prob:.3f}",
        "--sat_train_scenarios",
        c.sat_scenarios,
        "--sat_view_schedule",
        f"1@{c.sat_prob:.3f}:{c.sat_scenarios}",
        "--concat_sat_start_epoch",
        str(c.sat_start),
        "--lambda_dom",
        f"{c.lambda_dom:.4g}",
        "--lambda_adv",
        f"{c.lambda_adv:.4g}",
        "--grl_lambda",
        "1.0",
        "--lambda_orth",
        f"{c.lambda_orth:.4g}",
        "--lambda_cons",
        f"{c.lambda_cons:.4g}",
        "--lambda_group_ce",
        f"{c.lambda_group_ce:.4g}",
        "--lambda_proto",
        f"{c.lambda_proto:.4g}",
        "--lambda_supcon_id",
        f"{c.lambda_supcon:.4g}",
        "--lambda_fishr",
        f"{c.lambda_fishr:.4g}",
        "--lambda_feature_norm_guard",
        f"{c.lambda_feature_norm:.4g}",
        "--feature_norm_guard_mode",
        c.feature_norm_mode,
        "--feature_norm_guard_target",
        f"{c.feature_norm_target:.4g}",
        "--group_ce_mode",
        "smooth_dro_capped",
        "--group_ce_min_domains",
        "2",
        "--group_ce_top_frac",
        f"{c.group_top_frac:.3f}",
        "--groupdro_tau",
        f"{c.group_tau:.3f}",
        "--groupdro_cap",
        f"{c.group_cap:.3f}",
        "--fishr_min_domains",
        "2",
        "--aug_scale_min",
        f"{c.aug_scale_min:.3f}",
        "--aug_scale_max",
        f"{c.aug_scale_max:.3f}",
        "--late_aug_min_scale",
        f"{max(c.aug_scale_min, min(c.aug_scale_max, 0.16)):.3f}",
    ]
    args.append("--use_aug" if c.use_aug else "--no_use_aug")
    args.append("--use_mixstyle" if c.use_mixstyle else "--no_use_mixstyle")
    if c.use_mixstyle:
        args.extend(
            [
                "--mixstyle_p",
                f"{c.mixstyle_p:.3f}",
                "--mixstyle_strength",
                f"{c.mixstyle_strength:.3f}",
                "--mixstyle_mix",
                "same_tx_crossdomain",
                "--mixstyle_fallback",
                "skip",
                "--mixstyle_late_start",
                str(max(70, c.swad_start + 20)),
                "--mixstyle_late_ramp_epochs",
                "35",
                "--mixstyle_late_min_p",
                f"{max(0.02, c.mixstyle_p * 0.5):.3f}",
                "--mixstyle_late_min_strength",
                f"{max(0.10, c.mixstyle_strength * 0.65):.3f}",
            ]
        )
    if c.lambda_proto > 0.0:
        args.append("--use_proto_memory")
    else:
        args.append("--no_use_proto_memory")
    return args


def render_launcher(run_id: str, candidates: Sequence[Candidate]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="${{ROOT:-{REMOTE_ROOT}}}"',
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
        '  if [[ "${DRY_RUN}" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then',
        "    echo 0",
        "    return 0",
        "  fi",
        '  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \\',
        "    | sed '/^$/d' | wc -l | tr -d ' ' || echo 0",
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
        "declare -A INITIAL_BY_GPU=()",
        "declare -A LAUNCHED_BY_GPU=()",
        "snapshot_capacity() {",
        "  local gpu",
        "  for gpu in 0 1 2 3 4 5 6 7; do",
        '    INITIAL_BY_GPU[${gpu}]="$(gpu_process_count "${gpu}")"',
        "    LAUNCHED_BY_GPU[${gpu}]=0",
        "  done",
        "}",
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
        '  echo "[CEN51-FDLACSRFD] candidate=${candidate_id} run=${run_name} shots=${shots} gpu=${gpu} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        "  printf '[CEN51-FDLACSRFD-CMD]'",
        '  print_cmd "${cmd[@]}"',
        '  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi',
        '  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then',
        '    mkdir -p "${LOG_ROOT}"',
        '    printf "%s\\t%s\\t%s\\t%s\\t%s\\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"',
        "    return 0",
        "  fi",
        "  local initial_count local_count",
        '  initial_count="${INITIAL_BY_GPU[${gpu}]:-0}"',
        '  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"',
        "  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then",
        '    mkdir -p "${LOG_ROOT}"',
        '    printf "%s\\t%s\\t%s\\tgpu=%s initial_count=%s local_count=%s max=%s\\n" "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${initial_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"',
        "    return 0",
        "  fi",
        '  mkdir -p "${LOG_ROOT}" "${run_dir}"',
        '  nohup "${cmd[@]}" > "${log_path}" 2>&1 &',
        '  local pid="$!"',
        '  LAUNCHED_BY_GPU[${gpu}]=$(( local_count + 1 ))',
        '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${candidate_id}" "${run_name}" "${shots}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
        "}",
        "",
        "BASE_ARGS=(",
        bash_items(base_args()).rstrip(),
        ")",
        "",
        'if [[ "${DRY_RUN}" != "1" ]]; then',
        '  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }',
        "fi",
        'cd "${ROOT}"',
        "snapshot_capacity",
        'echo "[CEN51-FDLACSRFD] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"',
        'echo "[CEN51-FDLACSRFD] initial_gpu_counts: ${INITIAL_BY_GPU[*]}"',
        "",
    ]
    for c in candidates:
        lines.append(f"run_candidate {q(c.cid)} {q(c.run_name)} {c.shots} {c.gpu} \\")
        args = candidate_args(c)
        for idx, item in enumerate(args):
            suffix = " \\" if idx < len(args) - 1 else ""
            lines.append(f"  {q(item)}{suffix}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_report(run_id: str, candidates: Sequence[Candidate], script_path: Path, matrix_path: Path) -> str:
    gpu_counts: dict[int, int] = {}
    shot_counts: dict[int, int] = {}
    for c in candidates:
        gpu_counts[c.gpu] = gpu_counts.get(c.gpu, 0) + 1
        shot_counts[c.shots] = shot_counts.get(c.shots, 0) + 1

    rows = [
        "| ID | shots | GPU | seed | sat p/start | FD losses | norm | purpose |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for c in candidates:
        losses = (
            f"dom={c.lambda_dom}, adv={c.lambda_adv}, gce={c.lambda_group_ce}, "
            f"proto={c.lambda_proto}, sup={c.lambda_supcon}, fishr={c.lambda_fishr}"
        )
        norm = f"{c.feature_norm_mode}:{c.lambda_feature_norm}@{c.feature_norm_target}"
        rows.append(
            f"| `{c.cid}` | {c.shots} | {c.gpu} | {c.seed} | "
            f"{c.sat_prob}/{c.sat_start} | {losses} | {norm} | {c.rationale} |"
        )

    return f"""# {run_id}

## Objective

Per-combo CEN51 full-DG shot-aware LACSR-FD verification for K=5/10/20/30/50.
The matrix keeps CVS/CEN51 as the backbone and keeps satellite augmentation
enabled as full-DG for every candidate. No candidate uses `concat_sat_ce_only`;
the satellite view is concatenated into the main batch, so TX CE, domain/GRL,
GroupCE, prototype, SupCon, Fishr, and feature-norm guard see the satellite
samples through the normal main loss path.

## Diagnosis From Previous Runs

- Full-DG is the best per-combo K5/K10 line, but K5 validation is just below
  90 and K20 strict UDU regresses below LACSR/RIEI-FD.
- LACSR is the most coherent K20/K50 high-shot rescue, but its earlier
  satellite rescue path was not the default full-DG concat path.
- RIEI-FD improves validation through nuisance suppression and norm guard, but
  it is not the best strict line at K5/K10.
- The new matrix therefore tests hinge norm guards at K5/K10, relaxed K20
  full-DG, and LACSR-like K30/K50 growth with full-DG satellite kept on.

## Capacity Plan

- Candidates: {len(candidates)}
- GPU counts: {json.dumps(gpu_counts, sort_keys=True)}
- Shot counts: {json.dumps(shot_counts, sort_keys=True)}
- Launcher default: `MAX_TRAIN_PER_GPU=2`; use a higher value only after
  preflight and capacity audit, and record it in this report.

## Local Artifacts

- Generator: `tools/cen51_fulldg_shotaware_lacsr_fd_matrix.py`
- Launcher: `{script_path.as_posix()}`
- Matrix JSON: `{matrix_path.as_posix()}`
- Report: `automation_reports/CV-SincNet/{run_id}/report.md`

## Candidate Matrix

{chr(10).join(rows)}

## Success Criteria

- K5: improve the joint envelope: strict leader `73.78 strict / 89.95 val`
  and validation leader `72.91 strict / 90.81 val`; target val >= 90
  while moving strict toward or above 74.
- K10: improve over full-DG K10 `76.25 strict / 93.01 val`.
- K20: repair the full-DG dip and match or beat LACSR/RIEI-FD (`strict >= 77.6`,
  `val >= 95`).
- K30: reach strict >= 80 and keep val >= 97.
- K50: recover LACSR high-shot strength (`strict >= 84.1`, `val >= 98`) while
  keeping full-DG satellite enabled.
- Stability: seed-replicated branches should stay within 2-3pp strict and
  1-2pp validation.

## Verification Plan

- `conda activate ssr-gpu; python -m py_compile tools/cen51_fulldg_shotaware_lacsr_fd_matrix.py code/train.py`
- `conda activate ssr-gpu; python tools/cen51_fulldg_shotaware_lacsr_fd_matrix.py --run-id {run_id}`
- `bash -n code/scripts/launch_{run_id}.sh`
- Remote: dry-run the launcher, confirm no `concat_sat_ce_only`, and verify
  startup logs contain `[CONFIG-CONCAT-SAT]` / `[CONCAT-SAT-AUG]` with
  `ce_only=0`.

## Risks

- Hinge feature-norm may under-regularize K5 if the threshold is too high.
- K20 remains the main brittle region because full-DG pressure can dilute clean
  identity before enough samples support all-scenario invariance.
- Full-DG satellite doubles the supervised path; if validation rises but strict
  drops, the candidate is a source-fit branch, not the default.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=Path("automation_reports") / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=Path("code") / "scripts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_fulldg_lacsr_fd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidates = make_candidates()
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"

    script_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    matrix_path.write_text(
        json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(run_id, candidates, script_path, matrix_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": len(candidates),
                "launcher": str(script_path),
                "matrix": str(matrix_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
