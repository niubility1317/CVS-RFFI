#!/usr/bin/env python
"""Generate the comprehensive CVS-SAFD vs RIEI/DRIFT low-shot matrix.

This planner uses one unified CVS low-shot algorithm below 100 samples per
combo: CEN51-SAFD full-DG. At K >= 100 it must restore the original
CEN51_R04 ratio=0.1 recipe exactly; K only selects the controller prior below
that boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cen51_lowshot_config_search import BASE_PARAMS, SAT_SCENARIOS, arg_pairs
from cen51_r04_config import cen51_r04_ratio_params, should_restore_cen51_r04


SELECTED_CVS_ALGORITHM = "CEN51_SAFD_FULLDG"
SHOTS: Sequence[int] = (10, 20, 30, 50, 100)
METHODS: Sequence[str] = (
    "cvs_safd",
    "riei_fixed_nosat",
    "riei_fixed_sat",
    "drift_fixed_nosat",
    "drift_fixed_sat",
)
GPU_IDS: Sequence[int] = tuple(range(8))
WISIG_PKL_TOKEN = "__WISIG_PKL__"
OUT_DIR_TOKEN = "__OUT_DIR__"
ROOT_RUN_ID_PREFIX = "cvs_lacsr_riei_drift_comprehensive"


@dataclass(frozen=True)
class Profile:
    profile_id: str
    train_rxs: str
    train_days: str
    test_rxs: str = "7,8,9,10,11"
    test_days: str = "2,3"
    rationale: str = ""

    @property
    def train_rx_count(self) -> int:
        return len([rx for rx in self.train_rxs.split(",") if rx])


PROFILES: Sequence[Profile] = (
    Profile("rx7_all_d01", "0,1,2,3,4,5,6", "0,1", rationale="default seven training receivers over two days"),
    Profile("rx7_all_d0", "0,1,2,3,4,5,6", "0", rationale="one-day training stress test"),
    Profile("rx3_sp_d01", "0,3,6", "0,1", rationale="sparse low/mid/high receiver identity set"),
    Profile("rx5_sp_d01", "0,1,3,5,6", "0,1", rationale="five-receiver mixed identity set"),
    Profile("rx3_lo_d01", "0,1,2", "0,1", rationale="low-index receiver identity set"),
    Profile("rx3_hi_d01", "4,5,6", "0,1", rationale="high-index receiver identity set"),
    Profile("rx5_lo_d01", "0,1,2,3,4", "0,1", rationale="five low/mid receivers"),
    Profile("rx5_hi_d01", "2,3,4,5,6", "0,1", rationale="five mid/high receivers"),
)


def shell_join(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _shell_word(token: str) -> str:
    if token == WISIG_PKL_TOKEN:
        return '"${WISIG_PKL}"'
    if token == OUT_DIR_TOKEN:
        return '"${out_dir}"'
    if token.startswith(f"{OUT_DIR_TOKEN}/"):
        suffix = token[len(OUT_DIR_TOKEN) + 1 :]
        return f'"${{out_dir}}/{suffix}"'
    return shlex.quote(str(token))


def _with_common_wisig_args(profile: Profile, shots: int, seed: int) -> Dict[str, object]:
    return {
        "wisig_pkl": WISIG_PKL_TOKEN,
        "wisig_protocol": "cvs_day_rx",
        "wisig_equalized": 1,
        "wisig_domain": "rx_day",
        "wisig_out_len": 256,
        "wisig_train_ratio": 0.1,
        "wisig_val_ratio": "-1.0",
        "wisig_guard_gap": 8,
        "wisig_train_days": profile.train_days,
        "wisig_test_days": profile.test_days,
        "wisig_train_rxs": profile.train_rxs,
        "wisig_test_rxs": profile.test_rxs,
        "wisig_split_strategy": "random",
        "wisig_cap_strategy": "random",
        "wisig_max_day123_per_combo": 0,
        "wisig_max_train_per_combo": shots,
        "wisig_max_val_per_combo": 0,
        "wisig_max_test_per_combo": 0,
        "eval_batch_size": 256,
        "num_workers": 0,
        "prefetch_factor": 2,
        "seed": seed,
    }


def _baseline_common_args(job: "Job") -> List[str]:
    params = _with_common_wisig_args(job.profile, job.shots, job.seed)
    args = arg_pairs(params)
    args.extend(
        [
            "--eval_sat_channel",
            "--eval_sat_on",
            "main",
            "--eval_sat_scenarios",
            SAT_SCENARIOS,
            "--sat_eval_max_batches",
            "0",
        ]
    )
    if job.sat_train:
        args.extend(
            [
                "--use_sat_channel_view_aug",
                "--sat_train_scenarios",
                SAT_SCENARIOS,
                "--sat_view_prob",
                "1.0",
                "--sat_view_seed",
                "2027",
            ]
        )
    args.extend(["--output_dir", OUT_DIR_TOKEN, "--epochs", "200", "--no_test_on_val_improve", "--batch_size", "64"])
    return args


def _riei_args(job: "Job") -> List[str]:
    args = _baseline_common_args(job)
    args.extend(
        [
            "--lr_all",
            "0.0001",
            "--lr_fed",
            "0.0001",
            "--lambda_mi",
            "1.2",
            "--lambda_ie",
            "1.2",
            "--ce_reduction",
            "sum",
            "--mi_reduction",
            "sum",
            "--ie_reduction",
            "sum",
            "--paper_eval_last_n",
            "1",
            "--lambda_feature_norm",
            "0.0001",
            "--paper_eval_name",
            f"{job.method}_last1",
        ]
    )
    return args


def _drift_args(job: "Job") -> List[str]:
    args = _baseline_common_args(job)
    args.extend(
        [
            "--lr",
            "0.0001",
            "--lambda_grl",
            "1.0",
            "--grl_coeff",
            "1.0",
            "--lambda_center",
            "0.01",
            "--center_mode",
            "ema",
            "--center_momentum",
            "0.95",
            "--lambda_mse",
            "0.02",
            "--no-normalize_features_for_mse",
            "--mse_reduction",
            "sum",
            "--domain_discriminator_layers",
            "2",
            "--grl_schedule",
            "constant",
            "--paper_eval_last_n",
            "1",
            "--mse_cap",
            "4000",
            "--paper_eval_name",
            f"{job.method}_last1",
        ]
    )
    return args


def safd_fulldg_params(shots: int) -> Dict[str, object]:
    """CEN51-SAFD full-DG controller prior for per-combo few-shot runs."""
    if shots <= 10:
        return {
            "batch_size": 128,
            "epochs": 195,
            "test_eval_start_epoch": 31,
            "test_eval_interval": 10,
            "no_enable_pa_aux": True,
            "no_enable_dac_aux": True,
            "no_aug_enable_pa_normal": True,
            "aug_p_pa": 0.0,
            "aug_p_dac": 0.0,
            "lambda_cls_pa": 0.0,
            "lambda_pa_joint_inv": 0.0,
            "lambda_pa_kl": 0.0,
            "lambda_pa_reg": 0.0,
            "use_aug": False,
            "no_use_aug": True,
            "use_mixstyle": False,
            "no_use_mixstyle": True,
            "use_concat_sat_channel_aug": True,
            "no_use_sat_consistency": True,
            "lambda_sat_cls": 0.0,
            "lambda_sat_cons": 0.0,
            "concat_sat_ce_weight": 0.0,
            "sat_cons_start_epoch": 999,
            "sat_view_prob": 0.10,
            "sat_train_scenarios": "clear_leo,mixed_orbit",
            "sat_view_schedule": "1@0.100:clear_leo,mixed_orbit",
            "concat_sat_start_epoch": 70,
            "lambda_dom": 0.42,
            "lambda_adv": 0.12,
            "grl_lambda": 1.0,
            "lambda_orth": 0.015,
            "lambda_cons": 0.004,
            "lambda_group_ce": 0.006,
            "lambda_proto": 0.0008,
            "lambda_supcon_id": 0.0008,
            "lambda_fishr": 0.0,
            "lambda_feature_norm_guard": 0.00045,
            "feature_norm_guard_mode": "hinge",
            "feature_norm_guard_target": 8.5,
            "group_ce_mode": "smooth_dro_capped",
            "group_ce_min_domains": 2,
            "group_ce_top_frac": 0.16,
            "groupdro_tau": 0.35,
            "groupdro_cap": 0.42,
            "fishr_min_domains": 2,
            "use_proto_memory": True,
            "swad_start_epoch": 50,
            "swad_tolerance": 0.85,
            "primary_udu_weight": 0.82,
        }
    if shots <= 20:
        return {
            "batch_size": 256,
            "epochs": 200,
            "test_eval_start_epoch": 31,
            "test_eval_interval": 10,
            "no_enable_pa_aux": True,
            "no_enable_dac_aux": True,
            "no_aug_enable_pa_normal": True,
            "aug_p_pa": 0.0,
            "aug_p_dac": 0.0,
            "lambda_cls_pa": 0.0,
            "lambda_pa_joint_inv": 0.0,
            "lambda_pa_kl": 0.0,
            "lambda_pa_reg": 0.0,
            "use_aug": False,
            "no_use_aug": True,
            "use_mixstyle": False,
            "no_use_mixstyle": True,
            "use_concat_sat_channel_aug": True,
            "no_use_sat_consistency": True,
            "lambda_sat_cls": 0.0,
            "lambda_sat_cons": 0.0,
            "concat_sat_ce_weight": 0.0,
            "sat_cons_start_epoch": 999,
            "sat_view_prob": 0.12,
            "sat_train_scenarios": "clear_leo,mixed_orbit",
            "sat_view_schedule": "1@0.120:clear_leo,mixed_orbit",
            "concat_sat_start_epoch": 1,
            "lambda_dom": 0.42,
            "lambda_adv": 0.15,
            "grl_lambda": 1.0,
            "lambda_orth": 0.020,
            "lambda_cons": 0.008,
            "lambda_group_ce": 0.010,
            "lambda_proto": 0.0015,
            "lambda_supcon_id": 0.0015,
            "lambda_fishr": 0.0,
            "lambda_feature_norm_guard": 0.00004,
            "feature_norm_guard_mode": "l2",
            "feature_norm_guard_target": 0.0,
            "group_ce_mode": "smooth_dro_capped",
            "group_ce_min_domains": 2,
            "group_ce_top_frac": 0.18,
            "groupdro_tau": 0.37,
            "groupdro_cap": 0.48,
            "fishr_min_domains": 2,
            "use_proto_memory": True,
            "swad_start_epoch": 55,
            "swad_tolerance": 0.85,
            "primary_udu_weight": 0.82,
        }
    if shots <= 30:
        return {
            "batch_size": 256,
            "epochs": 200,
            "test_eval_start_epoch": 31,
            "test_eval_interval": 10,
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
            "use_concat_sat_channel_aug": True,
            "no_use_sat_consistency": True,
            "lambda_sat_cls": 0.0,
            "lambda_sat_cons": 0.0,
            "concat_sat_ce_weight": 0.0,
            "sat_cons_start_epoch": 999,
            "sat_view_prob": 0.18,
            "sat_train_scenarios": "clear_leo,mixed_orbit",
            "sat_view_schedule": "1@0.180:clear_leo,mixed_orbit",
            "concat_sat_start_epoch": 25,
            "lambda_dom": 0.50,
            "lambda_adv": 0.20,
            "grl_lambda": 1.0,
            "lambda_orth": 0.024,
            "lambda_cons": 0.012,
            "lambda_group_ce": 0.018,
            "lambda_proto": 0.0025,
            "lambda_supcon_id": 0.0025,
            "lambda_fishr": 0.0002,
            "lambda_feature_norm_guard": 0.00004,
            "feature_norm_guard_mode": "l2",
            "feature_norm_guard_target": 0.0,
            "group_ce_mode": "smooth_dro_capped",
            "group_ce_min_domains": 2,
            "group_ce_top_frac": 0.20,
            "groupdro_tau": 0.39,
            "groupdro_cap": 0.52,
            "fishr_min_domains": 2,
            "aug_scale_min": 0.10,
            "aug_scale_max": 0.32,
            "late_aug_min_scale": 0.16,
            "mixstyle_p": 0.025,
            "mixstyle_strength": 0.24,
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_late_start": 80,
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": 0.020,
            "mixstyle_late_min_strength": 0.156,
            "use_proto_memory": True,
            "swad_start_epoch": 60,
            "swad_tolerance": 0.85,
            "primary_udu_weight": 0.82,
        }
    if shots <= 50:
        return {
            "batch_size": 256,
            "epochs": 200,
            "test_eval_start_epoch": 31,
            "test_eval_interval": 10,
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
            "use_concat_sat_channel_aug": True,
            "no_use_sat_consistency": True,
            "lambda_sat_cls": 0.0,
            "lambda_sat_cons": 0.0,
            "concat_sat_ce_weight": 0.0,
            "sat_cons_start_epoch": 999,
            "sat_view_prob": 0.22,
            "sat_train_scenarios": SAT_SCENARIOS,
            "sat_view_schedule": f"1@0.220:{SAT_SCENARIOS}",
            "concat_sat_start_epoch": 25,
            "lambda_dom": 0.52,
            "lambda_adv": 0.24,
            "grl_lambda": 1.0,
            "lambda_orth": 0.025,
            "lambda_cons": 0.012,
            "lambda_group_ce": 0.026,
            "lambda_proto": 0.0040,
            "lambda_supcon_id": 0.0040,
            "lambda_fishr": 0.0003,
            "lambda_feature_norm_guard": 0.00003,
            "feature_norm_guard_mode": "l2",
            "feature_norm_guard_target": 0.0,
            "group_ce_mode": "smooth_dro_capped",
            "group_ce_min_domains": 2,
            "group_ce_top_frac": 0.22,
            "groupdro_tau": 0.40,
            "groupdro_cap": 0.55,
            "fishr_min_domains": 2,
            "aug_scale_min": 0.05,
            "aug_scale_max": 0.22,
            "late_aug_min_scale": 0.16,
            "mixstyle_p": 0.070,
            "mixstyle_strength": 0.20,
            "mixstyle_mix": "same_tx_crossdomain",
            "mixstyle_fallback": "skip",
            "mixstyle_late_start": 85,
            "mixstyle_late_ramp_epochs": 35,
            "mixstyle_late_min_p": 0.035,
            "mixstyle_late_min_strength": 0.130,
            "use_proto_memory": True,
            "swad_start_epoch": 65,
            "swad_tolerance": 0.85,
            "primary_udu_weight": 0.82,
        }
    params = safd_fulldg_params(50)
    params.update(
        {
            "sat_view_prob": 0.18,
            "sat_view_schedule": f"1@0.180:{SAT_SCENARIOS}",
            "concat_sat_start_epoch": 35,
            "lambda_dom": 0.46,
            "lambda_adv": 0.20,
            "lambda_group_ce": 0.020,
            "lambda_proto": 0.0030,
            "lambda_supcon_id": 0.0030,
            "group_ce_top_frac": 0.18,
            "mixstyle_p": 0.045,
        }
    )
    return params


def _cvs_args(job: "Job") -> List[str]:
    if should_restore_cen51_r04(job.shots):
        params = cen51_r04_ratio_params(seed=job.seed)
        params.update(
            {
                "run_name": job.run_name,
                "latest_save_path": f"{OUT_DIR_TOKEN}/latest_model.pth",
                "best_save_path": f"{OUT_DIR_TOKEN}/best_val_model.pth",
                "best_primary_save_path": f"{OUT_DIR_TOKEN}/best_primary_ood_model.pth",
                "best_unseen_day_unseen_rx_save_path": f"{OUT_DIR_TOKEN}/best_strict_udu_model.pth",
                "best_worst_rx_save_path": f"{OUT_DIR_TOKEN}/best_worst_rx_model.pth",
                "ema_save_path": f"{OUT_DIR_TOKEN}/ema_model.pth",
                "swa_save_path": f"{OUT_DIR_TOKEN}/swa_model.pth",
                "swad_save_path": f"{OUT_DIR_TOKEN}/swad_model.pth",
            }
        )
        return arg_pairs(params)

    params: Dict[str, object] = {}
    params.update(BASE_PARAMS)
    params.update(safd_fulldg_params(job.shots))
    params.update(_with_common_wisig_args(job.profile, job.shots, job.seed))
    params.update(
        {
            "sat_eval_max_batches": 0,
            "run_name": job.run_name,
            "latest_save_path": f"{OUT_DIR_TOKEN}/latest_model.pth",
            "best_save_path": f"{OUT_DIR_TOKEN}/best_val_model.pth",
            "best_primary_save_path": f"{OUT_DIR_TOKEN}/best_primary_ood_model.pth",
            "best_unseen_day_unseen_rx_save_path": f"{OUT_DIR_TOKEN}/best_strict_udu_model.pth",
            "best_worst_rx_save_path": f"{OUT_DIR_TOKEN}/best_worst_rx_model.pth",
            "ema_save_path": f"{OUT_DIR_TOKEN}/ema_model.pth",
            "swa_save_path": f"{OUT_DIR_TOKEN}/swa_model.pth",
            "swad_save_path": f"{OUT_DIR_TOKEN}/swad_model.pth",
        }
    )
    return arg_pairs(params)


@dataclass(frozen=True)
class Job:
    index: int
    method: str
    shots: int
    profile: Profile
    gpu: int
    seed: int = 1337

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    @property
    def cvs_algorithm(self) -> str:
        if self.method != "cvs_safd":
            return ""
        if should_restore_cen51_r04(self.shots):
            return "CEN51_R04"
        return SELECTED_CVS_ALGORITHM

    @property
    def sat_train(self) -> bool:
        return self.method == "cvs_safd" or self.method.endswith("_sat")

    @property
    def method_family(self) -> str:
        if self.method.startswith("riei"):
            return "RIEI"
        if self.method.startswith("drift"):
            return "DRIFT"
        return "CVS"

    @property
    def module(self) -> str:
        if self.method.startswith("riei"):
            return "baselines.riei_fd.train"
        if self.method.startswith("drift"):
            return "baselines.drift.train"
        return ""

    @property
    def run_name(self) -> str:
        if self.method == "cvs_safd" and should_restore_cen51_r04(self.shots):
            return f"cvs_cen51_r04_fs{self.shots:03d}_ratio010_seed{self.seed}"
        return f"{self.method}_fs{self.shots:03d}_{self.profile_id}_seed{self.seed}"

    @property
    def job_id(self) -> str:
        return self.run_name

    @property
    def log_name(self) -> str:
        return f"{self.run_name}.log"

    def command_args(self) -> List[str]:
        if self.method == "cvs_safd":
            return _cvs_args(self)
        if self.method.startswith("riei"):
            return _riei_args(self)
        if self.method.startswith("drift"):
            return _drift_args(self)
        raise ValueError(f"unsupported method: {self.method}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "method": self.method,
            "method_family": self.method_family,
            "cvs_algorithm": self.cvs_algorithm,
            "shots": self.shots,
            "profile_id": self.profile_id,
            "train_rx_count": self.profile.train_rx_count,
            "train_rxs": self.profile.train_rxs,
            "train_days": self.profile.train_days,
            "test_rxs": self.profile.test_rxs,
            "test_days": self.profile.test_days,
            "gpu": self.gpu,
            "seed": self.seed,
            "sat_train": self.sat_train,
            "r04_restore": self.method == "cvs_safd" and should_restore_cen51_r04(self.shots),
            "module": self.module,
            "run_name": self.run_name,
            "command_args": self.command_args(),
        }


def make_jobs(
    gpu_ids: Sequence[int] = GPU_IDS,
    seed: int = 1337,
    exclude_job_ids: Iterable[str] | None = None,
    methods: Sequence[str] = METHODS,
) -> List[Job]:
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    excluded = set(exclude_job_ids or [])
    jobs: List[Job] = []
    idx = 0
    for profile in PROFILES:
        for shots in SHOTS:
            for method in methods:
                if (
                    method == "cvs_safd"
                    and should_restore_cen51_r04(shots)
                    and profile.profile_id != PROFILES[0].profile_id
                ):
                    continue
                gpu = gpu_ids[idx % len(gpu_ids)]
                job = Job(index=idx, method=method, shots=shots, profile=profile, gpu=gpu, seed=seed)
                if job.job_id not in excluded:
                    jobs.append(job)
                idx += 1
    return jobs


def _render_cmd_assignment(job: Job) -> List[str]:
    lines = ["  CMD=("]
    if job.method == "cvs_safd":
        prefix = ['"${PYTHON}"', "-u", '"${TRAIN_SCRIPT}"']
    else:
        prefix = ['"${PYTHON}"', "-u", "-m", job.module]
    lines.extend(f"    {word}" for word in prefix)
    lines.extend(f"    {_shell_word(arg)}" for arg in job.command_args())
    lines.append("  )")
    return lines


def render_launcher(run_id: str, jobs: Sequence[Job]) -> str:
    method_list = list(dict.fromkeys(job.method for job in jobs))
    lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"',
        'PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"',
        'TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"',
        f'RUN_ID="${{RUN_ID:-{run_id}}}"',
        'RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/${RUN_ID}}"',
        'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
        'WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"',
        'MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-3}"',
        'QUEUE_SLOT_POLL_SECONDS="${QUEUE_SLOT_POLL_SECONDS:-60}"',
        'DRY_RUN="${DRY_RUN:-0}"',
        'SKIP_DONE="${SKIP_DONE:-1}"',
        'STREAM_LOGS="${STREAM_LOGS:-0}"',
        'ONLY_JOB="${ONLY_JOB:-}"',
        "",
        'while [ "$#" -gt 0 ]; do',
        '  case "$1" in',
        "    --dry-run) DRY_RUN=1; shift ;;",
        '    --only=*) ONLY_JOB="${1#--only=}"; shift ;;',
        '    --no-skip-done) SKIP_DONE=0; shift ;;',
        '    --stream-logs) STREAM_LOGS=1; shift ;;',
        '    --max-train-per-gpu) MAX_TRAIN_PER_GPU="$2"; shift 2 ;;',
        '    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;',
        "  esac",
        "done",
        "",
        'export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"',
        'mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"',
        'STAMP="$(date +%Y%m%d_%H%M%S)"',
        'SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"',
        'MANIFEST="${RUN_ROOT}/manifest_${STAMP}.tsv"',
        'QUEUE_DIR="${LOG_ROOT}/queues_${STAMP}"',
        'QUEUE_EVENTS="${LOG_ROOT}/queue_events_${STAMP}.jsonl"',
        'printf "job_id\\tmethod\\tshots\\tprofile\\ttrain_rx_count\\ttrain_days\\ttrain_rxs\\ttest_days\\ttest_rxs\\tgpu\\tsat_train\\tlog_file\\toutput_dir\\tcommand\\n" > "${MANIFEST}"',
        ': > "${QUEUE_EVENTS}"',
        "",
        "log_msg() {",
        '  echo "$@" | tee -a "${SCHED_LOG}"',
        "}",
        "",
        "format_cmd() {",
        '  printf "%q " "$@"',
        "}",
        "",
        "gpu_process_count() {",
        '  local gpu="$1"',
        '  if ! command -v nvidia-smi >/dev/null 2>&1; then echo 0; return 0; fi',
        '  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \\',
        "    | sed '/^$/d' | wc -l | tr -d ' '",
        "}",
        "",
        "declare -A QUEUE_FILES=()",
        "QUEUE_GPUS=()",
        'QUEUE_FILE_RESULT=""',
        "",
        "queue_file_for_gpu() {",
        '  local gpu="$1"',
        '  if [ -n "${QUEUE_FILES[$gpu]:-}" ]; then',
        '    QUEUE_FILE_RESULT="${QUEUE_FILES[$gpu]}"',
        "    return 0",
        "  fi",
        '  mkdir -p "${QUEUE_DIR}"',
        '  local queue_file="${QUEUE_DIR}/gpu_${gpu}.sh"',
        '  QUEUE_FILES[$gpu]="${queue_file}"',
        '  QUEUE_GPUS+=("${gpu}")',
        "  {",
        "    printf '#!/usr/bin/env bash\\n'",
        "    printf 'set -euo pipefail\\n'",
        '    printf "GPU_ID=%q\\n" "${gpu}"',
        '    printf "MAX_ACTIVE=%q\\n" "${MAX_TRAIN_PER_GPU}"',
        '    printf "POLL_SECONDS=%q\\n" "${QUEUE_SLOT_POLL_SECONDS}"',
        '    printf "QUEUE_EVENTS=%q\\n" "${QUEUE_EVENTS}"',
        "    cat <<'EOF'",
        "",
        "json_event() {",
        '  local event="$1" job="$2" pid="${3:-}"',
        '  printf \'{"ts":"%s","gpu":"%s","event":"%s","job_id":"%s","pid":"%s","running_count":%s}\\n\' "$(date -Iseconds)" "${GPU_ID}" "${event}" "${job}" "${pid}" "${RUNNING_COUNT:-0}" >> "${QUEUE_EVENTS}"',
        "}",
        "",
        "gpu_uuid_for_index() {",
        "  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null \\",
        "    | awk -F, -v target=\"${GPU_ID}\" '{gsub(/ /,\"\",$1); gsub(/ /,\"\",$2); if ($1 == target) {print $2; exit}}'",
        "}",
        "",
        'TARGET_GPU_UUID="$(gpu_uuid_for_index)"',
        "",
        "gpu_process_count_queue() {",
        "  if ! command -v nvidia-smi >/dev/null 2>&1; then",
        "    echo 0",
        "    return 0",
        "  fi",
        '  if [ -z "${TARGET_GPU_UUID}" ]; then',
        "    echo 999",
        "    return 0",
        "  fi",
        '  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \\',
        "    | awk -F, -v uuid=\"${TARGET_GPU_UUID}\" '{gsub(/ /,\"\",$1); if ($1 == uuid) count++} END {print count + 0}'",
        "}",
        "",
        "wait_for_one_job() {",
        "  if ! wait -n; then",
        "    FAILED=1",
        "  fi",
        "  if (( RUNNING_COUNT > 0 )); then",
        "    RUNNING_COUNT=$(( RUNNING_COUNT - 1 ))",
        "  fi",
        "}",
        "",
        "wait_for_gpu_slot() {",
        "  local visible_count observed_external_count allowed_count",
        "  while true; do",
        '    visible_count="$(gpu_process_count_queue)"',
        "    observed_external_count=$(( visible_count - RUNNING_COUNT ))",
        "    if (( observed_external_count < 0 )); then",
        "      observed_external_count=0",
        "    fi",
        "    allowed_count=$(( MAX_ACTIVE - observed_external_count ))",
        "    if (( allowed_count < 0 )); then",
        "      allowed_count=0",
        "    fi",
        "    if (( RUNNING_COUNT < allowed_count )); then",
        "      break",
        "    fi",
        "    if (( RUNNING_COUNT > 0 )); then",
        "      wait_for_one_job",
        "    else",
        '      echo "[QUEUE][GPU ${GPU_ID}] waiting visible_count=${visible_count} external_count=${observed_external_count} running_count=${RUNNING_COUNT} max=${MAX_ACTIVE} at $(date +%F_%T)" >&2',
        '      sleep "${POLL_SECONDS}"',
        "    fi",
        "  done",
        "}",
        "",
        'INITIAL_EXTERNAL_COUNT="$(gpu_process_count_queue)"',
        'echo "[QUEUE][GPU ${GPU_ID}] queue_start initial_external_count=${INITIAL_EXTERNAL_COUNT} max_active=${MAX_ACTIVE} at $(date +%F_%T)"',
        'json_event queue_start queue ""',
        "RUNNING_COUNT=0",
        "FAILED=0",
        "EOF",
        '  } > "${queue_file}"',
        '  chmod +x "${queue_file}"',
        '  QUEUE_FILE_RESULT="${queue_file}"',
        "}",
        "",
        "append_to_gpu_queue() {",
        '  local gpu="$1" job_id="$2" log_file="$3" out_dir="$4"',
        "  local queue_file",
        '  queue_file_for_gpu "${gpu}"',
        '  queue_file="${QUEUE_FILE_RESULT}"',
        "  {",
        '    printf \'\\necho "[QUEUE][GPU %s] job_start %s at $(date +%%F_%%T)"\\n\' "${gpu}" "${job_id}"',
        "    printf 'wait_for_gpu_slot\\n'",
        "    printf 'json_event %q %q\\n' job_start \"${job_id}\"",
        "    printf 'mkdir -p %q\\n' \"${out_dir}\"",
        "    printf '(\\n'",
        "    printf '  set +e\\n'",
        "    printf '  json_event %q %q \\\"\\\"\\n' job_exec \"${job_id}\"",
        '    printf \'  echo "[QUEUE][GPU %s] job_exec %s at $(date +%%F_%%T)"\\n\' "${gpu}" "${job_id}"',
        '    printf "  CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 " "${gpu}"',
        '    format_cmd "${CMD[@]}"',
        '    if [ "${STREAM_LOGS}" = "1" ]; then',
        "      printf '2>&1 | tee %q\\n' \"${log_file}\"",
        "      printf '  rc=${PIPESTATUS[0]}\\n'",
        "    else",
        "      printf '> %q 2>&1\\n' \"${log_file}\"",
        "      printf '  rc=$?\\n'",
        "    fi",
        "    printf '  json_event %q %q \\\"\\\"\\n' job_done \"${job_id}\"",
        '    printf \'  echo "[QUEUE][GPU %s] job_done %s rc=${rc} at $(date +%%F_%%T)"\\n\' "${gpu}" "${job_id}"',
        "    printf '  exit \"${rc}\"\\n'",
        "    printf ') &\\n'",
        "    printf 'pid=$!\\n'",
        "    printf 'RUNNING_COUNT=$(( RUNNING_COUNT + 1 ))\\n'",
        "    printf 'json_event %q %q \"${pid}\"\\n' job_pid \"${job_id}\"",
        '    printf \'echo "[QUEUE][GPU %s] job_pid %s pid=${pid}"\\n\' "${gpu}" "${job_id}"',
        '  } >> "${queue_file}"',
        "}",
        "",
        "finalize_refill_queues() {",
        "  local gpu queue_file",
        '  for gpu in "${QUEUE_GPUS[@]}"; do',
        '    queue_file="${QUEUE_FILES[$gpu]}"',
        "    cat >> \"${queue_file}\" <<'EOF'",
        "",
        "while (( RUNNING_COUNT > 0 )); do",
        "  wait_for_one_job",
        "done",
        "json_event queue_done queue \"\"",
        'echo "[QUEUE][GPU ${GPU_ID}] queue_done failed=${FAILED} at $(date +%F_%T)"',
        'exit "${FAILED}"',
        "EOF",
        "  done",
        "}",
        "",
        "launch_refill_queues() {",
        "  local gpu queue_file queue_log pid",
        "  finalize_refill_queues",
        '  : > "${LOG_ROOT}/launch_pids.tsv"',
        '  for gpu in "${QUEUE_GPUS[@]}"; do',
        '    queue_file="${QUEUE_FILES[$gpu]}"',
        '    queue_log="${LOG_ROOT}/gpu_${gpu}_queue_${STAMP}.log"',
        '    nohup bash "${queue_file}" > "${queue_log}" 2>&1 &',
        '    pid="$!"',
        '    printf "queue\\tgpu_%s\\t%s\\t%s\\t%s\\t%s\\n" "${gpu}" "${pid}" "${queue_log}" "${queue_file}" "${QUEUE_DIR}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
        '    log_msg "[CVS-LACSR-COMP][GPU ${gpu}] refill_queue_pid=${pid} queue=${queue_file} log=${queue_log}"',
        "  done",
        "}",
        "",
        "queue_job() {",
        '  local job_id="$1" method="$2" shots="$3" profile="$4" train_rx_count="$5" train_days="$6" train_rxs="$7" test_days="$8" test_rxs="$9" gpu="${10}" sat_train="${11}"',
        '  local out_dir="${RUN_ROOT}/${job_id}"',
        '  local log_file="${LOG_ROOT}/${job_id}.log"',
        '  if [ -n "${ONLY_JOB}" ] && [ "${ONLY_JOB}" != "${job_id}" ] && [ "${ONLY_JOB}" != "${method}" ] && [ "${ONLY_JOB}" != "${profile}" ]; then',
        "    return 0",
        "  fi",
        '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${job_id}" "${method}" "${shots}" "${profile}" "${train_rx_count}" "${train_days}" "${train_rxs}" "${test_days}" "${test_rxs}" "${gpu}" "${sat_train}" "${log_file}" "${out_dir}" "$(format_cmd "${CMD[@]}")" >> "${MANIFEST}"',
        '  log_msg "[CVS-LACSR-COMP][${job_id}][GPU ${gpu}] method=${method} shots=${shots} profile=${profile} sat_train=${sat_train}"',
        '  if [ "${SKIP_DONE}" = "1" ] && { [ -f "${out_dir}/metrics.json" ] || [ -f "${out_dir}/best_strict_udu_model.pth" ]; }; then',
        '    log_msg "[CVS-LACSR-COMP][${job_id}] skip existing output=${out_dir}"',
        "    return 0",
        "  fi",
        '  if [ "${DRY_RUN}" = "1" ]; then',
        "    return 0",
        "  fi",
        '  mkdir -p "${out_dir}" "$(dirname "${log_file}")"',
        '  append_to_gpu_queue "${gpu}" "${job_id}" "${log_file}" "${out_dir}"',
        "}",
        "",
        'if [ "${DRY_RUN}" != "1" ]; then',
        '  [ -f "${TRAIN_SCRIPT}" ] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }',
        '  [ -f "${WISIG_PKL}" ] || { echo "[ERROR] missing WISIG_PKL: ${WISIG_PKL}" >&2; exit 2; }',
        "fi",
        'cd "${ROOT}"',
        f'log_msg "[CVS-LACSR-COMP] run_id=${{RUN_ID}} selected_cvs={SELECTED_CVS_ALGORITHM} jobs={len(jobs)}"',
        f'log_msg "[CVS-SAFD-COMP] shots={",".join(str(x) for x in SHOTS)} methods={",".join(method_list)} max_train_per_gpu=${{MAX_TRAIN_PER_GPU}}"',
        "",
    ]
    for job in jobs:
        lines.append(f'  out_dir="${{RUN_ROOT}}/{job.job_id}"')
        lines.extend(_render_cmd_assignment(job))
        lines.append(
            "  queue_job "
            + shell_join(
                [
                    job.job_id,
                    job.method,
                    str(job.shots),
                    job.profile_id,
                    str(job.profile.train_rx_count),
                    job.profile.train_days,
                    job.profile.train_rxs,
                    job.profile.test_days,
                    job.profile.test_rxs,
                    str(job.gpu),
                    "1" if job.sat_train else "0",
                ]
            )
        )
        lines.append("")
    lines.extend(
        [
            'if [ "${DRY_RUN}" != "1" ]; then',
            "  launch_refill_queues",
            "fi",
            'log_msg "[CVS-LACSR-COMP] queued_jobs=$(($(wc -l < "${MANIFEST}") - 1)) manifest=${MANIFEST} queue_events=${QUEUE_EVENTS}"',
            "exit 0",
        ]
    )
    return "\n".join(lines) + "\n"


def family_evidence(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))

    def family_from_name(name: str) -> str:
        if "LACSR" in name:
            return "CEN51_LACSR"
        if "FULLDG" in name:
            return "CEN51_FULLDG"
        if "RIEIFD" in name:
            return "CEN51_RIEIFD_GUARD"
        if "CEN51_LAC" in name:
            return "CEN51_LAC"
        if "CEN51_R04" in name:
            return "CEN51_R04"
        return "OTHER"

    best_by_family_shot: Dict[tuple[str, int], Dict[str, str]] = {}
    for row in rows:
        try:
            shots = int(float(row.get("shots") or "0"))
            strict = float(row.get("final_primary_strict_udu") or "nan")
        except ValueError:
            continue
        if shots not in SHOTS:
            continue
        family = family_from_name(row.get("run_name", ""))
        if family == "OTHER":
            continue
        key = (family, shots)
        old = best_by_family_shot.get(key)
        if old is None or strict > float(old.get("final_primary_strict_udu") or "-inf"):
            best_by_family_shot[key] = row

    by_family: Dict[str, List[Dict[str, str]]] = {}
    for (family, _shots), row in best_by_family_shot.items():
        by_family.setdefault(family, []).append(row)

    summary: List[Dict[str, object]] = []
    for family, family_rows in sorted(by_family.items()):
        strict_vals = [float(row.get("final_primary_strict_udu") or "nan") for row in family_rows]
        sat_min_vals = [float(row.get("final_best_sat_min") or "nan") for row in family_rows]
        summary.append(
            {
                "family": family,
                "covered_shots": sorted(int(float(row.get("shots") or 0)) for row in family_rows),
                "mean_strict_udu": round(sum(strict_vals) / len(strict_vals), 3),
                "min_strict_udu": round(min(strict_vals), 3),
                "mean_sat_min": round(sum(sat_min_vals) / len(sat_min_vals), 3),
            }
        )
    return summary


def matrix_payload(run_id: str, jobs: Sequence[Job], methods: Sequence[str]) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Compare one selected CVS low-shot algorithm against RIEI/DRIFT fixed no-sat and sat variants across shots and receiver/day stress profiles.",
        "selected_cvs_algorithm": SELECTED_CVS_ALGORITHM,
        "shots": list(SHOTS),
        "methods": list(methods),
        "max_train_per_gpu": 3,
        "profiles": [profile.__dict__ for profile in PROFILES],
        "gpu_counts": dict(Counter(job.gpu for job in jobs)),
        "jobs": [job.to_dict() for job in jobs],
    }


def render_report(run_id: str, jobs: Sequence[Job], evidence: Sequence[Dict[str, object]], methods: Sequence[str]) -> str:
    per_gpu = Counter(job.gpu for job in jobs)
    per_method = Counter(job.method for job in jobs)
    profile_lines = "\n".join(
        f"| {p.profile_id} | {p.train_rx_count} | {p.train_days} | {p.train_rxs} | {p.test_days} | {p.test_rxs} | {p.rationale} |"
        for p in PROFILES
    )
    method_lines = "\n".join(f"| {method} | {per_method[method]} |" for method in methods)
    gpu_lines = "\n".join(f"| {gpu} | {per_gpu[gpu]} | 3 |" for gpu in sorted(per_gpu))
    evidence_lines = "\n".join(
        f"| {row['family']} | {','.join(str(x) for x in row['covered_shots'])} | {row['mean_strict_udu']} | {row['min_strict_udu']} | {row['mean_sat_min']} |"
        for row in evidence
    )
    if not evidence_lines:
        evidence_lines = "| not_loaded | n/a | n/a | n/a | n/a |"

    return f"""# CVS-SAFD vs RIEI/DRIFT Comprehensive Matrix

## Run
- run_id: `{run_id}`
- operator/agent: Codex main agent with read-only subagent scouts
- objective: run the selected strongest CVS low-shot algorithm family, then compare it offline with already completed RIEI/DRIFT fixed no-sat and sat variants.
- selected CVS algorithm: `{SELECTED_CVS_ALGORITHM}` from this script's `safd_fulldg_params()` controller prior.
- important correction: this is not a per-shot CVS_BEST_CHAIN. K=10/20/30/50/100 all use the same `CEN51_SAFD_FULLDG` method family.

## Matrix
- shots: `10,20,30,50,100`
- methods in this launch: `{','.join(methods)}`
- profiles: 8 receiver/day stress profiles
- total jobs: `{len(jobs)}`
- scheduler: one refill queue per GPU; each queue allows `MAX_TRAIN_PER_GPU=3`; when one job finishes, `wait -n` frees a slot and the next queued job starts.

| method | jobs |
| --- | ---: |
{method_lines}

| profile | train_rx_count | train_days | train_rxs | test_days | test_rxs | rationale |
| --- | ---: | --- | --- | --- | --- | --- |
{profile_lines}

| gpu | queued_jobs | concurrent_cap |
| ---: | ---: | ---: |
{gpu_lines}

## CVS Selection Evidence
The local candidate summary supports using one low-shot family instead of mixing shot-specific winners:

| family | covered_shots | mean_strict_udu | min_strict_udu | mean_sat_min |
| --- | --- | ---: | ---: | ---: |
{evidence_lines}

Decision: `CEN51_SAFD_FULLDG` is the current strongest CVS few-shot direction below K100 because it keeps the CEN51 backbone, keeps satellite view in the full DG loss path, and uses a unified shot-aware controller prior instead of fixed LACSR rescue parameters or a per-shot best-chain. At K>=100, CVS restores the original `CEN51_R04_sat_joint_guard_no_overdrive_r010` ratio path.

## Key Configuration
- protocol: `cvs_day_rx`
- train ratio: `0.1`
- validation ratio: `-1.0` so baseline data loading does not override the train ratio
- exact few-shot cap: `--wisig_max_train_per_combo K` for few-shot jobs below K100; CVS K>=100 omits the cap and restores the original CEN51_R04 ratio path
- test receivers: `7,8,9,10,11`
- test days: `2,3`
- seed: `1337`
- baseline epochs: `200`
- RIEI fixed delta: `--lambda_feature_norm 0.0001`
- DRIFT fixed delta: `--mse_cap 4000`
- no-sat variants: satellite evaluation enabled, but no satellite view augmentation during training
- CVS-SAFD below K100: full-DG satellite view augmentation enabled during training and satellite evaluation enabled; CVS K>=100: original CEN51_R04 satellite schedule and loss stack

## Local Files
- generator: `tools/cvs_lacsr_riei_drift_comprehensive_matrix.py`
- launcher: `code/scripts/launch_{run_id}.sh`
- matrix json: `automation_reports/CV-SincNet/{run_id}/artifacts/comprehensive_matrix.json`
- planned jobs TSV: `automation_reports/CV-SincNet/{run_id}/artifacts/planned_jobs.tsv`
- supervisor handoff: `automation_reports/CV-SincNet/{run_id}/supervisor_handoff.md`

## Launch Contract
Remote command after sync:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
bash code/scripts/launch_{run_id}.sh --max-train-per-gpu 3
```

Expected remote outputs:
- `runs/{run_id}/manifest_<timestamp>.tsv`
- `logs/{run_id}/launch_pids.tsv`
- `logs/{run_id}/queue_events_<timestamp>.jsonl`
- `logs/{run_id}/gpu_<gpu>_queue_<timestamp>.log`
- one run directory per job under `runs/{run_id}/`

## Risks And Follow-Up
- This launch is CVS-only when generated with `--methods cvs_safd`: 40 jobs, 5 queued per GPU. Queue workers should remain running until their GPU queue drains.
- A one-day profile is included as `rx7_all_d0`; other receiver-count and identity profiles keep two training days.
- If a job fails, keep its log and queue event entry; do not delete or overwrite outputs. Requeue failed job ids explicitly with `ONLY_JOB=<job_id>` after diagnosis.
"""


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_outputs(
    run_id: str,
    force: bool = False,
    exclude_job_ids: Iterable[str] | None = None,
    methods: Sequence[str] = METHODS,
) -> Dict[str, str]:
    excluded = set(exclude_job_ids or [])
    jobs = make_jobs(exclude_job_ids=excluded, methods=methods)
    report_dir = Path("automation_reports") / "CV-SincNet" / run_id
    artifacts_dir = report_dir / "artifacts"
    script_path = Path("code") / "scripts" / f"launch_{run_id}.sh"
    report_path = report_dir / "report.md"
    matrix_path = artifacts_dir / "comprehensive_matrix.json"
    planned_tsv_path = artifacts_dir / "planned_jobs.tsv"
    handoff_path = report_dir / "supervisor_handoff.md"

    for path in (script_path, report_path, matrix_path, planned_tsv_path, handoff_path):
        if path.exists() and not force:
            raise SystemExit(f"refusing to overwrite existing file without --force: {path}")

    evidence = family_evidence(Path("analysis_tmp") / "riei_drift_cvs_comparison_20260610" / "cvs_candidate_results.csv")
    payload = matrix_payload(run_id, jobs, methods)
    payload["selection_evidence"] = evidence
    payload["excluded_job_ids"] = sorted(excluded)

    write_text_lf(script_path, render_launcher(run_id, jobs))
    write_text_lf(report_path, render_report(run_id, jobs, evidence, methods))
    write_text_lf(matrix_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with planned_tsv_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "method",
                "shots",
                "profile_id",
                "train_rx_count",
                "train_days",
                "train_rxs",
                "test_days",
                "test_rxs",
                "gpu",
                "sat_train",
                "module",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "method": job.method,
                    "shots": job.shots,
                    "profile_id": job.profile_id,
                    "train_rx_count": job.profile.train_rx_count,
                    "train_days": job.profile.train_days,
                    "train_rxs": job.profile.train_rxs,
                    "test_days": job.profile.test_days,
                    "test_rxs": job.profile.test_rxs,
                    "gpu": job.gpu,
                    "sat_train": int(job.sat_train),
                    "module": job.module,
                }
            )
    write_text_lf(
        handoff_path,
        f"""# Supervisor Handoff

- run_id: `{run_id}`
- selected CVS algorithm: `{SELECTED_CVS_ALGORITHM}`
- total jobs: `{len(jobs)}`
- per GPU cap: `3`
- monitor without killing or modifying jobs.
- remote paths:
  - launcher: `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_{run_id}.sh`
  - logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}`
  - runs: `/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}`

Read-only monitor checks:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
test -d logs/{run_id} && ls -1 logs/{run_id}
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
test -f logs/{run_id}/launch_pids.tsv && cat logs/{run_id}/launch_pids.tsv
find logs/{run_id} -maxdepth 1 -name 'gpu_*_queue_*.log' -print -exec tail -n 20 {{}} \\;
find logs/{run_id} -maxdepth 1 -name 'queue_events_*.jsonl' -print -exec tail -n 20 {{}} \\;
```
""",
    )

    return {
        "script": str(script_path),
        "report": str(report_path),
        "matrix": str(matrix_path),
        "planned_jobs": str(planned_tsv_path),
        "handoff": str(handoff_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"{ROOT_RUN_ID_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--write", action="store_true", help="write report, matrix, planned jobs TSV, and launcher")
    parser.add_argument("--force", action="store_true", help="overwrite generated files for the same run id")
    parser.add_argument("--exclude-job-ids", default="", help="comma-separated job ids to omit from this launcher")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated methods to include, e.g. cvs_safd")
    parser.add_argument("--print-summary", action="store_true", help="print matrix summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    excluded = {item.strip() for item in args.exclude_job_ids.split(",") if item.strip()}
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; allowed={','.join(METHODS)}")
    jobs = make_jobs(exclude_job_ids=excluded, methods=methods)
    if args.print_summary or not args.write:
        per_gpu = Counter(job.gpu for job in jobs)
        print(f"run_id={args.run_id}")
        print(f"selected_cvs_algorithm={SELECTED_CVS_ALGORITHM}")
        print(f"jobs={len(jobs)} shots={','.join(str(x) for x in SHOTS)} methods={','.join(methods)}")
        print("per_gpu=" + ",".join(f"{gpu}:{count}" for gpu, count in sorted(per_gpu.items())))
    if args.write:
        paths = write_outputs(args.run_id, force=args.force, exclude_job_ids=excluded, methods=methods)
        for key, value in paths.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
