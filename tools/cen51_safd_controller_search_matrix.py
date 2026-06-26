#!/usr/bin/env python3
"""Generate a queued six-hour CEN51-SAFD diagnostic-controller search matrix.

The matrix is designed for mechanism search, not final promotion. It probes how
the SAFD controller should react to sample count, domain-label choice,
train-day coverage, train-receiver coverage, and the dominant SAFD deficits.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from cen51_lowshot_config_search import arg_pairs
from cen51_r04_config import cen51_r04_ratio_params, should_restore_cen51_r04


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
ALL_SAT = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LIGHT_SAT = "clear_leo,mixed_orbit"
DEFAULT_TRAIN_RXS = "0,1,2,3,4,5,6"
DEFAULT_TEST_RXS = "7,8,9,10,11"


@dataclass(frozen=True)
class Experiment:
    cid: str
    run_name: str
    axis: str
    actuator: str
    shots: int
    gpu: int
    seed: int
    epochs: int
    swad_start: int
    batch_size: int
    wisig_domain: str
    train_days: str
    test_days: str
    train_rxs: str
    test_rxs: str
    split_strategy: str
    cap_strategy: str
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
    hypothesis: str
    success_gate: str


def q(value: object) -> str:
    return shlex.quote(str(value))


def bash_items(items: Iterable[object], indent: str = "  ") -> str:
    return "".join(f"{indent}{q(item)}\n" for item in items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    parser.add_argument("--max-active-per-gpu", type=int, default=3)
    parser.add_argument("--scheduler-hours", type=float, default=6.0)
    return parser.parse_args()


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
        "--wisig_equalized",
        "1",
        "--wisig_train_ratio",
        "0.1",
        "--wisig_val_ratio",
        "-1.0",
        "--wisig_guard_gap",
        "8",
        "--test_eval_policy",
        "interval_final",
        "--test_eval_start_epoch",
        "31",
        "--test_eval_interval",
        "20",
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


def prior(shots: int) -> dict[str, object]:
    if should_restore_cen51_r04(shots):
        return {
            "epochs": 200,
            "swad_start": 70,
            "sat_prob": 1.0,
            "sat_start": 1,
            "sat_scenarios": ALL_SAT,
            "lambda_dom": 1.0,
            "lambda_adv": 0.5,
            "lambda_orth": 0.05,
            "lambda_cons": 0.1,
            "lambda_group_ce": 0.088,
            "lambda_proto": 0.016,
            "lambda_supcon": 0.022,
            "lambda_fishr": 0.002,
            "lambda_feature_norm": 0.0,
            "feature_norm_mode": "l2",
            "feature_norm_target": 0.0,
            "group_top_frac": 0.20,
            "group_tau": 0.37,
            "group_cap": 0.48,
            "use_aug": True,
            "use_mixstyle": True,
            "mixstyle_p": 0.18,
            "mixstyle_strength": 0.70,
            "aug_scale_min": 0.10,
            "aug_scale_max": 0.35,
        }
    if shots <= 5:
        return {
            "epochs": 115,
            "swad_start": 45,
            "sat_prob": 0.06,
            "sat_start": 1,
            "sat_scenarios": LIGHT_SAT,
            "lambda_dom": 0.34,
            "lambda_adv": 0.12,
            "lambda_orth": 0.012,
            "lambda_cons": 0.004,
            "lambda_group_ce": 0.004,
            "lambda_proto": 0.0006,
            "lambda_supcon": 0.0006,
            "lambda_fishr": 0.0,
            "lambda_feature_norm": 0.0008,
            "feature_norm_mode": "hinge",
            "feature_norm_target": 6.0,
            "group_top_frac": 0.14,
            "group_tau": 0.32,
            "group_cap": 0.40,
            "use_aug": False,
            "use_mixstyle": False,
            "mixstyle_p": 0.0,
            "mixstyle_strength": 0.0,
            "aug_scale_min": 0.0,
            "aug_scale_max": 0.0,
        }
    if shots <= 15:
        return {
            "epochs": 125,
            "swad_start": 50,
            "sat_prob": 0.10,
            "sat_start": 70,
            "sat_scenarios": LIGHT_SAT,
            "lambda_dom": 0.42,
            "lambda_adv": 0.12,
            "lambda_orth": 0.015,
            "lambda_cons": 0.004,
            "lambda_group_ce": 0.006,
            "lambda_proto": 0.0008,
            "lambda_supcon": 0.0008,
            "lambda_fishr": 0.0,
            "lambda_feature_norm": 0.00045,
            "feature_norm_mode": "hinge",
            "feature_norm_target": 8.5,
            "group_top_frac": 0.16,
            "group_tau": 0.35,
            "group_cap": 0.42,
            "use_aug": False,
            "use_mixstyle": False,
            "mixstyle_p": 0.0,
            "mixstyle_strength": 0.0,
            "aug_scale_min": 0.0,
            "aug_scale_max": 0.0,
        }
    if shots <= 25:
        return {
            "epochs": 130,
            "swad_start": 55,
            "sat_prob": 0.12,
            "sat_start": 1,
            "sat_scenarios": LIGHT_SAT,
            "lambda_dom": 0.42,
            "lambda_adv": 0.15,
            "lambda_orth": 0.020,
            "lambda_cons": 0.008,
            "lambda_group_ce": 0.010,
            "lambda_proto": 0.0015,
            "lambda_supcon": 0.0015,
            "lambda_fishr": 0.0,
            "lambda_feature_norm": 0.00004,
            "feature_norm_mode": "l2",
            "feature_norm_target": 0.0,
            "group_top_frac": 0.18,
            "group_tau": 0.37,
            "group_cap": 0.48,
            "use_aug": False,
            "use_mixstyle": False,
            "mixstyle_p": 0.0,
            "mixstyle_strength": 0.0,
            "aug_scale_min": 0.0,
            "aug_scale_max": 0.0,
        }
    if shots <= 40:
        return {
            "epochs": 135,
            "swad_start": 60,
            "sat_prob": 0.18,
            "sat_start": 25,
            "sat_scenarios": LIGHT_SAT,
            "lambda_dom": 0.50,
            "lambda_adv": 0.20,
            "lambda_orth": 0.024,
            "lambda_cons": 0.012,
            "lambda_group_ce": 0.018,
            "lambda_proto": 0.0025,
            "lambda_supcon": 0.0025,
            "lambda_fishr": 0.0002,
            "lambda_feature_norm": 0.00004,
            "feature_norm_mode": "l2",
            "feature_norm_target": 0.0,
            "group_top_frac": 0.20,
            "group_tau": 0.39,
            "group_cap": 0.52,
            "use_aug": True,
            "use_mixstyle": True,
            "mixstyle_p": 0.025,
            "mixstyle_strength": 0.24,
            "aug_scale_min": 0.10,
            "aug_scale_max": 0.32,
        }
    return {
        "epochs": 145,
        "swad_start": 65,
        "sat_prob": 0.22 if shots < 100 else 0.18,
        "sat_start": 25 if shots < 100 else 35,
        "sat_scenarios": ALL_SAT,
        "lambda_dom": 0.52 if shots < 100 else 0.46,
        "lambda_adv": 0.24 if shots < 100 else 0.20,
        "lambda_orth": 0.025,
        "lambda_cons": 0.012,
        "lambda_group_ce": 0.026 if shots < 100 else 0.020,
        "lambda_proto": 0.0040 if shots < 100 else 0.0030,
        "lambda_supcon": 0.0040 if shots < 100 else 0.0030,
        "lambda_fishr": 0.0003,
        "lambda_feature_norm": 0.00003,
        "feature_norm_mode": "l2",
        "feature_norm_target": 0.0,
        "group_top_frac": 0.22 if shots < 100 else 0.18,
        "group_tau": 0.40,
        "group_cap": 0.55,
        "use_aug": True,
        "use_mixstyle": True,
        "mixstyle_p": 0.070 if shots < 100 else 0.045,
        "mixstyle_strength": 0.20,
        "aug_scale_min": 0.05,
        "aug_scale_max": 0.22,
    }


def apply_actuator(params: dict[str, object], actuator: str) -> dict[str, object]:
    p = dict(params)
    if actuator == "r04_restore":
        return p
    if actuator == "rx_repair":
        p["lambda_dom"] = float(p["lambda_dom"]) + 0.04
        p["lambda_adv"] = float(p["lambda_adv"]) + 0.04
        p["lambda_group_ce"] = float(p["lambda_group_ce"]) * 1.45
        p["lambda_orth"] = float(p["lambda_orth"]) * 1.15
        p["group_top_frac"] = min(0.30, float(p["group_top_frac"]) + 0.04)
        p["group_cap"] = min(0.62, float(p["group_cap"]) + 0.04)
    elif actuator == "sat_repair":
        p["sat_prob"] = min(0.30, float(p["sat_prob"]) * 1.50 + 0.02)
        p["sat_start"] = max(1, min(int(p["sat_start"]), 25))
        p["sat_scenarios"] = ALL_SAT
        p["lambda_cons"] = float(p["lambda_cons"]) + 0.002
    elif actuator == "late_repair":
        p["swad_start"] = max(40, int(p["swad_start"]) - 10)
        p["sat_prob"] = max(0.05, float(p["sat_prob"]) * 0.80)
        p["sat_start"] = max(int(p["sat_start"]), 55)
        p["mixstyle_p"] = float(p["mixstyle_p"]) * 0.50
        p["aug_scale_max"] = min(float(p["aug_scale_max"]), 0.20)
    elif actuator == "val_repair":
        for key in ("lambda_dom", "lambda_adv", "lambda_group_ce", "lambda_proto", "lambda_supcon", "lambda_fishr"):
            p[key] = float(p[key]) * 0.72
        p["sat_prob"] = max(0.04, float(p["sat_prob"]) * 0.70)
        p["sat_start"] = max(int(p["sat_start"]), 80)
        p["use_aug"] = False
        p["use_mixstyle"] = False
        p["mixstyle_p"] = 0.0
        p["aug_scale_min"] = 0.0
        p["aug_scale_max"] = 0.0
    elif actuator == "clean_repair":
        p["lambda_cons"] = float(p["lambda_cons"]) * 1.30 + 0.001
        p["lambda_proto"] = float(p["lambda_proto"]) * 1.25
        p["lambda_supcon"] = float(p["lambda_supcon"]) * 1.25
        p["lambda_adv"] = float(p["lambda_adv"]) + 0.02
    elif actuator == "prior":
        pass
    else:
        raise ValueError(f"unknown actuator: {actuator}")
    return p


def exp(
    cid: str,
    axis: str,
    actuator: str,
    shots: int,
    seed: int,
    hypothesis: str,
    success_gate: str,
    *,
    wisig_domain: str = "rx_day",
    train_days: str = "0,1",
    test_days: str = "2,3",
    train_rxs: str = DEFAULT_TRAIN_RXS,
    test_rxs: str = DEFAULT_TEST_RXS,
    split_strategy: str = "random",
    cap_strategy: str = "random",
) -> Experiment:
    p = apply_actuator(prior(shots), actuator)
    return Experiment(
        cid=cid,
        run_name=f"CEN51_SAFD6H_{cid}",
        axis=axis,
        actuator=actuator,
        shots=shots,
        gpu=-1,
        seed=seed,
        batch_size=128,
        wisig_domain=wisig_domain,
        train_days=train_days,
        test_days=test_days,
        train_rxs=train_rxs,
        test_rxs=test_rxs,
        split_strategy=split_strategy,
        cap_strategy=cap_strategy,
        hypothesis=hypothesis,
        success_gate=success_gate,
        **p,
    )


def assign_gpus(rows: Sequence[Experiment]) -> list[Experiment]:
    assigned: list[Experiment] = []
    counts = {gpu: 0 for gpu in range(8)}
    for idx, row in enumerate(rows):
        gpu = idx % 8
        counts[gpu] += 1
        assigned.append(replace(row, gpu=gpu))
    assert all(value == len(rows) // 8 for value in counts.values()), counts
    return assigned


def make_candidates() -> list[Experiment]:
    rows: list[Experiment] = []

    for shots, seed in [(5, 2030), (10, 2030), (15, 2029), (20, 2028), (30, 2029), (50, 1337), (80, 2029), (100, 1337)]:
        if should_restore_cen51_r04(shots):
            rows.append(
                exp(
                    f"A_R04_GE{shots:03d}_S{seed}",
                    "restore_original",
                    "r04_restore",
                    shots,
                    seed,
                    "Boundary control: K>=100 must restore the original CEN51_R04 ratio path.",
                    "Command must match CEN51_R04 ratio=0.1 behavior: no per-combo cap, no SAFD low-shot overrides.",
                )
            )
        else:
            rows.append(exp(f"A_PRIOR_K{shots:03d}_S{seed}", "quantity_prior", "prior", shots, seed, "Map SAFD prior behavior as per-combo sample count changes.", "No collapse; report D-vector rather than promotion."))

    actuator_rows = [
        (5, 2028, "late_repair"),
        (5, 2029, "val_repair"),
        (5, 2030, "clean_repair"),
        (10, 2028, "sat_repair"),
        (10, 2028, "rx_repair"),
        (10, 2029, "clean_repair"),
        (10, 2030, "val_repair"),
        (20, 1337, "rx_repair"),
        (20, 2029, "late_repair"),
        (20, 2028, "sat_repair"),
        (20, 2029, "val_repair"),
        (30, 2030, "rx_repair"),
        (30, 2028, "clean_repair"),
        (30, 2030, "val_repair"),
        (50, 2028, "rx_repair"),
        (50, 2029, "late_repair"),
    ]
    for shots, seed, actuator in actuator_rows:
        rows.append(exp(f"B_{actuator.upper()}_K{shots:03d}_S{seed}", "deficit_actuator", actuator, shots, seed, f"Test whether {actuator} reduces the matching SAFD deficit.", "Matching deficit should improve without >0.5pp clean strict regression vs prior."))

    for shots, seed in [(10, 2028), (30, 2029), (50, 2029)]:
        rows.append(exp(f"C_DOMAIN_RX_K{shots:03d}_S{seed}", "domain_mode", "prior", shots, seed, "Use receiver-only domain labels to test whether day labels inject noise under few-shot.", "Compare D_rx and D_clean against rx_day prior.", wisig_domain="rx"))
        rows.append(exp(f"C_DOMAIN_DAY_K{shots:03d}_S{seed}", "domain_mode", "prior", shots, seed, "Use day-only domain labels to test whether receiver labels overconstrain the embedding.", "Compare D_late and D_rx against rx_day prior.", wisig_domain="day"))

    day_profiles = [
        ("DAY0", "0", "1,2,3", "single train day exposes whether date diversity is required."),
        ("DAY12", "1,2", "0,3", "shifted two-day source tests whether the 0,1 default is special."),
        ("DAY012", "0,1,2", "3", "three source days test whether more source dates reduce shortcut without harming identity."),
    ]
    for shots, seed in [(10, 2030), (30, 2029), (50, 1337)]:
        for tag, train_days, test_days, note in day_profiles:
            rows.append(exp(f"D_{tag}_K{shots:03d}_S{seed}", "day_coverage", "prior", shots, seed, note, "Track Q_val, Q_clean, Q_rx, and Q_late under day coverage shift.", train_days=train_days, test_days=test_days))

    rx_profiles = [
        ("RX4", "0,1,2,3", "7,8,9,10,11", "few train receivers test receiver shortcut pressure."),
        ("RXSPARSE4", "0,2,4,6", "7,8,9,10,11", "sparse train receivers test receiver diversity vs count."),
        ("RX9", "0,1,2,3,4,5,6,7,8", "9,10,11", "more train receivers test whether RX floor improves with source diversity."),
    ]
    for shots, seed in [(10, 2028), (30, 2030), (50, 2029)]:
        for tag, train_rxs, test_rxs, note in rx_profiles:
            rows.append(exp(f"E_{tag}_K{shots:03d}_S{seed}", "rx_coverage", "prior", shots, seed, note, "Track D_rx, D_val, and strict UDU under receiver coverage shift.", train_rxs=train_rxs, test_rxs=test_rxs))

    assert len(rows) == 48, len(rows)
    return assign_gpus(rows)


def candidate_args(c: Experiment) -> list[str]:
    args = [
        "--wisig_domain",
        c.wisig_domain,
        "--wisig_split_strategy",
        c.split_strategy,
        "--wisig_cap_strategy",
        c.cap_strategy,
        "--wisig_train_days",
        c.train_days,
        "--wisig_test_days",
        c.test_days,
        "--wisig_train_rxs",
        c.train_rxs,
        "--wisig_test_rxs",
        c.test_rxs,
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
    args.append("--use_proto_memory" if c.lambda_proto > 0.0 else "--no_use_proto_memory")
    return args


def command_for(run_id: str, c: Experiment) -> str:
    run_dir = f"{REMOTE_ROOT}/runs/{run_id}/{c.run_name}"
    if should_restore_cen51_r04(c.shots):
        params = cen51_r04_ratio_params(seed=c.seed)
        params.update(
            {
                "run_name": c.run_name,
                "latest_save_path": f"{run_dir}/latest_model.pth",
                "best_save_path": f"{run_dir}/best_val_model.pth",
                "best_primary_save_path": f"{run_dir}/best_primary_ood_model.pth",
                "best_unseen_day_unseen_rx_save_path": f"{run_dir}/best_strict_udu_model.pth",
                "best_worst_rx_save_path": f"{run_dir}/best_worst_rx_model.pth",
                "ema_save_path": f"{run_dir}/ema_model.pth",
                "swa_save_path": f"{run_dir}/swa_model.pth",
                "swad_save_path": f"{run_dir}/swad_model.pth",
            }
        )
        args = [
            "env",
            f"CUDA_VISIBLE_DEVICES={c.gpu}",
            f"PYTHONPATH={REMOTE_ROOT}/code:{REMOTE_ROOT}",
            "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
            "-u",
            f"{REMOTE_ROOT}/code/train.py",
            *arg_pairs(params),
        ]
        return " ".join(q(item) for item in args)

    args = [
        "env",
        f"CUDA_VISIBLE_DEVICES={c.gpu}",
        f"PYTHONPATH={REMOTE_ROOT}/code:{REMOTE_ROOT}",
        "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
        "-u",
        f"{REMOTE_ROOT}/code/train.py",
        *base_args(),
        *candidate_args(c),
        "--run_name",
        c.run_name,
        "--wisig_max_train_per_combo",
        str(c.shots),
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


def render_launcher(run_id: str, candidates: Sequence[Experiment], max_active_per_gpu: int, scheduler_hours: float) -> str:
    max_seconds = int(scheduler_hours * 3600)
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
        f'MAX_ACTIVE_PER_GPU="${{MAX_ACTIVE_PER_GPU:-{max_active_per_gpu}}}"',
        f'MAX_SCHEDULER_SECONDS="${{MAX_SCHEDULER_SECONDS:-{max_seconds}}}"',
        'POLL_SECONDS="${POLL_SECONDS:-60}"',
        'DRY_RUN="${DRY_RUN:-0}"',
        "",
        'for arg in "$@"; do',
        '  case "${arg}" in',
        '    --dry-run) DRY_RUN=1 ;;',
        '    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;',
        "  esac",
        "done",
        "",
        "gpu_process_count() {",
        '  local gpu="$1"',
        '  if [[ "${DRY_RUN}" == "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then echo 0; return 0; fi',
        '  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed \'/^$/d\' | wc -l | tr -d \' \' || echo 0',
        "}",
        "",
        "BASE_ARGS=(",
        bash_items(base_args()).rstrip(),
        ")",
        "",
        "CAND_ID=()",
        "CAND_RUN=()",
        "CAND_AXIS=()",
        "CAND_ACTUATOR=()",
        "CAND_SHOTS=()",
        "CAND_GPU=()",
        "CAND_CMD=()",
        "STATUS=()",
        "PID=()",
        "LOG_PATH=()",
        "",
        "add_candidate() {",
        '  CAND_ID+=("$1"); CAND_RUN+=("$2"); CAND_AXIS+=("$3"); CAND_ACTUATOR+=("$4"); CAND_SHOTS+=("$5"); CAND_GPU+=("$6"); CAND_CMD+=("$7")',
        '  STATUS+=("queued"); PID+=(""); LOG_PATH+=("${LOG_ROOT}/$2.out")',
        "}",
        "",
    ]
    for c in candidates:
        lines.append(
            "add_candidate "
            f"{q(c.cid)} {q(c.run_name)} {q(c.axis)} {q(c.actuator)} {q(c.shots)} {q(c.gpu)} {q(command_for(run_id, c))}"
        )
    lines.extend(
        [
            "",
            "launch_idx() {",
            '  local i="$1" run="${CAND_RUN[$1]}" gpu="${CAND_GPU[$1]}" log_path="${LOG_PATH[$1]}" run_dir="${RUNS_ROOT}/${run}"',
            '  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then',
            '    STATUS[$i]="blocked_path"; printf "%s\\t%s\\tBLOCKED_PATH\\t%s\\t%s\\n" "${CAND_ID[$i]}" "${run}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"; return 0',
            "  fi",
            '  mkdir -p "${LOG_ROOT}" "${run_dir}"',
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\tSTART\\t%s\\n" "$(date -Is)" "${CAND_ID[$i]}" "${run}" "${gpu}" "${CAND_SHOTS[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            '  if [[ "${DRY_RUN}" == "1" ]]; then echo "[DRY-RUN] ${CAND_CMD[$i]}"; STATUS[$i]="dry_run"; return 0; fi',
            '  bash -lc "${CAND_CMD[$i]}" > "${log_path}" 2>&1 &',
            '  PID[$i]="$!"; STATUS[$i]="running"',
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${CAND_ID[$i]}" "${run}" "${CAND_SHOTS[$i]}" "${gpu}" "${PID[$i]}" "${CAND_AXIS[$i]}" "${CAND_ACTUATOR[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
            "}",
            "",
            "reap_finished() {",
            "  local i rc",
            '  for i in "${!STATUS[@]}"; do',
            '    if [[ "${STATUS[$i]}" == "running" ]]; then',
            '      if ! kill -0 "${PID[$i]}" 2>/dev/null; then',
            '        if wait "${PID[$i]}"; then rc=0; else rc="$?"; fi',
            '        STATUS[$i]="done_${rc}"',
            '        printf "%s\\t%s\\t%s\\t%s\\t%s\\tDONE\\trc=%s\\n" "$(date -Is)" "${CAND_ID[$i]}" "${CAND_RUN[$i]}" "${CAND_GPU[$i]}" "${CAND_SHOTS[$i]}" "${rc}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "      fi",
            "    fi",
            "  done",
            "}",
            "",
            "queued_left() {",
            "  local i n=0",
            '  for i in "${!STATUS[@]}"; do [[ "${STATUS[$i]}" == "queued" ]] && n=$((n + 1)); done',
            '  echo "${n}"',
            "}",
            "",
            "running_left() {",
            "  local i n=0",
            '  for i in "${!STATUS[@]}"; do [[ "${STATUS[$i]}" == "running" ]] && n=$((n + 1)); done',
            '  echo "${n}"',
            "}",
            "",
            "launch_available() {",
            "  local gpu i current launched capacity",
            "  for gpu in 0 1 2 3 4 5 6 7; do",
            '    current="$(gpu_process_count "${gpu}")"',
            '    [[ "${current}" =~ ^[0-9]+$ ]] || current=0',
            "    capacity=$((MAX_ACTIVE_PER_GPU - current))",
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
            '[[ -f "${TRAIN_SCRIPT}" || "${DRY_RUN}" == "1" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }',
            'cd "${ROOT}"',
            'echo "[CEN51-SAFD6H] run_id=${RUN_ID} dry_run=${DRY_RUN} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} max_seconds=${MAX_SCHEDULER_SECONDS}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            'for gpu in 0 1 2 3 4 5 6 7; do echo "[CEN51-SAFD6H] initial gpu=${gpu} count=$(gpu_process_count "${gpu}")" | tee -a "${LOG_ROOT}/scheduler_events.tsv"; done',
            'START_TS="$(date +%s)"',
            "while true; do",
            "  reap_finished",
            '  NOW_TS="$(date +%s)"',
            "  if (( NOW_TS - START_TS < MAX_SCHEDULER_SECONDS )); then",
            "    launch_available",
            "  fi",
            '  q_left="$(queued_left)"; r_left="$(running_left)"',
            '  echo "[CEN51-SAFD6H] heartbeat=$(date -Is) queued=${q_left} running=${r_left}" | tee -a "${LOG_ROOT}/scheduler_heartbeat.log"',
            '  if [[ "${q_left}" == "0" && "${r_left}" == "0" ]]; then break; fi',
            '  if [[ "${DRY_RUN}" == "1" ]]; then break; fi',
            '  sleep "${POLL_SECONDS}"',
            "done",
            'echo "[CEN51-SAFD6H] scheduler_complete $(date -Is)" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "",
        ]
    )
    return "\n".join(lines)


def render_report(run_id: str, candidates: Sequence[Experiment], script_path: Path, matrix_path: Path, max_active_per_gpu: int, scheduler_hours: float) -> str:
    gpu_counts: dict[int, int] = {}
    axis_counts: dict[str, int] = {}
    shot_counts: dict[int, int] = {}
    for c in candidates:
        gpu_counts[c.gpu] = gpu_counts.get(c.gpu, 0) + 1
        axis_counts[c.axis] = axis_counts.get(c.axis, 0) + 1
        shot_counts[c.shots] = shot_counts.get(c.shots, 0) + 1

    rows = [
        "| ID | axis | actuator | K | GPU | seed | data profile | success gate |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for c in candidates:
        data = f"domain={c.wisig_domain}; days={c.train_days}->{c.test_days}; rxs={c.train_rxs}->{c.test_rxs}"
        rows.append(f"| `{c.cid}` | {c.axis} | {c.actuator} | {c.shots} | {c.gpu} | {c.seed} | {data} | {c.success_gate} |")

    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## Objective",
            "",
            "Six-hour queued search for the CEN51-SAFD Diagnostic Controller. This is a mechanism-search batch, not a final promotion batch. Each candidate is designed to expose which SAFD deficit should trigger which training actuator.",
            "",
            "## Capacity Plan",
            "",
            f"- candidates: {len(candidates)}",
            f"- max active per GPU: {max_active_per_gpu}",
            f"- scheduler launch window: about {scheduler_hours:.1f} hours; after that no new jobs are launched, existing jobs are allowed to finish",
            f"- GPU candidate counts: `{json.dumps(gpu_counts, sort_keys=True)}`",
            f"- axis counts: `{json.dumps(axis_counts, sort_keys=True)}`",
            f"- shot counts: `{json.dumps(shot_counts, sort_keys=True)}`",
            "",
            "## Experimental Axes",
            "",
        "- `quantity_prior`: map K=5/10/15/20/30/50/80 to a continuous controller prior.",
        "- `restore_original`: K>=100 boundary control; restore original CEN51_R04 ratio path exactly.",
            "- `deficit_actuator`: test whether the SAFD repair actions actually reduce the intended deficit.",
            "- `domain_mode`: compare `rx_day`, `rx`, and `day` domain labels.",
            "- `day_coverage`: change the number and identity of source days.",
            "- `rx_coverage`: change the number and diversity of source receivers.",
            "",
            "All candidates keep CVS/CEN51 as the backbone and keep satellite augmentation as full-DG concat view. No candidate uses `--concat_sat_ce_only`.",
            "",
            "## Candidate Matrix",
            "",
            *rows,
            "",
            "## Validation",
            "",
            "After completion, pull full stdout logs and run:",
            "",
            "```powershell",
            "conda activate ssr-gpu",
            f"python tools\\cen51_fewshot_stability_validator.py --log-dir <local-log-dir> --matrix-json {matrix_path} --out-dir analysis_tmp\\{run_id}\\stability_validation --late-window 30 --no-fail",
            f"python tools\\cen51_safd_score.py --summary-csv analysis_tmp\\{run_id}\\stability_validation\\stability_summary.csv --out-dir analysis_tmp\\{run_id}\\safd_score",
            "```",
            "",
            "## Local/Remote Artifacts",
            "",
            f"- launcher: `{script_path}`",
            f"- matrix: `{matrix_path}`",
            "- remote root: `/home/szu2070436088/2510044040/CV-SincNet`",
            "- expected remote logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}`",
            "- expected remote runs: `/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}`",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_safd_controller_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidates = make_candidates()
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"
    manifest_path = report_dir / "manifest.tsv"

    script_path.write_text(render_launcher(run_id, candidates, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(run_id, candidates, script_path, matrix_path, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("cid\trun_name\taxis\tactuator\tshots\tgpu\tseed\twisig_domain\ttrain_days\ttest_days\ttrain_rxs\ttest_rxs\thypothesis\n")
        for c in candidates:
            handle.write(
                "\t".join(
                    [
                        c.cid,
                        c.run_name,
                        c.axis,
                        c.actuator,
                        str(c.shots),
                        str(c.gpu),
                        str(c.seed),
                        c.wisig_domain,
                        c.train_days,
                        c.test_days,
                        c.train_rxs,
                        c.test_rxs,
                        c.hypothesis,
                    ]
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": len(candidates),
                "launcher": str(script_path),
                "matrix": str(matrix_path),
                "manifest": str(manifest_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
