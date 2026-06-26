#!/usr/bin/env python
"""Generate a large CEN51 domain-metric controller exploration batch.

This batch expands the completed anchor-fit run into a controller-oriented
search. The key idea is to treat strict UDU, receiver floor, satellite floor,
and late rollback as separate control signals for different loss families
instead of optimizing one decorative aggregate metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cen51_fulldg_fewshot_comprehensive_matrix as fdcomp  # noqa: E402
import cen51_safd_anchor_fit_matrix as anchorfit  # noqa: E402
from cen51_lac_sat_rescue_matrix import sat_rescue_params  # noqa: E402
from cen51_lowshot_config_search import BASE_PARAMS as LAC_BASE_PARAMS  # noqa: E402
from cen51_lowshot_config_search import arg_pairs  # noqa: E402


REMOTE_ROOT = anchorfit.REMOTE_ROOT
ALL_SAT = anchorfit.ALL_SAT
LIGHT_SAT = anchorfit.LIGHT_SAT
RUN_PREFIX = "CEN51_DMCTRL"
DESIRED_SHOT_COUNTS = {5: 12, 10: 12, 20: 12, 30: 20, 50: 20, 100: 20}


def q(value: object) -> str:
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    parser.add_argument("--max-active-per-gpu", type=int, default=3)
    parser.add_argument("--scheduler-hours", type=float, default=14.0)
    return parser.parse_args()


def rename(exp: anchorfit.Experiment, cid: str | None = None) -> anchorfit.Experiment:
    new_cid = cid or exp.cid
    return replace(exp, cid=new_cid, run_name=f"{RUN_PREFIX}_{new_cid}")


def comp_by_cid() -> dict[str, object]:
    return {candidate.cid: candidate for candidate in fdcomp.make_candidates()}


def fd_anchor(candidate: object, cid: str, axis: str, action: str, hypothesis: str, success_gate: str) -> anchorfit.Experiment:
    return rename(anchorfit.fd_exp(candidate, cid, axis, action, hypothesis, success_gate))


def fd_scale(
    candidate: object,
    *,
    cid: str,
    axis: str,
    action: str,
    pressure: float = 1.0,
    sat: float = 1.0,
    rx: float = 1.0,
    seed: int | None = None,
    swad_start: int | None = None,
    epochs: int | None = None,
    hypothesis: str,
    success_gate: str,
) -> anchorfit.Experiment:
    base = candidate
    if seed is not None or swad_start is not None or epochs is not None:
        updates = {}
        if seed is not None:
            updates["seed"] = seed
        if swad_start is not None:
            updates["swad_start"] = swad_start
        if epochs is not None:
            updates["epochs"] = epochs
        base = replace(base, **updates)
    return rename(
        anchorfit.scale_fd(
            base,
            cid=cid,
            axis=axis,
            action=action,
            pressure=pressure,
            sat=sat,
            rx=rx,
            hypothesis=hypothesis,
            success_gate=success_gate,
        )
    )


def fd_no_sat(candidate: object, cid: str, hypothesis: str, success_gate: str) -> anchorfit.Experiment:
    return rename(anchorfit.no_sat_fd(candidate, cid, hypothesis, success_gate))


def lac_args(shots: int, *, seed: int = 1337, updates: dict[str, object] | None = None) -> list[str]:
    params = dict(LAC_BASE_PARAMS)
    params.update(sat_rescue_params(shots))
    params["wisig_train_ratio"] = 0.1
    params["seed"] = seed
    params["test_eval_start_epoch"] = 31
    params["test_eval_interval"] = 10
    if updates:
        params.update(updates)
    return arg_pairs(params)


def lac_exp(
    cid: str,
    axis: str,
    action: str,
    hypothesis: str,
    success_gate: str,
    *,
    seed: int = 1337,
    updates: dict[str, object] | None = None,
) -> anchorfit.Experiment:
    anchor = anchorfit.ANCHORS[100]
    return anchorfit.Experiment(
        cid=cid,
        run_name=f"{RUN_PREFIX}_{cid}",
        shot=100,
        gpu=-1,
        seed=seed,
        axis=axis,
        action=action,
        anchor_name=anchor.name,
        target_strict=anchor.strict_udu,
        target_overall=anchor.overall,
        hypothesis=hypothesis,
        success_gate=success_gate,
        args=lac_args(100, seed=seed, updates=updates),
    )


def make_pools() -> dict[int, deque[anchorfit.Experiment]]:
    comp = comp_by_cid()
    k5 = comp["FS005_BEST_HINGE6_SATMIN_2030"]
    k5_seed = comp["FS005_BEST_HINGE6_SATMIN_1337"]
    k10 = comp["FS010_BEST_IDFIRST_LATE_2030"]
    k10_seed = comp["FS010_BEST_IDFIRST_LATE_2029"]
    k20 = comp["FS020_BEST_RIEIFD_LIGHT_2028"]
    k20_seed1 = comp["FS020_BEST_RIEIFD_LIGHT_1337"]
    k20_seed2 = comp["FS020_BEST_RIEIFD_LIGHT_2029"]
    k30 = comp["FS030_BEST_RXFLOOR_CAP_2029"]
    k30_seed = comp["FS030_BEST_RXFLOOR_CAP_2030"]
    k50 = comp["FS050_BEST_CAP_RELAX_1337"]
    k50_seed = comp["FS050_BEST_CAP_RELAX_2029"]

    pools: dict[int, list[anchorfit.Experiment]] = {
        5: [
            fd_anchor(k5, "K005_A00_ANCHOR2030", "anchor_replay", "anchor", "Replay K5 anchor before controller expansion.", "strict>=74.5 sat_floor>=28.5."),
            fd_anchor(k5_seed, "K005_A01_SEED1337", "seed_check", "seed", "Check K5 low-shot seed variance.", "strict>=74.0 and late_drop<=1.0."),
            fd_scale(k5, cid="K005_S01_SAT_P080", axis="sat_gate", action="sat_light", sat=1.33, hypothesis="K5 sat floor deficit should respond to tiny sat exposure.", success_gate="sat_floor improves with strict>=74.5."),
            fd_scale(k5, cid="K005_S02_SAT_P093", axis="sat_gate", action="sat_mid", sat=1.55, hypothesis="Retest the previous best K5 sat gate.", success_gate="strict>=75.0 sat_floor>=30."),
            fd_scale(k5, cid="K005_S03_SAT_P105", axis="sat_gate", action="sat_high", sat=1.75, hypothesis="Upper sat gate point for K5 before clean strict breaks.", success_gate="sat_floor up; strict loss <=0.3 from S02."),
            fd_scale(k5, cid="K005_B01_RX_SAT_BAL", axis="balanced_metric", action="rx_sat_bal", rx=1.05, sat=1.55, hypothesis="Combine tiny RX repair with proven sat gate.", success_gate="strict>=75 sat_floor>=30 rx_floor>=61."),
            fd_scale(k5, cid="K005_R01_RX_LIGHT", axis="rx_floor", action="rx_light", rx=1.08, hypothesis="RX floor is secondary at K5; test only a small step.", success_gate="rx_floor improves without strict loss."),
            fd_scale(k5, cid="K005_R02_RX_STRONG", axis="rx_floor", action="rx_strong", rx=1.18, hypothesis="Negative control for over-using RX pressure at K5.", success_gate="reject if strict or sat floor drops."),
            fd_scale(k5, cid="K005_C01_PRESS_CLAMP", axis="pressure_clamp", action="pressure_down", pressure=0.86, sat=1.55, hypothesis="Low-shot may need less invariance plus sat gate.", success_gate="strict>=75 and late_drop<=0.5."),
            fd_scale(k5, cid="K005_C02_ID_KEEP", axis="identity_guard", action="id_keep", pressure=0.78, sat=1.45, hypothesis="Probe whether lighter DG preserves TX signal at K5.", success_gate="overall and strict do not fall."),
            fd_no_sat(k5, "K005_N01_NOSAT_NEG", "No-sat negative control for K5.", "Should not beat sat-gated branches on sat_floor."),
            fd_scale(k5, cid="K005_S04_SAT_P093_SEED2029", axis="sat_gate", action="sat_mid_seed", sat=1.55, seed=2029, hypothesis="Seed-check the K5 sat gate controller decision.", success_gate="strict>=74.5 sat_floor>=29.5."),
            fd_scale(k5, cid="K005_B02_BAL_SEED1337", axis="balanced_metric", action="rx_sat_bal_seed", rx=1.05, sat=1.55, seed=1337, hypothesis="Seed-check balanced K5 controller.", success_gate="strict>=74 sat_floor>=29."),
            fd_scale(k5, cid="K005_C03_EARLY_SWAD", axis="stability", action="early_swad", pressure=0.92, sat=1.55, swad_start=45, hypothesis="Earlier SWAD may help low-shot stability.", success_gate="latest_drop<=0.5 strict>=74.8."),
            fd_scale(k5, cid="K005_S05_SAT_P070", axis="sat_gate", action="sat_min", sat=1.17, hypothesis="Lower sat gate point for response curve.", success_gate="strict>=74.8 sat_floor above anchor."),
            fd_scale(k5, cid="K005_B03_STRICT_GUARD", axis="identity_guard", action="strict_guard", pressure=0.88, rx=1.03, sat=1.40, hypothesis="Conservative all-around K5 controller sample.", success_gate="strict>=75 with no floor collapse."),
        ],
        10: [
            fd_anchor(k10, "K010_A00_ANCHOR2030", "anchor_replay", "anchor", "Replay K10 ID-first anchor.", "strict>=76.3 sat_floor near 29."),
            fd_anchor(k10_seed, "K010_A01_SEED2029", "seed_check", "seed", "K10 seed check.", "strict>=75.5 and sat_floor>=29."),
            fd_scale(k10, cid="K010_R01_RX115", axis="rx_floor", action="rx_mid", rx=1.15, hypothesis="Retest K10 RX repair that had good overall.", success_gate="overall>=81.9 strict>=76.5."),
            fd_scale(k10, cid="K010_R02_RX108", axis="rx_floor", action="rx_light", rx=1.08, hypothesis="Smaller RX step may preserve sat floor.", success_gate="strict>=76.5 sat_floor>=28.8."),
            fd_scale(k10, cid="K010_S01_SAT108", axis="sat_gate", action="sat_light", sat=1.08, hypothesis="K10 no-sat strict was misleading; add light sat floor protection.", success_gate="sat_floor>=29 strict>=76.3."),
            fd_scale(k10, cid="K010_S02_SAT122", axis="sat_gate", action="sat_mid", sat=1.22, hypothesis="K10 sat response upper point.", success_gate="sat_floor improves with strict loss<=0.3."),
            fd_scale(k10, cid="K010_B01_RXSAT", axis="balanced_metric", action="rx_sat_bal", rx=1.08, sat=1.15, hypothesis="Balance RX and sat deficits at K10.", success_gate="strict>=76.5 sat_floor>=29."),
            fd_scale(k10, cid="K010_C01_CLAMP090", axis="pressure_clamp", action="pressure_down", pressure=0.90, sat=1.15, hypothesis="Mild clamp may avoid hurting sat floor unlike CPI_CLAMP.", success_gate="strict>=76.3 sat_floor>=29."),
            fd_scale(k10, cid="K010_C02_CLAMP082", axis="pressure_clamp", action="pressure_down_strong", pressure=0.82, sat=0.95, hypothesis="Retest strong clamp as negative boundary.", success_gate="reject if sat_floor remains low."),
            fd_no_sat(k10, "K010_N01_NOSAT_NEG", "No-sat negative control for K10 misleading strict.", "Reject if sat_floor stays far below 29."),
            fd_scale(k10, cid="K010_B02_RXSAT_SEED1337", axis="balanced_metric", action="rx_sat_bal_seed", rx=1.08, sat=1.15, seed=1337, hypothesis="Seed-check K10 balanced controller.", success_gate="strict>=75.8 sat_floor>=28.7."),
            fd_scale(k10, cid="K010_S03_SAT135", axis="sat_gate", action="sat_high", sat=1.35, hypothesis="Check where K10 satellite gate starts hurting clean strict.", success_gate="strict loss<=0.5."),
            fd_scale(k10, cid="K010_R03_RX125_NEG", axis="rx_floor", action="rx_strong", rx=1.25, hypothesis="Upper RX pressure boundary at K10.", success_gate="reject if strict or sat floor drops."),
            fd_scale(k10, cid="K010_C03_EARLY_SWAD", axis="stability", action="early_swad", pressure=0.94, sat=1.12, swad_start=55, hypothesis="Earlier SWAD with light sat protection.", success_gate="latest_drop<=0.5 strict>=76.3."),
            fd_scale(k10, cid="K010_B03_ID_KEEP", axis="identity_guard", action="id_keep", pressure=0.86, rx=1.05, sat=1.20, hypothesis="Conservative K10 identity-preserving controller.", success_gate="strict>=76.3 sat_floor>=29."),
            fd_scale(k10, cid="K010_S04_SAT115_SEED2028", axis="sat_gate", action="sat_mid_seed", sat=1.15, seed=2028, hypothesis="Seed-check K10 sat protection.", success_gate="strict>=75.8 sat_floor>=28.8."),
        ],
        20: [
            fd_anchor(k20, "K020_A00_ANCHOR2028", "anchor_replay", "anchor", "Replay K20 anchor; previous repairs were not better.", "strict>=77.3 and sat_floor>=32."),
            fd_anchor(k20_seed1, "K020_A01_SEED1337", "seed_check", "seed", "K20 seed guard.", "strict>=76.5 late_drop<=1."),
            fd_anchor(k20_seed2, "K020_A02_SEED2029", "seed_check", "seed", "K20 instability seed guard.", "reject if late_drop>2."),
            fd_scale(k20, cid="K020_C01_CLAMP085", axis="pressure_clamp", action="pressure_down", pressure=0.85, sat=1.05, hypothesis="K20 may prefer anchor with only tiny clamp.", success_gate="strict>=77 sat_floor>=32."),
            fd_scale(k20, cid="K020_C02_CLAMP075_NEG", axis="pressure_clamp", action="pressure_down_strong", pressure=0.75, sat=1.00, hypothesis="Negative boundary for too little DG pressure.", success_gate="reject if strict drops."),
            fd_scale(k20, cid="K020_S01_SAT115", axis="sat_gate", action="sat_light", sat=1.15, hypothesis="Small sat gain without the prior RX floor damage.", success_gate="strict loss<=0.3 rx_floor>=60."),
            fd_scale(k20, cid="K020_S02_SAT135", axis="sat_gate", action="sat_mid", sat=1.35, hypothesis="K20 sat response point.", success_gate="reject if rx floor falls below 60."),
            fd_scale(k20, cid="K020_R01_RX105", axis="rx_floor", action="rx_tiny", rx=1.05, hypothesis="Only a tiny RX step is defensible after prior RX repair failed.", success_gate="rx_floor improves without strict loss."),
            fd_scale(k20, cid="K020_R02_RX112_NEG", axis="rx_floor", action="rx_mid", rx=1.12, hypothesis="RX pressure boundary for K20.", success_gate="reject if rx_floor still drops."),
            fd_scale(k20, cid="K020_B01_BAL", axis="balanced_metric", action="rx_sat_bal", rx=1.04, sat=1.12, hypothesis="Balanced tiny repair around the K20 anchor.", success_gate="strict>=77.2 sat_floor>=32 rx_floor>=60."),
            fd_scale(k20, cid="K020_B02_BAL_CLAMP", axis="balanced_metric", action="clamp_sat_bal", pressure=0.90, rx=1.03, sat=1.15, hypothesis="Conservative K20 controller.", success_gate="strict>=77.2 no floor collapse."),
            fd_scale(k20, cid="K020_C03_EARLY_SWAD", axis="stability", action="early_swad", pressure=0.92, sat=1.05, swad_start=60, hypothesis="K20 unstable seed suggests SWAD guard.", success_gate="late_drop<=1 strict>=77."),
            fd_scale(k20, cid="K020_S03_SAT110_SEED1337", axis="sat_gate", action="sat_seed", sat=1.10, seed=1337, hypothesis="Seed-check small sat gate at K20.", success_gate="strict>=76.5 sat_floor>=31."),
            fd_scale(k20, cid="K020_B03_ID_KEEP", axis="identity_guard", action="id_keep", pressure=0.88, sat=1.08, hypothesis="Protect identity if K20 repairs over-regularize.", success_gate="strict>=77."),
            fd_scale(k20, cid="K020_R03_GROUP_LIGHT", axis="rx_floor", action="group_light", rx=1.03, pressure=0.98, hypothesis="Minimal group-DRO response curve point.", success_gate="rx_floor not worse than anchor."),
            fd_scale(k20, cid="K020_A03_ANCHOR_SEED2030", axis="seed_check", action="seed", seed=2030, hypothesis="Third K20 anchor-family seed for variance estimate.", success_gate="strict>=76.5 latest_drop<=1.5."),
        ],
        30: [
            fd_anchor(k30, "K030_A00_ANCHOR2029", "anchor_replay", "anchor", "Replay K30 anchor.", "strict>=78 rx_floor>=65."),
            fd_anchor(k30_seed, "K030_A01_SEED2030", "seed_check", "seed", "K30 seed guard; previous seed had low rx floor.", "rx_floor>=60 or reject seed family."),
            fd_scale(k30, cid="K030_C01_CLAMP086", axis="pressure_clamp", action="pressure_down", pressure=0.86, sat=0.85, hypothesis="Retest successful K30 CPI clamp.", success_gate="strict>=79."),
            fd_scale(k30, cid="K030_C02_CLAMP080", axis="pressure_clamp", action="pressure_down_mid", pressure=0.80, sat=0.85, hypothesis="Lower pressure point for K30 response curve.", success_gate="strict>=78.7 sat_floor>=33."),
            fd_scale(k30, cid="K030_C03_CLAMP092", axis="pressure_clamp", action="pressure_down_light", pressure=0.92, sat=0.92, hypothesis="Mild clamp may preserve sat floor.", success_gate="strict>=78.8 sat_floor>=34."),
            fd_scale(k30, cid="K030_B01_CLAMP_SAT", axis="balanced_metric", action="clamp_sat_bal", pressure=0.86, sat=1.00, hypothesis="Add slight sat protection on top of clamp.", success_gate="strict>=78.8 sat_floor>=34."),
            fd_scale(k30, cid="K030_B02_CLAMP_RX", axis="balanced_metric", action="clamp_rx_bal", pressure=0.86, rx=1.05, sat=0.90, hypothesis="Clamp but restore RX floor.", success_gate="strict>=78.8 rx_floor>=66."),
            fd_scale(k30, cid="K030_R01_RX108", axis="rx_floor", action="rx_light", rx=1.08, hypothesis="K30 RX repair was stable; test smaller RX point.", success_gate="strict>=78.4 rx_floor>=65.5."),
            fd_scale(k30, cid="K030_R02_RX115", axis="rx_floor", action="rx_mid", rx=1.15, hypothesis="Upper RX pressure boundary for K30.", success_gate="reject if strict falls below 78.3."),
            fd_scale(k30, cid="K030_S01_SAT150", axis="sat_gate", action="sat_mid", sat=1.50, pressure=0.90, hypothesis="Sat floor can improve at K30 but should not dominate.", success_gate="sat_floor>=35 strict>=78.2."),
            fd_scale(k30, cid="K030_S02_SAT200_NEG", axis="sat_gate", action="sat_high_neg", sat=2.00, pressure=0.95, hypothesis="Negative boundary for blind sat scaling.", success_gate="reject if strict below anchor."),
            fd_scale(k30, cid="K030_C04_EARLY_SWAD", axis="stability", action="early_swad", pressure=0.86, sat=0.90, swad_start=58, hypothesis="K30 clamp plus earlier SWAD.", success_gate="latest_drop<=0.7 strict>=79."),
            fd_scale(k30, cid="K030_C05_LATE_SWAD", axis="stability", action="late_swad", pressure=0.86, sat=0.90, swad_start=75, hypothesis="SWAD start sensitivity for K30 clamp.", success_gate="strict>=78.8."),
            fd_scale(k30, cid="K030_B03_ID_KEEP", axis="identity_guard", action="id_keep", pressure=0.82, rx=1.02, sat=0.95, hypothesis="Identity-preserving low pressure boundary.", success_gate="strict>=78.5."),
            fd_scale(k30, cid="K030_B04_CLAMP_SEED2028", axis="balanced_metric", action="clamp_seed", pressure=0.86, sat=0.90, seed=2028, hypothesis="Seed-check K30 clamp.", success_gate="strict>=78.5 rx_floor>=64."),
            fd_scale(k30, cid="K030_B05_CLAMP_SEED2030", axis="balanced_metric", action="clamp_seed", pressure=0.86, sat=0.90, seed=2030, hypothesis="Check whether clamp fixes weak seed2030.", success_gate="strict>=77.5 rx_floor improves."),
            fd_scale(k30, cid="K030_B06_CLAMP_SAT105", axis="balanced_metric", action="clamp_sat_light", pressure=0.86, sat=1.05, hypothesis="K30 clamp with a tiny sat-floor compensation.", success_gate="strict>=78.8 sat_floor>=34."),
            fd_scale(k30, cid="K030_B07_CLAMP_RXSAT", axis="balanced_metric", action="clamp_rx_sat", pressure=0.86, rx=1.04, sat=1.03, hypothesis="K30 all-metric small-step controller.", success_gate="strict>=78.8 rx_floor>=65 sat_floor>=34."),
            fd_scale(k30, cid="K030_A02_ANCHOR_SEED1337", axis="seed_check", action="seed", seed=1337, hypothesis="K30 anchor-family third seed.", success_gate="strict>=77.5 and rx_floor>=63."),
            fd_scale(k30, cid="K030_C06_SWAD65", axis="stability", action="swad65", pressure=0.86, sat=0.92, swad_start=65, hypothesis="K30 clamp SWAD midpoint.", success_gate="strict>=78.8 latest_drop<=0.8."),
        ],
        50: [
            fd_anchor(k50, "K050_A00_ANCHOR1337", "anchor_replay", "anchor", "Replay K50 anchor.", "strict>=81.5."),
            fd_anchor(k50_seed, "K050_A01_SEED2029", "seed_check", "seed", "K50 seed guard.", "strict>=80 and latest_drop<=1.5."),
            fd_scale(k50, cid="K050_R01_RX106", axis="rx_floor", action="rx_light", rx=1.06, hypothesis="Smaller RX repair around successful K50 direction.", success_gate="strict>=82.3."),
            fd_scale(k50, cid="K050_R02_RX110", axis="rx_floor", action="rx_mid", rx=1.10, hypothesis="Retest successful K50 RX repair.", success_gate="strict>=82.7."),
            fd_scale(k50, cid="K050_R03_RX116", axis="rx_floor", action="rx_high", rx=1.16, hypothesis="Upper RX repair boundary at K50.", success_gate="reject if strict or sat floor falls."),
            fd_scale(k50, cid="K050_S01_SAT115", axis="sat_gate", action="sat_light", sat=1.15, hypothesis="K50 floor-balance branch.", success_gate="strict>=82.3 sat_floor>=35."),
            fd_scale(k50, cid="K050_S02_SAT120", axis="sat_gate", action="sat_mid", sat=1.20, hypothesis="Retest K50 SAT_GATE_LIGHT.", success_gate="strict>=82.4 overall>=88.6."),
            fd_scale(k50, cid="K050_S03_SAT130", axis="sat_gate", action="sat_high", sat=1.30, hypothesis="K50 sat upper response point.", success_gate="reject if strict drops below 82."),
            fd_scale(k50, cid="K050_B01_RXSAT", axis="balanced_metric", action="rx_sat_bal", rx=1.08, sat=1.15, hypothesis="Joint K50 RX and sat repair.", success_gate="strict>=82.7 sat_floor>=35."),
            fd_scale(k50, cid="K050_B02_RXSAT_HIGH", axis="balanced_metric", action="rx_sat_high", rx=1.12, sat=1.20, hypothesis="Higher joint repair boundary.", success_gate="strict>=82.5 and no floor collapse."),
            fd_scale(k50, cid="K050_C01_CLAMP085", axis="pressure_clamp", action="pressure_down", pressure=0.85, sat=0.90, hypothesis="K50 clamp was weaker; keep as boundary.", success_gate="reject unless strict>=82."),
            fd_scale(k50, cid="K050_C02_CLAMP_RX", axis="balanced_metric", action="clamp_rx_bal", pressure=0.92, rx=1.10, sat=1.05, hypothesis="RX repair with mild pressure clamp.", success_gate="strict>=82.5 rx_floor improves."),
            fd_scale(k50, cid="K050_C03_EARLY_SWAD", axis="stability", action="early_swad", rx=1.10, swad_start=72, hypothesis="K50 RX repair with earlier SWAD.", success_gate="strict>=82.7 latest_drop<=0.8."),
            fd_scale(k50, cid="K050_B03_RX_SEED2028", axis="rx_floor", action="rx_seed", rx=1.10, seed=2028, hypothesis="Seed-check K50 RX repair.", success_gate="strict>=81.5."),
            fd_scale(k50, cid="K050_S04_SAT_SEED2028", axis="sat_gate", action="sat_seed", sat=1.20, seed=2028, hypothesis="Seed-check K50 sat gate.", success_gate="strict>=81.5 sat_floor>=34.5."),
            fd_scale(k50, cid="K050_B04_ID_KEEP", axis="identity_guard", action="id_keep", pressure=0.90, rx=1.05, sat=1.15, hypothesis="Conservative K50 controller.", success_gate="strict>=82.2."),
            fd_scale(k50, cid="K050_B05_RXSAT_SEED2030", axis="balanced_metric", action="rx_sat_seed", rx=1.08, sat=1.15, seed=2030, hypothesis="Seed-check K50 balanced RX/SAT branch.", success_gate="strict>=81.8 sat_floor>=34.5."),
            fd_scale(k50, cid="K050_S05_SAT140_NEG", axis="sat_gate", action="sat_high_neg", sat=1.40, hypothesis="Upper sat boundary for K50.", success_gate="reject if strict<82."),
            fd_scale(k50, cid="K050_R04_RX104_SWAD", axis="rx_floor", action="rx_tiny_swad", rx=1.04, swad_start=72, hypothesis="Tiny RX repair plus SWAD guard.", success_gate="strict>=82.2 latest_drop<=0.8."),
            fd_scale(k50, cid="K050_C04_CLAMP_SAT", axis="balanced_metric", action="clamp_sat_bal", pressure=0.92, sat=1.18, hypothesis="K50 floor-balanced clamp/sat branch.", success_gate="strict>=82 sat_floor>=35."),
        ],
        100: [
            lac_exp("K100_A00_ANCHOR1337", "anchor_replay", "anchor", "Replay K100 LACSR anchor under this queue.", "strict>=82.5 and late_drop<=1.7."),
            lac_exp("K100_A01_SEED2028", "seed_check", "seed", "K100 seed guard.", "strict>=80 and no late collapse.", seed=2028),
            lac_exp("K100_S01_CONSERVE050", "sat_gate", "sat_conserve", "Retest successful K100 sat conserve.", "strict>=83 sat floor tradeoff recorded.", updates={"sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65, "sat_view_schedule": f"1@0.40:{LIGHT_SAT};140@0.55:{ALL_SAT}"}),
            lac_exp("K100_S02_CONSERVE045", "sat_gate", "sat_conserve_low", "Lower K100 satellite pressure to test clean strict recovery.", "strict improves without overall loss.", updates={"sat_view_prob": 0.45, "concat_sat_ce_weight": 0.58, "sat_view_schedule": f"1@0.35:{LIGHT_SAT};150@0.50:{ALL_SAT}"}),
            lac_exp("K100_S03_CONSERVE060", "sat_gate", "sat_conserve_mid", "Slightly higher K100 sat conserve point.", "strict>=83 sat_floor>=38.", updates={"sat_view_prob": 0.60, "concat_sat_ce_weight": 0.74, "sat_view_schedule": f"1@0.45:{LIGHT_SAT};130@0.65:{ALL_SAT}"}),
            lac_exp("K100_S04_BOOST_NEG", "sat_gate", "sat_boost_neg", "Negative boundary: high sat pressure should be rejected if strict falls.", "reject if strict<82.", updates={"sat_view_prob": 0.90, "concat_sat_ce_weight": 1.05, "sat_view_schedule": f"1@0.75:{LIGHT_SAT};100@0.90:{ALL_SAT}"}),
            lac_exp("K100_C01_GLOBAL_CLAMP", "pressure_clamp", "pressure_down", "Reduce global DG pressure with conservative sat.", "latest_drop improves and strict>=83.", updates={"lambda_adv": 0.30, "lambda_cons": 0.065, "lambda_group_ce": 0.060, "lambda_proto": 0.011, "lambda_supcon_id": 0.014, "lambda_fishr": 0.0011, "sat_view_prob": 0.55, "concat_sat_ce_weight": 0.70}),
            lac_exp("K100_C02_ID_KEEP", "identity_guard", "id_keep", "Keep TX identity by lowering proto/supcon/cons pressure.", "strict improves over anchor.", updates={"lambda_cons": 0.055, "lambda_proto": 0.009, "lambda_supcon_id": 0.011, "lambda_group_ce": 0.060, "sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65}),
            lac_exp("K100_R01_RX_LIGHT", "rx_floor", "rx_light", "RX floor is low, but previous RX repair failed; use only a light step.", "rx floor improves without strict loss.", updates={"lambda_adv": 0.36, "lambda_group_ce": 0.082, "group_ce_top_frac": 0.28, "groupdro_cap": 0.58, "sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65}),
            lac_exp("K100_R02_RX_STRONG_NEG", "rx_floor", "rx_strong_neg", "Negative boundary for K100 RX pressure.", "reject if strict drops.", updates={"lambda_adv": 0.42, "lambda_group_ce": 0.100, "group_ce_top_frac": 0.34, "groupdro_cap": 0.68}),
            lac_exp("K100_B01_CLAMP_RX", "balanced_metric", "clamp_rx_bal", "Balanced K100 clamp plus light RX repair.", "strict>=83 rx floor not worse.", updates={"lambda_adv": 0.34, "lambda_cons": 0.060, "lambda_group_ce": 0.074, "lambda_proto": 0.010, "lambda_supcon_id": 0.013, "group_ce_top_frac": 0.28, "groupdro_cap": 0.58, "sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65}),
            lac_exp("K100_B02_STRICT_FIRST", "identity_guard", "strict_first", "Strict-first K100 controller prioritizes TX identity over floors.", "strict>=83.5 even if floors lag.", updates={"lambda_adv": 0.28, "lambda_cons": 0.050, "lambda_group_ce": 0.052, "lambda_proto": 0.008, "lambda_supcon_id": 0.010, "sat_view_prob": 0.42, "concat_sat_ce_weight": 0.55}),
            lac_exp("K100_B03_SAT_FLOOR_GUARD", "balanced_metric", "sat_floor_guard", "Minimum sat guard with identity-protect weights.", "sat_floor>=38 strict>=83.", updates={"lambda_adv": 0.30, "lambda_group_ce": 0.060, "lambda_proto": 0.009, "lambda_supcon_id": 0.011, "sat_view_prob": 0.58, "concat_sat_ce_weight": 0.72, "sat_view_schedule": f"1@0.42:{LIGHT_SAT};120@0.60:{ALL_SAT}"}),
            lac_exp("K100_C03_EARLY_SWAD", "stability", "early_swad", "Earlier averaging for K100 rollback control.", "latest_drop<=1.0 strict>=83.", updates={"swad_start_epoch": 80, "sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65}),
            lac_exp("K100_A02_ANCHOR_SEED2030", "seed_check", "seed", "Third K100 seed for variance guard.", "strict>=81 and no large rollback.", seed=2030),
            lac_exp("K100_S05_CONSERVE050_SEED2028", "sat_gate", "sat_conserve_seed", "Seed-check K100 sat conserve.", "strict>=82 sat_floor>=37.", seed=2028, updates={"sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65, "sat_view_schedule": f"1@0.40:{LIGHT_SAT};140@0.55:{ALL_SAT}"}),
            lac_exp("K100_S06_CONSERVE055_SEED2030", "sat_gate", "sat_conserve_seed", "K100 moderate conserve with third seed.", "strict>=82.5 sat_floor>=37.5.", seed=2030, updates={"sat_view_prob": 0.55, "concat_sat_ce_weight": 0.70, "sat_view_schedule": f"1@0.42:{LIGHT_SAT};135@0.60:{ALL_SAT}"}),
            lac_exp("K100_B04_STRICT_FIRST_SEED2028", "identity_guard", "strict_first_seed", "Seed-check strict-first K100 boundary.", "strict>=82 with low rollback.", seed=2028, updates={"lambda_adv": 0.28, "lambda_cons": 0.050, "lambda_group_ce": 0.052, "lambda_proto": 0.008, "lambda_supcon_id": 0.010, "sat_view_prob": 0.42, "concat_sat_ce_weight": 0.55}),
            lac_exp("K100_B05_CLAMP_RX_SEED2028", "balanced_metric", "clamp_rx_seed", "Seed-check balanced K100 clamp/RX.", "strict>=82 rx floor not worse.", seed=2028, updates={"lambda_adv": 0.34, "lambda_cons": 0.060, "lambda_group_ce": 0.074, "lambda_proto": 0.010, "lambda_supcon_id": 0.013, "group_ce_top_frac": 0.28, "groupdro_cap": 0.58, "sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65}),
            lac_exp("K100_C04_GLOBAL_CLAMP_STRONG", "pressure_clamp", "pressure_down_strong", "Lower K100 global pressure boundary.", "reject if overall/strict fall below conserve.", updates={"lambda_adv": 0.26, "lambda_cons": 0.050, "lambda_group_ce": 0.050, "lambda_proto": 0.008, "lambda_supcon_id": 0.010, "lambda_fishr": 0.0008, "sat_view_prob": 0.45, "concat_sat_ce_weight": 0.58}),
        ],
    }
    for shot, items in pools.items():
        desired = DESIRED_SHOT_COUNTS[shot]
        if len(items) < desired:
            raise AssertionError(f"K{shot} expected at least {desired} candidates, got {len(items)}")
        pools[shot] = items[:desired]
    return {shot: deque(items) for shot, items in pools.items()}


def assign_gpu_queues(pools: dict[int, deque[anchorfit.Experiment]]) -> list[anchorfit.Experiment]:
    gpu_templates = {
        0: [5, 30, 100, 10, 50, 100, 20, 30, 50, 5, 20, 50],
        1: [10, 50, 100, 20, 30, 50, 5, 30, 100, 10, 30, 100],
        2: [20, 30, 50, 5, 30, 100, 10, 50, 100, 5, 20, 50],
        3: [5, 30, 100, 20, 30, 50, 10, 50, 100, 10, 30, 100],
    }
    rows: list[anchorfit.Experiment] = []
    for gpu in range(8):
        gpu_sequence = gpu_templates[gpu % 4]
        for shot in gpu_sequence:
            rows.append(replace(pools[shot].popleft(), gpu=gpu))
    leftovers = {shot: len(queue) for shot, queue in pools.items()}
    if any(leftovers.values()):
        raise AssertionError(f"unassigned candidates remain: {leftovers}")
    if len(rows) != 96:
        raise AssertionError(f"expected 96 candidates, got {len(rows)}")
    run_names = [row.run_name for row in rows]
    if len(run_names) != len(set(run_names)):
        duplicates = [name for name, count in Counter(run_names).items() if count > 1]
        raise AssertionError(f"duplicate run names: {duplicates}")
    return rows


def command_for(run_id: str, c: anchorfit.Experiment) -> str:
    run_dir = f"{REMOTE_ROOT}/runs/{run_id}/{c.run_name}"
    args = [
        "env",
        f"CUDA_VISIBLE_DEVICES={c.gpu}",
        f"PYTHONPATH={REMOTE_ROOT}/code:{REMOTE_ROOT}",
        "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
        "-u",
        f"{REMOTE_ROOT}/code/train.py",
        *c.args,
        "--run_name",
        c.run_name,
        "--wisig_max_train_per_combo",
        str(c.shot),
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


def render_launcher(run_id: str, rows: Sequence[anchorfit.Experiment], max_active_per_gpu: int, scheduler_hours: float) -> str:
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
        "CAND_ID=(); CAND_RUN=(); CAND_SHOT=(); CAND_GPU=(); CAND_AXIS=(); CAND_ACTION=(); CAND_TARGET_STRICT=(); CAND_TARGET_OVERALL=(); CAND_CMD=()",
        "STATUS=(); PID=(); LOG_PATH=()",
        "",
        "add_candidate() {",
        '  CAND_ID+=("$1"); CAND_RUN+=("$2"); CAND_SHOT+=("$3"); CAND_GPU+=("$4"); CAND_AXIS+=("$5"); CAND_ACTION+=("$6"); CAND_TARGET_STRICT+=("$7"); CAND_TARGET_OVERALL+=("$8"); CAND_CMD+=("$9")',
        '  STATUS+=("queued"); PID+=(""); LOG_PATH+=("${LOG_ROOT}/$2.out")',
        "}",
        "",
    ]
    for c in rows:
        lines.append(
            "add_candidate "
            f"{q(c.cid)} {q(c.run_name)} {q(c.shot)} {q(c.gpu)} {q(c.axis)} {q(c.action)} "
            f"{q(f'{c.target_strict:.2f}')} {q(f'{c.target_overall:.2f}')} {q(command_for(run_id, c))}"
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
            '      printf "%s\\t%s\\t%s\\t%s\\t%s\\tDEFERRED_WINDOW\\n" "$(date -Is)" "${CAND_ID[$i]}" "${CAND_RUN[$i]}" "${CAND_GPU[$i]}" "${CAND_SHOT[$i]}" | tee -a "${LOG_ROOT}/deferred.tsv"',
            "    fi",
            "  done",
            "}",
            "",
            "launch_idx() {",
            '  local i="$1" cid="${CAND_ID[$1]}" run="${CAND_RUN[$1]}" gpu="${CAND_GPU[$1]}" log_path="${LOG_PATH[$1]}" run_dir="${RUNS_ROOT}/${CAND_RUN[$1]}"',
            '  if should_skip "${cid}" "${run}"; then STATUS[$i]="skipped_only"; return 0; fi',
            '  if [[ "${DRY_RUN}" == "1" ]]; then',
            '    mkdir -p "${LOG_ROOT}"',
            '    printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tDRY_RUN\\t%s\\n" "$(date -Is)" "${cid}" "${run}" "${gpu}" "${CAND_SHOT[$i]}" "${CAND_ACTION[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            '    echo "[DRY-RUN] ${CAND_CMD[$i]}"; STATUS[$i]="dry_run"; return 0',
            "  fi",
            '  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then',
            '    STATUS[$i]="blocked_path"; printf "%s\\t%s\\tBLOCKED_PATH\\t%s\\t%s\\n" "${cid}" "${run}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"; return 0',
            "  fi",
            '  mkdir -p "${LOG_ROOT}" "${run_dir}"',
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\tSTART\\t%s\\n" "$(date -Is)" "${cid}" "${run}" "${gpu}" "${CAND_SHOT[$i]}" "${CAND_ACTION[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            '  bash -lc "${CAND_CMD[$i]}" > "${log_path}" 2>&1 &',
            '  PID[$i]="$!"; STATUS[$i]="running"',
            '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${cid}" "${run}" "${CAND_SHOT[$i]}" "${gpu}" "${PID[$i]}" "${CAND_AXIS[$i]}" "${CAND_ACTION[$i]}" "${CAND_TARGET_STRICT[$i]}" "${CAND_TARGET_OVERALL[$i]}" "${log_path}" | tee -a "${LOG_ROOT}/launch_pids.tsv"',
            "}",
            "",
            "reap_finished() {",
            "  local i rc",
            '  for i in "${!STATUS[@]}"; do',
            '    if [[ "${STATUS[$i]}" == "running" ]]; then',
            '      if ! kill -0 "${PID[$i]}" 2>/dev/null; then',
            '        if wait "${PID[$i]}"; then rc=0; else rc="$?"; fi',
            '        STATUS[$i]="done_${rc}"',
            '        printf "%s\\t%s\\t%s\\t%s\\t%s\\tDONE\\trc=%s\\n" "$(date -Is)" "${CAND_ID[$i]}" "${CAND_RUN[$i]}" "${CAND_GPU[$i]}" "${CAND_SHOT[$i]}" "${rc}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "      fi",
            "    fi",
            "  done",
            "}",
            "",
            "queued_left() { local i n=0; for i in \"${!STATUS[@]}\"; do [[ \"${STATUS[$i]}\" == \"queued\" ]] && n=$((n + 1)); done; echo \"${n}\"; }",
            "running_left() { local i n=0; for i in \"${!STATUS[@]}\"; do [[ \"${STATUS[$i]}\" == \"running\" ]] && n=$((n + 1)); done; echo \"${n}\"; }",
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
            'echo "[CEN51-DMCTRL] run_id=${RUN_ID} dry_run=${DRY_RUN} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} max_seconds=${MAX_SCHEDULER_SECONDS}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            'for gpu in 0 1 2 3 4 5 6 7; do echo "[CEN51-DMCTRL] initial gpu=${gpu} count=$(gpu_process_count "${gpu}")" | tee -a "${LOG_ROOT}/scheduler_events.tsv"; done',
            'START_TS="$(date +%s)"',
            'WINDOW_EXPIRED=0',
            "while true; do",
            "  reap_finished",
            '  NOW_TS="$(date +%s)"',
            "  if (( NOW_TS - START_TS < MAX_SCHEDULER_SECONDS )); then",
            "    launch_available",
            "  elif [[ \"${WINDOW_EXPIRED}\" == \"0\" ]]; then",
            "    WINDOW_EXPIRED=1",
            "    defer_queued_after_window",
            "  fi",
            '  q_left="$(queued_left)"; r_left="$(running_left)"',
            '  echo "[CEN51-DMCTRL] heartbeat=$(date -Is) queued=${q_left} running=${r_left} window_expired=${WINDOW_EXPIRED}" | tee -a "${LOG_ROOT}/scheduler_heartbeat.log"',
            '  if [[ "${q_left}" == "0" && "${r_left}" == "0" ]]; then break; fi',
            '  if [[ "${DRY_RUN}" == "1" ]]; then break; fi',
            '  sleep "${POLL_SECONDS}"',
            "done",
            'echo "[CEN51-DMCTRL] scheduler_complete $(date -Is)" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(path: Path, rows: Sequence[anchorfit.Experiment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cid",
            "run_name",
            "shot",
            "gpu",
            "seed",
            "axis",
            "action",
            "anchor_name",
            "target_strict",
            "target_overall",
            "hypothesis",
            "success_gate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data.pop("args", None)
            writer.writerow(data)


def render_report(
    run_id: str,
    rows: Sequence[anchorfit.Experiment],
    script_path: Path,
    matrix_path: Path,
    manifest_path: Path,
    max_active_per_gpu: int,
    scheduler_hours: float,
) -> str:
    gpu_counts = Counter(row.gpu for row in rows)
    shot_counts = Counter(row.shot for row in rows)
    axis_counts = Counter(row.axis for row in rows)
    gpu_shots: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        gpu_shots[row.gpu].append(row.shot)

    table = [
        "| ID | K | GPU | axis/action | target strict/overall | success gate |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| `{row.cid}` | {row.shot} | {row.gpu} | {row.axis}/{row.action} | "
            f"{row.target_strict:.2f}/{row.target_overall:.2f} | {row.success_gate} |"
        )

    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## 目标",
            "",
            "基于 `cen51_safd_anchor_fit_20260612_022622` 的完整日志，继续优化域指标驱动的 CVS 少样本域泛化算法。"
            "本批不是单点追分，而是验证域指标能否把不同损失族分开控制：域对抗、receiver/group-DRO、身份保持、satellite exposure 与稳定性/模型平均。",
            "",
            "## 数学原则",
            "",
            "把每个 K 的训练看成受约束多目标问题：",
            "",
            "```text",
            "maximize   strict_udu(theta) + alpha * overall(theta)",
            "subject to rx_floor(theta)  >= rho_K",
            "           sat_floor(theta) >= sigma_K",
            "           late_drop(theta) <= tau_K",
            "```",
            "",
            "可落地的离散控制律为：",
            "",
            "- `rx_floor` 缺口 -> 调 `lambda_group_ce`, `group_ce_top_frac`, `groupdro_cap`，只在 K50 等已证明有效的区间加压。",
            "- `sat_floor` 缺口 -> 调 `sat_view_prob`, `sat_view_schedule`, `concat_sat_ce_weight`，但以 strict 不崩为硬门。",
            "- `strict_udu` 与 `late_drop` -> 调 `lambda_dom`, `lambda_adv`, SWAD 起点；当过压时先降域对抗而不是继续加正则。",
            "- 身份信号受损 -> 降 `lambda_proto`, `lambda_supcon_id`, `lambda_cons`，特别是 K5/K10/K30。",
            "",
            "文献依据：",
            "",
            "- DANN/JMLR: https://www.jmlr.org/papers/v17/15-239.html ，用域不可判别特征解释 `lambda_adv/lambda_dom` 的意义。",
            "- Group DRO/arXiv: https://arxiv.org/abs/1911.08731 ，支持 worst-group/receiver floor 目标，同时强调正则和早停。",
            "- Fishr/PMLR: https://proceedings.mlr.press/v162/rame22a.html ，支持跨域梯度统计一致性，但不能替代身份保持。",
            "- SWAD/NeurIPS: https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html ，支持用 flat minima/模型平均缓解 late rollback。",
            "",
            "## 批次规模与调度",
            "",
            f"- candidates: {len(rows)}",
            f"- max active per GPU: {max_active_per_gpu}",
            f"- scheduler launch window: {scheduler_hours:.1f} hours",
            f"- expected behavior: 每张 GPU 同时最多 3 个训练进程；任一进程结束后，从同一 GPU 队列补上下一候选。",
            f"- GPU candidate counts: `{json.dumps(dict(sorted(gpu_counts.items())), ensure_ascii=False)}`",
            f"- shot counts: `{json.dumps(dict(sorted(shot_counts.items())), ensure_ascii=False)}`",
            f"- axis counts: `{json.dumps(dict(sorted(axis_counts.items())), ensure_ascii=False)}`",
            f"- per-GPU K order: `{json.dumps({gpu: vals for gpu, vals in sorted(gpu_shots.items())}, ensure_ascii=False)}`",
            "",
            "## 实验轴",
            "",
            "- `sat_gate`: K5/K10/K50/K100 的 satellite exposure 响应曲线。",
            "- `rx_floor`: receiver floor/group-DRO 响应曲线，重点验证 K50 有效性与 K20/K100 边界。",
            "- `pressure_clamp`: 域对抗和多正则降压，重点 K30/K100。",
            "- `balanced_metric`: 按 rx/sat/strict 多指标共同约束的小步组合。",
            "- `identity_guard`: 保护 TX 身份信息，避免低样本下过强不变性。",
            "- `stability`: SWAD/late rollback 防护。",
            "- `seed_check` 与 negative controls: 防止把随机性或单指标错判成机制收益。",
            "",
            "## 候选矩阵",
            "",
            *table,
            "",
            "## 完成后解析",
            "",
            "```powershell",
            "conda activate ssr-gpu",
            f"python tools\\cen51_fewshot_stability_validator.py --log-dir <local-log-dir> --matrix-json {matrix_path} --out-dir analysis_tmp\\{run_id}\\stability_validation --late-window 30 --no-fail",
            f"python tools\\cen51_anchor_fit_score.py --summary-csv analysis_tmp\\{run_id}\\stability_validation\\stability_summary.csv --out-dir analysis_tmp\\{run_id}\\anchor_fit_score",
            "```",
            "",
            "## 本地/远端路径",
            "",
            f"- launcher: `{script_path}`",
            f"- matrix: `{matrix_path}`",
            f"- manifest: `{manifest_path}`",
            f"- remote logs: `{REMOTE_ROOT}/logs/{run_id}`",
            f"- remote runs: `{REMOTE_ROOT}/runs/{run_id}`",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_domain_metric_ctrl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows = assign_gpu_queues(make_pools())

    report_dir = args.output_root / run_id
    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    manifest_path = report_dir / "manifest.tsv"
    report_path = report_dir / "report.md"

    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_launcher(run_id, rows, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(manifest_path, rows)
    report_path.write_text(
        render_report(run_id, rows, script_path, matrix_path, manifest_path, args.max_active_per_gpu, args.scheduler_hours),
        encoding="utf-8",
    )
    print(json.dumps({
        "run_id": run_id,
        "candidates": len(rows),
        "launcher": str(script_path),
        "matrix": str(matrix_path),
        "manifest": str(manifest_path),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
