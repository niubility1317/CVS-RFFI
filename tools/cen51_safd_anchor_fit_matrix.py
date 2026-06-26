#!/usr/bin/env python
"""Generate the CEN51-SAFD anchor-fit experiment matrix.

This batch uses the current best per-shot candidates as numeric anchors, then
tests whether SAFD-style diagnostic actions are actually useful: pressure
clamp, receiver-floor repair, satellite gate adjustment, and seed confirmation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cen51_fulldg_fewshot_comprehensive_matrix as fdcomp  # noqa: E402
import cen51_fulldg_shotaware_lacsr_fd_matrix as fdv1  # noqa: E402
from cen51_lac_sat_rescue_matrix import sat_rescue_params  # noqa: E402
from cen51_lowshot_config_search import BASE_PARAMS as LAC_BASE_PARAMS  # noqa: E402
from cen51_lowshot_config_search import arg_pairs  # noqa: E402


REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
ALL_SAT = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LIGHT_SAT = "clear_leo,mixed_orbit"


@dataclass(frozen=True)
class Anchor:
    shot: int
    name: str
    strict_udu: float
    overall: float
    source: str


@dataclass(frozen=True)
class Experiment:
    cid: str
    run_name: str
    shot: int
    gpu: int
    seed: int
    axis: str
    action: str
    anchor_name: str
    target_strict: float
    target_overall: float
    hypothesis: str
    success_gate: str
    args: list[str]


ANCHORS: dict[int, Anchor] = {
    5: Anchor(5, "CEN51_FDCOMP_FS005_BEST_HINGE6_SATMIN_2030", 74.52, 79.70, "FDCOMP"),
    10: Anchor(10, "CEN51_FDCOMP_FS010_BEST_IDFIRST_LATE_2030", 76.27, 81.54, "FDCOMP"),
    20: Anchor(20, "CEN51_FDCOMP_FS020_BEST_RIEIFD_LIGHT_2028", 77.34, 83.58, "FDCOMP"),
    30: Anchor(30, "CEN51_FDCOMP_FS030_BEST_RXFLOOR_CAP_2029", 78.72, 85.53, "FDCOMP"),
    50: Anchor(50, "CEN51_FDCOMP_FS050_BEST_CAP_RELAX_1337", 82.31, 88.58, "FDCOMP"),
    100: Anchor(100, "CEN51_LACSR06_FS100_r010_anchor_r010", 84.05, 88.45, "LACSR"),
}


def q(value: object) -> str:
    return shlex.quote(str(value))


def bash_items(items: Iterable[object], indent: str = "  ") -> str:
    return "".join(f"{indent}{q(item)}\n" for item in items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    parser.add_argument("--max-active-per-gpu", type=int, default=2)
    parser.add_argument("--scheduler-hours", type=float, default=7.0)
    return parser.parse_args()


def by_cid(candidates: Sequence[fdv1.Candidate], cid: str) -> fdv1.Candidate:
    for candidate in candidates:
        if candidate.cid == cid:
            return candidate
    raise KeyError(cid)


def set_arg(args: list[str], flag: str, value: object) -> list[str]:
    out = list(args)
    if flag in out:
        idx = out.index(flag)
        if idx + 1 >= len(out):
            raise ValueError(f"flag has no value: {flag}")
        out[idx + 1] = str(value)
        return out
    out.extend([flag, str(value)])
    return out


def replace_flag(args: list[str], old: str, new: str) -> list[str]:
    out = list(args)
    if old in out:
        out[out.index(old)] = new
    elif new not in out:
        out.append(new)
    return out


def fd_args(candidate: fdv1.Candidate) -> list[str]:
    return fdv1.base_args() + fdv1.candidate_args(candidate)


def fd_exp(
    candidate: fdv1.Candidate,
    cid: str,
    axis: str,
    action: str,
    hypothesis: str,
    success_gate: str,
    *,
    args_override: list[str] | None = None,
) -> Experiment:
    anchor = ANCHORS[candidate.shots]
    return Experiment(
        cid=cid,
        run_name=f"CEN51_SAFDAF_{cid}",
        shot=candidate.shots,
        gpu=-1,
        seed=candidate.seed,
        axis=axis,
        action=action,
        anchor_name=anchor.name,
        target_strict=anchor.strict_udu,
        target_overall=anchor.overall,
        hypothesis=hypothesis,
        success_gate=success_gate,
        args=args_override if args_override is not None else fd_args(candidate),
    )


def scale_fd(
    base: fdv1.Candidate,
    *,
    cid: str,
    axis: str,
    action: str,
    pressure: float = 1.0,
    sat: float = 1.0,
    rx: float = 1.0,
    hypothesis: str,
    success_gate: str,
) -> Experiment:
    candidate = replace(
        base,
        lambda_dom=base.lambda_dom * pressure * rx,
        lambda_adv=base.lambda_adv * pressure * rx,
        lambda_group_ce=base.lambda_group_ce * pressure * rx,
        lambda_proto=base.lambda_proto * pressure,
        lambda_supcon=base.lambda_supcon * pressure,
        lambda_fishr=base.lambda_fishr * pressure,
        sat_prob=max(0.0, min(0.40, base.sat_prob * sat)),
        group_top_frac=max(0.10, min(0.30, base.group_top_frac * rx)),
        group_cap=max(0.35, min(0.62, base.group_cap * rx)),
    )
    return fd_exp(candidate, cid, axis, action, hypothesis, success_gate)


def no_sat_fd(base: fdv1.Candidate, cid: str, hypothesis: str, success_gate: str) -> Experiment:
    args = fd_args(base)
    args = replace_flag(args, "--use_concat_sat_channel_aug", "--no_use_concat_sat_channel_aug")
    args = set_arg(args, "--sat_view_prob", "0.0")
    args = set_arg(args, "--concat_sat_start_epoch", "999")
    args = set_arg(args, "--sat_view_schedule", f"1@0.000:{LIGHT_SAT}")
    return fd_exp(base, cid, "paired_control", "no_sat_control", hypothesis, success_gate, args_override=args)


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
) -> Experiment:
    anchor = ANCHORS[100]
    return Experiment(
        cid=cid,
        run_name=f"CEN51_SAFDAF_{cid}",
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


def make_candidates() -> list[Experiment]:
    comp = {candidate.cid: candidate for candidate in fdcomp.make_candidates()}

    k5 = comp["FS005_BEST_HINGE6_SATMIN_2030"]
    k10 = comp["FS010_BEST_IDFIRST_LATE_2030"]
    k20 = comp["FS020_BEST_RIEIFD_LIGHT_2028"]
    k30 = comp["FS030_BEST_RXFLOOR_CAP_2029"]
    k50 = comp["FS050_BEST_CAP_RELAX_1337"]

    rows: list[Experiment] = [
        fd_exp(k5, "K005_ANCHOR_HINGE6_2030", "anchor_replay", "anchor", "Replay the screenshot K5 strict leader as the fit origin.", "strict within -0.5pp of 74.52 and no late collapse."),
        fd_exp(k10, "K010_ANCHOR_IDFIRST_2030", "anchor_replay", "anchor", "Replay the screenshot K10 strict leader as the fit origin.", "strict within -0.5pp of 76.27 and val>=93."),
        fd_exp(k20, "K020_ANCHOR_RIEIFD_2028", "anchor_replay", "anchor", "Replay the screenshot K20 strict leader as the fit origin.", "strict within -0.5pp of 77.34 and rx_floor>=60."),
        fd_exp(k30, "K030_ANCHOR_RXFLOOR_2029", "anchor_replay", "anchor", "Replay the screenshot K30 strict leader as the fit origin.", "strict within -0.5pp of 78.72 and rx_floor>=65."),
        fd_exp(k50, "K050_ANCHOR_CAPRELAX_1337", "anchor_replay", "anchor", "Replay the screenshot K50 strict leader as the fit origin.", "strict within -0.5pp of 82.31 and rx_floor>=67."),
        lac_exp("K100_ANCHOR_LACSR06_1337", "anchor_replay", "anchor", "Replay the screenshot K100 LACSR06 r010 anchor.", "strict within -0.5pp of 84.05 and overall near 88.45."),
        fd_exp(comp["FS005_BEST_HINGE6_SATMIN_1337"], "K005_SEEDCHECK_1337", "seed_check", "seed", "Check whether the K5 anchor family is reproducible outside seed2030.", "strict>=74.0 and seed gap <=1.0pp."),
        fd_exp(comp["FS010_BEST_IDFIRST_LATE_2029"], "K010_SEEDCHECK_2029", "seed_check", "seed", "Check whether the K10 ID-first family is stable below the best seed.", "strict>=75.8 and val>=93."),
        fd_exp(comp["FS020_BEST_RIEIFD_LIGHT_2029"], "K020_SEEDCHECK_2029", "seed_check", "seed", "Expose K20 seed instability before changing the mechanism.", "strict>=76.5 or dominant deficit points to val/late."),
        fd_exp(comp["FS030_BEST_RXFLOOR_CAP_2030"], "K030_SEEDCHECK_2030", "seed_check", "seed", "Expose K30 RX-floor family seed spread.", "strict>=77.5 or rx deficit explains gap."),
        fd_exp(comp["FS050_BEST_CAP_RELAX_2029"], "K050_SEEDCHECK_2029", "seed_check", "seed", "Check K50 CAP_RELAX seed spread against the best seed.", "strict>=80.5 and rx_floor>=65."),
        fd_exp(comp["FS020_BEST_RIEIFD_LIGHT_1337"], "K020_SEEDCHECK_1337", "seed_check", "seed", "Second K20 seed check to separate random split from mechanism.", "strict>=76.5 and late_drop<=2."),
        scale_fd(k5, cid="K005_RX_REPAIR", axis="metric_action", action="rx_floor_repair", rx=1.18, hypothesis="If K5 remaining gap is receiver-floor leakage, modest RX pressure should raise strict without killing val.", success_gate="rx_floor improves and strict >= anchor-0.5pp."),
        scale_fd(k5, cid="K005_SAT_GATE", axis="metric_action", action="sat_gate_repair", sat=1.55, hypothesis="If sat_floor is the dominant deficit, a gated sat increase should lift sat_floor without strict loss >1pp.", success_gate="sat_floor improves; strict >=73.5."),
        scale_fd(k10, cid="K010_CPI_CLAMP", axis="metric_action", action="pressure_clamp", pressure=0.82, sat=0.80, hypothesis="If K10 is over-regularized late, a CPI clamp should preserve strict and improve latest gap.", success_gate="latest_drop shrinks and strict >=75.8."),
        scale_fd(k10, cid="K010_RX_REPAIR", axis="metric_action", action="rx_floor_repair", rx=1.15, hypothesis="If K10 strict gap is driven by rx floor, raise RX pressure only.", success_gate="rx_floor improves; val loss <=0.7pp."),
        scale_fd(k20, cid="K020_RX_REPAIR", axis="metric_action", action="rx_floor_repair", rx=1.16, hypothesis="K20 anchor is strict-strong but rx-limited; test targeted RX repair.", success_gate="rx_floor improves and strict >=77.0."),
        scale_fd(k20, cid="K020_SAT_GATE", axis="metric_action", action="sat_gate_repair", sat=1.45, hypothesis="K20 has enough samples for more sat coverage; metric should reveal clean/sat tradeoff.", success_gate="sat_floor improves with strict loss <=0.8pp."),
        scale_fd(k30, cid="K030_CPI_CLAMP", axis="metric_action", action="pressure_clamp", pressure=0.86, sat=0.85, hypothesis="If K30 RXFLOOR_CAP is near over-pressure, clamp should improve late stability.", success_gate="latest_drop shrinks with strict >=78.0."),
        scale_fd(k30, cid="K030_RX_REPAIR", axis="metric_action", action="rx_floor_repair", rx=1.12, hypothesis="K30 anchor still reports rx8 as hard receiver; targeted RX pressure should help.", success_gate="rx_floor improves and strict >=78.2."),
        scale_fd(k50, cid="K050_RX_REPAIR", axis="metric_action", action="rx_floor_repair", rx=1.10, hypothesis="K50 strict is high but rx floor remains below target; raise only RX-aware pressure.", success_gate="rx_floor improves; strict >=82.0."),
        scale_fd(k50, cid="K050_SAT_GATE_LIGHT", axis="metric_action", action="sat_gate_repair", sat=1.20, hypothesis="K50 can afford slightly more satellite pressure; useful metric must expose the tradeoff.", success_gate="sat_floor improves and strict >=81.5."),
        lac_exp("K100_CPI_CLAMP", "metric_action", "pressure_clamp", "K100 may be over-saturated; reduce global DG/sat pressure and check rollback/strict.", "strict >=84.0 or better latest stability.", updates={"lambda_adv": 0.34 * 0.82, "lambda_cons": 0.08 * 0.82, "lambda_group_ce": 0.075 * 0.82, "lambda_proto": 0.014 * 0.82, "lambda_supcon_id": 0.018 * 0.82, "lambda_fishr": 0.0015 * 0.82, "sat_view_prob": 0.66, "concat_sat_ce_weight": 0.82}),
        lac_exp("K100_RX_REPAIR", "metric_action", "rx_floor_repair", "K100 strict may still be rx-floor limited; increase worst-domain pressure without changing seed.", "rx floor improves and strict >=84.0.", updates={"lambda_adv": 0.40, "lambda_group_ce": 0.092, "group_ce_top_frac": 0.32, "groupdro_cap": 0.64}),
        no_sat_fd(k5, "K005_NOSAT_CONTROL", "Paired no-sat control for the K5 metric-action claims.", "If sat gate is useful, sat action beats no-sat on sat_floor without clean collapse."),
        no_sat_fd(k10, "K010_NOSAT_CONTROL", "Paired no-sat control for K10 ID-first late schedule.", "No-sat should clarify whether satellite load is useful or decorative."),
        scale_fd(k20, cid="K020_LOW_PRESSURE_CONTROL", axis="paired_control", action="low_pressure_control", pressure=0.70, sat=0.70, hypothesis="Control for over-clamping: too much pressure reduction should hurt rx/sat deficits.", success_gate="should not beat targeted RX/SAT actions on both strict and floor metrics."),
        scale_fd(k30, cid="K030_SAT_STRONG_CONTROL", axis="paired_control", action="sat_strong_control", sat=2.00, pressure=0.95, hypothesis="Control for blind satellite scaling; useful metric should reject it if clean strict drops.", success_gate="sat may improve, but strict loss >1pp marks bad tradeoff."),
        scale_fd(k50, cid="K050_CPI_CLAMP_CONTROL", axis="paired_control", action="pressure_clamp", pressure=0.78, sat=0.75, hypothesis="Check whether K50 anchor is already over-pressured or needs RX-specific repair.", success_gate="passes only if strict stays >=82 and latest gap shrinks."),
        lac_exp("K100_SAT_CONSERVE", "paired_control", "sat_conserve", "Conservative K100 satellite pressure tests whether LACSR06 gains are actually sat-driven.", "strict stays high; sat floor loss quantifies tradeoff.", updates={"sat_view_prob": 0.50, "concat_sat_ce_weight": 0.65, "sat_view_schedule": f"1@0.40:{LIGHT_SAT};140@0.55:{ALL_SAT}"}),
        lac_exp("K100_SAT_BOOST", "paired_control", "sat_boost", "Blind K100 satellite boost is a negative control for the metric gate.", "Reject if clean strict drops >1pp despite sat gain.", updates={"sat_view_prob": 0.95, "concat_sat_ce_weight": 1.15, "sat_view_schedule": f"1@0.80:{LIGHT_SAT};100@0.95:{ALL_SAT}"}),
        lac_exp("K100_SEEDCHECK_2028", "seed_check", "seed", "K100 anchor seed check; if this swings widely, metric conclusions need seed guard.", "strict >=83.5 and no large late rollback.", seed=2028),
    ]

    if len(rows) != 32:
        raise AssertionError(f"expected 32 candidates, got {len(rows)}")
    assigned: list[Experiment] = []
    for index, row in enumerate(rows):
        assigned.append(replace(row, gpu=index % 8))
    return assigned


def command_for(run_id: str, c: Experiment) -> str:
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


def render_launcher(run_id: str, rows: Sequence[Experiment], max_active_per_gpu: int, scheduler_hours: float) -> str:
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
            'echo "[CEN51-SAFDAF] run_id=${RUN_ID} dry_run=${DRY_RUN} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} max_seconds=${MAX_SCHEDULER_SECONDS}" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            'for gpu in 0 1 2 3 4 5 6 7; do echo "[CEN51-SAFDAF] initial gpu=${gpu} count=$(gpu_process_count "${gpu}")" | tee -a "${LOG_ROOT}/scheduler_events.tsv"; done',
            'START_TS="$(date +%s)"',
            "while true; do",
            "  reap_finished",
            '  NOW_TS="$(date +%s)"',
            "  if (( NOW_TS - START_TS < MAX_SCHEDULER_SECONDS )); then launch_available; fi",
            '  q_left="$(queued_left)"; r_left="$(running_left)"',
            '  echo "[CEN51-SAFDAF] heartbeat=$(date -Is) queued=${q_left} running=${r_left}" | tee -a "${LOG_ROOT}/scheduler_heartbeat.log"',
            '  if [[ "${q_left}" == "0" && "${r_left}" == "0" ]]; then break; fi',
            '  if [[ "${DRY_RUN}" == "1" ]]; then break; fi',
            '  sleep "${POLL_SECONDS}"',
            "done",
            'echo "[CEN51-SAFDAF] scheduler_complete $(date -Is)" | tee -a "${LOG_ROOT}/scheduler_events.tsv"',
            "",
        ]
    )
    return "\n".join(lines)


def render_report(run_id: str, rows: Sequence[Experiment], script_path: Path, matrix_path: Path, max_active_per_gpu: int, scheduler_hours: float) -> str:
    gpu_counts: dict[int, int] = {}
    shot_counts: dict[int, int] = {}
    axis_counts: dict[str, int] = {}
    for row in rows:
        gpu_counts[row.gpu] = gpu_counts.get(row.gpu, 0) + 1
        shot_counts[row.shot] = shot_counts.get(row.shot, 0) + 1
        axis_counts[row.axis] = axis_counts.get(row.axis, 0) + 1

    table = [
        "| ID | K | GPU | action | anchor strict/overall | success gate |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| `{row.cid}` | {row.shot} | {row.gpu} | {row.action} | "
            f"{row.target_strict:.2f}/{row.target_overall:.2f} | {row.success_gate} |"
        )

    anchors = [
        "| K | anchor | strict UDU | overall |",
        "|---:|---|---:|---:|",
    ]
    for shot in sorted(ANCHORS):
        anchor = ANCHORS[shot]
        anchors.append(f"| {shot} | `{anchor.name}` | {anchor.strict_udu:.2f} | {anchor.overall:.2f} |")

    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## 目标",
            "",
            "围绕截图中的各 K 最强候选做 6-8 小时 SAFD 锚点拟合实验。重点不是再堆一个固定参数表，而是验证诊断指标能不能产生可执行动作：降约束压力、补 receiver floor、调 satellite gate、或只做 seed guard。",
            "",
            "## 截图锚点",
            "",
            *anchors,
            "",
            "## 批次规模",
            "",
            f"- candidates: {len(rows)}",
            f"- max active per GPU: {max_active_per_gpu}",
            f"- scheduler launch window: {scheduler_hours:.1f} hours",
            f"- GPU candidate counts: `{json.dumps(gpu_counts, sort_keys=True)}`",
            f"- shot counts: `{json.dumps(shot_counts, sort_keys=True)}`",
            f"- axis counts: `{json.dumps(axis_counts, sort_keys=True)}`",
            "",
            "## 指标可用性判据",
            "",
            "- `anchor_gap`: 新候选相对截图 strict/overall 锚点的差值。",
            "- `late_penalty`: latest strict 相对 best/final strict 的回退；如果这个高，优先降压或提前 SWAD，而不是盲目加正则。",
            "- `rx_floor_deficit`: receiver floor 低时只允许 RX-aware 动作获胜，盲目全局加压视为负对照。",
            "- `sat_floor_deficit`: satellite floor 低时比较 gated sat 与 no-sat/strong-sat control，只有 clean strict 不崩才算有用。",
            "- 若某动作只改善一个展示指标但损伤 strict 或稳定性，则不进入 controller。",
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
            f"- remote logs: `{REMOTE_ROOT}/logs/{run_id}`",
            f"- remote runs: `{REMOTE_ROOT}/runs/{run_id}`",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_safd_anchor_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows = make_candidates()
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"
    manifest_path = report_dir / "manifest.tsv"

    script_path.write_text(render_launcher(run_id, rows, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(run_id, rows, script_path, matrix_path, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("cid\trun_name\tshot\tgpu\tseed\taxis\taction\tanchor_name\ttarget_strict\ttarget_overall\thypothesis\n")
        for row in rows:
            handle.write(
                "\t".join(
                    [
                        row.cid,
                        row.run_name,
                        str(row.shot),
                        str(row.gpu),
                        str(row.seed),
                        row.axis,
                        row.action,
                        row.anchor_name,
                        f"{row.target_strict:.2f}",
                        f"{row.target_overall:.2f}",
                        row.hypothesis,
                    ]
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "candidates": len(rows),
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
