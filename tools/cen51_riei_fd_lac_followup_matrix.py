#!/usr/bin/env python
"""Generate a RIEI-inspired CEN51 FD/LAC follow-up matrix.

This generator complements the already launched CEN51-LAC primary branches.
It does not add new training logic. Instead, it keeps the CVS backbone and
turns the RIEI few-shot lesson into bounded settings:

* extreme low-shot branches reduce SAT/domain pressure;
* receiver-floor probes keep explicit nuisance suppression;
* higher-shot anchors reintroduce identity capacity and SAT robustness.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from cen51_lac_validation_matrix import (
    DEFAULT_SPEC,
    DEFAULT_SUMMARY,
    SAT_SCENARIOS,
    lac_ablation_params,
    lac_primary_params,
    merge_params,
)
from cen51_lowshot_config_search import Candidate, ShotEvidence, load_evidence, render_launcher


COMMON_FOLLOWUP_PARAMS: Dict[str, object] = {
    "test_eval_start_epoch": 81,
    "test_eval_interval": 10,
    "seed": 1337,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 10.0,
    "collapse_guard_max_skipped_delta": 2,
}


RIEI_DRIFT_TARGETS: Dict[str, Dict[str, float]] = {
    "0.005": {
        "riei_nosat_clean_strict": 78.24,
        "riei_sat_clean_strict": 66.73,
        "riei_sat_avg_strict": 43.60,
        "drift_nosat_clean_strict": 60.17,
        "drift_sat_clean_strict": 60.42,
        "drift_sat_avg_strict": 18.92,
    },
    "0.01": {
        "riei_nosat_clean_strict": 68.50,
        "riei_sat_clean_strict": 70.29,
        "riei_sat_avg_strict": 47.31,
        "drift_nosat_clean_strict": 55.59,
        "drift_sat_clean_strict": 66.06,
        "drift_sat_avg_strict": 30.53,
    },
    "0.02": {
        "riei_nosat_clean_strict": 61.01,
        "riei_sat_clean_strict": 63.26,
        "riei_sat_avg_strict": 44.39,
        "drift_nosat_clean_strict": 65.72,
        "drift_sat_clean_strict": 62.84,
        "drift_sat_avg_strict": 41.02,
    },
    "0.05": {
        "riei_nosat_clean_strict": 66.39,
        "riei_sat_clean_strict": 57.86,
        "riei_sat_avg_strict": 42.76,
        "drift_nosat_clean_strict": 74.40,
        "drift_sat_clean_strict": 65.38,
        "drift_sat_avg_strict": 39.63,
    },
    "0.1": {
        "riei_nosat_clean_strict": 59.83,
        "riei_sat_clean_strict": 46.32,
        "riei_sat_avg_strict": 47.82,
        "drift_nosat_clean_strict": 77.02,
        "drift_sat_clean_strict": 57.30,
        "drift_sat_avg_strict": 42.89,
    },
}


def fs100_anchor_params() -> Dict[str, object]:
    """Guard the high-shot side so low-shot tuning does not cap identity signal."""

    return merge_params(
        COMMON_FOLLOWUP_PARAMS,
        {
            "batch_size": 256,
            "epochs": 200,
            "concat_sat_ce_weight": 0.95,
            "sat_view_prob": 0.78,
            "sat_view_schedule": f"1@0.65:clear_leo,mixed_orbit;130@0.90:{SAT_SCENARIOS}",
            "use_sat_consistency": True,
            "lambda_sat_cons": 0.0040,
            "sat_cons_start_epoch": 140,
            "lambda_adv": 0.40,
            "lambda_cons": 0.090,
            "lambda_group_ce": 0.080,
            "group_ce_top_frac": 0.30,
            "groupdro_tau": 0.45,
            "groupdro_cap": 0.60,
            "lambda_proto": 0.016,
            "lambda_supcon_id": 0.020,
            "lambda_fishr": 0.0020,
            "mixstyle_p": 0.18,
            "mixstyle_strength": 0.65,
            "mixstyle_late_start": 110,
            "mixstyle_late_min_p": 0.050,
            "mixstyle_late_min_strength": 0.32,
            "late_stable_start": 110,
            "late_stable_ramp_epochs": 25,
            "late_adv_min_scale": 0.60,
            "late_cons_min_scale": 0.40,
            "late_group_ce_min_scale": 0.60,
            "late_aug_min_scale": 0.66,
            "primary_udu_weight": 0.86,
            "swad_start_epoch": 80,
            "swad_tolerance": 0.55,
        },
    )


def candidate(
    idx: int,
    candidate_id: str,
    run_name: str,
    shots: int,
    gpu: int,
    strategy: str,
    rationale: str,
    gate: str,
    params: Dict[str, object],
) -> Candidate:
    return Candidate(
        candidate_id=f"CEN51_RIEIFD{idx:02d}_{candidate_id}",
        run_name=f"CEN51_RIEIFD{idx:02d}_{run_name}",
        shots=shots,
        gpu=gpu,
        strategy=strategy,
        rationale=rationale,
        success_gate=gate,
        params=merge_params(COMMON_FOLLOWUP_PARAMS, params),
    )


def make_candidates(evidence: Dict[int, ShotEvidence], gpu_order: Sequence[int]) -> List[Candidate]:
    if len(gpu_order) < 4:
        raise ValueError("gpu_order must contain at least 4 GPU ids")
    ev5 = evidence[5]
    ev10 = evidence[10]
    ev50 = evidence[50]
    candidates = [
        candidate(
            1,
            "FS050_lac_primary",
            "FS050_lac_primary_r010",
            50,
            int(gpu_order[0]),
            "higher-shot LAC guard",
            (
                f"FS050 R04 was stable but still receiver-floor limited: "
                f"best_udu={ev50.best_udu:.2f}, floor={ev50.receiver_floor:.2f}({ev50.receiver_floor_name})."
            ),
            "strict UDU >= 78.92, receiver floor >= 65.59, no SAT floor regression",
            lac_primary_params(50),
        ),
        candidate(
            2,
            "FS005_sat_min_ablate",
            "FS005_sat_min_ablate_r010",
            5,
            int(gpu_order[1]),
            "RIEI low-shot SAT-min ablation",
            (
                f"FS005 had rollback={ev5.val_drop:.2f}; RIEI no-sat suggests that "
                "extreme low shot should first protect the clean TX boundary."
            ),
            "strict UDU > 56.06, rollback <= 3, receiver floor >= 38.15",
            lac_ablation_params("sat_min", 5),
        ),
        candidate(
            3,
            "FS010_rx_floor_probe",
            "FS010_rx_floor_probe_r010",
            10,
            int(gpu_order[2]),
            "receiver-floor explicit suppression probe",
            (
                f"FS010 floor was {ev10.receiver_floor:.2f} on {ev10.receiver_floor_name}; "
                "increase receiver-floor pressure without enabling early SAT consistency."
            ),
            "strict UDU > 64.96, rollback <= 3, receiver floor >= 32.61 and preferably +8 points",
            lac_ablation_params("rx_floor", 10),
        ),
        candidate(
            4,
            "FS100_r010_anchor_guard",
            "FS100_r010_anchor_guard",
            100,
            int(gpu_order[3]),
            "ratio-0.1 high-shot anchor guard",
            (
                "Use the same r010 pool with a 100-per-combo cap to verify that "
                "low-shot FD/LAC tuning does not cap higher-shot identity capacity."
            ),
            "ALAS-lite, clean strict UDU, and receiver floor must not regress versus FS050; target DRIFT-nosat 0.1 clean strict 77.02 and RIEI+Sat sat avg strict 47.82",
            fs100_anchor_params(),
        ),
    ]
    return candidates


def payload_for(candidates: Sequence[Candidate], evidence: Dict[int, ShotEvidence]) -> Dict[str, object]:
    return {
        "objective": (
            "Complete the RIEI-inspired CEN51 FD/LAC follow-up without changing the CVS backbone: "
            "protect very-low-shot TX identity, suppress receiver/day/channel shortcuts, and verify "
            "that higher shots recover stronger performance."
        ),
        "comparison_target": {
            "cvs": "CENCEN51_R04_sat_joint_guard_no_overdrive few-shot R04 and current CEN51-LAC primary branches.",
            "baselines": "Fixed RIEI/DRIFT low-ratio sweeps, sat and no-sat variants.",
        },
        "selection_score": {
            "name": "ALAS-lite + monotonic guard",
            "formula": (
                "0.30*best_val + 0.35*best_strict_udu + 0.20*receiver_floor "
                "+ 0.10*sat_floor + 0.05*stability - rollback_penalty"
            ),
            "rollback_penalty": "max(0, best_val-final_val-3) + max(0, best_strict_udu-final_strict_udu-3)",
            "monotonic_guard": (
                "From FS020->FS030->FS050->FS100, ALAS-lite, clean strict UDU, and receiver floor "
                "must not drop by more than 1 point unless the branch is explicitly marked as a SAT-specialist ablation."
            ),
        },
        "riei_drift_targets": RIEI_DRIFT_TARGETS,
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
        "already_running_primary_branches": [
            "CEN51_LAC01_FS005_lac_primary",
            "CEN51_LAC02_FS010_lac_primary",
            "CEN51_LAC03_FS020_lac_primary",
            "CEN51_LAC04_FS030_lac_primary",
        ],
        "followup_candidates": [
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
        f"# CEN51 RIEI-inspired FD/LAC follow-up - {run_id}",
        "",
        "## Objective",
        "",
        str(payload["objective"]),
        "",
        "## Integrated mechanism",
        "",
        "- Keep the CVS/CEN51 backbone; do not replace it with RIEI or DRIFT.",
        "- Use existing z_id/z_dom, domain adversarial, covariance orthogonality, same-TX consistency, prototype/SupCon/Fishr, and gated SAT views as FD-lite proxies.",
        "- Very-low-shot settings reduce SAT consistency, Fishr, SupCon, prototype, and MixStyle pressure to avoid erasing scarce TX identity signal.",
        "- Higher-shot settings reintroduce SAT and nuisance suppression so extra samples improve the clean and strict UDU curve instead of overfitting receiver/day shortcuts.",
        "",
        "## Local artifacts",
        "",
        f"- matrix: `{matrix_path}`",
        f"- launcher: `{launcher_path}`",
        "",
        "## Selection score",
        "",
        f"- name: `{payload['selection_score']['name']}`",
        f"- formula: `{payload['selection_score']['formula']}`",
        f"- rollback penalty: `{payload['selection_score']['rollback_penalty']}`",
        f"- monotonic guard: {payload['selection_score']['monotonic_guard']}",
        "",
        "## Follow-up candidate matrix",
        "",
        "| candidate | shot cap | gpu | strategy | success gate |",
        "|---|---:|---:|---|---|",
    ]
    for cand in payload["followup_candidates"]:
        lines.append(
            f"| `{cand['candidate_id']}` | {cand['shots']} | {cand['gpu']} | {cand['strategy']} | {cand['success_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Already running branches not duplicated",
            "",
        ]
    )
    for branch in payload["already_running_primary_branches"]:
        lines.append(f"- `{branch}`")
    lines.extend(
        [
            "",
            "## Verification before N607 launch",
            "",
            "- `conda activate ssr-gpu; python -m py_compile tools/cen51_riei_fd_lac_followup_matrix.py tools/cen51_lac_validation_matrix.py tools/cen51_lowshot_config_search.py code/train.py`",
            "- `bash -n code/scripts/launch_<run_id>.sh`",
            "- `bash code/scripts/launch_<run_id>.sh --dry-run` on N607 after sync",
            "",
            "## Expected remote paths",
            "",
            f"- logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}/`",
            f"- runs: `/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}/`",
            "",
            "## Launch policy",
            "",
            "Do not launch while the current paper-original RIEI/DRIFT queue plus CEN51-LAC primary branches keep GPUs at the two-train-process capacity limit. Launch only after a fresh N607 preflight and capacity audit.",
            "",
            "## Metrics to inspect after completion",
            "",
            "- best/final val_tx and best-to-final rollback",
            "- best/final test_unseen_day_unseen_rx strict UDU",
            "- per-receiver floor, especially rx8/rx11",
            "- SAT strict UDU floor and average across clear/rain/low-elev/storm/mixed",
            "- ALAS-lite monotonicity from FS020/FS030/FS050/FS100",
            "",
            "## Current status",
            "",
            "Generated locally. This is a launch-ready follow-up queue, not running evidence until synced, dry-run verified, capacity-gated, launched, and startup-health checked.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--spec-json", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=Path("automation_reports") / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=Path("code") / "scripts")
    parser.add_argument(
        "--gpu-order",
        default="0,1,2,3",
        help="Comma-separated GPU ids assigned to follow-up candidates in order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_riei_fd_lac_followup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evidence = load_evidence(args.summary_csv)
    if args.spec_json and not args.spec_json.exists():
        raise SystemExit(f"missing spec json: {args.spec_json}")
    gpu_order = [int(part.strip()) for part in args.gpu_order.split(",") if part.strip()]
    candidates = make_candidates(evidence, gpu_order)

    report_dir = args.output_root / run_id
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = artifact_dir / "cen51_riei_fd_lac_followup_matrix.json"
    launcher_path = args.scripts_dir / f"launch_{run_id}.sh"
    report_path = report_dir / "report.md"

    payload = payload_for(candidates, evidence)
    matrix_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    launcher_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    report_path.write_text(render_report(run_id, payload, launcher_path, matrix_path), encoding="utf-8")

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
