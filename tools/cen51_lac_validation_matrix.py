#!/usr/bin/env python
"""Generate a CEN51-LAC validation matrix.

This is a launch-planning tool, not an online controller. It applies the
CEN51-LAC bounds from the design report to produce eight bounded experiments:
one primary LAC branch for each shot bucket plus three targeted ablations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from cen51_lowshot_config_search import (
    SAT_SCENARIOS,
    Candidate,
    ShotEvidence,
    arg_pairs,
    load_evidence,
    render_launcher,
)


DEFAULT_SUMMARY = (
    Path("automation_reports")
    / "CV-SincNet"
    / "fewshot_cen51_r04_20260608_194541"
    / "full_log_analysis"
    / "fewshot_curve_summary.csv"
)

DEFAULT_SPEC = (
    Path("automation_reports")
    / "CV-SincNet"
    / "cen51_adaptive_loss_controller_20260609_001622"
    / "artifacts"
    / "adaptive_loss_controller_spec.json"
)


COMMON_PARAMS: Dict[str, object] = {
    "test_eval_start_epoch": 81,
    "test_eval_interval": 10,
    "seed": 1337,
    "primary_udu_weight": 0.84,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 10.0,
    "collapse_guard_max_skipped_delta": 2,
}


def merge_params(*parts: Dict[str, object]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for part in parts:
        merged.update(part)
    return merged


def lac_primary_params(shots: int) -> Dict[str, object]:
    if shots <= 5:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 128,
                "epochs": 170,
                "concat_sat_ce_weight": 0.35,
                "sat_view_prob": 0.35,
                "sat_view_schedule": f"1@0.30:clear_leo,mixed_orbit;130@0.35:{SAT_SCENARIOS}",
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
                "swad_start_epoch": 50,
                "swad_tolerance": 0.80,
            },
        )
    if shots <= 10:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 128,
                "epochs": 180,
                "concat_sat_ce_weight": 0.45,
                "sat_view_prob": 0.45,
                "sat_view_schedule": f"1@0.35:clear_leo,mixed_orbit;135@0.45:{SAT_SCENARIOS}",
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
                "mixstyle_strength": 0.42,
                "mixstyle_late_start": 85,
                "mixstyle_late_min_p": 0.025,
                "mixstyle_late_min_strength": 0.20,
                "late_stable_start": 85,
                "late_stable_ramp_epochs": 25,
                "late_adv_min_scale": 0.38,
                "late_cons_min_scale": 0.22,
                "late_group_ce_min_scale": 0.38,
                "late_aug_min_scale": 0.48,
                "swad_start_epoch": 60,
                "swad_tolerance": 0.85,
            },
        )
    if shots <= 20:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 256,
                "epochs": 200,
                "concat_sat_ce_weight": 0.60,
                "sat_view_prob": 0.55,
                "sat_view_schedule": f"1@0.45:clear_leo,mixed_orbit;145@0.60:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0010,
                "sat_cons_start_epoch": 160,
                "lambda_adv": 0.24,
                "lambda_cons": 0.055,
                "lambda_group_ce": 0.050,
                "group_ce_top_frac": 0.22,
                "groupdro_tau": 0.35,
                "groupdro_cap": 0.48,
                "lambda_proto": 0.008,
                "lambda_supcon_id": 0.010,
                "lambda_fishr": 0.0005,
                "mixstyle_p": 0.13,
                "mixstyle_strength": 0.50,
                "mixstyle_late_start": 100,
                "mixstyle_late_min_p": 0.030,
                "mixstyle_late_min_strength": 0.24,
                "late_stable_start": 100,
                "late_stable_ramp_epochs": 25,
                "late_adv_min_scale": 0.45,
                "late_cons_min_scale": 0.28,
                "late_group_ce_min_scale": 0.45,
                "late_aug_min_scale": 0.55,
                "swad_start_epoch": 70,
                "swad_tolerance": 0.80,
            },
        )
    if shots <= 30:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 256,
                "epochs": 200,
                "concat_sat_ce_weight": 0.68,
                "sat_view_prob": 0.60,
                "sat_view_schedule": f"1@0.50:clear_leo,mixed_orbit;145@0.68:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0015,
                "sat_cons_start_epoch": 155,
                "lambda_adv": 0.28,
                "lambda_cons": 0.065,
                "lambda_group_ce": 0.060,
                "group_ce_top_frac": 0.24,
                "groupdro_tau": 0.38,
                "groupdro_cap": 0.52,
                "lambda_proto": 0.010,
                "lambda_supcon_id": 0.012,
                "lambda_fishr": 0.0008,
                "mixstyle_p": 0.14,
                "mixstyle_strength": 0.54,
                "mixstyle_late_start": 105,
                "mixstyle_late_min_p": 0.035,
                "mixstyle_late_min_strength": 0.26,
                "late_stable_start": 105,
                "late_stable_ramp_epochs": 25,
                "late_adv_min_scale": 0.50,
                "late_cons_min_scale": 0.32,
                "late_group_ce_min_scale": 0.50,
                "late_aug_min_scale": 0.58,
                "swad_start_epoch": 75,
                "swad_tolerance": 0.80,
            },
        )
    return merge_params(
        COMMON_PARAMS,
        {
            "batch_size": 256,
            "epochs": 200,
            "concat_sat_ce_weight": 0.82,
            "sat_view_prob": 0.72,
            "sat_view_schedule": f"1@0.60:clear_leo,mixed_orbit;140@0.82:{SAT_SCENARIOS}",
            "use_sat_consistency": True,
            "lambda_sat_cons": 0.0025,
            "sat_cons_start_epoch": 150,
            "lambda_adv": 0.34,
            "lambda_cons": 0.080,
            "lambda_group_ce": 0.075,
            "group_ce_top_frac": 0.28,
            "groupdro_tau": 0.42,
            "groupdro_cap": 0.58,
            "lambda_proto": 0.014,
            "lambda_supcon_id": 0.018,
            "lambda_fishr": 0.0015,
            "mixstyle_p": 0.17,
            "mixstyle_strength": 0.62,
            "mixstyle_late_start": 110,
            "mixstyle_late_min_p": 0.045,
            "mixstyle_late_min_strength": 0.30,
            "late_stable_start": 110,
            "late_stable_ramp_epochs": 25,
            "late_adv_min_scale": 0.55,
            "late_cons_min_scale": 0.36,
            "late_group_ce_min_scale": 0.55,
            "late_aug_min_scale": 0.62,
            "primary_udu_weight": 0.86,
            "swad_start_epoch": 80,
            "swad_tolerance": 0.80,
        },
    )


def lac_ablation_params(kind: str, shots: int) -> Dict[str, object]:
    base = lac_primary_params(shots)
    if kind == "sat_min":
        base.update(
            {
                "concat_sat_ce_weight": 0.20,
                "sat_view_prob": 0.20,
                "sat_view_schedule": "1@0.20:clear_leo,mixed_orbit",
                "lambda_sat_cons": 0.0,
                "sat_cons_start_epoch": 999,
                "mixstyle_p": 0.08,
                "mixstyle_strength": 0.30,
            }
        )
    elif kind == "rx_floor":
        base.update(
            {
                "concat_sat_ce_weight": 0.40,
                "sat_view_prob": 0.38,
                "sat_view_schedule": f"1@0.30:clear_leo,mixed_orbit;145@0.40:{SAT_SCENARIOS}",
                "lambda_sat_cons": 0.0,
                "sat_cons_start_epoch": 999,
                "lambda_adv": 0.22,
                "lambda_cons": 0.045,
                "lambda_group_ce": 0.040,
                "lambda_proto": 0.006,
                "lambda_supcon_id": 0.006,
                "lambda_fishr": 0.0,
                "primary_udu_weight": 0.90,
            }
        )
    elif kind == "sat_reentry":
        base.update(
            {
                "use_sat_consistency": True,
                "concat_sat_ce_weight": 0.70,
                "sat_view_prob": 0.60,
                "sat_view_schedule": f"1@0.45:clear_leo,mixed_orbit;145@0.70:{SAT_SCENARIOS}",
                "lambda_sat_cons": 0.0015,
                "sat_cons_start_epoch": 155,
                "lambda_group_ce": 0.052,
                "lambda_proto": 0.010,
                "lambda_supcon_id": 0.012,
            }
        )
    else:
        raise ValueError(f"unknown ablation kind: {kind}")
    return base


def make_candidates(evidence: Dict[int, ShotEvidence], gpu_order: Sequence[int]) -> List[Candidate]:
    specs = [
        (5, "lac_primary", "LAC very-low primary", lac_primary_params(5)),
        (10, "lac_primary", "LAC very-low primary", lac_primary_params(10)),
        (20, "lac_primary", "LAC low-mid primary", lac_primary_params(20)),
        (30, "lac_primary", "LAC stable-low primary", lac_primary_params(30)),
        (50, "lac_primary", "LAC medium primary", lac_primary_params(50)),
        (5, "sat_min_ablate", "very-low SAT pressure ablation", lac_ablation_params("sat_min", 5)),
        (10, "rx_floor_probe", "very-low receiver-floor probe", lac_ablation_params("rx_floor", 10)),
        (20, "sat_reentry_probe", "low-mid delayed SAT re-entry probe", lac_ablation_params("sat_reentry", 20)),
    ]
    if len(gpu_order) < len(specs):
        raise ValueError(f"gpu_order must contain at least {len(specs)} GPU ids")
    candidates: List[Candidate] = []
    for idx, (shots, suffix, strategy, params) in enumerate(specs, start=1):
        ev = evidence[shots]
        candidate_id = f"CEN51_LAC{idx:02d}_FS{shots:03d}_{suffix}"
        run_name = f"{candidate_id}_r010"
        if shots <= 10:
            gate = (
                f"strict UDU > {ev.best_udu:.2f}, rollback <= 3, "
                f"receiver floor >= {ev.receiver_floor:.2f}"
            )
        elif shots <= 20:
            gate = (
                f"strict UDU > {ev.best_udu:.2f}, receiver floor > {ev.receiver_floor:.2f}, "
                "clean val within 1 point"
            )
        else:
            gate = (
                f"strict UDU >= {ev.best_udu:.2f}, receiver floor >= {ev.receiver_floor:.2f}, "
                "no SAT floor regression"
            )
        rationale = (
            f"R04 baseline FS{shots:03d}: best_val={ev.best_val:.2f}@E{ev.best_val_epoch}, "
            f"best_udu={ev.best_udu:.2f}@E{ev.best_udu_epoch}, final_udu={ev.final_udu:.2f}, "
            f"receiver_floor={ev.receiver_floor:.2f}({ev.receiver_floor_name}), "
            f"rollback={ev.val_drop:.2f}."
        )
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                run_name=run_name,
                shots=shots,
                gpu=int(gpu_order[idx - 1]),
                strategy=strategy,
                rationale=rationale,
                success_gate=gate,
                params=params,
            )
        )
    return candidates


def validate_bounds(candidates: Sequence[Candidate], spec: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    bucket_for_shot = {
        5: "very_low",
        10: "very_low",
        20: "low_mid",
        30: "stable_low",
        50: "medium",
    }
    bounds = spec.get("bounds", {})
    for cand in candidates:
        bucket = bucket_for_shot[cand.shots]
        bucket_bounds = bounds.get(bucket, {})
        for key, value in cand.params.items():
            if key not in bucket_bounds:
                continue
            lo, hi = bucket_bounds[key]
            if not (float(lo) <= float(value) <= float(hi)):
                issues.append(
                    f"{cand.candidate_id}: {key}={value} outside {bucket} bounds [{lo}, {hi}]"
                )
    return issues


def payload_for(candidates: Sequence[Candidate], evidence: Dict[int, ShotEvidence]) -> Dict[str, object]:
    return {
        "objective": "Validate whether CEN51-LAC bounded loss-component settings improve low-shot robustness versus completed CEN51_R04 few-shot baselines.",
        "comparison_target": "CENCEN51_R04_sat_joint_guard_no_overdrive few-shot R04 results from 2026-06-08.",
        "selection_score": {
            "name": "ALAS-lite",
            "formula": "0.30*best_val + 0.35*best_strict_udu + 0.20*receiver_floor + 0.10*sat_floor + 0.05*stability - rollback_penalty",
            "rollback_penalty": "max(0, best_val-final_val-3) + max(0, best_strict_udu-final_strict_udu-3)",
            "promotion_rule": "Promote a shot bucket only when strict UDU and receiver floor do not regress and rollback shrinks materially.",
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


def render_report(run_id: str, payload: Dict[str, object], launcher_path: Path, matrix_path: Path, spec_path: Path) -> str:
    lines = [
        f"# CEN51-LAC validation experiment - {run_id}",
        "",
        "## Objective",
        "",
        str(payload["objective"]),
        "",
        "## Hypothesis",
        "",
        "- 5/10 shot: lowering high-risk SAT/DG/Fishr/SupCon pressure should shrink best-to-final rollback and improve strict UDU stability.",
        "- 20 shot: bounded delayed SAT re-entry and receiver-floor pressure should improve strict UDU or rx8 floor without clean-val regression.",
        "- 30/50 shot: bounded LAC settings should match the stable R04 frontier while avoiding satellite-specialist collapse.",
        "",
        "## Local artifacts",
        "",
        f"- spec: `{spec_path}`",
        f"- matrix: `{matrix_path}`",
        f"- launcher: `{launcher_path}`",
        "",
        "## Selection score",
        "",
        f"- name: `{payload['selection_score']['name']}`",
        f"- formula: `{payload['selection_score']['formula']}`",
        f"- rollback penalty: `{payload['selection_score']['rollback_penalty']}`",
        f"- promotion rule: {payload['selection_score']['promotion_rule']}",
        "",
        "## Candidate matrix",
        "",
        "| candidate | shot | gpu | strategy | success gate |",
        "|---|---:|---:|---|---|",
    ]
    for cand in payload["candidates"]:
        lines.append(
            f"| `{cand['candidate_id']}` | {cand['shots']} | {cand['gpu']} | {cand['strategy']} | {cand['success_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Verification before N607 launch",
            "",
            "- `conda activate ssr-gpu; python -m py_compile tools/cen51_lac_validation_matrix.py tools/cen51_lowshot_config_search.py code/train.py`",
            "- `bash -n code/scripts/launch_<run_id>.sh`",
            "- `bash code/scripts/launch_<run_id>.sh --dry-run` on N607 after sync",
            "",
            "## Expected remote paths",
            "",
            f"- logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}/`",
            f"- runs: `/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}/`",
            "",
            "## Metrics to inspect after completion",
            "",
            "- best/final `val_tx` and best-to-final rollback",
            "- best/final `test_unseen_day_unseen_rx` strict UDU",
            "- per-receiver floor, especially rx8/rx11",
            "- `[SAT-TEST]` strict UDU floor across clear/rain/low-elev/storm/mixed scenarios",
            "- warning count, skipped backward count, collapse guard activity",
            "",
            "## Current status",
            "",
            "Generated locally. Sync, remote dry-run, launch, and startup-health are required before this can be treated as running evidence.",
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
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids assigned to candidates in order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_lac_validate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evidence = load_evidence(args.summary_csv)
    spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
    gpu_order = [int(part.strip()) for part in args.gpu_order.split(",") if part.strip()]
    candidates = make_candidates(evidence, gpu_order)
    issues = validate_bounds(candidates, spec)
    if issues:
        raise SystemExit("bound validation failed:\n" + "\n".join(issues))

    report_dir = args.output_root / run_id
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = artifact_dir / "cen51_lac_validation_matrix.json"
    launcher_path = args.scripts_dir / f"launch_{run_id}.sh"
    report_path = report_dir / "report.md"

    payload = payload_for(candidates, evidence)
    matrix_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    launcher_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    report_path.write_text(render_report(run_id, payload, launcher_path, matrix_path, args.spec_json), encoding="utf-8")

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
