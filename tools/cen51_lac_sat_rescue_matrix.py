#!/usr/bin/env python
"""Generate CEN51-LAC satellite rescue candidates after low-shot log audit.

The completed FS005 LAC primary showed that the clean/strict low-shot problem
is mostly fixed, while the satellite floor remains below RIEI+Sat. This matrix
therefore keeps strong receiver/day shortcut suppression explicit and tests
gated satellite re-entry without changing the CEN51 backbone.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from cen51_lac_validation_matrix import DEFAULT_SUMMARY, SAT_SCENARIOS, merge_params
from cen51_lowshot_config_search import Candidate, ShotEvidence, load_evidence, render_launcher


COMMON_PARAMS: Dict[str, object] = {
    "test_eval_start_epoch": 81,
    "test_eval_interval": 10,
    "seed": 1337,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 10.0,
    "collapse_guard_max_skipped_delta": 2,
    "lambda_adv": 0.45,
    "lambda_cons": 0.08,
    "lambda_group_ce": 0.10,
    "group_ce_top_frac": 0.35,
    "groupdro_tau": 0.50,
    "groupdro_cap": 0.65,
    "late_adv_min_scale": 0.70,
    "late_cons_min_scale": 0.45,
    "late_group_ce_min_scale": 0.80,
    "late_aug_min_scale": 0.35,
    "primary_udu_weight": 0.86,
}


def sat_rescue_params(shots: int) -> Dict[str, object]:
    if shots <= 5:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 128,
                "epochs": 180,
                "concat_sat_ce_weight": 0.60,
                "sat_view_prob": 0.55,
                "sat_view_schedule": f"1@0.30:clear_leo,mixed_orbit;120@0.55:{SAT_SCENARIOS};150@0.65:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0010,
                "sat_cons_start_epoch": 140,
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
                "swad_start_epoch": 50,
                "swad_tolerance": 0.70,
            },
        )
    if shots <= 10:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 128,
                "epochs": 185,
                "concat_sat_ce_weight": 0.70,
                "sat_view_prob": 0.62,
                "sat_view_schedule": f"1@0.35:clear_leo,mixed_orbit;120@0.60:{SAT_SCENARIOS};150@0.75:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0015,
                "sat_cons_start_epoch": 140,
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
                "swad_start_epoch": 60,
                "swad_tolerance": 0.75,
            },
        )
    if shots <= 20:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 256,
                "epochs": 200,
                "concat_sat_ce_weight": 0.82,
                "sat_view_prob": 0.70,
                "sat_view_schedule": f"1@0.45:clear_leo,mixed_orbit;115@0.70:{SAT_SCENARIOS};145@0.86:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0025,
                "sat_cons_start_epoch": 135,
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
                "swad_start_epoch": 70,
                "swad_tolerance": 0.75,
            },
        )
    if shots <= 50:
        return merge_params(
            COMMON_PARAMS,
            {
                "batch_size": 256,
                "epochs": 200,
                "concat_sat_ce_weight": 0.95,
                "sat_view_prob": 0.78,
                "sat_view_schedule": f"1@0.55:clear_leo,mixed_orbit;105@0.78:{SAT_SCENARIOS};135@0.98:{SAT_SCENARIOS}",
                "use_sat_consistency": True,
                "lambda_sat_cons": 0.0040,
                "sat_cons_start_epoch": 130,
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
                "swad_start_epoch": 80,
                "swad_tolerance": 0.65,
            },
        )
    return merge_params(
        COMMON_PARAMS,
        {
            "batch_size": 256,
            "epochs": 200,
            "concat_sat_ce_weight": 1.05,
            "sat_view_prob": 0.82,
            "sat_view_schedule": f"1@0.65:clear_leo,mixed_orbit;100@0.82:{SAT_SCENARIOS};130@1.00:{SAT_SCENARIOS}",
            "use_sat_consistency": True,
            "lambda_sat_cons": 0.0050,
            "sat_cons_start_epoch": 125,
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
            "swad_start_epoch": 80,
            "swad_tolerance": 0.60,
        },
    )


def lowreg_control_params() -> Dict[str, object]:
    return merge_params(
        sat_rescue_params(5),
        {
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
            "late_adv_min_scale": 0.35,
            "late_cons_min_scale": 0.20,
            "late_group_ce_min_scale": 0.35,
            "late_aug_min_scale": 0.45,
            "primary_udu_weight": 0.84,
            "swad_tolerance": 0.80,
        },
    )


def make_candidates(evidence: Dict[int, ShotEvidence], gpu_order: Sequence[int]) -> List[Candidate]:
    specs = [
        (5, "FS005_sat_rescue", "very-low-shot satellite rescue", sat_rescue_params(5), "sat avg strict >= 43.60 while clean strict UDU stays >= 68"),
        (5, "FS005_lowreg_control", "CLI-respected low-regularization control", lowreg_control_params(), "test whether the originally intended low-reg LAC improves clean strict toward RIEI-nosat 78.24"),
        (10, "FS010_sat_rescue", "low-shot satellite rescue", sat_rescue_params(10), "sat avg strict >= 47.31 while clean strict UDU stays >= 72"),
        (20, "FS020_sat_rescue", "low-mid satellite rescue", sat_rescue_params(20), "sat avg strict >= 44.39 while clean strict UDU stays >= 75"),
        (50, "FS050_monotonic_anchor", "higher-shot monotonic anchor", sat_rescue_params(50), "strict UDU >= 78.92 and sat avg strict improves over LAC primary"),
        (100, "FS100_r010_anchor", "r010 high-shot anchor", sat_rescue_params(100), "clean strict beats DRIFT-nosat 77.02 and sat avg approaches RIEI+Sat 47.82"),
    ]
    if len(gpu_order) < len(specs):
        raise ValueError(f"gpu_order must contain at least {len(specs)} GPU ids")
    candidates: List[Candidate] = []
    for idx, (shots, suffix, strategy, params, gate) in enumerate(specs, start=1):
        ev = evidence.get(shots if shots <= 50 else 50)
        rationale = (
            "LAC primary fixed clean/strict rollback but left satellite strict near 36%; "
            f"R04 reference FS{ev.shots:03d}: best_udu={ev.best_udu:.2f}, final_val={ev.final_val:.2f}."
        )
        candidate_id = f"CEN51_LACSR{idx:02d}_{suffix}"
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                run_name=f"{candidate_id}_r010",
                shots=shots,
                gpu=int(gpu_order[idx - 1]),
                strategy=strategy,
                rationale=rationale,
                success_gate=gate,
                params=params,
            )
        )
    return candidates


def payload_for(candidates: Sequence[Candidate], evidence: Dict[int, ShotEvidence]) -> Dict[str, object]:
    return {
        "objective": "Rescue the satellite floor after CEN51-LAC fixed the low-shot clean/strict rollback.",
        "diagnosis": {
            "FS005_LAC01": "best_val=77.77, latest/final_val=77.64, best_strict_udu=70.35, sat_avg_strict≈36.27",
            "root_cause": "LAC cut high-risk SAT/Fishr/Proto pressure and unexpectedly retained strong receiver/day suppression because exp_group preset overrode several explicit lambda/schedule flags.",
            "code_fix_dependency": "code/train.py now restores explicit preset-sensitive CLI flags after exp_group/slim presets.",
        },
        "baseline_targets": {
            "RIEI+Sat_0.005_sat_avg_strict": 43.60,
            "RIEI+Sat_0.01_sat_avg_strict": 47.31,
            "RIEI+Sat_0.02_sat_avg_strict": 44.39,
            "DRIFT_noSat_0.1_clean_strict": 77.02,
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
        f"# CEN51-LAC satellite rescue experiment - {run_id}",
        "",
        "## Objective",
        "",
        str(payload["objective"]),
        "",
        "## Diagnosis",
        "",
        f"- FS005 LAC01: {payload['diagnosis']['FS005_LAC01']}",
        f"- root cause: {payload['diagnosis']['root_cause']}",
        f"- code dependency: {payload['diagnosis']['code_fix_dependency']}",
        "",
        "## Local artifacts",
        "",
        f"- matrix: `{matrix_path}`",
        f"- launcher: `{launcher_path}`",
        "",
        "## Candidate matrix",
        "",
        "| candidate | shot cap | gpu | strategy | success gate |",
        "|---|---:|---:|---|---|",
    ]
    for cand in payload["candidates"]:
        lines.append(
            f"| `{cand['candidate_id']}` | {cand['shots']} | {cand['gpu']} | {cand['strategy']} | {cand['success_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Launch policy",
            "",
            "Launch only after a fresh N607 preflight and process/GPU capacity audit. If only one slot is available, launch `CEN51_LACSR01_FS005_sat_rescue` first.",
            "",
            "## Verification before N607 launch",
            "",
            "- `conda activate ssr-gpu; python -m py_compile code/train.py tools/cen51_lac_sat_rescue_matrix.py tools/cen51_lowshot_config_search.py`",
            "- `conda activate ssr-gpu; PYTHONPATH=E:\\type10-7\\code;E:\\type10-7\\code\\FJMP python -m pytest tests/test_cen51_lowshot_cli_overrides.py tests/test_post_stage_trainers.py -q`",
            "- `bash -n code/scripts/launch_<run_id>.sh`",
            "- local dry-run must show exactly one `--wisig_max_train_per_combo` per candidate.",
            "",
            "## Expected remote paths",
            "",
            f"- logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/{run_id}/`",
            f"- runs: `/home/szu2070436088/2510044040/CV-SincNet/runs/{run_id}/`",
            "",
            "## Current status",
            "",
            "Generated locally. Not running until synced, remote dry-run verified, capacity-gated, launched, and startup-health checked.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=Path("automation_reports") / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=Path("code") / "scripts")
    parser.add_argument("--gpu-order", default="4,5,6,7,0,1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_lac_sat_rescue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evidence = load_evidence(args.summary_csv)
    gpu_order = [int(part.strip()) for part in args.gpu_order.split(",") if part.strip()]
    candidates = make_candidates(evidence, gpu_order)

    report_dir = args.output_root / run_id
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = artifact_dir / "cen51_lac_sat_rescue_matrix.json"
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
