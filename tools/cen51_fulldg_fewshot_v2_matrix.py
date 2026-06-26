#!/usr/bin/env python
"""Generate the CEN51 full-DG few-shot V2 follow-up matrix.

This matrix is intentionally scoped to per-combo caps below 100.  For 100+
samples per combo, the intended default is to return to the original CEN51
configuration rather than keep escalating few-shot-specific constraints.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from cen51_fulldg_shotaware_lacsr_fd_matrix import (
    ALL_SAT,
    LIGHT_SAT,
    Candidate,
    render_launcher,
)


def candidate(
    cid: str,
    shots: int,
    gpu: int,
    seed: int,
    epochs: int,
    swad_start: int,
    use_aug: bool,
    use_mixstyle: bool,
    sat_prob: float,
    sat_start: int,
    sat_scenarios: str,
    lambda_dom: float,
    lambda_adv: float,
    lambda_orth: float,
    lambda_cons: float,
    lambda_group_ce: float,
    lambda_proto: float,
    lambda_supcon: float,
    lambda_fishr: float,
    lambda_feature_norm: float,
    feature_norm_mode: str,
    feature_norm_target: float,
    group_top_frac: float,
    group_tau: float,
    group_cap: float,
    mixstyle_p: float,
    mixstyle_strength: float,
    aug_scale_min: float,
    aug_scale_max: float,
    rationale: str,
    success_gate: str,
) -> Candidate:
    return Candidate(
        cid=cid,
        run_name=f"CEN51_FDLOWV2_{cid}",
        shots=shots,
        gpu=gpu,
        seed=seed,
        epochs=epochs,
        swad_start=swad_start,
        batch_size=128 if shots <= 10 else 256,
        use_aug=use_aug,
        use_mixstyle=use_mixstyle,
        sat_prob=sat_prob,
        sat_start=sat_start,
        sat_scenarios=sat_scenarios,
        lambda_dom=lambda_dom,
        lambda_adv=lambda_adv,
        lambda_orth=lambda_orth,
        lambda_cons=lambda_cons,
        lambda_group_ce=lambda_group_ce,
        lambda_proto=lambda_proto,
        lambda_supcon=lambda_supcon,
        lambda_fishr=lambda_fishr,
        lambda_feature_norm=lambda_feature_norm,
        feature_norm_mode=feature_norm_mode,
        feature_norm_target=feature_norm_target,
        group_top_frac=group_top_frac,
        group_tau=group_tau,
        group_cap=group_cap,
        mixstyle_p=mixstyle_p,
        mixstyle_strength=mixstyle_strength,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        rationale=rationale,
        success_gate=success_gate,
    )


def make_candidates() -> list[Candidate]:
    return [
        candidate(
            "FS005_HINGE6_SATMIN_1337",
            5,
            0,
            1337,
            195,
            55,
            False,
            False,
            0.06,
            1,
            LIGHT_SAT,
            0.34,
            0.12,
            0.012,
            0.004,
            0.004,
            0.0006,
            0.0006,
            0.0,
            0.0008,
            "hinge",
            6.0,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "K5 best strict came from SATMIN hinge, but target=8 was mostly inactive; lower the hinge target and reduce early satellite dilution.",
            "strict >= 74.6, val >= 89.5, worst_rx >= 64.",
        ),
        candidate(
            "FS005_HINGE6_SATMIN_2028",
            5,
            1,
            2028,
            195,
            55,
            False,
            False,
            0.06,
            1,
            LIGHT_SAT,
            0.34,
            0.12,
            0.012,
            0.004,
            0.004,
            0.0006,
            0.0006,
            0.0,
            0.0008,
            "hinge",
            6.0,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "Seed2028 checks whether the K5 validation leader can keep val >= 90 after stricter norm gating.",
            "val >= 90.0, strict >= 73.5, worst_rx >= 58.",
        ),
        candidate(
            "FS005_IDFIRST_LATE_2029",
            5,
            2,
            2029,
            195,
            60,
            False,
            False,
            0.06,
            95,
            LIGHT_SAT,
            0.34,
            0.10,
            0.012,
            0.003,
            0.004,
            0.0005,
            0.0005,
            0.0,
            0.0008,
            "hinge",
            6.0,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "K5 seed2029 was the unstable low-val seed; delay full-DG satellite until the TX boundary is established.",
            "val >= 88.5, strict >= 72.5, seed gap versus 1337 <= 2.5 strict.",
        ),
        candidate(
            "FS005_RX8_FLOOR_2028",
            5,
            3,
            2028,
            195,
            60,
            False,
            False,
            0.08,
            45,
            LIGHT_SAT,
            0.42,
            0.14,
            0.020,
            0.006,
            0.008,
            0.0008,
            0.0008,
            0.0,
            0.0007,
            "hinge",
            6.5,
            0.12,
            0.30,
            0.38,
            0.0,
            0.0,
            0.0,
            0.0,
            "Explicitly test whether a small receiver-floor push can fix the rx8 bottleneck without crushing K5 validation.",
            "worst_rx >= 60, strict >= 73, val >= 89.",
        ),
        candidate(
            "FS010_IDFIRST_LATE_2028",
            10,
            4,
            2028,
            200,
            60,
            False,
            False,
            0.10,
            70,
            LIGHT_SAT,
            0.42,
            0.12,
            0.020,
            0.006,
            0.006,
            0.0008,
            0.0008,
            0.0,
            0.00045,
            "hinge",
            8.5,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "Replicate the K10 strict leader family on seed2028 to test stability.",
            "strict >= 76.8, val >= 93.0.",
        ),
        candidate(
            "FS010_IDFIRST_LATE_2029",
            10,
            5,
            2029,
            200,
            60,
            False,
            False,
            0.10,
            70,
            LIGHT_SAT,
            0.42,
            0.12,
            0.020,
            0.006,
            0.006,
            0.0008,
            0.0008,
            0.0,
            0.00045,
            "hinge",
            8.5,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "Seed2029 stability replicate for the K10 identity-first family.",
            "strict std within the K10 family <= 0.6, val >= 92.8.",
        ),
        candidate(
            "FS010_RXGUARD_SATFLOOR_2028",
            10,
            6,
            2028,
            200,
            60,
            False,
            False,
            0.14,
            1,
            LIGHT_SAT,
            0.50,
            0.16,
            0.025,
            0.006,
            0.010,
            0.0010,
            0.0010,
            0.0,
            0.0005,
            "hinge",
            8.5,
            0.14,
            0.32,
            0.40,
            0.0,
            0.0,
            0.0,
            0.0,
            "K10 RXGUARD gave the best satellite mean but weak worst-rx; keep it as a sat-floor/receiver-floor trade-off probe.",
            "strict >= 76.4, sat_mean >= 36.5, worst_rx >= 62.",
        ),
        candidate(
            "FS010_BAL_VAL_2030",
            10,
            7,
            2030,
            200,
            65,
            False,
            True,
            0.12,
            45,
            LIGHT_SAT,
            0.40,
            0.16,
            0.018,
            0.007,
            0.010,
            0.0012,
            0.0012,
            0.0,
            0.00010,
            "l2",
            0.0,
            0.16,
            0.34,
            0.42,
            0.020,
            0.20,
            0.0,
            0.0,
            "Keep the K10 high-validation LACSR shape, but reduce GroupCE/proto pressure to close the val-strict gap.",
            "val >= 94.0, strict >= 76.6.",
        ),
        candidate(
            "FS020_RIEIFD_LIGHT_2028",
            20,
            0,
            2028,
            205,
            70,
            False,
            False,
            0.12,
            1,
            LIGHT_SAT,
            0.42,
            0.15,
            0.020,
            0.008,
            0.010,
            0.0015,
            0.0015,
            0.0,
            0.00004,
            "l2",
            0.0,
            0.16,
            0.34,
            0.42,
            0.0,
            0.0,
            0.0,
            0.0,
            "K20 use_aug/mixstyle may be fitting source nuisance; test a clean RIEI-FD-like light branch.",
            "strict >= 77.0, val >= 93.5, worst_rx >= 65.",
        ),
        candidate(
            "FS020_RIEIFD_FDGATE_B_2029",
            20,
            1,
            2029,
            205,
            70,
            True,
            True,
            0.14,
            1,
            LIGHT_SAT,
            0.42,
            0.16,
            0.020,
            0.008,
            0.010,
            0.0015,
            0.0015,
            0.0,
            0.00005,
            "l2",
            0.0,
            0.16,
            0.34,
            0.42,
            0.020,
            0.20,
            0.10,
            0.28,
            "Keep the best K20 RIEIFD_FDGATE seed but reduce domain and GroupCE load.",
            "strict >= 77.2, val >= 94.0.",
        ),
        candidate(
            "FS020_RXFLOOR_LIGHT_1337",
            20,
            2,
            1337,
            205,
            70,
            True,
            True,
            0.16,
            45,
            LIGHT_SAT,
            0.46,
            0.18,
            0.022,
            0.010,
            0.014,
            0.0020,
            0.0020,
            0.0001,
            0.00005,
            "l2",
            0.0,
            0.14,
            0.32,
            0.40,
            0.025,
            0.24,
            0.10,
            0.30,
            "K20 late repair had acceptable floor but too much domain load; lower it and keep a delayed sat view.",
            "strict >= 77.0, worst_rx >= 66, val >= 94.",
        ),
        candidate(
            "FS030_RXFLOOR_CAP_2028",
            30,
            3,
            2028,
            210,
            75,
            True,
            True,
            0.18,
            25,
            LIGHT_SAT,
            0.50,
            0.20,
            0.024,
            0.012,
            0.018,
            0.0025,
            0.0025,
            0.0002,
            0.00004,
            "l2",
            0.0,
            0.14,
            0.32,
            0.40,
            0.025,
            0.24,
            0.10,
            0.32,
            "Combine the K30 RXFLOOR finding with the K50 CAP_RELAX pattern.",
            "strict >= 79, val >= 96.5, worst_rx >= 70.",
        ),
        candidate(
            "FS030_RXFLOOR_CAP_2029",
            30,
            4,
            2029,
            210,
            75,
            True,
            True,
            0.18,
            25,
            LIGHT_SAT,
            0.50,
            0.20,
            0.024,
            0.012,
            0.018,
            0.0025,
            0.0025,
            0.0002,
            0.00004,
            "l2",
            0.0,
            0.14,
            0.32,
            0.40,
            0.025,
            0.24,
            0.10,
            0.32,
            "Seed2029 replicate for the K30 RXFLOOR+CAP hypothesis.",
            "strict >= 79, val >= 96.5, seed gap <= 1.5 strict.",
        ),
        candidate(
            "FS030_IDCAP_RELAX_2030",
            30,
            5,
            2030,
            210,
            75,
            True,
            True,
            0.16,
            45,
            LIGHT_SAT,
            0.44,
            0.18,
            0.020,
            0.010,
            0.016,
            0.0025,
            0.0025,
            0.0002,
            0.00004,
            "l2",
            0.0,
            0.16,
            0.34,
            0.42,
            0.020,
            0.22,
            0.10,
            0.30,
            "K30 validation is high but strict is low; reduce domain pressure to preserve identity scale.",
            "strict >= 78.5, val >= 96.5.",
        ),
        candidate(
            "FS050_CAP_RELAX_2028",
            50,
            6,
            2028,
            215,
            80,
            True,
            True,
            0.22,
            25,
            ALL_SAT,
            0.52,
            0.24,
            0.030,
            0.014,
            0.026,
            0.0040,
            0.0040,
            0.0003,
            0.00003,
            "l2",
            0.0,
            0.20,
            0.38,
            0.46,
            0.030,
            0.28,
            0.10,
            0.35,
            "Replicate the best K50 CAP_RELAX with a different seed to test whether it is stable.",
            "strict >= 82, val >= 97.5, worst_rx >= 73.",
        ),
        candidate(
            "FS050_CAP_RELAX_PLUS_2029",
            50,
            7,
            2029,
            215,
            80,
            True,
            True,
            0.24,
            25,
            ALL_SAT,
            0.50,
            0.22,
            0.028,
            0.014,
            0.024,
            0.0035,
            0.0035,
            0.0002,
            0.000025,
            "l2",
            0.0,
            0.20,
            0.38,
            0.46,
            0.030,
            0.28,
            0.10,
            0.35,
            "Push K50 toward the old LACSR 84.11 strict by slightly relaxing nuisance losses while keeping all-scenario full-DG satellite.",
            "strict >= 83, val >= 97.7, sat_mean >= 39.",
        ),
    ]


def render_report(run_id: str, candidates: Sequence[Candidate], script_path: Path, matrix_path: Path) -> str:
    rows = [
        "| ID | shots | GPU | seed | sat p/start/scenarios | key change | target |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for c in candidates:
        rows.append(
            f"| `{c.cid}` | {c.shots} | {c.gpu} | {c.seed} | "
            f"{c.sat_prob}/{c.sat_start}/{c.sat_scenarios} | {c.rationale} | {c.success_gate} |"
        )

    return f"""# {run_id}

## Objective

Optimize per-combo CEN51 few-shot performance for K=5/10/20/30/50 while
keeping the default satellite path as full-DG concat.  This matrix is not meant
for K>=100; above that boundary the algorithm should fall back to the original
CEN51 high-shot setting.

## Evidence From Completed V1

- All 16 V1 runs completed and all used `--use_concat_sat_channel_aug` with
  `ce_only=0`; satellite samples entered the full TX/domain/GroupCE/proto/SupCon
  loss path.
- K10 was the only clearly stable low-shot regime: strict mean 76.49 with std
  0.38 and val mean 93.26.
- K5 still split into strict and validation leaders: best strict was 74.36 at
  val 89.09, while best validation reached about 90 but strict dropped.
- K20 and K30 were non-monotonic; full-DG pressure still suppressed target
  generalization relative to earlier LACSR anchors.
- K50 improved under CAP_RELAX to 82.15 strict / 97.46 val, but still did not
  recover the old LACSR 84.11 strict target.

## V2 Optimization Rules

1. K5: lower the hinge norm target from 8 to 6/6.5 and reduce early satellite
   probability.  The V1 hinge target was too loose because final z_id norms were
   around 2.3-2.8.
2. K10: treat identity-first late satellite as the current default candidate
   and verify it across seeds; keep one RXGUARD branch only for satellite floor.
3. K20/K30: reduce domain/GroupCE/Fishr pressure and test receiver-floor repair
   without making source validation the only selection signal.
4. K50: replicate CAP_RELAX and run a slightly more relaxed plus branch; this is
   the path with the best chance of restoring the high-shot advantage.

## Local Artifacts

- Generator: `tools/cen51_fulldg_fewshot_v2_matrix.py`
- Launcher: `{script_path.as_posix()}`
- Matrix JSON: `{matrix_path.as_posix()}`
- Parsed V1 summary: `analysis_tmp/cen51_fulldg_shotaware_lacsr_fd_20260610_161500/parsed/summary_runs.csv`

## Candidate Matrix

{chr(10).join(rows)}

## Success Criteria

- K5: reach a joint envelope of strict >= 74.6 and val >= 90 on at least one
  seed, without a seed gap above 2.5 strict points.
- K10: preserve the stable 76.5-77 strict band and val >= 93.
- K20: repair the V1 dip and reach strict >= 77.2 with val >= 94.
- K30: move toward strict >= 79 while keeping val >= 96.5.
- K50: reach strict >= 83, and ideally approach the old LACSR 84.11 target.

## Verification Before Launch

- `conda activate ssr-gpu; python -m py_compile tools/cen51_fulldg_fewshot_v2_matrix.py tools/cen51_fulldg_shotaware_lacsr_fd_matrix.py code/train.py`
- `conda activate ssr-gpu; python tools/cen51_fulldg_fewshot_v2_matrix.py --run-id {run_id}`
- `bash -n code/scripts/launch_{run_id}.sh`
- Remote dry-run must show 16 candidates, 16 `--use_concat_sat_channel_aug`,
  and 0 `concat_sat_ce_only`.

## Launch Policy

Run only after a fresh N607 preflight/capacity audit.  Default packing is two
jobs per GPU.  Do not overwrite V1 logs or checkpoints.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=Path("automation_reports") / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=Path("code") / "scripts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_fulldg_fewshot_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
