#!/usr/bin/env python
"""Generate a comprehensive confirmation matrix for the best CEN51 few-shot recipe.

The matrix is shot-aware by design. Completed V1/V2 evidence did not support a
single fixed parameter set for every K. The selected recipe is:

- K5: V2 HINGE6_SATMIN, because it was the only K5 branch with val >= 90 and
  late-stability pass.
- K10: V1 IDFIRST_LATE, because it retained the strongest strict UDU in K10.
- K20: V2 RIEIFD_LIGHT, because it combined the best K20 val and strict.
- K30: V2 RXFLOOR_CAP, because it gave the strongest K30 strict with stable tail.
- K50: V1 CAP_RELAX, because V2 regressed and V1 remained the strongest K50
  strict anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cen51_fulldg_fewshot_v2_matrix as v2  # noqa: E402
import cen51_fulldg_shotaware_lacsr_fd_matrix as v1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    return parser.parse_args()


def by_cid(candidates: Sequence[v1.Candidate], cid: str) -> v1.Candidate:
    for candidate in candidates:
        if candidate.cid == cid:
            return candidate
    raise KeyError(cid)


def clone(base: v1.Candidate, cid: str, seed: int, gpu: int, rationale: str, gate: str) -> v1.Candidate:
    return replace(
        base,
        cid=cid,
        run_name=f"CEN51_FDCOMP_{cid}",
        seed=seed,
        gpu=gpu,
        rationale=rationale,
        success_gate=gate,
    )


def make_candidates() -> list[v1.Candidate]:
    v1_candidates = v1.make_candidates()
    v2_candidates = v2.make_candidates()

    k5 = by_cid(v2_candidates, "FS005_HINGE6_SATMIN_1337")
    k10 = by_cid(v1_candidates, "FS010_IDFIRST_LATE_2030")
    k20 = by_cid(v2_candidates, "FS020_RIEIFD_LIGHT_2028")
    k30 = by_cid(v2_candidates, "FS030_RXFLOOR_CAP_2029")
    k50 = by_cid(v1_candidates, "FS050_CAP_RELAX_1337")

    rows: list[tuple[v1.Candidate, str, int, int, str, str]] = [
        (k5, "FS005_BEST_HINGE6_SATMIN_1337", 1337, 0, "K5 V2 best stable branch; confirms val>=90 without late collapse.", "val>=90 strict>=74 latest_drop<=2.5"),
        (k5, "FS005_BEST_HINGE6_SATMIN_2028", 2028, 1, "K5 V2 family seed confirmation.", "val>=90 strict>=74 latest_drop<=2.5"),
        (k5, "FS005_BEST_HINGE6_SATMIN_2029", 2029, 2, "K5 V2 family hard-seed confirmation.", "val>=90 strict>=74 latest_drop<=2.5"),
        (k5, "FS005_BEST_HINGE6_SATMIN_2030", 2030, 3, "K5 V2 family additional stability seed.", "val>=90 strict>=74 latest_drop<=2.5"),
        (k10, "FS010_BEST_IDFIRST_LATE_2028", 2028, 4, "K10 V1 strict leader family, seed confirmation.", "strict>=76.5 val>=93 latest_drop<=2.0"),
        (k10, "FS010_BEST_IDFIRST_LATE_2029", 2029, 5, "K10 V1 strict leader family, seed confirmation.", "strict>=76.5 val>=93 latest_drop<=2.0"),
        (k10, "FS010_BEST_IDFIRST_LATE_2030", 2030, 6, "K10 V1 strict leader replay of the observed best seed.", "strict>=76.8 val>=93 latest_drop<=2.0"),
        (k20, "FS020_BEST_RIEIFD_LIGHT_1337", 1337, 7, "K20 V2 RIEI-FD-light family seed confirmation.", "strict>=77 val>=95 rx_floor>=60"),
        (k20, "FS020_BEST_RIEIFD_LIGHT_2028", 2028, 0, "K20 V2 observed best seed replay.", "strict>=77.5 val>=95 rx_floor>=60"),
        (k20, "FS020_BEST_RIEIFD_LIGHT_2029", 2029, 1, "K20 V2 RIEI-FD-light family seed confirmation.", "strict>=77 val>=95 rx_floor>=60"),
        (k30, "FS030_BEST_RXFLOOR_CAP_2028", 2028, 2, "K30 V2 RXFLOOR+CAP family confirmation.", "strict>=78 val>=96 rx_floor>=65"),
        (k30, "FS030_BEST_RXFLOOR_CAP_2029", 2029, 3, "K30 V2 observed best seed replay.", "strict>=78.8 val>=96 rx_floor>=65"),
        (k30, "FS030_BEST_RXFLOOR_CAP_2030", 2030, 4, "K30 V2 RXFLOOR+CAP family additional seed.", "strict>=78 val>=96 rx_floor>=65"),
        (k50, "FS050_BEST_CAP_RELAX_1337", 1337, 5, "K50 V1 strict anchor replay; V2 regressed.", "strict>=82 val>=97.4 rx_floor>=68"),
        (k50, "FS050_BEST_CAP_RELAX_2028", 2028, 6, "K50 V1 CAP_RELAX family seed confirmation.", "strict>=81 val>=97.4 rx_floor>=68"),
        (k50, "FS050_BEST_CAP_RELAX_2029", 2029, 7, "K50 V1 CAP_RELAX family seed confirmation.", "strict>=81 val>=97.4 rx_floor>=68"),
    ]
    return [clone(*row) for row in rows]


def render_report(run_id: str, candidates: Sequence[v1.Candidate], script_path: Path, matrix_path: Path) -> str:
    rows = [
        "| ID | shots | GPU | seed | source family | success target |",
        "|---|---:|---:|---:|---|---|",
    ]
    for c in candidates:
        family = c.cid.rsplit("_", 1)[0]
        rows.append(f"| `{c.cid}` | {c.shots} | {c.gpu} | {c.seed} | `{family}` | {c.success_gate} |")

    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## Objective",
            "",
            "Run a comprehensive multi-seed confirmation of the best observed CEN51 full-DG few-shot recipe.",
            "The recipe is shot-aware rather than one fixed loss schedule, because V1/V2 evidence showed different",
            "families dominate different per-combo sample regimes.",
            "",
            "## Evidence Summary",
            "",
            "- V2 completion: 16/16 finished, 12/16 late-stability pass, 0/16 promotion pass under strict gates.",
            "- K5: V2 HINGE6_SATMIN reached val 90.18 and strict 74.30 with stability pass.",
            "- K10: V1 IDFIRST_LATE remained the strict leader at 76.99; V2 improved stability but lowered strict.",
            "- K20: V2 RIEIFD_LIGHT reached val 95.94 and strict 77.50 with stability pass.",
            "- K30: V2 RXFLOOR_CAP seed2029 reached strict 78.80 with stable late behavior.",
            "- K50: V1 CAP_RELAX seed1337 stayed strongest at strict 82.15; V2 CAP_RELAX/PLUS regressed.",
            "",
            "## Candidate Matrix",
            "",
            *rows,
            "",
            "## Validation Contract",
            "",
            "After completion, parse full stdout logs with:",
            "",
            "```powershell",
            f"conda activate ssr-gpu",
            f"python tools\\cen51_fewshot_stability_validator.py --log-dir <local-log-dir> --matrix-json {matrix_path} --out-dir analysis_tmp\\{run_id}\\stability_validation --late-window 30 --no-fail",
            "```",
            "",
            "Promotion requires candidate-level stability plus family-level seed consistency. Compare against V1/V2",
            "anchors; do not select by source validation alone.",
            "",
            "## Local Artifacts",
            "",
            f"- Launcher: `{script_path}`",
            f"- Matrix JSON: `{matrix_path}`",
            "- Validator: `tools/cen51_fewshot_stability_validator.py`",
            "",
            "## Launch Policy",
            "",
            "Use a fresh N607 preflight and capacity audit. Default launch uses `MAX_TRAIN_PER_GPU=3` with the",
            "GPU assignment in the matrix; reroute only if live capacity requires it and record that change.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_fulldg_fewshot_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidates = make_candidates()
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"

    script_path.write_text(v1.render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
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
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
