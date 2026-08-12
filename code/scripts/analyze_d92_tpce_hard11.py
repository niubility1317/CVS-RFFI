#!/usr/bin/env python3
"""Analyze one immutable D92 TPCE Hard11 artifact set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_tpce_hard11_analysis import analyze_d92_tpce_hard11  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix-manifest", required=True)
    result.add_argument("--run-root", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument("--baseline-paired-rows", default="E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/analysis/paired_rows.csv")
    result.add_argument("--per-old-class-rows", default="E:/type10-7/local_artifacts/d92_e0_full_only_target125_20260812_v1/analysis/per_old_class_rows.csv")
    result.add_argument("--truth-sidecar-root")
    result.add_argument("--output-root", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = Path(args.matrix_manifest).resolve(strict=True)
    if args.dry_run:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        print(json.dumps({"status": "DRY_RUN", "matrix_manifest": str(manifest), "job_count": payload.get("job_count"), "scene_arm_count": payload.get("scene_arm_count")}, ensure_ascii=True, sort_keys=True))
        return 0
    result = analyze_d92_tpce_hard11(manifest, run_root=args.run_root, method_lock_path=args.method_lock, baseline_paired_rows_path=args.baseline_paired_rows, per_old_class_rows_path=args.per_old_class_rows, truth_sidecar_root=args.truth_sidecar_root)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(f"immutable analysis output already exists: {output}")
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({key: value for key, value in result.items() if key not in {"paired_rows", "per_old_class_rows", "scenario_rows", "liveness_rows"}}, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (output / "gates.json").write_text(json.dumps(result["gates"], indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"], "output_root": str(output)}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
