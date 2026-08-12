#!/usr/bin/env python3
"""Analyze one immutable D92 CSOAS Hard9+K1 artifact set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_csoas_hard10 import HISTORICAL_BASELINE_PATH, HISTORICAL_PER_OLD_CLASS_PATH  # noqa: E402
from cvsrffi.stage2_d92_csoas_hard10_analysis import analyze_d92_csoas_hard10  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix-manifest", required=True)
    result.add_argument("--run-root", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument("--baseline-paired-rows", default=HISTORICAL_BASELINE_PATH)
    result.add_argument("--per-old-class-rows", default=HISTORICAL_PER_OLD_CLASS_PATH)
    result.add_argument("--truth-sidecar-root")
    result.add_argument("--output-root", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("outer_key\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _markdown(result: Mapping[str, Any]) -> str:
    aggregate, gates = result["aggregate"], result["gates"]
    lines = ["# D92 E0_FULL_CSOAS Hard9+K1分析", "", f"- 状态：`{result['status']}`", f"- 判定：`{result['verdict']}`", "- 范围：`DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN`；K1仅作liveness，不进入性能均值。", "", "## 八项严格方向", "", "|指标|候选均值|相对E0_FULL_ONLY|", "|---|---:|---:|"]
    for metric in ("h_old_new", "old_balanced_accuracy", "c_old_acc", "old_floor", "seen_new_acc", "average_forgetting", "new_to_old_rate", "old_to_new_rate"):
        lines.append(f"|{metric}|{100*aggregate[f'candidate_mean_{metric}']:.4f}%|{100*aggregate[f'mean_delta_{metric}_vs_e0']:.4f}%|")
    lines.extend(["", "## 冻结门", "", "|门|结果|观测|", "|---|---|---|"])
    for name, gate in gates.items():
        lines.append(f"|`{name}`|{'PASS' if gate.get('passed') else 'FAIL'}|{gate.get('observed')}|")
    lines.extend(["", "历史paired_rows、E0 raw score与per_old_class_rows均按配置SHA验证；不重跑D92/E0。", ""])
    return "\n".join(lines)


def main() -> int:
    args = parser().parse_args()
    manifest = Path(args.matrix_manifest).resolve(strict=True)
    if args.dry_run:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        print(json.dumps({"status": "DRY_RUN", "matrix_manifest": str(manifest), "job_count": payload.get("job_count"), "scene_arm_count": payload.get("scene_arm_count")}, ensure_ascii=True, sort_keys=True)); return 0
    result = analyze_d92_csoas_hard10(manifest, run_root=args.run_root, method_lock_path=args.method_lock, baseline_paired_rows_path=args.baseline_paired_rows, per_old_class_rows_path=args.per_old_class_rows, truth_sidecar_root=args.truth_sidecar_root)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(f"immutable analysis output already exists: {output}")
    output.mkdir(parents=True)
    summary = {key: value for key, value in result.items() if key not in {"paired_rows", "per_old_class_rows", "scenario_rows", "liveness_rows"}}
    _write_json(output / "summary.json", summary); _write_json(output / "gates.json", result["gates"]); _write_csv(output / "paired_rows.csv", result["paired_rows"]); _write_csv(output / "per_old_class_rows.csv", result["per_old_class_rows"]); _write_csv(output / "scenario_rows.csv", result["scenario_rows"]); _write_csv(output / "liveness_rows.csv", result["liveness_rows"]); (output / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"], "output_root": str(output)})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
