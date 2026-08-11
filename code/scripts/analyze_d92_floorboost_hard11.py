#!/usr/bin/env python3
"""Analyze one immutable D92 floor-boost Hard11 artifact set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keep direct ``python code/scripts/analyze_d92_floorboost_hard11.py``
# invocation identical to the established runner entry point.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_floorboost_hard11_analysis import (
    analyze_d92_floorboost_hard11,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix-manifest", required=True)
    result.add_argument("--run-root", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument(
        "--baseline-paired-rows",
        "--baseline-row-metrics",
        dest="baseline_paired_rows",
        default=(
            "E:/type10-7/local_artifacts/"
            "d92_e0_full_only_target125_20260812_v1/analysis/paired_rows.csv"
        ),
    )
    result.add_argument("--output-root", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    gates = result["gates"]
    lines = [
        "# D92 E0_FULL_MAXMIN_FLOORBOOST Hard11分析",
        "",
        f"- 状态：`{result['status']}`",
        f"- 判定：`{result['verdict']}`",
        "- 范围：`DEVELOPMENT_ONLY_FLOOR_HARD_SCREEN`；K1仅作liveness，不进入性能均值。",
        "",
        "## 性能摘要",
        "",
        "|指标|候选均值|相对D92|相对E0_FULL_ONLY|",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "h_old_new": "H_old_new",
        "old_acc": "旧类平衡准确率",
        "old_floor": "旧类最低准确率",
        "seen_new_acc": "已见新类准确率",
        "forgetting": "平均遗忘",
    }
    for metric, label in labels.items():
        lines.append(
            f"|{label}|{100*aggregate[f'candidate_mean_{metric}']:.4f}%|"
            f"{100*aggregate[f'mean_delta_{metric}_vs_d92']:.4f}%|"
            f"{100*aggregate[f'mean_delta_{metric}_vs_full_only']:.4f}%|"
        )
    lines.extend(["", "## 冻结门", "", "|门|结果|观测|阈值|", "|---|---|---:|---:|"])
    for name, gate in gates.items():
        observed = gate.get("observed")
        if isinstance(observed, float):
            observed = f"{observed:.8f}"
        lines.append(f"|`{name}`|{'PASS' if gate.get('passed') else 'FAIL'}|{observed}|{gate.get('threshold')}|")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 历史对照固定读取`d92_e0_full_only_target125_20260812_v1/analysis/paired_rows.csv`，不重跑历史。",
            "- 本Hard11仅为development floor hard screen；`ADVANCE_TO_FULL125`不是正式Target125性能结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parser().parse_args()
    manifest = Path(args.matrix_manifest).resolve(strict=True)
    if args.dry_run:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "matrix_manifest": str(manifest),
                    "job_count": payload.get("job_count"),
                    "scene_arm_count": payload.get("scene_arm_count"),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    result = analyze_d92_floorboost_hard11(
        manifest,
        run_root=args.run_root,
        method_lock_path=args.method_lock,
        baseline_paired_rows_path=args.baseline_paired_rows,
    )
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"immutable analysis output already exists: {output_root}")
    output_root.mkdir(parents=True)
    summary = {key: value for key, value in result.items() if key not in {"paired_rows", "scenario_rows", "liveness_rows"}}
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "gates.json", result["gates"])
    _write_csv(output_root / "paired_rows.csv", result["paired_rows"])
    _write_csv(output_root / "scenario_rows.csv", result["scenario_rows"])
    _write_csv(output_root / "liveness_rows.csv", result["liveness_rows"])
    _write_csv(output_root / "resource_by_slice.csv", result["resource_by_slice"])
    for key, filename in (
        ("by_receiver", "receiver_metrics.csv"),
        ("by_seed", "seed_metrics.csv"),
        ("by_slice", "slice_metrics.csv"),
    ):
        _write_csv(output_root / filename, result[key])
    (output_root / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"], "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
