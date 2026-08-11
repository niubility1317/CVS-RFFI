#!/usr/bin/env python3
"""Analyze the immutable E0_FULL_ONLY complete Target125 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsrffi.stage2_d92_e0_full_only_target125_analysis import (
    analyze_d92_e0_full_only_target125,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--method-lock", required=True)
    parser.add_argument("--baseline-row-metrics", required=True)
    parser.add_argument("--baseline-scenario-metrics", required=True)
    parser.add_argument("--baseline-per-tx-metrics", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.4f}%"


def _markdown(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    gates = result["gates"]
    lines = [
        "# D92 E0_FULL_ONLY完整Target125分析",
        "",
        f"- 状态：`{result['status']}`",
        f"- 判定：`{result['verdict']}`",
        f"- 完整门：`{'PASS' if result['all_gates_pass'] else 'FAIL'}`",
        "- 比较：同一125outer、同一三场景，E0_FULL_ONLY对原始D92冻结基线。",
        "",
        "## 总体结果",
        "",
        "|指标|E0_FULL_ONLY|原始D92|差值|",
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
            f"|{label}|{_pct(aggregate[f'candidate_mean_{metric}'])}|"
            f"{_pct(aggregate[f'baseline_mean_{metric}'])}|"
            f"{_pct(aggregate[f'mean_delta_{metric}'])}|"
        )
    lines.extend([
        "",
        "## 冻结晋级门",
        "",
        "|门|结果|观测|阈值|",
        "|---|---|---:|---:|",
    ])
    for name, gate in gates.items():
        observed = gate["observed"]
        if isinstance(observed, float):
            observed = f"{observed:.8f}"
        lines.append(
            f"|`{name}`|{'PASS' if gate['passed'] else 'FAIL'}|{observed}|{gate['threshold']}|"
        )
    lines.extend([
        "",
        "## 资源",
        "",
        f"- 注册wall中位数：{aggregate['median_registration_wall_time_ns'] / 1e6:.4f}ms。",
        f"- 注册CPU中位数：{aggregate['median_registration_process_cpu_time_ns'] / 1e6:.4f}ms。",
        f"- 注册增量峰值内存中位数：{aggregate['median_registration_incremental_peak_working_set_bytes'] / 1024:.2f}KiB。",
        "- 查询MAC与永久状态逐slice见`resource_by_slice.csv`；不把历史缺失的配对wall资源伪造成125行加速比。",
        "",
        "## 证据边界",
        "",
        "- 场景表报告H、旧类准确率、已见新类准确率和遗忘；score未提供逐场景逐旧类表，因此不虚构逐场景old floor。",
        "- 所有性能值来自完整125×3场景、评分在不可变预测提交后完成。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    result = analyze_d92_e0_full_only_target125(
        args.matrix_manifest,
        run_root=args.run_root,
        method_lock_path=args.method_lock,
        baseline_row_metrics_path=args.baseline_row_metrics,
        baseline_scenario_metrics_path=args.baseline_scenario_metrics,
        baseline_per_tx_metrics_path=args.baseline_per_tx_metrics,
    )
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"immutable analysis output already exists: {output_root}")
    output_root.mkdir(parents=True)

    raw_names = ("paired_rows", "scenario_rows", "per_old_class_rows")
    summary = {key: value for key, value in result.items() if key not in raw_names}
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "gates.json", result["gates"])
    _write_csv(output_root / "paired_rows.csv", result["paired_rows"])
    _write_csv(output_root / "scenario_rows.csv", result["scenario_rows"])
    _write_csv(output_root / "per_old_class_rows.csv", result["per_old_class_rows"])
    for key, filename in (
        ("by_receiver", "receiver_metrics.csv"),
        ("by_seed", "seed_metrics.csv"),
        ("by_slice", "slice_metrics.csv"),
        ("by_scenario", "scenario_metrics.csv"),
        ("resource_by_slice", "resource_by_slice.csv"),
        ("per_old_class", "per_old_class.csv"),
    ):
        _write_csv(output_root / filename, result[key])
    (output_root / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"], "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
