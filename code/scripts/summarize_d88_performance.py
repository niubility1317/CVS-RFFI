#!/usr/bin/env python3
"""Create the complete D88 ground-sigma Pareto-guard performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d87_performance.py"
SPEC = importlib.util.spec_from_file_location("d88_d87_summary", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D87 summary helper")
d87 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d87)
base, quant = d87.base, d87.quant
TARGET, FP32 = d87.TARGET, d87.FP32


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = d87.mechanism(rows)
    audits = [row["geometry_summary"]["d79_worstclass_margin_audit"] for row in rows]
    traces = [item for audit in audits for item in audit["optimizer_objective_trace"]]
    result["pareto_guard"] = {
        "all_rows_class_clean_nonincrease": all(
            audit["all_class_clean_ce_nonincrease_verified"] for audit in audits
        ),
        "clean_ce_delta_min_class": base.stats(
            audit["oof_clean_ce_delta_min_class"] for audit in audits
        ),
        "clean_ce_delta_max_class": base.stats(
            audit["oof_clean_ce_delta_max_class"] for audit in audits
        ),
        "guard_tolerance": base.stats(
            audit["clean_pareto_guard_tolerance"] for audit in audits
        ),
        "halfspace_projection_count": base.stats(
            audit["total_halfspace_projection_count"] for audit in audits
        ),
        "zero_common_direction_steps": base.stats(
            audit["zero_common_direction_step_count"] for audit in audits
        ),
        "per_step_delta_vs_initial_max": base.stats(
            item["clean_ce_max_class_delta_vs_initial"] for item in traces
        ),
        "projected_direction_frobenius": base.stats(
            item["projected_direction_frobenius"] for item in traces
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d88", "d87", "d85", "d81", "d79")
    for name in names:
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {name.upper(): getattr(args, f"{name}_log") for name in names}
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D88"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d88.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "stdout_stderr": {
            "stdout_size": args.stdout.stat().st_size,
            "stdout_sha256": base.sha256(args.stdout),
            "stderr_size": args.stderr.stat().st_size,
            "stderr_sha256": base.sha256(args.stderr),
            "stderr_empty": stderr_text == "",
            "stdout_nonempty": stdout_text != "",
        },
        "metadata": {
            "path": str(args.metadata),
            "sha256": base.sha256(args.metadata),
            "schema": metadata["schema"],
            "verified_training_row_count": metadata["verified_d87_training_row_count"],
            "verified_d88_target_row_count": metadata["verified_d88_target_row_count"],
            "verified_d88_active_count": metadata["verified_d88_active_count"],
            "query_opened": metadata["query_opened"],
            "forced_nonpromotable": metadata["forced_nonpromotable"],
        },
        "all_candidates": base.candidate_summary(logs["D88"]),
        "D88_INT8": {
            "aggregate": base.aggregate(target["D88"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D88"]).items()
            },
            "classes": base.class_summary(target["D88"]),
            "outer_rows": base.detailed_rows(target["D88"]),
            "mechanism": mechanism(target["D88"]),
            "training": d87.training(target["D88"]),
            "quantization": quant(target["D88"]),
            "resources": d87.resources(target["D88"]),
        },
        "D88_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D87", "D85", "D81", "D79"):
        comparison = base.matched_delta(target["D88"], target[baseline])
        result[f"D88_vs_{baseline}"] = {
            "D88": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": base.sha256(args.output),
        "D88_INT8": result["D88_INT8"]["aggregate"],
        "D88_FP32": result["D88_FP32_MATCHED"]["aggregate"],
        "D88_vs_D87": result["D88_vs_D87"],
        "D88_vs_D85": result["D88_vs_D85"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
