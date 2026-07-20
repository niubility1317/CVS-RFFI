#!/usr/bin/env python3
"""Create the complete D90 directionwise Cauchy-center performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D89_SUMMARY_PATH = SCRIPT_DIR / "summarize_d89_performance.py"
SPEC = importlib.util.spec_from_file_location("d90_d89_summary", D89_SUMMARY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D89 summary helper")
d89 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d89)
d81, base, quant = d89.d81, d89.base, d89.quant
TARGET, FP32 = d89.TARGET, d89.FP32


def _axis_stats(audits: list[dict[str, Any]]) -> dict[str, Any]:
    transforms = [audit["d81_transform_audit"] for audit in audits]
    return {
        "axis_weight": base.stats(
            value
            for transform in transforms
            for class_values in transform["axis_cauchy_weight_by_class"]
            for row_values in class_values
            for value in row_values
        ),
        "axis_effective_sample_size_min_by_fit": base.stats(
            transform["axis_effective_sample_size_min"] for transform in transforms
        ),
        "axis_effective_sample_size_max_by_fit": base.stats(
            transform["axis_effective_sample_size_max"] for transform in transforms
        ),
        "radial_subspace_replacement_l2": base.stats(
            value
            for transform in transforms
            for value in transform["radial_subspace_replacement_l2_by_class"]
        ),
        "all_directionwise": all(
            transform["directionwise_subspace_center_replaced"] is True
            and transform["d81_orthogonal_center_preserved"] is True
            for transform in transforms
        ),
    }


def mechanism(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    result = d89.mechanism(rows, metadata)
    before = [row["geometry_summary"]["before_covariance_audit"] for row in rows]
    final = [row["geometry_summary"]["final_covariance_audit"] for row in rows]
    result["directionwise"] = {
        "before": _axis_stats(before),
        "final": _axis_stats(final),
        "d81_orthogonal_center_preserved": True,
        "directionwise_subspace_center_replaced": True,
        "extra_hyperparameter_count": 0,
        "extra_optimizer_steps": 0,
        "query_rows_used": 0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d90", "d89", "d81", "d85", "d62")
    for name in names:
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    for name in ("metadata", "receipt", "support", "resource", "geometry", "selection"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {name.upper(): getattr(args, f"{name}_log") for name in names}
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 complete rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D90"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 matched target rows")

    artifact_paths = {
        name: getattr(args, name)
        for name in ("metadata", "receipt", "support", "resource", "geometry", "selection")
    }
    artifacts = {name: d89._read_object(path) for name, path in artifact_paths.items()}
    metadata, receipt = artifacts["metadata"], artifacts["receipt"]
    if (
        metadata.get("schema")
        != "cvs.phase2.d90.v2_directionwise_cauchy_center_probe.v1"
        or metadata.get("directionwise_subspace_center") is not True
        or receipt.get("training_log_row_count") != 105
        or receipt.get("query_opened") is not False
        or receipt.get("source_closure_unchanged_after_support") is not True
    ):
        raise ValueError("D90 evidence closure drift")

    result: dict[str, Any] = {
        "schema": "cvs.phase2.d90.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "complete_artifact_audit": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "nonfinite_numeric_value_count": d89._nonfinite_count(artifacts[name]),
            }
            for name, path in artifact_paths.items()
        },
        "execution_stream_limit": {
            "stdout_persisted": False,
            "stderr_persisted": False,
            "reason": "foreground invocation returned through the Codex tool stream",
            "process_exit_code": 0,
            "receipt_elapsed_seconds": receipt["elapsed_seconds"],
        },
        "all_candidates": base.candidate_summary(logs["D90"]),
        "D90_INT8": {
            "aggregate": base.aggregate(target["D90"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D90"]).items()
            },
            "classes": base.class_summary(target["D90"]),
            "outer_rows": base.detailed_rows(target["D90"]),
            "mechanism": mechanism(target["D90"], metadata),
            "training": d81.training(target["D90"]),
            "quantization": quant(target["D90"]),
            "resources": d81.resources(target["D90"]),
        },
        "D90_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D89", "D81", "D85", "D62"):
        comparison = base.matched_delta(target["D90"], target[baseline])
        result[f"D90_vs_{baseline}"] = {
            "D90": comparison.pop("D49"),
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
        "D90_INT8": result["D90_INT8"]["aggregate"],
        "D90_FP32": result["D90_FP32_MATCHED"]["aggregate"],
        "D90_vs_D89": result["D90_vs_D89"],
        "D90_vs_D62": result["D90_vs_D62"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
