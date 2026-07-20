#!/usr/bin/env python3
"""Create the complete D91 crossfit-consensus performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D87_SUMMARY_PATH = SCRIPT_DIR / "summarize_d87_performance.py"
SPEC = importlib.util.spec_from_file_location("d91_d87_summary", D87_SUMMARY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D87 summary helper")
d87 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d87)
base, quant = d87.base, d87.quant
TARGET, FP32 = d87.TARGET, d87.FP32


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nonfinite_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_nonfinite_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_nonfinite_count(item) for item in value)
    if isinstance(value, float):
        return int(not (-float("inf") < value < float("inf")))
    return 0


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = d87.mechanism(rows)
    audits = [row["geometry_summary"]["d79_worstclass_margin_audit"] for row in rows]
    result["consensus"] = {
        "factor": base.stats(audit["consensus_factor"] for audit in audits),
        "zero_factor_count": sum(audit["consensus_factor"] == 0.0 for audit in audits),
        "fold_gradient_norm_min": base.stats(
            audit["fold_gradient_norm_min"] for audit in audits
        ),
        "fold_gradient_norm_mean": base.stats(
            audit["fold_gradient_norm_mean"] for audit in audits
        ),
        "fold_gradient_norm_max": base.stats(
            audit["fold_gradient_norm_max"] for audit in audits
        ),
        "fold_gradient_cosine_min": base.stats(
            audit["fold_gradient_cosine_min"] for audit in audits
        ),
        "fold_gradient_cosine_mean": base.stats(
            audit["fold_gradient_cosine_mean"] for audit in audits
        ),
        "fold_gradient_cosine_max": base.stats(
            audit["fold_gradient_cosine_max"] for audit in audits
        ),
        "mean_unit_gradient_norm": base.stats(
            audit["mean_unit_gradient_norm"] for audit in audits
        ),
        "d87_unshrunk_residual_frobenius": base.stats(
            audit["d87_unshrunk_residual_frobenius"] for audit in audits
        ),
        "actual_to_unshrunk_residual_ratio": base.stats(
            audit["residual_frobenius"]
            / max(audit["d87_unshrunk_residual_frobenius"], 1e-30)
            for audit in audits
        ),
        "threshold_count": 0,
        "extra_hyperparameter_count": 0,
        "query_rows_used": 0,
    }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = d87.resources(rows)
    result["d91_extra_crossfit_lda_fit_count"] = base.stats(
        row["resource"]["descendant_extra_crossfit_lda_fit_count"] for row in rows
    )
    result["d91_extra_crossfit_lda_fit_macs"] = base.stats(
        row["resource"]["descendant_extra_crossfit_lda_fit_macs"] for row in rows
    )
    result["d91_actual_crossfit_lda_fit_count"] = base.stats(
        row["resource"]["descendant_actual_crossfit_lda_fit_count"] for row in rows
    )
    result["d91_actual_crossfit_lda_fit_macs"] = base.stats(
        row["resource"]["descendant_actual_crossfit_lda_fit_macs"] for row in rows
    )
    result["d91_fold_consensus_mac_upper_bound"] = base.stats(
        row["resource"]["descendant_extra_support_mac_upper_bound"] for row in rows
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d91", "d89", "d87", "d85", "d62")
    for name in names:
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    for name in (
        "stdout", "stderr", "metadata", "receipt", "support", "resource",
        "geometry", "selection",
    ):
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
    fp32 = [row for row in logs["D91"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 matched target rows")

    artifact_names = ("metadata", "receipt", "support", "resource", "geometry", "selection")
    artifact_paths = {name: getattr(args, name) for name in artifact_names}
    artifacts = {name: _read(path) for name, path in artifact_paths.items()}
    metadata, receipt = artifacts["metadata"], artifacts["receipt"]
    if (
        metadata.get("schema") != "cvs.phase2.d91.crossfit_consensus_sigma_probe.v1"
        or receipt.get("training_log_row_count") != 105
        or receipt.get("query_opened") is not False
        or receipt.get("source_closure_unchanged_after_support") is not True
        or metadata.get("verified_d91_target_row_count") != 30
    ):
        raise ValueError("D91 evidence closure drift")
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d91.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path), "size": path.stat().st_size,
                "sha256": base.sha256(path), "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "complete_artifact_audit": {
            name: {
                "path": str(path), "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "nonfinite_numeric_value_count": _nonfinite_count(artifacts[name]),
            }
            for name, path in artifact_paths.items()
        },
        "stdout_stderr": {
            "stdout_size": args.stdout.stat().st_size,
            "stdout_sha256": base.sha256(args.stdout),
            "stderr_size": args.stderr.stat().st_size,
            "stderr_sha256": base.sha256(args.stderr),
            "error_marker_counts": {
                marker: stdout_text.count(marker) + stderr_text.count(marker)
                for marker in ("Traceback", "RuntimeError", "OOM", "NaN", "Inf")
            },
        },
        "all_candidates": base.candidate_summary(logs["D91"]),
        "D91_INT8": {
            "aggregate": base.aggregate(target["D91"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D91"]).items()
            },
            "classes": base.class_summary(target["D91"]),
            "outer_rows": base.detailed_rows(target["D91"]),
            "mechanism": mechanism(target["D91"]),
            "training": d87.training(target["D91"]),
            "quantization": quant(target["D91"]),
            "resources": resources(target["D91"]),
        },
        "D91_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D89", "D87", "D85", "D62"):
        comparison = base.matched_delta(target["D91"], target[baseline])
        result[f"D91_vs_{baseline}"] = {
            "D91": comparison.pop("D49"),
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
        "D91_INT8": result["D91_INT8"]["aggregate"],
        "D91_FP32": result["D91_FP32_MATCHED"]["aggregate"],
        "D91_vs_D89": result["D91_vs_D89"],
        "D91_vs_D87": result["D91_vs_D87"],
        "D91_vs_D62": result["D91_vs_D62"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
