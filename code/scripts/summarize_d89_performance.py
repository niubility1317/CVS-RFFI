#!/usr/bin/env python3
"""Create the complete D89 v2 radius-reliability performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D81_SUMMARY_PATH = SCRIPT_DIR / "summarize_d81_performance.py"
SPEC = importlib.util.spec_from_file_location("d89_d81_summary", D81_SUMMARY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D81 summary helper")
d81 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d81)
base, quant = d81.base, d81.quant
TARGET, FP32 = d81.TARGET, d81.FP32


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _nonfinite_count(value: Any) -> int:
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, dict):
        return sum(_nonfinite_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_nonfinite_count(item) for item in value)
    return 0


def mechanism(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    before = [row["geometry_summary"]["before_covariance_audit"] for row in rows]
    final = [row["geometry_summary"]["final_covariance_audit"] for row in rows]
    ground = metadata["ground_audit"]
    ground_keys = (
        "schema",
        "spectrum_policy",
        "ground_domain_count",
        "ground_class_count",
        "ground_component_input_count",
        "radius_definition",
        "radius_min",
        "radius_median",
        "radius_mean",
        "radius_max",
        "cross_domain_signal_min",
        "cross_domain_signal_mean",
        "cross_domain_signal_max",
        "reliability_formula",
        "reliability_min",
        "reliability_mean",
        "reliability_max",
        "class_reliability_sum_min",
        "class_reliability_sum_max",
        "cell_weight_min",
        "cell_weight_max",
        "domain_weight_sum_max_abs_error",
        "weighted_center_max_abs_error",
        "weighted_covariance_trace",
        "quantization_noise_floor_policy",
        "quantization_noise_floor_removed_before_spectrum",
        "positive_numerical_rank",
        "participation_ratio_effective_rank",
        "retained_rank",
        "retained_signal_fraction",
        "basis_sha256",
        "spectral_weight_sha256",
        "ground_component_state",
        "ground_int8_component_logical_state_bytes",
        "ground_covariance_statistics_mac_upper_bound",
        "ground_aggregated_center_access",
        "ground_aggregated_p90_radius_access",
        "ground_sample_radius_access",
        "ground_sample_feature_access",
        "ground_target_identity_mapping_access",
        "ground_class_score_access",
        "ground_component_update_access",
        "dense_ground_bank_persisted",
        "ground_class_centers_discarded",
        "radius_hyperparameter_count",
        "radius_scan_count",
        "rank_scan_count",
    )
    return {
        "before": d81._transform_stats(before),
        "final": d81._transform_stats(final),
        "before_d62_active_fit_count": sum(
            any(audit["d62_final_accept_mask"]) for audit in before
        ),
        "final_d62_active_fit_count": sum(
            any(audit["d62_final_accept_mask"]) for audit in final
        ),
        "before_d62_accepted_row_count": sum(
            sum(audit["d62_final_accept_mask"]) for audit in before
        ),
        "final_d62_accepted_row_count": sum(
            sum(audit["d62_final_accept_mask"]) for audit in final
        ),
        "ground": {key: ground[key] for key in ground_keys},
        "component_fit_execution_count": metadata["component_fit_execution_count"],
        "support_center_transform_execution_count": metadata[
            "support_center_transform_execution_count"
        ],
        "ground_component_bitwise_unchanged": metadata[
            "ground_component_bitwise_unchanged"
        ],
        "query_rows_used": 0,
        "optimizer_steps_extra": 0,
        "single_affine_state_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d89", "d81", "d85", "d62", "d88")
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
    fp32 = [row for row in logs["D89"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 matched target rows")

    artifact_paths = {
        name: getattr(args, name)
        for name in ("metadata", "receipt", "support", "resource", "geometry", "selection")
    }
    artifacts = {name: _read_object(path) for name, path in artifact_paths.items()}
    metadata, receipt = artifacts["metadata"], artifacts["receipt"]
    if (
        metadata.get("schema") != "cvs.phase2.d89.v2_radius_cauchy_center_probe.v1"
        or receipt.get("training_log_row_count") != 105
        or receipt.get("query_opened") is not False
        or receipt.get("source_closure_unchanged_after_support") is not True
    ):
        raise ValueError("D89 evidence closure drift")

    result: dict[str, Any] = {
        "schema": "cvs.phase2.d89.full_performance_summary.v1",
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
                "nonfinite_numeric_value_count": _nonfinite_count(artifacts[name]),
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
        "all_candidates": base.candidate_summary(logs["D89"]),
        "D89_INT8": {
            "aggregate": base.aggregate(target["D89"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D89"]).items()
            },
            "classes": base.class_summary(target["D89"]),
            "outer_rows": base.detailed_rows(target["D89"]),
            "mechanism": mechanism(target["D89"], metadata),
            "training": d81.training(target["D89"]),
            "quantization": quant(target["D89"]),
            "resources": d81.resources(target["D89"]),
        },
        "D89_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D81", "D85", "D62", "D88"):
        comparison = base.matched_delta(target["D89"], target[baseline])
        result[f"D89_vs_{baseline}"] = {
            "D89": comparison.pop("D49"),
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
        "D89_INT8": result["D89_INT8"]["aggregate"],
        "D89_FP32": result["D89_FP32_MATCHED"]["aggregate"],
        "D89_vs_D81": result["D89_vs_D81"],
        "D89_vs_D85": result["D89_vs_D85"],
        "D89_vs_D62": result["D89_vs_D62"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
