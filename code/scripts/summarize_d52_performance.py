#!/usr/bin/env python3
"""Create the full D52 performance ledger from complete matched logs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d50_performance.py"
SPEC = importlib.util.spec_from_file_location("d52_summary_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load summary helper")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
base = helper.base
TARGET = "D42-USLDA-INT8"
MATCHED_FP32 = "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "resultant": base.stats(
                value for audit in audits for value in audit["d52_resultant_norm_by_class"]
            ),
            "gamma": base.stats(
                value for audit in audits for value in audit["d52_gamma_by_class"]
            ),
            "raw_direction_l2": base.stats(
                sum(value * value for value in vector) ** 0.5
                for audit in audits
                for vector in audit["d52_raw_median_mean_direction_fp64"]
            ),
            "unit_direction_l2": base.stats(
                sum(value * value for value in vector) ** 0.5
                for audit in audits
                for vector in audit["d52_unit_median_mean_direction_fp64"]
            ),
            "base_discriminant_l2": base.stats(
                value
                for audit in audits
                for value in audit["d52_base_discriminant_norm_by_class"]
            ),
            "correction_bound_l2": base.stats(
                value
                for audit in audits
                for value in audit["d52_correction_bound_by_class"]
            ),
            "correction_l2": base.stats(
                sum(value * value for value in vector) ** 0.5
                for audit in audits
                for vector in audit["d52_coefficient_correction_fp64"]
            ),
            "row_norm_error": base.stats(
                audit["d52_support_row_norm_max_abs_error"] for audit in audits
            ),
            "correction_bound_max_abs_gap": max(
                abs(
                    sum(value * value for value in vector) ** 0.5 - bound
                )
                for audit in audits
                for vector, bound in zip(
                    audit["d52_coefficient_correction_fp64"],
                    audit["d52_correction_bound_by_class"],
                )
            ),
        }
    result["by_scene"] = {}
    for scene, group in base.scene_groups(rows).items():
        result["by_scene"][scene] = {}
        for phase in ("before", "final"):
            audits = [
                row["geometry_summary"][f"{phase}_covariance_audit"] for row in group
            ]
            result["by_scene"][scene][phase] = {
                "gamma_mean": base.mean(
                    value for audit in audits for value in audit["d52_gamma_by_class"]
                ),
                "correction_l2_mean": base.mean(
                    sum(value * value for value in vector) ** 0.5
                    for audit in audits
                    for vector in audit["d52_coefficient_correction_fp64"]
                ),
                "correction_l2_max": max(
                    sum(value * value for value in vector) ** 0.5
                    for audit in audits
                    for vector in audit["d52_coefficient_correction_fp64"]
                ),
            }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d52_extra_adaptation_mac_equivalents",
        "d52_coordinate_median_comparison_upper_bound",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
    )
    result = {key: base.stats(row["resource"][key] for row in rows) for key in keys}
    first = rows[0]["resource"]
    result["invariants"] = {
        key: first[key]
        for key in (
            "runtime_device",
            "deployment_precision",
            "coefficient_dtype",
            "intercept_dtype",
            "query_rows_used_for_fit",
            "query_features_used_for_fit",
            "query_labels_used_for_fit",
            "query_role_oracle_access",
            "query_class_quota_access",
            "query_true_batch_class_count_access",
            "query_batch_global_assignment",
            "query_dependent_batch_optimization",
            "source_sample_access",
            "clean_sample_access",
            "dense_query_graph_bytes",
            "d52_resource_reuses_d45_exact_inventory",
        )
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d52-log", type=Path, required=True)
    parser.add_argument("--d45-log", type=Path, required=True)
    parser.add_argument("--d46-log", type=Path, required=True)
    parser.add_argument("--d51-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "D52": args.d52_log,
        "D45": args.d45_log,
        "D46": args.d46_log,
        "D51": args.d51_log,
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected complete 105-row D52/D45/D46/D51 logs")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D52"] if row["candidate_id"] == MATCHED_FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target/matched rows")
    output = {
        "schema": "cvs.phase2.d52.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D52"]),
        "D52_INT8": {
            "aggregate": base.aggregate(target["D52"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D52"]).items()
            },
            "classes": base.class_summary(target["D52"]),
            "outer_rows": base.detailed_rows(target["D52"]),
            "mechanism": mechanism(target["D52"]),
            "training": base.training_summary(target["D52"]),
            "quantization": helper.quantization_summary(target["D52"]),
            "resources": resources(target["D52"]),
        },
        "D52_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
        "D52_vs_D45": base.matched_delta(target["D52"], target["D45"]),
        "D52_vs_D46": base.matched_delta(target["D52"], target["D46"]),
        "D52_vs_D51": base.matched_delta(target["D52"], target["D51"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
