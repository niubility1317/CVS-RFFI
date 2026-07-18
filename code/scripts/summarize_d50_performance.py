#!/usr/bin/env python3
"""Create the full D50 performance ledger from complete D50/D45/D46 logs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "summarize_d49_performance.py"
SPEC = importlib.util.spec_from_file_location("d50_summary_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load the full performance summary helper")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

TARGET = "D42-USLDA-INT8"
MATCHED_FP32 = "D42-USLDA-FP32-MATCHED"


def mechanism_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "full_weight": base.stats(
                value for audit in audits for value in audit["d50_full_weight_by_class"]
            ),
            "block_weight": base.stats(
                value for audit in audits for value in audit["d50_block_weight_by_class"]
            ),
            "d45_anchor_z0": base.stats(audit["d50_d45_anchor_z0"] for audit in audits),
            "median_center": base.stats(audit["d50_median_center"] for audit in audits),
            "median_mean_abs_difference": base.stats(
                audit["d50_median_mean_abs_difference"] for audit in audits
            ),
            "centered_delta": base.stats(
                value
                for audit in audits
                for value in audit["d50_centered_median_delta_by_class"]
            ),
            "centered_delta_abs": base.stats(
                abs(value)
                for audit in audits
                for value in audit["d50_centered_median_delta_by_class"]
            ),
            "post_log_odds": base.stats(
                value
                for audit in audits
                for value in audit["d50_post_log_odds_by_class"]
            ),
            "anchor_error_max": max(
                audit["d50_post_log_odds_mean_anchor_error"] for audit in audits
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
                "full_weight_mean": base.mean(
                    value
                    for audit in audits
                    for value in audit["d50_full_weight_by_class"]
                ),
                "median_mean_abs_difference_mean": base.mean(
                    audit["d50_median_mean_abs_difference"] for audit in audits
                ),
                "centered_delta_abs_mean": base.mean(
                    abs(value)
                    for audit in audits
                    for value in audit["d50_centered_median_delta_by_class"]
                ),
            }
    return result


def quantization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sum_keys = (
        "matched_fp32_before_argmax_change_count",
        "matched_fp32_outer_argmax_change_count",
        "int8_fp32_margin_sign_flip_count",
        "int8_vs_fp32_before_support_argmax_change_count",
        "int8_vs_fp32_final_support_argmax_change_count",
    )
    result = {
        key: sum(int(row["resource"][key]) for row in rows) for key in sum_keys
    }
    result["int8_fp32_max_score_abs_error"] = base.stats(
        row["int8_fp32_max_score_abs_error"] for row in rows
    )
    for key in ("old_new_margin_min", "new_old_margin_min", "new_new_margin_min"):
        result[key] = base.stats(row[key] for row in rows)
    return result


def resource_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d46_estimated_reliability_scoring_macs",
        "d46_estimated_classwise_affine_fusion_macs",
        "d47_additional_adaptation_mac_equivalents",
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
            "d47_resource_reuses_d46_exact_inventory",
        )
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d50-log", required=True, type=Path)
    parser.add_argument("--d45-log", required=True, type=Path)
    parser.add_argument("--d46-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logs = {
        "D50": base.load_jsonl(args.d50_log),
        "D45": base.load_jsonl(args.d45_log),
        "D46": base.load_jsonl(args.d46_log),
    }
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected complete 105-row D50/D45/D46 logs")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D50"] if row["candidate_id"] == MATCHED_FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target/matched rows per log")
    output = {
        "schema": "cvs.phase2.d50.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in (
                ("D50", args.d50_log),
                ("D45", args.d45_log),
                ("D46", args.d46_log),
            )
        },
        "all_candidates": base.candidate_summary(logs["D50"]),
        "D50_INT8": {
            "aggregate": base.aggregate(target["D50"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D50"]).items()
            },
            "classes": base.class_summary(target["D50"]),
            "outer_rows": base.detailed_rows(target["D50"]),
            "mechanism": mechanism_summary(target["D50"]),
            "training": base.training_summary(target["D50"]),
            "quantization": quantization_summary(target["D50"]),
            "resources": resource_summary(target["D50"]),
        },
        "D50_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
        "D50_vs_D45": base.matched_delta(target["D50"], target["D45"]),
        "D50_vs_D46": base.matched_delta(target["D50"], target["D46"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
