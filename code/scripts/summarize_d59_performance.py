#!/usr/bin/env python3
"""Create the complete D59 performance and mechanism ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d50_performance.py"
SPEC = importlib.util.spec_from_file_location("d59_summary_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D59 summary helper")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
base = helper.base
TARGET = "D42-USLDA-INT8"
MATCHED_FP32 = "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scalar_fields = (
        "d59_full_condition_number",
        "d59_block_condition_number",
        "d59_midpoint_condition_number",
        "d59_off_block_energy_fraction",
        "d59_affine_distance_block_to_full",
        "d59_affine_distance_block_to_midpoint",
        "d59_affine_distance_midpoint_to_full",
        "d59_riccati_relative_frobenius_residual",
        "d59_affine_half_distance_max_abs_error",
    )
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "fit_count": len(audits),
            "active_fit_count": sum(audit["d59_midpoint_active"] is True for audit in audits),
            **{
                field: base.stats(audit[field] for audit in audits)
                for field in scalar_fields
            },
            "unique_full_covariance_sha256": len(
                {audit["d59_full_covariance_sha256"] for audit in audits}
            ),
            "unique_block_covariance_sha256": len(
                {audit["d59_block_covariance_sha256"] for audit in audits}
            ),
            "unique_midpoint_covariance_sha256": len(
                {audit["d59_midpoint_covariance_sha256"] for audit in audits}
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
                "fit_count": len(audits),
                "off_block_energy_fraction_mean": base.mean(
                    audit["d59_off_block_energy_fraction"] for audit in audits
                ),
                "midpoint_condition_number_mean": base.mean(
                    audit["d59_midpoint_condition_number"] for audit in audits
                ),
                "affine_distance_block_to_full_mean": base.mean(
                    audit["d59_affine_distance_block_to_full"] for audit in audits
                ),
            }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d59_spd_midpoint_active_fit_count",
        "d59_spd_midpoint_dense_algebra_mac_equivalent_upper_bound",
        "d59_query_extra_macs",
        "d59_persistent_state_extra_bytes",
        "d59_optimizer_steps_extra",
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
    first = rows[0]["resource"]
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {
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
                "d59_resource_single_affine_state_only",
            )
        },
    }


def gates(
    d59_rows: list[dict[str, Any]], d46_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate = base.aggregate(d59_rows)
    reference = base.aggregate(d46_rows)
    maximize = (
        "before_old",
        "after_old",
        "seen_new",
        "H_old_new_same_row_mean",
        "joint_floor_same_row_mean",
        "min_before_class_mean",
        "min_after_class_mean",
        "min_new_class_mean",
    )
    aggregate_non_degradation = {
        field: candidate[field] >= reference[field] - 1.0e-12 for field in maximize
    }
    aggregate_non_degradation["forgetting_same_row_mean"] = (
        candidate["forgetting_same_row_mean"]
        <= reference["forgetting_same_row_mean"] + 1.0e-12
    )
    final_floor_strict = any(
        candidate[field] > reference[field] + 1.0e-12
        for field in (
            "joint_floor_same_row_mean",
            "min_after_class_mean",
            "min_new_class_mean",
        )
    )
    scene_non_degradation: dict[str, dict[str, bool]] = {}
    candidate_scenes = base.scene_groups(d59_rows)
    reference_scenes = base.scene_groups(d46_rows)
    for scene in candidate_scenes:
        new = base.aggregate(candidate_scenes[scene])
        old = base.aggregate(reference_scenes[scene])
        scene_non_degradation[scene] = {
            field: new[field] >= old[field] - 1.0e-12 for field in maximize
        }
        scene_non_degradation[scene]["forgetting_same_row_mean"] = (
            new["forgetting_same_row_mean"]
            <= old["forgetting_same_row_mean"] + 1.0e-12
        )
    confusion_non_degradation = {
        key: candidate["confusion"][key] <= reference["confusion"][key]
        for key in candidate["confusion"]
    }
    matched = base.matched_delta(d59_rows, d46_rows)
    quantization = helper.quantization_summary(d59_rows)
    quantization_zero = all(
        int(quantization[key]) == 0
        for key in (
            "matched_fp32_before_argmax_change_count",
            "matched_fp32_outer_argmax_change_count",
            "int8_fp32_margin_sign_flip_count",
        )
    )
    all_pass = (
        all(aggregate_non_degradation.values())
        and final_floor_strict
        and all(all(values.values()) for values in scene_non_degradation.values())
        and all(confusion_non_degradation.values())
        and matched["changed_prediction_hash_rows"] >= 1
        and quantization_zero
    )
    return {
        "aggregate_non_degradation": aggregate_non_degradation,
        "final_floor_strict_improvement": final_floor_strict,
        "scene_non_degradation": scene_non_degradation,
        "confusion_non_degradation": confusion_non_degradation,
        "changed_prediction_hash_rows_vs_d46": matched["changed_prediction_hash_rows"],
        "quantization_zero_change_and_flip": quantization_zero,
        "all_preregistered_performance_gates_pass": all_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    versions = ("d59", "d46", "d45", "d42", "d43full", "d43block", "d58")
    for version in versions:
        parser.add_argument(f"--{version}-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name.upper(): getattr(args, f"{name}_log") for name in versions}
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected complete 105-row matched logs")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D59"] if row["candidate_id"] == MATCHED_FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows per version")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d59.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D59"]),
        "D59_INT8": {
            "aggregate": base.aggregate(target["D59"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D59"]).items()
            },
            "classes": base.class_summary(target["D59"]),
            "outer_rows": base.detailed_rows(target["D59"]),
            "mechanism": mechanism(target["D59"]),
            "training": base.training_summary(target["D59"]),
            "quantization": helper.quantization_summary(target["D59"]),
            "resources": resources(target["D59"]),
        },
        "D59_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
        "preregistered_gates_vs_D46": gates(target["D59"], target["D46"]),
    }
    for baseline in versions[1:]:
        result[f"D59_vs_{baseline.upper()}"] = base.matched_delta(
            target["D59"], target[baseline.upper()]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
