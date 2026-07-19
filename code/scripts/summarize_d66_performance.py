#!/usr/bin/env python3
"""Create the complete D66 ground-reliability performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d66_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = h.mechanism(rows)
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[f"d66_{phase}"] = {
            "fit_count": len(audits),
            "ground_component_input_count": base.stats(
                audit["d66_ground_component_input_count"] for audit in audits
            ),
            "ground_z_scale_sha256_unique_count": len(
                {audit["d66_ground_z_scale_sha256"] for audit in audits}
            ),
            "ground_z_scale_min": base.stats(
                audit["d66_ground_z_scale_min"] for audit in audits
            ),
            "ground_z_scale_max": base.stats(
                audit["d66_ground_z_scale_max"] for audit in audits
            ),
            "compilation_max_abs_error": base.stats(
                audit["d66_compilation_max_abs_error"] for audit in audits
            ),
            "compilation_tolerance": base.stats(
                audit["d66_compilation_tolerance"] for audit in audits
            ),
            "ground_component_used": all(
                audit["d66_ground_int8_component_used"] for audit in audits
            ),
            "ground_component_update_access": any(
                audit["d66_ground_component_update_access"] for audit in audits
            ),
            "shared_transform_all_registered_classes": all(
                audit["d66_shared_transform_all_registered_classes"]
                for audit in audits
            ),
            "role_class_scene_or_query_branch": any(
                audit["d66_old_new_role_specific_branch"]
                or audit["d66_class_id_specific_formula"]
                or audit["d66_scene_receiver_handle_specific_branch"]
                or audit["d66_uses_outer_held_or_query"]
                for audit in audits
            ),
            "single_affine_state_only": all(
                audit["d66_compiled_single_affine_state_only"] for audit in audits
            ),
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d62_additional_lda_fit_macs",
        "d62_fisher_dense_algebra_mac_equivalent_upper_bound",
        "d62_gate_scalar_mac_equivalents",
        "d66_ground_reliability_statistics_macs",
        "d66_shared_transform_application_macs",
        "d66_total_added_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "d66_compiled_affine_state_bytes",
        "d66_component_inclusive_persistent_state_bytes",
        "registry_state_bytes",
        "d66_ground_component_logical_state_bytes",
        "d66_transient_dequantized_ground_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "ground_int8_component_input_count",
        "ground_int8_update_access",
        "d66_ground_int8_component_used",
        "d66_ground_component_update_access",
        "d66_ground_z_scale_sha256",
        "d66_query_extra_macs",
        "d66_persistent_compiled_transform_bytes",
        "d66_single_affine_state_only",
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
        "persistent_state_cap_pass",
    )
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in invariants},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("d66", "d65", "d64", "d63", "d62", "d61", "d46"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name.upper(): getattr(args, f"{name}_log")
        for name in ("d66", "d65", "d64", "d63", "d62", "d61", "d46")
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D66"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d66.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D66"]),
        "D66_INT8": {
            "aggregate": base.aggregate(target["D66"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D66"]).items()
            },
            "classes": base.class_summary(target["D66"]),
            "outer_rows": base.detailed_rows(target["D66"]),
            "mechanism": mechanism(target["D66"]),
            "training": base.training_summary(target["D66"]),
            "quantization": quant(target["D66"]),
            "resources": resources(target["D66"]),
        },
        "D66_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D65", "D64", "D63", "D62", "D61", "D46"):
        result[f"D66_vs_{baseline}"] = base.matched_delta(
            target["D66"], target[baseline]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
