#!/usr/bin/env python3
"""Create the complete D80 ground-covariance performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D78_SUMMARY = SCRIPT_DIR / "summarize_d78_performance.py"
SPEC = importlib.util.spec_from_file_location("d80_summary_helper", D78_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D78 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def mechanism(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    before = [row["geometry_summary"]["before_covariance_audit"] for row in rows]
    final = [row["geometry_summary"]["final_covariance_audit"] for row in rows]
    numeric = (
        "d80_target_degrees_of_freedom",
        "d80_ground_independent_domain_degrees_of_freedom",
        "d80_ground_shrinkage_weight",
        "d80_ground_z_trace_match_scale",
        "d80_target_z_covariance_trace",
        "d80_posterior_eigenvalue_min",
        "d80_posterior_eigenvalue_max",
        "d80_posterior_condition_number",
        "covariance_equation_residual_max",
        "within_class_residual_rank",
        "within_class_residual_energy",
    )
    ground = metadata["ground_audit"]
    result = {
        "fit_count": len(before) + len(final),
        "before": {key: base.stats(item[key] for item in before) for key in numeric},
        "final": {key: base.stats(item[key] for item in final) for key in numeric},
        "before_boundary_status": _counts(item["d62_boundary_status"] for item in before),
        "final_boundary_status": _counts(item["d62_boundary_status"] for item in final),
        "before_accepted_row_count": base.stats(
            sum(item["d62_final_accept_mask"]) for item in before
        ),
        "final_accepted_row_count": base.stats(
            sum(item["d62_final_accept_mask"]) for item in final
        ),
        "ground": {
            key: ground[key]
            for key in (
                "ground_registry_domain_count",
                "ground_domain_count",
                "ground_class_count",
                "ground_component_input_count",
                "ground_covariance_degrees_of_freedom",
                "ground_independent_domain_degrees_of_freedom",
                "ground_residual_numerical_rank",
                "ground_residual_effective_rank",
                "ground_raw_covariance_trace",
                "quantization_noise_floor",
                "covariance_eigenvalue_min",
                "covariance_eigenvalue_max",
                "covariance_condition_number",
                "covariance_sha256",
                "ground_bundle_contains_sample_radius",
                "ground_bundle_contains_sample_count",
            )
        },
        "ground_component_formal_phase2_eligible": ground[
            "component_formal_phase2_eligible"
        ],
        "ground_component_provenance_status": ground[
            "component_provenance_status"
        ],
        "ground_class_score_access": False,
        "full_and_block_component_prior_injected": True,
        "class_id_specific_formula": False,
        "old_new_role_specific_branch": False,
        "query_rows_used": 0,
        "optimizer_steps_extra": 0,
        "single_affine_state_only": True,
    }
    result["by_scene"] = {
        scene: {
            "before_ground_weight": base.stats(
                row["geometry_summary"]["before_covariance_audit"][
                    "d80_ground_shrinkage_weight"
                ]
                for row in group
            ),
            "final_ground_weight": base.stats(
                row["geometry_summary"]["final_covariance_audit"][
                    "d80_ground_shrinkage_weight"
                ]
                for row in group
            ),
            "before_condition": base.stats(
                row["geometry_summary"]["before_covariance_audit"][
                    "d80_posterior_condition_number"
                ]
                for row in group
            ),
            "final_condition": base.stats(
                row["geometry_summary"]["final_covariance_audit"][
                    "d80_posterior_condition_number"
                ]
                for row in group
            ),
        }
        for scene, group in base.scene_groups(rows).items()
    }
    return result


def training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [row["training_trace"] for row in rows]
    records = [item for trace in traces for item in trace]
    return {
        "complete_trace_length": base.stats(len(trace) for trace in traces),
        "all_trace_lengths_20": all(len(trace) == 20 for trace in traces),
        "stage2b_record_count": len(records),
        "stage2b_loss": base.stats(item["loss"] for item in records),
        "stage2b_ce_loss": base.stats(item["ce_loss"] for item in records),
        "stage2b_support_accuracy": base.stats(
            item["support_accuracy"] for item in records
        ),
        "d80_extra_optimizer_records": 0,
        "query_rows_used": 0,
    }


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d80_ground_covariance_statistics_macs",
        "d80_posterior_covariance_mac_upper_bound",
        "d80_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "peak_trainable_parameters",
        "persistent_state_bytes",
        "d80_compiled_affine_state_bytes",
        "d80_ground_component_logical_state_bytes",
        "d80_component_inclusive_persistent_state_bytes",
        "d80_ground_covariance_transient_fp64_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "total_optimizer_steps",
        "stage2c_optimizer_steps",
        "d80_optimizer_steps_extra",
        "d80_trainable_parameters_extra",
        "d80_query_extra_macs",
        "d80_query_extra_state_bytes",
        "d80_ground_component_input_count",
        "dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d80_single_affine_state_only",
        "d80_ground_component_update_access",
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
        "persistent_state_cap_pass",
        "optimizer_step_cap_pass",
        "trainable_parameter_cap_pass",
    )
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in invariants},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d80", "d79", "d78", "d77", "d66", "d62")
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
    fp32 = [row for row in logs["D80"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    markers = (
        "Traceback",
        "RuntimeError",
        "KeyError",
        "OOM",
        "OutOfMemory",
        "Killed",
        "NaN",
        "Inf",
    )
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d80.full_performance_summary.v1",
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
            "stdout_path": str(args.stdout),
            "stdout_size": args.stdout.stat().st_size,
            "stdout_sha256": base.sha256(args.stdout),
            "stdout_full_text_length": len(stdout_text),
            "stderr_path": str(args.stderr),
            "stderr_size": args.stderr.stat().st_size,
            "stderr_sha256": base.sha256(args.stderr),
            "stderr_full_text_length": len(stderr_text),
            "error_marker_counts": {
                marker: stdout_text.count(marker) + stderr_text.count(marker)
                for marker in markers
            },
        },
        "metadata": metadata,
        "all_candidates": base.candidate_summary(logs["D80"]),
        "D80_INT8": {
            "aggregate": base.aggregate(target["D80"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D80"]).items()
            },
            "classes": base.class_summary(target["D80"]),
            "outer_rows": base.detailed_rows(target["D80"]),
            "mechanism": mechanism(target["D80"], metadata),
            "training": training(target["D80"]),
            "quantization": quant(target["D80"]),
            "resources": resources(target["D80"]),
        },
        "D80_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D79", "D78", "D77", "D66", "D62"):
        comparison = base.matched_delta(target["D80"], target[baseline])
        result[f"D80_vs_{baseline}"] = {
            "D80": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": base.sha256(args.output),
                "D80_INT8": result["D80_INT8"]["aggregate"],
                "D80_FP32": result["D80_FP32_MATCHED"]["aggregate"],
                "D80_vs_D62": result["D80_vs_D62"],
                "D80_vs_D79": result["D80_vs_D79"],
                "quantization": result["D80_INT8"]["quantization"],
                "mechanism": result["D80_INT8"]["mechanism"],
                "resources": result["D80_INT8"]["resources"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
