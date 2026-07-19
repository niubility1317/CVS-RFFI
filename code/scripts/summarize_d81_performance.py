#!/usr/bin/env python3
"""Create the complete D81 ground-spectrum robust-center performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d78_performance.py"
SPEC = importlib.util.spec_from_file_location("d81_summary_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D78 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _transform_stats(audits: list[dict[str, Any]]) -> dict[str, Any]:
    transforms = [audit["d81_transform_audit"] for audit in audits]
    return {
        "fit_count": len(transforms),
        "center_shift_l2_max": base.stats(
            transform["center_shift_l2_max"] for transform in transforms
        ),
        "center_shift_l2_all_classes": base.stats(
            value
            for transform in transforms
            for value in transform["center_shift_l2_by_class"]
        ),
        "normalized_weight_min": base.stats(
            transform["normalized_weight_min"] for transform in transforms
        ),
        "normalized_weight_max": base.stats(
            transform["normalized_weight_max"] for transform in transforms
        ),
        "effective_sample_size_all_classes": base.stats(
            value
            for transform in transforms
            for value in transform["effective_sample_size_by_class"]
        ),
        "within_class_residual_max_abs_error": base.stats(
            transform["within_class_residual_max_abs_error"]
            for transform in transforms
        ),
        "fft96_rf32_max_abs_error": base.stats(
            transform["fft96_rf32_max_abs_error"] for transform in transforms
        ),
        "all_class_symmetric": all(
            transform["class_id_specific_formula"] is False
            and transform["old_new_role_specific_branch"] is False
            and transform["scene_receiver_handle_specific_branch"] is False
            for transform in transforms
        ),
        "all_query_free": all(
            transform["uses_outer_held_or_query"] is False
            and int(transform["query_rows_used"]) == 0
            for transform in transforms
        ),
    }


def mechanism(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    before = [row["geometry_summary"]["before_covariance_audit"] for row in rows]
    final = [row["geometry_summary"]["final_covariance_audit"] for row in rows]
    ground = metadata["ground_audit"]
    result = {
        "before": _transform_stats(before),
        "final": _transform_stats(final),
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
        "ground": {
            key: ground[key]
            for key in (
                "ground_registry_domain_count",
                "ground_domain_count",
                "ground_class_count",
                "ground_component_input_count",
                "ground_residual_numerical_rank",
                "ground_residual_effective_rank",
                "d81_positive_numerical_rank",
                "d81_participation_ratio_effective_rank",
                "d81_retained_rank",
                "d81_retained_signal_fraction",
                "d81_rank_policy",
                "d81_rank_scan_count",
                "d81_basis_sha256",
                "d81_spectral_weight_sha256",
                "ground_bundle_contains_sample_radius",
                "ground_bundle_contains_sample_count",
                "component_formal_phase2_eligible",
                "component_provenance_status",
            )
        },
        "component_fit_execution_count": metadata[
            "component_fit_execution_count"
        ],
        "support_center_transform_execution_count": metadata[
            "support_center_transform_execution_count"
        ],
        "ground_component_bitwise_unchanged": metadata[
            "ground_component_bitwise_unchanged"
        ],
        "verified_all_transform_center_shift_l2_min": metadata[
            "verified_d81_center_shift_l2_min"
        ],
        "verified_all_transform_center_shift_l2_max": metadata[
            "verified_d81_center_shift_l2_max"
        ],
        "verified_all_transform_normalized_weight_min": metadata[
            "verified_d81_normalized_weight_min"
        ],
        "verified_all_transform_effective_sample_size_min": metadata[
            "verified_d81_effective_sample_size_min"
        ],
        "query_rows_used": 0,
        "optimizer_steps_extra": 0,
        "single_affine_state_only": True,
    }
    result["by_scene"] = {
        scene: {
            "before": _transform_stats(
                [row["geometry_summary"]["before_covariance_audit"] for row in group]
            ),
            "final": _transform_stats(
                [row["geometry_summary"]["final_covariance_audit"] for row in group]
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
        "d81_extra_optimizer_records": 0,
        "query_rows_used": 0,
    }


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d81_ground_spectrum_statistics_macs",
        "d81_support_center_translation_mac_upper_bound",
        "d81_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "peak_trainable_parameters",
        "persistent_state_bytes",
        "d81_compiled_affine_state_bytes",
        "d81_ground_component_logical_state_bytes",
        "d81_component_inclusive_persistent_state_bytes",
        "d81_ground_basis_transient_fp64_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "total_optimizer_steps",
        "stage2c_optimizer_steps",
        "d81_optimizer_steps_extra",
        "d81_trainable_parameters_extra",
        "d81_query_extra_macs",
        "d81_query_extra_state_bytes",
        "d81_ground_component_input_count",
        "d81_ground_retained_rank",
        "dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d81_single_affine_state_only",
        "d81_ground_component_update_access",
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
    names = ("d81", "d80", "d79", "d62")
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
    fp32 = [row for row in logs["D81"] if row["candidate_id"] == FP32]
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
        "schema": "cvs.phase2.d81.full_performance_summary.v1",
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
        "all_candidates": base.candidate_summary(logs["D81"]),
        "D81_INT8": {
            "aggregate": base.aggregate(target["D81"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D81"]).items()
            },
            "classes": base.class_summary(target["D81"]),
            "outer_rows": base.detailed_rows(target["D81"]),
            "mechanism": mechanism(target["D81"], metadata),
            "training": training(target["D81"]),
            "quantization": quant(target["D81"]),
            "resources": resources(target["D81"]),
        },
        "D81_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D80", "D79", "D62"):
        comparison = base.matched_delta(target["D81"], target[baseline])
        result[f"D81_vs_{baseline}"] = {
            "D81": comparison.pop("D49"),
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
                "D81_INT8": result["D81_INT8"]["aggregate"],
                "D81_FP32": result["D81_FP32_MATCHED"]["aggregate"],
                "D81_vs_D62": result["D81_vs_D62"],
                "quantization": result["D81_INT8"]["quantization"],
                "mechanism": result["D81_INT8"]["mechanism"],
                "resources": result["D81_INT8"]["resources"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
