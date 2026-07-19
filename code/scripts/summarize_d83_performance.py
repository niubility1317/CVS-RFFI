#!/usr/bin/env python3
"""Create the complete D83 ground-precision-loading performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d78_performance.py"
SPEC = importlib.util.spec_from_file_location("d83_summary_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D78 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _stats(audits: list[dict[str, Any]], key: str) -> dict[str, float]:
    return base.stats(audit[key] for audit in audits)


def _mechanism(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    before = [row["geometry_summary"]["before_covariance_audit"] for row in rows]
    final = [row["geometry_summary"]["final_covariance_audit"] for row in rows]
    all_audits = before + final
    transforms = [audit["d83_transform_audit"] for audit in all_audits]
    ground = metadata["ground_audit"]
    return {
        "outer_fit_count": len(all_audits),
        "component_fit_execution_count": metadata["component_fit_execution_count"],
        "support_center_transform_execution_count": metadata[
            "support_center_transform_execution_count"
        ],
        "d62_active_fit_count": sum(
            any(audit["d62_final_accept_mask"]) for audit in all_audits
        ),
        "d62_accepted_row_count": sum(
            sum(audit["d62_final_accept_mask"]) for audit in all_audits
        ),
        "center_shift_l2": base.stats(
            value
            for transform in transforms
            for value in transform["center_shift_l2_by_class"]
        ),
        "normalized_weight_min": base.stats(
            transform["normalized_weight_min"] for transform in transforms
        ),
        "effective_sample_size": base.stats(
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
        "loading_scale": _stats(all_audits, "d83_loading_scale"),
        "loading_trace": _stats(all_audits, "d83_loading_trace"),
        "loading_mean_retained_direction": _stats(
            all_audits, "d83_loading_mean_retained_direction"
        ),
        "loading_to_target_mean_variance_ratio": _stats(
            all_audits, "d83_loading_to_target_mean_variance_ratio"
        ),
        "target_z_mean_variance": _stats(
            all_audits, "d83_target_z_mean_variance"
        ),
        "posterior_eigenvalue_min": _stats(
            all_audits, "d83_posterior_eigenvalue_min"
        ),
        "posterior_eigenvalue_max": _stats(
            all_audits, "d83_posterior_eigenvalue_max"
        ),
        "covariance_equation_residual_max": _stats(
            all_audits, "d83_covariance_equation_residual_max"
        ),
        "all_class_symmetric": all(
            not audit["d83_class_id_specific_formula"]
            and not audit["d83_old_new_role_specific_branch"]
            and not audit["d83_scene_receiver_handle_specific_branch"]
            for audit in all_audits
        ),
        "all_query_free": all(
            not audit["d83_uses_outer_held_or_query"]
            and int(audit["d83_query_rows_used"]) == 0
            for audit in all_audits
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
                "d83_positive_numerical_rank",
                "d83_participation_ratio_effective_rank",
                "d83_retained_rank",
                "d83_retained_signal_fraction",
                "d83_rank_policy",
                "d83_rank_scan_count",
                "d83_basis_sha256",
                "d83_spectral_weight_sha256",
                "ground_bundle_contains_sample_radius",
                "ground_bundle_contains_sample_count",
                "component_formal_phase2_eligible",
                "component_provenance_status",
            )
        },
        "ground_component_bitwise_unchanged": metadata[
            "ground_component_bitwise_unchanged"
        ],
    }


def _training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [row["training_trace"] for row in rows]
    records = [item for trace in traces for item in trace]
    return {
        "complete_trace_length": base.stats(len(trace) for trace in traces),
        "all_trace_lengths_20": all(len(trace) == 20 for trace in traces),
        "record_count": len(records),
        "loss": base.stats(item["loss"] for item in records),
        "ce_loss": base.stats(item["ce_loss"] for item in records),
        "support_accuracy": base.stats(
            item["support_accuracy"] for item in records
        ),
        "d83_extra_optimizer_records": 0,
        "query_rows_used": 0,
    }


def _resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d83_ground_spectrum_statistics_macs",
        "d83_support_center_translation_mac_upper_bound",
        "d83_covariance_loading_mac_upper_bound",
        "d83_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "peak_trainable_parameters",
        "persistent_state_bytes",
        "d83_compiled_affine_state_bytes",
        "d83_ground_component_logical_state_bytes",
        "d83_component_inclusive_persistent_state_bytes",
        "d83_ground_basis_transient_fp64_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "total_optimizer_steps",
        "stage2c_optimizer_steps",
        "d83_optimizer_steps_extra",
        "d83_trainable_parameters_extra",
        "d83_query_extra_macs",
        "d83_query_extra_state_bytes",
        "d83_ground_component_input_count",
        "d83_ground_retained_rank",
        "dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d83_single_affine_state_only",
        "d83_ground_component_update_access",
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
    for name in ("d83", "d81", "d62"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name.upper(): getattr(args, f"{name}_log")
        for name in ("d83", "d81", "d62")
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D83"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    markers = (
        "Traceback", "RuntimeError", "KeyError", "OOM", "OutOfMemory",
        "Killed", "NaN", "Inf",
    )
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d83.full_performance_summary.v1",
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
            "stderr_size": args.stderr.stat().st_size,
            "error_marker_counts": {
                marker: stdout_text.count(marker) + stderr_text.count(marker)
                for marker in markers
            },
        },
        "metadata": metadata,
        "all_candidates": base.candidate_summary(logs["D83"]),
        "D83_INT8": {
            "aggregate": base.aggregate(target["D83"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D83"]).items()
            },
            "classes": base.class_summary(target["D83"]),
            "outer_rows": base.detailed_rows(target["D83"]),
            "mechanism": _mechanism(target["D83"], metadata),
            "training": _training(target["D83"]),
            "quantization": quant(target["D83"]),
            "resources": _resources(target["D83"]),
        },
        "D83_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D81", "D62"):
        comparison = base.matched_delta(target["D83"], target[baseline])
        result[f"D83_vs_{baseline}"] = {
            "D83": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": base.sha256(args.output),
        "D83_INT8": result["D83_INT8"]["aggregate"],
        "D83_vs_D81": result["D83_vs_D81"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
