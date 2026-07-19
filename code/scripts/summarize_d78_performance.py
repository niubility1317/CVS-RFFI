#!/usr/bin/env python3
"""Create the complete D78 ground-tangent performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D77_SUMMARY = SCRIPT_DIR / "summarize_d77_performance.py"
SPEC = importlib.util.spec_from_file_location("d78_summary_helper", D77_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D77 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["geometry_summary"]["d78_worstclass_margin_audit"] for row in rows]
    tangents = [row["geometry_summary"]["d78_ground_tangent_audit"] for row in rows]
    numeric = (
        "tangent_rank",
        "crossfit_fold_count",
        "crossfit_held_row_count",
        "crossfit_unique_lda_coefficient_count",
        "temperature_from_initial_mean_class_loss",
        "initial_objective",
        "final_objective",
        "objective_delta",
        "initial_worst_class_top2_loss",
        "final_worst_class_top2_loss",
        "initial_mean_class_top2_loss",
        "final_mean_class_top2_loss",
        "base_margin_min",
        "base_margin_mean",
        "final_margin_min",
        "final_margin_mean",
        "nonpositive_margin_count_before",
        "nonpositive_margin_count_after",
        "trust_radius_frobenius",
        "tangent_coefficient_frobenius",
        "residual_frobenius",
        "oof_ce_before_mean",
        "oof_ce_after_mean",
        "oof_ce_delta_mean",
        "oof_ce_delta_max_class",
        "oof_ce_delta_min_class",
        "oof_base_correct_count",
        "oof_updated_correct_count",
        "oof_correct_delta",
        "support_prediction_change_count",
        "support_accuracy_base",
        "support_accuracy_updated",
    )
    result: dict[str, Any] = {
        "fit_count": len(audits),
        "status_counts": _counts(audit["status"] for audit in audits),
        "active_count": sum(audit["residual_active"] for audit in audits),
        "fallback_count": sum(not audit["residual_active"] for audit in audits),
        "unique_residual_sha256": len({audit["residual_sha256"] for audit in audits}),
        "ground_component_input_count": _counts(
            audit["ground_component_input_count"] for audit in audits
        ),
        "ground_component_formal_phase2_eligible": False,
        "ground_component_provenance_status": _counts(
            audit["ground_component_provenance_status"] for audit in audits
        ),
        "ground_class_score_access": False,
        "uses_outer_held_or_query_for_fit": False,
        "class_id_specific_formula": False,
        "class_permutation_equivariant": True,
        "single_affine_state_only": True,
        "tangent": {
            key: base.stats(value[key] for value in tangents)
            for key in (
                "ground_registry_domain_count",
                "ground_domain_count",
                "ground_component_input_count",
                "numerical_rank",
                "tangent_rank",
                "singular_value_max",
                "singular_value_min_kept",
                "retained_energy_fraction",
                "basis_orthonormality_max_abs_error",
            )
        },
        "unique_tangent_projector_sha256": len(
            {value["projector_sha256"] for value in tangents}
        ),
    }
    result.update({key: base.stats(audit[key] for audit in audits) for key in numeric})
    result["per_class_top2_loss_delta"] = {
        str(index): base.stats(
            audit["oof_per_class_top2_loss_after"][index]
            - audit["oof_per_class_top2_loss_before"][index]
            for audit in audits
        )
        for index in range(11)
    }
    result["per_class_oof_ce_delta"] = {
        str(index): base.stats(audit["oof_per_class_ce_delta"][index] for audit in audits)
        for index in range(11)
    }
    result["optimizer"] = {
        "iteration_count": _counts(len(audit["optimizer_objective_trace"]) for audit in audits),
        "accepted_step_all_iterations": base.stats(
            item["accepted_step"]
            for audit in audits
            for item in audit["optimizer_objective_trace"]
        ),
        "backtracking_count_all_iterations": base.stats(
            item["backtracking_count"]
            for audit in audits
            for item in audit["optimizer_objective_trace"]
        ),
        "gradient_frobenius_all_iterations": base.stats(
            item["gradient_frobenius"]
            for audit in audits
            for item in audit["optimizer_objective_trace"]
        ),
        "monotone_all_steps": all(
            item["objective_after"] <= item["objective_before"] + 1e-10
            for audit in audits
            for item in audit["optimizer_objective_trace"]
        ),
    }
    result["by_scene"] = {
        scene: {
            "active_count": sum(
                row["geometry_summary"]["d78_worstclass_margin_audit"]["residual_active"]
                for row in group
            ),
            **{
                key: base.stats(
                    row["geometry_summary"]["d78_worstclass_margin_audit"][key]
                    for row in group
                )
                for key in (
                    "objective_delta",
                    "nonpositive_margin_count_before",
                    "nonpositive_margin_count_after",
                    "oof_ce_delta_mean",
                    "oof_correct_delta",
                    "residual_frobenius",
                    "support_prediction_change_count",
                )
            },
        }
        for scene, group in base.scene_groups(rows).items()
    }
    return result


def training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [row["training_trace"] for row in rows]
    stage2b = [
        item
        for trace in traces
        for item in trace
        if item["phase"] == "stage2b_fullbatch_old_adaptation"
    ]
    tangent = [
        item
        for trace in traces
        for item in trace
        if item["phase"] == "stage2c_ground_tangent_worstclass_top2_margin"
    ]
    return {
        "complete_trace_length": base.stats(len(trace) for trace in traces),
        "all_trace_lengths_40": all(len(trace) == 40 for trace in traces),
        "stage2b_record_count": len(stage2b),
        "stage2b_loss": base.stats(item["loss"] for item in stage2b),
        "stage2b_ce_loss": base.stats(item["ce_loss"] for item in stage2b),
        "stage2b_support_accuracy": base.stats(item["support_accuracy"] for item in stage2b),
        "tangent_optimizer_record_count": len(tangent),
        "tangent_objective": base.stats(item["loss"] for item in tangent),
        "tangent_oof_ce": base.stats(item["ce_loss"] for item in tangent),
        "tangent_accepted_step": base.stats(item["line_search_gamma"] for item in tangent),
        "tangent_oof_support_accuracy": base.stats(item["support_accuracy"] for item in tangent),
        "stage2c_optimizer_steps": base.stats(
            row["resource"]["stage2c_optimizer_steps"] for row in rows
        ),
        "query_rows_used": 0,
    }


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d78_crossfit_fold_count",
        "d78_crossfit_held_row_count",
        "d78_crossfit_lda_fit_count",
        "d78_crossfit_lda_fit_macs",
        "d78_top2_setup_mac_upper_bound",
        "d78_worstclass_optimizer_mac_upper_bound",
        "d78_top2_ce_audit_mac_upper_bound",
        "d78_tangent_projection_macs",
        "d78_affine_compile_mac_equivalents",
        "d78_ground_statistics_macs",
        "d78_non_lda_added_adaptation_macs",
        "d78_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "peak_trainable_parameters",
        "persistent_state_bytes",
        "d78_compiled_affine_state_bytes",
        "d78_ground_component_logical_state_bytes",
        "d78_component_inclusive_persistent_state_bytes",
        "d78_transient_dequantized_ground_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "total_optimizer_steps",
        "stage2c_optimizer_steps",
        "d78_transient_tangent_parameter_count",
        "d78_ground_component_input_count",
        "d78_dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d78_single_affine_state_only",
        "d78_ground_component_update_access",
        "d78_ground_class_score_access",
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
    names = ("d78", "d77", "d75", "d66", "d62")
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
    fp32 = [row for row in logs["D78"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    error_markers = (
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
        "schema": "cvs.phase2.d78.full_performance_summary.v1",
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
                for marker in error_markers
            },
        },
        "metadata": metadata,
        "all_candidates": base.candidate_summary(logs["D78"]),
        "D78_INT8": {
            "aggregate": base.aggregate(target["D78"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D78"]).items()
            },
            "classes": base.class_summary(target["D78"]),
            "outer_rows": base.detailed_rows(target["D78"]),
            "mechanism": mechanism(target["D78"]),
            "training": training(target["D78"]),
            "quantization": quant(target["D78"]),
            "resources": resources(target["D78"]),
        },
        "D78_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D77", "D75", "D66", "D62"):
        comparison = base.matched_delta(target["D78"], target[baseline])
        result[f"D78_vs_{baseline}"] = {
            "D78": comparison.pop("D49"),
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
                "D78_INT8": result["D78_INT8"]["aggregate"],
                "D78_vs_D62": result["D78_vs_D62"],
                "mechanism": result["D78_INT8"]["mechanism"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
