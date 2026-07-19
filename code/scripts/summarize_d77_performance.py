#!/usr/bin/env python3
"""Create the complete D77 ground-preconditioned performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d77_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["geometry_summary"]["d77_common_descent_audit"] for row in rows]
    preconditioners = [
        row["geometry_summary"]["d77_ground_preconditioner_audit"] for row in rows
    ]
    numeric = (
        "crossfit_fold_count",
        "crossfit_held_row_count",
        "crossfit_unique_lda_coefficient_count",
        "minimum_common_descent_inner_product",
        "maximum_common_descent_inner_product",
        "preconditioned_direction_norm_sq",
        "lipschitz_min",
        "lipschitz_max",
        "analytic_raw_step",
        "trust_cap_frobenius",
        "trust_scale",
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
        "unique_residual_sha256": len(
            {audit["residual_sha256"] for audit in audits}
        ),
        "unique_gradient_sha256": len(
            {audit["class_gradient_sha256"] for audit in audits}
        ),
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
        "preconditioner": {
            key: base.stats(value[key] for value in preconditioners)
            for key in (
                "reliability_min",
                "reliability_mean",
                "reliability_max",
                "preconditioner_min",
                "preconditioner_mean",
                "preconditioner_max",
                "preconditioner_z_geometric_mean",
                "preconditioner_condition_number",
            )
        },
        "unique_preconditioner_sha256": len(
            {value["preconditioner_sha256"] for value in preconditioners}
        ),
    }
    result.update({key: base.stats(audit[key] for audit in audits) for key in numeric})
    result["per_class_oof_ce_delta"] = {
        str(index): base.stats(audit["oof_per_class_ce_delta"][index] for audit in audits)
        for index in range(11)
    }
    result["simplex_weight_by_class"] = {
        str(index): base.stats(audit["simplex_weights"][index] for audit in audits)
        for index in range(11)
    }
    result["frank_wolfe"] = {
        "iteration_count": _counts(
            len(audit["optimizer_objective_trace"]) for audit in audits
        ),
        "objective_initial": base.stats(
            audit["optimizer_objective_trace"][0]["objective_before"]
            for audit in audits
        ),
        "objective_final": base.stats(
            audit["optimizer_objective_trace"][-1]["objective_after"]
            for audit in audits
        ),
        "objective_delta": base.stats(
            audit["optimizer_objective_trace"][-1]["objective_after"]
            - audit["optimizer_objective_trace"][0]["objective_before"]
            for audit in audits
        ),
        "line_search_gamma_all_steps": base.stats(
            item["line_search_gamma"]
            for audit in audits
            for item in audit["optimizer_objective_trace"]
        ),
        "tied_vertex_count_all_steps": base.stats(
            item["tied_vertex_count"]
            for audit in audits
            for item in audit["optimizer_objective_trace"]
        ),
    }
    result["by_scene"] = {
        scene: {
            "active_count": sum(
                row["geometry_summary"]["d77_common_descent_audit"][
                    "residual_active"
                ]
                for row in group
            ),
            **{
                key: base.stats(
                    row["geometry_summary"]["d77_common_descent_audit"][key]
                    for row in group
                )
                for key in (
                    "oof_ce_delta_mean",
                    "oof_ce_delta_max_class",
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
    fw = [
        item
        for trace in traces
        for item in trace
        if item["phase"]
        == "stage2c_ground_preconditioned_common_descent_frank_wolfe"
    ]
    return {
        "complete_trace_length": base.stats(len(trace) for trace in traces),
        "all_trace_lengths_40": all(len(trace) == 40 for trace in traces),
        "stage2b_record_count": len(stage2b),
        "stage2b_loss": base.stats(item["loss"] for item in stage2b),
        "stage2b_ce_loss": base.stats(item["ce_loss"] for item in stage2b),
        "stage2b_support_accuracy": base.stats(
            item["support_accuracy"] for item in stage2b
        ),
        "frank_wolfe_record_count": len(fw),
        "frank_wolfe_loss": base.stats(item["loss"] for item in fw),
        "frank_wolfe_oof_ce_loss": base.stats(item["ce_loss"] for item in fw),
        "frank_wolfe_line_search_gamma": base.stats(
            item["line_search_gamma"] for item in fw
        ),
        "frank_wolfe_support_accuracy": base.stats(
            item["support_accuracy"] for item in fw
        ),
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
        "d77_crossfit_fold_count",
        "d77_crossfit_held_row_count",
        "d77_crossfit_lda_fit_count",
        "d77_crossfit_lda_fit_macs",
        "d77_oof_gradient_mac_upper_bound",
        "d77_frank_wolfe_mac_upper_bound",
        "d77_oof_ce_audit_mac_upper_bound",
        "d77_preconditioner_application_macs",
        "d77_affine_compile_mac_equivalents",
        "d77_ground_statistics_macs",
        "d77_non_lda_added_adaptation_macs",
        "d77_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "peak_trainable_parameters",
        "persistent_state_bytes",
        "d77_compiled_affine_state_bytes",
        "d77_ground_component_logical_state_bytes",
        "d77_component_inclusive_persistent_state_bytes",
        "d77_transient_dequantized_ground_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "total_optimizer_steps",
        "stage2c_optimizer_steps",
        "d77_ground_component_input_count",
        "d77_dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d77_single_affine_state_only",
        "d77_ground_component_update_access",
        "d77_ground_class_score_access",
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
    names = ("d77", "d75", "d74", "d73", "d66", "d62")
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
    fp32 = [row for row in logs["D77"] if row["candidate_id"] == FP32]
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
        "schema": "cvs.phase2.d77.full_performance_summary.v1",
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
        "all_candidates": base.candidate_summary(logs["D77"]),
        "D77_INT8": {
            "aggregate": base.aggregate(target["D77"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D77"]).items()
            },
            "classes": base.class_summary(target["D77"]),
            "outer_rows": base.detailed_rows(target["D77"]),
            "mechanism": mechanism(target["D77"]),
            "training": training(target["D77"]),
            "quantization": quant(target["D77"]),
            "resources": resources(target["D77"]),
        },
        "D77_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D75", "D74", "D73", "D66", "D62"):
        comparison = base.matched_delta(target["D77"], target[baseline])
        result[f"D77_vs_{baseline}"] = {
            "D77": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": base.sha256(args.output),
        "D77_INT8": result["D77_INT8"]["aggregate"],
        "D77_vs_D62": result["D77_vs_D62"],
        "mechanism": result["D77_INT8"]["mechanism"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
