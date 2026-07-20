#!/usr/bin/env python3
"""Create the complete D87 v2 ground-radius sigma-margin performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d79_performance.py"
SPEC = importlib.util.spec_from_file_location("d87_summary_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D79 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["geometry_summary"]["d79_worstclass_margin_audit"] for row in rows]
    tangents = [row["geometry_summary"]["d79_ground_tangent_audit"] for row in rows]
    numeric = (
        "effective_rank",
        "counterfactual_domain_count",
        "crossfit_fold_count",
        "crossfit_held_row_count",
        "initial_objective",
        "final_objective",
        "objective_delta",
        "initial_worst_class_sigma_ce",
        "final_worst_class_sigma_ce",
        "initial_mean_sigma_ce",
        "final_mean_sigma_ce",
        "oof_clean_ce_before_mean",
        "oof_clean_ce_after_mean",
        "oof_clean_ce_delta_max_class",
        "oof_clean_correct_before",
        "oof_clean_correct_after",
        "trust_radius_frobenius",
        "residual_frobenius",
        "bias_residual_frobenius",
        "residual_logit_at_support_center_max_abs",
        "support_prediction_change_count",
        "support_accuracy_base",
        "support_accuracy_updated",
    )
    result = {
        "fit_count": len(audits),
        "active_count": sum(bool(audit["residual_active"]) for audit in audits),
        "unique_residual_sha256": len({audit["residual_sha256"] for audit in audits}),
        "unique_bias_residual_sha256": len(
            {audit["bias_residual_sha256"] for audit in audits}
        ),
        "sigma_weights": {"original": 0.5, "plus": 0.25, "minus": 0.25},
        "all_grouped_physical_crossfit": all(
            audit["physical_group_crossfit_preserved"] for audit in audits
        ),
        "all_views_not_physical_samples": all(
            not audit["counterfactual_views_count_as_physical_samples"]
            for audit in audits
        ),
        "all_query_free": all(int(audit["query_rows_used"]) == 0 for audit in audits),
        "all_class_symmetric": all(
            audit["old_new_role_specific_branch"] is False
            and audit["class_id_specific_formula"] is False
            for audit in audits
        ),
        "geometry": {
            key: base.stats(value[key] for value in tangents)
            for key in (
                "counterfactual_domain_count",
                "effective_rank",
                "counterfactual_amplitude_min",
                "counterfactual_amplitude_mean",
                "counterfactual_amplitude_max",
                "sigma_covariance_max_abs_error",
                "basis_orthonormality_max_abs_error",
            )
        },
        "optimizer": {
            "iteration_count": {
                str(length): sum(
                    len(audit["optimizer_objective_trace"]) == length
                    for audit in audits
                )
                for length in sorted(
                    {len(audit["optimizer_objective_trace"]) for audit in audits}
                )
            },
            "accepted_step": base.stats(
                item["accepted_step"]
                for audit in audits
                for item in audit["optimizer_objective_trace"]
            ),
            "backtracking_count": base.stats(
                item["backtracking_count"]
                for audit in audits
                for item in audit["optimizer_objective_trace"]
            ),
            "gradient_frobenius": base.stats(
                item["gradient_frobenius"]
                for audit in audits
                for item in audit["optimizer_objective_trace"]
            ),
            "monotone_all_steps": all(
                item["objective_after"] <= item["objective_before"] + 1.0e-10
                for audit in audits
                for item in audit["optimizer_objective_trace"]
            ),
        },
    }
    result.update({key: base.stats(audit[key] for audit in audits) for key in numeric})
    result["by_scene"] = {
        scene: {
            key: base.stats(
                row["geometry_summary"]["d79_worstclass_margin_audit"][key]
                for row in group
            )
            for key in (
                "objective_delta",
                "oof_clean_ce_delta_max_class",
                "residual_frobenius",
                "support_prediction_change_count",
            )
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
    sigma = [
        item
        for trace in traces
        for item in trace
        if item["phase"] == "stage2c_centered_ground_tangent_worstclass_top2_margin"
    ]
    return {
        "complete_trace_length": base.stats(len(trace) for trace in traces),
        "stage2b_record_count": len(stage2b),
        "stage2b_loss": base.stats(item["loss"] for item in stage2b),
        "stage2b_support_accuracy": base.stats(item["support_accuracy"] for item in stage2b),
        "sigma_optimizer_record_count": len(sigma),
        "sigma_objective": base.stats(item["loss"] for item in sigma),
        "sigma_clean_oof_ce": base.stats(item["ce_loss"] for item in sigma),
        "sigma_support_accuracy": base.stats(item["support_accuracy"] for item in sigma),
        "query_rows_used": 0,
    }


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d79_crossfit_lda_fit_macs",
        "d79_worstclass_optimizer_mac_upper_bound",
        "d79_non_lda_added_adaptation_macs",
        "d79_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "peak_trainable_parameters",
        "persistent_state_bytes",
        "d79_compiled_affine_state_bytes",
        "d79_ground_component_logical_state_bytes",
        "d79_component_inclusive_persistent_state_bytes",
        "d79_transient_dequantized_ground_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "stage2c_optimizer_steps",
        "d79_transient_tangent_parameter_count",
        "d79_dense_query_graph_bytes",
        "d79_query_extra_mac_equivalents",
        "d79_query_extra_state_bytes",
    )
    first = rows[0]["resource"]
    invariant_keys = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d79_single_affine_state_only",
        "query_rows_used_for_fit",
        "query_features_used_for_fit",
        "query_labels_used_for_fit",
        "query_role_oracle_access",
        "query_class_quota_access",
        "query_dependent_batch_optimization",
        "source_sample_access",
        "clean_sample_access",
        "persistent_state_cap_pass",
        "optimizer_step_cap_pass",
        "trainable_parameter_cap_pass",
    )
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in invariant_keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d87", "d85", "d81", "d79", "d78")
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
    fp32 = [row for row in logs["D87"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d87.full_performance_summary.v1",
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
            "stdout_sha256": base.sha256(args.stdout),
            "stderr_size": args.stderr.stat().st_size,
            "stderr_sha256": base.sha256(args.stderr),
            "error_marker_counts": {
                marker: stdout_text.count(marker) + stderr_text.count(marker)
                for marker in ("Traceback", "RuntimeError", "OOM", "NaN", "Inf")
            },
        },
        "metadata": json.loads(args.metadata.read_text(encoding="utf-8-sig")),
        "all_candidates": base.candidate_summary(logs["D87"]),
        "D87_INT8": {
            "aggregate": base.aggregate(target["D87"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D87"]).items()
            },
            "classes": base.class_summary(target["D87"]),
            "outer_rows": base.detailed_rows(target["D87"]),
            "mechanism": mechanism(target["D87"]),
            "training": training(target["D87"]),
            "quantization": quant(target["D87"]),
            "resources": resources(target["D87"]),
        },
        "D87_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D85", "D81", "D79", "D78"):
        comparison = base.matched_delta(target["D87"], target[baseline])
        result[f"D87_vs_{baseline}"] = {
            "D87": comparison.pop("D49"),
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
                "D87_INT8": result["D87_INT8"]["aggregate"],
                "D87_FP32": result["D87_FP32_MATCHED"]["aggregate"],
                "D87_vs_D85": result["D87_vs_D85"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
