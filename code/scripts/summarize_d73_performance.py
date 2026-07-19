#!/usr/bin/env python3
"""Create the complete D73 conflict-projected metric performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d73_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["geometry_summary"]["d73_metric_audit"] for row in rows]
    final_d62 = [
        row["geometry_summary"]["final_covariance_audit"] for row in rows
    ]
    return {
        "fit_count": len(audits),
        "status_counts": _counts(audit["status"] for audit in audits),
        "stage2c_step_count": int(
            sum(audit["stage2c_step_count"] for audit in audits)
        ),
        "conflict_projection_count": int(
            sum(audit["conflict_projection_active"] for audit in audits)
        ),
        "task_gradient_cosine": base.stats(
            audit["task_gradient_cosine"] for audit in audits
        ),
        "old_loss_before": base.stats(
            audit["old_loss_before"] for audit in audits
        ),
        "old_loss_after": base.stats(
            audit["old_loss_after"] for audit in audits
        ),
        "old_loss_change": base.stats(
            audit["old_loss_after"] - audit["old_loss_before"]
            for audit in audits
        ),
        "new_loss_before": base.stats(
            audit["new_loss_before"] for audit in audits
        ),
        "new_loss_after": base.stats(
            audit["new_loss_after"] for audit in audits
        ),
        "new_loss_change": base.stats(
            audit["new_loss_after"] - audit["new_loss_before"]
            for audit in audits
        ),
        "old_support_accuracy_before": base.stats(
            audit["old_support_accuracy_before"] for audit in audits
        ),
        "old_support_accuracy_after": base.stats(
            audit["old_support_accuracy_after"] for audit in audits
        ),
        "new_support_accuracy_before": base.stats(
            audit["new_support_accuracy_before"] for audit in audits
        ),
        "new_support_accuracy_after": base.stats(
            audit["new_support_accuracy_after"] for audit in audits
        ),
        "old_gradient_l2": base.stats(
            audit["old_gradient_l2"] for audit in audits
        ),
        "new_gradient_l2": base.stats(
            audit["new_gradient_l2"] for audit in audits
        ),
        "delta_l2": base.stats(audit["delta_l2"] for audit in audits),
        "delta_rms": base.stats(audit["delta_rms"] for audit in audits),
        "delta_max_abs": base.stats(
            audit["delta_max_abs"] for audit in audits
        ),
        "old_first_order_change": base.stats(
            audit["old_first_order_change"] for audit in audits
        ),
        "new_first_order_change": base.stats(
            audit["new_first_order_change"] for audit in audits
        ),
        "unique_direction_sha256": len(
            {audit["direction_sha256"] for audit in audits}
        ),
        "final_d62_boundary_status_counts": _counts(
            audit["d62_boundary_status"] for audit in final_d62
        ),
        "final_d62_active_fit_count": int(
            sum(any(audit["d62_final_accept_mask"]) for audit in final_d62)
        ),
        "final_d62_accepted_row_count": int(
            sum(sum(audit["d62_final_accept_mask"]) for audit in final_d62)
        ),
        "ground_component_input_count": 0,
        "uses_outer_held_or_query_for_fit": False,
        "query_role_branch": False,
        "single_affine_state_only": True,
        "by_scene": {
            scene: {
                "task_gradient_cosine": base.stats(
                    row["geometry_summary"]["d73_metric_audit"][
                        "task_gradient_cosine"
                    ]
                    for row in group
                ),
                "old_loss_change": base.stats(
                    row["geometry_summary"]["d73_metric_audit"][
                        "old_loss_after"
                    ]
                    - row["geometry_summary"]["d73_metric_audit"][
                        "old_loss_before"
                    ]
                    for row in group
                ),
                "new_loss_change": base.stats(
                    row["geometry_summary"]["d73_metric_audit"][
                        "new_loss_after"
                    ]
                    - row["geometry_summary"]["d73_metric_audit"][
                        "new_loss_before"
                    ]
                    for row in group
                ),
            }
            for scene, group in base.scene_groups(rows).items()
        },
    }


def training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage2b_rows: list[dict[str, Any]] = []
    trace_lengths: list[int] = []
    for row in rows:
        trace = list(row["training_trace"])
        trace_lengths.append(len(trace))
        copied = dict(row)
        copied["training_trace"] = trace[:20]
        stage2b_rows.append(copied)
    return {
        "complete_trace_length": base.stats(trace_lengths),
        "all_trace_lengths_21": all(length == 21 for length in trace_lengths),
        "stage2b_epoch1_20": base.training_summary(stage2b_rows),
        "stage2c_epoch21": {
            key: base.stats(
                row["geometry_summary"]["d73_metric_audit"][key]
                for row in rows
            )
            for key in (
                "old_loss_before",
                "old_loss_after",
                "new_loss_before",
                "new_loss_after",
                "old_gradient_l2",
                "new_gradient_l2",
                "task_gradient_cosine",
                "delta_l2",
            )
        },
        "query_rows_used": 0,
    }


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d73_stage2c_step_count",
        "d73_metric_parameter_count",
        "d73_additional_component_fit_count",
        "d73_additional_lda_fit_macs",
        "d73_fisher_dense_mac_upper_bound",
        "d73_gate_scalar_mac_equivalents",
        "d73_gradient_mac_equivalent_upper_bound",
        "d73_base_metric_adaptation_macs",
        "d73_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "d73_query_extra_mac_equivalents",
        "d73_persistent_state_extra_bytes",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "stage2c_optimizer_steps",
        "d73_ground_component_input_count",
        "d73_dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d73_single_affine_state_only",
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
    )
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in invariants},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d73", "d72", "d71", "d62", "d61")
    for name in names:
        parser.add_argument(f"--{name}-log", required=True, type=Path)
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
    fp32 = [row for row in logs["D73"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d73.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D73"]),
        "D73_INT8": {
            "aggregate": base.aggregate(target["D73"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D73"]).items()
            },
            "classes": base.class_summary(target["D73"]),
            "outer_rows": base.detailed_rows(target["D73"]),
            "mechanism": mechanism(target["D73"]),
            "training": training(target["D73"]),
            "quantization": quant(target["D73"]),
            "resources": resources(target["D73"]),
        },
        "D73_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D72", "D71", "D62", "D61"):
        comparison = base.matched_delta(target["D73"], target[baseline])
        result[f"D73_vs_{baseline}"] = {
            "D73": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
