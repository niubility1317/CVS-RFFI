#!/usr/bin/env python3
"""Create the complete D70 atomic-lifecycle performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d70_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        masks = [np.asarray(audit["d70_final_accept_mask"], dtype=bool) for audit in audits]
        accepted_by_class = np.sum(np.stack(masks), axis=0).astype(int)
        active_rows = [
            {
                "scenario": row["scenario"],
                "fold_index": int(row["fold_index"]),
                "mask": audit["d70_final_accept_mask"],
                "gate_status": audit["d70_gate_status"],
                "base_positive": audit["d70_base_positive_by_class"],
                "base_false_positive": audit["d70_base_false_positive_by_class"],
                "joint_positive": audit["d70_joint_positive_by_class"],
                "joint_false_positive": audit["d70_joint_false_positive_by_class"],
            }
            for row, audit, mask in zip(rows, audits, masks)
            if np.any(mask)
        ]
        coefficient_delta = []
        bias_delta = []
        for audit in audits:
            actual = np.asarray(audit["d70_actual_coefficient_fp32"], dtype=np.float64)
            base_joint = np.asarray(
                audit["d70_base_joint_coefficient_fp32"], dtype=np.float64
            )
            actual_bias = np.asarray(audit["d70_actual_intercept_fp32"], dtype=np.float64)
            base_bias = np.asarray(
                audit["d70_base_joint_intercept_fp32"], dtype=np.float64
            )
            coefficient_delta.append(float(np.linalg.norm(actual - base_joint)))
            bias_delta.append(float(np.linalg.norm(actual_bias - base_bias)))
        result[phase] = {
            "fit_count": len(audits),
            "gate_status_counts": _counts(audit["d70_gate_status"] for audit in audits),
            "active_fit_count": int(sum(np.any(mask) for mask in masks)),
            "accepted_old_row_count": int(sum(np.sum(mask) for mask in masks)),
            "accepted_count_by_old_class": accepted_by_class.tolist(),
            "active_rows": active_rows,
            "partition_count": int(
                sum(len(audit["d70_partition_audit"]) for audit in audits)
            ),
            "partition_exact_once": all(
                audit["d70_partition_audit"] == []
                or sorted(
                    index
                    for partition in audit["d70_partition_audit"]
                    for index in partition["held_indices"]
                )
                == list(range(int(audit["d70_class_count"] * audit["d70_actual_k"])))
                for audit in audits
            ),
            "compiled_support_accuracy": base.stats(
                audit["d70_compiled_support_accuracy"] for audit in audits
            ),
            "coefficient_l2_delta_vs_d62_joint": base.stats(coefficient_delta),
            "bias_l2_delta_vs_d62_joint": base.stats(bias_delta),
            "new_rows_match_joint_d62": all(
                audit["d70_new_rows_match_joint_d62"] for audit in audits
            ),
            "ground_component_input_count": base.stats(
                audit["d70_ground_component_input_count"] for audit in audits
            ),
            "uses_outer_held_or_query": any(
                audit["d70_uses_outer_held_or_query"] for audit in audits
            ),
            "role_class_scene_or_query_branch": any(
                audit["d70_old_new_role_specific_query_branch"]
                or audit["d70_class_id_specific_formula"]
                or audit["d70_scene_receiver_handle_specific_branch"]
                or audit["d70_query_joint_optimization"]
                for audit in audits
            ),
            "single_affine_state_only": all(
                audit["d70_single_affine_state_only"] for audit in audits
            ),
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d70_inner_d62_fit_count",
        "d70_inner_component_fit_count",
        "d70_inner_lda_fit_macs",
        "d70_inner_fisher_dense_mac_upper_bound",
        "d70_held_score_macs",
        "d70_gate_scalar_mac_equivalents",
        "d70_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "d70_query_extra_macs",
        "d70_persistent_state_extra_bytes",
        "d70_optimizer_steps_extra",
        "d70_ground_component_input_count",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d70_resource_single_affine_state_only",
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
    names = ("d70", "d69", "d68", "d67", "d66", "d65", "d62")
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
    fp32 = [row for row in logs["D70"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d70.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D70"]),
        "D70_INT8": {
            "aggregate": base.aggregate(target["D70"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D70"]).items()
            },
            "classes": base.class_summary(target["D70"]),
            "outer_rows": base.detailed_rows(target["D70"]),
            "mechanism": mechanism(target["D70"]),
            "training": base.training_summary(target["D70"]),
            "quantization": quant(target["D70"]),
            "resources": resources(target["D70"]),
        },
        "D70_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D69", "D68", "D67", "D66", "D65", "D62"):
        comparison = base.matched_delta(target["D70"], target[baseline])
        result[f"D70_vs_{baseline}"] = {
            "D70": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
