#!/usr/bin/env python3
"""Create the complete D69 frozen-append performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d69_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _flatten(values: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for group in values for value in group]


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        old_abs: list[float] = []
        old_l2: list[float] = []
        old_bias: list[float] = []
        for audit in audits:
            old_count = int(audit["d69_old_class_count"])
            actual_coef = np.asarray(
                audit["d69_actual_coefficient_fp32"], dtype=np.float64
            )
            joint_coef = np.asarray(
                audit["d69_joint_d62_coefficient_fp32"], dtype=np.float64
            )
            actual_bias = np.asarray(
                audit["d69_actual_intercept_fp32"], dtype=np.float64
            )
            joint_bias = np.asarray(
                audit["d69_joint_d62_intercept_fp32"], dtype=np.float64
            )
            delta = actual_coef[:old_count] - joint_coef[:old_count]
            bias_delta = actual_bias[:old_count] - joint_bias[:old_count]
            old_abs.append(float(np.max(np.abs(delta))))
            old_l2.append(float(np.linalg.norm(delta)))
            old_bias.append(float(np.max(np.abs(bias_delta))))
        result[phase] = {
            "fit_count": len(audits),
            "phase_counts": _counts(audit["d69_phase"] for audit in audits),
            "old_rows_bitwise_unchanged": all(
                audit["d69_old_row_fp32_bitwise_unchanged"] for audit in audits
            ),
            "new_rows_match_joint_d62": all(
                audit["d69_new_row_fp32_matches_joint_d62"] for audit in audits
            ),
            "before_is_exact_joint_d62": all(
                audit["d69_actual_row_sha256"]
                == audit["d69_joint_d62_row_sha256"]
                for audit in audits
            ),
            "compiled_support_accuracy": base.stats(
                audit["d69_compiled_support_accuracy"] for audit in audits
            ),
            "old_row_max_abs_delta_vs_joint_d62": base.stats(old_abs),
            "old_row_l2_delta_vs_joint_d62": base.stats(old_l2),
            "old_bias_max_abs_delta_vs_joint_d62": base.stats(old_bias),
            "d62_boundary_status": _counts(
                audit["d62_boundary_status"] for audit in audits
            ),
            "d62_accepted_row_count": base.stats(
                sum(audit["d62_final_accept_mask"]) for audit in audits
            ),
            "d62_atomic_safe": _counts(
                str(bool(audit["d62_joint_atomic_safe"])) for audit in audits
            ),
            "ground_component_input_count": base.stats(
                audit["d69_ground_component_input_count"] for audit in audits
            ),
            "uses_outer_held_or_query": any(
                audit["d69_uses_outer_held_or_query"] for audit in audits
            ),
            "role_class_scene_or_query_branch": any(
                audit["d69_old_new_role_specific_query_branch"]
                or audit["d69_class_id_specific_formula"]
                or audit["d69_scene_receiver_handle_specific_branch"]
                or audit["d69_query_joint_optimization"]
                for audit in audits
            ),
            "single_affine_state_only": all(
                audit["d69_single_affine_state_only"] for audit in audits
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
        "estimated_metric_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "d69_append_row_count",
        "d69_query_extra_macs",
        "d69_persistent_state_extra_bytes",
        "d69_optimizer_steps_extra",
        "d69_ground_component_input_count",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d69_int8_old_rows_bitwise_unchanged",
        "d69_fp32_old_rows_bitwise_unchanged",
        "d69_resource_single_affine_state_only",
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
    names = ("d69", "d68", "d67", "d66", "d65", "d62")
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
    fp32 = [row for row in logs["D69"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d69.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D69"]),
        "D69_INT8": {
            "aggregate": base.aggregate(target["D69"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D69"]).items()
            },
            "classes": base.class_summary(target["D69"]),
            "outer_rows": base.detailed_rows(target["D69"]),
            "mechanism": mechanism(target["D69"]),
            "training": base.training_summary(target["D69"]),
            "quantization": quant(target["D69"]),
            "resources": resources(target["D69"]),
        },
        "D69_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D68", "D67", "D66", "D65", "D62"):
        comparison = base.matched_delta(target["D69"], target[baseline])
        result[f"D69_vs_{baseline}"] = {
            "D69": comparison.pop("D49"),
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
