#!/usr/bin/env python3
"""Create the complete D74 orthogonal-nuisance performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d74_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["geometry_summary"]["d74_projection_audit"] for row in rows]
    numeric = (
        "centroid_span_rank",
        "orthogonal_residual_rank",
        "projection_rank",
        "projection_removed_rank",
        "removed_residual_energy_fraction",
        "within_residual_energy_before",
        "within_residual_energy_after",
        "within_residual_energy_removed_fraction",
        "support_accuracy_base",
        "support_accuracy_projected",
        "support_prediction_change_count",
        "centroid_direction_max_abs",
        "centroid_pairwise_squared_distance_drift_max",
        "projector_annihilation_l2",
        "projector_idempotence_max_abs_error",
        "projector_symmetry_max_abs_error",
        "compiled_direction_coefficient_max_abs",
    )
    result: dict[str, Any] = {
        "fit_count": len(audits),
        "status_counts": _counts(audit["status"] for audit in audits),
        "projection_active_count": sum(audit["projection_active"] for audit in audits),
        "unique_direction_sha256": len({audit["direction_sha256"] for audit in audits}),
        "ground_component_input_count": 0,
        "uses_outer_held_or_query_for_fit": False,
        "class_id_specific_formula": False,
        "class_permutation_equivariant": True,
        "single_affine_state_only": True,
    }
    result.update(
        {key: base.stats(audit[key] for audit in audits) for key in numeric}
    )
    result["by_scene"] = {
        scene: {
            key: base.stats(
                row["geometry_summary"]["d74_projection_audit"][key]
                for row in group
            )
            for key in (
                "removed_residual_energy_fraction",
                "within_residual_energy_removed_fraction",
                "support_prediction_change_count",
                "support_accuracy_projected",
            )
        }
        for scene, group in base.scene_groups(rows).items()
    }
    return result


def training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trace_lengths = [len(row["training_trace"]) for row in rows]
    return {
        "complete_trace_length": base.stats(trace_lengths),
        "all_trace_lengths_20": all(length == 20 for length in trace_lengths),
        "stage2b_epoch1_20": base.training_summary(rows),
        "stage2c_optimizer_steps": base.stats(
            row["resource"]["stage2c_optimizer_steps"] for row in rows
        ),
        "projection_is_closed_form_no_optimizer_step": True,
        "query_rows_used": 0,
    }


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d74_projection_removed_rank",
        "d74_additional_component_fit_count",
        "d74_additional_lda_fit_macs",
        "d74_fisher_dense_mac_upper_bound",
        "d74_gate_scalar_mac_equivalents",
        "d74_projection_mac_equivalent_upper_bound",
        "d74_affine_compile_mac_equivalents",
        "d74_base_metric_adaptation_macs",
        "d74_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "d74_query_extra_mac_equivalents",
        "d74_persistent_state_extra_bytes",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "stage2c_optimizer_steps",
        "d74_ground_component_input_count",
        "d74_dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d74_single_affine_state_only",
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
    names = ("d74", "d73", "d72", "d71", "d62", "d61")
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
    fp32 = [row for row in logs["D74"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d74.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D74"]),
        "D74_INT8": {
            "aggregate": base.aggregate(target["D74"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D74"]).items()
            },
            "classes": base.class_summary(target["D74"]),
            "outer_rows": base.detailed_rows(target["D74"]),
            "mechanism": mechanism(target["D74"]),
            "training": training(target["D74"]),
            "quantization": quant(target["D74"]),
            "resources": resources(target["D74"]),
        },
        "D74_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D73", "D72", "D71", "D62", "D61"):
        comparison = base.matched_delta(target["D74"], target[baseline])
        result[f"D74_vs_{baseline}"] = {
            "D74": comparison.pop("D49"),
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
