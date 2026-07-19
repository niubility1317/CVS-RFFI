#!/usr/bin/env python3
"""Create the complete D67 row-stacking performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d67_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _flatten(values: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for group in values for value in group]


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = h.mechanism(rows)
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        alpha = _flatten(audit["d67_alpha_by_class"] for audit in audits)
        solver = [audit["d67_solver_audit"] for audit in audits]
        result[f"d67_{phase}"] = {
            "fit_count": len(audits),
            "class_alpha": base.stats(alpha),
            "row_alpha_mean": base.stats(audit["d67_alpha_mean"] for audit in audits),
            "row_alpha_max": base.stats(audit["d67_alpha_max"] for audit in audits),
            "d62_boundary_count": base.stats(
                audit["d67_alpha_d62_boundary_count"] for audit in audits
            ),
            "d65_boundary_count": base.stats(
                audit["d67_alpha_d65_boundary_count"] for audit in audits
            ),
            "interior_count": base.stats(
                audit["d67_alpha_interior_count"] for audit in audits
            ),
            "risk_d62": base.stats(_flatten(item["risk_d62"] for item in solver)),
            "risk_d65": base.stats(_flatten(item["risk_d65"] for item in solver)),
            "risk_stacked": base.stats(
                _flatten(item["risk_stacked"] for item in solver)
            ),
            "compiled_support_accuracy": base.stats(
                audit["d67_compiled_support_accuracy"] for audit in audits
            ),
            "compile_float32_error_max": base.stats(
                audit["d67_compile_float32_error_max"] for audit in audits
            ),
            "crossfit_fold_count": base.stats(
                audit["d67_crossfit_fold_count"] for audit in audits
            ),
            "hyperparameter_count": base.stats(
                audit["d67_hyperparameter_count"] for audit in audits
            ),
            "uses_outer_held_or_query": any(
                audit["d67_uses_outer_held_or_query"] for audit in audits
            ),
            "role_class_scene_or_query_branch": any(
                audit["d67_old_new_role_specific_query_branch"]
                or audit["d67_class_id_specific_formula"]
                or audit["d67_scene_receiver_handle_specific_branch"]
                or audit["d67_query_joint_optimization"]
                for audit in audits
            ),
            "single_affine_state_only": all(
                audit["d67_single_affine_state_only"] for audit in audits
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
        "d67_crossfit_fold_count_per_stage",
        "d67_d65_expert_covariance_fit_count",
        "d67_d65_expert_total_adaptation_macs",
        "d67_inner_d62_lda_fit_count",
        "d67_inner_d62_total_adaptation_macs",
        "d67_standardization_stacking_scalar_macs",
        "d67_total_added_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "d67_persistent_state_extra_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "d67_optimizer_steps_extra",
    )
    first = rows[0]["resource"]
    missing = [key for key in keys if key not in first]
    if missing:
        raise KeyError(f"missing D67 resource keys: {missing}")
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d67_query_extra_macs",
        "d67_resource_single_affine_state_only",
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
    names = ("d67", "d66", "d65", "d64", "d63", "d62", "d61", "d46")
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
    fp32 = [row for row in logs["D67"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d67.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D67"]),
        "D67_INT8": {
            "aggregate": base.aggregate(target["D67"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D67"]).items()
            },
            "classes": base.class_summary(target["D67"]),
            "outer_rows": base.detailed_rows(target["D67"]),
            "mechanism": mechanism(target["D67"]),
            "training": base.training_summary(target["D67"]),
            "quantization": quant(target["D67"]),
            "resources": resources(target["D67"]),
        },
        "D67_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D66", "D65", "D64", "D63", "D62", "D61", "D46"):
        result[f"D67_vs_{baseline}"] = base.matched_delta(
            target["D67"], target[baseline]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
