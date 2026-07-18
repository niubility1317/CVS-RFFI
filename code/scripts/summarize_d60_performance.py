#!/usr/bin/env python3
"""Create the complete D60 performance and spectral-stability ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D59_SUMMARY = SCRIPT_DIR / "summarize_d59_performance.py"
SPEC = importlib.util.spec_from_file_location("d60_summary_helper", D59_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D59 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.helper.quantization_summary
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "fit_count": len(audits),
            "stability": base.stats(
                value for audit in audits for value in audit["d60_stability_by_mode"]
            ),
            "per_fit_stability_mean": base.stats(audit["d60_stability_mean"] for audit in audits),
            "zero_mode_count": base.stats(audit["d60_stability_zero_count"] for audit in audits),
            "near_one_mode_count": base.stats(audit["d60_stability_near_one_count"] for audit in audits),
            "fold_rayleigh_mean_abs": base.stats(audit["d60_fold_rayleigh_mean_abs"] for audit in audits),
            "contracted_condition_number": base.stats(
                audit["d60_contracted_covariance_condition_number"] for audit in audits
            ),
            "contracted_normalized_eigenvalue_min": base.stats(
                audit["d60_contracted_normalized_eigenvalue_min"] for audit in audits
            ),
            "contracted_normalized_eigenvalue_max": base.stats(
                audit["d60_contracted_normalized_eigenvalue_max"] for audit in audits
            ),
            "unique_fold_rayleigh_sha256": len({audit["d60_fold_rayleigh_sha256"] for audit in audits}),
            "all_partitions_exact_once": all(
                audit["d60_inner_partition"]["held_support_row_exact_once_coverage"]
                for audit in audits
            ),
        }
    result["by_scene"] = {}
    for scene, group in base.scene_groups(rows).items():
        result["by_scene"][scene] = {
            phase: {
                "stability_mean": base.mean(
                    row["geometry_summary"][f"{phase}_covariance_audit"]["d60_stability_mean"]
                    for row in group
                ),
                "condition_number_mean": base.mean(
                    row["geometry_summary"][f"{phase}_covariance_audit"]["d60_contracted_covariance_condition_number"]
                    for row in group
                ),
            }
            for phase in ("before", "final")
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count", "estimated_lda_fit_macs",
        "d60_inner_covariance_fit_count", "d60_inner_covariance_fit_macs",
        "d60_spectral_dense_algebra_mac_equivalent_upper_bound",
        "estimated_adaptation_macs", "estimated_metric_adaptation_macs",
        "estimated_macs_per_query", "trainable_parameters", "persistent_state_bytes",
        "registry_state_bytes", "peak_cuda_memory_bytes", "adaptation_epochs", "optimizer_steps",
    )
    first = rows[0]["resource"]
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in (
            "runtime_device", "deployment_precision", "coefficient_dtype", "intercept_dtype",
            "query_rows_used_for_fit", "query_features_used_for_fit", "query_labels_used_for_fit",
            "query_role_oracle_access", "query_class_quota_access", "query_true_batch_class_count_access",
            "query_batch_global_assignment", "query_dependent_batch_optimization", "source_sample_access",
            "clean_sample_access", "dense_query_graph_bytes", "d60_resource_single_affine_state_only",
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("d60", "d46", "d59", "d58"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {name.upper(): getattr(args, f"{name}_log") for name in ("d60", "d46", "d59", "d58")}
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {name: [row for row in rows if row["candidate_id"] == TARGET] for name, rows in logs.items()}
    fp32 = [row for row in logs["D60"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d60.full_performance_summary.v1",
        "input": {name: {"path": str(path), "size": path.stat().st_size, "sha256": base.sha256(path), "rows": len(logs[name])} for name, path in paths.items()},
        "all_candidates": base.candidate_summary(logs["D60"]),
        "D60_INT8": {
            "aggregate": base.aggregate(target["D60"]),
            "by_scene": {scene: base.aggregate(group) for scene, group in base.scene_groups(target["D60"]).items()},
            "classes": base.class_summary(target["D60"]),
            "outer_rows": base.detailed_rows(target["D60"]),
            "mechanism": mechanism(target["D60"]),
            "training": base.training_summary(target["D60"]),
            "quantization": quant(target["D60"]),
            "resources": resources(target["D60"]),
        },
        "D60_FP32_MATCHED": {"aggregate": base.aggregate(fp32)},
    }
    for baseline in ("D46", "D59", "D58"):
        result[f"D60_vs_{baseline}"] = base.matched_delta(target["D60"], target[baseline])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
