#!/usr/bin/env python3
"""Create the full D57 performance ledger from complete matched logs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_PATH = SCRIPT_DIR / "summarize_d50_performance.py"
SPEC = importlib.util.spec_from_file_location("d57_summary_helper", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load summary helper")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
base = helper.base
TARGET = "D42-USLDA-INT8"
MATCHED_FP32 = "D42-USLDA-FP32-MATCHED"


def _l2(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def _phase_mechanism(audits: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(audit["d57_boundary_status"]) for audit in audits)
    initial_counts = [sum(audit["d57_initial_accept_mask"]) for audit in audits]
    final_counts = [sum(audit["d57_final_accept_mask"]) for audit in audits]
    crossfit_values = [
        value
        for audit in audits
        if audit["d57_crossfit_delta_by_fold"] is not None
        for fold in audit["d57_crossfit_delta_by_fold"]
        for value in fold
    ]
    return {
        "status_counts": dict(sorted(statuses.items())),
        "atomic_fallback_fit_count": sum(
            bool(audit["d57_joint_atomic_fallback"]) for audit in audits
        ),
        "exact_fallback_fit_count": sum(count == 0 for count in final_counts),
        "active_fit_count": sum(count > 0 for count in final_counts),
        "initial_accepted_coordinate_count": base.stats(initial_counts),
        "final_accepted_coordinate_count": base.stats(final_counts),
        "base_positive_correct": base.stats(
            value
            for audit in audits
            for value in audit["d57_base_positive_correct_by_class"]
        ),
        "coordinate_positive_correct": base.stats(
            value
            for audit in audits
            for value in audit["d57_coordinate_positive_correct_by_class"]
        ),
        "joint_positive_correct": base.stats(
            value
            for audit in audits
            for value in audit["d57_joint_positive_correct_by_class"]
        ),
        "base_false_positive": base.stats(
            value
            for audit in audits
            for value in audit["d57_base_false_positive_by_class"]
        ),
        "coordinate_false_positive": base.stats(
            value
            for audit in audits
            for value in audit["d57_coordinate_false_positive_by_class"]
        ),
        "joint_false_positive": base.stats(
            value
            for audit in audits
            for value in audit["d57_joint_false_positive_by_class"]
        ),
        "coordinate_positive_strict_improvement_count": sum(
            adjusted > original
            for audit in audits
            for adjusted, original in zip(
                audit["d57_coordinate_positive_correct_by_class"],
                audit["d57_base_positive_correct_by_class"],
            )
        ),
        "coordinate_false_positive_strict_improvement_count": sum(
            adjusted < original
            for audit in audits
            for adjusted, original in zip(
                audit["d57_coordinate_false_positive_by_class"],
                audit["d57_base_false_positive_by_class"],
            )
        ),
        "crossfit_delta": base.stats(crossfit_values),
        "crossfit_delta_abs": base.stats(abs(value) for value in crossfit_values),
        "centered_intercept_compensation": base.stats(
            value
            for audit in audits
            for value in audit["d57_centered_intercept_compensation_fp64"]
        ),
        "compensation_l1": base.stats(
            sum(abs(value) for value in audit["d57_centered_intercept_compensation_fp64"])
            for audit in audits
        ),
        "compensation_l2": base.stats(
            _l2(audit["d57_centered_intercept_compensation_fp64"])
            for audit in audits
        ),
        "compensation_abs_max": base.stats(
            max(abs(value) for value in audit["d57_centered_intercept_compensation_fp64"])
            for audit in audits
        ),
        "compensation_sum_abs_error": base.stats(
            abs(sum(audit["d57_centered_intercept_compensation_fp64"]))
            for audit in audits
        ),
        "coefficient_delta_l2": base.stats(
            _l2(
                [
                    actual - baseline
                    for actual_row, baseline_row in zip(
                        audit["d57_actual_coefficient_fp32"],
                        audit["d57_base_coefficient_fp32"],
                    )
                    for actual, baseline in zip(actual_row, baseline_row)
                ]
            )
            for audit in audits
        ),
        "intercept_delta_l2": base.stats(
            _l2(
                [
                    actual - baseline
                    for actual, baseline in zip(
                        audit["d57_actual_intercept_fp32"],
                        audit["d57_base_intercept_fp32"],
                    )
                ]
            )
            for audit in audits
        ),
    }


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        phase: _phase_mechanism(
            [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        )
        for phase in ("before", "final")
    }
    result["by_scene"] = {}
    for scene, group in base.scene_groups(rows).items():
        audits = [row["geometry_summary"]["final_covariance_audit"] for row in group]
        result["by_scene"][scene] = {
            "fit_count": len(audits),
            "active_fit_count": sum(any(audit["d57_final_accept_mask"]) for audit in audits),
            "atomic_fallback_fit_count": sum(
                bool(audit["d57_joint_atomic_fallback"]) for audit in audits
            ),
            "accepted_coordinate_count_mean": base.mean(
                sum(audit["d57_final_accept_mask"]) for audit in audits
            ),
            "compensation_l2_mean": base.mean(
                _l2(audit["d57_centered_intercept_compensation_fp64"])
                for audit in audits
            ),
            "compensation_abs_max": max(
                abs(value)
                for audit in audits
                for value in audit["d57_centered_intercept_compensation_fp64"]
            ),
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d56_additional_lda_fit_count",
        "d56_additional_lda_fit_macs",
        "d56_extra_adaptation_mac_equivalents",
        "d56_additional_comparison_count",
        "d57_additional_lda_fit_count",
        "d57_additional_lda_fit_macs",
        "d57_extra_adaptation_mac_equivalents",
        "d57_additional_comparison_count",
        "estimated_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
    )
    result = {key: base.stats(row["resource"][key] for row in rows) for key in keys}
    first = rows[0]["resource"]
    result["invariants"] = {
        key: first[key]
        for key in (
            "runtime_device",
            "deployment_precision",
            "coefficient_dtype",
            "intercept_dtype",
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
            "d56_resource_reuses_d46_query_state",
            "d57_resource_reuses_d56_exact_fit_inventory",
        )
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    versions = ("d57", "d45", "d46", "d51", "d52", "d53", "d54", "d55", "d56")
    for version in versions:
        parser.add_argument(f"--{version}-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {version.upper(): getattr(args, f"{version}_log") for version in versions}
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected complete 105-row D45-D57 matched logs")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D57"] if row["candidate_id"] == MATCHED_FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target/matched rows")
    output = {
        "schema": "cvs.phase2.d57.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D57"]),
        "D57_INT8": {
            "aggregate": base.aggregate(target["D57"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D57"]).items()
            },
            "classes": base.class_summary(target["D57"]),
            "outer_rows": base.detailed_rows(target["D57"]),
            "mechanism": mechanism(target["D57"]),
            "training": base.training_summary(target["D57"]),
            "quantization": helper.quantization_summary(target["D57"]),
            "resources": resources(target["D57"]),
        },
        "D57_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in versions[1:]:
        output[f"D57_vs_{baseline.upper()}"] = base.matched_delta(
            target["D57"], target[baseline.upper()]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
