#!/usr/bin/env python3
"""Create the complete D72 leave-one head-bagging performance ledger."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d72_summary_helper", D62_SUMMARY)
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
        audits = [
            row["geometry_summary"][f"d72_{phase}_bagging_audit"]
            for row in rows
        ]
        inner = [
            item
            for audit in audits
            for item in audit["inner_audit_summary"]
        ]
        result[phase] = {
            "fit_count": len(audits),
            "status_counts": _counts(audit["status"] for audit in audits),
            "leave_one_fit_count": int(
                sum(audit["leave_one_fit_count"] for audit in audits)
            ),
            "inner_k_shot_counts": _counts(
                audit["inner_k_shot"] for audit in audits
            ),
            "partition_count": int(
                sum(len(audit["partition_audit"]) for audit in audits)
            ),
            "partition_exact_once": all(
                audit["partition_exact_once"] for audit in audits
            ),
            "train_held_overlap_max": max(
                part["train_held_overlap_count"]
                for audit in audits
                for part in audit["partition_audit"]
            ),
            "support_prediction_change_count": int(
                sum(audit["support_prediction_change_count"] for audit in audits)
            ),
            "support_accuracy_base": base.stats(
                audit["support_accuracy_base"] for audit in audits
            ),
            "support_accuracy_bagged": base.stats(
                audit["support_accuracy_bagged"] for audit in audits
            ),
            "coefficient_dispersion_rms": base.stats(
                audit["coefficient_dispersion_rms"] for audit in audits
            ),
            "coefficient_dispersion_max_row_l2": base.stats(
                audit["coefficient_dispersion_max_row_l2"] for audit in audits
            ),
            "intercept_dispersion_rms": base.stats(
                audit["intercept_dispersion_rms"] for audit in audits
            ),
            "class_common_coefficient_center_max_abs": base.stats(
                audit["class_common_coefficient_center_max_abs"]
                for audit in audits
            ),
            "class_common_intercept_center_abs": base.stats(
                audit["class_common_intercept_center_abs"] for audit in audits
            ),
            "inner_d62_boundary_status_counts": _counts(
                item["d62_boundary_status"] for item in inner
            ),
            "inner_d62_accept_count": base.stats(
                item["d62_accept_count"] for item in inner
            ),
            "ground_component_input_count": 0,
            "uses_outer_held_or_query_for_fit": False,
            "role_class_scene_or_query_branch": False,
            "single_affine_state_only": True,
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d72_inner_d62_fit_count",
        "d72_inner_component_fit_count",
        "d72_inner_lda_fit_macs",
        "d72_inner_fisher_dense_mac_upper_bound",
        "d72_gate_scalar_mac_equivalents",
        "d72_mean_scalar_mac_equivalents",
        "d72_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "d72_query_extra_mac_equivalents",
        "d72_persistent_state_extra_bytes",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "d72_ground_component_input_count",
        "d72_dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d72_single_affine_state_only",
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
    names = ("d72", "d71", "d70", "d69", "d68", "d67", "d66", "d65", "d62")
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
    fp32 = [row for row in logs["D72"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d72.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D72"]),
        "D72_INT8": {
            "aggregate": base.aggregate(target["D72"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D72"]).items()
            },
            "classes": base.class_summary(target["D72"]),
            "outer_rows": base.detailed_rows(target["D72"]),
            "mechanism": mechanism(target["D72"]),
            "training": base.training_summary(target["D72"]),
            "quantization": quant(target["D72"]),
            "resources": resources(target["D72"]),
        },
        "D72_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D71", "D70", "D69", "D68", "D67", "D66", "D65", "D62"):
        comparison = base.matched_delta(target["D72"], target[baseline])
        result[f"D72_vs_{baseline}"] = {
            "D72": comparison.pop("D49"),
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
