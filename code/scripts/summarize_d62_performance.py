#!/usr/bin/env python3
"""Create the complete D62 performance and Fisher-row gate ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D61_SUMMARY = SCRIPT_DIR / "summarize_d61_performance.py"
SPEC = importlib.util.spec_from_file_location("d62_summary_helper", D61_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D61 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "fit_count": len(audits),
            "status_count": dict(Counter(audit["d62_boundary_status"] for audit in audits)),
            "active_fit_count": sum(any(audit["d62_final_accept_mask"]) for audit in audits),
            "atomic_fallback_count": sum(not audit["d62_joint_atomic_safe"] for audit in audits),
            "initial_accept_count": base.stats(
                sum(audit["d62_initial_accept_mask"]) for audit in audits
            ),
            "final_accept_count": base.stats(
                sum(audit["d62_final_accept_mask"]) for audit in audits
            ),
            "base_positive": base.stats(
                value for audit in audits for value in audit["d62_base_positive_correct_by_class"]
            ),
            "coordinate_positive_delta": base.stats(
                candidate - original
                for audit in audits
                for candidate, original in zip(
                    audit["d62_coordinate_positive_correct_by_class"],
                    audit["d62_base_positive_correct_by_class"],
                )
            ),
            "coordinate_false_positive_delta": base.stats(
                candidate - original
                for audit in audits
                for candidate, original in zip(
                    audit["d62_coordinate_false_positive_by_class"],
                    audit["d62_base_false_positive_by_class"],
                )
            ),
            "full_fisher_rank": base.stats(
                audit["d62_full_outer_transform_audit"]["d61_machine_rank"]
                for audit in audits
            ),
            "full_fisher_gain": base.stats(
                value
                for audit in audits
                for value in audit["d62_full_outer_transform_audit"]["d61_gain_by_mode"]
            ),
        }
    result["by_scene"] = {
        scene: {
            phase: {
                "active_fit_count": sum(
                    any(row["geometry_summary"][f"{phase}_covariance_audit"]["d62_final_accept_mask"])
                    for row in group
                ),
                "accepted_row_count": sum(
                    sum(row["geometry_summary"][f"{phase}_covariance_audit"]["d62_final_accept_mask"])
                    for row in group
                ),
                "atomic_fallback_count": sum(
                    not row["geometry_summary"][f"{phase}_covariance_audit"]["d62_joint_atomic_safe"]
                    for row in group
                ),
            }
            for phase in ("before", "final")
        }
        for scene, group in base.scene_groups(rows).items()
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
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
    )
    first = rows[0]["resource"]
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {
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
                "d62_resource_single_affine_state_only",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("d62", "d61", "d46", "d57", "d56"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name.upper(): getattr(args, f"{name}_log")
        for name in ("d62", "d61", "d46", "d57", "d56")
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D62"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d62.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D62"]),
        "D62_INT8": {
            "aggregate": base.aggregate(target["D62"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D62"]).items()
            },
            "classes": base.class_summary(target["D62"]),
            "outer_rows": base.detailed_rows(target["D62"]),
            "mechanism": mechanism(target["D62"]),
            "training": base.training_summary(target["D62"]),
            "quantization": quant(target["D62"]),
            "resources": resources(target["D62"]),
        },
        "D62_FP32_MATCHED": {"aggregate": base.aggregate(fp32)},
    }
    for baseline in ("D61", "D46", "D57", "D56"):
        result[f"D62_vs_{baseline}"] = base.matched_delta(
            target["D62"], target[baseline]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
