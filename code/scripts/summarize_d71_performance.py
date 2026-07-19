#!/usr/bin/env python3
"""Create the complete D71 top-2 centroid-reranker performance ledger."""

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
SPEC = importlib.util.spec_from_file_location("d71_summary_helper", D62_SUMMARY)
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
            row["geometry_summary"][f"d71_{phase}_reranker_audit"]
            for row in rows
        ]
        active = [
            {
                "scenario": row["scenario"],
                "fold_index": int(row["fold_index"]),
                "gate_status": audit["gate_status"],
                "accepted_pairs": audit["accepted_pairs"],
                "base_positive": audit["base_positive"],
                "base_false_positive": audit["base_false_positive"],
                "joint_positive": audit["joint_positive"],
                "joint_false_positive": audit["joint_false_positive"],
                "int8_pair_state_bytes": audit["int8_pair_state_bytes"],
            }
            for row, audit in zip(rows, audits)
            if int(audit["active_pair_count"]) > 0
        ]
        pair_deltas = [
            int(candidate) - int(original)
            for audit in audits
            for original_pair, candidate_pair in zip(
                audit["pair_base_correct"], audit["pair_candidate_correct"]
            )
            for original, candidate in zip(original_pair, candidate_pair)
        ]
        result[phase] = {
            "fit_count": len(audits),
            "gate_status_counts": _counts(audit["gate_status"] for audit in audits),
            "active_fit_count": int(
                sum(int(audit["active_pair_count"]) > 0 for audit in audits)
            ),
            "accepted_pair_count": int(
                sum(int(audit["active_pair_count"]) for audit in audits)
            ),
            "accepted_pair_count_per_fit": base.stats(
                audit["active_pair_count"] for audit in audits
            ),
            "initial_pair_count_per_fit": base.stats(
                sum(audit["initial_accept_mask"]) for audit in audits
            ),
            "pair_restricted_correct_delta": base.stats(pair_deltas),
            "active_rows": active,
            "partition_count": int(
                sum(len(audit["partition_audit"]) for audit in audits)
            ),
            "partition_exact_once": all(
                audit["partition_exact_once"] for audit in audits
            ),
            "direction_quantization_error_max": base.stats(
                audit["direction_quantization_error_max"] for audit in audits
            ),
            "bias_quantization_error_max": base.stats(
                audit["bias_quantization_error_max"] for audit in audits
            ),
            "int8_pair_state_bytes": base.stats(
                audit["int8_pair_state_bytes"] for audit in audits
            ),
            "ground_component_input_count": 0,
            "uses_outer_held_or_query_for_fit": False,
            "role_class_scene_or_query_branch": False,
            "top2_only_no_third_class_introduction": True,
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d62_additional_component_fit_count",
        "d71_inner_d62_fit_count",
        "d71_inner_component_fit_count",
        "d71_inner_lda_fit_macs",
        "d71_inner_fisher_dense_mac_upper_bound",
        "d71_pair_fit_and_score_scalar_mac_equivalents",
        "d71_held_base_score_macs",
        "d71_gate_scalar_mac_equivalents",
        "d71_total_added_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "d71_query_extra_mac_equivalents",
        "d71_int8_pair_state_extra_bytes",
        "d71_combined_int8_persistent_state_bytes",
        "d71_before_active_pair_count",
        "d71_final_active_pair_count",
        "trainable_parameters",
        "persistent_state_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "d71_ground_component_input_count",
        "d71_dense_query_graph_bytes",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d71_top2_only",
        "d71_single_affine_state_only",
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
    names = ("d71", "d70", "d69", "d68", "d67", "d66", "d65", "d62")
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
    fp32 = [row for row in logs["D71"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d71.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D71"]),
        "D71_INT8": {
            "aggregate": base.aggregate(target["D71"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D71"]).items()
            },
            "classes": base.class_summary(target["D71"]),
            "outer_rows": base.detailed_rows(target["D71"]),
            "mechanism": mechanism(target["D71"]),
            "training": base.training_summary(target["D71"]),
            "quantization": quant(target["D71"]),
            "resources": resources(target["D71"]),
        },
        "D71_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D70", "D69", "D68", "D67", "D66", "D65", "D62"):
        comparison = base.matched_delta(target["D71"], target[baseline])
        result[f"D71_vs_{baseline}"] = {
            "D71": comparison.pop("D49"),
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
