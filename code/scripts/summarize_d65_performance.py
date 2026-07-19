#!/usr/bin/env python3
"""Create the complete D65 append-only performance and lifecycle ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d65_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    covariance_pair_match = 0
    for row in rows:
        before = row["geometry_summary"]["before_covariance_audit"]
        final = row["geometry_summary"]["final_covariance_audit"]
        covariance_pair_match += (
            before["d65_stage2b_covariance_sha256"]
            == final["d65_stage2b_covariance_sha256"]
        )
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "fit_count": len(audits),
            "phase_count": dict(Counter(audit["d65_phase"] for audit in audits)),
            "covariance_sha256_unique_count": len(
                {audit["d65_stage2b_covariance_sha256"] for audit in audits}
            ),
            "covariance_condition_number": base.stats(
                audit["d65_stage2b_covariance_condition_number"] for audit in audits
            ),
            "residual_rank": base.stats(
                audit["d65_stage2b_residual_rank"] for audit in audits
            ),
            "residual_energy": base.stats(
                audit["d65_stage2b_residual_energy"] for audit in audits
            ),
            "compiled_support_accuracy": base.stats(
                audit["d65_compiled_support_accuracy"] for audit in audits
            ),
            "unit_covariance_fallback_count": sum(
                bool(audit["unit_covariance_fallback"]) for audit in audits
            ),
            "appended_class_count": base.stats(
                audit["d65_appended_class_count"] for audit in audits
            ),
            "old_row_fp32_bitwise_unchanged": all(
                audit["d65_old_row_fp32_bitwise_unchanged"] for audit in audits
            ),
            "single_affine_state_only": all(
                audit["d65_single_affine_state_only"] for audit in audits
            ),
            "role_or_scene_specific_branch": any(
                audit["d65_old_new_role_specific_query_branch"]
                or audit["d65_scene_receiver_handle_specific_branch"]
                or audit["d65_class_id_specific_formula"]
                for audit in audits
            ),
        }
    result["before_final_covariance_sha256_match_count"] = covariance_pair_match
    result["by_scene"] = {
        scene: {
            phase: {
                "fit_count": len(group),
                "compiled_support_accuracy": base.stats(
                    row["geometry_summary"][f"{phase}_covariance_audit"][
                        "d65_compiled_support_accuracy"
                    ]
                    for row in group
                ),
                "covariance_condition_number": base.stats(
                    row["geometry_summary"][f"{phase}_covariance_audit"][
                        "d65_stage2b_covariance_condition_number"
                    ]
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
        "d65_stage2b_covariance_fit_count",
        "d65_append_row_count",
        "d65_append_row_macs",
        "estimated_metric_adaptation_macs",
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
    invariant_keys = (
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
        "d65_int8_old_rows_bitwise_unchanged",
        "d65_fp32_old_rows_bitwise_unchanged",
        "d65_query_extra_macs",
        "d65_persistent_state_extra_bytes",
        "d65_optimizer_steps_extra",
        "d65_resource_single_affine_state_only",
    )
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in invariant_keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("d65", "d64", "d63", "d62", "d61", "d46"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name.upper(): getattr(args, f"{name}_log")
        for name in ("d65", "d64", "d63", "d62", "d61", "d46")
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D65"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d65.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D65"]),
        "D65_INT8": {
            "aggregate": base.aggregate(target["D65"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D65"]).items()
            },
            "classes": base.class_summary(target["D65"]),
            "outer_rows": base.detailed_rows(target["D65"]),
            "mechanism": mechanism(target["D65"]),
            "training": base.training_summary(target["D65"]),
            "quantization": quant(target["D65"]),
            "resources": resources(target["D65"]),
        },
        "D65_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D64", "D63", "D62", "D61", "D46"):
        result[f"D65_vs_{baseline}"] = base.matched_delta(
            target["D65"], target[baseline]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
