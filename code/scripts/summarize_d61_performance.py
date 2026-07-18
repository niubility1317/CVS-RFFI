#!/usr/bin/env python3
"""Create the complete D61 performance and Fisher-residual ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D59_SUMMARY = SCRIPT_DIR / "summarize_d59_performance.py"
SPEC = importlib.util.spec_from_file_location("d61_summary_helper", D59_SUMMARY)
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
            "machine_rank": base.stats(audit["d61_machine_rank"] for audit in audits),
            "gain": base.stats(
                value for audit in audits for value in audit["d61_gain_by_mode"]
            ),
            "per_fit_gain_mean": base.stats(audit["d61_gain_mean"] for audit in audits),
            "transform_condition_number": base.stats(
                audit["d61_transform_condition_number"] for audit in audits
            ),
            "compiled_fp32_score_drift": base.stats(
                audit["d61_compiled_affine_score_drift_max"] for audit in audits
            ),
            "compiled_relative_score_drift": base.stats(
                audit["d61_compiled_affine_relative_score_drift_max"] for audit in audits
            ),
            "unique_transform_sha256": len(
                {audit["d61_transform_sha256"] for audit in audits}
            ),
            "identity_primary_all": all(audit["d61_identity_primary"] for audit in audits),
            "covariance_coordinates_unchanged_all": all(
                audit["d61_covariance_coordinates_unchanged"] for audit in audits
            ),
        }
    result["by_scene"] = {
        scene: {
            phase: {
                "rank_mean": base.mean(
                    row["geometry_summary"][f"{phase}_covariance_audit"]["d61_machine_rank"]
                    for row in group
                ),
                "gain_mean": base.mean(
                    row["geometry_summary"][f"{phase}_covariance_audit"]["d61_gain_mean"]
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
        "d61_component_transform_fit_count",
        "d61_dense_algebra_mac_equivalent_upper_bound",
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
                "d61_resource_single_affine_state_only",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("d61", "d46", "d59", "d60", "d58"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name.upper(): getattr(args, f"{name}-log".replace("-", "_"))
        for name in ("d61", "d46", "d59", "d60", "d58")
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D61"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d61.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D61"]),
        "D61_INT8": {
            "aggregate": base.aggregate(target["D61"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D61"]).items()
            },
            "classes": base.class_summary(target["D61"]),
            "outer_rows": base.detailed_rows(target["D61"]),
            "mechanism": mechanism(target["D61"]),
            "training": base.training_summary(target["D61"]),
            "quantization": quant(target["D61"]),
            "resources": resources(target["D61"]),
        },
        "D61_FP32_MATCHED": {"aggregate": base.aggregate(fp32)},
    }
    for baseline in ("D46", "D59", "D60", "D58"):
        result[f"D61_vs_{baseline}"] = base.matched_delta(
            target["D61"], target[baseline]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
