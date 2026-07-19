#!/usr/bin/env python3
"""Create the complete D68 signed-calibration performance ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d68_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _flatten(values: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for group in values for value in group]


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        orientation = _flatten(audit["d68_orientation_by_class"] for audit in audits)
        recomputed = _flatten(
            audit["d68_recomputed_orientation_by_class"] for audit in audits
        )
        raw_risk = _flatten(audit["d68_crossfit_risk_raw"] for audit in audits)
        signed_risk = _flatten(
            audit["d68_crossfit_risk_signed"] for audit in audits
        )
        agreement = _flatten(
            audit["d68_orientation_fold_agreement_count"] for audit in audits
        )
        result[phase] = {
            "fit_count": len(audits),
            "orientation_negative_count": int(sum(value < 0.0 for value in orientation)),
            "recomputed_orientation_negative_count": int(
                sum(value < 0.0 for value in recomputed)
            ),
            "orientation_changed_by_freeze_count": int(
                sum(left != right for left, right in zip(orientation, recomputed))
            ),
            "crossfit_delta": base.stats(
                _flatten(audit["d68_crossfit_delta"] for audit in audits)
            ),
            "fold_agreement_count": base.stats(agreement),
            "risk_raw": base.stats(raw_risk),
            "risk_signed": base.stats(signed_risk),
            "risk_mean_delta": float(
                sum(signed_risk) / len(signed_risk)
                - sum(raw_risk) / len(raw_risk)
            ),
            "compiled_support_accuracy": base.stats(
                audit["d68_compiled_support_accuracy"] for audit in audits
            ),
            "compile_float32_error_max": base.stats(
                audit["d68_compile_float32_error_max"] for audit in audits
            ),
            "crossfit_fold_count": base.stats(
                audit["d68_crossfit_fold_count"] for audit in audits
            ),
            "partition_count": int(
                sum(len(audit["d68_crossfit_partition_audit"]) for audit in audits)
            ),
            "old_rows_bitwise_unchanged": all(
                audit["d68_old_row_fp32_bitwise_unchanged"] for audit in audits
            ),
            "common_affine_sha256_unique_count": len(
                {audit["d68_stage2b_common_affine_sha256"] for audit in audits}
            ),
            "ground_component_input_count": base.stats(
                audit["d68_ground_component_input_count"] for audit in audits
            ),
            "uses_outer_held_or_query": any(
                audit["d68_uses_outer_held_or_query"] for audit in audits
            ),
            "role_class_scene_or_query_branch": any(
                audit["d68_old_new_role_specific_query_branch"]
                or audit["d68_class_id_specific_formula"]
                or audit["d68_scene_receiver_handle_specific_branch"]
                or audit["d68_query_joint_optimization"]
                for audit in audits
            ),
            "single_affine_state_only": all(
                audit["d68_single_affine_state_only"] for audit in audits
            ),
        }
    return result


def resources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d68_crossfit_fold_count_per_stage",
        "d68_full_d65_covariance_fit_count",
        "d68_inner_d65_covariance_fit_count",
        "d68_inner_d65_total_adaptation_macs",
        "d68_calibration_scalar_macs",
        "d68_total_added_adaptation_macs",
        "estimated_metric_adaptation_macs",
        "estimated_adaptation_macs",
        "estimated_macs_per_query",
        "trainable_parameters",
        "persistent_state_bytes",
        "d68_persistent_state_extra_bytes",
        "registry_state_bytes",
        "peak_cuda_memory_bytes",
        "adaptation_epochs",
        "optimizer_steps",
        "d68_optimizer_steps_extra",
    )
    first = rows[0]["resource"]
    invariants = (
        "runtime_device",
        "deployment_precision",
        "coefficient_dtype",
        "intercept_dtype",
        "d68_ground_component_input_count",
        "d68_query_extra_macs",
        "d68_resource_single_affine_state_only",
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
    names = ("d68", "d67", "d66", "d65", "d64", "d63", "d62", "d61", "d46")
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
    fp32 = [row for row in logs["D68"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d68.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D68"]),
        "D68_INT8": {
            "aggregate": base.aggregate(target["D68"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D68"]).items()
            },
            "classes": base.class_summary(target["D68"]),
            "outer_rows": base.detailed_rows(target["D68"]),
            "mechanism": mechanism(target["D68"]),
            "training": base.training_summary(target["D68"]),
            "quantization": quant(target["D68"]),
            "resources": resources(target["D68"]),
        },
        "D68_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D67", "D66", "D65", "D64", "D63", "D62", "D61", "D46"):
        comparison = base.matched_delta(target["D68"], target[baseline])
        result[f"D68_vs_{baseline}"] = {
            "D68": comparison.pop("D49"),
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
