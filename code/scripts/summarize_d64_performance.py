#!/usr/bin/env python3
"""Create the complete D64 performance and all-pairs mechanism ledger."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D62_SUMMARY = SCRIPT_DIR / "summarize_d62_performance.py"
SPEC = importlib.util.spec_from_file_location("d64_summary_helper", D62_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D62 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _pair_values(audits: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(pair[key])
        for audit in audits
        for pair in audit["d64_pair_audits"]
        if pair.get(key) is not None
    ]


def mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        pair_count = sum(int(audit["d64_pair_count"]) for audit in audits)
        pair_audits = [pair for audit in audits for pair in audit["d64_pair_audits"]]
        result[phase] = {
            "fit_count": len(audits),
            "pair_fit_count": pair_count,
            "pair_audit_count": len(pair_audits),
            "pair_count_closure_pass": pair_count == len(pair_audits),
            "pair_count_per_fit": base.stats(audit["d64_pair_count"] for audit in audits),
            "pair_margin_rms": base.stats(_pair_values(audits, "d64_pair_margin_rms")),
            "pair_support_accuracy": base.stats(
                _pair_values(audits, "d64_pair_support_accuracy")
            ),
            "pair_covariance_condition_number": base.stats(
                _pair_values(audits, "d43_covariance_condition_number")
            ),
            "pair_unit_covariance_fallback_count": sum(
                bool(pair.get("unit_covariance_fallback", False))
                for pair in pair_audits
            ),
            "compiled_support_accuracy": base.stats(
                audit["d64_compiled_support_accuracy"] for audit in audits
            ),
            "coefficient_source_count": dict(
                Counter(audit["coefficient_source"] for audit in audits)
            ),
            "single_affine_state_only": all(
                audit["d64_single_affine_state_only"] for audit in audits
            ),
            "pair_graph_persisted_for_query": any(
                audit["d64_pair_graph_persisted_for_query"] for audit in audits
            ),
            "query_joint_optimization": any(
                audit["d64_query_joint_optimization"] for audit in audits
            ),
            "role_or_scene_specific_branch": any(
                audit["d64_old_new_role_specific_branch"]
                or audit["d64_scene_receiver_handle_specific_branch"]
                or audit["d64_class_id_specific_formula"]
                for audit in audits
            ),
        }
    result["by_scene"] = {
        scene: {
            phase: {
                "fit_count": len(group),
                "pair_fit_count": sum(
                    row["geometry_summary"][f"{phase}_covariance_audit"]["d64_pair_count"]
                    for row in group
                ),
                "compiled_support_accuracy": base.stats(
                    row["geometry_summary"][f"{phase}_covariance_audit"][
                        "d64_compiled_support_accuracy"
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
        "d64_pair_fit_count",
        "d64_pair_margin_normalization_macs",
        "d64_pair_affine_compilation_macs",
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
        "d64_query_extra_macs",
        "d64_persistent_state_extra_bytes",
        "d64_optimizer_steps_extra",
        "d64_resource_single_affine_state_only",
    )
    return {
        **{key: base.stats(row["resource"][key] for row in rows) for key in keys},
        "invariants": {key: first[key] for key in invariant_keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("d64", "d63", "d62", "d61", "d46"):
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name.upper(): getattr(args, f"{name}_log")
        for name in ("d64", "d63", "d62", "d61", "d46")
    }
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D64"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d64.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "all_candidates": base.candidate_summary(logs["D64"]),
        "D64_INT8": {
            "aggregate": base.aggregate(target["D64"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D64"]).items()
            },
            "classes": base.class_summary(target["D64"]),
            "outer_rows": base.detailed_rows(target["D64"]),
            "mechanism": mechanism(target["D64"]),
            "training": base.training_summary(target["D64"]),
            "quantization": quant(target["D64"]),
            "resources": resources(target["D64"]),
        },
        "D64_FP32_MATCHED": {"aggregate": base.aggregate(fp32)},
    }
    for baseline in ("D63", "D62", "D61", "D46"):
        result[f"D64_vs_{baseline}"] = base.matched_delta(
            target["D64"], target[baseline]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
