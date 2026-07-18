#!/usr/bin/env python3
"""Summarize every D49 development row and its matched D45 baseline.

The script intentionally reads the complete JSONL surfaces.  It emits a
machine-readable JSON artifact used to build the experiment report; it does
not select a candidate or alter any experiment state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


TARGET = "D42-USLDA-INT8"
MATCHED_FP32 = "D42-USLDA-FP32-MATCHED"
SCENE_ORDER = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return fmean(materialized) if materialized else math.nan


def stats(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    return {
        "min": min(materialized),
        "mean": mean(materialized),
        "max": max(materialized),
    }


def per_class_means(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for handle, value in row[metric]["per_class_accuracy"].items():
            values[handle].append(float(value))
    return {handle: mean(items) for handle, items in sorted(values.items())}


def confusion(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "final_argmax_old_to_new_count",
        "final_argmax_new_to_old_count",
        "final_argmax_new_to_new_count",
    )
    return {key: sum(int(row.get(key, 0)) for row in rows) for key in keys}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before_pc = per_class_means(rows, "before_old")
    after_pc = per_class_means(rows, "after_old")
    new_pc = per_class_means(rows, "after_new")
    return {
        "row_count": len(rows),
        "before_old": mean(row["before_old"]["overall_accuracy"] for row in rows),
        "after_old": mean(row["after_old"]["overall_accuracy"] for row in rows),
        "seen_new": mean(row["after_new"]["overall_accuracy"] for row in rows),
        "H_old_new_same_row_mean": mean(row["H_old_new"] for row in rows),
        "forgetting_same_row_mean": mean(row["forgetting"] for row in rows),
        "joint_floor_same_row_mean": mean(row["joint_floor"] for row in rows),
        "min_before_class_mean": min(before_pc.values()),
        "min_after_class_mean": min(after_pc.values()),
        "min_new_class_mean": min(new_pc.values()),
        "mean_row_before_floor": mean(
            row["before_old"]["class_floor_accuracy"] for row in rows
        ),
        "mean_row_after_floor": mean(
            row["after_old"]["class_floor_accuracy"] for row in rows
        ),
        "mean_row_new_floor": mean(
            row["after_new"]["class_floor_accuracy"] for row in rows
        ),
        "confusion": confusion(rows),
    }


def scene_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    return {scene: grouped[scene] for scene in SCENE_ORDER if scene in grouped}


def candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    return {
        candidate: aggregate(candidate_rows)
        for candidate, candidate_rows in sorted(grouped.items())
    }


def detailed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in sorted(rows, key=lambda item: (SCENE_ORDER.index(item["scenario"]), item["fold_index"])):
        result.append(
            {
                "scenario": row["scenario"],
                "fold_index": row["fold_index"],
                "before_old": row["before_old"]["overall_accuracy"],
                "after_old": row["after_old"]["overall_accuracy"],
                "seen_new": row["after_new"]["overall_accuracy"],
                "H_old_new": row["H_old_new"],
                "forgetting": row["forgetting"],
                "joint_floor": row["joint_floor"],
                "before_floor": row["before_old"]["class_floor_accuracy"],
                "after_floor": row["after_old"]["class_floor_accuracy"],
                "new_floor": row["after_new"]["class_floor_accuracy"],
                "old_to_new": row["final_argmax_old_to_new_count"],
                "new_to_old": row["final_argmax_new_to_old_count"],
                "new_to_new": row["final_argmax_new_to_new_count"],
                "prediction_sha256": row["outer_prediction_sha256"],
            }
        )
    return result


def class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "aggregate": {
            "before_old": per_class_means(rows, "before_old"),
            "after_old": per_class_means(rows, "after_old"),
            "seen_new": per_class_means(rows, "after_new"),
        },
        "by_scene": {},
    }
    for scene, group in scene_groups(rows).items():
        result["by_scene"][scene] = {
            "before_old": per_class_means(group, "before_old"),
            "after_old": per_class_means(group, "after_old"),
            "seen_new": per_class_means(group, "after_new"),
        }
    return result


def matched_delta(
    d49_rows: list[dict[str, Any]], d45_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    d49_by_key = {(row["scenario"], row["fold_index"]): row for row in d49_rows}
    d45_by_key = {(row["scenario"], row["fold_index"]): row for row in d45_rows}
    if d49_by_key.keys() != d45_by_key.keys():
        raise ValueError("D49 and D45 matched-row keys differ")

    def delta_row(new: dict[str, Any], old: dict[str, Any]) -> dict[str, float]:
        return {
            "before_old": new["before_old"]["overall_accuracy"]
            - old["before_old"]["overall_accuracy"],
            "after_old": new["after_old"]["overall_accuracy"]
            - old["after_old"]["overall_accuracy"],
            "seen_new": new["after_new"]["overall_accuracy"]
            - old["after_new"]["overall_accuracy"],
            "H_old_new": new["H_old_new"] - old["H_old_new"],
            "forgetting": new["forgetting"] - old["forgetting"],
            "joint_floor": new["joint_floor"] - old["joint_floor"],
            "before_floor": new["before_old"]["class_floor_accuracy"]
            - old["before_old"]["class_floor_accuracy"],
            "after_floor": new["after_old"]["class_floor_accuracy"]
            - old["after_old"]["class_floor_accuracy"],
            "new_floor": new["after_new"]["class_floor_accuracy"]
            - old["after_new"]["class_floor_accuracy"],
        }

    row_deltas = []
    for key in sorted(d49_by_key, key=lambda item: (SCENE_ORDER.index(item[0]), item[1])):
        new, old = d49_by_key[key], d45_by_key[key]
        delta = delta_row(new, old)
        row_deltas.append(
            {
                "scenario": key[0],
                "fold_index": key[1],
                "prediction_hash_changed": new["outer_prediction_sha256"]
                != old["outer_prediction_sha256"],
                **delta,
            }
        )

    def summarize_delta(items: list[dict[str, Any]]) -> dict[str, float]:
        metric_keys = (
            "before_old",
            "after_old",
            "seen_new",
            "H_old_new",
            "forgetting",
            "joint_floor",
            "before_floor",
            "after_floor",
            "new_floor",
        )
        return {key: mean(item[key] for item in items) for key in metric_keys}

    by_scene = {
        scene: summarize_delta([row for row in row_deltas if row["scenario"] == scene])
        for scene in SCENE_ORDER
    }
    aggregate_d49 = aggregate(d49_rows)
    aggregate_d45 = aggregate(d45_rows)
    class_floor_delta = {
        "min_before_class_mean": aggregate_d49["min_before_class_mean"]
        - aggregate_d45["min_before_class_mean"],
        "min_after_class_mean": aggregate_d49["min_after_class_mean"]
        - aggregate_d45["min_after_class_mean"],
        "min_new_class_mean": aggregate_d49["min_new_class_mean"]
        - aggregate_d45["min_new_class_mean"],
    }
    scene_class_floor_delta = {}
    d49_scene = scene_groups(d49_rows)
    d45_scene = scene_groups(d45_rows)
    for scene in SCENE_ORDER:
        new_agg, old_agg = aggregate(d49_scene[scene]), aggregate(d45_scene[scene])
        scene_class_floor_delta[scene] = {
            key: new_agg[key] - old_agg[key]
            for key in (
                "min_before_class_mean",
                "min_after_class_mean",
                "min_new_class_mean",
            )
        }
    return {
        "D49": aggregate_d49,
        "D45": aggregate_d45,
        "aggregate_mean_delta": summarize_delta(row_deltas),
        "aggregate_class_floor_delta": class_floor_delta,
        "by_scene_mean_delta": by_scene,
        "by_scene_class_floor_delta": scene_class_floor_delta,
        "changed_prediction_hash_rows": sum(
            bool(row["prediction_hash_changed"]) for row in row_deltas
        ),
        "row_deltas": row_deltas,
        "confusion_delta": {
            key: aggregate_d49["confusion"][key] - aggregate_d45["confusion"][key]
            for key in aggregate_d49["confusion"]
        },
    }


def mechanism_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in ("before", "final"):
        audits = [row["geometry_summary"][f"{phase}_covariance_audit"] for row in rows]
        result[phase] = {
            "d45_weight": stats(audit["d49_d45_weight"] for audit in audits),
            "cosine_weight": stats(audit["d49_cosine_weight"] for audit in audits),
            "d45_nested_macro_ce": stats(
                audit["d49_d45_nested_macro_class_ce"] for audit in audits
            ),
            "cosine_nested_macro_ce": stats(
                audit["d49_cosine_nested_macro_class_ce"] for audit in audits
            ),
            "d45_full_support_logit_rms": stats(
                audit["d49_d45_full_support_logit_rms"] for audit in audits
            ),
            "cosine_full_support_logit_rms": stats(
                audit["d49_cosine_full_support_logit_rms"] for audit in audits
            ),
            "prototype_resultant_norm": stats(
                value
                for audit in audits
                for value in audit["d49_cosine_prototype_resultant_norm_by_class"]
            ),
        }
    result["by_scene"] = {}
    for scene, group in scene_groups(rows).items():
        result["by_scene"][scene] = {}
        for phase in ("before", "final"):
            audits = [
                row["geometry_summary"][f"{phase}_covariance_audit"] for row in group
            ]
            result["by_scene"][scene][phase] = {
                "d45_weight_mean": mean(audit["d49_d45_weight"] for audit in audits),
                "cosine_weight_mean": mean(audit["d49_cosine_weight"] for audit in audits),
                "d45_nested_macro_ce_mean": mean(
                    audit["d49_d45_nested_macro_class_ce"] for audit in audits
                ),
                "cosine_nested_macro_ce_mean": mean(
                    audit["d49_cosine_nested_macro_class_ce"] for audit in audits
                ),
            }
    return result


def training_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in row["training_trace"]:
            by_epoch[int(item["epoch"])].append(item)
    result = {}
    for epoch, items in sorted(by_epoch.items()):
        result[str(epoch)] = {
            "loss_mean": mean(item["loss"] for item in items),
            "loss_min": min(item["loss"] for item in items),
            "loss_max": max(item["loss"] for item in items),
            "support_accuracy_mean": mean(item["support_accuracy"] for item in items),
            "gradient_norm_mean": mean(item["gradient_norm"] for item in items),
            "ce_loss_mean": mean(item["ce_loss"] for item in items),
            "prototype_anchor_loss_mean": mean(
                item["prototype_anchor_loss"] for item in items
            ),
            "query_rows_used_sum": sum(item["query_rows_used"] for item in items),
        }
    return result


def quantization_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sum_keys = (
        "matched_fp32_before_argmax_change_count",
        "matched_fp32_outer_argmax_change_count",
        "int8_fp32_margin_sign_flip_count",
        "int8_vs_fp32_before_support_argmax_change_count",
        "int8_vs_fp32_final_support_argmax_change_count",
        "d49_fp32_exact_top_tie_count",
        "d49_int8_exact_top_tie_count",
    )
    result = {
        key: sum(int(row["resource"][key]) for row in rows) for key in sum_keys
    }
    result["int8_fp32_max_score_abs_error"] = stats(
        row["int8_fp32_max_score_abs_error"] for row in rows
    )
    for key in ("old_new_margin_min", "new_old_margin_min", "new_new_margin_min"):
        result[key] = stats(row[key] for row in rows)
    return result


def resource_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = (
        "lda_closed_form_fit_count",
        "estimated_lda_fit_macs",
        "d49_extra_adaptation_macs",
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
    result = {
        key: stats(row["resource"][key] for row in rows) for key in numeric_keys
    }
    first = rows[0]["resource"]
    result["invariants"] = {
        key: first[key]
        for key in (
            "runtime_device",
            "deployment_precision",
            "coefficient_dtype",
            "intercept_dtype",
            "d49_cuda_peak_memory_measured",
            "d49_host_fp64_peak_memory_measured",
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
            "d49_k8_exact_292_lda_fit_count_pass",
        )
    }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d49-log", type=Path, required=True)
    parser.add_argument("--d45-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    d49_all = load_jsonl(args.d49_log)
    d45_all = load_jsonl(args.d45_log)
    d49 = [row for row in d49_all if row["candidate_id"] == TARGET]
    d49_fp32 = [row for row in d49_all if row["candidate_id"] == MATCHED_FP32]
    d45 = [row for row in d45_all if row["candidate_id"] == TARGET]
    if not (len(d49_all) == 105 and len(d45_all) == 105):
        raise ValueError("expected complete 105-row D49 and D45 logs")
    if not (len(d49) == len(d49_fp32) == len(d45) == 15):
        raise ValueError("expected 15 target/matched rows")

    output = {
        "schema": "cvs.phase2.d49.full_performance_summary.v1",
        "input": {
            "d49_log": str(args.d49_log),
            "d49_log_size": args.d49_log.stat().st_size,
            "d49_log_sha256": sha256(args.d49_log),
            "d45_log": str(args.d45_log),
            "d45_log_size": args.d45_log.stat().st_size,
            "d45_log_sha256": sha256(args.d45_log),
            "d49_rows_read": len(d49_all),
            "d45_rows_read": len(d45_all),
        },
        "all_candidates": candidate_summary(d49_all),
        "D49_INT8": {
            "aggregate": aggregate(d49),
            "by_scene": {
                scene: aggregate(group) for scene, group in scene_groups(d49).items()
            },
            "classes": class_summary(d49),
            "outer_rows": detailed_rows(d49),
            "mechanism": mechanism_summary(d49),
            "training": training_summary(d49),
            "quantization": quantization_summary(d49),
            "resources": resource_summary(d49),
        },
        "D49_FP32_MATCHED": {
            "aggregate": aggregate(d49_fp32),
            "by_scene": {
                scene: aggregate(group)
                for scene, group in scene_groups(d49_fp32).items()
            },
        },
        "D49_vs_D45": matched_delta(d49, d45),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
