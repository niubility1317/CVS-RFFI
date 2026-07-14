"""Audit and summarize a completed extreme-light Stage2-C matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
REQUIRED_ARTIFACTS = (
    "metrics.json",
    "split_manifest.json",
    "resolved_config.json",
    "score_table.csv",
    "detailed_metrics.json",
    "detailed_metrics.csv",
    "loss_trace.json",
    "loss_trace.csv",
    "row_manifest.json",
)
METRICS = (
    "old_acc",
    "seen_new_acc",
    "H_old_new",
    "min_old_class_acc",
    "min_seen_new_class_acc",
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"metric group must be finite and non-empty: {values}")
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    half = 1.96 * std / math.sqrt(count) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": min(values),
        "max": max(values),
    }


def _iter_groups(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> Iterable[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))


def _finite_trace(trace: Sequence[dict[str, Any]], path: Path) -> None:
    for row in trace:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"non-finite {key} in {path}: {row}")


def _int_value(value: Any) -> int:
    return 0 if value is None else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.run_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    thresholds = manifest["success_thresholds"]
    expected_rows = int(manifest["row_count"])
    metrics_paths = sorted(args.run_root.rglob("metrics.json"))
    if len(metrics_paths) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} complete rows, found {len(metrics_paths)}")

    per_run: list[dict[str, Any]] = []
    per_scenario: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    permission_violations: list[str] = []
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"missing artifacts in {run_dir}: {missing}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        trace = json.loads((run_dir / "loss_trace.json").read_text(encoding="utf-8"))
        _finite_trace(trace, run_dir / "loss_trace.json")
        arm = str(config["matrix_arm"])
        count = int(config["matrix_new_class_count"])
        receiver = str(config["target_receiver_labels"][0])
        seed = int(config["seed"])
        k_shot = int(config["k_shot"])
        metric = payload["metrics"]
        scenario_payload = payload["metrics_by_scenario"]
        if tuple(scenario_payload) != SCENARIOS:
            raise ValueError(f"scenario mismatch in {metrics_path}: {tuple(scenario_payload)}")
        checks = {
            "support_query_overlap": split.get("support_query_overlap") is False,
            "target_query_used_for_training": split.get("target_query_used_for_training") is False,
            "target_query_used_for_model_selection": split.get("target_query_used_for_model_selection") is False,
            "query_used_for_transductive_inference": split.get("query_used_for_transductive_inference") is False,
            "query_used_for_joint_decision": split.get("query_used_for_joint_decision") is False,
            "non_deployment_oracle_diagnostic": split.get("non_deployment_oracle_diagnostic") is False,
            "per_sample_argmax": split.get("qknnv42_decision_mode") == "per_sample_argmax",
        }
        for key, passed in checks.items():
            if not passed:
                permission_violations.append(f"{run_dir}:{key}")
        new_threshold = float(thresholds[f"seen_new_acc_{count}"])
        row = {
            "experiment_id": str(payload["experiment_id"]),
            "arm": arm,
            "new_class_count": count,
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
            "old_acc": float(metric["old_acc_mean"]),
            "seen_new_acc": float(metric["seen_new_acc_mean"]),
            "H_old_new": float(metric["H_old_new_mean"]),
            "min_old_class_acc": float(metric["min_old_class_acc"]),
            "min_seen_new_class_acc": float(metric["min_seen_new_class_acc"]),
            "old_threshold": float(thresholds["old_acc"]),
            "min_old_threshold": float(thresholds["min_old_class_acc"]),
            "seen_new_threshold": new_threshold,
            "old_pass": float(metric["old_acc_mean"]) >= float(thresholds["old_acc"]),
            "min_old_pass": float(metric["min_old_class_acc"]) >= float(thresholds["min_old_class_acc"]),
            "seen_new_pass": float(metric["seen_new_acc_mean"]) >= new_threshold,
            "run_dir": str(run_dir),
        }
        row["joint_pass"] = bool(row["old_pass"] and row["min_old_pass"] and row["seen_new_pass"])
        scenario_resources = list(scenario_payload.values())
        macs_per_query = []
        for scenario, item in scenario_payload.items():
            direct = item.get("estimated_macs_per_query")
            if direct is not None:
                macs_per_query.append(int(direct))
            else:
                query_count = int(split["splits_by_scenario"][scenario]["query_count"])
                macs_per_query.append(int(math.ceil(float(item.get("estimated_head_macs", 0)) / max(1, query_count))))
        row.update(
            {
                "trainable_parameters_max": max(_int_value(item.get("trainable_parameters")) for item in scenario_resources),
                "persistent_state_bytes_max": max(_int_value(item.get("persistent_state_bytes_with_post_adapter", item.get("persistent_state_bytes"))) for item in scenario_resources),
                "peak_device_memory_bytes_max": max(_int_value(item.get("peak_device_memory_bytes")) for item in scenario_resources),
                "estimated_macs_per_query_max": max(macs_per_query),
                "adaptation_latency_sec_mean": statistics.fmean(float(item.get("adaptation_latency_sec", 0.0)) for item in scenario_resources),
                "onboard_scoring_latency_per_query_ms_mean": statistics.fmean(float(item.get("onboard_scoring_latency_per_query_ms", 0.0)) for item in scenario_resources),
                "dense_graph_bytes_max": max(int(item.get("dense_graph_peak_bytes_lower_bound", 0)) for item in scenario_resources),
                "query_features_used_for_adaptation": any(bool(item.get("query_features_used_for_adaptation", item.get("feature_adapter_uses_query", False))) for item in scenario_resources),
                "role_oracle_used": any(bool(item.get("role_oracle_used", False)) for item in scenario_resources),
                "equal_class_quota_used": any(bool(item.get("equal_class_quota_used", False)) for item in scenario_resources),
            }
        )
        per_run.append(row)
        for scenario, values in scenario_payload.items():
            per_scenario.append(
                {
                    "arm": arm,
                    "new_class_count": count,
                    "receiver": receiver,
                    "seed": seed,
                    "k_shot": k_shot,
                    "scenario": scenario,
                    "old_acc": float(values["old_acc"]),
                    "seen_new_acc": float(values["seen_new_acc"]),
                    "H_old_new": float(values["H_old_new"]),
                }
            )
        phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in trace:
            phases[str(item.get("scenario", "unknown"))].append(item)
        final_losses: list[float] = []
        initial_losses: list[float] = []
        for scenario, items in phases.items():
            initial_losses.append(float(items[0]["loss"]))
            final_losses.append(float(items[-1]["loss"]))
        loss_rows.append(
            {
                "arm": arm,
                "new_class_count": count,
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "trace_rows": len(trace),
                "loss_initial_mean": statistics.fmean(initial_losses),
                "loss_final_mean": statistics.fmean(final_losses),
                "loss_decreased_all_scenarios": all(final < initial for initial, final in zip(initial_losses, final_losses)),
            }
        )

    if permission_violations:
        raise RuntimeError(f"permission/protocol violations: {permission_violations}")
    if len({row["experiment_id"] for row in per_run}) != expected_rows:
        raise RuntimeError("experiment IDs are not unique")

    baseline = {
        (row["new_class_count"], row["receiver"], row["seed"], row["k_shot"]): row
        for row in per_run
        if row["arm"] == "baseline_single_qknn"
    }
    paired: list[dict[str, Any]] = []
    if baseline:
        for row in per_run:
            if row["arm"] == "baseline_single_qknn":
                continue
            key = (row["new_class_count"], row["receiver"], row["seed"], row["k_shot"])
            if key not in baseline:
                raise RuntimeError(f"missing paired baseline cell: {key}")
            reference = baseline[key]
            item = {field: row[field] for field in ("arm", "new_class_count", "receiver", "seed", "k_shot")}
            for metric in METRICS:
                item[f"baseline_{metric}"] = reference[metric]
                item[f"candidate_{metric}"] = row[metric]
                item[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
            paired.append(item)

    group_fields = ("arm", "new_class_count", "k_shot")
    group_summary: list[dict[str, Any]] = []
    for group, items in _iter_groups(per_run, group_fields):
        summary = dict(zip(group_fields, group))
        for metric in METRICS:
            summary.update({f"{metric}_{key}": value for key, value in _stats([float(item[metric]) for item in items]).items()})
        summary.update(
            {
                "joint_pass_count": sum(bool(item["joint_pass"]) for item in items),
                "old_pass_count": sum(bool(item["old_pass"]) for item in items),
                "min_old_pass_count": sum(bool(item["min_old_pass"]) for item in items),
                "seen_new_pass_count": sum(bool(item["seen_new_pass"]) for item in items),
                "trainable_parameters_max": max(int(item["trainable_parameters_max"]) for item in items),
                "persistent_state_bytes_max": max(int(item["persistent_state_bytes_max"]) for item in items),
                "peak_device_memory_bytes_max": max(int(item["peak_device_memory_bytes_max"]) for item in items),
                "estimated_macs_per_query_max": max(int(item["estimated_macs_per_query_max"]) for item in items),
                "adaptation_latency_sec_mean": statistics.fmean(float(item["adaptation_latency_sec_mean"]) for item in items),
                "onboard_scoring_latency_per_query_ms_mean": statistics.fmean(float(item["onboard_scoring_latency_per_query_ms_mean"]) for item in items),
                "dense_graph_bytes_max": max(int(item["dense_graph_bytes_max"]) for item in items),
            }
        )
        group_summary.append(summary)

    row_logs = sorted(args.log_root.rglob("k_*.log"))
    complete_markers = 0
    failed_markers = 0
    trace_markers = 0
    for path in row_logs:
        text = path.read_text(encoding="utf-8")
        complete_markers += text.count("[ROW-COMPLETE]")
        failed_markers += text.count("[ROW-FAILED]")
        trace_markers += text.count("[LOSS-TRACE]")
    if len(row_logs) != expected_rows or complete_markers != expected_rows or failed_markers:
        raise RuntimeError(
            f"log audit failed: logs={len(row_logs)} complete={complete_markers} failed={failed_markers}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_run_results.csv", per_run)
    _write_csv(args.output_dir / "per_scenario_results.csv", per_scenario)
    _write_csv(args.output_dir / "group_summary.csv", group_summary)
    _write_csv(args.output_dir / "paired_deltas_vs_baseline.csv", paired)
    _write_csv(args.output_dir / "loss_summary.csv", loss_rows)
    audit = {
        "schema": "cvs_stage2c_extreme_light_summary_v1",
        "mode": manifest["mode"],
        "expected_rows": expected_rows,
        "complete_rows": len(per_run),
        "scenario_rows": len(per_scenario),
        "row_logs": len(row_logs),
        "row_complete_markers": complete_markers,
        "row_failed_markers": failed_markers,
        "loss_trace_markers": trace_markers,
        "permission_violation_count": len(permission_violations),
        "joint_pass_count": sum(bool(row["joint_pass"]) for row in per_run),
        "paired_baseline_present": bool(baseline),
        "paired_delta_rows": len(paired),
        "loss_nonfinite_count": 0,
        "support_query_overlap_count": 0,
        "query_training_or_model_selection_count": 0,
        "thresholds": thresholds,
    }
    (args.output_dir / "final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
