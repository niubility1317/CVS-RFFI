#!/usr/bin/env python3
"""Audit and aggregate the strict full-feature qKNNV42 non-Oracle 125-task run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PRIMARY_METRICS = ("old_acc", "seen_new_acc", "H_old_new")
EXPECTED_K = (1, 2, 5, 10, 20)
EXPECTED_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_FILES = {
    "detailed_metrics.csv",
    "detailed_metrics.json",
    "loss_trace.csv",
    "loss_trace.json",
    "metrics.json",
    "resolved_config.json",
    "score_table.csv",
    "split_manifest.json",
}


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else math.nan


def ci95(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_k(path: Path):
    match = re.search(r"[\\/]k_(\d+)[\\/]", str(path))
    if not match:
        raise RuntimeError(f"cannot infer K from {path}")
    return int(match.group(1))


def summarize(rows, fields, metrics=PRIMARY_METRICS + ("average_forgetting",)):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = {field: value for field, value in zip(fields, key)}
        out["count"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = statistics.pstdev(values)
            out[f"{metric}_ci95"] = ci95(values)
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)
        output.append(out)
    return output


def parse_current(root: Path):
    runs = []
    scenarios = []
    per_class = []
    metric_payloads = []
    for path in sorted(root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_payloads.append((path, payload))
        metrics = payload["metrics"]
        row = {
            "arm": "full_nonoracle_strict",
            "experiment_id": payload["experiment_id"],
            "receiver": payload["target_receiver_label"],
            "seed": int(payload["seed"]),
            "k_shot": infer_k(path),
            "old_acc": float(metrics["old_acc_mean"]),
            "seen_new_acc": float(metrics["seen_new_acc_mean"]),
            "H_old_new": float(metrics["H_old_new_mean"]),
            "average_forgetting": float(metrics["average_forgetting_mean"]),
            "min_old_class_acc": float(metrics["min_old_class_acc"]),
            "min_seen_new_class_acc": float(metrics["min_seen_new_class_acc"]),
            "run_dir": str(path.parent),
        }
        row["min_old_new"] = min(row["old_acc"], row["seen_new_acc"])
        row["joint_score"] = mean(row[metric] for metric in PRIMARY_METRICS)
        runs.append(row)
        for scenario, values in sorted(payload["metrics_by_scenario"].items()):
            scenario_row = {
                "arm": row["arm"],
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "scenario": scenario,
                "old_acc": float(values["old_acc"]),
                "seen_new_acc": float(values["seen_new_acc"]),
                "H_old_new": float(values["H_old_new"]),
                "average_forgetting": float(values["average_forgetting"]),
                "old_acc_before_increment": float(values["old_acc_before_increment"]),
                "old_to_seen_new_rate": float(values["old_to_seen_new_rate"]),
                "seen_new_to_old_rate": float(values["seen_new_to_old_rate"]),
                "adaptation_latency_sec": float(values["adaptation_latency_sec"]),
                "onboard_scoring_latency_per_query_ms": float(values["onboard_scoring_latency_per_query_ms"]),
                "persistent_state_bytes_with_post_adapter": int(values["persistent_state_bytes_with_post_adapter"]),
                "dense_graph_peak_bytes_lower_bound": int(values["dense_graph_peak_bytes_lower_bound"]),
                "estimated_head_macs_with_post_adapter": int(values["estimated_head_macs_with_post_adapter"]),
                "persistent_state_bytes_total": int(values["persistent_state_bytes_with_post_adapter"])
                + int(values["aux_persistent_state_bytes"]),
                "dense_graph_peak_bytes_total_lower_bound": int(values["dense_graph_peak_bytes_lower_bound"])
                + int(values["aux_dense_graph_bytes_lower_bound"]),
                "estimated_head_macs_total": int(values["estimated_head_macs_with_post_adapter"])
                + int(values["aux_estimated_head_macs"]),
            }
            scenarios.append(scenario_row)
        for label, value in sorted(metrics["per_class_accuracy_across_scenarios"].items()):
            per_class.append({
                "arm": row["arm"],
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "class_label": label,
                "class_role": "target_new" if label in {"1-16", "1-18"} else "target_old",
                "accuracy": float(value),
            })
    if len(runs) != 125:
        raise RuntimeError(f"expected 125 current runs, got {len(runs)}")
    return runs, scenarios, per_class, metric_payloads


def paired_compare(current, historical_path: Path):
    historical = read_csv(historical_path)
    index = {}
    for row in historical:
        key = (row["arm"], row["receiver"], int(row["seed"]), int(row["k_shot"]))
        index[key] = row
    output = []
    refs = ("full_legacy_oracle_strict", "singleview_fft96_strict")
    for row in current:
        for ref_arm in refs:
            key = (ref_arm, row["receiver"], row["seed"], row["k_shot"])
            if key not in index:
                raise RuntimeError(f"missing historical pair {key}")
            ref = index[key]
            out = {
                "reference_arm": ref_arm,
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
            }
            for metric in PRIMARY_METRICS:
                current_value = float(row[metric])
                ref_value = float(ref[metric])
                out[f"current_{metric}"] = current_value
                out[f"reference_{metric}"] = ref_value
                out[f"delta_{metric}"] = current_value - ref_value
            output.append(out)
    return output


def paired_summary(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = {field: value for field, value in zip(fields, key)}
        out["count"] = len(group)
        for metric in PRIMARY_METRICS:
            values = [float(row[f"delta_{metric}"]) for row in group]
            out[f"delta_{metric}_mean"] = mean(values)
            out[f"delta_{metric}_ci95"] = ci95(values)
            out[f"{metric}_wins"] = sum(value > 1e-12 for value in values)
            out[f"{metric}_ties"] = sum(abs(value) <= 1e-12 for value in values)
            out[f"{metric}_losses"] = sum(value < -1e-12 for value in values)
        output.append(out)
    return output


def audit_feature_manifest(path: Path):
    rows = [row for row in read_csv(path) if row["arm"] == "full_legacy_oracle_strict"]
    errors = []
    if len(rows) != 5:
        errors.append(f"expected 5 full feature manifests, got {len(rows)}")
    for row in rows:
        expected = {
            "checkpoint_load_strict": "true",
            "missing_keys": "0",
            "unexpected_keys": "0",
            "skipped_mismatch": "0",
            "state_tensor_count": "195",
            "num_domains": "14",
            "view_count": "5",
            "fft_dim": "96",
        }
        for field, value in expected.items():
            if str(row[field]).lower() != value:
                errors.append(f"feature manifest {row['item']} {field}={row[field]} expected={value}")
    return rows, errors


def audit_logs(root: Path):
    patterns = {
        "traceback": re.compile(r"Traceback"),
        "runtime_error": re.compile(r"RuntimeError"),
        "value_error": re.compile(r"ValueError"),
        "oom": re.compile(r"CUDA out of memory", re.IGNORECASE),
        "killed": re.compile(r"(?:^|\s)Killed(?:\s|$)", re.MULTILINE),
        "nan_or_inf": re.compile(r"(?:^|[^A-Za-z])(?:nan|inf)(?:[^A-Za-z]|$)", re.IGNORECASE),
    }
    files = []
    hits = {name: [] for name in patterns}
    total_bytes = 0
    total_lines = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        files.append(str(path))
        total_bytes += len(raw)
        total_lines += len(text.splitlines())
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits[name].append(str(path))
    summaries = []
    for path in sorted(root.glob("worker_*_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "error_hit_counts": {name: len(paths) for name, paths in hits.items()},
        "error_hit_examples": {name: paths[:5] for name, paths in hits.items()},
        "worker_summary_count": len(summaries),
        "assigned": sum(int(row.get("assigned", 0)) for row in summaries),
        "completed": sum(int(row.get("completed", 0)) for row in summaries),
        "failed": sum(int(row.get("failed", 0)) for row in summaries),
        "skipped": sum(int(row.get("skipped", 0)) for row in summaries),
    }


def audit_run(root: Path, metric_payloads, feature_errors, log_audit):
    errors = list(feature_errors)
    counters = Counter()
    query_sets = {}
    support_sets = defaultdict(dict)
    role_oracle_values = set()
    quota_values = set()
    decision_modes = set()
    adapter_modes = set()
    labelprop_modes = set()
    expected_boolean_failures = []
    nonfinite_losses = []
    score_rows = 0
    loss_rows = 0
    file_set_failures = []

    for metrics_path, payload in metric_payloads:
        run_dir = metrics_path.parent
        k_shot = infer_k(metrics_path)
        receiver = payload["target_receiver_label"]
        seed = int(payload["seed"])
        counters[("receiver", receiver)] += 1
        counters[("seed", seed)] += 1
        counters[("k", k_shot)] += 1
        names = {path.name for path in run_dir.iterdir() if path.is_file()}
        if names != EXPECTED_FILES:
            file_set_failures.append({"run_dir": str(run_dir), "missing": sorted(EXPECTED_FILES - names), "extra": sorted(names - EXPECTED_FILES)})

        manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        checks = {
            "target_query_used_for_training": False,
            "target_query_used_for_model_selection": False,
            "query_used_for_transductive_inference": True,
            "query_used_for_joint_decision": False,
            "support_query_overlap": False,
            "satellite_tta_view_count": 5,
            "qknnv42_aux_feature_dim": 96,
            "qknnv42_decision_mode": "per_sample_argmax",
            "qknnv42_labelprop_mode": "dense_transductive",
            "qknnv42_feature_adapter_mode": "support_diag_whiten_fisher",
            "non_deployment_oracle_diagnostic": False,
        }
        for field, expected in checks.items():
            if manifest.get(field) != expected:
                expected_boolean_failures.append(f"{run_dir}:{field}={manifest.get(field)!r} expected={expected!r}")
        if tuple(sorted(manifest["splits_by_scenario"])) != tuple(sorted(EXPECTED_SCENARIOS)):
            expected_boolean_failures.append(f"{run_dir}:unexpected scenarios")
        for scenario, split in manifest["splits_by_scenario"].items():
            support = set(split["support_sample_ids"])
            query = set(split["query_sample_ids"])
            if support & query:
                expected_boolean_failures.append(f"{run_dir}:{scenario}:support/query overlap={len(support & query)}")
            key = (receiver, seed, scenario)
            if key in query_sets and query_sets[key] != query:
                expected_boolean_failures.append(f"{run_dir}:{scenario}:query set changed across K")
            query_sets[key] = query
            support_sets[key][k_shot] = support

        for scenario, values in payload["metrics_by_scenario"].items():
            role_oracle_values.add(values["role_oracle_used"])
            quota_values.add(values["equal_class_quota_used"])
            decision_modes.add(values["decision_mode"])
            adapter_modes.add(values["feature_adapter_mode"])
            labelprop_modes.add(values["labelprop_mode"])
            scenario_checks = {
                "role_oracle_used": False,
                "equal_class_quota_used": False,
                "decision_batch_state_required": False,
                "decision_workspace_bytes_lower_bound": 0,
                "estimated_decision_cubic_work_units": 0,
                "feature_adapter_uses_query": False,
                "feature_adapter_updates_adv3b02": False,
                "query_labels_used_for_adaptation": False,
                "query_query_graph_used": True,
                "query_batch_state_required": True,
                "aux_feature_enabled": True,
                "aux_feature_dim": 96,
            }
            for field, expected in scenario_checks.items():
                if values.get(field) != expected:
                    expected_boolean_failures.append(f"{run_dir}:{scenario}:{field}={values.get(field)!r} expected={expected!r}")

        with (run_dir / "loss_trace.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        loss_rows += len(rows)
        for row in rows:
            value = float(row["loss"])
            if not math.isfinite(value):
                nonfinite_losses.append(f"{run_dir}:{row}")
        with (run_dir / "score_table.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            score_rows += sum(1 for _ in csv.DictReader(handle))

    nested_failures = []
    for key, by_k in sorted(support_sets.items()):
        if tuple(sorted(by_k)) != EXPECTED_K:
            nested_failures.append(f"{key}:K={sorted(by_k)}")
            continue
        for lower, upper in zip(EXPECTED_K, EXPECTED_K[1:]):
            if not by_k[lower].issubset(by_k[upper]):
                nested_failures.append(f"{key}:K{lower} not subset K{upper}")

    if counters[("k", 1)] != 25 or any(counters[("k", k)] != 25 for k in EXPECTED_K):
        errors.append(f"K counts invalid: {dict((k, counters[('k', k)]) for k in EXPECTED_K)}")
    receiver_counts = {key[1]: value for key, value in counters.items() if key[0] == "receiver"}
    seed_counts = {str(key[1]): value for key, value in counters.items() if key[0] == "seed"}
    if sorted(receiver_counts.values()) != [25] * 5:
        errors.append(f"receiver counts invalid: {receiver_counts}")
    if sorted(seed_counts.values()) != [25] * 5:
        errors.append(f"seed counts invalid: {seed_counts}")
    if file_set_failures:
        errors.append(f"artifact file set failures={len(file_set_failures)}")
    if expected_boolean_failures:
        errors.append(f"protocol/mechanism failures={len(expected_boolean_failures)}")
    if nested_failures:
        errors.append(f"support nesting failures={len(nested_failures)}")
    if nonfinite_losses:
        errors.append(f"nonfinite losses={len(nonfinite_losses)}")
    if loss_rows != 375:
        errors.append(f"expected 375 loss rows, got {loss_rows}")
    if score_rows != 60000:
        errors.append(f"expected 60000 score rows, got {score_rows}")
    if log_audit["worker_summary_count"] != 5 or log_audit["completed"] != 125 or log_audit["failed"] != 0:
        errors.append(f"worker summaries invalid: {log_audit}")
    if any(log_audit["error_hit_counts"].values()):
        errors.append(f"log error markers: {log_audit['error_hit_counts']}")

    return {
        "artifact_complete": not errors,
        "errors": errors,
        "metrics_files": len(metric_payloads),
        "expected_files_per_run": len(EXPECTED_FILES),
        "file_set_failure_count": len(file_set_failures),
        "file_set_failure_examples": file_set_failures[:5],
        "receiver_counts": receiver_counts,
        "seed_counts": seed_counts,
        "k_counts": {str(k): counters[("k", k)] for k in EXPECTED_K},
        "scenario_metric_rows": len(metric_payloads) * len(EXPECTED_SCENARIOS),
        "role_oracle_values": sorted(role_oracle_values),
        "equal_class_quota_values": sorted(quota_values),
        "decision_modes": sorted(decision_modes),
        "feature_adapter_modes": sorted(adapter_modes),
        "labelprop_modes": sorted(labelprop_modes),
        "protocol_mechanism_failure_count": len(expected_boolean_failures),
        "protocol_mechanism_failure_examples": expected_boolean_failures[:10],
        "query_identity_groups": len(query_sets),
        "support_nesting_groups": len(support_sets),
        "support_nesting_failure_count": len(nested_failures),
        "support_nesting_failure_examples": nested_failures[:10],
        "loss_rows": loss_rows,
        "nonfinite_loss_rows": len(nonfinite_losses),
        "score_rows": score_rows,
        "log_audit": log_audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--historical-results-csv", type=Path, required=True)
    parser.add_argument("--feature-audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs, scenarios, per_class, metric_payloads = parse_current(args.run_root)
    comparisons = paired_compare(runs, args.historical_results_csv)
    feature_rows, feature_errors = audit_feature_manifest(args.feature_audit_csv)
    log_audit = audit_logs(args.log_root)
    audit = audit_run(args.run_root, metric_payloads, feature_errors, log_audit)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_run_results.csv", runs)
    write_csv(output / "per_scenario_results.csv", scenarios)
    write_csv(output / "per_class_results.csv", per_class)
    write_csv(output / "summary_overall.csv", summarize(runs, ["arm"]))
    write_csv(output / "summary_by_k.csv", summarize(runs, ["arm", "k_shot"]))
    write_csv(output / "summary_by_receiver.csv", summarize(runs, ["arm", "receiver"]))
    write_csv(output / "summary_by_scenario.csv", summarize(scenarios, ["arm", "scenario"]))
    write_csv(output / "incremental_summary_by_k.csv", summarize(
        scenarios,
        ["arm", "k_shot"],
        metrics=(
            "old_acc_before_increment",
            "old_acc",
            "seen_new_acc",
            "H_old_new",
            "average_forgetting",
            "old_to_seen_new_rate",
            "seen_new_to_old_rate",
        ),
    ))
    write_csv(output / "summary_per_class.csv", summarize(per_class, ["arm", "class_role", "class_label"], metrics=("accuracy",)))
    write_csv(output / "resource_by_k.csv", summarize(
        scenarios,
        ["arm", "k_shot"],
        metrics=(
            "adaptation_latency_sec",
            "onboard_scoring_latency_per_query_ms",
            "persistent_state_bytes_with_post_adapter",
            "dense_graph_peak_bytes_lower_bound",
            "estimated_head_macs_with_post_adapter",
            "persistent_state_bytes_total",
            "dense_graph_peak_bytes_total_lower_bound",
            "estimated_head_macs_total",
        ),
    ))
    write_csv(output / "paired_results.csv", comparisons)
    write_csv(output / "paired_summary.csv", paired_summary(comparisons, ["reference_arm"]))
    write_csv(output / "paired_summary_by_k.csv", paired_summary(comparisons, ["reference_arm", "k_shot"]))
    write_csv(output / "feature_manifest_audit.csv", feature_rows)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output),
        "artifact_complete": audit["artifact_complete"],
        "errors": audit["errors"],
        "overall": summarize(runs, ["arm"])[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
