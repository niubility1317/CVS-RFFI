#!/usr/bin/env python3
"""Audit and aggregate the qKNNV42 non-dense adapter-epoch sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = ("singlehead_fft96_paired", "E2", "E5", "E10", "E20", "E30", "E60")
EPOCH_BY_ARM = {"singlehead_fft96_paired": 0, "E2": 2, "E5": 5, "E10": 10, "E20": 20, "E30": 30, "E60": 60}
K_GRID = (1, 2, 5, 10, 20)
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_TASK_FILES = {
    "detailed_metrics.csv", "detailed_metrics.json", "loss_trace.csv", "loss_trace.json",
    "metrics.json", "resolved_config.json", "score_table.csv", "split_manifest.json",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else math.nan


def ci95(values):
    values = [float(value) for value in values]
    return 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def infer_k(path: Path):
    match = re.search(r"[\\/]k_(\d+)[\\/]", str(path))
    if not match:
        raise RuntimeError(f"cannot infer K from {path}")
    return int(match.group(1))


def parse_tasks(run_root: Path):
    rows, scenario_rows, errors = [], [], []
    split_groups = {}
    score_count = loss_count = detailed_count = 0
    for path in sorted(run_root.rglob("metrics.json")):
        relative = path.relative_to(run_root)
        arm = relative.parts[0]
        if arm == "singlehead_fft96":
            # Preserved pre-repair result from a different feature sample pool.
            continue
        if arm not in ARMS:
            errors.append(f"unknown_arm:{arm}:{path}")
            continue
        directory = path.parent
        names = {item.name for item in directory.iterdir() if item.is_file()}
        missing = sorted(EXPECTED_TASK_FILES - names)
        if missing:
            errors.append(f"missing_task_files:{arm}:{directory}:{','.join(missing)}")
            continue
        payload = read_json(path)
        manifest = read_json(directory / "split_manifest.json")
        resolved = read_json(directory / "resolved_config.json")
        k_shot = infer_k(path)
        metrics = payload["metrics"]
        scenarios = payload["metrics_by_scenario"]
        old_before = mean(item["old_acc_before_increment"] for item in scenarios.values())
        old_to_new = mean(item["old_to_seen_new_rate"] for item in scenarios.values())
        new_to_old = mean(item["seen_new_to_old_rate"] for item in scenarios.values())
        rows.append({
            "arm": arm,
            "adapter_epoch": EPOCH_BY_ARM[arm],
            "resource_tier": resolved.get("adapter_resource_tier", ""),
            "receiver": payload["target_receiver_label"],
            "seed": int(payload["seed"]),
            "k_shot": k_shot,
            "old_before": old_before,
            "old_acc": float(metrics["old_acc_mean"]),
            "seen_new_acc": float(metrics["seen_new_acc_mean"]),
            "H_old_new": float(metrics["H_old_new_mean"]),
            "average_forgetting": float(metrics["average_forgetting_mean"]),
            "old_to_seen_new_rate": old_to_new,
            "seen_new_to_old_rate": new_to_old,
            "min_old_class_acc": float(metrics["min_old_class_acc"]),
            "min_seen_new_class_acc": float(metrics["min_seen_new_class_acc"]),
            "run_dir": str(directory),
        })
        score_count += len(read_csv(directory / "score_table.csv"))
        loss_rows = read_csv(directory / "loss_trace.csv")
        loss_count += len(loss_rows)
        detailed_count += len(read_csv(directory / "detailed_metrics.csv"))
        for loss_row in loss_rows:
            value = float(loss_row["loss"])
            if not math.isfinite(value):
                errors.append(f"nonfinite_loss:{directory}")
        required_manifest = {
            "support_query_overlap": False,
            "qknnv42_decision_mode": "per_sample_argmax",
            "qknnv42_labelprop_mode": "disabled",
            "non_deployment_oracle_diagnostic": False,
            "query_used_for_joint_decision": False,
            "query_used_for_transductive_inference": False,
            "target_query_used_for_training": False,
            "target_query_used_for_model_selection": False,
            "qknnv42_aux_feature_dim": 96,
        }
        for field, expected in required_manifest.items():
            if manifest.get(field) != expected:
                errors.append(f"manifest:{arm}:{directory}:{field}={manifest.get(field)!r}")
        expected_views = 1 if arm == "singlehead_fft96_paired" else 5
        if int(manifest.get("satellite_tta_view_count", -1)) != expected_views:
            errors.append(f"view_count:{arm}:{directory}:{manifest.get('satellite_tta_view_count')}")
        for scenario, item in scenarios.items():
            if scenario not in SCENARIOS:
                errors.append(f"unexpected_scenario:{scenario}:{directory}")
            for field in (
                "feature_adapter_uses_query", "query_labels_used_for_adaptation",
                "query_query_graph_used", "query_batch_state_required", "role_oracle_used",
                "equal_class_quota_used", "decision_batch_state_required",
            ):
                if item.get(field) is not False:
                    errors.append(f"scenario_flag:{arm}:{directory}:{scenario}:{field}={item.get(field)!r}")
            if item.get("labelprop_mode") != "disabled" or item.get("decision_mode") != "per_sample_argmax":
                errors.append(f"scenario_mode:{arm}:{directory}:{scenario}")
            if item.get("aux_feature_enabled") is not True or int(item.get("aux_feature_dim", -1)) != 96:
                errors.append(f"scenario_fft:{arm}:{directory}:{scenario}")
            scenario_rows.append({
                "arm": arm, "adapter_epoch": EPOCH_BY_ARM[arm],
                "receiver": payload["target_receiver_label"], "seed": int(payload["seed"]),
                "k_shot": k_shot, "scenario": scenario,
                "old_before": float(item["old_acc_before_increment"]),
                "old_acc": float(item["old_acc"]), "seen_new_acc": float(item["seen_new_acc"]),
                "H_old_new": float(item["H_old_new"]),
                "average_forgetting": float(item["average_forgetting"]),
                "head_macs": int(item["estimated_head_macs_with_post_adapter"]) + int(item["aux_estimated_head_macs"]),
                "persistent_state_bytes": int(item["persistent_state_bytes_with_post_adapter"]) + int(item["aux_persistent_state_bytes"]),
                "scoring_latency_per_query_ms": float(item["onboard_scoring_latency_per_query_ms"]),
            })
        for scenario, item in manifest["splits_by_scenario"].items():
            split_groups[(arm, payload["target_receiver_label"], int(payload["seed"]), scenario, k_shot)] = {
                "support": set(item["support_sample_ids"]),
                "query": tuple(item["query_sample_ids"]),
            }
    return rows, scenario_rows, split_groups, {
        "score_count": score_count, "loss_count": loss_count, "detailed_count": detailed_count,
        "errors": errors,
    }


def audit_splits(groups):
    errors = []
    base_keys = {(arm, rx, seed, scenario) for arm, rx, seed, scenario, _ in groups}
    for base in sorted(base_keys):
        entries = {k: groups[base + (k,)] for k in K_GRID}
        queries = {entry["query"] for entry in entries.values()}
        if len(queries) != 1:
            errors.append(f"query_changed_across_k:{base}")
        for low, high in zip(K_GRID, K_GRID[1:]):
            if not entries[low]["support"] <= entries[high]["support"]:
                errors.append(f"support_not_nested:{base}:{low}->{high}")
        for k, entry in entries.items():
            if len(entry["support"]) != 8 * k or len(entry["query"]) != 160:
                errors.append(f"split_count:{base}:K{k}:{len(entry['support'])}:{len(entry['query'])}")
            if entry["support"] & set(entry["query"]):
                errors.append(f"support_query_overlap:{base}:K{k}")
    paired_keys = {(rx, seed, scenario, k) for _, rx, seed, scenario, k in groups}
    for rx, seed, scenario, k in sorted(paired_keys):
        entries = [groups[(arm, rx, seed, scenario, k)] for arm in ARMS]
        if len({entry["query"] for entry in entries}) != 1:
            errors.append(f"query_changed_across_arms:{rx}:{seed}:{scenario}:K{k}")
        if len({tuple(sorted(entry["support"])) for entry in entries}) != 1:
            errors.append(f"support_changed_across_arms:{rx}:{seed}:{scenario}:K{k}")
    return errors


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    metrics = (
        "old_before", "old_acc", "seen_new_acc", "H_old_new", "average_forgetting",
        "old_to_seen_new_rate", "seen_new_to_old_rate", "min_old_class_acc", "min_seen_new_class_acc",
    )
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = {field: value for field, value in zip(fields, key)}
        out["count"] = len(group)
        for metric in metrics:
            values = [row[metric] for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = statistics.pstdev(values)
            out[f"{metric}_ci95"] = ci95(values)
        output.append(out)
    return output


def audit_logs(log_root: Path):
    patterns = {
        "traceback": re.compile(r"Traceback"), "runtime_error": re.compile(r"RuntimeError"),
        "oom": re.compile(r"CUDA out of memory", re.I), "killed": re.compile(r"(?:^|\s)Killed(?:\s|$)", re.M),
        "nan": re.compile(r"\bnan\b", re.I), "inf": re.compile(r"\binf\b", re.I),
    }
    hits = {key: [] for key in patterns}
    file_count = byte_count = line_count = 0
    adapter_trace = []
    for path in sorted(log_root.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        file_count += 1; byte_count += len(raw); line_count += len(text.splitlines())
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits[name].append(str(path))
        if path.name.startswith("launcher_e") and path.suffix == ".out":
            arm = path.parent.name
            for line in text.splitlines():
                if not line.startswith("{") or '"epoch"' not in line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "loss" in item and "epoch" in item:
                    adapter_trace.append({"arm": arm, **item})
    return {
        "file_count": file_count, "byte_count": byte_count, "line_count": line_count,
        "error_hit_counts": {key: len(value) for key, value in hits.items()},
        "error_hit_examples": {key: value[:5] for key, value in hits.items()},
    }, adapter_trace


def audit_features(feature_root: Path):
    rows, errors = [], []
    for path in sorted(feature_root.rglob("*.npz")):
        with np.load(path, allow_pickle=True) as payload:
            manifest = json.loads(str(payload["manifest_json"].item()))
        audit = manifest.get("checkpoint_load_audit", {})
        arm = next((part for part in path.parts if re.fullmatch(r"E(?:2|5|10|20|30|60)", part)), "")
        row = {
            "arm": arm, "feature_path": str(path),
            "checkpoint_load_strict": bool(manifest.get("checkpoint_load_strict")),
            "missing_keys": int(audit.get("missing_keys", -1)),
            "unexpected_keys": int(audit.get("unexpected_keys", -1)),
            "shape_mismatch": int(audit.get("skipped_mismatch", -1)),
            "tta_view_count": int(manifest.get("satellite_tta_view_count", -1)),
            "fft_dim": int(manifest.get("aux_fft_logmag_dim", -1)),
            "adapter_epochs": int(manifest.get("adapter", {}).get("epochs", -1)),
            "uses_target_labels_for_training": bool(manifest.get("uses_target_labels_for_training", False)),
        }
        rows.append(row)
        expected_epoch = EPOCH_BY_ARM.get(arm, -999)
        if not row["checkpoint_load_strict"] or (row["missing_keys"], row["unexpected_keys"], row["shape_mismatch"]) != (0, 0, 0):
            errors.append(f"checkpoint_manifest:{path}")
        if row["tta_view_count"] != 5 or row["fft_dim"] != 96 or row["adapter_epochs"] != expected_epoch:
            errors.append(f"feature_resource_manifest:{path}:{row}")
        if row["uses_target_labels_for_training"]:
            errors.append(f"target_label_training:{path}")
    return rows, errors


def audit_baseline_features(feature_root: Path):
    rows, errors = [], []
    for path in sorted(feature_root.rglob("*.npz")):
        with np.load(path, allow_pickle=True) as payload:
            manifest = json.loads(str(payload["manifest_json"].item()))
        audit = manifest.get("checkpoint_load_audit", {})
        adapter = manifest.get("adapter", {})
        row = {
            "arm": "singlehead_fft96_paired", "feature_path": str(path),
            "checkpoint_load_strict": bool(manifest.get("checkpoint_load_strict")),
            "missing_keys": int(audit.get("missing_keys", -1)),
            "unexpected_keys": int(audit.get("unexpected_keys", -1)),
            "shape_mismatch": int(audit.get("skipped_mismatch", -1)),
            "tta_view_count": int(manifest.get("satellite_tta_view_count", -1)),
            "fft_dim": int(manifest.get("aux_fft_logmag_dim", -1)),
            "adapter_epochs": int(adapter.get("epochs", -1)),
            "uses_target_labels_for_training": bool(adapter.get("uses_target_labels_for_training", True)),
        }
        rows.append(row)
        if not row["checkpoint_load_strict"] or (row["missing_keys"], row["unexpected_keys"], row["shape_mismatch"]) != (0, 0, 0):
            errors.append(f"baseline_checkpoint_manifest:{path}")
        if row["tta_view_count"] != 1 or row["fft_dim"] != 96 or row["adapter_epochs"] != 0:
            errors.append(f"baseline_resource_manifest:{path}:{row}")
        if row["uses_target_labels_for_training"] or adapter.get("skip_adapter_training") is not True:
            errors.append(f"baseline_training_scope:{path}:{adapter}")
    return rows, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--baseline-feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, scenario_rows, split_groups, task_audit = parse_tasks(args.run_root)
    split_errors = audit_splits(split_groups)
    log_audit, adapter_trace = audit_logs(args.log_root)
    feature_rows, feature_errors = audit_features(args.feature_root)
    baseline_feature_rows, baseline_feature_errors = audit_baseline_features(args.baseline_feature_root)
    feature_rows.extend(baseline_feature_rows)
    feature_errors.extend(baseline_feature_errors)
    count_by_arm = {arm: sum(row["arm"] == arm for row in rows) for arm in ARMS}
    errors = list(task_audit.pop("errors")) + split_errors + feature_errors
    if len(rows) != 875 or any(value != 125 for value in count_by_arm.values()):
        errors.append(f"task_count:{len(rows)}:{count_by_arm}")
    if len(scenario_rows) != 2625 or task_audit["score_count"] != 420000 or task_audit["loss_count"] != 2625:
        errors.append(f"record_counts:{len(scenario_rows)}:{task_audit}")
    if len(feature_rows) != 35:
        errors.append(f"feature_count:{len(feature_rows)}")
    if any(log_audit["error_hit_counts"].values()):
        errors.append(f"log_errors:{log_audit['error_hit_counts']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "task_rows.csv", rows)
    write_csv(args.output_dir / "scenario_rows.csv", scenario_rows)
    write_csv(args.output_dir / "summary_by_arm_k.csv", summarize(rows, ("adapter_epoch", "arm", "k_shot")))
    write_csv(args.output_dir / "summary_by_arm.csv", summarize(rows, ("adapter_epoch", "arm")))
    write_csv(args.output_dir / "feature_manifest_audit.csv", feature_rows)
    write_csv(args.output_dir / "adapter_loss_trace.csv", adapter_trace)
    audit = {
        "status": "PASS" if not errors else "FAIL", "task_count": len(rows),
        "count_by_arm": count_by_arm, "scenario_count": len(scenario_rows),
        **task_audit, "split_group_count": len(split_groups), "split_error_count": len(split_errors),
        "feature_count": len(feature_rows), "feature_error_count": len(feature_errors),
        "adapter_trace_count": len(adapter_trace), "log_audit": log_audit,
        "errors": errors[:200], "error_count": len(errors),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
