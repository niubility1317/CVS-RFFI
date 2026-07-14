#!/usr/bin/env python3
"""Aggregate strict qKNNV42 dual-125 reruns and compare compatibility diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("old_acc", "seen_new_acc", "H_old_new")


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else math.nan


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_arm(root: Path, arm: str):
    rows = []
    scenarios = []
    for path in sorted(root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = re.search(r"[\\/]k_(\d+)[\\/]", str(path))
        if not match:
            raise RuntimeError(f"cannot infer K from {path}")
        metrics = payload["metrics"]
        row = {
            "arm": arm,
            "experiment_id": payload["experiment_id"],
            "receiver": payload["target_receiver_label"],
            "seed": int(payload["seed"]),
            "k_shot": int(match.group(1)),
            "old_acc": float(metrics["old_acc_mean"]),
            "seen_new_acc": float(metrics["seen_new_acc_mean"]),
            "H_old_new": float(metrics["H_old_new_mean"]),
            "average_forgetting": float(metrics["average_forgetting_mean"]),
            "run_dir": str(path.parent),
        }
        row["min_old_new"] = min(row["old_acc"], row["seen_new_acc"])
        row["joint_score"] = mean(row[m] for m in METRICS)
        rows.append(row)
        for scenario, values in payload["metrics_by_scenario"].items():
            scenarios.append({
                "arm": arm,
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "scenario": scenario,
                "old_acc": float(values["old_acc"]),
                "seen_new_acc": float(values["seen_new_acc"]),
                "H_old_new": float(values["H_old_new"]),
                "average_forgetting": float(values["average_forgetting"]),
            })
    if len(rows) != 125:
        raise RuntimeError(f"{arm}: expected 125 rows, got {len(rows)}")
    return rows, scenarios


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = {field: value for field, value in zip(fields, key)}
        out["count"] = len(group)
        for metric in METRICS + ("average_forgetting",):
            values = [float(row[metric]) for row in group]
            out[metric] = mean(values)
            out[f"{metric}_std"] = statistics.pstdev(values)
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)
        output.append(out)
    return output


def load_diagnostic(path: Path):
    index = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["arm"], row["receiver"], int(row["seed"]), int(row["k_shot"]))
            index[key] = {metric: float(row[metric]) for metric in METRICS}
    if len(index) != 250:
        raise RuntimeError(f"expected 250 compatibility rows, got {len(index)}")
    return index


def paired(rows, reference, include_arm=True):
    output = []
    for row in rows:
        diagnostic_arm = row["arm"].removesuffix("_strict")
        key = (diagnostic_arm, row["receiver"], row["seed"], row["k_shot"])
        ref = reference[key]
        out = {field: row[field] for field in ("arm", "experiment_id", "receiver", "seed", "k_shot")}
        for metric in METRICS:
            out[f"{metric}_strict"] = row[metric]
            out[f"{metric}_compatibility"] = ref[metric]
            out[f"delta_{metric}"] = row[metric] - ref[metric]
        output.append(out)
    return output


def pair_arms(rows):
    index = {(row["arm"], row["receiver"], row["seed"], row["k_shot"]): row for row in rows}
    output = []
    for key, full in sorted(index.items()):
        arm, receiver, seed, k_shot = key
        if arm != "full_legacy_oracle_strict":
            continue
        light = index[("singleview_fft96_strict", receiver, seed, k_shot)]
        out = {"receiver": receiver, "seed": seed, "k_shot": k_shot}
        for metric in METRICS:
            out[f"{metric}_full"] = full[metric]
            out[f"{metric}_light"] = light[metric]
            out[f"delta_{metric}"] = full[metric] - light[metric]
        output.append(out)
    return output


def delta_summary(rows, group_field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(row)
    return [{
        group_field: key,
        "count": len(group),
        **{f"delta_{metric}": mean(row[f"delta_{metric}"] for row in group) for metric in METRICS},
    } for key, group in sorted(groups.items())]


def audit_run_root(root: Path, arm: str):
    split_paths = sorted(root.rglob("split_manifest.json"))
    loss_paths = sorted(root.rglob("loss_trace.csv"))
    score_paths = sorted(root.rglob("score_table.csv"))
    overlap_violations = []
    loss_rows = 0
    nonfinite_loss_rows = 0
    gradient_updates = set()
    for path in split_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("support_query_overlap") is not False:
            overlap_violations.append(str(path))
        for scenario, split in payload["splits_by_scenario"].items():
            overlap = set(split["support_sample_ids"]) & set(split["query_sample_ids"])
            if overlap:
                overlap_violations.append(f"{path}:{scenario}:{len(overlap)}")
    for path in loss_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                loss_rows += 1
                loss = float(row["loss"])
                if not math.isfinite(loss):
                    nonfinite_loss_rows += 1
                gradient_updates.add(int(row["gradient_updates"]))
    return {
        "arm": arm,
        "metrics_files": len(list(root.rglob("metrics.json"))),
        "split_manifest_files": len(split_paths),
        "loss_trace_files": len(loss_paths),
        "score_table_files": len(score_paths),
        "loss_rows": loss_rows,
        "nonfinite_loss_rows": nonfinite_loss_rows,
        "gradient_updates_values": sorted(gradient_updates),
        "support_query_overlap_violations": len(overlap_violations),
        "support_query_overlap_examples": overlap_violations[:5],
    }


def audit_logs(roots):
    patterns = {
        "traceback": re.compile(r"Traceback"),
        "oom": re.compile(r"CUDA out of memory", re.IGNORECASE),
        "killed": re.compile(r"(?:^|\s)Killed(?:\s|$)", re.MULTILINE),
        "nan_or_inf": re.compile(r"(?:^|[^A-Za-z])(?:nan|inf)(?:[^A-Za-z]|$)", re.IGNORECASE),
    }
    files = []
    hits = {name: [] for name in patterns}
    total_bytes = 0
    total_lines = 0
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            total_bytes += len(raw)
            text = raw.decode("utf-8-sig", errors="replace")
            total_lines += len(text.splitlines())
            files.append(str(path))
            for name, pattern in patterns.items():
                if pattern.search(text):
                    hits[name].append(str(path))
    return {
        "root_count": len(roots),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "error_hit_counts": {name: len(paths) for name, paths in hits.items()},
        "error_hit_examples": {name: paths[:5] for name, paths in hits.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--compatibility-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    light, light_scenarios = load_arm(args.light_root, "singleview_fft96_strict")
    full, full_scenarios = load_arm(args.full_root, "full_legacy_oracle_strict")
    rows = light + full
    scenarios = light_scenarios + full_scenarios
    comparison = paired(rows, load_diagnostic(args.compatibility_csv))
    cross_arm = pair_arms(rows)
    by_arm = summarize(rows, ["arm"])
    by_k = summarize(rows, ["arm", "k_shot"])
    by_receiver = summarize(rows, ["arm", "receiver"])
    by_scenario = summarize(scenarios, ["arm", "scenario"])
    compatibility_delta = delta_summary(comparison, "arm")
    cross_summary = [{
        "comparison": "full_minus_light_strict",
        "count": len(cross_arm),
        **{f"delta_{metric}": mean(row[f"delta_{metric}"] for row in cross_arm) for metric in METRICS},
    }]
    ranked = sorted(rows, key=lambda row: (min(row["old_acc"], row["seen_new_acc"]), row["H_old_new"]), reverse=True)
    thresholds = [{
        "arm": arm,
        "count": len(group),
        "old_ge_80": sum(row["old_acc"] >= 0.8 for row in group),
        "new_ge_80": sum(row["seen_new_acc"] >= 0.8 for row in group),
        "H_ge_80": sum(row["H_old_new"] >= 0.8 for row in group),
        "old_and_new_ge_80": sum(row["old_acc"] >= 0.8 and row["seen_new_acc"] >= 0.8 for row in group),
    } for arm, group in (("singleview_fft96_strict", light), ("full_legacy_oracle_strict", full))]
    artifact_audit = [
        audit_run_root(args.light_root, "singleview_fft96_strict"),
        audit_run_root(args.full_root, "full_legacy_oracle_strict"),
    ]
    log_audit = audit_logs(args.log_root)

    outputs = {
        "per_run_results.csv": rows,
        "per_scenario_results.csv": scenarios,
        "summary_by_arm.csv": by_arm,
        "summary_by_k.csv": by_k,
        "summary_by_receiver.csv": by_receiver,
        "summary_by_scenario.csv": by_scenario,
        "paired_strict_vs_compatibility.csv": comparison,
        "paired_strict_vs_compatibility_summary.csv": compatibility_delta,
        "paired_full_minus_light_strict.csv": cross_arm,
        "paired_full_minus_light_strict_summary.csv": cross_summary,
        "ranked_joint_rows.csv": ranked,
        "threshold_counts.csv": thresholds,
    }
    for filename, payload in outputs.items():
        write_csv(args.output_dir / filename, payload)
    summary = {
        "strict_row_count": len(rows),
        "strict_scenario_row_count": len(scenarios),
        "summary_by_arm": by_arm,
        "strict_minus_compatibility": compatibility_delta,
        "full_minus_light_strict": cross_summary,
        "threshold_counts": thresholds,
        "artifact_audit": artifact_audit,
        "log_audit": log_audit,
        "top_joint_rows": ranked[:10],
        "bottom_joint_rows": list(reversed(ranked[-10:])),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
