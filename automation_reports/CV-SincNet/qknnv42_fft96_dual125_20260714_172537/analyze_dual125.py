#!/usr/bin/env python3
"""Aggregate the two qKNNV42 125-run arms without mixing metric rows."""

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
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else math.nan


def load_arm(root: Path, arm: str):
    rows = []
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
        row["joint_score"] = mean([row["old_acc"], row["seen_new_acc"], row["H_old_new"]])
        rows.append(row)
    return rows


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out = {field: value for field, value in zip(fields, key)}
        out["count"] = len(group)
        for metric in METRICS + ("average_forgetting", "min_old_new", "joint_score"):
            out[metric] = mean(row[metric] for row in group)
        for metric in METRICS:
            values = [row[metric] for row in group]
            out[f"{metric}_std"] = statistics.pstdev(values)
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)
        output.append(out)
    return output


def load_baseline(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "cvs_qknnv42":
                continue
            key = (row["receiver"], int(row["seed"]), int(row["k_shot"]))
            rows[key] = {metric: float(row[metric]) for metric in METRICS}
    return rows


def paired_baseline(rows, baseline):
    output = []
    for row in rows:
        key = (row["receiver"], row["seed"], row["k_shot"])
        if key not in baseline:
            continue
        out = {k: row[k] for k in ("arm", "experiment_id", "receiver", "seed", "k_shot")}
        for metric in METRICS:
            out[f"{metric}_candidate"] = row[metric]
            out[f"{metric}_baseline"] = baseline[key][metric]
            out[f"delta_{metric}"] = row[metric] - baseline[key][metric]
        output.append(out)
    return output


def paired_arms(rows, left_arm, right_arm):
    index = {
        (row["arm"], row["receiver"], row["seed"], row["k_shot"]): row
        for row in rows
    }
    output = []
    for key, left in sorted(index.items()):
        arm, receiver, seed, k_shot = key
        if arm != left_arm:
            continue
        right = index[(right_arm, receiver, seed, k_shot)]
        out = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
        }
        for metric in METRICS:
            out[f"{metric}_{left_arm}"] = left[metric]
            out[f"{metric}_{right_arm}"] = right[metric]
            out[f"delta_{metric}"] = left[metric] - right[metric]
        output.append(out)
    return output


def scenario_rows(root: Path, arm: str):
    output = []
    for path in sorted(root.rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = re.search(r"[\\/]k_(\d+)[\\/]", str(path))
        for scenario, metrics in payload["metrics_by_scenario"].items():
            output.append({
                "arm": arm,
                "receiver": payload["target_receiver_label"],
                "seed": int(payload["seed"]),
                "k_shot": int(match.group(1)),
                "scenario": scenario,
                "old_acc": float(metrics["old_acc"]),
                "seen_new_acc": float(metrics["seen_new_acc"]),
                "H_old_new": float(metrics["H_old_new"]),
                "average_forgetting": float(metrics["average_forgetting"]),
                "min_old_new": min(float(metrics["old_acc"]), float(metrics["seen_new_acc"])),
                "joint_score": mean([metrics["old_acc"], metrics["seen_new_acc"], metrics["H_old_new"]]),
            })
    return output


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arms = [("singleview_fft96", args.light_root)]
    if args.full_root and args.full_root.exists():
        arms.append(("full_legacy_oracle", args.full_root))

    rows = []
    scenarios = []
    for arm, root in arms:
        arm_rows = load_arm(root, arm)
        if len(arm_rows) != 125:
            raise RuntimeError(f"{arm}: expected 125 metrics rows, got {len(arm_rows)}")
        rows.extend(arm_rows)
        scenarios.extend(scenario_rows(root, arm))

    baseline = load_baseline(args.baseline_csv)
    paired = paired_baseline(rows, baseline)
    cross_arm = []
    if len(arms) == 2:
        cross_arm = paired_arms(rows, "full_legacy_oracle", "singleview_fft96")
    by_arm = summarize(rows, ["arm"])
    by_k = summarize(rows, ["arm", "k_shot"])
    by_receiver = summarize(rows, ["arm", "receiver"])
    by_scenario = summarize(scenarios, ["arm", "scenario"])

    ranked = sorted(rows, key=lambda r: (r["min_old_new"], r["H_old_new"], r["joint_score"]), reverse=True)
    thresholds = []
    for arm, root in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        thresholds.append({
            "arm": arm,
            "count": len(arm_rows),
            "old_ge_80": sum(row["old_acc"] >= 0.8 for row in arm_rows),
            "new_ge_80": sum(row["seen_new_acc"] >= 0.8 for row in arm_rows),
            "H_ge_80": sum(row["H_old_new"] >= 0.8 for row in arm_rows),
            "old_and_new_ge_80": sum(row["old_acc"] >= 0.8 and row["seen_new_acc"] >= 0.8 for row in arm_rows),
        })

    delta_summary = summarize([
        {
            "arm": row["arm"],
            "old_acc": row["delta_old_acc"],
            "seen_new_acc": row["delta_seen_new_acc"],
            "H_old_new": row["delta_H_old_new"],
            "average_forgetting": 0.0,
            "min_old_new": min(row["delta_old_acc"], row["delta_seen_new_acc"]),
            "joint_score": mean([row["delta_old_acc"], row["delta_seen_new_acc"], row["delta_H_old_new"]]),
        }
        for row in paired
    ], ["arm"])
    cross_arm_delta_summary = []
    if cross_arm:
        cross_arm_delta_summary = [{
            "left_arm": "full_legacy_oracle",
            "right_arm": "singleview_fft96",
            "count": len(cross_arm),
            **{
                f"delta_{metric}": mean(row[f"delta_{metric}"] for row in cross_arm)
                for metric in METRICS
            },
        }]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_run_results.csv", rows)
    write_csv(args.output_dir / "per_scenario_results.csv", scenarios)
    write_csv(args.output_dir / "summary_by_arm.csv", by_arm)
    write_csv(args.output_dir / "summary_by_k.csv", by_k)
    write_csv(args.output_dir / "summary_by_receiver.csv", by_receiver)
    write_csv(args.output_dir / "summary_by_scenario.csv", by_scenario)
    write_csv(args.output_dir / "paired_vs_original_qknnv42.csv", paired)
    write_csv(args.output_dir / "paired_delta_summary.csv", delta_summary)
    write_csv(args.output_dir / "paired_full_minus_light.csv", cross_arm)
    write_csv(args.output_dir / "paired_full_minus_light_summary.csv", cross_arm_delta_summary)
    write_csv(args.output_dir / "ranked_joint_rows.csv", ranked)
    write_csv(args.output_dir / "threshold_counts.csv", thresholds)
    summary = {
        "arm_count": len(arms),
        "row_count": len(rows),
        "scenario_row_count": len(scenarios),
        "baseline_qknnv42_row_count": len(baseline),
        "summary_by_arm": by_arm,
        "paired_delta_summary": delta_summary,
        "paired_full_minus_light_summary": cross_arm_delta_summary,
        "threshold_counts": thresholds,
        "top_joint_rows": ranked[:10],
        "bottom_joint_rows": list(reversed(ranked[-10:])),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
