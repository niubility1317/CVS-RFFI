#!/usr/bin/env python
"""Score direct ground-classifier predictions on existing 125 old-query splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, values in sorted(groups.items()):
        correct = sum(int(row["correct"]) for row in values)
        total = sum(int(row["total"]) for row in values)
        output.append(
            {
                **{key: value for key, value in zip(keys, group)},
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "row_count": len(values),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    score_rows = _read_csv(Path(args.score_table))
    prediction = {
        (row["scenario"], row["sample_id"]): int(row["correct"])
        for row in score_rows
    }
    manifests = sorted(Path(args.split_root).glob("rx_*/seed_*/k_*/cvs_qknnv42/split_manifest.json"))
    if len(manifests) != int(args.expected_tasks):
        raise ValueError(f"expected {args.expected_tasks} manifests, got {len(manifests)}")
    per_scenario: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receiver = str(manifest["target_receiver_labels"][0])
        seed = int(manifest["seed"])
        k_shot = int(manifest["k_shot"])
        task_correct = 0
        task_total = 0
        for scenario, split in sorted(manifest["splits_by_scenario"].items()):
            old_ids = [
                sample_id
                for sample_id in split["query_sample_ids"]
                if str(sample_id).startswith("target_old|")
            ]
            missing = [sample_id for sample_id in old_ids if (scenario, sample_id) not in prediction]
            if missing:
                raise KeyError(
                    f"missing {len(missing)} predictions for receiver={receiver} seed={seed} k={k_shot} "
                    f"scenario={scenario}; first={missing[0]}"
                )
            correct = sum(prediction[(scenario, sample_id)] for sample_id in old_ids)
            total = len(old_ids)
            per_scenario.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "k_shot": k_shot,
                    "scenario": scenario,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total,
                }
            )
            task_correct += correct
            task_total += total
        per_task.append(
            {
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "correct": task_correct,
                "total": task_total,
                "old_acc": task_correct / task_total,
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "query125_per_task.csv", per_task)
    _write_csv(output_dir / "query125_per_scenario.csv", per_scenario)
    summary_rows: list[dict[str, Any]] = []
    for level, keys in (
        ("overall", tuple()),
        ("receiver", ("receiver",)),
        ("k_shot", ("k_shot",)),
        ("seed", ("seed",)),
        ("scenario", ("scenario",)),
        ("receiver_k", ("receiver", "k_shot")),
    ):
        source = per_task if level != "scenario" else per_scenario
        for row in _summarize(source, keys):
            summary_rows.append({"level": level, **row})
    _write_csv(output_dir / "query125_summary.csv", summary_rows)
    task_values = [float(row["old_acc"]) for row in per_task]
    result = {
        "schema": "adv3b02_ground_direct_old_query125_v1",
        "task_count": len(per_task),
        "scenario_task_count": len(per_scenario),
        "overall_accuracy": sum(int(row["correct"]) for row in per_task)
        / sum(int(row["total"]) for row in per_task),
        "task_macro_mean": sum(task_values) / len(task_values),
        "task_min": min(task_values),
        "task_max": max(task_values),
        "below_70_task_count": sum(value < 0.70 for value in task_values),
        "below_80_task_count": sum(value < 0.80 for value in task_values),
        "support_labels_used": False,
        "k_shot_role": "query_subset_selection_only",
    }
    (output_dir / "query125_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-tasks", type=int, default=125)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
