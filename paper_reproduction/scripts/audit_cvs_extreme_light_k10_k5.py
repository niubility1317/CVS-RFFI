"""Audit matched K10-primary and K5-sensitivity Stage2-C rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
DROP_METRICS = ("old_acc", "min_old_class_acc", "seen_new_acc", "H_old_new")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _accuracy(rows: Iterable[dict[str, str]]) -> float:
    values = list(rows)
    if not values:
        raise ValueError("accuracy group is empty")
    return sum(str(row["true_label"]) == str(row["predicted_label"]) for row in values) / len(values)


def scenario_metrics(
    rows: list[dict[str, str]], *, old_labels: set[str], new_labels: set[str]
) -> dict[str, float]:
    old_rows = [row for row in rows if str(row["true_label"]) in old_labels]
    new_rows = [row for row in rows if str(row["true_label"]) in new_labels]
    old_acc = _accuracy(old_rows)
    seen_new_acc = _accuracy(new_rows)
    per_old = {
        label: _accuracy(row for row in old_rows if str(row["true_label"]) == label)
        for label in sorted(old_labels)
    }
    harmonic = 0.0 if old_acc + seen_new_acc <= 0.0 else 2.0 * old_acc * seen_new_acc / (old_acc + seen_new_acc)
    return {
        "old_acc": old_acc,
        "min_old_class_acc": min(per_old.values()),
        "seen_new_acc": seen_new_acc,
        "H_old_new": harmonic,
    }


def validate_nested_ids(
    support5: list[str], support10: list[str], query5: list[str], query10: list[str]
) -> dict[str, bool]:
    def counts(values: list[str]) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for value in values:
            parts = str(value).split("|")
            if len(parts) < 2:
                raise ValueError(f"malformed physical sample ID: {value}")
            result[parts[1]] += 1
        return dict(result)

    count5 = counts(support5)
    count10 = counts(support10)
    return {
        "query_ids_identical": list(query5) == list(query10),
        "k5_support_subset_k10": set(support5) <= set(support10),
        "k5_exact_per_class": bool(count5) and all(value == 5 for value in count5.values()),
        "k10_exact_per_class": bool(count10) and all(value == 10 for value in count10.values()),
        "registered_class_sets_identical": set(count5) == set(count10),
    }


def _load_row(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    score_rows = list(csv.DictReader((run_dir / "score_table.csv").open(encoding="utf-8", newline="")))
    by_scenario = {
        scenario: scenario_metrics(
            [row for row in score_rows if str(row["scenario"]) == scenario],
            old_labels=set(map(str, config["target_old_tx_labels"])),
            new_labels=set(map(str, config["target_new_tx_labels"])),
        )
        for scenario in SCENARIOS
    }
    return {"config": config, "metrics": metrics, "split": split, "scenario_metrics": by_scenario, "run_dir": run_dir}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "matrix_manifest.json").read_text(encoding="utf-8"))
    primary_k = int(manifest.get("primary_k_shot") or 10)
    sensitivity_k = int(manifest.get("sensitivity_k_shot") or 5)
    max_drop = float(manifest.get("k5_max_drop_pp") or 3.0) / 100.0
    if primary_k != 10 or sensitivity_k != 5:
        raise ValueError(f"formal K audit requires primary=10 and sensitivity=5, got {primary_k}/{sensitivity_k}")
    if int(manifest.get("support_pool_max_k") or 0) != primary_k:
        raise ValueError("support_pool_max_k must equal primary K10 for matched query construction")
    paths = sorted(args.run_root.rglob("metrics.json"))
    if len(paths) != int(manifest["row_count"]):
        raise RuntimeError("matrix is incomplete")
    loaded = [_load_row(path.parent) for path in paths]
    groups: dict[tuple[str, int, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in loaded:
        config = row["config"]
        key = (
            str(config["matrix_arm"]),
            int(config["matrix_new_class_count"]),
            str(config["target_receiver_labels"][0]),
            int(config["seed"]),
        )
        groups[key][int(config["k_shot"])] = row
    pair_rows: list[dict[str, Any]] = []
    arm_gate_values: dict[str, list[bool]] = defaultdict(list)
    thresholds = manifest["success_thresholds"]
    for key, pair in sorted(groups.items()):
        if set(pair) != {sensitivity_k, primary_k}:
            raise RuntimeError(f"missing matched K5/K10 rows for {key}: {sorted(pair)}")
        arm, count, receiver, seed = key
        k5, k10 = pair[sensitivity_k], pair[primary_k]
        k10_metric = k10["metrics"]
        k10_absolute_pass = (
            float(k10_metric["old_acc_mean"]) >= float(thresholds["old_acc"])
            and float(k10_metric["min_old_class_acc"]) >= float(thresholds["min_old_class_acc"])
            and float(k10_metric["seen_new_acc_mean"]) >= float(thresholds[f"seen_new_acc_{count}"])
        )
        pair_pass = k10_absolute_pass
        for scenario in SCENARIOS:
            split5 = k5["split"]["splits_by_scenario"][scenario]
            split10 = k10["split"]["splits_by_scenario"][scenario]
            nesting = validate_nested_ids(
                split5["support_sample_ids"], split10["support_sample_ids"],
                split5["query_sample_ids"], split10["query_sample_ids"],
            )
            metric5 = k5["scenario_metrics"][scenario]
            metric10 = k10["scenario_metrics"][scenario]
            item: dict[str, Any] = {
                "arm": arm,
                "new_class_count": count,
                "receiver": receiver,
                "seed": seed,
                "scenario": scenario,
                "k10_absolute_pass": k10_absolute_pass,
                **nesting,
            }
            for metric in DROP_METRICS:
                drop = float(metric10[metric]) - float(metric5[metric])
                item[f"k10_{metric}"] = metric10[metric]
                item[f"k5_{metric}"] = metric5[metric]
                item[f"k5_drop_{metric}_pp"] = 100.0 * drop
                item[f"k5_drop_{metric}_pass"] = drop <= max_drop + 1.0e-12
            item["split_nesting_pass"] = all(nesting.values())
            item["k5_all_metric_drop_pass"] = all(bool(item[f"k5_drop_{metric}_pass"]) for metric in DROP_METRICS)
            item["matched_cell_pass"] = bool(
                k10_absolute_pass and item["split_nesting_pass"] and item["k5_all_metric_drop_pass"]
            )
            pair_pass = pair_pass and bool(item["matched_cell_pass"])
            pair_rows.append(item)
        arm_gate_values[arm].append(pair_pass)
    arm_rows = [
        {
            "arm": arm,
            "matched_pair_count": len(values),
            "matched_pair_pass_count": sum(values),
            "global_arm_pass": all(values),
        }
        for arm, values in sorted(arm_gate_values.items())
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "k10_k5_matched_cells.csv", pair_rows)
    _write_csv(args.output_dir / "arm_gate.csv", arm_rows)
    audit = {
        "schema": "cvs_stage2c_extreme_light_k10_k5_audit_v1",
        "primary_k_shot": primary_k,
        "sensitivity_k_shot": sensitivity_k,
        "k5_max_drop_pp": 100.0 * max_drop,
        "matched_scenario_cell_count": len(pair_rows),
        "matched_scenario_cell_pass_count": sum(bool(row["matched_cell_pass"]) for row in pair_rows),
        "global_passing_arm_count": sum(bool(row["global_arm_pass"]) for row in arm_rows),
        "support_query_nesting_violation_count": sum(not bool(row["split_nesting_pass"]) for row in pair_rows),
        "thresholds": thresholds,
    }
    (args.output_dir / "final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
