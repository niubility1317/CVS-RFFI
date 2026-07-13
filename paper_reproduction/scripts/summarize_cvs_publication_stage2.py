from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from paper_reproduction.scripts.run_cvs_publication_matrix import (
    DEFAULT_K,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    PHASE_METHODS,
    _artifact_status,
    build_rows,
)


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
REFERENCE_METHOD = {"stage2b": "cvs_opgac", "stage2c": "cvs_qknnv42"}
METRIC_KEYS = {
    "stage2b": (
        "target_old_accuracy",
        "target_old_accuracy_before_adaptation",
        "target_old_accuracy_delta",
    ),
    "stage2c": (
        "old_acc",
        "seen_new_acc",
        "H_old_new",
        "old_acc_before_increment",
        "average_forgetting",
    ),
}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric group")
    if not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError(f"non-finite values in metric group: {values}")
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


def _run_metric(
    payload: dict[str, Any],
    scenario_metrics: dict[str, Any],
    metric: str,
    *,
    run_dir: str,
) -> tuple[float, str]:
    mean_key = f"{metric}_mean"
    metrics = payload.get("metrics", {})
    if mean_key in metrics:
        value = float(metrics[mean_key])
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite {mean_key} in {run_dir}/metrics.json")
        return value, "payload_mean"
    values: list[float] = []
    for scenario in SCENARIOS:
        if metric not in scenario_metrics.get(scenario, {}):
            raise KeyError(f"{mean_key} and scenario {metric} missing from {run_dir}/metrics.json")
        value = float(scenario_metrics[scenario][metric])
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite {metric} for {scenario} in {run_dir}")
        values.append(value)
    return statistics.fmean(values), "scenario_mean_fallback"


def summarize_groups(
    rows: Iterable[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    metric_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result = {field: value for field, value in zip(group_fields, group)}
        for metric in metric_keys:
            stats = _stats([float(item[metric]) for item in items])
            result.update({f"{metric}_{key}": value for key, value in stats.items()})
        output.append(result)
    return output


def paired_rows(
    rows: Sequence[dict[str, Any]],
    *,
    phase: str,
    metric_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    reference_method = REFERENCE_METHOD[phase]
    key_fields = ("receiver", "k_shot", "seed")
    reference = {
        tuple(row[field] for field in key_fields): row
        for row in rows
        if row["method"] == reference_method
    }
    expected = len(DEFAULT_RECEIVERS) * len(DEFAULT_K) * len(DEFAULT_SEEDS)
    if len(reference) != expected:
        raise ValueError(f"{phase} reference rows must contain {expected} paired cells, got {len(reference)}")
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == reference_method:
            continue
        key = tuple(row[field] for field in key_fields)
        if key not in reference:
            raise ValueError(f"missing paired reference for {phase} {key}")
        ref = reference[key]
        item = {
            "phase": phase,
            "reference_method": reference_method,
            "candidate_method": row["method"],
            **{field: row[field] for field in key_fields},
        }
        for metric in metric_keys:
            item[f"reference_{metric}"] = float(ref[metric])
            item[f"candidate_{metric}"] = float(row[metric])
            item[f"delta_{metric}"] = float(row[metric]) - float(ref[metric])
        output.append(item)
    return output


def _paired_summary(
    rows: Sequence[dict[str, Any]], *, metric_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["candidate_method"], row["k_shot"])].append(row)
    output: list[dict[str, Any]] = []
    for group, items in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        phase, method, k_shot = group
        result: dict[str, Any] = {
            "phase": phase,
            "candidate_method": method,
            "reference_method": REFERENCE_METHOD[phase],
            "k_shot": k_shot,
        }
        for metric in metric_keys:
            deltas = [float(item[f"delta_{metric}"]) for item in items]
            stats = _stats(deltas)
            result.update({f"delta_{metric}_{key}": value for key, value in stats.items()})
            result[f"delta_{metric}_wins"] = sum(value > 0 for value in deltas)
            result[f"delta_{metric}_ties"] = sum(value == 0 for value in deltas)
            result[f"delta_{metric}_losses"] = sum(value < 0 for value in deltas)
        output.append(result)
    return output


def collect_phase(
    *,
    phase: str,
    run_root: Path,
    log_root: Path,
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = build_rows(
        phase=phase,
        methods=PHASE_METHODS[phase],
        receivers=DEFAULT_RECEIVERS,
        k_grid=DEFAULT_K,
        seeds=DEFAULT_SEEDS,
        output_root=run_root,
        log_root=log_root,
    )
    per_run: list[dict[str, Any]] = []
    per_scenario: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for row in rows:
        status = _artifact_status(row)
        if not status["complete"]:
            incomplete.append({**row.__dict__, "artifact_status": status})
            continue
        payload = json.loads((Path(row.run_dir) / "metrics.json").read_text(encoding="utf-8"))
        method = str(payload.get("method", payload.get("method_id", row.method)))
        if method != row.method:
            raise ValueError(f"method mismatch for {row.run_dir}: expected {row.method}, got {method}")
        scenario_metrics = payload.get("metrics_by_scenario", {})
        if tuple(scenario_metrics) != SCENARIOS:
            raise ValueError(
                f"scenario order/content mismatch for {row.run_dir}: {tuple(scenario_metrics)}"
            )
        base = {
            "phase": phase,
            "method": row.method,
            "receiver": row.receiver,
            "k_shot": row.k_shot,
            "seed": row.seed,
            "split_seed": row.split_seed,
            "experiment_id": row.experiment_id,
            "run_dir": row.run_dir,
        }
        run_item = dict(base)
        for metric in METRIC_KEYS[phase]:
            value, source = _run_metric(
                payload,
                scenario_metrics,
                metric,
                run_dir=row.run_dir,
            )
            run_item[metric] = value
            run_item[f"{metric}_aggregation_source"] = source
        per_run.append(run_item)
        for scenario in SCENARIOS:
            scenario_item = {**base, "scenario": scenario}
            for metric in METRIC_KEYS[phase]:
                if metric not in scenario_metrics[scenario]:
                    raise KeyError(f"{metric} missing for {scenario} in {row.run_dir}")
                scenario_item[metric] = float(scenario_metrics[scenario][metric])
            per_scenario.append(scenario_item)
    if incomplete and not allow_incomplete:
        raise RuntimeError(f"{phase} has {len(incomplete)} incomplete rows; refuse publication summary")
    return per_run, per_scenario, incomplete


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize full CVS Stage2 publication matrices")
    parser.add_argument("--stage2b-root", type=Path, required=True)
    parser.add_argument("--stage2c-root", type=Path, required=True)
    parser.add_argument("--stage2b-log-root", type=Path, required=True)
    parser.add_argument("--stage2c-log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_run: list[dict[str, Any]] = []
    all_scenario: list[dict[str, Any]] = []
    all_incomplete: list[dict[str, Any]] = []
    all_paired: list[dict[str, Any]] = []
    all_paired_summary: list[dict[str, Any]] = []
    for phase, run_root, log_root in (
        ("stage2b", args.stage2b_root, args.stage2b_log_root),
        ("stage2c", args.stage2c_root, args.stage2c_log_root),
    ):
        per_run, per_scenario, incomplete = collect_phase(
            phase=phase,
            run_root=run_root,
            log_root=log_root,
            allow_incomplete=bool(args.allow_incomplete),
        )
        all_run.extend(per_run)
        all_scenario.extend(per_scenario)
        all_incomplete.extend(incomplete)
        if not incomplete:
            phase_paired = paired_rows(per_run, phase=phase, metric_keys=METRIC_KEYS[phase])
            all_paired.extend(phase_paired)
            all_paired_summary.extend(_paired_summary(phase_paired, metric_keys=METRIC_KEYS[phase]))

    method_k_summary: list[dict[str, Any]] = []
    receiver_k_summary: list[dict[str, Any]] = []
    for phase in ("stage2b", "stage2c"):
        phase_rows = [row for row in all_run if row["phase"] == phase]
        if phase_rows:
            method_k_summary.extend(
                summarize_groups(
                    phase_rows,
                    group_fields=("phase", "method", "k_shot"),
                    metric_keys=METRIC_KEYS[phase],
                )
            )
            receiver_k_summary.extend(
                summarize_groups(
                    phase_rows,
                    group_fields=("phase", "method", "receiver", "k_shot"),
                    metric_keys=METRIC_KEYS[phase],
                )
            )

    _write_csv(args.output_dir / "per_run_results.csv", all_run)
    _write_csv(args.output_dir / "per_scenario_results.csv", all_scenario)
    _write_csv(args.output_dir / "method_k_summary.csv", method_k_summary)
    _write_csv(args.output_dir / "receiver_k_summary.csv", receiver_k_summary)
    _write_csv(args.output_dir / "paired_deltas_vs_cvs.csv", all_paired)
    _write_csv(args.output_dir / "paired_delta_summary.csv", all_paired_summary)
    (args.output_dir / "incomplete_rows.json").write_text(
        json.dumps(all_incomplete, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "cvs_publication_stage2_summary_v1",
        "scenarios": SCENARIOS,
        "expected_rows_per_phase": 500,
        "per_run_row_count": len(all_run),
        "per_scenario_row_count": len(all_scenario),
        "incomplete_row_count": len(all_incomplete),
        "allow_incomplete": bool(args.allow_incomplete),
        "reference_methods": REFERENCE_METHOD,
        "confidence_interval": "normal_approximation_1.96_standard_errors_over_independent_receiver_seed_runs",
    }
    (args.output_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
