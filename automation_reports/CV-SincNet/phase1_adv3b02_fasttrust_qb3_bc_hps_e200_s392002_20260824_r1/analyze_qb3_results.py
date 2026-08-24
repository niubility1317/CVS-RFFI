#!/usr/bin/env python3
"""Parse the complete QB3 E200 logs and final evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CANDIDATES = [
    "E200_C0_BC_NO_U_ID",
    "E200_C1_BC_H",
    "E200_C2_BC_H_PSET",
    "E200_C3_BC_H_PSET_PCOND",
    "E200_C4_BC_U_FEATURE_ANCHOR",
]

SCENARIO_FILES = {
    "clean": "metrics_clean.json",
    "leo_clear_weak": "metrics_leo_clear_weak.json",
    "leo_low_elev_weak": "metrics_leo_low_elev_weak.json",
    "leo_rain_weak": "metrics_leo_rain_weak.json",
}

CURVE_KEYS = [
    "epoch_time_s",
    "lr",
    "train_loss",
    "train_tx_acc",
    "val_tx_acc",
    "test_tx_acc",
    "stage_source_val_sat_mean_tx",
    "stage_source_val_sat_floor_tx",
    "stage_source_val_sat_receiver_floor",
    "protected_overall_tx",
    "protected_strict_udu",
    "protected_receiver_floor",
    "protected_sat_mean_tx",
    "protected_sat_floor_tx",
    "protected_sat_receiver_floor",
    "train_loss_adv_confusion_labeled",
    "train_loss_adv_discriminator_labeled",
    "train_loss_u_adv",
    "train_rc4_domain_confusion_loss",
    "train_rc4_domain_discriminator_loss",
    "train_rc4_hard_count",
    "train_rc4_partial_count",
    "train_rc4_representation_count",
    "train_rc4_hard_effective_coverage",
    "train_rc4_partial_effective_coverage",
    "train_rc4_effective_weighted_coverage",
    "train_rc4_candidate_size_mean",
    "train_rc4_risk_mean",
    "train_rc4_estimated_error_mean",
    "train_rc4_p_correct_mean",
    "train_rc4_p_set_safe_mean",
    "train_rc4_p_exclusion_safe_mean",
    "train_rc4_partial_safety_threshold",
    "train_rc4_partial_set_loss",
    "train_rc4_partial_conditional_loss",
    "train_rc4_feature_anchor_loss",
    "train_rc4_hard_tail_scale",
    "train_rc4_partial_set_tail_scale",
    "train_rc4_partial_conditional_tail_scale",
    "train_rc4_components_finite",
    "train_skipped_nonfinite_loss",
    "train_skipped_nonfinite_grad",
    "nonfinite_train_metric_count",
    "nonfinite_val_metric_count",
    "nonfinite_test_metric_count",
]

SNAPSHOT_EPOCHS = [1, 10, 11, 40, 41, 90, 91, 100, 130, 160, 161, 180, 181, 190, 200]
SEGMENTS = [(1, 10), (11, 40), (41, 90), (91, 160), (161, 180), (181, 200)]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def finite_numbers(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield prefix, float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from finite_numbers(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from finite_numbers(item, f"{prefix}[{index}]")


def numeric_values(records: list[dict[str, Any]], key: str, lo: int = 1, hi: int = 200) -> list[float]:
    result: list[float] = []
    for record in records:
        epoch = int(record["epoch"])
        value = record.get(key)
        if lo <= epoch <= hi and isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                result.append(float(value))
    return result


def metric_stats(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    pairs = [
        (int(record["epoch"]), float(record[key]))
        for record in records
        if isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
        and math.isfinite(float(record[key]))
    ]
    if not pairs:
        return None
    best_max = max(pairs, key=lambda item: item[1])
    best_min = min(pairs, key=lambda item: item[1])
    return {
        "first": pairs[0],
        "last": pairs[-1],
        "max": best_max,
        "min": best_min,
        "mean": statistics.fmean(value for _, value in pairs),
    }


def first_nonzero_epoch(records: list[dict[str, Any]], key: str) -> int | None:
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and abs(float(value)) > 1e-12:
            return int(record["epoch"])
    return None


def positive_metric_epochs(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {"epoch": int(record["epoch"]), "value": float(record[key])}
        for record in records
        if isinstance(record.get(key), (int, float))
        and not isinstance(record.get(key), bool)
        and math.isfinite(float(record[key]))
        and float(record[key]) > 0.0
    ]


def scan_log(path: Path) -> dict[str, Any]:
    patterns = {
        "traceback": re.compile(r"Traceback", re.I),
        "oom": re.compile(r"out of memory|CUDA OOM", re.I),
        "killed": re.compile(r"\bKilled\b|SIGKILL", re.I),
        "runtime_error": re.compile(r"RuntimeError", re.I),
        "nan_inf": re.compile(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", re.I),
        "nonfinite": re.compile(r"non[-_ ]?finite", re.I),
        "warning": re.compile(r"warning|\[WARN", re.I),
        "anomaly": re.compile(r"anomaly|skipped_nonfinite", re.I),
    }
    counts = Counter()
    first: dict[str, dict[str, Any]] = {}
    line_count = 0
    for line_count, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        for name, pattern in patterns.items():
            if pattern.search(line):
                counts[name] += 1
                first.setdefault(name, {"line": line_count, "text": line[:500]})
    return {"lines": line_count, "counts": dict(counts), "first": first}


def anomaly_metadata(path: Path) -> dict[str, Any]:
    def safe_scalar(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value

    try:
        import torch

        packet = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "schema": packet.get("schema"),
            "candidate_id": packet.get("candidate_id"),
            "epoch": packet.get("epoch"),
            "batch_index": packet.get("batch_index"),
            "loss_finite": packet.get("loss_finite"),
            "loss": safe_scalar(packet.get("loss")),
            "grad_before_clip": safe_scalar(packet.get("grad_before_clip")),
            "grad_total": safe_scalar(packet.get("grad_total")),
            "skipped_nonfinite_loss": packet.get("skipped_nonfinite_loss"),
            "skipped_nonfinite_grad": packet.get("skipped_nonfinite_grad"),
            "route_counts": packet.get("rc4_route_counts"),
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"load_error": f"{type(exc).__name__}: {exc}"}


def parse_candidate(root: Path, candidate: str) -> dict[str, Any]:
    directory = root / candidate
    records = [json.loads(line) for line in (directory / "metrics_epoch.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    epochs = [int(record["epoch"]) for record in records]
    nonfinite_values: list[dict[str, Any]] = []
    for record in records:
        for key, value in finite_numbers(record):
            if not math.isfinite(value):
                nonfinite_values.append({"epoch": int(record["epoch"]), "key": key, "value": repr(value)})

    evals: dict[str, Any] = {}
    all_rx_scenario: list[float] = []
    leo_rx_scenario: list[float] = []
    for scenario, filename in SCENARIO_FILES.items():
        data = load_json(directory / filename)
        aggregate = data["aggregate"]
        rows = data["rows"]
        row_map = {str(row["rx_idx"]): float(row["tx_acc"]) for row in rows}
        values = list(row_map.values())
        all_rx_scenario.extend(values)
        if scenario != "clean":
            leo_rx_scenario.extend(values)
        audit = data.get("reconstruction_audit", {})
        evals[scenario] = {
            "accuracy": float(aggregate["tx_acc"]),
            "correct": int(aggregate["tx_correct"]),
            "total": int(aggregate["tx_total"]),
            "receiver_floor": min(values),
            "receiver_ceiling": max(values),
            "per_receiver": row_map,
            "checkpoint_epoch": data.get("checkpoint_epoch"),
            "strict_reconstruction": bool(audit.get("strict_requested"))
            and bool(audit.get("checkpoint_load_strict"))
            and not bool(audit.get("fallback_used"))
            and int(audit.get("missing_keys", -1)) == 0
            and int(audit.get("unexpected_keys", -1)) == 0
            and int(audit.get("shape_mismatches", -1)) == 0,
        }

    leo_accuracies = [evals[name]["accuracy"] for name in SCENARIO_FILES if name != "clean"]
    epoch_times = numeric_values(records, "epoch_time_s")
    curve_stats = {key: stats for key in CURVE_KEYS if (stats := metric_stats(records, key)) is not None}
    snapshots = {
        str(epoch): {
            key: records[epoch - 1].get(key)
            for key in CURVE_KEYS
            if key in records[epoch - 1]
        }
        for epoch in SNAPSHOT_EPOCHS
    }
    segments: dict[str, Any] = {}
    segment_keys = [
        "epoch_time_s",
        "train_loss",
        "val_tx_acc",
        "stage_source_val_sat_mean_tx",
        "stage_source_val_sat_floor_tx",
        "stage_source_val_sat_receiver_floor",
        "protected_overall_tx",
        "protected_strict_udu",
        "protected_receiver_floor",
        "protected_sat_mean_tx",
        "protected_sat_floor_tx",
        "train_rc4_hard_count",
        "train_rc4_partial_count",
        "train_rc4_representation_count",
        "train_rc4_hard_effective_coverage",
        "train_rc4_partial_effective_coverage",
        "train_rc4_effective_weighted_coverage",
        "train_rc4_p_correct_mean",
        "train_rc4_p_set_safe_mean",
        "train_rc4_risk_mean",
        "train_rc4_partial_set_loss",
        "train_rc4_partial_conditional_loss",
        "train_rc4_feature_anchor_loss",
    ]
    for lo, hi in SEGMENTS:
        segment = {}
        for key in segment_keys:
            values = numeric_values(records, key, lo, hi)
            if values:
                segment[key] = statistics.fmean(values)
        segments[f"E{lo}-E{hi}"] = segment

    rc4_keys = sorted({key for record in records for key in record if "rc4" in key.lower()})
    first_active = {
        key: epoch
        for key in rc4_keys
        if (epoch := first_nonzero_epoch(records, key)) is not None
    }
    terminal = load_json(directory / "phase1_terminal_status.json")
    resource = load_json(directory / "phase1_resource_summary.json")
    status = (directory / "status.txt").read_text(encoding="utf-8-sig").strip()
    return {
        "status": status,
        "epochs": {
            "records": len(records),
            "sequence_complete": epochs == list(range(1, 201)),
            "first": epochs[0],
            "last": epochs[-1],
        },
        "structured_nonfinite_values": nonfinite_values,
        "positive_metric_epochs": {
            key: positive_metric_epochs(records, key)
            for key in [
                "train_skipped_nonfinite_loss",
                "train_skipped_nonfinite_grad",
                "nonfinite_train_metric_count",
                "nonfinite_val_metric_count",
                "nonfinite_test_metric_count",
            ]
        },
        "curve_stats": curve_stats,
        "snapshots": snapshots,
        "segments": segments,
        "first_active_epoch": first_active,
        "epoch_time": {
            "sum_seconds": sum(epoch_times),
            "mean_seconds": statistics.fmean(epoch_times),
            "median_seconds": statistics.median(epoch_times),
            "p95_seconds": sorted(epoch_times)[max(0, math.ceil(0.95 * len(epoch_times)) - 1)],
            "min_seconds": min(epoch_times),
            "max_seconds": max(epoch_times),
        },
        "resource": resource,
        "terminal": terminal,
        "final_eval": {
            "scenarios": evals,
            "leo_mean": statistics.fmean(leo_accuracies),
            "leo_scenario_floor": min(leo_accuracies),
            "leo_receiver_scenario_floor": min(leo_rx_scenario),
            "four_scenario_receiver_floor": min(all_rx_scenario),
        },
        "train_log": scan_log(directory / "train.log"),
        "dispatcher_log": scan_log(directory / f"{candidate}.log"),
        "anomaly": anomaly_metadata(directory / "first_rc4_anomaly.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = {candidate: parse_candidate(args.input_root, candidate) for candidate in CANDIDATES}
    control = candidates[CANDIDATES[0]]["final_eval"]
    comparison = {}
    for candidate, data in candidates.items():
        final_eval = data["final_eval"]
        comparison[candidate] = {
            "clean": final_eval["scenarios"]["clean"]["accuracy"],
            "leo_clear_weak": final_eval["scenarios"]["leo_clear_weak"]["accuracy"],
            "leo_low_elev_weak": final_eval["scenarios"]["leo_low_elev_weak"]["accuracy"],
            "leo_rain_weak": final_eval["scenarios"]["leo_rain_weak"]["accuracy"],
            "leo_mean": final_eval["leo_mean"],
            "leo_scenario_floor": final_eval["leo_scenario_floor"],
            "leo_receiver_scenario_floor": final_eval["leo_receiver_scenario_floor"],
            "four_scenario_receiver_floor": final_eval["four_scenario_receiver_floor"],
            "delta_vs_c0": {
                "clean": final_eval["scenarios"]["clean"]["accuracy"] - control["scenarios"]["clean"]["accuracy"],
                "leo_mean": final_eval["leo_mean"] - control["leo_mean"],
                "leo_scenario_floor": final_eval["leo_scenario_floor"] - control["leo_scenario_floor"],
                "leo_receiver_scenario_floor": final_eval["leo_receiver_scenario_floor"] - control["leo_receiver_scenario_floor"],
                "four_scenario_receiver_floor": final_eval["four_scenario_receiver_floor"] - control["four_scenario_receiver_floor"],
            },
        }

    output = {
        "schema": "cvs.phase1.fasttrust_qb3_full_analysis.v1",
        "input_root": str(args.input_root),
        "candidates": candidates,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(output), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
