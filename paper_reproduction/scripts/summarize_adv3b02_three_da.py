"""Audit and summarize the 375-row shared-ADV3B02 Stage2-B DA matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHODS = ("protonet_cda", "mrior_sda", "dadda_sda")
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
KS = (1, 2, 5, 10, 20)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data or not all(math.isfinite(value) for value in data):
        raise FloatingPointError(f"invalid metric group: {data[:5]}")
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "n": len(data), "mean": mean, "std": std,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "min": min(data), "max": max(data),
    }


def _group(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    checkpoint_shas: set[str] = set()
    score_total = 0
    for metrics_path in sorted(args.run_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "adv3b02_stage2b_supervised_da_v2":
            continue
        method = str(payload["method_id"])
        receiver = str(payload["target_receiver_label"])
        k_shot = int(payload["k_shot"])
        seed = int(payload["seed"])
        manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        trace = json.loads((run_dir / "loss_trace.json").read_text(encoding="utf-8"))
        detailed = json.loads((run_dir / "detailed_metrics.json").read_text(encoding="utf-8"))
        with (run_dir / "score_table.csv").open("r", encoding="utf-8", newline="") as handle:
            scores = list(csv.DictReader(handle))
        score_total += len(scores)
        expected_trace = 3 if method == "protonet_cda" else 33
        checks = {
            "method": method in METHODS,
            "k": k_shot in KS,
            "strict": manifest.get("checkpoint_load_strict") is True,
            "zero_load_audit": all(
                int(manifest.get("checkpoint_load_audit", {}).get(key, -1)) == 0
                for key in ("missing_keys", "unexpected_keys", "shape_mismatch")
            ),
            "no_overlap": manifest.get("support_query_overlap") is False,
            "no_query_training": manifest.get("target_query_used_for_training") is False,
            "no_query_selection": manifest.get("target_query_used_for_model_selection") is False,
            "score_count": len(scores) == 360,
            "trace_count": len(trace) == expected_trace,
            "scenarios": tuple(payload["metrics_by_scenario"]) == SCENARIOS,
            "leo_weak_only": manifest.get("phase2_sample_view_policy")
            == "leo_weak_only_no_clean_access",
            "no_clean_sample": manifest.get("clean_sample_access") is False,
            "no_clean_derived": manifest.get("clean_derived_signal_access") is False,
            "no_clean_dataset_reachability": manifest.get("phase2_clean_dataset_reachable")
            is False,
            "no_clean_cache_reachability": manifest.get("phase2_clean_cache_reachable")
            is False,
            "no_clean_control_flow": manifest.get("phase2_clean_control_flow_reachable")
            is False,
            "sealed_phase1_checkpoint": manifest.get("phase2_pretrained_artifact_policy")
            == "sealed_phase1_checkpoint_only",
            "overlay_before_phase2": manifest.get("overlay_applied_before_phase2") is True,
            "per_sample_query": manifest.get("phase2_query_decision_policy")
            == "per_sample_all_registered_classes",
            "no_role_oracle": manifest.get("phase2_query_role_oracle_access") is False,
            "no_true_batch_class_count": manifest.get("phase2_query_true_batch_class_count_access") is False,
            "no_class_quota": manifest.get("phase2_query_class_quota_access") is False,
            "no_global_assignment": manifest.get("phase2_query_batch_global_assignment")
            is False,
            "target_cache_verified": isinstance(
                manifest.get("target_leo_weak_cache_audit"), dict
            ),
        }
        if method == "protonet_cda":
            checks["source_cache_not_opened"] = (
                manifest.get("source_leo_weak_cache_used") is False
                and manifest.get("source_cache_declared_but_not_opened") is True
            )
        else:
            checks["source_cache_verified"] = (
                manifest.get("source_leo_weak_cache_used") is True
                and isinstance(manifest.get("source_leo_weak_cache_audit"), dict)
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{run_dir}:{failed}")
        checkpoint_shas.add(str(manifest.get("checkpoint_sha256", "")))
        for trace_row in trace:
            try:
                loss = float(trace_row["loss"])
            except Exception as exc:
                errors.append(f"{run_dir}:invalid_loss:{exc!r}")
                continue
            if not math.isfinite(loss):
                errors.append(f"{run_dir}:nonfinite_loss")
            trace_rows.append({
                "method": method, "receiver": receiver, "k_shot": k_shot, "seed": seed,
                **trace_row,
            })
        metrics = payload["metrics"]
        rows.append({
            "method": method, "receiver": receiver, "k_shot": k_shot, "seed": seed,
            "experiment_id": payload["experiment_id"], "run_dir": str(run_dir),
            "target_old_accuracy": float(metrics["target_old_accuracy_mean"]),
            "before_accuracy": float(metrics["target_old_accuracy_before_adaptation_mean"]),
            "delta": float(metrics["target_old_accuracy_delta_mean"]),
            "adaptation_latency_sec": float(metrics["adaptation_latency_sec_mean"]),
            "adv3b02_gradient_updates": int(manifest["adv3b02_gradient_updates"]),
        })
        for scenario, values in payload["metrics_by_scenario"].items():
            scenario_rows.append({
                "method": method, "receiver": receiver, "k_shot": k_shot, "seed": seed,
                "scenario": scenario,
                "target_old_accuracy": float(values["target_old_accuracy"]),
                "before_accuracy": float(values["target_old_accuracy_before_adaptation"]),
                "delta": float(values["target_old_accuracy_delta"]),
            })
        for detail in detailed:
            if detail.get("group_type") == "per_transmitter":
                class_rows.append({
                    "method": method, "receiver": receiver, "k_shot": k_shot, "seed": seed,
                    "scenario": detail["scenario"],
                    "transmitter": detail["transmitter_label"],
                    "accuracy": float(detail["accuracy"]),
                })

    expected_rows = len(METHODS) * 5 * 5 * 5
    if len(rows) != expected_rows:
        errors.append(f"row_count={len(rows)},expected={expected_rows}")
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if len(method_rows) != 125:
            errors.append(f"{method}:row_count={len(method_rows)}")
        for k_shot in KS:
            count = sum(row["k_shot"] == k_shot for row in method_rows)
            if count != 25:
                errors.append(f"{method}:k={k_shot}:count={count}")
    if len(checkpoint_shas) != 1 or "" in checkpoint_shas:
        errors.append(f"checkpoint_sha_set={sorted(checkpoint_shas)}")
    paired_before: dict[tuple[str, int, int], set[float]] = defaultdict(set)
    for row in rows:
        paired_before[(row["receiver"], row["k_shot"], row["seed"])].add(row["before_accuracy"])
    inconsistent = [key for key, values in paired_before.items() if len(values) != 1]
    if inconsistent:
        errors.append(f"paired_before_inconsistent={inconsistent[:5]}")

    method_k: list[dict[str, Any]] = []
    for (method, k_shot), values in sorted(_group(rows, ("method", "k_shot")).items()):
        item: dict[str, Any] = {"method": method, "k_shot": k_shot}
        for metric in ("target_old_accuracy", "before_accuracy", "delta", "adaptation_latency_sec"):
            item.update({f"{metric}_{key}": value for key, value in _stats(row[metric] for row in values).items()})
        item["wins"] = sum(row["delta"] > 0 for row in values)
        item["ties"] = sum(row["delta"] == 0 for row in values)
        item["losses"] = sum(row["delta"] < 0 for row in values)
        method_k.append(item)

    method_overall: list[dict[str, Any]] = []
    for (method,), values in sorted(_group(rows, ("method",)).items()):
        item = {"method": method}
        for metric in ("target_old_accuracy", "before_accuracy", "delta", "adaptation_latency_sec"):
            item.update({f"{metric}_{key}": value for key, value in _stats(row[metric] for row in values).items()})
        item["wins"] = sum(row["delta"] > 0 for row in values)
        item["ties"] = sum(row["delta"] == 0 for row in values)
        item["losses"] = sum(row["delta"] < 0 for row in values)
        method_overall.append(item)

    receiver_summary: list[dict[str, Any]] = []
    for (method, receiver), values in sorted(_group(rows, ("method", "receiver")).items()):
        receiver_summary.append({
            "method": method, "receiver": receiver,
            **{f"target_old_accuracy_{key}": value for key, value in _stats(row["target_old_accuracy"] for row in values).items()},
            **{f"delta_{key}": value for key, value in _stats(row["delta"] for row in values).items()},
        })

    scenario_summary: list[dict[str, Any]] = []
    for (method, scenario), values in sorted(_group(scenario_rows, ("method", "scenario")).items()):
        scenario_summary.append({
            "method": method, "scenario": scenario,
            **{f"target_old_accuracy_{key}": value for key, value in _stats(row["target_old_accuracy"] for row in values).items()},
            **{f"delta_{key}": value for key, value in _stats(row["delta"] for row in values).items()},
        })

    class_summary: list[dict[str, Any]] = []
    for (method, transmitter), values in sorted(_group(class_rows, ("method", "transmitter")).items()):
        class_summary.append({
            "method": method, "transmitter": transmitter,
            **{f"accuracy_{key}": value for key, value in _stats(row["accuracy"] for row in values).items()},
        })

    loss_summary: list[dict[str, Any]] = []
    for (method,), values in sorted(_group(trace_rows, ("method",)).items()):
        item: dict[str, Any] = {"method": method, "trace_count": len(values)}
        for metric in ("loss", "dvkl", "estimate_zeta", "mmd", "lmmd", "alpha", "source_ce", "target_support_ce"):
            present = [float(row[metric]) for row in values if row.get(metric, "") not in ("", None)]
            if present:
                item.update({f"{metric}_{key}": value for key, value in _stats(present).items()})
        loss_summary.append(item)

    for filename, values in (
        ("per_run_results.csv", rows), ("per_scenario_results.csv", scenario_rows),
        ("method_k_summary.csv", method_k), ("method_overall_summary.csv", method_overall),
        ("receiver_summary.csv", receiver_summary), ("scenario_summary.csv", scenario_summary),
        ("class_summary.csv", class_summary), ("loss_summary.csv", loss_summary),
    ):
        _write_csv(args.output_dir / filename, values)
    audit = {
        "schema": "adv3b02_three_da_audit_v1", "artifact_complete": not errors,
        "row_count": len(rows), "score_row_count": score_total,
        "trace_row_count": len(trace_rows), "checkpoint_sha256": sorted(checkpoint_shas),
        "errors": errors, "method_counts": {
            method: sum(row["method"] == method for row in rows) for method in METHODS
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
