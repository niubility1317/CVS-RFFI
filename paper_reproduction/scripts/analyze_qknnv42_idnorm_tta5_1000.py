#!/usr/bin/env python3
"""Audit and summarize the completed qKNN id_norm TTA5 1000-task matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EPOCHS = (1, 2, 5, 10, 20, 30, 60)
ARMS = ("singlehead_fft96",) + tuple(f"E{value}_idnorm_tta5" for value in EPOCHS)
EXPECTED_TASKS_PER_ARM = 125
EXPECTED_ADAPTERS = 875
EXPECTED_TOTAL_LOSS_ROWS = sum(EPOCHS) * EXPECTED_TASKS_PER_ARM


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("mean requires at least one value")
    return sum(items) / len(items)


def _sample_std(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if len(items) <= 1:
        return 0.0
    center = _mean(items)
    return math.sqrt(sum((value - center) ** 2 for value in items) / (len(items) - 1))


def _paired_ci95(values: list[float]) -> float:
    return 1.96 * _sample_std(values) / math.sqrt(max(1, len(values)))


def collect_results(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("metrics.json")):
        relative = path.relative_to(results_root)
        arm = relative.parts[0]
        if arm not in ARMS:
            continue
        result = _read_json(path)
        resolved = _read_json(path.parent / "resolved_config.json")
        manifest = _read_json(path.parent / "split_manifest.json")
        metrics = result["metrics"]
        scenario_metrics = result["metrics_by_scenario"]
        for scenario, detail in scenario_metrics.items():
            forbidden = {
                key: detail.get(key)
                for key in (
                    "feature_adapter_uses_query",
                    "query_labels_used_for_adaptation",
                    "query_query_graph_used",
                    "query_batch_state_required",
                    "role_oracle_used",
                    "equal_class_quota_used",
                    "decision_batch_state_required",
                )
                if detail.get(key) is not False
            }
            if forbidden:
                raise ValueError(f"forbidden query/Oracle state in {path}:{scenario}:{forbidden}")
        expected_views = 1 if arm == "singlehead_fft96" else 5
        checks = {
            "support_query_overlap": manifest.get("support_query_overlap") is False,
            "per_sample": manifest.get("qknnv42_decision_mode") == "per_sample_argmax",
            "no_dense": manifest.get("qknnv42_labelprop_mode") == "disabled",
            "no_oracle": manifest.get("non_deployment_oracle_diagnostic") is False,
            "views": int(manifest.get("satellite_tta_view_count", -1)) == expected_views,
            "leo_only": manifest.get("phase2_sample_view_policy")
            == "leo_weak_only_no_clean_access",
            "no_clean": manifest.get("clean_sample_access") is False,
            "resource_claim": manifest.get("resource_diagnostic_only")
            is (arm != "singlehead_fft96"),
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"protocol audit failed for {path}: {failed}")
        receiver = str(result["target_receiver_label"])
        seed = int(result["seed"])
        k_shot = int(manifest["k_shot"])
        epochs = 0 if arm == "singlehead_fft96" else int(arm.split("_", 1)[0][1:])
        row = {
            "task_id": str(result["experiment_id"]),
            "arm": arm,
            "epochs": epochs,
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
            "old_class_count": len(resolved["target_old_tx_labels"]),
            "new_class_count": len(resolved["target_new_tx_labels"]),
            "old_acc": float(metrics["old_acc_mean"]),
            "new_acc": float(metrics["seen_new_acc_mean"]),
            "H": float(metrics["H_old_new_mean"]),
            "average_forgetting": float(metrics["average_forgetting_mean"]),
            "min_old_class_acc": float(metrics["min_old_class_acc"]),
            "min_new_class_acc": float(metrics["min_seen_new_class_acc"]),
            "query_views": expected_views,
            "trainable_parameters": 0,
            "adapter_state_bytes_fp16": 0,
            "resource_tier": "baseline",
            "metrics_json": str(path),
            "metrics_sha256": _sha256(path),
        }
        if arm != "singlehead_fft96":
            details = list(scenario_metrics.values())
            if any(int(detail.get("post_feature_adapter_parameter_count", -1)) != 289_685 for detail in details):
                raise ValueError(f"adapter parameter provenance drift: {path}")
            row.update(
                {
                    "trainable_parameters": 289_685,
                    "adapter_state_bytes_fp16": 579_370,
                    "resource_tier": "non_extreme_light_large_adapter_diagnostic",
                }
            )
        rows.append(row)
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 result rows, found {len(rows)}")
    counts = {arm: sum(row["arm"] == arm for row in rows) for arm in ARMS}
    if any(count != EXPECTED_TASKS_PER_ARM for count in counts.values()):
        raise ValueError(f"arm counts are incomplete: {counts}")
    return rows


def collect_training(adapters_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapters: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for path in sorted(adapters_root.rglob("training_manifest.json")):
        manifest = _read_json(path)
        if manifest.get("method") != "support_only_id_norm_late_feature_tta5_v1":
            continue
        contract = manifest.get("optimizer_sample_contract", {})
        resources = manifest.get("resources", {})
        audit = manifest.get("checkpoint_load_audit", {})
        checks = {
            "exact_params": int(resources.get("trainable_parameters", -1)) == 289_685,
            "exact_checkpoint_params": int(resources.get("original_checkpoint_trainable_parameters", -1)) == 289_685,
            "fp16_bytes": int(resources.get("adapter_state_bytes_fp16", -1)) == 579_370,
            "tta5": int(manifest.get("query_view_count", -1)) == 5,
            "support_only": manifest.get("support_only") is True,
            "leo_only": manifest.get("phase2_sample_view_policy") == "leo_weak_only_no_clean_access",
            "no_clean_access": manifest.get("clean_sample_access") is False,
            "no_clean": contract.get("clean_samples_used") is False,
            "no_source": contract.get("source_samples_used") is False,
            "no_proxy": contract.get("proxy_samples_used") is False,
            "no_query": contract.get("query_samples_used") is False,
            "strict_missing": int(audit.get("missing_keys", -1)) == 0,
            "strict_unexpected": int(audit.get("unexpected_keys", -1)) == 0,
            "strict_mismatch": int(audit.get("skipped_mismatch", -1)) == 0,
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"training provenance audit failed for {path}: {failed}")
        epochs = int(manifest["epochs"])
        trace = _read_json(Path(manifest["loss_trace_json"]))
        if len(trace) != epochs:
            raise ValueError(f"incomplete trace for {path}: {len(trace)} != {epochs}")
        arm = f"E{epochs}_idnorm_tta5"
        adapter_row = {
            "adapter_id": path.parent.name,
            "arm": arm,
            "epochs": epochs,
            "receiver": str(manifest["receiver"]),
            "seed": int(manifest["seed"]),
            "k_shot": int(manifest["k_shot"]),
            "physical_support_count": int(contract["physical_support_count"]),
            "support_view_count": int(contract["support_view_count"]),
            "gradient_updates": int(manifest["runtime"]["gradient_updates"]),
            "adaptation_wall_seconds": float(manifest["runtime"]["adaptation_wall_seconds"]),
            "peak_cuda_memory_bytes": int(manifest["runtime"]["peak_cuda_memory_bytes"]),
            "first_loss": float(trace[0]["loss"]),
            "last_loss": float(trace[-1]["loss"]),
            "first_support_acc": float(trace[0]["support_train_acc"]),
            "last_support_acc": float(trace[-1]["support_train_acc"]),
            "manifest_json": str(path),
            "manifest_sha256": _sha256(path),
        }
        adapters.append(adapter_row)
        loss_rows.extend(
            {"adapter_id": path.parent.name, "arm": arm, **trace_row}
            for trace_row in trace
        )
    if len(adapters) != EXPECTED_ADAPTERS:
        raise ValueError(f"expected {EXPECTED_ADAPTERS} adapters, found {len(adapters)}")
    if len(loss_rows) != EXPECTED_TOTAL_LOSS_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_LOSS_ROWS} loss rows, found {len(loss_rows)}"
        )
    return adapters, loss_rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {
        (row["receiver"], row["seed"], row["k_shot"]): row
        for row in rows
        if row["arm"] == "singlehead_fft96"
    }
    arm_summary: list[dict[str, Any]] = []
    by_k: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        deltas = [
            row["H"] - baseline[(row["receiver"], row["seed"], row["k_shot"])]["H"]
            for row in arm_rows
        ]
        arm_summary.append(
            {
                "arm": arm,
                "epochs": arm_rows[0]["epochs"],
                "task_count": len(arm_rows),
                "old_acc_mean": _mean(row["old_acc"] for row in arm_rows),
                "new_acc_mean": _mean(row["new_acc"] for row in arm_rows),
                "H_mean": _mean(row["H"] for row in arm_rows),
                "forgetting_mean": _mean(row["average_forgetting"] for row in arm_rows),
                "paired_delta_H_mean": _mean(deltas),
                "paired_delta_H_ci95_halfwidth": _paired_ci95(deltas),
                "query_views": arm_rows[0]["query_views"],
                "trainable_parameters": arm_rows[0]["trainable_parameters"],
                "adapter_state_bytes_fp16": arm_rows[0]["adapter_state_bytes_fp16"],
                "resource_tier": arm_rows[0]["resource_tier"],
            }
        )
        for k_shot in (1, 2, 5, 10, 20):
            group = [row for row in arm_rows if row["k_shot"] == k_shot]
            k_delta = [
                row["H"] - baseline[(row["receiver"], row["seed"], row["k_shot"])]["H"]
                for row in group
            ]
            by_k.append(
                {
                    "arm": arm,
                    "epochs": group[0]["epochs"],
                    "k_shot": k_shot,
                    "task_count": len(group),
                    "old_acc_mean": _mean(row["old_acc"] for row in group),
                    "new_acc_mean": _mean(row["new_acc"] for row in group),
                    "H_mean": _mean(row["H"] for row in group),
                    "forgetting_mean": _mean(row["average_forgetting"] for row in group),
                    "paired_delta_H_mean": _mean(k_delta),
                    "paired_delta_H_ci95_halfwidth": _paired_ci95(k_delta),
                }
            )
    return arm_summary, by_k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--adapters-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_results(args.results_root)
    adapters, loss_rows = collect_training(args.adapters_root)
    arm_summary, by_k = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "task_rows.csv", rows)
    _write_csv(args.output_dir / "adapter_rows.csv", adapters)
    _write_csv(args.output_dir / "loss_rows.csv", loss_rows)
    _write_csv(args.output_dir / "arm_summary.csv", arm_summary)
    _write_csv(args.output_dir / "epoch_by_k.csv", by_k)
    payload = {
        "audit_status": "PASS",
        "result_task_count": len(rows),
        "adapter_count": len(adapters),
        "loss_row_count": len(loss_rows),
        "expected_total_loss_rows": EXPECTED_TOTAL_LOSS_ROWS,
        "arm_summary": arm_summary,
        "epoch_by_k": by_k,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
