#!/usr/bin/env python3
"""Audit and summarize the completed task-specific support-only qKNN 875 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from paper_reproduction.scripts.run_cvs_qknnv42_support_only_taskadapt_875 import (
    EPOCHS,
    EXPECTED_EVAL_FILES,
    K_GRID,
    SCENARIOS,
)


ARMS = ("singlehead_fft96",) + tuple(f"E{epoch}" for epoch in EPOCHS)
METRICS = (
    "old_before",
    "old_acc",
    "seen_new_acc",
    "H_old_new",
    "average_forgetting",
    "old_to_seen_new_rate",
    "seen_new_to_old_rate",
    "min_old_class_acc",
    "min_seen_new_class_acc",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows)


def ci95(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return 0.0 if len(rows) < 2 else 1.96 * statistics.stdev(rows) / math.sqrt(len(rows))


def _task_files_complete(run_dir: Path) -> bool:
    return all((run_dir / name).is_file() and (run_dir / name).stat().st_size > 0 for name in EXPECTED_EVAL_FILES)


def parse_matrix(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(run_root / "matrix_manifest.json")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    split_groups: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    training_split_groups: dict[tuple[str, str, int, int], set[str]] = {}
    completed_by_arm: Counter[str] = Counter()
    trace_epoch_rows = 0
    for task in manifest["tasks"]:
        arm = str(task["arm"])
        receiver = str(task["receiver"])
        seed = int(task["seed"])
        k_shot = int(task["k_shot"])
        epochs = int(task["epochs"])
        task_id = str(task["task_id"])
        run_dir = Path(task["eval_run_dir"])
        if not _task_files_complete(run_dir):
            errors.append(f"incomplete_eval:{task_id}:{run_dir}")
            continue
        payload = read_json(run_dir / "metrics.json")
        split = read_json(run_dir / "split_manifest.json")
        metrics = payload["metrics"]
        scenarios = payload["metrics_by_scenario"]
        required_split = {
            "support_query_overlap": False,
            "qknnv42_decision_mode": "per_sample_argmax",
            "qknnv42_labelprop_mode": "disabled",
            "non_deployment_oracle_diagnostic": False,
            "query_used_for_joint_decision": False,
            "query_used_for_transductive_inference": False,
            "target_query_used_for_training": False,
            "target_query_used_for_model_selection": False,
            "qknnv42_aux_feature_dim": 96,
            "satellite_tta_view_count": 1,
        }
        for key, expected in required_split.items():
            if split.get(key) != expected:
                errors.append(f"split_flag:{task_id}:{key}={split.get(key)!r}")
        scenario_values: list[dict[str, Any]] = []
        for scenario in SCENARIOS:
            detail = scenarios.get(scenario)
            if not isinstance(detail, dict):
                errors.append(f"missing_scenario:{task_id}:{scenario}")
                continue
            for key in (
                "feature_adapter_uses_query",
                "query_labels_used_for_adaptation",
                "query_query_graph_used",
                "query_batch_state_required",
                "role_oracle_used",
                "equal_class_quota_used",
                "decision_batch_state_required",
            ):
                if detail.get(key) is not False:
                    errors.append(f"forbidden_state:{task_id}:{scenario}:{key}={detail.get(key)!r}")
            if detail.get("aux_feature_enabled") is not True or int(detail.get("aux_feature_dim", -1)) != 96:
                errors.append(f"fft96:{task_id}:{scenario}")
            scenario_values.append(detail)
            item = split["splits_by_scenario"][scenario]
            support = set(item["support_sample_ids"])
            query = tuple(item["query_sample_ids"])
            if support & set(query):
                errors.append(f"support_query_overlap:{task_id}:{scenario}")
            split_groups[(arm, receiver, seed, k_shot, scenario)] = {
                "support": support,
                "query": query,
            }
        if len(scenario_values) != len(SCENARIOS):
            continue
        resources = {
            "adapter_parameters": 0,
            "adapter_state_bytes_fp16": 0,
            "adapter_macs_per_query": 0,
            "adaptation_wall_seconds": 0.0,
            "peak_cuda_memory_bytes": 0,
            "resource_tier": "baseline",
        }
        if epochs:
            adapter_dir = Path(task["adapter_run_dir"])
            training = read_json(adapter_dir / "training_manifest.json")
            trace = read_json(adapter_dir / "loss_trace.json")
            contract = training.get("optimizer_sample_contract", {})
            required_training = {
                "support_only": True,
                "query_update_forbidden": True,
                "query_labels_used_for_training": False,
                "old_new_role_used_by_optimizer": False,
                "class_quota_used_at_inference": False,
                "query_view_count": 1,
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "epochs": epochs,
                "new_count": 2,
            }
            for key, expected in required_training.items():
                if training.get(key) != expected:
                    errors.append(f"training_flag:{task_id}:{key}={training.get(key)!r}")
            if contract.get("roles") != ["target_old_support", "target_new_support"]:
                errors.append(f"training_roles:{task_id}:{contract.get('roles')!r}")
            for key in ("clean_samples_used", "source_samples_used", "proxy_samples_used", "query_samples_used"):
                if contract.get(key) is not False:
                    errors.append(f"forbidden_training_sample:{task_id}:{key}")
            if len(trace) != epochs:
                errors.append(f"trace_length:{task_id}:{len(trace)}!={epochs}")
            for index, item in enumerate(trace, 1):
                trace_epoch_rows += 1
                if int(item.get("epoch", -1)) != index:
                    errors.append(f"trace_epoch:{task_id}:{index}")
                for key in ("loss", "prototype_ce", "feature_anchor", "input_residual_mse"):
                    if not math.isfinite(float(item.get(key, math.nan))):
                        errors.append(f"trace_nonfinite:{task_id}:{index}:{key}")
            physical_support = set(training["split"]["physical_support_ids"])
            training_split_groups[(arm, receiver, seed, k_shot)] = physical_support
            for scenario in SCENARIOS:
                eval_support = split_groups[(arm, receiver, seed, k_shot, scenario)]["support"]
                if physical_support != eval_support:
                    errors.append(f"train_eval_support_mismatch:{task_id}:{scenario}")
            raw_resources = training["resources"]
            runtime = training["runtime"]
            resources = {
                "adapter_parameters": int(raw_resources["trainable_parameters"]),
                "adapter_state_bytes_fp16": int(raw_resources["adapter_state_bytes_fp16"]),
                "adapter_macs_per_query": int(raw_resources["adapter_macs_per_query"]),
                "adaptation_wall_seconds": float(runtime["adaptation_wall_seconds"]),
                "peak_cuda_memory_bytes": int(runtime["peak_cuda_memory_bytes"]),
                "resource_tier": str(training["resource_tier"]),
            }
        row = {
            "task_id": task_id,
            "arm": arm,
            "epochs": epochs,
            "resource_tier": resources["resource_tier"],
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
            "old_before": mean(item["old_acc_before_increment"] for item in scenario_values),
            "old_acc": float(metrics["old_acc_mean"]),
            "seen_new_acc": float(metrics["seen_new_acc_mean"]),
            "H_old_new": float(metrics["H_old_new_mean"]),
            "average_forgetting": float(metrics["average_forgetting_mean"]),
            "old_to_seen_new_rate": mean(item["old_to_seen_new_rate"] for item in scenario_values),
            "seen_new_to_old_rate": mean(item["seen_new_to_old_rate"] for item in scenario_values),
            "min_old_class_acc": float(metrics["min_old_class_acc"]),
            "min_seen_new_class_acc": float(metrics["min_seen_new_class_acc"]),
            **resources,
            "run_dir": str(run_dir),
        }
        if not all(math.isfinite(float(row[key])) for key in METRICS):
            errors.append(f"nonfinite_metric:{task_id}")
        rows.append(row)
        completed_by_arm[arm] += 1

    expected_trace_rows = 125 * sum(EPOCHS)
    if trace_epoch_rows != expected_trace_rows:
        errors.append(f"total_trace_rows:{trace_epoch_rows}!={expected_trace_rows}")
    for arm in ARMS:
        if completed_by_arm[arm] != 125:
            errors.append(f"arm_count:{arm}:{completed_by_arm[arm]}!=125")
    errors.extend(audit_splits(split_groups))
    return rows, {
        "task_count": len(rows),
        "completed_by_arm": dict(completed_by_arm),
        "trace_epoch_rows": trace_epoch_rows,
        "expected_trace_epoch_rows": expected_trace_rows,
        "errors": errors,
    }


def audit_splits(groups: dict[tuple[str, str, int, int, str], dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    task_bases = {(rx, seed, k, scenario) for _, rx, seed, k, scenario in groups}
    for rx, seed, k_shot, scenario in sorted(task_bases):
        entries = [groups.get((arm, rx, seed, k_shot, scenario)) for arm in ARMS]
        if any(entry is None for entry in entries):
            errors.append(f"missing_arm_split:{rx}:{seed}:K{k_shot}:{scenario}")
            continue
        if len({entry["query"] for entry in entries if entry}) != 1:
            errors.append(f"query_changed_across_arms:{rx}:{seed}:K{k_shot}:{scenario}")
        if len({tuple(sorted(entry["support"])) for entry in entries if entry}) != 1:
            errors.append(f"support_changed_across_arms:{rx}:{seed}:K{k_shot}:{scenario}")
    nested_bases = {(arm, rx, seed, scenario) for arm, rx, seed, _, scenario in groups}
    for arm, rx, seed, scenario in sorted(nested_bases):
        entries = {k: groups.get((arm, rx, seed, k, scenario)) for k in K_GRID}
        if any(entry is None for entry in entries.values()):
            errors.append(f"missing_k_split:{arm}:{rx}:{seed}:{scenario}")
            continue
        if len({entry["query"] for entry in entries.values() if entry}) != 1:
            errors.append(f"query_changed_across_k:{arm}:{rx}:{seed}:{scenario}")
        for low, high in zip(K_GRID, K_GRID[1:]):
            if not entries[low]["support"] <= entries[high]["support"]:
                errors.append(f"support_not_nested:{arm}:{rx}:{seed}:{scenario}:{low}->{high}")
        for k_shot, entry in entries.items():
            if len(entry["support"]) != 8 * k_shot or len(entry["query"]) != 160:
                errors.append(
                    f"split_count:{arm}:{rx}:{seed}:{scenario}:K{k_shot}:"
                    f"{len(entry['support'])}:{len(entry['query'])}"
                )
    return errors


def summarize(rows: list[dict[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        item = {field: value for field, value in zip(group_fields, key)}
        item["count"] = len(group)
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
            item[f"{metric}_ci95"] = ci95(values)
        for metric in (
            "adapter_parameters",
            "adapter_state_bytes_fp16",
            "adapter_macs_per_query",
            "adaptation_wall_seconds",
            "peak_cuda_memory_bytes",
        ):
            item[f"{metric}_mean"] = mean(float(row[metric]) for row in group)
        output.append(item)
    return output


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["arm"], row["receiver"], row["seed"], row["k_shot"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["arm"] == "singlehead_fft96":
            continue
        baseline = by_key[("singlehead_fft96", row["receiver"], row["seed"], row["k_shot"])]
        output.append(
            {
                "arm": row["arm"],
                "epochs": row["epochs"],
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "delta_old_acc_pp": 100.0 * (row["old_acc"] - baseline["old_acc"]),
                "delta_seen_new_acc_pp": 100.0 * (row["seen_new_acc"] - baseline["seen_new_acc"]),
                "delta_H_pp": 100.0 * (row["H_old_new"] - baseline["H_old_new"]),
                "delta_forgetting_pp": 100.0
                * (row["average_forgetting"] - baseline["average_forgetting"]),
            }
        )
    return output


def summarize_deltas(rows: list[dict[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    delta_metrics = (
        "delta_old_acc_pp",
        "delta_seen_new_acc_pp",
        "delta_H_pp",
        "delta_forgetting_pp",
    )
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        item = {field: value for field, value in zip(group_fields, key)}
        item["count"] = len(group)
        for metric in delta_metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
            item[f"{metric}_ci95"] = ci95(values)
        output.append(item)
    return output


def audit_logs(log_root: Path) -> dict[str, Any]:
    patterns = {
        "traceback": re.compile(r"Traceback"),
        "runtime_error": re.compile(r"RuntimeError"),
        "oom": re.compile(r"CUDA out of memory", re.I),
        "killed": re.compile(r"(?:^|\s)Killed(?:\s|$)", re.M),
        "nonfinite": re.compile(r"\b(?:nan|inf)\b", re.I),
        "task_failed": re.compile(r"\[TASK-FAILED\]"),
    }
    hits = {key: [] for key in patterns}
    file_count = byte_count = line_count = 0
    for path in sorted(log_root.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        file_count += 1
        byte_count += len(raw)
        line_count += len(text.splitlines())
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits[name].append(str(path))
    return {
        "file_count": file_count,
        "byte_count": byte_count,
        "line_count": line_count,
        "hit_counts": {key: len(value) for key, value in hits.items()},
        "hit_examples": {key: value[:10] for key, value in hits.items()},
    }


def collect_training_traces(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(run_root / "matrix_manifest.json")
    task_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        epochs = int(task["epochs"])
        if epochs == 0:
            continue
        training = read_json(Path(task["adapter_run_dir"]) / "training_manifest.json")
        trace = read_json(Path(task["adapter_run_dir"]) / "loss_trace.json")
        first, last = trace[0], trace[-1]
        task_rows.append(
            {
                "task_id": task["task_id"],
                "arm": task["arm"],
                "epochs": epochs,
                "receiver": task["receiver"],
                "seed": int(task["seed"]),
                "k_shot": int(task["k_shot"]),
                "loss_first": float(first["loss"]),
                "loss_last": float(last["loss"]),
                "loss_delta": float(last["loss"]) - float(first["loss"]),
                "prototype_ce_first": float(first["prototype_ce"]),
                "prototype_ce_last": float(last["prototype_ce"]),
                "feature_anchor_last": float(last["feature_anchor"]),
                "input_residual_mse_last": float(last["input_residual_mse"]),
                "support_train_acc_first": float(first["support_train_acc"]),
                "support_train_acc_last": float(last["support_train_acc"]),
                "support_train_acc_delta": float(last["support_train_acc"])
                - float(first["support_train_acc"]),
                "adaptation_wall_seconds": float(training["runtime"]["adaptation_wall_seconds"]),
                "peak_cuda_memory_bytes": int(training["runtime"]["peak_cuda_memory_bytes"]),
            }
        )
        for item in trace:
            epoch_rows.append(
                {
                    "task_id": task["task_id"],
                    "arm": task["arm"],
                    "receiver": task["receiver"],
                    "seed": int(task["seed"]),
                    "k_shot": int(task["k_shot"]),
                    **item,
                }
            )
    return task_rows, epoch_rows


def summarize_training(
    rows: list[dict[str, Any]], group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    metrics = (
        "loss_first",
        "loss_last",
        "loss_delta",
        "prototype_ce_first",
        "prototype_ce_last",
        "feature_anchor_last",
        "input_residual_mse_last",
        "support_train_acc_first",
        "support_train_acc_last",
        "support_train_acc_delta",
        "adaptation_wall_seconds",
        "peak_cuda_memory_bytes",
    )
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        item = {field: value for field, value in zip(group_fields, key)}
        item["count"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
            item[f"{metric}_ci95"] = ci95(values)
        item["loss_decreased_task_fraction"] = mean(
            float(row["loss_delta"]) < 0.0 for row in group
        )
        output.append(item)
    return output


def summarize_loss_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["arm"]), int(row["epoch"]))].append(row)
    output: list[dict[str, Any]] = []
    for (arm, epoch), group in sorted(groups.items()):
        output.append(
            {
                "arm": arm,
                "epoch": epoch,
                "count": len(group),
                "loss_mean": mean(row["loss"] for row in group),
                "prototype_ce_mean": mean(row["prototype_ce"] for row in group),
                "feature_anchor_mean": mean(row["feature_anchor"] for row in group),
                "input_residual_mse_mean": mean(row["input_residual_mse"] for row in group),
                "support_train_acc_mean": mean(row["support_train_acc"] for row in group),
                "epoch_seconds_mean": mean(row["epoch_seconds"] for row in group),
            }
        )
    return output


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def build_markdown(by_arm_k: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> str:
    delta_summary = summarize_deltas(deltas, ("arm", "k_shot")) if deltas else []
    delta_lookup = {(row["arm"], row["k_shot"]): row for row in delta_summary}
    lines = [
        "# qKNNV42逐任务support-only适应875任务主表",
        "",
        "类别协议：6个旧类+2个已注册新类。每格为5 receiver×5 seed=25次任务均值。",
        "",
        "| arm | K | old_before | old_acc | new_acc | H | 遗忘 | Δold | Δnew | ΔH |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_arm_k:
        delta = delta_lookup.get((row["arm"], row["k_shot"]))
        delta_cells = (
            ("—", "—", "—")
            if delta is None
            else (
                f"{delta['delta_old_acc_pp_mean']:+.2f}pp",
                f"{delta['delta_seen_new_acc_pp_mean']:+.2f}pp",
                f"{delta['delta_H_pp_mean']:+.2f}pp",
            )
        )
        lines.append(
            "| {arm} | {k} | {before} | {old} | {new} | {h} | {forget} | {do} | {dn} | {dh} |".format(
                arm=row["arm"],
                k=row["k_shot"],
                before=_pct(row["old_before_mean"]),
                old=_pct(row["old_acc_mean"]),
                new=_pct(row["seen_new_acc_mean"]),
                h=_pct(row["H_old_new_mean"]),
                forget=_pct(row["average_forgetting_mean"]),
                do=delta_cells[0],
                dn=delta_cells[1],
                dh=delta_cells[2],
            )
        )
    lines.extend(
        [
            "",
            "说明：Δ列均为与同receiver、同seed、同K的无adapter基线做配对后再求均值。E30为性能放宽档；E60为不可晋级资源控制档。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, protocol = parse_matrix(args.run_root)
    logs = audit_logs(args.log_root)
    by_arm_k = summarize(rows, ("arm", "k_shot"))
    by_arm = summarize(rows, ("arm",))
    by_receiver_arm_k = summarize(rows, ("receiver", "arm", "k_shot"))
    deltas = paired_deltas(rows)
    delta_by_arm_k = summarize_deltas(deltas, ("arm", "k_shot")) if deltas else []
    training_rows, epoch_rows = collect_training_traces(args.run_root)
    training_by_arm = summarize_training(training_rows, ("arm",))
    training_by_arm_k = summarize_training(training_rows, ("arm", "k_shot"))
    loss_curve = summarize_loss_curve(epoch_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "task_rows.csv", rows)
    write_csv(args.output_dir / "summary_by_arm_k.csv", by_arm_k)
    write_csv(args.output_dir / "summary_by_arm.csv", by_arm)
    write_csv(args.output_dir / "summary_by_receiver_arm_k.csv", by_receiver_arm_k)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    write_csv(args.output_dir / "paired_delta_summary_by_arm_k.csv", delta_by_arm_k)
    write_csv(args.output_dir / "training_task_rows.csv", training_rows)
    write_csv(args.output_dir / "training_summary_by_arm.csv", training_by_arm)
    write_csv(args.output_dir / "training_summary_by_arm_k.csv", training_by_arm_k)
    write_csv(args.output_dir / "loss_curve_by_arm_epoch.csv", loss_curve)
    audit = {
        "status": "PASS" if not protocol["errors"] and not any(logs["hit_counts"].values()) else "FAIL",
        "protocol": protocol,
        "logs": logs,
    }
    write_json(args.output_dir / "audit.json", audit)
    (args.output_dir / "main_table.md").write_text(
        build_markdown(by_arm_k, deltas), encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True, allow_nan=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
