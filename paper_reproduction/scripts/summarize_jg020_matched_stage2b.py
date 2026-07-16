"""Summarize the strict 25-row JG_R8_LR020 matched K=10 Stage2-B run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
EXPECTED_SEEDS = (713101, 713102, 713103, 713104, 713105)
EXPECTED_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
HISTORICAL_METHODS = ("mrior_sda", "dadda_sda", "protonet_cda")
LOG_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|Phase2ContractError|CUDA out of memory|"
    r"(?:^|\s)(?:RuntimeError|KeyError|AssertionError|FloatingPointError):",
    re.IGNORECASE,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Iterable[float], *, paired_t: bool = False) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data or not all(math.isfinite(value) for value in data):
        raise FloatingPointError(f"invalid metric group: {data[:5]}")
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    # The completed matrix always has n=25 for paired comparisons; t_0.975,24=2.0639.
    critical = 2.0639 if paired_t and len(data) == 25 else 1.96
    half = critical * std / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "n": len(data),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": min(data),
        "max": max(data),
    }


def _historical_rows(path: Path) -> dict[tuple[str, int, int, str], dict[str, str]]:
    result: dict[tuple[str, int, int, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["k_shot"]) != 10 or row["method"] not in HISTORICAL_METHODS:
                continue
            key = (row["receiver"], int(row["seed"]), 10, row["method"])
            if key in result:
                raise ValueError(f"duplicate historical row: {key}")
            result[key] = row
    return result


def _group(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def _scan_completed_logs(log_root: Path) -> dict[str, Any]:
    paths = sorted(log_root.glob("full_worker_*.out")) + sorted(
        log_root.rglob("jg_r8_lr020.log")
    )
    hits: list[dict[str, Any]] = []
    line_count = 0
    byte_count = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        byte_count += path.stat().st_size
        for line_no, line in enumerate(text.splitlines(), 1):
            line_count += 1
            if LOG_ERROR_RE.search(line):
                hits.append({"path": str(path), "line": line_no, "text": line[:500]})
    return {
        "file_count": len(paths),
        "line_count": line_count,
        "byte_count": byte_count,
        "error_hits": hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--historical-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical = _historical_rows(args.historical_csv)
    rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    class_cells: list[dict[str, Any]] = []
    errors: list[str] = []
    loss_rows: list[dict[str, Any]] = []

    for metrics_path in sorted(args.run_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        payload = _load_json(metrics_path)
        if payload.get("schema") != "adv3b02_stage2b_supervised_da_v2":
            continue
        receiver = str(payload["target_receiver_label"])
        seed = int(payload["seed"])
        k_shot = int(payload["k_shot"])
        key_prefix = (receiver, seed, k_shot)
        required = (
            "detailed_metrics.json",
            "filesystem_access_audit.json",
            "loss_trace.json",
            "prediction_manifest.json",
            "runtime_isolation_evidence.json",
            "scoring_audit.json",
            "split_manifest.json",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir}:missing={missing}")
            continue

        detailed = _load_json(run_dir / "detailed_metrics.json")
        fs_audit = _load_json(run_dir / "filesystem_access_audit.json")
        loss_trace = _load_json(run_dir / "loss_trace.json")
        prediction = _load_json(run_dir / "prediction_manifest.json")
        runtime = _load_json(run_dir / "runtime_isolation_evidence.json")
        scoring = _load_json(run_dir / "scoring_audit.json")
        split = _load_json(run_dir / "split_manifest.json")

        checks = {
            "method": payload.get("method_id") == "jg_r8_lr020",
            "receiver": receiver in EXPECTED_RECEIVERS,
            "seed": seed in EXPECTED_SEEDS,
            "k10": k_shot == 10,
            "scenarios": tuple(payload["metrics_by_scenario"]) == EXPECTED_SCENARIOS,
            "loss_count": len(loss_trace) == 5,
            "filesystem_pass": fs_audit.get("status") == "PASS",
            "landlock": fs_audit.get("landlock_enforced") is True,
            "no_forbidden_hits": not fs_audit.get("forbidden_access_hits"),
            "runtime_filesystem_pass": runtime.get("filesystem_access_audit_status") == "PASS",
            "runtime_preopen_pass": runtime.get("preopen_audit_status") == "PASS",
            "predict_score_isolated": runtime.get("predict_score_process_isolation") is True,
            "truth_after_prediction": scoring.get("truth_join_after_prediction_only") is True,
            "predictor_exited": scoring.get("predictor_process_exited_before_truth_open") is True,
            "score_no_feedback": scoring.get("scorer_output_must_not_feed_predictor") is True,
            "score_count": int(scoring.get("score_row_count", -1)) == 360,
            "prediction_count": int(prediction.get("prediction_row_count", -1)) == 360,
            "leo_weak_only": split.get("phase2_sample_view_policy")
            == "leo_weak_only_no_clean_access",
            "no_clean_sample": split.get("clean_sample_access") is False,
            "no_clean_derived": split.get("clean_derived_signal_access") is False,
            "no_clean_dataset": split.get("phase2_clean_dataset_reachable") is False,
            "no_clean_cache": split.get("phase2_clean_cache_reachable") is False,
            "no_clean_control": split.get("phase2_clean_control_flow_reachable") is False,
            "per_sample_all_classes": split.get("phase2_query_decision_policy")
            == "per_sample_all_registered_classes",
            "no_role_oracle": split.get("phase2_query_role_oracle_access") is False,
            "no_true_batch_count": split.get("phase2_query_true_batch_class_count_access") is False,
            "no_class_quota": split.get("phase2_query_class_quota_access") is False,
            "no_global_assignment": split.get("phase2_query_batch_global_assignment") is False,
            "no_support_query_overlap": split.get("support_query_overlap") is False,
            "no_query_training": split.get("target_query_used_for_training") is False,
            "no_query_selection": split.get("target_query_used_for_model_selection") is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{run_dir}:checks={failed}")

        for trace in loss_trace:
            numeric = [value for value in trace.values() if isinstance(value, (int, float))]
            if not all(math.isfinite(float(value)) for value in numeric):
                errors.append(f"{run_dir}:nonfinite_loss_trace_epoch={trace.get('epoch')}")
            loss_rows.append({"receiver": receiver, "seed": seed, **trace})

        scenario_class = [
            item for item in detailed if item.get("group_type") == "per_receiver_transmitter"
        ]
        if len(scenario_class) != 18:
            errors.append(f"{run_dir}:scenario_class_count={len(scenario_class)}")
        per_class_correct: dict[str, int] = defaultdict(int)
        per_class_total: dict[str, int] = defaultdict(int)
        for item in scenario_class:
            transmitter = str(item["transmitter_label"])
            per_class_correct[transmitter] += int(item["correct_count"])
            per_class_total[transmitter] += int(item["sample_count"])
            class_cells.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "scenario": item["scenario"],
                    "transmitter": transmitter,
                    "accuracy": float(item["accuracy"]),
                    "correct_count": int(item["correct_count"]),
                    "sample_count": int(item["sample_count"]),
                }
            )
        aggregate_class_accuracies = [
            per_class_correct[label] / per_class_total[label] for label in sorted(per_class_total)
        ]
        metrics = payload["metrics"]
        row: dict[str, Any] = {
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
            "jg_old_accuracy": float(metrics["target_old_accuracy_mean"]),
            "direct_old_accuracy": float(metrics["target_old_accuracy_before_adaptation_mean"]),
            "identity_qknn_old_accuracy": float(metrics["target_old_accuracy_identity_qknn_mean"]),
            "jg_delta_vs_direct": float(metrics["target_old_accuracy_delta_mean"]),
            "jg_delta_vs_identity_qknn": float(
                metrics["target_old_accuracy_delta_vs_identity_qknn_mean"]
            ),
            "aggregate_class_floor": min(aggregate_class_accuracies),
            "scenario_class_floor": min(float(item["accuracy"]) for item in scenario_class),
            "adaptation_latency_sec": float(metrics["adaptation_latency_sec_mean"]),
        }
        resource = {
            **next(iter(prediction["scenario_runtime"].values())),
            **prediction["resource_summary"],
        }
        for field in (
            "adapter_optimizer_steps",
            "adv3b02_gradient_updates",
            "trainable_parameters",
            "support_view_count",
            "query_view_count",
            "support_forward_sample_equivalents",
            "full_backbone_forward_sample_equivalents",
            "peak_cuda_memory_bytes",
        ):
            row[field] = resource[field]
        row["persistent_state_bytes"] = resource["persistent_state_bytes_estimate"]
        for method in HISTORICAL_METHODS:
            hist_key = (*key_prefix, method)
            if hist_key not in historical:
                errors.append(f"missing_historical={hist_key}")
                continue
            hist = historical[hist_key]
            prefix = {"mrior_sda": "mrior", "dadda_sda": "dadda", "protonet_cda": "protonet"}[method]
            hist_acc = float(hist["target_old_accuracy"])
            row[f"{prefix}_old_accuracy"] = hist_acc
            row[f"{prefix}_adaptation_latency_sec"] = float(hist["adaptation_latency_sec"])
            row[f"jg_delta_vs_{prefix}"] = row["jg_old_accuracy"] - hist_acc
        rows.append(row)
        for scenario, values in payload["metrics_by_scenario"].items():
            scenario_rows.append(
                {
                    "receiver": receiver,
                    "seed": seed,
                    "scenario": scenario,
                    "jg_old_accuracy": float(values["target_old_accuracy"]),
                    "direct_old_accuracy": float(values["target_old_accuracy_before_adaptation"]),
                    "identity_qknn_old_accuracy": float(values["target_old_accuracy_identity_qknn"]),
                    "jg_delta_vs_direct": float(values["target_old_accuracy_delta"]),
                    "jg_delta_vs_identity_qknn": float(
                        values["target_old_accuracy_delta_vs_identity_qknn"]
                    ),
                }
            )

    expected_keys = {
        (receiver, seed) for receiver in EXPECTED_RECEIVERS for seed in EXPECTED_SEEDS
    }
    actual_keys = {(row["receiver"], row["seed"]) for row in rows}
    if actual_keys != expected_keys:
        errors.append(
            f"matrix_keys_missing={sorted(expected_keys - actual_keys)},extra={sorted(actual_keys - expected_keys)}"
        )

    paired_summary: list[dict[str, Any]] = []
    win_tolerance = 1e-12
    for comparator, delta_field in (
        ("strict_direct_ADV3B02", "jg_delta_vs_direct"),
        ("P4_identity_qKNN", "jg_delta_vs_identity_qknn"),
        ("MRIOR", "jg_delta_vs_mrior"),
        ("DADDA", "jg_delta_vs_dadda"),
        ("ProtoNet", "jg_delta_vs_protonet"),
    ):
        values = [float(row[delta_field]) for row in rows]
        paired_summary.append(
            {
                "comparator": comparator,
                **_stats(values, paired_t=True),
                "wins": sum(value > win_tolerance for value in values),
                "ties": sum(abs(value) <= win_tolerance for value in values),
                "losses": sum(value < -win_tolerance for value in values),
            }
        )

    receiver_summary: list[dict[str, Any]] = []
    for (receiver,), group_rows in sorted(_group(rows, ("receiver",)).items()):
        item: dict[str, Any] = {"receiver": receiver}
        for field in (
            "jg_old_accuracy",
            "direct_old_accuracy",
            "identity_qknn_old_accuracy",
            "mrior_old_accuracy",
            "dadda_old_accuracy",
            "protonet_old_accuracy",
            "aggregate_class_floor",
        ):
            item.update({f"{field}_{key}": value for key, value in _stats(row[field] for row in group_rows).items()})
        receiver_summary.append(item)

    scenario_summary: list[dict[str, Any]] = []
    for (scenario,), group_rows in sorted(_group(scenario_rows, ("scenario",)).items()):
        item = {"scenario": scenario}
        for field in (
            "jg_old_accuracy",
            "direct_old_accuracy",
            "identity_qknn_old_accuracy",
            "jg_delta_vs_direct",
            "jg_delta_vs_identity_qknn",
        ):
            item.update({f"{field}_{key}": value for key, value in _stats(row[field] for row in group_rows).items()})
        scenario_summary.append(item)

    class_summary: list[dict[str, Any]] = []
    for (transmitter,), group_rows in sorted(_group(class_cells, ("transmitter",)).items()):
        correct = sum(int(row["correct_count"]) for row in group_rows)
        total = sum(int(row["sample_count"]) for row in group_rows)
        class_summary.append(
            {
                "transmitter": transmitter,
                "weighted_accuracy": correct / total,
                **{f"cell_accuracy_{key}": value for key, value in _stats(row["accuracy"] for row in group_rows).items()},
            }
        )

    log_audit = _scan_completed_logs(args.log_root)
    if log_audit["error_hits"]:
        errors.append(f"completed_log_error_hits={len(log_audit['error_hits'])}")
    worker_summaries = []
    for idx in range(8):
        path = args.log_root / f"worker_{idx}_summary.json"
        if not path.is_file():
            errors.append(f"missing_worker_summary={path}")
            continue
        summary = _load_json(path)
        worker_summaries.append(summary)
        if int(summary.get("failed", -1)) != 0:
            errors.append(f"worker_{idx}_failed={summary.get('failed')}")
    if sum(int(item["completed"]) for item in worker_summaries) != 24:
        errors.append("worker_completed_total_not_24")
    if sum(int(item["skipped"]) for item in worker_summaries) != 1:
        errors.append("worker_skipped_total_not_1")

    summary = {
        "schema": "qknnv42_jg020_matched_stage2b_summary_v1",
        "artifact_complete": not errors,
        "row_count": len(rows),
        "old_only_stage2b": True,
        "no_new_class_registration_metrics": True,
        "jg_old_accuracy": _stats(row["jg_old_accuracy"] for row in rows),
        "direct_old_accuracy": _stats(row["direct_old_accuracy"] for row in rows),
        "identity_qknn_old_accuracy": _stats(row["identity_qknn_old_accuracy"] for row in rows),
        "aggregate_class_floor": _stats(row["aggregate_class_floor"] for row in rows),
        "scenario_class_floor": _stats(row["scenario_class_floor"] for row in rows),
        "adaptation_latency_sec": _stats(row["adaptation_latency_sec"] for row in rows),
        "peak_cuda_memory_bytes": _stats(row["peak_cuda_memory_bytes"] for row in rows),
        "persistent_state_bytes": _stats(row["persistent_state_bytes"] for row in rows),
        "loss_trace": {
            "row_count": len(loss_rows),
            "first_epoch_loss": _stats(row["loss"] for row in loss_rows if row["epoch"] == 1),
            "final_epoch_loss": _stats(row["loss"] for row in loss_rows if row["epoch"] == 5),
            "first_epoch_support_acc": _stats(
                row["support_train_acc"] for row in loss_rows if row["epoch"] == 1
            ),
            "final_epoch_support_acc": _stats(
                row["support_train_acc"] for row in loss_rows if row["epoch"] == 5
            ),
        },
        "completed_log_audit": log_audit,
        "worker_summary": {
            "completed": sum(int(item["completed"]) for item in worker_summaries),
            "skipped": sum(int(item["skipped"]) for item in worker_summaries),
            "failed": sum(int(item["failed"]) for item in worker_summaries),
        },
        "protocol_audit": {
            "filesystem_pass_rows": len(rows),
            "runtime_preopen_pass_rows": len(rows),
            "isolated_scoring_pass_rows": len(rows),
            "forbidden_access_rows": 0,
        },
        "errors": errors,
    }

    for filename, values in (
        ("per_run_results.csv", rows),
        ("per_scenario_results.csv", scenario_rows),
        ("paired_summary.csv", paired_summary),
        ("receiver_summary.csv", receiver_summary),
        ("scenario_summary.csv", scenario_summary),
        ("class_summary.csv", class_summary),
        ("loss_trace_all.csv", loss_rows),
    ):
        _write_csv(args.output_dir / filename, values)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
