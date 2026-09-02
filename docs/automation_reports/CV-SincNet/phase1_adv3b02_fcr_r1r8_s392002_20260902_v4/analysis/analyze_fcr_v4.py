from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SAMPLE_ID = re.compile(
    r"^tx(?P<tx>\d+):rx(?P<rx>\d+):day(?P<day>\d+):eq(?P<eq>\d+):sig(?P<sig>\d+)$"
)
FCR_LINE = re.compile(
    r"^\[FCR\] stage=(?P<stage>\S+) scales=(?P<scales>\{[^}]*\}) "
    r"components=(?P<components>\{[^}]*\}) active_pairs=(?P<active>(?:[0-9.]+|nan)) "
    r"capability_reasons=(?P<capabilities>\{[^}]*\}) "
    r"freeze_decoder_for_necessity=(?P<freeze>[01])$"
)
ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "RuntimeError:",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Killed",
    "ComplexHalf",
)


def finite(values):
    return [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else float("nan")


def score_records(records: list[dict], row: str) -> dict:
    by_scenario: dict[str, list[tuple[dict, dict[str, int]]]] = defaultdict(list)
    for record in records:
        if record.get("row_id") != row:
            raise ValueError(f"{row}: row_id mismatch")
        match = SAMPLE_ID.fullmatch(str(record.get("sample_id", "")))
        if match is None:
            raise ValueError(f"{row}: non-reversible or malformed sample_id")
        scenario = str(record.get("scenario"))
        if scenario not in SCENARIOS:
            raise ValueError(f"{row}: unexpected scenario {scenario}")
        predicted = int(record["predicted_class"])
        if predicted < 0:
            raise ValueError(f"{row}: negative prediction")
        by_scenario[scenario].append((record, {key: int(value) for key, value in match.groupdict().items()}))
    if tuple(sorted(by_scenario)) != tuple(sorted(SCENARIOS)):
        raise ValueError(f"{row}: scenario set mismatch")

    id_sets = []
    scenario_scores = {}
    for scenario in SCENARIOS:
        items = by_scenario[scenario]
        ids = [str(record["sample_id"]) for record, _ in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{row}/{scenario}: duplicate sample_id")
        id_sets.append(set(ids))
        true_values = sorted({meta["tx"] for _, meta in items})
        predicted_values = sorted({int(record["predicted_class"]) for record, _ in items})
        classes = sorted(set(true_values) | set(predicted_values))
        confusion = {str(t): {str(p): 0 for p in classes} for t in classes}
        class_total = Counter()
        class_correct = Counter()
        receiver_total = Counter()
        receiver_correct = Counter()
        day_total = Counter()
        day_correct = Counter()
        prediction_hist = Counter()
        for record, meta in items:
            truth = meta["tx"]
            prediction = int(record["predicted_class"])
            correct = truth == prediction
            class_total[truth] += 1
            class_correct[truth] += int(correct)
            receiver_total[meta["rx"]] += 1
            receiver_correct[meta["rx"]] += int(correct)
            day_total[meta["day"]] += 1
            day_correct[meta["day"]] += int(correct)
            prediction_hist[prediction] += 1
            confusion[str(truth)][str(prediction)] += 1
        per_class = {
            str(label): percent(class_correct[label], class_total[label]) for label in sorted(class_total)
        }
        per_receiver = {
            str(label): percent(receiver_correct[label], receiver_total[label])
            for label in sorted(receiver_total)
        }
        per_day = {
            str(label): percent(day_correct[label], day_total[label]) for label in sorted(day_total)
        }
        correct = sum(class_correct.values())
        scenario_scores[scenario] = {
            "n": len(items),
            "correct": int(correct),
            "micro_acc": percent(correct, len(items)),
            "macro_acc": mean(per_class.values()),
            "class_floor": min(per_class.values()),
            "receiver_floor": min(per_receiver.values()),
            "day_floor": min(per_day.values()),
            "per_class_acc": per_class,
            "per_receiver_acc": per_receiver,
            "per_day_acc": per_day,
            "prediction_hist": {str(key): int(value) for key, value in sorted(prediction_hist.items())},
            "confusion": confusion,
        }
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError(f"{row}: scenario sample sets differ")
    return {
        "records": len(records),
        "samples_per_scenario": len(id_sets[0]),
        "truth_join": "parsed_reversible_sample_id",
        "strict_opaque_truth_last": False,
        "scenarios": scenario_scores,
        "four_scenario_mean_micro": mean(
            scenario_scores[scenario]["micro_acc"] for scenario in SCENARIOS
        ),
        "three_leo_mean_micro": mean(
            scenario_scores[scenario]["micro_acc"] for scenario in SCENARIOS[1:]
        ),
        "three_leo_worst_micro": min(
            scenario_scores[scenario]["micro_acc"] for scenario in SCENARIOS[1:]
        ),
    }


def parse_training(row_dir: Path) -> dict:
    with (row_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) != 200:
        raise ValueError(f"{row_dir.name}: expected 200 metrics rows, got {len(metrics)}")
    epochs = [int(item["epoch"]) for item in metrics]
    if epochs != list(range(1, 201)):
        raise ValueError(f"{row_dir.name}: epoch sequence mismatch")
    log_text = (row_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    log_lines = log_text.splitlines()
    jsonl_lines = (row_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    jsonl = [json.loads(line) for line in jsonl_lines if line.strip()]
    if len(jsonl) != 200 or [int(item["epoch"]) for item in jsonl] != epochs:
        raise ValueError(f"{row_dir.name}: logs.jsonl epoch closure mismatch")

    parsed_fcr = []
    for line in log_lines:
        match = FCR_LINE.fullmatch(line)
        if match:
            parsed_fcr.append(
                {
                    "stage": match.group("stage"),
                    "scales": json.loads(match.group("scales")),
                    "components": json.loads(match.group("components")),
                    "active_pairs": float(match.group("active")),
                    "capabilities": json.loads(match.group("capabilities")),
                    "freeze_decoder_for_necessity": bool(int(match.group("freeze"))),
                }
            )
    if len(parsed_fcr) != 200:
        raise ValueError(f"{row_dir.name}: expected 200 FCR rows, got {len(parsed_fcr)}")

    val = [float(item["val_tx_acc"]) for item in metrics]
    train_loss_pairs = [
        (int(item["epoch"]), float(item["train_loss"]))
        for item in metrics
        if item["train_loss"] not in (None, "") and math.isfinite(float(item["train_loss"]))
    ]
    train_acc_pairs = [
        (int(item["epoch"]), float(item["train_tx_acc"]))
        for item in metrics
        if item["train_tx_acc"] not in (None, "") and math.isfinite(float(item["train_tx_acc"]))
    ]
    missing_train_loss_epochs = [
        int(item["epoch"])
        for item in metrics
        if item["train_loss"] in (None, "") or not math.isfinite(float(item["train_loss"]))
    ]
    test_pairs = [
        (int(item["epoch"]), float(item["test_tx_acc"]))
        for item in metrics
        if item["test_tx_acc"] not in (None, "") and math.isfinite(float(item["test_tx_acc"]))
    ]
    best_val_index = max(range(len(val)), key=val.__getitem__)
    stage_counts = Counter(item["stage"] for item in parsed_fcr)
    component_keys = sorted({key for item in parsed_fcr for key in item["components"]})
    component_summary = {}
    for key in component_keys:
        values = [float(item["components"].get(key, 0.0)) for item in parsed_fcr]
        finite_values = [value for value in values if math.isfinite(value)]
        component_summary[key] = {
            "mean_finite": mean(finite_values) if finite_values else None,
            "max_finite": max(finite_values) if finite_values else None,
            "finite_active_epochs": sum(abs(value) > 0.0 for value in finite_values),
            "nan_epochs": sum(not math.isfinite(value) for value in values),
            "first_nan_epoch": next(
                (epochs[index] for index, value in enumerate(values) if not math.isfinite(value)),
                None,
            ),
        }
    active_pairs = [item["active_pairs"] for item in parsed_fcr]
    finite_active_pairs = [value for value in active_pairs if math.isfinite(value)]
    return {
        "epoch_rows": len(metrics),
        "logs_jsonl_rows": len(jsonl),
        "stdout_lines": len(log_lines),
        "best_val_tx_acc": val[best_val_index],
        "best_val_epoch": epochs[best_val_index],
        "final_val_tx_acc": val[-1],
        "start_train_loss": train_loss_pairs[0][1] if train_loss_pairs else None,
        "min_train_loss": min((value for _, value in train_loss_pairs), default=None),
        "min_train_loss_epoch": min(train_loss_pairs, key=lambda item: item[1])[0]
        if train_loss_pairs
        else None,
        "last_finite_train_loss": train_loss_pairs[-1][1] if train_loss_pairs else None,
        "last_finite_train_loss_epoch": train_loss_pairs[-1][0] if train_loss_pairs else None,
        "missing_train_loss_epoch_count": len(missing_train_loss_epochs),
        "first_missing_train_loss_epoch": missing_train_loss_epochs[0]
        if missing_train_loss_epochs
        else None,
        "last_missing_train_loss_epoch": missing_train_loss_epochs[-1]
        if missing_train_loss_epochs
        else None,
        "max_train_tx_acc": max((value for _, value in train_acc_pairs), default=None),
        "last_finite_train_tx_acc": train_acc_pairs[-1][1] if train_acc_pairs else None,
        "last_finite_train_tx_acc_epoch": train_acc_pairs[-1][0] if train_acc_pairs else None,
        "last_periodic_test_epoch": test_pairs[-1][0] if test_pairs else None,
        "last_periodic_test_acc": test_pairs[-1][1] if test_pairs else None,
        "best_periodic_test_acc": max((value for _, value in test_pairs), default=None),
        "skipped_backward_batches": int(metrics[-1]["skipped_backward_batches_so_far"]),
        "epoch_time_total_s": sum(float(item["epoch_time_s"]) for item in metrics),
        "train_time_total_s": sum(float(item["train_time_s"]) for item in metrics),
        "eval_time_total_s": sum(float(item["eval_time_s"]) for item in metrics),
        "fcr_stage_counts": dict(sorted(stage_counts.items())),
        "fcr_component_summary": component_summary,
        "active_pair_epoch_mean_finite": mean(finite_active_pairs) if finite_active_pairs else None,
        "active_pair_epoch_max_finite": max(finite_active_pairs) if finite_active_pairs else None,
        "active_pair_nan_epochs": sum(not math.isfinite(value) for value in active_pairs),
        "active_pair_first_nan_epoch": next(
            (epochs[index] for index, value in enumerate(active_pairs) if not math.isfinite(value)),
            None,
        ),
        "unsafe_backward_warning_count": sum(
            "unsafe backward/step skipped" in line for line in log_lines
        ),
        "error_marker_counts": {
            marker: log_text.count(marker) for marker in ERROR_MARKERS
        },
        "training_finished_marker": "Training finished." in log_text,
        "predictions_ready_marker": "[FCR-PREDICTIONS-READY]" in log_text,
        "diagnostics_marker": "[FCR-DIAGNOSTICS]" in log_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    output = {
        "schema": "adv3b02_fcr_v4_independent_analysis:v1",
        "run_id": "phase1_adv3b02_fcr_r1r8_s392002_20260902_v4",
        "truth_join": {
            "method": "parse tx/rx/day/eq/sig from prediction sample_id",
            "independent_process": True,
            "strict_opaque_truth_last": False,
            "limitation": "sample_id reversibly encodes transmitter truth",
        },
        "rows": {},
    }
    csv_rows = []
    row_summary = []
    per_class_rows = []
    for index in range(1, 9):
        row = f"R{index}"
        row_dir = base / row
        prediction_path = row_dir / "fcr_predictions.json"
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        score = score_records(payload["records"], row)
        training = parse_training(row_dir)
        diagnostics = json.loads((row_dir / "fcr_diagnostics.json").read_text(encoding="utf-8"))
        status = (row_dir / "status.txt").read_text(encoding="utf-8").strip()
        output["rows"][row] = {
            "status": status,
            "prediction_schema": payload.get("schema"),
            "training": training,
            "score": score,
            "diagnostics": diagnostics,
        }
        row_summary.append(
            {
                "row": row,
                "status": status,
                "best_val_tx_acc": training["best_val_tx_acc"],
                "best_val_epoch": training["best_val_epoch"],
                "missing_train_loss_epoch_count": training["missing_train_loss_epoch_count"],
                "first_missing_train_loss_epoch": training["first_missing_train_loss_epoch"],
                "skipped_backward_batches": training["skipped_backward_batches"],
                "fcr_nan_epochs": training["active_pair_nan_epochs"],
                "clean_acc": score["scenarios"]["clean"]["micro_acc"],
                "leo_clear_weak_acc": score["scenarios"]["leo_clear_weak"]["micro_acc"],
                "leo_low_elev_weak_acc": score["scenarios"]["leo_low_elev_weak"]["micro_acc"],
                "leo_rain_weak_acc": score["scenarios"]["leo_rain_weak"]["micro_acc"],
                "three_leo_mean_acc": score["three_leo_mean_micro"],
                "three_leo_worst_acc": score["three_leo_worst_micro"],
                "clean_class_floor": score["scenarios"]["clean"]["class_floor"],
                "zf_domain_probe": diagnostics["zf_domain_probe"],
                "zn_domain_probe": diagnostics["zn_domain_probe"],
                "effective_rank": diagnostics["effective_rank"],
                "gram_condition": diagnostics["gram_condition"],
                "peak_vram_mb": diagnostics["peak_vram_mb"],
                "latency_ms": diagnostics["latency_ms"],
                "train_time_s": diagnostics["train_time_s"],
            }
        )
        for scenario in SCENARIOS:
            scored = score["scenarios"][scenario]
            csv_rows.append(
                {
                    "row": row,
                    "status": status,
                    "scenario": scenario,
                    "n": scored["n"],
                    "micro_acc": scored["micro_acc"],
                    "macro_acc": scored["macro_acc"],
                    "class_floor": scored["class_floor"],
                    "receiver_floor": scored["receiver_floor"],
                    "best_val_tx_acc": training["best_val_tx_acc"],
                    "best_val_epoch": training["best_val_epoch"],
                    "skipped_backward_batches": training["skipped_backward_batches"],
                    "train_time_s": diagnostics["train_time_s"],
                    "peak_vram_mb": diagnostics["peak_vram_mb"],
                    "latency_ms": diagnostics["latency_ms"],
                }
            )
            for class_id, class_acc in scored["per_class_acc"].items():
                per_class_rows.append(
                    {
                        "row": row,
                        "scenario": scenario,
                        "class_id": class_id,
                        "class_acc": class_acc,
                        "class_n": sum(scored["confusion"][class_id].values()),
                        "prediction_hist": json.dumps(
                            scored["prediction_hist"], ensure_ascii=False, sort_keys=True
                        ),
                    }
                )
    (base / "analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (base / "scenario_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (base / "row_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_summary[0]))
        writer.writeheader()
        writer.writerows(row_summary)
    with (base / "per_class_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
