#!/usr/bin/env python3
"""Build a reproducible, evidence-bounded ECRS-V1 run data package.

The input is a read-only local snapshot containing R1--R8 stdout, CSV/JSONL
epoch logs, and (when available) the source-only ``ecrs_v1_diagnostics.pt``
artifacts.  The script never reads target truth beyond the evaluation lines
already emitted by the training program and never modifies the source snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RUN_ID = "phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2"
RUN_PREFIX = "ADV3B02_ECRS_R"
MILESTONES = {1, 10, 20, 40, 41, 60, 79, 80, 90, 91, 100, 120, 150, 180, 190, 194, 195, 196, 197, 198, 199, 200}
NUMERIC_SUMMARY_FIELDS = (
    "train_loss",
    "train_tx_acc",
    "val_tx_acc",
    "val_dom_acc",
    "epoch_time_s",
    "train_time_s",
    "eval_time_s",
    "skipped_backward_batches_this_epoch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stage_for_epoch(epoch: int) -> str:
    if epoch <= 40:
        return "Stage2_canonical_content_cycle"
    if epoch <= 90:
        return "Stage3_split_pair"
    return "Stage4_discriminative_gate"


def schedule_event(epoch: int) -> str:
    return {
        1: "Stage2开始；clear LEO，p=0.30",
        41: "Stage3开始；low-elevation/rain LEO，p=0.60",
        80: "卫星分类权重进入既定有效区间",
        91: "Stage4开始；三类LEO，p=0.80",
        200: "训练终点与最终评估",
    }.get(epoch, "milestone")


def as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(clean) if clean else None


def fmt_number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def locate_log_root(snapshot: Path) -> Path:
    direct = snapshot / f"{RUN_PREFIX}1.out"
    if direct.exists():
        return snapshot
    named = snapshot / RUN_ID
    if (named / f"{RUN_PREFIX}1.out").exists():
        return named
    matches = list(snapshot.rglob(f"{RUN_PREFIX}1.out"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot identify one log root beneath {snapshot}: {matches}")
    return matches[0].parent


def load_epoch_logs(log_root: Path, rung: int) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str]]:
    run_name = f"{RUN_PREFIX}{rung}"
    csv_path = log_root / run_name / "metrics.csv"
    jsonl_path = log_root / run_name / "logs.jsonl"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = []
    with jsonl_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    json_rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{jsonl_path}:{line_no}: {exc}") from exc
    stdout_lines = (log_root / f"{run_name}.out").read_text(encoding="utf-8", errors="replace").splitlines()
    return csv_rows, json_rows, stdout_lines


def parse_evaluations(rung: int, lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    receivers: list[dict[str, Any]] = []
    primary: dict[str, Any] = {}
    main_re = re.compile(
        r"^\[(FINAL-(?:BEST|PRIMARY))\] val_tx=([\d.]+)% \| test_overall_tx=([\d.]+)%"
        r"(?: \| strict_udu=([\d.]+)% \| score=([\d.]+))?"
    )
    group_re = re.compile(
        r"^\[(FINAL-(?:BEST|PRIMARY))\] (seen_day_unseen_rx|unseen_day_unseen_rx).*?: tx=([\d.]+)% \((\d+)/(\d+)\)"
    )
    rx_re = re.compile(
        r"^\[(FINAL-(?:BEST|PRIMARY))\] rx=(\d+) on (seen_days|unseen_days)=.*?: tx=([\d.]+)% \((\d+)/(\d+)\)"
    )
    sat_re = re.compile(
        r"^\[(FINAL-(?:BEST|PRIMARY))\] \[SAT-TEST\] scenario=([^ ]+).*?overall_tx=([\d.]+)% strict_udu=([\d.]+)%.*?\((\d+)/(\d+)\)"
    )
    for line in lines:
        match = main_re.search(line)
        if match:
            checkpoint = match.group(1)
            row = {
                "rung": f"R{rung}", "checkpoint": checkpoint, "scope": "main",
                "scenario": "clean", "val_tx_acc_pct": float(match.group(2)),
                "overall_tx_acc_pct": float(match.group(3)),
                "strict_udu_tx_acc_pct": as_float(match.group(4)),
                "primary_score": as_float(match.group(5)), "correct": "", "total": "",
            }
            evaluations.append(row)
            if checkpoint == "FINAL-PRIMARY":
                primary.update(row)
            continue
        match = group_re.search(line)
        if match:
            evaluations.append({
                "rung": f"R{rung}", "checkpoint": match.group(1), "scope": match.group(2),
                "scenario": "clean", "val_tx_acc_pct": "", "overall_tx_acc_pct": float(match.group(3)),
                "strict_udu_tx_acc_pct": "", "primary_score": "",
                "correct": int(match.group(4)), "total": int(match.group(5)),
            })
            continue
        match = rx_re.search(line)
        if match:
            receivers.append({
                "rung": f"R{rung}", "checkpoint": match.group(1), "receiver_id": int(match.group(2)),
                "day_scope": match.group(3), "tx_acc_pct": float(match.group(4)),
                "correct": int(match.group(5)), "total": int(match.group(6)),
            })
            continue
        match = sat_re.search(line)
        if match:
            evaluations.append({
                "rung": f"R{rung}", "checkpoint": match.group(1), "scope": "main",
                "scenario": match.group(2), "val_tx_acc_pct": "",
                "overall_tx_acc_pct": float(match.group(3)),
                "strict_udu_tx_acc_pct": float(match.group(4)), "primary_score": "",
                "correct": int(match.group(5)), "total": int(match.group(6)),
            })
    return evaluations, receivers, primary


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _hash_test_mask(group_ids: list[str]) -> list[bool]:
    return [hashlib.sha1(str(group).encode("utf-8")).digest()[0] < 51 for group in group_ids]


def nearest_centroid_probe(features: Any, labels: Any, group_ids: list[str]) -> dict[str, Any]:
    import torch

    x = features.detach().cpu().float().reshape(features.shape[0], -1)
    y = labels.detach().cpu().long().reshape(-1)
    valid = y >= 0
    mask_list = _hash_test_mask(group_ids)
    test = torch.tensor(mask_list, dtype=torch.bool) & valid
    train = (~torch.tensor(mask_list, dtype=torch.bool)) & valid
    classes = sorted(set(int(v) for v in y[valid].tolist()))
    if not classes or not bool(train.any()) or not bool(test.any()):
        return {"accuracy_pct": None, "majority_pct": None, "train_n": int(train.sum()), "test_n": int(test.sum()), "classes": len(classes)}
    mu = x[train].mean(dim=0)
    sigma = x[train].std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - mu) / sigma
    centroids = []
    kept_classes = []
    for cls in classes:
        cls_mask = train & (y == cls)
        if bool(cls_mask.any()):
            centroids.append(z[cls_mask].mean(dim=0))
            kept_classes.append(cls)
    centroid_matrix = torch.stack(centroids)
    distances = torch.cdist(z[test], centroid_matrix)
    prediction_indices = distances.argmin(dim=1).tolist()
    predicted = torch.tensor([kept_classes[index] for index in prediction_indices], dtype=torch.long)
    truth = y[test]
    accuracy = float(predicted.eq(truth).float().mean().item() * 100.0)
    counts = Counter(int(v) for v in truth.tolist())
    majority = 100.0 * max(counts.values()) / max(1, len(truth))
    return {"accuracy_pct": accuracy, "majority_pct": majority, "train_n": int(train.sum()), "test_n": int(test.sum()), "classes": len(kept_classes)}


def load_diagnostics(path: Path, rung: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    directional_a: list[float] = []
    directional_b: list[float] = []
    covariance_means: list[float] = []
    block_values: list[list[float]] = [[], [], [], []]
    coef_magnitudes: list[float] = []
    anchor_magnitudes: list[float] = []
    z_norms: list[float] = []
    source_only = 0
    synchronized = 0
    sample_count = 0
    all_features: dict[str, list[Any]] = {key: [] for key in ("z_resp", "s_hat_summary", "c_fp", "gamma_nuis", "canonical_nuisance")}
    all_labels: dict[str, list[Any]] = {key: [] for key in ("tx_labels", "receiver_labels", "view_labels")}
    all_groups: list[str] = []
    for record_index, record in enumerate(records):
        source_only += int(bool(record.get("source_only")))
        meta = record["pair_metadata"]
        synchronized += int(bool(meta.get("synchronized_crop")))
        directional_a.append(float(record["directional_prediction"]["clean_to_leo_nmse"]))
        directional_b.append(float(record["directional_prediction"]["leo_to_clean_nmse"]))
        surface = record["surface_export"]
        covariance_means.append(float(surface["resp_cov_diag"].float().mean()))
        blocks = surface["block_identifiability"].float()
        for index in range(4):
            block_values[index].extend(float(v) for v in blocks[:, index].tolist())
        coef_magnitudes.append(float(surface["resp_coef"].abs().mean()))
        anchor_magnitudes.append(float(surface["resp_anchor"].abs().mean()))
        probe = record["probe_payload"]
        z_norms.append(float(probe["z_resp"].float().norm(dim=1).mean()))
        n = int(probe["z_resp"].shape[0])
        sample_count += n
        ids = list(meta.get("physical_sample_id", []))
        if len(ids) != n:
            ids = [f"record{record_index}:sample{index}" for index in range(n)]
        all_groups.extend(str(item) for item in ids)
        for key in all_features:
            all_features[key].append(probe[key])
        for key in all_labels:
            all_labels[key].append(probe[key])
    summary = {
        "rung": f"R{rung}", "schema": payload.get("schema"), "feature_schema": payload.get("feature_schema"),
        "epoch": payload.get("epoch"), "artifact_rung": payload.get("rung"), "record_count": len(records),
        "sample_count": sample_count, "source_only_records": source_only,
        "synchronized_crop_records": synchronized,
        "clean_to_leo_nmse_mean": mean(directional_a), "clean_to_leo_nmse_median": quantile(directional_a, 0.5),
        "clean_to_leo_nmse_p95": quantile(directional_a, 0.95),
        "leo_to_clean_nmse_mean": mean(directional_b), "leo_to_clean_nmse_median": quantile(directional_b, 0.5),
        "leo_to_clean_nmse_p95": quantile(directional_b, 0.95),
        "resp_cov_diag_mean": mean(covariance_means), "resp_coef_abs_mean": mean(coef_magnitudes),
        "resp_anchor_abs_mean": mean(anchor_magnitudes), "z_resp_l2_mean": mean(z_norms),
    }
    for index, label in enumerate(("pa", "iq", "memory", "dac")):
        summary[f"ident_{label}_mean"] = mean(block_values[index])
        summary[f"ident_{label}_p05"] = quantile(block_values[index], 0.05)

    features_cat = {key: torch.cat(value, dim=0) for key, value in all_features.items()}
    labels_cat = {key: torch.cat(value, dim=0) for key, value in all_labels.items()}
    probes: list[dict[str, Any]] = []
    tasks = (
        ("z_resp", "tx_labels"), ("z_resp", "receiver_labels"), ("z_resp", "view_labels"),
        ("s_hat_summary", "tx_labels"), ("s_hat_summary", "receiver_labels"), ("s_hat_summary", "view_labels"),
        ("c_fp", "tx_labels"), ("c_fp", "receiver_labels"), ("c_fp", "view_labels"),
        ("gamma_nuis", "receiver_labels"), ("gamma_nuis", "view_labels"),
        ("canonical_nuisance", "receiver_labels"), ("canonical_nuisance", "view_labels"),
    )
    for feature_name, label_name in tasks:
        result = nearest_centroid_probe(features_cat[feature_name], labels_cat[label_name], all_groups)
        probes.append({"rung": f"R{rung}", "source": "diagnostic_payload", "feature": feature_name, "target": label_name, **result})

    first = records[0]
    controls = first.get("negative_controls", {})
    if controls:
        control_features = {}
        for key in ("quality_only_tx_probe", "raw_coefficient", "whitened_coefficient", "anchor_surface"):
            control_features[key] = torch.cat([controls["clean"][key], controls["leo"][key]], dim=0)
        n = int(control_features["quality_only_tx_probe"].shape[0])
        first_groups = list(first["pair_metadata"].get("physical_sample_id", []))[:n]
        if len(first_groups) != n:
            first_groups = [f"control:{index}" for index in range(n)]
        tx = first["probe_payload"]["tx_labels"][:n]
        for feature_name, feature in control_features.items():
            result = nearest_centroid_probe(feature, tx, [str(item) for item in first_groups])
            probes.append({"rung": f"R{rung}", "source": "negative_control_first_record", "feature": feature_name, "target": "tx_labels", **result})
        summary["negative_control_basis_controls"] = ";".join(controls["clean"].get("basis_controls", []))
        summary["negative_control_records"] = 1
    else:
        summary["negative_control_basis_controls"] = ""
        summary["negative_control_records"] = 0
    return summary, probes


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.resolve()
    output = args.output.resolve()
    log_root = locate_log_root(snapshot)
    output.mkdir(parents=True, exist_ok=True)

    epoch_rows_out: list[dict[str, Any]] = []
    run_summary: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    diary_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    receivers: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for rung in range(1, 9):
        csv_rows, json_rows, stdout_lines = load_epoch_logs(log_root, rung)
        csv_epochs = [int(row["epoch"]) for row in csv_rows]
        json_epochs = [int(row["epoch"]) for row in json_rows]
        expected_epochs = list(range(1, len(csv_rows) + 1))
        parity_ok = len(csv_rows) == len(json_rows) and csv_epochs == json_epochs == expected_epochs
        if not parity_ok:
            anomalies.append({
                "rung": f"R{rung}", "severity": "ERROR", "category": "epoch_log_parity",
                "epoch": "", "detail": f"csv={len(csv_rows)},jsonl={len(json_rows)},continuous={csv_epochs == expected_epochs}",
            })
        for row in csv_rows:
            enriched = {"rung": f"R{rung}", "stage": stage_for_epoch(int(row["epoch"])), **row}
            epoch_rows_out.append(enriched)

        row_evaluations, row_receivers, primary = parse_evaluations(rung, stdout_lines)
        evaluations.extend(row_evaluations)
        receivers.extend(row_receivers)
        fatal_lines = [line.strip() for line in stdout_lines if "Traceback (most recent call last)" in line or "RuntimeError:" in line]
        completed = bool(primary) and len(csv_rows) == 200
        if completed:
            state = "ARTIFACTS_COMPLETE"
        elif fatal_lines:
            state = "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE"
        else:
            state = "SNAPSHOT_INCOMPLETE_RUNNING_OR_PENDING"
        if fatal_lines:
            for detail in fatal_lines:
                anomalies.append({"rung": f"R{rung}", "severity": "FATAL", "category": "runtime_exception", "epoch": len(csv_rows), "detail": detail})
        if state == "SNAPSHOT_INCOMPLETE_RUNNING_OR_PENDING":
            anomalies.append({"rung": f"R{rung}", "severity": "INFO", "category": "snapshot_cutoff", "epoch": len(csv_rows), "detail": "该快照尚未包含训练终点；不能据此给出最终性能。"})
        skipped_total = int(float(csv_rows[-1].get("skipped_backward_batches_so_far") or 0)) if csv_rows else 0
        if skipped_total:
            anomalies.append({"rung": f"R{rung}", "severity": "WARN", "category": "skipped_backward_batches", "epoch": len(csv_rows), "detail": f"累计跳过反向批次={skipped_total}"})
        run_summary.append({
            "rung": f"R{rung}", "state": state, "epochs_expected": 200, "epochs_observed": len(csv_rows),
            "csv_rows": len(csv_rows), "jsonl_rows": len(json_rows), "epoch_parity_ok": parity_ok,
            "final_train_loss": as_float(csv_rows[-1].get("train_loss")) if csv_rows else None,
            "final_train_tx_acc_pct": as_float(csv_rows[-1].get("train_tx_acc")) if csv_rows else None,
            "final_val_tx_acc_pct": as_float(csv_rows[-1].get("val_tx_acc")) if csv_rows else None,
            "best_val_tx_acc_pct": max((as_float(row.get("val_tx_acc")) or -math.inf) for row in csv_rows) if csv_rows else None,
            "primary_clean_overall_pct": primary.get("overall_tx_acc_pct", ""),
            "primary_clean_strict_udu_pct": primary.get("strict_udu_tx_acc_pct", ""),
            "primary_score": primary.get("primary_score", ""), "skipped_backward_batches": skipped_total,
            "fatal_exception_count": int(bool(fatal_lines)),
        })

        for stage in ("Stage2_canonical_content_cycle", "Stage3_split_pair", "Stage4_discriminative_gate"):
            subset = [row for row in csv_rows if stage_for_epoch(int(row["epoch"])) == stage]
            if not subset:
                continue
            stage_row: dict[str, Any] = {
                "rung": f"R{rung}", "stage": stage, "epoch_start": int(subset[0]["epoch"]),
                "epoch_end": int(subset[-1]["epoch"]), "epoch_count": len(subset),
            }
            for field in NUMERIC_SUMMARY_FIELDS:
                values = [as_float(row.get(field)) for row in subset]
                stage_row[f"{field}_mean"] = mean(values)
                stage_row[f"{field}_start"] = values[0]
                stage_row[f"{field}_end"] = values[-1]
            stage_row["val_tx_acc_best"] = max(as_float(row.get("val_tx_acc")) or -math.inf for row in subset)
            stage_rows.append(stage_row)

        previous: dict[str, str] | None = None
        for row in csv_rows:
            epoch = int(row["epoch"])
            if epoch not in MILESTONES and epoch != len(csv_rows):
                continue
            train_loss = as_float(row.get("train_loss"))
            val_acc = as_float(row.get("val_tx_acc"))
            diary_rows.append({
                "rung": f"R{rung}", "epoch": epoch, "stage": stage_for_epoch(epoch),
                "event": schedule_event(epoch), "lr": row.get("lr", ""), "train_loss": train_loss,
                "train_tx_acc_pct": row.get("train_tx_acc", ""), "val_tx_acc_pct": val_acc,
                "val_dom_acc_pct": row.get("val_dom_acc", ""), "epoch_time_s": row.get("epoch_time_s", ""),
                "skipped_backward_batches_this_epoch": row.get("skipped_backward_batches_this_epoch", ""),
                "delta_train_loss_from_previous_milestone": "" if previous is None else (
                    "" if train_loss is None or as_float(previous.get("train_loss")) is None else train_loss - float(previous["train_loss"])
                ),
                "delta_val_tx_acc_from_previous_milestone": "" if previous is None else (
                    "" if val_acc is None or as_float(previous.get("val_tx_acc")) is None else val_acc - float(previous["val_tx_acc"])
                ),
            })
            previous = row

        diagnostic_path = snapshot / "diagnostics" / f"R{rung}" / "ecrs_v1_diagnostics.pt"
        if diagnostic_path.exists():
            diagnostic_summary, row_probes = load_diagnostics(diagnostic_path, rung)
            diagnostics.append(diagnostic_summary)
            probes.extend(row_probes)

    write_csv(output / "run_summary.csv", run_summary)
    write_csv(output / "epoch_metrics_full.csv", epoch_rows_out)
    write_csv(output / "stage_summary.csv", stage_rows)
    write_csv(output / "training_diary.csv", diary_rows)
    write_csv(output / "evaluations.csv", evaluations)
    write_csv(output / "receiver_results.csv", receivers)
    write_csv(output / "diagnostics_summary.csv", diagnostics)
    write_csv(output / "probe_results.csv", probes)
    write_csv(output / "anomalies.csv", anomalies)
    manifest = {
        "schema": "adv3b02_ecrs_v1_analysis_v1", "run_id": RUN_ID,
        "source_snapshot": str(snapshot), "log_root": str(log_root),
        "row_count": len(run_summary), "epoch_record_count": len(epoch_rows_out),
        "evaluation_record_count": len(evaluations), "receiver_record_count": len(receivers),
        "diagnostic_rungs": [row["rung"] for row in diagnostics],
        "probe_method": "deterministic SHA1 group 80/20 split; train-standardized nearest centroid; paired views share split",
        "limitations": [
            "Probe scores are descriptive source-only diagnostics, not registered target-query results.",
            "Snapshot-incomplete rows have no final-performance claim.",
            "NaN in disabled/inapplicable telemetry is preserved as blank rather than classified as a numerical failure.",
        ],
        "files": [
            "run_summary.csv", "epoch_metrics_full.csv", "stage_summary.csv", "training_diary.csv",
            "evaluations.csv", "receiver_results.csv", "diagnostics_summary.csv", "probe_results.csv", "anomalies.csv",
        ],
    }
    (output / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **{key: manifest[key] for key in ("row_count", "epoch_record_count", "evaluation_record_count", "receiver_record_count", "diagnostic_rungs")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
