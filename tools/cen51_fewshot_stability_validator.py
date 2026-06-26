#!/usr/bin/env python3
"""Validate late-training stability for CEN51 few-shot experiments.

The validator is intentionally log-based so it can be applied to older runs
without reloading checkpoints. It separates two questions:

1. stability_status: did the run avoid late collapse?
2. promotion_status: is the run strong enough to promote as a default?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "RuntimeError:",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Killed",
    "unrecognized arguments:",
)


THRESHOLDS: dict[int, dict[str, float]] = {
    5: {
        "val_drop": 1.50,
        "strict_drop": 2.00,
        "latest_strict_drop": 2.50,
        "val_late_std": 1.00,
        "strict_late_span": 2.50,
        "swad_gap": 1.70,
        "ema_gap": 3.00,
        "rx_floor": 55.00,
        "sat_floor": 28.00,
        "target_val": 90.00,
        "target_strict": 74.60,
    },
    10: {
        "val_drop": 1.00,
        "strict_drop": 1.50,
        "latest_strict_drop": 2.00,
        "val_late_std": 0.80,
        "strict_late_span": 2.00,
        "swad_gap": 1.50,
        "ema_gap": 2.50,
        "rx_floor": 60.00,
        "sat_floor": 30.00,
        "target_val": 93.00,
        "target_strict": 76.50,
    },
    20: {
        "val_drop": 1.00,
        "strict_drop": 1.50,
        "latest_strict_drop": 2.00,
        "val_late_std": 0.70,
        "strict_late_span": 1.80,
        "swad_gap": 1.20,
        "ema_gap": 2.00,
        "rx_floor": 62.00,
        "sat_floor": 31.00,
        "target_val": 94.00,
        "target_strict": 77.00,
    },
    30: {
        "val_drop": 0.80,
        "strict_drop": 1.20,
        "latest_strict_drop": 1.80,
        "val_late_std": 0.55,
        "strict_late_span": 1.50,
        "swad_gap": 1.20,
        "ema_gap": 2.00,
        "rx_floor": 65.00,
        "sat_floor": 33.00,
        "target_val": 96.50,
        "target_strict": 79.00,
    },
    50: {
        "val_drop": 0.80,
        "strict_drop": 1.00,
        "latest_strict_drop": 1.50,
        "val_late_std": 0.45,
        "strict_late_span": 1.30,
        "swad_gap": 1.00,
        "ema_gap": 1.80,
        "rx_floor": 70.00,
        "sat_floor": 34.00,
        "target_val": 97.50,
        "target_strict": 83.00,
    },
}

FAMILY_STRICT_STD_MAX = {
    5: 2.50,
    10: 0.80,
    20: 1.50,
    30: 1.50,
    50: 1.20,
}


@dataclass
class EpochRecord:
    epoch: int
    val_tx: float | None = None
    train_tx: float | None = None
    test_overall: float | None = None
    strict_udu: float | None = None
    sat_strict: dict[str, float] = field(default_factory=dict)


@dataclass
class FinalRecord:
    val_tx: float | None = None
    overall_tx: float | None = None
    strict_udu: float | None = None
    rx_unseen: dict[str, float] = field(default_factory=dict)
    sat_strict: dict[str, float] = field(default_factory=dict)
    worst_rx: float | None = None


@dataclass
class RunRecord:
    name: str
    path: str
    shot: int | None = None
    declared_epochs: int | None = None
    epochs: dict[int, EpochRecord] = field(default_factory=dict)
    final_primary: FinalRecord = field(default_factory=FinalRecord)
    final_best: FinalRecord = field(default_factory=FinalRecord)
    final_avg: dict[str, FinalRecord] = field(default_factory=dict)
    error_markers: list[str] = field(default_factory=list)
    config_lines: list[str] = field(default_factory=list)


def pct(value: str) -> float:
    return float(value)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def extract_shot(name: str) -> int | None:
    match = re.search(r"FS0*([0-9]+)", name.upper())
    if not match:
        return None
    return int(match.group(1))


def extract_family(name: str) -> str:
    return re.sub(r"_[0-9]{3,}$", "", name)


def get_epoch(record: RunRecord, epoch: int | None) -> EpochRecord | None:
    if epoch is None:
        return None
    if epoch not in record.epochs:
        record.epochs[epoch] = EpochRecord(epoch=epoch)
    return record.epochs[epoch]


def parse_final_line(line: str, final: FinalRecord) -> None:
    val_match = re.search(rf"val_tx=({FLOAT_RE})%", line)
    overall_match = re.search(rf"(?:test_overall|overall_tx|overall)=({FLOAT_RE})%", line)
    strict_match = re.search(rf"(?:strict_udu|unseen_day_unseen_rx)=({FLOAT_RE})%", line)
    worst_match = re.search(rf"worst_rx=({FLOAT_RE})%", line)
    if val_match:
        final.val_tx = pct(val_match.group(1))
    if overall_match:
        final.overall_tx = pct(overall_match.group(1))
    if strict_match:
        final.strict_udu = pct(strict_match.group(1))
    if worst_match:
        final.worst_rx = pct(worst_match.group(1))


def parse_log(path: Path) -> RunRecord:
    record = RunRecord(name=path.stem, path=str(path), shot=extract_shot(path.stem))
    current_epoch: int | None = None
    in_test_split = False
    final_scope: tuple[str, str | None] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            for marker in ERROR_MARKERS:
                if marker in line and marker not in record.error_markers:
                    record.error_markers.append(marker)

            if line.startswith("[CONFIG") or line.startswith("[INFO]") or line.startswith("[DATASET]"):
                if len(record.config_lines) < 40:
                    record.config_lines.append(line)

            epoch_match = re.search(r"\[EPOCH-BEGIN\]\s+E([0-9]+)/([0-9]+)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                record.declared_epochs = int(epoch_match.group(2))
                get_epoch(record, current_epoch)
                in_test_split = False
                final_scope = None
                continue

            epoch = get_epoch(record, current_epoch)

            train_match = re.search(rf"\[TRAIN\]\s+tx=({FLOAT_RE})%", line)
            if train_match and epoch is not None:
                epoch.train_tx = pct(train_match.group(1))

            val_match = re.search(rf"\[VAL\]\s+tx=({FLOAT_RE})%", line)
            if val_match and epoch is not None:
                epoch.val_tx = pct(val_match.group(1))

            test_match = re.search(rf"\[TEST\]\s+overall_tx=({FLOAT_RE})%", line)
            if test_match and epoch is not None:
                epoch.test_overall = pct(test_match.group(1))

            if line.startswith("[TEST-SPLIT]"):
                in_test_split = True
                continue

            if in_test_split:
                split_match = re.search(rf"^\s*([A-Za-z0-9_]+)(?:\([^)]*\))?:\s+tx=({FLOAT_RE})%", line)
                if split_match and epoch is not None:
                    split_name = split_match.group(1)
                    if split_name == "unseen_day_unseen_rx":
                        epoch.strict_udu = pct(split_match.group(2))
                    continue
                if line.startswith("[") and not line.startswith("[TEST-SPLIT]"):
                    in_test_split = False

            sat_epoch_match = re.search(rf"\[SAT-TEST\]\s+scenario=([A-Za-z0-9_]+).*strict_udu=({FLOAT_RE})%", line)
            if sat_epoch_match and epoch is not None:
                epoch.sat_strict[sat_epoch_match.group(1)] = pct(sat_epoch_match.group(2))

            is_final_sat_line = "[SAT-TEST]" in line

            if line.startswith("[FINAL-PRIMARY]"):
                final_scope = ("primary", None)
                if not is_final_sat_line:
                    parse_final_line(line, record.final_primary)
            elif line.startswith("[FINAL-BEST]"):
                final_scope = ("best", None)
                if not is_final_sat_line:
                    parse_final_line(line, record.final_best)
            else:
                avg_match = re.search(r"\[FINAL-AVG\]\s+mode=([A-Za-z0-9_]+)", line)
                if avg_match:
                    mode = avg_match.group(1).lower()
                    final_scope = ("avg", mode)
                    final = record.final_avg.setdefault(mode, FinalRecord())
                    if not is_final_sat_line:
                        parse_final_line(line, final)
                avg_mode_match = re.search(r"\[FINAL-AVG\]\[([A-Za-z0-9_]+)\]", line)
                if avg_mode_match:
                    mode = avg_mode_match.group(1).lower()
                    final_scope = ("avg", mode)
                    record.final_avg.setdefault(mode, FinalRecord())

            target_final: FinalRecord | None = None
            if final_scope == ("primary", None):
                target_final = record.final_primary
            elif final_scope == ("best", None):
                target_final = record.final_best
            elif final_scope is not None and final_scope[0] == "avg":
                target_final = record.final_avg.setdefault(final_scope[1] or "unknown", FinalRecord())

            if target_final is not None:
                split_strict_match = re.search(rf"unseen_day_unseen_rx\([^)]*\):\s+tx=({FLOAT_RE})%", line)
                if split_strict_match and not is_final_sat_line:
                    target_final.strict_udu = pct(split_strict_match.group(1))

                rx_match = re.search(rf"rx=([0-9]+)\s+on unseen_days[^:]*:\s+tx=({FLOAT_RE})%", line)
                if rx_match:
                    target_final.rx_unseen[rx_match.group(1)] = pct(rx_match.group(2))

                final_sat_match = re.search(rf"\[SAT-TEST\]\s+scenario=([A-Za-z0-9_]+).*strict_udu=({FLOAT_RE})%", line)
                if final_sat_match:
                    target_final.sat_strict[final_sat_match.group(1)] = pct(final_sat_match.group(2))

    return record


def finite_values(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None and not math.isnan(value)]


def nearest_threshold(shot: int | None) -> dict[str, float]:
    if shot in THRESHOLDS:
        return THRESHOLDS[shot]  # type: ignore[index]
    if shot is None:
        return THRESHOLDS[10]
    nearest = min(THRESHOLDS, key=lambda key: abs(key - shot))
    return THRESHOLDS[nearest]


def summarize_run(record: RunRecord, late_window: int) -> dict[str, Any]:
    thresholds = nearest_threshold(record.shot)
    sorted_epochs = [record.epochs[key] for key in sorted(record.epochs)]
    max_epoch = sorted_epochs[-1].epoch if sorted_epochs else None
    completed = bool(
        max_epoch is not None
        and (
            record.declared_epochs is None
            or max_epoch >= record.declared_epochs - 1
            or record.final_primary.val_tx is not None
        )
    )

    val_series = [(epoch.epoch, epoch.val_tx) for epoch in sorted_epochs if epoch.val_tx is not None]
    strict_series = [(epoch.epoch, epoch.strict_udu) for epoch in sorted_epochs if epoch.strict_udu is not None]
    sat_values_epoch = [value for epoch in sorted_epochs for value in epoch.sat_strict.values()]

    best_val_epoch, best_val = max(val_series, key=lambda item: item[1]) if val_series else (None, None)
    final_val = record.final_primary.val_tx
    if final_val is None and val_series:
        final_val = val_series[-1][1]
    val_drop = (best_val - final_val) if best_val is not None and final_val is not None else None

    best_strict_epoch, best_strict = max(strict_series, key=lambda item: item[1]) if strict_series else (None, None)
    latest_strict_epoch, latest_strict = strict_series[-1] if strict_series else (None, None)
    final_strict = record.final_primary.strict_udu
    if final_strict is None and strict_series:
        final_strict = strict_series[-1][1]
    strict_drop = (best_strict - final_strict) if best_strict is not None and final_strict is not None else None
    latest_strict_drop = (
        best_strict - latest_strict if best_strict is not None and latest_strict is not None else None
    )

    last_epoch_cut = (max_epoch - late_window + 1) if max_epoch is not None else None
    late_vals = [value for epoch, value in val_series if last_epoch_cut is None or epoch >= last_epoch_cut]
    late_stricts = [value for epoch, value in strict_series if last_epoch_cut is None or epoch >= last_epoch_cut]
    late_val_std = pstdev(late_vals) if len(late_vals) >= 2 else None
    late_strict_span = (max(late_stricts) - min(late_stricts)) if len(late_stricts) >= 2 else None

    primary_rx_values = list(record.final_primary.rx_unseen.values())
    if record.final_primary.worst_rx is not None:
        primary_rx_values.append(record.final_primary.worst_rx)
    primary_rx_floor = min(primary_rx_values) if primary_rx_values else None

    primary_sat_values = list(record.final_primary.sat_strict.values())
    if not primary_sat_values:
        primary_sat_values = sat_values_epoch[-4:]
    primary_sat_floor = min(primary_sat_values) if primary_sat_values else None

    swad = record.final_avg.get("swad")
    ema = record.final_avg.get("ema")
    swad_gap = (
        final_strict - swad.strict_udu
        if final_strict is not None and swad is not None and swad.strict_udu is not None
        else None
    )
    ema_gap = (
        final_strict - ema.strict_udu
        if final_strict is not None and ema is not None and ema.strict_udu is not None
        else None
    )

    checks: dict[str, tuple[str, str, bool]] = {}

    def check(name: str, ok: bool | None, detail: str, missing_is_warn: bool = True, hard: bool = True) -> None:
        if ok is None:
            checks[name] = ("WARN" if missing_is_warn else "FAIL", detail, hard)
        else:
            checks[name] = ("PASS" if ok else "FAIL", detail, hard)

    check("completed", completed and not record.error_markers, f"max_epoch={max_epoch}, errors={len(record.error_markers)}", False)
    check("val_drop", val_drop is not None and val_drop <= thresholds["val_drop"], f"{val_drop} <= {thresholds['val_drop']}")
    check(
        "strict_drop",
        strict_drop is not None and strict_drop <= thresholds["strict_drop"],
        f"{strict_drop} <= {thresholds['strict_drop']}",
    )
    check(
        "latest_strict_drop",
        latest_strict_drop is not None and latest_strict_drop <= thresholds["latest_strict_drop"],
        f"{latest_strict_drop} <= {thresholds['latest_strict_drop']}",
    )
    check(
        "late_val_std",
        late_val_std is None or late_val_std <= thresholds["val_late_std"],
        f"{late_val_std} <= {thresholds['val_late_std']}",
    )
    check(
        "late_strict_span",
        late_strict_span is None or late_strict_span <= thresholds["strict_late_span"],
        f"{late_strict_span} <= {thresholds['strict_late_span']}",
    )
    check("swad_gap", swad_gap is None or swad_gap <= thresholds["swad_gap"], f"{swad_gap} <= {thresholds['swad_gap']}")
    check(
        "ema_gap",
        ema_gap is None or ema_gap <= thresholds["ema_gap"],
        f"{ema_gap} <= {thresholds['ema_gap']}",
        hard=False,
    )
    check(
        "rx_floor",
        primary_rx_floor is None or primary_rx_floor >= thresholds["rx_floor"],
        f"{primary_rx_floor} >= {thresholds['rx_floor']}",
        hard=False,
    )
    check(
        "sat_floor",
        primary_sat_floor is None or primary_sat_floor >= thresholds["sat_floor"],
        f"{primary_sat_floor} >= {thresholds['sat_floor']}",
        hard=False,
    )

    hard_failures = [key for key, (status, _, hard) in checks.items() if hard and status == "FAIL"]
    stability_status = "PASS" if not hard_failures else "FAIL"

    promotion_failures: list[str] = []
    if stability_status != "PASS":
        promotion_failures.extend(hard_failures)
    if final_val is None or final_val < thresholds["target_val"]:
        promotion_failures.append("target_val")
    if final_strict is None or final_strict < thresholds["target_strict"]:
        promotion_failures.append("target_strict")
    if primary_rx_floor is None or primary_rx_floor < thresholds["rx_floor"]:
        promotion_failures.append("rx_floor")
    if primary_sat_floor is None or primary_sat_floor < thresholds["sat_floor"]:
        promotion_failures.append("sat_floor")
    promotion_status = "PASS" if not promotion_failures else "FAIL"

    collapse_index = 0.0
    for metric, gate in (
        (val_drop, thresholds["val_drop"]),
        (strict_drop, thresholds["strict_drop"]),
        (late_val_std, thresholds["val_late_std"]),
        (late_strict_span, thresholds["strict_late_span"]),
        (swad_gap, thresholds["swad_gap"]),
        (ema_gap, thresholds["ema_gap"]),
    ):
        if metric is not None and gate > 0:
            collapse_index += max(0.0, metric / gate - 1.0)
    if primary_rx_floor is not None and thresholds["rx_floor"] > 0:
        collapse_index += max(0.0, thresholds["rx_floor"] / max(primary_rx_floor, 1e-6) - 1.0)
    if primary_sat_floor is not None and thresholds["sat_floor"] > 0:
        collapse_index += max(0.0, thresholds["sat_floor"] / max(primary_sat_floor, 1e-6) - 1.0)

    return {
        "run": record.name,
        "family": extract_family(record.name),
        "path": record.path,
        "shot": record.shot,
        "stability_status": stability_status,
        "promotion_status": promotion_status,
        "failure_reasons": ";".join(hard_failures),
        "promotion_reasons": ";".join(promotion_failures),
        "collapse_index": round(collapse_index, 4),
        "max_epoch": max_epoch,
        "declared_epochs": record.declared_epochs,
        "best_val": safe_float(best_val),
        "best_val_epoch": best_val_epoch,
        "final_val": safe_float(final_val),
        "val_drop": safe_float(val_drop),
        "best_strict": safe_float(best_strict),
        "best_strict_epoch": best_strict_epoch,
        "latest_strict": safe_float(latest_strict),
        "latest_strict_epoch": latest_strict_epoch,
        "final_strict": safe_float(final_strict),
        "strict_drop": safe_float(strict_drop),
        "latest_strict_drop": safe_float(latest_strict_drop),
        "late_val_std": safe_float(late_val_std),
        "late_strict_span": safe_float(late_strict_span),
        "swad_strict": safe_float(swad.strict_udu if swad else None),
        "swad_gap": safe_float(swad_gap),
        "ema_strict": safe_float(ema.strict_udu if ema else None),
        "ema_gap": safe_float(ema_gap),
        "rx_floor": safe_float(primary_rx_floor),
        "sat_floor": safe_float(primary_sat_floor),
        "target_val": thresholds["target_val"],
        "target_strict": thresholds["target_strict"],
        "checks_json": json.dumps(checks, ensure_ascii=False, sort_keys=True),
    }


def load_matrix(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {str(item.get("run_name", item.get("id", ""))): item for item in payload if isinstance(item, dict)}
    if isinstance(payload, dict):
        matrix = payload.get("candidates", payload.get("runs", payload))
        if isinstance(matrix, list):
            return {str(item.get("run_name", item.get("id", ""))): item for item in matrix if isinstance(item, dict)}
    return {}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, log_dir: Path, late_window: int) -> None:
    def fmt(value: Any) -> str:
        number = safe_float(value)
        if number is None:
            return "nan"
        return f"{number:.2f}"

    def fmt4(value: Any) -> str:
        number = safe_float(value)
        if number is None:
            return "nan"
        return f"{number:.4f}"

    lines: list[str] = []
    lines.append("# CEN51 Few-Shot Stability Validation")
    lines.append("")
    lines.append(f"- log_dir: `{log_dir}`")
    lines.append(f"- late_window: `{late_window}` epochs")
    lines.append(f"- runs: {len(rows)}")
    lines.append("")
    if not rows:
        lines.append("No logs were parsed.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    by_shot: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        shot = int(row["shot"] or 0)
        by_shot.setdefault(shot, []).append(row)

    lines.append("## Shot Summary")
    lines.append("")
    lines.append("| shot | runs | stability_pass | promotion_pass | best_final_strict | best_final_val | lowest_collapse_index |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for shot in sorted(by_shot):
        group = by_shot[shot]
        stability_pass = sum(1 for row in group if row["stability_status"] == "PASS")
        promotion_pass = sum(1 for row in group if row["promotion_status"] == "PASS")
        best_final_strict = max((safe_float(row["final_strict"]) or float("-inf")) for row in group)
        best_final_val = max((safe_float(row["final_val"]) or float("-inf")) for row in group)
        lowest_collapse = min((safe_float(row["collapse_index"]) or 0.0) for row in group)
        lines.append(
            f"| {shot} | {len(group)} | {stability_pass} | {promotion_pass} | "
            f"{best_final_strict:.2f} | {best_final_val:.2f} | {lowest_collapse:.4f} |"
        )

    lines.append("")
    lines.append("## Family Summary")
    lines.append("")
    lines.append("| shot | family | runs | stable | promotable | mean_val | mean_strict | strict_std | max_latest_drop | family_status |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    by_family: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        shot = int(row["shot"] or 0)
        by_family.setdefault((shot, str(row["family"])), []).append(row)
    for (shot, family), group in sorted(by_family.items()):
        vals = finite_values([safe_float(row["final_val"]) for row in group])
        stricts = finite_values([safe_float(row["final_strict"]) for row in group])
        latest_drops = finite_values([safe_float(row["latest_strict_drop"]) for row in group])
        stable_count = sum(1 for row in group if row["stability_status"] == "PASS")
        promote_count = sum(1 for row in group if row["promotion_status"] == "PASS")
        strict_std = pstdev(stricts) if len(stricts) >= 2 else None
        std_gate = FAMILY_STRICT_STD_MAX.get(shot, 1.50)
        if promote_count >= 2 and stable_count == len(group) and strict_std is not None and strict_std <= std_gate:
            family_status = "DEFAULT_READY"
        elif promote_count >= 1:
            family_status = "NEED_SEED_CONFIRMATION"
        elif stable_count >= 1:
            family_status = "STABLE_BUT_WEAK"
        else:
            family_status = "REJECT_LATE"
        lines.append(
            f"| {shot} | `{family}` | {len(group)} | {stable_count} | {promote_count} | "
            f"{fmt(mean(vals) if vals else None)} | "
            f"{fmt(mean(stricts) if stricts else None)} | "
            f"{fmt(strict_std)} | "
            f"{fmt(max(latest_drops) if latest_drops else None)} | {family_status} |"
        )

    lines.append("")
    lines.append("## Candidate Ranking")
    lines.append("")
    lines.append(
        "| rank | run | shot | stability | promote | final_val | final_strict | latest_strict | val_drop | "
        "latest_drop | rx_floor | sat_floor | collapse_index | reasons |"
    )
    lines.append("|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    ranked = sorted(
        rows,
        key=lambda row: (
            row["promotion_status"] != "PASS",
            row["stability_status"] != "PASS",
            -(safe_float(row["final_strict"]) or -1.0),
            safe_float(row["collapse_index"]) or 999.0,
        ),
    )
    for idx, row in enumerate(ranked, start=1):
        reasons = row["promotion_reasons"] or row["failure_reasons"] or "-"
        lines.append(
            f"| {idx} | `{row['run']}` | {row['shot']} | {row['stability_status']} | {row['promotion_status']} | "
            f"{fmt(row['final_val'])} | "
            f"{fmt(row['final_strict'])} | "
            f"{fmt(row['latest_strict'])} | "
            f"{fmt(row['val_drop'])} | "
            f"{fmt(row['latest_strict_drop'])} | "
            f"{fmt(row['rx_floor'])} | "
            f"{fmt(row['sat_floor'])} | "
            f"{fmt4(row['collapse_index'])} | {reasons} |"
        )

    lines.append("")
    lines.append("## Gate Meaning")
    lines.append("")
    lines.append("- stability_status checks late collapse: completion, error markers, val drop, primary strict drop, latest strict drop, late-window oscillation, and SWAD gap.")
    lines.append("- promotion_status additionally requires shot-specific final validation, strict UDU, receiver-floor, and satellite-floor targets.")
    lines.append("- family_status marks DEFAULT_READY only when at least two seeds in the same family are promotable, all parsed seeds are stable, and strict standard deviation is below the shot-specific family gate.")
    lines.append("- A run can be stable but not promotable; that means the schedule did not collapse, but its accuracy target is still insufficient.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--matrix-json", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--late-window", type=int, default=30)
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing reports.")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not log_dir.exists():
        raise SystemExit(f"log dir does not exist: {log_dir}")

    _matrix = load_matrix(args.matrix_json)
    logs = sorted(
        path
        for path in log_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".log", ".txt", ".out"}
    )
    rows = [summarize_run(parse_log(path), late_window=args.late_window) for path in logs]
    rows.sort(key=lambda row: (int(row["shot"] or 0), str(row["run"])))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "stability_summary.csv")
    (args.out_dir / "stability_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(rows, args.out_dir / "stability_report.md", log_dir=log_dir, late_window=args.late_window)

    failed = [row for row in rows if row["stability_status"] != "PASS"]
    promoted = [row for row in rows if row["promotion_status"] == "PASS"]
    print(f"[STABILITY] parsed={len(rows)} stability_pass={len(rows) - len(failed)} promotion_pass={len(promoted)}")
    print(f"[STABILITY] report={args.out_dir / 'stability_report.md'}")
    if failed and not args.no_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
