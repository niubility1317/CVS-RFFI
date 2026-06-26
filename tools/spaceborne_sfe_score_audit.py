from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


KNOWN_GROUPS = ("old", "new")


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(value: object, default: int = -1) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _read_score_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source_file"] = str(path)
                row["true_label"] = _parse_int(row.get("true_label"))
                row["predicted_label"] = _parse_int(row.get("predicted_label"))
                row["accepted"] = _parse_bool(row.get("accepted"))
                row["true_group"] = str(row.get("true_group") or "other_known").strip() or "other_known"
                row["query_tx_id"] = str(row.get("query_tx_id") or "")
                row["gate_reason"] = str(row.get("gate_reason") or "")
                for field in (
                    "seen_new_evidence",
                    "seen_new_support_affinity",
                    "seen_new_support_residual",
                    "seen_new_anchor_similarity",
                    "seen_new_anchor_delta",
                ):
                    row[field] = _parse_float(row.get(field))
                rows.append(row)
    return rows


def _row_is_correct(row: dict) -> bool:
    group = row["true_group"]
    if group == "unknown":
        return (not row["accepted"]) or row["predicted_label"] == -1
    return row["accepted"] and row["predicted_label"] == row["true_label"]


def _outcome(row: dict, old_labels: set[int], new_labels: set[int]) -> str:
    group = row["true_group"]
    pred = row["predicted_label"]
    if group == "unknown":
        if row["accepted"] and pred != -1:
            return "unknown_false_accept"
        return "unknown_rejected"
    if _row_is_correct(row):
        return f"{group}_correct"
    if (not row["accepted"]) or pred == -1:
        return "known_rejected"
    if group == "new" and pred in old_labels:
        return "new_to_old"
    if group == "new":
        return "new_to_other_known"
    if group == "old" and pred in new_labels:
        return "old_to_new"
    if group == "old":
        return "old_to_other_known"
    return "known_to_wrong_known"


def _summarize_bucket(key_name: str, rows: list[dict], key_fn) -> list[dict]:
    buckets: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    summary = []
    for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0])):
        count = len(bucket)
        accepted = sum(1 for row in bucket if row["accepted"])
        correct = sum(1 for row in bucket if _row_is_correct(row))
        item = {
            key_name: key,
            "count": count,
            "accepted": accepted,
            "rejected": count - accepted,
            "correct": correct,
            "accuracy": _safe_rate(correct, count),
            "accepted_rate": _safe_rate(accepted, count),
        }
        summary.append(item)
    return summary


def _summarize_evidence(rows: list[dict]) -> list[dict]:
    summary = []
    fields = (
        "seen_new_evidence",
        "seen_new_support_affinity",
        "seen_new_support_residual",
        "seen_new_anchor_similarity",
        "seen_new_anchor_delta",
    )
    groups = _group_by(rows, lambda row: row["true_group"])
    for group, bucket_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        item = {"true_group": group, "count": len(bucket_rows)}
        for field in fields:
            values = [float(row[field]) for row in bucket_rows if row.get(field) is not None]
            item[f"{field}_count"] = len(values)
            item[f"{field}_mean"] = _safe_mean(values)
            item[f"{field}_min"] = min(values) if values else None
            item[f"{field}_max"] = max(values) if values else None
        summary.append(item)
    return summary


def summarize_score_tables(score_tables: Iterable[str | Path]) -> dict:
    paths = [Path(path) for path in score_tables]
    rows = _read_score_rows(paths)
    old_labels = {row["true_label"] for row in rows if row["true_group"] == "old"}
    new_labels = {row["true_label"] for row in rows if row["true_group"] == "new"}

    known_rows = [row for row in rows if row["true_group"] in KNOWN_GROUPS]
    unknown_rows = [row for row in rows if row["true_group"] == "unknown"]
    old_rows = [row for row in rows if row["true_group"] == "old"]
    new_rows = [row for row in rows if row["true_group"] == "new"]
    outcomes = Counter(_outcome(row, old_labels, new_labels) for row in rows)
    new_outcomes = Counter(_outcome(row, old_labels, new_labels) for row in new_rows)

    known_correct = sum(1 for row in known_rows if _row_is_correct(row))
    unknown_rejected = sum(1 for row in unknown_rows if _row_is_correct(row))
    unknown_false_accept = len(unknown_rows) - unknown_rejected
    new_correct = sum(1 for row in new_rows if _row_is_correct(row))
    old_correct = sum(1 for row in old_rows if _row_is_correct(row))

    confusion_counts: Counter[tuple[str, str]] = Counter()
    gate_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        confusion_counts[(row["true_group"], _outcome(row, old_labels, new_labels))] += 1
        gate_counts[(row["true_group"], row["gate_reason"])] += 1

    group_summary = _summarize_bucket("true_group", rows, lambda row: row["true_group"])
    per_tx_summary = []
    for (group, tx_id), bucket_rows in sorted(
        defaultdict(list, _group_by(rows, lambda row: (row["true_group"], row["query_tx_id"]))).items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        item = _summarize_bucket("query_tx_id", bucket_rows, lambda row: row["query_tx_id"])[0]
        item["true_group"] = group
        per_tx_summary.append(item)

    return {
        "files": [str(path) for path in paths],
        "overall": {
            "rows": len(rows),
            "known_rows": len(known_rows),
            "unknown_rows": len(unknown_rows),
            "known_accuracy": _safe_rate(known_correct, len(known_rows)),
            "old_accuracy": _safe_rate(old_correct, len(old_rows)),
            "new_accuracy": _safe_rate(new_correct, len(new_rows)),
            "unknown_rejection_rate": _safe_rate(unknown_rejected, len(unknown_rows)),
            "unknown_false_accept_rate": _safe_rate(unknown_false_accept, len(unknown_rows)),
            "new_to_old_count": int(outcomes["new_to_old"]),
            "new_rejected_count": int(new_outcomes["known_rejected"]),
            "unknown_false_accept_count": int(outcomes["unknown_false_accept"]),
        },
        "group_summary": group_summary,
        "confusion_summary": [
            {"true_group": group, "outcome": outcome, "count": count}
            for (group, outcome), count in sorted(confusion_counts.items())
        ],
        "gate_reason_summary": [
            {"true_group": group, "gate_reason": reason, "count": count}
            for (group, reason), count in sorted(gate_counts.items())
        ],
        "seen_new_evidence_summary": _summarize_evidence(rows),
        "per_tx_summary": per_tx_summary,
    }


def _group_by(rows: list[dict], key_fn) -> dict[object, list[dict]]:
    buckets: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    return buckets


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate CVS-SFE score-table diagnostics.")
    parser.add_argument("--score_table", action="append", required=True, help="Path to a score_table CSV. Repeatable.")
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_group_csv", type=Path, default=None)
    parser.add_argument("--out_confusion_csv", type=Path, default=None)
    parser.add_argument("--out_gate_csv", type=Path, default=None)
    parser.add_argument("--out_per_tx_csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = summarize_score_tables(args.score_table)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.out_group_csv:
        _write_csv(args.out_group_csv, summary["group_summary"])
    if args.out_confusion_csv:
        _write_csv(args.out_confusion_csv, summary["confusion_summary"])
    if args.out_gate_csv:
        _write_csv(args.out_gate_csv, summary["gate_reason_summary"])
    if args.out_per_tx_csv:
        _write_csv(args.out_per_tx_csv, summary["per_tx_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
