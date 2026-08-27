#!/usr/bin/env python3
"""Summarize independently scored canonical Phase2 prediction rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from cvsrffi.phase2_canonical_summary import (
    CanonicalSummaryError,
    summarize_scored_rows,
)
from cvsrffi.stage2_metric_scorer import FORMAL_PREDICTIONS_SCHEMA


CSV_COLUMNS = (
    "true_class_index",
    "receiver_label",
    "day_label",
    "scenario",
    "sample_count",
    "correct_count",
    "accuracy",
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize truth-last canonical Phase2 scored rows."
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    return parser.parse_args(argv)


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise CanonicalSummaryError(f"cannot read input {path}: {exc}") from exc
    if not text.strip():
        raise CanonicalSummaryError(f"input is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanonicalSummaryError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise CanonicalSummaryError(
                    f"JSONL row at {path}:{line_number} must be an object"
                )
            rows.append(row)
        if not rows:
            raise CanonicalSummaryError(f"input has no JSONL rows: {path}")
        return rows

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise CanonicalSummaryError(f"JSON root must be an object or list: {path}")
    if "schema" in payload or "predictions" in payload:
        if set(payload) != {"schema", "predictions"}:
            raise CanonicalSummaryError(f"formal prediction object schema drift: {path}")
        if payload["schema"] != FORMAL_PREDICTIONS_SCHEMA:
            raise CanonicalSummaryError(f"formal prediction schema mismatch: {path}")
        predictions = payload["predictions"]
        if not isinstance(predictions, list):
            raise CanonicalSummaryError(f"formal predictions must be a list: {path}")
        return predictions
    return [payload]


def _write_outputs(out_root: Path, summary: Mapping[str, Any]) -> None:
    if out_root.exists():
        raise FileExistsError(f"output root already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    summary_path = out_root / "summary.json"
    cell_path = out_root / "cell_metrics.csv"
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    with cell_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary["cell_metrics"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        rows: list[Mapping[str, Any]] = []
        for input_path in args.input:
            rows.extend(_load_rows(input_path))
        summary = summarize_scored_rows(rows)
        _write_outputs(args.out_root, summary)
    except (CanonicalSummaryError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
