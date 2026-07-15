#!/usr/bin/env python
"""Collect all locked Stage2-C benchmark cells into two immutable CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "cvs_stage2c_effective8_generated_execution_plan_v1"
EXPECTED_BENCHMARK_INVOCATIONS = 300
EXPECTED_FORMAL_ROWS = 900


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _benchmark_out_dir(command: Sequence[Any]) -> Path:
    values = [str(value) for value in command]
    if "--out_dir" not in values:
        raise ValueError(f"benchmark command lacks --out_dir: {values}")
    index = values.index("--out_dir")
    if index + 1 >= len(values):
        raise ValueError(f"benchmark command has empty --out_dir: {values}")
    return Path(values[index + 1])


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def collect_outputs(
    plan_manifest: Path,
    *,
    out_dir: Path,
    expected_invocations: int = EXPECTED_BENCHMARK_INVOCATIONS,
    expected_formal_rows: int = EXPECTED_FORMAL_ROWS,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite formal evidence directory: {out_dir}")
    plan = json.loads(plan_manifest.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("generated formal plan schema drift")
    counts = dict(plan.get("expected_counts", {}))
    benchmark_commands = list(dict(plan.get("commands", {})).get("benchmark", []))
    if (
        int(counts.get("benchmark_invocations", -1)) != int(expected_invocations)
        or len(benchmark_commands) != int(expected_invocations)
        or int(counts.get("formal_scenario_rows", -1)) != int(expected_formal_rows)
    ):
        raise ValueError("formal plan benchmark-count contract drift")

    combined_rows: list[dict[str, str]] = []
    combined_predictions: list[dict[str, str]] = []
    row_fields: list[str] | None = None
    prediction_fields: list[str] | None = None
    inputs: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()
    for command_index, command in enumerate(benchmark_commands):
        result_dir = _benchmark_out_dir(command)
        normalized_dir = str(result_dir)
        if normalized_dir in seen_dirs:
            raise ValueError(f"duplicate benchmark output directory: {result_dir}")
        seen_dirs.add(normalized_dir)
        row_path = result_dir / "formal_rows.csv"
        prediction_path = result_dir / "formal_predictions.csv"
        if not row_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"formal benchmark evidence is missing: {result_dir}")
        current_row_fields, current_rows = _read_csv(row_path)
        current_prediction_fields, current_predictions = _read_csv(prediction_path)
        if len(current_rows) != 3 or not current_predictions:
            raise ValueError(f"formal benchmark evidence cardinality drift: {result_dir}")
        if row_fields is None:
            row_fields = current_row_fields
            prediction_fields = current_prediction_fields
        elif current_row_fields != row_fields or current_prediction_fields != prediction_fields:
            raise ValueError(f"formal CSV schema drift: {result_dir}")
        combined_rows.extend(current_rows)
        combined_predictions.extend(current_predictions)
        inputs.append(
            {
                "command_index": int(command_index),
                "result_dir": normalized_dir,
                "formal_rows_sha256": _sha256_file(row_path),
                "formal_predictions_sha256": _sha256_file(prediction_path),
                "formal_row_count": len(current_rows),
                "formal_prediction_count": len(current_predictions),
            }
        )
    if len(combined_rows) != int(expected_formal_rows):
        raise ValueError("combined formal-row count drift")
    row_keys = {
        (
            row.get("receiver", ""),
            row.get("seed", ""),
            row.get("scenario", ""),
            row.get("new_class_count", ""),
            row.get("k_shot", ""),
        )
        for row in combined_rows
    }
    if len(row_keys) != len(combined_rows):
        raise ValueError("combined formal rows contain duplicate matrix cells")
    prediction_keys = {
        (
            row.get("receiver", ""),
            row.get("seed", ""),
            row.get("scenario", ""),
            row.get("new_class_count", ""),
            row.get("k_shot", ""),
            row.get("query_id", ""),
        )
        for row in combined_predictions
    }
    if len(prediction_keys) != len(combined_predictions):
        raise ValueError("combined formal predictions contain duplicate query cells")

    out_dir.mkdir(parents=True, exist_ok=False)
    rows_out = out_dir / "formal_rows.csv"
    predictions_out = out_dir / "formal_predictions.csv"
    _write_csv(rows_out, row_fields or (), combined_rows)
    _write_csv(predictions_out, prediction_fields or (), combined_predictions)
    manifest = {
        "schema": "cvs_stage2c_formal_evidence_collection_v1",
        "plan_manifest": str(plan_manifest),
        "plan_manifest_sha256": _sha256_file(plan_manifest),
        "benchmark_invocations": len(benchmark_commands),
        "formal_row_count": len(combined_rows),
        "formal_prediction_count": len(combined_predictions),
        "formal_rows": str(rows_out),
        "formal_rows_sha256": _sha256_file(rows_out),
        "formal_predictions": str(predictions_out),
        "formal_predictions_sha256": _sha256_file(predictions_out),
        "input_artifacts": inputs,
    }
    (out_dir / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan_manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = collect_outputs(args.plan_manifest, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
