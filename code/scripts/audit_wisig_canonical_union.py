#!/usr/bin/env python
"""Build a read-only audit of the canonical WiSig asset union."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.wisig_canonical_inventory import build_inventory  # noqa: E402


def _parse_assets(raw_assets: list[str]) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for raw_asset in raw_assets:
        if raw_asset.count("=") != 1:
            raise ValueError(f"malformed --asset value: {raw_asset!r}")
        name, path_text = raw_asset.split("=", 1)
        if not name or not path_text:
            raise ValueError(f"malformed --asset value: {raw_asset!r}")
        if name in assets:
            raise ValueError(f"duplicate asset name: {name}")
        assets[name] = Path(path_text)
    if not assets:
        raise ValueError("at least one --asset is required")
    return assets


def _write_json_exclusive(path: Path, payload: dict[str, object], created_paths: list[Path]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        created_paths.append(path)
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _write_csv_exclusive(
    path: Path,
    fieldnames: list[str],
    rows: list[tuple[object, ...]],
    created_paths: list[Path],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        created_paths.append(path)
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def _read_coverage_rows(sqlite_path: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(sqlite_path) as connection:
        return connection.execute(
            """
            SELECT canonical.tx_id, canonical.rx_id, canonical.day_id,
                   COUNT(DISTINCT canonical.physical_sample_id) AS record_count,
                   COUNT(DISTINCT sources.asset_name) AS asset_count
            FROM canonical_records AS canonical
            JOIN record_sources AS sources USING (physical_sample_id)
            GROUP BY canonical.tx_id, canonical.rx_id, canonical.day_id
            ORDER BY canonical.tx_id, canonical.rx_id, canonical.day_id
            """
        ).fetchall()


def _read_conflict_rows(sqlite_path: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(sqlite_path) as connection:
        return connection.execute(
            """
            SELECT physical_sample_id, first_iq_sha256, conflicting_iq_sha256, asset_name
            FROM identity_conflicts
            ORDER BY physical_sample_id, first_iq_sha256, conflicting_iq_sha256, asset_name
            """
        ).fetchall()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--sqlite-out", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--coverage-csv", required=True)
    parser.add_argument("--conflicts-csv", required=True)
    parser.add_argument("--equalized", default="1")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    output_paths = [
        Path(args.sqlite_out),
        Path(args.summary_json),
        Path(args.coverage_csv),
        Path(args.conflicts_csv),
    ]
    created_paths: list[Path] = []
    try:
        assets = _parse_assets(args.asset)
        if len({path.resolve() for path in output_paths}) != len(output_paths):
            raise ValueError("declared output paths must be distinct")
        existing_paths = [path for path in output_paths if path.exists()]
        if existing_paths:
            raise FileExistsError(f"declared output already exists: {existing_paths[0]}")
        summary = build_inventory(assets, output_paths[0], equalized=args.equalized)
        created_paths.append(output_paths[0])
        summary_payload: dict[str, object] = {
            "protocol_schema": "p2_min_v1",
            "equalized": str(args.equalized),
            **asdict(summary),
        }
        _write_json_exclusive(output_paths[1], summary_payload, created_paths)
        _write_csv_exclusive(
            output_paths[2],
            ["tx_id", "rx_id", "day_id", "record_count", "asset_count"],
            _read_coverage_rows(output_paths[0]),
            created_paths,
        )
        _write_csv_exclusive(
            output_paths[3],
            ["physical_sample_id", "first_iq_sha256", "conflicting_iq_sha256", "asset_name"],
            _read_conflict_rows(output_paths[0]),
            created_paths,
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        for path in reversed(created_paths):
            if path.exists():
                path.unlink()
        print(f"audit failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
