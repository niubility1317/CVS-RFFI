#!/usr/bin/env python
"""Build deterministic maximal-class canonical Phase2 split manifests."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_canonical_split import (  # noqa: E402
    CanonicalProfile,
    assign_scenes,
    build_split_manifest,
    rank_new_classes,
)


CLASS_SELECTION_SCHEMA = "cvs.phase2.canonical_class_selection.v1"


def _open_inventory_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"canonical inventory does not exist: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _eligible_records(connection: sqlite3.Connection) -> list[dict[str, object]]:
    columns = (
        "physical_sample_id",
        "tx_id",
        "rx_id",
        "day_id",
        "preferred_asset",
        "preferred_source_record_index",
        "eligible",
    )
    return [
        dict(zip(columns, row))
        for row in connection.execute(
            """
            SELECT physical_sample_id, tx_id, rx_id, day_id,
                   preferred_asset, preferred_source_record_index, eligible
            FROM canonical_records
            WHERE eligible = 1
            ORDER BY physical_sample_id
            """
        )
    ]


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    inventory_path = Path(args.inventory)
    profile_path = Path(args.profile)
    out_root = Path(args.out_root)
    if out_root.exists():
        print(f"split build failed: output root already exists: {out_root}", file=sys.stderr)
        return 2

    connection: sqlite3.Connection | None = None
    try:
        profile_payload = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        profile = CanonicalProfile.from_mapping(profile_payload)
        connection = _open_inventory_read_only(inventory_path)
        records = _eligible_records(connection)
        assignments = assign_scenes(records, seed=args.seed)
        plain_scenes = {
            sample_id: metadata.scene for sample_id, metadata in assignments.items()
        }
        class_selection = rank_new_classes(connection, profile, plain_scenes)
        manifests = {
            (policy, k): build_split_manifest(
                connection,
                profile,
                k,
                policy,
                scene_assignments=assignments,
                class_selection=class_selection,
                support_seed=args.seed,
            )
            for policy in profile.query_policies
            for k in profile.k_values
        }
        class_selection_payload = {
            "schema": CLASS_SELECTION_SCHEMA,
            "protocol_schema": profile.protocol_schema,
            "profile_id": profile.source_profile_id,
            "seed": args.seed,
            **{
                f"Y_new{size}": list(class_selection[size])
                for size in profile.new_class_sizes
            },
        }
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"split build failed: {error}", file=sys.stderr)
        return 2
    finally:
        if connection is not None:
            connection.close()

    try:
        out_root.mkdir()
        for policy in profile.query_policies:
            (out_root / policy).mkdir()
        _write_json_exclusive(out_root / "class_selection.json", class_selection_payload)
        for policy in profile.query_policies:
            for k in profile.k_values:
                _write_json_exclusive(
                    out_root / policy / f"k{k}.json",
                    manifests[(policy, k)].to_mapping(),
                )
    except OSError as error:
        print(f"split build failed while writing exclusive outputs: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
