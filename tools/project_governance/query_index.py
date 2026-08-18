"""Regenerable, read-only query index for completed governance scans."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


INDEX_SCHEMA_VERSION = 1

ASSET_HEADERS = (
    "asset_id",
    "scan_id",
    "location",
    "root_id",
    "relative_path",
    "display_name",
    "escaped_name",
    "asset_kind",
    "size_bytes",
    "mtime_utc",
    "access_status",
    "hash_status",
    "sha256",
    "experiment_id",
    "git_ownership",
    "evidence_role",
    "retention_class",
    "recommended_action",
    "decision_reason",
)
EXPERIMENT_HEADERS = (
    "experiment_id",
    "run_id",
    "experiment_state",
    "phase",
    "method_or_candidate",
    "report_path",
    "local_artifact_paths",
    "n607_artifact_paths",
    "git_commit",
    "process_evidence",
    "prediction_count",
    "score_count",
    "expected_artifacts",
    "observed_artifacts",
    "closure_gaps",
)
GIT_HEADERS = (
    "asset_id",
    "ownership",
    "repository_root",
    "common_git_dir",
    "branch",
    "head_commit",
    "status_summary",
    "linked_worktrees",
    "error",
)
RETENTION_HEADERS = (
    "asset_id",
    "retention_class",
    "rule_code",
    "reason",
    "evidence_asset_ids",
    "recommended_action",
)
DELETION_HEADERS = (
    "candidate_id",
    "location",
    "absolute_path",
    "asset_kind",
    "size_bytes",
    "reason",
    "evidence",
    "dependencies",
    "recoverability",
    "estimated_space_reclaim",
    "approval_state",
    "approved_scope",
    "execution_state",
)

REQUIRED_CSV = (
    "asset_inventory_local.csv",
    "asset_inventory_n607.csv",
    "experiment_index.csv",
    "git_ownership.csv",
    "retention_decisions.csv",
    "deletion_candidates.csv",
)


@dataclass(frozen=True)
class IndexBuildSummary:
    """Verified result of one offline index build."""

    scan_id: str
    database_path: Path
    table_counts: Mapping[str, int]


def _load_terminal_receipt(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable governance receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("governance receipt must be a JSON object")
    return payload


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _resolve_csv_sources(
    receipt: Mapping[str, object], external_root: Path
) -> Mapping[str, Path]:
    root = external_root.resolve(strict=True)
    entries = receipt.get("external_files")
    if not isinstance(entries, list):
        raise ValueError("receipt external_files must be a list")
    by_name: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("receipt external_files rows must be objects")
        raw_path = _require_str(entry.get("path"), "external_files.path")
        name = Path(raw_path).name
        if name in REQUIRED_CSV:
            if name in by_name:
                raise ValueError(f"duplicate receipt entry for {name}")
            by_name[name] = entry

    sources: dict[str, Path] = {}
    for name in REQUIRED_CSV:
        if name not in by_name:
            raise ValueError(f"receipt is missing required external file {name}")
        source = Path(_require_str(by_name[name].get("path"), f"{name}.path")).resolve(
            strict=True
        )
        if source.parent != root or source.name != name:
            raise ValueError(f"external file is outside the exact external root: {source}")
        expected_bytes = by_name[name].get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"invalid receipt byte count for {name}")
        if source.stat().st_size != expected_bytes:
            raise ValueError(f"external file byte count does not match receipt: {name}")
        sources[name] = source
    return MappingProxyType(sources)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE assets (
            asset_id TEXT PRIMARY KEY, scan_id TEXT NOT NULL, location TEXT NOT NULL,
            root_id TEXT NOT NULL, relative_path TEXT NOT NULL, display_name TEXT NOT NULL,
            escaped_name TEXT NOT NULL, asset_kind TEXT NOT NULL, size_bytes INTEGER,
            mtime_utc TEXT, access_status TEXT NOT NULL, hash_status TEXT NOT NULL,
            sha256 TEXT, experiment_id TEXT, git_ownership TEXT, evidence_role TEXT,
            retention_class TEXT, recommended_action TEXT, decision_reason TEXT
        );
        CREATE TABLE experiments (
            experiment_id TEXT PRIMARY KEY, run_id TEXT, experiment_state TEXT NOT NULL,
            phase TEXT, method_or_candidate TEXT, report_path TEXT,
            local_artifact_paths TEXT, n607_artifact_paths TEXT, git_commit TEXT,
            process_evidence TEXT, prediction_count INTEGER, score_count INTEGER,
            expected_artifacts TEXT, observed_artifacts TEXT, closure_gaps TEXT
        );
        CREATE TABLE git_ownership (
            asset_id TEXT NOT NULL, ownership TEXT NOT NULL, repository_root TEXT,
            common_git_dir TEXT, branch TEXT, head_commit TEXT, status_summary TEXT,
            linked_worktrees TEXT, error TEXT
        );
        CREATE TABLE retention (
            asset_id TEXT NOT NULL, retention_class TEXT NOT NULL, rule_code TEXT,
            reason TEXT, evidence_asset_ids TEXT, recommended_action TEXT
        );
        CREATE TABLE deletion_candidates (
            candidate_id TEXT, location TEXT, absolute_path TEXT, asset_kind TEXT,
            size_bytes INTEGER, reason TEXT, evidence TEXT, dependencies TEXT,
            recoverability TEXT, estimated_space_reclaim INTEGER, approval_state TEXT,
            approved_scope TEXT, execution_state TEXT
        );
        """
    )


def _integer_or_none(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"expected integer CSV value, got {value!r}") from exc


def _convert_row(table: str, row: Mapping[str, str]) -> tuple[object, ...]:
    headers: Sequence[str]
    integer_fields: frozenset[str]
    if table == "assets":
        headers, integer_fields = ASSET_HEADERS, frozenset({"size_bytes"})
    elif table == "experiments":
        headers = EXPERIMENT_HEADERS
        integer_fields = frozenset({"prediction_count", "score_count"})
    elif table == "git_ownership":
        headers, integer_fields = GIT_HEADERS, frozenset()
    elif table == "retention":
        headers, integer_fields = RETENTION_HEADERS, frozenset()
    elif table == "deletion_candidates":
        headers = DELETION_HEADERS
        integer_fields = frozenset({"size_bytes", "estimated_space_reclaim"})
    else:
        raise ValueError(f"unsupported index table {table}")
    return tuple(
        _integer_or_none(row[name]) if name in integer_fields else row[name]
        for name in headers
    )


def _import_csv(
    connection: sqlite3.Connection,
    *,
    table: str,
    source: Path,
    headers: Sequence[str],
    expected_scan_id: str | None = None,
) -> int:
    csv.field_size_limit(min(sys.maxsize, 1024 * 1024 * 1024))
    placeholders = ",".join("?" for _ in headers)
    columns = ",".join(headers)
    statement = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    count = 0
    batch: list[tuple[object, ...]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(headers):
            raise ValueError(f"unexpected CSV header for {source.name}")
        for row in reader:
            if None in row:
                raise ValueError(f"unexpected extra CSV fields in {source.name}")
            if expected_scan_id is not None and row.get("scan_id") != expected_scan_id:
                raise ValueError(f"asset scan_id mismatch in {source.name}")
            batch.append(_convert_row(table, row))
            count += 1
            if len(batch) >= 10_000:
                connection.executemany(statement, batch)
                batch.clear()
        if batch:
            connection.executemany(statement, batch)
    return count


def _receipt_count(receipt: Mapping[str, object], key: str) -> int:
    counts = receipt.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("receipt counts must be an object")
    value = counts.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid receipt count: {key}")
    return value


def _validate_receipt_for_index(receipt: Mapping[str, object]) -> str:
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported governance receipt schema")
    if receipt.get("terminal_state") != "COMPLETE":
        raise ValueError("only a COMPLETE governance receipt can be indexed")
    for field in ("source_asset_mutations", "moves", "overwrites", "deletions"):
        if receipt.get(field) != 0:
            raise ValueError(f"receipt {field} must be zero")
    return _require_str(receipt.get("scan_id"), "scan_id")


def _write_metadata(
    connection: sqlite3.Connection,
    *,
    receipt: Mapping[str, object],
    receipt_path: Path,
    external_root: Path,
    counts: Mapping[str, int],
) -> None:
    roots = receipt.get("roots") if isinstance(receipt.get("roots"), dict) else {}
    implementation = (
        receipt.get("implementation")
        if isinstance(receipt.get("implementation"), dict)
        else {}
    )
    values = {
        "schema_version": str(INDEX_SCHEMA_VERSION),
        "scan_id": str(receipt["scan_id"]),
        "receipt_path": str(receipt_path.resolve()),
        "external_root": str(external_root.resolve()),
        "created_at_utc": str(receipt.get("completed_at_utc", "")),
        "implementation_git_head": str(implementation.get("git_head", "")),
        "local_root": str(roots.get("local", "")),
        "n607_root": str(roots.get("n607", "")),
        "table_counts": json.dumps(dict(counts), sort_keys=True, separators=(",", ":")),
        "scan_error_counts": json.dumps(
            receipt.get("scan_error_counts", {}), sort_keys=True, separators=(",", ":")
        ),
    }
    connection.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)", values.items()
    )


def _create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX assets_location_path ON assets(location, root_id, relative_path);
        CREATE INDEX assets_experiment ON assets(experiment_id);
        CREATE INDEX assets_retention ON assets(retention_class);
        CREATE INDEX experiments_run_id ON experiments(run_id);
        CREATE INDEX experiments_state ON experiments(experiment_state);
        CREATE INDEX ownership_asset ON git_ownership(asset_id);
        CREATE INDEX ownership_repository ON git_ownership(repository_root);
        CREATE INDEX retention_asset ON retention(asset_id);
        CREATE INDEX retention_class ON retention(retention_class);
        """
    )


def build_index(
    *, receipt_path: Path, external_root: Path, database_path: Path
) -> IndexBuildSummary:
    """Stream one terminal governance scan into a new external SQLite index."""

    receipt_path = Path(receipt_path)
    external_root = Path(external_root)
    database_path = Path(database_path)
    receipt = _load_terminal_receipt(receipt_path)
    scan_id = _validate_receipt_for_index(receipt)
    sources = _resolve_csv_sources(receipt, external_root)
    if database_path.exists():
        raise FileExistsError(f"governance index target already exists: {database_path}")
    if database_path.parent.resolve(strict=True) != external_root.resolve(strict=True):
        raise ValueError("governance index must be created directly in external_root")
    temporary = database_path.with_name(f".{database_path.name}.building")
    if temporary.exists():
        raise FileExistsError(f"governance index temporary target exists: {temporary}")

    connection = sqlite3.connect(temporary)
    published = False
    try:
        _create_schema(connection)
        local_count = _import_csv(
            connection,
            table="assets",
            source=sources["asset_inventory_local.csv"],
            headers=ASSET_HEADERS,
            expected_scan_id=scan_id,
        )
        n607_count = _import_csv(
            connection,
            table="assets",
            source=sources["asset_inventory_n607.csv"],
            headers=ASSET_HEADERS,
            expected_scan_id=scan_id,
        )
        counts = {
            "assets": local_count + n607_count,
            "experiments": _import_csv(
                connection,
                table="experiments",
                source=sources["experiment_index.csv"],
                headers=EXPERIMENT_HEADERS,
            ),
            "git_ownership": _import_csv(
                connection,
                table="git_ownership",
                source=sources["git_ownership.csv"],
                headers=GIT_HEADERS,
            ),
            "retention": _import_csv(
                connection,
                table="retention",
                source=sources["retention_decisions.csv"],
                headers=RETENTION_HEADERS,
            ),
            "deletion_candidates": _import_csv(
                connection,
                table="deletion_candidates",
                source=sources["deletion_candidates.csv"],
                headers=DELETION_HEADERS,
            ),
        }
        expected = {
            "assets": _receipt_count(receipt, "assets"),
            "experiments": _receipt_count(receipt, "experiments"),
            "git_ownership": _receipt_count(receipt, "git_ownership"),
            "retention": _receipt_count(receipt, "retention_decisions"),
            "deletion_candidates": _receipt_count(receipt, "deletion_candidates"),
        }
        if counts != expected:
            raise ValueError(f"imported row counts do not match receipt: {counts} != {expected}")
        _write_metadata(
            connection,
            receipt=receipt,
            receipt_path=receipt_path,
            external_root=external_root,
            counts=counts,
        )
        _create_indexes(connection)
        connection.commit()
        connection.close()
        temporary.replace(database_path)
        published = True
    finally:
        try:
            connection.close()
        finally:
            if not published and temporary.exists():
                temporary.unlink()

    return IndexBuildSummary(
        scan_id=scan_id,
        database_path=database_path.resolve(),
        table_counts=MappingProxyType(counts),
    )
