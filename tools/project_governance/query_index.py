"""Regenerable, read-only query index for completed governance scans."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


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


@dataclass(frozen=True)
class LatestPointer:
    """Validated pointer to one immutable governance query baseline."""

    schema_version: int
    scan_id: str
    receipt_path: Path
    external_root: Path
    sqlite_path: Path
    created_at_utc: str
    implementation_git_head: str


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
    with database_path.open("xb"):
        pass
    connection = sqlite3.connect(database_path)
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
    finally:
        connection.close()

    return IndexBuildSummary(
        scan_id=scan_id,
        database_path=database_path.resolve(),
        table_counts=MappingProxyType(counts),
    )


def load_latest(pointer_path: Path) -> LatestPointer:
    """Load a strict, local latest pointer without opening the database."""

    pointer_path = Path(pointer_path)
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable governance latest pointer: {pointer_path}") from exc
    expected_keys = {
        "schema_version",
        "scan_id",
        "receipt_path",
        "external_root",
        "sqlite_path",
        "created_at_utc",
        "implementation_git_head",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("latest pointer must contain the exact schema keys")
    if payload["schema_version"] != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported latest pointer schema")
    scan_id = _require_str(payload["scan_id"], "latest.scan_id")
    receipt_path = Path(_require_str(payload["receipt_path"], "latest.receipt_path"))
    external_root = Path(_require_str(payload["external_root"], "latest.external_root"))
    sqlite_path = Path(_require_str(payload["sqlite_path"], "latest.sqlite_path"))
    if not receipt_path.is_absolute() or not external_root.is_absolute() or not sqlite_path.is_absolute():
        raise ValueError("latest pointer paths must be absolute")
    receipt_path = receipt_path.resolve(strict=True)
    external_root = external_root.resolve(strict=True)
    sqlite_path = sqlite_path.resolve(strict=True)
    if receipt_path.name != "scan_receipt.json" or receipt_path.parent.name != scan_id:
        raise ValueError("latest receipt path does not match scan_id")
    if external_root.name != scan_id or sqlite_path.parent != external_root:
        raise ValueError("latest external paths do not match scan_id")
    if sqlite_path.name != "governance.sqlite":
        raise ValueError("latest sqlite_path must name governance.sqlite")
    created_at = _require_str(payload["created_at_utc"], "latest.created_at_utc")
    git_head = _require_str(
        payload["implementation_git_head"], "latest.implementation_git_head"
    )
    if len(git_head) != 40 or any(character not in "0123456789abcdef" for character in git_head):
        raise ValueError("latest implementation_git_head must be 40 lowercase hex characters")
    return LatestPointer(
        schema_version=INDEX_SCHEMA_VERSION,
        scan_id=scan_id,
        receipt_path=receipt_path,
        external_root=external_root,
        sqlite_path=sqlite_path,
        created_at_utc=created_at,
        implementation_git_head=git_head,
    )


def _rows_as_dicts(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    names = tuple(description[0] for description in cursor.description or ())
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _require_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return limit


def _like_prefix(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


class QueryStore:
    """Bounded, read-only queries over one validated governance index."""

    def __init__(
        self,
        *,
        pointer: LatestPointer,
        receipt: Mapping[str, object],
        connection: sqlite3.Connection,
        metadata: Mapping[str, str],
    ) -> None:
        self.pointer = pointer
        self.receipt = receipt
        self._connection = connection
        self._metadata = metadata

    @classmethod
    def open(cls, pointer: LatestPointer) -> "QueryStore":
        receipt = _load_terminal_receipt(pointer.receipt_path)
        if _validate_receipt_for_index(receipt) != pointer.scan_id:
            raise ValueError("latest scan_id does not match receipt")
        uri = f"{pointer.sqlite_path.as_uri()}?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True)
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        except sqlite3.Error as exc:
            raise ValueError(f"unreadable governance SQLite index: {pointer.sqlite_path}") from exc
        metadata = {str(key): str(value) for key, value in rows}
        if metadata.get("schema_version") != str(INDEX_SCHEMA_VERSION):
            connection.close()
            raise ValueError("SQLite index schema_version does not match latest pointer")
        if metadata.get("scan_id") != pointer.scan_id:
            connection.close()
            raise ValueError("latest scan_id does not match SQLite metadata")
        if Path(metadata.get("external_root", "")).resolve() != pointer.external_root:
            connection.close()
            raise ValueError("latest external_root does not match SQLite metadata")
        if Path(metadata.get("receipt_path", "")).resolve() != pointer.receipt_path:
            connection.close()
            raise ValueError("latest receipt_path does not match SQLite metadata")
        return cls(
            pointer=pointer,
            receipt=receipt,
            connection=connection,
            metadata=MappingProxyType(metadata),
        )

    def close(self) -> None:
        self._connection.close()

    def status(self) -> dict[str, object]:
        table_counts = json.loads(self._metadata["table_counts"])
        scan_errors = json.loads(self._metadata.get("scan_error_counts", "{}"))
        warning_count = sum(
            value for value in scan_errors.values() if isinstance(value, int) and value > 0
        )
        return {
            "created_at_utc": self.pointer.created_at_utc,
            "database_path": str(self.pointer.sqlite_path),
            "receipt_path": str(self.pointer.receipt_path),
            "scan_error_counts": scan_errors,
            "scan_id": self.pointer.scan_id,
            "table_counts": table_counts,
            "terminal_state": self.receipt["terminal_state"],
            "warning_count": warning_count,
        }

    def _path_identity(self, query: str) -> tuple[str, str] | None:
        local_root = self._metadata.get("local_root", "").replace("\\", "/").rstrip("/")
        n607_root = self._metadata.get("n607_root", "").rstrip("/")
        local_query = query.replace("\\", "/")
        if local_root and (
            local_query.casefold() == local_root.casefold()
            or local_query.casefold().startswith(local_root.casefold() + "/")
        ):
            return "LOCAL", local_query[len(local_root) :].lstrip("/")
        if n607_root and (query == n607_root or query.startswith(n607_root + "/")):
            return "N607", query[len(n607_root) :].lstrip("/")
        return None

    def find_assets(self, query: str, *, limit: int = 20) -> dict[str, object]:
        limit = _require_limit(limit)
        query = _require_str(query, "query")
        columns = (
            "asset_id, scan_id, location, root_id, relative_path, display_name, "
            "asset_kind, size_bytes, access_status, experiment_id, git_ownership, "
            "evidence_role, retention_class, recommended_action, decision_reason"
        )
        cursor = self._connection.execute(
            f"SELECT {columns} FROM assets WHERE asset_id = ? LIMIT ?", (query, limit + 1)
        )
        items = _rows_as_dicts(cursor)
        identity = self._path_identity(query)
        if not items and identity is not None:
            location, relative_path = identity
            cursor = self._connection.execute(
                f"SELECT {columns} FROM assets "
                "WHERE location = ? AND (relative_path = ? OR relative_path LIKE ? ESCAPE '!') "
                "ORDER BY relative_path, asset_id LIMIT ?",
                (location, relative_path, _like_prefix(relative_path.rstrip("/") + "/"), limit + 1),
            )
            items = _rows_as_dicts(cursor)
        truncated = len(items) > limit
        return {
            "count": min(len(items), limit),
            "items": items[:limit],
            "query": query,
            "truncated": truncated,
        }

    def experiment(self, run_id: str) -> dict[str, object]:
        run_id = _require_str(run_id, "run_id")
        rows = _rows_as_dicts(
            self._connection.execute(
                "SELECT * FROM experiments WHERE run_id = ? ORDER BY experiment_id LIMIT 2",
                (run_id,),
            )
        )
        if not rows:
            return {"run_id": run_id, "status": "NOT_FOUND"}
        if len(rows) > 1:
            return {"count": len(rows), "run_id": run_id, "status": "AMBIGUOUS"}
        result = rows[0]
        counts = {
            str(location): int(count)
            for location, count in self._connection.execute(
                "SELECT location, COUNT(*) FROM assets WHERE experiment_id = ? GROUP BY location",
                (result["experiment_id"],),
            )
        }
        result["assets_by_location"] = counts
        result["status"] = "FOUND"
        return result

    def repo(self, path: str) -> dict[str, object]:
        path = _require_str(path, "path")
        identity = self._path_identity(path)
        if identity is not None:
            location, relative_path = identity
            asset_rows = self._connection.execute(
                "SELECT asset_id FROM assets WHERE location = ? AND relative_path = ? "
                "ORDER BY asset_id LIMIT 2",
                (location, relative_path),
            ).fetchall()
        else:
            asset_rows = self._connection.execute(
                "SELECT asset_id FROM assets WHERE asset_id = ? ORDER BY asset_id LIMIT 2",
                (path,),
            ).fetchall()
        if not asset_rows:
            return {"items": [], "path": path, "status": "NOT_FOUND"}
        if len(asset_rows) != 1:
            return {"items": [], "path": path, "status": "AMBIGUOUS"}
        asset_id = asset_rows[0][0]
        raw_items = _rows_as_dicts(
            self._connection.execute(
                "SELECT DISTINCT ownership, repository_root, common_git_dir, branch, "
                "head_commit, status_summary, linked_worktrees, error "
                "FROM git_ownership WHERE asset_id = ? "
                "ORDER BY repository_root, common_git_dir, branch, head_commit",
                (asset_id,),
            )
        )
        items: list[dict[str, object]] = []
        for row in raw_items:
            try:
                linked = json.loads(str(row.pop("linked_worktrees") or "[]"))
            except json.JSONDecodeError:
                linked = []
            status_summary = str(row.pop("status_summary") or "")
            row["linked_worktree_count"] = len(linked) if isinstance(linked, list) else 0
            row["status_state"] = (
                "CLEAN" if status_summary == "CLEAN" else "DIRTY_OR_UNKNOWN"
            )
            items.append(row)
        if not items:
            status = "NOT_FOUND"
        elif len(items) == 1:
            status = "FOUND"
        else:
            status = "AMBIGUOUS"
        return {"asset_id": asset_id, "items": items, "path": path, "status": status}

    def review(
        self, filters: Mapping[str, str] | None = None, *, limit: int = 20
    ) -> dict[str, object]:
        limit = _require_limit(limit)
        filters = dict(filters or {})
        allowed = {
            "location": "a.location",
            "retention_class": "r.retention_class",
            "experiment_state": "e.experiment_state",
            "ownership": "g.ownership",
        }
        if set(filters) - set(allowed):
            raise ValueError("unsupported review filter")
        clauses: list[str] = []
        parameters: list[object] = []
        for key in ("location", "retention_class", "experiment_state", "ownership"):
            if key in filters:
                clauses.append(f"{allowed[key]} = ?")
                parameters.append(_require_str(filters[key], key))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = (
            "SELECT DISTINCT a.asset_id, a.location, a.relative_path, a.access_status, "
            "a.experiment_id, r.retention_class, r.rule_code, r.reason, "
            "r.recommended_action, e.experiment_state, g.ownership "
            "FROM assets a "
            "LEFT JOIN retention r ON r.asset_id = a.asset_id "
            "LEFT JOIN experiments e ON e.experiment_id = a.experiment_id "
            "LEFT JOIN git_ownership g ON g.asset_id = a.asset_id"
            f"{where} ORDER BY a.location, a.relative_path, a.asset_id LIMIT ?"
        )
        parameters.append(limit + 1)
        items = _rows_as_dicts(self._connection.execute(query, parameters))
        authorized = self._connection.execute(
            "SELECT COUNT(*) FROM deletion_candidates "
            "WHERE approval_state = 'APPROVED' OR execution_state = 'AUTHORIZED'"
        ).fetchone()[0]
        truncated = len(items) > limit
        return {
            "authorized_deletion_count": int(authorized),
            "count": min(len(items), limit),
            "filters": filters,
            "items": items[:limit],
            "truncated": truncated,
        }
