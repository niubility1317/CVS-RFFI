"""Deterministic, read-only governance report emission.

The emitter consumes a :class:`~tools.project_governance.models.ScanBundle`
that has already been collected and classified.  It serializes facts only:
it never reads the governed roots, contacts N607, or exposes an execution
operation for deletion, movement, or overwrite.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    AccessStatus,
    ApprovalState,
    AssetRecord,
    DeletionCandidate,
    ExecutionState,
    ExperimentRecord,
    ExperimentState,
    GitOwnership,
    GitOwnershipRecord,
    Location,
    RetentionDecision,
    ScanBundle,
    ScopeResult,
)


DEFAULT_GIT_FILE_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_GIT_SCAN_MAX_BYTES = 50 * 1024 * 1024

_ARTIFACT_ORDER = (
    "report.md",
    "asset_inventory_local.csv",
    "asset_inventory_n607.csv",
    "experiment_index.csv",
    "git_ownership.csv",
    "retention_decisions.csv",
    "deletion_candidates.csv",
    "asset_inventory_full.json",
)
_RECEIPT_NAME = "scan_receipt.json"
_REQUESTED_N607_ROUTES = frozenset({"DIRECT", "LAB_BRIDGE", "NO_ROUTE"})
_OBSERVED_N607_STATES = frozenset({"VERIFIED", "FAILED", "UNKNOWN"})
_REQUIRED_METADATA = frozenset(
    {
        "local_root",
        "local_scopes",
        "n607_root",
        "n607_scopes",
        "implementation_git_head",
        "git_tracked_diff_state",
        "collector_versions",
        "n607_requested",
        "n607_route",
        "n607_preflight",
        "n607_disconnect",
        "n607_scan_error_count",
    }
)


@dataclass(frozen=True)
class EmissionResult:
    """Paths and receipt facts returned after a successful emission."""

    git_output_dir: Path
    external_output_dir: Path | None
    receipt: Mapping[str, Any]


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _json_value(value: Any) -> Any:
    """Convert immutable records into JSON-safe, deterministic values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(value[key])
            for key in sorted(value, key=lambda item: (str(item).casefold(), str(item)))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        ordered = sorted(
            value,
            key=lambda item: json.dumps(
                _json_value(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
        return [_json_value(item) for item in ordered]
    if isinstance(value, Path):
        return str(value)
    return value


def _cell(value: Any) -> str:
    """Render one CSV cell without losing structured evidence."""

    converted = _json_value(value)
    if converted is None:
        return ""
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(converted)


def _markdown_cell(value: Any) -> str:
    """Render one table cell without allowing evidence text to break Markdown rows."""

    return _cell(value).replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>").replace("|", "\\|")


def _record_dict(record: Any) -> dict[str, Any]:
    if not is_dataclass(record):
        raise TypeError(f"expected dataclass record, got {type(record)!r}")
    return {field.name: _json_value(getattr(record, field.name)) for field in fields(record)}


def _record_fields(record_type: type[Any]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(record_type))


def _sort_records(records: Iterable[Any], *, fields_for_key: Sequence[str]) -> tuple[Any, ...]:
    def key(record: Any) -> tuple[str, ...]:
        primary = tuple(_cell(getattr(record, field, None)).casefold() for field in fields_for_key)
        tie_breaker = json.dumps(
            _record_dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return primary + (tie_breaker,)

    return tuple(sorted(records, key=key))


def _csv_bytes(records: Iterable[Any], record_type: type[Any], *, sort_fields: Sequence[str]) -> bytes:
    values = _sort_records(records, fields_for_key=sort_fields)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    headers = _record_fields(record_type)
    writer.writerow(headers)
    for record in values:
        writer.writerow([_cell(getattr(record, header)) for header in headers])
    return output.getvalue().encode("utf-8-sig")


def _iso_utc(value: str | None) -> str:
    if value is None:
        current = datetime.now(timezone.utc)
        current -= timedelta(microseconds=current.microsecond)
        return current.isoformat()[:-6] + "Z"
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("timestamp must be an ISO 8601 value")
    try:
        iso_value = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO 8601 value") from exc
    if parsed.tzinfo is None:
        parsed = datetime(
            parsed.year,
            parsed.month,
            parsed.day,
            parsed.hour,
            parsed.minute,
            parsed.second,
            parsed.microsecond,
            tzinfo=timezone.utc,
        )
    parsed = parsed.astimezone(timezone.utc)
    parsed -= timedelta(microseconds=parsed.microsecond)
    return parsed.isoformat()[:-6] + "Z"


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _file_entry(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    payload = path.read_bytes()
    display_path = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path.resolve())
    return {
        "path": display_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _location_value(location: Location | str) -> str:
    return location.value if isinstance(location, Location) else str(location)


def _validate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("complete receipt metadata is required")
    missing = sorted(_REQUIRED_METADATA.difference(metadata))
    if missing:
        raise ValueError(f"receipt metadata is missing required fields: {', '.join(missing)}")
    values = dict(metadata)
    for field in (
        "local_root",
        "n607_root",
        "implementation_git_head",
        "git_tracked_diff_state",
        "n607_route",
        "n607_preflight",
        "n607_disconnect",
    ):
        value = values[field]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(
                f"receipt metadata field must be a trimmed non-empty string: {field}"
            )
    for field in ("local_scopes", "n607_scopes"):
        scopes = values[field]
        if (
            not isinstance(scopes, (tuple, list))
            or any(not isinstance(scope, str) or scope != scope.strip() for scope in scopes)
        ):
            raise ValueError(f"receipt metadata field must be an explicit string sequence: {field}")
    versions = values["collector_versions"]
    if not isinstance(versions, Mapping) or not versions:
        raise ValueError("receipt metadata collector_versions must be a non-empty mapping")
    if type(values["n607_requested"]) is not bool:
        raise ValueError("receipt metadata n607_requested must be boolean")
    error_count = values["n607_scan_error_count"]
    if type(error_count) is not int or error_count < 0:
        raise ValueError("n607_scan_error_count must be a non-negative integer")
    n607_states = (
        values["n607_route"],
        values["n607_preflight"],
        values["n607_disconnect"],
    )
    if values["n607_requested"]:
        route, preflight, disconnect = n607_states
        if route not in _REQUESTED_N607_ROUTES:
            raise ValueError("requested N607 receipt metadata has an uncontrolled route state")
        if preflight not in _OBSERVED_N607_STATES or disconnect not in _OBSERVED_N607_STATES:
            raise ValueError("requested N607 receipt metadata must carry controlled observed states")
    elif n607_states != ("NOT_REQUESTED", "NOT_REQUESTED", "NOT_REQUESTED") or error_count != 0:
        raise ValueError("unrequested N607 receipt metadata must use explicit NOT_REQUESTED states")
    return values


def _validate_deletion_candidates(candidates: Iterable[DeletionCandidate]) -> None:
    for candidate in candidates:
        if not isinstance(candidate, DeletionCandidate):
            raise ValueError("deletion candidate rows must use DeletionCandidate records")
        if (
            candidate.approval_state is not ApprovalState.AWAITING_USER_APPROVAL
            or candidate.execution_state is not ExecutionState.NOT_AUTHORIZED
            or candidate.approved_scope is not None
        ):
            raise ValueError(
                f"deletion candidate is not approval-only and unexecuted: {candidate.candidate_id}"
            )


def _payload_entry(name: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class ReportEmitter:
    """Write a fresh, deterministic governance report directory.

    ``metadata`` is an already collected fact mapping.  It is deliberately
    opaque to the emitter so this component cannot perform local or remote
    discovery as a side effect.
    """

    def __init__(
        self,
        bundle: ScanBundle,
        output_root: str | os.PathLike[str],
        external_output_root: str | os.PathLike[str],
        metadata: Mapping[str, Any] | None = None,
        *,
        git_file_max_bytes: int = DEFAULT_GIT_FILE_MAX_BYTES,
        git_scan_max_bytes: int = DEFAULT_GIT_SCAN_MAX_BYTES,
    ) -> None:
        if not isinstance(bundle, ScanBundle):
            raise TypeError("bundle must be a ScanBundle")
        if (
            not bundle.scan_id
            or bundle.scan_id in {".", ".."}
            or "/" in bundle.scan_id
            or "\\" in bundle.scan_id
        ):
            raise ValueError("scan_id must be a single path component")
        if git_file_max_bytes <= 0 or git_scan_max_bytes <= 0:
            raise ValueError("output thresholds must be positive")
        self.bundle = bundle
        self.output_root = Path(output_root).expanduser().resolve(strict=False)
        self.external_output_root = Path(external_output_root).expanduser().resolve(strict=False)
        self.metadata = _validate_metadata(metadata)
        _validate_deletion_candidates(bundle.deletion_candidates or ())
        for timestamp in (bundle.started_at_utc, bundle.completed_at_utc):
            if timestamp is not None:
                _iso_utc(timestamp)
        self.git_file_max_bytes = int(git_file_max_bytes)
        self.git_scan_max_bytes = int(git_scan_max_bytes)
        self._emission_now_utc: str | None = None

    @property
    def scan_id(self) -> str:
        return self.bundle.scan_id

    def _timestamp(self, value: str | None) -> str:
        if value:
            return _iso_utc(value)
        if self._emission_now_utc is None:
            self._emission_now_utc = _iso_utc(None)
        return self._emission_now_utc

    def _target_paths(self) -> tuple[Path, Path]:
        git_target = self.output_root / self.scan_id
        external_target = self.external_output_root / self.scan_id
        if git_target.exists():
            raise FileExistsError(f"governance output already exists: {git_target}")
        if external_target.exists():
            raise FileExistsError(f"external governance output already exists: {external_target}")
        return git_target, external_target

    def _write_exclusive(
        self,
        path: Path,
        payload: bytes | str,
        *,
        encoding: str = "utf-8",
        newline: str = "",
    ) -> None:
        """Write one artifact in exclusive mode; never remove partial output."""

        if isinstance(payload, bytes):
            with path.open("xb") as stream:
                stream.write(payload)
            return
        with path.open("x", encoding=encoding, newline=newline) as stream:
            stream.write(payload)

    def _records(self) -> dict[str, tuple[Any, ...]]:
        bundle = self.bundle
        return {
            "assets": tuple(bundle.assets or ()),
            "scope_results": tuple(bundle.scope_results or ()),
            "git_ownership": tuple(bundle.git_ownership or ()),
            "experiments": tuple(bundle.experiments or ()),
            "retention_decisions": tuple(bundle.retention_decisions or ()),
            "deletion_candidates": tuple(bundle.deletion_candidates or ()),
        }

    def _full_inventory(self, records: Mapping[str, tuple[Any, ...]]) -> dict[str, Any]:
        return {
            "schema_version": self.bundle.schema_version,
            "scan_id": self.scan_id,
            "operator": self.bundle.operator,
            "started_at_utc": self._timestamp(self.bundle.started_at_utc),
            "completed_at_utc": self._timestamp(self.bundle.completed_at_utc),
            "assets": [_record_dict(record) for record in _sort_records(records["assets"], fields_for_key=("location", "root_id", "relative_path", "asset_id"))],
            "scope_results": [_record_dict(record) for record in _sort_records(records["scope_results"], fields_for_key=("location", "root_id", "relative_path", "status"))],
            "git_ownership": [_record_dict(record) for record in _sort_records(records["git_ownership"], fields_for_key=("asset_id", "ownership"))],
            "experiments": [_record_dict(record) for record in _sort_records(records["experiments"], fields_for_key=("experiment_id",))],
            "retention_decisions": [_record_dict(record) for record in _sort_records(records["retention_decisions"], fields_for_key=("asset_id", "retention_class"))],
            "deletion_candidates": [_record_dict(record) for record in _sort_records(records["deletion_candidates"], fields_for_key=("candidate_id", "location", "absolute_path"))],
        }

    def _report(self, records: Mapping[str, tuple[Any, ...]]) -> str:
        assets = records["assets"]
        local_assets = tuple(asset for asset in assets if asset.location is Location.LOCAL)
        remote_assets = tuple(asset for asset in assets if asset.location is Location.N607)
        experiments = records["experiments"]
        states = Counter(_cell(getattr(item, "experiment_state", None)) for item in experiments)
        ownership = Counter(_cell(getattr(item, "ownership", None)) for item in records["git_ownership"])
        retention = Counter(_cell(getattr(item, "retention_class", None)) for item in records["retention_decisions"])
        gaps = [
            f"{_location_value(scope.location)}:{scope.relative_path}:{scope.status}"
            for scope in records["scope_results"]
            if str(scope.status).upper() != "VERIFIED"
        ]
        experiment_gaps = sorted(
            f"{experiment.experiment_id}:{gap}"
            for experiment in experiments
            for gap in (experiment.closure_gaps or ())
        )
        metadata = self.metadata
        n607_route = metadata["n607_route"]
        n607_preflight = metadata["n607_preflight"]
        n607_disconnect = metadata["n607_disconnect"]
        deletion_rows = records["deletion_candidates"]
        missing = "\u672a\u63d0\u4f9b"
        # Keep source text ASCII-safe while emitting the required Chinese report.
        zh = {
            "title": "\u9879\u76ee\u8d44\u4ea7\u6cbb\u7406\u626b\u63cf\u62a5\u544a",
            "evidence_first": "\u672c\u62a5\u544a\u9075\u5faa\u8bc1\u636e\u4f18\u5148\uff0c\u53ea\u6574\u7406\u5df2\u91c7\u96c6\u7684\u5143\u6570\u636e\u548c\u8bc1\u636e\u5f15\u7528\uff0c\u4e0d\u4f5c\u6027\u80fd\u7ed3\u8bba\u6216\u65b9\u6cd5\u9009\u62e9\u5224\u65ad\u3002",
            "scope": "\u626b\u63cf\u8303\u56f4\u4e0e\u65b0\u9c9c\u5ea6",
            "operator": "\u64cd\u4f5c\u8005",
            "start": "\u5f00\u59cb\u65f6\u95f4",
            "complete": "\u5b8c\u6210\u65f6\u95f4",
            "local_root": "\u672c\u5730\u6839",
            "remote_root": "N607\u6839",
            "surface": "\u627f\u8f7d\u9762",
            "asset": "\u8d44\u4ea7\u603b\u8868",
            "records": "\u8bb0\u5f55\u6570",
            "error_count": "SCAN_ERROR\u6570",
            "total": "\u5408\u8ba1",
            "experiment": "\u5b9e\u9a8c\u7d22\u5f15",
            "state": "\u72b6\u6001",
            "git": "Git\u5f52\u5c5e",
            "retention": "\u4fdd\u7559\u5206\u5e03",
            "retention_level": "\u4fdd\u7559\u7ea7\u522b",
            "deletion": "\u5f85\u5ba1\u6279\u5220\u9664\u6e05\u5355",
            "position": "\u4f4d\u7f6e",
            "path": "\u8def\u5f84",
            "kind": "\u8d44\u4ea7\u7c7b\u578b",
            "size": "\u5927\u5c0f",
            "reason": "\u539f\u56e0",
            "none": "\u65e0",
            "no_candidate": "\u65e0\u5019\u9009",
            "no_rows": "\u6ca1\u6709\u5f85\u5ba1\u6279\u6761\u76ee",
            "gaps": "\u8986\u76d6\u7f3a\u53e3\u4e0e\u9519\u8bef",
            "nonverified": "\u975eVERIFIED\u627f\u8f7d\u9762",
            "experiment_gaps": "\u5b9e\u9a8c\u95ed\u5408\u7f3a\u53e3",
            "asset_error": "\u8d44\u4ea7SCAN_ERROR",
            "experiment_error": "\u5b9e\u9a8cSCAN_ERROR",
            "n607_record_error": "N607\u534f\u8baeSCAN_ERROR",
            "remote": "N607\u8fde\u63a5\u7ed3\u679c",
            "route": "\u8def\u7531",
            "preflight": "\u9884\u68c0",
            "disconnect": "\u65ad\u8fde",
            "boundary": "\u53d8\u66f4\u8fb9\u754c",
            "zero": "\u5b9e\u9645\u79fb\u52a8\u3001\u8986\u76d6\u3001\u5220\u9664\u6570\u91cf\u4e3a0\u3002\u6240\u6709\u5f85\u5ba1\u6279\u6761\u76ee\u4fdd\u6301\u539f\u4f4d\uff0c\u6267\u884c\u72b6\u6001\u4e3a`NOT_AUTHORIZED`\uff1b\u672c\u62a5\u544a\u4e0d\u63d0\u4f9b\u6267\u884c\u63a5\u53e3\u3002",
        }
        lines = [
            f"# {zh['title']}",
            "",
            zh["evidence_first"],
            "",
            f"## {zh['scope']}",
            "",
            f"- scan_id\uff1a`{self.scan_id}`",
            f"- {zh['operator']}\uff1a`{self.bundle.operator or missing}`",
            f"- {zh['start']}\uff08UTC\uff09\uff1a`{self._timestamp(self.bundle.started_at_utc)}`",
            f"- {zh['complete']}\uff08UTC\uff09\uff1a`{self._timestamp(self.bundle.completed_at_utc)}`",
            f"- {zh['local_root']}\uff1a`{metadata['local_root']}`\uff1b{zh['surface']}\uff1a`{', '.join(map(str, metadata['local_scopes'])) or missing}`",
            f"- {zh['remote_root']}\uff1a`{metadata['n607_root']}`\uff1b{zh['surface']}\uff1a`{', '.join(map(str, metadata['n607_scopes'])) or missing}`",
            "",
            f"## {zh['asset']}",
            "",
            f"|{zh['position']}|{zh['records']}|{zh['error_count']}|",
            "|---|---:|---:|",
            f"|LOCAL|{len(local_assets)}|{sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in local_assets)}|",
            f"|N607|{len(remote_assets)}|{sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in remote_assets)}|",
            f"|{zh['total']}|{len(assets)}|{sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in assets)}|",
            "",
            f"## {zh['experiment']}",
            "",
            f"|{zh['state']}|{zh['records']}|",
            "|---|---:|",
        ]
        lines.extend(f"|{state.value}|{states[state.value]}|" for state in ExperimentState)
        lines.extend(["", f"## {zh['git']}", "", f"|{zh['git']}|{zh['records']}|", "|---|---:|"])
        lines.extend(f"|{key}|{ownership[key]}|" for key in sorted(ownership))
        if not ownership:
            lines.append(f"|{zh['none']}|0|")
        lines.extend(["", f"## {zh['retention']}", "", f"|{zh['retention_level']}|{zh['records']}|", "|---|---:|"])
        lines.extend(f"|{key}|{retention[key]}|" for key in sorted(retention))
        if not retention:
            lines.append(f"|{zh['none']}|0|")
        deletion_fields = _record_fields(DeletionCandidate)
        lines.extend(
            [
                "",
                f"## {zh['deletion']}",
                "",
                "|" + "|".join(deletion_fields) + "|",
                "|" + "|".join("---" for _ in deletion_fields) + "|",
            ]
        )
        for candidate in _sort_records(deletion_rows, fields_for_key=("candidate_id", "location", "absolute_path")):
            lines.append(
                "|"
                + "|".join(_markdown_cell(getattr(candidate, name)) for name in deletion_fields)
                + "|"
            )
        if not deletion_rows:
            lines.extend(["", zh["no_rows"]])
        lines.extend(["", f"## {zh['gaps']}", "", f"- {zh['nonverified']}\uff1a`{'; '.join(gaps) if gaps else zh['none']}`", f"- {zh['experiment_gaps']}\uff1a`{'; '.join(experiment_gaps) if experiment_gaps else zh['none']}`", f"- {zh['asset_error']}\uff1a`{sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in assets)}`", f"- {zh['experiment_error']}\uff1a`{sum(experiment.experiment_state is ExperimentState.SCAN_ERROR for experiment in experiments)}`", f"- {zh['n607_record_error']}\uff1a`{int(metadata.get('n607_scan_error_count', 0))}`", "", f"## {zh['remote']}", "", f"- {zh['route']}\uff1a`{n607_route}`", f"- {zh['preflight']}\uff1a`{n607_preflight}`", f"- {zh['disconnect']}\uff1a`{n607_disconnect}`", "", f"## {zh['boundary']}", "", zh["zero"], ""])
        return "\n".join(lines)

    def _payloads(self, records: Mapping[str, tuple[Any, ...]]) -> dict[str, bytes]:
        full_inventory = self._full_inventory(records)
        return {
            "report.md": self._report(records).encode("utf-8"),
            "asset_inventory_local.csv": _csv_bytes(
                (asset for asset in records["assets"] if asset.location is Location.LOCAL),
                AssetRecord,
                sort_fields=("location", "root_id", "relative_path", "asset_id"),
            ),
            "asset_inventory_n607.csv": _csv_bytes(
                (asset for asset in records["assets"] if asset.location is Location.N607),
                AssetRecord,
                sort_fields=("location", "root_id", "relative_path", "asset_id"),
            ),
            "experiment_index.csv": _csv_bytes(
                records["experiments"],
                ExperimentRecord,
                sort_fields=("experiment_id",),
            ),
            "git_ownership.csv": _csv_bytes(
                records["git_ownership"],
                GitOwnershipRecord,
                sort_fields=("asset_id", "ownership"),
            ),
            "retention_decisions.csv": _csv_bytes(
                records["retention_decisions"],
                RetentionDecision,
                sort_fields=("asset_id", "retention_class"),
            ),
            "deletion_candidates.csv": _csv_bytes(
                records["deletion_candidates"],
                DeletionCandidate,
                sort_fields=("candidate_id", "location", "absolute_path"),
            ),
            "asset_inventory_full.json": _json_bytes(full_inventory),
        }

    def _receipt_base(self, records: Mapping[str, tuple[Any, ...]]) -> dict[str, Any]:
        assets = records["assets"]
        scopes = records["scope_results"]
        experiments = records["experiments"]
        git_records = records["git_ownership"]
        metadata = self.metadata
        error_counts = {
            "assets": sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in assets),
            "scopes": sum(str(scope.status).upper() == "SCAN_ERROR" for scope in scopes),
            "experiments": sum(experiment.experiment_state is ExperimentState.SCAN_ERROR for experiment in experiments),
            "git_ownership": sum(record.ownership is GitOwnership.GIT_STATE_ERROR or bool(record.error) for record in git_records),
            "retention_decisions": 0,
            "deletion_candidates": 0,
            "n607_records": int(metadata["n607_scan_error_count"]),
        }
        local_scopes = metadata["local_scopes"]
        n607_scopes = metadata["n607_scopes"]
        deletion_rows = [
            _record_dict(candidate)
            for candidate in _sort_records(records["deletion_candidates"], fields_for_key=("candidate_id", "location", "absolute_path"))
        ]
        return {
            "schema_version": self.bundle.schema_version,
            "scan_id": self.scan_id,
            "operator": self.bundle.operator,
            "started_at_utc": self._timestamp(self.bundle.started_at_utc),
            "completed_at_utc": self._timestamp(self.bundle.completed_at_utc),
            "emitted_at_utc": self._timestamp(None),
            "roots": {
                "local": metadata["local_root"],
                "n607": metadata["n607_root"],
            },
            "scopes": {
                "local": list(local_scopes or ()),
                "n607": list(n607_scopes or ()),
            },
            "implementation": {
                "git_head": metadata["implementation_git_head"],
                "tracked_diff_state": metadata["git_tracked_diff_state"],
            },
            "collector_versions": _json_value(metadata["collector_versions"]),
            "counts": {
                "assets": len(assets),
                "assets_local": sum(asset.location is Location.LOCAL for asset in assets),
                "assets_n607": sum(asset.location is Location.N607 for asset in assets),
                "scopes": len(scopes),
                "experiments": len(experiments),
                "git_ownership": len(git_records),
                "retention_decisions": len(records["retention_decisions"]),
                "deletion_candidates": len(records["deletion_candidates"]),
            },
            "scan_error_counts": error_counts,
            "n607_evidence": {
                "requested": metadata["n607_requested"],
                "route": metadata["n607_route"],
                "preflight": metadata["n607_preflight"],
                "disconnect": metadata["n607_disconnect"],
            },
            "source_asset_mutations": 0,
            "moves": 0,
            "overwrites": 0,
            "deletions": 0,
            "deletion_rows": deletion_rows,
            "authorized_deletion_rows": sum(
                candidate.approval_state is not ApprovalState.AWAITING_USER_APPROVAL
                or candidate.execution_state is not ExecutionState.NOT_AUTHORIZED
                or candidate.approved_scope is not None
                for candidate in records["deletion_candidates"]
            ),
            "files": [],
            "external_files": [],
            "receipt_file": {"path": _RECEIPT_NAME, "written_last": True},
        }

    def _write_shards(
        self,
        git_target: Path,
        name: str,
        payload: bytes,
        *,
        external_path: Path,
        external_size: int,
        external_sha256: str,
    ) -> list[Path]:
        """Write complete row-aligned CSV shards, never truncate a row."""

        if not name.endswith(".csv"):
            return []
        try:
            logical_rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline="")))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"invalid CSV payload for sharding: {name}") from exc
        if not logical_rows:
            return []
        header = logical_rows[0]

        def encode_rows(rows: Sequence[Sequence[str]], *, bom: bool) -> bytes:
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows(rows)
            return output.getvalue().encode("utf-8-sig" if bom else "utf-8")

        header_payload = encode_rows((header,), bom=True)
        shards: list[Path] = []
        current: list[bytes] = []
        current_size = 0
        shard_number = 0

        def flush() -> None:
            nonlocal current, current_size, shard_number
            if len(current) <= 1:
                return
            shard_path = git_target / f"{name[:-4]}.{shard_number:03d}.part.csv"
            self._write_exclusive(shard_path, b"".join(current), encoding="utf-8")
            shards.append(shard_path)
            shard_number += 1
            current = []
            current_size = 0

        for row in logical_rows[1:]:
            row_payload = encode_rows((row,), bom=False)
            if len(row_payload) + len(header_payload) > self.git_file_max_bytes:
                raise ValueError(
                    f"CSV row cannot fit git shard threshold for {name}: "
                    f"row_bytes={len(row_payload)} header_bytes={len(header_payload)} "
                    f"limit={self.git_file_max_bytes}"
                )
            if not current:
                current = [header_payload]
                current_size = len(header_payload)
            if current_size + len(row_payload) > self.git_file_max_bytes:
                flush()
                current = [header_payload]
                current_size = len(header_payload)
            current.append(row_payload)
            current_size += len(row_payload)
        flush()
        summary = {
            "artifact": name,
            "external_path": str(external_path.resolve()),
            "external_bytes": external_size,
            "external_sha256": external_sha256,
            "shards": [path.name for path in shards],
            "row_count": max(0, len(logical_rows) - 1),
        }
        self._write_exclusive(git_target / f"{name[:-4]}.summary.json", _json_bytes(summary), encoding="utf-8")
        return shards

    def emit(self) -> EmissionResult:
        """Emit a fresh report and receipt, preserving partial output on error."""

        git_target, external_target = self._target_paths()
        self._emission_now_utc = _iso_utc(None)
        self.output_root.mkdir(parents=True, exist_ok=True)
        git_target.mkdir()
        records = self._records()
        payloads = self._payloads(records)
        small_receipt = self._receipt_base(records)
        small_receipt["artifact_route"] = "GIT_COMPLETE"
        small_receipt["files"] = [
            _payload_entry(name, payloads[name]) for name in sorted(payloads)
        ]
        small_receipt["external_files"] = []
        small_receipt_payload = _json_bytes(small_receipt)
        oversize = any(len(payload) > self.git_file_max_bytes for payload in payloads.values())
        oversize = oversize or len(small_receipt_payload) > self.git_file_max_bytes
        oversize = oversize or (
            sum(len(payload) for payload in payloads.values()) + len(small_receipt_payload)
            > self.git_scan_max_bytes
        )
        external_output: Path | None = None
        external_entries: list[dict[str, Any]] = []

        if not oversize:
            for name in _ARTIFACT_ORDER:
                self._write_exclusive(git_target / name, payloads[name], encoding="utf-8")
        else:
            external_target.parent.mkdir(parents=True, exist_ok=True)
            external_target.mkdir()
            external_output = external_target
            external_entry_by_name: dict[str, dict[str, Any]] = {}
            for name in _ARTIFACT_ORDER:
                external_path = external_target / name
                self._write_exclusive(external_path, payloads[name], encoding="utf-8")
                entry = _file_entry(external_path)
                external_entries.append(entry)
                external_entry_by_name[name] = entry

            for name in _ARTIFACT_ORDER:
                external_path = external_target / name
                entry = external_entry_by_name[name]
                if name == "report.md":
                    if len(payloads[name]) <= self.git_file_max_bytes:
                        self._write_exclusive(git_target / name, payloads[name], encoding="utf-8")
                    else:
                        compact = (
                            "# \u9879\u76ee\u8d44\u4ea7\u6cbb\u7406\u626b\u63cf\u62a5\u544a\n\n"
                            "\u5b8c\u6574\u62a5\u544a\u5df2\u4fdd\u5b58\u5230\u5916\u7f6e\u6e05\u5355\uff1b\u672cGit\u4fa7\u6587\u4ef6\u4ec5\u4fdd\u7559\u8def\u7531\u6458\u8981\u3002\n"
                            f"\u5916\u7f6e\u8def\u5f84\uff1a`{external_path.resolve()}`\n"
                        ).encode("utf-8")
                        self._write_exclusive(git_target / name, compact, encoding="utf-8")
                elif name.endswith(".csv"):
                    self._write_shards(
                        git_target,
                        name,
                        payloads[name],
                        external_path=external_path,
                        external_size=entry["bytes"],
                        external_sha256=entry["sha256"],
                    )
                else:
                    summary = {
                        "artifact": name,
                        "external_path": str(external_path.resolve()),
                        "external_bytes": entry["bytes"],
                        "external_sha256": entry["sha256"],
                    }
                    self._write_exclusive(git_target / f"{name}.summary.json", _json_bytes(summary), encoding="utf-8")

        oversized_git_files = [
            path
            for path in git_target.iterdir()
            if path.is_file() and path.stat().st_size > self.git_file_max_bytes
        ]
        if oversized_git_files:
            raise ValueError(
                "git output exceeds per-file threshold: "
                + ", ".join(path.name for path in sorted(oversized_git_files))
            )

        receipt = self._receipt_base(records)
        receipt["artifact_route"] = "EXTERNAL_COMPLETE_WITH_GIT_SHARDS" if oversize else "GIT_COMPLETE"
        receipt["files"] = [
            _file_entry(git_target / name, relative_to=git_target)
            for name in sorted(path.name for path in git_target.iterdir())
        ]
        receipt["external_files"] = sorted(external_entries, key=lambda item: item["path"])
        receipt_payload = _json_bytes(receipt)
        if len(receipt_payload) > self.git_file_max_bytes:
            raise ValueError(
                "scan receipt exceeds git file threshold: "
                f"bytes={len(receipt_payload)} limit={self.git_file_max_bytes}"
            )
        git_total_bytes = sum(
            path.stat().st_size for path in git_target.iterdir() if path.is_file()
        ) + len(receipt_payload)
        if git_total_bytes > self.git_scan_max_bytes:
            raise ValueError(
                "git output exceeds scan threshold: "
                f"bytes={git_total_bytes} limit={self.git_scan_max_bytes}"
            )
        self._write_exclusive(git_target / _RECEIPT_NAME, receipt_payload, encoding="utf-8")
        return EmissionResult(git_target, external_output, receipt)


__all__ = [
    "DEFAULT_GIT_FILE_MAX_BYTES",
    "DEFAULT_GIT_SCAN_MAX_BYTES",
    "EmissionResult",
    "ReportEmitter",
]
