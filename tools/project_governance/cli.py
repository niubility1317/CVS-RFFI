"""Safe, dependency-injected orchestration for project asset inventory.

This module deliberately contains no concrete process runner.  The top-level
entrypoint owns the narrowly allowlisted production runner; callers of this
module supply an N607 collector only when the explicit ``--include-n607``
option is selected.  The default local path therefore cannot construct or
contact N607.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .classify_retention import (
    RetentionEvidence,
    build_deletion_candidates,
    classify_retentions,
)
from .collect_git import GitCommandRunner, GitOwnershipMapper, RepositoryRecord
from .collect_local import LocalCollector
from .collect_n607 import (
    APPROVED_ENDPOINTS,
    AttemptReceipt,
    ConnectionEvidence,
    N607CollectionResult,
    RemoteOutcome,
)
from .config import GovernanceConfig, load_config
from .emit import ReportEmitter, ScanProgressJournal
from .index_experiments import ExperimentIndex, index_experiments
from .models import (
    AccessStatus,
    AssetKind,
    AssetRecord,
    ExperimentRecord,
    ExperimentState,
    GitOwnership,
    GitOwnershipRecord,
    HashStatus,
    Location,
    ScanBundle,
    ScopeResult,
)
from .paths import normalize_relative_path, stable_asset_id
from .query_index import QueryStore, build_index, load_latest


CLI_VERSION = "project-governance-cli-v1"
_REMOTE_SCOPE_STATUSES = frozenset({"VERIFIED", "NOT_PRESENT", "SCAN_ERROR"})
_REMOTE_ATTEMPT_LABELS = frozenset({"PREFLIGHT", "DIRECT", "LAB_BRIDGE"})
_REMOTE_CONNECTION_STATES = frozenset({"ESTABLISHED", "SYN_SENT"})
_MAX_ATTEMPT_STDERR_BYTES = 8192


@dataclass(frozen=True)
class ScanOutcome:
    """The small, JSON-safe result returned by the orchestration boundary."""

    exit_code: int
    scan_id: str
    output_dir: str | None
    external_output_dir: str | None
    remote_contacted: bool
    remote_outcome: str
    local_error_count: int
    remote_error_count: int
    message: str | None = None
    terminal_state: str = "NOT_STARTED"
    stage: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "scan_id": self.scan_id,
            "output_dir": self.output_dir,
            "external_output_dir": self.external_output_dir,
            "remote_contacted": self.remote_contacted,
            "remote_outcome": self.remote_outcome,
            "local_error_count": self.local_error_count,
            "remote_error_count": self.remote_error_count,
            "message": self.message,
            "terminal_state": self.terminal_state,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class _RemoteFacts:
    assets: tuple[AssetRecord, ...]
    scopes: tuple[ScopeResult, ...]
    error_scopes: tuple[ScopeResult, ...]
    processes: tuple[Mapping[str, object], ...]
    outcome: str
    route: str
    preflight: str
    disconnect: str
    attempts: tuple[Mapping[str, object], ...]
    active_training_observed: bool
    protocol_error_count: int
    error_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project_governance_inventory",
        description="Read-only project asset governance inventory.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="collect a fresh read-only inventory")
    scan.add_argument("--config", required=True, help="validated governance JSON configuration")
    scan.add_argument("--scan-id", required=True, help="new immutable output component")
    scan.add_argument("--output-root", required=True, help="Git-backed governance output root")
    scan.add_argument(
        "--external-output-root",
        required=True,
        help="separate external output root for oversized tables",
    )
    scan.add_argument("--operator", required=True, help="recorded human or service operator")
    scan.add_argument(
        "--include-n607",
        action="store_true",
        help="request the injected, read-only N607 collector",
    )
    scan.add_argument(
        "--print-plan",
        action="store_true",
        help="validate and print the exact plan without collecting or writing",
    )
    build = commands.add_parser(
        "build-index", help="build a local SQLite query index from one completed scan"
    )
    build.add_argument("--receipt", required=True, help="completed scan_receipt.json")
    build.add_argument("--external-root", required=True, help="exact external CSV directory")
    build.add_argument("--database", required=True, help="new external governance.sqlite")
    build.add_argument("--json", action="store_true", help="emit compact JSON")

    def add_latest(command: argparse.ArgumentParser) -> None:
        command.add_argument("--latest", required=True, help="validated latest.json pointer")
        command.add_argument("--json", action="store_true", help="emit compact JSON")

    status = commands.add_parser("status", help="show the latest governance baseline")
    add_latest(status)
    find = commands.add_parser("find", help="find an asset by ID or absolute path")
    find.add_argument("query")
    find.add_argument("--limit", type=int, default=20)
    add_latest(find)
    experiment = commands.add_parser("experiment", help="summarize one exact run ID")
    experiment.add_argument("run_id")
    add_latest(experiment)
    repo = commands.add_parser("repo", help="show Git ownership for one exact path")
    repo.add_argument("path")
    add_latest(repo)
    review = commands.add_parser("review", help="list bounded manual-review records")
    review.add_argument("--location", choices=("LOCAL", "N607"))
    review.add_argument("--retention-class")
    review.add_argument("--experiment-state")
    review.add_argument("--ownership")
    review.add_argument("--limit", type=int, default=20)
    add_latest(review)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only the approved non-mutating scan surface."""

    return build_parser().parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _require_trimmed(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a trimmed non-empty string")
    return value


def _scope_descriptions(scopes: Iterable[ScopeResult]) -> tuple[str, ...]:
    return tuple(
        f"{scope.relative_path or '.'}:{scope.status}"
        for scope in sorted(scopes, key=lambda item: (item.relative_path, item.status))
    )


def _require_verified_root_scope(
    scopes: Iterable[ScopeResult], *, location: Location, root_id: str, scan_id: str
) -> tuple[ScopeResult, ...]:
    """Fail closed when a collector does not verify exactly one configured root."""

    recorded = tuple(scopes)
    root_scopes = tuple(
        scope
        for scope in recorded
        if scope.location is location and scope.relative_path == ""
    )
    if len(root_scopes) == 1 and root_scopes[0].status == "VERIFIED":
        return recorded
    if len(root_scopes) == 1 and root_scopes[0].status == "SCAN_ERROR":
        return recorded
    if not root_scopes:
        return recorded + (
            ScopeResult(
                scan_id=scan_id,
                location=location,
                root_id=root_id,
                relative_path="",
                status="SCAN_ERROR",
                error=f"configured {location.value} root scope is missing",
            ),
        )
    error = f"configured {location.value} root scope is not uniquely VERIFIED"
    return tuple(
        replace(scope, status="SCAN_ERROR", asset_ids=None, error=scope.error or error)
        if scope.location is location and scope.relative_path == ""
        else scope
        for scope in recorded
    )


def _not_requested_remote() -> _RemoteFacts:
    return _RemoteFacts(
        assets=(),
        scopes=(),
        error_scopes=(),
        processes=(),
        outcome="NOT_REQUESTED",
        route="NOT_REQUESTED",
        preflight="NOT_REQUESTED",
        disconnect="NOT_REQUESTED",
        attempts=(),
        active_training_observed=False,
        protocol_error_count=0,
        error_count=0,
    )


def _metadata(
    config: GovernanceConfig,
    *,
    local_scopes: Iterable[ScopeResult] = (),
    remote: _RemoteFacts | None = None,
    implementation_git_head: str = "NOT_COLLECTED",
    git_tracked_diff_state: str = "NOT_COLLECTED",
) -> dict[str, object]:
    selected_remote = remote or _not_requested_remote()
    return {
        "local_root": config.local.root,
        "local_scopes": _scope_descriptions(local_scopes),
        "n607_root": config.n607.root,
        "n607_scopes": _scope_descriptions(selected_remote.scopes),
        "implementation_git_head": implementation_git_head,
        "git_tracked_diff_state": git_tracked_diff_state,
        "collector_versions": {
            "cli": CLI_VERSION,
            "local": "project-governance-local-v1",
            "git": "project-governance-git-v1",
            "n607": "project-governance-n607-v1",
        },
        "n607_requested": selected_remote.outcome != "NOT_REQUESTED",
        "n607_outcome": selected_remote.outcome,
        "n607_route": selected_remote.route,
        "n607_preflight": selected_remote.preflight,
        "n607_disconnect": selected_remote.disconnect,
        "n607_attempts": selected_remote.attempts,
        "n607_active_training_observed": selected_remote.active_training_observed,
        "n607_scan_error_count": selected_remote.protocol_error_count,
    }


def _validate_output_targets(
    args: argparse.Namespace, config: GovernanceConfig
) -> tuple[Path, Path]:
    """Reuse the emitter's non-writing validation before discovery starts."""

    scan_id = _require_trimmed(getattr(args, "scan_id", None), "scan_id")
    operator = _require_trimmed(getattr(args, "operator", None), "operator")
    output_root = _require_trimmed(getattr(args, "output_root", None), "output_root")
    external_output_root = _require_trimmed(
        getattr(args, "external_output_root", None), "external_output_root"
    )
    probe = ReportEmitter(
        ScanBundle(scan_id=scan_id, operator=operator),
        output_root=output_root,
        external_output_root=external_output_root,
        metadata=_metadata(config),
        git_file_max_bytes=config.output.git_file_max_bytes,
        git_scan_max_bytes=config.output.git_scan_max_bytes,
    )
    return probe._target_paths()


def _validate_request(args: argparse.Namespace, config: GovernanceConfig) -> tuple[Path, Path]:
    if not isinstance(config, GovernanceConfig):
        raise ValueError("config must be a GovernanceConfig")
    if getattr(args, "command", None) != "scan":
        raise ValueError("only the scan command is permitted")
    return _validate_output_targets(args, config)


def _asset_from_remote(record: Mapping[str, object], config: GovernanceConfig, scan_id: str) -> AssetRecord:
    if record.get("scan_id") != scan_id:
        raise ValueError("remote asset scan_id does not match the requested scan")
    if record.get("location") != Location.N607.value:
        raise ValueError("remote asset location is not N607")
    if record.get("root_id") != config.n607.root_id:
        raise ValueError("remote asset root_id does not match the configuration")
    relative_path = normalize_relative_path(str(record.get("relative_path", "")), location=Location.N607)
    asset_id = stable_asset_id(Location.N607, config.n607.root_id, relative_path)
    if record.get("asset_id") != asset_id:
        raise ValueError("remote asset_id is not the stable configured identity")
    display_name = record.get("display_name")
    escaped_name = record.get("escaped_name")
    if not isinstance(display_name, str) or not display_name:
        raise ValueError("remote asset display_name is invalid")
    if not isinstance(escaped_name, str) or not escaped_name:
        raise ValueError("remote asset escaped_name is invalid")
    size_bytes = record.get("size_bytes")
    if size_bytes is not None and (type(size_bytes) is not int or size_bytes < 0):
        raise ValueError("remote asset size_bytes is invalid")
    mtime_utc = record.get("mtime_utc")
    if mtime_utc is not None and not isinstance(mtime_utc, str):
        raise ValueError("remote asset mtime_utc is invalid")
    sha256 = record.get("sha256")
    if sha256 is not None and not isinstance(sha256, str):
        raise ValueError("remote asset sha256 is invalid")
    role = record.get("evidence_role")
    if role is not None and not isinstance(role, str):
        raise ValueError("remote asset evidence_role is invalid")
    try:
        asset_kind = AssetKind(str(record.get("asset_kind")))
        access_status = AccessStatus(str(record.get("access_status")))
        hash_status = HashStatus(str(record.get("hash_status")))
    except ValueError as exc:
        raise ValueError("remote asset vocabulary is invalid") from exc
    return AssetRecord(
        asset_id=asset_id,
        scan_id=scan_id,
        location=Location.N607,
        root_id=config.n607.root_id,
        relative_path=relative_path,
        display_name=display_name,
        escaped_name=escaped_name,
        asset_kind=asset_kind,
        size_bytes=size_bytes,
        mtime_utc=mtime_utc,
        access_status=access_status,
        hash_status=hash_status,
        sha256=sha256,
        evidence_role=role,
        decision_reason="REMOTE_COLLECTED",
    )


def _scope_from_remote(record: Mapping[str, object], config: GovernanceConfig, scan_id: str) -> ScopeResult:
    if record.get("scan_id") != scan_id:
        raise ValueError("remote scope scan_id does not match the requested scan")
    if record.get("location") != Location.N607.value:
        raise ValueError("remote scope location is not N607")
    if record.get("root_id") != config.n607.root_id:
        raise ValueError("remote scope root_id does not match the configuration")
    relative_raw = record.get("relative_path")
    if relative_raw == "":
        relative_path = ""
    else:
        relative_path = normalize_relative_path(str(relative_raw), location=Location.N607)
    status = record.get("status")
    if not isinstance(status, str) or status not in _REMOTE_SCOPE_STATUSES:
        raise ValueError("remote scope status is invalid")
    raw_ids = record.get("asset_ids")
    if raw_ids is None:
        asset_ids: tuple[str, ...] | None = None
    elif isinstance(raw_ids, list) and all(isinstance(item, str) for item in raw_ids):
        asset_ids = tuple(raw_ids)
    else:
        raise ValueError("remote scope asset_ids is invalid")
    error = record.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("remote scope error is invalid")
    return ScopeResult(
        scan_id=scan_id,
        location=Location.N607,
        root_id=config.n607.root_id,
        relative_path=relative_path,
        status=status,
        asset_ids=asset_ids,
        error=error,
    )


def _error_scope_from_remote(
    record: Mapping[str, object], config: GovernanceConfig, scan_id: str
) -> ScopeResult:
    if record.get("scan_id") != scan_id:
        raise ValueError("remote scan error scan_id does not match the requested scan")
    if record.get("location") != Location.N607.value:
        raise ValueError("remote scan error location is not N607")
    if record.get("root_id") != config.n607.root_id:
        raise ValueError("remote scan error root_id does not match the configuration")
    relative_raw = record.get("relative_path")
    if not isinstance(relative_raw, str):
        raise ValueError("remote scan error relative_path is invalid")
    relative_path = (
        ""
        if relative_raw == ""
        else normalize_relative_path(relative_raw, location=Location.N607)
    )
    operation = record.get("operation")
    error_type = record.get("error_type")
    error = record.get("error")
    if not isinstance(operation, str) or not operation:
        raise ValueError("remote scan error operation is invalid")
    if not isinstance(error_type, str) or not error_type:
        raise ValueError("remote scan error error_type is invalid")
    if not isinstance(error, str) or not error:
        raise ValueError("remote scan error message is invalid")
    structured_error = json.dumps(
        {
            "record_type": "SCAN_ERROR",
            "operation": operation,
            "error_type": error_type,
            "error": error,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ScopeResult(
        scan_id=scan_id,
        location=Location.N607,
        root_id=config.n607.root_id,
        relative_path=relative_path,
        status="SCAN_ERROR",
        asset_ids=None,
        error=structured_error,
    )


def _process_from_remote(record: Mapping[str, object]) -> Mapping[str, object]:
    pid = record.get("pid")
    ppid = record.get("ppid")
    cwd = record.get("cwd")
    cmdline = record.get("cmdline")
    if type(pid) is not int or pid <= 0:
        raise ValueError("remote process pid is invalid")
    if type(ppid) is not int or ppid < 0:
        raise ValueError("remote process ppid is invalid")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("remote process cwd is invalid")
    if not isinstance(cmdline, str) or not cmdline:
        raise ValueError("remote process cmdline is invalid")
    return {"pid": pid, "cwd": cwd, "cmdline": cmdline, "run_root": None}


def _connection_from_remote(
    connection: ConnectionEvidence, *, attempt_pids: frozenset[int]
) -> Mapping[str, object]:
    if not isinstance(connection, ConnectionEvidence):
        raise ValueError("remote attempt connection evidence has an invalid type")
    if type(connection.pid) is not int or connection.pid <= 0 or connection.pid not in attempt_pids:
        raise ValueError("remote attempt connection PID is invalid")
    if connection.endpoint not in APPROVED_ENDPOINTS:
        raise ValueError("remote attempt connection endpoint is not approved")
    if connection.state not in _REMOTE_CONNECTION_STATES:
        raise ValueError("remote attempt connection state is invalid")
    return {
        "pid": connection.pid,
        "endpoint": connection.endpoint,
        "state": connection.state,
    }


def _attempt_from_remote(attempt: AttemptReceipt) -> Mapping[str, object]:
    if not isinstance(attempt, AttemptReceipt):
        raise ValueError("remote receipt attempt has an invalid type")
    if attempt.label not in _REMOTE_ATTEMPT_LABELS:
        raise ValueError("remote receipt attempt label is invalid")
    if attempt.child_pid is not None and (
        type(attempt.child_pid) is not int or attempt.child_pid <= 0
    ):
        raise ValueError("remote receipt attempt child PID is invalid")
    if not isinstance(attempt.proxy_child_pids, tuple) or any(
        type(pid) is not int or pid <= 0 for pid in attempt.proxy_child_pids
    ):
        raise ValueError("remote receipt attempt proxy PIDs are invalid")
    if len(set(attempt.proxy_child_pids)) != len(attempt.proxy_child_pids):
        raise ValueError("remote receipt attempt proxy PIDs are not unique")
    if attempt.returncode is not None and type(attempt.returncode) is not int:
        raise ValueError("remote receipt attempt return code is invalid")
    for field_name in ("timed_out", "child_exited", "proxy_children_exited"):
        if type(getattr(attempt, field_name)) is not bool:
            raise ValueError(f"remote receipt attempt {field_name} is invalid")
    if attempt.disconnect_status not in {"VERIFIED", "UNKNOWN"}:
        raise ValueError("remote receipt attempt disconnect status is invalid")
    if not isinstance(attempt.stderr_tail, str):
        raise ValueError("remote receipt attempt stderr tail is invalid")
    if len(attempt.stderr_tail.encode("utf-8")) > _MAX_ATTEMPT_STDERR_BYTES:
        raise ValueError("remote receipt attempt stderr tail exceeds the bound")
    if not isinstance(attempt.lingering_connections, tuple):
        raise ValueError("remote receipt lingering connections are invalid")
    attempt_pids = frozenset(
        (() if attempt.child_pid is None else (attempt.child_pid,)) + attempt.proxy_child_pids
    )
    lingering = tuple(
        _connection_from_remote(connection, attempt_pids=attempt_pids)
        for connection in attempt.lingering_connections
    )
    return {
        "label": attempt.label,
        "child_pid": attempt.child_pid,
        "proxy_child_pids": attempt.proxy_child_pids,
        "returncode": attempt.returncode,
        "timed_out": attempt.timed_out,
        "child_exited": attempt.child_exited,
        "proxy_children_exited": attempt.proxy_children_exited,
        "disconnect_status": attempt.disconnect_status,
        "lingering_connections": lingering,
        "stderr_tail": attempt.stderr_tail,
    }


def _attempts_from_remote(
    attempts: tuple[AttemptReceipt, ...], *, route: str, outcome: str
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(attempts, tuple):
        raise ValueError("remote receipt attempts must be an immutable sequence")
    converted = tuple(_attempt_from_remote(attempt) for attempt in attempts)
    labels = tuple(str(attempt["label"]) for attempt in converted)
    if len(set(labels)) != len(labels):
        raise ValueError("remote receipt attempt labels are not unique")
    expected = {
        "DIRECT": ("PREFLIGHT", "DIRECT"),
        "LAB_BRIDGE": ("PREFLIGHT", "LAB_BRIDGE"),
    }.get(route)
    if expected is not None and labels != expected:
        raise ValueError("remote receipt attempts do not match the selected route")
    if route == "NO_ROUTE" and labels not in {(), ("PREFLIGHT",)}:
        raise ValueError("remote receipt attempts do not match the no-route outcome")
    if outcome == RemoteOutcome.VERIFIED.value:
        if any(
            attempt["child_pid"] is None
            or attempt["returncode"] != 0
            or attempt["timed_out"]
            or not attempt["child_exited"]
            or not attempt["proxy_children_exited"]
            or attempt["disconnect_status"] != "VERIFIED"
            or attempt["lingering_connections"]
            for attempt in converted
        ):
            raise ValueError("verified remote receipt has incomplete attempt evidence")
        if any(
            attempt["label"] in {"PREFLIGHT", "LAB_BRIDGE"}
            and not attempt["proxy_child_pids"]
            for attempt in converted
        ):
            raise ValueError("verified remote receipt lacks required proxy exit evidence")
    return converted


def _convert_remote(
    result: N607CollectionResult, config: GovernanceConfig, scan_id: str
) -> _RemoteFacts:
    receipt = result.receipt
    route = receipt.route or "NO_ROUTE"
    preflight = receipt.preflight_status
    disconnect = receipt.disconnect_status
    outcome = receipt.outcome.value
    if route not in {"DIRECT", "LAB_BRIDGE", "NO_ROUTE"}:
        raise ValueError("remote receipt route is invalid")
    if preflight not in {"DIRECT_READY", "DIRECT_PATH_UNAVAILABLE", "FAILED", "UNKNOWN"}:
        raise ValueError("remote receipt preflight state is invalid")
    if disconnect not in {"VERIFIED", "UNKNOWN"}:
        raise ValueError("remote receipt disconnect state is invalid")
    if type(receipt.active_training_observed) is not bool:
        raise ValueError("remote receipt active-training evidence is invalid")
    if receipt.error is not None:
        if not isinstance(receipt.error, str) or not receipt.error:
            raise ValueError("remote receipt error is invalid")
        if len(receipt.error.encode("utf-8")) > _MAX_ATTEMPT_STDERR_BYTES:
            raise ValueError("remote receipt error exceeds the bound")
        if outcome == RemoteOutcome.VERIFIED.value:
            raise ValueError("verified remote receipt cannot contain an error")
    attempts = _attempts_from_remote(receipt.attempts, route=route, outcome=outcome)
    assets: list[AssetRecord] = []
    scopes: list[ScopeResult] = []
    error_scopes: list[ScopeResult] = []
    processes: list[Mapping[str, object]] = []
    reported_error_count: int | None = None
    protocol_errors = 0
    for record in result.records:
        if not isinstance(record, Mapping):
            raise ValueError("remote collector returned a non-object record")
        record_type = record.get("record_type")
        if record_type == "ASSET":
            asset = _asset_from_remote(record, config, scan_id)
            assets.append(asset)
        elif record_type == "SCOPE":
            scope = _scope_from_remote(record, config, scan_id)
            scopes.append(scope)
        elif record_type == "PROCESS":
            if record.get("scan_id") != scan_id:
                raise ValueError("remote process scan_id does not match the requested scan")
            processes.append(_process_from_remote(record))
        elif record_type == "SCAN_ERROR":
            protocol_errors += 1
            error_scopes.append(_error_scope_from_remote(record, config, scan_id))
        elif record_type == "COLLECTION_COMPLETE":
            if reported_error_count is not None:
                raise ValueError("remote collection has multiple COLLECTION_COMPLETE records")
            count = record.get("scan_error_count")
            if type(count) is not int or count < 0:
                raise ValueError("remote completion scan_error_count is invalid")
            reported_error_count = count
    if outcome == RemoteOutcome.VERIFIED.value and reported_error_count is None:
        raise ValueError("verified remote collection lacks COLLECTION_COMPLETE evidence")
    if outcome == RemoteOutcome.VERIFIED.value and result.records[-1].get("record_type") != "COLLECTION_COMPLETE":
        raise ValueError("verified remote collection is not terminally closed")
    if reported_error_count is not None and reported_error_count != protocol_errors:
        raise ValueError("remote COLLECTION_COMPLETE scan_error_count does not close")
    closed_protocol_errors = (
        reported_error_count if reported_error_count is not None else protocol_errors
    )
    if (
        outcome != RemoteOutcome.VERIFIED.value
        and receipt.error is not None
        and not any(scope.relative_path == "" for scope in scopes)
    ):
        scopes.append(
            ScopeResult(
                scan_id=scan_id,
                location=Location.N607,
                root_id=config.n607.root_id,
                relative_path="",
                status="SCAN_ERROR",
                error=receipt.error,
            )
        )
    scopes = list(
        _require_verified_root_scope(
            scopes,
            location=Location.N607,
            root_id=config.n607.root_id,
            scan_id=scan_id,
        )
    )
    status_errors = sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in assets) + sum(
        scope.status == "SCAN_ERROR" for scope in scopes
    )
    return _RemoteFacts(
        assets=tuple(assets),
        scopes=tuple(scopes),
        error_scopes=tuple(error_scopes),
        processes=tuple(processes),
        outcome=outcome,
        route=route,
        preflight=preflight,
        disconnect=disconnect,
        attempts=attempts,
        active_training_observed=receipt.active_training_observed,
        protocol_error_count=closed_protocol_errors,
        error_count=max(closed_protocol_errors, status_errors),
    )


def _remote_failure_from_exception(error: Exception, *, scan_id: str, root_id: str) -> _RemoteFacts:
    return _RemoteFacts(
        assets=(),
        scopes=(
            ScopeResult(
                scan_id=scan_id,
                location=Location.N607,
                root_id=root_id,
                relative_path="",
                status="SCAN_ERROR",
                error=str(error),
            ),
        ),
        error_scopes=(),
        processes=(),
        outcome=RemoteOutcome.UNKNOWN.value,
        route="NO_ROUTE",
        preflight="UNKNOWN",
        disconnect="UNKNOWN",
        attempts=(),
        active_training_observed=False,
        protocol_error_count=0,
        error_count=1,
    )


def _make_n607_collector(
    factory: Callable[..., object], config: GovernanceConfig, scan_id: str
) -> object:
    return factory(config, scan_id)


def _ownership_and_assets(
    local_assets: tuple[AssetRecord, ...],
    remote_assets: tuple[AssetRecord, ...],
    *,
    repository_seeds: Iterable[str | Path],
    root_paths: Mapping[object, str | Path],
    git_runner: GitCommandRunner | None,
) -> tuple[
    tuple[AssetRecord, ...],
    tuple[GitOwnershipRecord, ...],
    tuple[RepositoryRecord, ...],
]:
    mapper = GitOwnershipMapper(
        repository_seeds,
        root_paths,
        git_runner,
    )
    ownership = mapper.map(local_assets)
    local_records = tuple(ownership[asset.asset_id] for asset in local_assets)
    attached_local = tuple(
        replace(asset, git_ownership=ownership[asset.asset_id].ownership)
        for asset in local_assets
    )
    remote_records = tuple(
        GitOwnershipRecord(asset_id=asset.asset_id, ownership=GitOwnership.REMOTE_NON_GIT)
        for asset in remote_assets
    )
    attached_remote = tuple(
        replace(asset, git_ownership=GitOwnership.REMOTE_NON_GIT) for asset in remote_assets
    )
    return attached_local + attached_remote, local_records + remote_records, mapper.repositories


def _implementation_state(
    records: Iterable[RepositoryRecord], *, implementation_repository: str | Path
) -> tuple[str, str]:
    target = Path(implementation_repository).resolve(strict=False)
    selected = tuple(
        record
        for record in records
        if Path(record.repository_root).resolve(strict=False) == target
    )
    heads = sorted(
        {record.head_commit for record in selected if isinstance(record.head_commit, str) and record.head_commit}
    )
    if len(heads) == 1:
        head = heads[0]
    else:
        head = "UNAVAILABLE"
    statuses = [record.status_summary for record in selected if record.status_summary is not None]
    if statuses and all(not status for status in statuses):
        tracked_state = "CLEAN"
    elif statuses:
        tracked_state = "DIRTY_OR_UNKNOWN"
    else:
        tracked_state = "UNAVAILABLE"
    return head, tracked_state


def _assets_with_experiment_ids(
    assets: Iterable[AssetRecord], experiments: ExperimentIndex
) -> tuple[AssetRecord, ...]:
    mapped: dict[str, str | None] = {}
    for experiment_id, claims in experiments.claims_by_experiment.items():
        for claim in claims:
            current = mapped.get(claim.source_asset_id)
            if current is None and claim.source_asset_id not in mapped:
                mapped[claim.source_asset_id] = experiment_id
            elif current != experiment_id:
                mapped[claim.source_asset_id] = None
    return tuple(
        replace(asset, experiment_id=mapped.get(asset.asset_id)) for asset in assets
    )


def _absolute_asset_path(asset: AssetRecord, root_paths: Mapping[str, str | Path]) -> str:
    root = str(root_paths[asset.root_id])
    if asset.location is Location.N607:
        return str(PurePosixPath(root).joinpath(*asset.relative_path.split("/")))
    return str(Path(root).joinpath(*asset.relative_path.split("/")))


def _retention_evidence(
    assets: Iterable[AssetRecord], experiments: Mapping[str, ExperimentRecord], root_paths: Mapping[str, str | Path]
) -> dict[str, RetentionEvidence]:
    evidence: dict[str, RetentionEvidence] = {}
    for asset in assets:
        experiment = experiments.get(asset.experiment_id) if asset.experiment_id else None
        role = (asset.evidence_role or "").casefold()
        evidence[asset.asset_id] = RetentionEvidence(
            evidence_asset_ids=(asset.asset_id,),
            active_process=(
                experiment is not None and experiment.experiment_state is ExperimentState.ACTIVE_LIVE
            ),
            experiment_state=experiment.experiment_state if experiment is not None else None,
            git_worktree_dependency=asset.git_ownership
            in {GitOwnership.TRACKED_GIT, GitOwnership.UNTRACKED_IN_GIT_WORKTREE},
            report_dependency="report" in role,
            manifest_dependency="manifest" in role,
            receipt_dependency="receipt" in role,
            provenance_known=False,
            purpose_known=False,
            deletion_candidate_requested=False,
            absolute_path=_absolute_asset_path(asset, root_paths),
            read_error=asset.access_status is AccessStatus.SCAN_ERROR,
            git_error=asset.git_ownership is GitOwnership.GIT_STATE_ERROR,
        )
    return evidence


def _with_retention(
    assets: Iterable[AssetRecord], decisions: Iterable[object]
) -> tuple[AssetRecord, ...]:
    decision_by_id = {decision.asset_id: decision for decision in decisions}
    updated: list[AssetRecord] = []
    for asset in assets:
        decision = decision_by_id[asset.asset_id]
        action = "KEEP" if decision.retention_class.value.startswith("KEEP") else "REVIEW"
        updated.append(
            replace(
                asset,
                retention_class=decision.retention_class,
                recommended_action=action,
                decision_reason=decision.rule_code,
            )
        )
    return tuple(updated)


def _local_error_count(
    assets: Iterable[AssetRecord], scopes: Iterable[ScopeResult], ownership: Iterable[GitOwnershipRecord]
) -> int:
    return (
        sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in assets)
        + sum(scope.status == "SCAN_ERROR" for scope in scopes)
        + sum(record.ownership is GitOwnership.GIT_STATE_ERROR for record in ownership)
    )


def _outcome_code(local_errors: int, remote: _RemoteFacts) -> int:
    if remote.outcome == RemoteOutcome.UNKNOWN.value or remote.disconnect == "UNKNOWN":
        return 3
    if local_errors or remote.error_count or remote.outcome == RemoteOutcome.FAILED.value:
        return 2
    return 0


def print_plan(args: argparse.Namespace, *, config: GovernanceConfig | None = None) -> dict[str, object]:
    """Validate and print a one-line, no-side-effect scan plan."""

    selected_config = config or load_config(getattr(args, "config", None), probe_local_paths=False)
    git_target, external_target = _validate_request(args, selected_config)
    payload = {
        "command": "scan",
        "scan_id": args.scan_id,
        "operator": args.operator,
        "local_root": selected_config.local.root,
        "n607_root": selected_config.n607.root,
        "local_scopes": [surface.relative_path for surface in selected_config.local.carrier_surfaces],
        "n607_scopes": [surface.relative_path for surface in selected_config.n607.carrier_surfaces],
        "control_evidence_max_depth": selected_config.discovery.control_evidence_max_depth,
        "hash_max_bytes": selected_config.discovery.hash_max_bytes,
        "text_read_max_bytes": selected_config.discovery.text_read_max_bytes,
        "git_file_max_bytes": selected_config.output.git_file_max_bytes,
        "git_scan_max_bytes": selected_config.output.git_scan_max_bytes,
        "output_targets": {"git": str(git_target), "external": str(external_target)},
        "n607_contact": bool(getattr(args, "include_n607", False)),
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return payload


def run_scan(
    args: argparse.Namespace,
    *,
    config: GovernanceConfig | None = None,
    git_runner: GitCommandRunner | None = None,
    repository_seeds: Iterable[str | Path] | None = None,
    implementation_repository: str | Path | None = None,
    n607_collector_factory: Callable[..., object] | None = None,
    clock: Callable[[], str] = _utc_now,
) -> ScanOutcome:
    """Execute the fixed read-only collection sequence through injected edges."""

    scan_id = getattr(args, "scan_id", "") if isinstance(getattr(args, "scan_id", ""), str) else ""
    try:
        selected_config = config or load_config(getattr(args, "config", None), probe_local_paths=False)
        git_target, external_target = _validate_request(args, selected_config)
    except (OSError, TypeError, ValueError) as exc:
        return ScanOutcome(4, scan_id, None, None, False, "NOT_STARTED", 0, 0, str(exc))
    try:
        journal = ScanProgressJournal.create(git_target, scan_id=args.scan_id)
    except (OSError, TypeError, ValueError) as exc:
        return ScanOutcome(
            4,
            args.scan_id,
            None,
            None,
            False,
            "NOT_STARTED",
            0,
            0,
            str(exc),
        )

    stage = "INITIALIZED"
    local_scopes: tuple[ScopeResult, ...] = ()
    git_records: tuple[GitOwnershipRecord, ...] = ()
    final_assets: tuple[AssetRecord, ...] = ()
    remote = _not_requested_remote()
    try:
        started_at_utc = clock()
        stage = "LOCAL"
        journal.begin_stage(stage)
        local_assets, local_scopes = LocalCollector(
            selected_config.local, selected_config.discovery, scan_id=args.scan_id
        ).collect()
        local_scopes = _require_verified_root_scope(
            local_scopes,
            location=Location.LOCAL,
            root_id=selected_config.local.root_id,
            scan_id=args.scan_id,
        )
        root_paths: dict[str, str | Path] = {
            selected_config.local.root_id: selected_config.local.root,
            selected_config.n607.root_id: selected_config.n607.root,
        }
        default_implementation_repository = Path(__file__).resolve().parents[2]
        selected_implementation_repository = Path(
            implementation_repository or default_implementation_repository
        ).resolve(strict=False)
        seeds = (
            tuple(repository_seeds)
            if repository_seeds is not None
            else (selected_implementation_repository,)
        )
        if not any(
            Path(seed).resolve(strict=False) == selected_implementation_repository for seed in seeds
        ):
            seeds = seeds + (selected_implementation_repository,)

        stage = "GIT"
        journal.begin_stage(stage)
        attached_assets, git_records, repositories = _ownership_and_assets(
            local_assets,
            (),
            repository_seeds=seeds,
            root_paths=root_paths,
            git_runner=git_runner,
        )

        stage = "N607"
        journal.begin_stage(stage)
        if bool(getattr(args, "include_n607", False)):
            if n607_collector_factory is None:
                remote = _remote_failure_from_exception(
                    RuntimeError("N607 collector was not supplied by the top-level entrypoint"),
                    scan_id=args.scan_id,
                    root_id=selected_config.n607.root_id,
                )
            else:
                try:
                    collector = _make_n607_collector(n607_collector_factory, selected_config, args.scan_id)
                    collected = collector.collect()
                    if not isinstance(collected, N607CollectionResult):
                        raise ValueError("N607 collector returned an invalid result")
                    remote = _convert_remote(collected, selected_config, args.scan_id)
                except Exception as exc:
                    remote = _remote_failure_from_exception(
                        exc,
                        scan_id=args.scan_id,
                        root_id=selected_config.n607.root_id,
                    )
        journal.record_n607_result(
            requested=bool(getattr(args, "include_n607", False)),
            outcome=remote.outcome,
            attempts=remote.attempts,
        )
        if remote.assets:
            attached_assets = attached_assets + tuple(
                replace(asset, git_ownership=GitOwnership.REMOTE_NON_GIT)
                for asset in remote.assets
            )
            git_records = git_records + tuple(
                GitOwnershipRecord(asset_id=asset.asset_id, ownership=GitOwnership.REMOTE_NON_GIT)
                for asset in remote.assets
            )

        stage = "INDEX"
        journal.begin_stage(stage)
        experiment_index = index_experiments(
            attached_assets,
            root_paths=root_paths,
            process_evidence=remote.processes,
        )
        indexed_assets = _assets_with_experiment_ids(attached_assets, experiment_index)

        stage = "RETENTION"
        journal.begin_stage(stage)
        evidence = _retention_evidence(indexed_assets, experiment_index, root_paths)
        decisions = classify_retentions(indexed_assets, evidence)
        candidates = build_deletion_candidates(indexed_assets, evidence)
        final_assets = _with_retention(indexed_assets, decisions)
        completed_at_utc = clock()
        head, tracked_state = _implementation_state(
            repositories,
            implementation_repository=selected_implementation_repository,
        )
        metadata = _metadata(
            selected_config,
            local_scopes=local_scopes,
            remote=remote,
            implementation_git_head=head,
            git_tracked_diff_state=tracked_state,
        )
        bundle = ScanBundle(
            scan_id=args.scan_id,
            operator=args.operator,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            assets=final_assets,
            scope_results=local_scopes + remote.scopes + remote.error_scopes,
            git_ownership=git_records,
            experiments=tuple(experiment_index.values()),
            retention_decisions=decisions,
            deletion_candidates=candidates,
        )

        stage = "EMISSION"
        journal.begin_stage(stage)
        emission = ReportEmitter(
            bundle,
            output_root=args.output_root,
            external_output_root=args.external_output_root,
            metadata=metadata,
            git_file_max_bytes=selected_config.output.git_file_max_bytes,
            git_scan_max_bytes=selected_config.output.git_scan_max_bytes,
            progress_journal=journal,
        ).emit()
        local_errors = _local_error_count(final_assets, local_scopes, git_records)
        return ScanOutcome(
            _outcome_code(local_errors, remote),
            args.scan_id,
            str(emission.git_output_dir),
            str(emission.external_output_dir) if emission.external_output_dir is not None else None,
            remote.outcome != "NOT_REQUESTED",
            remote.outcome,
            local_errors,
            remote.error_count,
            None,
            "COMPLETE",
            stage,
        )
    except KeyboardInterrupt:
        receipt_completed = journal.receipt_write_completed
        receipt_readback_unknown = journal.receipt_readback_unknown
        terminal_persisted = False
        if not receipt_completed and not receipt_readback_unknown:
            if journal.terminal_append_attempted:
                terminal_persisted = journal.terminal_append_persisted
            else:
                try:
                    terminal_persisted = journal.record_interrupt(stage)
                except BaseException:
                    # The original interrupt is the authoritative scan outcome even
                    # when durable-terminal journaling itself cannot be flushed.
                    terminal_persisted = False
        if receipt_completed:
            message = (
                "complete receipt exists; terminal receipt call was interrupted "
                f"during {stage}; journal was not modified"
            )
        elif receipt_readback_unknown:
            message = (
                "receipt readback UNKNOWN after terminal receipt call was interrupted "
                f"during {stage}; journal was not modified; inspect partial journal"
            )
        elif terminal_persisted:
            message = f"scan interrupted during {stage}; durable progress was preserved"
        else:
            message = (
                f"scan interrupted during {stage}; terminal progress could not be persisted; "
                "inspect partial journal"
            )
        return ScanOutcome(
            130,
            args.scan_id,
            str(git_target),
            str(external_target) if external_target.exists() else None,
            remote.outcome != "NOT_REQUESTED",
            remote.outcome,
            _local_error_count(final_assets, local_scopes, git_records),
            remote.error_count,
            message,
            "INTERRUPTED",
            stage,
        )
    except Exception as exc:
        receipt_completed = journal.receipt_write_completed
        receipt_readback_unknown = journal.receipt_readback_unknown
        terminal_persisted = False
        if not receipt_completed and not receipt_readback_unknown:
            if journal.terminal_append_attempted:
                terminal_persisted = journal.terminal_append_persisted
            else:
                try:
                    terminal_persisted = journal.record_failure(stage, exc)
                except BaseException:
                    # Preserve the scan failure rather than misreporting it as a
                    # pre-start CLI failure if the evidence append also fails.
                    terminal_persisted = False
        if receipt_completed:
            message = (
                "complete receipt exists; terminal receipt call raised "
                f"during {stage}: {type(exc).__name__}; journal was not modified"
            )
        elif receipt_readback_unknown:
            message = (
                "receipt readback UNKNOWN after terminal receipt call raised "
                f"during {stage}: {type(exc).__name__}; journal was not modified; "
                "inspect partial journal"
            )
        elif terminal_persisted:
            message = (
                f"report emission failed after scanning; durable progress was preserved during {stage}: {type(exc).__name__}"
                if stage == "EMISSION"
                else f"scan failed after initialization during {stage}; durable progress was preserved: {type(exc).__name__}"
            )
        else:
            message = (
                f"report emission failed after scanning; terminal progress could not be persisted; "
                f"inspect partial journal: {type(exc).__name__}"
                if stage == "EMISSION"
                else f"scan failed after initialization during {stage}; terminal progress could not be persisted; "
                f"inspect partial journal: {type(exc).__name__}"
            )
        return ScanOutcome(
            3
            if (
                receipt_readback_unknown
                or remote.outcome == RemoteOutcome.UNKNOWN.value
                or remote.disconnect == "UNKNOWN"
            )
            else 2,
            args.scan_id,
            str(git_target),
            str(external_target) if external_target.exists() else None,
            remote.outcome != "NOT_REQUESTED",
            remote.outcome,
            _local_error_count(final_assets, local_scopes, git_records),
            remote.error_count,
            message,
            "FAILED",
            stage,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    n607_collector_factory: Callable[..., object] | None = None,
) -> int:
    """CLI boundary for scans and bounded local governance queries."""

    args = parse_args(argv)
    if args.command != "scan":
        return _run_index_command(args)
    try:
        config = load_config(args.config, probe_local_paths=False)
        if args.print_plan:
            print_plan(args, config=config)
            return 0
        outcome = run_scan(
            args,
            config=config,
            n607_collector_factory=n607_collector_factory,
        )
    except KeyboardInterrupt:
        outcome = ScanOutcome(
            130,
            getattr(args, "scan_id", ""),
            None,
            None,
            False,
            "NOT_STARTED",
            0,
            0,
            "scan interrupted before durable progress initialization",
            "INTERRUPTED",
            "INITIALIZED",
        )
    except Exception as exc:
        outcome = ScanOutcome(4, getattr(args, "scan_id", ""), None, None, False, "NOT_STARTED", 0, 0, str(exc))
    print(json.dumps(outcome.as_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return outcome.exit_code


def _emit_index_payload(payload: Mapping[str, object], *, compact: bool) -> None:
    if compact:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return
    if "scan_id" in payload and "terminal_state" in payload:
        print(
            f"scan_id={payload['scan_id']} terminal_state={payload['terminal_state']} "
            f"warnings={payload.get('warning_count', 0)}"
        )
        return
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _run_index_command(args: argparse.Namespace) -> int:
    """Run one local-only index command before any scanner is constructed."""

    try:
        if args.command == "build-index":
            summary = build_index(
                receipt_path=Path(args.receipt),
                external_root=Path(args.external_root),
                database_path=Path(args.database),
            )
            payload: dict[str, object] = {
                "database_path": str(summary.database_path),
                "scan_id": summary.scan_id,
                "table_counts": dict(summary.table_counts),
            }
            exit_code = 0
        else:
            pointer = load_latest(Path(args.latest))
            store = QueryStore.open(pointer)
            try:
                if args.command == "status":
                    payload = store.status()
                    exit_code = 3 if payload["warning_count"] else 0
                elif args.command == "find":
                    payload = store.find_assets(args.query, limit=args.limit)
                    exit_code = 0
                elif args.command == "experiment":
                    payload = store.experiment(args.run_id)
                    exit_code = 0
                elif args.command == "repo":
                    payload = store.repo(args.path)
                    exit_code = 0
                elif args.command == "review":
                    filters = {
                        key: value
                        for key, value in {
                            "location": args.location,
                            "retention_class": args.retention_class,
                            "experiment_state": args.experiment_state,
                            "ownership": args.ownership,
                        }.items()
                        if value is not None
                    }
                    payload = store.review(filters, limit=args.limit)
                    exit_code = 0
                else:
                    raise ValueError(f"unsupported index command: {args.command}")
            finally:
                store.close()
    except FileExistsError as exc:
        payload = {"error": str(exc), "status": "REJECTED"}
        exit_code = 4
    except (OSError, ValueError) as exc:
        payload = {"error": str(exc), "status": "INDEX_UNAVAILABLE"}
        exit_code = 2
    _emit_index_payload(payload, compact=bool(args.json))
    return exit_code


__all__ = ["CLI_VERSION", "ScanOutcome", "build_parser", "main", "parse_args", "print_plan", "run_scan"]
