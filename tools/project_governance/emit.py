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
import re
import secrets
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    AccessStatus,
    ApprovalState,
    AssetKind,
    AssetRecord,
    DeletionCandidate,
    ExecutionState,
    ExperimentRecord,
    ExperimentState,
    GitOwnership,
    GitOwnershipRecord,
    HashStatus,
    Location,
    RetentionClass,
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
_PROGRESS_NAME = "scan_progress.ndjson"
_PROGRESS_SCHEMA_VERSION = 1
_PROGRESS_TOKEN_RE = re.compile(r"[0-9a-f]{48}\Z")
_PROGRESS_STAGES = ("LOCAL", "GIT", "N607", "INDEX", "RETENTION", "EMISSION")
_PROGRESS_TERMINAL_STATES = frozenset({"COMPLETE_PENDING_RECEIPT", "FAILED", "INTERRUPTED"})
_REQUESTED_N607_ROUTES = frozenset({"DIRECT", "LAB_BRIDGE", "NO_ROUTE"})
_REQUESTED_N607_PREFLIGHT_STATES = frozenset(
    {"DIRECT_READY", "DIRECT_PATH_UNAVAILABLE", "FAILED", "UNKNOWN"}
)
_REQUESTED_N607_DISCONNECT_STATES = frozenset({"VERIFIED", "UNKNOWN"})
_REQUESTED_N607_OUTCOMES = frozenset({"VERIFIED", "FAILED", "UNKNOWN"})
_N607_ATTEMPT_LABELS = frozenset({"PREFLIGHT", "DIRECT", "LAB_BRIDGE"})
_N607_CONNECTION_STATES = frozenset({"ESTABLISHED", "SYN_SENT"})
_N607_ENDPOINTS = frozenset({"172.31.111.215:22", "172.31.105.18:22"})
_MAX_N607_STDERR_TAIL_BYTES = 8192
_ROUTE_PREFLIGHT_STATE = {
    "DIRECT": "DIRECT_READY",
    "LAB_BRIDGE": "DIRECT_PATH_UNAVAILABLE",
}
_INVALID_WINDOWS_NAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_ALLOWED_SCOPE_STATUS = frozenset({"VERIFIED", "NOT_PRESENT", "SCAN_ERROR"})
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
        "n607_outcome",
        "n607_route",
        "n607_preflight",
        "n607_disconnect",
        "n607_attempts",
        "n607_active_training_observed",
        "n607_scan_error_count",
    }
)


class _ReceiptReadback(str, Enum):
    """The only safe conclusions after a receipt write raises."""

    MATCH = "MATCH"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EmissionResult:
    """Paths and receipt facts returned after a successful emission."""

    git_output_dir: Path
    external_output_dir: Path | None
    receipt: Mapping[str, Any]


@dataclass
class ScanProgressJournal:
    """Durable, append-only state for one non-overwriting scan target.

    The journal records only scan lifecycle facts.  It is intentionally not a
    command log, payload store, or asset mutation surface.  A terminal receipt
    closes the journal permanently; an interrupted process that never reaches
    a terminal record remains explicitly non-terminal/unknown on disk.
    """

    target: Path
    scan_id: str
    token: str
    windows_pid: int
    _last_stage: str = "INITIALIZED"
    _next_stage_index: int = 0
    _terminal_state: str | None = None
    _receipt_write_active: bool = False
    _receipt_write_completed: bool = False
    _expected_receipt_payload: bytes | None = None
    _receipt_readback: _ReceiptReadback | None = None
    _terminal_append_attempted: bool = False
    _terminal_append_persisted: bool = False

    @property
    def progress_path(self) -> Path:
        return self.target / _PROGRESS_NAME

    @property
    def receipt_path(self) -> Path:
        return self.target / _RECEIPT_NAME

    @property
    def current_stage(self) -> str:
        return self._last_stage

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def _require_token(value: object) -> str:
        if not isinstance(value, str) or _PROGRESS_TOKEN_RE.fullmatch(value) is None:
            raise ValueError("scan progress token must be a 48-character lowercase hexadecimal string")
        return value

    @classmethod
    def create(
        cls,
        target: str | os.PathLike[str],
        *,
        scan_id: str,
        token: str | None = None,
        windows_pid: int | None = None,
    ) -> "ScanProgressJournal":
        """Exclusively create the target and its first fsynced journal line."""

        _validate_scan_id(scan_id)
        selected_token = cls._require_token(token if token is not None else secrets.token_hex(24))
        selected_pid = os.getpid() if windows_pid is None else windows_pid
        if type(selected_pid) is not int or selected_pid <= 0:
            raise ValueError("scan progress Windows PID must be a positive integer")
        raw_target = Path(target).expanduser()
        if raw_target.is_symlink():
            raise ValueError("scan progress target must not be a symlink")
        resolved_target = raw_target.resolve(strict=False)
        if resolved_target.exists() or resolved_target.is_symlink():
            raise FileExistsError(f"governance output already exists: {resolved_target}")
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        resolved_target.mkdir()
        if resolved_target.is_symlink() or not resolved_target.is_dir():
            raise ValueError("scan progress target must be a real directory")
        journal = cls(resolved_target, scan_id, selected_token, selected_pid)
        journal._write_initial()
        return journal

    @classmethod
    def open_existing(
        cls,
        target: str | os.PathLike[str],
        *,
        scan_id: str,
        token: str,
    ) -> "ScanProgressJournal":
        """Attach only to a legal, token-owned precreated target."""

        _validate_scan_id(scan_id)
        raw_target = Path(target).expanduser()
        if raw_target.is_symlink():
            raise ValueError("scan progress target must not be a symlink")
        resolved_target = raw_target.resolve(strict=False)
        selected_token = cls._require_token(token)
        journal = cls(resolved_target, scan_id, selected_token, os.getpid())
        journal._restore_and_validate(require_receipt_absent=True)
        return journal

    def _base(self) -> dict[str, Any]:
        return {
            "schema_version": _PROGRESS_SCHEMA_VERSION,
            "scan_id": self.scan_id,
            "token": self.token,
            "windows_pid": self.windows_pid,
            "timestamp_utc": self._timestamp(),
        }

    @staticmethod
    def _write_and_sync(stream: Any, payload: bytes) -> None:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    def _write_initial(self) -> None:
        record = self._base()
        record.update({"event": "INITIALIZED", "stage": "INITIALIZED", "state": "INITIALIZED"})
        payload = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self.progress_path.open("xb") as stream:
            self._write_and_sync(stream, (payload + "\n").encode("utf-8"))

    def _append(self, record: Mapping[str, Any]) -> None:
        if self._receipt_write_completed or (
            self.receipt_path.exists() and not self._receipt_write_active
        ):
            raise RuntimeError("cannot append scan progress after scan_receipt.json exists")
        if self.target.is_symlink() or not self.target.is_dir():
            raise ValueError("scan progress target must remain a real directory")
        if self.progress_path.is_symlink() or not self.progress_path.is_file():
            raise ValueError("scan progress file must remain a real file")
        payload = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self.progress_path.open("ab") as stream:
            self._write_and_sync(stream, (payload + "\n").encode("utf-8"))

    def begin_stage(self, stage: str) -> None:
        if self._terminal_state is not None:
            raise RuntimeError("cannot start a stage after terminal scan progress")
        if self.receipt_path.exists():
            raise RuntimeError("cannot append scan progress after scan_receipt.json exists")
        if self._next_stage_index >= len(_PROGRESS_STAGES):
            raise ValueError("all scan progress stages already started")
        expected = _PROGRESS_STAGES[self._next_stage_index]
        if stage != expected:
            raise ValueError(f"scan progress stage must be {expected}, got {stage}")
        record = self._base()
        record.update({"event": "STAGE", "stage": stage, "state": "STARTED"})
        self._append(record)
        self._last_stage = stage
        self._next_stage_index += 1

    def ensure_emission_stage(self) -> None:
        """Advance a standalone emitter over already-supplied collection facts."""

        if self._terminal_state is not None:
            raise RuntimeError("cannot emit after terminal scan progress")
        if self._last_stage == "EMISSION":
            return
        while self._next_stage_index < len(_PROGRESS_STAGES):
            self.begin_stage(_PROGRESS_STAGES[self._next_stage_index])

    @staticmethod
    def _attempt_liveness(
        *, timed_out: bool, child_exited: bool, proxy_children_exited: bool
    ) -> str:
        if child_exited and proxy_children_exited:
            return "EXITED"
        if timed_out:
            return "LIVE_CHILD_UNKNOWN"
        return "UNKNOWN"

    @staticmethod
    def _normalize_attempts(attempts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise ValueError("scan progress N607 attempt must be a mapping")
            label = attempt.get("label")
            child_pid = attempt.get("child_pid")
            proxy_pids = attempt.get("proxy_child_pids")
            timed_out = attempt.get("timed_out")
            child_exited = attempt.get("child_exited")
            proxy_children_exited = attempt.get("proxy_children_exited")
            if not isinstance(label, str) or not label:
                raise ValueError("scan progress N607 attempt label is invalid")
            if child_pid is not None and (type(child_pid) is not int or child_pid <= 0):
                raise ValueError("scan progress N607 child PID is invalid")
            if not isinstance(proxy_pids, (tuple, list)) or any(
                type(pid) is not int or pid <= 0 for pid in proxy_pids
            ):
                raise ValueError("scan progress N607 proxy PIDs are invalid")
            if (
                type(timed_out) is not bool
                or type(child_exited) is not bool
                or type(proxy_children_exited) is not bool
            ):
                raise ValueError("scan progress N607 timeout/exit evidence is invalid")
            normalized.append(
                {
                    "label": label,
                    "child_pid": child_pid,
                    "proxy_child_pids": list(proxy_pids),
                    "timed_out": timed_out,
                    "child_exited": child_exited,
                    "proxy_children_exited": proxy_children_exited,
                    "liveness": ScanProgressJournal._attempt_liveness(
                        timed_out=timed_out,
                        child_exited=child_exited,
                        proxy_children_exited=proxy_children_exited,
                    ),
                }
            )
        return normalized

    @staticmethod
    def _validate_serialized_attempts(attempts: Any) -> None:
        """Validate the deliberately reduced N607 evidence stored in the journal.

        The journal persists the child and proxy exit facts alongside the
        derived liveness conclusion.  Reopening therefore validates both the
        independent evidence and the conclusion that follows from it.
        """

        if not isinstance(attempts, list):
            raise ValueError("scan progress N607 attempts must be a list")
        expected_keys = {
            "label",
            "child_pid",
            "proxy_child_pids",
            "timed_out",
            "child_exited",
            "proxy_children_exited",
            "liveness",
        }
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or set(attempt) != expected_keys:
                raise ValueError("scan progress N607 attempt is invalid")
            child_pid = attempt["child_pid"]
            proxy_pids = attempt["proxy_child_pids"]
            if (
                not isinstance(attempt["label"], str)
                or not attempt["label"]
                or (child_pid is not None and (type(child_pid) is not int or child_pid <= 0))
                or not isinstance(proxy_pids, list)
                or any(type(pid) is not int or pid <= 0 for pid in proxy_pids)
                or type(attempt["timed_out"]) is not bool
                or type(attempt["child_exited"]) is not bool
                or type(attempt["proxy_children_exited"]) is not bool
            ):
                raise ValueError("scan progress N607 attempt is invalid")
            expected_liveness = ScanProgressJournal._attempt_liveness(
                timed_out=attempt["timed_out"],
                child_exited=attempt["child_exited"],
                proxy_children_exited=attempt["proxy_children_exited"],
            )
            if attempt["liveness"] != expected_liveness:
                raise ValueError("scan progress N607 liveness does not match exit evidence")

    def record_n607_result(
        self,
        *,
        requested: bool,
        outcome: str,
        attempts: Iterable[Mapping[str, Any]],
    ) -> None:
        if self._last_stage != "N607" or self._terminal_state is not None:
            raise RuntimeError("N607 progress can only be recorded during the N607 stage")
        if not isinstance(requested, bool) or not isinstance(outcome, str) or not outcome:
            raise ValueError("scan progress N607 outcome is invalid")
        record = self._base()
        record.update(
            {
                "event": "N607_RESULT",
                "stage": "N607",
                "state": "COLLECTED",
                "n607_requested": requested,
                "n607_outcome": outcome,
                "attempts": self._normalize_attempts(attempts),
            }
        )
        self._append(record)

    def freeze_for_receipt(self) -> None:
        if self._last_stage != "EMISSION" or self._terminal_state is not None:
            raise RuntimeError("only an active EMISSION stage can freeze progress for receipt")
        record = self._base()
        record.update(
            {
                "event": "TERMINAL",
                "stage": "EMISSION",
                "terminal_state": "COMPLETE_PENDING_RECEIPT",
            }
        )
        self._append(record)
        self._terminal_state = "COMPLETE_PENDING_RECEIPT"

    @property
    def receipt_write_completed(self) -> bool:
        return self._receipt_write_completed

    @property
    def receipt_readback_unknown(self) -> bool:
        return self._receipt_readback is _ReceiptReadback.UNKNOWN

    @property
    def terminal_append_attempted(self) -> bool:
        return self._terminal_append_attempted

    @property
    def terminal_append_persisted(self) -> bool:
        return self._terminal_append_persisted

    def begin_receipt_write(self, expected_payload: bytes) -> None:
        """Allow one explicitly owned receipt write after the progress freeze."""

        if self._last_stage != "EMISSION" or self._terminal_state != "COMPLETE_PENDING_RECEIPT":
            raise RuntimeError("only frozen EMISSION progress can start receipt writing")
        if self._receipt_write_active or self._receipt_write_completed:
            raise RuntimeError("scan receipt writing has already started")
        if not isinstance(expected_payload, bytes):
            raise TypeError("expected scan receipt payload must be bytes")
        self._expected_receipt_payload = expected_payload
        self._receipt_write_active = True
        self._receipt_readback = None

    def _receipt_readback_state(self) -> _ReceiptReadback:
        """Read a just-written receipt without treating I/O uncertainty as partial."""

        if not self._receipt_write_active or self._expected_receipt_payload is None:
            return _ReceiptReadback.PARTIAL
        try:
            self.receipt_path.stat()
        except FileNotFoundError:
            return _ReceiptReadback.PARTIAL
        except OSError:
            return _ReceiptReadback.UNKNOWN
        try:
            with self.receipt_path.open("rb") as stream:
                payload = stream.read()
        except OSError:
            return _ReceiptReadback.UNKNOWN
        return (
            _ReceiptReadback.MATCH
            if payload == self._expected_receipt_payload
            else _ReceiptReadback.PARTIAL
        )

    def receipt_matches_expected_payload(self) -> bool:
        return self.reconcile_receipt_write() is _ReceiptReadback.MATCH

    def reconcile_receipt_write(self) -> _ReceiptReadback:
        """Freeze the outcome of the one receipt readback attempt.

        A failed readback can never be safely retried as a failed journal
        append: a complete receipt might exist but be temporarily unreadable.
        """

        if self._receipt_write_completed:
            return _ReceiptReadback.MATCH
        if self._receipt_readback is _ReceiptReadback.UNKNOWN:
            return _ReceiptReadback.UNKNOWN
        state = self._receipt_readback_state()
        self._receipt_readback = state
        if state is _ReceiptReadback.MATCH:
            self._receipt_write_active = False
            self._receipt_write_completed = True
        return state

    def mark_receipt_readback_unknown(self) -> None:
        """Conservatively quarantine a receipt when its verification raises."""

        if self._receipt_write_active and not self._receipt_write_completed:
            self._receipt_readback = _ReceiptReadback.UNKNOWN

    def mark_receipt_written(self) -> None:
        if self.reconcile_receipt_write() is not _ReceiptReadback.MATCH:
            raise RuntimeError("scan receipt bytes do not match the expected final payload")

    def record_failure(self, stage: str, error: BaseException) -> bool:
        return self._record_terminal(stage, "FAILED", error)

    def record_interrupt(self, stage: str) -> bool:
        return self._record_terminal(stage, "INTERRUPTED", KeyboardInterrupt())

    def _record_terminal(self, stage: str, terminal_state: str, error: BaseException) -> bool:
        if self._terminal_state in {"FAILED", "INTERRUPTED"}:
            return True
        if self._terminal_append_attempted:
            return self._terminal_append_persisted
        if self.receipt_readback_unknown:
            return False
        if self._receipt_write_completed or (
            self.receipt_path.exists() and not self._receipt_write_active
        ):
            raise RuntimeError("cannot append scan progress after scan_receipt.json exists")
        if terminal_state not in {"FAILED", "INTERRUPTED"}:
            raise ValueError("scan progress terminal state is invalid")
        if stage not in {"INITIALIZED", *_PROGRESS_STAGES}:
            raise ValueError("scan progress failure stage is invalid")
        record = self._base()
        record.update(
            {
                "event": "TERMINAL",
                "stage": stage,
                "terminal_state": terminal_state,
                "error_type": type(error).__name__,
            }
        )
        # This in-memory latch survives Emitter -> CLI exception handoff and
        # prevents a second terminal append after an ambiguous write failure.
        self._terminal_append_attempted = True
        try:
            self._append(record)
        except BaseException:
            if self._terminal_record_landed(record):
                self._terminal_state = terminal_state
                self._terminal_append_persisted = True
            return self._terminal_append_persisted
        self._terminal_state = terminal_state
        self._terminal_append_persisted = True
        return True

    def _terminal_record_landed(self, expected: Mapping[str, Any]) -> bool:
        """Confirm a complete terminal line only; unreadable evidence is unknown."""

        try:
            raw = self.progress_path.read_bytes()
            if not raw.endswith(b"\n"):
                return False
            lines = raw.decode("utf-8").splitlines()
            if not lines:
                return False
            landed = json.loads(lines[-1])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(landed, Mapping) and dict(landed) == dict(expected)

    def _restore_and_validate(self, *, require_receipt_absent: bool) -> None:
        if self.target.is_symlink() or not self.target.is_dir():
            raise ValueError("scan progress target must be a real directory, not a symlink")
        if require_receipt_absent and self.receipt_path.exists():
            raise FileExistsError("scan receipt already exists for the requested progress target")
        entries = tuple(self.target.iterdir())
        if {entry.name for entry in entries} != {_PROGRESS_NAME}:
            raise ValueError("scan progress target contains unexpected files")
        if self.progress_path.is_symlink() or not self.progress_path.is_file():
            raise ValueError("scan progress file must be a real file, not a symlink")
        try:
            raw = self.progress_path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("scan progress must be strict UTF-8") from exc
        if not text.endswith("\n"):
            raise ValueError("scan progress must end with an fsynced NDJSON line")
        try:
            records = [json.loads(line) for line in text.splitlines() if line]
        except json.JSONDecodeError as exc:
            raise ValueError("scan progress contains invalid NDJSON") from exc
        if not records:
            raise ValueError("scan progress is empty")
        initial = records[0]
        if not isinstance(initial, Mapping) or initial.get("event") != "INITIALIZED":
            raise ValueError("scan progress initial record is invalid")
        required_initial = {
            "schema_version",
            "scan_id",
            "token",
            "windows_pid",
            "timestamp_utc",
            "event",
            "stage",
            "state",
        }
        if set(initial) != required_initial:
            raise ValueError("scan progress initial record fields are invalid")
        if (
            initial["schema_version"] != _PROGRESS_SCHEMA_VERSION
            or initial["scan_id"] != self.scan_id
            or initial["token"] != self.token
            or initial["stage"] != "INITIALIZED"
            or initial["state"] != "INITIALIZED"
            or type(initial["windows_pid"]) is not int
            or initial["windows_pid"] <= 0
        ):
            if initial.get("token") != self.token:
                raise ValueError("scan progress token does not own this target")
            raise ValueError("scan progress initial record does not match the requested scan")
        self.windows_pid = initial["windows_pid"]
        expected_index = 0
        last_stage = "INITIALIZED"
        terminal_state: str | None = None
        n607_result_seen = False
        for record in records[1:]:
            if not isinstance(record, Mapping):
                raise ValueError("scan progress record is not an object")
            for field in ("schema_version", "scan_id", "token", "windows_pid"):
                if record.get(field) != initial.get(field):
                    raise ValueError("scan progress record does not match its initial identity")
            if not isinstance(record.get("timestamp_utc"), str) or not record["timestamp_utc"]:
                raise ValueError("scan progress record timestamp is invalid")
            event = record["event"]
            if terminal_state in {"FAILED", "INTERRUPTED"}:
                raise ValueError("scan progress contains records after a terminal state")
            if terminal_state == "COMPLETE_PENDING_RECEIPT" and event != "TERMINAL":
                raise ValueError("scan progress contains records after a frozen receipt state")
            if event == "STAGE":
                if set(record) != {
                    "schema_version", "scan_id", "token", "windows_pid", "timestamp_utc", "event", "stage", "state"
                }:
                    raise ValueError("scan progress stage record fields are invalid")
                if expected_index >= len(_PROGRESS_STAGES) or record["stage"] != _PROGRESS_STAGES[expected_index] or record.get("state") != "STARTED":
                    raise ValueError("scan progress stage order is invalid")
                last_stage = record["stage"]
                expected_index += 1
            elif event == "N607_RESULT":
                if (
                    set(record)
                    != {
                        "schema_version", "scan_id", "token", "windows_pid", "timestamp_utc", "event", "stage", "state", "n607_requested", "n607_outcome", "attempts"
                    }
                    or last_stage != "N607"
                    or record.get("stage") != "N607"
                    or record.get("state") != "COLLECTED"
                ):
                    raise ValueError("scan progress N607 record is invalid")
                if n607_result_seen:
                    raise ValueError("scan progress contains more than one N607 result")
                self._validate_serialized_attempts(record["attempts"])
                n607_result_seen = True
            elif event == "TERMINAL":
                expected_keys = {
                    "schema_version", "scan_id", "token", "windows_pid", "timestamp_utc", "event", "stage", "terminal_state"
                }
                if record.get("terminal_state") in {"FAILED", "INTERRUPTED"}:
                    expected_keys.add("error_type")
                if set(record) != expected_keys or record.get("terminal_state") not in _PROGRESS_TERMINAL_STATES:
                    raise ValueError("scan progress terminal record is invalid")
                record_terminal_state = record["terminal_state"]
                if record_terminal_state == "COMPLETE_PENDING_RECEIPT":
                    if terminal_state is not None or last_stage != "EMISSION" or record.get("stage") != "EMISSION":
                        raise ValueError("scan progress receipt freeze is not in EMISSION")
                elif record.get("stage") != last_stage:
                    raise ValueError("scan progress terminal state does not match the active stage")
                if terminal_state == "COMPLETE_PENDING_RECEIPT" and record_terminal_state not in {
                    "FAILED",
                    "INTERRUPTED",
                }:
                    raise ValueError("frozen receipt progress can only close as failed or interrupted")
                terminal_state = record_terminal_state
                last_stage = record["stage"]
            else:
                raise ValueError("scan progress event is invalid")
        self._last_stage = last_stage
        self._next_stage_index = expected_index
        self._terminal_state = terminal_state
        self._terminal_append_attempted = terminal_state in {"FAILED", "INTERRUPTED"}
        self._terminal_append_persisted = self._terminal_append_attempted

    def validate_for_emission(self, target: str | os.PathLike[str]) -> None:
        raw_target = Path(target).expanduser()
        if raw_target.is_symlink() or raw_target.resolve(strict=False) != self.target.resolve(strict=False):
            raise ValueError("scan progress target does not match the emitter target")
        self._restore_and_validate(require_receipt_absent=True)
        if self._terminal_state is not None:
            raise RuntimeError("cannot emit after terminal scan progress")


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


def _csv_cell(value: Any) -> str:
    """Render one spreadsheet-safe cell without changing JSON evidence."""

    converted = _json_value(value)
    rendered = _cell(value)
    if isinstance(converted, str) and rendered.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + rendered
    return rendered


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
        writer.writerow([_csv_cell(getattr(record, header)) for header in headers])
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


def _compact_json_bytes(payload: Any) -> bytes:
    """Return the same canonical JSON value with deterministic compact whitespace."""

    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
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


def _validate_n607_attempts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("receipt metadata n607_attempts must be an explicit sequence")
    required = {
        "label",
        "child_pid",
        "proxy_child_pids",
        "returncode",
        "timed_out",
        "child_exited",
        "proxy_children_exited",
        "disconnect_status",
        "lingering_connections",
        "stderr_tail",
    }
    normalized: list[dict[str, Any]] = []
    for attempt in value:
        if not isinstance(attempt, Mapping) or set(attempt) != required:
            raise ValueError("receipt metadata N607 attempt fields are invalid")
        label = attempt["label"]
        if label not in _N607_ATTEMPT_LABELS:
            raise ValueError("receipt metadata N607 attempt label is invalid")
        child_pid = attempt["child_pid"]
        if child_pid is not None and (type(child_pid) is not int or child_pid <= 0):
            raise ValueError("receipt metadata N607 child PID is invalid")
        proxy_pids = attempt["proxy_child_pids"]
        if not isinstance(proxy_pids, (tuple, list)) or any(
            type(pid) is not int or pid <= 0 for pid in proxy_pids
        ):
            raise ValueError("receipt metadata N607 proxy PIDs are invalid")
        if len(set(proxy_pids)) != len(proxy_pids):
            raise ValueError("receipt metadata N607 proxy PIDs are not unique")
        returncode = attempt["returncode"]
        if returncode is not None and type(returncode) is not int:
            raise ValueError("receipt metadata N607 return code is invalid")
        for field in ("timed_out", "child_exited", "proxy_children_exited"):
            if type(attempt[field]) is not bool:
                raise ValueError(f"receipt metadata N607 {field} is invalid")
        disconnect_status = attempt["disconnect_status"]
        if disconnect_status not in _REQUESTED_N607_DISCONNECT_STATES:
            raise ValueError("receipt metadata N607 attempt disconnect state is invalid")
        stderr_tail = attempt["stderr_tail"]
        if not isinstance(stderr_tail, str) or len(stderr_tail.encode("utf-8")) > _MAX_N607_STDERR_TAIL_BYTES:
            raise ValueError("receipt metadata N607 stderr tail is invalid")
        raw_connections = attempt["lingering_connections"]
        if not isinstance(raw_connections, (tuple, list)):
            raise ValueError("receipt metadata N607 lingering connections are invalid")
        attempt_pids = set(proxy_pids)
        if child_pid is not None:
            attempt_pids.add(child_pid)
        connections: list[dict[str, Any]] = []
        for connection in raw_connections:
            if not isinstance(connection, Mapping) or set(connection) != {"pid", "endpoint", "state"}:
                raise ValueError("receipt metadata N607 connection fields are invalid")
            pid = connection["pid"]
            endpoint = connection["endpoint"]
            state = connection["state"]
            if type(pid) is not int or pid <= 0 or pid not in attempt_pids:
                raise ValueError("receipt metadata N607 connection PID is invalid")
            if endpoint not in _N607_ENDPOINTS or state not in _N607_CONNECTION_STATES:
                raise ValueError("receipt metadata N607 connection endpoint or state is invalid")
            connections.append({"pid": pid, "endpoint": endpoint, "state": state})
        normalized.append(
            {
                "label": label,
                "child_pid": child_pid,
                "proxy_child_pids": tuple(proxy_pids),
                "returncode": returncode,
                "timed_out": attempt["timed_out"],
                "child_exited": attempt["child_exited"],
                "proxy_children_exited": attempt["proxy_children_exited"],
                "disconnect_status": disconnect_status,
                "lingering_connections": tuple(connections),
                "stderr_tail": stderr_tail,
            }
        )
    labels = tuple(attempt["label"] for attempt in normalized)
    if len(set(labels)) != len(labels):
        raise ValueError("receipt metadata N607 attempt labels are not unique")
    return tuple(normalized)


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
        "n607_outcome",
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
    if type(values["n607_active_training_observed"]) is not bool:
        raise ValueError("receipt metadata n607_active_training_observed must be boolean")
    values["n607_attempts"] = _validate_n607_attempts(values["n607_attempts"])
    error_count = values["n607_scan_error_count"]
    if type(error_count) is not int or error_count < 0:
        raise ValueError("n607_scan_error_count must be a non-negative integer")
    n607_states = (
        values["n607_outcome"],
        values["n607_route"],
        values["n607_preflight"],
        values["n607_disconnect"],
    )
    if values["n607_requested"]:
        outcome, route, preflight, disconnect = n607_states
        if outcome not in _REQUESTED_N607_OUTCOMES:
            raise ValueError("requested N607 receipt metadata has an uncontrolled outcome")
        if route not in _REQUESTED_N607_ROUTES:
            raise ValueError("requested N607 receipt metadata has an uncontrolled route state")
        if (
            preflight not in _REQUESTED_N607_PREFLIGHT_STATES
            or disconnect not in _REQUESTED_N607_DISCONNECT_STATES
        ):
            raise ValueError("requested N607 receipt metadata must carry controlled observed states")
        expected_preflight = _ROUTE_PREFLIGHT_STATE.get(route)
        if expected_preflight is not None and preflight != expected_preflight:
            raise ValueError("requested N607 route and preflight states are inconsistent")
        labels = tuple(attempt["label"] for attempt in values["n607_attempts"])
        expected_labels = {
            "DIRECT": ("PREFLIGHT", "DIRECT"),
            "LAB_BRIDGE": ("PREFLIGHT", "LAB_BRIDGE"),
        }.get(route)
        if expected_labels is not None and labels != expected_labels:
            raise ValueError("requested N607 attempts do not match the selected route")
        if route == "NO_ROUTE" and labels not in {(), ("PREFLIGHT",)}:
            raise ValueError("requested N607 attempts do not match the no-route state")
        if outcome == "VERIFIED":
            if route == "NO_ROUTE" or disconnect != "VERIFIED":
                raise ValueError("verified N607 metadata has incomplete terminal states")
            if any(
                attempt["child_pid"] is None
                or attempt["returncode"] != 0
                or attempt["timed_out"]
                or not attempt["child_exited"]
                or not attempt["proxy_children_exited"]
                or attempt["disconnect_status"] != "VERIFIED"
                or attempt["lingering_connections"]
                for attempt in values["n607_attempts"]
            ):
                raise ValueError("verified N607 metadata has incomplete attempt evidence")
            if any(
                attempt["label"] in {"PREFLIGHT", "LAB_BRIDGE"}
                and not attempt["proxy_child_pids"]
                for attempt in values["n607_attempts"]
            ):
                raise ValueError("verified N607 metadata lacks required proxy exit evidence")
        elif values["n607_active_training_observed"]:
            raise ValueError("non-verified N607 metadata cannot claim active training evidence")
    elif (
        n607_states
        != ("NOT_REQUESTED", "NOT_REQUESTED", "NOT_REQUESTED", "NOT_REQUESTED")
        or error_count != 0
        or values["n607_attempts"]
        or values["n607_active_training_observed"]
    ):
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


def _validate_scan_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("scan_id must be a trimmed non-empty string")
    windows_path = PureWindowsPath(value)
    reserved_stem = value.rstrip(" .").split(".", 1)[0].upper()
    if (
        value in {".", ".."}
        or windows_path.drive
        or windows_path.root
        or windows_path.anchor
        or windows_path.is_reserved()
        or len(windows_path.parts) != 1
        or any(
            character in _INVALID_WINDOWS_NAME_CHARS or ord(character) < 32
            for character in value
        )
        or value.endswith((" ", "."))
        or reserved_stem in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError("scan_id must be one safe path component without a drive or anchor")
    return value


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_is_within(left, right) or _path_is_within(right, left)


_RECORD_TYPES = {
    "assets": AssetRecord,
    "scope_results": ScopeResult,
    "git_ownership": GitOwnershipRecord,
    "experiments": ExperimentRecord,
    "retention_decisions": RetentionDecision,
    "deletion_candidates": DeletionCandidate,
}
_RECORD_ENUM_FIELDS = {
    "assets": (
        ("location", Location),
        ("asset_kind", AssetKind),
        ("access_status", AccessStatus),
        ("hash_status", HashStatus),
    ),
    "scope_results": (("location", Location),),
    "git_ownership": (("ownership", GitOwnership),),
    "experiments": (("experiment_state", ExperimentState),),
    "retention_decisions": (("retention_class", RetentionClass),),
    "deletion_candidates": (("location", Location), ("asset_kind", AssetKind)),
}


def _validate_bundle_records(bundle: ScanBundle) -> dict[str, tuple[Any, ...]]:
    records: dict[str, tuple[Any, ...]] = {}
    for collection, record_type in _RECORD_TYPES.items():
        raw_records = getattr(bundle, collection)
        try:
            selected = tuple(raw_records or ())
        except TypeError as exc:
            raise ValueError(f"{collection} must be an iterable of {record_type.__name__}") from exc
        for index, record in enumerate(selected):
            if not isinstance(record, record_type):
                raise ValueError(
                    f"{collection}[{index}] must be a {record_type.__name__} record"
                )
            for field_name, enum_type in _RECORD_ENUM_FIELDS[collection]:
                if not isinstance(getattr(record, field_name), enum_type):
                    raise ValueError(
                        f"{collection}[{index}].{field_name} must be {enum_type.__name__}"
                    )
            if isinstance(record, AssetRecord):
                for field_name, enum_type in (
                    ("git_ownership", GitOwnership),
                    ("retention_class", RetentionClass),
                ):
                    field_value = getattr(record, field_name)
                    if field_value is not None and not isinstance(field_value, enum_type):
                        raise ValueError(
                            f"{collection}[{index}].{field_name} must be {enum_type.__name__}"
                        )
            if isinstance(record, ScopeResult) and (
                type(record.status) is not str or record.status not in _ALLOWED_SCOPE_STATUS
            ):
                raise ValueError(
                    f"{collection}[{index}].status must be a canonical scope state"
                )
            if (
                isinstance(record, (AssetRecord, ScopeResult))
                and record.scan_id != bundle.scan_id
            ):
                raise ValueError(
                    f"{collection}[{index}].scan_id does not match bundle scan_id"
                )
        records[collection] = selected
    return records


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
        progress_journal: ScanProgressJournal | None = None,
    ) -> None:
        if not isinstance(bundle, ScanBundle):
            raise TypeError("bundle must be a ScanBundle")
        _validate_scan_id(bundle.scan_id)
        if (
            type(git_file_max_bytes) is not int
            or type(git_scan_max_bytes) is not int
            or git_file_max_bytes <= 0
            or git_scan_max_bytes <= 0
        ):
            raise ValueError("output thresholds must be positive")
        self.bundle = bundle
        self.output_root = Path(output_root).expanduser().resolve(strict=False)
        self.external_output_root = Path(external_output_root).expanduser().resolve(strict=False)
        if _paths_overlap(self.output_root, self.external_output_root):
            raise ValueError("Git and external output roots must not overlap")
        self.metadata = _validate_metadata(metadata)
        self._validated_records = _validate_bundle_records(bundle)
        _validate_deletion_candidates(self._validated_records["deletion_candidates"])
        for timestamp in (bundle.started_at_utc, bundle.completed_at_utc):
            if timestamp is not None:
                _iso_utc(timestamp)
        self.git_file_max_bytes = int(git_file_max_bytes)
        self.git_scan_max_bytes = int(git_scan_max_bytes)
        self._emission_now_utc: str | None = None
        if progress_journal is not None and not isinstance(progress_journal, ScanProgressJournal):
            raise TypeError("progress_journal must be a ScanProgressJournal or None")
        if progress_journal is not None and progress_journal.scan_id != self.scan_id:
            raise ValueError("progress journal scan_id does not match the bundle")
        self._progress_journal = progress_journal

    @property
    def scan_id(self) -> str:
        return self.bundle.scan_id

    def _timestamp(self, value: str | None) -> str:
        if value:
            return _iso_utc(value)
        if self._emission_now_utc is None:
            self._emission_now_utc = _iso_utc(None)
        return self._emission_now_utc

    @property
    def progress_journal(self) -> ScanProgressJournal:
        if self._progress_journal is None:
            raise RuntimeError("scan progress journal is not initialized")
        return self._progress_journal

    def _target_paths(self, *, allow_owned_progress: bool = False) -> tuple[Path, Path]:
        git_target = (self.output_root / self.scan_id).resolve(strict=False)
        external_target = (self.external_output_root / self.scan_id).resolve(strict=False)
        if (
            git_target.parent != self.output_root
            or external_target.parent != self.external_output_root
        ):
            raise ValueError("scan_id resolved outside an output root")
        if git_target.exists():
            if not allow_owned_progress or self._progress_journal is None:
                raise FileExistsError(f"governance output already exists: {git_target}")
            self._progress_journal.validate_for_emission(git_target)
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
        return dict(self._validated_records)

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
        n607_outcome = metadata["n607_outcome"]
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
            "outcome": "\u7ed3\u679c",
            "route": "\u8def\u7531",
            "preflight": "\u9884\u68c0",
            "disconnect": "\u65ad\u8fde",
            "attempts": "\u5c1d\u8bd5\u8bb0\u5f55\u6570",
            "active_training": "\u89c2\u6d4b\u5230\u6d3b\u52a8\u8bad\u7ec3",
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
        lines.extend(
            [
                "",
                f"## {zh['gaps']}",
                "",
                f"- {zh['nonverified']}\uff1a`{'; '.join(gaps) if gaps else zh['none']}`",
                f"- {zh['experiment_gaps']}\uff1a`{'; '.join(experiment_gaps) if experiment_gaps else zh['none']}`",
                f"- {zh['asset_error']}\uff1a`{sum(asset.access_status is AccessStatus.SCAN_ERROR for asset in assets)}`",
                f"- {zh['experiment_error']}\uff1a`{sum(experiment.experiment_state is ExperimentState.SCAN_ERROR for experiment in experiments)}`",
                f"- {zh['n607_record_error']}\uff1a`{int(metadata.get('n607_scan_error_count', 0))}`",
                "",
                f"## {zh['remote']}",
                "",
                f"- {zh['outcome']}\uff1a`{n607_outcome}`",
                f"- {zh['route']}\uff1a`{n607_route}`",
                f"- {zh['preflight']}\uff1a`{n607_preflight}`",
                f"- {zh['disconnect']}\uff1a`{n607_disconnect}`",
                f"- {zh['attempts']}\uff1a`{len(metadata['n607_attempts'])}`",
                f"- {zh['active_training']}\uff1a`{str(metadata['n607_active_training_observed']).lower()}`",
                "",
                f"## {zh['boundary']}",
                "",
                zh["zero"],
                "",
            ]
        )
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
                "outcome": metadata["n607_outcome"],
                "route": metadata["n607_route"],
                "preflight": metadata["n607_preflight"],
                "disconnect": metadata["n607_disconnect"],
                "active_training_observed": metadata["n607_active_training_observed"],
                "attempts": _json_value(metadata["n607_attempts"]),
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

    def _ensure_git_file_limit(self, git_target: Path) -> None:
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

    def _emit_into_initialized_target(
        self,
        git_target: Path,
        external_target: Path,
        journal: ScanProgressJournal,
    ) -> EmissionResult:
        """Emit artifacts after the exclusive journal target has been created."""

        self._emission_now_utc = _iso_utc(None)
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

        self._ensure_git_file_limit(git_target)
        journal.freeze_for_receipt()
        self._ensure_git_file_limit(git_target)
        receipt = self._receipt_base(records)
        receipt["terminal_state"] = "COMPLETE"
        receipt["artifact_route"] = "EXTERNAL_COMPLETE_WITH_GIT_SHARDS" if oversize else "GIT_COMPLETE"
        receipt["files"] = [
            _file_entry(git_target / name, relative_to=git_target)
            for name in sorted(path.name for path in git_target.iterdir())
            if name != _PROGRESS_NAME
        ]
        # Its fixed filename is part of the journal protocol.  Keep the
        # receipt evidence explicit without duplicating it in the potentially
        # large report-file manifest.
        progress_entry = _file_entry(journal.progress_path, relative_to=git_target)
        receipt["progress"] = {
            "bytes": progress_entry["bytes"],
            "sha256": progress_entry["sha256"],
        }
        receipt["external_files"] = sorted(external_entries, key=lambda item: item["path"])
        receipt_payload = _json_bytes(receipt)
        if len(receipt_payload) > self.git_file_max_bytes:
            receipt_payload = _compact_json_bytes(receipt)
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
        journal.begin_receipt_write(receipt_payload)
        self._write_exclusive(git_target / _RECEIPT_NAME, receipt_payload, encoding="utf-8")
        journal.mark_receipt_written()
        return EmissionResult(git_target, external_output, receipt)

    def emit(self) -> EmissionResult:
        """Emit a fresh report and make its receipt the last written artifact."""

        journal: ScanProgressJournal | None = None
        try:
            if self._progress_journal is None:
                git_target, external_target = self._target_paths()
                self.output_root.mkdir(parents=True, exist_ok=True)
                journal = ScanProgressJournal.create(git_target, scan_id=self.scan_id)
                self._progress_journal = journal
            else:
                git_target, external_target = self._target_paths(allow_owned_progress=True)
                journal = self._progress_journal
            journal.ensure_emission_stage()
            return self._emit_into_initialized_target(git_target, external_target, journal)
        except KeyboardInterrupt:
            if journal is not None:
                try:
                    receipt_state = journal.reconcile_receipt_write()
                except BaseException:
                    journal.mark_receipt_readback_unknown()
                    receipt_state = _ReceiptReadback.UNKNOWN
                if receipt_state is _ReceiptReadback.PARTIAL:
                    try:
                        journal.record_interrupt("EMISSION")
                    except BaseException:
                        # A journal append must never replace the original
                        # interrupt that occurred during emission.
                        pass
            raise
        except Exception as exc:
            if journal is not None:
                try:
                    receipt_state = journal.reconcile_receipt_write()
                except BaseException:
                    journal.mark_receipt_readback_unknown()
                    receipt_state = _ReceiptReadback.UNKNOWN
                if receipt_state is _ReceiptReadback.PARTIAL:
                    try:
                        journal.record_failure("EMISSION", exc)
                    except BaseException:
                        # Preserve the collection/emission error when its
                        # best-effort terminal record cannot be persisted.
                        pass
            raise


__all__ = [
    "DEFAULT_GIT_FILE_MAX_BYTES",
    "DEFAULT_GIT_SCAN_MAX_BYTES",
    "EmissionResult",
    "ReportEmitter",
    "ScanProgressJournal",
]
