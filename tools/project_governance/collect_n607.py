"""Read-only, short-connection metadata collection for the ordinary N607 account.

The remote program is supplied over stdin and writes NDJSON to stdout.  It
does not create a remote file.  Process creation and socket inspection live
behind a runner interface so unit tests never contact a network endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .config import DiscoveryConfig, LocationConfig
from .models import Location
from .paths import normalize_relative_path, stable_asset_id


SCHEMA_VERSION = 1
# Connection setup remains independently capped by ``ConnectTimeout=10``.
# The collection budget must cover the bounded depth-three metadata walk over
# the configured carrier surfaces; the real N607 inventory cannot complete in
# the former 45-second whole-command window.
SSH_TIMEOUT_SECONDS = 15 * 60
PREFLIGHT_TIMEOUT_SECONDS = 45
DIRECT_ROUTE = "DIRECT"
BRIDGE_ROUTE = "LAB_BRIDGE"
APPROVED_N607_HOST = "172.31.111.215:22"
APPROVED_BRIDGE_HOST = "172.31.105.18:22"
APPROVED_ENDPOINTS = (APPROVED_N607_HOST, APPROVED_BRIDGE_HOST)
DEFAULT_PREFLIGHT_SCRIPT = Path("E:/type10-7/tools/n607_ssh_preflight.ps1")
DEFAULT_SSH_CONFIG = DEFAULT_PREFLIGHT_SCRIPT.with_name("n607_ssh_config")
MAX_NDJSON_LINE_BYTES = 16 * 1024 * 1024
MAX_NDJSON_TOTAL_BYTES = 256 * 1024 * 1024
_BRIDGE_PROXY = (
    "ProxyCommand=ssh -i C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18 "
    "-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "
    "-W %h:%p administrator@172.31.105.18"
)
_BRIDGE_COMMAND = (
    "ssh",
    "-i",
    "C:/Users/lh594/.ssh/id_ed25519_n607",
    "-o",
    "BatchMode=yes",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    _BRIDGE_PROXY,
    "szu2070436088@172.31.111.215",
    "python3",
    "-",
)
_PATH_UNAVAILABLE_MARKERS = (
    "connection timed out",
    "operation timed out",
    "no route to host",
    "network is unreachable",
    "connection refused",
)
_SOCKET_STATES = {"ESTABLISHED", "SYN_SENT"}


class RemoteOutcome(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ConnectionEvidence:
    pid: int
    endpoint: str
    state: str


@dataclass(frozen=True)
class ConnectionCheck:
    completed: bool
    connections: tuple[ConnectionEvidence, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    child_pid: int | None
    proxy_child_pids: tuple[int, ...] = ()
    returncode: int | None = None
    child_exited: bool = False
    proxy_children_exited: bool = False
    stdout_lines: tuple[str | bytes, ...] = ()
    stderr_tail: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class AttemptReceipt:
    label: str
    child_pid: int | None
    proxy_child_pids: tuple[int, ...]
    returncode: int | None
    timed_out: bool
    child_exited: bool
    proxy_children_exited: bool
    disconnect_status: str
    lingering_connections: tuple[ConnectionEvidence, ...]
    stderr_tail: str


@dataclass(frozen=True)
class N607Receipt:
    outcome: RemoteOutcome
    route: str | None
    preflight_status: str
    disconnect_status: str
    attempts: tuple[AttemptReceipt, ...]
    active_training_observed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class N607CollectionResult:
    records: tuple[dict[str, object], ...]
    receipt: N607Receipt


class CommandRunner(Protocol):
    def run(
        self,
        command: tuple[str, ...],
        *,
        input_text: str | None,
        timeout_seconds: int,
        label: str,
        stdout_line_handler: Callable[[str | bytes], None] | None = None,
    ) -> CommandResult: ...

    def check_connections(
        self, *, attempt_pids: tuple[int, ...], endpoints: tuple[str, ...]
    ) -> ConnectionCheck | tuple[ConnectionEvidence, ...]: ...


_REMOTE_PAYLOAD_BODY = r'''
import datetime
import hashlib
import json
import os
import socket
import stat
import unicodedata
import urllib.parse

_emitted = 0
_scan_errors = 0
_asset_ids_emitted = set()


def _emit(record_type, **fields):
    global _emitted
    record = {
        "schema_version": SCHEMA_VERSION,
        "scan_id": SCAN_ID,
        "record_type": record_type,
    }
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)
    _emitted += 1


def _error(relative_path, operation, error):
    global _scan_errors
    _scan_errors += 1
    _emit(
        "SCAN_ERROR",
        location="N607",
        root_id=ROOT_ID,
        relative_path=relative_path,
        operation=operation,
        error_type=type(error).__name__,
        error=str(error),
    )


def _normalize_relative(value):
    normalized = unicodedata.normalize("NFC", value)
    if normalized.startswith("/"):
        raise ValueError("absolute relative path")
    parts = []
    for part in normalized.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise ValueError("path escapes root")
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        raise ValueError("empty relative path")
    return "/".join(parts)


def _join_relative(parent, name):
    return _normalize_relative(name if not parent else parent + "/" + name)


def _absolute(relative_path):
    normalized = _normalize_relative(relative_path)
    candidate = os.path.normpath(os.path.join(ROOT, *normalized.split("/")))
    root_norm = os.path.normpath(ROOT)
    if candidate != root_norm and not candidate.startswith(root_norm + os.sep):
        raise ValueError("path escapes configured root")
    return candidate


def _asset_id(relative_path):
    normalized = _normalize_relative(relative_path)
    return "asset:N607:{root}:{path}".format(
        root=urllib.parse.quote(ROOT_ID, safe="-._~"),
        path=urllib.parse.quote(normalized, safe="/-._~"),
    )


def _mtime_utc(value):
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _kind(metadata):
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def _bounded_bytes(path, limit):
    chunks = []
    total = 0
    with open(path, "rb") as stream:
        while total <= limit:
            chunk = stream.read(min(1048576, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    payload = b"".join(chunks)
    if len(payload) > limit:
        raise OverflowError("bounded read size exceeded")
    return payload


def _hash_file(path, size_bytes, suffix):
    if suffix in PROTECTED_SUFFIXES:
        return "METADATA_ONLY", None
    if suffix not in CONTROL_SUFFIXES:
        return "METADATA_ONLY", None
    if size_bytes > HASH_MAX_BYTES:
        return "NOT_HASHED_SIZE_LIMIT", None
    payload = _bounded_bytes(path, HASH_MAX_BYTES)
    return "SHA256", hashlib.sha256(payload).hexdigest()


def _record_asset(path, relative_path, evidence_role, metadata=None):
    try:
        metadata = metadata if metadata is not None else os.lstat(path)
        asset_kind = _kind(metadata)
        name = relative_path.rsplit("/", 1)[-1]
        suffix = os.path.splitext(name)[1].casefold()
        hash_status = "METADATA_ONLY"
        digest = None
        access_status = "OK"
        if asset_kind == "file":
            try:
                hash_status, digest = _hash_file(path, metadata.st_size, suffix)
            except (OSError, OverflowError, UnicodeError) as error:
                access_status = "SCAN_ERROR"
                hash_status = "ERROR"
                _error(relative_path, "bounded_file_read", error)
        fields = {
            "asset_id": _asset_id(relative_path),
            "location": "N607",
            "root_id": ROOT_ID,
            "relative_path": _normalize_relative(relative_path),
            "display_name": unicodedata.normalize("NFC", name),
            "escaped_name": unicodedata.normalize("NFC", name).encode(
                "unicode_escape"
            ).decode("ascii"),
            "asset_kind": asset_kind,
            "size_bytes": metadata.st_size,
            "mtime_utc": _mtime_utc(metadata.st_mtime),
            "access_status": access_status,
            "hash_status": hash_status,
            "sha256": digest,
            "evidence_role": evidence_role,
        }
        if fields["asset_id"] in _asset_ids_emitted:
            return fields["asset_id"], asset_kind
        _asset_ids_emitted.add(fields["asset_id"])
        _emit("ASSET", **fields)
        return fields["asset_id"], asset_kind
    except (OSError, ValueError) as error:
        _error(relative_path, "lstat", error)
        return None, "error"


def _scan_control(directory, relative_directory, depth):
    if depth > MAX_DEPTH:
        return
    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: unicodedata.normalize("NFC", item.name))
    except OSError as error:
        _error(relative_directory, "scandir", error)
        return
    for entry in entries:
        try:
            relative_path = _join_relative(relative_directory, entry.name)
            metadata = os.lstat(entry.path)
            asset_kind = _kind(metadata)
            if asset_kind == "symlink":
                _record_asset(entry.path, relative_path, "LINK", metadata)
            elif asset_kind == "directory":
                if entry.name.casefold() in SUMMARY_DIRECTORY_NAMES:
                    _record_asset(
                        entry.path, relative_path, "PREDICTION_SCORE_SUMMARY", metadata
                    )
                else:
                    _scan_control(entry.path, relative_path, depth + 1)
            elif asset_kind == "file" and os.path.splitext(entry.name)[1].casefold() in CONTROL_SUFFIXES:
                _record_asset(entry.path, relative_path, "CONTROL_EVIDENCE", metadata)
        except (OSError, ValueError) as error:
            _error(relative_directory, "control_evidence", error)


def _scan_direct(directory, relative_directory, tag, descend_units):
    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: unicodedata.normalize("NFC", item.name))
    except FileNotFoundError:
        _emit(
            "SCOPE",
            location="N607",
            root_id=ROOT_ID,
            relative_path=relative_directory,
            status="NOT_PRESENT",
            asset_ids=[],
        )
        return
    except OSError as error:
        _error(relative_directory, "scandir", error)
        _emit(
            "SCOPE",
            location="N607",
            root_id=ROOT_ID,
            relative_path=relative_directory,
            status="SCAN_ERROR",
            asset_ids=[],
        )
        return
    asset_ids = []
    units = []
    for entry in entries:
        try:
            relative_path = _join_relative(relative_directory, entry.name)
            metadata = os.lstat(entry.path)
            asset_id, asset_kind = _record_asset(entry.path, relative_path, tag, metadata)
            if asset_id is not None:
                asset_ids.append(asset_id)
            if descend_units and asset_kind == "directory":
                units.append((entry.path, relative_path))
        except (OSError, ValueError) as error:
            _error(relative_directory, "direct_entry", error)
    _emit(
        "SCOPE",
        location="N607",
        root_id=ROOT_ID,
        relative_path=relative_directory,
        status="VERIFIED",
        asset_ids=asset_ids,
    )
    for unit_path, unit_relative_path in units:
        _scan_control(unit_path, unit_relative_path, 1)


def _safe_carrier(relative_path):
    current = ROOT
    accumulated = []
    try:
        for component in _normalize_relative(relative_path).split("/"):
            accumulated.append(component)
            current = os.path.join(current, component)
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                _record_asset(current, "/".join(accumulated), "LINK", metadata)
                _error(
                    relative_path,
                    "carrier_boundary",
                    ValueError("carrier path crosses a symbolic link"),
                )
                return None
        return current
    except FileNotFoundError:
        return _absolute(relative_path)
    except (OSError, ValueError) as error:
        _error(relative_path, "carrier_boundary", error)
        return None


def _read_process_text(path, limit):
    payload = _bounded_bytes(path, limit)
    return payload.decode("utf-8", errors="backslashreplace")


def _processes():
    try:
        with os.scandir("/proc") as iterator:
            entries = tuple(iterator)
    except OSError as error:
        _error("", "proc_scandir", error)
        return
    unreadable_count = 0
    unreadable_pids = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        proc_root = "/proc/" + entry.name
        try:
            cwd = os.readlink(proc_root + "/cwd")
            root_norm = os.path.normpath(ROOT)
            cwd_norm = os.path.normpath(cwd)
            if cwd_norm != root_norm and not cwd_norm.startswith(root_norm + os.sep):
                continue
            cmdline = _read_process_text(proc_root + "/cmdline", 65536)
            cmdline = " ".join(part for part in cmdline.split("\x00") if part)
            status = _read_process_text(proc_root + "/status", 65536)
            ppid = None
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    raw_ppid = line.split(":", 1)[1].strip()
                    ppid = int(raw_ppid) if raw_ppid.isdigit() else None
                    break
            if ppid is None:
                raise ValueError("PPid is missing from process status")
            lowered = cmdline.casefold()
            training_like = any(token in lowered for token in ("train", "runner", "launch"))
            _emit(
                "PROCESS",
                pid=int(entry.name),
                ppid=ppid,
                cwd=cwd,
                cmdline=cmdline,
                training_like=training_like,
            )
        except (OSError, UnicodeError, ValueError, OverflowError):
            unreadable_count += 1
            if len(unreadable_pids) < 10:
                unreadable_pids.append(entry.name)
    if unreadable_count:
        _error(
            "",
            "proc_partial_visibility",
            RuntimeError(
                "unreadable_processes={count};sample_pids={pids}".format(
                    count=unreadable_count,
                    pids=",".join(unreadable_pids),
                )
            ),
        )


def _main():
    _emit(
        "SERVER_INFO",
        hostname=socket.gethostname(),
        server_time_utc=datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        root=ROOT,
    )
    try:
        root_metadata = os.lstat(ROOT)
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("configured root is not a direct directory")
        _scan_direct(ROOT, "", "ROOT_DIRECT", False)
        for carrier in CARRIER_SURFACES:
            carrier_path = _safe_carrier(carrier)
            if carrier_path is not None:
                _scan_direct(carrier_path, carrier, "CARRIER_DIRECT:" + carrier, True)
            else:
                _emit(
                    "SCOPE",
                    location="N607",
                    root_id=ROOT_ID,
                    relative_path=carrier,
                    status="SCAN_ERROR",
                    asset_ids=[],
                )
        _processes()
    except (OSError, ValueError) as error:
        _error("", "root_lstat", error)
        _emit(
            "SCOPE",
            location="N607",
            root_id=ROOT_ID,
            relative_path="",
            status="SCAN_ERROR",
            asset_ids=[],
        )
        for carrier in CARRIER_SURFACES:
            _emit(
                "SCOPE",
                location="N607",
                root_id=ROOT_ID,
                relative_path=carrier,
                status="SCAN_ERROR",
                asset_ids=[],
            )
    _emit(
        "COLLECTION_COMPLETE",
        record_count=_emitted,
        scan_error_count=_scan_errors,
    )


if __name__ == "__main__":
    _main()
'''


def build_remote_payload(
    location_config: LocationConfig,
    discovery_config: DiscoveryConfig,
    *,
    scan_id: str,
) -> str:
    """Build the self-contained, read-only Python program streamed to N607."""

    if location_config.location is not Location.N607:
        raise ValueError("remote payload requires an N607 configuration")
    if not isinstance(scan_id, str) or not scan_id.strip() or scan_id != scan_id.strip():
        raise ValueError("scan_id must be a canonical non-empty string")
    values = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "SCAN_ID": scan_id,
        "ROOT": location_config.root,
        "ROOT_ID": location_config.root_id,
        "CARRIER_SURFACES": tuple(surface.relative_path for surface in location_config.carrier_surfaces),
        "MAX_DEPTH": discovery_config.control_evidence_max_depth,
        "HASH_MAX_BYTES": discovery_config.hash_max_bytes,
        "CONTROL_SUFFIXES": (".json", ".md", ".markdown", ".py", ".sh", ".toml", ".yaml", ".yml"),
        "PROTECTED_SUFFIXES": (".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".h5", ".mat", ".tar", ".zip", ".7z"),
        "SUMMARY_DIRECTORY_NAMES": ("prediction", "predictions", "score", "scores"),
    }
    assignments = "\n".join(
        f"{name} = {json.dumps(value, ensure_ascii=True, separators=(',', ':'))}"
        for name, value in values.items()
    )
    return assignments + "\n" + _REMOTE_PAYLOAD_BODY


def _preflight_command(script: Path) -> tuple[str, ...]:
    return (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    )


def _direct_command(ssh_config: Path) -> tuple[str, ...]:
    return (
        "ssh",
        "-F",
        str(ssh_config),
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "N607",
        "python3",
        "-",
    )


def _attempt_pids(result: CommandResult) -> tuple[int, ...]:
    values = (() if result.child_pid is None else (result.child_pid,)) + result.proxy_child_pids
    return tuple(dict.fromkeys(values))


def _connection_check(
    runner: CommandRunner, result: CommandResult
) -> tuple[str, tuple[ConnectionEvidence, ...], str | None]:
    attempt_pids = _attempt_pids(result)
    raw = runner.check_connections(attempt_pids=attempt_pids, endpoints=APPROVED_ENDPOINTS)
    check = raw if isinstance(raw, ConnectionCheck) else ConnectionCheck(True, tuple(raw))
    if not check.completed:
        return "UNKNOWN", check.connections, check.error or "disconnect check did not complete"
    lingering = tuple(
        item
        for item in check.connections
        if item.pid in attempt_pids
        and item.endpoint in APPROVED_ENDPOINTS
        and item.state.upper() in _SOCKET_STATES
    )
    if (
        result.child_pid is None
        or not result.child_exited
        or not result.proxy_children_exited
        or lingering
    ):
        return "UNKNOWN", lingering, "attempt process or socket disconnect is not proven"
    return "VERIFIED", (), None


def _receipt_for_attempt(
    label: str,
    result: CommandResult,
    disconnect_status: str,
    lingering: tuple[ConnectionEvidence, ...],
) -> AttemptReceipt:
    return AttemptReceipt(
        label=label,
        child_pid=result.child_pid,
        proxy_child_pids=result.proxy_child_pids,
        returncode=result.returncode,
        timed_out=result.timed_out,
        child_exited=result.child_exited,
        proxy_children_exited=result.proxy_children_exited,
        disconnect_status=disconnect_status,
        lingering_connections=lingering,
        stderr_tail=result.stderr_tail,
    )


def _classify_preflight(
    result: CommandResult, *, expected_ssh_config: Path
) -> tuple[str, str | None]:
    try:
        decoded_lines = tuple(
            line.decode("utf-8") if isinstance(line, bytes) else line
            for line in result.stdout_lines
        )
    except UnicodeDecodeError:
        return "FAILED", "preflight output is not strict UTF-8"
    text = "\n".join(decoded_lines) + "\n" + result.stderr_tail
    folded = text.casefold()
    path_folded = folded.translate(str.maketrans({"\\": "/"}))
    config_valid = folded.count("config ok: n607 is direct") == 1
    config_path_valid = (
        f"ssh config: {expected_ssh_config.as_posix().casefold()}" in path_folded
    )
    identity_valid = folded.count("identity file ok:") >= 1
    remote_user_valid = "user=szu2070436088" in folded
    project_visible = "project_root=/home/szu2070436088/2510044040/cv-sincnet" in folded
    completion_marker = "preflight ok:" in folded
    if result.timed_out or result.returncode is None:
        return "UNKNOWN", "N607 preflight timed out or has no terminal exit evidence"
    if (
        result.returncode == 0
        and config_valid
        and config_path_valid
        and identity_valid
        and remote_user_valid
        and project_visible
        and completion_marker
    ):
        return "DIRECT_READY", None
    if (
        result.returncode != 0
        and config_valid
        and config_path_valid
        and identity_valid
        and any(marker in folded for marker in _PATH_UNAVAILABLE_MARKERS)
    ):
        return "DIRECT_PATH_UNAVAILABLE", None
    if (
        not config_valid
        or not config_path_valid
        or not identity_valid
        or (result.returncode == 0 and not remote_user_valid)
    ):
        return "FAILED", "preflight identity/config evidence is missing or ambiguous"
    return "FAILED", "preflight failed without an authorized bridge-fallback classification"


_ALLOWED_RECORD_TYPES = {
    "SERVER_INFO",
    "ASSET",
    "SCOPE",
    "PROCESS",
    "SCAN_ERROR",
    "COLLECTION_COMPLETE",
}
_ALLOWED_ASSET_KINDS = {"file", "directory", "symlink", "other"}
_ALLOWED_ACCESS_STATUS = {"OK", "SCAN_ERROR"}
_ALLOWED_HASH_STATUS = {
    "SHA256",
    "METADATA_ONLY",
    "NOT_HASHED_SIZE_LIMIT",
    "ERROR",
}
_ALLOWED_SCOPE_STATUS = {"VERIFIED", "NOT_PRESENT", "SCAN_ERROR"}


def _is_int(value: object, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _canonical_n607_path(value: object, *, allow_empty: bool) -> str:
    if value == "" and allow_empty:
        return ""
    if not isinstance(value, str):
        raise ValueError("N607 path is not a string")
    normalized = normalize_relative_path(value, location=Location.N607)
    if normalized != value:
        raise ValueError("N607 path is not canonical")
    return normalized


def _parse_ndjson(
    lines: Sequence[str],
    *,
    scan_id: str,
    expected_root: str,
    expected_carriers: Sequence[str],
) -> tuple[tuple[dict[str, object], ...], bool]:
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        if not isinstance(line, str) or not line.strip():
            raise ValueError(f"malformed NDJSON line {index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed NDJSON line {index}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"NDJSON line {index} is not an object")
        schema_version = value.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SCHEMA_VERSION
            or value.get("scan_id") != scan_id
        ):
            raise ValueError(f"NDJSON identity mismatch at line {index}")
        record_type = value.get("record_type")
        if record_type not in _ALLOWED_RECORD_TYPES:
            raise ValueError(f"NDJSON record type missing at line {index}")
        if "relative_path" in value:
            try:
                _canonical_n607_path(value["relative_path"], allow_empty=True)
            except ValueError as exc:
                raise ValueError(f"NDJSON path is not canonical at line {index}") from exc
        if record_type == "SERVER_INFO":
            if (
                not isinstance(value.get("hostname"), str)
                or not value.get("hostname")
                or not isinstance(value.get("server_time_utc"), str)
                or not str(value.get("server_time_utc")).endswith("Z")
                or value.get("root") != expected_root
            ):
                raise ValueError(f"NDJSON server evidence is incomplete at line {index}")
        if record_type == "ASSET":
            if value.get("location") != "N607" or value.get("root_id") != "N607_CVS_SINCNET":
                raise ValueError(f"NDJSON asset scope mismatch at line {index}")
            relative_path = value.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"NDJSON asset path missing at line {index}")
            if value.get("asset_id") != stable_asset_id(
                Location.N607, "N607_CVS_SINCNET", relative_path
            ):
                raise ValueError(f"NDJSON asset identity mismatch at line {index}")
            if (
                value.get("asset_kind") not in _ALLOWED_ASSET_KINDS
                or not _is_int(value.get("size_bytes"))
                or not isinstance(value.get("mtime_utc"), str)
                or not str(value.get("mtime_utc")).endswith("Z")
                or value.get("access_status") not in _ALLOWED_ACCESS_STATUS
                or value.get("hash_status") not in _ALLOWED_HASH_STATUS
                or not isinstance(value.get("display_name"), str)
                or not isinstance(value.get("escaped_name"), str)
                or not isinstance(value.get("evidence_role"), str)
            ):
                raise ValueError(f"NDJSON asset fields are incomplete at line {index}")
            digest = value.get("sha256")
            if value.get("hash_status") == "SHA256":
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"NDJSON asset hash is invalid at line {index}")
            elif digest is not None:
                raise ValueError(f"NDJSON metadata-only asset has a hash at line {index}")
            if "evidence_text" in value and not isinstance(value["evidence_text"], str):
                raise ValueError(f"NDJSON evidence text is invalid at line {index}")
        elif record_type == "SCOPE":
            if value.get("location") != "N607" or value.get("root_id") != "N607_CVS_SINCNET":
                raise ValueError(f"NDJSON scope mismatch at line {index}")
            try:
                _canonical_n607_path(value.get("relative_path"), allow_empty=True)
            except ValueError as exc:
                raise ValueError(f"NDJSON scope path is invalid at line {index}") from exc
            asset_ids = value.get("asset_ids")
            if (
                value.get("status") not in _ALLOWED_SCOPE_STATUS
                or not isinstance(asset_ids, list)
                or any(not isinstance(asset_id, str) or not asset_id for asset_id in asset_ids)
                or len(asset_ids) != len(set(asset_ids))
            ):
                raise ValueError(f"NDJSON scope fields are incomplete at line {index}")
        elif record_type == "PROCESS":
            pid = value.get("pid")
            ppid = value.get("ppid")
            cwd = value.get("cwd")
            root_prefix = expected_root.rstrip("/") + "/"
            if (
                not _is_int(pid, minimum=1)
                or not _is_int(ppid)
                or not isinstance(cwd, str)
                or (cwd != expected_root and not cwd.startswith(root_prefix))
                or not isinstance(value.get("cmdline"), str)
                or not isinstance(value.get("training_like"), bool)
            ):
                raise ValueError(f"NDJSON process fields are incomplete at line {index}")
        elif record_type == "SCAN_ERROR":
            try:
                _canonical_n607_path(value.get("relative_path"), allow_empty=True)
            except ValueError as exc:
                raise ValueError(f"NDJSON scan error path is invalid at line {index}") from exc
            if (
                value.get("location") != "N607"
                or value.get("root_id") != "N607_CVS_SINCNET"
                or not isinstance(value.get("operation"), str)
                or not value.get("operation")
                or not isinstance(value.get("error_type"), str)
                or not value.get("error_type")
                or not isinstance(value.get("error"), str)
                or not value.get("error")
            ):
                raise ValueError(f"NDJSON scan error fields are incomplete at line {index}")
        records.append(value)
    completions = [item for item in records if item.get("record_type") == "COLLECTION_COMPLETE"]
    if len(completions) != 1 or not records or records[-1] is not completions[0]:
        raise ValueError("NDJSON completion marker is missing")
    server_records = [item for item in records if item.get("record_type") == "SERVER_INFO"]
    if (
        len(server_records) != 1
        or records[0] is not server_records[0]
        or server_records[0].get("root") != expected_root
    ):
        raise ValueError("NDJSON server/root evidence is incomplete")
    scope_records = [item for item in records if item.get("record_type") == "SCOPE"]
    expected_scope_paths = ("",) + tuple(expected_carriers)
    observed_scope_paths = tuple(str(item.get("relative_path")) for item in scope_records)
    if (
        len(observed_scope_paths) != len(expected_scope_paths)
        or set(observed_scope_paths) != set(expected_scope_paths)
    ):
        raise ValueError("NDJSON root/carrier scope coverage is incomplete")
    root_scope = next(scope for scope in scope_records if scope.get("relative_path") == "")
    if root_scope.get("status") != "VERIFIED":
        raise ValueError("NDJSON root scope must be VERIFIED")
    asset_ids = [
        str(item.get("asset_id")) for item in records if item.get("record_type") == "ASSET"
    ]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("NDJSON asset identities are not unique")
    known_asset_ids = set(asset_ids)
    if any(
        asset_id not in known_asset_ids
        for scope in scope_records
        for asset_id in scope.get("asset_ids", [])
    ):
        raise ValueError("NDJSON scope references an unknown asset")
    completion = records[-1]
    record_count = completion.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count != len(records) - 1
    ):
        raise ValueError("NDJSON record count does not close")
    scan_errors = sum(item.get("record_type") == "SCAN_ERROR" for item in records[:-1])
    scan_error_count = completion.get("scan_error_count")
    if (
        isinstance(scan_error_count, bool)
        or not isinstance(scan_error_count, int)
        or scan_error_count != scan_errors
    ):
        raise ValueError("NDJSON scan error count does not close")
    active_training = any(
        item.get("record_type") == "PROCESS" and item.get("training_like") is True
        for item in records
    )
    return tuple(records), active_training


class _NDJSONStreamValidator:
    """Strictly decode and parse one bounded UTF-8 object at a time."""

    def __init__(
        self,
        *,
        scan_id: str,
        expected_root: str,
        expected_carriers: Sequence[str],
    ) -> None:
        self._scan_id = scan_id
        self._expected_root = expected_root
        self._expected_carriers = tuple(expected_carriers)
        self._lines: list[str] = []
        self._total_bytes = 0
        self._error: str | None = None

    def feed(self, raw_line: str | bytes) -> None:
        if self._error is not None:
            return
        try:
            payload = raw_line.encode("utf-8") if isinstance(raw_line, str) else raw_line
            if not isinstance(payload, bytes):
                raise TypeError("stdout line is not bytes or text")
            if payload.endswith(b"\n"):
                payload = payload[:-1]
                if payload.endswith(b"\r"):
                    payload = payload[:-1]
            if b"\n" in payload or b"\r" in payload:
                raise ValueError("stdout callback supplied multiple lines")
            if len(payload) > MAX_NDJSON_LINE_BYTES:
                raise ValueError("NDJSON line exceeds the bounded size")
            self._total_bytes += len(payload) + 1
            if self._total_bytes > MAX_NDJSON_TOTAL_BYTES:
                raise ValueError("NDJSON stream exceeds the bounded total size")
            decoded = payload.decode("utf-8")
            value = json.loads(decoded)
            if not isinstance(value, dict):
                raise ValueError("NDJSON line is not an object")
            self._lines.append(decoded)
        except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            self._error = f"malformed streaming NDJSON: {exc}"

    def finish(self) -> tuple[tuple[dict[str, object], ...], bool]:
        if self._error is not None:
            raise ValueError(self._error)
        return _parse_ndjson(
            self._lines,
            scan_id=self._scan_id,
            expected_root=self._expected_root,
            expected_carriers=self._expected_carriers,
        )


class N607Collector:
    """Run the approved direct-first collection protocol through an injected runner."""

    def __init__(
        self,
        location_config: LocationConfig,
        discovery_config: DiscoveryConfig,
        *,
        scan_id: str,
        runner: CommandRunner,
        preflight_script: Path | None = None,
    ) -> None:
        if location_config.location is not Location.N607:
            raise ValueError("N607Collector requires an N607 configuration")
        self._location = location_config
        self._discovery = discovery_config
        self._scan_id = scan_id
        self._runner = runner
        # The approved script reads its pinned SSH config from the same
        # directory.  Linked Git worktrees intentionally do not carry that
        # host-local config, so the production default remains on the fixed
        # project control plane.
        self._preflight_script = preflight_script or DEFAULT_PREFLIGHT_SCRIPT
        self._ssh_config = self._preflight_script.with_name("n607_ssh_config")

    def collect(self) -> N607CollectionResult:
        attempts: list[AttemptReceipt] = []
        preflight = self._run_attempt(
            "PREFLIGHT",
            _preflight_command(self._preflight_script),
            input_text=None,
            timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS,
            stdout_line_handler=None,
        )
        preflight_result, disconnect_status, lingering, disconnect_error = preflight
        attempts.append(_receipt_for_attempt("PREFLIGHT", preflight_result, disconnect_status, lingering))
        if disconnect_status != "VERIFIED":
            return self._result(
                RemoteOutcome.UNKNOWN,
                None,
                "UNKNOWN",
                attempts,
                disconnect_status,
                disconnect_error,
            )

        preflight_status, preflight_error = _classify_preflight(
            preflight_result, expected_ssh_config=self._ssh_config
        )
        if preflight_status in {"DIRECT_READY", "DIRECT_PATH_UNAVAILABLE"} and not preflight_result.proxy_child_pids:
            return self._result(
                RemoteOutcome.UNKNOWN,
                None,
                preflight_status,
                attempts,
                "UNKNOWN",
                "preflight SSH child exit evidence was not captured",
            )
        if preflight_status == "UNKNOWN":
            return self._result(
                RemoteOutcome.UNKNOWN,
                None,
                preflight_status,
                attempts,
                disconnect_status,
                preflight_error,
            )
        if preflight_status == "FAILED":
            return self._result(
                RemoteOutcome.FAILED,
                None,
                preflight_status,
                attempts,
                disconnect_status,
                preflight_error,
            )

        route = DIRECT_ROUTE if preflight_status == "DIRECT_READY" else BRIDGE_ROUTE
        command = _direct_command(self._ssh_config) if route == DIRECT_ROUTE else _BRIDGE_COMMAND
        payload = build_remote_payload(self._location, self._discovery, scan_id=self._scan_id)
        stream_validator = _NDJSONStreamValidator(
            scan_id=self._scan_id,
            expected_root=self._location.root,
            expected_carriers=tuple(
                surface.relative_path for surface in self._location.carrier_surfaces
            ),
        )
        command_result, disconnect_status, lingering, disconnect_error = self._run_attempt(
            route,
            command,
            input_text=payload,
            timeout_seconds=SSH_TIMEOUT_SECONDS,
            stdout_line_handler=stream_validator.feed,
        )
        attempts.append(_receipt_for_attempt(route, command_result, disconnect_status, lingering))
        if route == BRIDGE_ROUTE and not command_result.proxy_child_pids:
            return self._result(
                RemoteOutcome.UNKNOWN,
                route,
                preflight_status,
                attempts,
                "UNKNOWN",
                "bridge proxy child exit evidence was not captured",
            )
        if disconnect_status != "VERIFIED" or command_result.timed_out or command_result.returncode is None:
            return self._result(
                RemoteOutcome.UNKNOWN,
                route,
                preflight_status,
                attempts,
                disconnect_status,
                disconnect_error or "collection terminal/disconnect evidence is incomplete",
            )
        if command_result.returncode != 0:
            return self._result(
                RemoteOutcome.FAILED,
                route,
                preflight_status,
                attempts,
                disconnect_status,
                f"collection command exited {command_result.returncode}",
            )
        try:
            records, active_training = stream_validator.finish()
        except ValueError as exc:
            return self._result(
                RemoteOutcome.FAILED,
                route,
                preflight_status,
                attempts,
                disconnect_status,
                str(exc),
            )
        receipt = N607Receipt(
            outcome=RemoteOutcome.VERIFIED,
            route=route,
            preflight_status=preflight_status,
            disconnect_status=disconnect_status,
            attempts=tuple(attempts),
            active_training_observed=active_training,
        )
        return N607CollectionResult(records=records, receipt=receipt)

    def _run_attempt(
        self,
        label: str,
        command: tuple[str, ...],
        *,
        input_text: str | None,
        timeout_seconds: int,
        stdout_line_handler: Callable[[str | bytes], None] | None,
    ) -> tuple[CommandResult, str, tuple[ConnectionEvidence, ...], str | None]:
        result = self._runner.run(
            command,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
            label=label,
            stdout_line_handler=stdout_line_handler,
        )
        disconnect_status, lingering, error = _connection_check(self._runner, result)
        return result, disconnect_status, lingering, error

    @staticmethod
    def _result(
        outcome: RemoteOutcome,
        route: str | None,
        preflight_status: str,
        attempts: list[AttemptReceipt],
        disconnect_status: str,
        error: str | None,
    ) -> N607CollectionResult:
        return N607CollectionResult(
            records=(),
            receipt=N607Receipt(
                outcome=outcome,
                route=route,
                preflight_status=preflight_status,
                disconnect_status=disconnect_status,
                attempts=tuple(attempts),
                error=error,
            ),
        )


__all__ = [
    "APPROVED_BRIDGE_HOST",
    "APPROVED_N607_HOST",
    "AttemptReceipt",
    "BRIDGE_ROUTE",
    "CommandResult",
    "ConnectionCheck",
    "ConnectionEvidence",
    "DEFAULT_PREFLIGHT_SCRIPT",
    "DEFAULT_SSH_CONFIG",
    "DIRECT_ROUTE",
    "N607CollectionResult",
    "N607Collector",
    "N607Receipt",
    "RemoteOutcome",
    "build_remote_payload",
]
