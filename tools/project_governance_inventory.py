"""Read-only executable entrypoint for the project governance inventory.

The package-level CLI intentionally has no process launcher.  This module is
the sole concrete command-runner boundary: it accepts only the reviewed N607
preflight, the collector's exact direct/bridge SSH commands, and a bounded
``netstat.exe`` disconnect probe.  It never sends a signal, ends a process,
or retries a command.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence


_WORKTREE_ROOT = Path(__file__).resolve().parents[1]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from tools.project_governance.cli import main as _cli_main
from tools.project_governance.collect_n607 import (
    APPROVED_ENDPOINTS,
    DEFAULT_PREFLIGHT_SCRIPT,
    DEFAULT_SSH_CONFIG,
    MAX_NDJSON_LINE_BYTES,
    CommandResult,
    ConnectionCheck,
    ConnectionEvidence,
    N607Collector,
    _BRIDGE_COMMAND,
    _direct_command,
)
from tools.project_governance.config import GovernanceConfig


_NETSTAT_COMMAND = ("netstat.exe", "-ano", "-p", "tcp")
_MAX_STDERR_TAIL_BYTES = 8192
_STDOUT_READ_LIMIT = MAX_NDJSON_LINE_BYTES + 3
_STDERR_READ_LIMIT = _MAX_STDERR_TAIL_BYTES + 1
_STREAM_JOIN_SECONDS = 2.0
_TRUNCATION_MARKER = b"[stderr truncated]\n"


class _Tracker(Protocol):
    @property
    def proxy_child_pids(self) -> tuple[int, ...]: ...

    def wait_for_exit(self) -> bool: ...

    def close(self) -> None: ...


class _UnavailableTracker:
    """Fail closed when descendant process evidence cannot be collected."""

    def __init__(self, *, required: bool) -> None:
        self._required = required

    @property
    def proxy_child_pids(self) -> tuple[int, ...]:
        return ()

    def wait_for_exit(self) -> bool:
        return not self._required

    def close(self) -> None:
        return None


if os.name == "nt":
    from ctypes import wintypes

    _TH32CS_SNAPPROCESS = 0x00000002
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _SYNCHRONIZE = 0x00100000
    _WAIT_OBJECT_0 = 0
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    _kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _kernel32.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
    _kernel32.Process32FirstW.restype = wintypes.BOOL
    _kernel32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
    _kernel32.Process32NextW.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL


class _WindowsDescendantTracker:
    """Keep handles for observed descendants so exit evidence is PID-safe."""

    def __init__(self, root_pid: int, required: bool) -> None:
        self._root_pid = root_pid
        self._required = required
        self._handles: dict[int, object] = {}
        self._unopened: set[int] = set()
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        if os.name != "nt":
            self._unsupported = True
            return
        self._unsupported = False
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    @property
    def proxy_child_pids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(set(self._handles).union(self._unopened)))

    def _watch(self) -> None:
        while not self._stopped.is_set():
            self._capture_once()
            self._stopped.wait(0.02)

    def _capture_once(self) -> None:
        if self._unsupported:
            return
        snapshot = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if snapshot in {None, _INVALID_HANDLE_VALUE}:
            return
        parents: dict[int, int] = {}
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            if not _kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                return
            while True:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
                if not _kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
        finally:
            _kernel32.CloseHandle(snapshot)

        descendants: list[int] = []
        for pid in parents:
            if pid == self._root_pid:
                continue
            parent = parents.get(pid)
            seen: set[int] = set()
            while parent is not None and parent not in seen:
                if parent == self._root_pid:
                    descendants.append(pid)
                    break
                seen.add(parent)
                parent = parents.get(parent)

        for pid in descendants:
            with self._lock:
                if pid in self._handles or pid in self._unopened:
                    continue
            handle = _kernel32.OpenProcess(
                _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid,
            )
            with self._lock:
                if handle:
                    self._handles[pid] = handle
                else:
                    self._unopened.add(pid)

    def wait_for_exit(self) -> bool:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=_STREAM_JOIN_SECONDS)
        if self._unsupported:
            return not self._required
        with self._lock:
            if self._required and not self._handles and not self._unopened:
                return False
            if self._unopened:
                return False
            handles = tuple(self._handles.values())
        return all(_kernel32.WaitForSingleObject(handle, 0) == _WAIT_OBJECT_0 for handle in handles)

    def close(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=_STREAM_JOIN_SECONDS)
        if self._unsupported:
            return
        with self._lock:
            handles = tuple(self._handles.values())
            self._handles.clear()
        for handle in handles:
            _kernel32.CloseHandle(handle)


def _default_tracker(root_pid: int, required: bool) -> _Tracker:
    if os.name != "nt":
        return _UnavailableTracker(required=required)
    return _WindowsDescendantTracker(root_pid, required)


class _BoundedByteTail:
    """Thread-safe bounded stderr tail with explicit truncation evidence."""

    def __init__(self) -> None:
        self._payload = bytearray()
        self._truncated = False
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        raw = bytes(chunk)
        with self._lock:
            self._payload.extend(raw)
            if len(self._payload) > _MAX_STDERR_TAIL_BYTES:
                del self._payload[: len(self._payload) - _MAX_STDERR_TAIL_BYTES]
                self._truncated = True

    def text(self) -> str:
        with self._lock:
            payload = bytes(self._payload)
            truncated = self._truncated
        decoded = payload.decode("utf-8", errors="replace")
        encoded = decoded.encode("utf-8")
        if truncated or len(encoded) > _MAX_STDERR_TAIL_BYTES:
            retained = _MAX_STDERR_TAIL_BYTES - len(_TRUNCATION_MARKER)
            bounded = encoded[-retained:].decode("utf-8", errors="ignore")
            return _TRUNCATION_MARKER.decode("ascii") + bounded
        return decoded


def _tail(chunks: Sequence[bytes]) -> str:
    tail = _BoundedByteTail()
    for chunk in chunks:
        tail.append(chunk)
    return tail.text()


def _is_exact_preflight(command: tuple[str, ...]) -> bool:
    return command == (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(DEFAULT_PREFLIGHT_SCRIPT),
    )


def _allowed_command(command: tuple[str, ...], label: str) -> tuple[tuple[str, ...], bool]:
    direct = _direct_command(DEFAULT_SSH_CONFIG)
    if label == "PREFLIGHT" and _is_exact_preflight(command):
        return command, True
    if label == "DIRECT" and command == direct:
        return command, False
    if label == "LAB_BRIDGE" and command == _BRIDGE_COMMAND:
        return command, True
    raise ValueError("command is not an approved read-only collector command")


class ProductionCommandRunner:
    """Narrow command implementation for the injected N607 collector only."""

    def __init__(
        self,
        *,
        popen_factory: Callable[..., object] = subprocess.Popen,
        tracker_factory: Callable[[int, bool], _Tracker] = _default_tracker,
    ) -> None:
        self._popen_factory = popen_factory
        self._tracker_factory = tracker_factory

    def run(
        self,
        command: tuple[str, ...],
        *,
        input_text: str | None,
        timeout_seconds: int,
        label: str,
        stdout_line_handler: Callable[[str | bytes], None] | None = None,
    ) -> CommandResult:
        normalized_command = tuple(str(item) for item in command)
        _, proxy_required = _allowed_command(normalized_command, label)
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        if input_text is not None and not isinstance(input_text, str):
            raise ValueError("input_text must be text or None")
        try:
            process = self._popen_factory(
                normalized_command,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError as exc:
            return CommandResult(
                command=normalized_command,
                child_pid=None,
                returncode=None,
                child_exited=False,
                proxy_children_exited=False,
                stderr_tail=f"unable to start approved command: {exc}",
            )

        child_pid = getattr(process, "pid", None)
        if type(child_pid) is not int or child_pid <= 0:
            return CommandResult(
                command=normalized_command,
                child_pid=None,
                returncode=None,
                child_exited=False,
                proxy_children_exited=False,
                stderr_tail="approved command did not expose a usable child PID",
            )
        tracker = self._tracker_factory(child_pid, proxy_required)
        stdout_lines: list[bytes] = []
        stderr_tail = _BoundedByteTail()
        callback_errors: list[str] = []

        def read_stdout() -> None:
            stream = process.stdout
            while True:
                line = stream.readline(_STDOUT_READ_LIMIT)
                if not line:
                    return
                raw = bytes(line)
                if label == "PREFLIGHT" or stdout_line_handler is None:
                    stdout_lines.append(raw)
                if stdout_line_handler is not None:
                    try:
                        stdout_line_handler(raw)
                    except Exception as exc:
                        callback_errors.append(f"stdout callback failed: {exc}")

        def read_stderr() -> None:
            stream = process.stderr
            while True:
                line = stream.readline(_STDERR_READ_LIMIT)
                if not line:
                    return
                stderr_tail.append(bytes(line))

        reader_threads = [
            threading.Thread(target=read_stdout, daemon=True),
            threading.Thread(target=read_stderr, daemon=True),
        ]
        for thread in reader_threads:
            thread.start()

        writer_thread: threading.Thread | None = None
        writer_errors: list[str] = []
        if input_text is not None:
            def write_stdin() -> None:
                try:
                    process.stdin.write(input_text.encode("utf-8"))
                    process.stdin.flush()
                except (OSError, ValueError) as exc:
                    writer_errors.append(f"stdin write failed: {exc}")
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass

            writer_thread = threading.Thread(target=write_stdin, daemon=True)
            writer_thread.start()

        timed_out = False
        child_exited = False
        returncode: int | None = None
        try:
            returncode = process.wait(timeout=timeout_seconds)
            child_exited = True
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as exc:
            stderr_tail.append(str(exc).encode("utf-8", errors="replace"))

        if child_exited:
            for thread in reader_threads:
                thread.join(timeout=_STREAM_JOIN_SECONDS)
            if any(thread.is_alive() for thread in reader_threads):
                child_exited = False
                stderr_tail.append(b"stdout or stderr did not close after child exit")
            if writer_thread is not None:
                writer_thread.join(timeout=_STREAM_JOIN_SECONDS)
        proxy_child_pids = tracker.proxy_child_pids
        proxy_children_exited = tracker.wait_for_exit() if child_exited else False
        tracker.close()
        if callback_errors:
            for error in callback_errors:
                stderr_tail.append(error.encode("utf-8", errors="replace"))
        if writer_errors:
            for error in writer_errors:
                stderr_tail.append(error.encode("utf-8", errors="replace"))
        return CommandResult(
            command=normalized_command,
            child_pid=child_pid,
            proxy_child_pids=proxy_child_pids,
            returncode=returncode,
            child_exited=child_exited,
            proxy_children_exited=proxy_children_exited,
            stdout_lines=tuple(stdout_lines),
            stderr_tail=stderr_tail.text(),
            timed_out=timed_out,
        )

    def check_connections(
        self, *, attempt_pids: tuple[int, ...], endpoints: tuple[str, ...]
    ) -> ConnectionCheck:
        if tuple(endpoints) != tuple(APPROVED_ENDPOINTS):
            raise ValueError("disconnect probe endpoints are not the approved N607 endpoints")
        if any(type(pid) is not int or pid <= 0 for pid in attempt_pids):
            raise ValueError("disconnect probe PIDs must be positive integers")
        try:
            process = self._popen_factory(
                _NETSTAT_COMMAND,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            return ConnectionCheck(completed=False, error="netstat disconnect probe timed out")
        except OSError as exc:
            return ConnectionCheck(completed=False, error=f"netstat disconnect probe failed: {exc}")
        if getattr(process, "returncode", None) != 0:
            return ConnectionCheck(completed=False, error=_tail([bytes(stderr)]))
        try:
            text = bytes(stdout).decode("utf-8", errors="replace")
        except TypeError:
            return ConnectionCheck(completed=False, error="netstat disconnect probe returned invalid bytes")
        wanted = set(attempt_pids)
        allowed = set(endpoints)
        found: list[ConnectionEvidence] = []
        for raw_line in text.splitlines():
            fields = raw_line.split()
            if len(fields) < 5 or fields[0].upper() != "TCP":
                continue
            remote, state, pid_text = fields[2], fields[3].upper(), fields[4]
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if pid in wanted and remote in allowed:
                found.append(ConnectionEvidence(pid=pid, endpoint=remote, state=state))
        return ConnectionCheck(completed=True, connections=tuple(found))


def _n607_collector_factory(config: GovernanceConfig, scan_id: str) -> N607Collector:
    return N607Collector(
        config.n607,
        config.discovery,
        scan_id=scan_id,
        runner=ProductionCommandRunner(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return _cli_main(argv, n607_collector_factory=_n607_collector_factory)


if __name__ == "__main__":
    raise SystemExit(main())
