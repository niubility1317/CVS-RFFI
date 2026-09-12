"""Production bubblewrap/strace execution boundary for a Stage2 predictor.

This module never imports training, dataset, legacy-loader, or scorer code.  It
validates the truth-free request before launch, constructs the only supported
predictor argv, executes it in a no-network bubblewrap mount namespace, and
turns successful ``open*`` syscalls into sealed, tamper-evident post-run evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import posixpath
import re
import secrets
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .phase2_bwrap_policy import build_phase2_bwrap_command
from .phase2_pre_run_evidence import (
    FORMAL_LAUNCH_BLOCKERS,
    verify_phase2_pre_run_evidence,
)
from .phase2_runtime_contract import (
    POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    validate_phase2_contract,
    validate_predictor_request,
)
from .stage2_prediction_artifact import verify_prediction_artifact


TRACE_NAME = "filesystem_access_trace.log"
AUDIT_NAME = "filesystem_access_audit.json"
STDOUT_RECEIPT_NAME = "predictor_stdout_receipt.json"
POST_EVIDENCE_NAME = "phase2_diagnostic_post_run_runtime_evidence.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SYSCALL_RE = re.compile(r"\b(open|openat|openat2)\(")
_EXECVE_RE = re.compile(r"\bexecve\(")
_RESULT_RE = re.compile(r"\)\s+=\s+(-?\d+)(?:<([^>]*)>)?(?:\s|$)")
_QUOTED_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_BRACKET_PID_RE = re.compile(r"^\[pid\s+(\d+)\]\s+")
_PLAIN_PID_RE = re.compile(r"^(\d+)\s+")
_RESUMED_RE = re.compile(r"^<\.\.\.\s+(open|openat|openat2)\s+resumed>(.*)$")
_EXECVE_RESUMED_RE = re.compile(r"^<\.\.\.\s+execve\s+resumed>(.*)$")
_SENSITIVE_COMPONENTS = frozenset(
    {
        "truth",
        "truth_sidecar",
        "scorer",
        "scoring",
        "scoring_manifest",
        "clean",
        "raw",
        "clean_cache",
        "clean_dataset",
        "dataset_wisig",
        "manysig",
        "manytx",
    }
)


class Phase2IsolatedRunnerError(RuntimeError):
    """Raised when production isolation or its post-run evidence fails closed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_regular_nofollow(path: Path) -> bytes:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Phase2IsolatedRunnerError(f"input must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise Phase2IsolatedRunnerError(f"input identity changed before open: {path}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise Phase2IsolatedRunnerError(f"input was truncated during read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_exclusive_readonly(path: Path, data: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    if stat.S_IMODE(os.lstat(path).st_mode) & 0o222:
        raise Phase2IsolatedRunnerError(f"evidence could not be made read-only: {path}")
    return _sha256_bytes(data)


def _write_json_exclusive_readonly(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_exclusive_readonly(path, _canonical_json(payload))


def build_production_predictor_argv(
    *, expected_seal_sha256: str, device: str = "cuda:0", batch_size: int = 256
) -> list[str]:
    """Return the only supported in-sandbox predictor command-line arguments."""

    if _SHA256_RE.fullmatch(expected_seal_sha256) is None:
        raise Phase2IsolatedRunnerError("expected package seal SHA256 is invalid")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise Phase2IsolatedRunnerError("batch_size must be a positive integer")
    if not isinstance(device, str) or not device or any(ch.isspace() for ch in device):
        raise Phase2IsolatedRunnerError("device must be a non-empty token")
    return [
        "/runtime/code/scripts/run_cvs_stage2_predictor.py",
        "--request-json",
        "/sealed/request.json",
        "--predictor-package-root",
        "/sealed/package",
        "--detached-seal-path",
        "/sealed/package.seal.json",
        "--expected-seal-sha256",
        expected_seal_sha256,
        "--output-root",
        "/output",
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]


def _decode_strace_string(token: str) -> str:
    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError) as exc:
        raise Phase2IsolatedRunnerError("strace path string could not be decoded") from exc
    if not isinstance(value, str) or "\x00" in value:
        raise Phase2IsolatedRunnerError("strace path is not a valid string")
    return value


def _normalise_open_path(path: str, *, dirfd: str, cwd: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    if dirfd == "AT_FDCWD":
        base = cwd
    else:
        match = re.search(r"<([^>]*)>", dirfd)
        if match is None or not match.group(1).startswith("/"):
            raise Phase2IsolatedRunnerError(f"relative open path has unresolved dirfd: {dirfd}")
        base = match.group(1)
    return posixpath.normpath(posixpath.join(base, path))


def _trace_pid_and_body(line: str) -> tuple[str, str]:
    for pattern in (_BRACKET_PID_RE, _PLAIN_PID_RE):
        match = pattern.match(line)
        if match is not None:
            return match.group(1), line[match.end() :]
    return "main", line


def _bound_exec_index(
    text: str,
    *,
    expected_executable: str | Path,
    expected_entrypoint: str | Path | None = None,
) -> int:
    expected = posixpath.normpath(str(expected_executable).replace("\\", "/"))
    expected_entry = (
        None
        if expected_entrypoint is None
        else posixpath.normpath(str(expected_entrypoint).replace("\\", "/"))
    )
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        _pid, body = _trace_pid_and_body(raw_line)
        exec_match = _EXECVE_RE.search(body)
        if exec_match is None:
            continue
        result_match = _RESULT_RE.search(body)
        if result_match is None or int(result_match.group(1)) < 0:
            continue
        quoted = _QUOTED_RE.findall(body[exec_match.start() : result_match.start() + 1])
        if not quoted:
            raise Phase2IsolatedRunnerError("successful execve has no parseable executable path")
        decoded = [
            posixpath.normpath(_decode_strace_string(token).replace("\\", "/"))
            for token in quoted
        ]
        # strace quotes the exec path first, followed by argv[0], argv[1], ...
        # The fixed Python script must occupy argv[1].  Searching every argv
        # token is unsafe because the trusted launcher itself carries the
        # predictor path as a --predictor-entry argument.
        entrypoint_matches = (
            expected_entry is None
            or (len(decoded) >= 3 and decoded[2] == expected_entry)
        )
        if decoded[0] == expected and entrypoint_matches:
            return index
    raise Phase2IsolatedRunnerError("strace did not observe the bound predictor Python execve")


def _predictor_trace_suffix(
    text: str,
    *,
    expected_executable: str | Path,
    expected_entrypoint: str | Path | None = None,
) -> str:
    """Drop outer isolation setup calls and retain the predictor phase."""

    lines = text.splitlines()
    index = _bound_exec_index(
        text,
        expected_executable=expected_executable,
        expected_entrypoint=expected_entrypoint,
    )
    return "\n".join(lines[index + 1 :]) + (
        "\n" if index + 1 < len(lines) else ""
    )


def _complete_trace_lines(text: str) -> list[tuple[int, str]]:
    completed: list[tuple[int, str]] = []
    pending: dict[tuple[str, str], tuple[int, str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        pid, body = _trace_pid_and_body(raw_line)
        if "<unfinished ...>" in body:
            syscall_match = _SYSCALL_RE.search(body)
            if syscall_match is None:
                continue
            key = (pid, syscall_match.group(1))
            if key in pending:
                raise Phase2IsolatedRunnerError("strace contains duplicate unfinished open syscall")
            pending[key] = (line_number, body.replace("<unfinished ...>", ""))
            continue
        resumed = _RESUMED_RE.match(body)
        if resumed is not None:
            key = (pid, resumed.group(1))
            if key not in pending:
                raise Phase2IsolatedRunnerError("strace contains an orphan resumed open syscall")
            original_line, prefix = pending.pop(key)
            completed.append((original_line, prefix + resumed.group(2)))
            continue
        completed.append((line_number, body))
    if pending:
        raise Phase2IsolatedRunnerError("strace ends with an unfinished open syscall")
    return completed


def parse_successful_execve_trace(text: str) -> list[dict[str, Any]]:
    """Parse successful execve calls after the bound predictor entrypoint."""

    result: list[dict[str, Any]] = []
    pending: dict[str, tuple[int, str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        pid, body = _trace_pid_and_body(raw_line)
        if "<unfinished ...>" in body and _EXECVE_RE.search(body) is not None:
            if pid in pending:
                raise Phase2IsolatedRunnerError(
                    "strace contains duplicate unfinished execve syscall"
                )
            pending[pid] = (line_number, body.replace("<unfinished ...>", ""))
            continue
        resumed = _EXECVE_RESUMED_RE.match(body)
        if resumed is not None:
            if pid not in pending:
                raise Phase2IsolatedRunnerError(
                    "strace contains an orphan resumed execve syscall"
                )
            original_line, prefix = pending.pop(pid)
            line_number = original_line
            body = prefix + resumed.group(1)
        exec_match = _EXECVE_RE.search(body)
        if exec_match is None:
            continue
        result_match = _RESULT_RE.search(body)
        if result_match is None or int(result_match.group(1)) < 0:
            continue
        quoted = _QUOTED_RE.findall(body[exec_match.start() : result_match.start() + 1])
        if not quoted:
            raise Phase2IsolatedRunnerError(
                f"successful execve has no parseable executable path at line {line_number}"
            )
        result.append(
            {
                "line_number": line_number,
                "executable": posixpath.normpath(
                    _decode_strace_string(quoted[0]).replace("\\", "/")
                ),
            }
        )
    if pending:
        raise Phase2IsolatedRunnerError("strace ends with an unfinished execve syscall")
    return result


def parse_successful_open_trace(text: str, *, cwd: str = "/runtime/code") -> list[dict[str, Any]]:
    """Parse successful open/openat/openat2 calls into an aggregated ledger."""

    aggregate: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in _complete_trace_lines(text):
        syscall_match = _SYSCALL_RE.search(raw_line)
        if syscall_match is None:
            continue
        result_match = _RESULT_RE.search(raw_line)
        if result_match is None:
            continue
        if int(result_match.group(1)) < 0:
            continue
        call = raw_line[syscall_match.start() : result_match.start() + 1]
        quoted = _QUOTED_RE.findall(call)
        if not quoted:
            raise Phase2IsolatedRunnerError(
                f"successful open syscall has no parseable path at line {line_number}"
            )
        syscall = syscall_match.group(1)
        path = _decode_strace_string(quoted[0])
        arguments = call[call.find("(") + 1 :]
        dirfd = "AT_FDCWD" if syscall == "open" else arguments.split(",", 1)[0].strip()
        normalised = _normalise_open_path(path, dirfd=dirfd, cwd=cwd)
        resolved_target = result_match.group(2)
        if resolved_target:
            resolved_target = resolved_target.removesuffix(" (deleted)")
            if resolved_target.startswith("/"):
                normalised = posixpath.normpath(resolved_target)
        row = aggregate.setdefault(
            normalised,
            {"path": normalised, "successful_open_count": 0, "syscalls": set()},
        )
        row["successful_open_count"] += 1
        row["syscalls"].add(syscall)
    ledger: list[dict[str, Any]] = []
    for path in sorted(aggregate):
        row = aggregate[path]
        ledger.append(
            {
                "path": row["path"],
                "successful_open_count": row["successful_open_count"],
                "syscalls": sorted(row["syscalls"]),
            }
        )
    return ledger


def _path_inside(path: str, root: str) -> bool:
    path_obj = PurePosixPath(path)
    root_obj = PurePosixPath(posixpath.normpath(root))
    return path_obj == root_obj or root_obj in path_obj.parents


def _sensitive_path(path: str) -> bool:
    components = [part.lower() for part in PurePosixPath(path).parts]
    return any(
        part.split(".", 1)[0] in _SENSITIVE_COMPONENTS
        or part.startswith(("truth_", "scorer_", "scoring_", "clean_", "raw_"))
        for part in components
    ) or path.lower().endswith(".pkl")


def _normalise_package_member_paths(
    values: Iterable[str], *, context: str
) -> list[str]:
    root = PurePosixPath("/sealed/package")
    result: set[str] = set()
    for raw in values:
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise Phase2IsolatedRunnerError(f"{context} contains an invalid package member")
        candidate = PurePosixPath(raw)
        if candidate.is_absolute():
            normalised = PurePosixPath(posixpath.normpath(raw))
            if normalised == root or root not in normalised.parents:
                raise Phase2IsolatedRunnerError(
                    f"{context} member is outside /sealed/package: {raw}"
                )
        else:
            if any(part in {"", ".", ".."} for part in candidate.parts):
                raise Phase2IsolatedRunnerError(
                    f"{context} contains an unsafe relative package member: {raw}"
                )
            normalised = root.joinpath(candidate)
        result.add(normalised.as_posix())
    return sorted(result)


def audit_open_ledger(
    ledger: Sequence[Mapping[str, Any]],
    *,
    system_read_roots: Iterable[str | Path],
    forbidden_project_roots: Iterable[str] = (),
    sealed_package_members: Iterable[str] | None = None,
    required_package_members: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit the actual ledger against the sandbox-visible path allowlist."""

    system_roots = sorted({posixpath.normpath(str(path)) for path in system_read_roots})
    exact_package_members = (
        None
        if sealed_package_members is None
        else _normalise_package_member_paths(
            sealed_package_members, context="sealed_package_members"
        )
    )
    required_package = _normalise_package_member_paths(
        required_package_members, context="required_package_members"
    )
    if exact_package_members is not None and not set(required_package).issubset(
        exact_package_members
    ):
        raise Phase2IsolatedRunnerError(
            "required_package_members must be included in sealed_package_members"
        )
    allowed_roots = ["/runtime/code", "/sealed/package", "/output", "/tmp", "/proc", "/dev"]
    allowed_roots.extend(system_roots)
    allowed_exact = ["/sealed/package.seal.json", "/sealed/request.json"]
    forbidden_project = sorted({posixpath.normpath(path) for path in forbidden_project_roots})
    violations: list[dict[str, str]] = []
    opened_paths = {str(row["path"]) for row in ledger}
    required_exact = ["/sealed/package.seal.json", "/sealed/request.json"]
    required_roots = ["/runtime/code", "/sealed/package", "/output"]
    for required in required_exact:
        if required not in opened_paths:
            violations.append({"path": required, "reason": "required_runtime_input_not_opened"})
    for required in required_roots:
        if not any(_path_inside(path, required) for path in opened_paths):
            violations.append({"path": required, "reason": "required_runtime_root_not_opened"})
    for required in required_package:
        if required not in opened_paths:
            violations.append({"path": required, "reason": "required_package_member_not_opened"})
    for row in ledger:
        path = str(row["path"])
        if _sensitive_path(path):
            violations.append({"path": path, "reason": "truth_scorer_or_clean_sensitive_path"})
            continue
        if (
            exact_package_members is not None
            and path != "/sealed/package"
            and _path_inside(path, "/sealed/package")
            and path not in exact_package_members
        ):
            violations.append({"path": path, "reason": "sealed_package_member_not_allowlisted"})
            continue
        if any(_path_inside(path, root) for root in forbidden_project):
            violations.append({"path": path, "reason": "host_project_path_outside_mount_allowlist"})
            continue
        if path not in allowed_exact and not any(_path_inside(path, root) for root in allowed_roots):
            violations.append({"path": path, "reason": "path_outside_runtime_allowlist"})
    return {
        "status": "PASS" if not violations else "FAIL",
        "allowed_exact_paths": allowed_exact,
        "allowed_read_roots": allowed_roots,
        "required_exact_paths": required_exact,
        "required_open_roots": required_roots,
        "sealed_package_exact_members": exact_package_members,
        "required_package_members": required_package,
        "forbidden_host_project_roots": forbidden_project,
        "opened_file_ledger": [dict(row) for row in ledger],
        "unique_opened_path_count": len(ledger),
        "successful_open_count": sum(int(row["successful_open_count"]) for row in ledger),
        "violations": violations,
    }


def _parse_predictor_stdout(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Phase2IsolatedRunnerError("predictor stdout is not one JSON object") from exc
    if not isinstance(payload, dict):
        raise Phase2IsolatedRunnerError("predictor stdout root is not an object")
    for field in ("artifact_sha256", "seal_sha256", "request_sha256"):
        if _SHA256_RE.fullmatch(str(payload.get(field, ""))) is None:
            raise Phase2IsolatedRunnerError(f"predictor stdout digest is absent: {field}")
    return payload


def execute_phase2_isolated(
    *,
    bwrap: str | Path,
    strace_executable: str | Path,
    runtime_closure_root: str | Path,
    pre_run_evidence_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    expected_package_seal_sha256: str,
    request_json: str | Path,
    output_root: str | Path,
    python_executable: str | Path,
    system_read_roots: Sequence[str | Path],
    gpu_devices: Sequence[str | Path] = (),
    forbidden_roots: Sequence[str | Path] = (),
    forbidden_project_roots: Sequence[str] = (),
    sealed_package_members: Sequence[str] | None = None,
    required_package_members: Sequence[str] = (),
    device: str = "cuda:0",
    batch_size: int = 256,
    timeout_seconds: int | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute one diagnostic predictor cell; formal launch remains hard-blocked."""

    output = Path(output_root).resolve(strict=True)
    if Path(output_root).is_symlink() or not output.is_dir():
        raise Phase2IsolatedRunnerError("output root must be a non-symlink directory")
    if any(output.iterdir()):
        raise Phase2IsolatedRunnerError("production output root must be empty")
    if not forbidden_roots:
        raise Phase2IsolatedRunnerError(
            "production runner requires at least one separate scorer/truth root"
        )

    request_source = Path(request_json)
    request_bytes = _read_regular_nofollow(request_source)
    request_path = request_source.resolve(strict=True)
    request_sha256 = _sha256_bytes(request_bytes)
    try:
        request = json.loads(request_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase2IsolatedRunnerError("request is not UTF-8 JSON") from exc
    if not isinstance(request, dict):
        raise Phase2IsolatedRunnerError("request root must be an object")
    validate_predictor_request(request)
    pre_evidence = request["phase2_runtime_isolation_evidence"]
    if set(pre_evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise Phase2IsolatedRunnerError("request pre-run evidence does not use the exact 9-field schema")
    if pre_evidence["os_isolation_mode"] != "bwrap_readonly_mounts":
        raise Phase2IsolatedRunnerError("production runner requires bwrap_readonly_mounts evidence")
    verified_pre_run = verify_phase2_pre_run_evidence(
        evidence_root=pre_run_evidence_root,
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        bwrap_executable=bwrap,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_roots,
        expected_evidence=pre_evidence,
    )
    runtime_root = verified_pre_run["runtime_root"]
    package_seal_sha256 = str(pre_evidence["sealed_inference_package_sha256"])
    if package_seal_sha256 != str(expected_package_seal_sha256).lower():
        raise Phase2IsolatedRunnerError("request package seal is not the external trusted seal")
    seal_bytes = _read_regular_nofollow(Path(detached_seal))
    if _sha256_bytes(seal_bytes) != package_seal_sha256:
        raise Phase2IsolatedRunnerError("detached package seal does not match request evidence")
    output_relative = PurePosixPath(str(request["output_contract"]["relative_path"]))
    if output_relative.parent != PurePosixPath(".") or not output_relative.name:
        raise Phase2IsolatedRunnerError("prediction artifact must be a direct output child")

    temp_trace_name = f".phase2_open_trace.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    temp_trace = output.parent / temp_trace_name
    strace_path = Path(strace_executable).resolve(strict=True)
    predictor_argv = build_production_predictor_argv(
        expected_seal_sha256=package_seal_sha256,
        device=device,
        batch_size=batch_size,
    )
    try:
        bwrap_command = build_phase2_bwrap_command(
            bwrap=str(Path(bwrap).resolve(strict=True)),
            runtime_root=runtime_root,
            package_root=package_root,
            detached_seal=detached_seal,
            request_json=request_json,
            output_root=output,
            python_executable=python_executable,
            predictor_argv=predictor_argv,
            system_read_roots=system_read_roots,
            trusted_system_read_roots=verified_pre_run["trusted_system_read_roots"],
            gpu_devices=gpu_devices,
            forbidden_roots=forbidden_roots,
        )
        # Run strace outside bubblewrap.  The tracer owns the only writable
        # handle to this parent-side path; bubblewrap closes any inherited
        # non-stdio descriptor before the predictor is exec'd, and the trace
        # path is not mounted into the sandbox.
        command = [
            str(strace_path),
            "-f",
            "-qq",
            "-yy",
            "-s",
            "4096",
            "-e",
            "trace=execve,open,openat,openat2",
            "-o",
            str(temp_trace),
            *bwrap_command,
        ]
        completed = command_runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception:
        if temp_trace.exists():
            os.chmod(temp_trace, 0o444)
        raise

    trace_bytes = _read_regular_nofollow(temp_trace)
    trace_sha256 = _write_exclusive_readonly(output / TRACE_NAME, trace_bytes)
    os.chmod(temp_trace, 0o600)
    temp_trace.unlink()
    try:
        trace_text = trace_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Phase2IsolatedRunnerError("strace output is not UTF-8") from exc
    predictor_trace_text = _predictor_trace_suffix(
        trace_text, expected_executable=Path(python_executable).resolve(strict=True)
    )
    additional_execves = parse_successful_execve_trace(predictor_trace_text)
    if additional_execves:
        raise Phase2IsolatedRunnerError(
            "bound predictor launched an additional successful execve: "
            + ", ".join(str(item["executable"]) for item in additional_execves)
        )
    ledger = parse_successful_open_trace(predictor_trace_text)
    audit_core = audit_open_ledger(
        ledger,
        system_read_roots=system_read_roots,
        forbidden_project_roots=forbidden_project_roots,
        sealed_package_members=sealed_package_members,
        required_package_members=required_package_members,
    )

    stdout_text = str(completed.stdout or "")
    stderr_text = str(completed.stderr or "")
    predictor_result: dict[str, Any] | None = None
    if int(completed.returncode) == 0:
        predictor_result = _parse_predictor_stdout(stdout_text)
    stdout_receipt = {
        "schema": "cvs.phase2.predictor_stdout_receipt.v1",
        "status": "PASS" if predictor_result is not None else "FAIL",
        "returncode": int(completed.returncode),
        "stdout_sha256": _sha256_bytes(stdout_text.encode("utf-8")),
        "stdout_size_bytes": len(stdout_text.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(stderr_text.encode("utf-8")),
        "stderr_size_bytes": len(stderr_text.encode("utf-8")),
        "request_sha256": request_sha256,
        "predictor_result": predictor_result,
    }
    stdout_receipt_sha256 = _write_json_exclusive_readonly(
        output / STDOUT_RECEIPT_NAME, stdout_receipt
    )

    prediction_artifact_sha256: str | None = None
    prediction_seal_sha256: str | None = None
    if predictor_result is not None:
        artifact_path = output / output_relative.name
        verified = verify_prediction_artifact(
            artifact_path,
            expected_artifact_sha256=str(predictor_result["artifact_sha256"]),
            expected_seal_sha256=str(predictor_result["seal_sha256"]),
        )
        prediction_artifact_sha256 = str(verified["artifact_sha256"])
        prediction_seal_sha256 = str(verified["seal_sha256"])
        if predictor_result["request_sha256"] != request_sha256:
            raise Phase2IsolatedRunnerError("predictor stdout request digest mismatch")
        manifest = verified["manifest"]
        expected_stage = "Stage2-B" if request["stage"] == "stage2b" else "Stage2-C"
        binding_pairs = {
            "stage": expected_stage,
            "row_id": request["row_id"],
            "receiver": request["receiver"],
            "k_shot": request["k_shot"],
            "candidate_lock_sha256": request["candidate_lock_sha256"],
            "package_root_sha256": request["package_root_sha256"],
            "package_seal_sha256": package_seal_sha256,
        }
        if any(manifest[key] != value for key, value in binding_pairs.items()):
            raise Phase2IsolatedRunnerError("prediction manifest/request binding mismatch")

    audit_document = {
        "schema": "cvs.phase2.filesystem_access_audit.v1",
        "status": audit_core["status"],
        "request_sha256": request_sha256,
        "pre_run_evidence_sha256": _sha256_bytes(_canonical_json(pre_evidence)),
        "trace_sha256": trace_sha256,
        "predictor_stdout_receipt_sha256": stdout_receipt_sha256,
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_seal_sha256": prediction_seal_sha256,
        "trace_scope": "after_bound_predictor_python_execve",
        "predictor_python_executable": str(Path(python_executable).resolve(strict=True)),
        **{key: value for key, value in audit_core.items() if key != "status"},
    }
    filesystem_audit_sha256 = _write_json_exclusive_readonly(
        output / AUDIT_NAME, audit_document
    )
    if int(completed.returncode) != 0:
        raise Phase2IsolatedRunnerError(
            f"isolated predictor failed with return code {completed.returncode}"
        )
    if audit_core["status"] != "PASS":
        raise Phase2IsolatedRunnerError("actual opened-file ledger violates the runtime allowlist")
    assert prediction_artifact_sha256 is not None and prediction_seal_sha256 is not None

    verified_after_run = verify_phase2_pre_run_evidence(
        evidence_root=pre_run_evidence_root,
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        bwrap_executable=bwrap,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_roots,
        expected_evidence=pre_evidence,
    )
    if verified_after_run["binding_sha256"] != verified_pre_run["binding_sha256"]:
        raise Phase2IsolatedRunnerError("pre-run evidence binding changed during prediction")

    post_evidence = {
        **pre_evidence,
        "filesystem_access_audit_sha256": filesystem_audit_sha256,
        "filesystem_access_audit_status": "PASS",
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_seal_sha256": prediction_seal_sha256,
    }
    if set(post_evidence) != set(POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise Phase2IsolatedRunnerError("post-run evidence does not use the exact field set")
    validate_phase2_contract(
        {**request, "phase2_runtime_isolation_evidence": post_evidence},
        evidence_phase="post_run",
    )
    post_contract_sha256 = _sha256_bytes(_canonical_json(post_evidence))
    diagnostic_post_evidence = {
        "schema": "cvs.phase2.diagnostic_post_run_runtime_evidence.v1",
        "status": "LOCAL_DIAGNOSTIC_PASS",
        "formal_launch_authority": False,
        "formal_launch_blockers": list(FORMAL_LAUNCH_BLOCKERS),
        "protocol_valid_claim_allowed": False,
        "formal_post_run_contract_evidence": post_evidence,
        "formal_post_run_contract_sha256": post_contract_sha256,
    }
    diagnostic_post_evidence_sha256 = _write_json_exclusive_readonly(
        output / POST_EVIDENCE_NAME, diagnostic_post_evidence
    )
    return {
        "schema": "cvs.phase2.isolated_runner_result.v1",
        "status": "LOCAL_DIAGNOSTIC_PASS",
        "formal_launch_authority": False,
        "formal_launch_blockers": list(FORMAL_LAUNCH_BLOCKERS),
        "command": command,
        "request_sha256": request_sha256,
        "trace": str(output / TRACE_NAME),
        "trace_sha256": trace_sha256,
        "filesystem_access_audit": str(output / AUDIT_NAME),
        "filesystem_access_audit_sha256": filesystem_audit_sha256,
        "predictor_stdout_receipt": str(output / STDOUT_RECEIPT_NAME),
        "predictor_stdout_receipt_sha256": stdout_receipt_sha256,
        "diagnostic_post_run_runtime_evidence": str(output / POST_EVIDENCE_NAME),
        "diagnostic_post_run_runtime_evidence_sha256": diagnostic_post_evidence_sha256,
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_seal_sha256": prediction_seal_sha256,
    }


__all__ = [
    "AUDIT_NAME",
    "POST_EVIDENCE_NAME",
    "STDOUT_RECEIPT_NAME",
    "TRACE_NAME",
    "Phase2IsolatedRunnerError",
    "audit_open_ledger",
    "build_production_predictor_argv",
    "execute_phase2_isolated",
    "parse_successful_execve_trace",
    "parse_successful_open_trace",
]
