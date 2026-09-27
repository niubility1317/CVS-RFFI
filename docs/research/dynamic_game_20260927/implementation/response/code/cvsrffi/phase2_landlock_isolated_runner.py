"""Landlock+seccomp execution boundary for one diagnostic Stage2 predictor."""

from __future__ import annotations

import json
import os
import posixpath
import re
import secrets
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .phase2_isolated_runner import (
    AUDIT_NAME,
    POST_EVIDENCE_NAME,
    STDOUT_RECEIPT_NAME,
    TRACE_NAME,
    Phase2IsolatedRunnerError,
    _bound_exec_index,
    _canonical_json,
    _parse_predictor_stdout,
    _predictor_trace_suffix,
    _read_regular_nofollow,
    _sha256_bytes,
    _trace_pid_and_body,
    _write_exclusive_readonly,
    _write_json_exclusive_readonly,
    audit_open_ledger,
    parse_successful_execve_trace,
    parse_successful_open_trace,
)
from .phase2_landlock_pre_run_evidence import (
    LANDLOCK_FORMAL_BLOCKERS,
    LANDLOCK_ISOLATION_MODE,
    verify_phase2_landlock_pre_run_evidence,
)
from .phase2_runtime_contract import (
    POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    validate_phase2_contract,
    validate_predictor_request,
)
from .stage2_prediction_artifact import verify_prediction_artifact


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RESULT_RE = re.compile(r"\)\s+=\s+(-?\d+)(?:\s|$)")
_NETWORK_SYSCALL_RE = re.compile(
    r"\b(socket|socketpair|connect|bind|listen|accept|accept4|sendto|sendmsg|"
    r"sendmmsg|recvfrom|recvmsg|recvmmsg)\("
)


def _read_request(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_nofollow(path)
    try:
        request = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase2IsolatedRunnerError("request is not UTF-8 JSON") from exc
    if not isinstance(request, dict):
        raise Phase2IsolatedRunnerError("request root must be an object")
    validate_predictor_request(request)
    return request, payload


def _successful_line_index(
    lines: Sequence[str],
    *,
    start: int,
    pattern: str,
    context: str,
) -> int:
    compiled = re.compile(pattern)
    for index in range(start, len(lines)):
        _pid, body = _trace_pid_and_body(lines[index])
        result = _RESULT_RE.search(body)
        if (
            compiled.search(body) is not None
            and result is not None
            and int(result.group(1)) >= 0
        ):
            return index
    raise Phase2IsolatedRunnerError(f"strace lacks successful {context}")


def audit_landlock_lifecycle(
    trace_text: str,
    *,
    expected_executable: str | Path,
    expected_entrypoint: str | Path,
) -> dict[str, Any]:
    """Prove that isolation was installed before the bound predictor exec."""

    lines = trace_text.splitlines()
    ruleset = _successful_line_index(
        lines,
        start=0,
        pattern=r"\blandlock_create_ruleset\(",
        context="Landlock ruleset creation",
    )
    add_rule = _successful_line_index(
        lines,
        start=ruleset + 1,
        pattern=r"\blandlock_add_rule\(",
        context="Landlock path rule",
    )
    no_new_privs = _successful_line_index(
        lines,
        start=add_rule + 1,
        pattern=r"\bprctl\(PR_SET_NO_NEW_PRIVS,\s*1\b",
        context="PR_SET_NO_NEW_PRIVS",
    )
    restrict_self = _successful_line_index(
        lines,
        start=no_new_privs + 1,
        pattern=r"\blandlock_restrict_self\(",
        context="Landlock restrict_self",
    )
    seccomp = _successful_line_index(
        lines,
        start=restrict_self + 1,
        pattern=r"\bseccomp\(SECCOMP_SET_MODE_FILTER\b",
        context="seccomp filter load",
    )
    predictor_exec = _bound_exec_index(
        trace_text,
        expected_executable=expected_executable,
        expected_entrypoint=expected_entrypoint,
    )
    if predictor_exec <= seccomp:
        raise Phase2IsolatedRunnerError(
            "predictor execve occurred before Landlock+seccomp activation"
        )
    return {
        "status": "PASS",
        "ruleset_create_line": ruleset + 1,
        "first_path_rule_line": add_rule + 1,
        "no_new_privileges_line": no_new_privs + 1,
        "restrict_self_line": restrict_self + 1,
        "seccomp_filter_line": seccomp + 1,
        "predictor_exec_line": predictor_exec + 1,
        "order": [
            "landlock_create_ruleset",
            "landlock_add_rule",
            "PR_SET_NO_NEW_PRIVS",
            "landlock_restrict_self",
            "seccomp_filter",
            "predictor_execve",
        ],
    }


def _map_host_path(path: str, mapping: Mapping[str, str]) -> str:
    normalized = posixpath.normpath(path)
    for host, logical in sorted(
        (
            (posixpath.normpath(str(source)), posixpath.normpath(str(target)))
            for source, target in mapping.items()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if normalized == host:
            return logical
        if normalized.startswith(host.rstrip("/") + "/"):
            relative = normalized[len(host.rstrip("/")) + 1 :]
            return posixpath.normpath(posixpath.join(logical, relative))
    return normalized


def map_open_ledger(
    ledger: Iterable[Mapping[str, Any]],
    *,
    host_to_logical: Mapping[str, str],
) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for raw in ledger:
        path = _map_host_path(str(raw["path"]), host_to_logical)
        row = aggregate.setdefault(
            path,
            {"path": path, "successful_open_count": 0, "syscalls": set()},
        )
        row["successful_open_count"] += int(raw["successful_open_count"])
        row["syscalls"].update(str(value) for value in raw["syscalls"])
    return [
        {
            "path": path,
            "successful_open_count": aggregate[path]["successful_open_count"],
            "syscalls": sorted(aggregate[path]["syscalls"]),
        }
        for path in sorted(aggregate)
    ]


def _network_attempts(trace_text: str) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for line_number, raw in enumerate(trace_text.splitlines(), 1):
        _pid, body = _trace_pid_and_body(raw)
        match = _NETWORK_SYSCALL_RE.search(body)
        if match is not None:
            attempts.append({"line_number": line_number, "syscall": match.group(1)})
    return attempts


def _launcher_command(
    *,
    python_executable: Path,
    launcher: Path,
    runtime_root: Path,
    package_root: Path,
    detached_seal: Path,
    request_json: Path,
    output_root: Path,
    predictor_entry: Path,
    expected_seal_sha256: str,
    system_read_roots: Sequence[Path],
    gpu_devices: Sequence[Path],
    device: str,
    batch_size: int,
) -> list[str]:
    command = [
        str(python_executable),
        str(launcher),
        "--runtime-root",
        str(runtime_root),
        "--package-root",
        str(package_root),
        "--detached-seal",
        str(detached_seal),
        "--request-json",
        str(request_json),
        "--output-root",
        str(output_root),
        "--python-executable",
        str(python_executable),
        "--predictor-entry",
        str(predictor_entry),
        "--expected-seal-sha256",
        expected_seal_sha256,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    for root in system_read_roots:
        command.extend(["--system-read-root", str(root)])
    for path in gpu_devices:
        command.extend(["--gpu-device", str(path)])
    return command


def execute_phase2_landlock_isolated(
    *,
    landlock_launcher: str | Path,
    landlock_policy_module: str | Path,
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
    """Execute one fail-closed diagnostic cell with equivalent isolation."""

    output = Path(output_root).resolve(strict=True)
    if Path(output_root).is_symlink() or not output.is_dir():
        raise Phase2IsolatedRunnerError("output root must be a non-symlink directory")
    if any(output.iterdir()):
        raise Phase2IsolatedRunnerError("Landlock output root must be empty")
    if not forbidden_roots:
        raise Phase2IsolatedRunnerError(
            "Landlock runner requires at least one separate scorer/truth root"
        )
    if (
        _SHA256_RE.fullmatch(str(expected_package_seal_sha256).lower())
        is None
    ):
        raise Phase2IsolatedRunnerError("expected package seal SHA256 is invalid")

    request_path = Path(request_json).resolve(strict=True)
    request, request_bytes = _read_request(Path(request_json))
    request_sha256 = _sha256_bytes(request_bytes)
    pre_evidence = request["phase2_runtime_isolation_evidence"]
    if set(pre_evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise Phase2IsolatedRunnerError(
            "request pre-run evidence does not use the exact 9-field schema"
        )
    if pre_evidence["os_isolation_mode"] != LANDLOCK_ISOLATION_MODE:
        raise Phase2IsolatedRunnerError(
            "Landlock runner requires equivalent_verified_isolation evidence"
        )

    launcher = Path(landlock_launcher).resolve(strict=True)
    policy = Path(landlock_policy_module).resolve(strict=True)
    python = Path(python_executable).resolve(strict=True)
    strace = Path(strace_executable).resolve(strict=True)
    package = Path(package_root).resolve(strict=True)
    seal = Path(detached_seal).resolve(strict=True)
    system_roots = [Path(value).resolve(strict=True) for value in system_read_roots]
    gpu = [Path(value).resolve(strict=True) for value in gpu_devices]
    verified_pre = verify_phase2_landlock_pre_run_evidence(
        evidence_root=pre_run_evidence_root,
        runtime_closure_root=runtime_closure_root,
        package_root=package,
        detached_seal=seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        landlock_launcher=launcher,
        landlock_policy_module=policy,
        strace_executable=strace,
        python_executable=python,
        system_read_roots=system_roots,
        forbidden_scorer_truth_roots=forbidden_roots,
        expected_evidence=pre_evidence,
    )
    runtime_root = Path(verified_pre["runtime_root"]).resolve(strict=True)
    verified_package_members = list(
        verified_pre.get("sealed_package_members", [])
    )
    if sealed_package_members is None:
        sealed_package_members = verified_package_members
    if not required_package_members:
        required_package_members = verified_package_members
    predictor_entry = (
        runtime_root / "scripts" / "run_cvs_stage2_predictor.py"
    ).resolve(strict=True)
    package_seal_sha256 = str(pre_evidence["sealed_inference_package_sha256"])
    if package_seal_sha256 != str(expected_package_seal_sha256).lower():
        raise Phase2IsolatedRunnerError(
            "request package seal is not the external trusted seal"
        )
    if _sha256_bytes(_read_regular_nofollow(seal)) != package_seal_sha256:
        raise Phase2IsolatedRunnerError(
            "detached package seal does not match request evidence"
        )
    output_relative = PurePosixPath(
        str(request["output_contract"]["relative_path"])
    )
    if output_relative.parent != PurePosixPath(".") or not output_relative.name:
        raise Phase2IsolatedRunnerError(
            "prediction artifact must be a direct output child"
        )

    temp_trace = (
        output.parent
        / f".phase2_landlock_trace.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    launcher_command = _launcher_command(
        python_executable=python,
        launcher=launcher,
        runtime_root=runtime_root,
        package_root=package,
        detached_seal=seal,
        request_json=request_path,
        output_root=output,
        predictor_entry=predictor_entry,
        expected_seal_sha256=package_seal_sha256,
        system_read_roots=system_roots,
        gpu_devices=gpu,
        device=device,
        batch_size=batch_size,
    )
    command = [
        str(strace),
        "-f",
        "-qq",
        "-yy",
        "-s",
        "4096",
        "-e",
        (
            "trace=execve,open,openat,openat2,prctl,landlock_create_ruleset,"
            "landlock_add_rule,landlock_restrict_self,seccomp,socket,socketpair,"
            "connect,bind,listen,accept,accept4,sendto,sendmsg,sendmmsg,recvfrom,"
            "recvmsg,recvmmsg"
        ),
        "-o",
        str(temp_trace),
        *launcher_command,
    ]
    try:
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
    lifecycle = audit_landlock_lifecycle(
        trace_text,
        expected_executable=python,
        expected_entrypoint=predictor_entry,
    )
    predictor_trace = _predictor_trace_suffix(
        trace_text,
        expected_executable=python,
        expected_entrypoint=predictor_entry,
    )
    additional_execves = parse_successful_execve_trace(predictor_trace)
    if additional_execves:
        raise Phase2IsolatedRunnerError(
            "bound predictor launched an additional successful execve: "
            + ", ".join(str(item["executable"]) for item in additional_execves)
        )
    network_attempts = _network_attempts(predictor_trace)
    if network_attempts:
        raise Phase2IsolatedRunnerError(
            "bound predictor attempted a network syscall after isolation"
        )
    host_ledger = parse_successful_open_trace(
        predictor_trace, cwd=str(runtime_root)
    )
    host_to_logical = {
        str(runtime_root): "/runtime/code",
        str(package): "/sealed/package",
        str(seal): "/sealed/package.seal.json",
        str(request_path): "/sealed/request.json",
        str(output): "/output",
    }
    ledger = map_open_ledger(
        host_ledger,
        host_to_logical=host_to_logical,
    )
    audit_core = audit_open_ledger(
        ledger,
        system_read_roots=system_roots,
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
            raise Phase2IsolatedRunnerError(
                "predictor stdout request digest mismatch"
            )
        manifest = verified["manifest"]
        expected_stage = (
            "Stage2-B" if request["stage"] == "stage2b" else "Stage2-C"
        )
        bindings = {
            "stage": expected_stage,
            "row_id": request["row_id"],
            "receiver": request["receiver"],
            "k_shot": request["k_shot"],
            "candidate_lock_sha256": request["candidate_lock_sha256"],
            "package_root_sha256": request["package_root_sha256"],
            "package_seal_sha256": package_seal_sha256,
        }
        if any(manifest[key] != value for key, value in bindings.items()):
            raise Phase2IsolatedRunnerError(
                "prediction manifest/request binding mismatch"
            )

    audit_document = {
        "schema": "cvs.phase2.filesystem_access_audit.v1",
        "status": audit_core["status"],
        "request_sha256": request_sha256,
        "pre_run_evidence_sha256": _sha256_bytes(
            _canonical_json(pre_evidence)
        ),
        "trace_sha256": trace_sha256,
        "predictor_stdout_receipt_sha256": stdout_receipt_sha256,
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_seal_sha256": prediction_seal_sha256,
        "trace_scope": "full_launcher_lifecycle_and_after_bound_predictor_execve",
        "predictor_python_executable": str(python),
        "predictor_entrypoint": str(predictor_entry),
        "landlock_lifecycle_audit": lifecycle,
        "network_syscall_attempts_after_predictor_exec": network_attempts,
        "host_to_logical_path_mapping": host_to_logical,
        "host_opened_file_ledger": host_ledger,
        **{key: value for key, value in audit_core.items() if key != "status"},
    }
    filesystem_audit_sha256 = _write_json_exclusive_readonly(
        output / AUDIT_NAME, audit_document
    )
    if int(completed.returncode) != 0:
        raise Phase2IsolatedRunnerError(
            f"Landlock predictor failed with return code {completed.returncode}"
        )
    if audit_core["status"] != "PASS":
        raise Phase2IsolatedRunnerError(
            "actual opened-file ledger violates the runtime allowlist"
        )
    assert prediction_artifact_sha256 is not None
    assert prediction_seal_sha256 is not None

    verified_after = verify_phase2_landlock_pre_run_evidence(
        evidence_root=pre_run_evidence_root,
        runtime_closure_root=runtime_closure_root,
        package_root=package,
        detached_seal=seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        landlock_launcher=launcher,
        landlock_policy_module=policy,
        strace_executable=strace,
        python_executable=python,
        system_read_roots=system_roots,
        forbidden_scorer_truth_roots=forbidden_roots,
        expected_evidence=pre_evidence,
    )
    if verified_after["binding_sha256"] != verified_pre["binding_sha256"]:
        raise Phase2IsolatedRunnerError(
            "pre-run evidence binding changed during prediction"
        )

    post_evidence = {
        **pre_evidence,
        "filesystem_access_audit_sha256": filesystem_audit_sha256,
        "filesystem_access_audit_status": "PASS",
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_seal_sha256": prediction_seal_sha256,
    }
    if set(post_evidence) != set(POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise Phase2IsolatedRunnerError(
            "post-run evidence does not use the exact field set"
        )
    validate_phase2_contract(
        {**request, "phase2_runtime_isolation_evidence": post_evidence},
        evidence_phase="post_run",
    )
    post_contract_sha256 = _sha256_bytes(_canonical_json(post_evidence))
    diagnostic_post = {
        "schema": "cvs.phase2.diagnostic_post_run_runtime_evidence.v1",
        "status": "LOCAL_DIAGNOSTIC_PASS",
        "formal_launch_authority": False,
        "formal_launch_blockers": list(LANDLOCK_FORMAL_BLOCKERS),
        "protocol_valid_claim_allowed": False,
        "formal_post_run_contract_evidence": post_evidence,
        "formal_post_run_contract_sha256": post_contract_sha256,
    }
    diagnostic_post_sha256 = _write_json_exclusive_readonly(
        output / POST_EVIDENCE_NAME, diagnostic_post
    )
    return {
        "schema": "cvs.phase2.landlock_isolated_runner_result.v1",
        "status": "LOCAL_DIAGNOSTIC_PASS",
        "formal_launch_authority": False,
        "formal_launch_blockers": list(LANDLOCK_FORMAL_BLOCKERS),
        "command": command,
        "request_sha256": request_sha256,
        "trace": str(output / TRACE_NAME),
        "trace_sha256": trace_sha256,
        "filesystem_access_audit": str(output / AUDIT_NAME),
        "filesystem_access_audit_sha256": filesystem_audit_sha256,
        "predictor_stdout_receipt": str(output / STDOUT_RECEIPT_NAME),
        "predictor_stdout_receipt_sha256": stdout_receipt_sha256,
        "diagnostic_post_run_runtime_evidence": str(
            output / POST_EVIDENCE_NAME
        ),
        "diagnostic_post_run_runtime_evidence_sha256": (
            diagnostic_post_sha256
        ),
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_seal_sha256": prediction_seal_sha256,
    }


__all__ = [
    "audit_landlock_lifecycle",
    "execute_phase2_landlock_isolated",
    "map_open_ledger",
]
