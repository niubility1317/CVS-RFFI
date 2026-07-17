"""Audited bubblewrap/strace runner for fixed SOMP-H process entries."""

from __future__ import annotations

import json
import hashlib
import os
import secrets
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from cvsrffi.phase2_bwrap_policy import build_phase2_bwrap_command
from cvsrffi.phase2_isolated_runner import (
    _predictor_trace_suffix,
    _read_regular_nofollow,
    _write_exclusive_readonly,
    _write_json_exclusive_readonly,
    audit_open_ledger,
    parse_successful_execve_trace,
    parse_successful_open_trace,
)
from cvsrffi.phase2_pre_run_evidence import (
    SOMPH_APPLY_PROFILE,
    SOMPH_ENROLLMENT_PROFILE,
    verify_phase2_pre_run_evidence,
)
from cvsrffi.somph_head_artifact import verify_somph_head_artifact
from cvsrffi.somph_prediction_artifact import verify_somph_prediction_artifact
from cvsrffi.somph_predictor_entry import (
    APPLY_RECEIPT_NAME,
    ENROLLMENT_RECEIPT_NAME,
)
from cvsrffi.somph_diagnostic_bundle_loader import (
    load_verified_somph_head_capsule,
    preflight_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import (
    APPLY_ONLY,
    ENROLLMENT_ONLY,
)
from cvsrffi.somph_runtime_request import (
    validate_somph_apply_request,
    validate_somph_enrollment_request,
)
from cvsrffi.stage2_predictor_runtime import load_json_artifact_same_fd


TRACE_NAME = "somph_filesystem_access_trace.log"
AUDIT_NAME = "somph_filesystem_access_audit.json"
STDOUT_RECEIPT_NAME = "somph_stdout_receipt.json"

_ENTRYPOINT_BY_PROFILE = {
    ENROLLMENT_ONLY: "/runtime/code/scripts/run_cvs_somph_enrollment.py",
    APPLY_ONLY: "/runtime/code/scripts/run_cvs_somph_apply.py",
}


class SomphIsolatedRunnerError(RuntimeError):
    """Raised when a fixed SOMP-H process violates its OS boundary."""


def _request(path: str | Path, *, profile: str) -> tuple[dict[str, Any], str]:
    raw = _read_regular_nofollow(Path(path))
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SomphIsolatedRunnerError(
            "SOMP-H isolated request is not UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SomphIsolatedRunnerError("SOMP-H isolated request is not an object")
    try:
        safe = (
            validate_somph_enrollment_request(value)
            if profile == ENROLLMENT_ONLY
            else validate_somph_apply_request(value)
        )
    except ValueError as exc:
        raise SomphIsolatedRunnerError(str(exc)) from exc
    return safe, hashlib.sha256(raw).hexdigest()


def _fixed_argv(*, profile: str, expected_seal_sha256: str) -> list[str]:
    return [
        _ENTRYPOINT_BY_PROFILE[profile],
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
    ]


def _parse_stdout(text: str, *, profile: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SomphIsolatedRunnerError(
            "SOMP-H process stdout is not one JSON object"
        ) from exc
    if not isinstance(payload, dict) or payload.get("profile") != profile:
        raise SomphIsolatedRunnerError("SOMP-H stdout profile drift")
    expected = {
        "schema",
        "profile",
        "request_sha256",
        "execution_receipt_sha256",
        "formal_launch_authority",
        "diagnostic_only",
    }
    expected |= (
        {
            "head_output_leaf",
            "head_capsule_sha256",
            "enrollment_binding_sha256",
        }
        if profile == ENROLLMENT_ONLY
        else {"prediction_output_leaf", "artifact_sha256", "seal_sha256"}
    )
    expected_schema = (
        "cvs.phase2.somph_enrollment_stdout.v1"
        if profile == ENROLLMENT_ONLY
        else "cvs.phase2.somph_apply_stdout.v1"
    )
    if (
        set(payload) != expected
        or payload.get("schema") != expected_schema
        or payload["formal_launch_authority"] is not False
        or payload["diagnostic_only"] is not True
    ):
        raise SomphIsolatedRunnerError("SOMP-H stdout exact trust fields missing")
    for field in (
        "request_sha256",
        "execution_receipt_sha256",
        "head_capsule_sha256",
        "enrollment_binding_sha256",
        "artifact_sha256",
        "seal_sha256",
    ):
        if field in payload and (
            not isinstance(payload[field], str)
            or len(payload[field]) != 64
            or any(character not in "0123456789abcdef" for character in payload[field])
        ):
            raise SomphIsolatedRunnerError(
                f"SOMP-H stdout digest is invalid: {field}"
            )
    return payload


def _isolation_profile(profile: str) -> str:
    return (
        SOMPH_ENROLLMENT_PROFILE
        if profile == ENROLLMENT_ONLY
        else SOMPH_APPLY_PROFILE
    )


def _trusted_sha256(value: str, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SomphIsolatedRunnerError(
            f"{context} must be an external lowercase SHA256 trust root"
        )
    return value


def _execution_receipt(
    output: Path,
    *,
    profile: str,
    stdout: dict[str, Any],
    request_sha256: str,
    manifest: dict[str, Any],
    expected_seal: str,
    output_sha256: str,
    output_seal_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    receipt_name = (
        ENROLLMENT_RECEIPT_NAME
        if profile == ENROLLMENT_ONLY
        else APPLY_RECEIPT_NAME
    )
    raw = _read_regular_nofollow(output / receipt_name)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != stdout["execution_receipt_sha256"]:
        raise SomphIsolatedRunnerError(
            "SOMP-H execution receipt SHA256 drift"
        )
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SomphIsolatedRunnerError(
            "SOMP-H execution receipt is not UTF-8 JSON"
        ) from exc
    if not isinstance(receipt, dict):
        raise SomphIsolatedRunnerError(
            "SOMP-H execution receipt root is not an object"
        )
    canonical = (
        json.dumps(
            receipt,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise SomphIsolatedRunnerError(
            "SOMP-H execution receipt is not canonical immutable JSON"
        )
    common = {
        "schema",
        "status",
        "diagnostic_only",
        "formal_launch_authority",
        "formal_metric_claim_allowed",
        "request_sha256",
        "package_root_sha256",
        "package_seal_sha256",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "overlay_provenance_sha256",
        "preopen_audit",
        "resource",
        "peak_cuda_memory_bytes",
    }
    specific = (
        {"head_capsule_sha256", "enrollment_binding_sha256"}
        if profile == ENROLLMENT_ONLY
        else {
            "head_capsule_sha256",
            "enrollment_binding_sha256",
            "prediction_artifact_sha256",
            "prediction_seal_sha256",
        }
    )
    expected_schema = (
        "cvs.phase2.somph_enrollment_execution_receipt.v1"
        if profile == ENROLLMENT_ONLY
        else "cvs.phase2.somph_apply_execution_receipt.v1"
    )
    if (
        set(receipt) != common | specific
        or receipt.get("schema") != expected_schema
        or receipt.get("status") != "LOCAL_PROTOCOL_REPAIR_REQUIRED"
        or receipt.get("diagnostic_only") is not True
        or receipt.get("formal_launch_authority") is not False
        or receipt.get("formal_metric_claim_allowed") is not False
        or receipt.get("request_sha256") != request_sha256
        or receipt.get("package_root_sha256")
        != manifest["package_root_sha256"]
        or receipt.get("package_seal_sha256") != expected_seal
        or receipt.get("phase1_checkpoint_sha256")
        != manifest["phase1_checkpoint_sha256"]
        or receipt.get("feature_runtime_sha256")
        != manifest["feature_runtime_sha256"]
        or receipt.get("method_lock_sha256") != manifest["method_lock_sha256"]
        or receipt.get("overlay_provenance_sha256")
        != manifest["overlay_provenance_sha256"]
        or not isinstance(receipt.get("preopen_audit"), dict)
        or not isinstance(receipt.get("resource"), dict)
        or not isinstance(receipt.get("peak_cuda_memory_bytes"), int)
        or isinstance(receipt.get("peak_cuda_memory_bytes"), bool)
        or receipt["peak_cuda_memory_bytes"] < 0
    ):
        raise SomphIsolatedRunnerError(
            "SOMP-H execution receipt trust binding drift"
        )
    resource_schema = (
        "cvs.phase2.somph_enrollment_resource_receipt.v1"
        if profile == ENROLLMENT_ONLY
        else "cvs.phase2.somph_apply_resource_receipt.v1"
    )
    if receipt["resource"].get("schema") != resource_schema:
        raise SomphIsolatedRunnerError(
            "SOMP-H execution resource receipt schema drift"
        )
    if profile == ENROLLMENT_ONLY:
        if (
            receipt["head_capsule_sha256"] != output_sha256
            or receipt["head_capsule_sha256"]
            != stdout["head_capsule_sha256"]
            or receipt["enrollment_binding_sha256"]
            != stdout["enrollment_binding_sha256"]
        ):
            raise SomphIsolatedRunnerError(
                "SOMP-H enrollment receipt/output binding drift"
            )
    elif (
        receipt["head_capsule_sha256"] != manifest["head_capsule_sha256"]
        or receipt["enrollment_binding_sha256"]
        != manifest["head_enrollment_binding_sha256"]
        or receipt["prediction_artifact_sha256"] != output_sha256
        or receipt["prediction_artifact_sha256"] != stdout["artifact_sha256"]
        or receipt["prediction_seal_sha256"] != output_seal_sha256
        or receipt["prediction_seal_sha256"] != stdout["seal_sha256"]
    ):
        raise SomphIsolatedRunnerError(
            "SOMP-H apply receipt/output binding drift"
        )
    return receipt, digest


def _audit_exact_output_members(
    output: Path, *, expected_names: set[str]
) -> None:
    actual: set[str] = set()
    for path in output.iterdir():
        if path.is_symlink() or not path.is_file():
            raise SomphIsolatedRunnerError(
                "SOMP-H output contains a non-regular member"
            )
        actual.add(path.name)
    if actual != expected_names:
        raise SomphIsolatedRunnerError(
            "SOMP-H output exact member allowlist mismatch"
        )


def _execute_somph_isolated_impl(
    *,
    profile: str,
    bwrap: str | Path,
    strace_executable: str | Path,
    runtime_closure_root: str | Path,
    pre_run_evidence_root: str | Path,
    expected_pre_run_binding_sha256: str,
    expected_runtime_closure_sha256: str,
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
    timeout_seconds: int | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run one enrollment-only or apply-only process; formal claims stay blocked."""

    if profile not in _ENTRYPOINT_BY_PROFILE:
        raise SomphIsolatedRunnerError("SOMP-H isolated profile invalid")
    request, request_sha256 = _request(request_json, profile=profile)
    expected_seal = str(expected_package_seal_sha256).lower()
    if request["package_seal_sha256"] != expected_seal:
        raise SomphIsolatedRunnerError(
            "SOMP-H request is detached from external package seal"
        )
    output_raw = Path(output_root)
    output = output_raw.resolve(strict=True)
    if output_raw.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise SomphIsolatedRunnerError(
            "SOMP-H isolated output root must be an empty non-symlink directory"
        )
    if not forbidden_roots:
        raise SomphIsolatedRunnerError(
            "SOMP-H runner requires physically separate forbidden scorer/truth roots"
        )
    isolation_profile = _isolation_profile(profile)
    expected_pre_run_binding = _trusted_sha256(
        expected_pre_run_binding_sha256,
        context="SOMP-H pre-run evidence binding",
    )
    expected_runtime_closure = _trusted_sha256(
        expected_runtime_closure_sha256,
        context="SOMP-H runtime closure",
    )
    verified_pre_run = verify_phase2_pre_run_evidence(
        evidence_root=pre_run_evidence_root,
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_seal,
        bwrap_executable=bwrap,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_roots,
        isolation_profile=isolation_profile,
    )
    if (
        verified_pre_run["binding_sha256"] != expected_pre_run_binding
        or verified_pre_run["runtime_code_sha256"] != expected_runtime_closure
    ):
        raise SomphIsolatedRunnerError(
            "SOMP-H pre-run evidence/runtime closure external trust root mismatch"
        )
    manifest, _seal, preopen = preflight_somph_predictor_bundle(
        package_root,
        detached_seal_path=detached_seal,
        expected_seal_sha256=expected_seal,
    )
    if manifest["profile"] != profile:
        raise SomphIsolatedRunnerError("SOMP-H package/profile mismatch")
    member_paths = [
        item["relative_path"] for item in manifest["members"]
    ] + ["package_manifest.json"]
    runtime_root = verified_pre_run["runtime_root"]
    entrypoint = _ENTRYPOINT_BY_PROFILE[profile]
    entry_source = Path(runtime_root) / entrypoint.removeprefix("/runtime/code/")
    if not entry_source.is_file():
        raise SomphIsolatedRunnerError("fixed SOMP-H entrypoint is absent from closure")

    temp_trace = output.parent / (
        f".somph_open_trace.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    strace_path = Path(strace_executable).resolve(strict=True)
    bwrap_command = build_phase2_bwrap_command(
        bwrap=str(Path(bwrap).resolve(strict=True)),
        runtime_root=runtime_root,
        package_root=package_root,
        detached_seal=detached_seal,
        request_json=request_json,
        output_root=output,
        python_executable=python_executable,
        predictor_argv=_fixed_argv(
            profile=profile, expected_seal_sha256=expected_seal
        ),
        system_read_roots=system_read_roots,
        trusted_system_read_roots=verified_pre_run[
            "trusted_system_read_roots"
        ],
        gpu_devices=gpu_devices,
        forbidden_roots=forbidden_roots,
    )
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
    trace_bytes = _read_regular_nofollow(temp_trace)
    trace_sha256 = _write_exclusive_readonly(output / TRACE_NAME, trace_bytes)
    os.chmod(temp_trace, 0o600)
    temp_trace.unlink()
    try:
        trace_text = trace_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SomphIsolatedRunnerError("SOMP-H strace output is not UTF-8") from exc
    predictor_trace = _predictor_trace_suffix(
        trace_text,
        expected_executable=Path(python_executable).resolve(strict=True),
    )
    execves = parse_successful_execve_trace(predictor_trace)
    if execves:
        raise SomphIsolatedRunnerError(
            "SOMP-H process launched an additional successful execve"
        )
    ledger = parse_successful_open_trace(predictor_trace)
    audit_core = audit_open_ledger(
        ledger,
        system_read_roots=system_read_roots,
        forbidden_project_roots=forbidden_project_roots,
        sealed_package_members=member_paths,
        required_package_members=member_paths,
    )
    stdout_text = str(completed.stdout or "")
    stderr_text = str(completed.stderr or "")
    result = (
        _parse_stdout(stdout_text, profile=profile)
        if int(completed.returncode) == 0
        else None
    )
    if result is not None and result["request_sha256"] != request_sha256:
        raise SomphIsolatedRunnerError("SOMP-H stdout request digest mismatch")
    if result is not None and profile == ENROLLMENT_ONLY:
        method_descriptor = next(
            item for item in manifest["members"] if item["kind"] == "method_lock"
        )
        method_lock = load_json_artifact_same_fd(
            package_root, method_descriptor
        )
        verified_output = verify_somph_head_artifact(
            output / result["head_output_leaf"],
            method_lock=method_lock,
            expected_enrollment_binding_sha256=result[
                "enrollment_binding_sha256"
            ],
            expected_head_capsule_sha256=result["head_capsule_sha256"],
        )
        output_sha256 = verified_output["head_capsule_sha256"]
        output_seal_sha256 = None
    elif result is not None:
        verified_output = verify_somph_prediction_artifact(
            output / result["prediction_output_leaf"],
            expected_artifact_sha256=result["artifact_sha256"],
            expected_seal_sha256=result["seal_sha256"],
        )
        output_sha256 = verified_output["artifact_sha256"]
        output_seal_sha256 = verified_output["seal_sha256"]
    else:
        verified_output = None
        output_sha256 = None
        output_seal_sha256 = None
    if result is None:
        raise SomphIsolatedRunnerError(
            f"SOMP-H isolated process failed with return code {completed.returncode}"
        )
    assert output_sha256 is not None
    execution_receipt, execution_receipt_sha256 = _execution_receipt(
        output,
        profile=profile,
        stdout=result,
        request_sha256=request_sha256,
        manifest=manifest,
        expected_seal=expected_seal,
        output_sha256=output_sha256,
        output_seal_sha256=output_seal_sha256,
    )
    output_leaf = (
        result["head_output_leaf"]
        if profile == ENROLLMENT_ONLY
        else result["prediction_output_leaf"]
    )
    receipt_name = (
        ENROLLMENT_RECEIPT_NAME
        if profile == ENROLLMENT_ONLY
        else APPLY_RECEIPT_NAME
    )
    _audit_exact_output_members(
        output,
        expected_names={output_leaf, receipt_name, TRACE_NAME},
    )

    manifest_after, _seal_after, preopen_after = preflight_somph_predictor_bundle(
        package_root,
        detached_seal_path=detached_seal,
        expected_seal_sha256=expected_seal,
    )
    if (
        manifest_after["package_root_sha256"] != manifest["package_root_sha256"]
        or preopen_after["manifest_sha256"] != preopen["manifest_sha256"]
    ):
        raise SomphIsolatedRunnerError("SOMP-H package changed during execution")
    verified_after_run = verify_phase2_pre_run_evidence(
        evidence_root=pre_run_evidence_root,
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_seal,
        bwrap_executable=bwrap,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_roots,
        isolation_profile=isolation_profile,
    )
    if (
        verified_after_run["binding_sha256"]
        != expected_pre_run_binding
        or verified_after_run["runtime_code_sha256"]
        != expected_runtime_closure
    ):
        raise SomphIsolatedRunnerError(
            "SOMP-H pre-run evidence binding changed during execution"
        )
    if audit_core["status"] != "PASS":
        raise SomphIsolatedRunnerError(
            "SOMP-H opened-file ledger violates the exact allowlist"
        )
    stdout_receipt = {
        "schema": "cvs.phase2.somph_stdout_receipt.v2",
        "diagnostic_only": True,
        "status": "VALIDATED_LOCAL_SUBPROCESS_OUTPUT",
        "returncode": int(completed.returncode),
        "request_sha256": request_sha256,
        "stdout_sha256": hashlib.sha256(
            stdout_text.encode("utf-8")
        ).hexdigest(),
        "stderr_sha256": hashlib.sha256(
            stderr_text.encode("utf-8")
        ).hexdigest(),
        "execution_receipt_sha256": execution_receipt_sha256,
        "result": result,
    }
    stdout_receipt_sha256 = _write_json_exclusive_readonly(
        output / STDOUT_RECEIPT_NAME, stdout_receipt
    )
    audit = {
        "schema": "cvs.phase2.somph_filesystem_access_audit.v1",
        "diagnostic_only": True,
        "status": audit_core["status"],
        "control_state": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "profile": profile,
        "request_sha256": request_sha256,
        "runtime_closure_sha256": verified_pre_run["runtime_code_sha256"],
        "pre_run_evidence_binding_sha256": verified_pre_run[
            "binding_sha256"
        ],
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": expected_seal,
        "trace_sha256": trace_sha256,
        "stdout_receipt_sha256": stdout_receipt_sha256,
        "output_sha256": output_sha256,
        "output_seal_sha256": output_seal_sha256,
        "execution_receipt_sha256": execution_receipt_sha256,
        "trace_scope": "after_bound_somph_python_execve",
        **{key: value for key, value in audit_core.items() if key != "status"},
    }
    audit_sha256 = _write_json_exclusive_readonly(output / AUDIT_NAME, audit)
    _audit_exact_output_members(
        output,
        expected_names={
            output_leaf,
            receipt_name,
            TRACE_NAME,
            STDOUT_RECEIPT_NAME,
            AUDIT_NAME,
        },
    )
    return {
        "schema": "cvs.phase2.somph_isolated_runner_result.v1",
        "diagnostic_only": True,
        "status": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "profile": profile,
        "command": command,
        "request_sha256": request_sha256,
        "runtime_closure_sha256": verified_pre_run["runtime_code_sha256"],
        "pre_run_evidence_binding_sha256": verified_pre_run[
            "binding_sha256"
        ],
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": expected_seal,
        "output_sha256": output_sha256,
        "output_seal_sha256": output_seal_sha256,
        "execution_receipt": execution_receipt,
        "execution_receipt_sha256": execution_receipt_sha256,
        "filesystem_access_audit": str(output / AUDIT_NAME),
        "filesystem_access_audit_sha256": audit_sha256,
        "trace": str(output / TRACE_NAME),
        "trace_sha256": trace_sha256,
        "verified_output": verified_output,
    }


def execute_somph_isolated(
    *,
    profile: str,
    bwrap: str | Path,
    strace_executable: str | Path,
    runtime_closure_root: str | Path,
    pre_run_evidence_root: str | Path,
    expected_pre_run_binding_sha256: str,
    expected_runtime_closure_sha256: str,
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
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute the production path with the non-injectable subprocess runner."""

    return _execute_somph_isolated_impl(
        profile=profile,
        bwrap=bwrap,
        strace_executable=strace_executable,
        runtime_closure_root=runtime_closure_root,
        pre_run_evidence_root=pre_run_evidence_root,
        expected_pre_run_binding_sha256=expected_pre_run_binding_sha256,
        expected_runtime_closure_sha256=expected_runtime_closure_sha256,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        request_json=request_json,
        output_root=output_root,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        gpu_devices=gpu_devices,
        forbidden_roots=forbidden_roots,
        forbidden_project_roots=forbidden_project_roots,
        timeout_seconds=timeout_seconds,
        command_runner=subprocess.run,
    )


__all__ = [
    "APPLY_ONLY",
    "AUDIT_NAME",
    "ENROLLMENT_ONLY",
    "STDOUT_RECEIPT_NAME",
    "TRACE_NAME",
    "SomphIsolatedRunnerError",
    "execute_somph_isolated",
]
