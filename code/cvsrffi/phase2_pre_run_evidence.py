"""Build fail-closed Phase2 evidence for conditions verified before prediction.

This module deliberately does not claim that the sandbox has already run.
Actual mount and opened-file evidence is produced only by
``phase2_isolated_runner`` after the Linux ``bwrap`` process exits.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .phase2_runtime_closure import verify_phase2_runtime_closure
from .phase2_runtime_contract import PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS
from .stage2_predictor_bundle import (
    preflight_stage2_predictor_package,
    sha256_file,
)


ATTESTATION_NAME = "os_isolation_attestation.json"
PREOPEN_AUDIT_NAME = "preopen_audit_receipt.json"
PRE_RUN_EVIDENCE_NAME = "runtime_isolation_evidence.json"
PRE_RUN_BUNDLE_MEMBERS = (
    ATTESTATION_NAME,
    PREOPEN_AUDIT_NAME,
    PRE_RUN_EVIDENCE_NAME,
)
FORMAL_LAUNCH_BLOCKERS = (
    "TRUSTED_ADAPTER_HEAD_TTA_PROVENANCE_NOT_YET_BOUND",
    "FIXED_INODE_OR_DIFFERENT_UID_INPUT_SNAPSHOT_NOT_YET_PROVEN",
    "REAL_LINUX_ISOLATION_SMOKE_NOT_YET_PASSED",
)

PHASE2_BWRAP_POLICY_CONTRACT = {
    "schema": "cvs.phase2.bwrap_policy_contract.v1",
    "runtime_mount_target": "/runtime/code",
    "package_mount_target": "/sealed/package",
    "detached_seal_mount_target": "/sealed/package.seal.json",
    "request_mount_target": "/sealed/request.json",
    "single_write_root_target": "/output",
    "entrypoint": "/runtime/code/scripts/run_cvs_stage2_predictor.py",
    "required_isolation": ["unshare_all", "no_network", "cap_drop_all", "clearenv"],
    "required_trace_syscalls": ["open", "openat", "openat2"],
    "trace_sink": "parent_owned_inherited_fd_outside_output_mount",
    "post_run_open_ledger_required": True,
}

GENERIC_PREDICTOR_PROFILE = "stage2_predictor"
SOMPH_ENROLLMENT_PROFILE = "somph_enrollment_only"
SOMPH_APPLY_PROFILE = "somph_apply_only"
ISOLATION_PROFILES = frozenset(
    {
        GENERIC_PREDICTOR_PROFILE,
        SOMPH_ENROLLMENT_PROFILE,
        SOMPH_APPLY_PROFILE,
    }
)


class Phase2PreRunEvidenceError(ValueError):
    """Raised when a formal pre-run isolation condition is not satisfied."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _resolved_directory(path: str | Path, *, context: str) -> Path:
    source = Path(path)
    resolved = source.resolve(strict=True)
    if source.is_symlink() or not resolved.is_dir():
        raise Phase2PreRunEvidenceError(f"{context} must be a non-symlink directory")
    return resolved


def _executable_descriptor(path: str | Path, *, context: str) -> dict[str, Any]:
    source = Path(path)
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise Phase2PreRunEvidenceError(f"{context} must resolve to a regular file")
    if not os.access(resolved, os.X_OK):
        raise Phase2PreRunEvidenceError(f"{context} is not executable")
    return {
        "requested_path": str(source),
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _regular_descriptor(path: str | Path, *, context: str) -> dict[str, Any]:
    source = Path(path)
    resolved = source.resolve(strict=True)
    if source.is_symlink() or not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise Phase2PreRunEvidenceError(f"{context} must be a regular non-symlink file")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _policy_contract(isolation_profile: str) -> dict[str, Any]:
    if isolation_profile not in ISOLATION_PROFILES:
        raise Phase2PreRunEvidenceError("unsupported Phase2 isolation profile")
    entrypoint = {
        GENERIC_PREDICTOR_PROFILE: "/runtime/code/scripts/run_cvs_stage2_predictor.py",
        SOMPH_ENROLLMENT_PROFILE: "/runtime/code/scripts/run_cvs_somph_enrollment.py",
        SOMPH_APPLY_PROFILE: "/runtime/code/scripts/run_cvs_somph_apply.py",
    }[isolation_profile]
    return {
        **PHASE2_BWRAP_POLICY_CONTRACT,
        "schema": "cvs.phase2.bwrap_policy_contract.v2",
        "isolation_profile": isolation_profile,
        "entrypoint": entrypoint,
    }


def _controller_descriptors(
    isolation_profile: str,
) -> dict[str, dict[str, Any]]:
    code_root = Path(__file__).resolve().parents[1]
    members = {
        "phase2_pre_run_evidence": Path(__file__),
        "phase2_runtime_closure": Path(__file__).with_name("phase2_runtime_closure.py"),
        "phase2_bwrap_policy": Path(__file__).with_name("phase2_bwrap_policy.py"),
    }
    if isolation_profile == GENERIC_PREDICTOR_PROFILE:
        members.update(
            {
                "phase2_isolated_runner": Path(__file__).with_name(
                    "phase2_isolated_runner.py"
                ),
                "run_cvs_stage2_bwrap_isolated": (
                    code_root / "scripts/run_cvs_stage2_bwrap_isolated.py"
                ),
            }
        )
    else:
        entry_script = (
            "run_cvs_somph_enrollment.py"
            if isolation_profile == SOMPH_ENROLLMENT_PROFILE
            else "run_cvs_somph_apply.py"
        )
        members.update(
            {
                "somph_isolated_runner": Path(__file__).with_name(
                    "somph_isolated_runner.py"
                ),
                "somph_predictor_bundle": Path(__file__).with_name(
                    "somph_predictor_bundle.py"
                ),
                "somph_predictor_entry": Path(__file__).with_name(
                    "somph_predictor_entry.py"
                ),
                "somph_fixed_entry_script": code_root / "scripts" / entry_script,
                "run_cvs_somph_bwrap_isolated": (
                    code_root / "scripts/run_cvs_somph_bwrap_isolated.py"
                ),
            }
        )
    return {
        name: _regular_descriptor(path, context=f"trusted controller {name}")
        for name, path in members.items()
    }


def _read_json_regular(path: Path, *, context: str) -> tuple[dict[str, Any], bytes]:
    descriptor = _regular_descriptor(path, context=context)
    payload = Path(str(descriptor["path"])).read_bytes()
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase2PreRunEvidenceError(f"{context} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Phase2PreRunEvidenceError(f"{context} root must be an object")
    return value, payload


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _attestation_document(
    *,
    isolation_profile: str,
    closure_root: Path,
    runtime_root: Path,
    package: Path,
    seal_path: Path,
    trusted_package_seal_sha256: str,
    executables: Mapping[str, Any],
    system_roots: list[Path],
    trusted_system_roots: list[Path],
    forbidden: list[Path],
) -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.os_isolation_attestation.v3",
        "status": "PASS",
        "claim_scope": "pre_run_prerequisites_only",
        "host_system": "Linux",
        "isolation_mode": "bwrap_readonly_mounts",
        "isolation_profile": isolation_profile,
        "trusted_package_seal_sha256": trusted_package_seal_sha256,
        "trusted_controller_members": _controller_descriptors(isolation_profile),
        "policy_contract": _policy_contract(isolation_profile),
        "executables": dict(executables),
        "system_read_roots": [str(path) for path in system_roots],
        "trusted_system_root_allowlist": [str(path) for path in trusted_system_roots],
        "runtime_closure_root": str(closure_root),
        "runtime_mount": {"source": str(runtime_root), "target": "/runtime/code", "mode": "ro"},
        "package_mount": {"source": str(package), "target": "/sealed/package", "mode": "ro"},
        "detached_seal_mount": {"source": str(seal_path), "target": "/sealed/package.seal.json", "mode": "ro"},
        "request_mount_target": "/sealed/request.json",
        "single_write_root_target": "/output",
        "network_namespace_unshared": True,
        "capabilities_dropped": "ALL",
        "environment_cleared": True,
        "scorer_truth_roots_not_mounted": [str(path) for path in forbidden],
        "actual_open_ledger_required_post_run": True,
        "post_run_pass_not_claimed": True,
        "formal_launch_authority": False,
        "formal_launch_blockers": list(FORMAL_LAUNCH_BLOCKERS),
    }


def _audit_document(
    *,
    closure: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seal: Mapping[str, Any],
    package_audit: Mapping[str, Any],
    seal_sha256: str,
    attestation_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "cvs.phase2.preopen_audit_receipt.v4",
        "status": "PASS",
        "runtime_closure": dict(closure),
        "package_root_sha256": manifest["package_root_sha256"],
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "sealed_inference_package_sha256": seal_sha256,
        "package_preopen_audit": dict(package_audit),
        "os_isolation_attestation_sha256": attestation_sha256,
        "predictor_scorer_physical_disjointness": "PASS",
        "query_iq_materialized_during_preopen": False,
        "post_run_filesystem_access_audit_pending": True,
    }


def _preflight_profile_package(
    isolation_profile: str,
    package: Path,
    *,
    seal_path: Path,
    seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isolation_profile == GENERIC_PREDICTOR_PROFILE:
        manifest, seal, package_audit = preflight_stage2_predictor_package(
            package,
            detached_seal_path=seal_path,
            expected_seal_sha256=seal_sha256,
        )
        if package_audit.get("status") != "PASS":
            raise Phase2PreRunEvidenceError(
                "predictor package pre-open audit did not pass"
            )
        return manifest, seal, package_audit

    from .somph_predictor_bundle import (
        APPLY_ONLY,
        ENROLLMENT_ONLY,
        preflight_somph_predictor_bundle,
    )

    manifest, seal, package_audit = preflight_somph_predictor_bundle(
        package,
        detached_seal_path=seal_path,
        expected_seal_sha256=seal_sha256,
    )
    expected_profile = (
        ENROLLMENT_ONLY
        if isolation_profile == SOMPH_ENROLLMENT_PROFILE
        else APPLY_ONLY
    )
    if manifest.get("profile") != expected_profile:
        raise Phase2PreRunEvidenceError(
            "SOMP-H package does not match the requested isolation profile"
        )
    if package_audit.get("status") != "STRUCTURAL_SELF_CONSISTENCY_PASS":
        raise Phase2PreRunEvidenceError(
            "SOMP-H package structural pre-open audit did not pass"
        )
    return manifest, seal, package_audit


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _trusted_system_root_allowlist() -> list[Path]:
    """Return fixed dependency roots; callers cannot nominate data roots."""

    candidates = [Path(sys.prefix), Path("/usr"), Path("/etc")]
    result: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_dir() and resolved not in result:
            result.append(resolved)
    return result


def _validated_system_roots(values: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    trusted = _trusted_system_root_allowlist()
    if not trusted:
        raise Phase2PreRunEvidenceError("trusted system root allowlist is empty")
    roots = [_resolved_directory(path, context="system read root") for path in values]
    if not roots:
        raise Phase2PreRunEvidenceError("at least one system read root is required")
    if len(set(roots)) != len(roots):
        raise Phase2PreRunEvidenceError("duplicate system read root")
    unexpected = [str(path) for path in roots if path not in trusted]
    if unexpected:
        raise Phase2PreRunEvidenceError(
            f"system read root is outside the fixed trusted allowlist: {unexpected}"
        )
    return roots, trusted


def _write_exclusive_readonly(path: Path, payload: bytes) -> str:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return _sha256_bytes(payload)


def build_phase2_pre_run_evidence(
    *,
    runtime_closure_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    expected_package_seal_sha256: str,
    output_root: str | Path,
    bwrap_executable: str | Path,
    strace_executable: str | Path,
    python_executable: str | Path,
    system_read_roots: Iterable[str | Path],
    forbidden_scorer_truth_roots: Iterable[str | Path],
    isolation_profile: str = GENERIC_PREDICTOR_PROFILE,
) -> dict[str, Any]:
    """Verify immutable inputs and publish exact 9-field pre-run evidence.

    The output attests only prerequisites and the fixed isolation policy.  A
    PASS suitable for formal scoring still requires the post-run strace audit.
    """

    if platform.system() != "Linux":
        raise Phase2PreRunEvidenceError(
            "formal bwrap pre-run evidence must be produced on the Linux execution host"
        )
    if isolation_profile not in ISOLATION_PROFILES:
        raise Phase2PreRunEvidenceError("unsupported Phase2 isolation profile")
    closure_root = _resolved_directory(runtime_closure_root, context="runtime closure root")
    closure = verify_phase2_runtime_closure(
        closure_root, expected_profile=isolation_profile
    )
    runtime_root = Path(str(closure["runtime_root"])).resolve(strict=True)
    package = _resolved_directory(package_root, context="predictor package root")
    seal_path = Path(detached_seal).resolve(strict=True)
    if Path(detached_seal).is_symlink() or not seal_path.is_file():
        raise Phase2PreRunEvidenceError("detached seal must be a regular non-symlink file")

    forbidden = [
        _resolved_directory(path, context="scorer/truth root")
        for path in forbidden_scorer_truth_roots
    ]
    if not forbidden:
        raise Phase2PreRunEvidenceError("at least one physically separate scorer/truth root is required")
    visible_inputs = [runtime_root, package, seal_path]
    for left_index, left in enumerate(visible_inputs):
        for right in visible_inputs[left_index + 1 :]:
            if _overlaps(left, right):
                raise Phase2PreRunEvidenceError("predictor-visible roots must be physically disjoint")
    if any(_overlaps(root, visible) for root in forbidden for visible in visible_inputs):
        raise Phase2PreRunEvidenceError("scorer/truth root overlaps predictor-visible input")

    system_roots, trusted_system_roots = _validated_system_roots(system_read_roots)
    if any(
        _overlaps(system_root, project_root)
        for system_root in system_roots
        for project_root in (*visible_inputs, *forbidden)
    ):
        raise Phase2PreRunEvidenceError("system read root overlaps project or scorer/truth root")

    executables = {
        "bwrap": _executable_descriptor(bwrap_executable, context="bwrap executable"),
        "strace": _executable_descriptor(strace_executable, context="strace executable"),
        "python": _executable_descriptor(python_executable, context="python executable"),
    }
    for name, descriptor in executables.items():
        resolved = Path(str(descriptor["resolved_path"]))
        if not _inside(resolved, system_roots):
            raise Phase2PreRunEvidenceError(
                f"{name} executable is outside declared read-only system roots"
            )

    trusted_seal_sha256 = str(expected_package_seal_sha256).lower()
    if not _valid_sha256(trusted_seal_sha256):
        raise Phase2PreRunEvidenceError("trusted package seal SHA256 is invalid")
    seal_sha256 = sha256_file(seal_path)
    if seal_sha256 != trusted_seal_sha256:
        raise Phase2PreRunEvidenceError("detached seal does not match the external trusted SHA256")
    manifest, seal, package_audit = _preflight_profile_package(
        isolation_profile,
        package,
        seal_path=seal_path,
        seal_sha256=seal_sha256,
    )

    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite pre-run evidence root: {output}")
    intended_output = output.resolve(strict=False)
    if any(_overlaps(intended_output, root) for root in (*visible_inputs, *forbidden)):
        raise Phase2PreRunEvidenceError("evidence output overlaps predictor or scorer/truth roots")

    attestation = _attestation_document(
        isolation_profile=isolation_profile,
        closure_root=closure_root,
        runtime_root=runtime_root,
        package=package,
        seal_path=seal_path,
        trusted_package_seal_sha256=trusted_seal_sha256,
        executables=executables,
        system_roots=system_roots,
        trusted_system_roots=trusted_system_roots,
        forbidden=forbidden,
    )
    attestation_payload = _canonical_json(attestation)
    attestation_sha256 = _sha256_bytes(attestation_payload)
    audit = _audit_document(
        closure=closure,
        manifest=manifest,
        seal=seal,
        package_audit=package_audit,
        seal_sha256=seal_sha256,
        attestation_sha256=attestation_sha256,
    )
    audit_payload = _canonical_json(audit)
    audit_sha256 = _sha256_bytes(audit_payload)
    evidence = {
        "sealed_inference_package_sha256": seal_sha256,
        "package_root_sha256": manifest["package_root_sha256"],
        "runtime_code_sha256": closure["root_sha256"],
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "os_isolation_mode": "bwrap_readonly_mounts",
        "os_isolation_attestation_sha256": attestation_sha256,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": audit_sha256,
        "predict_score_process_isolation": True,
    }

    output.mkdir(parents=True, exist_ok=False)
    written_attestation_sha256 = _write_exclusive_readonly(
        output / ATTESTATION_NAME, attestation_payload
    )
    written_audit_sha256 = _write_exclusive_readonly(
        output / PREOPEN_AUDIT_NAME, audit_payload
    )
    _write_exclusive_readonly(output / PRE_RUN_EVIDENCE_NAME, _canonical_json(evidence))
    if written_attestation_sha256 != attestation_sha256 or written_audit_sha256 != audit_sha256:
        raise Phase2PreRunEvidenceError("pre-run evidence changed while being published")
    verified_bundle = verify_phase2_pre_run_evidence(
        evidence_root=output,
        runtime_closure_root=closure_root,
        package_root=package,
        detached_seal=seal_path,
        expected_package_seal_sha256=trusted_seal_sha256,
        bwrap_executable=bwrap_executable,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_roots,
        forbidden_scorer_truth_roots=forbidden,
        expected_evidence=evidence,
        isolation_profile=isolation_profile,
    )
    return {
        "schema": "cvs.phase2.pre_run_evidence_build_result.v1",
        "status": "PASS",
        "isolation_profile": isolation_profile,
        "runtime_isolation_evidence": str(output / PRE_RUN_EVIDENCE_NAME),
        "os_isolation_attestation": str(output / ATTESTATION_NAME),
        "preopen_audit_receipt": str(output / PREOPEN_AUDIT_NAME),
        "evidence": evidence,
        "binding_sha256": verified_bundle["binding_sha256"],
        "post_run_filesystem_access_audit_pending": True,
        "formal_launch_authority": False,
        "formal_launch_blockers": list(FORMAL_LAUNCH_BLOCKERS),
    }


def verify_phase2_pre_run_evidence(
    *,
    evidence_root: str | Path,
    runtime_closure_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    expected_package_seal_sha256: str,
    bwrap_executable: str | Path,
    strace_executable: str | Path,
    python_executable: str | Path,
    system_read_roots: Iterable[str | Path],
    forbidden_scorer_truth_roots: Iterable[str | Path],
    expected_evidence: Mapping[str, Any] | None = None,
    isolation_profile: str = GENERIC_PREDICTOR_PROFILE,
) -> dict[str, Any]:
    """Re-verify the complete pre-run bundle against the actual runner inputs."""

    if platform.system() != "Linux":
        raise Phase2PreRunEvidenceError(
            "formal bwrap pre-run evidence must be verified on the Linux execution host"
        )
    if isolation_profile not in ISOLATION_PROFILES:
        raise Phase2PreRunEvidenceError("unsupported Phase2 isolation profile")
    bundle_root = _resolved_directory(evidence_root, context="pre-run evidence root")
    actual_members = {path.name for path in bundle_root.iterdir()}
    if actual_members != set(PRE_RUN_BUNDLE_MEMBERS):
        raise Phase2PreRunEvidenceError("pre-run evidence bundle member allowlist mismatch")
    attestation, attestation_payload = _read_json_regular(
        bundle_root / ATTESTATION_NAME, context="OS isolation attestation"
    )
    audit, audit_payload = _read_json_regular(
        bundle_root / PREOPEN_AUDIT_NAME, context="pre-open audit receipt"
    )
    evidence, evidence_payload = _read_json_regular(
        bundle_root / PRE_RUN_EVIDENCE_NAME, context="runtime isolation evidence"
    )
    for value, payload, context in (
        (attestation, attestation_payload, "OS isolation attestation"),
        (audit, audit_payload, "pre-open audit receipt"),
        (evidence, evidence_payload, "runtime isolation evidence"),
    ):
        if payload != _canonical_json(value):
            raise Phase2PreRunEvidenceError(f"{context} is not canonical immutable JSON")
    if set(evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise Phase2PreRunEvidenceError("runtime isolation evidence field set mismatch")
    if expected_evidence is not None and dict(expected_evidence) != evidence:
        raise Phase2PreRunEvidenceError("request evidence does not match the immutable evidence bundle")

    closure_root = _resolved_directory(runtime_closure_root, context="runtime closure root")
    closure = verify_phase2_runtime_closure(
        closure_root, expected_profile=isolation_profile
    )
    runtime_root = Path(str(closure["runtime_root"])).resolve(strict=True)
    package = _resolved_directory(package_root, context="predictor package root")
    seal_source = Path(detached_seal)
    seal_path = seal_source.resolve(strict=True)
    if seal_source.is_symlink() or not seal_path.is_file():
        raise Phase2PreRunEvidenceError("detached seal must be a regular non-symlink file")
    trusted_seal_sha256 = str(expected_package_seal_sha256).lower()
    if not _valid_sha256(trusted_seal_sha256):
        raise Phase2PreRunEvidenceError("trusted package seal SHA256 is invalid")
    seal_sha256 = sha256_file(seal_path)
    if seal_sha256 != trusted_seal_sha256:
        raise Phase2PreRunEvidenceError("detached seal does not match the external trusted SHA256")

    forbidden = [
        _resolved_directory(path, context="scorer/truth root")
        for path in forbidden_scorer_truth_roots
    ]
    if not forbidden:
        raise Phase2PreRunEvidenceError("at least one physically separate scorer/truth root is required")
    visible_inputs = [runtime_root, package, seal_path]
    for left_index, left in enumerate(visible_inputs):
        for right in visible_inputs[left_index + 1 :]:
            if _overlaps(left, right):
                raise Phase2PreRunEvidenceError("predictor-visible roots must be physically disjoint")
    if any(_overlaps(root, visible) for root in forbidden for visible in visible_inputs):
        raise Phase2PreRunEvidenceError("scorer/truth root overlaps predictor-visible input")

    system_roots, trusted_system_roots = _validated_system_roots(system_read_roots)
    if any(
        _overlaps(system_root, project_root)
        for system_root in system_roots
        for project_root in (*visible_inputs, *forbidden)
    ):
        raise Phase2PreRunEvidenceError("system read root overlaps project or scorer/truth root")
    executables = {
        "bwrap": _executable_descriptor(bwrap_executable, context="bwrap executable"),
        "strace": _executable_descriptor(strace_executable, context="strace executable"),
        "python": _executable_descriptor(python_executable, context="python executable"),
    }
    for name, descriptor in executables.items():
        if not _inside(Path(str(descriptor["resolved_path"])), system_roots):
            raise Phase2PreRunEvidenceError(
                f"{name} executable is outside declared read-only system roots"
            )

    manifest, seal, package_audit = _preflight_profile_package(
        isolation_profile,
        package,
        seal_path=seal_path,
        seal_sha256=trusted_seal_sha256,
    )
    expected_attestation = _attestation_document(
        isolation_profile=isolation_profile,
        closure_root=closure_root,
        runtime_root=runtime_root,
        package=package,
        seal_path=seal_path,
        trusted_package_seal_sha256=trusted_seal_sha256,
        executables=executables,
        system_roots=system_roots,
        trusted_system_roots=trusted_system_roots,
        forbidden=forbidden,
    )
    if attestation != expected_attestation:
        raise Phase2PreRunEvidenceError("OS isolation attestation does not match actual runner inputs")
    attestation_sha256 = _sha256_bytes(attestation_payload)
    expected_audit = _audit_document(
        closure=closure,
        manifest=manifest,
        seal=seal,
        package_audit=package_audit,
        seal_sha256=seal_sha256,
        attestation_sha256=attestation_sha256,
    )
    if audit != expected_audit:
        raise Phase2PreRunEvidenceError("pre-open audit receipt does not match current immutable inputs")
    audit_sha256 = _sha256_bytes(audit_payload)
    expected_fields = {
        "sealed_inference_package_sha256": seal_sha256,
        "package_root_sha256": manifest["package_root_sha256"],
        "runtime_code_sha256": closure["root_sha256"],
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "os_isolation_mode": "bwrap_readonly_mounts",
        "os_isolation_attestation_sha256": attestation_sha256,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": audit_sha256,
        "predict_score_process_isolation": True,
    }
    if evidence != expected_fields:
        raise Phase2PreRunEvidenceError("runtime isolation evidence cross-binding mismatch")
    return {
        "schema": "cvs.phase2.pre_run_evidence_verification.v1",
        "status": "PASS",
        "isolation_profile": isolation_profile,
        "runtime_root": str(runtime_root),
        "trusted_system_read_roots": [str(path) for path in trusted_system_roots],
        "runtime_code_sha256": closure["root_sha256"],
        "package_root_sha256": manifest["package_root_sha256"],
        "sealed_inference_package_sha256": seal_sha256,
        "binding_sha256": _sha256_bytes(
            attestation_payload + audit_payload + evidence_payload
        ),
        "evidence": evidence,
        "post_run_filesystem_access_audit_pending": True,
        "formal_launch_authority": False,
        "formal_launch_blockers": list(FORMAL_LAUNCH_BLOCKERS),
    }


__all__ = [
    "ATTESTATION_NAME",
    "PREOPEN_AUDIT_NAME",
    "PRE_RUN_EVIDENCE_NAME",
    "PRE_RUN_BUNDLE_MEMBERS",
    "PHASE2_BWRAP_POLICY_CONTRACT",
    "FORMAL_LAUNCH_BLOCKERS",
    "GENERIC_PREDICTOR_PROFILE",
    "ISOLATION_PROFILES",
    "SOMPH_APPLY_PROFILE",
    "SOMPH_ENROLLMENT_PROFILE",
    "Phase2PreRunEvidenceError",
    "build_phase2_pre_run_evidence",
    "verify_phase2_pre_run_evidence",
]
