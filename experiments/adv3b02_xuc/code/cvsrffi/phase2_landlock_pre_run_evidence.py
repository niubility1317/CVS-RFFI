"""Build and verify pre-run evidence for Landlock+seccomp Phase2 isolation."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping

from .phase2_landlock_policy import query_landlock_abi
from .phase2_pre_run_evidence import (
    ATTESTATION_NAME,
    PREOPEN_AUDIT_NAME,
    PRE_RUN_EVIDENCE_NAME,
    _canonical_json,
    _executable_descriptor,
    _overlaps,
    _regular_descriptor,
    _resolved_directory,
    _sha256_bytes,
    _validated_system_roots,
    _valid_sha256,
    _write_exclusive_readonly,
)
from .phase2_runtime_closure import verify_phase2_runtime_closure
from .phase2_runtime_contract import PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS
from .stage2_predictor_bundle import (
    preflight_stage2_predictor_package,
    sha256_file,
)


LANDLOCK_ISOLATION_MODE = "equivalent_verified_isolation"
LANDLOCK_BACKEND = "landlock_v4_seccomp_strace"
LANDLOCK_EVIDENCE_SCHEMA = "cvs.phase2.landlock_pre_run_evidence.v1"
LANDLOCK_ATTESTATION_SCHEMA = "cvs.phase2.landlock_attestation.v1"
LANDLOCK_PREOPEN_SCHEMA = "cvs.phase2.landlock_preopen_audit.v1"
LANDLOCK_FORMAL_BLOCKERS = (
    "FIXED_INODE_OR_DIFFERENT_UID_INPUT_SNAPSHOT_NOT_YET_PROVEN",
    "LANDLOCK_DOES_NOT_HIDE_HOST_PATH_NAMES",
    "REAL_FOUR_STATE_LINUX_ISOLATION_SMOKE_NOT_YET_PASSED",
)


class Phase2LandlockEvidenceError(ValueError):
    """Raised when equivalent Landlock isolation evidence is incomplete."""


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    descriptor = _regular_descriptor(path, context=context)
    try:
        payload = json.loads(
            Path(str(descriptor["path"])).read_text(encoding="utf-8-sig")
        )
    except json.JSONDecodeError as exc:
        raise Phase2LandlockEvidenceError(f"{context} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise Phase2LandlockEvidenceError(f"{context} must be a JSON object")
    return payload


def _collect(
    *,
    runtime_closure_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    expected_package_seal_sha256: str,
    landlock_launcher: str | Path,
    landlock_policy_module: str | Path,
    strace_executable: str | Path,
    python_executable: str | Path,
    system_read_roots: Iterable[str | Path],
    forbidden_scorer_truth_roots: Iterable[str | Path],
) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise Phase2LandlockEvidenceError(
            "Landlock pre-run evidence must be produced on Linux"
        )
    closure_root = _resolved_directory(
        runtime_closure_root, context="runtime closure root"
    )
    closure = verify_phase2_runtime_closure(
        closure_root, expected_profile="stage2_predictor"
    )
    runtime_root = Path(str(closure["runtime_root"])).resolve(strict=True)
    package = _resolved_directory(package_root, context="predictor package root")
    seal = Path(detached_seal).resolve(strict=True)
    if Path(detached_seal).is_symlink() or not seal.is_file():
        raise Phase2LandlockEvidenceError(
            "detached seal must be a regular non-symlink file"
        )
    forbidden = [
        _resolved_directory(path, context="scorer/truth root")
        for path in forbidden_scorer_truth_roots
    ]
    if not forbidden:
        raise Phase2LandlockEvidenceError(
            "at least one separate scorer/truth root is required"
        )
    visible = [runtime_root, package, seal]
    for index, left in enumerate(visible):
        for right in visible[index + 1 :]:
            if _overlaps(left, right):
                raise Phase2LandlockEvidenceError(
                    "predictor-visible roots must be physically disjoint"
                )
    if any(_overlaps(root, item) for root in forbidden for item in visible):
        raise Phase2LandlockEvidenceError(
            "scorer/truth root overlaps predictor-visible inputs"
        )
    system_roots, trusted_system_roots = _validated_system_roots(
        system_read_roots
    )
    if any(
        _overlaps(system_root, project_root)
        for system_root in system_roots
        for project_root in (*visible, *forbidden)
    ):
        raise Phase2LandlockEvidenceError(
            "system root overlaps package/runtime/scorer"
        )
    launcher_descriptor = _regular_descriptor(
        landlock_launcher, context="Landlock launcher"
    )
    policy_descriptor = _regular_descriptor(
        landlock_policy_module, context="Landlock policy module"
    )
    launcher_path = Path(str(launcher_descriptor["path"]))
    policy_path = Path(str(policy_descriptor["path"]))
    if launcher_path.parents[1] != policy_path.parents[1]:
        raise Phase2LandlockEvidenceError(
            "Landlock launcher and policy must share one reviewed code root"
        )
    executables = {
        "strace": _executable_descriptor(
            strace_executable, context="strace executable"
        ),
        "python": _executable_descriptor(
            python_executable, context="Python executable"
        ),
    }
    for name, descriptor in executables.items():
        resolved = Path(str(descriptor["resolved_path"]))
        if not any(
            resolved == root or root in resolved.parents for root in system_roots
        ):
            raise Phase2LandlockEvidenceError(
                f"{name} is outside declared system read roots"
            )
    landlock_abi = query_landlock_abi()
    if landlock_abi < 4:
        raise Phase2LandlockEvidenceError(
            f"Landlock ABI4 required, observed {landlock_abi}"
        )
    trusted_seal = str(expected_package_seal_sha256).lower()
    if not _valid_sha256(trusted_seal) or sha256_file(seal) != trusted_seal:
        raise Phase2LandlockEvidenceError(
            "detached seal does not match the external trusted SHA256"
        )
    manifest, package_seal, package_audit = preflight_stage2_predictor_package(
        package,
        detached_seal_path=seal,
        expected_seal_sha256=trusted_seal,
    )
    if package_audit.get("status") != "PASS":
        raise Phase2LandlockEvidenceError("predictor package preflight failed")
    sealed_package_members = sorted(
        {
            "package_manifest.json",
            *(
                str(item["relative_path"])
                for item in manifest.get("members", [])
            ),
        }
    )
    host_to_logical = {
        str(runtime_root): "/runtime/code",
        str(package): "/sealed/package",
        str(seal): "/sealed/package.seal.json",
    }
    attestation = {
        "schema": LANDLOCK_ATTESTATION_SCHEMA,
        "status": "PASS",
        "claim_scope": "pre_run_prerequisites_only",
        "host_system": "Linux",
        "os_isolation_mode": LANDLOCK_ISOLATION_MODE,
        "isolation_backend": LANDLOCK_BACKEND,
        "filesystem_namespace_unshared": False,
        "direct_host_path_execution": True,
        "landlock_abi": landlock_abi,
        "landlock_filesystem_default_deny": True,
        "landlock_tcp_bind_connect_default_deny": True,
        "seccomp_socket_syscalls_default_deny": True,
        "no_new_privileges_required": True,
        "inherited_nonstdio_fd_policy": "close_before_landlock_and_predictor_exec",
        "full_lifecycle_strace_required": True,
        "predictor_phase_open_ledger_required": True,
        "runtime_closure_root": str(closure_root),
        "runtime_root": str(runtime_root),
        "package_root": str(package),
        "detached_seal": str(seal),
        "system_read_roots": [str(path) for path in system_roots],
        "trusted_system_root_allowlist": [
            str(path) for path in trusted_system_roots
        ],
        "forbidden_scorer_truth_roots": [str(path) for path in forbidden],
        "host_to_logical_path_mapping": host_to_logical,
        "landlock_launcher": launcher_descriptor,
        "landlock_policy_module": policy_descriptor,
        "executables": executables,
        "formal_launch_authority": False,
        "formal_launch_blockers": list(LANDLOCK_FORMAL_BLOCKERS),
    }
    attestation_sha = _sha256_bytes(_canonical_json(attestation))
    audit = {
        "schema": LANDLOCK_PREOPEN_SCHEMA,
        "status": "PASS",
        "runtime_closure": closure,
        "package_root_sha256": manifest["package_root_sha256"],
        "artifact_member_allowlist_sha256": package_seal[
            "artifact_member_allowlist_sha256"
        ],
        "sealed_inference_package_sha256": trusted_seal,
        "package_preopen_audit": package_audit,
        "landlock_attestation_sha256": attestation_sha,
    }
    audit_sha = _sha256_bytes(_canonical_json(audit))
    evidence = {
        "sealed_inference_package_sha256": trusted_seal,
        "package_root_sha256": manifest["package_root_sha256"],
        "runtime_code_sha256": closure["root_sha256"],
        "artifact_member_allowlist_sha256": package_seal[
            "artifact_member_allowlist_sha256"
        ],
        "os_isolation_mode": LANDLOCK_ISOLATION_MODE,
        "os_isolation_attestation_sha256": attestation_sha,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": audit_sha,
        "predict_score_process_isolation": True,
    }
    if set(evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise Phase2LandlockEvidenceError("Landlock evidence schema drift")
    return {
        "closure": closure,
        "runtime_root": runtime_root,
        "package": package,
        "seal": seal,
        "forbidden": forbidden,
        "system_roots": system_roots,
        "trusted_system_roots": trusted_system_roots,
        "attestation": attestation,
        "audit": audit,
        "evidence": evidence,
        "sealed_package_members": sealed_package_members,
    }


def build_phase2_landlock_pre_run_evidence(
    *,
    runtime_closure_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    expected_package_seal_sha256: str,
    output_root: str | Path,
    landlock_launcher: str | Path,
    landlock_policy_module: str | Path,
    strace_executable: str | Path,
    python_executable: str | Path,
    system_read_roots: Iterable[str | Path],
    forbidden_scorer_truth_roots: Iterable[str | Path],
) -> dict[str, Any]:
    collected = _collect(
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        landlock_launcher=landlock_launcher,
        landlock_policy_module=landlock_policy_module,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_scorer_truth_roots,
    )
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write_exclusive_readonly(
        output / ATTESTATION_NAME,
        _canonical_json(collected["attestation"]),
    )
    _write_exclusive_readonly(
        output / PREOPEN_AUDIT_NAME,
        _canonical_json(collected["audit"]),
    )
    _write_exclusive_readonly(
        output / PRE_RUN_EVIDENCE_NAME,
        _canonical_json(collected["evidence"]),
    )
    verified = verify_phase2_landlock_pre_run_evidence(
        evidence_root=output,
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        landlock_launcher=landlock_launcher,
        landlock_policy_module=landlock_policy_module,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_scorer_truth_roots,
        expected_evidence=collected["evidence"],
    )
    return {
        "schema": LANDLOCK_EVIDENCE_SCHEMA,
        "status": "PASS",
        "runtime_isolation_evidence": str(output / PRE_RUN_EVIDENCE_NAME),
        "os_isolation_attestation": str(output / ATTESTATION_NAME),
        "preopen_audit_receipt": str(output / PREOPEN_AUDIT_NAME),
        "evidence": collected["evidence"],
        "binding_sha256": verified["binding_sha256"],
        "post_run_filesystem_access_audit_pending": True,
        "formal_launch_authority": False,
        "formal_launch_blockers": list(LANDLOCK_FORMAL_BLOCKERS),
    }


def verify_phase2_landlock_pre_run_evidence(
    *,
    evidence_root: str | Path,
    runtime_closure_root: str | Path,
    package_root: str | Path,
    detached_seal: str | Path,
    expected_package_seal_sha256: str,
    landlock_launcher: str | Path,
    landlock_policy_module: str | Path,
    strace_executable: str | Path,
    python_executable: str | Path,
    system_read_roots: Iterable[str | Path],
    forbidden_scorer_truth_roots: Iterable[str | Path],
    expected_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _resolved_directory(evidence_root, context="Landlock evidence root")
    attestation = _read_json(root / ATTESTATION_NAME, context="Landlock attestation")
    audit = _read_json(root / PREOPEN_AUDIT_NAME, context="Landlock preopen audit")
    evidence = _read_json(
        root / PRE_RUN_EVIDENCE_NAME, context="Landlock runtime evidence"
    )
    collected = _collect(
        runtime_closure_root=runtime_closure_root,
        package_root=package_root,
        detached_seal=detached_seal,
        expected_package_seal_sha256=expected_package_seal_sha256,
        landlock_launcher=landlock_launcher,
        landlock_policy_module=landlock_policy_module,
        strace_executable=strace_executable,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
        forbidden_scorer_truth_roots=forbidden_scorer_truth_roots,
    )
    if (
        attestation != collected["attestation"]
        or audit != collected["audit"]
        or evidence != collected["evidence"]
    ):
        raise Phase2LandlockEvidenceError(
            "Landlock pre-run evidence no longer matches current inputs"
        )
    if expected_evidence is not None and dict(expected_evidence) != evidence:
        raise Phase2LandlockEvidenceError(
            "request Landlock evidence differs from the evidence bundle"
        )
    binding = _sha256_bytes(
        _canonical_json(
            {
                "attestation_sha256": sha256_file(root / ATTESTATION_NAME),
                "preopen_audit_sha256": sha256_file(root / PREOPEN_AUDIT_NAME),
                "runtime_evidence_sha256": sha256_file(
                    root / PRE_RUN_EVIDENCE_NAME
                ),
            }
        )
    )
    return {
        "status": "PASS",
        "binding_sha256": binding,
        "runtime_root": str(collected["runtime_root"]),
        "trusted_system_read_roots": [
            str(path) for path in collected["trusted_system_roots"]
        ],
        "evidence": evidence,
        "attestation": attestation,
        "sealed_package_members": collected["sealed_package_members"],
    }


__all__ = [
    "LANDLOCK_BACKEND",
    "LANDLOCK_FORMAL_BLOCKERS",
    "LANDLOCK_ISOLATION_MODE",
    "Phase2LandlockEvidenceError",
    "build_phase2_landlock_pre_run_evidence",
    "verify_phase2_landlock_pre_run_evidence",
]
