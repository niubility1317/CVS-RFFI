#!/usr/bin/env python
"""Build strict Landlock+sealed-memfd pre-run evidence for one Phase2 package."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_candidate_capsule import sha256_file  # noqa: E402
from cvsrffi.phase2_memfd_snapshot import REQUIRED_SEALS  # noqa: E402
from cvsrffi.phase2_runtime_closure import verify_phase2_runtime_closure  # noqa: E402
from cvsrffi.phase2_runtime_contract import (  # noqa: E402
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    preflight_stage2_predictor_package,
)


LANDLOCK_CREATE_RULESET_VERSION = 1


def _canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_descriptor(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    metadata = source.stat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"evidence input must be a regular non-symlink file: {path}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": int(metadata.st_size),
    }


def _directory(path: Path, *, context: str) -> Path:
    value = path.resolve(strict=True)
    if path.is_symlink() or not value.is_dir():
        raise ValueError(f"{context} must be a non-symlink directory")
    return value


def _landlock_abi() -> int:
    if platform.system() != "Linux":
        raise ValueError("formal Landlock evidence must be built on Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    return int(libc.syscall(444, 0, 0, LANDLOCK_CREATE_RULESET_VERSION))


def _write_new_readonly(path: Path, payload: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short evidence write: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o444)
    return _sha_bytes(payload)


def build(args: argparse.Namespace) -> dict[str, Any]:
    closure_root = _directory(Path(args.runtime_closure_root), context="runtime closure")
    closure = verify_phase2_runtime_closure(closure_root)
    package_root = _directory(Path(args.package_root), context="predictor package")
    scorer_root = _directory(Path(args.scorer_root), context="scorer root")
    if package_root == scorer_root or package_root in scorer_root.parents or scorer_root in package_root.parents:
        raise ValueError("predictor and scorer roots must be physically disjoint")
    seal_path = Path(args.detached_seal).resolve(strict=True)
    expected_seal = str(args.expected_package_seal_sha256).lower()
    manifest, seal, package_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=seal_path,
        expected_seal_sha256=expected_seal,
    )
    capsule_path = Path(args.candidate_capsule)
    expected_capsule = str(args.expected_candidate_capsule_sha256).lower()
    if sha256_file(capsule_path) != expected_capsule:
        raise ValueError("candidate capsule does not match the external trust root")
    runtime_receipt = Path(args.runtime_config_receipt)
    receipt_payload = json.loads(runtime_receipt.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(receipt_payload, dict)
        or receipt_payload.get("status") != "PASS"
        or receipt_payload.get("candidate_capsule_sha256") != expected_capsule
        or receipt_payload.get("derivation_uses_target_query") is not False
    ):
        raise ValueError("runtime-config receipt is not bound to the trusted capsule")
    abi = _landlock_abi()
    if abi < 1:
        raise ValueError("Landlock is unavailable")
    executables = {
        "python": _regular_descriptor(Path(args.python_executable)),
        "strace": _regular_descriptor(Path(args.strace_executable)),
        "landlock_launcher": _regular_descriptor(Path(args.landlock_launcher)),
    }
    runtime_root = Path(str(closure["runtime_root"]))
    runtime_files = sorted(path for path in runtime_root.rglob("*") if path.is_file())
    runtime_dirs = {runtime_root, runtime_root.parent}
    for path in runtime_files:
        cursor = path.parent
        while True:
            runtime_dirs.add(cursor)
            if cursor == runtime_root:
                break
            cursor = cursor.parent
    allowlist = {
        "schema": "cvs_phase2_landlock_allowlist_v1",
        "read_files": [str(path.resolve(strict=True)) for path in runtime_files],
        "runtime_code_list_dirs": sorted(str(path.resolve(strict=True)) for path in runtime_dirs),
        "forbidden_predictor_artifact_tokens": [
            "truth_sidecar",
            "scoring_manifest",
            ".pkl",
            "manysig",
            "manytx",
            "clean",
            "raw",
        ],
    }
    allowlist_raw = _canonical(allowlist)
    attestation = {
        "schema": "cvs.phase2.landlock_memfd_attestation.v1",
        "status": "PASS",
        "host_system": "Linux",
        "kernel": platform.release(),
        "landlock_abi": abi,
        "isolation_mode": "equivalent_verified_isolation",
        "landlock_no_new_privs_required": True,
        "network_access_allowed": False,
        "package_paths_allowed_after_exec": False,
        "request_paths_allowed_after_exec": False,
        "truth_or_scorer_roots_allowed": False,
        "immutable_snapshot_mechanism": "sealed_memfd_inherited_fd_only",
        "required_memfd_seals": int(REQUIRED_SEALS),
        "same_uid_replace_restore_reachable": False,
        "actual_open_ledger_required_post_run": True,
        "executables": executables,
        "runtime_closure_root_sha256": closure["root_sha256"],
        "candidate_capsule_sha256": expected_capsule,
        "runtime_config_receipt_sha256": sha256_file(runtime_receipt),
        "landlock_allowlist_sha256": _sha_bytes(allowlist_raw),
    }
    attestation_raw = _canonical(attestation)
    preopen = {
        "schema": "cvs.phase2.landlock_memfd_preopen_audit.v1",
        "status": "PASS",
        "package_preflight": package_audit,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_seal_sha256": expected_seal,
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "runtime_code_sha256": closure["root_sha256"],
        "candidate_capsule_sha256": expected_capsule,
        "runtime_config_receipt_sha256": sha256_file(runtime_receipt),
        "landlock_allowlist_sha256": _sha_bytes(allowlist_raw),
        "query_iq_materialized": False,
        "memfd_snapshot_required_before_predictor_exec": True,
    }
    preopen_raw = _canonical(preopen)
    evidence = {
        "sealed_inference_package_sha256": expected_seal,
        "package_root_sha256": manifest["package_root_sha256"],
        "runtime_code_sha256": closure["root_sha256"],
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "os_isolation_mode": "equivalent_verified_isolation",
        "os_isolation_attestation_sha256": _sha_bytes(attestation_raw),
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": _sha_bytes(preopen_raw),
        "predict_score_process_isolation": True,
    }
    if set(evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise ValueError("pre-run evidence exact field set drift")
    output = Path(args.output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _write_new_readonly(output / "os_isolation_attestation.json", attestation_raw)
    _write_new_readonly(output / "preopen_audit_receipt.json", preopen_raw)
    _write_new_readonly(output / "runtime_isolation_evidence.json", _canonical(evidence))
    _write_new_readonly(output / "landlock_allowlist.json", allowlist_raw)
    return {
        "schema": "cvs.phase2.landlock_memfd_pre_run_build.v1",
        "status": "PASS",
        "runtime_isolation_evidence": str(output / "runtime_isolation_evidence.json"),
        "candidate_capsule_sha256": expected_capsule,
        "landlock_abi": abi,
        "landlock_allowlist": str(output / "landlock_allowlist.json"),
        "formal_launch_authority": False,
        "next_gate": "REAL_N607_LANDLOCK_MEMFD_SMOKE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-closure-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-package-seal-sha256", required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--candidate-capsule", type=Path, required=True)
    parser.add_argument("--expected-candidate-capsule-sha256", required=True)
    parser.add_argument("--runtime-config-receipt", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--strace-executable", type=Path, required=True)
    parser.add_argument("--landlock-launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
