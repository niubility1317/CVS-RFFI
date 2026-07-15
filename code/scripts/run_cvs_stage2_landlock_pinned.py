#!/usr/bin/env python
"""Run one formal Phase2 cell under Landlock, seccomp, and sealed memfd inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import secrets
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_isolated_runner import (  # noqa: E402
    _parse_predictor_stdout,
    _predictor_trace_suffix,
    _write_exclusive_readonly,
    _write_json_exclusive_readonly,
    parse_successful_open_trace,
)
from cvsrffi.phase2_memfd_snapshot import build_sealed_memfd_snapshot  # noqa: E402
from cvsrffi.phase2_runtime_closure import verify_phase2_runtime_closure  # noqa: E402
from cvsrffi.phase2_runtime_contract import (  # noqa: E402
    POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    validate_phase2_contract,
    validate_predictor_request,
)
from cvsrffi.stage2_prediction_artifact import verify_prediction_artifact  # noqa: E402
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    preflight_stage2_predictor_package,
    sha256_file,
)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path) -> bytes:
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"runner input must be a regular non-symlink file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ValueError(f"runner input identity changed: {path}")
        data = b""
        while len(data) < opened.st_size:
            chunk = os.read(fd, opened.st_size - len(data))
            if not chunk:
                raise ValueError(f"runner input was truncated: {path}")
            data += chunk
        return data
    finally:
        os.close(fd)


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path)
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError(f"runner evidence is not canonical JSON: {path}")
    return value, raw


def _inside(path: str, root: str) -> bool:
    value = PurePosixPath(path)
    base = PurePosixPath(posixpath.normpath(root))
    return value == base or base in value.parents


def _audit_ledger(
    ledger: list[dict[str, Any]],
    *,
    runtime_root: Path,
    output_root: Path,
    system_roots: list[Path],
    forbidden_roots: list[Path],
    package_root: Path,
) -> dict[str, Any]:
    allowed = [str(runtime_root), str(output_root), "/proc", "/dev", "/sys", "/run", "/tmp"]
    allowed.extend(str(path) for path in system_roots)
    forbidden = [str(path) for path in [*forbidden_roots, package_root]]
    sensitive = ("truth", "scoring", "scorer", "clean", "raw", "manysig", "manytx")
    violations: list[dict[str, str]] = []
    paths = [str(row["path"]) for row in ledger]
    if not any(_inside(path, str(runtime_root)) for path in paths):
        violations.append({"path": str(runtime_root), "reason": "runtime_closure_not_observed"})
    if not any(_inside(path, str(output_root)) for path in paths):
        violations.append({"path": str(output_root), "reason": "output_root_not_observed"})
    for path in paths:
        lower_parts = [part.lower() for part in PurePosixPath(path).parts]
        if any(token in part for part in lower_parts for token in sensitive):
            violations.append({"path": path, "reason": "sensitive_path_opened"})
            continue
        if any(_inside(path, root) for root in forbidden):
            violations.append({"path": path, "reason": "forbidden_host_root_opened"})
            continue
        if not any(_inside(path, root) for root in allowed):
            violations.append({"path": path, "reason": "path_outside_allowlist"})
    return {
        "status": "PASS" if not violations else "FAIL",
        "isolation_mode": "landlock_seccomp_sealed_memfd",
        "package_source_path_opened_after_predictor_exec": any(
            _inside(path, str(package_root)) for path in paths
        ),
        "opened_file_ledger": ledger,
        "unique_opened_path_count": len(ledger),
        "successful_open_count": sum(int(row["successful_open_count"]) for row in ledger),
        "violations": violations,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve(strict=True)
    if Path(args.output_root).is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise ValueError("formal output root must be an empty non-symlink directory")
    request_path = Path(args.request_json)
    request_raw = _read_regular(request_path)
    request = json.loads(request_raw.decode("utf-8-sig"))
    if not isinstance(request, dict):
        raise ValueError("request root must be an object")
    validate_predictor_request(request)
    evidence_root = Path(args.pre_run_evidence_root).resolve(strict=True)
    evidence, evidence_raw = _read_canonical_json(
        evidence_root / "runtime_isolation_evidence.json"
    )
    attestation, attestation_raw = _read_canonical_json(
        evidence_root / "os_isolation_attestation.json"
    )
    preopen, preopen_raw = _read_canonical_json(
        evidence_root / "preopen_audit_receipt.json"
    )
    allowlist_path = evidence_root / "landlock_allowlist.json"
    allowlist_raw = _read_regular(allowlist_path)
    if set(evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise ValueError("pre-run evidence exact schema drift")
    if request["phase2_runtime_isolation_evidence"] != evidence:
        raise ValueError("request/pre-run evidence mismatch")
    if (
        evidence["os_isolation_mode"] != "equivalent_verified_isolation"
        or evidence["os_isolation_attestation_sha256"] != _sha(attestation_raw)
        or evidence["preopen_audit_receipt_sha256"] != _sha(preopen_raw)
        or attestation.get("immutable_snapshot_mechanism")
        != "sealed_memfd_inherited_fd_only"
        or attestation.get("same_uid_replace_restore_reachable") is not False
        or attestation.get("network_access_allowed") is not False
        or attestation.get("landlock_allowlist_sha256") != _sha(allowlist_raw)
        or preopen.get("landlock_allowlist_sha256") != _sha(allowlist_raw)
    ):
        raise ValueError("Landlock/memfd pre-run evidence binding failed")
    closure = verify_phase2_runtime_closure(Path(args.runtime_closure_root))
    runtime_root = Path(str(closure["runtime_root"])).resolve(strict=True)
    package_root = Path(args.package_root).resolve(strict=True)
    seal_path = Path(args.detached_seal).resolve(strict=True)
    manifest, _seal, package_audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=seal_path,
        expected_seal_sha256=str(args.expected_package_seal_sha256).lower(),
    )
    if (
        package_audit.get("status") != "PASS"
        or manifest["package_root_sha256"] != request["package_root_sha256"]
        or closure["root_sha256"] != request["runtime_code_sha256"]
    ):
        raise ValueError("runtime closure or package changed after pre-run evidence")
    snapshot = build_sealed_memfd_snapshot(
        package_root=package_root,
        detached_seal=seal_path,
        request_json=request_path,
        manifest=manifest,
    )
    snapshot_receipt_raw = _canonical(snapshot.receipt)
    trace_temp = output.parent / f".phase2_landlock_trace.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    runtime_attestation = output / "landlock_runtime_attestation.json"
    python = Path(args.python_executable).resolve(strict=True)
    predictor = runtime_root / "scripts/run_cvs_stage2_predictor.py"
    command = [
        str(Path(args.strace_executable).resolve(strict=True)),
        "-f",
        "-qq",
        "-yy",
        "-s",
        "4096",
        "-e",
        "trace=execve,open,openat,openat2,socket,connect,bind,listen",
        "-o",
        str(trace_temp),
        str(python),
        str(Path(args.landlock_launcher).resolve(strict=True)),
        "--allowlist",
        str(allowlist_path),
        "--write-dir",
        str(output),
        "--runtime-read-dir",
        str(Path(sys.prefix).resolve(strict=True)),
        "--require-pinned-inputs",
        "--attestation-out",
        str(runtime_attestation),
        "--",
        str(python),
        str(predictor),
        "--request-json",
        str(request_path),
        "--predictor-package-root",
        str(package_root),
        "--detached-seal-path",
        str(seal_path),
        "--expected-seal-sha256",
        str(args.expected_package_seal_sha256).lower(),
        "--output-root",
        str(output),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "NVIDIA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        }
    }
    environment.update(snapshot.environment)
    environment.update(
        {
            "PYTHONPATH": str(runtime_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.timeout_seconds,
            pass_fds=snapshot.pass_fds,
            env=environment,
        )
    finally:
        snapshot.close()
    trace_raw = _read_regular(trace_temp)
    os.chmod(trace_temp, 0o600)
    trace_temp.unlink()
    trace_sha = _write_exclusive_readonly(output / "filesystem_access_trace.log", trace_raw)
    trace_text = trace_raw.decode("utf-8")
    wrapper_trace = _predictor_trace_suffix(trace_text, expected_executable=python)
    predictor_trace = _predictor_trace_suffix(wrapper_trace, expected_executable=python)
    ledger = parse_successful_open_trace(predictor_trace)
    forbidden_roots = [Path(value).resolve(strict=True) for value in args.forbidden_root]
    system_roots = [Path(sys.prefix).resolve(strict=True), Path("/usr"), Path("/etc"), Path("/lib"), Path("/lib64")]
    system_roots = [path.resolve(strict=True) for path in system_roots if path.exists()]
    audit = _audit_ledger(
        ledger,
        runtime_root=runtime_root,
        output_root=output,
        system_roots=system_roots,
        forbidden_roots=forbidden_roots,
        package_root=package_root,
    )
    runtime_attestation_payload, runtime_attestation_raw = _read_canonical_json(
        runtime_attestation
    )
    if (
        runtime_attestation_payload.get("status") != "PASS"
        or runtime_attestation_payload.get("landlock_enforced") is not True
        or runtime_attestation_payload.get("network_syscalls_seccomp_denied") is not True
        or runtime_attestation_payload.get("pinned_memfd_inputs_required") is not True
    ):
        raise ValueError("runtime Landlock/seccomp attestation failed")
    snapshot_receipt_sha = _write_exclusive_readonly(
        output / "memfd_snapshot_receipt.json", snapshot_receipt_raw
    )
    audit.update(
        {
            "request_sha256": _sha(request_raw),
            "trace_sha256": trace_sha,
            "memfd_snapshot_receipt_sha256": snapshot_receipt_sha,
            "landlock_runtime_attestation_sha256": _sha(runtime_attestation_raw),
        }
    )
    audit_sha = _write_json_exclusive_readonly(output / "filesystem_access_audit.json", audit)
    stdout = str(completed.stdout or "")
    stderr = str(completed.stderr or "")
    predictor_result = _parse_predictor_stdout(stdout) if completed.returncode == 0 else None
    stdout_receipt = {
        "schema": "cvs.phase2.predictor_stdout_receipt.v1",
        "status": "PASS" if predictor_result is not None else "FAIL",
        "returncode": int(completed.returncode),
        "stdout_sha256": _sha(stdout.encode("utf-8")),
        "stdout_size_bytes": len(stdout.encode("utf-8")),
        "stderr_sha256": _sha(stderr.encode("utf-8")),
        "stderr_size_bytes": len(stderr.encode("utf-8")),
        "request_sha256": _sha(request_raw),
        "predictor_result": predictor_result,
    }
    stdout_sha = _write_json_exclusive_readonly(
        output / "predictor_stdout_receipt.json", stdout_receipt
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Landlock predictor failed with return code {completed.returncode}")
    if audit["status"] != "PASS" or audit["package_source_path_opened_after_predictor_exec"]:
        raise RuntimeError("post-run open ledger violated the formal allowlist")
    assert predictor_result is not None
    artifact = output / Path(request["output_contract"]["relative_path"]).name
    verified = verify_prediction_artifact(
        artifact,
        expected_artifact_sha256=str(predictor_result["artifact_sha256"]),
        expected_seal_sha256=str(predictor_result["seal_sha256"]),
    )
    post = {
        **evidence,
        "filesystem_access_audit_sha256": audit_sha,
        "filesystem_access_audit_status": "PASS",
        "prediction_artifact_sha256": verified["artifact_sha256"],
        "prediction_seal_sha256": verified["seal_sha256"],
    }
    if set(post) != set(POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise ValueError("post-run evidence exact field set drift")
    validate_phase2_contract(
        {**request, "phase2_runtime_isolation_evidence": post}, evidence_phase="post_run"
    )
    formal = {
        "schema": "cvs.phase2.formal_post_run_runtime_evidence.v1",
        "status": "PROTOCOL_VALID",
        "formal_launch_authority": True,
        "protocol_valid_claim_allowed": True,
        "candidate_capsule_sha256": attestation["candidate_capsule_sha256"],
        "runtime_config_receipt_sha256": attestation["runtime_config_receipt_sha256"],
        "predictor_stdout_receipt_sha256": stdout_sha,
        "formal_post_run_contract_evidence": post,
        "formal_post_run_contract_sha256": _sha(_canonical(post)),
    }
    formal_sha = _write_json_exclusive_readonly(
        output / "phase2_formal_post_run_runtime_evidence.json", formal
    )
    return {
        "schema": "cvs.phase2.landlock_pinned_runner_result.v1",
        "status": "PROTOCOL_VALID",
        "formal_launch_authority": True,
        "command": command,
        "prediction_artifact_sha256": verified["artifact_sha256"],
        "prediction_seal_sha256": verified["seal_sha256"],
        "filesystem_access_audit_sha256": audit_sha,
        "formal_post_run_runtime_evidence_sha256": formal_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-closure-root", type=Path, required=True)
    parser.add_argument("--pre-run-evidence-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-package-seal-sha256", required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--strace-executable", type=Path, required=True)
    parser.add_argument("--landlock-launcher", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
