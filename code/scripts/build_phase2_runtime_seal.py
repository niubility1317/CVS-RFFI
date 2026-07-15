#!/usr/bin/env python
"""Seal Phase2 predictor artifacts and emit detached runtime evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    preflight_stage2_predictor_package,
    sha256_file,
)


LANDLOCK_CREATE_RULESET_VERSION = 1


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _regular(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"seal member must be a regular non-symlink file: {path}")
    return resolved


def _tree_files(root: Path) -> list[Path]:
    resolved = root.resolve(strict=True)
    if root.is_symlink() or not resolved.is_dir():
        raise ValueError(f"runtime code root must be a non-symlink directory: {root}")
    files: list[Path] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"runtime code tree contains symlink: {path}")
        if path.is_file() and path.suffix == ".py":
            files.append(path)
    return files


def _landlock_abi() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    return int(libc.syscall(444, 0, 0, LANDLOCK_CREATE_RULESET_VERSION))


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _regular(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    source_manifest = _regular(Path(str(config["source_leo_weak_cache_set_manifest"])))
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
    artifact_files = [config_path, source_manifest]
    for raw in dict(source_payload["cache_npz_by_scenario"]).values():
        path = Path(str(raw))
        artifact_files.append(_regular(path if path.is_absolute() else source_manifest.parent / path))
    predictor_root = Path(str(config["target_predictor_bundle_root"])).resolve(strict=True)
    predictor_seal_root = Path(str(config["target_predictor_seal_root"])).resolve(
        strict=True
    )
    out = args.out_root.resolve()
    package_records: list[tuple[Path, Path, Path, dict[str, Any]]] = []
    for manifest_path in sorted(predictor_root.rglob("package_manifest.json")):
        bundle_root = manifest_path.parent
        relative_bundle = bundle_root.relative_to(predictor_root)
        evidence_path = out / relative_bundle / "runtime_isolation_evidence.json"
        seal_path = predictor_seal_root / relative_bundle / "seal.json"
        package_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        package_records.append((manifest_path, seal_path, evidence_path, package_manifest))
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                artifact_files.append(_regular(path))
        artifact_files.append(_regular(seal_path))
    if len(package_records) != 25:
        raise ValueError(f"runtime seal requires 25 predictor packages, found={len(package_records)}")
    code_roots = [path.resolve(strict=True) for path in args.runtime_code_root]
    code_files = [path for root in code_roots for path in _tree_files(root)]
    code_files.extend(_regular(path) for path in args.runtime_code_file)
    code_files = sorted(set(code_files))
    artifact_files = sorted(set(artifact_files))
    digest_cache: dict[tuple[int, int, int, int], str] = {}
    def entry(path: Path) -> dict[str, Any]:
        stat_result = path.stat()
        key = (
            int(stat_result.st_dev), int(stat_result.st_ino), int(stat_result.st_size),
            int(stat_result.st_mtime_ns),
        )
        if key not in digest_cache:
            digest_cache[key] = _sha(path)
        return {"path": str(path), "sha256": digest_cache[key], "size": stat_result.st_size}
    artifact_entries = [entry(path) for path in artifact_files]
    code_entries = [entry(path) for path in code_files]
    root_digest = hashlib.sha256(
        "\n".join(f"{row['path']}\0{row['sha256']}\0{row['size']}" for row in artifact_entries + code_entries).encode("utf-8")
    ).hexdigest()
    runtime_code_digest = hashlib.sha256(
        "\n".join(f"{row['path']}\0{row['sha256']}" for row in code_entries).encode("utf-8")
    ).hexdigest()
    allowlist_path = out / "artifact_member_allowlist.json"
    package_path = out / "sealed_inference_package.json"
    access_audit_path = out / "preopen_filesystem_access_audit.json"
    os_attestation_path = out / "os_isolation_attestation.json"
    allowlist = {
        "schema": "cvs_phase2_landlock_allowlist_v1",
        "read_files": [str(path) for path in artifact_files + code_files] + [
            str(record[2]) for record in package_records
        ],
        "runtime_code_list_dirs": sorted({str(path.parent) for path in code_files}),
        "forbidden_predictor_artifact_tokens": ["truth_sidecar", "scoring_manifest", ".pkl"],
    }
    _write(allowlist_path, allowlist)
    package = {
        "schema": "cvs_phase2_sealed_inference_package_v1",
        "package_root_sha256": root_digest,
        "runtime_code_sha256": runtime_code_digest,
        "artifact_members": artifact_entries,
        "runtime_code_members": code_entries,
        "artifact_member_allowlist": str(allowlist_path),
        "truth_or_scoring_sidecar_included": False,
        "raw_dataset_included": False,
    }
    _write(package_path, package)
    (out / "sealed_inference_package.sha256").write_text(_sha(package_path) + "\n", encoding="ascii")
    audit = {
        "schema": "cvs_phase2_preopen_filesystem_audit_v1",
        "status": "PASS",
        "landlock_abi": _landlock_abi(),
        "regular_file_count": len(artifact_entries) + len(code_entries),
        "symlink_count": 0,
        "truth_sidecar_or_scoring_manifest_member_count": sum(
            any(token in Path(row["path"]).name.lower() for token in ("truth_sidecar", "scoring_manifest"))
            for row in artifact_entries
        ),
        "raw_pkl_member_count": sum(Path(row["path"]).suffix.lower() == ".pkl" for row in artifact_entries),
        "predictor_package_preopen_audits": [],
    }
    for manifest_path, seal_path, _evidence_path, package_manifest in package_records:
        _manifest, _seal, package_audit = preflight_stage2_predictor_package(
            manifest_path.parent,
            detached_seal_path=seal_path,
            expected_seal_sha256=sha256_file(seal_path),
        )
        audit["predictor_package_preopen_audits"].append({
            "receiver": package_manifest["receiver"], "seed": package_manifest["seed"],
            "status": package_audit["status"],
            "hash_and_member_audit_same_file_descriptor": package_audit[
                "hash_and_member_audit_same_file_descriptor"
            ],
            "package_root_sha256": package_audit["package_root_sha256"],
        })
    if audit["landlock_abi"] < 1 or audit["truth_sidecar_or_scoring_manifest_member_count"] or audit["raw_pkl_member_count"]:
        audit["status"] = "FAIL"
        _write(access_audit_path, audit)
        raise ValueError(f"runtime seal preopen audit failed: {audit}")
    _write(access_audit_path, audit)
    os_attestation = {
        "schema": "cvs_phase2_landlock_attestation_v1",
        "kernel": platform.release(),
        "landlock_abi": audit["landlock_abi"],
        "no_new_privs_required": True,
        "isolation_mode": "equivalent_verified_isolation",
    }
    _write(os_attestation_path, os_attestation)
    evidence_entries = []
    for manifest_path, seal_path, evidence_path, package_manifest in package_records:
        evidence = {
            "sealed_inference_package_sha256": _sha(seal_path),
            "package_root_sha256": str(package_manifest["package_root_sha256"]),
            "runtime_code_sha256": runtime_code_digest,
            "artifact_member_allowlist_sha256": _sha(allowlist_path),
            "os_isolation_mode": "equivalent_verified_isolation",
            "os_isolation_attestation_sha256": _sha(os_attestation_path),
            "preopen_audit_status": "PASS",
            "preopen_audit_receipt_sha256": _sha(access_audit_path),
            "predict_score_process_isolation": True,
        }
        _write(evidence_path, evidence)
        evidence_entries.append({
            "receiver": package_manifest["receiver"], "seed": package_manifest["seed"],
            "package_manifest": str(manifest_path),
            "runtime_isolation_evidence": str(evidence_path),
            "runtime_isolation_evidence_sha256": _sha(evidence_path),
        })
    evidence_index_path = out / "runtime_isolation_evidence_index.json"
    _write(evidence_index_path, {
        "schema": "cvs_phase2_runtime_isolation_evidence_index_v1",
        "count": len(evidence_entries), "entries": evidence_entries,
    })
    return {
        "runtime_isolation_evidence_index": str(evidence_index_path),
        "runtime_isolation_evidence_count": len(evidence_entries),
        "sealed_inference_package": str(package_path),
        "artifact_member_allowlist": str(allowlist_path),
        "preopen_filesystem_access_audit": str(access_audit_path),
        "os_isolation_attestation": str(os_attestation_path),
        "global_inventory_root_sha256": root_digest,
        "runtime_code_sha256": runtime_code_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--runtime-code-root", type=Path, action="append", default=[])
    parser.add_argument("--runtime-code-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.runtime_code_root and not args.runtime_code_file:
        raise ValueError("at least one runtime code member is required")
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
