#!/usr/bin/env python
"""Build one exact-schema Phase2 request from a verified sealed predictor package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_runtime_contract import (  # noqa: E402
    PHASE2_FULL_CONTRACT,
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    validate_predictor_request,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    open_regular_member_same_fd,
    preflight_stage2_predictor_package,
    sha256_file,
)


def _read_json_regular(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{context} root must be an object")
    return payload


def _read_member_json(
    package_root: Path, descriptor: dict[str, Any], *, context: str
) -> dict[str, Any]:
    with open_regular_member_same_fd(package_root, descriptor["relative_path"]) as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} root must be an object")
    return payload


def _request_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": item["relative_path"],
        "sha256": item["sha256"],
        "size_bytes": item["size_bytes"],
        "artifact_role": item["artifact_role"],
        "schema": item["schema"],
    }


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    package_root = Path(args.predictor_package_root).resolve()
    seal_path = Path(args.detached_seal_path).resolve()
    expected_seal_sha256 = str(args.expected_seal_sha256).lower()
    runtime_evidence = _read_json_regular(
        Path(args.runtime_evidence_json), context="runtime evidence"
    )
    if set(runtime_evidence) != set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS):
        raise ValueError("runtime evidence must contain the exact pre-run field set")

    manifest, seal, _audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=seal_path,
        expected_seal_sha256=expected_seal_sha256,
    )
    if runtime_evidence["sealed_inference_package_sha256"] != expected_seal_sha256:
        raise ValueError("runtime evidence sealed package digest mismatch")
    if runtime_evidence["package_root_sha256"] != manifest["package_root_sha256"]:
        raise ValueError("runtime evidence package root digest mismatch")
    if (
        runtime_evidence["artifact_member_allowlist_sha256"]
        != seal["artifact_member_allowlist_sha256"]
    ):
        raise ValueError("runtime evidence member allowlist digest mismatch")

    members = {item["artifact_role"]: item for item in manifest["members"]}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        if f"support:{scenario}" not in members or f"query:{scenario}" not in members:
            raise ValueError("formal scenario artifacts are absent from the sealed package")
    k_shot = int(args.k_shot)
    if k_shot < 1 or k_shot > int(manifest["support_pool_max_k"]):
        raise ValueError("k_shot is outside the sealed nested support pool")
    tta_policy = _read_member_json(
        package_root, members["tta_policy"], context="TTA policy"
    )
    request = {
        "schema_version": "cvs.phase2.predict_request.v2",
        "request_id": str(args.request_id),
        "row_id": str(args.row_id),
        "stage": manifest["stage"],
        "receiver": manifest["receiver"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "k_shot": k_shot,
        "satellite_seed": int(manifest["seed"]),
        "candidate_lock_sha256": manifest["candidate_lock_sha256"],
        "package_root_sha256": manifest["package_root_sha256"],
        "runtime_code_sha256": runtime_evidence["runtime_code_sha256"],
        "registered_class_count": manifest["registered_class_count"],
        "registered_classes": manifest["registered_classes"],
        "support_artifacts": [
            _request_descriptor(members[f"support:{scenario}"])
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        ],
        "query_artifacts": [
            _request_descriptor(members[f"query:{scenario}"])
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        ],
        "checkpoint_artifact": _request_descriptor(members["checkpoint"]),
        "adapter_artifact": _request_descriptor(members["adapter"]),
        "head_artifact": _request_descriptor(members["head"]),
        "tta_policy": tta_policy,
        "tta_policy_sha256": members["tta_policy"]["sha256"],
        "output_contract": {
            "schema": "cvs.phase2.prediction.v2",
            "relative_path": str(args.output_relative_path),
            "sealed_immutable_required": True,
        },
        "phase2_runtime_isolation_evidence": runtime_evidence,
        **{key: manifest[key] for key in PHASE2_FULL_CONTRACT},
    }
    validate_predictor_request(request)
    output = Path(args.output_json).resolve()
    _write_json_new(output, request)
    return {
        "request_json": str(output),
        "request_sha256": sha256_file(output),
        "request_id": request["request_id"],
        "package_root_sha256": request["package_root_sha256"],
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "k_shot": k_shot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-package-root", type=Path, required=True)
    parser.add_argument("--detached-seal-path", type=Path, required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--runtime-evidence-json", type=Path, required=True)
    parser.add_argument("--k-shot", type=int, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output-relative-path", default="prediction_artifact.cvspred")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_request(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
