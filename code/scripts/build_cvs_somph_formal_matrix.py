#!/usr/bin/env python
"""Write the offline formal Stage2-B/C SOMP-H matrix contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.somph_formal_matrix import build_formal_matrix  # noqa: E402


CONTROLLER_POLICY_SCHEMA = "cvs.phase2.somph_offline_controller_policy.v1"
CONTROLLER_POLICY_KEYS = {
    "schema",
    "allowed_offline_controller_root",
    "forbidden_phase2_roots",
    "formal_launch_authority",
}
LOCKED_POLICY_ID = "qknnv42_stage2bc_extreme_light_route_20260716"
LOCKED_POLICY_PATH = Path(
    r"E:\type10-7\automation_reports\CV-SincNet"
    r"\qknnv42_stage2bc_extreme_light_route_20260716"
    r"\offline_controller\controller_policy.json"
)
LOCKED_POLICY_SHA256 = (
    "f079fc52123c06da51a30da6b40e32c02451bef31f301dc05f445224dd2d3c74"
)


def _load_controller_policy(
    path: Path, *, expected_sha256: str
) -> tuple[Path, list[Path]]:
    if (
        len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
    ):
        raise ValueError("expected controller policy SHA256 must be lowercase hex")
    parent = path.parent.resolve()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent / path.name, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("controller policy external SHA256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("controller policy JSON invalid") from exc
    if not isinstance(payload, dict) or set(payload) != CONTROLLER_POLICY_KEYS:
        raise ValueError("controller policy exact schema drift")
    if payload.get("schema") != CONTROLLER_POLICY_SCHEMA:
        raise ValueError("controller policy schema drift")
    if payload.get("formal_launch_authority") is not False:
        raise ValueError("controller policy cannot authorize launch")
    root_raw = payload.get("allowed_offline_controller_root")
    forbidden_raw = payload.get("forbidden_phase2_roots")
    if not isinstance(root_raw, str) or not root_raw:
        raise ValueError("controller policy allowed root invalid")
    if (
        not isinstance(forbidden_raw, list)
        or not forbidden_raw
        or any(not isinstance(value, str) or not value for value in forbidden_raw)
    ):
        raise ValueError("controller policy forbidden roots invalid")
    controller_root = Path(root_raw).resolve()
    forbidden_roots = [Path(value).resolve() for value in forbidden_raw]
    for forbidden in forbidden_roots:
        if (
            controller_root == forbidden
            or forbidden in controller_root.parents
            or controller_root in forbidden.parents
        ):
            raise ValueError("controller policy roots overlap")
    return controller_root, forbidden_roots


def _locked_controller_policy(policy_id: str) -> tuple[Path, str]:
    if policy_id != LOCKED_POLICY_ID:
        raise ValueError("controller policy id is not pre-registered in this release")
    return LOCKED_POLICY_PATH, LOCKED_POLICY_SHA256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-policy-id", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    policy_path, policy_sha256 = _locked_controller_policy(
        args.controller_policy_id
    )
    controller_root, forbidden_roots = _load_controller_policy(
        policy_path,
        expected_sha256=policy_sha256,
    )
    try:
        output.relative_to(controller_root)
    except ValueError as exc:
        raise ValueError("output must stay inside the offline controller root") from exc
    for forbidden in forbidden_roots:
        if (
            output == forbidden
            or forbidden in output.parents
            or controller_root == forbidden
            or forbidden in controller_root.parents
            or controller_root in forbidden.parents
        ):
            raise ValueError("offline controller root overlaps a forbidden Phase2 root")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_formal_matrix(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    temporary = output.with_name(output.name + ".tmp-" + str(os.getpid()))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(output, 0o444)
    print(json.dumps({"output": str(output), "size_bytes": len(payload)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
