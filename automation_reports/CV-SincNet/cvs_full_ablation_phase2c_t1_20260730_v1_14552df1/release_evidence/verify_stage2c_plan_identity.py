"""Bind the exact 1425-row Stage2-C source and sealed plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

from cvsrffi.full_ablation_spec import validate_plan_rows
from cvsrffi.stage2_ablation_release import (
    sha256_object,
    validate_sealed_stage2_plan,
)


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7282101, 7282102, 7282103)
VARIANT_K = {
    20: (1, 2, 5, 10),
    5: (10,),
}
ARMS = (
    "P2-A0",
    "P2-B0",
    "P2-BASE-ADAPTER-HEAD",
    "P2-BASE-COSINE",
    "P2-BASE-DIAG-LDA",
    "P2-BASE-EUCLIDEAN",
    "P2-BASE-FULL-BLOCK-LDA",
    "P2-BASE-POOLED-LW-LDA",
    "P2-BASE-QKNN",
    "P2-C3",
    "P2-D0",
    "P2-D1",
    "P2-D2",
    "P2-E0",
    "P2-F0",
    "P2-F1",
    "P2-F2",
    "P2-F3",
    "P2-FULL",
)
EXPECTED_LOGICAL = 1425
EXPECTED_PHYSICAL = 1350
NEW_CLASS_DRAW_SEED = 7282401


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected SHA-256 must be lowercase hexadecimal")
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"plan is not a regular file: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"plan SHA-256 drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plan must be a JSON object")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--expected-source-plan-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--sealed-plan", type=Path)
    parser.add_argument("--expected-sealed-plan-sha256")
    parser.add_argument("--expected-run-id")
    return parser.parse_args()


def _validate_source(
    plan: dict[str, Any],
    *,
    expected_git_commit: str,
) -> None:
    rows = list(plan.get("rows") or [])
    validate_plan_rows(rows)
    expected_identities = {
        (receiver, method_seed, k_shot, new_class_count)
        for receiver in RECEIVERS
        for method_seed in METHOD_SEEDS
        for new_class_count, k_values in VARIANT_K.items()
        for k_shot in k_values
    }
    identity_counts: Counter[tuple[str, int, int, int]] = Counter()
    row_keys: set[str] = set()
    arms: set[str] = set()
    for row in rows:
        identity = (
            str(row["receiver_id"]),
            int(row["method_seed"]),
            int(row["k_shot"]),
            int(row["new_class_count"]),
        )
        identity_counts[identity] += 1
        row_key = str(row["row_key"])
        if row_key in row_keys:
            raise ValueError("Stage2-C source plan contains duplicate row keys")
        row_keys.add(row_key)
        arms.add(str(row["ablation_id"]))
        method_seed = int(row["method_seed"])
        if (
            row.get("phase") != "stage2c"
            or row.get("stage") != "screening"
            or row.get("protocol_schema") != "p2_min_v1"
            or int(row["support_seed"]) != method_seed + 100
            or int(row["query_seed"]) != method_seed + 200
            or int(row["new_class_draw_seed"]) != NEW_CLASS_DRAW_SEED
            or row.get("git_commit") != expected_git_commit
        ):
            raise ValueError("Stage2-C source row identity drift")
    if (
        plan.get("schema") != "cvs.full_ablation.plan.v1"
        or plan.get("phase") != "phase2"
        or plan.get("stage") != "screening"
        or plan.get("phase2_matrix") != "stage2c"
        or plan.get("git_commit") != expected_git_commit
        or plan.get("python_environment_id") != "CVS-RFFI"
        or plan.get("formal_launch_authority") is not False
        or int(plan.get("logical_row_count", -1)) != EXPECTED_LOGICAL
        or len(rows) != EXPECTED_LOGICAL
        or set(identity_counts) != expected_identities
        or any(count != len(ARMS) for count in identity_counts.values())
        or arms != set(ARMS)
    ):
        raise ValueError("Stage2-C source plan matrix drift")


def main() -> int:
    args = _parse_args()
    source_path = args.source_plan.absolute()
    source = _load(source_path, args.expected_source_plan_sha256)
    _validate_source(source, expected_git_commit=args.expected_git_commit)
    result: dict[str, Any] = {
        "status": "SOURCE_PLAN_VERIFIED",
        "source_plan": str(source_path),
        "source_plan_sha256": args.expected_source_plan_sha256,
        "git_commit": args.expected_git_commit,
        "logical_row_count": EXPECTED_LOGICAL,
        "identity_count": 75,
        "arm_count": len(ARMS),
    }
    if args.sealed_plan is not None:
        if not args.expected_sealed_plan_sha256 or not args.expected_run_id:
            raise ValueError("sealed-plan verification requires SHA-256 and run ID")
        sealed_path = args.sealed_plan.absolute()
        sealed = _load(sealed_path, args.expected_sealed_plan_sha256)
        validate_sealed_stage2_plan(sealed)
        physical_rows = list(sealed.get("physical_rows") or [])
        sealed_logical_count = sum(
            len(list(physical.get("logical_rows") or []))
            for physical in physical_rows
        )
        if (
            sealed.get("run_id") != args.expected_run_id
            or sealed.get("git_commit") != args.expected_git_commit
            or sealed.get("formal_launch_authority") is not True
            or int(sealed.get("logical_row_count", -1)) != EXPECTED_LOGICAL
            or int(sealed.get("physical_execution_count", -1))
            != EXPECTED_PHYSICAL
            or sealed.get("source_plan_sha256") != sha256_object(source)
            or sealed_logical_count != EXPECTED_LOGICAL
            or len(physical_rows) != EXPECTED_PHYSICAL
        ):
            raise ValueError("Stage2-C sealed plan identity drift")
        result.update(
            {
                "status": "SEALED_PLAN_VERIFIED",
                "sealed_plan": str(sealed_path),
                "sealed_plan_sha256": args.expected_sealed_plan_sha256,
                "run_id": args.expected_run_id,
                "physical_execution_count": EXPECTED_PHYSICAL,
            }
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
