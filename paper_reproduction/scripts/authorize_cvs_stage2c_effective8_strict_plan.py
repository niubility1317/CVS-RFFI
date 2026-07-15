#!/usr/bin/env python3
"""Authorize the strict 300-cell matrix only from a copied N607 smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import validate_strict_plan


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def authorize(plan: Mapping[str, Any], smoke: Mapping[str, Any]) -> dict[str, Any]:
    plan = validate_strict_plan(plan)
    if plan.get("launch_authority") is not False:
        raise ValueError("only a fail-closed pre-smoke plan can be authorized")
    cell = smoke.get("cell_receipt")
    if not isinstance(cell, dict):
        raise ValueError("smoke cell receipt is absent")
    checks = (
        smoke.get("schema") == "cvs.stage2c.effective8.n607_landlock_smoke.v1",
        smoke.get("status") == "PASS",
        smoke.get("matrix_launch_authority_recommended") is True,
        smoke.get("candidate_capsule_sha256") == plan["candidate_capsule_sha256"],
        smoke.get("package_id") == plan["smoke_package_id"],
        int(smoke.get("k_shot", -1)) == int(plan["smoke_k_shot"]),
        cell.get("status") == "PROTOCOL_VALID",
        cell.get("package_id") == plan["smoke_package_id"],
        int(cell.get("k_shot", -1)) == int(plan["smoke_k_shot"]),
        cell.get("candidate_capsule_sha256") == plan["candidate_capsule_sha256"],
        int(cell.get("formal_scenario_row_count", -1)) == 3,
        smoke.get("cell_receipt_sha256") == hashlib.sha256(_canonical(cell)).hexdigest(),
    )
    if not all(checks):
        raise ValueError("N607 Landlock smoke receipt is not authority-bearing")
    authorized = deepcopy(plan)
    authorized["launch_authority"] = True
    authorized["authority_state"] = "N607_LANDLOCK_SMOKE_PASS"
    authorized["n607_smoke_receipt_sha256"] = hashlib.sha256(_canonical(smoke)).hexdigest()
    validate_strict_plan(authorized)
    return authorized


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan_manifest.read_text(encoding="utf-8-sig"))
    smoke = json.loads(args.smoke_receipt.read_text(encoding="utf-8-sig"))
    authorized = authorize(plan, smoke)
    with args.output_manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(authorized, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"launch_authority": True, "authority_state": authorized["authority_state"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
