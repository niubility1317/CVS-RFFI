#!/usr/bin/env python3
"""Seal a Phase2 plan against reusable feature-cache bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_ablation_release import seal_stage2_plan


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--binding-registry", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument(
        "--python-environment-id",
        default="CVS-RFFI",
    )
    parser.add_argument("--review-p0-count", type=int, required=True)
    parser.add_argument("--review-p1-count", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shared-view-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite sealed plan: {output}"
        )
    plan = seal_stage2_plan(
        _load(args.plan),
        _load(args.binding_registry),
        run_id=args.run_id,
        request_root=args.request_root,
        run_root=args.run_root,
        log_root=args.log_root,
        python_environment_id=args.python_environment_id,
        review_p0_count=args.review_p0_count,
        review_p1_count=args.review_p1_count,
        device=args.device,
        shared_view_count=args.shared_view_count,
        write_requests=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            plan,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "logical_row_count": plan["logical_row_count"],
                "physical_execution_count": plan[
                    "physical_execution_count"
                ],
                "reused_physical_count": plan[
                    "reused_physical_count"
                ],
                "alias_logical_count": plan[
                    "alias_logical_count"
                ],
                "formal_launch_authority": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
