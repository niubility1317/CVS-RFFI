#!/usr/bin/env python3
"""Build J1/J2/J3 source-free deltas for the paired R5D92 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_sf_d3_erbt_plan import build_d3_config  # noqa: E402
from cvsrffi.target_only_progressive_adapt import SFTAPFTConfig  # noqa: E402
from cvsrffi.target_only_progressive_runner import (  # noqa: E402
    run_sf_tapft_deploy_no_query,
)


METHODS = ("J1_R0", "J2_R3", "J3_R5D92_G")


def _build_adapt_config(
    data_plan: Mapping[str, Any],
    method_matrix: Mapping[str, Any],
    scenario: str,
    method: str,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown R5D92 method: {method}")
    if method_matrix.get("schema") != "cvs.sf_tapft.slim_matrix.v1":
        raise ValueError("R5D92 method matrix schema mismatch")
    raw = method_matrix.get("base_sf_tapft")
    if not isinstance(raw, Mapping):
        raise ValueError("R5D92 method matrix is missing base_sf_tapft")
    method_config = SFTAPFTConfig(**dict(raw))
    if (
        method_config.trainability_profile != "p1_head_norm"
        or method_config.norm_rules != (("t3", "weight_bias"),)
        or method_config.phase_steps != (300, 150, 70)
    ):
        raise ValueError("R5D92 E0 method lock drift")
    config = build_d3_config(data_plan, scenario)
    config["candidate_id"] = f"{method}_{scenario.upper()}"
    config["sf_tapft"] = dict(raw)
    config["sf_tapft"]["validation_steps"] = []
    config["sf_tapft"]["rse_snapshot_steps"] = []
    config["sf_tapft"]["checkpoint_average_top_k"] = 1
    config["sf_tapft"]["oof_temperature_calibration"] = False
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-plan", type=Path, required=True)
    parser.add_argument("--method-matrix", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--registered-support", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_plan = json.loads(args.data_plan.read_text(encoding="utf-8-sig"))
    method_matrix = json.loads(args.method_matrix.read_text(encoding="utf-8-sig"))
    config = _build_adapt_config(data_plan, method_matrix, args.scenario, args.method)
    if args.method == "J1_R0":
        mode, options = "fixed", {}
    elif args.method == "J2_R3":
        mode = "delta_ensemble"
        options = {"ensemble_count": 2, "per_class": 8, "polish_steps": 30}
    else:
        if args.registered_support is None:
            raise ValueError("J3_R5D92_G requires --registered-support")
        mode = "r5d92"
        options = {
            "registered_support_path": str(args.registered_support),
            "steps": [250, 350, 520],
            "polish_steps": 30,
            "d92_seed": 713101,
            "soft_budget_seconds": 240.0,
        }
    result = run_sf_tapft_deploy_no_query(
        config,
        args.output_root,
        device=args.device,
        deployment_inplace=False,
        emit_clean_single_bundle=False,
        rse_mode=mode,
        rse_options=options,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
