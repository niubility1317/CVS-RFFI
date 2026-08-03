#!/usr/bin/env python3
"""Run the one sealed A-only D128-A-ONE18 truth-free prediction entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi import stage2_d127_s0_package_adapter as adapter
from cvsrffi import stage2_d128_a_one18 as one


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-plan", required=True, type=Path)
    parser.add_argument("--prepared-plan-sha256", required=True)
    parser.add_argument("--method-lock", required=True, type=Path)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--d106-context", required=True, type=Path)
    parser.add_argument("--d106-context-sha256", required=True)
    parser.add_argument("--phase1-a-asset-bundle", required=True, type=Path)
    parser.add_argument("--phase1-a-asset-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, _plan_file_sha = adapter.load_d127_s0_prepared_plan(
        args.prepared_plan, expected_sha256=args.prepared_plan_sha256
    )
    prepared = one.materialize_d128_a_one18_rows(
        method_lock_path=args.method_lock,
        expected_method_lock_sha256=args.method_lock_sha256,
        d106_context_path=args.d106_context,
        expected_d106_context_sha256=args.d106_context_sha256,
        prepared_plan=plan,
        device=args.device,
    )
    model, _checkpoint_receipt = one.load_d128_checkpoint(
        checkpoint_path=args.checkpoint, prepared_plan=plan, device=args.device
    )
    asset, asset_receipt = one.load_d128_a_single_candidate_asset(
        bundle_dir=args.phase1_a_asset_bundle,
        expected_manifest_sha256=args.phase1_a_asset_manifest_sha256,
        prepared_plan=plan,
        device=args.device,
    )
    prediction = one.run_d128_a_one18_prediction(
        model=model,
        asset=asset,
        prepared=prepared,
        prepared_plan=plan,
        phase1_asset_manifest_sha256=args.phase1_a_asset_manifest_sha256,
        phase1_asset_receipt=asset_receipt,
    )
    output = one.write_d128_a_one18_prediction_exclusive(args.output, prediction, prepared_plan=plan)
    value: dict[str, Any] = {
        "status": "D128_A_ONE18_PREDICTION_COMPLETE",
        "candidate_id": one.CANDIDATE_ID,
        "row_pair_count": prediction["row_pair_count"],
        "state_row_count": prediction["state_row_count"],
        "truth_loaded": False,
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
        "prediction_sha256": prediction["prediction_sha256"],
    }
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
