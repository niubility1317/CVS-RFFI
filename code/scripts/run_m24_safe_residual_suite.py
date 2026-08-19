#!/usr/bin/env python3
"""Run a frozen ERBT-IDR M2.4 D0-D10 same-row prediction suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cvsrffi.stage2_m24_row_executor import run_m24_row_from_caches
from cvsrffi.stage2_m24_safe_residual import D1, M24_ARMS


SUITE_SCHEMA = "cvs.erbt_idr.m24.prediction_suite.v1"


def _arm_seed_plan(seed: int, arms: tuple[str, ...]) -> dict[str, int]:
    return {arm: int(seed) for arm in arms}


def _write(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o444)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-feature-cache-payload", required=True)
    parser.add_argument("--base-feature-cache-manifest", required=True)
    parser.add_argument("--base-feature-cache-payload-sha256", required=True)
    parser.add_argument("--base-feature-cache-manifest-sha256", required=True)
    parser.add_argument("--overlay-payload", required=True)
    parser.add_argument("--overlay-manifest", required=True)
    parser.add_argument("--overlay-payload-sha256", required=True)
    parser.add_argument("--overlay-manifest-sha256", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--row-prefix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arm", action="append", choices=M24_ARMS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    requested = tuple(args.arm) if args.arm else M24_ARMS
    arms = tuple(arm for arm in M24_ARMS if arm in requested)
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate M2.4 arm")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate M2.4 arm")
    if D1 not in arms:
        raise ValueError("every M2.4 suite requires the D1 parity reference")
    root = Path(args.output_root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    seed_plan = _arm_seed_plan(args.seed, arms)
    entries = []
    for arm in arms:
        arm_root = root / arm
        row_id = f"{args.row_prefix}__{arm}"
        receipt = run_m24_row_from_caches(
            arm=arm,
            row_id=row_id,
            receiver=args.receiver,
            base_feature_cache_payload=args.base_feature_cache_payload,
            base_feature_cache_manifest=args.base_feature_cache_manifest,
            base_feature_cache_payload_sha256=args.base_feature_cache_payload_sha256,
            base_feature_cache_manifest_sha256=args.base_feature_cache_manifest_sha256,
            overlay_payload=args.overlay_payload,
            overlay_manifest=args.overlay_manifest,
            overlay_payload_sha256=args.overlay_payload_sha256,
            overlay_manifest_sha256=args.overlay_manifest_sha256,
            output_root=arm_root,
            seed=seed_plan[arm],
            device=args.device,
        )
        entries.append({
            "arm": arm,
            "row_id": row_id,
            "receipt_path": str(arm_root / "row_execution_receipt.json"),
            "prediction_path": receipt["prediction"]["path"],
            "prediction_artifact_sha256": receipt["prediction"]["artifact_sha256"],
            "prediction_seal_sha256": receipt["prediction"]["seal_sha256"],
        })
        if arm == D1 and receipt["d1_historical_parity"]["prediction_disagreements"] != 0:
            raise RuntimeError("D1 physical256 parity failed; later M2.4 arms are not runnable")
    suite = {
        "schema": SUITE_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "receiver": args.receiver,
        "row_prefix": args.row_prefix,
        "method_seed": int(args.seed),
        "arm_seed_plan": seed_plan,
        "arms": list(arms),
        "entries": entries,
        "query_truth_opened": False,
    }
    _write(root / "suite_index.json", suite)
    print(json.dumps(suite, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
