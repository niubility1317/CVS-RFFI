#!/usr/bin/env python3
"""Run the B0/B3/C1/C2 ERBT-IDR M2.8 paired screen or full125 grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_m24_row_executor import run_m28_local_flip_risk_row_from_base_cache
from cvsrffi.stage2_m24_safe_residual import D1
from cvsrffi.stage2_m25_anchored_residual import B3
from cvsrffi.stage2_m28_local_flip_risk import C1, C2
from scripts.run_m24_d1_refit_matrix import (
    DEFAULT_CONDITIONS,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    _cache_root,
    _canonical_manifest_sha,
    _write_exclusive,
)


MATRIX_SCHEMA = "cvs.erbt_idr.m28.local_flip_risk_prediction_matrix.v1"
EVIDENCE_ARMS = (D1, B3, C1, C2)
SCREEN_RECEIVERS = ("3-19", "8-8")
SCREEN_SEEDS = (7282101,)
SCREEN_CONDITIONS = ((5, 20), (10, 5))


def matrix_spec(kind: str) -> dict[str, Any]:
    if kind == "screen":
        receivers, seeds, conditions = SCREEN_RECEIVERS, SCREEN_SEEDS, SCREEN_CONDITIONS
    elif kind == "full125":
        receivers, seeds, conditions = DEFAULT_RECEIVERS, DEFAULT_SEEDS, DEFAULT_CONDITIONS
    else:
        raise ValueError("matrix kind must be screen or full125")
    identities = len(receivers) * len(seeds) * len(conditions)
    return {
        "kind": kind,
        "receivers": tuple(receivers),
        "seeds": tuple(seeds),
        "conditions": tuple(conditions),
        "paired_input_identity_count": identities,
        "expected_method_rows": identities * len(EVIDENCE_ARMS),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--matrix-kind", choices=("screen", "full125"), required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--supplemental-feature-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-workers", type=int, default=2)
    return parser


def _validate_receipt(task: dict[str, Any], receipt: dict[str, Any]) -> None:
    if receipt.get("status") != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED":
        raise ValueError("receipt is not truth-unopened prediction-complete")
    for field in ("arm", "receiver", "k_shot", "new_class_count"):
        if receipt.get(field) != task.get(field):
            raise ValueError(f"task/receipt identity drift: {field}")
    if task["arm"] == D1:
        parity = receipt.get("d1_historical_parity") or {}
        if parity.get("prediction_disagreements") != 0 or parity.get(
            "before_prediction_disagreements"
        ) != 0:
            raise ValueError(f"B0 parity drift: {parity}")
    if task["arm"] in (C1, C2):
        if any(
            audit.get("query_application", {}).get("row_source_allowlist")
            != ["B0", "B3"]
            for audit in receipt.get("scenario_audit", {}).values()
        ):
            raise ValueError("M2.8 row-source allowlist drift")


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(task["manifest_path"])
    payload_path = Path(task["payload_path"])
    output_root = Path(task["output_root"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    receipt = run_m28_local_flip_risk_row_from_base_cache(
        arm=str(task["arm"]),
        row_id=str(task["row_id"]),
        receiver=str(task["receiver"]),
        base_feature_cache_payload=payload_path,
        base_feature_cache_manifest=manifest_path,
        base_feature_cache_payload_sha256=str(manifest["payload_sha256"]),
        base_feature_cache_manifest_sha256=_canonical_manifest_sha(manifest_path),
        output_root=output_root / str(task["row_id"]),
        seed=int(task["method_seed"]),
        device=str(task["device"]),
    )
    _validate_receipt(task, receipt)
    return {
        "row_id": task["row_id"],
        "arm": receipt["arm"],
        "receiver": receipt["receiver"],
        "method_seed": task["method_seed"],
        "k_shot": receipt["k_shot"],
        "new_class_count": receipt["new_class_count"],
        "support_seed": int(manifest["support_seed"]),
        "query_seed": int(manifest["query_seed"]),
        "new_class_draw_seed": int(manifest["new_class_draw_seed"]),
        "capsule_id": str(manifest["capsule_id"]),
        "split_id": str(manifest["split_id"]),
        "feature_cache_root": str(manifest_path.parent),
        "receipt_path": str(
            output_root / str(task["row_id"]) / "row_execution_receipt.json"
        ),
        "prediction": receipt["prediction"],
    }


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError("max-workers must be between 1 and 8")
    spec = matrix_spec(args.matrix_kind)
    roots = [Path(args.feature_root).absolute()]
    if args.supplemental_feature_root:
        roots.append(Path(args.supplemental_feature_root).absolute())
    feature_roots = tuple(roots)
    output_root = Path(args.output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    tasks: list[dict[str, Any]] = []
    for receiver in spec["receivers"]:
        for method_seed in spec["seeds"]:
            for k_shot, new_count in spec["conditions"]:
                cache_root = _cache_root(
                    feature_roots, receiver, method_seed, k_shot, new_count
                )
                for arm in EVIDENCE_ARMS:
                    row_id = f"rx{receiver}_m{method_seed}_k{k_shot}_new{new_count}__{arm}"
                    tasks.append(
                        {
                            "row_id": row_id,
                            "arm": arm,
                            "receiver": receiver,
                            "method_seed": method_seed,
                            "k_shot": k_shot,
                            "new_class_count": new_count,
                            "manifest_path": str(cache_root / "features.manifest.json"),
                            "payload_path": str(cache_root / "features.npz"),
                            "output_root": str(output_root),
                            "device": args.device,
                        }
                    )
    if len(tasks) != spec["expected_method_rows"]:
        raise ValueError("M2.8 task matrix size drift")
    completed: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            print(
                json.dumps(
                    {"completed": row["row_id"], "count": len(completed)},
                    sort_keys=True,
                ),
                flush=True,
            )
    completed.sort(key=lambda row: str(row["row_id"]))
    matrix = {
        "schema": MATRIX_SCHEMA,
        "run_id": str(args.run_id),
        "matrix_kind": args.matrix_kind,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "row_count": len(completed),
        "paired_input_identity_count": spec["paired_input_identity_count"],
        "method_rows_per_arm": spec["paired_input_identity_count"],
        "scenario_unit_count": len(completed) * 3,
        "primary_d92_e0_baseline": "P2-A1_NO_RF32",
        "reference_arm": D1,
        "performance_branch": B3,
        "receivers": list(spec["receivers"]),
        "method_seeds": list(spec["seeds"]),
        "conditions": [
            {"k_shot": k, "new_class_count": n}
            for k, n in spec["conditions"]
        ],
        "arms": list(EVIDENCE_ARMS),
        "feature_roots": [str(root) for root in feature_roots],
        "entries": completed,
        "query_truth_opened": False,
    }
    _write_exclusive(output_root / "matrix_index.json", matrix)
    print(
        json.dumps(
            {"status": matrix["status"], "row_count": len(completed)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
