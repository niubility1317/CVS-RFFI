#!/usr/bin/env python3
"""Assemble the complete held gate package from independently produced evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-json", type=Path, required=True)
    parser.add_argument("--d102-provenance-json", type=Path, required=True)
    parser.add_argument("--tx-probe-json", type=Path, required=True)
    parser.add_argument("--runner-resource-json", type=Path, required=True)
    parser.add_argument("--matrix-status-json", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable held receipt output exists: {output}")
    scores = _read(args.scores_json)
    d102 = _read(args.d102_provenance_json)
    tx = _read(args.tx_probe_json)
    runner = _read(args.runner_resource_json)
    matrix = _read(args.matrix_status_json)
    split = _read(args.source_split_manifest)
    if (
        len(scores.get("performance_rows", ())) != 63
        or len(scores.get("day_stability_rows", ())) != 49
        or tx.get("fold_count") != 7
        or d102.get("fold_count") != 49
        or split.get("protocol_schema") != "p2_min_v1"
        or split.get("partition", {}).get("counts")
        != {"L_s": 588, "U_s": 5292, "source_val": 2520}
        or matrix.get("status") != "ARTIFACTS_COMPLETE"
        or matrix.get("fit_manifest_validation_pass") is not True
        or matrix.get("fit_access_receipt_count") != 246
    ):
        raise ValueError("held evidence coverage or split drift")
    required_runner = {
        "total_gpu_hours",
        "peak_memory_bytes",
        "run_root_bytes",
        "completed_fit_count",
        "completed_meta_steps",
    }
    if set(runner) != required_runner:
        raise ValueError("runner resource key closure drift")
    resource = {
        **runner,
        "phase2_state_bytes": int(
            scores["resource_component"]["phase2_state_bytes"]
        ),
        "post_backbone_mac_per_query": int(
            scores["resource_component"]["post_backbone_mac_per_query"]
        ),
    }
    access = {
        "protocol_schema": "p2_min_v1",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_validation_ratio": 0.30,
        "u_s_tx_label_access": False,
        "source_validation_gradient_access": False,
        "source_validation_asset_access": False,
        "target_access": False,
        "formal_query_access": False,
        "query_fit_rows": 0,
        "derived_from_fit_access_receipt_count": int(
            matrix["fit_access_receipt_count"]
        ),
        "all_fit_manifests_identity_bound": bool(
            matrix["fit_manifest_validation_pass"]
        ),
    }
    receivers = sorted(
        {str(row["held_receiver"]) for row in scores["performance_rows"]}
    )
    classes = sorted(
        {
            str(row["held_class"])
            for row in scores["performance_rows"]
            if row.get("held_class") is not None
        }
    )
    if len(receivers) != 7 or len(classes) != 6:
        raise ValueError("held receiver/class coverage drift")
    result = {
        "receiver_ids": receivers,
        "class_ids": classes,
        "performance_rows": scores["performance_rows"],
        "day_stability_rows": scores["day_stability_rows"],
        "d102_provenance": d102,
        "tx_probe_rows": tx["folds"],
        "quantization_receipt": scores["quantization_receipt"],
        "resource_receipt": resource,
        "access_receipt": access,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
