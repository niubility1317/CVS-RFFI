#!/usr/bin/env python3
"""Independent truth-side scorer for immutable D103-R2 prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    score_prediction_artifact,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--truth-open-event-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root.resolve(strict=True)
    truth_path = args.truth_json.resolve(strict=True)
    output = args.output_json.resolve()
    event_path = args.truth_open_event_json.resolve()
    if (
        output.exists()
        or output.is_symlink()
        or event_path.exists()
        or event_path.is_symlink()
    ):
        raise FileExistsError(f"immutable score output exists: {output}")
    prediction_manifest_path = prediction_root / "prediction_manifest.json"
    manifest = _read_json(prediction_manifest_path)
    prediction_manifest_sha = sha256_file(prediction_manifest_path)
    prediction_manifest_mtime_ns = prediction_manifest_path.stat().st_mtime_ns
    event_time_ns = time.time_ns()
    if prediction_manifest_mtime_ns >= event_time_ns:
        raise ValueError("prediction manifest was not committed before truth open")
    event = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.truth_first_open.v1",
        "prediction_manifest_sha256": prediction_manifest_sha,
        "prediction_manifest_mtime_ns": prediction_manifest_mtime_ns,
        "truth_first_open_unix_ns": event_time_ns,
        "truth_opened_after_prediction_manifest_committed": True,
        "predictor_truth_access": False,
    }
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text(
        json.dumps(event, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    truth = _read_json(truth_path)
    if (
        manifest.get("row_count") != 63
        or manifest.get("query_truth_access") is not False
        or truth.get("package_count") != 21
        or truth.get("predictor_access") is not False
    ):
        raise ValueError("prediction/truth coverage drift")
    truth_by_package = {row["package_id"]: row for row in truth["packages"]}
    performance_rows = []
    agreement = []
    flips = []
    state_bytes = []
    query_mac = []
    for row in manifest["rows"]:
        artifact_path = prediction_root / row["path"]
        if artifact_path.stat().st_mtime_ns > prediction_manifest_mtime_ns:
            raise ValueError("prediction row was modified after manifest commit")
        if sha256_file(artifact_path) != row["sha256"]:
            raise ValueError("prediction artifact SHA drift")
        artifact = _read_json(artifact_path)
        truth_row = truth_by_package[row["package_id"]]
        if artifact["query_physical_ids"] != truth_row["query_physical_ids"]:
            raise ValueError("truth-side physical row alignment drift")
        scored = score_prediction_artifact(
            artifact, truth_row["query_truth_labels"]
        )
        scored["prediction_artifact_committed_before_truth"] = True
        scored["d102_prediction_committed_before_truth"] = True
        scored["truth_open_event_sha256"] = sha256_file(event_path)
        performance_rows.append(scored)
        int8 = artifact["int8_audit"]
        agreement.append(float(int8["top1_agreement"]))
        flips.append(int(int8["large_margin_flip_count"]))
        resource = artifact["resource_audit"]
        state_bytes.append(
            int(
                resource["actual_serialized_state_bytes"]
                if resource["actual_serialized_state_bytes"] is not None
                else resource["numeric_bundle_state_bytes"]
            )
        )
        query_mac.append(int(resource["post_backbone_mac_per_query"]))
    result = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.held_scores.v1",
        "performance_rows": performance_rows,
        "day_stability_rows": manifest["day_stability_rows"],
        "quantization_receipt": {
            "top1_agreement": min(agreement),
            "large_margin_flip_count": sum(flips),
            "persistent_fp_sidecar": False,
            "learning_arrays_int8_only": True,
            "row_count": len(agreement),
        },
        "resource_component": {
            "phase2_state_bytes": max(state_bytes),
            "post_backbone_mac_per_query": max(query_mac),
            "row_count": len(state_bytes),
        },
        "prediction_manifest_sha256": prediction_manifest_sha,
        "truth_sha256": sha256_file(truth_path),
        "truth_open_event_sha256": sha256_file(event_path),
        "prediction_artifact_committed_before_truth": True,
        "target_access": False,
        "formal_query_access": False,
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
