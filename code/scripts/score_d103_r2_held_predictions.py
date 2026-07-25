#!/usr/bin/env python3
"""Independent truth-side scorer for immutable D103-R2 prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    package_id,
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
    rows = manifest.get("rows")
    day_stability_rows = manifest.get("day_stability_rows")
    if (
        set(manifest)
        != {
            "schema",
            "row_count",
            "rows",
            "day_stability_rows",
            "query_truth_access",
            "target_access",
            "formal_query_access",
            "sealed_at_unix_ns",
        }
        or manifest.get("schema")
        != "cvs.d103_r2.rxid_crossreceiver.held_predictions.v1"
        or manifest.get("row_count") != 63
        or manifest.get("query_truth_access") is not False
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
        or not isinstance(rows, list)
        or len(rows) != 63
        or not isinstance(day_stability_rows, list)
        or len(day_stability_rows) != 49
    ):
        raise ValueError("prediction coverage drift before truth open")
    actual_keys: list[tuple[str, str | None, int]] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "held_receiver",
                "held_class",
                "K",
                "package_id",
                "path",
                "sha256",
            }
            or not isinstance(row.get("held_receiver"), str)
            or not str(row["held_receiver"])
            or (
                row.get("held_class") is not None
                and (
                    not isinstance(row.get("held_class"), str)
                    or not str(row["held_class"])
                )
            )
            or type(row.get("K")) is not int
            or int(row["K"]) not in (1, 5, 10)
        ):
            raise ValueError("prediction manifest row closure drift")
        actual_keys.append(
            (
                str(row["held_receiver"]),
                (
                    None
                    if row["held_class"] is None
                    else str(row["held_class"])
                ),
                int(row["K"]),
            )
        )
    receivers = sorted({key[0] for key in actual_keys})
    classes = sorted(
        {key[1] for key in actual_keys if key[1] is not None}
    )
    expected_keys = {
        (receiver, None, k_shot)
        for receiver in receivers
        for k_shot in (1, 5, 10)
    } | {
        (receiver, class_id, 1)
        for receiver in receivers
        for class_id in classes
    }
    if (
        len(receivers) != 7
        or len(classes) != 6
        or len(set(actual_keys)) != 63
        or set(actual_keys) != expected_keys
    ):
        raise ValueError("prediction identity matrix drift before truth open")

    prevalidated_rows: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    seen_artifact_paths: set[Path] = set()
    prediction_package_ids: set[str] = set()
    for row in rows:
        raw_path = str(row["path"])
        expected_sha = str(row["sha256"])
        artifact_path = (prediction_root / raw_path).resolve(strict=True)
        if (
            Path(raw_path).is_absolute()
            or not artifact_path.is_relative_to(prediction_root)
            or artifact_path in seen_artifact_paths
            or len(expected_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha)
            or artifact_path.stat().st_mtime_ns > prediction_manifest_mtime_ns
            or sha256_file(artifact_path) != expected_sha
        ):
            raise ValueError("prediction artifact was not sealed before truth open")
        artifact = _read_json(artifact_path)
        if (
            artifact.get("held_receiver") != row["held_receiver"]
            or artifact.get("held_class") != row["held_class"]
            or artifact.get("K") != row["K"]
            or str(row["package_id"])
            != package_id(str(row["held_receiver"]), int(row["K"]))
            ):
            raise ValueError("prediction package identity drift before truth open")
        seen_artifact_paths.add(artifact_path)
        prediction_package_ids.add(str(row["package_id"]))
        prevalidated_rows.append((row, artifact))
    if len(prediction_package_ids) != 21:
        raise ValueError("prediction package coverage drift before truth open")

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
    truth_packages = truth.get("packages")
    if (
        set(truth)
        != {"schema", "package_count", "packages", "predictor_access"}
        or truth.get("schema")
        != "cvs.d103_r2.rxid_crossreceiver.held_truth.v1"
        or truth.get("package_count") != 21
        or truth.get("predictor_access") is not False
        or not isinstance(truth_packages, list)
        or len(truth_packages) != 21
    ):
        raise ValueError("prediction/truth coverage drift")
    truth_by_package = {
        str(row["package_id"]): row for row in truth_packages
    }
    if (
        any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "package_id",
                "query_physical_ids",
                "query_truth_labels",
            }
            for row in truth_packages
        )
        or
        len(truth_by_package) != 21
        or set(truth_by_package) != prediction_package_ids
    ):
        raise ValueError("prediction/truth package identity drift")
    performance_rows = []
    agreement = []
    flips = []
    state_bytes = []
    query_mac = []
    for row, artifact in prevalidated_rows:
        truth_row = truth_by_package[str(row["package_id"])]
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
        "day_stability_rows": day_stability_rows,
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
