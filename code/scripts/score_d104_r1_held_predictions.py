#!/usr/bin/env python3
"""Independent truth-side scorer for sealed D104 252-unit predictions."""

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
    canonical_sha256,
    package_id,
    sha256_file,
)
from cvsrffi.stage2_d104_held_execution import (  # noqa: E402
    score_d104_prediction_artifact,
    validate_d104_prediction_artifact_without_truth,
)
from cvsrffi.stage2_d104_rxid_angq import ARMS  # noqa: E402
from cvsrffi.stage2_d104_source_split import SPLIT_ID  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--truth-json", type=Path, required=True)
    parser.add_argument("--truth-input-seal-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--truth-open-event-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root.resolve(strict=True)
    truth_path = args.truth_json.resolve(strict=True)
    truth_input_seal_path = args.truth_input_seal_json.resolve(strict=True)
    output = args.output_json.resolve()
    event_path = args.truth_open_event_json.resolve()
    if (
        output.exists()
        or output.is_symlink()
        or event_path.exists()
        or event_path.is_symlink()
    ):
        raise FileExistsError("immutable D104 scorer output already exists")
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest = _read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    manifest_mtime = manifest_path.stat().st_mtime_ns
    rows = manifest.get("rows")
    stability = manifest.get("day_stability_rows")
    truth_input_seal = _read_json(truth_input_seal_path)
    if (
        manifest.get("schema")
        != "cvs.d104_r1.rxid_angq.held_predictions.v1"
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("row_count") != 63
        or manifest.get("arm_row_prediction_unit_count") != 252
        or manifest.get("all_arm_prediction_receipts_unique") is not True
        or manifest.get("query_truth_access") is not False
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_state_updates") != 0
        or not isinstance(manifest.get("package_manifest_sha256"), str)
        or manifest.get("truth_input_seal_sha256")
        != sha256_file(truth_input_seal_path)
        or not isinstance(manifest.get("method_lock_sha256"), str)
        or not isinstance(manifest.get("registered_classes"), list)
        or canonical_sha256(manifest["registered_classes"])
        != manifest.get("registered_class_root_sha256")
        or not isinstance(rows, list)
        or len(rows) != 63
        or not isinstance(stability, list)
        or len(stability) != 49
    ):
        raise ValueError("D104 prediction manifest closure drift before truth open")
    actual_keys = []
    arm_receipts = []
    prevalidated: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    package_ids: set[str] = set()
    query_root_by_package: dict[str, str] = {}
    scorer_input_seals: list[str] = []
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
                "arm_prediction_receipts",
                "prediction_receipt_sha256",
                "scorer_input_seal_sha256",
                "method_lock_sha256",
                "registered_class_root_sha256",
                "support_physical_id_root_sha256",
                "query_physical_id_root_sha256",
                "int8_gate_pass",
            }
            or set(row["arm_prediction_receipts"]) != set(ARMS)
        ):
            raise ValueError("D104 prediction manifest row closure drift")
        key = (str(row["held_receiver"]), row["held_class"], int(row["K"]))
        actual_keys.append(key)
        raw_path = str(row["path"])
        artifact_path = (prediction_root / raw_path).resolve(strict=True)
        if (
            Path(raw_path).is_absolute()
            or not artifact_path.is_relative_to(prediction_root)
            or artifact_path in seen_paths
            or artifact_path.stat().st_mtime_ns > manifest_mtime
            or sha256_file(artifact_path) != row["sha256"]
        ):
            raise ValueError("D104 prediction artifact seal drift")
        artifact = _read_json(artifact_path)
        validated = validate_d104_prediction_artifact_without_truth(artifact)
        if (
            artifact.get("held_receiver") != row["held_receiver"]
            or artifact.get("held_class") != row["held_class"]
            or artifact.get("K") != row["K"]
            or artifact.get("arm_prediction_receipts")
            != row["arm_prediction_receipts"]
            or validated["prediction_receipt_sha256"]
            != row["prediction_receipt_sha256"]
            or validated["scorer_input_seal_sha256"]
            != row["scorer_input_seal_sha256"]
            or validated["method_lock_sha256"]
            != row["method_lock_sha256"]
            or validated["method_lock_sha256"]
            != manifest["method_lock_sha256"]
            or validated["registered_classes"]
            != manifest["registered_classes"]
            or validated["registered_class_root_sha256"]
            != row["registered_class_root_sha256"]
            or validated["registered_class_root_sha256"]
            != manifest["registered_class_root_sha256"]
            or validated["support_physical_id_root_sha256"]
            != row["support_physical_id_root_sha256"]
            or validated["query_physical_id_root_sha256"]
            != row["query_physical_id_root_sha256"]
            or package_id(str(row["held_receiver"]), int(row["K"]))
            != row["package_id"]
        ):
            raise ValueError("D104 prediction artifact identity drift")
        seen_paths.add(artifact_path)
        package_ids.add(str(row["package_id"]))
        prior_query_root = query_root_by_package.setdefault(
            str(row["package_id"]),
            str(row["query_physical_id_root_sha256"]),
        )
        if prior_query_root != row["query_physical_id_root_sha256"]:
            raise ValueError("D104 package query root drift across matched rows")
        arm_receipts.extend(row["arm_prediction_receipts"].values())
        scorer_input_seals.append(str(row["scorer_input_seal_sha256"]))
        prevalidated.append((row, artifact))
    receivers = sorted({key[0] for key in actual_keys})
    classes = sorted({key[1] for key in actual_keys if key[1] is not None})
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
        or len(package_ids) != 21
        or len(arm_receipts) != 252
        or len(set(arm_receipts)) != 252
        or canonical_sha256(scorer_input_seals)
        != manifest.get("scorer_input_root_sha256")
    ):
        raise ValueError("D104 63/252 identity matrix drift before truth open")
    if (
        set(truth_input_seal)
        != {
            "schema",
            "split_id",
            "package_count",
            "package_ids",
            "query_physical_id_roots",
            "truth_package_root_sha256",
            "source_val_scorer_manifest_sha256",
            "source_val_scorer_archive_sha256",
            "predictor_truth_access",
        }
        or truth_input_seal.get("schema")
        != "cvs.d104_r1.rxid_angq.truth_input_seal.v1"
        or truth_input_seal.get("split_id") != SPLIT_ID
        or truth_input_seal.get("package_count") != 21
        or truth_input_seal.get("predictor_truth_access") is not False
        or set(truth_input_seal.get("package_ids", ())) != package_ids
        or truth_input_seal.get("query_physical_id_roots")
        != query_root_by_package
    ):
        raise ValueError("D104 truth input seal drift before truth open")

    event_time = time.time_ns()
    if manifest_mtime >= event_time:
        raise ValueError("D104 manifest was not committed before truth open")
    event = {
        "schema": "cvs.d104_r1.rxid_angq.truth_first_open.v1",
        "split_id": SPLIT_ID,
        "prediction_manifest_sha256": manifest_sha,
        "package_manifest_sha256": manifest["package_manifest_sha256"],
        "truth_input_seal_sha256": sha256_file(truth_input_seal_path),
        "truth_package_root_sha256": truth_input_seal[
            "truth_package_root_sha256"
        ],
        "scorer_input_root_sha256": manifest["scorer_input_root_sha256"],
        "prediction_manifest_mtime_ns": manifest_mtime,
        "truth_first_open_unix_ns": event_time,
        "sealed_row_count": 63,
        "sealed_arm_row_prediction_unit_count": 252,
        "truth_opened_after_all_predictions_committed": True,
        "predictor_truth_access": False,
    }
    _write_json_new(event_path, event)
    truth = _read_json(truth_path)
    truth_packages = truth.get("packages")
    if (
        truth.get("schema") != "cvs.d104_r1.rxid_angq.held_truth.v2"
        or truth.get("split_id") != SPLIT_ID
        or truth.get("package_count") != 21
        or truth.get("predictor_access") is not False
        or not isinstance(truth_packages, list)
        or len(truth_packages) != 21
    ):
        raise ValueError("D104 truth package closure drift")
    truth_by_package = {
        str(row["package_id"]): row
        for row in truth_packages
        if isinstance(row, Mapping)
        and set(row)
        == {"package_id", "query_physical_ids", "query_truth_labels"}
    }
    if len(truth_by_package) != 21 or set(truth_by_package) != package_ids:
        raise ValueError("D104 truth/prediction package identity drift")
    if canonical_sha256(truth_packages) != truth_input_seal[
        "truth_package_root_sha256"
    ]:
        raise ValueError("D104 opened truth package root drift")

    performance_rows = []
    head_agreement = []
    joint_agreement = []
    head_flips = []
    joint_flips = []
    resource_bytes = []
    adaptation_mac = []
    query_mac_before = []
    query_mac_after = []
    query_mac_delta = []
    vector_ops = []
    peak_temporary = []
    for row, artifact in prevalidated:
        truth_row = truth_by_package[str(row["package_id"])]
        if artifact["query_physical_ids"] != truth_row["query_physical_ids"]:
            raise ValueError("D104 truth physical alignment drift")
        scored = score_d104_prediction_artifact(
            artifact,
            truth_row["query_truth_labels"],
        )
        scored["prediction_artifact_committed_before_truth"] = True
        scored["truth_open_event_sha256"] = sha256_file(event_path)
        scored["scorer_input_seal_sha256"] = artifact[
            "scorer_input_seal_sha256"
        ]
        scored["resource_receipts"] = artifact["resource_receipts"]
        scored["int8_audit"] = artifact["int8_audit"]
        scored.pop("score_receipt_sha256", None)
        scored["score_receipt_sha256"] = canonical_sha256(scored)
        performance_rows.append(scored)
        int8 = artifact["int8_audit"]
        head_agreement.append(float(int8["M_HEAD"]["top1_agreement"]))
        joint_agreement.append(float(int8["M_JOINT"]["top1_agreement"]))
        head_flips.append(
            int(int8["M_HEAD"]["teacher_winner_margin_flip_count"])
        )
        joint_flips.append(
            int(int8["M_JOINT"]["teacher_winner_margin_flip_count"])
        )
        for receipt in artifact["resource_receipts"].values():
            resource_bytes.append(int(receipt["actual_serialized_state_bytes_after"]))
            adaptation_mac.append(int(receipt["adaptation_mac_total"]))
            vector_ops.append(
                int(receipt["adaptation_vector_elementwise_ops_total"])
            )
            peak_temporary.append(
                int(receipt["peak_temporary_bytes_upper_bound"])
            )
            query_mac_before.append(int(receipt["query_mac_before"]))
            query_mac_after.append(int(receipt["query_mac_after"]))
            query_mac_delta.append(int(receipt["query_mac_delta"]))
    result = {
        "schema": "cvs.d104_r1.rxid_angq.held_scores.v1",
        "split_id": SPLIT_ID,
        "performance_rows": performance_rows,
        "day_stability_rows": stability,
        "quantization_receipt": {
            "M_HEAD_min_top1_agreement": min(head_agreement),
            "M_HEAD_margin_flip_count": sum(head_flips),
            "M_JOINT_min_top1_agreement": min(joint_agreement),
            "M_JOINT_margin_flip_count": sum(joint_flips),
            "row_count": 63,
            "persistent_fp_sidecar": False,
        },
        "resource_component": {
            "phase2_state_bytes_max": max(resource_bytes),
            "adaptation_mac_total_max": max(adaptation_mac),
            "adaptation_vector_elementwise_ops_total_max": max(vector_ops),
            "peak_temporary_bytes_max": max(peak_temporary),
            "query_mac_before_max": max(query_mac_before),
            "query_mac_after_max": max(query_mac_after),
            "query_mac_delta_max": max(query_mac_delta),
            "query_mac_delta_min": min(query_mac_delta),
            "query_mac_delta_sum": sum(query_mac_delta),
            "row_count": 63,
        },
        "evidence_binding": {
            "package_manifest_sha256": manifest["package_manifest_sha256"],
            "truth_input_seal_sha256": sha256_file(truth_input_seal_path),
            "truth_package_root_sha256": truth_input_seal[
                "truth_package_root_sha256"
            ],
            "source_val_scorer_manifest_sha256": truth_input_seal[
                "source_val_scorer_manifest_sha256"
            ],
            "source_val_scorer_archive_sha256": truth_input_seal[
                "source_val_scorer_archive_sha256"
            ],
            "method_lock_sha256": manifest["method_lock_sha256"],
            "registered_class_root_sha256": manifest[
                "registered_class_root_sha256"
            ],
            "scorer_input_root_sha256": manifest[
                "scorer_input_root_sha256"
            ],
        },
        "prediction_manifest_sha256": manifest_sha,
        "truth_sha256": sha256_file(truth_path),
        "truth_open_event_sha256": sha256_file(event_path),
        "prediction_artifact_committed_before_truth": True,
        "target_access": False,
    }
    result["score_set_receipt_sha256"] = canonical_sha256(result)
    _write_json_new(output, result)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
