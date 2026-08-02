#!/usr/bin/env python3
"""Run the frozen three-arm D112 source-held G1 and score separately.

``predict`` has no truth argument. ``score`` first verifies the complete,
immutable 63-row prediction set and its package/truth-seal chain, then opens
truth in a separate call. This is source-held evidence, not Target Phase2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import run_d106_g1_sourceheld_one_shot as d106  # noqa: E402
from scripts import run_d110_g1_sourceheld_one_shot as d110  # noqa: E402
from cvsrffi.stage2_d112_g0_source_bundle import (  # noqa: E402
    build_d112_g0_source_bundle,
)
from cvsrffi.stage2_d112_seam_bundle import (  # noqa: E402
    build_d112_source_held_g1_bundle,
)
from cvsrffi.stage2_d112_seam_qknn import (  # noqa: E402
    audit_d112_seam_state,
    fit_d112_ground_head_source_held_g1_state,
    fit_d112_seam_source_held_g1_state,
    predict_d112_seam_source_held_g1,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


ARMS = ("M0", "M_HEAD_GROUND", "M_JOINT_SEAM")
EFFECT_PAIRS = {
    "HEAD_GROUND_VS_M0": ("M_HEAD_GROUND", "M0"),
    "SEAM_MOTION_AT_HEAD": ("M_JOINT_SEAM", "M_HEAD_GROUND"),
    "JOINT_VS_M0": ("M_JOINT_SEAM", "M0"),
}
METRIC_NAMES = (
    "old_balanced_accuracy",
    "seen_new_accuracy",
    "H_old_new",
    "old_floor",
)
CANDIDATE_ID = "D112_SEAM_QKNN"
SPLIT_ID = d110.SPLIT_ID
PACKAGE_SCHEMA = d110.PACKAGE_SCHEMA
PREDICTION_SCHEMA = "cvs.d112.seam_qknn.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d112.seam_qknn.sourceheld.scores.v1"


class D112G1Error(ValueError):
    """Raised when the frozen D112 G1 identity or lifecycle drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return d106._file_sha(path)


def _read_json(path: Path) -> dict[str, Any]:
    return d106._read_json(path)


def _write_new(path: Path, value: Any) -> None:
    d106._write_new(path, value)


def _run_id(value: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 160:
        raise D112G1Error("run ID must be a short non-empty string")
    return value


def prepare(args: argparse.Namespace) -> int:
    """Reuse the already frozen D110 split/package builder without reselection."""

    # The package remains explicitly identified as the frozen D110-selected
    # data surface. D112 binds its SHA and truth seal later; it does not relabel
    # or rebuild the selected physical-ID set.
    return d110.prepare(args)


def _g1_bundle(args: argparse.Namespace, truth_input_seal_sha256: str):
    g0 = build_d112_g0_source_bundle(
        args.d106_tap_archive.resolve(strict=True),
        receipt_path=args.d106_tap_receipt.resolve(strict=True),
        checkpoint_sha256=args.checkpoint_sha256,
        expected_tap_sha256=args.d106_tap_archive_sha256,
    )
    return build_d112_source_held_g1_bundle(
        class_registry=g0.class_registry,
        g=g0.g,
        q0=g0.q0,
        U=g0.U,
        sigma0_r=g0.sigma0_r,
        sigma0_amb=g0.sigma0_amb,
        v_g_r=g0.v_g_r,
        v_g_amb=g0.v_g_amb,
        tau_h_r=g0.tau_h_r,
        checkpoint_sha256=str(g0.manifest["checkpoint_sha256"]),
        source_aggregate_sha256=str(g0.manifest["source_aggregate_sha256"]),
        phase1_seal_sha256=_file_sha(args.d106_tap_receipt.resolve(strict=True)),
        source_held_split_sha256=truth_input_seal_sha256,
        global_bundle_valid=bool(g0.manifest["global_bundle_valid"]),
        global_invalid_reason=str(g0.manifest["global_invalid_reason"]),
        g_quantization_l2_error_bound=g0.g_quantization_l2_error_bound,
        q0_quantization_l2_error_bound=g0.q0_quantization_l2_error_bound,
        U_operator_error_upper_bound=g0.U_operator_error_upper_bound,
        endpoint_quantization_chord_mse=g0.endpoint_quantization_chord_mse,
    )


def _baseline_predictions(bank: Any, query: np.ndarray, registry: Sequence[str]) -> list[str]:
    logits = score_zid_student_t_logits(
        bank,
        query,
        metric=identity_shared_psd_metric(config=bank.config),
    )
    return d106._argmax(logits, registry)


def predict(args: argparse.Namespace) -> int:
    """Commit all 63 x three truth-free predictions."""

    root = args.package_root.resolve(strict=True)
    output = args.output_dir.resolve()
    run_id = _run_id(args.run_id)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D112 G1 output exists: {output}")
    manifest_path = root / "package_manifest.json"
    manifest = _read_json(manifest_path)
    receivers = tuple(str(value) for value in manifest.get("receiver_ids", ()))
    classes = tuple(str(value) for value in manifest.get("class_ids", ()))
    packages = manifest.get("packages")
    truth_seal_sha = manifest.get("truth_input_seal_sha256")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("candidate_id") != d110.CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("query_truth_present") is not False
        or manifest.get("target_access") is not False
        or len(receivers) != 7
        or len(classes) != 6
        or not isinstance(packages, list)
        or len(packages) != 21
        or not isinstance(truth_seal_sha, str)
        or len(truth_seal_sha) != 64
    ):
        raise D112G1Error("D112 package manifest closure drift")
    by_key = {(str(row["held_receiver"]), int(row["K"])): row for row in packages}
    if set(by_key) != {(receiver, k) for receiver in receivers for k in d110.K_VALUES}:
        raise D112G1Error("D112 package matrix is not complete")
    bundle = _g1_bundle(args, str(truth_seal_sha))
    if tuple(bundle.class_registry) != classes:
        raise D112G1Error("D112 Phase1/package registry drift")

    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    cache: dict[tuple[str, int], tuple[np.ndarray, list[str], str, Any, Any, Any]] = {}
    rows: list[dict[str, Any]] = []
    for receiver, held_class, k_shot in d110.fixed_row_specs(receivers, classes):
        key = (receiver, k_shot)
        package_row = by_key[key]
        if key not in cache:
            support, labels, query, query_ids, package_sha = d110._load_package(
                root,
                package_row,
                classes=classes,
                k_shot=k_shot,
            )
            bank = build_typed_zid_support_bank(
                support,
                labels,
                classes,
                config=d106._lock(k_shot, package_sha),
            )
            head = fit_d112_ground_head_source_held_g1_state(bundle, bank)
            joint = fit_d112_seam_source_held_g1_state(bundle, bank)
            cache[key] = (query, query_ids, package_sha, bank, head, joint)
        query, query_ids, package_sha, bank, head, joint = cache[key]
        arm_predictions = {
            "M0": _baseline_predictions(bank, query, classes),
            "M_HEAD_GROUND": list(
                predict_d112_seam_source_held_g1(head, bank, query)
            ),
            "M_JOINT_SEAM": list(
                predict_d112_seam_source_held_g1(joint, bank, query)
            ),
        }
        if set(arm_predictions) != set(ARMS):
            raise D112G1Error("D112 three-arm prediction closure drift")
        audits = {
            "M_HEAD_GROUND": dict(audit_d112_seam_state(head)),
            "M_JOINT_SEAM": dict(audit_d112_seam_state(joint)),
        }
        if any(
            audit["query_rows_used_for_fit"] != 0
            or audit["query_state_updates"] != 0
            for audit in audits.values()
        ):
            raise D112G1Error("D112 query lifecycle drift")
        row = {
            "schema": PREDICTION_SCHEMA + ".row",
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "run_id": run_id,
            "held_receiver": receiver,
            "held_class": held_class,
            "K": k_shot,
            "package_id": str(package_row["package_id"]),
            "registered_classes": list(classes),
            "query_physical_ids": query_ids,
            "arm_predictions": arm_predictions,
            "shared_component_receipts": {
                "package_sha256": package_sha,
                "d106_tap_archive_sha256": args.d106_tap_archive_sha256,
                "d106_tap_receipt_sha256": _file_sha(
                    args.d106_tap_receipt.resolve(strict=True)
                ),
                "d112_bundle_content_root_sha256": bundle.manifest[
                    "content_root_sha256"
                ],
                "student_t_lock_sha256": bank.config.lock_digest,
                "arm_state_audits": audits,
            },
            "query_truth_access": False,
            "target_access": False,
            "formal_p2_authority": False,
            "query_state_updates": 0,
        }
        row["prediction_receipt_sha256"] = _sha(row)
        path = row_root / f"{_sha({'receiver': receiver, 'held_class': held_class, 'K': k_shot})}.json"
        _write_new(path, row)
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": str(package_row["package_id"]),
                "path": str(Path("rows") / path.name),
                "sha256": _file_sha(path),
                "prediction_receipt_sha256": row["prediction_receipt_sha256"],
            }
        )
    if len(rows) != 63 or len({row["prediction_receipt_sha256"] for row in rows}) != 63:
        raise D112G1Error("D112 prediction coverage did not close at 63 rows")
    result = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "run_id": run_id,
        "arms": list(ARMS),
        "row_count": 63,
        "arm_row_prediction_unit_count": 63 * len(ARMS),
        "rows": rows,
        "package_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": truth_seal_sha,
        "d112_bundle_content_root_sha256": bundle.manifest["content_root_sha256"],
        "query_truth_access": False,
        "target_access": False,
        "query_state_updates": 0,
        "sourceheld_non_target": True,
        "formal_p2_authority": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    result["prediction_set_receipt_sha256"] = _sha(result)
    _write_new(output / "prediction_manifest.json", result)
    print(output / "prediction_manifest.json")
    return 0


def _validate_truth_open_binding(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    prediction_root = args.prediction_root.resolve(strict=True)
    prediction_manifest_path = prediction_root / "prediction_manifest.json"
    prediction_manifest = _read_json(prediction_manifest_path)
    if (
        prediction_manifest.get("schema") != PREDICTION_SCHEMA
        or prediction_manifest.get("candidate_id") != CANDIDATE_ID
        or prediction_manifest.get("split_id") != SPLIT_ID
        or prediction_manifest.get("arms") != list(ARMS)
        or prediction_manifest.get("row_count") != 63
        or prediction_manifest.get("arm_row_prediction_unit_count") != 63 * len(ARMS)
        or prediction_manifest.get("query_truth_access") is not False
        or _sha(
            {
                key: value
                for key, value in prediction_manifest.items()
                if key != "prediction_set_receipt_sha256"
            }
        )
        != prediction_manifest.get("prediction_set_receipt_sha256")
    ):
        raise D112G1Error("D112 prediction manifest binding drift")
    truth_seal_path = args.truth_input_seal_json.resolve(strict=True)
    if _file_sha(truth_seal_path) != prediction_manifest.get("truth_input_seal_sha256"):
        raise D112G1Error("D112 truth-input seal SHA drift")
    package_manifest_path = (truth_seal_path.parent.parent / "package_manifest.json").resolve(
        strict=True
    )
    package_manifest = _read_json(package_manifest_path)
    packages = package_manifest.get("packages")
    truth_seal = _read_json(truth_seal_path)
    if (
        _file_sha(package_manifest_path)
        != prediction_manifest.get("package_manifest_sha256")
        or package_manifest.get("schema") != PACKAGE_SCHEMA
        or package_manifest.get("candidate_id") != d110.CANDIDATE_ID
        or package_manifest.get("split_id") != SPLIT_ID
        or package_manifest.get("query_truth_present") is not False
        or not isinstance(packages, list)
        or len(packages) != 21
        or package_manifest.get("truth_input_seal_sha256") != _file_sha(truth_seal_path)
        or truth_seal.get("split_id") != SPLIT_ID
        or truth_seal.get("package_count") != 21
        or truth_seal.get("predictor_truth_access") is not False
        or set(truth_seal.get("package_ids", ()))
        != {str(row.get("package_id")) for row in packages}
    ):
        raise D112G1Error("D112 package/truth-seal chain drift")
    return prediction_manifest_path, prediction_manifest


def score(args: argparse.Namespace) -> int:
    """Open truth only after validating the complete three-arm prediction seal."""

    manifest_path, manifest = _validate_truth_open_binding(args)
    output = args.output_json.resolve()
    event_path = args.truth_open_event_json.resolve()
    if output.exists() or event_path.exists():
        raise FileExistsError("immutable D112 G1 score/event output exists")
    root = args.prediction_root.resolve(strict=True)
    entries = manifest.get("rows")
    if not isinstance(entries, list) or len(entries) != 63:
        raise D112G1Error("D112 prediction row list drift")
    artifacts = []
    package_ids: set[str] = set()
    query_ids_by_package: dict[str, list[str]] = {}
    for entry in entries:
        relative = Path(str(entry["path"]))
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root) or _file_sha(path) != entry["sha256"]:
            raise D112G1Error("D112 prediction row seal drift")
        artifact = _read_json(path)
        if _sha(
            {
                key: value
                for key, value in artifact.items()
                if key != "prediction_receipt_sha256"
            }
        ) != artifact.get("prediction_receipt_sha256"):
            raise D112G1Error("D112 prediction row receipt drift")
        package_id = str(artifact.get("package_id"))
        query_ids = artifact.get("query_physical_ids")
        if (
            artifact.get("schema") != PREDICTION_SCHEMA + ".row"
            or artifact.get("candidate_id") != CANDIDATE_ID
            or artifact.get("query_truth_access") is not False
            or artifact.get("target_access") is not False
            or artifact.get("formal_p2_authority") is not False
            or artifact.get("query_state_updates") != 0
            or set(artifact.get("arm_predictions", {})) != set(ARMS)
            or not isinstance(query_ids, list)
            or not query_ids
            or any(len(artifact["arm_predictions"][arm]) != len(query_ids) for arm in ARMS)
            or any(
                entry.get(name) != artifact.get(name)
                for name in (
                    "held_receiver",
                    "held_class",
                    "K",
                    "package_id",
                    "prediction_receipt_sha256",
                )
            )
        ):
            raise D112G1Error("D112 prediction row lifecycle drift")
        prior = query_ids_by_package.setdefault(package_id, query_ids)
        if prior != query_ids:
            raise D112G1Error("D112 package query IDs drift across matched rows")
        package_ids.add(package_id)
        artifacts.append(artifact)
    receivers = tuple(sorted({str(row["held_receiver"]) for row in artifacts}))
    classes = tuple(str(value) for value in artifacts[0]["registered_classes"])
    actual_rows = {
        (str(row["held_receiver"]), row["held_class"], int(row["K"]))
        for row in artifacts
    }
    if (
        len(package_ids) != 21
        or len(receivers) != 7
        or len(classes) != 6
        or any(tuple(row["registered_classes"]) != classes for row in artifacts)
        or actual_rows != set(d110.fixed_row_specs(receivers, classes))
    ):
        raise D112G1Error("D112 fixed 63-row prediction coverage drift")

    seal_path = args.truth_input_seal_json.resolve(strict=True)
    event = {
        "schema": SCORE_SCHEMA + ".truth_open_event",
        "prediction_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "prediction_manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "truth_opened_after_all_predictions_committed": True,
        "opened_at_unix_ns": time.time_ns(),
    }
    if event["opened_at_unix_ns"] <= event["prediction_manifest_mtime_ns"]:
        raise D112G1Error("truth-open timestamp is not after prediction seal")
    _write_new(event_path, event)
    truth = _read_json(args.truth_json.resolve(strict=True))
    truth_packages = truth.get("packages")
    truth_seal = _read_json(seal_path)
    if (
        truth.get("split_id") != SPLIT_ID
        or truth.get("package_count") != 21
        or truth.get("predictor_access") is not False
        or not isinstance(truth_packages, list)
        or len(truth_packages) != 21
        or d106.canonical_sha256(truth_packages)
        != truth_seal.get("truth_package_root_sha256")
    ):
        raise D112G1Error("D112 independent truth closure drift")
    truth_by_package = {str(row["package_id"]): row for row in truth_packages}
    if set(truth_by_package) != package_ids:
        raise D112G1Error("D112 truth/prediction package identity drift")

    negative = {
        name: {metric: 0 for metric in METRIC_NAMES} for name in EFFECT_PAIRS
    }
    scored_rows = []
    for artifact in artifacts:
        matching = truth_by_package[str(artifact["package_id"])]
        if matching["query_physical_ids"] != artifact["query_physical_ids"]:
            raise D112G1Error("D112 truth/prediction physical-ID alignment drift")
        labels = np.asarray(matching["query_truth_labels"], dtype=str)
        metrics = {
            arm: d106._metric(
                labels,
                artifact["arm_predictions"][arm],
                artifact["registered_classes"],
                artifact["held_class"],
            )
            for arm in ARMS
        }
        effects: dict[str, dict[str, float | None]] = {}
        for name, (left, right) in EFFECT_PAIRS.items():
            effects[name] = {}
            for metric in METRIC_NAMES:
                left_value = metrics[left][metric]
                right_value = metrics[right][metric]
                delta = (
                    None
                    if left_value is None or right_value is None
                    else float(left_value) - float(right_value)
                )
                effects[name][metric] = delta
                if delta is not None and delta < 0.0:
                    negative[name][metric] += 1
        scored_rows.append(
            {
                "held_receiver": artifact["held_receiver"],
                "held_class": artifact["held_class"],
                "K": artifact["K"],
                "arm_metrics": metrics,
                "same_row_effects": effects,
                "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
            }
        )
    result = {
        "schema": SCORE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "arms": list(ARMS),
        "performance_rows": scored_rows,
        "negative_tail_row_counts": negative,
        "prediction_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "truth_sha256": _file_sha(args.truth_json.resolve(strict=True)),
        "truth_open_event_sha256": _file_sha(event_path),
        "prediction_artifact_committed_before_truth": True,
        "target_access": False,
    }
    result["score_set_receipt_sha256"] = _sha(result)
    _write_new(output, result)
    print(output)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preparer = commands.add_parser("prepare")
    preparer.add_argument("--source-val-archive", type=Path, required=True)
    preparer.add_argument("--source-val-manifest", type=Path, required=True)
    preparer.add_argument("--output-dir", type=Path, required=True)
    predictor = commands.add_parser("predict")
    predictor.add_argument("--package-root", type=Path, required=True)
    predictor.add_argument("--d106-tap-archive", type=Path, required=True)
    predictor.add_argument("--d106-tap-receipt", type=Path, required=True)
    predictor.add_argument("--d106-tap-archive-sha256", required=True)
    predictor.add_argument("--checkpoint-sha256", required=True)
    predictor.add_argument("--run-id", required=True)
    predictor.add_argument("--output-dir", type=Path, required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--prediction-root", type=Path, required=True)
    scorer.add_argument("--truth-json", type=Path, required=True)
    scorer.add_argument("--truth-input-seal-json", type=Path, required=True)
    scorer.add_argument("--truth-open-event-json", type=Path, required=True)
    scorer.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    return predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
