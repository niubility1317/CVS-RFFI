#!/usr/bin/env python3
"""Run the immutable three-stage GRB-JP4-CFM Phase1-held54 falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    while _value in sys.path:
        sys.path.remove(_value)
for _value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, _value)

from cvsrffi import grb_jp4_cfm_phase1_held_falsifier as held  # noqa: E402
from cvsrffi.phase1_grb_jp4_cfm_bundle import (  # noqa: E402
    canonical_array_sha256,
)


TAP_SCHEMA = "cvs.phase1.jp4_tap_archive.v1"
COVERAGE_SCHEMA = "cvs.phase1.singleobs_dual_feature_coverage_receipt.v1"
TAP_MEMBERS = (
    "z_id",
    "hidden",
    "pre_relu",
    "joint_weight",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "class_ids",
    "observation_ids",
)
MANIFEST_FIELDS = {
    "schema",
    "status",
    "artifact_stage",
    "formal_phase2_eligible",
    "bundle_created",
    "target25_release_authorized",
    "exact_member_allowlist",
    "array_sha256",
    "artifact",
    "row_count",
    "inputs",
    "runtime_audit",
    "access_audit",
    "selection",
}


class Held54RunnerError(ValueError):
    """Raised when a release-stage artifact or execution contract drifts."""


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: str, name: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise Held54RunnerError(f"{name} must be a lowercase SHA256")
    return text


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise Held54RunnerError(f"{path} must contain one JSON object")
    return value


def _write_jsonl_new(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        for row in rows:
            data = json.dumps(
                row,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            handle.write(data + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    with output.open("xb") as handle:
        handle.write(data + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_phase1_inputs(
    *,
    archive_path: str | Path,
    manifest_path: str | Path,
    checkpoint_sha256: str,
    coverage_receipt_path: str | Path,
    coverage_receipt_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    archive_file = Path(archive_path)
    manifest_file = Path(manifest_path)
    coverage_file = Path(coverage_receipt_path)
    expected_checkpoint = _require_sha(checkpoint_sha256, "checkpoint_sha256")
    expected_coverage = _require_sha(
        coverage_receipt_sha256, "coverage_receipt_sha256"
    )
    if _sha_file(coverage_file) != expected_coverage:
        raise Held54RunnerError("coverage receipt SHA256 drift")
    coverage = _json(coverage_file)
    if (
        coverage.get("schema") != COVERAGE_SCHEMA
        or coverage.get("artifact_stage")
        != "phase1_offline_before_target_access"
        or coverage.get("target_access") is not False
        or coverage.get("query_access") is not False
        or coverage.get("held_fold_selected") is not False
        or coverage.get("pre_registered_coverage_gate_passed") is not True
    ):
        raise Held54RunnerError("coverage receipt legality drift")
    manifest = _json(manifest_file)
    if (
        set(manifest) != MANIFEST_FIELDS
        or manifest.get("schema") != TAP_SCHEMA
        or manifest.get("status") != "DEVELOPMENT_ONLY_NOT_FORMAL"
        or manifest.get("artifact_stage")
        != "phase1_offline_before_target_access"
        or manifest.get("formal_phase2_eligible") is not False
        or manifest.get("bundle_created") is not False
        or manifest.get("target25_release_authorized") is not False
        or manifest.get("exact_member_allowlist") != list(TAP_MEMBERS)
        or manifest.get("inputs", {}).get("checkpoint_sha256")
        != expected_checkpoint
        or manifest.get("access_audit")
        != {
            "source_validation_weak_iq_access": True,
            "clean_iq_access": False,
            "target_access": False,
            "query_access": False,
            "received_iq_persisted": False,
            "raw_iq_persisted": False,
        }
        or manifest.get("selection", {}).get(
            "selected_observations_per_physical_id"
        )
        != 1
    ):
        raise Held54RunnerError("tap archive manifest legality drift")
    archive_sha = _sha_file(archive_file)
    if (
        manifest.get("artifact")
        != {"path": archive_file.name, "sha256": archive_sha}
    ):
        raise Held54RunnerError("tap archive artifact binding drift")
    with np.load(archive_file, allow_pickle=False) as archive:
        if tuple(archive.files) != TAP_MEMBERS:
            raise Held54RunnerError("tap archive member allowlist/order drift")
        arrays = {name: np.asarray(archive[name]) for name in TAP_MEMBERS}
    if (
        manifest.get("row_count") != len(arrays["labels"])
        or set(manifest.get("array_sha256", {})) != set(TAP_MEMBERS)
        or any(
            canonical_array_sha256(arrays[name])
            != manifest["array_sha256"][name]
            for name in TAP_MEMBERS
        )
    ):
        raise Held54RunnerError("tap archive array receipt drift")
    binding = {
        "archive_schema": TAP_SCHEMA,
        "archive_sha256": archive_sha,
        "manifest_sha256": _sha_file(manifest_file),
        "checkpoint_sha256": expected_checkpoint,
        "coverage_sha256": expected_coverage,
    }
    return arrays, binding


def _load_prediction(path: str | Path) -> dict[str, Any]:
    prediction = _json(path)
    if (
        prediction.get("schema") != held.SCHEMA + ".prediction.v1"
        or prediction.get("candidate") != held.CANDIDATE
        or prediction.get("evaluation_scope") != held.SCOPE
        or prediction.get("target25_authorized") is not False
        or type(prediction.get("COMMIT")) is not str
        or len(prediction.get("rows", [])) != held.ROW_COUNT
    ):
        raise Held54RunnerError("prediction artifact outer contract drift")
    return prediction


def _load_score(path: str | Path) -> dict[str, Any]:
    score = _json(path)
    if (
        score.get("schema") != held.SCHEMA + ".score.v1"
        or score.get("candidate") != held.CANDIDATE
        or score.get("evaluation_scope") != held.SCOPE
        or score.get("target25_authorized") is not False
        or len(score.get("metrics", [])) != held.ROW_COUNT
    ):
        raise Held54RunnerError("score artifact outer contract drift")
    return score


def build_stage(args: argparse.Namespace) -> dict[str, Any]:
    arrays, binding = _load_phase1_inputs(
        archive_path=args.archive,
        manifest_path=args.manifest,
        checkpoint_sha256=args.checkpoint_sha256,
        coverage_receipt_path=args.coverage_receipt,
        coverage_receipt_sha256=args.coverage_receipt_sha256,
    )
    packet, query, truth = held.build_packet(
        arrays,
        coverage_sha256=args.coverage_receipt_sha256,
        artifact_binding=binding,
    )
    receipt = held.write_build_artifacts(args.output_dir, packet, query, truth)
    row_receipts = [
        {
            "stage": "BUILD",
            "status": "SUCCESS",
            "row_index": index,
            "row_id": row["row_id"],
            "pseudo_new": row["pseudo_new"],
            "scene": row["scene"],
            "K": row["K"],
            "support_rows": len(row["support_physical_ids"]),
            "query_rows": len(row["query_ids"]),
            "fit_variants": list(row["fit_states"]),
            "full_arm_state_bytes": row["resource"]["full_arm_state_bytes"],
            "target25_authorized": False,
        }
        for index, row in enumerate(packet["rows"])
    ]
    _write_jsonl_new(args.row_receipt, row_receipts)
    return {
        "stage": "BUILD",
        "status": "SUCCESS",
        "rows": len(row_receipts),
        "held_receiver": packet["held_receiver"],
        "packet_sha256": packet["packet_sha256"],
        "build_receipt_sha256": receipt["receipt_sha256"],
        "target25_authorized": False,
    }


def predict_stage(args: argparse.Namespace) -> dict[str, Any]:
    packet, query = held.load_prediction_inputs(args.build_dir)
    prediction = held.predict_packet(packet, query)
    file_sha = held.write_prediction_artifact(args.output, prediction)
    row_receipts = [
        {
            "stage": "PREDICT",
            "status": "SUCCESS",
            "row_index": index,
            "row_id": row["row_id"],
            "query_rows": len(row["query_ids"]),
            "arms_before": sorted(row["before"]),
            "arms_after": sorted(row["after"]),
            "counterfactuals": sorted(row["counterfactuals"]),
            "prediction_commit": prediction["COMMIT"],
            "target25_authorized": False,
        }
        for index, row in enumerate(prediction["rows"])
    ]
    _write_jsonl_new(args.row_receipt, row_receipts)
    return {
        "stage": "PREDICT",
        "status": "SUCCESS",
        "rows": len(row_receipts),
        "prediction_commit": prediction["COMMIT"],
        "prediction_file_sha256": file_sha,
        "truth_parsed": False,
        "target25_authorized": False,
    }


def score_stage(args: argparse.Namespace) -> dict[str, Any]:
    prediction = _load_prediction(args.prediction)
    packet, _query, truth = held.load_build_artifacts(args.build_dir)
    score = held.score_packet(
        packet,
        prediction,
        truth,
        commit=prediction["COMMIT"],
        truth_sha256=truth["truth_sha256"],
    )
    file_sha = held.write_score_artifact(args.output, score)
    row_receipts = [
        {
            "stage": "SCORE",
            "status": "SUCCESS",
            "row_index": index,
            "row_id": row["row_id"],
            "pseudo_new": row["pseudo_new"],
            "scene": row["scene"],
            "K": row["K"],
            "arms": row["arms"],
            "resource": row["resource"],
            "prediction_commit": score["COMMIT"],
            "target25_authorized": False,
        }
        for index, row in enumerate(score["metrics"])
    ]
    _write_jsonl_new(args.row_receipt, row_receipts)
    return {
        "stage": "SCORE",
        "status": "SUCCESS",
        "rows": len(row_receipts),
        "prediction_commit": score["COMMIT"],
        "score_file_sha256": file_sha,
        "held_proxy_gate_pass": score["held_proxy_gate_pass"],
        "verdict": score["verdict"],
        "target25_authorized": False,
    }


def audit_stage(args: argparse.Namespace) -> dict[str, Any]:
    arrays, binding = _load_phase1_inputs(
        archive_path=args.archive,
        manifest_path=args.manifest,
        checkpoint_sha256=args.checkpoint_sha256,
        coverage_receipt_path=args.coverage_receipt,
        coverage_receipt_sha256=args.coverage_receipt_sha256,
    )
    packet, query, truth = held.load_build_artifacts(args.build_dir)
    prediction = _load_prediction(args.prediction)
    score = _load_score(args.score)
    expected_score = held.score_packet(
        packet,
        prediction,
        truth,
        commit=prediction["COMMIT"],
        truth_sha256=truth["truth_sha256"],
    )
    if score != expected_score:
        raise Held54RunnerError("score artifact does not replay from prediction/truth")
    audit = held.audit_label_permutation(
        arrays,
        coverage_sha256=args.coverage_receipt_sha256,
        artifact_binding=binding,
        packet=packet,
        query=query,
        truth=truth,
        prediction=prediction,
        score=score,
    )
    if audit.get("gate_pass") is not True:
        raise Held54RunnerError("label permutation equivariance audit failed")
    _write_json_new(args.output, audit)
    return {
        "stage": "AUDIT_LABEL_PERMUTATION",
        "status": "SUCCESS",
        "rows_refit_and_compared": held.ROW_COUNT,
        "audit_file_sha256": _sha_file(args.output),
        "gate_pass": True,
        "target25_authorized": False,
    }


def _phase1_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--coverage-receipt", required=True)
    parser.add_argument("--coverage-receipt-sha256", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    build = subparsers.add_parser("build")
    _phase1_arguments(build)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--row-receipt", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--build-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--row-receipt", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--build-dir", required=True)
    score.add_argument("--prediction", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--row-receipt", required=True)
    audit = subparsers.add_parser("audit-labels")
    _phase1_arguments(audit)
    audit.add_argument("--build-dir", required=True)
    audit.add_argument("--prediction", required=True)
    audit.add_argument("--score", required=True)
    audit.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    functions = {
        "build": build_stage,
        "predict": predict_stage,
        "score": score_stage,
        "audit-labels": audit_stage,
    }
    result = functions[args.stage](args)
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
