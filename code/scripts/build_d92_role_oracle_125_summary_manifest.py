#!/usr/bin/env python3
"""Seal 125 row-level paired records into the licensed Oracle summary input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from scripts import summarize_d92_role_oracle_125 as summary
from cvsrffi.stage2_d92_role_oracle_query_evaluation import (
    CANDIDATE_D92_ROLE_ORACLE,
    LICENSE_STATUS,
)


SCHEMA = "cvs.phase2.d92.licensed_role_oracle.summary_manifest_builder.v1"


class D92RoleOracleManifestError(ValueError):
    """Raised when the fresh 125 row set is incomplete or mixed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, raw: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short summary manifest write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def build_manifest(
    matrix_manifest_path: str | Path,
    output_root: str | Path,
    reference_d92_root: str | Path,
    expected_ground_manifest_sha256: str,
) -> dict[str, Any]:
    matrix_path = Path(matrix_manifest_path).resolve(strict=True)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(matrix, dict)
        or matrix.get("candidate") != CANDIDATE_D92_ROLE_ORACLE
        or matrix.get("claim_scope") != LICENSE_STATUS
        or int(matrix.get("job_count", -1)) != 125
        or int(matrix.get("scenario_pair_count", -1)) != 375
        or not isinstance(matrix.get("jobs"), list)
        or len(matrix["jobs"]) != 125
    ):
        raise D92RoleOracleManifestError("licensed 125 matrix manifest drift")
    output = Path(output_root).resolve()
    reference_root = Path(reference_d92_root).resolve(strict=True)
    expected_ground_sha = str(expected_ground_manifest_sha256).lower()
    if (
        len(expected_ground_sha) != 64
        or any(value not in "0123456789abcdef" for value in expected_ground_sha)
        or matrix.get("ground_manifest_sha256") != expected_ground_sha
        or not matrix.get("ground_component_dir")
    ):
        raise D92RoleOracleManifestError("matrix ground authority drift")
    if output.exists():
        raise FileExistsError(output)
    chunks: list[bytes] = []
    seen: set[tuple[str, int, int, int]] = set()
    record_count = 0
    reference_prediction_hashes: list[dict[str, str]] = []
    reference_registration_pair_match_count = 0
    reference_state_authority_match_count = 0
    reference_prediction_match_count = 0
    reference_semantic_score_hashes: list[dict[str, str]] = []
    for job in matrix["jobs"]:
        key = (
            str(job["receiver"]),
            int(job["seed"]),
            int(job["k_shot"]),
            int(job["new_class_count"]),
        )
        if key in seen:
            raise D92RoleOracleManifestError("duplicate 125 row key")
        seen.add(key)
        job_root = Path(job["output_root"]).resolve(strict=True)
        receipt_path = job_root / "pipeline_receipt.json"
        records_path = job_root / "scorer" / "paired_query_records.jsonl"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        reference_receipt_path = (
            reference_root / "jobs" / str(job["job_id"]) / "pipeline_receipt.json"
        )
        reference_receipt = json.loads(
            reference_receipt_path.resolve(strict=True).read_text(
                encoding="utf-8-sig"
            )
        )
        fresh_score = json.loads(
            (job_root / "scorer" / "diag_cosine_score.json")
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
        reference_score = json.loads(
            (
                reference_root
                / "jobs"
                / str(job["job_id"])
                / "scorer"
                / "diag_cosine_score.json"
            )
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
        semantic_fields = (
            "before",
            "after",
            "old_forgetting_pp",
            "per_old_class_floor_before",
            "per_old_class_floor_after",
            "query_truth_fed_back_to_predictor",
            "query_truth_joined_only_after_immutable_predictions",
        )
        fresh_semantic = {field: fresh_score.get(field) for field in semantic_fields}
        reference_semantic = {
            field: reference_score.get(field) for field in semantic_fields
        }
        if fresh_semantic != reference_semantic:
            raise D92RoleOracleManifestError(
                "fresh/reference D92 semantic score drift"
            )
        reference_semantic_score_hashes.append(
            {
                "job_id": str(job["job_id"]),
                "semantic_score_sha256": hashlib.sha256(
                    _canonical(fresh_semantic)
                ).hexdigest(),
            }
        )
        if (
            receipt.get("candidate") != CANDIDATE_D92_ROLE_ORACLE
            or receipt.get("result_label") != LICENSE_STATUS
            or receipt.get("formal_protocol_valid") is not False
            or receipt.get("promotion_eligible") is not False
            or receipt.get("receiver") != key[0]
            or int(receipt.get("seed", -1)) != key[1]
            or int(receipt.get("k_shot", -1)) != key[2]
            or int(receipt.get("new_class_count", -1)) != key[3]
            or receipt.get("ground_manifest_sha256") != expected_ground_sha
            or receipt.get("ground_component_dir")
            != matrix["ground_component_dir"]
        ):
            raise D92RoleOracleManifestError("row pipeline receipt drift")
        version_fields = (
            "receiver",
            "seed",
            "k_shot",
            "new_class_count",
            "row_manifest_sha256",
            "authority_commit_sha256",
            "phase1_checkpoint_sha256",
            "sealed_feature_runtime_sha256",
            "method_lock_sha256",
        )
        if any(
            receipt.get(field) != reference_receipt.get(field)
            for field in version_fields
        ):
            raise D92RoleOracleManifestError(
                "fresh/reference D92 version or row authority drift"
            )
        if (
            reference_receipt.get("candidate")
            != "d92_registration_balanced_covariance"
        ):
            raise D92RoleOracleManifestError("reference candidate is not D92")
        if (
            receipt.get("registration_pair_final_sha256")
            == reference_receipt.get("registration_pair_final_sha256")
        ):
            reference_registration_pair_match_count += 1
        for state in ("before", "after"):
            for field in (
                "head_capsule_sha256",
                "enrollment_binding_sha256",
                "apply_package_root_sha256",
                "apply_package_seal_sha256",
            ):
                if receipt.get("states", {}).get(state, {}).get(field) == (
                    reference_receipt.get("states", {}).get(state, {}).get(field)
                ):
                    reference_state_authority_match_count += 1
        expected_records_sha = receipt.get("paired_query_records", {}).get(
            "records_sha256"
        )
        if expected_records_sha != _sha256(records_path):
            raise D92RoleOracleManifestError("row paired records SHA drift")
        raw = records_path.read_bytes()
        if not raw.endswith(b"\n"):
            raise D92RoleOracleManifestError("row record file is not canonical JSONL")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if not rows or any(
            (
                row.get("row_id") != job["job_id"]
                or row.get("receiver") != key[0]
                or int(row.get("seed", -1)) != key[1]
                or int(row.get("k_shot", -1)) != key[2]
                or int(row.get("new_class_count", -1)) != key[3]
            )
            for row in rows
        ):
            raise D92RoleOracleManifestError("row record metadata drift")
        chunks.append(raw)
        record_count += len(rows)
        for state in ("before", "after"):
            fresh_prediction = (
                job_root
                / "diag"
                / "baseline"
                / state
                / "prediction_artifact.npz"
            )
            reference_prediction = (
                reference_root
                / "jobs"
                / str(job["job_id"])
                / "diag"
                / state
                / "prediction_artifact.npz"
            )
            fresh_sha = _sha256(fresh_prediction)
            reference_sha = _sha256(reference_prediction.resolve(strict=True))
            if fresh_sha == reference_sha:
                reference_prediction_match_count += 1
            reference_prediction_hashes.append(
                {
                    "job_id": str(job["job_id"]),
                    "state": state,
                    "fresh_prediction_sha256": fresh_sha,
                    "reference_prediction_sha256": reference_sha,
                }
            )
    expected = {
        (receiver, seed, k_shot, new_count)
        for receiver in summary.EXPECTED_RECEIVERS
        for seed in summary.EXPECTED_SEEDS
        for k_shot, new_count in summary.EXPECTED_SLICES
    }
    if seen != expected:
        raise D92RoleOracleManifestError("completed rows do not equal locked 125")
    output.mkdir(parents=True)
    records_path = output / "paired_query_records.jsonl"
    records_sha = _write_new(records_path, b"".join(chunks))
    matrix_sha = _sha256(matrix_path)
    manifest = {
        "schema": summary.SCHEMA,
        "claim_scope": LICENSE_STATUS,
        "fresh_run": True,
        "fresh_pairing_id": hashlib.sha256(
            f"{matrix_sha}:{records_sha}".encode("ascii")
        ).hexdigest(),
        "run_id": Path(matrix["output_root"]).name,
        "candidate": CANDIDATE_D92_ROLE_ORACLE,
        "job_count": 125,
        "scenario_pair_count": 375,
        "variants": list(summary.VARIANTS),
        "old_class_count": 6,
        "receivers": list(summary.EXPECTED_RECEIVERS),
        "seeds": list(summary.EXPECTED_SEEDS),
        "slices": [
            {"k_shot": k_shot, "new_class_count": new_count}
            for k_shot, new_count in summary.EXPECTED_SLICES
        ],
        "scenarios": list(summary.EXPECTED_SCENARIOS),
        "records_format": "jsonl",
        "records_path": records_path.name,
        "records_sha256": records_sha,
        "record_count": record_count,
        "source_matrix_manifest_sha256": matrix_sha,
        "reference_d92_root": str(reference_root),
        "ground_manifest_sha256": expected_ground_sha,
        "reference_d92_prediction_root_sha256": hashlib.sha256(
            _canonical({"rows": reference_prediction_hashes})
        ).hexdigest(),
        "fresh_no_oracle_same_run_paired": True,
        "historical_reference_d92_audit_complete": True,
        "historical_reference_d92_semantically_equivalent": True,
        "historical_reference_semantic_score_match_count": 125,
        "historical_reference_semantic_score_total": 125,
        "historical_reference_semantic_score_root_sha256": hashlib.sha256(
            _canonical({"rows": reference_semantic_score_hashes})
        ).hexdigest(),
        "historical_reference_registration_pair_match_count": reference_registration_pair_match_count,
        "historical_reference_registration_pair_total": 125,
        "historical_reference_state_authority_match_count": reference_state_authority_match_count,
        "historical_reference_state_authority_total": 1000,
        "historical_reference_prediction_match_count": reference_prediction_match_count,
        "historical_reference_prediction_total": 250,
        "fresh_no_oracle_bit_exact_to_d92_retry2": reference_prediction_match_count
        == 250,
        "formal_protocol_valid": False,
        "promotion_eligible": False,
    }
    manifest_path = output / "summary_manifest.json"
    manifest_sha = _write_new(manifest_path, _canonical(manifest))
    return {
        "schema": SCHEMA,
        "status": LICENSE_STATUS,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "records_path": str(records_path),
        "records_sha256": records_sha,
        "record_count": record_count,
        "job_count": 125,
        "scenario_pair_count": 375,
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-d92-root", required=True)
    parser.add_argument("--expected-ground-manifest-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_manifest(
                args.matrix_manifest,
                args.output_root,
                args.reference_d92_root,
                args.expected_ground_manifest_sha256,
            ),
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
