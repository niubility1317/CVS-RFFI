#!/usr/bin/env python3
"""Audit and summarize the locked D1 SOMP-H 125-job stability tranche."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.leo_weak_cache import post_channel_iq_sha256  # noqa: E402
from cvsrffi.somph_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
)


MANIFEST_SCHEMA = "cvs.phase2.somph_diag_125_stability.v1"
PIPELINE_SCHEMA = "cvs.phase2.somph_diag_row_pipeline.v2"
SCORE_SCHEMA = "cvs.phase2.diag_cosine_dev_pair_score.v1"
SUMMARY_SCHEMA = "cvs.phase2.somph_diag_125_summary.v1"
GATES_SCHEMA = "cvs.phase2.somph_diag_125_gates.v1"
CANDIDATE = "d1_historical_diag_fftrf"
DIRECT_STATUS = "MISSING_NOT_RUN"
QUERY_PACKAGE_BINDING_STATUS = "PIPELINE_COMMIT_RECEIPT_QUERY_SHA_BOUND"
EXPECTED_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
EXPECTED_SEEDS = (713102, 713103, 713104, 713105, 713106)
EXPECTED_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
PREDICTION_MEMBERS = (
    "query_tokens",
    "scenarios",
    "predicted_class_handles",
)


class StabilitySummaryError(ValueError):
    """Raised when the completed tranche is incomplete or not internally bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _regular_readonly(path: Path, *, name: str) -> Path:
    source = path.absolute()
    mode = source.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise StabilitySummaryError(f"{name} must be a regular non-symlink file")
    resolved = source.resolve(strict=True)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise StabilitySummaryError(f"{name} must be immutable/read-only: {resolved}")
    return resolved


def _load_json(path: Path, *, name: str, readonly: bool = True) -> dict[str, Any]:
    source = _regular_readonly(path, name=name) if readonly else path.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise StabilitySummaryError(f"{name} root must be an object")
    return payload


def _load_readonly_json_snapshot(
    path: Path, *, name: str
) -> tuple[dict[str, Any], str]:
    """Parse and hash one immutable JSON snapshot from the same file descriptor."""

    raw, digest = _read_readonly_bytes_snapshot(path, name=name)
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StabilitySummaryError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise StabilitySummaryError(f"{name} root must be an object")
    return payload, digest


def _read_readonly_bytes_snapshot(
    path: Path, *, name: str
) -> tuple[bytes, str]:
    """Read and hash one immutable file from the same descriptor."""

    source = path.absolute()
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise StabilitySummaryError(f"{name} must be a regular non-symlink file")
    if before.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise StabilitySummaryError(f"{name} must be immutable/read-only")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise StabilitySummaryError(f"{name} identity changed before open")
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                raise StabilitySummaryError(f"{name} was truncated")
            raw += chunk
    finally:
        os.close(descriptor)
    return raw, hashlib.sha256(raw).hexdigest()


def _close(left: Any, right: Any, *, field: str, tolerance: float = 1.0e-12) -> None:
    if left is None or right is None:
        if left is not right:
            raise StabilitySummaryError(f"metric nullability drift: {field}")
        return
    if not math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0):
        raise StabilitySummaryError(f"metric binding drift: {field}")


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        raise StabilitySummaryError("cannot aggregate an empty metric")
    return float(statistics.fmean(rows))


def _harmonic(old: float, new: float) -> float:
    return 0.0 if old + new <= 0.0 else 2.0 * old * new / (old + new)


def _read_prediction(path: Path) -> tuple[dict[str, np.ndarray], str]:
    raw, digest = _read_readonly_bytes_snapshot(
        path, name="diag prediction"
    )
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if tuple(archive.files) != PREDICTION_MEMBERS:
            raise StabilitySummaryError("diag prediction exact schema drift")
        values = {name: archive[name].astype(str) for name in archive.files}
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1 or lengths == {0}:
        raise StabilitySummaryError("diag prediction row alignment drift")
    keys = list(zip(values["scenarios"].tolist(), values["query_tokens"].tolist()))
    if len(keys) != len(set(keys)):
        raise StabilitySummaryError("diag prediction scenario/token duplication")
    return values, digest


def _read_receipt_bound_npz(
    path: Path, *, expected_sha256: str
) -> dict[str, np.ndarray]:
    """Snapshot one writable staging NPZ only through an immutable receipt digest."""

    source = path.absolute()
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise StabilitySummaryError("receipt-bound query must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise StabilitySummaryError("receipt-bound query identity changed before open")
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            if not chunk:
                raise StabilitySummaryError("receipt-bound query was truncated")
            raw += chunk
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != str(expected_sha256):
        raise StabilitySummaryError("receipt-bound query SHA mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _read_truth(
    path: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], str]:
    payload, digest = _load_readonly_json_snapshot(
        path, name="truth sidecar"
    )
    if (
        payload.get("schema") != "cvs.phase2.query_truth_sidecar.v2"
        or not isinstance(payload.get("rows"), list)
    ):
        raise StabilitySummaryError("truth sidecar schema drift")
    by_token: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    for raw in payload["rows"]:
        if not isinstance(raw, dict):
            raise StabilitySummaryError("truth sidecar row drift")
        token = str(raw["query_token"])
        if token in by_token:
            raise StabilitySummaryError("truth sidecar token duplication")
        row = {
            "query_token": token,
            "true_class_handle": str(raw["true_class_handle"]),
            "transmitter_label": str(raw["transmitter_label"]),
            "evaluation_role": str(raw["evaluation_role"]),
            "physical_sample_id": str(raw["physical_sample_id"]),
        }
        if row["evaluation_role"] not in {"target_old", "target_new"}:
            raise StabilitySummaryError("truth sidecar role drift")
        by_token[token] = row
        rows.append(row)
    return by_token, rows, digest


def _scenario_details(
    prediction: Mapping[str, np.ndarray],
    truth: Mapping[str, Mapping[str, str]],
    *,
    state: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    joined: list[dict[str, Any]] = []
    for scenario, token, predicted in zip(
        prediction["scenarios"].tolist(),
        prediction["query_tokens"].tolist(),
        prediction["predicted_class_handles"].tolist(),
    ):
        target = truth.get(token)
        if target is None:
            raise StabilitySummaryError("prediction token absent from truth sidecar")
        joined.append(
            {
                "scenario": scenario,
                "tx": target["transmitter_label"],
                "role": target["evaluation_role"],
                "correct": int(predicted == target["true_class_handle"]),
            }
        )
    result: dict[str, dict[str, Any]] = {}
    per_tx: list[dict[str, Any]] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        selected = [row for row in joined if row["scenario"] == scenario]
        if not selected:
            raise StabilitySummaryError("prediction scenario coverage drift")
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in selected:
            grouped[(row["role"], row["tx"])].append(int(row["correct"]))
        old_values = [
            value
            for (role, _tx), values in grouped.items()
            if role == "target_old"
            for value in values
        ]
        new_values = [
            value
            for (role, _tx), values in grouped.items()
            if role == "target_new"
            for value in values
        ]
        old_tx_acc = [
            _mean(values)
            for (role, _tx), values in grouped.items()
            if role == "target_old"
        ]
        old_acc = _mean(old_values)
        new_acc = _mean(new_values) if new_values else None
        result[scenario] = {
            "old_acc": old_acc,
            "old_floor": min(old_tx_acc),
            "seen_new_acc": new_acc,
            "h_old_new": _harmonic(old_acc, new_acc) if new_acc is not None else None,
        }
        for (role, tx), values in sorted(grouped.items()):
            per_tx.append(
                {
                    "state": state,
                    "scenario": scenario,
                    "role": role,
                    "tx": tx,
                    "count": len(values),
                    "accuracy": _mean(values),
                }
            )
    return result, per_tx


def _verify_iq_hashes(iq: np.ndarray, stored: np.ndarray, *, context: str) -> None:
    hashes = np.asarray(stored).astype(str)
    if len(iq) != len(hashes):
        raise StabilitySummaryError(f"{context} IQ/hash alignment drift")
    for index, expected in enumerate(hashes.tolist()):
        if post_channel_iq_sha256(np.asarray(iq[index])) != expected:
            raise StabilitySummaryError(f"{context} post-channel IQ SHA mismatch")


def _support_rank0(
    path: Path, *, expected_class_count: int
) -> list[tuple[int, int, str, int]]:
    source = _regular_readonly(path, name="support package")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "support_leo_weak_iq",
            "support_class_indices",
            "support_rank_within_class",
            "support_tokens",
            "support_post_channel_iq_sha256",
            "support_satellite_seeds",
        }
        if required - set(archive.files):
            raise StabilitySummaryError("support package evidence members missing")
        iq = archive["support_leo_weak_iq"]
        labels = archive["support_class_indices"].astype(np.int64)
        ranks = archive["support_rank_within_class"].astype(np.int64)
        hashes = archive["support_post_channel_iq_sha256"].astype(str)
        seeds = archive["support_satellite_seeds"].astype(np.int64)
        tokens = archive["support_tokens"].astype(str)
    lengths = {len(iq), len(labels), len(ranks), len(hashes), len(seeds), len(tokens)}
    if lengths == {0} or len(lengths) != 1:
        raise StabilitySummaryError("support package full-array alignment drift")
    if len(tokens) != len(set(tokens.tolist())):
        raise StabilitySummaryError("support package token duplication")
    mask = ranks == 0
    rank0_labels = labels[mask]
    if (
        int(expected_class_count) < 1
        or len(rank0_labels) != int(expected_class_count)
        or set(rank0_labels.tolist()) != set(range(int(expected_class_count)))
        or any(
            int(np.sum(rank0_labels == class_index)) != 1
            for class_index in range(int(expected_class_count))
        )
    ):
        raise StabilitySummaryError("support package does not have one rank0 per class")
    _verify_iq_hashes(iq[mask], hashes[mask], context="rank0 support")
    return sorted(
        (
            int(label),
            int(rank),
            str(digest),
            int(seed),
        )
        for label, rank, digest, seed in zip(
            labels[mask], ranks[mask], hashes[mask], seeds[mask]
        )
    )


def _query_physical_iq(
    path: Path,
    truth: Mapping[str, Mapping[str, str]],
    *,
    expected_sha256: str,
) -> list[tuple[str, str, int]]:
    archive = _read_receipt_bound_npz(path, expected_sha256=expected_sha256)
    required = {
        "query_leo_weak_iq",
        "query_tokens",
        "query_post_channel_iq_sha256",
        "query_satellite_seeds",
    }
    if required - set(archive):
        raise StabilitySummaryError("query package evidence members missing")
    iq = archive["query_leo_weak_iq"]
    tokens = archive["query_tokens"].astype(str)
    hashes = archive["query_post_channel_iq_sha256"].astype(str)
    seeds = archive["query_satellite_seeds"].astype(np.int64)
    lengths = {len(iq), len(tokens), len(hashes), len(seeds)}
    if lengths == {0} or len(lengths) != 1:
        raise StabilitySummaryError("query package full-array alignment drift")
    if len(tokens) != len(set(tokens.tolist())):
        raise StabilitySummaryError("query package token duplication")
    if set(tokens.tolist()) != set(truth):
        raise StabilitySummaryError("query package token set does not equal truth sidecar")
    _verify_iq_hashes(iq, hashes, context="query")
    rows = []
    for token, digest, seed in zip(tokens.tolist(), hashes.tolist(), seeds.tolist()):
        target = truth.get(token)
        if target is None:
            raise StabilitySummaryError("query package token absent from truth")
        rows.append((target["physical_sample_id"], digest, int(seed)))
    return sorted(rows)


def _truth_physical_query_ids(
    truth: Mapping[str, Mapping[str, str]]
) -> list[str]:
    rows = [str(row["physical_sample_id"]) for row in truth.values()]
    if not rows or len(rows) != len(set(rows)):
        raise StabilitySummaryError("truth physical query ID set is empty or duplicated")
    return sorted(rows)


def _query_package_tokens(path: Path, *, expected_sha256: str) -> set[str]:
    archive = _read_receipt_bound_npz(path, expected_sha256=expected_sha256)
    required = {
        "query_leo_weak_iq",
        "query_tokens",
        "query_post_channel_iq_sha256",
        "query_satellite_seeds",
    }
    if required - set(archive):
        raise StabilitySummaryError("query package evidence members missing")
    iq = archive["query_leo_weak_iq"]
    tokens = archive["query_tokens"].astype(str)
    hashes = archive["query_post_channel_iq_sha256"].astype(str)
    seeds = archive["query_satellite_seeds"].astype(np.int64)
    lengths = {len(iq), len(tokens), len(hashes), len(seeds)}
    if lengths == {0} or len(lengths) != 1:
        raise StabilitySummaryError("query package full-array alignment drift")
    if len(tokens) != len(set(tokens.tolist())):
        raise StabilitySummaryError("query package token duplication")
    _verify_iq_hashes(iq, hashes, context="query")
    return set(tokens.tolist())


def _job_root(matrix_root: Path, job: Mapping[str, Any]) -> Path:
    return matrix_root / "jobs" / str(job["job_id"])


def _query_member_sha_from_receipt(
    *,
    root: Path,
    state: str,
    scenario: str,
    pipeline: Mapping[str, Any],
) -> str:
    diag_root = root / "diag" / state
    receipt_path = diag_root / "execution_receipt.json"
    commit_path = diag_root / "COMMIT.json"
    receipt, receipt_sha = _load_readonly_json_snapshot(
        receipt_path, name="diag execution receipt"
    )
    commit, commit_sha = _load_readonly_json_snapshot(
        commit_path, name="diag COMMIT"
    )
    if (
        pipeline["states"][state].get("diag_commit_sha256") != commit_sha
        or pipeline["states"][state].get("execution_receipt_sha256")
        != receipt_sha
        or commit.get("schema") != "cvs.phase2.diag_cosine_exploration_commit.v1"
        or commit.get("execution_receipt_sha256") != receipt_sha
        or not isinstance(commit.get("members"), list)
        or not any(
            item.get("relative_path") == "execution_receipt.json"
            and item.get("sha256") == receipt_sha
            for item in commit["members"]
            if isinstance(item, dict)
        )
        or commit.get("prediction_artifact_sha256")
        != pipeline["states"][state]["prediction_artifact_sha256"]
    ):
        raise StabilitySummaryError(f"execution receipt/COMMIT binding drift: {state}")
    apply = receipt.get("preopen_audit", {}).get("apply")
    expected_root = pipeline["states"][state]["apply_package_root_sha256"]
    if (
        not isinstance(apply, dict)
        or apply.get("schema") != "cvs.phase2.somph_preopen_audit.v1"
        or apply.get("profile") != "apply_only"
        or apply.get("status") != "STRUCTURAL_SELF_CONSISTENCY_PASS"
        or apply.get("hash_and_member_audit_same_file_descriptor") is not True
        or apply.get("iq_payload_materialized") is not True
        or apply.get("package_root_sha256") != expected_root
        or apply.get("artifact_member_allowlist_sha256") != expected_root
        or receipt.get("apply_package_root_sha256") != expected_root
        or receipt.get("apply_package_seal_sha256")
        != pipeline["states"][state]["apply_package_seal_sha256"]
        or not isinstance(apply.get("manifest_sha256"), str)
        or len(apply["manifest_sha256"]) != 64
        or set(apply.get("materialized_scenarios", []))
        != set(FORMAL_LEO_WEAK_SCENARIOS)
        or not isinstance(apply.get("opened_members"), list)
    ):
        raise StabilitySummaryError(f"immutable apply preopen binding drift: {state}")
    relative_path = f"query_{scenario}.npz"
    matches = [
        item
        for item in apply["opened_members"]
        if isinstance(item, dict) and item.get("relative_path") == relative_path
    ]
    if (
        len(matches) != 1
        or matches[0].get("status") != "PASS"
        or not isinstance(matches[0].get("sha256"), str)
        or len(matches[0]["sha256"]) != 64
        or int(matches[0].get("size_bytes", 0)) <= 0
    ):
        raise StabilitySummaryError(
            f"immutable query opened-member receipt drift: {state}/{scenario}"
        )
    return str(matches[0]["sha256"])


def _audit_job(
    matrix_root: Path,
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = _job_root(matrix_root, job)
    pipeline_path = root / "pipeline_receipt.json"
    score_path = root / "scorer" / "diag_cosine_score.json"
    truth_path = root / "offline" / "scorer" / "truth_sidecar.json"
    row_manifest_path = root / "offline" / "scorer" / "row_manifest.json"
    final_pair_path = root / "scorer" / "registration_pair.final.json"
    pipeline, pipeline_sha256 = _load_readonly_json_snapshot(
        pipeline_path, name="pipeline receipt"
    )
    score, score_sha256 = _load_readonly_json_snapshot(
        score_path, name="diag score"
    )
    row_manifest = _load_json(row_manifest_path, name="row manifest")
    final_pair, final_pair_sha256 = _load_readonly_json_snapshot(
        final_pair_path, name="registration pair"
    )
    expected = {
        "receiver": str(job["receiver"]),
        "seed": int(job["seed"]),
        "k_shot": int(job["k_shot"]),
        "new_class_count": int(job["new_class_count"]),
        "candidate": CANDIDATE,
    }
    if (
        pipeline.get("schema") != PIPELINE_SCHEMA
        or pipeline.get("status") != "DEVELOPMENT_ROW_COMPLETE"
        or pipeline.get("formal_launch_authority") is not False
        or any(pipeline.get(key) != value for key, value in expected.items())
    ):
        raise StabilitySummaryError(f"pipeline/job binding drift: {job['job_id']}")
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("candidate") != CANDIDATE
        or score.get("query_truth_joined_only_after_immutable_predictions") is not True
        or score.get("query_truth_fed_back_to_predictor") is not False
    ):
        raise StabilitySummaryError(f"score contract drift: {job['job_id']}")
    if score_sha256 != pipeline.get("score_artifact_sha256"):
        raise StabilitySummaryError(f"score SHA binding drift: {job['job_id']}")
    if (
        row_manifest.get("receiver") != expected["receiver"]
        or row_manifest.get("seed") != expected["seed"]
        or row_manifest.get("k_shot") != expected["k_shot"]
        or row_manifest.get("new_class_count") != expected["new_class_count"]
        or _canonical_sha256(row_manifest) != pipeline.get("row_manifest_sha256")
    ):
        raise StabilitySummaryError(f"row manifest binding drift: {job['job_id']}")
    if final_pair_sha256 != pipeline.get("registration_pair_final_sha256"):
        raise StabilitySummaryError(f"registration-pair SHA drift: {job['job_id']}")
    if (
        final_pair.get("old_support_physical_ids_sha256_before")
        != final_pair.get("old_support_physical_ids_sha256_after")
        or final_pair.get("old_query_physical_ids_sha256_before")
        != final_pair.get("old_query_physical_ids_sha256_after")
    ):
        raise StabilitySummaryError(f"within-row B/C physical split drift: {job['job_id']}")

    truth, _truth_rows, truth_sha256 = _read_truth(truth_path)
    state_details: dict[str, dict[str, dict[str, Any]]] = {}
    query_sha_by_state: dict[str, dict[str, str]] = {}
    query_binding_by_state: dict[str, dict[str, str]] = {}
    prediction_tokens_by_state: dict[str, dict[str, set[str]]] = {}
    per_tx_rows: list[dict[str, Any]] = []
    old_truth_tokens = {
        token
        for token, row in truth.items()
        if row["evaluation_role"] == "target_old"
    }
    all_truth_tokens = set(truth)
    for state in ("before", "after"):
        prediction_path = root / "diag" / state / "prediction_artifact.npz"
        prediction, prediction_sha256 = _read_prediction(prediction_path)
        if prediction_sha256 != pipeline["states"][state]["prediction_artifact_sha256"]:
            raise StabilitySummaryError(f"prediction SHA binding drift: {job['job_id']}:{state}")
        score_sha_field = f"{state}_prediction_sha256"
        if prediction_sha256 != score.get(score_sha_field):
            raise StabilitySummaryError(f"prediction/score binding drift: {job['job_id']}:{state}")
        details, state_tx = _scenario_details(prediction, truth, state=state)
        state_details[state] = details
        query_sha_by_state[state] = {}
        query_binding_by_state[state] = {}
        prediction_tokens_by_state[state] = {}
        per_tx_rows.extend(state_tx)
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            prediction_tokens = set(
                prediction["query_tokens"][
                    prediction["scenarios"] == scenario
                ].tolist()
            )
            expected_truth_tokens = (
                old_truth_tokens if state == "before" else all_truth_tokens
            )
            if prediction_tokens != expected_truth_tokens:
                raise StabilitySummaryError(
                    f"immutable prediction/truth exact coverage drift: "
                    f"{job['job_id']}:{state}:{scenario}"
                )
            prediction_tokens_by_state[state][scenario] = prediction_tokens
            expected_query_sha = _query_member_sha_from_receipt(
                root=root,
                state=state,
                scenario=scenario,
                pipeline=pipeline,
            )
            query_sha_by_state[state][scenario] = expected_query_sha
            query_binding_by_state[state][scenario] = (
                QUERY_PACKAGE_BINDING_STATUS
            )
            package_tokens = _query_package_tokens(
                root
                / "offline"
                / "predictor"
                / state
                / "apply_only_staging"
                / f"query_{scenario}.npz",
                expected_sha256=expected_query_sha,
            )
            if package_tokens != prediction_tokens:
                raise StabilitySummaryError(
                    f"pipeline-bound query/prediction exact coverage drift: "
                    f"{job['job_id']}:{state}:{scenario}"
                )
            scored = score[state]["by_scenario"][scenario]
            derived = details[scenario]
            _close(derived["old_acc"], scored["old_acc"], field=f"{job['job_id']}:{state}:{scenario}:old")
            _close(
                derived["seen_new_acc"],
                scored["seen_new_acc"],
                field=f"{job['job_id']}:{state}:{scenario}:new",
            )
            _close(
                derived["h_old_new"],
                scored["h_old_new"],
                field=f"{job['job_id']}:{state}:{scenario}:H",
            )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before_tokens = prediction_tokens_by_state["before"][scenario]
        after_tokens = prediction_tokens_by_state["after"][scenario]
        if before_tokens != (after_tokens & old_truth_tokens):
            raise StabilitySummaryError(
                f"before/after immutable old query token drift: "
                f"{job['job_id']}:{scenario}"
            )
    if truth_sha256 != score.get("truth_sidecar_sha256"):
        raise StabilitySummaryError(f"truth SHA binding drift: {job['job_id']}")

    row = {
        "job_id": job["job_id"],
        "receiver": expected["receiver"],
        "seed": expected["seed"],
        "k_shot": expected["k_shot"],
        "new_class_count": expected["new_class_count"],
        "candidate": CANDIDATE,
        "b_old_acc": float(score["before"]["old_acc"]),
        "c_old_acc": float(score["after"]["old_acc"]),
        "b_old_floor": float(score["per_old_class_floor_before"]),
        "c_old_floor": float(score["per_old_class_floor_after"]),
        "seen_new_acc": float(score["after"]["seen_new_acc"]),
        "h_old_new": float(score["after"]["h_old_new"]),
        "average_forgetting": float(score["old_forgetting_pp"]) / 100.0,
        "old_adaptation_gain": -float(score["old_forgetting_pp"]) / 100.0,
        "direct_adv3b02_status": DIRECT_STATUS,
        "direct_adv3b02_old_acc": None,
        "delta_vs_direct_ADV3B02_K1": None,
        "pipeline_receipt_sha256": pipeline_sha256,
        "score_sha256": score_sha256,
    }
    scenario_rows = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before = state_details["before"][scenario]
        after = state_details["after"][scenario]
        scenario_rows.append(
            {
                "job_id": job["job_id"],
                "receiver": expected["receiver"],
                "seed": expected["seed"],
                "k_shot": expected["k_shot"],
                "new_class_count": expected["new_class_count"],
                "scenario": scenario,
                "b_old_acc": before["old_acc"],
                "c_old_acc": after["old_acc"],
                "b_old_floor": before["old_floor"],
                "c_old_floor": after["old_floor"],
                "seen_new_acc": after["seen_new_acc"],
                "h_old_new": after["h_old_new"],
                "average_forgetting": before["old_acc"] - after["old_acc"],
                "old_adaptation_gain": after["old_acc"] - before["old_acc"],
                "direct_adv3b02_status": DIRECT_STATUS,
                "direct_adv3b02_old_acc": None,
                "delta_vs_direct_ADV3B02_K1": None,
            }
        )
    for tx_row in per_tx_rows:
        tx_row.update(
            {
                "job_id": job["job_id"],
                "receiver": expected["receiver"],
                "seed": expected["seed"],
                "k_shot": expected["k_shot"],
                "new_class_count": expected["new_class_count"],
            }
        )
    return row, scenario_rows, per_tx_rows, {
        "truth": truth,
        "root": root,
        "query_sha_by_state": query_sha_by_state,
        "query_binding_by_state": query_binding_by_state,
    }


def _group_mean(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    fields = (
        "b_old_acc",
        "c_old_acc",
        "b_old_floor",
        "c_old_floor",
        "seen_new_acc",
        "h_old_new",
        "average_forgetting",
        "old_adaptation_gain",
    )
    result = []
    for group_key, group_rows in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["row_count"] = len(group_rows)
        for field in fields:
            item[f"{field}_mean"] = _mean(row[field] for row in group_rows)
            item[f"{field}_min"] = min(float(row[field]) for row in group_rows)
        result.append(item)
    return result


def _pooled_old_floor(
    per_tx_rows: list[dict[str, Any]],
    *,
    k_shot: int,
    new_class_count: int,
    state: str,
) -> float:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in per_tx_rows:
        if (
            int(row["k_shot"]) == k_shot
            and int(row["new_class_count"]) == new_class_count
            and row["state"] == state
            and row["role"] == "target_old"
        ):
            grouped[str(row["tx"])].append((int(row["count"]), float(row["accuracy"])))
    if not grouped:
        raise StabilitySummaryError("pooled old-class floor has no rows")
    accuracies = []
    for rows in grouped.values():
        total = sum(count for count, _accuracy in rows)
        correct = sum(count * accuracy for count, accuracy in rows)
        accuracies.append(correct / total)
    return min(accuracies)


def _build_gates(
    scenario_rows: list[dict[str, Any]],
    per_tx_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (
            row["receiver"],
            int(row["seed"]),
            int(row["k_shot"]),
            int(row["new_class_count"]),
            row["scenario"],
        ): row
        for row in scenario_rows
    }
    k10 = {}
    thresholds = {5: 0.92, 10: 0.90, 20: 0.86}
    for new_count, new_threshold in thresholds.items():
        rows = [
            row
            for row in scenario_rows
            if int(row["k_shot"]) == 10
            and int(row["new_class_count"]) == new_count
        ]
        b_old = _mean(row["b_old_acc"] for row in rows)
        c_old = _mean(row["c_old_acc"] for row in rows)
        seen_new = _mean(row["seen_new_acc"] for row in rows)
        h_value = _mean(row["h_old_new"] for row in rows)
        floor_before = _pooled_old_floor(
            per_tx_rows, k_shot=10, new_class_count=new_count, state="before"
        )
        floor_after = _pooled_old_floor(
            per_tx_rows, k_shot=10, new_class_count=new_count, state="after"
        )
        checks = {
            "b_old_acc_ge_0p92": b_old >= 0.92,
            "c_old_acc_ge_0p92": c_old >= 0.92,
            "b_pooled_old_class_floor_ge_0p88": floor_before >= 0.88,
            "c_pooled_old_class_floor_ge_0p88": floor_after >= 0.88,
            f"seen_new_acc_ge_{new_threshold:.2f}": seen_new >= new_threshold,
        }
        k10[str(new_count)] = {
            "row_count": len(rows),
            "b_old_acc": b_old,
            "c_old_acc": c_old,
            "b_pooled_old_class_floor": floor_before,
            "c_pooled_old_class_floor": floor_after,
            "seen_new_acc": seen_new,
            "h_old_new": h_value,
            "checks": checks,
            "pass": all(checks.values()),
        }

    drops = []
    for receiver in EXPECTED_RECEIVERS:
        for seed in EXPECTED_SEEDS:
            for scenario in FORMAL_LEO_WEAK_SCENARIOS:
                k5 = by_key[(receiver, seed, 5, 20, scenario)]
                k10_row = by_key[(receiver, seed, 10, 20, scenario)]
                row = {
                    "receiver": receiver,
                    "seed": seed,
                    "scenario": scenario,
                }
                for field in ("c_old_acc", "c_old_floor", "seen_new_acc", "h_old_new"):
                    row[f"{field}_drop_pp"] = 100.0 * (
                        float(k10_row[field]) - float(k5[field])
                    )
                row["pass"] = all(
                    row[f"{field}_drop_pp"] <= 3.0 + 1.0e-12
                    for field in ("c_old_acc", "c_old_floor", "seen_new_acc", "h_old_new")
                )
                drops.append(row)

    k1_rows = [
        row
        for row in scenario_rows
        if int(row["k_shot"]) == 1 and int(row["new_class_count"]) == 20
    ]
    receiver_gain = {
        receiver: _mean(
            row["old_adaptation_gain"]
            for row in k1_rows
            if row["receiver"] == receiver
        )
        for receiver in EXPECTED_RECEIVERS
    }
    overall_gain = _mean(row["old_adaptation_gain"] for row in k1_rows)
    k1_checks = {
        "overall_old_adaptation_gain_ge_0": overall_gain >= 0.0,
        "every_receiver_old_adaptation_gain_ge_0": all(
            value >= 0.0 for value in receiver_gain.values()
        ),
    }
    k10_pass = all(item["pass"] for item in k10.values())
    k5_pass = all(row["pass"] for row in drops)
    k1_pass = all(k1_checks.values())
    performance_pass = k10_pass and k5_pass and k1_pass
    return {
        "schema": GATES_SCHEMA,
        "k10_absolute": k10,
        "k5_matched_new20": {
            "comparison_count": len(drops),
            "all_four_metrics_drop_le_3pp": k5_pass,
            "worst_drop_pp": {
                field: max(row[f"{field}_drop_pp"] for row in drops)
                for field in ("c_old_acc", "c_old_floor", "seen_new_acc", "h_old_new")
            },
            "comparisons": drops,
        },
        "k1_old_adaptation_gain": {
            "row_count": len(k1_rows),
            "overall_mean": overall_gain,
            "receiver_mean": receiver_gain,
            "checks": k1_checks,
            "pass": all(k1_checks.values()),
        },
        "direct_ADV3B02_K1": {
            "status": DIRECT_STATUS,
            "overall_delta_ge_0p02": None,
            "paired_95ci_lower_gt_0": None,
            "every_receiver_delta_ge_0": None,
            "pass": None,
        },
        "executed_performance_gates_pass": performance_pass,
        "executed_performance_status": (
            "PASS" if performance_pass else "FAIL"
        ),
        "overall_status": (
            "INCOMPLETE_DIRECT_ADV3B02_NOT_RUN_PERFORMANCE_PASS"
            if performance_pass
            else "INCOMPLETE_DIRECT_ADV3B02_NOT_RUN_PERFORMANCE_FAIL"
        ),
    }


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        raw = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write for summary JSON")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise StabilitySummaryError(f"refusing to write empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def summarize(matrix_manifest: str | Path, output_root: str | Path) -> dict[str, Any]:
    manifest_path = _regular_readonly(Path(matrix_manifest), name="matrix manifest")
    manifest, manifest_sha256 = _load_readonly_json_snapshot(
        manifest_path, name="matrix manifest"
    )
    matrix_root = manifest_path.parent
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"summary output already exists: {output}")
    jobs = manifest.get("jobs")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("candidate") != CANDIDATE
        or manifest.get("job_count") != 125
        or manifest.get("row_pair_count") != 125
        or manifest.get("scenario_pair_count") != 375
        or manifest.get("scenario_state_metric_count") != 750
        or manifest.get("receivers") != list(EXPECTED_RECEIVERS)
        or manifest.get("confirmation_seeds") != list(EXPECTED_SEEDS)
        or not isinstance(jobs, list)
        or len(jobs) != 125
    ):
        raise StabilitySummaryError("locked matrix manifest drift")
    observed_slices = {
        (int(item["k_shot"]), int(item["new_class_count"])) for item in jobs
    }
    if observed_slices != set(EXPECTED_SLICES):
        raise StabilitySummaryError("locked stability slice drift")
    expected_keys = {
        (receiver, seed, k_shot, new_count)
        for receiver in EXPECTED_RECEIVERS
        for seed in EXPECTED_SEEDS
        for k_shot, new_count in EXPECTED_SLICES
    }
    observed_keys = [
        (
            str(item["receiver"]),
            int(item["seed"]),
            int(item["k_shot"]),
            int(item["new_class_count"]),
        )
        for item in jobs
    ]
    job_ids = [str(item["job_id"]) for item in jobs]
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != expected_keys
        or len(job_ids) != len(set(job_ids))
    ):
        raise StabilitySummaryError("125-job Cartesian key/job_id uniqueness drift")

    row_metrics: list[dict[str, Any]] = []
    scenario_metrics: list[dict[str, Any]] = []
    per_tx_metrics: list[dict[str, Any]] = []
    evidence: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for job in jobs:
        row, scenarios, tx_rows, audit = _audit_job(matrix_root, job)
        row_metrics.append(row)
        scenario_metrics.extend(scenarios)
        per_tx_metrics.extend(tx_rows)
        evidence_key = (
            str(job["receiver"]),
            int(job["seed"]),
            int(job["k_shot"]),
            int(job["new_class_count"]),
        )
        if evidence_key in evidence:
            raise StabilitySummaryError("job evidence key overwrite attempt")
        evidence[evidence_key] = audit
    if len(row_metrics) != 125 or len(scenario_metrics) != 375:
        raise StabilitySummaryError("completed metric row count drift")

    nesting = []
    for receiver in EXPECTED_RECEIVERS:
        for seed in EXPECTED_SEEDS:
            k1 = evidence[(receiver, seed, 1, 20)]
            k10 = evidence[(receiver, seed, 10, 20)]
            k1_physical_ids = _truth_physical_query_ids(k1["truth"])
            k10_physical_ids = _truth_physical_query_ids(k10["truth"])
            if k1_physical_ids != k10_physical_ids:
                raise StabilitySummaryError(
                    f"K1/K10 readonly truth physical query drift: {receiver}/{seed}"
                )
            for scenario in FORMAL_LEO_WEAK_SCENARIOS:
                support_leaf = f"support_{scenario}.npz"
                k1_support = _support_rank0(
                    k1["root"]
                    / "offline"
                    / "predictor"
                    / "after"
                    / "enrollment_only"
                    / support_leaf,
                    expected_class_count=6 + 20,
                )
                k10_support = _support_rank0(
                    k10["root"]
                    / "offline"
                    / "predictor"
                    / "after"
                    / "enrollment_only"
                    / support_leaf,
                    expected_class_count=6 + 20,
                )
                query_leaf = f"query_{scenario}.npz"
                k1_query = _query_physical_iq(
                    k1["root"]
                    / "offline"
                    / "predictor"
                    / "after"
                    / "apply_only_staging"
                    / query_leaf,
                    k1["truth"],
                    expected_sha256=k1["query_sha_by_state"]["after"][scenario],
                )
                k10_query = _query_physical_iq(
                    k10["root"]
                    / "offline"
                    / "predictor"
                    / "after"
                    / "apply_only_staging"
                    / query_leaf,
                    k10["truth"],
                    expected_sha256=k10["query_sha_by_state"]["after"][scenario],
                )
                if k1_support != k10_support:
                    raise StabilitySummaryError(
                        f"K1/K10 rank0 support SHA drift: {receiver}/{seed}/{scenario}"
                    )
                if k1_query != k10_query:
                    raise StabilitySummaryError(
                        f"K1/K10 postrun query IQ SHA consistency drift: "
                        f"{receiver}/{seed}/{scenario}"
                    )
                nesting.append(
                    {
                        "receiver": receiver,
                        "seed": seed,
                        "scenario": scenario,
                        "rank0_support_post_channel_iq_sha_match": True,
                        "readonly_truth_physical_query_ids_match": True,
                        "pipeline_bound_post_channel_iq_sha_match": True,
                        "query_package_binding_status": (
                            QUERY_PACKAGE_BINDING_STATUS
                        ),
                        "rank0_support_count": len(k1_support),
                        "readonly_truth_query_count": len(k1_physical_ids),
                        "receipt_bound_staging_query_count": len(k1_query),
                    }
                )

    receiver_metrics = _group_mean(
        scenario_metrics, ("receiver", "k_shot", "new_class_count")
    )
    slice_metrics = _group_mean(
        scenario_metrics, ("k_shot", "new_class_count", "scenario")
    )
    gates = _build_gates(scenario_metrics, per_tx_metrics)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": (
            "INCOMPLETE_DIRECT_BASELINE_PERFORMANCE_PASS"
            if gates["executed_performance_gates_pass"]
            else "INCOMPLETE_DIRECT_BASELINE_PERFORMANCE_FAIL"
        ),
        "claim_scope": manifest.get("claim_scope"),
        "formal_launch_authority": False,
        "matrix_manifest": str(manifest_path),
        "matrix_manifest_sha256": manifest_sha256,
        "candidate": CANDIDATE,
        "job_count": len(row_metrics),
        "scenario_pair_count": len(scenario_metrics),
        "scenario_state_metric_count": 2 * len(scenario_metrics),
        "per_tx_metric_count": len(per_tx_metrics),
        "receiver_metric_count": len(receiver_metrics),
        "support_query_nesting_audit_count": len(nesting),
        "support_query_nesting_all_strict_bindings_pass": all(
            row["rank0_support_post_channel_iq_sha_match"]
            and row["readonly_truth_physical_query_ids_match"]
            and row["pipeline_bound_post_channel_iq_sha_match"]
            for row in nesting
        ),
        "direct_adv3b02_status": DIRECT_STATUS,
        "query_package_commit_binding_status": QUERY_PACKAGE_BINDING_STATUS,
        "result_boundary": (
            "D1 independent stability summary with strict pipeline-to-COMMIT-"
            "to-execution-receipt-to-opened-query-member binding. Direct "
            "ADV3B02 K1 delta and paired confidence interval were not run."
        ),
        "slice_metrics": slice_metrics,
        "support_query_nesting_audit": nesting,
        "gates": gates,
    }

    output.mkdir(parents=True, exist_ok=False)
    _write_csv_new(output / "row_metrics.csv", row_metrics)
    _write_csv_new(output / "scenario_metrics.csv", scenario_metrics)
    _write_csv_new(output / "receiver_metrics.csv", receiver_metrics)
    _write_csv_new(output / "per_tx_metrics.csv", per_tx_metrics)
    _write_json_new(output / "gates.json", gates)
    _write_json_new(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = summarize(args.matrix_manifest, args.output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
