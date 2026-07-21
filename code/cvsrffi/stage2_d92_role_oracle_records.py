"""Normalize one paired D92 Role-Oracle row into auditable query records."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "cvs.phase2.d92.licensed_role_oracle.row_records.v1"


class D92RoleOracleRecordsError(ValueError):
    """Raised when paired row evidence cannot be normalized exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _registry_sha(old: tuple[str, ...], new: tuple[str, ...]) -> str:
    return hashlib.sha256(
        _canonical({"old": list(old), "new": list(new)})
    ).hexdigest()


def _readonly_snapshot(path: Path, *, name: str) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise D92RoleOracleRecordsError(f"{name} cannot be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o222:
            raise D92RoleOracleRecordsError(
                f"{name} must be a read-only regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()


def _prediction(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        expected = (
            "query_tokens",
            "scenarios",
            "predicted_class_handles",
        )
        if tuple(archive.files) != expected:
            raise D92RoleOracleRecordsError("prediction exact schema drift")
        result = {name: archive[name].astype(str) for name in expected}
    if len({len(value) for value in result.values()}) != 1:
        raise D92RoleOracleRecordsError("prediction row alignment drift")
    return result


def _shared(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        expected = (
            "query_tokens",
            "scenarios",
            "registered_class_handles",
            "model_state_sha256",
            "scores",
        )
        if tuple(archive.files) != expected:
            raise D92RoleOracleRecordsError("shared score exact schema drift")
        result = {
            "query_tokens": archive["query_tokens"].astype(str),
            "scenarios": archive["scenarios"].astype(str),
            "registered_class_handles": archive[
                "registered_class_handles"
            ].astype(str),
            "model_state_sha256": archive["model_state_sha256"].astype(str),
            "scores": np.asarray(archive["scores"], dtype=np.float32),
        }
    count = len(result["query_tokens"])
    if (
        result["scores"].shape
        != (count, len(result["registered_class_handles"]))
        or len(result["scenarios"]) != count
        or len(result["model_state_sha256"]) != count
    ):
        raise D92RoleOracleRecordsError("shared score row alignment drift")
    return result


def _truth(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "cvs.phase2.query_truth_sidecar.v2"
        or not isinstance(payload.get("rows"), list)
    ):
        raise D92RoleOracleRecordsError("truth sidecar schema drift")
    result: dict[str, dict[str, str]] = {}
    for row in payload["rows"]:
        if not isinstance(row, dict) or not {
            "query_token",
            "true_class_handle",
            "evaluation_role",
        }.issubset(row):
            raise D92RoleOracleRecordsError("truth sidecar row drift")
        token = str(row["query_token"])
        role = str(row["evaluation_role"])
        if token in result or role not in {"target_old", "target_new"}:
            raise D92RoleOracleRecordsError("truth token/role drift")
        result[token] = {
            "true_class": str(row["true_class_handle"]),
            "true_role": role,
        }
    return result


def _receipt(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise D92RoleOracleRecordsError("execution receipt drift")
    return value


def _verify_bundle(
    root: Path, *, kind: str
) -> dict[str, Any]:
    if kind == "baseline":
        receipt_name = "execution_receipt.json"
        primary_name = "prediction_artifact.npz"
        commit_schema = "cvs.phase2.diag_cosine_exploration_commit.v1"
        receipt_schema = "cvs.phase2.diag_cosine_exploration_receipt.v1"
        primary_key = "prediction_artifact_sha256"
    elif kind == "oracle":
        receipt_name = "execution_receipt.json"
        primary_name = "prediction_artifact.npz"
        commit_schema = "cvs.phase2.d92.licensed_role_oracle.commit.v1"
        receipt_schema = "cvs.phase2.d92.licensed_role_oracle.execution_receipt.v1"
        primary_key = "prediction_artifact_sha256"
    elif kind == "shared":
        receipt_name = "receipt.json"
        primary_name = "shared_score_matrix.npz"
        commit_schema = "cvs.phase2.d92.licensed_role_oracle.shared_score_commit.v1"
        receipt_schema = "cvs.phase2.d92.licensed_role_oracle.shared_score.v1"
        primary_key = "shared_score_matrix_sha256"
    else:
        raise D92RoleOracleRecordsError("unknown paired evidence bundle kind")
    receipt_raw, receipt_sha = _readonly_snapshot(
        root / receipt_name, name=f"{kind} receipt"
    )
    primary_raw, primary_sha = _readonly_snapshot(
        root / primary_name, name=f"{kind} primary artifact"
    )
    commit_raw, commit_sha = _readonly_snapshot(root / "COMMIT.json", name=f"{kind} COMMIT")
    try:
        receipt = json.loads(receipt_raw.decode("utf-8-sig"))
        commit = json.loads(commit_raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise D92RoleOracleRecordsError(f"{kind} evidence JSON drift") from exc
    members = {
        item.get("relative_path"): (item.get("sha256"), int(item.get("size_bytes", -1)))
        for item in commit.get("members", [])
        if isinstance(item, dict)
    }
    if (
        receipt.get("schema") != receipt_schema
        or commit.get("schema") != commit_schema
        or commit.get("execution_receipt_sha256", commit.get("receipt_sha256"))
        != receipt_sha
        or commit.get(primary_key) != primary_sha
        or members.get(receipt_name) != (receipt_sha, len(receipt_raw))
        or members.get(primary_name) != (primary_sha, len(primary_raw))
    ):
        raise D92RoleOracleRecordsError(f"{kind} COMMIT closure drift")
    return {
        "receipt": receipt,
        "primary_raw": primary_raw,
        "primary_sha256": primary_sha,
        "commit_sha256": commit_sha,
    }


def build_d92_role_oracle_row_records(
    *,
    paired_evaluation_root: str | Path,
    truth_sidecar_path: str | Path,
    output_path: str | Path,
    row_id: str,
    receiver: str,
    seed: int,
    k_shot: int,
    new_class_count: int,
) -> dict[str, Any]:
    """Join truth only after both prediction COMMIT files exist."""

    root = Path(paired_evaluation_root).resolve(strict=True)
    verified: dict[tuple[str, str], dict[str, Any]] = {}
    for state in ("before", "after"):
        verified[(state, "baseline")] = _verify_bundle(
            root / "baseline" / state, kind="baseline"
        )
        verified[(state, "oracle")] = _verify_bundle(
            root / "oracle" / state, kind="oracle"
        )
        verified[(state, "shared")] = _verify_bundle(
            root / "shared_scores" / state, kind="shared"
        )
    truth = _truth(Path(truth_sidecar_path))
    records: list[dict[str, Any]] = []
    old_classes: tuple[str, ...] | None = None
    for state in ("before", "after"):
        baseline = _prediction(verified[(state, "baseline")]["primary_raw"])
        oracle = _prediction(verified[(state, "oracle")]["primary_raw"])
        shared = _shared(verified[(state, "shared")]["primary_raw"])
        for field in ("query_tokens", "scenarios"):
            if not np.array_equal(baseline[field], oracle[field]) or not np.array_equal(
                baseline[field], shared[field]
            ):
                raise D92RoleOracleRecordsError(f"paired {state} {field} drift")
        classes = tuple(shared["registered_class_handles"].tolist())
        if state == "before":
            old_classes = classes
            new_classes: tuple[str, ...] = ()
        else:
            if old_classes is None or classes[: len(old_classes)] != old_classes:
                raise D92RoleOracleRecordsError("old registry prefix drift")
            new_classes = classes[len(old_classes) :]
            if len(new_classes) != int(new_class_count):
                raise D92RoleOracleRecordsError("new registry count drift")
        registry_sha = _registry_sha(old_classes, new_classes)
        baseline_receipt = verified[(state, "baseline")]["receipt"]
        query_payload_sha = str(baseline_receipt.get("apply_package_root_sha256", ""))
        if len(query_payload_sha) != 64:
            raise D92RoleOracleRecordsError("query payload receipt SHA missing")
        score_contract_sha = verified[(state, "shared")]["commit_sha256"]
        for index, token in enumerate(baseline["query_tokens"].tolist()):
            if token not in truth:
                raise D92RoleOracleRecordsError("query token is absent from truth")
            target = truth[token]
            if state == "before" and target["true_role"] != "target_old":
                raise D92RoleOracleRecordsError("before query role drift")
            score_vector_sha = hashlib.sha256(
                _canonical({"classes": list(classes)})
                + np.ascontiguousarray(shared["scores"][index]).tobytes(order="C")
            ).hexdigest()
            common = {
                "row_id": str(row_id),
                "receiver": str(receiver),
                "seed": int(seed),
                "k_shot": int(k_shot),
                "new_class_count": int(new_class_count),
                "scenario": str(baseline["scenarios"][index]),
                "state": state,
                "query_token": token,
                "true_class": target["true_class"],
                "true_role": target["true_role"],
                "score_contract_sha256": score_contract_sha,
                "model_state_sha256": str(shared["model_state_sha256"][index]),
                "score_vector_sha256": score_vector_sha,
                "query_payload_sha256": query_payload_sha,
                "registered_classes_sha256": registry_sha,
                "old_registered_classes": list(old_classes),
                "new_registered_classes": list(new_classes),
            }
            records.extend(
                [
                    {
                        **common,
                        "variant": "baseline",
                        "predicted_class": str(
                            baseline["predicted_class_handles"][index]
                        ),
                    },
                    {
                        **common,
                        "variant": "role_oracle",
                        "predicted_class": str(
                            oracle["predicted_class_handles"][index]
                        ),
                    },
                ]
            )
    if not records:
        raise D92RoleOracleRecordsError("paired row records are empty")
    destination = Path(output_path)
    raw = (
        "".join(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in records
        )
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short paired record write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(destination, stat.S_IREAD)
    return {
        "schema": SCHEMA,
        "status": "LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE",
        "record_count": len(records),
        "query_pair_count": len(records) // 2,
        "records_path": str(destination),
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "truth_join_started_after_both_prediction_commits": True,
    }


__all__ = [
    "D92RoleOracleRecordsError",
    "SCHEMA",
    "build_d92_role_oracle_row_records",
]
