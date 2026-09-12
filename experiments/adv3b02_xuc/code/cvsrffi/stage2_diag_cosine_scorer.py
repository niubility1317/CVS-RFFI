"""Post-prediction scorer for immutable diag-cosine Stage2-B/C artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


PREDICTION_MEMBERS = (
    "query_tokens",
    "scenarios",
    "predicted_class_handles",
)


class DiagCosineScorerError(ValueError):
    """Raised when prediction/truth evidence cannot be joined exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_prediction(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise DiagCosineScorerError("prediction artifact must be a regular file")
    if source.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise DiagCosineScorerError("prediction artifact must be read-only")
    with np.load(source, allow_pickle=False) as archive:
        if tuple(archive.files) != PREDICTION_MEMBERS:
            raise DiagCosineScorerError("prediction artifact exact schema drift")
        result = {name: archive[name].astype(str) for name in archive.files}
    lengths = {len(value) for value in result.values()}
    if lengths == {0} or len(lengths) != 1:
        raise DiagCosineScorerError("prediction artifact rows are empty or misaligned")
    keys = list(zip(result["scenarios"].tolist(), result["query_tokens"].tolist()))
    if len(keys) != len(set(keys)):
        raise DiagCosineScorerError("prediction scenario/token key is duplicated")
    if set(result["scenarios"].tolist()) != set(FORMAL_LEO_WEAK_SCENARIOS):
        raise DiagCosineScorerError("prediction scenario registry drift")
    return result


def _read_truth(path: str | Path) -> dict[str, dict[str, str]]:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise DiagCosineScorerError("truth sidecar must be a regular file")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "cvs.phase2.query_truth_sidecar.v2"
        or not isinstance(payload.get("rows"), list)
    ):
        raise DiagCosineScorerError("truth sidecar schema drift")
    result: dict[str, dict[str, str]] = {}
    for row in payload["rows"]:
        if (
            not isinstance(row, dict)
            or set(
                (
                    "query_token",
                    "true_class_handle",
                    "transmitter_label",
                    "evaluation_role",
                )
            )
            - set(row)
        ):
            raise DiagCosineScorerError("truth sidecar row schema drift")
        token = str(row["query_token"])
        role = str(row["evaluation_role"])
        if token in result or role not in {"target_old", "target_new"}:
            raise DiagCosineScorerError("truth token/role drift")
        result[token] = {
            "true_class_handle": str(row["true_class_handle"]),
            "transmitter_label": str(row["transmitter_label"]),
            "evaluation_role": role,
        }
    if not result:
        raise DiagCosineScorerError("truth sidecar is empty")
    return result


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    total = old_accuracy + new_accuracy
    return 0.0 if total <= 0.0 else 2.0 * old_accuracy * new_accuracy / total


def _score_state(
    prediction: Mapping[str, np.ndarray],
    truth: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    handle_roles: dict[str, str] = {}
    for row in truth.values():
        handle = row["true_class_handle"]
        role = row["evaluation_role"]
        previous = handle_roles.setdefault(handle, role)
        if previous != role:
            raise DiagCosineScorerError("registered class handle has mixed roles")
    rows: list[dict[str, Any]] = []
    for scenario, token, predicted in zip(
        prediction["scenarios"].tolist(),
        prediction["query_tokens"].tolist(),
        prediction["predicted_class_handles"].tolist(),
    ):
        if token not in truth:
            raise DiagCosineScorerError("prediction token is absent from truth sidecar")
        target = truth[token]
        rows.append(
            {
                "scenario": scenario,
                "token": token,
                "tx": target["transmitter_label"],
                "role": target["evaluation_role"],
                "true": target["true_class_handle"],
                "predicted": predicted,
                "predicted_role": handle_roles.get(predicted),
                "correct": int(predicted == target["true_class_handle"]),
            }
        )
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        old = [row for row in selected if row["role"] == "target_old"]
        new = [row for row in selected if row["role"] == "target_new"]
        old_acc = float(np.mean([row["correct"] for row in old])) if old else None
        new_acc = float(np.mean([row["correct"] for row in new])) if new else None
        by_scenario[scenario] = {
            "query_count": len(selected),
            "old_acc": old_acc,
            "seen_new_acc": new_acc,
            "h_old_new": (
                _harmonic(old_acc, new_acc)
                if old_acc is not None and new_acc is not None
                else None
            ),
            "old_to_new_rate": (
                float(
                    np.mean(
                        [
                            int(row["predicted_role"] == "target_new")
                            for row in old
                        ]
                    )
                )
                if old
                else None
            ),
            "new_to_old_rate": (
                float(
                    np.mean(
                        [
                            int(row["predicted_role"] == "target_old")
                            for row in new
                        ]
                    )
                )
                if new
                else None
            ),
        }
    by_tx: dict[str, dict[str, Any]] = {}
    for tx in sorted({row["tx"] for row in rows}):
        selected = [row for row in rows if row["tx"] == tx]
        by_tx[tx] = {
            "role": selected[0]["role"],
            "count": len(selected),
            "accuracy": float(np.mean([row["correct"] for row in selected])),
        }
    old = [row for row in rows if row["role"] == "target_old"]
    new = [row for row in rows if row["role"] == "target_new"]
    old_acc = float(np.mean([row["correct"] for row in old])) if old else None
    new_acc = float(np.mean([row["correct"] for row in new])) if new else None
    return {
        "query_count": len(rows),
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "h_old_new": (
            _harmonic(old_acc, new_acc)
            if old_acc is not None and new_acc is not None
            else None
        ),
        "by_scenario": by_scenario,
        "by_tx": by_tx,
    }


def score_diag_cosine_pair(
    *,
    before_prediction_path: str | Path,
    after_prediction_path: str | Path,
    truth_sidecar_path: str | Path,
    output_path: str | Path,
    candidate: str,
) -> dict[str, Any]:
    """Join truth only after both immutable prediction streams exist."""

    before_path = Path(before_prediction_path).resolve(strict=True)
    after_path = Path(after_prediction_path).resolve(strict=True)
    truth_path = Path(truth_sidecar_path).resolve(strict=True)
    before = _score_state(_read_prediction(before_path), _read_truth(truth_path))
    truth = _read_truth(truth_path)
    after = _score_state(_read_prediction(after_path), truth)
    if before["seen_new_acc"] is not None or after["seen_new_acc"] is None:
        raise DiagCosineScorerError("before/after registration role coverage drift")
    old_before = float(before["old_acc"])
    old_after = float(after["old_acc"])
    old_before_by_tx = {
        tx: row["accuracy"]
        for tx, row in before["by_tx"].items()
        if row["role"] == "target_old"
    }
    old_after_by_tx = {
        tx: row["accuracy"]
        for tx, row in after["by_tx"].items()
        if row["role"] == "target_old"
    }
    if set(old_before_by_tx) != set(old_after_by_tx):
        raise DiagCosineScorerError("matched old-class registry drift")
    payload = {
        "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
        "claim_scope": "development_only_not_formal_confirmation",
        "candidate": str(candidate),
        "query_truth_joined_only_after_immutable_predictions": True,
        "query_truth_fed_back_to_predictor": False,
        "before_prediction_sha256": _sha256_file(before_path),
        "after_prediction_sha256": _sha256_file(after_path),
        "truth_sidecar_sha256": _sha256_file(truth_path),
        "before": before,
        "after": after,
        "old_forgetting_pp": 100.0 * (old_before - old_after),
        "per_old_class_floor_before": min(old_before_by_tx.values()),
        "per_old_class_floor_after": min(old_after_by_tx.values()),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(destination, stat.S_IREAD)
    return {
        **payload,
        "score_artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }


__all__ = [
    "DiagCosineScorerError",
    "score_diag_cosine_pair",
]
