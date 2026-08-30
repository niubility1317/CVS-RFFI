"""Independent truth-last scoring for WISER-RF representation probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_PROBE_MEMBERS = {
    "P1_SOURCE_HEAD": "p1_predictions",
    "P2_SOURCE_PROTOTYPE": "p2_predictions",
    "P3_OLD_D92": "p3_predictions",
}


def _probe_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    classes = np.unique(truth)
    per_class = {
        str(int(class_id)): float(np.mean(prediction[truth == class_id] == class_id))
        for class_id in classes
    }
    values = np.asarray(list(per_class.values()), dtype=np.float64)
    return {
        "balanced_accuracy": float(values.mean()),
        "floor": float(values.min()),
        "accuracy": float(np.mean(prediction == truth)),
        "per_class_accuracy": per_class,
    }


def _geometry(features: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if features.ndim != 2 or features.shape[0] != truth.shape[0]:
        raise ValueError("query feature/truth geometry drift")
    classes = np.unique(truth)
    centers = np.stack([features[truth == class_id].mean(axis=0) for class_id in classes])
    center_by_row = np.stack(
        [centers[int(np.where(classes == class_id)[0][0])] for class_id in truth]
    )
    within = float(np.mean(np.sum((features - center_by_row) ** 2, axis=1)))
    global_center = centers.mean(axis=0)
    between = float(np.mean(np.sum((centers - global_center) ** 2, axis=1)))
    return {
        "within_trace": within,
        "between_trace": between,
        "between_within_ratio": between / max(within, 1.0e-12),
    }


def score_wiser_predictions(
    predictions_path: str | Path,
    receipt_path: str | Path,
    truth_path: str | Path,
) -> Mapping[str, Any]:
    """Join opaque tokens only after a complete truth-blind prediction receipt."""

    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8-sig"))
    if (
        receipt.get("status") != "PREDICTIONS_COMPLETE"
        or receipt.get("query_truth_opened") is not False
        or receipt.get("query_role_opened") is not False
        or receipt.get("support_state_frozen_before_query") is not True
    ):
        raise ValueError("WISER prediction is not truth-last eligible")
    truth_payload = json.loads(Path(truth_path).read_text(encoding="utf-8-sig"))
    if str(truth_payload.get("receiver")) != str(receipt.get("receiver")):
        raise ValueError("WISER truth receiver binding drift")
    rows = truth_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("WISER truth rows are missing")
    truth_lookup: dict[str, int] = {}
    for row in rows:
        token = str(row["query_token"])
        class_id = int(row["true_class_index"])
        if token in truth_lookup and truth_lookup[token] != class_id:
            raise ValueError("WISER duplicate truth token drift")
        truth_lookup[token] = class_id

    with np.load(Path(predictions_path), allow_pickle=False) as arrays:
        required = {"query_tokens", "query_z_id", *_PROBE_MEMBERS.values()}
        if not required.issubset(arrays.files):
            raise ValueError("WISER prediction members are incomplete")
        tokens = np.asarray(arrays["query_tokens"]).astype(str)
        expected_tokens = receipt.get("expected_query_tokens")
        if not isinstance(expected_tokens, list):
            raise ValueError("WISER frozen query-token registry is missing")
        expected = tuple(str(token) for token in expected_tokens)
        if (
            tokens.ndim != 1
            or len(tokens) != int(receipt.get("query_rows", -1))
            or tuple(tokens.tolist()) != expected
            or len(set(expected)) != len(expected)
            or not set(expected).issubset(truth_lookup)
        ):
            raise ValueError("WISER truth token join drift")
        truth = np.asarray([truth_lookup[token] for token in tokens], dtype=np.int64)
        old_mask = truth < 6
        if not bool(old_mask.all()) or set(truth.tolist()) != set(range(6)):
            raise ValueError("WISER representation probe needs complete six-old-class coverage")
        old_truth = truth[old_mask]
        probes = {}
        for probe_name, member in _PROBE_MEMBERS.items():
            prediction = np.asarray(arrays[member], dtype=np.int64)
            if prediction.shape != truth.shape:
                raise ValueError(f"WISER {probe_name} prediction geometry drift")
            probes[probe_name] = _probe_metrics(prediction[old_mask], old_truth)
        features = np.asarray(arrays["query_z_id"], dtype=np.float64)
        if features.shape[0] != len(tokens) or not np.isfinite(features).all():
            raise ValueError("WISER query feature closure drift")
        geometry = _geometry(features[old_mask], old_truth)

    return {
        "schema": "cvs.phase2.wiser_rf.truth_last_score.v1",
        "status": "ANALYZED",
        "arm": str(receipt["arm"]),
        "receiver": str(receipt["receiver"]),
        "scenario": str(receipt["scenario"]),
        "query_rows": int(len(tokens)),
        "old_query_rows": int(old_mask.sum()),
        "probes": probes,
        "geometry": geometry,
        "truth_join_after_prediction_only": True,
        "truth_handle_alignment_verified": True,
    }


__all__ = ["score_wiser_predictions"]
