#!/usr/bin/env python
"""Source-calibrated OOD geometry baseline for Phase1 feature packages.

This diagnostic evaluates cosine-prototype distance, diagonal Mahalanobis
distance, kNN distance, and optional logit energy without using target_unknown
rows for threshold fitting. It is a read-only baseline, not a deployable
Stage2-C success claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.gate_metrics import binary_reject_metrics  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _as_str_array(value: np.ndarray, n: int) -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [canonical_tx_id(arr.item())] * int(n)
    return [canonical_tx_id(v) for v in arr.reshape(-1).tolist()]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{path} does not contain features")
        features = np.asarray(data["features"], dtype=np.float32)
        n = int(features.shape[0])

        def pick(key: str, default: np.ndarray) -> np.ndarray:
            return np.asarray(data[key]) if key in data.files else default

        manifest: dict[str, Any] = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        payload: dict[str, Any] = {
            "features": features,
            "tx_logits": np.asarray(data["tx_logits"], dtype=np.float32) if "tx_logits" in data.files else None,
            "dataset_role": _as_str_array(pick("dataset_role", np.asarray([""] * n)), n),
            "tx_ids": _as_str_array(pick("tx_ids", np.asarray([""] * n)), n),
            "rx_ids": _as_str_array(pick("rx_ids", np.asarray([""] * n)), n),
            "day_ids": _as_str_array(pick("day_ids", np.asarray([""] * n)), n),
            "sat_scenarios": _as_str_array(pick("sat_scenarios", np.asarray([""] * n)), n),
            "channel_views": _as_str_array(pick("channel_views", np.asarray([""] * n)), n),
            "manifest": manifest,
        }
    return payload


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1.0e-8)


def _class_map(source_tx_ids: Sequence[str]) -> dict[str, int]:
    return {canonical_tx_id(tx): i for i, tx in enumerate(source_tx_ids)}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _build_source_geometry(
    z: np.ndarray,
    tx_ids: Sequence[str],
    roles: Sequence[str],
    *,
    source_tx_ids: Sequence[str],
    calibration_roles: set[str],
    var_floor: float,
) -> dict[str, Any]:
    tx_to_class = _class_map(source_tx_ids)
    cal_indices = [
        i
        for i, role in enumerate(roles)
        if role in calibration_roles and canonical_tx_id(tx_ids[i]) in tx_to_class
    ]
    if not cal_indices:
        raise ValueError("no source calibration rows are available")

    prototypes: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    source_by_class: dict[str, np.ndarray] = {}
    for tx in source_tx_ids:
        tx_c = canonical_tx_id(tx)
        idx = [i for i in cal_indices if canonical_tx_id(tx_ids[i]) == tx_c]
        if not idx:
            raise ValueError(f"no source calibration rows for tx={tx_c}")
        rows = z[np.asarray(idx, dtype=int)]
        proto = _normalize_rows(rows.mean(axis=0, keepdims=True))[0]
        var = np.var(rows - proto.reshape(1, -1), axis=0) + float(var_floor)
        prototypes[tx_c] = proto
        variances[tx_c] = var
        source_by_class[tx_c] = rows
    source_matrix = z[np.asarray(cal_indices, dtype=int)]
    return {
        "calibration_indices": cal_indices,
        "prototypes": prototypes,
        "variances": variances,
        "source_matrix": source_matrix,
        "source_by_class": source_by_class,
    }


def _score_rows(z: np.ndarray, geometry: dict[str, Any], *, knn_k: int) -> dict[str, np.ndarray | list[str]]:
    prototypes: dict[str, np.ndarray] = geometry["prototypes"]
    variances: dict[str, np.ndarray] = geometry["variances"]
    labels = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[label] for label in labels], axis=0)
    cosine_dist = 1.0 - np.clip(z @ proto_matrix.T, -1.0, 1.0)
    pred_pos = np.argmin(cosine_dist, axis=1)
    min_cosine_dist = cosine_dist[np.arange(z.shape[0]), pred_pos]
    mahal = np.zeros_like(cosine_dist)
    for j, label in enumerate(labels):
        delta = z - prototypes[label].reshape(1, -1)
        mahal[:, j] = np.sqrt(np.sum((delta * delta) / variances[label].reshape(1, -1), axis=1))
    min_mahal = mahal[np.arange(z.shape[0]), np.argmin(mahal, axis=1)]
    source_matrix = np.asarray(geometry["source_matrix"], dtype=np.float32)
    sims = np.clip(z @ source_matrix.T, -1.0, 1.0)
    dists = 1.0 - sims
    cal_indices = list(geometry.get("calibration_indices", []))
    for source_col, global_row in enumerate(cal_indices):
        if 0 <= int(global_row) < dists.shape[0] and source_col < dists.shape[1]:
            dists[int(global_row), source_col] = np.inf
    kth = min(max(int(knn_k), 1), dists.shape[1])
    knn_dist = np.partition(dists, kth - 1, axis=1)[:, kth - 1]
    return {
        "pred_tx_id": [labels[int(i)] for i in pred_pos],
        "cosine_dist": min_cosine_dist,
        "mahalanobis": min_mahal,
        "knn_dist": knn_dist,
    }


def _source_thresholds(
    scores: dict[str, np.ndarray | list[str]],
    payload: dict[str, Any],
    *,
    source_tx_ids: Sequence[str],
    calibration_roles: set[str],
    distance_quantile: float,
    energy_quantile: float,
) -> dict[str, Any]:
    source_set = {canonical_tx_id(tx) for tx in source_tx_ids}
    roles = payload["dataset_role"]
    tx_ids = payload["tx_ids"]
    pred_tx = list(scores["pred_tx_id"])
    mask = np.asarray(
        [
            role in calibration_roles and canonical_tx_id(tx_ids[i]) in source_set and pred_tx[i] == canonical_tx_id(tx_ids[i])
            for i, role in enumerate(roles)
        ],
        dtype=bool,
    )
    if not bool(mask.any()):
        raise ValueError("no correctly assigned source rows are available for threshold calibration")
    out: dict[str, Any] = {
        "cosine_dist_max": float(np.quantile(np.asarray(scores["cosine_dist"])[mask], float(distance_quantile))),
        "mahalanobis_max": float(np.quantile(np.asarray(scores["mahalanobis"])[mask], float(distance_quantile))),
        "knn_dist_max": float(np.quantile(np.asarray(scores["knn_dist"])[mask], float(distance_quantile))),
        "distance_quantile": float(distance_quantile),
        "source_correct_count": int(mask.sum()),
        "source_threshold_count": int(mask.size),
    }
    logits = payload.get("tx_logits")
    if logits is not None:
        energy = -torch.logsumexp(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
        out["energy_max"] = float(np.quantile(energy[mask], float(energy_quantile)))
        out["energy_quantile"] = float(energy_quantile)
    return out


def _validate_no_unknown_calibration(calibration_roles: set[str], unknown_roles: set[str]) -> None:
    overlap = sorted(calibration_roles & unknown_roles)
    if overlap:
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: unknown query roles cannot be used for OOD threshold calibration: "
            + ",".join(overlap)
        )
    forbidden = {"target_unknown", "unknown", "unknown_query"}
    if calibration_roles & forbidden:
        raise RuntimeError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: target_unknown/unknown calibration roles are forbidden"
        )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    z = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids must define Phase1 known class order")
    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    calibration_roles = _parse_roles(args.calibration_roles)
    _validate_no_unknown_calibration(calibration_roles, unknown_roles)

    geometry = _build_source_geometry(
        z,
        payload["tx_ids"],
        payload["dataset_role"],
        source_tx_ids=source_tx_ids,
        calibration_roles=calibration_roles,
        var_floor=float(args.var_floor),
    )
    scores = _score_rows(z, geometry, knn_k=int(args.knn_k))
    thresholds = _source_thresholds(
        scores,
        payload,
        source_tx_ids=source_tx_ids,
        calibration_roles=calibration_roles,
        distance_quantile=float(args.distance_quantile),
        energy_quantile=float(args.energy_quantile),
    )

    logits = payload.get("tx_logits")
    energy = None
    if logits is not None:
        energy = -torch.logsumexp(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    source_set = {canonical_tx_id(tx) for tx in source_tx_ids}
    explicit_unknown = {canonical_tx_id(tx) for tx in parse_tx_id_list(args.unknown_tx_ids)}
    rows: list[dict[str, Any]] = []
    known_total = known_closed_correct = known_accepted = known_correct_full = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    y_unknown: list[bool] = []
    accepted_flags: list[bool] = []
    reject_scores: list[float] = []
    for i, role in enumerate(payload["dataset_role"]):
        tx = canonical_tx_id(payload["tx_ids"][i])
        pred_tx = str(scores["pred_tx_id"][i])
        cosine_ok = float(np.asarray(scores["cosine_dist"])[i]) <= float(thresholds["cosine_dist_max"])
        mahal_ok = float(np.asarray(scores["mahalanobis"])[i]) <= float(thresholds["mahalanobis_max"])
        knn_ok = float(np.asarray(scores["knn_dist"])[i]) <= float(thresholds["knn_dist_max"])
        energy_ok = True
        if energy is not None and bool(args.use_energy_gate):
            energy_ok = float(energy[i]) <= float(thresholds["energy_max"])
        accepted = cosine_ok and mahal_ok and knn_ok and energy_ok
        is_known_query = role in known_roles and tx in source_set
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        closed_correct = bool(is_known_query and pred_tx == tx)
        if is_known_query:
            known_total += 1
            known_closed_correct += int(closed_correct)
            known_accepted += int(accepted)
            known_correct_full += int(accepted and closed_correct)
            known_correct_accepted += int(accepted and closed_correct)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            accepted_flags.append(bool(accepted))
            reject_scores.append(float(np.asarray(scores["mahalanobis"])[i]))
        rows.append(
            {
                "row": i,
                "role": role,
                "tx_id": tx,
                "rx_id": payload["rx_ids"][i],
                "day_id": payload["day_ids"][i],
                "channel_view": payload["channel_views"][i],
                "sat_scenario": payload["sat_scenarios"][i],
                "is_known_query": int(is_known_query),
                "is_unknown_query": int(is_unknown_query),
                "pred_tx_id": pred_tx,
                "accepted": int(accepted),
                "closed_correct_known": int(closed_correct),
                "accepted_correct_known": int(bool(accepted and closed_correct)),
                "cosine_dist": f"{float(np.asarray(scores['cosine_dist'])[i]):.8f}",
                "mahalanobis": f"{float(np.asarray(scores['mahalanobis'])[i]):.8f}",
                "knn_dist": f"{float(np.asarray(scores['knn_dist'])[i]):.8f}",
                "energy": "" if energy is None else f"{float(energy[i]):.8f}",
            }
        )

    metrics = {
        "phase": "phase1_source_calibrated_ood_geometry_baseline",
        "verdict_scope": "diagnostic_only_not_stage2c_success",
        "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(args.feature_npz),
        "source_tx_ids": source_tx_ids,
        "known_query_roles": sorted(known_roles),
        "unknown_query_roles": sorted(unknown_roles),
        "target_unknown_tx_ids": sorted(explicit_unknown),
        "calibration_roles": sorted(calibration_roles),
        "target_unknown_training_count": 0,
        "uses_unknown_query_for_threshold": False,
        "uses_target_receiver_for_ground_training": False,
        "gate_policy": {
            "cosine_prototype": True,
            "diag_mahalanobis": True,
            "knn_distance": True,
            "energy": bool(energy is not None and args.use_energy_gate),
        },
        "thresholds": thresholds,
        "known_query_count": int(known_total),
        "known_closed_accuracy_no_reject": _safe_rate(known_closed_correct, known_total),
        "known_full_accuracy_after_reject": _safe_rate(known_correct_full, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
        "known_coverage": _safe_rate(known_accepted, known_total),
        "unknown_query_count": int(unknown_total),
        "unknown_FAR": _safe_rate(unknown_accepted, unknown_total),
        "unknown_reject_rate": _safe_rate(unknown_total - unknown_accepted, unknown_total),
        "passes_unknown_far_target": bool(unknown_total > 0 and (unknown_accepted / max(1, unknown_total)) <= float(args.unknown_far_target)),
        "row_count": len(rows),
    }
    if y_unknown:
        extra = binary_reject_metrics(
            torch.as_tensor(y_unknown, dtype=torch.bool),
            torch.as_tensor(reject_scores, dtype=torch.float32),
            torch.as_tensor(accepted_flags, dtype=torch.bool),
        )
        metrics.update(extra)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.score_table_csv:
        Path(args.score_table_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.score_table_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["row"])
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", required=True)
    p.add_argument("--source_tx_ids", required=True)
    p.add_argument("--unknown_tx_ids", default="")
    p.add_argument("--known_query_roles", default="target_old,target_new")
    p.add_argument("--unknown_query_roles", default="target_unknown")
    p.add_argument("--calibration_roles", default="source")
    p.add_argument("--distance_quantile", type=float, default=0.95)
    p.add_argument("--energy_quantile", type=float, default=0.95)
    p.add_argument("--unknown_far_target", type=float, default=0.05)
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--var_floor", type=float, default=1.0e-4)
    p.add_argument("--use_energy_gate", action="store_true")
    p.add_argument("--output_json", default="")
    p.add_argument("--score_table_csv", default="")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
