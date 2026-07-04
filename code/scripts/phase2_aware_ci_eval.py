#!/usr/bin/env python
"""AWARE-CI adaptive old-safe collaborative inference for Stage2-C features.

AWARE-CI is a deployment-oriented diagnostic for satellite-swarm RFFI. It keeps
the ADV3B02 feature extractor frozen, builds class-conditional old/seen-new
prototypes from source old rows plus target K-shot support, uses source-side
proxy_unknown only for open-set calibration, and evaluates target_unknown rows
only after all thresholds are fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluation.collaborative_open_set_qknn_eval import parse_collab_counts  # noqa: E402
from phase2_collaborative_open_set_qknn_eval import _normalize_rows, load_feature_npz  # noqa: E402
from phase2_proxy_adapter_ci_eval import AdapterTrainingPlan, build_training_plan  # noqa: E402

UNKNOWN_LABEL = "__unknown__"


@dataclass(frozen=True)
class AwareProfile:
    name: str
    description: str
    known_quantile: float
    reject_scale: float
    accept_scale: float
    min_label_votes: int
    min_label_vote_fraction: float
    reject_vote_fraction: float
    min_reliability: float


PROFILES: tuple[AwareProfile, ...] = (
    AwareProfile(
        name="aware_old_safe",
        description="old retention first; reject only with strong multi-risk evidence",
        known_quantile=0.995,
        reject_scale=1.25,
        accept_scale=1.00,
        min_label_votes=1,
        min_label_vote_fraction=0.34,
        reject_vote_fraction=0.75,
        min_reliability=0.20,
    ),
    AwareProfile(
        name="aware_balanced",
        description="balanced old-safe and unknown-risk fusion",
        known_quantile=0.990,
        reject_scale=1.10,
        accept_scale=0.95,
        min_label_votes=2,
        min_label_vote_fraction=0.40,
        reject_vote_fraction=0.60,
        min_reliability=0.25,
    ),
    AwareProfile(
        name="aware_unknown_probe",
        description="diagnostic probe with stronger unknown rejection pressure",
        known_quantile=0.970,
        reject_scale=0.95,
        accept_scale=0.90,
        min_label_votes=2,
        min_label_vote_fraction=0.46,
        reject_vote_fraction=0.50,
        min_reliability=0.30,
    ),
)


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    missing = sorted(set(names) - known)
    if missing:
        raise argparse.ArgumentTypeError(f"unknown AWARE-CI profile(s): {', '.join(missing)}")
    return names


def _stable_order(payload: Mapping[str, Any], indices: Sequence[int], seed: int) -> list[int]:
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    day_ids = np.asarray(payload["day_ids"]).astype(str)
    sig_ids = np.asarray(payload["sig_ids"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)

    def key(i: int) -> tuple[str, str, str, str, int]:
        return (str(rx_ids[i]), str(tx_ids[i]), str(day_ids[i]), str(sig_ids[i]), (int(i) + int(seed)) % 1000003)

    return sorted([int(i) for i in indices], key=key)


def _query_indices(payload: Mapping[str, Any], plan: AdapterTrainingPlan, query_per_class: int, seed: int) -> list[int]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    support = set(int(i) for i in plan.support_indices)
    selected: list[int] = []
    for role in ["target_old", "target_new", "target_unknown"]:
        role_mask = roles == role
        for rx in sorted({str(v) for v in rx_ids[role_mask].tolist()}):
            rx_mask = role_mask & (rx_ids == rx)
            for tx in sorted({str(v) for v in tx_ids[rx_mask].tolist()}):
                idx = [int(i) for i in np.where(rx_mask & (tx_ids == tx))[0].tolist() if int(i) not in support]
                selected.extend(_stable_order(payload, idx, int(seed))[: int(query_per_class)])
    return selected


def _labels_for_indices(payload: Mapping[str, Any], indices: Sequence[int]) -> list[str]:
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    return [str(tx_ids[int(i)]) for i in indices]


def _build_prototypes(features: np.ndarray, labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    arr = np.asarray([str(v) for v in labels], dtype=object)
    label_values = sorted({str(v) for v in arr.tolist()})
    protos = []
    for label in label_values:
        protos.append(features[arr == label].mean(axis=0, keepdims=True)[0])
    return label_values, _normalize_rows(np.vstack(protos).astype(np.float32))


def _diag_variance(features: np.ndarray, labels: Sequence[str], label_values: Sequence[str], floor: float) -> dict[str, np.ndarray]:
    arr = np.asarray([str(v) for v in labels], dtype=object)
    out: dict[str, np.ndarray] = {}
    for label in label_values:
        subset = features[arr == str(label)]
        if subset.shape[0] <= 1:
            out[str(label)] = np.ones(features.shape[1], dtype=np.float32) * float(floor)
        else:
            out[str(label)] = np.maximum(subset.var(axis=0).astype(np.float32), float(floor))
    return out


def _topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    kk = max(1, min(int(k), values.shape[1]))
    part = np.partition(values, values.shape[1] - kk, axis=1)[:, -kk:]
    return part.mean(axis=1)


def _raw_components(
    features: np.ndarray,
    indices: Sequence[int],
    *,
    prototypes: np.ndarray,
    label_values: Sequence[str],
    diag_vars: Mapping[str, np.ndarray],
    memory_features: np.ndarray,
    proxy_centroid: np.ndarray,
    known_centroid: np.ndarray,
    qknn_k: int,
) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=int)
    x = _normalize_rows(features[idx])
    sims = x @ prototypes.T
    pred_pos = np.argmax(sims, axis=1)
    pred_scores = sims[np.arange(sims.shape[0]), pred_pos]
    sorted_scores = np.sort(sims, axis=1)
    second = sorted_scores[:, -2] if sims.shape[1] >= 2 else np.zeros_like(pred_scores)
    margin = pred_scores - second
    proto_dist = np.maximum(0.0, 1.0 - pred_scores)
    mem_sims = x @ memory_features.T
    knn_dist = np.maximum(0.0, 1.0 - _topk_mean(mem_sims, int(qknn_k)))
    maha = np.zeros(x.shape[0], dtype=np.float64)
    for row, pos in enumerate(pred_pos.tolist()):
        label = str(label_values[int(pos)])
        diff = x[row] - prototypes[int(pos)]
        maha[row] = float(np.mean((diff * diff) / np.asarray(diag_vars[label], dtype=np.float32)))
    exp_scores = np.exp(sims - sims.max(axis=1, keepdims=True))
    probs = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1) / max(math.log(sims.shape[1]), 1e-6)
    proxy_gap = (x @ proxy_centroid.reshape(-1)) - (x @ known_centroid.reshape(-1))
    return {
        "pred_pos": pred_pos.astype(int),
        "pred_score": pred_scores.astype(np.float64),
        "margin": margin.astype(np.float64),
        "proto_dist": proto_dist.astype(np.float64),
        "knn_dist": knn_dist.astype(np.float64),
        "maha": maha.astype(np.float64),
        "entropy": entropy.astype(np.float64),
        "proxy_gap": proxy_gap.astype(np.float64),
    }


def _fit_aware_model(payload: Mapping[str, Any], plan: AdapterTrainingPlan, args: argparse.Namespace) -> dict[str, Any]:
    features = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    known_indices = [*plan.source_old_indices, *plan.support_indices]
    known_labels = _labels_for_indices(payload, known_indices)
    label_values, prototypes = _build_prototypes(features[np.asarray(known_indices, dtype=int)], known_labels)
    diag_vars = _diag_variance(
        features[np.asarray(known_indices, dtype=int)],
        known_labels,
        label_values,
        floor=float(args.maha_var_floor),
    )
    memory_features = _normalize_rows(features[np.asarray(known_indices, dtype=int)])
    proxy_idx = np.asarray(plan.proxy_unknown_indices, dtype=int)
    proxy_centroid = _normalize_rows(features[proxy_idx].mean(axis=0, keepdims=True))[0]
    known_centroid = _normalize_rows(features[np.asarray(known_indices, dtype=int)].mean(axis=0, keepdims=True))[0]
    known = _raw_components(
        features,
        known_indices,
        prototypes=prototypes,
        label_values=label_values,
        diag_vars=diag_vars,
        memory_features=memory_features,
        proxy_centroid=proxy_centroid,
        known_centroid=known_centroid,
        qknn_k=int(args.qknn_k),
    )
    proxy = _raw_components(
        features,
        proxy_idx,
        prototypes=prototypes,
        label_values=label_values,
        diag_vars=diag_vars,
        memory_features=memory_features,
        proxy_centroid=proxy_centroid,
        known_centroid=known_centroid,
        qknn_k=int(args.qknn_k),
    )
    scales = {
        "proto_dist": float(np.quantile(known["proto_dist"], float(args.component_known_quantile))) + 1e-6,
        "knn_dist": float(np.quantile(known["knn_dist"], float(args.component_known_quantile))) + 1e-6,
        "maha": float(np.quantile(known["maha"], float(args.component_known_quantile))) + 1e-6,
        "entropy": float(np.quantile(known["entropy"], float(args.component_known_quantile))) + 1e-6,
        "proxy_gap": max(float(np.quantile(proxy["proxy_gap"], 0.50) - np.quantile(known["proxy_gap"], 0.90)), 1e-6),
    }
    known_score = _open_score(known, scales, args)
    proxy_score = _open_score(proxy, scales, args)
    receiver_reliability = _receiver_reliability(payload, known_indices, known_score)
    return {
        "features": features,
        "label_values": label_values,
        "prototypes": prototypes,
        "diag_vars": diag_vars,
        "memory_features": memory_features,
        "proxy_centroid": proxy_centroid,
        "known_centroid": known_centroid,
        "scales": scales,
        "known_open_score": known_score,
        "proxy_open_score": proxy_score,
        "receiver_reliability": receiver_reliability,
        "state_bytes": {
            "prototype_fp16_bytes": int(len(label_values) * features.shape[1] * 2),
            "diag_var_fp16_bytes": int(len(label_values) * features.shape[1] * 2),
            "centroid_fp16_bytes": int(features.shape[1] * 2 * 2),
            "scalar_calibration_bytes": int((len(scales) + len(receiver_reliability)) * 4),
        },
    }


def _open_score(components: Mapping[str, np.ndarray], scales: Mapping[str, float], args: argparse.Namespace) -> np.ndarray:
    proxy_term = np.maximum(0.0, np.asarray(components["proxy_gap"], dtype=np.float64)) / float(scales["proxy_gap"])
    score = (
        float(args.proto_weight) * np.asarray(components["proto_dist"], dtype=np.float64) / float(scales["proto_dist"])
        + float(args.knn_weight) * np.asarray(components["knn_dist"], dtype=np.float64) / float(scales["knn_dist"])
        + float(args.maha_weight) * np.asarray(components["maha"], dtype=np.float64) / float(scales["maha"])
        + float(args.entropy_weight) * np.asarray(components["entropy"], dtype=np.float64) / float(scales["entropy"])
        + float(args.proxy_weight) * proxy_term
    )
    return score.astype(np.float64)


def _receiver_reliability(payload: Mapping[str, Any], known_indices: Sequence[int], known_score: np.ndarray) -> dict[str, float]:
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    by_rx: dict[str, list[float]] = defaultdict(list)
    for idx, score in zip(known_indices, known_score.tolist()):
        by_rx[str(rx_ids[int(idx)])].append(float(score))
    medians = {rx: float(np.median(values)) for rx, values in by_rx.items() if values}
    if not medians:
        return {}
    global_q = float(np.quantile(list(medians.values()), 0.75)) + 1e-6
    return {rx: float(np.clip(1.0 - value / (2.0 * global_q), 0.05, 1.0)) for rx, value in medians.items()}


def _role_name(role: str) -> str:
    if role == "target_old":
        return "old"
    if role == "target_new":
        return "seen_new"
    if role == "target_unknown":
        return "unknown"
    return str(role)


def _build_event_rows(payload: Mapping[str, Any], plan: AdapterTrainingPlan, model: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    query_idx = _query_indices(payload, plan, int(args.query_per_class), int(args.seed))
    comps = _raw_components(
        np.asarray(model["features"], dtype=np.float32),
        query_idx,
        prototypes=np.asarray(model["prototypes"], dtype=np.float32),
        label_values=list(model["label_values"]),
        diag_vars=model["diag_vars"],
        memory_features=np.asarray(model["memory_features"], dtype=np.float32),
        proxy_centroid=np.asarray(model["proxy_centroid"], dtype=np.float32),
        known_centroid=np.asarray(model["known_centroid"], dtype=np.float32),
        qknn_k=int(args.qknn_k),
    )
    open_score = _open_score(comps, model["scales"], args)
    label_values = list(model["label_values"])
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    reliability = {str(k): float(v) for k, v in dict(model["receiver_reliability"]).items()}
    per_key_rank: Counter[tuple[str, str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for pos, idx in enumerate(query_idx):
        role = _role_name(str(roles[int(idx)]))
        true_label = str(tx_ids[int(idx)])
        receiver_id = str(rx_ids[int(idx)])
        pred_label = str(label_values[int(comps["pred_pos"][pos])])
        rank_key = (role, true_label, receiver_id)
        per_key_rank[rank_key] += 1
        event_rank = per_key_rank[rank_key]
        rows.append(
            {
                "event_id": f"{role}|{true_label}|rank{event_rank:04d}",
                "role": role,
                "true_label": true_label,
                "receiver_id": receiver_id,
                "pred_label": pred_label,
                "pred_score": float(comps["pred_score"][pos]),
                "margin": float(comps["margin"][pos]),
                "open_score": float(open_score[pos]),
                "proto_dist": float(comps["proto_dist"][pos]),
                "knn_dist": float(comps["knn_dist"][pos]),
                "maha": float(comps["maha"][pos]),
                "entropy": float(comps["entropy"][pos]),
                "proxy_gap": float(comps["proxy_gap"][pos]),
                "receiver_reliability": float(reliability.get(receiver_id, 0.50)),
                "bytes": float(args.evidence_packet_bytes),
                "latency_ms": float(args.receiver_latency_ms),
            }
        )
    return rows


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["receiver_reliability"]),
            float(row["open_score"]),
            -float(row["margin"]),
            str(row["receiver_id"]),
        ),
    )[: int(k)]


def _fuse_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: AwareProfile,
    threshold: float,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    votes: Counter[str] = Counter(str(row["pred_label"]) for row in rows)
    best_label, best_votes = votes.most_common(1)[0] if votes else (UNKNOWN_LABEL, 0)
    best_fraction = best_votes / max(len(rows), 1)
    label_rows = [row for row in rows if str(row["pred_label"]) == str(best_label)]
    label_score = float(np.mean([float(row["open_score"]) for row in label_rows])) if label_rows else float("inf")
    label_reliability = float(np.mean([float(row["receiver_reliability"]) for row in label_rows])) if label_rows else 0.0
    reject_votes = sum(float(row["open_score"]) >= float(threshold) * float(profile.reject_scale) for row in rows)
    reject_fraction = reject_votes / max(len(rows), 1)
    known_accept = (
        int(best_votes) >= int(profile.min_label_votes)
        and best_fraction >= float(profile.min_label_vote_fraction)
        and label_score <= float(threshold) * float(profile.accept_scale)
        and label_reliability >= float(profile.min_reliability)
    )
    unknown_confirm = reject_fraction >= float(profile.reject_vote_fraction) and not known_accept
    if known_accept:
        output_label = str(best_label)
        decision = "accept_known"
    elif unknown_confirm:
        output_label = UNKNOWN_LABEL
        decision = "reject_unknown"
    else:
        output_label = UNKNOWN_LABEL
        decision = "defer"
    total_bytes = float(sum(float(row["bytes"]) for row in rows))
    latency = float(max((float(row["latency_ms"]) for row in rows), default=0.0))
    first = rows[0]
    return {
        "event_id": str(first["event_id"]),
        "role": str(first["role"]),
        "true_label": str(first["true_label"]),
        "output_label": output_label,
        "decision": decision,
        "candidate_label": str(best_label),
        "candidate_votes": int(best_votes),
        "candidate_vote_fraction": float(best_fraction),
        "mean_open_score": float(np.mean([float(row["open_score"]) for row in rows])) if rows else 0.0,
        "reject_vote_fraction": float(reject_fraction),
        "participating_receivers": int(len(rows)),
        "bytes_per_event": total_bytes,
        "latency_ms": latency,
        "resource_pass": (
            (float(max_event_bytes) <= 0.0 or total_bytes <= float(max_event_bytes))
            and (float(max_event_latency_ms) <= 0.0 or latency <= float(max_event_latency_ms))
        ),
    }


def _finalize(events: Sequence[Mapping[str, Any]], old_labels: set[str], seen_labels: set[str]) -> dict[str, Any]:
    role_total = Counter()
    role_correct = Counter()
    role_defer = Counter()
    per_total: dict[str, Counter[str]] = {"old": Counter(), "seen_new": Counter()}
    per_correct: dict[str, Counter[str]] = {"old": Counter(), "seen_new": Counter()}
    for item in events:
        role = str(item["role"])
        true_label = str(item["true_label"])
        output = str(item["output_label"])
        decision = str(item["decision"])
        role_total[role] += 1
        if decision == "defer":
            role_defer[role] += 1
        if role in {"old", "seen_new"}:
            per_total[role][true_label] += 1
            if output == true_label:
                role_correct[role] += 1
                per_correct[role][true_label] += 1
        elif role == "unknown" and output == UNKNOWN_LABEL and decision == "reject_unknown":
            role_correct[role] += 1
    old_rates = {label: (per_correct["old"][label] / per_total["old"][label] if per_total["old"][label] else 0.0) for label in sorted(old_labels)}
    seen_rates = {label: (per_correct["seen_new"][label] / per_total["seen_new"][label] if per_total["seen_new"][label] else 0.0) for label in sorted(seen_labels)}
    known_total = role_total["old"] + role_total["seen_new"]
    known_defer = role_defer["old"] + role_defer["seen_new"]
    participants = [int(item["participating_receivers"]) for item in events]
    return {
        "old_total": int(role_total["old"]),
        "seen_new_total": int(role_total["seen_new"]),
        "unknown_total": int(role_total["unknown"]),
        "old_acc": role_correct["old"] / role_total["old"] if role_total["old"] else 0.0,
        "seen_new_acc": role_correct["seen_new"] / role_total["seen_new"] if role_total["seen_new"] else 0.0,
        "unknown_reject": role_correct["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0,
        "unknown_FAR": 1.0 - (role_correct["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0),
        "known_defer": known_defer / known_total if known_total else 0.0,
        "unknown_defer": role_defer["unknown"] / role_total["unknown"] if role_total["unknown"] else 0.0,
        "min_old": min(old_rates.values()) if old_rates else 0.0,
        "min_seen": min(seen_rates.values()) if seen_rates else 0.0,
        "avg_participating_receivers": float(np.mean(participants)) if participants else 0.0,
        "p95_participating_receivers": float(np.quantile(participants, 0.95)) if participants else 0.0,
        "per_old_class_acc": old_rates,
        "per_seen_new_class_acc": seen_rates,
    }


def _target_pass(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["old_acc"]) >= float(row["target_old_acc"])
        and float(row["min_old"]) >= float(row["target_min_old"])
        and float(row["seen_new_acc"]) >= float(row["target_seen_new_acc"])
        and float(row["min_seen"]) >= float(row["target_min_seen"])
        and float(row["unknown_reject"]) >= float(row["target_unknown_reject"])
        and bool(row["resource_pass"])
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_aware_ci(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_npz(args.feature_npz)
    plan = build_training_plan(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
    )
    model = _fit_aware_model(payload, plan, args)
    event_rows = _build_event_rows(payload, plan, model, args)
    receiver_ids = sorted({str(row["receiver_id"]) for row in event_rows})
    counts = parse_collab_counts(str(args.collab_counts), receiver_count=len(receiver_ids))
    requested = set(_profile_names(str(args.profiles)))
    profiles = [profile for profile in PROFILES if profile.name in requested]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        groups[str(row["event_id"])].append(row)
    old_labels = sorted({str(row["true_label"]) for row in event_rows if str(row["role"]) == "old"})
    seen_labels = sorted({str(row["true_label"]) for row in event_rows if str(row["role"]) == "seen_new"})
    known_scores = np.asarray(model["known_open_score"], dtype=np.float64)
    summary_rows: list[dict[str, Any]] = []
    profile_results: dict[str, Any] = {}
    for profile in profiles:
        threshold = float(np.quantile(known_scores, float(profile.known_quantile)))
        count_results: dict[str, Any] = {}
        for k in counts:
            fused = []
            for rows in groups.values():
                if len(rows) < int(k) and str(args.collab_group_policy) in {"exact_k", "same_max_budget"}:
                    continue
                selected = _select_receivers(rows, min(int(k), len(rows)))
                if selected:
                    fused.append(
                        _fuse_event(
                            selected,
                            profile=profile,
                            threshold=threshold,
                            max_event_bytes=float(args.max_event_bytes),
                            max_event_latency_ms=float(args.max_event_latency_ms),
                        )
                    )
            metrics = _finalize(fused, set(old_labels), set(seen_labels))
            metrics["bytes_per_event"] = sum(float(item["bytes_per_event"]) for item in fused) / max(len(fused), 1)
            metrics["latency_ms"] = max((float(item["latency_ms"]) for item in fused), default=0.0)
            metrics["resource_pass"] = all(bool(item["resource_pass"]) for item in fused) if fused else False
            row = {
                "profile": profile.name,
                "profile_description": profile.description,
                "collab_count": int(k),
                "calibrated_known_threshold": threshold,
                "old_acc": float(metrics["old_acc"]),
                "min_old": float(metrics["min_old"]),
                "seen_new_acc": float(metrics["seen_new_acc"]),
                "min_seen": float(metrics["min_seen"]),
                "unknown_reject": float(metrics["unknown_reject"]),
                "unknown_FAR": float(metrics["unknown_FAR"]),
                "known_defer": float(metrics["known_defer"]),
                "unknown_defer": float(metrics["unknown_defer"]),
                "bytes_per_event": float(metrics["bytes_per_event"]),
                "latency_ms": float(metrics["latency_ms"]),
                "avg_participating_receivers": float(metrics["avg_participating_receivers"]),
                "p95_participating_receivers": float(metrics["p95_participating_receivers"]),
                "target_old_acc": float(args.target_old_acc),
                "target_min_old": float(args.target_min_old),
                "target_seen_new_acc": float(args.target_seen_new_acc),
                "target_min_seen": float(args.target_min_seen),
                "target_unknown_reject": float(args.target_unknown_reject),
                "resource_pass": bool(metrics["resource_pass"]),
            }
            row["target_pass"] = _target_pass(row)
            summary_rows.append(row)
            count_results[str(k)] = {**metrics, "threshold": threshold}
        profile_results[profile.name] = {"profile": profile.__dict__, "counts": count_results}
    best_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["target_pass"],
            row["old_acc"] >= 0.80,
            row["seen_new_acc"] >= 0.78,
            row["unknown_reject"],
            row["old_acc"],
            row["seen_new_acc"],
        ),
        reverse=True,
    )
    state_bytes = dict(model["state_bytes"])
    state_bytes["total_fp16_state_bytes"] = int(sum(state_bytes.values()))
    result = {
        "algorithm": "AWARE-CI",
        "feature_npz": str(args.feature_npz),
        "alignment_policy": "receiver_domain_ranked_diagnostic",
        "target_unknown_eval_only": True,
        "training_counts": {
            "source_old": len(plan.source_old_indices),
            "proxy_unknown": len(plan.proxy_unknown_indices),
            "target_support": len(plan.support_indices),
            "target_unknown_eval_only": len(plan.target_unknown_indices),
            "target_unknown_training_count": 0,
        },
        "calibration": {
            "component_scales": model["scales"],
            "known_score_quantiles": np.quantile(np.asarray(model["known_open_score"], dtype=np.float64), [0.5, 0.9, 0.95, 0.99, 0.995]).tolist(),
            "proxy_score_quantiles": np.quantile(np.asarray(model["proxy_open_score"], dtype=np.float64), [0.1, 0.5, 0.9]).tolist(),
        },
        "state_bytes": state_bytes,
        "target_receivers": receiver_ids,
        "collab_counts_requested": str(args.collab_counts),
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_joint_row": best_rows[0] if best_rows else None,
        "target_gates": {
            "old_acc": float(args.target_old_acc),
            "min_old": float(args.target_min_old),
            "seen_new_acc": float(args.target_seen_new_acc),
            "min_seen": float(args.target_min_seen),
            "unknown_reject": float(args.target_unknown_reject),
        },
        "run_command_argv": [str(item) for item in sys.argv],
        "run_cwd": str(Path.cwd()),
    }
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, event_rows)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--output_summary_csv", type=Path)
    p.add_argument("--output_evidence_csv", type=Path)
    p.add_argument("--profiles", default="all")
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070704)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--component_known_quantile", type=float, default=0.95)
    p.add_argument("--proto_weight", type=float, default=0.35)
    p.add_argument("--knn_weight", type=float, default=0.30)
    p.add_argument("--maha_weight", type=float, default=0.20)
    p.add_argument("--entropy_weight", type=float, default=0.10)
    p.add_argument("--proxy_weight", type=float, default=0.05)
    p.add_argument("--maha_var_floor", type=float, default=1e-3)
    p.add_argument("--evidence_packet_bytes", type=float, default=96.0)
    p.add_argument("--receiver_latency_ms", type=float, default=0.35)
    p.add_argument("--max_event_bytes", type=float, default=1152.0)
    p.add_argument("--max_event_latency_ms", type=float, default=20.0)
    p.add_argument("--target_old_acc", type=float, default=0.99)
    p.add_argument("--target_min_old", type=float, default=0.95)
    p.add_argument("--target_seen_new_acc", type=float, default=0.97)
    p.add_argument("--target_min_seen", type=float, default=0.93)
    p.add_argument("--target_unknown_reject", type=float, default=0.99)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = run_aware_ci(args)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    print(json.dumps({"best_joint_row": result["best_joint_row"], "target_receivers": result["target_receivers"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
