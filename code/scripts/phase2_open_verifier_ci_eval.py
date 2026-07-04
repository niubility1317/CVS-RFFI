#!/usr/bin/env python
"""OVC-CI open-verifier collaborative inference for Stage2-C qknn8 features.

The module trains a compact open-set verifier from source old rows, source-side
proxy_unknown rows, and target old/seen-new K-shot support. Target unknown rows
are evaluation-only. The verifier operates on prototype-derived scalar evidence,
so deployment only needs class prototypes plus a tiny linear risk head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import _normalize_rows, load_feature_npz  # noqa: E402
from phase2_proxy_adapter_ci_eval import AdapterTrainingPlan, build_training_plan  # noqa: E402
from evaluation.collaborative_open_set_qknn_eval import parse_collab_counts  # noqa: E402

UNKNOWN_LABEL = "__unknown__"


@dataclass(frozen=True)
class VerifierProfile:
    name: str
    description: str
    known_risk_quantile: float
    unknown_risk_scale: float
    min_label_votes: int
    min_label_vote_fraction: float
    max_known_risk: float
    reject_vote_fraction: float


PROFILES: tuple[VerifierProfile, ...] = (
    VerifierProfile(
        name="ovc_old_guard",
        description="old-retention constrained verifier",
        known_risk_quantile=0.995,
        unknown_risk_scale=1.00,
        min_label_votes=1,
        min_label_vote_fraction=0.34,
        max_known_risk=0.98,
        reject_vote_fraction=0.67,
    ),
    VerifierProfile(
        name="ovc_balanced",
        description="balanced verifier with majority unknown confirmation",
        known_risk_quantile=0.990,
        unknown_risk_scale=0.95,
        min_label_votes=2,
        min_label_vote_fraction=0.40,
        max_known_risk=0.92,
        reject_vote_fraction=0.60,
    ),
    VerifierProfile(
        name="ovc_unknown_guard",
        description="stricter unknown rejection diagnostic",
        known_risk_quantile=0.970,
        unknown_risk_scale=0.90,
        min_label_votes=2,
        min_label_vote_fraction=0.46,
        max_known_risk=0.86,
        reject_vote_fraction=0.50,
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
        raise argparse.ArgumentTypeError(f"unknown OVC-CI profile(s): {', '.join(missing)}")
    return names


def _build_prototypes(features: np.ndarray, labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    label_arr = np.asarray([str(v) for v in labels], dtype=object)
    label_values = sorted({str(v) for v in label_arr.tolist()})
    protos = []
    for label in label_values:
        protos.append(_normalize_rows(features[label_arr == label].mean(axis=0, keepdims=True))[0])
    return label_values, _normalize_rows(np.vstack(protos).astype(np.float32))


def _stable_order(payload: Mapping[str, Any], indices: Sequence[int], seed: int) -> list[int]:
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    day_ids = np.asarray(payload["day_ids"]).astype(str)
    sig_ids = np.asarray(payload["sig_ids"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)

    def key(i: int) -> tuple[str, str, str, str, int]:
        return (str(rx_ids[i]), str(tx_ids[i]), str(day_ids[i]), str(sig_ids[i]), (int(i) + int(seed)) % 1000003)

    return sorted([int(i) for i in indices], key=key)


def _evidence_features(
    features: np.ndarray,
    prototypes: np.ndarray,
    indices: Sequence[int],
    *,
    proxy_centroid: np.ndarray,
    known_centroid: np.ndarray,
    entropy_temperature: float,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    x = _normalize_rows(features[idx])
    scores = x @ prototypes.T
    sorted_scores = np.sort(scores, axis=1)
    best = sorted_scores[:, -1]
    second = sorted_scores[:, -2] if scores.shape[1] >= 2 else np.zeros_like(best)
    margin = best - second
    top3 = sorted_scores[:, -min(3, scores.shape[1]) :].mean(axis=1)
    temp = max(float(entropy_temperature), 1e-6)
    exp_scores = np.exp((scores - scores.max(axis=1, keepdims=True)) / temp)
    probs = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1) / max(math.log(scores.shape[1]), 1e-6)
    proxy_sim = x @ proxy_centroid.reshape(-1)
    known_sim = x @ known_centroid.reshape(-1)
    return np.vstack([best, margin, top3, entropy, proxy_sim, known_sim, proxy_sim - known_sim]).T.astype(np.float32)


def _train_open_verifier(
    features: np.ndarray,
    plan: AdapterTrainingPlan,
    payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(str(args.device) if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    known_indices = [*plan.source_old_indices, *plan.support_indices]
    known_labels = [str(tx_ids[i]) for i in known_indices]
    labels, prototypes = _build_prototypes(features[np.asarray(known_indices, dtype=int)], known_labels)
    proxy_idx = np.asarray(plan.proxy_unknown_indices, dtype=int)
    proxy_centroid = _normalize_rows(features[proxy_idx].mean(axis=0, keepdims=True))[0]
    known_centroid = _normalize_rows(features[np.asarray(known_indices, dtype=int)].mean(axis=0, keepdims=True))[0]
    x_known = _evidence_features(
        features,
        prototypes,
        known_indices,
        proxy_centroid=proxy_centroid,
        known_centroid=known_centroid,
        entropy_temperature=float(args.entropy_temperature),
    )
    x_proxy = _evidence_features(
        features,
        prototypes,
        proxy_idx,
        proxy_centroid=proxy_centroid,
        known_centroid=known_centroid,
        entropy_temperature=float(args.entropy_temperature),
    )
    train_x = np.vstack([x_known, x_proxy]).astype(np.float32)
    train_y = np.concatenate([np.zeros(x_known.shape[0]), np.ones(x_proxy.shape[0])]).astype(np.float32)
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0).astype(np.float32), 1e-6)
    z = torch.as_tensor((train_x - mean[None, :]) / std[None, :], dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    weight = torch.zeros((z.shape[1],), dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.AdamW([weight, bias], lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_weight = torch.as_tensor(float(x_known.shape[0]) / max(float(x_proxy.shape[0]), 1.0), dtype=torch.float32, device=device)
    start = time.perf_counter()
    losses: list[float] = []
    for _ in range(int(args.verifier_epochs)):
        logits = z @ weight + bias
        risk = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        known_risk = risk[: x_known.shape[0]]
        loss = loss + float(args.known_risk_penalty) * F.relu(known_risk - float(args.known_risk_target)).mean()
        loss = loss + float(args.weight_l2) * (weight**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weight, bias], float(args.grad_clip))
        opt.step()
        losses.append(float(loss.detach().item()))
    with torch.no_grad():
        train_risk = torch.sigmoid(z @ weight + bias).detach().cpu().numpy()
    return {
        "label_values": labels,
        "prototypes": prototypes,
        "proxy_centroid": proxy_centroid.astype(np.float32),
        "known_centroid": known_centroid.astype(np.float32),
        "feature_mean": mean,
        "feature_std": std,
        "weight": weight.detach().cpu().numpy().astype(np.float32),
        "bias": float(bias.detach().cpu().item()),
        "train_risk": train_risk.astype(np.float32),
        "train_known_count": int(x_known.shape[0]),
        "train_proxy_count": int(x_proxy.shape[0]),
        "train_seconds": float(time.perf_counter() - start),
        "loss_trace_tail": losses[-5:],
        "device": str(device),
    }


def _risk_scores(model: Mapping[str, Any], features: np.ndarray, indices: Sequence[int], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = _evidence_features(
        features,
        np.asarray(model["prototypes"], dtype=np.float32),
        indices,
        proxy_centroid=np.asarray(model["proxy_centroid"], dtype=np.float32),
        known_centroid=np.asarray(model["known_centroid"], dtype=np.float32),
        entropy_temperature=float(args.entropy_temperature),
    )
    z = (phi - np.asarray(model["feature_mean"], dtype=np.float32)[None, :]) / np.asarray(model["feature_std"], dtype=np.float32)[None, :]
    logits = z @ np.asarray(model["weight"], dtype=np.float32) + float(model["bias"])
    risk = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    scores = _normalize_rows(features[np.asarray(indices, dtype=int)]) @ np.asarray(model["prototypes"], dtype=np.float32).T
    pred_pos = np.argmax(scores, axis=1)
    pred_labels = np.asarray(model["label_values"], dtype=object)[pred_pos].astype(str)
    pred_scores = scores[np.arange(scores.shape[0]), pred_pos]
    return risk.astype(np.float64), pred_labels, pred_scores.astype(np.float64)


def _query_indices(payload: Mapping[str, Any], plan: AdapterTrainingPlan, query_per_class: int, seed: int) -> list[int]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    support = set(int(i) for i in plan.support_indices)
    selected: list[int] = []
    for role in ["target_old", "target_new", "target_unknown"]:
        for rx in sorted({str(v) for v in rx_ids[roles == role].tolist()}):
            for tx in sorted({str(v) for v in tx_ids[(roles == role) & (rx_ids == rx)].tolist()}):
                idx = [i for i in np.where((roles == role) & (rx_ids == rx) & (tx_ids == tx))[0].tolist() if int(i) not in support]
                selected.extend(_stable_order(payload, idx, int(seed))[: int(query_per_class)])
    return selected


def _role_name(role: str) -> str:
    if role == "target_old":
        return "old"
    if role == "target_new":
        return "seen_new"
    if role == "target_unknown":
        return "unknown"
    return str(role)


def _build_event_rows(
    payload: Mapping[str, Any],
    model: Mapping[str, Any],
    plan: AdapterTrainingPlan,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    features = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    query_idx = _query_indices(payload, plan, int(args.query_per_class), int(args.seed))
    risk, pred_labels, pred_scores = _risk_scores(model, features, query_idx, args)
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    per_key_rank: Counter[tuple[str, str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for local_pos, idx in enumerate(query_idx):
        role = _role_name(str(roles[idx]))
        true_label = str(tx_ids[idx])
        receiver_id = str(rx_ids[idx])
        rank_key = (role, true_label, receiver_id)
        per_key_rank[rank_key] += 1
        event_rank = per_key_rank[rank_key]
        rows.append(
            {
                "event_id": f"{role}|{true_label}|rank{event_rank:04d}",
                "role": role,
                "true_label": true_label,
                "receiver_id": receiver_id,
                "risk": float(risk[local_pos]),
                "pred_label": str(pred_labels[local_pos]),
                "pred_score": float(pred_scores[local_pos]),
                "bytes": float(args.evidence_packet_bytes),
                "latency_ms": float(args.receiver_latency_ms),
            }
        )
    return rows


def _select_receivers(rows: Sequence[Mapping[str, Any]], k: int, policy: str) -> list[Mapping[str, Any]]:
    if str(policy) == "risk_prior":
        return sorted(rows, key=lambda row: (float(row["risk"]), -float(row["pred_score"]), str(row["receiver_id"])))[: int(k)]
    return sorted(rows, key=lambda row: str(row["receiver_id"]))[: int(k)]


def _fuse_event(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: VerifierProfile,
    threshold: float,
    max_event_bytes: float,
    max_event_latency_ms: float,
) -> dict[str, Any]:
    risk_values = [float(row["risk"]) for row in rows]
    reject_votes = sum(v >= float(threshold) * float(profile.unknown_risk_scale) for v in risk_values)
    reject_fraction = reject_votes / max(len(rows), 1)
    votes: Counter[str] = Counter(str(row["pred_label"]) for row in rows)
    best_label, best_votes = votes.most_common(1)[0] if votes else (UNKNOWN_LABEL, 0)
    best_fraction = best_votes / max(len(rows), 1)
    label_risk_values = [float(row["risk"]) for row in rows if str(row["pred_label"]) == str(best_label)]
    label_risk = sum(label_risk_values) / max(len(label_risk_values), 1)
    known_accept = (
        int(best_votes) >= int(profile.min_label_votes)
        and best_fraction >= float(profile.min_label_vote_fraction)
        and label_risk <= min(float(profile.max_known_risk), float(threshold) * float(profile.unknown_risk_scale))
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
    total_bytes = sum(float(row["bytes"]) for row in rows)
    latency = max((float(row["latency_ms"]) for row in rows), default=0.0)
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
        "mean_risk": float(sum(risk_values) / max(len(risk_values), 1)),
        "reject_vote_fraction": float(reject_fraction),
        "receiver_count": int(len(rows)),
        "bytes_per_event": float(total_bytes),
        "latency_ms": float(latency),
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


def run_ovc_ci(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_feature_npz(args.feature_npz)
    features = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    plan = build_training_plan(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
    )
    model = _train_open_verifier(features, plan, payload, args)
    event_rows = _build_event_rows(payload, model, plan, args)
    receiver_ids = sorted({str(row["receiver_id"]) for row in event_rows})
    old_labels = sorted({str(row["true_label"]) for row in event_rows if str(row["role"]) == "old"})
    seen_labels = sorted({str(row["true_label"]) for row in event_rows if str(row["role"]) == "seen_new"})
    counts = parse_collab_counts(str(args.collab_counts), receiver_count=len(receiver_ids))
    requested = set(_profile_names(str(args.profiles)))
    profiles = [profile for profile in PROFILES if profile.name in requested]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        groups[str(row["event_id"])].append(row)
    train_risk = np.asarray(model["train_risk"], dtype=np.float64)
    known_count = int(model["train_known_count"])
    summary_rows: list[dict[str, Any]] = []
    profile_results: dict[str, Any] = {}
    for profile in profiles:
        threshold = float(np.quantile(train_risk[:known_count], float(profile.known_risk_quantile)))
        count_results: dict[str, Any] = {}
        for k in counts:
            fused = []
            for rows in groups.values():
                if len(rows) < int(k) and str(args.collab_group_policy) in {"exact_k", "same_max_budget"}:
                    continue
                selected = _select_receivers(rows, min(int(k), len(rows)), str(args.receiver_selection_policy))
                if not selected:
                    continue
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
            if bool(args.include_event_results):
                metrics["event_results"] = fused
            count_results[str(k)] = metrics
            row = {
                "profile": profile.name,
                "profile_description": profile.description,
                "collab_count": int(k),
                "risk_threshold": threshold,
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
                "target_old_acc": float(args.target_old_acc),
                "target_min_old": float(args.target_min_old),
                "target_seen_new_acc": float(args.target_seen_new_acc),
                "target_min_seen": float(args.target_min_seen),
                "target_unknown_reject": float(args.target_unknown_reject),
                "resource_pass": bool(metrics["resource_pass"]),
            }
            row["target_pass"] = _target_pass(row)
            summary_rows.append(row)
        profile_results[profile.name] = {
            "profile": profile.__dict__,
            "risk_threshold": threshold,
            "receiver_count": len(receiver_ids),
            "observed_receiver_ids": receiver_ids,
            "counts": count_results,
        }
    best_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["target_pass"],
            row["old_acc"] >= 0.80,
            row["unknown_reject"],
            row["old_acc"],
            row["seen_new_acc"],
            -row["known_defer"],
        ),
        reverse=True,
    )
    state_bytes = {
        "prototype_fp16_bytes": int(len(model["label_values"]) * features.shape[1] * 2),
        "verifier_fp16_bytes": int((len(model["weight"]) + len(model["feature_mean"]) + len(model["feature_std"]) + 1) * 2),
        "centroid_fp16_bytes": int(features.shape[1] * 2 * 2),
    }
    state_bytes["total_fp16_state_bytes"] = sum(state_bytes.values())
    result = {
        "algorithm": "OVC-CI",
        "feature_npz": str(args.feature_npz),
        "target_unknown_eval_only": True,
        "training_counts": {
            "source_old": len(plan.source_old_indices),
            "proxy_unknown": len(plan.proxy_unknown_indices),
            "target_support": len(plan.support_indices),
            "target_unknown_eval_only": len(plan.target_unknown_indices),
            "target_unknown_training_count": 0,
        },
        "state_bytes": state_bytes,
        "train_metrics": {
            "device": model["device"],
            "train_seconds": model["train_seconds"],
            "loss_trace_tail": model["loss_trace_tail"],
            "known_risk_quantiles": np.quantile(train_risk[:known_count], [0.5, 0.9, 0.95, 0.99, 0.995]).tolist(),
            "proxy_risk_quantiles": np.quantile(train_risk[known_count:], [0.1, 0.5, 0.9]).tolist(),
        },
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
    p.add_argument("--collab_group_policy", default="available_up_to_k", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--receiver_selection_policy", default="fixed_receiver_order", choices=["fixed_receiver_order", "risk_prior"])
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070601)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verifier_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--weight_l2", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--entropy_temperature", type=float, default=0.05)
    p.add_argument("--known_risk_target", type=float, default=0.03)
    p.add_argument("--known_risk_penalty", type=float, default=5.0)
    p.add_argument("--evidence_packet_bytes", type=float, default=48.0)
    p.add_argument("--receiver_latency_ms", type=float, default=0.25)
    p.add_argument("--max_event_bytes", type=float, default=1152.0)
    p.add_argument("--max_event_latency_ms", type=float, default=20.0)
    p.add_argument("--include_event_results", action="store_true")
    p.add_argument("--target_old_acc", type=float, default=0.99)
    p.add_argument("--target_min_old", type=float, default=0.95)
    p.add_argument("--target_seen_new_acc", type=float, default=0.97)
    p.add_argument("--target_min_seen", type=float, default=0.93)
    p.add_argument("--target_unknown_reject", type=float, default=0.99)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = run_ovc_ci(args)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    print(json.dumps({"best_joint_row": result["best_joint_row"], "target_receivers": result["target_receivers"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
