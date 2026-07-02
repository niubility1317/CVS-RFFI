#!/usr/bin/env python
"""Evaluate Phase1-only class-prototype open-set rejection.

The evaluator consumes repeated satellite-view feature exports, groups repeated
views by sample metadata, builds source-old class prototypes from the frozen
Phase1 embedding, and rejects samples whose grouped embedding is too far from
the predicted old-class prototype. Thresholds are calibrated only from source
old and optional source-side proxy unknown groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.gate_metrics import binary_reject_metrics  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


def _as_str_array(value: np.ndarray, n: int) -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [str(arr.item())] * int(n)
    return [canonical_tx_id(v) for v in arr.reshape(-1).tolist()]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files or "tx_logits" not in data.files:
            raise ValueError(f"{path} must contain features and tx_logits")
        features = torch.as_tensor(np.asarray(data["features"]), dtype=torch.float32)
        logits = torch.as_tensor(np.asarray(data["tx_logits"]), dtype=torch.float32)
        n = int(features.shape[0])

        def pick(key: str, default: np.ndarray) -> np.ndarray:
            return np.asarray(data[key]) if key in data.files else default

        manifest: dict[str, Any] = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        return {
            "features": features,
            "tx_logits": logits,
            "dataset_role": _as_str_array(pick("dataset_role", np.asarray([""] * n)), n),
            "tx_ids": _as_str_array(pick("tx_ids", np.asarray([""] * n)), n),
            "rx_ids": _as_str_array(pick("rx_ids", np.asarray([""] * n)), n),
            "day_ids": _as_str_array(pick("day_ids", np.asarray([""] * n)), n),
            "eq_ids": _as_str_array(pick("eq_ids", np.asarray([""] * n)), n),
            "sig_ids": _as_str_array(pick("sig_ids", np.asarray([str(i) for i in range(n)])), n),
            "manifest": manifest,
        }


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _class_tx_map(source_tx_ids: Sequence[str]) -> dict[int, str]:
    return {i: canonical_tx_id(tx) for i, tx in enumerate(source_tx_ids)}


def _group_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
    for i, role in enumerate(payload["dataset_role"]):
        key = (
            str(role),
            canonical_tx_id(payload["tx_ids"][i]),
            canonical_tx_id(payload["rx_ids"][i]),
            canonical_tx_id(payload["day_ids"][i]),
            canonical_tx_id(payload["eq_ids"][i]),
            canonical_tx_id(payload["sig_ids"][i]),
        )
        groups[key].append(int(i))

    out: list[dict[str, Any]] = []
    features = payload["features"]
    logits = payload["tx_logits"]
    for key, idx in sorted(groups.items()):
        idx_t = torch.tensor(idx, dtype=torch.long)
        z = features[idx_t].float()
        lo = logits[idx_t].float()
        probs = F.softmax(lo, dim=1)
        mean_logits = lo.mean(dim=0)
        mean_probs = F.softmax(mean_logits, dim=0)
        pred_class = int(mean_logits.argmax().item())
        conf, pred = probs.max(dim=1)
        margin = float((mean_probs.topk(k=min(2, mean_probs.numel())).values.diff().abs()[0]).item()) if mean_probs.numel() >= 2 else 0.0
        out.append(
            {
                "key": key,
                "role": key[0],
                "tx_id": key[1],
                "rx_id": key[2],
                "day_id": key[3],
                "eq_id": key[4],
                "sig_id": key[5],
                "view_count": len(idx),
                "z_mean": z.mean(dim=0),
                "z_std": z.std(dim=0, unbiased=False),
                "mean_logits": mean_logits,
                "pred_class": pred_class,
                "mean_confidence": float(mean_probs.max().item()),
                "view_confidence": float(conf.mean().item()),
                "vote_frac": float((pred == pred_class).float().mean().item()),
                "entropy": float((-(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()).item()),
                "margin": margin,
            }
        )
    return out


def _standardize(train_x: torch.Tensor, all_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (all_x - mean) / std, {"feature_dim": int(all_x.size(1)), "train_count": int(train_x.size(0))}


def _prototype_scores(
    x: torch.Tensor,
    groups: Sequence[Mapping[str, Any]],
    train_known_mask: torch.Tensor,
    class_to_tx: Mapping[int, str],
    *,
    metric: str,
    confidence_weight: float,
    entropy_weight: float,
    margin_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x = x[train_known_mask]
    if train_x.numel() == 0:
        raise ValueError("no train-known source groups found")
    prototypes: dict[str, torch.Tensor] = {}
    variances: dict[str, torch.Tensor] = {}
    fallback_proto = train_x.mean(dim=0)
    fallback_var = train_x.var(dim=0, unbiased=False).clamp_min(1e-4)
    for tx in sorted({canonical_tx_id(g["tx_id"]) for i, g in enumerate(groups) if bool(train_known_mask[i])}):
        idx = torch.tensor([i for i, g in enumerate(groups) if bool(train_known_mask[i]) and canonical_tx_id(g["tx_id"]) == tx], dtype=torch.long)
        cls_x = x[idx]
        prototypes[tx] = cls_x.mean(dim=0)
        variances[tx] = cls_x.var(dim=0, unbiased=False).clamp_min(1e-4)

    raw_scores: list[float] = []
    proto_tx: list[str] = []
    metric_name = str(metric).lower()
    for i, g in enumerate(groups):
        pred_tx = class_to_tx.get(int(g["pred_class"]), str(g["pred_class"]))
        proto = prototypes.get(pred_tx, fallback_proto)
        var = variances.get(pred_tx, fallback_var)
        z = x[i]
        if metric_name == "cosine":
            score = 1.0 - float(F.cosine_similarity(z.unsqueeze(0), proto.unsqueeze(0)).item())
        elif metric_name == "euclidean":
            score = float(torch.norm(z - proto, p=2).item())
        elif metric_name == "diag_mahalanobis":
            score = float((((z - proto) ** 2) / var).mean().item())
        else:
            raise ValueError(f"unknown metric={metric!r}")
        score += float(confidence_weight) * (1.0 - float(g["mean_confidence"]))
        score += float(entropy_weight) * float(g["entropy"])
        score -= float(margin_weight) * float(g["margin"])
        raw_scores.append(float(score))
        proto_tx.append(pred_tx)
    return np.asarray(raw_scores, dtype=np.float64), {
        "metric": metric_name,
        "confidence_weight": float(confidence_weight),
        "entropy_weight": float(entropy_weight),
        "margin_weight": float(margin_weight),
        "prototype_count": len(prototypes),
        "prototype_tx_ids": sorted(prototypes),
        "predicted_tx_ids": proto_tx,
    }


def _quantile(values: Sequence[float], q: float, fallback: float) -> float:
    if not values:
        return float(fallback)
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _thresholds(
    scores: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
    pred_tx_ids: Sequence[str],
    train_known_mask: torch.Tensor,
    proxy_mask: torch.Tensor,
    *,
    policy: str,
    source_q: float,
    proxy_q: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    source_scores = [float(scores[i]) for i in range(len(groups)) if bool(train_known_mask[i])]
    proxy_scores = [float(scores[i]) for i in range(len(groups)) if bool(proxy_mask[i])]
    global_source_t = _quantile(source_scores, source_q, float(np.max(scores)))
    global_proxy_t = _quantile(proxy_scores, proxy_q, global_source_t)
    tx_ids = sorted(set(pred_tx_ids))
    thresholds: dict[str, float] = {}
    source_thresholds: dict[str, float] = {}
    proxy_thresholds: dict[str, float] = {}
    for tx in tx_ids:
        src_vals = [float(scores[i]) for i, pred_tx in enumerate(pred_tx_ids) if pred_tx == tx and bool(train_known_mask[i])]
        proxy_vals = [float(scores[i]) for i, pred_tx in enumerate(pred_tx_ids) if pred_tx == tx and bool(proxy_mask[i])]
        src_t = _quantile(src_vals, source_q, global_source_t)
        proxy_t = _quantile(proxy_vals, proxy_q, global_proxy_t)
        if policy == "source_accept":
            t = src_t
        elif policy == "proxy_far":
            t = proxy_t
        elif policy == "min_source_proxy":
            t = min(src_t, proxy_t)
        elif policy == "max_source_proxy":
            t = max(src_t, proxy_t)
        else:
            raise ValueError(f"unknown threshold_policy={policy!r}")
        thresholds[tx] = float(t)
        source_thresholds[tx] = float(src_t)
        proxy_thresholds[tx] = float(proxy_t)

    def _accept_rate(mask: torch.Tensor) -> float:
        total = 0
        acc = 0
        for i, pred_tx in enumerate(pred_tx_ids):
            if bool(mask[i]):
                total += 1
                acc += int(float(scores[i]) <= thresholds[pred_tx])
        return 0.0 if total <= 0 else float(acc) / float(total)

    return thresholds, {
        "threshold_policy": policy,
        "threshold_scope": "per_predicted_old_class",
        "source_accept_quantile": float(source_q),
        "proxy_far_quantile": float(proxy_q),
        "global_source_threshold": global_source_t,
        "global_proxy_threshold": global_proxy_t,
        "source_accept_rate_at_threshold": _accept_rate(train_known_mask),
        "proxy_false_accept_rate_at_threshold": _accept_rate(proxy_mask),
        "class_thresholds": thresholds,
        "class_source_thresholds": source_thresholds,
        "class_proxy_thresholds": proxy_thresholds,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    groups = _group_payload(payload)
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids must define Phase1 known class order")
    class_to_tx = _class_tx_map(source_tx_ids)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    train_known_roles = _parse_roles(args.train_known_roles)
    proxy_roles = _parse_roles(args.proxy_unknown_roles)
    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}

    x_raw = torch.stack([g["z_mean"] for g in groups], dim=0)
    source_train_base_mask = torch.tensor(
        [g["role"] in train_known_roles and canonical_tx_id(g["tx_id"]) in source_known for g in groups],
        dtype=torch.bool,
    )
    source_closed_correct_mask = torch.tensor(
        [
            bool(class_to_tx.get(int(g["pred_class"]), str(g["pred_class"])) == canonical_tx_id(g["tx_id"]))
            for g in groups
        ],
        dtype=torch.bool,
    )
    train_known_mask = source_train_base_mask & (
        source_closed_correct_mask if bool(args.train_known_correct_only) else torch.ones_like(source_train_base_mask, dtype=torch.bool)
    )
    proxy_mask = torch.tensor([g["role"] in proxy_roles for g in groups], dtype=torch.bool)
    if bool(args.source_incorrect_as_proxy):
        proxy_mask = proxy_mask | (source_train_base_mask & ~source_closed_correct_mask)
    if not bool(train_known_mask.any()):
        raise ValueError("no train-known source groups found")
    if not bool(proxy_mask.any()):
        raise ValueError("no proxy unknown groups found")

    x_all, scale_info = _standardize(x_raw[train_known_mask], x_raw)
    scores, proto_info = _prototype_scores(
        x_all,
        groups,
        train_known_mask,
        class_to_tx,
        metric=str(args.metric),
        confidence_weight=float(args.confidence_weight),
        entropy_weight=float(args.entropy_weight),
        margin_weight=float(args.margin_weight),
    )
    pred_tx_ids = list(proto_info.pop("predicted_tx_ids"))
    thresholds, th_info = _thresholds(
        scores,
        groups,
        pred_tx_ids,
        train_known_mask,
        proxy_mask,
        policy=str(args.threshold_policy),
        source_q=float(args.source_accept_quantile),
        proxy_q=float(args.proxy_far_quantile),
    )

    rows: list[dict[str, Any]] = []
    y_unknown: list[bool] = []
    reject_scores: list[float] = []
    accepted_flags: list[bool] = []
    known_total = known_closed_correct = known_accepted = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    for i, g in enumerate(groups):
        tx = canonical_tx_id(g["tx_id"])
        pred_tx = pred_tx_ids[i]
        score = float(scores[i])
        threshold = float(thresholds[pred_tx])
        accepted = score <= threshold
        is_known_query = g["role"] in known_roles and tx in source_known
        is_unknown_query = g["role"] in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
        closed_correct = bool(is_known_query and pred_tx == tx)
        if is_known_query:
            known_total += 1
            known_closed_correct += int(closed_correct)
            known_accepted += int(accepted)
            if accepted:
                known_correct_accepted += int(pred_tx == tx)
        if is_unknown_query:
            unknown_total += 1
            unknown_accepted += int(accepted)
        if is_known_query or is_unknown_query:
            y_unknown.append(bool(is_unknown_query))
            reject_scores.append(score)
            accepted_flags.append(bool(accepted))
        rows.append(
            {
                "group": i,
                "role": g["role"],
                "tx_id": tx,
                "rx_id": g["rx_id"],
                "day_id": g["day_id"],
                "sig_id": g["sig_id"],
                "view_count": g["view_count"],
                "pred_tx_id": pred_tx,
                "accepted": int(accepted),
                "is_known_query": int(is_known_query),
                "is_unknown_query": int(is_unknown_query),
                "closed_correct_known": int(closed_correct),
                "accepted_correct_known": int(bool(accepted and closed_correct)),
                "unknown_score": f"{score:.8f}",
                "threshold": f"{threshold:.8f}",
                "mean_confidence": f"{float(g['mean_confidence']):.8f}",
                "vote_frac": f"{float(g['vote_frac']):.8f}",
                "entropy": f"{float(g['entropy']):.8f}",
                "margin": f"{float(g['margin']):.8f}",
            }
        )

    known_closed_accuracy = _safe_rate(known_closed_correct, known_total)
    known_full_accuracy = _safe_rate(known_correct_accepted, known_total)
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    old_drop_pp = None
    if known_closed_accuracy is not None and known_full_accuracy is not None:
        old_drop_pp = 100.0 * (float(known_closed_accuracy) - float(known_full_accuracy))
    metrics = {
        "phase": "phase1_only_source_prototype_reject",
        "threshold_scope": "source_old_and_source_proxy_unknown_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(args.feature_npz),
        "source_tx_ids": source_tx_ids,
        "group_count": len(groups),
        "source_train_base_count": int(source_train_base_mask.sum().item()),
        "source_train_closed_correct_count": int((source_train_base_mask & source_closed_correct_mask).sum().item()),
        "train_known_correct_only": bool(args.train_known_correct_only),
        "source_incorrect_as_proxy": bool(args.source_incorrect_as_proxy),
        "train_known_count": int(train_known_mask.sum().item()),
        "proxy_unknown_count": int(proxy_mask.sum().item()),
        "scoring": {**proto_info, **scale_info},
        "threshold": th_info,
        "known_query_count": int(known_total),
        "known_closed_accuracy_no_reject": known_closed_accuracy,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy_after_reject": known_full_accuracy,
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
        "old_retention_vs_closed": None if known_closed_correct <= 0 else float(known_correct_accepted) / float(known_closed_correct),
        "old_drop_pp_vs_closed": old_drop_pp,
        "max_old_drop_pp": float(args.max_old_drop_pp),
        "unknown_query_count": int(unknown_total),
        "unknown_FAR": unknown_far,
        "unknown_reject_rate": None if unknown_total == 0 else 1.0 - float(unknown_accepted) / float(unknown_total),
        "unknown_far_target": float(args.unknown_far_target),
        "passes_unknown_far_target": None if unknown_far is None else float(unknown_far) <= float(args.unknown_far_target),
        "passes_old_drop_target": None if old_drop_pp is None else float(old_drop_pp) <= float(args.max_old_drop_pp),
        "passes_dual_target": None
        if unknown_far is None or old_drop_pp is None
        else (float(unknown_far) <= float(args.unknown_far_target) and float(old_drop_pp) <= float(args.max_old_drop_pp)),
        "manifest": payload.get("manifest", {}),
    }
    if y_unknown:
        metrics.update(
            binary_reject_metrics(
                torch.tensor(y_unknown, dtype=torch.bool),
                torch.tensor(reject_scores, dtype=torch.float32),
                torch.tensor(accepted_flags, dtype=torch.bool),
            )
        )
    if args.score_table_csv:
        _write_score_table(args.score_table_csv, rows)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _write_score_table(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--unknown_tx_ids", default="")
    parser.add_argument("--train_known_roles", default="source")
    parser.add_argument("--proxy_unknown_roles", default="proxy_unknown")
    parser.add_argument("--known_query_roles", default="target_old")
    parser.add_argument("--unknown_query_roles", default="target_unknown")
    parser.add_argument("--train_known_correct_only", action="store_true")
    parser.add_argument("--source_incorrect_as_proxy", action="store_true")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "diag_mahalanobis"])
    parser.add_argument("--confidence_weight", type=float, default=0.0)
    parser.add_argument("--entropy_weight", type=float, default=0.0)
    parser.add_argument("--margin_weight", type=float, default=0.0)
    parser.add_argument("--threshold_policy", default="source_accept", choices=["source_accept", "proxy_far", "min_source_proxy", "max_source_proxy"])
    parser.add_argument("--source_accept_quantile", type=float, default=0.995)
    parser.add_argument("--proxy_far_quantile", type=float, default=0.05)
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--max_old_drop_pp", type=float, default=2.0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--score_table_csv", default="")
    return parser.parse_args(argv)


def main() -> int:
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
