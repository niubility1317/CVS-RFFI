#!/usr/bin/env python
"""Evaluate Phase1-only kNN density rejection on frozen features."""

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


def _group_payload(payload: Mapping[str, Any], *, feature_reduce: str) -> list[dict[str, Any]]:
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
    reduce = str(feature_reduce).lower()
    for key, idx in sorted(groups.items()):
        idx_t = torch.tensor(idx, dtype=torch.long)
        z = features[idx_t].float()
        lo = logits[idx_t].float()
        probs = F.softmax(lo, dim=1)
        mean_logits = lo.mean(dim=0)
        pred_class = int(mean_logits.argmax().item())
        conf, pred = probs.max(dim=1)
        if reduce == "mean":
            feat = z.mean(dim=0)
        elif reduce == "best_conf":
            feat = z[int(conf.argmax().item())]
        elif reduce == "median":
            feat = z.median(dim=0).values
        else:
            raise ValueError(f"unknown feature_reduce={feature_reduce!r}")
        out.append(
            {
                "role": key[0],
                "tx_id": key[1],
                "rx_id": key[2],
                "day_id": key[3],
                "eq_id": key[4],
                "sig_id": key[5],
                "view_count": len(idx),
                "features": feat,
                "pred_class": pred_class,
                "mean_confidence": float(conf.mean().item()),
                "min_confidence": float(conf.min().item()),
                "vote_frac": float((pred == pred_class).float().mean().item()),
            }
        )
    return out


def _standardize(train_x: torch.Tensor, all_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (all_x - mean) / std, {"feature_dim": int(all_x.size(1)), "train_count": int(train_x.size(0))}


def _knn_scores(
    x: torch.Tensor,
    groups: Sequence[Mapping[str, Any]],
    train_known_mask: torch.Tensor,
    class_to_tx: Mapping[int, str],
    *,
    k: int,
    distance: str,
    exclude_self: bool,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    metric = str(distance).lower()
    train_by_tx: dict[str, torch.Tensor] = {}
    for tx in sorted({canonical_tx_id(g["tx_id"]) for i, g in enumerate(groups) if bool(train_known_mask[i])}):
        idx = torch.tensor([i for i, g in enumerate(groups) if bool(train_known_mask[i]) and canonical_tx_id(g["tx_id"]) == tx], dtype=torch.long)
        train_by_tx[tx] = x[idx]
    fallback = x[train_known_mask]
    scores: list[float] = []
    pred_tx_ids: list[str] = []
    for i, g in enumerate(groups):
        pred_tx = class_to_tx.get(int(g["pred_class"]), str(g["pred_class"]))
        bank = train_by_tx.get(pred_tx, fallback)
        query = x[i].unsqueeze(0)
        if metric == "cosine":
            d = 1.0 - F.cosine_similarity(query, bank, dim=1)
        elif metric == "euclidean":
            d = torch.norm(bank - query, dim=1, p=2)
        else:
            raise ValueError(f"unknown distance={distance!r}")
        if bool(exclude_self) and bool(train_known_mask[i]) and canonical_tx_id(g["tx_id"]) == pred_tx and d.numel() > 1:
            d = d.masked_fill(d <= 1e-8, float("inf"))
        kk = max(1, min(int(k), int(d.numel())))
        score = float(torch.topk(d, k=kk, largest=False).values.mean().item())
        scores.append(score)
        pred_tx_ids.append(pred_tx)
    return np.asarray(scores, dtype=np.float64), pred_tx_ids, {
        "distance": metric,
        "knn_k": int(k),
        "exclude_self": bool(exclude_self),
        "class_count": len(train_by_tx),
    }


def _quantile(values: Sequence[float], q: float, fallback: float) -> float:
    if not values:
        return float(fallback)
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _thresholds(
    scores: np.ndarray,
    pred_tx_ids: Sequence[str],
    train_known_mask: torch.Tensor,
    proxy_mask: torch.Tensor,
    *,
    policy: str,
    source_q: float,
    proxy_q: float,
    class_conditional: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    source_scores = [float(scores[i]) for i in range(len(scores)) if bool(train_known_mask[i])]
    proxy_scores = [float(scores[i]) for i in range(len(scores)) if bool(proxy_mask[i])]
    global_source_t = _quantile(source_scores, source_q, float(np.max(scores)))
    global_proxy_t = _quantile(proxy_scores, proxy_q, global_source_t)
    tx_ids = sorted(set(pred_tx_ids)) if bool(class_conditional) else ["__global__"]
    thresholds: dict[str, float] = {}
    for tx in tx_ids:
        if tx == "__global__":
            src_t = global_source_t
            proxy_t = global_proxy_t
        else:
            src_vals = [float(scores[i]) for i, pred_tx in enumerate(pred_tx_ids) if pred_tx == tx and bool(train_known_mask[i])]
            proxy_vals = [float(scores[i]) for i, pred_tx in enumerate(pred_tx_ids) if pred_tx == tx and bool(proxy_mask[i])]
            src_t = _quantile(src_vals, source_q, global_source_t)
            proxy_t = _quantile(proxy_vals, proxy_q, global_proxy_t)
        if policy == "source_accept":
            threshold = src_t
        elif policy == "proxy_far":
            threshold = proxy_t
        elif policy == "min_source_proxy":
            threshold = min(src_t, proxy_t)
        elif policy == "max_source_proxy":
            threshold = max(src_t, proxy_t)
        else:
            raise ValueError(f"unknown threshold_policy={policy!r}")
        thresholds[tx] = float(threshold)

    def pick(pred_tx: str) -> float:
        return thresholds[pred_tx] if bool(class_conditional) and pred_tx in thresholds else thresholds["__global__"]

    def accept_rate(mask: torch.Tensor) -> float:
        total = 0
        acc = 0
        for i, pred_tx in enumerate(pred_tx_ids):
            if bool(mask[i]):
                total += 1
                acc += int(float(scores[i]) <= pick(pred_tx))
        return 0.0 if total <= 0 else float(acc) / float(total)

    return thresholds, {
        "threshold_policy": policy,
        "threshold_scope": "per_predicted_old_class" if bool(class_conditional) else "global",
        "source_accept_quantile": float(source_q),
        "proxy_far_quantile": float(proxy_q),
        "global_source_threshold": global_source_t,
        "global_proxy_threshold": global_proxy_t,
        "source_accept_rate_at_threshold": accept_rate(train_known_mask),
        "proxy_false_accept_rate_at_threshold": accept_rate(proxy_mask),
        "class_thresholds": thresholds,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    groups = _group_payload(payload, feature_reduce=str(args.feature_reduce))
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    class_to_tx = _class_tx_map(source_tx_ids)
    source_known = {canonical_tx_id(x) for x in source_tx_ids}
    train_known_roles = _parse_roles(args.train_known_roles)
    proxy_roles = _parse_roles(args.proxy_unknown_roles)
    known_roles = _parse_roles(args.known_query_roles)
    unknown_roles = _parse_roles(args.unknown_query_roles)
    explicit_unknown = {canonical_tx_id(x) for x in parse_tx_id_list(args.unknown_tx_ids)}

    x_raw = torch.stack([g["features"] for g in groups], dim=0)
    source_train_base_mask = torch.tensor([g["role"] in train_known_roles and canonical_tx_id(g["tx_id"]) in source_known for g in groups], dtype=torch.bool)
    source_closed_correct_mask = torch.tensor(
        [class_to_tx.get(int(g["pred_class"]), str(g["pred_class"])) == canonical_tx_id(g["tx_id"]) for g in groups],
        dtype=torch.bool,
    )
    train_known_mask = source_train_base_mask & (source_closed_correct_mask if bool(args.train_known_correct_only) else torch.ones_like(source_train_base_mask, dtype=torch.bool))
    proxy_mask = torch.tensor([g["role"] in proxy_roles for g in groups], dtype=torch.bool)
    if bool(args.source_incorrect_as_proxy):
        proxy_mask = proxy_mask | (source_train_base_mask & ~source_closed_correct_mask)
    if not bool(train_known_mask.any()):
        raise ValueError("no train-known source groups found")
    if not bool(proxy_mask.any()):
        raise ValueError("no proxy unknown groups found")

    x_all, scale_info = _standardize(x_raw[train_known_mask], x_raw)
    scores, pred_tx_ids, score_info = _knn_scores(
        x_all,
        groups,
        train_known_mask,
        class_to_tx,
        k=int(args.knn_k),
        distance=str(args.distance),
        exclude_self=bool(args.exclude_self),
    )
    thresholds, th_info = _thresholds(
        scores,
        pred_tx_ids,
        train_known_mask,
        proxy_mask,
        policy=str(args.threshold_policy),
        source_q=float(args.source_accept_quantile),
        proxy_q=float(args.proxy_far_quantile),
        class_conditional=bool(args.class_conditional_threshold),
    )

    def pick_threshold(pred_tx: str) -> float:
        if bool(args.class_conditional_threshold) and pred_tx in thresholds:
            return thresholds[pred_tx]
        return thresholds["__global__"]

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
        threshold = pick_threshold(pred_tx)
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
            }
        )

    known_closed_accuracy = _safe_rate(known_closed_correct, known_total)
    known_full_accuracy = _safe_rate(known_correct_accepted, known_total)
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    old_drop_pp = None if known_closed_accuracy is None or known_full_accuracy is None else 100.0 * (float(known_closed_accuracy) - float(known_full_accuracy))
    metrics = {
        "phase": "phase1_only_knn_density_reject",
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
        "scoring": {**score_info, **scale_info, "feature_reduce": str(args.feature_reduce)},
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
        metrics.update(binary_reject_metrics(torch.tensor(y_unknown, dtype=torch.bool), torch.tensor(reject_scores, dtype=torch.float32), torch.tensor(accepted_flags, dtype=torch.bool)))
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
    parser.add_argument("--feature_reduce", default="mean", choices=["mean", "best_conf", "median"])
    parser.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--exclude_self", action="store_true")
    parser.add_argument("--class_conditional_threshold", action="store_true")
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
