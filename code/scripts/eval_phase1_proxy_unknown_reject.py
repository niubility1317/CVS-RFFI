#!/usr/bin/env python
"""Evaluate a Phase1-only proxy-unknown rejection head.

The detector is trained on frozen Phase1 features/logits using source old
samples as known and source-domain proxy unknown samples as non-old. Target
old and target unknown rows are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
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
        if "features" not in data.files:
            raise ValueError(f"{path} does not contain features")
        if "tx_logits" not in data.files:
            raise ValueError(f"{path} does not contain tx_logits; re-export features with updated exporter")
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
            "sat_scenarios": _as_str_array(pick("sat_scenarios", np.asarray([""] * n)), n),
            "channel_views": _as_str_array(pick("channel_views", np.asarray([""] * n)), n),
            "manifest": manifest,
        }


def _parse_roles(text: str) -> set[str]:
    return {str(x).strip() for x in str(text or "").split(",") if str(x).strip()}


def _class_tx_map(source_tx_ids: Sequence[str]) -> dict[int, str]:
    items = [canonical_tx_id(x) for x in source_tx_ids]
    return {i: item for i, item in enumerate(items)}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _feature_matrix(features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    top2 = torch.topk(logits, k=min(2, logits.size(1)), dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1] if top2.values.size(1) > 1 else torch.full_like(conf, 0.0)
    energy = -torch.logsumexp(logits, dim=1)
    scalars = torch.stack([conf, margin, energy], dim=1)
    return torch.cat([features.float(), logits.float(), scalars.float()], dim=1)


def _standardize(train_x: torch.Tensor, all_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (all_x - mean) / std, {
        "feature_dim": int(all_x.size(1)),
        "train_count": int(train_x.size(0)),
    }


def _train_linear_head(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    torch.manual_seed(int(seed))
    w = torch.zeros((x.size(1), 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=float(lr))
    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32)
    losses: list[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), y.float(), pos_weight=pos_weight)
        loss = loss + float(l2) * (w.pow(2).mean())
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    return w.detach(), b.detach(), losses


def _train_head_scores(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_all: torch.Tensor,
    *,
    head_type: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    head = str(head_type).lower()
    if head == "linear":
        w, b, losses = _train_linear_head(x_train, y_train, epochs=epochs, lr=lr, l2=l2, seed=seed)
        scores = torch.sigmoid((x_all @ w + b).reshape(-1)).detach().cpu().numpy()
        return scores, losses, {"head_type": "linear", "hidden_dim": 0}
    if head != "mlp":
        raise ValueError(f"unknown head_type={head_type!r}")
    torch.manual_seed(int(seed))
    model = nn.Sequential(
        nn.Linear(x_train.size(1), int(hidden_dim)),
        nn.ReLU(),
        nn.Linear(int(hidden_dim), 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(l2))
    pos = float((y_train > 0.5).sum().item())
    neg = float((y_train <= 0.5).sum().item())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], dtype=torch.float32)
    losses: list[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        logits = model(x_train).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train.float(), pos_weight=pos_weight)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    with torch.no_grad():
        scores = torch.sigmoid(model(x_all).reshape(-1)).detach().cpu().numpy()
    return scores, losses, {"head_type": "mlp", "hidden_dim": int(hidden_dim)}


def _threshold(
    source_scores: np.ndarray,
    proxy_scores: np.ndarray,
    *,
    policy: str,
    source_accept_quantile: float,
    proxy_far_quantile: float,
) -> tuple[float, dict[str, Any]]:
    source_t = float(np.quantile(source_scores, float(source_accept_quantile)))
    proxy_t = float(np.quantile(proxy_scores, float(proxy_far_quantile)))
    policy_text = str(policy)
    if policy_text == "source_accept":
        threshold = source_t
    elif policy_text == "proxy_far":
        threshold = proxy_t
    elif policy_text == "min_source_proxy":
        threshold = min(source_t, proxy_t)
    elif policy_text == "max_source_proxy":
        threshold = max(source_t, proxy_t)
    else:
        raise ValueError(f"unknown threshold_policy={policy!r}")
    source_accept = sum(1 for value in source_scores.tolist() if float(value) <= threshold) / max(
        1,
        int(source_scores.shape[0]),
    )
    proxy_accept = sum(1 for value in proxy_scores.tolist() if float(value) <= threshold) / max(
        1,
        int(proxy_scores.shape[0]),
    )
    return threshold, {
        "threshold_policy": policy_text,
        "unknown_score_threshold": float(threshold),
        "source_accept_quantile": float(source_accept_quantile),
        "proxy_far_quantile": float(proxy_far_quantile),
        "source_threshold": source_t,
        "proxy_threshold": proxy_t,
        "source_accept_rate_at_threshold": float(source_accept),
        "proxy_false_accept_rate_at_threshold": float(proxy_accept),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    features = payload["features"]
    logits = payload["tx_logits"]
    roles = payload["dataset_role"]
    tx_ids = payload["tx_ids"]
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

    train_known_mask = torch.tensor(
        [role in train_known_roles and canonical_tx_id(tx) in source_known for role, tx in zip(roles, tx_ids)],
        dtype=torch.bool,
    )
    proxy_mask = torch.tensor([role in proxy_roles for role in roles], dtype=torch.bool)
    if not bool(train_known_mask.any()):
        raise ValueError("no train-known source rows found")
    if not bool(proxy_mask.any()):
        raise ValueError("no proxy unknown rows found")

    raw_x = _feature_matrix(features, logits)
    train_mask = train_known_mask | proxy_mask
    x_all, scale_info = _standardize(raw_x[train_mask], raw_x)
    y_train = torch.where(proxy_mask[train_mask], torch.ones_like(proxy_mask[train_mask], dtype=torch.float32), torch.zeros_like(proxy_mask[train_mask], dtype=torch.float32))
    unknown_scores, losses, head_info = _train_head_scores(
        x_all[train_mask],
        y_train,
        x_all,
        head_type=str(getattr(args, "head_type", "linear")),
        hidden_dim=int(getattr(args, "hidden_dim", 64)),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
    )
    threshold, th_info = _threshold(
        unknown_scores[train_known_mask.numpy()],
        unknown_scores[proxy_mask.numpy()],
        policy=str(args.threshold_policy),
        source_accept_quantile=float(args.source_accept_quantile),
        proxy_far_quantile=float(args.proxy_far_quantile),
    )

    probs = F.softmax(logits, dim=1)
    _, pred = probs.max(dim=1)
    rows: list[dict[str, Any]] = []
    y_unknown: list[bool] = []
    reject_scores: list[float] = []
    accepted_flags: list[bool] = []
    known_total = known_closed_correct = known_accepted = known_correct_accepted = 0
    unknown_total = unknown_accepted = 0
    for i in range(logits.size(0)):
        role = str(roles[i])
        tx = canonical_tx_id(tx_ids[i])
        pred_class = int(pred[i].item())
        pred_tx = class_to_tx.get(pred_class, str(pred_class))
        score = float(unknown_scores[i])
        accepted = score <= threshold
        is_known_query = role in known_roles and tx in source_known
        is_unknown_query = role in unknown_roles and (not explicit_unknown or tx in explicit_unknown)
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
                "row": i,
                "role": role,
                "tx_id": tx,
                "rx_id": payload["rx_ids"][i],
                "day_id": payload["day_ids"][i],
                "channel_view": payload["channel_views"][i],
                "sat_scenario": payload["sat_scenarios"][i],
                "is_train_known": int(bool(train_known_mask[i].item())),
                "is_proxy_unknown": int(bool(proxy_mask[i].item())),
                "is_known_query": int(is_known_query),
                "is_unknown_query": int(is_unknown_query),
                "pred_class": pred_class,
                "pred_tx_id": pred_tx,
                "accepted": int(accepted),
                "closed_correct_known": int(closed_correct),
                "accepted_correct_known": int(bool(accepted and closed_correct)),
                "unknown_score": f"{score:.8f}",
            }
        )

    known_closed_accuracy = _safe_rate(known_closed_correct, known_total)
    known_full_accuracy = _safe_rate(known_correct_accepted, known_total)
    unknown_far = _safe_rate(unknown_accepted, unknown_total)
    old_drop_pp = None
    if known_closed_accuracy is not None and known_full_accuracy is not None:
        old_drop_pp = 100.0 * (float(known_closed_accuracy) - float(known_full_accuracy))
    metrics = {
        "phase": "phase1_only_proxy_unknown_reject",
        "threshold_scope": "source_old_and_source_proxy_unknown_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(args.feature_npz),
        "source_tx_ids": source_tx_ids,
        "known_query_roles": sorted(known_roles),
        "unknown_query_roles": sorted(unknown_roles),
        "proxy_unknown_roles": sorted(proxy_roles),
        "target_unknown_tx_ids": sorted(explicit_unknown),
        "train_known_count": int(train_known_mask.sum().item()),
        "proxy_unknown_count": int(proxy_mask.sum().item()),
        "training": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "seed": int(args.seed),
            "loss_start": float(losses[0]) if losses else None,
            "loss_end": float(losses[-1]) if losses else None,
            **head_info,
            **scale_info,
        },
        "threshold": th_info,
        "known_query_count": int(known_total),
        "known_closed_accuracy_no_reject": known_closed_accuracy,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_full_accuracy_after_reject": known_full_accuracy,
        "known_accepted_accuracy": _safe_rate(known_correct_accepted, known_accepted),
        "old_retention_vs_closed": None
        if known_closed_correct <= 0
        else float(known_correct_accepted) / float(known_closed_correct),
        "old_drop_pp_vs_closed": old_drop_pp,
        "max_old_drop_pp": float(args.max_old_drop_pp),
        "unknown_query_count": int(unknown_total),
        "unknown_FAR": unknown_far,
        "unknown_reject_rate": None if unknown_total == 0 else 1.0 - float(unknown_accepted) / float(unknown_total),
        "unknown_far_target": float(args.unknown_far_target),
        "passes_unknown_far_target": None
        if unknown_far is None
        else float(unknown_far) <= float(args.unknown_far_target),
        "passes_old_drop_target": None
        if old_drop_pp is None
        else float(old_drop_pp) <= float(args.max_old_drop_pp),
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
    parser.add_argument("--threshold_policy", default="source_accept", choices=["source_accept", "proxy_far", "min_source_proxy", "max_source_proxy"])
    parser.add_argument("--source_accept_quantile", type=float, default=0.995)
    parser.add_argument("--proxy_far_quantile", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--head_type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4070203)
    parser.add_argument("--unknown_far_target", type=float, default=0.05)
    parser.add_argument("--max_old_drop_pp", type=float, default=3.0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--score_table_csv", default="")
    return parser.parse_args(argv)


def main() -> int:
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
