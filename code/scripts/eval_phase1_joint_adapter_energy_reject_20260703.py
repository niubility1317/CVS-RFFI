#!/usr/bin/env python
"""Train a source-only joint LEO repair adapter and open-set oldness gate.

This diagnostic keeps the Phase1 backbone frozen. Training uses only:
- source clean/LEO paired old-class features;
- source-receiver proxy_unknown LEO features for an open-set oldness loss.

Target receiver old/unknown rows are used only for final sat-only evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KEY_FIELDS = ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x.strip()) for x in str(text).split(",") if x.strip()]


def _as_str(data, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.asarray([""] * n, dtype=str)
    arr = np.asarray(data[key])
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch")
    return arr.astype(str)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        logits = np.asarray(data["tx_logits"], dtype=np.float32) if "tx_logits" in data.files else None
        n = int(features.shape[0])
        manifest = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        return {
            "path": str(path),
            "features": features,
            "tx_logits": logits,
            "dataset_role": _as_str(data, "dataset_role", n),
            "tx_ids": _as_str(data, "tx_ids", n),
            "rx_ids": _as_str(data, "rx_ids", n),
            "day_ids": _as_str(data, "day_ids", n),
            "eq_ids": _as_str(data, "eq_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
            "sat_scenarios": _as_str(data, "sat_scenarios", n),
            "manifest": manifest,
        }


def _row_key(payload: dict, i: int) -> tuple[str, ...]:
    return tuple(str(payload[field][i]) for field in KEY_FIELDS)


def _source_pair_indices(clean: dict, sat: dict, source_roles: set[str]) -> list[tuple[int, int]]:
    clean_map: dict[tuple[str, ...], int] = {}
    for i, role in enumerate(clean["dataset_role"]):
        if str(role) in source_roles:
            clean_map.setdefault(_row_key(clean, i), int(i))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[str, ...]] = set()
    for i, role in enumerate(sat["dataset_role"]):
        if str(role) not in source_roles:
            continue
        key = _row_key(sat, i)
        if key in seen:
            continue
        seen.add(key)
        if key in clean_map:
            pairs.append((int(clean_map[key]), int(i)))
    return pairs


def _make_pairs(clean: dict, sats: list[dict], source_roles: set[str], source_tx_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, ...]], dict[str, int]]:
    clean_parts = []
    sat_parts = []
    label_text: list[str] = []
    keys: list[tuple[str, ...]] = []
    counts: dict[str, int] = {}
    allowed = set(source_tx_ids)
    for sat_i, sat in enumerate(sats):
        pairs = _source_pair_indices(clean, sat, source_roles)
        counts[str(sat["path"])] = len(pairs)
        for ci, si in pairs:
            tx = canonical_tx_id(sat["tx_ids"][si])
            if tx not in allowed:
                continue
            clean_parts.append(clean["features"][ci])
            sat_parts.append(sat["features"][si])
            label_text.append(tx)
            keys.append((f"sat{sat_i}", *_row_key(sat, si)))
    if not clean_parts:
        raise ValueError("no source clean/LEO pairs found")
    label_map = {tx: i for i, tx in enumerate(source_tx_ids)}
    y = np.asarray([label_map[tx] for tx in label_text], dtype=np.int64)
    return (
        np.asarray(clean_parts, dtype=np.float32),
        np.asarray(sat_parts, dtype=np.float32),
        y,
        keys,
        counts,
    )


def _stable_split(keys: list[tuple[str, ...]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import hashlib

    scores = []
    for key in keys:
        digest = hashlib.sha256(("|".join([str(seed), *key])).encode("utf-8")).digest()
        scores.append(int.from_bytes(digest[:8], "little") / float(2**64 - 1))
    arr = np.asarray(scores, dtype=np.float64)
    val = arr < float(val_fraction)
    if val.all() or (~val).all():
        order = np.argsort(arr)
        n_val = max(1, min(len(order) - 1, int(round(len(order) * float(val_fraction)))))
        val = np.zeros(len(order), dtype=bool)
        val[order[:n_val]] = True
    return np.where(~val)[0], np.where(val)[0]


def _sample_proxy(proxy_npzs: list[Path], max_proxy: int, seed: int) -> np.ndarray:
    parts = []
    for path in proxy_npzs:
        payload = _load_npz(path)
        role = payload["dataset_role"].astype(str)
        idx = np.where(role == "proxy_unknown")[0]
        if idx.size:
            parts.append(payload["features"][idx])
    if not parts:
        raise ValueError("no proxy_unknown rows found in proxy_npz")
    x = np.concatenate(parts, axis=0).astype(np.float32)
    if max_proxy > 0 and x.shape[0] > max_proxy:
        rng = np.random.default_rng(seed)
        keep = rng.choice(x.shape[0], size=int(max_proxy), replace=False)
        x = x[np.sort(keep)]
    return x


def _make_clean_prototypes(clean_x: np.ndarray, labels: np.ndarray, classes: int, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(clean_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    protos = []
    for c in range(classes):
        idx = torch.where(y == c)[0]
        if idx.numel() == 0:
            raise ValueError(f"missing clean prototype class {c}")
        protos.append(x.index_select(0, idx).mean(dim=0))
    return torch.stack(protos, dim=0)


def _proto_logits(x: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.normalize(x.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).t() / max(float(temperature), 1e-6)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, kind: str, hidden: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.kind = kind
        self.alpha = float(alpha)
        self.norm = nn.LayerNorm(dim)
        if kind == "linear":
            self.net = nn.Linear(dim, dim)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(dim, int(hidden)), nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden), dim))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.net(self.norm(x))


class OldnessHead(nn.Module):
    def __init__(self, dim: int, kind: str, hidden: int, dropout: float) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(hidden)), nn.ReLU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden), 1))
        else:
            raise ValueError(kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


@torch.no_grad()
def _alignment(adapter: nn.Module, sat_x: torch.Tensor, clean_x: torch.Tensor, y: torch.Tensor, prototypes: torch.Tensor, temp: float) -> dict[str, float]:
    before = _proto_logits(sat_x, prototypes, temp).argmax(dim=1)
    z = adapter(sat_x)
    after_logits = _proto_logits(z, prototypes, temp)
    after = after_logits.argmax(dim=1)
    return {
        "pair_mse_before": float(F.mse_loss(sat_x, clean_x).item()),
        "pair_mse_after": float(F.mse_loss(z, clean_x).item()),
        "pair_cos_before": float(F.cosine_similarity(sat_x, clean_x, dim=1).mean().item()),
        "pair_cos_after": float(F.cosine_similarity(z, clean_x, dim=1).mean().item()),
        "proto_acc_before": float((before == y).float().mean().item()),
        "proto_acc_after": float((after == y).float().mean().item()),
        "mean_residual_norm": float((z - sat_x).norm(dim=1).mean().item()),
    }


def _train_model(args: argparse.Namespace, clean_x_np: np.ndarray, sat_x_np: np.ndarray, y_np: np.ndarray, keys: list[tuple[str, ...]], proxy_x_np: np.ndarray, source_tx_ids: list[str]) -> tuple[nn.Module, nn.Module, torch.Tensor, dict]:
    device = torch.device(str(args.device) if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    clean_x = torch.as_tensor(clean_x_np, dtype=torch.float32, device=device)
    sat_x = torch.as_tensor(sat_x_np, dtype=torch.float32, device=device)
    proxy_x = torch.as_tensor(proxy_x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    prototypes = _make_clean_prototypes(clean_x_np, y_np, len(source_tx_ids), device=device)
    train_idx_np, val_idx_np = _stable_split(keys, float(args.val_fraction), int(args.seed))
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)
    adapter = ResidualAdapter(sat_x.shape[1], str(args.adapter_kind), int(args.hidden_dim), float(args.alpha), float(args.dropout)).to(device)
    head = OldnessHead(sat_x.shape[1], str(args.head_kind), int(args.head_hidden_dim), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=float(args.lr), weight_decay=float(args.weight_decay))
    bce_pos_weight = torch.tensor([float(args.oldness_pos_weight)], dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(args.seed))
    losses = []
    for epoch in range(int(args.epochs)):
        adapter.train()
        head.train()
        order = train_idx[torch.randperm(train_idx.numel(), device=device)]
        epoch_loss = 0.0
        steps = 0
        for start in range(0, int(order.numel()), int(args.batch_size)):
            idx = order[start : start + int(args.batch_size)]
            if idx.numel() == 0:
                continue
            pidx_np = rng.integers(0, proxy_x.shape[0], size=int(min(idx.numel() * int(args.proxy_batch_mult), proxy_x.shape[0])))
            pidx = torch.as_tensor(pidx_np, dtype=torch.long, device=device)
            z_sat = sat_x.index_select(0, idx)
            z_clean = clean_x.index_select(0, idx)
            cls = y.index_select(0, idx)
            z_old = adapter(z_sat)
            z_proxy = adapter(proxy_x.index_select(0, pidx))
            logits_old = _proto_logits(z_old, prototypes.detach(), float(args.proto_temperature))
            logits_proxy = _proto_logits(z_proxy, prototypes.detach(), float(args.proto_temperature))
            true_logit = logits_old.gather(1, cls.view(-1, 1)).reshape(-1)
            proxy_max = logits_proxy.max(dim=1).values
            pair_loss = F.smooth_l1_loss(z_old, z_clean)
            cos_loss = (1.0 - F.cosine_similarity(z_old, z_clean, dim=1)).mean()
            ce_loss = F.cross_entropy(logits_old, cls)
            residual_loss = ((z_old - z_sat) ** 2).mean()
            oldness_logits = torch.cat([head(z_old), head(z_proxy)], dim=0)
            oldness_y = torch.cat([torch.ones(z_old.shape[0], device=device), torch.zeros(z_proxy.shape[0], device=device)], dim=0)
            oldness_loss = F.binary_cross_entropy_with_logits(oldness_logits, oldness_y, pos_weight=bce_pos_weight)
            proxy_loss = F.softplus(proxy_max - float(args.proxy_logit_cap)).mean()
            old_conf_loss = F.softplus(float(args.old_logit_floor) - true_logit).mean()
            loss = (
                float(args.pair_weight) * pair_loss
                + float(args.cos_weight) * cos_loss
                + float(args.proto_ce_weight) * ce_loss
                + float(args.residual_weight) * residual_loss
                + float(args.oldness_weight) * oldness_loss
                + float(args.proxy_proto_weight) * proxy_loss
                + float(args.old_conf_weight) * old_conf_loss
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(head.parameters()), float(args.grad_clip))
            opt.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
        if steps:
            losses.append(epoch_loss / steps)
    adapter.eval()
    head.eval()
    metrics = {
        "train_alignment": _alignment(adapter, sat_x.index_select(0, train_idx), clean_x.index_select(0, train_idx), y.index_select(0, train_idx), prototypes, float(args.proto_temperature)),
        "val_alignment": _alignment(adapter, sat_x.index_select(0, val_idx), clean_x.index_select(0, val_idx), y.index_select(0, val_idx), prototypes, float(args.proto_temperature)),
        "final_train_loss": losses[-1] if losses else None,
        "epochs": int(args.epochs),
        "train_pair_count": int(train_idx.numel()),
        "val_pair_count": int(val_idx.numel()),
        "proxy_train_count": int(proxy_x.shape[0]),
    }
    return adapter, head, prototypes.detach(), metrics


@torch.no_grad()
def _score_payload(payload: dict, adapter: nn.Module, head: nn.Module, prototypes: torch.Tensor, args: argparse.Namespace, source_tx_ids: list[str]) -> tuple[list[dict], dict[str, np.ndarray]]:
    device = next(adapter.parameters()).device
    x = torch.as_tensor(payload["features"], dtype=torch.float32, device=device)
    z = adapter(x)
    proto_logits_t = _proto_logits(z, prototypes, float(args.proto_temperature))
    old_prob_t = torch.sigmoid(head(z))
    proto_logits = np.asarray(proto_logits_t.detach().cpu().tolist(), dtype=np.float32)
    old_prob = np.asarray(old_prob_t.detach().cpu().tolist(), dtype=np.float32)
    grouped: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for i in range(payload["features"].shape[0]):
        grouped[(
            str(payload["dataset_role"][i]),
            canonical_tx_id(payload["tx_ids"][i]),
            str(payload["rx_ids"][i]),
            str(payload["day_ids"][i]),
            str(payload["sig_ids"][i]),
        )].append(i)
    class_to_tx = {i: tx for i, tx in enumerate(source_tx_ids)}
    rows = []
    score_parts = {"old_prob": [], "proto_max": [], "proto_margin": [], "fused_rank": []}
    for key, idx in sorted(grouped.items()):
        role, tx, rx, day, sig = key
        lo = proto_logits[idx].mean(axis=0)
        op = float(old_prob[idx].mean())
        pred = int(lo.argmax())
        pred_tx = class_to_tx.get(pred, str(pred))
        sorted_lo = np.sort(lo)
        margin = float(sorted_lo[-1] - sorted_lo[-2]) if lo.size >= 2 else float(sorted_lo[-1])
        proto_max = float(lo.max())
        rows.append({
            "role": role,
            "tx_id": tx,
            "rx_id": rx,
            "day_id": day,
            "sig_id": sig,
            "pred_tx_id": pred_tx,
            "is_known_query": role == "target_old" and tx in set(source_tx_ids),
            "is_unknown_query": role == "target_unknown",
            "closed_correct_known": role == "target_old" and tx == pred_tx,
            "old_prob": op,
            "proto_max": proto_max,
            "proto_margin": margin,
        })
        score_parts["old_prob"].append(op)
        score_parts["proto_max"].append(proto_max)
        score_parts["proto_margin"].append(margin)
    for key in score_parts:
        if key == "fused_rank":
            continue
        score_parts[key] = np.asarray(score_parts[key], dtype=np.float64)
    # Rank-normalized fusion is fit per payload from source/proxy rows only.
    source_mask = np.asarray([r["role"] == "source" for r in rows], dtype=bool)
    proxy_mask = np.asarray([r["role"] == "proxy_unknown" for r in rows], dtype=bool)
    fused = np.zeros(len(rows), dtype=np.float64)
    for key, weight in [("old_prob", 0.50), ("proto_max", 0.30), ("proto_margin", 0.20)]:
        vals = np.asarray(score_parts[key], dtype=np.float64)
        fit = vals[source_mask | proxy_mask]
        lo = float(np.quantile(fit, 0.01)) if fit.size else float(vals.min())
        hi = float(np.quantile(fit, 0.99)) if fit.size else float(vals.max())
        scaled = (vals - lo) / max(hi - lo, 1e-6)
        fused += weight * np.clip(scaled, 0.0, 1.0)
    score_parts["fused_rank"] = fused
    for i, value in enumerate(fused):
        rows[i]["fused_rank"] = float(value)
    return rows, score_parts


def _safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def _metrics(accept: np.ndarray, known: np.ndarray, unknown: np.ndarray, closed: np.ndarray) -> dict:
    known_total = int(known.sum())
    unknown_total = int(unknown.sum())
    known_closed = int((known & closed).sum())
    known_correct_after = int((known & closed & accept).sum())
    known_accepted = int((known & accept).sum())
    unknown_accepted = int((unknown & accept).sum())
    closed_acc = _safe_rate(known_closed, known_total)
    full_acc = _safe_rate(known_correct_after, known_total)
    far = _safe_rate(unknown_accepted, unknown_total)
    old_drop = 100.0 * (closed_acc - full_acc)
    return {
        "unknown_FAR": far,
        "known_closed_accuracy_no_reject": closed_acc,
        "known_full_accuracy_after_reject": full_acc,
        "old_drop_pp_vs_closed": old_drop,
        "known_coverage": _safe_rate(known_accepted, known_total),
        "known_accepted_accuracy": _safe_rate(known_correct_after, known_accepted),
        "known_query_count": known_total,
        "unknown_query_count": unknown_total,
        "passes_unknown_far_target": bool(far <= 0.05),
        "passes_old_drop_target": bool(old_drop <= 2.0),
        "passes_dual_target": bool(far <= 0.05 and old_drop <= 2.0),
    }


def _evaluate_run(run_dir: Path, adapter: nn.Module, head: nn.Module, prototypes: torch.Tensor, args: argparse.Namespace, source_tx_ids: list[str]) -> list[dict]:
    npz = run_dir / str(args.sat_npz_relpath)
    payload = _load_npz(npz)
    rows, scores = _score_payload(payload, adapter, head, prototypes, args, source_tx_ids)
    source = np.asarray([r["role"] == "source" for r in rows], dtype=bool)
    proxy = np.asarray([r["role"] == "proxy_unknown" for r in rows], dtype=bool)
    known = np.asarray([bool(r["is_known_query"]) for r in rows], dtype=bool)
    unknown = np.asarray([bool(r["is_unknown_query"]) for r in rows], dtype=bool)
    closed = np.asarray([bool(r["closed_correct_known"]) for r in rows], dtype=bool)
    pred_tx = np.asarray([str(r["pred_tx_id"]) for r in rows])
    target_rxs = sorted({r["rx_id"] for r in rows if r["role"] in {"target_old", "target_unknown"}})
    out = []
    for score_name in ["old_prob", "proto_max", "proto_margin", "fused_rank"]:
        score = np.asarray(scores[score_name], dtype=np.float64)
        for policy in [
            "source_accept",
            "proxy_far",
            "max_source_proxy",
            "mean_source_proxy",
            "class_source_accept",
            "class_proxy_far",
            "class_max_source_proxy",
            "class_mean_source_proxy",
        ]:
            for source_q in [0.001, 0.005, 0.010, 0.020]:
                proxy_qs = [0.95] if policy == "source_accept" else [0.90, 0.95, 0.97, 0.99]
                for proxy_q in proxy_qs:
                    source_t = float(np.quantile(score[source], source_q))
                    proxy_t = float(np.quantile(score[proxy], proxy_q))
                    if policy == "source_accept":
                        threshold = source_t
                    elif policy == "proxy_far":
                        threshold = proxy_t
                    elif policy == "max_source_proxy":
                        threshold = max(source_t, proxy_t)
                    elif policy == "mean_source_proxy":
                        threshold = 0.5 * (source_t + proxy_t)
                    else:
                        class_thresholds = {}
                        for tx in source_tx_ids:
                            sm = source & (pred_tx == tx)
                            pm = proxy & (pred_tx == tx)
                            st = float(np.quantile(score[sm], source_q)) if sm.any() else source_t
                            pt = float(np.quantile(score[pm], proxy_q)) if pm.any() else proxy_t
                            if policy == "class_source_accept":
                                class_thresholds[tx] = st
                            elif policy == "class_proxy_far":
                                class_thresholds[tx] = pt
                            elif policy == "class_max_source_proxy":
                                class_thresholds[tx] = max(st, pt)
                            elif policy == "class_mean_source_proxy":
                                class_thresholds[tx] = 0.5 * (st + pt)
                            else:
                                raise ValueError(policy)
                        threshold = float(np.mean(list(class_thresholds.values())))
                    if policy.startswith("class_"):
                        th_vec = np.asarray([class_thresholds.get(str(tx), threshold) for tx in pred_tx], dtype=np.float64)
                        accept = score >= th_vec
                    else:
                        accept = score >= threshold
                    m = _metrics(accept, known, unknown, closed)
                    m.update({
                        "run_id": run_dir.name,
                        "run_tag": str(args.run_tag),
                        "target_rx": ",".join(target_rxs),
                        "mode": "joint_adapter_oldness_gate",
                        "score_name": score_name,
                        "threshold_policy": policy,
                        "source_accept_quantile": source_q,
                        "proxy_far_quantile": proxy_q,
                        "threshold": threshold,
                        "source_threshold": source_t,
                        "proxy_threshold": proxy_t,
                        "source_train_groups": int(source.sum()),
                        "proxy_train_groups": int(proxy.sum()),
                    })
                    out.append(m)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--clean_npz", type=Path, required=True)
    p.add_argument("--train_sat_npz", type=Path, action="append", required=True)
    p.add_argument("--proxy_npz", type=Path, action="append", required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--metrics_json", type=Path, default=None)
    p.add_argument("--run_tag", default="V9")
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--sat_npz_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--source_roles", default="source")
    p.add_argument("--adapter_kind", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--head_kind", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--head_hidden_dim", type=int, default=96)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--proxy_batch_mult", type=int, default=1)
    p.add_argument("--max_proxy_samples", type=int, default=9000)
    p.add_argument("--lr", type=float, default=0.0006)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--pair_weight", type=float, default=0.55)
    p.add_argument("--cos_weight", type=float, default=2.0)
    p.add_argument("--proto_ce_weight", type=float, default=1.3)
    p.add_argument("--residual_weight", type=float, default=0.08)
    p.add_argument("--oldness_weight", type=float, default=0.55)
    p.add_argument("--oldness_pos_weight", type=float, default=1.5)
    p.add_argument("--proxy_proto_weight", type=float, default=0.18)
    p.add_argument("--proxy_logit_cap", type=float, default=4.5)
    p.add_argument("--old_conf_weight", type=float, default=0.12)
    p.add_argument("--old_logit_floor", type=float, default=7.5)
    p.add_argument("--proto_temperature", type=float, default=0.07)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=4070351)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    source_roles = {x.strip() for x in str(args.source_roles).split(",") if x.strip()}
    clean = _load_npz(args.clean_npz)
    sats = [_load_npz(p) for p in args.train_sat_npz]
    clean_x, sat_x, y, keys, pair_counts = _make_pairs(clean, sats, source_roles, source_tx_ids)
    proxy_x = _sample_proxy(args.proxy_npz, int(args.max_proxy_samples), int(args.seed))
    adapter, head, prototypes, train_metrics = _train_model(args, clean_x, sat_x, y, keys, proxy_x, source_tx_ids)
    rows = []
    for run_dir in sorted(args.runs_root.glob(args.run_glob)):
        if (run_dir / str(args.sat_npz_relpath)).is_file():
            rows.extend(_evaluate_run(run_dir, adapter, head, prototypes, args, source_tx_ids))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["run_id"]
    leading = ["run_id", "target_rx", "mode", "score_name", "threshold_policy"]
    fields = leading + [f for f in fields if f not in leading]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "phase": "phase1_source_only_joint_adapter_oldness_gate_v9",
        "run_tag": str(args.run_tag),
        "rows": len(rows),
        "dual_pass": int(sum(1 for r in rows if bool(r.get("passes_dual_target")))),
        "out_csv": str(args.out_csv),
        "source_pair_count": int(sat_x.shape[0]),
        "source_pair_count_by_train_npz": pair_counts,
        "proxy_sample_count": int(proxy_x.shape[0]),
        "uses_target_clean": False,
        "uses_target_labels_for_training": False,
        "uses_unknown_query_for_threshold": False,
        "config": {
            "adapter_kind": str(args.adapter_kind),
            "head_kind": str(args.head_kind),
            "hidden_dim": int(args.hidden_dim),
            "head_hidden_dim": int(args.head_hidden_dim),
            "alpha": float(args.alpha),
            "epochs": int(args.epochs),
            "pair_weight": float(args.pair_weight),
            "cos_weight": float(args.cos_weight),
            "proto_ce_weight": float(args.proto_ce_weight),
            "residual_weight": float(args.residual_weight),
            "oldness_weight": float(args.oldness_weight),
            "oldness_pos_weight": float(args.oldness_pos_weight),
            "proxy_proto_weight": float(args.proxy_proto_weight),
            "proxy_logit_cap": float(args.proxy_logit_cap),
            "old_conf_weight": float(args.old_conf_weight),
            "old_logit_floor": float(args.old_logit_floor),
            "proto_temperature": float(args.proto_temperature),
            "seed": int(args.seed),
        },
        "train_metrics": train_metrics,
    }
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
