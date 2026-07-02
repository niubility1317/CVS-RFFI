#!/usr/bin/env python
"""Fit a source-only LEO feature repair adapter and apply it to a sat NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402


KEY_FIELDS = ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")


def _as_str_array(value: np.ndarray, n: int) -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [canonical_tx_id(arr.item())] * int(n)
    return [canonical_tx_id(v) for v in arr.reshape(-1).tolist()]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{path} must contain features")
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
        return {
            "arrays": {key: np.asarray(data[key]) for key in data.files if key != "manifest_json"},
            "features": features,
            "tx_logits": np.asarray(data["tx_logits"], dtype=np.float32) if "tx_logits" in data.files else None,
            "dataset_role": _as_str_array(pick("dataset_role", np.asarray([""] * n)), n),
            "tx_ids": _as_str_array(pick("tx_ids", np.asarray([""] * n)), n),
            "rx_ids": _as_str_array(pick("rx_ids", np.asarray([""] * n)), n),
            "day_ids": _as_str_array(pick("day_ids", np.asarray([""] * n)), n),
            "eq_ids": _as_str_array(pick("eq_ids", np.asarray([""] * n)), n),
            "sig_ids": _as_str_array(pick("sig_ids", np.asarray([str(i) for i in range(n)])), n),
            "manifest": manifest,
        }


def _row_key(payload: Mapping[str, Any], i: int) -> tuple[str, ...]:
    return tuple(str(payload[field][i]) for field in KEY_FIELDS)


def _source_pair_indices(clean: Mapping[str, Any], sat: Mapping[str, Any], source_roles: set[str]) -> list[tuple[int, int]]:
    clean_map: dict[tuple[str, ...], int] = {}
    for i, role in enumerate(clean["dataset_role"]):
        if str(role) not in source_roles:
            continue
        key = _row_key(clean, i)
        clean_map.setdefault(key, int(i))
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
    if not pairs:
        raise ValueError("no source clean/satellite feature pairs found")
    return pairs


def _stable_split(keys: Sequence[tuple[str, ...]], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    for key in keys:
        digest = hashlib.sha256(("|".join([str(seed), *key])).encode("utf-8")).digest()
        scores.append(int.from_bytes(digest[:8], "little") / float(2**64 - 1))
    scores_arr = np.asarray(scores, dtype=np.float64)
    val = scores_arr < float(val_fraction)
    if val.all() or (~val).all():
        order = np.argsort(scores_arr)
        n_val = max(1, min(len(order) - 1, int(round(len(order) * float(val_fraction)))))
        val = np.zeros(len(order), dtype=bool)
        val[order[:n_val]] = True
    return np.where(~val)[0], np.where(val)[0]


class LinearResidualAdapter(nn.Module):
    def __init__(self, dim: int, alpha: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.alpha = float(alpha)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.proj(self.norm(x))


class MLPResidualAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, int(hidden))
        self.up = nn.Linear(int(hidden), dim)
        self.drop = nn.Dropout(float(dropout))
        self.alpha = float(alpha)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.drop(F.gelu(self.down(self.norm(x)))))


class AffineAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _build_adapter(kind: str, dim: int, hidden: int, alpha: float, dropout: float) -> nn.Module:
    mode = str(kind).lower()
    if mode == "linear_residual":
        return LinearResidualAdapter(dim, alpha)
    if mode == "mlp_residual":
        return MLPResidualAdapter(dim, hidden, alpha, dropout)
    if mode == "affine":
        return AffineAdapter(dim)
    if mode == "identity":
        return nn.Identity()
    raise ValueError(f"unknown adapter_kind={kind!r}")


def _make_clean_prototypes(clean_x: torch.Tensor, labels: Sequence[str], source_tx_ids: Sequence[str]) -> torch.Tensor:
    protos = []
    label_arr = [canonical_tx_id(x) for x in labels]
    for tx in source_tx_ids:
        idx = torch.tensor([i for i, label in enumerate(label_arr) if label == canonical_tx_id(tx)], dtype=torch.long, device=clean_x.device)
        if idx.numel() == 0:
            raise ValueError(f"no clean source features for tx={tx}")
        protos.append(clean_x.index_select(0, idx).mean(dim=0))
    return torch.stack(protos, dim=0)


def _proto_logits(x: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    x_n = F.normalize(x.float(), dim=1)
    p_n = F.normalize(prototypes.float(), dim=1)
    return x_n @ p_n.t() / max(float(temperature), 1.0e-6)


@torch.no_grad()
def _alignment_metrics(
    adapter: nn.Module,
    sat_x: torch.Tensor,
    clean_x: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    before_logits = _proto_logits(sat_x, prototypes, temperature)
    before_pred = before_logits.argmax(dim=1)
    adapted = adapter(sat_x)
    after_logits = _proto_logits(adapted, prototypes, temperature)
    after_pred = after_logits.argmax(dim=1)
    return {
        "pair_mse_before": float(F.mse_loss(sat_x, clean_x).item()),
        "pair_mse_after": float(F.mse_loss(adapted, clean_x).item()),
        "pair_cos_before": float(F.cosine_similarity(sat_x, clean_x, dim=1).mean().item()),
        "pair_cos_after": float(F.cosine_similarity(adapted, clean_x, dim=1).mean().item()),
        "proto_acc_before": float((before_pred == labels).float().mean().item()),
        "proto_acc_after": float((after_pred == labels).float().mean().item()),
        "mean_residual_norm": float((adapted - sat_x).norm(dim=1).mean().item()),
        "mean_feature_norm_after": float(adapted.norm(dim=1).mean().item()),
    }


def fit_apply(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    clean = _load_npz(args.clean_npz)
    sat = _load_npz(args.sat_npz)
    source_tx_ids = parse_tx_id_list(args.source_tx_ids)
    if not source_tx_ids:
        raise ValueError("--source_tx_ids is required")
    source_roles = {x.strip() for x in str(args.source_roles).split(",") if x.strip()}
    pairs = _source_pair_indices(clean, sat, source_roles)
    clean_idx = [i for i, _ in pairs]
    sat_idx = [j for _, j in pairs]
    pair_keys = [_row_key(sat, j) for j in sat_idx]

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    clean_x = torch.as_tensor(clean["features"][clean_idx], dtype=torch.float32, device=device)
    sat_x = torch.as_tensor(sat["features"][sat_idx], dtype=torch.float32, device=device)
    source_labels_text = [canonical_tx_id(sat["tx_ids"][j]) for j in sat_idx]
    label_map = {canonical_tx_id(tx): i for i, tx in enumerate(source_tx_ids)}
    labels = torch.tensor([label_map[tx] for tx in source_labels_text], dtype=torch.long, device=device)
    prototypes = _make_clean_prototypes(clean_x, source_labels_text, source_tx_ids)

    train_idx_np, val_idx_np = _stable_split(pair_keys, float(args.val_fraction), int(args.seed))
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)
    adapter = _build_adapter(str(args.adapter_kind), int(sat_x.shape[1]), int(args.hidden_dim), float(args.alpha), float(args.dropout)).to(device)

    if str(args.adapter_kind).lower() != "identity":
        opt = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        for epoch in range(int(args.epochs)):
            adapter.train()
            perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
            for start in range(0, int(perm.numel()), int(args.batch_size)):
                idx = perm[start : start + int(args.batch_size)]
                z_sat = sat_x.index_select(0, idx)
                z_clean = clean_x.index_select(0, idx)
                y = labels.index_select(0, idx)
                z_hat = adapter(z_sat)
                logits = _proto_logits(z_hat, prototypes.detach(), float(args.proto_temperature))
                pair_loss = F.smooth_l1_loss(z_hat, z_clean)
                cos_loss = (1.0 - F.cosine_similarity(z_hat, z_clean, dim=1)).mean()
                ce_loss = F.cross_entropy(logits, y)
                residual_loss = ((z_hat - z_sat) ** 2).mean()
                loss = (
                    float(args.pair_weight) * pair_loss
                    + float(args.cos_weight) * cos_loss
                    + float(args.proto_ce_weight) * ce_loss
                    + float(args.residual_weight) * residual_loss
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                opt.step()
    adapter.eval()
    train_metrics = _alignment_metrics(adapter, sat_x.index_select(0, train_idx), clean_x.index_select(0, train_idx), labels.index_select(0, train_idx), prototypes, float(args.proto_temperature))
    val_metrics = _alignment_metrics(adapter, sat_x.index_select(0, val_idx), clean_x.index_select(0, val_idx), labels.index_select(0, val_idx), prototypes, float(args.proto_temperature))

    all_features = torch.as_tensor(sat["features"], dtype=torch.float32, device=device)
    with torch.no_grad():
        adapted = adapter(all_features).detach().cpu().numpy().astype(np.float32)
        adapted_logits = _proto_logits(torch.as_tensor(adapted, dtype=torch.float32, device=device), prototypes, float(args.proto_temperature)).detach().cpu().numpy().astype(np.float32)

    out_arrays = dict(sat["arrays"])
    out_arrays["features"] = adapted
    out_arrays["tx_logits"] = adapted_logits
    base_manifest = dict(sat.get("manifest", {}))
    base_manifest["leo_feature_adapter"] = {
        "enabled": True,
        "training_scope": "source_clean_to_source_satellite_pairs_only",
        "clean_npz": str(args.clean_npz),
        "sat_npz": str(args.sat_npz),
        "adapter_kind": str(args.adapter_kind),
        "source_roles": sorted(source_roles),
        "source_pair_count": len(pairs),
        "train_pair_count": int(train_idx.numel()),
        "val_pair_count": int(val_idx.numel()),
        "uses_target_clean": False,
        "uses_target_labels": False,
        "uses_unknown_query_for_training": False,
        "logits": "cosine_to_source_clean_prototypes_after_adapter",
    }
    out_arrays["manifest_json"] = np.asarray(json.dumps(base_manifest, ensure_ascii=True))
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out_arrays)

    metrics = {
        "phase": "phase1_source_only_leo_feature_repair_adapter",
        "clean_npz": str(args.clean_npz),
        "sat_npz": str(args.sat_npz),
        "out_npz": str(args.out_npz),
        "source_tx_ids": source_tx_ids,
        "adapter_kind": str(args.adapter_kind),
        "feature_dim": int(sat_x.shape[1]),
        "source_pair_count": len(pairs),
        "train_pair_count": int(train_idx.numel()),
        "val_pair_count": int(val_idx.numel()),
        "loss_weights": {
            "pair_weight": float(args.pair_weight),
            "cos_weight": float(args.cos_weight),
            "proto_ce_weight": float(args.proto_ce_weight),
            "residual_weight": float(args.residual_weight),
        },
        "train_alignment": train_metrics,
        "val_alignment": val_metrics,
        "protocol": {
            "uses_source_clean_pairs": True,
            "uses_target_clean": False,
            "uses_target_labels": False,
            "uses_unknown_query_for_training": False,
            "test_features_written_from_satellite_npz_only": True,
        },
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.adapter_out:
        Path(args.adapter_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_state": adapter.state_dict(),
                "adapter_kind": str(args.adapter_kind),
                "source_tx_ids": source_tx_ids,
                "feature_dim": int(sat_x.shape[1]),
                "metrics": metrics,
            },
            args.adapter_out,
        )
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_npz", required=True)
    parser.add_argument("--sat_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--source_roles", default="source")
    parser.add_argument("--adapter_kind", default="mlp_residual", choices=["identity", "linear_residual", "mlp_residual", "affine"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--pair_weight", type=float, default=1.0)
    parser.add_argument("--cos_weight", type=float, default=0.5)
    parser.add_argument("--proto_ce_weight", type=float, default=0.25)
    parser.add_argument("--residual_weight", type=float, default=0.01)
    parser.add_argument("--proto_temperature", type=float, default=0.07)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=4070301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--adapter_out", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    metrics = fit_apply(parse_args(argv))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
