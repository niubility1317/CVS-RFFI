#!/usr/bin/env python
"""Target-old-only MLP adapter upper-bound diagnostic.

This is a non-deployment diagnostic. It uses only `target_old` feature rows,
splits them into support/query per old TX, trains a small fixed-epoch MLP on
support rows, and evaluates old target query accuracy. Rows from target_new,
target_unknown, and proxy_unknown are ignored for fitting, normalization,
thresholding, early stopping, and model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


def _canonical(value: object) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def _as_str_array(value: np.ndarray, n: int, default: str = "") -> list[str]:
    arr = np.asarray(value)
    if arr.shape == ():
        return [_canonical(arr.item() or default)] * int(n)
    rows = arr.reshape(-1).tolist()
    if len(rows) < n:
        rows = rows + [default] * (n - len(rows))
    return [_canonical(v if v is not None else default) for v in rows[:n]]


def _parse_csv(text: str) -> list[str]:
    return [_canonical(x.strip()) for x in str(text or "").split(",") if x.strip()]


def _manifest_list(manifest: dict[str, Any], key: str) -> list[str]:
    value = manifest.get(key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_csv(value)
    if isinstance(value, (list, tuple, set)):
        return [_canonical(x) for x in value]
    return [_canonical(value)]


def _load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ValueError(f"{path} does not contain features")
        features = np.asarray(data["features"], dtype=np.float32)
        n = int(features.shape[0])

        def pick(key: str, default: str = "") -> np.ndarray:
            return np.asarray(data[key]) if key in data.files else np.asarray([default] * n)

        manifest: dict[str, Any] = {}
        if "manifest_json" in data.files:
            try:
                manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
            except Exception:
                manifest = {}
        return {
            "features": features,
            "dataset_role": _as_str_array(pick("dataset_role"), n),
            "tx_ids": _as_str_array(pick("tx_ids"), n),
            "rx_ids": _as_str_array(pick("rx_ids"), n),
            "day_ids": _as_str_array(pick("day_ids"), n),
            "sig_ids": _as_str_array(pick("sig_ids"), n),
            "manifest": manifest,
        }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1.0e-8)


def _target_old_indices(payload: dict[str, Any], target_old_tx_ids: Sequence[str]) -> dict[str, list[int]]:
    target_set = {_canonical(tx) for tx in target_old_tx_ids}
    by_tx: dict[str, list[tuple[tuple[str, str, str, int], int]]] = {tx: [] for tx in target_set}
    for i, role in enumerate(payload["dataset_role"]):
        tx = _canonical(payload["tx_ids"][i])
        if role != "target_old" or tx not in target_set:
            continue
        key = (str(payload["rx_ids"][i]), str(payload["day_ids"][i]), str(payload["sig_ids"][i]), int(i))
        by_tx[tx].append((key, i))
    return {tx: [i for _, i in sorted(rows, key=lambda item: item[0])] for tx, rows in by_tx.items()}


def _safe_rate(num: int, den: int) -> float | None:
    return None if int(den) <= 0 else float(num) / float(den)


def _index_hash(indices: Sequence[int]) -> str:
    text = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_design(
    z: np.ndarray,
    support_idx: Sequence[int],
    query_idx: Sequence[int],
    *,
    standardize: bool,
    l2_normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(z, dtype=np.float32)
    if l2_normalize:
        x = _normalize_rows(x)
    x_support = x[np.asarray(support_idx, dtype=int)]
    x_query = x[np.asarray(query_idx, dtype=int)]
    if standardize:
        mu = x_support.mean(axis=0, keepdims=True)
        sigma = x_support.std(axis=0, keepdims=True)
        x_support = (x_support - mu) / np.maximum(sigma, 1.0e-6)
        x_query = (x_query - mu) / np.maximum(sigma, 1.0e-6)
    return x_support.astype(np.float32), x_query.astype(np.float32)


class _AdapterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda was requested but is not available")
    return device


def _evaluate_one(
    z: np.ndarray,
    by_tx: dict[str, list[int]],
    *,
    k: int,
    seed: int,
    epochs: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    standardize: bool,
    l2_normalize: bool,
    device: torch.device,
) -> dict[str, Any]:
    labels = sorted(tx for tx, rows in by_tx.items() if rows)
    support_idx: list[int] = []
    support_labels: list[str] = []
    query_idx: list[int] = []
    query_labels: list[str] = []
    invalid_classes: list[str] = []
    for tx in labels:
        rows = list(by_tx[tx])
        support = rows[: max(0, int(k))]
        query = rows[max(0, int(k)) :]
        if len(support) < int(k) or not query:
            invalid_classes.append(tx)
        support_idx.extend(support)
        support_labels.extend([tx] * len(support))
        query_idx.extend(query)
        query_labels.extend([tx] * len(query))
    if not support_idx:
        raise ValueError(f"k={k} leaves no support rows")
    if not query_idx:
        raise ValueError(f"k={k} leaves no query rows")
    if invalid_classes:
        return {
            "k": int(k),
            "seed": int(seed),
            "epochs": int(epochs),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "dropout": float(dropout),
            "valid_row": False,
            "invalid_reason": "class_has_insufficient_support_or_empty_query",
            "support_count": int(len(support_idx)),
            "query_count": int(len(query_idx)),
            "support_query_overlap_count": int(len(set(support_idx) & set(query_idx))),
            "support_index_sha256": _index_hash(support_idx),
            "query_index_sha256": _index_hash(query_idx),
            "train_acc": None,
            "train_loss_final": None,
            "old_acc": None,
            "macro_old_acc": None,
            "min_old_class_acc": None,
            "invalid_classes": invalid_classes,
            "per_tx_query_count": {tx: len(list(by_tx[tx])[max(0, int(k)) :]) for tx in labels},
            "per_tx_accuracy": {tx: None for tx in labels},
            "confusion": {tx: {pred: 0 for pred in labels} for tx in labels},
        }

    x_support, x_query = _prepare_design(
        z,
        support_idx,
        query_idx,
        standardize=standardize,
        l2_normalize=l2_normalize,
    )
    label_to_col = {label: j for j, label in enumerate(labels)}
    y_support = [label_to_col[label] for label in support_labels]
    y_query = [label_to_col[label] for label in query_labels]

    _seed_everything(int(seed))
    model = _AdapterMLP(x_support.shape[1], int(hidden_dim), len(labels), float(dropout)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.CrossEntropyLoss()
    # Some remote PyTorch/NumPy builds reject torch.from_numpy despite receiving
    # a numpy.ndarray. Materializing through lists keeps this diagnostic portable.
    xs = torch.tensor(x_support.tolist(), dtype=torch.float32, device=device)
    ys = torch.tensor(y_support, dtype=torch.long, device=device)
    xq = torch.tensor(x_query.tolist(), dtype=torch.float32, device=device)

    model.train()
    last_loss = None
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xs), ys)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        train_pred = model(xs).argmax(dim=1).detach().cpu().tolist()
        pred_pos = model(xq).argmax(dim=1).detach().cpu().tolist()
    pred_labels = [labels[int(i)] for i in pred_pos]
    true_labels = [labels[int(i)] for i in y_query]
    correct = [int(p == y_true) for p, y_true in zip(pred_labels, true_labels)]

    per_tx_total = {tx: 0 for tx in labels}
    per_tx_correct = {tx: 0 for tx in labels}
    confusion = {tx: {pred: 0 for pred in labels} for tx in labels}
    for y_true, ok in zip(true_labels, correct):
        per_tx_total[y_true] += 1
        per_tx_correct[y_true] += int(ok)
    for y_true, pred in zip(true_labels, pred_labels):
        confusion[y_true][pred] += 1
    per_tx_acc = {
        tx: _safe_rate(per_tx_correct.get(tx, 0), per_tx_total.get(tx, 0))
        for tx in labels
    }
    valid_accs = [v for v in per_tx_acc.values() if v is not None]
    return {
        "k": int(k),
        "seed": int(seed),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "valid_row": True,
        "invalid_reason": "",
        "support_count": int(len(support_idx)),
        "query_count": int(len(query_idx)),
        "support_query_overlap_count": int(len(set(support_idx) & set(query_idx))),
        "support_index_sha256": _index_hash(support_idx),
        "query_index_sha256": _index_hash(query_idx),
        "train_acc": _safe_rate(sum(int(p == y) for p, y in zip(train_pred, y_support)), int(len(y_support))),
        "train_loss_final": last_loss,
        "old_acc": _safe_rate(sum(correct), len(correct)),
        "macro_old_acc": None if not valid_accs else float(sum(valid_accs) / len(valid_accs)),
        "min_old_class_acc": None if not valid_accs else float(min(valid_accs)),
        "invalid_classes": invalid_classes,
        "per_tx_query_count": per_tx_total,
        "per_tx_accuracy": per_tx_acc,
        "confusion": confusion,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_npz(args.feature_npz)
    target_old_tx_ids = _parse_csv(args.target_old_tx_ids)
    if not target_old_tx_ids:
        target_old_tx_ids = [_canonical(x) for x in payload["manifest"].get("target_old_tx_ids", [])]
    if not target_old_tx_ids:
        raise ValueError("--target_old_tx_ids or manifest target_old_tx_ids is required")
    by_tx = _target_old_indices(payload, target_old_tx_ids)
    if not any(by_tx.values()):
        raise ValueError("no target_old rows are available for target-old-only MLP adapter")

    k_values = [int(k) for k in _parse_csv(args.k_values)]
    seeds = [int(seed) for seed in _parse_csv(args.seeds)]
    if not k_values or not seeds:
        raise ValueError("--k_values and --seeds must be non-empty")

    roles = list(payload["dataset_role"])
    source_receiver_ids = _parse_csv(args.source_receiver_ids) or _manifest_list(payload["manifest"], "source_receiver_ids")
    target_receiver_ids = _parse_csv(args.target_receiver_ids) or _manifest_list(payload["manifest"], "target_receiver_ids")
    receiver_disjoint = None
    if source_receiver_ids and target_receiver_ids:
        receiver_disjoint = not bool(set(source_receiver_ids) & set(target_receiver_ids))
    selected_target_set = {_canonical(tx) for tx in target_old_tx_ids}
    observed_target_old_rx_ids = sorted(
        {
            _canonical(rx)
            for role, tx, rx in zip(payload["dataset_role"], payload["tx_ids"], payload["rx_ids"])
            if role == "target_old" and _canonical(tx) in selected_target_set
        }
    )
    observed_target_old_rx_within_target_receiver_ids = None
    if target_receiver_ids:
        observed_target_old_rx_within_target_receiver_ids = set(observed_target_old_rx_ids).issubset(
            set(target_receiver_ids)
        )
    ignored = sum(1 for role in roles if role != "target_old")
    z = np.asarray(payload["features"], dtype=np.float32)
    device = _choose_device(args.device)
    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for k in k_values:
        for seed in seeds:
            item = _evaluate_one(
                z,
                by_tx,
                k=int(k),
                seed=int(seed),
                epochs=int(args.epochs),
                hidden_dim=int(args.hidden_dim),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                dropout=float(args.dropout),
                standardize=not bool(args.no_standardize),
                l2_normalize=not bool(args.no_l2_normalize),
                device=device,
            )
            item["ignored_non_target_old_rows"] = int(ignored)
            results.append(item)
            summary_rows.append(
                {
                    "k": item["k"],
                    "seed": item["seed"],
                    "epochs": item["epochs"],
                    "hidden_dim": item["hidden_dim"],
                    "lr": f"{item['lr']:.8g}",
                    "weight_decay": f"{item['weight_decay']:.8g}",
                    "dropout": f"{item['dropout']:.8g}",
                    "valid_row": item["valid_row"],
                    "invalid_reason": item["invalid_reason"],
                    "support_count": item["support_count"],
                    "query_count": item["query_count"],
                    "old_acc": "" if item["old_acc"] is None else f"{float(item['old_acc']):.8f}",
                    "macro_old_acc": "" if item["macro_old_acc"] is None else f"{float(item['macro_old_acc']):.8f}",
                    "min_old_class_acc": "" if item["min_old_class_acc"] is None else f"{float(item['min_old_class_acc']):.8f}",
                    "train_acc": "" if item["train_acc"] is None else f"{float(item['train_acc']):.8f}",
                    "support_query_overlap_count": item["support_query_overlap_count"],
                    "invalid_classes": ",".join(item["invalid_classes"]),
                }
            )
    metrics = {
        "phase": "target_old_only_mlp_adapter_upper_bound_diagnostic",
        "verdict_scope": "non_deployment_target_old_only_diagnostic",
        "feature_npz": str(args.feature_npz),
        "target_old_tx_ids": target_old_tx_ids,
        "k_values": k_values,
        "seeds": seeds,
        "epochs": int(args.epochs),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "device": str(device),
        "standardize_support_only": not bool(args.no_standardize),
        "l2_normalize": not bool(args.no_l2_normalize),
        "source_receiver_ids": source_receiver_ids,
        "target_receiver_ids": target_receiver_ids,
        "receiver_split_disjoint": receiver_disjoint,
        "observed_target_old_rx_ids": observed_target_old_rx_ids,
        "observed_target_old_rx_within_target_receiver_ids": observed_target_old_rx_within_target_receiver_ids,
        "target_channel_view": args.target_channel_view or payload["manifest"].get("target_channel_view", ""),
        "target_old_tx_ids_from_manifest": _manifest_list(payload["manifest"], "target_old_tx_ids"),
        "model_selection_scope": "reported_grid_diagnostic_not_deployment_selection",
        "target_new_training_count": 0,
        "target_unknown_training_count": 0,
        "target_unknown_calibration_count": 0,
        "uses_unknown_query_for_threshold": False,
        "uses_unknown_for_model_selection": False,
        "early_stopping_source": "fixed_epoch_no_query_early_stopping",
        "support_query_source": "target_old_rows_only_deterministic_split",
        "ignored_role_counts": {
            role: int(sum(1 for item in roles if item == role))
            for role in sorted(set(roles))
            if role != "target_old"
        },
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary_csv and summary_rows:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", required=True)
    p.add_argument("--target_old_tx_ids", default="")
    p.add_argument("--source_receiver_ids", default="")
    p.add_argument("--target_receiver_ids", default="")
    p.add_argument("--target_channel_view", default="")
    p.add_argument("--k_values", default="1,2,5,10,20,50")
    p.add_argument("--seeds", default="1")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_standardize", action="store_true")
    p.add_argument("--no_l2_normalize", action="store_true")
    p.add_argument("--output_json", default="")
    p.add_argument("--summary_csv", default="")
    return p.parse_args(argv)


def evaluate(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(evaluate(argv), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
