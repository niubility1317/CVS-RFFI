#!/usr/bin/env python3
"""Raw-IQ kNN baseline for CVS Stage2-C multi-new-class probes.

This diagnostic reuses the feature NPZ metadata only for protocol alignment:
role labels, TX/RX/day/eq/sig identifiers, satellite scenario tags, and the
existing support/query split policy. The classifier itself sees flattened IQ
samples, not model features.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
for path in (str(SCRIPT_DIR), str(CODE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import phase2_qknn_active_support_select as active
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list


def _parse_csv(text: str | None) -> list[str]:
    out: list[str] = []
    for item in str(text or "").replace(";", ",").split(","):
        value = item.strip()
        if value:
            out.append(value)
    return out


def _load_pickle(path: str) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "data" not in data:
        raise ValueError(f"unexpected WiSig pickle payload: {path}")
    if "tx_list" not in data and "node_list" in data:
        data["tx_list"] = data["node_list"]
    if "capture_date_list" not in data:
        data["capture_date_list"] = [None]
    if "rx_list" not in data:
        data["rx_list"] = [None]
    if "equalized_list" not in data:
        data["equalized_list"] = [0]
    return data


def _index_map(values: list[Any]) -> dict[str, int]:
    return {canonical_tx_id(value): int(i) for i, value in enumerate(values)}


def _fetch_iq(
    stores: dict[str, dict[str, Any]],
    store_key: str,
    *,
    tx: str,
    rx: str,
    day: str,
    eq: str,
    sig: str,
    out_len: int,
) -> np.ndarray:
    store = stores[store_key]
    maps = store["_maps"]
    tx_i = maps["tx"][canonical_tx_id(tx)]
    rx_i = maps["rx"][canonical_tx_id(rx)]
    day_i = maps["day"][canonical_tx_id(day)]
    eq_i = maps["eq"][canonical_tx_id(eq)]
    sig_i = int(sig)
    raw = np.asarray(store["data"][tx_i][rx_i][day_i][eq_i][sig_i], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"unexpected IQ shape for {tx}/{rx}/{day}/{eq}/{sig}: {raw.shape}")
    if raw.shape[0] == 2:
        x = raw
    elif raw.shape[1] == 2:
        x = raw.T
    else:
        raise ValueError(f"unexpected IQ shape for {tx}/{rx}/{day}/{eq}/{sig}: {raw.shape}")
    if x.shape[1] > int(out_len):
        start = (x.shape[1] - int(out_len)) // 2
        x = x[:, start : start + int(out_len)]
    elif x.shape[1] < int(out_len):
        pad = np.zeros((2, int(out_len)), dtype=np.float32)
        left = (int(out_len) - x.shape[1]) // 2
        pad[:, left : left + x.shape[1]] = x
        x = pad
    power = np.mean(x[0] * x[0] + x[1] * x[1])
    return (x / np.sqrt(power + 1e-12)).astype(np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _apply_view(
    batch: np.ndarray,
    scenarios: np.ndarray,
    *,
    view: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    if view == "clean":
        return batch.reshape(batch.shape[0], -1)
    rows: list[np.ndarray] = []
    gen = make_torch_generator(device, int(seed))
    for scenario in sorted({str(v) for v in scenarios.tolist()}):
        local = np.where(scenarios.astype(str) == scenario)[0]
        x = torch.from_numpy(batch[local]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            y, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
        out = y.detach().cpu().numpy().reshape(len(local), -1)
        rows.extend((int(i), out[pos]) for pos, i in enumerate(local.tolist()))
    rows.sort(key=lambda item: item[0])
    return np.stack([row for _idx, row in rows], axis=0)


def _knn_predict(
    support: np.ndarray,
    support_labels: np.ndarray,
    query: np.ndarray,
    *,
    vote_k: int,
) -> np.ndarray:
    sims = query @ support.T
    k = max(1, min(int(vote_k), int(support.shape[0])))
    top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    pred: list[str] = []
    for row, cols in enumerate(top):
        labels = support_labels[cols]
        if k == 1:
            pred.append(str(labels[0]))
            continue
        totals: dict[str, float] = {}
        for label, col in zip(labels.tolist(), cols.tolist()):
            totals[str(label)] = totals.get(str(label), 0.0) + float(sims[row, col])
        pred.append(max(totals.items(), key=lambda item: (item[1], item[0]))[0])
    return np.asarray(pred, dtype=object)


def _acc_dict(truth: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in labels:
        mask = truth == label
        out[str(label)] = float(np.mean(pred[mask] == truth[mask])) if bool(mask.any()) else float("nan")
    return out


def _build_stores(feature_npz: Path, args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    z = np.load(feature_npz, allow_pickle=True)
    manifest = json.loads(str(z["manifest_json"].item()))
    old_pkl = str(args.old_pkl or manifest["target_old"]["pkl"])
    new_pkl = str(args.new_pkl or manifest["target_unknown"]["pkl"])
    stores = {"target_old": _load_pickle(old_pkl), "target_unknown": _load_pickle(new_pkl)}
    for store in stores.values():
        store["_maps"] = {
            "tx": _index_map(list(store.get("tx_list", []))),
            "rx": _index_map(list(store.get("rx_list", []))),
            "day": _index_map(list(store.get("capture_date_list", []))),
            "eq": _index_map(list(store.get("equalized_list", []))),
        }
    return stores, manifest


def evaluate(args: argparse.Namespace) -> list[dict[str, Any]]:
    feature_npz = Path(args.feature_npz)
    z = np.load(feature_npz, allow_pickle=True)
    stores, manifest = _build_stores(feature_npz, args)
    tx_ids = np.asarray(z["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(z["dataset_role"], dtype=object).astype(str)
    scenarios = np.asarray(z["sat_scenarios"], dtype=object).astype(str)
    features_for_split = _normalize_rows(np.asarray(z["features"], dtype=np.float32))
    source_probs = np.zeros((features_for_split.shape[0], 1), dtype=np.float32)
    source_label_to_idx: dict[str, int] = {}
    source_prototypes: dict[str, np.ndarray] = {}

    old_labels = [str(v) for v in parse_tx_id_list(args.old_tx_ids)]
    if not old_labels:
        old_labels = [str(v) for v in manifest["target_old"]["tx_labels"]]
    all_new = [str(v) for v in manifest["target_unknown"]["tx_labels"]]
    requested_new = [str(v) for v in parse_tx_id_list(args.new_tx_ids)]
    new_labels = requested_new if requested_new else all_new[: int(args.new_count)]
    class_labels = old_labels + new_labels

    raw_cache: dict[int, np.ndarray] = {}

    def raw_for_indices(indices: np.ndarray) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in indices.astype(int).tolist():
            if i not in raw_cache:
                role = str(roles[i])
                store_key = "target_old" if role == str(args.old_role) else "target_unknown"
                raw_cache[i] = _fetch_iq(
                    stores,
                    store_key,
                    tx=str(z["tx_ids"][i]),
                    rx=str(z["rx_ids"][i]),
                    day=str(z["day_ids"][i]),
                    eq=str(z["eq_ids"][i]),
                    sig=str(z["sig_ids"][i]),
                    out_len=int(args.out_len),
                )
            out.append(raw_cache[i])
        return np.stack(out, axis=0)

    rows: list[dict[str, Any]] = []
    device = torch.device(str(args.device))
    for shot in [int(v) for v in _parse_csv(args.k_values)]:
        query_per = int(args.max_per_class) - shot if int(args.query_per_class) <= 0 else int(args.query_per_class)
        old_raw = active._build_active_splits(
            tx_ids=tx_ids,
            roles=roles,
            features=features_for_split,
            scenarios=scenarios,
            source_probs=source_probs,
            source_label_to_idx=source_label_to_idx,
            source_prototypes=source_prototypes,
            labels=old_labels,
            role=str(args.old_role),
            k=shot,
            query_per_class=query_per,
            pool_per_class=shot,
            policy=str(args.policy),
            seed=int(args.seed),
            exclude_pool_from_query=False,
        )
        new_raw = active._build_active_splits(
            tx_ids=tx_ids,
            roles=roles,
            features=features_for_split,
            scenarios=scenarios,
            source_probs=source_probs,
            source_label_to_idx=source_label_to_idx,
            source_prototypes=source_prototypes,
            labels=new_labels,
            role=str(args.new_role),
            k=shot,
            query_per_class=query_per,
            pool_per_class=shot,
            policy=str(args.policy),
            seed=int(args.seed),
            exclude_pool_from_query=False,
        )
        if set(old_raw) != set(old_labels) or set(new_raw) != set(new_labels):
            raise RuntimeError(f"cannot build complete split for K={shot}")
        old_splits = active._as_eval_splits(old_raw)
        new_splits = active._as_eval_splits(new_raw)
        support_indices: list[int] = []
        support_labels: list[str] = []
        query_indices: list[int] = []
        query_truth: list[str] = []
        for label in old_labels:
            support, query = old_splits[label]
            support_indices.extend(support.tolist())
            support_labels.extend([label] * int(support.size))
            query_indices.extend(query.tolist())
            query_truth.extend([label] * int(query.size))
        old_query_count = len(query_indices)
        for label in new_labels:
            support, query = new_splits[label]
            support_indices.extend(support.tolist())
            support_labels.extend([label] * int(support.size))
            query_indices.extend(query.tolist())
            query_truth.extend([label] * int(query.size))
        support_idx = np.asarray(support_indices, dtype=int)
        query_idx = np.asarray(query_indices, dtype=int)
        support_y = np.asarray(support_labels, dtype=object).astype(str)
        truth = np.asarray(query_truth, dtype=object).astype(str)
        for view in _parse_csv(args.views):
            support_raw = raw_for_indices(support_idx)
            query_raw = raw_for_indices(query_idx)
            support_x = _normalize_rows(
                _apply_view(
                    support_raw,
                    scenarios[support_idx],
                    view=view,
                    args=args,
                    device=device,
                    seed=int(args.seed) + 1701 + shot,
                )
            )
            query_x = _normalize_rows(
                _apply_view(
                    query_raw,
                    scenarios[query_idx],
                    view=view,
                    args=args,
                    device=device,
                    seed=int(args.seed) + 2701 + shot,
                )
            )
            for vote_k in [int(v) for v in _parse_csv(args.vote_k_values)]:
                pred = _knn_predict(support_x, support_y, query_x, vote_k=vote_k)
                old_truth = truth[:old_query_count]
                old_pred = pred[:old_query_count]
                new_truth = truth[old_query_count:]
                new_pred = pred[old_query_count:]
                per_old = _acc_dict(old_truth, old_pred, old_labels)
                per_new = _acc_dict(new_truth, new_pred, new_labels)
                row = {
                    "view": view,
                    "shot_k": int(shot),
                    "vote_k": int(vote_k),
                    "new_class_count": int(len(new_labels)),
                    "old_class_count": int(len(old_labels)),
                    "query_per_class": int(query_per),
                    "old_acc": float(np.mean(old_pred == old_truth)),
                    "min_old_class_acc": float(min(per_old.values())),
                    "seen_new_acc": float(np.mean(new_pred == new_truth)),
                    "min_seen_new_class_acc": float(min(per_new.values())),
                    "query_per_old_acc": per_old,
                    "query_per_new_acc": per_new,
                    "old_query_count": int(old_truth.size),
                    "new_query_count": int(new_truth.size),
                    "stored_raw_support_count": int(support_idx.size),
                    "stored_raw_support_scalars": int(support_idx.size * 2 * int(args.out_len)),
                    "feature_npz": str(feature_npz),
                    "old_pkl": str(args.old_pkl or manifest["target_old"]["pkl"]),
                    "new_pkl": str(args.new_pkl or manifest["target_unknown"]["pkl"]),
                    "new_tx_ids": ",".join(new_labels),
                    "policy": str(args.policy),
                    "seed": int(args.seed),
                }
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--old_pkl", default="")
    parser.add_argument("--new_pkl", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--old_tx_ids", default="")
    parser.add_argument("--new_tx_ids", default="")
    parser.add_argument("--new_count", type=int, default=10)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--k_values", default="5,10")
    parser.add_argument("--vote_k_values", default="1")
    parser.add_argument("--views", default="clean,leo")
    parser.add_argument("--policy", default="stable_first")
    parser.add_argument("--seed", type=int, default=421027)
    parser.add_argument("--max_per_class", type=int, default=80)
    parser.add_argument("--query_per_class", type=int, default=0)
    parser.add_argument("--out_len", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    args = parser.parse_args()

    rows = evaluate(args)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, ensure_ascii=False, indent=2)
    fieldnames = [
        "view",
        "shot_k",
        "vote_k",
        "new_class_count",
        "old_class_count",
        "query_per_class",
        "old_acc",
        "min_old_class_acc",
        "seen_new_acc",
        "min_seen_new_class_acc",
        "query_per_old_acc",
        "query_per_new_acc",
        "old_query_count",
        "new_query_count",
        "stored_raw_support_count",
        "stored_raw_support_scalars",
        "new_tx_ids",
        "policy",
        "seed",
        "feature_npz",
        "old_pkl",
        "new_pkl",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["query_per_old_acc"] = json.dumps(row["query_per_old_acc"], ensure_ascii=False, sort_keys=True)
            out["query_per_new_acc"] = json.dumps(row["query_per_new_acc"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"rows": rows, "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
