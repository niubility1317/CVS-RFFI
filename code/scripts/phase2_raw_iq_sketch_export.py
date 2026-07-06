#!/usr/bin/env python3
"""Export compact raw-IQ sketches aligned to a Phase2 feature NPZ.

This is a diagnostic bridge for qKNN: it reads raw WiSig IQ samples only at
export time and writes low-dimensional deterministic sketches. Deployment
experiments can then store quantized sketch support codes instead of raw IQ.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict pkl: {path}")
    return obj


def _index_map(values: list[Any]) -> dict[str, int]:
    return {str(value): index for index, value in enumerate(values)}


def _sample_from_compact(
    obj: dict[str, Any],
    *,
    tx: str,
    rx: str,
    day: str,
    eq: str,
    sig: str,
) -> np.ndarray:
    tx_i = _index_map(obj["tx_list"])[str(tx)]
    rx_i = _index_map(obj["rx_list"])[str(rx)]
    day_i = _index_map(obj["capture_date_list"])[str(day)]
    eq_i = _index_map(obj["equalized_list"])[str(eq)]
    sig_i = int(sig)
    arr = np.asarray(obj["data"][tx_i][rx_i][day_i][eq_i])
    if sig_i < 0 or sig_i >= int(arr.shape[0]):
        raise IndexError(
            f"sig index out of range for tx={tx} rx={rx} day={day} eq={eq}: "
            f"sig={sig_i}, available={arr.shape[0]}"
        )
    return np.asarray(arr[sig_i], dtype=np.float32)


def _preprocess_iq(x: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    flat = flat - np.mean(flat, dtype=np.float64).astype(np.float32)
    norm = float(np.linalg.norm(flat))
    if norm <= 1e-8:
        return flat
    return flat / norm


def _projection_matrix(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mat = rng.standard_normal((int(input_dim), int(output_dim))).astype(np.float32)
    mat /= np.sqrt(float(max(1, input_dim)))
    return mat


def _sketch_batch(raw_rows: np.ndarray, *, dim: int, seed: int) -> np.ndarray:
    flat = np.stack([_preprocess_iq(row) for row in raw_rows], axis=0).astype(np.float32)
    proj = _projection_matrix(flat.shape[1], int(dim), int(seed))
    sketch = flat @ proj
    sketch = np.tanh(sketch).astype(np.float32)
    sketch -= sketch.mean(axis=1, keepdims=True)
    sketch /= np.maximum(np.linalg.norm(sketch, axis=1, keepdims=True), 1e-8)
    return sketch.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--manysig_pkl", required=True)
    parser.add_argument("--manytx_pkl", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--sketch_dim", type=int, default=96)
    parser.add_argument("--projection_seed", type=int, default=60741)
    parser.add_argument(
        "--manytx_roles",
        default="target_unknown,proxy_unknown,target_new",
        help="dataset_role values read from ManyTx; all other rows use ManySig",
    )
    args = parser.parse_args()

    feature_path = Path(args.feature_npz)
    data = np.load(feature_path, allow_pickle=True)
    required = ["tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids", "dataset_role"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"feature_npz missing metadata keys: {missing}")

    manysig = _load_pickle(Path(args.manysig_pkl))
    manytx = _load_pickle(Path(args.manytx_pkl))
    manytx_roles = {item.strip() for item in str(args.manytx_roles).split(",") if item.strip()}

    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    rx_ids = np.asarray(data["rx_ids"], dtype=object).astype(str)
    day_ids = np.asarray(data["day_ids"], dtype=object).astype(str)
    eq_ids = np.asarray(data["eq_ids"], dtype=object).astype(str)
    sig_ids = np.asarray(data["sig_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)

    raw_rows: list[np.ndarray] = []
    source_counts = {"ManySig": 0, "ManyTx": 0}
    for index in range(int(tx_ids.size)):
        use_manytx = str(roles[index]) in manytx_roles
        source = manytx if use_manytx else manysig
        source_counts["ManyTx" if use_manytx else "ManySig"] += 1
        raw_rows.append(
            _sample_from_compact(
                source,
                tx=str(tx_ids[index]),
                rx=str(rx_ids[index]),
                day=str(day_ids[index]),
                eq=str(eq_ids[index]),
                sig=str(sig_ids[index]),
            )
        )

    sketch = _sketch_batch(np.stack(raw_rows, axis=0), dim=int(args.sketch_dim), seed=int(args.projection_seed))
    manifest = {
        "source_feature_npz": str(feature_path),
        "manysig_pkl": str(args.manysig_pkl),
        "manytx_pkl": str(args.manytx_pkl),
        "sketch_dim": int(args.sketch_dim),
        "projection_seed": int(args.projection_seed),
        "source_counts": source_counts,
        "stored_raw_support_count": 0,
        "method": "dc_removed_l2_raw_iq_random_projection_tanh_l2",
    }

    out: dict[str, np.ndarray] = {"features": sketch, "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True))}
    for key in data.files:
        if key in {"features", "manifest_json"}:
            continue
        out[key] = data[key]
    np.savez_compressed(args.output_npz, **out)
    print(json.dumps({"output_npz": str(args.output_npz), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
