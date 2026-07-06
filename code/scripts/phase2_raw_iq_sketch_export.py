#!/usr/bin/env python3
"""Export compact LEO/raw-IQ sketches aligned to a Phase2 feature NPZ.

This is a diagnostic bridge for qKNN: it reads raw WiSig IQ samples only at
export time and writes low-dimensional deterministic sketches. Deployment
experiments can then store quantized sketch support codes instead of raw IQ.
"""

from __future__ import annotations

import argparse
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
    while path in sys.path:
        sys.path.remove(path)
for path in (str(REPO_ROOT), str(CODE_ROOT), str(SCRIPT_DIR)):
    sys.path.insert(0, path)

from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator


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


def _as_iq2t(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"unexpected IQ rank: {arr.shape}")
    if arr.shape[0] == 2:
        out = arr
    elif arr.shape[1] == 2:
        out = arr.T
    else:
        raise ValueError(f"unexpected IQ shape: {arr.shape}")
    return np.asarray(out, dtype=np.float32, order="C")


def _preprocess_iq(x: np.ndarray) -> np.ndarray:
    flat = _as_iq2t(x).reshape(-1)
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


def _sketch_batch_with_projection(raw_rows: np.ndarray, *, projection: np.ndarray) -> np.ndarray:
    flat = np.stack([_preprocess_iq(row) for row in raw_rows], axis=0).astype(np.float32)
    if flat.shape[1] != int(projection.shape[0]):
        raise ValueError(f"projection input mismatch: flat={flat.shape}, projection={projection.shape}")
    sketch = flat @ projection
    sketch = np.tanh(sketch).astype(np.float32)
    sketch -= sketch.mean(axis=1, keepdims=True)
    sketch /= np.maximum(np.linalg.norm(sketch, axis=1, keepdims=True), 1e-8)
    return sketch.astype(np.float32)


def _sketch_batch(raw_rows: np.ndarray, *, dim: int, seed: int) -> np.ndarray:
    flat_dim = int(np.asarray(raw_rows[0], dtype=np.float32).size)
    proj = _projection_matrix(flat_dim, int(dim), int(seed))
    return _sketch_batch_with_projection(raw_rows, projection=proj)


def _scenario_array(data: np.lib.npyio.NpzFile, count: int, fallback: str) -> np.ndarray:
    if "sat_scenarios" not in data.files:
        return np.asarray([str(fallback)] * int(count), dtype=object)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    if int(scenarios.size) != int(count):
        raise ValueError(f"sat_scenarios length mismatch: {scenarios.size} != {count}")
    return scenarios


def _to_torch_float(array: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(array, dtype=np.float32, order="C")
    try:
        return torch.as_tensor(arr.copy(), dtype=torch.float32, device=device)
    except Exception:
        return torch.tensor(arr.tolist(), dtype=torch.float32, device=device)


def _sketch_leo_batch(
    raw_rows: np.ndarray,
    scenarios: np.ndarray,
    *,
    projection: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    raw = np.stack([_as_iq2t(row) for row in raw_rows], axis=0).astype(np.float32)
    out = np.zeros((raw.shape[0], int(projection.shape[1])), dtype=np.float32)
    device = torch.device(str(args.device))
    views = max(1, int(args.leo_tta_views))
    batch_size = max(1, int(args.batch_size))
    scenarios = np.asarray(scenarios, dtype=object).astype(str)
    for start in range(0, raw.shape[0], batch_size):
        end = min(raw.shape[0], start + batch_size)
        local_raw = raw[start:end]
        local_scenarios = scenarios[start:end]
        local_acc = np.zeros((end - start, int(projection.shape[1])), dtype=np.float32)
        for view_i in range(views):
            view_rows: list[tuple[int, np.ndarray]] = []
            for scenario in sorted({str(v) for v in local_scenarios.tolist()}):
                pos = np.where(local_scenarios == scenario)[0]
                x = _to_torch_float(local_raw[pos], device)
                gen = make_torch_generator(device, int(args.channel_seed) + 1009 * view_i + start)
                with torch.no_grad():
                    y, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
                y_np = y.detach().cpu().numpy().astype(np.float32)
                sketch = _sketch_batch_with_projection(y_np, projection=projection)
                view_rows.extend((int(i), sketch[j]) for j, i in enumerate(pos.tolist()))
            view_rows.sort(key=lambda item: item[0])
            local_acc += np.stack([row for _idx, row in view_rows], axis=0)
        local_acc /= float(views)
        local_acc -= local_acc.mean(axis=1, keepdims=True)
        local_acc /= np.maximum(np.linalg.norm(local_acc, axis=1, keepdims=True), 1e-8)
        out[start:end] = local_acc.astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--manysig_pkl", required=True)
    parser.add_argument("--manytx_pkl", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--sketch_dim", type=int, default=96)
    parser.add_argument("--projection_seed", type=int, default=60741)
    parser.add_argument("--channel_view", choices=["leo", "clean"], default="leo")
    parser.add_argument("--leo_tta_views", type=int, default=5)
    parser.add_argument("--default_sat_scenario", default="leo_clear_weak")
    parser.add_argument("--channel_seed", type=int, default=960741)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
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
    scenarios = _scenario_array(data, int(tx_ids.size), str(args.default_sat_scenario))

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

    raw_array = np.stack([_as_iq2t(row) for row in raw_rows], axis=0)
    projection = _projection_matrix(raw_array.shape[1] * raw_array.shape[2], int(args.sketch_dim), int(args.projection_seed))
    if str(args.channel_view) == "leo":
        sketch = _sketch_leo_batch(raw_array, scenarios, projection=projection, args=args)
    else:
        sketch = _sketch_batch_with_projection(raw_array, projection=projection)
    source_manifest = json.loads(str(data["manifest_json"].item())) if "manifest_json" in data.files else {}
    manifest = {
        "source_feature_npz": str(feature_path),
        "manysig_pkl": str(args.manysig_pkl),
        "manytx_pkl": str(args.manytx_pkl),
        "sketch_dim": int(args.sketch_dim),
        "projection_seed": int(args.projection_seed),
        "source_counts": source_counts,
        "stored_raw_support_count": 0,
        "method": (
            "leo_tta_dc_removed_l2_raw_iq_random_projection_tanh_l2"
            if str(args.channel_view) == "leo"
            else "dc_removed_l2_raw_iq_random_projection_tanh_l2"
        ),
        "channel_view": "satellite/LEO" if str(args.channel_view) == "leo" else "clean/control",
        "uses_target_clean": bool(str(args.channel_view) != "leo"),
        "applies_star_ground_channel": bool(str(args.channel_view) == "leo"),
        "star_ground_channel_impl": str(source_manifest.get("star_ground_channel_impl", "simplified_leo_residual")),
        "sat_scenarios": sorted({str(v) for v in scenarios.tolist()}),
        "leo_tta_views": int(args.leo_tta_views) if str(args.channel_view) == "leo" else 0,
        "channel_seed": int(args.channel_seed),
        "sat_fs_hz": float(args.sat_fs_hz),
        "sat_fc_hz": float(args.sat_fc_hz),
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
