#!/usr/bin/env python
"""Build source-only local residual-memory feature repairs for Phase1 target1.

The repair memory is built from source clean/LEO pairs only. Each inference
feature retrieves nearby source LEO features, averages their clean-minus-LEO
residuals, and applies the correction only when the sample is close enough to
the source LEO memory under a threshold calibrated from source clean rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(x) for x in parse_csv(text)]


def parse_int_csv(text: str) -> list[int]:
    return [int(x) for x in parse_csv(text)]


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x) for x in parse_csv(text)]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    if "features" not in arrays:
        raise ValueError(f"{path} missing features")
    return arrays


def as_str(arrays: dict[str, np.ndarray], key: str, n: int, default: str = "") -> np.ndarray:
    if key not in arrays:
        return np.asarray([default] * n, dtype=str)
    arr = np.asarray(arrays[key])
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch {arr.shape[0]} != {n}")
    return arr.astype(str)


def tx_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    n = int(arrays["features"].shape[0])
    return np.asarray([canonical_tx_id(x) for x in as_str(arrays, "tx_ids", n)], dtype=str)


def role_array(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return as_str(arrays, "dataset_role", int(arrays["features"].shape[0]))


def row_keys(arrays: dict[str, np.ndarray]) -> list[tuple[str, ...]]:
    n = int(arrays["features"].shape[0])
    fields = [
        role_array(arrays),
        tx_array(arrays),
        as_str(arrays, "rx_ids", n),
        as_str(arrays, "day_ids", n),
        as_str(arrays, "eq_ids", n),
        as_str(arrays, "sig_ids", n),
    ]
    return [tuple(str(col[i]) for col in fields) for i in range(n)]


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-6)


def make_prototypes(clean: dict[str, np.ndarray], source_tx_ids: Sequence[str]) -> np.ndarray:
    feats = np.asarray(clean["features"], dtype=np.float32)
    tx = tx_array(clean)
    protos = []
    for item in source_tx_ids:
        mask = tx == canonical_tx_id(item)
        if not bool(mask.any()):
            raise ValueError(f"no clean source rows for tx={item}")
        proto = feats[mask].mean(axis=0)
        protos.append((proto / max(float(np.linalg.norm(proto)), 1.0e-6)).astype(np.float32))
    return np.stack(protos, axis=0)


def pair_source(clean: dict[str, np.ndarray], sats: Sequence[dict[str, np.ndarray]], source_tx_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    source_set = {canonical_tx_id(x) for x in source_tx_ids}
    clean_keys = row_keys(clean)
    clean_roles = role_array(clean)
    clean_tx = tx_array(clean)
    clean_map: dict[tuple[str, ...], int] = {}
    for i, key in enumerate(clean_keys):
        if clean_roles[i] == "source" and clean_tx[i] in source_set:
            clean_map.setdefault(key, i)
    clean_rows = []
    leo_rows = []
    for sat in sats:
        sat_keys = row_keys(sat)
        sat_roles = role_array(sat)
        sat_tx = tx_array(sat)
        seen = set()
        for j, key in enumerate(sat_keys):
            if key in seen:
                continue
            seen.add(key)
            if sat_roles[j] != "source" or sat_tx[j] not in source_set or key not in clean_map:
                continue
            clean_rows.append(clean["features"][clean_map[key]])
            leo_rows.append(sat["features"][j])
    if not clean_rows:
        raise ValueError("no source clean/LEO pairs")
    return np.asarray(clean_rows, dtype=np.float32), np.asarray(leo_rows, dtype=np.float32)


def proto_logits(features: np.ndarray, prototypes: np.ndarray, temperature: float) -> np.ndarray:
    return (l2n(np.asarray(features, dtype=np.float32)) @ prototypes.T) / max(float(temperature), 1.0e-6)


def topk_local_residual(
    features: np.ndarray,
    memory_features_n: np.ndarray,
    residuals: np.ndarray,
    *,
    k: int,
    temperature: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = l2n(np.asarray(features, dtype=np.float32))
    k = max(1, min(int(k), int(memory_features_n.shape[0])))
    corrections = np.zeros_like(np.asarray(features, dtype=np.float32))
    max_sims = np.zeros((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), x.shape[0])
        sims = x[start:end] @ memory_features_n.T
        idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(vals, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        max_sims[start:end] = vals[:, 0]
        z = vals / max(float(temperature), 1.0e-6)
        z = z - z.max(axis=1, keepdims=True)
        weights = np.exp(z)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
        corrections[start:end] = np.einsum("bk,bkd->bd", weights.astype(np.float32), residuals[idx])
    return corrections.astype(np.float32), max_sims.astype(np.float32)


def maxsim_to_memory(features: np.ndarray, memory_features_n: np.ndarray, chunk_size: int) -> np.ndarray:
    x = l2n(np.asarray(features, dtype=np.float32))
    out = np.zeros((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), x.shape[0])
        out[start:end] = (x[start:end] @ memory_features_n.T).max(axis=1)
    return out


def write_npz(
    input_npz: Path,
    output_npz: Path,
    prototypes: np.ndarray,
    memory_features_n: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
    *,
    k: int,
    alpha: float,
    gate_scale: float,
    local_temperature: float,
    proto_temperature: float,
    chunk_size: int,
    clean_control: bool,
) -> None:
    arrays = load_npz(input_npz)
    corrections, max_sims = topk_local_residual(
        arrays["features"],
        memory_features_n,
        residuals,
        k=k,
        temperature=local_temperature,
        chunk_size=chunk_size,
    )
    gate = 1.0 / (1.0 + np.exp(-(max_sims - float(threshold)) / max(float(gate_scale), 1.0e-6)))
    repaired = np.asarray(arrays["features"], dtype=np.float32) + float(alpha) * gate.reshape(-1, 1).astype(np.float32) * corrections
    out = dict(arrays)
    out["features"] = repaired.astype(np.float32)
    out["tx_logits"] = proto_logits(repaired, prototypes, proto_temperature).astype(np.float32)
    out["local_residual_gate"] = gate.astype(np.float32)
    out["local_residual_maxsim"] = max_sims.astype(np.float32)
    manifest: dict[str, Any] = {}
    if "manifest_json" in arrays:
        try:
            manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
        except Exception:
            manifest = {}
    manifest.update({
        "local_residual_memory_repair": {
            "enabled": True,
            "training_scope": "source_clean_to_source_leo_pairs_only",
            "features_changed": True,
            "k": int(k),
            "alpha": float(alpha),
            "threshold": float(threshold),
            "gate_scale": float(gate_scale),
            "local_temperature": float(local_temperature),
            "clean_control": bool(clean_control),
            "uses_target_clean": False,
            "uses_target_labels": False,
            "uses_unknown_query_for_training": False,
        }
    })
    out["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **out)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--source_clean_npz", type=Path, required=True)
    p.add_argument("--train_sat_npz", type=Path, action="append", required=True)
    p.add_argument("--run_ids", required=True)
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--ks", default="8,32,128")
    p.add_argument("--alphas", default="0.25,0.50,0.75")
    p.add_argument("--threshold_quantiles", default="0.05,0.10,0.20")
    p.add_argument("--gate_scale", type=float, default=0.03)
    p.add_argument("--local_temperature", type=float, default=0.05)
    p.add_argument("--proto_temperature", type=float, default=0.07)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--sat_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz")
    p.add_argument("--clean_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz")
    args = p.parse_args(argv)

    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    clean = load_npz(args.source_clean_npz)
    sats = [load_npz(path) for path in args.train_sat_npz]
    clean_x, leo_x = pair_source(clean, sats, source_tx_ids)
    residuals = (clean_x - leo_x).astype(np.float32)
    memory_features_n = l2n(leo_x.astype(np.float32))
    source_clean_maxsim = maxsim_to_memory(clean_x, memory_features_n, int(args.chunk_size))
    thresholds = {float(q): float(np.quantile(source_clean_maxsim, float(q))) for q in parse_float_csv(args.threshold_quantiles)}
    prototypes = make_prototypes(clean, source_tx_ids)

    rows = []
    run_ids = parse_csv(args.run_ids)
    for k in parse_int_csv(args.ks):
        for q, threshold in thresholds.items():
            qtag = f"Q{int(q * 100 + 0.5):02d}"
            for alpha in parse_float_csv(args.alphas):
                atag = f"A{int(alpha * 100 + 0.5):03d}"
                variant = f"LEOMEM1_K{int(k):03d}_{qtag}_{atag}"
                for run_id in run_ids:
                    run_dir = args.runs_root / run_id
                    sat_npz = run_dir / str(args.sat_relpath)
                    clean_npz = run_dir / str(args.clean_relpath)
                    if not sat_npz.exists():
                        rows.append({"run_id": run_id, "variant": variant, "status": "missing_sat", "path": str(sat_npz)})
                        continue
                    out_dir = run_dir / variant
                    write_npz(
                        sat_npz,
                        out_dir / "features_leo_repaired.npz",
                        prototypes,
                        memory_features_n,
                        residuals,
                        threshold,
                        k=int(k),
                        alpha=float(alpha),
                        gate_scale=float(args.gate_scale),
                        local_temperature=float(args.local_temperature),
                        proto_temperature=float(args.proto_temperature),
                        chunk_size=int(args.chunk_size),
                        clean_control=False,
                    )
                    if clean_npz.exists():
                        write_npz(
                            clean_npz,
                            out_dir / "features_clean_repaired.npz",
                            prototypes,
                            memory_features_n,
                            residuals,
                            threshold,
                            k=int(k),
                            alpha=float(alpha),
                            gate_scale=float(args.gate_scale),
                            local_temperature=float(args.local_temperature),
                            proto_temperature=float(args.proto_temperature),
                            chunk_size=int(args.chunk_size),
                            clean_control=True,
                        )
                    rows.append({"run_id": run_id, "variant": variant, "status": "ok"})
    print(json.dumps({
        "phase": "phase1_local_residual_memory_v25",
        "source_pair_count": int(clean_x.shape[0]),
        "source_clean_maxsim_quantiles": {str(k): v for k, v in thresholds.items()},
        "rows": rows,
        "uses_target_clean": False,
        "uses_target_labels": False,
        "uses_unknown_query_for_training": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
