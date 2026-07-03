#!/usr/bin/env python
"""Build source-only residual-subspace feature repairs for Phase1 target1 audit.

The repair is feature-only: source clean/LEO pairs estimate a LEO residual
subspace and per-TX residual directions. At inference, the correction is gated
by source-only clean-vs-LEO likelihood in that residual subspace and blended by
prototype-predicted TX weights. No target clean, target labels, or unknown query
labels are used for fitting.
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


def pair_source(clean: dict[str, np.ndarray], sats: Sequence[dict[str, np.ndarray]], source_tx_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    labels = []
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
            labels.append(canonical_tx_id(sat_tx[j]))
    if not clean_rows:
        raise ValueError("no source clean/LEO pairs")
    return (
        np.asarray(clean_rows, dtype=np.float32),
        np.asarray(leo_rows, dtype=np.float32),
        np.asarray(labels, dtype=str),
    )


def fit_model(clean_x: np.ndarray, leo_x: np.ndarray, labels: np.ndarray, source_tx_ids: Sequence[str], rank: int) -> dict[str, Any]:
    residual = np.asarray(clean_x - leo_x, dtype=np.float32)
    mean_residual = residual.mean(axis=0)
    centered = residual - mean_residual.reshape(1, -1)
    _, _, vt = np.linalg.svd(centered.astype(np.float32), full_matrices=False)
    basis = vt[: int(rank)].astype(np.float32)
    proj = basis.T @ basis

    tx_delta = {}
    for tx in source_tx_ids:
        mask = labels == canonical_tx_id(tx)
        delta = residual[mask].mean(axis=0) if bool(mask.any()) else mean_residual
        tx_delta[canonical_tx_id(tx)] = (proj @ delta.astype(np.float32)).astype(np.float32)

    clean_q = clean_x @ basis.T
    leo_q = leo_x @ basis.T
    clean_mu = clean_q.mean(axis=0)
    leo_mu = leo_q.mean(axis=0)
    clean_var = clean_q.var(axis=0) + 1.0e-4
    leo_var = leo_q.var(axis=0) + 1.0e-4
    return {
        "basis": basis,
        "tx_delta": tx_delta,
        "clean_mu": clean_mu.astype(np.float32),
        "leo_mu": leo_mu.astype(np.float32),
        "clean_var": clean_var.astype(np.float32),
        "leo_var": leo_var.astype(np.float32),
    }


def log_diag_gauss(q: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (((q - mu.reshape(1, -1)) ** 2) / var.reshape(1, -1) + np.log(var.reshape(1, -1))).sum(axis=1)


def proto_logits(features: np.ndarray, prototypes: np.ndarray, temperature: float) -> np.ndarray:
    return (l2n(np.asarray(features, dtype=np.float32)) @ prototypes.T) / max(float(temperature), 1.0e-6)


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / max(float(temperature), 1.0e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1.0e-8)


def repair_features(features: np.ndarray, prototypes: np.ndarray, model: dict[str, Any], source_tx_ids: Sequence[str], *, alpha: float, gate_scale: float, mode: str, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    basis = model["basis"]
    q = x @ basis.T
    ll_clean = log_diag_gauss(q, model["clean_mu"], model["clean_var"])
    ll_leo = log_diag_gauss(q, model["leo_mu"], model["leo_var"])
    gate = 1.0 / (1.0 + np.exp(-float(gate_scale) * (ll_leo - ll_clean)))
    logits = proto_logits(x, prototypes, temperature)
    if mode == "hardtx":
        pred = logits.argmax(axis=1)
        corr = np.stack([model["tx_delta"][canonical_tx_id(source_tx_ids[int(i)])] for i in pred], axis=0)
    elif mode == "softtx":
        weights = softmax(logits, 1.0)
        deltas = np.stack([model["tx_delta"][canonical_tx_id(tx)] for tx in source_tx_ids], axis=0)
        corr = weights @ deltas
    elif mode == "global":
        deltas = np.stack([model["tx_delta"][canonical_tx_id(tx)] for tx in source_tx_ids], axis=0)
        corr = np.repeat(deltas.mean(axis=0, keepdims=True), x.shape[0], axis=0)
    else:
        raise ValueError(f"unknown repair mode {mode}")
    repaired = x + float(alpha) * gate.reshape(-1, 1).astype(np.float32) * corr.astype(np.float32)
    return repaired.astype(np.float32), gate.astype(np.float32)


def write_npz(input_npz: Path, output_npz: Path, prototypes: np.ndarray, model: dict[str, Any], source_tx_ids: Sequence[str], *, alpha: float, gate_scale: float, mode: str, rank: int, temperature: float, clean_control: bool) -> None:
    arrays = load_npz(input_npz)
    repaired, gate = repair_features(arrays["features"], prototypes, model, source_tx_ids, alpha=alpha, gate_scale=gate_scale, mode=mode, temperature=temperature)
    out = dict(arrays)
    out["features"] = repaired
    out["tx_logits"] = proto_logits(repaired, prototypes, temperature).astype(np.float32)
    out["leo_residual_gate"] = gate
    manifest = {}
    if "manifest_json" in arrays:
        try:
            manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
        except Exception:
            manifest = {}
    manifest.update({
        "residual_subspace_repair": {
            "enabled": True,
            "training_scope": "source_clean_to_source_leo_pairs_only",
            "features_changed": True,
            "mode": mode,
            "rank": int(rank),
            "alpha": float(alpha),
            "gate_scale": float(gate_scale),
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
    p.add_argument("--ranks", default="4,8,16")
    p.add_argument("--alphas", default="0.25,0.50,0.75")
    p.add_argument("--modes", default="softtx,hardtx,global")
    p.add_argument("--gate_scale", type=float, default=0.35)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--sat_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz")
    p.add_argument("--clean_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz")
    args = p.parse_args(argv)

    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    clean = load_npz(args.source_clean_npz)
    sats = [load_npz(path) for path in args.train_sat_npz]
    clean_x, leo_x, labels = pair_source(clean, sats, source_tx_ids)
    prototypes = make_prototypes(clean, source_tx_ids)
    ranks = parse_int_csv(args.ranks)
    alphas = parse_float_csv(args.alphas)
    modes = parse_csv(args.modes)
    run_ids = parse_csv(args.run_ids)

    rows = []
    for rank in ranks:
        model = fit_model(clean_x, leo_x, labels, source_tx_ids, int(rank))
        for mode in modes:
            for alpha in alphas:
                variant = f"LEOSUB1_{mode.upper()}_R{int(rank):02d}_A{int(round(alpha * 100)):03d}"
                for run_id in run_ids:
                    run_dir = args.runs_root / run_id
                    sat_npz = run_dir / str(args.sat_relpath)
                    clean_npz = run_dir / str(args.clean_relpath)
                    if not sat_npz.is_file():
                        rows.append({"run_id": run_id, "variant": variant, "status": "missing_sat"})
                        continue
                    out_dir = run_dir / variant
                    write_npz(sat_npz, out_dir / "features_leo_repaired.npz", prototypes, model, source_tx_ids, alpha=alpha, gate_scale=float(args.gate_scale), mode=mode, rank=int(rank), temperature=float(args.temperature), clean_control=False)
                    if clean_npz.is_file():
                        write_npz(clean_npz, out_dir / "features_clean_repaired.npz", prototypes, model, source_tx_ids, alpha=alpha, gate_scale=float(args.gate_scale), mode=mode, rank=int(rank), temperature=float(args.temperature), clean_control=True)
                    rows.append({"run_id": run_id, "variant": variant, "status": "ok"})

    summary = {
        "phase": "phase1_residual_subspace_repair_v24",
        "rows": rows,
        "source_pair_count": int(clean_x.shape[0]),
        "variants": sorted({r["variant"] for r in rows}),
        "uses_target_clean": False,
        "uses_target_labels": False,
        "uses_unknown_query_for_training": False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
