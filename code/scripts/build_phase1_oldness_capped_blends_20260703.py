#!/usr/bin/env python
"""Build oldness-capped repair features for target1 audit.

For each sample, use the repaired/blended feature only when its source-prototype
max logit does not exceed the identity feature's max logit by more than a cap.
This is label-free and is intended to prevent unknown oldness/FAR worsening.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SOURCE_TX_IDS = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(x) for x in parse_csv(text)]


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x) for x in parse_csv(text)]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if "features" not in arrays:
        raise ValueError(f"{path} missing features")
    return arrays


def as_tx(arr: np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(arr)
    if value.shape == ():
        return np.asarray([canonical_tx_id(value.item())] * n, dtype=str)
    if value.shape[0] != n:
        raise ValueError(f"tx length mismatch: {value.shape[0]} != {n}")
    return np.asarray([canonical_tx_id(x) for x in value.reshape(-1)], dtype=str)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-6)


def make_prototypes(clean_npz: Path, source_tx_ids: list[str]) -> np.ndarray:
    arrays = load_npz(clean_npz)
    features = np.asarray(arrays["features"], dtype=np.float32)
    tx = as_tx(arrays["tx_ids"], int(features.shape[0]))
    protos = []
    for tx_id in source_tx_ids:
        mask = tx == canonical_tx_id(tx_id)
        if not bool(mask.any()):
            raise ValueError(f"no source clean prototype rows for tx={tx_id}")
        proto = features[mask].mean(axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1.0e-6)
        protos.append(proto.astype(np.float32))
    return np.stack(protos, axis=0)


def proto_logits(features: np.ndarray, prototypes: np.ndarray, temperature: float) -> np.ndarray:
    return (l2n(features.astype(np.float32)) @ l2n(prototypes.astype(np.float32)).T / max(float(temperature), 1.0e-6)).astype(np.float32)


def cap_tag(value: float) -> str:
    return f"CAP{int(round(float(value) * 1000)):03d}"


def gate_arrays(
    base_arrays: dict[str, np.ndarray],
    candidate_arrays: dict[str, np.ndarray],
    prototypes: np.ndarray,
    cap: float,
    proto_temperature: float,
    manifest_patch: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    base_features = np.asarray(base_arrays["features"], dtype=np.float32)
    cand_features = np.asarray(candidate_arrays["features"], dtype=np.float32)
    if base_features.shape != cand_features.shape:
        raise ValueError(f"feature shape mismatch: {base_features.shape} != {cand_features.shape}")
    base_logits = proto_logits(base_features, prototypes, proto_temperature)
    cand_logits = proto_logits(cand_features, prototypes, proto_temperature)
    gate = cand_logits.max(axis=1) <= (base_logits.max(axis=1) + float(cap))
    out_features = np.where(gate[:, None], cand_features, base_features).astype(np.float32)
    out = dict(base_arrays)
    out["features"] = out_features
    out["tx_logits"] = proto_logits(out_features, prototypes, proto_temperature)
    manifest: dict[str, Any] = {}
    if "manifest_json" in base_arrays:
        try:
            manifest = json.loads(str(np.asarray(base_arrays["manifest_json"]).item()))
        except Exception:
            manifest = {}
    manifest.update(manifest_patch)
    out["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
    stats = {
        "gate_accept_rate": float(gate.mean()) if gate.size else float("nan"),
        "mean_base_oldness": float(base_logits.max(axis=1).mean()) if gate.size else float("nan"),
        "mean_candidate_oldness": float(cand_logits.max(axis=1).mean()) if gate.size else float("nan"),
        "mean_output_oldness": float(out["tx_logits"].max(axis=1).mean()) if gate.size else float("nan"),
    }
    return out, stats


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", required=True)
    p.add_argument("--source_clean_npz", required=True)
    p.add_argument("--run_ids", required=True)
    p.add_argument("--source_tx_ids", default=",".join(SOURCE_TX_IDS))
    p.add_argument("--candidate_variants", required=True)
    p.add_argument("--caps", default="0.00,0.05")
    p.add_argument("--sat_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz")
    p.add_argument("--clean_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz")
    p.add_argument("--repaired_relpath", default="{variant}/features_leo_repaired.npz")
    p.add_argument("--clean_repaired_relpath", default="{variant}/features_clean_repaired.npz")
    p.add_argument("--proto_temperature", type=float, default=0.07)
    args = p.parse_args(argv)

    runs_root = Path(args.runs_root)
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    variants = parse_csv(args.candidate_variants)
    caps = parse_float_csv(args.caps)
    run_ids = parse_csv(args.run_ids)
    prototypes = make_prototypes(Path(args.source_clean_npz), source_tx_ids)

    rows = []
    for run_id in run_ids:
        run_dir = runs_root / run_id
        sat_path = run_dir / args.sat_relpath
        clean_path = run_dir / args.clean_relpath
        if not sat_path.exists() or not clean_path.exists():
            rows.append({"run_id": run_id, "status": "missing_identity_npz"})
            continue
        sat_base = load_npz(sat_path)
        clean_base = load_npz(clean_path)
        for variant in variants:
            repair_path = run_dir / args.repaired_relpath.format(variant=variant)
            clean_repair_path = run_dir / args.clean_repaired_relpath.format(variant=variant)
            if not repair_path.exists() or not clean_repair_path.exists():
                rows.append({"run_id": run_id, "variant": variant, "status": "missing_candidate_npz"})
                continue
            repair = load_npz(repair_path)
            clean_repair = load_npz(clean_repair_path)
            for cap in caps:
                tag = cap_tag(cap)
                out_variant = f"{variant}_{tag}"
                out_dir = run_dir / out_variant
                out_dir.mkdir(parents=True, exist_ok=True)
                patch = {
                    "leo_feature_adapter": {
                        "enabled": True,
                        "adapter_kind": "oldness_capped_identity_fallback",
                        "candidate_variant": variant,
                        "oldness_cap": float(cap),
                        "training_scope": "post_export_gate_no_target_label_fit",
                        "uses_target_clean": False,
                        "uses_target_labels": False,
                        "uses_unknown_query_for_training": False,
                        "logits": "cosine_to_source_clean_prototypes_after_oldness_cap_gate",
                    }
                }
                sat_out, sat_stats = gate_arrays(sat_base, repair, prototypes, cap, float(args.proto_temperature), patch)
                clean_out, clean_stats = gate_arrays(
                    clean_base,
                    clean_repair,
                    prototypes,
                    cap,
                    float(args.proto_temperature),
                    {"leo_feature_adapter_clean_control": {**patch["leo_feature_adapter"], "clean_apply_npz": str(clean_path)}},
                )
                np.savez(out_dir / "features_leo_repaired.npz", **sat_out)
                np.savez(out_dir / "features_clean_repaired.npz", **clean_out)
                rows.append(
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "out_variant": out_variant,
                        "cap": float(cap),
                        "status": "written",
                        "sat_gate_accept_rate": sat_stats["gate_accept_rate"],
                        "clean_gate_accept_rate": clean_stats["gate_accept_rate"],
                    }
                )
    print(json.dumps({"rows": rows, "written": sum(1 for r in rows if r.get("status") == "written")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
