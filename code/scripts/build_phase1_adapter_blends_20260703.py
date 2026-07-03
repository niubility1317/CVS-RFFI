#!/usr/bin/env python
"""Build conservative identity/adapter feature blends for target1 audit.

The blend is a post-export source-protocol diagnostic: it does not use target
labels or target clean data to choose the blend. Target labels are used only by
the downstream strong target1 evaluator.
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


def load_npz(path: Path) -> dict[str, Any]:
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


def blend_arrays(
    base_arrays: dict[str, np.ndarray],
    repair_arrays: dict[str, np.ndarray],
    prototypes: np.ndarray,
    blend: float,
    proto_temperature: float,
    manifest_patch: dict[str, Any],
) -> dict[str, np.ndarray]:
    base_features = np.asarray(base_arrays["features"], dtype=np.float32)
    repair_features = np.asarray(repair_arrays["features"], dtype=np.float32)
    if base_features.shape != repair_features.shape:
        raise ValueError(f"feature shape mismatch: {base_features.shape} != {repair_features.shape}")
    out = dict(base_arrays)
    blended = ((1.0 - float(blend)) * base_features + float(blend) * repair_features).astype(np.float32)
    out["features"] = blended
    out["tx_logits"] = proto_logits(blended, prototypes, proto_temperature)
    manifest: dict[str, Any] = {}
    if "manifest_json" in base_arrays:
        try:
            manifest = json.loads(str(np.asarray(base_arrays["manifest_json"]).item()))
        except Exception:
            manifest = {}
    manifest.update(manifest_patch)
    out["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
    return out


def blend_tag(value: float) -> str:
    return f"BLEND{int(round(float(value) * 1000)):03d}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", required=True)
    p.add_argument("--source_clean_npz", required=True)
    p.add_argument("--run_ids", required=True)
    p.add_argument("--source_tx_ids", default=",".join(SOURCE_TX_IDS))
    p.add_argument("--variants", required=True)
    p.add_argument("--blends", default="0.10,0.20,0.35,0.50,0.65,0.80")
    p.add_argument("--sat_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz")
    p.add_argument("--clean_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz")
    p.add_argument("--repaired_relpath", default="{variant}/features_leo_repaired.npz")
    p.add_argument("--clean_repaired_relpath", default="{variant}/features_clean_repaired.npz")
    p.add_argument("--proto_temperature", type=float, default=0.07)
    p.add_argument("--manifest_label", default="phase1_v20_identity_adapter_blend")
    args = p.parse_args(argv)

    runs_root = Path(args.runs_root)
    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    variants = parse_csv(args.variants)
    blends = parse_float_csv(args.blends)
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
                rows.append({"run_id": run_id, "variant": variant, "status": "missing_repaired_npz"})
                continue
            repair = load_npz(repair_path)
            clean_repair = load_npz(clean_repair_path)
            for blend in blends:
                tag = blend_tag(blend)
                out_variant = f"{variant}_{tag}"
                out_dir = run_dir / out_variant
                out_dir.mkdir(parents=True, exist_ok=True)
                patch = {
                    "leo_feature_adapter": {
                        "enabled": True,
                        "adapter_kind": "identity_adapter_feature_blend",
                        "base_variant": variant,
                        "blend_weight": float(blend),
                        "training_scope": "post_export_blend_no_target_label_fit",
                        "uses_target_clean": False,
                        "uses_target_labels": False,
                        "uses_unknown_query_for_training": False,
                        "logits": "cosine_to_source_clean_prototypes_after_blend",
                    }
                }
                sat_out = blend_arrays(sat_base, repair, prototypes, blend, float(args.proto_temperature), patch)
                clean_out = blend_arrays(
                    clean_base,
                    clean_repair,
                    prototypes,
                    blend,
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
                        "blend": float(blend),
                        "status": "written",
                        "sat_out": str(out_dir / "features_leo_repaired.npz"),
                        "clean_out": str(out_dir / "features_clean_repaired.npz"),
                    }
                )
    print(json.dumps({"rows": rows, "written": sum(1 for r in rows if r.get("status") == "written")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
