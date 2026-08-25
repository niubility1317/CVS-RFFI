#!/usr/bin/env python3
"""Build the immutable source-only M2.9 TASR48 aggregate bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cvsrffi.stage2_m29_tasr import (
    build_phase1_tasr_bundle,
    publish_phase1_tasr_bundle,
)
from scripts.build_m26_phase1_spectral_anchor import (
    _class_mapping_from_binding,
    _scalar_string,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--class-binding-json", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--rank", type=int, default=8)
    return parser


def build_from_source_npz(
    source_npz: str | Path,
    *,
    checkpoint_sha256: str,
    class_binding_json: str | Path,
    rank: int,
):
    path = Path(source_npz)
    with np.load(path, allow_pickle=False) as payload:
        required = {"fft_logmag_features", "raw_labels", "dataset_role", "rx_ids", "manifest_json"}
        if not required.issubset(payload.files):
            raise ValueError(f"source NPZ is missing {sorted(required - set(payload.files))}")
        fft = np.asarray(payload["fft_logmag_features"], dtype=np.float32)
        labels = np.asarray(payload["raw_labels"]).astype(str)
        roles = np.asarray(payload["dataset_role"]).astype(str)
        receivers = np.asarray(payload["rx_ids"]).astype(str)
        manifest = json.loads(_scalar_string(payload["manifest_json"], "manifest_json"))
        embedded_checkpoint = str(manifest["source_checkpoint_sha256"]).lower()
        if "source_checkpoint_sha256" in payload.files and _scalar_string(
            payload["source_checkpoint_sha256"], "source_checkpoint_sha256"
        ).lower() != embedded_checkpoint:
            raise ValueError("source checkpoint identities disagree")
    if not (len(fft) == len(labels) == len(roles) == len(receivers)):
        raise ValueError("source NPZ row geometry drift")
    selected = roles == "source"
    if not np.any(selected):
        raise ValueError("source NPZ has no source-only rows")
    if embedded_checkpoint != str(checkpoint_sha256).lower():
        raise ValueError("source NPZ/checkpoint identity drift")
    classes, raw_order, tx_order, binding_schema = _class_mapping_from_binding(
        class_binding_json,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha256,
    )
    if set(labels[selected].tolist()) != set(raw_order):
        raise ValueError("source label set does not match the Phase1 class binding")
    mapping = dict(zip(raw_order, classes))
    mapped_labels = np.asarray([mapping[value] for value in labels[selected].tolist()])
    bundle, audit = build_phase1_tasr_bundle(
        fft[selected],
        mapped_labels,
        receivers[selected],
        class_registry=classes,
        checkpoint_sha256=checkpoint_sha256,
        dataset_roles=roles[selected],
        rank=int(rank),
    )
    evidence = {
        **dict(audit),
        "source_npz": str(path.absolute()),
        "source_npz_total_rows": int(len(fft)),
        "source_selected_rows": int(np.sum(selected)),
        "non_source_rows_read_for_statistics": 0,
        "class_registry": list(bundle.class_registry),
        "receiver_registry": list(bundle.receiver_registry),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "class_binding_json": str(Path(class_binding_json).absolute()),
        "class_binding_schema": binding_schema,
        "source_label_order": list(raw_order),
        "phase1_tx_order": list(tx_order),
    }
    return bundle, evidence


def main() -> int:
    args = _parser().parse_args()
    bundle_path = Path(args.output_bundle).absolute()
    audit_path = Path(args.audit_json).absolute()
    if bundle_path.exists() or audit_path.exists():
        raise FileExistsError("M2.9 outputs are immutable and must not already exist")
    if bundle_path.parent != audit_path.parent:
        raise ValueError("M2.9 bundle and audit must share one control directory")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if not bundle_path.parent.is_dir():
        raise FileNotFoundError("M2.9 control directory could not be created")
    bundle, audit = build_from_source_npz(
        args.source_npz,
        checkpoint_sha256=args.checkpoint_sha256,
        class_binding_json=args.class_binding_json,
        rank=args.rank,
    )
    publish_phase1_tasr_bundle(bundle_path, bundle)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PHASE1_TASR_BUNDLE_COMPLETE", **audit}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
