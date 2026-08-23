#!/usr/bin/env python3
"""Build the source-only checkpoint-bound six-class M2.6 anchor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cvsrffi.stage2_m26_spectral_anchor import (
    IDENTITY_DIM,
    FFT_DIM,
    build_phase1_spectral_anchor,
    publish_m26_spectral_anchor,
)


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _class_mapping_from_binding(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    checkpoint_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    binding_path = Path(path).absolute()
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    entries = binding.get("entries")
    logit_order = manifest.get("logit_class_order")
    class_id_to_tx = manifest.get("class_id_to_tx")
    if (
        str(binding.get("checkpoint_sha256", "")).lower()
        != str(checkpoint_sha256).lower()
        or not isinstance(entries, list)
        or len(entries) != 6
        or not isinstance(logit_order, list)
        or len(logit_order) != 6
        or not isinstance(class_id_to_tx, list)
        or len(class_id_to_tx) < 6
    ):
        raise ValueError("Phase1 class binding/checkpoint geometry drift")
    ordered_entries = sorted(entries, key=lambda item: int(item.get("class_index", -1)))
    registry: list[str] = []
    raw_order: list[str] = []
    tx_order: list[str] = []
    for class_index, (entry, raw_label) in enumerate(zip(ordered_entries, logit_order)):
        direct_index = int(entry.get("direct_logit_index", -1))
        if (
            int(entry.get("class_index", -1)) != class_index
            or direct_index != int(raw_label)
            or direct_index < 0
            or direct_index >= len(class_id_to_tx)
            or str(entry.get("phase1_tx", "")) != str(class_id_to_tx[direct_index])
        ):
            raise ValueError("Phase1 class binding does not match export manifest")
        registry.append(str(entry.get("registered_class_handle", "")))
        raw_order.append(str(raw_label))
        tx_order.append(str(class_id_to_tx[direct_index]))
    if (
        len(set(registry)) != 6
        or any(not item for item in registry)
        or len(set(raw_order)) != 6
        or len(set(tx_order)) != 6
    ):
        raise ValueError("Phase1 class binding contains duplicate or empty identities")
    return (
        tuple(registry),
        tuple(raw_order),
        tuple(tx_order),
        str(binding.get("schema", "")),
    )


def build_from_source_npz(
    *,
    source_npz: str | Path,
    output_path: str | Path,
    audit_path: str | Path,
    checkpoint_sha256: str,
    class_binding_json: str | Path,
) -> dict[str, Any]:
    source = Path(source_npz).absolute()
    output = Path(output_path).absolute()
    audit_output = Path(audit_path).absolute()
    if output.exists() or audit_output.exists():
        raise FileExistsError("M2.6 anchor output is non-overwriting")
    if not output.parent.is_dir() or not audit_output.parent.is_dir():
        raise FileNotFoundError("M2.6 anchor output parent is missing")
    with np.load(source, allow_pickle=False) as arrays:
        required = {"features", "fft_logmag_features", "raw_labels", "dataset_role"}
        if not required.issubset(arrays.files):
            raise ValueError("source feature NPZ lacks required M2.6 members")
        identity = np.asarray(arrays["features"], dtype=np.float32)
        fft = np.asarray(arrays["fft_logmag_features"], dtype=np.float32)
        raw_labels = np.asarray(arrays["raw_labels"]).astype(str)
        roles = np.asarray(arrays["dataset_role"]).astype(str)
        if "manifest_json" not in arrays.files:
            raise ValueError("source feature NPZ lacks required manifest_json identity")
        manifest = json.loads(_scalar_string(arrays["manifest_json"], "manifest_json"))
        embedded_checkpoint = str(manifest["source_checkpoint_sha256"])
        if "source_checkpoint_sha256" in arrays.files and _scalar_string(
            arrays["source_checkpoint_sha256"], "source_checkpoint_sha256"
        ).lower() != embedded_checkpoint.lower():
            raise ValueError("source checkpoint identities disagree")
    registry, raw_order, tx_order, binding_schema = _class_mapping_from_binding(
        class_binding_json,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha256,
    )
    if (
        identity.ndim != 2
        or identity.shape[1] != IDENTITY_DIM
        or fft.shape != (len(identity), FFT_DIM)
        or raw_labels.shape != (len(identity),)
        or roles.shape != (len(identity),)
        or not np.all(np.isfinite(identity))
        or not np.all(np.isfinite(fft))
        or embedded_checkpoint.lower() != str(checkpoint_sha256).lower()
    ):
        raise ValueError("source feature/checkpoint geometry drift")
    source_mask = roles == "source"
    if not np.any(source_mask):
        raise ValueError("source feature NPZ contains no source rows")
    source_identity = identity[source_mask]
    source_fft = fft[source_mask]
    source_raw_labels = raw_labels[source_mask]
    source_roles = roles[source_mask]
    if set(source_raw_labels.tolist()) != set(raw_order):
        raise ValueError("source label set does not match source label order")
    mapping = dict(zip(raw_order, registry))
    labels = np.asarray([mapping[item] for item in source_raw_labels.tolist()])
    rows = np.concatenate([source_identity, source_fft], axis=1)
    component, build_audit = build_phase1_spectral_anchor(
        rows,
        labels,
        class_registry=registry,
        checkpoint_sha256=str(checkpoint_sha256).lower(),
        dataset_roles=source_roles,
    )
    audit = {
        "schema": "cvs.erbt_idr.m26.phase1_spectral_anchor_publish_audit.v1",
        "status": "VERIFIED",
        "source_npz": str(source),
        "output_path": str(output),
        "checkpoint_sha256": component.checkpoint_sha256,
        "component_id": component.component_id,
        "class_registry": list(registry),
        "source_label_order": list(raw_order),
        "phase1_tx_order": list(tx_order),
        "class_binding_json": str(Path(class_binding_json).absolute()),
        "class_binding_schema": binding_schema,
        "input_row_count": int(len(identity)),
        "input_target_row_count": int(np.count_nonzero(~source_mask)),
        "query_rows_used": 0,
        "target_rows_used": 0,
        **dict(build_audit),
    }
    publish_m26_spectral_anchor(output, component)
    _write_json_exclusive(audit_output, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--class-binding-json", required=True)
    args = parser.parse_args()
    audit = build_from_source_npz(
        source_npz=args.source_npz,
        output_path=args.output,
        audit_path=args.audit_output,
        checkpoint_sha256=args.checkpoint_sha256,
        class_binding_json=args.class_binding_json,
    )
    print(json.dumps({"status": audit["status"], "source_row_count": audit["source_row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
