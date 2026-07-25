#!/usr/bin/env python3
"""Build the 49 source-only D102 diagnostic comparators before scorer access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    build_phase1_rb_metabias4_bundle,
    save_phase1_rb_metabias4_bundle,
)


STATUS = "DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE"
METHOD_LOCK_SHA256 = (
    "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f"
)
ORIGINAL_REJECTED_RECEIPT_SHA256 = (
    "01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2"
)
EXPECTED_KEYS = {
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: str, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _construction_code_sha256() -> str:
    module_path = ROOT / "cvsrffi" / "phase1_rb_metabias4_bundle.py"
    script_path = Path(__file__).resolve()
    return hashlib.sha256(
        (
            _sha256_file(module_path)
            + "\0"
            + _sha256_file(script_path)
        ).encode("ascii")
    ).hexdigest()


def _physical_root(
    physical_ids: np.ndarray,
    receiver_ids: np.ndarray,
    labels: np.ndarray,
    *,
    held_receiver: str,
    held_class: str | None,
) -> str:
    keep = receiver_ids != held_receiver
    if held_class is not None:
        keep &= labels != held_class
    payload = json.dumps(
        sorted(physical_ids[keep].astype(str).tolist()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-archive", type=Path, required=True)
    parser.add_argument("--labeled-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = args.labeled_archive.resolve(strict=True)
    manifest_path = args.labeled_manifest.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable comparator root exists: {output}")
    checkpoint_sha = _require_sha(args.checkpoint_sha256, "checkpoint")
    runtime_sha = _require_sha(args.runtime_sha256, "runtime")
    code_sha = _require_sha(args.code_sha256, "code")
    if code_sha != _construction_code_sha256():
        raise ValueError("D102 construction code SHA binding drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("role") != "L_s"
        or manifest.get("fraction") != 0.07
        or manifest.get("tx_visibility") != "visible"
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
        or manifest.get("archive_sha256") != _sha256_file(archive_path)
        or manifest.get("checkpoint_sha256") != checkpoint_sha
        or manifest.get("runtime_sha256") != runtime_sha
    ):
        raise ValueError("L_s manifest binding drift")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_KEYS:
            raise ValueError("L_s member closure drift")
        source = {name: np.array(archive[name], copy=True) for name in archive.files}
    labels = source["tx_labels"].astype(str)
    receivers = tuple(sorted(set(source["receiver_ids"].astype(str).tolist())))
    classes = tuple(sorted(set(labels.tolist())))
    if len(receivers) != 7 or len(classes) != 6:
        raise ValueError("D102 comparator plan requires exactly 7 receivers and 6 classes")
    tap = {
        "pre_relu": np.asarray(source["pre_relu"], dtype=np.float32),
        "z_dom": np.asarray(source["z_dom"], dtype=np.float32),
        "labels": labels,
        "receiver_ids": source["receiver_ids"].astype(str),
        "day_ids": source["day_ids"].astype(str),
        "physical_ids": source["physical_ids"].astype(str),
        "class_ids": np.asarray(classes, dtype=str),
    }
    output.mkdir(parents=True, exist_ok=False)
    fold_rows: list[dict[str, Any]] = []
    for receiver in receivers:
        specs = [(receiver, None), *[(receiver, class_id) for class_id in classes]]
        for held_receiver, held_class in specs:
            bundle = build_phase1_rb_metabias4_bundle(
                tap,
                checkpoint_sha256=checkpoint_sha,
                runtime_sha256=runtime_sha,
                method_lock_sha256=METHOD_LOCK_SHA256,
                excluded_receivers=(held_receiver,),
                excluded_classes=(() if held_class is None else (held_class,)),
            )
            fold_name = hashlib.sha256(
                f"{held_receiver}\0{held_class or ''}".encode("utf-8")
            ).hexdigest()[:20]
            saved = save_phase1_rb_metabias4_bundle(output / fold_name, bundle)
            fold_rows.append(
                {
                    "held_receiver": held_receiver,
                    "held_class": held_class,
                    "bundle_relative_path": fold_name,
                    "bundle_content_root_sha256": saved["content_root_sha256"],
                    "l_s_physical_root_sha256": _physical_root(
                        tap["physical_ids"],
                        tap["receiver_ids"],
                        labels,
                        held_receiver=held_receiver,
                        held_class=held_class,
                    ),
                    "query_rows_used_for_fit": 0,
                }
            )
    provenance = {
        "status": STATUS,
        "fold_count": len(fold_rows),
        "folds": fold_rows,
        "original_rejected_receipt_sha256": ORIGINAL_REJECTED_RECEIPT_SHA256,
        "method_lock_sha256": METHOD_LOCK_SHA256,
        "code_sha256": code_sha,
        "built_before_source_validation_open": True,
        "target_access": False,
        "formal_query_access": False,
    }
    if len(fold_rows) != 49:
        raise RuntimeError("D102 comparator build did not close at 49 folds")
    provenance_path = output / "d102_provenance.json"
    provenance_path.write_bytes(_canonical(provenance))
    print(provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
