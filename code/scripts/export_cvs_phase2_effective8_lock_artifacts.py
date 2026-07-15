#!/usr/bin/env python
"""Export source-locked head and TTA artifacts from a v14 candidate lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = Path(args.candidate_lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    if lock.get("schema") != "cvs_stage2c_source_candidate_lock_v2":
        raise ValueError("candidate lock schema drift")
    candidate = dict(lock.get("locked_candidate", {}))
    head_payload = {"schema": "cvs.symmetric_head_lock.v1", **dict(candidate["head"])}
    adaptive = dict(candidate["adaptive_tta"])
    thresholds = dict(adaptive["thresholds"])
    # v14 locked three active decision thresholds.  The current deployment
    # image persists six float32 slots, so encode the three historically absent
    # controls as exact no-ops instead of fitting anything on target queries.
    compatibility_defaults = {
        "base_stop_min_score": -1.0e9,
        "shift3_stop_min_score": -1.0e9,
        "fusion_std_penalty": 0.0,
    }
    tta_payload = {
        "schema": "cvs.phase2.adaptive_rxlight_tta.v1",
        "mode": "adaptive_1_3_5",
        "base_views": 1,
        "max_views": 5,
        "base_stop_margin": thresholds["base_stop_margin"],
        "shift3_stop_margin": thresholds["shift3_stop_margin"],
        "shift3_max_disagreement": thresholds["shift3_max_disagreement"],
        "base_stop_min_score": thresholds.get(
            "base_stop_min_score", compatibility_defaults["base_stop_min_score"]
        ),
        "shift3_stop_min_score": thresholds.get(
            "shift3_stop_min_score", compatibility_defaults["shift3_stop_min_score"]
        ),
        "fusion_std_penalty": thresholds.get(
            "fusion_std_penalty", compatibility_defaults["fusion_std_penalty"]
        ),
        "calibration_scope": "source_validation",
        "uses_query_labels": False,
        "uses_query_role": False,
        "uses_class_quota": False,
    }
    output = Path(args.out_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite lock-artifact root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    head = output / "symmetric_head_lock.json"
    tta = output / "adaptive_tta_policy.json"
    _write(head, head_payload)
    _write(tta, tta_payload)
    receipt = {
        "schema": "cvs.phase2.effective8_lock_artifact_export.v1",
        "status": "PASS",
        "candidate_lock_sha256": _sha(lock_path),
        "head_lock_sha256": _sha(head),
        "tta_policy_sha256": _sha(tta),
        "target_query_used": False,
        "compatibility_defaults": {
            key: value for key, value in compatibility_defaults.items() if key not in thresholds
        },
    }
    receipt_path = output / "lock_artifact_receipt.json"
    _write(receipt_path, receipt)
    return {"head_lock": str(head), "tta_policy": str(tta), "receipt": str(receipt_path), **receipt}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
