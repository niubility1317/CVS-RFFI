#!/usr/bin/env python
"""Derive the only strict runtime configs from an externally trusted v14 capsule."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_candidate_capsule import (  # noqa: E402
    load_and_validate_candidate_capsule,
    sha256_file,
)


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"runtime-config input must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"runtime-config input must be a JSON object: {path}")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    capsule_path = Path(args.candidate_capsule)
    candidate_lock_path = Path(args.candidate_lock)
    capsule = load_and_validate_candidate_capsule(
        capsule_path,
        expected_capsule_sha256=str(args.expected_candidate_capsule_sha256).lower(),
        candidate_lock_path=candidate_lock_path,
    )
    source_stats_path = Path(args.source_feature_stats)
    tta_path = Path(args.tta_policy)
    if sha256_file(source_stats_path) != capsule["source_feature_stats"]["sha256"]:
        raise ValueError("source-feature statistics do not match the trusted capsule")
    if sha256_file(tta_path) != capsule["tta_policy"]["sha256"]:
        raise ValueError("TTA policy does not match the trusted capsule")
    candidate_lock = _read_json(candidate_lock_path)
    candidate = dict(candidate_lock["locked_candidate"])
    selected = dict(dict(candidate["head"])["selected"])
    # v14 predates the two optional symmetric-head controls.  Zero is the
    # exact identity/no-penalty representation and introduces no fitted state.
    selected.setdefault("gram_mix", 0.0)
    selected.setdefault("uncertainty_penalty", 0.0)
    with np.load(source_stats_path, allow_pickle=False) as payload:
        mean = np.asarray(payload["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(payload["std"], dtype=np.float32).reshape(-1)
        fft_dim = int(np.asarray(payload["fft_dim"]).reshape(-1)[0])
        fft_weight = float(np.asarray(payload["fft_weight"]).reshape(-1)[0])
    if mean.shape != std.shape or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("source-feature statistics are malformed")
    if fft_dim != 96 or fft_weight != 2.0:
        raise ValueError("effective8 runtime requires locked FFT96 weight2")
    resource = dict(capsule["resource_accounting"])
    adapter = {
        "schema": "cvs.phase2.feature_adapter.v1",
        "mode": "identity",
        "trainable_parameters": int(resource["adapter_trainable_parameters"]),
        "adapt_epochs": int(resource["ground_adapter_train_epochs"]),
        "persistent_state_bytes": int(resource["deployment_incremental_persistent_bytes"]),
        "fft_dim": fft_dim,
        "fft_weight": fft_weight,
    }
    head = {
        "schema": "cvs.phase2.symmetric_locked_head.v1",
        "mode": "three_leo_support_symmetric_locked",
        "selected": selected,
        "source_feature_mean": mean.tolist(),
        "source_feature_std": std.tolist(),
        "variance_floor": 0.05,
        "storage_dtype": "fp16",
    }
    tta = _read_json(tta_path)
    required_tta = {
        "schema",
        "mode",
        "base_views",
        "max_views",
        "base_stop_margin",
        "shift3_stop_margin",
        "shift3_max_disagreement",
        "base_stop_min_score",
        "shift3_stop_min_score",
        "fusion_std_penalty",
        "calibration_scope",
        "uses_query_labels",
        "uses_query_role",
        "uses_class_quota",
    }
    if set(tta) != required_tta:
        raise ValueError("trusted TTA policy exact schema drift")
    output = Path(args.out_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime-config root: {output}")
    output.mkdir(parents=True, exist_ok=False)
    adapter_path = output / "effective8_runtime_adapter.json"
    head_path = output / "effective8_runtime_head.json"
    tta_out = output / "effective8_runtime_tta.json"
    _write_new(adapter_path, adapter)
    _write_new(head_path, head)
    _write_new(tta_out, tta)
    receipt = {
        "schema": "cvs.phase2.effective8_runtime_config_receipt.v1",
        "status": "PASS",
        "candidate_capsule_sha256": sha256_file(capsule_path),
        "candidate_lock_sha256": sha256_file(candidate_lock_path),
        "source_feature_stats_sha256": sha256_file(source_stats_path),
        "adapter_config_sha256": sha256_file(adapter_path),
        "head_config_sha256": sha256_file(head_path),
        "tta_config_sha256": sha256_file(tta_out),
        "derivation_uses_target_query": False,
    }
    receipt_path = output / "runtime_config_receipt.json"
    _write_new(receipt_path, receipt)
    return {
        "runtime_config_root": str(output),
        "adapter": str(adapter_path),
        "head": str(head_path),
        "tta": str(tta_out),
        "receipt": str(receipt_path),
        **receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-capsule", type=Path, required=True)
    parser.add_argument("--expected-candidate-capsule-sha256", required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--source-feature-stats", type=Path, required=True)
    parser.add_argument("--tta-policy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
