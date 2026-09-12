"""Phase2-external truth sidecar helpers; never import this inside predictor code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cvsrffi.stage2_predictor_bundle import sha256_file


SCORING_MANIFEST_SCHEMA = "cvs.phase2.scoring_sidecar_manifest.v2"


def load_verified_scoring_sidecar(path: str | Path):
    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("scoring manifest must be a regular non-symlink file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    required = {
        "schema", "predictor_package_root_sha256", "predictor_package_seal_sha256",
        "truth_sidecar_json", "truth_sidecar_sha256",
        "scorer_output_must_not_feed_predictor",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("scoring manifest exact schema drift")
    if payload.get("schema") != SCORING_MANIFEST_SCHEMA:
        raise ValueError("scoring manifest schema drift")
    if payload.get("scorer_output_must_not_feed_predictor") is not True:
        raise ValueError("scorer feedback guard missing")
    truth_path = manifest_path.parent / str(payload["truth_sidecar_json"])
    if truth_path.is_symlink() or not truth_path.is_file():
        raise ValueError("truth sidecar must be a regular non-symlink file")
    if sha256_file(truth_path) != str(payload["truth_sidecar_sha256"]):
        raise ValueError("truth sidecar hash mismatch")
    truth = json.loads(truth_path.read_text(encoding="utf-8-sig"))
    if not isinstance(truth, dict) or not isinstance(truth.get("rows"), list):
        raise ValueError("truth sidecar payload drift")
    return truth, payload, {
        "scoring_manifest": str(manifest_path),
        "scoring_manifest_sha256": sha256_file(manifest_path),
        "truth_sidecar": str(truth_path),
        "truth_sidecar_sha256": sha256_file(truth_path),
    }
