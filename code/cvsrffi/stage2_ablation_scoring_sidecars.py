"""Stage-scoped scorer-side publication for the full Phase2 ablation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.stage2_metric_scorer import (
    SCORING_MANIFEST_SCHEMA,
    TRUTH_SIDECAR_SCHEMA,
    _validate_truth_rows,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


class Stage2AblationScoringSidecarError(ValueError):
    """Raised when a stage-scoped scorer sidecar cannot be sealed."""


def _exclusive_readonly_json(
    path: Path, payload: Mapping[str, Any]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while sealing scoring sidecar")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)
    return sha256_bytes(data)


def publish_stage2a_scoring_sidecar(
    *,
    source_stage2b_truth_path: str | Path,
    expected_source_truth_sha256: str,
    predictor_package_root_sha256: str,
    predictor_package_seal_sha256: str,
    output_root: str | Path,
) -> dict[str, Any]:
    """Reuse sealed Stage2-B truth rows with only the top-level stage changed."""

    source = Path(source_stage2b_truth_path)
    if sha256_file(source) != str(expected_source_truth_sha256).lower():
        raise Stage2AblationScoringSidecarError(
            "source Stage2-B truth detached hash mismatch"
        )
    truth = json.loads(source.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(truth, dict)
        or truth.get("schema") != TRUTH_SIDECAR_SCHEMA
        or truth.get("stage") != "stage2b"
    ):
        raise Stage2AblationScoringSidecarError(
            "source truth is not a Stage2-B scorer sidecar"
        )
    stage2a_truth = dict(truth)
    stage2a_truth["stage"] = "stage2a"
    _validate_truth_rows(stage2a_truth)

    root = Path(output_root).absolute()
    truth_path = root / "truth_sidecar.json"
    manifest_path = root / "scoring_manifest.json"
    truth_sha256 = _exclusive_readonly_json(
        truth_path, stage2a_truth
    )
    manifest = {
        "schema": SCORING_MANIFEST_SCHEMA,
        "predictor_package_root_sha256": str(
            predictor_package_root_sha256
        ).lower(),
        "predictor_package_seal_sha256": str(
            predictor_package_seal_sha256
        ).lower(),
        "truth_sidecar_json": truth_path.name,
        "truth_sidecar_sha256": truth_sha256,
        "scorer_output_must_not_feed_predictor": True,
    }
    manifest_sha256 = _exclusive_readonly_json(
        manifest_path, manifest
    )
    return {
        "stage": "stage2a",
        "truth_sidecar_path": str(truth_path),
        "truth_sidecar_sha256": truth_sha256,
        "scoring_manifest_path": str(manifest_path),
        "scoring_manifest_sha256": manifest_sha256,
        "source_stage2b_truth_sha256": str(
            expected_source_truth_sha256
        ).lower(),
        "truth_rows_reused_without_data_revalidation": True,
    }


__all__ = [
    "Stage2AblationScoringSidecarError",
    "publish_stage2a_scoring_sidecar",
]
