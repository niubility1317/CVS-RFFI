"""Single-candidate D92-Lite-FULL288 Target125 adapter.

The existing sealed D92 package materializer and Target125 I/O plane are
reused.  This adapter changes only the candidate core and method-lock
projection; it deliberately uses the standard 288-dimensional materializer,
with no additional backbone forward or query-side state.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import threading

from . import stage2_d108_target125_runner as d108_runner
from . import stage2_d92_lite_target125 as d131_adapter
from . import stage2_d92_full288_target125_core as core


CANDIDATE_ID = core.CANDIDATE_ID
PROTOCOL_SCHEMA = core.PROTOCOL_SCHEMA
ARMS = (core.TRANSPORT_ARM,)
PHASES = d108_runner.PHASES
SCENES = d108_runner.SCENES
SHARD_COUNT = d108_runner.SHARD_COUNT
OUTER_JOB_COUNT = d108_runner.OUTER_JOB_COUNT
SCENE_ROW_COUNT = d108_runner.SCENE_ROW_COUNT
ARM_PAIR_COUNT = SCENE_ROW_COUNT
SURFACE_COUNT = SCENE_ROW_COUNT * len(PHASES)

PREDICTION_MANIFEST_SCHEMA = (
    "cvs.phase2.d138.d92_lite_full288.target125.prediction_manifest.v1"
)
PREDICTION_ARTIFACT_SCHEMA = (
    "cvs.phase2.d138.d92_lite_full288.target125.prediction_artifact.v1"
)
PREDICTION_SHARD_SCHEMA = (
    "cvs.phase2.d138.d92_lite_full288.target125.prediction_shard.v1"
)
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d138.d92_lite_full288.target125.smoke_receipt.v1"
SMOKE_PREDICTIONS_SCHEMA = (
    "cvs.phase2.d138.d92_lite_full288.target125.smoke_predictions.v1"
)
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d138.d92_lite_full288.target125.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = (
    "cvs.phase2.d138.d92_lite_full288.target125.truth_open_event.v1"
)
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d138.d92_lite_full288.target125.score_manifest.v1"
FORMAL_ISOLATION_ENV = "CVS_D138_FULL288_FORMAL_ISOLATED_PROCESS"
_PROJECTION_LOCK = threading.RLock()


class D92Full288Target125Error(ValueError):
    """Raised when the full-288 Target125 adapter fails closed."""


def _verify_method_lock(
    method_lock_path: str | Path, expected_method_lock_sha256: str
) -> dict[str, Any]:
    source = Path(method_lock_path)
    if (
        source.is_symlink()
        or not source.is_file()
        or expected_method_lock_sha256 != core.METHOD_LOCK_SHA256
    ):
        raise D92Full288Target125Error("full-288 method-lock path or SHA drift")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != core.METHOD_LOCK_SHA256:
        raise D92Full288Target125Error("full-288 method-lock file SHA drift")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D92Full288Target125Error("full-288 method lock is not UTF-8 JSON") from error
    representation = document.get("representation") if isinstance(document, dict) else None
    matrix = document.get("matrix") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema") != core.METHOD_LOCK_SCHEMA
        or document.get("candidate_id") != CANDIDATE_ID
        or document.get("protocol_schema") != PROTOCOL_SCHEMA
        or not isinstance(matrix, dict)
        or matrix.get("prediction_surface_count") != SURFACE_COUNT
        or not isinstance(representation, dict)
        or representation.get("source_runtime_sha256") != core.SOURCE_RUNTIME_SHA256
    ):
        raise D92Full288Target125Error("full-288 method-lock identity or runtime drift")
    return dict(document)


def _require_formal_isolation() -> None:
    if os.environ.get(FORMAL_ISOLATION_ENV) != "1" or threading.active_count() != 1:
        raise D92Full288Target125Error(
            "formal full-288 projection requires a dedicated single-thread process"
        )


@contextmanager
def _base_projection():
    replacements = {
        "core": core,
        "CANDIDATE_ID": CANDIDATE_ID,
        "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
        "ARMS": ARMS,
        "ARM_PAIR_COUNT": ARM_PAIR_COUNT,
        "SURFACE_COUNT": SURFACE_COUNT,
        "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
        "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
        "PREDICTION_SHARD_SCHEMA": PREDICTION_SHARD_SCHEMA,
        "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
        "SMOKE_PREDICTIONS_SCHEMA": SMOKE_PREDICTIONS_SCHEMA,
        "TRUTH_CATALOG_SCHEMA": TRUTH_CATALOG_SCHEMA,
        "TRUTH_OPEN_EVENT_SCHEMA": TRUTH_OPEN_EVENT_SCHEMA,
        "SCORE_MANIFEST_SCHEMA": SCORE_MANIFEST_SCHEMA,
        "FORMAL_ISOLATION_ENV": FORMAL_ISOLATION_ENV,
        "TRUTH_PRIMARY_CANDIDATE_ARM": CANDIDATE_ID,
        "TRUTH_SYSTEM_DIAGNOSTIC_ONLY": True,
        "_verify_method_lock": _verify_method_lock,
    }
    with _PROJECTION_LOCK:
        original = {name: getattr(d131_adapter, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(d131_adapter, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(d131_adapter, name, value)


def prepare_d92_full288_target125_run(
    *, method_lock_path: str | Path, expected_method_lock_sha256: str, **kwargs: Any
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    with _base_projection():
        return d131_adapter.prepare_d92_lite_target125_run(
            method_lock_path=method_lock_path,
            expected_method_lock_sha256=expected_method_lock_sha256,
            **kwargs,
        )


def smoke_d92_full288_target125_prepared_state(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
    output_dir: Path,
    row_index: int = 0,
    scene_index: int = 0,
    device: str = "cpu",
    feature_batch_size: int = 64,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    _require_formal_isolation()
    with _base_projection():
        return d131_adapter.smoke_d92_lite_target125_prepared_state(
            plan_manifest_path=plan_manifest_path,
            expected_plan_file_sha256=expected_plan_file_sha256,
            context_manifest_path=context_manifest_path,
            expected_context_file_sha256=expected_context_file_sha256,
            output_dir=output_dir,
            row_index=row_index,
            scene_index=scene_index,
            device=device,
            feature_batch_size=feature_batch_size,
            method_lock_path=method_lock_path,
            expected_method_lock_sha256=expected_method_lock_sha256,
            state_materializer=None,
            pair_builder=core.build_d92_full288_pair,
            query_scorer=core.score,
        )


def predict_d92_full288_target125(
    *,
    plan_manifest_path: Path | None = None,
    expected_plan_file_sha256: str | None = None,
    context_manifest_path: Path | None = None,
    expected_context_file_sha256: str | None = None,
    output_dir: Path,
    device: str = "cpu",
    feature_batch_size: int = 64,
    method_lock_path: str | Path,
    expected_method_lock_sha256: str,
    shard_index: int | None = None,
    shard_manifest_paths: tuple[Path, ...] | list[Path] | None = None,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    _require_formal_isolation()
    with _base_projection():
        if shard_manifest_paths is not None:
            return d131_adapter.predict_d92_lite_target125(
                output_dir=output_dir,
                method_lock_path=method_lock_path,
                expected_method_lock_sha256=expected_method_lock_sha256,
                shard_manifest_paths=shard_manifest_paths,
            )
        if any(
            value is None
            for value in (
                plan_manifest_path,
                expected_plan_file_sha256,
                context_manifest_path,
                expected_context_file_sha256,
                shard_index,
            )
        ):
            raise D92Full288Target125Error("full-288 shard prediction requires prepared inputs")
        return d131_adapter.predict_d92_lite_target125(
            plan_manifest_path=Path(plan_manifest_path),
            expected_plan_file_sha256=str(expected_plan_file_sha256),
            context_manifest_path=Path(context_manifest_path),
            expected_context_file_sha256=str(expected_context_file_sha256),
            output_dir=output_dir,
            device=device,
            feature_batch_size=feature_batch_size,
            method_lock_path=method_lock_path,
            expected_method_lock_sha256=expected_method_lock_sha256,
            shard_index=int(shard_index),
            state_materializer=None,
            pair_builder=core.build_d92_full288_pair,
            query_scorer=core.score,
        )


def validate_d92_full288_prediction_manifest(**kwargs: Any) -> dict[str, Any]:
    with _base_projection():
        return d131_adapter.validate_d92_lite_target125_prediction_manifest(**kwargs)


def build_d92_full288_truth_catalog(**kwargs: Any) -> dict[str, Any]:
    with _base_projection():
        return d131_adapter.build_d92_lite_target125_truth_catalog(**kwargs)


def score_d92_full288_target125(**kwargs: Any) -> dict[str, Any]:
    with _base_projection():
        return d131_adapter.score_d92_lite_target125(**kwargs)


__all__ = [
    "CANDIDATE_ID",
    "D92Full288Target125Error",
    "FORMAL_ISOLATION_ENV",
    "PREDICTION_MANIFEST_SCHEMA",
    "SHARD_COUNT",
    "SURFACE_COUNT",
    "build_d92_full288_truth_catalog",
    "prepare_d92_full288_target125_run",
    "predict_d92_full288_target125",
    "score_d92_full288_target125",
    "smoke_d92_full288_target125_prepared_state",
    "validate_d92_full288_prediction_manifest",
]
