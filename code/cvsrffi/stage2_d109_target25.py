"""Thin D109-SCRC adapter over the verified D108 Target25 plane.

The D108 frozen seed-713102 matrix, prepared inputs, real materializer,
immutable publication, and independent truth scorer remain unchanged.  Only
the already-frozen D109 pair/score core and D109 Target25 artifact identity are
projected into that plane.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import stage2_d108_target25 as d108_target25
from . import stage2_d109_d92_core as d109_core
from . import stage2_d109_target125 as d109_target125


CANDIDATE_ID = d109_target125.CANDIDATE_ID
PROTOCOL_SCHEMA = d108_target25.PROTOCOL_SCHEMA
FROZEN_SEED = d108_target25.FROZEN_SEED
ARMS = d108_target25.ARMS
PHASES = d108_target25.PHASES
SCENES = d108_target25.SCENES
RECEIVERS = d108_target25.RECEIVERS
SLICES = d108_target25.SLICES
SHARD_COUNT = d108_target25.SHARD_COUNT
OUTER_JOB_COUNT = d108_target25.OUTER_JOB_COUNT
SCENE_ROW_COUNT = d108_target25.SCENE_ROW_COUNT
ARM_PAIR_COUNT = d108_target25.ARM_PAIR_COUNT
SURFACE_COUNT = d108_target25.SURFACE_COUNT
TRUTH_SURFACE_COUNT = d108_target25.TRUTH_SURFACE_COUNT

PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d109.scrc.target25.prediction_manifest.v1"
PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.d109.scrc.target25.prediction_artifact.v1"
PREDICTION_SHARD_SCHEMA = "cvs.phase2.d109.scrc.target25.prediction_shard.v1"
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d109.scrc.target25.smoke_receipt.v1"
SMOKE_PREDICTIONS_SCHEMA = "cvs.phase2.d109.scrc.target25.smoke_predictions.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d109.scrc.target25.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d109.scrc.target25.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d109.scrc.target25.score_manifest.v1"

StateMaterializer = d108_target25.StateMaterializer
PairBuilder = d108_target25.PairBuilder
QueryScorer = d108_target25.QueryScorer

_RUNNER_REPLACEMENTS = {
    **d108_target25._RUNNER_REPLACEMENTS,
    "CANDIDATE_ID": CANDIDATE_ID,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "PREDICTION_SHARD_SCHEMA": PREDICTION_SHARD_SCHEMA,
    "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
    "SMOKE_PREDICTIONS_SCHEMA": SMOKE_PREDICTIONS_SCHEMA,
}
_TRUTH_REPLACEMENTS = {
    **d108_target25._TRUTH_REPLACEMENTS,
    "CANDIDATE_ID": CANDIDATE_ID,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "TRUTH_CATALOG_SCHEMA": TRUTH_CATALOG_SCHEMA,
    "TRUTH_OPEN_EVENT_SCHEMA": TRUTH_OPEN_EVENT_SCHEMA,
    "SCORE_MANIFEST_SCHEMA": SCORE_MANIFEST_SCHEMA,
}
_D108_TARGET25_REPLACEMENTS = {
    "_RUNNER_REPLACEMENTS": _RUNNER_REPLACEMENTS,
    "_TRUTH_REPLACEMENTS": _TRUTH_REPLACEMENTS,
}
_D109_SMOKE_REPLACEMENTS = {
    "OUTER_JOB_COUNT": OUTER_JOB_COUNT,
    "SURFACE_COUNT": SURFACE_COUNT,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "PREDICTION_SHARD_SCHEMA": PREDICTION_SHARD_SCHEMA,
    "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
    "SMOKE_PREDICTIONS_SCHEMA": SMOKE_PREDICTIONS_SCHEMA,
    "TRUTH_CATALOG_SCHEMA": TRUTH_CATALOG_SCHEMA,
    "TRUTH_OPEN_EVENT_SCHEMA": TRUTH_OPEN_EVENT_SCHEMA,
    "SCORE_MANIFEST_SCHEMA": SCORE_MANIFEST_SCHEMA,
    "_prepared_inputs": d108_target25._prepared_target25_inputs,
}


class D109Target25Error(ValueError):
    """Raised when the D109 projection cannot preserve the Target25 plane."""


def prepare_d109_target25_run(**kwargs: Any) -> dict[str, Any]:
    """Reuse the D108 Target25 preparation without data revalidation."""

    try:
        return d108_target25.prepare_d108_target25_run(**kwargs)
    except d108_target25.D108Target25Error as error:
        raise D109Target25Error("D109 Target25 prepare reuse failed closed") from error


def smoke_d109_target25_prepared_state(
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
    state_materializer: StateMaterializer | None = None,
    pair_builder: PairBuilder | None = None,
    query_scorer: QueryScorer | None = None,
    d92_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Smoke one frozen Target25 row with the submitted D109 core."""

    try:
        with d109_target125._temporary_identity(
            d109_target125, _D109_SMOKE_REPLACEMENTS
        ):
            return d109_target125.smoke_d109_target125_prepared_state(
                plan_manifest_path=plan_manifest_path,
                expected_plan_file_sha256=expected_plan_file_sha256,
                context_manifest_path=context_manifest_path,
                expected_context_file_sha256=expected_context_file_sha256,
                output_dir=output_dir,
                row_index=row_index,
                scene_index=scene_index,
                device=device,
                feature_batch_size=feature_batch_size,
                state_materializer=state_materializer,
                pair_builder=pair_builder,
                query_scorer=query_scorer,
                d92_fit=d92_fit,
            )
    except (
        d108_target25.D108Target25Error,
        d109_target125.D109Target125Error,
        d109_core.D109D92CoreError,
    ) as error:
        raise D109Target25Error("D109 Target25 smoke failed closed") from error


def predict_d109_target25(
    *,
    plan_manifest_path: Path | None = None,
    expected_plan_file_sha256: str | None = None,
    context_manifest_path: Path | None = None,
    expected_context_file_sha256: str | None = None,
    output_dir: Path,
    device: str = "cpu",
    feature_batch_size: int = 64,
    shard_index: int | None = None,
    shard_manifest_paths: Sequence[Path] | None = None,
    state_materializer: StateMaterializer | None = None,
    pair_builder: PairBuilder | None = None,
    query_scorer: QueryScorer | None = None,
    d92_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Publish one D109 Target25 shard or merge exactly eight shards."""

    try:
        builder = pair_builder
        scorer = query_scorer
        if shard_manifest_paths is None:
            builder, scorer = d109_target125._resolve_d109_core(
                pair_builder, query_scorer
            )
        with d109_target125._temporary_identity(
            d108_target25, _D108_TARGET25_REPLACEMENTS
        ):
            return d108_target25.predict_d108_target25(
                plan_manifest_path=plan_manifest_path,
                expected_plan_file_sha256=expected_plan_file_sha256,
                context_manifest_path=context_manifest_path,
                expected_context_file_sha256=expected_context_file_sha256,
                output_dir=output_dir,
                device=device,
                feature_batch_size=feature_batch_size,
                shard_index=shard_index,
                shard_manifest_paths=shard_manifest_paths,
                state_materializer=state_materializer,
                pair_builder=builder,
                query_scorer=scorer,
                d92_fit=d92_fit,
            )
    except (
        d108_target25.D108Target25Error,
        d109_target125.D109Target125Error,
        d109_core.D109D92CoreError,
    ) as error:
        raise D109Target25Error("D109 Target25 publication failed closed") from error


def validate_d109_target25_prediction_manifest(
    *,
    prediction_manifest_path: Path,
    expected_prediction_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        with d109_target125._temporary_identity(
            d108_target25, _D108_TARGET25_REPLACEMENTS
        ):
            return d108_target25.validate_d108_target25_prediction_manifest(
                prediction_manifest_path=prediction_manifest_path,
                expected_prediction_manifest_file_sha256=(
                    expected_prediction_manifest_file_sha256
                ),
            )
    except d108_target25.D108Target25Error as error:
        raise D109Target25Error("D109 Target25 validation failed closed") from error


def build_d109_target25_truth_catalog(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    try:
        with d109_target125._temporary_identity(
            d108_target25, _D108_TARGET25_REPLACEMENTS
        ):
            return d108_target25.build_d108_target25_truth_catalog(
                prediction_manifest_path=prediction_manifest_path,
                expected_prediction_manifest_file_sha256=(
                    expected_prediction_manifest_file_sha256
                ),
                plan_manifest_path=plan_manifest_path,
                expected_plan_file_sha256=expected_plan_file_sha256,
                context_manifest_path=context_manifest_path,
                expected_context_file_sha256=expected_context_file_sha256,
                output_path=output_path,
            )
    except d108_target25.D108Target25Error as error:
        raise D109Target25Error("D109 Target25 truth build failed closed") from error


def score_d109_target25(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        with d109_target125._temporary_identity(
            d108_target25, _D108_TARGET25_REPLACEMENTS
        ):
            return d108_target25.score_d108_target25(
                prediction_manifest_path=prediction_manifest_path,
                expected_prediction_manifest_file_sha256=(
                    expected_prediction_manifest_file_sha256
                ),
                truth_catalog_path=truth_catalog_path,
                expected_truth_catalog_file_sha256=(
                    expected_truth_catalog_file_sha256
                ),
                output_dir=output_dir,
            )
    except d108_target25.D108Target25Error as error:
        raise D109Target25Error("D109 Target25 scoring failed closed") from error


__all__ = [
    "ARMS",
    "ARM_PAIR_COUNT",
    "CANDIDATE_ID",
    "D109Target25Error",
    "FROZEN_SEED",
    "OUTER_JOB_COUNT",
    "PHASES",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "PREDICTION_SHARD_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SCENES",
    "SCENE_ROW_COUNT",
    "SCORE_MANIFEST_SCHEMA",
    "SHARD_COUNT",
    "SMOKE_PREDICTIONS_SCHEMA",
    "SMOKE_RECEIPT_SCHEMA",
    "SURFACE_COUNT",
    "TRUTH_CATALOG_SCHEMA",
    "TRUTH_OPEN_EVENT_SCHEMA",
    "TRUTH_SURFACE_COUNT",
    "build_d109_target25_truth_catalog",
    "predict_d109_target25",
    "prepare_d109_target25_run",
    "score_d109_target25",
    "smoke_d109_target25_prepared_state",
    "validate_d109_target25_prediction_manifest",
]
