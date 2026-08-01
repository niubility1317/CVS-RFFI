"""Thin D109 Target125 adapter over the verified D108 execution plane.

The D108 prepared plan/context, sealed D92 materializer, frozen 125 matrix,
immutable shard publication, and independent truth scorer remain unchanged.
This module injects only the frozen D109 D92/SCRC pair and gives prediction,
smoke, truth, and score artifacts a distinct D109 identity.  The temporary
identity switch is process-local, lock-guarded, and exception-safe.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Any

import numpy as np

from . import stage2_d108_target125_runner as d108_runner
from . import stage2_d108_truth_scorer as d108_truth
from . import stage2_d109_d92_core as d109_core


CANDIDATE_ID = "D109-SCRC/r1"
PROTOCOL_SCHEMA = "p2_min_v1"
PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d109.scrc.target125.prediction_manifest.v1"
PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.d109.scrc.target125.prediction_artifact.v1"
PREDICTION_SHARD_SCHEMA = "cvs.phase2.d109.scrc.target125.prediction_shard.v1"
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d109.scrc.target125.smoke_receipt.v1"
SMOKE_PREDICTIONS_SCHEMA = "cvs.phase2.d109.scrc.target125.smoke_predictions.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d109.scrc.target125.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d109.scrc.target125.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d109.scrc.target125.score_manifest.v1"

ARMS = d108_runner.ARMS
PHASES = d108_runner.PHASES
SCENES = d108_runner.SCENES
SHARD_COUNT = d108_runner.SHARD_COUNT
OUTER_JOB_COUNT = d108_runner.OUTER_JOB_COUNT
SURFACE_COUNT = d108_runner.SURFACE_COUNT

StateMaterializer = d108_runner.StateMaterializer
PairBuilder = d108_runner.PairBuilder
QueryScorer = d108_runner.QueryScorer

_IDENTITY_LOCK = threading.RLock()
_PREDICTION_REPLACEMENTS = {
    "CANDIDATE_ID": CANDIDATE_ID,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "PREDICTION_SHARD_SCHEMA": PREDICTION_SHARD_SCHEMA,
    "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
    "SMOKE_PREDICTIONS_SCHEMA": SMOKE_PREDICTIONS_SCHEMA,
}
_TRUTH_REPLACEMENTS: dict[str, Any] = {
    "CANDIDATE_ID": CANDIDATE_ID,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "TRUTH_CATALOG_SCHEMA": TRUTH_CATALOG_SCHEMA,
    "TRUTH_OPEN_EVENT_SCHEMA": TRUTH_OPEN_EVENT_SCHEMA,
    "SCORE_MANIFEST_SCHEMA": SCORE_MANIFEST_SCHEMA,
    "_PREDICTION_MANIFEST_FIELDS": frozenset(
        d108_truth._PREDICTION_MANIFEST_FIELDS
    )
    | {"shard_count", "shard_receipts"},
}


class D109Target125Error(ValueError):
    """Raised when the D109 adapter cannot preserve the D108 sealed plane."""


@contextmanager
def _temporary_identity(module: Any, replacements: Mapping[str, Any]):
    """Apply a bounded identity projection and restore every global exactly."""

    with _IDENTITY_LOCK:
        original = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(module, name, value)


def _prepared_inputs(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return d108_runner._prepared_inputs(
            plan_manifest_path=plan_manifest_path,
            expected_plan_file_sha256=expected_plan_file_sha256,
            context_manifest_path=context_manifest_path,
            expected_context_file_sha256=expected_context_file_sha256,
        )
    except d108_runner.D108Target125RunnerError as error:
        raise D109Target125Error("D108 prepared input reuse failed closed") from error


def _resolve_d109_core(
    pair_builder: PairBuilder | None, query_scorer: QueryScorer | None
) -> tuple[PairBuilder, QueryScorer]:
    builder = d109_core.build_d109_d92_pair if pair_builder is None else pair_builder
    scorer = d109_core.score if query_scorer is None else query_scorer
    if not callable(builder) or not callable(scorer):
        raise D109Target125Error("D109 pair/score API must be callable")
    return builder, scorer


def prepare_d109_target125_run(**kwargs: Any) -> dict[str, Any]:
    """Reuse the verified D108 prepared D92/materialization input plane."""

    try:
        return d108_runner.prepare_d108_target125_run(**kwargs)
    except d108_runner.D108Target125RunnerError as error:
        raise D109Target125Error("D109 prepare reuse failed closed") from error


def smoke_d109_target125_prepared_state(
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
    """Exercise one D108-materialized row with the four frozen D109 arms."""

    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    if (
        type(row_index) is not int
        or row_index not in range(OUTER_JOB_COUNT)
        or type(scene_index) is not int
        or scene_index not in range(len(SCENES))
    ):
        raise D109Target125Error("smoke row/scene index drift")
    try:
        destination = d108_runner._output_dir_new(output_dir, "D109 smoke")
        materializer = state_materializer or d108_runner._D108RealStateMaterializer(
            plan=plan, device=device, support_batch_size=feature_batch_size
        )
        builder, scorer = _resolve_d109_core(pair_builder, query_scorer)
        pair_device, resolved_d92_fit = d108_runner._pair_runtime_bindings(
            materializer, device=device, d92_fit=d92_fit
        )
        row = context["rows"][row_index]
        scene = SCENES[scene_index]
        before, after = d108_runner._materialize_pair(materializer, row, scene)
        pair = d108_runner._build_pair(
            before,
            after,
            row=row,
            scene=scene,
            plan=plan,
            pair_builder=builder,
            device=pair_device,
            d92_fit=resolved_d92_fit,
        )
        surfaces: list[dict[str, Any]] = []
        for arm in ARMS:
            for phase, materialized in (("before", before), ("after", after)):
                labels = d108_runner._predict_labels(
                    pair,
                    materialized,
                    arm=arm,
                    phase=phase,
                    query_scorer=scorer,
                )
                query_ids = list(materialized.query_physical_ids)
                predicted = list(labels)
                surfaces.append(
                    {
                        "arm": arm,
                        "phase": phase,
                        "registered_classes": list(materialized.registered_classes),
                        "ordered_query_physical_ids": query_ids,
                        "ordered_query_physical_ids_sha256": (
                            d108_runner.canonical_sha256(query_ids)
                        ),
                        "predicted_labels": predicted,
                        "predicted_labels_sha256": (
                            d108_runner.canonical_sha256(predicted)
                        ),
                    }
                )
        predictions: dict[str, Any] = {
            "schema": SMOKE_PREDICTIONS_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "protocol_schema": PROTOCOL_SCHEMA,
            "outer_id": row["outer_id"],
            "receiver": row["receiver"],
            "seed": row["seed"],
            "k_shot": row["k_shot"],
            "new_count": row["new_count"],
            "scene": scene,
            "surfaces": surfaces,
            "access_ledger": dict(d108_runner.ACCESS_LEDGER),
            "truth_open": False,
            "immutable": True,
        }
        predictions["smoke_predictions_receipt_sha256"] = (
            d108_runner.canonical_sha256(predictions)
        )
        predictions_path = destination / "smoke_predictions.json"
        predictions_file_sha = d108_runner._write_json_new(
            predictions_path, predictions
        )
        receipt: dict[str, Any] = {
            "schema": SMOKE_RECEIPT_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "protocol_schema": PROTOCOL_SCHEMA,
            "status": "D109_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS",
            "outer_id": row["outer_id"],
            "scene": scene,
            "arms": list(ARMS),
            "phases": list(PHASES),
            "support_batch_size": feature_batch_size,
            "query_batch_size": 1,
            "query_truth_access": False,
            "query_fit_access": False,
            "query_update_access": False,
        }
        receipt["smoke_receipt_sha256"] = d108_runner.canonical_sha256(receipt)
        receipt_path = destination / "smoke_receipt.json"
        receipt_file_sha = d108_runner._write_json_new(receipt_path, receipt)
        return {
            **receipt,
            "smoke_receipt": str(receipt_path),
            "smoke_receipt_file_sha256": receipt_file_sha,
            "smoke_predictions": str(predictions_path),
            "smoke_predictions_file_sha256": predictions_file_sha,
            "smoke_predictions_receipt_sha256": predictions[
                "smoke_predictions_receipt_sha256"
            ],
        }
    except d108_runner.D108Target125RunnerError as error:
        raise D109Target125Error("D109 smoke failed closed") from error


def predict_d109_target125(
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
    """Publish one immutable D109 shard or merge exactly eight shards."""

    try:
        if shard_manifest_paths is not None:
            if any(
                value is not None
                for value in (
                    plan_manifest_path,
                    expected_plan_file_sha256,
                    context_manifest_path,
                    expected_context_file_sha256,
                    shard_index,
                    state_materializer,
                    pair_builder,
                    query_scorer,
                    d92_fit,
                )
            ):
                raise D109Target125Error(
                    "merge accepts only shard manifests and output_dir"
                )
            with _temporary_identity(d108_runner, _PREDICTION_REPLACEMENTS):
                return d108_runner._merge_shards(
                    shard_manifest_paths=shard_manifest_paths,
                    output_dir=output_dir,
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
            raise D109Target125Error(
                "shard predict requires prepared inputs and shard_index"
            )
        plan, context = _prepared_inputs(
            plan_manifest_path=Path(plan_manifest_path),
            expected_plan_file_sha256=str(expected_plan_file_sha256),
            context_manifest_path=Path(context_manifest_path),
            expected_context_file_sha256=str(expected_context_file_sha256),
        )
        builder, scorer = _resolve_d109_core(pair_builder, query_scorer)
        with _temporary_identity(d108_runner, _PREDICTION_REPLACEMENTS):
            return d108_runner._predict_shard(
                plan=plan,
                context=context,
                output_dir=output_dir,
                shard_index=int(shard_index),
                device=device,
                feature_batch_size=feature_batch_size,
                state_materializer=state_materializer,
                pair_builder=builder,
                query_scorer=scorer,
                d92_fit=d92_fit,
            )
    except d108_runner.D108Target125RunnerError as error:
        raise D109Target125Error(
            f"D109 prediction publication failed closed: {error}"
        ) from error


def validate_d109_target125_prediction_manifest(
    *,
    prediction_manifest_path: Path,
    expected_prediction_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        with _temporary_identity(d108_runner, _PREDICTION_REPLACEMENTS):
            return d108_runner.validate_d108_target125_prediction_manifest(
                prediction_manifest_path=prediction_manifest_path,
                expected_prediction_manifest_file_sha256=(
                    expected_prediction_manifest_file_sha256
                ),
            )
    except d108_runner.D108Target125RunnerError as error:
        raise D109Target125Error("D109 prediction validation failed closed") from error


def build_d109_target125_truth_catalog(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Reuse the D108 truth join with D109 prediction/truth identities."""

    validate_d109_target125_prediction_manifest(
        prediction_manifest_path=Path(prediction_manifest_path),
        expected_prediction_manifest_file_sha256=(
            expected_prediction_manifest_file_sha256
        ),
    )
    try:
        with _temporary_identity(d108_truth, _TRUTH_REPLACEMENTS):
            return d108_truth.build_d108_target125_truth_catalog(
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
    except d108_truth.D108TruthScorerError as error:
        raise D109Target125Error("D109 truth-catalog build failed closed") from error


def score_d109_target125(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reuse the complete D108 truth-side metric implementation for D109."""

    validate_d109_target125_prediction_manifest(
        prediction_manifest_path=Path(prediction_manifest_path),
        expected_prediction_manifest_file_sha256=(
            expected_prediction_manifest_file_sha256
        ),
    )
    try:
        with _temporary_identity(d108_truth, _TRUTH_REPLACEMENTS):
            return d108_truth.score_d108_target125(
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
    except d108_truth.D108TruthScorerError as error:
        raise D109Target125Error("D109 truth-side scoring failed closed") from error


__all__ = [
    "ARMS",
    "CANDIDATE_ID",
    "D109Target125Error",
    "OUTER_JOB_COUNT",
    "PHASES",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "PREDICTION_SHARD_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SCENES",
    "SCORE_MANIFEST_SCHEMA",
    "SHARD_COUNT",
    "SMOKE_PREDICTIONS_SCHEMA",
    "SMOKE_RECEIPT_SCHEMA",
    "SURFACE_COUNT",
    "TRUTH_CATALOG_SCHEMA",
    "TRUTH_OPEN_EVENT_SCHEMA",
    "build_d109_target125_truth_catalog",
    "predict_d109_target125",
    "prepare_d109_target125_run",
    "score_d109_target125",
    "smoke_d109_target125_prepared_state",
    "validate_d109_target125_prediction_manifest",
]
