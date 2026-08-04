"""Thin single-candidate D92-Lite160 Target125 adapter over D108 I/O.

The sealed D92 packages, real feature materializer, eight-shard publication,
truth opening, and metric implementation are reused.  Only the existing
``M_JOINT`` matrix entries are used as transport slots; they carry the D131
D92-Lite system and do not execute the D108 joint mechanism.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import threading
from typing import Any

from . import stage2_d108_target125_runner as d108_runner
from . import stage2_d108_truth_scorer as d108_truth
from . import stage2_d92_lite_target125_core as core


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
SCENE_ARM_METRIC_ROW_COUNT = SCENE_ROW_COUNT
OUTER_ARM_AGGREGATE_ROW_COUNT = OUTER_JOB_COUNT

PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.prediction_manifest.v1"
PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.prediction_artifact.v1"
PREDICTION_SHARD_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.prediction_shard.v1"
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.smoke_receipt.v1"
SMOKE_PREDICTIONS_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.smoke_predictions.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d131.d92_lite160.target125.score_manifest.v1"

StateMaterializer = d108_runner.StateMaterializer
PairBuilder = d108_runner.PairBuilder
QueryScorer = d108_runner.QueryScorer

_LOCK = threading.RLock()
FORMAL_ISOLATION_ENV = "CVS_D131_FORMAL_ISOLATED_PROCESS"
# Candidate adapters may project these labels while reusing this
# single-candidate truth/scoring implementation. Keep the historical D92-Lite
# defaults here so the original adapter remains traceable.
TRUTH_PRIMARY_CANDIDATE_ARM = "D92_LITE160_SYSTEM"
TRUTH_SYSTEM_DIAGNOSTIC_ONLY = True


class D92LiteTarget125Error(ValueError):
    """Raised when the thin D92-Lite Target125 adapter fails closed."""


def _verify_method_lock(
    method_lock_path: str | Path, expected_method_lock_sha256: str
) -> dict[str, Any]:
    source = Path(method_lock_path)
    if (
        source.is_symlink()
        or not source.is_file()
        or type(expected_method_lock_sha256) is not str
        or expected_method_lock_sha256 != core.METHOD_LOCK_SHA256
    ):
        raise D92LiteTarget125Error("D131 method-lock path or expected SHA drift")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != core.METHOD_LOCK_SHA256:
        raise D92LiteTarget125Error("D131 method-lock file SHA drift")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D92LiteTarget125Error("D131 method lock is not UTF-8 JSON") from error
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != core.METHOD_LOCK_SCHEMA
        or document.get("candidate_id") != "D131-D92-LITE160-QTIE/r2"
        or document.get("protocol_schema") != PROTOCOL_SCHEMA
        or document.get("matrix", {}).get("prediction_surface_count") != SURFACE_COUNT
    ):
        raise D92LiteTarget125Error("D131 method-lock identity or matrix drift")
    return dict(document)


def _require_formal_isolation() -> None:
    if os.environ.get(FORMAL_ISOLATION_ENV) != "1" or threading.active_count() != 1:
        raise D92LiteTarget125Error(
            "formal D131 projection requires a dedicated single-thread Python process"
        )


def _filtered_matrix() -> Any:
    matrix = d108_runner.freeze_d108_target125_matrix()
    arm_pairs = tuple(item for item in matrix.arm_pairs if item.arm in ARMS)
    surfaces = tuple(item for item in matrix.surfaces if item.arm in ARMS)
    if (
        len(matrix.outer_rows) != OUTER_JOB_COUNT
        or len(matrix.scene_rows) != SCENE_ROW_COUNT
        or len(arm_pairs) != ARM_PAIR_COUNT
        or len(surfaces) != SURFACE_COUNT
    ):
        raise D92LiteTarget125Error("single-arm matrix projection coverage drift")
    return SimpleNamespace(
        outer_rows=matrix.outer_rows,
        scene_rows=matrix.scene_rows,
        arm_pairs=arm_pairs,
        surfaces=surfaces,
        matrix_receipt_sha256=matrix.matrix_receipt_sha256,
    )


def _audit_single_arm_surface_coverage(values: Any) -> None:
    iterable = values.keys() if isinstance(values, Mapping) else values
    actual = tuple(str(item) for item in iterable)
    expected = tuple(item.surface_id for item in _filtered_matrix().surfaces)
    if len(actual) != SURFACE_COUNT or len(set(actual)) != SURFACE_COUNT or set(actual) != set(expected):
        raise D92LiteTarget125Error("single-arm surface coverage drift")


@contextmanager
def _prediction_projection():
    _require_formal_isolation()
    filtered = _filtered_matrix()
    replacements = {
        "CANDIDATE_ID": CANDIDATE_ID,
        "ARMS": ARMS,
        "ARM_PAIR_COUNT": ARM_PAIR_COUNT,
        "SURFACE_COUNT": SURFACE_COUNT,
        "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
        "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
        "PREDICTION_SHARD_SCHEMA": PREDICTION_SHARD_SCHEMA,
        "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
        "SMOKE_PREDICTIONS_SCHEMA": SMOKE_PREDICTIONS_SCHEMA,
        "freeze_d108_target125_matrix": lambda: filtered,
        "audit_surface_coverage": _audit_single_arm_surface_coverage,
    }
    with _LOCK:
        original = {name: getattr(d108_runner, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(d108_runner, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(d108_runner, name, value)


@contextmanager
def _truth_projection():
    _require_formal_isolation()
    original_summary = d108_truth._score_summary

    def single_summary(prediction: Mapping[str, Any], *, truth_catalog_size: int):
        coverage, resources, verdict = original_summary(
            prediction, truth_catalog_size=truth_catalog_size
        )
        coverage = dict(coverage)
        coverage["four_arm_causal_coverage_verified"] = False
        coverage["single_candidate_before_after_coverage_verified"] = True
        verdict = dict(verdict)
        verdict.update(
            {
                "primary_candidate_arm": TRUTH_PRIMARY_CANDIDATE_ARM,
                "transport_arm": core.TRANSPORT_ARM,
                "transport_arm_is_D108_joint_mechanism": False,
                "causal_arms": [],
                "causal_table_preserved": False,
                "system_diagnostic_only": TRUTH_SYSTEM_DIAGNOSTIC_ONLY,
            }
        )
        return coverage, resources, verdict

    replacements = {
        "CANDIDATE_ID": CANDIDATE_ID,
        "ARMS": ARMS,
        "ARM_PAIR_COUNT": ARM_PAIR_COUNT,
        "SURFACE_COUNT": SURFACE_COUNT,
        "SCENE_ARM_METRIC_ROW_COUNT": SCENE_ARM_METRIC_ROW_COUNT,
        "OUTER_ARM_AGGREGATE_ROW_COUNT": OUTER_ARM_AGGREGATE_ROW_COUNT,
        "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
        "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
        "TRUTH_CATALOG_SCHEMA": TRUTH_CATALOG_SCHEMA,
        "TRUTH_OPEN_EVENT_SCHEMA": TRUTH_OPEN_EVENT_SCHEMA,
        "SCORE_MANIFEST_SCHEMA": SCORE_MANIFEST_SCHEMA,
        "_PREDICTION_MANIFEST_FIELDS": frozenset(d108_truth._PREDICTION_MANIFEST_FIELDS)
        | {"shard_count", "shard_receipts"},
        "_score_summary": single_summary,
    }
    with _LOCK:
        original = {name: getattr(d108_truth, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(d108_truth, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(d108_truth, name, value)


def _prepared_inputs(
    *, plan_manifest_path: Path, expected_plan_file_sha256: str,
    context_manifest_path: Path, expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return d108_runner._prepared_inputs(
            plan_manifest_path=plan_manifest_path,
            expected_plan_file_sha256=expected_plan_file_sha256,
            context_manifest_path=context_manifest_path,
            expected_context_file_sha256=expected_context_file_sha256,
        )
    except d108_runner.D108Target125RunnerError as error:
        raise D92LiteTarget125Error("D108 prepared-input reuse failed closed") from error


def prepare_d92_lite_target125_run(
    *, method_lock_path: str | Path, expected_method_lock_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    return d108_runner.prepare_d108_target125_run(**kwargs)


def smoke_d92_lite_target125_prepared_state(
    *, plan_manifest_path: Path, expected_plan_file_sha256: str,
    context_manifest_path: Path, expected_context_file_sha256: str,
    output_dir: Path, row_index: int = 0, scene_index: int = 0,
    device: str = "cpu", feature_batch_size: int = 64,
    method_lock_path: str | Path, expected_method_lock_sha256: str,
    state_materializer: StateMaterializer | None = None,
    pair_builder: PairBuilder | None = None, query_scorer: QueryScorer | None = None,
    d92_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    _require_formal_isolation()
    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    if row_index not in range(OUTER_JOB_COUNT) or scene_index not in range(len(SCENES)):
        raise D92LiteTarget125Error("smoke row/scene index drift")
    materializer = state_materializer or d108_runner._D108RealStateMaterializer(
        plan=plan, device=device, support_batch_size=feature_batch_size
    )
    builder = core.build_d92_lite_pair if pair_builder is None else pair_builder
    scorer = core.score if query_scorer is None else query_scorer
    pair_device, resolved_fit = d108_runner._pair_runtime_bindings(
        materializer, device=device, d92_fit=d92_fit
    )
    row = context["rows"][row_index]
    scene = SCENES[scene_index]
    destination = d108_runner._output_dir_new(output_dir, "D92-Lite smoke")
    before, after = d108_runner._materialize_pair(materializer, row, scene)
    pair = d108_runner._build_pair(
        before, after, row=row, scene=scene, plan=plan, pair_builder=builder,
        device=pair_device, d92_fit=resolved_fit,
    )
    surfaces = []
    for phase, state in (("before", before), ("after", after)):
        labels = d108_runner._predict_labels(
            pair, state, arm=core.TRANSPORT_ARM, phase=phase, query_scorer=scorer
        )
        surfaces.append(
            {
                "arm": core.TRANSPORT_ARM,
                "phase": phase,
                "registered_classes": list(state.registered_classes),
                "ordered_query_physical_ids": list(state.query_physical_ids),
                "predicted_labels": list(labels),
            }
        )
    predictions = {
        "schema": SMOKE_PREDICTIONS_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "outer_id": row["outer_id"], "scene": scene, "surfaces": surfaces,
        "access_ledger": dict(d108_runner.ACCESS_LEDGER), "truth_open": False,
        "immutable": True,
    }
    predictions["smoke_predictions_receipt_sha256"] = d108_runner.canonical_sha256(predictions)
    predictions_path = destination / "smoke_predictions.json"
    predictions_file_sha = d108_runner._write_json_new(predictions_path, predictions)
    receipt = {
        "schema": SMOKE_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "status": "D92_LITE160_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS",
        "outer_id": row["outer_id"], "scene": scene,
        "transport_arm": core.TRANSPORT_ARM,
        "transport_arm_is_D108_joint_mechanism": False,
        "method_lock_sha256": core.METHOD_LOCK_SHA256,
        "phases": list(PHASES), "support_batch_size": feature_batch_size,
        "query_batch_size": 1, "query_truth_access": False,
        "query_fit_access": False, "query_update_access": False,
        "query_selection_access": False,
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
    }


def predict_d92_lite_target125(
    *, plan_manifest_path: Path | None = None,
    expected_plan_file_sha256: str | None = None,
    context_manifest_path: Path | None = None,
    expected_context_file_sha256: str | None = None,
    output_dir: Path, device: str = "cpu", feature_batch_size: int = 64,
    method_lock_path: str | Path, expected_method_lock_sha256: str,
    shard_index: int | None = None,
    shard_manifest_paths: Sequence[Path] | None = None,
    state_materializer: StateMaterializer | None = None,
    pair_builder: PairBuilder | None = None, query_scorer: QueryScorer | None = None,
    d92_fit: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
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
                raise D92LiteTarget125Error(
                    "merge accepts only shard manifests and output_dir"
                )
            with _prediction_projection():
                return d108_runner._merge_shards(
                    shard_manifest_paths=shard_manifest_paths, output_dir=output_dir
                )
        if any(value is None for value in (
            plan_manifest_path, expected_plan_file_sha256,
            context_manifest_path, expected_context_file_sha256, shard_index,
        )):
            raise D92LiteTarget125Error("shard prediction requires prepared inputs")
        plan, context = _prepared_inputs(
            plan_manifest_path=Path(plan_manifest_path),
            expected_plan_file_sha256=str(expected_plan_file_sha256),
            context_manifest_path=Path(context_manifest_path),
            expected_context_file_sha256=str(expected_context_file_sha256),
        )
        with _prediction_projection():
            return d108_runner._predict_shard(
                plan=plan, context=context, output_dir=output_dir,
                shard_index=int(shard_index), device=device,
                feature_batch_size=feature_batch_size,
                state_materializer=state_materializer,
                pair_builder=core.build_d92_lite_pair if pair_builder is None else pair_builder,
                query_scorer=core.score if query_scorer is None else query_scorer,
                d92_fit=d92_fit,
            )
    except d108_runner.D108Target125RunnerError as error:
        raise D92LiteTarget125Error("D92-Lite prediction failed closed") from error


def validate_d92_lite_target125_prediction_manifest(
    *, prediction_manifest_path: Path,
    expected_prediction_manifest_file_sha256: str | None = None,
    method_lock_path: str | Path, expected_method_lock_sha256: str,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    with _prediction_projection():
        return d108_runner.validate_d108_target125_prediction_manifest(
            prediction_manifest_path=prediction_manifest_path,
            expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
        )


def build_d92_lite_target125_truth_catalog(
    *, method_lock_path: str | Path, expected_method_lock_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    validate_d92_lite_target125_prediction_manifest(
        prediction_manifest_path=Path(kwargs["prediction_manifest_path"]),
        expected_prediction_manifest_file_sha256=kwargs[
            "expected_prediction_manifest_file_sha256"
        ],
        method_lock_path=method_lock_path,
        expected_method_lock_sha256=expected_method_lock_sha256,
    )
    with _truth_projection():
        return d108_truth.build_d108_target125_truth_catalog(**kwargs)


def score_d92_lite_target125(
    *, method_lock_path: str | Path, expected_method_lock_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    _verify_method_lock(method_lock_path, expected_method_lock_sha256)
    validate_d92_lite_target125_prediction_manifest(
        prediction_manifest_path=Path(kwargs["prediction_manifest_path"]),
        expected_prediction_manifest_file_sha256=kwargs[
            "expected_prediction_manifest_file_sha256"
        ],
        method_lock_path=method_lock_path,
        expected_method_lock_sha256=expected_method_lock_sha256,
    )
    with _truth_projection():
        return d108_truth.score_d108_target125(**kwargs)


__all__ = [
    "ARM_PAIR_COUNT", "ARMS", "CANDIDATE_ID", "D92LiteTarget125Error",
    "OUTER_JOB_COUNT", "PHASES", "SCENES", "SHARD_COUNT", "SURFACE_COUNT",
    "build_d92_lite_target125_truth_catalog", "predict_d92_lite_target125",
    "prepare_d92_lite_target125_run", "score_d92_lite_target125",
    "smoke_d92_lite_target125_prepared_state",
    "validate_d92_lite_target125_prediction_manifest",
]
