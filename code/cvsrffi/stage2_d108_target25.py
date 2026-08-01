"""Frozen-seed D108 Target25 adapter over the verified Target125 plane.

The adapter validates the complete prepared D108 inputs, selects only seed
713102 across the five receivers and five frozen slices, and reuses the real
D92 materializer, D108 four-arm core, immutable publication, and truth scorer.
No method constant, fit, data permission, or query boundary is changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from . import stage2_d108_matrix_protocol as matrix125
from . import stage2_d108_target125_runner as runner125
from . import stage2_d108_truth_scorer as truth125


FROZEN_SEED = 713102
OUTER_JOB_COUNT = 25
SCENE_ROW_COUNT = 75
ARM_PAIR_COUNT = 300
SURFACE_COUNT = 600
TRUTH_SURFACE_COUNT = 150
SCENE_ARM_METRIC_ROW_COUNT = 300
OUTER_ARM_AGGREGATE_ROW_COUNT = 100
SHARD_COUNT = runner125.SHARD_COUNT

CANDIDATE_ID = matrix125.CANDIDATE_ID
PROTOCOL_SCHEMA = matrix125.PROTOCOL_SCHEMA
MATRIX_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.matrix.v1"
PREDICTION_MANIFEST_SCHEMA = (
    "cvs.phase2.d108.cbrrc_smme.target25.prediction_manifest.v1"
)
PREDICTION_ARTIFACT_SCHEMA = (
    "cvs.phase2.d108.cbrrc_smme.target25.prediction_artifact.v1"
)
PREDICTION_SHARD_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.prediction_shard.v1"
SMOKE_RECEIPT_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.smoke_receipt.v1"
SMOKE_PREDICTIONS_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.smoke_predictions.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target25.score_manifest.v1"

ARMS = matrix125.ARMS
PHASES = matrix125.PHASES
SCENES = matrix125.SCENES
RECEIVERS = matrix125.RECEIVERS
SLICES = matrix125.TARGET125_SLICES
ACCESS_LEDGER = matrix125.ACCESS_LEDGER

StateMaterializer = runner125.StateMaterializer
PairBuilder = runner125.PairBuilder
QueryScorer = runner125.QueryScorer

_IDENTITY_LOCK = threading.RLock()
_ORIGINAL_SCORE_SUMMARY = truth125._score_summary


class D108Target25Error(ValueError):
    """Raised when the frozen-seed projection cannot fail closed."""


@dataclass(frozen=True, slots=True)
class D108Target25Plan:
    outer_rows: tuple[matrix125.D108OuterRow, ...]
    scene_rows: tuple[matrix125.D108SceneRow, ...]
    arm_pairs: tuple[matrix125.D108ArmPair, ...]
    surfaces: tuple[matrix125.D108Surface, ...]
    matrix_receipt_sha256: str

    def receipt_payload(self) -> dict[str, Any]:
        payload = _matrix_payload(self)
        return {**payload, "matrix_receipt_sha256": self.matrix_receipt_sha256}


def _matrix_payload(plan: D108Target25Plan) -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT,
        "receivers": list(RECEIVERS),
        "seeds": [FROZEN_SEED],
        "slices": [
            {"k_shot": k_shot, "new_count": new_count}
            for k_shot, new_count in SLICES
        ],
        "scenes": list(SCENES),
        "arms": list(ARMS),
        "phases": list(PHASES),
        "access_ledger": dict(ACCESS_LEDGER),
        "outer_rows": [row.as_dict() for row in plan.outer_rows],
        "scene_rows": [row.as_dict() for row in plan.scene_rows],
        "arm_pairs": [pair.as_dict() for pair in plan.arm_pairs],
        "surfaces": [surface.as_dict() for surface in plan.surfaces],
    }


def freeze_d108_target25_matrix() -> D108Target25Plan:
    full = matrix125.freeze_d108_target125_matrix()
    outer = tuple(row for row in full.outer_rows if row.seed == FROZEN_SEED)
    outer_ids = {row.outer_id for row in outer}
    scene = tuple(row for row in full.scene_rows if row.outer_id in outer_ids)
    pairs = tuple(row for row in full.arm_pairs if row.outer_id in outer_ids)
    surfaces = tuple(row for row in full.surfaces if row.outer_id in outer_ids)
    provisional = D108Target25Plan(outer, scene, pairs, surfaces, "0" * 64)
    receipt = matrix125.canonical_sha256(_matrix_payload(provisional))
    plan = D108Target25Plan(outer, scene, pairs, surfaces, receipt)
    if (
        len(plan.outer_rows) != OUTER_JOB_COUNT
        or len(plan.scene_rows) != SCENE_ROW_COUNT
        or len(plan.arm_pairs) != ARM_PAIR_COUNT
        or len(plan.surfaces) != SURFACE_COUNT
        or {row.seed for row in plan.outer_rows} != {FROZEN_SEED}
        or plan.matrix_receipt_sha256
        != matrix125.canonical_sha256(_matrix_payload(plan))
    ):
        raise D108Target25Error("Target25 matrix closure drift")
    return plan


def audit_d108_target25_surface_coverage(surface_ids: Sequence[str]) -> None:
    values = list(surface_ids)
    expected = {
        surface.surface_id for surface in freeze_d108_target25_matrix().surfaces
    }
    if (
        len(values) != SURFACE_COUNT
        or len(set(values)) != SURFACE_COUNT
        or set(values) != expected
    ):
        raise D108Target25Error("Target25 surface coverage drift")


def _score_summary_target25(
    prediction: Mapping[str, Any], *, truth_catalog_size: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coverage, resources, verdict = _ORIGINAL_SCORE_SUMMARY(
        prediction, truth_catalog_size=truth_catalog_size
    )
    verdict = dict(verdict)
    verdict["coverage_verdict"] = "COMPLETE_25_TRUTH_OPEN_AND_SCORED"
    return dict(coverage), dict(resources), verdict


_RUNNER_REPLACEMENTS: dict[str, Any] = {
    "OUTER_JOB_COUNT": OUTER_JOB_COUNT,
    "SCENE_ROW_COUNT": SCENE_ROW_COUNT,
    "ARM_PAIR_COUNT": ARM_PAIR_COUNT,
    "SURFACE_COUNT": SURFACE_COUNT,
    "freeze_d108_target125_matrix": freeze_d108_target25_matrix,
    "audit_surface_coverage": audit_d108_target25_surface_coverage,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "PREDICTION_SHARD_SCHEMA": PREDICTION_SHARD_SCHEMA,
    "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
    "SMOKE_PREDICTIONS_SCHEMA": SMOKE_PREDICTIONS_SCHEMA,
}


def _load_target25_truth_inputs(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return _prepared_target25_inputs(
        plan_manifest_path=Path(kwargs["plan_manifest_path"]),
        expected_plan_file_sha256=kwargs["expected_plan_file_sha256"],
        context_manifest_path=Path(kwargs["context_manifest_path"]),
        expected_context_file_sha256=kwargs["expected_context_file_sha256"],
    )


_TRUTH_REPLACEMENTS: dict[str, Any] = {
    "OUTER_JOB_COUNT": OUTER_JOB_COUNT,
    "SCENE_ROW_COUNT": SCENE_ROW_COUNT,
    "ARM_PAIR_COUNT": ARM_PAIR_COUNT,
    "SURFACE_COUNT": SURFACE_COUNT,
    "TRUTH_SURFACE_COUNT": TRUTH_SURFACE_COUNT,
    "SCENE_ARM_METRIC_ROW_COUNT": SCENE_ARM_METRIC_ROW_COUNT,
    "OUTER_ARM_AGGREGATE_ROW_COUNT": OUTER_ARM_AGGREGATE_ROW_COUNT,
    "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
    "PREDICTION_ARTIFACT_SCHEMA": PREDICTION_ARTIFACT_SCHEMA,
    "TRUTH_CATALOG_SCHEMA": TRUTH_CATALOG_SCHEMA,
    "TRUTH_OPEN_EVENT_SCHEMA": TRUTH_OPEN_EVENT_SCHEMA,
    "SCORE_MANIFEST_SCHEMA": SCORE_MANIFEST_SCHEMA,
    "_PREDICTION_MANIFEST_FIELDS": frozenset(
        truth125._PREDICTION_MANIFEST_FIELDS
    )
    | {"shard_count", "shard_receipts"},
    "_load_prepared_d108_truth_inputs": _load_target25_truth_inputs,
    "_score_summary": _score_summary_target25,
}


@contextmanager
def _temporary_projection(module: Any, replacements: Mapping[str, Any]):
    with _IDENTITY_LOCK:
        original = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(module, name, value)


def _prepared_target25_inputs(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        plan, context = runner125._prepared_inputs(
            plan_manifest_path=plan_manifest_path,
            expected_plan_file_sha256=expected_plan_file_sha256,
            context_manifest_path=context_manifest_path,
            expected_context_file_sha256=expected_context_file_sha256,
        )
    except runner125.D108Target125RunnerError as error:
        raise D108Target25Error("full D108 prepared inputs failed closed") from error
    matrix = freeze_d108_target25_matrix()
    rows = [row for row in plan["rows"] if row.get("seed") == FROZEN_SEED]
    if len(rows) != OUTER_JOB_COUNT or rows != [
        row for row in context["rows"] if row.get("seed") == FROZEN_SEED
    ]:
        raise D108Target25Error("prepared seed-713102 row projection drift")
    identity = dict(plan["identity"])
    identity["matrix_receipt_sha256"] = matrix.matrix_receipt_sha256
    projected_plan = {
        **plan,
        "matrix_protocol": matrix.receipt_payload(),
        "identity": identity,
        "rows": rows,
    }
    projected_plan.pop("plan_receipt_sha256", None)
    projected_plan["plan_receipt_sha256"] = matrix125.canonical_sha256(projected_plan)
    projected_context = {
        **context,
        "plan_receipt_sha256": projected_plan["plan_receipt_sha256"],
        "identity": identity,
        "rows": rows,
    }
    projected_context.pop("context_receipt_sha256", None)
    projected_context["context_receipt_sha256"] = matrix125.canonical_sha256(
        projected_context
    )
    return projected_plan, projected_context


def prepare_d108_target25_run(**kwargs: Any) -> dict[str, Any]:
    """Prepare the existing D108 inputs and return the frozen Target25 receipt."""

    try:
        result = runner125.prepare_d108_target125_run(**kwargs)
    except runner125.D108Target125RunnerError as error:
        raise D108Target25Error("D108 prepare reuse failed closed") from error
    matrix = freeze_d108_target25_matrix()
    return {
        **result,
        "target25_matrix_receipt_sha256": matrix.matrix_receipt_sha256,
        "target25_outer_job_count": OUTER_JOB_COUNT,
        "target25_surface_count": SURFACE_COUNT,
        "target25_seed": FROZEN_SEED,
    }


def smoke_d108_target25_prepared_state(
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
    plan, context = _prepared_target25_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    if row_index not in range(OUTER_JOB_COUNT):
        raise D108Target25Error("smoke row index must be in 0..24")
    if scene_index not in range(len(SCENES)):
        raise D108Target25Error("smoke scene index must be in 0..2")
    try:
        with _temporary_projection(runner125, _RUNNER_REPLACEMENTS):
            destination = runner125._output_dir_new(output_dir, "Target25 smoke")
            materializer = state_materializer or runner125._D108RealStateMaterializer(
                plan=plan, device=device, support_batch_size=feature_batch_size
            )
            builder, scorer = runner125._resolve_core(pair_builder, query_scorer)
            pair_device, fit = runner125._pair_runtime_bindings(
                materializer, device=device, d92_fit=d92_fit
            )
            row = context["rows"][row_index]
            scene = SCENES[scene_index]
            before, after = runner125._materialize_pair(materializer, row, scene)
            pair = runner125._build_pair(
                before,
                after,
                row=row,
                scene=scene,
                plan=plan,
                pair_builder=builder,
                device=pair_device,
                d92_fit=fit,
            )
            surfaces: list[dict[str, Any]] = []
            for arm in ARMS:
                for phase, state in (("before", before), ("after", after)):
                    labels = runner125._predict_labels(
                        pair, state, arm=arm, phase=phase, query_scorer=scorer
                    )
                    query_ids = list(state.query_physical_ids)
                    predicted = list(labels)
                    surfaces.append(
                        {
                            "arm": arm,
                            "phase": phase,
                            "registered_classes": list(state.registered_classes),
                            "ordered_query_physical_ids": query_ids,
                            "ordered_query_physical_ids_sha256": (
                                matrix125.canonical_sha256(query_ids)
                            ),
                            "predicted_labels": predicted,
                            "predicted_labels_sha256": (
                                matrix125.canonical_sha256(predicted)
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
                "access_ledger": dict(ACCESS_LEDGER),
                "truth_open": False,
                "immutable": True,
            }
            predictions["smoke_predictions_receipt_sha256"] = (
                matrix125.canonical_sha256(predictions)
            )
            predictions_path = destination / "smoke_predictions.json"
            predictions_sha = runner125._write_json_new(
                predictions_path, predictions
            )
            receipt: dict[str, Any] = {
                "schema": SMOKE_RECEIPT_SCHEMA,
                "candidate_id": CANDIDATE_ID,
                "protocol_schema": PROTOCOL_SCHEMA,
                "status": "D108_TARGET25_REAL_CHECKPOINT_NO_QUERY_FIT_SMOKE_PASS",
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
            receipt["smoke_receipt_sha256"] = matrix125.canonical_sha256(receipt)
            receipt_path = destination / "smoke_receipt.json"
            receipt_sha = runner125._write_json_new(receipt_path, receipt)
            return {
                **receipt,
                "smoke_receipt": str(receipt_path),
                "smoke_receipt_file_sha256": receipt_sha,
                "smoke_predictions": str(predictions_path),
                "smoke_predictions_file_sha256": predictions_sha,
            }
    except runner125.D108Target125RunnerError as error:
        raise D108Target25Error(f"Target25 smoke failed closed: {error}") from error


def predict_d108_target25(
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
                raise D108Target25Error("merge accepts only shard manifests and output_dir")
            with _temporary_projection(runner125, _RUNNER_REPLACEMENTS):
                return runner125._merge_shards(
                    shard_manifest_paths=shard_manifest_paths, output_dir=output_dir
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
            raise D108Target25Error("shard predict requires prepared inputs")
        plan, context = _prepared_target25_inputs(
            plan_manifest_path=Path(plan_manifest_path),
            expected_plan_file_sha256=str(expected_plan_file_sha256),
            context_manifest_path=Path(context_manifest_path),
            expected_context_file_sha256=str(expected_context_file_sha256),
        )
        with _temporary_projection(runner125, _RUNNER_REPLACEMENTS):
            return runner125._predict_shard(
                plan=plan,
                context=context,
                output_dir=output_dir,
                shard_index=int(shard_index),
                device=device,
                feature_batch_size=feature_batch_size,
                state_materializer=state_materializer,
                pair_builder=pair_builder,
                query_scorer=query_scorer,
                d92_fit=d92_fit,
            )
    except runner125.D108Target125RunnerError as error:
        raise D108Target25Error(f"Target25 publication failed closed: {error}") from error


def validate_d108_target25_prediction_manifest(
    *,
    prediction_manifest_path: Path,
    expected_prediction_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        with _temporary_projection(runner125, _RUNNER_REPLACEMENTS):
            return runner125.validate_d108_target125_prediction_manifest(
                prediction_manifest_path=prediction_manifest_path,
                expected_prediction_manifest_file_sha256=(
                    expected_prediction_manifest_file_sha256
                ),
            )
    except runner125.D108Target125RunnerError as error:
        raise D108Target25Error(f"Target25 validation failed closed: {error}") from error


def build_d108_target25_truth_catalog(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    validate_d108_target25_prediction_manifest(
        prediction_manifest_path=Path(prediction_manifest_path),
        expected_prediction_manifest_file_sha256=(
            expected_prediction_manifest_file_sha256
        ),
    )
    try:
        with _temporary_projection(truth125, _TRUTH_REPLACEMENTS):
            return truth125.build_d108_target125_truth_catalog(
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
    except truth125.D108TruthScorerError as error:
        raise D108Target25Error(f"Target25 truth build failed closed: {error}") from error


def score_d108_target25(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    validate_d108_target25_prediction_manifest(
        prediction_manifest_path=Path(prediction_manifest_path),
        expected_prediction_manifest_file_sha256=(
            expected_prediction_manifest_file_sha256
        ),
    )
    try:
        with _temporary_projection(truth125, _TRUTH_REPLACEMENTS):
            return truth125.score_d108_target125(
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
    except truth125.D108TruthScorerError as error:
        raise D108Target25Error(f"Target25 score failed closed: {error}") from error


__all__ = [
    "ARMS",
    "ARM_PAIR_COUNT",
    "CANDIDATE_ID",
    "D108Target25Error",
    "FROZEN_SEED",
    "OUTER_JOB_COUNT",
    "PHASES",
    "PREDICTION_MANIFEST_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SCENES",
    "SCENE_ROW_COUNT",
    "SHARD_COUNT",
    "SURFACE_COUNT",
    "build_d108_target25_truth_catalog",
    "freeze_d108_target25_matrix",
    "predict_d108_target25",
    "prepare_d108_target25_run",
    "score_d108_target25",
    "smoke_d108_target25_prepared_state",
    "validate_d108_target25_prediction_manifest",
]
