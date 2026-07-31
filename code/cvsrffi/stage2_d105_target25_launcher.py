"""Executable Target25 adapter for the real paired D105 query evaluator.

The runner owns immutable coverage and truth separation.  This adapter owns
the missing execution bridge: an immutable context manifest reconstructs the
exact D92 package refs, split authorities, Phase1 asset, checkpoint and qKNN
lock for every frozen row. It binds each row to one requested GPU, runs
``evaluate_d105_query_row`` exactly once, and then serves its already sealed
S_B/S_C states to the Target25 runner. It deliberately cannot evaluate an
after-only state. The public callback constructor remains injectable only for
local tests; the formal CLI uses the manifest-backed loader.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from .stage2_d105_target25_runner import (
    D105Target25ExecutionSummary,
    D105Target25OuterRow,
    D105Target25Plan,
    D105Target25PredictionOutput,
    D105Target25PredictionRequest,
    D105Target25Run,
    D105Target25RunnerError,
    LEO_SCENARIOS,
    OUTER_ROW_COUNT,
    PROTOCOL_SCHEMA,
    canonical_sha256,
    execute_d105_target25_predictions,
)


CONTEXT_MANIFEST_SCHEMA = "cvs.phase2.d105.target25_context_manifest.v1"


class D105Target25LauncherError(ValueError):
    """Raised when an evaluator/context cannot satisfy the frozen launcher."""


D105Target25ContextFactory = Callable[[D105Target25OuterRow, int], Any]
D105Target25Evaluator = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class D105Target25EvaluatorExecution:
    """Observable local bridge state; it contains no labels or metrics."""

    row_id: str
    gpu_id: int
    evaluation: Any


def _rows_by_id(plan: D105Target25Plan) -> dict[str, D105Target25OuterRow]:
    if type(plan) is not D105Target25Plan:
        raise D105Target25LauncherError("exact frozen Target25 plan required")
    rows = {row.row_id: row for row in plan.rows}
    if len(rows) != len(plan.rows):
        raise D105Target25LauncherError("frozen Target25 row-ID closure drift")
    return rows


def _validate_complete_row_evaluation(
    evaluation: Any, *, row: D105Target25OuterRow, request: D105Target25PredictionRequest
) -> None:
    """Require a full S_B/S_C×three-scenario result before exposing one state."""

    pairs = getattr(evaluation, "scenario_pairs", None)
    if (
        str(getattr(evaluation, "receiver", "")) != row.receiver
        or getattr(evaluation, "seed", None) != request.seed
        or getattr(evaluation, "k_shot", None) != row.k_shot
        or not isinstance(pairs, tuple)
        or tuple(getattr(pair, "scenario", None) for pair in pairs) != LEO_SCENARIOS
        or not callable(getattr(evaluation, "target25_output_for", None))
    ):
        raise D105Target25LauncherError(
            "D105 evaluator did not return the complete frozen Target25 row"
        )
    for pair in pairs:
        before = getattr(pair, "before", None)
        after = getattr(pair, "after", None)
        if (
            getattr(before, "stage", None) != "S_B"
            or getattr(before, "registration_state", None) != "BEFORE_REGISTRATION"
            or getattr(after, "stage", None) != "S_C"
            or getattr(after, "registration_state", None) != "AFTER_REGISTRATION"
            or not set(getattr(before, "query_physical_ids", ()))
            < set(getattr(after, "query_physical_ids", ()))
        ):
            raise D105Target25LauncherError(
                "D105 evaluator did not seal a strict before/after state pair"
            )


def make_d105_target25_evaluator_predictor(
    plan: D105Target25Plan,
    context_factory: D105Target25ContextFactory,
    *,
    evaluate_row: D105Target25Evaluator | None = None,
) -> Callable[[D105Target25PredictionRequest], D105Target25PredictionOutput]:
    """Build a truth-free callback backed by one real evaluator pass per row.

    The callback is intentionally lazy only to fit the runner's health-stop
    semantics.  The first S_B request for a row runs the evaluator over all
    three scenarios and both states; the following S_C request merely obtains
    the paired sealed view from that completed evaluation.
    """

    rows = _rows_by_id(plan)
    if not callable(context_factory):
        raise D105Target25LauncherError("GPU-bound D105 context factory is required")
    if evaluate_row is None:
        from .stage2_d105_query_evaluation import evaluate_d105_query_row

        evaluate_row = evaluate_d105_query_row
    if not callable(evaluate_row):
        raise D105Target25LauncherError("D105 row evaluator must be callable")
    completed: dict[str, D105Target25EvaluatorExecution] = {}

    def predictor(request: D105Target25PredictionRequest) -> D105Target25PredictionOutput:
        row = rows.get(request.row_id)
        if row is None or (
            request.receiver != row.receiver
            or request.k_shot != row.k_shot
            or request.new_count != row.new_count
        ):
            raise D105Target25LauncherError("Target25 request/row binding drift")
        execution = completed.get(row.row_id)
        if execution is None:
            context = context_factory(row, request.gpu_id)
            evaluation = evaluate_row(context)
            _validate_complete_row_evaluation(
                evaluation, row=row, request=request
            )
            execution = D105Target25EvaluatorExecution(
                row_id=row.row_id,
                gpu_id=request.gpu_id,
                evaluation=evaluation,
            )
            completed[row.row_id] = execution
        elif execution.gpu_id != request.gpu_id:
            raise D105Target25LauncherError("row GPU assignment drift after evaluation")
        output = execution.evaluation.target25_output_for(request)
        if type(output) is not D105Target25PredictionOutput:
            raise D105Target25LauncherError(
                "D105 evaluator adapter did not return exact Target25 output"
            )
        return output

    return predictor


def execute_d105_target25_with_evaluator(
    run: D105Target25Run,
    context_factory: D105Target25ContextFactory,
    *,
    evaluate_row: D105Target25Evaluator | None = None,
) -> D105Target25ExecutionSummary:
    """Execute the formal matrix through the complete paired D105 evaluator."""

    try:
        predictor = make_d105_target25_evaluator_predictor(
            run.plan, context_factory, evaluate_row=evaluate_row
        )
        return execute_d105_target25_predictions(run, predictor)
    except D105Target25RunnerError:
        raise
    except D105Target25LauncherError:
        raise
    except Exception as error:
        raise D105Target25LauncherError(
            "D105 Target25 evaluator launch preparation failed"
        ) from error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D105Target25LauncherError(f"{name} must be a lowercase SHA256")
    return text


def _read_immutable_json(path: Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D105Target25LauncherError(
            f"context manifest must be a regular immutable JSON file: {source}"
        )
    if source.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25LauncherError("context manifest is writable")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105Target25LauncherError("context manifest JSON cannot be read") from error
    if type(value) is not dict:
        raise D105Target25LauncherError("context manifest must contain a JSON object")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable output already exists: {destination}")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise D105Target25LauncherError("context manifest parent is unsafe")
    payload = _canonical_bytes(value) + b"\n"
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(destination, stat.S_IREAD)
    if destination.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25LauncherError("context manifest remained writable")
    return hashlib.sha256(payload).hexdigest()


def seal_d105_target25_context_manifest(
    path: Path, document_without_receipt: Mapping[str, Any]
) -> str:
    """Publish a read-only context manifest; loading performs full preflight."""

    if not isinstance(document_without_receipt, Mapping):
        raise D105Target25LauncherError("context manifest input must be an object")
    document = dict(document_without_receipt)
    if "context_manifest_receipt_sha256" in document:
        raise D105Target25LauncherError("context manifest receipt must be runner-generated")
    document["context_manifest_receipt_sha256"] = canonical_sha256(document)
    return _write_immutable_json(Path(path), document)


def _existing_path(value: Any, *, name: str, directory: bool) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or path.is_symlink() or not path.exists():
        raise D105Target25LauncherError(f"{name} must be an existing absolute non-symlink path")
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        expected = "directory" if directory else "file"
        raise D105Target25LauncherError(f"{name} must be a regular {expected}")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_reference(value: Any, *, name: str) -> Any:
    from .stage2_d105_query_evaluation import D105SealedPackageRef

    expected = {
        "package_root",
        "detached_seal_path",
        "expected_seal_sha256",
        "formal_policy_path",
        "formal_policy_authorization_path",
        "signed_policy_authorization_envelope_path",
        "expected_signed_policy_authorization_envelope_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise D105Target25LauncherError(f"{name} package reference field closure drift")
    root = _existing_path(value["package_root"], name=f"{name}.package_root", directory=True)
    seal = _existing_path(
        value["detached_seal_path"], name=f"{name}.detached_seal_path", directory=False
    )
    policy = _existing_path(
        value["formal_policy_path"],
        name=f"{name}.formal_policy_path",
        directory=False,
    )
    authorization = _existing_path(
        value["formal_policy_authorization_path"],
        name=f"{name}.formal_policy_authorization_path",
        directory=False,
    )
    envelope = _existing_path(
        value["signed_policy_authorization_envelope_path"],
        name=f"{name}.signed_policy_authorization_envelope_path",
        directory=False,
    )
    expected_envelope_sha256 = _require_sha256(
        value["expected_signed_policy_authorization_envelope_sha256"],
        f"{name}.expected_signed_policy_authorization_envelope_sha256",
    )
    if _sha256_file(envelope) != expected_envelope_sha256:
        raise D105Target25LauncherError(
            f"{name} signed policy authorization envelope SHA256 drift"
        )
    return D105SealedPackageRef(
        package_root=root,
        detached_seal_path=seal,
        expected_seal_sha256=_require_sha256(
            value["expected_seal_sha256"], f"{name}.expected_seal_sha256"
        ),
        formal_policy_path=policy,
        formal_policy_authorization_path=authorization,
        signed_policy_authorization_envelope_path=envelope,
        expected_signed_policy_authorization_envelope_sha256=(
            expected_envelope_sha256
        ),
    )


def _phase1_authority(value: Any, *, plan: D105Target25Plan) -> Any:
    from .stage2_d105_query_evaluation import D105Phase1BundleAuthority

    expected = {
        "bundle_dir",
        "manifest_sha256",
        "bundle_wire_sha256",
        "validated_bundle_id_sha256",
        "validator_receipt_sha256",
        "expected_content_root_sha256",
        "checkpoint_sha256",
        "candidate_runtime_manifest_path",
        "candidate_method_lock_path",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise D105Target25LauncherError("Phase1 authority field closure drift")
    authority = D105Phase1BundleAuthority(
        bundle_dir=_existing_path(value["bundle_dir"], name="phase1.bundle_dir", directory=True),
        manifest_sha256=_require_sha256(value["manifest_sha256"], "phase1.manifest_sha256"),
        bundle_wire_sha256=_require_sha256(
            value["bundle_wire_sha256"], "phase1.bundle_wire_sha256"
        ),
        validated_bundle_id_sha256=_require_sha256(
            value["validated_bundle_id_sha256"], "phase1.validated_bundle_id_sha256"
        ),
        validator_receipt_sha256=_require_sha256(
            value["validator_receipt_sha256"], "phase1.validator_receipt_sha256"
        ),
        expected_content_root_sha256=_require_sha256(
            value["expected_content_root_sha256"], "phase1.expected_content_root_sha256"
        ),
        checkpoint_sha256=_require_sha256(
            value["checkpoint_sha256"], "phase1.checkpoint_sha256"
        ),
        candidate_runtime_manifest_path=_existing_path(
            value["candidate_runtime_manifest_path"],
            name="phase1.candidate_runtime_manifest_path",
            directory=False,
        ),
        candidate_method_lock_path=_existing_path(
            value["candidate_method_lock_path"],
            name="phase1.candidate_method_lock_path",
            directory=False,
        ),
        d105_candidate_runtime_manifest_sha256=_require_sha256(
            value["d105_candidate_runtime_manifest_sha256"],
            "phase1.d105_candidate_runtime_manifest_sha256",
        ),
        d105_candidate_method_lock_sha256=_require_sha256(
            value["d105_candidate_method_lock_sha256"],
            "phase1.d105_candidate_method_lock_sha256",
        ),
    )
    from .stage2_d105_phase1_bundle import (
        load_d105_candidate_method_lock,
        load_d105_candidate_runtime_manifest,
    )

    try:
        runtime = load_d105_candidate_runtime_manifest(
            authority.candidate_runtime_manifest_path,
            expected_checkpoint_sha256=authority.checkpoint_sha256,
        )
        candidate_lock = load_d105_candidate_method_lock(
            authority.candidate_method_lock_path,
            expected_checkpoint_sha256=authority.checkpoint_sha256,
            expected_runtime_sha256=runtime[
                "d105_candidate_runtime_manifest_sha256"
            ],
        )
    except ValueError as error:
        raise D105Target25LauncherError(
            "Phase1 canonical candidate identity preflight failed"
        ) from error
    if (
        runtime["d105_candidate_runtime_manifest_sha256"]
        != plan.d105_candidate_runtime_manifest_sha256
        or candidate_lock["d105_candidate_method_lock_sha256"]
        != plan.d105_candidate_method_lock_sha256
        or authority.d105_candidate_runtime_manifest_sha256
        != plan.d105_candidate_runtime_manifest_sha256
        or authority.d105_candidate_method_lock_sha256
        != plan.d105_candidate_method_lock_sha256
    ):
        raise D105Target25LauncherError("Phase1 authority/candidate identity drift")
    return authority


def _qknn_lock(value: Any, *, row: D105Target25OuterRow) -> Any:
    from .stage2_zid_student_t_qknn import Phase1ZIDStudentTLock

    expected = {field.name for field in fields(Phase1ZIDStudentTLock)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise D105Target25LauncherError("qKNN lock field closure drift")
    try:
        lock = Phase1ZIDStudentTLock(**dict(value))
    except (TypeError, ValueError) as error:
        raise D105Target25LauncherError("qKNN lock cannot be reconstructed") from error
    if lock.active_k != row.k_shot:
        raise D105Target25LauncherError("qKNN lock K-shot/row drift")
    return lock


def _split_authorities(
    value: Any, *, row: D105Target25OuterRow
) -> tuple[Any, ...]:
    from .stage2_d105_query_evaluation import D105SplitAuthority

    expected_fields = {
        "registration_state",
        "scenario",
        "capsule_id",
        "split_id",
        "validator_receipt_sha256",
        "support_token_root_sha256",
        "query_token_root_sha256",
        "protocol_schema",
        "phase2_data_status",
    }
    if not isinstance(value, list) or len(value) != 6:
        raise D105Target25LauncherError("split authority must close at six state/scenario rows")
    expected_states = {
        ("BEFORE_REGISTRATION", scenario.scenario): scenario.before
        for scenario in row.scenarios
    } | {
        ("AFTER_REGISTRATION", scenario.scenario): scenario.after
        for scenario in row.scenarios
    }
    authorities = []
    observed: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise D105Target25LauncherError("split authority field closure drift")
        try:
            authority = D105SplitAuthority(**dict(item))
        except (TypeError, ValueError) as error:
            raise D105Target25LauncherError("split authority cannot be reconstructed") from error
        key = (authority.registration_state, authority.scenario)
        state = expected_states.get(key)
        if key in observed or state is None or (
            authority.capsule_id != state.capsule_id
            or authority.split_id != state.split_id
            or authority.validator_receipt_sha256 != state.authority_receipt_sha256
            or authority.support_token_root_sha256 != state.support_physical_root_sha256
            or authority.query_token_root_sha256 != state.query_physical_root_sha256
            or authority.protocol_schema != PROTOCOL_SCHEMA
            or authority.phase2_data_status != "VALIDATED_ONCE"
        ):
            raise D105Target25LauncherError("split authority/Target25 state binding drift")
        observed.add(key)
        authorities.append(authority)
    if set(expected_states) != observed:
        raise D105Target25LauncherError("split authority state/scenario coverage drift")
    return tuple(authorities)


@dataclass(frozen=True, slots=True)
class _PreparedContextRow:
    row: D105Target25OuterRow
    before_enrollment: Any
    before_apply: Any
    after_enrollment: Any
    after_apply: Any
    split_authorities: tuple[Any, ...]
    phase1_bundle: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    data_feature_runtime_sha256: str
    data_materialization_lock_sha256: str
    qknn_lock: Any
    feature_batch_size: int
    score_chunk_size: int | None


def load_d105_target25_context_factory(
    path: Path, plan: D105Target25Plan
) -> D105Target25ContextFactory:
    """Load the complete JSON-backed D92/D105 context surface for Target25.

    This is the formal launcher entry: it checks exact row coverage, package
    and checkpoint paths, Phase1 formal-asset authority, qKNN locks and all six
    before/after split roots before returning GPU-bound real evaluator contexts.
    """

    from .stage2_d105_phase1_bundle import load_d105_phase1_asset
    from .stage2_d105_query_evaluation import D105QueryEvaluationContext

    if type(plan) is not D105Target25Plan:
        raise D105Target25LauncherError("exact frozen Target25 plan required")
    document = _read_immutable_json(Path(path))
    expected_document_fields = {
        "schema",
        "plan_receipt_sha256",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "rows",
        "context_manifest_receipt_sha256",
    }
    if set(document) != expected_document_fields or (
        document.get("schema") != CONTEXT_MANIFEST_SCHEMA
        or document.get("plan_receipt_sha256") != plan.plan_receipt_sha256
        or document.get("claim_scope") != plan.claim_scope
        or document.get("formal_launch_authority") is not plan.formal_launch_authority
        or document.get("authority_envelope_root_sha256")
        != plan.authority_envelope_root_sha256
        or any(
            document.get(name) != getattr(plan, name)
            for name in (
                "data_feature_runtime_sha256",
                "data_materialization_lock_sha256",
                "d105_candidate_runtime_manifest_sha256",
                "d105_candidate_method_lock_sha256",
            )
        )
        or document.get("context_manifest_receipt_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in document.items()
                if key != "context_manifest_receipt_sha256"
            }
        )
        or not isinstance(document.get("rows"), list)
        or len(document["rows"]) != OUTER_ROW_COUNT
    ):
        raise D105Target25LauncherError("context manifest top-level closure drift")
    row_fields = {
        "row_id",
        "receiver",
        "k_shot",
        "new_count",
        "before_enrollment",
        "before_apply",
        "after_enrollment",
        "after_apply",
        "split_authorities",
        "phase1_bundle",
        "checkpoint_path",
        "checkpoint_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "qknn_lock",
        "feature_batch_size",
        "score_chunk_size",
    }
    planned_rows = _rows_by_id(plan)
    serialized_rows = document["rows"]
    if [
        item.get("row_id") for item in serialized_rows if isinstance(item, Mapping)
    ] != [row.row_id for row in plan.rows]:
        raise D105Target25LauncherError("context manifest frozen row order drift")
    prepared: dict[str, _PreparedContextRow] = {}
    phase1_checked: dict[tuple[str, str], Any] = {}
    checkpoint_checked: dict[tuple[str, str], None] = {}
    for value in serialized_rows:
        if not isinstance(value, Mapping) or set(value) != row_fields:
            raise D105Target25LauncherError("context manifest row field closure drift")
        row = planned_rows.get(str(value["row_id"]))
        if row is None or (
            value.get("receiver") != row.receiver
            or value.get("k_shot") != row.k_shot
            or value.get("new_count") != row.new_count
            or value.get("data_feature_runtime_sha256")
            != plan.data_feature_runtime_sha256
            or value.get("data_materialization_lock_sha256")
            != plan.data_materialization_lock_sha256
        ):
            raise D105Target25LauncherError("context manifest row/plan binding drift")
        phase1 = _phase1_authority(value["phase1_bundle"], plan=plan)
        checkpoint = _existing_path(
            value["checkpoint_path"], name="checkpoint_path", directory=False
        )
        checkpoint_sha = _require_sha256(value["checkpoint_sha256"], "checkpoint_sha256")
        data_runtime_sha = _require_sha256(
            value["data_feature_runtime_sha256"], "data_feature_runtime_sha256"
        )
        data_lock_sha = _require_sha256(
            value["data_materialization_lock_sha256"],
            "data_materialization_lock_sha256",
        )
        if (
            checkpoint_sha != phase1.checkpoint_sha256
            or data_runtime_sha != plan.data_feature_runtime_sha256
            or data_lock_sha != plan.data_materialization_lock_sha256
        ):
            raise D105Target25LauncherError("context/Phase1 common identity drift")
        checkpoint_key = (str(checkpoint), checkpoint_sha)
        if checkpoint_key not in checkpoint_checked:
            if _sha256_file(checkpoint) != checkpoint_sha:
                raise D105Target25LauncherError("checkpoint SHA256 drift")
            checkpoint_checked[checkpoint_key] = None
        phase1_key = (str(phase1.bundle_dir), phase1.manifest_sha256)
        if phase1_key not in phase1_checked:
            try:
                asset = load_d105_phase1_asset(
                    phase1.bundle_dir, require_formal_phase2_eligible=True
                )
            except ValueError as error:
                raise D105Target25LauncherError(
                    "formal D105 Phase1 asset preflight failed"
                ) from error
            if (
                asset.manifest_sha256 != phase1.manifest_sha256
                or asset.manifest.get("bundle_wire_sha256") != phase1.bundle_wire_sha256
                or str(asset.validated_bundle_id_sha256)
                != phase1.validated_bundle_id_sha256
                or str(asset.validator_receipt_sha256)
                != phase1.validator_receipt_sha256
                or asset.bundle.content_root_sha256 != phase1.expected_content_root_sha256
                or asset.bundle.checkpoint_sha256 != phase1.checkpoint_sha256
                or asset.bundle.runtime_sha256
                != phase1.d105_candidate_runtime_manifest_sha256
                or asset.bundle.method_lock_sha256
                != phase1.d105_candidate_method_lock_sha256
            ):
                raise D105Target25LauncherError("formal Phase1 asset authority drift")
            phase1_checked[phase1_key] = asset
        feature_batch_size = value["feature_batch_size"]
        score_chunk_size = value["score_chunk_size"]
        if (
            type(feature_batch_size) is not int
            or feature_batch_size < 1
            or (
                score_chunk_size is not None
                and (type(score_chunk_size) is not int or score_chunk_size < 1)
            )
        ):
            raise D105Target25LauncherError("context batch/chunk configuration drift")
        prepared[row.row_id] = _PreparedContextRow(
            row=row,
            before_enrollment=_package_reference(
                value["before_enrollment"], name="before_enrollment"
            ),
            before_apply=_package_reference(value["before_apply"], name="before_apply"),
            after_enrollment=_package_reference(
                value["after_enrollment"], name="after_enrollment"
            ),
            after_apply=_package_reference(value["after_apply"], name="after_apply"),
            split_authorities=_split_authorities(value["split_authorities"], row=row),
            phase1_bundle=phase1,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            data_feature_runtime_sha256=data_runtime_sha,
            data_materialization_lock_sha256=data_lock_sha,
            qknn_lock=_qknn_lock(value["qknn_lock"], row=row),
            feature_batch_size=feature_batch_size,
            score_chunk_size=score_chunk_size,
        )
    if len(prepared) != OUTER_ROW_COUNT:
        raise D105Target25LauncherError("context manifest row uniqueness drift")

    def factory(row: D105Target25OuterRow, gpu_id: int) -> Any:
        if type(row) is not D105Target25OuterRow or type(gpu_id) is not int or gpu_id < 0:
            raise D105Target25LauncherError("context factory row/GPU input drift")
        item = prepared.get(row.row_id)
        if item is None or item.row is not row:
            raise D105Target25LauncherError("context factory received an unsealed row")
        return D105QueryEvaluationContext(
            before_enrollment=item.before_enrollment,
            before_apply=item.before_apply,
            after_enrollment=item.after_enrollment,
            after_apply=item.after_apply,
            split_authorities=item.split_authorities,
            phase1_bundle=item.phase1_bundle,
            checkpoint_path=item.checkpoint_path,
            checkpoint_sha256=item.checkpoint_sha256,
            data_feature_runtime_sha256=item.data_feature_runtime_sha256,
            data_materialization_lock_sha256=item.data_materialization_lock_sha256,
            qknn_lock=item.qknn_lock,
            device=f"cuda:{gpu_id}",
            feature_batch_size=item.feature_batch_size,
            score_chunk_size=item.score_chunk_size,
        )

    return factory


__all__ = [
    "CONTEXT_MANIFEST_SCHEMA",
    "D105Target25ContextFactory",
    "D105Target25EvaluatorExecution",
    "D105Target25LauncherError",
    "execute_d105_target25_with_evaluator",
    "load_d105_target25_context_factory",
    "make_d105_target25_evaluator_predictor",
    "seal_d105_target25_context_manifest",
]
