"""Minimal executable prepare/predict/score runner for D106 Target25.

Prediction has no truth input.  A local state-input factory owns the bounded
opening of the sealed support/query IQ references and returns arguments for
``evaluate_d106_target25_state``.  The runner seals all four arms first and
only then invokes the frozen K-only router.  Scoring validates the complete
25/75/300/600 prediction closure and writes a truth-open event before the
independent catalog is opened.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any

import numpy as np

from .stage2_d105_feature_tap import extract_d105_feature_tap
from .stage2_d105_query_evaluation import (
    _default_model_loader,
    _device,
    _query_rows,
    _support_rows,
    _tap_rows,
)
from .somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
from .stage2_diag_cosine_exploration import _validate_matched_packages
from .stage2_d106_k_conditioned_router import (
    ARMS,
    ROUTE_BY_K,
    route_d106_k_conditioned_prediction,
)
from .stage2_d106_matrix_protocol import (
    LEO_SCENARIOS,
    MATCHED_ARM_PAIR_COUNT,
    OUTER_JOB_COUNT,
    SCENARIO_ROW_COUNT,
    STATE_SURFACE_COUNT,
    STATES,
    canonical_sha256,
    freeze_d106_matrix_protocol,
)
from .stage2_d106_rcmr_2v_qknn import load_d106_rcmr_2v_method_lock
from .stage2_d106_rcmr_g0 import PREDECESSOR_NUMERIC_LOCK
from .stage2_d106_rdce_asset import (
    D106RDCEAssetLineage,
    WIRE_MAGIC,
    deserialize_d106_rdce_asset,
)
from .stage2_d106_rdce_runtime import (
    D106RDCESupportRows,
    ROW_AUTHORITY_SCHEMA,
    load_d106_rdce_row_authority,
)
from .stage2_d106_target25_evaluator import (
    PLAN_STATE_SCHEMA,
    evaluate_d106_target25_state,
    load_d106_paired_features,
    load_d106_target25_plan_state,
    publish_d106_paired_features,
    publish_d106_target25_plan_state,
)
from .stage2_d106_target25_inputs import (
    CONTEXT_SCHEMA,
    PLAN_SCHEMA,
    prepare_d106_target25_inputs,
)
from .stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle
from .stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
)


PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d106.target25.prediction_manifest.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d106.target25.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d106.target25.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d106.target25.score_manifest.v1"
METHODS = (*ARMS, "ROUTED")
_PREDICTION_MANIFEST_FIELDS = {
    "schema",
    "matrix_receipt_sha256",
    "plan_receipt_sha256",
    "context_receipt_sha256",
    "kcr_route_lock_sha256",
    "outer_job_count",
    "scenario_row_count",
    "matched_arm_pair_count",
    "state_surface_count",
    "state_prediction_count",
    "target_access",
    "clean_source_runtime_access",
    "query_fit_count",
    "query_update_count",
    "query_selection_count",
    "rows",
    "prediction_manifest_receipt_sha256",
}


class D106Target25RunnerError(ValueError):
    """Raised when a Target25 execution or artifact fails closed."""


StateInputFactory = Callable[[Mapping[str, Any]], Mapping[str, Any]]
StateEvaluator = Callable[..., Mapping[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D106Target25RunnerError("canonical JSON payload is invalid") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D106Target25RunnerError(f"{name} must be a lowercase SHA256")
    return value


def _regular_file(path: Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D106Target25RunnerError(f"{name} must be a regular non-symlink file")
    return source.resolve(strict=True)


def _read_json(
    path: Path, *, name: str, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    source = _regular_file(path, name)
    if expected_file_sha256 is not None and _sha256_file(source) != _sha(
        expected_file_sha256, f"expected {name} file SHA256"
    ):
        raise D106Target25RunnerError(f"{name} file SHA mismatch")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise D106Target25RunnerError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise D106Target25RunnerError(f"{name} must contain an object")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    raw = _canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(raw).hexdigest()


def _receipt(document: Mapping[str, Any], field: str, name: str) -> str:
    expected = _sha(document.get(field), f"{name} {field}")
    payload = {key: value for key, value in document.items() if key != field}
    if canonical_sha256(payload) != expected:
        raise D106Target25RunnerError(f"{name} canonical receipt drift")
    return expected


def _sequence(value: Any, name: str, expected_len: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (
        expected_len is not None and len(value) != expected_len
    ):
        qualifier = "" if expected_len is None else f" with {expected_len} items"
        raise D106Target25RunnerError(f"{name} must be a list{qualifier}")
    return value


_RAW_ROW_FIELDS = {
    "job_id",
    "source_d92_job_id",
    "receiver",
    "seed",
    "k_shot",
    "new_count",
    "packages",
}
_RAW_PACKAGE_FIELDS = {
    "package_root",
    "detached_seal_path",
    "expected_seal_sha256",
}


def _load_raw_package(value: Mapping[str, Any]):
    if not isinstance(value, Mapping) or set(value) != _RAW_PACKAGE_FIELDS:
        raise D106Target25RunnerError("raw D92 package reference closure drift")
    root = Path(str(value["package_root"]))
    seal = Path(str(value["detached_seal_path"]))
    expected = _sha(value["expected_seal_sha256"], "D92 package seal SHA256")
    try:
        payloads, manifest, audit = load_verified_somph_predictor_bundle(
            root,
            detached_seal_path=seal,
            expected_seal_sha256=expected,
        )
    except Exception as error:
        raise D106Target25RunnerError("sealed D92 package verification failed") from error
    if not isinstance(manifest, Mapping) or not isinstance(payloads, Mapping):
        raise D106Target25RunnerError("sealed D92 package materialization drift")
    return payloads, dict(manifest), dict(audit)


def _derived_state(
    *,
    row: Mapping[str, Any],
    scene: str,
    state_name: str,
    support_ref: Mapping[str, Any],
    query_ref: Mapping[str, Any],
    support_loaded: tuple[Any, Mapping[str, Any], Mapping[str, Any]],
    query_loaded: tuple[Any, Mapping[str, Any], Mapping[str, Any]],
    old_registry: tuple[str, ...] | None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    support_payloads, support_manifest, _support_audit = support_loaded
    query_payloads, query_manifest, _query_audit = query_loaded
    try:
        _validate_matched_packages(support_manifest, query_manifest)
    except Exception as error:
        raise D106Target25RunnerError("D92 support/query package pairing drift") from error
    if scene not in support_payloads or scene not in query_payloads:
        raise D106Target25RunnerError("D92 package scenario missing")
    registry = tuple(
        str(item.get("class_handle", ""))
        for item in support_manifest.get("registered_classes", [])
        if isinstance(item, Mapping)
    )
    if not registry or len(set(registry)) != len(registry):
        raise D106Target25RunnerError("D92 registered-class manifest drift")
    for manifest in (support_manifest, query_manifest):
        if (
            manifest.get("receiver") != row["receiver"]
            or manifest.get("seed") != row["seed"]
            or manifest.get("k_shot") != row["k_shot"]
        ):
            raise D106Target25RunnerError("D92 package row binding drift")
    try:
        _support_iq, _support_labels, support_ids = _support_rows(
            support_payloads[scene],
            registered_classes=registry,
            active_k=row["k_shot"],
        )
        _query_iq, query_ids = _query_rows(query_payloads[scene])
    except Exception as error:
        raise D106Target25RunnerError("D92 physical-ID materialization drift") from error
    support_ids = tuple(str(value) for value in support_ids)
    query_ids = tuple(str(value) for value in query_ids)
    if set(support_ids).intersection(query_ids):
        raise D106Target25RunnerError("D92 support/query physical IDs overlap")
    if state_name == "before":
        old = registry
        new: tuple[str, ...] = ()
    else:
        if old_registry is None or registry[: len(old_registry)] != old_registry:
            raise D106Target25RunnerError("D92 before/after registry prefix drift")
        old = old_registry
        new = registry[len(old_registry) :]
        if len(new) != row["new_count"]:
            raise D106Target25RunnerError("D92 new-class count drift")
    support_root = canonical_sha256(sorted(support_ids))
    query_root = canonical_sha256(sorted(query_ids))
    capsule_id = canonical_sha256(
        {
            "schema": "cvs.phase2.d106.d92_sealed_package_capsule.v1",
            "row": row["source_d92_job_id"],
            "state": state_name,
            "support_seal": support_ref["expected_seal_sha256"],
            "query_seal": query_ref["expected_seal_sha256"],
            "support_package_root": support_manifest.get("package_root_sha256"),
            "query_package_root": query_manifest.get("package_root_sha256"),
        }
    )
    split_id = canonical_sha256(
        {
            "schema": "cvs.phase2.d106.d92_materialized_split.v1",
            "capsule_id": capsule_id,
            "scene": scene,
            "state": state_name,
            "support_physical_ids": list(support_ids),
            "query_physical_ids": list(query_ids),
        }
    )
    validator_receipt = canonical_sha256(
        {
            "schema": "cvs.phase2.d106.d92_seal_verification_receipt.v1",
            "support_seal": support_ref["expected_seal_sha256"],
            "query_seal": query_ref["expected_seal_sha256"],
            "support_package_root": support_manifest.get("package_root_sha256"),
            "query_package_root": query_manifest.get("package_root_sha256"),
            "support_query_disjoint": True,
        }
    )
    registration_state = (
        "BEFORE_REGISTRATION" if state_name == "before" else "AFTER_REGISTRATION"
    )
    plan_state: dict[str, Any] = {
        "state": state_name,
        "registration_state": registration_state,
        "registered_classes": list(registry),
        "old_classes": list(old),
        "new_classes": list(new),
        "capsule_id": capsule_id,
        "split_id": split_id,
        "authority_receipt_sha256": validator_receipt,
        "support_physical_root_sha256": support_root,
        "query_physical_root_sha256": query_root,
    }
    plan_state["state_input_receipt_sha256"] = canonical_sha256(plan_state)
    context_state = {
        **plan_state,
        "support_received_iq_ref": dict(support_ref),
        "query_received_iq_ref": dict(query_ref),
    }
    return plan_state, context_state, support_ids, query_ids


def _expand_raw_rows(
    plan: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_plan_rows = _sequence(plan.get("rows"), "raw plan rows", OUTER_JOB_COUNT)
    raw_context_rows = _sequence(context.get("rows"), "raw context rows", OUTER_JOB_COUNT)
    if raw_plan_rows != raw_context_rows:
        raise D106Target25RunnerError("raw plan/context row drift")
    package_cache: dict[tuple[tuple[str, str], ...], tuple[Any, dict[str, Any], dict[str, Any]]] = {}

    def loaded(ref: Mapping[str, Any]):
        key = tuple(sorted((str(name), str(value)) for name, value in ref.items()))
        if key not in package_cache:
            package_cache[key] = _load_raw_package(ref)
        return package_cache[key]

    expanded_plan_rows: list[dict[str, Any]] = []
    expanded_context_rows: list[dict[str, Any]] = []
    physical: dict[
        tuple[str, int, int, str, str], tuple[tuple[str, ...], tuple[str, ...]]
    ] = {}
    for raw_row in raw_plan_rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != _RAW_ROW_FIELDS:
            raise D106Target25RunnerError("raw D92 row closure drift")
        packages = raw_row.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != {
            "before_enrollment", "before_apply", "after_enrollment", "after_apply"
        }:
            raise D106Target25RunnerError("raw D92 four-package closure drift")
        before_registry: tuple[str, ...] | None = None
        plan_scenes: list[dict[str, Any]] = []
        context_scenes: list[dict[str, Any]] = []
        for scene in LEO_SCENARIOS:
            plan_states: list[dict[str, Any]] = []
            context_states: list[dict[str, Any]] = []
            for state_name in STATES:
                support_ref = packages[f"{state_name}_enrollment"]
                query_ref = packages[f"{state_name}_apply"]
                plan_state, context_state, support_ids, query_ids = _derived_state(
                    row=raw_row,
                    scene=scene,
                    state_name=state_name,
                    support_ref=support_ref,
                    query_ref=query_ref,
                    support_loaded=loaded(support_ref),
                    query_loaded=loaded(query_ref),
                    old_registry=before_registry,
                )
                if state_name == "before":
                    before_registry = tuple(plan_state["registered_classes"])
                physical[
                    (
                        raw_row["receiver"],
                        raw_row["k_shot"],
                        raw_row["new_count"],
                        scene,
                        state_name,
                    )
                ] = (
                    support_ids,
                    query_ids,
                )
                plan_states.append(plan_state)
                context_states.append(context_state)
            scenario_row_id = f"{raw_row['job_id']}::{scene}"
            plan_scenes.append(
                {"scenario_row_id": scenario_row_id, "scenario": scene, "states": plan_states}
            )
            context_scenes.append(
                {"scenario_row_id": scenario_row_id, "scenario": scene, "states": context_states}
            )
        base = {name: raw_row[name] for name in ("job_id", "receiver", "seed", "k_shot", "new_count")}
        expanded_plan_rows.append({**base, "scenarios": plan_scenes})
        expanded_context_rows.append({**base, "scenarios": context_scenes})
    for receiver in {row["receiver"] for row in raw_plan_rows}:
        for scene in LEO_SCENARIOS:
            for state_name in STATES:
                short = physical[(receiver, 5, 20, scene, state_name)]
                long = physical[(receiver, 10, 20, scene, state_name)]
                if not set(short[0]).issubset(long[0]) or canonical_sha256(sorted(short[1])) != canonical_sha256(sorted(long[1])):
                    raise D106Target25RunnerError(
                        "D92 K5 support/query pairing differs from matched K10"
                    )
    return ({**dict(plan), "rows": expanded_plan_rows}, {**dict(context), "rows": expanded_context_rows})


def _prepared_inputs(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_json(
        plan_manifest_path,
        name="D106 Target25 plan",
        expected_file_sha256=expected_plan_file_sha256,
    )
    context = _read_json(
        context_manifest_path,
        name="D106 Target25 context",
        expected_file_sha256=expected_context_file_sha256,
    )
    if plan.get("schema") != PLAN_SCHEMA or context.get("schema") != CONTEXT_SCHEMA:
        raise D106Target25RunnerError("prepared plan/context schema drift")
    plan_receipt = _receipt(plan, "plan_receipt_sha256", "plan")
    context_receipt = _receipt(context, "context_receipt_sha256", "context")
    matrix = freeze_d106_matrix_protocol()
    if (
        plan.get("protocol_schema") != "p2_min_v1"
        or context.get("protocol_schema") != "p2_min_v1"
        or plan.get("matrix_protocol") != matrix.receipt_payload()
        or plan.get("identity") != context.get("identity")
        or plan.get("identity", {}).get("matrix_receipt_sha256")
        != matrix.matrix_receipt_sha256
        or context.get("plan_receipt_sha256") != plan_receipt
        or not context_receipt
    ):
        raise D106Target25RunnerError("prepared plan/context identity drift")
    raw_rows = plan.get("rows")
    if (
        isinstance(raw_rows, list)
        and raw_rows
        and all(isinstance(row, Mapping) and set(row) == _RAW_ROW_FIELDS for row in raw_rows)
    ):
        plan, context = _expand_raw_rows(plan, context)
    plan_rows = _sequence(plan.get("rows"), "plan rows", OUTER_JOB_COUNT)
    context_rows = _sequence(context.get("rows"), "context rows", OUTER_JOB_COUNT)
    if [row.get("job_id") for row in plan_rows if isinstance(row, Mapping)] != [
        job.job_id for job in matrix.jobs
    ] or [row.get("job_id") for row in context_rows if isinstance(row, Mapping)] != [
        job.job_id for job in matrix.jobs
    ]:
        raise D106Target25RunnerError("prepared row coverage/order drift")
    for plan_row, context_row, job in zip(
        plan_rows, context_rows, matrix.jobs, strict=True
    ):
        if not isinstance(plan_row, Mapping) or not isinstance(context_row, Mapping):
            raise D106Target25RunnerError("prepared row must be an object")
        for name, expected in (
            ("job_id", job.job_id),
            ("receiver", job.receiver),
            ("seed", job.seed),
            ("k_shot", job.k_shot),
            ("new_count", job.new_count),
        ):
            if plan_row.get(name) != expected or context_row.get(name) != expected:
                raise D106Target25RunnerError("prepared row binding drift")
        plan_scenarios = _sequence(
            plan_row.get("scenarios"), "plan row scenarios", len(LEO_SCENARIOS)
        )
        context_scenarios = _sequence(
            context_row.get("scenarios"),
            "context row scenarios",
            len(LEO_SCENARIOS),
        )
        for plan_scene, context_scene, expected_scene in zip(
            plan_scenarios, context_scenarios, LEO_SCENARIOS, strict=True
        ):
            if (
                not isinstance(plan_scene, Mapping)
                or not isinstance(context_scene, Mapping)
                or plan_scene.get("scenario") != expected_scene
                or context_scene.get("scenario") != expected_scene
                or plan_scene.get("scenario_row_id")
                != context_scene.get("scenario_row_id")
            ):
                raise D106Target25RunnerError("prepared scenario binding drift")
            plan_states = _sequence(plan_scene.get("states"), "plan states", 2)
            context_states = _sequence(context_scene.get("states"), "context states", 2)
            for plan_state, context_state, expected_state in zip(
                plan_states, context_states, STATES, strict=True
            ):
                if (
                    not isinstance(plan_state, Mapping)
                    or not isinstance(context_state, Mapping)
                    or plan_state.get("state") != expected_state
                    or context_state.get("state") != expected_state
                ):
                    raise D106Target25RunnerError("prepared state binding drift")
                projected = {
                    key: value
                    for key, value in context_state.items()
                    if key not in {"support_received_iq_ref", "query_received_iq_ref"}
                }
                if dict(plan_state) != projected:
                    raise D106Target25RunnerError("plan/context state projection drift")
                state_payload = {
                    key: value
                    for key, value in plan_state.items()
                    if key != "state_input_receipt_sha256"
                }
                if plan_state.get("state_input_receipt_sha256") != canonical_sha256(
                    state_payload
                ):
                    raise D106Target25RunnerError("state input receipt drift")
                if not isinstance(context_state.get("support_received_iq_ref"), Mapping) or not isinstance(
                    context_state.get("query_received_iq_ref"), Mapping
                ):
                    raise D106Target25RunnerError("received-IQ locator missing")
    return plan, context


def prepare_d106_target25_run(**kwargs: Any) -> dict[str, Any]:
    """Forward the prepare stage to the immutable D106 input builder."""

    return prepare_d106_target25_inputs(**kwargs)


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _load_rdce_asset_from_wire(
    path: Path,
    *,
    expected_wire_sha256: str,
    expected_checkpoint_sha256: str,
    expected_method_lock_sha256: str,
):
    source = _regular_file(path, "RDCE wire")
    raw = source.read_bytes()
    expected = _sha(expected_wire_sha256, "expected RDCE wire SHA256")
    if hashlib.sha256(raw).hexdigest() != expected or not raw.startswith(WIRE_MAGIC):
        raise D106Target25RunnerError("RDCE wire external SHA/magic drift")
    offset = len(WIRE_MAGIC)
    if len(raw) < offset + 4:
        raise D106Target25RunnerError("RDCE wire header is truncated")
    header_size = struct.unpack(">I", raw[offset : offset + 4])[0]
    offset += 4
    if header_size < 1 or offset + header_size > len(raw):
        raise D106Target25RunnerError("RDCE wire header length drift")
    header_raw = raw[offset : offset + header_size]
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Target25RunnerError("RDCE wire header is not UTF-8 JSON") from error
    if (
        not isinstance(header, Mapping)
        or _canonical_bytes(header) != header_raw
        or not isinstance(header.get("asset"), Mapping)
    ):
        raise D106Target25RunnerError("RDCE wire canonical header drift")
    asset_header = header["asset"]
    lineage_names = (
        "checkpoint_sha256",
        "runtime_sha256",
        "method_lock_sha256",
        "split_id",
        "tap_sha256",
        "construction_code_sha256",
        "content_root_sha256",
        "source_receipt_sha256",
        "tap_receipt_sha256",
        "tap_authority_sha256",
    )
    if any(name not in asset_header for name in lineage_names):
        raise D106Target25RunnerError("RDCE wire lineage field closure drift")
    lineage = D106RDCEAssetLineage(
        **{name: asset_header[name] for name in lineage_names}
    )
    if (
        lineage.checkpoint_sha256 != expected_checkpoint_sha256
        or lineage.method_lock_sha256 != expected_method_lock_sha256
    ):
        raise D106Target25RunnerError("RDCE lineage checkpoint/method-lock drift")
    return deserialize_d106_rdce_asset(
        raw,
        expected_wire_sha256=expected,
        expected_lineage=lineage,
    )


class _D106RealStateMaterializer:
    """Bounded D92 IQ/checkpoint adapter; it contains no D105 method state."""

    def __init__(
        self,
        *,
        plan: Mapping[str, Any],
        artifact_root: Path,
        checkpoint_path: Path,
        expected_checkpoint_sha256: str,
        rdce_wire_path: Path,
        expected_rdce_wire_sha256: str,
        rcmr_lock_path: Path,
        expected_rcmr_lock_sha256: str,
        device: str,
        feature_batch_size: int,
    ) -> None:
        if type(feature_batch_size) is not int or feature_batch_size < 1:
            raise D106Target25RunnerError("feature_batch_size must be positive")
        assets = plan["identity"]["assets"]
        supplied = {
            "checkpoint": (
                _regular_file(checkpoint_path, "checkpoint"),
                _sha(expected_checkpoint_sha256, "expected checkpoint SHA256"),
            ),
            "rdce_wire": (
                _regular_file(rdce_wire_path, "RDCE wire"),
                _sha(expected_rdce_wire_sha256, "expected RDCE wire SHA256"),
            ),
            "rcmr_lock": (
                _regular_file(rcmr_lock_path, "RCMR lock"),
                _sha(expected_rcmr_lock_sha256, "expected RCMR lock SHA256"),
            ),
        }
        for name, (path, digest) in supplied.items():
            expected_asset = assets.get(name)
            if (
                not isinstance(expected_asset, Mapping)
                or expected_asset.get("path") != str(path)
                or expected_asset.get("sha256") != digest
                or _sha256_file(path) != digest
            ):
                raise D106Target25RunnerError(f"prepared {name} binding drift")
        self.checkpoint_path, self.checkpoint_sha256 = supplied["checkpoint"]
        self.checkpoint_bytes = self.checkpoint_path.read_bytes()
        self.device = _device(device)
        self.feature_batch_size = feature_batch_size
        self.artifact_root = artifact_root
        self.artifact_root.mkdir()
        self.model = None
        self.model_input_len: int | None = None
        self.model_load_receipt_sha256: str | None = None
        self.package_cache: dict[
            tuple[tuple[str, str], ...],
            tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]],
        ] = {}
        rdce_lock = assets.get("rdce_lock")
        if not isinstance(rdce_lock, Mapping):
            raise D106Target25RunnerError("prepared RDCE lock binding missing")
        self.rdce_asset = _load_rdce_asset_from_wire(
            supplied["rdce_wire"][0],
            expected_wire_sha256=supplied["rdce_wire"][1],
            expected_checkpoint_sha256=self.checkpoint_sha256,
            expected_method_lock_sha256=_sha(
                rdce_lock.get("sha256"), "prepared RDCE method-lock SHA256"
            ),
        )
        self.rcmr_method_lock = load_d106_rcmr_2v_method_lock(
            supplied["rcmr_lock"][0],
            expected_sha256=supplied["rcmr_lock"][1],
        )

    def _package(self, value: Mapping[str, Any]):
        key = tuple(sorted((str(name), str(item)) for name, item in value.items()))
        cached = self.package_cache.get(key)
        if cached is None:
            cached = _load_raw_package(value)
            self.package_cache[key] = cached
        return cached

    def _model_for(self, input_len: int):
        if self.model is None:
            try:
                model, audit = _default_model_loader(
                    self.checkpoint_bytes, input_len, self.device
                )
            except Exception as error:
                raise D106Target25RunnerError("checkpoint safe loader failed") from error
            self.model = model
            self.model_input_len = input_len
            self.model_load_receipt_sha256 = canonical_sha256(dict(audit))
        elif self.model_input_len != input_len:
            raise D106Target25RunnerError("received-IQ input length drift")
        return self.model

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        index = request.get("artifact_index")
        if type(index) is not int or index < 0:
            raise D106Target25RunnerError("state artifact index missing")
        state_root = self.artifact_root / f"state-{index:03d}"
        if state_root.exists() or state_root.is_symlink():
            raise FileExistsError(f"immutable state artifact exists: {state_root}")
        state_root.mkdir()
        support_payloads, support_manifest, _support_audit = self._package(
            request["support_received_iq_ref"]
        )
        query_payloads, query_manifest, _query_audit = self._package(
            request["query_received_iq_ref"]
        )
        try:
            _validate_matched_packages(support_manifest, query_manifest)
        except Exception as error:
            raise D106Target25RunnerError("D92 support/query package pairing drift") from error
        scene = request["scenario"]
        if scene not in support_payloads or scene not in query_payloads:
            raise D106Target25RunnerError("D92 package scenario missing")
        registry = tuple(request["registered_classes"])
        package_registry = tuple(
            str(item.get("class_handle", ""))
            for item in support_manifest.get("registered_classes", [])
            if isinstance(item, Mapping)
        )
        if (
            package_registry != registry
            or support_manifest.get("receiver") != request["receiver"]
            or query_manifest.get("receiver") != request["receiver"]
            or support_manifest.get("seed") != request["seed"]
            or query_manifest.get("seed") != request["seed"]
            or support_manifest.get("k_shot") != request["k_shot"]
            or query_manifest.get("k_shot") != request["k_shot"]
        ):
            raise D106Target25RunnerError("D92 package row/registry binding drift")
        try:
            support_iq, support_labels, support_ids = _support_rows(
                support_payloads[scene],
                registered_classes=registry,
                active_k=request["k_shot"],
            )
            query_iq, query_ids = _query_rows(query_payloads[scene])
        except Exception as error:
            raise D106Target25RunnerError("D92 IQ/token materialization drift") from error
        if (
            canonical_sha256(sorted(support_ids))
            != request["support_physical_root_sha256"]
            or canonical_sha256(sorted(query_ids))
            != request["query_physical_root_sha256"]
        ):
            raise D106Target25RunnerError("materialized physical-ID root drift")
        if support_iq.shape[1:] != query_iq.shape[1:]:
            raise D106Target25RunnerError("support/query IQ shape drift")
        model = self._model_for(int(support_iq.shape[-1]))
        combined = np.ascontiguousarray(
            np.concatenate((support_iq, query_iq), axis=0), dtype=np.float32
        )
        try:
            signed, _zdom, tap_receipt = _tap_rows(
                model,
                combined,
                device=self.device,
                batch_size=self.feature_batch_size,
                feature_extractor=extract_d105_feature_tap,
            )
        except Exception as error:
            raise D106Target25RunnerError("same-model state feature forward failed") from error
        support_signed = np.ascontiguousarray(signed[: len(support_iq)], dtype=np.float32)
        query_signed = np.ascontiguousarray(signed[len(support_iq) :], dtype=np.float32)
        support_plus = np.maximum(support_signed, np.float32(0.0))
        query_plus = np.maximum(query_signed, np.float32(0.0))
        runtime_sha = support_manifest.get("feature_runtime_sha256")
        if (
            runtime_sha != query_manifest.get("feature_runtime_sha256")
            or runtime_sha != self.rdce_asset.runtime_sha256
        ):
            raise D106Target25RunnerError("feature runtime/RDCE lineage drift")
        received_pair_sha = canonical_sha256(
            {
                "support": request["support_received_iq_ref"][
                    "expected_seal_sha256"
                ],
                "query": request["query_received_iq_ref"][
                    "expected_seal_sha256"
                ],
            }
        )
        forward_receipt = canonical_sha256(
            {
                "schema": "cvs.phase2.d106.target25.same_model_forward.v1",
                "row_id": request["evaluation_row_id"],
                "received_iq_pair_sha256": received_pair_sha,
                "checkpoint_sha256": self.checkpoint_sha256,
                "model_load_receipt_sha256": self.model_load_receipt_sha256,
                "tap_receipt_sha256": tap_receipt,
                "support_rows": len(support_iq),
                "query_rows": len(query_iq),
                "query_fit_count": 0,
                "query_update_count": 0,
            }
        )
        feature_path = state_root / "paired_features.npz"
        feature_receipt_path = state_root / "paired_features.receipt.json"
        published = publish_d106_paired_features(
            feature_path,
            feature_receipt_path,
            received_iq_package_seal_sha256=received_pair_sha,
            checkpoint_sha256=self.checkpoint_sha256,
            runtime_sha256=runtime_sha,
            forward_receipt_sha256=forward_receipt,
            support_plus=support_plus,
            support_signed=support_signed,
            query_plus=query_plus,
            query_signed=query_signed,
            support_physical_ids=support_ids,
            query_physical_ids=query_ids,
        )
        features = load_d106_paired_features(
            feature_path,
            feature_receipt_path,
            expected_receipt_sha256=published["feature_receipt_sha256"],
        )
        plan_projection = {
            "schema": PLAN_STATE_SCHEMA,
            "row_id": request["evaluation_row_id"],
            "receiver": request["receiver"],
            "scene": scene,
            "active_k": request["k_shot"],
            "registered_classes": list(registry),
            "capsule_id": request["capsule_id"],
            "split_id": request["split_id"],
            "validator_receipt_sha256": request["validator_receipt_sha256"],
            "seed": request["seed"],
            "support_physical_root_sha256": request[
                "support_physical_root_sha256"
            ],
            "query_physical_root_sha256": request["query_physical_root_sha256"],
            "paired_feature_receipt_sha256": features.receipt_sha256,
        }
        plan_state_path = state_root / "plan_state.json"
        plan_state_sha = publish_d106_target25_plan_state(
            plan_state_path, projection=plan_projection
        )
        plan_state = load_d106_target25_plan_state(
            plan_state_path, expected_receipt_sha256=plan_state_sha
        )
        row_identity = canonical_sha256(
            {
                "schema": "cvs.phase2.d106.target25.qknn_row_identity.v1",
                "received_iq_pair_sha256": received_pair_sha,
                "paired_feature_receipt_sha256": features.receipt_sha256,
                "forward_receipt_sha256": forward_receipt,
                "numeric_lock": dict(PREDECESSOR_NUMERIC_LOCK),
            }
        )
        qknn_lock = Phase1ZIDStudentTLock(
            active_k=request["k_shot"],
            phase1_lodo_receipt_sha256=row_identity,
            quantization_margin_audit_sha256=features.receipt_sha256,
            **dict(PREDECESSOR_NUMERIC_LOCK),
        )
        bank = build_typed_zid_support_bank(
            features.support_plus,
            support_labels,
            registry,
            config=qknn_lock,
        )
        split_handle = TypedValidatedOnceP2SplitHandle(
            capsule_id=request["capsule_id"],
            split_id=request["split_id"],
            validator_receipt_sha256=request["validator_receipt_sha256"],
            support_physical_root_sha256=request["support_physical_root_sha256"],
            query_physical_root_sha256=request["query_physical_root_sha256"],
            support_query_disjoint=True,
        )
        label_array = np.asarray(support_labels, dtype=np.str_)
        id_array = np.asarray(support_ids, dtype=np.str_)
        support_rows = D106RDCESupportRows(
            support_z_id=np.ascontiguousarray(features.support_plus, dtype=np.float32),
            support_labels=label_array,
            support_physical_ids=id_array,
            qknn_bank=bank,
            split_handle=split_handle,
            row_id=request["evaluation_row_id"],
            seed=request["seed"],
        )
        authority_document = {
            "schema": ROW_AUTHORITY_SCHEMA,
            "capsule_id": request["capsule_id"],
            "split_id": request["split_id"],
            "validator_receipt_sha256": request["validator_receipt_sha256"],
            "row_id": request["evaluation_row_id"],
            "seed": request["seed"],
            "active_k": request["k_shot"],
            "registered_classes": list(registry),
            "support_z_id_receipt": _array_receipt(support_rows.support_z_id),
            "support_labels_receipt": _array_receipt(label_array),
            "support_physical_ids_receipt": _array_receipt(id_array),
            "ordered_support_physical_ids_sha256": canonical_sha256(
                list(support_ids)
            ),
            "qknn_bank_sha256": bank.bank_receipt_sha256,
            "support_physical_root_sha256": request[
                "support_physical_root_sha256"
            ],
            "query_physical_root_sha256": request["query_physical_root_sha256"],
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "support_query_disjoint": True,
        }
        authority_path = state_root / "rdce_row_authority.json"
        authority_raw = _canonical_bytes(authority_document)
        authority_path.write_bytes(authority_raw)
        authority_sha = hashlib.sha256(authority_raw).hexdigest()
        row_authority = load_d106_rdce_row_authority(
            authority_path, expected_authority_sha256=authority_sha
        )
        return {
            "plan_state": plan_state,
            "paired_features": features,
            "support_rows": support_rows,
            "rdce_asset": self.rdce_asset,
            "rdce_row_authority": row_authority,
            "rcmr_method_lock": self.rcmr_method_lock,
        }


def _reject_forbidden_evaluator_inputs(value: Mapping[str, Any]) -> None:
    for key in value:
        lowered = str(key).lower()
        if any(token in lowered for token in ("truth", "metric", "score", "selector")):
            raise D106Target25RunnerError(
                f"state-input factory exposed forbidden evaluator field: {key}"
            )


def _state_request(
    *,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    state: Mapping[str, Any],
    device: str,
    artifact_index: int,
) -> dict[str, Any]:
    state_name = str(state["state"])
    evaluation_row_id = (
        f"{row['job_id']}::{scenario['scenario']}::{state_name}"
    )
    return {
        "schema": "cvs.phase2.d106.target25.state_input_request.v1",
        "artifact_index": artifact_index,
        "evaluation_row_id": evaluation_row_id,
        "job_id": row["job_id"],
        "receiver": row["receiver"],
        "seed": row["seed"],
        "k_shot": row["k_shot"],
        "new_count": row["new_count"],
        "scenario": scenario["scenario"],
        "state": state_name,
        "registration_state": state["registration_state"],
        "registered_classes": state["registered_classes"],
        "old_classes": state["old_classes"],
        "new_classes": state["new_classes"],
        "capsule_id": state["capsule_id"],
        "split_id": state["split_id"],
        "validator_receipt_sha256": state["authority_receipt_sha256"],
        "support_physical_root_sha256": state["support_physical_root_sha256"],
        "query_physical_root_sha256": state["query_physical_root_sha256"],
        "support_received_iq_ref": state["support_received_iq_ref"],
        "query_received_iq_ref": state["query_received_iq_ref"],
        "assets": plan["identity"]["assets"],
        "device": device,
        "target_access": "sealed_received_iq_support_query_only",
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection": False,
    }


def _validate_prediction_row(
    row: Mapping[str, Any], *, request: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(row, Mapping):
        raise D106Target25RunnerError("state evaluator must return a mapping")
    prediction = dict(row)
    if (
        prediction.get("row_id") != request["evaluation_row_id"]
        or prediction.get("receiver") != request["receiver"]
        or prediction.get("scene") != request["scenario"]
        or prediction.get("K") != request["k_shot"]
        or prediction.get("registered_classes") != request["registered_classes"]
        or prediction.get("query_truth_access") is not False
        or prediction.get("query_role_access") is not False
        or prediction.get("query_selection") is not False
        or prediction.get("query_state_updates") != 0
    ):
        raise D106Target25RunnerError("state prediction request binding drift")
    query_ids = prediction.get("query_physical_ids")
    if not isinstance(query_ids, list) or not query_ids or len(set(query_ids)) != len(query_ids):
        raise D106Target25RunnerError("state prediction query order drift")
    if canonical_sha256(sorted(query_ids)) != request["query_physical_root_sha256"]:
        raise D106Target25RunnerError("state prediction query root drift")
    routed = route_d106_k_conditioned_prediction(
        active_k=request["k_shot"], row_prediction=prediction
    ).as_dict()
    routed["route_receipt_sha256"] = canonical_sha256(routed)
    return prediction, routed


def smoke_d106_target25_state(
    *,
    state_request: Mapping[str, Any],
    state_input_factory: StateInputFactory,
    state_evaluator: StateEvaluator = evaluate_d106_target25_state,
) -> dict[str, Any]:
    """Run one real-checkpoint-capable state path without any truth argument."""

    if not callable(state_input_factory) or not callable(state_evaluator):
        raise D106Target25RunnerError("state factory/evaluator must be callable")
    evaluator_inputs = state_input_factory(state_request)
    if not isinstance(evaluator_inputs, Mapping):
        raise D106Target25RunnerError("state-input factory must return a mapping")
    _reject_forbidden_evaluator_inputs(evaluator_inputs)
    prediction, routed = _validate_prediction_row(
        state_evaluator(**dict(evaluator_inputs)), request=state_request
    )
    return {"prediction_row": prediction, "routed_prediction": routed}


def smoke_d106_target25_prepared_state(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    rdce_wire_path: Path,
    expected_rdce_wire_sha256: str,
    rcmr_lock_path: Path,
    expected_rcmr_lock_sha256: str,
    output_dir: Path,
    row_index: int = 0,
    scenario_index: int = 0,
    state_index: int = 0,
    device: str = "cpu",
    feature_batch_size: int = 64,
) -> dict[str, Any]:
    """Run one prepared real-checkpoint state without exposing a truth input."""

    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    if (
        type(row_index) is not int
        or type(scenario_index) is not int
        or type(state_index) is not int
        or row_index not in range(OUTER_JOB_COUNT)
        or scenario_index not in range(len(LEO_SCENARIOS))
        or state_index not in range(len(STATES))
    ):
        raise D106Target25RunnerError("smoke row/scenario/state index drift")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable smoke output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D106Target25RunnerError("unsafe smoke output parent")
    destination.mkdir()
    materializer = _D106RealStateMaterializer(
        plan=plan,
        artifact_root=destination / "state_inputs",
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        rdce_wire_path=rdce_wire_path,
        expected_rdce_wire_sha256=expected_rdce_wire_sha256,
        rcmr_lock_path=rcmr_lock_path,
        expected_rcmr_lock_sha256=expected_rcmr_lock_sha256,
        device=device,
        feature_batch_size=feature_batch_size,
    )
    row = context["rows"][row_index]
    scenario = row["scenarios"][scenario_index]
    state = scenario["states"][state_index]
    request = _state_request(
        plan=plan,
        row=row,
        scenario=scenario,
        state=state,
        device=device,
        artifact_index=0,
    )
    result = smoke_d106_target25_state(
        state_request=request,
        state_input_factory=materializer,
    )
    receipt: dict[str, Any] = {
        "schema": "cvs.phase2.d106.target25.real_checkpoint_smoke.v1",
        "row_index": row_index,
        "scenario_index": scenario_index,
        "state_index": state_index,
        "prediction_receipt_sha256": result["prediction_row"][
            "prediction_receipt_sha256"
        ],
        "route_receipt_sha256": result["routed_prediction"][
            "route_receipt_sha256"
        ],
        "query_truth_access": False,
        "query_fit_count": 0,
        "query_update_count": 0,
        "query_selection_count": 0,
    }
    receipt["smoke_receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = destination / "smoke_receipt.json"
    receipt_file_sha = _write_json_new(receipt_path, receipt)
    return {
        **receipt,
        "smoke_receipt": str(receipt_path),
        "smoke_receipt_file_sha256": receipt_file_sha,
    }


def predict_d106_target25(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    rdce_wire_path: Path | None = None,
    expected_rdce_wire_sha256: str | None = None,
    rcmr_lock_path: Path | None = None,
    expected_rcmr_lock_sha256: str | None = None,
    feature_batch_size: int = 64,
    state_input_factory: StateInputFactory | None = None,
    state_evaluator: StateEvaluator = evaluate_d106_target25_state,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute and seal all 150 state evaluations and their four-arm surfaces."""

    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable prediction output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D106Target25RunnerError("unsafe prediction output parent")
    destination.mkdir()
    if state_input_factory is None:
        if any(
            value is None
            for value in (
                checkpoint_path,
                expected_checkpoint_sha256,
                rdce_wire_path,
                expected_rdce_wire_sha256,
                rcmr_lock_path,
                expected_rcmr_lock_sha256,
            )
        ):
            raise D106Target25RunnerError(
                "production prediction requires checkpoint/RDCE/RCMR paths and SHAs"
            )
        state_input_factory = _D106RealStateMaterializer(
            plan=plan,
            artifact_root=destination / "state_inputs",
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            rdce_wire_path=rdce_wire_path,
            expected_rdce_wire_sha256=expected_rdce_wire_sha256,
            rcmr_lock_path=rcmr_lock_path,
            expected_rcmr_lock_sha256=expected_rcmr_lock_sha256,
            device=device,
            feature_batch_size=feature_batch_size,
        )
    if not callable(state_input_factory) or not callable(state_evaluator):
        raise D106Target25RunnerError("state materializer/evaluator must be callable")
    prediction_rows: list[dict[str, Any]] = []
    artifact_index = 0
    for row in context["rows"]:
        scenes: list[dict[str, Any]] = []
        for scenario in row["scenarios"]:
            states: list[dict[str, Any]] = []
            for state in scenario["states"]:
                request = _state_request(
                    plan=plan,
                    row=row,
                    scenario=scenario,
                    state=state,
                    device=device,
                    artifact_index=artifact_index,
                )
                artifact_index += 1
                evaluated = smoke_d106_target25_state(
                    state_request=request,
                    state_input_factory=state_input_factory,
                    state_evaluator=state_evaluator,
                )
                states.append(
                    {
                        "state": state["state"],
                        "registration_state": state["registration_state"],
                        "state_input_receipt_sha256": state[
                            "state_input_receipt_sha256"
                        ],
                        "query_physical_root_sha256": state[
                            "query_physical_root_sha256"
                        ],
                        **evaluated,
                    }
                )
            scenes.append(
                {
                    "scenario_row_id": scenario["scenario_row_id"],
                    "scenario": scenario["scenario"],
                    "states": states,
                }
            )
        prediction_rows.append(
            {
                "job_id": row["job_id"],
                "receiver": row["receiver"],
                "seed": row["seed"],
                "k_shot": row["k_shot"],
                "new_count": row["new_count"],
                "scenarios": scenes,
            }
        )
    manifest: dict[str, Any] = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "matrix_receipt_sha256": plan["identity"]["matrix_receipt_sha256"],
        "plan_receipt_sha256": plan["plan_receipt_sha256"],
        "context_receipt_sha256": context["context_receipt_sha256"],
        "kcr_route_lock_sha256": plan["identity"]["assets"][
            "kcr_route_lock"
        ]["sha256"],
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
        "state_prediction_count": OUTER_JOB_COUNT * len(LEO_SCENARIOS) * len(STATES),
        "target_access": "sealed_received_iq_support_query_only",
        "clean_source_runtime_access": False,
        "query_fit_count": 0,
        "query_update_count": 0,
        "query_selection_count": 0,
        "rows": prediction_rows,
    }
    manifest["prediction_manifest_receipt_sha256"] = canonical_sha256(manifest)
    manifest_path = destination / "prediction_manifest.json"
    file_sha = _write_json_new(manifest_path, manifest)
    return {
        "prediction_manifest": str(manifest_path),
        "prediction_manifest_file_sha256": file_sha,
        "prediction_manifest_receipt_sha256": manifest[
            "prediction_manifest_receipt_sha256"
        ],
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
    }


def _validated_predictions(
    prediction: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    expected_context_receipt_sha256: str,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    if prediction.get("schema") != PREDICTION_MANIFEST_SCHEMA:
        raise D106Target25RunnerError("prediction manifest schema drift")
    if set(prediction) != _PREDICTION_MANIFEST_FIELDS:
        raise D106Target25RunnerError("prediction manifest field closure drift")
    _receipt(
        prediction,
        "prediction_manifest_receipt_sha256",
        "prediction manifest",
    )
    expected_counts = {
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
        "state_prediction_count": OUTER_JOB_COUNT * len(LEO_SCENARIOS) * len(STATES),
    }
    if any(prediction.get(name) != value for name, value in expected_counts.items()):
        raise D106Target25RunnerError("prediction 25/75/300/600 count closure drift")
    if (
        prediction.get("matrix_receipt_sha256")
        != plan["identity"]["matrix_receipt_sha256"]
        or prediction.get("plan_receipt_sha256") != plan["plan_receipt_sha256"]
        or prediction.get("context_receipt_sha256")
        != expected_context_receipt_sha256
        or prediction.get("kcr_route_lock_sha256")
        != plan["identity"]["assets"]["kcr_route_lock"]["sha256"]
        or prediction.get("clean_source_runtime_access") is not False
        or prediction.get("query_fit_count") != 0
        or prediction.get("query_update_count") != 0
        or prediction.get("query_selection_count") != 0
    ):
        raise D106Target25RunnerError("prediction access/identity closure drift")
    rows = _sequence(prediction.get("rows"), "prediction rows", OUTER_JOB_COUNT)
    plan_rows = plan["rows"]
    if [row.get("job_id") for row in rows if isinstance(row, Mapping)] != [
        row["job_id"] for row in plan_rows
    ]:
        raise D106Target25RunnerError("prediction row order/coverage drift")
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    arm_pairs = 0
    surfaces = 0
    for row, plan_row in zip(rows, plan_rows, strict=True):
        if not isinstance(row, Mapping) or any(
            row.get(name) != plan_row[name]
            for name in ("job_id", "receiver", "seed", "k_shot", "new_count")
        ):
            raise D106Target25RunnerError("prediction row binding drift")
        scenes = _sequence(row.get("scenarios"), "prediction scenarios", 3)
        for scene, plan_scene in zip(scenes, plan_row["scenarios"], strict=True):
            if (
                not isinstance(scene, Mapping)
                or scene.get("scenario") != plan_scene["scenario"]
                or scene.get("scenario_row_id") != plan_scene["scenario_row_id"]
            ):
                raise D106Target25RunnerError("prediction scenario binding drift")
            states = _sequence(scene.get("states"), "prediction states", 2)
            for state, plan_state in zip(states, plan_scene["states"], strict=True):
                if (
                    not isinstance(state, Mapping)
                    or state.get("state") != plan_state["state"]
                    or state.get("registration_state")
                    != plan_state["registration_state"]
                    or state.get("state_input_receipt_sha256")
                    != plan_state["state_input_receipt_sha256"]
                    or state.get("query_physical_root_sha256")
                    != plan_state["query_physical_root_sha256"]
                ):
                    raise D106Target25RunnerError("prediction state binding drift")
                request = {
                    "evaluation_row_id": (
                        f"{row['job_id']}::{scene['scenario']}::{state['state']}"
                    ),
                    "receiver": row["receiver"],
                    "scenario": scene["scenario"],
                    "k_shot": row["k_shot"],
                    "registered_classes": plan_state["registered_classes"],
                    "query_physical_root_sha256": plan_state[
                        "query_physical_root_sha256"
                    ],
                }
                prediction_row, routed = _validate_prediction_row(
                    state.get("prediction_row"), request=request
                )
                if state.get("routed_prediction") != routed:
                    raise D106Target25RunnerError("K-route artifact closure drift")
                result[(row["job_id"], scene["scenario"], state["state"])] = {
                    "prediction_row": prediction_row,
                    "routed_prediction": routed,
                    "plan_state": plan_state,
                }
                surfaces += len(ARMS)
            arm_pairs += len(ARMS)
    if (
        len(result) != OUTER_JOB_COUNT * len(LEO_SCENARIOS) * len(STATES)
        or arm_pairs != MATCHED_ARM_PAIR_COUNT
        or surfaces != STATE_SURFACE_COUNT
    ):
        raise D106Target25RunnerError("prediction structural surface closure drift")
    return result


def _truth_catalog(
    path: Path, *, expected_file_sha256: str, event_path: Path
) -> dict[str, Any]:
    # The event must already be durable before any path stat/read/hash occurs.
    if not event_path.is_file():
        raise D106Target25RunnerError("truth-open event was not published")
    return _read_json(
        path,
        name="independent truth catalog",
        expected_file_sha256=expected_file_sha256,
    )


def _accuracy(predictions: Sequence[str], truth: Sequence[str]) -> float:
    if len(predictions) != len(truth) or not truth:
        raise D106Target25RunnerError("metric row has empty/misaligned predictions")
    return 100.0 * sum(left == right for left, right in zip(predictions, truth, strict=True)) / len(truth)


def _metric_row(
    *,
    plan_row: Mapping[str, Any],
    method: str,
    predictions: Mapping[tuple[str, str, str], Mapping[str, Any]],
    truths: Mapping[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]],
) -> dict[str, Any]:
    before_pred: list[str] = []
    before_truth: list[str] = []
    after_old_pred: list[str] = []
    after_old_truth: list[str] = []
    after_new_pred: list[str] = []
    after_new_truth: list[str] = []
    old_per_class: dict[str, list[bool]] = {
        value: [] for value in plan_row["scenarios"][0]["states"][0]["old_classes"]
    }
    for scenario in LEO_SCENARIOS:
        for state_name in STATES:
            key = (plan_row["job_id"], scenario, state_name)
            prediction_row = predictions[key]["prediction_row"]
            routed = predictions[key]["routed_prediction"]
            query_ids, labels = truths[key]
            if tuple(prediction_row["query_physical_ids"]) != query_ids:
                raise D106Target25RunnerError("truth/prediction query order drift")
            values = (
                tuple(routed["predictions"])
                if method == "ROUTED"
                else tuple(prediction_row["arm_predictions"][method])
            )
            plan_state = predictions[key]["plan_state"]
            old = set(plan_state["old_classes"])
            new = set(plan_state["new_classes"])
            if any(label not in set(plan_state["registered_classes"]) for label in labels):
                raise D106Target25RunnerError("truth label falls outside state registry")
            if state_name == "before":
                before_pred.extend(values)
                before_truth.extend(labels)
            else:
                for predicted, label in zip(values, labels, strict=True):
                    if label in old:
                        after_old_pred.append(predicted)
                        after_old_truth.append(label)
                        old_per_class[label].append(predicted == label)
                    elif label in new:
                        after_new_pred.append(predicted)
                        after_new_truth.append(label)
                    else:
                        raise D106Target25RunnerError("after truth role partition drift")
    b_old = _accuracy(before_pred, before_truth)
    a_old = _accuracy(after_old_pred, after_old_truth)
    seen_new = _accuracy(after_new_pred, after_new_truth)
    floors = [100.0 * sum(values) / len(values) for values in old_per_class.values() if values]
    if len(floors) != len(old_per_class):
        raise D106Target25RunnerError("old-class floor coverage drift")
    harmonic = 0.0 if a_old + seen_new == 0.0 else 2.0 * a_old * seen_new / (a_old + seen_new)
    values = (b_old, a_old, seen_new, harmonic, b_old - a_old, min(floors))
    if any(not math.isfinite(value) for value in values):
        raise D106Target25RunnerError("non-finite score row")
    row = {
        "job_id": plan_row["job_id"],
        "receiver": plan_row["receiver"],
        "seed": plan_row["seed"],
        "k_shot": plan_row["k_shot"],
        "new_count": plan_row["new_count"],
        "method": method,
        "selected_arm": ROUTE_BY_K[plan_row["k_shot"]] if method == "ROUTED" else method,
        "B_old": b_old,
        "A_old": a_old,
        "A_old_floor": min(floors),
        "seen_new": seen_new,
        "H_old_new": harmonic,
        "forgetting": b_old - a_old,
    }
    row["metric_row_receipt_sha256"] = canonical_sha256(row)
    return row


def score_d106_target25(
    *,
    plan_manifest_path: Path,
    expected_plan_file_sha256: str,
    context_manifest_path: Path,
    expected_context_file_sha256: str,
    prediction_manifest_path: Path,
    expected_prediction_file_sha256: str,
    truth_catalog_path: Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate predictions, record truth opening, then score all five methods."""

    plan, context = _prepared_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    prediction = _read_json(
        prediction_manifest_path,
        name="D106 prediction manifest",
        expected_file_sha256=expected_prediction_file_sha256,
    )
    predictions = _validated_predictions(
        prediction,
        plan,
        expected_context_receipt_sha256=context["context_receipt_sha256"],
    )
    expected_truth_sha = _sha(
        expected_truth_catalog_file_sha256, "expected truth catalog file SHA256"
    )
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable score output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D106Target25RunnerError("unsafe score output parent")
    destination.mkdir()
    event: dict[str, Any] = {
        "schema": TRUTH_OPEN_EVENT_SCHEMA,
        "prediction_manifest_receipt_sha256": prediction[
            "prediction_manifest_receipt_sha256"
        ],
        "expected_truth_catalog_file_sha256": expected_truth_sha,
        "prediction_closure_verified": True,
        "outer_job_count": OUTER_JOB_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "matched_arm_pair_count": MATCHED_ARM_PAIR_COUNT,
        "state_surface_count": STATE_SURFACE_COUNT,
    }
    event["truth_open_event_receipt_sha256"] = canonical_sha256(event)
    event_path = destination / "truth_open_event.json"
    event_file_sha = _write_json_new(event_path, event)

    truth = _truth_catalog(
        truth_catalog_path,
        expected_file_sha256=expected_truth_sha,
        event_path=event_path,
    )
    if truth.get("schema") != TRUTH_CATALOG_SCHEMA:
        raise D106Target25RunnerError("truth catalog schema drift")
    truth_receipt = _receipt(truth, "truth_catalog_receipt_sha256", "truth catalog")
    if truth.get("matrix_receipt_sha256") != plan["identity"]["matrix_receipt_sha256"]:
        raise D106Target25RunnerError("truth catalog matrix binding drift")
    truth_rows = _sequence(truth.get("rows"), "truth rows", OUTER_JOB_COUNT)
    if [row.get("job_id") for row in truth_rows if isinstance(row, Mapping)] != [
        row["job_id"] for row in plan["rows"]
    ]:
        raise D106Target25RunnerError("truth row coverage/order drift")
    truths: dict[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for row in truth_rows:
        scenes = _sequence(row.get("scenarios"), "truth scenarios", 3)
        for scene, scenario in zip(scenes, LEO_SCENARIOS, strict=True):
            if not isinstance(scene, Mapping) or scene.get("scenario") != scenario:
                raise D106Target25RunnerError("truth scenario binding drift")
            states = _sequence(scene.get("states"), "truth states", 2)
            for state, state_name in zip(states, STATES, strict=True):
                if not isinstance(state, Mapping) or state.get("state") != state_name:
                    raise D106Target25RunnerError("truth state binding drift")
                query = state.get("query_physical_ids")
                labels = state.get("labels")
                if (
                    not isinstance(query, list)
                    or not isinstance(labels, list)
                    or not query
                    or len(query) != len(labels)
                    or len(set(query)) != len(query)
                    or any(type(value) is not str or not value for value in query + labels)
                ):
                    raise D106Target25RunnerError("truth query/label closure drift")
                key = (row["job_id"], scenario, state_name)
                truths[key] = (tuple(query), tuple(labels))
    if set(truths) != set(predictions):
        raise D106Target25RunnerError("truth/prediction surface coverage drift")
    metric_rows = [
        _metric_row(
            plan_row=plan_row,
            method=method,
            predictions=predictions,
            truths=truths,
        )
        for plan_row in plan["rows"]
        for method in METHODS
    ]
    score: dict[str, Any] = {
        "schema": SCORE_MANIFEST_SCHEMA,
        "matrix_receipt_sha256": plan["identity"]["matrix_receipt_sha256"],
        "prediction_manifest_receipt_sha256": prediction[
            "prediction_manifest_receipt_sha256"
        ],
        "truth_catalog_receipt_sha256": truth_receipt,
        "truth_open_event_receipt_sha256": event[
            "truth_open_event_receipt_sha256"
        ],
        "metric_row_count": OUTER_JOB_COUNT * len(METHODS),
        "methods": list(METHODS),
        "rows": metric_rows,
    }
    score["score_manifest_receipt_sha256"] = canonical_sha256(score)
    score_path = destination / "score_manifest.json"
    score_file_sha = _write_json_new(score_path, score)
    return {
        "truth_open_event": str(event_path),
        "truth_open_event_file_sha256": event_file_sha,
        "score_manifest": str(score_path),
        "score_manifest_file_sha256": score_file_sha,
        "score_manifest_receipt_sha256": score[
            "score_manifest_receipt_sha256"
        ],
        "metric_row_count": len(metric_rows),
    }


__all__ = [
    "D106Target25RunnerError",
    "METHODS",
    "PREDICTION_MANIFEST_SCHEMA",
    "SCORE_MANIFEST_SCHEMA",
    "TRUTH_CATALOG_SCHEMA",
    "TRUTH_OPEN_EVENT_SCHEMA",
    "predict_d106_target25",
    "prepare_d106_target25_run",
    "score_d106_target25",
    "smoke_d106_target25_prepared_state",
    "smoke_d106_target25_state",
]
