"""Support-only four-state runtime for NEXT-R2 CVFR-BSSDG proxy24."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_next_r2_bssdg as bssdg
from . import stage2_next_r2_cvfr as cvfr
from . import stage2_next_r2_matrix as matrix


SCHEMA = "cvs.stage2.next_r2.proxy24.runtime.v1"
STATE_RECEIPT_SCHEMA = "cvs.stage2.next_r2.proxy24.state_receipt.v1"
SEALED_MANIFEST_SCHEMA = "cvs.stage2.next_r2.proxy24.sealed_manifest.v1"


class NextR2RuntimeError(ValueError):
    """A state binding, protocol isolation, or prediction closure failed."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _id_root(values: Sequence[str]) -> str:
    return _sha_bytes(matrix.canonical_bytes({"ids": tuple(values)}))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise NextR2RuntimeError("runtime receipt contains an unsupported value")


def _features(value: object, *, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype("<f4")
        or array.ndim != 2
        or array.shape[1] != cvfr.Z_DIM
        or array.shape[0] < 1
        or (rows is not None and array.shape[0] != rows)
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        expected = "N" if rows is None else str(rows)
        raise NextR2RuntimeError(
            f"{name} must be finite C-contiguous float32 [{expected},{cvfr.Z_DIM}]"
        )
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _handles(
    values: Sequence[str], *, name: str, expected: int, unique: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise NextR2RuntimeError(f"{name} must be a row-aligned handle sequence")
    result = tuple(values)
    if (
        len(result) != expected
        or any(not isinstance(item, str) or not item for item in result)
        or (unique and len(set(result)) != expected)
    ):
        qualifier = "unique " if unique else ""
        raise NextR2RuntimeError(f"{name} must contain {expected} {qualifier}handles")
    return result


@dataclass(frozen=True, slots=True)
class NextR2StateInputs:
    """Prediction inputs for one state; query truth is deliberately absent."""

    outer_key_id: str
    state_id: str
    capsule_id: str
    split_id: str
    active_k: int
    registered_classes: tuple[str, ...]
    support_canonical: np.ndarray
    support_phase_plus: np.ndarray
    support_phase_minus: np.ndarray
    support_labels: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    query_canonical: np.ndarray
    query_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.state_id not in matrix.STATE_IDS:
            raise NextR2RuntimeError("state_id is outside the frozen four-state registry")
        if not isinstance(self.outer_key_id, str) or not self.outer_key_id:
            raise NextR2RuntimeError("outer_key_id must be nonempty")
        if not isinstance(self.capsule_id, str) or not self.capsule_id:
            raise NextR2RuntimeError("capsule_id must be nonempty")
        if not isinstance(self.split_id, str) or not self.split_id:
            raise NextR2RuntimeError("split_id must be nonempty")
        if self.active_k not in matrix.K_VALUES:
            raise NextR2RuntimeError("active_k must be 1 or 5")
        classes = _handles(
            self.registered_classes,
            name="registered_classes",
            expected=len(self.registered_classes),
            unique=True,
        )
        if len(classes) not in (matrix.CLASS_COUNT - 1, matrix.CLASS_COUNT):
            raise NextR2RuntimeError("registered class count must be five or six")
        expected_support = len(classes) * self.active_k
        expected_query = len(classes) * matrix.QUERY_PER_CLASS
        support_ids = _handles(
            self.support_physical_ids,
            name="support_physical_ids",
            expected=expected_support,
            unique=True,
        )
        query_ids = _handles(
            self.query_physical_ids,
            name="query_physical_ids",
            expected=expected_query,
            unique=True,
        )
        if set(support_ids) & set(query_ids):
            raise NextR2RuntimeError("support/query physical IDs must be disjoint")
        labels = _handles(
            self.support_labels,
            name="support_labels",
            expected=expected_support,
            unique=False,
        )
        if set(labels) != set(classes) or any(labels.count(item) != self.active_k for item in classes):
            raise NextR2RuntimeError("support labels must provide balanced K for every class")
        canonical = _features(
            self.support_canonical, name="support_canonical", rows=expected_support
        )
        plus = _features(
            self.support_phase_plus, name="support_phase_plus", rows=expected_support
        )
        minus = _features(
            self.support_phase_minus, name="support_phase_minus", rows=expected_support
        )
        query = _features(self.query_canonical, name="query_canonical", rows=expected_query)
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_physical_ids", query_ids)
        object.__setattr__(self, "support_canonical", canonical)
        object.__setattr__(self, "support_phase_plus", plus)
        object.__setattr__(self, "support_phase_minus", minus)
        object.__setattr__(self, "query_canonical", query)

    @property
    def base_input_sha256(self) -> str:
        return matrix.canonical_sha256(
            {
                "outer_key_id": self.outer_key_id,
                "capsule_id": self.capsule_id,
                "split_id": self.split_id,
                "registration": "REG1" if self.state_id in matrix.REG1_STATES else "REG0",
                "active_k": self.active_k,
                "registered_classes": self.registered_classes,
                "support_canonical_sha256": _array_sha(self.support_canonical),
                "support_plus_sha256": _array_sha(self.support_phase_plus),
                "support_minus_sha256": _array_sha(self.support_phase_minus),
                "support_label_root": _id_root(self.support_labels),
                "support_physical_id_root": _id_root(self.support_physical_ids),
                "query_canonical_sha256": _array_sha(self.query_canonical),
                "query_physical_id_root": _id_root(self.query_physical_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class NextR2StateResult:
    outer_key_id: str
    state_id: str
    registered_classes: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    scores: np.ndarray
    predictions: tuple[str, ...]
    cvfr_state: cvfr.CVFRState | None
    bssdg_state: bssdg.BSSDGState
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.state_id not in matrix.STATE_IDS:
            raise NextR2RuntimeError("result state_id drift")
        scores = np.asarray(self.scores)
        if (
            scores.dtype != np.dtype("<f4")
            or scores.shape != (len(self.query_physical_ids), len(self.registered_classes))
            or not scores.flags.c_contiguous
            or not np.isfinite(scores).all()
        ):
            raise NextR2RuntimeError("result scores shape/value drift")
        if len(self.predictions) != len(self.query_physical_ids):
            raise NextR2RuntimeError("result predictions/query row drift")
        if any(item not in self.registered_classes for item in self.predictions):
            raise NextR2RuntimeError("result prediction is outside the registered classes")
        if not isinstance(self.receipt, Mapping):
            raise NextR2RuntimeError("result receipt must be a mapping")
        if self.receipt.get("schema") != STATE_RECEIPT_SCHEMA:
            raise NextR2RuntimeError("result receipt schema drift")
        frozen_scores = np.array(scores, dtype=np.float32, copy=True, order="C")
        frozen_scores.setflags(write=False)
        object.__setattr__(self, "scores", frozen_scores)
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "receipt", _freeze(dict(self.receipt)))


def _state_input_pair_equal(left: NextR2StateInputs, right: NextR2StateInputs) -> bool:
    return (
        left.registered_classes == right.registered_classes
        and left.support_labels == right.support_labels
        and left.support_physical_ids == right.support_physical_ids
        and left.query_physical_ids == right.query_physical_ids
        and left.base_input_sha256 == right.base_input_sha256
    )


def _subset_rows_equal(
    *,
    subset_ids: tuple[str, ...],
    superset_ids: tuple[str, ...],
    subset_arrays: tuple[np.ndarray, ...],
    superset_arrays: tuple[np.ndarray, ...],
    subset_labels: tuple[str, ...] | None = None,
    superset_labels: tuple[str, ...] | None = None,
) -> bool:
    if len(subset_arrays) != len(superset_arrays):
        return False
    positions = {physical_id: index for index, physical_id in enumerate(superset_ids)}
    if len(positions) != len(superset_ids) or any(item not in positions for item in subset_ids):
        return False
    for subset_index, physical_id in enumerate(subset_ids):
        superset_index = positions[physical_id]
        if subset_labels is not None and (
            superset_labels is None
            or subset_labels[subset_index] != superset_labels[superset_index]
        ):
            return False
        if any(
            not np.array_equal(left[subset_index], right[superset_index])
            for left, right in zip(subset_arrays, superset_arrays, strict=True)
        ):
            return False
    return True


def validate_four_state_inputs(
    outer_key: matrix.NextR2OuterKey,
    inputs: Mapping[str, NextR2StateInputs],
) -> Mapping[str, NextR2StateInputs]:
    if tuple(inputs) != matrix.STATE_IDS or any(
        not isinstance(value, NextR2StateInputs) for value in inputs.values()
    ):
        raise NextR2RuntimeError("four-state inputs must follow the frozen state order")
    for state_id, value in inputs.items():
        expected_classes = matrix.registered_classes_for_state(outer_key, state_id)
        if (
            value.outer_key_id != outer_key.outer_key_id
            or value.state_id != state_id
            or value.active_k != outer_key.active_k
            or value.registered_classes != expected_classes
        ):
            raise NextR2RuntimeError("state inputs drifted from the outer-key plan")
    if (
        len({value.capsule_id for value in inputs.values()}) != 1
        or len({value.split_id for value in inputs.values()}) != 1
    ):
        raise NextR2RuntimeError(
            "all four states must share one capsule_id and split_id"
        )
    if not _state_input_pair_equal(inputs["DA0_REG0"], inputs["DA1_REG0"]):
        raise NextR2RuntimeError("DA0/DA1 REG0 must share exact base inputs")
    if not _state_input_pair_equal(inputs["DA0_REG1"], inputs["DA1_REG1"]):
        raise NextR2RuntimeError("DA0/DA1 REG1 must share exact base inputs")
    reg0 = inputs["DA0_REG0"]
    reg1 = inputs["DA0_REG1"]
    if (
        outer_key.held_class in reg0.registered_classes
        or outer_key.held_class not in reg1.registered_classes
        or set(reg0.support_physical_ids) - set(reg1.support_physical_ids)
        or set(reg0.query_physical_ids) - set(reg1.query_physical_ids)
    ):
        raise NextR2RuntimeError("REG1/REG0 registration isolation drift")
    if not _subset_rows_equal(
        subset_ids=reg0.support_physical_ids,
        superset_ids=reg1.support_physical_ids,
        subset_arrays=(
            reg0.support_canonical,
            reg0.support_phase_plus,
            reg0.support_phase_minus,
        ),
        superset_arrays=(
            reg1.support_canonical,
            reg1.support_phase_plus,
            reg1.support_phase_minus,
        ),
        subset_labels=reg0.support_labels,
        superset_labels=reg1.support_labels,
    ) or not _subset_rows_equal(
        subset_ids=reg0.query_physical_ids,
        superset_ids=reg1.query_physical_ids,
        subset_arrays=(reg0.query_canonical,),
        superset_arrays=(reg1.query_canonical,),
    ):
        raise NextR2RuntimeError("REG0 rows must be exact retained-only REG1 subsets")
    return MappingProxyType(dict(inputs))


def execute_next_r2_state(value: NextR2StateInputs) -> NextR2StateResult:
    """Fit current-state DA/head and predict canonical queries without truth."""

    if not isinstance(value, NextR2StateInputs):
        raise NextR2RuntimeError("execute_next_r2_state requires typed inputs")
    if value.state_id in matrix.DA1_STATES:
        cvfr_binding = cvfr.CVFRSupportBinding(
            capsule_id=value.capsule_id,
            split_id=value.split_id,
            outer_key=value.outer_key_id,
            state_id=value.state_id,
            k=value.active_k,
            registered_classes=value.registered_classes,
            support_physical_ids=value.support_physical_ids,
        )
        cvfr_fit_started = time.perf_counter_ns()
        cvfr_state = cvfr.fit_cvfr_support(
            value.support_canonical,
            value.support_phase_plus,
            value.support_phase_minus,
            value.support_labels,
            cvfr_binding,
        )
        cvfr_fit_latency_ns = time.perf_counter_ns() - cvfr_fit_started
        cvfr_support_transform_started = time.perf_counter_ns()
        support_z = cvfr.transform_cvfr(
            value.support_canonical,
            cvfr_state,
            expected_binding_digest=cvfr_binding.digest,
        )
        cvfr_support_transform_latency_ns = (
            time.perf_counter_ns() - cvfr_support_transform_started
        )
        cvfr_query_transform_started = time.perf_counter_ns()
        query_z = cvfr.transform_cvfr(
            value.query_canonical,
            cvfr_state,
            expected_binding_digest=cvfr_binding.digest,
        )
        cvfr_query_transform_latency_ns = (
            time.perf_counter_ns() - cvfr_query_transform_started
        )
        cvfr_wire = cvfr_state.to_wire()
        cvfr_status = cvfr_state.status
        cvfr_wire_sha: str | None = _sha_bytes(cvfr_wire)
        cvfr_receipt: Mapping[str, Any] | None = cvfr_state.receipt
        cvfr_transform_cost: Mapping[str, Any] | None = {
            "schema": "cvs.stage2.next_r2.cvfr_transform_integration_cost.v1",
            "transform_call_count": 2,
            "support_transform_rows": int(value.support_canonical.shape[0]),
            "query_transform_rows": int(value.query_canonical.shape[0]),
            "per_transform_fixed_helmert_h_a_multiplications": cvfr.Z_DIM * cvfr.SCALE_DIM,
            "per_transform_fixed_helmert_h_a_additions": cvfr.Z_DIM * (cvfr.SCALE_DIM - 1),
            "per_transform_fixed_exp_evaluations": cvfr.Z_DIM,
            "per_transform_fixed_rms_shift_multiplications": cvfr.Z_DIM,
            "per_row_scale_multiplications": cvfr.Z_DIM,
            "per_row_shift_additions": cvfr.Z_DIM,
            "core_320_field_is_scale_shift_per_row_only": True,
            "is_end_to_end_transform_total": False,
            "totalisation_cost_excluded_from_this_breakdown": True,
        }
    else:
        cvfr_state = None
        cvfr_fit_latency_ns = None
        cvfr_support_transform_latency_ns = None
        cvfr_query_transform_latency_ns = None
        support_z = np.ascontiguousarray(
            cvfr.totalize_rows(value.support_canonical), dtype=np.float32
        )
        query_z = np.ascontiguousarray(
            cvfr.totalize_rows(value.query_canonical), dtype=np.float32
        )
        cvfr_status = "DA0_IDENTITY_NO_CVFR_FIT"
        cvfr_wire_sha = None
        cvfr_receipt = None
        cvfr_transform_cost = None

    support_sha = _array_sha(support_z)
    state_input_digest = matrix.canonical_sha256(
        {
            "schema": "cvs.stage2.next_r2.bssdg_state_input_binding.v1",
            "support_z_sha256": support_sha,
            "capsule_id": value.capsule_id,
            "split_id": value.split_id,
            "outer_key_id": value.outer_key_id,
            "state_id": value.state_id,
            "support_physical_id_root": _id_root(value.support_physical_ids),
            "registered_classes": value.registered_classes,
        }
    )
    head_binding = bssdg.BSSDGBinding(
        state_name=value.state_id,
        registration_name="REG1" if value.state_id in matrix.REG1_STATES else "REG0",
        canonical_sha256=state_input_digest,
    )
    bssdg_fit_started = time.perf_counter_ns()
    head_state = bssdg.fit_bssdg(
        support_z,
        value.support_labels,
        value.registered_classes,
        k_shot=value.active_k,
        binding=head_binding,
    )
    bssdg_fit_latency_ns = time.perf_counter_ns() - bssdg_fit_started
    head_state = bssdg.roundtrip_bssdg_state(head_state)
    bssdg.verify_bssdg_binding(head_state, head_binding)
    bssdg_score_started = time.perf_counter_ns()
    scores = bssdg.score_bssdg(head_state, query_z)
    bssdg_score_latency_ns = time.perf_counter_ns() - bssdg_score_started
    maxima = np.max(scores, axis=1, keepdims=True)
    tie_rows = np.flatnonzero(np.sum(scores == maxima, axis=1) > 1)
    if len(tie_rows):
        raise bssdg.BSSDGExactTieError(
            f"exact top tie in query rows {tuple(int(item) for item in tie_rows)}"
        )
    winners = np.argmax(scores, axis=1)
    predictions = tuple(head_state.classes[int(index)] for index in winners)
    head_wire = bssdg.serialize_bssdg_state(head_state)

    receipt: dict[str, Any] = {
        "schema": STATE_RECEIPT_SCHEMA,
        "runtime_schema": SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "outer_key_id": value.outer_key_id,
        "state_id": value.state_id,
        "active_k": value.active_k,
        "registered_classes": value.registered_classes,
        "base_input_sha256": value.base_input_sha256,
        "support_physical_id_root": _id_root(value.support_physical_ids),
        "query_physical_id_root": _id_root(value.query_physical_ids),
        "support_query_disjoint": True,
        "query_truth_input_count": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "all_registered_classes_scored": True,
        "new_class_metric": "NA" if value.state_id not in matrix.REG1_STATES else "DEFINED_AFTER_SEPARATE_SCORING",
        "harmonic_metric": "NA" if value.state_id not in matrix.REG1_STATES else "DEFINED_AFTER_SEPARATE_SCORING",
        "cvfr_status": cvfr_status,
        "cvfr_wire_sha256": cvfr_wire_sha,
        "cvfr_receipt": cvfr_receipt,
        "cvfr_transform_cost": cvfr_transform_cost,
        "execution_latency_ns": {
            "measurement": "perf_counter_ns_single_execution_observation",
            "cvfr_fit": cvfr_fit_latency_ns,
            "cvfr_support_transform": cvfr_support_transform_latency_ns,
            "cvfr_query_transform": cvfr_query_transform_latency_ns,
            "bssdg_fit": bssdg_fit_latency_ns,
            "bssdg_score": bssdg_score_latency_ns,
            "excluded_from_cvfr_bssdg_deploy_state_digest": True,
        },
        "bssdg_state_sha256": head_state.state_sha256,
        "bssdg_wire_sha256": _sha_bytes(head_wire),
        "bssdg_fit_receipt": head_state.fit_receipt,
        "bssdg_resource_receipt": bssdg.bssdg_resource_receipt(head_state),
        "support_z_sha256": support_sha,
        "bssdg_state_input_digest": state_input_digest,
        "bssdg_state_input_digest_fields": (
            "support_z_sha256",
            "capsule_id",
            "split_id",
            "outer_key_id",
            "state_id",
            "support_physical_id_root",
            "registered_classes",
        ),
        "query_z_sha256": _array_sha(query_z),
        "scores_sha256": _array_sha(scores),
        "predictions_sha256": _sha_bytes(
            matrix.canonical_bytes({"predictions": predictions})
        ),
    }
    receipt["state_receipt_sha256"] = matrix.canonical_sha256(receipt)
    return NextR2StateResult(
        outer_key_id=value.outer_key_id,
        state_id=value.state_id,
        registered_classes=value.registered_classes,
        query_physical_ids=value.query_physical_ids,
        scores=np.ascontiguousarray(scores, dtype=np.float32),
        predictions=predictions,
        cvfr_state=cvfr_state,
        bssdg_state=head_state,
        receipt=receipt,
    )


def execute_next_r2_outer_key(
    outer_key: matrix.NextR2OuterKey,
    inputs: Mapping[str, NextR2StateInputs],
) -> tuple[NextR2StateResult, ...]:
    validated = validate_four_state_inputs(outer_key, inputs)
    results = tuple(execute_next_r2_state(validated[state]) for state in matrix.STATE_IDS)
    # A fresh BSSDG state is mandatory even when DA1 legally resolves to identity.
    if len({id(item.bssdg_state) for item in results}) != len(matrix.STATE_IDS):
        raise NextR2RuntimeError("four-state runtime reused a BSSDG state object")
    return results


def state_seal(result: NextR2StateResult) -> Mapping[str, Any]:
    if not isinstance(result, NextR2StateResult):
        raise NextR2RuntimeError("state_seal requires a typed result")
    payload: dict[str, Any] = {
        "schema": STATE_RECEIPT_SCHEMA,
        "outer_key_id": result.outer_key_id,
        "state_id": result.state_id,
        "registered_classes": result.registered_classes,
        "query_physical_id_root": _id_root(result.query_physical_ids),
        "scores_sha256": _array_sha(result.scores),
        "predictions_sha256": _sha_bytes(
            matrix.canonical_bytes({"predictions": result.predictions})
        ),
        "state_receipt_sha256": result.receipt["state_receipt_sha256"],
        "cvfr_status": result.receipt["cvfr_status"],
        "bssdg_state_sha256": result.bssdg_state.state_sha256,
    }
    payload["state_seal_sha256"] = matrix.canonical_sha256(payload)
    return MappingProxyType(payload)


def build_next_r2_sealed_manifest(
    plan: Mapping[str, Any],
    state_artifacts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    frozen_plan = matrix.validate_next_r2_proxy24_plan(plan)
    artifacts = tuple(state_artifacts)
    if len(artifacts) != matrix.STATE_PREDICTION_COUNT:
        raise NextR2RuntimeError("sealed manifest requires all 96 state artifacts")
    expected = tuple(
        (str(key["outer_key_id"]), state_id)
        for key in frozen_plan["keys"]
        for state_id in matrix.STATE_IDS
    )
    observed: list[tuple[str, str]] = []
    artifact_paths: set[str] = set()
    ready: list[dict[str, Any]] = []

    def valid_sha(value: str) -> bool:
        if len(value) != 64 or value.lower() != value:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    def valid_relative_path(value: object, *, suffix: str) -> bool:
        """Match the scorer's run-root-relative artifact path contract."""
        if not isinstance(value, str) or not value or "\\" in value:
            return False
        path = PurePosixPath(value)
        return (
            not path.is_absolute()
            and not any(part in {"", ".", ".."} for part in path.parts)
            and path.name.lower().endswith(suffix)
        )

    for item in artifacts:
        if not isinstance(item, Mapping):
            raise NextR2RuntimeError("state artifact must be a mapping")
        pair = (str(item.get("outer_key_id", "")), str(item.get("state_id", "")))
        observed.append(pair)
        json_path = item.get("json_path")
        npz_path = item.get("npz_path")
        json_sha = str(item.get("json_sha256", ""))
        npz_sha = str(item.get("npz_sha256", ""))
        seal_sha = str(item.get("state_seal_sha256", ""))
        if (
            not valid_relative_path(json_path, suffix=".json")
            or not valid_relative_path(npz_path, suffix=".npz")
            or str(json_path) in artifact_paths
            or str(npz_path) in artifact_paths
        ):
            raise NextR2RuntimeError("state artifact paths are invalid or repeated")
        if (
            not valid_sha(json_sha)
            or not valid_sha(npz_sha)
            or not valid_sha(seal_sha)
        ):
            raise NextR2RuntimeError("state artifact hashes are invalid")
        artifact_paths.update((str(json_path), str(npz_path)))
        ready.append(dict(item))
    if tuple(observed) != expected:
        raise NextR2RuntimeError("state artifact order/coverage drift")
    payload: dict[str, Any] = {
        "schema": SEALED_MANIFEST_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "matrix_sha256": frozen_plan["matrix_sha256"],
        "outer_key_count": matrix.OUTER_KEY_COUNT,
        "state_prediction_count": matrix.STATE_PREDICTION_COUNT,
        "all_states_sealed": True,
        "sealed_before_scoring": True,
        "truth_opened": False,
        "states": tuple(ready),
    }
    payload["sealed_manifest_sha256"] = matrix.canonical_sha256(payload)
    return MappingProxyType(payload)


__all__ = [
    "NextR2RuntimeError",
    "NextR2StateInputs",
    "NextR2StateResult",
    "SCHEMA",
    "SEALED_MANIFEST_SCHEMA",
    "STATE_RECEIPT_SCHEMA",
    "build_next_r2_sealed_manifest",
    "execute_next_r2_outer_key",
    "execute_next_r2_state",
    "state_seal",
    "validate_four_state_inputs",
]
