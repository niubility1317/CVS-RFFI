"""Support-only FA-RDCE3 -> direct signed-unit qKNN core for Target125.

The only Phase2 fit is one Fisher-anchored rank-three shift for K=5/K=10,
using REG0's six old-class support rows.  It is reused by object identity in
REG1.  K=1 never constructs an FA state: DA1 states are exact aliases of
their DA0 qKNN state, logits, predictions, and resource receipt.

R1 is produced by the frozen FA-RDCE3 signed-unit map and is passed directly
to the qKNN codec/scorer.  The qKNN implementation deliberately validates
unit rows without re-normalising them: neither ReLU nor a second L2 map can
silently change the prescribed R1 representation.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_next_r4_fa_rdce3 as r4
from . import stage2_next_r5_fa_target125_matrix as matrix
from .stage2_adv3b02_ts_drqknn_bcrr import phase1_qknn_lock
from .stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


CANDIDATE_ID = matrix.CANDIDATE_ID
PROTOCOL_SCHEMA = matrix.PROTOCOL_SCHEMA
Z_DIM = matrix.FEATURE_DIM
OLD_CLASS_COUNT = matrix.OLD_CLASS_COUNT
ALLOWED_K = (1, 5, 10)
FA_FIT_K = (5, 10)
FA_RANK = r4.RANK
INT8_MAX = 127

TARGET_ASSET_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.phase1_asset.v2"
TARGET_ASSET_WIRE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.phase1_asset_wire.v2"
RUNTIME_BINDING_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.runtime_binding.v2"
RUNTIME_STATE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.runtime_state.v1"
QKNN_STATE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.direct_qknn_state.v2"
FOUR_STATE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.four_state.v1"
SCORE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.four_state_score.v2"
RESOURCE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3.target125.resource.v1"

R0_REPRESENTATION = "same_iq_sealed_relu_signed_totalized_zid160"
R1_REPRESENTATION = "fa_rdce3_once_rdce_signed_unit_zid160"
FIT_MODE_FISHER_CLOSED_FORM = "FISHER_CLOSED_FORM"
FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE = "POSTERIOR_ZERO_FIXED_RDCE"
FIT_MODE_K1_STRICT_BYPASS = "FA_STRICT_BYPASS"
QKNN_TIE_POLICY = "highest_logit_then_min_registered_class_index"


class NextR5FATarget125CoreError(ValueError):
    """Raised when the frozen Target125 FA/qKNN contract drifts."""


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
        raise NextR5FATarget125CoreError("canonical core payload is invalid") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise NextR5FATarget125CoreError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR5FATarget125CoreError(f"{name} must be a lowercase SHA256") from error
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise NextR5FATarget125CoreError(f"{name} must be a non-empty exact string")
    return value


def _classes(value: Sequence[str], name: str, *, expected: int | None = None) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise NextR5FATarget125CoreError(f"{name} must be a string sequence")
    result = tuple(_text(item, name) for item in value)
    if not result or len(set(result)) != len(result):
        raise NextR5FATarget125CoreError(f"{name} must be a unique nonempty registry")
    if expected is not None and len(result) != expected:
        raise NextR5FATarget125CoreError(f"{name} cardinality drift")
    return result


def _class_indices(
    value: Sequence[int],
    name: str,
    *,
    expected: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise NextR5FATarget125CoreError(f"{name} must be an integer sequence")
    result = tuple(value)
    if (
        (expected is not None and len(result) != expected)
        or not result
        or any(type(item) is not int for item in result)
        or result != tuple(range(len(result)))
    ):
        raise NextR5FATarget125CoreError(f"{name} must be one continuous zero-based index sequence")
    return result


def _source_old_class_order_sha256(classes: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_bytes(list(_classes(classes, "source old classes", expected=OLD_CLASS_COUNT))))


def _physical_ids(value: Sequence[str], name: str, *, expected: int | None = None) -> tuple[str, ...]:
    result = _classes(value, name, expected=expected)
    return result


def _physical_root(value: Sequence[str]) -> str:
    values = _physical_ids(value, "physical IDs")
    return _sha256_bytes("\n".join(values).encode("utf-8"))


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _unit_rows(
    value: Any,
    name: str,
    *,
    nonnegative: bool | None,
    expected_rows: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR5FATarget125CoreError(f"{name} must be a numpy float32 array")
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or len(rows) < 1
        or (expected_rows is not None and len(rows) != expected_rows)
        or not np.isfinite(rows).all()
    ):
        raise NextR5FATarget125CoreError(f"{name} must be finite float32 [N,{Z_DIM}]")
    if nonnegative is True and np.any(rows < 0.0):
        raise NextR5FATarget125CoreError(f"{name} must be canonical nonnegative R0")
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR5FATarget125CoreError(f"{name} must already be unit length")
    return np.ascontiguousarray(rows, dtype=np.float64)


def _representation_rows(value: Any, representation: str, name: str) -> np.ndarray:
    if representation == R0_REPRESENTATION:
        # The trusted runtime preserves its sealed ReLU-unit row whenever it
        # is nonzero.  Only an exactly-zero sealed row may be totalized with
        # its byte-bound, same-IQ pre-ReLU signed direction.
        return _unit_rows(value, name, nonnegative=None)
    if representation == R1_REPRESENTATION:
        return _unit_rows(value, name, nonnegative=None)
    raise NextR5FATarget125CoreError("representation rule is outside the frozen contract")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _freeze_mapping(item)
            if isinstance(item, Mapping)
            else tuple(item)
            if isinstance(item, (list, tuple))
            else item
            for key, item in value.items()
        }
    )


def _target_asset_payload(asset: "Target125FARDCE3Asset") -> dict[str, Any]:
    wire = r4.serialize_fa_rdce3_phase1_asset(asset.fa_asset)
    return {
        "schema": TARGET_ASSET_WIRE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "asset_schema": TARGET_ASSET_SCHEMA,
        "old_classes": list(asset.fa_asset.old_classes),
        "source_class_indices": list(asset.source_class_indices),
        "source_old_class_order_sha256": asset.source_old_class_order_sha256,
        "aggregate_samples_per_class": list(asset.fa_asset.aggregate_samples_per_class),
        "phase1_source_only": True,
        "phase1_source_rows_retained": False,
        "phase1_per_row_features_retained": False,
        "target_support_rows_used": 0,
        "target_query_rows_used": 0,
        "query_rows_used_for_fit": 0,
        "underlying_fa_rdce3_wire_b64": base64.b64encode(wire).decode("ascii"),
    }


@dataclass(frozen=True, slots=True)
class Target125FARDCE3Asset:
    """A Target125-labelled wrapper around the reusable aggregate-only FA wire."""

    fa_asset: r4.FARDCE3Phase1Asset
    source_class_indices: tuple[int, ...]
    source_old_class_order_sha256: str
    schema: str = TARGET_ASSET_SCHEMA
    asset_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.fa_asset) is not r4.FARDCE3Phase1Asset or self.schema != TARGET_ASSET_SCHEMA:
            raise NextR5FATarget125CoreError("Target125 FA asset type/schema drift")
        if len(self.fa_asset.old_classes) != OLD_CLASS_COUNT:
            raise NextR5FATarget125CoreError("Target125 FA asset must carry exactly six old classes")
        if any(count < 2 for count in self.fa_asset.aggregate_samples_per_class):
            raise NextR5FATarget125CoreError("Target125 FA asset lost source aggregation proof")
        indices = _class_indices(
            self.source_class_indices,
            "source_class_indices",
            expected=OLD_CLASS_COUNT,
        )
        order_sha = _require_sha256(
            self.source_old_class_order_sha256,
            "source_old_class_order_sha256",
        )
        if order_sha != _source_old_class_order_sha256(self.fa_asset.old_classes):
            raise NextR5FATarget125CoreError("Target125 FA source old-class order root drift")
        object.__setattr__(self, "source_class_indices", indices)
        object.__setattr__(self, "source_old_class_order_sha256", order_sha)
        object.__setattr__(
            self,
            "asset_sha256",
            _sha256_bytes(_canonical_bytes(_target_asset_payload(self))),
        )

    @property
    def old_classes(self) -> tuple[str, ...]:
        return self.fa_asset.old_classes

    @property
    def checkpoint_sha256(self) -> str:
        return self.fa_asset.checkpoint_sha256

    @property
    def phase1_bundle_sha256(self) -> str:
        return self.fa_asset.phase1_bundle_sha256

    @property
    def phase1_aggregate_receipt_sha256(self) -> str:
        return self.fa_asset.phase1_aggregate_receipt_sha256

    @property
    def method_lock_sha256(self) -> str:
        return self.fa_asset.method_lock_sha256


def build_target_fa_asset(
    *,
    old_classes: Sequence[str],
    aggregate_samples_per_class: Sequence[int],
    class_centers_3d: np.ndarray,
    fisher_precision_3d: np.ndarray,
    residual_variance_3d: np.ndarray,
    fisher_radius: np.ndarray,
    rdce_kappa_3d: np.ndarray,
    basis_3x160: np.ndarray,
    checkpoint_sha256: str,
    phase1_bundle_sha256: str,
    phase1_aggregate_receipt_sha256: str,
    method_lock_sha256: str,
    source_class_indices: Sequence[int] | None = None,
    source_old_class_order_sha256: str | None = None,
) -> Target125FARDCE3Asset:
    """Build a six-old-class Target125 asset from source-only aggregates.

    There is deliberately no Target support/query, raw IQ, source row, source
    feature, physical-ID, or receiver argument in this construction boundary.
    """

    classes = _classes(old_classes, "old_classes", expected=OLD_CLASS_COUNT)
    indices = _class_indices(
        tuple(range(OLD_CLASS_COUNT)) if source_class_indices is None else source_class_indices,
        "source_class_indices",
        expected=OLD_CLASS_COUNT,
    )
    derived_order_sha = _source_old_class_order_sha256(classes)
    if source_old_class_order_sha256 is not None and _require_sha256(
        source_old_class_order_sha256,
        "source_old_class_order_sha256",
    ) != derived_order_sha:
        raise NextR5FATarget125CoreError("source old-class order root drift")
    counts = tuple(aggregate_samples_per_class)
    if len(counts) != OLD_CLASS_COUNT or any(type(item) is not int or item < 2 for item in counts):
        raise NextR5FATarget125CoreError("aggregate_samples_per_class must prove six-class aggregation")
    try:
        underlying = r4.build_fa_rdce3_phase1_asset(
            old_classes=classes,
            aggregate_samples_per_class=counts,
            class_centers_3d=class_centers_3d,
            fisher_precision_3d=fisher_precision_3d,
            residual_variance_3d=residual_variance_3d,
            fisher_radius=fisher_radius,
            rdce_kappa_3d=rdce_kappa_3d,
            basis_3x160=basis_3x160,
            checkpoint_sha256=_require_sha256(checkpoint_sha256, "checkpoint_sha256"),
            phase1_bundle_sha256=_require_sha256(phase1_bundle_sha256, "phase1_bundle_sha256"),
            phase1_aggregate_receipt_sha256=_require_sha256(
                phase1_aggregate_receipt_sha256,
                "phase1_aggregate_receipt_sha256",
            ),
            method_lock_sha256=_require_sha256(method_lock_sha256, "method_lock_sha256"),
        )
    except r4.NextR4FARDCE3Error as error:
        raise NextR5FATarget125CoreError("source-only FA aggregate construction failed") from error
    return Target125FARDCE3Asset(
        underlying,
        source_class_indices=indices,
        source_old_class_order_sha256=derived_order_sha,
    )


def validate_target_fa_asset(asset: Target125FARDCE3Asset) -> None:
    if type(asset) is not Target125FARDCE3Asset:
        raise NextR5FATarget125CoreError("Target125 FA asset must use the exact wrapper type")
    if asset.asset_sha256 != _sha256_bytes(_canonical_bytes(_target_asset_payload(asset))):
        raise NextR5FATarget125CoreError("Target125 FA asset receipt drift")


def serialize_target_fa_asset(asset: Target125FARDCE3Asset) -> bytes:
    validate_target_fa_asset(asset)
    payload = _target_asset_payload(asset)
    payload["asset_sha256"] = asset.asset_sha256
    return _canonical_bytes(payload)


def deserialize_target_fa_asset(value: bytes) -> Target125FARDCE3Asset:
    if not isinstance(value, bytes):
        raise NextR5FATarget125CoreError("Target125 FA asset wire must be bytes")
    try:
        payload = json.loads(value.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR5FATarget125CoreError("Target125 FA asset wire is invalid JSON") from error
    if _canonical_bytes(payload) != value:
        raise NextR5FATarget125CoreError("Target125 FA asset wire is not canonical")
    required = {
        "schema",
        "candidate_id",
        "asset_schema",
        "old_classes",
        "source_class_indices",
        "source_old_class_order_sha256",
        "aggregate_samples_per_class",
        "phase1_source_only",
        "phase1_source_rows_retained",
        "phase1_per_row_features_retained",
        "target_support_rows_used",
        "target_query_rows_used",
        "query_rows_used_for_fit",
        "underlying_fa_rdce3_wire_b64",
        "asset_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise NextR5FATarget125CoreError("Target125 FA asset wire field closure drift")
    if (
        payload["schema"] != TARGET_ASSET_WIRE_SCHEMA
        or payload["candidate_id"] != CANDIDATE_ID
        or payload["asset_schema"] != TARGET_ASSET_SCHEMA
        or payload["phase1_source_only"] is not True
        or payload["phase1_source_rows_retained"] is not False
        or payload["phase1_per_row_features_retained"] is not False
        or payload["target_support_rows_used"] != 0
        or payload["target_query_rows_used"] != 0
        or payload["query_rows_used_for_fit"] != 0
    ):
        raise NextR5FATarget125CoreError("Target125 FA asset source-only contract drift")
    try:
        inner_wire = base64.b64decode(
            str(payload["underlying_fa_rdce3_wire_b64"]).encode("ascii"),
            validate=True,
        )
        underlying = r4.deserialize_fa_rdce3_phase1_asset(inner_wire)
    except (ValueError, UnicodeEncodeError, r4.NextR4FARDCE3Error) as error:
        raise NextR5FATarget125CoreError("Target125 FA asset underlying wire is invalid") from error
    asset = Target125FARDCE3Asset(
        underlying,
        source_class_indices=_class_indices(
            payload["source_class_indices"],
            "source_class_indices",
            expected=OLD_CLASS_COUNT,
        ),
        source_old_class_order_sha256=_require_sha256(
            payload["source_old_class_order_sha256"],
            "source_old_class_order_sha256",
        ),
    )
    if (
        tuple(payload["old_classes"]) != asset.old_classes
        or tuple(payload["source_class_indices"]) != asset.source_class_indices
        or payload["source_old_class_order_sha256"] != asset.source_old_class_order_sha256
        or tuple(payload["aggregate_samples_per_class"]) != asset.fa_asset.aggregate_samples_per_class
        or _require_sha256(payload["asset_sha256"], "asset_sha256") != asset.asset_sha256
        or serialize_target_fa_asset(asset) != value
    ):
        raise NextR5FATarget125CoreError("Target125 FA asset roundtrip drift")
    return asset


@dataclass(frozen=True, slots=True)
class Target125FARuntimeBinding:
    """Opaque row/physical-ID binding; it carries no query labels or features."""

    checkpoint_sha256: str
    capsule_id: str
    split_id: str
    outer_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    source_pool_k: int
    scene: str
    registration_phase: str
    registered_classes: tuple[str, ...]
    registered_class_indices: tuple[int, ...]
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    protocol_schema: str = PROTOCOL_SCHEMA
    phase2_data_status: str = "VALIDATED_ONCE"
    schema: str = RUNTIME_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != RUNTIME_BINDING_SCHEMA
            or self.protocol_schema != PROTOCOL_SCHEMA
            or self.phase2_data_status != "VALIDATED_ONCE"
            or type(self.seed) is not int
            or self.seed not in matrix.SEEDS
            or self.k_shot not in ALLOWED_K
            or type(self.new_count) is not int
            or self.registration_phase not in ("REG0", "REG1")
        ):
            raise NextR5FATarget125CoreError("runtime binding lifecycle drift")
        for name in ("checkpoint_sha256", "capsule_id", "split_id"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        expected_outer = matrix.make_outer_id(
            self.receiver,
            self.seed,
            self.k_shot,
            self.new_count,
        )
        if self.outer_id != expected_outer or self.source_pool_k != matrix.source_pool_k_for(
            self.k_shot,
            self.new_count,
        ):
            raise NextR5FATarget125CoreError("runtime binding outer/source-pool drift")
        if self.scene not in matrix.SCENES:
            raise NextR5FATarget125CoreError("runtime binding scene drift")
        registry = _classes(self.registered_classes, "registered_classes")
        if (
            (self.registration_phase == "REG0" and len(registry) != OLD_CLASS_COUNT)
            or (self.registration_phase == "REG1" and len(registry) <= OLD_CLASS_COUNT)
        ):
            raise NextR5FATarget125CoreError("runtime binding registration registry drift")
        indices = _class_indices(
            self.registered_class_indices,
            "registered_class_indices",
            expected=len(registry),
        )
        if (
            (self.registration_phase == "REG0" and indices != tuple(range(OLD_CLASS_COUNT)))
            or (
                self.registration_phase == "REG1"
                and indices[:OLD_CLASS_COUNT] != tuple(range(OLD_CLASS_COUNT))
            )
        ):
            raise NextR5FATarget125CoreError("runtime binding registered class-index drift")
        support = _physical_ids(
            self.support_physical_ids,
            "support_physical_ids",
            expected=len(registry) * self.k_shot,
        )
        query = _physical_ids(self.query_physical_ids, "query_physical_ids")
        if set(support).intersection(query):
            raise NextR5FATarget125CoreError("support/query physical-ID overlap is forbidden")
        object.__setattr__(self, "registered_classes", registry)
        object.__setattr__(self, "registered_class_indices", indices)
        object.__setattr__(self, "support_physical_ids", support)
        object.__setattr__(self, "query_physical_ids", query)

    @property
    def support_physical_root_sha256(self) -> str:
        return _physical_root(self.support_physical_ids)

    @property
    def query_physical_root_sha256(self) -> str:
        return _physical_root(self.query_physical_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "outer_id": self.outer_id,
            "receiver": self.receiver,
            "seed": self.seed,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "source_pool_k": self.source_pool_k,
            "scene": self.scene,
            "registration_phase": self.registration_phase,
            "registered_classes": list(self.registered_classes),
            "registered_class_indices": list(self.registered_class_indices),
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "query_physical_root_sha256": self.query_physical_root_sha256,
            "protocol_schema": self.protocol_schema,
            "phase2_data_status": self.phase2_data_status,
        }

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.as_dict()))


def _runtime_payload(
    *,
    asset: Target125FARDCE3Asset,
    binding: Target125FARuntimeBinding,
    a_fp16: np.ndarray,
    fit_mode: str,
    support_content_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_STATE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "asset_sha256": asset.asset_sha256,
        "runtime_binding": binding.as_dict(),
        "runtime_binding_sha256": binding.binding_sha256,
        "a_fp16": _array_receipt(a_fp16),
        "fit_mode": fit_mode,
        "support_content_sha256": support_content_sha256,
        "fit_input": "REG0_old_class_support_only",
        "phase1_source_rows_used": 0,
        "phase1_loo_folds_used": 0,
        "new_class_support_rows_used_for_da": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_gradient_calls": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_batch_dependency": False,
        "target_optimizer_steps": 0,
        "dynamic_numeric_bytes": FA_RANK * np.dtype("<f2").itemsize,
    }


@dataclass(frozen=True, slots=True)
class Target125FARuntimeState:
    """One immutable K5/K10 FA state fitted from REG0 old support exactly once."""

    asset: Target125FARDCE3Asset
    binding: Target125FARuntimeBinding
    a_fp16: np.ndarray
    fit_mode: str
    support_content_sha256: str
    fa_state_receipt_sha256: str
    schema: str = RUNTIME_STATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.asset) is not Target125FARDCE3Asset
            or type(self.binding) is not Target125FARuntimeBinding
            or self.schema != RUNTIME_STATE_SCHEMA
            or self.binding.registration_phase != "REG0"
            or self.binding.k_shot not in FA_FIT_K
            or self.binding.registered_class_indices != self.asset.source_class_indices
            or self.binding.checkpoint_sha256 != self.asset.checkpoint_sha256
            or self.fit_mode not in (
                FIT_MODE_FISHER_CLOSED_FORM,
                FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE,
            )
        ):
            raise NextR5FATarget125CoreError("FA runtime state contract drift")
        value = np.asarray(self.a_fp16)
        if value.dtype != np.dtype("<f2") or value.shape != (FA_RANK,) or not np.isfinite(value).all():
            raise NextR5FATarget125CoreError("FA runtime state shift dtype/shape drift")
        a = _readonly(value, np.dtype("<f2"))
        raw = a.astype(np.float64)
        if self.fit_mode == FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE and np.any(raw != 0.0):
            raise NextR5FATarget125CoreError("zero-posterior mode must retain exact zero shift")
        if self.fit_mode == FIT_MODE_FISHER_CLOSED_FORM and not np.any(raw != 0.0):
            raise NextR5FATarget125CoreError("closed-form mode cannot hide zero shift")
        fisher = r4.decode_fa_rdce3_fisher_precision(self.asset.fa_asset).astype(np.float64)
        radius = r4.decode_fa_rdce3_radius(self.asset.fa_asset)
        norm = math.sqrt(float(np.sum(fisher * np.square(raw))))
        if not math.isfinite(norm) or norm > radius + 1.0e-7 * max(1.0, radius):
            raise NextR5FATarget125CoreError("FA FP16 shift escaped frozen Fisher radius")
        support_sha = _require_sha256(self.support_content_sha256, "support_content_sha256")
        receipt = _require_sha256(self.fa_state_receipt_sha256, "fa_state_receipt_sha256")
        if receipt != _sha256_bytes(
            _canonical_bytes(
                _runtime_payload(
                    asset=self.asset,
                    binding=self.binding,
                    a_fp16=a,
                    fit_mode=self.fit_mode,
                    support_content_sha256=support_sha,
                )
            )
        ):
            raise NextR5FATarget125CoreError("FA runtime receipt drift")
        object.__setattr__(self, "a_fp16", a)
        object.__setattr__(self, "support_content_sha256", support_sha)
        object.__setattr__(self, "fa_state_receipt_sha256", receipt)

    @property
    def dynamic_numeric_bytes(self) -> int:
        return int(self.a_fp16.nbytes)

    @property
    def resource_receipt(self) -> Mapping[str, Any]:
        return _freeze_mapping(
            {
                "schema": RESOURCE_SCHEMA,
                "candidate_id": CANDIDATE_ID,
                "kind": "fa_rdce3_runtime",
                "runtime_state_sha256": self.fa_state_receipt_sha256,
                "active_k": self.binding.k_shot,
                "dynamic_numeric_bytes": self.dynamic_numeric_bytes,
                "fit_mac": OLD_CLASS_COUNT * self.binding.k_shot * FA_RANK * Z_DIM,
                "query_mac_per_row": 2 * FA_RANK * Z_DIM,
                "query_rows_used_for_fit": 0,
                "new_class_support_rows_used_for_da": 0,
            }
        )


def _quantize_shift_toward_zero(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != (FA_RANK,) or not np.isfinite(raw).all():
        raise NextR5FATarget125CoreError("FA posterior shift is non-finite")
    result = np.asarray(raw, dtype=np.dtype("<f2"))
    if not np.isfinite(result).all():
        raise NextR5FATarget125CoreError("FA posterior shift cannot fit FP16")
    for index, item in enumerate(result):
        if abs(float(item)) > abs(float(raw[index])):
            result[index] = np.nextafter(item, np.float16(0.0), dtype=np.float16)
    if np.any(raw != 0.0) and not np.any(result != 0.0):
        raise NextR5FATarget125CoreError("nonzero FA posterior underflowed its fixed state")
    return _readonly(result, np.dtype("<f2"))


def _support_content_sha(
    asset: Target125FARDCE3Asset,
    binding: Target125FARuntimeBinding,
    support: Mapping[int, np.ndarray],
    labels: Sequence[str],
    physical_ids: Sequence[str],
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": "cvs.phase2.next_r5.fa_rdce3.target125.reg0_support_content.v1",
                "asset_sha256": asset.asset_sha256,
                "source_class_indices": list(asset.source_class_indices),
                "source_old_class_order_sha256": asset.source_old_class_order_sha256,
                "row_local_old_class_handles": list(binding.registered_classes),
                "arrays_by_source_class_index": [
                    _array_receipt(support[class_index])
                    for class_index in asset.source_class_indices
                ],
                "labels": list(labels),
                "support_physical_root_sha256": _physical_root(physical_ids),
            }
        )
    )


def fit_fa_rdce3(
    asset: Target125FARDCE3Asset,
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    k_shot: int,
    *,
    binding: Target125FARuntimeBinding,
) -> Target125FARuntimeState:
    """Fit the sole K5/K10 closed-form FA state from REG0 old support.

    The function has no query, new-support, truth, role, score, or optimiser
    argument.  K1 is rejected rather than converted into a fake zero shift.
    """

    validate_target_fa_asset(asset)
    if type(binding) is not Target125FARuntimeBinding:
        raise NextR5FATarget125CoreError("FA fit requires an exact runtime binding")
    if (
        type(k_shot) is not int
        or k_shot not in FA_FIT_K
        or binding.k_shot != k_shot
        or binding.registration_phase != "REG0"
        or binding.registered_class_indices != asset.source_class_indices
        or tuple(old_registered_classes) != binding.registered_classes
    ):
        raise NextR5FATarget125CoreError("FA fit K/REG0 registry drift")
    labels = tuple(_text(item, "old_support_labels") for item in old_support_labels)
    expected_rows = OLD_CLASS_COUNT * k_shot
    rows = _unit_rows(
        old_support_features,
        "old_support_features",
        nonnegative=None,
        expected_rows=expected_rows,
    )
    if len(labels) != expected_rows or len(binding.support_physical_ids) != expected_rows:
        raise NextR5FATarget125CoreError("FA fit old-support row/physical binding drift")
    support: dict[int, np.ndarray] = {}
    for class_index, class_handle in zip(
        binding.registered_class_indices,
        binding.registered_classes,
        strict=True,
    ):
        positions = [index for index, label in enumerate(labels) if label == class_handle]
        if len(positions) != k_shot:
            raise NextR5FATarget125CoreError("FA fit requires balanced complete old support")
        support[class_index] = np.ascontiguousarray(rows[positions], dtype=np.float64)
    if any(label not in binding.registered_classes for label in labels):
        raise NextR5FATarget125CoreError("FA fit support label is outside old registry")
    basis = r4.decode_fa_rdce3_basis(asset.fa_asset).astype(np.float64)
    centers = r4.decode_fa_rdce3_centers(asset.fa_asset).astype(np.float64)
    fisher = r4.decode_fa_rdce3_fisher_precision(asset.fa_asset).astype(np.float64)
    variance = r4.decode_fa_rdce3_residual_variance(asset.fa_asset).astype(np.float64)
    residual_sum = np.zeros(FA_RANK, dtype=np.float64)
    for class_index in asset.source_class_indices:
        residual_sum += np.sum(support[class_index] @ basis.T - centers[class_index][None, :], axis=0)
    precision = fisher + float(OLD_CLASS_COUNT * k_shot) / variance
    raw_a = (residual_sum / variance) / precision
    if not np.isfinite(raw_a).all() or np.any(precision <= 0.0):
        raise NextR5FATarget125CoreError("FA Fisher posterior is undefined")
    fisher_norm_sq = float(np.sum(fisher * np.square(raw_a)))
    if not math.isfinite(fisher_norm_sq) or fisher_norm_sq < 0.0:
        raise NextR5FATarget125CoreError("FA Fisher direction is undefined")
    if fisher_norm_sq == 0.0:
        a_fp16 = _readonly(np.zeros(FA_RANK, dtype=np.dtype("<f2")), np.dtype("<f2"))
        fit_mode = FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE
    else:
        radius = r4.decode_fa_rdce3_radius(asset.fa_asset)
        norm = math.sqrt(fisher_norm_sq)
        if norm > radius:
            raw_a = raw_a * (radius / norm)
        a_fp16 = _quantize_shift_toward_zero(raw_a)
        fit_mode = FIT_MODE_FISHER_CLOSED_FORM
    support_sha = _support_content_sha(asset, binding, support, labels, binding.support_physical_ids)
    payload = _runtime_payload(
        asset=asset,
        binding=binding,
        a_fp16=a_fp16,
        fit_mode=fit_mode,
        support_content_sha256=support_sha,
    )
    return Target125FARuntimeState(
        asset=asset,
        binding=binding,
        a_fp16=a_fp16,
        fit_mode=fit_mode,
        support_content_sha256=support_sha,
        fa_state_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
    )


def reuse_fa_rdce3_state_for_reg1(
    state: Target125FARuntimeState,
    *,
    reg1_binding: Target125FARuntimeBinding,
) -> Target125FARuntimeState:
    """Return the exact REG0 FA state object for REG1; refitting is impossible."""

    if type(state) is not Target125FARuntimeState or type(reg1_binding) is not Target125FARuntimeBinding:
        raise NextR5FATarget125CoreError("REG1 FA reuse requires exact typed state/binding")
    old = state.binding
    same_outer = (
        old.checkpoint_sha256 == reg1_binding.checkpoint_sha256
        and old.capsule_id == reg1_binding.capsule_id
        and old.split_id == reg1_binding.split_id
        and old.outer_id == reg1_binding.outer_id
        and old.receiver == reg1_binding.receiver
        and old.seed == reg1_binding.seed
        and old.k_shot == reg1_binding.k_shot
        and old.new_count == reg1_binding.new_count
        and old.source_pool_k == reg1_binding.source_pool_k
        and old.scene == reg1_binding.scene
    )
    if (
        not same_outer
        or reg1_binding.registration_phase != "REG1"
        or reg1_binding.registered_class_indices[:OLD_CLASS_COUNT]
        != state.asset.source_class_indices
        or reg1_binding.registered_classes[:OLD_CLASS_COUNT]
        != state.binding.registered_classes
        or len(reg1_binding.registered_classes) <= OLD_CLASS_COUNT
    ):
        raise NextR5FATarget125CoreError("REG1 FA reuse binding drift")
    return state


def fa_rdce3_reg1_reuse_receipt(
    state: Target125FARuntimeState,
    *,
    reg1_binding: Target125FARuntimeBinding,
) -> Mapping[str, Any]:
    reused = reuse_fa_rdce3_state_for_reg1(state, reg1_binding=reg1_binding)
    payload = {
        "schema": "cvs.phase2.next_r5.fa_rdce3.target125.reg1_reuse.v1",
        "candidate_id": CANDIDATE_ID,
        "da1_reg0_state_sha256": state.fa_state_receipt_sha256,
        "da1_reg1_state_sha256": reused.fa_state_receipt_sha256,
        "same_state_object": reused is state,
        "bitwise_state_reuse": reused.a_fp16.tobytes(order="C") == state.a_fp16.tobytes(order="C"),
        "reg0_fit_calls": 1,
        "reg1_fit_calls": 0,
        "new_class_support_rows_used_for_da": 0,
    }
    payload["reuse_receipt_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return _freeze_mapping(payload)


def transform_fa_rdce3(
    features: np.ndarray,
    state: Target125FARuntimeState,
) -> np.ndarray:
    """Apply the one prescribed FA shift/RDCE/final-signed-unit map.

    It performs the final signed-unit map exactly once.  qKNN receives this
    output directly and is forbidden from re-normalising it.
    """

    if type(state) is not Target125FARuntimeState:
        raise NextR5FATarget125CoreError("FA transform requires an exact runtime state")
    r0_rows = _unit_rows(features, "R0 features", nonnegative=None)
    basis = r4.decode_fa_rdce3_basis(state.asset.fa_asset).astype(np.float64)
    kappa = r4.decode_fa_rdce3_kappa(state.asset.fa_asset).astype(np.float64)
    shift = state.a_fp16.astype(np.float64)
    projected = r0_rows @ basis.T
    coeff = 1.0 - np.sqrt(1.0 - kappa)
    transformed = r0_rows - (
        shift[None, :] + coeff[None, :] * (projected - shift[None, :])
    ) @ basis
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise NextR5FATarget125CoreError("FA transform produced an undefined signed direction")
    result = np.asarray(transformed / norms, dtype=np.float32)
    _unit_rows(result, "R1 signed-unit result", nonnegative=None)
    return _readonly(result, np.dtype(np.float32))


def _qknn_payload(
    *,
    classes: tuple[str, ...],
    active_k: int,
    representation: str,
    codes: np.ndarray,
    scales: np.ndarray,
    indices: np.ndarray,
    class_scales: np.ndarray,
    support_physical_root_sha256: str,
    qknn_lock: Phase1ZIDStudentTLock,
) -> dict[str, Any]:
    return {
        "schema": QKNN_STATE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "classes": list(classes),
        "active_k": active_k,
        "representation": representation,
        "codes_qint8": _array_receipt(codes),
        "scales_fp16": _array_receipt(scales),
        "class_indices_int16": _array_receipt(indices),
        "class_scales_fp16": _array_receipt(class_scales),
        "support_physical_root_sha256": support_physical_root_sha256,
        "qknn_lock_digest": qknn_lock.lock_digest,
        "support_only": True,
        "all_registered_classes_scored": True,
        "tie_policy": QKNN_TIE_POLICY,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_batch_dependency": False,
        "class_quota_access": False,
        "global_reassignment_access": False,
        "r1_second_normalization": False,
    }


@dataclass(frozen=True, slots=True)
class Target125QKNNState:
    """INT8 support-only qKNN state that consumes pre-normalized R0/R1 rows."""

    classes: tuple[str, ...]
    active_k: int
    representation: str
    codes_qint8: np.ndarray
    scales_fp16: np.ndarray
    class_indices_int16: np.ndarray
    class_scales_fp16: np.ndarray
    support_physical_root_sha256: str
    qknn_lock: Phase1ZIDStudentTLock
    qknn_state_receipt_sha256: str
    schema: str = QKNN_STATE_SCHEMA

    def __post_init__(self) -> None:
        classes = _classes(self.classes, "qKNN classes")
        if (
            self.schema != QKNN_STATE_SCHEMA
            or self.active_k not in ALLOWED_K
            or self.representation not in (R0_REPRESENTATION, R1_REPRESENTATION)
            or type(self.qknn_lock) is not Phase1ZIDStudentTLock
            or self.qknn_lock.active_k != self.active_k
        ):
            raise NextR5FATarget125CoreError("qKNN state lifecycle drift")
        rows = len(classes) * self.active_k
        codes = np.asarray(self.codes_qint8)
        scales = np.asarray(self.scales_fp16)
        indices = np.asarray(self.class_indices_int16)
        class_scales = np.asarray(self.class_scales_fp16)
        if (
            codes.dtype != np.int8
            or codes.shape != (rows, Z_DIM)
            or scales.dtype != np.dtype("<f2")
            or scales.shape != (rows,)
            or indices.dtype != np.dtype("<i2")
            or indices.shape != (rows,)
            or class_scales.dtype != np.dtype("<f2")
            or class_scales.shape != (len(classes),)
            or np.any(codes == np.int8(-128))
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
            or not np.isfinite(class_scales).all()
            or np.any(class_scales <= 0.0)
            or np.any(indices < 0)
            or np.any(indices >= len(classes))
        ):
            raise NextR5FATarget125CoreError("qKNN state numeric closure drift")
        counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
        if any(value != self.active_k for value in counts):
            raise NextR5FATarget125CoreError("qKNN balanced support closure drift")
        support_root = _require_sha256(
            self.support_physical_root_sha256,
            "support_physical_root_sha256",
        )
        receipt = _require_sha256(self.qknn_state_receipt_sha256, "qknn_state_receipt_sha256")
        codes = _readonly(codes, np.dtype(np.int8))
        scales = _readonly(scales, np.dtype("<f2"))
        indices = _readonly(indices, np.dtype("<i2"))
        class_scales = _readonly(class_scales, np.dtype("<f2"))
        payload = _qknn_payload(
            classes=classes,
            active_k=self.active_k,
            representation=self.representation,
            codes=codes,
            scales=scales,
            indices=indices,
            class_scales=class_scales,
            support_physical_root_sha256=support_root,
            qknn_lock=self.qknn_lock,
        )
        if receipt != _sha256_bytes(_canonical_bytes(payload)):
            raise NextR5FATarget125CoreError("qKNN state receipt drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "codes_qint8", codes)
        object.__setattr__(self, "scales_fp16", scales)
        object.__setattr__(self, "class_indices_int16", indices)
        object.__setattr__(self, "class_scales_fp16", class_scales)
        object.__setattr__(self, "support_physical_root_sha256", support_root)
        object.__setattr__(self, "qknn_state_receipt_sha256", receipt)

    @property
    def dynamic_numeric_bytes(self) -> int:
        return int(
            self.codes_qint8.nbytes
            + self.scales_fp16.nbytes
            + self.class_indices_int16.nbytes
            + self.class_scales_fp16.nbytes
        )

    @property
    def resource_receipt(self) -> Mapping[str, Any]:
        return _freeze_mapping(
            {
                "schema": RESOURCE_SCHEMA,
                "candidate_id": CANDIDATE_ID,
                "kind": "direct_signed_unit_qknn",
                "qknn_state_sha256": self.qknn_state_receipt_sha256,
                "active_k": self.active_k,
                "class_count": len(self.classes),
                "support_rows": len(self.codes_qint8),
                "dynamic_numeric_bytes": self.dynamic_numeric_bytes,
                "query_mac_per_row": len(self.codes_qint8) * Z_DIM,
                "all_registered_classes_scored": True,
                "tie_policy": QKNN_TIE_POLICY,
                "r1_second_normalization": False,
            }
        )


def _quantize_direct_unit_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT8 codec with no post-decode unit renormalisation."""

    source = np.asarray(rows, dtype=np.float32)
    codes = np.zeros(source.shape, dtype=np.int8)
    scales = np.zeros(len(source), dtype=np.dtype("<f2"))
    decoded = np.zeros(source.shape, dtype=np.float64)
    minimum_scale = float(np.finfo(np.float16).tiny)
    for index, row in enumerate(source):
        scale = np.float16(max(float(np.max(np.abs(row))) / INT8_MAX, minimum_scale))
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise NextR5FATarget125CoreError("qKNN direct-unit quantization scale overflow")
        code = np.clip(np.rint(row / float(scale)), -INT8_MAX, INT8_MAX).astype(np.int8)
        codes[index] = code
        scales[index] = scale
        decoded[index] = code.astype(np.float64) * float(scale)
    return codes, scales, decoded


def _class_scales(
    decoded: np.ndarray,
    indices: np.ndarray,
    class_count: int,
    lock: Phase1ZIDStudentTLock,
) -> np.ndarray:
    if lock.active_k == 1:
        return np.full(class_count, lock.shared_h0, dtype=np.float64)
    values: list[float] = []
    for class_index in range(class_count):
        local = decoded[indices == class_index]
        cosine = np.clip(local @ local.T, -1.0, 1.0)
        distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
        upper = distance[np.triu_indices(lock.active_k, 1)]
        if len(upper) == 0:
            raise NextR5FATarget125CoreError("qKNN class scale has no within-class pairs")
        empirical = float(np.mean(upper))
        shrunk = (
            empirical + lock.scale_prior_strength * lock.shared_h0**2
        ) / (1.0 + lock.scale_prior_strength)
        values.append(
            float(
                np.clip(
                    math.sqrt(max(shrunk, np.finfo(np.float64).tiny)),
                    lock.shared_h0 * lock.scale_min_ratio,
                    lock.shared_h0 * lock.scale_max_ratio,
                )
            )
        )
    return np.asarray(values, dtype=np.float64)


def fit_qknn(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    support_physical_ids: Sequence[str],
    representation: str = R0_REPRESENTATION,
) -> Target125QKNNState:
    """Fit one support-only all-registered-class qKNN state.

    R1 inputs are already signed-unit.  This function merely validates that
    fact; it never applies ReLU or a second normalisation.
    """

    classes = _classes(registered_classes, "registered_classes")
    rows = _representation_rows(support_features, representation, "support_features")
    labels = tuple(_text(item, "support_labels") for item in support_labels)
    physical_ids = _physical_ids(
        support_physical_ids,
        "support_physical_ids",
        expected=len(rows),
    )
    if len(labels) != len(rows) or any(label not in classes for label in labels):
        raise NextR5FATarget125CoreError("qKNN support label/row closure drift")
    class_index = {name: index for index, name in enumerate(classes)}
    indices = np.asarray([class_index[label] for label in labels], dtype=np.dtype("<i2"))
    counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
    if any(count < 1 for count in counts) or len(set(counts)) != 1 or counts[0] not in ALLOWED_K:
        raise NextR5FATarget125CoreError("qKNN support must be balanced K={1,5,10}")
    active_k = counts[0]
    lock = phase1_qknn_lock(active_k)
    if type(lock) is not Phase1ZIDStudentTLock or lock.active_k != active_k:
        raise NextR5FATarget125CoreError("sealed qKNN lock/K drift")
    # Physical IDs establish deterministic storage order only; they never
    # select or score a support row.
    order = np.asarray(
        sorted(range(len(rows)), key=lambda index: (int(indices[index]), physical_ids[index])),
        dtype=np.intp,
    )
    ordered_rows = np.asarray(rows[order], dtype=np.float32)
    ordered_indices = np.asarray(indices[order], dtype=np.dtype("<i2"))
    codes, scales, decoded = _quantize_direct_unit_rows(ordered_rows)
    raw_class_scales = _class_scales(decoded, ordered_indices, len(classes), lock)
    class_scales = np.asarray(raw_class_scales, dtype=np.dtype("<f2"))
    if not np.isfinite(class_scales).all() or np.any(class_scales <= 0.0):
        raise NextR5FATarget125CoreError("qKNN class bandwidth FP16 closure drift")
    root = _physical_root(physical_ids)
    payload = _qknn_payload(
        classes=classes,
        active_k=active_k,
        representation=representation,
        codes=codes,
        scales=scales,
        indices=ordered_indices,
        class_scales=class_scales,
        support_physical_root_sha256=root,
        qknn_lock=lock,
    )
    return Target125QKNNState(
        classes=classes,
        active_k=active_k,
        representation=representation,
        codes_qint8=codes,
        scales_fp16=scales,
        class_indices_int16=ordered_indices,
        class_scales_fp16=class_scales,
        support_physical_root_sha256=root,
        qknn_lock=lock,
        qknn_state_receipt_sha256=_sha256_bytes(_canonical_bytes(payload)),
    )


def score_qknn(state: Target125QKNNState, query_features: np.ndarray) -> np.ndarray:
    """Score each query independently over every registered class."""

    if type(state) is not Target125QKNNState:
        raise NextR5FATarget125CoreError("qKNN score requires an exact typed state")
    query = _representation_rows(query_features, state.representation, "query_features")
    support = (
        state.codes_qint8.astype(np.float64)
        * state.scales_fp16.astype(np.float64)[:, None]
    )
    cosine = np.clip(query @ support.T, -1.0, 1.0)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    columns: list[np.ndarray] = []
    lock = state.qknn_lock
    for class_index in range(len(state.classes)):
        local = distance[:, state.class_indices_int16 == class_index]
        if local.shape[1] != state.active_k:
            raise NextR5FATarget125CoreError("qKNN class support count drift while scoring")
        h = float(state.class_scales_fp16[class_index])
        kernel = (
            -lock.kernel_volume_gamma * lock.kernel_effective_dim * math.log(h)
            - 0.5
            * (lock.student_nu + lock.kernel_effective_dim)
            * np.log1p(local / (lock.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(
            maximum[:, 0] + np.log(np.sum(np.exp(kernel - maximum), axis=1)) - math.log(state.active_k)
        )
    result = np.asarray(np.stack(columns, axis=1), dtype=np.float32)
    if result.shape != (len(query), len(state.classes)) or not np.isfinite(result).all():
        raise NextR5FATarget125CoreError("qKNN logits shape/value drift")
    return _readonly(result, np.dtype(np.float32))


def _predict_qknn_logits(
    state: Target125QKNNState,
    logits: np.ndarray,
) -> tuple[str, ...]:
    """Choose the smallest frozen registry index among exact highest logits."""

    values = np.asarray(logits)
    if (
        type(state) is not Target125QKNNState
        or values.dtype != np.float32
        or values.ndim != 2
        or values.shape != (len(values), len(state.classes))
        or len(values) < 1
        or not np.isfinite(values).all()
    ):
        raise NextR5FATarget125CoreError("qKNN prediction logits/state drift")
    maxima = np.max(values, axis=1, keepdims=True)
    winners = values == maxima
    # np.argmax returns the first True, which is the minimum class index in
    # the already frozen registered-class order. No query metadata is read.
    indices = np.argmax(winners, axis=1)
    return tuple(state.classes[int(index)] for index in indices.tolist())


def predict_qknn(state: Target125QKNNState, query_features: np.ndarray) -> tuple[str, ...]:
    logits = score_qknn(state, query_features)
    return _predict_qknn_logits(state, logits)


@dataclass(frozen=True, slots=True)
class Target125FAFourState:
    """Support states for the four explicit DA/registration surfaces."""

    reg0_binding: Target125FARuntimeBinding
    reg1_binding: Target125FARuntimeBinding
    da0_reg0: Target125QKNNState
    da1_reg0: Target125QKNNState
    da0_reg1: Target125QKNNState
    da1_reg1: Target125QKNNState
    fa_state: Target125FARuntimeState | None
    reg1_reuse_receipt: Mapping[str, Any]
    schema: str = FOUR_STATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != FOUR_STATE_SCHEMA
            or type(self.reg0_binding) is not Target125FARuntimeBinding
            or type(self.reg1_binding) is not Target125FARuntimeBinding
            or self.reg0_binding.registration_phase != "REG0"
            or self.reg1_binding.registration_phase != "REG1"
            or self.reg1_binding.registered_class_indices[:OLD_CLASS_COUNT]
            != self.reg0_binding.registered_class_indices
            or self.reg1_binding.registered_classes[:OLD_CLASS_COUNT]
            != self.reg0_binding.registered_classes
        ):
            raise NextR5FATarget125CoreError("four-state binding closure drift")
        states = (self.da0_reg0, self.da1_reg0, self.da0_reg1, self.da1_reg1)
        if any(type(item) is not Target125QKNNState for item in states):
            raise NextR5FATarget125CoreError("four-state qKNN state type drift")
        if (
            self.da0_reg0.classes != self.reg0_binding.registered_classes
            or self.da1_reg0.classes != self.reg0_binding.registered_classes
            or self.da0_reg1.classes != self.reg1_binding.registered_classes
            or self.da1_reg1.classes != self.reg1_binding.registered_classes
        ):
            raise NextR5FATarget125CoreError("four-state qKNN registry drift")
        active_k = self.reg0_binding.k_shot
        if (
            self.reg1_binding.k_shot != active_k
            or any(item.active_k != active_k for item in states)
        ):
            raise NextR5FATarget125CoreError("four-state K drift")
        if active_k == 1:
            if (
                self.fa_state is not None
                or self.da1_reg0 is not self.da0_reg0
                or self.da1_reg1 is not self.da0_reg1
                or dict(self.reg1_reuse_receipt).get("fit_mode") != FIT_MODE_K1_STRICT_BYPASS
            ):
                raise NextR5FATarget125CoreError("K1 requires exact FA/qKNN alias states")
        else:
            if (
                type(self.fa_state) is not Target125FARuntimeState
                or self.da1_reg0.representation != R1_REPRESENTATION
                or self.da1_reg1.representation != R1_REPRESENTATION
                or dict(self.reg1_reuse_receipt).get("same_state_object") is not True
            ):
                raise NextR5FATarget125CoreError("K5/K10 FA state/reuse closure drift")


def _combined_binding_ids(
    reg0_binding: Target125FARuntimeBinding,
    reg1_binding: Target125FARuntimeBinding,
    new_physical_ids: Sequence[str],
) -> tuple[str, ...]:
    new_ids = _physical_ids(
        new_physical_ids,
        "new_support_physical_ids",
        expected=(len(reg1_binding.registered_classes) - OLD_CLASS_COUNT)
        * reg1_binding.k_shot,
    )
    combined = reg0_binding.support_physical_ids + new_ids
    if tuple(combined) != reg1_binding.support_physical_ids:
        raise NextR5FATarget125CoreError("REG1 support physical-ID binding drift")
    return combined


def build_fa_qknn_four_state(
    asset: Target125FARDCE3Asset,
    *,
    reg0_binding: Target125FARuntimeBinding,
    reg1_binding: Target125FARuntimeBinding,
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_support_physical_ids: Sequence[str],
) -> Target125FAFourState:
    """Build all four support states without query input or target optimisation."""

    validate_target_fa_asset(asset)
    if (
        type(reg0_binding) is not Target125FARuntimeBinding
        or type(reg1_binding) is not Target125FARuntimeBinding
        or reg0_binding.registered_class_indices != asset.source_class_indices
        or reg1_binding.registered_class_indices[:OLD_CLASS_COUNT]
        != asset.source_class_indices
        or reg1_binding.registered_classes[:OLD_CLASS_COUNT]
        != reg0_binding.registered_classes
        or reg0_binding.k_shot != reg1_binding.k_shot
    ):
        raise NextR5FATarget125CoreError("four-state asset/binding registry drift")
    k_shot = reg0_binding.k_shot
    old_rows = _unit_rows(
        old_support_features,
        "old_support_features",
        nonnegative=None,
        expected_rows=OLD_CLASS_COUNT * k_shot,
    )
    new_rows = _unit_rows(
        new_support_features,
        "new_support_features",
        nonnegative=None,
        expected_rows=(len(reg1_binding.registered_classes) - OLD_CLASS_COUNT) * k_shot,
    )
    old_labels = tuple(_text(item, "old_support_labels") for item in old_support_labels)
    new_labels = tuple(_text(item, "new_support_labels") for item in new_support_labels)
    if (
        len(old_labels) != len(old_rows)
        or len(new_labels) != len(new_rows)
        or any(item not in reg0_binding.registered_classes for item in old_labels)
        or any(item not in reg1_binding.registered_classes[OLD_CLASS_COUNT:] for item in new_labels)
    ):
        raise NextR5FATarget125CoreError("four-state support label partition drift")
    all_rows = np.ascontiguousarray(np.concatenate((old_rows, new_rows), axis=0), dtype=np.float32)
    all_labels = old_labels + new_labels
    all_ids = _combined_binding_ids(reg0_binding, reg1_binding, new_support_physical_ids)
    da0_reg0 = fit_qknn(
        np.asarray(old_rows, dtype=np.float32),
        old_labels,
        reg0_binding.registered_classes,
        support_physical_ids=reg0_binding.support_physical_ids,
        representation=R0_REPRESENTATION,
    )
    da0_reg1 = fit_qknn(
        all_rows,
        all_labels,
        reg1_binding.registered_classes,
        support_physical_ids=all_ids,
        representation=R0_REPRESENTATION,
    )
    if k_shot == 1:
        bypass = _freeze_mapping(
            {
                "schema": "cvs.phase2.next_r5.fa_rdce3.target125.k1_bypass.v1",
                "candidate_id": CANDIDATE_ID,
                "fit_mode": FIT_MODE_K1_STRICT_BYPASS,
                "fa_fit_calls": 0,
                "fa_dynamic_numeric_bytes": 0,
                "DA1_REG0_alias_of": "DA0_REG0",
                "DA1_REG1_alias_of": "DA0_REG1",
                "query_rows_used_for_fit": 0,
                "new_class_support_rows_used_for_da": 0,
            }
        )
        return Target125FAFourState(
            reg0_binding=reg0_binding,
            reg1_binding=reg1_binding,
            da0_reg0=da0_reg0,
            da1_reg0=da0_reg0,
            da0_reg1=da0_reg1,
            da1_reg1=da0_reg1,
            fa_state=None,
            reg1_reuse_receipt=bypass,
        )
    fa_state = fit_fa_rdce3(
        asset,
        np.asarray(old_rows, dtype=np.float32),
        old_labels,
        reg0_binding.registered_classes,
        k_shot,
        binding=reg0_binding,
    )
    reused = reuse_fa_rdce3_state_for_reg1(fa_state, reg1_binding=reg1_binding)
    old_r1 = transform_fa_rdce3(np.asarray(old_rows, dtype=np.float32), fa_state)
    new_r1 = transform_fa_rdce3(np.asarray(new_rows, dtype=np.float32), reused)
    all_r1 = np.ascontiguousarray(np.concatenate((old_r1, new_r1), axis=0), dtype=np.float32)
    da1_reg0 = fit_qknn(
        old_r1,
        old_labels,
        reg0_binding.registered_classes,
        support_physical_ids=reg0_binding.support_physical_ids,
        representation=R1_REPRESENTATION,
    )
    da1_reg1 = fit_qknn(
        all_r1,
        all_labels,
        reg1_binding.registered_classes,
        support_physical_ids=all_ids,
        representation=R1_REPRESENTATION,
    )
    return Target125FAFourState(
        reg0_binding=reg0_binding,
        reg1_binding=reg1_binding,
        da0_reg0=da0_reg0,
        da1_reg0=da1_reg0,
        da0_reg1=da0_reg1,
        da1_reg1=da1_reg1,
        fa_state=fa_state,
        reg1_reuse_receipt=fa_rdce3_reg1_reuse_receipt(
            fa_state,
            reg1_binding=reg1_binding,
        ),
    )


@dataclass(frozen=True, slots=True)
class Target125FAFourStateScores:
    """Query-only immutable output; REG0 has no new/H metric value."""

    logits_by_state: Mapping[str, np.ndarray]
    predictions_by_state: Mapping[str, tuple[str, ...]]
    audit: Mapping[str, Any]
    schema: str = SCORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCORE_SCHEMA or set(self.logits_by_state) != set(matrix.STATES) or set(
            self.predictions_by_state
        ) != set(matrix.STATES):
            raise NextR5FATarget125CoreError("four-state score field closure drift")
        if (
            self.audit.get("query_rows_used_for_fit") != 0
            or self.audit.get("query_truth_access") is not False
            or self.audit.get("tie_policy") != QKNN_TIE_POLICY
        ):
            raise NextR5FATarget125CoreError("four-state score query-access ledger drift")


def _assert_query_binding(
    binding: Target125FARuntimeBinding,
    physical_ids: Sequence[str],
    rows: np.ndarray,
    name: str,
) -> None:
    ids = _physical_ids(physical_ids, f"{name}_physical_ids", expected=len(rows))
    if tuple(ids) != binding.query_physical_ids:
        raise NextR5FATarget125CoreError(f"{name} query physical-ID binding drift")


def score_fa_qknn_four_state(
    state: Target125FAFourState,
    *,
    reg0_query_features: np.ndarray,
    reg0_query_physical_ids: Sequence[str],
    reg1_query_features: np.ndarray,
    reg1_query_physical_ids: Sequence[str],
) -> Target125FAFourStateScores:
    """Score all four surfaces without a query fit, update, role, or truth input."""

    if type(state) is not Target125FAFourState:
        raise NextR5FATarget125CoreError("four-state score requires exact support state")
    reg0_query = _unit_rows(
        reg0_query_features,
        "reg0_query_features",
        nonnegative=None,
    )
    reg1_query = _unit_rows(
        reg1_query_features,
        "reg1_query_features",
        nonnegative=None,
    )
    _assert_query_binding(
        state.reg0_binding,
        reg0_query_physical_ids,
        reg0_query,
        "REG0",
    )
    _assert_query_binding(
        state.reg1_binding,
        reg1_query_physical_ids,
        reg1_query,
        "REG1",
    )
    da0_reg0_logits = score_qknn(state.da0_reg0, np.asarray(reg0_query, dtype=np.float32))
    da0_reg1_logits = score_qknn(state.da0_reg1, np.asarray(reg1_query, dtype=np.float32))
    da0_reg0_predictions = _predict_qknn_logits(state.da0_reg0, da0_reg0_logits)
    da0_reg1_predictions = _predict_qknn_logits(state.da0_reg1, da0_reg1_logits)
    if state.reg0_binding.k_shot == 1:
        logits = {
            "DA0_REG0": da0_reg0_logits,
            "DA1_REG0": da0_reg0_logits,
            "DA0_REG1": da0_reg1_logits,
            "DA1_REG1": da0_reg1_logits,
        }
        predictions = {
            "DA0_REG0": da0_reg0_predictions,
            "DA1_REG0": da0_reg0_predictions,
            "DA0_REG1": da0_reg1_predictions,
            "DA1_REG1": da0_reg1_predictions,
        }
        audit = {
            "schema": SCORE_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "fit_mode": FIT_MODE_K1_STRICT_BYPASS,
            "tie_policy": QKNN_TIE_POLICY,
            "exact_logit_alias": True,
            "exact_prediction_alias": True,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_truth_access": False,
            "query_role_access": False,
            "query_batch_dependency": False,
            "class_quota_access": False,
            "global_reassignment_access": False,
        }
    else:
        assert state.fa_state is not None
        r1_reg0 = transform_fa_rdce3(np.asarray(reg0_query, dtype=np.float32), state.fa_state)
        r1_reg1 = transform_fa_rdce3(np.asarray(reg1_query, dtype=np.float32), state.fa_state)
        da1_reg0_logits = score_qknn(state.da1_reg0, r1_reg0)
        da1_reg1_logits = score_qknn(state.da1_reg1, r1_reg1)
        logits = {
            "DA0_REG0": da0_reg0_logits,
            "DA1_REG0": da1_reg0_logits,
            "DA0_REG1": da0_reg1_logits,
            "DA1_REG1": da1_reg1_logits,
        }
        predictions = {
            name: _predict_qknn_logits(source, values)
            for name, source, values in (
                ("DA0_REG0", state.da0_reg0, da0_reg0_logits),
                ("DA1_REG0", state.da1_reg0, da1_reg0_logits),
                ("DA0_REG1", state.da0_reg1, da0_reg1_logits),
                ("DA1_REG1", state.da1_reg1, da1_reg1_logits),
            )
        }
        audit = {
            "schema": SCORE_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "fit_mode": FIT_MODE_FISHER_CLOSED_FORM,
            "tie_policy": QKNN_TIE_POLICY,
            "exact_logit_alias": False,
            "exact_prediction_alias": False,
            "r1_second_normalization": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_truth_access": False,
            "query_role_access": False,
            "query_batch_dependency": False,
            "class_quota_access": False,
            "global_reassignment_access": False,
        }
    return Target125FAFourStateScores(
        logits_by_state=MappingProxyType(logits),
        predictions_by_state=MappingProxyType(predictions),
        audit=_freeze_mapping(audit),
    )


__all__ = [
    "ALLOWED_K",
    "CANDIDATE_ID",
    "FA_FIT_K",
    "FA_RANK",
    "FIT_MODE_FISHER_CLOSED_FORM",
    "FIT_MODE_K1_STRICT_BYPASS",
    "FIT_MODE_POSTERIOR_ZERO_FIXED_RDCE",
    "NextR5FATarget125CoreError",
    "OLD_CLASS_COUNT",
    "PROTOCOL_SCHEMA",
    "QKNN_TIE_POLICY",
    "R0_REPRESENTATION",
    "R1_REPRESENTATION",
    "Target125FARDCE3Asset",
    "Target125FAFourState",
    "Target125FAFourStateScores",
    "Target125FARuntimeBinding",
    "Target125FARuntimeState",
    "Target125QKNNState",
    "Z_DIM",
    "build_fa_qknn_four_state",
    "build_target_fa_asset",
    "deserialize_target_fa_asset",
    "fa_rdce3_reg1_reuse_receipt",
    "fit_fa_rdce3",
    "fit_qknn",
    "predict_qknn",
    "reuse_fa_rdce3_state_for_reg1",
    "score_fa_qknn_four_state",
    "score_qknn",
    "serialize_target_fa_asset",
    "transform_fa_rdce3",
    "validate_target_fa_asset",
]
