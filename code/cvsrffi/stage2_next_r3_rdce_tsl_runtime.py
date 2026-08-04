"""Frozen RDCE×TSL-160 four-state, six-arm integration.

This module is deliberately a narrow Phase2 runtime.  It receives an already
sealed D106 RDCE state and never exposes an API for fitting it again.  The
only target-side fitting performed here is the support-only qKNN/F/TSL head
fit on one of two immutable caches.  Query rows are only scored.

``REG0`` and ``REG1`` each emit ``R0Q/R0F/R0L/R1Q/R1F/R1L``.  The four named
causal states are views over those two six-arm bundles:

* ``DA0_REG0`` -> ``REG0/R0``;
* ``DA1_REG0`` -> ``REG0/R1``;
* ``DA0_REG1`` -> ``REG1/R0``;
* ``DA1_REG1`` -> ``REG1/R1``.

The R1 state fitted on REG0 support is intentionally reused byte-for-byte in
REG1.  REG1 is accepted only when its support starts with the exact canonical
REG0 support rows and appends new-class support; it may not refit or replace
the RDCE state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_d106_rdce_runtime as rdce
from . import stage2_d129_joint6_heads as d129
from . import stage2_next_r3_tsl160 as tsl
from . import stage2_zid_student_t_qknn as qknn


RUNTIME_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.runtime.v1"
BRIDGE_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.bridge.v1"
CACHE_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.cache.v1"
REGISTRATION_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.registration.v1"
FOUR_STATE_SCHEMA = "cvs.stage2.next_r3.rdce_tsl160.four_state.v1"

Z_DIM = 160
CANONICAL_REPRESENTATION_RULE = "d106_canonical_normalized_relu_zid160"
ARM_IDS = ("R0Q", "R0F", "R0L", "R1Q", "R1F", "R1L")
REGISTRATION_STATES = ("REG0", "REG1")
FOUR_STATE_IDS = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")


class NextR3RDCETSLRuntimeError(ValueError):
    """Raised when the frozen R3 bridge or four-state closure drifts."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _array_receipt(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise NextR3RDCETSLRuntimeError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise NextR3RDCETSLRuntimeError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise NextR3RDCETSLRuntimeError(f"{name} must be a non-empty exact string")
    return value


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return MappingProxyType(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, tuple | list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value, dtype=value.dtype)
    return value


def _registry(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_require_text(value, name=name) for value in values)
    if len(result) < 2 or len(set(result)) != len(result):
        raise NextR3RDCETSLRuntimeError(f"{name} must be a unique registry of at least two classes")
    return result


def _tokens(values: Sequence[str], *, expected: int, name: str) -> tuple[str, ...]:
    result = tuple(_require_text(value, name=name) for value in values)
    if len(result) != expected or len(set(result)) != len(result):
        raise NextR3RDCETSLRuntimeError(f"{name} must be unique and aligned to its rows")
    return result


def _raw_prerelu160(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR3RDCETSLRuntimeError(f"{name} must be a numpy float32 array")
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise NextR3RDCETSLRuntimeError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return _readonly(value, dtype=np.float32)


def _unit_zid160(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR3RDCETSLRuntimeError(f"{name} must be a numpy float32 array")
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != Z_DIM
        or not np.isfinite(value).all()
    ):
        raise NextR3RDCETSLRuntimeError(f"{name} must be finite float32 [N,{Z_DIM}]")
    norms = np.linalg.norm(np.asarray(value, dtype=np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR3RDCETSLRuntimeError(f"{name} is not a unit z_id160 representation")
    return _readonly(value, dtype=np.float32)


def _physical_root(values: Sequence[str]) -> str:
    return _canonical_sha256(sorted(values))


def _ordered_physical_root(values: Sequence[str]) -> str:
    return _canonical_sha256(list(values))


def _require_zero_query_receipt(receipt: Mapping[str, Any], *, name: str) -> None:
    for field in (
        "query_rows_used_for_fit",
        "query_state_updates",
        "query_selection_count",
        "query_gradient_calls",
        "global_reassignment_calls",
    ):
        if field in receipt and receipt[field] != 0:
            raise NextR3RDCETSLRuntimeError(f"{name} receipt has non-zero {field}")
    for field in ("query_role_access", "query_truth_access", "query_batch_dependency"):
        if field in receipt and receipt[field] is not False:
            raise NextR3RDCETSLRuntimeError(f"{name} receipt has forbidden {field}")


def _state_fingerprint(state: rdce.D106RDCERuntimeState) -> tuple[Any, ...]:
    return (
        state.runtime_receipt_sha256,
        state.asset.binding_sha256,
        state.attenuation.tobytes(order="C"),
        state.capsule_id,
        state.split_id,
        state.row_id,
        state.seed,
        state.qknn_bank_sha256,
        state.support_physical_root_sha256,
        state.support_binding_sha256,
        state.active_k,
        state.registered_class_count,
        state.query_rows_used_for_fit,
        state.query_state_updates,
        state.target_optimizer_steps,
    )


def _representation_context_sha256(
    *,
    bridge: "NextR3RDCEBridgeBinding",
    da_state: rdce.D106RDCERuntimeState,
    representation: str,
) -> str:
    """Bind a TSL cache to the exact R0/R1 representation contract.

    The context intentionally identifies the R1 RDCE *transform* but does not
    reinterpret the R0 Phase1 prior as an R1 covariance object.  That keeps
    the frozen prior an ambient-axis source anchor on both representations.
    """

    if representation == "R0":
        mode = tsl.CANONICAL_R0
    elif representation == "R1":
        mode = tsl.RDCE_R1_SIGNED_UNIT
    else:
        raise NextR3RDCETSLRuntimeError("TSL representation must be R0 or R1")
    payload: dict[str, Any] = {
        "schema": "cvs.stage2.next_r3.rdce_tsl160.representation_context.v1",
        "bridge_sha256": bridge.binding_sha256,
        "checkpoint_sha256": bridge.checkpoint_sha256,
        "received_iq_root_sha256": bridge.received_iq_root_sha256,
        "tap_sha256": bridge.tap_sha256,
        "representation": representation,
        "representation_mode": mode,
        "representation_rule": bridge.representation_rule,
        "representation_rule_sha256": bridge.representation_rule_sha256,
        "prior_semantics": tsl.PRIOR_SEMANTICS,
        "prior_transported_by_rdce": False,
        "r1_covariance_claim": False,
    }
    if representation == "R1":
        payload.update(
            {
                "rdce_runtime_state_sha256": da_state.runtime_receipt_sha256,
                "rdce_asset_binding_sha256": da_state.asset.binding_sha256,
                "rdce_support_binding_sha256": da_state.support_binding_sha256,
            }
        )
    return _canonical_sha256(payload)


@dataclass(frozen=True, slots=True)
class NextR3RDCEBridgeBinding:
    """Exact bridge between frozen D106 state, received IQ, and TSL binding."""

    checkpoint_sha256: str
    capsule_id: str
    split_id: str
    row_id: str
    seed: int
    received_iq_root_sha256: str
    tap_sha256: str
    representation_rule_sha256: str
    phase1_physical_id_root_sha256: str
    phase1_seal_sha256: str
    outer_fold_id: str
    representation_rule: str = CANONICAL_REPRESENTATION_RULE
    schema: str = BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BRIDGE_SCHEMA:
            raise NextR3RDCETSLRuntimeError("R3 bridge schema drift")
        for name in (
            "checkpoint_sha256",
            "capsule_id",
            "split_id",
            "received_iq_root_sha256",
            "tap_sha256",
            "representation_rule_sha256",
            "phase1_physical_id_root_sha256",
            "phase1_seal_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        object.__setattr__(self, "row_id", _require_text(self.row_id, name="row_id"))
        object.__setattr__(
            self, "outer_fold_id", _require_text(self.outer_fold_id, name="outer_fold_id")
        )
        if type(self.seed) is not int or self.seed < 0:
            raise NextR3RDCETSLRuntimeError("R3 bridge seed must be a non-negative exact integer")
        if self.representation_rule != CANONICAL_REPRESENTATION_RULE:
            raise NextR3RDCETSLRuntimeError("R3 bridge forbids a non-D106 canonical representation")

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "checkpoint_sha256": self.checkpoint_sha256,
                "capsule_id": self.capsule_id,
                "split_id": self.split_id,
                "row_id": self.row_id,
                "seed": self.seed,
                "received_iq_root_sha256": self.received_iq_root_sha256,
                "tap_sha256": self.tap_sha256,
                "representation_rule": self.representation_rule,
                "representation_rule_sha256": self.representation_rule_sha256,
                "phase1_physical_id_root_sha256": self.phase1_physical_id_root_sha256,
                "phase1_seal_sha256": self.phase1_seal_sha256,
                "outer_fold_id": self.outer_fold_id,
            }
        )


@dataclass(frozen=True, slots=True)
class NextR3RegistrationInput:
    """One REG0 or REG1 sealed row, with no query labels or roles."""

    registration_state: str
    received_iq_root_sha256: str
    support_pre_relu160: np.ndarray
    query_pre_relu160: np.ndarray
    support_labels: Sequence[str]
    registered_classes: Sequence[str]
    support_physical_ids: Sequence[str]
    query_physical_ids: Sequence[str]

    def __post_init__(self) -> None:
        if self.registration_state not in REGISTRATION_STATES:
            raise NextR3RDCETSLRuntimeError("registration_state must be REG0 or REG1")
        support = _raw_prerelu160(self.support_pre_relu160, name="support_pre_relu160")
        query = _raw_prerelu160(self.query_pre_relu160, name="query_pre_relu160")
        classes = _registry(self.registered_classes, name="registered_classes")
        labels = tuple(_require_text(value, name="support_labels") for value in self.support_labels)
        if len(labels) != len(support) or any(value not in classes for value in labels):
            raise NextR3RDCETSLRuntimeError("support labels/registry closure drift")
        counts = tuple(labels.count(class_id) for class_id in classes)
        if len(set(counts)) != 1 or counts[0] not in {1, 5}:
            raise NextR3RDCETSLRuntimeError("R3 requires balanced K1 or K5 support")
        support_ids = _tokens(
            self.support_physical_ids,
            expected=len(support),
            name="support_physical_ids",
        )
        query_ids = _tokens(
            self.query_physical_ids,
            expected=len(query),
            name="query_physical_ids",
        )
        if set(support_ids) & set(query_ids):
            raise NextR3RDCETSLRuntimeError("support/query physical IDs must be disjoint")
        object.__setattr__(
            self,
            "received_iq_root_sha256",
            _require_sha256(
                self.received_iq_root_sha256, name="received_iq_root_sha256"
            ),
        )
        object.__setattr__(self, "support_pre_relu160", support)
        object.__setattr__(self, "query_pre_relu160", query)
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_physical_ids", query_ids)

    @property
    def active_k(self) -> int:
        return self.support_labels.count(self.registered_classes[0])


@dataclass(frozen=True, slots=True)
class NextR3FeatureCache:
    """One immutable, physical-order-bound R0 or R1 cache shared by Q/F/L."""

    representation: str
    registration_state: str
    support_zid160: np.ndarray
    query_zid160: np.ndarray
    support_labels: tuple[str, ...]
    registered_classes: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.representation not in {"R0", "R1"}:
            raise NextR3RDCETSLRuntimeError("cache representation must be R0 or R1")
        if self.registration_state not in REGISTRATION_STATES:
            raise NextR3RDCETSLRuntimeError("cache registration state drift")
        support = _unit_zid160(self.support_zid160, name="cache support_zid160")
        query = _unit_zid160(self.query_zid160, name="cache query_zid160")
        classes = _registry(self.registered_classes, name="cache registered_classes")
        labels = tuple(_require_text(value, name="cache support_labels") for value in self.support_labels)
        if len(labels) != len(support) or any(value not in classes for value in labels):
            raise NextR3RDCETSLRuntimeError("cache support labels/registry drift")
        support_ids = _tokens(
            self.support_physical_ids,
            expected=len(support),
            name="cache support physical IDs",
        )
        query_ids = _tokens(
            self.query_physical_ids,
            expected=len(query),
            name="cache query physical IDs",
        )
        if set(support_ids) & set(query_ids):
            raise NextR3RDCETSLRuntimeError("cache support/query physical IDs overlap")
        observed = dict(self.receipt)
        expected = _canonical_sha256(
            {
                "schema": CACHE_SCHEMA,
                "representation": self.representation,
                "registration_state": self.registration_state,
                "support": _array_receipt(support),
                "query": _array_receipt(query),
                "support_labels": list(labels),
                "registered_classes": list(classes),
                "support_physical_ids": list(support_ids),
                "query_physical_ids": list(query_ids),
            }
        )
        if observed.get("cache_sha256") != expected:
            raise NextR3RDCETSLRuntimeError("R3 cache receipt drift")
        object.__setattr__(self, "support_zid160", support)
        object.__setattr__(self, "query_zid160", query)
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_labels", labels)
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_physical_ids", query_ids)
        object.__setattr__(self, "receipt", _freeze(observed))

    @property
    def cache_sha256(self) -> str:
        return str(self.receipt["cache_sha256"])

    @property
    def support_physical_root_sha256(self) -> str:
        return _physical_root(self.support_physical_ids)

    @property
    def query_physical_root_sha256(self) -> str:
        return _physical_root(self.query_physical_ids)

    @property
    def ordered_support_physical_root_sha256(self) -> str:
        return _ordered_physical_root(self.support_physical_ids)

    @property
    def ordered_query_physical_root_sha256(self) -> str:
        return _ordered_physical_root(self.query_physical_ids)


@dataclass(frozen=True, slots=True)
class NextR3Arm:
    """One strict-top, per-query arm sharing one representation cache."""

    arm_id: str
    head: str
    cache: NextR3FeatureCache
    logits: np.ndarray
    predictions: tuple[str, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.cache) is not NextR3FeatureCache:
            raise NextR3RDCETSLRuntimeError("R3 arm requires an exact feature cache")
        if self.head not in {"Q", "F", "L"} or self.arm_id != f"{self.cache.representation}{self.head}":
            raise NextR3RDCETSLRuntimeError("R3 arm identity/head drift")
        logits = self.logits
        if (
            not isinstance(logits, np.ndarray)
            or logits.dtype != np.float32
            or logits.ndim != 2
            or logits.shape != (len(self.cache.query_physical_ids), len(self.cache.registered_classes))
            or not np.isfinite(logits).all()
        ):
            raise NextR3RDCETSLRuntimeError("R3 arm logits shape/dtype/finite drift")
        if len(self.predictions) != len(logits) or any(
            prediction not in self.cache.registered_classes
            for prediction in self.predictions
        ):
            raise NextR3RDCETSLRuntimeError("R3 arm prediction registry drift")
        try:
            logits.setflags(write=False)
        except ValueError as error:
            raise NextR3RDCETSLRuntimeError("R3 arm logits must be safely frozen") from error
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "receipt", _freeze(dict(self.receipt)))


@dataclass(frozen=True, slots=True)
class NextR3RegistrationResult:
    """The six arms and receipts for one registration state."""

    registration_state: str
    caches: Mapping[str, NextR3FeatureCache]
    arms: Mapping[str, NextR3Arm]
    head_fits: Mapping[str, Any]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.registration_state not in REGISTRATION_STATES:
            raise NextR3RDCETSLRuntimeError("registration result state drift")
        if tuple(self.caches) != ("R0", "R1") or tuple(self.arms) != ARM_IDS:
            raise NextR3RDCETSLRuntimeError("registration result cache/arm closure drift")
        for representation, cache in self.caches.items():
            if type(cache) is not NextR3FeatureCache or cache.representation != representation:
                raise NextR3RDCETSLRuntimeError("registration cache identity drift")
            if cache.registration_state != self.registration_state:
                raise NextR3RDCETSLRuntimeError("registration cache state drift")
        for arm_id, arm in self.arms.items():
            if type(arm) is not NextR3Arm or arm.arm_id != arm_id:
                raise NextR3RDCETSLRuntimeError("registration arm identity drift")
            if arm.cache is not self.caches[arm.cache.representation]:
                raise NextR3RDCETSLRuntimeError("Q/F/L must share the exact cache object")
        object.__setattr__(self, "caches", MappingProxyType(dict(self.caches)))
        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))
        object.__setattr__(self, "head_fits", MappingProxyType(dict(self.head_fits)))
        object.__setattr__(self, "receipt", _freeze(dict(self.receipt)))


@dataclass(frozen=True, slots=True)
class NextR3RuntimeResult:
    """Two six-arm bundles plus explicit four-state causal views."""

    bridge: NextR3RDCEBridgeBinding
    da1_reg0_state_sha256: str
    da1_reg1_state_sha256: str
    reg0: NextR3RegistrationResult
    reg1: NextR3RegistrationResult
    four_state: Mapping[str, Mapping[str, NextR3Arm]]
    four_state_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]
    runtime_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.bridge) is not NextR3RDCEBridgeBinding:
            raise NextR3RDCETSLRuntimeError("runtime result bridge type drift")
        state0 = _require_sha256(
            self.da1_reg0_state_sha256, name="da1_reg0_state_sha256"
        )
        state1 = _require_sha256(
            self.da1_reg1_state_sha256, name="da1_reg1_state_sha256"
        )
        if state0 != state1:
            raise NextR3RDCETSLRuntimeError("DA1_REG1 must reuse the exact DA1_REG0 state SHA")
        if (
            type(self.reg0) is not NextR3RegistrationResult
            or self.reg0.registration_state != "REG0"
            or type(self.reg1) is not NextR3RegistrationResult
            or self.reg1.registration_state != "REG1"
        ):
            raise NextR3RDCETSLRuntimeError("runtime registration result closure drift")
        if tuple(self.four_state) != FOUR_STATE_IDS:
            raise NextR3RDCETSLRuntimeError("four-state result closure drift")
        expected = {
            "DA0_REG0": self.reg0.caches["R0"],
            "DA1_REG0": self.reg0.caches["R1"],
            "DA0_REG1": self.reg1.caches["R0"],
            "DA1_REG1": self.reg1.caches["R1"],
        }
        for state_id, arms in self.four_state.items():
            if tuple(arms) != ("Q", "F", "L"):
                raise NextR3RDCETSLRuntimeError("four-state head closure drift")
            for head, arm in arms.items():
                if arm.cache is not expected[state_id] or arm.head != head:
                    raise NextR3RDCETSLRuntimeError("four-state cache/arm binding drift")
        object.__setattr__(self, "da1_reg0_state_sha256", state0)
        object.__setattr__(self, "da1_reg1_state_sha256", state1)
        object.__setattr__(
            self,
            "four_state",
            MappingProxyType(
                {state_id: MappingProxyType(dict(arms)) for state_id, arms in self.four_state.items()}
            ),
        )
        object.__setattr__(self, "four_state_receipt", _freeze(dict(self.four_state_receipt)))
        object.__setattr__(self, "resource_receipt", _freeze(dict(self.resource_receipt)))
        object.__setattr__(self, "runtime_receipt", _freeze(dict(self.runtime_receipt)))


@dataclass(frozen=True, slots=True)
class _QHeadResult:
    bank: qknn.TypedINT8ZIDSupportBank
    metric: qknn.TypedSharedPSDMetric
    logits: np.ndarray
    receipt: Mapping[str, Any]


def _canonicalize_input(
    value: NextR3RegistrationInput,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        support = tsl.canonical_d106_relu_zid160(value.support_pre_relu160)
        query = tsl.canonical_d106_relu_zid160(value.query_pre_relu160)
    except Exception as error:
        raise NextR3RDCETSLRuntimeError(
            "D106 canonical normalized-ReLU bridge rejected input"
        ) from error
    support = _unit_zid160(support, name="canonical support_zid160")
    query = _unit_zid160(query, name="canonical query_zid160")
    if np.any(support < 0.0) or np.any(query < 0.0):
        raise NextR3RDCETSLRuntimeError("canonical normalized-ReLU bridge produced a signed fallback")
    return support, query


def _build_cache(
    *,
    representation: str,
    registration: NextR3RegistrationInput,
    support: np.ndarray,
    query: np.ndarray,
) -> NextR3FeatureCache:
    payload = {
        "schema": CACHE_SCHEMA,
        "representation": representation,
        "registration_state": registration.registration_state,
        "support": _array_receipt(support),
        "query": _array_receipt(query),
        "support_labels": list(registration.support_labels),
        "registered_classes": list(registration.registered_classes),
        "support_physical_ids": list(registration.support_physical_ids),
        "query_physical_ids": list(registration.query_physical_ids),
    }
    receipt = {
        "schema": CACHE_SCHEMA,
        "cache_sha256": _canonical_sha256(payload),
        "representation": representation,
        "registration_state": registration.registration_state,
        "feature_space": (
            CANONICAL_REPRESENTATION_RULE
            if representation == "R0"
            else "d106_rdce_phi_on_canonical_normalized_relu_zid160"
        ),
        "support_query_feature_cache_shared_by_three_heads": True,
        "signed_pre_relu_fallback_used": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    return NextR3FeatureCache(
        representation=representation,
        registration_state=registration.registration_state,
        support_zid160=support,
        query_zid160=query,
        support_labels=tuple(registration.support_labels),
        registered_classes=tuple(registration.registered_classes),
        support_physical_ids=tuple(registration.support_physical_ids),
        query_physical_ids=tuple(registration.query_physical_ids),
        receipt=receipt,
    )


def _validate_bridge_core(
    *,
    bridge: NextR3RDCEBridgeBinding,
    da1_reg0_state: rdce.D106RDCERuntimeState,
    reg0: NextR3RegistrationInput,
    reg1: NextR3RegistrationInput,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    tsl_runtime_binding: tsl.TSL160RuntimeBinding,
) -> None:
    if type(da1_reg0_state) is not rdce.D106RDCERuntimeState:
        raise NextR3RDCETSLRuntimeError("R3 requires an exact D106 RDCE runtime state")
    if da1_reg0_state.is_formal_deployable is not True:
        raise NextR3RDCETSLRuntimeError("R3 refuses a non-deployable D106 math state")
    if type(qknn_lock) is not qknn.Phase1ZIDStudentTLock:
        raise NextR3RDCETSLRuntimeError("R3 requires an exact frozen qKNN lock")
    if type(tsl_runtime_binding) is not tsl.TSL160RuntimeBinding:
        raise NextR3RDCETSLRuntimeError("R3 requires an exact TSL160 runtime binding")
    if (
        reg0.registration_state != "REG0"
        or reg1.registration_state != "REG1"
        or reg0.received_iq_root_sha256 != bridge.received_iq_root_sha256
        or reg1.received_iq_root_sha256 != bridge.received_iq_root_sha256
    ):
        raise NextR3RDCETSLRuntimeError("R3 received-IQ REG0/REG1 bridge drift")
    state = da1_reg0_state
    if (
        state.asset.checkpoint_sha256 != bridge.checkpoint_sha256
        or state.asset.tap_sha256 != bridge.tap_sha256
        or state.capsule_id != bridge.capsule_id
        or state.split_id != bridge.split_id
        or state.row_id != bridge.row_id
        or state.seed != bridge.seed
        or state.active_k != reg0.active_k
        or state.registered_class_count != len(reg0.registered_classes)
        or qknn_lock.active_k != reg0.active_k
        or qknn_lock.active_k != reg1.active_k
    ):
        raise NextR3RDCETSLRuntimeError("D106 state/checkpoint/capsule/tap/K bridge drift")
    if (
        tsl_runtime_binding.outer_fold_id != bridge.outer_fold_id
        or tsl_runtime_binding.checkpoint_sha256 != bridge.checkpoint_sha256
        or tsl_runtime_binding.representation_rule_sha256
        != bridge.representation_rule_sha256
        or tsl_runtime_binding.phase1_physical_id_root_sha256
        != bridge.phase1_physical_id_root_sha256
        or tsl_runtime_binding.phase1_seal_sha256 != bridge.phase1_seal_sha256
    ):
        raise NextR3RDCETSLRuntimeError("TSL/D106 bridge binding drift")
    if (
        state.query_rows_used_for_fit != 0
        or state.query_state_updates != 0
        or state.target_optimizer_steps != 0
    ):
        raise NextR3RDCETSLRuntimeError("D106 state violates query-read-only closure")


def _validate_reg1_append(
    *,
    reg0: NextR3RegistrationInput,
    reg1: NextR3RegistrationInput,
    reg0_support: np.ndarray,
    reg1_support: np.ndarray,
) -> None:
    old_classes = reg0.registered_classes
    old_rows = len(reg0_support)
    if (
        reg1.registered_classes[: len(old_classes)] != old_classes
        or len(reg1.registered_classes) <= len(old_classes)
        or reg1.active_k != reg0.active_k
        or reg1.support_labels[:old_rows] != reg0.support_labels
        or reg1.support_physical_ids[:old_rows] != reg0.support_physical_ids
        or not np.array_equal(reg1_support[:old_rows], reg0_support)
    ):
        raise NextR3RDCETSLRuntimeError(
            "REG1 must byte-preserve REG0 support and append new support only"
        )
    new_classes = set(reg1.registered_classes[len(old_classes) :])
    suffix_labels = reg1.support_labels[old_rows:]
    if not suffix_labels or any(label not in new_classes for label in suffix_labels):
        raise NextR3RDCETSLRuntimeError("REG1 support suffix must contain only appended classes")
    if any(suffix_labels.count(class_id) != reg0.active_k for class_id in new_classes):
        raise NextR3RDCETSLRuntimeError("REG1 appended classes violate frozen K")


def _q_head(cache: NextR3FeatureCache, lock: qknn.Phase1ZIDStudentTLock) -> _QHeadResult:
    try:
        bank = qknn.build_typed_zid_support_bank(
            cache.support_zid160,
            cache.support_labels,
            cache.registered_classes,
            config=lock,
        )
        metric = qknn.identity_shared_psd_metric(config=lock)
        logits = qknn.score_zid_student_t_logits(
            bank, cache.query_zid160, metric=metric
        )
        resource = qknn.audit_runtime_state(bank, metric)
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("R3 qKNN support-only compile/score failed") from error
    receipt = {
        "schema": "cvs.stage2.next_r3.rdce_tsl160.qknn.v1",
        "representation": cache.representation,
        "registration_state": cache.registration_state,
        "cache_sha256": cache.cache_sha256,
        "qknn_bank_sha256": bank.bank_receipt_sha256,
        "qknn_metric_sha256": metric.metric_receipt_sha256,
        "resource": dict(resource),
        "support_only_fit": True,
        "all_registered_classes_scored": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "query_truth_access": False,
        "query_batch_dependency": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
    }
    _require_zero_query_receipt(receipt, name="qKNN")
    return _QHeadResult(bank=bank, metric=metric, logits=logits, receipt=_freeze(receipt))


def _f_head(
    *,
    cache: NextR3FeatureCache,
    q_logits: np.ndarray,
    old_class_count: int,
) -> tuple[np.ndarray, Any | None, Mapping[str, Any]]:
    if cache.registration_state == "REG0":
        receipt = {
            "schema": "cvs.stage2.next_r3.rdce_tsl160.fallback_f.v1",
            "fit_mode": "historical_pre_registration_qknn_fallback",
            "cache_sha256": cache.cache_sha256,
            "underlying_qknn_logit_object_reused": True,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_role_access": False,
            "query_truth_access": False,
            "query_batch_dependency": False,
        }
        return q_logits, None, _freeze(receipt)
    if cache.registration_state != "REG1":
        raise NextR3RDCETSLRuntimeError("F head registration state drift")
    if cache.support_labels.count(cache.registered_classes[0]) == 1:
        receipt = {
            "schema": "cvs.stage2.next_r3.rdce_tsl160.fallback_f.v1",
            "fit_mode": "exact_qknn_logit_object_alias",
            "cache_sha256": cache.cache_sha256,
            "underlying_qknn_logit_object_reused": True,
            "historical_k1_equivalence_claim": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_role_access": False,
            "query_truth_access": False,
            "query_batch_dependency": False,
        }
        return q_logits, None, _freeze(receipt)
    try:
        fit = d129.fit_d92_full160(
            cache.support_zid160,
            cache.support_labels,
            cache.registered_classes,
            old_class_count=old_class_count,
        )
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("historical Full160 F support-only fit failed") from error
    if type(fit.state) is not d129.D129AffineHeadState:
        raise NextR3RDCETSLRuntimeError("REG1/K5 Full160 F must produce an affine state")
    try:
        logits = d129.score_d129_affine_head(fit.state, cache.query_zid160)
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("historical Full160 F query score failed") from error
    receipt = {
        "schema": "cvs.stage2.next_r3.rdce_tsl160.full160_f.v1",
        "fit_mode": fit.fit_receipt["fit_mode"],
        "cache_sha256": cache.cache_sha256,
        "old_class_count_derived_from_reg0_prefix": old_class_count,
        "head_state_sha256": fit.state.state_sha256,
        "fit_receipt": dict(fit.fit_receipt),
        "resource_receipt": dict(fit.resource_receipt),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "query_truth_access": False,
        "query_batch_dependency": False,
    }
    _require_zero_query_receipt(receipt, name="Full160 F")
    _require_zero_query_receipt(dict(fit.fit_receipt), name="Full160 F fit")
    return logits, fit, _freeze(receipt)


def _l_head(
    *,
    cache: NextR3FeatureCache,
    q_logits: np.ndarray,
    prior: tsl.TSL160Phase1Prior,
    runtime_binding: tsl.TSL160RuntimeBinding,
    representation_context_sha256: str,
) -> tuple[np.ndarray, Any, Mapping[str, Any]]:
    if cache.representation == "R0":
        representation_mode = tsl.CANONICAL_R0
    elif cache.representation == "R1":
        representation_mode = tsl.RDCE_R1_SIGNED_UNIT
    else:
        raise NextR3RDCETSLRuntimeError("TSL cache representation must be R0 or R1")
    support_cache_sha256 = tsl.tsl160_cache_sha256(cache.support_zid160)
    query_cache_sha256 = tsl.tsl160_cache_sha256(cache.query_zid160)
    try:
        fit = tsl.fit_tsl160(
            cache.support_zid160,
            cache.support_labels,
            cache.registered_classes,
            prior=prior,
            runtime_binding=runtime_binding,
            representation_mode=representation_mode,
            representation_context_sha256=representation_context_sha256,
            support_cache_sha256=support_cache_sha256,
        )
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("TSL-160 support-only fit failed") from error
    try:
        tsl.validate_tsl160_fit_binding(fit, runtime_binding)
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("TSL-160 fit binding validation failed") from error
    if cache.support_labels.count(cache.registered_classes[0]) == 1:
        try:
            logits = tsl.alias_k1_qknn_logits(
                fit, q_logits, runtime_binding=runtime_binding
            )
        except Exception as error:
            raise NextR3RDCETSLRuntimeError("TSL-160 K1 alias failed") from error
        if logits is not q_logits:
            raise NextR3RDCETSLRuntimeError("TSL-160 K1 must reuse the exact qKNN logit object")
        fit_mode = "exact_qknn_logit_object_alias"
    else:
        try:
            logits = tsl.score_tsl160_affine(
                fit,
                cache.query_zid160,
                runtime_binding=runtime_binding,
                representation_mode=representation_mode,
                representation_context_sha256=representation_context_sha256,
                query_cache_sha256=query_cache_sha256,
            )
        except Exception as error:
            raise NextR3RDCETSLRuntimeError("TSL-160 affine query score failed") from error
        fit_mode = "tsl160_affine"
    fit_receipt = dict(getattr(fit, "fit_receipt"))
    resource_receipt = dict(getattr(fit, "resource_receipt"))
    if (
        fit_receipt.get("representation_mode") != representation_mode
        or fit_receipt.get("representation_context_sha256")
        != representation_context_sha256
        or fit_receipt.get("support_cache_sha256") != support_cache_sha256
        or fit_receipt.get("prior_semantics") != tsl.PRIOR_SEMANTICS
        or fit_receipt.get("prior_transported_by_rdce") is not False
        or fit_receipt.get("r1_covariance_claim") is not False
    ):
        raise NextR3RDCETSLRuntimeError("TSL-160 fit receipt representation/prior drift")
    _require_zero_query_receipt(fit_receipt, name="TSL-160 fit")
    receipt = {
        "schema": "cvs.stage2.next_r3.rdce_tsl160.tsl.v1",
        "fit_mode": fit_mode,
        "cache_sha256": cache.cache_sha256,
        "representation_mode": representation_mode,
        "representation_context_sha256": representation_context_sha256,
        "tsl_support_cache_sha256": support_cache_sha256,
        "tsl_query_cache_sha256": query_cache_sha256,
        "prior_semantics": tsl.PRIOR_SEMANTICS,
        "prior_transported_by_rdce": False,
        "r1_covariance_claim": False,
        "fit_receipt": fit_receipt,
        "resource_receipt": resource_receipt,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_role_access": False,
        "query_truth_access": False,
        "query_batch_dependency": False,
    }
    _require_zero_query_receipt(receipt, name="TSL-160")
    return logits, fit, _freeze(receipt)


def _strict_arm(
    *,
    cache: NextR3FeatureCache,
    head: str,
    logits: np.ndarray,
    head_receipt: Mapping[str, Any],
) -> NextR3Arm:
    if not isinstance(logits, np.ndarray) or logits.dtype != np.float32:
        raise NextR3RDCETSLRuntimeError("head must return exact float32 logits")
    if logits.shape != (len(cache.query_physical_ids), len(cache.registered_classes)):
        raise NextR3RDCETSLRuntimeError("head logits shape does not match shared cache")
    try:
        tsl.require_unique_float32_top(logits)
    except Exception as error:
        raise NextR3RDCETSLRuntimeError(
            f"{cache.registration_state}/{cache.representation}{head} exact float32 top-tie closure failed"
        ) from error
    predictions = tuple(
        cache.registered_classes[int(index)] for index in np.argmax(logits, axis=1)
    )
    receipt = {
        "schema": "cvs.stage2.next_r3.rdce_tsl160.arm.v1",
        "arm_id": f"{cache.representation}{head}",
        "registration_state": cache.registration_state,
        "representation": cache.representation,
        "head": head,
        "cache_sha256": cache.cache_sha256,
        "query_physical_root_sha256": cache.query_physical_root_sha256,
        "same_representation_cache_shared": True,
        "all_registered_classes_scored": True,
        "independent_per_query": True,
        "exact_float32_top_tie_closed": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_batch_dependency": False,
        "head_receipt": dict(head_receipt),
    }
    return NextR3Arm(
        arm_id=f"{cache.representation}{head}",
        head=head,
        cache=cache,
        logits=logits,
        predictions=predictions,
        receipt=receipt,
    )


def _run_registration(
    *,
    bridge: NextR3RDCEBridgeBinding,
    registration: NextR3RegistrationInput,
    canonical_support: np.ndarray,
    canonical_query: np.ndarray,
    da_state: rdce.D106RDCERuntimeState,
    scoring_context: Any,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    tsl_prior: tsl.TSL160Phase1Prior,
    tsl_runtime_binding: tsl.TSL160RuntimeBinding,
    old_class_count: int,
) -> NextR3RegistrationResult:
    r0 = _build_cache(
        representation="R0",
        registration=registration,
        support=canonical_support,
        query=canonical_query,
    )
    before = _state_fingerprint(da_state)
    try:
        r1_support = rdce.transform_d106_rdce_zid(
            da_state, canonical_support, context=scoring_context
        )
        r1_query = rdce.transform_d106_rdce_query(
            da_state, canonical_query, context=scoring_context
        )
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("RDCE R1 transform failed") from error
    if before != _state_fingerprint(da_state):
        raise NextR3RDCETSLRuntimeError("RDCE query transform mutated the frozen DA state")
    r1 = _build_cache(
        representation="R1",
        registration=registration,
        support=r1_support,
        query=r1_query,
    )
    if (
        r0.support_physical_ids != r1.support_physical_ids
        or r0.query_physical_ids != r1.query_physical_ids
        or r0.support_physical_root_sha256 != r1.support_physical_root_sha256
        or r0.query_physical_root_sha256 != r1.query_physical_root_sha256
    ):
        raise NextR3RDCETSLRuntimeError("R0/R1 physical cache order/root drift")

    q_results = {"R0": _q_head(r0, qknn_lock), "R1": _q_head(r1, qknn_lock)}
    f_logits: dict[str, np.ndarray] = {}
    f_fits: dict[str, Any | None] = {}
    f_receipts: dict[str, Mapping[str, Any]] = {}
    l_logits: dict[str, np.ndarray] = {}
    l_fits: dict[str, Any] = {}
    l_receipts: dict[str, Mapping[str, Any]] = {}
    for representation, cache in (("R0", r0), ("R1", r1)):
        f_logits[representation], f_fits[representation], f_receipts[representation] = _f_head(
            cache=cache,
            q_logits=q_results[representation].logits,
            old_class_count=old_class_count,
        )
        l_logits[representation], l_fits[representation], l_receipts[representation] = _l_head(
            cache=cache,
            q_logits=q_results[representation].logits,
            prior=tsl_prior,
            runtime_binding=tsl_runtime_binding,
            representation_context_sha256=_representation_context_sha256(
                bridge=bridge,
                da_state=da_state,
                representation=representation,
            ),
        )
        if registration.active_k == 1 and (
            f_logits[representation] is not q_results[representation].logits
            or l_logits[representation] is not q_results[representation].logits
        ):
            raise NextR3RDCETSLRuntimeError("K1 F/L must be exact qKNN logit-object aliases")

    arms = {
        "R0Q": _strict_arm(
            cache=r0, head="Q", logits=q_results["R0"].logits, head_receipt=q_results["R0"].receipt
        ),
        "R0F": _strict_arm(cache=r0, head="F", logits=f_logits["R0"], head_receipt=f_receipts["R0"]),
        "R0L": _strict_arm(cache=r0, head="L", logits=l_logits["R0"], head_receipt=l_receipts["R0"]),
        "R1Q": _strict_arm(
            cache=r1, head="Q", logits=q_results["R1"].logits, head_receipt=q_results["R1"].receipt
        ),
        "R1F": _strict_arm(cache=r1, head="F", logits=f_logits["R1"], head_receipt=f_receipts["R1"]),
        "R1L": _strict_arm(cache=r1, head="L", logits=l_logits["R1"], head_receipt=l_receipts["R1"]),
    }
    metric_availability: Mapping[str, Any]
    if registration.registration_state == "REG0":
        metric_availability = MappingProxyType(
            {"seen_new_acc": "N/A", "H_old_new": "N/A", "reason": "new_class_not_registered"}
        )
    else:
        metric_availability = MappingProxyType(
            {"seen_new_acc": "scorer_only", "H_old_new": "scorer_only"}
        )
    receipt = {
        "schema": REGISTRATION_SCHEMA,
        "registration_state": registration.registration_state,
        "active_k": registration.active_k,
        "registered_classes": list(registration.registered_classes),
        "old_class_count": old_class_count,
        "new_class_count": len(registration.registered_classes) - old_class_count,
        "arm_ids": list(ARM_IDS),
        "r0_cache_sha256": r0.cache_sha256,
        "r1_cache_sha256": r1.cache_sha256,
        "r0_r1_support_order_byte_identical": r0.support_physical_ids == r1.support_physical_ids,
        "r0_r1_query_order_byte_identical": r0.query_physical_ids == r1.query_physical_ids,
        "tsl_representation_context_sha256": {
            representation: l_receipts[representation]["representation_context_sha256"]
            for representation in ("R0", "R1")
        },
        "qknn_bank_sha256": {
            "R0": q_results["R0"].bank.bank_receipt_sha256,
            "R1": q_results["R1"].bank.bank_receipt_sha256,
        },
        "head_fit_calls": {
            "R0Q": 1,
            "R0F": 0 if f_fits["R0"] is None else 1,
            "R0L": 1,
            "R1Q": 1,
            "R1F": 0 if f_fits["R1"] is None else 1,
            "R1L": 1,
        },
        "metric_availability": dict(metric_availability),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "query_batch_dependency": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
        "global_reassignment_calls": 0,
    }
    _require_zero_query_receipt(receipt, name="registration")
    return NextR3RegistrationResult(
        registration_state=registration.registration_state,
        caches=MappingProxyType({"R0": r0, "R1": r1}),
        arms=MappingProxyType(arms),
        head_fits=MappingProxyType(
            {
                "R0F": f_fits["R0"],
                "R0L": l_fits["R0"],
                "R1F": f_fits["R1"],
                "R1L": l_fits["R1"],
            }
        ),
        receipt=receipt,
    )


def _four_state_view(
    reg0: NextR3RegistrationResult, reg1: NextR3RegistrationResult
) -> Mapping[str, Mapping[str, NextR3Arm]]:
    return MappingProxyType(
        {
            "DA0_REG0": MappingProxyType(
                {head: reg0.arms[f"R0{head}"] for head in ("Q", "F", "L")}
            ),
            "DA1_REG0": MappingProxyType(
                {head: reg0.arms[f"R1{head}"] for head in ("Q", "F", "L")}
            ),
            "DA0_REG1": MappingProxyType(
                {head: reg1.arms[f"R0{head}"] for head in ("Q", "F", "L")}
            ),
            "DA1_REG1": MappingProxyType(
                {head: reg1.arms[f"R1{head}"] for head in ("Q", "F", "L")}
            ),
        }
    )


def _rdce_asset_numeric_bytes(asset: rdce.D106RDCEAsset) -> int:
    arrays = (
        asset.basis_codes_qint8,
        asset.basis_scales_fp16,
        asset.tau_codes_qint8,
        asset.tau_scales_fp16,
        asset.spectrum_codes_qint8,
        asset.spectrum_scales_fp16,
    )
    return int(sum(value.nbytes for value in arrays))


def execute_next_r3_four_state(
    *,
    bridge: NextR3RDCEBridgeBinding,
    da1_reg0_state: rdce.D106RDCERuntimeState,
    reg0: NextR3RegistrationInput,
    reg1: NextR3RegistrationInput,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
    tsl_prior: tsl.TSL160Phase1Prior,
    tsl_runtime_binding: tsl.TSL160RuntimeBinding,
) -> NextR3RuntimeResult:
    """Execute R3's frozen REG0/REG1 × R0/R1 × Q/F/L closure.

    The signature intentionally contains no query labels, query roles, quotas,
    source/clean data, scorer output, optimizer, or global-assignment hook.
    ``da1_reg0_state`` must have been fitted through the formal D106 loader on
    REG0 support before this call; this function only transforms it.
    """

    _validate_bridge_core(
        bridge=bridge,
        da1_reg0_state=da1_reg0_state,
        reg0=reg0,
        reg1=reg1,
        qknn_lock=qknn_lock,
        tsl_runtime_binding=tsl_runtime_binding,
    )
    reg0_support, reg0_query = _canonicalize_input(reg0)
    reg1_support, reg1_query = _canonicalize_input(reg1)
    _validate_reg1_append(
        reg0=reg0,
        reg1=reg1,
        reg0_support=reg0_support,
        reg1_support=reg1_support,
    )
    before = _state_fingerprint(da1_reg0_state)
    try:
        scoring_context = rdce.prepare_d106_rdce_scoring_context(da1_reg0_state)
    except Exception as error:
        raise NextR3RDCETSLRuntimeError("unable to prepare sealed RDCE scoring context") from error
    reg0_result = _run_registration(
        bridge=bridge,
        registration=reg0,
        canonical_support=reg0_support,
        canonical_query=reg0_query,
        da_state=da1_reg0_state,
        scoring_context=scoring_context,
        qknn_lock=qknn_lock,
        tsl_prior=tsl_prior,
        tsl_runtime_binding=tsl_runtime_binding,
        old_class_count=len(reg0.registered_classes),
    )
    if (
        reg0_result.receipt["qknn_bank_sha256"]["R0"]
        != da1_reg0_state.qknn_bank_sha256
        or reg0_result.caches["R0"].support_physical_root_sha256
        != da1_reg0_state.support_physical_root_sha256
    ):
        raise NextR3RDCETSLRuntimeError(
            "D106 state does not byte-bind the actual REG0 canonical support cache"
        )
    reg1_result = _run_registration(
        bridge=bridge,
        registration=reg1,
        canonical_support=reg1_support,
        canonical_query=reg1_query,
        da_state=da1_reg0_state,
        scoring_context=scoring_context,
        qknn_lock=qknn_lock,
        tsl_prior=tsl_prior,
        tsl_runtime_binding=tsl_runtime_binding,
        old_class_count=len(reg0.registered_classes),
    )
    old_rows = len(reg0.support_physical_ids)
    if (
        not np.array_equal(
            reg1_result.caches["R0"].support_zid160[:old_rows],
            reg0_result.caches["R0"].support_zid160,
        )
        or not np.array_equal(
            reg1_result.caches["R1"].support_zid160[:old_rows],
            reg0_result.caches["R1"].support_zid160,
        )
    ):
        raise NextR3RDCETSLRuntimeError("REG1 did not reuse the frozen old-support representations")
    if before != _state_fingerprint(da1_reg0_state):
        raise NextR3RDCETSLRuntimeError("R3 execution mutated the frozen DA1_REG0 state")

    four_state = _four_state_view(reg0_result, reg1_result)
    four_state_receipt = {
        "schema": FOUR_STATE_SCHEMA,
        "state_order": list(FOUR_STATE_IDS),
        "states": {
            state_id: {
                "registration_state": "REG0" if state_id.endswith("REG0") else "REG1",
                "representation": "R0" if state_id.startswith("DA0") else "R1",
                "arm_ids": [arm.arm_id for arm in arms.values()],
                "cache_sha256": next(iter(arms.values())).cache.cache_sha256,
                "query_physical_root_sha256": next(iter(arms.values())).cache.query_physical_root_sha256,
                "seen_new_acc": "N/A" if state_id.endswith("REG0") else "scorer_only",
                "H_old_new": "N/A" if state_id.endswith("REG0") else "scorer_only",
            }
            for state_id, arms in four_state.items()
        },
        "da1_reg0_state_sha256": da1_reg0_state.runtime_receipt_sha256,
        "da1_reg1_state_sha256": da1_reg0_state.runtime_receipt_sha256,
        "da1_reg1_reuses_da1_reg0_state_sha": True,
        "reg1_da_state_refit_calls": 0,
        "reg1_support_policy": "byte_preserve_reg0_prefix_append_new_support_only",
    }
    resource_receipt = {
        "schema": "cvs.stage2.next_r3.rdce_tsl160.resource.v1",
        "rdce_phase1_numeric_payload_bytes": _rdce_asset_numeric_bytes(da1_reg0_state.asset),
        "rdce_row_dynamic_numeric_state_bytes": int(da1_reg0_state.attenuation.nbytes),
        "rdce_query_projection_macs_per_sample": 2 * 3 * Z_DIM,
        "rdce_state_sha256": da1_reg0_state.runtime_receipt_sha256,
        "r0_r1_cache_count_per_registration": 2,
        "heads_per_representation": 3,
        "registration_resources": {
            state: {
                "head_fit_calls": dict(result.receipt["head_fit_calls"]),
                "qknn_bank_sha256": dict(result.receipt["qknn_bank_sha256"]),
                "l_resource_by_representation": {
                    representation: dict(
                        getattr(result.head_fits[f"{representation}L"], "resource_receipt")
                    )
                    for representation in ("R0", "R1")
                },
            }
            for state, result in (("REG0", reg0_result), ("REG1", reg1_result))
        },
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "phase2_optimizer_steps": 0,
        "phase2_backward_calls": 0,
        "dense_matrix_count_by_tsl": 0,
        "spectral_factorization_count_by_tsl": 0,
        "linear_solve_count_by_tsl": 0,
    }
    runtime_receipt = {
        "schema": RUNTIME_SCHEMA,
        "bridge_sha256": bridge.binding_sha256,
        "checkpoint_sha256": bridge.checkpoint_sha256,
        "received_iq_root_sha256": bridge.received_iq_root_sha256,
        "tap_sha256": bridge.tap_sha256,
        "representation_rule": bridge.representation_rule,
        "representation_rule_sha256": bridge.representation_rule_sha256,
        "r1_tsl_prior_semantics": tsl.PRIOR_SEMANTICS,
        "r1_tsl_prior_transported_by_rdce": False,
        "r1_tsl_covariance_claim": False,
        "tsl_representation_context_sha256": {
            "REG0": dict(reg0_result.receipt)["tsl_representation_context_sha256"],
            "REG1": dict(reg1_result.receipt)["tsl_representation_context_sha256"],
        },
        "reg0_r0_cache_sha256": reg0_result.caches["R0"].cache_sha256,
        "reg0_r1_cache_sha256": reg0_result.caches["R1"].cache_sha256,
        "reg1_r0_cache_sha256": reg1_result.caches["R0"].cache_sha256,
        "reg1_r1_cache_sha256": reg1_result.caches["R1"].cache_sha256,
        "same_da_state_sha_across_reg0_reg1": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_input_exists": False,
        "query_role_input_exists": False,
        "class_quota_input_exists": False,
        "true_batch_class_count_input_exists": False,
        "global_reassignment_calls": 0,
        "source_runtime_access": False,
        "clean_runtime_access": False,
        "phase2_optimizer_steps": 0,
        "phase2_backward_calls": 0,
        "all_registered_classes_scored": True,
        "class_label_permutation_equivariant": True,
        "exact_float32_top_tie_fail_closed": True,
    }
    _require_zero_query_receipt(resource_receipt, name="R3 resource")
    _require_zero_query_receipt(runtime_receipt, name="R3 runtime")
    return NextR3RuntimeResult(
        bridge=bridge,
        da1_reg0_state_sha256=da1_reg0_state.runtime_receipt_sha256,
        da1_reg1_state_sha256=da1_reg0_state.runtime_receipt_sha256,
        reg0=reg0_result,
        reg1=reg1_result,
        four_state=four_state,
        four_state_receipt=four_state_receipt,
        resource_receipt=resource_receipt,
        runtime_receipt=runtime_receipt,
    )


__all__ = [
    "ARM_IDS",
    "BRIDGE_SCHEMA",
    "CANONICAL_REPRESENTATION_RULE",
    "FOUR_STATE_IDS",
    "FOUR_STATE_SCHEMA",
    "NextR3Arm",
    "NextR3FeatureCache",
    "NextR3RDCEBridgeBinding",
    "NextR3RDCETSLRuntimeError",
    "NextR3RegistrationInput",
    "NextR3RegistrationResult",
    "NextR3RuntimeResult",
    "REGISTRATION_STATES",
    "RUNTIME_SCHEMA",
    "execute_next_r3_four_state",
]
