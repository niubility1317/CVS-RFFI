"""Thin, label-independent execution closure for one frozen NEXT-R1 row.

The module deliberately owns no checkpoint loader, asset builder, scorer, or
experiment runner.  Callers provide narrow functional-forward and historical
head callbacks; this layer only binds one frozen matrix row, builds the common
R0/R1 signed-pre-ReLU160 caches, and seals Q/F/L predictions before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import stage2_next_r1_assets as assets
from . import stage2_next_r1_fabr as fabr
from . import stage2_next_r1_matrix as matrix
from . import stage2_next_r1_tsl as tsl


RUNTIME_SCHEMA = "cvs.stage2.next_r1.runtime.v1"
PREDICTION_RECEIPT_SCHEMA = "cvs.stage2.next_r1.runtime.prediction_receipt.v1"
RESOURCE_RECEIPT_SCHEMA = "cvs.stage2.next_r1.runtime.resource_receipt.v1"
FORWARD_RECEIPT_SCHEMA = "cvs.stage2.next_r1.runtime.forward_receipt.v1"
SMOKE_RECEIPT_SCHEMA = "cvs.stage2.next_r1.runtime.smoke_receipt.v1"
CHECKPOINT_SMOKE_SCHEMA = "cvs.stage2.next_r1.runtime.real_checkpoint_smoke.v1"
ROW_SEAL_SCHEMA = "cvs.stage2.next_r1.runtime.row_seal.v1"
SEALED_MANIFEST_SCHEMA = "cvs.stage2.next_r1.runtime.sealed_manifest.v1"


class NextR1RuntimeError(ValueError):
    """Raised when one NEXT-R1 row cannot close without a fallback."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NextR1RuntimeError(f"{name} must be a lowercase SHA256")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    return _sha256_bytes(
        json.dumps(
            {"dtype": array.dtype.str, "shape": tuple(int(item) for item in array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        + np.ascontiguousarray(array).tobytes(order="C")
    )


def _physical_root(values: Sequence[str]) -> str:
    return _sha256_bytes("\n".join(values).encode("utf-8"))


def _frozen_logits(value: object, *, rows: int, classes: int, arm_id: str) -> np.ndarray:
    logits = np.asarray(value)
    if (
        logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape != (rows, classes)
        or not np.isfinite(logits).all()
    ):
        raise NextR1RuntimeError(
            f"{arm_id} logits must be finite float32 [{rows},{classes}]"
        )
    frozen = np.array(logits, dtype=np.float32, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True, init=False)
class NextR1VerifiedCheckpointSmoke:
    """Typed closure for a completed real-checkpoint smoke receipt."""

    actual_checkpoint_sha256: str
    representation_rule_sha256: str
    builder_bundle_sha256: str
    row_phase1_seal_sha256: str
    receipt_sha256: str
    completed: bool
    verification_mode: str

    @classmethod
    def _create(
        cls,
        *,
        actual_checkpoint_sha256: str,
        representation_rule_sha256: str,
        builder_bundle_sha256: str,
        row_phase1_seal_sha256: str,
        receipt_sha256: str,
        verification_mode: str,
    ) -> "NextR1VerifiedCheckpointSmoke":
        value = object.__new__(cls)
        object.__setattr__(value, "actual_checkpoint_sha256", actual_checkpoint_sha256)
        object.__setattr__(value, "representation_rule_sha256", representation_rule_sha256)
        object.__setattr__(value, "builder_bundle_sha256", builder_bundle_sha256)
        object.__setattr__(value, "row_phase1_seal_sha256", row_phase1_seal_sha256)
        object.__setattr__(value, "receipt_sha256", receipt_sha256)
        object.__setattr__(value, "completed", True)
        object.__setattr__(value, "verification_mode", verification_mode)
        value.__post_init__()
        return value

    @classmethod
    def for_test_only(
        cls,
        *,
        bundle: assets.NextR1Phase1AssetBundle,
        row_phase1_seal_sha256: str,
    ) -> "NextR1VerifiedCheckpointSmoke":
        """Create an explicit synthetic receipt for focused unit tests only."""

        if type(bundle) is not assets.NextR1Phase1AssetBundle:
            raise NextR1RuntimeError("test-only smoke requires an exact asset bundle")
        row_seal = _require_sha256(
            row_phase1_seal_sha256, name="row_phase1_seal_sha256"
        )
        if row_seal != bundle.receipt["row_phase1_seal_sha256"]:
            raise NextR1RuntimeError("test-only smoke row Phase1 seal drift")
        payload = {
            "schema": CHECKPOINT_SMOKE_SCHEMA,
            "completed": True,
            "actual_checkpoint_sha256": bundle.receipt["checkpoint_sha256"],
            "representation_rule_sha256": bundle.receipt["representation_rule_sha256"],
            "builder_bundle_sha256": bundle.bundle_sha256,
            "row_phase1_seal_sha256": row_seal,
            "verification_mode": "test_only_synthetic",
        }
        return cls._create(
            actual_checkpoint_sha256=payload["actual_checkpoint_sha256"],
            representation_rule_sha256=payload["representation_rule_sha256"],
            builder_bundle_sha256=payload["builder_bundle_sha256"],
            row_phase1_seal_sha256=row_seal,
            receipt_sha256=_canonical_sha256(payload),
            verification_mode="test_only_synthetic",
        )

    def __post_init__(self) -> None:
        for name in (
            "actual_checkpoint_sha256",
            "representation_rule_sha256",
            "builder_bundle_sha256",
            "row_phase1_seal_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))
        if self.completed is not True or self.verification_mode not in (
            "verified_external_receipt",
            "test_only_synthetic",
        ):
            raise NextR1RuntimeError("checkpoint smoke is not a completed verified receipt")


def verify_next_r1_checkpoint_smoke(
    receipt: Mapping[str, Any],
    *,
    bundle: assets.NextR1Phase1AssetBundle,
    row_phase1_seal_sha256: str,
) -> NextR1VerifiedCheckpointSmoke:
    """Validate an external completed smoke and return the production runtime type."""

    if type(bundle) is not assets.NextR1Phase1AssetBundle:
        raise NextR1RuntimeError("checkpoint smoke requires an exact asset bundle")
    if not isinstance(receipt, Mapping):
        raise NextR1RuntimeError("checkpoint smoke receipt must be a mapping")
    payload = dict(receipt)
    observed = payload.pop("checkpoint_smoke_receipt_sha256", None)
    if not isinstance(observed, str) or observed != _canonical_sha256(payload):
        raise NextR1RuntimeError("checkpoint smoke receipt SHA256 drift")
    required = {
        "schema",
        "completed",
        "actual_checkpoint_sha256",
        "representation_rule_sha256",
        "builder_bundle_sha256",
        "row_phase1_seal_sha256",
    }
    if not required.issubset(payload) or payload["schema"] != CHECKPOINT_SMOKE_SCHEMA:
        raise NextR1RuntimeError("checkpoint smoke receipt schema/fields drift")
    row_seal = _require_sha256(
        row_phase1_seal_sha256, name="row_phase1_seal_sha256"
    )
    if (
        payload["completed"] is not True
        or payload["actual_checkpoint_sha256"] != bundle.receipt["checkpoint_sha256"]
        or payload["representation_rule_sha256"]
        != bundle.receipt["representation_rule_sha256"]
        or payload["builder_bundle_sha256"] != bundle.bundle_sha256
        or payload["row_phase1_seal_sha256"] != row_seal
        or row_seal != bundle.receipt["row_phase1_seal_sha256"]
    ):
        raise NextR1RuntimeError("checkpoint smoke/bundle/row binding drift")
    return NextR1VerifiedCheckpointSmoke._create(
        actual_checkpoint_sha256=payload["actual_checkpoint_sha256"],
        representation_rule_sha256=payload["representation_rule_sha256"],
        builder_bundle_sha256=payload["builder_bundle_sha256"],
        row_phase1_seal_sha256=row_seal,
        receipt_sha256=observed,
        verification_mode="verified_external_receipt",
    )


@dataclass(frozen=True, slots=True)
class NextR1FeatureCache:
    """One immutable signed-pre-ReLU160 cache and its physical row order."""

    z160: np.ndarray
    physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = np.asarray(self.z160)
        if (
            rows.dtype != np.float32
            or rows.ndim != 2
            or rows.shape[1] != fabr.Z_DIM
            or rows.shape[0] < 1
            or not np.isfinite(rows).all()
        ):
            raise NextR1RuntimeError("signed-pre-ReLU160 cache must be finite float32 [N,160]")
        norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
            raise NextR1RuntimeError("cache is not the sealed signed-pre-ReLU160 representation")
        ids = tuple(self.physical_ids)
        if (
            len(ids) != rows.shape[0]
            or len(set(ids)) != len(ids)
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise NextR1RuntimeError("cache physical IDs must be nonempty and unique")
        frozen = np.array(rows, dtype=np.float32, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "z160", frozen)
        object.__setattr__(self, "physical_ids", ids)

    @property
    def feature_sha256(self) -> str:
        return _array_sha256(self.z160)

    @property
    def physical_id_root_sha256(self) -> str:
        return _physical_root(self.physical_ids)


@dataclass(frozen=True, slots=True)
class NextR1ArmContext:
    """The only feature view exposed to caller-owned Q/F logical heads."""

    representation_id: str
    support: NextR1FeatureCache
    query: NextR1FeatureCache
    support_labels: tuple[str, ...]
    registered_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.representation_id not in ("R0", "R1"):
            raise NextR1RuntimeError("NEXT-R1 representation must be R0 or R1")
        if type(self.support) is not NextR1FeatureCache or type(self.query) is not NextR1FeatureCache:
            raise NextR1RuntimeError("NEXT-R1 arm context requires exact feature caches")
        classes = tuple(self.registered_classes)
        labels = tuple(self.support_labels)
        if (
            len(classes) != matrix.CLASS_COUNT
            or len(set(classes)) != len(classes)
            or len(labels) != len(self.support.physical_ids)
            or any(not isinstance(item, str) or item not in classes for item in labels)
        ):
            raise NextR1RuntimeError("NEXT-R1 arm context class/support closure drift")
        counts = tuple(labels.count(class_id) for class_id in classes)
        if len(set(counts)) != 1 or counts[0] not in matrix.K_VALUES:
            raise NextR1RuntimeError("NEXT-R1 arm context requires balanced K1 or K5 support")
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_labels", labels)


ArmCallback = Callable[[NextR1ArmContext], np.ndarray]


@dataclass(frozen=True, slots=True)
class NextR1RowSeal:
    """Digest-only row closure suitable for a pre-scoring 84-row manifest."""

    row_id: str
    active_k: int
    held_receiver: str
    held_class: str
    matrix_sha256: str
    binding_sha256: str
    prediction_receipt_sha256: str
    resource_receipt_sha256: str
    forward_receipt_sha256: str
    smoke_receipt_sha256: str
    schema: str = ROW_SEAL_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != ROW_SEAL_SCHEMA
            or not self.row_id
            or self.active_k not in matrix.K_VALUES
            or not self.held_receiver
            or not self.held_class
        ):
            raise NextR1RuntimeError("NEXT-R1 row seal identity drift")
        for name in (
            "matrix_sha256",
            "binding_sha256",
            "prediction_receipt_sha256",
            "resource_receipt_sha256",
            "forward_receipt_sha256",
            "smoke_receipt_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))

    def wire_mapping(self) -> Mapping[str, Any]:
        return _freeze(
            {
                "schema": self.schema,
                "row_id": self.row_id,
                "active_k": self.active_k,
                "held_receiver": self.held_receiver,
                "held_class": self.held_class,
                "matrix_sha256": self.matrix_sha256,
                "binding_sha256": self.binding_sha256,
                "prediction_receipt_sha256": self.prediction_receipt_sha256,
                "resource_receipt_sha256": self.resource_receipt_sha256,
                "forward_receipt_sha256": self.forward_receipt_sha256,
                "smoke_receipt_sha256": self.smoke_receipt_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class NextR1RuntimeResult:
    """Immutable result of one row, still deliberately independent of scoring."""

    row: matrix.NextR1LocoRow
    contexts: Mapping[str, NextR1ArmContext]
    fabr_state: fabr.FABRState
    tsl_fits: Mapping[str, tsl.TSLFit]
    arm_logits: Mapping[str, np.ndarray]
    arm_predictions: Mapping[str, np.ndarray]
    prediction_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]
    forward_receipt: Mapping[str, Any]
    smoke_receipt: Mapping[str, Any]
    row_seal: NextR1RowSeal

    def __post_init__(self) -> None:
        if type(self.row) is not matrix.NextR1LocoRow:
            raise NextR1RuntimeError("runtime result requires an exact NEXT-R1 row")
        if type(self.fabr_state) is not fabr.FABRState:
            raise NextR1RuntimeError("runtime result requires an exact FABR state")
        if type(self.row_seal) is not NextR1RowSeal:
            raise NextR1RuntimeError("runtime result requires a row seal")
        if tuple(self.contexts) != ("R0", "R1") or tuple(self.tsl_fits) != ("R0", "R1"):
            raise NextR1RuntimeError("runtime result representation closure drift")
        if tuple(self.arm_logits) != matrix.ARM_IDS or tuple(self.arm_predictions) != matrix.ARM_IDS:
            raise NextR1RuntimeError("runtime result arm closure drift")
        object.__setattr__(self, "contexts", MappingProxyType(dict(self.contexts)))
        object.__setattr__(self, "tsl_fits", MappingProxyType(dict(self.tsl_fits)))
        object.__setattr__(self, "arm_logits", MappingProxyType(dict(self.arm_logits)))
        object.__setattr__(self, "arm_predictions", MappingProxyType(dict(self.arm_predictions)))
        object.__setattr__(self, "prediction_receipt", _freeze(self.prediction_receipt))
        object.__setattr__(self, "resource_receipt", _freeze(self.resource_receipt))
        object.__setattr__(self, "forward_receipt", _freeze(self.forward_receipt))
        object.__setattr__(self, "smoke_receipt", _freeze(self.smoke_receipt))


def _matching_plan_row(plan: Mapping[str, Any], row: matrix.NextR1LocoRow) -> Mapping[str, Any]:
    matched = [item for item in plan["rows"] if item.get("row_id") == row.row_id]
    if len(matched) != 1:
        raise NextR1RuntimeError("NEXT-R1 row is absent from the frozen matrix")
    expected = matched[0]
    if (
        expected.get("candidate_id") != row.candidate_id
        or expected.get("held_receiver") != row.held_receiver
        or expected.get("held_class") != row.held_class
        or expected.get("active_k") != row.active_k
        or tuple(expected.get("retained_classes", ())) != row.retained_classes
        or tuple(expected.get("registered_classes", ())) != row.registered_classes
    ):
        raise NextR1RuntimeError("NEXT-R1 frozen matrix row drift")
    return expected


def _validate_row_inputs(
    *,
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    row: matrix.NextR1LocoRow,
    bundle: assets.NextR1Phase1AssetBundle,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if type(row) is not matrix.NextR1LocoRow:
        raise NextR1RuntimeError("NEXT-R1 runtime accepts an exact row value")
    if type(bundle) is not assets.NextR1Phase1AssetBundle:
        raise NextR1RuntimeError("NEXT-R1 runtime accepts an exact builder asset bundle")
    try:
        frozen_plan = matrix.validate_next_r1_plan(plan)
        frozen_binding = matrix.validate_next_r1_binding(binding)
    except matrix.NextR1MatrixError as error:
        raise NextR1RuntimeError("NEXT-R1 matrix/binding validation failed") from error
    _matching_plan_row(frozen_plan, row)
    expected_row_id = (
        frozen_binding["k1_row_id"] if row.active_k == 1 else frozen_binding["k5_row_id"]
    )
    if (
        expected_row_id != row.row_id
        or frozen_binding["held_receiver"] != row.held_receiver
        or frozen_binding["held_class"] != row.held_class
        or tuple(frozen_binding["registered_classes"]) != row.registered_classes
        or frozen_binding["phase1_seal_sha256"]
        != bundle.receipt["row_phase1_seal_sha256"]
    ):
        raise NextR1RuntimeError("NEXT-R1 row/Phase1 seal binding drift")
    return frozen_plan, frozen_binding


def _support_binding(
    binding: Mapping[str, Any], row: matrix.NextR1LocoRow
) -> Mapping[str, tuple[str, ...]]:
    name = "k1_support_ids_by_class" if row.active_k == 1 else "k5_support_ids_by_class"
    values = binding[name]
    if not isinstance(values, Mapping):
        raise NextR1RuntimeError("NEXT-R1 support binding is malformed")
    result = {class_id: tuple(values[class_id]) for class_id in row.registered_classes}
    if any(len(result[class_id]) != row.active_k for class_id in row.registered_classes):
        raise NextR1RuntimeError("NEXT-R1 support binding K drift")
    return MappingProxyType(result)


def _assert_support_cache_binding(
    cache: NextR1FeatureCache,
    labels: Sequence[str],
    expected: Mapping[str, tuple[str, ...]],
    classes: tuple[str, ...],
) -> None:
    if len(labels) != len(cache.physical_ids):
        raise NextR1RuntimeError("NEXT-R1 support labels/cache row drift")
    actual: dict[str, list[str]] = {class_id: [] for class_id in classes}
    for label, physical_id in zip(labels, cache.physical_ids, strict=True):
        if label not in actual:
            raise NextR1RuntimeError("NEXT-R1 support label falls outside registry")
        actual[label].append(physical_id)
    for class_id in classes:
        if set(actual[class_id]) != set(expected[class_id]) or len(actual[class_id]) != len(expected[class_id]):
            raise NextR1RuntimeError("NEXT-R1 support physical binding drift")


def _expected_query_ids(binding: Mapping[str, Any], classes: tuple[str, ...]) -> frozenset[str]:
    values = binding["query_ids_by_class"]
    if not isinstance(values, Mapping):
        raise NextR1RuntimeError("NEXT-R1 query physical binding is malformed")
    ids = tuple(item for class_id in classes for item in values[class_id])
    if len(ids) != matrix.QUERY_COUNT or len(set(ids)) != len(ids):
        raise NextR1RuntimeError("NEXT-R1 query physical binding count drift")
    return frozenset(ids)


def _assert_query_cache_binding(
    cache: NextR1FeatureCache, expected_ids: frozenset[str]
) -> None:
    if len(cache.physical_ids) != len(expected_ids) or frozenset(cache.physical_ids) != expected_ids:
        raise NextR1RuntimeError("NEXT-R1 query physical binding drift")


def _cached_support_forward(
    callback: fabr.ForwardWithCoeff,
    token: Any,
) -> tuple[fabr.ForwardWithCoeff, dict[bytes, fabr.FABRForwardBatch], list[np.ndarray]]:
    if not callable(callback):
        raise NextR1RuntimeError("NEXT-R1 support functional forward must be callable")
    captured: dict[bytes, fabr.FABRForwardBatch] = {}
    calls: list[np.ndarray] = []

    def wrapped(value: Any, coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        if value is not token:
            raise NextR1RuntimeError("NEXT-R1 support token substitution is forbidden")
        batch = callback(value, coefficient)
        if type(batch) is not fabr.FABRForwardBatch:
            raise NextR1RuntimeError("NEXT-R1 functional forward must return FABRForwardBatch")
        coeff = np.asarray(coefficient)
        key = np.ascontiguousarray(coeff, dtype=np.float32).tobytes(order="C")
        existing = captured.get(key)
        if existing is not None and (
            existing.physical_ids != batch.physical_ids
            or not np.array_equal(existing.features, batch.features)
        ):
            raise NextR1RuntimeError("same support coefficient produced cache drift")
        captured.setdefault(key, batch)
        calls.append(np.array(coeff, dtype=np.float32, copy=True, order="C"))
        return batch

    return wrapped, captured, calls


def _query_cache(
    callback: fabr.ForwardWithCoeff,
    token: Any,
    coefficient: np.ndarray,
    *,
    name: str,
) -> NextR1FeatureCache:
    if not callable(callback):
        raise NextR1RuntimeError("NEXT-R1 query functional forward must be callable")
    try:
        batch = callback(token, coefficient)
    except Exception as error:
        raise NextR1RuntimeError(f"{name} query functional forward failed") from error
    if type(batch) is not fabr.FABRForwardBatch:
        raise NextR1RuntimeError("NEXT-R1 query functional forward must return FABRForwardBatch")
    return NextR1FeatureCache(batch.features, batch.physical_ids)


def _invoke_arm_callback(
    callback: ArmCallback, context: NextR1ArmContext, *, arm_id: str
) -> np.ndarray:
    if not callable(callback):
        raise NextR1RuntimeError(f"{arm_id} callback must be callable")
    try:
        return callback(context)
    except Exception as error:
        raise NextR1RuntimeError(f"{arm_id} callback failed") from error


def _receipt_with_sha256(value: Mapping[str, Any], *, field: str) -> Mapping[str, Any]:
    payload = dict(value)
    payload[field] = _canonical_sha256(payload)
    return _freeze(payload)


def _validate_embedded_receipt(
    receipt: Mapping[str, Any], *, field: str, row_id: str
) -> str:
    if not isinstance(receipt, Mapping):
        raise NextR1RuntimeError("NEXT-R1 result receipt must be a mapping")
    payload = dict(receipt)
    observed = payload.pop(field, None)
    if payload.get("row_id") != row_id:
        raise NextR1RuntimeError("NEXT-R1 result receipt row ID drift")
    if not isinstance(observed, str) or observed != _canonical_sha256(payload):
        raise NextR1RuntimeError("NEXT-R1 result receipt SHA256 drift")
    return observed


def _execute_next_r1_row_impl(
    *,
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    row: matrix.NextR1LocoRow,
    bundle: assets.NextR1Phase1AssetBundle,
    verified_checkpoint_smoke: NextR1VerifiedCheckpointSmoke,
    support_token: Any,
    query_token: Any,
    support_labels: Sequence[str],
    support_forward_with_coeff: fabr.ForwardWithCoeff,
    query_forward_with_coeff: fabr.ForwardWithCoeff,
    q_callback: ArmCallback,
    frozen_f_callback: ArmCallback,
    frozen_f_archive_sha256: str,
    allow_test_smoke: bool,
) -> NextR1RuntimeResult:
    """Execute exactly one frozen row and seal all six logical arms.

    The support forward is the only callback reachable while FABR fits.  The
    query forward and logical-head callbacks receive caches only after the
    support-only state is closed.  A typed, completed checkpoint smoke must be
    bound before any row artifact can be sealed.
    """

    frozen_plan, frozen_binding = _validate_row_inputs(
        plan=plan, binding=binding, row=row, bundle=bundle
    )
    if type(verified_checkpoint_smoke) is not NextR1VerifiedCheckpointSmoke:
        raise NextR1RuntimeError("NEXT-R1 runtime requires a typed verified checkpoint smoke")
    if (
        verified_checkpoint_smoke.verification_mode == "test_only_synthetic"
        and allow_test_smoke is not True
    ):
        raise NextR1RuntimeError("production runtime rejects test-only checkpoint smoke")
    if (
        verified_checkpoint_smoke.completed is not True
        or verified_checkpoint_smoke.actual_checkpoint_sha256
        != bundle.receipt["checkpoint_sha256"]
        or verified_checkpoint_smoke.representation_rule_sha256
        != bundle.receipt["representation_rule_sha256"]
        or verified_checkpoint_smoke.builder_bundle_sha256 != bundle.bundle_sha256
        or verified_checkpoint_smoke.row_phase1_seal_sha256
        != frozen_binding["phase1_seal_sha256"]
    ):
        raise NextR1RuntimeError("verified checkpoint smoke/bundle/row binding drift")
    archive_sha256 = _require_sha256(
        frozen_f_archive_sha256, name="frozen_f_archive_sha256"
    )
    if not callable(frozen_f_callback):
        raise NextR1RuntimeError("NEXT-R1 requires a frozen historical-F callback")
    labels = tuple(support_labels)
    expected_support = _support_binding(frozen_binding, row)
    expected_rows = row.active_k * len(row.registered_classes)
    if len(labels) != expected_rows:
        raise NextR1RuntimeError("NEXT-R1 support label count does not match frozen K")
    if any(not isinstance(item, str) or item not in row.registered_classes for item in labels):
        raise NextR1RuntimeError("NEXT-R1 support labels fall outside the registry")
    if any(labels.count(class_id) != row.active_k for class_id in row.registered_classes):
        raise NextR1RuntimeError("NEXT-R1 support labels are not balanced at frozen K")

    wrapped_support, captured_support, support_calls = _cached_support_forward(
        support_forward_with_coeff, support_token
    )
    try:
        fabr_state = fabr.fit_fabr_support(
            bundle.fabr_asset,
            support_token,
            labels,
            row.registered_classes,
            wrapped_support,
            support_physical_ids=tuple(
                item for class_id in row.registered_classes for item in expected_support[class_id]
            ),
            runtime_binding=fabr.FABRRuntimeBinding(
                actual_checkpoint_sha256=verified_checkpoint_smoke.actual_checkpoint_sha256,
                phase1_seal_sha256=bundle.receipt["row_phase1_seal_sha256"],
                representation_rule_sha256=bundle.receipt["representation_rule_sha256"],
            ),
        )
    except Exception as error:
        raise NextR1RuntimeError("NEXT-R1 support-only FABR fit failed") from error
    if (
        fabr_state.active_k != row.active_k
        or fabr_state.registered_classes != row.registered_classes
        or fabr_state.asset_sha256 != fabr.fabr_asset_sha256(bundle.fabr_asset)
    ):
        raise NextR1RuntimeError("NEXT-R1 FABR state binding drift")
    zero = np.zeros(fabr.RANK, dtype=np.float32)
    zero_key = zero.tobytes(order="C")
    final_coeff = fabr_state.coeff_float32
    final_key = final_coeff.tobytes(order="C")
    if zero_key not in captured_support or final_key not in captured_support:
        raise NextR1RuntimeError("NEXT-R1 FABR cache did not retain R0/R1 support")
    if len(support_calls) != 6:
        raise NextR1RuntimeError("NEXT-R1 FABR support forward count drift")

    r0_support = NextR1FeatureCache(
        captured_support[zero_key].features, captured_support[zero_key].physical_ids
    )
    r1_support = NextR1FeatureCache(
        captured_support[final_key].features, captured_support[final_key].physical_ids
    )
    _assert_support_cache_binding(r0_support, labels, expected_support, row.registered_classes)
    _assert_support_cache_binding(r1_support, labels, expected_support, row.registered_classes)
    if r0_support.physical_ids != r1_support.physical_ids:
        raise NextR1RuntimeError("NEXT-R1 R0/R1 support cache physical order drift")

    r0_query = _query_cache(
        query_forward_with_coeff, query_token, zero, name="R0"
    )
    r1_query = _query_cache(
        query_forward_with_coeff, query_token, final_coeff, name="R1"
    )
    expected_query = _expected_query_ids(frozen_binding, row.registered_classes)
    _assert_query_cache_binding(r0_query, expected_query)
    _assert_query_cache_binding(r1_query, expected_query)
    if r0_query.physical_ids != r1_query.physical_ids:
        raise NextR1RuntimeError("NEXT-R1 R0/R1 query cache physical order drift")

    contexts: Mapping[str, NextR1ArmContext] = MappingProxyType(
        {
            "R0": NextR1ArmContext("R0", r0_support, r0_query, labels, row.registered_classes),
            "R1": NextR1ArmContext("R1", r1_support, r1_query, labels, row.registered_classes),
        }
    )
    class_count = len(row.registered_classes)
    query_rows = len(r0_query.physical_ids)
    q_logits = {
        representation: _frozen_logits(
            _invoke_arm_callback(q_callback, context, arm_id=f"{representation}Q"),
            rows=query_rows,
            classes=class_count,
            arm_id=f"{representation}Q",
        )
        for representation, context in contexts.items()
    }
    tsl_runtime_binding = tsl.TSLRuntimeBinding(
        checkpoint_sha256=verified_checkpoint_smoke.actual_checkpoint_sha256,
        representation_rule_sha256=bundle.receipt["representation_rule_sha256"],
        phase1_seal_sha256=bundle.receipt["row_phase1_seal_sha256"],
    )
    fitter = tsl.TailSafeLite(bundle.tsl_prior, runtime_binding=tsl_runtime_binding)
    tsl_fits: dict[str, tsl.TSLFit] = {}
    f_callback_invocations = 0
    if row.active_k == 1:
        f_logits: dict[str, np.ndarray] = {}
        l_logits: dict[str, np.ndarray] = {}
        for representation, context in contexts.items():
            fit = fitter.fit(context.support.z160, context.support_labels, context.registered_classes)
            if type(fit.state) is not tsl.TSLK1AliasState:
                raise NextR1RuntimeError("NEXT-R1 K1 TSL must be an exact Q alias")
            tsl_fits[representation] = fit
            f_value = tsl.alias_qknn_logits(
                fit.state, q_logits[representation], runtime_binding=tsl_runtime_binding
            )
            l_value = tsl.alias_qknn_logits(
                fit.state, q_logits[representation], runtime_binding=tsl_runtime_binding
            )
            if f_value is not q_logits[representation] or l_value is not q_logits[representation]:
                raise NextR1RuntimeError("NEXT-R1 K1 Q/F/L alias object drift")
            fabr.require_exact_logit_alias(q_logits[representation], f_value)
            fabr.require_exact_logit_alias(q_logits[representation], l_value)
            f_logits[representation] = f_value
            l_logits[representation] = l_value
    else:
        f_logits = {}
        l_logits = {}
        for representation, context in contexts.items():
            f_callback_invocations += 1
            f_logits[representation] = _frozen_logits(
                _invoke_arm_callback(
                    frozen_f_callback, context, arm_id=f"{representation}F"
                ),
                rows=query_rows,
                classes=class_count,
                arm_id=f"{representation}F",
            )
            fit = fitter.fit(context.support.z160, context.support_labels, context.registered_classes)
            if type(fit.state) is not tsl.TSLAffineHeadState:
                raise NextR1RuntimeError("NEXT-R1 K5 TSL must produce an affine state")
            tsl_fits[representation] = fit
            l_logits[representation] = _frozen_logits(
                tsl.score_affine(
                    fit.state, context.query.z160, runtime_binding=tsl_runtime_binding
                ).logits,
                rows=query_rows,
                classes=class_count,
                arm_id=f"{representation}L",
            )

    arm_logits: Mapping[str, np.ndarray] = MappingProxyType(
        {
            "R0Q": q_logits["R0"],
            "R0F": f_logits["R0"],
            "R0L": l_logits["R0"],
            "R1Q": q_logits["R1"],
            "R1F": f_logits["R1"],
            "R1L": l_logits["R1"],
        }
    )
    arm_predictions: dict[str, np.ndarray] = {}
    for arm_id in matrix.ARM_IDS:
        logits = arm_logits[arm_id]
        try:
            tsl.require_unique_float32_top(logits)
            arm_predictions[arm_id] = fabr.strict_top1_predictions(logits)
        except (fabr.FABRError, tsl.TailSafeLiteError) as error:
            raise NextR1RuntimeError(f"{arm_id} exact float32 top-tie closure failed") from error

    cache_receipt = {
        representation: {
            "support_rows": len(context.support.physical_ids),
            "query_rows": len(context.query.physical_ids),
            "support_z160_sha256": context.support.feature_sha256,
            "query_z160_sha256": context.query.feature_sha256,
            "support_physical_id_root_sha256": context.support.physical_id_root_sha256,
            "query_physical_id_root_sha256": context.query.physical_id_root_sha256,
            "signed_pre_relu160": True,
        }
        for representation, context in contexts.items()
    }
    forward_receipt = _receipt_with_sha256(
        {
            "schema": FORWARD_RECEIPT_SCHEMA,
            "row_id": row.row_id,
            "support_base_forward_calls": fabr_state.resource_receipt.base_support_forward_calls,
            "support_perturbation_forward_calls": fabr_state.resource_receipt.perturbation_support_forward_calls,
            "support_final_forward_calls": fabr_state.resource_receipt.final_support_forward_calls,
            "support_forward_calls_observed": len(support_calls),
            "r0_query_forward_calls": 1,
            "r1_query_forward_calls": 1,
            "support_only_fabr_fit": True,
            "r0_r1_cache_representation": "same_signed_pre_relu160_rule",
            "cache": cache_receipt,
        },
        field="forward_receipt_sha256",
    )
    resource_receipt = _receipt_with_sha256(
        {
            "schema": RESOURCE_RECEIPT_SCHEMA,
            "row_id": row.row_id,
            "active_k": row.active_k,
            "registered_class_count": class_count,
            "arm_ids": list(matrix.ARM_IDS),
            "common_r0_cache_shared": True,
            "q_callback_invocations": 2,
            "frozen_f_callback_invocations": f_callback_invocations,
            "fabr_resource": {
                "active_k": fabr_state.resource_receipt.active_k,
                "registered_class_count": fabr_state.resource_receipt.registered_class_count,
                "asset_numeric_payload_bytes": fabr_state.resource_receipt.asset_numeric_payload_bytes,
                "dynamic_numeric_state_bytes": fabr_state.resource_receipt.dynamic_numeric_state_bytes,
                "support_fit_mac_equivalent": fabr_state.resource_receipt.support_fit_mac_equivalent,
                "support_base_forward_calls": fabr_state.resource_receipt.base_support_forward_calls,
                "support_perturbation_forward_calls": fabr_state.resource_receipt.perturbation_support_forward_calls,
                "support_final_forward_calls": fabr_state.resource_receipt.final_support_forward_calls,
                "protocol_closed": fabr_state.resource_receipt.protocol_closed,
            },
            "tsl_resource_by_representation": {
                representation: dict(fit.resource_receipt)
                for representation, fit in tsl_fits.items()
            },
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        },
        field="resource_receipt_sha256",
    )
    prediction_receipt = _receipt_with_sha256(
        {
            "schema": PREDICTION_RECEIPT_SCHEMA,
            "row_id": row.row_id,
            "active_k": row.active_k,
            "registered_classes": list(row.registered_classes),
            "matrix_sha256": frozen_plan["matrix_sha256"],
            "binding_sha256": frozen_binding["binding_sha256"],
            "fabr_asset_sha256": fabr.fabr_asset_sha256(bundle.fabr_asset),
            "builder_bundle_sha256": bundle.bundle_sha256,
            "verified_checkpoint_smoke_sha256": verified_checkpoint_smoke.receipt_sha256,
            "fabr_support_root_sha256": fabr_state.support_root_sha256,
            "frozen_f_archive_sha256": archive_sha256,
            "arm_logit_sha256": {
                arm_id: _array_sha256(arm_logits[arm_id]) for arm_id in matrix.ARM_IDS
            },
            "arm_prediction_sha256": {
                arm_id: _array_sha256(arm_predictions[arm_id]) for arm_id in matrix.ARM_IDS
            },
            "all_registered_classes_scored": True,
            "independent_per_sample": True,
            "common_r0_cache_shared": True,
            "k1_qfl_exact_alias": row.active_k == 1,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        },
        field="prediction_receipt_sha256",
    )
    smoke_receipt = _receipt_with_sha256(
        {
            "schema": SMOKE_RECEIPT_SCHEMA,
            "row_id": row.row_id,
            "runtime_schema": RUNTIME_SCHEMA,
            "same_checkpoint_sha256": True,
            "checkpoint_sha256": bundle.receipt["checkpoint_sha256"],
            "phase1_seal_sha256": bundle.receipt["row_phase1_seal_sha256"],
            "representation_rule_sha256": bundle.receipt["representation_rule_sha256"],
            "builder_bundle_sha256": bundle.bundle_sha256,
            "verified_checkpoint_smoke_sha256": verified_checkpoint_smoke.receipt_sha256,
            "checkpoint_smoke_verification_mode": verified_checkpoint_smoke.verification_mode,
            "narrow_forward_callbacks_exercised": True,
            "actual_checkpoint_archive_smoke_required": True,
            "actual_checkpoint_archive_smoke_completed": True,
            "support_only_fabr_fit": True,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        },
        field="smoke_receipt_sha256",
    )
    row_seal = NextR1RowSeal(
        row_id=row.row_id,
        active_k=row.active_k,
        held_receiver=row.held_receiver,
        held_class=row.held_class,
        matrix_sha256=frozen_plan["matrix_sha256"],
        binding_sha256=frozen_binding["binding_sha256"],
        prediction_receipt_sha256=prediction_receipt["prediction_receipt_sha256"],
        resource_receipt_sha256=resource_receipt["resource_receipt_sha256"],
        forward_receipt_sha256=forward_receipt["forward_receipt_sha256"],
        smoke_receipt_sha256=smoke_receipt["smoke_receipt_sha256"],
    )
    return NextR1RuntimeResult(
        row=row,
        contexts=contexts,
        fabr_state=fabr_state,
        tsl_fits=MappingProxyType(tsl_fits),
        arm_logits=arm_logits,
        arm_predictions=MappingProxyType(arm_predictions),
        prediction_receipt=prediction_receipt,
        resource_receipt=resource_receipt,
        forward_receipt=forward_receipt,
        smoke_receipt=smoke_receipt,
        row_seal=row_seal,
    )


def execute_next_r1_row(
    *,
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    row: matrix.NextR1LocoRow,
    bundle: assets.NextR1Phase1AssetBundle,
    verified_checkpoint_smoke: NextR1VerifiedCheckpointSmoke,
    support_token: Any,
    query_token: Any,
    support_labels: Sequence[str],
    support_forward_with_coeff: fabr.ForwardWithCoeff,
    query_forward_with_coeff: fabr.ForwardWithCoeff,
    q_callback: ArmCallback,
    frozen_f_callback: ArmCallback,
    frozen_f_archive_sha256: str,
) -> NextR1RuntimeResult:
    """Execute one production row using a verified external checkpoint smoke."""

    return _execute_next_r1_row_impl(
        plan=plan,
        binding=binding,
        row=row,
        bundle=bundle,
        verified_checkpoint_smoke=verified_checkpoint_smoke,
        support_token=support_token,
        query_token=query_token,
        support_labels=support_labels,
        support_forward_with_coeff=support_forward_with_coeff,
        query_forward_with_coeff=query_forward_with_coeff,
        q_callback=q_callback,
        frozen_f_callback=frozen_f_callback,
        frozen_f_archive_sha256=frozen_f_archive_sha256,
        allow_test_smoke=False,
    )


def build_next_r1_sealed_manifest(
    plan: Mapping[str, Any], results: Sequence[NextR1RuntimeResult]
) -> Mapping[str, Any]:
    """Prove all 84 frozen rows are sealed before a separate scorer opens them."""

    try:
        frozen_plan = matrix.validate_next_r1_plan(plan)
    except matrix.NextR1MatrixError as error:
        raise NextR1RuntimeError("NEXT-R1 manifest requires a valid frozen plan") from error
    if isinstance(results, (str, bytes)) or len(results) != matrix.ROW_COUNT:
        raise NextR1RuntimeError("NEXT-R1 sealed manifest requires all 84 runtime results")
    expected = {item["row_id"]: item for item in frozen_plan["rows"]}
    observed: dict[str, NextR1RuntimeResult] = {}
    receipt_hashes: dict[str, set[str]] = {
        "prediction": set(),
        "resource": set(),
        "forward": set(),
        "smoke": set(),
    }
    for result in results:
        if type(result) is not NextR1RuntimeResult:
            raise NextR1RuntimeError("NEXT-R1 manifest requires exact runtime results")
        row_id = result.row.row_id
        if row_id in observed or row_id not in expected:
            raise NextR1RuntimeError("NEXT-R1 manifest row identity drift")
        row = expected[row_id]
        if (
            result.row.active_k != row["active_k"]
            or result.row.held_receiver != row["held_receiver"]
            or result.row.held_class != row["held_class"]
            or tuple(row["retained_classes"]) != result.row.retained_classes
            or tuple(row["registered_classes"]) != result.row.registered_classes
        ):
            raise NextR1RuntimeError("NEXT-R1 manifest row binding drift")
        receipt_values = {
            "prediction": _validate_embedded_receipt(
                result.prediction_receipt,
                field="prediction_receipt_sha256",
                row_id=row_id,
            ),
            "resource": _validate_embedded_receipt(
                result.resource_receipt,
                field="resource_receipt_sha256",
                row_id=row_id,
            ),
            "forward": _validate_embedded_receipt(
                result.forward_receipt,
                field="forward_receipt_sha256",
                row_id=row_id,
            ),
            "smoke": _validate_embedded_receipt(
                result.smoke_receipt,
                field="smoke_receipt_sha256",
                row_id=row_id,
            ),
        }
        for kind, value in receipt_values.items():
            if value in receipt_hashes[kind]:
                raise NextR1RuntimeError(
                    f"NEXT-R1 manifest repeats a row-specific {kind} receipt SHA256"
                )
            receipt_hashes[kind].add(value)
        seal = result.row_seal
        if (
            seal.row_id != row_id
            or seal.matrix_sha256 != frozen_plan["matrix_sha256"]
            or seal.active_k != result.row.active_k
            or seal.held_receiver != result.row.held_receiver
            or seal.held_class != result.row.held_class
            or seal.prediction_receipt_sha256 != receipt_values["prediction"]
            or seal.resource_receipt_sha256 != receipt_values["resource"]
            or seal.forward_receipt_sha256 != receipt_values["forward"]
            or seal.smoke_receipt_sha256 != receipt_values["smoke"]
            or result.prediction_receipt.get("matrix_sha256") != frozen_plan["matrix_sha256"]
            or result.prediction_receipt.get("binding_sha256") != seal.binding_sha256
            or result.smoke_receipt.get("actual_checkpoint_archive_smoke_completed") is not True
        ):
            raise NextR1RuntimeError("NEXT-R1 result/row-seal receipt closure drift")
        observed[row_id] = result
    if set(observed) != set(expected):
        raise NextR1RuntimeError("NEXT-R1 sealed manifest coverage drift")
    payload = {
        "schema": SEALED_MANIFEST_SCHEMA,
        "matrix_sha256": frozen_plan["matrix_sha256"],
        "candidate_id": matrix.CANDIDATE_ID,
        "row_count": matrix.ROW_COUNT,
        "all_rows_sealed": True,
        "sealed_before_scoring": True,
        "rows": [
            dict(observed[item["row_id"]].row_seal.wire_mapping())
            for item in frozen_plan["rows"]
        ],
    }
    payload["sealed_manifest_sha256"] = _canonical_sha256(payload)
    return _freeze(payload)


__all__ = [
    "ArmCallback",
    "CHECKPOINT_SMOKE_SCHEMA",
    "FORWARD_RECEIPT_SCHEMA",
    "NextR1ArmContext",
    "NextR1FeatureCache",
    "NextR1RowSeal",
    "NextR1RuntimeError",
    "NextR1RuntimeResult",
    "NextR1VerifiedCheckpointSmoke",
    "PREDICTION_RECEIPT_SCHEMA",
    "RESOURCE_RECEIPT_SCHEMA",
    "ROW_SEAL_SCHEMA",
    "RUNTIME_SCHEMA",
    "SEALED_MANIFEST_SCHEMA",
    "SMOKE_RECEIPT_SCHEMA",
    "build_next_r1_sealed_manifest",
    "execute_next_r1_row",
    "verify_next_r1_checkpoint_smoke",
]
