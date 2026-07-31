"""D105-CBRC-MB4 shared, support-only Stage2 domain adaptation.

This module deliberately contains no classifier, query truth surface, or
ground multi-prototype path.  It consumes the sealed Phase1 MetaBias4
aggregate bundle and legal current-row support only, estimates one common
four-dimensional coefficient, and exposes the canonical ``z_id`` transform
for a separately owned target-support qKNN head.

The first D105 revision keeps ``ground_old_multiprototype_enabled=false``;
therefore ``gamma_ground`` is exactly one.  Low-information support and a
zero deployed coefficient are legal, prediction-complete data outcomes rather
than technical failures.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.rxid_metabias4_bundle import (
    AMAX_FP16,
    CODE_DIM,
    DOMAIN_DIM,
    LAMBDA0_FP16,
    RXIDMetaBias4Bundle,
    RXIDMetaBias4BundleError,
    Z_DIM,
    serialize_rxid_metabias4_bundle,
)


SCHEMA = "cvs.phase2.d105.cbrc.state.v1"
BUNDLE_HANDLE_SCHEMA = "cvs.phase1.d105.cbrc.bundle_handle.v1"
FIT_AUDIT_SCHEMA = "cvs.phase2.d105.cbrc.fit_audit.v1"
RESOURCE_AUDIT_SCHEMA = "cvs.phase2.d105.cbrc.resource_audit.v1"
WIRE_MAGIC = b"CVSD105C\x00\x01"

ALLOWED_STAGES = ("S_B", "S_C")
ALLOWED_K = (1, 5, 10)
IRLS_STEPS = 4
EPS = 1.0e-12
MAX_STATE_BYTES = 80 * 1024
MAX_POST_BACKBONE_MAC_PER_QUERY = 262_144

ACTIVE = "ACTIVE"
FALLBACK_LEGAL_DATA = "FALLBACK_LEGAL_DATA"


class D105CBRCError(ValueError):
    """Raised when the typed D105 protocol or numeric closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        convert(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise D105CBRCError(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze audit structures retained by typed runtime state."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(member) for key, member in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(member) for member in value)
    if isinstance(value, np.ndarray):
        return _readonly(value, value.dtype)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deep_copy_public(value: Any) -> Any:
    """Return a detached mutable audit copy without leaking state aliases."""

    if isinstance(value, Mapping):
        return {
            str(key): _deep_copy_public(member) for key, member in value.items()
        }
    if isinstance(value, tuple):
        return [_deep_copy_public(member) for member in value]
    if isinstance(value, list):
        return [_deep_copy_public(member) for member in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_receipt(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _roundtrip_rows(value: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Use D103's reviewed INT8/FP16 row conversion boundary explicitly."""

    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise D105CBRCError("row roundtrip requires finite rank-2 values")
    if normalize:
        norms = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms <= EPS):
            raise D105CBRCError("normalized roundtrip contains a zero row")
        rows = np.asarray(rows.astype(np.float64) / norms, dtype=np.float32)
    maximum = np.max(np.abs(rows.astype(np.float64)), axis=1)
    if np.any(maximum <= EPS):
        raise D105CBRCError("row roundtrip contains a zero row")
    scales = maximum / 127.0
    codes = np.clip(
        np.rint(rows.astype(np.float64) / scales[:, None]), -127.0, 127.0
    ).astype(np.int8)
    decoded = np.asarray(
        codes.astype(np.float32)
        * scales.astype(np.float16).astype(np.float32)[:, None],
        dtype=np.float32,
    )
    if normalize:
        norms = np.linalg.norm(decoded.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms <= EPS):
            raise D105CBRCError("decoded normalized roundtrip contains a zero row")
        decoded = np.asarray(decoded.astype(np.float64) / norms, dtype=np.float32)
    return decoded


def _receipt_root(bundle: RXIDMetaBias4Bundle) -> str:
    """Hash the immutable receipt handles, without reopening Phase1 provenance."""

    return _sha256_bytes(
        _canonical_bytes(
            {
                "training_receipt_sha256": bundle.training_receipt_sha256,
                "nested_receipt_sha256": bundle.nested_receipt_sha256,
                "tx_probe_receipt_sha256": bundle.tx_probe_receipt_sha256,
                "aggregation_receipt_sha256": bundle.aggregation_receipt_sha256,
                "quantization_receipt_sha256": bundle.quantization_receipt_sha256,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class D105CBRCBundleHandle:
    """The bounded runtime identity check for a sealed Phase1 aggregate bundle."""

    validated_bundle_id_sha256: str
    validator_receipt_sha256: str
    expected_content_root_sha256: str
    checkpoint_sha256: str
    runtime_sha256: str
    method_lock_sha256: str
    receipt_root_sha256: str
    target_rows: int = 0
    schema: str = BUNDLE_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BUNDLE_HANDLE_SCHEMA or self.target_rows != 0:
            raise D105CBRCError("D105 bundle handle schema/target_rows drift")
        for field in (
            "validated_bundle_id_sha256",
            "validator_receipt_sha256",
            "expected_content_root_sha256",
            "checkpoint_sha256",
            "runtime_sha256",
            "method_lock_sha256",
            "receipt_root_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.validated_bundle_id_sha256 == self.expected_content_root_sha256:
            raise D105CBRCError(
                "validated bundle_id must be independent from the content root"
            )
        expected_validator_receipt = compute_d105_bundle_validator_receipt(
            validated_bundle_id_sha256=self.validated_bundle_id_sha256,
            expected_content_root_sha256=self.expected_content_root_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            runtime_sha256=self.runtime_sha256,
            method_lock_sha256=self.method_lock_sha256,
            receipt_root_sha256=self.receipt_root_sha256,
        )
        if self.validator_receipt_sha256 != expected_validator_receipt:
            raise D105CBRCError(
                "validator receipt does not bind the sealed bundle manifest"
            )


def compute_d105_bundle_validator_receipt(
    *,
    validated_bundle_id_sha256: str,
    expected_content_root_sha256: str,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    receipt_root_sha256: str,
) -> str:
    """Canonical receipt emitted by the independent Phase1 bundle validator."""

    payload = {
        "schema": BUNDLE_HANDLE_SCHEMA,
        "validated_bundle_id_sha256": _require_sha256(
            validated_bundle_id_sha256, "validated_bundle_id_sha256"
        ),
        "expected_content_root_sha256": _require_sha256(
            expected_content_root_sha256, "expected_content_root_sha256"
        ),
        "checkpoint_sha256": _require_sha256(
            checkpoint_sha256, "checkpoint_sha256"
        ),
        "runtime_sha256": _require_sha256(runtime_sha256, "runtime_sha256"),
        "method_lock_sha256": _require_sha256(
            method_lock_sha256, "method_lock_sha256"
        ),
        "receipt_root_sha256": _require_sha256(
            receipt_root_sha256, "receipt_root_sha256"
        ),
        "target_rows": 0,
    }
    return _sha256_bytes(_canonical_bytes(payload))


def compute_d105_bundle_receipt_root(bundle: RXIDMetaBias4Bundle) -> str:
    """Expose the receipt-root field used by the external validator seal."""

    _revalidate_bundle_payload(bundle)
    return _receipt_root(bundle)


def _revalidate_bundle_payload(bundle: RXIDMetaBias4Bundle) -> str:
    """Recompute the complete payload root without copying the bundle arrays."""

    if type(bundle) is not RXIDMetaBias4Bundle:
        raise D105CBRCError("D105 requires an exact RXIDMetaBias4Bundle")
    try:
        actual_root = bundle._content_root()
    except (AttributeError, TypeError, ValueError) as error:
        raise D105CBRCError("D105 complete bundle payload validation failed") from error
    stored_root = _require_sha256(
        bundle.content_root_sha256, "actual bundle content root"
    )
    if actual_root != stored_root:
        raise D105CBRCError("D105 complete bundle payload validation failed")
    return actual_root


def _serialize_revalidated_bundle(bundle: RXIDMetaBias4Bundle) -> bytes:
    _revalidate_bundle_payload(bundle)
    try:
        return serialize_rxid_metabias4_bundle(bundle)
    except RXIDMetaBias4BundleError as error:
        raise D105CBRCError("D105 complete bundle serialization failed") from error


def make_d105_cbrc_bundle_handle(
    bundle: RXIDMetaBias4Bundle,
    *,
    validated_bundle_id_sha256: str,
    validator_receipt_sha256: str,
    expected_content_root_sha256: str,
) -> D105CBRCBundleHandle:
    """Bind an external Phase1 validator seal to one exact bundle payload.

    None of the three required validator values is inferred from the runtime
    bundle.  In particular, the independently sealed ``bundle_id`` must not be
    the payload content root.
    """

    _revalidate_bundle_payload(bundle)
    validated_id = _require_sha256(
        validated_bundle_id_sha256, "validated_bundle_id_sha256"
    )
    validator_receipt = _require_sha256(
        validator_receipt_sha256, "validator_receipt_sha256"
    )
    expected_root = _require_sha256(
        expected_content_root_sha256, "expected_content_root_sha256"
    )
    if expected_root != bundle.content_root_sha256:
        raise D105CBRCError("external expected content root does not match payload")
    return D105CBRCBundleHandle(
        validated_bundle_id_sha256=validated_id,
        validator_receipt_sha256=validator_receipt,
        expected_content_root_sha256=expected_root,
        checkpoint_sha256=bundle.checkpoint_sha256,
        runtime_sha256=bundle.runtime_sha256,
        method_lock_sha256=bundle.method_lock_sha256,
        receipt_root_sha256=_receipt_root(bundle),
    )


def _validate_bundle_handle(
    bundle: RXIDMetaBias4Bundle, handle: D105CBRCBundleHandle
) -> str:
    if type(handle) is not D105CBRCBundleHandle:
        raise D105CBRCError("D105 requires an exact typed bundle handle")
    actual_root = _revalidate_bundle_payload(bundle)
    actual = {
        "expected_content_root_sha256": bundle.content_root_sha256,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "runtime_sha256": bundle.runtime_sha256,
        "method_lock_sha256": bundle.method_lock_sha256,
        "receipt_root_sha256": _receipt_root(bundle),
    }
    expected = {key: getattr(handle, key) for key in actual}
    if actual != expected:
        raise D105CBRCError(
            "D105 external bundle seal/content root/payload identity drift"
        )
    return actual_root


def _canonical_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise D105CBRCError(f"{name} must be non-empty, non-blank, and unique")
    return result


def validate_d105_physical_split(
    support_physical_ids: Sequence[str], query_physical_ids: Sequence[str]
) -> None:
    """Validate non-truth physical-ID split closure without accepting query data."""

    support = _canonical_ids(support_physical_ids, "support physical IDs")
    query = _canonical_ids(query_physical_ids, "query physical IDs")
    overlap = sorted(set(support).intersection(query))
    if overlap:
        raise D105CBRCError("support/query physical IDs overlap")


def _validate_lifecycle(
    stage: str,
    registered_classes: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if stage not in ALLOWED_STAGES:
        raise D105CBRCError("stage must be S_B or S_C")
    registry = _canonical_ids(registered_classes, "registered classes")
    old = _canonical_ids(old_classes, "old classes")
    new = tuple(str(value) for value in new_classes)
    if len(set(new)) != len(new) or any(not value for value in new):
        raise D105CBRCError("new classes must be unique and non-blank")
    if set(old).intersection(new):
        raise D105CBRCError("old/new lifecycle classes overlap")
    if stage == "S_B":
        if new or set(registry) != set(old) or len(old) < 2:
            raise D105CBRCError("S_B requires exactly at least two old classes")
    else:
        if len(old) < 2 or len(new) < 2 or set(registry) != set(old).union(new):
            raise D105CBRCError(
                "S_C requires at least two old and two new registered classes"
            )
    return registry, tuple(sorted(old)), tuple(sorted(new))


def _validate_support_inputs(
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    registry: tuple[str, ...],
    *,
    active_k: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    if type(active_k) is not int or active_k not in ALLOWED_K:
        raise D105CBRCError("active_k must be one of 1, 5, 10")
    zdom = np.asarray(support_zdom)
    if (
        zdom.dtype != np.float32
        or zdom.ndim != 2
        or zdom.shape[1] != Z_DIM
        or len(zdom) < 1
        or not np.isfinite(zdom).all()
    ):
        raise D105CBRCError("support z_dom contract drift")
    labels = tuple(str(value) for value in support_labels)
    identifiers = _canonical_ids(support_physical_ids, "support physical IDs")
    if len(labels) != len(zdom) or len(identifiers) != len(zdom):
        raise D105CBRCError("support arrays/labels/physical IDs must align")
    if any(label not in registry for label in labels) or any(
        labels.count(label) != active_k for label in registry
    ):
        raise D105CBRCError("support labels require balanced K-shot registry coverage")
    order = np.asarray(sorted(range(len(identifiers)), key=lambda index: identifiers[index]))
    return (
        np.ascontiguousarray(zdom[order], dtype=np.float32),
        tuple(labels[index] for index in order),
        tuple(identifiers[index] for index in order),
        order,
    )


def _validate_pre_relu(support_pre_relu: np.ndarray, expected_rows: int) -> None:
    pre = np.asarray(support_pre_relu)
    if (
        pre.dtype != np.float32
        or pre.ndim != 2
        or pre.shape != (expected_rows, Z_DIM)
        or not np.isfinite(pre).all()
    ):
        raise D105CBRCError("support pre-ReLU z_id alignment contract drift")


def compute_d105_support_binding_root(
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    *,
    active_k: int,
    stage: str,
) -> str:
    """Bind both support feature views, labels, physical IDs, and lifecycle roles."""

    registry, old, new = _validate_lifecycle(
        stage, registered_classes, old_classes, new_classes
    )
    zdom, labels, physical_ids, order = _validate_support_inputs(
        support_zdom,
        support_labels,
        support_physical_ids,
        registry,
        active_k=active_k,
    )
    _validate_pre_relu(support_pre_relu, len(zdom))
    pre = np.ascontiguousarray(np.asarray(support_pre_relu)[order], dtype=np.float32)
    lifecycle = {
        label: ("old" if label in set(old) else "new") for label in sorted(registry)
    }
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": "cvs.phase2.d105.cbrc.support_binding.v1",
                "protocol_schema": "p2_min_v1",
                "stage": stage,
                "active_k": active_k,
                "support_pre_relu": _array_receipt(pre),
                "support_zdom": _array_receipt(zdom),
                "support_labels": labels,
                "support_physical_ids": physical_ids,
                "registered_classes": registry,
                "lifecycle_role_by_label": lifecycle,
            }
        )
    )


def _canonical_zid(pre_relu: np.ndarray) -> np.ndarray:
    value = np.asarray(pre_relu)
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or value.shape[1] != Z_DIM
        or len(value) < 1
        or not np.isfinite(value).all()
    ):
        raise D105CBRCError("pre-ReLU z_id transform contract drift")
    relu = np.maximum(value.astype(np.float64), 0.0)
    norms = np.linalg.norm(relu, axis=1, keepdims=True)
    if np.any(norms <= EPS) or not np.isfinite(norms).all():
        raise D105CBRCError("canonical z_id contains a zero ReLU row")
    return np.ascontiguousarray(relu / norms, dtype=np.float32)


def _decoded_b(bundle: RXIDMetaBias4Bundle) -> np.ndarray:
    return _roundtrip_rows(bundle.decode_b()).astype(np.float64)


def transform_d105_canonical(
    bundle: RXIDMetaBias4Bundle,
    bundle_handle: D105CBRCBundleHandle,
    coefficient_fp16: np.ndarray,
    pre_relu: np.ndarray,
) -> np.ndarray:
    """Apply the frozen common non-linear canonical transform read-only."""

    _validate_bundle_handle(bundle, bundle_handle)
    return _transform_validated_canonical(bundle, coefficient_fp16, pre_relu)


def _transform_validated_canonical(
    bundle: RXIDMetaBias4Bundle,
    coefficient_fp16: np.ndarray,
    pre_relu: np.ndarray,
) -> np.ndarray:
    """Transform after the caller has validated the complete sealed payload."""

    coefficient = np.asarray(coefficient_fp16)
    if (
        coefficient.dtype != np.float16
        or coefficient.shape != (CODE_DIM,)
        or not np.isfinite(coefficient).all()
    ):
        raise D105CBRCError("D105 coefficient contract drift")
    pre = np.asarray(pre_relu)
    if (
        pre.dtype != np.float32
        or pre.ndim != 2
        or pre.shape[1] != Z_DIM
        or len(pre) < 1
        or not np.isfinite(pre).all()
    ):
        raise D105CBRCError("pre-ReLU z_id transform contract drift")
    if np.all(coefficient == np.float16(0.0)):
        return _canonical_zid(pre)
    bias = _decoded_b(bundle) @ coefficient.astype(np.float64)
    shifted = pre.astype(np.float64) + bias[None, :]
    return _canonical_zid(np.ascontiguousarray(shifted, dtype=np.float32))


def _support_domain_moments(
    bundle: RXIDMetaBias4Bundle, support_zdom: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], int]:
    """Compute support-only MetaBias4 posterior moments in float64 transiently."""

    zdom = np.asarray(support_zdom)
    u = _roundtrip_rows(bundle.decode_u()).astype(np.float64)
    representation = zdom.astype(np.float64) @ u.T
    norms = np.linalg.norm(representation, axis=1, keepdims=True)
    if np.any(norms <= EPS) or not np.isfinite(norms).all():
        raise D105CBRCError("encoded support domain norm drift")
    representation /= norms
    bank_g = _roundtrip_rows(bundle.decode_bank_g(), normalize=True).astype(np.float64)
    similarity = np.clip(representation @ bank_g.T, -1.0, 1.0)
    scaled = similarity / float(bundle.temperature_fp16)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= np.sum(weights, axis=1, keepdims=True)
    sigma = np.asarray(bundle.decode_bank_sigma(), dtype=np.float16).astype(np.float64)
    coverage = np.sum(
        weights * np.exp(-(1.0 - similarity) / (sigma[None, :] ** 2)), axis=1
    )
    bank_precision = np.asarray(
        bundle.decode_bank_precision(), dtype=np.float16
    ).astype(np.float64)
    bank_t = np.asarray(bundle.decode_bank_t(), dtype=np.float16).astype(np.float64)
    precision = coverage[:, None] * (weights @ bank_precision)
    means = weights @ bank_t
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(coverage).all()
        or np.any(coverage <= 0.0)
        or np.any(coverage > 1.0 + 1.0e-12)
        or not np.isfinite(precision).all()
        or np.any(precision <= 0.0)
        or not np.isfinite(means).all()
    ):
        raise D105CBRCError("support domain encoding closure failed")
    count = int(len(zdom))
    cells = int(bundle.bank_count)
    mac = {
        "domain_projection": count * Z_DIM * DOMAIN_DIM,
        "bank_similarity": count * cells * DOMAIN_DIM,
        "posterior_mean": count * cells * CODE_DIM,
        "posterior_precision": count * cells * CODE_DIM,
        "coverage_weight": count * cells,
    }
    temporary_bytes = int(
        u.nbytes
        + representation.nbytes
        + bank_g.nbytes
        + similarity.nbytes
        + scaled.nbytes
        + weights.nbytes
        + sigma.nbytes
        + bank_precision.nbytes
        + bank_t.nbytes
        + coverage.nbytes
        + precision.nbytes
        + means.nbytes
    )
    return means, precision, coverage, mac, temporary_bytes


def _class_statistics(
    means: np.ndarray,
    precision: np.ndarray,
    labels: tuple[str, ...],
    class_groups: tuple[tuple[str, ...], ...],
    *,
    active_k: int,
    lambda0: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in class_groups:
        for label in group:
            indices = np.asarray([index for index, value in enumerate(labels) if value == label])
            if len(indices) != active_k:
                raise D105CBRCError("balanced support count drift during class statistics")
            if active_k == 1:
                moment = means[int(indices[0])]
                class_precision = lambda0.copy()
                class_b = lambda0 * moment
            else:
                class_precision = np.mean(precision[indices], axis=0)
                class_b = np.mean(precision[indices] * means[indices], axis=0)
            if (
                class_precision.shape != (CODE_DIM,)
                or class_b.shape != (CODE_DIM,)
                or not np.isfinite(class_precision).all()
                or not np.isfinite(class_b).all()
                or np.any(class_precision <= 0.0)
            ):
                raise D105CBRCError("class sufficient-statistic closure failed")
            result[label] = (class_precision, class_b)
    return result


def _project_a(
    value: np.ndarray, lambda0: np.ndarray, limits: np.ndarray, radius: float
) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64)
    if candidate.shape != (CODE_DIM,) or not np.isfinite(candidate).all():
        raise D105CBRCError("D105 coefficient solve produced non-finite values")
    boxed = np.clip(candidate, -limits, limits)
    quadratic = float(np.sum(lambda0 * boxed * boxed))
    if not math.isfinite(quadratic) or quadratic < 0.0:
        raise D105CBRCError("D105 ellipsoid projection drift")
    if quadratic > radius * radius:
        boxed = radius * boxed / math.sqrt(quadratic)
    if (
        not np.isfinite(boxed).all()
        or np.any(np.abs(boxed) > limits + 1.0e-10)
        or float(np.sum(lambda0 * boxed * boxed)) > radius * radius + 1.0e-10
    ):
        raise D105CBRCError("D105 coefficient projection closure failed")
    return boxed


def _task_weights(
    groups: tuple[tuple[str, ...], ...],
    masses: tuple[float, ...],
    raw: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return task-mass-preserving class weights, with a uniform fallback."""

    result: dict[str, float] = {}
    for group, mass in zip(groups, masses):
        if raw is None:
            values = np.ones(len(group), dtype=np.float64)
        else:
            values = np.asarray([float(raw[label]) for label in group], dtype=np.float64)
        total = float(np.sum(values))
        if not np.isfinite(values).all() or total <= EPS:
            values = np.ones(len(group), dtype=np.float64)
            total = float(len(group))
        for label, value in zip(group, values):
            result[label] = float(mass * value / total)
    if not np.isfinite(np.asarray(list(result.values()), dtype=np.float64)).all():
        raise D105CBRCError("task-balanced IRLS weights are non-finite")
    return result


def _task_median(
    values: Mapping[str, float],
    groups: tuple[tuple[str, ...], ...],
    masses: tuple[float, ...],
) -> float:
    medians = []
    for group in groups:
        item = float(np.median(np.asarray([values[label] for label in group])))
        if not math.isfinite(item):
            raise D105CBRCError("task median is non-finite")
        medians.append(item)
    result = float(sum(mass * item for mass, item in zip(masses, medians)))
    if not math.isfinite(result):
        raise D105CBRCError("task-balanced median is non-finite")
    return result


def _solve_irls(
    statistics: Mapping[str, tuple[np.ndarray, np.ndarray]],
    groups: tuple[tuple[str, ...], ...],
    masses: tuple[float, ...],
    lambda0: np.ndarray,
    limits: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the fixed task-balanced four-step diagonal Huber-IRLS solve."""

    labels = tuple(label for group in groups for label in group)
    if not labels:
        raise D105CBRCError("IRLS requires at least one class")
    initial_weights = _task_weights(groups, masses)

    def solve_with(weights: Mapping[str, float]) -> np.ndarray:
        system = lambda0.copy()
        rhs = np.zeros(CODE_DIM, dtype=np.float64)
        for label in labels:
            class_precision, class_b = statistics[label]
            system += float(weights[label]) * class_precision
            rhs += float(weights[label]) * class_b
        if np.any(system <= 0.0) or not np.isfinite(system).all() or not np.isfinite(rhs).all():
            raise D105CBRCError("D105 diagonal system is not SPD")
        return _project_a(rhs / system, lambda0, limits, radius)

    coefficient = solve_with(initial_weights)
    residuals = {
        label: float(
            np.linalg.norm(
                np.sqrt(statistics[label][0])
                * (coefficient - statistics[label][1] / statistics[label][0])
            )
        )
        for label in labels
    }
    kappa = max(EPS, _task_median(residuals, groups, masses))
    uniform_fallback_groups = 0
    task_weight_history: list[dict[str, float]] = []
    for _ in range(IRLS_STEPS):
        raw = {
            label: min(1.0, kappa / (residuals[label] + EPS)) for label in labels
        }
        for group in groups:
            group_values = np.asarray([raw[label] for label in group], dtype=np.float64)
            if not np.isfinite(group_values).all() or float(np.sum(group_values)) <= EPS:
                uniform_fallback_groups += 1
        weights = _task_weights(groups, masses, raw)
        task_weight_history.append(
            {
                "old": float(sum(weights[label] for label in groups[0])),
                "new": (
                    0.0
                    if len(groups) == 1
                    else float(sum(weights[label] for label in groups[1]))
                ),
            }
        )
        coefficient = solve_with(weights)
        residuals = {
            label: float(
                np.linalg.norm(
                    np.sqrt(statistics[label][0])
                    * (coefficient - statistics[label][1] / statistics[label][0])
                )
            )
            for label in labels
        }
        if not np.isfinite(np.asarray(list(residuals.values()), dtype=np.float64)).all():
            raise D105CBRCError("D105 IRLS residual closure failed")
    return coefficient, {
        "kappa": float(kappa),
        "final_task_weights": {label: float(weights[label]) for label in labels},
        "task_weight_sums_per_iteration": task_weight_history,
        "uniform_fallback_group_count": int(uniform_fallback_groups),
        "final_residual_task_median": float(_task_median(residuals, groups, masses)),
    }


def _loo_groups(
    groups: tuple[tuple[str, ...], ...], excluded: str
) -> tuple[tuple[str, ...], ...]:
    result = tuple(
        tuple(label for label in group if label != excluded) for group in groups
    )
    if any(not group for group in result):
        raise D105CBRCError("LOO would drop an entire lifecycle task")
    return result


def _solver_mac(class_count: int) -> int:
    """Exact logical scalar-MAC accounting used by this implementation receipt."""

    if class_count < 1:
        raise D105CBRCError("solver MAC requires a positive class count")
    # Initial weighted Lambda/b accumulation: 2*D*C.  Per IRLS iteration:
    # D*C residual products plus 2*D*C weighted accumulations.  Division,
    # sqrt, exp, comparisons, and reductions are reported outside the MAC unit.
    return int(2 * CODE_DIM * class_count + IRLS_STEPS * 3 * CODE_DIM * class_count)


@dataclass(frozen=True, slots=True)
class D105CBRCSupportSolution:
    """Transient support-only CBRC coefficient solution for review and state fit."""

    coefficient_fp16: np.ndarray
    rho: float
    status: str
    fit_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient_fp16)
        audit = dict(self.fit_audit)
        if (
            coefficient.dtype != np.float16
            or coefficient.shape != (CODE_DIM,)
            or not np.isfinite(coefficient).all()
            or self.status not in (ACTIVE, FALLBACK_LEGAL_DATA)
            or not math.isfinite(float(self.rho))
            or not 0.0 <= float(self.rho) <= 1.0
            or audit.get("support_only") is not True
            or audit.get("query_rows_used_for_fit") != 0
        ):
            raise D105CBRCError("D105 support solution contract drift")
        if self.status == FALLBACK_LEGAL_DATA and np.any(coefficient != 0.0):
            raise D105CBRCError("legal D105 fallback must carry an exact zero coefficient")
        object.__setattr__(self, "coefficient_fp16", _readonly(coefficient, np.float16))
        object.__setattr__(self, "fit_audit", _deep_freeze(audit))


def solve_d105_cbrc_support(
    bundle: RXIDMetaBias4Bundle,
    bundle_handle: D105CBRCBundleHandle,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    *,
    active_k: int,
    stage: str,
) -> D105CBRCSupportSolution:
    """Fit the shared coefficient from support only; no query surface exists."""

    _validate_bundle_handle(bundle, bundle_handle)
    registry, old, new = _validate_lifecycle(
        stage, registered_classes, old_classes, new_classes
    )
    zdom, labels, physical_ids, _ = _validate_support_inputs(
        support_zdom,
        support_labels,
        support_physical_ids,
        registry,
        active_k=active_k,
    )
    groups = (old,) if stage == "S_B" else (old, new)
    masses = (1.0,) if stage == "S_B" else (0.5, 0.5)
    lambda0 = np.asarray(bundle.lambda0_fp16, dtype=np.float64)
    limits = np.asarray(bundle.amax_fp16, dtype=np.float64)
    radius = float(bundle.radius_fp16)
    if (
        lambda0.shape != (CODE_DIM,)
        or limits.shape != (CODE_DIM,)
        or not np.isfinite(lambda0).all()
        or np.any(lambda0 <= 0.0)
        or not np.isfinite(limits).all()
        or np.any(limits <= 0.0)
        or not math.isfinite(radius)
        or radius <= 0.0
    ):
        raise D105CBRCError("D105 sealed prior/constraint drift")

    means, precision, coverage, encoding_mac, domain_temporary_bytes = (
        _support_domain_moments(bundle, zdom)
    )
    statistics = _class_statistics(
        means,
        precision,
        labels,
        groups,
        active_k=active_k,
        lambda0=lambda0,
    )
    coefficient, main_audit = _solve_irls(
        statistics, groups, masses, lambda0, limits, radius
    )
    b_matrix = _decoded_b(bundle)
    shift = b_matrix @ coefficient
    loo_similarity: dict[str, float] = {}
    loo_solver_mac = 0
    for label in tuple(item for group in groups for item in group):
        remaining_groups = _loo_groups(groups, label)
        remaining_stats = {key: value for key, value in statistics.items() if key != label}
        left_out, _ = _solve_irls(
            remaining_stats, remaining_groups, masses, lambda0, limits, radius
        )
        left_shift = b_matrix @ left_out
        denominator = float(np.linalg.norm(shift) + np.linalg.norm(left_shift) + EPS)
        similarity = 1.0 - float(np.linalg.norm(shift - left_shift)) / denominator
        loo_similarity[label] = float(np.clip(similarity, 0.0, 1.0))
        loo_solver_mac += _solver_mac(len(statistics) - 1)
    rho = float(_task_median(loo_similarity, groups, masses))
    gamma_ground = 1.0
    deployed = np.asarray(rho * gamma_ground * coefficient, dtype=np.float16)
    if not np.isfinite(deployed).all():
        raise D105CBRCError("D105 deployed coefficient became non-finite")
    status = ACTIVE if np.any(deployed != np.float16(0.0)) else FALLBACK_LEGAL_DATA
    class_mode = "unified_lambda0" if active_k == 1 else "support_posterior_mean"
    class_targets = np.stack(
        [
            statistics[label][1] / statistics[label][0]
            for group in groups
            for label in group
        ]
    )
    centered_targets = class_targets - np.mean(class_targets, axis=0, keepdims=True)
    data_information_rank = int(np.linalg.matrix_rank(centered_targets, tol=EPS))
    support_statistic_mac = int(len(zdom) * CODE_DIM * 2)
    main_solver_mac = _solver_mac(len(statistics))
    b_projection_mac = int((len(statistics) + 1) * Z_DIM * CODE_DIM)
    statistics_bytes = int(
        sum(first.nbytes + second.nbytes for first, second in statistics.values())
    )
    posterior_output_bytes = int(means.nbytes + precision.nbytes + coverage.nbytes)
    class_target_bytes = int(class_targets.nbytes + centered_targets.nbytes)
    # One solve keeps 4D coefficient/system/RHS/projection/residual scratch
    # together with one C-vector each for residual, robust q, and normalized
    # task weight.  The LOO path additionally keeps the current 4D left-out
    # coefficient.  This is a deterministic numeric-workspace upper bound,
    # not a wall-clock allocator measurement.
    solver_vector_bytes = int(
        (
            9 * CODE_DIM
            + 3 * len(statistics)
        )
        * np.dtype(np.float64).itemsize
    )
    solver_phase_bytes = int(
        posterior_output_bytes
        + b_matrix.nbytes
        + statistics_bytes
        + class_target_bytes
        + solver_vector_bytes
        + shift.nbytes
        + (Z_DIM * np.dtype(np.float64).itemsize)
    )
    # Phase A owns the domain-posterior dense arrays.  Phase B begins after
    # those local work arrays are released; it retains only the returned
    # posterior outputs, class statistics, B, one main shift, and one current
    # LOO shift.  The peak is therefore max(A, B), not their sum.
    temporary_bytes = int(max(domain_temporary_bytes, solver_phase_bytes))
    lifecycle_root = _sha256_bytes(
        _canonical_bytes(
            {
                "stage": stage,
                "old": old,
                "new": new,
                "role_by_label": {
                    label: ("old" if label in set(old) else "new")
                    for label in sorted(registry)
                },
            }
        )
    )
    audit: dict[str, Any] = {
        "schema": FIT_AUDIT_SCHEMA,
        "stage": stage,
        "active_k": int(active_k),
        "status": status,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_truth_read": False,
        "query_state_updates": 0,
        "all_registered_classes_compete": True,
        "registry_class_count": len(registry),
        "old_class_count": len(old),
        "new_class_count": len(new),
        "task_mass_old": 1.0 if stage == "S_B" else 0.5,
        "task_mass_new": 0.0 if stage == "S_B" else 0.5,
        "task_weight_sums": {
            "old": float(sum(main_audit["final_task_weights"][label] for label in old)),
            "new": float(sum(main_audit["final_task_weights"].get(label, 0.0) for label in new)),
        },
        "k1_precision_mode": class_mode,
        "k1_unified_lambda0": bool(active_k == 1),
        "irls_steps": IRLS_STEPS,
        "irls_kappa": float(main_audit["kappa"]),
        "irls_uniform_group_fallback_count": int(
            main_audit["uniform_fallback_group_count"]
        ),
        "irls_final_residual_task_median": float(
            main_audit["final_residual_task_median"]
        ),
        "irls_task_weight_sums_per_iteration": main_audit[
            "task_weight_sums_per_iteration"
        ],
        "rho_loo": rho,
        "rho_old_median": float(
            np.median(np.asarray([loo_similarity[label] for label in old]))
        ),
        "rho_new_median": (
            None
            if stage == "S_B"
            else float(np.median(np.asarray([loo_similarity[label] for label in new])))
        ),
        "gamma_ground": gamma_ground,
        "ground_old_multiprototype_enabled": False,
        "legal_low_information_fallback": status == FALLBACK_LEGAL_DATA,
        "data_information_rank": data_information_rank,
        "data_information_rank_basis": "centered_class_coefficient_targets_without_lambda0",
        "lifecycle_role_binding_root_sha256": lifecycle_root,
        "coverage_min": float(np.min(coverage)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_max": float(np.max(coverage)),
        "support_physical_ids_unique": len(set(physical_ids)) == len(physical_ids),
        "loo_excluded_class_count": len(loo_similarity),
        "loo_self_exclusion": True,
        "support_encoding_algorithmic_mac": int(sum(encoding_mac.values())),
        "support_statistic_algorithmic_mac": support_statistic_mac,
        "irls_main_algorithmic_mac": main_solver_mac,
        "irls_loo_algorithmic_mac": int(loo_solver_mac),
        "b_projection_algorithmic_mac": b_projection_mac,
        "b_projection_count": len(statistics) + 1,
        "mac_accounting_convention": "logical scalar multiply-accumulates only; divisions/sqrt/exp/reductions separate",
        "temporary_fit_bytes": temporary_bytes,
        "temporary_solver_vector_bytes": solver_vector_bytes,
        "temporary_class_target_bytes": class_target_bytes,
        "temporary_domain_phase_bytes": int(domain_temporary_bytes),
        "temporary_solver_phase_bytes": solver_phase_bytes,
        "temporary_fit_accounting": "deterministic numeric upper bound=max(domain-posterior phase, solver phase); solver includes posterior outputs, class stats, class/centered targets, B, main/current LOO shifts, and named 4D/C-vector solve workspaces",
        "persistent_decoded_learning_arrays": False,
        "old_new_query_role_access": False,
        "class_quota_access": False,
        "target25_authorized": False,
    }
    return D105CBRCSupportSolution(
        coefficient_fp16=deployed,
        rho=rho,
        status=status,
        fit_audit=audit,
    )


def _state_receipt(
    bundle_handle: D105CBRCBundleHandle,
    stage: str,
    active_k: int,
    coefficient_fp16: np.ndarray,
    support_receipt_sha256: str,
    status: str,
    fit_audit: Mapping[str, Any],
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "schema": SCHEMA,
                "bundle_handle": {
                    "validated_bundle_id_sha256": (
                        bundle_handle.validated_bundle_id_sha256
                    ),
                    "validator_receipt_sha256": (
                        bundle_handle.validator_receipt_sha256
                    ),
                    "expected_content_root_sha256": (
                        bundle_handle.expected_content_root_sha256
                    ),
                    "checkpoint_sha256": bundle_handle.checkpoint_sha256,
                    "runtime_sha256": bundle_handle.runtime_sha256,
                    "method_lock_sha256": bundle_handle.method_lock_sha256,
                    "receipt_root_sha256": bundle_handle.receipt_root_sha256,
                    "target_rows": 0,
                },
                "stage": stage,
                "active_k": active_k,
                "coefficient_fp16": _array_receipt(coefficient_fp16),
                "support_receipt_sha256": support_receipt_sha256,
                "status": status,
                "fit_audit": fit_audit,
                "query_state_updates": 0,
                "query_rows_used_for_fit": 0,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class D105CBRCState:
    """Immutable runtime state: sealed Phase1 bundle plus one FP16 4D code."""

    bundle: RXIDMetaBias4Bundle
    bundle_handle: D105CBRCBundleHandle
    stage: str
    active_k: int
    coefficient_fp16: np.ndarray
    support_receipt_sha256: str
    status: str
    fit_audit: Mapping[str, Any]
    state_receipt_sha256: str
    query_state_updates: int = 0
    query_rows_used_for_fit: int = 0
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient_fp16)
        audit = dict(self.fit_audit)
        if (
            self.schema != SCHEMA
            or self.stage not in ALLOWED_STAGES
            or type(self.active_k) is not int
            or self.active_k not in ALLOWED_K
            or coefficient.dtype != np.float16
            or coefficient.shape != (CODE_DIM,)
            or not np.isfinite(coefficient).all()
            or self.status not in (ACTIVE, FALLBACK_LEGAL_DATA)
            or type(self.query_state_updates) is not int
            or self.query_state_updates != 0
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
            or audit.get("status") != self.status
            or audit.get("query_rows_used_for_fit") != 0
        ):
            raise D105CBRCError("D105 state lifecycle/type drift")
        if self.status == FALLBACK_LEGAL_DATA and np.any(coefficient != 0.0):
            raise D105CBRCError("legal fallback state must carry an exact zero code")
        _validate_bundle_handle(self.bundle, self.bundle_handle)
        _require_sha256(self.support_receipt_sha256, "support_receipt_sha256")
        _require_sha256(self.state_receipt_sha256, "state_receipt_sha256")
        expected = _state_receipt(
            self.bundle_handle,
            self.stage,
            self.active_k,
            coefficient,
            self.support_receipt_sha256,
            self.status,
            audit,
        )
        if expected != self.state_receipt_sha256:
            raise D105CBRCError("D105 state receipt verification failed")
        object.__setattr__(self, "coefficient_fp16", _readonly(coefficient, np.float16))
        object.__setattr__(self, "fit_audit", _deep_freeze(audit))

    @property
    def active(self) -> bool:
        return self.status == ACTIVE


def _validate_state_integrity(state: D105CBRCState) -> None:
    """Fail closed if any sealed payload, code, audit, or receipt has drifted."""

    if type(state) is not D105CBRCState:
        raise D105CBRCError("D105 runtime requires an exact typed state")
    _validate_bundle_handle(state.bundle, state.bundle_handle)
    coefficient = np.asarray(state.coefficient_fp16)
    if (
        coefficient.dtype != np.float16
        or coefficient.shape != (CODE_DIM,)
        or not np.isfinite(coefficient).all()
        or (
            state.status == FALLBACK_LEGAL_DATA
            and np.any(coefficient != np.float16(0.0))
        )
    ):
        raise D105CBRCError("D105 runtime coefficient integrity drift")
    expected = _state_receipt(
        state.bundle_handle,
        state.stage,
        state.active_k,
        coefficient,
        state.support_receipt_sha256,
        state.status,
        state.fit_audit,
    )
    if expected != state.state_receipt_sha256:
        raise D105CBRCError("D105 runtime state receipt verification failed")


def fit_d105_cbrc_state(
    bundle: RXIDMetaBias4Bundle,
    bundle_handle: D105CBRCBundleHandle,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    support_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    old_classes: Sequence[str],
    new_classes: Sequence[str],
    *,
    active_k: int,
    stage: str,
    support_receipt_sha256: str,
) -> D105CBRCState:
    """Create an immutable D105 support-only state for later read-only queries."""

    supplied_support_root = _require_sha256(
        support_receipt_sha256, "support_receipt_sha256"
    )
    computed_support_root = compute_d105_support_binding_root(
        support_pre_relu,
        support_zdom,
        support_labels,
        support_physical_ids,
        registered_classes,
        old_classes,
        new_classes,
        active_k=active_k,
        stage=stage,
    )
    if supplied_support_root != computed_support_root:
        raise D105CBRCError(
            "support receipt does not bind pre-ReLU/z_dom/labels/physical IDs"
        )
    solution = solve_d105_cbrc_support(
        bundle,
        bundle_handle,
        support_zdom,
        support_labels,
        support_physical_ids,
        registered_classes,
        old_classes,
        new_classes,
        active_k=active_k,
        stage=stage,
    )
    audit = dict(solution.fit_audit)
    audit["support_pre_relu_alignment_checked"] = True
    audit["support_pre_relu_zdom_physical_id_bound"] = True
    audit["support_binding_root_sha256"] = computed_support_root
    receipt = _state_receipt(
        bundle_handle,
        stage,
        active_k,
        solution.coefficient_fp16,
        computed_support_root,
        solution.status,
        audit,
    )
    return D105CBRCState(
        bundle=bundle,
        bundle_handle=bundle_handle,
        stage=stage,
        active_k=active_k,
        coefficient_fp16=solution.coefficient_fp16,
        support_receipt_sha256=computed_support_root,
        status=solution.status,
        fit_audit=audit,
        state_receipt_sha256=receipt,
    )


def transform_d105_cbrc(state: D105CBRCState, query_pre_relu: np.ndarray) -> np.ndarray:
    """Transform arbitrary support/query batches with zero state update."""

    _validate_state_integrity(state)
    return _transform_validated_canonical(
        state.bundle, state.coefficient_fp16, query_pre_relu
    )


def serialize_d105_cbrc_state(state: D105CBRCState) -> bytes:
    """Serialize only the sealed Phase1 bundle, FP16 code, and immutable receipts."""

    _validate_state_integrity(state)
    bundle_wire = _serialize_revalidated_bundle(state.bundle)
    header = _canonical_bytes(
        {
            "schema": state.schema,
            "state_receipt_sha256": state.state_receipt_sha256,
            "support_receipt_sha256": state.support_receipt_sha256,
            "status": state.status,
            "bundle_handle": {
                "validated_bundle_id_sha256": (
                    state.bundle_handle.validated_bundle_id_sha256
                ),
                "validator_receipt_sha256": (
                    state.bundle_handle.validator_receipt_sha256
                ),
                "expected_content_root_sha256": (
                    state.bundle_handle.expected_content_root_sha256
                ),
                "checkpoint_sha256": state.bundle_handle.checkpoint_sha256,
                "runtime_sha256": state.bundle_handle.runtime_sha256,
                "method_lock_sha256": state.bundle_handle.method_lock_sha256,
                "receipt_root_sha256": state.bundle_handle.receipt_root_sha256,
                "target_rows": 0,
            },
            "bundle_wire_sha256": _sha256_bytes(bundle_wire),
            "coefficient_fp16": _array_receipt(state.coefficient_fp16),
            "fit_audit": state.fit_audit,
            "query_state_updates": 0,
            "query_rows_used_for_fit": 0,
            "persistent_fp16_or_fp32_learned_sidecar": False,
        }
    )
    return b"".join(
        (
            WIRE_MAGIC,
            struct.pack("<Q", len(header)),
            header,
            struct.pack("<Q", len(bundle_wire)),
            bundle_wire,
            state.coefficient_fp16.tobytes(order="C"),
        )
    )


def audit_d105_cbrc_resources(state: D105CBRCState) -> dict[str, Any]:
    """Return the exact D105-side algorithmic resource receipt.

    qKNN/head work is intentionally absent and must be added by the integration
    layer; this module owns only the shared DA transform.
    """

    _validate_state_integrity(state)
    audit = state.fit_audit
    query_matmul = Z_DIM * CODE_DIM if state.active else 0
    state_bytes = len(serialize_d105_cbrc_state(state))
    support_fit_mac = int(
        audit["support_encoding_algorithmic_mac"]
        + audit["support_statistic_algorithmic_mac"]
        + audit["irls_main_algorithmic_mac"]
        + audit["irls_loo_algorithmic_mac"]
        + audit["b_projection_algorithmic_mac"]
    )
    result = {
        "schema": RESOURCE_AUDIT_SCHEMA,
        "status": state.status,
        "actual_serialized_state_bytes": state_bytes,
        "numeric_bundle_state_bytes": state.bundle.numeric_state_bytes,
        "coefficient_fp16_bytes": int(state.coefficient_fp16.nbytes),
        "fp32_persistent_sidecar_bytes": 0,
        "fp16_persistent_learned_sidecar_bytes": 0,
        "persistent_decoded_learning_arrays": False,
        "trainable_parameters_stage2": 0,
        "optimizer_steps_stage2": 0,
        "support_fit_algorithmic_mac": support_fit_mac,
        "irls_steps": IRLS_STEPS,
        "loo_fit_algorithmic_mac": int(audit["irls_loo_algorithmic_mac"]),
        "b_projection_algorithmic_mac": int(
            audit["b_projection_algorithmic_mac"]
        ),
        "fit_task_weight_sums_per_iteration": audit[
            "irls_task_weight_sums_per_iteration"
        ],
        "temporary_fit_bytes": int(audit["temporary_fit_bytes"]),
        "temporary_domain_phase_bytes": int(
            audit["temporary_domain_phase_bytes"]
        ),
        "temporary_solver_phase_bytes": int(
            audit["temporary_solver_phase_bytes"]
        ),
        "temporary_solver_vector_bytes": int(
            audit["temporary_solver_vector_bytes"]
        ),
        "temporary_class_target_bytes": int(
            audit["temporary_class_target_bytes"]
        ),
        "query_da_matmul_mac_per_query": query_matmul,
        "query_extra_dot_product_mac": 0,
        "post_backbone_da_mac_per_query": query_matmul,
        "query_integrity_content_root_recomputations": 1,
        "query_integrity_numeric_payload_read_bytes": int(
            state.bundle.numeric_state_bytes
        ),
        "query_integrity_numeric_temporary_bytes": 0,
        "state_gate_bytes": MAX_STATE_BYTES,
        "query_gate_mac": MAX_POST_BACKBONE_MAC_PER_QUERY,
        "passes_state_gate": state_bytes < MAX_STATE_BYTES,
        "passes_query_da_mac_gate": query_matmul <= MAX_POST_BACKBONE_MAC_PER_QUERY,
        "query_state_updates": 0,
        "query_batch_dependency": False,
        "ground_old_multiprototype_enabled": False,
    }
    return _deep_copy_public(result)


__all__ = [
    "ACTIVE",
    "ALLOWED_K",
    "ALLOWED_STAGES",
    "D105CBRCBundleHandle",
    "D105CBRCError",
    "D105CBRCState",
    "D105CBRCSupportSolution",
    "FALLBACK_LEGAL_DATA",
    "IRLS_STEPS",
    "MAX_POST_BACKBONE_MAC_PER_QUERY",
    "MAX_STATE_BYTES",
    "audit_d105_cbrc_resources",
    "compute_d105_bundle_receipt_root",
    "compute_d105_bundle_validator_receipt",
    "compute_d105_support_binding_root",
    "fit_d105_cbrc_state",
    "make_d105_cbrc_bundle_handle",
    "serialize_d105_cbrc_state",
    "solve_d105_cbrc_support",
    "transform_d105_canonical",
    "transform_d105_cbrc",
    "validate_d105_physical_split",
]
