"""Truth-free, single-logical-row runtime for NEXT-R4.

This module is intentionally the narrow integration point for the frozen
FA-RDCE3 domain state, the direct qKNN base head, and CER-PLR160.  It knows
nothing about labels of query rows, query roles, quotas, scoring, selection,
or promotion.  A call consumes exactly one K-specific logical row and emits
the four explicitly named causal states:

``DA0_REG0`` / ``DA1_REG0`` / ``DA0_REG1`` / ``DA1_REG1``.

The R1 qKNN route deliberately does *not* call the public qKNN scorer: that
entry point normalizes its input.  R1 is already the final signed-unit
FA-RDCE3 representation and must not receive another ReLU, translation, or
L2 map.  The direct path retains the frozen INT8 per-support-row wire and the
same Student-t/class-logsumexp formula, but uses explicit cosine denominators
for both the query score and the K5 class-scale estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_next_r4_cer_plr160 as cer
from . import stage2_next_r4_fa_rdce3 as fa
from . import stage2_next_r4_matrix as matrix
from . import stage2_zid_student_t_qknn as qknn


RUNTIME_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.logical_row_runtime.v1"
DIRECT_QKNN_SCHEMA = "cvs.stage2.next_r4.direct_qknn_int8_cosine.v1"
QUERY_ISOLATION_SCHEMA = "cvs.stage2.next_r4.query_isolation.v1"
RESOURCE_SCHEMA = "cvs.stage2.next_r4.logical_row_resource.v1"

Z_DIM = 160
STATE_IDS = matrix.STATE_IDS
ARM_IDS = matrix.ARM_IDS
_R0_STATES = ("DA0_REG0", "DA0_REG1")
_R1_STATES = ("DA1_REG0", "DA1_REG1")
_REG0_STATES = ("DA0_REG0", "DA1_REG0")
_REG1_STATES = ("DA0_REG1", "DA1_REG1")
_UNIT_ATOL = 2.0e-6
_EPSILON = float(np.finfo(np.float64).eps)


class NextR4RuntimeError(ValueError):
    """Raised when the frozen R4 row-level runtime contract drifts."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, tuple | list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.ascontiguousarray(value).copy()
        copied.setflags(write=False)
        return copied
    if isinstance(value, np.generic):
        return value.item()
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return MappingProxyType(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def _id_root(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise NextR4RuntimeError(f"{name} must be a lowercase SHA256")
    return value


def _tokens(value: Sequence[str], *, name: str, expected: int | None = None) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise NextR4RuntimeError(f"{name} must be a sequence of opaque IDs")
    result = tuple(value)
    if (
        (expected is not None and len(result) != expected)
        or not result
        or any(type(item) is not str or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise NextR4RuntimeError(f"{name} must contain unique nonempty opaque IDs")
    return result


def _registry(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise NextR4RuntimeError(f"{name} must be an ordered class registry")
    result = tuple(value)
    if len(result) < 2 or any(type(item) is not str or not item for item in result) or len(set(result)) != len(result):
        raise NextR4RuntimeError(f"{name} must contain unique nonempty classes")
    return result


def _readonly_float32(value: np.ndarray, *, name: str, representation: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise NextR4RuntimeError(f"{name} must be a numpy float32 array")
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] != Z_DIM
        or not np.isfinite(rows).all()
    ):
        raise NextR4RuntimeError(f"{name} must be finite float32 [N,{Z_DIM}]")
    if representation == "R0" and bool(np.any(rows < np.float32(0.0))):
        raise NextR4RuntimeError(f"{name} must be canonical nonnegative R0")
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_UNIT_ATOL):
        raise NextR4RuntimeError(f"{name} must already be unit {representation}")
    copied = np.ascontiguousarray(rows, dtype=np.float32).copy()
    copied.setflags(write=False)
    return copied


def _byte_identical(left: np.ndarray, right: np.ndarray) -> bool:
    lhs = np.ascontiguousarray(left)
    rhs = np.ascontiguousarray(right)
    return lhs.dtype == rhs.dtype and lhs.shape == rhs.shape and lhs.tobytes(order="C") == rhs.tobytes(order="C")


@dataclass(frozen=True, slots=True)
class NextR4RegistrationInput:
    """One R0-only support/query package for REG0 or REG1.

    Query IDs deliberately have no class, truth, role, or quota field.  The
    matrix binding, rather than this runtime input, proves their common order.
    """

    registration_id: str
    registered_classes: Sequence[str]
    support_r0_by_class: Mapping[str, np.ndarray]
    support_physical_ids_by_class: Mapping[str, Sequence[str]]
    query_r0_zid160: np.ndarray
    query_physical_ids: Sequence[str]
    query_observation_ids: Sequence[str]

    def __post_init__(self) -> None:
        if self.registration_id not in matrix.REGISTRATION_IDS:
            raise NextR4RuntimeError("registration_id must be REG0 or REG1")
        classes = _registry(self.registered_classes, name="registered_classes")
        if (
            not isinstance(self.support_r0_by_class, Mapping)
            or not isinstance(self.support_physical_ids_by_class, Mapping)
            or set(self.support_r0_by_class) != set(classes)
            or set(self.support_physical_ids_by_class) != set(classes)
        ):
            raise NextR4RuntimeError("support maps must close exactly over registered classes")
        support: dict[str, np.ndarray] = {}
        physical: dict[str, tuple[str, ...]] = {}
        active_k: int | None = None
        all_support_ids: list[str] = []
        for class_id in classes:
            rows = _readonly_float32(
                self.support_r0_by_class[class_id],
                name=f"support_r0_by_class[{class_id!r}]",
                representation="R0",
            )
            ids = _tokens(
                self.support_physical_ids_by_class[class_id],
                name=f"support_physical_ids_by_class[{class_id!r}]",
                expected=len(rows),
            )
            if active_k is None:
                active_k = len(rows)
            if len(rows) != active_k:
                raise NextR4RuntimeError("support must be balanced K-shot by class")
            support[class_id] = rows
            physical[class_id] = ids
            all_support_ids.extend(ids)
        if active_k not in matrix.K_VALUES or len(set(all_support_ids)) != len(all_support_ids):
            raise NextR4RuntimeError("support must have globally unique frozen K1/K5 physical IDs")
        query = _readonly_float32(
            self.query_r0_zid160, name="query_r0_zid160", representation="R0"
        )
        query_ids = _tokens(self.query_physical_ids, name="query_physical_ids", expected=len(query))
        observation_ids = _tokens(
            self.query_observation_ids, name="query_observation_ids", expected=len(query)
        )
        if set(all_support_ids) & (set(query_ids) | set(observation_ids)):
            raise NextR4RuntimeError("support physical IDs overlap query physical/observation IDs")
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_r0_by_class", MappingProxyType(support))
        object.__setattr__(self, "support_physical_ids_by_class", MappingProxyType(physical))
        object.__setattr__(self, "query_r0_zid160", query)
        object.__setattr__(self, "query_physical_ids", query_ids)
        object.__setattr__(self, "query_observation_ids", observation_ids)

    @property
    def active_k(self) -> int:
        return len(self.support_r0_by_class[self.registered_classes[0]])

    @property
    def support_r0_zid160(self) -> np.ndarray:
        return _freeze(np.concatenate([self.support_r0_by_class[item] for item in self.registered_classes], axis=0))

    @property
    def support_labels(self) -> tuple[str, ...]:
        return tuple(
            class_id for class_id in self.registered_classes for _ in range(self.active_k)
        )

    @property
    def support_physical_ids(self) -> tuple[str, ...]:
        return tuple(
            physical_id
            for class_id in self.registered_classes
            for physical_id in self.support_physical_ids_by_class[class_id]
        )


def _class_scales_direct(
    *, decoded_support: np.ndarray, class_indices: np.ndarray, classes: tuple[str, ...], lock: qknn.Phase1ZIDStudentTLock
) -> np.ndarray:
    if lock.active_k == 1:
        return np.full(len(classes), float(lock.shared_h0), dtype=np.float64)
    values: list[float] = []
    for class_index in range(len(classes)):
        local = decoded_support[class_indices == class_index].astype(np.float64, copy=False)
        norms = np.linalg.norm(local, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= _EPSILON):
            raise NextR4RuntimeError("direct qKNN decoded support has an undefined cosine denominator")
        cosine = (local @ local.T) / (norms[:, None] * norms[None, :])
        cosine = np.clip(cosine, -1.0, 1.0)
        distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
        upper = distance[np.triu_indices(lock.active_k, 1)]
        if len(upper) != lock.active_k * (lock.active_k - 1) // 2:
            raise NextR4RuntimeError("direct qKNN K5 class-scale closure drift")
        empirical = float(np.mean(upper, dtype=np.float64))
        shrunk = (empirical + float(lock.scale_prior_strength) * float(lock.shared_h0) ** 2) / (1.0 + float(lock.scale_prior_strength))
        values.append(
            float(
                np.clip(
                    math.sqrt(max(shrunk, _EPSILON)),
                    float(lock.shared_h0) * float(lock.scale_min_ratio),
                    float(lock.shared_h0) * float(lock.scale_max_ratio),
                )
            )
        )
    return np.asarray(values, dtype=np.float64)


def _build_direct_qknn_bank(
    *, support: np.ndarray, labels: tuple[str, ...], classes: tuple[str, ...], lock: qknn.Phase1ZIDStudentTLock
) -> qknn.TypedINT8ZIDSupportBank:
    """Compile the qKNN INT8 wire without a post-R0/R1 L2 operation."""

    if type(lock) is not qknn.Phase1ZIDStudentTLock:
        raise NextR4RuntimeError("NEXT-R4 requires the exact frozen qKNN lock")
    rows = np.asarray(support)
    if rows.dtype != np.float32 or rows.shape != (len(labels), Z_DIM):
        raise NextR4RuntimeError("direct qKNN support layout drift")
    if lock.active_k not in matrix.K_VALUES:
        raise NextR4RuntimeError("direct qKNN lock K drift")
    index_by_class = {class_id: index for index, class_id in enumerate(classes)}
    indices = np.asarray([index_by_class[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(indices == index)) for index in range(len(classes)))
    if len(set(counts)) != 1 or counts[0] != lock.active_k:
        raise NextR4RuntimeError("direct qKNN balanced support/lock K drift")
    codes = np.zeros(rows.shape, dtype=np.int8)
    scales = np.empty(len(rows), dtype=np.float16)
    minimum = float(np.finfo(np.float16).tiny)
    for index, row in enumerate(rows):
        maximum = float(np.max(np.abs(row)))
        scale = np.float16(max(maximum / 127.0, minimum))
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise NextR4RuntimeError("direct qKNN INT8 scale is not representable")
        encoded = np.clip(np.rint(row.astype(np.float64) / float(scale)), -127.0, 127.0).astype(np.int8)
        if np.any(encoded == np.int8(-128)):
            raise NextR4RuntimeError("direct qKNN INT8 code range drift")
        codes[index] = encoded
        scales[index] = scale
    order_keys = [
        (int(indices[index]), np.ascontiguousarray(codes[index]).tobytes(), np.ascontiguousarray(scales[index]).tobytes(), index)
        for index in range(len(codes))
    ]
    order = np.asarray(sorted(range(len(order_keys)), key=order_keys.__getitem__), dtype=np.int64)
    codes, scales, indices = codes[order], scales[order], indices[order]
    decoded = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
    class_scales = _class_scales_direct(
        decoded_support=decoded, class_indices=indices, classes=classes, lock=lock
    )
    class_scales16 = np.asarray(class_scales, dtype=np.float16)
    if not np.isfinite(class_scales16).all() or np.any(class_scales16 <= 0.0):
        raise NextR4RuntimeError("direct qKNN class scales are not representable")
    input_after_quantization = rows[order].astype(np.float64, copy=False)
    reconstruction_error = np.abs(decoded - input_after_quantization)
    input_norms = np.linalg.norm(input_after_quantization, axis=1)
    decoded_norms = np.linalg.norm(decoded, axis=1)
    if np.any(input_norms <= _EPSILON) or np.any(decoded_norms <= _EPSILON):
        raise NextR4RuntimeError("direct qKNN quantization produced undefined norms")
    reconstruction_cosine = np.sum(decoded * input_after_quantization, axis=1) / (decoded_norms * input_norms)
    audit = {
        "schema": DIRECT_QKNN_SCHEMA,
        "feature_space": "z_id160_only",
        "support_only": True,
        "single_received_observation": True,
        "support_rows": int(len(codes)),
        "class_count": int(len(classes)),
        "support_counts": list(counts),
        "per_vector_scale": True,
        "quantization_error_mean": float(np.mean(reconstruction_error)),
        "quantization_error_max": float(np.max(reconstruction_error)),
        "reconstruction_cosine_mean": float(np.mean(reconstruction_cosine)),
        "reconstruction_cosine_min": float(np.min(reconstruction_cosine)),
        "class_scale_source": "direct_explicit_cosine_support_only_uniform_class_formula",
        "class_count_normalization": "logsumexp_minus_log_Kc",
        "same_formula_all_registered_classes": True,
        "pre_quantization_l2_normalization_applied": False,
        "decoded_support_post_quantization_l2_normalization_applied": False,
        "query_rows_used_for_fit": 0,
        "config_lock_digest": lock.lock_digest,
    }
    # The typed bank is still the frozen qKNN wire/receipt type.  Only its
    # compiler avoids the public normalize_zid_rows helper after R0/R1.
    payload = qknn._bank_payload(  # type: ignore[attr-defined]
        classes=classes,
        counts=counts,
        codes=codes,
        scales=scales,
        class_indices=indices,
        class_scales=class_scales16,
        config=lock,
        quantization=audit,
    )
    return qknn.TypedINT8ZIDSupportBank(
        classes=classes,
        support_counts=counts,
        codes_qint8=codes,
        scales_fp16=scales,
        class_indices_int16=indices,
        class_scales_fp16=class_scales16,
        active_k=lock.active_k,
        config_lock_digest=lock.lock_digest,
        config=lock,
        quantization_audit=audit,
        bank_receipt_sha256=qknn._canonical_sha256(payload),  # type: ignore[attr-defined]
    )


def _score_direct_qknn(
    *, bank: qknn.TypedINT8ZIDSupportBank, query: np.ndarray, metric: qknn.TypedSharedPSDMetric
) -> np.ndarray:
    """Student-t qKNN with an explicit cosine denominator and no L2 map."""

    if metric.exact_identity is not True:
        raise NextR4RuntimeError("NEXT-R4 direct qKNN is frozen to the identity shared metric")
    support = bank.codes_qint8.astype(np.float64) * bank.scales_fp16.astype(np.float64)[:, None]
    query64 = np.asarray(query, dtype=np.float64)
    support_norm = np.linalg.norm(support, axis=1)
    query_norm = np.linalg.norm(query64, axis=1)
    if (
        not np.isfinite(support_norm).all()
        or not np.isfinite(query_norm).all()
        or np.any(support_norm <= _EPSILON)
        or np.any(query_norm <= _EPSILON)
    ):
        raise NextR4RuntimeError("direct qKNN precision-cosine denominator is undefined")
    cosine = (query64 @ support.T) / (query_norm[:, None] * support_norm[None, :])
    distance = np.maximum(2.0 * (1.0 - np.clip(cosine, -1.0, 1.0)), 0.0)
    columns: list[np.ndarray] = []
    for class_index, expected in enumerate(bank.support_counts):
        local = distance[:, bank.class_indices_int16 == class_index]
        if local.shape[1] != expected:
            raise NextR4RuntimeError("direct qKNN class support count drift during score")
        h = float(bank.class_scales_fp16[class_index])
        nu = float(bank.config.student_nu)
        d_eff = float(bank.config.kernel_effective_dim)
        kernel = (
            -float(bank.config.kernel_volume_gamma) * d_eff * math.log(h)
            -0.5 * (nu + d_eff) * np.log1p(local / (nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(maximum[:, 0] + np.log(np.sum(np.exp(kernel - maximum), axis=1)) - math.log(expected))
    logits64 = np.stack(columns, axis=1)
    if not np.isfinite(logits64).all():
        raise NextR4RuntimeError("direct qKNN Student-t logits became non-finite")
    logits = np.ascontiguousarray(logits64, dtype=np.float32)
    logits.setflags(write=False)
    return logits


def _q_head(
    *, support: np.ndarray, labels: tuple[str, ...], classes: tuple[str, ...], query: np.ndarray, representation: str, lock: qknn.Phase1ZIDStudentTLock
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    bank = _build_direct_qknn_bank(support=support, labels=labels, classes=classes, lock=lock)
    metric = qknn.identity_shared_psd_metric(config=lock)
    logits = _score_direct_qknn(bank=bank, query=query, metric=metric)
    try:
        cer.require_unique_float32_top(logits)
    except Exception as error:
        raise NextR4RuntimeError("TIE_UNRESOLVED: direct qKNN final float32 top tie") from error
    resource = qknn.audit_runtime_state(bank, metric)
    receipt = {
        "schema": DIRECT_QKNN_SCHEMA,
        "representation": representation,
        "qknn_lock_digest": lock.lock_digest,
        "qknn_bank_sha256": bank.bank_receipt_sha256,
        "qknn_metric_sha256": metric.metric_receipt_sha256,
        "qknn_metric_rank": metric.effective_rank,
        "int8_per_row_support_quantization": True,
        "pre_quantization_l2_normalization_applied": False,
        "decoded_support_post_quantization_l2_normalization_applied": False,
        "query_post_representation_l2_normalization_applied": False,
        "post_representation_l2_normalization_applied": False,
        "post_representation_relu_applied": False,
        "post_representation_translation_applied": False,
        "explicit_precision_cosine_denominator": True,
        "k5_class_scale_explicit_cosine_denominator": lock.active_k == 5,
        "student_t_kernel": True,
        "class_logsumexp_minus_log_k": True,
        "all_registered_classes_scored": True,
        "independent_per_query": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "class_quota_access": False,
        "global_reassignment_calls": 0,
    }
    return logits, _freeze(receipt), _freeze(resource)


def _predictions(logits: np.ndarray, classes: tuple[str, ...]) -> tuple[str, ...]:
    try:
        cer.require_unique_float32_top(logits)
    except Exception as error:
        raise NextR4RuntimeError("TIE_UNRESOLVED: final float32 top tie") from error
    return tuple(classes[int(index)] for index in np.argmax(logits, axis=1))


def _arm(
    *, arm_id: str, logits: np.ndarray, classes: tuple[str, ...], q_receipt: Mapping[str, Any], head_receipt: Mapping[str, Any], exact_alias: bool, unique_prediction: bool, head_status: str, no_head_function_reason: str | None = None
) -> Mapping[str, Any]:
    if arm_id not in ARM_IDS:
        raise NextR4RuntimeError("unknown NEXT-R4 arm")
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1] != len(classes):
        raise NextR4RuntimeError("arm logits dtype/shape drift")
    receipt = {
        "schema": "cvs.stage2.next_r4.arm.v1",
        "arm_id": arm_id,
        "head_status": head_status,
        "exact_qknn_alias": exact_alias,
        "alias_target_arm": "Q" if exact_alias else None,
        "unique_prediction": unique_prediction,
        "all_registered_classes_scored": True,
        "independent_per_query": True,
        "exact_float32_top_tie_closed": True,
        "qknn_receipt": _plain(q_receipt),
        "head_receipt": _plain(head_receipt),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_truth_access": False,
        "query_role_access": False,
        "class_quota_access": False,
        "global_reassignment_calls": 0,
    }
    if no_head_function_reason is not None:
        receipt["no_head_function_reason"] = no_head_function_reason
    return _freeze({"predictions": _predictions(logits, classes), "receipt": receipt})


def _run_state(
    *, state_id: str, registration: NextR4RegistrationInput, support: np.ndarray, query: np.ndarray, representation: str, lock: qknn.Phase1ZIDStudentTLock
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if state_id not in STATE_IDS or representation not in {"R0", "R1"}:
        raise NextR4RuntimeError("state/representation identity drift")
    if state_id in _R0_STATES and representation != "R0":
        raise NextR4RuntimeError("DA0 state must consume R0")
    if state_id in _R1_STATES and representation != "R1":
        raise NextR4RuntimeError("DA1 state must consume R1")
    expected_registration = "REG0" if state_id in _REG0_STATES else "REG1"
    if registration.registration_id != expected_registration:
        raise NextR4RuntimeError("state/registration closure drift")
    checked_support = _readonly_float32(support, name=f"{state_id} support", representation=representation)
    checked_query = _readonly_float32(query, name=f"{state_id} query", representation=representation)
    q_logits, q_receipt, q_resource = _q_head(
        support=checked_support,
        labels=registration.support_labels,
        classes=registration.registered_classes,
        query=checked_query,
        representation=representation,
        lock=lock,
    )
    try:
        head_fit = cer.fit_cer_plr160(
            checked_support,
            registration.support_labels,
            registration.registered_classes,
            qknn_lock=lock,
            representation=representation,
        )
        if registration.active_k == 1:
            h_logits = cer.alias_k1_qknn_logits(head_fit, q_logits)
            if h_logits is not q_logits:
                raise NextR4RuntimeError("K1 CER H must be the exact qKNN logit object")
            exact_alias, unique_prediction = True, False
        else:
            h_logits = cer.score_cer_plr160(head_fit, q_logits, checked_query)
            no_function = type(head_fit.state) is cer.CERPLR160NoFunctionAliasState
            if no_function and h_logits is not q_logits:
                raise NextR4RuntimeError("NO_HEAD_FUNCTION must preserve the exact qKNN logit object")
            exact_alias, unique_prediction = no_function, not no_function
    except NextR4RuntimeError:
        raise
    except Exception as error:
        raise NextR4RuntimeError(f"{state_id} CER-PLR160 fit/score failed") from error
    q_arm = _arm(
        arm_id="Q", logits=q_logits, classes=registration.registered_classes,
        q_receipt=q_receipt, head_receipt={"head": "qKNN_base", "head_status": "FUNCTIONAL"},
        exact_alias=False, unique_prediction=True, head_status="FUNCTIONAL",
    )
    h_fit_receipt = _plain(head_fit.fit_receipt)
    h_arm = _arm(
        arm_id="H", logits=h_logits, classes=registration.registered_classes,
        q_receipt=q_receipt, head_receipt={"fit_receipt": h_fit_receipt, "resource_receipt": _plain(head_fit.resource_receipt)},
        exact_alias=exact_alias, unique_prediction=unique_prediction,
        head_status=str(h_fit_receipt.get("head_status")),
        no_head_function_reason=(
            str(h_fit_receipt.get("no_head_function_reason"))
            if h_fit_receipt.get("head_status") == "NO_HEAD_FUNCTION"
            else None
        ),
    )
    state = {
        "state_id": state_id,
        "state_name_zh": matrix.STATE_NAMES_ZH[state_id],
        "registered_classes": list(registration.registered_classes),
        "query_physical_ids": list(registration.query_physical_ids),
        "query_observation_ids": list(registration.query_observation_ids),
        "arms": {"Q": _plain(q_arm), "H": _plain(h_arm)},
    }
    resource = {
        "state_id": state_id,
        "representation": representation,
        "qknn": {
            **_plain(q_receipt),
            "runtime_resource_audit": _plain(q_resource),
        },
        "cer_plr160": _plain(head_fit.resource_receipt),
        "cer_head_status": head_fit.fit_receipt.get("head_status"),
        "post_representation_relu_applied": False,
        "post_representation_l2_normalization_applied": False,
        "post_representation_translation_applied": False,
    }
    return _freeze(state), _freeze(resource)


def _validate_shared_binding(
    *, row: matrix.NextR4ProxyRow, binding_receipt: Mapping[str, Any], reg0: NextR4RegistrationInput, reg1: NextR4RegistrationInput
) -> Mapping[str, Any]:
    try:
        binding = matrix.validate_next_r4_binding(binding_receipt)
    except Exception as error:
        raise NextR4RuntimeError("shared physical binding receipt drift") from error
    expected_row_key = "k1_row_id" if row.active_k == 1 else "k5_row_id"
    if (
        binding.get(expected_row_key) != row.row_id
        or binding.get("held_receiver") != row.held_receiver
        or binding.get("held_class") != row.held_class
        or tuple(binding.get("registered_classes", ())) != row.all_registered_classes
    ):
        raise NextR4RuntimeError("shared physical binding/current logical row drift")
    support_key = "k1_support_ids_by_class" if row.active_k == 1 else "k5_support_ids_by_class"
    support_map = binding.get(support_key)
    if not isinstance(support_map, Mapping):
        raise NextR4RuntimeError("shared physical binding lacks current-K support IDs")
    for class_id in row.retained_classes:
        if tuple(support_map.get(class_id, ())) != reg0.support_physical_ids_by_class[class_id]:
            raise NextR4RuntimeError("REG0 physical support IDs drift from shared binding")
    for class_id in row.all_registered_classes:
        if tuple(support_map.get(class_id, ())) != reg1.support_physical_ids_by_class[class_id]:
            raise NextR4RuntimeError("REG1 physical support IDs drift from shared binding")
    flattened_query = tuple(binding.get("query_physical_ids", ()))
    flattened_observation = tuple(binding.get("query_observation_ids", ()))
    if (
        reg0.query_physical_ids != flattened_query
        or reg1.query_physical_ids != flattened_query
        or reg0.query_observation_ids != flattened_observation
        or reg1.query_observation_ids != flattened_observation
        or not _byte_identical(reg0.query_r0_zid160, reg1.query_r0_zid160)
    ):
        raise NextR4RuntimeError("four-state common query physical/observation bytes/order drift")
    return binding


def _validate_row_inputs(
    *, row: matrix.NextR4ProxyRow, asset: fa.FARDCE3Phase1Asset, binding: fa.FARDCE3RuntimeBinding, reg0: NextR4RegistrationInput, reg1: NextR4RegistrationInput, lock: qknn.Phase1ZIDStudentTLock
) -> None:
    if type(row) is not matrix.NextR4ProxyRow:
        raise NextR4RuntimeError("runtime requires an exact NEXT-R4 logical row")
    if type(asset) is not fa.FARDCE3Phase1Asset or type(binding) is not fa.FARDCE3RuntimeBinding:
        raise NextR4RuntimeError("runtime requires exact FA-RDCE3 Phase1 asset/binding")
    if type(lock) is not qknn.Phase1ZIDStudentTLock:
        raise NextR4RuntimeError("runtime requires an exact frozen qKNN lock")
    if (
        reg0.registration_id != "REG0"
        or reg1.registration_id != "REG1"
        or reg0.registered_classes != row.retained_classes
        or reg1.registered_classes != row.all_registered_classes
        or reg0.active_k != row.active_k
        or reg1.active_k != row.active_k
        or lock.active_k != row.active_k
        or asset.old_classes != row.retained_classes
        or binding.old_classes != row.retained_classes
        or binding.active_k != row.active_k
        or binding.row_id != row.row_id
        or binding.checkpoint_sha256 != asset.checkpoint_sha256
    ):
        raise NextR4RuntimeError("row/REG0/REG1/FA/qKNN K-class binding drift")
    if binding.support_physical_root_sha256 != _id_root(reg0.support_physical_ids):
        raise NextR4RuntimeError("FA binding support physical root drift")
    for class_id in row.retained_classes:
        if (
            reg1.support_physical_ids_by_class[class_id] != reg0.support_physical_ids_by_class[class_id]
            or not _byte_identical(reg1.support_r0_by_class[class_id], reg0.support_r0_by_class[class_id])
        ):
            raise NextR4RuntimeError("REG1 must byte-preserve every REG0 old-class support row")


def _query_isolation_receipt() -> Mapping[str, Any]:
    return _freeze(
        {
            "schema": QUERY_ISOLATION_SCHEMA,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_truth_access": False,
            "query_role_access": False,
            "class_quota_access": False,
            "true_batch_class_count_access": False,
            "query_batch_dependency": False,
            "global_reassignment_calls": 0,
            "source_runtime_access": False,
            "clean_runtime_access": False,
            "phase2_optimizer_steps": 0,
            "phase2_backward_calls": 0,
            "post_representation_relu_applied": False,
            "post_representation_l2_normalization_applied": False,
            "post_representation_translation_applied": False,
        }
    )


def execute_next_r4_logical_row(
    *,
    row: matrix.NextR4ProxyRow,
    binding_receipt: Mapping[str, Any],
    fa_asset: fa.FARDCE3Phase1Asset,
    fa_binding: fa.FARDCE3RuntimeBinding,
    reg0: NextR4RegistrationInput,
    reg1: NextR4RegistrationInput,
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> Mapping[str, Any]:
    """Run exactly one frozen K-specific row without opening query truth.

    ``binding_receipt`` is a previously sealed *physical* paired-K receipt.
    It intentionally does not contain an FA state digest: K1/K5 may have
    different support and therefore different FA states.  This call fits one
    REG0 state once, then proves only the required same-K DA1_REG0→DA1_REG1
    object/byte reuse.
    """

    _validate_row_inputs(
        row=row, asset=fa_asset, binding=fa_binding, reg0=reg0, reg1=reg1, lock=qknn_lock
    )
    shared_binding = _validate_shared_binding(
        row=row, binding_receipt=binding_receipt, reg0=reg0, reg1=reg1
    )
    try:
        da_state = fa.fit_fa_rdce3_reg0(
            fa_asset, dict(reg0.support_r0_by_class), binding=fa_binding
        )
    except Exception as error:
        raise NextR4RuntimeError("FA-RDCE3 REG0 support-only fit failed") from error
    state_before = (
        da_state.runtime_receipt_sha256,
        da_state.a_fp16.tobytes(order="C"),
        id(da_state),
    )
    try:
        reused_state = fa.reuse_fa_rdce3_state_for_reg1(
            da_state, registered_classes=reg1.registered_classes
        )
        core_reuse = fa.fa_rdce3_reg1_reuse_receipt(
            da_state, registered_classes=reg1.registered_classes
        )
    except Exception as error:
        raise NextR4RuntimeError("FA-RDCE3 REG1 reuse proof failed") from error
    if reused_state is not da_state or reused_state.a_fp16.tobytes(order="C") != da_state.a_fp16.tobytes(order="C"):
        raise NextR4RuntimeError("DA1_REG1 must reuse the exact DA1_REG0 FA state object/bytes")
    try:
        fa_reuse = matrix.validate_fa_state_reuse(
            {"DA1_REG0": da_state.runtime_receipt_sha256, "DA1_REG1": reused_state.runtime_receipt_sha256}
        )
        r1_reg0_support = fa.transform_fa_rdce3_r1(da_state, reg0.support_r0_zid160)
        r1_reg0_query = fa.transform_fa_rdce3_r1(da_state, reg0.query_r0_zid160)
        r1_reg1_support = fa.transform_fa_rdce3_r1(reused_state, reg1.support_r0_zid160)
        r1_reg1_query = fa.transform_fa_rdce3_r1(reused_state, reg1.query_r0_zid160)
    except Exception as error:
        raise NextR4RuntimeError("FA-RDCE3 R1 transform/reuse receipt failed") from error
    if state_before != (
        da_state.runtime_receipt_sha256,
        da_state.a_fp16.tobytes(order="C"),
        id(da_state),
    ):
        raise NextR4RuntimeError("query transform mutated the frozen FA state")
    if not _byte_identical(r1_reg0_query, r1_reg1_query):
        raise NextR4RuntimeError("DA1 REG0/REG1 common query R1 bytes drift")
    states_and_resources = {
        "DA0_REG0": _run_state(
            state_id="DA0_REG0", registration=reg0, support=reg0.support_r0_zid160,
            query=reg0.query_r0_zid160, representation="R0", lock=qknn_lock,
        ),
        "DA1_REG0": _run_state(
            state_id="DA1_REG0", registration=reg0, support=r1_reg0_support,
            query=r1_reg0_query, representation="R1", lock=qknn_lock,
        ),
        "DA0_REG1": _run_state(
            state_id="DA0_REG1", registration=reg1, support=reg1.support_r0_zid160,
            query=reg1.query_r0_zid160, representation="R0", lock=qknn_lock,
        ),
        "DA1_REG1": _run_state(
            state_id="DA1_REG1", registration=reg1, support=r1_reg1_support,
            query=r1_reg1_query, representation="R1", lock=qknn_lock,
        ),
    }
    registrations = {
        "REG0": {
            "registered_classes": list(reg0.registered_classes),
            "states": {
                state_id: _plain(states_and_resources[state_id][0]) for state_id in _REG0_STATES
            },
        },
        "REG1": {
            "registered_classes": list(reg1.registered_classes),
            "states": {
                state_id: _plain(states_and_resources[state_id][0]) for state_id in _REG1_STATES
            },
        },
    }
    resource = {
        "schema": RESOURCE_SCHEMA,
        "row_id": row.row_id,
        "active_k": row.active_k,
        "fa_rdce3": _plain(fa.fa_rdce3_resource_receipt(da_state)),
        "fa_rdce3_reg1_reuse_core_receipt": _plain(core_reuse),
        "qknn_lock_digest": qknn_lock.lock_digest,
        "states": {state_id: _plain(value[1]) for state_id, value in states_and_resources.items()},
        "metric_availability_by_state": {
            "DA0_REG0": {"seen_new_acc": "N/A", "H_old_new": "N/A", "reason": "new_class_not_registered"},
            "DA1_REG0": {"seen_new_acc": "N/A", "H_old_new": "N/A", "reason": "new_class_not_registered"},
            "DA0_REG1": {"seen_new_acc": "scorer_only", "H_old_new": "scorer_only"},
            "DA1_REG1": {"seen_new_acc": "scorer_only", "H_old_new": "scorer_only"},
        },
        "post_representation_relu_applied": False,
        "post_representation_l2_normalization_applied": False,
        "post_representation_translation_applied": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    return _freeze(
        {
            "row_id": row.row_id,
            "held_receiver": row.held_receiver,
            "held_class": row.held_class,
            "active_k": row.active_k,
            "binding_receipt": _plain(shared_binding),
            "fa_state_reuse_receipt": _plain(fa_reuse),
            "registrations": registrations,
            "resource_receipt": resource,
            "query_isolation_receipt": _plain(_query_isolation_receipt()),
        }
    )


run_next_r4_logical_row = execute_next_r4_logical_row
execute_next_r4_four_state = execute_next_r4_logical_row


__all__ = [
    "DIRECT_QKNN_SCHEMA",
    "NextR4RegistrationInput",
    "NextR4RuntimeError",
    "QUERY_ISOLATION_SCHEMA",
    "RESOURCE_SCHEMA",
    "RUNTIME_SCHEMA",
    "execute_next_r4_logical_row",
    "execute_next_r4_four_state",
    "run_next_r4_logical_row",
]
