"""Runner-neutral same-row S0 prediction surface for D127's four core arms.

This module deliberately receives already materialized base and adapted
``z_id160`` feature caches.  It does not import a DA candidate, raw IQ, a
checkpoint hook, a scorer, or any query-side fitting API.  One call closes a
single row over the same registered-class support, opaque query order, and
frozen qKNN lock:

* ``M0``: base cache + frozen Student-t qKNN;
* ``M_DA``: caller-provided adapted cache + the same qKNN rule;
* ``M_L92``: base cache + D92-Lite; and
* ``M_JOINT``: the *same* adapted cache + D92-Lite.

K1 is not an approximate Lite path.  It binds a typed D92-Lite alias receipt
to the exact ``M0``/``M_DA`` qKNN logits object.  K5/K10 fit their own active
D92-Lite state from their corresponding support cache.  Query arrays are only
scored; every output is per-sample, read-only, and truth-free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_d127_d92_lite as d92_lite
from . import stage2_zid_student_t_qknn as qknn


Z_DIM = 160
S0_ACTIVE_K_VALUES = frozenset({1, 5, 10})
ROW_RECEIPT_SCHEMA = "cvs.stage2.d127.joint_screen.row_receipt.v1"
ARM_RECEIPT_SCHEMA = "cvs.stage2.d127.joint_screen.arm_receipt.v1"

M0 = "M0"
M_DA = "M_DA"
M_L92 = "M_L92"
M_JOINT = "M_JOINT"
_ARM_CONTRACTS = {
    M0: ("base_zid160", "phase1_locked_student_t_qknn"),
    M_DA: ("adapted_zid160", "phase1_locked_student_t_qknn"),
    M_L92: ("base_zid160", "d92_lite_dr_oas_lda"),
    M_JOINT: ("adapted_zid160", "d92_lite_dr_oas_lda"),
}


class D127JointScreenError(ValueError):
    """Raised when the frozen same-row four-arm contract is violated."""


def _freeze(value: Any) -> Any:
    """Freeze receipt-only JSON-like data without retaining mutable lists."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise D127JointScreenError("receipt must be a mapping")
    return _freeze(value)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    rows = np.ascontiguousarray(value)
    return {
        "dtype": rows.dtype.str,
        "shape": list(rows.shape),
        "nbytes": int(rows.nbytes),
        "sha256": hashlib.sha256(rows.tobytes(order="C")).hexdigest(),
    }


def _zid_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or rows.shape[0] < 1
        or not np.isfinite(rows).all()
    ):
        raise D127JointScreenError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return rows


def _registered_classes(values: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(str(value) for value in values)
    if len(classes) < 2 or len(set(classes)) != len(classes) or any(not value for value in classes):
        raise D127JointScreenError(
            "registered_classes must contain at least two unique non-empty classes"
        )
    return classes


def _opaque_query_ids(values: Sequence[str], *, rows: int) -> tuple[str, ...]:
    query_ids = tuple(str(value) for value in values)
    if (
        len(query_ids) != rows
        or not query_ids
        or len(set(query_ids)) != len(query_ids)
        or any(not value for value in query_ids)
    ):
        raise D127JointScreenError(
            "opaque_query_ids must be unique non-empty values aligned to query rows"
        )
    return query_ids


def _support_contract(
    support: np.ndarray,
    labels_value: Sequence[str],
    classes: tuple[str, ...],
    lock: qknn.Phase1ZIDStudentTLock,
) -> tuple[tuple[str, ...], int]:
    labels = tuple(str(value) for value in labels_value)
    if len(labels) != len(support) or any(label not in classes for label in labels):
        raise D127JointScreenError(
            "support_labels must align to support rows and close over registered_classes"
        )
    counts = tuple(sum(label == class_id for label in labels) for class_id in classes)
    if any(count < 1 for count in counts) or len(set(counts)) != 1:
        raise D127JointScreenError(
            "all registered classes require the same positive K-shot support"
        )
    active_k = counts[0]
    if active_k not in S0_ACTIVE_K_VALUES:
        raise D127JointScreenError("joint screen supports only K1, K5, or K10")
    if active_k != lock.active_k:
        raise D127JointScreenError("support K-shot does not match frozen qKNN lock")
    return labels, active_k


def _readonly_logits(
    value: np.ndarray, *, rows: int, classes: tuple[str, ...], arm_id: str
) -> np.ndarray:
    logits = np.asarray(value)
    if (
        logits.dtype != np.float32
        or logits.shape != (rows, len(classes))
        or not np.isfinite(logits).all()
        or logits.flags.writeable
    ):
        raise D127JointScreenError(
            f"{arm_id} must produce immutable finite float32 [query_rows,class_count] logits"
        )
    return logits


def _predictions(logits: np.ndarray, classes: tuple[str, ...]) -> tuple[str, ...]:
    maxima = np.max(logits, axis=1, keepdims=True)
    top_counts = np.sum(logits == maxima, axis=1)
    if np.any(top_counts != 1):
        raise D127JointScreenError(
            "exact top-logit tie is undefined; prediction fails closed"
        )
    return tuple(classes[int(index)] for index in np.argmax(logits, axis=1))


def _cache_receipt(
    support: np.ndarray, query: np.ndarray, *, representation: str
) -> dict[str, Any]:
    payload = {
        "representation": representation,
        "support_zid": _array_receipt(support),
        "query_zid": _array_receipt(query),
    }
    return {**payload, "cache_sha256": _canonical_sha256(payload)}


def _qknn_arm_receipt(
    *,
    arm_id: str,
    row_input_sha256: str,
    cache: Mapping[str, Any],
    bank: qknn.TypedINT8ZIDSupportBank,
    metric: qknn.TypedSharedPSDMetric,
    classes: tuple[str, ...],
    query_rows: int,
) -> Mapping[str, Any]:
    return _frozen_mapping(
        {
            "schema": ARM_RECEIPT_SCHEMA,
            "arm_id": arm_id,
            "row_input_sha256": row_input_sha256,
            "representation": cache["representation"],
            "head": "phase1_locked_student_t_qknn",
            "registered_classes": list(classes),
            "active_k": bank.active_k,
            "support_bank_receipt_sha256": bank.bank_receipt_sha256,
            "qknn_lock_digest": bank.config_lock_digest,
            "qknn_metric_receipt_sha256": metric.metric_receipt_sha256,
            "qknn_metric_exact_identity": metric.exact_identity,
            "cache_sha256": cache["cache_sha256"],
            "query_rows": query_rows,
            "all_registered_classes_scored": True,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_batch_dependency": False,
            "per_sample_decision": "argmax_over_all_registered_class_columns",
        }
    )


def _lite_arm_receipt(
    *,
    arm_id: str,
    row_input_sha256: str,
    cache: Mapping[str, Any],
    fitted: d92_lite.D92LiteFit,
    score: d92_lite.D92LiteScore,
    classes: tuple[str, ...],
    alias_receipt: d92_lite.D92LiteQKNNAliasReceipt | None,
    underlying_qknn: D127JointArmPrediction,
) -> Mapping[str, Any]:
    receipt: dict[str, Any] = {
        "schema": ARM_RECEIPT_SCHEMA,
        "arm_id": arm_id,
        "row_input_sha256": row_input_sha256,
        "representation": cache["representation"],
        "head": "d92_lite_dr_oas_lda",
        "registered_classes": list(classes),
        "active_k": int(fitted.state.active_k),
        "cache_sha256": cache["cache_sha256"],
        "d92_lite_state_receipt_sha256": fitted.state.state_receipt_sha256,
        "d92_lite_fit": dict(fitted.fit_receipt),
        "d92_lite_resource": dict(fitted.resource_receipt),
        "d92_lite_score": dict(score.score_receipt),
        "all_registered_classes_scored": True,
        "query_rows": int(len(score.logits)),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "per_sample_decision": "argmax_over_all_registered_class_columns",
    }
    if alias_receipt is not None:
        receipt.update(
            {
                "fit_mode": "exact_qknn_alias",
                "qknn_alias_receipt_schema": alias_receipt.schema,
                "qknn_alias_query_binding_sha256": alias_receipt.query_binding_sha256,
                "qknn_alias_logits_sha256": alias_receipt.qknn_logits_sha256,
                "underlying_qknn_arm": underlying_qknn.arm_id,
                "underlying_qknn_logit_object_reused": score.logits is underlying_qknn.logits,
                "underlying_qknn_logit_sha256": _array_receipt(
                    underlying_qknn.logits
                )["sha256"],
            }
        )
    else:
        receipt["fit_mode"] = "diagonal_oas_form"
    return _frozen_mapping(receipt)


@dataclass(frozen=True, slots=True)
class D127JointArmPrediction:
    """One immutable four-arm output, bound to one row and class-column order."""

    arm_id: str
    representation: str
    head: str
    classes: tuple[str, ...]
    logits: np.ndarray
    predictions: tuple[str, ...]
    receipt: Mapping[str, Any]
    qknn_alias_receipt: d92_lite.D92LiteQKNNAliasReceipt | None = None

    def __post_init__(self) -> None:
        contract = _ARM_CONTRACTS.get(self.arm_id)
        classes = _registered_classes(self.classes)
        if contract is None or (self.representation, self.head) != contract:
            raise D127JointScreenError("four-arm identity/representation/head contract drift")
        logits = _readonly_logits(
            self.logits,
            rows=len(self.predictions),
            classes=classes,
            arm_id=self.arm_id,
        )
        predictions = tuple(str(value) for value in self.predictions)
        if len(predictions) != len(logits) or any(value not in classes for value in predictions):
            raise D127JointScreenError("predictions must align to query rows and registered classes")
        receipt = _frozen_mapping(self.receipt)
        is_alias = receipt.get("fit_mode") == "exact_qknn_alias"
        if is_alias:
            if (
                self.arm_id not in {M_L92, M_JOINT}
                or type(self.qknn_alias_receipt) is not d92_lite.D92LiteQKNNAliasReceipt
            ):
                raise D127JointScreenError("K1 D92-Lite arm requires an exact typed qKNN alias receipt")
        elif self.qknn_alias_receipt is not None:
            raise D127JointScreenError("only K1 D92-Lite arms may retain an alias receipt")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "receipt", receipt)


@dataclass(frozen=True, slots=True)
class D127JointFourArmResult:
    """Same-row immutable outputs for ``M0``, ``M_DA``, ``M_L92``, ``M_JOINT``."""

    m0: D127JointArmPrediction
    m_da: D127JointArmPrediction
    m_l92: D127JointArmPrediction
    m_joint: D127JointArmPrediction
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        arms = (self.m0, self.m_da, self.m_l92, self.m_joint)
        expected = (M0, M_DA, M_L92, M_JOINT)
        if tuple(arm.arm_id for arm in arms) != expected:
            raise D127JointScreenError("four-arm output order/identity drift")
        classes = self.m0.classes
        if any(arm.classes != classes for arm in arms[1:]):
            raise D127JointScreenError("all four arms must preserve one class-column order")
        row_receipt = _frozen_mapping(self.receipt)
        if row_receipt.get("schema") != ROW_RECEIPT_SCHEMA:
            raise D127JointScreenError("joint four-arm row receipt schema drift")
        row_input_sha256 = row_receipt.get("row_input_sha256")
        if not isinstance(row_input_sha256, str) or len(row_input_sha256) != 64:
            raise D127JointScreenError("joint four-arm row binding is missing")
        if any(arm.receipt.get("row_input_sha256") != row_input_sha256 for arm in arms):
            raise D127JointScreenError("four-arm outputs do not share an exact row binding")
        active_k = row_receipt.get("active_k")
        if active_k == 1:
            if (
                self.m_l92.logits is not self.m0.logits
                or self.m_joint.logits is not self.m_da.logits
            ):
                raise D127JointScreenError("K1 must reuse exact M0/M_DA qKNN logits")
        else:
            if self.m_l92.qknn_alias_receipt is not None or self.m_joint.qknn_alias_receipt is not None:
                raise D127JointScreenError("active D92-Lite arms cannot carry K1 alias receipts")
        object.__setattr__(self, "receipt", row_receipt)

    @property
    def arms(self) -> tuple[D127JointArmPrediction, ...]:
        return (self.m0, self.m_da, self.m_l92, self.m_joint)


@dataclass(frozen=True, slots=True)
class D127JointTwoArmResult:
    """One common or adapted two-arm pair for cross-candidate S0 reuse."""

    pair_kind: str
    qknn_arm: D127JointArmPrediction
    lite_arm: D127JointArmPrediction
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "common": (M0, M_L92),
            "adapted": (M_DA, M_JOINT),
        }.get(self.pair_kind)
        if expected is None or (self.qknn_arm.arm_id, self.lite_arm.arm_id) != expected:
            raise D127JointScreenError("two-arm pair identity drift")
        if self.qknn_arm.classes != self.lite_arm.classes:
            raise D127JointScreenError("two-arm class registry drift")
        receipt = _frozen_mapping(self.receipt)
        if (
            receipt.get("schema") != ROW_RECEIPT_SCHEMA
            or receipt.get("pair_kind") != self.pair_kind
            or receipt.get("query_rows_used_for_fit") != 0
            or receipt.get("query_state_updates") != 0
            or receipt.get("query_selection_count") != 0
        ):
            raise D127JointScreenError("two-arm receipt drift")
        object.__setattr__(self, "receipt", receipt)

    @property
    def arms(self) -> tuple[D127JointArmPrediction, D127JointArmPrediction]:
        return (self.qknn_arm, self.lite_arm)


def _run_d127_two_arm(
    *,
    pair_kind: str,
    support_zid: np.ndarray,
    query_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> D127JointTwoArmResult:
    if type(qknn_lock) is not qknn.Phase1ZIDStudentTLock:
        raise D127JointScreenError("qknn_lock must be an exact frozen Phase1 lock")
    if pair_kind not in {"common", "adapted"}:
        raise D127JointScreenError("unknown two-arm pair kind")
    representation = "base_zid160" if pair_kind == "common" else "adapted_zid160"
    qknn_arm_id = M0 if pair_kind == "common" else M_DA
    lite_arm_id = M_L92 if pair_kind == "common" else M_JOINT
    support = _zid_rows(support_zid, name=f"{representation}_support")
    query = _zid_rows(query_zid, name=f"{representation}_query")
    classes = _registered_classes(registered_classes)
    labels, active_k = _support_contract(support, support_labels, classes, qknn_lock)
    query_ids = _opaque_query_ids(opaque_query_ids, rows=len(query))
    cache = _cache_receipt(support, query, representation=representation)
    row_binding = {
        "schema": ROW_RECEIPT_SCHEMA,
        "pair_kind": pair_kind,
        "registered_classes": list(classes),
        "support_labels": list(labels),
        "opaque_query_ids": list(query_ids),
        "active_k": active_k,
        "qknn_lock_digest": qknn_lock.lock_digest,
        "cache": cache,
    }
    row_input_sha256 = _canonical_sha256(row_binding)
    metric = qknn.identity_shared_psd_metric(config=qknn_lock)
    bank, raw_logits = _build_qknn_logits(
        support_zid=support,
        query_zid=query,
        support_labels=labels,
        classes=classes,
        lock=qknn_lock,
        metric=metric,
    )
    logits = _readonly_logits(raw_logits, rows=len(query), classes=classes, arm_id=qknn_arm_id)
    qknn_arm = D127JointArmPrediction(
        arm_id=qknn_arm_id,
        representation=representation,
        head="phase1_locked_student_t_qknn",
        classes=classes,
        logits=logits,
        predictions=_predictions(logits, classes),
        receipt=_qknn_arm_receipt(
            arm_id=qknn_arm_id,
            row_input_sha256=row_input_sha256,
            cache=cache,
            bank=bank,
            metric=metric,
            classes=classes,
            query_rows=len(query),
        ),
    )
    lite_arm = _make_lite_arm(
        arm_id=lite_arm_id,
        representation=representation,
        support_zid=support,
        query_zid=query,
        support_labels=labels,
        classes=classes,
        opaque_query_ids=query_ids,
        active_k=active_k,
        cache=cache,
        row_input_sha256=row_input_sha256,
        underlying_qknn=qknn_arm,
    )
    receipt = {
        **row_binding,
        "row_input_sha256": row_input_sha256,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "all_registered_classes_scored": True,
        "adaptation_calls_in_joint_interface": 0,
        "arms": {
            arm.arm_id: {
                "logits_sha256": _array_receipt(arm.logits)["sha256"],
                "predictions": list(arm.predictions),
                "receipt_schema": arm.receipt["schema"],
            }
            for arm in (qknn_arm, lite_arm)
        },
    }
    return D127JointTwoArmResult(
        pair_kind=pair_kind,
        qknn_arm=qknn_arm,
        lite_arm=lite_arm,
        receipt=receipt,
    )


def run_d127_common_two_arm(
    *,
    base_support_zid: np.ndarray,
    base_query_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> D127JointTwoArmResult:
    """Compute M0/M_L92 once for one row, reusable across all candidates."""

    return _run_d127_two_arm(
        pair_kind="common",
        support_zid=base_support_zid,
        query_zid=base_query_zid,
        support_labels=support_labels,
        registered_classes=registered_classes,
        opaque_query_ids=opaque_query_ids,
        qknn_lock=qknn_lock,
    )


def run_d127_adapted_two_arm(
    *,
    adapted_support_zid: np.ndarray,
    adapted_query_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> D127JointTwoArmResult:
    """Compute M_DA/M_JOINT for one candidate using one adapted cache."""

    return _run_d127_two_arm(
        pair_kind="adapted",
        support_zid=adapted_support_zid,
        query_zid=adapted_query_zid,
        support_labels=support_labels,
        registered_classes=registered_classes,
        opaque_query_ids=opaque_query_ids,
        qknn_lock=qknn_lock,
    )


def _build_qknn_logits(
    *,
    support_zid: np.ndarray,
    query_zid: np.ndarray,
    support_labels: tuple[str, ...],
    classes: tuple[str, ...],
    lock: qknn.Phase1ZIDStudentTLock,
    metric: qknn.TypedSharedPSDMetric,
) -> tuple[qknn.TypedINT8ZIDSupportBank, np.ndarray]:
    bank = qknn.build_typed_zid_support_bank(
        support_zid,
        support_labels,
        classes,
        config=lock,
    )
    if (
        bank.classes != classes
        or bank.active_k != lock.active_k
        or bank.config_lock_digest != lock.lock_digest
    ):
        raise D127JointScreenError("qKNN bank did not close over the frozen same-row rule")
    logits = qknn.score_zid_student_t_logits(bank, query_zid, metric=metric)
    return bank, logits


def _make_lite_arm(
    *,
    arm_id: str,
    representation: str,
    support_zid: np.ndarray,
    query_zid: np.ndarray,
    support_labels: tuple[str, ...],
    classes: tuple[str, ...],
    opaque_query_ids: tuple[str, ...],
    active_k: int,
    cache: Mapping[str, Any],
    row_input_sha256: str,
    underlying_qknn: D127JointArmPrediction,
) -> D127JointArmPrediction:
    fitted = d92_lite.fit_d92_lite(support_zid, support_labels, classes)
    if fitted.state.classes != classes or fitted.state.active_k != active_k:
        raise D127JointScreenError("D92-Lite fit class/K closure drift")
    alias_receipt: d92_lite.D92LiteQKNNAliasReceipt | None = None
    if active_k == 1:
        if type(fitted.state) is not d92_lite.D92LiteQKNNAlias:
            raise D127JointScreenError("K1 D92-Lite must compile only an exact qKNN alias")
        alias_receipt = d92_lite.make_qknn_alias_receipt(
            classes=classes,
            query_zid=query_zid,
            opaque_query_ids=opaque_query_ids,
            qknn_logits=underlying_qknn.logits,
        )
        score = d92_lite.score_d92_lite(
            fitted.state,
            query_zid,
            qknn_logits=underlying_qknn.logits,
            qknn_alias_receipt=alias_receipt,
        )
        if score.logits is not underlying_qknn.logits:
            raise D127JointScreenError("K1 D92-Lite did not preserve the exact qKNN logits object")
    else:
        if type(fitted.state) is not d92_lite.D92LiteQuantizedLDAState:
            raise D127JointScreenError("K5/K10 D92-Lite must compile an active quantized affine head")
        score = d92_lite.score_d92_lite(fitted.state, query_zid)
    logits = _readonly_logits(
        score.logits,
        rows=len(query_zid),
        classes=classes,
        arm_id=arm_id,
    )
    return D127JointArmPrediction(
        arm_id=arm_id,
        representation=representation,
        head="d92_lite_dr_oas_lda",
        classes=classes,
        logits=logits,
        predictions=_predictions(logits, classes),
        receipt=_lite_arm_receipt(
            arm_id=arm_id,
            row_input_sha256=row_input_sha256,
            cache=cache,
            fitted=fitted,
            score=score,
            classes=classes,
            alias_receipt=alias_receipt,
            underlying_qknn=underlying_qknn,
        ),
        qknn_alias_receipt=alias_receipt,
    )


def run_d127_joint_four_arm(
    *,
    base_support_zid: np.ndarray,
    adapted_support_zid: np.ndarray,
    base_query_zid: np.ndarray,
    adapted_query_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    opaque_query_ids: Sequence[str],
    qknn_lock: qknn.Phase1ZIDStudentTLock,
) -> D127JointFourArmResult:
    """Build and score the frozen same-row four-arm S0 core.

    ``adapted_*`` are caller-owned, already computed feature caches.  This
    function intentionally exposes no adapter/gradient/hook input and calls no
    DA core, so ``M_DA`` and ``M_JOINT`` consume the identical cache objects
    rather than fitting or applying adaptation a second time.
    """

    if type(qknn_lock) is not qknn.Phase1ZIDStudentTLock:
        raise D127JointScreenError("qknn_lock must be an exact frozen Phase1 lock")
    base_support = _zid_rows(base_support_zid, name="base_support_zid")
    adapted_support = _zid_rows(adapted_support_zid, name="adapted_support_zid")
    base_query = _zid_rows(base_query_zid, name="base_query_zid")
    adapted_query = _zid_rows(adapted_query_zid, name="adapted_query_zid")
    if base_support.shape != adapted_support.shape or base_query.shape != adapted_query.shape:
        raise D127JointScreenError(
            "base/adapted support and query caches must have identical same-row shapes"
        )
    classes = _registered_classes(registered_classes)
    labels, active_k = _support_contract(base_support, support_labels, classes, qknn_lock)
    query_ids = _opaque_query_ids(opaque_query_ids, rows=len(base_query))

    base_cache = _cache_receipt(base_support, base_query, representation="base_zid160")
    adapted_cache = _cache_receipt(
        adapted_support, adapted_query, representation="adapted_zid160"
    )
    row_binding = {
        "schema": ROW_RECEIPT_SCHEMA,
        "registered_classes": list(classes),
        "support_labels": list(labels),
        "opaque_query_ids": list(query_ids),
        "active_k": active_k,
        "qknn_lock_digest": qknn_lock.lock_digest,
        "base_cache": base_cache,
        "adapted_cache": adapted_cache,
    }
    row_input_sha256 = _canonical_sha256(row_binding)

    # One identity metric object and one frozen lock define the same qKNN rule
    # for M0 and M_DA.  Their support banks differ only by the supplied cache.
    metric = qknn.identity_shared_psd_metric(config=qknn_lock)
    base_bank, base_logits = _build_qknn_logits(
        support_zid=base_support,
        query_zid=base_query,
        support_labels=labels,
        classes=classes,
        lock=qknn_lock,
        metric=metric,
    )
    adapted_bank, adapted_logits = _build_qknn_logits(
        support_zid=adapted_support,
        query_zid=adapted_query,
        support_labels=labels,
        classes=classes,
        lock=qknn_lock,
        metric=metric,
    )
    m0_logits = _readonly_logits(
        base_logits, rows=len(base_query), classes=classes, arm_id=M0
    )
    m_da_logits = _readonly_logits(
        adapted_logits, rows=len(adapted_query), classes=classes, arm_id=M_DA
    )
    m0 = D127JointArmPrediction(
        arm_id=M0,
        representation="base_zid160",
        head="phase1_locked_student_t_qknn",
        classes=classes,
        logits=m0_logits,
        predictions=_predictions(m0_logits, classes),
        receipt=_qknn_arm_receipt(
            arm_id=M0,
            row_input_sha256=row_input_sha256,
            cache=base_cache,
            bank=base_bank,
            metric=metric,
            classes=classes,
            query_rows=len(base_query),
        ),
    )
    m_da = D127JointArmPrediction(
        arm_id=M_DA,
        representation="adapted_zid160",
        head="phase1_locked_student_t_qknn",
        classes=classes,
        logits=m_da_logits,
        predictions=_predictions(m_da_logits, classes),
        receipt=_qknn_arm_receipt(
            arm_id=M_DA,
            row_input_sha256=row_input_sha256,
            cache=adapted_cache,
            bank=adapted_bank,
            metric=metric,
            classes=classes,
            query_rows=len(adapted_query),
        ),
    )
    m_l92 = _make_lite_arm(
        arm_id=M_L92,
        representation="base_zid160",
        support_zid=base_support,
        query_zid=base_query,
        support_labels=labels,
        classes=classes,
        opaque_query_ids=query_ids,
        active_k=active_k,
        cache=base_cache,
        row_input_sha256=row_input_sha256,
        underlying_qknn=m0,
    )
    m_joint = _make_lite_arm(
        arm_id=M_JOINT,
        representation="adapted_zid160",
        support_zid=adapted_support,
        query_zid=adapted_query,
        support_labels=labels,
        classes=classes,
        opaque_query_ids=query_ids,
        active_k=active_k,
        cache=adapted_cache,
        row_input_sha256=row_input_sha256,
        underlying_qknn=m_da,
    )
    row_receipt = _frozen_mapping(
        {
            **row_binding,
            "row_input_sha256": row_input_sha256,
            "same_row_four_arm_binding": True,
            "same_qknn_lock_and_identity_metric_for_m0_m_da": True,
            "adapted_cache_reused_for_m_da_and_m_joint": True,
            "adaptation_calls_in_joint_interface": 0,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_batch_dependency": False,
            "all_registered_classes_scored": True,
            "arms": {
                arm.arm_id: {
                    "logits_sha256": _array_receipt(arm.logits)["sha256"],
                    "predictions": list(arm.predictions),
                    "receipt_schema": arm.receipt["schema"],
                }
                for arm in (m0, m_da, m_l92, m_joint)
            },
        }
    )
    return D127JointFourArmResult(
        m0=m0,
        m_da=m_da,
        m_l92=m_l92,
        m_joint=m_joint,
        receipt=row_receipt,
    )


__all__ = [
    "ARM_RECEIPT_SCHEMA",
    "D127JointArmPrediction",
    "D127JointFourArmResult",
    "D127JointTwoArmResult",
    "D127JointScreenError",
    "M0",
    "M_DA",
    "M_JOINT",
    "M_L92",
    "ROW_RECEIPT_SCHEMA",
    "S0_ACTIVE_K_VALUES",
    "Z_DIM",
    "run_d127_joint_four_arm",
    "run_d127_common_two_arm",
    "run_d127_adapted_two_arm",
]
