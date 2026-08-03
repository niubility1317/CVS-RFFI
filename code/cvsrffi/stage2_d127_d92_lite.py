"""Support-only D92-Lite diagonal OAS-form LDA for normalized ``z_id160``.

The active K5/K10 path only compiles a class-symmetric INT8/FP16 affine head.
K1 deliberately compiles no LDA parameters: its state is an explicit alias to
the caller's already-computed qKNN logits.  Query rows never enter fitting and
scoring never changes either kind of state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


Z_DIM = 160
INT8_MAX = 127.0
ACTIVE_K_VALUES = frozenset({5, 10})
STATE_SCHEMA = "cvs.stage2.d127.d92_lite.dr_oas_lda.v1"
ALIAS_SCHEMA = "cvs.stage2.d127.d92_lite.qknn_alias.v1"
ALIAS_RECEIPT_SCHEMA = "cvs.stage2.d127.d92_lite.qknn_alias_receipt.v1"


class D92LiteError(ValueError):
    """Raised when the frozen support-only D92-Lite contract is violated."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    registry = tuple(str(value) for value in values)
    if (
        len(registry) < 2
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
    ):
        raise D92LiteError("registry must contain at least two unique non-empty classes")
    return registry


def _finite_zid160_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D92LiteError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(rows)


def normalize_zid160_rows(value: np.ndarray) -> np.ndarray:
    """Return immutable L2-normalized float32 z_id160 rows in the same order."""

    rows = _finite_zid160_rows(value, name="z_id160 rows").astype(np.float64)
    norms = np.sqrt(np.sum(rows * rows, axis=1, keepdims=True))
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise D92LiteError("z_id160 rows must have positive finite L2 norm")
    return _readonly(rows / norms, np.float32)


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _opaque_query_ids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result) or any(not value for value in result):
        raise D92LiteError("opaque query IDs must be unique non-empty values")
    return result


def _query_binding_sha256(query_zid: np.ndarray, opaque_query_ids: tuple[str, ...]) -> str:
    return _canonical_digest(
        {
            "opaque_query_ids": list(opaque_query_ids),
            "query_zid": _array_digest(query_zid),
        }
    )


def _require_sha256(value: str, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise D92LiteError(f"{field} must be a lowercase SHA256")
    return digest


@dataclass(frozen=True, slots=True)
class D92LiteQKNNAlias:
    """K1 state: exactly reuse qKNN logits rather than fitting a head."""

    classes: tuple[str, ...]
    active_k: int = 1
    schema: str = ALIAS_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        if self.schema != ALIAS_SCHEMA or int(self.active_k) != 1:
            raise D92LiteError("K1 alias state schema/K drift")
        object.__setattr__(self, "classes", classes)

    @property
    def numeric_state_bytes(self) -> int:
        """D92-Lite adds no deployment array when K1 is a qKNN alias."""

        return 0

    @property
    def state_receipt_sha256(self) -> str:
        return _canonical_digest(
            {
                "active_k": 1,
                "classes": list(self.classes),
                "schema": self.schema,
            }
        )


@dataclass(frozen=True, slots=True)
class D92LiteQKNNAliasReceipt:
    """Immutable binding of qKNN logits to class columns and query row order."""

    classes: tuple[str, ...]
    opaque_query_ids: tuple[str, ...]
    query_binding_sha256: str
    qknn_logits_sha256: str
    schema: str = ALIAS_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        query_ids = _opaque_query_ids(self.opaque_query_ids)
        if self.schema != ALIAS_RECEIPT_SCHEMA:
            raise D92LiteError("qKNN alias receipt schema drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "opaque_query_ids", query_ids)
        object.__setattr__(
            self,
            "query_binding_sha256",
            _require_sha256(self.query_binding_sha256, field="query binding"),
        )
        object.__setattr__(
            self,
            "qknn_logits_sha256",
            _require_sha256(self.qknn_logits_sha256, field="qKNN logits"),
        )


@dataclass(frozen=True, slots=True)
class D92LiteQuantizedLDAState:
    """The entire K5/K10 deployment numeric state: INT8 W plus FP16 scale/b."""

    classes: tuple[str, ...]
    active_k: int
    q_int8: np.ndarray
    scale_fp16: np.ndarray
    intercept_fp16: np.ndarray
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        classes = _registry(self.classes)
        codes = np.asarray(self.q_int8)
        scales = np.asarray(self.scale_fp16)
        intercept = np.asarray(self.intercept_fp16)
        class_count = len(classes)
        if (
            self.schema != STATE_SCHEMA
            or int(self.active_k) not in ACTIVE_K_VALUES
            or codes.dtype != np.int8
            or codes.shape != (class_count, Z_DIM)
            or np.any(codes == np.int8(-128))
            or scales.dtype != np.float16
            or scales.shape != (class_count,)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
            or intercept.dtype != np.float16
            or intercept.shape != (class_count,)
            or not np.isfinite(intercept).all()
        ):
            raise D92LiteError("D92-Lite deployment state shape/dtype/range drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "active_k", int(self.active_k))
        object.__setattr__(self, "q_int8", _readonly(codes, np.int8))
        object.__setattr__(self, "scale_fp16", _readonly(scales, np.float16))
        object.__setattr__(self, "intercept_fp16", _readonly(intercept, np.float16))

    @property
    def numeric_state_bytes(self) -> int:
        return int(
            self.q_int8.nbytes
            + self.scale_fp16.nbytes
            + self.intercept_fp16.nbytes
        )

    @property
    def state_receipt_sha256(self) -> str:
        return _canonical_digest(
            {
                "active_k": self.active_k,
                "classes": list(self.classes),
                "intercept_fp16": _array_digest(self.intercept_fp16),
                "q_int8": _array_digest(self.q_int8),
                "scale_fp16": _array_digest(self.scale_fp16),
                "schema": self.schema,
            }
        )


D92LiteState = D92LiteQKNNAlias | D92LiteQuantizedLDAState


@dataclass(frozen=True, slots=True)
class D92LiteFit:
    """Composable support-only fit result with non-deployment audit receipts."""

    state: D92LiteState
    fit_receipt: Mapping[str, Any]
    resource_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.state) not in {
            D92LiteQKNNAlias,
            D92LiteQuantizedLDAState,
        }:
            raise D92LiteError("D92-Lite fit must contain an exact state type")
        object.__setattr__(self, "fit_receipt", MappingProxyType(dict(self.fit_receipt)))
        object.__setattr__(
            self, "resource_receipt", MappingProxyType(dict(self.resource_receipt))
        )


@dataclass(frozen=True, slots=True)
class D92LiteScore:
    """Independent-query logits and the corresponding zero-update receipt."""

    logits: np.ndarray
    score_receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits)
        if (
            logits.dtype != np.float32
            or logits.ndim != 2
            or logits.shape[0] < 1
            or logits.shape[1] < 2
            or not np.isfinite(logits).all()
        ):
            raise D92LiteError("D92-Lite logits must be finite float32 [N,C]")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "score_receipt", MappingProxyType(dict(self.score_receipt)))


def _support_registry(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int]:
    rows = _finite_zid160_rows(support_zid, name="support_zid")
    registry = _registry(classes)
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(rows) or any(value not in registry for value in labels):
        raise D92LiteError("support labels must close exactly over the registered classes")
    class_to_index = {value: index for index, value in enumerate(registry)}
    indices = np.asarray([class_to_index[value] for value in labels], dtype=np.int64)
    counts = np.bincount(indices, minlength=len(registry))
    if np.any(counts < 1) or len(set(int(value) for value in counts)) != 1:
        raise D92LiteError("all registered classes require the same positive K-shot support")
    active_k = int(counts[0])
    if len(rows) != len(registry) * active_k:
        raise D92LiteError("support registry row count/K closure drift")
    return rows, registry, indices, active_k


def _resource_receipt(state: D92LiteState) -> dict[str, Any]:
    class_count = len(state.classes)
    if type(state) is D92LiteQKNNAlias:
        return {
            "schema": "cvs.stage2.d127.d92_lite.resource.v1",
            "active_k": 1,
            "class_count": class_count,
            "feature_dim": Z_DIM,
            "fit_mode": "exact_qknn_alias",
            # These zeros are only the additional D92-Lite cost.  The qKNN
            # state and MACs remain mandatory, caller-owned formal receipts.
            "d92_lite_incremental_deployed_numeric_state_bytes": 0,
            "d92_lite_incremental_state_formula": "0_B_for_K1_qknn_alias",
            "d92_lite_incremental_query_macs_per_sample": 0,
            "d92_lite_incremental_query_state_bytes": 0,
            "underlying_qknn_resource_required": True,
            "underlying_qknn_resource_included": False,
            "underlying_qknn_resource_receipt_binding": "caller_formal_receipt_required",
            "dense_matrix_elements_constructed": 0,
            "spectral_factorization_count": 0,
            "linear_system_solve_count": 0,
        }
    deployed = state.numeric_state_bytes
    expected = 164 * class_count
    if deployed != expected:
        raise D92LiteError("D92-Lite 164C deployment byte formula drift")
    return {
        "schema": "cvs.stage2.d127.d92_lite.resource.v1",
        "active_k": state.active_k,
        "class_count": class_count,
        "feature_dim": Z_DIM,
        "fit_mode": "diagonal_oas_form",
        "weight_int8_bytes": int(state.q_int8.nbytes),
        "scale_fp16_bytes": int(state.scale_fp16.nbytes),
        "intercept_fp16_bytes": int(state.intercept_fp16.nbytes),
        "deployed_numeric_state_bytes": deployed,
        "d92_lite_incremental_state_formula": "160C+2C+2C=164C_B",
        "query_head_macs_per_sample": Z_DIM * class_count,
        "dense_matrix_elements_constructed": 0,
        "spectral_factorization_count": 0,
        "linear_system_solve_count": 0,
        "query_state_bytes": 0,
    }


def _compile_diagonal_oas_state(
    normalized_support: np.ndarray,
    registry: tuple[str, ...],
    class_indices: np.ndarray,
    active_k: int,
) -> tuple[D92LiteQuantizedLDAState, dict[str, Any]]:
    """Fit the frozen diagonal OAS-form expression without a dense matrix path."""

    rows64 = np.asarray(normalized_support, dtype=np.float64)
    class_count = len(registry)
    means = np.stack(
        [rows64[class_indices == index].mean(axis=0) for index in range(class_count)]
    )
    residuals = rows64 - means[class_indices]
    n_eff = class_count * (active_k - 1)
    scatter = np.sum(residuals * residuals, axis=0, dtype=np.float64) / float(n_eff)
    total = float(np.sum(scatter, dtype=np.float64))
    second_moment = float(np.sum(scatter * scatter, dtype=np.float64))
    tau = total / float(Z_DIM)
    delta = second_moment - total * total / float(Z_DIM)
    if (
        not np.isfinite(scatter).all()
        or not all(math.isfinite(value) for value in (total, second_moment, tau, delta))
        or total <= 0.0
    ):
        raise D92LiteError("P0: diagonal OAS-form support statistics are non-finite or t<=0")
    if delta <= 0.0:
        shrinkage = 1.0
    else:
        numerator = (1.0 - 2.0 / Z_DIM) * second_moment + total * total
        denominator = (float(n_eff) + 1.0 - 2.0 / Z_DIM) * delta
        shrinkage = min(1.0, numerator / denominator)
    if not math.isfinite(shrinkage) or shrinkage < 0.0:
        raise D92LiteError("P0: diagonal OAS-form shrinkage is invalid")
    variance_floor = max(
        float(np.finfo(np.float64).tiny),
        float(np.finfo(np.float64).eps) * max(1.0, tau),
    )
    variance = np.maximum(
        (1.0 - shrinkage) * scatter + shrinkage * tau,
        variance_floor,
    )
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise D92LiteError("P0: diagonal OAS-form variance floor closure failed")
    weights = means / variance[None, :]
    intercepts = -0.5 * np.sum(means * means / variance[None, :], axis=1)
    weights -= weights.mean(axis=0, keepdims=True)
    intercepts -= intercepts.mean()
    if not np.isfinite(weights).all() or not np.isfinite(intercepts).all():
        raise D92LiteError("P0: affine head became non-finite before quantization")

    scale_floor = float(np.finfo(np.float16).tiny)
    scale_limit = float(np.finfo(np.float16).max)
    maximum = np.max(np.abs(weights), axis=1)
    scales64 = np.maximum(maximum / INT8_MAX, scale_floor)
    if not np.isfinite(scales64).all() or np.any(scales64 > scale_limit):
        raise D92LiteError("P0: per-class FP16 quantization scale is not representable")
    scales = scales64.astype(np.float16)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise D92LiteError("P0: per-class FP16 quantization scale underflowed")
    codes = np.clip(
        np.rint(weights / scales.astype(np.float64)[:, None]), -INT8_MAX, INT8_MAX
    ).astype(np.int8)
    intercept16 = intercepts.astype(np.float16)
    if not np.isfinite(intercept16).all():
        raise D92LiteError("P0: FP16 intercept is not representable")
    state = D92LiteQuantizedLDAState(
        classes=registry,
        active_k=active_k,
        q_int8=codes,
        scale_fp16=scales,
        intercept_fp16=intercept16,
    )
    receipt = {
        "schema": "cvs.stage2.d127.d92_lite.fit.v1",
        "fit_mode": "diagonal_oas_form",
        "feature_dim": Z_DIM,
        "class_count": class_count,
        "active_k": active_k,
        "support_rows": int(len(normalized_support)),
        "residual_degrees_of_freedom": n_eff,
        "shrinkage": float(shrinkage),
        "variance_floor": variance_floor,
        "support_only": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "same_formula_all_registered_classes": True,
        "class_label_permutation_equivariant": True,
        "common_affine_centered_before_quantization": True,
        "prequantized_weight_class_mean_max_abs": float(
            np.max(np.abs(weights.mean(axis=0)))
        ),
        "prequantized_intercept_class_mean_abs": float(abs(intercepts.mean())),
        "dense_matrix_elements_constructed": 0,
        "spectral_factorization_count": 0,
        "linear_system_solve_count": 0,
        "state_receipt_sha256": state.state_receipt_sha256,
    }
    return state, receipt


def fit_d92_lite(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
) -> D92LiteFit:
    """Fit a K5/K10 head from target support, or make the strict K1 qKNN alias.

    The function intentionally has no query, truth, role, quota, or global
    assignment input.  K1 returns before any diagonal statistic is computed.
    """

    rows, registry, class_indices, active_k = _support_registry(
        support_zid, support_labels, classes
    )
    if active_k == 1:
        state = D92LiteQKNNAlias(classes=registry)
        receipt = {
            "schema": "cvs.stage2.d127.d92_lite.fit.v1",
            "fit_mode": "exact_qknn_alias",
            "feature_dim": Z_DIM,
            "class_count": len(registry),
            "active_k": 1,
            "support_rows": int(len(rows)),
            "support_only": True,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "same_formula_all_registered_classes": True,
            "class_label_permutation_equivariant": True,
            "diagonal_statistics_computed": False,
            "deployed_lda_arrays": False,
            "state_receipt_sha256": state.state_receipt_sha256,
        }
        return D92LiteFit(
            state=state,
            fit_receipt=receipt,
            resource_receipt=_resource_receipt(state),
        )
    if active_k not in ACTIVE_K_VALUES:
        raise D92LiteError("D92-Lite activates only for K1, K5, or K10")
    normalized = normalize_zid160_rows(rows)
    state, receipt = _compile_diagonal_oas_state(
        normalized, registry, class_indices, active_k
    )
    return D92LiteFit(
        state=state,
        fit_receipt=receipt,
        resource_receipt=_resource_receipt(state),
    )


def _alias_logits(
    logits: np.ndarray, *, query_count: int, class_count: int
) -> np.ndarray:
    value = np.asarray(logits)
    if (
        value.dtype != np.float32
        or value.shape != (query_count, class_count)
        or not np.isfinite(value).all()
    ):
        raise D92LiteError("K1 qKNN alias logits must be finite float32 [N,C]")
    return value


def make_qknn_alias_receipt(
    *,
    classes: Sequence[str],
    query_zid: np.ndarray,
    opaque_query_ids: Sequence[str],
    qknn_logits: np.ndarray,
) -> D92LiteQKNNAliasReceipt:
    """Bind a qKNN result to exact class columns and opaque query row identity.

    This is an ephemeral caller-side receipt, never part of D92-Lite deployment
    state.  It records neither query truth nor any role/quota information.
    """

    registry = _registry(classes)
    raw_query = _finite_zid160_rows(query_zid, name="query_zid")
    query_ids = _opaque_query_ids(opaque_query_ids)
    if len(query_ids) != len(raw_query):
        raise D92LiteError("opaque query ID/query row count drift")
    logits = _alias_logits(
        qknn_logits,
        query_count=len(raw_query),
        class_count=len(registry),
    )
    return D92LiteQKNNAliasReceipt(
        classes=registry,
        opaque_query_ids=query_ids,
        query_binding_sha256=_query_binding_sha256(raw_query, query_ids),
        qknn_logits_sha256=_array_digest(logits)["sha256"],
    )


def _bound_alias_logits(
    state: D92LiteQKNNAlias,
    query_zid: np.ndarray,
    qknn_logits: np.ndarray,
    receipt: D92LiteQKNNAliasReceipt,
) -> np.ndarray:
    if type(receipt) is not D92LiteQKNNAliasReceipt:
        raise D92LiteError("K1 qKNN alias requires an exact immutable alias receipt")
    if receipt.classes != state.classes:
        raise D92LiteError("K1 qKNN alias class-column order drift")
    if len(receipt.opaque_query_ids) != len(query_zid):
        raise D92LiteError("K1 qKNN alias query identity/order length drift")
    logits = _alias_logits(
        qknn_logits,
        query_count=len(query_zid),
        class_count=len(state.classes),
    )
    expected_query = _query_binding_sha256(query_zid, receipt.opaque_query_ids)
    if receipt.query_binding_sha256 != expected_query:
        raise D92LiteError("K1 qKNN alias query identity/order binding drift")
    if receipt.qknn_logits_sha256 != _array_digest(logits)["sha256"]:
        raise D92LiteError("K1 qKNN alias logits binding drift")
    return logits


def score_d92_lite_logits(
    state: D92LiteState,
    query_zid: np.ndarray,
    *,
    qknn_logits: np.ndarray | None = None,
    qknn_alias_receipt: D92LiteQKNNAliasReceipt | None = None,
) -> np.ndarray:
    """Score independent query rows over all classes without changing state.

    A K1 state returns the supplied qKNN logits object unchanged only after its
    immutable receipt binds exact class columns and opaque query row order.
    """

    if type(state) not in {D92LiteQKNNAlias, D92LiteQuantizedLDAState}:
        raise D92LiteError("scoring requires an exact D92-Lite state")
    raw_query = _finite_zid160_rows(query_zid, name="query_zid")
    if type(state) is D92LiteQKNNAlias:
        if qknn_logits is None or qknn_alias_receipt is None:
            raise D92LiteError(
                "K1 D92-Lite requires exact qKNN logits and an alias receipt"
            )
        return _bound_alias_logits(
            state, raw_query, qknn_logits, qknn_alias_receipt
        )
    if qknn_logits is not None or qknn_alias_receipt is not None:
        raise D92LiteError("K5/K10 D92-Lite cannot accept qKNN alias data")
    query = normalize_zid160_rows(raw_query).astype(np.float64)
    dot_products = query @ state.q_int8.astype(np.float64).T
    logits = (
        dot_products * state.scale_fp16.astype(np.float64)[None, :]
        + state.intercept_fp16.astype(np.float64)[None, :]
    )
    if not np.isfinite(logits).all():
        raise D92LiteError("D92-Lite INT8/FP16 affine scoring became non-finite")
    return _readonly(logits, np.float32)


def score_d92_lite(
    state: D92LiteState,
    query_zid: np.ndarray,
    *,
    qknn_logits: np.ndarray | None = None,
    qknn_alias_receipt: D92LiteQKNNAliasReceipt | None = None,
) -> D92LiteScore:
    """Return logits together with a per-sample, zero-update score receipt."""

    logits = score_d92_lite_logits(
        state,
        query_zid,
        qknn_logits=qknn_logits,
        qknn_alias_receipt=qknn_alias_receipt,
    )
    return D92LiteScore(
        logits=logits,
        score_receipt={
            "schema": "cvs.stage2.d127.d92_lite.score.v1",
            "state_receipt_sha256": state.state_receipt_sha256,
            "query_rows": int(len(logits)),
            "class_count": len(state.classes),
            "fit_mode": (
                "exact_qknn_alias"
                if type(state) is D92LiteQKNNAlias
                else "diagonal_oas_form"
            ),
            "all_registered_classes_scored": True,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_batch_dependency": False,
        },
    )


def d92_lite_resource_receipt(state: D92LiteState) -> Mapping[str, Any]:
    """Expose an immutable receipt for the exact deployment resource layout."""

    if type(state) not in {D92LiteQKNNAlias, D92LiteQuantizedLDAState}:
        raise D92LiteError("resource receipt requires an exact D92-Lite state")
    return MappingProxyType(_resource_receipt(state))


__all__ = [
    "ACTIVE_K_VALUES",
    "ALIAS_RECEIPT_SCHEMA",
    "ALIAS_SCHEMA",
    "D92LiteError",
    "D92LiteFit",
    "D92LiteQKNNAlias",
    "D92LiteQKNNAliasReceipt",
    "D92LiteQuantizedLDAState",
    "D92LiteScore",
    "D92LiteState",
    "INT8_MAX",
    "STATE_SCHEMA",
    "Z_DIM",
    "d92_lite_resource_receipt",
    "fit_d92_lite",
    "make_qknn_alias_receipt",
    "normalize_zid160_rows",
    "score_d92_lite",
    "score_d92_lite_logits",
]
