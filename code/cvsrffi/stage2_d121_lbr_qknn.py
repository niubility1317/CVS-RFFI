"""D121 Local Binary Rival qKNN over the frozen Student-t support bank.

This module adds one immutable, support-only rival edge for every support
row.  A query first receives the ordinary M0 per-support Student-t logits;
the edge then applies a single, non-recursive log-sigmoid margin.  There are
no fitted parameters, query-side state updates, role inputs, or truth inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_zid_student_t_qknn as _qknn
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    decode_zid_support_bank,
    normalize_zid_rows,
)


SCHEMA = "cvs.stage2.d121.lbr_qknn.v1"
MAX_SUPPORT_ROWS = 260


class D121LBRQKNNError(ValueError):
    """Raised when the fixed D121 LBR construction or score path drifts."""


class RivalTieUnresolvedError(D121LBRQKNNError):
    """Raised when physical-ID/content-hash tie keys cannot select one rival."""


class ClassScoreTieUnresolvedError(D121LBRQKNNError):
    """Raised when two final class scores are bitwise equal."""


def _canonical_bytes(value: Any) -> bytes:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _same_binary64(left: float, right: float) -> bool:
    return bool(
        np.asarray(np.float64(left)).view(np.uint64)
        == np.asarray(np.float64(right)).view(np.uint64)
    )


def _same_binary32(left: np.float32, right: np.float32) -> bool:
    return bool(
        np.asarray(np.float32(left)).view(np.uint32)
        == np.asarray(np.float32(right)).view(np.uint32)
    )


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D121LBRQKNNError(f"{field} must be a lowercase SHA256")
    return text


def _require_support_physical_ids(value: Sequence[str], *, rows: int) -> tuple[str, ...]:
    result = tuple(value)
    if (
        len(result) != rows
        or len(set(result)) != rows
        or any(type(item) is not str or not item for item in result)
    ):
        raise D121LBRQKNNError(
            "support physical IDs must be unique, non-empty strings in bank order"
        )
    return result


def _default_content_hashes(bank: TypedINT8ZIDSupportBank) -> tuple[str, ...]:
    result = []
    for codes, scale in zip(bank.codes_qint8, bank.scales_fp16, strict=True):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(codes).tobytes(order="C"))
        digest.update(np.ascontiguousarray(scale).tobytes(order="C"))
        result.append(digest.hexdigest())
    return tuple(result)


def _require_content_hashes(
    value: Sequence[str] | None, *, bank: TypedINT8ZIDSupportBank
) -> tuple[str, ...]:
    if value is None:
        return _default_content_hashes(bank)
    result = tuple(_require_sha256(item, field="support content hash") for item in value)
    if len(result) != bank.support_row_count:
        raise D121LBRQKNNError("support content hash count must equal the support bank")
    return result


def _verify_build_inputs(
    bank: TypedINT8ZIDSupportBank, metric: TypedSharedPSDMetric
) -> None:
    if type(bank) is not TypedINT8ZIDSupportBank or type(metric) is not TypedSharedPSDMetric:
        raise D121LBRQKNNError("D121 LBR requires exact typed qKNN bank and metric")
    if bank.config_lock_digest != metric.config_lock_digest:
        raise D121LBRQKNNError("D121 LBR bank/metric Phase1 lock drift")
    if bank.support_row_count < 2 or bank.support_row_count > MAX_SUPPORT_ROWS:
        raise D121LBRQKNNError("D121 LBR support-row bound drift")
    if len(bank.classes) < 2 or any(count != bank.active_k for count in bank.support_counts):
        raise D121LBRQKNNError("D121 LBR requires balanced multi-class support")


def _support_distance_matrix(
    bank: TypedINT8ZIDSupportBank, metric: TypedSharedPSDMetric
) -> np.ndarray:
    support = decode_zid_support_bank(bank).astype(np.float64)
    cosine = _qknn._precision_cosine(support, support, metric)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    if distance.shape != (bank.support_row_count, bank.support_row_count) or not np.isfinite(
        distance
    ).all():
        raise D121LBRQKNNError("D121 LBR support metric distance is invalid")
    return np.ascontiguousarray(distance, dtype=np.float64)


def _build_rival_indices(
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    physical_ids: tuple[str, ...],
    content_hashes: tuple[str, ...],
) -> np.ndarray:
    distance = _support_distance_matrix(bank, metric)
    class_indices = np.asarray(bank.class_indices_int16, dtype=np.int16)
    physical_hashes = tuple(_sha256_text(value) for value in physical_ids)
    rival_indices = np.empty(bank.support_row_count, dtype=np.uint16)

    for index in range(bank.support_row_count):
        foreign = np.flatnonzero(class_indices != class_indices[index])
        if len(foreign) == 0:
            raise D121LBRQKNNError("D121 LBR support row has no foreign-class rival")
        foreign_distance = distance[index, foreign]
        minimum = float(np.min(foreign_distance))
        minimum_bits = np.asarray(np.float64(minimum)).view(np.uint64)
        tied = foreign[
            np.asarray(foreign_distance, dtype=np.float64).view(np.uint64) == minimum_bits
        ]
        keys = {
            int(candidate): (physical_hashes[int(candidate)], content_hashes[int(candidate)])
            for candidate in tied.tolist()
        }
        if len(set(keys.values())) != len(keys):
            raise RivalTieUnresolvedError("RIVAL_TIE_UNRESOLVED")
        rival_indices[index] = np.uint16(min(keys, key=keys.__getitem__))

    if np.any(class_indices[rival_indices.astype(np.int64)] == class_indices):
        raise D121LBRQKNNError("D121 LBR rival graph contains a same-class edge")
    return _readonly(rival_indices, np.uint16)


def _support_identity_root(
    physical_ids: tuple[str, ...], content_hashes: tuple[str, ...]
) -> str:
    return _sha256(
        {
            "schema": SCHEMA + ".support_identity.v1",
            "physical_id_sha256": [_sha256_text(value) for value in physical_ids],
            "content_sha256": list(content_hashes),
        }
    )


@dataclass(frozen=True, slots=True)
class LBRQKNNState:
    """Immutable support graph for one Phase2 registered support bank."""

    classes: tuple[str, ...]
    rival_indices_uint16: np.ndarray
    bank_receipt_sha256: str
    config_lock_digest: str
    metric_receipt_sha256: str
    support_identity_root_sha256: str
    rival_index_root_sha256: str
    resource_receipt: Mapping[str, Any]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        rivals = np.asarray(self.rival_indices_uint16)
        if (
            self.schema != SCHEMA
            or len(self.classes) < 2
            or len(set(self.classes)) != len(self.classes)
            or rivals.dtype != np.uint16
            or rivals.ndim != 1
            or len(rivals) < 2
            or len(rivals) > MAX_SUPPORT_ROWS
            or rivals.flags.writeable
            or not isinstance(self.resource_receipt, Mapping)
        ):
            raise D121LBRQKNNError("D121 LBR state shape/schema drift")
        for field, value in (
            ("bank receipt", self.bank_receipt_sha256),
            ("config lock", self.config_lock_digest),
            ("metric receipt", self.metric_receipt_sha256),
            ("support identity root", self.support_identity_root_sha256),
            ("rival index root", self.rival_index_root_sha256),
        ):
            _require_sha256(value, field=field)
        if int(self.resource_receipt.get("persistent_numeric_bytes", -1)) != int(rivals.nbytes):
            raise D121LBRQKNNError("D121 LBR resource receipt byte drift")


@dataclass(frozen=True, slots=True)
class LBRQKNNScoreTrace:
    """Read-only base and LBR support logits for one independent query batch."""

    base_support_logits_fp64: np.ndarray
    lbr_support_logits_fp64: np.ndarray
    class_logits_fp32: np.ndarray
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        base = np.asarray(self.base_support_logits_fp64)
        lbr = np.asarray(self.lbr_support_logits_fp64)
        classes = np.asarray(self.class_logits_fp32)
        if (
            self.schema != SCHEMA
            or base.dtype != np.float64
            or lbr.dtype != np.float64
            or classes.dtype != np.float32
            or base.ndim != 2
            or lbr.shape != base.shape
            or classes.ndim != 2
            or classes.shape[0] != base.shape[0]
            or not np.isfinite(base).all()
            or not np.isfinite(lbr).all()
            or not np.isfinite(classes).all()
            or base.flags.writeable
            or lbr.flags.writeable
            or classes.flags.writeable
        ):
            raise D121LBRQKNNError("D121 LBR score trace shape/schema drift")


def build_lbr_qknn_state(
    bank: TypedINT8ZIDSupportBank,
    support_physical_ids_in_bank_order: Sequence[str],
    *,
    metric: TypedSharedPSDMetric,
    support_content_hashes_in_bank_order: Sequence[str] | None = None,
) -> LBRQKNNState:
    """Compile fixed foreign-class rival edges using support data only.

    ``support_physical_ids_in_bank_order`` must use the exact canonical support
    row order already stored in ``bank``.  Query data is intentionally absent.
    """

    _verify_build_inputs(bank, metric)
    physical_ids = _require_support_physical_ids(
        support_physical_ids_in_bank_order, rows=bank.support_row_count
    )
    content_hashes = _require_content_hashes(
        support_content_hashes_in_bank_order, bank=bank
    )
    rivals = _build_rival_indices(bank, metric, physical_ids, content_hashes)
    identity_root = _support_identity_root(physical_ids, content_hashes)
    rival_root = _sha256(
        {
            "schema": SCHEMA + ".rival_graph.v1",
            "bank_receipt_sha256": bank.bank_receipt_sha256,
            "metric_receipt_sha256": metric.metric_receipt_sha256,
            "support_identity_root_sha256": identity_root,
            "rival_indices_uint16": [int(value) for value in rivals],
        }
    )
    resource = MappingProxyType(
        {
            "support_row_count": int(bank.support_row_count),
            "rival_index_dtype": "uint16",
            "persistent_numeric_bytes": int(rivals.nbytes),
            "max_persistent_numeric_bytes_at_n260": 520,
            "enrollment_distance_mac_upper_bound": int(
                bank.support_row_count
                * (bank.support_row_count - 1)
                * _qknn.Z_DIM
            ),
            "extra_query_scalar_operations": int(bank.support_row_count),
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "parameter_scan_count": 0,
        }
    )
    return LBRQKNNState(
        classes=tuple(bank.classes),
        rival_indices_uint16=rivals,
        bank_receipt_sha256=bank.bank_receipt_sha256,
        config_lock_digest=bank.config_lock_digest,
        metric_receipt_sha256=metric.metric_receipt_sha256,
        support_identity_root_sha256=identity_root,
        rival_index_root_sha256=rival_root,
        resource_receipt=resource,
    )


def _verify_score_inputs(
    state: LBRQKNNState,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
) -> None:
    _verify_build_inputs(bank, metric)
    if type(state) is not LBRQKNNState:
        raise D121LBRQKNNError("D121 LBR scoring requires an exact state")
    rivals = np.asarray(state.rival_indices_uint16)
    if (
        state.classes != tuple(bank.classes)
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
        or state.metric_receipt_sha256 != metric.metric_receipt_sha256
        or rivals.shape != (bank.support_row_count,)
        or np.any(rivals.astype(np.int64) >= bank.support_row_count)
    ):
        raise D121LBRQKNNError("D121 LBR state/bank/metric binding drift")
    classes = np.asarray(bank.class_indices_int16, dtype=np.int16)
    if np.any(classes[rivals.astype(np.int64)] == classes):
        raise D121LBRQKNNError("D121 LBR state contains same-class rival edges")


def _base_support_student_t_logits(
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
    metric: TypedSharedPSDMetric,
) -> np.ndarray:
    query = normalize_zid_rows(query_zid).astype(np.float64)
    support = decode_zid_support_bank(bank).astype(np.float64)
    cosine = _qknn._precision_cosine(query, support, metric)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    class_scales = np.asarray(bank.class_scales_fp16, dtype=np.float64)
    class_indices = np.asarray(bank.class_indices_int16, dtype=np.int16)
    kernel = np.empty_like(distance, dtype=np.float64)
    for class_index, expected in enumerate(bank.support_counts):
        mask = class_indices == class_index
        if int(np.count_nonzero(mask)) != expected:
            raise D121LBRQKNNError("D121 LBR support class count drift")
        scale = float(class_scales[class_index])
        kernel[:, mask] = (
            -bank.config.kernel_volume_gamma
            * bank.config.kernel_effective_dim
            * math.log(scale)
            - 0.5
            * (bank.config.student_nu + bank.config.kernel_effective_dim)
            * np.log1p(
                distance[:, mask]
                / (bank.config.student_nu * scale * scale)
            )
        )
    if not np.isfinite(kernel).all():
        raise D121LBRQKNNError("D121 LBR base support logits became non-finite")
    return np.ascontiguousarray(kernel, dtype=np.float64)


def _class_logsumexp_minus_log_k(
    support_logits: np.ndarray, bank: TypedINT8ZIDSupportBank
) -> np.ndarray:
    values = np.asarray(support_logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != bank.support_row_count:
        raise D121LBRQKNNError("D121 LBR support-logit shape drift")
    result = np.empty((len(values), len(bank.classes)), dtype=np.float64)
    indices = np.asarray(bank.class_indices_int16, dtype=np.int16)
    for class_index, expected in enumerate(bank.support_counts):
        local = values[:, indices == class_index]
        if local.shape[1] != expected:
            raise D121LBRQKNNError("D121 LBR class aggregation support count drift")
        maximum = np.max(local, axis=1, keepdims=True)
        result[:, class_index] = (
            maximum[:, 0]
            + np.log(np.sum(np.exp(local - maximum), axis=1))
            - math.log(expected)
        )
    if not np.isfinite(result).all():
        raise D121LBRQKNNError("D121 LBR class logits became non-finite")
    return np.ascontiguousarray(result, dtype=np.float64)


def unique_lbr_argmax(class_logits: np.ndarray, classes: Sequence[str]) -> tuple[str, ...]:
    """Return decisions only when every final row has one bitwise winner."""

    scores = np.asarray(class_logits)
    registry = tuple(classes)
    if (
        scores.dtype != np.float32
        or scores.ndim != 2
        or scores.shape[0] < 1
        or scores.shape[1] != len(registry)
        or len(registry) < 2
        or len(set(registry)) != len(registry)
        or not np.isfinite(scores).all()
    ):
        raise D121LBRQKNNError("D121 LBR final class-score layout drift")
    output: list[str] = []
    for row in scores:
        maximum = np.max(row)
        winners = [
            index
            for index, value in enumerate(row)
            if _same_binary32(np.float32(value), np.float32(maximum))
        ]
        if len(winners) != 1:
            raise ClassScoreTieUnresolvedError("CLASS_SCORE_TIE_UNRESOLVED")
        output.append(registry[winners[0]])
    return tuple(output)


def score_lbr_qknn_trace(
    state: LBRQKNNState,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
    *,
    metric: TypedSharedPSDMetric,
) -> LBRQKNNScoreTrace:
    """Score independent queries with base logits computed once and read once."""

    _verify_score_inputs(state, bank, metric)
    base = _base_support_student_t_logits(bank, query_zid, metric)
    rivals = np.asarray(state.rival_indices_uint16, dtype=np.int64)
    # This is l_i - log(1 + exp(l_n_i - l_i)); no corrected logit is read back.
    corrected = base - np.logaddexp(0.0, base[:, rivals] - base)
    class_logits = _class_logsumexp_minus_log_k(corrected, bank).astype(np.float32)
    class_logits = _readonly(class_logits, np.float32)
    unique_lbr_argmax(class_logits, state.classes)
    return LBRQKNNScoreTrace(
        base_support_logits_fp64=_readonly(base, np.float64),
        lbr_support_logits_fp64=_readonly(corrected, np.float64),
        class_logits_fp32=class_logits,
    )


def score_lbr_qknn_logits(
    state: LBRQKNNState,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
    *,
    metric: TypedSharedPSDMetric,
) -> np.ndarray:
    """Return only the frozen D121 class-score matrix."""

    return score_lbr_qknn_trace(state, bank, query_zid, metric=metric).class_logits_fp32


def audit_lbr_qknn_state(state: LBRQKNNState) -> Mapping[str, Any]:
    """Expose compact support-state facts without predictions or query data."""

    if type(state) is not LBRQKNNState:
        raise D121LBRQKNNError("D121 LBR audit requires an exact state")
    return MappingProxyType(
        {
            "schema": state.schema,
            "support_row_count": int(len(state.rival_indices_uint16)),
            "rival_index_root_sha256": state.rival_index_root_sha256,
            "support_identity_root_sha256": state.support_identity_root_sha256,
            "resource_receipt": dict(state.resource_receipt),
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }
    )


__all__ = [
    "ClassScoreTieUnresolvedError",
    "D121LBRQKNNError",
    "LBRQKNNScoreTrace",
    "LBRQKNNState",
    "MAX_SUPPORT_ROWS",
    "RivalTieUnresolvedError",
    "SCHEMA",
    "audit_lbr_qknn_state",
    "build_lbr_qknn_state",
    "score_lbr_qknn_logits",
    "score_lbr_qknn_trace",
    "unique_lbr_argmax",
]
