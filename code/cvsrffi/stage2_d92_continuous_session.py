"""Frozen-DA, cumulative-support D92 continuous registration core.

This module deliberately owns only support-time state transition.  A caller
must inject the already-frozen E0/D81 support transform when creating the DA
anchor; there is no production identity-transform default.  Every successful
arrival rebuilds one support-only D92 FULL affine head and publishes it through
one D42 residual-int8 codec pass.  Query, truth, role, quota, and global
assignment inputs do not exist in this API.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d92_registration_balanced_covariance as d92


FEATURE_DIM = int(d42.FEATURE_DIM)
OLD_CLASS_COUNT = int(d92.OLD_CLASS_COUNT)
K_SHOT = 10
MAX_NEW_CLASSES = 5
SCHEMA = "cvs.phase2.d92.continuous_session.d42_residual_int8.v1"
METHOD_ID = "D92_E0_CUMULATIVE_REPLAY_SESSION_V1"

SupportTransform = Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]


class D92ContinuousSessionError(RuntimeError):
    """Raised when an immutable continuous-registration transition drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _canonical_row_key(row: np.ndarray) -> tuple[bytes, bytes]:
    """Match D92's float32-first, float64-tie-break support ordering."""

    row64 = np.ascontiguousarray(np.asarray(row, dtype=np.float64))
    row32 = np.ascontiguousarray(np.asarray(row64, dtype=np.float32))
    return row32.tobytes(order="C"), row64.tobytes(order="C")


def _callable_identity(transform: SupportTransform) -> str:
    module = str(getattr(transform, "__module__", ""))
    qualname = str(
        getattr(transform, "__qualname__", getattr(transform, "__name__", ""))
    )
    identity = f"{module}:{qualname}".strip(":")
    if not identity:
        raise D92ContinuousSessionError("frozen DA support transform identity missing")
    return identity


@dataclass(frozen=True)
class SupportPacket:
    """One immutable K10 class package, unopened until its arrival session."""

    handle: str
    rows: Any
    physical_tokens: Sequence[str]
    package_id: str
    arrival_session: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.handle, str)
            or not self.handle
            or not isinstance(self.package_id, str)
            or not self.package_id
            or isinstance(self.physical_tokens, (str, bytes))
            or not isinstance(self.arrival_session, (int, np.integer))
            or isinstance(self.arrival_session, (bool, np.bool_))
            or int(self.arrival_session) < 0
        ):
            raise D92ContinuousSessionError("continuous support packet metadata drift")


@dataclass(frozen=True)
class _SupportRecord:
    handle: str
    rows: np.ndarray
    physical_tokens: tuple[str, ...]
    package_id: str


def _materialize_packet(packet: SupportPacket) -> _SupportRecord:
    """Open a packet only after its session metadata has been accepted."""

    if not isinstance(packet, SupportPacket):
        raise D92ContinuousSessionError("continuous support packet type drift")
    tokens = tuple(packet.physical_tokens)
    if (
        len(tokens) != K_SHOT
        or any(not isinstance(token, str) or not token for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise D92ContinuousSessionError("duplicate physical support token")
    rows = np.asarray(packet.rows, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape != (K_SHOT, FEATURE_DIM)
        or not np.isfinite(rows).all()
    ):
        raise D92ContinuousSessionError(
            f"continuous support packet must be finite float32 [{K_SHOT},{FEATURE_DIM}]"
        )
    order = sorted(
        range(K_SHOT), key=lambda index: (_canonical_row_key(rows[index]), tokens[index])
    )
    ordered = np.asarray(rows[np.asarray(order, dtype=np.int64)], dtype=np.float32)
    ordered_tokens = tuple(tokens[index] for index in order)
    return _SupportRecord(
        handle=packet.handle,
        rows=_readonly(ordered, np.float32),
        physical_tokens=ordered_tokens,
        package_id=packet.package_id,
    )


def _canonical_records(records: Iterable[_SupportRecord]) -> tuple[_SupportRecord, ...]:
    result = tuple(sorted(tuple(records), key=lambda record: record.handle))
    handles = tuple(record.handle for record in result)
    if len(result) != len(set(handles)):
        raise D92ContinuousSessionError("continuous support class handle repeated")
    return result


@dataclass(frozen=True)
class FrozenDAAnchor:
    """The immutable old registry and injected E0/D81 support transform."""

    old_records: tuple[_SupportRecord, ...]
    da_anchor_id: str
    log_diag_fp32: np.ndarray
    support_transform: SupportTransform
    support_transform_identity: str

    def __post_init__(self) -> None:
        log_diag = np.asarray(self.log_diag_fp32)
        if (
            log_diag.dtype != np.float32
            or log_diag.shape != (FEATURE_DIM,)
            or not np.isfinite(log_diag).all()
        ):
            raise D92ContinuousSessionError("frozen DA log_diag must be finite float32[288]")
        object.__setattr__(self, "log_diag_fp32", _readonly(log_diag, np.float32))

    @classmethod
    def from_old_support(
        cls,
        packets: Sequence[SupportPacket],
        *,
        da_anchor_id: str,
        support_transform: SupportTransform,
        log_diag_fp32: np.ndarray,
        support_transform_identity: str | None = None,
    ) -> "FrozenDAAnchor":
        """Freeze the already-available E0 old registry and D81 transform.

        ``support_transform`` is mandatory.  Real callers pass the frozen E0
        transform (including its ground robust-center state); tests may inject
        a transparent fixture only at this boundary.
        """

        if not isinstance(da_anchor_id, str) or not da_anchor_id:
            raise D92ContinuousSessionError("frozen DA anchor identity missing")
        if not callable(support_transform):
            raise D92ContinuousSessionError("frozen DA support transform missing")
        supplied = tuple(packets)
        if len(supplied) != OLD_CLASS_COUNT:
            raise D92ContinuousSessionError("frozen DA anchor requires six old classes")
        if any(packet.arrival_session != 0 for packet in supplied):
            raise D92ContinuousSessionError("old support must belong to DA anchor session 0")
        records = _canonical_records(_materialize_packet(packet) for packet in supplied)
        tokens = tuple(
            token for record in records for token in record.physical_tokens
        )
        if len(tokens) != len(set(tokens)):
            raise D92ContinuousSessionError("duplicate physical support token")
        transform_identity = (
            str(support_transform_identity)
            if support_transform_identity is not None
            else _callable_identity(support_transform)
        )
        if not transform_identity:
            raise D92ContinuousSessionError("frozen DA support transform identity missing")
        return cls(
            old_records=records,
            da_anchor_id=da_anchor_id,
            log_diag_fp32=log_diag_fp32,
            support_transform=support_transform,
            support_transform_identity=transform_identity,
        )


@dataclass(frozen=True)
class SessionLedger:
    """Append-only arrived-package ledger; future support is never retained."""

    anchor: FrozenDAAnchor
    arrived_records: tuple[_SupportRecord, ...]
    registered_tokens: frozenset[str]
    next_session: int

    @classmethod
    def start(cls, anchor: FrozenDAAnchor) -> "SessionLedger":
        if not isinstance(anchor, FrozenDAAnchor):
            raise D92ContinuousSessionError("continuous session requires frozen DA anchor")
        tokens = frozenset(
            token for record in anchor.old_records for token in record.physical_tokens
        )
        return cls(
            anchor=anchor,
            arrived_records=(),
            registered_tokens=tokens,
            next_session=1,
        )


@dataclass(frozen=True)
class ContinuousSessionStatistics:
    """D92-equivalent prefix statistics for one-to-four arrived classes."""

    rows: np.ndarray
    labels: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    old_covariance: np.ndarray
    new_covariance: np.ndarray
    class_count: int
    k_shot: int
    covariance_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _readonly(self.rows, np.float64))
        object.__setattr__(self, "labels", _readonly(self.labels, np.int64))
        object.__setattr__(self, "means", _readonly(self.means, np.float64))
        object.__setattr__(self, "covariance", _readonly(self.covariance, np.float64))
        object.__setattr__(
            self, "old_covariance", _readonly(self.old_covariance, np.float64)
        )
        object.__setattr__(
            self, "new_covariance", _readonly(self.new_covariance, np.float64)
        )
        object.__setattr__(
            self, "covariance_audit", MappingProxyType(dict(self.covariance_audit))
        )


@dataclass(frozen=True)
class ContinuousD42State:
    """D42 residual-int8 head fields, without a second continuous patch."""

    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag_fp32: np.ndarray
    coef1_qint8: np.ndarray
    coef2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    intercept_fp16: np.ndarray
    covariance_policy: str
    da_anchor_id: str
    support_transform_identity: str

    def __post_init__(self) -> None:
        classes = tuple(self.classes)
        count = len(classes)
        valid = (
            self.schema == SCHEMA
            and self.old_class_count == OLD_CLASS_COUNT
            and OLD_CLASS_COUNT < count <= OLD_CLASS_COUNT + MAX_NEW_CLASSES
            and len(set(classes)) == count
            and all(isinstance(value, str) and value for value in classes)
            and self.log_diag_fp32.shape == (FEATURE_DIM,)
            and self.log_diag_fp32.dtype == np.float32
            and np.isfinite(self.log_diag_fp32).all()
            and self.coef1_qint8.shape == (count, FEATURE_DIM)
            and self.coef1_qint8.dtype == np.int8
            and self.coef2_qint8.shape == (count, FEATURE_DIM)
            and self.coef2_qint8.dtype == np.int8
            and self.scale1_fp16.shape == (count, len(d42.BLOCK_SLICES))
            and self.scale1_fp16.dtype == np.float16
            and self.scale2_fp16.shape == (count, len(d42.BLOCK_SLICES))
            and self.scale2_fp16.dtype == np.float16
            and self.intercept_fp16.shape == (count,)
            and self.intercept_fp16.dtype == np.float16
            and np.isfinite(self.scale1_fp16).all()
            and np.isfinite(self.scale2_fp16).all()
            and np.isfinite(self.intercept_fp16).all()
            and bool(np.all(self.scale1_fp16 > 0))
            and bool(np.all(self.scale2_fp16 > 0))
            and isinstance(self.covariance_policy, str)
            and bool(self.covariance_policy)
            and isinstance(self.da_anchor_id, str)
            and bool(self.da_anchor_id)
            and isinstance(self.support_transform_identity, str)
            and bool(self.support_transform_identity)
        )
        if not valid:
            raise D92ContinuousSessionError("continuous D42 codec state drift")
        object.__setattr__(self, "classes", classes)
        for name, dtype in (
            ("log_diag_fp32", np.float32),
            ("coef1_qint8", np.int8),
            ("coef2_qint8", np.int8),
            ("scale1_fp16", np.float16),
            ("scale2_fp16", np.float16),
            ("intercept_fp16", np.float16),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def registry_state_bytes(self) -> int:
        metadata = {
            "classes": list(self.classes),
            "covariance_policy": self.covariance_policy,
            "da_anchor_id": self.da_anchor_id,
            "old_class_count": self.old_class_count,
            "schema": self.schema,
            "support_transform_identity": self.support_transform_identity,
        }
        return len(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        )

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.registry_state_bytes
            + self.log_diag_fp32.nbytes
            + self.coef1_qint8.nbytes
            + self.coef2_qint8.nbytes
            + self.scale1_fp16.nbytes
            + self.scale2_fp16.nbytes
            + self.intercept_fp16.nbytes
        )

    @property
    def persistent_state_sha256(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            "classes": list(self.classes),
            "covariance_policy": self.covariance_policy,
            "da_anchor_id": self.da_anchor_id,
            "old_class_count": self.old_class_count,
            "schema": self.schema,
            "support_transform_identity": self.support_transform_identity,
        }
        digest.update(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        )
        for value in (
            self.log_diag_fp32,
            self.coef1_qint8,
            self.coef2_qint8,
            self.scale1_fp16,
            self.scale2_fp16,
            self.intercept_fp16,
        ):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        return digest.hexdigest()

    def to_d42_unified_state(self) -> d42.D42UnifiedShrinkageLDAState:
        """Return the original D42 scorer state without a second codec pass.

        The D42 constructor is the only validation boundary for this adapter.
        It receives the exact published residual-int8 arrays and frozen E0
        log-diagonal, while the continuous-only ledger metadata stays outside
        the query state.
        """

        return d42.D42UnifiedShrinkageLDAState(
            schema=d42.SCHEMA_INT8,
            classes=self.classes,
            old_class_count=self.old_class_count,
            log_diag_fp32=self.log_diag_fp32,
            coef1_qint8=self.coef1_qint8,
            coef2_qint8=self.coef2_qint8,
            scale1_fp16=self.scale1_fp16,
            scale2_fp16=self.scale2_fp16,
            intercept_fp16=self.intercept_fp16,
            coef_fp32=np.zeros((0, FEATURE_DIM), dtype=np.float32),
            intercept_fp32=np.zeros(0, dtype=np.float32),
            covariance_policy=self.covariance_policy,
        )


def to_d42_unified_state(
    state: ContinuousD42State,
) -> d42.D42UnifiedShrinkageLDAState:
    """Explicit one-way adapter to the original D42 F0 scorer state."""

    if not isinstance(state, ContinuousD42State):
        raise D92ContinuousSessionError("continuous D42 state conversion drift")
    return state.to_d42_unified_state()


@dataclass(frozen=True)
class ContinuousSessionResult:
    """One successful support-only state transition and its publication receipt."""

    ledger: SessionLedger
    state: ContinuousD42State
    statistics: ContinuousSessionStatistics | d92.RegistrationBalancedStatistics
    coefficient: np.ndarray
    intercept: np.ndarray
    transformed_rows: np.ndarray
    targets: np.ndarray
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", _readonly(self.coefficient, np.float32))
        object.__setattr__(self, "intercept", _readonly(self.intercept, np.float32))
        object.__setattr__(
            self, "transformed_rows", _readonly(self.transformed_rows, np.float32)
        )
        object.__setattr__(self, "targets", _readonly(self.targets, np.int64))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def _validate_arrival_metadata(
    ledger: SessionLedger, packets: tuple[SupportPacket, ...]
) -> None:
    if not packets:
        raise D92ContinuousSessionError("continuous session requires arriving support")
    for packet in packets:
        if not isinstance(packet, SupportPacket):
            raise D92ContinuousSessionError("continuous support packet type drift")
        if packet.arrival_session > ledger.next_session:
            raise D92ContinuousSessionError("future support package is not open")
        if packet.arrival_session < ledger.next_session:
            raise D92ContinuousSessionError("past support package cannot be replayed")


def _advance_ledger(
    ledger: SessionLedger, packets: tuple[SupportPacket, ...]
) -> SessionLedger:
    _validate_arrival_metadata(ledger, packets)
    materialized = _canonical_records(_materialize_packet(packet) for packet in packets)
    existing_handles = {
        record.handle for record in ledger.anchor.old_records + ledger.arrived_records
    }
    existing_packages = {
        record.package_id for record in ledger.anchor.old_records + ledger.arrived_records
    }
    seen_tokens = set(ledger.registered_tokens)
    for record in materialized:
        if record.handle in existing_handles:
            raise D92ContinuousSessionError("continuous support class handle repeated")
        if record.package_id in existing_packages:
            raise D92ContinuousSessionError("continuous support package repeated")
        if any(token in seen_tokens for token in record.physical_tokens):
            raise D92ContinuousSessionError("duplicate physical support token")
        existing_handles.add(record.handle)
        existing_packages.add(record.package_id)
        seen_tokens.update(record.physical_tokens)
    arrived = _canonical_records(ledger.arrived_records + materialized)
    if len(arrived) > MAX_NEW_CLASSES:
        raise D92ContinuousSessionError("continuous registration exceeds new-class prefix five")
    all_tokens = frozenset(
        token
        for record in ledger.anchor.old_records + arrived
        for token in record.physical_tokens
    )
    expected_token_count = (OLD_CLASS_COUNT + len(arrived)) * K_SHOT
    if len(all_tokens) != expected_token_count:
        raise D92ContinuousSessionError("duplicate physical support token")
    return SessionLedger(
        anchor=ledger.anchor,
        arrived_records=arrived,
        registered_tokens=all_tokens,
        next_session=ledger.next_session + 1,
    )


def _support_registry(
    ledger: SessionLedger,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    records = ledger.anchor.old_records + ledger.arrived_records
    class_count = len(records)
    if class_count <= OLD_CLASS_COUNT or class_count > OLD_CLASS_COUNT + MAX_NEW_CLASSES:
        raise D92ContinuousSessionError("continuous registered-class prefix drift")
    classes = tuple(record.handle for record in records)
    rows = np.concatenate([record.rows for record in records], axis=0)
    targets = np.concatenate(
        [np.full(K_SHOT, index, dtype=np.int64) for index in range(class_count)], axis=0
    )
    return classes, _readonly(rows, np.float32), _readonly(targets, np.int64)


def _apply_frozen_support_transform(
    ledger: SessionLedger,
    rows: np.ndarray,
    targets: np.ndarray,
    class_count: int,
) -> np.ndarray:
    try:
        transformed = ledger.anchor.support_transform(
            rows, targets, int(class_count), K_SHOT
        )
    except D92ContinuousSessionError:
        raise
    except Exception as exc:  # pragma: no cover - defensive transform boundary
        raise D92ContinuousSessionError("frozen DA support transform failed") from exc
    result = np.asarray(transformed, dtype=np.float32)
    if (
        result.shape != (class_count * K_SHOT, FEATURE_DIM)
        or not np.isfinite(result).all()
    ):
        raise D92ContinuousSessionError("frozen DA support transform output drift")
    return _readonly(result, np.float32)


def _canonical_means(
    rows: np.ndarray, labels: np.ndarray, class_count: int
) -> np.ndarray:
    return np.stack(
        [
            np.sum(
                d92._canonical_class_rows(rows[labels == index]),
                axis=0,
                dtype=np.float64,
            )
            / float(K_SHOT)
            for index in range(class_count)
        ]
    )


def _single_class_bridge_covariance(rows: np.ndarray) -> np.ndarray:
    """The unique S1 bridge: StandardScaler followed by Ledoit-Wolf."""

    source = np.asarray(rows, dtype=np.float64)
    centered = source - source.mean(axis=0, keepdims=True)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(centered)
    standardized_covariance, _ = ledoit_wolf(standardized)
    covariance = (
        scaler.scale_[:, None]
        * standardized_covariance
        * scaler.scale_[None, :]
    )
    return 0.5 * (covariance + covariance.T)


def _build_prefix_statistics(
    d42_module: Any,
    transformed: np.ndarray,
    targets: np.ndarray,
    new_class_count: int,
) -> ContinuousSessionStatistics:
    class_count = OLD_CLASS_COUNT + int(new_class_count)
    rows = np.asarray(transformed, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    means = _canonical_means(rows, labels, class_count)
    old_indices = np.arange(OLD_CLASS_COUNT, dtype=np.int64)
    new_indices = np.arange(OLD_CLASS_COUNT, class_count, dtype=np.int64)
    old_covariance = d92._group_covariance(d42_module, rows, labels, old_indices)
    if new_class_count == 1:
        new_covariance = _single_class_bridge_covariance(rows[labels == OLD_CLASS_COUNT])
        covariance_policy = "standard_scaler_ledoit_wolf_singleton"
    else:
        new_covariance = d92._group_covariance(d42_module, rows, labels, new_indices)
        covariance_policy = "sklearn_lsqr_auto_shrinkage_equal_prior"
    covariance = d92.TASK_WEIGHT * old_covariance + d92.TASK_WEIGHT * new_covariance
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) <= 0.0:
        raise D92ContinuousSessionError("continuous D92 covariance is not positive definite")
    audit: dict[str, Any] = {
        "solver": "lsqr_equivalent_explicit_solve",
        "shrinkage": (
            "standard_scaler_ledoit_wolf_singleton"
            if new_class_count == 1
            else "auto_per_registration_task_then_fixed_equal_average"
        ),
        "prior_policy": "equal_1_over_registered_class_count",
        "covariance_policy": covariance_policy,
        "unit_covariance_fallback": False,
        "support_rows": int(len(rows)),
        "class_count": class_count,
        "k_shot": K_SHOT,
        "d92_status": "registration_balanced_active",
        "d92_registration_balanced_active": True,
        "d92_old_class_count": OLD_CLASS_COUNT,
        "d92_new_class_count": int(new_class_count),
        "d92_old_covariance_weight": d92.TASK_WEIGHT,
        "d92_new_covariance_weight": d92.TASK_WEIGHT,
        "d92_weight_source": "fixed_equal_stage2b_stage2c_task_priority",
        "d92_formula": "Sigma_shared=0.5*Sigma_old_auto+0.5*Sigma_new_auto",
        "d92_weight_scan_count": 0,
        "d92_hyperparameter_scan_count": 0,
        "d92_query_rows_used": 0,
        "d92_query_role_oracle_access": False,
        "d92_scene_receiver_seed_specific_branch": False,
        "d92_class_id_specific_formula": False,
        "d92_registration_state_support_only": True,
        "d92_old_covariance_trace": float(np.trace(old_covariance)),
        "d92_new_covariance_trace": float(np.trace(new_covariance)),
        "d92_balanced_covariance_trace": float(np.trace(covariance)),
        "d92_balanced_eigenvalue_min": float(np.min(eigenvalues)),
        "d92_balanced_eigenvalue_max": float(np.max(eigenvalues)),
        "d92_shared_covariance_estimation_count": 1,
        # S1 still has two group-local covariance estimators: old sklearn
        # auto-shrinkage plus the singleton StandardScaler/Ledoit-Wolf bridge.
        "d92_group_local_shrinkage_estimation_count": 2,
        "d92_continuous_singleton_bridge": new_class_count == 1,
    }
    return ContinuousSessionStatistics(
        rows=rows,
        labels=labels,
        means=means,
        covariance=covariance,
        old_covariance=old_covariance,
        new_covariance=new_covariance,
        class_count=class_count,
        k_shot=K_SHOT,
        covariance_audit=audit,
    )


def _compile_prefix_affine(
    statistics: ContinuousSessionStatistics,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    covariance = np.asarray(statistics.covariance, dtype=np.float64).copy()
    coefficient64 = np.linalg.solve(covariance, statistics.means.T).T
    equation_residual = float(
        np.max(np.abs(covariance @ coefficient64.T - statistics.means.T))
    )
    priors = np.full(statistics.class_count, 1.0 / statistics.class_count, dtype=np.float64)
    intercept64 = -0.5 * np.diag(statistics.means @ coefficient64.T) + np.log(priors)
    coefficient64 -= coefficient64.mean(axis=0, keepdims=True)
    intercept64 -= intercept64.mean()
    if not math.isfinite(equation_residual):
        raise D92ContinuousSessionError("continuous D92 covariance solve became non-finite")
    audit = dict(statistics.covariance_audit)
    audit.update(
        {
            "d92_covariance_arm": "full",
            "d92_class_common_affine_omitted_before_fp32": True,
            "d92_centered_coefficient_mean_max_abs": float(
                np.max(np.abs(coefficient64.mean(axis=0)))
            ),
            "d92_centered_intercept_mean_abs": float(abs(intercept64.mean())),
            "d92_covariance_equation_residual_max": equation_residual,
            "covariance_equation_residual_max": equation_residual,
            "d92_shared_covariance_reused": True,
            "d92_shared_statistics_compile_arm": "full",
            "d92_compiled_covariance_eigvalsh_count": 0,
            "d92_compiled_covariance_solve_count": 1,
            "d92_compiled_covariance_dense_288_solve_count": 1,
            "d92_compiled_covariance_eigenvalue_min": float(
                statistics.covariance_audit["d92_balanced_eigenvalue_min"]
            ),
        }
    )
    return coefficient64.astype(np.float32), intercept64.astype(np.float32), audit


def _compile_d42_state(
    d42_module: Any,
    *,
    classes: tuple[str, ...],
    coefficient: np.ndarray,
    intercept: np.ndarray,
    covariance_policy: str,
    anchor: FrozenDAAnchor,
) -> tuple[ContinuousD42State, dict[str, float]]:
    code1, code2, scale1, scale2, decoded = d42_module._quantize_coefficients(
        np.asarray(coefficient, dtype=np.float32)
    )
    intercept16 = np.asarray(intercept, dtype=np.float16)
    if not np.isfinite(intercept16).all():
        raise D92ContinuousSessionError("continuous D42 intercept FP16 overflow")
    state = ContinuousD42State(
        schema=SCHEMA,
        classes=classes,
        old_class_count=OLD_CLASS_COUNT,
        log_diag_fp32=anchor.log_diag_fp32,
        coef1_qint8=code1,
        coef2_qint8=code2,
        scale1_fp16=scale1,
        scale2_fp16=scale2,
        intercept_fp16=intercept16,
        covariance_policy=covariance_policy,
        da_anchor_id=anchor.da_anchor_id,
        support_transform_identity=anchor.support_transform_identity,
    )
    coefficient_error = np.abs(decoded - np.asarray(coefficient, dtype=np.float32))
    intercept_error = np.abs(intercept16.astype(np.float32) - np.asarray(intercept, dtype=np.float32))
    return state, {
        "coefficient_quantization_error_mean": float(np.mean(coefficient_error)),
        "coefficient_quantization_error_max": float(np.max(coefficient_error)),
        "intercept_quantization_error_mean": float(np.mean(intercept_error)),
        "intercept_quantization_error_max": float(np.max(intercept_error)),
    }


def advance_session(
    ledger: SessionLedger,
    arriving: Sequence[SupportPacket],
    *,
    d42_module: Any = d42,
) -> ContinuousSessionResult:
    """Publish one cumulative D92 FULL/D42 head from legal arrived support only."""

    if not isinstance(ledger, SessionLedger) or ledger.next_session < 1:
        raise D92ContinuousSessionError("continuous session ledger drift")
    if d42_module is None or not hasattr(d42_module, "_quantize_coefficients"):
        raise D92ContinuousSessionError("continuous D42 codec dependency drift")
    packets = tuple(arriving)
    next_ledger = _advance_ledger(ledger, packets)
    classes, support_rows, targets = _support_registry(next_ledger)
    transformed = _apply_frozen_support_transform(
        next_ledger, support_rows, targets, len(classes)
    )
    new_class_count = len(next_ledger.arrived_records)
    if new_class_count == MAX_NEW_CLASSES:
        statistics = d92.build_registration_balanced_statistics(
            d42_module, transformed, targets, len(classes), K_SHOT
        )
        coefficient, intercept, compile_audit = d92.compile_registration_balanced_affine(
            d42_module, statistics, arm="full"
        )
        original_e0_equivalent = True
        original_builder_used = True
    else:
        statistics = _build_prefix_statistics(
            d42_module, transformed, targets, new_class_count
        )
        coefficient, intercept, compile_audit = _compile_prefix_affine(statistics)
        original_e0_equivalent = False
        original_builder_used = False
    state, codec_audit = _compile_d42_state(
        d42_module,
        classes=classes,
        coefficient=coefficient,
        intercept=intercept,
        covariance_policy=str(statistics.covariance_audit["covariance_policy"]),
        anchor=next_ledger.anchor,
    )
    audit: dict[str, Any] = dict(statistics.covariance_audit)
    audit.update(compile_audit)
    audit.update(codec_audit)
    audit.update(
        {
            "method_id": METHOD_ID,
            "lifecycle_state": f"DA1_REG1_S{ledger.next_session}",
            "d92_continuous_session_index": int(ledger.next_session),
            "d92_continuous_new_class_prefix_count": new_class_count,
            "d92_continuous_frozen_da_anchor_reused": True,
            "d92_continuous_da_anchor_id": next_ledger.anchor.da_anchor_id,
            "d92_continuous_support_transform_identity": (
                next_ledger.anchor.support_transform_identity
            ),
            "d92_continuous_bridge_active": new_class_count == 1,
            "d92_continuous_bridge_policy": (
                "standard_scaler_ledoit_wolf_singleton"
                if new_class_count == 1
                else "not_active"
            ),
            "d92_continuous_original_e0_equivalent": original_e0_equivalent,
            "d92_continuous_s5_original_builder_used": original_builder_used,
            "d92_continuous_full_solve_count": 1,
            "d92_continuous_d42_codec_count": 1,
            "d92_continuous_d81_transform_count": 1,
            "d92_continuous_support_rows": int(len(transformed)),
            "d92_continuous_registered_class_count": int(len(classes)),
            "d92_continuous_query_state_bytes": state.persistent_state_bytes,
            "d92_continuous_query_state_sha256": state.persistent_state_sha256,
            "d92_continuous_query_macs": int(len(classes) * FEATURE_DIM),
            "d92_continuous_resource_receipt_status": "NOT_MEASURED_CORE",
            "d92_continuous_incremental_peak_bytes": None,
            "d92_continuous_registration_wall_ms": None,
            "d92_continuous_peak_budget_bytes": 4 * 1024 * 1024,
            "d92_continuous_wall_budget_ms": 300,
            "future_support_open_sentinel": 0,
            "past_token_duplicate_count": 0,
            "query_fit_access": False,
            "query_update_access": False,
            "query_selection_access": False,
            "query_truth_access": False,
            "query_role_oracle_access": False,
            "query_class_quota_access": False,
            "query_global_reassignment": False,
            "clean_source_access": False,
        }
    )
    return ContinuousSessionResult(
        ledger=next_ledger,
        state=state,
        statistics=statistics,
        coefficient=coefficient,
        intercept=intercept,
        transformed_rows=transformed,
        targets=targets,
        audit=audit,
    )


__all__ = [
    "ContinuousD42State",
    "ContinuousSessionResult",
    "ContinuousSessionStatistics",
    "D92ContinuousSessionError",
    "FEATURE_DIM",
    "FrozenDAAnchor",
    "K_SHOT",
    "MAX_NEW_CLASSES",
    "METHOD_ID",
    "OLD_CLASS_COUNT",
    "SCHEMA",
    "SessionLedger",
    "SupportPacket",
    "advance_session",
    "to_d42_unified_state",
]
