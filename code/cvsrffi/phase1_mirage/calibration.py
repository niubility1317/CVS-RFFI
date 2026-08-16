"""Source-only MIRAGE threshold calibration and immutable decision tables.

This module deliberately accepts only typed source validation score tables.  It
does not load files, inspect target artifacts, or retain mutable model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import torch

from .head import DecisionThresholds, OpenHeadOutput, decide
from .protocol import Permission, Phase1DataPolicy, Phase1PolicyError, ProxyRole, SourcePartition


class CalibrationProtocolError(ValueError):
    """Raised when a score table, role, or frozen-decision boundary is invalid."""


class NoDeployableSeparation(RuntimeError):
    """Raised when no source-only threshold tuple satisfies known-FRR safety."""


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationProtocolError(f"{field_name} must be a non-empty string")
    return value


def _require_unit_interval(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CalibrationProtocolError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise CalibrationProtocolError(f"{field_name} must be finite")
    if not 0.0 <= numeric <= 1.0:
        raise CalibrationProtocolError(f"{field_name} must lie in [0, 1]")
    return numeric


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationProtocolError(f"{field_name} must be a non-negative integer")
    return value


def _require_group_key(value: object, field_name: str) -> object:
    if isinstance(value, str) and not value:
        raise CalibrationProtocolError(f"{field_name} must not be empty")
    try:
        hash(value)
    except TypeError as error:
        raise CalibrationProtocolError(f"{field_name} must be hashable") from error
    return value


@dataclass(frozen=True)
class KnownScoreRow:
    """One immutable V_cal or V_select known-class query score."""

    physical_id: str
    query_id: str
    quality: float
    unknown_risk: float
    inside_registered_support: bool
    predicted_class: int
    true_class: int
    receiver: object
    day: object
    scene: object
    fold: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_id", _require_identifier(self.physical_id, "physical_id"))
        object.__setattr__(self, "query_id", _require_identifier(self.query_id, "query_id"))
        object.__setattr__(self, "quality", _require_unit_interval(self.quality, "quality"))
        object.__setattr__(self, "unknown_risk", _require_unit_interval(self.unknown_risk, "unknown_risk"))
        if not isinstance(self.inside_registered_support, bool):
            raise CalibrationProtocolError("inside_registered_support must be bool")
        object.__setattr__(self, "predicted_class", _require_nonnegative_int(self.predicted_class, "predicted_class"))
        object.__setattr__(self, "true_class", _require_nonnegative_int(self.true_class, "true_class"))
        object.__setattr__(self, "receiver", _require_group_key(self.receiver, "receiver"))
        object.__setattr__(self, "day", _require_group_key(self.day, "day"))
        object.__setattr__(self, "scene", _require_group_key(self.scene, "scene"))
        object.__setattr__(self, "fold", _require_nonnegative_int(self.fold, "fold"))


@dataclass(frozen=True)
class ProxyScoreRow:
    """One immutable P_cal or P_select proxy-unknown query score."""

    physical_id: str
    query_id: str
    quality: float
    unknown_risk: float
    inside_registered_support: bool
    predicted_class: int
    fold: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_id", _require_identifier(self.physical_id, "physical_id"))
        object.__setattr__(self, "query_id", _require_identifier(self.query_id, "query_id"))
        object.__setattr__(self, "quality", _require_unit_interval(self.quality, "quality"))
        object.__setattr__(self, "unknown_risk", _require_unit_interval(self.unknown_risk, "unknown_risk"))
        if not isinstance(self.inside_registered_support, bool):
            raise CalibrationProtocolError("inside_registered_support must be bool")
        object.__setattr__(self, "predicted_class", _require_nonnegative_int(self.predicted_class, "predicted_class"))
        object.__setattr__(self, "fold", _require_nonnegative_int(self.fold, "fold"))


def _validate_rows(rows: object, expected_type: type[KnownScoreRow] | type[ProxyScoreRow], table_name: str) -> tuple[Any, ...]:
    if not isinstance(rows, tuple):
        raise CalibrationProtocolError(f"{table_name}.rows must be an immutable tuple")
    if not rows:
        raise CalibrationProtocolError(f"{table_name}.rows must not be empty")
    if any(not isinstance(row, expected_type) for row in rows):
        raise CalibrationProtocolError(f"{table_name}.rows contain an invalid row type")
    physical_ids = [row.physical_id for row in rows]
    query_ids = [row.query_id for row in rows]
    if len(set(physical_ids)) != len(physical_ids):
        raise CalibrationProtocolError(f"{table_name} has duplicate physical_id")
    if len(set(query_ids)) != len(query_ids):
        raise CalibrationProtocolError(f"{table_name} has duplicate query_id")
    folds = {row.fold for row in rows}
    if len(folds) != 1:
        raise CalibrationProtocolError(f"{table_name} must contain exactly one fold")
    return rows


def _validate_update_count(value: object, table_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationProtocolError(f"{table_name}.update_count must be a non-negative integer")
    return value


def _column_tuple(value: object, field_name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise CalibrationProtocolError(f"{field_name} must be an iterable column")
    return tuple(value)


def _same_length_columns(**columns: object) -> dict[str, tuple[object, ...]]:
    normalized = {name: _column_tuple(value, name) for name, value in columns.items()}
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise CalibrationProtocolError("score table columns must have the same length")
    if not lengths or next(iter(lengths)) == 0:
        raise CalibrationProtocolError("score table columns must not be empty")
    return normalized


@dataclass(frozen=True)
class KnownScoreTable:
    """An immutable known-query score table explicitly bound to a validation role."""

    role: SourcePartition
    rows: tuple[KnownScoreRow, ...]
    update_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.role, SourcePartition) or self.role not in {
            SourcePartition.V_CAL,
            SourcePartition.V_SELECT,
        }:
            raise CalibrationProtocolError("known score table role must be V_cal or V_select")
        object.__setattr__(self, "rows", _validate_rows(self.rows, KnownScoreRow, "known score table"))
        object.__setattr__(self, "update_count", _validate_update_count(self.update_count, "known score table"))

    @property
    def fold(self) -> int:
        return self.rows[0].fold

    @property
    def physical_ids(self) -> frozenset[str]:
        return frozenset(row.physical_id for row in self.rows)

    @property
    def query_ids(self) -> frozenset[str]:
        return frozenset(row.query_id for row in self.rows)

    @classmethod
    def from_columns(
        cls,
        *,
        role: SourcePartition,
        physical_ids: Iterable[str],
        query_ids: Iterable[str],
        qualities: Iterable[float],
        unknown_risks: Iterable[float],
        inside_registered_support: Iterable[bool],
        predicted_classes: Iterable[int],
        true_classes: Iterable[int],
        receivers: Iterable[object],
        days: Iterable[object],
        scenes: Iterable[object],
        folds: Iterable[int],
        update_count: int = 0,
    ) -> "KnownScoreTable":
        columns = _same_length_columns(
            physical_ids=physical_ids,
            query_ids=query_ids,
            qualities=qualities,
            unknown_risks=unknown_risks,
            inside_registered_support=inside_registered_support,
            predicted_classes=predicted_classes,
            true_classes=true_classes,
            receivers=receivers,
            days=days,
            scenes=scenes,
            folds=folds,
        )
        rows = tuple(
            KnownScoreRow(
                physical_id=physical_id,
                query_id=query_id,
                quality=quality,
                unknown_risk=unknown_risk,
                inside_registered_support=inside,
                predicted_class=predicted,
                true_class=true,
                receiver=receiver,
                day=day,
                scene=scene,
                fold=fold,
            )
            for physical_id, query_id, quality, unknown_risk, inside, predicted, true, receiver, day, scene, fold in zip(
                columns["physical_ids"],
                columns["query_ids"],
                columns["qualities"],
                columns["unknown_risks"],
                columns["inside_registered_support"],
                columns["predicted_classes"],
                columns["true_classes"],
                columns["receivers"],
                columns["days"],
                columns["scenes"],
                columns["folds"],
                strict=True,
            )
        )
        return cls(role=role, rows=rows, update_count=update_count)


@dataclass(frozen=True)
class ProxyScoreTable:
    """An immutable proxy-unknown score table explicitly bound to a proxy role."""

    role: ProxyRole
    rows: tuple[ProxyScoreRow, ...]
    update_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.role, ProxyRole) or self.role not in {ProxyRole.P_CAL, ProxyRole.P_SELECT}:
            raise CalibrationProtocolError("proxy score table role must be P_cal or P_select")
        object.__setattr__(self, "rows", _validate_rows(self.rows, ProxyScoreRow, "proxy score table"))
        object.__setattr__(self, "update_count", _validate_update_count(self.update_count, "proxy score table"))

    @property
    def fold(self) -> int:
        return self.rows[0].fold

    @property
    def physical_ids(self) -> frozenset[str]:
        return frozenset(row.physical_id for row in self.rows)

    @property
    def query_ids(self) -> frozenset[str]:
        return frozenset(row.query_id for row in self.rows)

    @classmethod
    def from_columns(
        cls,
        *,
        role: ProxyRole,
        physical_ids: Iterable[str],
        query_ids: Iterable[str],
        qualities: Iterable[float],
        unknown_risks: Iterable[float],
        inside_registered_support: Iterable[bool],
        predicted_classes: Iterable[int],
        folds: Iterable[int],
        update_count: int = 0,
    ) -> "ProxyScoreTable":
        columns = _same_length_columns(
            physical_ids=physical_ids,
            query_ids=query_ids,
            qualities=qualities,
            unknown_risks=unknown_risks,
            inside_registered_support=inside_registered_support,
            predicted_classes=predicted_classes,
            folds=folds,
        )
        rows = tuple(
            ProxyScoreRow(
                physical_id=physical_id,
                query_id=query_id,
                quality=quality,
                unknown_risk=unknown_risk,
                inside_registered_support=inside,
                predicted_class=predicted,
                fold=fold,
            )
            for physical_id, query_id, quality, unknown_risk, inside, predicted, fold in zip(
                columns["physical_ids"],
                columns["query_ids"],
                columns["qualities"],
                columns["unknown_risks"],
                columns["inside_registered_support"],
                columns["predicted_classes"],
                columns["folds"],
                strict=True,
            )
        )
        return cls(role=role, rows=rows, update_count=update_count)


@dataclass(frozen=True)
class KnownDecisionRow:
    """A known score row after one frozen, per-query head decision."""

    row: KnownScoreRow
    label: int
    registered: bool
    explicit_unknown: bool
    deferred: bool

    def __post_init__(self) -> None:
        if not isinstance(self.row, KnownScoreRow):
            raise CalibrationProtocolError("known decision row must retain a KnownScoreRow")
        _validate_decision_flags(self.label, self.registered, self.explicit_unknown, self.deferred)


@dataclass(frozen=True)
class ProxyDecisionRow:
    """A proxy score row after one frozen, per-query head decision."""

    row: ProxyScoreRow
    label: int
    registered: bool
    explicit_unknown: bool
    deferred: bool

    def __post_init__(self) -> None:
        if not isinstance(self.row, ProxyScoreRow):
            raise CalibrationProtocolError("proxy decision row must retain a ProxyScoreRow")
        _validate_decision_flags(self.label, self.registered, self.explicit_unknown, self.deferred)


def _validate_decision_flags(label: object, registered: object, explicit_unknown: object, deferred: object) -> None:
    if isinstance(label, bool) or not isinstance(label, int):
        raise CalibrationProtocolError("decision label must be an integer")
    flags = (registered, explicit_unknown, deferred)
    if any(not isinstance(flag, bool) for flag in flags) or sum(flags) != 1:
        raise CalibrationProtocolError("decision states must be exactly one of registered, unknown, or defer")


_FROZEN_TABLE_FACTORY_SEAL = object()


def _sha256_payload(value: object) -> str:
    return sha256(repr(value).encode("utf-8")).hexdigest()


def _valid_role_pair(known_role: object, proxy_role: object) -> bool:
    return (known_role, proxy_role) in {
        (SourcePartition.V_CAL, ProxyRole.P_CAL),
        (SourcePartition.V_SELECT, ProxyRole.P_SELECT),
    }


def _validate_decision_row_collections(
    known_rows: object,
    proxy_rows: object,
    *,
    known_role: object,
    proxy_role: object,
    proxy_update_count: object,
) -> tuple[tuple[KnownDecisionRow, ...], tuple[ProxyDecisionRow, ...], int, int]:
    """Fail closed on a forged or internally inconsistent frozen decision payload."""

    if not _valid_role_pair(known_role, proxy_role):
        raise CalibrationProtocolError("frozen decision roles must be paired validation roles")
    if not isinstance(known_rows, tuple) or not known_rows:
        raise CalibrationProtocolError("known decision rows must be a non-empty tuple")
    if not isinstance(proxy_rows, tuple) or not proxy_rows:
        raise CalibrationProtocolError("proxy decision rows must be a non-empty tuple")
    if any(not isinstance(row, KnownDecisionRow) for row in known_rows):
        raise CalibrationProtocolError("known decision rows are malformed")
    if any(not isinstance(row, ProxyDecisionRow) for row in proxy_rows):
        raise CalibrationProtocolError("proxy decision rows are malformed")
    update_count = _validate_update_count(proxy_update_count, "proxy decision table")
    known_physical_ids = [row.row.physical_id for row in known_rows]
    known_query_ids = [row.row.query_id for row in known_rows]
    proxy_physical_ids = [row.row.physical_id for row in proxy_rows]
    proxy_query_ids = [row.row.query_id for row in proxy_rows]
    for label, identities in (
        ("known physical_id", known_physical_ids),
        ("known query_id", known_query_ids),
        ("proxy physical_id", proxy_physical_ids),
        ("proxy query_id", proxy_query_ids),
    ):
        if len(set(identities)) != len(identities):
            raise CalibrationProtocolError(f"frozen decision table has duplicate {label}")
    known_ids = set(known_physical_ids) | set(known_query_ids)
    proxy_ids = set(proxy_physical_ids) | set(proxy_query_ids)
    if known_ids & proxy_ids:
        raise CalibrationProtocolError("frozen decision table has forbidden known/proxy ID overlap")
    folds = {row.row.fold for row in known_rows} | {row.row.fold for row in proxy_rows}
    if len(folds) != 1:
        raise CalibrationProtocolError("frozen decision table must contain exactly one fold")
    return known_rows, proxy_rows, update_count, next(iter(folds))


def _decision_rows_digest(rows: tuple[KnownDecisionRow, ...] | tuple[ProxyDecisionRow, ...]) -> str:
    return _sha256_payload(
        tuple(
            (
                row.row.physical_id,
                row.row.query_id,
                row.row.quality,
                row.row.unknown_risk,
                row.row.inside_registered_support,
                row.row.predicted_class,
                getattr(row.row, "true_class", None),
                getattr(row.row, "receiver", None),
                getattr(row.row, "day", None),
                getattr(row.row, "scene", None),
                row.row.fold,
                row.label,
                row.registered,
                row.explicit_unknown,
                row.deferred,
            )
            for row in rows
        )
    )


@dataclass(frozen=True)
class DecisionSourceReceipt:
    """Immutable role and source-row hashes that bind a factory-created decision table."""

    known_role: SourcePartition
    proxy_role: ProxyRole
    fold: int
    known_row_count: int
    proxy_row_count: int
    known_rows_sha256: str
    proxy_rows_sha256: str
    proxy_update_count: int

    def __post_init__(self) -> None:
        if not _valid_role_pair(self.known_role, self.proxy_role):
            raise CalibrationProtocolError("decision source receipt roles are invalid")
        object.__setattr__(self, "fold", _require_nonnegative_int(self.fold, "decision source receipt fold"))
        object.__setattr__(
            self,
            "known_row_count",
            _require_nonnegative_int(self.known_row_count, "decision source receipt known_row_count"),
        )
        object.__setattr__(
            self,
            "proxy_row_count",
            _require_nonnegative_int(self.proxy_row_count, "decision source receipt proxy_row_count"),
        )
        if self.known_row_count == 0 or self.proxy_row_count == 0:
            raise CalibrationProtocolError("decision source receipt row counts must be positive")
        for field_name in ("known_rows_sha256", "proxy_rows_sha256"):
            digest = getattr(self, field_name)
            if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
                raise CalibrationProtocolError(f"{field_name} must be a lowercase SHA256 digest")
        object.__setattr__(
            self,
            "proxy_update_count",
            _validate_update_count(self.proxy_update_count, "decision source receipt"),
        )


def _make_source_receipt(
    known_rows: object,
    proxy_rows: object,
    *,
    known_role: object,
    proxy_role: object,
    proxy_update_count: object,
) -> DecisionSourceReceipt:
    known, proxy, updates, fold = _validate_decision_row_collections(
        known_rows,
        proxy_rows,
        known_role=known_role,
        proxy_role=proxy_role,
        proxy_update_count=proxy_update_count,
    )
    return DecisionSourceReceipt(
        known_role=known_role,
        proxy_role=proxy_role,
        fold=fold,
        known_row_count=len(known),
        proxy_row_count=len(proxy),
        known_rows_sha256=_decision_rows_digest(known),
        proxy_rows_sha256=_decision_rows_digest(proxy),
        proxy_update_count=updates,
    )


@dataclass(frozen=True, init=False)
class FrozenDecisionTable:
    """Factory-created scores and decisions for one candidate, fold, and threshold tuple."""

    candidate_id: str
    known_role: SourcePartition
    proxy_role: ProxyRole
    thresholds: DecisionThresholds
    known_rows: tuple[KnownDecisionRow, ...]
    proxy_rows: tuple[ProxyDecisionRow, ...]
    proxy_update_count: int
    source_receipt: DecisionSourceReceipt

    @property
    def fold(self) -> int:
        return self.source_receipt.fold

    @property
    def table_id(self) -> str:
        """Return a deterministic hash bound to the factory source receipt and decisions."""

        payload = (
            self.candidate_id,
            self.source_receipt,
            float(self.thresholds.tau_q),
            float(self.thresholds.tau_reg),
            float(self.thresholds.tau_unk),
        )
        return _sha256_payload(payload)


def _create_frozen_decision_table(
    *,
    candidate_id: object,
    known_role: object,
    proxy_role: object,
    thresholds: object,
    known_rows: object,
    proxy_rows: object,
    proxy_update_count: object,
    _factory_seal: object,
) -> FrozenDecisionTable:
    """Construct a table only for a module-private factory call with validated rows."""

    if _factory_seal is not _FROZEN_TABLE_FACTORY_SEAL:
        raise CalibrationProtocolError("FrozenDecisionTable must be created by the validated factory")
    candidate = _require_identifier(candidate_id, "candidate_id")
    if not isinstance(thresholds, DecisionThresholds):
        raise CalibrationProtocolError("thresholds must be DecisionThresholds")
    source_receipt = _make_source_receipt(
        known_rows,
        proxy_rows,
        known_role=known_role,
        proxy_role=proxy_role,
        proxy_update_count=proxy_update_count,
    )
    table = object.__new__(FrozenDecisionTable)
    object.__setattr__(table, "candidate_id", candidate)
    object.__setattr__(table, "known_role", known_role)
    object.__setattr__(table, "proxy_role", proxy_role)
    object.__setattr__(table, "thresholds", thresholds)
    object.__setattr__(table, "known_rows", known_rows)
    object.__setattr__(table, "proxy_rows", proxy_rows)
    object.__setattr__(table, "proxy_update_count", source_receipt.proxy_update_count)
    object.__setattr__(table, "source_receipt", source_receipt)
    return table


def _validate_frozen_decision_table(table: object) -> FrozenDecisionTable:
    """Recompute the source receipt before any downstream metric consumes a table."""

    if not isinstance(table, FrozenDecisionTable):
        raise CalibrationProtocolError("metric inputs must be a FrozenDecisionTable")
    if not isinstance(table.thresholds, DecisionThresholds):
        raise CalibrationProtocolError("frozen decision table thresholds are invalid")
    expected_receipt = _make_source_receipt(
        table.known_rows,
        table.proxy_rows,
        known_role=table.known_role,
        proxy_role=table.proxy_role,
        proxy_update_count=table.proxy_update_count,
    )
    if table.source_receipt != expected_receipt:
        raise CalibrationProtocolError("frozen decision table source receipt mismatch")
    return table


def _pair_identity_overlap(known_scores: KnownScoreTable, proxy_scores: ProxyScoreTable) -> set[str]:
    known_ids = set(known_scores.physical_ids) | set(known_scores.query_ids)
    proxy_ids = set(proxy_scores.physical_ids) | set(proxy_scores.query_ids)
    return known_ids & proxy_ids


def _validate_role_pair(
    known_scores: KnownScoreTable,
    proxy_scores: ProxyScoreTable,
    *,
    expected_known_role: SourcePartition,
    expected_proxy_role: ProxyRole,
    permission: Permission,
) -> None:
    if not isinstance(known_scores, KnownScoreTable) or not isinstance(proxy_scores, ProxyScoreTable):
        raise CalibrationProtocolError("calibration requires immutable KnownScoreTable and ProxyScoreTable inputs")
    if known_scores.role is not expected_known_role:
        raise CalibrationProtocolError(f"known score table must use {expected_known_role.value}")
    if proxy_scores.role is not expected_proxy_role:
        raise CalibrationProtocolError(f"proxy score table must use {expected_proxy_role.value}")
    if known_scores.fold != proxy_scores.fold:
        raise CalibrationProtocolError("known and proxy score tables must use the same fold")
    if known_scores.update_count != 0 or proxy_scores.update_count != 0:
        raise CalibrationProtocolError("validation score tables require update_count == 0")
    overlap = _pair_identity_overlap(known_scores, proxy_scores)
    if overlap:
        raise CalibrationProtocolError("known and proxy score table ID overlap is forbidden")
    policy = Phase1DataPolicy()
    try:
        policy.require_proxy_origin(expected_proxy_role, expected_known_role)
        policy.require_permission(expected_proxy_role, permission)
    except Phase1PolicyError as error:
        raise CalibrationProtocolError("source role permission check failed") from error


def _head_decisions(
    known_scores: KnownScoreTable,
    proxy_scores: ProxyScoreTable,
    thresholds: DecisionThresholds,
    *,
    candidate_id: str,
) -> FrozenDecisionTable:
    if not isinstance(thresholds, DecisionThresholds):
        raise CalibrationProtocolError("thresholds must be DecisionThresholds")
    combined: tuple[KnownScoreRow | ProxyScoreRow, ...] = known_scores.rows + proxy_scores.rows
    num_classes = 1 + max(
        max(row.predicted_class for row in combined),
        max(row.true_class for row in known_scores.rows),
    )
    batch_size = len(combined)
    class_scores = torch.full((batch_size, num_classes), -1.0, dtype=torch.float32)
    class_distances = torch.ones((batch_size, num_classes), dtype=torch.float32)
    radius_margins = torch.full((batch_size, num_classes), 0.1, dtype=torch.float32)
    for index, row in enumerate(combined):
        class_scores[index, row.predicted_class] = 0.0
        if row.inside_registered_support:
            radius_margins[index, row.predicted_class] = -0.1
            class_distances[index, row.predicted_class] = 0.0
    output = OpenHeadOutput(
        class_scores=class_scores,
        class_distances=class_distances,
        radius_margins=radius_margins,
        energy=torch.zeros(batch_size, dtype=torch.float32),
        unknown_risk=torch.tensor([row.unknown_risk for row in combined], dtype=torch.float32),
    )
    quality = torch.tensor([row.quality for row in combined], dtype=torch.float32)
    with torch.no_grad():
        decision = decide(output, quality=quality, thresholds=thresholds)
    known_count = len(known_scores.rows)
    known_rows = tuple(
        KnownDecisionRow(
            row=row,
            label=int(decision.labels[index].item()),
            registered=bool(decision.registered[index].item()),
            explicit_unknown=bool(decision.explicit_unknown[index].item()),
            deferred=bool(decision.deferred[index].item()),
        )
        for index, row in enumerate(known_scores.rows)
    )
    proxy_rows = tuple(
        ProxyDecisionRow(
            row=row,
            label=int(decision.labels[known_count + index].item()),
            registered=bool(decision.registered[known_count + index].item()),
            explicit_unknown=bool(decision.explicit_unknown[known_count + index].item()),
            deferred=bool(decision.deferred[known_count + index].item()),
        )
        for index, row in enumerate(proxy_scores.rows)
    )
    return _create_frozen_decision_table(
        candidate_id=candidate_id,
        known_role=known_scores.role,
        proxy_role=proxy_scores.role,
        thresholds=thresholds,
        known_rows=known_rows,
        proxy_rows=proxy_rows,
        proxy_update_count=proxy_scores.update_count,
        _factory_seal=_FROZEN_TABLE_FACTORY_SEAL,
    )


def freeze_calibration_decisions(
    known_scores: KnownScoreTable,
    proxy_scores: ProxyScoreTable,
    thresholds: DecisionThresholds,
) -> FrozenDecisionTable:
    """Apply an already chosen threshold only to V_cal/P_cal for calibration receipts."""

    _validate_role_pair(
        known_scores,
        proxy_scores,
        expected_known_role=SourcePartition.V_CAL,
        expected_proxy_role=ProxyRole.P_CAL,
        permission=Permission.CALIBRATE,
    )
    return _head_decisions(known_scores, proxy_scores, thresholds, candidate_id="CALIBRATION")


def freeze_selection_decisions(
    known_scores: KnownScoreTable,
    proxy_scores: ProxyScoreTable,
    thresholds: DecisionThresholds,
    *,
    candidate_id: str,
) -> FrozenDecisionTable:
    """Apply frozen thresholds only to V_select/P_select; never re-calibrate them."""

    _validate_role_pair(
        known_scores,
        proxy_scores,
        expected_known_role=SourcePartition.V_SELECT,
        expected_proxy_role=ProxyRole.P_SELECT,
        permission=Permission.SELECT_MODEL,
    )
    return _head_decisions(known_scores, proxy_scores, thresholds, candidate_id=candidate_id)


def _require_decision_table(table: object) -> FrozenDecisionTable:
    return _validate_frozen_decision_table(table)


def known_false_rejection_rate(table: FrozenDecisionTable) -> float:
    """Return known FRR, where explicit unknown and defer both count as rejection."""

    resolved = _require_decision_table(table)
    return sum(not row.registered for row in resolved.known_rows) / len(resolved.known_rows)


def known_registered_coverage(table: FrozenDecisionTable) -> float:
    """Return known registered coverage; classification correctness is intentionally separate."""

    resolved = _require_decision_table(table)
    return sum(row.registered for row in resolved.known_rows) / len(resolved.known_rows)


def known_defer_rate(table: FrozenDecisionTable) -> float:
    """Return known defer rate without relabeling defer as explicit unknown."""

    resolved = _require_decision_table(table)
    return sum(row.deferred for row in resolved.known_rows) / len(resolved.known_rows)


def proxy_explicit_rejection_rate(table: FrozenDecisionTable) -> float:
    """Return the proxy rate that is explicitly unknown; defer never counts as success."""

    resolved = _require_decision_table(table)
    return sum(row.explicit_unknown for row in resolved.proxy_rows) / len(resolved.proxy_rows)


def _grid_values(values: Sequence[float], *, maximum_points: int = 33) -> tuple[float, ...]:
    unique = tuple(sorted({float(value) for value in values} | {0.0, 1.0}))
    if len(unique) <= maximum_points:
        return unique
    positions = {
        round(index * (len(unique) - 1) / (maximum_points - 1))
        for index in range(maximum_points)
    }
    return tuple(unique[index] for index in sorted(positions))


def empirical_threshold_grid(
    known_scores: KnownScoreTable,
    proxy_scores: ProxyScoreTable,
) -> tuple[DecisionThresholds, ...]:
    """Enumerate a deterministic finite source-only empirical threshold grid."""

    if not isinstance(known_scores, KnownScoreTable) or not isinstance(proxy_scores, ProxyScoreTable):
        raise CalibrationProtocolError("empirical grid requires immutable score tables")
    combined: tuple[KnownScoreRow | ProxyScoreRow, ...] = known_scores.rows + proxy_scores.rows
    quality_values = _grid_values([row.quality for row in combined])
    risk_values = _grid_values([row.unknown_risk for row in combined])
    return tuple(
        DecisionThresholds(tau_q=tau_q, tau_reg=tau_reg, tau_unk=tau_unk)
        for tau_q in quality_values
        for tau_reg in risk_values
        for tau_unk in risk_values
        if tau_reg <= tau_unk
    )


def calibrate_thresholds(
    known_scores: KnownScoreTable,
    proxy_scores: ProxyScoreTable,
    *,
    max_known_frr: float = 0.10,
) -> DecisionThresholds:
    """Freeze one threshold tuple using V_cal/P_cal and no selection or target inputs.

    Feasible tuples satisfy the known-FRR safety limit.  The deterministic
    lexicographic objective is proxy explicit rejection, V_cal registered
    coverage, and low V_cal defer.  Threshold magnitude only resolves an exact
    objective tie and never reads V_select, P_select, or target information.
    """

    _validate_role_pair(
        known_scores,
        proxy_scores,
        expected_known_role=SourcePartition.V_CAL,
        expected_proxy_role=ProxyRole.P_CAL,
        permission=Permission.CALIBRATE,
    )
    allowed_frr = _require_unit_interval(max_known_frr, "max_known_frr")
    if allowed_frr > 0.10:
        raise CalibrationProtocolError("formal max_known_frr must not exceed 0.10")
    best_thresholds: DecisionThresholds | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None
    for thresholds in empirical_threshold_grid(known_scores, proxy_scores):
        decisions = _head_decisions(
            known_scores,
            proxy_scores,
            thresholds,
            candidate_id="CALIBRATION",
        )
        if known_false_rejection_rate(decisions) > allowed_frr + 1e-12:
            continue
        key = (
            proxy_explicit_rejection_rate(decisions),
            known_registered_coverage(decisions),
            -known_defer_rate(decisions),
            -float(thresholds.tau_q),
            -float(thresholds.tau_reg),
            -float(thresholds.tau_unk),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = thresholds
    if best_thresholds is None:
        raise NoDeployableSeparation("NO_DEPLOYABLE_SEPARATION")
    return best_thresholds


__all__ = [
    "CalibrationProtocolError",
    "DecisionSourceReceipt",
    "FrozenDecisionTable",
    "KnownDecisionRow",
    "KnownScoreRow",
    "KnownScoreTable",
    "NoDeployableSeparation",
    "ProxyDecisionRow",
    "ProxyScoreRow",
    "ProxyScoreTable",
    "calibrate_thresholds",
    "empirical_threshold_grid",
    "freeze_calibration_decisions",
    "freeze_selection_decisions",
    "known_defer_rate",
    "known_false_rejection_rate",
    "known_registered_coverage",
    "proxy_explicit_rejection_rate",
]
