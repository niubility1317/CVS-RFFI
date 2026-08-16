"""Machine-executable Phase1 MIRAGE-OWDG data-permission policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import ClassVar, Collection, Iterable, Mapping, Union


class Phase1PolicyError(ValueError):
    """Raised when a Phase1 source or target permission rule is violated."""


class SourcePartition(str, Enum):
    """Immutable source partitions after the one-time physical-ID split."""

    L_S = "L_s"
    U_S = "U_s"
    V_CAL = "V_cal"
    V_SELECT = "V_select"


class ProxyRole(str, Enum):
    """Source-only proxy roles derived from exactly one source partition."""

    PROXY_TRAIN = "proxy_train"
    P_CAL = "P_cal"
    P_SELECT = "P_select"


class TargetRole(str, Enum):
    """Truth-blind target roles; neither role is a development input."""

    TARGET_KNOWN = "target_known"
    TARGET_UNKNOWN = "target_unknown"


class Permission(str, Enum):
    """Operations that can change a Phase1 development decision or state."""

    TRAIN = "train"
    REJECTION_GRADIENT = "rejection_gradient"
    CALIBRATE = "calibrate"
    SELECT_MODEL = "select_model"
    SELECTIVE_RERUN = "selective_rerun"


PolicyRole = Union[SourcePartition, ProxyRole, TargetRole]


@dataclass(frozen=True)
class SampleIdentity:
    """The identity metadata required to verify a source physical-ID partition."""

    physical_sample_id: str
    tx_id: str


@dataclass(frozen=True)
class Phase1DataPolicy:
    """Authorize Phase1 source development while keeping target evaluation blind."""

    SPLIT_FRACTIONS: ClassVar[Mapping[SourcePartition, Fraction]] = MappingProxyType(
        {
            SourcePartition.L_S: Fraction(7, 100),
            SourcePartition.U_S: Fraction(63, 100),
            SourcePartition.V_CAL: Fraction(15, 100),
            SourcePartition.V_SELECT: Fraction(15, 100),
        }
    )
    _PARTITION_ORDER: ClassVar[tuple[SourcePartition, ...]] = tuple(SPLIT_FRACTIONS)
    _PROXY_ORIGINS: ClassVar[Mapping[ProxyRole, SourcePartition]] = MappingProxyType(
        {
            ProxyRole.PROXY_TRAIN: SourcePartition.L_S,
            ProxyRole.P_CAL: SourcePartition.V_CAL,
            ProxyRole.P_SELECT: SourcePartition.V_SELECT,
        }
    )
    _PERMISSIONS: ClassVar[Mapping[PolicyRole, frozenset[Permission]]] = MappingProxyType(
        {
            SourcePartition.L_S: frozenset({Permission.TRAIN}),
            SourcePartition.U_S: frozenset({Permission.TRAIN}),
            SourcePartition.V_CAL: frozenset(),
            SourcePartition.V_SELECT: frozenset(),
            ProxyRole.PROXY_TRAIN: frozenset({Permission.REJECTION_GRADIENT}),
            ProxyRole.P_CAL: frozenset({Permission.CALIBRATE}),
            ProxyRole.P_SELECT: frozenset({Permission.SELECT_MODEL}),
            TargetRole.TARGET_KNOWN: frozenset(),
            TargetRole.TARGET_UNKNOWN: frozenset(),
        }
    )

    def partition_counts(self, sample_count: int) -> dict[SourcePartition, int]:
        """Return a deterministic integer allocation for the approved split ratios."""

        if sample_count < 0:
            raise Phase1PolicyError("sample_count must be non-negative")
        raw = {
            partition: self.SPLIT_FRACTIONS[partition] * sample_count
            for partition in self._PARTITION_ORDER
        }
        counts = {partition: raw[partition].numerator // raw[partition].denominator for partition in raw}
        remaining = sample_count - sum(counts.values())
        ranked = sorted(
            self._PARTITION_ORDER,
            key=lambda partition: (
                -(raw[partition] - counts[partition]),
                self._PARTITION_ORDER.index(partition),
            ),
        )
        for partition in ranked[:remaining]:
            counts[partition] += 1
        return counts

    def validate_source_partitions(
        self,
        *,
        l_s: Iterable[SampleIdentity],
        u_s: Iterable[SampleIdentity],
        v_cal: Iterable[SampleIdentity],
        v_select: Iterable[SampleIdentity],
    ) -> None:
        """Require physical-ID disjointness without imposing TX-identity disjointness."""

        used_ids: set[str] = set()
        for samples in (l_s, u_s, v_cal, v_select):
            for sample in samples:
                if not sample.physical_sample_id:
                    raise Phase1PolicyError("physical_sample_id must be non-empty")
                if sample.physical_sample_id in used_ids:
                    raise Phase1PolicyError(
                        f"physical_sample_id reused across source partitions: {sample.physical_sample_id}"
                    )
                used_ids.add(sample.physical_sample_id)

    def proxy_origin_is_allowed(self, proxy_role: ProxyRole, source_partition: SourcePartition) -> bool:
        """Return whether a source partition is the sole legal origin for a proxy role."""

        return self._PROXY_ORIGINS[proxy_role] is source_partition

    def require_proxy_origin(self, proxy_role: ProxyRole, source_partition: SourcePartition) -> None:
        """Fail closed when a proxy role is materialized from the wrong source partition."""

        if not self.proxy_origin_is_allowed(proxy_role, source_partition):
            required = self._PROXY_ORIGINS[proxy_role].value
            raise Phase1PolicyError(f"{proxy_role.value} must originate from {required}")

    def allows(self, role: PolicyRole, permission: Permission) -> bool:
        """Return the immutable permission decision for one data role and operation."""

        return permission in self._PERMISSIONS.get(role, frozenset())

    def require_permission(self, role: PolicyRole, permission: Permission) -> None:
        """Fail closed for a prohibited state-changing operation."""

        if not self.allows(role, permission):
            raise Phase1PolicyError(f"{role.value} is not permitted to {permission.value}")

    def validate_target_unknown_identities(
        self,
        *,
        target_unknown_tx_ids: Collection[str],
        source_train_tx_ids: Collection[str],
        source_validation_tx_ids: Collection[str],
    ) -> None:
        """Require true target-unknown TX identities to be absent from source development."""

        source_tx_ids = set(source_train_tx_ids) | set(source_validation_tx_ids)
        overlap = set(target_unknown_tx_ids) & source_tx_ids
        if overlap:
            rendered = ", ".join(sorted(overlap))
            raise Phase1PolicyError(
                "target unknown TX identity overlaps source train/validation TX: " + rendered
            )
