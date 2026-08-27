"""Validated canonical Phase2 profiles and deterministic coverage selection."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


PROFILE_SCHEMA = "cvs.phase2.canonical_union_profile.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
RECEIVER_TIER_NAMES = ("dense", "single_day", "many_tx")
FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
QUERY_POLICIES = ("MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE")
NEW_CLASS_SIZES = (5, 10, 20)
K_VALUES = (1, 5, 10, 20)


def _string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence")
    return tuple(str(item) for item in value)


def _integer_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence")
    items = tuple(value)
    if any(type(item) is not int for item in items):
        raise ValueError(f"{key} must contain exact integers")
    return items


def _has_duplicates(values: Sequence[str]) -> bool:
    return len(values) != len(set(values))


@dataclass(frozen=True)
class CanonicalProfile:
    schema: str
    protocol_schema: str
    source_profile_id: str
    source_receivers: tuple[str, ...]
    receiver_tiers: Mapping[str, tuple[str, ...]]
    old_tx_ids: tuple[str, ...]
    new_tx_candidates: tuple[str, ...]
    new_class_sizes: tuple[int, ...]
    k_values: tuple[int, ...]
    k_max: int
    scenarios: tuple[str, ...]
    query_policies: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CanonicalProfile":
        """Validate the v1 profile schema and normalize every collection immutably."""

        if not isinstance(payload, Mapping):
            raise ValueError("profile payload must be a mapping")
        schema = str(payload.get("schema", ""))
        if schema != PROFILE_SCHEMA:
            raise ValueError(f"schema must be {PROFILE_SCHEMA}")
        protocol_schema = str(payload.get("protocol_schema", ""))
        if protocol_schema != PROTOCOL_SCHEMA:
            raise ValueError(f"protocol_schema must be {PROTOCOL_SCHEMA}")

        tiers_value = payload.get("receiver_tiers")
        if not isinstance(tiers_value, Mapping) or set(tiers_value) != set(RECEIVER_TIER_NAMES):
            raise ValueError(f"receiver tier names must be exactly {RECEIVER_TIER_NAMES!r}")
        receiver_tiers: dict[str, tuple[str, ...]] = {}
        target_receivers: list[str] = []
        for tier_name in tiers_value:
            tier_receivers = tiers_value[tier_name]
            if isinstance(tier_receivers, (str, bytes)) or not isinstance(tier_receivers, Sequence):
                raise ValueError(f"receiver tier {tier_name} must be a sequence")
            normalized = tuple(str(receiver) for receiver in tier_receivers)
            receiver_tiers[str(tier_name)] = normalized
            target_receivers.extend(normalized)
        if _has_duplicates(target_receivers):
            raise ValueError("receiver IDs must be unique within and across tiers")
        if not target_receivers:
            raise ValueError("at least one target receiver is required")

        source_receivers = _string_tuple(payload, "source_receivers")
        if _has_duplicates(source_receivers):
            raise ValueError("source receiver IDs must be unique")
        overlap = set(source_receivers).intersection(target_receivers)
        if overlap:
            raise ValueError(f"R_s and R_t must be disjoint; R_t overlap={sorted(overlap)!r}")

        old_tx_ids = _string_tuple(payload, "old_tx_ids")
        if len(old_tx_ids) != 6:
            raise ValueError("old_tx_ids must contain exactly six IDs")
        if _has_duplicates(old_tx_ids):
            raise ValueError("old_tx_ids must contain six unique IDs")

        new_tx_candidates = _string_tuple(payload, "new_tx_candidates")
        if len(new_tx_candidates) != 22:
            raise ValueError("new_tx_candidates must contain exactly 22 IDs")
        if _has_duplicates(new_tx_candidates):
            raise ValueError("new_tx_candidates must contain 22 unique IDs")
        if set(old_tx_ids).intersection(new_tx_candidates):
            raise ValueError("old_tx_ids and new_tx_candidates must be disjoint")

        new_class_sizes = _integer_tuple(payload, "new_class_sizes")
        if new_class_sizes != NEW_CLASS_SIZES:
            raise ValueError(f"new_class_sizes must be {NEW_CLASS_SIZES!r}")
        if max(new_class_sizes) > len(new_tx_candidates):
            raise ValueError("new_class_sizes cannot exceed the candidate count")

        k_values = _integer_tuple(payload, "k_values")
        if k_values != K_VALUES:
            raise ValueError(f"k_values must be {K_VALUES!r}")
        k_max_value = payload.get("k_max")
        if type(k_max_value) is not int:
            raise ValueError("k_max must contain exact integers")
        k_max = k_max_value
        if k_max != 20:
            raise ValueError("k_max must be 20")

        scenarios = _string_tuple(payload, "scenarios")
        if scenarios != FORMAL_LEO_WEAK_SCENARIOS:
            raise ValueError(f"scenarios must be {FORMAL_LEO_WEAK_SCENARIOS!r}")
        query_policies = _string_tuple(payload, "query_policies")
        if query_policies != QUERY_POLICIES:
            raise ValueError(f"query_policies must be {QUERY_POLICIES!r}")

        return cls(
            schema=schema,
            protocol_schema=protocol_schema,
            source_profile_id=str(payload.get("source_profile_id", "")),
            source_receivers=source_receivers,
            receiver_tiers=MappingProxyType(receiver_tiers),
            old_tx_ids=old_tx_ids,
            new_tx_candidates=new_tx_candidates,
            new_class_sizes=new_class_sizes,
            k_values=k_values,
            k_max=k_max,
            scenarios=scenarios,
            query_policies=query_policies,
        )


def eligible_receivers(
    connection: sqlite3.Connection,
    *,
    registered_tx_ids: Sequence[str],
    candidate_receivers: Sequence[str],
    scenario_by_sample: Mapping[str, str],
    k: int,
) -> tuple[str, ...]:
    """Return input-order receivers with at least K eligible rows per class and scene."""

    if k <= 0:
        raise ValueError("k must be positive")
    registered = tuple(str(tx_id) for tx_id in registered_tx_ids)
    receivers = tuple(str(rx_id) for rx_id in candidate_receivers)
    if _has_duplicates(registered):
        raise ValueError("duplicate registered TX inputs are not allowed")
    if _has_duplicates(receivers):
        raise ValueError("duplicate candidate receiver inputs are not allowed")

    registered_set = set(registered)
    receiver_set = set(receivers)
    counts: Counter[tuple[str, str, str]] = Counter()
    for sample_id, tx_id, rx_id in connection.execute(
        """
        SELECT physical_sample_id, tx_id, rx_id
        FROM canonical_records
        WHERE eligible = 1
        """
    ):
        tx_label = str(tx_id)
        rx_label = str(rx_id)
        if tx_label not in registered_set or rx_label not in receiver_set:
            continue
        scene = scenario_by_sample.get(str(sample_id))
        if scene in FORMAL_LEO_WEAK_SCENARIOS:
            counts[(rx_label, tx_label, scene)] += 1

    return tuple(
        rx_id
        for rx_id in receivers
        if all(
            counts[(rx_id, tx_id, scene)] >= k
            for tx_id in registered
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        )
    )


def rank_new_classes(
    connection: sqlite3.Connection,
    profile: CanonicalProfile,
    scenario_by_sample: Mapping[str, str],
) -> Mapping[int, tuple[str, ...]]:
    """Rank all candidates once using only eligible canonical inventory coverage."""

    target_receivers = tuple(
        receiver
        for tier_receivers in profile.receiver_tiers.values()
        for receiver in tier_receivers
    )
    target_set = set(target_receivers)
    candidate_set = set(profile.new_tx_candidates)
    all_days = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT day_id FROM canonical_records WHERE eligible = 1 ORDER BY day_id"
        )
    )
    first_three_days = set(all_days[:3])
    non_single_day_receivers = target_set.difference(profile.receiver_tiers["single_day"])

    valid_rows: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    required_receiver_days: set[tuple[str, str]] = set()
    for sample_id, tx_id, rx_id, day_id in connection.execute(
        """
        SELECT physical_sample_id, tx_id, rx_id, day_id
        FROM canonical_records
        WHERE eligible = 1
        """
    ):
        tx_label = str(tx_id)
        rx_label = str(rx_id)
        day_label = str(day_id)
        if rx_label not in target_set:
            continue
        required_receiver_days.add((rx_label, day_label))
        if tx_label not in candidate_set:
            continue
        scene = scenario_by_sample.get(str(sample_id))
        if scene in FORMAL_LEO_WEAK_SCENARIOS:
            valid_rows[tx_label].append((rx_label, day_label, scene))

    ranking_keys: dict[str, tuple[int, int, int, int, str]] = {}
    for tx_id in profile.new_tx_candidates:
        rows = valid_rows[tx_id]
        receiver_scene_counts = Counter((rx_id, scene) for rx_id, _, scene in rows)
        all_receiver_kmax_feasible = all(
            receiver_scene_counts[(rx_id, scene)] >= profile.k_max
            for rx_id in target_receivers
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        )
        first_three_day_coverage = len(
            {
                day_id
                for rx_id, day_id, _ in rows
                if rx_id in non_single_day_receivers and day_id in first_three_days
            }
        )
        unique_query_capacity = len(rows)
        cell_counts = Counter(rows)
        required_cells = (
            (rx_id, day_id, scene)
            for rx_id, day_id in sorted(required_receiver_days)
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        )
        min_receiver_day_scene_count = min(
            (cell_counts[cell] for cell in required_cells),
            default=0,
        )
        ranking_keys[tx_id] = (
            -int(all_receiver_kmax_feasible),
            -int(first_three_day_coverage),
            -int(unique_query_capacity),
            -int(min_receiver_day_scene_count),
            str(tx_id),
        )

    ranked = tuple(sorted(profile.new_tx_candidates, key=ranking_keys.__getitem__))
    return MappingProxyType({size: ranked[:size] for size in profile.new_class_sizes})
