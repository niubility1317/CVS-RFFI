"""Validated canonical Phase2 profiles and deterministic coverage selection."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


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
SPLIT_MANIFEST_SCHEMA = "cvs.phase2.canonical_split_manifest.v1"


@dataclass(frozen=True)
class SceneAssignment:
    scene: str
    scene_rank: int


@dataclass(frozen=True)
class SplitRow:
    physical_sample_id: str
    source_asset: str
    source_record_index: int
    tx_id: str
    rx_id: str
    day_id: str
    scene: str
    role: str
    rank: int


@dataclass(frozen=True)
class SplitManifest:
    protocol_schema: str
    profile_id: str
    query_policy: str
    k: int
    registered_tx_ids: tuple[str, ...]
    eligible_receivers: tuple[str, ...]
    rows: tuple[SplitRow, ...]
    capsule_id: str
    split_id: str

    @property
    def support_ids(self) -> tuple[str, ...]:
        return tuple(row.physical_sample_id for row in self.rows if row.role == "support")

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(row.physical_sample_id for row in self.rows if row.role == "query")

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return tuple(row.physical_sample_id for row in self.rows)

    def to_mapping(self) -> dict[str, object]:
        rows = [
            {
                "physical_sample_id": row.physical_sample_id,
                "source_asset": row.source_asset,
                "source_record_index": row.source_record_index,
                "tx_id": row.tx_id,
                "rx_id": row.rx_id,
                "day_id": row.day_id,
                "scene": row.scene,
                "role": row.role,
                "rank": row.rank,
            }
            for row in self.rows
        ]
        return {
            "schema": SPLIT_MANIFEST_SCHEMA,
            "protocol_schema": self.protocol_schema,
            "profile_id": self.profile_id,
            "query_policy": self.query_policy,
            "k": self.k,
            "registered_tx_ids": list(self.registered_tx_ids),
            "eligible_receivers": list(self.eligible_receivers),
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "counts": {
                "registered_tx_count": len(self.registered_tx_ids),
                "eligible_receiver_count": len(self.eligible_receivers),
                "eligible_count": len(self.eligible_ids),
                "support_count": len(self.support_ids),
                "query_count": len(self.query_ids),
                "row_count": len(self.rows),
            },
            "rows": rows,
        }


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


def _row_field(row: object, key: str) -> Any:
    if isinstance(row, Mapping):
        if key not in row:
            raise ValueError(f"malformed identity row: missing {key}")
        return row[key]
    try:
        return row[key]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        try:
            return getattr(row, key)
        except AttributeError:
            raise ValueError(f"malformed identity row: missing {key}") from None


def _optional_row_field(row: object, key: str, default: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return getattr(row, key, default)


def _identity_text(row: object, key: str) -> str:
    value = _row_field(row, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"malformed identity row: {key} must be nonempty text")
    return value


def _hash_rank(seed: int, physical_sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{physical_sample_id}".encode("utf-8")).digest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def assign_scenes(
    records: Iterable[object],
    seed: int = 713101,
) -> Mapping[str, SceneAssignment]:
    """Assign one deterministic formal scene to each eligible canonical record."""

    if type(seed) is not int:
        raise ValueError("scene seed must be an exact integer")
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for raw_row in records:
        eligible = _optional_row_field(raw_row, "eligible", 1)
        if type(eligible) is not int or eligible not in (0, 1):
            raise ValueError("malformed identity row: eligible must be exactly 0 or 1")
        if eligible == 0:
            continue
        physical_sample_id = _identity_text(raw_row, "physical_sample_id")
        tx_id = _identity_text(raw_row, "tx_id")
        rx_id = _identity_text(raw_row, "rx_id")
        day_id = _identity_text(raw_row, "day_id")
        if physical_sample_id in seen:
            raise ValueError(f"duplicate physical ID: {physical_sample_id}")
        seen.add(physical_sample_id)
        groups[(tx_id, rx_id, day_id)].append(physical_sample_id)

    assignments: dict[str, SceneAssignment] = {}
    for group_key in sorted(groups):
        tx_id, rx_id, day_id = group_key
        ranked_ids = sorted(
            groups[group_key],
            key=lambda sample_id: (_hash_rank(seed, sample_id), sample_id),
        )
        offset_digest = hashlib.sha256(
            f"{seed}|{tx_id}|{rx_id}|{day_id}".encode("utf-8")
        ).digest()
        offset = int.from_bytes(offset_digest, byteorder="big") % len(FORMAL_LEO_WEAK_SCENARIOS)
        rotated = FORMAL_LEO_WEAK_SCENARIOS[offset:] + FORMAL_LEO_WEAK_SCENARIOS[:offset]
        next_rank: Counter[str] = Counter()
        for index, physical_sample_id in enumerate(ranked_ids):
            scene = rotated[index % len(rotated)]
            assignments[physical_sample_id] = SceneAssignment(
                scene=scene,
                scene_rank=next_rank[scene],
            )
            next_rank[scene] += 1
    return MappingProxyType(dict(sorted(assignments.items())))


def _normalize_inventory_records(connection_or_records: object) -> tuple[dict[str, object], ...]:
    if isinstance(connection_or_records, sqlite3.Connection):
        raw_rows: Iterable[object] = connection_or_records.execute(
            """
            SELECT physical_sample_id, tx_id, rx_id, day_id,
                   preferred_asset, preferred_source_record_index, eligible
            FROM canonical_records
            WHERE eligible = 1
            ORDER BY physical_sample_id
            """
        ).fetchall()
        columns = (
            "physical_sample_id",
            "tx_id",
            "rx_id",
            "day_id",
            "preferred_asset",
            "preferred_source_record_index",
            "eligible",
        )
        raw_rows = (dict(zip(columns, raw_row)) for raw_row in raw_rows)
    else:
        if isinstance(connection_or_records, (str, bytes, Mapping)):
            raise ValueError("connection_or_records must be a SQLite connection or record iterable")
        try:
            raw_rows = iter(connection_or_records)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError(
                "connection_or_records must be a SQLite connection or record iterable"
            ) from None

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        eligible = _optional_row_field(raw_row, "eligible", 1)
        if type(eligible) is not int or eligible not in (0, 1):
            raise ValueError("eligible must be exactly 0 or 1")
        if eligible == 0:
            continue
        physical_sample_id = _identity_text(raw_row, "physical_sample_id")
        if physical_sample_id in seen:
            raise ValueError(f"duplicate physical ID: {physical_sample_id}")
        seen.add(physical_sample_id)
        source_asset = _row_field(raw_row, "preferred_asset")
        source_record_index = _row_field(raw_row, "preferred_source_record_index")
        if not isinstance(source_asset, str) or not source_asset.strip():
            raise ValueError("preferred_asset must be nonempty text")
        if type(source_record_index) is not int or source_record_index < 0:
            raise ValueError("preferred_source_record_index must be a nonnegative exact integer")
        normalized.append(
            {
                "physical_sample_id": physical_sample_id,
                "tx_id": _identity_text(raw_row, "tx_id"),
                "rx_id": _identity_text(raw_row, "rx_id"),
                "day_id": _identity_text(raw_row, "day_id"),
                "preferred_asset": source_asset,
                "preferred_source_record_index": source_record_index,
                "eligible": 1,
            }
        )
    return tuple(sorted(normalized, key=lambda row: str(row["physical_sample_id"])))


def _records_connection(records: Sequence[Mapping[str, object]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE canonical_records (
          physical_sample_id TEXT PRIMARY KEY,
          tx_id TEXT NOT NULL,
          rx_id TEXT NOT NULL,
          day_id TEXT NOT NULL,
          preferred_asset TEXT NOT NULL,
          preferred_source_record_index INTEGER NOT NULL,
          eligible INTEGER NOT NULL CHECK (eligible IN (0,1))
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO canonical_records (
          physical_sample_id, tx_id, rx_id, day_id,
          preferred_asset, preferred_source_record_index, eligible
        ) VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        [
            (
                row["physical_sample_id"],
                row["tx_id"],
                row["rx_id"],
                row["day_id"],
                row["preferred_asset"],
                row["preferred_source_record_index"],
            )
            for row in records
        ],
    )
    return connection


def _normalize_scene_assignments(
    records: Sequence[Mapping[str, object]],
    scene_assignments: Mapping[str, object],
) -> dict[str, SceneAssignment]:
    if not isinstance(scene_assignments, Mapping):
        raise ValueError("scene_assignments must be a mapping")
    expected_ids = {str(row["physical_sample_id"]) for row in records}
    actual_ids = {str(sample_id) for sample_id in scene_assignments}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids.difference(actual_ids))
        extra = sorted(actual_ids.difference(expected_ids))
        raise ValueError(
            f"scene_assignments must exactly cover eligible IDs; missing={missing[:3]!r}, extra={extra[:3]!r}"
        )

    normalized: dict[str, SceneAssignment] = {}
    for physical_sample_id in sorted(expected_ids):
        metadata = scene_assignments[physical_sample_id]
        if isinstance(metadata, SceneAssignment):
            assignment = metadata
        elif isinstance(metadata, Mapping):
            scene = metadata.get("scene")
            scene_rank = metadata.get("scene_rank")
            assignment = SceneAssignment(scene=str(scene), scene_rank=scene_rank)  # type: ignore[arg-type]
        else:
            try:
                assignment = SceneAssignment(
                    scene=getattr(metadata, "scene"),
                    scene_rank=getattr(metadata, "scene_rank"),
                )
            except AttributeError:
                raise ValueError(
                    f"scene assignment metadata is malformed for {physical_sample_id}"
                ) from None
        if assignment.scene not in FORMAL_LEO_WEAK_SCENARIOS:
            raise ValueError(f"non-formal scene for {physical_sample_id}")
        if type(assignment.scene_rank) is not int or assignment.scene_rank < 0:
            raise ValueError(f"scene_rank must be a nonnegative exact integer for {physical_sample_id}")
        normalized[physical_sample_id] = assignment

    records_by_id = {str(row["physical_sample_id"]): row for row in records}
    ranks: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for physical_sample_id, assignment in normalized.items():
        row = records_by_id[physical_sample_id]
        ranks[
            (
                str(row["tx_id"]),
                str(row["rx_id"]),
                str(row["day_id"]),
                assignment.scene,
            )
        ].append(assignment.scene_rank)
    for cell, cell_ranks in ranks.items():
        if sorted(cell_ranks) != list(range(len(cell_ranks))):
            raise ValueError(f"scene_rank sequence is not zero-based and contiguous for {cell!r}")
    return normalized


def _normalize_class_selection(
    class_selection: Mapping[object, object],
    profile: CanonicalProfile,
    computed: Mapping[int, tuple[str, ...]],
) -> Mapping[int, tuple[str, ...]]:
    if not isinstance(class_selection, Mapping):
        raise ValueError("class_selection must be a mapping")
    normalized: dict[int, tuple[str, ...]] = {}
    for size in profile.new_class_sizes:
        candidates = (
            class_selection.get(size),
            class_selection.get(str(size)),
            class_selection.get(f"Y_new{size}"),
        )
        raw_value = next((candidate for candidate in candidates if candidate is not None), None)
        if isinstance(raw_value, (str, bytes)) or not isinstance(raw_value, Sequence):
            raise ValueError(f"class_selection is missing Y_new{size}")
        value = tuple(str(tx_id) for tx_id in raw_value)
        if value != computed[size]:
            raise ValueError(f"class_selection Y_new{size} does not match deterministic ranking")
        normalized[size] = value
    return MappingProxyType(normalized)


def _ordered_split_rows(
    rows: Iterable[SplitRow],
    registered_tx_ids: Sequence[str],
    eligible_rx_ids: Sequence[str],
) -> tuple[SplitRow, ...]:
    tx_order = {tx_id: index for index, tx_id in enumerate(registered_tx_ids)}
    rx_order = {rx_id: index for index, rx_id in enumerate(eligible_rx_ids)}
    scene_order = {scene: index for index, scene in enumerate(FORMAL_LEO_WEAK_SCENARIOS)}
    role_order = {"support": 0, "query": 1}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                rx_order[row.rx_id],
                scene_order[row.scene],
                tx_order[row.tx_id],
                role_order[row.role],
                row.rank,
                row.day_id,
                row.physical_sample_id,
            ),
        )
    )


def build_split_manifest(
    connection_or_records: object,
    profile: CanonicalProfile,
    k: int,
    query_policy: str,
    *,
    scene_assignments: Mapping[str, object],
    class_selection: Mapping[object, object],
    support_seed: int = 713101,
) -> SplitManifest:
    """Build one deterministic maximal-class Phase2 split manifest."""

    if not isinstance(profile, CanonicalProfile) or profile.protocol_schema != PROTOCOL_SCHEMA:
        raise ValueError(f"profile must use protocol_schema={PROTOCOL_SCHEMA}")
    if type(k) is not int or k not in profile.k_values:
        raise ValueError(f"K must be one of {profile.k_values!r}")
    if query_policy not in profile.query_policies:
        raise ValueError(f"query policy must be one of {profile.query_policies!r}")
    if type(support_seed) is not int:
        raise ValueError("support_seed must be an exact integer")

    records = _normalize_inventory_records(connection_or_records)
    if not records:
        raise ValueError("canonical inventory contains no eligible records")
    assignments = _normalize_scene_assignments(records, scene_assignments)
    plain_scenes = {
        physical_sample_id: assignment.scene
        for physical_sample_id, assignment in assignments.items()
    }
    working_connection = _records_connection(records)
    try:
        computed_selection = rank_new_classes(working_connection, profile, plain_scenes)
        normalized_selection = _normalize_class_selection(
            class_selection,
            profile,
            computed_selection,
        )
        registered_tx_ids = profile.old_tx_ids + normalized_selection[20]
        if _has_duplicates(registered_tx_ids):
            raise ValueError("registered TX IDs must be unique")

        all_target_receivers = tuple(
            receiver
            for tier_receivers in profile.receiver_tiers.values()
            for receiver in tier_receivers
        )
        candidate_receivers = (
            profile.receiver_tiers["dense"]
            if query_policy == "BALANCED_4DAY_CORE"
            else all_target_receivers
        )
        eligibility_k = profile.k_max if query_policy == "BALANCED_4DAY_CORE" else k
        retained_receivers = eligible_receivers(
            working_connection,
            registered_tx_ids=registered_tx_ids,
            candidate_receivers=candidate_receivers,
            scenario_by_sample=plain_scenes,
            k=eligibility_k,
        )
    finally:
        working_connection.close()
    if not retained_receivers:
        raise ValueError(f"profile is infeasible for {query_policy} at K={k}")

    registered_set = set(registered_tx_ids)
    target_set = set(all_target_receivers)
    retained_receiver_set = set(retained_receivers)
    relevant_capsule_rows = [
        row
        for row in records
        if str(row["tx_id"]) in registered_set and str(row["rx_id"]) in target_set
    ]
    if not relevant_capsule_rows:
        raise ValueError("maximal registered class pool has no eligible target records")
    capsule_id = _canonical_sha256(
        {
            "protocol_schema": profile.protocol_schema,
            "registered_tx_ids": list(registered_tx_ids),
            "samples": [
                {
                    "physical_sample_id": str(row["physical_sample_id"]),
                    "scene": assignments[str(row["physical_sample_id"])].scene,
                    "scene_rank": assignments[str(row["physical_sample_id"])].scene_rank,
                }
                for row in relevant_capsule_rows
            ],
        }
    )

    retained_rows = [
        row
        for row in records
        if str(row["tx_id"]) in registered_set
        and str(row["rx_id"]) in retained_receiver_set
    ]
    rows_by_support_cell: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in retained_rows:
        physical_sample_id = str(row["physical_sample_id"])
        rows_by_support_cell[
            (
                str(row["rx_id"]),
                assignments[physical_sample_id].scene,
                str(row["tx_id"]),
            )
        ].append(row)
    support_ranks: dict[str, int] = {}
    for rx_id in retained_receivers:
        for scene in FORMAL_LEO_WEAK_SCENARIOS:
            for tx_id in registered_tx_ids:
                cell = rows_by_support_cell[(rx_id, scene, tx_id)]
                if len(cell) < eligibility_k:
                    raise ValueError(
                        f"support shortfall for rx={rx_id}, scene={scene}, tx={tx_id}"
                    )
                ranked = sorted(
                    cell,
                    key=lambda row: (
                        _hash_rank(support_seed, str(row["physical_sample_id"])),
                        str(row["physical_sample_id"]),
                    ),
                )
                for rank, row in enumerate(ranked):
                    support_ranks[str(row["physical_sample_id"])] = rank

    selected_query_ids: set[str]
    if query_policy == "MAXQ_ALL_UNIQUE":
        selected_query_ids = {
            str(row["physical_sample_id"])
            for row in retained_rows
            if support_ranks[str(row["physical_sample_id"])] >= k
        }
    else:
        required_days = tuple(sorted({str(row["day_id"]) for row in retained_rows}))
        if len(required_days) != 4:
            raise ValueError(
                f"BALANCED_4DAY_CORE requires exactly four target days, got {required_days!r}"
            )
        query_pools: dict[
            tuple[str, str, str, str], list[Mapping[str, object]]
        ] = defaultdict(list)
        for row in retained_rows:
            physical_sample_id = str(row["physical_sample_id"])
            if support_ranks[physical_sample_id] < profile.k_max:
                continue
            query_pools[
                (
                    str(row["tx_id"]),
                    str(row["rx_id"]),
                    str(row["day_id"]),
                    assignments[physical_sample_id].scene,
                )
            ].append(row)
        required_cells = tuple(
            (tx_id, rx_id, day_id, scene)
            for tx_id in registered_tx_ids
            for rx_id in retained_receivers
            for day_id in required_days
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        )
        common_capacity = min((len(query_pools[cell]) for cell in required_cells), default=0)
        if common_capacity <= 0:
            raise ValueError("BALANCED_4DAY_CORE has an absent or zero query cell")
        selected_query_ids = set()
        for cell in required_cells:
            ranked_pool = sorted(
                query_pools[cell],
                key=lambda row: (
                    _hash_rank(support_seed, str(row["physical_sample_id"])),
                    str(row["physical_sample_id"]),
                ),
            )
            selected_query_ids.update(
                str(row["physical_sample_id"])
                for row in ranked_pool[:common_capacity]
            )

    output_rows: list[SplitRow] = []
    support_ids: set[str] = set()
    for row in retained_rows:
        physical_sample_id = str(row["physical_sample_id"])
        rank = support_ranks[physical_sample_id]
        if rank < k:
            role = "support"
            support_ids.add(physical_sample_id)
        elif physical_sample_id in selected_query_ids:
            role = "query"
        else:
            continue
        output_rows.append(
            SplitRow(
                physical_sample_id=physical_sample_id,
                source_asset=str(row["preferred_asset"]),
                source_record_index=int(row["preferred_source_record_index"]),
                tx_id=str(row["tx_id"]),
                rx_id=str(row["rx_id"]),
                day_id=str(row["day_id"]),
                scene=assignments[physical_sample_id].scene,
                role=role,
                rank=rank,
            )
        )
    if support_ids.intersection(selected_query_ids):
        raise ValueError("support/query physical IDs are not disjoint")
    expected_support_count = len(retained_receivers) * len(registered_tx_ids) * len(
        FORMAL_LEO_WEAK_SCENARIOS
    ) * k
    if len(support_ids) != expected_support_count:
        raise ValueError(
            f"support count mismatch: expected {expected_support_count}, got {len(support_ids)}"
        )

    ordered_rows = _ordered_split_rows(
        output_rows,
        registered_tx_ids,
        retained_receivers,
    )
    split_id = _canonical_sha256(
        {
            "capsule_id": capsule_id,
            "profile_id": profile.source_profile_id,
            "query_policy": query_policy,
            "k": k,
            "registered_tx_ids": list(registered_tx_ids),
            "eligible_receivers": list(retained_receivers),
            "rows": [
                {
                    "physical_sample_id": row.physical_sample_id,
                    "role": row.role,
                    "rank": row.rank,
                }
                for row in ordered_rows
            ],
        }
    )
    return SplitManifest(
        protocol_schema=profile.protocol_schema,
        profile_id=profile.source_profile_id,
        query_policy=query_policy,
        k=k,
        registered_tx_ids=registered_tx_ids,
        eligible_receivers=retained_receivers,
        rows=ordered_rows,
        capsule_id=capsule_id,
        split_id=split_id,
    )
