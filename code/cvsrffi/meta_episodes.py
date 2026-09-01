from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


class EpisodeKind(str, Enum):
    SAME_DOMAIN = "Q_SAME_DOMAIN"
    RX_HOLDOUT = "Q_RX_HOLDOUT"
    DAY_CHANNEL_HOLDOUT = "Q_DAY_CHANNEL_HOLDOUT"
    CLEAN_TO_LEO = "Q_CLEAN_TO_LEO"
    LEO_CROSS = "Q_LEO_CROSS"


_DEFAULT_EPISODE_WEIGHTS = {
    EpisodeKind.SAME_DOMAIN: 0.40,
    EpisodeKind.RX_HOLDOUT: 0.20,
    EpisodeKind.DAY_CHANNEL_HOLDOUT: 0.15,
    EpisodeKind.CLEAN_TO_LEO: 0.15,
    EpisodeKind.LEO_CROSS: 0.10,
}

MARC_OT_CANONICAL_K = (1, 2, 5, 10, 20)
MARC_OT_LEO_WEAK_SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)


@dataclass(frozen=True)
class MetaSampleRef:
    dataset_index: int
    tx_i: int
    rx_i: int
    day_i: int
    eq_i: int
    capture_block_i: int
    physical_sample_id: str
    role: str
    view: str


@dataclass(frozen=True)
class MetaEpisode:
    kind: EpisodeKind
    support: tuple[MetaSampleRef, ...]
    query_adapt: tuple[MetaSampleRef, ...]
    query_guard: tuple[MetaSampleRef, ...]
    adapt_class_ids: frozenset[int]
    guard_class_ids: frozenset[int]
    k_shot: int
    seed: int
    query_per_class: int = 2


def _episode_kind(value: Any) -> EpisodeKind:
    if isinstance(value, EpisodeKind):
        return value
    try:
        return EpisodeKind(str(value))
    except ValueError:
        try:
            return EpisodeKind[str(value)]
        except KeyError as exc:
            raise ValueError(f"Unknown meta episode kind: {value!r}") from exc


def _normalise_episode_weights(
    weights: Mapping[Any, float] | None,
) -> Dict[EpisodeKind, float]:
    normalised = {kind: 0.0 for kind in EpisodeKind}
    source = _DEFAULT_EPISODE_WEIGHTS if weights is None else weights
    for key, value in source.items():
        kind = _episode_kind(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"episode weight for {kind.value} must be numeric") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"episode weight for {kind.value} must be finite and non-negative")
        normalised[kind] = numeric
    if sum(normalised.values()) <= 0.0:
        raise ValueError("episode_weights must contain at least one positive weight")
    return normalised


@dataclass(frozen=True)
class MetaEpisodeSamplerConfig:
    k_choices: tuple[int, ...] = MARC_OT_CANONICAL_K
    query_per_class: int = 2
    allowed_roles: tuple[str, ...] | None = None
    training: bool = True
    episode_weights: Mapping[Any, float] = field(
        default_factory=lambda: dict(_DEFAULT_EPISODE_WEIGHTS)
    )
    partial_coverage_probability: float = 0.30
    partial_class_fraction: tuple[float, float] = (0.50, 0.80)

    def __post_init__(self) -> None:
        choices = tuple(int(value) for value in self.k_choices)
        if not choices or any(value <= 0 for value in choices):
            raise ValueError("k_choices must contain at least one positive integer")
        object.__setattr__(self, "k_choices", choices)

        query_per_class = int(self.query_per_class)
        if query_per_class <= 0:
            raise ValueError("query_per_class must be positive")
        object.__setattr__(self, "query_per_class", query_per_class)

        expected_roles = ("L_s",) if bool(self.training) else ("V_cal", "V_select")
        if self.allowed_roles is None:
            roles = expected_roles
        else:
            roles = tuple(str(role) for role in self.allowed_roles)
        if roles != expected_roles:
            sampler_name = "training" if bool(self.training) else "evaluation"
            raise ValueError(
                f"{sampler_name} sampler allowed_roles must be exactly {expected_roles!r}; "
                f"got {roles!r}"
            )
        object.__setattr__(self, "allowed_roles", roles)
        object.__setattr__(self, "training", bool(self.training))
        object.__setattr__(
            self,
            "episode_weights",
            _normalise_episode_weights(self.episode_weights),
        )

        probability = float(self.partial_coverage_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("partial_coverage_probability must be in [0, 1]")
        object.__setattr__(self, "partial_coverage_probability", probability)

        try:
            lower, upper = tuple(float(value) for value in self.partial_class_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("partial_class_fraction must contain two numeric bounds") from exc
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0.0 < lower <= upper < 1.0
        ):
            raise ValueError("partial_class_fraction must satisfy 0 < lower <= upper < 1")
        object.__setattr__(self, "partial_class_fraction", (lower, upper))


class HierarchicalMetaEpisodeSampler:
    """Sample source-only episodes with explicit cross-domain relationships."""

    _LEO_PREFIX = "leo_"
    _LEO_SUFFIX = "_weak"

    def __init__(
        self,
        refs: Sequence[MetaSampleRef],
        config: MetaEpisodeSamplerConfig | None = None,
    ) -> None:
        self.config = config if config is not None else MetaEpisodeSamplerConfig()
        self.refs = tuple(refs)
        if not self.refs:
            raise ValueError("Meta episode sampler requires a non-empty sample pool")

        allowed_roles = set(self.config.allowed_roles or ())
        invalid_roles = sorted({str(ref.role) for ref in self.refs if ref.role not in allowed_roles})
        if invalid_roles:
            raise ValueError(
                "Meta episode sample pool contains role(s) outside allowed_roles: "
                f"{invalid_roles!r}; allowed_roles={self.config.allowed_roles!r}"
            )

        self._by_class: Dict[int, Tuple[MetaSampleRef, ...]] = {}
        grouped: Dict[int, List[MetaSampleRef]] = defaultdict(list)
        for ref in self.refs:
            if not str(ref.physical_sample_id):
                raise ValueError("Meta episode refs require a non-empty physical_sample_id")
            grouped[int(ref.tx_i)].append(ref)
        for class_id, rows in grouped.items():
            self._by_class[int(class_id)] = tuple(
                sorted(rows, key=lambda row: (int(row.dataset_index), str(row.view)))
            )
        self._class_ids = tuple(sorted(self._by_class))
        self._descriptors = frozenset(self._descriptor_set(self.refs))
        self._candidate_plan_cache: Dict[
            EpisodeKind,
            tuple[tuple[dict[str, Any], dict[str, Any]], ...],
        ] = {}
        self._pool_cache: Dict[
            tuple[int, tuple[tuple[str, Any], ...]],
            Dict[str, MetaSampleRef],
        ] = {}

    @staticmethod
    def _is_leo(view: str) -> bool:
        value = str(view)
        return value.startswith(HierarchicalMetaEpisodeSampler._LEO_PREFIX) and value.endswith(
            HierarchicalMetaEpisodeSampler._LEO_SUFFIX
        )

    @staticmethod
    def _row_matches(row: MetaSampleRef, spec: Mapping[str, Any]) -> bool:
        for attr, expected in spec.items():
            if attr == "view":
                if str(row.view) != str(expected):
                    return False
            elif attr == "is_leo":
                if HierarchicalMetaEpisodeSampler._is_leo(row.view) != bool(expected):
                    return False
            elif int(getattr(row, attr)) != int(expected):
                return False
        return True

    def _pool(self, class_id: int, spec: Mapping[str, Any]) -> Dict[str, MetaSampleRef]:
        cache_key = (int(class_id), tuple(sorted((str(key), value) for key, value in spec.items())))
        cached = self._pool_cache.get(cache_key)
        if cached is not None:
            return cached
        unique: Dict[str, MetaSampleRef] = {}
        for row in self._by_class[int(class_id)]:
            if self._row_matches(row, spec):
                physical_id = str(row.physical_sample_id)
                previous = unique.get(physical_id)
                if previous is None or int(row.dataset_index) < int(previous.dataset_index):
                    unique[physical_id] = row
        self._pool_cache[cache_key] = unique
        return unique

    @staticmethod
    def _descriptor_set(refs: Iterable[MetaSampleRef]) -> set[tuple[Any, ...]]:
        return {
            (
                int(row.rx_i),
                int(row.day_i),
                int(row.eq_i),
                int(row.capture_block_i),
                str(row.view),
            )
            for row in refs
        }

    def _candidate_plans(
        self,
        kind: EpisodeKind,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        descriptors = self._descriptors
        plans: set[tuple[tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]]] = set()

        def add(support_spec: dict[str, Any], query_spec: dict[str, Any]) -> None:
            support_key = tuple(sorted(support_spec.items()))
            query_key = tuple(sorted(query_spec.items()))
            plans.add((support_key, query_key))

        if kind is EpisodeKind.SAME_DOMAIN:
            for rx_i, day_i, eq_i, block_i, view in descriptors:
                add(
                    {"rx_i": rx_i, "day_i": day_i, "eq_i": eq_i, "view": view},
                    {
                        "rx_i": rx_i,
                        "day_i": day_i,
                        "eq_i": eq_i,
                        "capture_block_i": block_i,
                        "view": view,
                    },
                )
        elif kind is EpisodeKind.RX_HOLDOUT:
            # Receiver holdout is a knowledge holdout, not a cross-receiver
            # support shortcut: both inner support and hidden query belong to
            # pseudo-target receiver d.  The Phase1 entry separately removes
            # d's expert coordinates from the historical bank for this fold.
            for rx_i, day_i, eq_i, block_i, view in descriptors:
                add(
                    {"rx_i": rx_i, "day_i": day_i, "eq_i": eq_i, "view": view},
                    {
                        "rx_i": rx_i,
                        "day_i": day_i,
                        "eq_i": eq_i,
                        "capture_block_i": block_i,
                        "view": view,
                    },
                )
        elif kind is EpisodeKind.DAY_CHANNEL_HOLDOUT:
            grouped = defaultdict(list)
            for descriptor in descriptors:
                grouped[(descriptor[0], descriptor[2], descriptor[4])].append(descriptor)
            for rows in grouped.values():
                for support in rows:
                    for query in rows:
                        if support[1] == query[1]:
                            continue
                        add(
                            {
                                "rx_i": support[0],
                                "day_i": support[1],
                                "eq_i": support[2],
                                "view": support[4],
                            },
                            {
                                "rx_i": query[0],
                                "day_i": query[1],
                                "eq_i": query[2],
                                "capture_block_i": query[3],
                                "view": query[4],
                            },
                        )
        elif kind is EpisodeKind.CLEAN_TO_LEO:
            grouped = defaultdict(list)
            for descriptor in descriptors:
                grouped[descriptor[:3]].append(descriptor)
            for rows in grouped.values():
                clean_rows = [row for row in rows if row[4] == "clean"]
                leo_rows = [row for row in rows if self._is_leo(row[4])]
                for rx_i, day_i, eq_i, _block_i, _view in clean_rows:
                    for leo_rx, leo_day, leo_eq, leo_block, leo_view in leo_rows:
                        add(
                            {
                                "rx_i": rx_i,
                                "day_i": day_i,
                                "eq_i": eq_i,
                                "view": "clean",
                            },
                            {
                                "rx_i": leo_rx,
                                "day_i": leo_day,
                                "eq_i": leo_eq,
                                "capture_block_i": leo_block,
                                "view": leo_view,
                            },
                        )
        elif kind is EpisodeKind.LEO_CROSS:
            grouped = defaultdict(list)
            for descriptor in descriptors:
                if self._is_leo(descriptor[4]):
                    grouped[descriptor[:3]].append(descriptor)
            for rows in grouped.values():
                for support in rows:
                    for query in rows:
                        if support[4] == query[4]:
                            continue
                        add(
                            {
                                "rx_i": support[0],
                                "day_i": support[1],
                                "eq_i": support[2],
                                "view": support[4],
                            },
                            {
                                "rx_i": query[0],
                                "day_i": query[1],
                                "eq_i": query[2],
                                "capture_block_i": query[3],
                                "view": query[4],
                            },
                        )
        else:  # pragma: no cover - protected by EpisodeKind and config normalization
            raise ValueError(f"Unsupported meta episode kind: {kind!r}")

        return [
            (dict(support_key), dict(query_key))
            for support_key, query_key in sorted(plans, key=repr)
        ]

    def _cached_candidate_plans(
        self,
        kind: EpisodeKind,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        cached = self._candidate_plan_cache.get(kind)
        if cached is None:
            cached = tuple(self._candidate_plans(kind))
            self._candidate_plan_cache[kind] = cached
        return list(cached)

    def _plan_available(
        self,
        plan: tuple[dict[str, Any], dict[str, Any]],
        adapt_class_ids: Sequence[int],
        guard_class_ids: Sequence[int],
        k_shot: int,
    ) -> bool:
        support_spec, query_spec = plan
        query_count = int(self.config.query_per_class)
        for class_id in adapt_class_ids:
            support_pool = self._pool(class_id, support_spec)
            query_pool = self._pool(class_id, query_spec)
            if len(support_pool) < k_shot or len(query_pool) < query_count:
                return False
            if len(set(support_pool).union(query_pool)) < k_shot + query_count:
                return False
        for class_id in guard_class_ids:
            if len(self._pool(class_id, query_spec)) < query_count:
                return False
        return True

    @staticmethod
    def _choose_rows(
        pool: Mapping[str, MetaSampleRef],
        count: int,
        rng: random.Random,
        blocked_ids: set[str],
    ) -> tuple[MetaSampleRef, ...]:
        candidates = [
            row for physical_id, row in pool.items() if str(physical_id) not in blocked_ids
        ]
        if len(candidates) < int(count):
            raise ValueError(
                "Meta episode pool does not contain enough disjoint physical samples: "
                f"need {int(count)}, available {len(candidates)}"
            )
        rng.shuffle(candidates)
        selected = candidates[: int(count)]
        return tuple(sorted(selected, key=lambda row: int(row.dataset_index)))

    def _sample_partition(
        self,
        plan: tuple[dict[str, Any], dict[str, Any]],
        adapt_class_ids: Sequence[int],
        guard_class_ids: Sequence[int],
        k_shot: int,
        rng: random.Random,
    ) -> tuple[tuple[MetaSampleRef, ...], tuple[MetaSampleRef, ...], tuple[MetaSampleRef, ...]]:
        support_spec, query_spec = plan
        query_count = int(self.config.query_per_class)
        used_ids: set[str] = set()
        support_rows: list[MetaSampleRef] = []
        query_adapt_rows: list[MetaSampleRef] = []
        query_guard_rows: list[MetaSampleRef] = []

        class_order = list(adapt_class_ids) + list(guard_class_ids)
        rng.shuffle(class_order)
        adapt_set = set(adapt_class_ids)
        for class_id in class_order:
            query_pool = self._pool(class_id, query_spec)
            if class_id in adapt_set:
                support_pool = self._pool(class_id, support_spec)
                selected_query = self._choose_rows(query_pool, query_count, rng, used_ids)
                used_ids.update(str(row.physical_sample_id) for row in selected_query)
                selected_support = self._choose_rows(support_pool, k_shot, rng, used_ids)
                used_ids.update(str(row.physical_sample_id) for row in selected_support)
                support_rows.extend(selected_support)
                query_adapt_rows.extend(selected_query)
            else:
                selected_guard = self._choose_rows(query_pool, query_count, rng, used_ids)
                used_ids.update(str(row.physical_sample_id) for row in selected_guard)
                query_guard_rows.extend(selected_guard)

        return (
            tuple(sorted(support_rows, key=lambda row: (int(row.tx_i), int(row.dataset_index)))),
            tuple(sorted(query_adapt_rows, key=lambda row: (int(row.tx_i), int(row.dataset_index)))),
            tuple(sorted(query_guard_rows, key=lambda row: (int(row.tx_i), int(row.dataset_index)))),
        )

    def _sample_class_sets(self, rng: random.Random) -> tuple[frozenset[int], frozenset[int]]:
        classes = list(self._class_ids)
        if len(classes) <= 1 or rng.random() >= self.config.partial_coverage_probability:
            return frozenset(classes), frozenset()

        lower, upper = self.config.partial_class_fraction
        min_count = max(1, int(math.ceil(len(classes) * lower)))
        max_count = min(len(classes) - 1, int(math.floor(len(classes) * upper)))
        if max_count < min_count:
            min_count = max_count = max(1, len(classes) - 1)
        adapt_count = rng.randint(min_count, max_count)
        adapt = frozenset(rng.sample(classes, adapt_count))
        guard = frozenset(set(classes).difference(adapt))
        return adapt, guard

    def _choose_kind(self, rng: random.Random) -> EpisodeKind:
        kinds = list(EpisodeKind)
        weights = [float(self.config.episode_weights.get(kind, 0.0)) for kind in kinds]
        return rng.choices(kinds, weights=weights, k=1)[0]

    def sample(self, seed: int) -> MetaEpisode:
        """Return one deterministic episode for ``seed`` or reject its pool."""
        seed_i = int(seed)
        rng = random.Random(seed_i)
        kind = self._choose_kind(rng)
        k_shot = int(rng.choice(self.config.k_choices))
        return self._sample_requested(
            kind=kind,
            k_shot=k_shot,
            seed=seed_i,
            rng=rng,
        )

    def sample_requested(
        self,
        *,
        kind: EpisodeKind | str,
        k_shot: int,
        seed: int,
        support_view: str | None = None,
        query_view: str | None = None,
    ) -> MetaEpisode:
        """Sample one explicit coverage cell without weakening its pool checks."""

        resolved_kind = _episode_kind(kind)
        if (
            isinstance(k_shot, bool)
            or not isinstance(k_shot, int)
            or int(k_shot) not in self.config.k_choices
        ):
            raise ValueError("requested k_shot is outside sampler k_choices")
        return self._sample_requested(
            kind=resolved_kind,
            k_shot=int(k_shot),
            seed=int(seed),
            support_view=support_view,
            query_view=query_view,
        )

    def _sample_requested(
        self,
        *,
        kind: EpisodeKind,
        k_shot: int,
        seed: int,
        support_view: str | None = None,
        query_view: str | None = None,
        rng: random.Random | None = None,
    ) -> MetaEpisode:
        seed_i = int(seed)
        rng = random.Random(seed_i) if rng is None else rng
        adapt_class_ids, guard_class_ids = self._sample_class_sets(rng)

        plans = self._cached_candidate_plans(kind)
        if support_view is not None:
            plans = [
                plan for plan in plans if str(plan[0].get("view")) == str(support_view)
            ]
        if query_view is not None:
            plans = [
                plan for plan in plans if str(plan[1].get("view")) == str(query_view)
            ]
        rng.shuffle(plans)
        valid_plans = [
            plan
            for plan in plans
            if self._plan_available(plan, sorted(adapt_class_ids), sorted(guard_class_ids), k_shot)
        ]
        if not valid_plans:
            raise ValueError(
                f"Cannot construct {kind.value} meta episode: no domain plan has "
                f"enough disjoint physical samples for k_shot={k_shot} and "
                f"query_per_class={self.config.query_per_class}"
            )

        last_error: ValueError | None = None
        for plan in valid_plans:
            try:
                support, query_adapt, query_guard = self._sample_partition(
                    plan,
                    sorted(adapt_class_ids),
                    sorted(guard_class_ids),
                    k_shot,
                    rng,
                )
                break
            except ValueError as exc:
                last_error = exc
        else:
            raise ValueError(
                f"Cannot construct {kind.value} meta episode with globally disjoint "
                "physical sample IDs"
            ) from last_error

        episode = MetaEpisode(
            kind=kind,
            support=support,
            query_adapt=query_adapt,
            query_guard=query_guard,
            adapt_class_ids=frozenset(adapt_class_ids),
            guard_class_ids=frozenset(guard_class_ids),
            k_shot=k_shot,
            seed=seed_i,
            query_per_class=int(self.config.query_per_class),
        )
        self._assert_episode(episode)
        return episode

    def _assert_episode(self, episode: MetaEpisode) -> None:
        rows = episode.support + episode.query_adapt + episode.query_guard
        if not rows:
            raise ValueError("Meta episode cannot be empty")
        allowed_roles = set(self.config.allowed_roles or ())
        if any(row.role not in allowed_roles for row in rows):
            raise ValueError("Meta episode contains a row outside allowed_roles")
        support_ids = [str(row.physical_sample_id) for row in episode.support]
        query_ids = [
            str(row.physical_sample_id)
            for row in episode.query_adapt + episode.query_guard
        ]
        if len(set(support_ids)) != len(support_ids):
            raise ValueError("Meta episode support contains duplicate physical_sample_id")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Meta episode query contains duplicate physical_sample_id")
        if set(support_ids).intersection(query_ids):
            raise ValueError("Meta episode support/query physical_sample_id overlap")

        support_classes = frozenset(int(row.tx_i) for row in episode.support)
        adapt_query_classes = frozenset(int(row.tx_i) for row in episode.query_adapt)
        guard_query_classes = frozenset(int(row.tx_i) for row in episode.query_guard)
        if support_classes != episode.adapt_class_ids:
            raise ValueError("Meta episode support classes do not match adapt_class_ids")
        if not adapt_query_classes.issubset(episode.adapt_class_ids):
            raise ValueError("query_adapt contains a class outside adapt_class_ids")
        if guard_query_classes != episode.guard_class_ids:
            raise ValueError("query_guard classes do not match guard_class_ids")
        if episode.adapt_class_ids.intersection(episode.guard_class_ids):
            raise ValueError("adapt_class_ids and guard_class_ids must be disjoint")


def _partition_ids(rows: Sequence[MetaSampleRef], *, name: str) -> set[str]:
    values = [str(row.physical_sample_id) for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"episode {name} physical IDs must be nonempty and unique")
    return set(values)


def _single_value(rows: Sequence[MetaSampleRef], field_name: str) -> Any:
    values = {getattr(row, field_name) for row in rows}
    if len(values) != 1:
        raise ValueError(
            f"declared episode kind relation requires one {field_name} per partition"
        )
    return next(iter(values))


def _class_counts(rows: Sequence[MetaSampleRef]) -> Counter[int]:
    return Counter(int(row.tx_i) for row in rows)


def validate_episode_semantics(
    episode: MetaEpisode,
    *,
    source_receiver_ids: Sequence[int] | None = None,
    allowed_roles: Sequence[str] = ("L_s",),
) -> Mapping[str, Any]:
    """Fail closed on forged episode metadata before a bank meta step."""

    if not isinstance(episode, MetaEpisode):
        raise TypeError("episode must be a MetaEpisode")
    kind = _episode_kind(episode.kind)
    if (
        isinstance(episode.k_shot, bool)
        or not isinstance(episode.k_shot, int)
        or episode.k_shot not in MARC_OT_CANONICAL_K
    ):
        raise ValueError("episode k_shot must be one of 1/2/5/10/20")
    if (
        isinstance(episode.query_per_class, bool)
        or not isinstance(episode.query_per_class, int)
        or episode.query_per_class <= 0
    ):
        raise ValueError("episode query_per_class must be a positive integer")
    if not episode.support or not episode.query_adapt or not episode.query_guard:
        raise ValueError("bank episode requires support, query_adapt and query_guard rows")

    partitions = {
        "support": episode.support,
        "query_adapt": episode.query_adapt,
        "query_guard": episode.query_guard,
    }
    ids = {
        name: _partition_ids(rows, name=name) for name, rows in partitions.items()
    }
    for left, right in (
        ("support", "query_adapt"),
        ("support", "query_guard"),
        ("query_adapt", "query_guard"),
    ):
        if ids[left].intersection(ids[right]):
            raise ValueError("episode physical partitions must remain disjoint without overlap")

    all_rows = episode.support + episode.query_adapt + episode.query_guard
    roles = {str(row.role) for row in all_rows}
    allowed = {str(role) for role in allowed_roles}
    if not allowed or not roles.issubset(allowed):
        raise ValueError("episode contains a role outside the source allowlist")
    if source_receiver_ids is not None:
        receiver_allowlist = {int(value) for value in source_receiver_ids}
        if not receiver_allowlist or any(
            int(row.rx_i) not in receiver_allowlist for row in all_rows
        ):
            raise ValueError("episode receiver is outside the source allowlist")

    support_counts = _class_counts(episode.support)
    adapt_counts = _class_counts(episode.query_adapt)
    guard_counts = _class_counts(episode.query_guard)
    if set(support_counts) != set(episode.adapt_class_ids) or any(
        count != episode.k_shot for count in support_counts.values()
    ):
        raise ValueError("episode support per-class K does not match k_shot metadata")
    if set(adapt_counts) != set(episode.adapt_class_ids) or any(
        count != episode.query_per_class for count in adapt_counts.values()
    ):
        raise ValueError("episode query_adapt per-class K does not match role metadata")
    if set(guard_counts) != set(episode.guard_class_ids) or any(
        count != episode.query_per_class for count in guard_counts.values()
    ):
        raise ValueError("episode query_guard per-class K does not match role metadata")
    if set(episode.adapt_class_ids).intersection(episode.guard_class_ids):
        raise ValueError("episode adapt and guard class roles must be disjoint")

    support = episode.support
    query = episode.query_adapt + episode.query_guard
    support_rx = _single_value(support, "rx_i")
    query_rx = _single_value(query, "rx_i")
    support_day = _single_value(support, "day_i")
    query_day = _single_value(query, "day_i")
    support_eq = _single_value(support, "eq_i")
    query_eq = _single_value(query, "eq_i")
    support_blocks = {int(row.capture_block_i) for row in support}
    if not support_blocks:
        raise ValueError("episode support requires a nonempty capture_block_i set")
    query_block = _single_value(query, "capture_block_i")
    support_view = str(_single_value(support, "view"))
    query_view = str(_single_value(query, "view"))

    relation_valid = False
    if kind is EpisodeKind.SAME_DOMAIN:
        relation_valid = (
            support_rx == query_rx
            and support_day == query_day
            and support_eq == query_eq
            and support_view == query_view
        )
    elif kind is EpisodeKind.RX_HOLDOUT:
        relation_valid = (
            support_rx == query_rx
            and support_day == query_day
            and support_eq == query_eq
            and support_view == query_view
        )
    elif kind is EpisodeKind.DAY_CHANNEL_HOLDOUT:
        relation_valid = (
            support_rx == query_rx
            and support_eq == query_eq
            and support_view == query_view
            and support_day != query_day
        )
    elif kind is EpisodeKind.CLEAN_TO_LEO:
        relation_valid = (
            support_rx == query_rx
            and support_day == query_day
            and support_eq == query_eq
            and support_view == "clean"
            and query_view in MARC_OT_LEO_WEAK_SCENES
        )
    elif kind is EpisodeKind.LEO_CROSS:
        relation_valid = (
            support_rx == query_rx
            and support_day == query_day
            and support_eq == query_eq
            and support_view in MARC_OT_LEO_WEAK_SCENES
            and query_view in MARC_OT_LEO_WEAK_SCENES
            and support_view != query_view
        )
    if not relation_valid:
        raise ValueError(f"declared episode kind {kind.value} does not match row relation")
    return {
        "kind": kind.value,
        "k_shot": int(episode.k_shot),
        "query_per_class": int(episode.query_per_class),
        "support_view": support_view,
        "query_view": query_view,
        "support_capture_block_count": len(support_blocks),
        "query_capture_block": int(query_block),
        "pseudo_target_receiver": int(query_rx),
        "receiver_knowledge_holdout": kind is EpisodeKind.RX_HOLDOUT,
        "source_only": True,
    }


def _semantic_key_from_audit(audit: Mapping[str, Any]) -> tuple[str, int, str, str]:
    kind = str(audit["kind"])
    k_shot = int(audit["k_shot"])
    if kind in {
        EpisodeKind.RX_HOLDOUT.value,
        EpisodeKind.DAY_CHANNEL_HOLDOUT.value,
    }:
        return (kind, k_shot, "*", "*")
    return (
        kind,
        k_shot,
        str(audit["support_view"]),
        str(audit["query_view"]),
    )


def marc_ot_episode_semantic_key(
    episode: MetaEpisode,
    *,
    source_receiver_ids: Sequence[int],
) -> tuple[str, int, str, str]:
    """Return the canonical coverage cell after complete semantic validation."""

    return _semantic_key_from_audit(
        validate_episode_semantics(
            episode,
            source_receiver_ids=source_receiver_ids,
        )
    )


def _required_marc_ot_coverage_counter() -> Counter[tuple[str, int, str, str]]:
    cells: list[tuple[str, int, str, str]] = []
    for k_shot in MARC_OT_CANONICAL_K:
        cells.extend(
            (
                (EpisodeKind.RX_HOLDOUT.value, k_shot, "*", "*"),
                (EpisodeKind.DAY_CHANNEL_HOLDOUT.value, k_shot, "*", "*"),
            )
        )
        cells.extend(
            (EpisodeKind.CLEAN_TO_LEO.value, k_shot, "clean", scene)
            for scene in MARC_OT_LEO_WEAK_SCENES
        )
        cells.extend(
            (
                EpisodeKind.LEO_CROSS.value,
                k_shot,
                support_scene,
                query_scene,
            )
            for support_scene in MARC_OT_LEO_WEAK_SCENES
            for query_scene in MARC_OT_LEO_WEAK_SCENES
            if support_scene != query_scene
        )
    return Counter(cells)


def sample_marc_ot_coverage_schedule(
    sampler: HierarchicalMetaEpisodeSampler,
    *,
    seed: int = 0,
) -> tuple[MetaEpisode, ...]:
    """Materialize every frozen MARC-OT software coverage cell deterministically."""

    if not isinstance(sampler, HierarchicalMetaEpisodeSampler):
        raise TypeError("sampler must be a HierarchicalMetaEpisodeSampler")
    episodes: list[MetaEpisode] = []
    offset = 0

    def add(
        kind: EpisodeKind,
        k_shot: int,
        support_view: str | None = None,
        query_view: str | None = None,
    ) -> None:
        nonlocal offset
        episodes.append(
            sampler.sample_requested(
                kind=kind,
                k_shot=k_shot,
                seed=int(seed) + offset,
                support_view=support_view,
                query_view=query_view,
            )
        )
        offset += 1

    for k_shot in MARC_OT_CANONICAL_K:
        add(EpisodeKind.RX_HOLDOUT, k_shot)
        add(EpisodeKind.DAY_CHANNEL_HOLDOUT, k_shot)
        for scene in MARC_OT_LEO_WEAK_SCENES:
            add(EpisodeKind.CLEAN_TO_LEO, k_shot, "clean", scene)
        for support_scene in MARC_OT_LEO_WEAK_SCENES:
            for query_scene in MARC_OT_LEO_WEAK_SCENES:
                if support_scene != query_scene:
                    add(
                        EpisodeKind.LEO_CROSS,
                        k_shot,
                        support_scene,
                        query_scene,
                    )
    return tuple(episodes)


def audit_marc_ot_episode_coverage(
    episodes: Sequence[MetaEpisode],
    *,
    source_receiver_ids: Sequence[int],
    require_complete: bool,
) -> Mapping[str, Any]:
    """Audit scheduled cells separately from episodes actually trained."""

    rows = tuple(episodes)
    if not rows:
        raise ValueError("MARC-OT episode coverage is empty")
    audits = tuple(
        validate_episode_semantics(
            episode,
            source_receiver_ids=source_receiver_ids,
        )
        for episode in rows
    )
    k_values = tuple(sorted({int(row["k_shot"]) for row in audits}))
    receiver_k = tuple(
        sorted(
            {
                int(row["k_shot"])
                for row in audits
                if row["kind"] == EpisodeKind.RX_HOLDOUT.value
            }
        )
    )
    day_k = tuple(
        sorted(
            {
                int(row["k_shot"])
                for row in audits
                if row["kind"] == EpisodeKind.DAY_CHANNEL_HOLDOUT.value
            }
        )
    )
    clean_cells = {
        (int(row["k_shot"]), str(row["query_view"]))
        for row in audits
        if row["kind"] == EpisodeKind.CLEAN_TO_LEO.value
    }
    cross_cells = {
        (
            int(row["k_shot"]),
            str(row["support_view"]),
            str(row["query_view"]),
        )
        for row in audits
        if row["kind"] == EpisodeKind.LEO_CROSS.value
    }
    required_clean = {
        (k_shot, scene)
        for k_shot in MARC_OT_CANONICAL_K
        for scene in MARC_OT_LEO_WEAK_SCENES
    }
    required_cross = {
        (k_shot, support_scene, query_scene)
        for k_shot in MARC_OT_CANONICAL_K
        for support_scene in MARC_OT_LEO_WEAK_SCENES
        for query_scene in MARC_OT_LEO_WEAK_SCENES
        if support_scene != query_scene
    }
    observed_counter = Counter(_semantic_key_from_audit(row) for row in audits)
    required_counter = _required_marc_ot_coverage_counter()
    if require_complete and (
        len(rows) != 55
        or observed_counter != required_counter
        or k_values != MARC_OT_CANONICAL_K
        or receiver_k != MARC_OT_CANONICAL_K
        or day_k != MARC_OT_CANONICAL_K
        or clean_cells != required_clean
        or cross_cells != required_cross
    ):
        raise ValueError(
            "MARC-OT software coverage is incomplete for canonical K/domain/scene cells"
        )
    return {
        "episode_count": len(rows),
        "semantic_cell_count": len(observed_counter),
        "software_supported_k": k_values,
        "receiver_holdout_k": receiver_k,
        "day_capture_holdout_k": day_k,
        "clean_to_leo_scenes": tuple(
            sorted({scene for _k, scene in clean_cells})
        ),
        "leo_cross_scene_pairs": tuple(
            sorted({(support, query) for _k, support, query in cross_cells})
        ),
        "coverage_complete": bool(require_complete),
    }


@dataclass(frozen=True)
class DomainEpisode:
    meta_train_indices: List[int]
    meta_val_indices: List[int]
    meta_train_domains: List[int]
    meta_val_domain: int


def sample_rxday_episode(
    domain_ids: Sequence[int],
    *,
    seed: int = 0,
    meta_train_domain_count: int = 2,
    max_samples_per_domain: int = 0,
) -> DomainEpisode:
    """Sample a source-domain episode for first-order MLDG style checks."""
    by_domain: Dict[int, List[int]] = {}
    for idx, did in enumerate(domain_ids):
        by_domain.setdefault(int(did), []).append(int(idx))
    domains = sorted(by_domain)
    if len(domains) < 2:
        raise ValueError("Meta-SSL MLDG episodes require at least two source domains.")

    rng = np.random.default_rng(int(seed))
    shuffled = [domains[int(i)] for i in rng.permutation(len(domains)).tolist()]
    val_domain = int(shuffled[0])
    train_domains = [int(d) for d in shuffled[1:1 + max(1, int(meta_train_domain_count))]]
    if not train_domains:
        train_domains = [int(shuffled[1])]

    def pick(domain: int) -> List[int]:
        values = list(by_domain[int(domain)])
        if int(max_samples_per_domain) > 0 and len(values) > int(max_samples_per_domain):
            take = rng.permutation(len(values))[: int(max_samples_per_domain)].tolist()
            values = sorted(values[int(i)] for i in take)
        return values

    meta_train: List[int] = []
    for domain in train_domains:
        meta_train.extend(pick(domain))
    meta_val = pick(val_domain)
    if not meta_train or not meta_val:
        raise ValueError("Sampled Meta-SSL episode is empty.")
    return DomainEpisode(
        meta_train_indices=sorted(meta_train),
        meta_val_indices=sorted(meta_val),
        meta_train_domains=sorted(train_domains),
        meta_val_domain=val_domain,
    )
