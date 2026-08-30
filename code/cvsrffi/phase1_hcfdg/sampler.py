"""Source-only fold selection and rectangular HCF-DG episodes.

The sampler deliberately works on source metadata only.  In particular,
``q_phys`` is accepted by the fold selector as the source-side physical
environment evidence; no target or Phase2 metadata is consulted here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


_BATCH_TX_COUNT = 6
_DOMAIN_COUNT = 4
_SAMPLES_PER_CELL = 4
_EPOCH_STRIDE = 1_000_003
_EPISODE_TYPES = ("receiver", "day", "channel")
_EPISODE_PROBABILITIES = (0.65, 0.225, 0.125)


def _integer_array(name: str, values: Any) -> np.ndarray:
    """Return a non-empty one-dimensional integer metadata array."""

    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    try:
        converted = array.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integer IDs") from exc
    if not np.all(array == converted):
        raise ValueError(f"{name} must contain integer IDs")
    return np.asarray(converted, dtype=np.int64)


def _q_phys_array(q_phys: Any) -> np.ndarray:
    """Normalize physical metadata to a finite ``(N, Q)`` float array."""

    try:
        array = np.asarray(q_phys, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("q_phys must be a finite numeric array") from exc
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("q_phys must have shape (N, Q) with Q > 0")
    if not np.isfinite(array).all():
        raise ValueError("q_phys must contain only finite values")
    return array


def select_center_and_far_receivers(
    q_phys: np.ndarray,
    receiver_ids: np.ndarray,
) -> tuple[int, int]:
    """Select source-style center and far receivers from physical metadata.

    The physical features are standardized with the supplied source pool.
    The center is the receiver whose centroid has the smallest mean distance
    to the other receiver centroids.  The far receiver is the one farthest
    from the source-wide centroid.  Receiver IDs provide deterministic
    ascending tie breaks for both choices.
    """

    physical = _q_phys_array(q_phys)
    receivers = _integer_array("receiver_ids", receiver_ids)
    if physical.shape[0] != receivers.shape[0]:
        raise ValueError("q_phys and receiver_ids must have the same length")

    unique_receivers = np.unique(receivers)
    if unique_receivers.size < 2:
        raise ValueError("at least two receivers are required")

    mean = physical.mean(axis=0)
    scale = physical.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (physical - mean) / scale

    centroids = np.vstack(
        [standardized[receivers == receiver].mean(axis=0) for receiver in unique_receivers]
    )
    pairwise_distance = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2
    )
    np.fill_diagonal(pairwise_distance, 0.0)
    mean_other_distance = pairwise_distance.sum(axis=1) / float(unique_receivers.size - 1)
    center_position = min(
        range(unique_receivers.size),
        key=lambda position: (float(mean_other_distance[position]), int(unique_receivers[position])),
    )

    source_centroid = standardized.mean(axis=0)
    distance_to_source = np.linalg.norm(centroids - source_centroid[None, :], axis=1)
    far_position = min(
        range(unique_receivers.size),
        key=lambda position: (-float(distance_to_source[position]), int(unique_receivers[position])),
    )

    return int(unique_receivers[center_position]), int(unique_receivers[far_position])


@dataclass(frozen=True, eq=False)
class EpisodeDescriptor:
    """A fixed rectangular episode and its source-domain masks."""

    indices: tuple[int, ...]
    tx_ids: tuple[int, ...]
    receiver_ids: tuple[int, ...]
    day_ids: tuple[int, ...]
    channel_ids: tuple[int, ...]
    domain_ids: tuple[int, ...]
    episode_type: str
    query_domain: int
    support_mask: np.ndarray
    query_mask: np.ndarray
    valid_tx_mask: np.ndarray
    episode_seed: int

    def __eq__(self, other: object) -> bool:
        """Compare descriptors by value, including NumPy mask contents."""

        if not isinstance(other, EpisodeDescriptor):
            return NotImplemented
        return (
            self.indices == other.indices
            and self.tx_ids == other.tx_ids
            and self.receiver_ids == other.receiver_ids
            and self.day_ids == other.day_ids
            and self.channel_ids == other.channel_ids
            and self.domain_ids == other.domain_ids
            and self.episode_type == other.episode_type
            and self.query_domain == other.query_domain
            and self.episode_seed == other.episode_seed
            and np.array_equal(self.support_mask, other.support_mask)
            and np.array_equal(self.query_mask, other.query_mask)
            and np.array_equal(self.valid_tx_mask, other.valid_tx_mask)
        )


def _metadata_value(metadata: Any, key: str) -> Any:
    if isinstance(metadata, Mapping):
        try:
            return metadata[key]
        except KeyError as exc:
            raise ValueError(f"metadata is missing {key}") from exc
    try:
        return metadata[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        try:
            return getattr(metadata, key)
        except AttributeError as exc:
            raise ValueError(f"metadata is missing {key}") from exc


class HCFDGEpisodeBatchSampler:
    """Yield source-only ``6 TX x 4 domain x 4 sample`` episodes.

    ``metadata`` must provide one-dimensional ``tx_ids``, ``receiver_ids``,
    ``day_ids`` and ``channel_ids`` arrays plus finite ``q_phys``.  A
    ``domain_ids`` array is optional; when absent, each unique
    ``(receiver, day, channel)`` triple receives a stable integer ID.

    The iterator is unbounded by default so callers can request exactly the
    number of optimizer updates they need.  ``episodes_per_epoch`` can be
    supplied for finite data-loader integration.
    """

    def __init__(
        self,
        metadata: Mapping[str, Any] | Any,
        *,
        seed: int = 0,
        episodes_per_epoch: int | None = None,
    ) -> None:
        try:
            self.seed = int(seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("seed must be a non-negative integer") from exc
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if episodes_per_epoch is not None:
            try:
                episodes_per_epoch = int(episodes_per_epoch)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("episodes_per_epoch must be positive") from exc
            if episodes_per_epoch <= 0:
                raise ValueError("episodes_per_epoch must be positive")

        tx_ids = _integer_array("tx_ids", _metadata_value(metadata, "tx_ids"))
        receiver_ids = _integer_array("receiver_ids", _metadata_value(metadata, "receiver_ids"))
        day_ids = _integer_array("day_ids", _metadata_value(metadata, "day_ids"))
        channel_ids = _integer_array("channel_ids", _metadata_value(metadata, "channel_ids"))
        q_phys = _q_phys_array(_metadata_value(metadata, "q_phys"))

        lengths = {len(tx_ids), len(receiver_ids), len(day_ids), len(channel_ids), len(q_phys)}
        if len(lengths) != 1:
            raise ValueError("metadata arrays must have the same length")

        optional_domain_ids: np.ndarray | None = None
        try:
            raw_domain_ids = _metadata_value(metadata, "domain_ids")
        except ValueError:
            raw_domain_ids = None
        if raw_domain_ids is not None:
            optional_domain_ids = _integer_array("domain_ids", raw_domain_ids)
            if len(optional_domain_ids) != len(tx_ids):
                raise ValueError("metadata arrays must have the same length")

        self._tx_ids = tx_ids
        self._receiver_ids = receiver_ids
        self._day_ids = day_ids
        self._channel_ids = channel_ids
        self._q_phys = q_phys
        self.episodes_per_epoch = episodes_per_epoch
        self._epoch = 0

        self.center_receiver, self.far_receiver = select_center_and_far_receivers(
            self._q_phys, self._receiver_ids
        )

        if np.unique(self._tx_ids).size < _BATCH_TX_COUNT:
            raise ValueError("metadata must contain at least six TX")

        if optional_domain_ids is None:
            triples = list(
                zip(
                    self._receiver_ids.tolist(),
                    self._day_ids.tolist(),
                    self._channel_ids.tolist(),
                )
            )
            unique_triples = sorted(set(triples))
            triple_to_domain = {triple: position for position, triple in enumerate(unique_triples)}
            domain_ids = np.asarray(
                [triple_to_domain[triple] for triple in triples], dtype=np.int64
            )
            self._domain_metadata = {
                int(domain_id): tuple(int(value) for value in triple)
                for domain_id, triple in enumerate(unique_triples)
            }
        else:
            domain_ids = optional_domain_ids
            self._domain_metadata = {}
            for domain_id in np.unique(domain_ids).tolist():
                positions = np.flatnonzero(domain_ids == domain_id)
                first = int(positions[0])
                triple = (
                    int(self._receiver_ids[first]),
                    int(self._day_ids[first]),
                    int(self._channel_ids[first]),
                )
                triples = list(
                    zip(
                        self._receiver_ids[positions].tolist(),
                        self._day_ids[positions].tolist(),
                        self._channel_ids[positions].tolist(),
                    )
                )
                if any(tuple(int(value) for value in row) != triple for row in triples):
                    raise ValueError("each domain_id must identify one receiver/day/channel triple")
                self._domain_metadata[int(domain_id)] = triple

        self._domain_ids_by_row = domain_ids
        self._tx_values = tuple(int(value) for value in np.unique(self._tx_ids).tolist())
        self._domain_values = tuple(int(value) for value in np.unique(domain_ids).tolist())
        if len(self._domain_values) < _DOMAIN_COUNT:
            raise ValueError("metadata must contain at least four domains")
        if len({self._domain_metadata[domain_id][0] for domain_id in self._domain_values}) < 3:
            raise ValueError("metadata domains must cover at least three receivers")

        self._cells: dict[tuple[int, int], tuple[int, ...]] = {}
        for tx_id in self._tx_values:
            for domain_id in self._domain_values:
                positions = np.flatnonzero(
                    (self._tx_ids == tx_id) & (self._domain_ids_by_row == domain_id)
                )
                self._cells[(tx_id, domain_id)] = tuple(int(position) for position in positions.tolist())

    def __iter__(self) -> Iterator[EpisodeDescriptor]:
        generator = np.random.default_rng(self._epoch_seed())
        emitted = 0
        while self.episodes_per_epoch is None or emitted < self.episodes_per_epoch:
            yield self._sample_episode(generator)
            emitted += 1

    def __len__(self) -> int:
        if self.episodes_per_epoch is None:
            raise TypeError("an unbounded HCF-DG sampler has no finite length")
        return self.episodes_per_epoch

    def set_epoch(self, epoch: int) -> None:
        try:
            resolved = int(epoch)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("epoch must be a non-negative integer") from exc
        self._epoch = max(0, resolved)

    def _epoch_seed(self) -> int:
        return self.seed + self._epoch * _EPOCH_STRIDE

    def _select_tx_for_domains(self, domains: tuple[int, ...]) -> tuple[int, ...] | None:
        """Choose six TXs, retaining a source TX that covers each domain.

        A sparse domain may have only one source TX.  Keeping that TX in the
        rectangle lets the domain remain observable without filling another
        TX's cell with borrowed samples.  The remaining positions are later
        represented by same-TX placeholders and masked invalid.
        """

        def cell_score(tx_id: int) -> tuple[int, int, int]:
            cells = [self._cells[(tx_id, domain_id)] for domain_id in domains]
            complete = sum(len(cell) >= _SAMPLES_PER_CELL for cell in cells)
            nonempty = sum(bool(cell) for cell in cells)
            usable = sum(min(len(cell), _SAMPLES_PER_CELL) for cell in cells)
            return int(complete), int(nonempty), int(usable)

        selected: list[int] = []
        uncovered = set(domains)
        while uncovered:
            candidates = [
                tx_id
                for tx_id in self._tx_values
                if tx_id not in selected
                and any(self._cells[(tx_id, domain_id)] for domain_id in uncovered)
            ]
            if not candidates:
                return None
            selected_tx = max(
                candidates,
                key=lambda tx_id: (
                    sum(bool(self._cells[(tx_id, domain_id)]) for domain_id in uncovered),
                    *cell_score(tx_id),
                    -tx_id,
                ),
            )
            selected.append(selected_tx)
            uncovered = {
                domain_id
                for domain_id in uncovered
                if not self._cells[(selected_tx, domain_id)]
            }
            if len(selected) > _BATCH_TX_COUNT:
                return None

        remaining = [tx_id for tx_id in self._tx_values if tx_id not in selected]
        remaining.sort(key=lambda tx_id: (*cell_score(tx_id), -tx_id), reverse=True)
        selected.extend(remaining[: _BATCH_TX_COUNT - len(selected)])
        if len(selected) != _BATCH_TX_COUNT:
            return None
        return tuple(sorted(selected))

    def _valid_domain_combinations(
        self,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int, int]]]:
        """Return four-domain rectangles with a six-TX source coverage plan."""

        candidates: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int, int]]
        ] = []
        for combination in combinations(self._domain_values, _DOMAIN_COUNT):
            receiver_coverage = len(
                {self._domain_metadata[domain_id][0] for domain_id in combination}
            )
            if receiver_coverage < 3:
                continue
            selected_tx = self._select_tx_for_domains(combination)
            if selected_tx is None:
                continue
            cells = [
                self._cells[(tx_id, domain_id)]
                for tx_id in selected_tx
                for domain_id in combination
            ]
            complete_count = sum(len(cell) >= _SAMPLES_PER_CELL for cell in cells)
            nonempty_count = sum(bool(cell) for cell in cells)
            usable_count = sum(min(len(cell), _SAMPLES_PER_CELL) for cell in cells)
            score = (
                int(complete_count),
                int(nonempty_count),
                int(usable_count),
                int(receiver_coverage),
            )
            candidates.append((combination, selected_tx, score))

        if not candidates:
            raise ValueError(
                "metadata must contain a six-TX rectangular pool across four domains"
            )
        return candidates

    def _choose_rectangle(
        self, generator: np.random.Generator
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        candidates = self._valid_domain_combinations()
        best_score = max(candidate[2] for candidate in candidates)
        candidates = [candidate for candidate in candidates if candidate[2] == best_score]
        candidate_position = int(generator.integers(0, len(candidates)))
        domains, selected_tx, _ = candidates[candidate_position]
        return domains, selected_tx

    def _choose_query_domain(
        self,
        domains: tuple[int, ...],
        episode_type: str,
        generator: np.random.Generator,
    ) -> int:
        factor_position = {"receiver": 0, "day": 1, "channel": 2}[episode_type]
        factor_counts: dict[int, int] = {}
        for domain_id in domains:
            factor = self._domain_metadata[domain_id][factor_position]
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        queryable_domains = tuple(
            domain_id
            for domain_id in domains
            if any(
                self._cells[(tx_id, domain_id)]
                and sum(
                    bool(self._cells[(tx_id, other_domain)])
                    for other_domain in domains
                    if other_domain != domain_id
                )
                >= 2
                for tx_id in self._tx_values
            )
        )
        queryable_domains = queryable_domains or tuple(
            domain_id
            for domain_id in domains
            if any(self._cells[(tx_id, domain_id)] for tx_id in self._tx_values)
        )
        unique_factor_domains = tuple(
            domain_id
            for domain_id in queryable_domains
            if factor_counts[self._domain_metadata[domain_id][factor_position]] == 1
        )
        pool = unique_factor_domains or queryable_domains
        if not pool:
            raise ValueError("selected rectangle has no queryable domain")
        return int(pool[int(generator.integers(0, len(pool)))])

    def _sample_cell(
        self,
        tx_id: int,
        domain_id: int,
        generator: np.random.Generator,
    ) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        pool = self._cells[(tx_id, domain_id)]
        if not pool:
            same_tx = tuple(
                index
                for other_domain in self._domain_values
                for index in self._cells[(tx_id, other_domain)]
            )
            if not same_tx:
                raise ValueError("selected rectangle contains a TX without source samples")
            placeholder = int(same_tx[int(generator.integers(0, len(same_tx)))])
            return (placeholder,) * _SAMPLES_PER_CELL, (False,) * _SAMPLES_PER_CELL
        permutation = generator.permutation(len(pool))
        available = tuple(pool[int(position)] for position in permutation.tolist())
        if len(available) >= _SAMPLES_PER_CELL:
            return available[:_SAMPLES_PER_CELL], (True,) * _SAMPLES_PER_CELL

        pad_count = _SAMPLES_PER_CELL - len(available)
        padding = tuple(
            int(available[int(position)])
            for position in generator.integers(0, len(available), size=pad_count).tolist()
        )
        return (
            available + padding,
            (True,) * len(available) + (False,) * pad_count,
        )

    def _sample_episode(self, generator: np.random.Generator) -> EpisodeDescriptor:
        episode_seed = int(generator.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        episode_generator = np.random.default_rng(episode_seed)
        episode_type = str(
            episode_generator.choice(_EPISODE_TYPES, p=_EPISODE_PROBABILITIES)
        )
        domains, selected_tx = self._choose_rectangle(episode_generator)
        query_domain = self._choose_query_domain(domains, episode_type, episode_generator)

        indices: list[int] = []
        tx_ids: list[int] = []
        receiver_ids: list[int] = []
        day_ids: list[int] = []
        channel_ids: list[int] = []
        domain_ids: list[int] = []
        sample_valid: list[bool] = []

        for tx_id in selected_tx:
            for domain_id in domains:
                cell_indices, cell_valid = self._sample_cell(tx_id, domain_id, episode_generator)
                receiver_id, day_id, channel_id = self._domain_metadata[domain_id]
                for index, is_valid in zip(cell_indices, cell_valid):
                    indices.append(int(index))
                    tx_ids.append(int(tx_id))
                    receiver_ids.append(int(receiver_id))
                    day_ids.append(int(day_id))
                    channel_ids.append(int(channel_id))
                    domain_ids.append(int(domain_id))
                    sample_valid.append(bool(is_valid))

        domain_array = np.asarray(domain_ids, dtype=np.int64)
        valid_array = np.asarray(sample_valid, dtype=bool)
        eligible_tx = {
            tx_id: sum(bool(self._cells[(tx_id, domain_id)]) for domain_id in domains if domain_id != query_domain)
            >= 2
            for tx_id in selected_tx
        }
        valid_tx_mask = valid_array & np.asarray(
            [eligible_tx[int(tx_id)] for tx_id in tx_ids], dtype=bool
        )
        query_mask = valid_tx_mask & (domain_array == query_domain)
        support_mask = valid_tx_mask & (domain_array != query_domain)

        return EpisodeDescriptor(
            indices=tuple(indices),
            tx_ids=tuple(tx_ids),
            receiver_ids=tuple(receiver_ids),
            day_ids=tuple(day_ids),
            channel_ids=tuple(channel_ids),
            domain_ids=tuple(domain_ids),
            episode_type=episode_type,
            query_domain=int(query_domain),
            support_mask=support_mask,
            query_mask=query_mask,
            valid_tx_mask=valid_tx_mask,
            episode_seed=episode_seed,
        )


__all__ = [
    "EpisodeDescriptor",
    "HCFDGEpisodeBatchSampler",
    "select_center_and_far_receivers",
]
