from __future__ import annotations

from collections import Counter
from itertools import islice

import numpy as np
import pytest

from cvsrffi.phase1_hcfdg.sampler import (
    EpisodeDescriptor,
    HCFDGEpisodeBatchSampler,
    select_center_and_far_receivers,
)


def _complete_metadata() -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, int, int, float]] = []
    domains = (
        (1, 1, 1),
        (3, 1, 1),
        (4, 2, 1),
        (8, 2, 2),
    )
    for tx_id in range(10, 16):
        for receiver_id, day_id, channel_id in domains:
            for sample_id in range(4):
                rows.append(
                    (
                        tx_id,
                        receiver_id,
                        day_id,
                        channel_id,
                        float(receiver_id) + sample_id * 0.01,
                    )
                )
    return {
        "tx_ids": np.asarray([row[0] for row in rows], dtype=np.int64),
        "receiver_ids": np.asarray([row[1] for row in rows], dtype=np.int64),
        "day_ids": np.asarray([row[2] for row in rows], dtype=np.int64),
        "channel_ids": np.asarray([row[3] for row in rows], dtype=np.int64),
        "q_phys": np.asarray([[row[4]] for row in rows], dtype=np.float64),
    }


def test_episode_has_six_tx_four_domains_and_four_samples_per_cell():
    metadata = _complete_metadata()
    sampler = HCFDGEpisodeBatchSampler(metadata, seed=392002)

    episode = next(iter(sampler))

    assert isinstance(episode, EpisodeDescriptor)
    assert len(episode.indices) == 96
    assert len(set(episode.tx_ids)) == 6
    assert len(set(episode.domain_ids)) == 4
    assert len(set(episode.receiver_ids)) >= 3
    assert episode.query_mask.sum() > 0
    assert episode.support_mask.sum() > 0
    assert not np.any(episode.support_mask & episode.query_mask)
    assert episode.valid_tx_mask.shape == (96,)
    assert episode.query_domain in set(episode.domain_ids)

    cell_counts = Counter(zip(episode.tx_ids, episode.domain_ids))
    assert set(cell_counts.values()) == {4}


def test_center_and_far_receivers_use_source_qphys_only():
    q_phys = np.asarray([[0.0], [11.0], [10.0], [12.0], [100.0]])
    receiver_ids = np.asarray([1, 3, 4, 6, 8])

    center, far = select_center_and_far_receivers(q_phys, receiver_ids)

    assert (center, far) == (3, 8)


def test_receiver_id_ties_use_numeric_ascending_order():
    q_phys = np.asarray([[-1.0], [1.0], [0.0]])
    receiver_ids = np.asarray([2, 10, 3])

    center, far = select_center_and_far_receivers(q_phys, receiver_ids)

    assert center == 3
    assert far == 2


def test_sampler_replays_the_same_episode_after_resetting_epoch():
    metadata = _complete_metadata()
    sampler = HCFDGEpisodeBatchSampler(metadata, seed=392002)

    sampler.set_epoch(7)
    first = next(iter(sampler))
    sampler.set_epoch(7)
    replay = next(iter(sampler))
    sampler.set_epoch(8)
    next_epoch = next(iter(sampler))

    assert first == replay
    assert first.episode_seed != next_epoch.episode_seed
    assert first.indices != next_epoch.indices or first.episode_type != next_epoch.episode_type


def test_candidate_rectangles_are_constructed_once_and_replay_stays_stable(monkeypatch):
    construction_calls = 0
    original = HCFDGEpisodeBatchSampler._valid_domain_combinations

    def counted_construction(self):
        nonlocal construction_calls
        construction_calls += 1
        return original(self)

    monkeypatch.setattr(
        HCFDGEpisodeBatchSampler,
        "_valid_domain_combinations",
        counted_construction,
    )
    sampler = HCFDGEpisodeBatchSampler(_complete_metadata(), seed=392002)

    sampler.set_epoch(7)
    first = list(islice(iter(sampler), 12))
    sampler.set_epoch(7)
    replay = list(islice(iter(sampler), 12))

    assert first == replay
    assert construction_calls == 1


def test_set_epoch_rejects_negative_epoch():
    sampler = HCFDGEpisodeBatchSampler(_complete_metadata(), seed=392002)

    with pytest.raises(ValueError, match="non-negative integer"):
        sampler.set_epoch(-1)


def test_episode_types_follow_the_frozen_probabilities():
    sampler = HCFDGEpisodeBatchSampler(_complete_metadata(), seed=392002)
    episodes = list(islice(iter(sampler), 4000))
    counts = Counter(episode.episode_type for episode in episodes)

    assert set(counts) == {"receiver", "day", "channel"}
    assert counts["receiver"] / 4000 == pytest.approx(0.65, abs=0.03)
    assert counts["day"] / 4000 == pytest.approx(0.225, abs=0.025)
    assert counts["channel"] / 4000 == pytest.approx(0.125, abs=0.02)


def test_incomplete_cell_is_masked_without_borrowing_another_tx():
    metadata = _complete_metadata()
    incomplete_index = 12
    keep = np.ones(len(metadata["tx_ids"]), dtype=bool)
    keep[incomplete_index + 1 : incomplete_index + 4] = False
    incomplete = {key: values[keep] for key, values in metadata.items()}

    episode = next(iter(HCFDGEpisodeBatchSampler(incomplete, seed=392002)))

    tx10_positions = np.flatnonzero(np.asarray(episode.tx_ids) == 10)
    assert tx10_positions.size == 16
    assert not np.all(episode.valid_tx_mask[tx10_positions])
    assert all(episode.tx_ids[position] == 10 for position in tx10_positions)


def test_sampler_rejects_metadata_without_a_rectangular_source_pool():
    metadata = _complete_metadata()
    metadata = {key: values[:16] for key, values in metadata.items()}

    with pytest.raises(ValueError, match="at least six TX"):
        HCFDGEpisodeBatchSampler(metadata, seed=392002)
