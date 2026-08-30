from __future__ import annotations

import pytest
import torch

from cvsrffi.phase1_bicad_xr.sampler import (
    BalancedIndexPool,
    StructuredEpisode,
    build_structured_episode,
)


def test_structured_episode_masks_missing_cells_without_duplication():
    tx = [0, 0, 1, 1, 1]
    receiver = [0, 1, 0, 0, 1]
    day = [10, 11, 20, 21, 22]

    episode = build_structured_episode(
        tx,
        receiver,
        day,
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert isinstance(episode, StructuredEpisode)
    assert episode.valid_cells.tolist() == [[False, False], [True, False]]
    assert len(episode.indices) == len(set(episode.indices))
    assert episode.tx.tolist() == [1, 1]
    assert episode.receiver.tolist() == [0, 0]


def test_episode_never_fills_a_missing_tx_rx_cell_from_another_tx():
    episode = build_structured_episode(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [10, 11, 20, 21],
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(1),
    )

    assert episode.valid_cells.tolist() == [[True, False], [False, True]]
    assert set(zip(episode.tx.tolist(), episode.receiver.tolist())) == {(0, 0), (1, 1)}
    assert set(episode.indices) == {0, 1, 2, 3}


def test_structured_episode_uses_only_complete_cells_and_preserves_day_alignment():
    tx = [0, 0, 0, 1, 1, 1]
    receiver = [0, 0, 1, 0, 1, 1]
    day = [100, 101, 102, 200, 201, 202]

    episode = build_structured_episode(
        tx,
        receiver,
        day,
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(3),
    )

    assert episode.valid_cells.tolist() == [[True, False], [False, True]]
    assert episode.day.tolist() == [day[index] for index in episode.indices]
    assert episode.tx.tolist() == [tx[index] for index in episode.indices]
    assert episode.receiver.tolist() == [receiver[index] for index in episode.indices]


def test_same_generator_seed_reproduces_the_same_episode():
    args = ([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [3, 4, 5, 6, 7, 8])

    first = build_structured_episode(
        *args,
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(392001),
    )
    replay = build_structured_episode(
        *args,
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(392001),
    )

    assert first.indices == replay.indices
    assert torch.equal(first.tx, replay.tx)
    assert torch.equal(first.receiver, replay.receiver)
    assert torch.equal(first.day, replay.day)
    assert torch.equal(first.valid_cells, replay.valid_cells)


def test_balanced_index_pool_samples_each_complete_cell_without_repairing_it():
    pool = BalancedIndexPool(
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [10, 11, 20, 21, 22],
        num_classes=2,
        num_receivers=2,
    )

    episode = pool.sample(
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert episode.valid_cells.tolist() == [[False, False], [True, False]]
    assert set(episode.indices) == {2, 3}


def test_sampler_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="same length"):
        build_structured_episode([0], [0, 1], [0], samples_per_cell=1)

    with pytest.raises(ValueError, match="non-empty"):
        build_structured_episode([], [], [], samples_per_cell=1)


@pytest.mark.parametrize(
    ("tx", "receiver", "num_classes", "num_receivers"),
    [
        ([-1, 0], [0, 0], None, None),
        ([0, 2], [0, 0], 2, None),
        ([0, 0], [0, 2], None, 2),
    ],
)
def test_sampler_rejects_out_of_range_labels(tx, receiver, num_classes, num_receivers):
    with pytest.raises(ValueError, match="range"):
        build_structured_episode(
            tx,
            receiver,
            [0] * len(tx),
            samples_per_cell=1,
            num_classes=num_classes,
            num_receivers=num_receivers,
        )


@pytest.mark.parametrize("samples_per_cell", [0, -1, 1.5, True])
def test_sampler_rejects_invalid_samples_per_cell(samples_per_cell):
    with pytest.raises(ValueError, match="samples_per_cell"):
        build_structured_episode([0], [0], [0], samples_per_cell=samples_per_cell)


def test_sampler_rejects_non_integral_metadata_labels():
    with pytest.raises(ValueError, match="integer"):
        build_structured_episode([0.5], [0], [0], samples_per_cell=1)
