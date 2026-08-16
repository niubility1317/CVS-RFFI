"""Behavioral checks for balanced, role-safe Phase1 source proxy episodes."""

from __future__ import annotations

import hashlib
import importlib

import pytest
import torch


def _proxy_api():
    """Import the Task 3 API inside tests so RED proves the missing boundary."""

    try:
        module = importlib.import_module("cvsrffi.phase1_mirage.proxy")
    except ModuleNotFoundError as error:
        if error.name == "cvsrffi.phase1_mirage.proxy":
            pytest.fail("missing role-balanced Phase1 proxy episode boundary")
        raise
    return module.ProxyProtocolError, module.build_proxy_episode, module.proxy_class_for_episode


def test_train_proxy_accepts_only_labeled_training_role():
    ProxyProtocolError, build_proxy_episode, _ = _proxy_api()
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    with pytest.raises(ProxyProtocolError, match="train_l"):
        build_proxy_episode(labels, split_role="train_u", seed=9, episode_index=0)


@pytest.mark.parametrize("split_role", ("train_l", "val_cal", "val_select"))
def test_only_approved_labeled_split_roles_build_proxy_episodes(split_role: str):
    _, build_proxy_episode, _ = _proxy_api()

    episode = build_proxy_episode(
        torch.tensor([0, 0, 1, 1, 2, 2]),
        split_role=split_role,
        seed=9,
        episode_index=0,
    )

    assert episode.schedule_receipt["split_role"] == split_role


@pytest.mark.parametrize("split_role", ("unknown", "target_known", "target_unknown", ""))
def test_unrecognized_or_target_roles_fail_closed(split_role: str):
    ProxyProtocolError, build_proxy_episode, _ = _proxy_api()

    with pytest.raises(ProxyProtocolError, match="split_role"):
        build_proxy_episode(
            torch.tensor([0, 0, 1, 1, 2, 2]),
            split_role=split_role,
            seed=9,
            episode_index=0,
        )


def test_proxy_class_is_absent_from_registered_mask_and_rows():
    _, build_proxy_episode, _ = _proxy_api()
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    episode = build_proxy_episode(labels, split_role="train_l", seed=9, episode_index=0)

    assert not episode.registered_class_mask[episode.proxy_class].item()
    assert set(labels[episode.proxy_rows].tolist()) == {episode.proxy_class}
    assert episode.proxy_rows.numel() + episode.registered_rows.numel() == labels.numel()
    assert episode.proxy_class not in set(labels[episode.registered_rows].tolist())


def test_proxy_schedule_is_seed_deterministic_and_balanced_over_one_cycle():
    _, _, proxy_class_for_episode = _proxy_api()
    class_ids = (0, 1, 2, 3)
    expected_offset = int(hashlib.sha256(b"9:proxy").hexdigest(), 16) % len(class_ids)

    first_cycle = [
        proxy_class_for_episode(class_ids, seed=9, episode_index=episode_index)
        for episode_index in range(len(class_ids))
    ]
    repeated_cycle = [
        proxy_class_for_episode(class_ids, seed=9, episode_index=episode_index)
        for episode_index in range(len(class_ids))
    ]

    assert first_cycle == [class_ids[(expected_offset + episode_index) % len(class_ids)] for episode_index in range(len(class_ids))]
    assert repeated_cycle == first_cycle
    assert sorted(first_cycle) == list(class_ids)


def test_label_permutation_preserves_full_cycle_role_balance_and_receipts():
    _, build_proxy_episode, _ = _proxy_api()
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    relabeled = torch.tensor([2, 2, 0, 0, 1, 1])

    original_cycle = [
        build_proxy_episode(labels, split_role="val_select", seed=9, episode_index=episode_index)
        for episode_index in range(3)
    ]
    relabeled_cycle = [
        build_proxy_episode(relabeled, split_role="val_select", seed=9, episode_index=episode_index)
        for episode_index in range(3)
    ]
    original_proxy_counts = torch.zeros(labels.numel(), dtype=torch.int64)
    relabeled_proxy_counts = torch.zeros(relabeled.numel(), dtype=torch.int64)
    for original, permuted in zip(original_cycle, relabeled_cycle):
        original_proxy_counts[original.proxy_rows] += 1
        relabeled_proxy_counts[permuted.proxy_rows] += 1
        assert dict(permuted.schedule_receipt) == dict(original.schedule_receipt)

    assert torch.equal(original_proxy_counts, torch.ones_like(original_proxy_counts))
    assert torch.equal(relabeled_proxy_counts, torch.ones_like(relabeled_proxy_counts))


def test_schedule_receipt_contains_only_permutation_invariant_metadata():
    _, build_proxy_episode, _ = _proxy_api()

    episode = build_proxy_episode(
        torch.tensor([0, 0, 1, 1, 2, 2]),
        split_role="train_l",
        seed=9,
        episode_index=0,
    )

    assert dict(episode.schedule_receipt) == {
        "class_count": 3,
        "proxy_row_count": 2,
        "registered_row_count": 4,
        "split_role": "train_l",
        "seed": 9,
        "episode_index": 0,
    }
    with pytest.raises(TypeError):
        episode.schedule_receipt["class_count"] = 4


@pytest.mark.parametrize(
    ("labels", "message"),
    (
        (torch.tensor([], dtype=torch.int64), "non-empty"),
        (torch.tensor([0, 0, 1, 1]), "at least three"),
        (torch.tensor([-1, -1, 0, 0, 1, 1]), "non-negative"),
        (torch.tensor([0, 0, 2, 2, 3, 3]), "contiguous"),
        (torch.tensor([[0, 1, 2], [0, 1, 2]]), "one-dimensional"),
    ),
)
def test_invalid_label_batches_fail_closed(labels: torch.Tensor, message: str):
    ProxyProtocolError, build_proxy_episode, _ = _proxy_api()

    with pytest.raises(ProxyProtocolError, match=message):
        build_proxy_episode(labels, split_role="train_l", seed=9, episode_index=0)
