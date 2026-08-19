import copy

import pytest
import torch

from cvsrffi.muse_ssdg import MUSEClassificationPrototypeBank, MUSETemporalMemory


def _assert_temporal_state_equal(left, right):
    assert left["stability_steps"] == right["stability_steps"]
    assert left["frozen"] == right["frozen"]
    assert left["entries"] == right["entries"]


def _assert_prototype_state_equal(left, right):
    assert left["feature_dim"] == right["feature_dim"]
    assert left["unlabeled_weight"] == right["unlabeled_weight"]
    assert left["frozen"] == right["frozen"]
    assert left["counts"] == right["counts"]
    assert left["domain_counts"] == right["domain_counts"]
    assert set(left["prototypes"]) == set(right["prototypes"])
    for class_id in left["prototypes"]:
        assert torch.equal(left["prototypes"][class_id], right["prototypes"][class_id])


def test_temporal_memory_requires_three_consecutive_same_predictions():
    memory = MUSETemporalMemory()
    key = ("rx_0", "day_0", "eq_0", "sig_0", 7)

    assert memory.observe([key], torch.tensor([2]), torch.tensor([0.8]), 1).tolist() == [False]
    assert memory.observe([key], torch.tensor([2]), torch.tensor([0.9]), 2).tolist() == [False]
    assert memory.observe([key], torch.tensor([2]), torch.tensor([0.95]), 3).tolist() == [True]

    assert memory.observe([key], torch.tensor([3]), torch.tensor([0.95]), 4).tolist() == [False]
    assert memory.observe([key], torch.tensor([3]), torch.tensor([0.95]), 5).tolist() == [False]
    assert memory.observe([key], torch.tensor([3]), torch.tensor([0.95]), 6).tolist() == [True]


def test_temporal_memory_rejects_stability_threshold_below_three():
    with pytest.raises(ValueError, match="three"):
        MUSETemporalMemory(stability_steps=2)

    state = MUSETemporalMemory().state_dict()
    state["stability_steps"] = 2
    with pytest.raises(ValueError, match="three"):
        MUSETemporalMemory().load_state_dict(state)


def test_temporal_memory_state_round_trip_and_freeze_are_observable():
    key = ("rx_0", "day_0", "eq_0", "sig_0", 7)
    memory = MUSETemporalMemory()
    for epoch in (1, 2):
        memory.observe([key], torch.tensor([2]), torch.tensor([0.8]), epoch)

    state = memory.state_dict()
    restored = MUSETemporalMemory()
    restored.load_state_dict(copy.deepcopy(state))
    _assert_temporal_state_equal(restored.state_dict(), state)

    memory.freeze()
    frozen_state = copy.deepcopy(memory.state_dict())
    memory.observe(
        [("rx_1", "day_1", "eq_1", "sig_1", 8)],
        torch.tensor([9]),
        torch.tensor([0.99]),
        3,
    )
    _assert_temporal_state_equal(memory.state_dict(), frozen_state)


def test_frozen_temporal_memory_rejects_load_and_round_trip_stays_frozen():
    key = ("rx_0", "day_0", "eq_0", "sig_0", 7)
    memory = MUSETemporalMemory()
    memory.observe([key], torch.tensor([2]), torch.tensor([0.8]), 1)
    memory.freeze()
    frozen_state = copy.deepcopy(memory.state_dict())

    replacement = MUSETemporalMemory()
    replacement.observe(
        [("rx_1", "day_1", "eq_1", "sig_1", 8)],
        torch.tensor([9]),
        torch.tensor([0.99]),
        1,
    )
    with pytest.raises(RuntimeError, match="frozen"):
        memory.load_state_dict(replacement.state_dict())
    _assert_temporal_state_equal(memory.state_dict(), frozen_state)

    restored = MUSETemporalMemory()
    restored.load_state_dict(copy.deepcopy(frozen_state))
    assert restored.state_dict()["frozen"] is True
    restored.observe(
        [("rx_2", "day_2", "eq_2", "sig_2", 9)],
        torch.tensor([10]),
        torch.tensor([0.99]),
        2,
    )
    _assert_temporal_state_equal(restored.state_dict(), frozen_state)


def test_prototype_bank_updates_only_high_and_stable_samples():
    bank = MUSEClassificationPrototypeBank(feature_dim=2)
    bank.observe(
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]),
        pseudo=torch.tensor([4, 4, 5]),
        domains=["d0", "d1", "d0"],
        high_mask=torch.tensor([True, False, True]),
        stable_mask=torch.tensor([True, True, False]),
        unlabeled_weight=0.075,
    )
    state = bank.state_dict()
    assert set(state["prototypes"]) == {4}
    assert torch.allclose(state["prototypes"][4], torch.tensor([1.0, 0.0]))


def test_prototype_bank_validates_unlabeled_weight_at_construction():
    assert MUSEClassificationPrototypeBank(feature_dim=2).unlabeled_weight == 0.075
    for value in (0.049, 0.101):
        with pytest.raises(ValueError, match="unlabeled_weight"):
            MUSEClassificationPrototypeBank(feature_dim=2, unlabeled_weight=value)


def test_prototype_bank_state_round_trip_and_freeze_are_observable():
    bank = MUSEClassificationPrototypeBank(feature_dim=2, unlabeled_weight=0.08)
    bank.observe(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([4]),
        ["d0"],
        torch.tensor([True]),
        torch.tensor([True]),
        0.08,
    )
    state = bank.state_dict()
    restored = MUSEClassificationPrototypeBank(feature_dim=2)
    restored.load_state_dict(copy.deepcopy(state))
    _assert_prototype_state_equal(restored.state_dict(), state)

    bank.freeze()
    frozen_state = copy.deepcopy(bank.state_dict())
    bank.observe(
        torch.tensor([[8.0, 9.0]]),
        torch.tensor([4]),
        ["d0"],
        torch.tensor([True]),
        torch.tensor([True]),
        0.08,
    )
    _assert_prototype_state_equal(bank.state_dict(), frozen_state)


def test_frozen_prototype_bank_rejects_load_and_round_trip_stays_frozen():
    bank = MUSEClassificationPrototypeBank(feature_dim=2)
    bank.observe(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([4]),
        ["d0"],
        torch.tensor([True]),
        torch.tensor([True]),
        0.075,
    )
    bank.freeze()
    frozen_state = copy.deepcopy(bank.state_dict())

    replacement = MUSEClassificationPrototypeBank(feature_dim=2)
    replacement.observe(
        torch.tensor([[8.0, 9.0]]),
        torch.tensor([5]),
        ["d1"],
        torch.tensor([True]),
        torch.tensor([True]),
        0.075,
    )
    with pytest.raises(RuntimeError, match="frozen"):
        bank.load_state_dict(replacement.state_dict())
    _assert_prototype_state_equal(bank.state_dict(), frozen_state)

    restored = MUSEClassificationPrototypeBank(feature_dim=2)
    restored.load_state_dict(copy.deepcopy(frozen_state))
    assert restored.state_dict()["frozen"] is True
    restored.observe(
        torch.tensor([[8.0, 9.0]]),
        torch.tensor([4]),
        ["d0"],
        torch.tensor([True]),
        torch.tensor([True]),
        0.075,
    )
    _assert_prototype_state_equal(restored.state_dict(), frozen_state)
