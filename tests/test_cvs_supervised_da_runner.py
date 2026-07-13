from __future__ import annotations

import torch

from paper_reproduction.cvs_aligned.supervised_da_runner import (
    _nearest_prototype,
    _parametric_optimizer,
    _set_method_learning_rate,
    _validate_config,
)


def _config() -> dict:
    return {
        "method_id": "protonet_cda",
        "stage": "Stage2-B",
        "target_new_tx_labels": [],
        "target_unknown_tx_labels": [],
        "target_receiver_labels": ["20-1"],
        "base_steps": 2,
        "adapt_steps": 2,
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
    }


def test_supervised_da_config_requires_one_target_receiver() -> None:
    config = _config()
    config["target_receiver_labels"] = ["20-1", "3-19"]
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "exactly one target receiver" in str(exc)
    else:
        raise AssertionError("pooled target-receiver adaptation should fail closed")


def test_supervised_da_config_rejects_clean_formal_test() -> None:
    config = _config()
    config["target_channel_scenarios"] = ["clean"]
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "non-clean satellite" in str(exc)
    else:
        raise AssertionError("clean-only formal test should fail closed")


def test_supervised_da_stage2b_rejects_new_classes() -> None:
    config = _config()
    config["target_new_tx_labels"] = ["new"]
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "target-old classes only" in str(exc)
    else:
        raise AssertionError("Stage2-B target-new support should fail closed")


def test_nearest_prototype_uses_labeled_target_support() -> None:
    support = torch.tensor([[0.0, 0.0], [0.2, 0.0], [4.0, 4.0], [4.2, 4.0]])
    labels = torch.tensor([2, 2, 7, 7])
    query = torch.tensor([[0.1, 0.1], [4.1, 3.9]])
    assert _nearest_prototype(support, labels, query).tolist() == [2, 7]


def test_mrior_uses_reproduction_adam_profile() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer, profile = _parametric_optimizer(
        {"mrior_adapt_learning_rate": 0.0007}, model, method="mrior_sda", phase="adapt"
    )
    assert isinstance(optimizer, torch.optim.Adam)
    assert profile["learning_rate"] == 0.0007
    assert profile["weight_decay"] == 0.0


def test_dadda_uses_paper_sgd_inverse_schedule() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer, profile = _parametric_optimizer({}, model, method="dadda_sda", phase="base")
    assert isinstance(optimizer, torch.optim.SGD)
    assert profile["learning_rate"] == 0.0001
    assert _set_method_learning_rate(optimizer, profile, step=1, total_steps=11) == 0.0001
    final_lr = _set_method_learning_rate(optimizer, profile, step=11, total_steps=11)
    assert final_lr < 0.0001
