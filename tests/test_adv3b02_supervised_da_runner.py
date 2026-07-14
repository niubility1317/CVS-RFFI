from __future__ import annotations

import torch

from paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner import (
    _nearest_prototype,
    _validate_config,
)


def _config() -> dict:
    return {
        "method_id": "mrior_sda", "stage": "Stage2-B",
        "target_new_tx_labels": [], "target_unknown_tx_labels": [],
        "target_receiver_labels": ["20-1"], "k_shot": 5, "adapt_steps": 2,
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
    }


def test_adv3b02_da_protocol_accepts_old_only() -> None:
    _validate_config(_config())


def test_adv3b02_da_protocol_rejects_new_classes() -> None:
    config = _config()
    config["target_new_tx_labels"] = ["new"]
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "target-old" in str(exc)
    else:
        raise AssertionError("Stage2-B accepted target-new classes")


def test_adv3b02_protonet_support_prediction() -> None:
    support = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[0.0, 0.1], [3.0, 3.1]])
    assert _nearest_prototype(support, labels, query).tolist() == [0, 1]
