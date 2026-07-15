from __future__ import annotations

import torch

from paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner import (
    _nearest_prototype,
    _target_cache_manifest_path,
    _target_split_from_cache,
    _validate_config,
)

import numpy as np


def _config() -> dict:
    return {
        "method_id": "mrior_sda", "stage": "Stage2-B",
        "target_new_tx_labels": [], "target_unknown_tx_labels": [],
        "target_receiver_labels": ["20-1"], "target_old_tx_labels": ["a", "b"],
        "source_receiver_labels": ["s0"], "k_shot": 5, "support_pool_max_k": 20,
        "query_per_tx": 2, "adapt_steps": 2, "seed": 713101, "split_seed": 713101,
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "target_channel_view": "leo_weak_only",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False, "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "source_leo_weak_cache_set_manifest": "source/cache_set.json",
        "target_leo_weak_cache_root": "target",
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


def test_adv3b02_da_protocol_rejects_raw_dataset_path() -> None:
    config = _config()
    config["manysig_pkl"] = "ManySig.pkl"
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "raw/clean inputs" in str(exc)
    else:
        raise AssertionError("Phase2 config accepted a raw dataset path")


def test_target_cache_path_is_receiver_and_seed_specific() -> None:
    path = _target_cache_manifest_path(_config())
    assert path.as_posix().endswith("target/rx_20_1/seed_713101/cache_set.json")


def test_target_split_keeps_query_fixed_after_max_support_pool() -> None:
    count_per_class = 22
    tx_ids = np.asarray(["a"] * count_per_class + ["b"] * count_per_class)
    arrays = {
        "dataset_role": np.asarray(["target_old"] * (2 * count_per_class)),
        "rx_ids": np.asarray(["20-1"] * (2 * count_per_class)),
        "tx_ids": tx_ids,
        "sample_ids": np.asarray([f"s{index}" for index in range(2 * count_per_class)]),
        "day_ids": np.asarray(["0"] * (2 * count_per_class)),
        "sig_ids": np.asarray([str(index) for index in range(2 * count_per_class)]),
    }
    config = _config()
    split_k5 = _target_split_from_cache(arrays, config)
    config["k_shot"] = 1
    split_k1 = _target_split_from_cache(arrays, config)
    assert split_k1["query_sample_ids"] == split_k5["query_sample_ids"]
    assert split_k1["support_sample_ids"] == ["s0", f"s{count_per_class}"]
