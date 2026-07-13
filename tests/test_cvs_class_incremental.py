import pytest
import torch

from paper_reproduction.cvs_aligned.class_incremental import (
    RUNNERS,
    _detailed_breakdown,
    validate_class_incremental_manifest,
)


def _payload(method="csil"):
    return {
        "method": method,
        "source_receiver_labels": ["source-rx"],
        "target_receiver_labels": ["target-rx"],
        "target_old_tx_labels": ["old-a", "old-b"],
        "target_new_tx_labels": ["new-a"],
        "target_unknown_tx_labels": [],
        "k_shot": 5,
        "query_per_tx": 20,
        "target_channel_view": "satellite/LEO",
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "unknown_rejection_enabled": False,
        "threshold_scope": "support_only_no_unknown_query",
        "base_steps": 20,
        "increment_steps": 20,
    }


@pytest.mark.parametrize("method", ["csil", "mopc_hr", "orthogonal_incremental"])
def test_stage2c_manifest_marks_cvs_extension_and_prevents_query_training(method):
    checked = validate_class_incremental_manifest(_payload(method))
    assert checked["stage"] == "Stage2-C"
    assert checked["cvs_extension"] is True
    assert checked["target_labels_scope"] == "registered_old_and_new_support_only"
    assert checked["query_used_for_training"] is False
    assert checked["query_used_for_model_selection"] is False
    assert checked["unknown_rejection_enabled"] is False


def test_stage2c_manifest_rejects_open_set_and_missing_new_classes():
    open_set = _payload()
    open_set["unknown_rejection_enabled"] = True
    open_set["target_unknown_tx_labels"] = ["unknown-a"]
    with pytest.raises(ValueError, match="excludes unknown"):
        validate_class_incremental_manifest(open_set)

    no_new = _payload()
    no_new["target_new_tx_labels"] = []
    with pytest.raises(ValueError, match="requires target_new"):
        validate_class_incremental_manifest(no_new)


def test_stage2c_manifest_rejects_receiver_and_label_overlap():
    receiver_overlap = _payload()
    receiver_overlap["target_receiver_labels"] = ["source-rx"]
    with pytest.raises(ValueError, match="R_s and R_t"):
        validate_class_incremental_manifest(receiver_overlap)

    label_overlap = _payload()
    label_overlap["target_new_tx_labels"] = ["old-a"]
    with pytest.raises(ValueError, match="Y_old and Y_new"):
        validate_class_incremental_manifest(label_overlap)


@pytest.mark.parametrize("method", ["csil", "mopc_hr", "orthogonal_incremental"])
def test_stage2c_method_adapters_execute_without_query_in_training(method):
    torch.manual_seed(3)
    source_batch = {
        "iq": torch.randn(4, 2, 64),
        "label": torch.tensor([10, 10, 11, 11]),
    }
    support_x = torch.randn(6, 2, 64)
    support_y = torch.tensor([0, 0, 1, 1, 2, 2])
    query_x = torch.randn(6, 2, 64)
    query_y = torch.tensor([0, 0, 1, 1, 2, 2])
    config = {
        "seed": 3,
        "base_steps": 1,
        "increment_steps": 1,
        "old_support_steps": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "embedding_dim": 8,
        "csil_embedding_dim": 8,
        "csil_added_embedding_dim": 4,
        "orthogonal_top_k": 1,
    }

    predicted, pre_old, info = RUNNERS[method](
        config,
        [source_batch],
        [10, 11],
        support_x,
        support_y,
        query_x,
        query_y,
        {0, 1},
        {2},
        torch.device("cpu"),
    )

    assert predicted.shape == query_y.shape
    assert 0.0 <= pre_old <= 1.0
    assert info["trainable_parameters"] > 0
    assert info["paper_mechanisms"]
    assert [row["phase"] for row in info["loss_trace"]] == ["base", "old_support", "increment"]
    assert all(torch.isfinite(torch.tensor(row["loss"])).item() for row in info["loss_trace"])


def test_detailed_breakdown_reports_receiver_transmitter_and_confusion_rows():
    predicted = torch.tensor([0, 1, 2, 1])
    truth = torch.tensor([0, 0, 2, 2])
    metadata = [
        {"rx_label": "rx-a", "tx_label": "tx-a", "role": "target_old_query"},
        {"rx_label": "rx-a", "tx_label": "tx-a", "role": "target_old_query"},
        {"rx_label": "rx-a", "tx_label": "tx-b", "role": "target_new_query"},
        {"rx_label": "rx-a", "tx_label": "tx-b", "role": "target_new_query"},
    ]
    rows = _detailed_breakdown(predicted, truth, metadata, scenario="leo_clear_weak")
    tx_a = next(
        row
        for row in rows
        if row["group_type"] == "per_receiver_transmitter"
        and row["receiver_label"] == "rx-a"
        and row["transmitter_label"] == "tx-a"
    )
    tx_b = next(
        row
        for row in rows
        if row["group_type"] == "per_receiver_transmitter"
        and row["receiver_label"] == "rx-a"
        and row["transmitter_label"] == "tx-b"
    )
    assert tx_a["sample_count"] == 2
    assert tx_a["accuracy"] == 0.5
    assert '"0->1": 1' in tx_a["confusion_json"]
    assert tx_b["accuracy"] == 0.5
    assert {row["group_type"] for row in rows} == {
        "per_receiver",
        "per_transmitter",
        "per_receiver_transmitter",
        "per_receiver_transmitter_day",
    }
