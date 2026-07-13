from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from paper_reproduction.cvs_aligned.supervised_da import (
    dadda_sda_objective,
    mrior_sda_objective,
    validate_supervised_da_manifest,
)


def test_mrior_sda_uses_true_target_support_labels_and_backpropagates() -> None:
    source_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    target_logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]], requires_grad=True)
    source_est = torch.tensor([[0.2], [0.4]], requires_grad=True)
    target_est = torch.tensor([[0.1], [0.3]], requires_grad=True)
    result = mrior_sda_objective(
        {"tx_logits": source_logits, "estimate_logits": source_est},
        {"tx_logits": target_logits, "estimate_logits": target_est},
        source_labels=torch.tensor([0, 1]),
        target_support_labels=torch.tensor([1, 0]),
        target_ce_weight=0.75,
        dvkl_weight=0.01,
    )
    expected_target_ce = F.cross_entropy(target_logits, torch.tensor([1, 0]))
    assert torch.allclose(result["target_support_ce"], expected_target_ce)
    assert torch.allclose(result["class_balance_weights"], torch.ones(2))
    assert torch.allclose(result["weighted_ce"], 0.5 * result["source_ce"] + 0.375 * result["target_support_ce"])
    result["loss"].backward()
    assert source_logits.grad is not None
    assert target_logits.grad is not None
    assert source_est.grad is not None
    assert target_est.grad is not None


def test_dadda_sda_uses_true_labels_for_target_ce_and_lmmd() -> None:
    source_local = torch.tensor([[0.0, 0.0], [2.0, 2.0]], requires_grad=True)
    target_local = torch.tensor([[2.1, 2.0], [0.1, 0.0]], requires_grad=True)
    source_global = source_local.clone().detach().requires_grad_(True)
    target_global = target_local.clone().detach().requires_grad_(True)
    source_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    target_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    result = dadda_sda_objective(
        {"global_features": source_global, "local_features": source_local, "logits": source_logits},
        {"global_features": target_global, "local_features": target_local, "logits": target_logits},
        source_labels=torch.tensor([0, 1]),
        target_support_labels=torch.tensor([1, 0]),
        target_ce_weight=1.0,
        alignment_weight=0.5,
    )
    assert torch.allclose(
        result["target_support_ce"], F.cross_entropy(target_logits, torch.tensor([1, 0]))
    )
    assert 0.0 <= float(result["alpha"]) <= 1.0
    result["loss"].backward()
    assert source_local.grad is not None
    assert target_local.grad is not None
    assert source_logits.grad is not None
    assert target_logits.grad is not None


def test_supervised_da_manifest_blocks_support_query_leakage() -> None:
    payload = {
        "method_id": "mrior_sda",
        "cvs_extension": True,
        "stage": "Stage2-B",
        "k_shot": 5,
        "target_labels_scope": "registered_support_only",
        "target_old_support_sample_ids": ["s0", "s1"],
        "target_old_query_sample_ids": ["q0", "q1"],
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
    }
    checked = validate_supervised_da_manifest(payload)
    assert checked["support_query_disjoint"] is True
    assert checked["paper_faithful_claim_allowed"] is False
    payload["target_old_query_sample_ids"] = ["s1"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_supervised_da_manifest(payload)


@pytest.mark.parametrize("method_id", ["protonet_cda", "mrior_sda", "dadda_sda"])
def test_supervised_da_manifest_accepts_only_registered_target_support(method_id: str) -> None:
    checked = validate_supervised_da_manifest(
        {
            "method_id": method_id,
            "cvs_extension": True,
            "stage": "Stage2-B",
            "k_shot": 10,
            "target_labels_scope": "registered_support_only",
            "target_old_support_sample_ids": ["support"],
            "target_old_query_sample_ids": ["query"],
        }
    )
    assert checked["supervised_target_support"] is True
