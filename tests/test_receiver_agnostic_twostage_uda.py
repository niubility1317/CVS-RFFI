from __future__ import annotations

import torch


def test_receiver_agnostic_model_matches_paper_shapes():
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet

    model = ReceiverAgnosticUDANet(num_tx=6, feature_dim=128, classifier_hidden_dim=128)
    batch = torch.randn(4, 2, 256)

    outputs = model(batch, grl_lambda=0.5, return_activations=True)

    assert outputs["features"].shape == (4, 128)
    assert outputs["tx_logits"].shape == (4, 6)
    assert outputs["domain_logits"].shape == (4, 1)
    assert len(outputs["activations"]) == 4


def test_lmmd_loss_uses_target_probabilities_without_target_labels():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import lmmd_loss

    source_features = torch.randn(5, 4, requires_grad=True)
    target_features = torch.randn(6, 4, requires_grad=True)
    source_labels = torch.tensor([0, 1, 2, 0, 1])
    target_probs = torch.softmax(torch.randn(6, 3), dim=1)

    loss = lmmd_loss(source_features, target_features, source_labels, target_probs, num_classes=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert source_features.grad is not None
    assert target_features.grad is not None


def test_uncertainty_sampling_orders_hard_target_samples():
    from paper_reproduction.receiver_agnostic_twostage_uda.sampling import rank_uncertain_samples

    logits = torch.tensor(
        [
            [6.0, 0.1, 0.0],
            [1.1, 1.0, 0.9],
            [2.0, 1.9, 0.1],
            [3.0, 0.2, 0.1],
        ]
    )

    assert rank_uncertain_samples(logits, strategy="entropy", k=2).tolist() == [1, 2]
    assert rank_uncertain_samples(logits, strategy="margin", k=2).tolist() == [1, 2]
    assert rank_uncertain_samples(logits, strategy="least_confidence", k=2).tolist() == [1, 2]


def test_paper_faithful_protocol_builds_receiver_ratio_plan():
    from paper_reproduction.receiver_agnostic_twostage_uda.protocol import (
        build_receiver_ratio_plan,
        validate_paper_faithful_config,
    )

    config = {
        "paper_scope": "paper_faithful",
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "source_receiver_counts": [1, 2, 3, 4, 6],
        "tx_count": 6,
        "target_unlabeled_allowed": True,
        "cvs_extension": False,
    }

    checked = validate_paper_faithful_config(config)
    plan = build_receiver_ratio_plan(checked)

    assert [row["ratio"] for row in plan] == ["1:11", "2:10", "3:9", "4:8", "6:6"]
    assert plan[-1]["table_i_target_receiver_count"] == 6
    assert checked["claim_boundary"] == "paper-faithful closed-set cross-receiver UDA"


def test_paper_faithful_protocol_rejects_cvs_extension_mixing():
    from paper_reproduction.receiver_agnostic_twostage_uda.protocol import validate_paper_faithful_config

    config = {
        "paper_scope": "paper_faithful",
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "source_receiver_counts": [6],
        "tx_count": 6,
        "target_unlabeled_allowed": True,
        "cvs_extension": True,
    }

    try:
        validate_paper_faithful_config(config)
    except ValueError as exc:
        assert "cvs_extension" in str(exc)
    else:
        raise AssertionError("paper-faithful config must reject cvs_extension mixing")
