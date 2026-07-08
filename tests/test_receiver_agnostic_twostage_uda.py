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
    assert checked["claim_boundary"] == "paper-faithful closed-set cross-receiver DA with unlabeled target adaptation"


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


def test_iotj2024_model_exposes_classifier_and_estimate_networks():
    from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, feature_dim=128, hidden_dim=128)
    batch = torch.randn(3, 2, 256)

    outputs = model(batch)

    assert outputs["features"].shape == (3, 128)
    assert outputs["tx_logits"].shape == (3, 6)
    assert outputs["estimate_logits"].shape == (3, 1)
    assert "domain_logits" not in outputs


def test_dv_kl_alignment_and_gada_objective_follow_paper_terms():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import (
        dv_kl_domain_alignment,
        gada_minimax_objective,
    )

    source_outputs = {
        "tx_logits": torch.tensor([[3.0, 0.1], [0.2, 2.2]], requires_grad=True),
        "estimate_logits": torch.tensor([[0.6], [0.8]], requires_grad=True),
    }
    target_outputs = {
        "tx_logits": torch.tensor([[2.0, 0.5], [0.3, 2.4], [1.7, 0.2]], requires_grad=True),
        "estimate_logits": torch.tensor([[0.1], [0.4], [0.2]], requires_grad=True),
    }
    target_probs = torch.softmax(target_outputs["tx_logits"], dim=1)
    target_mask = torch.tensor([True, True, False])
    target_pseudo = torch.tensor([0, 1, 0])
    class_weights = torch.tensor([1.0, 1.5])

    expected_kl = source_outputs["estimate_logits"].mean() - torch.logsumexp(
        target_outputs["estimate_logits"].flatten(), dim=0
    ) + torch.log(torch.tensor(3.0))
    assert torch.allclose(dv_kl_domain_alignment(source_outputs["estimate_logits"], target_outputs["estimate_logits"]), expected_kl)

    terms = gada_minimax_objective(
        source_outputs,
        target_outputs,
        source_labels=torch.tensor([0, 1]),
        target_pseudo_labels=target_pseudo,
        target_mask=target_mask,
        class_weights=class_weights,
        mu=0.5,
        kl_weight=0.005,
    )

    assert set(terms) == {"loss", "loss_weighted_ce", "loss_source", "loss_target", "loss_kl"}
    assert terms["loss"].requires_grad
    assert torch.isfinite(terms["loss"])


def test_cpl_thresholds_pseudo_labels_and_class_weights_match_paper_direction():
    from paper_reproduction.receiver_agnostic_twostage_uda.losses import (
        adaptive_pseudo_labels,
        class_balance_weights,
        curriculum_thresholds,
    )

    thresholds = curriculum_thresholds(torch.tensor([10.0, 5.0, 1.0]), base_tau=0.7)

    assert torch.allclose(thresholds, torch.tensor([0.7, 0.35, 0.07]))

    probs = torch.tensor(
        [
            [0.69, 0.20, 0.11],
            [0.40, 0.45, 0.15],
            [0.15, 0.10, 0.75],
        ]
    )
    labels, mask = adaptive_pseudo_labels(probs, thresholds)

    assert labels.tolist() == [0, 1, 2]
    assert mask.tolist() == [False, True, True]

    weights = class_balance_weights(
        predicted_counts=torch.tensor([30.0, 10.0, 5.0]),
        total_seen=45,
        prior=torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )

    assert weights[0] < weights[1] < weights[2]


def test_protocol_dry_run_names_iotj2024_gada_matrix_not_dann_lmmd():
    from paper_reproduction.receiver_agnostic_twostage_uda.train import build_dry_run_payload

    config = {
        "paper_scope": "paper_faithful",
        "cvs_extension": False,
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "source_receiver_counts": [1, 2, 3, 4, 6],
        "tx_count": 6,
        "target_unlabeled_allowed": True,
        "source_target_tasks": ["d01->d23", "1-1->8-8", "7-7->8-8", "14-7->3-19"],
    }

    payload = build_dry_run_payload(config)
    method_names = {name for row in payload["receiver_ratio_plan"] for name in row["compare_methods"]}

    assert payload["paper"].startswith("Mitigating Receiver Impact")
    assert payload["algorithm"] == "GAD adversarial training with DV-KL domain alignment and adaptive pseudo-labeling"
    assert "proposed_GAD_DVKL_CPL_class_weighting" in method_names
    assert "DANN_plus_LMMD_subdomain_adaptation" not in method_names
