from __future__ import annotations

import torch


class CountingOptimizer:
    """Count optimizer calls without asserting a real update direction."""

    def __init__(self, params):
        self.params = list(params)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1
        for param in self.params:
            param.grad = None

    def step(self):
        self.step_calls += 1


def test_iotj2024_model_exposes_classifier_and_estimate_networks():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, feature_dim=128, hidden_dim=128)
    batch = torch.randn(3, 2, 256)

    outputs = model(batch)

    assert outputs["features"].shape == (3, 128)
    assert outputs["tx_logits"].shape == (3, 6)
    assert outputs["estimate_logits"].shape == (3, 1)
    assert "domain_logits" not in outputs


def test_iotj2024_model_can_export_classifier_without_estimate_network():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, feature_dim=128, hidden_dim=128)
    batch = torch.randn(2, 2, 256)

    logits = model.classify(batch)

    assert logits.shape == (2, 6)


def test_dv_kl_alignment_and_gada_objective_follow_paper_terms():
    from paper_reproduction.mitigating_receiver_impact_da.losses import (
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
    target_mask = torch.tensor([True, True, False])
    target_pseudo = torch.tensor([0, 1, 0])
    class_weights = torch.tensor([1.0, 1.5])

    expected_kl = source_outputs["estimate_logits"].mean() - torch.logsumexp(
        target_outputs["estimate_logits"].flatten(), dim=0
    ) + torch.log(torch.tensor(3.0))
    assert torch.allclose(
        dv_kl_domain_alignment(source_outputs["estimate_logits"], target_outputs["estimate_logits"]),
        expected_kl,
    )

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
    from paper_reproduction.mitigating_receiver_impact_da.losses import (
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

    equal_probs = torch.tensor([[0.70, 0.20, 0.10]])
    equal_labels, equal_mask = adaptive_pseudo_labels(equal_probs, torch.tensor([0.70, 0.35, 0.07]))
    assert equal_labels.tolist() == [0]
    assert equal_mask.tolist() == [False]

    weights = class_balance_weights(
        predicted_counts=torch.tensor([30.0, 10.0, 5.0]),
        total_seen=45,
        prior=torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )

    assert weights[0] < weights[1] < weights[2]


def test_algorithm1_batch_step_updates_estimator_m_times_then_ec_once():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    torch.manual_seed(0)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    source_x = torch.randn(4, 2, 256)
    source_y = torch.tensor([0, 1, 2, 1])
    target_x = torch.randn(5, 2, 256)
    state = PseudoLabelState(num_classes=3)
    optimizer_t = CountingOptimizer(model.estimate_network.parameters())
    ec_params = list(model.feature_extractor.parameters()) + list(model.classifier.parameters())
    optimizer_ec = CountingOptimizer(ec_params)

    result = gada_batch_step(
        model,
        source_x,
        source_y,
        target_x,
        state=state,
        optimizer_t=optimizer_t,
        optimizer_ec=optimizer_ec,
        estimate_steps=3,
        base_tau=0.7,
        mu=0.5,
        kl_weight=0.005,
    )

    assert optimizer_t.step_calls == 3
    assert optimizer_ec.step_calls == 1
    assert result["estimate_steps"] == 3
    assert state.total_seen == 5
    assert int(state.pseudo_counts.sum().item()) == int(result["target_selected"].item())
    assert int(state.predicted_counts.sum().item()) == 5
    assert torch.isfinite(result["loss"])


def test_algorithm1_batch_step_defaults_to_paper_m7_estimator_updates():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    torch.manual_seed(1)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    optimizer_t = CountingOptimizer(model.estimate_network.parameters())
    optimizer_ec = CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters()))

    result = gada_batch_step(
        model,
        torch.randn(4, 2, 256),
        torch.tensor([0, 1, 2, 1]),
        torch.randn(5, 2, 256),
        state=PseudoLabelState(num_classes=3),
        optimizer_t=optimizer_t,
        optimizer_ec=optimizer_ec,
    )

    assert optimizer_t.step_calls == 7
    assert optimizer_ec.step_calls == 1
    assert result["estimate_steps"] == 7


def test_protocol_dry_run_names_iotj2024_gada_matrix_not_receiver_agnostic_twostage():
    from paper_reproduction.mitigating_receiver_impact_da.train import build_dry_run_payload

    config = {
        "paper_scope": "paper_faithful",
        "cvs_extension": False,
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "tx_count": 6,
        "target_unlabeled_allowed": True,
        "source_target_tasks": ["d01->d23", "1-1->8-8", "7-7->8-8", "14-7->3-19"],
    }

    payload = build_dry_run_payload(config)
    method_names = {name for row in payload["paper_task_plan"] for name in row["compare_method_ids"]}
    display_methods = {name for row in payload["paper_task_plan"] for name in row["paper_display_methods"]}

    assert payload["paper"].startswith("Mitigating Receiver Impact")
    assert payload["algorithm"] == "GAD adversarial training with DV-KL domain alignment and adaptive pseudo-labeling"
    assert "Proposed_GAD_DVKL_CPL_class_weighting" in method_names
    assert display_methods == {"Source only", "DANN", "MCD", "SHOT", "Proposed"}
    assert payload["target_labels_scope"] == "evaluation_only"
    assert "DANN_plus_LMMD_subdomain_adaptation" not in method_names
    assert "target_labeled_retrain_upper_bound" not in method_names
    assert "receiver_ratio_plan" not in payload
    assert all("table_i_target_receiver_count" not in row for row in payload["paper_task_plan"])


def test_protocol_rejects_target_label_training_scope():
    from paper_reproduction.mitigating_receiver_impact_da.train import build_dry_run_payload

    config = {
        "paper_scope": "paper_faithful",
        "cvs_extension": False,
        "dataset": "WiSig ManySig",
        "total_receivers": 12,
        "tx_count": 6,
        "target_unlabeled_allowed": True,
        "target_labels_scope": "training_allowed",
    }

    try:
        build_dry_run_payload(config)
    except ValueError as exc:
        assert "target_labels_scope" in str(exc)
    else:
        raise AssertionError("target_labels_scope other than evaluation_only must be rejected")
