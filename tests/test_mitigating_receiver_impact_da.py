from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
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


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def _synthetic_manysig_compact() -> dict:
    data = []
    rx_labels = ["1-1", "1-19", "3-19", "7-7", "8-8", "14-7", "rx6", "rx7", "rx8", "rx9", "rx10", "rx11"]
    for tx_i in range(6):
        tx_rows = []
        for rx_i in range(12):
            rx_rows = []
            for day_i in range(4):
                eq_rows = []
                for _eq_i in range(2):
                    samples = np.zeros((2, 320, 2), dtype=np.float32)
                    samples[:, :, 0] = float(tx_i + 1)
                    samples[:, :, 1] = float(rx_i + day_i)
                    samples[:, 0, 0] = 100.0 + tx_i
                    eq_rows.append(samples)
                rx_rows.append(eq_rows)
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": [f"tx{i}" for i in range(6)],
        "rx_list": rx_labels,
        "capture_date_list": ["2021_03_01", "2021_03_08", "2021_03_15", "2021_03_23"],
        "equalized_list": [0, 1],
    }


def test_iotj2024_model_exposes_classifier_and_estimate_networks():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, feature_dim=128, hidden_dim=128)
    batch = torch.randn(3, 2, 256)

    outputs = model(batch)

    assert outputs["features"].shape == (3, 128)
    assert outputs["tx_logits"].shape == (3, 6)
    assert outputs["estimate_logits"].shape == (3, 1)
    assert "domain_logits" not in outputs


def test_iotj2024_default_model_uses_standard_resnet18_widths():
    from torch import nn

    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6)
    batch = torch.randn(2, 2, 256)

    outputs = model(batch)

    assert model.feature_extractor.stem[0].out_channels == 64
    assert model.feature_extractor.layer4[-1].bn2.num_features == 512
    assert isinstance(model.feature_extractor.projection, nn.Identity)
    assert outputs["features"].shape == (2, 512)


def test_iotj2024_linked_template_profile_matches_eight_block_resnet1d():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, model_profile="pytorch_template_resnet18_hypothesis_v1")
    outputs = model(torch.randn(2, 2, 256))

    assert len(model.feature_extractor.blocks) == 8
    assert [block.out_channels for block in model.feature_extractor.blocks] == [64, 64, 128, 128, 256, 256, 512, 512]
    assert outputs["features"].shape == (2, 512)
    assert outputs["tx_logits"].shape == (2, 6)
    assert model.classifier.net[1].in_features == 512
    assert isinstance(model.estimate_network.net[1], torch.nn.LeakyReLU)


def test_iotj2024_model_can_export_classifier_without_estimate_network():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, feature_dim=128, hidden_dim=128)
    batch = torch.randn(2, 2, 256)

    logits = model.classify(batch)

    assert logits.shape == (2, 6)


def test_iotj2024_model_exports_ec_only_inference_state_without_t():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=6, feature_dim=128, hidden_dim=128)

    inference_state = model.inference_state_dict()

    assert inference_state
    assert all(key.startswith(("feature_extractor.", "classifier.")) for key in inference_state)
    assert not any(key.startswith("estimate_network.") for key in inference_state)


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


def test_weighted_ce_supports_paper_and_released_trainer_reductions():
    import torch.nn.functional as F

    from paper_reproduction.mitigating_receiver_impact_da.losses import _weighted_ce

    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.5, 1.5]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([0.5, 2.0])
    per_sample = F.cross_entropy(logits, labels, reduction="none")

    paper = _weighted_ce(logits, labels, weights, reduction_mode="paper_sample_mean")
    released = _weighted_ce(logits, labels, weights, reduction_mode="pytorch_weighted_mean")

    assert torch.allclose(paper, (per_sample * weights[labels]).mean())
    assert torch.allclose(released, F.cross_entropy(logits, labels, weight=weights))
    assert not torch.allclose(paper, released)


def test_official_mine_stabilized_objective_matches_released_trainer_formula():
    from paper_reproduction.mitigating_receiver_impact_da.losses import mine_kl_stabilized_objective

    source = torch.tensor([[0.2], [0.4], [0.6]], requires_grad=True)
    target = torch.tensor([[0.1], [0.3]], requires_grad=True)

    terms = mine_kl_stabilized_objective(source, target, ma_et=1.0, ma_rate=0.01)
    et_mean = torch.exp(target.flatten()).mean()
    expected_ma = 0.99 * torch.tensor(1.0) + 0.01 * et_mean
    expected_loss = source.flatten().mean() - (1.0 / expected_ma).detach() * et_mean
    expected_kl = source.flatten().mean() - torch.log(et_mean + 1e-4)

    assert torch.allclose(terms["ma_et"], expected_ma.detach())
    assert torch.allclose(terms["loss"], expected_loss)
    assert torch.allclose(terms["kl"], expected_kl)
    assert terms["loss"].requires_grad


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


def test_official_epoch_state_uses_target_indices_and_zero_count_weights():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState

    state = PseudoLabelState(num_classes=3, target_size=5, target_batches=2)
    labels = torch.tensor([0, 1, 1])
    confidence = torch.tensor([0.8, 0.2, 0.9])

    mask = state.official_threshold_mask(labels, confidence, base_tau=0.7)
    assert mask.tolist() == [True, False, True]

    state.update(
        labels,
        labels[mask],
        target_indices=torch.tensor([0, 2, 4]),
        target_mask=mask,
    )
    assert state.total_seen == 3
    assert state.predicted_counts.tolist() == [1.0, 2.0, 0.0]
    assert state.pseudo_counts.tolist() == [1.0, 1.0, 0.0]

    weights = state.class_weights(
        prior=torch.full((3,), 1.0 / 3.0),
        device=torch.device("cpu"),
    )
    assert torch.allclose(weights, torch.tensor([1.0, 0.5, 1.0]), atol=1e-6)

    state.reset_epoch()
    assert state.total_seen == 0
    assert state.predicted_counts.sum().item() == 0
    assert state.pseudo_counts.sum().item() == 0


def test_class_balance_weights_default_matches_paper_eq9_ratio():
    from paper_reproduction.mitigating_receiver_impact_da.losses import class_balance_weights

    weights = class_balance_weights(
        predicted_counts=torch.tensor([30.0, 10.0, 5.0]),
        total_seen=45,
        prior=torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )

    assert torch.allclose(weights, torch.tensor([0.5, 1.5, 3.0]), atol=1e-6)


def test_class_balance_weights_are_smoothed_clipped_and_mean_normalized():
    from paper_reproduction.mitigating_receiver_impact_da.losses import class_balance_weights

    weights = class_balance_weights(
        predicted_counts=torch.tensor([100.0, 0.0, 0.0]),
        total_seen=100,
        prior=torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        smoothing=10.0,
        clip_min=0.1,
        clip_max=10.0,
        mean_normalize=True,
    )

    assert torch.isfinite(weights).all()
    assert float(weights.max()) <= 10.0
    assert float(weights.min()) >= 0.1
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_estimator_update_features_do_not_mutate_feature_extractor_state():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import _estimate_outputs
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    torch.manual_seed(4)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    model.train()
    before = {key: value.detach().clone() for key, value in model.feature_extractor.state_dict().items()}

    _estimate_outputs(model, torch.randn(4, 2, 256), torch.randn(5, 2, 256))

    after = model.feature_extractor.state_dict()
    changed = [key for key, value in before.items() if not torch.equal(value, after[key])]
    assert changed == []
    assert model.training
    assert model.feature_extractor.training


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
        target_y_audit=torch.tensor([0, 1, 2, 1, 0]),
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
    assert set(["target_selected_correct", "target_audit_total", "target_pred_correct"]).issubset(result)
    assert int(result["target_audit_total"].item()) == 5
    assert torch.isfinite(result["loss"])


def test_table3_component_flags_disable_exact_objective_paths():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

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
        estimate_steps=7,
        use_domain_alignment=False,
        use_pseudo=False,
        use_class_weight=False,
    )

    assert optimizer_t.step_calls == 0
    assert optimizer_ec.step_calls == 1
    assert int(result["target_selected"].item()) == 0
    assert float(result["loss_target"].item()) == 0.0
    assert float(result["loss_kl"].item()) == 0.0
    assert torch.equal(result["class_weight_vector"], torch.ones(3))


@pytest.mark.parametrize(
    ("use_domain_alignment", "use_pseudo", "use_class_weight", "source_scale", "target_scale"),
    [
        (True, False, False, 0.5, 0.0),
        (False, True, False, 0.5, 0.5),
        (False, False, True, 0.5, 0.0),
        (True, True, False, 0.5, 0.5),
        (True, False, True, 0.5, 0.0),
        (False, True, True, 0.5, 0.5),
        (True, True, True, 0.5, 0.5),
    ],
)
def test_table3_seven_component_combinations_preserve_paper_mu_scales(
    use_domain_alignment,
    use_pseudo,
    use_class_weight,
    source_scale,
    target_scale,
):
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

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
        estimate_steps=1,
        base_tau=0.7,
        mu=0.5,
        use_domain_alignment=use_domain_alignment,
        use_pseudo=use_pseudo,
        use_class_weight=use_class_weight,
    )

    assert float(result["source_ce_scale"].item()) == source_scale
    assert float(result["target_ce_scale"].item()) == target_scale
    assert optimizer_t.step_calls == (1 if use_domain_alignment else 0)
    assert optimizer_ec.step_calls == 1
    if not use_pseudo:
        assert int(result["target_selected"].item()) == 0
    if not use_class_weight:
        assert torch.equal(result["class_weight_vector"], torch.ones(3))
    assert torch.allclose(
        result["loss_weighted_ce"],
        source_scale * result["loss_source"] + target_scale * result["loss_target"],
    )


def test_component_source_only_diagnostic_does_not_update_bn_from_target():
    from copy import deepcopy

    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    source_only_reference = deepcopy(model)
    source_x = torch.randn(4, 2, 256)
    source_y = torch.tensor([0, 1, 2, 1])
    target_x = torch.randn(5, 2, 256) + 10.0

    source_only_reference.train()
    source_only_reference(source_x)
    gada_batch_step(
        model,
        source_x,
        source_y,
        target_x,
        state=PseudoLabelState(num_classes=3),
        optimizer_t=CountingOptimizer(model.estimate_network.parameters()),
        optimizer_ec=CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters())),
        estimate_steps=1,
        use_domain_alignment=False,
        use_pseudo=False,
        use_class_weight=False,
    )

    reference_buffers = dict(source_only_reference.feature_extractor.named_buffers())
    actual_buffers = dict(model.feature_extractor.named_buffers())
    assert reference_buffers.keys() == actual_buffers.keys()
    assert all(torch.equal(reference_buffers[name], actual_buffers[name]) for name in reference_buffers)


def test_algorithm1_ec_kl_updates_feature_extractor_on_training_path():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.losses import dv_kl_domain_alignment

    class ModeSensitiveFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]]))
            self.forward_calls = 0

        def forward(self, x, *, return_activations=False):
            self.forward_calls += 1
            base = x[:, :1, :1].flatten(1) @ self.weight
            scale = 10.0 if self.training else 1.0
            return base * scale, []

    class TinyGADModel(torch.nn.Module):
        num_tx = 2

        def __init__(self):
            super().__init__()
            self.feature_extractor = ModeSensitiveFeatureExtractor()
            self.classifier = torch.nn.Linear(1, 2, bias=False)
            self.estimate_network = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.classifier.weight.zero_()
                self.estimate_network.weight.fill_(1.0)

        def forward(self, x):
            features, _ = self.feature_extractor(x, return_activations=False)
            return {
                "features": features,
                "tx_logits": self.classifier(features),
                "estimate_logits": self.estimate_network(features),
            }

    model = TinyGADModel()
    source_x = torch.zeros(2, 2, 256)
    target_x = torch.zeros(2, 2, 256)
    source_x[:, 0, 0] = torch.tensor([0.0, 1.0])
    target_x[:, 0, 0] = torch.tensor([2.0, 3.0])
    optimizer_t = CountingOptimizer(model.estimate_network.parameters())
    optimizer_ec = CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters()))

    result = gada_batch_step(
        model,
        source_x,
        torch.tensor([0, 1]),
        target_x,
        state=PseudoLabelState(num_classes=2),
        optimizer_t=optimizer_t,
        optimizer_ec=optimizer_ec,
        estimate_steps=1,
    )
    expected_training_path = dv_kl_domain_alignment(torch.tensor([[0.0], [10.0]]), torch.tensor([[20.0], [30.0]]))

    assert torch.allclose(result["estimate_zeta"], expected_training_path, atol=1e-5)
    assert torch.allclose(result["loss_kl"], expected_training_path, atol=1e-5)
    assert model.feature_extractor.forward_calls == 2


def test_official_mine_resets_ma_for_each_t_step_and_reuses_last_for_ec(monkeypatch):
    import paper_reproduction.mitigating_receiver_impact_da.algorithm as algorithm
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    observed_ma: list[float] = []

    def fake_mine(source, target, *, ma_et=1.0, ma_rate=0.01, eps=1e-4):
        del ma_rate, eps
        observed_ma.append(float(torch.as_tensor(ma_et).mean().item()))
        objective = source.mean() - target.mean()
        return {"kl": objective, "ma_et": torch.tensor(2.0), "loss": objective}

    monkeypatch.setattr(algorithm, "mine_kl_stabilized_objective", fake_mine)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    algorithm.gada_batch_step(
        model,
        torch.randn(4, 2, 256),
        torch.tensor([0, 1, 2, 1]),
        torch.randn(4, 2, 256),
        state=algorithm.PseudoLabelState(num_classes=3),
        optimizer_t=CountingOptimizer(model.estimate_network.parameters()),
        optimizer_ec=CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters())),
        estimate_steps=3,
        kl_estimator_mode="mine_ma",
    )

    assert observed_ma == [1.0, 1.0, 1.0, 2.0]


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
        "paper_scope": "paper_equations_bounded",
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
    assert payload["method_id"] == "mitigating_receiver_impact_da"
    assert payload["algorithm"] == "GAD adversarial training with DV-KL domain alignment and adaptive pseudo-labeling"
    assert "Proposed_GAD_DVKL_CPL_class_weighting" in method_names
    assert display_methods == {"Source only", "DANN", "MCD", "SHOT", "Proposed"}
    assert payload["target_labels_scope"] == "evaluation_only"
    assert "first-batch class-weight fallback" in payload["paper_unspecified_fields"]
    assert "empty pseudo-label target loss fallback" in payload["paper_unspecified_fields"]
    assert "zero-count CPL threshold floor" in payload["paper_unspecified_fields"]
    assert payload["paper_evidence_targets"]["Table II"] == "task/display-method plan only"
    assert payload["paper_evidence_targets"]["Table III"] == "not reproduced in dry-run"
    assert payload["paper_evidence_targets"]["Table IV"] == "not reproduced in dry-run"
    assert payload["paper_evidence_targets"]["Fig.5-7"] == "not reproduced in dry-run"
    assert "DANN_plus_LMMD_subdomain_adaptation" not in method_names
    assert "target_labeled_retrain_upper_bound" not in method_names
    assert "receiver_ratio_plan" not in payload
    assert all("table_i_target_receiver_count" not in row for row in payload["paper_task_plan"])


def test_protocol_rejects_target_label_training_scope():
    from paper_reproduction.mitigating_receiver_impact_da.train import build_dry_run_payload

    config = {
        "paper_scope": "paper_equations_bounded",
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


def test_manysig_task_builder_matches_table2_receiver_and_day_tasks():
    from paper_reproduction.mitigating_receiver_impact_da.data import build_manysig_task_datasets

    cross_rx = build_manysig_task_datasets(_synthetic_manysig_compact(), task="14-7->3-19", max_samples_per_combo=1)
    source_x, source_y, _source_d, source_meta = cross_rx["source"][0]
    target_x, target_y, _target_d, target_meta = cross_rx["target"][0]

    assert source_x.shape == (2, 256)
    assert target_x.shape == (2, 256)
    assert int(source_y) in range(6)
    assert int(target_y) in range(6)
    assert source_meta["rx"] == "14-7"
    assert target_meta["rx"] == "3-19"
    assert cross_rx["meta"]["target_label_role"] == "hidden_for_UDA_training_available_for_final_accuracy"

    cross_day = build_manysig_task_datasets(_synthetic_manysig_compact(), task="d01->d23", max_samples_per_combo=1)
    source_days = {meta["day"] for *_unused, meta in [cross_day["source"][idx] for idx in range(len(cross_day["source"]))]}
    target_days = {meta["day"] for *_unused, meta in [cross_day["target"][idx] for idx in range(len(cross_day["target"]))]}
    assert source_days == {"2021_03_01"}
    assert target_days == {"2021_03_23"}


def test_paper_train_loaders_drop_partial_batches():
    from paper_reproduction.mitigating_receiver_impact_da.data import build_manysig_task_loaders

    loaders = build_manysig_task_loaders(
        _synthetic_manysig_compact(),
        task="14-7->3-19",
        batch_size=5,
        max_samples_per_combo=1,
        seed=3,
    )

    assert len(loaders["source"].dataset) == 24
    assert len(loaders["source"]) == 4
    assert len(loaders["target_train"]) == 4
    assert all(batch["iq"].shape[0] == 5 for batch in loaders["source"])
    target_batch = next(iter(loaders["target_train"]))
    assert "label" not in target_batch
    assert all("tx" not in meta and "tx_i" not in meta for meta in target_batch["meta"])
    assert all(meta["target_label_visible"] is False for meta in target_batch["meta"])
    assert len(loaders["target_eval"]) == 5


def test_paper_dataset_centers_then_power_normalizes_iq():
    from paper_reproduction.mitigating_receiver_impact_da.data import build_manysig_task_datasets

    built = build_manysig_task_datasets(
        _synthetic_manysig_compact(), task="14-7->3-19", max_samples_per_combo=1
    )
    iq, *_ = built["source"][0]

    assert torch.allclose(iq.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert torch.allclose((iq.square().sum(dim=0)).mean(), torch.tensor(1.0), atol=1e-5)
    assert built["meta"]["preprocessing"]["center"] is True


def test_table2_runner_smoke_trains_source_only_and_proposed_rows(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["14-7->3-19"],
        methods=["source_only", "proposed"],
        output_dir=tmp_path,
        epochs=2,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        base_tau=0.95,
        class_prior_mode="source",
        enable_target_label_audit=True,
        seed=3,
        device="cpu",
    )

    assert result["method_id"] == "mitigating_receiver_impact_da"
    assert result["result_claim_status"] == "diagnostic_only"
    assert result["reproduction_profile"] == "diagnostic_extension"
    assert [row["method"] for row in result["rows"]] == ["source_only", "proposed"]
    assert all(row["task"] == "14-7->3-19" for row in result["rows"])
    assert result["rows"][0]["target_labels_scope"] == "evaluation_only"
    assert result["rows"][1]["target_labels_scope"] == "training_time_audit_and_final_evaluation_no_optimization"
    assert all(0.0 <= row["target_accuracy"] <= 1.0 for row in result["rows"])
    assert len(result["rows"][0]["history"]) == 2
    assert len(result["rows"][1]["history"]) == 2
    assert "source_pretrain_history" in result["rows"][1]
    assert len(result["rows"][1]["source_pretrain_history"]) == 2
    assert result["rows"][1]["target_true_hist"] == [4, 4, 4, 4, 4, 4]
    assert len(result["rows"][1]["target_accuracy_by_class"]) == 6
    assert len(result["rows"][1]["target_confusion_matrix"]) == 6
    assert result["base_tau"] == 0.95
    assert result["class_prior_mode"] == "source"
    assert result["rows"][1]["class_prior_mode"] == "source"
    assert torch.allclose(torch.tensor(result["rows"][1]["class_prior"]), torch.full((6,), 1.0 / 6.0))
    assert result["rows"][1]["source_pretrain_target_audit"]["total"] > 0
    assert {row["tau"] for row in result["rows"][1]["source_pretrain_target_audit"]["tau_sweep"]} == {0.7, 0.95}
    assert (tmp_path / "14-7_to_3-19_source_only.pt").exists()
    assert (tmp_path / "14-7_to_3-19_proposed.pt").exists()


def test_table2_runner_defaults_to_paper_uniform_prior_and_tau(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["14-7->3-19"],
        methods=["proposed"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        seed=4,
        device="cpu",
    )

    row = result["rows"][0]
    assert result["base_tau"] == 0.7
    assert result["class_prior_mode"] == "uniform"
    assert row["class_prior_mode"] == "uniform"
    assert torch.allclose(torch.tensor(row["class_prior"]), torch.full((6,), 1.0 / 6.0))
    assert result["pseudo_state_scope"] == "epoch"
    assert result["batch_pairing"] == "zip_min"
    assert result["weighted_ce_reduction"] == "paper_sample_mean"


def test_table2_runner_records_step_lr_scheduler_path(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["14-7->3-19"],
        methods=["proposed"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        source_pretrain_epochs=0,
        lr_scheduler_mode="step",
        lr_step_size=1,
        lr_gamma=0.6,
        seed=6,
        device="cpu",
    )

    row = result["rows"][0]
    assert result["lr_scheduler_mode"] == "step"
    assert result["lr_step_size"] == 1
    assert result["lr_gamma"] == 0.6
    assert row["lr_scheduler_mode"] == "step"
    assert row["lr_scheduler_active"] is True
    assert row["history"][0]["lr_ec"] < result["learning_rate"]
    assert row["history"][0]["lr_t"] < result["learning_rate"]


def test_table2_runner_records_pseudo_floor_and_quota_controls(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["14-7->3-19"],
        methods=["proposed"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        source_pretrain_epochs=0,
        pseudo_threshold_floor=0.4,
        pseudo_quota_mode="balanced_topk",
        pseudo_quota_per_class=2,
        seed=6,
        device="cpu",
    )

    row = result["rows"][0]
    assert result["pseudo_threshold_floor"] == 0.4
    assert result["pseudo_quota_mode"] == "balanced_topk"
    assert result["pseudo_quota_per_class"] == 2
    assert row["pseudo_threshold_floor"] == 0.4
    assert row["pseudo_quota_mode"] == "balanced_topk"
    assert row["pseudo_quota_per_class"] == 2
    assert "target_pseudo_selected_hist" in row["history"][0]
    assert row["status"] == "completed_diagnostic_only"
    assert row["claim_status"] == "diagnostic_only"
    assert row["reproduction_profile"] == "diagnostic_extension"


def test_table2_runner_official_compat_records_released_trainer_path(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["14-7->3-19"],
        methods=["proposed"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        target_model_selection="target_loss_best",
        official_compat=True,
        seed=5,
        device="cpu",
    )

    row = result["rows"][0]
    assert result["official_compat"] is True
    assert result["kl_estimator_mode"] == "mine_ma"
    assert result["pseudo_threshold_mode"] == "official"
    assert result["pseudo_score_mode"] == "logit"
    assert result["class_weight_timing"] == "current"
    assert result["pseudo_state_scope"] == "epoch"
    assert result["batch_pairing"] == "zip_min"
    assert result["weighted_ce_reduction"] == "pytorch_weighted_mean"
    assert result["source_pretrain_epochs"] == 0
    assert row["official_compat"] is True
    assert row["target_model_selection"] == "target_loss_best"
    assert row["status"] == "completed_diagnostic_only"
    assert row["reproduction_profile"] == "oracle_diagnostic"
    assert row["claim_status"] == "diagnostic_only"
    assert row["best_target_loss_epoch"] == 1
    assert len(row["target_eval_history"]) == 1


def test_table2_runner_official_compat_safe_pseudo_keeps_official_state_but_probability_cpl(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.train import run_table2_reproduction

    result = run_table2_reproduction(
        _synthetic_manysig_compact(),
        tasks=["14-7->3-19"],
        methods=["proposed"],
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        max_samples_per_combo=1,
        max_batches_per_epoch=1,
        target_model_selection="target_loss_best",
        official_compat=True,
        official_compat_safe_pseudo=True,
        seed=5,
        device="cpu",
    )

    row = result["rows"][0]
    assert result["official_compat"] is True
    assert result["official_compat_safe_pseudo"] is True
    assert result["kl_estimator_mode"] == "mine_ma"
    assert result["pseudo_threshold_mode"] == "paper"
    assert result["pseudo_score_mode"] == "probability"
    assert result["class_weight_timing"] == "current"
    assert result["pseudo_state_scope"] == "epoch"
    assert result["batch_pairing"] == "zip_min"
    assert result["source_pretrain_epochs"] == 0
    assert row["official_compat"] is True
    assert row["official_compat_safe_pseudo"] is True
    assert row["reproduction_profile"] == "oracle_diagnostic"
    assert row["claim_status"] == "diagnostic_only"


def test_reproduction_profiles_fail_closed_for_strict_released_and_diagnostics():
    from paper_reproduction.mitigating_receiver_impact_da.train import _classify_reproduction_claim

    base = {
        "official_compat": False,
        "official_compat_safe_pseudo": False,
        "class_prior_mode": "uniform",
        "class_weight_smoothing": 0.0,
        "class_weight_clip_min": None,
        "class_weight_clip_max": None,
        "class_weight_mean_normalize": False,
        "kl_estimator_mode": "dvkl",
        "pseudo_threshold_mode": "paper",
        "pseudo_score_mode": "probability",
        "pseudo_threshold_floor": 0.0,
        "pseudo_quota_mode": "none",
        "class_weight_timing": "previous",
        "pseudo_state_scope": "epoch",
        "batch_pairing": "zip_min",
        "adapt_start_epoch": 0,
        "label_smoothing": 0.0,
        "target_model_selection": "final",
        "weighted_ce_reduction": "paper_sample_mean",
        "lr_scheduler_mode": "none",
        "model_profile": "standard_resnet18",
        "learning_rate": 0.0006,
        "estimate_steps": 7,
        "base_tau": 0.7,
        "mu": 0.5,
        "kl_weight": 0.005,
        "use_domain_alignment": True,
        "use_pseudo": True,
        "use_class_weight": True,
        "max_samples_per_combo": None,
        "max_batches_per_epoch": None,
    }

    strict = _classify_reproduction_claim(**base)
    assert strict["reproduction_profile"] == "paper_equations_bounded"
    assert strict["claim_status"] == "bounded_paper_reproduction"

    released_args = dict(base)
    released_args.update(
        official_compat=True,
        class_prior_mode="source",
        kl_estimator_mode="mine_ma",
        pseudo_threshold_mode="official",
        pseudo_score_mode="logit",
        class_weight_timing="current",
        adapt_start_epoch=10,
        weighted_ce_reduction="pytorch_weighted_mean",
    )
    released = _classify_reproduction_claim(**released_args)
    assert released["reproduction_profile"] == "released_trainer_semantics_bounded"
    assert released["claim_status"] == "bounded_released_trainer_semantics"

    quota_args = dict(base)
    quota_args.update(pseudo_quota_mode="balanced_topk")
    diagnostic = _classify_reproduction_claim(**quota_args)
    assert diagnostic["reproduction_profile"] == "diagnostic_extension"
    assert diagnostic["claim_status"] == "diagnostic_only"

    ablation_args = dict(base)
    ablation_args.update(use_domain_alignment=False)
    ablation = _classify_reproduction_claim(**ablation_args)
    assert ablation["reproduction_profile"] == "paper_ablation_diagnostic"
    assert ablation["claim_status"] == "diagnostic_only"

    composite_args = dict(ablation_args)
    composite_args.update(model_profile="pytorch_template_resnet18_hypothesis_v1", max_batches_per_epoch=1)
    composite = _classify_reproduction_claim(**composite_args)
    assert composite["reproduction_profile"] == "paper_ablation_diagnostic"
    assert len(composite["claim_reasons"]) >= 3

    architecture_args = dict(base)
    architecture_args.update(model_profile="pytorch_template_resnet18_hypothesis_v1")
    architecture = _classify_reproduction_claim(**architecture_args)
    assert architecture["reproduction_profile"] == "architecture_hypothesis_diagnostic"
    assert architecture["claim_status"] == "diagnostic_only"

    scalar_args = dict(base)
    scalar_args.update(mu=0.6)
    scalar = _classify_reproduction_claim(**scalar_args)
    assert scalar["reproduction_profile"] == "diagnostic_extension"
    assert scalar["claim_status"] == "diagnostic_only"

    oracle_args = dict(base)
    oracle_args.update(target_model_selection="target_loss_best")
    oracle = _classify_reproduction_claim(**oracle_args)
    assert oracle["reproduction_profile"] == "oracle_diagnostic"
    assert oracle["claim_status"] == "diagnostic_only"


def test_source_class_prior_is_counted_from_labeled_source_index():
    from dataclasses import dataclass

    from paper_reproduction.mitigating_receiver_impact_da.train import _source_class_prior_from_dataset

    @dataclass(frozen=True)
    class Item:
        tx_i: int

    class SourceDataset:
        index = [Item(0), Item(0), Item(1), Item(2)]

        def __len__(self):
            return len(self.index)

    prior = _source_class_prior_from_dataset(SourceDataset(), num_classes=3)

    assert torch.allclose(prior, torch.tensor([0.5, 0.25, 0.25]))


def test_source_class_prior_falls_back_when_index_lacks_tx_labels():
    from dataclasses import dataclass

    from paper_reproduction.mitigating_receiver_impact_da.train import _source_class_prior_from_dataset

    @dataclass(frozen=True)
    class Item:
        tx_i: int | None = None

    class SourceDataset:
        index = [Item(0), Item(None)]

        def __len__(self):
            return 3

        def __getitem__(self, idx):
            labels = [2, 2, 1]
            return torch.zeros(2, 256), labels[idx], 0, {}

    prior = _source_class_prior_from_dataset(SourceDataset(), num_classes=3)

    assert torch.allclose(prior, torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0]))


def test_target_prediction_audit_reports_tau_precision_and_coverage():
    from paper_reproduction.mitigating_receiver_impact_da.train import _audit_target_predictions

    class FixedModel:
        def eval(self):
            return self

        def classify(self, x):
            return torch.tensor(
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 3.0],
                    [1.0, 2.0],
                ]
            )[: x.shape[0]]

    loader = [
        {
            "iq": torch.zeros(4, 2, 256),
            "label": torch.tensor([0, 1, 0, 0]),
        }
    ]

    audit = _audit_target_predictions(FixedModel(), loader, device="cpu", tau_values=(0.7, 0.95))

    assert audit["target_pred_acc"] == 0.5
    by_tau = {row["tau"]: row for row in audit["tau_sweep"]}
    assert by_tau[0.7]["selected"] == 4
    assert by_tau[0.7]["selected_acc"] == 0.5
    assert audit["target_true_hist"] == [3, 1]
    assert audit["target_pred_hist"] == [1, 3]
    assert audit["target_acc_by_true_class"] == [1.0 / 3.0, 1.0]
    assert by_tau[0.7]["selected_true_hist"] == [3, 1]
    assert by_tau[0.7]["selected_acc_by_true_class"] == [1.0 / 3.0, 1.0]
    assert by_tau[0.95]["selected"] == 2
    assert by_tau[0.95]["selected_acc"] == 1.0


def test_target_metric_evaluation_reports_per_class_confusion():
    from paper_reproduction.mitigating_receiver_impact_da.train import _evaluate_target_metrics

    class FixedModel:
        num_tx = 3

        def eval(self):
            return self

        def classify(self, x):
            return torch.tensor(
                [
                    [5.0, 1.0, 0.0],
                    [0.0, 5.0, 1.0],
                    [0.0, 4.0, 3.0],
                    [1.0, 0.0, 5.0],
                ]
            )[: x.shape[0]]

    loader = [
        {
            "iq": torch.zeros(4, 2, 256),
            "label": torch.tensor([0, 1, 2, 2]),
        }
    ]

    metrics = _evaluate_target_metrics(FixedModel(), loader, device="cpu", include_loss=True)

    assert metrics["target_accuracy"] == 0.75
    assert metrics["target_true_hist"] == [1, 1, 2]
    assert metrics["target_pred_hist"] == [1, 2, 1]
    assert metrics["target_correct_by_class"] == [1, 1, 1]
    assert metrics["target_accuracy_by_class"] == [1.0, 1.0, 0.5]
    assert metrics["target_confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert "target_loss" in metrics


def test_pseudo_threshold_floor_and_balanced_quota_limit_selected_labels():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, _select_pseudo_labels

    state = PseudoLabelState(num_classes=2)
    logits = torch.log(
        torch.tensor(
            [
                [0.91, 0.09],
                [0.88, 0.12],
                [0.72, 0.28],
                [0.15, 0.85],
            ]
        )
    )

    labels, mask, confidence, thresholds = _select_pseudo_labels(
        logits,
        state,
        base_tau=0.7,
        threshold_mode="paper",
        score_mode="probability",
        threshold_floor=0.8,
        quota_mode="balanced_topk",
        quota_per_class=1,
    )

    assert labels.tolist() == [0, 0, 0, 1]
    assert torch.allclose(confidence, torch.tensor([0.91, 0.88, 0.72, 0.85]))
    assert torch.allclose(thresholds, torch.tensor([0.8, 0.8]))
    assert mask.tolist() == [True, False, False, True]


def test_algorithm1_training_loop_runs_epochs_and_writes_checkpoint(tmp_path):
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
    from paper_reproduction.mitigating_receiver_impact_da.train import run_gada_training_loop

    torch.manual_seed(2)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    optimizer_t = CountingOptimizer(model.estimate_network.parameters())
    optimizer_ec = CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters()))
    source_batches = [
        {"iq": torch.randn(3, 2, 256), "label": torch.tensor([0, 1, 2])},
        {"iq": torch.randn(3, 2, 256), "label": torch.tensor([1, 2, 0])},
    ]
    target_batches = [{"iq": torch.randn(4, 2, 256), "label": torch.tensor([0, 1, 2, 1])}]
    checkpoint_path = tmp_path / "gada_smoke.pt"

    result = run_gada_training_loop(
        model,
        source_batches,
        target_batches,
        optimizer_t=optimizer_t,
        optimizer_ec=optimizer_ec,
        epochs=2,
        checkpoint_path=checkpoint_path,
    )

    assert result["epochs"] == 2
    assert result["target_label_audit_enabled"] is False
    assert result["batches"] == 2
    assert len(result["history"]) == 2
    assert "target_pseudo_selected_acc" in result["history"][0]
    assert "target_pred_acc" in result["history"][0]
    assert result["history"][0]["target_pred_acc"] is None
    assert result["history"][0]["target_true_hist"] == [0, 0, 0]
    assert "target_pred_hist" in result["history"][0]
    assert "target_pseudo_selected_hist" in result["history"][0]
    assert "target_pred_acc_by_true_class" in result["history"][0]
    assert "target_pseudo_selected_acc_by_pred_class" in result["history"][0]
    assert "class_weight_mean_by_class" in result["history"][0]
    assert "pseudo_threshold_mean_by_class" in result["history"][0]
    assert "estimate_zeta_mean" in result["history"][0]
    assert result["state"]["total_seen"] == 4
    assert optimizer_t.step_calls == 14
    assert optimizer_ec.step_calls == 2
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["paper"] == "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation"
    assert checkpoint["algorithm"] == "Algorithm 1 GAD training loop"
    assert checkpoint["epoch"] == 2
    assert "model_state_dict" in checkpoint


def test_real_batch_norm_buffers_update_once_per_source_and_target_forward():
    from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    tracked = model.feature_extractor.stem[1].num_batches_tracked
    before = int(tracked.item())

    gada_batch_step(
        model,
        torch.randn(4, 2, 256),
        torch.tensor([0, 1, 2, 1]),
        torch.randn(4, 2, 256),
        state=PseudoLabelState(num_classes=3),
        optimizer_t=CountingOptimizer(model.estimate_network.parameters()),
        optimizer_ec=CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters())),
        estimate_steps=1,
    )

    assert int(tracked.item()) - before == 2


def test_reseeding_makes_model_initialization_independent_of_prior_rng_use():
    from paper_reproduction.common.wisig_runtime import set_seed
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet

    set_seed(17)
    first = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    _ = torch.randn(100)
    set_seed(17)
    second = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_algorithm1_training_loop_steps_lr_schedulers_once_per_batch():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
    from paper_reproduction.mitigating_receiver_impact_da.train import run_gada_training_loop

    torch.manual_seed(7)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    optimizer_t = CountingOptimizer(model.estimate_network.parameters())
    optimizer_ec = CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters()))
    scheduler_t = CountingScheduler()
    scheduler_ec = CountingScheduler()
    source_batches = [
        {"iq": torch.randn(3, 2, 256), "label": torch.tensor([0, 1, 2])},
        {"iq": torch.randn(3, 2, 256), "label": torch.tensor([1, 2, 0])},
    ]
    target_batches = [{"iq": torch.randn(4, 2, 256), "label": torch.tensor([0, 1, 2, 1])}]

    result = run_gada_training_loop(
        model,
        source_batches,
        target_batches,
        optimizer_t=optimizer_t,
        optimizer_ec=optimizer_ec,
        scheduler_t=scheduler_t,
        scheduler_ec=scheduler_ec,
        epochs=2,
    )

    assert result["batches"] == 2
    assert result["lr_scheduler_active"] is True
    assert scheduler_t.step_calls == 2
    assert scheduler_ec.step_calls == 2
    assert optimizer_t.step_calls == 14
    assert optimizer_ec.step_calls == 2
    assert "lr_ec" in result["history"][0]
    assert "lr_t" in result["history"][0]


def test_algorithm1_training_loop_records_pseudo_floor_and_quota():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
    from paper_reproduction.mitigating_receiver_impact_da.train import run_gada_training_loop

    torch.manual_seed(8)
    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)
    source_batches = [{"iq": torch.randn(3, 2, 256), "label": torch.tensor([0, 1, 2])}]
    target_batches = [{"iq": torch.randn(6, 2, 256), "label": torch.tensor([0, 1, 2, 0, 1, 2])}]

    result = run_gada_training_loop(
        model,
        source_batches,
        target_batches,
        optimizer_t=CountingOptimizer(model.estimate_network.parameters()),
        optimizer_ec=CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters())),
        epochs=1,
        pseudo_threshold_floor=0.3,
        pseudo_quota_mode="balanced_topk",
        pseudo_quota_per_class=2,
    )

    assert result["pseudo_threshold_floor"] == 0.3
    assert result["pseudo_quota_mode"] == "balanced_topk"
    assert result["pseudo_quota_per_class"] == 2
    assert result["history"][0]["target_selected"] <= 6
    assert max(result["history"][0]["target_pseudo_selected_hist"]) <= 2


def test_algorithm1_training_loop_requires_unlabeled_target_batches():
    from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
    from paper_reproduction.mitigating_receiver_impact_da.train import run_gada_training_loop

    model = ReceiverImpactGADNet(num_tx=3, feature_dim=8, hidden_dim=8)

    try:
        run_gada_training_loop(
            model,
            [{"iq": torch.randn(3, 2, 256), "label": torch.tensor([0, 1, 2])}],
            [],
            optimizer_t=CountingOptimizer(model.estimate_network.parameters()),
            optimizer_ec=CountingOptimizer(list(model.feature_extractor.parameters()) + list(model.classifier.parameters())),
            epochs=1,
        )
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("GAD training must require unlabeled target batches")


def test_cli_non_dry_run_fails_before_writing_output(tmp_path):
    output_path = tmp_path / "should_not_exist.json"
    command = [
        sys.executable,
        "-m",
        "paper_reproduction.mitigating_receiver_impact_da.train",
        "--config",
        "paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json",
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)

    assert completed.returncode != 0
    assert "formal WiSig training CLI is intentionally gated" in completed.stderr
    assert not output_path.exists()
