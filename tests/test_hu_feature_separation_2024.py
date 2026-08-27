import torch
import pytest

from paper_reproduction.hu_feature_separation_2024.finetune import fine_tune_tx_step, tx_finetune_parameters
from paper_reproduction.hu_feature_separation_2024.losses import feature_separation_loss
from paper_reproduction.hu_feature_separation_2024.model import FeatureSeparationNet
from paper_reproduction.hu_feature_separation_2024.representation import build_fusion_representation, welch_psd_256


def test_fusion_builds_paper_three_stream_shape_deterministically():
    iq = torch.randn(2, 2, 256)
    fusion_a = build_fusion_representation(iq)
    fusion_b = build_fusion_representation(iq)
    assert fusion_a.shape == (2, 3, 256)
    assert torch.allclose(fusion_a, fusion_b)
    assert torch.allclose(fusion_a[:, :2], iq)


def test_welch_representation_rejects_nonpaper_iq_shape():
    with pytest.raises(ValueError, match="2, 256"):
        welch_psd_256(torch.randn(1, 2, 255))


def test_feature_separation_network_emits_tx_rx_features_and_logits():
    model = FeatureSeparationNet(num_tx=6, num_rx=3)
    outputs = model(torch.randn(2, 2, 256))
    assert outputs["tx_features"].shape == (2, 256)
    assert outputs["rx_features"].shape == (2, 256)
    assert outputs["tx_logits"].shape == (2, 6)
    assert outputs["rx_logits"].shape == (2, 3)


def test_feature_separation_loss_has_all_paper_terms_and_gradients():
    model = FeatureSeparationNet(num_tx=3, num_rx=2)
    outputs = model(torch.randn(3, 2, 256))
    loss, terms = feature_separation_loss(
        outputs,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 0]),
        lambda_similarity=0.25,
        lambda_tx_entropy=0.5,
        lambda_rx_entropy=0.75,
    )
    assert set(terms) == {"tx_ce", "rx_ce", "similarity", "tx_entropy", "rx_entropy", "total"}
    assert loss.item() == pytest.approx(terms["total"].item())
    loss.backward()
    assert model.tx_classifier.weight.grad is not None
    assert model.rx_classifier.weight.grad is not None


def test_tx_finetuning_uses_only_tx_labels_and_keeps_rx_classifier_frozen():
    model = FeatureSeparationNet(num_tx=3, num_rx=2)
    optimizer = torch.optim.SGD(tx_finetune_parameters(model), lr=0.01)
    rx_before = model.rx_classifier.weight.detach().clone()
    result = fine_tune_tx_step(model, torch.randn(3, 2, 256), torch.tensor([0, 1, 2]), optimizer)
    assert result["tx_ce"] > 0
    assert torch.equal(model.rx_classifier.weight.detach(), rx_before)
