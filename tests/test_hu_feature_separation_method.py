import pytest
import torch

from paper_reproduction.hu_feature_separation_2024.model import AttentionResNet18, FeatureSeparationNet
from paper_reproduction.hu_feature_separation_2024.preprocess import preprocess_iq, synchronize_iq
from paper_reproduction.hu_feature_separation_2024.representation import welch_psd_256


def test_preprocess_uses_zero_mean_unit_rms_default_without_changing_shape():
    iq = torch.arange(512, dtype=torch.float32).reshape(1, 2, 256)
    processed = preprocess_iq(iq)
    assert processed.shape == (1, 2, 256)
    assert processed.mean(dim=-1).abs().max().item() < 1e-6
    assert processed.square().mean(dim=(1, 2)).sqrt().item() == pytest.approx(1.0)


def test_preamble_synchronization_extracts_a_256_sample_leading_window():
    preamble = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    payload = torch.randn(2, 256)
    leading_noise = torch.zeros(2, 5)
    iq = torch.cat((leading_noise, preamble, payload), dim=1).unsqueeze(0)
    synchronized = synchronize_iq(iq, preamble)
    assert synchronized.shape == (1, 2, 256)
    assert torch.allclose(synchronized[0], payload)


def test_welch_default_is_deterministic_density_stream_for_complex_256_sample_iq():
    iq = torch.randn(2, 2, 256)
    first = welch_psd_256(iq)
    second = welch_psd_256(iq)
    assert first.shape == (2, 1, 256)
    assert torch.all(first >= 0)
    assert torch.allclose(first, second)


def test_figure6_encoder_has_five_residual_stages_and_only_shared_feature_attention():
    encoder = AttentionResNet18()
    assert encoder.stage_channels == (16, 32, 64, 128, 256)
    assert encoder.residual_stage_count == 5
    assert sum(module.__class__.__name__ == "ChannelAttention" for module in encoder.modules()) == 1
    assert encoder(torch.randn(2, 3, 256)).shape == (2, 512)


def test_feature_separation_network_remains_three_stream_compatible():
    outputs = FeatureSeparationNet(num_tx=6, num_rx=12)(torch.randn(2, 3, 256))
    assert outputs["tx_logits"].shape == (2, 6)
    assert outputs["rx_logits"].shape == (2, 12)
