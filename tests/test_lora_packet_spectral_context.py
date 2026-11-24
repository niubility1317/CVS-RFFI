import torch

from experiments.pamf_lora_main.packet_spectral_context import PacketSpectralContext, SpectralPatchMixer
from experiments.pamf_lora_main.tail_pilot_source import WIDTHS


def test_packet_attention_exposes_only_the_requested_cross_chirp_context():
    torch.manual_seed(4)
    local = SpectralPatchMixer(packet=False).eval()
    packet = SpectralPatchMixer(packet=True).eval()
    packet.load_state_dict(local.state_dict())
    signals = torch.randn(1, 6, 256, 8)
    changed = signals.clone()
    changed[:, 1] *= 1.4
    with torch.no_grad():
        first_local = local.mix(signals)
        first_packet = packet.mix(signals)
        torch.testing.assert_close(first_local[:, 0], local.mix(changed)[:, 0], atol=0, rtol=0)
        assert not torch.allclose(first_packet[:, 0], packet.mix(changed)[:, 0])
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in packet.parameters())


def test_spectral_readout_trains_without_moving_the_carrier():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    model = PacketSpectralContext(WIDTHS, packet=True)
    y = torch.randn(2, 6, 256, dtype=torch.complex64) * .6 + 1.
    q = torch.rand(2, 6, 1)
    coarse, restored = model(y, q, torch.ones_like(y))
    assert restored.shape == coarse.shape == y.shape
    torch.testing.assert_close(restored.mean(-1), y.mean(-1), atol=1e-6, rtol=1e-6)
    (restored - torch.ones_like(y)).abs().square().mean().backward()
    grad = model.spectral_mixer.readout.weight.grad
    assert grad.isfinite().all() and grad.abs().sum() > 0
    with torch.no_grad():
        model.spectral_mixer.readout.weight.normal_(0, .1)
        _, changed = model(y, q, torch.ones_like(y))
        assert not torch.allclose(changed, restored)
        torch.testing.assert_close(changed.mean(-1), y.mean(-1), atol=1e-6, rtol=1e-6)
