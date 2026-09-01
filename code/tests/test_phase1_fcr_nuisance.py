import torch

from cvsrffi.phase1_fcr_nuisance import StructuredNuisanceEncoder
from cvsrffi.phase1_fcr_types import FCRConfig


def test_nuisance_encoder_uses_only_a_small_structured_latent() -> None:
    config = FCRConfig()
    encoder = StructuredNuisanceEncoder(config)
    out = encoder(torch.randn(3, 2, config.input_len), torch.randn(3, 3))

    assert out.z_ch.shape == (3, config.channel_dim)
    assert out.z_rx.shape == (3, config.receiver_dim)
    assert out.z_sync.shape == (3, config.sync_dim)
    assert out.z_gain.shape == (3, config.gain_dim)
    assert out.eta_pred.shape == (3, 3)
    assert sum(part.size(1) for part in (out.z_ch, out.z_rx, out.z_sync, out.z_gain)) == 33
    assert 33 < 2 * config.input_len
    assert not any("skip" in name.lower() for name, _ in encoder.named_modules())
    assert all(part.ndim == 2 for part in (out.z_ch, out.z_rx, out.z_sync, out.z_gain))


def test_nuisance_encoder_is_finite_and_bounded_for_zero_and_near_zero_iq() -> None:
    config = FCRConfig()
    encoder = StructuredNuisanceEncoder(config)
    iq = torch.zeros(2, 2, config.input_len)
    iq[1, :, 17] = 1.0e-9
    eta_hat = torch.zeros(2, 3)

    out = encoder(iq, eta_hat)

    for part in (out.z_ch, out.z_rx, out.z_sync, out.z_gain, out.eta_pred):
        assert torch.isfinite(part).all()
    assert out.z_sync[:, 0].abs().max() <= torch.pi
    assert out.z_sync[:, 1].abs().max() <= 0.25
    assert out.z_sync[:, 2].abs().max() <= 0.01
    assert out.z_sync[:, 3].abs().max() <= 8.0
    assert out.z_sync[:, 4].abs().max() <= 0.02
    assert out.z_sync[:, 5].abs().max() <= 0.10
    assert out.z_gain.abs().max() <= 2.0
