import inspect

import torch

from cvsrffi.phase1_fcr_decoder import PhysicsOrderedDecoder
from cvsrffi.phase1_fcr_nuisance import NuisanceOutput
from cvsrffi.phase1_fcr_types import FCRConfig


def _nuisance(batch_size: int, *, requires_grad: bool = False) -> NuisanceOutput:
    return NuisanceOutput(
        z_ch=torch.randn(batch_size, 16, requires_grad=requires_grad),
        z_rx=torch.randn(batch_size, 8, requires_grad=requires_grad),
        z_sync=torch.randn(batch_size, 6, requires_grad=requires_grad),
        z_gain=torch.randn(batch_size, 3, requires_grad=requires_grad),
        eta_pred=torch.randn(batch_size, 3, requires_grad=requires_grad),
    )


def test_decoder_enforces_physical_order_shapes_and_variance_bounds() -> None:
    config = FCRConfig(variance_floor=0.02, variance_ceiling=0.20)
    decoder = PhysicsOrderedDecoder(config)
    s_hat = torch.randn(3, config.input_len, dtype=torch.complex64)
    delta_f = torch.randn(3, config.input_len, dtype=torch.complex64)

    out = decoder(s_hat, delta_f, _nuisance(3))

    assert out.mu_iq.shape == (3, 2, config.input_len)
    assert out.log_variance.shape == (3, config.input_len)
    assert decoder.call_trace == ("content", "fingerprint", "channel_receiver")
    torch.testing.assert_close(out.delta_f, delta_f)
    variance = out.log_variance.exp()
    assert (variance >= config.variance_floor).all()
    assert (variance <= config.variance_ceiling).all()
    assert torch.isfinite(out.mu_iq).all()
    assert torch.isfinite(out.log_variance).all()


def test_decoder_has_no_raw_iq_argument_and_is_finite_for_zero_content() -> None:
    config = FCRConfig()
    decoder = PhysicsOrderedDecoder(config)
    parameters = tuple(inspect.signature(decoder.forward).parameters)
    assert parameters == ("s_hat", "delta_f", "nuisance")

    zero = torch.zeros(2, config.input_len, dtype=torch.complex64)
    out = decoder(zero, zero, _nuisance(2))

    assert torch.isfinite(out.mu_iq).all()
    assert torch.isfinite(out.log_variance).all()
    torch.testing.assert_close(out.delta_f, zero)


def test_decoder_backpropagates_to_content_fingerprint_and_all_nuisance_parts() -> None:
    config = FCRConfig()
    decoder = PhysicsOrderedDecoder(config)
    s_hat = torch.randn(2, config.input_len, dtype=torch.complex64, requires_grad=True)
    delta_f = torch.randn(2, config.input_len, dtype=torch.complex64, requires_grad=True)
    nuisance = _nuisance(2, requires_grad=True)

    out = decoder(s_hat, delta_f, nuisance)
    (out.mu_iq.square().mean() + out.log_variance.mean()).backward()

    for tensor in (s_hat, delta_f, nuisance.z_ch, nuisance.z_rx, nuisance.z_sync, nuisance.z_gain):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
