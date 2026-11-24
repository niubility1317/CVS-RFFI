import torch

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.dura_carrier_action import (
    CARRIER_BINS, CarrierRestrictedAction, CarrierRestrictedDura,
)
from experiments.pamf_lora_main.dura_prior_width import PacketDura


def test_restricted_action_matches_native_projection_and_adjoint():
    rng = torch.Generator().manual_seed(10)
    count = 256

    def complex_noise(shape):
        return torch.randn(*shape, dtype=torch.complex64, generator=rng)

    native = Action(complex_noise((2, 7)), 1 + .3 * complex_noise((2, count)),
                    .1 * complex_noise((2, count)))
    phase = torch.rand(2, count, generator=rng)
    rotation = torch.polar(torch.ones_like(phase), phase)
    restricted = CarrierRestrictedAction(native, rotation)
    x, y = (complex_noise((2, count)) for _ in range(2))
    keep = torch.fft.fftfreq(count).abs() <= max(CARRIER_BINS) / count

    def project(wave):
        coeff = torch.fft.fft(wave * rotation.conj(), norm='ortho')
        return torch.fft.ifft(coeff * keep, norm='ortho') * rotation

    protected = project(x)
    result = restricted(x)
    torch.testing.assert_close(result, x + project(native(protected) - protected), atol=3e-6, rtol=1e-5)
    torch.testing.assert_close((x.conj() * restricted.adjoint(y)).real.sum(),
                               (result.conj() * y).real.sum(), atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(project(result - x), result - x, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(restricted(x - protected), x - protected, atol=2e-6, rtol=1e-5)


def test_same_initialization_and_joint_gradients_through_restricted_dura():
    torch.manual_seed(78)
    ordinary = PacketDura(wide=True).state_dict()
    torch.manual_seed(78)
    model = CarrierRestrictedDura()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, ordinary[name], rtol=0., atol=0.)
    received = 1 + .2 * torch.randn(1, 6, 256, dtype=torch.complex64)
    phase = torch.linspace(0.3, 9., 256).unsqueeze(0).expand_as(received.real)
    rotation = torch.polar(torch.ones_like(phase), phase)
    coarse, fine = model(received, None, rotation)
    desired = 1 + .15 * torch.randn_like(received)
    ((coarse - desired).abs().square().mean() + (fine - desired).abs().square().mean()).backward()
    for layer in (model.restorer.frozen.coarse, model.restorer.frozen.estimator,
                  *model.restorer.prior):
        grad = [p.grad for p in layer.parameters() if p.grad is not None]
        assert grad and all(x.isfinite().all() for x in grad)
        assert any(x.abs().sum() > 0 for x in grad)
