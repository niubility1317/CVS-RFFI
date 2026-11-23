import torch

from experiments.pamf_lora_main.denoise_components import bands
from experiments.pamf_lora_main.packet_subspace import (CAP, PacketSubspace, component_risk,
                                                         components, target_vectors)


def test_packet_axis_projection_and_spectral_support():
    torch.manual_seed(59)
    y = torch.randn(2, 6, 256, dtype=torch.complex64)
    q = torch.full((2, 6, 1), .9)
    model = PacketSubspace(factorized=True)
    a, b = components(y)
    torch.testing.assert_close(a+b, y)
    torch.testing.assert_close(b.mean(1), torch.zeros_like(b[:, 0]), atol=5e-7, rtol=0.)
    torch.testing.assert_close((a.conj()*b).sum(1), torch.zeros_like(b[:, 0]), atol=1e-6, rtol=0.)

    output = model(y, .85*y, q)
    assert torch.count_nonzero(output) == 0
    targets = target_vectors(.85*y, .8*y+torch.roll(y, 2, -1)*.3)
    component_risk(output, targets['factorized']).backward()
    assert model.spectral.out.weight.grad.norm() > 0
    assert model.temporal.out.weight.grad.norm() > 0

    with torch.no_grad():
        model.spectral.out.bias.copy_(torch.linspace(.1, .8, 12))
        model.temporal.out.bias.copy_(torch.linspace(-.9, .3, 12))
    first, second = model.parts(y, .85*y, q)
    torch.testing.assert_close(first, first[:, :1].expand_as(first))
    torch.testing.assert_close(second.mean(1), torch.zeros_like(second[:, 0]), atol=1e-6, rtol=0.)
    result = model(y, .85*y, q)
    a, b = bands(256, y.device)
    assert torch.count_nonzero(result[..., ~(a | b)]) == 0
    assert (result.abs().square().mean((1, 2)).sqrt() <= CAP+1e-6).all()
