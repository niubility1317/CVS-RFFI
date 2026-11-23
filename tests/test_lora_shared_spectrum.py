import torch

from experiments.pamf_lora_main.denoise_components import bands
from experiments.pamf_lora_main.packet_subspace import bounded_near
from experiments.pamf_lora_main.packet_subspace_cross_source import band_norm, physical_components
from experiments.pamf_lora_main.shared_spectrum import (
    SharedSpectrum, common_inputs, errors, match_common, packet_mean,
)


def test_shared_rotation_preserves_physical_dose_and_individual_detail():
    torch.manual_seed(77)
    y = torch.randn(2, 6, 256, dtype=torch.complex64)
    q = torch.full((2, 6, 1), .75)
    rms = .2+torch.rand_like(q)
    prior = bounded_near(torch.randn_like(y))
    net = SharedSpectrum()
    features, anchor = common_inputs(y, .8*y, prior, q, rms)
    first = net(features, anchor)
    assert torch.equal(first, anchor)
    target = match_common(anchor, packet_mean(bounded_near(1j*prior), rms))
    common, fine = errors(first, target).unbind()
    assert common>0 and fine>0
    assert torch.autograd.grad(fine, net.refiner.out.weight)[0].norm()>0
    with torch.no_grad():
        net.refiner.out.bias.copy_(torch.tensor([.4, -.9]))
    updated = prior+(net(features, anchor)-anchor)[:, None]
    torch.testing.assert_close(physical_components(updated, rms)[1],
                               physical_components(prior, rms)[1])
    a, b = bands(prior.shape[-1], prior.device)
    for selected in (a, b):
        torch.testing.assert_close(band_norm(updated[..., selected], rms),
                                   band_norm(prior[..., selected], rms))
    assert torch.equal(updated[..., ~(a|b)], prior[..., ~(a|b)])
