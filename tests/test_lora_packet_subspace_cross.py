import torch

from experiments.pamf_lora_main.denoise_components import bands
from experiments.pamf_lora_main.packet_subspace import bounded_near
from experiments.pamf_lora_main.packet_subspace_cross_source import (
    band_norm, physical_components, substitutes,
)


def test_weighted_swaps_match_each_actual_band_dose():
    torch.manual_seed(46)
    learned, free, exact = (bounded_near(torch.randn(3, 6, 256, dtype=torch.complex64))
                            for _ in range(3))
    weight = torch.rand(3, 6, 1) + .2
    variation = substitutes(learned, free, exact, weight)
    assert len(variation) == 5
    assert torch.equal(variation[0], substitutes(learned, free, -exact, weight)[0])
    for band in bands(learned.shape[-1], learned.device):
        common, deviation = physical_components(learned[..., band], weight)
        v = (common.conj()*deviation*weight.square()).sum((1, 2))
        torch.testing.assert_close(v, torch.zeros_like(v), atol=1e-6, rtol=0.)
        for value in variation:
            torch.testing.assert_close(band_norm(value[..., band], weight),
                                       band_norm(learned[..., band], weight))
    assert all(torch.equal(z[..., 0], learned[..., 0]) for z in variation)
