import numpy as np
import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.native_detail import (
    CAP, NativeDetail, band_windows, bounded_components, feature_pack, total_risk,
)


def test_native_neighboring_band_exposes_an_out_of_band_tone():
    clock = np.arange(8192)
    y = 1. + np.exp(2j * np.pi * .25 * clock)
    side = band_windows(y, 0)
    assert side.shape == (6, 3, 2, 256)
    energy = (side ** 2).sum(2).mean((0, 2))
    assert energy[0] > .99 and max(energy[1:]) < 1e-4


def test_component_cap_keeps_noncarrier_and_preserves_decomposition():
    torch.manual_seed(9)
    n = torch.randn(2, 6, 256, dtype=torch.complex64)
    s = torch.randn_like(n)
    a, b = bounded_components(n, s)
    assert (a + b).abs().square().mean(-1).sqrt().max() <= CAP + 1e-6
    torch.testing.assert_close((a + b).mean(-1), torch.zeros_like(a[..., 0]), atol=1e-6, rtol=0.)


def test_native_features_reach_independent_component_heads():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    observed = torch.randn(2, 6, 2, 256)
    sidebands = torch.randn(2, 6, 3, 2, 256)
    features, _ = feature_pack(observed, observed, sidebands, PacketRestorer('consensus'))
    model = NativeDetail()
    expected = [torch.randn(2, 6, 256, dtype=torch.complex64) for _ in range(2)]
    loss = total_risk(model(features), bounded_components(*expected))
    loss.backward()
    assert features.shape == (2, 6, 10, 256)
    assert torch.isfinite(loss) and model.head.out.weight.grad.flatten(1).norm(dim=1).gt(0).all()
