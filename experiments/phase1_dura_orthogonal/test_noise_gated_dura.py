import torch
from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate, NoiseGatedDuraTimeVarying


def test_gate_respects_calibrated_bounds_and_uses_each_record():
    anchor = torch.ones(3, 2, 256)
    variance = torch.tensor([10.**-4., 10.**-3.3, 10.**-2.5])
    gate = noise_gate(anchor, variance).flatten()
    torch.testing.assert_close(gate, torch.tensor([0., .5, 1.]), atol=1e-6, rtol=0)


def test_low_noise_returns_anchor_even_when_correction_nonzero():
    torch.manual_seed(2)
    model = NoiseGatedDuraTimeVarying(width=8, taps=5)
    torch.nn.init.normal_(model.wave.weight, std=.3)
    x, anchor, innovation = (torch.randn(2, 2, 256) for _ in range(3))
    power = anchor.square().mean((1, 2))
    variance = power * torch.tensor([1e-5, 1e-2])
    out = model(x, anchor, innovation, variance)
    torch.testing.assert_close(out[0], anchor[0], rtol=0, atol=0)
    assert not torch.equal(out[1], anchor[1])
