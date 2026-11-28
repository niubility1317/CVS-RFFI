import torch

from experiments.pamf_lora_main.dura_real_gain_wisig_packet_gate import PacketGate, gated


def test_packet_gate_is_per_packet_bounded_and_differentiable():
    torch.manual_seed(19)
    gate = PacketGate()
    observed = torch.randn(3, 256, dtype=torch.complex64)
    baseline = torch.randn_like(observed)
    candidate = baseline + .1 * torch.randn_like(observed)
    weight = gate(observed, baseline, candidate)
    assert (weight > 0).all() and (weight < 1).all()
    torch.testing.assert_close(weight[:1], gate(observed[:1], baseline[:1], candidate[:1]))
    result = gated(gate, observed, baseline, candidate)
    result.abs().square().mean().backward()
    assert gate.net[-1].weight.grad is not None
    assert gate.net[-1].weight.grad.norm() > 0
