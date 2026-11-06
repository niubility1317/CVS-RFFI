import torch

from experiments.phase1_dura_orthogonal.dura_experts import DuraExperts


def test_experts_initialize_at_dura_and_each_head_receives_gradient():
    torch.manual_seed(31)
    model = DuraExperts()
    received = torch.randn(2, 2, 256)
    anchor = torch.randn_like(received)
    innovation = torch.randn_like(received)
    variance = torch.full((2,), .1)
    output = model(received, anchor, innovation, variance)
    torch.testing.assert_close(output, anchor, rtol=0, atol=0)
    output.square().mean().backward()
    for expert in model.experts:
        for head in (expert.wave, expert.filter_head):
            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
            assert head.weight.grad.abs().sum() > 0


def test_gate_uses_each_record_without_batch_statistics():
    torch.manual_seed(32)
    model = DuraExperts().eval()
    for expert in model.experts:
        torch.nn.init.normal_(expert.wave.weight, std=.01)
    torch.nn.init.normal_(model.gate[-1].weight, std=.1)
    received, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.tensor([.1, .2, .3])
    with torch.no_grad():
        together = model(received, anchor, innovation, variance)
        separate = torch.cat([model(received[i:i+1], anchor[i:i+1], innovation[i:i+1], variance[i:i+1])
                              for i in range(3)])
    torch.testing.assert_close(together, separate, rtol=2e-5, atol=2e-6)
    model(received, anchor, innovation, variance).square().mean().backward()
    assert model.gate[-1].weight.grad is not None
    assert model.gate[-1].weight.grad.abs().sum() > 0
