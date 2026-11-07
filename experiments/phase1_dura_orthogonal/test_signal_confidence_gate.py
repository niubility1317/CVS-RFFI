import torch
from experiments.phase1_dura_orthogonal.signal_confidence_gate import SignalConfidenceGate, source_gate_loss


def inputs(n=3):
    torch.manual_seed(21)
    anchor = torch.randn(n, 2, 256)
    received = torch.randn_like(anchor)
    innovation = torch.randn_like(anchor)
    proposal = anchor + .1 * torch.randn_like(anchor)
    variance = torch.full((n,), .05)
    return received, anchor, innovation, proposal, variance


def test_initial_half_step_zero_direction_and_scale_equivariance():
    model = SignalConfidenceGate().eval()
    x, anchor, direction, proposal, variance = inputs()
    with torch.no_grad():
        result = model(x, anchor, direction, proposal, variance)
        same = model(x, anchor, direction, anchor, variance)
        scale = 3.0
        scaled = model(x*scale, anchor*scale, direction*scale, proposal*scale, variance*scale**2)
    torch.testing.assert_close(result, (anchor+proposal)*.5)
    torch.testing.assert_close(same, anchor)
    torch.testing.assert_close(scaled, result*scale, atol=1e-6, rtol=1e-6)


def test_source_loss_trains_gate_toward_a_better_proposal():
    model = SignalConfidenceGate().train()
    x, anchor, direction, proposal, variance = inputs()
    opt = torch.optim.SGD(model.parameters(), lr=.2)
    before = source_gate_loss(model(x, anchor, direction, proposal, variance), anchor, proposal).item()
    opt.zero_grad()
    source_gate_loss(model(x, anchor, direction, proposal, variance), anchor, proposal).backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    opt.step()
    after = source_gate_loss(model(x, anchor, direction, proposal, variance), anchor, proposal).item()
    assert after < before
