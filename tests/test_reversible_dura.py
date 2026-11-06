import torch

from experiments.phase1_dura_orthogonal.reversible_dura import ReversibleDura


def test_zero_initialization_matches_dura_and_gradients_reach_all_heads():
    torch.manual_seed(4)
    model = ReversibleDura(width=8)
    x = torch.randn(4, 2, 256)
    anchor = torch.randn_like(x)
    out = model(x, anchor, torch.randn_like(x), torch.ones(4))
    assert torch.equal(out, anchor)
    (out-x).square().mean().backward()
    assert all(layer.head.weight.grad is not None and torch.isfinite(layer.head.weight.grad).all()
               and torch.count_nonzero(layer.head.weight.grad) for layer in model.layers)


def test_nonzero_couplings_are_reversible_and_ignore_external_context():
    torch.manual_seed(5)
    model = ReversibleDura(width=8)
    for layer in model.layers:
        torch.nn.init.normal_(layer.head.weight, std=.15)
        torch.nn.init.normal_(layer.head.bias, std=.15)
    anchor = torch.randn(7, 2, 256)
    out = model(torch.randn_like(anchor), anchor, torch.randn_like(anchor), torch.ones(7))
    torch.testing.assert_close(model.inverse(out), anchor, atol=1e-6, rtol=1e-5)
    other = model(torch.zeros_like(anchor), anchor, torch.zeros_like(anchor), torch.ones(7)*2)
    assert torch.equal(out, other)
