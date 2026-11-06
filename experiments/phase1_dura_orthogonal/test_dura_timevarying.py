import torch
from experiments.phase1_dura_orthogonal.dura_timevarying import DuraTimeVaryingRecovery


def test_zero_heads_reproduce_anchor_and_both_paths_backpropagate():
    torch.manual_seed(73)
    model = DuraTimeVaryingRecovery(width=8, taps=5)
    x = torch.randn(3, 2, 256)
    anchor = torch.randn_like(x)
    innovation = torch.randn_like(x)
    variance = torch.rand(3)+.2
    output = model(x, anchor, innovation, variance)
    torch.testing.assert_close(output, anchor, rtol=0, atol=0)
    (output-x).square().mean().backward()
    assert torch.isfinite(model.wave.weight.grad).all()
    assert torch.isfinite(model.filter_head.weight.grad).all()
    assert torch.count_nonzero(model.filter_head.weight.grad)


def test_output_is_per_record_in_batched_inference():
    torch.manual_seed(11)
    model = DuraTimeVaryingRecovery(width=8, taps=5)
    torch.nn.init.normal_(model.filter_head.weight, std=.1)
    torch.nn.init.normal_(model.wave.weight, std=.1)
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.rand(3)+.2
    batched = model(x, anchor, innovation, variance)
    separate = torch.cat([model(x[i:i+1], anchor[i:i+1], innovation[i:i+1], variance[i:i+1]) for i in range(3)])
    torch.testing.assert_close(batched, separate, rtol=1e-5, atol=1e-6)
