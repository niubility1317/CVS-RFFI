import torch
from experiments.phase1_dura_orthogonal.adaptive_inverse import widely_linear_filter
from experiments.phase1_dura_orthogonal.timevarying_inverse import timevarying_filter, TimeVaryingInverseRecovery


def test_constant_timevarying_taps_match_stationary_filter():
    torch.manual_seed(8)
    x = torch.randn(3, 2, 32)
    taps = torch.randn(3, 4, 5)
    dynamic = taps.unsqueeze(-1).expand(-1, -1, -1, 32)
    torch.testing.assert_close(timevarying_filter(x, dynamic), widely_linear_filter(x, taps), rtol=1e-5, atol=1e-6)


def test_new_model_starts_as_received_and_both_heads_get_gradients():
    torch.manual_seed(9)
    model = TimeVaryingInverseRecovery(width=8, taps=5)
    x = torch.randn(2, 2, 256)
    output = model(x)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    (output-x.roll(2, -1)).square().mean().backward()
    assert torch.isfinite(model.filter_head.weight.grad).all()
    assert torch.count_nonzero(model.filter_head.weight.grad)
    assert torch.isfinite(model.wave.weight.grad).all()
