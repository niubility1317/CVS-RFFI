import torch
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral, deterministic_linear_2x


def test_deterministic_upsample_matches_linear_forward_and_gradient():
    for length in (1, 2, 17):
        torch.manual_seed(length)
        x = torch.randn(2, 3, length, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        expected = torch.nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        actual = deterministic_linear_2x(y)
        torch.testing.assert_close(actual, expected, rtol=1e-15, atol=1e-15)
        weight = torch.randn_like(actual)
        expected_grad, = torch.autograd.grad((expected*weight).sum(), x)
        actual_grad, = torch.autograd.grad((actual*weight).sum(), y)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-15, atol=1e-15)


def test_zero_heads_preserve_dura_and_backpropagate():
    torch.manual_seed(9)
    model = DuraMultiscaleSpectral(width=8, taps=5)
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = anchor.square().mean((1, 2))*1e-2
    output = model(x, anchor, innovation, variance)
    torch.testing.assert_close(output, anchor, rtol=0, atol=0)
    (output-x).square().mean().backward()
    assert torch.isfinite(model.wave.weight.grad).all()
    assert torch.isfinite(model.filter_head.weight.grad).all()
    assert torch.count_nonzero(model.filter_head.weight.grad)


def test_low_noise_gate_and_global_spectral_parameters():
    torch.manual_seed(11)
    model = DuraMultiscaleSpectral(width=8, taps=5)
    torch.nn.init.normal_(model.wave.weight, std=.2)
    x, anchor, innovation = (torch.randn(2, 2, 256) for _ in range(3))
    variance = anchor.square().mean((1, 2))*torch.tensor([1e-5, 1e-2])
    out = model(x, anchor, innovation, variance)
    torch.testing.assert_close(out[0], anchor[0], rtol=0, atol=0)
    assert not torch.equal(out[1], anchor[1])
    out.square().mean().backward()
    assert model.spectral_real.grad is not None and torch.isfinite(model.spectral_real.grad).all()
    assert torch.count_nonzero(model.spectral_real.grad)


def test_spectral_refinement_is_per_record():
    torch.manual_seed(13)
    model = DuraMultiscaleSpectral(width=8, taps=5)
    torch.nn.init.normal_(model.wave.weight, std=.1)
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = anchor.square().mean((1, 2))*1e-2
    batch = model(x, anchor, innovation, variance)
    separate = torch.cat([model(x[i:i+1], anchor[i:i+1], innovation[i:i+1], variance[i:i+1]) for i in range(3)])
    torch.testing.assert_close(batch, separate, rtol=1e-5, atol=1e-6)
