import pytest
import torch

from experiments.phase1_residual_diffusion.model import (ResidualDiffusion, alpha, forward_noise,
    reverse_step, STEPS)


def test_forward_and_reverse_share_the_same_noise_trajectory():
    torch.manual_seed(17)
    clean = torch.randn(3, 2, 256)
    noise = torch.randn_like(clean)
    t = torch.ones(3)
    value = forward_noise(clean, t, noise)
    for i in range(STEPS, 0, -1):
        t = torch.full((3,), i/STEPS)
        following = torch.full_like(t, (i-1)/STEPS)
        value = reverse_step(value, clean, t, following)
        torch.testing.assert_close(value, forward_noise(clean, following, noise), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(value, clean, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(alpha(torch.zeros(1)), torch.ones(1))


def test_zero_head_returns_anchor_and_fixed_latent_is_batch_independent():
    torch.manual_seed(12)
    model = ResidualDiffusion(width=16).eval()
    received = torch.randn(2, 2, 256)
    anchor = torch.randn_like(received)
    innovation = torch.randn_like(received)
    variance = torch.tensor([.01, .05])
    with torch.no_grad():
        out = model(received, anchor, innovation, variance)
        first = model(received[:1], anchor[:1], innovation[:1], variance[:1])
    assert torch.equal(out, anchor)
    assert torch.equal(first, out[:1])


def test_nonzero_model_is_deterministic_and_record_independent():
    torch.manual_seed(42)
    model = ResidualDiffusion(width=16).eval()
    torch.nn.init.normal_(model.output.weight, std=.001)
    received = torch.randn(2, 2, 256)
    anchor, innovation = torch.randn_like(received), torch.randn_like(received)
    variance = torch.tensor([.01, .05])
    with torch.no_grad():
        out = model(received, anchor, innovation, variance)
        again = model(received, anchor, innovation, variance)
        first = model(received[:1], anchor[:1], innovation[:1], variance[:1])
    assert torch.equal(out, again)
    torch.testing.assert_close(out[:1], first, atol=2e-5, rtol=1e-5)
    assert torch.isfinite(out).all()


def test_incorrect_diffusion_checkpoint_rejected(monkeypatch, tmp_path):
    from experiments.phase1_residual_diffusion import ground
    bad = dict(schema='unrelated', status='SOURCE_TRAINED')
    monkeypatch.setattr(ground.core, 'read', lambda *a: bad)
    monkeypatch.setattr(ground.torch, 'load', lambda *a, **k: bad)
    with pytest.raises(ValueError, match='checkpoint'):
        ground.load_fixed(tmp_path, 'source', 'fixed', 'cpu')


def test_gpu_noise_objective_has_finite_nonzero_gradient():
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU required')
    torch.manual_seed(16)
    model = ResidualDiffusion(width=192).cuda()
    y = torch.randn((2, 2, 256), device='cuda')
    anchor = torch.randn_like(y)
    innovation = torch.randn_like(y)
    variance = torch.full((2,), .05, device='cuda')
    noise = torch.randn_like(y)
    t = torch.tensor([.001, .999], device='cuda')
    clean = torch.randn_like(y)*.2
    noisy = forward_noise(clean, t, noise)
    prediction = model.estimate(y, anchor, innovation, variance, noisy, t)
    (prediction-clean).square().mean().backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()
    assert model.output.weight.grad.abs().sum() > 0
