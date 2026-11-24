import torch

from experiments.pamf_lora_main.dura_identity_action import (
    IdentityAction, IdentityActionDura, solve_identity_map,
)
from experiments.pamf_lora_main.dura_prior_width import PacketDura
from experiments.pamf_uncertainty_restore.models import SCALE, weighted_map


def test_identity_cg_matches_closed_form_under_original_variance():
    rng = torch.Generator().manual_seed(220)
    observation = torch.randn(2, 256, dtype=torch.complex64, generator=rng)
    prior = torch.randn(2, 256, dtype=torch.complex64, generator=rng)
    initial = torch.randn(2, 256, dtype=torch.complex64, generator=rng)
    variance = torch.tensor([.004, .13])[:, None]
    result = weighted_map(IdentityAction(), observation, prior, variance, steps=2, initial=initial)
    expected = solve_identity_map(IdentityAction(), observation, prior, variance, steps=2, initial=initial)
    torch.testing.assert_close(result, expected, atol=1.2e-6, rtol=2e-6)


def test_exact_map_has_correct_variance_gradient_at_both_ends_of_learned_range():
    rng = torch.Generator().manual_seed(223)
    y, prior, target = (torch.randn(5, 256, dtype=torch.complex128, generator=rng) for _ in range(3))
    variance = torch.logspace(-8, 0, 5, dtype=torch.float64)[:, None].requires_grad_()
    result = solve_identity_map(IdentityAction(), y, prior, variance, steps=2, initial=y)
    residual = result - target
    derivative = SCALE**2 * (prior - y) / (SCALE**2 + variance).square()
    expected = 2 * (residual.detach().conj() * derivative.detach()).real.sum(-1, keepdim=True)
    actual, = torch.autograd.grad(residual.abs().square().sum(), variance)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-10)


def test_variance_only_matches_native_context_and_keeps_joint_gradients():
    torch.manual_seed(232)
    ordinary = PacketDura(wide=True)
    torch.manual_seed(232)
    identity = IdentityActionDura()
    for name, weight in identity.state_dict().items():
        torch.testing.assert_close(weight, ordinary.state_dict()[name], atol=0., rtol=0.)
    wave = 1 + .1 * torch.randn(1, 6, 256, dtype=torch.complex64)
    rotation = torch.ones_like(wave)
    native = torch.stack((wave.real, wave.imag), dim=-2).flatten(0, 1)
    estimator = identity.restorer.frozen.estimator
    with torch.no_grad():
        estimator.global_head.weight.normal_(std=.07)
        base = identity.restorer.frozen.coarse(native)
        _, native_variance = estimator(native, base)
        torch.testing.assert_close(estimator.variance_only(native, base), native_variance,
                                   atol=0., rtol=0.)
    expected = torch.randn_like(wave)
    optimizer = torch.optim.Adam(identity.parameters(), lr=3e-4)
    for _ in range(2):
        optimizer.zero_grad()
        coarse, restored = identity(wave, None, rotation)
        ((coarse - expected).abs().square().mean() + (restored - expected).abs().square().mean()).backward()
        optimizer.step()
    for layer in (identity.restorer.frozen.coarse, estimator.core.encoder, *identity.restorer.prior):
        gradients = [p.grad for p in layer.parameters() if p.grad is not None]
        assert gradients and all(g.isfinite().all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    head_gradient = estimator.global_head.weight.grad
    assert (head_gradient[:-1] == 0).all() and head_gradient[-1].abs().sum() > 0
    assert all(p.grad is None for module in (estimator.core.decode, estimator.core.out)
               for p in module.parameters())
