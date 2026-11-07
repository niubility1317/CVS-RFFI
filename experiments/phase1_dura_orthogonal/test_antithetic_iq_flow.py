import pytest
import torch

from experiments.phase1_dura_orthogonal.antithetic_iq_flow import AntitheticIQFlow, perturbation_scale, stochastic_flow_loss


def test_initialization_preserves_dura_and_fixed_latent_is_reproducible():
    model = AntitheticIQFlow(width=4)
    again = AntitheticIQFlow(width=4)
    assert torch.equal(model.fixed_latent, again.fixed_latent)
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.tensor([0., .01, 10.])
    with torch.no_grad():
        assert torch.equal(model(x, anchor, innovation, variance), anchor)
        scale = perturbation_scale(anchor, variance)
    assert scale.shape == (3, 1, 1) and scale[0].item() == 0.
    assert (scale <= .25*anchor.square().mean((1, 2), keepdim=True).sqrt()+1e-7).all()


def test_stochastic_loss_trains_and_rejects_invalid_time():
    model = AntitheticIQFlow(width=4)
    x, anchor, innovation, clean, latent = (torch.randn(3, 2, 256) for _ in range(5))
    variance, t = torch.full((3,), .02), torch.tensor([0., .5, 1.])
    loss = stochastic_flow_loss(model, x, anchor, innovation, variance, clean, t, latent)
    loss.backward()
    assert torch.isfinite(loss) and model.out.weight.grad is not None
    assert torch.isfinite(model.out.weight.grad).all() and model.out.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='time'):
        stochastic_flow_loss(model, x, anchor, innovation, variance, clean, torch.tensor([0., 2., 1.]), latent)


def test_time_check_omission_preserves_valid_loss():
    model = AntitheticIQFlow(width=4)
    x, anchor, innovation, clean, latent = (torch.randn(3, 2, 256) for _ in range(5))
    variance, t = torch.full((3,), .02), torch.tensor([0., .5, 1.])
    full = stochastic_flow_loss(model, x, anchor, innovation, variance, clean, t, latent)
    omitted = stochastic_flow_loss(model, x, anchor, innovation, variance, clean, t, latent, validate_time=False)
    assert torch.equal(full, omitted)


def test_antithetic_output_is_batch_order_invariant_and_clean_gated():
    model = AntitheticIQFlow(width=4)
    with torch.no_grad():
        model.out.weight.normal_()
        model.out.bias.normal_()
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.tensor([0., .1, .2])
    order = torch.tensor([2, 0, 1])
    with torch.no_grad():
        result = model(x, anchor, innovation, variance)
        reordered = model(x[order], anchor[order], innovation[order], variance[order])
    assert torch.equal(result[0], anchor[0])
    assert torch.allclose(result[order], reordered, rtol=1e-6, atol=1e-7)
