import torch

from experiments.phase1_dura_orthogonal.monotone_envelope import (
    DuraMonotoneEnvelope, envelope_loss, monotone_factor)


def test_zero_initialized_recovery_is_exact_dura_anchor():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(4, 2, 256, generator=generator)
    anchor = torch.randn(4, 2, 256, generator=generator)
    innovation = torch.randn(4, 2, 256, generator=generator)
    model = DuraMonotoneEnvelope(width=8)
    assert torch.equal(model(x, anchor, innovation, torch.ones(4)), anchor)


def test_monotone_radial_mapping_preserves_order_and_is_positive():
    radius = torch.linspace(.001, 6., 256).reshape(1, 1, -1)
    logits = torch.tensor([[-1.4, .4, -.5, .2, .9, -1., 1.1, .3]])
    mapped = radius * monotone_factor(radius, logits)
    assert torch.all(mapped > 0) and torch.all(mapped[..., 1:] > mapped[..., :-1])
    assert torch.equal(monotone_factor(radius, torch.zeros_like(logits)), torch.ones_like(radius))


def test_envelope_loss_is_phase_invariant_and_differentiable():
    generator = torch.Generator().manual_seed(42)
    clean = torch.randn(4, 2, 256, generator=generator)
    rotated = torch.stack((-clean[:, 1], clean[:, 0]), 1)
    assert envelope_loss(rotated, clean).item() < 1e-12
    output = (clean + .2*torch.randn(clean.shape, generator=generator)).requires_grad_()
    loss = envelope_loss(output, clean)
    loss.backward()
    assert loss > 0 and torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0
