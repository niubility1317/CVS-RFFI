import torch

from experiments.phase1_dura_orthogonal.quartic_recovery import DuraQuarticRecovery, quartic_loss
from experiments.phase1_dura_orthogonal.source_quartic_closure import closure_features


def test_zero_initialization_and_low_noise_gate_keep_anchor_exact():
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(4, 2, 256, generator=generator)
    anchor = torch.randn(4, 2, 256, generator=generator)
    innovation = torch.randn(4, 2, 256, generator=generator)
    model = DuraQuarticRecovery(width=8)
    assert torch.equal(model(x, anchor, innovation, torch.ones(4)), anchor)
    with torch.no_grad():
        model.spectral_head.weight.fill_(.1)
        model.spectral_head.bias.fill_(.1)
    variance = anchor.square().mean((1, 2))*1e-8
    assert torch.equal(model(x, anchor, innovation, variance), anchor)


def test_quartic_loss_matches_probe_features_and_has_finite_gradient():
    generator = torch.Generator().manual_seed(32)
    clean = torch.randn(4, 2, 256, generator=generator)
    output = (clean + .15*torch.randn(clean.shape, generator=generator)).requires_grad_()
    expected = (closure_features(output)-closure_features(clean)).square().mean()
    loss = quartic_loss(output, clean)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert loss > 0 and torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0


def test_both_recovery_paths_receive_finite_gradients():
    generator = torch.Generator().manual_seed(33)
    x = torch.randn(2, 2, 256, generator=generator)
    anchor = torch.randn(2, 2, 256, generator=generator)
    innovation = torch.randn(2, 2, 256, generator=generator)
    target = torch.randn(2, 2, 256, generator=generator)
    model = DuraQuarticRecovery(width=8)
    output = model(x, anchor, innovation, torch.ones(2))
    ((output-target).square().mean() + .1*quartic_loss(output, target)).backward()
    for head in (model.base.wave, model.base.filter_head, model.spectral_head):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
