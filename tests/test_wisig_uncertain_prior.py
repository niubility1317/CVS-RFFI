import torch

from experiments.pamf_unrolled_restore.wisig_uncertain_prior import (
    UncertaintyPrior, fit_loss, perturb, received_features,
)


def test_shared_perturbation_and_joint_update():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(71)
    source = torch.randn((8, 256), dtype=torch.complex64, generator=rng)
    equal = torch.stack((source, source), dim=1)
    paired = perturb(equal, rng)
    torch.testing.assert_close(paired[:, 0], paired[:, 1], rtol=0, atol=0)
    model = UncertaintyPrior()
    value, metrics = fit_loss(model, source, rng, preserve=.20)
    value.backward()
    assert torch.isfinite(metrics).all()
    assert model.spectral[-1].weight.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.detail.parameters())


def test_coherent_anchor_uses_relative_frequency_without_mutating_input():
    rng = torch.Generator().manual_seed(12)
    cycle = torch.randn((4, 20), dtype=torch.complex64, generator=rng).repeat(1, 13)[:, :256]
    time = torch.arange(256)
    y = cycle * torch.exp(1j * .052 * time)
    variance, confidence, _ = received_features(y)
    assert variance.max() < 1e-10
    assert confidence.min() > .9999
    _, result, _ = UncertaintyPrior().eval()(y)
    torch.testing.assert_close(result, y, rtol=0, atol=1e-6)
