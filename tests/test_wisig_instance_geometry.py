import torch

from experiments.pamf_repl_view_shuffle.instance_geometry import descriptor, relation_loss


def test_spectral_descriptor_invariance():
    generator = torch.Generator().manual_seed(31)
    signal = torch.randn(8, 256, dtype=torch.complex64, generator=generator)
    torch.testing.assert_close(descriptor(signal), descriptor((2. + 3.j) * signal.roll(13, -1)),
                               rtol=2e-4, atol=2e-5)


def test_relation_preserves_matches_and_supplies_gradients():
    generator = torch.Generator().manual_seed(31)
    reference = torch.randn(8, 256, dtype=torch.complex64, generator=generator)
    original = reference[:, None].expand(-1, 2, -1)
    restored = original.roll(1, 0).clone().requires_grad_()
    loss = relation_loss(restored, reference)
    assert loss > relation_loss(original, reference) + .5
    loss.backward()
    assert torch.isfinite(restored.grad).all()
    assert restored.grad.abs().sum() > 0
