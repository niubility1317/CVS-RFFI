import torch
from experiments.phase1_dura_orthogonal.moment_relational import moment_features, moment_relational_loss


def test_moments_invariant_to_global_complex_gain():
    torch.manual_seed(1)
    x = torch.randn(4, 2, 256)
    z = torch.complex(x[:, 0], x[:, 1]) * (1.7+2.3j)
    transformed = torch.stack((z.real, z.imag), dim=1)
    torch.testing.assert_close(moment_features(x), moment_features(transformed), rtol=1e-5, atol=1e-6)


def test_relational_loss_zero_at_match_and_has_gradients():
    torch.manual_seed(3)
    clean = torch.randn(4, 2, 256)
    assert moment_relational_loss(clean, clean).item() == 0.
    out = (clean + .2*torch.randn_like(clean)).requires_grad_()
    loss = moment_relational_loss(out, clean)
    assert loss > 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(out.grad).all() and torch.count_nonzero(out.grad)
