import torch
from experiments.phase1_dura_orthogonal.iq_geometry import iq_geometry_loss


def test_geometry_zero_for_identical_source_and_finite_gradient():
    torch.manual_seed(4)
    clean = torch.randn(8, 2, 256)
    assert abs(iq_geometry_loss(clean, clean).item()) < 1e-7
    output = (clean + .4*torch.randn_like(clean)).requires_grad_()
    loss = iq_geometry_loss(output, clean)
    assert loss > 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(output.grad).all() and torch.count_nonzero(output.grad)


def test_geometry_is_invariant_to_consistent_batch_permutation():
    torch.manual_seed(5)
    clean = torch.randn(6, 2, 256)
    output = clean+.3*torch.randn_like(clean)
    perm = torch.tensor([4, 0, 5, 2, 1, 3])
    torch.testing.assert_close(iq_geometry_loss(output, clean), iq_geometry_loss(output[perm], clean[perm]))
