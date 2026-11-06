import torch

from experiments.phase1_dura_orthogonal.source_nonlinear_residual import basis, fit_group, solve


def test_complex_residual_fit_uses_same_features_and_heldout_records():
    torch.manual_seed(79)
    x = torch.randn(10, 2, 256)
    a = torch.randn(10, 2, 256)
    features, _ = basis(x,a)
    coeff = torch.tensor([.2+.1j,-.1+.05j,.05-.1j,.15+.02j])
    target = (features[:,:,:4]*coeff).sum(-1)
    sel = torch.arange(10) < 7
    gram, rhs = fit_group(features,target,torch.ones(10),sel,4)
    fitted = solve(gram,rhs).squeeze(-1)
    heldout = (features[~sel,:,:4]*fitted).sum(-1)
    assert torch.mean((heldout-target[~sel]).abs().square()) < 1e-5
    assert torch.equal(features[:,:,1],features[:,:,0].conj())
    assert features.shape == (10,256,12)
