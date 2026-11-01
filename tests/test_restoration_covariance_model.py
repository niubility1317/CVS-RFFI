import torch
from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance,source_action_loss
from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior


def test_initial_posterior_and_strict_state_reload():
    g=scratch_covariance(9).eval();mean=torch.linspace(-.1,.2,512);cov=torch.diag(torch.linspace(.01,.4,512))
    g.bind_prior(mean,cov);x=torch.randn(2,2,256)
    with torch.no_grad():actual,aux=g(x,return_aux=True)
    expected,_=gaussian_posterior(torch.eye(512,dtype=torch.float64).expand(2,-1,-1),x.flatten(1).double(),mean.double(),cov.double(),torch.full((2,),.0005,dtype=torch.float64))
    torch.testing.assert_close(actual,expected.float().reshape(2,2,256),rtol=1e-5,atol=1e-6)
    other=scratch_covariance(2).eval();other.load_state_dict(g.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(other(x),actual,rtol=0,atol=0)
    assert aux['relative_solve_residual'].max()<1e-8


def test_operator_only_training_reaches_all_action_heads():
    g=scratch_covariance(9);x=torch.randn(2,2,256)
    aux=g.estimate(x)
    h=torch.zeros(2,2,64);h[:,0,0]=1;h[:,0,1]=.2
    c=torch.zeros(2,4,256);c[:,0]=.8;c[:,1]=.1;c[:,2]=.05
    loss,info=source_action_loss(aux,h,c,torch.tensor([.01,.02]))
    loss.mean().backward()
    for head in (g.kernel_head,g.coefficient_head,g.noise_head):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum()>0
    assert not g.prior_mean.requires_grad and not g.prior_covariance.requires_grad
    assert all(p.dtype==torch.float32 for p in g.state_dict().values())
