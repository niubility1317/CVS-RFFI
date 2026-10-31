import pytest
import torch
from experiments.pamf_dura_efficiency.bounded_tangent import cap_metric,eligibility
from experiments.pamf_dura_efficiency.clean_tangent import tangent_loss


def test_cap_preserves_weak_metric_and_bounds_eigenvalues():
    jac=torch.zeros(2,5,2,256,dtype=torch.float64)
    for k,value in enumerate([100.,10.,1.,.1,.01]):jac[0,k].flatten()[k]=value
    jac[1]=jac[0]/1000
    energy=torch.tensor([256.,256.],dtype=torch.float64);weight=.0033130025035573727
    bounded,info=cap_metric(jac,energy,weight)
    raw=jac.flatten(2).transpose(1,2)@jac.flatten(2)
    actual=bounded.flatten(2).transpose(1,2)@bounded.flatten(2)
    eigen=torch.linalg.eigvalsh(actual)
    assert (weight*energy[:,None]*eigen).max()<=1+1e-10
    torch.testing.assert_close(actual[1],raw[1],atol=1e-12,rtol=1e-10)
    assert info['clipped_record'].tolist()==[True,False]


def test_loss_and_IQ_gradient_bound_for_arbitrary_errors():
    torch.manual_seed(15);clean=torch.randn(7,2,256);jac=torch.randn(7,5,2,256)*torch.logspace(-2,2,7)[:,None,None,None]
    energy=clean.square().sum((1,2));weight=.0033130025035573727;bounded,_=cap_metric(jac,energy,weight)
    restored=(clean+torch.randn_like(clean)*20).requires_grad_(True)
    guide=weight*tangent_loss(restored,clean,bounded);iq=(restored-clean).square().sum((1,2))/energy
    gradient=torch.autograd.grad(guide.sum(),restored)[0];iqgrad=2*(restored-clean)/energy[:,None,None]
    assert torch.all(guide<=iq*(1+2e-5))
    assert torch.all(gradient.flatten(1).norm(dim=1)<=iqgrad.flatten(1).norm(dim=1)*(1+2e-5))


def test_zero_and_invalid_cache_boundaries():
    zero=torch.zeros(2,5,2,256);bounded,info=cap_metric(zero,torch.ones(2),.1)
    assert torch.equal(bounded,zero) and not info['clipped_record'].any()
    with pytest.raises(ValueError):cap_metric(zero,torch.zeros(2),.1)
    with pytest.raises(ValueError):cap_metric(zero[:,:4],torch.ones(2),.1)
    with pytest.raises(ValueError):cap_metric(zero,torch.ones(2),0.)


def test_fixed_two_view_qualification_requires_pressure_and_alignment():
    def row():return dict(count=512,defined=512,raw_pressure_p95=2.,raw_mean_cosine=.65,bounded_mean_cosine=.64,
        bounded_nonnegative=400,rx_bounded_mean={str(rx):.4 for rx in [1,3,4,6,8]},maximum_bounded_pressure=.99)
    rows={'1':row(),'20':row()};assert eligibility(.2,1.000001,rows)['eligible']
    rows['20']['bounded_mean_cosine']=.60;assert not eligibility(.2,1.000001,rows)['eligible']
    rows['20']=row();rows['1']['raw_pressure_p95']=.9;assert not eligibility(.2,1.000001,rows)['eligible']
    rows['1']=row();assert not eligibility(.01,1.,rows)['eligible']
    assert not eligibility(.2,1.,rows)['automatic_training']
