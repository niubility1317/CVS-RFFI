import pytest
import torch
from experiments.pamf_dura_efficiency.soft_reference_tangent import minimum_shift,prototype_tangent_loss,calibration_decision
from experiments.pamf_dura_efficiency.tangent_balance import balanced_tangent_loss


def test_minimum_shift_is_minimum_norm_and_reports_unreachable_direction():
    j=torch.zeros(2,5,2,256,dtype=torch.float64)
    for k in range(5):j[0,k,0,k]=k+1
    j[1,0,0,0]=2
    offset=torch.ones(2,5,dtype=torch.float64)
    shift,info=minimum_shift(j,offset)
    torch.testing.assert_close(shift[0,0,:5],1/torch.arange(1,6,dtype=torch.float64))
    assert torch.count_nonzero(shift[0,:,5:])==0 and info['rank'].tolist()==[5,1]
    torch.testing.assert_close(info['relative_residual'][0],torch.tensor(0.,dtype=torch.float64))
    assert info['relative_residual'][1]>.8


def test_offset_guidance_has_the_actual_capped_gradient_and_detaches_references():
    g=torch.Generator().manual_seed(61)
    x=torch.randn(3,2,256,generator=g,dtype=torch.float64,requires_grad=True)
    r=(x.detach()+.03*torch.randn(3,2,256,generator=g,dtype=torch.float64)).requires_grad_(True)
    j=torch.randn(3,5,2,256,generator=g,dtype=torch.float64,requires_grad=True)
    offset=torch.randn(3,5,generator=g,dtype=torch.float64,requires_grad=True);alpha=.0033130025035573727
    value,info=prototype_tangent_loss(r,x,j,offset,alpha);(alpha*value.sum()).backward()
    delta=r.detach()-x.detach();residual=(j.detach()*delta[:,None]).sum((2,3))-offset.detach()
    raw=2*alpha*(residual[:,:,None,None]*j.detach()).sum(1)
    torch.testing.assert_close(r.grad,raw*info['weight'][:,None,None])
    assert (r.grad.flatten(1).norm(dim=1)<=info['IQ_gradient_norm']*(1+1e-10)).all()
    assert x.grad is None and j.grad is None and offset.grad is None


def test_zero_offset_matches_previous_balanced_loss_and_clean_point_is_stationary():
    g=torch.Generator().manual_seed(62);x=torch.randn(4,2,256,generator=g)
    r=(x+.02*torch.randn(4,2,256,generator=g)).requires_grad_(True);j=torch.randn(4,5,2,256,generator=g);alpha=.0033130025035573727
    old,_=balanced_tangent_loss(r,x,j,alpha);new,_=prototype_tangent_loss(r,x,j,torch.zeros(4,5),alpha)
    torch.testing.assert_close(new,old)
    r=x.clone().requires_grad_(True);value,_=prototype_tangent_loss(r,x,j,torch.ones(4,5),alpha)
    grad=torch.autograd.grad(value.sum(),r)[0];assert torch.count_nonzero(grad)==0


def test_fixed_feasibility_and_both_cached_gradient_views_are_required():
    feasibility=dict(linear_residual_max=1e-4,shift_NMSE_mean=.0001,shift_NMSE_p95=.001,
        nonlinear_relative_error_median=.4,nonlinear_relative_error_p95=.9,choices_preserved=512)
    row=dict(count=512,defined=512,defined_IQ=512,mean_cosine=.6,mean_IQ_cosine=.1,nonnegative=400,
        RX_mean_cosine={str(r):.5 for r in [1,3,4,6,8]},maximum_balanced_pressure=1.)
    views={'1':dict(row),'20':dict(row)}
    assert calibration_decision(feasibility,views)['eligible']
    feasibility['choices_preserved']=511
    assert not calibration_decision(feasibility,views)['eligible']
    with pytest.raises(ValueError):calibration_decision(feasibility,{'1':dict(row)})
