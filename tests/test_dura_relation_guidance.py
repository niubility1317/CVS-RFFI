import pytest
import torch
from experiments.pamf_dura_efficiency.relation_guidance import relation_error,bounded_relation,CAP


def test_cross_receiver_pair_loss_matches_explicit_distances_and_zero_reference():
    torch.manual_seed(101);s=torch.randn(3,160,requires_grad=True);t=torch.randn(3,160,requires_grad=True);rx=torch.tensor([1,1,3])
    error,info=relation_error(s,t,rx);u=s/s.norm(dim=1,keepdim=True);v=t.detach()/t.detach().norm(dim=1,keepdim=True)
    for i in range(3):
        peers=[j for j in range(3) if rx[j]!=rx[i]]
        wanted=torch.stack([1-(v[i]*v[j]).sum() for j in peers]);actual=torch.stack([1-(u[i]*u[j]).sum() for j in peers])
        torch.testing.assert_close(error[i],(actual-wanted).square().mean()/wanted.square().mean().clamp_min(1e-4))
    assert info['peers'].tolist()==[1,1,2]
    gradients=torch.autograd.grad(error.sum(),[s,t],allow_unused=True);assert gradients[1] is None and gradients[0].abs().sum()>0
    same,_=relation_error(t.detach(),t.detach(),rx);torch.testing.assert_close(same,torch.zeros(3),atol=0,rtol=0)


def test_relations_allow_shared_rotations_and_empty_cross_receiver_graph_is_explicit():
    torch.manual_seed(102);t=torch.randn(3,160);q,_=torch.linalg.qr(torch.randn(160,160));rx=torch.tensor([1,3,4])
    e,_=relation_error(t@q,t,rx);assert e.max()<1e-12
    s=torch.randn(3,160,requires_grad=True);e,info=relation_error(s,t,torch.ones(3,dtype=torch.int64))
    assert not info['active'].any() and info['peers'].sum()==0 and e.sum()==0
    assert torch.autograd.grad(e.sum(),s)[0].abs().sum()==0
    with pytest.raises(ValueError):relation_error(s,t,rx.float())


def test_coupled_gradient_uses_both_endpoints_and_actual_IQ_cap_without_target_gradients():
    torch.manual_seed(103);r=torch.randn(3,2,256,requires_grad=True);clean=torch.randn_like(r);target=torch.randn(3,160,requires_grad=True)
    signature=r.flatten(1)[:,:160];error,_=relation_error(signature,target,torch.tensor([1,3,4]))
    raw=torch.autograd.grad(error.sum(),r,retain_graph=True)[0]
    guide,info=bounded_relation(r,clean,error);actual=torch.autograd.grad(guide.sum(),r)[0]
    torch.testing.assert_close(actual,raw*info['weight'][:,None,None],atol=1e-7,rtol=2e-6)
    iqnorm=2*(r.detach()-clean).flatten(1).norm(dim=1)/(512*clean.square().mean((1,2)).clamp_min(1e-4))
    assert torch.all(actual.flatten(1).norm(dim=1)<=CAP*iqnorm*(1+2e-5)) and target.grad is None
    z=torch.randn(3,2,256,requires_grad=True);e,_=relation_error(z.flatten(1)[:,:160],target,torch.tensor([1,3,4]))
    g,_=bounded_relation(z,z.detach(),e);assert torch.autograd.grad(g.sum(),z)[0].abs().sum()==0
