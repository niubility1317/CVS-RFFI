import copy
import pytest
import torch
from experiments.pamf_dura_efficiency.exact_reference import exact_guide,feature_phi
from experiments.pamf_dura_efficiency.soft_reference_tangent import ALPHA


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.project=torch.nn.Linear(512,160)
    def features(self,x):return torch.tanh(self.project(x.flatten(1)))


def fixture():
    torch.manual_seed(123);reader=Reader().eval().requires_grad_(False)
    g=torch.nn.Linear(512,512);basis=torch.linalg.qr(torch.randn(160,10)).Q
    return reader,g,basis,torch.randn(3,2,256),torch.randn(3,2,256),torch.randn(3,10)*4


def test_injected_G_gradient_matches_weighted_exact_F_objective_with_one_F_reverse():
    reader,g,q,y,x,target=fixture();target=target*10;other=copy.deepcopy(g)
    r=g(y.flatten(1)).reshape_as(x);_,phi=feature_phi(reader,r,q)
    error=(phi-target).square().sum(1)
    raw=torch.autograd.grad((ALPHA*error).sum(),r,retain_graph=True)[0]
    IQnorm=2*(r.detach()-x).flatten(1).norm(dim=1)/(512*x.square().mean((1,2)).clamp_min(1e-4))
    weight=(IQnorm/raw.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp(max=1).detach()
    expected=ALPHA*error*weight
    ((r-x).square().mean((1,2))+expected).mean().backward()
    calls=[];hook=reader.project.register_full_backward_hook(lambda *args:calls.append(1))
    q=q.requires_grad_();target=target.requires_grad_();x=x.requires_grad_()
    restored=other(y.flatten(1)).reshape_as(x)
    value,info=exact_guide(reader,restored,x,target,q)
    torch.testing.assert_close(value.detach(),expected.detach())
    ((restored-x.detach()).square().mean((1,2))+value).mean().backward()
    hook.remove();assert len(calls)==1
    for a,b in zip(g.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
    assert x.grad is target.grad is q.grad is None
    assert all(p.grad is None for p in reader.parameters())
    assert (info['bounded_pressure']<=1+2e-5).all() and (info['weight']<1).any()


def test_clean_stationarity_and_weak_guide_are_preserved():
    reader,_,q,y,x,target=fixture();r=x.clone().requires_grad_()
    value,info=exact_guide(reader,r,x,target,q)
    grad=torch.autograd.grad(value.sum(),r)[0]
    assert not grad.any() and not value.any() and not info['weight'].any()
    r=y.clone().requires_grad_();_,phi=feature_phi(reader,r,q)
    close_target=phi.detach()+1e-5
    value,info=exact_guide(reader,r,x,close_target,q)
    actual=torch.autograd.grad(value.sum(),r)[0]
    assert torch.equal(info['weight'],torch.ones(3))
    torch.testing.assert_close(actual,info['raw_gradient'])


def test_frozen_eval_teacher_and_exact10_contract_are_required():
    reader,g,q,y,x,target=fixture();r=g(y.flatten(1)).reshape_as(x)
    with pytest.raises(ValueError):exact_guide(reader.train(),r,x,target,q)
    with pytest.raises(ValueError):exact_guide(reader.eval().requires_grad_(True),r,x,target,q)
    with pytest.raises(ValueError):exact_guide(reader.requires_grad_(False),r,x,target[:,:5],q[:,:5])
