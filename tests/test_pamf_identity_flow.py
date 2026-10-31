import pytest
import torch

from experiments.pamf_identity_flow.model import (
    IdentityConditionalFlow, flow_path, source_flow_loss, frozen_identity_consistency,
)


def test_flow_path_endpoints_and_constant_velocity():
    torch.manual_seed(4)
    x,y,noise = [torch.randn(2,2,256) for _ in range(3)]
    state,target = flow_path(x,y,torch.zeros(2),noise,.02)
    torch.testing.assert_close(state,y+.02*noise)
    torch.testing.assert_close(state+target,x)
    state,_ = flow_path(x,y,torch.ones(2),noise,.02)
    torch.testing.assert_close(state,x)
    model=IdentityConditionalFlow(width=8,layers=1,heads=2).eval()
    with torch.no_grad():
        model.output.bias.fill_(1.)
    torch.testing.assert_close(model.integrate(y,torch.zeros_like(y),steps=4),y+1,rtol=1e-5,atol=2e-6)


def test_ensemble_uses_explicit_draws_and_preserves_batch_independence():
    model=IdentityConditionalFlow(width=8,layers=1,heads=2).eval()
    y=torch.randn(2,2,256)
    noises=torch.randn(3,2,2,256)
    mean,var=model.ensemble(y,noises,steps=2)
    expected=y+.02*noises.mean(0)
    torch.testing.assert_close(mean,expected,rtol=1e-5,atol=2e-6)
    torch.testing.assert_close(var,(.02*noises).var(0,unbiased=False),rtol=1e-5,atol=2e-6)
    one=model.integrate(y[:1],noises[0,:1],steps=2)
    batch=model.integrate(y,noises[0],steps=2)[:1]
    torch.testing.assert_close(one,batch,rtol=1e-5,atol=2e-6)
    with torch.no_grad():torch.nn.init.normal_(model.output.weight,std=.01)
    one=model.integrate(y[:1],noises[0,:1],steps=2)
    batch=model.integrate(y,noises[0],steps=2)[:1]
    torch.testing.assert_close(one,batch,rtol=1e-5,atol=2e-6)
    with pytest.raises(ValueError):model.ensemble(y,noises[:1],steps=2)


def test_flow_and_identity_gradients_keep_teacher_frozen():
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(512,12,bias=False)
        def features(self,x):return self.linear(x.flatten(1))
    teacher=Reader().eval().requires_grad_(False)
    model=IdentityConditionalFlow(width=8,layers=1,heads=2)
    clean,received,noise=[torch.randn(2,2,256) for _ in range(3)]
    t=torch.tensor([.2,.7])
    flow,endpoint=source_flow_loss(model,clean,received,t,noise)
    anchor,agreement=frozen_identity_consistency(teacher,clean,endpoint,endpoint+.01*noise)
    (flow+anchor+agreement).mean().backward()
    assert torch.isfinite(flow).all() and model.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    optimizer.step();optimizer.zero_grad()
    next_loss,_=source_flow_loss(model,clean,received,t,noise)
    next_loss.mean().backward()
    assert model.input.weight.grad is not None and model.input.weight.grad.abs().sum()>0
    teacher.train()
    with pytest.raises(ValueError):frozen_identity_consistency(teacher,clean,endpoint,endpoint)
