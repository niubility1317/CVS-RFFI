import pytest
import torch
from experiments.pamf_dura_efficiency.gradient_components import split_gradients,local_steps
from experiments.pamf_dura_efficiency.exact_reference import feature_phi
from experiments.pamf_dura_efficiency.soft_reference_tangent import ALPHA


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(512,160)
    def features(self,x):return torch.tanh(self.layer(x.flatten(1)))


def test_head_aux_sum_matches_independent_full_objective_without_teacher_or_target_grads():
    torch.manual_seed(17);reader=Reader().eval().requires_grad_(False)
    iq=torch.randn(4,2,256,requires_grad=True);basis=torch.linalg.qr(torch.randn(160,10)).Q.requires_grad_()
    target=torch.randn(4,10,requires_grad=True)
    result=split_gradients(reader,iq,target,basis,retain_graph=True)
    _,phi=feature_phi(reader,iq,basis)
    expected=torch.autograd.grad((ALPHA*(phi-target.detach()).square().sum(1)).sum(),iq)[0]
    torch.testing.assert_close(result['head_gradient']+result['aux_gradient'],expected,atol=2e-8,rtol=2e-5)
    assert target.grad is basis.grad is None and all(p.grad is None for p in reader.parameters())


def test_local_steps_use_the_same_budget_and_bounded_gradients_including_cancellation():
    torch.manual_seed(19);x=torch.randn(4,2,256);r=x+.3*torch.randn_like(x)
    gh=torch.randn_like(x);ga=torch.randn_like(x)
    values=local_steps(r,x,gh,ga)
    IQ=values['IQ_gradient'];norm=IQ.flatten(1).norm(dim=1)
    for name in ['head','joint']:
        assert (values[name+'_bounded_gradient'].flatten(1).norm(dim=1)<=norm*(1+2e-6)).all()
        torch.testing.assert_close(values[name+'_step'].flatten(1).norm(dim=1),values['step_budget'])
        assert (values[name+'_step'].square().sum((1,2))/x.square().sum((1,2))<=.0001*(1+2e-6)).all()
    zero=local_steps(x,x,gh,ga)
    assert not zero['head_step'].any() and not zero['joint_step'].any()
    cancel=local_steps(r,x,-IQ,torch.zeros_like(IQ))
    assert not cancel['head_step'].any() and not cancel['joint_step'].any()


def test_requires_frozen_eval_teacher_and_original_source_shapes():
    reader=Reader().eval().requires_grad_(False);iq=torch.randn(2,2,256,requires_grad=True)
    basis=torch.eye(160)[:,:10];target=torch.zeros(2,10)
    with pytest.raises(ValueError):split_gradients(reader.train(),iq,target,basis)
    with pytest.raises(ValueError):split_gradients(reader.eval().requires_grad_(True),iq,target,basis)
    with pytest.raises(ValueError):split_gradients(reader.requires_grad_(False),iq,target[:,:5],basis[:,:5])
