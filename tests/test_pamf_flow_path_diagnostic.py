import pytest
import torch
from torch import nn
from tools.diagnose_pamf_flow_path import diagnose


class Prior(nn.Module):
    def field(self,point,condition,time):return condition+.2*point


class Model(nn.Module):
    def __init__(self):super().__init__();self.prior=Prior()
    def context(self,received):return torch.zeros_like(received),None,None,received
    def forward(self,received,noise):
        point=torch.zeros_like(received)
        for k in range(4):point=point+self.prior.field(point,received,received.new_full((len(received),),k/4))/4
        return point


def test_same_batch_teacher_path_and_free_running_are_distinguished():
    model=Model().eval();r=torch.full((3,2,4),.25);x=torch.ones_like(r);noise=torch.ones_like(r)
    out=diagnose(model,x,r,noise,sigma=0.,steps=4,scale=1.)
    assert len(out['teacher_path'])==4 and len(out['free_running_steps'])==4
    assert out['teacher_path'][0]['endpoint_nmse']['mean']==pytest.approx(.5625)
    assert out['teacher_path'][-1]['endpoint_nmse']['mean']==pytest.approx(.0225,abs=1e-7)
    assert out['free_running_prior_nmse']['mean']>.5
    assert out['teacher_path_uses_clean_endpoint'] and not out['free_running_uses_clean']


def test_clean_diagnostics_do_not_change_actual_model_output_or_rng():
    model=Model().eval();r=torch.full((2,2,4),.25);noise=torch.ones_like(r)
    expected=model(r,noise);state=torch.get_rng_state().clone()
    a=diagnose(model,torch.ones_like(r),r,noise,sigma=0.,steps=4,scale=1.)
    b=diagnose(model,torch.ones_like(r)*2,r,noise,sigma=0.,steps=4,scale=1.)
    assert a['full_restored_mean'] == b['full_restored_mean'] == float(expected.mean())
    torch.testing.assert_close(model(r,noise),expected);assert torch.equal(state,torch.get_rng_state())
    assert model.prior.field.__func__ is Prior.field


def test_training_mode_or_incompatible_inputs_rejected():
    r=torch.ones((2,2,4));noise=torch.ones_like(r)
    with pytest.raises(ValueError):diagnose(Model(),r,r,noise,sigma=0.,steps=4,scale=1.)
    with pytest.raises(ValueError):diagnose(Model().eval(),r,r[:1],noise,sigma=0.,steps=4,scale=1.)
    with pytest.raises(ValueError):diagnose(Model().eval(),r,r,noise,sigma=0.,steps=0,scale=1.)
