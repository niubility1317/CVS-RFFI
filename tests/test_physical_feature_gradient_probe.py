import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_phase3_restoration import physical_feature_gradient_probe as probe


def test_scene_role_selection_is_fixed_and_never_uses_labels():
    roles=np.full(63000,'X',dtype='<U3');scenes=np.full(63000,'unknown',dtype='<U24')
    cursor=0
    for scene in probe.SCENES:
        for role in ('L_s','U_s'):
            roles[cursor:cursor+3]=role;scenes[cursor:cursor+3]=scene;cursor+=3
    chosen=probe.select_rows(roles,scenes)
    assert len(chosen)==28 and len({x[0] for x in chosen})==28
    assert [i for i,_,_ in chosen[:4]]==[0,1,3,4]
    with pytest.raises(ValueError):probe.select_rows(roles[:-1],scenes[:-1])


class Physics(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.tensor(.9));self.log_noise=nn.Parameter(torch.tensor(-2.))
        self.register_buffer('prior_mean',torch.zeros(512));self.register_buffer('prior_covariance',torch.eye(512))
    def estimate(self,x):
        return dict(action_matrix=self.gain.double()*torch.eye(512,dtype=torch.double)[None].expand(len(x),-1,-1),
                    variance=self.log_noise.exp().expand(len(x)))


class Detail(nn.Module):
    def forward(self,base,context):return base+.1*context[:,:2]+.01*context[:,2:3]


class Model(nn.Module):
    def __init__(self):super().__init__();self.physical=Physics();self.detail=Detail()


def test_connected_posterior_preserves_forward_and_reaches_physical_heads():
    torch.set_num_threads(2);m=Model();x=torch.linspace(-1,1,512).reshape(1,2,256)
    with torch.no_grad(): reference,_,_=probe.connected_forward(m,x)
    result,_,residual=probe.connected_forward(m,x)
    torch.testing.assert_close(result,reference,rtol=0,atol=0)
    grads=torch.autograd.grad(result.square().mean(),(m.physical.gain,m.physical.log_noise))
    assert all(torch.isfinite(g) and g.abs()>0 for g in grads) and residual.max()<1e-8


def test_gradient_statistics_preserve_direction_and_report_zero():
    a=torch.tensor([1.,0.]);b=torch.tensor([-2.,0.]);s=probe.pair_statistics(a,b)
    assert s['ratio']==pytest.approx(2.) and s['cosine']==pytest.approx(-1.)
    zero=probe.pair_statistics(a,torch.zeros_like(a))
    assert zero['cosine'] is None and zero['ratio']==0 and zero['second_zero']


def test_original_estimator_uses_same_autograd_mode_as_connected_path():
    from experiments.pamf_phase3_restoration.covariance_detail import scratch_covariance_detail
    torch.set_num_threads(2)
    model=scratch_covariance_detail(2026092701).eval()
    x=torch.randn(1,2,256)
    reference=model(x).detach()
    output,_,residual=probe.connected_forward(model,x)
    torch.testing.assert_close(output,reference,rtol=2e-5,atol=2e-6)
    gradient=torch.autograd.grad(output.square().mean(),model.physical.coefficient_head.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0 and residual.max()<1e-8
