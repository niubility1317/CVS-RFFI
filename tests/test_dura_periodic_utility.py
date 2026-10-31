import pytest
import torch
from experiments.pamf_dura_efficiency.periodic_utility import periodic_projection,counterfactuals,decision


def test_projection_is_dense_orthogonal_and_keeps_exact_window():
    torch.manual_seed(1);x=torch.randn(3,2,256,dtype=torch.float64)
    p=periodic_projection(x);torch.testing.assert_close(periodic_projection(p),p)
    dense=torch.zeros(80,80,dtype=torch.float64)
    for i in range(80):
        for j in range(80):dense[i,j]=.25 if i%20==j%20 else 0
    torch.testing.assert_close(p[:,:,80:160],x[:,:,80:160]@dense.T)
    r=x[:,:,80:160]-p[:,:,80:160]
    torch.testing.assert_close((p[:,:,80:160]*r).sum(-1),torch.zeros(3,2,dtype=torch.float64),atol=2e-14,rtol=0)
    assert not p[:,:,:80].count_nonzero() and not p[:,:,160:].count_nonzero()


def test_paired_correction_and_global_control_have_equal_energy():
    torch.manual_seed(2);a=torch.randn(4,2,256,dtype=torch.float64);x=torch.randn_like(a)
    cases,d=counterfactuals(a,x)
    torch.testing.assert_close(d['periodic_delta']+d['residual_delta'],d['window_error'])
    torch.testing.assert_close((cases['residual']-a).square().sum((1,2)),(cases['matched_global']-a).square().sum((1,2)))
    for name in ['periodic','residual']:
        assert torch.equal(cases[name][:,:,:80],a[:,:,:80]) and torch.equal(cases[name][:,:,160:],a[:,:,160:])
    unchanged,_=counterfactuals(a,a)
    assert all(torch.equal(v,a) for v in unchanged.values())
    with pytest.raises(ValueError):counterfactuals(a[:,:,:128],x[:,:,:128])


def test_both_views_extra_utility_and_receiver_protection_are_required():
    def groups(extra=2,rxloss=False):
        base=dict(teacher_correct=100,feature_cosine_error=.1)
        row=dict(count=512,cases=dict(native=base,residual=dict(teacher_correct=104,feature_cosine_error=.09),matched_global=dict(teacher_correct=104-extra),periodic=base))
        values={'all':row}
        for r in [1,3,4,6,8]:values['rx:'+str(r)]=dict(cases=dict(native=dict(teacher_correct=10),residual=dict(teacher_correct=9 if rxloss and r==8 else 10)))
        for scene in ['clean','s1','s2','s3','s4','s5','s6']:values['scene:'+scene]=dict(cases=dict(native=dict(teacher_correct=10),residual=dict(teacher_correct=10)))
        return values
    assert decision({'1':groups(),'20':groups()})['eligible']
    assert not decision({'1':groups(),'20':groups(extra=1)})['eligible']
    assert not decision({'1':groups(),'20':groups(rxloss=True)})['eligible']
