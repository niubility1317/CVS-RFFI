"""Full-source frontend telemetry must describe actual frozen forwards."""
import copy
import pytest
import torch
from experiments.cvs_frontfilter_identity.model import build,VARIANTS
from experiments.cvs_frontfilter_identity.source import (frontfilter_forward_snapshot,accumulate_frontfilter_groups,finalize_frontfilter_groups)

@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_once_snapshot_has_no_state_update(variant):
    torch.set_num_threads(2);torch.manual_seed(5)
    model=build(variant).eval();x=torch.randn(3,2,256)
    with torch.no_grad():
        if variant.endswith('static'):model.frontfilter.coeff_raw.fill_(.02)
        else:model.frontfilter.context[-1].bias.fill_(.02)
        expected=model.features(x);gx_expected,coeff_expected=model.frontfilter(x)
    state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    hooks=len(model.frontfilter._forward_hooks)
    emb,gx,coeff,kernel=frontfilter_forward_snapshot(model,x)
    assert torch.equal(emb,expected) and torch.equal(gx,gx_expected) and torch.equal(coeff,coeff_expected)
    assert kernel.shape==(3,2,5) and len(model.frontfilter._forward_hooks)==hooks
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())

def test_population_moments_and_zero_input_are_counted():
    x=torch.ones(3,2,4);x[0].zero_();gx=x.clone();gx[1]*=1.1;gx[2]*=.9
    coefficients=torch.zeros(3,2,4);coefficients[:,0,0]=torch.tensor([0.,.1,.2])
    kernel=torch.zeros(3,2,5);kernel[:,0,0]=.1
    batch=dict(label=torch.tensor([0,0,1]),receiver=torch.tensor([1,1,3]),day=torch.tensor([1,1,2]))
    groups={};accumulate_frontfilter_groups(groups,batch,x,gx,coefficients,kernel)
    records=finalize_frontfilter_groups(groups,expected_count=3,expected_groups=2)
    a,b=records
    assert a['count']==2 and a['input_norm_ratio_eligible_count']==1
    assert a['relative_input_change_mean']==pytest.approx(.05,abs=1e-6)
    assert a['input_norm_ratio_min']==pytest.approx(1.1)
    assert a['coefficient_mean'][0]==pytest.approx(.05)
    assert a['coefficient_trace_variance']==pytest.approx(.0025)
    assert b['coefficient_trace_variance']==pytest.approx(0.,abs=1e-12)
    with pytest.raises(ValueError):finalize_frontfilter_groups(groups)

def test_snapshot_rejects_bypass_and_removes_hook_on_error():
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__();self.frontfilter=torch.nn.Identity()
        def features(self,x):return x
    model=Broken()
    with pytest.raises(ValueError):frontfilter_forward_snapshot(model,torch.ones(1,2,256))
    assert len(model.frontfilter._forward_hooks)==0
    def fails(x):raise RuntimeError('testfailure')
    model.features=fails
    with pytest.raises(RuntimeError):frontfilter_forward_snapshot(model,torch.ones(1,2,256))
    assert len(model.frontfilter._forward_hooks)==0

def test_nonfinite_telemetry_is_rejected():
    x=torch.ones(1,2,4);gx=x.clone();gx[0,0,0]=float('nan')
    batch=dict(label=torch.tensor([0]),receiver=torch.tensor([1]),day=torch.tensor([1]))
    with pytest.raises(FloatingPointError):accumulate_frontfilter_groups({},batch,x,gx,torch.zeros(1,2,4),torch.zeros(1,2,5))
