"""Public/disposable math and model checks; no formal data or weights."""
import math
import torch
import pytest
from torch.nn import functional as F
from experiments.cvs_phase_curvature_identity.model import build,VARIANTS,PATHS,curvature_contract,curvature_delta,curvature_transform
from experiments.cvs_adaptive_volterra_identity.model import build as control_build

torch.set_num_threads(2)

def rotate(z,phase):
    r,i=z.unbind(1);c=phase.cos();s=phase.sin()
    while c.ndim<r.ndim:c=c.unsqueeze(0);s=s.unsqueeze(0)
    return torch.stack((r*c-i*s,r*s+i*c),1)

@pytest.mark.parametrize('variant',VARIANTS)
def test_initial_state_and_full_function_match_control(variant):
    torch.manual_seed(77);control=control_build('adaptive_volterra_lag4')
    torch.manual_seed(77);model=build(variant)
    assert sum(p.numel() for p in model.parameters())==202561
    original=control.state_dict();actual=model.state_dict()
    assert all(torch.equal(value,actual[key]) for key,value in original.items())
    assert set(actual)-set(original)=={'core.'+path+'.conv.memory_raw' for path in PATHS}
    x=torch.randn(5,2,256)
    for training in (True,False):
        control.train(training);model.train(training)
        torch.manual_seed(182);expected=control(x)
        torch.manual_seed(182);observed=model(x)
        assert torch.equal(observed,expected)
    for raw in model.memory_parameters():assert raw.item()==0.

@pytest.mark.parametrize('pair',((1,4),(2,4)))
def test_affine_phase_covariance_and_causal_batch_independence(pair):
    torch.manual_seed(2);x=torch.randn(3,2,5,40,dtype=torch.float64)
    phase=torch.arange(40,dtype=torch.float64)*.16+.51;raw=torch.tensor(.3,dtype=torch.float64)
    result=curvature_transform(x,raw,*pair)
    assert torch.allclose(curvature_transform(rotate(x,phase),raw,*pair),rotate(result,phase),atol=2e-14,rtol=2e-14)
    assert torch.equal(result,torch.cat([curvature_transform(row[None],raw,*pair) for row in x]))
    other=x.clone();other[...,25:]+=torch.randn_like(other[...,25:])
    assert torch.equal(curvature_transform(other,raw,*pair)[...,:25],result[...,:25])
    assert torch.equal(result[...,:sum(pair)],x[...,:sum(pair)])

@pytest.mark.parametrize('pair',((1,4),(2,4)))
@pytest.mark.parametrize('amplitude',(.2,1.,5.))
def test_constant_amplitude_tone_is_null_including_clipping_and_prefix(pair,amplitude):
    phase=torch.arange(32,dtype=torch.float64)*.27-.8
    z=torch.stack((phase.cos(),phase.sin()),0)[None,:,None]*amplitude
    assert curvature_delta(z,*pair).abs().max()<2e-14

@pytest.mark.parametrize('pair',((1,4),(2,4)))
def test_quadratic_phase_matches_closed_form(pair):
    a,b=pair;time=torch.arange(40,dtype=torch.float64);kappa=.017
    phase=.23+.1*time+kappa*time.square();z=torch.stack((phase.cos(),phase.sin()),0)[None,:,None]
    observed=curvature_delta(z,a,b)
    expected=(rotate(z,z.new_full((40,),-2*kappa*a*b))-z)/4
    assert torch.allclose(observed[...,a+b:],expected[...,a+b:],atol=2e-14,rtol=2e-14)
    assert observed[...,a+b:].abs().max()>.01

@pytest.mark.parametrize('pair',((1,4),(2,4)))
def test_delta_bound_degenerate_input_and_real_backward(pair):
    torch.manual_seed(9);z=(torch.randn(3,2,5,40)*100).requires_grad_();raw=torch.tensor(.2,requires_grad=True)
    delta=curvature_delta(z,*pair)
    assert delta.square().sum(1).sqrt().max()<=4.+1e-6
    result=curvature_transform(z,raw,*pair);result.square().mean().backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(raw.grad) and raw.grad!=0
    for scale in (0.,1e-9):
        small=(torch.randn(3,2,5,40)*scale).requires_grad_();parameter=torch.tensor(.2,requires_grad=True)
        output=curvature_transform(small,parameter,*pair);output.sum().backward()
        assert torch.isfinite(output).all() and torch.isfinite(small.grad).all() and torch.isfinite(parameter.grad)

@pytest.mark.parametrize('variant',VARIANTS)
def test_all_six_gates_have_actual_CE_gradients_updates_and_diagnostics(variant):
    torch.manual_seed(19);model=build(variant);x=torch.randn(8,2,256);labels=torch.arange(8)%6
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    model.train();loss=F.cross_entropy(model(x),labels);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(raw.grad.abs()>0 for raw in model.memory_parameters())
    optimizer.step();assert all(raw.item()!=0 for raw in model.memory_parameters())
    model.eval();before={key:value.clone() for key,value in model.state_dict().items()}
    diagnostic=model.diagnostics(x);records=diagnostic['feature_curvature']['records']
    assert [row['block'] for row in records]==list(PATHS)
    assert all(row['input_formula_max_abs_error']==row['prefix_correction_max_abs_error']==0 for row in records)
    assert all(row['actual_relative_output_change_mean']>0 for row in records)
    assert diagnostic['normalization']['blocks']==6 and len(diagnostic['adaptive_input']['records'])==1
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    actual=model.contract();assert actual['feature_curvature_active'] and actual['new_trainable_parameters']==6
    assert actual['actual_feature_curvature_parameter_shapes']==[[]]*6
    assert actual==curvature_contract(variant)

@pytest.mark.parametrize('variant',VARIANTS)
def test_nonzero_gates_retain_whole_constant_phase_property(variant):
    torch.manual_seed(31);model=build(variant).eval()
    with torch.no_grad():
        for i,raw in enumerate(model.memory_parameters()):raw.fill_(.07*(i+1))
        x=torch.randn(4,2,256);phase=x.new_full((256,),.72)
        assert torch.allclose(model(rotate(x,phase)),model(x),atol=1e-4,rtol=1e-4)

def test_unregistered_delays_shape_and_variant_rejected():
    with pytest.raises(ValueError):curvature_contract('other')
    with pytest.raises(ValueError):curvature_delta(torch.randn(2,2,3,20),1,3)
    with pytest.raises(ValueError):curvature_delta(torch.randn(2,2,20),1,4)
    with pytest.raises(ValueError):curvature_transform(torch.randn(2,2,3,20),torch.zeros(2),1,4)
