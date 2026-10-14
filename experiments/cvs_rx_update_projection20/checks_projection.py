"""Closed-form/KKT projection checks; no data, model, or optimizer updates."""
import json
import random

import numpy as np
import torch

from .projection import project, SOURCE_RX


def _gradients(rows, *, dtype=torch.float64):
    result={rx:(None,None) for rx in SOURCE_RX}
    for rx,row in rows.items():
        result[rx]=(torch.tensor([row[0]],dtype=dtype),torch.tensor([row[1]],dtype=dtype))
    return result


def _delta(x,y, *, dtype=torch.float64):
    return (torch.tensor([x],dtype=dtype),torch.tensor([y],dtype=dtype))


def _flat(result): return torch.cat(result['projected_delta']).double()


def _raises(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Expected malformed/nonfinite source gradient rejection')


def check():
    # Projection of(2,3) onto x+y<=0 is(-.5,.5), by removing the
    # outward unit normal component. This is an independent closed form.
    d0=_delta(2.,3.);g=_gradients({1:(1.,1.)})
    half=project(d0,g)
    assert torch.allclose(_flat(half),torch.tensor([-.5,.5],dtype=torch.float64),atol=2e-14,rtol=0)
    assert abs(half['summary']['objective']-6.25)<2e-13
    # Two orthogonal outward normals project to the negative-quadrant vertex.
    quadrant=project(d0,_gradients({1:(1.,0.),3:(0.,1.)}))
    assert _flat(quadrant).abs().max()<2e-14 and quadrant['summary']['retained_norm_ratio']<1e-14
    # Opposite normals on both axes constrain the feasible cone to exactly0.
    conflict=project(_delta(2.,-3.),_gradients({1:(1.,0.),3:(-1.,0.),4:(0.,1.),6:(0.,-1.)}))
    assert _flat(conflict).abs().max()<2e-14
    # Duplicate/scaled rows define the same halfspace, including singular Gram
    # active sets. No row strength becomes a source-specific weight.
    related=project(d0,_gradients({1:(1.,1.),3:(2.,2.),4:(3.,3.)}))
    assert torch.allclose(_flat(related),_flat(half),atol=2e-14,rtol=0)
    assert related['summary']['rank_deficient_active_sets']>0
    # Near-parallel constraints: x<=0 dominates x+1e-5*y<=0 at this point.
    near=project(_delta(1.,-1.),_gradients({1:(1.,0.),3:(1.,1e-5)}))
    assert torch.allclose(_flat(near),torch.tensor([0.,-1.],dtype=torch.float64),atol=2e-13,rtol=0)
    # Nearly opposite rows constrain y<=-|x|/1e-5; (0,1) projects to0.
    near_opposite=project(_delta(0.,1.),_gradients({1:(1.,1e-5),3:(-1.,1e-5)}))
    assert _flat(near_opposite).abs().max()<2e-12
    # Positive rescaling of risk rows does not change the geometric constraints.
    scaled=project(d0,_gradients({1:(1e-30,1e-30),3:(1e20,1e20)}))
    assert torch.allclose(_flat(scaled),_flat(half),atol=2e-14,rtol=0)
    # Feasible increments return the original tuple tensors without rounding.
    feasible_d0=_delta(-2.,3.);feasible=project(feasible_d0,_gradients({1:(1.,0.)}))
    assert feasible['projected_delta'] is feasible_d0 and feasible['summary']['status']=='already_feasible'
    zeros=project(_delta(0.,0.),_gradients({1:(1.,0.)}))
    assert zeros['summary']['original_norm']==0 and zeros['summary']['retained_norm_ratio']==1.
    unconstrained=project(d0,_gradients({}))
    assert unconstrained['projected_delta'] is d0 and unconstrained['summary']['nonzero_constraints']==0
    # Independently evaluate KKT on the returned delta and reported normalized
    # duals rather than rebuilding the enumerator in the test.
    for original,rows,result in ((d0,g,half),(d0,_gradients({1:(1.,0.),3:(0.,1.)}),quadrant),
                                 (_delta(0.,1.),_gradients({1:(1.,1e-5),3:(-1.,1e-5)}),near_opposite)):
        original_vector=torch.cat(original);delta=_flat(result);stationarity=delta-original_vector
        for rx,values in rows.items():
            gradient=torch.tensor([0. if value is None else float(value) for value in values],dtype=torch.float64)
            norm=gradient.norm();dual=result['summary']['by_RX'][str(rx)]['normalized_dual']
            assert dual>=0
            if norm:
                directional=float(gradient@delta)
                assert directional<=result['summary']['by_RX'][str(rx)]['FP64_raw_risk_tolerance']
                assert abs(dual*float((gradient/norm)@delta))<2e-10
                stationarity+=dual*gradient/norm
        assert stationarity.norm()<2e-11
        assert result['summary']['active_sets_enumerated']<=32
        json.dumps(result['summary'],allow_nan=False)
    # The FP32-cast residual is recomputed independently from the actual output
    # tensors; it does not claim to measure theta_after-theta_before.
    fp32_d0=_delta(2.,3.,dtype=torch.float32);fp32_g=_gradients({1:(1.,1.4)},dtype=torch.float32)
    fp32=project(fp32_d0,fp32_g);row=torch.cat(fp32_g[1]).double()
    assert all(out.dtype==old.dtype and out.device==old.device for out,old in zip(fp32['projected_delta'],fp32_d0))
    assert abs(float(row@_flat(fp32))-fp32['summary']['by_RX']['1']['cast_directional_risk'])<1e-15
    assert fp32['summary']['actual_parameter_addition_not_measured']
    # Tiny AdamW increments retain an equally tiny numeric tolerance; no
    # max(1, parameter magnitude) can admit an outward raw update.
    tiny=project(_delta(2e-20,3e-20),_gradients({1:(1.,1.)}))
    assert torch.allclose(_flat(tiny),_flat(half)*1e-20,atol=2e-33,rtol=0)
    assert tiny['summary']['normalized_risk_tolerance']<1e-30
    input_values=[*d0,*[v for values in g.values() for v in values if v is not None]]
    before=[(v.clone(),v._version,v.grad) for v in input_values]
    rng=torch.get_rng_state().clone();python_rng=random.getstate();numpy_rng=np.random.get_state()
    project(d0,g)
    assert all(torch.equal(v,b[0]) and v._version==b[1] and v.grad is b[2] for v,b in zip(input_values,before))
    assert torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate()
    current=np.random.get_state();assert current[0]==numpy_rng[0] and np.array_equal(current[1],numpy_rng[1]) and current[2:]==numpy_rng[2:]
    _raises(lambda:project((),g))
    _raises(lambda:project(d0,{1:g[1]}))
    _raises(lambda:project(d0,dict(g,query=g[1])))
    wrong=dict(g);wrong[1]=(torch.zeros(2),torch.zeros(1));_raises(lambda:project(d0,wrong))
    nonfinite=dict(g);nonfinite[1]=(torch.tensor([float('nan')]),None);_raises(lambda:project(d0,nonfinite))
    _raises(lambda:project(_delta(float('inf'),0.),g))
    _raises(lambda:project(d0,g,role='U_s'))
    return dict(status='PASS',torch=torch.__version__,device='cpu',closed_form_halfspace=True,
        orthogonal_and_conflicting_cones=True,zero_related_near_related_gradients=True,
        positive_row_scaling_invariant=True,KKT_independently_checked=True,
        feasible_input_exactly_preserved=True,FP32_cast_risk_separately_measured=True,
        tiny_delta_tolerance_scaled=True,inputs_and_RNG_unchanged=True,negative_cases=True,
        projection_norm=half['summary']['projection_norm'],retained_norm_ratio=half['summary']['retained_norm_ratio'],
        data_reads=0,optimizer_updates=0,
        boundary='Pure numeric projection only; no provenance, actual parameter-addition or efficacy claim')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
