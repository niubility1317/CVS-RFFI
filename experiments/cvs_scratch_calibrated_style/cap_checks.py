"""CPU-only bounded joint-cap correctness and native objective checks."""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from .joint_cap import apply_joint_cap,parameter_groups,GROUPS

CAP=dict(enabled=True,rho=.2,eps=1e-12)


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Module()
        self.base.time_fuse=nn.Linear(2,1,bias=False)
        self.base.t2=nn.Linear(2,1,bias=False)
        self.base.low_hz_=nn.Parameter(torch.tensor([2.],dtype=torch.float64))
        self.base.complex_filter=nn.Parameter(torch.tensor([1.+2.j,2.-1.j],dtype=torch.complex128))
        self.base.unused=nn.Parameter(torch.ones(3,dtype=torch.float64))
        self.base.frozen=nn.Parameter(torch.ones(1),requires_grad=False)
        self.double()


def _norm(loss,parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return math.sqrt(sum(float(v.detach().abs().double().square().sum()) for v in values if v is not None))


def toy_checks():
    torch.manual_seed(202611);m=Toy();named,grouped=parameter_groups(m)
    assert {id(p) for _,p in named}=={id(p) for p in m.base.parameters() if p.requires_grad}
    assert sum(len(v) for v in grouped.values())==len(named)
    assert any(n=='low_hz_' for n,p in grouped['rest']) and any(p.is_complex() for n,p in grouped['rest'])
    ps=tuple(p for _,p in named)
    identity=sum(p.abs().square().sum() for p in ps)
    regularizer=10*m.base.time_fuse.weight.square().sum()+20*m.base.t2.weight.square().sum()+5*m.base.complex_filter.abs().square().sum()+30*m.base.low_hz_.square().sum()
    state={n:p.clone() for n,p in m.named_parameters()};rng=torch.random.get_rng_state().clone()
    m.base.low_hz_.grad=torch.ones_like(m.base.low_hz_)
    effective,detail=apply_joint_cap(m,identity,regularizer,CAP)
    assert detail['gradient_evaluations']==2 and 0<detail['scalar']<1
    for group,rows in grouped.items():
        p=tuple(v for _,v in rows)
        measured=_norm(effective,p);reference=_norm(identity,p)
        assert measured<=.2*reference+1e-10
        assert math.isclose(measured,detail['groups'][group]['capped_R'],rel_tol=1e-10,abs_tol=1e-12)
        assert math.isclose(_norm(regularizer,p),detail['groups'][group]['raw_R'],rel_tol=1e-10,abs_tol=1e-12)
    actual=torch.autograd.grad(effective,ps,retain_graph=True,allow_unused=True)
    original=torch.autograd.grad(regularizer,ps,retain_graph=True,allow_unused=True)
    for a,b in zip(actual,original):
        if b is None:assert a is None
        else:torch.testing.assert_close(a,b*detail['scalar'])
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(p,state[n]) for n,p in m.named_parameters())
    assert torch.equal(m.base.low_hz_.grad,torch.ones_like(m.base.low_hz_))
    assert all(p.grad is None for n,p in m.named_parameters() if n!='base.low_hz_')
    # Zero R => s=1; zero identity with nonzero R => s=0.
    _,zero_r=apply_joint_cap(m,identity,regularizer*0,CAP)
    assert zero_r['scalar']==1 and zero_r['all_regularizer_groups_zero']
    zero,zero_i=apply_joint_cap(m,identity*0,regularizer,CAP)
    assert zero_i['scalar']==0 and float(zero.detach())==0
    # Constant losses may legitimately have no autograd graph.
    _,constant=apply_joint_cap(m,torch.tensor(0.),torch.tensor(0.),CAP)
    assert constant['scalar']==1 and constant['gradient_evaluations']==0
    with patch('torch.autograd.grad',side_effect=AssertionError('Disabled cap differentiated')):
        same,disabled=apply_joint_cap(m,identity,regularizer,dict(CAP,enabled=False))
    assert same is regularizer and disabled['gradient_evaluations']==0 and disabled['seconds']==0
    # The cap sees the summed layers, including cancellation/cross terms.
    r1=100*m.base.time_fuse.weight.sum();r2=-99*m.base.time_fuse.weight.sum()
    _,combined=apply_joint_cap(m,identity,r1+r2,CAP)
    assert math.isclose(combined['groups']['shallow']['raw_R'],math.sqrt(2),rel_tol=1e-7)
    assert combined['groups']['shallow']['raw_R']<_norm(r1,tuple(p for _,p in grouped['shallow']))
    for bad_id,bad_r in ((identity*float('nan'),regularizer),(identity,regularizer*float('inf'))):
        try:apply_joint_cap(m,bad_id,bad_r,CAP)
        except FloatingPointError:pass
        else:raise AssertionError('Nonfinite loss accepted')
    # Finite forward with nonfinite derivative is independently rejected.
    with torch.no_grad():m.base.low_hz_.zero_()
    identity=sum(p.abs().square().sum() for p in ps)
    singular=m.base.low_hz_.sqrt().sum()
    try:apply_joint_cap(m,identity,singular,CAP)
    except FloatingPointError:pass
    else:raise AssertionError('Nonfinite gradient accepted')
    return dict(complex_norms_and_Hz_covered=True,complete_disjoint_groups=True,detached_scalar_gradient_identity=True,
        joint_layer_cancellation_measured=True,zero_cases=True,nonfinite_rejected=True,
        disabled_no_autograd=True,parameters_grads_RNG_unchanged=True,detail=detail)


def native_checks():
    from . import design as d
    from .model import build,groups
    from .objective import objective,BASE_WEIGHTED_KEYS
    from experiments.cvs_e20_boundary_style.objective import objective as previous_objective
    from experiments.cvs_e20_direction_audit.checks import _fixture
    c=d.config(next(r for r in d.rows() if r['arm']=='CAP_M_BOTH'))
    m=build(c,'cpu');groups(m,c);m.multilevel.fixed_coefficients={k:1. for k in c['multilevel']['levels']}
    L=_fixture();L['x']=torch.randn(360,2,256,generator=torch.Generator().manual_seed(887700))
    li=torch.arange(32);m.train()
    frozen={k:v.clone() for k,v in m.state_dict().items()}
    parameters=tuple(p for p in m.base.parameters() if p.requires_grad)
    # Disabled control is exactly the previous objective for identical inputs/RNG.
    disabled=dict(c,multilevel=dict(c['multilevel'],joint_gradient_cap=dict(CAP,enabled=False)))
    torch.manual_seed(991)
    old_total,old_parts,old_weighted,old_detail=previous_objective(disabled,m,None,L,li,None,None,448,'cpu')
    old_grad=torch.autograd.grad(old_total,parameters,allow_unused=True)
    old_values={k:float(v.detach()) for k,v in old_weighted.items()}
    m.load_state_dict(frozen);torch.manual_seed(991)
    total,parts,weighted,detail=objective(disabled,m,None,L,li,None,None,448,'cpu')
    new_grad=torch.autograd.grad(total,parameters,allow_unused=True)
    assert old_values=={k:float(v.detach()) for k,v in weighted.items()}
    for a,b in zip(old_grad,new_grad):
        if a is None:assert b is None
        else:torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert detail['joint_gradient_cap']['gradient_evaluations']==0
    # Enabled cap uses actual combined weighted Fishr; no parameter update.
    m.load_state_dict(frozen);torch.manual_seed(991)
    total,parts,weighted,detail=objective(c,m,None,L,li,None,None,448,'cpu')
    cap=detail['joint_gradient_cap'];assert cap['enabled'] and cap['gradient_evaluations']==2
    assert detail['fishr_budget']['scope']=='raw_pre_joint_cap'
    torch.testing.assert_close(total,sum(weighted[k] for k in BASE_WEIGHTED_KEYS))
    torch.testing.assert_close(weighted['fishr'].detach(),torch.tensor(cap['effective_weighted_fishr']))
    identity=sum(weighted[k] for k in BASE_WEIGHTED_KEYS if k!='fishr')
    _,partition=parameter_groups(m)
    for name,rows in partition.items():
        p=tuple(v for _,v in rows)
        assert _norm(weighted['fishr'],p)<=.2*_norm(identity,p)+1e-6
    grads=torch.autograd.grad(total,parameters,allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert any(g is not None and g.abs().sum()>0 for g in grads)
    assert all(p.grad is None for p in m.parameters())
    # Warmup and nonauxiliary steps must never invoke the cap implementation.
    with patch('experiments.cvs_scratch_calibrated_style.objective.apply_joint_cap',side_effect=AssertionError('Unexpected cap')):
        for step in (444,445):
            _,_,_,inactive=objective(c,m,None,L,li,None,None,step,'cpu')
            assert inactive['joint_gradient_cap']['gradient_evaluations']==0
    # JOINT: exercise nonzero cached-peer KL and selected labeled style CE.
    joint_c=d.config(next(r for r in d.rows() if r['arm']=='CAP_JOINT'))
    joint=build(joint_c,'cpu');groups(joint,joint_c);joint.train()
    joint.multilevel.fixed_coefficients={k:1. for k in joint_c['multilevel']['levels']}
    U=dict(x=L['x'][:16].clone(),rx=L['rx'][:16].clone(),day=L['day'][:16].clone(),ids=list(range(90000,90016)))
    bank=dict(ids=U['ids'],teacher_seeds=joint_c['teacher']['seeds'],q=torch.randn(16,6).softmax(1),
        peer_disagreement=torch.ones(16,dtype=torch.bool),peer_JS=torch.full((16,),.1))
    with patch('experiments.cvs_scratch_calibrated_style.objective.apply_joint_cap',wraps=apply_joint_cap) as applied:
        jt,jp,jw,jd=objective(joint_c,joint,bank,L,torch.arange(32),U,torch.arange(16),1332,'cpu')
    assert float(jw['KL'].detach())>0 and float(jw['style_ce'].detach())>0
    assert applied.call_count==1
    actual_id,actual_raw=applied.call_args.args[1:3]
    torch.testing.assert_close(actual_id,sum(jw[key] for key in BASE_WEIGHTED_KEYS if key!='fishr'))
    torch.testing.assert_close(jw['KL_same']+jw['KL_disagree'],jw['KL'])
    torch.testing.assert_close(jt,sum(jw[key] for key in BASE_WEIGHTED_KEYS))
    jps=tuple(p for p in joint.base.parameters() if p.requires_grad)
    for scaled,raw in zip(torch.autograd.grad(jw['fishr'],jps,retain_graph=True,allow_unused=True),
                          torch.autograd.grad(actual_raw,jps,retain_graph=True,allow_unused=True)):
        if raw is None:assert scaled is None
        else:
            # FP32 mixed second derivatives round differently when the scalar
            # enters backward before reductions rather than after them.
            expected=raw*jd['joint_gradient_cap']['scalar']
            assert float((scaled-expected).norm())<=2e-5*float(expected.norm())+5e-7
    assert all(g is None or torch.isfinite(g).all() for g in torch.autograd.grad(jt,jps,allow_unused=True))
    return dict(raw_disabled_loss_and_gradients_exact=True,actual_native_backward_finite=True,
        complete_identity_group_bound=True,no_warmup_or_nonaux_cap=True,
        audit_KL_not_double_counted=True,nonzero_JOINT_KL_and_style_in_identity=True,
        actual_cap=cap,joint_cap=jd['joint_gradient_cap'])


def run():
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    # Existing shared helpers fork all available CUDA generators even on CPU;
    # suppress that availability path so this bounded check creates no GPU context.
    with patch('torch.cuda.is_available',return_value=False):
        result=dict(status='PASS',device='cpu',target_access=False,GPU_used=False,toy=toy_checks(),native=native_checks())
    assert not torch.cuda.is_initialized()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('toy','native')}))
