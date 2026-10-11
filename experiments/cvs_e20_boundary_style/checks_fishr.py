"""Bounded margin-sensitivity derivative and native second-order checks."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from torch.nn import functional as F
from .multilevel import BoundarySensitivity, normalize_scores, identity_margin, calibrate, fixed_fishr, LEVELS
from experiments.cvs_e20_budget_fishr.multilevel import _parameter_groups, _norm
from experiments.cvs_e20_direction_audit.checks import Model, _fixture, _config


def _rng():
    return [torch.random.get_rng_state().clone()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _state(model):
    return ({k:v.clone() for k,v in model.state_dict().items()}, [m.training for m in model.modules()],
            [None if p.grad is None else p.grad.clone() for p in model.parameters()], _rng())


def _unchanged(model, state):
    weights,modes,gradients,random=state
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
    assert modes==[m.training for m in model.modules()]
    assert all((p.grad is None and v is None) or (v is not None and torch.equal(p.grad,v)) for p,v in zip(model.parameters(),gradients))
    assert all(torch.equal(a,b) for a,b in zip(random,_rng()))


def _derivative_checks(device):
    torch.manual_seed(1277)
    model=Model(2026092701).to(device).double()
    model.multilevel=BoundarySensitivity(model.base,2026092701);model.eval()
    x=torch.randn(7,2,64,device=device,dtype=torch.float64);y=torch.arange(7,device=device)%6
    plain=model(x)
    with model.multilevel.capture() as (probes,shapes):
        identity=model(x)
    assert torch.equal(plain,identity)
    margin=identity_margin(identity,y)
    leaves=tuple(v for level in LEVELS for v in probes[level])
    grads=torch.autograd.grad(margin.sum(),leaves,retain_graph=True,create_graph=True)
    errors={};h=1e-4
    for i,level in enumerate(LEVELS):
        for j,name in enumerate(('a','b')):
            displacement=torch.zeros_like(probes[level][j]);displacement[0,1,0]=h
            offset=(displacement,0.) if j==0 else (0.,displacement)
            negative=tuple(-v if torch.is_tensor(v) else v for v in offset)
            with model.multilevel.capture({level:offset}):plus=identity_margin(model(x),y).sum()
            with model.multilevel.capture({level:negative}):minus=identity_margin(model(x),y).sum()
            estimate=(plus-minus)/(2*h);exact=grads[2*i+j][0,1,0]
            torch.testing.assert_close(exact,estimate,rtol=2e-4,atol=1e-8)
            errors[level+'.'+name]=float((estimate-exact).detach().abs())
    # Each virtual coordinate is packet-local in eval. A sum of margins must
    # equal the sample's own derivative, not an unintended cross-batch gradient.
    own=torch.autograd.grad(margin[0],probes['time_down'][0],retain_graph=True)[0]
    torch.testing.assert_close(own[0],grads[0][0],rtol=1e-9,atol=1e-12)
    assert torch.count_nonzero(own[1:])==0
    zero=torch.zeros(4,12,device=device,dtype=torch.float64,requires_grad=True)
    normalized,stats=normalize_scores(zero)
    assert torch.equal(normalized,zero) and stats['low_norm_rate']==1.
    derivative=torch.autograd.grad(normalized.square().sum(),zero)[0]
    assert torch.isfinite(derivative).all() and torch.count_nonzero(derivative)==0
    s=torch.randn_like(zero,requires_grad=True)
    score,_=normalize_scores(s);score2,_=normalize_scores(s*10)
    torch.testing.assert_close(score,score2,rtol=1e-9,atol=1e-9)
    return errors


def _exercise(model,L,c,device,do_calibration):
    model.multilevel=BoundarySensitivity(model.base,c['model_seed']).to(device)
    assert list(model.multilevel.parameters())==[] and model.multilevel.state_dict()=={}
    assert not hasattr(model.multilevel,'heads')
    model.eval()
    with torch.no_grad():
        clean=model(L['x'][:12].to(device))
        with model.multilevel.capture():
            identity=model(L['x'][:12].to(device))
    assert torch.equal(clean,identity)
    model.train();initial=_state(model);hooks=sum(len(m._forward_hooks) for m in model.modules())
    group=_parameter_groups(model);routing={}
    for levels in ([LEVELS[0]],[LEVELS[1]],list(LEVELS)):
        mc=dict(c['multilevel'],statistic='margin',sensitivity_eps=1e-6,fishr_levels=levels)
        ce,terms,detail=model.multilevel.loss(model,L,448,device,mc)
        assert float(ce.detach())==0 and _norm(ce,tuple(model.base.parameters()))==0
        assert set(terms)==set(levels) and detail['auxiliary_heads']==0
        assert detail['aux_forwards']==2 and detail['auxiliary_packets']==120
        row={}
        for level,term in terms.items():
            norms={key:_norm(term,ps) for key,ps in group.items()}
            assert norms[level]>0 and all(torch.isfinite(torch.tensor(v)) for v in norms.values())
            row[level]=norms
            assert detail['levels'][level]['sensitivity_A']['score_requires_grad']
            assert detail['levels'][level]['sensitivity_B']['score_requires_grad']
        routing['+'.join(levels)]=row
        _unchanged(model,initial)
    # A subsequent ordinary optimizer backward must still traverse the live
    # second derivatives after eval flags have been restored by the helper.
    main_ce=F.cross_entropy(model(L['x'][:30].to(device)),L['y'][:30].to(device))
    params=tuple(p for p in model.base.parameters() if p.requires_grad)
    g=torch.autograd.grad(main_ce+sum(terms.values())/2,params,allow_unused=True)
    assert any(v is not None and bool(v.abs().sum()>0) for v in g)
    # That ordinary train forward may update native BN buffers; re-establish
    # a baseline before subsequent pure diagnostic checks.
    initial=_state(model)
    for step in (449,450,451):
        ce,empty,detail=model.multilevel.loss(model,L,step,device,mc)
        assert not empty and detail['aux_forwards']==0
    model.eval();ce,empty,detail=model.multilevel.loss(model,L,448,device,mc)
    assert not empty and detail['aux_forwards']==0
    model.train()
    if do_calibration:
        initial=_state(model)
        with TemporaryDirectory() as directory:
            result=calibrate(c,model,L,device,directory)
            try:calibrate(c,model,L,device,directory)
            except FileExistsError:pass
            else:raise AssertionError('Calibration overwrite accepted')
        assert result['calibration_reference']=='own_margin_term' and result['pairs']==12
        assert result['model_forwards']==24 and result['extra_source_readouts']==1440
        _unchanged(model,initial)
    else:
        model.multilevel.fixed_coefficients={k:1. for k in LEVELS};result=None
    initial=_state(model)
    _,terms,_,ce=model.multilevel._pair(model,L,452,device,c['multilevel'],calibration=True)
    loss,detail=fixed_fishr(model,ce,terms,c['multilevel'],.3)
    expected=sum(v*model.multilevel.fixed_coefficients[k]*.3 for k,v in terms.items())/2
    torch.testing.assert_close(loss,expected)
    _norm(loss,group['time_down']);_unchanged(model,initial)
    assert sum(len(m._forward_hooks) for m in model.modules())==hooks
    assert list(model.multilevel.parameters())==[] and model.multilevel.state_dict()=={}
    return dict(routing=routing,calibration=result,fixed_budget=detail,native_affine_identity_exact=True)


def run(device='cpu'):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    errors=_derivative_checks(device)
    seed=2026092701;c=_config(seed)
    c['multilevel'].update(statistic='margin',sensitivity_eps=1e-6)
    toy=_exercise(Model(seed).to(device),_fixture(),c,device,True)
    from experiments.cvs_early_geometry import design as native_design
    from experiments.cvs_early_geometry.model import build
    native=build(native_design.config(next(r for r in native_design.rows() if r['arm']=='PHYS')),device)
    L=_fixture();L['x']=torch.randn(360,2,256,generator=torch.Generator().manual_seed(88971))
    native_result=_exercise(native,L,c,device,True)
    return dict(status='PASS',device=device,target_access=False,finite_difference_errors=errors,
        affine_identity_exact=True,zero_parameter_no_head=True,zero_score_stable=True,
        first_derivative_packet_local=True,second_derivative_identity_path=True,
        pair_eval_state_modes_RNG_gradients_unchanged=True,inactive_no_forward=True,
        post_context_backward_valid=True,own_term_calibration=True,source_days=[1,2,3],
        toy=toy,native=native_result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('toy','native')},allow_nan=False))
