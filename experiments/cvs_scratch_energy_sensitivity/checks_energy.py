"""Synthetic derivative, calibration lifecycle and persisted-state regression."""
import argparse,json,math
import contextlib,io
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from .multilevel import BoundarySensitivity,LEVELS,identity_margin
from .calibration import begin_calibration,advance_calibration,require_calibration
from experiments.cvs_e20_direction_audit.checks import Model,_fixture
from experiments.cvs_e20_boundary_style.checks_fishr import _state,_unchanged


def rejected(call):
    try:call()
    except (ValueError,FileExistsError):return
    raise AssertionError('Invalid state accepted')


def derivative_checks(device,objective):
    torch.manual_seed(1632)
    model=Model(d.SEEDS[0]).to(device).double().eval()
    model.multilevel=BoundarySensitivity(model.base,d.SEEDS[0],objective)
    x=torch.randn(7,2,64,dtype=torch.float64,device=device);y=torch.arange(7,device=device)%6
    logits,raw,_=model.multilevel.raw_sensitivities(model,x,y)
    h=1e-4;errors={}
    def loss():
        z=model(x)
        return F.cross_entropy(z,y,reduction='sum') if objective=='ce' else identity_margin(z,y).sum()
    for level in LEVELS:
        for j,label in enumerate(('a','b')):
            delta=torch.zeros(7,4,1,dtype=torch.float64,device=device);delta[0,1,0]=h
            offset=(delta,0.) if j==0 else (0.,delta)
            with model.multilevel.capture({level:offset}):plus=loss()
            with model.multilevel.capture({level:tuple(-v for v in offset)}):minus=loss()
            numerical=(plus-minus)/(2*h);exact=raw[level][0,j*4+1]
            torch.testing.assert_close(exact,numerical,rtol=2e-4,atol=1e-9)
            errors[level+'.'+label]=float((exact-numerical).abs().detach())
    # Live CE confidence is required also in the mixed derivative.
    margin=BoundarySensitivity(model.base,d.SEEDS[0],'margin')
    _,gm,_=margin.raw_sensitivities(model,x,y)
    if objective=='ce':
        py=logits.softmax(1).gather(1,y[:,None])
        for level in LEVELS:torch.testing.assert_close(raw[level],-(1-py)*gm[level],rtol=1e-9,atol=1e-12)
    model.multilevel.fixed_scales={k:.037 for k in LEVELS}
    _,scores,_=model.multilevel.sensitivities(model,x,y)
    for level in LEVELS:
        torch.testing.assert_close(scores[level],raw[level]/.037,rtol=1e-9,atol=1e-12)
        rms=scores[level].square().mean(1).sqrt()
        assert float((rms.max()-rms.min()).detach())>1e-7
    weight=model.base.id_backbone.cls_head.weight
    # Downstream classifier perturbation keeps detached feature statistics fixed.
    energy=sum(s.square().sum() for s in scores.values())
    exact=torch.autograd.grad(energy,weight)[0][0,1]
    original=float(weight[0,1].detach())
    def energy_at(value):
        with torch.no_grad():weight[0,1]=value
        _,s,_=model.multilevel.sensitivities(model,x,y)
        return sum(v.square().sum() for v in s.values()).detach()
    numerical=(energy_at(original+h)-energy_at(original-h))/(2*h)
    with torch.no_grad():weight[0,1]=original
    torch.testing.assert_close(exact,numerical,rtol=2e-4,atol=1e-8)
    errors['mixed_second']=float((exact-numerical).abs())
    return errors


def lifecycle(device,arm):
    torch.manual_seed(991)
    c=d.config(next(r for r in d.rows() if r['arm']==arm));seed=c['model_seed']
    model=Model(seed).to(device);model.multilevel=BoundarySensitivity(model.base,seed,c['multilevel']['sensitivity_objective'])
    model.train();L=_fixture();initial=_state(model);calls=[];pairs=[]
    hook=model.register_forward_pre_hook(lambda m,args:calls.append(len(args[0])))
    original_pair=model.multilevel.pair_indices
    def pair(*args,**kwargs):
        result=original_pair(*args,**kwargs);pairs.append((result[0].tolist(),result[1].tolist()));return result
    model.multilevel.pair_indices=pair
    with TemporaryDirectory() as directory:
        current=begin_calibration(c,model,L,device,directory)
        assert current['status']=='PENDING_CALIBRATION' and not calls
        current,changed=advance_calibration(c,model,L,device,directory,current,443)
        assert not changed and not calls
        require_calibration(c,model,current,444)
        rejected(lambda:require_calibration(c,model,current,448))
        current,changed=advance_calibration(c,model,L,device,directory,current,444)
        assert changed and len(calls)==48 and sum(calls)==2880
        assert current['model_forwards']==48 and current['extra_source_readouts']==2880
        # Coefficient _pair calls the deterministic sampler twice: actual + telemetry.
        assert pairs[:12]==pairs[12::2]==pairs[13::2]
        _unchanged(model,initial)
        assert current['scale_calibration']['fixed_scales']==model.multilevel.fixed_scales
        for level in LEVELS:
            measured=math.sqrt(sum(r['sums'][level] for r in current['scale_calibration']['records'])/sum(r['coordinates'][level] for r in current['scale_calibration']['records']))
            assert math.isclose(measured,model.multilevel.fixed_scales[level],rel_tol=1e-12)
        require_calibration(c,model,current,448)
        again,changed=advance_calibration(c,model,L,device,directory,current,445)
        assert not changed and again is current and len(calls)==48
        rejected(lambda:model.multilevel.measure_scales(model,L,device))
        state=model.multilevel.export_state();restored=BoundarySensitivity(model.base,seed,model.multilevel.objective)
        restored.restore_state(state);assert restored.export_state()==state
        rejected(lambda:restored.restore_state(state))
        invalid=[None,{},dict(state,objective='other')]
        for value in (0.,-1.,float('nan'),float('inf')):
            bad=deepcopy(state);bad['fixed_scales'][LEVELS[0]]=value;invalid.append(bad)
        for bad in invalid:rejected(lambda:BoundarySensitivity(model.base,seed,model.multilevel.objective).restore_state(bad))
        _,terms,detail,_=model.multilevel._pair(model,L,448,device,c['multilevel'],True)
        assert all(len(detail['levels'][k]['sensitivity_A']['raw_RMS_by_RX_TX'])==30 for k in LEVELS)
        total=sum(terms.values());total.backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.base.parameters())
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert list(model.multilevel.parameters())==[] and model.multilevel.state_dict()=={}
    hook.remove()
    return dict(calibration_forwards=48,L_readouts=2880,same_pairs=True,state_RNG_modes_gradients_unchanged=True,checkpoint_state_rejection=True,live_second_derivative=True)


def run(device):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    result={arm:dict(derivatives=derivative_checks(device,'ce' if arm=='CE_ENERGY' else 'margin'),lifecycle=lifecycle(device,arm)) for arm in d.ARMS}
    torch.backends.cudnn.deterministic=False
    return dict(status='PASS',device=device,synthetic_only=True,target_access=False,arms=result,source_entry=source_entry())


def source_entry():
    """Execute the real training entry through the first objective boundary."""
    from . import source,peer_cache,provenance
    from experiments.cvs_e20_budget_fishr.checks import fixture_parent
    from experiments.cvs_early_geometry.checks import fixture
    L,_=fixture();U=dict(x=L['x'],rx=L['rx'],day=L['day'],ids=L['ids'])
    class StopAtFirstObjective(Exception):pass
    results=[]
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm));_,proof=fixture_parent(c)
        teachers={str(seed):dict(proof,seed=seed,checkpoint=provenance.teacher_path(seed)) for seed in c['teacher']['seeds']}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'release_commit.txt').write_text('synthetic-only',encoding='utf-8');c['output_root']=str(root/'source')
            def boundary(config,model,bank,l,li,u,ui,step,device):
                assert step==1 and ui is None and not hasattr(model.multilevel,'fixed_scales')
                initial=json.loads((root/'source/initial_resolved_config.json').read_text(encoding='utf-8'))
                assert initial['fishr_calibration']['status']=='PENDING_CALIBRATION'
                assert not (root/'source/calibration.json').exists()
                raise StopAtFirstObjective
            with contextlib.redirect_stdout(io.StringIO()),patch.object(d,'validate',lambda c:c),patch.object(d,'ROOT',root),patch.object(source,'load_source',return_value=(L,U,L,{})),patch.object(peer_cache,'load',return_value={'provenance':teachers}),patch.object(source,'objective',side_effect=boundary):
                try:source.train(c,'cpu')
                except StopAtFirstObjective:results.append(arm)
                else:raise AssertionError('Training entry did not reach expected boundary')
    return dict(status='PASS',arms=results,identity_updates=0,source_IQ=False,target_access=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.device);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],device=a.device)))
