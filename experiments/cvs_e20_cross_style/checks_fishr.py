"""Bounded native paired-Fishr/calibration checks; no real datasets or targets."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import design as d
from . import multilevel as m
from experiments.cvs_e20_budget_fishr.multilevel import fishr_variance,_norm,_parameter_groups


def run(device='cpu'):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    from experiments.cvs_early_geometry import design as fixture_design
    from experiments.cvs_early_geometry.model import build
    from experiments.cvs_e20_moment_audit.checks_fishr import run as moment_checks
    mathematical=moment_checks()
    configs={r['arm']:d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]}
    model=build(fixture_design.config(next(r for r in fixture_design.rows() if r['arm']=='PHYS')),device)
    model.multilevel=m.MultiLevel(model.base,seed=d.SEEDS[0]).to(device);model.train()
    gen=torch.Generator().manual_seed(7702)
    L=dict(role='L_s',x=torch.randn(360,2,256,generator=gen),y=torch.arange(6).repeat_interleave(20).repeat(3),
        rx=torch.tensor([1,3,4,6,8]).repeat_interleave(4).repeat(18),day=torch.arange(3).repeat_interleave(120),ids=list(range(360)))
    def rng():return [torch.random.get_rng_state().clone()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    state={k:v.clone() for k,v in model.state_dict().items()};random_before=rng()
    pair_samples=[]
    for step in (448,452,456,460,464,468):
        a,b,day,sizes=model.multilevel.pair_indices(L,step)
        ca,cb,_,_=model.multilevel.pair_indices(L,step,calibration=True)
        assert len(a)==len(b)==60 and len(set(a.tolist()))==len(set(b.tolist()))==60
        assert bool((L['day'][a]==day).all()) and bool((L['day'][b]==day).all())
        assert not torch.equal(a,ca) and not torch.equal(b,cb)
        assert all(len(pool)==2 for pool in [a[(L['rx'][a]==r)&(L['y'][a]==k)] for r in (1,3,4,6,8) for k in range(6)])
        pair_samples.append(dict(day=day,overlap=len(set(a.tolist())&set(b.tolist()))))
    assert {v['day'] for v in pair_samples}=={0,1,2} and any(v['overlap']>0 for v in pair_samples)
    assert all(torch.equal(a,b) for a,b in zip(random_before,rng()))
    for mask in (L['day']!=2,~((L['day']==1)&(L['rx']==8)&(L['y']==5))):
        bad={key:([value[i] for i in torch.where(mask)[0].tolist()] if key=='ids' else value[mask] if torch.is_tensor(value) else value) for key,value in L.items()}
        try:model.multilevel.pair_indices(bad,448)
        except ValueError:pass
        else:raise AssertionError('Missing day/cell silently accepted')
    base=tuple(model.base.parameters());heads=tuple(model.multilevel.heads.parameters());groups=_parameter_groups(model)
    routing={}
    for arm in ('OLD_PAIR','F_SHALLOW','F_MIDDLE','F_BOTH','JOINT'):
        config=configs[arm]['multilevel'];before=rng()
        with patch.object(m,'linear_sample_gradients',wraps=m.linear_sample_gradients) as gradient_calls:
            ce,terms,detail=model.multilevel.loss(model,L,448,device,config)
        assert _norm(ce,base)==0 and _norm(ce,heads)>0
        assert all(torch.equal(a,b) for a,b in zip(before,rng()))
        assert detail['aux_forwards']==2 and detail['auxiliary_packets']==120 and detail['distinct_PID']==120-detail['cross_batch_PID_overlap']
        assert set(terms)==set(config['fishr_levels']) and len(gradient_calls.call_args_list)==2*len(terms)
        norms={}
        for level,term in terms.items():
            offset=list(terms).index(level)*2;calls=gradient_calls.call_args_list[offset:offset+2]
            features=[call.args[0] for call in calls]
            assert all(_norm(term,(value,))>0 for value in features) and _norm(term,heads)==0
            norms[level]={name:_norm(term,ps) for name,ps in groups.items()}
            assert norms[level][level]>0
            if level=='time_down':assert norms[level]['t2.pool']==0
            if arm=='OLD_PAIR':
                recomputed=[fishr_variance(call.args[0],call.args[1],call.args[2],L['rx'][model.multilevel.pair_indices(L,448)[j]].to(device))[0] for j,call in enumerate(calls)]
                torch.testing.assert_close(term,(recomputed[0]+recomputed[1])/2,rtol=2e-6,atol=1e-12)
        routing[arm]=dict(pair=detail,gradient_norms=norms)
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
    Path('local_artifacts').mkdir(exist_ok=True)
    calibration={};initial_modes=[module.training for module in model.modules()]
    with TemporaryDirectory(prefix='cross_calibration_',dir='local_artifacts') as directory:
        for arm in ('OLD_PAIR','F_SHALLOW','F_MIDDLE','F_BOTH','JOINT'):
            before=rng();out=Path(directory)/arm
            report=m.calibrate(configs[arm],model,L,device,out)
            calibration[arm]=report['fixed_coefficients']
            assert report['extra_source_readouts']==1440 and report['model_forwards']==24 and report['pairs']==12
            assert len(report['records'])==12 and set(report['fixed_coefficients'])==set(m.LEVELS)
            assert all(torch.equal(a,b) for a,b in zip(before,rng()))
            assert [module.training for module in model.modules()]==initial_modes
            assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
            assert not any('fixed_coefficients' in key for key in model.state_dict())
            try:m.calibrate(configs[arm],model,L,device,out)
            except FileExistsError:pass
            else:raise AssertionError('Calibration overwritten')
    assert all(values==calibration['OLD_PAIR'] for values in calibration.values())
    weighted={}
    for arm in ('OLD_PAIR','F_SHALLOW','F_MIDDLE','F_BOTH','JOINT'):
        config=configs[arm]['multilevel'];_,terms,info=model.multilevel.loss(model,L,448,device,config)
        ce=F.cross_entropy(model(L['x'][:60].to(device)),L['y'][:60].to(device));loss,detail=m.fixed_fishr(model,ce,terms,config,.4)
        expected=sum(term*model.multilevel.fixed_coefficients[level]*.4 for level,term in terms.items())/2
        torch.testing.assert_close(loss,expected,atol=1e-10,rtol=1e-6)
        _,changed=m.fixed_fishr(model,ce*7,terms,config,.4)
        assert all(detail['levels'][level]['coefficient']==changed['levels'][level]['coefficient'] for level in m.LEVELS)
        assert all(detail['levels'][level]['coefficient']==model.multilevel.fixed_coefficients[level]*.4 for level in terms)
        weighted[arm]=detail
    # Negative stochastic losses must survive fixed weighting unchanged.
    config=configs['F_BOTH']['multilevel'];negative={level:-term.abs() for level,term in terms.items()}
    loss,_=m.fixed_fishr(model,ce,negative,config,1.);assert float(loss.detach())<0
    assert all(p.grad is None for p in model.parameters())
    return dict(status='PASS',device=device,target_access=False,mathematical=mathematical,independent_pairs=pair_samples,
        calibration_identical_across_arms=calibration,calibration_state_modes_RNG_unchanged=True,calibration_cost_packets=1440,
        both_feature_paths_live=True,Fishr_head_gradient_zero=True,aux_CE_identity_gradient_zero=True,
        old_pair_exact_two_batch_average=True,equal_training_pair_budget=True,fixed_coefficients_not_state_buffers=True,
        no_current_batch_rescale=True,negative_cross_loss_not_clipped=True,missing_days_cells_rejected=True,
        routing=routing,fixed_budget=weighted)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('routing','fixed_budget','mathematical')},allow_nan=False))
