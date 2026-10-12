"""Conditional nullspace, unbiased finite-pool cross estimator and live mixed derivatives."""
import argparse,json,math,contextlib,io
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from . import design as d
from .multilevel import BoundarySensitivity,interaction_means,LEVELS
from .calibration import begin_calibration,advance_calibration,require_calibration
from experiments.cvs_e20_direction_audit.checks import Model,_fixture
from experiments.cvs_e20_boundary_style.checks_fishr import _state,_unchanged
from experiments.cvs_scratch_energy_sensitivity.checks_energy import derivative_checks,rejected
from experiments.cvs_equivariant_identity.precision import numerical_context

def algebra(device):
    torch.manual_seed(107)
    y=torch.arange(6,device=device).repeat_interleave(10);rx=torch.tensor([1,3,4,6,8],device=device).repeat_interleave(2).repeat(6)
    tx=torch.randn(6,1,1,7,device=device,dtype=torch.float64);receiver=torch.randn(1,5,1,7,device=device,dtype=torch.float64)
    additive=(tx+receiver).expand(6,5,2,7).reshape(60,7)
    b,_=interaction_means(additive,y,rx);assert b.abs().max()<1e-12
    g=torch.randn(60,7,device=device,dtype=torch.float64,requires_grad=True)
    a,m=interaction_means(g,y,rx);shifted,_=interaction_means(g+additive,y,rx)
    torch.testing.assert_close(a,shifted,rtol=1e-12,atol=1e-12)
    a.square().mean().backward();assert torch.isfinite(g.grad).all() and bool(g.grad.abs().sum()>0)
    assert a.mean(0).abs().max()<1e-12 and a.mean(1).abs().max()<1e-12
    rejected(lambda:interaction_means(g[:-1],y[:-1],rx[:-1]))
    # Independent cross-product expectation is exactly population interaction energy.
    from experiments.cvs_scratch_energy_sensitivity.moment_identifiability import exhaustive_cross_expectation
    return dict(additive_TX_RX_nullspace=True,interaction_nonzero_grad=True,finite_pool=exhaustive_cross_expectation(device))

def lifecycle(device,arm):
    c=d.config(next(r for r in d.rows() if r['arm']==arm));torch.manual_seed(991)
    model=Model(c['model_seed']).to(device).double();mc=c['multilevel']
    model.multilevel=BoundarySensitivity(model.base,c['model_seed'],mc['sensitivity_objective'],mc['conditional_components'])
    L=_fixture();L['x']=L['x'].double();model.train();before=_state(model);calls=[];pairs=[]
    hook=model.register_forward_pre_hook(lambda m,args:calls.append(len(args[0])))
    sampler=model.multilevel.pair_indices
    def sample(*a,**kw):
        value=sampler(*a,**kw);pairs.append((value[0].tolist(),value[1].tolist()));return value
    model.multilevel.pair_indices=sample
    with TemporaryDirectory() as tmp:
        current=begin_calibration(c,model,L,device,tmp)
        current,ready=advance_calibration(c,model,L,device,tmp,current,443);assert not ready and not calls
        rejected(lambda:require_calibration(c,model,current,448))
        current,ready=advance_calibration(c,model,L,device,tmp,current,444)
        assert ready and len(calls)==48 and sum(calls)==2880 and pairs[:12]==pairs[12:]
        from .analyze_source import _check_calibration
        assert _check_calibration(Path(tmp),c)==current
        _unchanged(model,before);require_calibration(c,model,current,448)
        state=model.multilevel.export_state();restored=BoundarySensitivity(model.base,c['model_seed'],model.multilevel.objective,model.multilevel.components)
        restored.restore_state(state);assert restored.export_state()==state
        for bad in (None,{},dict(state,components=['bad']),dict(state,fixed_component_coefficients={}),dict(state,fixed_coefficients={'time_down':2.,'t2.pool':1.})):
            rejected(lambda:BoundarySensitivity(model.base,c['model_seed'],model.multilevel.objective,model.multilevel.components).restore_state(bad))
        model.eval();_,terms,detail,_=model.multilevel._pair(model,L,448,device,mc,True)
        for level in LEVELS:
            v=detail['levels'][level]
            expected=sum(v['component_values'][k]*v['component_coefficients'][k] for k in mc['conditional_components'])/len(mc['conditional_components'])
            assert math.isclose(float(terms[level].detach()),expected,rel_tol=1e-12,abs_tol=1e-12)
        weight=model.base.id_backbone.cls_head.weight
        exact=torch.autograd.grad(sum(terms.values()),weight)[0][0,1]
        initial=float(weight[0,1].detach());h=1e-4
        def value(offset):
            with torch.no_grad():weight[0,1]=initial+offset
            _,terms,_,_=model.multilevel._pair(model,L,448,device,mc,True)
            return sum(terms.values()).detach()
        numerical=(value(h)-value(-h))/(2*h)
        with torch.no_grad():weight[0,1]=initial
        torch.testing.assert_close(exact,numerical,rtol=3e-4,atol=1e-8)
        assert not list(model.multilevel.parameters()) and model.multilevel.state_dict()=={}
        hook.remove()
        return dict(calibration_forwards=48,calibration_L=2880,same_AB_pairs=True,state_unchanged=True,components=mc['conditional_components'],
                    gradient_calibration_evaluations=current['coefficient_gradient_evaluations'],mixed_derivative_error=float(abs(exact-numerical)),checkpoint_rejection=True)

def run(device):
    torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY):
        mathcheck=algebra(device)
        derivatives={name:derivative_checks(device,name) for name in ('ce','margin')}
        results={arm:lifecycle(device,arm) for arm in d.ARMS}
    return dict(status='PASS',device=device,synthetic_only=True,target_access=False,algebra=mathcheck,raw_derivatives=derivatives,arms=results,source_entry=source_entry())

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
    result=run(a.device);d.write(a.output,result);print(json.dumps(dict(status=result['status'],device=a.device)))
