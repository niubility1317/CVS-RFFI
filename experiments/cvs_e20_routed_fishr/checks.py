"""Native bounded equivalence/routing checks; no source/target data required."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from . import multilevel as m
from .objective import objective
from experiments.cvs_e20_budget_fishr.multilevel import MultiLevel as OldMultiLevel,_norm,_parameter_groups
from experiments.cvs_e20_budget_fishr.objective import objective as old_objective


def run(device='cpu'):
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    from experiments.cvs_early_geometry import design as fixture_design
    from experiments.cvs_early_geometry.model import build
    from experiments.cvs_e20_budget_fishr import design as previous_design
    from experiments.cvs_e20_budget_fishr.multilevel_checks import duplicate_pid_checks
    configs={r['arm']:d.config(r) for r in d.rows() if r['model_seed']==d.SEEDS[0]}
    for c in configs.values():d.validate(c)
    model=build(fixture_design.config(next(r for r in fixture_design.rows() if r['arm']=='PHYS')),device)
    old_heads=OldMultiLevel(model.base,seed=d.SEEDS[0]).to(device)
    new_heads=m.MultiLevel(model.base,seed=d.SEEDS[0]).to(device)
    new_heads.load_state_dict(old_heads.state_dict());model.multilevel=old_heads;model.train()
    source_rng=torch.Generator().manual_seed(2934)
    L=dict(x=torch.randn(180,2,256,generator=source_rng),y=torch.arange(6).repeat_interleave(10).repeat(3),
        rx=torch.tensor([1,3,4,6,8]).repeat_interleave(2).repeat(18),day=torch.arange(3).repeat_interleave(60),ids=list(range(180)))
    class Hidden(dict):
        def __getitem__(self,key):
            if key=='y':raise AssertionError('U truth read')
            return super().__getitem__(key)
    U=Hidden(role='U_s',x=torch.randn(180,2,256,generator=source_rng),rx=L['rx'].clone(),day=L['day'].clone(),ids=list(range(1000,1180)))
    li=torch.arange(128);ui=torch.tensor(list(range(127))+[0]);step=1332
    initial={n:t.clone() for n,t in model.state_dict().items()}
    def seed():
        torch.manual_seed(923)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(923)
    def rng():return [torch.random.get_rng_state().clone()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    def trial(function,config,heads):
        model.multilevel=heads;model.load_state_dict(initial);model.zero_grad(set_to_none=True);model.train();seed()
        total,parts,weighted,info=function(config,model,None,L,li,U,ui,step,device)
        total.backward()
        grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
        return dict(loss=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},grads=grads,rng=rng(),info=info)
    old_config=previous_design.config(next(r for r in previous_design.rows() if r['arm']=='JOINT' and r['model_seed']==d.SEEDS[0]))
    old=trial(old_objective,old_config,old_heads);new=trial(objective,configs['COUPLED'],new_heads)
    assert old['loss']==new['loss'] and old['parts']==new['parts']
    assert old['grads'].keys()==new['grads'].keys()
    for name in old['grads']:
        a,b=old['grads'][name],new['grads'][name]
        assert (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)),name
    assert all(torch.equal(a,b) for a,b in zip(old['rng'],new['rng']))
    assert old['info']['U_style']==new['info']['U_style']
    model.multilevel=new_heads;model.load_state_dict(initial);model.zero_grad(set_to_none=True)
    base_parameters=tuple(model.base.parameters());head_parameters=tuple(model.multilevel.heads.parameters());native_groups=_parameter_groups(model)
    layer_results={}
    for arm in ('F_SHALLOW','F_MIDDLE','F_BOTH','STYLE','JOINT'):
        mc=configs[arm]['multilevel'];seed();state=rng()
        with patch.object(m,'fishr_variance',wraps=m.fishr_variance) as variance:
            ce,terms,details=model.multilevel.loss(model,L,step,device,mc)
            called=[call.args[0].shape[1] for call in variance.call_args_list]
        assert called==[48 if level=='time_down' else 96 for level in mc['fishr_levels']],(arm,called)
        assert set(terms)==set(mc['fishr_levels'])
        assert all(torch.equal(a,b) for a,b in zip(state,rng()))
        assert _norm(ce,base_parameters)==0 and _norm(ce,head_parameters)>0
        norms={}
        for level,term in terms.items():
            assert _norm(term,head_parameters)==0
            norms[level]={name:_norm(term,ps) for name,ps in native_groups.items()}
            assert norms[level][level]>0
            if level=='time_down':assert norms[level]['t2.pool']==0
        main_ce=torch.nn.functional.cross_entropy(model(L['x'][li].to(device)),L['y'][li].to(device))
        fishr,budget=m.budget_fishr(model,main_ce,terms,mc,1.)
        if terms:
            manual=torch.stack([terms[level]*budget['levels'][level]['coefficient'] if level in terms else main_ce*0 for level in m.LEVELS]).mean()
            assert torch.equal(fishr,manual)
            for level in m.LEVELS:
                record=budget['levels'][level]
                if level not in terms:
                    assert record['reason']=='inactive_by_design' and not record['technical_failure'] and record['coefficient']==0
                else:assert record['achieved_direct_contribution_ratio']==record['achieved_ratio_pre_level_mean']/2
        else:assert float(fishr.detach())==0
        layer_results[arm]=dict(computed_feature_dimensions=called,CE_backbone_norm=0,CE_head_norm=_norm(ce,head_parameters),fishr_gradient_norms=norms,budget=budget)
    # Local route-only fixture: STYLE with style disabled, retaining its head CE.
    # This is a regression probe, not an extra registered training arm.
    head_only=deepcopy(configs['STYLE']);head_only['multilevel'].update(mixstyle_u=False,style_levels=[])
    def update(config):
        model.load_state_dict(initial);model.zero_grad(set_to_none=True);model.train();seed()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
        loss,_,_,_=objective(config,model,None,L,li,U,ui,step,device);loss.backward()
        gradients={n:p.grad.clone() if p.grad is not None else None for n,p in model.base.named_parameters()}
        optimizer.step()
        return {n:t.clone() for n,t in model.base.state_dict().items()},gradients,rng(),{n:t.clone() for n,t in model.multilevel.state_dict().items()}
    plain,gplain,rplain,hplain=update(configs['SSL']);routed,grouted,rrouted,hrouted=update(head_only)
    assert all(torch.equal(plain[n],routed[n]) for n in plain)
    assert all((gplain[n] is None and grouted[n] is None) or (gplain[n] is not None and grouted[n] is not None and torch.equal(gplain[n],grouted[n])) for n in gplain)
    assert all(torch.equal(a,b) for a,b in zip(rplain,rrouted)) and any(not torch.equal(hplain[n],hrouted[n]) for n in hplain)
    duplicate=duplicate_pid_checks(device)
    try:objective(configs['SSL'],model,None,L,li,dict(U,y=object()),ui,step,device)
    except ValueError:pass
    else:raise AssertionError('U truth accepted')
    return dict(status='PASS',device=device,target_access=False,registered_arms=list(configs),
        coupled_old_JOINT_loss_parts_all_gradients_RNG_bitwise_equal=True,coupled_style_details_equal=True,
        detached_CE_identity_gradient_zero_heads_nonzero=True,Fishr_heads_gradient_zero=True,
        only_registered_Fishr_layers_computed=True,fixed_denominator_two=True,
        head_only_SSL_identity_gradient_optimizer_step_RNG_bitwise_equal=True,head_only_heads_updated=True,
        native_U_s_without_labels=True,U_truth_poison_rejected=True,repeated_PID=duplicate,layers=layer_results)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('layers','repeated_PID')},allow_nan=False))
