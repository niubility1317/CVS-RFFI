"""Budget-Fishr and hidden-label-safe U-style checks, no target access."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .multilevel import MultiLevel,LEVELS,linear_sample_gradients,fishr_variance,budget_fishr,_norm,_parameter_groups,weak_mixstyle_u
from experiments.cvs_early_geometry.multilevel import weak_mixstyle


def duplicate_pid_checks(device='cpu'):
    """Focused P1 regression; does not build models or launch a matrix."""
    import numpy as np
    value=torch.randn(8,4,16,generator=torch.Generator().manual_seed(551)).to(device)
    labels=torch.tensor([0,0,0,0,1,1,12,13],device=device)
    rx=torch.tensor([1,3,4,6,1,3,1,3],device=device)
    day=torch.zeros(8,dtype=torch.long,device=device)
    unique=list(range(8));duplicate=[10,10,11,12,20,21,20,20]
    torch_before=torch.random.get_rng_state().clone();numpy_before=np.random.get_state()
    cuda_before=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None
    # Multiple deterministic seeds exercise both Bernoulli paths and exact donor
    # and Beta draw trajectories when IDs are unique.
    for seed in range(32):
        old,old_info=weak_mixstyle(value,labels,rx,day,unique,seed,p=.5,alpha=.1,eta=.3)
        new,new_info=weak_mixstyle_u(value,labels,rx,day,unique,seed,p=.5,alpha=.1,eta=.3)
        assert torch.equal(old,new) and old_info==new_info
        changed,info=weak_mixstyle_u(value,labels,rx,day,duplicate,seed,p=1.,alpha=.1,eta=.3)
        assert info['eligible_packets']==6 and info['changed_packets']>0
        for i,j in enumerate(info['donors']):
            if i<6:
                assert duplicate[i]!=duplicate[j] and labels[i]==labels[j] and rx[i]!=rx[j] and day[i]==day[j]
            else:
                assert i==j and torch.equal(changed[i],value[i])
    assert torch.equal(torch_before,torch.random.get_rng_state())
    numpy_after=np.random.get_state()
    assert numpy_before[0]==numpy_after[0] and np.array_equal(numpy_before[1],numpy_after[1]) and numpy_before[2:]==numpy_after[2:]
    if cuda_before is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all()))
    try:weak_mixstyle(value,labels,rx,day,duplicate,1,p=1.,eta=.3)
    except ValueError:pass
    else:raise AssertionError('L auxiliary uniqueness check was weakened')
    return dict(status='PASS',device=device,seeds=32,U_repeated_PID_allowed=True,
                same_physical_PID_donor_forbidden=True,other_PID_legal_donors_used=True,
                low_confidence_unique_groups_unmixed=True,unique_PID_old_new_bitwise_equal=True,
                unique_PID_random_draw_trajectory_equal=True,global_RNG_unchanged=True,L_uniqueness_preserved=True)


def run(device='cpu'):
    torch.set_num_threads(2)
    repeated_pid_regression=duplicate_pid_checks(device)
    generator=torch.Generator().manual_seed(33011)
    feature=torch.randn(12,9,generator=generator,dtype=torch.float64).to(device).requires_grad_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(344);head=nn.Linear(9,6).to(device=device,dtype=torch.float64)
    labels=torch.arange(12,device=device)%6
    samples=linear_sample_gradients(feature,head,labels)
    auto=[]
    for loss in F.cross_entropy(head(feature),labels,reduction='none'):
        w,b=torch.autograd.grad(loss,(head.weight,head.bias),retain_graph=True)
        auto.append(torch.cat((w,b[:,None]),1))
    auto=torch.stack(auto)
    torch.testing.assert_close(samples,auto,rtol=1e-10,atol=1e-12)
    penalty,_=fishr_variance(feature,head,labels,torch.tensor([1]*6+[3]*6,device=device))
    assert all(g is None for g in torch.autograd.grad(penalty,tuple(head.parameters()),retain_graph=True,allow_unused=True))
    assert _norm(penalty,(feature,))>0
    from experiments.cvs_early_geometry import design as old
    from experiments.cvs_early_geometry.model import build
    model=build(old.config(next(row for row in old.rows() if row['arm']=='PHYS')),device)
    auxiliary=MultiLevel(model.base,seed=7711).to(device)
    model.train();auxiliary.train()
    source=dict(x=torch.randn(180,2,256,generator=generator),
                y=torch.arange(6).repeat_interleave(10).repeat(3),
                rx=torch.tensor([1,3,4,6,8]).repeat_interleave(2).repeat(18),
                day=torch.arange(3).repeat_interleave(60),ids=list(range(180)))
    cfg=dict(enabled=True,interval=4,batch_size=60,levels=list(LEVELS),mixstyle_l=False,mixstyle_u=False,
             mixstyle_p=1.,mixstyle_alpha=.1,mixstyle_strength=.3,fishr_mode='budget',
             fishr_ratio=.1,fishr_eps=1e-12,fishr_cap=10000.,fishr_raw_weight=2.,style_confidence=.95)
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    x=source['x'][:60].to(device);y=source['y'][:60].to(device)
    main_ce=F.cross_entropy(model(x),y)
    rng=torch.random.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None
    ce,terms,detail=auxiliary.loss(model,source,4,device,cfg)
    assert torch.equal(rng,torch.random.get_rng_state())
    if cuda is not None:assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    assert all(g is None for g in torch.autograd.grad(sum(terms.values()),tuple(auxiliary.heads.parameters()),retain_graph=True,allow_unused=True))
    assert _norm(ce,tuple(auxiliary.heads.parameters()))>0
    weighted,budget=budget_fishr(model,main_ce,terms,cfg,ramp=.7)
    assert weighted>0 and torch.isfinite(weighted)
    for level,metrics in budget['levels'].items():
        assert metrics['coefficient']>0 and metrics['main_ce_gradient_norm']>0
        assert metrics['raw_fishr_gradient_norm']>0
        if not metrics['cap_active']:
            assert abs(metrics['achieved_ratio_pre_level_mean']-.07)<1e-6
            assert abs(metrics['achieved_direct_contribution_ratio']-.035)<1e-6
    cap_loss,cap_detail=budget_fishr(model,main_ce,terms,dict(cfg,fishr_cap=1e-6),ramp=1)
    assert all(v['cap_active'] and v['coefficient']==1e-6 for v in cap_detail['levels'].values())
    raw,raw_detail=budget_fishr(model,main_ce,terms,dict(cfg,fishr_mode='raw'),ramp=.5)
    torch.testing.assert_close(raw,torch.stack(list(terms.values())).mean())
    null,null_detail=budget_fishr(model,main_ce,{k:v*0 for k,v in terms.items()},cfg,ramp=1)
    assert null==0 and all(v['coefficient']==0 and v['reason'] for v in null_detail['levels'].values())
    invalid,invalid_detail=budget_fishr(model,main_ce,{k:v*float('nan') for k,v in terms.items()},cfg,ramp=1)
    assert invalid==0 and all(v['coefficient']==0 and v['reason'] for v in invalid_detail['levels'].values())
    json.dumps(invalid_detail,allow_nan=False)
    both_cfg=dict(cfg,mixstyle_l=True)
    both_ce,both_terms,both_detail=auxiliary.loss(model,source,4,device,both_cfg)
    assert both_detail['aux_forwards']==2
    for level in LEVELS:torch.testing.assert_close(terms[level],both_terms[level])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())

    class PoisonLabels(dict):
        def __getitem__(self,key):
            if key=='y':raise AssertionError('Hidden U truth accessed')
            return super().__getitem__(key)
    u=PoisonLabels(x=x,rx=source['rx'][:60].to(device),day=source['day'][:60].to(device),ids=list(range(60)))
    probabilities=torch.full((60,6),.004,device=device)
    pseudo=torch.arange(6,device=device).repeat_interleave(10)
    probabilities.scatter_(1,pseudo[:,None],.98)
    probabilities[:4]=1/6
    ucfg=dict(cfg,mixstyle_u=True)
    hooks=sum(len(m._forward_hooks) for m in model.modules())
    dropout_modes=[(m,m.training) for m in model.modules() if isinstance(m,nn.modules.dropout._DropoutNd)]
    # Compare consumption by one ordinary versus one styled strong forward.
    cpu_start=torch.random.get_rng_state().clone()
    cuda_start=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None
    model(u['x'])
    cpu_end=torch.random.get_rng_state().clone()
    cuda_end=torch.cuda.get_rng_state_all() if cuda_start is not None else None
    torch.random.set_rng_state(cpu_start)
    if cuda_start is not None:torch.cuda.set_rng_state_all(cuda_start)
    with auxiliary.main_style(probabilities,u['rx'],u['day'],u['ids'],4,ucfg) as u_detail:
        assert all(m.training==mode for m,mode in dropout_modes)
        styled=model(u['x'])
    assert torch.equal(cpu_end,torch.random.get_rng_state())
    if cuda_end is not None:assert all(torch.equal(a,b) for a,b in zip(cuda_end,torch.cuda.get_rng_state_all()))
    assert u_detail['high_confidence']==56 and not u_detail['used_U_truth']
    for record in u_detail['levels'].values():
        for i,j in enumerate(record['donors']):
            if i!=j:
                assert i>=4 and j>=4 and pseudo[i]==pseudo[j] and u['rx'][i]!=u['rx'][j]
                assert u['day'][i]==u['day'][j]
    assert sum(len(m._forward_hooks) for m in model.modules())==hooks
    assert all(m.training==mode for m,mode in dropout_modes)
    model.eval()
    with torch.no_grad():
        baseline=model(x)
        with auxiliary.main_style(probabilities,u['rx'],u['day'],u['ids'],4,ucfg) as disabled:
            evaluated=model(x)
        torch.testing.assert_close(baseline,evaluated,rtol=0,atol=0)
        assert not disabled['levels']
    model.train()
    try:
        with auxiliary.main_style(probabilities,u['rx'],u['day'],u['ids'],4,ucfg):
            raise RuntimeError('intentional cleanup test')
    except RuntimeError:pass
    assert not auxiliary._active and sum(len(m._forward_hooks) for m in model.modules())==hooks
    # Warm heads are the sole update recipient when features are cached/detached.
    cache=auxiliary.extract_clean_features(model,x)
    assert not any(v.requires_grad for v in cache.values())
    warmed=auxiliary.probe_ce(cache,y)
    assert _norm(warmed,tuple(auxiliary.heads.parameters()))>0
    assert all(g is None for g in torch.autograd.grad(warmed,tuple(model.parameters()),allow_unused=True,retain_graph=True))
    return dict(status='PASS',device=device,target_access=False,
        analytic_parameter_gradient_max_error=float((samples-auto).abs().max().detach()),
        Fishr_backbone_only=True,heads_learn_by_CE=True,actual_real_layer_shapes={k:v['shape'] for k,v in detail['levels'].items()},
        budget=budget,cap_behavior=True,zero_gradient_safe=True,nonfinite_gradient_zero_JSONsafe=True,
        raw_coefficient_preserved=True,
        auxiliary_rng_preserved=True,U_truth_poison_not_accessed=True,U_both_donors_confident=True,
        main_dropout_unchanged=True,main_rng_consumption_unchanged=True,hooks_cleaned=True,
        inference_bitwise_unchanged=True,cached_warm_heads_only=True,repeated_pid_regression=repeated_pid_regression)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    parser.add_argument('--duplicate-pid-only',action='store_true')
    args=parser.parse_args();result=duplicate_pid_checks(args.device) if args.duplicate_pid_only else run(args.device)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))
