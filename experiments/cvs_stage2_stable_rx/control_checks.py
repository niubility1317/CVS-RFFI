"""Focused estimator/control checks and CPU source-only publication preflight."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch
import torch
from . import design as d
from .receiver import fit_receiver,load_receiver


def compare_streams(cache,old,new,steps):
    from experiments.cvs_stage2_decouple.source import SourceStream
    a=SourceStream(len(cache['ids']),load_receiver(old),123)
    b=SourceStream(len(cache['ids']),load_receiver(new),123)
    for step in range(1,steps+1):
        ia,ra=a.step(step);ib,rb=b.step(step)
        if not torch.equal(ia,ib):raise ValueError('Ordinary/relation physical sample stream differs')
        if ra is not None:
            for x,y in zip(ra['relations'],rb['relations']):
                for key in ('group_ids','positions','block_id','tx','day'):
                    if x[key]!=y[key]:raise ValueError('Relation group selection differs')


def check():
    from experiments.cvs_stage2_geometry.receiver_checks import synthetic_cache
    from experiments.cvs_stage2_geometry.receiver import fit_receiver as oldfit
    from . import evaluate as e,dispatch as q
    torch.set_num_threads(2)
    cache=synthetic_cache();rng=torch.get_rng_state().clone()
    payload=fit_receiver(cache,123)
    assert torch.equal(rng,torch.get_rng_state()) and len(payload['folds'])==6
    assert all(abs(f['c']-.64)<1e-9 for f in payload['folds'])
    assert all(abs(b['alpha']-.25)<1e-12 for b in payload['blocks'])
    for kind in ('held_tx','held_day'):
        modified=deepcopy(cache)
        mask=modified['y']<2 if kind=='held_tx' else modified['day']==0
        modified['parent_feature'][mask,:2]*=-1
        altered=fit_receiver(modified,123)
        for rx,value in payload['folds'][0]['rx_offsets'].items():
            torch.testing.assert_close(value,altered['folds'][0]['rx_offsets'][rx],rtol=0,atol=0)
    compare_streams(cache,oldfit(cache,123),payload,8880)
    bank=load_receiver(payload);r=bank.sample_relations(torch.Generator().manual_seed(19))
    z=cache['parent_feature'][r['indices']].clone().requires_grad_(True)
    loss,detail=bank.identity_loss(z,r['indices'],r);loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm()>0
    assert detail['variance_gradient_retained'] and detail['signed_estimate_not_clamped']
    for row in d.rows():
        c=d.validate(d.config(row));prior=d.previous.config(dict(row,arm='B3',row_id=f"B3-s{row['model_seed']}"))
        for key in ('optimizer','losses','enabled','action_policy','action','schedule','parent_contract','epochs','updates'):
            assert c[key]==prior[key]
        try:d.validate(dict(c,epochs=39))
        except ValueError:pass
        else:raise AssertionError('Changed budget accepted')
    assert q.choose_gpu({0:dict(pids={1,2,3,4},free_mb=99999)}) is None
    assert q.choose_gpu({0:dict(pids={1,2,3},free_mb=99999)})==0
    count=dict(status='SCORED_COMPLETE',models=3,views=1,metric_records=42,day_metric_records=12,
        query_count_per_view=168000,prediction_count=504000,truth_last=True,independent_recount='VERIFIED')
    q.require_scored(count,3)
    with patch.object(e,'snapshot',return_value=[d.config(d.rows()[0])]),\
         patch.object(e,'prediction_preflight',side_effect=ValueError('incomplete')),\
         patch.object(d,'read',side_effect=AssertionError('truth must stay closed')):
        try:e.score()
        except ValueError:pass
        else:raise AssertionError('Incomplete predictions scored')
    results=[]
    for arm in ('P0',*d.CONTROL_ARMS,*d.ARMS):
        for seed in d.SEEDS:
            for dim in ('overall','receiver'):
                results.append(dict(arm=arm,model_seed=seed,row_id=f'{arm}-s{seed}',dimension=dim,accuracy=.8,macro_f1=.79))
    assert len(e.summarize(results))==5 and len(e.paired_results(results))==5
    return dict(status='PASS',source_only=True,target_access=False,
        checks=['closed_form_shrink_reliability','held_TX_exclusion','held_day_exclusion',
        'same_8880_sample_stream','finite_full160_gradient','unchanged_B3_objective_budget',
        'capacity_four','truth_last_boundary','complete_control_summary'])


def remote_preflight():
    from experiments.cvs_stage2_geometry import dispatch as oldq
    from experiments.cvs_stage2_geometry import evaluate as old_eval
    from experiments.cvs_stage2_geometry.identity_screen import RUN as SCREEN
    from experiments.cvs_stage2_geometry.actions import load_actions
    from experiments.cvs_stage2_geometry.source import forward_objective
    from experiments.cvs_stage2_decouple.common import build_model,verify_parent,parameter_groups
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context
    torch.set_num_threads(2);started=time.perf_counter()
    oldq.require_preparations()
    screen=Path(d.PROJECT)/'runs'/SCREEN
    sd=d.read(screen/'completion.json');summary=d.read(screen/'summary.json')
    if (sd.get('status')!='SOURCE_DIAGNOSTIC_COMPLETE' or sd.get('rows_completed')!=3
            or sd.get('target_access') is not False or sd.get('identity_updates')!=0
            or {r['seed'] for r in summary}!=set(d.SEEDS)
            or not all(r['decision']['no_material_identity_damage'] for r in summary)):
        raise ValueError('Preregistered source V identity screen has not passed')
    output=d.RELEASE_BASE/'receiver_payloads';output.mkdir(exist_ok=False)
    rows={};records=[]
    with torch.random.fork_rng(devices=[]),numerical_context(d.FULL_FP32_POLICY):
        for row in d.rows():
            seed=row['model_seed'];c=d.config(row);folder=Path(c['preparation'])
            provenance=verify_parent(c['parent_contract'])
            # Only fixed source metadata is read here, never target predictions
            # or scores. Fail before spending a new E40 if controls differ.
            for arm in d.CONTROL_ARMS:
                control=old_eval.snapshot(f'{arm}-s{seed}')[0]
                old_done=d.read(Path(control['output_root'])/'completion.json')
                if (old_done.get('status')!='SOURCE_TRAINED' or old_done.get('config')!=control
                        or old_done.get('epoch')!=40 or old_done.get('optimizer_steps')!=8880
                        or old_done.get('target_access') is not False):
                    raise ValueError('Fixed source control completion differs')
                for key in ('parent_contract','source_contract','optimizer','schedule','epochs','updates','numerical_policy'):
                    if control[key]!=c[key]:raise ValueError('Fixed control recipe differs: '+key)
            packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE))
            verify_contract(d.read(folder/'preparation_complete.json')['source_contract'],packed['contract'])
            cache=packed['labeled']
            for key,role,count in [('labeled','L_s',6300),('validation','V',27000)]:
                ids=packed[key]['ids']
                if len(ids)!=count or len(set(ids))!=count or sorted(ids)!=sorted(packed['contract']['role_ids'][role]):
                    raise ValueError('Cached source physical roles differ')
            payload=fit_receiver(cache,seed)
            old=torch.load(folder/'receiver.pt',map_location='cpu',weights_only=False)
            compare_streams(cache,old,payload,8880)
            bank=load_receiver(payload)
            if len(payload['folds'])!=9 or payload['max_rank']!=4:raise ValueError('Actual stable RX fold contract differs')
            if any(f['fit_ids']!=2800 for f in payload['folds']):raise ValueError('Four TX x two days x five RX x70 fit expected')
            torch.save(payload,Path(c['receiver_payload']))
            diag=dict(fit_seconds=payload['fit_seconds'],folds=[{k:v for k,v in f.items()
                if k not in ('rx_offsets','fit_source_ids','error_spectrum')} for f in payload['folds']],
                alphas=[b['alpha'] for b in payload['blocks']],method=payload['method'],same_8880_stream=True)
            d.write(output/f's{seed}.json',diag);rows[str(seed)]=diag
            model=build_model(c['parent_contract'],'cpu');groups,_=parameter_groups(model);model.train()
            action_payload=torch.load(folder/'actions.pt',map_location='cpu',weights_only=False)
            if action_payload['seed']!=seed or action_payload['source_ids']!=cache['ids']:raise ValueError('Actions source provenance differs')
            actions=load_actions(action_payload,'cpu')
            relation=bank.sample_relations(torch.Generator().manual_seed(seed+723))
            indices=torch.cat((torch.arange(96),relation['indices']))
            x,y,margin=(cache[k][indices] for k in ('x','y','parent_margin'))
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            domain_calls=[];hook=model.dom_backbone.register_forward_hook(lambda *a:domain_calls.append(1))
            try:
                total,weighted,parts,observed=forward_objective(c,model,x,y,margin,888,
                    actions=actions,relation=relation,bank=bank)
                grad=torch.autograd.grad(weighted['receiver'],groups[0]['params'],retain_graph=True,allow_unused=True)
                rnorm=float(sum((g.detach().square().sum() for g in grad if g is not None),torch.zeros(())).sqrt())
                total.backward()
                if not torch.isfinite(total) or not rnorm>0:raise ValueError('Nonfinite objective or inactive stable R gradient')
                if domain_calls or any(p.grad is None or not torch.isfinite(p.grad).all() for g in groups for p in g['params']):
                    raise ValueError('Unexpected domain forward or missing/nonfinite identity gradient')
                if any(not torch.equal(before[n],p) for n,p in model.named_parameters()):raise ValueError('Preflight changed parent')
                if any(p.grad is not None for p in actions.networks.parameters()):raise ValueError('Auxiliary not frozen')
                records.append(dict(seed=seed,parent=provenance,loss=float(total.detach()),receiver_encoder_gradient_norm=rnorm,
                    domain_forwards=0,optimizer_constructed=False,parameters_unchanged=True,target_access=False))
            finally:hook.remove()
            del model,total,weighted,parts,actions,before,packed,cache,bank,payload,old
    done=dict(status='FROZEN_STABLE_RX',method=d.METHOD,seeds=list(d.SEEDS),rows=rows,target_access=False,
        identity_updates=0,fit_only='L_s; held TX/day excluded',source_screen=SCREEN)
    d.write(output/'complete.json',done)
    return dict(status='PASS',source_only=True,target_access=False,identity_updates=0,
        formal_parent_models=3,formal_parent_forward_backward=records,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--remote-preflight',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():raise FileExistsError('Preserve prior preflight')
    result=check()
    if args.remote_preflight:result=dict(status='PASS',synthetic=result,remote=remote_preflight())
    if args.output:d.write(args.output,result)
    print(json.dumps(result,indent=2))
