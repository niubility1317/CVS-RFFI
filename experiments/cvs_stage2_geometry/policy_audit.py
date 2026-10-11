"""Frozen B2 source audit of action freshness and task-aligned auxiliary fitting.

No identity update, target reader, deployment policy or training fallback.
New policies are diagnostic artifacts, frozen before any V scoring.
"""
import json
import math
import time
from pathlib import Path

import torch

from . import actions as a, design as d
from .objectives import margin_risk
from experiments.cvs_stage2_decouple.common import margins
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState

RUN = '20261011-phase1-stage2-policy-risk-source-manysig-m3-r01'
RELEASE = 'cvs_stage2_policy_risk_source_20261011_r01'
METHODS = ('coverage', 'old', 'fresh_raw', 'fresh_risk')
BATCH = 128


def search_loss(clean, changed, objective):
    if objective == 'raw':
        return -(clean.detach() - changed).mean()
    if objective == 'risk':
        return -margin_risk(clean, changed).mean()
    raise ValueError('Unregistered auxiliary objective')


def fit(model, cache, clean_margin, seed, objective):
    a.old._validate_cache(cache)
    if cache.get('role') != 'L_s' or objective not in ('raw', 'risk'):
        raise ValueError('Fit consumes only labeled source L_s')
    if clean_margin.shape != cache['y'].shape or not torch.isfinite(clean_margin).all():
        raise ValueError('Invalid frozen subject clean margins')
    nets, logs = {}, []
    with ReadOnlyState(model) as preservation:
        model.eval().requires_grad_(False)
        for ki, kind in enumerate(a.KINDS):
            # Exactly same initialization, permutations and proposal draws across objectives.
            g = a._generator(seed + 17041 * (ki + 1))
            net = a.old._new_net(kind, g, 'cpu')
            optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
            for epoch in range(2):
                order = torch.randperm(len(cache['y']), generator=g)
                for start in range(0, len(order), BATCH):
                    ix = order[start:start+BATCH]
                    x, y = cache['x'][ix], cache['y'][ix]
                    optimizer.zero_grad(set_to_none=True)
                    p = a._proposal(net, x, kind, g, 'conditional', cache['state'][ix])
                    ax = a.apply_proposal(x, p)
                    changed = margins(model(ax), y)
                    loss = search_loss(clean_margin[ix], changed, objective)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite search loss')
                    loss.backward()
                    norm = sum(v.grad.detach().square().sum() for v in net.parameters() if v.grad is not None).sqrt()
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite auxiliary gradient')
                    optimizer.step()
                    logs.append(dict(kind=kind,pass_index=epoch+1,batch=start//BATCH,samples=len(ix),
                        loss=float(loss.detach()),gradient_norm=float(norm),
                        finite_risk=float(margin_risk(clean_margin[ix],changed).detach().mean()),
                        margin_drop=float((clean_margin[ix]-changed).detach().mean()),
                        conditional_samples=int(p['conditional_mask'].sum())))
                print(json.dumps(dict(seed=seed,objective=objective,kind=kind,fit_pass=epoch+1)),flush=True)
            net.eval().requires_grad_(False)
            nets[kind] = net
    return nets, dict(objective=objective,steps=logs,auxiliary_updates=len(logs),
        sample_occurrences=4*len(cache['y']),identity_state=preservation.report,
        source_ids=list(cache['ids']),validation_fit=False)


@torch.no_grad()
def clean_forward(model, cache):
    return torch.cat([model(cache['x'][start:start+BATCH]).cpu() for start in range(0,len(cache['y']),BATCH)])


def summarize(values, labels, cache):
    def stats(mask):
        result = {key:float(value[mask].double().mean()) for key,value in values.items()}
        risk=values['finite_risk'][mask].double();k=max(1,math.ceil(len(risk)*.01))
        result.update(count=len(risk),risk_top1pct_share=float(risk.topk(k).values.sum()/risk.sum()) if float(risk.sum())>0 else 0.,
                      errors=int((values['correct'][mask]==0).sum()),
                      new_errors=int(values['clean_correct_to_error'][mask].sum()))
        return result
    out=dict(all=stats(torch.ones(len(labels),dtype=torch.bool)),strata={})
    for key in ('y','rx','day'):
        tags=cache[key]
        out['strata'][key]={str(v):stats(tags==v) for v in sorted(set(tags.tolist()))}
    out['strata']['tx_rx']={f'{y}/{rx}':stats((labels==y)&(cache['rx']==rx))
        for y in sorted(set(labels.tolist())) for rx in sorted(set(cache['rx'].tolist()))}
    return out


@torch.no_grad()
def evaluate(model, cache, logits, banks, seed, role, output):
    clean=margins(logits,cache['y']);clean_correct=logits.argmax(1)==cache['y'];results=[];saved={}
    for ki,kind in enumerate(a.KINDS):
        for method in METHODS:
            # New private held-out proposal stream. Every method restarts identically.
            rng_seed=seed+941003+ki*1009+(0 if role=='L_s' else 500009)
            g=a._generator(rng_seed);parts={};normalized=[];radii=[]
            for start in range(0,len(clean),BATCH):
                x=cache['x'][start:start+BATCH];y=cache['y'][start:start+BATCH]
                net=None if method=='coverage' else banks[method][kind]
                p=a._proposal(net,x,kind,g,'coverage' if method=='coverage' else 'conditional')
                ax=a.apply_proposal(x,p)
                bounds=a.old._parameter_summary(p['parameters'],kind)
                if not bounds['finite'] or not bounds['within_registered_bounds']:
                    raise ValueError('Policy exceeded registered physical bounds')
                z=model(ax);u=margins(z,y);risk=margin_risk(clean[start:start+BATCH],u)
                obs=dict(a._observations(x,ax,p),finite_risk=risk,finite_risk_active=(risk>0).float(),
                    correct=(z.argmax(1)==y).float(),clean_correct_to_error=(clean_correct[start:start+BATCH]&(z.argmax(1)!=y)).float(),
                    margin=u,margin_drop=clean[start:start+BATCH]-u,
                    raw_drop_without_violation=((clean[start:start+BATCH]>u)&(risk==0)).float())
                for key,val in obs.items():parts.setdefault(key,[]).append(val.cpu())
                normalized.append(p['normalized_parameters'].cpu());radii.append(p['radius'].cpu())
            vals={key:torch.cat(v) for key,v in parts.items()};r=torch.cat(radii);coords=torch.cat(normalized)
            if not all(torch.isfinite(v).all() for v in vals.values()):raise ValueError('Nonfinite evaluation')
            error=float((coords.abs().amax(1)-r).abs().max())
            if error>1e-6:raise ValueError('Radius mismatch')
            result=dict(role=role,kind=kind,method=method,evaluation_seed=rng_seed,radius_max_error=error,**summarize(vals,cache['y'],cache))
            saved[kind+'/'+method]=dict(values=vals,normalized_parameters=coords,radius=r)
            results.append(result)
        base=saved[kind+'/coverage']['radius']
        if not all(torch.equal(base,saved[kind+'/'+m]['radius']) for m in METHODS):raise ValueError('Unpaired radii')
    paired=[]
    for kind in a.KINDS:
        for candidate,control in [('fresh_raw','old'),('fresh_risk','old'),('fresh_risk','fresh_raw'),('fresh_risk','coverage')]:
            c=saved[kind+'/'+candidate]['values'];b=saved[kind+'/'+control]['values']
            paired.append(dict(kind=kind,candidate=candidate,control=control,
                mean_risk_difference=float((c['finite_risk']-b['finite_risk']).double().mean()),
                newly_exposed_errors=int(((c['correct']==0)&(b['correct']>0)).sum()),
                no_longer_exposed_errors=int(((c['correct']>0)&(b['correct']==0)).sum())))
    torch.save(dict(source_ids=list(cache['ids']),role=role,policies=saved,clean_logits=logits,target_access=False),output)
    return dict(role=role,metrics=results,paired=paired)


def advancement(roles):
    v=next(r for r in roles if r['role']=='V')
    means={m:sum(x['all']['finite_risk'] for x in v['metrics'] if x['method']==m)/2 for m in METHODS}
    checks={f'risk_beats_{control}':means['fresh_risk']>means[control] for control in ('old','fresh_raw','coverage')}
    return dict(passed=all(checks.values()),checks=checks,pooled_equal_LT_risk=means,
                freshness_raw_gain=means['fresh_raw']-means['old'],
                interpretation='Stronger legal source stress only; no identity performance or target claim')


def run():
    from experiments.cvs_stage2_decouple.common import verify_parent,build_model,IdentityView
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from . import evaluate as evaluator
    out=Path(d.PROJECT)/'runs'/RUN;out.mkdir(parents=True,exist_ok=False)
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Wrong registration')
    commit=(d.ROOT/'release_commit.txt').read_text().strip();spec.update(status='RUNNING',execution_commit=commit);spec['code']['commit']=commit
    d.write(out/'experiment.json',spec);torch.set_num_threads(2);started=time.perf_counter();summary=[]
    def event(status,**kw):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**kw))+'\n')
    event('RUNNING')
    try:
        with numerical_context(d.FULL_FP32_POLICY):
            for seed in d.SEEDS:
                pc=d.parent_config(seed);verify_parent(pc);folder=d.AUDIT_BASE/f's{seed}'
                prepared=d.read(folder/'preparation_complete.json')
                if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['parent_contract']!=pc or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Invalid preparation')
                packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False)
                verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(prepared['source_contract'],packed['contract'])
                for role,key,count in [('L_s','labeled',6300),('V','validation',27000)]:
                    cache=packed[key]
                    if len(cache['ids'])!=count or len(set(cache['ids']))!=count or sorted(cache['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source role mismatch')
                if set(packed['labeled']['ids'])&set(packed['validation']['ids']):raise ValueError('Overlapping roles')
                c=d.config(dict(arm='B2',model_seed=seed,row_id=f'B2-s{seed}'));evaluator.snapshot(c['row_id']);evaluator.source_provenance(c)
                path=Path(c['output_root'])/'final_ssdg.pth';ck=torch.load(path,map_location='cpu',weights_only=False);evaluator.checkpoint_provenance(c,ck);del ck
                model=IdentityView(build_model(pc,'cpu',checkpoint=path));row=out/f's{seed}';row.mkdir()
                payload=torch.load(folder/'actions.pt',map_location='cpu',weights_only=False)
                if payload['seed']!=seed or payload['source_ids']!=packed['labeled']['ids']:raise ValueError('Old policy provenance differs')
                banks={'old':a.load_actions(payload,'cpu').networks};fit_reports={}
                with ReadOnlyState(model) as frozen:
                    model.eval().requires_grad_(False)
                    clean_l=clean_forward(model,packed['labeled']);u=margins(clean_l,packed['labeled']['y'])
                    for objective in ('raw','risk'):
                        banks['fresh_'+objective],fit_reports[objective]=fit(model,packed['labeled'],u,seed,objective)
                    # Persist frozen policies before the first V forward; no V enters fitting.
                    torch.save(dict(networks={m:{k:n.state_dict() for k,n in nets.items()} for m,nets in banks.items()},
                        fit_reports=fit_reports,subject_checkpoint=str(path),seed=seed,target_access=False,diagnostic_only=True),row/'policies.pt')
                    roles=[evaluate(model,packed['labeled'],clean_l,banks,seed,'L_s',row/'labeled_metrics.pt')]
                    clean_v=clean_forward(model,packed['validation'])
                    roles.append(evaluate(model,packed['validation'],clean_v,banks,seed,'V',row/'validation_metrics.pt'))
                decision=advancement(roles)
                result=dict(seed=seed,subject_checkpoint=str(path),subject='fixed B2 E40',fit_reports=fit_reports,roles=roles,
                    clean_accuracy={role:float((logit.argmax(1)==cache['y']).double().mean()) for role,logit,cache in [('L_s',clean_l,packed['labeled']),('V',clean_v,packed['validation'])]},
                    advancement=decision,identity_state=frozen.report,identity_updates=0,auxiliary_updates=400,
                    new_identity_forwards=350100,target_access=False,validation_fit=False,execution_commit=commit)
                if sum(v['auxiliary_updates'] for v in fit_reports.values())!=400:raise ValueError('Unexpected fitting budget')
                d.write(row/'audit.json',result);d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,identity_updates=0,auxiliary_updates=400,target_access=False))
                summary.append({k:v for k,v in result.items() if k!='fit_reports'})
                print(json.dumps(dict(seed=seed,advancement=decision)),flush=True);event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
                del model,banks
        spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for r in spec['rows']:r['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(out/'experiment.json',spec);d.write(out/'summary.json',summary)
        d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,identity_updates=0,auxiliary_updates=1200,
            new_identity_forwards=1050300,target_access=False,validation_fit=False,primary_advanced=all(r['advancement']['passed'] for r in summary),
            elapsed_seconds=time.perf_counter()-started,execution_commit=commit));event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':run()
