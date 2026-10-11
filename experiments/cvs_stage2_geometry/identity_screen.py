"""Frozen source-V identity preservation under L-only stable RX subtraction.

This is a feature counterfactual diagnostic, not an inference algorithm or a
claim about a subsequently optimized network. V labels only score frozen logits.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from . import design as d
from .receiver import _source,_build_groups,_check_features
from .conditional_audit import center_blocks

RUN='20261011-phase1-stage2-stable-rx-identity-source-manysig-m3-r01'
RELEASE='cvs_stage2_stable_rx_identity_source_20261011_r01'
SHRINK=.4
PRIMARY_ALPHA=.25
ALPHAS=(0.,PRIMARY_ALPHA,1.)


def estimate_rx(means,groups,blocks,held_day,fit_tx=None):
    """Equal source cell means; no validation labels/features or held day fit."""
    fit_tx=set(range(6)) if fit_tx is None else set(fit_tx)
    centered=center_blocks(means,blocks);buckets={}
    for i,g in enumerate(groups):
        if g['tx'] in fit_tx and g['day']!=held_day:
            buckets.setdefault(int(g['rx']),[]).append(centered[i])
    if not buckets:raise ValueError('No fitting source groups')
    return {rx:SHRINK*torch.stack(values).mean(0) for rx,values in buckets.items()}


def intervention(features,rx,day,bank,alpha):
    offsets=torch.stack([bank[int(dy)][int(r)] for r,dy in zip(rx.tolist(),day.tolist())]).to(features)
    modified=features-alpha*offsets
    if modified.norm(dim=1).min()<1e-4:raise ValueError('Degenerate identity normalization')
    return F.normalize(modified,dim=1,eps=1e-4)


def logits_from_features(features,weight):
    return 30.*F.linear(features,F.normalize(weight,dim=1,eps=1e-4))


def identity_safe(parent,primary):
    # Integer comparisons implement exactly 0.2pp overall / 0.5pp per RX.
    total=(primary['overall']['errors']-parent['overall']['errors'])*1000<=2*parent['overall']['count']
    receivers=all((primary['receiver'][r]['errors']-p['errors'])*1000<=5*p['count']
                  for r,p in parent['receiver'].items())
    return total and receivers


def performance(logits,labels,parent_logits,rx,day):
    def values(mask):
        actual=logits[mask];y=labels[mask];parent=parent_logits[mask]
        pred=actual.argmax(1);p=parent.argmax(1)
        margin=actual.gather(1,y[:,None]).squeeze(1)-actual.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
        pm=parent.gather(1,y[:,None]).squeeze(1)-parent.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
        return dict(count=len(y),accuracy=float((pred==y).double().mean()),
            errors=int((pred!=y).sum()),broken=int(((p==y)&(pred!=y)).sum()),
            repaired=int(((p!=y)&(pred==y)).sum()),margin_mean=float(margin.mean()),
            margin_delta_mean=float((margin-pm).mean()),
            parent_correct_margin_drop_mean=float((pm-margin)[p==y].mean()) if (p==y).any() else None,
            finite_keep_risk=float((.5*((pm.clamp_max(d.MARGIN_REQUIRED)-margin)/d.MARGIN_REQUIRED).clamp_min(0).square()*(pm>0)).mean()))
    out=dict(overall=values(torch.ones(len(labels),dtype=torch.bool)),receiver={},day={},tx={})
    for name,column in [('receiver',rx),('day',day),('tx',labels)]:
        for key in sorted(column.unique().tolist()):out[name][str(key)]=values(column==key)
    return out


def analyze(cache,validation,weight):
    data=_source(cache);z=cache['parent_feature'].detach().cpu().double();_check_features(z,len(data['ids']))
    weight=weight.detach().cpu().double()
    if weight.shape!=(6,160) or not torch.isfinite(weight).all():raise ValueError('Original cosine weight shape differs')
    check=logits_from_features(z,weight);y=data['y']
    margin=check.gather(1,y[:,None]).squeeze(1)-check.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
    if not torch.allclose(margin,data['parent_margin'].double(),atol=2e-4,rtol=2e-5):
        raise ValueError('Cached joint160 / original cosine head does not reproduce L_s parent')
    groups,blocks,_,_=_build_groups(data)
    means=torch.stack([z[g['indices']].mean(0) for g in groups])
    bank={day:estimate_rx(means,groups,blocks,day) for day in sorted(data['day'].unique().tolist())}
    vz=validation['feature'].double();vy=validation['y'].long();vr=validation['rx'].long();vd=validation['day'].long()
    _check_features(vz,len(vy));baseline=logits_from_features(vz,weight)
    rows=[]
    for alpha in ALPHAS:
        modified=intervention(vz,vr,vd,bank,alpha)
        result=performance(logits_from_features(modified,weight),vy,baseline,vr,vd)
        result.update(alpha=alpha,feature_displacement_mean=float((modified-vz).norm(dim=1).mean()),
            labels_used_in_transform=False,validation_fit=False)
        rows.append(result)
    parent=rows[0];primary=rows[1]
    delta=primary['overall']['accuracy']-parent['overall']['accuracy']
    min_rx=min(primary['receiver'][r]['accuracy']-parent['receiver'][r]['accuracy'] for r in parent['receiver'])
    return dict(source_count=len(z),validation_count=len(vy),source_groups=len(groups),
        rows=rows,primary_alpha=PRIMARY_ALPHA,source_shrink=SHRINK,
        decision=dict(accuracy_delta=delta,min_receiver_delta=min_rx,
            no_material_identity_damage=identity_safe(parent,primary),
            registered_tolerances=dict(overall_drop=.002,any_receiver_drop=.005)),
        source_head_margin_max_error=float((margin-data['parent_margin']).abs().max()),
        target_access=False,identity_updates=0,validation_fit=False,
        interpretation='known source RX/day feature counterfactual only; neither target inference nor realized training performance'),bank


def run(output=None):
    from experiments.cvs_stage2_decouple.common import verify_parent
    from experiments.cvs_phase1_stack.recover import verify_contract
    out=Path(output) if output else Path(d.PROJECT)/'runs'/RUN;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();commit=(d.ROOT/'release_commit.txt').read_text().strip()
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Preregistration mismatch')
    spec.update(status='RUNNING',execution_commit=commit);spec['code']['commit']=commit;d.write(out/'experiment.json',spec)
    def event(status,**kwargs):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**kwargs))+'\n')
    event('RUNNING');summary=[]
    try:
        for seed in d.SEEDS:
            begin=time.perf_counter();source=d.AUDIT_BASE/f's{seed}';done=d.read(source/'preparation_complete.json');c=d.parent_config(seed)
            if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['target_access'] or done['identity_updates'] or done['parent_contract']!=c:raise ValueError('Preparation provenance mismatch')
            provenance=verify_parent(c)
            packed=torch.load(source/'source_cache.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE));cache=packed['labeled'];vc=packed['validation']
            for role,observations in [('L_s',cache),('V',vc)]:
                if sorted(observations['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError(role+' IDs differ')
            response=torch.load(source/'source_responses.pt',map_location='cpu',weights_only=False)
            if response['source_ids']!=cache['ids']:raise ValueError('Source response IDs differ')
            validation=response['validation']
            for key in ('y','rx','day'):
                if not torch.equal(validation[key],vc[key]):raise ValueError('V ordering differs')
            checkpoint=torch.load(c['parent_checkpoint'],map_location='cpu',weights_only=False)
            key='id_backbone.encoder.core.id_backbone.cls_head.weight'
            result,bank=analyze(cache,validation,checkpoint['model'][key]);row=out/f's{seed}';row.mkdir()
            result.update(seed=seed,parent_provenance=provenance,input_source=str(source),execution_commit=commit,
                elapsed_seconds=time.perf_counter()-begin)
            d.write(row/'audit.json',result);torch.save(bank,row/'source_rx_offsets.pt')
            d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,target_access=False,identity_updates=0,audit=str(row/'audit.json')))
            summary.append(dict(seed=seed,decision=result['decision'],rows=[dict(alpha=r['alpha'],**r['overall']) for r in result['rows']]))
            print(json.dumps(summary[-1]),flush=True);event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
        d.write(out/'summary.json',summary);spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(out/'experiment.json',spec);d.write(out/'completion.json',dict(run_id=RUN,status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,
            target_access=False,identity_updates=0,execution_commit=commit,elapsed_seconds=time.perf_counter()-started))
        event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();run(a.output)
