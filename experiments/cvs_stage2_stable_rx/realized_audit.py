"""Source-only population geometry after fixed E40; no parameter optimization.

This measures realization of pre-existing R targets, not a new target selected
from observed test scores. All comparisons use all L_s and source V packets.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_stage2_geometry.features import identity
from experiments.cvs_stage2_geometry.receiver import _check_features

RUN='20261011-phase1-stage2-realized-geometry-source-manysig-m3-r01'
RELEASE='cvs_stage2_realized_geometry_source_20261011_r01'
ARMS=('P0','B0','B2','B3','S3')
HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'


def geometry(current,parent,payload):
    """Equal blocks then pairs; exactly the complete-population frozen R risk.

    target=-alpha*(p_g-p_h), update=(mu_g-mu_h)-(mu0_g-mu0_h).
    Progress is a signed least-squares coefficient, not an accuracy or a
    percentage constrained to [0,1]. Per-pair off-target energy is descriptive.
    """
    current=torch.as_tensor(current).detach().cpu().double()
    parent=torch.as_tensor(parent).detach().cpu().double()
    groups,blocks,folds=(payload[k] for k in ('groups','blocks','folds'))
    if current.shape!=parent.shape or current.shape!=(len(groups),160):raise ValueError('Group mean alignment differs')
    records=[];sums={k:0. for k in ('target_energy','update_energy','alignment','error_energy','parallel_energy')}
    for b in blocks:
        pairs=list(combinations(b['group_ids'],2));c=float(folds[b['fold']]['c']);alpha=float(b['alpha'])
        for g,h in pairs:
            target=-alpha*(groups[g]['potential']-groups[h]['potential']).double()
            update=(current[g]-current[h])-(parent[g]-parent[h])
            te=float(target.square().sum());ue=float(update.square().sum());alignment=float(update@target)
            err=float((update-target).square().sum());parallel=alignment**2/te if te>1e-20 else 0.
            w=1./(len(blocks)*len(pairs))
            values=dict(target_energy=te,update_energy=ue,alignment=alignment,error_energy=err,parallel_energy=parallel)
            for k,v in values.items():sums[k]+=w*c*v
            records.append(dict(block=b['block_id'],group_ids=[g,h],tx=b['tx'],day=b['day'],c=c,alpha=alpha,weight=w,**values))
    denominator=sums['target_energy']
    if abs(sums['error_energy']-(sums['update_energy']+denominator-2*sums['alignment']))>1e-10:
        raise ValueError('Population error decomposition failed')
    summary=dict(**sums,off_target_energy=sums['update_energy']-sums['parallel_energy'],
        population_risk=.5*sums['error_energy'],parent_population_risk=.5*denominator,
        progress_coefficient=sums['alignment']/denominator if denominator>1e-20 else None,
        error_over_parent=sums['error_energy']/denominator if denominator>1e-20 else None,
        anchor_term=.5*sums['update_energy'],cross_term=-sums['alignment'],constant_term=.5*denominator,
        relation_count=len(records),blocks=len(blocks),positive_reliability_relations=sum(r['c']>0 for r in records),
        weighting='equal_block_then_equal_pair_then_frozen_scalar_c',zero_signal=denominator<=1e-20)
    return dict(summary=summary,relations=records)


def group_means(features,cache,payload):
    _check_features(features,len(cache['ids']))
    masks=[];means=[];counts=[]
    for group in payload['groups']:
        mask=(cache['y']==group['tx'])&(cache['day']==group['day'])&(cache['rx']==group['rx'])
        if int(mask.sum())<2:raise ValueError('Missing complete source TX/day/RX cell')
        masks.append(mask);means.append(features[mask].double().mean(0));counts.append(int(mask.sum()))
    if not torch.stack(masks).sum(0).eq(1).all():raise ValueError('Source cells do not partition every packet once')
    return torch.stack(means),counts


def scores(z,weight,y,rx):
    logits=30*F.linear(z.double(),F.normalize(weight.double(),dim=1,eps=1e-4))
    margin=logits.gather(1,y[:,None]).squeeze(1)-logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
    pred=logits.argmax(1)
    result=dict(count=len(y),accuracy=float((pred==y).double().mean()),errors=int((pred!=y).sum()),
                margin_mean=float(margin.mean()),mean_ce=float(F.cross_entropy(logits,y)),receiver={})
    for r in sorted(rx.unique().tolist()):
        mask=rx==r;result['receiver'][str(r)]=dict(count=int(mask.sum()),accuracy=float((pred[mask]==y[mask]).double().mean()),
           margin_mean=float(margin[mask].mean()))
    return result


@torch.no_grad()
def forward_features(model,cache):
    model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()};zs=[]
    for begin in range(0,len(cache['x']),256):
        logits,z=identity(model,cache['x'][begin:begin+256],verify=True)
        zs.append(z.cpu())
    if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Read-only source forward mutated model')
    return torch.cat(zs)


def run():
    from experiments.cvs_stage2_decouple.common import verify_parent,build_model
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_stage2_geometry import evaluate as old_eval
    from . import evaluate as new_eval
    from experiments.cvs_equivariant_identity.precision import numerical_context
    output=Path(d.PROJECT)/'runs'/RUN;output.mkdir(parents=True,exist_ok=False)
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Preregistered run mismatch')
    commit=(d.ROOT/'release_commit.txt').read_text().strip();spec.update(status='RUNNING',execution_commit=commit)
    spec['code']['commit']=commit;d.write(output/'experiment.json',spec)
    def event(status,**kwargs):
        with (output/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**kwargs))+'\n')
    torch.set_num_threads(2);started=time.perf_counter();summary=[];event('RUNNING')
    try:
      with numerical_context(d.FULL_FP32_POLICY):
        for seed in d.SEEDS:
            begin=time.perf_counter();folder=d.AUDIT_BASE/f's{seed}';pc=d.parent_config(seed)
            provenance=verify_parent(pc)
            prepared=d.read(folder/'preparation_complete.json')
            if (prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['parent_contract']!=pc
                    or prepared['target_access'] or prepared['identity_updates']!=0):raise ValueError('Parent cache source provenance differs')
            packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(prepared['source_contract'],packed['contract'])
            response=torch.load(folder/'source_responses.pt',map_location='cpu',weights_only=False)
            if response['source_ids']!=packed['labeled']['ids']:raise ValueError('Source response order differs')
            caches={'L_s':packed['labeled'],'V':packed['validation']}
            for role,cache in caches.items():
                if sorted(cache['ids'])!=sorted(packed['contract']['role_ids'][role]) or len(set(cache['ids']))!=len(cache['ids']):raise ValueError('Exact source IDs differ')
                if len(cache['ids'])!=(6300 if role=='L_s' else 27000):raise ValueError('Source role counts differ')
            for k in ('y','rx','day'):
                if not torch.equal(response['validation'][k],caches['V'][k]):raise ValueError('V feature metadata order differs')
            parents={'L_s':caches['L_s']['parent_feature'],'V':response['validation']['feature']}
            parent_checkpoint=torch.load(pc['parent_checkpoint'],map_location='cpu',weights_only=False)
            parent_weight=parent_checkpoint['model'][HEAD];del parent_checkpoint
            banks={'B3':torch.load(folder/'receiver.pt',map_location='cpu',weights_only=False),
                   'S3':torch.load(d.RELEASE_BASE/'receiver_payloads'/f's{seed}.pt',map_location='cpu',weights_only=False)}
            for name,bank in banks.items():
                if bank['seed']!=seed or bank['source_ids']!=caches['L_s']['ids'] or not torch.equal(bank['parent_feature'],parents['L_s']):raise ValueError('Frozen R bank parent/IDs differ')
                if name=='S3' and bank.get('method')!=d.METHOD:raise ValueError('Stable estimator method differs')
            for a,b in zip(banks['B3']['groups'],banks['S3']['groups']):
                if any(a[k]!=b[k] for k in ('tx','rx','day','indices','block_id')):raise ValueError('Old/new R groups differ')
            if len(banks['S3']['groups'])!=90 or any(g['quality_bin']!=0 for g in banks['S3']['groups']):
                raise ValueError('Diagnostic requires actual collapsed ninety source cells')
            row=output/f's{seed}';row.mkdir();means={};measurements=[];models=[]
            for arm in ARMS:
                tick=time.perf_counter()
                if arm=='P0':
                    vectors=parents;weight=parent_weight;model=None;model_info=dict(arm=arm,checkpoint=pc['parent_checkpoint'],parent_provenance=provenance)
                else:
                    evaluator=new_eval if arm=='S3' else old_eval
                    c=evaluator.snapshot(f'{arm}-s{seed}')[0];evaluator.source_provenance(c)
                    path=Path(c['output_root'])/'final_ssdg.pth'
                    ck=torch.load(path,map_location='cpu',weights_only=False);evaluator.checkpoint_provenance(c,ck)
                    weight=ck['model'][HEAD].clone();del ck
                    model=build_model(pc,'cpu',checkpoint=path)
                    vectors={role:forward_features(model,cache) for role,cache in caches.items()}
                    model_info=dict(arm=arm,checkpoint=str(path),config=c,source_only_readback=True,parameters_unchanged=True)
                means[arm]={}
                for role,cache in caches.items():
                    z=vectors[role];m,counts=group_means(z,cache,banks['S3']);means[arm][role]=m
                    metrics=dict(actual=scores(z,weight,cache['y'],cache['rx']),
                        current_features_parent_head=scores(z,parent_weight,cache['y'],cache['rx']),
                        parent_features_current_head=scores(parents[role],weight,cache['y'],cache['rx']))
                    for bank_name,bank in banks.items():
                        result=geometry(m,means['P0'][role],bank)
                        measurements.append(dict(arm=arm,role=role,bank=bank_name,**result))
                    d.write(row/f'{arm}_{role}_identity.json',metrics)
                model_info.update(seconds=time.perf_counter()-tick,source_packets_forwarded=0 if arm=='P0' else 33300)
                models.append(model_info);print(json.dumps(dict(seed=seed,arm=arm,status='SOURCE_FORWARDED',seconds=model_info['seconds'])),flush=True)
                if model is not None:del model
                del vectors
            contrasts=[]
            for arm,control in [('B3','B2'),('S3','B2'),('S3','B3'),('B2','B0')]:
                for role in caches:
                    for bank_name,bank in banks.items():
                        contrasts.append(dict(arm=arm,control=control,role=role,bank=bank_name,
                            **geometry(means[arm][role],means[control][role],bank)))
            torch.save(dict(group_means=means,group_order=[{k:g[k] for k in ('tx','rx','day','block_id')} for g in banks['S3']['groups']],
                no_packet_features_saved=True,target_access=False),row/'source_group_means.pt')
            audit=dict(seed=seed,measurements=measurements,contrasts=contrasts,models=models,
                source_count=6300,validation_count=27000,identity_updates=0,target_access=False,
                selection=False,optimizer_constructed=False,execution_commit=commit,elapsed_seconds=time.perf_counter()-begin)
            d.write(row/'audit.json',audit);d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,target_access=False,identity_updates=0))
            summary.append(dict(seed=seed,measurements=[{k:v for k,v in x.items() if k!='relations'} for x in measurements],
                contrasts=[{k:v for k,v in x.items() if k!='relations'} for x in contrasts]))
            event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
        d.write(output/'summary.json',summary)
        spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for r in spec['rows']:r['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(output/'experiment.json',spec)
        d.write(output/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,identity_updates=0,
            target_access=False,models=15,forwarded_models=12,source_packets_forwarded=399600,
            elapsed_seconds=time.perf_counter()-started,execution_commit=commit))
        event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(output/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':
    run()
