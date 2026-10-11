"""Frozen source decision geometry: no fitting, parameter update or target use."""
from itertools import combinations
from pathlib import Path
import json,time
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_stage2_stable_rx.realized_audit import (
    HEAD,forward_features,group_means,validate_complete_cells)

RUN='20261011-phase1-stage2-decision-geometry-source-manysig-m3-r01'
RELEASE='cvs_stage2_decision_geometry_source_20261011_r01'
ARMS=('P0','B2','S3','Z3')


def centered_head(weight):
    w=F.normalize(weight.detach().cpu().double(),dim=1,eps=1e-4)
    return w-w.mean(0,keepdim=True)


def basis(weight):
    w=centered_head(weight);_,s,vh=torch.linalg.svd(w,full_matrices=False)
    rank=int((s>max(1e-12,float(s[0])*1e-10)).sum())
    if rank>5:raise ValueError('Six-class centered classifier rank exceeds five')
    q=vh[:rank].T
    if not torch.allclose(w@q@q.T,w,rtol=1e-8,atol=1e-10):raise ValueError('Classifier contrast reconstruction failed')
    return q,dict(rank=rank,singular_values=s.tolist(),threshold=max(1e-12,float(s[0])*1e-10))


def decision_geometry(current,parent,bank,head):
    current,parent=(torch.as_tensor(v).detach().cpu().double() for v in (current,parent))
    q,info=basis(head);w=centered_head(head);records=[]
    totals={scope:{key:0. for key in ('target','update','alignment','error','parent_difference','prediction_error')}
            for scope in ('full','visible','remaining')}
    for b in bank['blocks']:
        pairs=list(combinations(b['group_ids'],2));c=float(bank['folds'][b['fold']]['c']);a=float(b['alpha'])
        for g,h in pairs:
            p=(bank['groups'][g]['potential']-bank['groups'][h]['potential']).double()
            t=-a*p;u=(current[g]-current[h])-(parent[g]-parent[h]);base=parent[g]-parent[h]
            raw={}
            for scope,transform in [('full',lambda x:x),('visible',lambda x:x@q),('remaining',lambda x:x-(x@q)@q.T)]:
                ts,us,bs,ps=(transform(v) for v in (t,u,base,p))
                values=dict(target=float(ts.square().sum()),update=float(us.square().sum()),alignment=float(us@ts),
                    error=float((us-ts).square().sum()),parent_difference=float(bs.square().sum()),prediction_error=float((bs-ps).square().sum()))
                raw[scope]=values
                for key,v in values.items():totals[scope][key]+=c*v/(len(bank['blocks'])*len(pairs))
            records.append(dict(block=b['block_id'],group_ids=[g,h],c=c,alpha=a,energies=raw,
                target_centered_logit_rms=float((30*w@t).square().mean().sqrt())))
    for key in totals['full']:
        if abs(totals['full'][key]-totals['visible'][key]-totals['remaining'][key])>1e-9:
            raise ValueError('Visible/remaining energy decomposition failed')
    for scope,x in totals.items():
        x['progress']=x['alignment']/x['target'] if x['target']>1e-20 else None
        x['error_over_parent']=x['error']/x['target'] if x['target']>1e-20 else None
        x['prediction_mse_over_zero']=x['prediction_error']/x['parent_difference'] if x['parent_difference']>1e-20 else None
    return dict(head=info,energies=totals,target_visible_fraction=totals['visible']['target']/totals['full']['target'] if totals['full']['target']>1e-20 else None,
        relation_count=len(records),weighting='equal_block_pair_times_original_c',relations=records)


def fixed_rival(z,head,y):
    logits=30*z.double()@centered_head(head).T
    return logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).argmax(1)


def contrast(x,y,rival):return x.gather(1,y[:,None]).squeeze(1)-x.gather(1,rival[:,None]).squeeze(1)


def describe(x):
    x=x.double()
    return dict(count=len(x),mean=float(x.mean()),mean_abs=float(x.abs().mean()),rms=float(x.square().mean().sqrt()),
        min=float(x.min()),max=float(x.max()),q05=float(torch.quantile(x,.05)),q50=float(torch.quantile(x,.5)),q95=float(torch.quantile(x,.95)),
        positive=int((x>0).sum()),negative=int((x<0).sum()))


def pair_decisions(zs,ws,zz,wz,z0,w0,y):
    zs,zz,z0=(z.double() for z in (zs,zz,z0));ws,wz=centered_head(ws),centered_head(wz)
    rival=fixed_rival(z0,w0,y)
    logits_s=30*zs@ws.T;logits_z=30*zz@wz.T
    feature=30*(zs-zz)@((ws+wz)/2).T;head=30*((zs+zz)/2)@(ws-wz).T
    if not torch.allclose(feature+head,logits_s-logits_z,rtol=1e-8,atol=1e-10):raise ValueError('Symmetric logit decomposition failed')
    actual=lambda logits:logits.gather(1,y[:,None]).squeeze(1)-logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
    return dict(fixed_parent_rival=dict(total=describe(contrast(logits_s-logits_z,y,rival)),
        feature=describe(contrast(feature,y,rival)),head=describe(contrast(head,y,rival))),
        actual_max_rival_margin_difference=describe(actual(logits_s)-actual(logits_z)),
        changed_prediction_count=int((logits_s.argmax(1)!=logits_z.argmax(1)).sum()),
        fixed_parent_rival_changed_count=dict(
            S3=int((logits_s.masked_fill(F.one_hot(y,6).bool(),-torch.inf).argmax(1)!=rival).sum()),
            Z3=int((logits_z.masked_fill(F.one_hot(y,6).bool(),-torch.inf).argmax(1)!=rival).sum())),
        symmetric_decomposition_exact=True,normalization='already normalized joint160; no post-hoc feature transform')


def target_margins(z0,w0,cache,bank):
    target=torch.zeros_like(z0,dtype=torch.float64);seen=torch.zeros(len(z0),dtype=torch.long)
    for g in bank['groups']:
        mask=(cache['y']==g['tx'])&(cache['day']==g['day'])&(cache['rx']==g['rx'])
        target[mask]=-float(bank['blocks'][g['block_id']]['alpha'])*g['potential'].double();seen[mask]+=1
    if not seen.eq(1).all():raise ValueError('Target potential packet assignment differs')
    rival=fixed_rival(z0,w0,cache['y']);w=centered_head(w0)
    logits=30*z0.double()@w.T;delta=30*target@w.T
    m=contrast(logits,cache['y'],rival);dm=contrast(delta,cache['y'],rival)
    return dict(parent_margin=describe(m),linear_target_margin_shift=describe(dm),
        positive_parent_margin_crossed=int(((m>0)&(m+dm<=0)).sum()),
        nonpositive_parent_margin_repaired=int(((m<=0)&(m+dm>0)).sum()),
        interpretation='Algebraic group-potential addition in normalized feature coordinates, not realizable packet counterfactual or training; fixed P0 competitor')


def run():
    from experiments.cvs_stage2_decouple.common import verify_parent,build_model
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_stage2_geometry import evaluate as old_eval
    from experiments.cvs_stage2_stable_rx import evaluate as stable_eval
    from . import evaluate as anchor_eval
    from experiments.cvs_equivariant_identity.precision import numerical_context
    output=Path(d.PROJECT)/'runs'/RUN;output.mkdir(parents=True,exist_ok=False)
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Preregistered diagnostic differs')
    commit=(d.ROOT/'release_commit.txt').read_text().strip();spec.update(status='RUNNING',execution_commit=commit);spec['code']['commit']=commit
    d.write(output/'experiment.json',spec)
    def event(status,**extra):
        with (output/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**extra))+'\n')
    torch.set_num_threads(2);start=time.perf_counter();summary=[];event('RUNNING')
    try:
      with numerical_context(d.FULL_FP32_POLICY):
        for seed in d.SEEDS:
            folder=d.AUDIT_BASE/f's{seed}';pc=d.parent_config(seed);verify_parent(pc)
            prepared=d.read(folder/'preparation_complete.json')
            if prepared['parent_contract']!=pc or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Parent cache provenance differs')
            packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(prepared['source_contract'],packed['contract'])
            responses=torch.load(folder/'source_responses.pt',map_location='cpu',weights_only=False)
            caches={'L_s':packed['labeled'],'V':packed['validation']}
            if responses['source_ids']!=caches['L_s']['ids']:raise ValueError('Parent source cache order differs')
            for role,cache in caches.items():
                if len(cache['ids'])!=(6300 if role=='L_s' else 27000) or len(set(cache['ids']))!=len(cache['ids']) or sorted(cache['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source physical roles differ')
            for key in ('y','rx','day'):
                if not torch.equal(responses['validation'][key],caches['V'][key]):raise ValueError('Parent V order differs')
            vectors={'P0':{'L_s':caches['L_s']['parent_feature'],'V':responses['validation']['feature']}}
            parent=torch.load(pc['parent_checkpoint'],map_location='cpu',weights_only=False);weights={'P0':parent['model'][HEAD].clone()};del parent
            bank=torch.load(d.stable.RELEASE_BASE/'receiver_payloads'/f's{seed}.pt',map_location='cpu',weights_only=False)
            if bank['seed']!=seed or bank['method']!=d.stable.METHOD or bank['source_ids']!=caches['L_s']['ids'] or not torch.equal(bank['parent_feature'],vectors['P0']['L_s']):raise ValueError('S3 frozen RX bank differs')
            validate_complete_cells(bank,caches['L_s']);row=output/f's{seed}';row.mkdir();models=[]
            for arm,evaluator in [('B2',old_eval),('S3',stable_eval),('Z3',anchor_eval)]:
                tick=time.perf_counter();c=evaluator.snapshot(f'{arm}-s{seed}')[0];evaluator.source_provenance(c)
                path=Path(c['output_root'])/'final_ssdg.pth';ck=torch.load(path,map_location='cpu',weights_only=False);evaluator.checkpoint_provenance(c,ck)
                weights[arm]=ck['model'][HEAD].clone();del ck
                model=build_model(pc,'cpu',checkpoint=path);vectors[arm]={role:forward_features(model,cache) for role,cache in caches.items()};del model
                models.append(dict(arm=arm,checkpoint=str(path),config=c,parameters_unchanged=True,seconds=time.perf_counter()-tick))
                print(json.dumps(dict(seed=seed,arm=arm,status='SOURCE_FORWARDED',seconds=models[-1]['seconds'])),flush=True)
            measured=[];target=[];pairs=[];means={}
            for role,cache in caches.items():
                means[role]={arm:group_means(vectors[arm][role],cache,bank)[0] for arm in ARMS}
                for arm in ARMS:
                    for scope,head in [('P0',weights['P0']),('own',weights[arm])]:
                        measured.append(dict(arm=arm,role=role,head_scope=scope,**decision_geometry(means[role][arm],means[role]['P0'],bank,head)))
                measured.append(dict(arm='S3_minus_Z3',role=role,head_scope='P0',
                    **decision_geometry(means[role]['S3'],means[role]['Z3'],bank,weights['P0'])))
                pairs.append(dict(role=role,**pair_decisions(vectors['S3'][role],weights['S3'],vectors['Z3'][role],weights['Z3'],vectors['P0'][role],weights['P0'],cache['y'])))
                target.append(dict(role=role,**target_margins(vectors['P0'][role],weights['P0'],cache,bank)))
            result=dict(seed=seed,measurements=measured,pair_decisions=pairs,target_margins=target,models=models,
                source_count=6300,validation_count=27000,target_access=False,identity_updates=0,optimizer_constructed=False,
                classifier_basis='actual normalized centered head; no learned projection',execution_commit=commit)
            d.write(row/'audit.json',result);d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,target_access=False,identity_updates=0))
            torch.save(dict(means=means,heads=weights,group_order=[{k:g[k] for k in ('tx','day','rx','block_id')} for g in bank['groups']],target_access=False),row/'group_means_heads.pt')
            summary.append(dict(seed=seed,measurements=[{k:v for k,v in x.items() if k!='relations'} for x in measured],pair_decisions=pairs,target_margins=target))
            event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
        d.write(output/'summary.json',summary);spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for r in spec['rows']:r['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(output/'experiment.json',spec);d.write(output/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,
            identity_updates=0,target_access=False,models=12,forwarded_models=9,source_packets_forwarded=299700,
            elapsed_seconds=time.perf_counter()-start,execution_commit=commit));event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(output/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':run()
