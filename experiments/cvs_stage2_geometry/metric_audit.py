"""Fixed full-dimensional source metric probes; no identity or target update."""
from itertools import combinations
from pathlib import Path
import json,time
import torch
from torch.nn import functional as F
from . import design as d
from .geometry_audit import OMEGA,RIDGE

RUN='20261011-phase1-stage2-soft-metric-source-manysig-m3-r01'
RELEASE='cvs_stage2_soft_metric_source_20261011_r01'
METHODS=('isotropic','packet','packet_L','packet_T','packet_R','packet_LTR')
HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'


def validate_features(z,y,rx,day,j):
    if z.ndim!=2 or z.shape[1]!=160 or j.shape!=(len(z),160,8):raise ValueError('Expected joint160 and eight local responses')
    if not torch.isfinite(z).all() or not torch.isfinite(j).all():raise ValueError('Nonfinite source feature')
    if not torch.allclose(z.norm(dim=1),z.new_ones(len(z)),rtol=0,atol=2e-5):raise ValueError('Features not normalized')
    if set(y.tolist())!=set(range(6)) or min(len(rx.unique()),len(day.unique()))<2:raise ValueError('Complete source classes/multiple conditions required')
    for c in (y,rx,day):
        if c.shape!=(len(z),) or not torch.equal(c,c.long()):raise ValueError('Invalid source role')


def fit_components(z,y,rx,day,j):
    """Every argument contains fitting L_s rows only; no validation inputs."""
    z=z.double();j=j.double();validate_features(z,y,rx,day,j)
    ys=list(range(6));rs=sorted(rx.unique().tolist());ds=sorted(day.unique().tolist())
    means={};within=[];counts=[]
    for a in ys:
        for b in ds:
            for c in rs:
                mask=(y==a)&(day==b)&(rx==c);v=z[mask]
                if len(v)<2:raise ValueError('Missing source cell')
                mu=v.mean(0);means[a,b,c]=mu;delta=v-mu
                within.append(delta.T@delta/len(v));counts.append(len(v))
    if len(set(counts))!=1:raise ValueError('Probe requires balanced full physical cells')
    mu=torch.stack([torch.stack([means[a,b,c] for b in ds for c in rs]).mean(0) for a in ys])
    cov=dict(packet=torch.stack(within).mean(0))
    for name,start in [('L',0),('T',4)]:cov[name]=OMEGA*torch.einsum('ndk,nek->de',j[:,:,start:start+4],j[:,:,start:start+4])/len(j)
    centered={(a,b,c):means[a,b,c]-torch.stack([means[a,b,r] for r in rs]).mean(0) for a in ys for b in ds for c in rs}
    cr=torch.zeros(160,160,dtype=torch.double);folds=[]
    for held_tx in ((0,1),(2,3),(4,5)):
        for held_day in ds:
            fit_y=[a for a in ys if a not in held_tx];fit_day=[b for b in ds if b!=held_day]
            offsets={r:.4*torch.stack([centered[a,b,r] for a in fit_y for b in fit_day]).mean(0) for r in rs}
            truths=[];predictions=[]
            for a in held_tx:
                for r,s in combinations(rs,2):
                    truths.append(means[a,held_day,r]-means[a,held_day,s]);predictions.append(offsets[r]-offsets[s])
            truth=torch.stack(truths);pred=torch.stack(predictions)
            zero=float(truth.square().sum(1).mean());error=float((pred-truth).square().sum(1).mean())
            c=max(0.,min(1.,1-error/zero)) if zero>1e-12 else 0.
            cr+=c*(pred.T@pred/len(pred))/(3*len(ds))
            folds.append(dict(held_tx=list(held_tx),held_day=held_day,fit_tx=fit_y,fit_days=fit_day,c=c,mse=error,mse_zero=zero,pairs=len(pred)))
    cov['R']=cr
    return mu,cov,dict(groups=len(means),packets_per_group=counts[0],rxs=rs,days=ds,folds=folds,
        covariance_traces={k:float(v.trace()) for k,v in cov.items()},R_scope='cross-TX and cross-day predictions entirely inside this outer fitting subset; shrink=.4, full-vector reliability; no validation')


def make_metric(cov,method):
    eye=torch.eye(160,dtype=torch.double)
    if method=='isotropic':return eye,dict(min_eigenvalue=1.,max_eigenvalue=1.,condition_number=1.)
    if method not in METHODS:raise ValueError('Unknown fixed metric')
    c=cov['packet']+RIDGE*eye
    for name in ('L','T','R'):
        if method=='packet_'+name or method=='packet_LTR':c=c+cov[name]
    values,vectors=torch.linalg.eigh((c+c.T)/2)
    if float(values.min())<=0:raise ValueError('Nonpositive metric')
    transform=(vectors*values.rsqrt()[None])@vectors.T
    if not torch.allclose(transform@c@transform,eye,atol=2e-8,rtol=2e-8):raise ValueError('Whitening reconstruction failed')
    return transform,dict(min_eigenvalue=float(values.min()),max_eigenvalue=float(values.max()),condition_number=float(values.max()/values.min()))


def logits(z,centers,transform):
    # Test-time rows are independent. No RX/day/label inputs or fitted test state.
    return 30*F.normalize(z.double()@transform,dim=1,eps=1e-12)@F.normalize(centers@transform,dim=1,eps=1e-12).T


def score(actual,y,parent,rx,day):
    pred=actual.argmax(1);base=parent.argmax(1)
    def part(mask):
        p=pred[mask];yy=y[mask];old=base[mask]
        cm=torch.bincount(yy*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
        return dict(count=len(yy),correct=int((p==yy).sum()),accuracy=float((p==yy).double().mean()),
            macro_f1=float((2*cm.diag().double()/den.clamp_min(1)).mean()),
            broken=int(((old==yy)&(p!=yy)).sum()),repaired=int(((old!=yy)&(p==yy)).sum()),confusion=cm.tolist())
    result=dict(overall=part(torch.ones(len(y),dtype=torch.bool)))
    for name,column in [('rx',rx),('day',day),('tx',y)]:result[name]={str(v):part(column==v) for v in sorted(column.unique().tolist())}
    return result


def evaluate_fold(cache,response,head,kind,held):
    n=len(cache['y']);mask=torch.ones(n,dtype=torch.bool) if kind=='all' else cache[kind]!=held
    valid=response['validation'];vmask=torch.ones(len(valid['y']),dtype=torch.bool) if kind=='all' else valid[kind]==held
    if kind!='all' and bool((cache[kind][mask]==held).any()):raise ValueError('Held condition leaked into fit')
    z=cache['parent_feature'];y=cache['y'];j=response['jacobian']
    centers,cov,fit=fit_components(z[mask],y[mask],cache['rx'][mask],cache['day'][mask],j[mask])
    vz=valid['feature'][vmask];vy=valid['y'][vmask];vr=valid['rx'][vmask];vd=valid['day'][vmask]
    parent=30*vz.double()@F.normalize(head.double(),dim=1,eps=1e-4).T
    rows=[dict(method='parent',metrics=score(parent,vy,parent,vr,vd))];parameters={}
    for method in METHODS:
        transform,diag=make_metric(cov,method);out=logits(vz,centers,transform)
        rows.append(dict(method=method,metric=diag,metrics=score(out,vy,parent,vr,vd)))
        parameters[method]=dict(centers=centers,transform=transform)
    fit_ids=[v for v,m in zip(cache['ids'],mask.tolist()) if m]
    return dict(kind=kind,held=held,fit_count=int(mask.sum()),validation_count=int(vmask.sum()),fit=fit,rows=rows,fit_ids=fit_ids),parameters


def run():
    from experiments.cvs_stage2_decouple.common import verify_parent
    from experiments.cvs_phase1_stack.recover import verify_contract
    out=Path(d.PROJECT)/'runs'/RUN;out.mkdir(parents=True,exist_ok=False)
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Wrong registration')
    commit=(d.ROOT/'release_commit.txt').read_text().strip();spec.update(status='RUNNING',execution_commit=commit);spec['code']['commit']=commit
    d.write(out/'experiment.json',spec);torch.set_num_threads(2);started=time.perf_counter();summary=[]
    def event(status,**kwargs):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**kwargs))+'\n')
    event('RUNNING')
    try:
        for seed in d.SEEDS:
            pc=d.parent_config(seed);verify_parent(pc);folder=d.AUDIT_BASE/f's{seed}';prepared=d.read(folder/'preparation_complete.json')
            if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['parent_contract']!=pc or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Invalid parent cache provenance')
            packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False);response=torch.load(folder/'source_responses.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(prepared['source_contract'],packed['contract'])
            cache=packed['labeled'];valid=packed['validation']
            for role,c,count in [('L_s',cache,6300),('V',valid,27000)]:
                if len(c['ids'])!=count or len(set(c['ids']))!=count or sorted(c['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source role IDs differ')
            if set(cache['ids'])&set(valid['ids']) or response['source_ids']!=cache['ids']:raise ValueError('Physical role overlap/order differs')
            for k in ('y','rx','day'):
                if not torch.equal(response['validation'][k],valid[k]):raise ValueError('Validation cached order differs')
            checkpoint=torch.load(pc['parent_checkpoint'],map_location='cpu',weights_only=False);head=checkpoint['model'][HEAD].clone();del checkpoint
            parent=30*cache['parent_feature'].double()@F.normalize(head.double(),dim=1,eps=1e-4).T
            margin=parent.gather(1,cache['y'][:,None]).squeeze(1)-parent.masked_fill(F.one_hot(cache['y'],6).bool(),-torch.inf).max(1).values
            if not torch.allclose(margin,cache['parent_margin'].double(),rtol=2e-5,atol=2e-4):raise ValueError('Actual parent feature interface differs')
            cases=[('all',None)]+[(kind,v) for kind in ('rx','day') for v in sorted(cache[kind].unique().tolist())]
            row=out/f's{seed}';row.mkdir();results=[];parameters={}
            for kind,held in cases:
                result,param=evaluate_fold(cache,response,head,kind,held);results.append(result);parameters[f'{kind}:{held}']=param
                print(json.dumps(dict(seed=seed,condition=kind,held=held,fit=result['fit_count'],validation=result['validation_count'])),flush=True)
            result=dict(seed=seed,cases=results,identity_updates=0,target_access=False,validation_fit=False,
                independent_unknown_RX_claim=False,parent_seen_all_source_conditions=True,head_readout_fits=len(cases)*len(METHODS),new_identity_forwards=0,execution_commit=commit)
            d.write(row/'audit.json',result);torch.save(parameters,row/'metric_readouts.pt')
            d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,identity_updates=0,target_access=False))
            summary.append({**result,'cases':[{k:v for k,v in case.items() if k!='fit_ids'} for case in results]})
            event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
        spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(out/'experiment.json',spec);d.write(out/'summary.json',summary)
        d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,identity_updates=0,target_access=False,
            new_identity_forwards=0,validation_fit=False,metric_readout_fits=162,elapsed_seconds=time.perf_counter()-started,execution_commit=commit));event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':run()
