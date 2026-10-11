"""One-anchor conditional source RX transport, independently scored outcomes."""
from itertools import combinations
from pathlib import Path
import hashlib,json,time
import torch
from torch.nn import functional as F
from . import design as d

RUN='20261011-phase1-stage2-conditional-transport-source-manysig-m3-r01'
RELEASE='cvs_stage2_conditional_transport_source_20261011_r01'
SALT='source-relation-split-v1'
METHODS=('zero','shared_rx','conditional_joint','conditional_state')
FOLDS=((0,1),(2,3),(4,5))
RIDGE=.1
SHRINK=.4
HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'


def source_observations(cache):
    """Only L_s observations: V has no input path into this constructor."""
    ys=sorted(cache['y'].unique().tolist());ds=sorted(cache['day'].unique().tolist());rs=sorted(cache['rx'].unique().tolist())
    if ys!=list(range(6)) or len(ds)!=3 or len(rs)!=5:raise ValueError('Expected6TX/3day/5RX')
    ids=cache['ids']
    if len(ids)!=6300 or len(set(ids))!=len(ids):raise ValueError('L_s IDs differ')
    z=cache['parent_feature'].double();state=cache['state'].double()
    if z.shape!=(6300,160) or state.shape!=(6300,40) or not torch.isfinite(z).all() or not torch.isfinite(state).all():raise ValueError('Invalid parent/state features')
    if not torch.allclose(z.norm(dim=1),torch.ones(6300,dtype=torch.double),atol=2e-5,rtol=0):raise ValueError('joint160 not normalized')
    observations={};splits=[]
    for y in ys:
        for day in ds:
            for rx in rs:
                indices=torch.where((cache['y']==y)&(cache['day']==day)&(cache['rx']==rx))[0].tolist()
                if len(indices)!=70:raise ValueError('Expected70 distinct packets per complete cell')
                indices=sorted(indices,key=lambda i:hashlib.sha256((SALT+'|'+str(ids[i])).encode('utf-8')).digest())
                a,b=indices[:35],indices[35:]
                observations[y,day,rx]=dict(jointA=z[a].mean(0),jointB=z[b].mean(0),stateA=state[a].mean(0),stateB=state[b].mean(0))
                splits.append(dict(tx=y,day=day,rx=rx,A_indices=a,B_indices=b))
    return observations,dict(txs=ys,days=ds,rxs=rs,splits=splits)


def fit_affine(x,y):
    x=x.double();y=y.double()
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)<2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Invalid affine fit')
    center=x.mean(0);target=y.mean(0);dx=x-center;dy=y-target
    scale=float(dx.square().sum(1).mean().sqrt());weight=x.new_zeros((x.shape[1],y.shape[1]))
    if scale>1e-12:
        xx=dx/scale;n=len(x)
        dual=torch.linalg.solve(xx@xx.T/n+RIDGE*torch.eye(n,dtype=x.dtype),dy/n)
        weight=xx.T@dual
    else:scale=1.
    return dict(center=center,scale=scale,mean=target,weight=weight,fit_count=len(x))


def predict_affine(model,x):
    return SHRINK*(model['mean']+(x.double()-model['center'])/model['scale']@model['weight'])


def fit_transport(fit,rxs):
    """The supplied mapping already excludes both held TX and held day."""
    cells=sorted({(y,day) for y,day,rx in fit});models={}
    if len(cells)!=8 or len(fit)!=40:raise ValueError('Expected4TX x2day x5RX fitting cells')
    for anchor in rxs:
        for destination in rxs:
            if destination==anchor:continue
            xj=[];xs=[];target=[]
            for y,day in cells:
                a=fit[y,day,anchor];b=fit[y,day,destination]
                for side,other in [('A','B'),('B','A')]:
                    xj.append(a['joint'+side]);xs.append(a['state'+side]);target.append(b['joint'+other]-a['joint'+other])
            target=torch.stack(target)
            models[anchor,destination]=dict(shared_rx=SHRINK*target.mean(0),conditional_joint=fit_affine(torch.stack(xj),target),conditional_state=fit_affine(torch.stack(xs),target))
    return models


def star_predictions(models,anchor,input_joint,input_state,rxs):
    """No destination observation, class label, day or V argument is accepted."""
    predictions={name:torch.zeros(len(rxs),160,dtype=torch.double) for name in METHODS}
    for i,destination in enumerate(rxs):
        if destination==anchor:continue
        entry=models[anchor,destination];predictions['shared_rx'][i]=entry['shared_rx']
        predictions['conditional_joint'][i]=predict_affine(entry['conditional_joint'],input_joint)
        predictions['conditional_state'][i]=predict_affine(entry['conditional_state'],input_state)
    return predictions


def frozen_predictions(observations,layout,head):
    head=F.normalize(head.double(),dim=1,eps=1e-4);records=[];saved={};folds=[];rxs=layout['rxs']
    for held_tx in FOLDS:
        for held_day in layout['days']:
            fit={k:v for k,v in observations.items() if k[0] not in held_tx and k[1]!=held_day}
            models=fit_transport(fit,rxs);fi=len(folds);saved[fi]=models
            folds.append(dict(fold=fi,held_tx=list(held_tx),held_day=held_day,fit_tx=sorted({k[0] for k in fit}),fit_days=sorted({k[1] for k in fit}),fit_cells=[list(k) for k in sorted(fit)],fit_packets=2800,
                affine_fits=40,records_per_affine_fit=16,shared_cells_per_pair=8))
            for y in held_tx:
                for anchor in rxs:
                    observation=observations[y,held_day,anchor];x=observation['jointA']
                    outputs=star_predictions(models,anchor,x,observation['stateA'],rxs)
                    logit=30*head@x;logit[y]=-torch.inf;rival=int(logit.argmax())
                    records.append(dict(fold=fi,tx=y,day=held_day,anchor=anchor,rival=rival,predictions=outputs))
    return records,saved,folds


def validation_outcomes(validation,layout):
    if validation['feature'].shape!=(27000,160) or not torch.isfinite(validation['feature']).all():raise ValueError('Invalid V joint160')
    outcomes={}
    for y in layout['txs']:
        for day in layout['days']:
            for rx in layout['rxs']:
                mask=(validation['y']==y)&(validation['day']==day)&(validation['rx']==rx)
                if int(mask.sum())!=300:raise ValueError('Expected300 V packets per cell')
                outcomes[y,day,rx]=validation['feature'][mask].double().mean(0)
    return outcomes


def score_predictions(predictions,outcomes,rxs,head,role):
    w=F.normalize(head.double(),dim=1,eps=1e-4);wc=w-w.mean(0);details=[]
    for record in predictions:
        y=record['tx'];day=record['day'];rival=record['rival'];v=30*(w[y]-w[rival])
        for i,j in combinations(range(len(rxs)),2):
            truth=outcomes[y,day,rxs[i]]-outcomes[y,day,rxs[j]];true_logit=30*wc@truth;true_margin=v@truth
            for method in METHODS:
                p=record['predictions'][method][i]-record['predictions'][method][j]
                logit=30*wc@p;margin=v@p;denom=float(p.norm()*truth.norm())
                details.append(dict(role=role,fold=record['fold'],tx=y,day=day,anchor=record['anchor'],rx_pair=[rxs[i],rxs[j]],method=method,rival=rival,
                    full_error=float((p-truth).square().sum()),full_zero=float(truth.square().sum()),
                    logit_error=float((logit-true_logit).square().sum()),logit_zero=float(true_logit.square().sum()),
                    margin_error=float((margin-true_margin).square()),margin_zero=float(true_margin.square()),
                    predicted_norm=float(p.norm()),true_norm=float(truth.norm()),alignment=float(p@truth),
                    cosine=float(p@truth)/denom if denom>1e-12 else 0.))
    def aggregate(rows):
        out=dict(relations=len(rows))
        for space in ('full','logit','margin'):
            error=sum(x[space+'_error'] for x in rows)/len(rows);zero=sum(x[space+'_zero'] for x in rows)/len(rows)
            out[space]=dict(mse=error,zero=zero,ratio=error/zero if zero>1e-12 else None)
        out.update(max_predicted_norm=max(x['predicted_norm'] for x in rows),mean_direction_cosine=sum(x['cosine'] for x in rows)/len(rows))
        return out
    metrics={method:aggregate([r for r in details if r['method']==method]) for method in METHODS}
    folds=[dict(fold=fi,method=method,**aggregate([r for r in details if r['fold']==fi and r['method']==method])) for fi in range(9) for method in METHODS]
    return dict(role=role,metrics=metrics,folds=folds,relations=details)


def advancement(ls,validation):
    p=validation['metrics']['conditional_joint'];b=validation['metrics']['shared_rx']
    checks={space+'_V_better_than_shared_and_zero':p[space]['mse']<min(b[space]['mse'],p[space]['zero']) for space in ('full','logit','margin')}
    checks['L_s_B_full_better_than_shared']=ls['metrics']['conditional_joint']['full']['mse']<ls['metrics']['shared_rx']['full']['mse']
    return dict(checks=checks,passed=all(checks.values()),primary='conditional_joint',alternative_selection=False)


def run():
    from experiments.cvs_stage2_decouple.common import verify_parent
    from experiments.cvs_phase1_stack.recover import verify_contract
    out=Path(d.PROJECT)/'runs'/RUN;out.mkdir(parents=True,exist_ok=False);spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=RUN:raise ValueError('Wrong registration')
    commit=(d.ROOT/'release_commit.txt').read_text().strip();spec.update(status='RUNNING',execution_commit=commit);spec['code']['commit']=commit;d.write(out/'experiment.json',spec)
    torch.set_num_threads(2);started=time.perf_counter();summary=[]
    def event(status,**kwargs):
        with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(at=time.time(),status=status,**kwargs))+'\n')
    event('RUNNING')
    try:
        for seed in d.SEEDS:
            pc=d.parent_config(seed);verify_parent(pc);folder=d.AUDIT_BASE/f's{seed}';prepared=d.read(folder/'preparation_complete.json')
            if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['parent_contract']!=pc or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Invalid source preparation provenance')
            packed=torch.load(folder/'source_cache.pt',map_location='cpu',weights_only=False);response=torch.load(folder/'source_responses.pt',map_location='cpu',weights_only=False)
            verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(prepared['source_contract'],packed['contract']);cache=packed['labeled'];valid=packed['validation']
            for role,c,count in [('L_s',cache,6300),('V',valid,27000)]:
                if len(c['ids'])!=count or len(set(c['ids']))!=count or sorted(c['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source physical roles differ')
            if set(cache['ids'])&set(valid['ids']) or response['source_ids']!=cache['ids']:raise ValueError('Source role overlap/order differs')
            for key in ('y','rx','day'):
                if not torch.equal(response['validation'][key],valid[key]):raise ValueError('V cache order differs')
            checkpoint=torch.load(pc['parent_checkpoint'],map_location='cpu',weights_only=False);head=checkpoint['model'][HEAD].clone();del checkpoint
            logit=30*cache['parent_feature'].double()@F.normalize(head.double(),dim=1,eps=1e-4).T
            margin=logit.gather(1,cache['y'][:,None]).squeeze(1)-logit.masked_fill(F.one_hot(cache['y'],6).bool(),-torch.inf).max(1).values
            if not torch.allclose(margin,cache['parent_margin'].double(),atol=2e-4,rtol=2e-5):raise ValueError('Parent interface differs')
            observations,layout=source_observations(cache);row=out/f's{seed}';row.mkdir()
            predictions,models,folds=frozen_predictions(observations,layout,head)
            # Predictions fixed before connecting either independent outcome.
            torch.save(dict(predictions=predictions,models=models,layout=layout,target_access=False,validation_input=False),row/'transport_models.pt')
            ls_out={k:v['jointB'] for k,v in observations.items()}
            ls=score_predictions(predictions,ls_out,layout['rxs'],head,'L_s_B')
            vout=validation_outcomes(response['validation'],layout);vs=score_predictions(predictions,vout,layout['rxs'],head,'V')
            decision=advancement(ls,vs)
            result=dict(seed=seed,folds=folds,splits=layout,roles=[ls,vs],advancement=decision,identity_updates=0,new_identity_forwards=0,target_access=False,validation_fit=False,validation_input=False,execution_commit=commit)
            d.write(row/'audit.json',result);d.write(row/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',seed=seed,identity_updates=0,target_access=False))
            summary.append(dict(seed=seed,roles=[{k:v for k,v in part.items() if k!='relations'} for part in (ls,vs)],advancement=decision))
            print(json.dumps(dict(seed=seed,advancement=decision,source_complete=True)),flush=True);event('SOURCE_DIAGNOSTIC_COMPLETE',row_id=f's{seed}')
        spec['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        for row in spec['rows']:row['status']='SOURCE_DIAGNOSTIC_COMPLETE'
        d.write(out/'experiment.json',spec);d.write(out/'summary.json',summary)
        d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows_completed=3,identity_updates=0,new_identity_forwards=0,target_access=False,validation_fit=False,validation_input=False,
            affine_fits=1080,primary_advanced=all(r['advancement']['passed'] for r in summary),elapsed_seconds=time.perf_counter()-started,execution_commit=commit));event('SOURCE_DIAGNOSTIC_COMPLETE')
    except Exception as exc:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(exc)));event('FAILED',error=repr(exc));raise


if __name__=='__main__':run()
