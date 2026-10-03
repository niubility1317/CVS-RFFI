"""Recount paired frozen logits and descriptive source-label geometry."""
import numpy as np


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-4)


def summarize(x):
    if not len(x):return dict(count=0,mean=None)
    return dict(count=len(x),mean=float(np.mean(x)))


def classification(logits,truth):
    logits=logits.astype(np.float64);n=len(truth);pred=logits.argmax(1)
    shift=logits-logits.max(1,keepdims=True)
    ce=np.log(np.exp(shift).sum(1))-shift[np.arange(n),truth]
    other=logits.copy();other[np.arange(n),truth]=-np.inf
    margin=logits[np.arange(n),truth]-other.max(1)
    return pred,ce,margin


def scores(pred,truth):
    cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(truth,pred),1)
    tp=np.diag(cm);denom=cm.sum(0)+cm.sum(1)
    return dict(count=len(truth),accuracy=float(np.mean(pred==truth)),macro_f1=float(np.mean(2*tp/np.maximum(denom,1))),confusion=cm.tolist())


def scatter(z,truth,rx,day):
    z=unit(z.astype(float));grand=z.mean(0);total=float(np.square(z-grand).sum());between=0.;domain=0.;within=0.
    for tx in sorted(set(truth.tolist())):
        mask=truth==tx;mu=z[mask].mean(0);between+=int(mask.sum())*float(np.square(mu-grand).sum())
        for r in sorted(set(rx[mask].tolist())):
            for d in sorted(set(day[mask&(rx==r)].tolist())):
                sel=mask&(rx==r)&(day==d);mean=z[sel].mean(0)
                domain+=int(sel.sum())*float(np.square(mean-mu).sum());within+=float(np.square(z[sel]-mean).sum())
    if not np.isclose(total,between+domain+within,rtol=1e-10,atol=1e-9):raise ValueError('Scatter decomposition failed')
    return dict(total=total,TX=None if total<=1e-20 else between/total,
        RX_day_within_TX=None if total<=1e-20 else domain/total,within_cell=None if total<=1e-20 else within/total)


def analyze_arrays(q,require_complete=True):
    names=('all_on','auxiliary_off');features=('base_frequency','relation','embedding_on','embedding_off')
    required={'ids','truth','receiver','day','classifier_weight','classifier_scale',*names,*features}
    if set(q)!=required:raise ValueError('Saved field set differs')
    n=len(q['ids']);truth=q['truth'];rx=q['receiver'];day=q['day']
    if n==0 or len(set(q['ids'].tolist()))!=n:raise ValueError('Empty or duplicate IDs')
    if any(q[k].shape!=(n,) for k in ('truth','receiver','day')):raise ValueError('Metadata shape mismatch')
    if any(not np.issubdtype(q[k].dtype,np.integer) for k in ('truth','receiver','day')) or np.any((truth<0)|(truth>=6)):raise ValueError('Invalid coordinates')
    if any(q[k].shape!=(n,160) for k in features) or any(q[k].shape!=(n,6) for k in names) or q['classifier_weight'].shape!=(6,160):raise ValueError('Feature/logit shape mismatch')
    if any(not np.isfinite(q[k]).all() for k in required-{'ids'}):raise ValueError('Nonfinite evidence')
    if float(q['classifier_scale'])!=30.:raise ValueError('Frozen classifier scale differs')
    if require_complete:
        if n!=27000 or set(rx)!={1,3,4,6,8} or set(day)!={1,2,3}:raise ValueError('Incomplete source V')
        if any(np.count_nonzero((truth==t)&(rx==r)&(day==d))!=300 for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)):raise ValueError('Incomplete TX/RX/day cells')
    weight=unit(q['classifier_weight'].astype(float));errors={}
    for name,key in zip(names,('embedding_on','embedding_off')):
        errors[name]=float(np.max(np.abs(30*unit(q[key].astype(float))@weight.T-q[name])))
    if max(errors.values())>5e-5:raise ValueError('Logits fail independent feature/head reconstruction')
    values={name:classification(q[name],truth) for name in names}
    on=values['all_on'][0]==truth;off=values['auxiliary_off'][0]==truth
    partitions=dict(both_correct=on&off,both_wrong=~on&~off,helped=on&~off,hurt=~on&off)
    groups=[('ALL',np.ones(n,dtype=bool))]+[('RX'+str(r),rx==r) for r in sorted(set(rx.tolist()))]
    groups += [('TX'+str(t)+'_RX'+str(r)+'_D'+str(d),(truth==t)&(rx==r)&(day==d)) for t in sorted(set(truth.tolist())) for r in sorted(set(rx.tolist())) for d in sorted(set(day.tolist()))]
    rows=[]
    for group,mask in groups:
        if not mask.any():continue
        row=dict(group=group,count=int(mask.sum()),helped=int((partitions['helped']&mask).sum()),hurt=int((partitions['hurt']&mask).sum()))
        for name,(pred,ce,margin) in values.items():
            row[name]=dict(scores(pred[mask],truth[mask]),CE=float(ce[mask].mean()),margin=float(margin[mask].mean()))
        rows.append(row)
    return dict(count=n,groups=rows,recomputed_logit_max_errors=errors,
        partitions={p:dict(count=int(mask.sum()),**{name:dict(CE=summarize(v[1][mask]),margin=summarize(v[2][mask])) for name,v in values.items()}) for p,mask in partitions.items()},
        prediction_changes=int(np.count_nonzero(values['all_on'][0]!=values['auxiliary_off'][0])),
        geometry={key:scatter(q[key],truth,rx,day) for key in features},
        source_only=True,model_fitting=False,claim='Within frozen checkpoint intervention; normalized label scatter is descriptive, not causal TX/RX separation or retraining effect')
