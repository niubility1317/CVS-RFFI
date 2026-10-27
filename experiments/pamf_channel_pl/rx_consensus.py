"""L-only nonparametric cross-RX evidence; never an inference head replacement."""
import numpy as np

RULES=('A_same_top1','B_within_top2')
THRESHOLD=.95
MARGIN=.05
MIN_EXTRA=135


def unit(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite feature matrix required')
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if (norm<=1e-8).any():raise ValueError('Nonzero features required')
    return x/norm


def bank(features,labels,rx,source_rxs,role='L_s'):
    if role!='L_s':raise ValueError('Only true labeled training data may build prototypes')
    x=unit(features);y=np.asarray(labels);r=np.asarray(rx);receivers=np.asarray(source_rxs)
    if (y.shape!=(len(x),) or r.shape!=y.shape or y.dtype.kind not in 'iu' or r.dtype.kind not in 'iu'
        or len(receivers)!=5 or len(set(receivers.tolist()))!=5 or not np.isin(r,receivers).all()
        or not np.isin(y,np.arange(6)).all()):raise ValueError('Original five-RX six-class L metadata required')
    counts=np.zeros((5,6),dtype=np.int64);centers=np.empty((5,6,x.shape[1]),dtype=np.float32)
    for i,receiver in enumerate(receivers):
        for j in range(6):
            mask=(r==receiver)&(y==j);counts[i,j]=int(mask.sum())
            if not counts[i,j]:raise ValueError('Every true L RX/TX prototype needs evidence')
            centers[i,j]=x[mask].mean(axis=0,dtype=np.float64)
    centers=unit(centers.reshape(-1,x.shape[1])).reshape(5,6,-1)
    return centers,counts


def decisions(features,logits,rx,centers,source_rxs):
    x=unit(features);q=np.asarray(logits,dtype=np.float32);r=np.asarray(rx);p=np.asarray(centers,dtype=np.float32)
    receivers=np.asarray(source_rxs)
    if (q.shape!=(len(x),6) or not np.isfinite(q).all() or r.shape!=(len(x),)
        or p.shape!=(5,6,x.shape[1]) or len(receivers)!=5 or len(set(receivers.tolist()))!=5
        or not np.isin(r,receivers).all() or not np.isfinite(p).all()
        or not np.allclose(np.linalg.norm(p,axis=2),1,atol=1e-5)):
        raise ValueError('Compatible frozen raw features/logits and five L-only prototype banks required')
    prob=np.exp(q-q.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
    order=np.argsort(prob,axis=1,kind='stable');top1=order[:,-1];top2=order[:,-2]
    accepted=prob.max(axis=1)>=THRESHOLD
    similarity=np.einsum('nf,rcf->nrc',x,p);rank=np.argsort(similarity,axis=2,kind='stable')
    vote=rank[:,:,-1];runner=rank[:,:,-2]
    margin=np.take_along_axis(similarity,vote[:,:,None],axis=2)[:,:,0]-np.take_along_axis(similarity,runner[:,:,None],axis=2)[:,:,0]
    other=r[:,None]!=receivers[None,:]
    if not np.all(other.sum(axis=1)==4):raise ValueError('Exactly own RX excluded from consensus')
    # Own RX never affects agreement or minimum margin.
    low=np.where(other,vote,6).min(axis=1);high=np.where(other,vote,-1).max(axis=1)
    min_margin=np.where(other,margin,np.inf).min(axis=1)
    consensus=(low==high)&(min_margin>=MARGIN)
    mass=np.take_along_axis(prob,order[:,-2:],axis=1).sum(axis=1)
    eligible=(~accepted)&consensus&(mass>=THRESHOLD)
    extra_a=eligible&(low==top1);extra_b=eligible&((low==top1)|(low==top2))
    return dict(base_accepted=accepted,base_label=top1.astype(np.int64),consensus_label=low.astype(np.int64),
        consensus=consensus,min_other_margin=min_margin,top2_mass=mass,
        A_same_top1=extra_a,B_within_top2=extra_b)


def score(pred,truth):
    y=np.asarray(truth)
    if y.shape!=(27000,) or y.dtype.kind not in 'iu' or not np.isin(y,np.arange(6)).all():raise ValueError('Original source V only')
    base=np.asarray(pred['base_accepted']);label=np.asarray(pred['base_label']);new=np.asarray(pred['consensus_label'])
    if base.dtype!=np.bool_ or base.shape!=y.shape or label.shape!=y.shape or new.shape!=y.shape:raise ValueError('Complete frozen predictions required')
    rows=[]
    for rule in RULES:
        extra=np.asarray(pred[rule])
        if extra.dtype!=np.bool_ or extra.shape!=y.shape or (extra&base).any():raise ValueError('Original raw95 accepted set must remain unchanged')
        n=int(extra.sum());correct=int((extra&(new==y)).sum());combined=np.where(base,label,new)
        rows.append(dict(rule=rule,extra_accepted=n,extra_correct=correct,extra_precision_percent=100*correct/n if n else None,
            extra_coverage_pp=100*n/len(y),source_diagnostic_pass=n>=MIN_EXTRA and 100*correct>=99*n,
            total_accepted=int((base|extra).sum()),total_correct=int(((base|extra)&(combined==y)).sum()),
            relabeled_extra=int((extra&(new!=label)).sum())))
    passing=[r for r in rows if r['source_diagnostic_pass']]
    selected=max(passing,key=lambda r:(r['extra_accepted'],-RULES.index(r['rule'])))['rule'] if passing else None
    return dict(rules=rows,selected_for_later_source_pilot=selected,base_accepted=int(base.sum()),
        base_correct=int((base&(label==y)).sum()),source_count=len(y),target_reads=0,U_truth_reads=0,
        scope='Source diagnostic only; pass permits a separately registered source training pilot, never target or seed expansion')
