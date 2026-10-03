"""Descriptive geometry on frozen source artifacts; no fitting or model choice."""
import numpy as np


def summarize(values):
    a=np.asarray(values,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite descriptive statistic')
    if not a.size:return dict(count=0,mean=None,p10=None,median=None,p90=None,maximum=None)
    return dict(count=len(a),mean=float(a.mean()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))


def unit(z,epsilon=1e-4):
    return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),epsilon)


def factor_scatter(z,truth,receiver,day):
    """Nested TX -> TX/RX/day mean scatter, descriptive and not causal."""
    z=np.asarray(z,dtype=np.float64);truth=np.asarray(truth);receiver=np.asarray(receiver);day=np.asarray(day)
    grand=z.mean(0);total=float(np.square(z-grand).sum());between_tx=0.;within_tx_domain=0.;within_cell=0.;cells=[]
    for tx in sorted(set(truth.tolist())):
        mask=truth==tx;mu=z[mask].mean(0);between_tx+=int(mask.sum())*float(np.square(mu-grand).sum())
        for rx in sorted(set(receiver[mask].tolist())):
            for d in sorted(set(day[mask&(receiver==rx)].tolist())):
                sel=mask&(receiver==rx)&(day==d);mean=z[sel].mean(0);n=int(sel.sum())
                shift=float(np.square(mean-mu).sum());within_tx_domain+=n*shift
                within_cell+=float(np.square(z[sel]-mean).sum())
                cells.append(dict(tx=int(tx),receiver=int(rx),day=int(d),count=n,mean_shift_from_tx=float(np.sqrt(shift))))
    if not np.isclose(total,between_tx+within_tx_domain+within_cell,rtol=1e-10,atol=1e-9):raise ValueError('Nested scatter does not reconcile')
    return dict(total=total,between_tx=between_tx,within_tx_between_rx_day=within_tx_domain,within_cell=within_cell,
                fractions={name:None if total<=1e-20 else value/total for name,value in [('between_tx',between_tx),('within_tx_between_rx_day',within_tx_domain),('within_cell',within_cell)]},cells=cells,
                interpretation='Observed nested source-label mean differences, not independent/counterfactual TX or channel factors')


def logit_statistics(logits,truth):
    maximum=logits.max(1);shift=logits-maximum[:,None];logsum=np.log(np.exp(shift).sum(1))
    logp=shift-logsum[:,None];prob=np.exp(logp)
    other=logits.copy();other[np.arange(len(truth)),truth]=-np.inf
    return dict(ce=-logp[np.arange(len(truth)),truth],true_probability=prob[np.arange(len(truth)),truth],confidence=prob.max(1),margin=logits[np.arange(len(truth)),truth]-other.max(1),prediction=logits.argmax(1))


def analyze_arrays(q,require_complete=True):
    required={'ids','truth','receiver','day','classifier_weight','classifier_scale','base','response','coefficients','all_on','auxiliary_off'}
    if set(q)!=required:raise ValueError('Geometry artifact fields differ')
    truth=q['truth'];rx=q['receiver'];day=q['day'];n=len(truth)
    if len(set(q['ids'].tolist()))!=n or any(len(q[k])!=n for k in ('ids','receiver','day','base','response','coefficients','all_on','auxiliary_off')):raise ValueError('Unpaired source rows')
    if any(not np.issubdtype(q[k].dtype,np.integer) for k in ('truth','receiver','day')) or np.any((truth<0)|(truth>=6)):raise ValueError('Invalid source class')
    if q['base'].shape!=(n,160) or q['response'].shape!=(n,160) or q['coefficients'].shape!=(n,8) or any(q[k].shape!=(n,6) for k in ('all_on','auxiliary_off')) or q['classifier_weight'].shape!=(6,160):raise ValueError('Geometry shape differs')
    if any(not np.isfinite(q[k]).all() for k in required-{'ids'}):raise ValueError('Nonfinite geometry artifact')
    if float(q['classifier_scale'])!=30.:raise ValueError('Classifier scale differs')
    if require_complete:
        if n!=27000 or set(rx)!={1,3,4,6,8} or set(day)!={1,2,3}:raise ValueError('Incomplete source geometry')
        if any(np.count_nonzero((truth==t)&(rx==r)&(day==d))!=300 for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)):raise ValueError('Source TX/RX/day cell counts differ')
    base=q['base'].astype(np.float64);res=q['response'].astype(np.float64);joint=base+res
    bnorm=np.linalg.norm(base,axis=1);rnorm=np.linalg.norm(res,axis=1);dot=np.sum(base*res,axis=1)
    valid=(bnorm>1e-12)&(rnorm>1e-12);cos=dot/np.maximum(bnorm*rnorm,1e-24)
    base_unit=unit(base);joint_unit=unit(joint);res_unit=unit(res);movement=joint_unit-base_unit
    weight=unit(q['classifier_weight'].astype(np.float64));center=weight-weight.mean(0)
    _,sv,vt=np.linalg.svd(center,full_matrices=False);rank=int(np.sum(sv>max(sv[0]*1e-8,1e-10)));basis=vt[:rank]
    recomputed={'all_on':30.*(joint_unit@weight.T),'auxiliary_off':30.*(base_unit@weight.T)}
    errors={name:float(np.max(np.abs(value-q[name]))) for name,value in recomputed.items()}
    if any(value>5e-5 for value in errors.values()):raise ValueError('Saved logits do not match frozen features/head')
    direction_fraction=lambda z:np.square(z@basis.T).sum(1)/np.maximum(np.square(z).sum(1),1e-24)
    scalars=dict(base_norm=bnorm,response_norm=rnorm,response_base_cosine=cos[valid],parallel_response_energy_fraction=np.square(cos[valid]),
                 tangent_response_relative=np.sqrt(np.maximum(np.square(rnorm)-np.square(dot/np.maximum(bnorm,1e-12)),0))/np.maximum(bnorm,1e-12),
                 unit_embedding_movement=np.linalg.norm(movement,axis=1),response_head_span_energy_fraction=direction_fraction(res),movement_head_span_energy_fraction=direction_fraction(movement))
    stats={name:logit_statistics(q[name].astype(np.float64),truth) for name in ('all_on','auxiliary_off')}
    allcorrect=stats['all_on']['prediction']==truth;basecorrect=stats['auxiliary_off']['prediction']==truth
    masks=dict(both_correct=allcorrect&basecorrect,both_wrong=~allcorrect&~basecorrect,helped=allcorrect&~basecorrect,hurt=~allcorrect&basecorrect)
    classified={}
    for name,v in stats.items():
        classified[name]=dict(accuracy=float((v['prediction']==truth).mean()),ce=float(v['ce'].mean()),metrics={k:summarize(v[k]) for k in ('ce','margin','confidence','true_probability')},
            partitions={part:dict(count=int(mask.sum()),**{k:summarize(v[k][mask]) for k in ('ce','margin','confidence')}) for part,mask in masks.items()},
            receiver={str(r):dict(count=int((rx==r).sum()),accuracy=float((v['prediction'][rx==r]==truth[rx==r]).mean()),ce=float(v['ce'][rx==r].mean())) for r in sorted(set(rx))})
    return dict(count=n,valid_cosine_packets=int(valid.sum()),head_difference_rank=rank,recomputed_logit_max_errors=errors,
                geometry={name:summarize(value) for name,value in scalars.items()},classification=classified,
                prediction_changes=int(np.count_nonzero(stats['all_on']['prediction']!=stats['auxiliary_off']['prediction'])),
                nested_scatter={name:factor_scatter(value,truth,rx,day) for name,value in [('base_unit',base_unit),('response_unit',res_unit),('joint_unit',joint_unit),('compensation_coefficients',q['coefficients'])]},
                source_only=True,model_fitting=False,interpretation='Normalized feature geometry and observed source-label scatter; classifier span excludes common-logit shifts. Not causal disentanglement or independent generalization evidence.')
