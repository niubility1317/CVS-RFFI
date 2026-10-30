"""Frozen-density anonymous registration; no IQ, IO, labels in discovery or query fit."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from . import star_density_core as density
from . import star_macro12_risk as risk
from .star_consensus_readout import _masks

SOURCE_SCHEMA='pamf_phase3_signal_register_source_v1'
ANONYMOUS_SCHEMA='pamf_phase3_signal_register_anonymous_v1'
METHODS=('pooled_REG0','predicted_REG0','pooled_REG1','predicted_REG1')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
RECEIVERS,SCENES=risk.RECEIVERS,risk.SCENES
HARD_NAMES,DIAGNOSTIC_NAMES=risk.HARD_NAMES,risk.DIAGNOSTIC_NAMES
DIMENSION=160
MIN_SAMPLES=8
MIN_CLUSTER=16


def _state(state):
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=density.SCHEMA or state['known_means'].shape!=(90,160):
        raise ValueError('Only the original frozen old-six density160 is allowed')
    return state


def _white(state,x):
    _state(state)
    with np.errstate(over='ignore',invalid='ignore'):
        z=(density.unit(x)-state['origin'])@state['whitener']
    if not np.isfinite(z).all():raise ValueError('Finite original whitened coordinates required')
    return z


def _ids(ids,n):
    ids=np.asarray(ids)
    if ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)==''):
        raise ValueError('Unique aligned nonempty opaque Unicode IDs required')
    return ids


def _float(a,shape,name):
    a=np.asarray(a)
    if a.dtype!=np.float64 or a.shape!=shape or not np.isfinite(a).all():
        raise ValueError('Finite FP64 '+name+' with aligned shape required')
    return a


def _closed(a,n):
    a=np.asarray(a)
    if a.shape!=(n,) or a.dtype.kind not in 'iu' or np.any((a<0)|(a>=6)):
        raise ValueError('Aligned original old-six predictions required')
    return a


def validate_source(source):
    if (not isinstance(source,dict) or set(source)!={'schema','eps'}
            or not isinstance(source['schema'],np.ndarray) or source['schema'].shape!=()
            or source['schema'].dtype.kind!='U' or source['schema'].item()!=SOURCE_SCHEMA):
        raise ValueError('Exact source-radius aggregate schema required')
    eps=_float(source['eps'],(),'source radius')
    if eps<1e-6:raise ValueError('Positive source epsilon with the fixed floor required')
    return source


def validate_anonymous(anonymous):
    if (not isinstance(anonymous,dict) or set(anonymous)!={'schema','means','variances','anonymous_ids'}
            or not isinstance(anonymous['schema'],np.ndarray) or anonymous['schema'].shape!=()
            or anonymous['schema'].dtype.kind!='U' or anonymous['schema'].item()!=ANONYMOUS_SCHEMA):
        raise ValueError('Exact anonymous Gaussian aggregate schema required')
    means=np.asarray(anonymous['means'])
    if means.ndim!=2:raise ValueError('Anonymous means must be [K,160]')
    k=len(means);_float(means,(k,160),'anonymous means');v=_float(anonymous['variances'],(k,),'anonymous variances')
    names=np.asarray(anonymous['anonymous_ids'])
    expected=['anon-'+str(j+1).zfill(4) for j in range(k)]
    if np.any(v<density.VARIANCE_FLOOR) or names.shape!=(k,) or names.dtype.kind!='U' or names.tolist()!=expected:
        raise ValueError('Positive anonymous variances and canonical ordered anonymous IDs required')
    return anonymous


def source_radius(state,x,y,rx,day):
    """95th percentile of each source point's seventh other same-mode neighbor."""
    z=_white(state,x);n=len(z);y=_closed(y,n)
    rx,day,_,_=density._domains(rx,day,n)
    distances=[]
    for c in range(6):
        for r in RECEIVERS:
            for d in (1,2,3):
                rows=np.flatnonzero((y==c)&(rx==r)&(day==d))
                if len(rows)!=70:raise ValueError('Every source class/RX/day must have exactly 70 physical rows')
                points=z[rows];sq=density._distances(points,points);np.fill_diagonal(sq,np.inf)
                distances.append(np.sqrt(np.partition(sq,6,axis=1)[:,6]))
    eps=max(float(np.quantile(np.concatenate(distances),.95)),1e-6)
    source=dict(schema=np.asarray(SOURCE_SCHEMA),eps=np.asarray(eps,np.float64));validate_source(source)
    return source,dict(source_count=n,source_groups=90,rows_per_group=70,other_neighbor=7,quantile=.95,
        min_samples=MIN_SAMPLES,source_only=True,self_excluded=True,source_density_refitted=False)


def discover(state,source,U_features,base_scores,ids):
    """All-U DBSCAN; qualification uses a strict majority of original predicted rejects."""
    from sklearn.cluster import DBSCAN
    validate_source(source);white=_white(state,U_features);n=len(white);ids=_ids(ids,n)
    base=_float(base_scores,(n,2),'old pooled/predicted margins');order=np.argsort(ids,kind='stable')
    labels=DBSCAN(eps=float(source['eps']),min_samples=MIN_SAMPLES,metric='euclidean',n_jobs=1).fit_predict(white[order])
    members=[np.flatnonzero(labels==label) for label in set(labels.tolist())-{-1}]
    members.sort(key=lambda rows:ids[order[rows[0]]])
    canonical=np.full(n,-1,np.int64);assigned=np.full(n,-1,np.int64);rejected=base[:,1]<0
    means=[];variances=[];clusters=[]
    for label,sorted_rows in enumerate(members):
        rows=order[sorted_rows];canonical[rows]=label;count=len(rows);rejects=int(rejected[rows].sum())
        qualified=count>=MIN_CLUSTER and rejects*2>count;slot=None
        if qualified:
            slot=len(means);assigned[rows]=slot;mean=white[rows].mean(axis=0)
            radial=float(np.square(white[rows]-mean).sum()/(count*DIMENSION));alpha=count/(count+DIMENSION)
            means.append(mean);variances.append(max(alpha*radial+(1-alpha)*float(state['pooled_L_variance']),density.VARIANCE_FLOOR))
        clusters.append(dict(dbscan_label=label,size=count,rejected_count=rejects,rejected_fraction=rejects/count,
            qualified=qualified,anonymous_index=slot,minimum_opaque_id=str(ids[rows[0]])))
    anonymous=dict(schema=np.asarray(ANONYMOUS_SCHEMA),means=np.asarray(means,np.float64).reshape(-1,160),
        variances=np.asarray(variances,np.float64),anonymous_ids=np.asarray(['anon-'+str(j+1).zfill(4) for j in range(len(means))],dtype='U'))
    validate_anonymous(anonymous)
    assignments=dict(ids=ids.copy(),assignments=assigned,dbscan_labels=canonical,rejected=rejected.copy())
    diag=dict(U_count=n,all_U_rows_used=True,registered_count=len(means),clusters=clusters,noise_count=int((canonical<0).sum()),
        eps=float(source['eps']),min_samples=MIN_SAMPLES,minimum_registered_size=MIN_CLUSTER,strict_rejected_majority=True,
        true_unknown_count_used=False,true_U_labels_used=False,source_density_refitted=False)
    return anonymous,assignments,diag


def anonymous_ratios(state,anonymous,x):
    validate_anonymous(anonymous);z=_white(state,x);k=len(anonymous['means'])
    if not k:return np.empty((len(z),0),np.float64)
    variance=anonymous['variances'];dist=density._distances(z,anonymous['means'])
    # Each column is one class-conditional density: there is no 1/K prior.
    lognew=-DIMENSION/2*np.log(2*np.pi*variance)[None]-dist/(2*variance[None])
    background=density._log_density(state,z,'background')-float(state['log_abs_det_whitener'])
    result=lognew-background[:,None]
    if not np.isfinite(result).all():raise ValueError('Finite anonymous likelihood ratios required')
    return result


def _decision(closed,old,newmax,newindex,tau):
    with np.errstate(over='ignore',invalid='ignore'):new=newmax-tau
    if not np.isfinite(new).all():raise ValueError('Finite registration margins required')
    winner=new>old;decision=np.maximum(old,new)
    prediction=np.where(winner,6+newindex,closed);prediction=np.where(decision<0,-1,prediction)
    return prediction.astype(np.int64),decision


def predict(closed,base_scores,new_ratios,registration_tau):
    base=np.asarray(base_scores);n=len(base) if base.ndim==2 else 0
    if not n:raise ValueError('Nonempty aligned prediction rows required')
    base=_float(base,(n,2),'base margins');closed=_closed(closed,n);new=np.asarray(new_ratios)
    if new.ndim!=2:raise ValueError('Anonymous ratios must be [N,K]')
    new=_float(new,(n,new.shape[1]),'anonymous ratios');tau=_float(registration_tau,(2,),'registration thresholds')
    p=np.empty((n,4),np.int64);scores=np.empty((n,4),np.float64)
    for b in range(2):p[:,b]=np.where(base[:,b]>=0,closed,-1)
    scores[:,:2]=base
    if not new.shape[1]:p[:,2:]=p[:,:2];scores[:,2:]=base
    else:
        index=new.argmax(axis=1);maximum=new[np.arange(n),index]
        for b in range(2):p[:,b+2],scores[:,b+2]=_decision(closed,base[:,b],maximum,index,tau[b])
    return p,scores


def _group(y,baseline,prediction):
    counts=np.bincount(y,minlength=6);coverage=bool(np.all(counts>0))
    accepted=np.bincount(y[prediction>=0],minlength=6);correct=np.bincount(y[prediction==y],minlength=6)
    original=np.bincount(y[baseline==y],minlength=6)
    result=dict(count=len(y),class_counts=counts.tolist(),accepted_by_class=accepted.tolist(),correct_old_by_class=correct.tolist(),
        baseline_correct_by_class=original.tolist(),coverage=coverage,constraints_passed=False,
        known_macro_acceptance=None,known_open_macro_accuracy=None,original_closed_macro_accuracy=None,original_closed_minus_new_open=None)
    if coverage:
        def macro(values):return sum((Fraction(int(a),int(b)) for a,b in zip(values,counts)),Fraction())/6
        ck,ak,f0=macro(accepted),macro(correct),macro(original);loss=f0-ak
        result.update(known_macro_acceptance=float(ck),known_open_macro_accuracy=float(ak),original_closed_macro_accuracy=float(f0),
            original_closed_minus_new_open=float(loss),constraints_passed=ck>=Fraction(9,10) and loss<=Fraction(1,50))
    return result


def calibrate(ids,y,closed,baseline,base_scores,new_ratios,rx,scenario):
    """Smallest finite whole-tie new-class threshold under known-only hard12 risk."""
    ids=np.asarray(ids);n=len(ids) if ids.ndim==1 else 0
    if not n:raise ValueError('Nonempty source-known calibration rows required')
    _ids(ids,n);y=_closed(y,n);closed=_closed(closed,n);baseline=_closed(baseline,n)
    base=_float(base_scores,(n,2),'base margins');new=np.asarray(new_ratios)
    if new.ndim!=2:raise ValueError('Aligned anonymous ratios [N,K] required')
    new=_float(new,(n,new.shape[1]),'anonymous ratios');rx=np.asarray(rx);scenario=np.asarray(scenario)
    if (rx.shape!=(n,) or rx.dtype.kind not in 'iu' or not np.isin(rx,RECEIVERS).all()
            or scenario.shape!=(n,) or scenario.dtype.kind!='U' or not np.isin(scenario,SCENES).all()):
        raise ValueError('Only original source RX and six preregistered star scenes may calibrate')
    masks=dict(_masks(rx,scenario));coverage={name:set(y[mask].tolist())==set(range(6)) for name,mask in masks.items()}
    missing={name:sorted(set(range(6))-set(y[mask].tolist())) for name,mask in masks.items() if not coverage[name]}
    coverage_report=dict(hard_groups_complete=all(coverage[name] for name in HARD_NAMES),
        all_groups_complete=all(coverage.values()),missing_known_classes_by_group=missing)
    k=new.shape[1];maximum=new.max(axis=1) if k else None;index=new.argmax(axis=1) if k else None
    thresholds=[];methods={}
    for b,name in enumerate(METHODS[2:]):
        oldpred=np.where(base[:,b]>=0,closed,-1)
        def assess(prediction,names):return {key:_group(y[masks[key]],baseline[masks[key]],prediction[masks[key]]) for key in names}
        control=assess(oldpred,HARD_NAMES);control_pass=all(row['constraints_passed'] for row in control.values())
        def trial(threshold,all_groups=False):
            pred=_decision(closed,base[:,b],maximum,index,threshold)[0] if k else oldpred
            groups=assess(pred,masks if all_groups else HARD_NAMES)
            return groups,bool(control_pass and all(groups[key]['constraints_passed'] for key in HARD_NAMES))
        if k:
            with np.errstate(over='ignore',invalid='ignore'):candidates=np.unique(maximum-np.maximum(base[:,b],0))
            if not np.isfinite(candidates).all():raise ValueError('Finite whole-tie registration candidates required')
        else:candidates=np.asarray([0.],np.float64)
        sentinel=False;last=len(candidates)-1
        if k and control_pass and not trial(candidates[-1])[1]:
            following=np.nextafter(candidates[-1],np.inf)
            if not np.isfinite(following):raise ValueError('Cannot express finite no-invasion registration threshold')
            candidates=np.append(candidates,following);last+=1;sentinel=True
            if not trial(following)[1]:raise ValueError('Finite nextafter threshold failed to restore the original valid control')
        selected=last
        if control_pass:
            left,right=0,last
            while left<right:
                middle=(left+right)//2
                if trial(candidates[middle])[1]:right=middle
                else:left=middle+1
            selected=left
        threshold=float(candidates[selected]);groups,passed=trial(threshold,True);previous=None;previous_pass=None;failures={}
        if selected:
            previous=float(candidates[selected-1]);before,previous_pass=trial(previous)
            failures={key:row for key,row in before.items() if not row['constraints_passed']}
        thresholds.append(threshold)
        methods[name]=dict(registration_threshold=threshold,selected_threshold=threshold,selected_candidate_index=selected,
            candidate_count=len(candidates),previous_threshold=previous,previous_constraints_passed=previous_pass,
            previous_failure_groups=failures,groups=groups,coverage=coverage_report,constraints_passed=passed,
            base_control_constraints_passed=bool(control_pass),no_anonymous_classes=k==0,
            finite_sentinel_added=sentinel,sentinel_reason='FP64 subtraction at the largest algebraic tie did not restore the control' if sentinel else None)
    tau=np.asarray(thresholds,np.float64)
    report=dict(method_names=list(METHODS),registration_thresholds=tau.tolist(),methods=methods,known_count=n,
        group_count=42,hard_group_count=12,diagnostic_group_count=30,hard_names=list(HARD_NAMES),diagnostic_names=list(DIAGNOSTIC_NAMES),
        coverage=coverage_report,constraints_passed=all(row['constraints_passed'] for row in methods.values()),known_only=True,
        source_V_labels_used=True,source_V_threshold_fit=True,source_V_neural_fit=False,unknown_inputs_used=False,
        novel_inputs_used=False,target_reads=0,limits=dict(known_macro_acceptance_min=.90,original_closed_minus_new_open=.02),
        exact_limits=dict(known_macro_frr='1/10',original_closed_minus_new_open='1/50'),
        acceptance='joint old/anonymous competition; zero accepted; old wins ties',
        risk_scope='Empirical known-V hard12 protection; all30 RX-by-scene groups remain diagnostic')
    return tau,report
