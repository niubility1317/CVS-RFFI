"""Deterministic source-OE threshold allocation under original-F0 known risks.

Temporary sorted queues allocate six independent predicted-class thresholds.
Only the six finite thresholds deploy; neither OE rows nor risk curves do.
The greedy stopping point is maximal in its fixed moves, not globally optimal.
"""
from __future__ import annotations

import numpy as np

from . import star_consensus_readout as risk

RUN_ID='20261030-phase3-star-oe-risk-budget-s2026092701-r01'
RECEIVERS,SCENES=risk.RECEIVERS,risk.SCENES
LIMITS=np.asarray([.10,.02],np.float64)
TOLERANCE=1e-12


def _ids(value,n):
    return (value.shape==(n,) and value.dtype.kind=='U' and len(set(value.tolist()))==n
        and not np.any(np.char.strip(value)==''))


def _valid(ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain):
    values=list(map(np.asarray,(ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain)))
    ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain=values
    n=len(ids) if ids.ndim==1 else 0;m=len(oe_ids) if oe_ids.ndim==1 else 0
    if (not n or not m or not _ids(ids,n) or not _ids(oe_ids,m) or set(ids.tolist())&set(oe_ids.tolist())
            or any(v.shape!=(n,) or v.dtype.kind not in 'iu' for v in (y,closed,baseline,rx))
            or any(np.any((v<0)|(v>=6)) for v in (y,closed,baseline)) or not np.isin(rx,RECEIVERS).all()
            or score.shape!=(n,) or score.dtype.kind!='f' or not np.isfinite(score).all()
            or scenario.shape!=(n,) or scenario.dtype.kind!='U' or not np.isin(scenario,SCENES).all()
            or oe_closed.shape!=(m,) or oe_closed.dtype.kind not in 'iu' or np.any((oe_closed<0)|(oe_closed>=6))
            or oe_score.shape!=(m,) or oe_score.dtype.kind!='f' or not np.isfinite(oe_score).all()
            or oe_domain.shape!=(m,) or oe_domain.dtype.kind not in 'iu' or set(oe_domain.tolist())!=set(range(15))):
        raise ValueError('Aligned finite source V and disjoint source OE membership/15-domain arrays required')
    values[4]=score.astype(np.float64,copy=False);values[9]=oe_score.astype(np.float64,copy=False)
    return values


def _groups(y,closed,baseline,accepted,masks,coverage):
    return {name:risk._risk_group(y[mask],closed[mask],baseline[mask],accepted[mask].astype(np.float64),.5)
        if coverage[name] else dict(certified=False,constraints_passed=False,reason='missing_known_class',
            known_count=int(mask.sum()),class_counts=np.bincount(y[mask],minlength=6).tolist()) for name,mask in masks}


def _queues(ids,y,closed,score,oe_closed,oe_score,oe_domain,masks,counts,domain_counts,thresholds):
    queues=[]
    for c in range(6):
        oe=np.flatnonzero(oe_closed==c);levels,inverse=np.unique(oe_score[oe],return_inverse=True)
        domain_layers=np.zeros((len(levels),15),np.int64)
        np.add.at(domain_layers,(inverse,oe_domain[oe]),1)
        gains=(domain_layers/domain_counts[None,:]).sum(axis=1)/15
        ix=np.flatnonzero(closed==c);ix=ix[np.lexsort((ids[ix],score[ix]))]
        positions=np.searchsorted(score[ix],levels,side='right')
        coefficients=np.zeros((len(ix),len(masks)),np.float64)
        for gi,(_,mask) in enumerate(masks):coefficients[:,gi]=mask[ix]/(6.*counts[gi,y[ix]])
        prefix=np.zeros((len(ix)+1,len(masks),2),np.float64)
        prefix[1:,:,0]=np.cumsum(coefficients,axis=0)
        prefix[1:,:,1]=np.cumsum(coefficients*(closed[ix]==y[ix])[:,None],axis=0)
        initial_known=int(np.searchsorted(score[ix],thresholds[c],side='left'))
        queues.append(dict(levels=levels,gains=gains,risks=prefix[positions],known_counts=positions,
            oe_counts=domain_layers.sum(axis=1),index=int(np.searchsorted(levels,thresholds[c],side='left')),
            initial_risk=prefix[initial_known],selected_known=initial_known))
    return queues


def _candidate(c,queues,selected,current,base):
    q=queues[c];i=q['index']
    if i==len(q['levels']):return dict(class_index=c,reason='OE_levels_exhausted',feasible=False,failure_groups={})
    level=float(q['levels'][i])
    with np.errstate(over='ignore'):threshold=float(np.nextafter(level,np.inf))
    if not np.isfinite(threshold):
        return dict(class_index=c,reason='no_finite_nextafter',feasible=False,OE_level=level,threshold=None,failure_groups={})
    delta=q['risks'][i]-selected[c]
    proposal=base+(selected.sum(axis=0)-selected[c])+q['risks'][i]
    remaining=np.maximum(LIMITS-current,0.)
    ratios=np.zeros_like(delta);positive=delta>0
    np.divide(delta,remaining,out=ratios,where=positive&(remaining>0))
    ratios[positive&(remaining<=0)]=np.inf
    cost=float(ratios.max());gain=float(q['gains'][i]);feasible=bool(np.all(proposal<=LIMITS+TOLERANCE))
    return dict(class_index=c,index=i,OE_level=level,threshold=threshold,gain=gain,cost=cost,
        ratio=gain/(cost+TOLERANCE),feasible=feasible,proposal=proposal,
        reason='feasible' if feasible else 'known_risk_budget',failure_groups={})


def _objective(oe_score,oe_closed,oe_domain,thresholds,domain_counts):
    rejected=oe_score<thresholds[oe_closed]
    return float(np.mean(np.bincount(oe_domain[rejected],minlength=15)/domain_counts))


def calibrate(ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain,*,initial_thresholds=None):
    """Allocate thresholds from accept-all or a supplied monotone starting point."""
    ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain=_valid(
        ids,y,closed,baseline,score,rx,scenario,oe_ids,oe_closed,oe_score,oe_domain)
    masks=list(risk._masks(rx,scenario));counts=np.asarray([np.bincount(y[mask],minlength=6) for _,mask in masks])
    coverage={name:bool(np.all(counts[i]>0)) for i,(name,_) in enumerate(masks)}
    domain_counts=np.bincount(oe_domain,minlength=15)
    provided=initial_thresholds is not None
    if provided:
        thresholds=np.asarray(initial_thresholds)
        if thresholds.shape!=(6,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all():
            raise ValueError('The optional starting state must contain six finite FP64 thresholds')
        thresholds=thresholds.copy()
    else:thresholds=np.full(6,min(float(score.min()),float(oe_score.min())),np.float64)
    start_thresholds=thresholds.copy()
    initial_levels=[int(np.searchsorted(np.unique(oe_score[oe_closed==c]),thresholds[c],side='left')) for c in range(6)]
    objective_before=_objective(oe_score,oe_closed,oe_domain,thresholds,domain_counts)
    no_rejection=_groups(y,closed,baseline,np.ones(len(y),bool),masks,coverage)
    no_rejection_feasible=all(coverage.values()) and all(row['constraints_passed'] for row in no_rejection.values())
    initial=_groups(y,closed,baseline,score>=thresholds[closed],masks,coverage)
    feasible=all(coverage.values()) and all(row['constraints_passed'] for row in initial.values())
    trace=[];queues=[];next_rows=[];selected=np.zeros((6,len(masks),2),np.float64)
    if feasible:
        base=np.asarray([[0.,no_rejection[name]['encoder_closed_loss']] for name,_ in masks],np.float64)
        queues=_queues(ids,y,closed,score,oe_closed,oe_score,oe_domain,masks,counts,domain_counts,thresholds)
        for c,q in enumerate(queues):selected[c]=q['initial_risk']
        current=base+selected.sum(axis=0)
        while True:
            best=None
            for c in range(6):
                row=_candidate(c,queues,selected,current,base)
                if row['feasible'] and row['gain']>0 and (best is None or row['ratio']>best['ratio']):best=row
            if best is None:break
            c=best['class_index'];q=queues[c];i=q['index'];old_known=q['selected_known']
            thresholds[c]=best['threshold'];selected[c]=q['risks'][i];q['index']+=1;current=base+selected.sum(axis=0)
            q['selected_known']=int(q['known_counts'][i])
            trace.append(dict(step=len(trace)+1,class_index=c,OE_level=best['OE_level'],threshold=best['threshold'],
                gain=best['gain'],cost=best['cost'] if np.isfinite(best['cost']) else None,ratio=best['ratio'],
                OE_newly_rejected=int(q['oe_counts'][i]),known_newly_rejected=int(q['known_counts'][i])-old_known))
        for c in range(6):
            row=_candidate(c,queues,selected,current,base);proposal=row.pop('proposal',None)
            if proposal is not None:
                row['failure_groups']={name:dict(macro_frr=float(proposal[g,0]),original_closed_minus_new_open=float(proposal[g,1]))
                    for g,(name,_) in enumerate(masks) if np.any(proposal[g]>LIMITS+TOLERANCE)}
                if not np.isfinite(row['cost']):row['cost']=None
            next_rows.append(row)
    else:
        next_rows=[dict(class_index=c,reason='initial_known_constraints_infeasible',feasible=False,failure_groups={}) for c in range(6)]
    # Exact Fraction accounting decides the reported result, even if a float
    # candidate was only feasible within the explicit numerical tolerance.
    groups=_groups(y,closed,baseline,score>=thresholds[closed],masks,coverage)
    passed=all(coverage.values()) and all(row['constraints_passed'] for row in groups.values())
    state={'conditional_thresholds':thresholds}
    report=dict(conditional_thresholds=thresholds.tolist(),constraints_passed=bool(passed),groups=groups,group_count=42,
        initialization='provided_thresholds' if provided else 'accept_all',initial_thresholds=start_thresholds.tolist(),
        objective_before=objective_before,objective_after=_objective(oe_score,oe_closed,oe_domain,thresholds,domain_counts),trace=trace,
        coverage=dict(known_groups_complete=all(coverage.values()),missing_known_classes_by_group={name:np.flatnonzero(counts[i]==0).tolist()
            for i,(name,_) in enumerate(masks) if not coverage[name]},known_predicted_class_counts=np.bincount(closed,minlength=6).tolist(),
            OE_domains_complete=True,OE_domain_counts=domain_counts.tolist()),
        initial_feasible=bool(feasible),no_rejection_feasible=bool(no_rejection_feasible),zero_rejection_fallback=bool(not provided and not feasible),
        initial_state_preserved_on_failure=bool(provided and not feasible),
        maximal=bool(feasible and not any(row['feasible'] for row in next_rows)),global_optimum_claimed=False,
        selected_prefix_counts=[int(np.count_nonzero((oe_closed==c)&(oe_score<thresholds[c]))) for c in range(6)],
        selected_OE_levels=[q['index'] for q in queues] if queues else initial_levels,next_candidates=next_rows,
        known_count=len(y),OE_count=len(oe_score),source_only=True,target_reads=0,novel_inputs_used=False,
        source_V_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,
        source_V_neural_fit=False,source_V_normalizer_fit=False,neural_updates=0,
        objective='source OE rejection with equal 1/15 RX/day domain mass',
        limits=dict(known_macro_acceptance_min=.9,known_macro_frr=.1,original_closed_minus_new_open=.02),
        numerical_tolerance=TOLERANCE,final_risk_accounting='exact Fraction via original _risk_group',
        routing='frozen predicted class only',acceptance_rule='score >= conditional_thresholds[closed]; intact ties',
        algorithm='deterministic greedy feasible OE-layer risk-budget allocation; class index breaks equal ratios',
        deployment_state='six finite thresholds only; no source/OE row bank or risk curve',conformal_guarantee=False)
    return state,report


def apply(score,closed,state):
    """Return row-local FP64 margins; nonnegative values are accepted."""
    score,closed=np.asarray(score),np.asarray(closed)
    if (score.ndim!=1 or not len(score) or score.dtype.kind!='f' or not np.isfinite(score).all()
            or closed.shape!=score.shape or closed.dtype.kind not in 'iu' or np.any((closed<0)|(closed>=6))):
        raise ValueError('Aligned finite raw scores and frozen six-class predictions required')
    if not isinstance(state,dict) or set(state)!={'conditional_thresholds'}:raise ValueError('Only six deployment thresholds required')
    thresholds=np.asarray(state['conditional_thresholds'])
    if thresholds.dtype!=np.float64 or thresholds.shape!=(6,) or not np.isfinite(thresholds).all():
        raise ValueError('Six finite FP64 deployment thresholds required')
    with np.errstate(over='ignore',invalid='ignore'):result=score.astype(np.float64)-thresholds[closed]
    if not np.isfinite(result).all():raise ValueError('Finite deployment margins required')
    return result
