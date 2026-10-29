"""Frozen-F0 consensus readouts with source-known-only 90%/2pp calibration.

Geometry, 100D descriptor and domain-balanced 200-step CPU readout fitting are
the existing implementations, directly exported here. No IO, encoder/head
training, unknown calibration inputs, or target fitting lives in this module.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import source_open_readout as core
from . import scores as static
from .source_consensus import descriptor, DESCRIPTOR_SCHEMA
from .source_domain_oe_readout import fit_readouts
from .source_open_readout import fit_geometry, score_readouts


RUN_ID = '20261029-phase3-star-consensus-s2026092701-r01'
SCHEMA = 'pamf_phase3_star_consensus_prediction_v1'
CALIBRATION_STATUS = 'STAR_CONSENSUS_THRESHOLDS_FROZEN'
METHODS = static.METHODS + core.METHODS
PRIMARY = 'mlp16_oe'
PRIMARY_INDEX = 6
PRIMARY_PATH = 'P1'
RECEIVERS = (1,3,4,6,8)
SCENES = ('practical_high','practical_mid','practical_low_suburban',
          'practical_high_urban','practical_mid_urban','practical_low_urban')
FRR = Fraction(1,10)
TOTAL_LOSS = Fraction(1,50)


def score_compact(desc, static_scores, state):
    values = core._matrix(desc,'head-aligned consensus descriptors',100,fp32=True)
    scores = core._matrix(static_scores,'five fixed static knownness scores',5)
    if len(values) != len(scores):
        raise ValueError('Descriptor/static score row counts differ')
    return np.column_stack((scores,score_readouts(desc,state))).astype(np.float64,copy=False)


def all_scores(branches, logits, geometry, state):
    desc = descriptor(branches,logits,geometry)
    scores = static.knownness_scores(branches['fused160'],logits,geometry['branches']['fused160']['prototypes'])
    return score_compact(desc,scores,state)


def _macro(numerator, denominator):
    return sum((Fraction(int(n),int(d)) for n,d in zip(numerator,denominator)),Fraction())/6


def _masks(rx, scenario):
    yield 'overall',np.ones(len(rx),bool)
    for receiver in RECEIVERS:yield 'rx/'+str(receiver),rx==receiver
    for scene in SCENES:yield 'scene/'+scene,scenario==scene
    for receiver in RECEIVERS:
        for scene in SCENES:yield 'rxscene/'+str(receiver)+'/'+scene,(rx==receiver)&(scenario==scene)


def _local_maximum(y, closed, baseline, values):
    """Same tie-block prefix search as the existing risk code, with own limits."""
    order=np.argsort(values,kind='stable')
    levels,sizes=np.unique(values[order],return_counts=True)
    labels=y[order];blocks=np.repeat(np.arange(len(levels)),sizes)
    counts=np.zeros((len(levels),6),np.int64);correct=np.zeros_like(counts)
    np.add.at(counts,(blocks,labels),1)
    is_correct=closed[order]==labels
    np.add.at(correct,(blocks[is_correct],labels[is_correct]),1)
    prefix=np.vstack((np.zeros((1,6),np.int64),np.cumsum(counts,axis=0)))
    correct_prefix=np.vstack((np.zeros((1,6),np.int64),np.cumsum(correct,axis=0)))
    total,new_correct=prefix[-1],correct_prefix[-1]
    old_correct=np.bincount(y[baseline==y],minlength=6)
    fixed=_macro(old_correct,total)-_macro(new_correct,total)
    def feasible(i):
        return _macro(prefix[i],total)<=FRR and fixed+_macro(correct_prefix[i],total)<=TOTAL_LOSS
    left,right=0,len(levels)-1;no_rejection=feasible(0)
    if no_rejection:
        while left<right:
            middle=(left+right+1)//2
            if feasible(middle):left=middle
            else:right=middle-1
    return float(levels[left]),bool(no_rejection)


def _risk_group(y, closed, baseline, values, threshold):
    total=np.bincount(y,minlength=6)
    rejected=np.bincount(y[values<threshold],minlength=6)
    correct=np.bincount(y[closed==y],minlength=6)
    original=np.bincount(y[baseline==y],minlength=6)
    rejected_correct=np.bincount(y[(values<threshold)&(closed==y)],minlength=6)
    frr=_macro(rejected,total);old_closed=_macro(original,total);new_closed=_macro(correct,total)
    rejected_correct_risk=_macro(rejected_correct,total)
    new_open=new_closed-rejected_correct_risk;loss=old_closed-new_open
    return dict(certified=True,known_count=int(total.sum()),class_counts=total.tolist(),
        rejected_count=int(rejected.sum()),rejected_by_class=rejected.tolist(),
        rejected_correct_count=int(rejected_correct.sum()),rejected_correct_by_class=rejected_correct.tolist(),
        original_correct_by_class=original.tolist(),new_closed_correct_by_class=correct.tolist(),
        macro_frr=float(frr),known_macro_frr=float(frr),known_macro_acceptance=float(1-frr),
        original_closed_macro_accuracy=float(old_closed),new_closed_macro_accuracy=float(new_closed),
        new_open_macro_accuracy=float(new_open),known_open_macro_accuracy=float(new_open),
        encoder_closed_loss=float(old_closed-new_closed),correct_rejection_loss=float(rejected_correct_risk),
        original_closed_minus_new_open=float(loss),constraints_passed=bool(frr<=FRR and loss<=TOTAL_LOSS))


def calibrate_known(ids, y, closed, baseline, scores, rx, scenario, *, method_names=METHODS):
    """Maximal global scalar/method from source known V, accepting full ties.

    All 42 groups require six-class coverage. Missing coverage or already
    excessive original-F0 damage uses the minimum score (accept all), reports
    scientific failure, and does not borrow unknown scores or labels.
    """
    if not isinstance(method_names,(tuple,list,np.ndarray)) or np.ndim(method_names)!=1:
        raise ValueError('An explicit nonempty vector of unique method names is required')
    names=tuple(method_names)
    if not names or any(not isinstance(name,str) or not name.strip() for name in names) or len(set(names))!=len(names):
        raise ValueError('An explicit nonempty vector of unique method names is required')
    ids,y,closed,baseline,scores,rx,scenario=map(np.asarray,(ids,y,closed,baseline,scores,rx,scenario))
    n=len(ids) if ids.ndim==1 else 0
    if (not n or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or scores.shape!=(n,len(names)) or scores.dtype.kind!='f' or not np.isfinite(scores).all()
            or any(a.shape!=(n,) or a.dtype.kind not in 'iu' for a in (y,closed,baseline,rx))
            or any(np.any((a<0)|(a>=6)) for a in (y,closed,baseline)) or not np.isin(rx,RECEIVERS).all()
            or scenario.shape!=(n,) or scenario.dtype.kind!='U' or not np.isin(scenario,SCENES).all()):
        raise ValueError('Only aligned source-known V IDs/labels, complete six-class predictions and declared finite scores may calibrate')
    masks=list(_masks(rx,scenario));coverage={key:set(y[mask].tolist())==set(range(6)) for key,mask in masks}
    methods={};thresholds=[]
    for j,name in enumerate(names):
        values=scores[:,j]
        local={key:_local_maximum(y[mask],closed[mask],baseline[mask],values[mask]) for key,mask in masks if coverage[key]}
        feasible=all(coverage.values()) and all(row[1] for row in local.values())
        threshold=min(row[0] for row in local.values()) if feasible else float(values.min())
        groups={key:_risk_group(y[mask],closed[mask],baseline[mask],values[mask],threshold) if coverage[key] else
            dict(certified=False,constraints_passed=False,reason='missing_known_class',known_count=int(mask.sum()),
                 class_counts=np.bincount(y[mask],minlength=6).tolist()) for key,mask in masks}
        levels=np.unique(values);index=int(np.searchsorted(levels,threshold))
        with np.errstate(over='ignore'):
            following=float(levels[index+1]) if index+1<len(levels) else float(np.nextafter(levels[-1],np.inf))
        next_pass=feasible and np.isfinite(following) and all(_risk_group(
            y[mask],closed[mask],baseline[mask],values[mask],following)['constraints_passed'] for _,mask in masks)
        methods[name]=dict(threshold=float(threshold),no_rejection_feasible=bool(feasible),zero_rejection_fallback=not feasible,
            constraints_passed=bool(feasible and all(row['constraints_passed'] for row in groups.values())),
            all_groups_class_complete=all(coverage.values()),groups=groups,
            local_maximum_thresholds={key:row[0] for key,row in local.items()},
            selected_candidate_index=index,next_candidate_threshold=following if np.isfinite(following) else None,
            next_candidate_passed=bool(next_pass))
        thresholds.append(threshold)
    return np.asarray(thresholds,np.float64),dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,
        method_names=list(names),primary_method=PRIMARY if names==METHODS else None,primary_path=PRIMARY_PATH,known_count=n,group_count=42,
        class_count=6,receivers=list(RECEIVERS),scenario_order=list(SCENES),known_only=True,source_only=True,
        methods=methods,constraints_passed=all(row['constraints_passed'] for row in methods.values()),
        source_V_labels_used=True,source_V_metric_fit=False,source_V_readout_fit=False,source_V_score_normalization_fit=False,
        unknown_inputs_used=False,target_reads=0,baseline='original F0 same-input same-path complete head',
        limits=dict(known_macro_acceptance_min=.9,known_macro_frr=.1,original_closed_minus_new_open=.02),
        exact_limits=dict(known_macro_frr='1/10',original_closed_minus_new_open='1/50'),
        group_scope='overall + 5 RX + 6 actual scenes + 30 RX-by-scene; equal known-class macro',
        acceptance_rule='score >= maximal legal global scalar; intact ties; infeasible or missing coverage accepts all and stays false',
        risk_scope='Empirical source-V constraints; not an independent target guarantee')
