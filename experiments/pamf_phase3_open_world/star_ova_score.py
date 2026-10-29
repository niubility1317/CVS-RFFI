"""Independent source-only scores for two fixed OVA/membership readouts.

Four frozen scalar rules are diagnostics of two trained models. Identity
always comes from the original P1 six-class head. No target, fitting or
threshold selection occurs here; all class evidence and margins are checked
before opening independent source-heldout truth.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import star_consensus_score as common
from .metrics import open_set_metrics

RUN_ID='20261029-phase3-star-ova-evidence-s2026092701-r01'
PREDICTION_SCHEMA='pamf_phase3_star_ova_prediction_v1'
PREDICTION_STATUS='STAR_OVA_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_OVA_THRESHOLDS_FROZEN'
SCHEMA='pamf_phase3_star_ova_scores_v1'
STATUS='STAR_OVA_SCORES_COMPLETE'
MODEL_NAMES=('membership','ova_hard')
METHODS=('membership_routed','membership_max','ova_routed','ova_max')
PRIMARY='ova_max'
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','class_scores','closed','baseline_closed'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','class_scores','closed'}
COMPARISONS=(('ova_minus_membership_routed',METHODS[2],METHODS[0]),('ova_minus_membership_max',METHODS[3],METHODS[1]),
    ('membership_max_minus_routed',METHODS[1],METHODS[0]),('ova_max_minus_routed',METHODS[3],METHODS[2]))
_read,_write,_ids,_truth=common._read,common._write,common._ids,common._truth
_masks,_goals,_delta=common._masks,common._goals,common._delta


def _readout_gate(root):
    from .star_ova_run import load_readouts
    return load_readouts(Path(root)/'readouts.npz',Path(root)/'readouts.json')


def _trained(state):
    if (state.get('schema')!='pamf_phase3_star_ova_readouts_v1'
            or state.get('descriptor_schema')!='class_conditional_paired_branch_evidence_v1'
            or state.get('descriptor_dim')!=106 or state.get('class_count')!=6 or state.get('model_names')!=list(MODEL_NAMES)
            or set(state.get('models',{}))!=set(MODEL_NAMES)):
        raise ValueError('Exactly two frozen shared 106D/32/1 source readouts required')
    for key in ('mean','scale'):
        a=state[key]
        if a.shape!=(106,) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Finite source-only 106D normalizer required')
    if np.any(state['scale']<=0):raise ValueError('Positive source-only normalizer scale required')
    for model in state['models'].values():
        shapes=dict(weight1=(32,106),bias1=(32,),weight2=(1,32),bias2=(1,))
        if set(model)!=set(shapes):raise ValueError('Complete shared MLP parameter keys required')
        for key,shape in shapes.items():
            a=model[key]
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Finite FP32 shared MLP state required')
    training=state.get('training',{})
    if (training.get('steps_per_arm')!=1000 or training.get('total_optimizer_updates')!=2000
            or training.get('model_seed')!=2026092701 or training.get('initial_parameters_equal') is not True
            or training.get('parameter_counts')!={name:3457 for name in MODEL_NAMES}
            or set(training.get('losses',{}))!=set(MODEL_NAMES)
            or any(len(values)!=1000 or not np.isfinite(values).all() for values in training['losses'].values())):
        raise ValueError('Two complete matched-initialization fixed1000-step source traces required')


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('schema')!=PREDICTION_SCHEMA or marker.get('status')!=PREDICTION_STATUS or marker.get('run_id')!=RUN_ID
            or not isinstance(marker.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',marker['code_commit']) is None
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_method')!=PRIMARY
            or marker.get('model_names')!=list(MODEL_NAMES) or marker.get('readout_model_count')!=2
            or marker.get('readout_steps_per_arm')!=1000 or marker.get('readout_optimizer_updates_total')!=2000
            or marker.get('readout_parameter_counts')!={name:3457 for name in MODEL_NAMES}
            or marker.get('classification_path')!='P1' or marker.get('score_threshold')!=0.
            or marker.get('source_only') is not True
            or marker.get('known_predictions_ref')!='predict/known_predictions.npz'
            or marker.get('novel_predictions_ref')!='predict/novel_predictions.npz'
            or any(marker.get(key)!=0 for key in ('target_reads','source_novel_truth_reads','new_encoder_updates',
                'new_identity_head_updates','source_V_readout_fit','source_novel_fit_reads'))
            or any(marker.get(key) is not True for key in ('model_parameters_buffers_unchanged','L_aggregates_unchanged_during_V','all_thresholds_frozen_before_novel'))):
        raise ValueError('All four source-only scalar predictions and unchanged P1 head must complete before truth')
    if not (root/'readouts.npz').is_file() or not (root/'readouts.json').is_file():
        raise ValueError('Both frozen source readout aggregates and full traces must be present')
    state=_readout_gate(root);_trained(state)
    calibration=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'global','conditional'}:raise ValueError('Four global controls and four by-six source thresholds required')
        thresholds={k:saved[k].copy() for k in saved.files}
    if (thresholds['global'].dtype!=np.float64 or thresholds['global'].shape!=(4,)
            or thresholds['conditional'].dtype!=np.float64 or thresholds['conditional'].shape!=(4,6)
            or any(not np.isfinite(a).all() for a in thresholds.values())
            or calibration.get('schema')!=PREDICTION_SCHEMA or calibration.get('status')!=CALIBRATION_STATUS
            or any(calibration.get(k)!=marker[k] for k in ('run_id','code_commit'))
            or calibration.get('method_names')!=list(METHODS) or calibration.get('group_count')!=42
            or calibration.get('known_only') is not True or calibration.get('unknown_inputs_used') is not False
            or set(calibration.get('methods',{}))!=set(METHODS)
            or any(set(calibration['methods'][name].get('groups',{}))!=set(GROUP_NAMES) for name in METHODS)
            or not np.array_equal(calibration.get('conditional_thresholds'),thresholds['conditional'])
            or not np.array_equal(calibration.get('global_thresholds'),thresholds['global'])
            or any(not np.array_equal(calibration['methods'][name].get('conditional_thresholds'),thresholds['conditional'][j]) for j,name in enumerate(METHODS))):
        raise ValueError('Complete matching known-only four-scalar class-risk calibration required')
    rows=[]
    for role,fields,count_key in (('known',KNOWN_FIELDS,'known_count'),('novel',NOVEL_FIELDS,'novel_count')):
        n=marker.get(count_key)
        if type(n) is not int or n<=0:raise ValueError('Positive completed prediction counts required')
        with np.load(root/marker[role+'_predictions_ref'],allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Exact completed source fields required; novel TX/scene truth forbidden')
            data={k:saved[k].copy() for k in saved.files}
        if (not _ids(data['ids'],n) or data['scores'].shape!=(n,4) or data['scores'].dtype!=np.float64
                or not np.isfinite(data['scores']).all() or data['class_scores'].shape!=(n,2,6) or data['class_scores'].dtype!=np.float64
                or not np.isfinite(data['class_scores']).all() or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64
                or np.any((data['closed']<0)|(data['closed']>=6))):
            raise ValueError('Unique IDs, finite four margins/two-by-six evidence and frozen full-six P1 identities required')
        for key,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
            if data[key].shape!=(n,) or data[key].dtype!=np.int64 or not np.isin(data[key],roster).all():
                raise ValueError('Aligned original source RX/day required')
        c=data['closed'];evidence=data['class_scores'];ix=np.arange(n)
        raw=np.column_stack((evidence[ix,0,c],evidence[:,0].max(1),evidence[ix,1,c],evidence[:,1].max(1)))
        expected=raw-thresholds['conditional'][:,c].T
        if not np.allclose(data['scores'],expected,rtol=1e-12,atol=1e-12):
            raise ValueError('Four margins must match their routed/max operator and original-P1-predicted-class threshold')
        if role=='known':
            if (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                    or data['baseline_closed'].shape!=(n,) or data['baseline_closed'].dtype!=np.int64
                    or not np.array_equal(data['closed'],data['baseline_closed'])):
                raise ValueError('Known V labels and original F0 same-input P1 baseline must remain aligned')
        rows.append(data)
    if set(rows[0]['ids'].tolist())&set(rows[1]['ids'].tolist()):raise ValueError('Known V and source novel IDs must be disjoint')
    return marker,calibration,thresholds,rows[0],rows[1]


def _counts(la,ra,lr,rr,correct,km,um):
    a,b=la[km],ra[km];c=correct[km];x,z=lr[um],rr[um]
    return dict(known_accept_only_left=int(np.sum(a&~b)),known_accept_only_right=int(np.sum(b&~a)),known_accept_net=int(a.sum()-b.sum()),
        known_correct_accept_only_left=int(np.sum(a&~b&c)),known_correct_accept_only_right=int(np.sum(b&~a&c)),
        known_correct_accept_net=int(np.sum(a&c)-np.sum(b&c)),unknown_rejected_only_left=int(np.sum(x&~z)),
        unknown_rejected_only_right=int(np.sum(z&~x)),unknown_rejected_net=int(x.sum()-z.sum()))


def _ci(rows,key):
    result=[];z=1.959963984540054
    for r in rows:
        n=r['count'];k=r[key]
        if n:
            p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
            lo,hi=float(max(0.,c-h)),float(min(1.,c+h))
        else:lo=hi=None
        result.append(dict(label=r['label'],count=n,successes=k,lower=lo,upper=hi))
    return result


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent OVA-source score root is required')
    marker,calibration,thresholds,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Sole truth opening after all four predictions/readouts/thresholds.
    km=dict(_masks(known['rx_i'],known['scenario']));um=dict(_masks(novel['rx_i'],truth['scenario']));methods={}
    for j,method in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            return open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],
                truth['tx'][umask],novel['scores'][umask,j],0.,known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
        groups={key:calculate(mask,um[key]) for key,mask in km.items()};overall=groups['overall']
        methods[method]=dict(overall=overall,groups=groups,threshold=0.,
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='per true-class Wilson; descriptive one source-RX pool, not independent target or macro CI',
                A_K_per_class=_ci(overall['known_per_class'],'correct_accepted'),R_U_per_class=_ci(overall['unknown_per_class'],'rejected')),
            goals=_goals(groups,calibration['methods'][method].get('constraints_passed') is True),diagnostic_only=method!=PRIMARY)
    correct=known['closed']==known['y'];paired={}
    for key,left,right in COMPARISONS:
        li,ri=METHODS.index(left),METHODS.index(right);la=known['scores'][:,li]>=0.;ra=known['scores'][:,ri]>=0.
        lr=novel['scores'][:,li]<0.;rr=novel['scores'][:,ri]<0.
        def counts(kmask,umask):return _counts(la,ra,lr,rr,correct,kmask,umask)
        paired[key]=dict(left_method=left,right_method=right,direction='left minus right',
            factor='training objective, same scalar operator' if key.startswith('ova_minus') else 'max versus routed, same trained model',
            overall=_delta(methods[left]['overall'],methods[right]['overall']),
            groups={group:_delta(methods[left]['groups'][group],methods[right]['groups'][group]) for group in GROUP_NAMES},
            counts=counts(km['overall'],um['overall']),group_counts={group:counts(km[group],um[group]) for group in GROUP_NAMES},
            by_known_TX={str(c):counts(known['y']==c,np.zeros(len(novel['ids']),bool)) for c in range(6)},
            by_unknown_TX={tx:counts(np.zeros(len(known['ids']),bool),truth['tx']==tx) for tx in UNKNOWN_TX},
            same_physical_IDs_and_original_P1_head=True,threshold_scope='each scalar own pre-frozen source-known predicted-class thresholds')
    primary=methods[PRIMARY];passed=primary['goals']['passed']
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],method_names=list(METHODS),
        primary_method=PRIMARY,classification_path='P1',score_threshold=0.,methods=methods,primary_summary=primary['overall'],
        primary_goals=primary['goals'],paired_comparisons=paired,source_gate_passed=passed,
        next_step='SOURCE_OVA_READY_FOR_TARGET_FALSIFICATION' if passed else 'NO_PROMOTION',
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,
        known_count=len(known['ids']),novel_count=len(novel['ids']),known_protection_groups=42,
        readout_model_count=2,readout_parameters_per_model=3457,readout_parameters_total=6914,readout_optimizer_updates_total=2000,
        scalar_methods_are_not_additional_trained_models=True,identity_from_original_P1_head=True,OVA_max_is_not_identity_classification=True,
        required_unknown_groups='overall + 5 source RX + 6 scenes; RX-scene unknown diagnostic only',
        limits=dict(schema='pamf_phase3_scientific_targets_v3',known_macro_acceptance_min=.90,known_macro_frr_max=.10,
            source_unknown_macro_rejection_min=.90,original_F0_open_loss_max=.02),
        averaging='equal per required true TX; H-score from macro A_K and R_U, never acceptance',
        curves='actual four margins and original frozen P1 identity; class-specific shifts change ranking; no oracle threshold selection',
        method_scope='project source-conditioned shared evidence MLP; not native OVANet, target OEM, six linear heads or replacement identity head',
        source_RX_is_seen=True,source_novel_historically_exposed=True,target_generalization_confirmed=False,
        formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,
        primary_rule='only ova_max may become ready for a separate target falsification run; controls cannot replace it')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_comparisons')})
    print(STATUS,result['next_step'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_OVA_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);score(a.prediction_root,a.novel_truth,a.output);return 0


if __name__=='__main__':raise SystemExit(main())
