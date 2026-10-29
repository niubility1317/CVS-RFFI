"""Truth-last comparison of exact parents and a common-initialized policy."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

from . import star_common_seeded_risk_run as evaluation
from . import star_joint_ova_score as parent_score
from . import star_oe_risk_score as oe_parent_score
from .metrics import open_set_metrics

RUN_ID='20261030-phase3-star-common-seeded-risk-s2026092701-r01'
PARENT_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
PARENT_CODE='5aa041122396d48c49b8cb3d5912649bdc3217e1'
OE_PARENT_RUN='20261030-phase3-star-oe-risk-budget-s2026092701-r01'
OE_PARENT_CODE='585e10b07be1f6821edea08b4853582b92512aff'
SCHEMA='pamf_phase3_star_common_seeded_risk_scores_v1'
STATUS='STAR_COMMON_SEEDED_RISK_SCORES_COMPLETE'
METHODS=('common_tail','zero_initialized_budget','common_initialized_budget')
PRIMARY='common_initialized_budget';PRIMARY_ARM='joint'
RECEIVERS,SCENES,UNKNOWN_TX=parent_score.RECEIVERS,parent_score.SCENES,parent_score.UNKNOWN_TX
GROUP_NAMES=parent_score.GROUP_NAMES
_read,_write,_truth=parent_score._read,parent_score._write,parent_score._truth


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('run_id')!=RUN_ID or marker.get('schema')!=evaluation.SCHEMA or marker.get('status')!=evaluation.STATUS
            or marker.get('arm') not in ('detach_control','joint') or marker.get('method_names')!=list(METHODS)
            or marker.get('primary_arm')!=PRIMARY_ARM or marker.get('primary_method')!=PRIMARY
            or re.fullmatch('[0-9a-f]{40}',str(marker.get('code_commit')) or '') is None
            or marker.get('parent_run_id')!=PARENT_RUN or marker.get('parent_code_commit')!=PARENT_CODE
            or marker.get('OE_parent_run_id')!=OE_PARENT_RUN or marker.get('OE_parent_code_commit')!=OE_PARENT_CODE
            or marker.get('classification_path')!='P1' or marker.get('source_only') is not True
            or any(marker.get(k)!=v for k,v in dict(target_reads=0,source_L_reads=0,source_U_reads=0,source_novel_truth_reads=0,neural_updates=0,
                G_forward_rows=0,F_forward_rows=2,real_P1_forward_rows=0,synthetic_P0_forward_rows=2).items())
            or any(marker.get(k) is not True for k in ('all_thresholds_frozen_before_novel',
                'model_parameters_buffers_unchanged','aux_parameters_buffers_unchanged','monotone_thresholds_from_common'))):
        raise ValueError('Complete frozen common-seeded source predictions required before truth')
    pm,pc,pt,pk,pu=parent_score._prediction(marker['parent_prediction_root'])
    om,oc,ot,ok,ou,oe=oe_parent_score._prediction(marker['OE_prediction_root'])
    if (pm['run_id']!=PARENT_RUN or pm['code_commit']!=PARENT_CODE or om['run_id']!=OE_PARENT_RUN
            or om['code_commit']!=OE_PARENT_CODE or pm['arm']!=marker['arm'] or om['arm']!=marker['arm']
            or marker.get('parent_model_identity')!=pm['model_identity'] or om['parent_model_identity']!=pm['model_identity']):
        raise ValueError('Both exact source parent arms and matching frozen model identity required')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Only three six-class threshold policies required')
        tau=z['conditional'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or not np.array_equal(tau[0],pt['conditional'][2]) or not np.array_equal(tau[1],ot[1])
            or np.any(tau[2]<tau[0])
            or cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit','arm'))
            or cal.get('method_names')!=list(METHODS) or not np.array_equal(cal.get('conditional_thresholds'),tau)
            or cal.get('group_count')!=42 or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or cal.get('source_V_labels_used') is not True or cal.get('source_OE_membership_used') is not True
            or cal.get('source_OE_TX_or_K_used') is not False or cal.get('novel_inputs_used') is not False
            or set(cal.get('methods',{}))!=set(METHODS)
            or any(set(cal['methods'][name].get('groups',{}))!=set(GROUP_NAMES) for name in METHODS)
            or any(not np.array_equal(cal['methods'][name].get('conditional_thresholds'),tau[j]) for j,name in enumerate(METHODS))
            or not np.array_equal(cal['methods'][PRIMARY].get('initial_thresholds'),tau[0])):
        raise ValueError('Exact original controls, seeded initial state and monotone thresholds required')
    rows=[]
    for role,parent,old,key in (('known',pk,ok,'known_count'),('novel',pu,ou,'novel_count')):
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=set(parent):raise ValueError('Only unchanged public parent prediction fields allowed')
            data={k:z[k].copy() for k in z.files}
        n=len(parent['ids'])
        if (marker.get(key)!=n or any(not np.array_equal(data[k],parent[k]) for k in parent if k!='scores')
                or any(not np.array_equal(data[k],old[k]) for k in parent if k!='scores')
                or data['scores'].shape!=(n,3) or data['scores'].dtype!=np.float64 or not np.isfinite(data['scores']).all()):
            raise ValueError('Original default identity, raw logits, IDs, baselines and features must remain unchanged')
        raw=data['aux_logits'].astype(np.float64).max(1)
        if (not np.allclose(data['scores'],raw[:,None]-tau[:,data['closed']].T,rtol=0,atol=1e-12)
                or not np.array_equal(data['scores'][:,0],parent['scores'][:,2])
                or not np.array_equal(data['scores'][:,1],old['scores'][:,1])
                or np.any((data['scores'][:,2]>=0)&(data['scores'][:,0]<0))):
            raise ValueError('Three exact raw-score policies and new acceptance subset of common required')
        rows.append(data)
    with np.load(root/'OE_scores.npz',allow_pickle=False) as z:
        if set(z.files)!=evaluation.OE_FIELDS:raise ValueError('Only public OE membership/domain evidence allowed')
        current={k:z[k].copy() for k in z.files}
    if marker.get('OE_count')!=len(oe['ids']) or any(not np.array_equal(current[k],oe[k]) for k in oe):
        raise ValueError('Exact frozen parent OE score evidence required')
    return marker,cal,tau,*rows,oe


def _paired(known,novel,methods):
    correct=known['closed']==known['y'];ka=known['scores']>=0;ur=novel['scores']<0;result={}
    for j,name in enumerate(METHODS[:-1]):
        left,right=methods[PRIMARY]['overall'],methods[name]['overall']
        delta={key:None if left[key] is None or right[key] is None else float(left[key]-right[key])
            for key in ('known_macro_open_accuracy','known_macro_acceptance','unknown_macro_rejection','H_OS')}
        result[name]=dict(known_correct_gained=int(np.sum(correct&ka[:,2]&~ka[:,j])),
            known_correct_lost=int(np.sum(correct&~ka[:,2]&ka[:,j])),
            known_acceptance_gained=int(np.sum(ka[:,2]&~ka[:,j])),known_acceptance_lost=int(np.sum(~ka[:,2]&ka[:,j])),
            unknown_rejection_gained=int(np.sum(ur[:,2]&~ur[:,j])),unknown_rejection_lost=int(np.sum(~ur[:,2]&ur[:,j])),delta=delta)
    return result


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new independent score root is required')
    marker,cal,tau,known,novel,oe=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)
    km=dict(parent_score._masks(known['rx_i'],known['scenario']));um=dict(parent_score._masks(novel['rx_i'],truth['scenario']));methods={}
    for j,name in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            row=open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],truth['tx'][umask],
                novel['scores'][umask,j],0.,known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
            closed=[float(np.mean(known['closed'][kmask&(known['y']==c)]==c)) if np.any(kmask&(known['y']==c)) else None for c in known_roster]
            candidate=None if any(v is None for v in closed) else float(np.mean(closed))
            row.update(candidate_macro_closed_accuracy=candidate,
                candidate_closed_minus_open=None if candidate is None or row['known_macro_open_accuracy'] is None else candidate-row['known_macro_open_accuracy'])
            return row
        groups={key:calculate(mask,um[key]) for key,mask in km.items()};overall=groups['overall'];rejected=oe['score']<tau[j,oe['closed']]
        methods[name]=dict(overall=overall,groups=groups,
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='descriptive per-class Wilson',
                A_K_per_class=parent_score._ci(overall['known_per_class'],'correct_accepted'),
                R_U_per_class=parent_score._ci(overall['unknown_per_class'],'rejected')),
            OE_domain_macro_rejection=float(np.mean([np.mean(rejected[oe['domain']==d]) for d in range(15)])),OE_rejected=int(rejected.sum()),
            goals=parent_score._goals(groups,cal['methods'][name].get('constraints_passed') is True),diagnostic_only=name!=PRIMARY or marker['arm']!=PRIMARY_ARM)
    retrieval={name:parent_score.cross_rx_retrieval(novel[key],novel['ids'],novel['rx_i'],truth['tx'],UNKNOWN_TX)
        for name,key in (('candidate','candidate_fused_unit'),('original_F0','baseline_fused_unit'))}
    current,base=(retrieval[key]['macro_hit_rate'] for key in ('candidate','original_F0'));drop=None if current is None or base is None else base-current
    guard=dict(candidate_macro_hit_rate=current,original_F0_macro_hit_rate=base,macro_drop_from_original=drop,
        passed=drop is not None and drop<=.02+1e-12,max_allowed_drop=.02,not_anonymous_registration_accuracy=True)
    passed=methods[PRIMARY]['goals']['passed'] and guard['passed'];eligible=passed and marker['arm']==PRIMARY_ARM
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],arm=marker['arm'],
        method_names=list(METHODS),methods=methods,primary_arm=PRIMARY_ARM,primary_method=PRIMARY,primary_summary=methods[PRIMARY]['overall'],
        novel_cross_RX_retrieval=retrieval,novel_structure_guard=guard,paired_primary_minus=_paired(known,novel,methods),monotone_policy_verified=True,
        arm_source_gate_passed=bool(passed),eligible_primary=bool(eligible),
        next_step='SOURCE_COMMON_SEEDED_RISK_READY_FOR_TARGET_FALSIFICATION' if eligible else 'NO_PROMOTION',
        source_only=True,target_reads=0,truth_last=True,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_status='ACTIVE',
        source_RX_is_seen=True,source_novel_historically_exposed=True,parent_run_id=PARENT_RUN,parent_code_commit=PARENT_CODE,
        OE_parent_run_id=OE_PARENT_RUN,OE_parent_code_commit=OE_PARENT_CODE,
        baseline='Original F0 same-input P1 closed; never candidate closed',known_protection_groups=42)
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','novel_cross_RX_retrieval')})
    print(STATUS,marker['arm'],result['next_step'],flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_COMMON_SEEDED_RISK_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
