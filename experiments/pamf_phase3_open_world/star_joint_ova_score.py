"""Truth-last source metrics and original-paired cross-RX structural guard."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import star_joint_ova_eval as evaluation
from . import star_consensus_score as common
from .metrics import open_set_metrics
from .star_ova_score import _ci
from .star_representation_score import cross_rx_retrieval  # Pure fixed-neighbor diagnostic only; no old gates/calibration.

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_joint_ova_scores_v1'
STATUS='STAR_JOINT_OVA_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY='ova_max';PRIMARY_ARM='joint'
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
_read,_write,_ids,_truth=common._read,common._write,common._ids,common._truth
_masks,_goals=common._masks,common._goals


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json');identity=marker.get('model_identity',{})
    if (marker.get('schema')!=evaluation.SCHEMA or marker.get('status')!=evaluation.STATUS or marker.get('run_id')!=RUN_ID
            or not isinstance(marker.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',marker['code_commit']) is None
            or marker.get('arm') not in evaluation.ARMS or marker.get('method_names')!=list(METHODS)
            or marker.get('primary_method')!=PRIMARY or marker.get('primary_arm')!=PRIMARY_ARM
            or marker.get('classification_path')!='P1' or marker.get('score_threshold')!=0.
            or marker.get('baseline')!='cache original F0 same-input P1 closed; never candidate closed'
            or marker.get('source_only') is not True
            or any(marker.get(k)!=0 for k in ('target_reads','source_novel_truth_reads','source_L_records_consumed','source_OE_records_consumed','eval_neural_updates'))
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged','aux_parameters_buffers_unchanged',
                'all_thresholds_frozen_before_novel','original_F0_loaded_after_all_thresholds'))
            or any(identity.get(k)!=marker[k] for k in ('run_id','code_commit','arm')) or identity.get('updates')!=2220
            or identity.get('cumulative_updates')!=13320 or any(identity.get(k)!=2220 for k in ('encoder_updates','identity_head_updates','auxiliary_updates'))
            or identity.get('model_parameters')!=232993 or identity.get('aux_parameters')!=966 or identity.get('identity_head_replaced') is not False):
        raise ValueError('All frozen per-arm native/default-head/aux source predictions required before truth')
    cal=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'global','conditional'}:raise ValueError('Three global controls and eighteen predicted-class thresholds required')
        thresholds={key:z[key].copy() for key in z.files}
    if (thresholds['global'].shape!=(3,) or thresholds['conditional'].shape!=(3,6)
            or any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in thresholds.values())
            or cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit','arm')) or cal.get('method_names')!=list(METHODS)
            or cal.get('group_count')!=42 or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal.get('source_V_neural_fit') is not False or set(cal.get('methods',{}))!=set(METHODS)
            or any(set(cal['methods'][name].get('groups',{}))!=set(GROUP_NAMES) for name in METHODS)
            or not np.array_equal(cal.get('conditional_thresholds'),thresholds['conditional']) or not np.array_equal(cal.get('global_thresholds'),thresholds['global'])
            or any(not np.array_equal(cal['methods'][name].get('conditional_thresholds'),thresholds['conditional'][j]) for j,name in enumerate(METHODS))):
        raise ValueError('Matching complete known-V-only class-risk calibration and original baselines required')
    rows=[]
    for role,fields,key in (('known',evaluation.KNOWN_FIELDS,'known_count'),('novel',evaluation.NOVEL_FIELDS,'novel_count')):
        n=marker.get(key)
        if type(n) is not int or n<=0:raise ValueError('Positive completed source counts required')
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=fields:raise ValueError('Exact public prediction fields only; novel TX/scene/private labels forbidden')
            data={k:z[k].copy() for k in z.files}
        if (not _ids(data['ids'],n) or data['scores'].shape!=(n,3) or data['scores'].dtype!=np.float64 or not np.isfinite(data['scores']).all()
                or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64 or np.any((data['closed']<0)|(data['closed']>=6))):
            raise ValueError('Three finite margins and complete candidate default-head predictions required')
        for k in ('default_logits','aux_logits'):
            if data[k].shape!=(n,6) or data[k].dtype!=np.float32 or not np.isfinite(data[k]).all():
                raise ValueError('Finite FP32 full-six default and auxiliary logits required')
        for k,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
            if data[k].shape!=(n,) or data[k].dtype!=np.int64 or not np.isin(data[k],roster).all():
                raise ValueError('Aligned original source RX/day required')
        raw,closed=evaluation.scalars(data['default_logits'],data['aux_logits'])
        if not np.array_equal(closed,data['closed']) or not np.allclose(data['scores'],raw-thresholds['conditional'][:,closed].T,rtol=1e-12,atol=1e-12):
            raise ValueError('Energy/routed/max margins must use actual logits and candidate default-head class, never aux argmax identity')
        if role=='known':
            if (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                    or data['baseline_closed'].shape!=(n,) or data['baseline_closed'].dtype!=np.int64 or np.any((data['baseline_closed']<0)|(data['baseline_closed']>=6))):
                raise ValueError('Original-F0 P1 baselines and legal known labels required; candidate baseline substitution forbidden')
        else:
            for k in ('candidate_fused_unit','baseline_fused_unit'):
                a=data[k]
                if a.shape!=(n,160) or a.dtype!=np.float32 or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):
                    raise ValueError('Paired original/candidate finite unit features required for truth-side cross-RX diagnostic')
        rows.append(data)
    if set(rows[0]['ids'].tolist())&set(rows[1]['ids'].tolist()):raise ValueError('Source V and novel physical IDs must be disjoint')
    return marker,cal,thresholds,rows[0],rows[1]


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable per-arm joint-OVA score root is required')
    marker,cal,thresholds,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # All default/aux logits, features, margins and calibration precede truth.
    km=dict(_masks(known['rx_i'],known['scenario']));um=dict(_masks(novel['rx_i'],truth['scenario']));methods={}
    for j,name in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            row=open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],truth['tx'][umask],
                novel['scores'][umask,j],0.,known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
            closed=[float(np.mean(known['closed'][kmask&(known['y']==c)]==c)) if np.any(kmask&(known['y']==c)) else None for c in known_roster]
            candidate=None if any(v is None for v in closed) else float(np.mean(closed))
            row.update(candidate_macro_closed_accuracy=candidate,candidate_closed_minus_open=None if candidate is None or row['known_macro_open_accuracy'] is None else candidate-row['known_macro_open_accuracy'],
                original_F0_minus_candidate_closed=None if candidate is None or row['original_F0_macro_closed_accuracy'] is None else row['original_F0_macro_closed_accuracy']-candidate)
            return row
        groups={key:calculate(mask,um[key]) for key,mask in km.items()};overall=groups['overall']
        methods[name]=dict(overall=overall,groups=groups,threshold=0.,
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='per true-class Wilson, descriptive source pool only',
                A_K_per_class=_ci(overall['known_per_class'],'correct_accepted'),R_U_per_class=_ci(overall['unknown_per_class'],'rejected')),
            goals=_goals(groups,cal['methods'][name].get('constraints_passed') is True),diagnostic_only=name!=PRIMARY or marker['arm']!=PRIMARY_ARM)
    # Stable opaque-ID tie breaking, other-RX candidates only. TX is used only
    # after neighbor selection by the pure source diagnostic helper.
    retrieval={name:cross_rx_retrieval(novel[key],novel['ids'],novel['rx_i'],truth['tx'],UNKNOWN_TX)
        for name,key in (('candidate','candidate_fused_unit'),('original_F0','baseline_fused_unit'))}
    current,baseline=(retrieval[k]['macro_hit_rate'] for k in ('candidate','original_F0'))
    drop=None if current is None or baseline is None else baseline-current
    guard=dict(metric='source-novel macro cross-source-RX 1NN hit rate',candidate_macro_hit_rate=current,original_F0_macro_hit_rate=baseline,
        macro_drop_from_original=drop,max_allowed_drop=.02,passed=drop is not None and drop<=.02+1e-12,
        source_only_structural_proxy=True,not_anonymous_registration_accuracy=True,threshold_or_parameter_fitting=0)
    method_pass=methods[PRIMARY]['goals']['passed'];arm_pass=method_pass and guard['passed'];eligible=arm_pass and marker['arm']==PRIMARY_ARM
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],arm=marker['arm'],method_names=list(METHODS),methods=methods,
        primary_arm=PRIMARY_ARM,primary_method=PRIMARY,primary_path='P1',primary_summary=methods[PRIMARY]['overall'],
        primary_method_source_gate_passed=method_pass,novel_structure_guard=guard,novel_cross_RX_retrieval=retrieval,
        arm_source_gate_passed=bool(arm_pass),eligible_primary=bool(eligible),
        next_step='SOURCE_JOINT_OVA_READY_FOR_TARGET_FALSIFICATION' if eligible else 'NO_PROMOTION',
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,
        known_count=len(known['ids']),novel_count=len(novel['ids']),known_protection_groups=42,
        source_RX_is_seen=True,source_novel_historically_exposed=True,formal_target_accessed=False,final_phase3_complete=False,
        A_N=None,H_OW=None,full_goal_status='ACTIVE',
        limits=dict(known_macro_acceptance_min=.90,original_F0_open_loss_max=.02,source_unknown_rejection_min=.90,novel_cross_RX_1NN_drop_max=.02),
        baseline='cache original F0 same-input P1 closed; never candidate closed',
        primary_rule='Only joint/ova_max with known42, unknown overall/5RX/6scene and structure guard can prepare a separate target falsification; no control replacement',
        retrieval_scope='Same five source RX and historically exposed novel; no future-Q anonymous accuracy, prototype fitting or source bank deployment')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','novel_cross_RX_retrieval')})
    print(STATUS,marker['arm'],result['next_step'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_JOINT_OVA_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);score(a.prediction_root,a.novel_truth,a.output);return 0


if __name__=='__main__':raise SystemExit(main())
