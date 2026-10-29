"""Truth-last source evaluation of frozen predicted-class calibration margins.

The operating threshold is zero for both completed columns. Conditional
thresholds route only by the frozen P1 head prediction; true TX macro metrics
are computed independently here after all predictions have been checked.
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

RUN_ID='20261029-phase3-star-class-risk-s2026092701-r01'
PARENT_RUN_ID='20261029-phase3-star-paired-evidence-s2026092701-r01'
PARENT_CODE='bcc873cd9f9273133d74b5bc591c947b37ce16e1'
PREDICTION_SCHEMA='pamf_phase3_star_class_risk_prediction_v1'
PREDICTION_STATUS='STAR_CLASS_RISK_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CLASS_RISK_THRESHOLDS_FROZEN'
SCHEMA='pamf_phase3_star_class_risk_scores_v1'
STATUS='STAR_CLASS_RISK_SCORES_COMPLETE'
METHODS=('global_control','predicted_class')
PRIMARY=METHODS[-1]
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','closed'}
_read,_write,_ids,_truth=common._read,common._write,common._ids,common._truth
_masks,_goals,_delta=common._masks,common._goals,common._delta


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('schema')!=PREDICTION_SCHEMA or marker.get('status')!=PREDICTION_STATUS or marker.get('run_id')!=RUN_ID
            or not isinstance(marker.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',marker['code_commit']) is None
            or marker.get('parent_run_id')!=PARENT_RUN_ID or marker.get('parent_code_commit')!=PARENT_CODE
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_method')!=PRIMARY
            or marker.get('classification_path')!='P1' or marker.get('routing')!='predicted_class'
            or marker.get('use_predicted_class_only') is not True or marker.get('score_threshold')!=0.
            or marker.get('known_predictions_ref')!='predict/known_predictions.npz'
            or marker.get('novel_predictions_ref')!='predict/novel_predictions.npz' or marker.get('source_only') is not True
            or any(marker.get(key)!=0 for key in ('target_reads','source_novel_truth_reads','new_encoder_updates','new_identity_head_updates','new_readout_updates'))
            or any(marker.get(key) is not True for key in ('all_thresholds_frozen_before_novel','model_parameters_buffers_unchanged'))):
        raise ValueError('Both source-only frozen predicted-class margin columns must complete before truth')
    calibration=_read(root/'calibration/complete.json')
    with np.load(root/'thresholds.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'global','conditional'}:raise ValueError('One global and six predicted-class source thresholds required')
        thresholds={key:saved[key].copy() for key in saved.files}
    if (thresholds['global'].dtype!=np.float64 or thresholds['global'].shape!=(1,)
            or thresholds['conditional'].dtype!=np.float64 or thresholds['conditional'].shape!=(6,)
            or any(not np.isfinite(value).all() for value in thresholds.values())
            or calibration.get('schema')!=PREDICTION_SCHEMA or calibration.get('status')!=CALIBRATION_STATUS
            or any(calibration.get(key)!=marker[key] for key in ('run_id','code_commit'))
            or calibration.get('method_names')!=list(METHODS) or calibration.get('group_count')!=42
            or calibration.get('known_only') is not True or calibration.get('unknown_inputs_used') is not False
            or calibration.get('routing')!='predicted_class' or calibration.get('score_threshold')!=0.
            or set(calibration.get('methods',{}))!=set(METHODS)
            or calibration.get('global_threshold')!=float(thresholds['global'][0])
            or not np.array_equal(np.asarray(calibration.get('conditional_thresholds')),thresholds['conditional'])):
        raise ValueError('Matching complete known-only global and predicted-class thresholds required')
    rows=[]
    for role,fields,count_key in (('known',KNOWN_FIELDS,'known_count'),('novel',NOVEL_FIELDS,'novel_count')):
        n=marker.get(count_key)
        if type(n) is not int or n<=0:raise ValueError('Positive complete source prediction counts required')
        with np.load(root/marker[role+'_predictions_ref'],allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Exact public predictions required; novel TX/scene/labels forbidden')
            data={key:saved[key].copy() for key in saved.files}
        if (not _ids(data['ids'],n) or data['scores'].dtype!=np.float64 or data['scores'].shape!=(n,2)
                or not np.isfinite(data['scores']).all() or data['closed'].dtype!=np.int64 or data['closed'].shape!=(n,)
                or np.any((data['closed']<0)|(data['closed']>=6))):
            raise ValueError('Unique IDs, two finite FP64 margins and aligned frozen six-class P1 predictions required')
        for key,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
            if data[key].dtype!=np.int64 or data[key].shape!=(n,) or not np.isin(data[key],roster).all():
                raise ValueError('Aligned original source RX/day metadata required')
        routed=data['scores'][:,0]+thresholds['global'][0]-thresholds['conditional'][data['closed']]
        if not np.allclose(data['scores'][:,1],routed,rtol=1e-12,atol=1e-12):
            raise ValueError('Both margins must use the same raw score and only frozen-head predicted-class routing')
        if role=='known':
            if (data['y'].dtype!=np.int64 or data['y'].shape!=(n,) or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].dtype.kind!='U' or data['scenario'].shape!=(n,) or not np.isin(data['scenario'],SCENES).all()
                    or data['baseline_closed'].dtype!=np.int64 or data['baseline_closed'].shape!=(n,)
                    or not np.array_equal(data['closed'],data['baseline_closed'])):
                raise ValueError('Source V labels/scenes and unchanged same-input original F0 P1 closed baseline required')
        rows.append(data)
    if set(rows[0]['ids'].tolist())&set(rows[1]['ids'].tolist()):raise ValueError('Known V and heldout novel opaque IDs must be separate')
    return marker,calibration,thresholds,rows[0],rows[1]


def _pair_counts(primary_accept,global_accept,primary_reject,global_reject,correct,known_mask,unknown_mask):
    pa=primary_accept[known_mask];ga=global_accept[known_mask];identity=correct[known_mask]
    pr=primary_reject[unknown_mask];gr=global_reject[unknown_mask]
    return dict(known_accept_only_primary=int(np.sum(pa&~ga)),known_accept_only_global=int(np.sum(ga&~pa)),
        known_accept_net=int(pa.sum()-ga.sum()),known_correct_accept_only_primary=int(np.sum(pa&~ga&identity)),
        known_correct_accept_only_global=int(np.sum(ga&~pa&identity)),
        known_correct_accept_net=int(np.sum(pa&identity)-np.sum(ga&identity)),
        novel_reject_only_primary=int(np.sum(pr&~gr)),novel_reject_only_global=int(np.sum(gr&~pr)),
        unknown_rejected_net=int(pr.sum()-gr.sum()))


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable class-risk score root is required')
    marker,calibration,thresholds,known,novel=_prediction(prediction_root)
    # Sole private truth access follows both margin columns and calibration checks.
    truth=_truth(novel_truth,novel)
    known_masks=dict(_masks(known['rx_i'],known['scenario']));unknown_masks=dict(_masks(novel['rx_i'],truth['scenario']))
    methods={}
    for j,method in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            return open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],
                truth['tx'][umask],novel['scores'][umask,j],0.,known_roster,unknown_roster,
                baseline_prediction=known['baseline_closed'][kmask])
        groups={key:calculate(mask,unknown_masks[key]) for key,mask in known_masks.items()}
        methods[method]=dict(overall=groups['overall'],groups=groups,threshold=0.,
            by_known_TX={str(label):calculate(known['y']==label,unknown_masks['overall'],(label,)) for label in range(6)},
            by_unknown_TX={tx:calculate(known_masks['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            goals=_goals(groups,calibration['methods'][method].get('constraints_passed') is True),diagnostic_only=method!=PRIMARY)
    primary=methods[PRIMARY];reference=methods['global_control']
    pa=known['scores'][:,1]>=0.;ga=known['scores'][:,0]>=0.
    pr=novel['scores'][:,1]<0.;gr=novel['scores'][:,0]<0.;correct=known['closed']==known['y']
    def counts(kmask,umask):return _pair_counts(pa,ga,pr,gr,correct,kmask,umask)
    paired=dict(primary_method=PRIMARY,reference_method='global_control',direction='predicted_class minus global_control',
        same_physical_IDs_and_P1_default_head=True,operating_margin_threshold=0.,
        threshold_scope='pre-frozen known-V global tau or predicted-head-class tau; true labels never route inference',
        overall=_delta(primary['overall'],reference['overall']),
        groups={key:_delta(primary['groups'][key],reference['groups'][key]) for key in GROUP_NAMES},
        counts=counts(known_masks['overall'],unknown_masks['overall']),
        group_counts={key:counts(known_masks[key],unknown_masks[key]) for key in GROUP_NAMES})
    passed=primary['goals']['passed']
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],
        parent_run_id=marker['parent_run_id'],parent_code_commit=marker['parent_code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY,classification_path='P1',routing='predicted_class',
        score_threshold=0.,methods=methods,primary_summary=primary['overall'],primary_goals=primary['goals'],
        paired_primary_vs_global=paired,source_gate_passed=passed,
        next_step='SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION' if passed else 'NO_PROMOTION',
        global_threshold=float(thresholds['global'][0]),conditional_thresholds=thresholds['conditional'].tolist(),
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,
        known_count=len(known['ids']),novel_count=len(novel['ids']),known_protection_groups=42,
        limits=dict(schema='pamf_phase3_scientific_targets_v3',known_macro_acceptance_min=.90,known_macro_frr_max=.10,
            source_unknown_macro_rejection_min=.90,original_F0_open_loss_max=.02),
        required_unknown_groups='overall + 5 source RX + 6 scenes; RX-scene unknown diagnostic only',
        averaging='equal per required true TX, never predicted-class groups; H-score from macro A_K and R_U, never C_K',
        curves='AUROC/OSCR use fixed margins; predicted-class offsets can change cross-class ranking; no oracle threshold selection',
        conditional_shift_can_change_ranking=True,conformal_guarantee=False,
        source_RX_is_seen=True,source_novel_historically_exposed=True,target_generalization_confirmed=False,
        formal_target_accessed=False,final_phase3_complete=False,A_N=None,H_OW=None,
        primary_rule='only predicted_class may advance; global_control cannot replace the predeclared primary')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{key:value for key,value in result.items() if key not in ('methods','paired_primary_vs_global')})
    print(STATUS,result['next_step'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_CLASS_RISK_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prediction-root',required=True)
    parser.add_argument('--novel-truth',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);score(args.prediction_root,args.novel_truth,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
