"""Truth-last source scoring of the fixed paired-evidence 2x2 matrix.

Only completed scalar predictions are read. No model, IQ, fitting, threshold
selection or target access occurs. Step200 is a snapshot within each matched
1000-update trajectory, not an additional 200-step optimization.
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

RUN_ID='20261029-phase3-star-paired-evidence-s2026092701-r01'
PREDICTION_SCHEMA='pamf_phase3_star_paired_prediction_v1'
PREDICTION_STATUS='STAR_PAIRED_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_PAIRED_THRESHOLDS_FROZEN'
SCHEMA='pamf_phase3_star_paired_scores_v1'
STATUS='STAR_PAIRED_SCORES_COMPLETE'
METHODS=('energy','p1_mlp16_step200','paired_mlp16_step200','p1_mlp16_step1000','paired_mlp16_step1000')
PRIMARY=METHODS[-1]
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','closed'}
COMPARISONS=(('input_step200',METHODS[2],METHODS[1]),('input_step1000',METHODS[4],METHODS[3]),
    ('budget_p1_only',METHODS[3],METHODS[1]),('budget_paired',METHODS[4],METHODS[2]),('primary_vs_energy',METHODS[4],METHODS[0]))
_read,_write,_ids,_truth=common._read,common._write,common._ids,common._truth
_masks,_goals,_delta=common._masks,common._goals,common._delta


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('status')!=PREDICTION_STATUS or marker.get('schema')!=PREDICTION_SCHEMA or marker.get('run_id')!=RUN_ID
            or not isinstance(marker.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',marker['code_commit']) is None
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_method')!=PRIMARY
            or marker.get('classification_path')!='P1' or marker.get('rejection_input_paths')!=['P1','P0']
            or marker.get('control_p0_zeroed') is not True or marker.get('source_only') is not True
            or marker.get('known_predictions_ref')!='predict/known_predictions.npz'
            or marker.get('novel_predictions_ref')!='predict/novel_predictions.npz'
            or any(marker.get(k)!=0 for k in ('target_reads','source_novel_truth_reads','new_encoder_updates','new_identity_head_updates'))
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged','L_aggregates_unchanged_during_V','all_thresholds_frozen_before_novel'))):
        raise ValueError('All five fixed source-only predictions must complete before truth access')
    calibration=_read(root/'calibration/complete.json');thresholds=np.load(root/'thresholds.npy',allow_pickle=False)
    if (calibration.get('status')!=CALIBRATION_STATUS or calibration.get('schema')!=PREDICTION_SCHEMA
            or any(calibration.get(k)!=marker[k] for k in ('run_id','code_commit'))
            or calibration.get('method_names')!=list(METHODS) or calibration.get('group_count')!=42
            or calibration.get('known_only') is not True or calibration.get('unknown_inputs_used') is not False
            or set(calibration.get('methods',{}))!=set(METHODS) or thresholds.shape!=(5,)
            or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all()
            or any(calibration['methods'][name].get('threshold')!=float(thresholds[j]) for j,name in enumerate(METHODS))):
        raise ValueError('Five matching known-only frozen global thresholds required')
    rows=[]
    for role,fields,count_key in (('known',KNOWN_FIELDS,'known_count'),('novel',NOVEL_FIELDS,'novel_count')):
        n=marker.get(count_key)
        if type(n) is not int or n<=0:raise ValueError('Positive completed source prediction counts required')
        with np.load(root/marker[role+'_predictions_ref'],allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Exact source prediction fields required; novel labels/scenes/truth forbidden')
            data={key:saved[key].copy() for key in saved.files}
        if (not _ids(data['ids'],n) or data['scores'].shape!=(n,5) or data['scores'].dtype!=np.float64
                or not np.isfinite(data['scores']).all() or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64
                or np.any((data['closed']<0)|(data['closed']>=6))):
            raise ValueError('Unique IDs, finite FP64 five-score rows and one frozen full-six P1 identity required')
        for key,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
            if data[key].shape!=(n,) or data[key].dtype!=np.int64 or not np.isin(data[key],roster).all():
                raise ValueError('Original source RX/day metadata required')
        if role=='known':
            if (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                    or data['baseline_closed'].shape!=(n,) or data['baseline_closed'].dtype!=np.int64
                    or not np.array_equal(data['closed'],data['baseline_closed'])):
                raise ValueError('Source known V and original F0 P1 same-input closed baseline must remain aligned')
        rows.append(data)
    if set(rows[0]['ids'].tolist())&set(rows[1]['ids'].tolist()):raise ValueError('Known V and heldout novel physical IDs must be separate')
    return marker,calibration,thresholds,rows[0],rows[1]


def _pair_counts(left_accept,right_accept,left_reject,right_reject,correct,known_mask,unknown_mask):
    left=left_accept[known_mask];right=right_accept[known_mask];identity=correct[known_mask]
    lr=left_reject[unknown_mask];rr=right_reject[unknown_mask]
    return dict(known_accept_only_left=int(np.sum(left&~right)),known_accept_only_right=int(np.sum(right&~left)),
        known_accept_net=int(left.sum()-right.sum()),known_correct_accept_only_left=int(np.sum(left&~right&identity)),
        known_correct_accept_only_right=int(np.sum(right&~left&identity)),
        known_correct_accept_net=int(np.sum(left&identity)-np.sum(right&identity)),
        unknown_rejected_only_left=int(np.sum(lr&~rr)),unknown_rejected_only_right=int(np.sum(rr&~lr)),
        unknown_rejected_net=int(lr.sum()-rr.sum()))


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable paired-source score root is required')
    marker,calibration,thresholds,known,novel=_prediction(prediction_root)
    # Sole private truth access follows completion and validation of all five columns.
    truth=_truth(novel_truth,novel)
    known_masks=dict(_masks(known['rx_i'],known['scenario']));unknown_masks=dict(_masks(novel['rx_i'],truth['scenario']))
    methods={}
    for j,method in enumerate(METHODS):
        threshold=thresholds[j]
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            return open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],
                truth['tx'][umask],novel['scores'][umask,j],threshold,known_roster,unknown_roster,
                baseline_prediction=known['baseline_closed'][kmask])
        groups={key:calculate(mask,unknown_masks[key]) for key,mask in known_masks.items()}
        methods[method]=dict(overall=groups['overall'],groups=groups,threshold=float(threshold),
            by_known_TX={str(label):calculate(known['y']==label,unknown_masks['overall'],(label,)) for label in range(6)},
            by_unknown_TX={tx:calculate(known_masks['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            goals=_goals(groups,calibration['methods'][method].get('constraints_passed') is True),diagnostic_only=method!=PRIMARY)
    correct=known['closed']==known['y'];paired={}
    for key,left,right in COMPARISONS:
        li,ri=METHODS.index(left),METHODS.index(right)
        la=known['scores'][:,li]>=thresholds[li];ra=known['scores'][:,ri]>=thresholds[ri]
        lr=novel['scores'][:,li]<thresholds[li];rr=novel['scores'][:,ri]<thresholds[ri]
        def counts(kmask,umask):return _pair_counts(la,ra,lr,rr,correct,kmask,umask)
        paired[key]=dict(left_method=left,right_method=right,direction='left minus right',
            factor='input' if key.startswith('input_') else 'budget' if key.startswith('budget_') else 'reference',
            overall=_delta(methods[left]['overall'],methods[right]['overall']),
            groups={group:_delta(methods[left]['groups'][group],methods[right]['groups'][group]) for group in GROUP_NAMES},
            counts=counts(known_masks['overall'],unknown_masks['overall']),
            group_counts={group:counts(known_masks[group],unknown_masks[group]) for group in GROUP_NAMES},
            same_physical_IDs_and_P1_default_head=True,
            thresholds='method-specific frozen source-known calibration; no rescanning')
    main=methods[PRIMARY];passed=main['goals']['passed']
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],method_names=list(METHODS),
        primary_method=PRIMARY,classification_path='P1',rejection_input_paths=['P1','P0'],methods=methods,
        primary_summary=main['overall'],primary_goals=main['goals'],paired_comparisons=paired,
        source_gate_passed=passed,next_step='SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION' if passed else 'NO_PROMOTION',
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,
        known_count=len(known['ids']),novel_count=len(novel['ids']),known_protection_groups=42,
        limits=dict(schema='pamf_phase3_scientific_targets_v3',known_macro_acceptance_min=.9,known_macro_frr_max=.1,
            source_unknown_macro_rejection_min=.9,original_F0_open_loss_max=.02),
        required_unknown_groups='overall + 5 source RX + 6 scenes; RX-scene unknown diagnostic only',
        averaging='equal per required TX; H-score from macro A_K and R_U, never C_K',
        snapshot_budget='two 1000-update trajectories, total 2000 actual updates; step200 snapshots do not add 200 updates',
        source_RX_is_seen=True,source_novel_historically_exposed=True,target_generalization_confirmed=False,
        formal_target_accessed=False,final_phase3_complete=False,A_N=None,H_OW=None,
        primary_rule='only paired_mlp16_step1000 may advance; energy and 200-step/P1 controls cannot replace it')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{key:value for key,value in result.items() if key not in ('methods','paired_comparisons')})
    print(STATUS,result['next_step'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_PAIRED_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prediction-root',required=True,
        help='producer output root with complete.json, thresholds.npy, calibration/ and predict/')
    parser.add_argument('--novel-truth',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);score(args.prediction_root,args.novel_truth,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
