"""Independent truth-last source consensus scores under Phase3 targets v3.

No model, IQ, fitting, threshold choice or target access occurs here. Source
heldout TX labels are read only after every P0/P1 score and source threshold
is complete. Only the predeclared P1/MLP candidate can become source-eligible.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from .metrics import open_set_metrics

RUN_ID='20261029-phase3-star-consensus-s2026092701-r01'
PREDICTION_SCHEMA='pamf_phase3_star_consensus_prediction_v1'
PREDICTION_STATUS='STAR_CONSENSUS_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CONSENSUS_THRESHOLDS_FROZEN'
SCHEMA='pamf_phase3_star_consensus_scores_v1'
STATUS='STAR_CONSENSUS_SCORES_COMPLETE'
METHODS=('msp','maxlogit','margin','energy','cosine','linear_oe','mlp16_oe')
PATHS=('P0','P1')
PRIMARY_PATH='P1'
PRIMARY_METHOD='mlp16_oe'
RECEIVERS=(1,3,4,6,8)
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
UNKNOWN_TX=('19-15','19-16','7-1')  # Scorer-only required source-heldout roster.
GROUP_NAMES=('overall',)+tuple('rx/'+str(r) for r in RECEIVERS)+tuple('scene/'+s for s in SCENES)+tuple(
    'rxscene/'+str(r)+'/'+s for r in RECEIVERS for s in SCENES)
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','closed'}
_DELTA_FIELDS=('known_macro_acceptance','known_macro_open_accuracy','unknown_macro_rejection','H_OS',
               'auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _ids(ids,n):
    return ids.shape==(n,) and ids.dtype.kind=='U' and len(set(ids.tolist()))==n and not np.any(np.char.strip(ids)=='')


def _masks(rx,scenario):
    yield 'overall',np.ones(len(rx),bool)
    for r in RECEIVERS:yield 'rx/'+str(r),rx==r
    for s in SCENES:yield 'scene/'+s,scenario==s
    for r in RECEIVERS:
        for s in SCENES:yield 'rxscene/'+str(r)+'/'+s,(rx==r)&(scenario==s)


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('status')!=PREDICTION_STATUS or marker.get('schema')!=PREDICTION_SCHEMA or marker.get('run_id')!=RUN_ID
            or not isinstance(marker.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',marker['code_commit']) is None
            or marker.get('path_names')!=list(PATHS) or marker.get('method_names')!=list(METHODS)
            or marker.get('primary_path')!=PRIMARY_PATH or marker.get('primary_method')!=PRIMARY_METHOD
            or marker.get('known_predictions_ref')!='predict/known_predictions.npz'
            or marker.get('novel_predictions_ref')!='predict/novel_predictions.npz'
            or marker.get('source_only') is not True
            or any(marker.get(k)!=0 for k in ('target_reads','source_novel_truth_reads','new_encoder_updates','new_identity_head_updates'))
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged','L_aggregates_unchanged_during_V','all_thresholds_frozen_before_novel'))):
        raise ValueError('All source-only frozen P0/P1 seven-score predictions must complete before truth access')
    calibration={};thresholds={}
    for path in PATHS:
        folder=root/'by_path'/path;report=_read(folder/'calibration/complete.json')
        threshold=np.load(folder/'thresholds.npy',allow_pickle=False)
        if (report.get('status')!=CALIBRATION_STATUS or report.get('schema')!=PREDICTION_SCHEMA
                or any(report.get(k)!=marker[k] for k in ('run_id','code_commit')) or report.get('path')!=path
                or report.get('method_names')!=list(METHODS) or report.get('group_count')!=42
                or report.get('known_only') is not True or report.get('unknown_inputs_used') is not False
                or set(report.get('methods',{}))!=set(METHODS)
                or threshold.shape!=(7,) or threshold.dtype!=np.float64 or not np.isfinite(threshold).all()
                or any(report['methods'][name].get('threshold')!=float(threshold[j]) for j,name in enumerate(METHODS))):
            raise ValueError('Matching completed source known-only thresholds required for both paths')
        calibration[path]=report;thresholds[path]=threshold
    rows=[]
    for role,fields,count_key in (('known',KNOWN_FIELDS,'known_count'),('novel',NOVEL_FIELDS,'novel_count')):
        n=marker.get(count_key)
        if type(n) is not int or n<=0:raise ValueError('Positive complete prediction counts required')
        with np.load(root/marker[role+'_predictions_ref'],allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Only exact prediction fields allowed; novel truth/labels/scenes forbidden')
            data={key:saved[key].copy() for key in saved.files}
        if (not _ids(data['ids'],n) or data['scores'].shape!=(n,2,7) or data['scores'].dtype!=np.float64
                or not np.isfinite(data['scores']).all() or data['closed'].shape!=(n,2) or data['closed'].dtype!=np.int64
                or np.any((data['closed']<0)|(data['closed']>=6))):
            raise ValueError('Unique aligned IDs, finite FP64 scores and complete original six-class predictions required')
        for key,roster in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
            if data[key].shape!=(n,) or data[key].dtype!=np.int64 or not np.isin(data[key],roster).all():
                raise ValueError('Aligned original source RX/day metadata required')
        if role=='known':
            if (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or np.any((data['y']<0)|(data['y']>=6))
                    or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                    or data['baseline_closed'].shape!=(n,2) or data['baseline_closed'].dtype!=np.int64
                    or not np.array_equal(data['baseline_closed'],data['closed'])):
                raise ValueError('Known V labels/scenes and unchanged same-path original F0 baselines required')
        rows.append(data)
    if set(rows[0]['ids'].tolist())&set(rows[1]['ids'].tolist()):raise ValueError('Known V and source-heldout novel IDs must be separate')
    return marker,calibration,thresholds,rows[0],rows[1]


def _truth(path,novel):
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!={'ids','tx','rx_i','day_i','scenario'}:raise ValueError('Exact independent source novel truth fields required')
        data={key:saved[key].copy() for key in saved.files}
    n=len(novel['ids'])
    if not _ids(data['ids'],n) or set(data['ids'].tolist())!=set(novel['ids'].tolist()) or any(a.shape!=(n,) for a in data.values()):
        raise ValueError('Truth must uniquely cover exactly all completed novel opaque IDs')
    lookup={sid:i for i,sid in enumerate(data['ids'].tolist())};order=np.asarray([lookup[sid] for sid in novel['ids'].tolist()],np.int64)
    data={key:value[order] for key,value in data.items()}
    if (data['tx'].dtype.kind!='U' or not np.isin(data['tx'],UNKNOWN_TX).all()
            or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
            or any(data[key].dtype!=np.int64 or not np.array_equal(data[key],novel[key]) for key in ('rx_i','day_i'))):
        raise ValueError('Source-heldout TX/scenes and public RX/day alignment differ')
    return data


def _protected(row):
    return (row['coverage']['known_complete'] and row['known_macro_acceptance'] is not None
            and row['known_macro_acceptance']>=.90-1e-12 and row['original_closed_minus_new_open'] is not None
            and row['original_closed_minus_new_open']<=.02+1e-12)


def _goals(groups,calibrated):
    if set(groups)!=set(GROUP_NAMES):raise ValueError('All 42 predeclared source groups are required')
    known=all(_protected(row) for row in groups.values())
    required=GROUP_NAMES[:12]
    unknown=all(groups[key]['coverage']['unknown_complete'] and groups[key]['unknown_macro_rejection'] is not None
                and groups[key]['unknown_macro_rejection']>=.90-1e-12 for key in required)
    return dict(known_42_groups_passed=known,source_known_calibration_passed=bool(calibrated),
        unknown_overall_five_RX_six_scenes_90_passed=unknown,unknown_RX_scene_diagnostic_only=True,
        passed=bool(known and calibrated and unknown))


def _delta(left,right):
    return {key:None if left[key] is None or right[key] is None else float(left[key]-right[key]) for key in _DELTA_FIELDS}


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent source-score output root is required')
    marker,calibration,thresholds,known,novel=_prediction(prediction_root)
    # Sole private truth access: all input checks and both frozen paths precede it.
    truth=_truth(novel_truth,novel)
    known_masks=dict(_masks(known['rx_i'],known['scenario']));unknown_masks=dict(_masks(novel['rx_i'],truth['scenario']))
    paths={}
    for pi,path in enumerate(PATHS):
        methods={}
        for j,method in enumerate(METHODS):
            tau=thresholds[path][j]
            def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
                return open_set_metrics(known['y'][kmask],known['closed'][kmask,pi],known['scores'][kmask,pi,j],
                    truth['tx'][umask],novel['scores'][umask,pi,j],tau,known_roster,unknown_roster,
                    baseline_prediction=known['baseline_closed'][kmask,pi])
            groups={key:calculate(mask,unknown_masks[key]) for key,mask in known_masks.items()}
            methods[method]=dict(overall=groups['overall'],groups=groups,threshold=float(tau),
                by_known_TX={str(label):calculate(known['y']==label,unknown_masks['overall'],(label,)) for label in range(6)},
                by_unknown_TX={tx:calculate(known_masks['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
                goals=_goals(groups,calibration[path]['methods'][method].get('constraints_passed') is True),
                diagnostic_only=path!=PRIMARY_PATH or method!=PRIMARY_METHOD)
        paths[path]=dict(methods=methods,MLP_source_gate_passed=methods[PRIMARY_METHOD]['goals']['passed'])
    main=paths[PRIMARY_PATH]['methods'][PRIMARY_METHOD];reference=paths[PRIMARY_PATH]['methods']['energy']
    passed=main['goals']['passed'];pi=PATHS.index(PRIMARY_PATH)
    ka=known['scores'][:,pi,6]>=thresholds[PRIMARY_PATH][6];ke=known['scores'][:,pi,3]>=thresholds[PRIMARY_PATH][3]
    ur=novel['scores'][:,pi,6]<thresholds[PRIMARY_PATH][6];ue=novel['scores'][:,pi,3]<thresholds[PRIMARY_PATH][3]
    correct=known['closed'][:,pi]==known['y']
    paired=dict(path=PRIMARY_PATH,primary_method=PRIMARY_METHOD,reference_method='energy',same_physical_IDs_and_original_head=True,
        threshold_scope='each method uses its own pre-frozen source-known-only global threshold',
        overall=_delta(main['overall'],reference['overall']),groups={key:_delta(main['groups'][key],reference['groups'][key]) for key in GROUP_NAMES},
        counts=dict(known_accept_only_primary=int(np.sum(ka&~ke)),known_accept_only_energy=int(np.sum(ke&~ka)),
            known_correct_accept_only_primary=int(np.sum(ka&~ke&correct)),known_correct_accept_only_energy=int(np.sum(ke&~ka&correct)),
            novel_reject_only_primary=int(np.sum(ur&~ue)),novel_reject_only_energy=int(np.sum(ue&~ur))))
    result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=marker['code_commit'],path_names=list(PATHS),method_names=list(METHODS),
        primary_path=PRIMARY_PATH,primary_method=PRIMARY_METHOD,primary_summary=main['overall'],primary_goals=main['goals'],
        paths=paths,paired_primary_vs_energy=paired,source_gate_passed=passed,
        next_step='SOURCE_ELIGIBLE_FOR_FROZEN_TARGET_VALIDATION' if passed else 'NO_PROMOTION',
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,
        known_count=len(known['ids']),novel_count=len(novel['ids']),known_protection_groups=42,
        limits=dict(schema='pamf_phase3_scientific_targets_v3',known_macro_acceptance_min=.90,
            known_macro_frr_max=.10,source_unknown_macro_rejection_min=.90,original_F0_open_loss_max=.02),
        required_unknown_groups='overall + 5 source RX + 6 scenes; RX-scene unknown diagnostic only',
        averaging='equal per required true TX; H-score from macro A_K and R_U, never C_K',
        source_novel_historically_exposed=True,source_RX_is_seen=True,target_generalization_confirmed=False,
        formal_target_accessed=False,final_phase3_complete=False,A_N=None,H_OW=None,
        primary_rule='P1/mlp16_oe fixed before source observations; P0, linear and static scores cannot replace it')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{key:value for key,value in result.items() if key not in ('paths','paired_primary_vs_energy')})
    print(STATUS,result['next_step'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_CONSENSUS_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prediction-root',required=True,
        help='source run output root containing complete.json, by_path and predict/')
    parser.add_argument('--novel-truth',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);score(args.prediction_root,args.novel_truth,args.output);return 0


if __name__=='__main__':raise SystemExit(main())
