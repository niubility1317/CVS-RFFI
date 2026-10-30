"""Independent truth-last source and cross-RX PAMF-OW metrics, without fitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import star_open_world_eval as evaluation
from . import star_consensus_score as common
from .metrics import open_set_metrics
from .scoring import _truth as target_truth_load


RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_open_world_scores_v1'
STATUS='STAR_OPEN_WORLD_SCORES_COMPLETE'
TARGET_UNKNOWN=(6,7,8)  # Private scorer roster; never passed to prediction.
_read,_write=common._read,common._write


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method=evaluation.PRIMARY_METHOD,
        primary_arm='M',path='P1',prototype_count=6,prototype_fit_count=evaluation.COUNTS['L'],
        known_count=evaluation.COUNTS['V'],novel_count=evaluation.COUNTS['novel'],target_count=evaluation.Q_COUNT,
        target_RX='24-1',target_receiver_index=26,source_truth_reads=0,target_truth_reads=0,target_U_reads=0,target_Q_fit=False,
        source_V_neural_fit=False,source_V_threshold_fit=True,prototypes_frozen_before_V=True,
        thresholds_frozen_before_novel_and_target=True,frozen_models_verified=True,source_aggregates_unchanged=True,
        source_unknown_success_target_entry_gate=False,eval_neural_updates=0,source_bank_deployed=False,
        per_record_source_rows_deployed=0,all_new_IQ_satellite=True,formal_target_accessed=False,target_DA=False,target_REG=False)
    if any(m.get(k)!=v for k,v in expected.items()) or m.get('arm') not in evaluation.ARMS:
        raise ValueError('All source and future-Q predictions must be complete and frozen before any private truth')
    c=evaluation.validate_config(m['config'])
    if any(c[k]!=m[k] for k in ('run_id','code_commit','arm')) or evaluation._identity(m['model_identity'],c)!=m['parent_F0_identity']:
        raise ValueError('Actual trained/config/parent identities differ')
    prototypes=np.load(root/'prototypes.npy',allow_pickle=False);tau=np.load(root/'thresholds.npy',allow_pickle=False)
    cal=_read(root/'calibration/complete.json')
    if (prototypes.shape!=(6,64) or prototypes.dtype!=np.float32 or not np.isfinite(prototypes).all()
            or not np.allclose(np.linalg.norm(prototypes.astype(np.float64),axis=1),1.,rtol=0.,atol=2e-6)
            or tau.shape!=(6,) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,
                code_commit=m['code_commit'],arm=m['arm'],method=evaluation.PRIMARY_METHOD,selected_calibration_method='predicted_class',
                prototypes_fit_role='L',known_count=evaluation.COUNTS['V'],group_count=42,known_only=True,
                source_V_neural_fit=False,source_V_threshold_fit=True,unknown_inputs_used=False,target_reads=0).items())
            or cal.get('conditional_thresholds')!=tau.tolist()):
        raise ValueError('Matching source-L prototypes and known-only predicted-class thresholds required')
    rows={}
    for role,fields,n in (('known',evaluation.KNOWN_FIELDS,evaluation.COUNTS['V']),
            ('novel',evaluation.NOVEL_FIELDS,evaluation.COUNTS['novel']),('target',evaluation.TARGET_FIELDS,evaluation.Q_COUNT)):
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=fields:raise ValueError('Only exact completed public predictions allowed; no novel/Q truth')
            a={k:z[k].copy() for k in z.files}
        if (not common._ids(a['ids'],n) or any(a[k].shape!=(n,6) or a[k].dtype!=np.float32 or not np.isfinite(a[k]).all()
                for k in ('default_logits','relation_logits'))
                or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
                or a['scores'].shape!=(n,) or a['scores'].dtype!=np.float64 or not np.isfinite(a['scores']).all()
                or not np.allclose(a['scores'],a['relation_logits'].max(1).astype(np.float64)-tau[a['closed']],rtol=0.,atol=1e-12)):
            raise ValueError('Finite shared relation margins must reconstruct from original default-head predicted routing')
        if role in ('known','target'):
            if a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.isin(a['baseline_closed'],range(6)).all():
                raise ValueError('Same-P1 original F0 default-head baseline required')
        if role!='target':
            for key,roster in (('rx_i',common.RECEIVERS),('day_i',(1,2,3))):
                if a[key].shape!=(n,) or a[key].dtype!=np.int64 or not np.isin(a[key],roster).all():
                    raise ValueError('Aligned source public RX/day metadata required')
        if role=='known':
            if (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all()
                    or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],common.SCENES).all()):
                raise ValueError('Visible known-V labels and fixed received scenario metadata required')
            rebuilt,report=evaluation.risk.calibrate_known(a['ids'],a['y'],a['closed'],a['baseline_closed'],
                a['relation_logits'].max(1).astype(np.float64),a['rx_i'],a['scenario'])
            if (not np.array_equal(rebuilt['conditional_thresholds'],tau)
                    or any(report[k]!=cal[k] for k in ('methods','coverage','constraints_passed','common_tail_rate'))):
                raise ValueError('Exact known-only class-risk calibration must reconstruct before any private truth')
        rows[role]=a
    if any(set(rows[a]['ids'].tolist())&set(rows[b]['ids'].tolist()) for a,b in (('known','novel'),('known','target'),('novel','target'))):
        raise ValueError('Source calibration/heldout and future Q physical IDs must be disjoint')
    return m,cal,rows


def _metrics(y,closed,scores,uy,uscores,baseline,known_roster=tuple(range(6)),unknown_roster=common.UNKNOWN_TX):
    row=open_set_metrics(y,closed,scores,uy,uscores,0.,known_roster,unknown_roster,baseline_prediction=baseline)
    rates=[float(np.mean(closed[y==c]==c)) if np.any(y==c) else None for c in known_roster]
    current=None if any(v is None for v in rates) else float(np.mean(rates))
    original=row['original_F0_macro_closed_accuracy'];opened=row['known_macro_open_accuracy']
    row.update(candidate_macro_closed_accuracy=current,
        original_F0_minus_candidate_closed=None if original is None or current is None else original-current,
        candidate_closed_minus_open=None if current is None or opened is None else current-opened,
        known_closed_correct=int((closed==y).sum()),known_correct_accepted=int(((closed==y)&(scores>=0)).sum()),
        known_accepted=int((scores>=0).sum()),known_rejected=int((scores<0).sum()),unknown_rejected=int((uscores<0).sum()),
        A_K=opened,C_K=row['known_macro_acceptance'],R_U=row['unknown_macro_rejection'],H_score=row['H_OS'])
    for entry in row['known_per_class']:
        entry['candidate_closed_correct']=int((closed[y==entry['label']]==entry['label']).sum())
        entry['rejected']=entry['count']-entry['accepted'];entry['wrong_accepted']=entry['accepted']-entry['correct_accepted']
    return row


def _known_pass(row):
    return bool(row['coverage']['known_complete'] and row['known_macro_acceptance'] is not None
        and row['known_macro_acceptance']>=.90-1e-12 and row['original_closed_minus_new_open'] is not None
        and row['original_closed_minus_new_open']<=.02+1e-12)


def _full_pass(row):
    return bool(_known_pass(row) and row['coverage']['unknown_complete']
        and row['unknown_macro_rejection'] is not None and row['unknown_macro_rejection']>=.90-1e-12)


def score(prediction_root,novel_truth,target_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent PAMF-OW score root required')
    m,cal,rows=_prediction(prediction_root)
    known,novel,target=(rows[k] for k in ('known','novel','target'))
    # First private reads: every source/Q array, source aggregate and threshold passed.
    nt=common._truth(novel_truth,novel);qt=target_truth_load(target_truth,target['ids'])
    if (not np.isin(qt['y'],range(9)).all() or not np.all(qt['receiver']==26)
            or not np.isin(qt['scenario'],common.SCENES).all()):
        raise ValueError('Only fixed RX24-1 G10 independent future-Q truth is valid')
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],nt['scenario']))
    def source_metric(kmask,nmask,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
        return _metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask],nt['tx'][nmask],
            novel['scores'][nmask],known['baseline_closed'][kmask],kr,ur)
    source_groups={key:source_metric(mask,nm[key]) for key,mask in km.items()}
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool)
    source=dict(overall=source_groups['overall'],groups=source_groups,
        by_RX={str(r):source_groups['rx/'+str(r)] for r in common.RECEIVERS},
        by_scene={s:source_groups['scene/'+s] for s in common.SCENES},
        by_day={str(d):source_metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
        by_TX={str(c):source_metric(known['y']==c,alln,(c,)) for c in range(6)}|
              {str(c):source_metric(allk,nt['tx']==c,ur=(c,)) for c in common.UNKNOWN_TX},
        known_only_calibration_passed=bool(cal['methods']['predicted_class']['constraints_passed']),
        known_42_groups_passed=all(_known_pass(r) for r in source_groups.values()),
        source_novel_historically_exposed=True,source_RX_seen=True,target_entry_gate=False)
    y=qt['y'];k=y<6;u=~k
    def qmetric(mask,kr=tuple(range(6)),ur=TARGET_UNKNOWN):
        return _metrics(y[mask&k],target['closed'][mask&k],target['scores'][mask&k],y[mask&u],target['scores'][mask&u],
                        target['baseline_closed'][mask&k],kr,ur)
    allq=np.ones(len(y),bool);overall=qmetric(allq)
    scenes={s:qmetric(qt['scenario']==s) for s in common.SCENES}
    target_result=dict(overall=overall,by_scene=scenes,
        by_RX={str(r):qmetric(qt['receiver']==r) for r in np.unique(qt['receiver'])},
        by_day={'2021_03_15':overall},day_scope='fixed future-Q physical role day; no query-side state fit',
        by_TX={str(c):qmetric((y==c)|u,(c,)) for c in range(6)}|
              {str(c):qmetric(k|(y==c),ur=(c,)) for c in TARGET_UNKNOWN},
        overall_passed=_full_pass(overall),all_six_scenes_passed=all(_full_pass(r) for r in scenes.values()),
        required_scene_coverage_complete=all(r['coverage']['complete'] for r in scenes.values()),
        source_target_RX_disjoint=True,historically_exposed_development_RX=True,new_blind_confirmation=False)
    target_passed=bool(_full_pass(overall) and target_result['all_six_scenes_passed'])
    eligible=bool(m['arm']==evaluation.PRIMARY_ARM and target_passed)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],primary_arm='M',
        primary_method=evaluation.PRIMARY_METHOD,path='P1',source=source,target=target_result,primary_summary=overall,
        target_overall_and_six_scenes_passed=target_passed,eligible_primary=eligible,
        next_step='SOURCE_OPEN_WORLD_QUALIFIED_FOR_SEPARATE_UDA' if eligible else 'NO_PROMOTION',
        source_unknown_success_target_entry_gate=False,truth_last=True,formal_target_accessed=False,
        target_U_fit=False,target_Q_fit=False,target_DA=False,target_REG=False,adaptation_success=None,registration_success=None,
        A_N=None,H_OW=None,final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',
        identifier_parameters=284834,open_world_parameters=51841,training_container_parameters=952563,
        costs={key:m[key] for key in ('candidate_real_P1_forward_rows','candidate_synthetic_forward_rows',
            'original_F0_real_P1_forward_rows','original_F0_synthetic_forward_rows','G_real_forward_rows','G_synthetic_forward_rows')},
        metric_scope='True-TX macro A_K/C_K/R_U/H; missing required TX coverage is N/A, never dropped',
        loss_decomposition='Original F0 closed minus candidate closed + candidate closed minus open = original F0 closed minus open',
        primary_rule='M fixed; relation_max with source-only predicted-class thresholds; CK90/RU90/originalF0 loss2pp overall and all six target scenes; DA/REG not yet performed')
    output.mkdir(parents=True);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('source','target')})
    print(STATUS,m['arm'],flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_OPEN_WORLD_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True)
    p.add_argument('--target-truth',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    score(a.prediction_root,a.novel_truth,a.target_truth,a.output)


if __name__=='__main__':main()
