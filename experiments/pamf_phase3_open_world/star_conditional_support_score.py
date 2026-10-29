"""Truth-last source diagnostics for predicted-identity local support."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
from . import star_conditional_support_run as evaluation
from . import star_conditional_support_core as core
from . import star_local_support_run as parent
from . import star_local_support_score as previous
from . import star_joint_ova_score as common
from .metrics import open_set_metrics

RUN_ID='20261030-phase3-star-conditional-support-s2026092701-r01'
SCHEMA='pamf_phase3_star_conditional_support_scores_v1'
STATUS='STAR_CONDITIONAL_SUPPORT_SCORES_COMPLETE'
PARENT_RUN='20261030-phase3-star-local-support-s2026092701-r01'
PARENT_CODE='84cdc012f7d9641b283c407f6048bfe4e6e21cf6'
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
TEACHER_CODE='5aa041122396d48c49b8cb3d5912649bdc3217e1'
METHODS=('pooled_relative','predicted_identity_relative')
PRIMARY='predicted_identity_relative'
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
_read,_write,_truth=common._read,common._write,common._truth
_npz,_units=previous._npz,previous._units


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('schema')!=evaluation.SCHEMA or marker.get('status')!=evaluation.STATUS or marker.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(marker.get('code_commit'))) is None
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_method')!=PRIMARY
            or marker.get('parent_run_id')!=PARENT_RUN or marker.get('parent_code_commit')!=PARENT_CODE
            or marker.get('classification_path')!='P1' or marker.get('source_only') is not True
            or any(marker.get(k) is not True for k in ('parent_public_features_reused','control_scores_thresholds_acceptance_reproduced',
                'source_bank_frozen_before_V','source_bank_unchanged','model_parameters_buffers_unchanged','all_thresholds_frozen_before_novel'))
            or any(marker.get(k)!=v for k,v in dict(known_count=COUNTS['V'],novel_count=COUNTS['novel'],bank_L_count=COUNTS['L'],bank_OE_count=COUNTS['OE'],
                target_reads=0,source_novel_truth_reads=0,source_U_reads=0,F_forward_rows=2,synthetic_P0_forward_rows=2,G_forward_rows=0,
                new_V_forward_rows=0,new_novel_forward_rows=0,source_L_forward_rows=0,source_OE_forward_rows=0,neural_updates=0,G_updates=0,
                real_IQ_values_read=0,source_L_metadata_labels_read=COUNTS['L'],fresh_closed_vs_cached_baseline_changed_count=0).items())):
        raise ValueError('Complete frozen source predictions, original-head reuse and fixed input costs required before truth')
    folder=Path(marker['parent_prediction_root']);pm=_read(folder/'complete.json');identity=pm.get('model_identity',{})
    if (pm.get('schema')!=parent.SCHEMA or pm.get('status')!=parent.STATUS or pm.get('run_id')!=PARENT_RUN or pm.get('code_commit')!=PARENT_CODE
            or pm.get('method_names')!=list(parent.METHODS) or pm.get('primary_method')!=parent.PRIMARY_METHOD
            or pm.get('classification_path')!='P1' or pm.get('source_only') is not True
            or pm.get('default_head_replaced') is not False or pm.get('parameters')!=232993 or pm.get('score_threshold')!=0.
            or any(identity.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(pm.get(k)!=v for k,v in dict(known_count=COUNTS['V'],novel_count=COUNTS['novel'],bank_L_count=COUNTS['L'],bank_OE_count=COUNTS['OE'],
                target_reads=0,source_novel_truth_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,F_forward_rows=COUNTS['V']+COUNTS['novel']+2).items())
            or any(pm.get(k) is not True for k in ('all_thresholds_frozen_before_novel','source_bank_frozen_before_V','source_bank_unchanged','model_parameters_buffers_unchanged'))
            or any(marker.get(k)!=pm.get(k) for k in ('model_identity','dura_identity','cache_root','cache_identities','teacher_root','teacher_run_id','teacher_code_commit'))
            or pm.get('teacher_run_id')!=TEACHER_RUN or pm.get('teacher_code_commit')!=TEACHER_CODE):
        raise ValueError('Exact completed local-support parent, original F0 and source identities required')
    dura=pm.get('dura_identity',{})
    if any(dura.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()):
        raise ValueError('Fixed DURA original P1 ancestry required')
    bank=_npz(root/'source_bank.npz',evaluation.BANK_FIELDS);pb=_npz(folder/'source_bank.npz',parent.BANK_FIELDS)
    if any(not np.array_equal(bank[k],pb[k]) or bank[k].dtype!=pb[k].dtype for k in parent.BANK_FIELDS):
        raise ValueError('Original parent source-bank IDs, RX and FP32 unit features must remain identical')
    cacheL=Path(marker['cache_root'])/'L';ids=np.load(cacheL/'ids.npy',allow_pickle=False)
    with np.load(cacheL/'metadata.npz',allow_pickle=False) as z:
        if set(z.files)!={'y','rx_i','day_i','domain'}:raise ValueError('Only original public source-L labeled metadata allowed')
        y=z['y'].copy();rx=z['rx_i'].copy()
    if (not common._ids(ids,COUNTS['L']) or not np.array_equal(ids,bank['L_ids'])
            or _read(cacheL/'manifest.json')!=marker.get('cache_identities',{}).get('L')
            or y.shape!=(COUNTS['L'],) or y.dtype!=np.int64 or not np.isin(y,range(6)).all()
            or not np.array_equal(np.bincount(y,minlength=6),np.full(6,COUNTS['L']//6))
            or bank['L_y'].dtype!=np.int64 or not np.array_equal(bank['L_y'],y)
            or rx.dtype!=np.int64 or not np.array_equal(rx,bank['L_rx_i'])):
        raise ValueError('Source-L true labels must align exactly with original cache IDs and the immutable bank')
    seen=set()
    for role in ('L','OE'):
        n=COUNTS[role];sid=bank[role+'_ids'];features=bank[role+'_unit_features'];r=bank[role+'_rx_i']
        if (not common._ids(sid,n) or not _units(features,n) or r.shape!=(n,) or r.dtype!=np.int64 or not np.isin(r,RECEIVERS).all()
                or seen&set(sid.tolist())):raise ValueError('Finite original source unit features and disjoint bank roles required')
        seen.update(sid.tolist())
    tau=_npz(root/'thresholds.npz',{'conditional'})['conditional'];ptau=_npz(folder/'thresholds.npz',{'conditional'})['conditional']
    cal=_read(root/'calibration/complete.json');pc=_read(folder/'calibration/complete.json')
    if (tau.shape!=(2,6) or ptau.shape!=(3,6) or any(t.dtype!=np.float64 or not np.isfinite(t).all() for t in (tau,ptau))
            or not np.array_equal(tau[0],ptau[1])
            or cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit')) or cal.get('method_names')!=list(METHODS)
            or any(cal.get(k) is not True for k in ('primary_threshold_fit_known_only','control_thresholds_reused','source_only','source_L_TX_labels_used'))
            or cal.get('source_OE_TX_or_K_used') is not False or cal.get('target_reads')!=0 or cal.get('group_count')!=42
            or not np.array_equal(cal.get('conditional_thresholds'),tau) or set(cal.get('methods',{}))!=set(METHODS)
            or any(set(cal['methods'][name].get('groups',{}))!=set(GROUP_NAMES) for name in METHODS)
            or any(not np.array_equal(cal['methods'][name].get('conditional_thresholds'),tau[j]) for j,name in enumerate(METHODS))
            or pc.get('schema')!=parent.SCHEMA or pc.get('status')!=parent.CALIBRATION_STATUS
            or pc.get('run_id')!=PARENT_RUN or pc.get('code_commit')!=PARENT_CODE
            or pc.get('method_names')!=list(parent.METHODS) or pc.get('group_count')!=42 or pc.get('source_only') is not True or pc.get('target_reads')!=0
            or not np.array_equal(pc.get('conditional_thresholds'),ptau)
            or cal['methods'][METHODS[0]]!=pc.get('methods',{}).get(METHODS[0])):
        raise ValueError('Twelve frozen class thresholds, known42 calibration and exact parent control report required')
    # Rebuild only the fixed per-vector transform, never fit a query statistic or threshold.
    state=dict(L=bank['L_unit_features'].astype(np.float64),OE=bank['OE_unit_features'].astype(np.float64),L_y=y.copy())
    for role in ('L','OE'):state[role]/=np.sqrt(np.einsum('nd,nd->n',state[role],state[role],optimize=False))[:,None]
    rows=[]
    for role,fields,count in (('known',evaluation.KNOWN_FIELDS,COUNTS['V']),('novel',evaluation.NOVEL_FIELDS,COUNTS['novel'])):
        data=_npz(root/(role+'_predictions.npz'),fields);old=_npz(folder/(role+'_predictions.npz'),fields)
        if (any(not np.array_equal(data[k],old[k]) or data[k].dtype!=old[k].dtype for k in fields-{'raw_scores','scores'})
                or not common._ids(data['ids'],count) or seen&set(data['ids'].tolist())
                or not _units(data['unit_features'],count) or data['default_logits'].shape!=(count,6) or data['default_logits'].dtype!=np.float32
                or not np.isfinite(data['default_logits']).all() or data['closed'].shape!=(count,) or data['closed'].dtype!=np.int64
                or not np.array_equal(data['closed'],data['default_logits'].argmax(1))
                or any(data[k].shape!=(count,) or data[k].dtype!=np.int64 or not np.isin(data[k],r).all() for k,r in (('rx_i',RECEIVERS),('day_i',(1,2,3))))
                or any(data[k].shape!=(count,2) or data[k].dtype!=np.float64 or not np.isfinite(data[k]).all() for k in ('raw_scores','scores'))
                or any(old[k].shape!=(count,3) or old[k].dtype!=np.float64 or not np.isfinite(old[k]).all() for k in ('raw_scores','scores'))):
            raise ValueError('Exact parent identity/features/logits/meta and original default-head argmax required')
        if role=='known' and (data['y'].shape!=(count,) or data['y'].dtype!=np.int64 or not np.isin(data['y'],range(6)).all()
                or data['scenario'].shape!=(count,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                or data['baseline_closed'].shape!=(count,) or data['baseline_closed'].dtype!=np.int64 or not np.isin(data['baseline_closed'],range(6)).all()):
            raise ValueError('Unchanged source-V labels/scenes and original cached baseline required')
        seen.update(data['ids'].tolist());raw=core.score(data['unit_features'],data['closed'],state)
        if (not np.allclose(data['raw_scores'],raw,rtol=0,atol=1e-12) or not np.allclose(data['scores'],raw-tau[:,data['closed']].T,rtol=0,atol=1e-12)
                or not np.allclose(data['raw_scores'][:,0],old['raw_scores'][:,1],rtol=0,atol=1e-12)
                or not np.allclose(data['scores'][:,0],old['scores'][:,1],rtol=0,atol=1e-12)
                or not np.array_equal(data['scores'][:,0]>=0,old['scores'][:,1]>=0)):
            raise ValueError('Two fixed-bank raw scores and margins, original control scores/acceptance must reproduce')
        rows.append(data)
    return marker,cal,tau,*rows


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent source-score root is required')
    started=time.perf_counter();marker,cal,tau,known,novel=_prediction(prediction_root);verification=time.perf_counter()-started
    truth=_truth(novel_truth,novel)
    km=dict(common._masks(known['rx_i'],known['scenario']));um=dict(common._masks(novel['rx_i'],truth['scenario']));methods={}
    for j,name in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            row=open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],truth['tx'][umask],novel['scores'][umask,j],
                0.,known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
            closed=[float(np.mean(known['closed'][kmask&(known['y']==c)]==c)) if np.any(kmask&(known['y']==c)) else None for c in known_roster]
            row['fresh_F0_macro_closed_accuracy']=None if any(v is None for v in closed) else float(np.mean(closed));return row
        groups={key:calculate(mask,um[key]) for key,mask in km.items()};overall=groups['overall']
        methods[name]=dict(overall=overall,groups=groups,conditional_thresholds=tau[j].tolist(),
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='per true-class Wilson; descriptive source pool only',
                A_K_per_class=common._ci(overall['known_per_class'],'correct_accepted'),R_U_per_class=common._ci(overall['unknown_per_class'],'rejected')),
            goals=common._goals(groups,cal['methods'][name].get('constraints_passed') is True),diagnostic_only=name!=PRIMARY)
    correct=known['closed']==known['y'];ka=known['scores']>=0;ur=novel['scores']<0;main=methods[PRIMARY];control=methods[METHODS[0]];passed=bool(main['goals']['passed'])
    paired=dict(known_correct_gained=int(np.sum(correct&ka[:,1]&~ka[:,0])),known_correct_lost=int(np.sum(correct&~ka[:,1]&ka[:,0])),
        known_acceptance_gained=int(np.sum(ka[:,1]&~ka[:,0])),known_acceptance_lost=int(np.sum(~ka[:,1]&ka[:,0])),
        unknown_rejection_gained=int(np.sum(ur[:,1]&~ur[:,0])),unknown_rejection_lost=int(np.sum(~ur[:,1]&ur[:,0])),
        delta={key:None if main['overall'][key] is None or control['overall'][key] is None else float(main['overall'][key]-control['overall'][key])
            for key in ('known_macro_open_accuracy','known_macro_acceptance','unknown_macro_rejection','H_OS')})
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],row='original_F0',method_names=list(METHODS),methods=methods,
        primary_method=PRIMARY,primary_path='P1',primary_summary=main['overall'],primary_goals=main['goals'],source_gate_passed=passed,arm_source_gate_passed=passed,
        eligible_primary=False,target_eligible=False,source_free_export_required=True,source_bank_is_deployment_state=False,
        next_step='SOURCE_CONDITIONAL_SUPPORT_READY_FOR_SOURCE_FREE_EXPORT' if passed else 'NO_PROMOTION',paired_primary_minus_pooled_relative=paired,
        control_scores_thresholds_acceptance_reproduced=True,parent_run_id=PARENT_RUN,parent_code_commit=PARENT_CODE,
        model_identity=marker['model_identity'],teacher_run_id=TEACHER_RUN,teacher_code_commit=TEACHER_CODE,
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,
        formal_target_accessed=False,target_generalization_confirmed=False,source_RX_is_seen=True,source_novel_historically_exposed=True,known_protection_groups=42,
        source_L_TX_labels_used=True,source_V_labels_used_for_thresholds=True,source_OE_membership_used_for_bank=True,source_OE_TX_or_K_used=False,
        baseline='Unchanged parent original F0 cached P1 closed; never reset from fresh classification',
        primary_rule='Only predicted_identity_relative can pass this source diagnostic; full bank is not a source-free deployment state',
        fresh_closed_vs_cached_baseline_changed_count=int(np.sum(known['closed']!=known['baseline_closed'])),
        input_costs={k:marker[k] for k in ('F_forward_rows','G_forward_rows','neural_updates','G_updates','real_IQ_values_read','source_L_metadata_labels_read')},
        ancestor_costs=dict(parent_F_forward_rows=30152,historical_P1_restoration_reused=True,reuse_is_not_zero_cost=True),
        CPU_readout=dict(production_seconds=marker.get('cpu_readout_seconds'),production_rows=marker.get('cpu_readout_rows'),
            public_evidence_verification_seconds=verification,verification_is_not_deployment_time=True,not_end_to_end_B1_latency=True),
        storage=dict(neural_parameters_F=232993,neural_parameters_G=266951,neural_parameters_total=499944,aux_parameters=0,
            bank_FP32_feature_bytes=(COUNTS['L']+COUNTS['OE'])*160*4,working_FP64_feature_bytes=(COUNTS['L']+COUNTS['OE'])*160*8,
            L_y_bytes=COUNTS['L']*8,excludes_RX_IDs_indices_distance_workspace=True))
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result);_write(output/'complete.json',{k:v for k,v in result.items() if k!='methods'})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_CONDITIONAL_SUPPORT_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
