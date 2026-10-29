"""Independent truth-last metrics for frozen original-F0 local support."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np

from . import star_local_support_run as evaluation
from . import star_local_support_core as core
from . import star_joint_ova_score as common
from .metrics import open_set_metrics

RUN_ID='20261030-phase3-star-local-support-s2026092701-r01'
SCHEMA='pamf_phase3_star_local_support_scores_v1'
STATUS='STAR_LOCAL_SUPPORT_SCORES_COMPLETE'
METHODS=('pooled_known','pooled_relative','rx_consensus_relative')
PRIMARY='rx_consensus_relative'
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
TEACHER_CODE='5aa041122396d48c49b8cb3d5912649bdc3217e1'
TEACHER_SCHEMA='pamf_phase3_star_joint_ova_teacher_v1'
TEACHER_STATUS='STAR_JOINT_OVA_TEACHER_FROZEN'
CACHE_RUN='20261029-phase3-star-representation-s2026092701-r01'
CACHE_SCHEMA='pamf_phase3_star_representation_cache_role_v1'
CACHE_STATUS='STAR_REPRESENTATION_ROLE_COMPLETE'
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
_read,_write,_truth=common._read,common._write,common._truth


def _npz(path,fields):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(fields):raise ValueError('Exact public artifact fields required; private novel fields forbidden')
        return {key:z[key].copy() for key in z.files}


def _units(value,n):
    return (value.shape==(n,160) and value.dtype==np.float32 and np.isfinite(value).all()
        and np.allclose(np.linalg.norm(value.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6))


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json');identity=marker.get('model_identity',{})
    expected=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
    if (marker.get('run_id')!=RUN_ID or marker.get('schema')!=evaluation.SCHEMA or marker.get('status')!=evaluation.STATUS
            or re.fullmatch('[0-9a-f]{40}',str(marker.get('code_commit'))) is None
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_method')!=PRIMARY
            or marker.get('classification_path')!='P1' or marker.get('source_only') is not True
            or marker.get('score_threshold')!=0. or marker.get('parameters')!=232993 or marker.get('default_head_replaced') is not False
            or marker.get('source_bank_uses_OE_membership') is not True or marker.get('source_bank_uses_TX_labels') is not False
            or any(identity.get(k)!=v for k,v in expected.items())
            or marker.get('teacher_run_id')!=TEACHER_RUN or marker.get('teacher_code_commit')!=TEACHER_CODE
            or any(marker.get(k) is not True for k in ('all_thresholds_frozen_before_novel','source_bank_frozen_before_V',
                'source_bank_unchanged','model_parameters_buffers_unchanged'))
            or any(marker.get(k)!=v for k,v in dict(known_count=COUNTS['V'],novel_count=COUNTS['novel'],bank_L_count=COUNTS['L'],bank_OE_count=COUNTS['OE'],
                target_reads=0,source_novel_truth_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,
                F_forward_rows=COUNTS['V']+COUNTS['novel']+2,new_V_forward_rows=COUNTS['V'],new_novel_forward_rows=COUNTS['novel'],
                synthetic_P0_forward_rows=2,source_L_forward_rows=0,source_OE_forward_rows=0,source_U_reads=0,received_P0_values_read=0).items())):
        raise ValueError('Complete original-F0 source-only predictions and fixed input costs required before truth')
    dura=marker.get('dura_identity',{})
    if any(dura.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()):
        raise ValueError('Fixed DURA identity required for cached P1 inputs')
    teacher_root=Path(marker['teacher_root']);teacher=_read(teacher_root/'complete.json')
    if (teacher.get('schema')!=TEACHER_SCHEMA or teacher.get('status')!=TEACHER_STATUS
            or teacher.get('run_id')!=TEACHER_RUN or teacher.get('code_commit')!=TEACHER_CODE
            or teacher.get('parent_F0_identity')!=identity or teacher.get('dura_identity')!=dura
            or teacher.get('source_only') is not True or teacher.get('source_L_count')!=COUNTS['L'] or teacher.get('source_OE_count')!=COUNTS['OE']
            or teacher.get('input_path')!='cached_P1_only' or teacher.get('feature')!='unit_fused160'
            or teacher.get('model_parameters')!=232993 or teacher.get('model_buffers_unchanged') is not True
            or any(teacher.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','OE_truth_reads',
                'encoder_updates','identity_head_updates','G_updates','G_forward_rows'))):
        raise ValueError('Exact completed original-F0 P1 teacher required; updated r02 encoder is forbidden')
    bank=_npz(root/'source_bank.npz',evaluation.BANK_FIELDS)
    cache=Path(marker['cache_root']);handles=marker.get('cache_identities',{})
    if set(handles)!=set(COUNTS):raise ValueError('All four fixed public source-role identities required')
    roles={};seen=set()
    for role,n in COUNTS.items():
        folder=cache/role;manifest=_read(folder/'manifest.json');ids=np.load(folder/'ids.npy',allow_pickle=False)
        with np.load(folder/'metadata.npz',allow_pickle=False) as z:rx=z['rx_i'].copy()
        if (manifest!=handles[role] or manifest.get('schema')!=CACHE_SCHEMA or manifest.get('status')!=CACHE_STATUS
                or manifest.get('run_id')!=CACHE_RUN or manifest.get('role')!=role or manifest.get('count')!=n
                or manifest.get('source_only') is not True or manifest.get('all_new_IQ_satellite') is not True
                or manifest.get('received_input_path')!='P0' or manifest.get('restored_input_path')!='P1'
                or manifest.get('parent_F0_identity')!=identity or manifest.get('dura_identity')!=dura
                or not common._ids(ids,n) or rx.shape!=(n,) or rx.dtype!=np.int64 or not np.isin(rx,RECEIVERS).all()):
            raise ValueError('Fixed public source-role IDs, RX metadata and original F0/DURA identities must match')
        current=set(ids.tolist())
        if current&seen:raise ValueError('L/OE bank, V and novel physical IDs must all be disjoint')
        seen.update(current);roles[role]=dict(ids=ids,rx=rx)
        if role in ('L','OE'):
            original=_npz(teacher_root/(role+'.npz'),{'ids','unit_features'})
            if (teacher.get('cache_handles',{}).get(role)!=manifest
                    or not np.array_equal(original['ids'],ids) or not np.array_equal(bank[role+'_ids'],ids)
                    or not _units(original['unit_features'],n) or not _units(bank[role+'_unit_features'],n)
                    or not np.array_equal(bank[role+'_unit_features'],original['unit_features'])
                    or bank[role+'_rx_i'].dtype!=np.int64 or not np.array_equal(bank[role+'_rx_i'],rx)
                    or any(np.sum(rx==r)<8 for r in RECEIVERS)):
                raise ValueError('Exact original teacher FP32 features, source role IDs and five RX bank groups required')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Exactly eighteen fixed class thresholds required')
        tau=z['conditional'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit')) or cal.get('method_names')!=list(METHODS)
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal.get('source_only') is not True or cal.get('target_reads')!=0 or cal.get('group_count')!=42
            or not np.array_equal(cal.get('conditional_thresholds'),tau) or set(cal.get('methods',{}))!=set(METHODS)
            or any(set(cal['methods'][name].get('groups',{}))!=set(GROUP_NAMES) for name in METHODS)
            or any(not np.array_equal(cal['methods'][name].get('conditional_thresholds'),tau[j]) for j,name in enumerate(METHODS))):
        raise ValueError('Matching known-V-only frozen calibration, all 42 groups and eighteen thresholds required')
    # Only the specified per-vector transform; no fitted/query statistics or fit call.
    state={role:bank[role+'_unit_features'].astype(np.float64) for role in ('L','OE')}
    for role in ('L','OE'):
        state[role]/=np.sqrt(np.einsum('nd,nd->n',state[role],state[role],optimize=False))[:,None]
    state.update(L_rx=bank['L_rx_i'],OE_rx=bank['OE_rx_i']);rows=[]
    for role,cache_role,fields in (('known','V',evaluation.KNOWN_FIELDS),('novel','novel',evaluation.NOVEL_FIELDS)):
        n=COUNTS[cache_role];data=_npz(root/(role+'_predictions.npz'),fields)
        if (not np.array_equal(data['ids'],roles[cache_role]['ids']) or not np.array_equal(data['rx_i'],roles[cache_role]['rx'])
                or not _units(data['unit_features'],n) or data['default_logits'].shape!=(n,6) or data['default_logits'].dtype!=np.float32
                or not np.isfinite(data['default_logits']).all() or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64
                or not np.array_equal(data['closed'],data['default_logits'].argmax(1))
                or data['day_i'].shape!=(n,) or data['day_i'].dtype!=np.int64 or not np.isin(data['day_i'],(1,2,3)).all()
                or any(data[key].shape!=(n,3) or data[key].dtype!=np.float64 or not np.isfinite(data[key]).all() for key in ('raw_scores','scores'))):
            raise ValueError('Original default-head argmax, finite FP32 units/logits and FP64 scores required')
        if role=='known':
            baseline=np.load(cache/'V/baseline_closed.npy',allow_pickle=False)
            if (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or not np.isin(data['y'],range(6)).all()
                    or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                    or baseline.shape!=(n,2) or baseline.dtype!=np.int64 or not np.isin(baseline,range(6)).all()
                    or data['baseline_closed'].dtype!=np.int64 or not np.array_equal(data['baseline_closed'],baseline[:,1])):
                raise ValueError('Known V labels/scenes and original cached P1 closed baseline required; fresh baseline substitution forbidden')
        raw=core.score(data['unit_features'],state)
        if (not np.allclose(data['raw_scores'],raw,rtol=0,atol=1e-12)
                or not np.allclose(data['scores'],raw-tau[:,data['closed']].T,rtol=0,atol=1e-12)):
            raise ValueError('All local support raw scores and predicted-class margins must reconstruct from the fixed teacher bank')
        rows.append(data)
    return marker,cal,tau,*rows


def _paired(known,novel,methods):
    correct=known['closed']==known['y'];ka=known['scores']>=0;ur=novel['scores']<0;result={}
    for j,name in enumerate(METHODS[:-1]):
        left,right=methods[PRIMARY]['overall'],methods[name]['overall']
        result[name]=dict(known_correct_gained=int(np.sum(correct&ka[:,2]&~ka[:,j])),known_correct_lost=int(np.sum(correct&~ka[:,2]&ka[:,j])),
            known_acceptance_gained=int(np.sum(ka[:,2]&~ka[:,j])),known_acceptance_lost=int(np.sum(~ka[:,2]&ka[:,j])),
            unknown_rejection_gained=int(np.sum(ur[:,2]&~ur[:,j])),unknown_rejection_lost=int(np.sum(~ur[:,2]&ur[:,j])),
            delta={key:None if left[key] is None or right[key] is None else float(left[key]-right[key])
                for key in ('known_macro_open_accuracy','known_macro_acceptance','unknown_macro_rejection','H_OS')})
    return result


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent score root is required')
    started=time.perf_counter();marker,cal,tau,known,novel=_prediction(prediction_root)
    verification_seconds=time.perf_counter()-started
    truth=_truth(novel_truth,novel)  # Sole private read, after all public identity and score checks.
    km=dict(common._masks(known['rx_i'],known['scenario']));um=dict(common._masks(novel['rx_i'],truth['scenario']));methods={}
    for j,name in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            row=open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],truth['tx'][umask],
                novel['scores'][umask,j],0.,known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
            closed=[float(np.mean(known['closed'][kmask&(known['y']==c)]==c)) if np.any(kmask&(known['y']==c)) else None for c in known_roster]
            fresh=None if any(v is None for v in closed) else float(np.mean(closed))
            row.update(fresh_F0_macro_closed_accuracy=fresh,fresh_closed_minus_open=None if fresh is None or row['known_macro_open_accuracy'] is None else fresh-row['known_macro_open_accuracy'])
            return row
        groups={key:calculate(mask,um[key]) for key,mask in km.items()};overall=groups['overall']
        methods[name]=dict(overall=overall,groups=groups,threshold=0.,conditional_thresholds=tau[j].tolist(),
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='per true-class Wilson; descriptive source pool only',
                A_K_per_class=common._ci(overall['known_per_class'],'correct_accepted'),R_U_per_class=common._ci(overall['unknown_per_class'],'rejected')),
            goals=common._goals(groups,cal['methods'][name].get('constraints_passed') is True),diagnostic_only=name!=PRIMARY)
    passed=bool(methods[PRIMARY]['goals']['passed'])
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],row='original_F0',
        method_names=list(METHODS),methods=methods,primary_method=PRIMARY,primary_path='P1',primary_summary=methods[PRIMARY]['overall'],
        primary_goals=methods[PRIMARY]['goals'],source_gate_passed=passed,arm_source_gate_passed=passed,eligible_primary=passed,
        next_step='SOURCE_LOCAL_SUPPORT_READY_FOR_TARGET_FALSIFICATION' if passed else 'NO_PROMOTION',
        paired_primary_minus=_paired(known,novel,methods),fresh_closed_vs_cached_baseline_changed_count=int(np.sum(known['closed']!=known['baseline_closed'])),
        feature_structure=dict(preserved_by_construction=True,encoder_updates=0,new_structure_gate=False,not_anonymous_registration_accuracy=True),
        model_identity=marker['model_identity'],teacher_run_id=TEACHER_RUN,teacher_code_commit=TEACHER_CODE,
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,A_N=None,H_OW=None,
        full_goal_status='ACTIVE',final_phase3_complete=False,formal_target_accessed=False,target_generalization_confirmed=False,
        source_RX_is_seen=True,source_novel_historically_exposed=True,known_protection_groups=42,
        source_known_labels_used_for_thresholds=True,source_OE_membership_used_for_bank=True,source_OE_TX_or_K_used=False,
        threshold_fit_known_only=True,method_is_entirely_known_only=False,
        baseline='cached original F0 same-input P1 closed; fresh closed reported separately',
        primary_rule='Only fixed rx_consensus_relative with known42 and unknown overall/5RX/6scene protection can prepare separate target falsification',
        input_costs={key:marker[key] for key in ('F_forward_rows','G_forward_rows','neural_updates','G_updates','bank_L_count','bank_OE_count')},
        CPU_readout=dict(production_seconds=marker.get('cpu_readout_seconds'),production_rows=marker.get('cpu_readout_rows'),
            public_evidence_verification_seconds=verification_seconds,production_scope='V + novel blocking core.score calls only; excludes F and calibration',
            verification_is_not_deployment_time=True,not_end_to_end_B1_latency=True),
        storage=dict(neural_parameters_F=232993,neural_parameters_G=266951,neural_parameters_total=499944,aux_parameters=0,
            bank_FP32_feature_bytes=(COUNTS['L']+COUNTS['OE'])*160*4,working_FP64_feature_bytes=(COUNTS['L']+COUNTS['OE'])*160*8,
            excludes_RX_indices_and_distance_workspace=True,CPU_readout_is_not_end_to_end_B1_latency=True))
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{key:value for key,value in result.items() if key!='methods'})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_LOCAL_SUPPORT_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
