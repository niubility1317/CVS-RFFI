"""Truth-last source cosine scores with an unchanged external MD reference."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from . import star_metric_cosine as evaluation
from . import star_metric_embedding_run as parent
from . import star_metric_embedding_score as embedding
from . import star_joint_ova_score as common
from .metrics import open_set_metrics

RUN_ID='20261030-phase3-star-metric-cosine-s2026092701-r01'
SCHEMA='pamf_phase3_star_metric_cosine_scores_v1'
STATUS='STAR_METRIC_COSINE_SCORES_COMPLETE'
PARENT_RUN='20261030-phase3-star-metric-embedding-s2026092701-r01'
PARENT_CODE='efb82a4177e9b59533b7e71408ded6b63fc8733a'
PARENT_ARM='compact_oe_structure'
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
TEACHER_CODE='5aa041122396d48c49b8cb3d5912649bdc3217e1'
METHODS=('predicted_cosine','maximum_cosine');PRIMARY='maximum_cosine'
COUNTS=dict(V=27000,novel=3150)
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
_read,_write,_truth=common._read,common._write,common._truth
_npz,_units=embedding._npz,embedding._units


def _parent(marker):
    root=Path(marker['parent_prediction_root']);pm=_read(root/'complete.json');identity=pm.get('model_identity',{})
    if (pm.get('schema')!=parent.SCHEMA or pm.get('status')!=parent.STATUS or pm.get('run_id')!=PARENT_RUN or pm.get('code_commit')!=PARENT_CODE
            or pm.get('arm')!=PARENT_ARM or pm.get('g_parameters')!=20704 or pm.get('g_training_steps')!=1000
            or any(identity.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or pm.get('classification_path')!='P1' or pm.get('default_head_replaced') is not False or pm.get('source_only') is not True
            or any(pm.get(k)!=v for k,v in dict(known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=6300,source_OE_count=6134,
                g_training_L_exposures=120000,g_training_OE_exposures=128000,neural_F_updates=0,G_updates=0,source_U_reads=0,target_reads=0,
                source_novel_truth_reads=0,per_record_source_rows_retained=0,fresh_closed_vs_cached_baseline_changed_count=0).items())
            or any(pm.get(k) is not True for k in ('source_free_deployment_state','model_parameters_buffers_unchanged',
                'metric_parameters_buffers_frozen_for_V_novel','aggregate_frozen_before_V','all_thresholds_frozen_before_novel'))
            or any(marker.get(k)!=pm.get(k) for k in ('model_identity','dura_identity','cache_identities','cache_root','teacher_root','teacher_run_id','teacher_code_commit'))
            or pm.get('teacher_run_id')!=TEACHER_RUN or pm.get('teacher_code_commit')!=TEACHER_CODE):
        raise ValueError('Only the fixed completed structured main parent, original F0 and source-free g ancestry are allowed')
    dura=pm.get('dura_identity',{});handles=pm.get('cache_identities',{})
    if (any(dura.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items())
            or set(handles)!={'L','OE','V','novel'} or any(handles[r].get('role')!=r or handles[r].get('count')!=n
                or handles[r].get('parent_F0_identity')!=identity or handles[r].get('dura_identity')!=dura
                for r,n in dict(L=6300,OE=6134,**COUNTS).items())):
        raise ValueError('Matching frozen original-F0/DURA public source handles required')
    # This helper reads only four public deployment files, never source rows,
    # training traces, parent predictions or truth-derived parent score files.
    net,metric,tau=embedding._deployment(root,pm)
    cal=_read(root/'calibration/complete.json');name='predicted_identity_mahalanobis'
    if (cal.get('schema')!=parent.SCHEMA or cal.get('status')!=parent.CALIBRATION_STATUS
            or any(cal.get(k)!=pm[k] for k in ('run_id','code_commit','arm')) or cal.get('method_names')!=list(parent.METHODS)
            or cal.get('group_count')!=42 or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal.get('source_V_neural_fit') is not False or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or not np.array_equal(cal.get('conditional_thresholds'),tau)
            or not np.array_equal(cal.get('methods',{}).get(name,{}).get('conditional_thresholds'),tau[1])
            or set(cal.get('methods',{}).get(name,{}).get('groups',{}))!=set(GROUP_NAMES)):
        raise ValueError('Original frozen parent MD thresholds and known-only calibration required for saved-margin reference')
    return root,pm,net,metric,tau,cal


def _prediction(root):
    root=Path(root);marker=_read(root/'complete.json')
    if (marker.get('schema')!=evaluation.SCHEMA or marker.get('status')!=evaluation.STATUS or marker.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(marker.get('code_commit'))) is None
            or marker.get('parent_run_id')!=PARENT_RUN or marker.get('parent_code_commit')!=PARENT_CODE or marker.get('parent_arm')!=PARENT_ARM
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_method')!=PRIMARY
            or marker.get('source_only') is not True or marker.get('classification_path')!='P1' or marker.get('default_head_replaced') is not False
            or any(marker.get(k) is not True for k in ('prototypes_frozen_before_V','all_thresholds_frozen_before_novel',
                'source_free_deployment_state','parent_public_features_reused','deployed_g_unchanged','fresh_closed_is_parent_reuse'))
            or any(marker.get(k)!=v for k,v in dict(known_count=COUNTS['V'],novel_count=COUNTS['novel'],g_parameters=20704,inherited_g_training_steps=1000,
                new_neural_updates=0,F_forward_rows=0,G_forward_rows=0,g_forward_rows=2,synthetic_unit_smoke_rows=2,real_IQ_values_read=0,
                source_L_forward_rows=0,source_OE_forward_rows=0,source_L_metadata_labels_read=0,target_reads=0,source_U_reads=0,source_novel_truth_reads=0,
                per_record_source_rows_retained=0,cpu_readout_rows=COUNTS['V']+COUNTS['novel']).items())
            or not embedding._number(marker.get('cpu_readout_seconds')) or marker['cpu_readout_seconds']<0):
        raise ValueError('Complete frozen cosine source predictions and zero new source/real-IQ/target access required before truth')
    parent_root,pm,parent_net,metric,ptau,pc=_parent(marker);path=root/'deployment'
    if {p.name for p in path.iterdir()}!={'model.pt','prototypes.npz','thresholds.npz','complete.json'}:
        raise ValueError('Exactly four source-free cosine deployment files required; no source rows or covariance export')
    expected=dict(schema=evaluation.DEPLOYMENT_SCHEMA,status=evaluation.DEPLOYMENT_STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],
        parameters=20704,source_free=True,per_record_source_rows_retained=0)
    if _read(path/'complete.json')!=expected:raise ValueError('Exact source-free cosine deployment identity required')
    state=torch.load(path/'model.pt',map_location='cpu',weights_only=True);old_state=parent_net.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(embedding.MODEL_SHAPES)
            or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or tuple(v.shape)!=embedding.MODEL_SHAPES[k]
                or not bool(torch.isfinite(v).all()) or not torch.equal(v.double(),old_state[k]) for k,v in state.items())):
        raise ValueError('The exact four finite FP32 tensors of the fixed structured parent must be copied unchanged')
    net=embedding.core.MetricNet().cpu().float();net.load_state_dict(state,strict=True);net.double().eval()
    prototypes=_npz(path/'prototypes.npz',{'prototypes'})['prototypes'];expected_prototypes=evaluation.prototypes(metric['class_means'])
    if (prototypes.shape!=(6,160) or prototypes.dtype!=np.float64 or not np.isfinite(prototypes).all()
            or not np.allclose(prototypes,expected_prototypes,rtol=0,atol=1e-12)):
        raise ValueError('Prototypes must be the FP64 row-normalized frozen parent source-L class means')
    tau=_npz(path/'thresholds.npz',{'conditional'})['conditional'];cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(2,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit')) or cal.get('method_names')!=list(METHODS) or cal.get('primary_method')!=PRIMARY
            or cal.get('group_count')!=42 or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal.get('source_V_neural_fit') is not False or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or not np.array_equal(cal.get('conditional_thresholds'),tau) or set(cal.get('methods',{}))!=set(METHODS)
            or any(set(cal['methods'][name].get('groups',{}))!=set(GROUP_NAMES) or not np.array_equal(cal['methods'][name].get('conditional_thresholds'),tau[j]) for j,name in enumerate(METHODS))):
        raise ValueError('Twelve matching frozen cosine thresholds and all known42 calibration groups required')
    smoke=_read(root/'smoke.json')
    if (smoke.get('status')!='PASS' or smoke.get('actual_deployment_checkpoint') is not True or smoke.get('synthetic_unit_rows')!=2
            or smoke.get('prediction_run_id')!=PARENT_RUN or smoke.get('prediction_code_commit')!=PARENT_CODE or smoke.get('arm')!=PARENT_ARM
            or smoke.get('verification_precision')!=embedding.VERIFICATION_PRECISION or smoke.get('g_reference_forward_rows')!=2
            or any(smoke.get(k)!=0 for k in ('source_known_rows_read','source_novel_rows_read','private_truth_reads','neural_updates','F_forward_rows','G_forward_rows'))):
        raise ValueError('Producer actual-checkpoint synthetic-unit smoke must pass without source query or truth access')
    seen=set();rows=[];references=[];diff={}
    for role,fields,n in (('known',evaluation.KNOWN_FIELDS,COUNTS['V']),('novel',evaluation.NOVEL_FIELDS,COUNTS['novel'])):
        data=_npz(root/(role+'_predictions.npz'),fields);old=_npz(parent_root/(role+'_predictions.npz'),fields)
        if (any(not np.array_equal(data[k],old[k]) or data[k].dtype!=old[k].dtype for k in fields-{'raw_scores','scores'})
                or not common._ids(data['ids'],n) or seen&set(data['ids'].tolist()) or not _units(data['unit_features'],n) or not _units(data['metric_unit_features'],n)
                or data['default_logits'].shape!=(n,6) or data['default_logits'].dtype!=np.float32 or not np.isfinite(data['default_logits']).all()
                or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64 or not np.array_equal(data['closed'],data['default_logits'].argmax(1))
                or any(data[k].shape!=(n,) or data[k].dtype!=np.int64 or not np.isin(data[k],r).all() for k,r in (('rx_i',RECEIVERS),('day_i',(1,2,3))))
                or any(data[k].shape!=(n,2) or data[k].dtype!=np.float64 or not np.isfinite(data[k]).all() for k in ('raw_scores','scores'))
                or any(old[k].shape!=(n,2) or old[k].dtype!=np.float64 or not np.isfinite(old[k][:,1]).all() for k in ('raw_scores','scores'))):
            raise ValueError('Exact parent non-score arrays, original F0 identity and finite source prediction fields required')
        if role=='known' and (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or not np.isin(data['y'],range(6)).all()
                or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                or data['baseline_closed'].dtype!=np.int64 or not np.array_equal(data['baseline_closed'],data['closed'])):
            raise ValueError('Original known labels/scenes and unchanged cached P1 baseline required')
        rebuilt=[]
        with torch.no_grad():
            for start in range(0,n,128):
                rebuilt.append(np.asarray(net(torch.tensor(data['unit_features'][start:start+128].tolist(),dtype=torch.float64)).tolist(),np.float32))
        units=np.concatenate(rebuilt);diff[role]=float(np.abs(units.astype(np.float64)-data['metric_unit_features']).max())
        if not np.allclose(units,data['metric_unit_features'],rtol=1e-5,atol=2e-6):raise ValueError('Actual copied g CPU FP64 reference must reconstruct saved parent metric units')
        raw=evaluation.cosine_scores(data['metric_unit_features'],data['closed'],prototypes)
        if (not np.allclose(raw,data['raw_scores'],rtol=0,atol=1e-12)
                or not np.allclose(raw-tau[:,data['closed']].T,data['scores'],rtol=0,atol=1e-12)):
            raise ValueError('Both FP64 cosine scores and predicted-class threshold margins must reconstruct')
        if not np.allclose(old['raw_scores'][:,1]-ptau[1,data['closed']],old['scores'][:,1],rtol=0,atol=1e-12):
            raise ValueError('External MD reference must reuse the original saved raw/margin and original six thresholds')
        seen.update(data['ids'].tolist());rows.append(data);references.append(old['scores'][:,1].copy())
    return marker,cal,tau,*rows,dict(known_margin=references[0],novel_margin=references[1],thresholds=ptau[1],
        calibrated=pc['methods']['predicted_identity_mahalanobis'].get('constraints_passed') is True,unit_max_diff=diff)


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent cosine score root is required')
    started=time.perf_counter();marker,cal,tau,known,novel,reference=_prediction(prediction_root);seconds=time.perf_counter()-started
    truth=_truth(novel_truth,novel)
    km=dict(common._masks(known['rx_i'],known['scenario']));um=dict(common._masks(novel['rx_i'],truth['scenario']))
    def method(known_margin,novel_margin,calibrated):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            return open_set_metrics(known['y'][kmask],known['closed'][kmask],known_margin[kmask],truth['tx'][umask],novel_margin[umask],0.,
                known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
        groups={name:calculate(mask,um[name]) for name,mask in km.items()};overall=groups['overall']
        return dict(overall=overall,groups=groups,
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='per true-class Wilson; descriptive source pool only',
                A_K_per_class=common._ci(overall['known_per_class'],'correct_accepted'),R_U_per_class=common._ci(overall['unknown_per_class'],'rejected')),
            goals=common._goals(groups,calibrated))
    methods={name:dict(method(known['scores'][:,j],novel['scores'][:,j],cal['methods'][name].get('constraints_passed') is True),
        conditional_thresholds=tau[j].tolist(),diagnostic_only=name!=PRIMARY) for j,name in enumerate(METHODS)}
    old=method(reference['known_margin'],reference['novel_margin'],reference['calibrated'])
    old.update(conditional_thresholds=reference['thresholds'].tolist(),saved_parent_margin_reused=True,thresholds_refitted=False,
        deployable_from_new_four_files=False,external_matched_reference=True,diagnostic_only=True,parent_method='predicted_identity_mahalanobis')
    main=methods[PRIMARY];passed=bool(main['goals']['passed']);ka=known['scores'][:,1]>=0;kr=reference['known_margin']>=0
    ur=novel['scores'][:,1]<0;rr=reference['novel_margin']<0;correct=known['closed']==known['y']
    paired=dict(known_correct_gained=int(np.sum(correct&ka&~kr)),known_correct_lost=int(np.sum(correct&~ka&kr)),
        known_acceptance_gained=int(np.sum(ka&~kr)),known_acceptance_lost=int(np.sum(~ka&kr)),unknown_rejection_gained=int(np.sum(ur&~rr)),unknown_rejection_lost=int(np.sum(~ur&rr)),
        delta={k:None if main['overall'][k] is None or old['overall'][k] is None else float(main['overall'][k]-old['overall'][k])
            for k in ('known_macro_open_accuracy','known_macro_acceptance','unknown_macro_rejection','H_OS')})
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],method_names=list(METHODS),methods=methods,primary_method=PRIMARY,
        primary_path='P1',primary_summary=main['overall'],primary_goals=main['goals'],source_gate_passed=passed,eligible_primary=passed,
        next_step='SOURCE_METRIC_COSINE_READY_FOR_TARGET_FALSIFICATION' if passed else 'NO_PROMOTION',
        reference_original_mahalanobis=old,paired_primary_minus_original_mahalanobis=paired,parent_run_id=PARENT_RUN,parent_code_commit=PARENT_CODE,parent_arm=PARENT_ARM,
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,parent_private_score_reads=0,A_N=None,H_OW=None,full_goal_status='ACTIVE',
        final_phase3_complete=False,formal_target_accessed=False,target_generalization_confirmed=False,source_RX_is_seen=True,source_novel_historically_exposed=True,
        source_free_deployment_state=True,per_record_source_rows_retained=0,known_protection_groups=42,fresh_closed_is_parent_reuse=True,
        verification_precision=embedding.VERIFICATION_PRECISION,reconstruction=dict(CPU_FP64_reference_metric_unit_max_abs_diff=reference['unit_max_diff'],
            unit_rtol=1e-5,unit_atol=2e-6,FP64_cosine_margin_atol=1e-12,parent_L_class_means_normalization_verified=True,copied_parent_g_verified=True),
        input_costs=dict(producer_g_smoke_rows=2,new_F_forward_rows=0,new_G_forward_rows=0,new_neural_updates=0,real_IQ_values_read=0,
            source_L_OE_rows_read=0,scorer_g_reference_rows=COUNTS['V']+COUNTS['novel'],scorer_cosine_readout_rows=COUNTS['V']+COUNTS['novel']),
        ancestor_costs=dict(metric_run_optimizer_steps=2000,metric_run_g_forward_rows=605350,metric_run_F_synthetic_rows=6,
            original_parent_F_forward_rows=30152,historical_P1_G_cost_reused=True,reuse_is_not_zero_cost=True),
        timing=dict(producer_CPU_readout_seconds=marker['cpu_readout_seconds'],independent_public_reconstruction_seconds=seconds,not_end_to_end_B1_latency=True),
        storage=dict(neural_parameters=520648,g_FP32_payload_bytes=82816,prototypes_FP64_bytes=7680,threshold_FP64_bytes=96,container_overhead_excluded=True),
        primary_rule='Only maximum_cosine from the fixed structured main parent can prepare a separately registered target falsification')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','reference_original_mahalanobis')});print(STATUS,result['next_step'],flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_METRIC_COSINE_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
