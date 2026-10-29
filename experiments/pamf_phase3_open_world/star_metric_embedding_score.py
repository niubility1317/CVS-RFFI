"""Independent deployment reconstruction and truth-last metric-embedding scores."""
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
from . import star_metric_embedding_run as evaluation
from . import star_metric_embedding_core as core
from . import star_local_support_run as parent
from . import star_local_support_score as previous
from . import star_joint_ova_score as common
from .metrics import open_set_metrics

RUN_ID='20261030-phase3-star-metric-embedding-s2026092701-r01'
SCHEMA='pamf_phase3_star_metric_embedding_scores_v1'
STATUS='STAR_METRIC_EMBEDDING_SCORES_COMPLETE'
VERIFICATION_PRECISION='CPU_FP64_reference_from_FP32_weights'
PARENT_RUN='20261030-phase3-star-local-support-s2026092701-r01'
PARENT_CODE='84cdc012f7d9641b283c407f6048bfe4e6e21cf6'
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
TEACHER_CODE='5aa041122396d48c49b8cb3d5912649bdc3217e1'
DEPLOYMENT_SCHEMA='pamf_phase3_star_metric_embedding_deployment_v1'
DEPLOYMENT_STATUS='STAR_METRIC_EMBEDDING_DEPLOYMENT_FROZEN'
TRAINING_SCHEMA='pamf_phase3_star_metric_embedding_training_v1'
TRAINING_STATUS='STAR_METRIC_EMBEDDING_TRAINING_COMPLETE'
ARMS=('identity','compact_oe','compact_oe_structure')
METHODS=('minimum_relative','predicted_identity_mahalanobis')
PRIMARY_ARM='compact_oe_structure';PRIMARY='predicted_identity_mahalanobis'
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
MODEL_SHAPES={'fc1.weight':(64,160),'fc1.bias':(64,),'fc2.weight':(160,64),'fc2.bias':(160,)}
METRIC_SHAPES={'class_means':(6,160),'background_mean':(160,),'within_whitener':(160,160),'background_whitener':(160,160)}
LOSS_FIELDS=('total','known_contrast','OE_repulsion','OE_structure')
TRACE_FIELDS={'step',*LOSS_FIELDS,'g_grad_norm','L_indices','OE_indices','L_exposures','OE_exposures','L_unique','OE_unique'}
RECEIVERS,SCENES,UNKNOWN_TX=common.RECEIVERS,common.SCENES,common.UNKNOWN_TX
GROUP_NAMES=common.GROUP_NAMES
_read,_write,_truth=common._read,common._write,common._truth
_npz,_units=previous._npz,previous._units


def _number(value):return type(value) in (int,float) and np.isfinite(value)


def _source(marker):
    root=Path(marker['parent_prediction_root']);pm=_read(root/'complete.json');identity=pm.get('model_identity',{})
    if (pm.get('schema')!=parent.SCHEMA or pm.get('status')!=parent.STATUS or pm.get('run_id')!=PARENT_RUN or pm.get('code_commit')!=PARENT_CODE
            or any(identity.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(marker.get(k)!=pm.get(k) for k in ('model_identity','dura_identity','cache_root','cache_identities','teacher_run_id','teacher_code_commit'))
            or pm.get('teacher_run_id')!=TEACHER_RUN or pm.get('teacher_code_commit')!=TEACHER_CODE
            or pm.get('classification_path')!='P1' or pm.get('source_only') is not True or pm.get('default_head_replaced') is not False
            or any(pm.get(k)!=v for k,v in dict(target_reads=0,source_novel_truth_reads=0,source_U_reads=0,neural_updates=0,G_updates=0,G_forward_rows=0,
                known_count=COUNTS['V'],novel_count=COUNTS['novel'],bank_L_count=COUNTS['L'],bank_OE_count=COUNTS['OE'],
                F_forward_rows=COUNTS['V']+COUNTS['novel']+2,fresh_closed_vs_cached_baseline_changed_count=0).items())
            or any(pm.get(k) is not True for k in ('source_bank_frozen_before_V','source_bank_unchanged','model_parameters_buffers_unchanged','all_thresholds_frozen_before_novel'))):
        raise ValueError('Exact completed original-F0 source parent and public feature ancestry required')
    dura=pm.get('dura_identity',{})
    if any(dura.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()):
        raise ValueError('Original fixed DURA P1 ancestry required')
    handles=pm.get('cache_identities',{})
    if (set(handles)!=set(COUNTS) or any(handles[r].get('role')!=r or handles[r].get('count')!=n
            or handles[r].get('parent_F0_identity')!=identity or handles[r].get('dura_identity')!=dura for r,n in COUNTS.items())):
        raise ValueError('Four matching validated source role handles required')
    # Training indices need source IDs/labels/RX, not source feature values or IQ.
    with np.load(root/'source_bank.npz',allow_pickle=False) as z:
        if set(z.files)!=parent.BANK_FIELDS:raise ValueError('Exact public parent-bank fields required; OE TX/K forbidden')
        lids=z['L_ids'].copy();rx=z['L_rx_i'].copy();oeids=z['OE_ids'].copy()
    cache=Path(marker['cache_root'])/'L';ids=np.load(cache/'ids.npy',allow_pickle=False)
    with np.load(cache/'metadata.npz',allow_pickle=False) as z:
        if set(z.files)!={'y','rx_i','day_i','domain'}:raise ValueError('Exact legal source-L labeled metadata required')
        y=z['y'].copy();crx=z['rx_i'].copy()
    if (not common._ids(lids,COUNTS['L']) or not common._ids(oeids,COUNTS['OE']) or set(lids.tolist())&set(oeids.tolist())
            or not np.array_equal(ids,lids) or _read(cache/'manifest.json')!=handles['L']
            or y.shape!=(COUNTS['L'],) or y.dtype!=np.int64 or not np.isin(y,range(6)).all()
            or not np.array_equal(np.bincount(y,minlength=6),np.full(6,COUNTS['L']//6))
            or rx.shape!=(COUNTS['L'],) or rx.dtype!=np.int64 or not np.array_equal(rx,crx)
            or any(np.sum((y==c)&(rx==r))<4 for c in range(6) for r in RECEIVERS)):
        raise ValueError('Source-L true class/RX metadata and original bank IDs must align')
    return root,pm,y,rx,set(lids.tolist())|set(oeids.tolist())


def _training(root,marker,y,rx):
    report=_read(root/'training_complete.json');steps=0 if marker['arm']=='identity' else 1000
    required=dict(schema=TRAINING_SCHEMA,status=TRAINING_STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],arm=marker['arm'],g_parameters=20704,
        g_training_steps=steps,g_training_L_exposures=steps*120,g_training_OE_exposures=steps*128,g_training_forward_rows=steps*248,
        model_seed=2026092701,sampling_seed=2026103002,F_updates=0,G_updates=0,target_reads=0,trace_rows=steps)
    optimizer=None if not steps else dict(name='Adam',lr=.001,weight_decay=.0001)
    if (any(report.get(k)!=v for k,v in required.items()) or report.get('optimizer')!=optimizer
            or any(report.get(k) is not True for k in ('initialization_cpu_seeded','source_only','source_L_TX_RX_labels_used','g_parameters_finite'))
            or report.get('source_OE_TX_or_K_used') is not False or not _number(report.get('train_seconds')) or report['train_seconds']<0):
        raise ValueError('Exact fixed training budget, Adam settings and legal source supervision required')
    rows=[];seenL=set();seenOE=set()
    with (root/'training.jsonl').open(encoding='utf-8') as stream:
        for step,line in enumerate(stream,1):
            row=json.loads(line)
            if not isinstance(row,dict) or set(row)!=TRACE_FIELDS or step>steps or row.get('step')!=step:
                raise ValueError('Complete ordered training trace with exact public training fields required')
            if any(not _number(row[k]) or row[k]<0 for k in (*LOSS_FIELDS,'g_grad_norm')):
                raise ValueError('Finite nonnegative complete component losses and gradient norms required')
            expected=row['known_contrast']+10*row['OE_repulsion']+(10*row['OE_structure'] if marker['arm']==PRIMARY_ARM else 0.)
            if not np.isclose(row['total'],expected,rtol=1e-5,atol=1e-6):raise ValueError('Declared arm loss weights must match the complete loss trace')
            for key,n,count in (('L_indices',COUNTS['L'],120),('OE_indices',COUNTS['OE'],128)):
                v=row[key]
                if not isinstance(v,list) or len(v)!=count or any(type(i) is not int or i<0 or i>=n for i in v) or len(set(v))!=count:
                    raise ValueError('Exact without-replacement source-only training batch indices required')
            index=np.asarray(row['L_indices'],np.int64)
            if any(np.sum((y[index]==c)&(rx[index]==r))!=4 for c in range(6) for r in RECEIVERS):
                raise ValueError('Every training step must retain six-TX by five-source-RX balanced L sampling')
            seenL.update(row['L_indices']);seenOE.update(row['OE_indices'])
            if any(row[k]!=v for k,v in dict(L_exposures=step*120,OE_exposures=step*128,L_unique=len(seenL),OE_unique=len(seenOE)).items()):
                raise ValueError('Actual cumulative training exposure and unique coverage must match the trace')
            rows.append({k:row[k] for k in LOSS_FIELDS})
    if (len(rows)!=steps or report.get('g_training_L_unique')!=len(seenL) or report.get('g_training_OE_unique')!=len(seenOE)
            or report.get('loss_initial')!=(rows[0] if rows else None) or report.get('loss_final')!=(rows[-1] if rows else None)):
        raise ValueError('All fixed steps, actual unique coverage and initial/final loss records required')
    return report


def _deployment(root,marker):
    path=root/'deployment'
    if {p.name for p in path.iterdir()}!={'model.pt','metric.npz','thresholds.npz','complete.json'}:
        raise ValueError('Deployment must contain exactly four files and no retained source rows')
    dm=_read(path/'complete.json');expected=dict(schema=DEPLOYMENT_SCHEMA,status=DEPLOYMENT_STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],
        arm=marker['arm'],parameters=20704,source_free=True,per_record_source_rows_retained=0)
    if dm!=expected:raise ValueError('Exact source-free deployment identity required, without source sample metadata')
    state=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    if (not isinstance(state,dict) or set(state)!=set(MODEL_SHAPES)
            or any(not isinstance(v,torch.Tensor) or tuple(v.shape)!=MODEL_SHAPES[k] or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for k,v in state.items())
            or sum(v.numel() for v in state.values())!=20704):
        raise ValueError('Only four finite FP32 MetricNet parameter tensors may be deployed')
    if marker['arm']=='identity' and (bool(torch.count_nonzero(state['fc2.weight'])) or bool(torch.count_nonzero(state['fc2.bias']))):
        raise ValueError('Identity arm must retain the zero-initialized residual output layer')
    net=core.MetricNet().cpu().float();net.load_state_dict(state,strict=True);net.double().eval()
    metric=_npz(path/'metric.npz',METRIC_SHAPES)
    if any(v.dtype!=np.float64 or v.shape!=METRIC_SHAPES[k] or not np.isfinite(v).all() for k,v in metric.items()):
        raise ValueError('Exactly four finite FP64 source-L aggregate arrays required; no per-record source state')
    tau=_npz(path/'thresholds.npz',{'conditional'})['conditional']
    if tau.shape!=(2,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all():raise ValueError('Twelve finite FP64 class thresholds required')
    return net,metric,tau


def _reference_smoke(net,marker,output):
    x=torch.zeros((2,160),dtype=torch.float64);x[0,0]=1.;x[1,159]=1.
    with torch.no_grad():value=net(x)
    if value.dtype!=torch.float64 or tuple(value.shape)!=(2,160) or not bool(torch.isfinite(value).all()):
        raise ValueError('Actual deployed MetricNet FP64 reference smoke must produce finite unit vectors')
    norms=np.asarray(torch.linalg.vector_norm(value,dim=1).tolist(),np.float64)
    if not np.allclose(norms,1.,rtol=0,atol=2e-12):raise ValueError('Actual deployed MetricNet reference smoke must remain unit norm')
    _write(Path(output),dict(schema='pamf_phase3_star_metric_embedding_scorer_smoke_v1',status='PASS',
        prediction_run_id=marker['run_id'],prediction_code_commit=marker['code_commit'],arm=marker['arm'],
        actual_deployment_checkpoint=True,verification_precision=VERIFICATION_PRECISION,synthetic_unit_rows=2,
        unit_norms=norms.tolist(),parameters=20704,g_reference_forward_rows=2,F_forward_rows=0,G_forward_rows=0,
        source_known_rows_read=0,source_novel_rows_read=0,private_truth_reads=0,neural_updates=0))


def _prediction(root,smoke_output=None):
    root=Path(root);marker=_read(root/'complete.json');arm=marker.get('arm');steps=0 if arm=='identity' else 1000
    if (marker.get('schema')!=evaluation.SCHEMA or marker.get('status')!=evaluation.STATUS or marker.get('run_id')!=RUN_ID or arm not in ARMS
            or re.fullmatch('[0-9a-f]{40}',str(marker.get('code_commit'))) is None
            or marker.get('parent_run_id')!=PARENT_RUN or marker.get('parent_code_commit')!=PARENT_CODE
            or marker.get('method_names')!=list(METHODS) or marker.get('primary_arm')!=PRIMARY_ARM or marker.get('primary_method')!=PRIMARY
            or marker.get('classification_path')!='P1' or marker.get('default_head_replaced') is not False or marker.get('source_only') is not True
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged','metric_parameters_buffers_frozen_for_V_novel',
                'aggregate_frozen_before_V','all_thresholds_frozen_before_novel','source_free_deployment_state','parent_public_features_reused','source_L_TX_RX_labels_used'))
            or marker.get('source_OE_TX_or_K_used') is not False
            or any(marker.get(k)!=v for k,v in dict(known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
                g_parameters=20704,g_training_steps=steps,g_training_L_exposures=steps*120,g_training_OE_exposures=steps*128,
                g_forward_rows=steps*248+COUNTS['L']+COUNTS['V']+COUNTS['novel'],F_forward_rows=2,synthetic_P0_forward_rows=2,G_forward_rows=0,real_IQ_values_read=0,
                source_L_metadata_labels_read=COUNTS['L'],neural_F_updates=0,G_updates=0,source_U_reads=0,target_reads=0,source_novel_truth_reads=0,
                per_record_source_rows_retained=0,fresh_closed_vs_cached_baseline_changed_count=0).items())
            or any(not _number(marker.get(k)) or marker[k]<0 for k in ('cpu_readout_seconds','g_evaluation_seconds'))):
        raise ValueError('Complete original-head source predictions, frozen deployment and exact per-arm costs required before truth')
    parent_root,pm,y,rx,seen=_source(marker);training=_training(root,marker,y,rx);net,metric,tau=_deployment(root,marker)
    if smoke_output is not None:_reference_smoke(net,marker,smoke_output)
    cal=_read(root/'calibration/complete.json')
    if (cal.get('schema')!=evaluation.SCHEMA or cal.get('status')!=evaluation.CALIBRATION_STATUS
            or any(cal.get(k)!=marker[k] for k in ('run_id','code_commit','arm')) or cal.get('method_names')!=list(METHODS)
            or cal.get('group_count')!=42 or cal.get('source_only') is not True or cal.get('target_reads')!=0
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal.get('source_V_neural_fit') is not False
            or set(cal.get('methods',{}))!=set(METHODS) or not np.array_equal(cal.get('conditional_thresholds'),tau)
            or any(set(cal['methods'][name].get('groups',{}))!=set(GROUP_NAMES) or not np.array_equal(cal['methods'][name].get('conditional_thresholds'),tau[j]) for j,name in enumerate(METHODS))):
        raise ValueError('Matching frozen source-known calibration, all 42 groups and twelve thresholds required')
    rows=[];diff={}
    for role,fields,n in (('known',evaluation.KNOWN_FIELDS,COUNTS['V']),('novel',evaluation.NOVEL_FIELDS,COUNTS['novel'])):
        data=_npz(root/(role+'_predictions.npz'),fields);old=_npz(parent_root/(role+'_predictions.npz'),fields-{'metric_unit_features'})
        if (any(not np.array_equal(data[k],old[k]) or data[k].dtype!=old[k].dtype for k in fields-{'metric_unit_features','raw_scores','scores'})
                or not common._ids(data['ids'],n) or seen&set(data['ids'].tolist())
                or not _units(data['unit_features'],n) or not _units(data['metric_unit_features'],n)
                or data['default_logits'].shape!=(n,6) or data['default_logits'].dtype!=np.float32 or not np.isfinite(data['default_logits']).all()
                or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64 or not np.array_equal(data['closed'],data['default_logits'].argmax(1))
                or any(data[k].shape!=(n,) or data[k].dtype!=np.int64 or not np.isin(data[k],r).all() for k,r in (('rx_i',RECEIVERS),('day_i',(1,2,3))))
                or any(data[k].shape!=(n,2) or data[k].dtype!=np.float64 or not np.isfinite(data[k]).all() for k in ('raw_scores','scores'))):
            raise ValueError('Original parent features/default-head metadata and finite metric-unit predictions required')
        if role=='known' and (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or not np.isin(data['y'],range(6)).all()
                or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
                or data['baseline_closed'].dtype!=np.int64 or not np.array_equal(data['baseline_closed'],data['closed'])):
            raise ValueError('Original cached P1 baseline and unchanged known metadata required')
        rebuilt=[]
        with torch.no_grad():
            for start in range(0,n,128):
                x=torch.tensor(data['unit_features'][start:start+128].tolist(),dtype=torch.float64)
                rebuilt.append(np.asarray(net(x).tolist(),np.float32))
        units=np.concatenate(rebuilt);diff[role]=float(np.max(np.abs(units.astype(np.float64)-data['metric_unit_features'])))
        if not np.allclose(units,data['metric_unit_features'],rtol=1e-5,atol=2e-6):
            raise ValueError('Saved metric units must reconstruct from only deployed network parameters and original F0 units')
        # FP32 deployment weights are losslessly promoted for CPU FP64 reference
        # reconstruction. Readout still uses the verified production FP32 units.
        raw=core.metric_scores(data['metric_unit_features'],data['closed'],metric)
        if (not np.allclose(raw,data['raw_scores'],rtol=0,atol=1e-10)
                or not np.allclose(raw-tau[:,data['closed']].T,data['scores'],rtol=0,atol=1e-10)):
            raise ValueError('Both frozen aggregate distances and predicted-class margins must reconstruct')
        seen.update(data['ids'].tolist());rows.append(data)
    return marker,cal,tau,*rows,dict(training=training,CPU_metric_unit_max_abs_diff=diff,parent_F_forward_rows=pm['F_forward_rows'])


def score(prediction_root,novel_truth,output,*,scorer_run_id=None,scorer_code_commit=None,smoke_output=None):
    output=Path(output)
    if output.exists():raise FileExistsError('A new immutable independent per-arm score root is required')
    if ((scorer_run_id is None)!=(scorer_code_commit is None)
            or scorer_run_id is not None and (not isinstance(scorer_run_id,str) or not scorer_run_id.strip()
                or not isinstance(scorer_code_commit,str) or re.fullmatch('[0-9a-f]{40}',scorer_code_commit) is None)):
        raise ValueError('Explicit scorer run and forty-character code identity must be supplied together')
    started=time.perf_counter();marker,cal,tau,known,novel,evidence=_prediction(prediction_root,smoke_output=smoke_output);seconds=time.perf_counter()-started
    truth=_truth(novel_truth,novel)
    km=dict(common._masks(known['rx_i'],known['scenario']));um=dict(common._masks(novel['rx_i'],truth['scenario']));methods={}
    for j,name in enumerate(METHODS):
        def calculate(kmask,umask,known_roster=tuple(range(6)),unknown_roster=UNKNOWN_TX):
            row=open_set_metrics(known['y'][kmask],known['closed'][kmask],known['scores'][kmask,j],truth['tx'][umask],novel['scores'][umask,j],0.,
                known_roster,unknown_roster,baseline_prediction=known['baseline_closed'][kmask])
            values=[float(np.mean(known['closed'][kmask&(known['y']==c)]==c)) if np.any(kmask&(known['y']==c)) else None for c in known_roster]
            row['fresh_F0_macro_closed_accuracy']=None if any(v is None for v in values) else float(np.mean(values));return row
        groups={key:calculate(mask,um[key]) for key,mask in km.items()};overall=groups['overall']
        methods[name]=dict(overall=overall,groups=groups,conditional_thresholds=tau[j].tolist(),
            by_known_TX={str(c):calculate(known['y']==c,um['overall'],(c,)) for c in range(6)},
            by_unknown_TX={tx:calculate(km['overall'],truth['tx']==tx,unknown_roster=(tx,)) for tx in UNKNOWN_TX},
            confidence_intervals=dict(level=.95,method='per true-class Wilson; descriptive source pool only',
                A_K_per_class=common._ci(overall['known_per_class'],'correct_accepted'),R_U_per_class=common._ci(overall['unknown_per_class'],'rejected')),
            goals=common._goals(groups,cal['methods'][name].get('constraints_passed') is True),diagnostic_only=name!=PRIMARY or marker['arm']!=PRIMARY_ARM)
    retrieval={name:common.cross_rx_retrieval(novel[key],novel['ids'],novel['rx_i'],truth['tx'],UNKNOWN_TX)
        for name,key in (('metric','metric_unit_features'),('original_F0','unit_features'))}
    main=methods[PRIMARY];passed=bool(main['goals']['passed']);eligible=passed and marker['arm']==PRIMARY_ARM
    ka=known['scores']>=0;ur=novel['scores']<0;correct=known['closed']==known['y']
    paired=dict(known_correct_gained=int(np.sum(correct&ka[:,1]&~ka[:,0])),known_correct_lost=int(np.sum(correct&~ka[:,1]&ka[:,0])),
        known_acceptance_gained=int(np.sum(ka[:,1]&~ka[:,0])),known_acceptance_lost=int(np.sum(~ka[:,1]&ka[:,0])),
        unknown_rejection_gained=int(np.sum(ur[:,1]&~ur[:,0])),unknown_rejection_lost=int(np.sum(~ur[:,1]&ur[:,0])),
        delta={key:None if main['overall'][key] is None or methods[METHODS[0]]['overall'][key] is None
            else float(main['overall'][key]-methods[METHODS[0]]['overall'][key])
            for key in ('known_macro_open_accuracy','known_macro_acceptance','unknown_macro_rejection','H_OS')})
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],arm=marker['arm'],method_names=list(METHODS),methods=methods,
        scorer_run_id=marker['run_id'] if scorer_run_id is None else scorer_run_id,
        scorer_code_commit=marker['code_commit'] if scorer_code_commit is None else scorer_code_commit,verification_precision=VERIFICATION_PRECISION,
        primary_arm=PRIMARY_ARM,primary_method=PRIMARY,primary_path='P1',primary_summary=main['overall'],primary_goals=main['goals'],
        source_gate_passed=passed,arm_source_gate_passed=passed,eligible_primary=bool(eligible),
        next_step='SOURCE_METRIC_EMBEDDING_READY_FOR_TARGET_FALSIFICATION' if eligible else 'CONTROL_ONLY_COMPLETE' if passed else 'NO_PROMOTION',
        paired_primary_minus_minimum_relative=paired,novel_cross_RX_retrieval=retrieval,novel_structure_is_diagnostic_only=True,novel_structure_is_promotion_gate=False,
        source_free_deployment_state=True,per_record_source_rows_retained=0,model_identity=marker['model_identity'],parent_run_id=PARENT_RUN,parent_code_commit=PARENT_CODE,
        source_only=True,target_reads=0,truth_last=True,private_source_novel_truth_read=True,A_N=None,H_OW=None,full_goal_status='ACTIVE',final_phase3_complete=False,
        formal_target_accessed=False,target_generalization_confirmed=False,source_RX_is_seen=True,source_novel_historically_exposed=True,known_protection_groups=42,
        source_L_TX_RX_labels_used=True,source_OE_membership_used=True,source_OE_TX_or_K_used=False,source_V_labels_used_for_thresholds=True,
        training=evidence['training'],input_costs=dict({k:marker[k] for k in ('g_training_steps','g_training_L_exposures','g_training_OE_exposures',
            'g_forward_rows','F_forward_rows','synthetic_P0_forward_rows','G_forward_rows','real_IQ_values_read')},
            g_training_forward_rows=evidence['training']['g_training_forward_rows'],g_source_L_aggregate_rows=COUNTS['L'],g_evaluation_rows=COUNTS['V']+COUNTS['novel']),
        ancestor_costs=dict(parent_F_forward_rows=evidence['parent_F_forward_rows'],historical_P1_restoration_reused=True,reuse_is_not_zero_cost=True),
        timing=dict(production_CPU_readout_seconds=marker['cpu_readout_seconds'],production_g_evaluation_seconds=marker['g_evaluation_seconds'],
            independent_public_reconstruction_seconds=seconds,not_end_to_end_B1_latency=True),
        reconstruction=dict(CPU_FP64_reference_metric_unit_max_abs_diff=evidence['CPU_metric_unit_max_abs_diff'],metric_unit_rtol=1e-5,metric_unit_atol=2e-6,
            FP64_raw_margin_atol=1e-10,FP64_readout_uses_verified_production_metric_units=True),
        fresh_closed_is_parent_reuse=True,fresh_closed_vs_cached_baseline_changed_count=0,
        storage=dict(F_parameters=232993,G_parameters=266951,g_parameters=20704,total_neural_parameters=520648,g_FP32_bytes=82816,aggregate_FP64_bytes=418560,threshold_FP64_bytes=96),
        primary_rule='Only compact_oe_structure::predicted_identity_mahalanobis can prepare a separately registered target falsification; source 1NN is diagnostic only')
    output.mkdir(parents=True,exist_ok=False);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','novel_cross_RX_retrieval')});print(STATUS,marker['arm'],result['next_step'],flush=True);return result


def main():
    print(json.dumps(dict(event='STAR_METRIC_EMBEDDING_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    p.add_argument('--scorer-run-id');p.add_argument('--scorer-code-commit');p.add_argument('--smoke-output')
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output,scorer_run_id=a.scorer_run_id,scorer_code_commit=a.scorer_code_commit,smoke_output=a.smoke_output)


if __name__=='__main__':main()
