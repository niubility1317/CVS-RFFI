"""Frozen SupportMetric B -> C and independent R0 single-query predictions.

This source contains no query-label reader, score-based selection or launch.
Input validation retains the existing received-cache/capsule/packet closure.
"""
import argparse
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
from cvsrffi import d92_support_metric_joint_local_ridge as core
from cvsrffi.d92_branch_local_ridge import fit_branch_local_ridge
from cvsrffi.d92_ground_classifier_a import GroundFeatureContract
from cvsrffi.d92_proto_frame_ground_geometry import load_proto_frame_ground_geometry
from export_d92_ground_classifier_a_packet import load_packet
from export_d92_branch_features import (CACHE_SCHEMA,FEATURE_CONTRACT,local_core as cache_core,
    load_features,validate_capsule,read,peak_process_rss)
from d92_orbit_feature_cache import validate_split
from evaluate_d92_support_metric_joint_probe import (
    StateArchive,json_native,compact_event,training_text,_empty_work,_work_merge,WORK_KEYS,
    build_row_basis,_basis_refs,_small_measurements,
)
from evaluate_d92_branch_support_probe import csv_record

SCHEMA='d92_support_metric_joint_query_predictions_v1'
METHOD=core.METHOD
STATUS='COMPLETE'
TRAIN_SCOPE='query_benchmark_support_training'
SCOPE='SUPPORT_METRIC_JOINT_REPEAT_QUERY_PERFORMANCE'
BRANCHES=('z_id','fft','t_emb','f_emb','pa_local')
RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160,
    max_integer_bits=65536,max_fraction_operations=65536,max_secular_iterations=128)
WORK_SUM_KEYS=core.WORK_SUM_KEYS
WORK_MAX_KEYS=core.WORK_MAX_KEYS
WORK_AGGREGATION={key:'MAX' if key in WORK_MAX_KEYS else 'SUM' for key in WORK_KEYS}
DECISION_SCHEMA='d92_support_metric_single_query_decision_v1'
DECISION_POLICY='WITHIN_GROUP_RAW_ARGMAX_THEN_GATE_LOGPROB_GAP_LEXICAL_TIE'
STREAM_NAMES=('A','B','C','R0_B','R0_C')


def check(condition,message):
    if not condition:raise ValueError(message)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(json_native(value),stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def validate_config(config):
    check(type(config) is dict and set(config)=={'algorithm','support_metric_resources'}
        and config['algorithm']==core.FROZEN_CONFIG,'Frozen SupportMetric benchmark algorithm/config mismatch')
    limits=config['support_metric_resources']
    check(type(limits) is dict and set(limits)==set(RESOURCES)
        and all(type(v) is int and v>0 for v in limits.values()),'Explicit positive integer SupportMetric resources required')
    check(limits==RESOURCES,'Frozen SupportMetric benchmark technical resources mismatch')


def _inputs(*,run_id,row_id,release_commit,row_root,capsule,output,config,
            expected_capsule_id,expected_checkpoint_sha256,branch_features,ground_packet,
            ground_summary,ground_summary_already_deployed):
    validate_config(config)
    check(all(type(v) is str and v for v in (run_id,row_id)),'Explicit benchmark run/row required')
    check(type(release_commit) is str and re.fullmatch('[0-9a-f]{40}',release_commit),'Explicit actual release commit required')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    inputs=[Path(p).resolve() for p in (row_root,capsule,branch_features,ground_packet,ground_summary)]
    check(all(out!=p and not out.is_relative_to(p) for p in inputs),'Output overlaps original input directory')
    check(type(ground_summary_already_deployed) is bool,'Explicit ground-summary already-deployed boolean required')
    started=time.perf_counter()
    arrays,ids,producer,previous,provenance=load_features(branch_features=branch_features,capsule=capsule,
        row_root=row_root,expected_capsule_id=expected_capsule_id,expected_checkpoint_sha256=expected_checkpoint_sha256,
        config=dict(algorithm=cache_core().FROZEN_CONFIG))
    ids=np.asarray(ids).astype(str);ids.setflags(write=False)
    cache_seconds=time.perf_counter()-started
    old=list(producer['classes']);check(len(old)==6 and len(set(old))==6,'Exactly six declared original classes required')
    check(set(arrays)==set(BRANCHES),'Original five received feature blocks required')
    check(type(previous['seed']) is int and previous['seed']>=0,'Actual producer model seed required')
    started=time.perf_counter();head=load_packet(ground_packet);packet_seconds=time.perf_counter()-started
    contract=GroundFeatureContract(expected_checkpoint_sha256,'z_id','feat_joint','raw','float32',160)
    check(head.metadata.checkpoint_sha256==expected_checkpoint_sha256 and head.metadata.feature_contract==contract
        and set(head.classes)==set(old),'Ground packet/cache source identity or original classes mismatch')
    packet_marker=read(Path(ground_packet)/'complete.json');packet_meta=read(Path(ground_packet)/'metadata.json')
    packet_seed=packet_meta['existing_source_only_provenance'].get('model_seed')
    check(packet_seed==previous['seed'],'Ground packet/cache model seed mismatch')
    manifest=validate_capsule(Path(capsule),expected_capsule_id)
    paths=sorted((Path(capsule)/'splits').glob('*.json'));splits=[read(p) for p in paths]
    check(paths and len(paths)==manifest['split_count'] and len({s['split_id'] for s in splits})==len(splits)
        and all(p.stem==s['split_id'] for p,s in zip(paths,splits)),'Incomplete/ambiguous physical split matrix')
    tasks=[];metadata=[]
    for split in splits:
        support,query,labels=validate_split(split,manifest,ids,old)
        names=[split['registered_classes'][int(y)] for y in labels]
        old_mask=np.asarray([name in old for name in names],dtype=bool)
        sid=str(split['split_id']);check(sid,'Missing split identity')
        tasks.append((split,support,query,labels,old_mask))
        metadata.append(dict(split_id=sid,receiver=split['receiver'],scenario=split['scenario'],k=split['k'],
            new_count=len(split['registered_classes'])-6,support_seed=split['support_seed'],
            old_classes=sorted(old),registered_classes=sorted(split['registered_classes']),
            declared_registered_classes=list(split['registered_classes']),
            support_ids=sorted(ids[support].astype(str).tolist()),old_support_ids=sorted(ids[support[old_mask]].astype(str).tolist()),
            new_support_ids=sorted(ids[support[~old_mask]].astype(str).tolist()),
            query_ids=ids[query].astype(str).tolist(),query_count=len(query)))
    source_identity=dict(checkpoint_sha256=expected_checkpoint_sha256,model_seed=previous['seed'],
        source_only_verdict=provenance['verdict'],source_role_comparison=provenance['source_role_comparison'],
        checkpoint_epoch=provenance['checkpoint_epoch'],checkpoint_inheritance=provenance['checkpoint_inheritance'],
        target_access_before_freeze=provenance['target_access_before_freeze'],cache_schema=CACHE_SCHEMA,feature_contract=FEATURE_CONTRACT)
    ground_identity=dict(path=str(Path(ground_packet).resolve()),checkpoint_sha256=head.metadata.checkpoint_sha256,
        model_seed=previous['seed'],packet_declared_model_seed=packet_seed,ordered_classes=list(head.classes),
        scale=head.metadata.scale,norm_eps=head.metadata.norm_eps,feature_contract=asdict(head.metadata.feature_contract),
        source_only_verdict=head.metadata.source_only_verdict,packet_total_file_bytes=packet_marker['packet_total_file_bytes'],
        head_weight_file_bytes=(Path(ground_packet)/'head_weight.float32.bin').stat().st_size,
        head_weight_numeric_bytes=len(head.classes)*head.metadata.feature_contract.feature_dim*np.dtype(np.float32).itemsize,
        byte_scope='LOCAL_PACKET_FILE_AND_NATIVE_FLOAT32_WEIGHT_BUFFER; NOT_WIRE_OR_DEPLOYMENT_MEMORY')
    tick=time.perf_counter()
    frame,geometry_audit=load_proto_frame_ground_geometry(ground_summary,
        expected_checkpoint_sha256=expected_checkpoint_sha256,expected_classes=old,
        already_deployed=ground_summary_already_deployed)
    geometry_audit=json_native(geometry_audit)
    geometry_audit['total_geometry_binding_seconds']=time.perf_counter()-tick
    geometry_identity=dict(path=str(inputs[4]),checkpoint_sha256=expected_checkpoint_sha256,
        model_seed=previous['seed'],already_deployed=ground_summary_already_deployed,
        input_role='FROZEN_PHASE1_CENTER_ONLY_GEOMETRY',source_examples=False,
        prototype_teacher_targets=False,checkpoint_loaded=False)
    public=dict(schema=SCHEMA,method=METHOD,scope=SCOPE,run_id=run_id,row_id=row_id,release_commit=release_commit,
        capsule_id=expected_capsule_id,checkpoint_sha256=expected_checkpoint_sha256,model_seed=previous['seed'],
        algorithm=deepcopy(config['algorithm']),support_metric_resources=deepcopy(config['support_metric_resources']),
        run_binding=dict(prediction_output_root=str(out),row_root=str(inputs[0]),capsule=str(inputs[1]),
            branch_features=str(inputs[2]),ground_packet=str(inputs[3]),ground_summary=str(inputs[4])),
        ground_summary_already_deployed=ground_summary_already_deployed,ground_geometry_binding=geometry_audit,
        ground_geometry_identity=geometry_identity,
        source_identity=source_identity,ground_packet_identity=ground_identity,ordered_ground_classes=list(head.classes),
        old_classes=sorted(old),splits=metadata,split_count=len(splits),
        query_record_count=sum(v['query_count'] for v in metadata),query_fit_access=False,source_fit_access=False,truth_read=False,
        checkpoint_loaded=False,encoder_called=False,cross_split_adapted_state_reuse=False,
        source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',
        A_tie_policy='first_original_native_head_column',B_C_tie_policy='physical_class_id_ascending',
        C_decision_policy=DECISION_POLICY,C_decision_certificate_schema=DECISION_SCHEMA,
        fit_scope='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY',query_inference_scope='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS',
        technical_query_chunk_size=1,nominal_adapter_parameter_count=5,
        row_basis_construction_status='PENDING_EXPLICIT_FROZEN_Q_FACTORY',
        row_basis_namespace=dict(run_id=run_id,row_id=row_id,split_id=None,scope='row_frozen_ground_geometry',
            fold=None,trial=None,parent_k=None,train_k=None,state='ROW_SUPPORT_METRIC_BASIS'),
        old_teacher_scope='C_INNER_PRIOR_FROM_OLD_INNER_TRAIN; C_FINAL_PRIOR_FROM_ACTUAL_SPLIT_B',
        query_role_used=False,query_count_used_for_decision=False,view_aggregation=False,
        prototype_geometry_role='FROZEN_REFERENCE_ONLY_NOT_TEACHER',
        work_aggregation='SUM/MAX',actual_work_aggregation=WORK_AGGREGATION,
        paths={'R0':'INDEPENDENT_COMPLETE_SUPPORT_LOCAL_RIDGE_B_OLD_C_ALL',
            'R_SUPPORT_METRIC_seq':'ACTUAL_CURRENT_SPLIT_B_TO_C_SEQUENTIAL_INHERITANCE'},
        cache_resources=dict(cache_load_seconds=cache_seconds,ground_packet_load_seconds=packet_seconds,
            feature_array_bytes=producer['feature_array_bytes'],feature_file_bytes=producer['feature_file_bytes'],
            existing_checkpoint_file_bytes=producer['model_file_bytes'],encoder_calls_this_prediction=0))
    return dict(public=public,arrays=arrays,ids=ids,old=old,head=head,tasks=tasks,prototype_frame=frame)


def preflight(*,run_id,row_id,release_commit,row_root,capsule,output,config,
              expected_capsule_id,expected_checkpoint_sha256,branch_features,ground_packet,
            ground_summary,ground_summary_already_deployed):
    """Existing immutable input permission/binding check; no fit or inference."""
    return dict(_inputs(**locals())['public'],status='SUPPORT_METRIC_QUERY_PREFLIGHT_COMPLETE')


def _csv_from_jsonl(path):
    keys=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:keys.update(json.loads(line))
    with path.with_suffix('.csv').open('x',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=sorted(keys));writer.writeheader()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);writer.writerow(csv_record({k:row.get(k) for k in sorted(keys)}))


def _prediction(split_id,query_id,classes,scores,prediction=None):
    values=np.asarray(scores)
    check(values.shape==(1,len(classes)) and np.isfinite(values).all(),'Invalid singleton all-class prediction')
    pred=min(classes[j] for j in np.flatnonzero(values[0]==np.max(values[0]))) if prediction is None else str(prediction)
    check(pred in classes,'Fixed prediction is unregistered')
    return dict(split_id=split_id,query_id=str(query_id),classes=list(classes),scores=values[0].tolist(),prediction=pred)


def _structured_score(state,one):
    """The exact core single-record score path, retaining its intermediate evidence."""
    raw=core._raw(**one)
    check(len(raw['z_id'])==1,'Exactly one physical query required')
    ledger=core._Ledger();tick=time.perf_counter()
    try:scores,parts=state._proxy()._score(raw,ledger)
    except Exception as exc:
        original=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else getattr(exc,'audit',None)
        exc.support_metric_query_score_audit=dict(status='TECHNICAL_FAILURE',score_seconds=time.perf_counter()-tick,
            failed_operation_audit=original,**ledger.audit())
        raise
    check(parts is not None,'Registered C structural evidence required')
    B,h,g,lo,ln=parts;oldcols,newcols=state.head['old_columns'],state.head['new_columns']
    old=[state.classes[int(j)] for j in oldcols];new=[state.classes[int(j)] for j in newcols]
    check(B.shape==(1,len(old)) and h.shape==(1,len(new)) and len(g)==1,'Non-singleton structural evidence')
    bo,bn=int(np.argmax(B[0])),int(np.argmax(h[0]));gap=float(g[0]+lo[0,bo]-ln[0,bn])
    ow,nw=old[bo],new[bn];pred=ow if gap>0 else nw if gap<0 else min(ow,nw)
    cert=dict(schema=DECISION_SCHEMA,policy=DECISION_POLICY,new0_reuses_actual_B=False,
        old_classes=old,new_classes=new,old_raw_scores=B[0].tolist(),new_raw_scores=h[0].tolist(),
        gate_logit=float(g[0]),old_log_probabilities=lo[0].tolist(),new_log_probabilities=ln[0].tolist(),
        old_winner=ow,new_winner=nw,group_gap=gap,prediction=pred)
    json_native(cert)
    return scores,dict(score_seconds=time.perf_counter()-tick,single_record_all_registered_classes=True,
        **ledger.audit()),cert


def _baseline_arrays(state):
    return dict(original_train_b=state.support_background,original_train_a=state.support_auxiliary,
        alpha=state.alpha,reference_kernel=state.reference_kernel,reference_self=np.asarray(state.reference_self),
        center_mean=state.center_mean,center_grand=np.asarray(state.center_grand),
        tau=np.asarray([] if state.bandwidth_tau is None else [state.bandwidth_tau],dtype=np.float64),
        gamma=np.asarray([] if state.trace_scale is None else [state.trace_scale],dtype=np.float64))


def _stage_check(state,registry,support_ids,mode,actual_b,basis,audit):
    check(list(state.classes)==sorted(registry) and sorted(state.ids)==sorted(map(str,support_ids)),
        'Actual fitted support/classes mismatch')
    check(audit.get('status')=='COMPLETED' and isinstance(audit.get('final_state_ref'),Mapping)
        and audit.get('final_head_complete') is True,'Missing completed actual state')
    check(type(audit.get('optimizer_steps')) is int and 0<=audit['optimizer_steps']<=1
        and isinstance(audit.get('trials'),list) and len(audit['trials'])==audit.get('trial_count')
        and len(audit['trials'])<=12,'Frozen one-update/twelve-trial budget mismatch')
    check(sum(t.get('accepted') is True for t in audit['trials'])==audit['optimizer_steps'],
        'Accepted-update ledger mismatch')
    check(state.basis is basis and state.theta.shape==(basis.rank,), 'Actual immutable row U basis differs')
    check(audit.get('nominal_parameter_count')==5 and audit.get('effective_parameter_rank')==basis.rank
        and audit.get('trainable_parameter_count') in (0,basis.rank), 'Actual effective U-coordinate declaration missing')
    if mode=='C_seq':
        check(state.prior is actual_b,'C did not inherit same-split actual B')
        check(audit.get('final_prior_ref')==actual_b.audit_dict()['final_state_ref'],
            'C final prior reference differs from actual B')


def predict(*,run_id,row_id,release_commit,row_root,capsule,output,config,
            expected_capsule_id,expected_checkpoint_sha256,branch_features,ground_packet,
            ground_summary,ground_summary_already_deployed):
    kwargs=locals().copy();started=time.perf_counter();out=Path(output).resolve()
    loaded=_inputs(**kwargs);out.mkdir(parents=True,exist_ok=False)
    public=loaded['public'];arrays,ids,old,head=(loaded[k] for k in ('arrays','ids','old','head'))
    frame=loaded['prototype_frame'];archive=StateArchive(out);completed=[];stages=[];active=None
    basis=None;row_basis_owner=None
    active_query=None;active_score_work=None
    work={phase:_empty_work() for phase in ('row_basis','preparation','stage','score')};totals=_empty_work()
    baseline_audits=[];baseline_totals=dict(completed_head_fits=0,factorization_calls=0,triangular_solve_calls=0)
    score_counts=dict.fromkeys(STREAM_NAMES,0);score_seconds=dict.fromkeys(STREAM_NAMES,0.)
    structural_calls=0;structural_seconds=0.;structural_attempts=0;structural_attempt_seconds=0.
    high=dict(candidate_preparation_count=0,candidate_stage_count=0,final_candidate_head_fit_count=0,
        optimizer_steps=0,trial_count=0,accepted_trial_count=0,rejected_trial_count=0,
        ggn_step_count=0,ggn_parameter_direction_count=0,peak_resident_numeric_state_bytes=0,
        row_basis_construction_count=0,peak_effective_adapter_rank=0)
    callback_seconds=dict(state_archive_seconds=0.,training_event_seconds=0.)
    def account(phase,audit):
        check(isinstance(audit.get('operation_audits'),list),'Actual operation audits required')
        _work_merge(work[phase],audit['actual_work']);_work_merge(totals,audit['actual_work'])
    write(out/'startup.json',dict(public,config=config,argv=sys.argv,python=sys.executable,pid=os.getpid(),
        stage_namespace_scope=TRAIN_SCOPE,device='cpu',fixed_method_no_selection=True,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        hardware=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
            cpu_count=os.cpu_count(),torch_version=str(torch.__version__),dtype='float64; native_A_float32'),
        blas_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    try:
        active=dict(split_id=None,phase='ROW_SUPPORT_METRIC_BASIS')
        basis,row_basis_owner=build_row_basis(prototype_frame=frame,resources=config['support_metric_resources'],
            context=dict(run_id=run_id,row_id=row_id),state_callback=archive)
        account('row_basis',row_basis_owner)
        high.update(row_basis_construction_count=1,peak_effective_adapter_rank=basis.rank)
        write(out/'row_basis.json',row_basis_owner)
        with ExitStack() as stack:
            names={key:'predictions_'+key+'.jsonl' for key in STREAM_NAMES}
            names.update(alias='predictions.jsonl',events='training_events.jsonl',compact='training_events_compact.jsonl',
                stages='fit_stages.jsonl',preparations='preparations.jsonl',trace='fit_trace.jsonl',
                work='query_score_work.jsonl',text='training.log')
            streams={key:stack.enter_context((out/name).open('x',encoding='utf-8')) for key,name in names.items()}
            def emit(key,value):
                streams[key].write(json.dumps(json_native(value),ensure_ascii=False,allow_nan=False)+'\n');streams[key].flush()
            streams['text'].write('STARTUP '+json.dumps(json_native(dict(public,config=config)),allow_nan=False)+'\n')
            streams['text'].write('ROW_BASIS '+json.dumps(json_native(row_basis_owner),allow_nan=False)+'\n');streams['text'].flush()
            for (split,support,query,labels,old_mask),split_meta in zip(loaded['tasks'],public['splits']):
                sid=split['split_id'];state_by_name={};audit_by_name={};active=dict(split_id=sid,phase='R0_B')
                coords=dict(run_id=run_id,row_id=row_id,split_id=sid,scope=TRAIN_SCOPE,fold=None,trial=None,
                    parent_k=split['k'],train_k=split['k'])
                measurements={}
                def callbacks(state_name):
                    namespace=dict(coords,state=state_name)
                    def save(key,numeric):
                        tick=time.perf_counter()
                        try:
                            measurements[(state_name,key)]=_small_measurements(numeric)
                            for source in ('ggn','curvature'):
                                if source in numeric:
                                    value=np.asarray(numeric[source])
                                    if value.size<=25 and np.isfinite(value).all():
                                        measurements[(state_name,key)]['metric_GGN']=value.tolist()
                            saver=archive.failure if key in ('failure','preparation_failure') else archive
                            return saver(json.dumps(namespace,sort_keys=True,separators=(',',':'))+'/'+key,numeric)
                        finally:callback_seconds['state_archive_seconds']+=time.perf_counter()-tick
                    def log(value):
                        tick=time.perf_counter()
                        try:
                            event=_basis_refs(dict(value),row_basis_owner)
                            ref=event.get('state_ref')
                            observed=dict(measurements.get((state_name,ref.get('key')),{})) if isinstance(ref,Mapping) else {}
                            if str(event.get('event','')).endswith('GRADIENT'):
                                observed=dict(measurements.get((state_name,'initial'),{}),**observed)
                            if str(event.get('event','')).endswith('FINAL'):
                                final=event.get('audit',{});accepted=[v for v in final.get('trials',[]) if v.get('accepted')]
                                source=accepted[0]['state_ref'] if accepted else final.get('initial_state_ref')
                                if isinstance(source,Mapping):observed=dict(measurements.get((state_name,source.get('key')),{}),**observed)
                            event.update(observed,**namespace,executed_core_schema=value.get('schema'),
                                executed_core_method=value.get('method'),schema=SCHEMA,method=METHOD,
                                effective_parameter_rank=basis.rank,nominal_parameter_count=5,
                                row_basis_ref=row_basis_owner['row_basis_ref'],
                                class_statistics_scope='CURRENT_ACTUAL_OOF_FORWARD',
                                class_weights_unavailable_reason=None if 'class_weights' in observed else 'NO_GRADIENT_WEIGHTS_AT_THIS_FORWARD')
                            small=compact_event(event)
                            if 'metric_GGN' in event:small['metric_GGN']=event['metric_GGN']
                            emit('events',event);emit('compact',small)
                            text=training_text(event)
                            streams['text'].write(text+' binding='+json.dumps(namespace,ensure_ascii=False)+'\n')
                            streams['text'].write('SUPPORT_METRIC_JOINT_TRAINING '+json.dumps(json_native(small),
                                ensure_ascii=False,allow_nan=False)+'\n')
                            streams['text'].flush();print(text+' split='+sid,flush=True)
                        finally:callback_seconds['training_event_seconds']+=time.perf_counter()-tick
                    return save,log
                # R0 is independently fit using this split's full original support.
                for name,old_only in (('R0_B',True),('R0_C',False)):
                    if not old_only and not split_meta['new_support_ids']:
                        state_by_name[name]=state_by_name['R0_B'];audit_by_name[name]=audit_by_name['R0_B'];continue
                    selected=support[old_mask] if old_only else support
                    registry=old if old_only else split['registered_classes']
                    ys=np.asarray([old.index(split['registered_classes'][int(y)]) for y in labels[old_mask]],
                        dtype=np.int64) if old_only else labels
                    # LocalRidge intentionally returns the supplied class order.
                    # Preserve each physical class ID while requesting the
                    # canonical columns required by the five-stream protocol.
                    canonical_registry=sorted(registry)
                    canonical_ys=np.asarray([canonical_registry.index(registry[int(y)]) for y in ys],dtype=np.int64)
                    active=dict(split_id=sid,phase=name);tick=time.perf_counter()
                    state=fit_branch_local_ridge(**{k:arrays[k][selected] for k in BRANCHES},support_labels=canonical_ys,
                        support_ids=ids[selected],classes=canonical_registry,old_classes=old,arm='local_ridge')
                    audit=json_native(state.audit_dict());fit=audit['final_fit']
                    check(list(state.classes)==canonical_registry,'R0 fitted class registry mismatch')
                    save,_=callbacks(name);ref=save('final',_baseline_arrays(state))
                    audit.update(status='COMPLETED',final_state_ref=ref,fit_and_archive_seconds=time.perf_counter()-tick,
                        training_physical_ids=sorted(ids[selected].astype(str).tolist()),train_k=split['k'])
                    stage=dict(coords,state=name,mode='INDEPENDENT_R0_REFIT',audit=audit)
                    emit('stages',stage);baseline_audits.append(stage)
                    baseline_totals['completed_head_fits']+=1
                    baseline_totals['factorization_calls']+=fit['factorization_calls']
                    baseline_totals['triangular_solve_calls']+=2*fit['factorization_calls']+fit['effective_degrees_of_freedom_extra_triangular_solves']
                    state_by_name[name]=state;audit_by_name[name]=audit
                for name,mode in (('B_SUPPORT_METRIC','B'),('C_SUPPORT_METRIC_seq','C_seq')):
                    if mode=='C_seq' and not split_meta['new_support_ids']:
                        state_by_name[name]=state_by_name['B_SUPPORT_METRIC'];audit_by_name[name]=audit_by_name['B_SUPPORT_METRIC'];continue
                    selected=support[old_mask] if mode=='B' else support
                    registry=old if mode=='B' else split['registered_classes']
                    ys=np.asarray([old.index(split['registered_classes'][int(y)]) for y in labels[old_mask]],
                        dtype=np.int64) if mode=='B' else labels
                    prep_name='B_prepare' if mode=='B' else 'C_prepare';active=dict(split_id=sid,phase=prep_name)
                    save,log=callbacks(prep_name)
                    prepared=core.prepare_support_metric_joint_training(**{k:arrays[k][selected] for k in BRANCHES},
                        support_labels=ys,support_ids=ids[selected],classes=registry,old_classes=old,
                        inherited=None if mode=='B' else state_by_name['B_SUPPORT_METRIC'],context=dict(coords,stage=mode),
                        prototype_frame=frame,support_metric_basis=basis,**config['support_metric_resources'],log_callback=log,state_callback=save)
                    prep=_basis_refs(json_native(prepared.audit_dict()),row_basis_owner);account('preparation',prep);high['candidate_preparation_count']+=1
                    emit('preparations',dict(coords,state=prep_name,row_basis_ref=row_basis_owner['row_basis_ref'],
                        row_basis_rank=basis.rank,basis_certificate_ref=row_basis_owner['basis_certificate'],audit=prep))
                    active=dict(split_id=sid,phase=name);save,log=callbacks(name)
                    state=core.fit_support_metric_joint_local_ridge(prepared,mode=mode,log_callback=log,state_callback=save)
                    audit=json_native(state.audit_dict())
                    try:_stage_check(state,registry,ids[selected],mode,state_by_name.get('B_SUPPORT_METRIC'),basis,audit)
                    except Exception as state_error:
                        # A returned head can consume work before its state/ref
                        # validation fails; retain that audit as failed evidence.
                        state_error.audit=audit
                        if callable(getattr(state,'to_arrays',None)):
                            try:state_error.arrays=state.to_arrays()
                            except Exception as array_error:
                                state_error.state_arrays_error=str(array_error)
                        raise
                    audit=_basis_refs(audit,row_basis_owner)
                    account('stage',audit)
                    high['candidate_stage_count']+=1;high['final_candidate_head_fit_count']+=1
                    high['optimizer_steps']+=audit['optimizer_steps'];high['trial_count']+=audit['trial_count']
                    high['accepted_trial_count']+=audit['optimizer_steps'];high['rejected_trial_count']+=audit['trial_count']-audit['optimizer_steps']
                    ggn=sum(op['operation']=='support_metric_step' for op in audit['operation_audits'])
                    check(ggn<=1,'One actual GGN direction ceiling required')
                    high['ggn_step_count']+=ggn;high['ggn_parameter_direction_count']+=basis.rank*ggn
                    high['peak_resident_numeric_state_bytes']=max(high['peak_resident_numeric_state_bytes'],audit['resident_numeric_state_bytes'])
                    stage=dict(coords,state=name,mode=mode,row_basis_ref=row_basis_owner['row_basis_ref'],row_basis_rank=basis.rank,
                        basis_certificate_ref=row_basis_owner['basis_certificate'],training_physical_ids=sorted(ids[selected].astype(str).tolist()),audit=audit)
                    emit('stages',stage);stages.append(dict(split_id=sid,state=name,final_state_ref=audit['final_state_ref']))
                    state_by_name[name]=state;audit_by_name[name]=audit
                b,c=state_by_name['B_SUPPORT_METRIC'],state_by_name['C_SUPPORT_METRIC_seq'];reuse=c is b
                r0b,r0c=state_by_name['R0_B'],state_by_name['R0_C'];r0reuse=r0c is r0b
                active=dict(split_id=sid,phase='SINGLE_QUERY_INFERENCE')
                for physical_index in query:
                    active_query=str(ids[physical_index]);active_score_work={}
                    one={k:arrays[k][physical_index:physical_index+1] for k in BRANCHES}
                    values={}
                    tick=time.perf_counter()
                    tensor=torch.tensor([one['z_id'][0].tolist()],dtype=torch.float32,device='cpu')
                    native=head.score(z_id=tensor,feature_contract=head.metadata.feature_contract)
                    check(native.dtype==torch.float32,'Ground A native float32 required')
                    values['A']=np.asarray(native.detach().cpu().tolist(),dtype=np.float32)
                    score_seconds['A']+=time.perf_counter()-tick;score_counts['A']+=1;active_score_work['A_completed']=True
                    for key,state in (('R0_B',r0b),('R0_C',r0c)):
                        if key=='R0_C' and r0reuse:values[key]=values['R0_B'];continue
                        tick=time.perf_counter();values[key]=state.score(**one)
                        score_seconds[key]+=time.perf_counter()-tick;score_counts[key]+=1
                        active_score_work[key+'_completed']=True
                    tick=time.perf_counter();values['B'],bw=b.score_with_audit(**one)
                    account('score',bw);score_seconds['B']+=time.perf_counter()-tick;score_counts['B']+=1;active_score_work['B_work']=bw
                    predict_time=0.
                    if reuse:
                        values['C']=values['B'];cw=None
                        c_prediction=b.classes[int(np.argmax(values['B'][0]))]
                        decision=dict(schema=DECISION_SCHEMA,policy='EXACT_ACTUAL_B_REUSE',
                            new0_reuses_actual_B=True,prediction=c_prediction)
                    else:
                        tick=time.perf_counter();values['C'],cw,decision=_structured_score(c,one)
                        account('score',cw);score_seconds['C']+=time.perf_counter()-tick;score_counts['C']+=1
                        active_score_work.update(C_work=cw,C_reused_B_scores=False,C_structural_decision=decision,
                            C_public_predict_attempted=True,C_public_predict_completed=False,
                            C_public_predict_internal_work=None)
                        structural_attempts+=1;tick=time.perf_counter()
                        try:actual=core.predict_support_metric_joint_local_ridge(c,**one)
                        finally:
                            predict_time=time.perf_counter()-tick;structural_attempt_seconds+=predict_time
                            active_score_work['C_public_predict_seconds']=predict_time
                        check(len(actual)==1 and str(actual[0])==decision['prediction'],
                            'Public C predict differs from recorded structural decision')
                        active_score_work['C_public_predict_completed']=True
                        structural_calls+=1;structural_seconds+=predict_time;c_prediction=str(actual[0])
                    active_score_work.update(C_work=cw,C_reused_B_scores=reuse,C_structural_decision=decision)
                    emit('work',dict(split_id=sid,query_id=active_query,B_work=bw,C_work=cw,
                        R0_score_work=None,R0_work_unavailable_reason='ORIGINAL_SCORE_API_HAS_NO_OPERATION_LEDGER',
                        C_reused_B_scores=reuse,R0_C_reused_B_scores=r0reuse,
                        C_structural_decision=decision,C_structural_prediction=c_prediction,
                        C_public_predict_calls=0 if reuse else 1,C_public_predict_seconds=predict_time,
                        C_public_predict_internal_work=None,C_public_predict_internal_work_unavailable_reason=
                        'EXACT_B_REUSE_NO_EXTRA_CALL' if reuse else 'PUBLIC_PREDICT_HAS_NO_WORK_AUDIT',
                        scope='ACTUAL_SINGLETON_SCORE_AND_SEPARATE_PUBLIC_PREDICT; NO_LABEL_OR_ROLE'))
                    for key,state in (('A',head),('B',b),('C',c),('R0_B',r0b),('R0_C',r0c)):
                        pred=(head.classes[int(np.argmax(values[key][0]))] if key=='A' else c_prediction if key=='C' else None)
                        record=_prediction(sid,active_query,state.classes,values[key],pred);emit(key,record)
                        if key=='C':emit('alias',record)
                    active_query=None;active_score_work=None
                item=dict(split_meta,row_basis_ref=row_basis_owner['row_basis_ref'],row_basis_rank=basis.rank,
                    basis_certificate_ref=row_basis_owner['basis_certificate'],b_state_ref=audit_by_name['B_SUPPORT_METRIC']['final_state_ref'],
                    c_state_ref=audit_by_name['C_SUPPORT_METRIC_seq']['final_state_ref'],
                    c_inherited_from_b_state_ref=audit_by_name['B_SUPPORT_METRIC']['final_state_ref'],c_reuses_b=reuse,
                    r0_b_state_ref=audit_by_name['R0_B']['final_state_ref'],r0_c_state_ref=audit_by_name['R0_C']['final_state_ref'],
                    r0_c_reuses_b=r0reuse,B_classes=list(b.classes),C_classes=list(c.classes),
                    R0_B_classes=list(r0b.classes),R0_C_classes=list(r0c.classes),
                    support_only_fit=True,query_fit_access=False,source_fit_access=False)
                emit('trace',item);completed.append(item)
        for name in ('training_events_compact','fit_stages','preparations'):
            _csv_from_jsonl(out/(name+'.jsonl'))
        manifest=archive.finalize('COMPLETE')
        streams={key:dict(path='predictions_'+key+'.jsonl',record_count=public['query_record_count'],
            file_bytes=(out/('predictions_'+key+'.jsonl')).stat().st_size) for key in STREAM_NAMES}
        marker=dict(public,status=STATUS,pid=os.getpid(),splits=completed,streams=streams,**high,
            actual_work=totals,row_basis_actual_work=work['row_basis'],row_basis_ref=row_basis_owner['row_basis_ref'],
            row_basis_rank=basis.rank,row_basis_audit=row_basis_owner,basis_certificate=row_basis_owner['basis_certificate'],
            row_basis='row_basis.json',row_basis_construction_status='COMPLETE',
            preparation_actual_work=work['preparation'],stage_actual_work=work['stage'],
            score_actual_work=work['score'],actual_work_aggregation=WORK_AGGREGATION,work_aggregation='SUM/MAX',
            query_score_work=dict(path='query_score_work.jsonl',record_count=public['query_record_count'],
                file_bytes=(out/'query_score_work.jsonl').stat().st_size),
            compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),
            completed_split_count=len(completed),actual_stage_count=len(stages),
            actual_baseline_stage_count=len(baseline_audits),state_manifest='state_manifest.json',
            resources=dict(actual_core_work=totals,work_aggregation='SUM/MAX',counter_scope='COMPLETED_ROW_BASIS_PREP_STAGE_SCORE',
                original_R0_training_audits=baseline_audits,original_R0_completed_training_counters=baseline_totals,
                R0_count_scope='SUCCESSFULLY_RETURNED_ORIGINAL_BASELINE; FAILED_ORIGINAL_WORK_NOT_INFERRED',
                query_score_calls=score_counts,query_score_seconds=score_seconds,C_public_predict_calls=structural_calls,
                C_public_predict_seconds=structural_seconds,C_public_predict_internal_work=None,
                C_public_predict_internal_work_unavailable_reason='PUBLIC_PREDICT_HAS_NO_WORK_AUDIT',
                query_score_call_semantics=dict(A='native_float32_score',B='actual_B_score_with_audit',
                    C='exact_core_single_record_score_path_plus_public_predict',R0_B='original_full_support_local_ridge',
                    R0_C='independent_original_full_registered_support_local_ridge_or_exact_new0_B_reuse'),
                callback_seconds=callback_seconds,state_archive_seconds=manifest['archive_seconds'],
                basis_certificate_archive_seconds=row_basis_owner['basis_certificate']['archive_seconds'],
                archive_seconds_scope='NPZ_AND_CERTIFICATE_ACTUAL_NESTED_ARCHIVE_TIMES_NOT_ADDITIVE_TO_FIT_WALL',
                callback_seconds_scope='ACTUAL_NESTED_CALLBACKS; INCLUDED_IN_FIT_WALL_DO_NOT_ADD',
                wall_seconds=time.perf_counter()-started,state_archive_file_bytes=manifest['total_file_bytes'],
                state_archive_numeric_bytes=manifest['numeric_array_bytes'],peak_process_rss_bytes=peak_process_rss(),
                peak_gpu_memory_bytes=None,incremental_transfer_bytes=None,native_wire_bytes=None,energy=None,
                hardware=dict(device='cpu',platform=platform.platform(),machine=platform.machine(),
                    processor=platform.processor(),cpu_count=os.cpu_count(),torch_version=str(torch.__version__)),
                peak_process_rss_scope='HOST_PROCESS_LIFETIME_HIGH_WATER; NOT_SATELLITE_MEASUREMENT',
                ground_packet_file_bytes=public['ground_packet_identity']['packet_total_file_bytes'],
                ground_weight_file_bytes=public['ground_packet_identity']['head_weight_file_bytes'],
                ground_weight_numeric_bytes=public['ground_packet_identity']['head_weight_numeric_bytes'],
                ground_geometry_payload_audit=public['ground_geometry_binding']['ground_payload_audit'],
                newly_generated_ground_statistics_bytes=0),
            state_namespace_scope=TRAIN_SCOPE,source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN')
        write(out/'predictions_complete.json',marker)
        return json_native(marker)
    except Exception as exc:
        reporting_failures=[]
        def report_step(name,callback):
            try:return callback()
            except Exception as report_exc:
                reporting_failures.append(dict(step=name,error_type=type(report_exc).__name__,error=str(report_exc)))
                return None
        details=report_step('failed_work_audit',lambda:getattr(exc,'support_metric_query_score_audit',None) or (
            exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else getattr(exc,'audit',None)))
        failed_ref=details.get('failure_state_ref') if isinstance(details,Mapping) else None
        numeric=getattr(exc,'arrays',None)
        if failed_ref is None and isinstance(numeric,Mapping) and numeric:
            namespace=dict(run_id=run_id,row_id=row_id,split_id=(active or {}).get('split_id'),
                scope=TRAIN_SCOPE,fold=None,trial=None,state=(active or {}).get('phase'))
            failed_ref=report_step('failed_numeric_archive',lambda:archive.failure(json.dumps(namespace,sort_keys=True)+'/failure',numeric))
        if not (out/'state_manifest.json').exists():
            report_step('state_manifest_finalize',lambda:archive.finalize('TECHNICAL_FAILURE'))
        for name in ('training_events_compact','fit_stages','preparations'):
            if (out/(name+'.jsonl')).exists() and not (out/(name+'.csv')).exists():
                report_step(name+'_csv',lambda stem=name:_csv_from_jsonl(out/(stem+'.jsonl')))
        failure=dict(schema=SCHEMA,method=METHOD,status='TECHNICAL_FAILURE',run_id=run_id,row_id=row_id,
            release_commit=release_commit,support_metric_resources=deepcopy(config['support_metric_resources']),
            algorithm=deepcopy(config['algorithm']),active=active,error_type=type(exc).__name__,error=str(exc),
            original_cause=None if exc.__cause__ is None else dict(error_type=type(exc.__cause__).__name__,error=str(exc.__cause__)),
            completed_splits=completed,completed_stages=stages,completed_baseline_audits=baseline_audits,
            completed_actual_work=totals,completed_row_basis_actual_work=work['row_basis'],
            row_basis_audit=row_basis_owner,row_basis_ref=None if row_basis_owner is None else row_basis_owner['row_basis_ref'],
            row_basis_rank=None if basis is None else basis.rank,
            completed_preparation_actual_work=work['preparation'],
            completed_stage_actual_work=work['stage'],completed_score_actual_work=work['score'],
            completed_original_R0_counters=baseline_totals,
            failed_work_audit=details,failed_state_ref=failed_ref,
            failed_work_unknown=not isinstance(details,Mapping) or not isinstance(details.get('actual_work'),Mapping),
            work_scope='COMPLETED_RETURNED_ROW_BASIS_PREP_STAGE_SCORE; FAILED_WORK_SEPARATE_OR_UNKNOWN',
            active_query_id=active_query,active_completed_query_work=active_score_work,
            completed_query_score_calls=score_counts,completed_query_score_seconds=score_seconds,
            completed_C_public_predict_calls=structural_calls,completed_C_public_predict_seconds=structural_seconds,
            attempted_C_public_predict_calls=structural_attempts,attempted_C_public_predict_seconds=structural_attempt_seconds,
            callback_seconds=callback_seconds,query_work_scope='RETURNED_CALLS_ONLY; FAILING_UNMEASURED_CALL_WORK_UNKNOWN',
            query_fit_access=False,source_fit_access=False,truth_read=False,automatic_retry=False,reporting_failures=reporting_failures)
        report_step('technical_failure_write',lambda:write(out/'technical_failure.json',failure))
        if reporting_failures:
            report_step('reporting_failure_stderr',lambda:print('TECHNICAL_FAILURE evidence: '+json.dumps(json_native(failure),
                ensure_ascii=False,allow_nan=False),file=sys.stderr,flush=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run-id','row-id','release-commit','row-root','capsule','output','config','expected-capsule-id',
                'expected-checkpoint-sha256','branch-features','ground-packet','ground-summary'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--ground-summary-already-deployed',choices=('true','false'),required=True)
    parser.add_argument('--preflight-only',action='store_true');args=vars(parser.parse_args())
    only=args.pop('preflight_only');args['config']=read(args['config'])
    args['ground_summary_already_deployed']=args['ground_summary_already_deployed']=='true'
    result=preflight(**args) if only else predict(**args)
    print(json.dumps(json_native(result),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
