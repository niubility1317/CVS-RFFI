"""Frozen per-split Margin B -> C and native Ground A query predictions only."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import csv
import json
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
from cvsrffi import d92_margin_joint_local_ridge as core
from cvsrffi.d92_ground_classifier_a import GroundFeatureContract
from export_d92_ground_classifier_a_packet import load_packet
from export_d92_branch_features import (CACHE_SCHEMA,FEATURE_CONTRACT,local_core as cache_core,
    load_features,validate_capsule,read,peak_process_rss)
from d92_orbit_feature_cache import validate_split
from evaluate_d92_margin_joint_probe import StateArchive,json_native

SCHEMA='d92_margin_joint_query_predictions_v1'
METHOD='D92-MarginJointLocalRidge-v1'
STATUS='COMPLETE'
TRAIN_SCOPE='query_benchmark_support_training'
BRANCHES=('z_id','fft','t_emb','f_emb','pa_local')
PEAKS=('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes')


def check(condition,message):
    if not condition:raise ValueError(message)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(json_native(value),stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def validate_config(config):
    check(type(config) is dict and set(config)=={'algorithm','qp_resources'}
        and config['algorithm']==core.FROZEN_CONFIG,'Frozen Margin benchmark algorithm/config mismatch')
    limits=config['qp_resources']
    check(type(limits) is dict and set(limits)=={'max_transitions','max_factor_buffer_bytes'}
        and all(type(v) is int and v>0 for v in limits.values()),'Explicit positive integer QP resources required')


def _inputs(*,run_id,row_id,release_commit,row_root,capsule,output,config,
            expected_capsule_id,expected_checkpoint_sha256,branch_features,ground_packet):
    validate_config(config)
    check(all(type(v) is str and v for v in (run_id,row_id)),'Explicit benchmark run/row required')
    check(type(release_commit) is str and re.fullmatch('[0-9a-f]{40}',release_commit),'Explicit actual release commit required')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    inputs=[Path(p).resolve() for p in (row_root,capsule,branch_features,ground_packet)]
    check(all(out!=p and not out.is_relative_to(p) for p in inputs),'Output overlaps original input directory')
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
    check(packet_seed is None or packet_seed==previous['seed'],'Ground packet/cache model seed mismatch')
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
    public=dict(schema=SCHEMA,method=METHOD,run_id=run_id,row_id=row_id,release_commit=release_commit,
        capsule_id=expected_capsule_id,checkpoint_sha256=expected_checkpoint_sha256,model_seed=previous['seed'],
        algorithm=deepcopy(config['algorithm']),qp_resources=deepcopy(config['qp_resources']),
        run_binding=dict(prediction_output_root=str(out),row_root=str(inputs[0]),capsule=str(inputs[1]),
            branch_features=str(inputs[2]),ground_packet=str(inputs[3])),
        source_identity=source_identity,ground_packet_identity=ground_identity,ordered_ground_classes=list(head.classes),
        old_classes=sorted(old),splits=metadata,split_count=len(splits),
        query_record_count=sum(v['query_count'] for v in metadata),query_fit_access=False,source_fit_access=False,truth_read=False,
        checkpoint_loaded=False,encoder_called=False,cross_split_adapted_state_reuse=False,
        source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',
        A_tie_policy='first_original_native_head_column',B_C_tie_policy='physical_class_id_ascending',
        fit_scope='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY',query_inference_scope='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS',
        technical_query_chunk_size=1,
        cache_resources=dict(cache_load_seconds=cache_seconds,ground_packet_load_seconds=packet_seconds,
            feature_array_bytes=producer['feature_array_bytes'],feature_file_bytes=producer['feature_file_bytes'],
            existing_checkpoint_file_bytes=producer['model_file_bytes'],encoder_calls_this_prediction=0))
    return dict(public=public,arrays=arrays,ids=ids,old=old,head=head,tasks=tasks)


def preflight(*,run_id,row_id,release_commit,row_root,capsule,output,config,
              expected_capsule_id,expected_checkpoint_sha256,branch_features,ground_packet):
    """Existing immutable input permission/binding check; no fit or inference."""
    return dict(_inputs(**locals())['public'],status='MARGIN_QUERY_PREFLIGHT_COMPLETE')


def _csv_from_jsonl(path):
    keys=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:keys.update(json.loads(line))
    with path.with_suffix('.csv').open('x',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=sorted(keys));writer.writeheader()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);writer.writerow({k:'N/A' if row.get(k) is None else row[k] for k in sorted(keys)})


def _account(totals,audit,keys):
    for key in keys:
        value=audit.get(key)
        check(value is None or type(value) is int and value>=0,'Invalid measured work: '+key)
        if value is None or totals[key] is None:totals[key]=None
        elif key in PEAKS:totals[key]=max(totals[key],value)
        else:totals[key]+=value


def _prediction(split_id,query_id,classes,scores):
    values=np.asarray(scores)
    check(values.shape==(1,len(classes)) and np.isfinite(values).all(),'Invalid singleton all-class prediction')
    column=int(np.argmax(values[0]))
    return dict(split_id=split_id,query_id=str(query_id),classes=list(classes),scores=values[0].tolist(),prediction=classes[column])


def predict(*,run_id,row_id,release_commit,row_root,capsule,output,config,
            expected_capsule_id,expected_checkpoint_sha256,branch_features,ground_packet):
    kwargs=locals().copy();started=time.perf_counter();out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    # Validate explicit controls and readonly inputs before creating evidence.
    validate_config(config)
    check(all(type(v) is str and v for v in (run_id,row_id)),'Explicit run/row required')
    check(type(release_commit) is str and re.fullmatch('[0-9a-f]{40}',release_commit),'Actual release commit required')
    check(all(not out.is_relative_to(Path(p).resolve()) for p in (row_root,capsule,branch_features,ground_packet)),
        'Output overlaps original inputs')
    # _inputs requires an absent output. Finish readonly loading before exclusive
    # creation; a preflight/input failure cannot masquerade as a started fit.
    loaded=_inputs(**kwargs);out.mkdir(parents=True,exist_ok=False)
    public=loaded['public'];arrays,ids,old,head=(loaded[k] for k in ('arrays','ids','old','head'))
    archive=StateArchive(out);completed=[];stages=[];active=None;active_query=None;active_score_work=None
    totals=dict.fromkeys(tuple(core.AUDIT_COUNTERS)+PEAKS,0)
    score_counts=dict(A=0,B=0,C=0);score_seconds=dict(A=0.,B=0.,C=0.)
    write(out/'startup.json',dict(public,config=config,argv=sys.argv,python=sys.executable,pid=os.getpid(),
        stage_namespace_scope=TRAIN_SCOPE,device='cpu',fixed_method_no_selection=True))
    try:
        with ExitStack() as stack:
            streams={key:stack.enter_context((out/name).open('x',encoding='utf-8')) for key,name in
                dict(A='predictions_A.jsonl',B='predictions_B.jsonl',C='predictions_C.jsonl',alias='predictions.jsonl',
                    events='training_events.jsonl',compact='training_events_compact.jsonl',stages='fit_stages.jsonl',
                    preparations='preparations.jsonl',trace='fit_trace.jsonl',work='query_score_work.jsonl',text='training.log').items()}
            def emit(key,value):
                streams[key].write(json.dumps(json_native(value),ensure_ascii=False,allow_nan=False)+'\n');streams[key].flush()
            for (split,support,query,labels,old_mask),split_meta in zip(loaded['tasks'],public['splits']):
                sid=split['split_id'];active=dict(split_id=sid,phase='B_prepare');state_by_name={};audit_by_name={}
                coords=dict(run_id=run_id,row_id=row_id,split_id=sid,scope=TRAIN_SCOPE,fold=None,trial=None)
                def callbacks(state_name):
                    namespace=dict(coords,state=state_name)
                    def save(key,numeric):return archive(json.dumps(namespace,sort_keys=True,separators=(',',':'))+'/'+key,numeric)
                    def log(value):
                        event=dict(value,**namespace,schema=SCHEMA,method=METHOD)
                        emit('events',event)
                        small=core.compact_training_record(event);small.pop('text',None)
                        owner=value.get('audit',value.get('objective',value))
                        for peak in PEAKS:
                            if peak in owner:small[peak]=owner[peak]
                        small=json_native(small)
                        small.update(state=state_name,schema=SCHEMA,method=METHOD)
                        check(all(v is None or type(v) in (str,bool,int,float) for v in small.values()),'Nonscalar compact training event')
                        emit('compact',small)
                        text=value.get('text','MARGIN_JOINT '+str(value.get('event')))
                        streams['text'].write(str(text)+' binding='+json.dumps(namespace,ensure_ascii=False)+'\n');streams['text'].flush()
                        print(str(text)+' split='+sid,flush=True)
                    return save,log
                for name,mode in (('B_MARGIN','B'),('C_MARGIN_seq','C_seq')):
                    if mode=='C_seq' and not split_meta['new_support_ids']:
                        state_by_name[name]=state_by_name['B_MARGIN'];audit_by_name[name]=audit_by_name['B_MARGIN'];continue
                    selected=support[old_mask] if mode=='B' else support
                    registry=old if mode=='B' else split['registered_classes']
                    ys=np.asarray([old.index(split['registered_classes'][int(y)]) for y in labels[old_mask]],dtype=np.int64) if mode=='B' else labels
                    prep_name='B_prepare' if mode=='B' else 'C_prepare';active=dict(split_id=sid,phase=prep_name)
                    save,log=callbacks(prep_name)
                    prepared=core.prepare_margin_joint_training(**{k:arrays[k][selected] for k in BRANCHES},
                        support_labels=ys,support_ids=ids[selected],classes=registry,old_classes=old,
                        inherited=None if mode=='B' else state_by_name['B_MARGIN'],context=dict(coords,stage=mode),
                        **config['qp_resources'],log_callback=log,state_callback=save)
                    prep_audit=json_native(prepared.audit_dict());_account(totals,prep_audit,core.PREPARATION_COUNTERS)
                    emit('preparations',dict(coords,state=prep_name,audit=prep_audit))
                    active=dict(split_id=sid,phase=name);save,log=callbacks(name)
                    state=core.fit_margin_joint_local_ridge(prepared,mode=mode,log_callback=log,state_callback=save)
                    audit=json_native(state.audit_dict())
                    check(list(state.classes)==sorted(registry) and sorted(state.ids)==sorted(ids[selected].astype(str).tolist()),
                        'Actual fitted support/classes mismatch')
                    check(audit.get('status')=='COMPLETED' and audit.get('final_state_ref') is not None,'Missing completed actual state')
                    if mode=='C_seq':check(audit['preparation']['final_problem']['prior_ref']==audit_by_name['B_MARGIN']['final_state_ref'],
                        'C did not inherit the same-split actual B')
                    _account(totals,audit,core.STAGE_COUNTERS);_account(totals,audit,PEAKS)
                    stage=dict(coords,state=name,mode=mode,training_physical_ids=sorted(ids[selected].astype(str).tolist()),audit=audit)
                    emit('stages',stage);stages.append(dict(split_id=sid,state=name,final_state_ref=audit['final_state_ref']))
                    state_by_name[name]=state;audit_by_name[name]=audit
                b=state_by_name['B_MARGIN'];c=state_by_name['C_MARGIN_seq'];reuse=c is b
                active=dict(split_id=sid,phase='SINGLE_QUERY_INFERENCE')
                for physical_index in query:
                    active_query=str(ids[physical_index]);active_score_work={}
                    one={k:arrays[k][physical_index:physical_index+1] for k in BRANCHES}
                    tick=time.perf_counter()
                    tensor=torch.tensor([one['z_id'][0].tolist()],dtype=torch.float32,device='cpu')
                    a_scores=head.score(z_id=tensor,feature_contract=head.metadata.feature_contract)
                    check(a_scores.dtype==torch.float32,'Ground A native float32 required')
                    a_values=np.asarray(a_scores.cpu().tolist(),dtype=np.float32)
                    score_seconds['A']+=time.perf_counter()-tick;score_counts['A']+=1
                    active_score_work['A_completed']=True
                    tick=time.perf_counter();b_values,b_work=b.score_with_audit(**one)
                    score_seconds['B']+=time.perf_counter()-tick;score_counts['B']+=1
                    active_score_work['B_work']=b_work
                    if reuse:c_values=b_values;c_work=b_work
                    else:
                        tick=time.perf_counter();c_values,c_work=c.score_with_audit(**one)
                        score_seconds['C']+=time.perf_counter()-tick;score_counts['C']+=1
                    active_score_work.update(C_work=None if reuse else c_work,C_reused_B_scores=reuse)
                    emit('work',dict(split_id=sid,query_id=str(ids[physical_index]),B_work=b_work,
                        C_work=None if reuse else c_work,C_reused_B_scores=reuse,
                        scope='ACTUAL_SINGLETON_SCORE_AUDIT; C_REUSE_HAS_NO_EXTRA_SCORE_CALL'))
                    for key,classes,values in (('A',head.classes,a_values),('B',b.classes,b_values),('C',c.classes,c_values)):
                        record=_prediction(sid,ids[physical_index],classes,values);emit(key,record)
                        if key=='C':emit('alias',record)
                    active_query=None;active_score_work=None
                item=dict(split_meta,b_state_ref=audit_by_name['B_MARGIN']['final_state_ref'],
                    c_state_ref=audit_by_name['C_MARGIN_seq']['final_state_ref'],
                    c_inherited_from_b_state_ref=audit_by_name['B_MARGIN']['final_state_ref'],c_reuses_b=reuse,
                    B_classes=list(b.classes),C_classes=list(c.classes),support_only_fit=True,
                    query_fit_access=False,source_fit_access=False)
                emit('trace',item);completed.append(item)
        _csv_from_jsonl(out/'training_events_compact.jsonl')
        manifest=archive.finalize('COMPLETE')
        streams={key:dict(path='predictions_'+key+'.jsonl',record_count=public['query_record_count'],
            file_bytes=(out/('predictions_'+key+'.jsonl')).stat().st_size) for key in ('A','B','C')}
        marker=dict(public,status=STATUS,splits=completed,streams=streams,
            compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),
            completed_split_count=len(completed),actual_stage_count=len(stages),state_manifest='state_manifest.json',
            resources=dict(actual_training_counters=totals,work_aggregation='SUM',peak_aggregation='MAX',
                counter_scope='COMPLETED_PREPARATIONS_AND_COMPLETED_STAGE_AUDITS_ONLY',
                query_score_calls=score_counts,query_score_seconds=score_seconds,wall_seconds=time.perf_counter()-started,
                state_archive_file_bytes=manifest['total_file_bytes'],state_archive_numeric_bytes=manifest['numeric_array_bytes'],
                peak_process_rss_bytes=peak_process_rss(),peak_gpu_memory_bytes=None,incremental_transfer_bytes=None,energy=None,
                hardware=dict(device='cpu',platform=platform.platform(),machine=platform.machine(),torch_version=str(torch.__version__)),
                peak_process_rss_scope='Linux current CPU process lifetime high-water RSS; not GPU or satellite deployment measurement',
                ground_packet_file_bytes=public['ground_packet_identity']['packet_total_file_bytes'],
                ground_weight_file_bytes=public['ground_packet_identity']['head_weight_file_bytes'],
                ground_weight_numeric_bytes=public['ground_packet_identity']['head_weight_numeric_bytes']),
            state_namespace_scope=TRAIN_SCOPE,source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN')
        write(out/'predictions_complete.json',marker)
        return json_native(marker)
    except Exception as exc:
        details=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else getattr(exc,'audit',None)
        failed_ref=None;numeric=getattr(exc,'arrays',None)
        if numeric:
            namespace=dict(run_id=run_id,row_id=row_id,split_id=(active or {}).get('split_id'),scope=TRAIN_SCOPE,
                fold=None,trial=None,state=(active or {}).get('phase'))
            failed_ref=archive.failure(json.dumps(namespace,sort_keys=True)+'/failure',numeric)
        if not (out/'state_manifest.json').exists():archive.finalize('TECHNICAL_FAILURE')
        write(out/'technical_failure.json',dict(schema=SCHEMA,method=METHOD,status='TECHNICAL_FAILURE',
            run_id=run_id,row_id=row_id,release_commit=release_commit,active=active,
            error_type=type(exc).__name__,error=str(exc),completed_splits=completed,completed_stages=stages,
            completed_work_counters=totals,work_scope='COMPLETED_PREPARATIONS_AND_STAGES; FAILED_WORK_SEPARATE_OR_UNKNOWN',
            failed_work_audit=details,failed_state_ref=failed_ref,failed_work_unknown=details is None,
            active_query_id=active_query,active_completed_query_work=active_score_work,
            completed_query_score_calls=score_counts,completed_query_score_seconds=score_seconds,
            query_work_scope='RETURNED_CALLS_ONLY; ANY_FAILING_CALL_WORK_NOT_MEASURED_IS_UNKNOWN',
            query_fit_access=False,source_fit_access=False,truth_read=False,automatic_retry=False))
        if (out/'training_events_compact.jsonl').exists() and not (out/'training_events_compact.csv').exists():_csv_from_jsonl(out/'training_events_compact.jsonl')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run-id','row-id','release-commit','row-root','capsule','output','config','expected-capsule-id',
                'expected-checkpoint-sha256','branch-features','ground-packet'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=vars(parser.parse_args())
    only=args.pop('preflight_only');args['config']=read(args['config'])
    result=preflight(**args) if only else predict(**args)
    print(json.dumps(json_native(result),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
