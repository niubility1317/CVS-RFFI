"""Truth-last scoring of all frozen MarginJoint A/B/C query predictions.

This module never fits, imports a model, loads a ground packet or opens features.
The only NPZ member read is the capsule's opaque physical ID registry.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import re
import time

import numpy as np

METHOD='D92-MarginJointLocalRidge-v1'
PREDICTION_SCHEMA='d92_margin_joint_query_predictions_v1'
SCHEMA='d92_margin_joint_query_score_v1'
STATUS='MARGIN_JOINT_QUERY_SCORE_COMPLETE'
SCOPE='FROZEN_QUERY_BENCHMARK_TRUTH_LAST_THREE_STAGE_PAIRED'
KS=(1,5,10,20)
NEWS=(0,2,5,10,20)
METRICS=('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','C_h',
    'adaptation_gain_B_minus_A','total_old_accuracy_drop','C_abs_new_old_gap')
IDENTITY=('run_id','row_id','capsule_id','checkpoint_sha256','model_seed','release_commit')
SPLIT_IDENTITY=('split_id','receiver','scenario','k','new_count','support_seed')
STREAMS={s:'predictions_'+s+'.jsonl' for s in ('A','B','C')}
PEAK_COUNTERS=frozenset(('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes'))
RAW_FEATURE_CONTRACT=dict(input_shape=[2,256],view='original_received_observation',view_count=1,
    identity_feature_key='feat_joint',branch_keys=['z_id','fft','t_emb','f_emb','pa_local'],branch_dim=160,fft_dim=96,
    fft='historical_spectral_logmag_sketch',fft_norm_floor=1e-8,cache_dtype='float32',
    normalization='raw_native_aux_no_additional_branch_normalization')


def require(condition,message):
    if not condition:raise ValueError(message)


def _constant(value):
    raise ValueError('Nonfinite JSON number: '+value)


def _object(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate JSON key: '+key);result[key]=value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=_constant,object_pairs_hook=_object)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def strings(values,empty=False):
    return isinstance(values,list) and (empty or bool(values)) and all(type(v) is str and v for v in values) and len(set(values))==len(values)


def _integer(value,minimum=0):
    return type(value) is int and value>=minimum


def validate_config(config):
    require(isinstance(config,dict) and set(config)=={'algorithm','qp_resources'},'Wrong fixed benchmark config')
    require(config['algorithm'].get('schema')=='d92_margin_joint_local_ridge_v1'
        and config['algorithm'].get('method')==METHOD,'Wrong frozen method identity')
    limits=config['qp_resources']
    require(isinstance(limits,dict) and set(limits)=={'max_transitions','max_factor_buffer_bytes'}
        and all(_integer(v,1) for v in limits.values()),'Explicit QP resource limits required')


def _indices(value,count):
    return isinstance(value,list) and bool(value) and all(_integer(i) and i<count for i in value) and len(set(value))==len(value)


def load_capsule_metadata(capsule,expected_capsule_id):
    """Never read IQ or query labels: only manifest, splits and received.ids."""
    root=Path(capsule);manifest=read(root/'manifest.json')
    require(manifest.get('protocol_schema')=='p2_min_v1' and manifest.get('phase2_data_status')=='VALIDATED_ONCE'
        and manifest.get('capsule_id')==expected_capsule_id,'Capsule identity/protocol mismatch')
    with np.load(root/'received.npz',allow_pickle=False) as arrays:
        ids=arrays['ids']
        require(ids.ndim==1 and ids.dtype.kind in 'SU','Opaque capsule IDs must be text')
        physical=ids.astype(str).tolist()
    require(strings(physical),'Repeated or empty opaque capsule IDs')
    splits={};cells=set();old=None
    for path in sorted((root/'splits').glob('*.json')):
        s=read(path);sid=s['split_id'];classes=s['registered_classes']
        require(type(sid) is str and sid and sid not in splits and strings(classes),'Invalid capsule split registry')
        require(path.stem==sid and s.get('protocol_schema')=='p2_min_v1' and s.get('phase2_data_status')=='VALIDATED_ONCE'
            and s.get('capsule_id')==expected_capsule_id
            and not {'query_labels','query_truth','query_roles','query_class_counts'}&set(s),'Forbidden or mismatched split protocol metadata')
        require(_integer(s['k'],1) and s['k'] in KS and _integer(s['support_seed'])
            and len(classes)-6 in NEWS,'Wrong K/new/support seed axes')
        declared_old=classes[:6]
        if old is None:old=declared_old
        require(len(declared_old)==6 and declared_old==old,'Six-old class registry changed')
        require(all(type(s[k]) is str and s[k] for k in ('receiver','scenario')),'Invalid receiver/scenario')
        cell=(s['receiver'],s['scenario'],s['support_seed'],s['k'],len(classes)-6)
        require(cell not in cells,'Duplicate capsule matrix cell');cells.add(cell)
        si,qi=s['support_indices'],s['query_indices']
        require(_indices(si,len(physical)) and _indices(qi,len(physical)) and not set(si)&set(qi),'Invalid or overlapping support/query IDs')
        labels=s['support_labels']
        require(isinstance(labels,list) and len(si)==len(labels)==s['k']*len(classes)
            and all(_integer(y) and y<len(classes) for y in labels)
            and all(labels.count(j)==s['k'] for j in range(len(classes))),'Physical support K mismatch')
        support=[physical[i] for i in si];queries=[physical[i] for i in qi]
        if 'support_ids' in s:require(s['support_ids']==support,'Capsule support index/ID mismatch')
        old_support=[pid for pid,y in zip(support,labels) if classes[y] in old]
        new_support=[pid for pid,y in zip(support,labels) if classes[y] not in old]
        splits[sid]=dict(**{k:s[k] for k in ('split_id','receiver','scenario','k','support_seed')},
            new_count=len(classes)-6,declared_registered_classes=classes,registered_classes=sorted(classes),
            old_classes=sorted(old),support_ids=sorted(support),old_support_ids=sorted(old_support),new_support_ids=sorted(new_support),
            query_ids=queries,query_count=len(queries))
    pairs={(rx,scene,seed) for rx,scene,seed,_,_ in cells}
    expected={(rx,scene,seed,k,n) for rx,scene,seed in pairs for k in KS for n in NEWS}
    require(bool(splits) and cells==expected and manifest.get('split_count')==len(splits),'Incomplete capsule K/new matrix')
    return manifest,splits


def _source_identity(startup,complete,expected):
    source=complete['source_identity'];ground=complete['ground_packet_identity']
    require(startup['source_identity']==source and startup['ground_packet_identity']==ground,'Source identity changed between startup and completion')
    for item in (source,ground):
        require(item['checkpoint_sha256']==expected['checkpoint_sha256'] and item['model_seed']==expected['model_seed']
            and item['source_only_verdict']=='MATCHED_SOURCE_ONLY_SCRATCH','Source-only checkpoint identity mismatch')
    require(source['source_role_comparison']=='EXACT_MATCH' and type(source['checkpoint_epoch']) is int
        and source['checkpoint_epoch']==200 and source['checkpoint_inheritance']==[]
        and source['target_access_before_freeze'] is False,'Source provenance boundary mismatch')
    require(ground['ordered_classes']==complete['ordered_ground_classes'] and ground['feature_contract']==dict(
        checkpoint_sha256=expected['checkpoint_sha256'],cache_key='z_id',source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160),
        'Ground/native feature or class mapping changed')
    require(Path(ground['path']).resolve()==Path(complete['run_binding']['ground_packet']).resolve(),
        'Ground packet path differs from the explicit current row source')
    require(source['cache_schema']=='d92_branch_received_features_v1' and source['feature_contract']==RAW_FEATURE_CONTRACT,
        'Wrong current five-branch raw feature source contract')
    require(type(ground['scale']) in (int,float) and not isinstance(ground['scale'],bool)
        and math.isfinite(ground['scale']) and ground['scale']>0
        and type(ground['norm_eps']) in (int,float) and not isinstance(ground['norm_eps'],bool)
        and math.isfinite(ground['norm_eps']) and ground['norm_eps']>0
        and _integer(ground['packet_total_file_bytes'],1),'Ground packet declared scale/bytes invalid')


def _state_ref(ref,manifest,expected,state):
    require(isinstance(ref,dict) and ref.get('path') in manifest,'Head ref is not in current archive manifest')
    saved=manifest[ref['path']]
    require(set(ref)==set(saved) and all(ref[k]==saved[k] for k in saved if k!='arrays')
        and set(ref['arrays'])==set(saved['arrays']),'Head reference identity/inventory mismatch')
    for name,meta in saved['arrays'].items():
        require(ref['arrays'][name]==meta or ref['arrays'][name]=={k:meta[k] for k in ('shape','dtype','nbytes')},
            'Head reference numeric metadata mismatch')
        require(meta.get('all_finite') is True and type(meta.get('nonfinite_count')) is int
            and meta['nonfinite_count']==0,'Failed numeric metadata in complete head ref')
    require(saved.get('failed_numeric_state') is False,'Failed state ref cannot count as completed actual B/C')
    path=Path(ref['path'])
    require(not path.is_absolute() and len(path.parts)==2 and path.parts[0]=='state_arrays' and path.suffix=='.npz',
        'Head reference escaped current archive')
    namespace=json.loads(ref['namespace'],parse_constant=_constant,object_pairs_hook=_object)
    require(all(namespace.get(k)==expected[k] for k in ('run_id','row_id','split_id'))
        and namespace.get('scope')=='query_benchmark_support_training' and namespace.get('fold') is None
        and namespace.get('trial') is None and namespace.get('state')==state,'Head reference escaped actual current split')


def _stream(root,stage,marker,splits,ground):
    path=root/STREAMS[stage];item=marker['streams'][stage]
    require(item['path']==STREAMS[stage] and item['file_bytes']==path.stat().st_size,'Fixed stream file identity/bytes mismatch')
    records=[];grouped=defaultdict(list)
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()),'Empty prediction record')
            record=json.loads(line,parse_constant=_constant,object_pairs_hook=_object)
            require(set(record)=={'split_id','query_id','classes','scores','prediction'},'Prediction carries forbidden or missing fields')
            sid=record['split_id'];require(sid in splits,'Unexpected prediction split')
            classes=ground if stage=='A' else splits[sid]['old_classes'] if stage=='B' else splits[sid]['registered_classes']
            require(record['classes']==classes and record['query_id'] in splits[sid]['query_ids'],'Prediction class/opaque ID binding mismatch')
            scores=record['scores']
            require(isinstance(scores,list) and len(scores)==len(classes)
                and all(type(v) in (int,float) and not isinstance(v,bool) and math.isfinite(v) for v in scores),
                'Invalid complete finite class scores')
            winner=classes[max(range(len(scores)),key=scores.__getitem__)]
            require(record['prediction']==winner,'Wrong per-sample full-column argmax')
            records.append(record);grouped[sid].append(record)
    require(item['record_count']==len(records) and set(grouped)==set(splits),'Missing prediction stream coverage')
    for sid,split in splits.items():
        require([r['query_id'] for r in grouped[sid]]==split['query_ids'],'Duplicate, reordered or missing opaque queries')
    return records,dict(grouped)


def load_fixed_predictions(*,predictions,capsule,config,expected_binding):
    """Validate one complete row without opening any truth path."""
    validate_config(config);root=Path(predictions)
    require(not any((root/name).exists() for name in ('prediction_failed.json','technical_failure.json')),'Failed predictions cannot be scored')
    startup=read(root/'startup.json');marker=read(root/'predictions_complete.json')
    require(marker['schema']==startup['schema']==PREDICTION_SCHEMA and marker['method']==startup['method']==METHOD
        and marker['status']=='COMPLETE','Incomplete fixed query predictions')
    require(re.fullmatch('[0-9a-f]{40}',expected_binding['release_commit']) is not None,'Actual runtime release commit required')
    for item in (startup,marker):
        require(all(item.get(k)==expected_binding[k] for k in IDENTITY),'Prediction current row/source identity mismatch')
        require(item.get('query_fit_access') is False and item.get('source_fit_access') is False and item.get('truth_read') is False,
            'Forbidden fitting/truth access')
        require(item.get('algorithm')==config['algorithm'] and item.get('qp_resources')==config['qp_resources'],'Prediction config/resource mismatch')
    require(startup.get('config')==config,'Resolved predictor config mismatch')
    manifest,splits=load_capsule_metadata(capsule,expected_binding['capsule_id'])
    ground=marker['ordered_ground_classes']
    require(strings(ground) and len(ground)==6 and marker['old_classes']==sorted(ground)
        and all(set(ground)==set(s['old_classes']) for s in splits.values()),'Ground six native columns mismatch')
    _source_identity(startup,marker,expected_binding)
    archive=read(root/'state_manifest.json')
    require(archive['schema']=='d92_margin_joint_state_archive_v1' and archive['method']==METHOD and archive['status']=='COMPLETE',
        'Incomplete current numeric archive manifest')
    refs={ref['path']:ref for ref in archive['files']}
    require(len(refs)==archive['file_count']==len(archive['files']),'Duplicate numeric archive refs')
    actual={s['split_id']:s for s in marker['splits']}
    require(len(actual)==len(marker['splits'])==len(splits) and set(actual)==set(splits),'Incomplete prediction split metadata')
    require(marker['split_count']==marker['completed_split_count']==len(splits)
        and marker['query_record_count']==sum(s['query_count'] for s in splits.values()),'Wrong completed split/query counts')
    for sid,split in splits.items():
        item=actual[sid]
        require(all(item.get(k)==v for k,v in split.items()),'Prediction support/query/class metadata mismatch')
        require(item['B_classes']==split['old_classes'] and item['C_classes']==split['registered_classes']
            and item['support_only_fit'] is True and item['query_fit_access'] is False and item['source_fit_access'] is False,
            'Actual support-only fitted columns mismatch')
        ctx=dict(expected_binding,split_id=sid)
        _state_ref(item['b_state_ref'],refs,ctx,'B_MARGIN')
        require(item['c_inherited_from_b_state_ref']==item['b_state_ref'],'C did not inherit actual same-split B')
        reuse=split['new_count']==0
        require(item['c_reuses_b'] is reuse,'new0 reuse boundary mismatch')
        if reuse:require(item['c_state_ref']==item['b_state_ref'],'new0 must reuse the exact B state ref')
        else:_state_ref(item['c_state_ref'],refs,ctx,'C_MARGIN_seq')
    require(set(marker['streams'])==set(STREAMS),'Missing or unexpected prediction streams')
    streams={};all_records={}
    for stage in STREAMS:
        records,streams[stage]=_stream(root,stage,marker,splits,ground);all_records[stage]=records
    require([r['split_id'] for r in all_records['A']]==[r['split_id'] for r in all_records['B']]
        ==[r['split_id'] for r in all_records['C']],'Three stream global order mismatch')
    alias=[]
    with (root/'predictions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:alias.append(json.loads(line,parse_constant=_constant,object_pairs_hook=_object))
    require(alias==all_records['C'],'Public C alias differs from frozen C stream')
    require(marker['compatibility_predictions']==dict(path='predictions.jsonl',alias_of='C',file_bytes=(root/'predictions.jsonl').stat().st_size),
        'Public C alias metadata mismatch')
    return dict(binding=dict(expected_binding),startup=startup,complete=marker,splits=splits,streams=streams,
        predictions_root=str(root),capsule_manifest=manifest,ground_classes=ground)


def _statistics(parents):
    dimensions=dict(overall=(),by_k_new_count=('k','new_count'),
        by_receiver_scene=('receiver','scenario','k','new_count'),
        by_model_row=('model_seed','row_id','k','new_count'),
        by_support_seed=('support_seed','k','new_count'),
        by_model_receiver_scene_support_seed=('model_seed','row_id','receiver','scenario','support_seed','k','new_count'))
    result={}
    for name,dims in dimensions.items():
        groups=defaultdict(list)
        for parent in parents:
            populations=['all']+(['new_present'] if parent['new_count'] else ['new0'])
            for population in populations:
                for metric in METRICS:groups[(population,)+tuple(parent[d] for d in dims)+(metric,)].append(parent['metrics'][metric])
        rows=[]
        for key,values in sorted(groups.items(),key=repr):
            measured=[v for v in values if v is not None]
            rows.append(dict(zip(('population',)+dims+('metric',),key),parent_count=len(values),measured_parent_count=len(measured),
                null_parent_count=len(values)-len(measured),mean=None if not measured else math.fsum(measured)/len(measured),
                minimum=None if not measured else min(measured),maximum=None if not measured else max(measured)))
        result[name]=rows
    return result


def _truth_parent(fixed,sid,truth):
    split=fixed['splits'][sid];old=set(split['old_classes']);classes=split['registered_classes']
    rows={stage:fixed['streams'][stage][sid] for stage in STREAMS};by_stage={s:{r['query_id']:r for r in v} for s,v in rows.items()}
    old_ids=[];new_ids=[];targets={}
    for pid in split['query_ids']:
        require(pid in truth,'Fixed opaque query is missing from truth')
        target=truth[pid];name=target['transmitter']
        require(target['pool_role']=='query' and target['receiver']==split['receiver'] and target['scene']==split['scenario']
            and name in classes and type(target['old']) is bool and target['old']==(name in old),'Independent truth binding mismatch')
        targets[pid]=name;(old_ids if name in old else new_ids).append(pid)
    require(all(any(targets[pid]==name for pid in split['query_ids']) for name in classes),'Missing registered-class query coverage')
    def accuracy(stage,pids):return None if not pids else sum(by_stage[stage][pid]['prediction']==targets[pid] for pid in pids)/len(pids)
    a,b,co,cn=accuracy('A',old_ids),accuracy('B',old_ids),accuracy('C',old_ids),accuracy('C',new_ids)
    metrics=dict(A_old_accuracy=a,B_old_accuracy=b,C_old_accuracy=co,C_new_accuracy=cn,
        C_h=None if cn is None else (0. if co+cn==0 else 2*co*cn/(co+cn)),
        adaptation_gain_B_minus_A=b-a,total_old_accuracy_drop=b-co,C_abs_new_old_gap=None if cn is None else abs(co-cn))
    if split['new_count']==0:
        require(not new_ids and metrics['C_new_accuracy'] is metrics['C_h'] is metrics['C_abs_new_old_gap'] is None,'new0 new/H/gap must be N/A')
        require(all(by_stage['B'][pid]['prediction']==by_stage['C'][pid]['prediction']
            and by_stage['B'][pid]['scores']==by_stage['C'][pid]['scores'] for pid in split['query_ids']),'new0 changed the actual B function')
    counts={name:sum(v==name for v in targets.values()) for name in classes}
    class_accuracy={stage:{name:sum(by_stage[stage][pid]['prediction']==name for pid in split['query_ids'] if targets[pid]==name)/counts[name]
        for name in (split['old_classes'] if stage!='C' else classes)} for stage in STREAMS}
    metadata=next(v for v in fixed['complete']['splits'] if v['split_id']==sid)
    return dict(**fixed['binding'],**{k:split[k] for k in SPLIT_IDENTITY},old_class_count=6,registered_class_count=len(classes),
        ordered_ground_classes=fixed['ground_classes'],old_classes=split['old_classes'],registered_classes=classes,
        old_query_ids=old_ids,new_query_ids=new_ids,old_query_count=len(old_ids),new_query_count=len(new_ids),
        query_count=len(split['query_ids']),query_class_counts=counts,class_accuracy=class_accuracy,metrics=metrics,
        b_state_ref=metadata['b_state_ref'],c_state_ref=metadata['c_state_ref'])


def _mapped(path,spec,run_root):
    source=Path(path)
    if run_root is None:return source
    original=Path(spec['execution']['remote_run_root'])
    require(source.is_relative_to(original),'Declared output escaped current run root')
    return Path(run_root)/source.relative_to(original)


def aggregate_training_resources(rows):
    """Reported actual work only: missing is unknown, and two peaks use MAX."""
    audits=[row['resources']['actual_training_counters'] for row in rows]
    keys=set().union(*(set(v) for v in audits));totals={}
    for key in sorted(keys):
        values=[v.get(key) for v in audits]
        require(all(v is None or _integer(v) for v in values),'Invalid reported training counter: '+key)
        totals[key]=None if any(v is None for v in values) else max(values) if key in PEAK_COUNTERS else sum(values)
    return dict(actual_training_counters=totals,counter_aggregation='SUM_EXCEPT_TWO_QP_PEAKS_MAX',
        missing_counter_policy='UNKNOWN_NOT_ZERO',training_measurements=None,
        training_measurements_reason='SCORER_DOES_NOT_MEASURE_TRAINING_OR_DEPLOYMENT',peak_memory_bytes=None,
        peak_gpu_memory_bytes=None,deployment_numeric_state_bytes=None,incremental_network_transfer_bytes=None,energy=None)


def score_benchmark(*,spec,output,run_root=None):
    """Finish every prediction validation and independent reread before truth."""
    output=Path(output)
    require(not output.exists(),'Exclusive score output required')
    require(spec['schema']=='d92_margin_joint_query_benchmark_v1','Wrong current benchmark spec')
    validate_config(spec['benchmark']['config']);rows=spec['rows']
    require(isinstance(rows,list) and rows and strings([r['row_id'] for r in rows]),'Duplicate or missing declared rows')
    root=Path(spec['execution']['remote_run_root']) if run_root is None else Path(run_root)
    startup=read(root/'startup.json');complete=read(root/'complete.json')
    require(startup['schema']==complete['schema']==spec['schema'] and startup['status']=='MARGIN_QUERY_SUPERVISOR_STARTED'
        and startup['resolved_spec']==complete['resolved_spec']==spec
        and complete['run_id']==startup['run_id']==spec['run_id']
        and complete['group_id']==startup['group_id']==spec['group_id'],'Current supervisor spec/run mismatch')
    require(complete['status']=='MARGIN_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
        and complete['row_count']==complete['completed_row_count']==len(rows)
        and complete['all_predictions_fixed'] is True and complete['truth_read'] is False
        and complete['scorer_invoked'] is False and complete['automatic_retry'] is False
        and set(complete['rows'])==set(startup['rows'])=={r['row_id'] for r in rows}
        and all(v['status']=='COMPLETE' for v in complete['rows'].values()),
        'All declared rows must finish before truth access')
    runtime=complete['runtime_commit']
    require(runtime==startup['runtime_commit'] and re.fullmatch('[0-9a-f]{40}',runtime) is not None
        and complete['code_commit']==startup['code_commit']==spec['code']['commit'],'Runtime/preparation commit binding mismatch')
    fixed=[];cohorts=spec['benchmark']['cohorts']
    for row in rows:
        co=cohorts[row['cohort']]
        binding=dict(run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['expected_capsule_id'],
            checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],release_commit=runtime)
        path=_mapped(str(Path(row['output_root'])/'predictions'),spec,run_root)
        value=load_fixed_predictions(predictions=path,capsule=co['capsule'],config=spec['benchmark']['config'],expected_binding=binding)
        lane=complete['rows'][row['row_id']]
        require(lane['marker']==value['complete'] and lane['release_commit']==runtime
            and lane['expected_model_seed']==row['expected_model_seed'] and lane['expected_checkpoint_sha256']==row['expected_checkpoint_sha256']
            and lane['expected_capsule_id']==co['expected_capsule_id'] and lane['output_root']==row['output_root']
            and lane['prediction_output']==str(Path(row['output_root'])/'predictions'),'Supervisor/lane fixed marker binding mismatch')
        source_paths=lane['source_paths']
        require(source_paths==dict(row_root=row['row_root'],branch_features=row['branch_features'],ground_packet=row['ground_packet'],capsule=co['capsule']),
            'Supervisor source paths changed')
        run_binding=value['complete']['run_binding']
        require(all(Path(run_binding[k]).resolve()==Path(v).resolve() for k,v in source_paths.items())
            and Path(run_binding['prediction_output_root']).resolve()==Path(row['output_root']+'/predictions').resolve(),
            'Predictor source path/run output binding mismatch')
        fixed.append(value)
    # Reopen every fixed stream and completion/source identity after all rows
    # validate. A truth reader cannot run from a partially verified prefix.
    for row,old in zip(rows,fixed):
        co=cohorts[row['cohort']]
        reread=load_fixed_predictions(predictions=old['predictions_root'],capsule=co['capsule'],config=spec['benchmark']['config'],expected_binding=old['binding'])
        require(reread==old,'Frozen predictions changed during independent reread')
    require(read(root/'startup.json')==startup and read(root/'complete.json')==complete,'Supervisor changed before truth join')
    parents=[];truths={};old_sets={};started=time.perf_counter()
    for row,value in zip(rows,fixed):
        path=str(cohorts[row['cohort']]['truth'])
        if path not in truths:truths[path]=read(path)
        for sid in value['splits']:
            parent=_truth_parent(value,sid,truths[path]);parent['cohort']=row['cohort']
            key=(row['row_id'],parent['receiver'],parent['scenario'],parent['k'],parent['support_seed'])
            paired=(sorted(parent['old_query_ids']),value['splits'][sid]['old_support_ids'])
            require(old_sets.setdefault(key,paired)==paired,'Old physical queries/support changed across registration counts')
            parents.append(parent)
    resource_rows=[dict(binding=v['binding'],resources=v['complete']['resources']) for v in fixed]
    resources=aggregate_training_resources(resource_rows)
    resources['scoring_truth_join_seconds']=time.perf_counter()-started
    result=dict(schema=SCHEMA,method=METHOD,status=STATUS,scope=SCOPE,run_id=spec['run_id'],group_id=spec['group_id'],
        release_commit=runtime,code_commit=spec['code']['commit'],algorithm=spec['benchmark']['config']['algorithm'],
        qp_resources=spec['benchmark']['config']['qp_resources'],row_count=len(rows),parent_count=len(parents),
        parents=parents,statistics=_statistics(parents),
        current_metadata=dict(spec=deepcopy(spec),supervisor_startup=startup,supervisor_complete=complete,
            rows=[dict(binding=v['binding'],source_identity=v['complete']['source_identity'],ground_packet_identity=v['complete']['ground_packet_identity'],
                streams=v['complete']['streams'],split_count=len(v['splits']),resources=v['complete']['resources'],
                predictor_device=v['startup'].get('device')) for v in fixed]),
        prediction_validation_complete_before_truth=True,prediction_streams_independently_reread=True,
        query_fit_access=False,source_fit_access=False,training_performed=False,model_called=False,selection_feedback_forbidden=True,
        data_reuse_statement='Repeated frozen benchmark data; not a fresh independent confirmation',
        resources=resources)
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result)
    require(read(output)==result,'Score artifact independent readback failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run-root',type=Path)
    a=p.parse_args();result=score_benchmark(spec=read(a.spec),output=a.output,run_root=a.run_root)
    print(json.dumps(dict(status=result['status'],row_count=result['row_count'],parent_count=result['parent_count'])))


if __name__=='__main__':main()
