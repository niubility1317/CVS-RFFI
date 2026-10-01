"""Truth-last scoring of all frozen GroupBarrierJoint A/B/C query predictions.

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

METHOD='D92-GroupBarrierJointLocalRidge-v1'
PREDICTION_SCHEMA='d92_group_barrier_joint_query_predictions_v1'
SCHEMA='d92_group_barrier_joint_query_score_v1'
STATUS='GROUP_BARRIER_JOINT_QUERY_SCORE_COMPLETE'
SCOPE='FROZEN_QUERY_BENCHMARK_TRUTH_LAST_THREE_STAGE_PAIRED'
KS=(1,5,10,20)
NEWS=(0,2,5,10,20)
METRICS=('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','C_h',
    'adaptation_gain_B_minus_A','total_old_accuracy_drop','C_abs_new_old_gap',
    'A_old_macro_f1','B_old_macro_f1','C_old_macro_f1','C_new_macro_f1','C_macro_f1')
IDENTITY=('run_id','row_id','capsule_id','checkpoint_sha256','model_seed','release_commit')
SPLIT_IDENTITY=('split_id','receiver','scenario','k','new_count','support_seed')
STREAMS={s:'predictions_'+s+'.jsonl' for s in ('A','B','C')}
PEAK_COUNTERS=frozenset(('margin_qp_peak_factor_buffer_bytes','margin_qp_peak_explicit_solve_temporary_bytes',
    'gate_peak_factor_buffer_bytes','gate_peak_explicit_temporary_bytes','forward_cache_bytes'))
MODEL_SEEDS=(2026092701,2026092702)
COHORT_COUNTS={'rx3':900,'rx1':300}
RECEIVERS={'rx3':['19-1','8-14','8-7'],'rx1':['20-19']}
SCENARIOS=['practical_high','practical_mid','practical_low_urban']
SUPPORT_SEEDS=list(range(2026092711,2026092716))
FROZEN_RESOURCES=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160)
FACTOR_COMPONENTS=('inner_factorization_count','final_factorization_count','prior_factorization_count',
    'new_ridge_factorization_count','group_gate_forward_factorization_attempts','group_gate_adjoint_factorization_attempts')
DECISION_SCHEMA='d92_group_barrier_single_query_decision_v1'
DECISION_POLICY='WITHIN_GROUP_RAW_ARGMAX_THEN_GATE_LOGPROB_GAP_LEXICAL_TIE'
# Independent literal method contract, never imported from a fitting module.
FROZEN_ALGORITHM=dict(
    schema='d92_group_barrier_joint_local_ridge_v1',method=METHOD,
    candidate_status='SOURCE_ONLY_DRAFT_NOT_PUBLISHED',B='unchanged_CE_only_hard_ball_MarginJoint_B',
    C='new_only_affine_ridge_plus_all_pair_fixed_positive_barrier_Bernoulli_gate',
    B_objective='RMS_across_classes_of_cross_fold_mean_CE_only',C_objective='RMS_class_mean_CE_only',
    rank=8,input_dim=736,dictionary='first8_orthonormal_DCT_rows',learn_V=False,
    initialization_C='actual_current_B_U_no_reset',coordinate_ball_radius=.5,proximal_coefficient=0.,
    ridge_coefficient=1.,temperature=1.,barrier_average_original_objective_gap=1e-4,
    barrier_law='zeta=N*1e-4/(old_physical_count*new_class_count)_fixed_per_head',
    bandwidth='fixed_original_old_inner_train',trace_scale='fixed_actual_old_prior',max_iterations=4,max_trials=12,
    initial_step_size=.125,backtrack_factor=.5,free_new_intercept=True,free_gate_intercept=True,max_coordinates=5888,
    query_decision_policy='per_physical_sample_all_registered_classes',tie_break='physical_class_id_lexicographic',
    phase1_frozen=True,source_inputs=False,query_fit=False,encoder_backward=False,parameter_search=False,
    extra_ground_data_stat_payload_bytes=0,code_wire_payload_bytes=None,dtype='float64')
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
    require(isinstance(config,dict) and set(config)=={'algorithm','group_barrier_resources'},'Wrong fixed benchmark config')
    require(config['algorithm']==FROZEN_ALGORITHM,'Wrong frozen method algorithm')
    limits=config['group_barrier_resources']
    require(isinstance(limits,dict) and set(limits)=={'max_newton_iterations','max_line_search_trials','max_factor_buffer_bytes'}
        and all(_integer(v,1) for v in limits.values()),'Explicit GroupBarrier resource limits required')
    require(limits==FROZEN_RESOURCES,'Registered GroupBarrier resource limits changed')


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
            require(record['prediction'] in classes,'Prediction is outside all registered columns')
            if stage!='C' or splits[sid]['new_count']==0:
                winner=classes[max(range(len(scores)),key=scores.__getitem__)]
                require(record['prediction']==winner,'Wrong per-sample full-column argmax')
            records.append(record);grouped[sid].append(record)
    require(item['record_count']==len(records) and set(grouped)==set(splits),'Missing prediction stream coverage')
    for sid,split in splits.items():
        require([r['query_id'] for r in grouped[sid]]==split['query_ids'],'Duplicate, reordered or missing opaque queries')
    return records,dict(grouped)


def _vector(value,count,name):
    require(isinstance(value,list) and len(value)==count and all(type(v) in (int,float)
        and math.isfinite(v) for v in value),'Invalid decision vector: '+name)
    return np.asarray(value,dtype=np.float64)


def _numeric_readback(actual,expected,operation_count,name):
    # Binary64 forward arithmetic readback bound, independent of accuracy.
    expected=np.asarray(expected,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    scale=np.maximum(1.,np.abs(expected))
    bound=np.finfo(np.float64).eps*operation_count*scale
    require(actual.shape==expected.shape and np.all(np.isfinite(actual))
        and bool(np.all(np.abs(actual-expected)<=bound)),'Structural numeric readback differs: '+name)


def _seconds(value,name):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid measured seconds: '+name)
    return value


def _seconds_sum_readback(total,values,name):
    """Compare one clock scope only, allowing binary64 sequential SUM error."""
    total=_seconds(total,name)
    expected=math.fsum(_seconds(v,name) for v in values)
    count=max(1,len(values));eps=np.finfo(np.float64).eps
    require(count*eps<1,'Seconds readback accumulation bound is undefined')
    # gamma_n bounds nonnegative sequential addition; fsum adds one rounding.
    bound=(count*eps/(1-count*eps)+eps)*abs(expected)
    require(abs(total-expected)<=bound,'Measured seconds SUM readback differs: '+name)


def _work_seconds(work,name):
    require(isinstance(work,dict) and 'score_seconds' in work,'Missing returned score timing: '+name)
    for key,value in work.items():
        if key.endswith('_seconds'):_seconds(value,name+'.'+key)


def verify_structured_decision(certificate,c_record,b_record,split):
    require(isinstance(certificate,dict) and certificate.get('schema')==DECISION_SCHEMA,'Missing structural C decision certificate')
    require(certificate.get('prediction')==c_record['prediction'],'C prediction/certificate mismatch')
    if split['new_count']==0:
        require(set(certificate)=={'schema','policy','new0_reuses_actual_B','prediction'}
            and certificate['policy']=='EXACT_ACTUAL_B_REUSE' and certificate['new0_reuses_actual_B'] is True,
            'new0 structural certificate mismatch')
        require(c_record==b_record,'new0 changed actual B scores or prediction')
        return
    fields={'schema','policy','new0_reuses_actual_B','old_classes','new_classes','old_raw_scores','new_raw_scores',
        'gate_logit','old_log_probabilities','new_log_probabilities','old_winner','new_winner','group_gap','prediction'}
    require(set(certificate)==fields and certificate['policy']==DECISION_POLICY
        and certificate['new0_reuses_actual_B'] is False,'Wrong all-group structural decision policy')
    old=split['old_classes'];new=[v for v in split['registered_classes'] if v not in old]
    require(certificate['old_classes']==old and certificate['new_classes']==new,'Structural decision omitted registered columns')
    rawold=_vector(certificate['old_raw_scores'],len(old),'old raw')
    rawnew=_vector(certificate['new_raw_scores'],len(new),'new raw')
    # Actual B remains the old conditional classifier, not an independently fit teacher.
    require(certificate['old_raw_scores']==b_record['scores'],'C certificate did not use actual same-query B')
    lo=_vector(certificate['old_log_probabilities'],len(old),'old log probabilities')
    ln=_vector(certificate['new_log_probabilities'],len(new),'new log probabilities')
    def logsoftmax(values):
        shift=values-values.max();return shift-np.log(np.exp(shift).sum())
    _numeric_readback(lo,logsoftmax(rawold),len(old)+8,'old logsoftmax')
    _numeric_readback(ln,logsoftmax(rawnew),len(new)+8,'new logsoftmax')
    g=certificate['gate_logit'];gap=certificate['group_gap']
    require(type(g) in (int,float) and math.isfinite(g) and type(gap) in (int,float) and math.isfinite(gap),
        'Invalid gate scalar')
    oi=int(np.argmax(rawold));ni=int(np.argmax(rawnew));ow,nw=old[oi],new[ni]
    require(certificate['old_winner']==ow and certificate['new_winner']==nw,'Wrong raw within-group winner')
    require(gap==float(g+lo[oi]-ln[ni]),'Wrong structural group gap')
    expected=ow if gap>0 else nw if gap<0 else min(ow,nw)
    require(c_record['prediction']==expected,'Wrong public structured C decision')
    values={name:float(-np.logaddexp(0.,-g)+lo[i]) for i,name in enumerate(old)}
    values.update({name:float(-np.logaddexp(0.,g)+ln[i]) for i,name in enumerate(new)})
    _numeric_readback(c_record['scores'],[values[name] for name in split['registered_classes']],
        len(old)+len(new)+12,'all-column composed scores')


def _decision_stream(root,marker,splits,records):
    path=root/'query_score_work.jsonl';meta=marker['query_score_work']
    require(meta==dict(path=path.name,record_count=len(records['C']),file_bytes=path.stat().st_size),
        'Complete structural decision stream metadata differs')
    decisions=[]
    with path.open(encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            require(index<len(records['C']),'Unexpected structural decision')
            row=json.loads(line,parse_constant=_constant,object_pairs_hook=_object)
            require(set(row)=={'split_id','query_id','B_work','C_work','C_reused_B_scores','C_structural_decision',
                'C_structural_prediction','C_public_predict_calls','C_public_predict_seconds','C_public_predict_internal_work',
                'C_public_predict_internal_work_unavailable_reason','scope'},'Wrong query work schema')
            c=records['C'][index];b=records['B'][index]
            require((row.get('split_id'),row.get('query_id'))==(c['split_id'],c['query_id']),
                'Structural decision order/physical identity mismatch')
            verify_structured_decision(row.get('C_structural_decision'),c,b,splits[c['split_id']])
            reuse=splits[c['split_id']]['new_count']==0
            require(row['C_structural_prediction']==c['prediction'] and row['C_reused_B_scores'] is reuse
                and type(row['C_public_predict_calls']) is int and row['C_public_predict_calls']==int(not reuse)
                and type(row['C_public_predict_seconds']) in (int,float) and math.isfinite(row['C_public_predict_seconds'])
                and row['C_public_predict_seconds']>=0 and row['C_public_predict_internal_work'] is None,
                'Structured public inference work binding differs')
            require(isinstance(row['B_work'],dict) and (row['C_work'] is None if reuse else isinstance(row['C_work'],dict)),
                'Actual score work/reuse scope differs')
            _work_seconds(row['B_work'],'B_work')
            if not reuse:_work_seconds(row['C_work'],'C_work')
            require(row['C_public_predict_internal_work_unavailable_reason']==('EXACT_B_REUSE_NO_EXTRA_CALL' if reuse
                else 'PUBLIC_PREDICT_RECOMPUTES_GEOMETRY_WITHOUT_RETURNING_AUDIT'),'Unknown public inference work reason differs')
            require(row['scope']=='ACTUAL_SINGLETON_CORE_SCORE_GEOMETRY_PLUS_PUBLIC_STRUCTURAL_PREDICT; C_REUSE_HAS_NO_EXTRA_CALL',
                'Query work scope differs')
            decisions.append(row)
    require(len(decisions)==len(records['C']),'Missing structural query decisions')
    return decisions


def _fixed_resources(marker,decisions):
    resources=marker['resources'];counters=resources['actual_training_counters']
    require(all(k in counters and _integer(counters[k]) for k in FACTOR_COMPONENTS+tuple(PEAK_COUNTERS)),
        'Missing actual factor/peak resource schema')
    require(_integer(resources.get('actual_factorization_attempts'))
        and resources['actual_factorization_attempts']==sum(counters[k] for k in FACTOR_COMPONENTS),
        'Actual factorization ledger differs')
    calls=sum(row['C_public_predict_calls'] for row in decisions);n=len(decisions)
    require(isinstance(resources.get('query_score_calls'),dict)
        and all(_integer(v) for v in resources['query_score_calls'].values())
        and resources['query_score_calls']==dict(A=n,B=n,C=calls)
        and _integer(resources.get('C_public_predict_calls'))
        and resources['C_public_predict_calls']==calls,'Actual singleton query inference count differs')
    require(resources.get('work_aggregation')=='SUM' and resources.get('peak_aggregation')=='MAX',
        'Resource SUM/MAX scope differs')
    measured=resources.get('query_score_seconds')
    require(isinstance(measured,dict) and set(measured)==set(STREAMS),'Missing query score seconds')
    for stage,value in measured.items():_seconds(value,'query_score_seconds.'+stage)
    _seconds_sum_readback(resources.get('C_public_predict_seconds'),
        [row['C_public_predict_seconds'] for row in decisions],'C_public_predict_seconds')
    aggregate_training_resources([dict(resources=resources)])


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
        require(all(item.get(k) is False for k in ('checkpoint_loaded','encoder_called','cross_split_adapted_state_reuse')),
            'Forbidden checkpoint/encoder/cross-split adapted state use')
        require(item.get('C_decision_policy')==DECISION_POLICY and item.get('C_decision_certificate_schema')==DECISION_SCHEMA
            and item.get('fit_scope')=='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY'
            and item.get('query_inference_scope')=='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS'
            and item.get('A_tie_policy')=='first_original_native_head_column'
            and item.get('B_C_tie_policy')=='physical_class_id_ascending','Wrong singleton all-class prediction policy')
        require(item.get('algorithm')==config['algorithm'] and item.get('group_barrier_resources')==config['group_barrier_resources'],'Prediction config/resource mismatch')
    require(startup.get('config')==config,'Resolved predictor config mismatch')
    manifest,splits=load_capsule_metadata(capsule,expected_binding['capsule_id'])
    ground=marker['ordered_ground_classes']
    require(strings(ground) and len(ground)==6 and marker['old_classes']==sorted(ground)
        and all(set(ground)==set(s['old_classes']) for s in splits.values()),'Ground six native columns mismatch')
    _source_identity(startup,marker,expected_binding)
    archive=read(root/'state_manifest.json')
    require(archive['schema']=='d92_group_barrier_joint_state_archive_v1' and archive['method']==METHOD and archive['status']=='COMPLETE',
        'Incomplete current numeric archive manifest')
    refs={ref['path']:ref for ref in archive['files']}
    require(len(refs)==archive['file_count']==len(archive['files']),'Duplicate numeric archive refs')
    actual={s['split_id']:s for s in marker['splits']}
    require(len(actual)==len(marker['splits'])==len(splits) and set(actual)==set(splits),'Incomplete prediction split metadata')
    require(marker['split_count']==marker['completed_split_count']==len(splits)
        and marker['query_record_count']==sum(s['query_count'] for s in splits.values()),'Wrong completed split/query counts')
    require(marker.get('actual_stage_count')==sum(1+int(s['new_count']>0) for s in splits.values())
        and marker.get('state_manifest')=='state_manifest.json','Actual B/C stage completion count differs')
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
        else:_state_ref(item['c_state_ref'],refs,ctx,'C_GROUP_BARRIER_seq')
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
    decisions=_decision_stream(root,marker,splits,all_records)
    _fixed_resources(marker,decisions)
    return dict(binding=dict(expected_binding),startup=startup,complete=marker,splits=splits,streams=streams,
        predictions_root=str(root),capsule_manifest=manifest,ground_classes=ground,decisions=decisions)


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
    # First average physical parents within a model seed; only then summarize
    # the two registered model-seed means. SD is descriptive sample SD, not CI.
    seed_tables={}
    for name,dims in dict(overall=(),by_k_new_count=('k','new_count'),
            by_receiver_scene=('receiver','scenario','k','new_count'),
            by_support_seed=('support_seed','k','new_count')).items():
        buckets=defaultdict(lambda:defaultdict(list))
        for parent in parents:
            for population in ['all','new_present' if parent['new_count'] else 'new0']:
                for metric in METRICS:
                    key=(population,)+tuple(parent[d] for d in dims)+(metric,)
                    buckets[key][parent['model_seed']].append(parent['metrics'][metric])
        records=[]
        for key,seeds in sorted(buckets.items(),key=repr):
            require(set(seeds)==set(MODEL_SEEDS),'Missing paired model seed statistics')
            means={str(seed):(None if all(v is None for v in values)
                else math.fsum(v for v in values if v is not None)/sum(v is not None for v in values))
                for seed,values in sorted(seeds.items())}
            measured=[v for v in means.values() if v is not None]
            mean=None if not measured else math.fsum(measured)/len(measured)
            sd=None if len(measured)<2 else math.sqrt(math.fsum((v-mean)**2 for v in measured)/(len(measured)-1))
            records.append(dict(zip(('population',)+dims+('metric',),key),model_seed_means=means,
                model_seed_count=len(measured),mean=mean,sample_standard_deviation=sd,
                aggregation='PARENT_MEAN_WITHIN_MODEL_SEED_THEN_EQUAL_MODEL_SEED_MEAN',
                uncertainty_scope='DESCRIPTIVE_TWO_MODEL_SEED_SD_NOT_CONFIDENCE_INTERVAL'))
        seed_tables[name]=records
    result['model_seed_mean_sd']=seed_tables
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
    def macro_f1(stage,pids,names):
        if not pids or not names:return None
        values=[]
        for name in names:
            tp=sum(targets[pid]==name and by_stage[stage][pid]['prediction']==name for pid in pids)
            fp=sum(targets[pid]!=name and by_stage[stage][pid]['prediction']==name for pid in pids)
            fn=sum(targets[pid]==name and by_stage[stage][pid]['prediction']!=name for pid in pids)
            denominator=2*tp+fp+fn
            values.append(0. if denominator==0 else 2*tp/denominator)
        return math.fsum(values)/len(values)
    metrics=dict(A_old_accuracy=a,B_old_accuracy=b,C_old_accuracy=co,C_new_accuracy=cn,
        C_h=None if cn is None else (0. if co+cn==0 else 2*co*cn/(co+cn)),
        adaptation_gain_B_minus_A=b-a,total_old_accuracy_drop=b-co,C_abs_new_old_gap=None if cn is None else abs(co-cn),
        A_old_macro_f1=macro_f1('A',old_ids,split['old_classes']),B_old_macro_f1=macro_f1('B',old_ids,split['old_classes']),
        C_old_macro_f1=macro_f1('C',old_ids,split['old_classes']),
        C_new_macro_f1=macro_f1('C',new_ids,[c for c in classes if c not in old]),
        C_macro_f1=macro_f1('C',split['query_ids'],classes))
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
        macro_f1_scope='OLD_AND_NEW_METRICS_USE_THEIR_PHYSICAL_QUERY_SUBSET; C_MACRO_F1_USES_ALL_REGISTERED_QUERIES',
        b_state_ref=metadata['b_state_ref'],c_state_ref=metadata['c_state_ref'])


def _mapped(path,spec,run_root):
    source=Path(path)
    if run_root is None:return source
    original=Path(spec['execution']['remote_run_root'])
    require(source.is_relative_to(original),'Declared output escaped current run root')
    return Path(run_root)/source.relative_to(original)


def aggregate_training_resources(rows):
    """Reported actual work only; seconds may be floats, peaks are not summed."""
    audits=[row['resources']['actual_training_counters'] for row in rows]
    keys=set().union(*(set(v) for v in audits));totals={}
    for key in sorted(keys):
        values=[v.get(key) for v in audits]
        require(all(v is None or (type(v) in (int,float) and math.isfinite(v) and v>=0
            and (key.endswith('_seconds') or type(v) is int)) for v in values),'Invalid reported training counter: '+key)
        totals[key]=None if any(v is None for v in values) else max(values) if key in PEAK_COUNTERS else sum(values)
    return dict(actual_training_counters=totals,counter_aggregation='SUM_EXCEPT_DECLARED_PEAK_COUNTERS_MAX',
        peak_counter_keys=sorted(PEAK_COUNTERS),
        missing_counter_policy='UNKNOWN_NOT_ZERO',training_measurements=None,
        training_measurements_reason='SCORER_DOES_NOT_MEASURE_TRAINING_OR_DEPLOYMENT',peak_memory_bytes=None,
        peak_gpu_memory_bytes=None,deployment_numeric_state_bytes=None,incremental_network_transfer_bytes=None,energy=None)


def validate_declared_matrix(spec):
    rows=spec['rows'];cohorts=spec['benchmark']['cohorts']
    require(set(cohorts)==set(COHORT_COUNTS),'Both frozen receiver cohorts required')
    require(isinstance(rows,list) and len(rows)==4 and strings([r['row_id'] for r in rows]),
        'Exactly four distinct model/cohort rows required; subset scoring forbidden')
    require({(r['cohort'],r['expected_model_seed']) for r in rows}
        ==set(itertools.product(COHORT_COUNTS,MODEL_SEEDS)),'Fixed model/cohort cross product required')
    models={};outputs=set()
    for row in rows:
        require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',row['row_id']) is not None
            and Path(row['output_root'])==Path(spec['execution']['remote_run_root'])/row['row_id'],
            'Row output escaped the declared current run')
        require(re.fullmatch('[0-9a-f]{64}',row['expected_checkpoint_sha256']) is not None,'Invalid checkpoint SHA')
        identity=(row['expected_checkpoint_sha256'],row['ground_packet'])
        require(models.setdefault(row['expected_model_seed'],identity)==identity,'Same model seed changed checkpoint/packet across cohorts')
        require(row['output_root'] not in outputs,'Duplicate row output path');outputs.add(row['output_root'])
    require(len({value[0] for value in models.values()})==2,'Distinct model seeds require distinct checkpoint identities')
    for name,co in cohorts.items():
        expected=dict(receivers=RECEIVERS[name],scenarios=SCENARIOS,ks=list(KS),new_counts=list(NEWS),support_seeds=SUPPORT_SEEDS)
        require(co.get('matrix')==expected and type(co.get('expected_split_count')) is int
            and co['expected_split_count']==COHORT_COUNTS[name],'Declared full receiver/scene/support-seed/K/new matrix differs')
    return rows,cohorts


def _physical_matrix(value,co):
    matrix=co['matrix']
    expected=set(itertools.product(matrix['receivers'],matrix['scenarios'],matrix['support_seeds'],matrix['ks'],matrix['new_counts']))
    actual={(s['receiver'],s['scenario'],s['support_seed'],s['k'],s['new_count']) for s in value['splits'].values()}
    require(actual==expected and len(value['splits'])==co['expected_split_count'],'Incomplete actual physical full query matrix')


def score_benchmark(*,spec,output,run_root=None):
    """Finish every prediction validation and independent reread before truth."""
    output=Path(output)
    require(not output.exists(),'Exclusive score output required')
    require(spec['schema']=='d92_group_barrier_joint_query_benchmark_v1','Wrong current benchmark spec')
    validate_config(spec['benchmark']['config']);rows,cohorts=validate_declared_matrix(spec)
    root=Path(spec['execution']['remote_run_root']) if run_root is None else Path(run_root)
    startup=read(root/'startup.json');complete=read(root/'complete.json')
    require(startup['schema']==complete['schema']==spec['schema'] and startup['status']=='GROUP_BARRIER_QUERY_SUPERVISOR_STARTED'
        and startup['resolved_spec']==complete['resolved_spec']==spec
        and complete['run_id']==startup['run_id']==spec['run_id']
        and complete['group_id']==startup['group_id']==spec['group_id'],'Current supervisor spec/run mismatch')
    require(complete['status']=='GROUP_BARRIER_QUERY_BENCHMARK_PREDICTIONS_COMPLETE'
        and complete['row_count']==complete['completed_row_count']==len(rows)
        and complete['all_predictions_fixed'] is True and complete['truth_read'] is False
        and complete['scorer_invoked'] is False and complete['automatic_retry'] is False
        and set(complete['rows'])==set(startup['rows'])=={r['row_id'] for r in rows}
        and all(v['status']=='COMPLETE' for v in complete['rows'].values()),
        'All declared rows must finish before truth access')
    runtime=complete['runtime_commit']
    require(runtime==startup['runtime_commit'] and re.fullmatch('[0-9a-f]{40}',runtime) is not None
        and complete['code_commit']==startup['code_commit']==spec['code']['commit'],'Runtime/preparation commit binding mismatch')
    fixed=[];cohort_physical={}
    for row in rows:
        co=cohorts[row['cohort']]
        binding=dict(run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['expected_capsule_id'],
            checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],release_commit=runtime)
        path=_mapped(str(Path(row['output_root'])/'predictions'),spec,run_root)
        value=load_fixed_predictions(predictions=path,capsule=co['capsule'],config=spec['benchmark']['config'],expected_binding=binding)
        _physical_matrix(value,co)
        require(cohort_physical.setdefault(row['cohort'],value['splits'])==value['splits'],
            'Same cohort model rows changed physical support/query split metadata')
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
    require(sum(len(v['splits']) for v in fixed)==2400,'Exactly 2400 complete parents required before truth access')
    # Reopen every fixed stream and completion/source identity after all rows
    # validate. A truth reader cannot run from a partially verified prefix.
    for row,old in zip(rows,fixed):
        co=cohorts[row['cohort']]
        reread=load_fixed_predictions(predictions=old['predictions_root'],capsule=co['capsule'],config=spec['benchmark']['config'],expected_binding=old['binding'])
        require(reread==old,'Frozen predictions changed during independent reread')
    require(read(root/'startup.json')==startup and read(root/'complete.json')==complete,'Supervisor changed before truth join')
    resource_rows=[dict(binding=v['binding'],resources=v['complete']['resources']) for v in fixed]
    resources=aggregate_training_resources(resource_rows)
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
    resources['scoring_truth_join_seconds']=time.perf_counter()-started
    result=dict(schema=SCHEMA,method=METHOD,status=STATUS,scope=SCOPE,run_id=spec['run_id'],group_id=spec['group_id'],
        release_commit=runtime,code_commit=spec['code']['commit'],algorithm=spec['benchmark']['config']['algorithm'],
        group_barrier_resources=spec['benchmark']['config']['group_barrier_resources'],row_count=len(rows),parent_count=len(parents),
        parents=parents,statistics=_statistics(parents),
        current_metadata=dict(spec=deepcopy(spec),supervisor_startup=startup,supervisor_complete=complete,
            rows=[dict(binding=v['binding'],source_identity=v['complete']['source_identity'],ground_packet_identity=v['complete']['ground_packet_identity'],
                streams=v['complete']['streams'],split_count=len(v['splits']),resources=v['complete']['resources'],
                predictor_device=v['startup'].get('device')) for v in fixed]),
        prediction_validation_complete_before_truth=True,prediction_streams_independently_reread=True,
        query_fit_access=False,source_fit_access=False,training_performed=False,model_called=False,selection_feedback_forbidden=True,
        data_reuse_statement='Repeated frozen benchmark data; not a fresh independent confirmation',
        resources=resources,automatic_promotion=False,
        score_units='FRACTIONS; SUBTRACTIONS_ARE_FRACTION_DIFFERENCES',
        metric_definitions=dict(total_old_accuracy_drop='B_old_accuracy-C_old_accuracy',
            adaptation_gain_B_minus_A='B_old_accuracy-A_old_accuracy',
            C_h='HARMONIC_MEAN_COMPUTED_WITHIN_EACH_PHYSICAL_PARENT_BEFORE_AGGREGATION',
            C_abs_new_old_gap='ABS_C_OLD_MINUS_C_NEW_WITHIN_PARENT',
            K1='QUERY_BENCHMARK_IS_VALID; NOT_SUPPORT_HELD_DIAGNOSTICS',
            new0='C_EXACT_ACTUAL_B; C_NEW_ACCURACY/H/GAP/NEW_MACRO_F1_NULL'))
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
