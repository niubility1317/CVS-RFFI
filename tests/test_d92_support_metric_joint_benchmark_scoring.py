"""Literal fixed prediction contracts; no actual benchmark files are accessed."""
import ast
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__' and sys.argv[1:]==['--static-only']:
    for path in (ROOT/'tools/score_d92_support_metric_joint_benchmark.py',Path(__file__)):
        source=path.read_bytes().decode('utf-8',errors='strict')
        assert not any(ord(c)<32 and c not in '\n\r\t' for c in source)
        ast.parse(source,filename=str(path))
    print('UTF8/AST PASS: 2 owned ProtoFrame scorer/test files; no numerical imports')
    raise SystemExit(0)

from copy import deepcopy
import importlib.util
import itertools
import json
import numpy as np
import pytest

sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import score_d92_support_metric_joint_benchmark as score

OLD=['old_'+str(i) for i in range(6)]
NATIVE=list(reversed(OLD))
COMMIT='a'*40
CONFIG=dict(algorithm=deepcopy(score.FROZEN_ALGORITHM),support_metric_resources=deepcopy(score.FROZEN_RESOURCES))


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def lines(path,values):
    path.write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in values),encoding='utf-8')


def decision(old_scores,new_scores,g):
    def ls(values):
        values=np.asarray(values,dtype=np.float64);shift=values-values.max()
        return (shift-np.log(np.exp(shift).sum())).tolist()
    new=['new_'+str(i).zfill(2) for i in range(len(new_scores))]
    lo,ln=ls(old_scores),ls(new_scores);oi=int(np.argmax(old_scores));ni=int(np.argmax(new_scores))
    gap=g+lo[oi]-ln[ni];ow,nw=OLD[oi],new[ni]
    winner=ow if gap>0 else nw if gap<0 else min(ow,nw)
    cert=dict(schema=score.DECISION_SCHEMA,policy=score.DECISION_POLICY,new0_reuses_actual_B=False,
        old_classes=OLD,new_classes=new,old_raw_scores=old_scores,new_raw_scores=new_scores,gate_logit=g,
        old_log_probabilities=lo,new_log_probabilities=ln,old_winner=ow,new_winner=nw,group_gap=gap,prediction=winner)
    mapping={name:float(-np.logaddexp(0.,-g)+lo[i]) for i,name in enumerate(OLD)}
    mapping.update({name:float(-np.logaddexp(0.,g)+ln[i]) for i,name in enumerate(new)})
    return cert,[mapping[name] for name in sorted(OLD+new)]


def empty_work(**values):
    work=dict.fromkeys(score.WORK_KEYS,0);work.update(values);return work


def capsule_fixture(root,co,full=False):
    path=root/co;path.mkdir();(path/'splits').mkdir()
    matrix=dict(receivers=score.RECEIVERS[co],scenarios=score.SCENARIOS,ks=list(score.KS),
        new_counts=list(score.NEWS),support_seeds=score.SUPPORT_SEEDS)
    axes=[matrix[k] for k in ('receivers','scenarios','support_seeds','ks','new_counts')]
    if not full:axes=[axes[0][:1],axes[1][:1],axes[2][:1],axes[3],axes[4]]
    physical=[];positions={};truth={};splits=[];cid='synthetic_'+co
    def pos(pid):
        if pid not in positions:positions[pid]=len(physical);physical.append(pid)
        return positions[pid]
    for rx,scene,seed,k,new in itertools.product(*axes):
        classes=OLD+['new_'+str(i).zfill(2) for i in range(new)];support=[];labels=[];queries=[]
        for i,name in enumerate(classes):
            for j in range(k):support.append(pos('s_'+rx+'_'+scene+'_'+name+'_'+str(j)));labels.append(i)
            pid='q_'+rx+'_'+scene+'_'+name;queries.append(pos(pid))
            truth[pid]=dict(pool_role='query',receiver=rx,scene=scene,transmitter=name,old=name in OLD)
        sid='split_'+str(len(splits)).zfill(4)
        split=dict(split_id=sid,capsule_id=cid,protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',
            receiver=rx,scenario=scene,support_seed=seed,k=k,registered_classes=classes,
            support_indices=support,support_labels=labels,query_indices=queries)
        dump(path/'splits'/(sid+'.json'),split);splits.append(split)
    dump(path/'manifest.json',dict(capsule_id=cid,protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',split_count=len(splits)))
    np.savez(path/'received.npz',ids=np.asarray(physical),received=np.asarray([np.nan]))
    truth_path=root/(co+'_truth.json');dump(truth_path,truth)
    return dict(capsule=str(path),truth=str(truth_path),expected_capsule_id=cid,matrix=matrix,
        expected_split_count=score.COHORT_COUNTS[co]),truth


def prediction_rows(split,truth):
    records={s:[] for s in score.STREAMS};work=[];new=split['new_count']
    for pid in split['query_ids']:
        name=truth[pid]['transmitter'];bpred=name if name in OLD[:-1] else OLD[0]
        apred=name if name in OLD[:3] else OLD[4] if name==OLD[5] else OLD[5]
        aval=[float(c==apred) for c in NATIVE];bval=[float(c==bpred) for c in OLD]
        common=dict(split_id=split['split_id'],query_id=pid)
        a=dict(common,classes=NATIVE,scores=aval,prediction=apred)
        b=dict(common,classes=OLD,scores=bval,prediction=bpred)
        # R0 is deliberately distinct from native A and support-adapted B.
        rb=dict(common,classes=OLD,scores=[float(c==OLD[0]) for c in OLD],prediction=OLD[0])
        if new:
            nval=[float(c==name) for c in split['registered_classes'] if c not in OLD]
            cert,cval=decision(bval,nval,10. if name in OLD else -10.)
            c=dict(common,classes=split['registered_classes'],scores=cval,prediction=cert['prediction'])
            rc=dict(common,classes=split['registered_classes'],scores=[float(v==OLD[0]) for v in split['registered_classes']],prediction=OLD[0])
        else:
            c=deepcopy(b);rc=deepcopy(rb)
            cert=dict(schema=score.DECISION_SCHEMA,policy='EXACT_ACTUAL_B_REUSE',new0_reuses_actual_B=True,prediction=bpred)
        for stage,record in zip(score.STREAMS,(a,b,c,rb,rc)):records[stage].append(record)
        bw=dict(score_seconds=.001,single_record_all_registered_classes=True,actual_work=empty_work(raw_kernel_calls=1),
            operation_audits=[dict(operation='raw_kernel',audit={})])
        work.append(dict(common,B_work=bw,C_work=deepcopy(bw) if new else None,R0_score_work=None,
            R0_work_unavailable_reason='ORIGINAL_SCORE_API_HAS_NO_OPERATION_LEDGER',C_reused_B_scores=not bool(new),
            R0_C_reused_B_scores=not bool(new),C_structural_decision=cert,C_structural_prediction=c['prediction'],
            C_public_predict_calls=int(bool(new)),C_public_predict_seconds=.001 if new else 0.,
            C_public_predict_internal_work=None,C_public_predict_internal_work_unavailable_reason=(
                'PUBLIC_PREDICT_HAS_NO_WORK_AUDIT' if new else 'EXACT_B_REUSE_NO_EXTRA_CALL'),
            scope='ACTUAL_SINGLETON_SCORE_AND_SEPARATE_PUBLIC_PREDICT; NO_LABEL_OR_ROLE'))
    return records,work


def row_bundle(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True);root=tmp_path/'run';root.mkdir()
    cohort,truth=capsule_fixture(tmp_path,'rx1');_,splits=score.load_capsule_metadata(cohort['capsule'],cohort['expected_capsule_id'])
    rid='rx1_'+str(score.MODEL_SEEDS[0]);out=root/rid/'predictions';out.mkdir(parents=True);(out/'state_arrays').mkdir()
    row=dict(row_id=rid,cohort='rx1',expected_model_seed=score.MODEL_SEEDS[0],expected_checkpoint_sha256='1'*64,
        row_root=str(tmp_path/'source'),branch_features=str(tmp_path/'features'),ground_packet=str(tmp_path/'packet'),
        ground_summary=str(tmp_path/'geometry'),ground_summary_already_deployed=True,output_root=str(out.parent))
    binding=dict(run_id='synthetic_run',row_id=rid,release_commit=COMMIT,capsule_id=cohort['expected_capsule_id'],
        checkpoint_sha256='1'*64,model_seed=score.MODEL_SEEDS[0])
    source=dict(checkpoint_sha256='1'*64,model_seed=binding['model_seed'],source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',
        source_role_comparison='EXACT_MATCH',checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,
        cache_schema='d92_branch_received_features_v1',feature_contract=score.RAW_FEATURE_CONTRACT)
    ground=dict(path=row['ground_packet'],checkpoint_sha256='1'*64,model_seed=binding['model_seed'],ordered_classes=NATIVE,
        scale=30.,norm_eps=1e-4,packet_total_file_bytes=100,source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',feature_contract=dict(
            checkpoint_sha256='1'*64,cache_key='z_id',source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160))
    geometry=dict(path=row['ground_summary'],checkpoint_sha256='1'*64,model_seed=binding['model_seed'],already_deployed=True,
        input_role='FROZEN_PHASE1_CENTER_ONLY_GEOMETRY',source_examples=False,prototype_teacher_targets=False,checkpoint_loaded=False)
    geom_binding=dict(operation='load_existing_ground_center_fixed_proto_frame',frame_audit=dict(class_order=OLD,
        prototype_class_count=6,prototype_dimension=160,dictionary_coordinate_count=5,seconds=.1),
        ground_payload_audit=dict(already_deployed=True,total_file_bytes=100,incremental_transfer_bytes=0,
            source_samples_read=False,checkpoint_file_read=False,dense_bank_persisted=False,dequantized_domain_cache=False),
        component_read_seconds=.2,total_geometry_binding_seconds=.3,newly_generated_ground_statistics_bytes=0,
        **dict.fromkeys(('source_samples_read','source_per_record_features_read','checkpoint_loaded','residual_domain_reconstruction',
            'prototype_updated','prototype_teacher_targets','support_labels_read','query_read'),False))
    archive=[];metadata=[];records={s:[] for s in score.STREAMS};decisions=[];prep=[];stages=[];baseline=[]
    def ref(sid,state,numeric=None,key='final'):
        namespace=json.dumps(dict(run_id=binding['run_id'],row_id=rid,split_id=sid,
            scope='row_frozen_ground_geometry' if sid is None else 'query_benchmark_support_training',
            state=state,fold=None,trial=None,parent_k=None if sid is None else splits[sid]['k'],
            train_k=None if sid is None else splits[sid]['k']),sort_keys=True)
        relative='state_arrays/'+str(len(archive))+'.npz'
        numeric=dict(numeric or {},value=np.asarray([1.],dtype=np.float64),scalar=np.asarray(2.,dtype=np.float64),flag=np.asarray(True))
        np.savez_compressed(out/relative,**numeric)
        value=dict(key=key,path=relative,namespace=namespace,failed_numeric_state=False,
            file_bytes=(out/relative).stat().st_size,archive_seconds=0.,
            arrays={name:dict(shape=list(a.shape),dtype=str(a.dtype),nbytes=a.nbytes,all_finite=True,nonfinite_count=0) for name,a in numeric.items()})
        archive.append(value);return value
    basis_arrays=dict(original_dictionary_Q=np.zeros((160,5)),U=np.empty((160,0)),U_lower=np.empty((160,0)),
        U_upper=np.empty((160,0)),U_error_bounds=np.empty((160,0)),exact_column_indices=np.empty(0,dtype=np.int64),
        physical_gram=np.empty((0,0)),rank=np.asarray(0,dtype=np.int64))
    basis_ref=ref(None,'ROW_SUPPORT_METRIC_BASIS',basis_arrays,'basis')
    cert=dict(schema='d92_support_metric_basis_v1',shape=[160,5],rank=0,enclosure_bits=128,
        input_scope='EXACT_STORED_BINARY64_NOT_PRE_ROUNDING_PROTOTYPES',fraction_encoding='signed_hex_numerator_positive_hex_denominator',
        exact_state=dict(column_indices=[],orthogonal_residuals=[],norm_squared=[]),normalizations=[])
    dump(out/'basis_certificate.json',cert);cert_bytes=(out/'basis_certificate.json').stat().st_size
    ca=dict.fromkeys(score.BASIS_SUM+score.BASIS_MAX,0)
    ca.update(status='COMPLETE',exact_rank=0,enclosure_bits=128,max_integer_bits=65536,max_fraction_operations=65536,
        certificate_utf8_bytes=cert_bytes,numeric_array_payload_bytes=6400)
    cw=empty_work(basis_calls=1,**{'basis_'+k:ca[k] for k in score.BASIS_SUM+score.BASIS_MAX})
    cert_ref=dict(path='basis_certificate.json',file_bytes=cert_bytes,utf8_bytes=cert_bytes,archive_seconds=0.,
        scope='ORDINARY_METHOD_ARTIFACT_NOT_AUTHORIZATION')
    owner=dict(schema='d92_support_metric_row_basis_v1',method=score.METHOD,status='COMPLETE',row_basis_construction_count=1,
        input_role='FROZEN_GROUND_Q_ONLY_NO_SUPPORT_TEACHER',context=json.loads(basis_ref['namespace']),
        row_basis_ref=basis_ref,row_basis_rank=0,construction_audit=ca,basis_certificate=cert_ref,
        actual_work=cw,operation_audits=[dict(operation='basis',audit=ca)])
    dump(out/'row_basis.json',owner)
    ownership=dict(row_basis_ref=basis_ref,row_basis_rank=0,basis_certificate_ref=cert_ref)
    def candidate_arrays(s,old_only):
        classes=OLD if old_only else s['registered_classes'];ids=s['old_support_ids'] if old_only else s['support_ids']
        labels=np.asarray([classes.index(s['_support_label_mapping'][pid]) for pid in ids],dtype=np.int64)
        values=dict(basis_arrays,theta=np.empty(0),head_theta_padded=np.zeros(5),labels=labels)
        if old_only:values.update(alpha=np.zeros((len(ids),6)),intercept=np.zeros(6))
        else:
            q=s['new_count'];old=np.asarray([i for i,pid in enumerate(ids) if pid in s['old_support_ids']],dtype=np.int64)
            values.update(new_alpha=np.zeros((q*s['k'],q)),new_intercept=np.zeros(q),gate_alpha=np.zeros(len(ids)),
                gate_b=np.asarray(1.),gate_zeta=np.asarray(1e-4),gate_slacks=np.ones((6*s['k'],q)),lower_bounds=np.zeros((6*s['k'],q)),
                old_indices=old,gate_old_indices=old,gate_targets=np.isin(np.arange(len(ids)),old).astype(float))
        return values
    for sid,s in splits.items():
        ba=candidate_arrays(s,True);b,rb=ref(sid,'B_SUPPORT_METRIC',ba),ref(sid,'R0_B')
        if s['new_count']:
            with np.load(out/b['path'],allow_pickle=False) as saved:full_b={k:saved[k] for k in saved.files}
            c_arrays=candidate_arrays(s,False);c_arrays.update({'actual_B_'+k:v for k,v in full_b.items()})
            c,rc=ref(sid,'C_SUPPORT_METRIC_seq',c_arrays),ref(sid,'R0_C')
        else:c,rc=b,rb
        metadata.append(dict(s,**ownership,b_state_ref=b,c_state_ref=c,c_inherited_from_b_state_ref=b,c_reuses_b=not bool(s['new_count']),
            r0_b_state_ref=rb,r0_c_state_ref=rc,r0_c_reuses_b=not bool(s['new_count']),B_classes=OLD,C_classes=s['registered_classes'],
            R0_B_classes=OLD,R0_C_classes=s['registered_classes'],support_only_fit=True,query_fit_access=False,source_fit_access=False))
        coords=dict(run_id=binding['run_id'],row_id=rid,split_id=sid,scope='query_benchmark_support_training',fold=None,trial=None,
            parent_k=s['k'],train_k=s['k'])
        for state,r,old_only in (('B_SUPPORT_METRIC',b,True),('C_SUPPORT_METRIC_seq',c,False),('R0_B',rb,True),('R0_C',rc,False)):
            if not old_only and not s['new_count']:continue
            ids=s['old_support_ids'] if old_only else s['support_ids']
            if state.startswith('R0_'):
                audit=dict(status='COMPLETED',final_state_ref=r,training_physical_ids=ids,
                    final_fit=dict(factorization_calls=1,effective_degrees_of_freedom_extra_triangular_solves=2))
                v=dict(coords,state=state,mode='INDEPENDENT_R0_REFIT',audit=audit);baseline.append(v);stages.append(v)
            else:
                classes=OLD if old_only else s['registered_classes'];folds=[];fc=0 if s['k']==1 else min(3,s['k'])
                members={pid:i%fc for name in classes for i,pid in enumerate(sorted(x for x in ids if s['_support_label_mapping'][x]==name))} if fc else {}
                for j in range(fc):
                    ti=sorted(x for x in ids if members[x]!=j);hi=sorted(x for x in ids if members[x]==j)
                    oi=[x for x in ti if s['_support_label_mapping'][x] in OLD]
                    prior=None if old_only else ref(sid,'C_prepare',dict(basis_arrays,theta=np.empty(0),head_theta_padded=np.zeros(5),
                        K=np.eye(len(oi)),alpha=np.zeros((len(oi),6))), 'fold_'+str(j)+'_old_inner_prior')
                    folds.append(dict(fold=j,train_ids=ti,held_ids=hi,old_inner_train_ids=oi,prior_ref=prior))
                core_context=dict(coords,stage='B' if old_only else 'C_seq')
                pa=dict(context=core_context,training_physical_ids=ids,effective_parameter_rank=0,folds=folds,
                    full_prior_ref=None if old_only else b,basis_construction_charged_here=False,basis_audit=ca,
                    actual_work=empty_work(basis_binding_calls=1,basis_binding_physical_gram_evaluation_count=1),
                    operation_audits=[dict(operation='basis_binding',audit=dict(physical_gram_evaluation_count=1,wall_seconds=0.))])
                audit=dict(schema=score.FROZEN_ALGORITHM['schema'],method=score.METHOD,status='COMPLETED',final_head_complete=True,
                    context=core_context,config=CONFIG['algorithm'],limits=CONFIG['support_metric_resources'],classes=classes,training_physical_ids=ids,
                    mode='B' if old_only else 'C_seq',effective_parameter_rank=0,nominal_parameter_count=5,parameter_buffer_bytes=0,
                    no_held=s['k']==1,trainable_parameter_count=0,evidence_level=score.FROZEN_ALGORITHM['evidence_level'],complete_head_jvp_error_bound=None,
                    anchor_theta=[],theta=[],actual_updated_coordinate_count=0,deployment_numeric_state_bytes=64,
                    initial_objective=None,final_objective=None,trials=[],
                    final_state_ref=r,optimizer_steps=0,trial_count=0,operation_audits=[],resident_numeric_state_bytes=64,
                    final_prior_ref=None if old_only else b,
                    actual_work=empty_work())
                stages.append(dict(coords,**ownership,state=state,mode='B' if old_only else 'C_seq',training_physical_ids=ids,audit=audit))
                prep.append(dict(coords,**ownership,state='B_prepare' if old_only else 'C_prepare',audit=pa))
        r,w=prediction_rows(s,truth)
        for stage in score.STREAMS:records[stage]+=r[stage]
        decisions+=w
    for stage,name in score.STREAMS.items():lines(out/name,records[stage])
    lines(out/'predictions.jsonl',records['C']);lines(out/'query_score_work.jsonl',decisions)
    lines(out/'preparations.jsonl',prep);lines(out/'fit_stages.jsonl',stages)
    by_phase={}
    for r in archive:
        phase=json.loads(r['namespace'])['state'];v=by_phase.setdefault(phase,dict(file_count=0,file_bytes=0,numeric_array_bytes=0,archive_seconds=0.))
        v['file_count']+=1;v['file_bytes']+=r['file_bytes'];v['numeric_array_bytes']+=sum(a['nbytes'] for a in r['arrays'].values())
    archive_bytes=sum(r['file_bytes'] for r in archive);numeric_bytes=sum(a['nbytes'] for r in archive for a in r['arrays'].values())
    dump(out/'state_manifest.json',dict(schema='d92_support_metric_joint_state_archive_v1',method=score.METHOD,status='COMPLETE',files=archive,file_count=len(archive),
        total_file_bytes=archive_bytes,numeric_array_bytes=numeric_bytes,archive_seconds=0.,by_phase=by_phase))
    public=dict(binding,schema=score.PREDICTION_SCHEMA,method=score.METHOD,algorithm=CONFIG['algorithm'],support_metric_resources=CONFIG['support_metric_resources'],
        row_basis_construction_status='PENDING_EXPLICIT_FROZEN_Q_FACTORY',row_basis_namespace=owner['context'],
        ordered_ground_classes=NATIVE,old_classes=OLD,source_identity=source,ground_packet_identity=ground,
        ground_geometry_identity=geometry,ground_geometry_binding=geom_binding,ground_summary_already_deployed=True,
        run_binding=dict(prediction_output_root=str(out),capsule=cohort['capsule'],**{k:row[k] for k in ('row_root','branch_features','ground_packet','ground_summary')}),
        splits=list(splits.values()),split_count=len(splits),query_record_count=len(decisions),query_fit_access=False,source_fit_access=False,truth_read=False,
        checkpoint_loaded=False,encoder_called=False,cross_split_adapted_state_reuse=False,query_role_used=False,query_count_used_for_decision=False,
        view_aggregation=False,technical_query_chunk_size=1,nominal_adapter_parameter_count=5,prototype_geometry_role='FROZEN_REFERENCE_ONLY_NOT_TEACHER',
        C_decision_policy=score.DECISION_POLICY,C_decision_certificate_schema=score.DECISION_SCHEMA,A_tie_policy='first_original_native_head_column',
        B_C_tie_policy='physical_class_id_ascending',fit_scope='CURRENT_SPLIT_LEGAL_SUPPORT_ONLY',query_inference_scope='ONE_RECEIVED_RECORD_ALL_REGISTERED_COLUMNS')
    dump(out/'startup.json',dict(public,config=CONFIG))
    pw=score._merge_work([v['audit']['actual_work'] for v in prep]);sw=score._merge_work([v['audit']['actual_work'] for v in stages if not v['state'].startswith('R0_')])
    qw=score._merge_work([v[k]['actual_work'] for v in decisions for k in ('B_work','C_work') if v[k] is not None])
    calls=sum(v['C_public_predict_calls'] for v in decisions);n=len(decisions);stage_count=len(prep)
    resources=dict(actual_core_work=score._merge_work([cw,pw,sw,qw]),work_aggregation='SUM/MAX',counter_scope='COMPLETED_ROW_BASIS_PREP_STAGE_SCORE',
        state_archive_file_bytes=archive_bytes,state_archive_numeric_bytes=numeric_bytes,
        original_R0_training_audits=baseline,original_R0_completed_training_counters=dict(completed_head_fits=len(baseline),factorization_calls=len(baseline),triangular_solve_calls=4*len(baseline)),
        query_score_calls=dict(A=n,B=n,C=calls,R0_B=n,R0_C=calls),query_score_seconds=dict(A=.01,B=.02,C=.03,R0_B=.04,R0_C=.05),
        C_public_predict_calls=calls,C_public_predict_seconds=sum(v['C_public_predict_seconds'] for v in decisions),
        C_public_predict_internal_work=None,C_public_predict_internal_work_unavailable_reason='PUBLIC_PREDICT_HAS_NO_WORK_AUDIT')
    marker=dict(public,status='COMPLETE',splits=metadata,
        row_basis_ref=basis_ref,row_basis_rank=0,row_basis_audit=owner,basis_certificate=cert_ref,
        row_basis='row_basis.json',row_basis_construction_status='COMPLETE',
        row_basis_construction_count=1,row_basis_actual_work=cw,peak_effective_adapter_rank=0,
        streams={s:dict(path=name,record_count=len(records[s]),file_bytes=(out/name).stat().st_size) for s,name in score.STREAMS.items()},
        compatibility_predictions=dict(path='predictions.jsonl',alias_of='C',file_bytes=(out/'predictions.jsonl').stat().st_size),
        query_score_work=dict(path='query_score_work.jsonl',record_count=n,file_bytes=(out/'query_score_work.jsonl').stat().st_size),
        completed_split_count=len(splits),actual_stage_count=stage_count,actual_baseline_stage_count=stage_count,state_manifest='state_manifest.json',
        candidate_preparation_count=stage_count,candidate_stage_count=stage_count,final_candidate_head_fit_count=stage_count,
        optimizer_steps=0,trial_count=0,accepted_trial_count=0,rejected_trial_count=0,ggn_step_count=0,ggn_parameter_direction_count=0,
        peak_resident_numeric_state_bytes=64,resources=resources,actual_work=resources['actual_core_work'],preparation_actual_work=pw,
        stage_actual_work=sw,score_actual_work=qw,work_aggregation='SUM/MAX',actual_work_aggregation={k:'MAX' if k in score.PEAK_COUNTERS else 'SUM' for k in score.WORK_KEYS})
    dump(out/'predictions_complete.json',marker)
    return dict(row=row,co=cohort,root=out,binding=binding,truth=truth,marker=marker,public=public,config=CONFIG)


def load_one(case):
    return score.load_fixed_predictions(predictions=case['root'],capsule=case['co']['capsule'],config=CONFIG,expected_binding=case['binding'])


def test_literal_five_streams_native_A_distinct_R0_lineage_new0_and_parent_metrics(tmp_path):
    case=row_bundle(tmp_path);fixed=load_one(case)
    assert set(fixed['streams'])==set(score.STREAMS)
    for sid,s in fixed['splits'].items():
        p=score._truth_parent(fixed,sid,case['truth']);m=p['metrics']
        assert m['A_old_accuracy']==.5 and m['B_old_accuracy']==m['C_old_accuracy']==5/6
        assert m['R0_B_old_accuracy']==m['R0_C_old_accuracy']==1/6
        if s['new_count']:
            assert m['C_h']==pytest.approx(10/11) and m['R0_C_h']==0. and m['C_new_minus_R0']==1.
        else:assert m['C_new_accuracy'] is m['C_h'] is m['C_abs_new_old_gap'] is m['C_h_minus_R0'] is None
        assert m['B_old_macro_f1']==pytest.approx((4+2/3)/6)


@pytest.mark.parametrize('change',['inherit','namespace','source','geometry_teacher','geometry_path','runtime','limits','missing_R0',
    'duplicate','work_order','C_classes','forbidden_truth','ledger_SUM','ledger_MAX','R0_EDF','failed'])
def test_corruption_and_partial_reject_without_truth(tmp_path,monkeypatch,change):
    case=row_bundle(tmp_path);root=case['root'];marker=score.read(root/'predictions_complete.json')
    if change=='inherit':marker['splits'][1]['c_inherited_from_b_state_ref']=marker['splits'][0]['b_state_ref']
    elif change=='namespace':marker['splits'][1]['c_state_ref']['namespace']='{}'
    elif change=='source':marker['source_identity']['target_access_before_freeze']=True
    elif change=='geometry_teacher':marker['ground_geometry_binding']['prototype_teacher_targets']=True
    elif change=='geometry_path':marker['ground_geometry_identity']['path']='wrong'
    elif change=='runtime':marker['release_commit']='c'*40
    elif change=='limits':marker['support_metric_resources']['max_newton_iterations']+=1
    elif change=='ledger_SUM':marker['stage_actual_work']['ridge_calls']+=1
    elif change=='ledger_MAX':marker['actual_work']['gate_forward_peak_factor_buffer_bytes']+=1
    elif change=='R0_EDF':marker['resources']['original_R0_completed_training_counters']['triangular_solve_calls']-=2
    elif change=='failed':dump(root/'technical_failure.json',dict(status='FAILED'))
    elif change=='work_order':
        path=root/'query_score_work.jsonl';v=[json.loads(x) for x in path.read_text().splitlines()];v[0],v[1]=v[1],v[0]
        lines(path,v);marker['query_score_work']['file_bytes']=path.stat().st_size
    else:
        stage='R0_C' if change=='missing_R0' else 'C';path=root/score.STREAMS[stage]
        v=[json.loads(x) for x in path.read_text().splitlines()]
        if change=='missing_R0':v.pop()
        elif change=='duplicate':v[1]=deepcopy(v[0])
        elif change=='C_classes':v[0]['classes'].pop()
        else:v[0]['truth']='old_0'
        lines(path,v);marker['streams'][stage]['file_bytes']=path.stat().st_size
    dump(root/'predictions_complete.json',marker);original=score.read
    def guarded(path):
        assert not str(path).endswith('_truth.json'),'No prefix or invalid-row truth access'
        return original(path)
    monkeypatch.setattr(score,'read',guarded)
    with pytest.raises(ValueError):load_one(case)


@pytest.mark.parametrize('change',['missing_npz','shape','dtype','nonfinite','archive_totals',
    'undeclared_npz','wrong_C_stage_prior','wrong_C_preparation_prior','missing_C_stage_prior'])
def test_actual_numeric_archive_and_actual_C_inheritance_reject_before_truth(tmp_path,monkeypatch,change):
    case=row_bundle(tmp_path);root=case['root'];manifest=score.read(root/'state_manifest.json')
    ref=manifest['files'][0];path=root/ref['path']
    if change=='missing_npz':path.unlink()
    elif change in ('shape','dtype','nonfinite'):
        value=(np.asarray([1.,2.]) if change=='shape' else np.asarray([1],dtype=np.int32) if change=='dtype' else np.asarray([np.nan]))
        np.savez_compressed(path,value=value,scalar=np.asarray(2.),flag=np.asarray(True))
        # Keep declared file size current so the regression reaches the actual
        # array shape/dtype/finite check rather than only its physical byte gate.
        ref['file_bytes']=path.stat().st_size;dump(root/'state_manifest.json',manifest)
    elif change=='archive_totals':
        manifest['numeric_array_bytes']+=1;dump(root/'state_manifest.json',manifest)
    elif change=='undeclared_npz':np.savez_compressed(root/'state_arrays'/'undeclared.npz',value=np.asarray([1.]))
    else:
        is_prep=change=='wrong_C_preparation_prior';p=root/('preparations.jsonl' if is_prep else 'fit_stages.jsonl')
        records=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
        actual=next(v for v in records if v['state']==('C_prepare' if is_prep else 'C_SUPPORT_METRIC_seq'))
        key='full_prior_ref' if is_prep else 'final_prior_ref'
        if change=='missing_C_stage_prior':actual['audit'].pop(key)
        else:
            marker=score.read(root/'predictions_complete.json')
            other=next(s['b_state_ref'] for s in marker['splits'] if s['split_id']!=actual['split_id'])
            actual['audit'][key]=other
        lines(p,records)
    original=score.read
    def guarded(p):
        assert not str(p).endswith('_truth.json'),'Actual state/prior failure must precede truth'
        return original(p)
    monkeypatch.setattr(score,'read',guarded)
    with pytest.raises(ValueError):load_one(case)


def full_virtual_case(tmp_path,monkeypatch):
    """All 2400 literal parents exercise orchestration; row parser tested above.

    Avoid duplicating hundreds of MB of zero operation fields in this test.
    The loader is a literal fixed-input oracle here, never a fitting oracle.
    """
    template=row_bundle(tmp_path/'template');base=load_one(template)
    root=tmp_path/'full-run';root.mkdir();cohorts={};truths={};meta={};rows=[];fixed={};lanes={}
    for co in score.COHORT_COUNTS:
        cohorts[co],truths[co]=capsule_fixture(tmp_path,co,full=True)
        _,meta[co]=score.load_capsule_metadata(cohorts[co]['capsule'],cohorts[co]['expected_capsule_id'])
    for co,seed in itertools.product(score.COHORT_COUNTS,score.MODEL_SEEDS):
        rid=co+'_'+str(seed);row=dict(template['row'],row_id=rid,cohort=co,expected_model_seed=seed,
            expected_checkpoint_sha256=('1' if seed==score.MODEL_SEEDS[0] else '2')*64,output_root=str(root/rid),
            ground_packet='packet_'+str(seed),ground_summary='geometry_'+str(seed));rows.append(row)
        binding=dict(template['binding'],row_id=rid,capsule_id=cohorts[co]['expected_capsule_id'],model_seed=seed,checkpoint_sha256=row['expected_checkpoint_sha256'])
        streams={s:{} for s in score.STREAMS};metadata=[]
        for sid,s in meta[co].items():
            r,_=prediction_rows(s,truths[co])
            for stage in streams:streams[stage][sid]=r[stage]
            oldref=dict(synthetic='literal-current-B',split_id=sid,row_id=rid)
            metadata.append(dict(s,b_state_ref=oldref,c_state_ref=oldref,r0_b_state_ref=oldref,r0_c_state_ref=oldref))
        public=dict(template['public'],**binding,splits=list(meta[co].values()),run_binding=dict(
            prediction_output_root=str(Path(row['output_root'])/'predictions'),capsule=cohorts[co]['capsule'],
            **{k:row[k] for k in ('row_root','branch_features','ground_packet','ground_summary')}))
        marker=dict(template['marker'],**binding,splits=metadata,run_binding=public['run_binding'])
        value=dict(base,binding=binding,splits=meta[co],streams=streams,startup=public,complete=marker,
            predictions_root=str(Path(row['output_root'])/'predictions'))
        fixed[Path(value['predictions_root']).resolve()]=value
        lanes[rid]=dict(status='COMPLETE',release_commit=COMMIT,expected_model_seed=seed,expected_checkpoint_sha256=row['expected_checkpoint_sha256'],
            expected_capsule_id=cohorts[co]['expected_capsule_id'],output_root=row['output_root'],prediction_output=value['predictions_root'],marker=marker,
            source_paths={k:public['run_binding'][k] for k in ('row_root','branch_features','ground_packet','ground_summary','capsule')},
            preflight=dict(public,status='SUPPORT_METRIC_QUERY_PREFLIGHT_COMPLETE'))
    spec=dict(schema='d92_support_metric_joint_query_benchmark_v1',run_id='synthetic_run',group_id='synthetic_group',
        code=dict(commit='b'*40),execution=dict(remote_run_root=str(root)),benchmark=dict(config=CONFIG,cohorts=cohorts),rows=rows)
    startup=dict(schema=spec['schema'],status='SUPPORT_METRIC_QUERY_BENCHMARK_STARTED',run_id=spec['run_id'],group_id=spec['group_id'],
        runtime_commit=COMMIT,code_commit=spec['code']['commit'],resolved_spec=spec,rows={k:dict(status='PENDING') for k in lanes})
    complete=dict(startup,status='SUPPORT_METRIC_QUERY_BENCHMARK_PREDICTIONS_COMPLETE',rows=lanes,row_count=4,completed_row_count=4,
        declared_episode_count=2400,completed_episode_count=2400,
        all_predictions_fixed=True,truth_read=False,scorer_invoked=False,automatic_retry=False)
    dump(root/'startup.json',startup);dump(root/'complete.json',complete)
    validations=[]
    def loader(**kw):
        key=Path(kw['predictions']).resolve()
        validations.append(str(key));return deepcopy(fixed[key])
    monkeypatch.setattr(score,'load_fixed_predictions',loader)
    return dict(spec=spec,root=root,fixed=fixed,validations=validations,truth_paths={v['truth'] for v in cohorts.values()})


def test_all_declared_rows_full_matrix_independent_reread_before_truth_and_metrics(tmp_path,monkeypatch):
    case=full_virtual_case(tmp_path,monkeypatch);original=score.read;truth_reads=[]
    def guarded(path):
        if str(path) in case['truth_paths']:
            assert len(case['validations'])==8;truth_reads.append(str(path))
        return original(path)
    monkeypatch.setattr(score,'read',guarded)
    result=score.score_benchmark(spec=case['spec'],output=tmp_path/'score.json')
    assert result['row_count']==4 and result['parent_count']==2400 and len(truth_reads)==2
    assert result['prediction_streams_independently_reread'] and not result['model_called']
    assert all(p['metrics']['A_old_accuracy']==.5 for p in result['parents'])
    assert all(p['metrics']['R0_C_old_accuracy']==1/6 for p in result['parents'])
    assert len(result['statistics']['by_k_new_count'])>0 and len(result['statistics']['by_cohort'])>0
    assert all(v['sample_standard_deviation']==0. for v in result['statistics']['model_seed_mean_sd']['by_k_new_count'] if v['mean'] is not None)
    assert score.read(tmp_path/'score.json')==result


@pytest.mark.parametrize('change',['global_partial','last_row_matrix','changed_reread','subset'])
def test_full_run_failures_cannot_join_truth(tmp_path,monkeypatch,change):
    case=full_virtual_case(tmp_path,monkeypatch);original=score.read
    if change=='global_partial':
        value=original(case['root']/'complete.json');value['completed_row_count']=3;dump(case['root']/'complete.json',value)
    elif change=='last_row_matrix':case['fixed'][list(case['fixed'])[-1]]['splits'].popitem()
    elif change=='subset':case['spec']['rows'].pop()
    else:
        loader=score.load_fixed_predictions
        def changed(**kw):
            value=loader(**kw)
            if len(case['validations'])==8:value['streams']['B'][next(iter(value['splits']))][0]['prediction']='old_5'
            return value
        monkeypatch.setattr(score,'load_fixed_predictions',changed)
    def guarded(path):
        assert str(path) not in case['truth_paths'],'No partial truth join';return original(path)
    monkeypatch.setattr(score,'read',guarded)
    with pytest.raises(ValueError):score.score_benchmark(spec=case['spec'],output=tmp_path/'never.json')
    assert not (tmp_path/'never.json').exists()


def test_rounded_scores_keep_real_public_C_and_lexical_group_tie():
    b=[0.,1e-16,0.,0.,0.,0.];cert,values=decision(b,[0.,0.],100.);classes=sorted(OLD+cert['new_classes'])
    c=dict(classes=classes,scores=values,prediction='old_1');record=dict(classes=OLD,scores=b,prediction='old_1')
    assert classes[int(np.argmax(values))]=='old_0'
    score.verify_structured_decision(cert,c,record,dict(new_count=2,old_classes=OLD,registered_classes=classes))
    cert,values=decision([0.]*6,[0.]*6,0.)
    assert cert['group_gap']==0. and cert['prediction']=='new_00'
    score.verify_structured_decision(cert,dict(classes=sorted(OLD+cert['new_classes']),scores=values,prediction='new_00'),
        dict(classes=OLD,scores=[0.]*6,prediction=OLD[0]),dict(new_count=6,old_classes=OLD,registered_classes=sorted(OLD+cert['new_classes'])))


@pytest.mark.parametrize('key',['old_winner','new_winner','group_gap','gate_logit','prediction','old_raw_scores'])
def test_structured_certificate_tampering_rejected(key):
    cert,values=decision([0.,1.,0.,0.,0.,0.],[1.,0.],2.);b=dict(scores=[0.,1.,0.,0.,0.,0.])
    c=dict(scores=values,prediction=cert['prediction']);classes=sorted(OLD+cert['new_classes'])
    if key.endswith('winner') or key=='prediction':cert[key]='impossible'
    elif key=='old_raw_scores':cert[key][0]+=.1
    else:cert[key]+=1.
    with pytest.raises(ValueError):score.verify_structured_decision(cert,c,b,dict(new_count=2,old_classes=OLD,registered_classes=classes))


def test_SUM_MAX_unknown_work_no_training_imports_and_ids_plus_declared_numeric_states_only(tmp_path,monkeypatch):
    case=row_bundle(tmp_path);original=np.load;members=[]
    class IDsOnly:
        def __init__(self,path,**kw):self.data=original(path,**kw);self.ids_only=Path(path).name=='received.npz'
        def __enter__(self):return self
        def __exit__(self,*args):self.data.close()
        @property
        def files(self):return self.data.files
        def __getitem__(self,key):
            if self.ids_only:assert key=='ids';members.append(key)
            else:assert key in self.data.files
            return self.data[key]
    monkeypatch.setattr(np,'load',IDsOnly);fixed=load_one(case);assert members==['ids']
    resources=fixed['complete']['resources'];r2=deepcopy(resources);r2['actual_core_work']['gate_forward_peak_factor_buffer_bytes']=32
    result=score.aggregate_training_resources([dict(resources=resources),dict(resources=r2)])
    assert result['actual_core_work']['basis_binding_calls']==2*resources['actual_core_work']['basis_binding_calls']
    assert result['actual_core_work']['gate_forward_peak_factor_buffer_bytes']==32
    assert result['R0_query_internal_work'] is result['C_public_predict_internal_work'] is None
    tree=ast.parse((ROOT/'tools/score_d92_support_metric_joint_benchmark.py').read_text(encoding='utf-8'))
    modules={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
    assert not any(any(k in v for k in ('cvsrffi','torch','evaluate_','export_','run_','packet')) for v in modules)


def test_independent_algorithm_and_work_literals_match_frozen_source_AST():
    tree=ast.parse((ROOT/'code/cvsrffi/d92_support_metric_joint_local_ridge.py').read_text(encoding='utf-8'))
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SCHEMA','METHOD')}
    call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='FROZEN_CONFIG')
    # The new method explicitly reuses only this old geometry definition.
    # Resolve its literal in source, never import a fitting module or skip it.
    assert any(isinstance(n,ast.ImportFrom) and n.level==1 and any(
        a.name=='d92_proto_frame_joint_local_ridge' and a.asname=='geometry' for a in n.names) for n in tree.body)
    geometry=ast.parse((ROOT/'code/cvsrffi/d92_proto_frame_joint_local_ridge.py').read_text(encoding='utf-8'))
    geometry_config=next(n.value for n in geometry.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='FROZEN_CONFIG')
    kernel=ast.literal_eval(next(v.value for v in geometry_config.keywords if v.arg=='kernel'))
    actual={}
    for v in call.keywords:
        if isinstance(v.value,ast.Subscript):
            assert v.arg=='kernel' and isinstance(v.value.value,ast.Attribute)
            assert isinstance(v.value.value.value,ast.Name) and v.value.value.value.id=='geometry'
            assert v.value.value.attr=='FROZEN_CONFIG' and ast.literal_eval(v.value.slice)=='kernel'
            actual[v.arg]=kernel
        else:actual[v.arg]=constants[v.value.id] if isinstance(v.value,ast.Name) else ast.literal_eval(v.value)
    assert actual==score.FROZEN_ALGORITHM
    p=ast.parse((ROOT/'code/cvsrffi/d92_proto_frame_primitives.py').read_text(encoding='utf-8'))
    count=next(ast.literal_eval(n.value) for n in p.body if isinstance(n,ast.Assign) and n.targets[0].id=='_COUNT_KEYS')
    assert count==score.PRIMITIVE_COUNTS
    gate=ast.parse((ROOT/'code/cvsrffi/d92_group_barrier_gate.py').read_text(encoding='utf-8'))
    ledger=next(n for n in gate.body if isinstance(n,ast.FunctionDef) and n.name=='_ledger').body[0].value
    assert tuple(v.arg for v in ledger.keywords if not v.arg.startswith('peak_'))+('wall_seconds',)==score.GATE_SUM
    # Exporter's four native branches are distinct from its separately named
    # FFT96 sketch. Check the exact literal source contract without importing
    # its native/checkpoint dependencies or relaxing the scorer comparison.
    exporter=ast.parse((ROOT/'tools/export_d92_branch_support_features.py').read_text(encoding='utf-8'))
    assignments={n.targets[0].id:n.value for n in exporter.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name)}
    branches=ast.literal_eval(assignments['BRANCHES'])
    contract={v.arg:list(branches) if v.arg=='branch_keys' else ast.literal_eval(v.value)
        for v in assignments['FEATURE_CONTRACT'].keywords}
    assert contract==score.RAW_FEATURE_CONTRACT and 'fft' not in contract['branch_keys']


def test_seconds_bounds_nonfinite_unknown_keys_and_exclusive_output(tmp_path):
    values=[.1]*1000;score._seconds_sum_readback(sum(values),values,'public')
    with pytest.raises(ValueError):score._seconds_sum_readback(sum(values)+.0001,values,'public')
    for bad in (True,float('nan'),-.1):
        with pytest.raises(ValueError):score._validate_work(empty_work(ridge_seconds=bad),'bad')
    value=empty_work();value.pop('ridge_calls')
    with pytest.raises(ValueError):score._validate_work(value,'missing')
    output=tmp_path/'preserve.json';output.write_text('original',encoding='utf-8')
    with pytest.raises(ValueError):score.score_benchmark(spec={},output=output)
    assert output.read_text()=='original'
    config=deepcopy(CONFIG);config['support_metric_resources']['max_newton_iterations']=101
    with pytest.raises(ValueError):score.validate_config(config)


@pytest.mark.parametrize('fault',['owner_rank','owner_scope','certificate_bytes','certificate_rank',
    'construction_recharged','binding_gram_missing','basis_SUM','rank_direction_count','split_basis','full_head_claim'])
def test_support_metric_basis_rank_ownership_and_actual_work_reject_before_truth(tmp_path,monkeypatch,fault):
    case=row_bundle(tmp_path);root=case['root'];marker=score.read(root/'predictions_complete.json')
    if fault=='owner_rank':
        owner=score.read(root/'row_basis.json');owner['row_basis_rank']=1;dump(root/'row_basis.json',owner)
    elif fault=='owner_scope':
        owner=score.read(root/'row_basis.json');owner['context']['scope']='wrong';dump(root/'row_basis.json',owner)
    elif fault in ('certificate_bytes','certificate_rank'):
        cert=score.read(root/'basis_certificate.json')
        if fault=='certificate_bytes':cert['extra']='unbound'
        else:cert['rank']=1
        dump(root/'basis_certificate.json',cert)
    elif fault=='basis_SUM':marker['row_basis_actual_work']['basis_calls']=2
    elif fault=='rank_direction_count':marker['ggn_parameter_direction_count']=5
    elif fault=='split_basis':marker['splits'][0]['row_basis_rank']=1
    else:
        path=root/('fit_stages.jsonl' if fault=='full_head_claim' else 'preparations.jsonl')
        records=score._jsonl(path);record=next(v for v in records if not v['state'].startswith('R0_'))
        if fault=='full_head_claim':record['audit']['complete_head_jvp_error_bound']=0.
        elif fault=='construction_recharged':record['audit']['basis_construction_charged_here']=True
        else:record['audit']['actual_work']['basis_binding_physical_gram_evaluation_count']=0
        lines(path,records)
    dump(root/'predictions_complete.json',marker);original=score.read
    def no_truth(path):
        assert not str(path).endswith('_truth.json');return original(path)
    monkeypatch.setattr(score,'read',no_truth)
    with pytest.raises((ValueError,KeyError)):load_one(case)


@pytest.mark.parametrize('key',list(score.FROZEN_RESOURCES))
def test_all_six_explicit_resource_guards(key):
    for value in (True,0,-1,1.5):
        config=deepcopy(CONFIG);config['support_metric_resources'][key]=value
        with pytest.raises(ValueError):score.validate_config(config)
    config=deepcopy(CONFIG);config['support_metric_resources'].pop(key)
    with pytest.raises(ValueError):score.validate_config(config)


@pytest.mark.parametrize('new,k,rank0',[(2,1,False),(2,3,False),(0,3,True)])
def test_real_preflight_predict_archive_and_completion_with_separate_timing_scopes(tmp_path,monkeypatch,new,k,rank0):
    # Reuse only the owner's literal synthetic factory. The real numerical
    # producer/core/archive executes when root runs this test, never by scorer.
    path=ROOT/'tests/test_d92_support_metric_joint_benchmark_entry.py'
    module_spec=importlib.util.spec_from_file_location('literal_proto_producer_fixture',path)
    literal=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(literal)
    case=literal.fixture(tmp_path,monkeypatch,new=new,k=k,rank0=rank0)
    preflight=literal.entry.preflight(**case.args);marker=literal.entry.predict(**case.args)
    # This fixture is one declared split: row parser/real ABI and timing are
    # checked here; complete frozen 2400-parent matrix is covered separately.
    initial=preflight['splits'][0];capsule=dict(capsule_id=case.args['expected_capsule_id'])
    internal=deepcopy(initial);internal['_support_label_mapping']={str(case.ids[i]):case.split['registered_classes'][label]
        for i,label in zip(case.split['support_indices'],case.split['support_labels'])}
    monkeypatch.setattr(score,'load_capsule_metadata',lambda *a:(capsule,{initial['split_id']:internal}))
    row=dict(row_id=case.args['row_id'],cohort='literal',expected_model_seed=17,expected_checkpoint_sha256=literal.SHA,
        output_root=str(Path(case.args['output']).parent),**{k:case.args[k] for k in ('row_root','branch_features','ground_packet','ground_summary','ground_summary_already_deployed')})
    # Production output path itself is explicit; parent-of-predictions rule.
    co=dict(capsule=case.args['capsule'],expected_capsule_id=case.args['expected_capsule_id'],expected_split_count=1,
        matrix=dict(receivers=[initial['receiver']],scenarios=[initial['scenario']],support_seeds=[initial['support_seed']],ks=[k],new_counts=[new]))
    spec=dict(schema='d92_support_metric_joint_query_benchmark_v1',run_id=case.args['run_id'],rows=[row],benchmark=dict(config=CONFIG,cohorts={'literal':co}))
    # Make the independent clock difference deterministic without changing any
    # path, bytes, class order or frozen role used for identity comparison.
    preflight=deepcopy(preflight)
    preflight['ground_geometry_binding']['component_read_seconds']+=1.
    preflight['ground_geometry_binding']['frame_audit']['seconds']+=1.
    assert preflight['ground_geometry_binding']!=marker['ground_geometry_binding']
    checked=score.validate_row_output(case.args['output'],spec,row,literal.COMMIT,preflight)
    assert checked==marker
    changed=deepcopy(preflight);changed['ground_geometry_identity']['path']='incorrect'
    with pytest.raises(ValueError):score.validate_row_output(case.args['output'],spec,row,literal.COMMIT,changed)


def metric_certificate():
    """Production scalar-objective and step results on literal saved-score/JVP input.
    No head, source data, cache or fixture selected from observed results."""
    from cvsrffi import d92_proto_frame_primitives as primitive
    from cvsrffi import d92_support_metric_step as step
    scores=np.array([[.4,-.1],[-.2,.6],[.1,.2],[-.3,.4]])
    labels=np.array([0,1,0,1]);J=np.array([[[.1,.2],[-.2,.1]],[[.3,-.1],[.2,.1]],
        [[.1,.4],[-.1,.2]],[[-.2,.3],[.2,-.1]]])
    padded=np.zeros((4,2,5));padded[...,:2]=J
    rms=primitive.class_balanced_rmsce(scores=scores,labels=labels,score_jacobian=padded)
    gram=np.array([[1.,.125],[.125,1.5]])
    initial=dict(scores=scores,labels=labels,probabilities=rms.arrays['probabilities'],score_jacobian=J,
        gradient=rms.gradient[:2],curvature=rms.curvature[:2,:2],physical_gram=gram)
    result=step.solve_support_metric_step(gradient=initial['gradient'],ggn=initial['curvature'],physical_gram=gram,
        score_jvp=J,probabilities=initial['probabilities'],labels=labels,max_secular_iterations=128)
    saved=dict(score_jvp=J,probabilities=initial['probabilities'],labels=labels,**result.arrays)
    return initial,saved,result.audit_dict(),float(rms.loss)

def test_independent_saved_objective_Fisher_metric_and_KKT_readback():
    initial,saved,audit,loss=metric_certificate()
    score.verify_objective_arrays(initial,2,['a','b','c','d'],['old','new'],
        dict(a='old',b='new',c='old',d='new'),loss)
    direction,M=score.verify_metric_direction(initial,saved,audit,2)
    assert direction.shape==(2,) and float(direction@M@direction)<=.25
    assert audit['direction_error_bound'] is None and not audit['precise_original_problem_certificate']
    assert not np.array_equal(saved['fisher'],initial['curvature'])

@pytest.mark.parametrize('fault',['fisher','gram_identity','ggn_is_fisher','multiplier','direction','label_binding','certificate'])
def test_saved_metric_evidence_tampering_rejected(fault):
    initial,saved,audit,loss=metric_certificate()
    initial={k:v.copy() for k,v in initial.items()};saved={k:v.copy() for k,v in saved.items()}
    if fault=='fisher':saved['fisher'][0,0]+=.05
    elif fault=='gram_identity':saved['physical_gram']=np.eye(2)
    elif fault=='ggn_is_fisher':saved['ggn']=saved['fisher'].copy()
    elif fault=='multiplier':saved['multiplier']=np.asarray(-1.)
    elif fault=='direction':saved['direction']=saved['direction']+.25
    elif fault=='label_binding':saved['labels']=1-saved['labels']
    else:audit['precise_original_problem_certificate']=True
    with pytest.raises(ValueError):score.verify_metric_direction(initial,saved,audit,2)

def test_basis_U_enclosure_actual_Gram_and_rank_zero():
    U=np.zeros((160,2));U[0,0]=1.;U[1,1]=1.
    arrays=dict(original_dictionary_Q=np.zeros((160,5)),U=U,U_lower=U.copy(),U_upper=U.copy(),
        U_error_bounds=np.zeros_like(U),exact_column_indices=np.array([0,2]),physical_gram=U.T@U)
    score.verify_basis_arrays(arrays,2)
    changed=deepcopy(arrays);changed['physical_gram'][0,0]+=1e-5
    with pytest.raises(ValueError,match='Gram'):score.verify_basis_arrays(changed,2)
    changed=deepcopy(arrays);changed['U_upper'][0,0]=.9
    with pytest.raises(ValueError,match='enclosure'):score.verify_basis_arrays(changed,2)
    zero={k:(np.zeros((160,0)) if k in ('U','U_lower','U_upper','U_error_bounds') else v) for k,v in arrays.items()}
    zero.update(exact_column_indices=np.array([],dtype=int),physical_gram=np.zeros((0,0)))
    score.verify_basis_arrays(zero,0)

def test_basis_construction_SUM_and_step_peak_MAX_are_different_scopes():
    ops=[dict(operation='basis',audit=dict(fraction_operations_attempted=12,max_observed_integer_bits=400)),
         dict(operation='basis_binding',audit=dict(wall_seconds=.2,physical_gram_evaluation_count=1)),
         dict(operation='support_metric_step',audit=dict(factorization_attempts=2,peak_single_factor_input_output_bytes=144)),
         dict(operation='support_metric_step',audit=dict(factorization_attempts=3,peak_single_factor_input_output_bytes=96))]
    work=score._empty_work()
    for item in ops:
        op=item['operation'];work[op+'_calls']+=1
        for key,value in item['audit'].items():
            full=op+'_'+key
            if full in score.WORK_KEYS:work[full]=max(work[full],value) if full in score.WORK_MAX else work[full]+value
    assert score.verify_work(dict(actual_work=work,operation_audits=ops))==work
    assert work['basis_fraction_operations_attempted']==12 and work['basis_binding_calls']==1
    assert work['support_metric_step_factorization_attempts']==5
    assert work['support_metric_step_peak_single_factor_input_output_bytes']==144
    corrupt=work.copy();corrupt['support_metric_step_peak_single_factor_input_output_bytes']=240
    with pytest.raises(ValueError):score.verify_work(dict(actual_work=corrupt,operation_audits=ops))
