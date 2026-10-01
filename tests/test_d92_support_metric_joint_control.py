"""Literal control fixtures; no real configuration, dataset, packet or run is read.

--static-check parses only the four owned UTF-8 files before importing the
production modules. Root owns pytest, isolated imports and all external actions.
"""
if __name__ == '__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:] != ['--static-check']:
        raise SystemExit('Only --static-check is available as a script')
    root = Path(__file__).resolve().parents[1]
    for relative in ('tools/run_d92_support_metric_joint_probe.py',
                     'tools/preflight_d92_support_metric_joint_probe.py',
                     'tools/publish_d92_support_metric_joint_probe.py',
                     'tests/test_d92_support_metric_joint_control.py'):
        path = root / relative
        raw = path.read_bytes()
        ast.parse(raw.decode('utf-8', errors='strict'), filename=str(path))
        print('STATIC_UTF8_AST_OK', relative, len(raw))
    raise SystemExit(0)

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_d92_support_metric_joint_probe as supervisor
import preflight_d92_support_metric_joint_probe as preflight
import publish_d92_support_metric_joint_probe as publisher

RESOURCES = dict(max_newton_iterations=100, max_line_search_trials=64,
    max_factor_buffer_bytes=167772160, max_integer_bits=65536,
    max_fraction_operations=65536, max_secular_iterations=128)


def fixed_spec():
    """Entire fixture is a source literal, never an existing configs file."""
    spec = dict(schema=supervisor.SCHEMA, group_id='d92-support-metric-joint-support',
        run_id='SOURCE_ONLY_SYNTHETIC_SUPPORT_METRIC_CONTROL',
        spec_path='configs/SOURCE_ONLY_SYNTHETIC_SUPPORT_METRIC.json',
        code=dict(commit='a'*40, preparation_parent_commit='a'*40,
            cwd='/synthetic/releases/support_metric', environment='/synthetic/python'),
        execution=dict(cpu_lanes=2, blas_threads_per_lane=2, launch_owner='root',
            remote_run_root='/synthetic/runs/support_metric'),
        permissions=dict(query_use='none; query IQ/labels/truth/scores never read',
            source_samples=False, source_per_record_features=False,
            old_target_scores_for_adaptation=False, cross_run_result_tuning=False,
            adapted_state_reuse='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse',
            summary_inputs=True,
            prototype_use='frozen_center_feature_dictionary_only; no_teacher_targets_or_extra_support'),
        probe=dict(algorithm=deepcopy(supervisor.PROBE_CONFIG),
            support_metric_resources=deepcopy(RESOURCES), channel=deepcopy(supervisor.CHANNEL),
            candidate='R_SUPPORT_METRIC_seq', controls=['R0'], query_access=False,
            reuse_support_cache=True, interpretation='support_joint_pilot_no_direct_promotion',
            cohorts={}), rows=[])
    old=['1','10','11','2','20','3']
    for ci,cohort in enumerate(('rx3','rx1')):
        seed=71+ci
        receivers=[cohort,cohort+'_other']
        pairs=[[receivers[0],'practical_mid'],[receivers[1],'practical_low_urban']]
        selection=dict(receiver_scenes=pairs, support_seed=seed,
            ks=[1,5,10,20], new_counts=[0,2,5,10,20], splits=[])
        for rx,scene in pairs:
            for k in selection['ks']:
                for new in selection['new_counts']:
                    selection['splits'].append(dict(split_id=f'{cohort}_{rx}_{scene}_{k}_{new}',
                        receiver=rx,scenario=scene,k=k,new_count=new,support_seed=seed,
                        registered_classes=old+[f'new_{i:02}' for i in range(new)]))
        matrix=dict(receivers=receivers,scenarios=list(supervisor.SCENARIOS),
            ks=list(selection['ks']),new_counts=list(selection['new_counts']),support_seeds=[seed])
        spec['probe']['cohorts'][cohort]=dict(matrix=matrix,selection=selection,
            expected_split_count=120,selected_split_count=40,
            capsule_id='residual-noeq-SOURCE_ONLY_'+cohort,
            capsule='/synthetic/inputs/capsule_'+cohort)
        for mi,model in enumerate((2026092701,2026092702)):
            row_id=f'{cohort}_model_{model}'
            spec['rows'].append(dict(row_id=row_id,cohort=cohort,
                output_root=spec['execution']['remote_run_root']+'/'+row_id,
                method=supervisor.METHOD,initial_step_size=1.0,lr=None,
                support_features='/synthetic/inputs/cache_'+row_id,
                expected_checkpoint_sha256=('b' if mi else 'c')*64,
                ground_packet=f'/synthetic/inputs/native_A_model_{model}',
                ground_summary=f'/synthetic/inputs/centers_model_{model}',
                ground_summary_already_deployed=False,
                seeds=dict(model=model,split=None,data=None,augmentation=None,
                    support=seed,evaluation=None)))
    for cohort in spec['probe']['cohorts']:
        spec['probe']['cohorts'][cohort]['evaluator_config']=supervisor.evaluator_config(spec,cohort)
    spec['probe']['budget']=supervisor.budget_for_spec(spec)
    return spec


def test_literal_full_matrix_budget_counts_row_basis_once_and_path_binding():
    spec=supervisor.validate_spec(fixed_spec())
    total=spec['probe']['budget']['total']
    assert total['exact']['episodes']==160
    assert total['exact']['sequence_paths']==1800
    assert total['exact']['candidate_preparation_count']==3240
    assert total['exact']['final_candidate_head_fit_count']==3240
    assert total['exact']['row_basis_construction_count']==total['exact']['basis_calls']==4
    assert total['exact']['basis_binding_calls']==3240
    assert total['exact']['basis_binding_physical_gram_evaluation_count']==3240
    assert total['maximum']['support_metric_step_calls']==648
    assert total['maximum']['support_metric_step_fisher_construction_attempts']==648
    assert total['maximum']['support_metric_step_secular_iteration_count']==648*128
    assert total['maximum']['trial_count']==648*12
    for row in spec['rows']:
        exact=spec['probe']['budget']['per_row'][row['row_id']]['exact']
        assert exact['row_basis_construction_count']==exact['basis_calls']==1
        assert exact['basis_binding_calls']==exact['basis_binding_physical_gram_evaluation_count']==810


@pytest.mark.parametrize('fault', ['missing_resource','bool_resource','zero_resource','secular129',
    'old_initial_step','old_method','query','source','prototype_teacher','missing_summary',
    'incomplete_matrix','duplicate_row','owner','lanes','blas','budget'])
def test_literal_spec_rejects_direct_method_permission_and_matrix_faults(fault):
    spec=fixed_spec()
    if fault=='missing_resource':spec['probe']['support_metric_resources'].pop('max_integer_bits')
    elif fault=='bool_resource':spec['probe']['support_metric_resources']['max_fraction_operations']=True
    elif fault=='zero_resource':spec['probe']['support_metric_resources']['max_integer_bits']=0
    elif fault=='secular129':spec['probe']['support_metric_resources']['max_secular_iterations']=129
    elif fault=='old_initial_step':spec['rows'][0]['initial_step_size']=.125
    elif fault=='old_method':spec['probe']['algorithm']['damping']='I5'
    elif fault=='query':spec['probe']['query_access']=True
    elif fault=='source':spec['permissions']['source_samples']=True
    elif fault=='prototype_teacher':spec['permissions']['prototype_use']='prototype_teacher'
    elif fault=='missing_summary':spec['rows'][0].pop('ground_summary')
    elif fault=='incomplete_matrix':spec['probe']['cohorts']['rx3']['selection']['splits'].pop()
    elif fault=='duplicate_row':spec['rows'][1]['row_id']=spec['rows'][0]['row_id']
    elif fault=='owner':spec['execution']['launch_owner']='another_worker'
    elif fault=='lanes':spec['execution']['cpu_lanes']=3
    elif fault=='blas':spec['execution']['blas_threads_per_lane']=3
    else:spec['probe']['budget']['total']['exact']['basis_calls']+=1
    with pytest.raises((ValueError,KeyError)):
        supervisor.validate_spec(spec)


def test_new_command_and_resolved_config_preserve_complete_explicit_resources():
    spec=fixed_spec();row=spec['rows'][0]
    argv=supervisor.command(spec,row,'d'*40)
    assert argv[2].endswith('tools/evaluate_d92_support_metric_joint_probe.py')
    assert argv[argv.index('--ground-summary')+1]==row['ground_summary']
    assert argv[argv.index('--ground-summary-already-deployed')+1]=='false'
    assert '--query' not in argv and '--truth' not in argv
    config=supervisor.evaluator_config(spec,row['cohort'])
    assert set(config)=={'algorithm','producer_matrix','selection','support_metric_resources'}
    assert config['support_metric_resources']==RESOURCES
    assert config['algorithm']['initial_step']==1.
    assert config['algorithm']['max_updates']==1
    assert config['algorithm']['max_trials']==12


def test_preflight_is_existing_metadata_binding_not_numeric_revalidation():
    program=preflight.remote_script(fixed_spec())
    compile(program,'source-only-unexecuted-support-preflight','exec')
    assert 'np.load' not in program and 'torch.load' not in program
    assert 'VALIDATED_ONCE' in program
    assert 'ground_numeric_values_read=False' in program
    assert 'CURRENT_MATCHED_SOURCE_ONLY_V2' in program
    assert 'support_metric_resources' in program
    assert 'query_access=False' in program


def test_preflight_fake_transport_preserves_exclusive_evidence_and_no_retry(tmp_path):
    spec=fixed_spec();calls=[]
    answer=dict(status='VERIFIED',run_id=spec['run_id'],query_access=False,
        source_sample_access=False,feature_values_read=False,
        support_metric_resources=RESOURCES)
    def fake(argv,**kwargs):
        calls.append((argv,kwargs))
        return SimpleNamespace(returncode=0,stdout=json.dumps(answer),stderr='')
    output=tmp_path/'synthetic_preflight.json'
    assert preflight.preflight(spec,ssh_host='N607',ssh_config='SOURCE_ONLY_CONFIG',
        remote_python='/synthetic/python',output=output,run_fn=fake)==answer
    assert len(calls)==1
    with pytest.raises(FileExistsError):
        preflight.preflight(spec,ssh_host='N607',ssh_config='SOURCE_ONLY_CONFIG',
            remote_python='/synthetic/python',output=output,run_fn=fake)
    assert len(calls)==1


def test_release_is_explicit_runtime_closure_and_normal_user_only():
    spec=fixed_spec();paths=publisher.release_paths(spec)
    assert len(paths)==len(set(paths))
    for file in ('d92_support_metric_basis.py','d92_support_metric_step.py',
                 'd92_support_metric_joint_local_ridge.py'):
        assert 'code/cvsrffi/'+file in paths
    assert 'tools/evaluate_d92_support_metric_joint_probe.py' in paths
    assert 'tools/publish_d92_support_metric_joint_probe.py' in paths
    assert spec['spec_path'] in paths
    assert not any('analyze_d92_support_metric' in p for p in paths)
    assert not any(p.startswith(('automation_reports/','experiment_registry/')) for p in paths)
    assert not any(p.endswith(('.npz','.pth','.bin')) for p in paths)
    assert publisher.ENDPOINTS==('run_d92_support_metric_joint_probe',
        'evaluate_d92_support_metric_joint_probe','preflight_d92_support_metric_joint_probe')
    assert publisher.EXPECTED_REMOTE_USER=='szu2070436088'
    compile(publisher.REMOTE,'source-only-unexecuted-publisher','exec')
    compile(publisher.IMPORT_PROGRAM,'source-only-unexecuted-import-check','exec')
    assert 'Ordinary N607 user required' in publisher.REMOTE
    assert 'automatic_retry=False' in publisher.REMOTE


def marker_fixture(tmp_path,rank=2):
    from evaluate_d92_support_metric_joint_probe import COUNTERS,WORK_SUM_KEYS,WORK_MAX_KEYS
    spec=fixed_spec();row=deepcopy(spec['rows'][0]);co=spec['probe']['cohorts'][row['cohort']]
    cache=tmp_path/'synthetic-cache';cache.mkdir();row['support_features']=str(cache)
    physical=[dict(split_id=s['split_id'],support_ids=['source-only-'+str(i)])
        for i,s in enumerate(co['selection']['splits'])]
    plan=dict(capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],splits=physical)
    (cache/'support_splits.json').write_text(json.dumps(plan),encoding='utf-8')
    output=tmp_path/'synthetic-output';output.mkdir()
    marker=dict.fromkeys(COUNTERS,0)
    marker.update(spec['probe']['budget']['per_row'][row['row_id']]['exact'])
    keys=tuple(WORK_SUM_KEYS)+tuple(WORK_MAX_KEYS)
    phases={name:dict.fromkeys(keys,0) for name in
        ('row_basis_actual_work','preparation_actual_work','stage_actual_work','score_actual_work')}
    phases['row_basis_actual_work']['basis_calls']=1
    phases['preparation_actual_work']['basis_binding_calls']=810
    phases['preparation_actual_work']['basis_binding_physical_gram_evaluation_count']=810
    steps=162 if rank else 0
    phase=phases['stage_actual_work']
    phase.update(support_metric_step_calls=steps,
        support_metric_step_fisher_construction_attempts=steps,
        support_metric_step_fisher_constructions_completed=steps,
        support_metric_step_spectral_check_attempts=2*steps,
        support_metric_step_spectral_checks_completed=2*steps,
        support_metric_step_factorization_attempts=2*steps,
        support_metric_step_factorizations_completed=2*steps,
        support_metric_step_triangular_calls=3*steps,
        support_metric_step_triangular_calls_completed=3*steps,
        support_metric_step_triangular_rhs_columns=(rank+3)*steps,
        support_metric_step_triangular_rhs_elements=rank*(rank+3)*steps,
        support_metric_step_triangular_dense_work_units=rank**2*(rank+3)*steps,
        support_metric_step_secular_evaluation_count=steps)
    work={key:max(part[key] for part in phases.values()) if key in WORK_MAX_KEYS else
        sum(part[key] for part in phases.values()) for key in keys}
    marker.update(work)
    marker.update(ggn_step_count=steps,ggn_parameter_direction_count=rank*steps,
        peak_effective_adapter_rank=rank)
    # Controller reads archive metadata/existence/bytes only. Its independent
    # analyzer later verifies the actual NPZ values; this fixture is no model.
    state_dir=output/'state_arrays';state_dir.mkdir()
    state_file=state_dir/'00000000.npz';state_file.write_bytes(b'SOURCE_ONLY_METADATA_FIXTURE')
    context=dict(run_id=spec['run_id'],row_id=row['row_id'],split_id=None,
        scope='row_frozen_ground_geometry',fold=None,trial=None,parent_k=None,train_k=None,
        state='ROW_SUPPORT_METRIC_BASIS')
    ref=dict(path='state_arrays/00000000.npz',key='basis',namespace=json.dumps(context),
        arrays={},file_bytes=state_file.stat().st_size,failed_numeric_state=False)
    certificate=dict(schema='d92_support_metric_basis_v1',shape=[160,5],rank=rank,
        input_scope='EXACT_STORED_BINARY64_NOT_PRE_ROUNDING_PROTOTYPES')
    raw=(json.dumps(certificate)+'\n').encode('utf-8')
    (output/'basis_certificate.json').write_bytes(raw)
    marker.update(schema=supervisor.SCHEMA,method=supervisor.METHOD,
        run_id=spec['run_id'],row_id=row['row_id'],capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],
        release_commit='d'*40,query_rows_used=0,source_rows_used=0,truth_read=False,
        support_metric_resources=RESOURCES,pid=12345,status=supervisor.STATUS,
        workload_complete=True,algorithm=supervisor.PROBE_CONFIG,
        selection=co['selection'],producer_matrix=co['matrix'],
        ground_packet=row['ground_packet'],ground_summary=row['ground_summary'],
        ground_summary_already_deployed=False,
        selected_support_physical_ids={v['split_id']:v['support_ids'] for v in physical},
        state_archive_file_count=1,work_aggregation='SUM/MAX',actual_work=work,**phases,
        actual_work_aggregation={key:'MAX' if key in WORK_MAX_KEYS else 'SUM' for key in keys},
        row_basis_ref=ref,row_basis_rank=rank,row_basis='row_basis.json',
        basis_certificate=dict(path='basis_certificate.json',utf8_bytes=len(raw),
            file_bytes=len(raw),archive_seconds=0.,scope='ORDINARY_METHOD_ARTIFACT_NOT_AUTHORIZATION'))
    basis_audit=dict(schema='d92_support_metric_row_basis_v1',method=supervisor.METHOD,
        status='COMPLETE',row_basis_ref=ref,row_basis_rank=rank,
        basis_certificate=marker['basis_certificate'],row_basis_construction_count=1,
        construction_audit=dict(status='COMPLETE',exact_rank=rank,
            max_integer_bits=RESOURCES['max_integer_bits'],
            max_fraction_operations=RESOURCES['max_fraction_operations']),
        input_role='FROZEN_GROUND_Q_ONLY_NO_SUPPORT_TEACHER',context=context,
        actual_work=phases['row_basis_actual_work'],operation_audits=[])
    marker['row_basis_audit']=basis_audit
    (output/'row_basis.json').write_text(json.dumps(basis_audit),encoding='utf-8')
    startup=dict(marker,config=supervisor.evaluator_config(spec,row['cohort']),
        blas_environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),
        cuda_visible_devices='')
    (output/'startup.json').write_text(json.dumps(startup),encoding='utf-8')
    compact=[dict(s,query_rows_used=0,source_rows_used=0,row_basis_ref=ref,row_basis_rank=rank)
        for s in co['selection']['splits']]
    (output/'compact.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in compact),encoding='utf-8')
    manifest=dict(status='COMPLETE',schema='d92_support_metric_joint_state_archive_v1',
        method=supervisor.METHOD,file_count=1,files=[ref])
    (output/'state_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return spec,row,output,marker


@pytest.mark.parametrize('rank',[0,2,5])
def test_completed_marker_closes_row_basis_four_phases_and_actual_rank(tmp_path,rank):
    spec,row,output,marker=marker_fixture(tmp_path,rank)
    path=output/'probe_complete.json';path.write_text(json.dumps(marker),encoding='utf-8')
    assert supervisor.verify_marker(path,spec,row,commit='d'*40,expected_pid=12345)==marker


@pytest.mark.parametrize('fault',['phase_missing','phase_sum','negative_phase','peak_as_sum',
    'basis_recharge','basis_binding','nominal_direction','zero_steps','fisher_missing',
    'basis_ref','basis_rank','basis_certificate_bytes','basis_certificate_rank',
    'state_file_missing','duplicate_parent','integer_limit','secular_limit',
    'basis_owner_missing','basis_owner_mixed','basis_resource_mixed','rank_peak'])
def test_marker_rejects_partial_or_mixed_basis_and_real_work_metadata(tmp_path,fault):
    from evaluate_d92_support_metric_joint_probe import WORK_SUM_KEYS,WORK_MAX_KEYS
    spec,row,output,marker=marker_fixture(tmp_path)
    if fault=='phase_missing':marker['row_basis_actual_work'].pop(WORK_SUM_KEYS[0])
    elif fault=='phase_sum':marker['score_actual_work'][WORK_SUM_KEYS[0]]+=1
    elif fault=='negative_phase':marker['score_actual_work'][WORK_SUM_KEYS[0]]=-1
    elif fault=='peak_as_sum':marker['actual_work_aggregation'][WORK_MAX_KEYS[0]]='SUM'
    elif fault=='basis_recharge':marker['preparation_actual_work']['basis_calls']=1
    elif fault=='basis_binding':marker['preparation_actual_work']['basis_binding_calls']-=1
    elif fault=='nominal_direction':marker['ggn_parameter_direction_count']=5*marker['support_metric_step_calls']
    elif fault=='zero_steps':marker['support_metric_step_calls']=marker['ggn_step_count']=0
    elif fault=='fisher_missing':marker['support_metric_step_fisher_constructions_completed']-=1
    elif fault=='basis_ref':marker['row_basis_ref']=dict(marker['row_basis_ref'],path='other.npz')
    elif fault=='basis_rank':marker['row_basis_rank']=3
    elif fault=='basis_certificate_bytes':marker['basis_certificate']['utf8_bytes']+=1
    elif fault=='basis_certificate_rank':
        p=output/'basis_certificate.json';data=json.loads(p.read_text(encoding='utf-8'))
        data['rank']=3;p.write_text(json.dumps(data)+'\n',encoding='utf-8')
        marker['basis_certificate']['file_bytes']=marker['basis_certificate']['utf8_bytes']=p.stat().st_size
    elif fault=='state_file_missing':(output/'state_arrays/00000000.npz').unlink()
    elif fault=='duplicate_parent':
        p=output/'compact.jsonl';lines=p.read_text(encoding='utf-8').splitlines()
        lines[-1]=lines[0];p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    elif fault=='integer_limit':marker['basis_max_observed_integer_bits']=RESOURCES['max_integer_bits']+1
    elif fault=='secular_limit':marker['support_metric_step_secular_iteration_count']=162*128+1
    elif fault=='basis_owner_missing':(output/'row_basis.json').unlink()
    elif fault=='basis_owner_mixed':
        p=output/'row_basis.json';value=json.loads(p.read_text(encoding='utf-8'))
        value['context']['row_id']='another_row';p.write_text(json.dumps(value),encoding='utf-8')
    elif fault=='basis_resource_mixed':
        marker['row_basis_audit']['construction_audit']['max_integer_bits']+=1
        (output/'row_basis.json').write_text(json.dumps(marker['row_basis_audit']),encoding='utf-8')
    else:marker['peak_effective_adapter_rank']=5
    path=output/'probe_complete.json';path.write_text(json.dumps(marker),encoding='utf-8')
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):
        supervisor.verify_marker(path,spec,row,commit='d'*40,expected_pid=12345)


def test_publisher_independent_landing_readback_and_no_admin():
    spec=fixed_spec();commit='d'*40
    binding=dict(runtime_commit=commit)
    launch=dict(runtime_commit=commit,cwd=spec['code']['cwd'],
        run_root=spec['execution']['remote_run_root'],launch_owner='root',
        remote_user=publisher.EXPECTED_REMOTE_USER,pid=321)
    startup=dict(runtime_commit=commit,code_commit=spec['code']['commit'],
        resolved_spec=spec,run_id=spec['run_id'],group_id=spec['group_id'],
        schema=supervisor.SCHEMA,status=supervisor.STARTED,pid=321,
        truth_read=False,scorer_invoked=False)
    value=dict(metadata=dict(launch=launch,startup=startup))
    assert publisher.verify_landing(value,spec,binding)['status']=='VERIFIED'
    value['metadata']['launch']['remote_user']='root'
    with pytest.raises(ValueError):publisher.verify_landing(value,spec,binding)


def test_run_does_not_retry_failed_rows_or_reuse_existing_root(tmp_path):
    spec=fixed_spec();root=tmp_path/'exclusive_synthetic_run'
    spec['execution']['remote_run_root']=str(root)
    for row in spec['rows']:row['output_root']=str(root/row['row_id'])
    # Windows temp paths are local test transport; production validation remains
    # strict POSIX source/output separation. No declared input is opened.
    calls=[]
    bindings=dict(cache_bindings=[dict(row_id=r['row_id'],binding='VERIFIED') for r in spec['rows']])
    def fake_preflight(spec,**kwargs):return bindings
    def fail_launch(argv,log,cwd,started):
        calls.append(argv);started(7000+len(calls))
        raise RuntimeError('literal synthetic row process failure')
    # This test concerns the orchestration failure path, separately from the
    # independently tested literal spec validator.
    original=supervisor.validate_spec
    try:
        supervisor.validate_spec=lambda value:value
        with pytest.raises(RuntimeError):
            supervisor.run(spec,'d'*40,launch_fn=fail_launch,preflight_fn=fake_preflight)
        assert len(calls)==4
        complete=json.loads((root/'complete.json').read_text(encoding='utf-8'))
        assert complete['status']=='FAILED' and complete['completed_rows']==0
        assert complete['workload_complete'] is False and complete['automatic_retry'] is False
        assert all(v['automatic_retry'] is False for v in complete['rows'].values())
        with pytest.raises(ValueError):
            supervisor.run(spec,'d'*40,launch_fn=fail_launch,preflight_fn=fake_preflight)
        assert len(calls)==4
    finally:
        supervisor.validate_spec=original
