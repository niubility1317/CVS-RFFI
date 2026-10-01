"""Source-only control checks; no declared data paths are opened or launched."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))

import run_d92_proto_frame_joint_probe as supervisor
import preflight_d92_proto_frame_joint_probe as preflight
import publish_d92_proto_frame_joint_probe as publisher


def fixed_spec():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/'configs/d92_proto_frame_joint_support_20261002.json').read_text(encoding='utf-8'))


def test_fixed_complete_matrix_and_one_update_budget():
    spec=supervisor.validate_spec(fixed_spec())
    budget=supervisor.budget_for_spec(spec)['total']
    assert budget['exact']['episodes']==160
    assert budget['exact']['sequence_paths']==1800
    assert budget['exact']['candidate_stage_count']==3240
    assert budget['exact']['final_candidate_head_fit_count']==3240
    assert budget['maximum']['optimizer_steps']==648
    assert budget['maximum']['ggn_step_count']==648
    assert budget['maximum']['ggn_parameter_direction_count']==3240
    assert budget['maximum']['trial_count']==7776


@pytest.mark.parametrize('fault', ['summary_denied','teacher','missing_ground','deployment_missing','method_budget'])
def test_explicit_geometry_permission_and_deployment_contract(fault):
    spec=deepcopy(fixed_spec())
    if fault=='summary_denied':spec['permissions']['summary_inputs']=False
    elif fault=='teacher':spec['permissions']['prototype_use']='prototype_teacher_targets'
    elif fault=='missing_ground':spec['rows'][0].pop('ground_summary')
    elif fault=='deployment_missing':spec['rows'][0].pop('ground_summary_already_deployed')
    else:spec['probe']['algorithm']['max_updates']=4
    with pytest.raises(ValueError):supervisor.validate_spec(spec)


def test_command_includes_actual_ground_inputs_and_frozen_resources_are_resolved():
    spec=fixed_spec();row=spec['rows'][0]
    argv=supervisor.command(spec,row,'c'*40)
    assert argv[argv.index('--ground-summary')+1]==row['ground_summary']
    assert argv[argv.index('--ground-summary-already-deployed')+1]=='false'
    assert '--ground-packet' in argv
    assert '--query' not in argv and '--truth' not in argv
    resolved=supervisor.evaluator_config(spec,row['cohort'])
    assert resolved['proto_frame_resources']==spec['probe']['proto_frame_resources']
    assert resolved['algorithm']['coordinates']==5


def test_preflight_program_is_metadata_only_and_checks_existing_ground_binding():
    program=preflight.remote_script(fixed_spec())
    compile(program,'synthetic-only-preflight-text','exec')
    assert 'np.load' not in program and 'torch.load' not in program
    assert 'ground_numeric_values_read=False' in program
    assert 'CURRENT_MATCHED_SOURCE_ONLY_V2' in program
    assert 'Frozen reference geometry provenance/schema mismatch' in program


def test_release_closure_has_distinct_candidate_and_no_experiment_results():
    spec=fixed_spec();paths=publisher.release_paths(spec)
    assert len(paths)==len(set(paths))
    assert 'code/cvsrffi/d92_proto_frame_primitives.py' in paths
    assert 'code/cvsrffi/d92_proto_frame_ground_geometry.py' in paths
    assert 'code/cvsrffi/d92_proto_frame_joint_local_ridge.py' in paths
    assert spec['spec_path'] in paths
    assert not any(path.startswith(('automation_reports/','experiment_registry/')) for path in paths)
    assert all(not path.endswith(('.npz','.pth','.bin')) for path in paths)
    compile(publisher.REMOTE,'unexecuted-release-program','exec')


def marker_fixture(tmp_path):
    from evaluate_d92_proto_frame_joint_probe import COUNTERS,WORK_SUM_KEYS,WORK_MAX_KEYS
    spec=fixed_spec();row=deepcopy(spec['rows'][0]);co=spec['probe']['cohorts'][row['cohort']]
    cache=tmp_path/'synthetic-cache';cache.mkdir();row['support_features']=str(cache)
    rows=[]
    for index,selected in enumerate(co['selection']['splits']):
        rows.append(dict(split_id=selected['split_id'],support_ids=['synthetic-id-'+str(index)]))
    plan=dict(capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],splits=rows)
    (cache/'support_splits.json').write_text(json.dumps(plan),encoding='utf-8')
    output=tmp_path/'synthetic-output';output.mkdir()
    marker=dict.fromkeys(COUNTERS,0)
    marker.update(spec['probe']['budget']['per_row'][row['row_id']]['exact'])
    marker.update(schema=supervisor.SCHEMA,method=supervisor.METHOD,run_id=spec['run_id'],row_id=row['row_id'],
        capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],
        release_commit='c'*40,query_rows_used=0,source_rows_used=0,truth_read=False,
        proto_frame_resources=spec['probe']['proto_frame_resources'],pid=12345,status=supervisor.STATUS,
        workload_complete=True,algorithm=supervisor.PROBE_CONFIG,selection=co['selection'],producer_matrix=co['matrix'],
        ground_packet=row['ground_packet'],ground_summary=row['ground_summary'],
        ground_summary_already_deployed=row['ground_summary_already_deployed'],
        selected_support_physical_ids={r['split_id']:r['support_ids'] for r in rows},
        state_archive_file_count=0,work_aggregation='SUM/MAX')
    keys=tuple(WORK_SUM_KEYS)+tuple(WORK_MAX_KEYS)
    marker['actual_work']=dict.fromkeys(keys,0)
    for field in ('preparation_actual_work','stage_actual_work','score_actual_work'):
        marker[field]=dict.fromkeys(keys,0)
    marker['actual_work_aggregation']={key:'MAX' if key in WORK_MAX_KEYS else 'SUM' for key in keys}
    startup=dict(marker,config=supervisor.evaluator_config(spec,row['cohort']),
        blas_environment=dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),cuda_visible_devices='')
    (output/'startup.json').write_text(json.dumps(startup),encoding='utf-8')
    compact=[dict(selected,query_rows_used=0,source_rows_used=0) for selected in co['selection']['splits']]
    (output/'compact.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in compact),encoding='utf-8')
    (output/'state_manifest.json').write_text(json.dumps(dict(status='COMPLETE',schema='d92_proto_frame_joint_state_archive_v1',method=supervisor.METHOD,file_count=0)),encoding='utf-8')
    return spec,row,output,marker


def test_completed_marker_closes_declared_phases_and_physical_metadata(tmp_path):
    spec,row,output,marker=marker_fixture(tmp_path)
    path=output/'probe_complete.json';path.write_text(json.dumps(marker),encoding='utf-8')
    assert supervisor.verify_marker(path,spec,row,commit='c'*40,expected_pid=12345)==marker


@pytest.mark.parametrize('fault',['phase_sum','phase_missing','peak_declared_as_sum','budget'])
def test_completed_marker_rejects_inconsistent_measured_work(tmp_path,fault):
    from evaluate_d92_proto_frame_joint_probe import WORK_SUM_KEYS,WORK_MAX_KEYS
    spec,row,output,marker=marker_fixture(tmp_path)
    if fault=='phase_sum':marker['stage_actual_work'][WORK_SUM_KEYS[0]]=1
    elif fault=='phase_missing':marker['score_actual_work'].pop(WORK_SUM_KEYS[0])
    elif fault=='peak_declared_as_sum':marker['actual_work_aggregation'][WORK_MAX_KEYS[0]]='SUM'
    else:marker['final_candidate_head_fit_count']-=1
    path=output/'probe_complete.json';path.write_text(json.dumps(marker),encoding='utf-8')
    with pytest.raises(ValueError):supervisor.verify_marker(path,spec,row,commit='c'*40,expected_pid=12345)
