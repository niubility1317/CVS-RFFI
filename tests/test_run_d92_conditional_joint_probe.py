"""Per-row structural accounting and supervisor metadata, synthetic only."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from test_prepare_d92_conditional_joint_probe import spec
import run_d92_conditional_joint_probe as run
import evaluate_d92_conditional_joint_probe as entry


def marker(s,row):
    co=s['probe']['cohorts'][row['cohort']];counts=dict.fromkeys(entry.COUNTERS,0)
    counts.update(s['probe']['budget']['per_row'][row['row_id']]['exact'])
    counts.update(ajlr_preparation_count=counts['candidate_preparation_count'],ajlr_stage_count=counts['candidate_stage_count'],
        final_head_fit_count=counts['candidate_stage_count'])
    counts['head_fit_count']=counts['baseline_head_fit_count']+counts['final_head_fit_count']
    return dict(counts,status=run.STATUS,schema=run.SCHEMA,method=run.METHOD,run_id=s['run_id'],row_id=row['row_id'],
        capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],
        algorithm=run.PROBE_CONFIG,selection=co['selection'],producer_matrix=co['matrix'],query_rows_used=0,source_rows_used=0,truth_read=False)


@pytest.mark.parametrize('nrows,pairs',[(1,1),(3,1),(2,2)])
def test_budget_derived_from_declared_rows_not_global_four(nrows,pairs):
    s=spec(nrows,pairs);run.validate_spec(s);budget=s['probe']['budget']
    assert budget['total']['exact']['episodes']==nrows*pairs*20
    assert budget['total']['exact']['sequence_paths']==nrows*pairs*225
    assert budget['total']['exact']['baseline_head_fit_count']==nrows*pairs*405
    assert budget['total']['maximum']['factorization_count']>budget['total']['maximum']['head_fit_count']
    assert budget['total']['maximum']['spectral_diagnostic_count']==3*budget['total']['maximum']['projection_factorization_count']
    for key,total in budget['total']['exact'].items():assert total==sum(v['exact'][key] for v in budget['per_row'].values())


def test_marker_counts_allow_two_conditional_factors_and_reject_bad_bindings(tmp_path):
    s=spec();row=s['rows'][0];value=marker(s,row)
    value.update(final_factorization_count=2,projection_factorization_count=1,residual_factorization_count=1,
        spectral_diagnostic_count=3,factorization_count=2,completed_factorization_count=2,
        projection_triangular_solve_count=2,projection_triangular_rhs_count=6,
        projection_triangular_rhs_element_count=36,projection_triangular_dense_work_unit_count=216,
        residual_triangular_solve_count=2,residual_triangular_rhs_count=16,
        residual_triangular_rhs_element_count=32,residual_triangular_dense_work_unit_count=64)
    path=tmp_path/'marker.json';path.write_text(json.dumps(value),encoding='utf-8')
    assert run.verify_marker(path,s,row)==value
    for key,replacement in [('row_id','different'),('query_rows_used',1),('candidate_stage_count',1),('spectral_diagnostic_count',10**8),
            ('projection_triangular_solve_count',0),('residual_adjoint_factorization_count',1)]:
        changed=dict(value,**{key:replacement});path.write_text(json.dumps(changed),encoding='utf-8')
        with pytest.raises(ValueError):run.verify_marker(path,s,row)


def test_permissions_and_selection_and_output_boundaries():
    s=spec()
    for change in ('missing_cell','source','query','reuse','ball','output','config_overlap','config_backslash'):
        bad=deepcopy(s)
        if change=='missing_cell':bad['probe']['cohorts']['synthetic']['selection']['splits'].pop()
        elif change=='source':bad['permissions']['source_samples']=True
        elif change=='query':bad['probe']['query_access']=True
        elif change=='reuse':bad['permissions']['adapted_state_reuse']='cross_run'
        elif change=='ball':bad['probe']['algorithm']['coordinate_ball_radius']=.6
        elif change=='output':bad['rows'][0]['output_root']='/escaped'
        elif change=='config_overlap':
            co=bad['probe']['cohorts']['synthetic'];co['config_path']=bad['spec_path'];co['evaluation_config']=bad['code']['cwd']+'/'+bad['spec_path']
        elif change=='config_backslash':
            co=bad['probe']['cohorts']['synthetic'];co['config_path']='configs\\escaped.json';co['evaluation_config']=bad['code']['cwd']+'/'+co['config_path']
        with pytest.raises(ValueError):run.validate_spec(bad)
    command=run.command(s,s['rows'][0])
    assert command[command.index('--run-id')+1]==s['run_id']
    assert command[command.index('--row-id')+1]==s['rows'][0]['row_id']


def test_supervisor_uses_actual_row_count_and_exclusive_root(tmp_path):
    s=spec(3);s['execution']['remote_run_root']=str(tmp_path/'run').replace('\\','/')
    for row in s['rows']:row['output_root']=str(Path(s['execution']['remote_run_root'])/row['row_id'])
    # Metadata validation is separately tested with POSIX remote paths; simulate
    # only the local supervisor/marker I/O on the host's native temp directory.
    def fake_launch(argv,log,cwd,device):
        out=Path(argv[argv.index('--output')+1]);out.mkdir()
        row=next(r for r in s['rows'] if r['row_id']==argv[argv.index('--row-id')+1])
        (out/'probe_complete.json').write_text(json.dumps(marker(s,row)),encoding='utf-8')
    actual_read=run.read
    def read(path):
        if str(path)==s['probe']['cohorts']['synthetic']['evaluation_config']:
            co=s['probe']['cohorts']['synthetic'];return dict(algorithm=run.PROBE_CONFIG,producer_matrix=co['matrix'],selection=co['selection'])
        return actual_read(path)
    with patch.object(run,'validate_spec'),patch.object(run,'read',side_effect=read):
        run.run(s,'a'*40,launch_fn=fake_launch)
        with pytest.raises(FileExistsError):run.run(s,'a'*40,launch_fn=fake_launch)
    complete=json.loads((Path(s['execution']['remote_run_root'])/'complete.json').read_text())
    assert complete['model_rows']==complete['completed_rows']==3 and complete['episodes']==60
