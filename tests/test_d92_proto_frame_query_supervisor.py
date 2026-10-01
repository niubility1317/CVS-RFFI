"""Literal control-contract regressions; no data, weights, SSH or actual launch."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import run_d92_proto_frame_joint_benchmark as control
from cvsrffi.d92_proto_frame_joint_local_ridge import FROZEN_CONFIG


def literal_spec():
    run='/synthetic/runs/proto-frame';release='/synthetic/releases/proto-frame'
    cohorts={name:dict(capsule='/synthetic/input/'+name+'/capsule',truth='/synthetic/score-only/'+name+'/truth.json',
        expected_capsule_id='opaque-'+name,matrix=dict(control.MATRIX_BASE,receivers=control.RECEIVERS[name]),
        expected_split_count=900 if name=='rx3' else 300) for name in ('rx3','rx1')}
    rows=[]
    for co in ('rx3','rx1'):
        for seed,letter in ((2026092701,'a'),(2026092702,'b')):
            name=co+'-s'+str(seed)
            rows.append(dict(row_id=name,cohort=co,expected_model_seed=seed,expected_checkpoint_sha256=letter*64,
                row_root='/synthetic/input/models/s'+str(seed),branch_features='/synthetic/input/'+name+'/branches',
                ground_packet='/synthetic/input/models/s'+str(seed)+'/native-a',
                ground_summary='/synthetic/input/models/s'+str(seed)+'/summary',ground_summary_already_deployed=False,
                output_root=run+'/'+name))
    return dict(schema=control.SCHEMA,run_id='synthetic-proto-frame',group_id='synthetic-only',
        spec_path='configs/synthetic_proto_frame.json',code=dict(cwd=release,environment='/synthetic/python',commit='c'*40),
        execution=dict(remote_run_root=run,launch_owner='root',cpu_lanes=2,blas_threads=2),
        benchmark=dict(config=dict(algorithm=dict(FROZEN_CONFIG),proto_frame_resources=dict(control.RESOURCES)),cohorts=cohorts),rows=rows)


def test_actual_ground_summary_and_boolean_are_wired_to_predictor():
    spec=literal_spec();control.validate_spec(spec)
    row=spec['rows'][0];argv=control.command(spec,row,'d'*40,'/synthetic/resolved.json')
    assert argv[2].endswith('/tools/evaluate_d92_proto_frame_joint_benchmark.py')
    assert argv[argv.index('--ground-summary')+1]==row['ground_summary']
    assert argv[argv.index('--ground-summary-already-deployed')+1]=='false'
    row['ground_summary_already_deployed']=True
    argv=control.command(spec,row,'d'*40,'/synthetic/resolved.json',preflight=True)
    assert argv[argv.index('--ground-summary-already-deployed')+1]=='true'
    assert argv[-1]=='--preflight-only'
    assert '--truth' not in argv


@pytest.mark.parametrize('value',[None,0,1,'false','true'])
def test_deployment_flag_requires_an_actual_boolean(value):
    spec=literal_spec();spec['rows'][0]['ground_summary_already_deployed']=value
    with pytest.raises(ValueError,match='deployment flag'):control.validate_spec(spec)


def test_same_source_model_cannot_change_ground_geometry_across_cohorts():
    spec=literal_spec();spec['rows'][2]['ground_summary']='/synthetic/input/other-summary'
    with pytest.raises(ValueError,match='Same model seed'):control.validate_spec(spec)


def test_ground_summary_cannot_alias_the_new_run_output():
    spec=literal_spec();spec['rows'][0]['ground_summary']=spec['execution']['remote_run_root']+'/injected-summary'
    with pytest.raises(ValueError,match='Source/output overlap'):control.validate_spec(spec)


@pytest.mark.parametrize('mutation',[
    lambda s:s['benchmark']['config']['algorithm'].update(max_updates=2),
    lambda s:s['benchmark']['config']['proto_frame_resources'].update(max_factor_buffer_bytes=1),
    lambda s:s['benchmark']['cohorts']['rx3']['matrix'].update(ks=[1,5]),
    lambda s:s['execution'].update(cpu_lanes=3),
    lambda s:s['rows'].pop(),
])
def test_fixed_method_full_matrix_and_root_resource_contract_are_enforced(mutation):
    spec=deepcopy(literal_spec());mutation(spec)
    with pytest.raises(ValueError):control.validate_spec(spec)
