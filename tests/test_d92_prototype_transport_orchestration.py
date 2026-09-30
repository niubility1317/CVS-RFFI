"""Exercise the fixed support contract and bounded solver accounting without a launch."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_d92_prototype_transport_probe as run
import prepare_d92_prototype_transport_probe as prepare
import analyze_d92_prototype_transport_probe as analyze


@pytest.fixture
def documents():
    parent=json.loads((ROOT/prepare.PARENT).read_text(encoding='utf-8'))
    return prepare.documents(parent,commit='a'*40)


def marker(spec):
    row=spec['rows'][0];co=spec['probe']['cohorts'][row['cohort']]
    value=dict.fromkeys(run.COUNTERS,0)
    value.update({key:count//4 for key,count in run.EXACT_COUNTS.items()})
    value.update(status=run.STATUS,capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['seeds']['model'],
        algorithm=run.PROBE_CONFIG,selection=co['selection'],producer_matrix=co['matrix'],
        query_rows_used=0,source_rows_used=0,truth_read=False,head_fit_count=792)
    return value


def verify(tmp_path,spec,value):
    path=tmp_path/'marker.json';path.write_text(json.dumps(value),encoding='utf-8')
    return run.verify_marker(path,spec,spec['rows'][0])


def test_preparation_preserves_support_identity_and_records_actual_optimizer(documents):
    spec=documents[prepare.SPEC];run.validate_spec(spec)
    parent=json.loads((ROOT/prepare.PARENT).read_text(encoding='utf-8'))
    assert spec['run_id']==prepare.RUN=='20260930-phase2-d92-prototype-transport-support-m2-r01'
    assert spec['run_id']!=parent['run_id']
    assert spec['execution']['remote_run_root'].endswith('/'+prepare.RUN)
    assert all('/'+prepare.RUN+'/' in row['output_root'] for row in spec['rows'])
    assert spec['permissions']['query_use']==parent['permissions']['query_use']
    assert spec['checkpoint']['runtime_checkpoint_reload'] is False
    assert spec['probe']['candidate']=='R_transport_seq'
    assert spec['probe']['controls']==['R0','R_transport_reset']
    assert spec['probe']['expected_head_fits']==40608
    for row,old in zip(spec['rows'],parent['rows']):
        assert row['support_features']==old['support_features']
        assert row['expected_checkpoint_sha256']==old['expected_checkpoint_sha256']
        assert row['seeds']==old['seeds']
        assert row['lr'] is None and row['initial_step_size']==0.125
        assert 'max4 iterations/max3 Armijo trials' in row['optimizer']
    assert len(documents)==3
    for name,co in spec['probe']['cohorts'].items():
        assert documents[run.CONFIG_NAMES[name]]==dict(algorithm=run.PROBE_CONFIG,
            producer_matrix=co['matrix'],selection=co['selection'])


@pytest.mark.parametrize('field,value',[('candidate','R_transport_reset'),
    ('expected_head_fits',29376),('interpretation','query_confirmation')])
def test_wrong_method_budget_or_claim_is_rejected(documents,field,value):
    spec=deepcopy(documents[prepare.SPEC]);spec['probe'][field]=value
    with pytest.raises((RuntimeError,ValueError)):run.validate_spec(spec)


def test_no_fixed_eight_step_assumption_for_legitimate_zero_update_stage(documents,tmp_path):
    spec=documents[prepare.SPEC];value=marker(spec)
    value.update(trained_transport_stage_count=0,inner_objective_evaluation_count=1,
        inner_head_fit_count=3,inner_factorization_count=3,backward_evaluation_count=1,
        derivative_triangular_solve_count=6,head_fit_count=795,factorization_count=3)
    assert verify(tmp_path,spec,value)['optimizer_steps']==0


@pytest.mark.parametrize('field,value',[('query_rows_used',1),('truth_read',True),
    ('optimizer_steps',937),('inner_objective_evaluation_count',3043),
    ('head_fit_count',999),('trial_count',True)])
def test_bad_binding_or_impossible_accounting_rejected(documents,tmp_path,field,value):
    spec=documents[prepare.SPEC];data=marker(spec);data[field]=value
    with pytest.raises((RuntimeError,ValueError)):verify(tmp_path,spec,data)


@pytest.mark.parametrize('rows,episodes',[(3,120),(4,159)])
def test_independent_analysis_requires_complete_declared_matrix(rows,episodes):
    with pytest.raises(ValueError):analyze.completion_check(dict(status=run.STATUS,
        completed_rows=rows,model_rows=4,episodes=episodes))


def test_release_contains_core_dependencies():
    import publish_d92_prototype_transport_probe as publish
    required={'code/cvsrffi/d92_prototype_transport_local_ridge.py',
        'code/cvsrffi/d92_joint_channel_local_ridge.py',
        'tools/evaluate_d92_prototype_transport_probe.py',
        'tools/summarize_d92_prototype_transport_probe.py',
        'configs/d92_prototype_transport_frozen_20260930.json'}
    assert required<=set(publish.PATHS)
    assert 'GPU0 occupied' not in publish.transport.REMOTE
