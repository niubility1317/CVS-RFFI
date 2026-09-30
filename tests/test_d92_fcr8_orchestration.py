"""Bound the joint MC adapter work and preserve the registered support contract."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_d92_fcr8_probe as run
import prepare_d92_fcr8_probe as prepare
import analyze_d92_fcr8_probe as analyze


@pytest.fixture
def documents():
    parent = json.loads((ROOT / prepare.PARENT).read_text(encoding='utf-8'))
    return prepare.documents(parent, commit='a' * 40)


def marker(spec):
    row = spec['rows'][0]
    co = spec['probe']['cohorts'][row['cohort']]
    value = dict.fromkeys(run.COUNTERS, 0)
    value.update({key: count // 4 for key, count in run.EXACT_COUNTS.items()})
    value.update(status=run.STATUS, capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'], model_seed=row['seeds']['model'],
        algorithm=run.PROBE_CONFIG, selection=co['selection'], producer_matrix=co['matrix'],
        query_rows_used=0, source_rows_used=0, truth_read=False, head_fit_count=792)
    return value


def verify(tmp_path, spec, value):
    path = tmp_path / 'marker.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return run.verify_marker(path, spec, spec['rows'][0])


def test_preparation_preserves_physical_support_and_optimizer(documents):
    spec = documents[prepare.SPEC]
    run.validate_spec(spec)
    parent = json.loads((ROOT / prepare.PARENT).read_text(encoding='utf-8'))
    assert spec['run_id'] == prepare.RUN == '20260930-phase2-d92-fcr8-support-m2-r01'
    assert spec['group_id'] == 'd92-fcr8-support' and 'fcr8' in spec['tags']
    assert spec['run_id'] != '20260930-phase2-d92-mc-residual8-support-m2-r01'
    assert spec['execution']['remote_run_root'].endswith('/' + prepare.RUN)
    assert spec['probe']['candidate'] == 'R_FCR8_seq'
    assert spec['probe']['controls'] == ['R0', 'R_FCR8_reset_init']
    assert spec['probe']['expected_head_fits'] == 42336
    assert spec['permissions']['query_use'] == parent['permissions']['query_use']
    assert spec['checkpoint']['runtime_checkpoint_reload'] is False
    for row, old in zip(spec['rows'], parent['rows']):
        assert row['support_features'] == old['support_features']
        assert row['expected_checkpoint_sha256'] == old['expected_checkpoint_sha256']
        assert row['seeds'] == old['seeds']
        assert row['lr'] is None and row['initial_step_size'] == .125
        assert '/' + prepare.RUN + '/' in row['output_root']
        assert 'keep-risk margin constraint' in row['optimizer']
    assert len(documents) == 3


def test_zero_update_stage_still_counts_two_adjoint_channels_and_teacher(documents, tmp_path):
    spec = documents[prepare.SPEC]
    value = marker(spec)
    value.update(trained_fcr_stage_count=0, inner_objective_evaluation_count=1,
        inner_head_fit_count=3, inner_factorization_count=3, backward_evaluation_count=1,
        derivative_triangular_solve_count=12, teacher_head_fit_count=3,
        task_derivative_triangular_solve_count=6, keep_derivative_triangular_solve_count=6,
        teacher_factorization_count=3, head_fit_count=798, factorization_count=6)
    assert verify(tmp_path, spec, value)['optimizer_steps'] == 0


@pytest.mark.parametrize('field,value', [
    ('query_rows_used', 1), ('truth_read', True), ('optimizer_steps', 937),
    ('inner_objective_evaluation_count', 3043), ('teacher_head_fit_count', 433),
    ('head_fit_count', 999), ('trial_count', True), ('derivative_triangular_solve_count', 11233)])
def test_illegal_access_or_work_accounting_rejected(documents, tmp_path, field, value):
    spec = documents[prepare.SPEC]
    data = marker(spec)
    data[field] = value
    with pytest.raises((RuntimeError, ValueError)):
        verify(tmp_path, spec, data)


@pytest.mark.parametrize('field,value', [('candidate', 'R_FCR8_reset_init'),
    ('expected_head_fits', 40608), ('interpretation', 'query_confirmation')])
def test_frozen_method_budget_and_claim(documents, field, value):
    spec = deepcopy(documents[prepare.SPEC])
    spec['probe'][field] = value
    with pytest.raises((RuntimeError, ValueError)):
        run.validate_spec(spec)


@pytest.mark.parametrize('rows,episodes', [(3, 120), (4, 159)])
def test_analysis_requires_all_declared_parents(rows, episodes):
    with pytest.raises(ValueError):
        analyze.completion_check(dict(status=run.STATUS, completed_rows=rows,
            model_rows=4, episodes=episodes))


def test_release_includes_imported_core_and_entry_dependencies():
    import publish_d92_fcr8_probe as publish
    required = {'code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py',
        'code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py',
        'code/cvsrffi/d92_prototype_transport_local_ridge.py',
        'code/cvsrffi/d92_joint_channel_local_ridge.py',
        'tools/evaluate_d92_fcr8_probe.py',
        'tools/summarize_d92_fcr8_probe.py',
        'tools/evaluate_d92_prototype_transport_probe.py',
        'tools/summarize_d92_prototype_transport_probe.py',
        'configs/d92_fcr8_frozen_20260930.json',
        'configs/d92_prototype_transport_frozen_20260930.json'}
    assert required <= set(publish.PATHS)
    assert 'GPU0 occupied' not in publish.transport.REMOTE
