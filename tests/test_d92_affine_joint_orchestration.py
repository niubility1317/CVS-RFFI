"""Bind the two-path AffineJL pilot, physical cache reuse and actual solve accounting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_d92_affine_joint_probe as run
import prepare_d92_affine_joint_probe as prepare
import analyze_d92_affine_joint_probe as analyze


@pytest.fixture
def documents():
    parent = json.loads((ROOT/prepare.PARENT).read_text(encoding='utf-8'))
    return prepare.documents(parent, commit='a'*40)


def marker(spec):
    row = spec['rows'][0]
    co = spec['probe']['cohorts'][row['cohort']]
    value = dict.fromkeys(run.COUNTERS, 0)
    value.update({key: count//4 for key, count in run.EXACT_COUNTS.items()})
    value.update(status=run.STATUS, run_id=spec['run_id'], row_id=row['row_id'], capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'], model_seed=row['seeds']['model'],
        algorithm=run.PROBE_CONFIG, selection=co['selection'], producer_matrix=co['matrix'],
        query_rows_used=0, source_rows_used=0, truth_read=False, head_fit_count=810)
    return value


def verify(tmp_path, spec, value):
    path = tmp_path/'marker.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return run.verify_marker(path, spec, spec['rows'][0])


def test_only_one_structure_preserves_physical_support_source_and_six_seeds(documents):
    spec = documents[prepare.SPEC]
    run.validate_spec(spec)
    parent = json.loads((ROOT/prepare.PARENT).read_text(encoding='utf-8'))
    assert spec['run_id'] == '20261001-phase2-d92-affine-joint-support-m2-r01'
    assert spec['group_id'] == 'd92-affine-joint-support'
    assert spec['execution']['remote_run_root'].endswith('/'+prepare.RUN)
    assert spec['probe']['candidate'] == 'R_AFFINE_seq'
    assert spec['probe']['controls'] == ['R0']
    assert spec['probe']['expected_head_fits'] == 102600
    assert spec['probe']['exact_counts']['sequence_paths'] == 1800
    assert spec['probe']['expected_sequence_paths'] == 1800
    assert spec['checkpoint']['runtime_checkpoint_reload'] is False
    assert spec['permissions']['query_use'] == parent['permissions']['query_use']
    assert spec['probe']['cohorts']['rx1']['capsule'] == parent['probe']['cohorts']['rx1']['capsule']
    for row, old in zip(spec['rows'], parent['rows']):
        assert row['support_features'] == old['support_features']
        assert row['expected_checkpoint_sha256'] == old['expected_checkpoint_sha256']
        assert row['seeds'] == old['seeds']
        assert row['lr'] is None and row['initial_step_size'] == .125
        assert '/'+prepare.RUN+'/' in row['output_root']
        assert 'class-RMS CE' in row['optimizer'] and 'no keep channel' in row['optimizer']
    assert len(documents) == 3


def test_zero_update_can_have_initial_objective_backward_and_closed_heads(documents, tmp_path):
    spec = documents[prepare.SPEC]
    value = marker(spec)
    value.update(inner_objective_evaluation_count=1, inner_head_fit_count=3,
        inner_factorization_count=3, final_head_fit_count=1, final_factorization_count=1,
        prior_head_fit_count=3, prior_factorization_count=3, prior_triangular_solve_count=6,
        backward_evaluation_count=1, ce_adjoint_solve_count=3, derivative_triangular_solve_count=6,
        head_triangular_solve_count=8, ajlr_forward_evaluation_count=4,
        head_fit_count=817, factorization_count=7,
        head_triangular_rhs_count=56, head_triangular_rhs_element_count=336, head_triangular_dense_work_unit_count=2016,
        derivative_triangular_rhs_count=36, derivative_triangular_rhs_element_count=216, derivative_triangular_dense_work_unit_count=1296,
        prior_triangular_rhs_count=42, prior_triangular_rhs_element_count=252, prior_triangular_dense_work_unit_count=1512,
        intercept_fit_count=4, prior_intercept_fit_count=3)
    assert verify(tmp_path, spec, value)['optimizer_steps'] == 0


def test_baseline_EDF_solves_are_measured_separately(documents, tmp_path):
    spec = documents[prepare.SPEC]
    value = marker(spec)
    value.update(baseline_factorization_count=1, baseline_head_triangular_solve_count=2,
        baseline_effective_df_triangular_solve_count=2, baseline_triangular_solve_count=4,
        factorization_count=1)
    verify(tmp_path, spec, value)
    value['baseline_triangular_solve_count'] = 2
    with pytest.raises(ValueError, match='Baseline triangular'):
        verify(tmp_path, spec, value)


@pytest.mark.parametrize('field,value', [
    ('query_rows_used', 1), ('truth_read', True), ('optimizer_steps', 649),
    ('inner_objective_evaluation_count', 7939), ('prior_head_fit_count', 217),
    ('head_fit_count', 811), ('trial_count', True), ('derivative_triangular_solve_count', 3889),
    ('final_head_fit_count', 811), ('baseline_effective_df_triangular_solve_count', 1),
    ('ajlr_forward_evaluation_count', 1)])
def test_illegal_access_budget_or_impossible_accounting_rejected(documents, tmp_path, field, value):
    data = marker(documents[prepare.SPEC])
    data[field] = value
    with pytest.raises(ValueError):
        verify(tmp_path, documents[prepare.SPEC], data)


@pytest.mark.parametrize('field,value', [('candidate', 'R_AFFINE_reset'),
    ('controls', ['R0', 'R_AFFINE_reset']), ('expected_head_fits', 42336),
    ('interpretation', 'query_confirmation'), ('expected_sequence_paths', 1760)])
def test_frozen_structure_and_support_only_claim(documents, field, value):
    spec = deepcopy(documents[prepare.SPEC])
    spec['probe'][field] = value
    with pytest.raises(ValueError):
        run.validate_spec(spec)


@pytest.mark.parametrize('rows,episodes', [(3, 120), (4, 159)])
def test_independent_analysis_waits_for_declared_complete_parents(rows, episodes):
    with pytest.raises(ValueError):
        analyze.completion_check(dict(status=run.STATUS, completed_rows=rows,
            model_rows=4, episodes=episodes))


def test_release_has_pure_helper_dependencies_but_no_old_adapted_state():
    import publish_d92_affine_joint_probe as publish
    required = {'code/cvsrffi/d92_affine_joint_local_ridge.py',
        'code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py',
        'code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py',
        'code/cvsrffi/d92_prototype_transport_local_ridge.py',
        'code/cvsrffi/d92_joint_channel_local_ridge.py',
        'tools/evaluate_d92_affine_joint_probe.py', 'tools/summarize_d92_affine_joint_probe.py',
        'configs/d92_affine_joint_frozen_20261001.json'}
    assert required <= set(publish.PATHS)
    assert 'GPU0 occupied' not in publish.transport.REMOTE
    assert not any('/state_arrays/' in p or p.endswith(('.npz', '.pt', '.pth')) for p in publish.PATHS)


def test_declared_counter_sets_match_entry_and_core():
    import evaluate_d92_affine_joint_probe as entry
    assert set(run.COUNTERS) == set(entry.COUNTERS)
    assert run.EXACT_COUNTS == entry.EXACT_COUNTS
    assert run.MAX_COUNTS == entry.MAX_COUNTS


def test_runtime_command_passes_actual_run_and_row_binding(documents):
    spec = documents[prepare.SPEC]
    row = spec['rows'][0]
    command = run.command(spec, row)
    assert command[command.index('--run-id')+1] == spec['run_id']
    assert command[command.index('--row-id')+1] == row['row_id']


@pytest.mark.parametrize('field,value', [
    ('head_triangular_rhs_count', 1), ('prior_triangular_rhs_count', 1),
    ('derivative_triangular_rhs_element_count', 1), ('head_triangular_dense_work_unit_count', 1),
    ('intercept_fit_count', 1), ('run_id', 'different-run'), ('row_id', 'different-row')])
def test_rhs_or_actual_state_binding_cannot_be_fabricated(documents, tmp_path, field, value):
    spec = documents[prepare.SPEC]
    data = marker(spec)
    data[field] = value
    with pytest.raises(ValueError):
        verify(tmp_path, spec, data)
