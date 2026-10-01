"""No remote execution: dynamic analysis boundaries and source bundle contracts."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import analyze_d92_conditional_joint_probe as analyzer
from run_d92_conditional_joint_probe import budget_for_spec, KS, NEW_COUNTS


def _spec(rows=3):
    old = ['a' + str(j) for j in range(6)]
    selection = dict(receiver_scenes=[['rx', 'practical_high']], support_seed=2, ks=KS, new_counts=NEW_COUNTS,
        splits=[dict(split_id=str(k) + '_' + str(new), receiver='rx', scenario='practical_high', k=k, new_count=new,
            support_seed=2, registered_classes=old + ['z' + str(j).zfill(2) for j in range(new)]) for k in KS for new in NEW_COUNTS])
    return dict(run_id='synthetic_run', rows=[dict(row_id='row' + str(j), cohort='co') for j in range(rows)],
        probe=dict(cohorts=dict(co=dict(selection=selection))), code=dict(cwd='/release/synthetic', environment='/python/ssr-gpu/bin/python'),
        execution=dict(remote_run_root='/run/synthetic'), spec_path='configs/synthetic_spec.json')


@pytest.mark.parametrize('rows', [1, 2, 3, 6])
def test_dynamic_completion_and_summary_verification(rows):
    spec = _spec(rows); exact = budget_for_spec(spec)['total']['exact']
    done = dict(status=analyzer.STATUS, run_id=spec['run_id'], model_rows=rows, completed_rows=rows, **exact)
    analyzer.completion_check(done, spec)
    result = dict(status=analyzer.SUMMARY_STATUS, summary_schema=analyzer.SUMMARY_SCHEMA, schema=analyzer.SCHEMA,
        method=analyzer.METHOD, run_id=spec['run_id'], model_rows=rows, coverage=exact,
        query_rows_used=0, source_rows_used=0, actual_A=None)
    analyzer.summary_check(result, spec)
    done['episodes'] -= 1
    with pytest.raises(ValueError): analyzer.completion_check(done, spec)
    result['coverage'] = dict(exact, episodes=exact['episodes'] - 1)
    with pytest.raises(ValueError): analyzer.summary_check(result, spec)


@pytest.mark.parametrize('name', ['', '../escape', 'slash/path', 'two names', 'Name'])
def test_unsafe_analysis_release_names_rejected(name):
    with pytest.raises(ValueError): analyzer.remote_config(_spec(), name, 'a' * 40)


def test_remote_script_has_dynamic_coverage_exclusive_evidence_and_no_training_launch():
    cfg = analyzer.remote_config(_spec(), 'synthetic-analysis', 'a' * 40)
    compile(analyzer.REMOTE.replace('CONFIG', repr(cfg)), 'synthetic-remote-analysis', 'exec')
    assert len(cfg['rows']) == 3 and cfg['exact']['episodes'] == 60
    assert cfg['output'] == '/run/synthetic/results/support_summary'
    assert 'summarize_d92_conditional_joint_probe.py' in analyzer.REMOTE
    assert 'analysis_process.json' in analyzer.REMOTE and 'analysis_execution.json' in analyzer.REMOTE
    assert "release.mkdir(exist_ok=False)" in analyzer.REMOTE and "open('x'" in analyzer.REMOTE
    assert 'evaluate_d92_conditional_joint_probe.py' not in analyzer.REMOTE
    assert 'run_d92_conditional_joint_probe.py' not in analyzer.REMOTE


def test_analysis_whitelist_includes_actual_independent_math_dependencies():
    required = {'code/cvsrffi/d92_conditional_joint_local_ridge.py', 'code/cvsrffi/d92_conditional_affine_kernel.py',
        'tools/d92_conditional_analysis_math.py', 'tools/d92_affine_analysis_math.py',
        'tools/summarize_d92_affine_joint_probe.py', 'tools/evaluate_d92_affine_joint_probe.py',
        'tools/summarize_d92_conditional_joint_probe.py', 'tools/analyze_d92_conditional_joint_probe.py'}
    assert required <= set(analyzer.PATHS)
    assert len(analyzer.PATHS) == len(set(analyzer.PATHS))
