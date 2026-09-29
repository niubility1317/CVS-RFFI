"""Synthetic cache-to-summary coverage; no producer, Torch, or query fixtures."""
import contextlib
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import evaluate_d92_branch_local_margin_probe as entry
import summarize_d92_branch_local_margin_probe as summary
from test_evaluate_d92_branch_local_margin_probe import fixture, CACHE_NAME


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def write_records(path, records):
    path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records), encoding='utf-8')


def complete_fixture(root, ks=(1, 2, 5), constant=False):
    run = root / 'run'
    run.mkdir()
    rows, cohorts, totals = [], {}, dict(k1_episodes=0, oof_episodes=0,
        proxy_anchor_count=0, factorization_count=0, optimizer_steps=0)
    for cohort in ('rx1', 'rx3'):
        for seed in range(2026092701, 2026092705):
            rid = f'{cohort}-{seed}'
            lane = run / rid
            lane.mkdir()
            args = fixture(lane, ks=ks)
            args['output'] = lane / 'probe'
            args['expected_model_seed'] = seed
            if constant:
                path = args['support_features'] / CACHE_NAME
                with np.load(path, allow_pickle=False) as stored:
                    arrays = {key: stored[key] for key in stored.files}
                for name in entry.BRANCHES:
                    arrays[name] = np.ones_like(arrays[name])
                np.savez(path, **arrays)
            for name in ('features_complete.json', 'startup.json', 'checkpoint_provenance.json'):
                path = args['support_features'] / name
                value = entry.read(path)
                value['model_seed'] = seed
                if name == 'features_complete.json':
                    value['feature_file_bytes'] = (args['support_features'] / CACHE_NAME).stat().st_size
                if 'provenance' in value:
                    value['provenance']['model_seed'] = seed
                write(path, value)
            with contextlib.redirect_stdout(io.StringIO()):
                marker = entry.evaluate(**args)
            for key in totals:
                totals[key] += marker[key]
            rows.append(dict(row_id=rid, cohort=cohort, seeds=dict(model=seed),
                             expected_checkpoint_sha256=args['expected_checkpoint_sha256']))
            cohorts[cohort] = dict(matrix=args['config']['matrix'],
                capsule_id=args['expected_capsule_id'], expected_split_count=len(ks))
    spec = dict(run_id='synthetic-local-ridge', execution=dict(remote_run_root=str(run)), rows=rows,
                probe=dict(cohorts=cohorts, total_episodes=8 * len(ks)))
    write(run / 'startup.json', dict(spec=spec, query_access=False, source_sample_access=False,
                                    commit='synthetic', started=1.))
    write(run / 'complete.json', dict(status='SUPPORT_PROBE_COMPLETE', completed_rows=8,
        model_rows=8, episodes=8 * len(ks), query_access=False, source_sample_access=False,
        commit='synthetic', finished=2., **totals))
    write(run / 'state.json', {r['row_id']: dict(status='SUPPORT_PROBE_COMPLETE', episodes=len(ks)) for r in rows})
    return spec, run


@pytest.fixture(scope='module')
def clean_run(tmp_path_factory):
    return complete_fixture(tmp_path_factory.mktemp('local-ridge-summary'))


def copied_run(clean_run, tmp_path):
    spec, source = clean_run
    run = tmp_path / 'run'
    shutil.copytree(source, run)
    # All absolute support paths continue to refer only to the synthetic producer markers.
    return deepcopy(spec), run


def test_complete_parent_first_aggregation_reads_only_synthetic_evidence(clean_run, tmp_path, monkeypatch):
    spec, run = clean_run
    original, opened = Path.open, []

    def guarded(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get('mode', 'r')
        if 'r' in mode:
            opened.append(path)
            assert path.is_relative_to(run)
            assert path.name in {'startup.json', 'complete.json', 'state.json', 'probe_complete.json',
                                 'features_complete.json', 'fit_trace.jsonl', 'compact.jsonl'}
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, 'open', guarded)
        result = summary.summarize(spec=spec, run_root=run, output=tmp_path / 'summary')
    assert opened
    assert result['coverage']['optimizer_steps'] > 0
    assert result['coverage']['candidate_fits'] == 96
    assert {k: v for k, v in result['coverage'].items() if k not in ('optimizer_steps', 'candidate_fits')} == dict(episodes=24, parent_episodes=24, k1_numerical_only=8,
        standard_oof_parents=16, proxy_parents=16, proxy_anchors=56,
        standard_factorizations=120, proxy_factorizations=168, actual_factorizations=288,
        proxy_held_occurrences_per_arm=352, nominal_max_factorizations=288,
        degenerate_factorization_reductions=0)
    for arm in summary.ARMS:
        proxy = next(r for r in result['statistics'] if r['diagnostic'] == 'support_oneshot_proxy'
                     and r['metric'] == arm + '.h')
        assert proxy['count'] == 16  # Parent count, never anchors or repeated held occurrences.
        expected = []
        for row in spec['rows']:
            traces = list(summary.jsonlines(run / row['row_id'] / 'probe/fit_trace.jsonl'))
            expected.extend(r['oneshot_proxy']['parent_mean_metrics'][arm]['h'] for r in traces if r['k'] > 1)
        assert proxy['mean'] == pytest.approx(sum(expected) / len(expected))
    assert not result['automatic_promotion'] and result['selected_arm'] is None
    assert result['candidate_assessment']['standard']['passes'] is None
    assert 'ncm_equivalence' not in json.dumps(result)
    with (tmp_path / 'summary/by_parent_k_newcount.csv').open(encoding='utf-8', newline='') as stream:
        strata = list(csv.DictReader(stream))
    assert not any(r['k'] == '1' and r['diagnostic'] != 'parent_numerical' for r in strata)
    assert {r['metric'].split('.')[0] for r in strata if r['kind'] == 'arm'} == set(summary.ARMS)
    with pytest.raises(FileExistsError):
        summary.summarize(spec=spec, run_root=run, output=tmp_path / 'summary')


@pytest.mark.parametrize('fault', [
    'anchor', 'train_held', 'missing_held', 'stage_training_id', 'confusion', 'nll_count',
    'paired_count', 'parent_mean', 'k1_proxy', 'missing_parent', 'missing_arm',
    'tau', 'gamma', 'residual', 'trace_residual', 'distance_rule', 'bandwidth_rule',
    'physical_mass', 'extra_solve_count', 'stage_factor_count'])
def test_corrupt_evidence_rejected_before_output(clean_run, tmp_path, fault):
    spec, run = copied_run(clean_run, tmp_path)
    path = run / spec['rows'][0]['row_id'] / 'probe/fit_trace.jsonl'
    records = list(summary.jsonlines(path))
    record = records[1]
    trial = record['oneshot_proxy']['trials'][0]
    local = next(s for s in trial['stages'] if s['arm'] == 'local_ridge')
    if fault == 'anchor':
        trial['trial'] = 1
    elif fault == 'train_held':
        trial['training_ids'][0] = trial['held_ids'][0]
    elif fault == 'missing_held':
        trial['held_ids'].pop()
    elif fault == 'stage_training_id':
        local['training_physical_ids'][0] = trial['held_ids'][0]
    elif fault == 'confusion':
        trial['oof']['local_ridge']['confusion'][0][0] += 1
    elif fault == 'nll_count':
        trial['oof']['local_ridge']['classwise_count'][0] += 1
    elif fault == 'paired_count':
        trial['paired']['local_ridge_minus_branch_ridge']['counts']['both_wrong'] += 1
    elif fault == 'parent_mean':
        record['oneshot_proxy']['parent_mean_metrics']['local_ridge']['h'] += .1
    elif fault == 'k1_proxy':
        records[0]['oneshot_proxy'] = deepcopy(record['oneshot_proxy'])
    elif fault == 'missing_parent':
        records.pop()
    elif fault == 'missing_arm':
        trial['oof'].pop('interaction_ridge')
    elif fault == 'tau':
        local['bandwidth_tau'] *= 2
    elif fault == 'gamma':
        local['trace_scale'] *= 2
    elif fault == 'residual':
        local['normal_equation_residual'] = local['numerical_tolerance'] * 2
    elif fault == 'trace_residual':
        local['trace_relative_error'] = local['numerical_tolerance'] * 2
    elif fault == 'distance_rule':
        local['distance_rule'] = 'gram-subtraction'
    elif fault == 'bandwidth_rule':
        local['bandwidth_rule'] = 'held-selected'
    elif fault == 'physical_mass':
        local['physical_loss_mass'] = 1.
    elif fault == 'extra_solve_count':
        local['effective_degrees_of_freedom_extra_triangular_solves'] = 0
    elif fault == 'stage_factor_count':
        local['factorization_calls'] = 0
    write_records(path, records)
    with pytest.raises(ValueError):
        summary.summarize(spec=spec, run_root=run, output=tmp_path / 'summary')
    assert not (tmp_path / 'summary').exists()


@pytest.mark.parametrize('field', ['k1_episodes', 'oof_episodes', 'proxy_anchor_count',
                                  'factorization_count', 'optimizer_steps'])
def test_supervisor_totals_must_match_verified_lanes(clean_run, tmp_path, field):
    spec, run = copied_run(clean_run, tmp_path)
    path = run / 'complete.json'
    marker = entry.read(path)
    marker[field] += 1
    write(path, marker)
    with pytest.raises(ValueError):
        summary.summarize(spec=spec, run_root=run, output=tmp_path / 'summary')
    assert not (tmp_path / 'summary').exists()


def test_degenerate_candidate_counts_actual_reduced_factorizations(tmp_path):
    spec, run = complete_fixture(tmp_path, ks=(1, 2), constant=True)
    result = summary.summarize(spec=spec, run_root=run, output=tmp_path / 'summary')
    counts = result['coverage']
    assert counts['nominal_max_factorizations'] == 96
    assert counts['actual_factorizations'] == 64  # Two unchanged controls, zero local solves.
    assert counts['standard_factorizations'] == counts['proxy_factorizations'] == 32
    assert counts['degenerate_factorization_reductions'] == 32
    trace = list(summary.jsonlines(run / spec['rows'][0]['row_id'] / 'probe/fit_trace.jsonl'))[1]
    for fit in trace['folds'] + trace['oneshot_proxy']['trials']:
        local = next(s for s in fit['stages'] if s['arm'] == 'local_ridge')
        assert local['bandwidth_tau'] == 0 and local['trace_scale'] is None
        assert local['interaction_centered_trace'] == 0
        assert local['factorization_calls'] == 0


def screening_stats(old=.01):
    values = {}
    for diagnostic in ('physical_oof', 'support_oneshot_proxy'):
        for k in (5, 10, 20):
            for control in ('local_ridge',):
                for field in ('h', 'new_accuracy', 'old_accuracy'):
                    summary.add(values, (diagnostic, 'new_present', k, 'paired',
                        'local_margin_minus_' + control + '.' + field), old if field == 'old_accuracy' else .01)
    return values


@pytest.mark.parametrize('diagnostic', ['physical_oof', 'support_oneshot_proxy'])
@pytest.mark.parametrize('control', ['local_ridge'])
@pytest.mark.parametrize('k', [5, 10, 20])
@pytest.mark.parametrize('metric,bad', [('h', 0.), ('new_accuracy', -.001), ('old_accuracy', -.011)])
def test_screen_checks_each_parent_against_both_controls(diagnostic, control, k, metric, bad):
    values = screening_stats()
    key = (diagnostic, 'new_present', k, 'paired', 'local_margin_minus_' + control + '.' + metric)
    values.pop(key)
    summary.add(values, key, bad)
    result = summary.assessment(values, True)
    failed = 'standard' if diagnostic == 'physical_oof' else 'oneshot_proxy'
    other = 'oneshot_proxy' if failed == 'standard' else 'standard'
    assert result[failed]['passes'] is False
    assert result[other]['passes'] is True


def test_old_guard_pass_is_distinct_from_strict_joint_improvement():
    result = summary.assessment(screening_stats(old=-.005), True)
    for diagnostic in ('standard', 'oneshot_proxy'):
        assert result[diagnostic]['passes'] is True
        assert result[diagnostic]['old_new_h_all_strictly_improve'] is False
        assert len(result[diagnostic]['comparisons']) == 3
    assert result['automatic_promotion'] is False
    strict = summary.assessment(screening_stats(), True)
    assert strict['standard']['old_new_h_all_strictly_improve'] is True
    assert strict['oneshot_proxy']['old_new_h_all_strictly_improve'] is True


def test_screen_has_no_extra_newcount_gate_and_requires_complete_parent_evidence():
    # Two equally weighted new-count strata can differ in sign while their parent mean passes.
    values = screening_stats()
    for key in list(values):
        values.pop(key)
        summary.add(values, key, -.03)
        summary.add(values, key, .05)
    result = summary.assessment(values, True)
    assert result['standard']['passes'] is result['oneshot_proxy']['passes'] is True
    unavailable = summary.assessment(values, False)
    assert unavailable['standard']['passes'] is unavailable['oneshot_proxy']['passes'] is None
    values.pop(('physical_oof', 'new_present', 10, 'paired', 'local_margin_minus_local_ridge.h'))
    missing = summary.assessment(values, True)
    assert missing['standard']['available'] is False and missing['standard']['passes'] is None
    assert missing['oneshot_proxy']['passes'] is True


def test_old_only_and_true_k1_do_not_invent_new_class_or_held_metrics(tmp_path):
    args = fixture(tmp_path, ks=(1, 2))
    arrays, tasks, _, *_ = entry.load_support(support_features=args['support_features'],
        capsule=args['capsule'], expected_capsule_id=args['expected_capsule_id'],
        expected_checkpoint_sha256=args['expected_checkpoint_sha256'],
        expected_model_seed=args['expected_model_seed'],
        config=dict(algorithm=entry.CACHE_VALIDATION_CONFIG, matrix=args['config']['matrix']))
    for split, positions, labels in tasks:
        record = entry.probe_branch_local_margin(**{name: value[positions] for name, value in arrays.items()},
            support_labels=labels, support_ids=split['support_ids'],
            classes=split['registered_classes'], old_classes=split['registered_classes'])
        entry.compact_proxy(record, split['split_id'])
        record.update(split_id=split['split_id'], new_count=0, query_rows_used=0, source_rows_used=0)
        verified = summary.verify_record(record)
        if split['k'] == 1:
            assert verified['standard'] == verified['proxy'] == {}
        else:
            for diagnostic in ('standard', 'proxy'):
                for metrics in verified[diagnostic].values():
                    assert metrics['new_accuracy'] is None and metrics['h'] is None


def test_old_controls_are_descriptive_and_cannot_rescue_direct_baseline_failure():
    values = screening_stats()
    for diagnostic in ('physical_oof', 'support_oneshot_proxy'):
        for k in (5, 10, 20):
            for control in ('branch_ridge', 'interaction_ridge'):
                for field in ('h', 'new_accuracy', 'old_accuracy'):
                    summary.add(values, (diagnostic, 'new_present', k, 'paired',
                        'local_margin_minus_' + control + '.' + field), -.9)
    assert summary.assessment(values, True)['standard']['passes'] is True
    key = ('physical_oof', 'new_present', 5, 'paired', 'local_margin_minus_local_ridge.h')
    values.pop(key)
    summary.add(values, key, 0.)
    assert summary.assessment(values, True)['standard']['passes'] is False


@pytest.mark.parametrize('field', ['optimizer_steps', 'relative_duality_gap', 'relative_kkt_residual'])
def test_margin_solver_certificate_and_measured_steps_must_match(clean_run, tmp_path, field):
    spec, run = copied_run(clean_run, tmp_path)
    path = run / spec['rows'][0]['row_id'] / 'probe/fit_trace.jsonl'
    records = list(summary.jsonlines(path))
    stage = next(s for s in records[1]['folds'][0]['stages'] if s['arm'] == 'local_margin')
    stage[field] += 1
    write_records(path, records)
    with pytest.raises(ValueError):
        summary.summarize(spec=spec, run_root=run, output=tmp_path / 'summary')
