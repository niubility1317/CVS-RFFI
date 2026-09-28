"""Synthetic only: no real run artifacts or target scores are accessed."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import finalize_d92_osc_record as target
from summarize_d92_confirmation import summarize
from summarize_d92_repeated_benchmark import combine


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def spec(cohort):
    return dict(run_id=target.RUNS[cohort], permissions=dict(claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS'),
        rows=[dict(row_id=str(s), seeds=dict(model=s), source_root='/source/' + str(s),
                   reuse_row_root='/original/' + cohort + '/' + str(s), expected_checkpoint_sha256='a'*64)
              for s in range(2026092701, 2026092705)],
        data=dict(capsule_id=cohort, target_receivers=['a', 'b', 'c'] if cohort == 'rx3' else ['d'], scenarios=['x', 'y', 'z'],
            k=[1, 5, 10, 20], new_class_counts=[0, 2, 5, 10, 20], support_seeds=[11, 12, 13, 14, 15]),
        confirmation=dict(candidate_method='D92-OSC-v1', candidate_folder='osc',
            candidate_predictor='predict_d92_orbit_shared.py', candidate_mode='d92_osc_registration',
            capsule='/capsule/' + cohort, reuse_validated_capsule_id=cohort),
        metrics_plan=dict(acceptance=dict(old_max_drop=0.01, require_new_improvement=True)),
        joint_benchmark=dict(old_accuracy_scope='All paired cells at this K, including old-only registration'))


def score_rows(s, adverse=False):
    data = s['data']
    axes = ([r['seeds']['model'] for r in s['rows']], data['target_receivers'], data['scenarios'],
            data['k'], data['new_class_counts'], data['support_seeds'])
    result = []
    for cell in itertools.product(*axes):
        metadata = dict(zip(target.KEY, cell))
        classes = [f'old-{i}' for i in range(6)] + [f'new-{i}' for i in range(cell[4])]
        metadata.update(row_id=str(cell[0]), split_id=str(cell[1:]), classes=classes,
                        class_count=len(classes), query_count=100 * len(classes))
        for method in target.METHODS:
            old = 0.4 if cell[0] % 2 else 0.7
            new = 0.7 if cell[0] % 2 else 0.4
            if method == target.METHODS[1]:
                # Joint old falls by 2pp, while all-cell old rises by 0.4pp.
                old += -0.02 if cell[4] else 0.1
                new += 0.06 if cell[1] != 'd' else (-0.25 if adverse else -0.04)
            metrics = {m: old for m in target.METRICS}
            metrics.update(new_accuracy=new if cell[4] else None,
                harmonic_mean=2 * old * new / (old + new) if cell[4] else None,
                new_macro_f1=new if cell[4] else None, new_floor=new if cell[4] else None)
            class_accuracy = [old]*6 + [new]*cell[4]
            confusion = [[0]*len(classes) for _ in classes]
            for i, acc in enumerate(class_accuracy):
                confusion[i][i] = round(100*acc); confusion[i][(i+1)%len(classes)] = 100-round(100*acc)
            result.append(dict(metadata, method=method, class_accuracy=class_accuracy, confusion=confusion, **metrics))
        if cell[3:] == (1, 0, 11):
            result.append(dict(metadata, method='frozen_dg', class_accuracy=class_accuracy, confusion=confusion, **metrics))
    return dict(status='SCORED', selection_feedback_forbidden=True,
                claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS', results=result)


def fit_audit(splits):
    models = []
    for seed in range(2026092701, 2026092705):
        models.append(dict(row_id=str(seed), fits=splits, fixed_k1_fits=splits // 4, diagnostic_cv_fits=3 * splits // 4,
            optimizer_steps=0, analytical_fit_calls=13*splits//4, fold_fit_calls=9*splits//4, final_fit_calls=splits,
            final_status_counts={'ZERO_PHYSICAL_RESIDUAL': splits//4, 'SHARED_COVARIANCE_FIT': 3*splits//4},
            fold_status_counts={'SHARED_COVARIANCE_FIT': 9*splits//4},
            timing={k: dict(total=splits * v) for k, v in dict(fit_seconds=1.0, fit_call_seconds=1.1,
                query_score_seconds=0.1, prediction_write_seconds=0.01, total_seconds=1.25).items()},
            analytical_stage_timing={scope: {k: dict(total=splits*v*weight) for k, v in
                dict(medoid_seconds=.01, covariance_seconds=.02, solve_seconds=.1, fit_seconds=.15).items()}
                for scope, weight in dict(all=13/4, fold=9/4, final=1).items()},
            numeric_state_byte_ranges={k: dict(min=8200*6, max=8200*26) for k in ('head_bytes', 'persistent_state_bytes')},
            resources=dict(model_already_deployed=None, model_incremental_transfer_bytes=None, new_source_payload_bytes=0, ground_summary_used=False,
                existing_frozen_model_file_bytes=15992872 + seed % 4, feature_extraction_seconds=0,
                prior_producer_extraction_seconds=12, new_native_forward_count=0, reused_frozen_cache=True, checkpoint_loaded=False,
                received_cache_array_bytes=294400, received_cache_file_bytes=300000),
            peak_process_rss_bytes=50000000, trace_file='/synthetic/' + str(seed) + '/osc/fit_trace.jsonl'))
    return dict(status='VERIFIED', method='D92-OSC-v1', models=models, total_fits=4 * splits,
        fixed_k1_fits=splits, diagnostic_cv_fits=3 * splits, analytical_fit_calls=13*splits, optimizer_steps=0,
        query_rows_used_for_fit=0, source_rows_used_for_fit=0, new_source_payload_bytes=0,
        ground_summary_used=False, optimizer_convergence_claim=False, selection='none_cv_diagnostic_only', adapted_state_inherited=False)


@pytest.fixture(scope='module')
def synthetic():
    specs = [spec(c) for c in target.RUNS]
    scores = [score_rows(s) for s in specs]
    inputs = {c: dict(scores=d, summary=summarize(d, s)) for c, s, d in zip(target.RUNS, specs, scores)}
    return inputs, combine(specs, scores), specs


def workspace(tmp_path, synthetic):
    inputs, combined, _ = synthetic
    for cohort, run in target.RUNS.items():
        folder = tmp_path / 'automation_reports/CV-SincNet' / run
        folder.mkdir(parents=True)
        (folder / 'report.md').write_bytes(b'Historical RUNNING report\r\n')
        result = folder / 'results'
        dump(result / 'complete.json', dict(status='SCORED', commit='synthetic', records=7236 if cohort == 'rx3' else 2412))
        current = spec(cohort)
        dump(result / 'startup.json', dict(commit='synthetic', spec=current))
        prior = copy.deepcopy(current); prior['run_id'] = target.BASELINES[cohort]
        prior['data']['capsule_id'] = None
        prior['confirmation'].pop('reuse_validated_capsule_id')
        for row in prior['rows']: row['output_root'] = row['reuse_row_root']
        baseline = tmp_path / 'automation_reports/CV-SincNet' / target.BASELINES[cohort] / 'results'
        dump(baseline / 'startup.json', dict(commit='historical', spec=prior))
        dump(baseline / 'scores.json', inputs[cohort]['scores'])
        dump(result / 'fit_audit.json', fit_audit(900 if cohort == 'rx3' else 300))
        dump(result / 'arithmetic_audit.json', dict(status='VERIFIED'))
        dump(result / 'scores.json', inputs[cohort]['scores'])
        dump(result / 'summary/summary.json', inputs[cohort]['summary'])
    root = tmp_path / 'automation_reports/CV-SincNet' / target.RUNS['rx3'] / 'results/combined_rx4'
    dump(root / 'summary.json', combined)
    (root / 'report.md').write_bytes(b'Existing joint summary\r\n')
    return tmp_path


def test_equal_cells_both_scopes_and_h_mean(synthetic):
    inputs, summary, _ = synthetic
    result = target.interpret(inputs, summary)
    assert result['records'] == 9648 and result['preregistered_guard_pass'] is True
    assert result['candidate_promoted'] is result['goal_complete'] is False
    for row in result['per_k']:
        assert row['joint_cells'] == 960 and row['old_all_cells']['cells'] == 1200
        assert row['joint_delta_pp']['old_accuracy'] == pytest.approx(-2)
        assert row['old_all_cells']['delta_pp'] == pytest.approx(0.4)
        assert row['joint_delta_pp']['new_accuracy'] == pytest.approx(3.5)
        assert row['old_guard_pass'] is True
        base = row['joint_percent']['D92']
        harmonic_of_means = 2 * base['old_accuracy'] * base['new_accuracy'] / (base['old_accuracy'] + base['new_accuracy'])
        assert base['harmonic_mean'] != pytest.approx(harmonic_of_means)
        assert row['seed_consistency']['harmonic_mean']['total'] == 4
    assert len(result['per_model_seed_k']) == 16 and len(result['complete_k_by_new_count']) == 20


def test_failure_is_computed_not_hardcoded(synthetic):
    _, _, specs = synthetic
    scores = [score_rows(s, adverse=True) for s in specs]
    inputs = {c: dict(scores=d, summary=summarize(d, s)) for c, s, d in zip(target.RUNS, specs, scores)}
    result = target.interpret(inputs, combine(specs, scores))
    assert result['preregistered_guard_pass'] is False
    assert '未通过' in result['conclusion']


@pytest.mark.parametrize('damage', ['duplicate_cell', 'wrong_pair', 'missing_dg', 'wrong_weight', 'wrong_old_scope', 'duplicate_table', 'nan'])
def test_reject_incomplete_or_corrupt_summary(synthetic, damage):
    inputs, summary, _ = copy.deepcopy(synthetic)
    rows = inputs['rx3']['scores']['results']
    if damage == 'duplicate_cell': rows[3] = copy.deepcopy(rows[0])
    if damage == 'wrong_pair': rows[1]['split_id'] = 'unpaired'
    if damage == 'missing_dg': rows[:] = [r for r in rows if r['method'] != 'frozen_dg']
    if damage == 'wrong_weight': summary['tables']['per_k'][0]['new_accuracy'] += 0.02
    if damage == 'wrong_old_scope': summary['comparisons'][0]['old_guard_delta'] = -0.02
    if damage == 'duplicate_table': summary['tables']['per_k'][1] = copy.deepcopy(summary['tables']['per_k'][0])
    if damage == 'nan': rows[0]['old_accuracy'] = float('nan')
    with pytest.raises(ValueError): target.interpret(inputs, summary)


def test_training_cost_actual_schema_and_unknown_transfer():
    result = target.costs(fit_audit(300))
    assert result['optimizer_steps'] == 0
    assert result['analytical_fit_calls'] == 3900
    assert result['fold_fit_calls'] == 2700 and result['final_fit_calls'] == 1200
    assert result['model_already_deployed'] is result['model_incremental_transfer_bytes'] is None
    assert result['timing_totals_seconds']['feature_extraction_seconds'] == 0
    assert result['numeric_state_byte_ranges']['persistent_state_bytes'] == [49200, 213200]
    assert result['optimizer_convergence_claim'] is False
    assert 'checkpoint' in target.cost_text(result) and '没有优化器更新' in target.cost_text(result)


@pytest.mark.parametrize('damage', ['model_bytes', 'transfer', 'steps', 'count', 'source', 'nan', 'statebytes', 'stage_sum', 'inherited'])
def test_reject_bad_cost_metadata(damage):
    data = fit_audit(300)
    if damage == 'model_bytes': data['models'][0]['resources']['existing_frozen_model_file_bytes'] = None
    if damage == 'transfer': data['models'][0]['resources']['model_incremental_transfer_bytes'] = 0
    if damage == 'steps': data['optimizer_steps'] += 1
    if damage == 'count': data['models'][0]['fits'] -= 1
    if damage == 'source': data['models'][0]['resources']['new_source_payload_bytes'] = 10
    if damage == 'nan': data['models'][0]['timing']['fit_seconds']['total'] = float('nan')
    if damage == 'statebytes': data['models'][0]['numeric_state_byte_ranges']['head_bytes']['max'] += 1
    if damage == 'stage_sum': data['models'][0]['analytical_stage_timing']['all']['solve_seconds']['total'] += 1
    if damage == 'inherited': data['adapted_state_inherited'] = True
    with pytest.raises(ValueError): target.costs(data)


def test_both_terminal_before_any_scores(tmp_path, synthetic, monkeypatch):
    workspace(tmp_path, synthetic)
    path = tmp_path / 'automation_reports/CV-SincNet' / target.RUNS['rx1'] / 'results/complete.json'
    dump(path, dict(status='RUNNING'))
    original = target.read
    def guard(p):
        assert p.name != 'scores.json', 'Premature target score access'
        return original(p)
    monkeypatch.setattr(target, 'read', guard)
    with pytest.raises(ValueError, match='Both cohorts'): target.prepare(tmp_path)


def test_full_write_preserves_raw_history_and_refuses_repeat(tmp_path, synthetic):
    workspace(tmp_path, synthetic)
    files = [p for p in tmp_path.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    writes, appends = target.prepare(tmp_path)
    assert all(p.read_bytes() == b for p, b in before.items())  # dry-run is read only
    target.persist(writes, appends)
    reports = {p for p, _ in appends}
    for p, b in before.items():
        if p in reports: assert p.read_bytes().startswith(b)
        else: assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(b).digest()
    for p, data in writes: assert json.loads(p.read_text(encoding='utf-8')) == data
    after = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError): target.prepare(tmp_path)
    with pytest.raises(FileExistsError): target.persist(writes, appends)
    assert all(p.read_bytes() == b for p, b in after.items())


def test_conflicting_output_rejected_before_scores(tmp_path, synthetic, monkeypatch):
    workspace(tmp_path, synthetic)
    path = tmp_path / 'automation_reports/CV-SincNet' / target.RUNS['rx1'] / 'results/artifacts.json'
    dump(path, {'already': 'archived'})
    monkeypatch.setattr(target, 'read', lambda p: pytest.fail('Output conflict must be checked before inputs'))
    with pytest.raises(FileExistsError): target.prepare(tmp_path)


@pytest.mark.parametrize('damage', ['metric', 'confusion', 'class_accuracy', 'split', 'duplicate', 'missing_dg'])
def test_original_baseline_must_be_unchanged(synthetic, damage):
    inputs, _, _ = synthetic
    current = inputs['rx3']['scores']; original = copy.deepcopy(current)
    row = next(r for r in original['results'] if r['method'] == 'D92')
    if damage == 'metric': row['old_accuracy'] += .001
    if damage == 'confusion': row['confusion'][0][0] += 1
    if damage == 'class_accuracy': row['class_accuracy'][0] += .001
    if damage == 'split': row['split_id'] = 'wrong'
    if damage == 'duplicate': original['results'].append(copy.deepcopy(row))
    if damage == 'missing_dg': original['results'] = [r for r in original['results'] if r['method'] != 'frozen_dg']
    with pytest.raises(ValueError): target.compare_baseline(current, original, inputs['rx3']['summary']['matrix'])


def test_prior_candidate_not_used_in_baseline_comparison(synthetic):
    inputs, _, _ = synthetic; current = inputs['rx1']['scores']; original = copy.deepcopy(current)
    original['results'] = [r for r in original['results'] if r['method'] != 'D92-OSC-v1']
    original['results'].append({'method': 'unrelated-historical-candidate', 'unread_result': 'not consulted'})
    result = target.compare_baseline(current, original, inputs['rx1']['summary']['matrix'])
    assert result['unchanged_original_records'] == 1212


def test_recovery_run_ids_are_explicit_and_bound(tmp_path, synthetic):
    workspace(tmp_path, synthetic)
    base = tmp_path / 'automation_reports/CV-SincNet'
    runs = {c: run.replace('-r02', '-r03') for c, run in target.RUNS.items()}
    for cohort, old in target.RUNS.items():
        (base / old).rename(base / runs[cohort])
        startup_path = base / runs[cohort] / 'results/startup.json'
        startup = target.read(startup_path); startup['spec']['run_id'] = runs[cohort]; dump(startup_path, startup)
    writes, _ = target.prepare(tmp_path, runs)
    artifacts = [value for path, value in writes if path.name == 'artifacts.json']
    assert {value['run_id'] for value in artifacts} == set(runs.values())
    assert all(value['original_baseline_verification']['status'] == 'VERIFIED' for value in artifacts)


def test_missing_historical_scores_reports_path_without_writing(tmp_path, synthetic):
    workspace(tmp_path, synthetic)
    path = tmp_path / 'automation_reports/CV-SincNet' / target.BASELINES['rx3'] / 'results/scores.json'
    path.unlink()  # Temporary synthetic fixture only.
    with pytest.raises(FileNotFoundError) as error: target.prepare(tmp_path)
    assert Path(error.value.filename) == path
    assert not list(tmp_path.rglob('artifacts.json'))


@pytest.mark.parametrize('field', ['source_root', 'reuse_row_root', 'expected_checkpoint_sha256'])
def test_wrong_baseline_binding_fails_before_any_scores(tmp_path, synthetic, monkeypatch, field):
    workspace(tmp_path, synthetic)
    path = tmp_path / 'automation_reports/CV-SincNet' / target.RUNS['rx3'] / 'results/startup.json'
    startup = target.read(path); startup['spec']['rows'][0][field] = 'wrong'; dump(path, startup)
    original_read = target.read
    def guard(path):
        assert path.name != 'scores.json'; return original_read(path)
    monkeypatch.setattr(target, 'read', guard)
    with pytest.raises(ValueError): target.prepare(tmp_path)


def test_registered_matrix_mismatch_before_scores(tmp_path, synthetic, monkeypatch):
    workspace(tmp_path, synthetic)
    path = tmp_path / 'automation_reports/CV-SincNet' / target.RUNS['rx3'] / 'results/startup.json'
    startup = target.read(path); startup['spec']['data']['scenarios'][0] = 'unregistered'; dump(path, startup)
    original_read = target.read
    def guard(path):
        assert path.name != 'scores.json'; return original_read(path)
    monkeypatch.setattr(target, 'read', guard)
    with pytest.raises(ValueError, match='registered matrix'): target.prepare(tmp_path)
