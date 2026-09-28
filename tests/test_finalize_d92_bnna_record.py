"""Synthetic only: no real run artifacts or target scores are accessed."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import finalize_d92_bnna_record as target
from summarize_d92_confirmation import summarize
from summarize_d92_repeated_benchmark import combine


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def spec(cohort):
    return dict(run_id=target.RUNS[cohort], permissions=dict(claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS'),
        rows=[dict(seeds=dict(model=s)) for s in range(2026092701, 2026092705)],
        data=dict(target_receivers=['a', 'b', 'c'] if cohort == 'rx3' else ['d'], scenarios=['x', 'y', 'z'],
            k=[1, 5, 10, 20], new_class_counts=[0, 2, 5, 10, 20], support_seeds=[11, 12, 13, 14, 15]),
        confirmation=dict(candidate_method='D92-BNNA-v1', candidate_folder='bnna',
            candidate_predictor='predict_d92_bnna.py', candidate_mode='d92_bnna_registration',
            reuse_validated_capsule_id=cohort),
        metrics_plan=dict(acceptance=dict(old_max_drop=0.01, require_new_improvement=True)),
        joint_benchmark=dict(old_accuracy_scope='All paired cells at this K, including old-only registration'))


def score_rows(s, adverse=False):
    data = s['data']
    axes = ([r['seeds']['model'] for r in s['rows']], data['target_receivers'], data['scenarios'],
            data['k'], data['new_class_counts'], data['support_seeds'])
    result = []
    for cell in itertools.product(*axes):
        metadata = dict(zip(target.KEY, cell))
        metadata.update(row_id=str(cell[0]), split_id=str(cell[1:]), classes=['old', 'new'] if cell[4] else ['old'],
                        class_count=2 if cell[4] else 1, query_count=60 if cell[4] else 30)
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
            result.append(dict(metadata, method=method, **metrics))
        if cell[3:] == (1, 0, 11):
            result.append(dict(metadata, method='frozen_dg', **metrics))
    return dict(status='SCORED', selection_feedback_forbidden=True,
                claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS', results=result)


def fit_audit(splits):
    models = []
    for seed in range(2026092701, 2026092705):
        models.append(dict(row_id=str(seed), fits=splits, fixed_k1_fits=splits // 4, support_cv_fits=3 * splits // 4,
            optimizer_steps=208 * splits, final_optimizer_steps=64 * splits,
            final_status_counts={'FIXED_64_STEPS': splits}, selected_arm_counts={'trained': splits},
            timing={k: dict(total=splits * v) for k, v in dict(fit_seconds=1.0, fit_call_seconds=1.1,
                query_score_seconds=0.1, prediction_write_seconds=0.01, total_seconds=1.25).items()},
            numeric_state_byte_ranges={k: dict(min=v, max=v + 100) for k, v in dict(head_bytes=12288,
                basis_bytes=0, gate_bytes=0, persistent_state_bytes=12288).items()},
            resources=dict(model_incremental_transfer_bytes=None, new_source_payload_bytes=0, ground_summary_used=False,
                existing_frozen_model_file_bytes=15992872 + seed % 4, feature_extraction_seconds=12.0,
                received_cache_array_bytes=294400, received_cache_file_bytes=300000),
            peak_process_rss_bytes=50000000, trace_file='/synthetic/' + str(seed) + '/bnna/fit_trace.jsonl'))
    return dict(status='VERIFIED', method='D92-BNNA-v1', models=models, total_fits=4 * splits,
        fixed_k1_fits=splits, support_cv_fits=3 * splits, optimizer_steps=4 * 208 * splits,
        query_rows_used_for_fit=0, source_rows_used_for_fit=0, new_source_payload_bytes=0,
        ground_summary_used=False, optimizer_convergence_claim=False)


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
        dump(result / 'startup.json', dict(commit='synthetic'))
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
    assert result['optimizer_steps'] == 249600
    assert result['final_optimizer_steps'] == 76800
    assert result['fold_optimizer_steps'] == 172800
    assert result['model_already_deployed'] is result['model_incremental_transfer_bytes'] is None
    assert result['timing_totals_seconds']['feature_extraction_seconds'] == 48
    assert result['optimizer_convergence_claim'] is False
    assert 'checkpoint' in target.cost_text(result) and '64 步预算不代表' in target.cost_text(result)


@pytest.mark.parametrize('damage', ['model_bytes', 'transfer', 'steps', 'count', 'source', 'nan'])
def test_reject_bad_cost_metadata(damage):
    data = fit_audit(300)
    if damage == 'model_bytes': data['models'][0]['resources']['existing_frozen_model_file_bytes'] = None
    if damage == 'transfer': data['models'][0]['resources']['model_incremental_transfer_bytes'] = 0
    if damage == 'steps': data['optimizer_steps'] += 1
    if damage == 'count': data['models'][0]['fits'] -= 1
    if damage == 'source': data['models'][0]['resources']['new_source_payload_bytes'] = 10
    if damage == 'nan': data['models'][0]['timing']['fit_seconds']['total'] = float('nan')
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
