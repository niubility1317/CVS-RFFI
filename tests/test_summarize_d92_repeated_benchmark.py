"""Full synthetic cohorts verify pooling and forbid selective or duplicated rows."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_d92_repeated_benchmark import combine
from summarize_d92_confirmation import write_summary
from test_summarize_d92_confirmation import complete_scores, sfhead_spec, sfhead_scores


def cohorts():
    specs = [sfhead_spec(), sfhead_spec()]
    specs[0]['data']['target_receivers'] = ['19-1', '8-14', '8-7']
    data = [complete_scores(lambda method, seed: (.5, .5) if method == 'D92' else (.6, .6)), sfhead_scores()]
    for i, (s, d) in enumerate(zip(specs, data)):
        s['run_id'] = f'synthetic-cohort-{i}'
        s['permissions']['claim_scope'] = 'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS synthetic'
        s['metrics_plan']['acceptance'].update(require_harmonic_improvement=True,
            primary_scope='joint all-four-RX equal-cell per-K; cohort tables descriptive, not independent pass gates')
        s['confirmation'].update(candidate_method='D92-SGJoint-v1', candidate_folder='sgjoint',
            candidate_predictor='predict_d92_summary_joint.py', candidate_mode='d92_sgjoint_registration',
            reuse_validated_capsule_id=f'synthetic-capsule-{i}', expected_split_count=900 if i == 0 else 300,
            splits_per_model=900 if i == 0 else 300, predictions_total=7236 if i == 0 else 2412)
        for r in d['results']:
            if r['method'].startswith('D92-'):
                r['method'] = 'D92-SGJoint-v1'
                if i == 1 and r['new_count']:
                    for key in ('accuracy', 'old_accuracy', 'new_accuracy', 'harmonic_mean', 'macro_f1',
                                'old_macro_f1', 'new_macro_f1', 'old_floor', 'new_floor'):
                        r[key] = .8
    return specs, data


def test_equal_receiver_cells_not_equal_cohort_weight_and_explicit_exposure(tmp_path):
    specs, data = cohorts()
    result = combine(specs, data)
    assert result['rows'] == 9648
    assert result['preregistered_guard_pass'] is True
    assert len(result['source_cohorts']) == 2
    assert 'not a new independent confirmation' in result['claim_scope']
    for row in result['tables']['per_k']:
        assert row['cells'] == 960
        if row['method'] == 'D92-SGJoint-v1':
            assert row['harmonic_mean'] == pytest.approx(.65)
    write_summary(result, tmp_path / 'report')
    assert '重复基准比较' in (tmp_path / 'report/report.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        write_summary(result, tmp_path / 'report')


@pytest.mark.parametrize('fault', ['missing_cohort', 'missing_row', 'duplicate_run', 'duplicate_capsule',
                                  'unexposed_claim', 'overlap_receivers', 'different_support_seeds'])
def test_reject_incomplete_incompatible_or_mislabeled_pool(fault):
    specs, data = cohorts()
    if fault == 'missing_cohort':
        specs.pop(); data.pop()
    elif fault == 'missing_row':
        data[1]['results'].pop()
    elif fault == 'duplicate_run':
        specs[1]['run_id'] = specs[0]['run_id']
    elif fault == 'duplicate_capsule':
        specs[1]['confirmation']['reuse_validated_capsule_id'] = specs[0]['confirmation']['reuse_validated_capsule_id']
    elif fault == 'unexposed_claim':
        specs[1]['permissions']['claim_scope'] = 'Fresh independent confirmation'
    elif fault == 'overlap_receivers':
        specs[1]['data']['target_receivers'] = ['19-1']
        for r in data[1]['results']:
            r['receiver'] = '19-1'
    else:
        specs[1]['data']['support_seeds'][0] += 100
        for r in data[1]['results']:
            if r['support_seed'] == 2026092711:
                r['support_seed'] += 100
    with pytest.raises(ValueError):
        combine(specs, data)


def test_inputs_unchanged_after_pooling():
    specs, data = cohorts()
    before = deepcopy((specs, data))
    combine(specs, data)
    assert (specs, data) == before


def test_old_guard_includes_old_only_when_preregistered():
    specs,data=cohorts()
    for spec,scored in zip(specs,data):
        spec['joint_benchmark']={'old_accuracy_scope':'All paired cells at this K, including old-only registration'}
        for row in scored['results']:
            if row['method']=='D92-SGJoint-v1':
                row['old_accuracy']=.51 if row['new_count'] else 0.0
            elif row['method']=='D92':
                row['old_accuracy']=.5
    result=combine(specs,data)
    assert all(row['delta']['old_accuracy']==pytest.approx(.01) for row in result['comparisons'])
    assert all(row['old_guard_delta']==pytest.approx(-.092) and not row['old_guard'] for row in result['comparisons'])
    assert not result['preregistered_guard_pass']


@pytest.mark.parametrize('candidate_index',[3,4,5])
def test_new_candidate_pool_keeps_method_binding_and_strict_new_guard(candidate_index):
    from run_d92_confirmation import CANDIDATES,CANDIDATE_FIELDS
    candidate=CANDIDATES[candidate_index]
    specs,data=cohorts()
    for spec,scored in zip(specs,data):
        spec['confirmation'].update(dict(zip(CANDIDATE_FIELDS,candidate)))
        for row in scored['results']:
            if row['method']=='D92-SGJoint-v1':row['method']=candidate[0]
    result=combine(specs,data)
    assert result['methods']==['D92',candidate[0]]
    assert result['acceptance']['require_new_improvement']
    assert 'D92-SGJoint' not in result['claim_scope']


@pytest.mark.parametrize('field,value', [('require_harmonic_improvement', False),
                                        ('primary_scope', 'choose best receiver')])
def test_reject_changed_preregistered_acceptance(field, value):
    specs, data = cohorts()
    specs[0]['metrics_plan']['acceptance'][field] = value
    with pytest.raises(ValueError):
        combine(specs, data)
