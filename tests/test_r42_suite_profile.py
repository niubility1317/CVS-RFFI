"""No-GPU tests of profile accounting and strict support reuse semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pamf_phase2_practical import r42_suite_profile as profile


def test_batch_latency_quantiles_are_batch_calls_not_invented_record_quantiles():
    rows = [dict(sample_count=128, **{key: value for key in
        ('pipeline_seconds', 'G_seconds', 'F_seconds', 'NN_seconds', 'head_seconds')}) for value in (1., 3.)]
    summary = profile.latency_summary(rows, 128)
    assert summary['sample_count'] == 256 and summary['batch_count'] == 2
    assert summary['quantile_unit'] == 'whole_batch_call'
    assert summary['stages']['pipeline_seconds']['p50_seconds'] == 2.
    assert summary['stages']['pipeline_seconds']['amortized_seconds_per_record'] == 4 / 256
    assert profile.latency_summary([dict(rows[0], sample_count=1)], 1)['quantile_unit'] == 'individual_record_call'


def test_comparison_covers_kernel_state_but_not_timing_and_requires_same_selected_rule():
    cold = SimpleNamespace(model=SimpleNamespace(alpha=np.eye(2), tau=(np.float64(.5),),
                           ridge=1e-6, config=('sig', .5)), strength=.5, selection={'time': 1.})
    other = SimpleNamespace(model=SimpleNamespace(alpha=np.eye(2) + 1e-8, tau=(.5,),
                            ridge=1e-6, config=('sig', .5)), strength=.5, selection={'time': 2.})
    assert profile.compare_states(cold, other)['passed']
    other.model.ridge = 1e-5
    result = profile.compare_states(cold, other)
    assert not result['passed'] and '/model/ridge' in result['failed_members']
    other.model.ridge = 1e-6
    other.model.alpha[0, 0] += .1
    assert not profile.compare_states(cold, other)['passed']


def test_incremental_support_reuses_exact_old_ids_without_crossing_S_paths():
    old = {'x': np.arange(6).reshape(3, 2)}
    new = {'x': np.ones((2, 2))}
    merged = profile.merge_support(old, new, ['a', 'b', 'c'], ['a', 'b', 'c', 'd', 'e'], [0, 1, 5, 6, 6])
    np.testing.assert_array_equal(merged['x'][:3], old['x'])
    with pytest.raises(ValueError, match='IDs differ'):
        profile.merge_support(old, new, ['b', 'a', 'c'], ['a', 'b', 'c', 'd', 'e'], [0, 1, 5, 6, 6])


class FakeMeasure:
    def __init__(self, *args, **kwargs):
        self.result = {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def fake_snapshot():
    return {'isolation': 'shared_GPU', 'other_compute_processes': [{'pid': 123}]}


def test_support_three_repeats_and_full_refit_match_cold_C(monkeypatch, tmp_path):
    monkeypatch.setattr(profile, 'Measure', FakeMeasure)
    monkeypatch.setattr(profile, 'gpu0_snapshot', fake_snapshot)
    monkeypatch.setattr(profile, 'timed', lambda fn: (fn(), .01))
    monkeypatch.setattr(profile, 'fresh_engine', lambda engine: SimpleNamespace(setup=lambda: []))
    calls = []
    def extract(reader, engine, bridge, columns, arm, raw, restored, batch):
        calls.append((arm, restored, len(raw), batch))
        return {'x': raw[:, 0, :] + int(restored)}, dict(G_seconds=.01 if restored else 0., F_seconds=.02, NN_seconds=.03)
    monkeypatch.setattr(profile, 'extract_path', extract)
    fit_counts = []
    def fit(arm, features, support, spec, old_classes, rule):
        fit_counts.append((arm, spec['new_count'], len(features['x'])))
        return SimpleNamespace(features=features, alpha=features['x'] @ features['x'].T, ridge=1.)
    monkeypatch.setattr(profile, 'fit_head', fit)
    b = dict(x=np.arange(60).reshape(30, 1, 2), ids=np.arange(30), y=np.repeat(np.arange(6), 5))
    c = dict(x=np.arange(260).reshape(130, 1, 2), ids=np.arange(130), y=np.repeat(np.arange(26), 5))
    report = {'support': []}
    heads = profile.support_profiles(tmp_path, report, None, None, None, None, {0: b, 20: c},
        {0: dict(new_count=0, registered_classes=list(range(6))),
         20: dict(new_count=20, registered_classes=list(range(26)))}, {})
    assert len(report['support']) == 2 * 2 * 3 * 3
    assert set(heads) == {('R42', 0), ('R42', 1), ('LocalRidge', 0), ('LocalRidge', 1)}
    assert len(fit_counts) == 36
    assert all(n == (30 if count == 0 else 130) for _, count, n in fit_counts)
    assert sum(n == 100 for _, _, n, _ in calls) == 12
    for item in report['support']:
        if item['telemetry']['old_support_features_reused']:
            assert item['telemetry']['cold_C_comparison']['passed']
            assert item['telemetry']['cold_C_feature_comparison']['passed']
            assert item['telemetry']['B_coefficients_inherited'] is False


def test_query_hooks_separate_cold_ten_record_warmup_and_actual_batches(monkeypatch, tmp_path):
    monkeypatch.setattr(profile, 'Measure', FakeMeasure)
    monkeypatch.setattr(profile, 'gpu0_snapshot', fake_snapshot)
    monkeypatch.setattr(profile, 'fresh_engine', lambda engine: SimpleNamespace(setup=lambda: []))
    def call(reader, engine, bridge, columns, arm, state, raw, path):
        return dict(sample_count=len(raw), pipeline_seconds=.1, G_seconds=.01 if path != 'raw' else 0.,
                    F_seconds=.02, NN_seconds=.03 if arm == 'R42' else 0., head_seconds=.04)
    monkeypatch.setattr(profile, 'query_call', call)
    report = {'query': []}
    heads = {(arm, sp): object() for arm in ('R42', 'LocalRidge') for sp in (0, 1)}
    profile.query_profiles(tmp_path, report, None, None, None, None, heads, np.zeros((256, 2, 256)))
    assert len(report['query']) == 20
    for row in report['query']:
        telemetry = row['telemetry']
        assert sum(c['sample_count'] for c in telemetry['warmup_calls']) == 10
        assert telemetry['warmup_excluded_from_latency']
        summary = telemetry['measured_summary']
        assert summary['sample_count'] == 256
        expected = 256 if summary['requested_batch'] == 1 else 2
        assert summary['batch_count'] == expected


def test_query_fusion_runs_raw_and_restored_actual_extractors(monkeypatch):
    monkeypatch.setattr(profile, 'timed', lambda fn: (fn(), .01))
    paths = []
    def extract(reader, engine, bridge, columns, arm, raw, restored, batch):
        paths.append((arm, restored, batch, raw.copy()))
        return {'values': np.ones((len(raw), 2))}, dict(G_seconds=.5 if restored else 0., F_seconds=1., NN_seconds=.2)
    monkeypatch.setattr(profile, 'extract_path', extract)
    state = SimpleNamespace(evaluate=lambda features: (features['values'], np.ones(len(features['values']))))
    raw = np.ones((3, 2, 256), np.float32)
    result = profile.query_call(None, None, None, None, 'R42', state, raw, 'fusion')
    assert [(arm, restored, batch) for arm, restored, batch, _ in paths] == [('R42', False, 3), ('R42', True, 3)]
    np.testing.assert_array_equal(paths[0][3], paths[1][3])
    assert result['F_seconds'] == 2. and result['G_seconds'] == .5


def test_active_scientific_suite_prevents_any_profile_initialization(monkeypatch, tmp_path):
    from experiments.pamf_phase2_practical import milkr_screen
    monkeypatch.setattr(profile, 'read', lambda path: {'status': 'RUNNING'})
    def forbidden(*args):
        raise AssertionError('The ongoing scientific run cannot be initialized or modified')
    monkeypatch.setattr(milkr_screen, 'initialize', forbidden)
    with pytest.raises(ValueError, match='after scientific scoring'):
        profile.run({'run_root': str(tmp_path)}, tmp_path / 'profile')
    assert not (tmp_path / 'profile').exists()
