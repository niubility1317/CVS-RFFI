"""Pure NumPy full-cache fixtures for the frozen truth-last prediction boundary."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import evaluate_d92_branch_local_margin as mod
import export_d92_branch_features as cache


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def fixture(root, constant=False):
    capsule, features, row = root / 'capsule', root / 'branch_features', root / 'baseline'
    capsule.mkdir(); features.mkdir()
    (capsule / 'splits').mkdir()
    ids = np.array([f'physical-{i:03}' for i in range(9)])
    classes, old = ['TX9', 'TX2', 'TX7'], ['TX9']
    query = [6, 7, 8]
    sha, cid, seed = 'a' * 64, 'synthetic-capsule', 2026092701
    manifest = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
                    capsule_id=cid, received_count=len(ids), signal_shape=[2, 256], split_count=2)
    dump(capsule / 'manifest.json', manifest)
    # Deliberately unusable waveform payload: consumers may access only the ID member.
    np.savez(capsule / 'received.npz', ids=ids, iq=np.full((len(ids), 2, 256), np.nan))
    for name, k, support, labels in [('small', 1, [0, 2, 4], [0, 1, 2]),
                                    ('large', 2, list(range(6)), [0, 0, 1, 1, 2, 2])]:
        dump(capsule / 'splits' / (name + '.json'), dict(manifest, split_id=name, receiver='RX',
            scenario='scene', k=k, support_seed=7, registered_classes=classes,
            support_indices=support, support_labels=labels, query_indices=query))
    rng = np.random.default_rng(61)
    arrays = {key: (np.ones((len(ids), 96 if key == 'fft' else 160), dtype=np.float32)
                   if constant else rng.normal(size=(len(ids), 96 if key == 'fft' else 160)).astype(np.float32))
              for key in (*cache.BRANCHES, 'fft')}
    np.savez(features / cache.CACHE_NAME, **arrays, ids=ids, checkpoint_sha256=np.array(sha),
             capsule_id=np.array(cid), feature_contract_json=np.array(json.dumps(cache.FEATURE_CONTRACT)))
    provenance = dict(checkpoint_sha256=sha, classes=old, model_seed=seed,
        verdict='MATCHED_SOURCE_ONLY_SCRATCH', source_role_comparison='EXACT_MATCH',
        checkpoint_epoch=200, checkpoint_inheritance=[], target_access_before_freeze=False,
        model_file_bytes=100)
    common = dict(schema=cache.CACHE_SCHEMA, checkpoint_sha256=sha, capsule_id=cid,
        model_seed=seed, feature_contract=cache.FEATURE_CONTRACT, native_batch_size=1,
        native_eval=True, truth_read=False, source_data_access=False, adapted_state_inherited=False,
        encoder_updated=False, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_file_bytes=100)
    dump(features / 'startup.json', dict(common, provenance=provenance,
        config=dict(algorithm=deepcopy(cache.local_core().FROZEN_CONFIG)), query_fit_access=False))
    dump(features / 'checkpoint_provenance.json', provenance)
    dump(features / 'features_complete.json', dict(common, status='BRANCH_FEATURES_COMPLETE',
        classes=old, query_used_for_fitting=False, native_parameters_unchanged=True,
        native_buffers_unchanged=True, view_count_per_observation=1, dtype='float32',
        count=len(ids), native_physical_forward_count=len(ids), native_batch_calls=len(ids),
        identity_reference_checks=len(ids), identity_check_additional_encoder_forwards=0,
        shapes={key: list(value.shape) for key, value in arrays.items()},
        feature_array_bytes=sum(value.nbytes for value in arrays.values()), registry_array_bytes=ids.nbytes,
        feature_file_bytes=(features / cache.CACHE_NAME).stat().st_size, smoke_forward_count=1,
        smoke_batch_calls=1, native_total_physical_forward_count=len(ids) + 1,
        native_total_batch_calls=len(ids) + 1, support_cache_reuse_count=0,
        synthetic_smoke=dict(status='PASS', query_rows_read=0, frozen_state_unchanged=True),
        timing=dict(total_seconds=.01)))
    dump(row / 'd92_startup.json', dict(seed=seed, checkpoint_sha256=sha, capsule=str(capsule),
        features=str(row / 'received_features/received_features.npz'), truth_read=False, query_fit_access=False))
    dump(row / 'received_features/checkpoint_provenance.json', provenance)
    args = dict(row_root=row, capsule=capsule, output=root / 'predictions',
        branch_features=features, expected_capsule_id=cid, expected_checkpoint_sha256=sha,
        config=dict(algorithm=deepcopy(mod.local_core().FROZEN_CONFIG)))
    return args, arrays, ids, query


def predict(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.predict(**args)


def test_support_only_fit_singleton_query_and_sharedscore_compatible_artifacts(tmp_path, monkeypatch):
    args, arrays, ids, query = fixture(tmp_path)
    core = mod.local_core()
    actual = core.fit_branch_local_margin
    states, score_counts = [], []
    splits = {p.stem: mod.read(p) for p in (args['capsule'] / 'splits').glob('*.json')}
    by_support = {tuple(ids[s['support_indices']]): s for s in splits.values()}

    def fit(**kw):
        assert kw['arm'] == 'local_margin'
        split = by_support[tuple(kw['support_ids'])]
        assert set(kw['support_ids']).isdisjoint(ids[query])
        assert set(kw) == set(arrays) | {'support_labels', 'support_ids', 'classes', 'old_classes', 'arm', 'log_callback'}
        for key, value in arrays.items():
            np.testing.assert_array_equal(kw[key], value[split['support_indices']])
        state = actual(**kw)
        frozen = state.audit_dict()
        q = {key: value[query] for key, value in arrays.items()}
        scores = state.score(**q)
        singleton = np.concatenate([state.score(**{key: value[i:i+1] for key, value in q.items()})
                                    for i in range(len(query))])
        np.testing.assert_array_equal(scores, singleton)
        np.testing.assert_array_equal(scores[::-1], state.score(**{key: value[::-1] for key, value in q.items()}))
        assert state.audit_dict() == frozen
        states.append(state)

        def individual(**features):
            score_counts.append(len(features['z_id']))
            assert len(features['z_id']) == 1
            before = state.audit_dict()
            result = state.score(**features)
            assert state.audit_dict() == before
            return result

        return SimpleNamespace(classes=state.classes, audit_dict=state.audit_dict, score=individual)

    monkeypatch.setattr(core, 'fit_branch_local_margin', fit)
    with np.load(args['capsule'] / 'received.npz', allow_pickle=False) as archive:
        archive_type = type(archive)
    original_item = archive_type.__getitem__

    def guard(archive, key):
        assert key not in {'iq', 'labels', 'truth', 'roles'}
        return original_item(archive, key)

    monkeypatch.setattr(archive_type, '__getitem__', guard)
    marker = predict(args)
    assert len(states) == marker['split_count'] == 2
    assert marker['factorization_count'] == 0 and marker['optimizer_steps'] > 0
    assert marker['sweep_count'] == sum(s.audit_dict()['sweep_count'] for s in states)
    assert score_counts == [1] * 6
    assert marker['query_used_for_fitting'] is marker['truth_read'] is False
    assert marker['payload_audit']['feature_cache_reused'] is True
    assert marker['payload_audit']['native_total_physical_forward_count'] == 0
    traces, compact = lines(args['output'] / 'fit_trace.jsonl'), lines(args['output'] / 'compact.jsonl')
    for trace, small in zip(traces, compact):
        assert trace['folds'] == [] and trace['oof'] is None and trace['selected'] == 'local_margin'
        assert trace['head_bytes'] == trace['persistent_state_bytes'] == sum(trace['state_array_bytes'].values()) + trace['state_scalar_bytes']
        assert trace['factorization_count'] == trace['final_fit']['factorization_calls'] == 0
        assert [r['physical_id'] for r in trace['support_records']] == trace['final_fit']['training_physical_ids']
        assert small['final_fit']['bandwidth_tau'] == trace['final_fit']['bandwidth_tau']
        assert small['final_fit']['trace_scale'] == trace['final_fit']['trace_scale']
        assert small['final_fit']['nearest_other_class_squared_distance'] == mod.scalar_record(trace['final_fit']['nearest_other_class_squared_distance'])
        assert small['query_rows_used_for_fit'] == small['source_rows_used_for_fit'] == 0
        assert small['optimizer_steps'] == trace['optimizer_steps'] > 0
        assert small['epoch'] == trace['final_fit']['sweeps']
        assert trace['final_fit']['certified'] is True
    assert all(r['mode'] == mod.MODE and r['classes'] == ['TX9', 'TX2', 'TX7']
               and len(r['scores'][0]) == 3 for r in lines(args['output'] / 'predictions.jsonl'))
    import score_d92_confirmation as scorer
    records = scorer.validate_method_records(args['output'], 'D92-BranchLocalMargin-v1',
        mod.read(args['capsule'] / 'manifest.json'), ids, splits,
        dict(candidate_method='D92-BranchLocalMargin-v1', candidate_folder='branch_local_margin',
             candidate_predictor='evaluate_d92_branch_local_margin.py', candidate_mode=mod.MODE))
    assert len(records) == 2


def test_changing_and_reordering_other_queries_does_not_change_fit_or_remaining_scores(tmp_path):
    args, arrays, _, query = fixture(tmp_path)
    predict(args)
    before = {r['split_id']: r for r in lines(args['output'] / 'predictions.jsonl')}
    fits = {r['split_id']: r for r in lines(args['output'] / 'fit_trace.jsonl')}
    path = args['branch_features'] / cache.CACHE_NAME
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    for key in arrays:
        payload[key][query[-1]] *= -3
    np.savez(path, **payload)
    marker_path = args['branch_features'] / 'features_complete.json'
    marker = mod.read(marker_path); marker['feature_file_bytes'] = path.stat().st_size
    dump(marker_path, marker)
    for path in (args['capsule'] / 'splits').glob('*.json'):
        split = mod.read(path); split['query_indices'] = query[::-1]; dump(path, split)
    args['output'] = tmp_path / 'changed_predictions'
    predict(args)
    for record in lines(args['output'] / 'predictions.jsonl'):
        original = before[record['split_id']]
        new_by_id = dict(zip(record['query_ids'], record['scores']))
        for pid, score in zip(original['query_ids'][:-1], original['scores'][:-1]):
            np.testing.assert_array_equal(new_by_id[pid], score)
    for record in lines(args['output'] / 'fit_trace.jsonl'):
        original = fits[record['split_id']]['final_fit']
        for key in ('bandwidth_tau', 'trace_scale', 'interaction_centered_trace', 'radial_centered_trace',
                    'normal_equation_residual', 'training_physical_ids'):
            assert record['final_fit'][key] == original[key]


@pytest.mark.parametrize('fault', ['query_labels', 'query_roles', 'query_class_counts', 'overlap',
    'missing_split', 'config', 'checkpoint_sha', 'inherited_checkpoint', 'target_access',
    'model_seed', 'query_fit', 'wrong_cache_schema', 'extra_labels_member', 'physical_ids'])
def test_invalid_bindings_rejected_before_fit_or_output(tmp_path, monkeypatch, fault):
    args, *_ = fixture(tmp_path)
    monkeypatch.setattr(mod.local_core(), 'fit_branch_local_margin',
                        lambda **kw: pytest.fail('Invalid inputs must never fit'))
    path = args['capsule'] / 'splits/small.json'
    split = mod.read(path)
    if fault in ('query_labels', 'query_roles', 'query_class_counts'):
        split[fault] = [0, 1, 2]; dump(path, split)
    elif fault == 'overlap':
        split['query_indices'][0] = split['support_indices'][0]; dump(path, split)
    elif fault == 'missing_split':
        path.unlink()
    elif fault == 'config':
        args['config']['algorithm']['ridge_coefficient'] = 2
    elif fault in ('inherited_checkpoint', 'target_access'):
        path = args['branch_features'] / 'checkpoint_provenance.json'; value = mod.read(path)
        value['checkpoint_inheritance' if fault == 'inherited_checkpoint' else 'target_access_before_freeze'] = ['other'] if fault == 'inherited_checkpoint' else True
        dump(path, value)
        path = args['branch_features'] / 'startup.json'; startup = mod.read(path)
        startup['provenance'] = value; dump(path, startup)
    elif fault in ('checkpoint_sha', 'model_seed', 'query_fit', 'wrong_cache_schema'):
        path = args['branch_features'] / 'features_complete.json'; value = mod.read(path)
        key, bad = {'checkpoint_sha': ('checkpoint_sha256', 'b' * 64), 'model_seed': ('model_seed', 1),
                    'query_fit': ('query_used_for_fitting', True), 'wrong_cache_schema': ('schema', 'support-only')}[fault]
        value[key] = bad; dump(path, value)
    else:
        path = args['branch_features'] / cache.CACHE_NAME
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if fault == 'extra_labels_member': data['labels'] = np.zeros(len(data['ids']))
        else: data['ids'][0] = 'invalid-id'
        np.savez(path, **data)
    with pytest.raises(ValueError):
        predict(args)
    assert not args['output'].exists()


def test_degenerate_zero_scores_use_physical_ties_and_zero_actual_solves(tmp_path):
    args, _, ids, _ = fixture(tmp_path, constant=True)
    marker = predict(args)
    assert marker['factorization_count'] == 0
    for record in lines(args['output'] / 'predictions.jsonl'):
        np.testing.assert_array_equal(record['scores'], np.zeros((3, 3)))
        assert record['predicted_indices'] == [1, 1, 1]
    import score_d92_confirmation as scorer
    splits = {p.stem: mod.read(p) for p in (args['capsule'] / 'splits').glob('*.json')}
    scorer.validate_method_records(args['output'], 'D92-BranchLocalMargin-v1',
        mod.read(args['capsule'] / 'manifest.json'), ids, splits, dict(candidate_mode=mod.MODE))


def test_failure_after_completed_split_preserves_fit_evidence_without_completion(tmp_path, monkeypatch):
    args, *_ = fixture(tmp_path)
    core = mod.local_core(); original = core.fit_branch_local_margin; calls = []

    def fail_second(**kw):
        calls.append(kw)
        if len(calls) == 2:
            raise core.NumericalFailure('synthetic numerical failure', dict(factorization_calls=0, optimizer_steps=7,
                solver_state={'beta': [[1.]], 'alpha': [[1., -1.]], 'F': [[0., 0.]]},
                training_physical_ids=sorted(kw['support_ids']), completed_stages=[], arm='local_margin'))
        return original(**kw)

    monkeypatch.setattr(core, 'fit_branch_local_margin', fail_second)
    with pytest.raises(core.NumericalFailure): predict(args)
    assert not (args['output'] / 'predictions_complete.json').exists()
    failed = mod.read(args['output'] / 'technical_failure.json')
    assert failed['completed_splits'] == 1 and len(failed['completed_stages']) == 1
    assert 'optimization_trace' not in failed['completed_stages'][0]
    assert 'training_physical_ids' not in failed['completed_stages'][0]
    assert failed['current_fit'] is None
    prior = lines(args['output'] / 'fit_trace.jsonl')[0]['final_fit']
    assert prior['optimization_trace'] and prior['training_physical_ids']
    assert Path(failed['completed_evidence']['fit_trace']) == args['output'] / 'fit_trace.jsonl'
    assert failed['failure_context']['training_physical_ids'] == sorted(calls[-1]['support_ids'])
    assert failed['completed_factorization_count'] == 0
    assert failed['completed_optimizer_steps'] > 0
    assert failed['factorization_count'] == 0
    assert failed['optimizer_steps'] == failed['completed_optimizer_steps'] + 7
    assert failed['failure_context']['solver_state']['beta'] == [[1.]]
    assert len(lines(args['output'] / 'fit_trace.jsonl')) == len(lines(args['output'] / 'predictions.jsonl')) == 1
    assert (args['output'] / 'fit_stages.csv').exists()
    with pytest.raises(FileExistsError): predict(args)


def test_query_failure_preserves_completed_fit_but_no_prediction(tmp_path, monkeypatch):
    args, *_ = fixture(tmp_path)
    core = mod.local_core(); original = core.fit_branch_local_margin

    def invalid_score(**kw):
        state = original(**kw)
        return SimpleNamespace(classes=state.classes, audit_dict=state.audit_dict,
                               score=lambda **features: np.full((1, len(state.classes)), np.nan))

    monkeypatch.setattr(core, 'fit_branch_local_margin', invalid_score)
    with pytest.raises(ValueError, match='individual query scores'): predict(args)
    failed = mod.read(args['output'] / 'technical_failure.json')
    assert failed['completed_splits'] == 0 and len(failed['completed_stages']) == 1
    assert 'optimization_trace' not in failed['completed_stages'][0]
    assert failed['current_fit']['optimization_trace']
    assert failed['current_fit']['training_physical_ids']
    assert Path(failed['completed_evidence']['solver_sweeps']).is_file()
    assert failed['completed_factorization_count'] == 0
    assert failed['completed_optimizer_steps'] > 0
    assert lines(args['output'] / 'predictions.jsonl') == []
    assert len(lines(args['output'] / 'fit_stages.jsonl')) == 1
    assert not (args['output'] / 'predictions_complete.json').exists()


def test_import_does_not_require_torch():
    script = '''import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch forbidden')
sys.meta_path.insert(0, NoTorch())
sys.path[:0] = sys.argv[1:]
import evaluate_d92_branch_local_margin
evaluate_d92_branch_local_margin.local_core()
'''
    result = subprocess.run([sys.executable, '-s', '-c', script, str(ROOT / 'tools'),
                             str(ROOT / 'code')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_actual_sweep_streams_match_full_fit_and_do_not_include_arrays(tmp_path):
    import csv
    args, *_ = fixture(tmp_path)
    marker = predict(args)
    out = args['output']
    traces = lines(out / 'fit_trace.jsonl')
    rows = lines(out / 'solver_sweeps.jsonl')
    assert len(rows) == marker['sweep_count']
    assert marker['optimizer_steps'] == sum(t['optimizer_steps'] for t in traces)
    for record in traces:
        events = [r for r in rows if r['split_id'] == record['split_id']]
        assert len(events) == record['final_fit']['sweeps']
        assert events[-1]['optimizer_steps'] == record['optimizer_steps']
        assert record['optimizer_steps'] == len(events) * record['support_count']
        for event in events:
            assert not any(isinstance(v, list) for v in event.values())
            assert event['learning_rate'] is event['source_validation'] is None
            assert event['scope'] == 'final'
            assert event['source_validation_reason'] and event['learning_rate_reason']
    with (out / 'solver_sweeps.csv').open(encoding='utf-8', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == len(rows)
    text = (out / 'solver_sweeps.log').read_text(encoding='utf-8').splitlines()
    assert len(text) == len(rows)
    assert all('LR=N/A source_validation=N/A measured=' in line for line in text)
    assert len([r for r in traces if r['k'] == 1]) == 1
    assert next(r for r in traces if r['k'] == 1)['optimizer_steps'] > 0
