"""Actual OSC core integration on synthetic cache/support only."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
import predict_d92_orbit_shared as mod
from test_d92_orbit_feature_cache import inputs as cache_inputs, dump


def inputs(tmp_path, monkeypatch, k=1):
    args = cache_inputs(tmp_path, monkeypatch, k)
    args['config'] = {'algorithm': args.pop('algorithm')}; args['output'] = tmp_path / 'osc'
    return args


@pytest.mark.parametrize('k', [1, 2, 5])
def test_support_only_real_core_and_analytical_logs(tmp_path, monkeypatch, k):
    args = inputs(tmp_path, monkeypatch, k)
    with np.load(args['orbit_features'] / mod.CACHE_NAME) as data:
        z, f = data['identity_views'], data['fft']
    original_fit = mod.fit_orbit_shared; states = []
    def fit(**values):
        np.testing.assert_array_equal(values['support_identity_views'], z[:4*k])
        np.testing.assert_array_equal(values['support_fft'], f[:4*k])
        assert values['support_ids'].tolist() == [f'physical-{i}' for i in range(4*k)]
        state = original_fit(**values); states.append(state); return state
    monkeypatch.setattr(mod, 'fit_orbit_shared', fit)
    original = Path.open
    def guarded(path, *a, **kw):
        if not path.is_relative_to(args['output']):
            assert path.name not in ('truth.json', 'scores.json', 'fit_trace.jsonl', 'compact.jsonl', 'predictions.jsonl', 'final_ssdg.pth')
            assert 'ground' not in path.parts
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded)
    getitem = np.lib.npyio.NpzFile.__getitem__
    def member(data, key):
        assert key != 'iq'; return getitem(data, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', member)
    with threadpool_limits(limits=1):
        mod.predict(**args); state = states[0]
        scores = state.score(z[4*k:], f[4*k:])
        np.testing.assert_array_equal(scores, np.concatenate([state.score(z[i:i+1], f[i:i+1]) for i in range(4*k, len(z))]))
        np.testing.assert_array_equal(scores, state.score(z[4*k:][::-1], f[4*k:][::-1])[::-1])
    out = args['output']; record = mod.read(out / 'predictions.jsonl')
    small = mod.read(out / 'compact.jsonl'); trace = mod.read(out / 'fit_trace.jsonl')
    assert record['mode'] == 'd92_osc_registration'
    assert record['query_ids'] == [f'physical-{i}' for i in range(4*k, len(z))]
    assert small['fold_count'] == (0 if k == 1 else min(k, 3))
    assert small['selection'] == 'none_cv_diagnostic_only' and small['selected'] == 'fixed_orbit_shared'
    assert small['head_bytes'] == small['persistent_state_bytes'] == state.W.nbytes + state.b.nbytes == 4 * 8200
    assert small['optimizer_steps'] == 0 and trace['steps'] == []
    assert small['source_rows_used_for_fit'] == small['query_rows_used_for_fit'] == 0
    assert small['source_validation'] is None and 'user prohibits' in small['source_validation_reason']
    assert all(small[key] is None for key in ('learning_rate', 'gradient_norm', 'epoch'))
    assert 'Analytical' in small['unavailable_reason']
    assert all(small[key] >= 0 for key in ('fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds'))
    assert (small['oof'] is None) == (k == 1)
    if k > 1:
        assert small['oof'] == {key: trace['oof'][key] for key in ('macro_nll', 'old_nll', 'new_nll')}
        for fold in trace['folds']:
            assert not set(fold['train_ids']) & set(fold['heldout_ids'])
            assert set(fold['training']['training_physical_ids']) == set(fold['train_ids'])
            assert set(fold['train_ids']) | set(fold['heldout_ids']) == set(f'physical-{i}' for i in range(4*k))
    stages = [json.loads(line) for line in (out / 'fit_stages.jsonl').read_text().splitlines()]
    assert len(stages) == small['fold_count'] + 1 and stages[-1]['scope'] == 'final'
    assert all(not isinstance(value, (dict, list)) for row in stages for value in row.values())
    with (out / 'fit_stages.csv').open(encoding='utf-8', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == len(stages)
    for assignment in trace['physical_fold_assignment']:
        index = int(assignment['physical_id'].split('-')[-1])
        assert assignment['class_id'] == trace['registered_classes'][index // k]
    startup = mod.read(out / 'startup.json'); payload = startup['payload_audit']
    assert payload['new_ground_statistics_bytes'] == payload['new_source_payload_bytes'] == 0
    assert payload['model_file_bytes'] == 123456 and payload['model_already_deployed'] is None
    assert payload['model_incremental_transfer_bytes'] is None and 'unknown' in payload['model_deployment_unknown_reason']
    assert 'Complete training checkpoint package' in payload['model_file_bytes_scope']
    assert payload['reused_frozen_cache'] and payload['checkpoint_loaded'] is False
    assert startup['adapted_state_inherited'] is False and startup['feature_contract']['fft_norm_floor'] == 1e-8
    assert mod.read(out / 'predictions_complete.json')['payload_audit'] == payload
    with pytest.raises(FileExistsError): mod.predict(**args)


@pytest.mark.parametrize('fault', ['config', 'truth', 'overlap', 'missing_split', 'duplicate_split', 'old_registry'])
def test_invalid_input_never_fits_or_creates_output(tmp_path, monkeypatch, fault):
    args = inputs(tmp_path, monkeypatch)
    split_path = args['capsule'] / 'splits/one.json'; split = mod.read(split_path)
    if fault == 'config': args['config']['algorithm']['unexpected'] = True
    elif fault == 'truth': split['query_labels'] = [0, 1, 2]
    elif fault == 'overlap': split['query_indices'][0] = 0
    elif fault == 'old_registry': split['registered_classes'][:2] = split['registered_classes'][:2][::-1]
    else:
        manifest_path = args['capsule'] / 'manifest.json'; manifest = mod.read(manifest_path)
        manifest['split_count'] = 2; dump(manifest_path, manifest)
        if fault == 'duplicate_split': dump(args['capsule'] / 'splits/two.json', copy.deepcopy(split))
    dump(split_path, split)
    monkeypatch.setattr(mod, 'fit_orbit_shared', lambda **_: pytest.fail('Must reject before fit'))
    with pytest.raises(ValueError): mod.predict(**args)
    assert not args['output'].exists()


def test_physical_tie_policy():
    assert mod.stable_predictions(np.ones((2, 3)), ['z', 'a', 'b']).tolist() == [1, 1]


def test_failed_fit_has_no_complete_marker(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    def fail(**_): raise FloatingPointError('synthetic technical failure')
    monkeypatch.setattr(mod, 'fit_orbit_shared', fail)
    with pytest.raises(FloatingPointError): mod.predict(**args)
    assert not (args['output'] / 'predictions_complete.json').exists()
