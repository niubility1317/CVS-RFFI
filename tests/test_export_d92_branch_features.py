"""Complete branch cache tests using synthetic IQ and a tiny frozen encoder."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import export_d92_branch_features as mod
import export_d92_branch_support_features as support_mod
from test_export_d92_branch_support_features import fixture as support_fixture, dump, SHA


@pytest.fixture
def full_fixture(tmp_path, monkeypatch):
    original, iq, ids, support, query, infer = support_fixture.__wrapped__(tmp_path, monkeypatch)
    iq = np.nan_to_num(iq, nan=.125)
    np.savez(original['capsule']/'received.npz', iq=iq, ids=ids)
    _, provenance = support_mod.load_native()
    row = tmp_path/'baseline'
    dump(row/'d92_startup.json', dict(seed=original['seed'], checkpoint_sha256=SHA,
        capsule=str(original['capsule']), features=str(row/'received_features/received_features.npz'), truth_read=False, query_fit_access=False))
    dump(row/'received_features/checkpoint_provenance.json', provenance)
    monkeypatch.setattr(mod, 'load_native', lambda **kw: (infer, copy.deepcopy(provenance)))
    args = dict(row_root=row, source=original['checkpoint_root'], contract=original['source_contract'],
        native_code=original['native_code'], seed=original['seed'], capsule=original['capsule'], output=tmp_path/'full_features',
        config=dict(algorithm=copy.deepcopy(mod.local_core().FROZEN_CONFIG)), expected_capsule_id='capsule-test',
        expected_checkpoint_sha256=SHA, device='cpu', batch_size=3)
    return args, iq, ids, infer, support, query


def test_full_export_exact_singleton_forward_and_label_free_cache(full_fixture):
    args, iq, ids, infer, _, _ = full_fixture
    marker = mod.export(**args)
    assert marker['status'] == 'BRANCH_FEATURES_COMPLETE' and marker['schema'] != support_mod.CACHE_SCHEMA
    assert marker['native_batch_calls'] == marker['native_physical_forward_count'] == len(iq)
    assert marker['native_total_batch_calls'] == marker['native_total_physical_forward_count'] == len(iq)+1
    assert marker['smoke_forward_count'] == marker['smoke_batch_calls'] == 1
    assert marker['native_batch_size'] == marker['view_count_per_observation'] == 1
    assert marker['query_used_for_fitting'] is marker['source_data_access'] is marker['truth_read'] is False
    with np.load(args['output']/mod.CACHE_NAME, allow_pickle=False) as data:
        assert set(data.files) == {*mod.BRANCHES, 'fft', 'ids', 'capsule_id', 'checkpoint_sha256', 'feature_contract_json'}
        for index in reversed(range(len(iq))):
            one = infer(iq[index:index+1])
            for key in mod.BRANCHES: np.testing.assert_array_equal(data[key][index], one[key][0])
        np.testing.assert_array_equal(data['ids'], ids)
    with pytest.raises(FileExistsError): mod.export(**args)


def test_io_chunk_size_does_not_change_features(full_fixture, monkeypatch):
    args, iq, _, infer, _, _ = full_fixture
    mod.export(**args)
    original = args['output']; args['output'] = original.parent/'different_io_chunk'; args['batch_size'] = 1
    # Counter reset only; no changes to the frozen network or its buffers.
    infer.batch_calls = infer.physical_forward_count = infer.identity_reference_checks = 0
    mod.export(**args)
    with np.load(original/mod.CACHE_NAME) as left, np.load(args['output']/mod.CACHE_NAME) as right:
        for key in (*mod.BRANCHES, 'fft'): np.testing.assert_array_equal(left[key], right[key])


def test_load_features_reads_identifiers_but_never_original_iq(full_fixture, monkeypatch):
    args, _, _, _, _, _ = full_fixture; mod.export(**args)
    original = np.lib.npyio.NpzFile.__getitem__
    def guard(archive, key):
        assert key != 'iq', 'Predictor must not read waveform values'
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', guard)
    arrays, ids, marker, _, _ = mod.load_features(branch_features=args['output'], **{k:args[k] for k in
        ('row_root', 'capsule', 'expected_capsule_id', 'expected_checkpoint_sha256', 'config')})
    assert all(not value.flags.writeable for value in arrays.values()) and not ids.flags.writeable
    assert marker['count'] == len(ids)


@pytest.mark.parametrize('fault', ['support_schema', 'sha', 'capsule', 'seed', 'query_fit', 'frozen', 'native_batch', 'bytes', 'config'])
def test_cache_binding_or_scope_mismatch_rejected(full_fixture, fault):
    args, *_ = full_fixture; mod.export(**args)
    path = args['output']/'features_complete.json'; marker = mod.read(path)
    if fault == 'support_schema': marker['schema'] = support_mod.CACHE_SCHEMA
    if fault == 'sha': marker['checkpoint_sha256'] = 'b'*64
    if fault == 'capsule': marker['capsule_id'] = 'other'
    if fault == 'seed': marker['model_seed'] += 1
    if fault == 'query_fit': marker['query_used_for_fitting'] = True
    if fault == 'frozen': marker['native_parameters_unchanged'] = False
    if fault == 'native_batch': marker['native_batch_size'] = 2
    if fault == 'bytes': marker['feature_array_bytes'] += 1
    if fault == 'config': args['config']['algorithm']['ridge_coefficient'] = 2
    dump(path, marker)
    with pytest.raises(ValueError): mod.load_features(branch_features=args['output'], **{k:args[k] for k in
        ('row_root', 'capsule', 'expected_capsule_id', 'expected_checkpoint_sha256', 'config')})


def test_scalar_bridge_works_without_tensor_numpy(full_fixture, monkeypatch):
    args, *_ = full_fixture
    def disabled(*args, **kw): raise RuntimeError('ABI bridge prohibited')
    monkeypatch.setattr(torch.Tensor, 'numpy', disabled); monkeypatch.setattr(torch, 'as_tensor', disabled)
    assert mod.export(**args)['status'] == 'BRANCH_FEATURES_COMPLETE'


def test_complete_cache_rejects_labels_member(full_fixture):
    args, *_ = full_fixture; mod.export(**args)
    path = args['output']/mod.CACHE_NAME
    with np.load(path) as archive: data = {key:archive[key] for key in archive.files}
    np.savez(path, **data, labels=np.zeros(len(data['ids']), dtype=int))
    with pytest.raises(ValueError, match='members'): mod.load_features(branch_features=args['output'], **{k:args[k] for k in
        ('row_root', 'capsule', 'expected_capsule_id', 'expected_checkpoint_sha256', 'config')})


def test_synthetic_smoke_precedes_any_received_read(full_fixture, monkeypatch):
    args, _, _, infer, *_ = full_fixture
    original = mod.SupportIQ.take
    def guarded_take(reader, indices):
        assert infer.physical_forward_count >= 1
        assert infer.model.id_backbone.calls >= 1
        return original(reader, indices)
    monkeypatch.setattr(mod.SupportIQ, 'take', guarded_take)
    marker = mod.export(**args)
    assert marker['synthetic_smoke']['feature_shape'] == [1, 736]
    assert marker['synthetic_smoke']['query_rows_read'] == 0
    assert marker['timing']['synthetic_smoke_seconds'] > 0


def test_failed_synthetic_smoke_reads_no_received_rows(full_fixture, monkeypatch):
    args, *_ = full_fixture
    monkeypatch.setattr(mod.SupportIQ, 'take', lambda *a: pytest.fail('No received read before smoke succeeds'))
    monkeypatch.setattr(mod.local_core(), '_features', lambda **kw: np.full((1, 736), np.nan))
    with pytest.raises(ValueError, match='smoke'): mod.export(**args)
    assert not (args['output']/'features_complete.json').exists()
    assert mod.read(args['output']/'technical_failure.json')['status'] == 'TECHNICAL_FAILURE'
