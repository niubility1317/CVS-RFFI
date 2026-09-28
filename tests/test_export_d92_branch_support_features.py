"""Synthetic support-only boundaries; no actual model/data/remote access."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import export_d92_branch_support_features as mod

SHA = 'a'*64
CLASSES = ['old-z', 'old-a', 'new-n']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class TinyBackbone(torch.nn.Module):
    emb_dim = 160
    input_len = 256
    use_time_path = use_freq_path = use_pa_path = True
    use_dac_path = False

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 160)
        self.norm = torch.nn.BatchNorm1d(160)
        self.t_proj = torch.nn.Linear(160, 160)
        self.f_proj = torch.nn.Linear(160, 160)
        self.calls = 0

    def forward(self, x, **kwargs):
        assert not torch.is_grad_enabled() and not self.training
        assert kwargs['y'] is None and kwargs['domain_labels'] is None and kwargs['return_aux']
        self.calls += 1
        value = self.norm(self.linear(x.flatten(1)))
        return dict(feat_joint=value, t_emb=value*2, f_emb=-value,
                    pa_local=value*.5, logits=value[:, :2])


class TinyModel(torch.nn.Module):
    id_feature_key = 'feat_joint'

    def __init__(self):
        super().__init__(); self.id_backbone = TinyBackbone()

    def _pick_z_id(self, aux):
        return aux[self.id_feature_key]


def forward_module():
    def backbone_forward_compat(model, x, **kwargs): return model(x, **kwargs)
    def identity_only_feature_forward(model, x, name):
        assert name == 'z_id'
        aux = backbone_forward_compat(model.id_backbone, x, y=None, return_aux=True, domain_labels=None)
        return model._pick_z_id(aux).float(), aux['logits'].float()
    return SimpleNamespace(backbone_forward_compat=backbone_forward_compat,
                           identity_only_feature_forward=identity_only_feature_forward)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1); torch.manual_seed(921)
    capsule = tmp_path/'capsule'; capsule.mkdir()
    iq = np.random.default_rng(13).normal(size=(9, 2, 256)).astype(np.float32)
    support = [6, 1, 4, 0, 7, 3]
    query = [2, 5, 8]
    # Poison query values: any eager full-array finite check/FFT would fail.
    iq[query] = np.nan
    ids = np.asarray([f'physical-{8-i}' for i in range(len(iq))])
    np.savez(capsule/'received.npz', iq=iq, ids=ids)
    manifest = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE', capsule_id='capsule-test',
                    signal_shape=[2, 256], received_count=9, split_count=2, scenarios=['test-scene'])
    dump(capsule/'manifest.json', manifest)
    for sid, selected, k in [('small', support[::2], 1), ('large', support, 2)]:
        dump(capsule/'splits'/f'{sid}.json', dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
            capsule_id='capsule-test', split_id=sid, receiver='target-rx', scenario='test-scene', k=k,
            registered_classes=CLASSES, support_indices=selected, support_labels=np.repeat(np.arange(3), k).tolist(),
            query_indices=query, support_seed=12))
    infer = mod.FrozenBranches(TinyModel(), forward_module(), torch.device('cpu'))
    provenance = dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH', source_role_comparison='EXACT_MATCH',
        checkpoint_sha256=SHA, checkpoint_epoch=200, checkpoint_inheritance=[], target_access_before_freeze=False,
        classes=CLASSES[:2], model_seed=2026092701, model_file_bytes=123456, active_branches=infer.attributes)
    monkeypatch.setattr(mod, 'load_native', lambda **kw: (infer, copy.deepcopy(provenance)))
    args = dict(checkpoint_root=tmp_path/'unopened_source', native_code=tmp_path/'unopened_native',
        source_contract=tmp_path/'unopened_contract.json', source_receivers=['source-rx'], seed=2026092701,
        capsule=capsule, output=tmp_path/'output', expected_checkpoint_sha256=SHA,
        expected_capsule_id='capsule-test', device='cpu', batch_size=2)
    return args, iq, ids, support, query, infer


def test_support_only_mmap_deduplicates_and_exports_complete_bound_schema(fixture, monkeypatch):
    args, iq, ids, support, query, infer = fixture
    np_load = np.load
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('Exporter must not use eager np.load'))
    original_get = np.memmap.__getitem__; reads = []
    def guarded_get(array, index):
        assert isinstance(index, np.ndarray) and set(index).issubset(support)
        reads.extend(index.tolist()); return original_get(array, index)
    monkeypatch.setattr(np.memmap, '__getitem__', guarded_get)
    def no_bridge(*a, **kw): raise RuntimeError('Disabled NumPy tensor ABI bridge')
    monkeypatch.setattr(torch.Tensor, 'numpy', no_bridge)
    monkeypatch.setattr(torch, 'as_tensor', no_bridge)
    marker = mod.export(**args)
    assert set(reads) == set(support) and len(reads) == 6 and not set(reads) & set(query)
    assert infer.model.id_backbone.calls == marker['native_batch_calls'] == 3
    assert marker['native_physical_forward_count'] == marker['identity_reference_checks'] == marker['count'] == 6
    assert marker['identity_check_additional_encoder_forwards'] == marker['query_rows_read'] == 0
    assert marker['feature_array_bytes'] == 6*(4*160+96)*4
    assert marker['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE'
    assert marker['view_count_per_observation'] == 1 and marker['new_source_payload_bytes'] == 0
    assert marker['native_parameters_unchanged'] and marker['native_buffers_unchanged']
    assert set(marker['timing']) == {'metadata_seconds', 'checkpoint_load_seconds', 'support_read_seconds',
        'native_forward_seconds', 'fft_seconds', 'frozen_verify_seconds', 'write_seconds', 'total_seconds'}
    with np_load(args['output']/mod.CACHE_NAME, allow_pickle=False) as data:
        assert set(data.files) == {'ids', 'labels', 'indices', *mod.BRANCHES, 'fft', 'capsule_id',
                                   'checkpoint_sha256', 'feature_contract_json'}
        assert data['ids'].tolist() == sorted(ids[support].tolist())
        np.testing.assert_array_equal(data['ids'], ids[data['indices']])
        assert all(data[k].dtype == np.float32 for k in (*mod.BRANCHES, 'fft'))
        assert all(np.isfinite(data[k]).all() for k in (*mod.BRANCHES, 'fft'))
        assert json.loads(data['feature_contract_json'].item()) == mod.FEATURE_CONTRACT
    splits = mod.read(args['output']/'support_splits.json')
    assert len(splits['splits']) == 2 and all('query_indices' not in row for row in splits['splits'])
    for row in splits['splits']:
        assert row['support_ids'] == ids[row['support_indices']].tolist()
    startup = mod.read(args['output']/'startup.json')
    assert startup['provenance'] == mod.read(args['output']/'checkpoint_provenance.json')
    assert startup['model_already_deployed'] is startup['model_incremental_transfer_bytes'] is None
    with pytest.raises(FileExistsError): mod.export(**args)


@pytest.mark.parametrize('fault', ['compressed', 'shape', 'dtype', 'fortran', 'member', 'duplicate_ids'])
def test_unsupported_received_layout_rejected_without_waveform_read(fixture, fault):
    args, iq, ids, *_ = fixture
    if fault == 'shape': iq = iq.transpose(0, 2, 1)
    if fault == 'dtype': iq = iq.astype(np.float64)
    if fault == 'fortran': iq = np.asfortranarray(iq)
    if fault == 'duplicate_ids': ids[0] = ids[1]
    writer = np.savez_compressed if fault == 'compressed' else np.savez
    writer(args['capsule']/'received.npz', iq=iq, ids=ids, **({'truth': [0]} if fault == 'member' else {}))
    with pytest.raises(ValueError): mod.export(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('fault', ['capsule', 'missing_split', 'query_truth', 'overlap', 'cross_overlap', 'label', 'receiver', 'k'])
def test_bad_support_metadata_stops_before_encoder_load(fixture, monkeypatch, fault):
    args, _, _, _, _, _ = fixture
    monkeypatch.setattr(mod, 'load_native', lambda **kw: pytest.fail('Must validate metadata before model load'))
    path = args['capsule']/'splits/small.json'; row = mod.read(path)
    if fault == 'capsule': args['expected_capsule_id'] = 'wrong'
    if fault == 'missing_split': path.unlink()
    if fault == 'query_truth': row['query_labels'] = [0, 1, 2]
    if fault == 'overlap': row['query_indices'][0] = row['support_indices'][0]
    if fault == 'cross_overlap': row['query_indices'][0] = 1
    if fault == 'label': row['support_labels'] = [1, 0, 2]
    if fault == 'receiver': row['receiver'] = 'source-rx'
    if fault == 'k': row['k'] = 2
    if fault != 'missing_split': dump(path, row)
    with pytest.raises(ValueError): mod.export(**args)
    assert not args['output'].exists()


def test_mmap_reader_forbids_query_indices(fixture):
    args, _, _, support, query, _ = fixture
    reader = mod.SupportIQ(args['capsule']/'received.npz')
    try:
        reader.allow(support)
        with pytest.raises(ValueError, match='non-support'): reader.take(query)
        assert reader.rows_read == 0
    finally: reader.close()


@pytest.mark.parametrize('fault', ['parameter_data', 'buffer', 'training', 'requires_grad'])
def test_frozen_state_rejects_even_unversioned_parameter_changes(fixture, fault):
    _, _, _, _, _, infer = fixture
    if fault == 'parameter_data': next(infer.model.parameters()).data.add_(1)
    if fault == 'buffer': next(infer.model.buffers()).add_(1)
    if fault == 'training': infer.model.train()
    if fault == 'requires_grad': next(infer.model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError): infer.verify_frozen()


def test_mid_forward_state_change_leaves_failure_not_complete(fixture, monkeypatch):
    args, _, _, _, _, infer = fixture
    original = infer.forward.backbone_forward_compat
    def changed(model, x, **kwargs):
        result = original(model, x, **kwargs)
        if model is infer.model.id_backbone: next(model.buffers()).add_(1)
        return result
    infer.forward.backbone_forward_compat = changed
    with pytest.raises(ValueError, match='buffer'): mod.export(**args)
    assert mod.read(args['output']/'technical_failure.json')['status'] == 'TECHNICAL_FAILURE'
    assert not (args['output']/'features_complete.json').exists()


def test_fft_recipe_matches_existing_formula_and_is_sample_local(fixture):
    _, iq, _, support, _, _ = fixture
    import export_d92_bnna_features
    selected = iq[support]
    result = mod.historical_fft96(selected)
    np.testing.assert_array_equal(result, export_d92_bnna_features.local_core().received_fft96(selected).astype(np.float32))
    np.testing.assert_array_equal(result, np.concatenate([mod.historical_fft96(row[None]) for row in selected]))
    np.testing.assert_array_equal(result, mod.historical_fft96(selected[::-1])[::-1])


def test_encoder_matches_real_identity_helper_without_second_backbone_forward():
    import cvsrffi.identity_only_forward as native
    torch.manual_seed(94); model = TinyModel()
    infer = mod.FrozenBranches(model, native, torch.device('cpu'))
    iq = np.random.default_rng(7).normal(size=(2, 2, 256)).astype(np.float32)
    result = infer(iq)
    assert model.id_backbone.calls == infer.batch_calls == 1
    assert infer.identity_reference_checks == 2 and result['z_id'].shape == (2, 160)
    infer.verify_frozen()


@pytest.mark.parametrize('fault', ['dac', 'pa', 'time', 'identity'])
def test_unregistered_architecture_rejected(fault):
    model = TinyModel()
    if fault == 'dac': model.id_backbone.use_dac_path = True
    if fault == 'pa': model.id_backbone.use_pa_path = False
    if fault == 'time': model.id_backbone.use_time_path = False
    if fault == 'identity': model.id_feature_key = 'feat_cls'
    with pytest.raises(ValueError): mod.FrozenBranches(model, forward_module(), torch.device('cpu'))


def test_output_race_does_not_write_into_another_owners_directory(fixture, monkeypatch):
    args, _, _, _, _, infer = fixture
    original = mod.load_native
    def load(**kwargs):
        value = original(**kwargs)
        args['output'].mkdir(); (args['output']/'other-owner.txt').write_text('preserve', encoding='utf-8')
        return value
    monkeypatch.setattr(mod, 'load_native', load)
    with pytest.raises(FileExistsError): mod.export(**args)
    assert [p.name for p in args['output'].iterdir()] == ['other-owner.txt']


@pytest.mark.parametrize('fault', [None, 'sha', 'seed', 'epoch', 'ancestor'])
def test_exact_loader_verifies_lineage_before_any_forward(tmp_path, monkeypatch, fault):
    import cvsrffi.checkpoint_loading as loader
    source = tmp_path/'source'; source.mkdir()
    checkpoint = source/'final_ssdg.pth'; checkpoint.write_bytes(b'synthetic-model-not-a-pickle')
    contract = tmp_path/'contract.json'; dump(contract, {})
    events = []
    def verify(path, reference, seed):
        events.append('verify')
        assert path == source and seed == 2026092701
        return dict(classes=CLASSES[:2], num_classes=2), dict(seed=seed)
    monkeypatch.setattr(mod, 'verify_source', verify)
    payload = dict(epoch=200, args=dict(seed=2026092701, from_scratch=True, baseline_ckpt='', teacher_ckpt=''))
    if fault == 'seed': payload['args']['seed'] += 1
    if fault == 'epoch': payload['epoch'] -= 1
    if fault == 'ancestor': payload['args']['teacher_ckpt'] = '/unknown/model.pth'
    def load(*args, **kwargs): events.append('checkpoint'); return payload
    monkeypatch.setattr(torch, 'load', load)
    model = TinyModel()
    def exact(*args, **kwargs):
        events.append('exact'); return model, dict(checkpoint_load_strict=True, missing_keys=0, unexpected_keys=0)
    monkeypatch.setattr(loader, 'build_exact_ssdg_model_from_checkpoint', exact)
    kwargs = dict(checkpoint_root=source, native_code=ROOT/'code', source_contract=contract,
                  seed=2026092701, expected_checkpoint_sha256=mod.sha256(checkpoint), device='cpu')
    if fault == 'sha': kwargs['expected_checkpoint_sha256'] = SHA
    if fault is None:
        infer, provenance = mod.load_native(**kwargs)
        assert events == ['verify', 'checkpoint', 'exact']
        assert infer.batch_calls == model.id_backbone.calls == 0
        assert provenance['checkpoint_inheritance'] == [] and provenance['checkpoint_epoch'] == 200
    else:
        with pytest.raises(ValueError): mod.load_native(**kwargs)
        assert events == (['verify'] if fault == 'sha' else ['verify', 'checkpoint'])
    assert model.id_backbone.calls == 0


def test_nonfinite_support_fails_but_poisoned_query_is_never_validated(fixture):
    args, iq, ids, support, _, _ = fixture
    iq[support[0], 0, 0] = np.nan
    np.savez(args['capsule']/'received.npz', iq=iq, ids=ids)
    with pytest.raises(ValueError, match='Nonfinite support'): mod.export(**args)
    assert not (args['output']/'features_complete.json').exists()


def test_original_identity_reference_disagreement_is_rejected(fixture):
    _, iq, _, support, _, infer = fixture
    original = infer.forward.identity_only_feature_forward
    def wrong(*args):
        identity, logits = original(*args); return identity+1, logits
    infer.forward.identity_only_feature_forward = wrong
    with pytest.raises(ValueError, match='identity exporter'): infer(iq[support[:2]])
