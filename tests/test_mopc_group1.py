"""Synthetic source/C bindings; no target capsule or query files are opened."""
import json

import numpy as np
import pytest
import torch

from experiments.pamf_phase2_practical import group1, group1_source as source, group1_target as target
from experiments.pamf_phase2_practical.mopc_author import ARCHITECTURE_VERSION, MoPCAuthorWiSig256


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_mopc_catalog_and_source_recipe_bind_author_network_without_generic_factory(monkeypatch):
    row = group1.row('mopc_hr')
    assert row['active'] and row['unavailable_reason'] is None
    assert row['category'] == 'CIL' and row['B'] is None and row['C'] == 'native'
    assert row['native_scope'] == ARCHITECTURE_VERSION
    spec = source.recipe('mopc_hr')
    assert spec['epochs'] == 200 and spec['PL_start_epoch'] == 101
    assert spec['PL_threshold'] == .95 and spec['PL_weight'] == 1.
    assert spec['satellite_concat'] and spec['satellite_weight'] == .68
    assert spec['architecture']['architecture'] == ARCHITECTURE_VERSION
    assert spec['source_metadata'] == 'frozen_A_clean_L_s_old6_prototypes_only'
    assert spec['source_optimizer'] == 'Adam' and spec['source_learning_rate'] == .001
    assert spec['L_batch'] == 128 and spec['U_batch'] == 256
    def forbidden(*args, **kwargs):
        raise AssertionError('The generic comparison encoder cannot construct MoPC')
    monkeypatch.setattr(source.nets, 'factory', forbidden)
    model = source.model_for('mopc_hr', 'cpu')
    assert isinstance(model, MoPCAuthorWiSig256)
    assert source.logits(model.eval(), 'mopc_hr', torch.randn(2, 2, 256)).shape == (2, 6)


def test_frozen_source_checkpoint_exports_only_L_prototypes_and_roundtrips(tmp_path, monkeypatch):
    from experiments.practical_full_2x2 import cache
    torch.manual_seed(3)
    labels = torch.arange(6).repeat_interleave(2)
    class Data:
        def __init__(self, count, hidden=False):
            self.x = torch.randn(count, 2, 256)
            self.metadata = {'rx_i': np.ones(count, dtype=int)}
            if not hidden:
                self.metadata['y'] = labels.numpy()
        def __len__(self):
            return len(self.x)
    datasets = {'L_s': Data(12), 'U_s': Data(24, hidden=True), 'V': Data(12)}
    roles = []
    def source_data(config, role):
        roles.append(role)
        return datasets[role]
    class Reader:
        wait_seconds = 0.
        def __init__(self, config):
            pass
        def get(self, indices, *args):
            return (datasets['L_s'].x[indices] * 1.2).numpy(), {'applied_count': len(indices)}
    monkeypatch.setattr(source, 'source_data', source_data)
    monkeypatch.setattr(source, 'row', lambda profile: dict(method='mopc_hr', epochs=2, PL_start_epoch=2))
    monkeypatch.setattr(cache, 'Reader', Reader)
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'source_rxs': [1]}), encoding='utf-8')
    config = dict(run_root=str(tmp_path), source_contract=str(contract), source_cache='synthetic_source_only')
    output = tmp_path / 'source/mopc_hr-s42'
    source._train_native(config, 'mopc_hr', 42, output, 'cpu')
    records = [json.loads(line) for line in (output / 'epochs.jsonl').read_text().splitlines()]
    assert [record['PL_enabled'] for record in records] == [False, True]
    assert records[0]['PL_seen'] == 0 and records[1]['PL_seen'] == 24
    assert all(record['satellite_loss'] > 0 for record in records)
    payload = torch.load(output / 'source.pt', map_location='cpu', weights_only=False)
    assert payload['initialization'] == 'scratch' and payload['ancestors'] == []
    assert payload['native_state'] == {} and payload['target_contact'] is False
    metadata = payload['base_metadata']
    assert metadata['source_role'] == 'L_s' and metadata['source_rows'] == 12
    assert metadata['computed_after_A_freeze'] and metadata['computed_before_target_adaptation']
    assert metadata['source_class_ids'] == list(range(6))
    assert metadata['old_prototypes'].shape == (6, 256)
    assert [key for key, value in metadata.items() if isinstance(value, torch.Tensor)] == ['old_prototypes']
    bundle = source.load_source(config, 'mopc_hr', 42, 'cpu')
    z = source.extract(bundle['model'], 'mopc_hr', datasets['L_s'].x, 'cpu')
    expected = torch.stack([z[labels == label].mean(0) for label in range(6)])
    torch.testing.assert_close(bundle['base_metadata']['old_prototypes'], expected, rtol=0, atol=0)
    assert bundle['score_fn'](bundle['model'], datasets['L_s'].x[:1]).shape == (1, 6)
    assert roles == ['L_s', 'U_s', 'V']
    # A former generic payload has no author-network identity and cannot be reused.
    payload['recipe']['architecture'].pop('architecture')
    torch.save(payload, output / 'source.pt')
    with pytest.raises(ValueError, match='Source recipe mismatch'):
        source.load_source(config, 'mopc_hr', 42, 'cpu')


def test_direct_C_consumes_frozen_A_prototypes_and_native_five_class_sessions(monkeypatch):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration as cil
    torch.manual_seed(14)
    model = source.model_for('mopc_hr', 'cpu').eval()
    metadata = source.mopc_source_metadata(model, torch.randn(12, 2, 256),
                                           torch.arange(6).repeat_interleave(2), 'cpu')
    model.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    bundle = dict(model=model, method='mopc_hr', profile='mopc_hr', classes=tuple(range(6)),
                  base_metadata=metadata, native_state={})
    def forbidden(*args, **kwargs):
        raise AssertionError('Direct C cannot run old-target B adaptation')
    monkeypatch.setattr(target, 'fit_B', forbidden)
    monkeypatch.setattr(cil, 'prepare_registration_state', forbidden)
    original = cil.register_new_classes
    observed = {}
    def short_native(*args, **kwargs):
        observed.update(kwargs['config'])
        # Exercise actual native session transitions while keeping this no-query
        # implementation test independent of the formal twenty-epoch budget.
        kwargs['config'] = dict(kwargs['config'], mopc_epochs=1)
        return original(*args, **kwargs)
    monkeypatch.setattr(cil, 'register_new_classes', short_native)
    support = torch.randn(12, 2, 256)
    traces = []
    result = target.fit_C({}, bundle, support, torch.arange(6, 12).repeat_interleave(2),
                          tuple(range(6, 12)), 51, traces.append)
    assert observed['old_adaptation_epochs'] == 0
    assert observed['mopc_epochs'] == 20 and observed['mopc_stage_size'] == 5
    assert observed['mopc_semantics'] == 'paper' and observed['mopc_lr'] == .01
    assert [row['registered_count'] for row in traces] == [11, 12]
    assert result['classes'] == tuple(range(12))
    assert result['model'].fc.in_features == 256 and result['model'].fc.out_features == 12
    native = result['native_state']
    assert native.prototypes.shape == (12, 256)
    assert [stage['new_count'] for stage in native.protocol_metadata['incremental_stages']] == [5, 1]
    assert native.protocol_metadata['C_inherits_B_state'] is False
    assert native.protocol_metadata['B_executed'] is False
    assert result['resources']['actual_target_old_labels'] == 0
    assert result['resources']['actual_target_new_labels'] == 12
    assert result['resources']['C_initialization'] == 'A_no_B'
    assert result['score_fn'](result['model'], support[:1]).shape == (1, 12)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    scores_before = {name: value.clone() for name, value in result['model'].state_dict().items()}
    assert torch.isfinite(result['score_fn'](result['model'], support)).all()
    assert all(torch.equal(scores_before[name], value) for name, value in result['model'].state_dict().items())


def test_existing_dispatcher_selects_mopc_source_then_prediction_without_B_jobs(tmp_path):
    from experiments.pamf_phase2_practical.group1_dispatch import jobs
    matrix = jobs(dict(run_root=str(tmp_path), method_profiles=['mopc_hr'],
                       model_seeds=[42], capsules=['synthetic_unopened_capsule']))
    sources = [job for job in matrix if job['kind'] == 'source']
    assert len(sources) == 1 and sources[0]['profile'] == 'mopc_hr'
    predictions = [job for job in matrix if job['kind'] == 'predict']
    assert len(predictions) == 2
    assert sources[0]['id'] in predictions[0]['dependencies']
    assert all(job['profile'] == 'mopc_hr' for job in matrix)
