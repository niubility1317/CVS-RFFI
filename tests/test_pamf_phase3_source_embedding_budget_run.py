import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

import test_pamf_phase3_source_risk_run as old


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_budget_run.py'
    assert path.is_file(), 'Budget fit/predict is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_budget_run')


def setup(tmp_path, monkeypatch):
    m = module(); torch.set_num_threads(2)
    base = old.SourceRiskRunTests().fixture(tmp_path, m)
    export = tmp_path / 'new-export'; shutil.copytree(base / 'export', export)
    marker = dict(arm='supervised_only', config={'parent_run_root': str(base),
            'run_id': '20261029-phase3-source-embedding-budget-s2026092701-r01',
            'row_id': 'supervised_only-s2026092701', 'code_commit': 'a' * 40},
        encoder_state_identity={'arm': 'supervised_only', 'row_id': 'supervised_only-s2026092701'},
        parent_export_identity={'original': 'F6'}, source_OE_handles={}, counts=old.SMALL)
    monkeypatch.setattr(m, '_export_marker', lambda root: marker)
    monkeypatch.setattr(m.parent, 'COUNTS', old.SMALL)
    return m, base, export, marker


def test_own_l_geometry_and_200_readouts_then_baseline_risk_smoke_before_v_novel(tmp_path, monkeypatch):
    m, base, export, marker = setup(tmp_path, monkeypatch)
    loader = m.parent._load_features; roles = []
    def training(root, role, include_y=False):
        assert role in ('source_L', 'exposure_train'); roles.append(role)
        return loader(root, role, include_y)
    monkeypatch.setattr(m.parent, '_load_features', training)
    fit = m.fit(export, 2026092701, tmp_path / 'fit')
    assert roles == ['source_L', 'exposure_train']
    assert fit['status'] == 'SOURCE_BUDGET_READOUTS_FROZEN'
    assert fit['historical_encoder_updates'] == 2220 and fit['new_encoder_updates'] == 0
    assert fit['readout_steps_per_model'] == 200
    frozen = {name: (tmp_path / 'fit' / name).read_bytes() for name in ['geometry.npz', 'readouts.npz']}
    roles.clear(); output = tmp_path / 'predict'
    def evaluation(root, role, include_y=False):
        assert not include_y; roles.append(role)
        if role == 'source_V': assert (output / 'smoke.json').is_file()
        if role == 'novel_validation': assert (output / 'calibration/complete.json').is_file()
        return loader(root, role, include_y)
    monkeypatch.setattr(m.parent, '_load_features', evaluation)
    monkeypatch.setattr(m.parent.core, 'fit_readouts', lambda *a, **k: pytest.fail('No V/novel fitting'))
    pred = m.predict(export, tmp_path / 'fit', output)
    assert roles == ['source_L', 'source_V', 'novel_validation']
    assert pred['status'] == 'SOURCE_BUDGET_READOUT_PREDICTIONS_COMPLETE'
    assert pred['encoder_updates'] == pred['identity_head_updates'] == 2220
    assert pred['new_encoder_updates'] == pred['new_identity_head_updates'] == pred['new_readout_updates'] == 0
    cal = json.loads((output / 'calibration/complete.json').read_text())
    assert cal['schema'] == m.calibration.SCHEMA
    assert cal['limits'] == {'known_macro_frr': .05, 'original_closed_minus_new_open': .02}
    assert not cal['unknown_inputs_used']
    for name, contents in frozen.items(): assert contents == (tmp_path / 'fit' / name).read_bytes()
    with np.load(output / 'predictions.npz') as stored:
        assert stored['knownness'].shape == (312, 7)
        assert set(stored.files) == {'ids', 'method_names', 'closed_prediction', 'baseline_closed_prediction',
            'knownness', 'accepted', 'prediction', 'rx_i', 'day_i'}
        np.testing.assert_array_equal(stored['prediction'], np.where(stored['accepted'], stored['closed_prediction'][:, None], -1))
    with pytest.raises(FileExistsError): m.predict(export, tmp_path / 'fit', output)


def test_closed_damage_is_scientific_failure_not_predict_abort(tmp_path, monkeypatch):
    m, base, export, marker = setup(tmp_path, monkeypatch)
    m.fit(export, 2026092701, tmp_path / 'fit')
    path = export / m.parent.FEATURE_REFS['source_V']
    with np.load(path) as stored: arrays = {k: stored[k] for k in stored.files}
    # The original parent V remains correct; only the new encoder's closed head
    # is damaged, which must produce completed predictions, not a launch error.
    arrays['logits'] = np.full((300, 6), -5., np.float32)
    arrays['logits'][:, 0] = 5.
    np.savez(path, **arrays)
    m.predict(export, tmp_path / 'fit', tmp_path / 'predict')
    cal = json.loads((tmp_path / 'predict/calibration/complete.json').read_text())
    assert cal['encoder_protection_passed'] is False
    assert (tmp_path / 'predict/complete.json').is_file()


def test_wrong_fit_encoder_is_rejected_before_any_evaluation(tmp_path, monkeypatch):
    m, base, export, marker = setup(tmp_path, monkeypatch)
    m.fit(export, 2026092701, tmp_path / 'fit')
    f = tmp_path / 'fit/complete.json'; value = json.loads(f.read_text())
    value['encoder_state_identity'] = {'arm': 'uniform_oe'}; f.write_text(json.dumps(value))
    monkeypatch.setattr(m.parent, '_load_features', lambda *a, **k: pytest.fail('No V with wrong fit identity'))
    with pytest.raises(ValueError): m.predict(export, tmp_path / 'fit', tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_real_budget_export_schema_bridge_and_original_oe_roles(tmp_path):
    m = module(); run = '20261029-phase3-source-embedding-budget-s2026092701-r01'
    handles = {'exposure_train': {'capsule_id': 'source4', 'split_id': 'train', 'count': 4200},
               'novel_validation': {'capsule_id': 'novel3', 'split_id': 'validation', 'count': 3150}}
    identity = dict(run_id=run, row_id='supervised_only-s2026092701', arm='supervised_only',
        code_commit='a' * 40, checkpoint='/runs/' + run + '/rows/supervised_only/train/final.pt',
        updates=2220, cumulative_updates=13320)
    marker = dict(status='SOURCE_BUDGET_BRANCH_FEATURES_EXPORTED', arm='supervised_only',
        source_only=True, target_reads=0, original_U_s_reads=0, OE_truth_reads=0, optimizer_updates=0,
        new_encoder_updates=0, new_identity_head_updates=0, historical_encoder_updates=2220,
        historical_identity_head_updates=2220, counts=m.parent.COUNTS, feature_refs=m.parent.FEATURE_REFS,
        model_classes=6, parameters=232993, head_changed=False, default_head_replaced=False,
        encoder_state_identity=identity, source_OE_handles=handles,
        parent_export_identity={'source_OE_handles': handles, 'counts': m.parent.COUNTS})
    path = tmp_path / 'complete.json'; path.write_text(json.dumps(marker))
    assert m._export_marker(tmp_path)['encoder_state_identity'] == identity
    marker['source_OE_handles']['exposure_train']['count'] = 6134
    # The private source identities have distinct value objects after JSON IO.
    saved = json.loads(path.read_text()); saved['source_OE_handles']['exposure_train']['count'] = 6134
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='capsule'): m._export_marker(tmp_path)
