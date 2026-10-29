import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import test_pamf_phase3_source_embedding_readout_scoring as old


RUN = '20261029-phase3-source-embedding-budget-s2026092701-r01'


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_budget_scoring.py'
    assert path.is_file(), 'Budget scorer is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_budget_scoring')


def fixture(tmp_path, arm='supervised_only'):
    case = old.fixture(tmp_path, arm=arm)
    file = case['root'] / 'complete.json'; marker = json.loads(file.read_text())
    marker.update(status='SOURCE_BUDGET_READOUT_PREDICTIONS_COMPLETE', encoder_updates=2220,
        identity_head_updates=2220, new_readout_updates=0,
        calibration_schema='phase3_source_original_baseline_risk_v1')
    marker['encoder_state_identity'] = dict(run_id=RUN, row_id=arm + '-s2026092701', arm=arm,
        code_commit='a' * 40, checkpoint='/runs/' + RUN + '/rows/' + arm + '/train/final.pt',
        updates=2220, cumulative_updates=13320)
    file.write_text(json.dumps(marker)); return case


def run(case):
    return module().score(case['predictions'], case['known_features'], case['novel_truth'], case['output'])


@pytest.mark.parametrize('arm', ['supervised_only', 'uniform_oe'])
def test_budget_identity_and_only_mlp_promotion_on_each_arm(tmp_path, arm):
    case = fixture(tmp_path, arm); result = run(case)
    assert result['status'] == 'SCORED_SOURCE_BUDGET_READOUT_COMPLETE'
    assert result['encoder_updates'] == 2220 and result['new_encoder_updates'] == 0
    assert result['eligible_methods'] == ['mlp16_oe']
    assert result['next_step'] == 'SOURCE_EMBEDDING_BUDGET_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'
    assert not result['methods']['linear_oe']['eligible']
    assert not result['methods']['msp']['eligible']
    assert result['original_parent_baseline_verified']
    assert result['final_phase3_complete'] is False


@pytest.mark.parametrize('fault', ['old_steps', 'wrong_row', 'new_readout_updates', 'wrong_calibration'])
def test_wrong_budget_contract_fails_before_any_truth(tmp_path, monkeypatch, fault):
    case = fixture(tmp_path); path = case['root'] / 'complete.json'; marker = json.loads(path.read_text())
    if fault == 'old_steps': marker['encoder_updates'] = 222
    elif fault == 'wrong_row': marker['encoder_state_identity']['row_id'] = 'uniform_oe-s2026092701'
    elif fault == 'new_readout_updates': marker['new_readout_updates'] = 200
    else: marker['calibration_schema'] = 'phase3_source_known_risk_calibration_v1'
    path.write_text(json.dumps(marker)); original = np.load; opened = []
    def guard(file, *args, **kwargs):
        if Path(file) in (case['known_features'], case['novel_truth']):
            opened.append(str(file)); pytest.fail('Truth opened before budget checks')
        return original(file, *args, **kwargs)
    monkeypatch.setattr(module().np, 'load', guard)
    with pytest.raises(ValueError): run(case)
    assert opened == [] and not case['output'].exists()


def test_diagnostic_success_and_per_rx_closed_damage_cannot_promote(tmp_path):
    case = fixture(tmp_path)
    old.mutate(case, lambda d: d['accepted'].__setitem__((np.isin(d['ids'], case['novel_ids']), 6), True))
    result = run(case)
    assert result['methods']['linear_oe']['diagnostic_passed'] and result['selected_method'] is None
    root = tmp_path / 'second'; root.mkdir(); case = fixture(root)
    damaged = case['known_ids'][np.flatnonzero(case['known_rx'] == 8)[0]]
    old.mutate(case, lambda d: d['closed_prediction'].__setitem__(d['ids'] == damaged, 1))
    result = run(case)
    assert result['methods']['mlp16_oe']['goals']['overall']['passed']
    assert not result['methods']['mlp16_oe']['goals']['by_receiver']['8']['passed']
    assert result['next_step'] == 'NO_PROMOTION'
