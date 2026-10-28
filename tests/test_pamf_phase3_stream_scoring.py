"""Truth-last source-stream scoring, using small synthetic records only."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


STATES = ['F', 'D0R0', 'D1R0', 'D0R1', 'D1R1']


def scorer():
    return importlib.import_module('experiments.pamf_phase3_open_world.stream_scoring')


def write_npz(path, **values):
    np.savez(path, **values)


def change(path, **updates):
    with np.load(path, allow_pickle=False) as stored:
        values = {key: stored[key].copy() for key in stored.files}
    values.update(updates)
    write_npz(path, **values)


@pytest.fixture
def case(tmp_path):
    root = tmp_path / 'predictions'
    root.mkdir()
    ids = np.asarray(['v' + str(i) for i in range(12)])
    y = np.repeat(np.arange(6), 2)
    baseline = np.minimum(y, 3)
    baseline[0] = 1
    closed = np.repeat(baseline[:, None], 5, axis=1)
    closed[:8, 2] = y[:8]
    closed[:8, 4] = y[:8]
    prediction = closed.copy()
    prediction[8:, :] = -1
    prediction[8:10, 3:] = 4
    future = root / 'future_predictions.npz'
    write_npz(future, ids=ids, state_names=np.asarray(STATES), baseline_closed=baseline,
              closed_prediction=closed, knownness=np.ones((12, 5)),
              accepted=prediction != -1, prediction=prediction)
    u_ids = np.asarray(['u' + str(i) for i in range(12)])
    u_y = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
    assigned = np.full((12, 2), -1, dtype=np.int64)
    assigned[8:, 0] = [4, 4, 5, 5]
    assigned[8:, 1] = [4, 4, 4, 5]
    discovery = root / 'discovery_predictions.npz'
    write_npz(discovery, ids=u_ids, state_names=np.asarray(['D0', 'D1']),
              anonymous_prediction=assigned)
    registry = root / 'registry.npz'
    values = {}
    for name in ['D0', 'D1']:
        values[name + '_anonymous_ids'] = np.asarray([4, 5], dtype=np.int64)
        values[name + '_converged'] = np.asarray(True)
        values[name + '_cap_reached'] = np.asarray(False)
        values[name + '_promotion_eligible'] = np.asarray(True)
    write_npz(registry, **values)
    marker = dict(status='ALL_STREAM_PREDICTIONS_COMPLETE', truth_read=False,
                  query_fit=False, source_only=True, target_reads=0,
                  future_count=12, discovery_count=12)
    (root / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    u_truth, future_truth = tmp_path / 'u_truth.npz', tmp_path / 'future_truth.npz'
    # Deliberately reverse truth order: matching must use opaque IDs.
    write_npz(u_truth, ids=u_ids[::-1], y=u_y[::-1])
    write_npz(future_truth, ids=ids[::-1], y=y[::-1])
    return dict(future=future, discovery=discovery, registry=registry,
                u_truth=u_truth, future_truth=future_truth, output=tmp_path / 'score',
                root=root, y=y, u_y=u_y)


def run(case):
    return scorer().score(case['future'], case['discovery'], case['registry'],
                          case['u_truth'], case['future_truth'], case['output'])


def guard_truth(monkeypatch, case):
    module = scorer()
    original = np.load
    opened = []

    def guarded(path, *args, **kwargs):
        if Path(path) in (case['u_truth'], case['future_truth']):
            opened.append(str(path))
            raise AssertionError('Truth opened before prediction validation')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.np, 'load', guarded)
    return opened


def test_fixed_u_mapping_and_independent_future_unknown(case):
    change(case['future'], prediction=np.asarray([
        [1, 1, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3],
        [-1, -1, -1, 5, 4], [-1, -1, -1, 5, 4],
        [-1, -1, -1, 5, -1], [-1, -1, -1, 5, -1]]),
        accepted=np.asarray([[True] * 5] * 8 + [[False, False, False, True, True]] * 2
                            + [[False, False, False, True, False]] * 2))
    result = run(case)
    d0 = result['registries']['D0']
    assert d0['anonymous_to_source_label'] == {'4': 4}
    assert d0['duplicate_registration_count'] == 1
    assert d0['matched_purity'] == 1.
    assert d0['matched_discovery_coverage'] == .5
    future = result['states']['D0R1']['future']
    assert future['new4']['accuracy'] == 0.  # No new Hungarian on V to reward slot 5.
    assert future['later5']['rejection'] == 0.  # Unmatched anonymous is still accepted.
    assert future['old_macro_accuracy'] == .875


def test_source_only_complete_and_paired_old_metrics(case):
    result = run(case)
    assert result['status'] == 'SCORED_SOURCE_STREAM_COMPLETE'
    assert result['evidence_scope'] == 'source_proxy_only_not_target'
    assert result['next_step'] == 'SOURCE_JOINT_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'
    assert result['states']['D1R1']['eligible'] is True
    assert result['eligible_states'] == ['D1R1']
    assert result['states']['F']['eligible'] is False
    assert result['states']['D1R0']['future']['new4']['accuracy'] is None
    assert result['states']['D1R0']['future']['new4']['rejection'] == 1.
    assert result['paired_old_metrics']['domain_correction_gain']['R1'] == .125
    assert result['paired_old_metrics']['registration_damage']['D1'] == 0.
    assert result['states']['D1R1']['future']['old_macro_accuracy'] == 1.
    assert result['states']['D1R1']['future']['old_macro_frr'] == 0.
    assert json.loads((case['output'] / 'results.json').read_text()) == result
    assert json.loads((case['output'] / 'complete.json').read_text())['status'] == result['status']


@pytest.mark.parametrize('fault', ['missing_file', 'missing_marker', 'marker_flag', 'marker_count',
    'state_order', 'score_nan', 'accepted_dtype', 'closed_shape', 'wrong_old', 'bad_anon',
    'r0_anon', 'duplicate_id', 'u_future_overlap', 'registry_gap', 'registry_bool',
    'registry_flag', 'u_bad_anon', 'mismatched_r0_r1'])
def test_invalid_predictions_never_open_truth(case, monkeypatch, fault):
    if fault == 'missing_file':
        case['discovery'].unlink()
    elif fault == 'missing_marker':
        (case['root'] / 'complete.json').unlink()
    elif fault in ('marker_flag', 'marker_count'):
        marker = json.loads((case['root'] / 'complete.json').read_text())
        marker['target_reads' if fault == 'marker_flag' else 'future_count'] = False if fault == 'marker_flag' else 13
        (case['root'] / 'complete.json').write_text(json.dumps(marker))
    elif fault == 'state_order':
        change(case['future'], state_names=np.asarray(STATES[::-1]))
    elif fault == 'score_nan':
        values = np.ones((12, 5)); values[0, 0] = np.nan
        change(case['future'], knownness=values)
    elif fault == 'accepted_dtype':
        change(case['future'], accepted=np.ones((12, 5), dtype=np.int64))
    elif fault == 'closed_shape':
        change(case['future'], closed_prediction=np.zeros(12, dtype=np.int64))
    elif fault in ('wrong_old', 'bad_anon', 'r0_anon', 'mismatched_r0_r1'):
        with np.load(case['future']) as data:
            pred = data['prediction'].copy()
        if fault == 'wrong_old': pred[0, 0] = 0
        elif fault == 'bad_anon': pred[8, 3] = 8
        elif fault == 'r0_anon': pred[8, 1] = 4
        else: pred[0, 3] = -1
        change(case['future'], prediction=pred, accepted=pred != -1)
    elif fault in ('duplicate_id', 'u_future_overlap'):
        ids = np.asarray(['v' + str(i) for i in range(12)])
        ids[0] = ids[1] if fault == 'duplicate_id' else 'u0'
        change(case['future'], ids=ids)
    elif fault == 'registry_gap':
        change(case['registry'], D0_anonymous_ids=np.asarray([4, 6]))
    elif fault == 'registry_bool':
        change(case['registry'], D0_converged=np.asarray(1))
    elif fault == 'registry_flag':
        change(case['registry'], D0_promotion_eligible=np.asarray(False))
    else:
        bad = np.full((12, 2), 8, dtype=np.int64)
        change(case['discovery'], anonymous_prediction=bad)
    opened = guard_truth(monkeypatch, case)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(case)
    assert opened == []
    assert not case['output'].exists()


def test_nonconvergence_cap_and_missing_classes_cannot_promote(case):
    change(case['registry'], D0_converged=np.asarray(False), D0_promotion_eligible=np.asarray(False),
           D1_cap_reached=np.asarray(True), D1_promotion_eligible=np.asarray(False))
    result = run(case)
    assert result['next_step'] == 'NO_PROMOTION'
    assert not result['eligible_states']


def test_missing_later_class_is_not_a_pass(case):
    y = case['y'].copy(); y[y == 5] = 4
    write_npz(case['future_truth'], ids=np.asarray(['v' + str(i) for i in range(12)]), y=y)
    result = run(case)
    assert not result['eligible_states']
    assert result['states']['D1R1']['future']['later5']['rejection'] is None


def test_truth_ids_and_label_contract_are_strict(case):
    write_npz(case['u_truth'], ids=np.asarray(['wrong' + str(i) for i in range(12)]), y=case['u_y'])
    with pytest.raises(ValueError, match='coverage'):
        run(case)
    assert not case['output'].exists()


def test_old_misregistration_is_error_but_not_frr(case):
    with np.load(case['future']) as data:
        pred = data['prediction'].copy()
    pred[0, 1] = -1
    pred[0, 3] = 4
    change(case['future'], prediction=pred, accepted=pred != -1)
    result = run(case)
    row = result['states']['D0R1']['future']
    assert row['old_macro_frr'] == 0.
    assert row['old_misregistered_count'] == 1
    assert row['old_macro_accuracy'] == .875
    assert result['paired_old_metrics']['registration_damage']['D0'] == 0.


def test_no_novel_evidence_leaves_all_slots_unmatched(case):
    y = case['u_y'].copy(); y[y == 4] = 0
    write_npz(case['u_truth'], ids=np.asarray(['u' + str(i) for i in range(12)]), y=y)
    result = run(case)
    assert result['registries']['D1']['anonymous_to_source_label'] == {}
    assert result['states']['D1R1']['future']['new4']['accuracy'] == 0.
    assert result['next_step'] == 'NO_PROMOTION'


def test_empty_registry_is_legal_evidence_but_cannot_promote(case):
    changes = {}
    for bank in ('D0', 'D1'):
        changes[bank + '_anonymous_ids'] = np.asarray([], dtype=np.int64)
        changes[bank + '_promotion_eligible'] = np.asarray(False)
    change(case['registry'], **changes)
    change(case['discovery'], anonymous_prediction=np.full((12, 2), -1, dtype=np.int64))
    with np.load(case['future']) as stored:
        prediction = stored['prediction'].copy()
    prediction[:, 3], prediction[:, 4] = prediction[:, 1], prediction[:, 2]
    change(case['future'], prediction=prediction, accepted=prediction != -1)
    result = run(case)
    assert result['registries']['D0']['components'] == []
    assert result['registries']['D0']['anonymous_to_source_label'] == {}
    assert result['next_step'] == 'NO_PROMOTION'


def test_all_rejected_cannot_promote(case):
    change(case['future'], prediction=np.full((12, 5), -1, dtype=np.int64),
           accepted=np.zeros((12, 5), dtype=bool))
    result = run(case)
    assert result['states']['D1R1']['future']['old_macro_frr'] == 1.
    assert result['states']['D1R1']['future']['later5']['rejection'] == 1.
    assert result['next_step'] == 'NO_PROMOTION'


def test_paired_old_macro_does_not_include_new_classes_or_use_pooled_weights(case):
    with np.load(case['future']) as stored:
        values = {key: stored[key].copy() for key in stored.files}
    for key in ('baseline_closed', 'closed_prediction', 'knownness', 'accepted', 'prediction'):
        values[key] = np.concatenate([values[key], values[key][[0, 0]]], axis=0)
    values['ids'] = np.r_[values['ids'], ['v12', 'v13']]
    write_npz(case['future'], **values)
    marker_path = case['root'] / 'complete.json'
    marker = json.loads(marker_path.read_text()); marker['future_count'] = 14
    marker_path.write_text(json.dumps(marker))
    write_npz(case['future_truth'], ids=values['ids'], y=np.r_[case['y'], [0, 0]])
    result = run(case)
    assert result['states']['D0R1']['future']['old_macro_accuracy'] == .8125
    assert result['states']['D1R1']['future']['old_macro_accuracy'] == 1.
    assert result['paired_old_metrics']['domain_correction_gain']['R1'] == .1875


def test_r0_controls_pass_but_failed_registration_cannot_promote_joint(case):
    with np.load(case['future']) as stored:
        prediction = stored['prediction'].copy()
    prediction[8:10, 3:] = 5  # Both R1 states accept an unmatched anonymous slot.
    change(case['future'], prediction=prediction, accepted=prediction != -1)
    result = run(case)
    assert result['states']['D0R0']['control_passed'] is True
    assert result['states']['D1R0']['control_passed'] is True
    assert result['states']['D1R1']['future']['new4']['accuracy'] == 0.
    assert result['eligible_states'] == []
    assert result['next_step'] == 'NO_PROMOTION'


def test_d0_registration_pass_without_domain_gain_cannot_promote_joint(case):
    with np.load(case['future']) as stored:
        prediction, closed = stored['prediction'].copy(), stored['closed_prediction'].copy()
        baseline = stored['baseline_closed'].copy()
    for column in (2, 4):
        closed[:, column] = baseline
        prediction[:8, column] = baseline[:8]
    change(case['future'], prediction=prediction, closed_prediction=closed, accepted=prediction != -1)
    result = run(case)
    assert result['states']['D0R1']['control_passed'] is True
    assert result['paired_old_metrics']['domain_correction_gain']['R1'] == 0.
    assert result['states']['D1R1']['scientific_checks']['same_registration_domain_correction_old_gain_positive'] is False
    assert result['eligible_states'] == []
    assert result['next_step'] == 'NO_PROMOTION'
