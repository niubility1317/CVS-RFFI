"""Independent source-OE score contracts, with tiny synthetic observations."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


METHODS = ['msp', 'maxlogit', 'margin', 'energy', 'cosine', 'linear_oe', 'mlp16_oe']
RX = [1, 3, 4, 6, 8]
BRANCHES = ['fused160', 'time160', 'frequency160', 'pa160', 'reference29',
            'signature336', 'spectrum128', 'pa_moments48']


def scorer():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_open_scoring')


def change(path, **updates):
    with np.load(path, allow_pickle=False) as stored:
        values = {name: stored[name].copy() for name in stored.files}
    values.update(updates)
    np.savez(path, **values)


def fixture(tmp_path, known_repeats=None, novel_repeats=None):
    known_repeats = known_repeats or [2] * 6
    novel_repeats = novel_repeats or [1] * 3
    rows = [(r, y) for r in RX for y in range(6) for _ in range(known_repeats[y])]
    new_rows = [(r, tx) for r in RX for tx in range(3) for _ in range(novel_repeats[tx])]
    n, nu = len(rows), len(new_rows)
    permutation = np.random.default_rng(18).permutation(n + nu)
    ids = np.asarray([f'opaque-{value:05d}' for value in permutation])
    known_ids, novel_ids = ids[:n], ids[n:]
    y = np.asarray([row[1] for row in rows], np.int64)
    rx = np.asarray([row[0] for row in rows + new_rows], np.int64)
    day = np.ones(n + nu, np.int64)
    closed = np.r_[y, np.zeros(nu, np.int64)]
    accepted = np.repeat((np.arange(n + nu) < n)[:, None], 7, axis=1)
    score = np.where(accepted, 1., -1.).astype(np.float32)
    prediction = np.where(accepted, closed[:, None], -1)
    order = np.argsort(ids)
    root = tmp_path / 'predictions'; root.mkdir()
    path = root / 'predictions.npz'
    np.savez(path, ids=ids[order], method_names=np.asarray(METHODS),
             closed_prediction=closed[order], knownness=score[order], accepted=accepted[order],
             prediction=prediction[order], rx_i=rx[order], day_i=day[order])
    marker = dict(status='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', source_only=True, target_reads=0,
        truth_read=False, query_fit=False, known_count=n, novel_count=nu, method_names=METHODS)
    (root / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    known_features = tmp_path / 'known_features.npz'
    # Deliberately unreadable with allow_pickle=False. The scorer only needs
    # ids/y/rx/day, not logits or any feature branch payload.
    arrays = {name: np.asarray([{'must_not_materialize': True}], dtype=object)
              for name in ['logits'] + ['features__' + value for value in BRANCHES]}
    np.savez(known_features, ids=known_ids[::-1], y=y[::-1], rx_i=rx[:n][::-1], day_i=day[:n][::-1], **arrays)
    novel_truth = tmp_path / 'novel_truth.npz'
    tx = np.asarray(['device-' + str(row[1]) for row in new_rows])
    np.savez(novel_truth, ids=novel_ids[::-1], tx=tx[::-1])
    return dict(predictions=path, known_features=known_features, novel_truth=novel_truth,
                output=tmp_path / 'score', root=root, known_ids=known_ids, novel_ids=novel_ids,
                known_y=y, rx=rx, novel_tx=tx)


@pytest.fixture
def case(tmp_path):
    return fixture(tmp_path)


def run(case):
    return scorer().score(case['predictions'], case['known_features'], case['novel_truth'], case['output'])


def set_acceptance(case, mutate):
    with np.load(case['predictions']) as stored:
        ids, accepted, closed = stored['ids'], stored['accepted'].copy(), stored['closed_prediction']
    known = np.isin(ids, case['known_ids'])
    mutate(ids, known, accepted)
    change(case['predictions'], accepted=accepted, prediction=np.where(accepted, closed[:, None], -1))


def test_two_oe_pass_choose_smaller_linear_and_only_read_truth_metadata(case):
    result = run(case)
    assert result['eligible_methods'] == ['linear_oe', 'mlp16_oe']
    assert result['selected_method'] == 'linear_oe'
    assert result['methods']['msp']['diagnostic_passed'] is True
    assert result['methods']['msp']['eligible'] is False
    assert result['methods']['linear_oe']['overall']['known_macro_open_accuracy'] == 1.
    assert result['methods']['linear_oe']['overall']['unknown_macro_rejection'] == 1.
    assert result['methods']['linear_oe']['overall']['auroc_known'] == 1.
    assert result['methods']['linear_oe']['overall']['oscr'] == 1.
    assert result['evidence_scope'] == 'source_only_not_target'
    assert result['final_phase3_complete'] is False
    assert result['next_step'] == 'SOURCE_OPEN_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'
    complete = json.loads((case['output'] / 'complete.json').read_text())
    assert complete['status'] == 'SCORED_SOURCE_OPEN_COMPLETE'
    assert complete['selected_method'] == 'linear_oe'
    assert json.loads((case['output'] / 'results.json').read_text()) == result


def test_control_only_success_cannot_promote_oe(case):
    set_acceptance(case, lambda ids, known, values: values.__setitem__((~known, slice(5, 7)), True))
    result = run(case)
    assert result['methods']['msp']['diagnostic_passed'] is True
    assert result['methods']['linear_oe']['diagnostic_passed'] is False
    assert result['eligible_methods'] == [] and result['selected_method'] is None
    assert result['next_step'] == 'NO_PROMOTION'


def test_only_mlp_success_selects_mlp(case):
    set_acceptance(case, lambda ids, known, values: values.__setitem__((~known, 5), True))
    result = run(case)
    assert result['eligible_methods'] == ['mlp16_oe']
    assert result['selected_method'] == 'mlp16_oe'


@pytest.mark.parametrize('fault', ['marker', 'count', 'target_bool', 'methods', 'ids_duplicate', 'nan',
    'accepted_type', 'bad_prediction', 'closed_class', 'rx', 'day', 'missing_file'])
def test_bad_prediction_fails_before_any_truth_io(case, monkeypatch, fault):
    module = scorer()
    if fault in ('marker', 'count', 'target_bool'):
        marker = json.loads((case['root'] / 'complete.json').read_text())
        if fault == 'marker': marker['truth_read'] = True
        elif fault == 'count': marker['known_count'] += 1
        else: marker['target_reads'] = False
        (case['root'] / 'complete.json').write_text(json.dumps(marker))
    elif fault == 'methods': change(case['predictions'], method_names=np.asarray(METHODS[::-1]))
    elif fault == 'ids_duplicate':
        with np.load(case['predictions']) as stored: ids = stored['ids'].copy()
        ids[0] = ids[1]; change(case['predictions'], ids=ids)
    elif fault == 'nan':
        with np.load(case['predictions']) as stored: score = stored['knownness'].copy()
        score[0, 0] = np.nan; change(case['predictions'], knownness=score)
    elif fault == 'accepted_type':
        with np.load(case['predictions']) as stored: accepted = stored['accepted'].astype(np.int64)
        change(case['predictions'], accepted=accepted)
    elif fault == 'bad_prediction':
        with np.load(case['predictions']) as stored: values = stored['prediction'].copy()
        values[0, 0] = 6; change(case['predictions'], prediction=values)
    elif fault == 'closed_class':
        with np.load(case['predictions']) as stored: values = stored['closed_prediction'].copy()
        values[0] = 6; change(case['predictions'], closed_prediction=values)
    elif fault in ('rx', 'day'):
        with np.load(case['predictions']) as stored: values = stored['rx_i' if fault == 'rx' else 'day_i'].copy()
        values[0] = 99; change(case['predictions'], **{'rx_i' if fault == 'rx' else 'day_i': values})
    else: case['predictions'].unlink()
    original = np.load
    opened = []

    def guard(path, *args, **kwargs):
        if Path(path) in (case['known_features'], case['novel_truth']):
            opened.append(str(path)); raise AssertionError('Truth opened before complete prediction checks')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(module.np, 'load', guard)
    with pytest.raises((ValueError, FileNotFoundError)): run(case)
    assert opened == [] and not case['output'].exists()


def test_unknown_macro_uses_tx_not_record_weights(tmp_path):
    case = fixture(tmp_path, novel_repeats=[100, 1, 1])
    bad_ids = case['novel_ids'][case['novel_tx'] == 'device-1']
    set_acceptance(case, lambda ids, known, values: values.__setitem__((np.isin(ids, bad_ids), slice(None)), True))
    result = run(case)
    metrics = result['methods']['linear_oe']['overall']
    assert metrics['unknown_rejection'] > .99
    assert metrics['unknown_macro_rejection'] == pytest.approx(2 / 3)
    assert result['selected_method'] is None


def test_old_macro_drop_protects_rare_known_class(tmp_path):
    case = fixture(tmp_path, known_repeats=[1, 100, 100, 100, 100, 100])
    rare_ids = case['known_ids'][case['known_y'] == 0]
    set_acceptance(case, lambda ids, known, values: values.__setitem__((np.isin(ids, rare_ids), slice(None)), False))
    result = run(case)
    metrics = result['methods']['linear_oe']['overall']
    assert metrics['known_accuracy_drop'] < .02  # Legacy pooled diagnostic alone would miss it.
    assert metrics['known_macro_accuracy_drop'] == pytest.approx(1 / 6)
    assert result['selected_method'] is None


def test_known_label_and_unknown_tx_permutations_preserve_overall_metrics(case, tmp_path):
    first = run(case)
    # Keep identical physical observations and scores; only rename class coordinates.
    with np.load(case['known_features'], allow_pickle=True) as stored:
        known = {key: stored[key] for key in stored.files}
    permutation = np.asarray([4, 2, 5, 1, 0, 3])
    known['y'] = permutation[known['y']]
    np.savez(case['known_features'], **known)
    with np.load(case['predictions']) as stored:
        closed, accepted = permutation[stored['closed_prediction']], stored['accepted']
    change(case['predictions'], closed_prediction=closed, prediction=np.where(accepted, closed[:, None], -1))
    with np.load(case['novel_truth']) as stored:
        ids, tx = stored['ids'], stored['tx']
    renamed = np.asarray([{'device-0': 'z', 'device-1': 'x', 'device-2': 'y'}[value] for value in tx])
    np.savez(case['novel_truth'], ids=ids, tx=renamed)
    case['output'] = tmp_path / 'permuted_score'
    result = run(case)
    a, b = first['methods']['linear_oe']['overall'], result['methods']['linear_oe']['overall']
    for key in ('known_macro_accuracy_drop', 'known_macro_open_accuracy', 'known_macro_frr', 'unknown_macro_rejection', 'auroc_known', 'oscr'):
        assert a[key] == b[key]


def test_all_reject_does_not_pass(case):
    set_acceptance(case, lambda ids, known, values: values.__setitem__((slice(None), slice(None)), False))
    result = run(case)
    assert result['selected_method'] is None
    assert result['methods']['linear_oe']['overall']['unknown_macro_rejection'] == 1.


def test_truth_id_coverage_must_match(case):
    with np.load(case['novel_truth']) as stored:
        ids, tx = stored['ids'].copy(), stored['tx']
    ids[0] = case['known_ids'][0]
    np.savez(case['novel_truth'], ids=ids, tx=tx)
    with pytest.raises(ValueError, match='overlap|coverage'): run(case)
    assert not case['output'].exists()


def test_each_receiver_needs_all_three_novel_devices(case):
    with np.load(case['novel_truth']) as stored:
        ids, tx = stored['ids'], stored['tx'].copy()
    # Eliminate one TX only in RX8, preserving all three globally.
    candidate = case['novel_ids'][-1]
    tx[ids == candidate] = 'device-0'
    np.savez(case['novel_truth'], ids=ids, tx=tx)
    result = run(case)
    assert result['methods']['linear_oe']['overall']['unknown_classes_present'] == 3
    assert result['methods']['linear_oe']['by_receiver']['8']['unknown_classes_present'] == 2
    assert result['selected_method'] is None


def test_known_truth_metadata_must_match_prediction(case):
    with np.load(case['known_features'], allow_pickle=True) as stored:
        values = {key: stored[key].copy() for key in stored.files}
    values['rx_i'][0] = 1 if values['rx_i'][0] != 1 else 3
    np.savez(case['known_features'], **values)
    with pytest.raises(ValueError, match='receiver/day'): run(case)
    assert not case['output'].exists()


def test_missing_known_class_is_explicitly_not_a_pass(case):
    with np.load(case['known_features'], allow_pickle=True) as stored:
        values = {key: stored[key].copy() for key in stored.files}
    values['y'][values['y'] == 5] = 4
    np.savez(case['known_features'], **values)
    result = run(case)
    assert result['methods']['linear_oe']['overall']['known_classes_present'] == 5
    assert result['methods']['linear_oe']['goals']['overall']['checks']['all_six_known_classes_present'] is False
    assert result['selected_method'] is None


def test_empty_novel_tx_is_rejected(case):
    with np.load(case['novel_truth']) as stored:
        ids, tx = stored['ids'], stored['tx'].copy()
    tx[0] = ''
    np.savez(case['novel_truth'], ids=ids, tx=tx)
    with pytest.raises(ValueError, match='novel TX'): run(case)
