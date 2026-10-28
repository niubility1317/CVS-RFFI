"""Source-only embedding increment scores against the unchanged parent head."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


METHODS = ['msp', 'maxlogit', 'margin', 'energy', 'cosine']
RX = [1, 3, 4, 6, 8]
BRANCHES = ['fused160', 'time160', 'frequency160', 'pa160', 'reference29',
            'signature336', 'spectrum128', 'pa_moments48']


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_scoring')


def change(path, **updates):
    with np.load(path, allow_pickle=True) as stored:
        values = {key: stored[key].copy() for key in stored.files}
    values.update(updates)
    np.savez(path, **values)


def fixture(tmp_path, known_repeats=None, novel_repeats=None, arm='uniform_oe'):
    known_repeats = known_repeats or [20] * 6
    novel_repeats = novel_repeats or [2] * 3
    known = [(rx, y) for rx in RX for y in range(6) for _ in range(known_repeats[y])]
    novel = [(rx, y) for rx in RX for y in range(3) for _ in range(novel_repeats[y])]
    nk, nu = len(known), len(novel)
    # Names deliberately reveal no role. Physical-ID alignment cannot use prefixes.
    opaque = np.random.default_rng(13).permutation(nk + nu)
    ids = np.asarray(['opaque-%06d' % n for n in opaque])
    y = np.asarray([v[1] for v in known], np.int64)
    rx = np.asarray([v[0] for v in known + novel], np.int64)
    day = np.ones(nk + nu, np.int64)
    baseline = np.r_[y, np.zeros(nu, np.int64)]
    accepted = np.repeat((np.arange(nk + nu) < nk)[:, None], 5, axis=1)
    root = tmp_path / 'prediction'; root.mkdir()
    predictions = root / 'predictions.npz'
    order = np.argsort(ids)
    np.savez(predictions, ids=ids[order], method_names=np.asarray(METHODS),
        closed_prediction=baseline[order], baseline_closed_prediction=baseline[order],
        accepted=accepted[order], knownness=np.where(accepted, 1., -1.).astype(np.float32)[order],
        prediction=np.where(accepted, baseline[:, None], -1)[order], rx_i=rx[order], day_i=day[order])
    marker = dict(status='SOURCE_EMBEDDING_PREDICTIONS_COMPLETE', source_only=True,
        target_reads=0, known_count=nk, novel_count=nu, count=nk + nu,
        method_names=METHODS, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_truth_read=False, query_fit=False, encoder_updates=222, identity_head_updates=222,
        default_head_replaced=False, arm=arm, parent_export_identity={'model': 'frozen original F6'})
    (root / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    features = tmp_path / 'known_features.npz'
    logits = np.full((nk, 6), -2., np.float32); logits[np.arange(nk), y] = 2.
    # These arrays cannot be loaded with allow_pickle=False and are never needed.
    unused = {'features__' + name: np.asarray([{'forbidden': True}], dtype=object) for name in BRANCHES}
    np.savez(features, ids=ids[:nk][::-1], y=y[::-1], rx_i=rx[:nk][::-1],
             day_i=day[:nk][::-1], logits=logits[::-1], **unused)
    truth = tmp_path / 'novel_truth.npz'
    tx = np.asarray(['device-' + str(v[1]) for v in novel])
    np.savez(truth, ids=ids[nk:][::-1], tx=tx[::-1])
    return dict(predictions=predictions, known_features=features, novel_truth=truth,
        output=tmp_path / 'score', root=root, known_ids=ids[:nk], novel_ids=ids[nk:],
        known_y=y, known_rx=rx[:nk], novel_tx=tx)


@pytest.fixture
def case(tmp_path):
    return fixture(tmp_path)


def run(case):
    return module().score(case['predictions'], case['known_features'], case['novel_truth'], case['output'])


def mutate_predictions(case, mutate):
    with np.load(case['predictions']) as stored:
        data = {key: stored[key].copy() for key in stored.files}
    mutate(data)
    data['prediction'] = np.where(data['accepted'], data['closed_prediction'][:, None], -1)
    np.savez(case['predictions'], **data)


def test_uniform_msp_only_can_promote_and_parent_features_not_materialized(case):
    result = run(case)
    assert result['status'] == 'SCORED_SOURCE_EMBEDDING_COMPLETE'
    assert result['arm'] == 'uniform_oe'
    assert result['eligible_methods'] == ['msp'] and result['selected_method'] == 'msp'
    assert result['next_step'] == 'SOURCE_EMBEDDING_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'
    assert all(m['diagnostic_passed'] for m in result['methods'].values())
    assert not result['methods']['energy']['eligible']
    metrics = result['methods']['msp']['overall']
    assert metrics['baseline_known_macro_closed_accuracy'] == 1.
    assert metrics['known_macro_closed_accuracy'] == metrics['known_macro_open_accuracy'] == 1.
    assert metrics['encoder_closed_loss'] == metrics['total_open_loss'] == metrics['rejection_loss'] == 0.
    assert metrics['auroc_known'] == metrics['oscr'] == 1.
    assert len(result['methods']['msp']['by_known_class']) == 6
    assert result['final_phase3_complete'] is False and result['evidence_scope'] == 'source_only_not_target'
    assert json.loads((case['output'] / 'results.json').read_text()) == result
    assert json.loads((case['output'] / 'complete.json').read_text())['selected_method'] == 'msp'


def test_native_supervised_control_cannot_promote_even_when_all_metrics_pass(tmp_path):
    result = run(fixture(tmp_path, arm='supervised_only'))
    assert result['methods']['msp']['diagnostic_passed']
    assert result['eligible_methods'] == [] and result['selected_method'] is None
    assert result['next_step'] == 'NO_PROMOTION'


def test_diagnostic_best_score_cannot_replace_predeclared_msp(case):
    mutate_predictions(case, lambda d: d['accepted'].__setitem__((np.isin(d['ids'], case['novel_ids']), 0), True))
    result = run(case)
    assert result['methods']['maxlogit']['diagnostic_passed']
    assert not result['methods']['msp']['diagnostic_passed']
    assert result['eligible_methods'] == [] and result['next_step'] == 'NO_PROMOTION'


def test_new_head_damage_is_not_hidden_by_small_self_rejection_loss(case):
    damaged = [case['known_ids'][np.flatnonzero(case['known_rx'] == rx)[0]] for rx in RX]
    def damage(data):
        mask = np.isin(data['ids'], damaged)
        data['closed_prediction'][mask] = (data['closed_prediction'][mask] + 1) % 6
    mutate_predictions(case, damage)
    result = run(case); m = result['methods']['msp']
    assert m['overall']['rejection_loss'] == 0.
    assert m['overall']['encoder_closed_loss'] == pytest.approx(1 / 120)
    assert m['overall']['total_open_loss'] < .02
    assert not m['goals']['overall']['checks']['encoder_closed_loss_at_most_005']
    assert result['next_step'] == 'NO_PROMOTION'  # scientific failure still emits complete scores


def test_single_rx_encoder_damage_cannot_hide_in_overall_average(case):
    damaged = case['known_ids'][np.flatnonzero(case['known_rx'] == 8)[0]]
    mutate_predictions(case, lambda d: d['closed_prediction'].__setitem__(d['ids'] == damaged, 1))
    result = run(case); goals = result['methods']['msp']['goals']
    assert goals['overall']['passed']
    assert not goals['by_receiver']['8']['checks']['encoder_closed_loss_at_most_005']
    assert result['selected_method'] is None


def test_exact_half_percentage_point_encoder_boundary_remains_eligible(tmp_path):
    case = fixture(tmp_path, known_repeats=[100] * 6)
    damaged = np.concatenate([case['known_ids'][case['known_rx'] == rx][:3] for rx in RX])
    mutate_predictions(case, lambda d: d['closed_prediction'].__setitem__(np.isin(d['ids'], damaged), 1))
    result = run(case)
    assert result['methods']['msp']['overall']['encoder_closed_loss'] == pytest.approx(.005)
    assert result['selected_method'] == 'msp'


def test_total_open_loss_includes_encoder_and_rejection_loss(tmp_path):
    case = fixture(tmp_path, known_repeats=[100] * 6)
    damaged, rejected = [], []
    for rx in RX:
        ids = case['known_ids'][case['known_rx'] == rx]
        damaged.extend(ids[:2]); rejected.extend(ids[2:13])
    def damage(data):
        mask = np.isin(data['ids'], damaged)
        data['closed_prediction'][mask] = (data['closed_prediction'][mask] + 1) % 6
        data['accepted'][np.isin(data['ids'], rejected), :] = False
    mutate_predictions(case, damage)
    result = run(case); m = result['methods']['msp']
    assert m['overall']['encoder_closed_loss'] == pytest.approx(2 / 600)
    assert m['overall']['rejection_loss'] == pytest.approx(11 / 600)
    assert m['overall']['total_open_loss'] == pytest.approx(13 / 600)
    assert m['goals']['overall']['checks']['encoder_closed_loss_at_most_005']
    assert not m['goals']['overall']['checks']['total_open_loss_at_most_02']
    assert result['selected_method'] is None


@pytest.mark.parametrize('fault', ['missing_marker', 'marker_status', 'target_bool', 'arm',
    'truth', 'query', 'source', 'known_labels', 'calibration', 'head_replaced',
    'encoder_updates', 'head_updates_bool', 'parent_identity', 'count', 'methods',
    'duplicate_ids', 'nan', 'accepted_type', 'bad_prediction', 'baseline_range',
    'baseline_type', 'rx', 'day', 'missing_baseline', 'missing_file'])
def test_invalid_frozen_prediction_contract_fails_before_truth(case, monkeypatch, fault):
    scorer = module(); marker_file = case['root'] / 'complete.json'
    marker_faults = dict(marker_status=('status', 'INCOMPLETE'), target_bool=('target_reads', False),
        arm=('arm', 'other'), truth=('unknown_truth_read', True), query=('query_fit', True),
        source=('source_only', False), known_labels=('source_V_labels_read', False),
        calibration=('source_known_label_calibration', False), head_replaced=('default_head_replaced', True),
        encoder_updates=('encoder_updates', 221), head_updates_bool=('identity_head_updates', True),
        parent_identity=('parent_export_identity', {}), count=('count', 1))
    if fault == 'missing_marker': marker_file.unlink()
    elif fault in marker_faults:
        marker = json.loads(marker_file.read_text()); key, value = marker_faults[fault]
        marker[key] = value; marker_file.write_text(json.dumps(marker))
    elif fault == 'missing_file': case['predictions'].unlink()
    else:
        with np.load(case['predictions']) as stored: data = {k: stored[k].copy() for k in stored.files}
        if fault == 'methods': data['method_names'] = np.asarray(METHODS[::-1])
        elif fault == 'duplicate_ids': data['ids'][0] = data['ids'][1]
        elif fault == 'nan': data['knownness'][0, 0] = np.nan
        elif fault == 'accepted_type': data['accepted'] = data['accepted'].astype(np.int64)
        elif fault == 'bad_prediction': data['prediction'][0, 0] = 6
        elif fault == 'baseline_range': data['baseline_closed_prediction'][0] = 6
        elif fault == 'baseline_type': data['baseline_closed_prediction'] = data['baseline_closed_prediction'].astype(np.float32)
        elif fault == 'rx': data['rx_i'][0] = 99
        elif fault == 'day': data['day_i'][0] = 4
        elif fault == 'missing_baseline': del data['baseline_closed_prediction']
        np.savez(case['predictions'], **data)
    original, opened = np.load, []
    def guard(path, *args, **kwargs):
        if Path(path) in (case['known_features'], case['novel_truth']):
            opened.append(str(path)); raise AssertionError('Truth opened before frozen predictions validated')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(scorer.np, 'load', guard)
    with pytest.raises((ValueError, FileNotFoundError)): run(case)
    assert opened == [] and not case['output'].exists()


def test_original_baseline_mismatch_fails_before_novel_truth(case, monkeypatch):
    def damage(data):
        mask = data['ids'] == case['known_ids'][0]
        data['baseline_closed_prediction'][mask] = (data['baseline_closed_prediction'][mask] + 1) % 6
    mutate_predictions(case, damage)
    original = np.load
    def guard(path, *args, **kwargs):
        if Path(path) == case['novel_truth']: raise AssertionError('Novel truth opened for wrong parent baseline')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module().np, 'load', guard)
    with pytest.raises(ValueError, match='baseline'): run(case)
    assert not case['output'].exists()


@pytest.mark.parametrize('fault', ['coverage', 'metadata', 'logits'])
def test_truth_alignment_and_parent_head_contract(case, fault):
    if fault == 'coverage':
        with np.load(case['novel_truth']) as stored: ids, tx = stored['ids'].copy(), stored['tx']
        ids[0] = case['known_ids'][0]; np.savez(case['novel_truth'], ids=ids, tx=tx)
    elif fault == 'metadata':
        with np.load(case['known_features']) as stored: rx = stored['rx_i'].copy()
        rx[0] = 1 if rx[0] != 1 else 3; change(case['known_features'], rx_i=rx)
    else:
        with np.load(case['known_features']) as stored: logits = stored['logits'].copy()
        logits[0, 0] = np.nan; change(case['known_features'], logits=logits)
    with pytest.raises(ValueError): run(case)
    assert not case['output'].exists()


def test_unknown_macro_is_per_tx_not_frequency_weighted(tmp_path):
    case = fixture(tmp_path, novel_repeats=[100, 1, 1])
    accepted_ids = case['novel_ids'][case['novel_tx'] == 'device-1']
    mutate_predictions(case, lambda d: d['accepted'].__setitem__((np.isin(d['ids'], accepted_ids), slice(None)), True))
    result = run(case); metrics = result['methods']['msp']['overall']
    assert metrics['unknown_rejection'] > .99
    assert metrics['unknown_macro_rejection'] == pytest.approx(2 / 3)
    assert result['next_step'] == 'NO_PROMOTION'


def test_encoder_protection_is_per_class_not_pooled(tmp_path):
    case = fixture(tmp_path, known_repeats=[1, 100, 100, 100, 100, 100])
    rare = case['known_ids'][case['known_y'] == 0]
    mutate_predictions(case, lambda d: d['closed_prediction'].__setitem__(np.isin(d['ids'], rare), 1))
    result = run(case); metrics = result['methods']['msp']['overall']
    assert 1 - metrics['known_closed_accuracy'] < .005
    assert metrics['encoder_closed_loss'] == pytest.approx(1 / 6)
    assert metrics['known_per_class'][0]['encoder_closed_loss'] == 1.
    assert result['selected_method'] is None


def test_all_reject_and_missing_novel_rx_cell_cannot_pass(case, tmp_path):
    mutate_predictions(case, lambda d: d['accepted'].__setitem__((slice(None), slice(None)), False))
    result = run(case)
    assert result['methods']['msp']['overall']['unknown_macro_rejection'] == 1.
    assert result['selected_method'] is None
    other = tmp_path / 'other'; other.mkdir(); case = fixture(other)
    with np.load(case['novel_truth']) as stored: ids, tx = stored['ids'], stored['tx'].copy()
    rx8_last_tx = case['novel_ids'][-2:]; tx[np.isin(ids, rx8_last_tx)] = 'device-0'
    np.savez(case['novel_truth'], ids=ids, tx=tx)
    result = run(case)
    assert result['methods']['msp']['by_receiver']['8']['unknown_classes_present'] == 2
    assert result['selected_method'] is None


def test_missing_known_class_is_scored_as_ineligible(case):
    with np.load(case['known_features']) as stored: y = stored['y'].copy()
    y[y == 5] = 4; change(case['known_features'], y=y)
    result = run(case)
    assert not result['methods']['msp']['goals']['overall']['checks']['all_six_known_classes_present']
    assert result['selected_method'] is None


def test_class_renaming_and_prediction_row_order_leave_metrics_unchanged(case, tmp_path):
    first = run(case)
    permutation = np.asarray([4, 2, 5, 1, 0, 3])
    with np.load(case['known_features']) as stored: y, logits = stored['y'], stored['logits']
    change(case['known_features'], y=permutation[y], logits=logits[:, np.argsort(permutation)])
    def rename(data):
        data['closed_prediction'] = permutation[data['closed_prediction']]
        data['baseline_closed_prediction'] = permutation[data['baseline_closed_prediction']]
    mutate_predictions(case, rename)
    with np.load(case['predictions']) as stored: data = {k: stored[k].copy() for k in stored.files}
    for key in data:
        if key != 'method_names': data[key] = data[key][::-1]
    np.savez(case['predictions'], **data)
    case['output'] = tmp_path / 'renamed_score'; result = run(case)
    for key in ['baseline_known_macro_closed_accuracy', 'encoder_closed_loss', 'total_open_loss',
                'rejection_loss', 'known_macro_open_accuracy', 'unknown_macro_rejection', 'auroc_known', 'oscr']:
        assert result['methods']['msp']['overall'][key] == first['methods']['msp']['overall'][key]


def test_existing_output_is_not_overwritten(case):
    case['output'].mkdir(); (case['output'] / 'keep').write_text('keep')
    with pytest.raises(FileExistsError): run(case)
    assert (case['output'] / 'keep').read_text() == 'keep'


def test_cli_uses_the_independent_four_path_contract(case, capsys):
    assert module().main(['--predictions', str(case['predictions']), '--known-features', str(case['known_features']),
        '--novel-truth', str(case['novel_truth']), '--output', str(case['output'])]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'SCORED_SOURCE_EMBEDDING_COMPLETE'
