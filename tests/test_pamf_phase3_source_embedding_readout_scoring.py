"""Focused source E51/readout combination contracts; all IQ/features synthetic."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


METHODS = ['msp', 'maxlogit', 'margin', 'energy', 'cosine', 'linear_oe', 'mlp16_oe']
RX = [1, 3, 4, 6, 8]
ENCODER_RUN = '20261029-phase3-source-embedding-increment-s2026092701-r02'
ENCODER_COMMIT = '28d53ea9b38c19f642c59e31f178a3f65f40894d'
BRANCHES = ['fused160', 'time160', 'frequency160', 'pa160', 'reference29',
            'signature336', 'spectrum128', 'pa_moments48']


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_readout_scoring')


def fixture(tmp_path, arm='supervised_only', known_repeats=None, novel_repeats=None):
    known_repeats = known_repeats or [20] * 6
    novel_repeats = novel_repeats or [2] * 3
    rows = [(r, y) for r in RX for y in range(6) for _ in range(known_repeats[y])]
    unknown = [(r, tx) for r in RX for tx in range(3) for _ in range(novel_repeats[tx])]
    nk, nu = len(rows), len(unknown)
    ids = np.asarray(['opaque-%05d' % n for n in np.random.default_rng(7).permutation(nk + nu)])
    y = np.asarray([row[1] for row in rows], np.int64)
    rx = np.asarray([row[0] for row in rows + unknown], np.int64)
    day = np.ones(nk + nu, np.int64)
    closed = np.r_[y, np.zeros(nu, np.int64)]
    accepted = np.repeat((np.arange(nk + nu) < nk)[:, None], 7, axis=1)
    root = tmp_path / 'predict'; root.mkdir()
    pred = root / 'predictions.npz'
    order = np.argsort(ids)
    np.savez(pred, ids=ids[order], method_names=np.asarray(METHODS), closed_prediction=closed[order],
        baseline_closed_prediction=closed[order], knownness=np.where(accepted, 1., -1.).astype(np.float32)[order],
        accepted=accepted[order], prediction=np.where(accepted, closed[:, None], -1)[order],
        rx_i=rx[order], day_i=day[order])
    identity = dict(run_id=ENCODER_RUN, row_id=arm + '-s2026092701', arm=arm,
        code_commit=ENCODER_COMMIT, checkpoint='/runs/' + ENCODER_RUN + '/rows/' + arm + '/train/final.pt')
    marker = dict(status='SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE', arm=arm,
        source_only=True, target_reads=0, count=nk + nu, known_count=nk, novel_count=nu,
        method_names=METHODS, encoder_updates=222, identity_head_updates=222,
        new_encoder_updates=0, new_identity_head_updates=0, readout_steps_per_model=200,
        default_head_replaced=False, source_V_labels_read=True, source_known_label_calibration=True,
        unknown_truth_read=False, query_fit=False, parent_export_identity={'model': 'original frozen F6'},
        encoder_state_identity=identity)
    (root / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
    known = tmp_path / 'parentV.npz'
    logits = np.full((nk, 6), -2., np.float32); logits[np.arange(nk), y] = 2.
    unused = {'features__' + name: np.asarray([{'not_read': True}], dtype=object) for name in BRANCHES}
    np.savez(known, ids=ids[:nk][::-1], y=y[::-1], logits=logits[::-1],
             rx_i=rx[:nk][::-1], day_i=day[:nk][::-1], **unused)
    truth = tmp_path / 'truth.npz'
    tx = np.asarray(['device-' + str(row[1]) for row in unknown])
    np.savez(truth, ids=ids[nk:][::-1], tx=tx[::-1])
    return dict(predictions=pred, known_features=known, novel_truth=truth, output=tmp_path / 'score',
                root=root, known_ids=ids[:nk], novel_ids=ids[nk:], known_y=y, known_rx=rx[:nk], novel_tx=tx)


@pytest.fixture
def case(tmp_path):
    return fixture(tmp_path)


def run(case):
    return module().score(case['predictions'], case['known_features'], case['novel_truth'], case['output'])


def mutate(case, change):
    with np.load(case['predictions']) as stored: d = {key: stored[key].copy() for key in stored.files}
    change(d)
    d['prediction'] = np.where(d['accepted'], d['closed_prediction'][:, None], -1)
    np.savez(case['predictions'], **d)


@pytest.mark.parametrize('arm', ['supervised_only', 'uniform_oe'])
def test_only_mlp_can_promote_on_either_fixed_encoder_and_history_is_not_new_training(tmp_path, arm):
    case = fixture(tmp_path, arm=arm); result = run(case)
    assert result['status'] == 'SCORED_SOURCE_EMBEDDING_READOUT_COMPLETE'
    assert result['eligible_methods'] == ['mlp16_oe'] and result['selected_method'] == 'mlp16_oe'
    assert result['next_step'] == 'SOURCE_EMBEDDING_READOUT_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'
    assert result['arm'] == arm and result['encoder_state_identity']['arm'] == arm
    assert result['encoder_updates'] == result['identity_head_updates'] == 222
    assert result['new_encoder_updates'] == result['new_identity_head_updates'] == 0
    assert result['readout_steps_per_model'] == 200
    assert result['methods']['linear_oe']['diagnostic_passed'] and not result['methods']['linear_oe']['eligible']
    assert result['methods']['msp']['diagnostic_passed'] and not result['methods']['msp']['eligible']
    metrics = result['methods']['mlp16_oe']['overall']
    assert metrics['baseline_known_macro_closed_accuracy'] == 1.
    assert metrics['total_open_loss'] == metrics['encoder_closed_loss'] == metrics['rejection_loss'] == 0.
    assert len(result['methods']['mlp16_oe']['by_receiver']) == 5
    assert len(result['methods']['mlp16_oe']['by_known_class']) == 6
    assert len(result['methods']['mlp16_oe']['by_unknown_transmitter']) == 3
    assert result['final_phase3_complete'] is False
    assert result['evidence_scope'] == 'source_only_not_target' and not result['independent_confirmation']
    assert json.loads((case['output'] / 'results.json').read_text()) == result
    assert json.loads((case['output'] / 'complete.json').read_text())['selected_method'] == 'mlp16_oe'


def test_linear_and_static_success_cannot_replace_failed_predeclared_mlp(case):
    mutate(case, lambda d: d['accepted'].__setitem__((np.isin(d['ids'], case['novel_ids']), 6), True))
    result = run(case)
    assert result['methods']['linear_oe']['diagnostic_passed']
    assert result['methods']['energy']['diagnostic_passed']
    assert not result['methods']['mlp16_oe']['diagnostic_passed']
    assert result['eligible_methods'] == [] and result['selected_method'] is None
    assert result['next_step'] == 'NO_PROMOTION'


@pytest.mark.parametrize('fault', ['new_encoder', 'new_head_bool', 'history_as_zero', 'readout_steps',
    'encoder_arm', 'encoder_row', 'encoder_checkpoint', 'encoder_commit', 'parent_identity',
    'status', 'unknown_truth', 'five_columns', 'method_order', 'missing_baseline', 'duplicate_ids'])
def test_new_seven_method_and_encoder_contract_fail_before_any_truth(case, monkeypatch, fault):
    marker_file = case['root'] / 'complete.json'; marker = json.loads(marker_file.read_text())
    if fault == 'new_encoder': marker['new_encoder_updates'] = 222
    elif fault == 'new_head_bool': marker['new_identity_head_updates'] = False
    elif fault == 'history_as_zero': marker['encoder_updates'] = 0
    elif fault == 'readout_steps': marker['readout_steps_per_model'] = 199
    elif fault == 'encoder_arm': marker['encoder_state_identity']['arm'] = 'uniform_oe'
    elif fault == 'encoder_row': marker['encoder_state_identity']['row_id'] = 'uniform_oe-s2026092701'
    elif fault == 'encoder_checkpoint': marker['encoder_state_identity']['checkpoint'] = '/runs/' + ENCODER_RUN + '/rows/uniform_oe/train/final.pt'
    elif fault == 'encoder_commit': marker['encoder_state_identity']['code_commit'] = 'wrong'
    elif fault == 'parent_identity': marker['parent_export_identity'] = {}
    elif fault == 'status': marker['status'] = 'SOURCE_EMBEDDING_PREDICTIONS_COMPLETE'
    elif fault == 'unknown_truth': marker['unknown_truth_read'] = True
    else:
        with np.load(case['predictions']) as stored: d = {key: stored[key].copy() for key in stored.files}
        if fault == 'five_columns':
            for key in ['knownness', 'accepted', 'prediction']: d[key] = d[key][:, :5]
        elif fault == 'method_order': d['method_names'] = np.asarray(METHODS[::-1])
        elif fault == 'missing_baseline': del d['baseline_closed_prediction']
        elif fault == 'duplicate_ids': d['ids'][0] = d['ids'][1]
        np.savez(case['predictions'], **d)
    marker_file.write_text(json.dumps(marker))
    original, opened = np.load, []
    def guard(path, *args, **kwargs):
        if Path(path) in (case['known_features'], case['novel_truth']):
            opened.append(str(path)); raise AssertionError('Truth opened before prediction checks')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module().np, 'load', guard)
    with pytest.raises(ValueError): run(case)
    assert opened == [] and not case['output'].exists()


def test_wrong_original_f_baseline_rejected_before_novel_truth(case, monkeypatch):
    mutate(case, lambda d: d['baseline_closed_prediction'].__setitem__(d['ids'] == case['known_ids'][0], 1))
    original = np.load
    def guard(path, *args, **kwargs):
        if Path(path) == case['novel_truth']: raise AssertionError('Novel truth opened for false E50 baseline')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(module().np, 'load', guard)
    with pytest.raises(ValueError, match='baseline'): run(case)
    assert not case['output'].exists()


def test_one_rx_encoder_damage_completes_scores_but_cannot_promote(case):
    damaged = case['known_ids'][np.flatnonzero(case['known_rx'] == 8)[0]]
    mutate(case, lambda d: d['closed_prediction'].__setitem__(d['ids'] == damaged, 1))
    result = run(case); method = result['methods']['mlp16_oe']
    assert method['overall']['rejection_loss'] == 0.
    assert method['goals']['overall']['passed']
    assert not method['goals']['by_receiver']['8']['checks']['encoder_closed_loss_at_most_005']
    assert result['selected_method'] is None
    assert (case['output'] / 'complete.json').exists()


def test_unknown_macro_and_rare_known_class_guard_do_not_use_pooled_denominators(tmp_path):
    root = tmp_path / 'novel'; root.mkdir()
    case = fixture(root, novel_repeats=[100, 1, 1])
    bad = case['novel_ids'][case['novel_tx'] == 'device-1']
    mutate(case, lambda d: d['accepted'].__setitem__((np.isin(d['ids'], bad), 6), True))
    result = run(case); metrics = result['methods']['mlp16_oe']['overall']
    assert metrics['unknown_rejection'] > .99 and metrics['unknown_macro_rejection'] == pytest.approx(2 / 3)
    assert result['selected_method'] is None
    root = tmp_path / 'known'; root.mkdir()
    case = fixture(root, known_repeats=[1, 100, 100, 100, 100, 100])
    rare = case['known_ids'][case['known_y'] == 0]
    mutate(case, lambda d: d['closed_prediction'].__setitem__(np.isin(d['ids'], rare), 1))
    result = run(case); metrics = result['methods']['mlp16_oe']['overall']
    assert 1. - metrics['known_closed_accuracy'] < .005
    assert metrics['encoder_closed_loss'] == pytest.approx(1 / 6)
    assert result['selected_method'] is None


def test_missing_truth_id_coverage_is_rejected(case):
    with np.load(case['novel_truth']) as stored: ids, tx = stored['ids'].copy(), stored['tx']
    ids[0] = 'not-in-frozen-predictions'; np.savez(case['novel_truth'], ids=ids, tx=tx)
    with pytest.raises(ValueError, match='coverage'): run(case)
    assert not case['output'].exists()


def test_cli_preserves_immutable_output_and_four_path_interface(case, capsys):
    args = ['--predictions', str(case['predictions']), '--known-features', str(case['known_features']),
            '--novel-truth', str(case['novel_truth']), '--output', str(case['output'])]
    assert module().main(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'SCORED_SOURCE_EMBEDDING_READOUT_COMPLETE'
    with pytest.raises(FileExistsError): module().main(args)
