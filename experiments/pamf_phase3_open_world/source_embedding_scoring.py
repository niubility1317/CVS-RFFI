"""Truth-last source embedding scores with independent original-F protection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scores import METHODS
from .scoring import group_metrics, wilson95
from .source_open_scoring import (
    KNOWN_KEYS, NOVEL_KEYS, SOURCE_DAYS, SOURCE_RECEIVERS,
    _at_least, _at_most, _integer, _load, _read_json, _strings, _write,
)


PREDICTION_KEYS = {'ids', 'method_names', 'closed_prediction', 'baseline_closed_prediction',
                   'knownness', 'accepted', 'prediction', 'rx_i', 'day_i'}
ARMS = ('supervised_only', 'uniform_oe')
STATUS = 'SCORED_SOURCE_EMBEDDING_COMPLETE'


def _predictions(path):
    try:
        marker = _read_json(Path(path).parent / 'complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: completed source embedding predictions required') from exc
    if (not isinstance(marker, dict) or marker.get('status') != 'SOURCE_EMBEDDING_PREDICTIONS_COMPLETE'
            or marker.get('source_only') is not True or type(marker.get('target_reads')) is not int
            or marker['target_reads'] != 0 or marker.get('unknown_truth_read') is not False
            or marker.get('query_fit') is not False or marker.get('source_V_labels_read') is not True
            or marker.get('source_known_label_calibration') is not True
            or marker.get('default_head_replaced') is not False or marker.get('arm') not in ARMS
            or marker.get('method_names') != list(METHODS)
            or not isinstance(marker.get('parent_export_identity'), dict) or not marker['parent_export_identity']):
        raise ValueError('Truth closed: frozen five-method source embedding contract required')
    for key in ('encoder_updates', 'identity_head_updates'):
        if type(marker.get(key)) is not int or marker[key] != 222:
            raise ValueError('Truth closed: exactly 222 native encoder/head updates required')
    for key in ('known_count', 'novel_count', 'count'):
        if type(marker.get(key)) is not int or marker[key] <= 0:
            raise ValueError('Truth closed: positive integer prediction role counts required')
    if marker['count'] != marker['known_count'] + marker['novel_count']:
        raise ValueError('Truth closed: completed role counts differ')
    values = _load(path, PREDICTION_KEYS)
    _strings(values['ids'], 'prediction IDs', unique=True)
    names = values['method_names']
    if names.ndim != 1 or names.dtype.kind != 'U' or names.tolist() != list(METHODS):
        raise ValueError('Truth closed: fixed static-five method names/order required')
    n, m = len(values['ids']), len(METHODS)
    if n != marker['count']:
        raise ValueError('Truth closed: prediction count differs from completed coverage')
    for key in ('closed_prediction', 'baseline_closed_prediction'):
        _integer(values[key], (n,), key, allowed=np.arange(6))
    _integer(values['rx_i'], (n,), 'source receiver', allowed=SOURCE_RECEIVERS)
    _integer(values['day_i'], (n,), 'source day', allowed=SOURCE_DAYS)
    _integer(values['prediction'], (n, m), 'prediction', allowed=np.arange(-1, 6))
    scores, accepted = values['knownness'], values['accepted']
    if scores.shape != (n, m) or scores.dtype.kind != 'f' or not np.isfinite(scores).all():
        raise ValueError('Truth closed: finite floating knownness [N,5] required')
    if accepted.shape != (n, m) or accepted.dtype.kind != 'b':
        raise ValueError('Truth closed: boolean accepted [N,5] required')
    if not np.array_equal(values['prediction'], np.where(accepted, values['closed_prediction'][:, None], -1)):
        raise ValueError('Truth closed: accepted predictions must use the complete new default head')
    return values, marker


def _truth(known_features, novel_truth, pred, marker):
    # First label IO, after all frozen prediction checks. Large feature payloads
    # stay unopened; original parent logits independently fix the old baseline.
    known = _load(known_features, KNOWN_KEYS, ('ids', 'y', 'rx_i', 'day_i', 'logits'))
    _strings(known['ids'], 'known truth IDs', unique=True)
    nk = len(known['ids'])
    _integer(known['y'], (nk,), 'known labels', allowed=np.arange(6))
    _integer(known['rx_i'], (nk,), 'known source receiver', allowed=SOURCE_RECEIVERS)
    _integer(known['day_i'], (nk,), 'known source day', allowed=SOURCE_DAYS)
    logits = known['logits']
    if logits.shape != (nk, 6) or logits.dtype.kind != 'f' or not np.isfinite(logits).all():
        raise ValueError('Finite complete original parent logits [known,6] required')
    lookup = {value: j for j, value in enumerate(pred['ids'].tolist())}
    if nk != marker['known_count'] or not set(known['ids'].tolist()) <= set(lookup):
        raise ValueError('Known prediction/truth ID coverage differs')
    kp = np.asarray([lookup[value] for value in known['ids'].tolist()], np.int64)
    if (not np.array_equal(pred['rx_i'][kp], known['rx_i'])
            or not np.array_equal(pred['day_i'][kp], known['day_i'])):
        raise ValueError('Known prediction/truth receiver/day metadata differs')
    if not np.array_equal(pred['baseline_closed_prediction'][kp], np.argmax(logits, axis=1)):
        raise ValueError('Original parent baseline differs from independently loaded parent logits')
    # A wrong original baseline never triggers novel/private TX truth access.
    novel = _load(novel_truth, NOVEL_KEYS)
    _strings(novel['ids'], 'novel truth IDs', unique=True)
    _strings(novel['tx'], 'novel TX')
    if novel['tx'].shape != novel['ids'].shape:
        raise ValueError('Novel IDs and TX shape differ')
    ki, ni = set(known['ids'].tolist()), set(novel['ids'].tolist())
    if ki & ni:
        raise ValueError('Known/novel physical ID coverage overlaps')
    if len(novel['ids']) != marker['novel_count'] or set(lookup) != ki | ni:
        raise ValueError('Prediction/truth exact ID coverage differs')
    tx_to_label = {tx: 6 + j for j, tx in enumerate(sorted(set(novel['tx'].tolist())))}
    y = np.empty(len(pred['ids']), np.int64)
    y[kp] = known['y']
    positions = np.asarray([lookup[value] for value in novel['ids'].tolist()], np.int64)
    y[positions] = np.asarray([tx_to_label[value] for value in novel['tx'].tolist()], np.int64)
    return y, {label: tx for tx, label in tx_to_label.items()}


def _group(y, closed, baseline, knownness, accepted, label_to_tx):
    metrics = group_metrics(y, closed, knownness, accepted, 6)
    baseline_rates = []
    for row in metrics['known_per_class']:
        mask = y == row['label']
        correct = int(np.count_nonzero(baseline[mask] == row['label']))
        rate = correct / row['count']
        row.update(baseline_closed_accuracy=rate,
            baseline_closed_accuracy_interval95=wilson95(correct, row['count']),
            encoder_closed_loss=rate - row['closed_accuracy'],
            total_open_loss=rate - row['open_accuracy'],
            rejection_loss=row['closed_accuracy'] - row['open_accuracy'])
        baseline_rates.append(rate)
    known = y < 6
    count = metrics['known_count']
    correct = int(np.count_nonzero(known & (baseline == y)))
    baseline_macro = float(np.mean(baseline_rates)) if baseline_rates else None
    new_closed, new_open = metrics['known_macro_closed_accuracy'], metrics['known_macro_open_accuracy']
    metrics.update(baseline_known_closed_correct=correct,
        baseline_known_closed_accuracy=correct / count if count else None,
        baseline_known_macro_closed_accuracy=baseline_macro,
        encoder_closed_loss=None if baseline_macro is None else baseline_macro - new_closed,
        total_open_loss=None if baseline_macro is None else baseline_macro - new_open,
        rejection_loss=None if new_closed is None else new_closed - new_open,
        known_macro_accuracy_drop=None if new_closed is None else new_closed - new_open,
        all_rejected=not bool(np.any(accepted)))
    metrics['confidence_intervals95']['baseline_known_closed_accuracy'] = wilson95(correct, count)
    for row in metrics['unknown_per_class']:
        row['transmitter'] = label_to_tx[row['label']]
    return metrics


def _checks(metrics):
    checks = dict(nonempty_known_and_unknown=bool(metrics['known_count'] and metrics['unknown_count']),
        all_six_known_classes_present=metrics['known_classes_present'] == 6,
        all_three_novel_transmitters_present=metrics['unknown_classes_present'] == 3,
        not_all_rejected=not metrics['all_rejected'],
        unknown_macro_rejection_at_least_95=_at_least(metrics['unknown_macro_rejection'], .95),
        known_macro_frr_at_most_05=_at_most(metrics['known_macro_frr'], .05),
        encoder_closed_loss_at_most_005=_at_most(metrics['encoder_closed_loss'], .005),
        total_open_loss_at_most_02=_at_most(metrics['total_open_loss'], .02))
    return dict(checks=checks, passed=all(checks.values()))


def score(predictions, known_features, novel_truth, output):
    """Score source-only fixed STATIC5; only uniform_oe::msp may advance."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Source embedding score output must be a new directory')
    pred, marker = _predictions(predictions)
    y, label_to_tx = _truth(known_features, novel_truth, pred, marker)
    methods = {}
    for j, name in enumerate(METHODS):
        def metrics(mask):
            return _group(y[mask], pred['closed_prediction'][mask], pred['baseline_closed_prediction'][mask],
                          pred['knownness'][mask, j], pred['accepted'][mask, j], label_to_tx)
        overall = metrics(np.ones(len(y), dtype=bool))
        receivers = {str(rx): metrics(pred['rx_i'] == rx) for rx in SOURCE_RECEIVERS}
        overall_goals = _checks(overall)
        receiver_goals = {rx: _checks(value) for rx, value in receivers.items()}
        passed = overall_goals['passed'] and all(value['passed'] for value in receiver_goals.values())
        promotable = marker['arm'] == 'uniform_oe' and name == 'msp'
        methods[name] = dict(overall=overall, by_receiver=receivers,
            by_known_class={str(row['label']): dict(row) for row in overall['known_per_class']},
            by_unknown_transmitter={row['transmitter']: dict(row) for row in overall['unknown_per_class']},
            goals=dict(overall=overall_goals, by_receiver=receiver_goals, passed=passed),
            diagnostic_passed=passed, promotable_candidate=promotable, eligible=promotable and passed)
    eligible = ['msp'] if methods['msp']['eligible'] else []
    selected = 'msp' if eligible else None
    next_step = 'SOURCE_EMBEDDING_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if selected else 'NO_PROMOTION'
    result = dict(schema='pamf_phase3_source_embedding_scores_v1', status=STATUS, arm=marker['arm'],
        source_only=True, target_reads=0, truth_last=True, unknown_truth_read=True, query_fit=False,
        evidence_scope='source_only_not_target', independent_confirmation=False,
        source_novel_evaluation_reused=True, source_known_label_calibration=True,
        encoder_updates=222, identity_head_updates=222, default_head_replaced=False,
        original_parent_baseline_verified=True, parent_export_identity=marker['parent_export_identity'],
        known_count=marker['known_count'], novel_count=marker['novel_count'], count=len(y),
        novel_transmitter_count=len(label_to_tx),
        novel_label_to_transmitter={str(k): v for k, v in label_to_tx.items()},
        source_receivers=list(SOURCE_RECEIVERS), source_days=list(SOURCE_DAYS), method_names=list(METHODS),
        methods=methods, eligible_methods=eligible, selected_method=selected, next_step=next_step,
        final_phase3_complete=False, domain_adaptation_performed=False, anonymous_registration_performed=False,
        metric_definitions=dict(known_macro='Equal mean over known classes; rejected predictions count as incorrect',
            unknown_macro='Equal mean over private true TX; every accepted unknown is a false acceptance',
            new_closed='Complete incremented default six-class head, shared by all methods',
            baseline='Original frozen parent default logits argmax, independently verified on every known ID',
            encoder_closed_loss='Original baseline macro accuracy minus new closed macro accuracy',
            total_open_loss='Original baseline macro accuracy minus new open macro accuracy',
            rejection_loss='New closed macro accuracy minus new open macro accuracy',
            protection='Overall and each source RX: encoder loss <=.005, total open loss <=.02, FRR <=.05',
            selection='Only predeclared uniform_oe::msp may advance; native L-supervised increment control and four other scores diagnostic only',
            intervals='95% Wilson per-class/pooled received-record rates, no macro confidence interval claimed',
            scope='Repeated source clean development with known-source empirical risk calibration, not independent target confirmation'))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'results.json', result)
    _write(output / 'complete.json', dict(status=STATUS, source_only=True, target_reads=0,
        truth_last=True, arm=marker['arm'], original_parent_baseline_verified=True,
        evidence_scope=result['evidence_scope'], eligible_methods=eligible, selected_method=selected,
        next_step=next_step, final_phase3_complete=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'known-features', 'novel-truth', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = score(args.predictions, args.known_features, args.novel_truth, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'arm', 'eligible_methods', 'selected_method', 'next_step')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
