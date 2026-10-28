"""Independent truth-last source-OE rejection scoring; no adaptation/registration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .branch_export import DIMENSIONS
from .scores import METHODS as STATIC_METHODS
from .scoring import group_metrics


METHODS = tuple(STATIC_METHODS) + ('linear_oe', 'mlp16_oe')
OE_METHODS = ('linear_oe', 'mlp16_oe')
SOURCE_RECEIVERS = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
PREDICTION_KEYS = {'ids', 'method_names', 'closed_prediction', 'knownness', 'accepted', 'prediction', 'rx_i', 'day_i'}
KNOWN_KEYS = {'ids', 'logits', 'rx_i', 'day_i', 'y'} | {'features__' + name for name in DIMENSIONS}
NOVEL_KEYS = {'ids', 'tx'}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _load(path, expected, selected=None):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != expected:
            raise ValueError('NPZ schema differs: ' + str(path))
        return {key: stored[key].copy() for key in (selected if selected is not None else expected)}


def _strings(value, name, *, unique=False):
    if (value.ndim != 1 or value.dtype.kind != 'U' or not len(value)
            or np.any(np.char.strip(value) == '') or (unique and len(set(value.tolist())) != len(value))):
        raise ValueError('Nonempty Unicode strings required: ' + name)


def _integer(value, shape, name, *, allowed=None):
    if value.shape != shape or value.dtype.kind not in 'iu' or (allowed is not None and not np.isin(value, allowed).all()):
        raise ValueError('Integer shape/range differs: ' + name)


def _predictions(path):
    try:
        marker = _read_json(Path(path).parent / 'complete.json')
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: completed source-only predictions required') from exc
    if (not isinstance(marker, dict) or marker.get('status') != 'ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE'
            or marker.get('source_only') is not True or type(marker.get('target_reads')) is not int
            or marker['target_reads'] != 0 or marker.get('truth_read') is not False
            or marker.get('query_fit') is not False or marker.get('method_names') != list(METHODS)):
        raise ValueError('Truth closed: frozen truth-blind seven-method source predictions required')
    for key in ('known_count', 'novel_count'):
        if type(marker.get(key)) is not int or marker[key] <= 0:
            raise ValueError('Truth closed: positive known/novel marker counts required')
    values = _load(path, PREDICTION_KEYS)
    _strings(values['ids'], 'prediction IDs', unique=True)
    names = values['method_names']
    if names.ndim != 1 or names.dtype.kind != 'U' or names.tolist() != list(METHODS):
        raise ValueError('Truth closed: method names/order differ')
    n, m = len(values['ids']), len(METHODS)
    if n != marker['known_count'] + marker['novel_count']:
        raise ValueError('Truth closed: prediction count differs from completed role coverage')
    _integer(values['closed_prediction'], (n,), 'complete closed head', allowed=np.arange(6))
    _integer(values['rx_i'], (n,), 'source receiver', allowed=SOURCE_RECEIVERS)
    _integer(values['day_i'], (n,), 'source day', allowed=SOURCE_DAYS)
    _integer(values['prediction'], (n, m), 'prediction', allowed=np.arange(-1, 6))
    scores, accepted = values['knownness'], values['accepted']
    if scores.shape != (n, m) or scores.dtype.kind != 'f' or not np.isfinite(scores).all():
        raise ValueError('Truth closed: finite floating knownness [N,7] required')
    if accepted.shape != (n, m) or accepted.dtype.kind != 'b':
        raise ValueError('Truth closed: accepted must be boolean [N,7]')
    if not np.array_equal(values['prediction'], np.where(accepted, values['closed_prediction'][:, None], -1)):
        raise ValueError('Truth closed: predictions must use common frozen head and acceptance')
    return values, marker


def _truth(known_features, novel_truth, predictions, marker):
    # First truth IO. These feature arrays/logits are intentionally never materialized.
    known = _load(known_features, KNOWN_KEYS, ('ids', 'y', 'rx_i', 'day_i'))
    _strings(known['ids'], 'known truth IDs', unique=True)
    nk = len(known['ids'])
    _integer(known['y'], (nk,), 'known labels', allowed=np.arange(6))
    _integer(known['rx_i'], (nk,), 'known source receiver', allowed=SOURCE_RECEIVERS)
    _integer(known['day_i'], (nk,), 'known source day', allowed=SOURCE_DAYS)
    novel = _load(novel_truth, NOVEL_KEYS)
    _strings(novel['ids'], 'novel truth IDs', unique=True)
    _strings(novel['tx'], 'novel TX')
    if novel['tx'].shape != novel['ids'].shape:
        raise ValueError('Novel IDs and TX shape differ')
    known_ids, novel_ids = set(known['ids'].tolist()), set(novel['ids'].tolist())
    if known_ids & novel_ids:
        raise ValueError('Known/novel physical ID coverage overlaps')
    if (nk != marker['known_count'] or len(novel['ids']) != marker['novel_count']
            or set(predictions['ids'].tolist()) != known_ids | novel_ids):
        raise ValueError('Prediction/truth exact ID coverage differs')
    lookup = {value: j for j, value in enumerate(predictions['ids'].tolist())}
    known_positions = np.asarray([lookup[value] for value in known['ids'].tolist()], np.int64)
    novel_positions = np.asarray([lookup[value] for value in novel['ids'].tolist()], np.int64)
    if (not np.array_equal(predictions['rx_i'][known_positions], known['rx_i'])
            or not np.array_equal(predictions['day_i'][known_positions], known['day_i'])):
        raise ValueError('Known prediction/truth receiver/day metadata differs')
    transmitters = sorted(set(novel['tx'].tolist()))
    tx_to_label = {tx: 6 + j for j, tx in enumerate(transmitters)}
    y = np.empty(len(predictions['ids']), dtype=np.int64)
    y[known_positions] = known['y']
    y[novel_positions] = np.asarray([tx_to_label[value] for value in novel['tx'].tolist()], np.int64)
    return y, {label: tx for tx, label in tx_to_label.items()}


def _group(y, closed, knownness, accepted, label_to_tx):
    metrics = group_metrics(y, closed, knownness, accepted, 6)
    a, b = metrics['known_macro_closed_accuracy'], metrics['known_macro_open_accuracy']
    metrics['known_macro_accuracy_drop'] = None if a is None or b is None else a - b
    metrics['all_rejected'] = not bool(np.any(accepted))
    for row in metrics['unknown_per_class']:
        row['transmitter'] = label_to_tx[row['label']]
    return metrics


def _at_least(value, threshold):
    return value is not None and value >= threshold - 1e-12


def _at_most(value, threshold):
    return value is not None and value <= threshold + 1e-12


def _checks(metrics):
    checks = dict(nonempty_known_and_unknown=bool(metrics['known_count'] and metrics['unknown_count']),
        all_six_known_classes_present=metrics['known_classes_present'] == 6,
        all_three_novel_transmitters_present=metrics['unknown_classes_present'] == 3,
        not_all_rejected=not metrics['all_rejected'],
        unknown_macro_rejection_at_least_95=_at_least(metrics['unknown_macro_rejection'], .95),
        known_macro_frr_at_most_05=_at_most(metrics['known_macro_frr'], .05),
        known_macro_accuracy_drop_at_most_02=_at_most(metrics['known_macro_accuracy_drop'], .02))
    return dict(checks=checks, passed=all(checks.values()))


def score(predictions, known_features, novel_truth, output):
    """Score frozen seven-method source predictions; counts follow marker/ID truth."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Source score output must be a new directory')
    pred, marker = _predictions(predictions)
    y, label_to_tx = _truth(known_features, novel_truth, pred, marker)
    methods = {}
    for j, name in enumerate(METHODS):
        def metrics(mask):
            return _group(y[mask], pred['closed_prediction'][mask], pred['knownness'][mask, j],
                          pred['accepted'][mask, j], label_to_tx)
        overall = metrics(np.ones(len(y), dtype=bool))
        receivers = {str(rx): metrics(pred['rx_i'] == rx) for rx in SOURCE_RECEIVERS}
        overall_goals = _checks(overall)
        receiver_goals = {name: _checks(value) for name, value in receivers.items()}
        passed = overall_goals['passed'] and all(value['passed'] for value in receiver_goals.values())
        methods[name] = dict(overall=overall, by_receiver=receivers,
            by_unknown_transmitter={row['transmitter']: dict(row) for row in overall['unknown_per_class']},
            goals=dict(overall=overall_goals, by_receiver=receiver_goals, passed=passed),
            diagnostic_passed=passed, promotable_candidate=name in OE_METHODS,
            eligible=name in OE_METHODS and passed)
    # A static control is never a new OE candidate; parameter priority is fixed.
    eligible = [name for name in OE_METHODS if methods[name]['eligible']]
    selected = eligible[0] if eligible else None
    next_step = 'SOURCE_OPEN_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if selected else 'NO_PROMOTION'
    result = dict(schema='pamf_phase3_source_open_scores_v1', status='SCORED_SOURCE_OPEN_COMPLETE',
        source_only=True, target_reads=0, truth_last=True, query_fit=False, evidence_scope='source_only_not_target',
        known_count=marker['known_count'], novel_count=marker['novel_count'], novel_transmitter_count=len(label_to_tx),
        novel_label_to_transmitter={str(key): value for key, value in label_to_tx.items()},
        source_receivers=list(SOURCE_RECEIVERS), source_days=list(SOURCE_DAYS), method_names=list(METHODS),
        methods=methods, eligible_methods=eligible, selected_method=selected, next_step=next_step,
        candidate_parameter_counts=dict(linear_oe=101, mlp16_oe=1633), final_phase3_complete=False,
        domain_adaptation_performed=False, anonymous_registration_performed=False,
        metric_definitions=dict(known_macro='Equal mean over known labels 0..5; rejected predictions count as incorrect',
            unknown_macro='Equal mean over private true TX, independent of their record counts',
            protection='Gate uses known_macro_accuracy_drop; legacy known_accuracy_drop is pooled diagnostic only',
            rejection='Only prediction -1; every accepted unknown is a false acceptance',
            receiver_gate='Overall and each fixed source RX must contain six known classes and three novel TX',
            ranks='Known-positive AUROC and OSCR, O(N log N), ties admitted together',
            intervals='95% Wilson per-class/pooled received-record binomial rates, no macro CI claimed',
            selection='Only two OE candidates may advance; if both pass choose linear_oe with fewer parameters',
            scope='Source membership supervised rejection development; no unseen-RX, satellite, adaptation or registration success claim'))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'results.json', result)
    _write(output / 'complete.json', dict(status=result['status'], source_only=True, target_reads=0,
        truth_last=True, evidence_scope=result['evidence_scope'], eligible_methods=eligible,
        selected_method=selected, next_step=next_step, final_phase3_complete=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'known-features', 'novel-truth', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = score(args.predictions, args.known_features, args.novel_truth, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'eligible_methods', 'selected_method', 'next_step')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
