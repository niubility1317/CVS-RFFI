"""Independent truth-last metrics for frozen Phase3 predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PREDICTION_KEYS = {'ids', 'method_names', 'closed_prediction', 'knownness', 'accepted', 'prediction'}
TRUTH_KEYS = {'ids', 'y', 'receiver', 'scenario'}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _load_npz(path, keys):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != keys:
            raise ValueError('NPZ schema differs: ' + str(sorted(data.files)))
        return {key: data[key].copy() for key in keys}


def _strings(value, name, *, unique=False):
    if value.ndim != 1 or value.dtype.kind != 'U' or not len(value) or np.any(value == ''):
        raise ValueError(name + ' must contain nonempty unicode strings')
    if unique and len(set(value.tolist())) != len(value):
        raise ValueError(name + ' contains duplicates')


def _predictions(path, known_class_count):
    marker = Path(path).parent / 'complete.json'
    try:
        done = _read_json(marker)
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: prediction completion marker missing or invalid') from exc
    if (done.get('status') != 'ALL_PREDICTIONS_COMPLETE' or done.get('truth_read') is not False
            or done.get('query_fit') is not False):
        raise ValueError('Truth closed: all predictions must be complete, truth-blind and frozen')
    values = _load_npz(path, PREDICTION_KEYS)
    _strings(values['ids'], 'prediction ids', unique=True)
    _strings(values['method_names'], 'method names', unique=True)
    n, m = len(values['ids']), len(values['method_names'])
    closed = values['closed_prediction']
    if (closed.shape != (n,) or closed.dtype.kind not in 'iu' or np.any(closed < 0)
            or np.any(closed >= known_class_count)):
        raise ValueError('Truth closed: invalid frozen closed-set class prediction')
    score, accepted, predicted = (values[key] for key in ('knownness', 'accepted', 'prediction'))
    if score.shape != (n, m) or score.dtype.kind != 'f' or not np.isfinite(score).all():
        raise ValueError('Truth closed: finite knownness [N,M] required')
    if accepted.shape != (n, m) or accepted.dtype.kind != 'b':
        raise ValueError('Truth closed: boolean accepted [N,M] required')
    if predicted.shape != (n, m) or predicted.dtype.kind not in 'iu':
        raise ValueError('Truth closed: integer prediction [N,M] required')
    expected = np.where(accepted, closed[:, None], -1)
    if not np.array_equal(predicted, expected):
        raise ValueError('Truth closed: prediction differs from frozen acceptance and closed prediction')
    return values


def _truth(path, ids):
    values = _load_npz(path, TRUTH_KEYS)
    _strings(values['ids'], 'truth ids', unique=True)
    _strings(values['scenario'], 'truth scenarios')
    n = len(values['ids'])
    for key in ('y', 'receiver'):
        if values[key].shape != (n,) or values[key].dtype.kind not in 'iu':
            raise ValueError('Truth ' + key + ' must be integer [N]')
    if values['scenario'].shape != (n,) or np.any(values['y'] < 0):
        raise ValueError('Invalid truth scenario shape or negative class')
    if set(ids.tolist()) != set(values['ids'].tolist()):
        raise ValueError('Prediction/truth ID coverage differs')
    lookup = {item: i for i, item in enumerate(values['ids'].tolist())}
    order = np.asarray([lookup[item] for item in ids.tolist()], dtype=np.int64)
    return {key: value[order] for key, value in values.items()}


def wilson95(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1. + z * z / count
    center = (p + z * z / (2. * count)) / denominator
    half = z * np.sqrt(p * (1. - p) / count + z * z / (4. * count * count)) / denominator
    return [max(0., float(center - half)), min(1., float(center + half))]


def rank_metrics(known, correct_known, knownness):
    """Known-positive AUROC and OSCR, with tied scores admitted together."""
    known = np.asarray(known, dtype=bool)
    correct = np.asarray(correct_known, dtype=bool)
    scores = np.asarray(knownness)
    if (known.ndim != 1 or correct.shape != known.shape or scores.shape != known.shape
            or not np.isfinite(scores).all() or np.any(correct & ~known)):
        raise ValueError('Invalid rank metric inputs')
    nk, nu = int(known.sum()), int((~known).sum())
    if not nk or not nu:
        return None, None
    order = np.argsort(-scores, kind='mergesort')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    false_accept = np.r_[0., np.cumsum(~known[order], dtype=np.int64)[ends] / nu]
    true_accept = np.r_[0., np.cumsum(known[order], dtype=np.int64)[ends] / nk]
    correct_accept = np.r_[0., np.cumsum(correct[order], dtype=np.int64)[ends] / nk]
    dx = np.diff(false_accept)
    auroc = np.sum(dx * (true_accept[:-1] + true_accept[1:]) * .5)
    oscr = np.sum(dx * (correct_accept[:-1] + correct_accept[1:]) * .5)
    return float(np.clip(auroc, 0., 1.)), float(np.clip(oscr, 0., 1.))


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _mean(values):
    return float(np.mean(values)) if values else None


def group_metrics(y, closed, knownness, accepted, known_class_count):
    known = y < known_class_count
    unknown = ~known
    nk, nu = int(known.sum()), int(unknown.sum())
    correct = known & (closed == y)
    known_correct, known_correct_accepted = int(correct.sum()), int((correct & accepted).sum())
    known_rejected, unknown_accepted = int((known & ~accepted).sum()), int((unknown & accepted).sum())
    unknown_rejected = nu - unknown_accepted
    known_rows, unknown_rows = [], []
    for label in np.unique(y):
        mask = y == label
        count = int(mask.sum())
        rejected = int((mask & ~accepted).sum())
        if label < known_class_count:
            row = dict(label=int(label), count=count, rejected=rejected, frr=rejected / count,
                       closed_accuracy=float((closed[mask] == label).mean()),
                       open_accuracy=float(((closed[mask] == label) & accepted[mask]).mean()),
                       rejection_interval95=wilson95(rejected, count))
            known_rows.append(row)
        else:
            unknown_rows.append(dict(label=int(label), count=count, rejected=rejected,
                                     rejection=rejected / count, rejection_interval95=wilson95(rejected, count)))
    auroc, oscr = rank_metrics(known, correct, knownness)
    baseline, opened = _rate(known_correct, nk), _rate(known_correct_accepted, nk)
    return dict(count=len(y), known_count=nk, unknown_count=nu,
                known_classes_present=len(known_rows), unknown_classes_present=len(unknown_rows),
                known_closed_correct=known_correct, known_correct_accepted=known_correct_accepted,
                known_rejected=known_rejected, unknown_accepted=unknown_accepted, unknown_rejected=unknown_rejected,
                known_closed_accuracy=baseline, known_open_accuracy=opened,
                known_accuracy_drop=None if baseline is None else baseline - opened,
                known_macro_closed_accuracy=_mean([r['closed_accuracy'] for r in known_rows]),
                known_macro_open_accuracy=_mean([r['open_accuracy'] for r in known_rows]),
                known_frr=_rate(known_rejected, nk), known_macro_frr=_mean([r['frr'] for r in known_rows]),
                unknown_rejection=_rate(unknown_rejected, nu),
                unknown_macro_rejection=_mean([r['rejection'] for r in unknown_rows]),
                unknown_far=_rate(unknown_accepted, nu),
                auroc_known=auroc, oscr=oscr, known_per_class=known_rows, unknown_per_class=unknown_rows,
                confidence_intervals95=dict(known_closed_accuracy=wilson95(known_correct, nk),
                    known_open_accuracy=wilson95(known_correct_accepted, nk), known_frr=wilson95(known_rejected, nk),
                    unknown_rejection=wilson95(unknown_rejected, nu), unknown_far=wilson95(unknown_accepted, nu)))


def _goals(overall, scenarios):
    enough = bool(overall['known_count'] and overall['unknown_count'])
    checks = dict(nonempty_known_and_unknown=enough,
        unknown_macro_rejection_at_least_95=overall['unknown_macro_rejection'] is not None and overall['unknown_macro_rejection'] >= .95,
        each_scenario_unknown_rejection_at_least_95=bool(scenarios) and all(
            r['unknown_macro_rejection'] is not None and r['unknown_macro_rejection'] >= .95 for r in scenarios.values()),
        known_macro_frr_at_most_05=overall['known_macro_frr'] is not None and overall['known_macro_frr'] <= .05,
        known_accuracy_drop_at_most_02=overall['known_accuracy_drop'] is not None and overall['known_accuracy_drop'] <= .02 + 1e-12)
    return dict(checks, **{'pass': all(checks.values())})


def score(predictions, truth, output, known_class_count=6):
    if isinstance(known_class_count, bool) or not isinstance(known_class_count, int) or known_class_count <= 0:
        raise ValueError('Positive integer known_class_count required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Score output must be a new directory: ' + str(output))
    pred = _predictions(Path(predictions), known_class_count)
    actual = _truth(Path(truth), pred['ids'])  # First truth IO: predictions have passed every check.
    methods = {}
    for j, name in enumerate(pred['method_names'].tolist()):
        def metrics(mask):
            return group_metrics(actual['y'][mask], pred['closed_prediction'][mask], pred['knownness'][mask, j],
                                 pred['accepted'][mask, j], known_class_count)
        overall = metrics(np.ones(len(pred['ids']), dtype=bool))
        receivers = {str(value): metrics(actual['receiver'] == value) for value in np.unique(actual['receiver'])}
        scenarios = {str(value): metrics(actual['scenario'] == value) for value in np.unique(actual['scenario'])}
        unknown_tx = {str(value): metrics(actual['y'] == value) for value in np.unique(actual['y']) if value >= known_class_count}
        methods[name] = dict(overall=overall, by_receiver=receivers, by_scenario=scenarios,
                             by_unknown_transmitter=unknown_tx, goals=_goals(overall, scenarios))
    result = dict(schema='pamf_phase3_open_world_scores_v1', status='SCORED_COMPLETE',
                  truth_last=True, target_feedback_forbidden=True, known_class_count=known_class_count,
                  count=len(pred['ids']), methods=methods,
                  metric_definitions=dict(knownness='Higher scores indicate known registered classes',
                      auroc_known='Known is positive; tied thresholds enter together',
                      oscr='Area of correct-known acceptance versus unknown false acceptance',
                      macro='Equal mean across classes present in the reported stratum',
                      intervals='95% Wilson intervals for pooled and per-class binomial rates; no macro-mean interval claimed',
                      interval_unit='Received record; interval interpretation assumes independent physical observations',
                      defer='No defer decision in this prediction schema; -1 means rejected'))
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'scores.json', result)
    _write_json(output / 'complete.json', dict(status='SCORED_COMPLETE', truth_last=True,
                target_feedback_forbidden=True, count=len(pred['ids']), methods=list(methods),
                goals={name: row['goals']['pass'] for name, row in methods.items()}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--known-class-count', type=int, default=6)
    args = parser.parse_args(argv)
    result = score(args.predictions, args.truth, args.output, args.known_class_count)
    print(json.dumps(dict(status=result['status'], output=args.output,
          goals={name: row['goals']['pass'] for name, row in result['methods'].items()}), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
