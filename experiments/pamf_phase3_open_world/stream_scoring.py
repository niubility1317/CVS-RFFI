"""Independent truth-last scoring for the four-old-class source stream proxy.

This module never fits a predictor. Anonymous/TX matching uses U truth once;
that fixed matching, including unmatched slots, is then evaluated on future V.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scoring import wilson95


STATES = ('F', 'D0R0', 'D1R0', 'D0R1', 'D1R1')
DISCOVERY_STATES = ('D0', 'D1')
FUTURE_KEYS = {'ids', 'state_names', 'baseline_closed', 'closed_prediction',
               'knownness', 'accepted', 'prediction'}
DISCOVERY_KEYS = {'ids', 'state_names', 'anonymous_prediction'}
REGISTRY_KEYS = {name + suffix for name in DISCOVERY_STATES for suffix in
                 ('_anonymous_ids', '_converged', '_cap_reached', '_promotion_eligible')}
TRUTH_KEYS = {'ids', 'y'}


def _load_npz(path, keys):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != keys:
            raise ValueError('NPZ schema differs: ' + str(path))
        return {key: stored[key].copy() for key in keys}


def _ids(value, name):
    if (value.ndim != 1 or value.dtype.kind != 'U' or not len(value)
            or np.any(np.char.strip(value) == '') or len(set(value.tolist())) != len(value)):
        raise ValueError('Unique nonempty unicode IDs required: ' + name)


def _integer(value, shape, name, low=0, high=None):
    if (value.shape != shape or value.dtype.kind not in 'iu' or np.any(value < low)
            or (high is not None and np.any(value > high))):
        raise ValueError('Integer shape/range differs: ' + name)


def _state_names(value, expected):
    if value.ndim != 1 or value.dtype.kind != 'U' or value.tolist() != list(expected):
        raise ValueError('Fixed state names/order required')


def _predictions(future_path, discovery_path, registry_path):
    paths = [Path(path) for path in (future_path, discovery_path, registry_path)]
    if len({path.resolve().parent for path in paths}) != 1:
        raise ValueError('All prediction and registry files must share one directory')
    if len({path.resolve() for path in paths}) != 3 or not all(path.is_file() for path in paths):
        raise ValueError('Three distinct complete prediction/registry files required')
    try:
        marker = json.loads((paths[0].parent / 'complete.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ValueError('Truth closed: stream prediction completion marker missing/invalid') from exc
    if (not isinstance(marker, dict) or marker.get('status') != 'ALL_STREAM_PREDICTIONS_COMPLETE'
            or marker.get('truth_read') is not False or marker.get('query_fit') is not False
            or marker.get('source_only') is not True or type(marker.get('target_reads')) is not int
            or marker['target_reads'] != 0):
        raise ValueError('Truth closed: frozen truth-blind source-only stream required')
    for name in ('future_count', 'discovery_count'):
        if type(marker.get(name)) is not int or marker[name] <= 0:
            raise ValueError('Truth closed: positive marker counts required')

    future = _load_npz(paths[0], FUTURE_KEYS)
    discovery = _load_npz(paths[1], DISCOVERY_KEYS)
    raw_registry = _load_npz(paths[2], REGISTRY_KEYS)
    _ids(future['ids'], 'future')
    _ids(discovery['ids'], 'U')
    n, nu = len(future['ids']), len(discovery['ids'])
    if n != marker['future_count'] or nu != marker['discovery_count']:
        raise ValueError('Truth closed: prediction coverage/count differs from completion marker')
    if set(future['ids'].tolist()) & set(discovery['ids'].tolist()):
        raise ValueError('Truth closed: U and future physical IDs overlap')
    _state_names(future['state_names'], STATES)
    _state_names(discovery['state_names'], DISCOVERY_STATES)
    _integer(future['baseline_closed'], (n,), 'baseline_closed', high=3)
    _integer(future['closed_prediction'], (n, 5), 'closed_prediction', high=3)
    _integer(future['prediction'], (n, 5), 'prediction', low=-1)
    _integer(discovery['anonymous_prediction'], (nu, 2), 'anonymous_prediction', low=-1)
    scores, accepted = future['knownness'], future['accepted']
    if scores.shape != (n, 5) or scores.dtype.kind != 'f' or not np.isfinite(scores).all():
        raise ValueError('Truth closed: finite floating knownness [N,5] required')
    if accepted.shape != (n, 5) or accepted.dtype.kind != 'b':
        raise ValueError('Truth closed: boolean accepted [N,5] required')
    pred, closed = future['prediction'], future['closed_prediction']
    if not np.array_equal(accepted, pred != -1):
        raise ValueError('Truth closed: -1 must mean rejected and only rejected')
    old = (pred >= 0) & (pred < 4)
    if not np.array_equal(pred[old], closed[old]):
        raise ValueError('Truth closed: accepted old predictions must use the frozen complete head')
    for j in (0, 1, 3):
        if not np.array_equal(closed[:, j], future['baseline_closed']):
            raise ValueError('Truth closed: F/D0 must retain the original closed head output')
    if not np.array_equal(closed[:, 2], closed[:, 4]):
        raise ValueError('Truth closed: D1 R0/R1 must share corrected closed head outputs')
    if np.any(pred[:, :3] >= 4):
        raise ValueError('Truth closed: F/R0 cannot emit anonymous IDs')

    registry = {}
    for u_column, (name, r0, r1) in enumerate((('D0', 1, 3), ('D1', 2, 4))):
        vocabulary = raw_registry[name + '_anonymous_ids']
        _integer(vocabulary, (len(vocabulary),) if vocabulary.ndim else (-1,), name + ' IDs', low=4)
        if not np.array_equal(vocabulary, np.arange(4, 4 + len(vocabulary))):
            raise ValueError('Truth closed: anonymous IDs must be continuous from 4')
        flags = {}
        for suffix in ('converged', 'cap_reached', 'promotion_eligible'):
            value = raw_registry[name + '_' + suffix]
            if value.shape != () or value.dtype.kind != 'b':
                raise ValueError('Truth closed: registry flags must be boolean scalars')
            flags[suffix] = bool(value.item())
        expected = flags['converged'] and not flags['cap_reached'] and bool(len(vocabulary))
        if flags['promotion_eligible'] != expected:
            raise ValueError('Truth closed: registry eligibility disagrees with convergence/cap/slots')
        u_predictions = discovery['anonymous_prediction'][:, u_column]
        if (not np.isin(u_predictions, np.r_[-1, vocabulary]).all()
                or not np.isin(pred[:, r1], np.r_[-1, np.arange(4), vocabulary]).all()):
            raise ValueError('Truth closed: prediction contains an illegal anonymous ID')
        if set(u_predictions[u_predictions >= 4].tolist()) != set(vocabulary.tolist()):
            raise ValueError('Truth closed: each qualified slot needs U assignment coverage')
        # R0 and R1 reuse the same bank: only a qualified-new winner changes output.
        expected_r0 = np.where((pred[:, r1] >= 0) & (pred[:, r1] < 4), pred[:, r1], -1)
        if not np.array_equal(pred[:, r0], expected_r0):
            raise ValueError('Truth closed: matched R0/R1 old decisions differ')
        registry[name] = dict(flags, anonymous_ids=[int(v) for v in vocabulary])
    return future, discovery, registry


def _truth(path, ids, maximum):
    values = _load_npz(path, TRUTH_KEYS)
    _ids(values['ids'], 'truth')
    _integer(values['y'], (len(values['ids']),), 'truth labels', high=maximum)
    if set(values['ids'].tolist()) != set(ids.tolist()):
        raise ValueError('Prediction/truth ID coverage differs')
    lookup = {value: j for j, value in enumerate(values['ids'].tolist())}
    order = np.asarray([lookup[value] for value in ids.tolist()], dtype=np.int64)
    return values['y'][order]


def _rate(successes, count):
    return float(successes / count) if count else None


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _binomial(successes, count, name):
    return dict(count=int(count), **{name + '_count': int(successes), name: _rate(successes, count),
                                   name + '_interval95': wilson95(successes, count)})


def _registration(y, prediction, registry):
    counts = {str(label): int((y == label).sum()) for label in range(5)}
    components = []
    for anonymous_id in registry['anonymous_ids']:
        selected = prediction == anonymous_id
        class_counts = {str(label): int((selected & (y == label)).sum()) for label in range(5)}
        size, novel = int(selected.sum()), class_counts['4']
        components.append(dict(anonymous_id=anonymous_id, count=size, true_class_counts=class_counts,
                               discovery_purity=_rate(novel, size),
                               dominant_class_purity=_rate(max(class_counts.values()), size)))
    # One discovery TX in this source proxy. Ties select the smaller anonymous ID.
    positive = [row for row in components if row['true_class_counts']['4'] > 0]
    matched = min(positive, key=lambda row: (-row['true_class_counts']['4'], row['anonymous_id'])) if positive else None
    mapping = {} if matched is None else {str(matched['anonymous_id']): 4}
    assigned, old = prediction >= 4, y < 4
    novel_assigned = int(((y == 4) & assigned).sum())
    old_assigned = int((old & assigned).sum())
    return dict(registry, count=len(y), true_class_counts=counts, components=components,
                anonymous_to_source_label=mapping,
                matched_anonymous_id=None if matched is None else matched['anonymous_id'],
                matched_purity=None if matched is None else matched['discovery_purity'],
                matched_discovery_coverage=_rate(0 if matched is None else matched['true_class_counts']['4'], counts['4']),
                discovery_coverage=_rate(novel_assigned, counts['4']),
                registered_assignment_count=int(assigned.sum()), unassigned_count=int((~assigned).sum()),
                aggregate_discovery_purity=_rate(novel_assigned, int(assigned.sum())),
                old_count=int(old.sum()), old_misregistered_count=old_assigned,
                old_misregistration_rate=_rate(old_assigned, int(old.sum())),
                duplicate_registration_count=sum(row['true_class_counts']['4'] > 0 and
                    (matched is None or row['anonymous_id'] != matched['anonymous_id']) for row in components),
                unmatched_registration_count=len(components) - int(matched is not None),
                false_old_registration_count=sum(row['true_class_counts']['4'] == 0 for row in components))


def _future_metrics(y, prediction, baseline, closed, registered, mapping):
    old_rows, per_class = [], []
    for label in range(6):
        mask = y == label
        count = int(mask.sum())
        rejected = int((prediction[mask] == -1).sum())
        row = dict(label=label, **_binomial(rejected, count, 'rejection'))
        if label < 4:
            correct = int((prediction[mask] == label).sum())
            row.update(_binomial(correct, count, 'accuracy'))
            row.update(baseline_closed_accuracy=_rate(int((baseline[mask] == label).sum()), count),
                       same_input_closed_accuracy=_rate(int((closed[mask] == label).sum()), count))
            old_rows.append(row)
        per_class.append(row)
    old = y < 4
    old_count = int(old.sum())
    accuracy = _mean([row['accuracy'] for row in old_rows])
    baseline_accuracy = _mean([row['baseline_closed_accuracy'] for row in old_rows])
    same_input_accuracy = _mean([row['same_input_closed_accuracy'] for row in old_rows])
    new_mask, later_mask = y == 4, y == 5
    matched_ids = [int(key) for key, label in mapping.items() if label == 4]
    new_correct = int((new_mask & np.isin(prediction, matched_ids)).sum()) if registered else 0
    new4 = dict(per_class[4], registered=registered,
                accuracy=_rate(new_correct, int(new_mask.sum())) if registered else None,
                accuracy_count=new_correct if registered else None,
                accuracy_interval95=wilson95(new_correct, int(new_mask.sum())) if registered else None,
                unmatched_anonymous_accept_count=int((new_mask & (prediction >= 4) & ~np.isin(prediction, matched_ids)).sum()))
    unknown = ~old
    return dict(count=len(y), true_class_counts={str(k): int((y == k).sum()) for k in range(6)},
                all_six_classes_present=set(np.unique(y).tolist()) == set(range(6)),
                old_count=old_count, old_macro_accuracy=accuracy,
                old_macro_frr=_mean([row['rejection'] for row in old_rows]),
                old_macro_baseline_closed_accuracy=baseline_accuracy,
                old_macro_same_input_closed_accuracy=same_input_accuracy,
                old_accuracy_drop_from_baseline=None if accuracy is None else baseline_accuracy - accuracy,
                old_accuracy_drop_from_same_input_closed=None if accuracy is None else same_input_accuracy - accuracy,
                old_correct_count=int((old & (prediction == y)).sum()),
                old_correct_interval95=wilson95(int((old & (prediction == y)).sum()), old_count),
                old_rejected_count=int((old & (prediction == -1)).sum()),
                old_rejection_interval95=wilson95(int((old & (prediction == -1)).sum()), old_count),
                old_misregistered_count=int((old & (prediction >= 4)).sum()), old_per_class=old_rows,
                new4=new4, later5=dict(per_class[5], anonymous_accept_count=int((later_mask & (prediction >= 4)).sum())),
                initial_unknown_macro_rejection=_mean([per_class[4]['rejection'], per_class[5]['rejection']]),
                initial_unknown_pooled_rejection=_rate(int((unknown & (prediction == -1)).sum()), int(unknown.sum())),
                per_class=per_class, all_rejected=bool(np.all(prediction == -1)))


def _difference(left, right):
    return None if left is None or right is None else float(left - right)


def _at_least(value, threshold):
    return value is not None and value >= threshold - 1e-12


def _at_most(value, threshold):
    return value is not None and value <= threshold + 1e-12


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def score(future_predictions, discovery_predictions, registry, u_truth, future_truth, output):
    """Score a completed source-only stream; all six arguments are file/directory paths."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Score output must be a new directory')
    future, discovery, registry_values = _predictions(future_predictions, discovery_predictions, registry)
    # First truth IO occurs only after ALL prediction/registry checks above.
    u_y = _truth(u_truth, discovery['ids'], 4)
    y = _truth(future_truth, future['ids'], 5)
    registrations = {name: _registration(u_y, discovery['anonymous_prediction'][:, j], registry_values[name])
                     for j, name in enumerate(DISCOVERY_STATES)}
    states = {}
    for j, name in enumerate(STATES):
        bank = None if name == 'F' else name[:2]
        registered = name.endswith('R1')
        mapping = {} if bank is None else registrations[bank]['anonymous_to_source_label']
        metrics = _future_metrics(y, future['prediction'][:, j], future['baseline_closed'],
                                  future['closed_prediction'][:, j], registered, mapping)
        states[name] = dict(future=metrics, registry=bank, registered=registered)
    old_accuracy = {name: row['future']['old_macro_accuracy'] for name, row in states.items()}
    paired = dict(domain_correction_gain={r: _difference(old_accuracy['D1' + r], old_accuracy['D0' + r]) for r in ('R0', 'R1')},
                  registration_damage={d: _difference(old_accuracy[d + 'R0'], old_accuracy[d + 'R1']) for d in DISCOVERY_STATES})
    complete_u_classes = set(np.unique(u_y).tolist()) == set(range(5))
    for name, row in states.items():
        bank, metrics = row['registry'], row['future']
        reg = None if bank is None else registrations[bank]
        checks = dict(joint_state=name == 'D1R1', all_future_classes_present=metrics['all_six_classes_present'],
                      all_u_classes_present=complete_u_classes,
                      converged=reg is not None and reg['converged'],
                      cap_not_reached=reg is not None and not reg['cap_reached'],
                      qualified_component_exists=reg is not None and bool(reg['anonymous_ids']),
                      registry_promotion_eligible=reg is not None and reg['promotion_eligible'],
                      not_all_rejected=not metrics['all_rejected'],
                      later_unknown_rejection_at_least_95=_at_least(metrics['later5']['rejection'], .95),
                      old_macro_frr_at_most_05=_at_most(metrics['old_macro_frr'], .05),
                      old_accuracy_drop_from_baseline_at_most_02=_at_most(metrics['old_accuracy_drop_from_baseline'], .02))
        if row['registered']:
            checks.update(new4_accuracy_at_least_80=_at_least(metrics['new4']['accuracy'], .8),
                          registration_old_damage_at_most_02=_at_most(paired['registration_damage'][bank], .02))
        if bank == 'D1':
            gain = paired['domain_correction_gain'][name[-2:]]
            checks['same_registration_domain_correction_old_gain_positive'] = gain is not None and gain > 1e-12
        row.update(scientific_checks=checks,
                   control_passed=all(value for key, value in checks.items() if key != 'joint_state'),
                   eligible=all(checks.values()))
    eligible = [name for name in STATES if states[name]['eligible']]
    result = dict(schema='pamf_phase3_source_stream_scores_v1', status='SCORED_SOURCE_STREAM_COMPLETE',
                  evidence_scope='source_proxy_only_not_target', source_only=True, target_reads=0,
                  truth_last=True, query_fit=False, future_count=len(y), discovery_count=len(u_y),
                  state_names=list(STATES), registries=registrations, states=states, paired_old_metrics=paired,
                  eligible_states=eligible, final_six_class_method_selected=False,
                  next_step='SOURCE_JOINT_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if eligible else 'NO_PROMOTION',
                  metric_definitions=dict(old_accuracy='Equal mean over old classes 0..3; rejection and anonymous predictions are errors',
                      old_frr='Only prediction -1 is rejection; wrong old/anonymous acceptance is not FRR',
                      baseline='Original frozen head; corrected-head closed reference is separately reported',
                      anonymous_matching='One U-only count-maximizing match to source class 4; tie uses smaller anonymous ID; future never rematched',
                      initial_unknown='Labels 4 and 5 diagnostic only; after registration class 4 must be recognized',
                      later_unknown='Class 5 succeeds only when prediction is -1; unmatched anonymous remains acceptance',
                      intervals='95% Wilson per received-record pooled/per-class binomial rates; no macro CI claimed',
                      scope='F0 source L with hidden labels; source RX8 has cross-split historical exposure; not blind target evidence'))
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'results.json', result)
    _write(output / 'complete.json', dict(status=result['status'], source_only=True, target_reads=0,
                evidence_scope=result['evidence_scope'], truth_last=True, query_fit=False,
                eligible_states=eligible, next_step=result['next_step']))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('future-predictions', 'discovery-predictions', 'registry', 'u-truth', 'future-truth', 'output'):
        parser.add_argument('--' + argument, required=True)
    args = parser.parse_args(argv)
    result = score(args.future_predictions, args.discovery_predictions, args.registry,
                   args.u_truth, args.future_truth, args.output)
    print(json.dumps(dict(status=result['status'], next_step=result['next_step'], eligible_states=result['eligible_states'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
