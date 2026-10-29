"""Independent truth-last U matching and future-Q registration evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import development_transfer as transfer
from .scoring import _truth
from .source_risk_calibration import METHODS


RUN = '20261029-phase3-unlabeled-registration-s2026092701-r01'
SCHEMA = 'pamf_phase3_unlabeled_registration_scores_v1'
STATUS = 'SCORED_UNLABELED_REGISTRATION_COMPLETE'
STATES = ('DA0_REG0', 'DA1_REG0', 'DA0_REG1', 'DA1_REG1')
OLD_NAMES = ('14-10', '14-7', '20-15', '20-19', '6-15', '8-20')
PRIMARY = 'DA1_REG1'


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _prediction_gate(root):
    from .unlabeled_registration_target import _predictions
    return _predictions(root)


def _matching(y, assignments, candidate, count, arrivals):
    """Only U truth enters this one-to-one match; old labels are never targets."""
    table = np.asarray([[int(np.sum((assignments == c) & (y == label))) for label in arrivals]
                        for c in range(count)], np.int64).reshape(count, len(arrivals))
    mapping = {}
    if count and len(arrivals):
        from scipy.optimize import linear_sum_assignment
        rows, columns = linear_sum_assignment(-table)
        mapping = {int(row): int(arrivals[col]) for row, col in zip(rows, columns) if table[row, col] > 0}
    clusters = []; purity_numerator = 0; assigned_count = 0
    for c in range(count):
        mask = assignments == c; size = int(mask.sum())
        labels, counts = np.unique(y[mask], return_counts=True)
        composition = {str(int(label)): int(n) for label, n in zip(labels, counts)}
        old = int(np.sum(mask & (y < 6)))
        largest = int(counts.max()) if size else 0
        purity_numerator += largest; assigned_count += size
        clusters.append(dict(cluster_index=c, size=size, old_count=old, new_arrival_count=size-old,
            composition=composition, purity=None if not size else largest/size,
            matched_true_arrival=mapping.get(c), old_contamination=None if not size else old/size,
            true_arrival_labels_present=[int(label) for label in labels if label >= 6]))
    positive = table > 0
    return mapping, dict(mapping={str(k): v for k, v in mapping.items()},
        mapping_source='U truth only; maximum-count one-to-one Hungarian; zero cells remain unmatched',
        contingency=dict(cluster_indices=list(range(count)), true_arrival_labels=list(arrivals), counts=table.tolist()),
        clusters=clusters, prototype_count=count, matched_count=len(mapping), unmatched_cluster_indices=[c for c in range(count) if c not in mapping],
        candidate_count=int(candidate.sum()), noncandidate_count=int((~candidate).sum()),
        candidate_noise_count=int(np.sum(candidate & (assignments < 0))), unassigned_count=int(np.sum(assignments < 0)),
        old_contaminated_cluster_count=sum(c['old_count'] > 0 for c in clusters),
        pure_old_cluster_count=sum(c['size'] > 0 and c['old_count'] == c['size'] for c in clusters),
        merge_cluster_count=int(np.sum(positive.sum(axis=1) > 1)),
        split_true_arrival_count=int(np.sum(positive.sum(axis=0) > 1)),
        split_cluster_counts_by_true_arrival={str(label): int(n) for label, n in zip(arrivals, positive.sum(axis=0))},
        weighted_purity=None if not assigned_count else purity_numerator/assigned_count,
        U_clustering_diagnostic_only=True, future_registration_accuracy=False)


def _mapped(raw, mapping):
    result = raw.copy()
    anonymous = raw >= 6
    result[anonymous] = -2  # An accepted wrong identity, never a reject.
    for cluster, label in mapping.items(): result[raw == 6 + cluster] = label
    return result


def _class_rows(y, raw, mapped, baseline, labels, *, missing_zero):
    rows = []
    for label in labels:
        mask = y == label; count = int(mask.sum())
        correct = int(np.sum(mask & (mapped == label)))
        rejected = int(np.sum(mask & (raw == -1)))
        original_correct = int(np.sum(mask & (baseline == label)))
        rows.append(dict(label=int(label), count=count, correct_accepted=correct, rejected=rejected,
            accepted=count-rejected, open_accuracy=correct/count if count else (0. if missing_zero else None),
            acceptance=(count-rejected)/count if count else (0. if missing_zero else None),
            frr=rejected/count if count else (1. if missing_zero else None),
            original_F0_closed_accuracy=original_correct/count if count else (0. if missing_zero else None),
            original_F0_closed_correct=original_correct))
    return rows


def _mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def _rejection(y, raw, labels, complete):
    rows = []
    for label in labels:
        mask = y == label; n = int(mask.sum()); rejected = int(np.sum(mask & (raw == -1)))
        rows.append(dict(label=int(label), count=n, rejected=rejected,
                         rejection=rejected/n if n else (0. if complete else None)))
    return _mean(rows, 'rejection'), rows


def _metrics(y, raw, mapped, baseline, q, registered, arrivals, later, actual_unregistered, *, complete=True):
    present = set(np.unique(y).tolist())
    old_labels = list(range(6)) if complete else sorted(present & set(range(6)))
    new_labels = list(arrivals) if complete else sorted(present & set(arrivals))
    intended = list(range(6)) + list(arrivals)
    if not complete: intended = sorted(present & set(intended))
    old = _class_rows(y, raw, mapped, baseline, old_labels, missing_zero=complete)
    new = _class_rows(y, raw, mapped, baseline, new_labels, missing_zero=complete) if registered else []
    intended_rows = _class_rows(y, raw, mapped, baseline, intended, missing_zero=complete) if registered else []
    old_open, old_accept = _mean(old, 'open_accuracy'), _mean(old, 'acceptance')
    original = _mean(old, 'original_F0_closed_accuracy'); old_frr = _mean(old, 'frr')
    new_open = _mean(new, 'open_accuracy') if registered else None
    harmonic = (None if not registered or old_open is None or new_open is None else
                0. if old_open + new_open == 0 else 2 * old_open * new_open / (old_open + new_open))
    nonold = sorted(present - set(range(6)))
    all_nonold_rejection, all_nonold_rows = _rejection(y, raw, nonold, complete)
    planned, later_rows = _rejection(y, raw, later if complete else sorted(present & set(later)), complete)
    actual, actual_rows = _rejection(y, raw, actual_unregistered if complete else sorted(present & set(actual_unregistered)), complete)
    old_mask = y < 6; new_mask = np.isin(y, arrivals)
    result = dict(count=len(y), old_count=int(old_mask.sum()), old_accepted=int(np.sum(old_mask & (raw != -1))),
        old_rejected=int(np.sum(old_mask & (raw == -1))), old_correct_accepted=int(np.sum(old_mask & (mapped == y))),
        original_F0_closed_correct=int(np.sum(old_mask & (baseline == y))),
        old_macro_acceptance=old_accept, old_macro_frr=old_frr, old_macro_open_accuracy=old_open,
        original_F0_macro_closed_accuracy=original, original_F0_minus_old_open=None if original is None or old_open is None else original-old_open,
        old_per_class=old, old_class_coverage=all(row['count'] > 0 for row in old),
        new_macro_open_accuracy=new_open, new_macro_denominator=len(new_labels) if registered else None,
        new_arrival_count=int(new_mask.sum()), new_correct_accepted=int(np.sum(new_mask & (mapped == y))) if registered else None,
        new_per_class=new if registered else None, H=harmonic,
        intended_registered_macro_frr=_mean(intended_rows, 'frr') if registered else None,
        intended_registered_macro_acceptance=_mean(intended_rows, 'acceptance') if registered else None,
        intended_registered_class_count=len(intended) if registered else None,
        intended_registered_per_class=intended_rows if registered else None,
        planned_later_macro_rejection=planned, planned_later_per_class=later_rows,
        actual_unregistered_macro_rejection=actual, actual_unregistered_per_class=actual_rows,
        unmatched_anonymous_accepted=int(np.sum(mapped == -2)),
        all_future_nonold_macro_rejection=all_nonold_rejection,
        unknown_macro_rejection=all_nonold_rejection if not registered else None,
        known_macro_acceptance=old_accept, known_macro_frr=old_frr, known_macro_open_accuracy=old_open,
        original_closed_minus_new_open=None if original is None or old_open is None else original-old_open,
        unknown_per_class=all_nonold_rows if not registered else None)
    if not registered:
        ranked = transfer._metrics(y, baseline, baseline, q, raw != -1)
        result.update(auroc_known=ranked['auroc_known'], oscr=ranked['oscr'])
    else: result.update(auroc_known=None, oscr=None)
    return result


def _passes_old(row):
    return (row['old_macro_acceptance'] is not None and row['old_macro_acceptance'] >= .95
            and row['original_F0_minus_old_open'] is not None and row['original_F0_minus_old_open'] <= .02 + 1e-12)


def _registration_goals(row, control, required, minimum):
    def checks(current, reg0):
        retention = None if current['old_macro_open_accuracy'] is None or reg0['old_macro_open_accuracy'] is None else reg0['old_macro_open_accuracy']-current['old_macro_open_accuracy']
        return dict(old_known_and_absolute_F0_protection=_passes_old(current),
            old_retention_from_same_DA_REG0=retention is not None and retention <= .02 + 1e-12,
            intended_registered_macro_FRR=current['intended_registered_macro_frr'] is not None and current['intended_registered_macro_frr'] <= .05 + 1e-12,
            planned_later_unknown_protection=current['planned_later_macro_rejection'] is not None and current['planned_later_macro_rejection'] >= minimum)
    overall = checks(row['overall'], control['overall'])
    overall['new_arrival_macro_accuracy_80'] = row['overall']['new_macro_open_accuracy'] is not None and row['overall']['new_macro_open_accuracy'] >= .8
    coverage = set(row['by_scenario']) == required and set(control['by_scenario']) == required
    scenarios = {s: checks(r, control['by_scenario'][s]) for s, r in row['by_scenario'].items() if s in control['by_scenario']}
    return dict(overall=overall, by_scenario=scenarios, main_scene_coverage=coverage,
        new_accuracy_each_scene_reported_not_required=True,
        passed=coverage and all(overall.values()) and all(all(v.values()) for v in scenarios.values()))


def _delta(first, second):
    keys = ('old_macro_open_accuracy', 'old_macro_acceptance', 'old_macro_frr', 'original_F0_minus_old_open',
            'new_macro_open_accuracy', 'H', 'intended_registered_macro_frr', 'planned_later_macro_rejection')
    return {k: None if first[k] is None or second[k] is None else first[k]-second[k] for k in keys}


def score(predictions_root, U_truth, Q_truth, targets, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('A new immutable registration score output root is required')
    done, p, diag, _ = _prediction_gate(predictions_root)  # No truth IO before this complete target gate.
    if (set(p['ids'].tolist()) & set(p['U_ids'].tolist()) or not np.array_equal(diag['ids'], p['U_ids'])):
        raise ValueError('U and future Q must have disjoint physical IDs and aligned U diagnostics')
    spec, minimum = transfer._targets(targets, done['view_family'])
    if (spec.get('anonymous_registration', {}).get('future_independent_new_class_macro_accuracy_min') != .8
            or spec['anonymous_registration'].get('old_class_drop_from_pre_registration_max') != .02
            or spec.get('unlabeled_adaptation', {}).get('known_open_accuracy_gain_required') is not True):
        raise ValueError('Current adaptation gain and independent new80/old2pp targets required')
    u, future = _truth(U_truth, p['U_ids']), _truth(Q_truth, p['ids'])
    if not bool((u['receiver'] == 26).all()) or not bool((future['receiver'] == 26).all()):
        raise ValueError('Only RX24-1 index26 U and future Q development truth are allowed')
    uy, y, scenes = u['y'], future['y'], future['scenario']
    arrivals = sorted(int(label) for label in np.unique(uy) if label >= 6)
    q_new = set(int(label) for label in np.unique(y) if label >= 6)
    later = sorted(q_new - set(arrivals))
    mappings, clusters = {}, {}
    for da in ('DA0', 'DA1'):
        mappings[da], clusters[da] = _matching(uy, diag['assignments_' + da], diag['candidate_' + da], done['prototype_counts'][da], arrivals)
    rows = {}
    for i, name in enumerate(STATES):
        da = 'DA' + str(i % 2); registered = i >= 2; raw = p['prediction'][i]
        matched = mappings[da] if registered else {}
        actual_unregistered = sorted(q_new - (set(matched.values()) if registered else set()))
        mapped = _mapped(raw, matched) if registered else raw
        def metrics(mask, complete=True):
            return _metrics(y[mask], raw[mask], mapped[mask], p['baseline_closed_prediction'][mask],
                p['knownness'][i % 2, mask, 6], registered, arrivals, later, actual_unregistered, complete=complete)
        rows[name] = dict(overall=metrics(np.ones(len(y), bool)),
            by_scenario={str(s): metrics(scenes == s) for s in sorted(set(scenes.tolist()))},
            by_transmitter={str(int(label)): metrics(y == label, False) for label in np.unique(y)},
            registered=registered, matched_arrival_labels=sorted(matched.values()), actual_unregistered_labels=actual_unregistered,
            new_macro_denominator_rule='All true U arrivals, including undiscovered/unmatched; REG0 is N/A',
            unknown_rejection_rule='Only raw prediction -1; unmatched anonymous -2 remains accepted wrong')
    required = {'clean'} if done['view_family'] == 'clean_cross_receiver' else {'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban'}
    for name in STATES[:2]:
        goals = transfer._goals(rows[name]['overall'], rows[name]['by_scenario'], minimum, done['view_family'])
        rows[name]['goals'] = dict(checks=goals['checks'], passed=goals['pass'])
    registration_goals = {da: _registration_goals(rows[da + '_REG1'], rows[da + '_REG0'], required, minimum) for da in ('DA0', 'DA1')}
    for da in ('DA0', 'DA1'): rows[da + '_REG1']['goals'] = registration_goals[da]
    before, after = rows['DA0_REG0']['overall'], rows['DA1_REG0']['overall']
    gain = None if before['old_macro_open_accuracy'] is None or after['old_macro_open_accuracy'] is None else after['old_macro_open_accuracy']-before['old_macro_open_accuracy']
    adaptation_checks = dict(nonidentity_bias_applied=done['bias_applied'] is True,
        old_macro_open_accuracy_gain=gain is not None and gain > 0., condition_targets=rows['DA1_REG0']['goals']['passed'])
    adaptation_goals = dict(checks=adaptation_checks, old_macro_open_accuracy_gain=gain, passed=all(adaptation_checks.values()))
    pairs = [('DA1_REG0_minus_DA0_REG0', 'DA1_REG0', 'DA0_REG0'), ('DA1_REG1_minus_DA0_REG1', 'DA1_REG1', 'DA0_REG1'),
             ('DA0_REG1_minus_DA0_REG0', 'DA0_REG1', 'DA0_REG0'), ('DA1_REG1_minus_DA1_REG0', 'DA1_REG1', 'DA1_REG0')]
    deltas = {key: _delta(rows[a]['overall'], rows[b]['overall']) for key, a, b in pairs}
    did = {key: None if deltas['DA1_REG1_minus_DA1_REG0'][key] is None or deltas['DA0_REG1_minus_DA0_REG0'][key] is None else
           deltas['DA1_REG1_minus_DA1_REG0'][key]-deltas['DA0_REG1_minus_DA0_REG0'][key] for key in
           ('old_macro_open_accuracy', 'old_macro_acceptance', 'old_macro_frr', 'new_macro_open_accuracy', 'H')}
    diagnostics = {}
    for a, da in enumerate(('DA0', 'DA1')):
        methods = {}
        for j, method in enumerate(METHODS):
            accepted = p['diagnostic_reg0_prediction'][a, :, j] != -1
            def diagnostic(mask): return transfer._metrics(y[mask], p['closed_prediction'][mask], p['baseline_closed_prediction'][mask], p['knownness'][a, mask, j], accepted[mask])
            methods[method] = dict(overall=diagnostic(np.ones(len(y), bool)),
                by_scenario={str(s): diagnostic(scenes == s) for s in sorted(set(scenes.tolist()))},
                by_transmitter={str(int(label)): diagnostic(y == label) for label in np.unique(y)}, diagnostic_only=True)
        diagnostics[da] = methods
    passed = adaptation_goals['passed'] and registration_goals['DA1']['passed']
    result = dict(status=STATUS, schema=SCHEMA, run_id=done['run_id'], condition=done['condition'], view_family=done['view_family'],
        count=len(y), U_count=len(uy), truth_last=True, development_only=True, states=rows, cluster_diagnostics=clusters,
        U_true_arrival_labels=arrivals, planned_later_labels=later, public_old_class_names=list(OLD_NAMES),
        transmitter_group_keys='Truth-side integer transmitter labels; old names are public; no unknown roster enters prediction',
        deltas=deltas, difference_in_differences=did, diagnostic_methods_reg0=diagnostics,
        adaptation_goals=adaptation_goals, registration_goals=registration_goals,
        primary_state=PRIMARY, primary_method='mlp16_oe', primary_pass=passed,
        condition_unknown_rejection_min=minimum, targets_schema=spec['schema'], Q_truth_used_for_cluster_mapping=False,
        mapping_fitted_on='Independent U truth only after all prediction states complete', future_Q_fitting=0,
        new_macro_denominator='All true U arrivals, including undiscovered/unmatched devices; REG0 new/H are N/A',
        target_truth_used_for_prediction=False, algorithm_target_label_inputs=0, source_weights_read=0, source_labels_read=0,
        source_parameters_updated=0, identity_head_updates=0, optimizer_updates=0, formal_target_reads=0,
        target_U_truth_reads_by_predictor=0, business_registration_authorized=False, final_phase3_complete=False,
        U_cluster_metrics_are_future_evidence=False,
        evidence_scope='Development-only future-Q anonymous recognition; identities have no operational ownership/authorization',
        next_step='DEVELOPMENT_G0_G1_G2_PIPELINE_PASSED' if passed else 'NO_PROMOTION')
    output.mkdir(parents=True, exist_ok=False); _write(output / 'scores.json', result)
    _write(output / 'complete.json', {k: result[k] for k in ('status', 'schema', 'run_id', 'condition', 'truth_last',
        'primary_state', 'primary_method', 'primary_pass', 'next_step', 'final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='UNLABELED_REGISTRATION_SCORE_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))), flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-root', required=True); parser.add_argument('--U-truth', required=True)
    parser.add_argument('--Q-truth', required=True); parser.add_argument('--targets', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv); result = score(args.predictions_root, args.U_truth, args.Q_truth, args.targets, args.output)
    print(json.dumps(dict(status=result['status'], primary_pass=result['primary_pass'])), flush=True); return 0


if __name__ == '__main__': raise SystemExit(main())
