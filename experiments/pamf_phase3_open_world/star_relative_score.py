"""Independent six-scene G10 truth-last evaluation of four RMD states."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import development_transfer as transfer
from . import star_relative_core as core
from . import unlabeled_registration_score as prior
from .scoring import _truth


RUN = core.RUN
SCHEMA = 'pamf_phase3_star_relative_scores_v1'
STATUS = 'STAR_RELATIVE_SCORES_COMPLETE'
STATES = prior.STATES
METHODS = core.METHODS
PRIMARY = core.PRIMARY
SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _prediction_gate(root):
    from .star_relative_target import _predictions
    return _predictions(root)


def _mean(rows, key, complete):
    if not rows or (complete and any(row['count'] == 0 for row in rows)):
        return None
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def _rejection(y, raw, labels, complete):
    rows = []
    for label in labels:
        mask = y == label; count = int(mask.sum()); rejected = int(np.sum(mask & (raw == -1)))
        rows.append(dict(label=int(label), count=count, rejected=rejected, rejection=rejected/count if count else None))
    return _mean(rows, 'rejection', complete), rows


def _coverage(y, unknowns, arrivals, later, registered):
    present = set(int(label) for label in np.unique(y))
    def covered(labels): return bool(labels) and set(labels).issubset(present)
    old, unknown, new, future = (covered(labels) for labels in (list(range(6)), unknowns, arrivals, later))
    return dict(old_classes=old, unknown_classes=unknown, new_arrivals=new, planned_later=future,
        intended_registered=old and new, complete=old and unknown and (new and future if registered else True),
        missing_old_labels=sorted(set(range(6))-present), missing_unknown_labels=sorted(set(unknowns)-present),
        missing_arrival_labels=sorted(set(arrivals)-present), missing_later_labels=sorted(set(later)-present),
        unknown_roster_nonempty=bool(unknowns), new_arrival_roster_nonempty=bool(arrivals), later_roster_nonempty=bool(later))


def _metrics(y, raw, mapped, baseline, q, registered, unknowns, arrivals, later, actual_unregistered, *, complete=True):
    present = set(int(label) for label in np.unique(y))
    def labels(roster): return list(roster) if complete else sorted(present & set(roster))
    old = prior._class_rows(y, raw, mapped, baseline, labels(range(6)), missing_zero=False)
    new = prior._class_rows(y, raw, mapped, baseline, labels(arrivals), missing_zero=False) if registered else []
    intended = prior._class_rows(y, raw, mapped, baseline, labels(list(range(6))+arrivals), missing_zero=False) if registered else []
    old_open, old_accept, old_frr = (_mean(old, key, complete) for key in ('open_accuracy', 'acceptance', 'frr'))
    original = _mean(old, 'original_F0_closed_accuracy', complete)
    new_open = _mean(new, 'open_accuracy', complete) if registered else None
    harmonic = (None if not registered or old_open is None or new_open is None else
                0. if old_open + new_open == 0 else 2*old_open*new_open/(old_open+new_open))
    unknown_rejection, unknown_rows = _rejection(y, raw, labels(unknowns), complete)
    planned, later_rows = _rejection(y, raw, labels(later), complete)
    actual, actual_rows = _rejection(y, raw, labels(actual_unregistered), complete)
    coverage = _coverage(y, unknowns, arrivals, later, registered)
    if not complete: coverage['complete'] = False  # TX-only estimates cannot certify the full roster.
    old_mask = y < 6; new_mask = np.isin(y, arrivals)
    result = dict(count=len(y), coverage=coverage, macro_scope='complete required class roster' if complete else 'classes present in this TX slice',
        old_count=int(old_mask.sum()), old_accepted=int(np.sum(old_mask & (raw != -1))), old_rejected=int(np.sum(old_mask & (raw == -1))),
        old_correct_accepted=int(np.sum(old_mask & (mapped == y))), original_F0_closed_correct=int(np.sum(old_mask & (baseline == y))),
        old_macro_acceptance=old_accept, old_macro_frr=old_frr, old_macro_open_accuracy=old_open,
        original_F0_macro_closed_accuracy=original, original_F0_minus_old_open=None if original is None or old_open is None else original-old_open,
        old_per_class=old, old_class_coverage=coverage['old_classes'],
        new_macro_open_accuracy=new_open, new_macro_denominator=len(labels(arrivals)) if registered else None,
        global_true_arrival_denominator=len(arrivals), new_arrival_count=int(new_mask.sum()),
        new_correct_accepted=int(np.sum(new_mask & (mapped == y))) if registered else None, new_per_class=new if registered else None, H=harmonic,
        intended_registered_macro_frr=_mean(intended, 'frr', complete) if registered else None,
        intended_registered_macro_acceptance=_mean(intended, 'acceptance', complete) if registered else None,
        intended_registered_class_count=len(intended) if registered else None, intended_registered_per_class=intended if registered else None,
        planned_later_macro_rejection=planned, planned_later_per_class=later_rows,
        actual_unregistered_macro_rejection=actual, actual_unregistered_per_class=actual_rows,
        unmatched_anonymous_accepted=int(np.sum(mapped == -2)), all_future_nonold_macro_rejection=unknown_rejection,
        unknown_macro_rejection=unknown_rejection if not registered else None, unknown_per_class=unknown_rows if not registered else None,
        known_macro_acceptance=old_accept, known_macro_frr=old_frr, known_macro_open_accuracy=old_open,
        original_closed_minus_new_open=None if original is None or old_open is None else original-old_open)
    if not registered and coverage['old_classes'] and coverage['unknown_classes']:
        rank = transfer._metrics(y, baseline, baseline, q, raw != -1)
        result.update(auroc_known=rank['auroc_known'], oscr=rank['oscr'])
    else: result.update(auroc_known=None, oscr=None)
    return result


def _old_ok(row):
    return (row['coverage']['complete'] and row['old_macro_acceptance'] is not None and row['old_macro_acceptance'] >= .95
            and row['original_F0_minus_old_open'] is not None and row['original_F0_minus_old_open'] <= .02+1e-12)


def _reg0_checks(row, minimum):
    return dict(complete_required_class_coverage=row['coverage']['complete'], old_known_and_F0_protection=_old_ok(row),
        unknown_protection=row['coverage']['complete'] and row['unknown_macro_rejection'] is not None and row['unknown_macro_rejection'] >= minimum)


def _reg0_goals(row, observed, minimum):
    overall = _reg0_checks(row['overall'], minimum)
    scenes = {s: _reg0_checks(row['by_scenario'][s], minimum) for s in SCENES}
    coverage = observed == set(SCENES)
    return dict(overall=overall, by_scenario=scenes, main_scene_coverage=coverage,
                passed=coverage and all(overall.values()) and all(all(checks.values()) for checks in scenes.values()))


def _registration_goals(row, control, observed, minimum):
    def checks(current, previous):
        covered = current['coverage']['complete'] and previous['coverage']['complete']
        loss = current['old_loss_from_same_DA_REG0']
        return dict(complete_required_class_coverage=covered,
            old_known_and_absolute_F0_protection=covered and _old_ok(current),
            old_retention_from_same_DA_REG0=covered and loss is not None and loss <= .02+1e-12,
            intended_registered_macro_FRR=covered and current['intended_registered_macro_frr'] is not None and current['intended_registered_macro_frr'] <= .05+1e-12,
            planned_later_unknown_protection=covered and current['planned_later_macro_rejection'] is not None and current['planned_later_macro_rejection'] >= minimum)
    overall = checks(row['overall'], control['overall'])
    new = row['overall']['new_macro_open_accuracy']
    overall['new_arrival_macro_accuracy_80'] = row['overall']['coverage']['complete'] and new is not None and new >= .8
    scenes = {s: checks(row['by_scenario'][s], control['by_scenario'][s]) for s in SCENES}
    coverage = observed == set(SCENES)
    return dict(overall=overall, by_scenario=scenes, main_scene_coverage=coverage,
        new_accuracy_each_scene_reported_not_required=True,
        passed=coverage and all(overall.values()) and all(all(checks.values()) for checks in scenes.values()))


def _old_loss(current, previous):
    a, b = current['old_macro_open_accuracy'], previous['old_macro_open_accuracy']
    return None if a is None or b is None else b-a


def score(predictions_root, U_truth, Q_truth, targets, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('A new immutable star-relative score root is required')
    done, p, diag, _ = _prediction_gate(predictions_root)
    if (done.get('run_id') != RUN or done.get('condition') not in ('P0', 'P1')
            or done.get('view_family') != 'practical_cross_receiver' or done.get('all_new_IQ_satellite') is not True
            or done.get('encoder_models') != 1 or done.get('auxiliary_E51_read') is not False):
        raise ValueError('Only all-star P0/P1 with one frozen Phase1 F0 may use this scorer')
    if set(p['ids'].tolist()) & set(p['U_ids'].tolist()) or not np.array_equal(diag['ids'], p['U_ids']):
        raise ValueError('U and Q physical IDs must be disjoint and U diagnostics aligned')
    spec, minimum = transfer._targets(targets, 'practical_cross_receiver')
    if (spec.get('anonymous_registration', {}).get('future_independent_new_class_macro_accuracy_min') != .8
            or spec['anonymous_registration'].get('old_class_drop_from_pre_registration_max') != .02
            or spec.get('unlabeled_adaptation', {}).get('known_open_accuracy_gain_required') is not True):
        raise ValueError('Current source-frozen new80/old2pp and adaptation-gain targets required')
    u, future = _truth(U_truth, p['U_ids']), _truth(Q_truth, p['ids'])
    if not bool((u['receiver'] == 26).all()) or not bool((future['receiver'] == 26).all()):
        raise ValueError('Only RX24-1 index26 development U/Q truth is allowed')
    uy, y, scenarios = u['y'], future['y'], future['scenario']
    arrivals = sorted(int(label) for label in np.unique(uy) if label >= 6)
    unknowns = sorted(int(label) for label in np.unique(y) if label >= 6)
    later = sorted(set(unknowns)-set(arrivals)); observed = set(scenarios.tolist())
    all_scenes = sorted(set(SCENES) | observed)
    mapping, clusters = {}, {}
    for da in ('DA0', 'DA1'):
        mapping[da], clusters[da] = prior._matching(uy, diag['assignments_'+da], diag['candidate_'+da], done['prototype_counts'][da], arrivals)
    states = {}
    for index, name in enumerate(STATES):
        da = 'DA'+str(index % 2); registered = index >= 2; raw = p['prediction'][index]
        matched = mapping[da] if registered else {}; mapped = prior._mapped(raw, matched) if registered else raw
        unregistered = sorted(set(unknowns)-set(matched.values()))
        def metrics(mask, complete=True):
            return _metrics(y[mask], raw[mask], mapped[mask], p['baseline_closed_prediction'][mask],
                p['knownness'][index % 2, mask, 6], registered, unknowns, arrivals, later, unregistered, complete=complete)
        states[name] = dict(overall=metrics(np.ones(len(y), bool)),
            by_scenario={s: metrics(scenarios == s) for s in all_scenes},
            by_transmitter={str(int(label)): metrics(y == label, False) for label in np.unique(y)},
            registered=registered, matched_arrival_labels=sorted(matched.values()), actual_unregistered_labels=unregistered,
            new_macro_denominator_rule='All true U arrivals; missing Q observations give N/A, undiscovered observed arrivals give zero; REG0 N/A',
            unknown_rejection_rule='Only raw -1 rejects; unmatched anonymous -2 is accepted wrong')
    for da in ('DA0', 'DA1'):
        current, previous = states[da+'_REG1'], states[da+'_REG0']
        current['overall']['old_loss_from_same_DA_REG0'] = _old_loss(current['overall'], previous['overall'])
        for group in ('by_scenario', 'by_transmitter'):
            for key, row in current[group].items(): row['old_loss_from_same_DA_REG0'] = _old_loss(row, previous[group][key])
        previous['goals'] = _reg0_goals(previous, observed, minimum)
    registration_goals = {da: _registration_goals(states[da+'_REG1'], states[da+'_REG0'], observed, minimum) for da in ('DA0', 'DA1')}
    for da in ('DA0', 'DA1'): states[da+'_REG1']['goals'] = registration_goals[da]
    first, second = states['DA0_REG0']['overall'], states['DA1_REG0']['overall']
    gain = None if first['old_macro_open_accuracy'] is None or second['old_macro_open_accuracy'] is None else second['old_macro_open_accuracy']-first['old_macro_open_accuracy']
    checks = dict(nonidentity_bias_applied=done['bias_applied'] is True,
        old_macro_open_accuracy_gain=gain is not None and gain > 0., six_scene_condition_targets=states['DA1_REG0']['goals']['passed'])
    adaptation_goals = dict(checks=checks, old_macro_open_accuracy_gain=gain, passed=all(checks.values()))
    pairs = [('DA1_REG0_minus_DA0_REG0','DA1_REG0','DA0_REG0'),('DA1_REG1_minus_DA0_REG1','DA1_REG1','DA0_REG1'),
             ('DA0_REG1_minus_DA0_REG0','DA0_REG1','DA0_REG0'),('DA1_REG1_minus_DA1_REG0','DA1_REG1','DA1_REG0')]
    deltas = {key: prior._delta(states[a]['overall'], states[b]['overall']) for key, a, b in pairs}
    did = {key: None if deltas['DA1_REG1_minus_DA1_REG0'][key] is None or deltas['DA0_REG1_minus_DA0_REG0'][key] is None else
           deltas['DA1_REG1_minus_DA1_REG0'][key]-deltas['DA0_REG1_minus_DA0_REG0'][key]
           for key in ('old_macro_open_accuracy','old_macro_acceptance','old_macro_frr','new_macro_open_accuracy','H')}
    diagnostic = {}
    for index, da in enumerate(('DA0','DA1')):
        methods = {}
        for j, method in enumerate(METHODS):
            raw = p['diagnostic_reg0_prediction'][index,:,j]
            def metrics(mask, complete=True):
                return _metrics(y[mask], raw[mask], raw[mask], p['baseline_closed_prediction'][mask], p['knownness'][index,mask,j],
                    False, unknowns, arrivals, later, unknowns, complete=complete)
            methods[method] = dict(overall=metrics(np.ones(len(y),bool)),
                by_scenario={s: metrics(scenarios == s) for s in all_scenes},
                by_transmitter={str(int(label)): metrics(y == label, False) for label in np.unique(y)}, diagnostic_only=True)
        diagnostic[da] = methods
    passed = adaptation_goals['passed'] and registration_goals['DA1']['passed']
    result = dict(status=STATUS, schema=SCHEMA, run_id=RUN, condition=done['condition'], view_family='practical_cross_receiver',
        count=len(y), U_count=len(uy), truth_last=True, development_only=True, states=states, cluster_diagnostics=clusters,
        future_unknown_labels=unknowns, U_true_arrival_labels=arrivals, planned_later_labels=later,
        public_old_class_names=list(prior.OLD_NAMES), required_main_scenes=list(SCENES), observed_scenes=sorted(observed),
        missing_group_policy='N/A plus coverage false; no zero filling, resampling or empty-all certification',
        deltas=deltas, difference_in_differences=did, diagnostic_methods_reg0=diagnostic,
        adaptation_goals=adaptation_goals, registration_goals=registration_goals,
        primary_state='DA1_REG1', primary_method=PRIMARY, method_names=list(METHODS), primary_pass=passed,
        condition_unknown_rejection_min=minimum, targets_schema=spec['schema'], all_new_IQ_satellite=True,
        encoder_models=1, auxiliary_E51_read=False, Q_truth_used_for_cluster_mapping=False,
        mapping_fitted_on='Independent U truth only after complete predictions', future_Q_fitting=0,
        target_truth_used_for_prediction=False, algorithm_target_label_inputs=0, source_weights_read=0, source_labels_read=0,
        source_parameters_updated=0, identity_head_updates=0, optimizer_updates=0, formal_target_reads=0,
        target_U_truth_reads_by_predictor=0, source_sigma_role='Frozen current-path source L scores only',
        business_registration_authorized=False, final_phase3_complete=False, U_cluster_metrics_are_future_evidence=False,
        evidence_scope='Development-only anonymous future-Q recognition, not operational identity/ownership or total Phase3 completion',
        next_step='STAR_RELATIVE_DEVELOPMENT_PIPELINE_PASSED' if passed else 'NO_PROMOTION')
    output.mkdir(parents=True, exist_ok=False); _write(output/'scores.json',result)
    _write(output/'complete.json',{key:result[key] for key in ('status','schema','run_id','condition','truth_last','primary_state',
        'primary_method','primary_pass','next_step','final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_RELATIVE_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-root',required=True);parser.add_argument('--U-truth',required=True)
    parser.add_argument('--Q-truth',required=True);parser.add_argument('--targets',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);result=score(args.predictions_root,args.U_truth,args.Q_truth,args.targets,args.output)
    print(json.dumps(dict(status=result['status'],primary_pass=result['primary_pass'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
