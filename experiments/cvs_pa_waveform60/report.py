"""Only source decides extension; all fixed rows retain their seven-view test."""
import csv
from pathlib import Path
import numpy as np
from . import design as d

def source_decision():
    rows = {r['arm']: r for r in d.rows()}
    no_root = d.BASE / rows['NOAS_PAOFF']['row_id']
    as_root = d.BASE / rows['AS_PAOFF']['row_id']
    reference = Path(d.PROJECT) / 'runs' / d.REFERENCE_RUN
    paths = [no_root / 'source/completion.json', as_root / 'source/completion.json',
             as_root / 'source_qualification/complete.json',
             reference / 'B01-s2026092701/source_qualification/complete.json']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        result = dict(status='SOURCE_READOUT_PARTIAL', missing=missing,
                      source_qualified=False, formal_evaluation_always_required=True,
                      formal_scores_read=False, no_automatic_budget_or_seed_extension=True,
                      NN=0, target_feedback=False)
        d.write(d.BASE / 'source_matrix_decision.json', result)
        return result
    no, yes, q, old = [d.read(p) for p in paths]
    if (no['config'] != d.config(rows['NOAS_PAOFF']) or yes['config'] != d.config(rows['AS_PAOFF'])
            or q['config'] != d.config(rows['AS_PAOFF']) or old['config'] != d.old_config('AS_PAOFF')
            or no['target_access'] or yes['target_access'] or q['target_access'] or old['target_access']
            or q['formal_scores_read'] or old['formal_scores_read'] or q['training_updates']
            or q['NN'] != 12600 or no['final_source_metrics']['source_val_count'] != 27000
            or yes['final_source_metrics']['source_val_count'] != 27000):
        raise ValueError('Only exact frozen source-only readouts may decide extension')
    now, previous = np.asarray(q['source_accuracies'], float), np.asarray(old['source_accuracies'], float)
    if now.shape != (2, 3) or previous.shape != (2, 3) or not np.isfinite(now).all() or not np.isfinite(previous).all():
        raise ValueError('Complete fixed source two-draw/three-scene readout required')
    clean = float(100 * (yes['final_source_metrics']['source_val_accuracy'] - no['final_source_metrics']['source_val_accuracy']))
    midlow = float(100 * (now[:, 1:].mean() - previous[:, 1:].mean()))
    low = float(100 * (now[:, 2].mean() - previous[:, 2].mean()))
    checks = dict(fullV_AS_clean_within_new_NOAS_0p1pp=bool(clean >= -.1 - 1e-12),
                  mid_low_mean_gain_at_least_0p5pp=bool(midlow >= .5 - 1e-12),
                  low_not_below_originalAs30=bool(low >= -1e-12))
    result = dict(status='SOURCE_READOUT_DECIDED', checks=checks,
                  source_qualified=bool(all(checks.values())),
                  fullV_AS_clean_vs_new_NOAS_pp=clean, mid_low_mean_vs_originalAs30_pp=midlow,
                  low_vs_originalAs30_pp=low, source_accuracies=now.tolist(),
                  originalAs30_source_accuracies=previous.tolist(),
                  formal_scores_read=False, formal_evaluation_always_required=True,
                  no_automatic_budget_or_seed_extension=True, target_feedback=False,
                  model_seed_count=1, NN=0)
    d.write(d.BASE / 'source_matrix_decision.json', result)
    return result

def collect():
    records, values, cost, missing = [], {}, {}, []
    refs = [(r['arm'], d.BASE / r['row_id'], False) for r in d.rows()]
    refs += [('AS30', Path(d.PROJECT) / 'runs' / d.REFERENCE_RUN / 'B01-s2026092701', True),
             ('NOAS30', Path(d.PROJECT) / 'runs' / d.REFERENCE_RUN / 'B00-s2026092701', True)]
    for label, root, reused in refs:
        absent = [str(root / p) for p in ('prediction/complete.json', 'scoring/complete.json')
                  if not (root / p).is_file()]
        if absent and not reused:
            missing.append(dict(arm=label, artifacts=absent))
            continue
        prediction, scoring = d.read(root / 'prediction/complete.json'), d.read(root / 'scoring/complete.json')
        if (prediction['status'] != 'PREDICTIONS_COMPLETE' or prediction['views'] != list(d.VIEWS)
                or prediction['truth_read'] or scoring['status'] != 'SCORED_COMPLETE'
                or scoring['views'] != list(d.VIEWS) or not scoring['truth_last']):
            raise ValueError('All seven immutable predictions and independent truth-last scores required')
        rs = d.read(root / 'scoring/scores.json')['results']
        values[label] = {r['view']: r for r in rs if r['dimension'] == 'overall'}
        if set(values[label]) != set(d.VIEWS):
            raise ValueError('Seven actual view results required')
        records.extend(dict(r, comparison_arm=label, reused=reused) for r in rs)
        if not reused:
            c = d.config(next(r for r in d.rows() if r['arm'] == label))
            cost[label] = dict(new10=d.read(root / 'source/result.json'), inherited20=d.read(Path(c['parent_source']) / 'result.json'))
    with (d.BASE / 'per_row_seed.csv').open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['comparison_arm', 'reused', 'row_id', 'model_seed', 'view', 'dimension', 'stratum', 'accuracy', 'macro_f1'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    source = d.read(d.BASE / 'source_matrix_decision.json')
    status = ('FORMAL_PARTIAL' if missing else 'SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS'
              if source['status'] == 'SOURCE_READOUT_DECIDED' else 'FORMAL_COMPLETE_SOURCE_PARTIAL')
    result = dict(status=status, missing_rows=missing,
                  views=values, cost=cost, source_decision=source,
                  reference_formal_artifacts_reused=True, reference_reprediction=0, reference_rescoring=0,
                  new_model_rows=2, new_epochs_per_row=10, parent_epochs_per_row=20, cumulative_epochs_per_row=30,
                  new_updates_per_row=2220, parent_updates_per_row=4440, cumulative_updates_per_row=6660,
                  model_seed_count=1, source_FAIL_does_not_skip_formal=True,
                  formal_feedback=False, crossseed_confirmed=False, goal_complete=False)
    d.write(d.BASE / 'results.json', result)
    return result
