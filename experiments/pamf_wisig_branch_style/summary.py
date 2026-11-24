"""Matched metrics and correct/wrong transitions on fixed physical IDs."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.pipeline import done

from .plan import ARMS, VIEWS


def pooled(parts):
    cm = sum(np.asarray(p['confusion'], dtype=np.int64) for p in parts)
    counts = cm.sum(0) + cm.sum(1)
    macro_f1 = np.divide(2 * cm.diagonal(), counts, out=np.zeros(len(cm)), where=counts > 0).mean()
    return dict(count=int(cm.sum()), accuracy_pct=100 * float(cm.trace() / cm.sum()),
                macro_f1_pct=100 * float(macro_f1))


def analyze(runtime, jobs):
    root = Path(runtime['run_root'])
    rows = []
    stratum_rows = []
    paired_rows = []
    aggregate = {}
    for job in jobs:
        if job['kind'] != 'score' or not done(job):
            continue
        arm = job['arm']
        path = Path(job['completion']).with_name('scores.json')
        data = read(path)
        if not data['truth_last'] or not data['target_feedback_forbidden'] or not data['paired_reference']:
            raise ValueError('Missing fixed reference and truth-last evidence')
        all_scores = defaultdict(list)
        for item in data['results']:
            stratum_rows.append(dict(arm=arm, **item))
            if item['dimension'] == 'overall':
                all_scores[item['view']].append(item)
        all_pairs = defaultdict(list)
        for item in data['paired_raw']:
            paired_rows.append(dict(arm=arm, **item))
            if item['dimension'] == 'overall':
                all_pairs[item['view']].append(item)
        per_view = {}
        for view in VIEWS:
            value = pooled(all_scores[view])
            pairs = all_pairs[view]
            if (len(pairs) != 2 or sorted(p['count'] for p in pairs) != [42000, 126000]
                    or value['count'] != 168000):
                raise ValueError('Expected both fixed WiSig Main populations')
            repairs, damage = (sum(p[key] for p in pairs) for key in ('repairs', 'damage'))
            value.update(repairs=repairs, damage=damage, net=repairs-damage,
                         delta_historical_time_pp=100 * (repairs-damage) / value['count'])
            per_view[view] = value
            rows.append(dict(arm=arm, view=view, **value))
        aggregate[arm] = dict(
            clean=per_view['clean'], scenes=per_view,
            six_practical_mean={metric: statistics.mean(per_view[v][metric] for v in VIEWS[1:])
                                for metric in ('accuracy_pct', 'macro_f1_pct', 'delta_historical_time_pp')})
    contrasts = {}
    if 'time' in aggregate and 'none' in aggregate:
        for arm, result in aggregate.items():
            contrasts[arm] = {
                metric: {f'delta_vs_{base}_pp': result[metric]['accuracy_pct'] - aggregate[base][metric]['accuracy_pct']
                         for base in ('time', 'none')} for metric in ('clean', 'six_practical_mean')}
    factorial = {}
    factorial_arms = ('time-frequency-envelope', 'time', 'frequency-envelope', 'none')
    if all(arm in aggregate for arm in factorial_arms):
        for metric in ('clean', 'six_practical_mean'):
            factors = [aggregate[arm][metric]['accuracy_pct'] for arm in factorial_arms]
            factorial[metric] = factors[0] - factors[1] - factors[2] + factors[3]
    output = root / 'analysis'
    output.mkdir(exist_ok=False)
    for name, table in [('pooled', rows), ('strata', stratum_rows), ('paired_transitions', paired_rows)]:
        with (output / (name + '.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    missing = [arm for arm in ARMS if arm not in aggregate]
    write(output / 'summary.json', dict(
        status='PARTIAL' if missing else 'ANALYZED', model_seeds=1, rows=aggregate,
        missing=missing, versus_matched_controls=contrasts, interaction_pp=factorial,
        paired_reference='Original, frozen same-seed temporal MixStyle row',
        verdict_scope='Exploratory on previously evaluated Main split. No automatic multi-seed promotion.'))
