"""Same-row accuracy comparisons after all predeclared predictions are fixed."""
import argparse
import math
import statistics
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from .plan import METHODS, SEEDS, VIEWS


def overall(path, expected=None):
    d = read(path)
    if d.get('truth_last') is not True or d.get('status') != 'SCORED':
        raise ValueError('Only independent truth-last scores are comparable')
    if expected and any(d.get(k) != v for k, v in expected.items()):
        raise ValueError('Scores belong to another run/row/model seed')
    rows = [r for r in d['results'] if r['dimension'] == 'overall' and r['stratum'] == 'ALL'
            and r['test'] == 'main' and r['readout'] == 'raw']
    if len(rows) != 7 or {r['view'] for r in rows} != set(VIEWS) or any(r['count'] != 168000 for r in rows):
        raise ValueError('Full matched raw seven-view coverage required')
    correct = {}
    for row in rows:
        cm = row['confusion']
        if (len(cm) != 6 or any(len(r) != 6 for r in cm)
                or any(type(v) is not int or v < 0 for r in cm for v in r)
                or sum(sum(r) for r in cm) != row['count']):
            raise ValueError('Six-class confusion counts do not match full coverage')
        n = sum(cm[i][i] for i in range(6))
        reported = float(row['accuracy'])
        if not math.isfinite(reported) or abs(reported - n / row['count']) > 1e-12:
            raise ValueError('Reported accuracy differs from integer correct counts')
        correct[row['view']] = n
    accuracy = {view: 100. * n / 168000 for view, n in correct.items()}
    accuracy['six_mean'] = 100. * sum(correct[v] for v in VIEWS[1:]) / (6 * 168000)
    accuracy['correct_counts'] = correct
    return accuracy


def counts(value, view):
    if view == 'six_mean':
        return sum(value['correct_counts'][v] for v in VIEWS[1:]), 6 * 168000
    return value['correct_counts'][view], 168000


def declared_methods(rt):
    methods = tuple(rt.get('methods', METHODS))
    if (methods not in (METHODS, ('VIEWANCHOR',)) or len(rt['rows']) != 3 * len(methods)
            or {(r['method'], r['model_seed']) for r in rt['rows']}
            != {(method, seed) for method in methods for seed in SEEDS}):
        raise ValueError('Complete declared method/three-seed rows required exactly once')
    return methods


def summarize(rt, output):
    if read(Path(rt['run_root']) / 'state.json')['status'] != 'ALL_SCORED':
        raise ValueError('All declared target-test rows must finish before overall analysis')
    methods = declared_methods(rt)
    values = {}
    for seed in SEEDS:
        for arm, folder in rt['baseline_scores'][str(seed)].items():
            values[arm, seed] = overall(Path(folder) / 'scores.json',
                dict(model_seed=seed, row_id=arm + '-s' + str(seed)))
    for row in rt['rows']:
        c = read(row['score_config'])
        if c['model_seed'] != row['model_seed'] or c['row_id'] != row['method'] + '-s' + str(row['model_seed']):
            raise ValueError('Score configuration belongs to another method/seed')
        values[row['method'], row['model_seed']] = overall(Path(c['output_root']) / 'scores.json',
            {k: c[k] for k in ('run_id', 'row_id', 'model_seed')})
    result = []
    for method in ('F20', 'F21') + methods:
        for view in (*VIEWS, 'six_mean'):
            x = [values[method, seed][view] for seed in SEEDS]
            n = sum(counts(values[method, seed], view)[0] for seed in SEEDS)
            denominator = 3 * counts(values[method, SEEDS[0]], view)[1]
            result.append(dict(method=method, view=view, mean=100. * n / denominator, sd=statistics.stdev(x)))
    verdicts = []
    for method in methods:
        differences, exact = {}, {}
        for arm in ('F20', 'F21'):
            differences[arm] = {}
            for view in ('clean', 'six_mean'):
                net = [counts(values[method, seed], view)[0] - counts(values[arm, seed], view)[0] for seed in SEEDS]
                denominator = counts(values[method, SEEDS[0]], view)[1]
                delta = [100. * n / denominator for n in net]
                differences[arm][view] = dict(mean=100. * sum(net) / (3 * denominator), sd=statistics.stdev(delta),
                    per_seed=dict(zip(map(str, SEEDS), delta)), positive_seeds=sum(d > 0 for d in delta))
                exact.setdefault(arm, {})[view] = dict(net_correct=sum(net), denominator=3 * denominator)
            clean, six = exact[arm]['clean'], exact[arm]['six_mean']
            clean['passes_at_least_1pp'] = 100 * clean['net_correct'] >= clean['denominator']
            six['passes_strictly_more_than_1_5pp'] = 200 * six['net_correct'] > 3 * six['denominator']
        passed = all(d['clean']['passes_at_least_1pp'] and d['six_mean']['passes_strictly_more_than_1_5pp'] for d in exact.values())
        verdicts.append(dict(method=method, goal_met=passed, paired_differences_pp=differences, integer_goal_evidence=exact))
    write(output, dict(status='ANALYZED', raw_only=True, model_seeds=list(SEEDS),
        summary=result, verdicts=verdicts, goal_met=any(v['goal_met'] for v in verdicts),
        target_feedback_for_training=False, G_readouts_not_substituted=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    summarize(read(a.config), a.output)
