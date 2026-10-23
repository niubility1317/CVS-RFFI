"""Same-row accuracy comparisons after all predeclared predictions are fixed."""
import argparse
import statistics
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from .plan import METHODS, SEEDS, VIEWS


def overall(path):
    d = read(path)
    if d.get('truth_last') is not True:
        raise ValueError('Only independent truth-last scores are comparable')
    rows = [r for r in d['results'] if r['dimension'] == 'overall' and r['stratum'] == 'ALL'
            and r['test'] == 'main' and r['readout'] == 'raw']
    if len(rows) != 7 or {r['view'] for r in rows} != set(VIEWS) or any(r['count'] != 168000 for r in rows):
        raise ValueError('Full matched raw seven-view coverage required')
    accuracy = {r['view']: r['accuracy'] * 100. for r in rows}
    accuracy['six_mean'] = statistics.mean(accuracy[v] for v in VIEWS[1:])
    return accuracy


def summarize(rt, output):
    if read(Path(rt['run_root']) / 'state.json')['status'] != 'ALL_SCORED':
        raise ValueError('All six predeclared rows must finish before overall analysis')
    values = {}
    for seed in SEEDS:
        for arm, folder in rt['baseline_scores'][str(seed)].items():
            values[arm, seed] = overall(Path(folder) / 'scores.json')
    for row in rt['rows']:
        c = read(row['score_config'])
        values[row['method'], row['model_seed']] = overall(Path(c['output_root']) / 'scores.json')
    result = []
    for method in ('F20', 'F21') + METHODS:
        for view in (*VIEWS, 'six_mean'):
            x = [values[method, seed][view] for seed in SEEDS]
            result.append(dict(method=method, view=view, mean=statistics.mean(x), sd=statistics.stdev(x)))
    verdicts = []
    for method in METHODS:
        differences = {}
        for arm in ('F20', 'F21'):
            differences[arm] = {}
            for view in ('clean', 'six_mean'):
                delta = [values[method, seed][view] - values[arm, seed][view] for seed in SEEDS]
                differences[arm][view] = dict(mean=statistics.mean(delta), sd=statistics.stdev(delta),
                    per_seed=dict(zip(map(str, SEEDS), delta)), positive_seeds=sum(d > 0 for d in delta))
        passed = all(d['clean']['mean'] >= 1. and d['six_mean']['mean'] > 1.5 for d in differences.values())
        verdicts.append(dict(method=method, goal_met=passed, paired_differences_pp=differences))
    write(output, dict(status='ANALYZED', raw_only=True, model_seeds=list(SEEDS),
        summary=result, verdicts=verdicts, goal_met=any(v['goal_met'] for v in verdicts),
        target_feedback_for_training=False, G_readouts_not_substituted=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    summarize(read(a.config), a.output)
