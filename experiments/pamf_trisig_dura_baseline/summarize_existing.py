"""Read completed independent scores only; no IQ, truth or model access."""
import argparse
import json
from pathlib import Path
import statistics

RUN = '20261110-evaluation-trisig-dura-f0-s2026092702-03-m2-r01'
OLD = '20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03'
SEEDS = (2026092701, 2026092702, 2026092703)
TASKS = ('cross_rx', 'cross_both')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def load(project, seed):
    directory = (project / 'runs' / OLD / 'score' if seed == SEEDS[0]
                 else project / 'runs' / RUN / 'scores' / str(seed))
    done = json.loads((directory / 'complete.json').read_text())
    scores = json.loads((directory / 'scores.json').read_text())
    if done['status'] != 'SCORED_COMPLETE' or not scores['truth_last'] or not scores['target_feedback_forbidden']:
        raise ValueError('Completed independent score required')
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    args = parser.parse_args()
    all_results = {seed: load(args.project, seed) for seed in SEEDS}
    output = []
    for view in VIEWS:
        values, repairs, damage = [], [], []
        for seed, scores in all_results.items():
            rows = [r for r in scores['results'] if r['test'] in TASKS and r['readout'] == 'DURA_v1'
                    and r['view'] == view and r['dimension'] == 'overall']
            pairs = [r for r in scores['paired_raw'] if r['test'] in TASKS and r['readout'] == 'DURA_v1'
                     and r['view'] == view and r['dimension'] == 'overall']
            if len(rows) != 2 or sum(r['count'] for r in rows) != 168000 or len(pairs) != 2:
                raise ValueError('Two full original populations required')
            correct = sum(sum(r['confusion'][i][i] for i in range(6)) for r in rows)
            values.append(100.0 * correct / 168000)
            repairs.append(sum(r['repairs'] for r in pairs)); damage.append(sum(r['damage'] for r in pairs))
        output.append(dict(view=view, accuracy_percent=values, mean=statistics.mean(values),
                           sd=statistics.stdev(values), repairs_vs_raw=repairs, damage_vs_raw=damage))
    print(json.dumps(dict(model_seeds=SEEDS, test_count_per_seed=168000, results=output,
                          note='DURA versus same F raw; not method versus PL or candidate'), indent=2))


if __name__ == '__main__':
    main()
