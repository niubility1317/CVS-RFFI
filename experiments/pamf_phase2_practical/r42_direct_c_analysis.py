"""Aggregate completed direct-C scores and pair with frozen native CIL raw scores."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .io import read, require, write
from .r42_direct_c_dispatch import jobs


METRICS = ('H_macro', 'old_macro_accuracy', 'new_macro_accuracy', 'macro_f1', 'class_floor')


def stats(values):
    require(values and all(v[1] is not None for v in values), 'Missing metric in complete result')
    grouped = defaultdict(list)
    for seed, value in values:
        grouped[seed].append(float(value))
    require(len(grouped) == 4, 'Expected four model seeds')
    means = [mean(grouped[key]) for key in sorted(grouped)]
    return mean(value for _, value in values), stdev(means)


def save_csv(path, rows):
    require(rows, 'Empty result table')
    with Path(path).open('x', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rt, native_index, output):
    root = Path(rt['run_root'])
    done = read(root / 'scoring_complete.json')
    require(done['status'] == 'SCORED_COMPLETE' and done['units'] == 84 and done['rows'] == 10752,
            'Direct-C results incomplete')
    native = {}
    files = defaultdict(list)
    with Path(native_index).open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            files[row['metrics_file']].append(row)
    for path, entries in files.items():
        payload = read(path)
        for row in entries:
            key = (row['profile'], row['capsule'], int(row['model_seed']), row['split_id'])
            require(payload['model_seed'] == key[2], 'Wrong native score seed')
            matches = [r for r in payload['rows'] if r['split_id'] == row['split_id']
                       and r['stage'] == 'C' and r['input_path'] == 'P0']
            require(len(matches) == 1, 'Native score row missing or ambiguous')
            require(key not in native, 'Duplicate native score index')
            native[key] = matches[0]['metrics']
    require(len(native) == 3 * 84 * 16, 'Incomplete native CIL score index')
    aggregate = defaultdict(list)
    paired = defaultdict(list)
    units = []
    for task in jobs(rt):
        directory = root / 'scores' / task['key']
        complete = read(directory / 'complete.json')
        payload = read(directory / 'metrics.json')
        require(complete['status'] == 'SCORED_COMPLETE' and complete['rows'] == 128
                and payload['truth_last'] is True and payload['model_seed'] == task['seed'], 'Wrong score closure')
        for row in payload['rows']:
            k, n, arm, path = row['k'], row['new_count'], row['arm'], row['input_path']
            require(row['actual_target_old_labels'] == 0 and row['actual_target_new_labels'] == k*n,
                    'Wrong target label budget')
            entry = dict(capsule=task['capsule'], model_seed=task['seed'], split_id=row['split_id'],
                         K=k, N=n, arm=arm, input_path=path, actual_target_old_labels=0,
                         actual_target_new_labels=k*n)
            for metric in METRICS:
                value = row['metrics'][metric]
                require(value is not None, 'Missing direct-C metric')
                entry[metric] = value
                aggregate[(k, n, arm, path, metric)].append((task['seed'], value))
            units.append(entry)
            for profile in ('mopc_hr', 'orthogonal', 'lora_triplet_knn'):
                key = (profile, task['capsule'], task['seed'], row['split_id'])
                require(key in native, 'No matched native CIL split')
                for metric in METRICS:
                    value = native[key][metric]
                    require(value is not None, 'Missing native metric')
                    paired[(k, n, arm, path, profile, metric)].append((task['seed'], row['metrics'][metric] - value))
    require(len(units) == 10752, 'Wrong direct-C row count')
    summary, comparisons = [], []
    for key, values in sorted(aggregate.items()):
        require(len(values) == 84, 'Incomplete direct-C group')
        k, n, arm, path, metric = key
        avg, sd = stats(values)
        summary.append(dict(K=k, N=n, arm=arm, input_path=path, metric=metric, units=len(values),
                            mean_pct=100*avg, seed_sample_SD_pp=100*sd))
    for key, values in sorted(paired.items()):
        require(len(values) == 84, 'Incomplete native comparison')
        k, n, arm, path, profile, metric = key
        avg, sd = stats(values)
        comparisons.append(dict(K=k, N=n, arm=arm, input_path=path, native_profile=profile,
            native_input='S0Q0', matched_input=path == 'S0Q0', target_label_budget_equivalent=True,
            metric=metric, units=len(values), mean_gain_pp=100*avg, seed_sample_SD_pp=100*sd))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / 'direct_c_summary.csv', summary)
    save_csv(output / 'direct_c_vs_cil.csv', comparisons)
    save_csv(output / 'direct_c_units.csv', units)
    write(output / 'complete.json', dict(status='ANALYSIS_COMPLETE', units=84, rows=len(units),
          native_profiles=3, truth_access=False, exposed_development_evaluation=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--native-index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(read(args.runtime), args.native_index, args.output)


if __name__ == '__main__':
    main()
