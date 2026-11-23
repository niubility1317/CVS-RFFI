"""Audit the three-seed LoRa DURA objective from independent score artifacts."""

import argparse
from math import isclose
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .data import TARGET, VIEWS
from .full_packet_data import LIMIT
from .residual_dura import ROUTE
from .residual_dura_gate_run import MODES


SEEDS = (2026092701, 2026092702, 2026092703)
LOW_URBAN = 'practical_low_urban'
GOAL_COUNTS = dict(clean=64490, practical=307800, low_urban=38700)
SAMPLES_PER_VIEW = len(TARGET) * LIMIT
BASELINES = ('direct', 'coherent_half', 'complex_response_half')


def totals(by_view):
    counts = {view: sum(by_view[view].values()) for view in VIEWS}
    return dict(clean=counts['clean'], practical=sum(counts[v] for v in VIEWS[1:]),
                low_urban=counts[LOW_URBAN])


def pool(rows):
    return {key: sum(row[key] for row in rows) for key in GOAL_COUNTS}


def check_receiver_counts(counts):
    if set(counts) != set(TARGET) or any(type(n) is not int or not 0 <= n <= LIMIT for n in counts.values()):
        raise ValueError('Expected one valid count for each registered target receiver')
    return counts


def read_reference(path):
    artifact = read(path)
    if artifact['status'] != 'SCORED_COMPLETE' or artifact['truth_last'] is not True:
        raise ValueError('The reference must be independently scored from frozen predictions')
    by_mode = {mode: {seed: {} for seed in SEEDS} for mode in BASELINES}
    for row in artifact['by_seed']:
        mode = row['mode']
        if mode not in by_mode:
            continue
        seed, view = row['seed'], row['view']
        if seed not in SEEDS or view not in VIEWS or row['n'] != SAMPLES_PER_VIEW:
            raise ValueError('Unexpected seed, view or population in the historical reference')
        by_view = by_mode[mode][seed]
        if view in by_view:
            raise ValueError('Duplicate reference row')
        counts = check_receiver_counts({r: round(a * LIMIT) for r, a in row['by_rx'].items()})
        if (not all(isclose(row['by_rx'][r] * LIMIT, n, rel_tol=0., abs_tol=1e-7) for r, n in counts.items())
                or not isclose(row['accuracy'] * SAMPLES_PER_VIEW, sum(counts.values()),
                               rel_tol=0., abs_tol=1e-7)):
            raise ValueError('Historical accuracy and per-receiver counts disagree')
        by_view[view] = counts
    if any(set(by_view) != set(VIEWS) for views in by_mode.values() for by_view in views.values()):
        raise ValueError('Missing historical seed or view')
    if pool([totals(x) for x in by_mode['coherent_half'].values()]) != dict(
            clean=64490, practical=301599, low_urban=37596):
        raise ValueError('The historical report does not match the registered half-dose baseline')
    return by_mode


def read_candidate(path, reference):
    item = read(path)
    if (item['status'] != 'SCORED_COMPLETE' or item['truth_last'] is not True or item['practical'] != ROUTE
            or item['seed'] not in SEEDS or item['matrix_arrays'] != len(TARGET) * len(VIEWS) * len(MODES)):
        raise ValueError('Expected a completed, independently scored residual gate matrix')
    seed = item['seed']
    coverage = item['gate_activity']
    if set(coverage) != set(TARGET) or any(set(row) != set(VIEWS) for row in coverage.values()):
        raise ValueError('Candidate misses gate activity for a receiver or view')
    gate_activity = {view: check_receiver_counts({r: coverage[r][view] for r in TARGET}) for view in VIEWS}
    counts = {}
    for mode in (*MODES, 'full_direct', 'full_complex_response_half'):
        views = item['metrics'][mode]['views']
        if set(views) != set(VIEWS):
            raise ValueError('Candidate score artifact omits a registered practical view')
        counts[mode] = {}
        for view, row in views.items():
            by_rx = check_receiver_counts(row['by_rx'])
            if row['n'] != SAMPLES_PER_VIEW or row['correct'] != sum(by_rx.values()):
                raise ValueError('Candidate view and receiver counts disagree')
            counts[mode][view] = by_rx
    if (counts['full_direct'] != reference['direct'][seed]
            or counts['full_complex_response_half'] != reference['complex_response_half'][seed]):
        raise ValueError('Candidate scorer used a different historical full practical control')
    return seed, dict(modes=counts, gate_activity=gate_activity,
                      clean_control_prediction_changes=item['clean_control_prediction_changes'])


def audit(reference_path, candidate_paths):
    reference = read_reference(reference_path)
    candidates = {}
    for path in candidate_paths:
        seed, score = read_candidate(path, reference)
        if seed in candidates:
            raise ValueError(f'Duplicate candidate seed {seed}')
        candidates[seed] = score
    missing = [seed for seed in SEEDS if seed not in candidates]
    observed = pool([totals(x['modes']['dura_gated_half']) for x in candidates.values()])
    history = {mode: pool([totals(by_view) for by_view in rows.values()]) for mode, rows in reference.items()}
    met = not missing and all(observed[key] >= goal for key, goal in GOAL_COUNTS.items())
    status = 'INCOMPLETE' if missing else ('MET_COUNTS' if met else 'BELOW_COUNTS')
    return dict(status=status, required=GOAL_COUNTS, candidate_observed=observed,
                missing_seeds=missing, remaining_correct={key: max(goal - observed[key], 0)
                for key, goal in GOAL_COUNTS.items()}, per_view_per_seed=SAMPLES_PER_VIEW,
                full_practical_baselines=history, residual_practical_seeds=candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', dest='reference_path', required=True, type=Path)
    parser.add_argument('--candidate', dest='candidate_paths', action='append', default=[], type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = vars(parser.parse_args())
    output = args.pop('output')
    write(output, audit(**args))
