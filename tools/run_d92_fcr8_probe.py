"""Run the fixed support-only joint FCR8/LocalRidge adapter pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time

import run_d92_registration_diagnostic as baseline
from run_d92_branch_support_probe import read, write, launch

ROOT = Path(__file__).resolve().parents[1]
CHANNEL = baseline.CHANNEL
require = baseline.require
PROBE_CONFIG = read(ROOT/'configs/d92_fcr8_frozen_20260930.json')['algorithm']
CONFIG_NAMES = {c: f'configs/d92_fcr8_support_{c}_20260930.json' for c in ('rx3', 'rx1')}
STATUS = 'FCR8_PROBE_COMPLETE'
EXACT_COUNTS = dict(episodes=160, k1_episodes=40, oof_episodes=120,
    proxy_anchor_count=1400, sequence_paths=1760, baseline_head_fit_count=3168,
    fcr_preparation_count=3168, fcr_stage_count=4576,
    diagnostic_fit_count=0)
MAX_COUNTS = dict(trained_fcr_stage_count=936, optimizer_steps=3744,
    inner_objective_evaluation_count=12168, inner_head_fit_count=36504,
    inner_factorization_count=36504, final_head_fit_count=936,
    final_factorization_count=936, teacher_head_fit_count=1728, teacher_factorization_count=1728,
    head_fit_count=42336, factorization_count=42336,
    baseline_factorization_count=3168, derivative_triangular_solve_count=44928)
ACTUAL_DYNAMIC_COUNTS = ('backward_evaluation_count', 'accepted_trial_count',
    'rejected_trial_count', 'trial_count', 'trial_attempt_count', 'optimizer_iterations',
    'initial_inner_head_fit_count', 'initial_inner_factorization_count',
    'teacher_score_evaluation_count', 'teacher_score_physical_count',
    'task_derivative_triangular_solve_count', 'keep_derivative_triangular_solve_count',
    'prepared_distance_evaluation_count', 'fcr_forward_evaluation_count',
    'final_score_evaluation_count', 'final_score_physical_count',
    'latent_svd_count', 'dictionary_physical_evaluation_count')
COUNTERS = tuple(EXACT_COUNTS) + tuple(MAX_COUNTS) + ACTUAL_DYNAMIC_COUNTS


def validate_spec(spec):
    p = spec['probe']
    require(p['algorithm'] == PROBE_CONFIG and p['channel'] == CHANNEL, 'Frozen joint algorithm/channel mismatch')
    require(p['exact_counts'] == EXACT_COUNTS and p['maximum_counts'] == MAX_COUNTS,
        'Frozen joint pilot budget mismatch')
    require(p['expected_head_fits'] == MAX_COUNTS['head_fit_count'], 'Head upper bound mismatch')
    require(p['candidate'] == 'R_FCR8_seq' and p['controls'] == ['R0', 'R_FCR8_reset_init'], 'Joint/control mismatch')
    require(p['interpretation'] == 'support_joint_pilot_no_direct_promotion', 'Pilot interpretation mismatch')
    require(spec['permissions'].get('adapted_state_reuse') ==
        'within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse', 'Sequential inheritance scope mismatch')
    require(all(row.get('lr') is None and row.get('initial_step_size') == 0.125 for row in spec['rows']), 'Actual finite optimizer metadata mismatch')
    for name, co in p['cohorts'].items():
        require(co['evaluation_config'] == str(PurePosixPath(spec['code']['cwd'])/CONFIG_NAMES[name]),
            'Joint config escaped release')
    view = deepcopy(spec)
    view['probe'].update(algorithm=baseline.DIAGNOSTIC_CONFIG, expected_head_fits=3168)
    for name, co in view['probe']['cohorts'].items():
        co['evaluation_config'] = str(PurePosixPath(spec['code']['cwd'])/baseline.CONFIG_NAMES[name])
    baseline.validate_spec(view)


def command(spec, row):
    co = spec['probe']['cohorts'][row['cohort']]
    return [sys.executable, '-u', str(Path(spec['code']['cwd'])/'tools/evaluate_d92_fcr8_probe.py'),
        '--support-features', row['support_features'], '--capsule', co['capsule'],
        '--output', str(Path(row['output_root'])/'probe'), '--config', co['evaluation_config'],
        '--expected-capsule-id', co['capsule_id'], '--expected-checkpoint-sha256', row['expected_checkpoint_sha256'],
        '--expected-model-seed', str(row['seeds']['model'])]


def verify_marker(path, spec, row):
    marker = read(path)
    co = spec['probe']['cohorts'][row['cohort']]
    expected = dict(status=STATUS, capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'], model_seed=row['seeds']['model'],
        algorithm=PROBE_CONFIG, selection=co['selection'], producer_matrix=co['matrix'],
        query_rows_used=0, source_rows_used=0, **{k: v//4 for k, v in EXACT_COUNTS.items()})
    require(all(marker.get(k) == v for k, v in expected.items()), 'Incomplete or incorrectly bound joint pilot')
    for key in COUNTERS + ('query_rows_used', 'source_rows_used'):
        require(type(marker.get(key)) is int and marker[key] >= 0, 'Invalid counter: '+key)
    for key, total in MAX_COUNTS.items():
        require(marker[key] <= total//4, 'Joint pilot budget exceeded: '+key)
    require(marker['optimizer_steps'] <= 4*marker['trained_fcr_stage_count'], 'Finite update budget exceeded')
    require(marker['optimizer_steps'] == marker['accepted_trial_count'], 'Accepted update accounting mismatch')
    require(marker['trial_count'] == marker['accepted_trial_count'] + marker['rejected_trial_count'],
        'Trial outcome accounting mismatch')
    require(marker['trial_attempt_count'] == marker['trial_count'], 'Complete lane has an unfinished trial')
    # Updated-stage count excludes legitimate zero-gradient/zero-update stages.
    # Those stages still evaluate an initial objective and backward, so budget
    # their work against the declared information-stage upper bound, not updates.
    information_stage_limit = MAX_COUNTS['trained_fcr_stage_count']//4
    require(marker['trial_count'] <= 12*information_stage_limit
        and marker['backward_evaluation_count'] <= 4*information_stage_limit
        and marker['optimizer_iterations'] <= 4*information_stage_limit, 'Finite optimizer budget mismatch')
    require(marker['inner_head_fit_count'] <= 3*marker['inner_objective_evaluation_count'], 'Inner head count mismatch')
    require(marker['head_fit_count'] == sum(marker[k] for k in
        ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'teacher_head_fit_count')), 'Head accounting mismatch')
    require(marker['factorization_count'] == sum(marker[k] for k in
        ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'teacher_factorization_count')), 'Factorization accounting mismatch')
    for prefix in ('baseline', 'inner', 'final', 'teacher'):
        require(marker[prefix+'_factorization_count'] <= marker[prefix+'_head_fit_count'], 'Impossible factorization count')
    require(marker['derivative_triangular_solve_count'] <= 4*marker['inner_head_fit_count'], 'Impossible adjoint solve count')
    require(marker['derivative_triangular_solve_count'] == marker['task_derivative_triangular_solve_count']
        + marker['keep_derivative_triangular_solve_count'], 'Two-channel adjoint accounting mismatch')
    require(marker['initial_inner_head_fit_count'] <= marker['inner_head_fit_count']
        and marker['initial_inner_factorization_count'] <= marker['inner_factorization_count'],
        'Prepaid B initial heads exceed inner total')
    require(marker.get('truth_read') is False, 'Query truth access forbidden')
    return marker


def run(spec, commit, launch_fn=launch):
    validate_spec(spec)
    for co in spec['probe']['cohorts'].values():
        require(read(co['evaluation_config']) == dict(algorithm=PROBE_CONFIG,
            producer_matrix=co['matrix'], selection=co['selection']), 'Evaluator config/spec mismatch')
    root = Path(spec['execution']['remote_run_root'])
    root.mkdir(parents=True, exist_ok=False)
    rows = spec['rows']
    state = {r['row_id']: dict(status='PENDING') for r in rows}
    lock = threading.Lock()
    write(root/'startup.json', dict(spec=spec, commit=commit, pid=os.getpid(), argv=sys.argv,
        cpu_lanes=2, blas_threads_per_lane=2, query_access=False, source_sample_access=False,
        checkpoint_loaded=False, gpu_use=False, adapter_training=True, encoder_training=False,
        adapted_state_inherited=True, actual_A=None, maximum_optimizer_steps=3744, started=time.time()))
    def update(row, status, **fields):
        with lock:
            state[row['row_id']] = dict(status=status, updated=time.time(), **fields)
            write(root/'state.json', state)
        print(json.dumps(dict(row=row['row_id'], status=status, **fields)), flush=True)
    def work(row):
        try:
            out = Path(row['output_root'])
            out.mkdir(exist_ok=False)
            update(row, 'TRAINING_ON_SUPPORT')
            launch_fn(command(spec, row), out/'probe.log', Path(spec['code']['cwd']), 'probe')
            marker = verify_marker(out/'probe/probe_complete.json', spec, row)
            update(row, STATUS, **{k: marker[k] for k in COUNTERS})
        except Exception as exc:
            update(row, 'FAILED', error_type=type(exc).__name__, error=str(exc))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work, rows))
    complete = all(v['status'] == STATUS for v in state.values())
    write(root/'complete.json', dict(status=STATUS if complete else 'FAILED', commit=commit,
        model_rows=4, completed_rows=sum(v['status'] == STATUS for v in state.values()),
        **{k: sum(v.get(k, 0) for v in state.values()) for k in COUNTERS},
        query_access=False, source_sample_access=False, checkpoint_loaded=False, gpu_use=False,
        finished=time.time()))
    if not complete:
        raise RuntimeError('Joint pilot lane failed; healthy lanes retained, no retry')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    run(read(a.spec), a.commit)
