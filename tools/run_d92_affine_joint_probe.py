"""Run the frozen AffineJL residual-function joint support pilot in two CPU lanes."""
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
PROBE_CONFIG = read(ROOT/'configs/d92_affine_joint_frozen_20261001.json')
CONFIG_NAMES = {c: f'configs/d92_affine_joint_support_{c}_20261001.json' for c in ('rx3', 'rx1')}
STATUS = 'AFFINE_JOINT_PROBE_COMPLETE'
EXACT_COUNTS = dict(episodes=160, k1_episodes=40, oof_episodes=120,
    proxy_anchor_count=1400, sequence_paths=1800, baseline_head_fit_count=3240,
    ajlr_preparation_count=3240, ajlr_stage_count=3240, diagnostic_fit_count=0)
MAX_COUNTS = dict(trained_ajlr_stage_count=648, optimizer_steps=2592,
    trial_count=31104, trial_attempt_count=31104,
    inner_objective_evaluation_count=31752, inner_head_fit_count=95256,
    inner_factorization_count=95256, final_head_fit_count=3240,
    final_factorization_count=3240, prior_head_fit_count=864, prior_factorization_count=864,
    baseline_factorization_count=3240, head_fit_count=102600, factorization_count=102600,
    baseline_head_triangular_solve_count=6480,
    baseline_effective_df_triangular_solve_count=6480, baseline_triangular_solve_count=12960,
    ce_adjoint_solve_count=7776, derivative_triangular_solve_count=15552)
ACTUAL_DYNAMIC_COUNTS = (
    'latent_svd_count', 'dictionary_physical_evaluation_count',
    'prepared_distance_evaluation_count', 'original_distance_pair_count',
    'prior_triangular_solve_count', 'prior_score_evaluation_count', 'prior_score_physical_count',
    'reference_distance_evaluation_count', 'reference_distance_pair_count',
    'raw_distance_evaluation_count', 'raw_distance_pair_count',
    'kernel_evaluation_count', 'kernel_pair_count', 'optimizer_iterations',
    'head_triangular_solve_count', 'backward_evaluation_count',
    'accepted_trial_count', 'rejected_trial_count', 'ajlr_forward_evaluation_count',
    'adapter_physical_evaluation_count', 'final_score_evaluation_count', 'final_score_physical_count',
    'head_triangular_rhs_count', 'head_triangular_rhs_element_count', 'head_triangular_dense_work_unit_count',
    'derivative_triangular_rhs_count', 'derivative_triangular_rhs_element_count', 'derivative_triangular_dense_work_unit_count',
    'prior_triangular_rhs_count', 'prior_triangular_rhs_element_count', 'prior_triangular_dense_work_unit_count',
    'intercept_fit_count', 'intercept_addition_count', 'prior_intercept_fit_count', 'prior_intercept_addition_count')
COUNTERS = tuple(dict.fromkeys(tuple(EXACT_COUNTS) + tuple(MAX_COUNTS) + ACTUAL_DYNAMIC_COUNTS))


def validate_spec(spec):
    p = spec['probe']
    require(p['algorithm'] == PROBE_CONFIG and p['channel'] == CHANNEL, 'Frozen AffineJL algorithm/channel mismatch')
    require(p['exact_counts'] == EXACT_COUNTS and p['maximum_counts'] == MAX_COUNTS,
        'Frozen AffineJL pilot budget mismatch')
    require(p['expected_head_fits'] == MAX_COUNTS['head_fit_count'], 'Head upper bound mismatch')
    require(p['expected_sequence_paths'] == EXACT_COUNTS['sequence_paths'], 'Actual AffineJL path count mismatch')
    require(p['candidate'] == 'R_AFFINE_seq' and p['controls'] == ['R0'], 'AffineJL/control mismatch')
    require(p['interpretation'] == 'support_joint_pilot_no_direct_promotion', 'Pilot interpretation mismatch')
    require(spec['permissions'].get('adapted_state_reuse') ==
        'within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse', 'Sequential inheritance scope mismatch')
    require(all(row.get('lr') is None and row.get('initial_step_size') == .125 for row in spec['rows']),
        'Finite optimizer metadata mismatch')
    for name, co in p['cohorts'].items():
        require(co['evaluation_config'] == str(PurePosixPath(spec['code']['cwd'])/CONFIG_NAMES[name]),
            'AffineJL config escaped release')
    # The old diagnostic validator only checks the unchanged cache and selection
    # contract. Its 3168 is a compatibility view, never the actual AffineJL workload.
    view = deepcopy(spec)
    view['probe'].update(algorithm=baseline.DIAGNOSTIC_CONFIG, expected_head_fits=3168,
        expected_sequence_paths=1760)
    for name, co in view['probe']['cohorts'].items():
        co['evaluation_config'] = str(PurePosixPath(spec['code']['cwd'])/baseline.CONFIG_NAMES[name])
    baseline.validate_spec(view)


def command(spec, row):
    co = spec['probe']['cohorts'][row['cohort']]
    return [sys.executable, '-u', str(Path(spec['code']['cwd'])/'tools/evaluate_d92_affine_joint_probe.py'),
        '--support-features', row['support_features'], '--capsule', co['capsule'],
        '--output', str(Path(row['output_root'])/'probe'), '--config', co['evaluation_config'],
        '--expected-capsule-id', co['capsule_id'], '--expected-checkpoint-sha256', row['expected_checkpoint_sha256'],
        '--expected-model-seed', str(row['seeds']['model']), '--run-id', spec['run_id'], '--row-id', row['row_id']]


def verify_marker(path, spec, row):
    marker = read(path)
    co = spec['probe']['cohorts'][row['cohort']]
    expected = dict(status=STATUS, run_id=spec['run_id'], row_id=row['row_id'], capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'], model_seed=row['seeds']['model'],
        algorithm=PROBE_CONFIG, selection=co['selection'], producer_matrix=co['matrix'],
        query_rows_used=0, source_rows_used=0, **{k: v//4 for k, v in EXACT_COUNTS.items()})
    require(all(marker.get(k) == v for k, v in expected.items()), 'Incomplete or incorrectly bound AffineJL pilot')
    for key in COUNTERS + ('query_rows_used', 'source_rows_used'):
        require(type(marker.get(key)) is int and marker[key] >= 0, 'Invalid counter: '+key)
    for key, total in MAX_COUNTS.items():
        require(marker[key] <= total//4, 'AffineJL pilot budget exceeded: '+key)
    information_stage_limit = MAX_COUNTS['trained_ajlr_stage_count']//4
    require(marker['optimizer_steps'] <= 4*marker['trained_ajlr_stage_count'], 'Finite update budget exceeded')
    require(marker['optimizer_steps'] == marker['accepted_trial_count'], 'Accepted update accounting mismatch')
    require(marker['trial_count'] == marker['accepted_trial_count'] + marker['rejected_trial_count'],
        'Trial outcome accounting mismatch')
    require(marker['trial_attempt_count'] == marker['trial_count'], 'Complete lane has an unfinished trial')
    require(marker['trial_count'] <= 48*information_stage_limit
        and marker['backward_evaluation_count'] <= 4*information_stage_limit
        and marker['optimizer_iterations'] <= 4*information_stage_limit, 'Finite optimizer budget mismatch')
    require(marker['inner_head_fit_count'] <= 3*marker['inner_objective_evaluation_count'], 'Inner head count mismatch')
    require(marker['head_fit_count'] == sum(marker[k] for k in
        ('baseline_head_fit_count', 'inner_head_fit_count', 'final_head_fit_count', 'prior_head_fit_count')),
        'Head accounting mismatch')
    require(marker['factorization_count'] == sum(marker[k] for k in
        ('baseline_factorization_count', 'inner_factorization_count', 'final_factorization_count', 'prior_factorization_count')),
        'Factorization accounting mismatch')
    for prefix in ('baseline', 'inner', 'final', 'prior'):
        require(marker[prefix+'_factorization_count'] <= marker[prefix+'_head_fit_count'], 'Impossible factorization count')
    require(marker['head_triangular_solve_count'] == 2*(marker['inner_factorization_count']+marker['final_factorization_count']),
        'Student primal triangular solve accounting mismatch')
    require(marker['prior_triangular_solve_count'] == 2*marker['prior_factorization_count'], 'Prior primal accounting mismatch')
    require(marker['baseline_head_triangular_solve_count'] == 2*marker['baseline_factorization_count'],
        'Baseline primal accounting mismatch')
    require(marker['baseline_effective_df_triangular_solve_count'] <= 2*marker['baseline_factorization_count'],
        'Baseline EDF accounting mismatch')
    require(marker['baseline_triangular_solve_count'] == marker['baseline_head_triangular_solve_count']
        + marker['baseline_effective_df_triangular_solve_count'], 'Baseline triangular accounting mismatch')
    require(marker['derivative_triangular_solve_count'] == 2*marker['ce_adjoint_solve_count'],
        'CE adjoint accounting mismatch')
    require(marker['ce_adjoint_solve_count'] <= 3*marker['backward_evaluation_count'], 'Impossible CE adjoint count')
    require(marker['ajlr_forward_evaluation_count'] == marker['inner_head_fit_count']+marker['final_head_fit_count'],
        'Student forward accounting mismatch')
    # RHS sums count every actual triangular call, including rejected trials.
    for prefix, minimum_width, maximum_width, maximum_n in (('head', 7, 27, 520),
            ('derivative', 6, 26, 520), ('prior', 7, 7, 120)):
        calls = marker[prefix+'_triangular_solve_count']
        rhs = marker[prefix+'_triangular_rhs_count']
        elements = marker[prefix+'_triangular_rhs_element_count']
        dense = marker[prefix+'_triangular_dense_work_unit_count']
        require(minimum_width*calls <= rhs <= maximum_width*calls, 'RHS width accounting mismatch: '+prefix)
        require(6*rhs <= elements <= maximum_n*rhs, 'RHS element accounting mismatch: '+prefix)
        require(6*elements <= dense <= maximum_n*elements, 'Dense work accounting mismatch: '+prefix)
        require(elements*elements <= rhs*dense, 'Inconsistent weighted solve dimensions: '+prefix)
    require(marker['intercept_fit_count'] == marker['ajlr_forward_evaluation_count'], 'Student intercept fit accounting mismatch')
    require(marker['prior_intercept_fit_count'] == marker['prior_head_fit_count'], 'Prior intercept fit accounting mismatch')
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
        adapted_state_inherited=True, actual_A=None,
        maximum_optimizer_steps=MAX_COUNTS['optimizer_steps'], started=time.time()))
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
        raise RuntimeError('AffineJL pilot lane failed; healthy lanes retained, no retry')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    run(read(a.spec), a.commit)
