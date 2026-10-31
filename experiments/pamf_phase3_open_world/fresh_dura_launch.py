"""Frozen fresh-receiver raw/DURA comparison, followed by independent scoring."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from .star_representation_launch import _read, _write, _env, _gpu_free, _job

RUN = '20261031-phase3-fresh-dura-s2026092701-r01'
FIELDS = {'run_id', 'code_commit', 'run_root', 'release_root', 'python', 'launch_owner',
          'allowed_gpus', 'minimum_free_mb', 'max_added_workers_per_gpu', 'eval_config',
          'source_novel_truth', 'clean_truth', 'star_truth'}


def validate(r):
    from .fresh_dura_run import validate_config
    if (not isinstance(r, dict) or set(r) != FIELDS or r.get('run_id') != RUN
            or any(not isinstance(r.get(k), str) or not r[k].strip() for k in
                   ('code_commit', 'run_root', 'release_root', 'python', 'launch_owner',
                    'source_novel_truth', 'clean_truth', 'star_truth'))
            or re.fullmatch('[0-9a-f]{40}', r['code_commit']) is None
            or r['allowed_gpus'] != list(range(8)) or r['minimum_free_mb'] != 4096
            or r['max_added_workers_per_gpu'] != 2):
        raise ValueError('Exact fresh-receiver frozen-comparison runtime required')
    c = validate_config(r['eval_config'])
    if (c['run_id'] != RUN or c['code_commit'] != r['code_commit']
            or Path(c['output_root']) != Path(r['run_root']) / 'predict'):
        raise ValueError('Comparison identity and immutable output must agree')
    return r


def start(runtime):
    r = validate(_read(runtime))
    root = Path(r['run_root'])
    if root.exists():
        raise FileExistsError('Observe existing run; never duplicate or replace')
    root.mkdir()
    (root / 'logs').mkdir()
    (root / 'configs').mkdir()
    _write(root / 'configs/eval.json', r['eval_config'])
    argv = [r['python'], '-B', '-u', '-m', __package__ + '.fresh_dura_launch',
            'driver', '--runtime', str(runtime)]
    with (root / 'logs/driver.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=r['release_root'], env=_env(r),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(run_id=RUN, code_commit=r['code_commit'], pid=child.pid, argv=argv,
                  cwd=r['release_root'], launch_owner=r['launch_owner'], time=time.time())
    _write(root / 'launch.json', result)
    print(json.dumps(result), flush=True)


def _command(r, stage):
    root = Path(r['run_root'])
    prefix = [r['python'], '-B', '-u', '-m']
    if stage == 'predict':
        return prefix + [__package__ + '.fresh_dura_run', '--config',
                         str(root / 'configs/eval.json'), '--device', 'cuda:0']
    return prefix + [__package__ + '.fresh_dura_score', '--prediction-root', str(root / 'predict'),
        '--source-novel-truth', r['source_novel_truth'], '--clean-truth', r['clean_truth'],
        '--star-truth', r['star_truth'], '--output', str(root / 'score')]


def _terminal(r, stage):
    m = _read(Path(r['run_root']) / stage / 'complete.json')
    expected = dict(run_id=RUN, code_commit=r['code_commit'], final_phase3_complete=False)
    if stage == 'predict':
        expected.update(status='FRESH_DURA_PREDICTIONS_COMPLETE', all_six_paths_complete=True,
            all_DURA_main_conditions_complete=True, all_DURA_controls_reused_exactly=True,
            all_thresholds_frozen_before_source_raw_and_Q=True, private_truth_reads=0,
            target_U_reads=0, U_ID_metadata_reads=0, new_neural_updates=0,
            density_fit_updates=0, threshold_fit_updates=0)
    else:
        expected.update(status='FRESH_DURA_SCORES_COMPLETE', truth_last=True,
            all_six_predictions_before_truth=True, full_goal_status='ACTIVE',
            diagnostic_completed=True, adaptation_success=None, new_target_DA=False)
    if any(m.get(k) != value for k, value in expected.items()):
        raise ValueError('Fresh DURA ' + stage + ' terminal differs')
    if stage == 'score' and (type(m.get('development_gate_passed')) is not bool
                             or m.get('next_step') != 'NO_PROMOTION'):
        raise ValueError('Frozen diagnostic cannot promote a new candidate')
    return m


def driver(runtime):
    r = validate(_read(runtime))
    root = Path(r['run_root'])
    path = root / 'pipeline_state.json'
    if path.exists():
        raise FileExistsError('Observe original pipeline; never restart')
    state = dict(run_id=RUN, code_commit=r['code_commit'], pid=os.getpid(), status='RUNNING',
                 started_at=time.time(), jobs={})
    _write(path, state)
    try:
        for stage in ('predict', 'score'):
            gpu = None
            if stage == 'predict':
                available = _gpu_free(r)
                while not available:
                    state.update(status='WAITING_MEMORY')
                    _write(path, state)
                    time.sleep(20)
                    available = _gpu_free(r)
                gpu = available[0]
            state.update(status='RUNNING_' + stage.upper())
            _write(path, state)
            if not _job(r, state, path, stage, _command(r, stage), gpu):
                raise RuntimeError(stage + ' failed; preserve original output')
            m = _terminal(r, stage)
        state.update(status='ANALYZED_DEVELOPMENT', all_stages_complete=True,
            diagnostic_completed=True, adaptation_success=None, new_target_DA=False,
            development_gate_passed=m['development_gate_passed'], primary='PCD_predicted_ratio',
            next_step=m['next_step'], final_phase3_complete=False, full_goal_status='ACTIVE')
    except Exception as exc:
        state.update(status='FAILED_PRESERVED', error=str(exc), diagnostic_completed=False,
                     final_phase3_complete=False, full_goal_status='ACTIVE')
    state['finished_at'] = time.time()
    _write(path, state)
    print(state['status'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('start', 'driver'))
    p.add_argument('--runtime', required=True)
    args = p.parse_args()
    {'start': start, 'driver': driver}[args.mode](args.runtime)


if __name__ == '__main__':
    main()
