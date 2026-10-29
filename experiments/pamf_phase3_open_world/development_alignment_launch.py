"""Single-owner detached driver for two independent unlabeled alignment rows."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .launch import choose_gpu, read, write


CONDITIONS = ('C0', 'P1')
FIELDS = {'run_id', 'run_root', 'release_root', 'python', 'code_commit', 'launch_owner',
          'allowed_gpus', 'max_added_workers_per_gpu', 'minimum_free_mb', 'targets_path', 'rows'}


def validate_runtime(rt):
    if not isinstance(rt, dict) or set(rt) != FIELDS:
        raise ValueError('Only the fixed two-condition alignment runtime is accepted')
    for key in FIELDS - {'rows', 'allowed_gpus', 'max_added_workers_per_gpu', 'minimum_free_mb'}:
        if not isinstance(rt[key], str) or not rt[key].strip():
            raise ValueError('Nonempty runtime identity/path required: ' + key)
    if (len(rt['code_commit']) != 40 or any(x not in '0123456789abcdef' for x in rt['code_commit'])
            or rt['max_added_workers_per_gpu'] != 2 or type(rt['max_added_workers_per_gpu']) is not int
            or type(rt['minimum_free_mb']) is not int or rt['minimum_free_mb'] < 4096
            or not isinstance(rt['allowed_gpus'], list) or not rt['allowed_gpus']
            or any(type(x) is not int or x < 0 for x in rt['allowed_gpus'])
            or len(set(rt['allowed_gpus'])) != len(rt['allowed_gpus'])):
        raise ValueError('Actual commit and existing two-added-worker/free-memory policy required')
    root, release = Path(rt['run_root']), Path(rt['release_root'])
    if (not root.is_absolute() or root.name != rt['run_id'] or not release.is_absolute()
            or not Path(rt['targets_path']).is_absolute()
            or not isinstance(rt['rows'], dict) or set(rt['rows']) != set(CONDITIONS)):
        raise ValueError('Immutable output/release and both C0/P1 rows are required')
    from .development_alignment import validate_config
    for condition in CONDITIONS:
        row = rt['rows'][condition]
        if not isinstance(row, dict) or set(row) != {'predict_config', 'truth_path'}:
            raise ValueError('Only a public predictor configuration and separate scorer truth path are allowed')
        config = validate_config(row['predict_config'])
        if (config['run_id'] != rt['run_id'] or config['code_commit'] != rt['code_commit']
                or config['condition'] != condition
                or Path(config['output_root']).resolve() != (root / 'rows' / condition / 'predict').resolve()
                or not isinstance(row['truth_path'], str) or not Path(row['truth_path']).is_absolute()):
            raise ValueError('Runtime row identity, prediction output or scorer path differs')
    return rt


def commands(rt, condition, config_path):
    row = Path(rt['run_root']) / 'rows' / condition
    prefix = [rt['python'], '-B', '-u', '-m', 'experiments.pamf_phase3_open_world.development_alignment']
    return dict(predict=prefix + ['predict', '--config', str(config_path), '--device', 'cuda:0'],
        score=prefix + ['score', '--predictions-root', str(row / 'predict'),
            '--truth', rt['rows'][condition]['truth_path'], '--targets', rt['targets_path'],
            '--output', str(row / 'score')])


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root'])
    state = dict(status='RUNNING', run_id=rt['run_id'], code_commit=rt['code_commit'],
        owner=rt['launch_owner'], pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        started_at=time.time(), rows={}, foreign_processes_stopped=0,
        max_added_workers_per_gpu=2, max_concurrent_gpu_workers=1,
        formal_target_reads=0, final_phase3_complete=False)
    def record():
        write(root / 'pipeline_state.json', state, exclusive=False)
    record()
    try:
        for condition in CONDITIONS:
            row = root / 'rows' / condition
            row.mkdir(exist_ok=False)
            (row / 'logs').mkdir()
            config_path = root / 'configs' / (condition + '.predict.json')
            write(config_path, rt['rows'][condition]['predict_config'])
            stage_commands = commands(rt, condition, config_path)
            current = dict(status='RUNNING', jobs={})
            state['rows'][condition] = current
            record()
            for kind in ('predict', 'score'):
                gpu = choose_gpu(rt) if kind == 'predict' else None
                env = dict(os.environ, PYTHONPATH=rt['release_root'],
                    CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                log_path = row / 'logs' / (kind + '.log')
                job = dict(status='STARTING', argv=stage_commands[kind], cwd=rt['release_root'],
                    gpu=gpu, log=str(log_path), started_at=time.time())
                current['jobs'][kind] = job
                record()
                with log_path.open('xb') as stream:
                    process = subprocess.Popen(stage_commands[kind], cwd=rt['release_root'], env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    job.update(pid=process.pid, status='RUNNING')
                    record()
                    exitcode = process.wait()
                job.update(returncode=exitcode, finished_at=time.time())
                if exitcode:
                    job['status'] = 'FAILED'
                    raise RuntimeError(condition + ' ' + kind + ' failed; preserve partial and do not retry')
                marker = read(row / ('predict' if kind == 'predict' else 'score') / 'complete.json')
                job.update(status='COMPLETE', completion_status=marker.get('status'))
                record()
            current['status'] = 'COMPLETE'
            record()
        state.update(status='ANALYZED_DEVELOPMENT_ONLY', finished_at=time.time())
    except Exception as exc:
        state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), finished_at=time.time())
    record()
    return state


def launch(rt, runtime_path):
    validate_runtime(rt)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir()
    (root / 'rows').mkdir()
    (root / 'logs').mkdir()
    command = [rt['python'], '-B', '-u', '-m',
        'experiments.pamf_phase3_open_world.development_alignment_launch',
        '--runtime', str(Path(runtime_path).resolve()), '--dispatch']
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with (root / 'logs/driver.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=rt['release_root'], env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(status='LAUNCH_RETURNED_REQUIRES_POST_STATE', run_id=rt['run_id'], pid=process.pid,
        cwd=rt['release_root'], argv=command, owner=rt['launch_owner'], code_commit=rt['code_commit'],
        foreign_processes_stopped=0)
    write(root / 'launch.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--dispatch', action='store_true')
    args = parser.parse_args(argv)
    rt = read(args.runtime)
    result = dispatch(rt) if args.dispatch else launch(rt, args.runtime)
    print(result['status'], flush=True)
    return int(result['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
