"""One owned source-predict/independent-score chain; no target input route."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

RUN = '20261029-phase3-star-nuisance-s2026092701-r01'
FIELDS = {'run_id', 'code_commit', 'run_root', 'release_root', 'python', 'launch_owner',
          'allowed_gpus', 'minimum_free_mb', 'max_added_workers_per_gpu', 'source_config', 'novel_truth'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def validate(runtime):
    from .star_nuisance_source import validate_config
    if (not isinstance(runtime, dict) or set(runtime) != FIELDS or runtime.get('run_id') != RUN
            or runtime.get('max_added_workers_per_gpu') != 2 or runtime.get('minimum_free_mb') != 4096
            or runtime.get('allowed_gpus') != list(range(8))
            or any(not isinstance(runtime[k], str) or not runtime[k]
                   for k in ('run_root', 'release_root', 'python', 'launch_owner', 'novel_truth'))):
        raise ValueError('Only the fixed owned source-only launch runtime is supported')
    validate_config(runtime['source_config'])
    if (runtime['source_config']['run_id'] != RUN or runtime['source_config']['code_commit'] != runtime['code_commit']
            or Path(runtime['source_config']['output_root']) != Path(runtime['run_root'])/'prediction'):
        raise ValueError('Owned source prediction output and code identity differ')
    return runtime


def start(runtime_path):
    r = validate(_read(runtime_path)); root = Path(r['run_root'])
    if root.exists():
        raise FileExistsError('Observe the original run; never relaunch into an existing root')
    root.mkdir(); (root/'logs').mkdir(); (root/'configs').mkdir()
    _write(root/'configs/source.json', r['source_config'])
    command = [r['python'], '-B', '-u', '-m', __package__+'.star_nuisance_launch', 'driver', '--runtime', str(runtime_path)]
    env = dict(os.environ, PYTHONPATH=r['release_root'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'logs/driver.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=r['release_root'], env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(run_id=RUN, pid=child.pid, argv=command, cwd=r['release_root'], time=time.time(),
                  code_commit=r['code_commit'], launch_owner=r['launch_owner'])
    _write(root/'launch.json', record)
    print(json.dumps(record), flush=True)


def driver(runtime_path):
    r = validate(_read(runtime_path)); root = Path(r['run_root'])
    state = dict(run_id=RUN, code_commit=r['code_commit'], pid=os.getpid(), started_at=time.time(), status='RUNNING', jobs={})
    def save(): _write(root/'pipeline_state.json', state)
    save()
    env = dict(os.environ, PYTHONPATH=r['release_root'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        rows = [tuple(map(int, v.split(','))) for v in raw.splitlines()]
        free = [(m, g) for g, m in rows if g in r['allowed_gpus'] and m >= r['minimum_free_mb']]
        if not free:
            raise RuntimeError('No card currently has the preregistered actual free memory')
        gpu = max(free)[1]
        commands = [
            ('source_predict', [r['python'], '-B', '-u', '-m', __package__+'.star_nuisance_source', 'run',
                                '--config', str(root/'configs/source.json'), '--device', 'cuda:0'], gpu),
            ('source_score', [r['python'], '-B', '-u', '-m', __package__+'.star_nuisance_score',
                              '--source-root', str(root/'prediction'), '--novel-truth', r['novel_truth'], '--output', str(root/'score')], None)]
        for name, command, device in commands:
            jobenv = dict(env, CUDA_VISIBLE_DEVICES='' if device is None else str(device))
            job = dict(status='RUNNING', started_at=time.time(), argv=command, gpu=device)
            with (root/'logs'/f'{name}.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=r['release_root'], env=jobenv,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = child.pid; state['jobs'][name] = job; save()
                job['returncode'] = child.wait()
            job.update(finished_at=time.time(), status='COMPLETE' if job['returncode'] == 0 else 'FAILED'); save()
            if job['returncode']:
                raise RuntimeError(name+' failed; preserve partial outputs without restart')
        state['status'] = 'ANALYZED_SOURCE_ONLY'
    except Exception as exc:
        state.update(status='FAILED_PRESERVED', error=str(exc))
    state['finished_at'] = time.time(); save()
    print(state['status'], flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['start', 'driver']); p.add_argument('--runtime', required=True)
    a = p.parse_args(); return dict(start=start, driver=driver)[a.mode](a.runtime)


if __name__ == '__main__':
    main()
