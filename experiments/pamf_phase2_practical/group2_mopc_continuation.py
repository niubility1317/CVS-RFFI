"""Prepare a saved-B MoPC-only successor; never retry quarantined jobs."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

from .io import read, write, require
from .group2_resume import previous_process_is_live, require_same_method
from .group2_schedule import tasks, closed_prediction, validate_runtime, _queue, _smoke_gpu


def saved_b_jobs(rt, previous, manifest, job_files):
    """Bind only completed two-core jobs to their exact saved B outputs."""
    require(manifest['scope'] == 'ORTHOGONAL_LORA_ONLY_MOPC_PENDING', 'Wrong parent scope')
    require(manifest['truth_read'] is False and manifest['query_fit'] is False, 'Parent accessed truth or fit query')
    expected = {j['key']: j for j in tasks(rt)}
    reused = {j['key'] for j in manifest['reused']}
    deferred = {j['key'] for j in manifest['deferred']}
    cancelled = {j['key'] for j in manifest['cancelled']}
    require(not (reused & deferred or reused & cancelled or deferred & cancelled), 'Parent categories overlap')
    out = []
    seen = set()
    for path in job_files:
        old = read(path)
        key = old['key']
        require(key in expected and key not in seen and key not in reused | deferred | cancelled,
                'Duplicate, unknown or excluded task')
        original = expected[key]
        require(old['profile'] in ('protonet_cda', 'dadda_sda') and
                all(old[field] == original[field] for field in ('profile', 'seed', 'capsule')) and
                [tuple(p) for p in old['points']] == [tuple(p) for p in original['points']] and
                old['methods'] == ['orthogonal', 'lora'] and old.get('save_B_for_mopc') is True and
                Path(old['output']) == previous/'predictions'/key, 'Subset job identity differs')
        closed_prediction(old, len(old['points']) * 2)
        out.append(dict(original, methods=['mopc_hr'], load_B_for_mopc=old['output'], trace_registration=True))
        seen.add(key)
    require(len(out) == manifest['pending_count'] and seen | reused | deferred | cancelled == set(expected),
            'Parent matrix is incomplete')
    return out


def dispatch(rt):
    validate_runtime(rt)
    require(all(int(v) <= 2 for v in rt['gpu_limits'].values()), 'Per-GPU task limit exceeds two')
    previous = Path(rt['previous_run_root'])
    parent_rt = read(rt['previous_runtime_file'])
    require(str(previous) == parent_rt['run_root'] and rt['previous_runtime_file'] == parent_rt['runtime_file'],
            'Parent runtime differs')
    require_same_method(parent_rt, rt)
    state, done = read(previous/'worker_state.json'), read(previous/'subset_predictions_complete.json')
    require(state['status'] == 'COMPLETE' and not state['active'] and not state['failed'] and
            done['status'] == 'SUBSET_FIXED_MOPC_PENDING' and done['truth_read'] is False and
            done['query_fit'] is False, 'Parent is not a completed two-core subset')
    for path in [previous/'owner.json'] + list((previous/'launches').glob('*.json')):
        require(not previous_process_is_live(read(path).get('pid'), parent_rt), 'Parent process still active')
    manifest = read(previous/'resume_manifest.json')
    jobs = saved_b_jobs(rt, previous, manifest, sorted((previous/'jobs').glob('*.json')))
    require(done['completed_pending'] == len(jobs), 'Parent completed count differs')
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'launches', 'jobs'):
        (root/name).mkdir()
    for i, job in enumerate(jobs):
        write(root/'jobs'/f'{i:05d}.json', job)
    write(root/'resume_manifest.json', dict(scope='MOPC_ONLY_FROM_SAVED_B', previous_run_root=str(previous),
        reused=manifest['reused'], deferred=manifest['deferred'], cancelled=manifest['cancelled'],
        pending_count=len(jobs), truth_read=False, query_fit=False))
    write(root/'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv))
    gpu = _smoke_gpu(rt)
    while gpu is None:
        time.sleep(rt['poll_seconds'])
        gpu = _smoke_gpu(rt)
    cmd = [sys.executable, '-u', '-m', __package__+'.group2_method_subset', '--smoke', '--runtime', rt['runtime_file']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'logs'/'smoke.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root/'launches'/'smoke.json', dict(pid=proc.pid, gpu=gpu, cwd=str(Path.cwd()), argv=cmd))
        require(proc.wait() == 0, 'Real-source no-query smoke failed')
    old_claim = previous.parent/'.resume_successors'/(previous.name+'.json')
    require(not old_claim.exists(), 'Parent already has a successor')
    write(previous.parent/'.method_owner_reservations'/(previous.name+'.json'),
          dict(successor_runtime=rt['runtime_file'], successor_root=str(root), scope='MOPC_ONLY_FROM_SAVED_B'))
    _queue(rt, jobs, 'worker')
    for job in jobs:
        closed_prediction(job, len(job['points']))
    write(root/'mopc_predictions_complete.json', dict(status='MOPC_ONLY_FIXED_MERGE_PENDING',
        tasks=len(jobs), truth_read=False, query_fit=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    dispatch(read(parser.parse_args().runtime))
