"""Continue unstarted saved-B MoPC tasks from a retained numerical failure."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read, require, write
from .group2_mopc_continuation import saved_b_jobs
from .group2_mopc_retained import partition_retained_mopc
from .group2_resume import previous_process_is_live, require_same_method
from .group2_schedule import validate_runtime, closed_prediction, _queue, _smoke_gpu


def successor_manifest(parent, partition, previous):
    categories = [parent[k] for k in ('reused', 'deferred', 'cancelled')]
    categories += [partition[k] for k in ('completed', 'quarantined', 'pending')]
    keys = [job['key'] for group in categories for job in group]
    require(len(keys) == len(set(keys)), 'Task categories overlap')
    queue = partition['pending']
    require(all(job.get('profile') in ('protonet_cda', 'dadda_sda') and job.get('methods') == ['mopc_hr']
                for job in queue), 'Invalid pending tasks')
    manifest = dict(scope='MOPC_CONTINUATION_WITH_QUARANTINES', previous_run_root=str(previous),
        reused=parent['reused'], deferred=parent['deferred'], cancelled=parent['cancelled'],
        completed_mopc=partition['completed'], quarantined_mopc=partition['quarantined'],
        pending_count=len(queue), truth_read=False, query_fit=False)
    return manifest, queue


def dispatch(rt):
    validate_runtime(rt)
    require(all(int(v) <= 2 for v in rt['gpu_limits'].values()), 'Per-GPU task limit exceeds two')
    previous = Path(rt['previous_run_root'])
    parent_rt = read(rt['previous_runtime_file'])
    require(str(previous) == parent_rt['run_root'] and rt['previous_runtime_file'] == parent_rt['runtime_file'],
            'Parent runtime differs')
    require_same_method(parent_rt, rt)
    state, parent_manifest = read(previous/'worker_state.json'), read(previous/'resume_manifest.json')
    require(parent_manifest['scope'] == 'MOPC_ONLY_FROM_SAVED_B' and
            parent_manifest['truth_read'] is False and parent_manifest['query_fit'] is False,
            'Wrong retained parent')
    for path in [previous/'owner.json'] + list((previous/'launches').glob('*.json')):
        require(not previous_process_is_live(read(path).get('pid'), parent_rt), 'Parent process still active')
    middle_rt = read(parent_rt['previous_runtime_file'])
    require(parent_rt['previous_run_root'] == middle_rt['run_root'] and
            parent_rt['previous_runtime_file'] == middle_rt['runtime_file'], 'Middle parent differs')
    require_same_method(middle_rt, rt)
    subset_rt = read(middle_rt['previous_runtime_file'])
    subset = Path(middle_rt['previous_run_root'])
    require(str(subset) == subset_rt['run_root'] and middle_rt['previous_runtime_file'] == subset_rt['runtime_file'],
            'Subset parent differs')
    require_same_method(subset_rt, rt)
    subset_manifest = read(subset/'resume_manifest.json')
    for key in ('reused', 'deferred', 'cancelled', 'pending_count'):
        require(parent_manifest[key] == subset_manifest[key], 'Parent categories differ from subset')
    done = read(subset/'subset_predictions_complete.json')
    require(done['status'] == 'SUBSET_FIXED_MOPC_PENDING' and done['truth_read'] is False and
            done['query_fit'] is False, 'Subset is not fixed')
    jobs = saved_b_jobs(rt, subset, subset_manifest, sorted((subset/'jobs').glob('*.json')))
    require(done['completed_pending'] == len(jobs), 'Subset task count differs')
    partition = partition_retained_mopc(previous, state, jobs)
    manifest, queue = successor_manifest(parent_manifest, partition, previous)
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'launches', 'jobs'):
        (root/name).mkdir()
    for i, job in enumerate(queue):
        write(root/'jobs'/f'{i:05d}.json', job)
    write(root/'resume_manifest.json', manifest)
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
          dict(successor_runtime=rt['runtime_file'], successor_root=str(root), scope=manifest['scope']))
    _queue(rt, queue, 'worker')
    for job in queue:
        closed_prediction(job, len(job['points']))
    write(root/'mopc_predictions_complete.json', dict(status='MOPC_PENDING_FIXED_QUARANTINES_REMAIN',
          tasks=len(queue), retained_completed=len(partition['completed']),
          quarantined=[job['key'] for job in partition['quarantined']], truth_read=False, query_fit=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    dispatch(read(args.runtime))
