"""Transfer one Group1 owner without restarting its existing GPU workers.

Completed and in-flight jobs retain their original immutable artifacts. Only
not-yet-started PAMF predictions use the new release and output root.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import signal
import subprocess
import time

from .external_takeover import process
from .group1_dispatch import completed, jobs
from .io import atomic, read, require, write


def select_job(job, old_by, new_by):
    """Use the score implementation paired with the existing prediction root."""
    if job['origin'] != 'replacement' or job['kind'] == 'source':
        return job
    new = new_by[job['original_id']]
    if job['kind'] == 'predict':
        return new
    require(job['kind'] == 'score', 'Unexpected replacement job')
    predecessor = old_by[job['dependencies'][0]]
    old_complete = completed(predecessor)
    new_complete = completed(new_by[predecessor['original_id']])
    require(old_complete != new_complete, 'Prediction must exist in exactly one output root')
    return job if old_complete else new


def resolve_jobs(items, done, active, new_by, rt):
    """Persist the exact execution and scoring paths before retiring the owner."""
    result = []
    for original in items:
        j = dict(original)
        key = j['id']
        use_new = (j['origin'] == 'replacement' and j['kind'] != 'source'
                   and key not in done and key not in active)
        if use_new and j['kind'] == 'score':
            predecessor = j['dependencies'][0]
            use_new = predecessor not in done and predecessor not in active
        if use_new:
            new = new_by[j['original_id']]
            for field in ('output', 'completion', 'prediction'):
                if field in new:
                    j[field] = new[field]
            j['runtime_file'] = rt['runtime_file']
            j['release_root'] = rt['release_root']
        j['execution_version'] = 'cuda' if use_new else 'original'
        result.append(j)
    return result


class Adopted:
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if process(self.pid) else 0


def _parent(old):
    state = read(Path(old['run_root'])/'pipeline_state.json')
    require(state['status'] == 'RUNNING' and not state['counts']['FAILED']
            and not state['counts']['BLOCKED'], 'Existing owner is not healthy')
    pid = state['controller_pid']
    item = process(pid)
    require(item and item['uid'] == os.getuid() and item['cwd'] == old['release_root']
            and old['runtime_file'] in item['argv']
            and 'experiments.pamf_phase2_practical.group1_recovery' in item['argv']
            and '--job' not in item['argv'], 'Existing controller identity mismatch')
    return pid


def _workers(old, legacy, by):
    found = {}
    origins = {old['runtime_file']: ('replacement', old),
               legacy['runtime_file']: ('legacy', legacy)}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        item = process(int(path.name))
        if not item or item['uid'] != os.getuid() or '--job' not in item['argv']:
            continue
        matches = [x for x in origins if x in item['argv']]
        if not matches:
            continue
        require(len(matches) == 1 and 'experiments.pamf_phase2_practical.group1_dispatch' in item['argv'],
                'Unexpected process using a Group1 runtime')
        origin, cfg = origins[matches[0]]
        require(item['cwd'] == cfg['release_root'], 'Worker CWD mismatch')
        original = item['argv'][item['argv'].index('--job')+1]
        key = origin+'-'+original
        require(key in by and key not in found, 'Unknown or duplicate active job')
        try:
            env = dict(v.split('=', 1) for v in (path/'environ').read_bytes().decode().split('\0') if '=' in v)
        except (FileNotFoundError, ProcessLookupError):
            continue
        gpu = env.get('CUDA_VISIBLE_DEVICES', '')
        require(gpu == '' or gpu.isdigit(), 'Unexpected worker GPU setting')
        found[key] = dict(pid=item['pid'], gpu=int(gpu) if gpu else None,
            kind=by[key]['kind'], profile=by[key]['profile'], command=item['argv'],
            cwd=item['cwd'], runtime_file=cfg['runtime_file'], origin=origin, adopted=True,
            log=str(Path(cfg['run_root'])/'logs'/(original+'.log')), started_at=None)
    require(all(n <= 2 for n in Counter(v['gpu'] for v in found.values() if v['gpu'] is not None).values()),
            'Existing own workers exceed GPU limit')
    require(sum(v['kind'] == 'score' for v in found.values()) <= 1, 'Multiple existing scorers')
    return found


def _check_pending(items, old, legacy):
    for j in items:
        cfg = legacy if j['origin'] == 'legacy' else old
        log = Path(cfg['run_root'])/'logs'/(j['original_id']+'.log')
        require(not Path(j['output']).exists() and not log.exists(),
                'Unowned partial output or launched-job log: '+j['id'])


def _check_children(parent, observed):
    # A child still between fork and exec may not yet identify as a worker.
    children = Path('/proc', str(parent), 'task', str(parent), 'children').read_text().split()
    known = {v['pid'] for v in observed.values()}
    for child in children:
        pid = int(child)
        require(process(pid) is None or pid in known, 'Unidentified live controller child: '+child)


def dispatch(rt):
    old = read(rt['takeover_runtime'])
    legacy = read(old['recovery_runtime'])
    require(old['run_id'] == '20261030-phase2-group1-trisigtime-threepool-m4-r04', 'Unexpected previous run')
    require(rt['method_profiles'] == old['method_profiles'] == ['pamf_tri_sig_time_ssl_milkr'],
            'Wrong PAMF method')
    identity = {'run_id', 'release_root', 'run_root', 'runtime_file', 'code_commit',
                'launch_owner', 'authorization'}
    require(set(old).issubset(rt), 'New runtime omitted existing fields')
    for name in set(old)-identity:
        require(rt[name] == old[name], 'Scientific input changed: '+name)
    root = Path(rt['run_root'])
    old_root = Path(old['run_root'])
    items = read(old_root/'jobs.json')
    by = {j['id']: j for j in items}
    require(len(by) == len(items), 'Duplicate existing job ID')
    new_by = {j['id']: j for j in jobs(rt)}
    require(all(j['original_id'] in new_by for j in items if j['origin'] == 'replacement'),
            'New PAMF matrix does not match existing jobs')
    require(all(completed(j) for j in items if j['kind'] == 'source'), 'Source jobs not all complete')
    producer = read(Path(legacy['run_root'])/'augmentation_owner.json')
    require(process(producer['pid']) is not None, 'Source augmentation producer missing')
    require(not (old_root/'USER_STOP.json').exists() and not (Path(legacy['run_root'])/'USER_STOP.json').exists(),
            'Existing run has a stop request')
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    (root/'job_results').mkdir()
    write(root/'original_jobs.json', items)
    # This runs with the real frozen source models and no target/query data.
    from .cuda_takeover_smoke import run as smoke
    smoke(rt, root/'smoke.json')
    parent = _parent(old)
    os.kill(parent, signal.SIGSTOP)
    time.sleep(.3)
    terminated = False
    try:
        previous = read(old_root/'pipeline_state.json')
        require(previous['controller_pid'] == parent and previous['counts']['FAILED'] == 0
                and previous['counts']['BLOCKED'] == 0, 'Owner changed while preparing handoff')
        observed = _workers(old, legacy, by)
        _check_children(parent, observed)
        done = {j['id'] for j in items if completed(j)}
        for key, v in previous['active'].items():
            require(key in observed or key in done, 'Previous worker exited without a completion: '+key)
        require(not set(observed).intersection(done), 'Active job already has a completion')
        remaining = set(by)-set(observed)-done
        # An unowned partial output is not safe to restart or silently skip.
        _check_pending((by[key] for key in remaining), old, legacy)
        resolved = resolve_jobs(items, done, observed, new_by, rt)
        write(root/'jobs.json', resolved)
        by = {j['id']: j for j in resolved}
        write(root/'owner.json', dict(pid=os.getpid(), runtime=rt['runtime_file'],
            owner=rt['launch_owner'], replacement_for=old['run_id']))
        write(root/'handoff_prepared.json', dict(old_controller_pid=parent,
            adopted_workers=list(observed.values()), completed=len(done), pending=len(remaining)))
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        deadline = time.monotonic()+5
        while process(parent) and time.monotonic() < deadline:
            time.sleep(.1)
        require(process(parent) is None, 'Old controller did not exit')
        terminated = True
    finally:
        if not terminated and process(parent):
            os.kill(parent, signal.SIGCONT)
    write(root/'handoff.json', dict(status='VERIFIED', old_controller_pid=parent,
        old_controller_exited=True, model_workers_restarted=0,
        adopted_workers=list(observed.values()), scientific_config_unchanged=True))
    active = {key: (Adopted(v['pid']), v) for key, v in observed.items()}
    failed = {}
    blocked = set()
    profile_errors = set()
    while True:
        stop = any((p/'USER_STOP.json').exists() for p in (root, old_root, Path(legacy['run_root'])))
        for key, (child, item) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            good = code == 0 and completed(by[key])
            result = dict(item, status='COMPLETE' if good else 'FAILED', exit_code=code,
                          ended_at=time.time(), output=by[key]['output'],
                          completion=by[key]['completion'], prediction=by[key].get('prediction'))
            write(root/'job_results'/(key+'.json'), result)
            if good:
                done.add(key)
            else:
                failed[key] = result
                profile_errors.add((by[key]['origin'], by[key]['profile']))
            del active[key]
        for key in list(remaining):
            if ((by[key]['origin'], by[key]['profile']) in profile_errors
                    or any(dep in failed or dep in blocked for dep in by[key]['dependencies'])):
                blocked.add(key)
                remaining.remove(key)
        available = sorted((by[k] for k in remaining if all(d in done for d in by[k]['dependencies'])),
                           key=lambda j: (j['priority'], j['origin'] != 'replacement', j['id']))
        def launch(job, gpu):
            cfg = rt if job['execution_version'] == 'cuda' else (legacy if job['origin'] == 'legacy' else old)
            require(job['runtime_file'] == cfg['runtime_file'], 'Resolved job runtime mismatch')
            if job['kind'] == 'score':
                require(completed(by[job['dependencies'][0]]), 'Scorer prediction not complete')
                require(job['prediction'] == by[job['dependencies'][0]]['output'], 'Scorer prediction root mismatch')
            args = [cfg['python'], '-u', '-m', 'experiments.pamf_phase2_practical.group1_dispatch',
                    '--runtime', cfg['runtime_file'], '--job', job['original_id']]
            env = dict(os.environ, PYTHONPATH=cfg['release_root'],
                CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            log = Path(cfg['run_root'])/'logs'/(job['original_id']+'.log')
            with log.open('x') as stream:
                child = subprocess.Popen(args, cwd=cfg['release_root'], env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            item = dict(pid=child.pid, gpu=gpu, kind=job['kind'], profile=job['profile'],
                command=args, cwd=cfg['release_root'], runtime_file=cfg['runtime_file'],
                log=str(log), started_at=time.time(), origin=job['origin'], adopted=False,
                output=job['output'], completion=job['completion'], prediction=job.get('prediction'))
            active[job['id']] = (child, item)
            remaining.remove(job['id'])
            available.remove(job)
            print('CUDA_TAKEOVER_LAUNCH', job['id'], child.pid, gpu, cfg['run_id'], flush=True)
        if not stop:
            if not any(v['kind'] == 'score' for _, v in active.values()):
                scorer = next((j for j in available if not j['gpu']), None)
                if scorer:
                    launch(scorer, None)
            from .runner import occupancy
            devices = occupancy()
            for _ in (0, 1):
                for gpu in rt['allowed_gpus']:
                    own = sum(v['gpu'] == gpu for _, v in active.values())
                    if own >= 2 or devices[gpu]['free'] < rt['minimum_free_MiB']:
                        continue
                    job = next((j for j in available if j['gpu']), None)
                    if job is None:
                        continue
                    launch(job, gpu)
                    devices[gpu]['free'] -= rt['minimum_free_MiB']
        counts = dict(COMPLETE=len(done), FAILED=len(failed), BLOCKED=len(blocked),
                      RUNNING=len(active), QUEUED=len(remaining))
        status = 'STOP_REQUESTED' if stop else 'RUNNING'
        if not active and not remaining:
            status = 'COMPLETE' if not failed else 'PARTIAL_TECHNICAL_FAILURE'
        atomic(root/'pipeline_state.json', dict(status=status, run_id=rt['run_id'],
            controller_pid=os.getpid(), counts=counts, active={k:v for k,(_,v) in active.items()},
            updated_at=time.time(), failed_jobs=list(failed), previous_run=old['run_id']))
        if (not active and not remaining) or (stop and not active):
            break
        time.sleep(rt.get('poll_seconds', 10))
    if not stop:
        Path(rt['augmentation_cache'], 'stop').write_text('Group1 CUDA takeover completed\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    dispatch(read(args.runtime))
