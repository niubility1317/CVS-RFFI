"""Single-owner takeover retaining every completed C-backbone prediction.

The parent dispatcher is paused while its own workers finish. A failed
pre-transfer restores the verified parent; no healthy worker is stopped.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .io import read, require, write
from .group2_c_backbone_schedule import _gpu_free, _queue, prediction_closed, tasks, validate_runtime


SCIENCE_KEYS = ('data_root','K','N','capsules','model_seeds','b_methods','c_methods','c_source_roots',
    'hu_aux_roots_by_seed','pilot_roots','source_cache','source_contract','batch_size')
OPTIONAL_SCIENCE_KEYS = ('old_classes','new_classes','phase1_release','frozen_sources','source_run_root',
    'native_adaptation_overrides','native_registration_overrides','scope','device')


def check_parent_runtime(old, current):
    require(all(key in old and key in current and old[key] == current[key] for key in SCIENCE_KEYS) and
            all(old.get(key) == current.get(key) for key in OPTIONAL_SCIENCE_KEYS),
            'Parent scientific matrix or input differs')
    previous, updated = old.get('gpu_limits'), current.get('gpu_limits')
    require((previous == updated or (isinstance(previous, dict) and isinstance(updated, dict) and
            set(previous) == set(updated) and all(0 < int(previous[g]) <= int(updated[g]) <=
            (2 if g == '7' else 3) for g in previous))) and
            old.get('minimum_free_MiB') == current.get('minimum_free_MiB'), 'Parent GPU policy differs')


def worker_index(argv, runtime):
    try:
        if ('experiments.pamf_phase2_practical.group2_c_backbone_schedule' not in argv or
                'worker' not in argv or argv[argv.index('--runtime')+1] != runtime):
            return None
        index = int(argv[argv.index('--index')+1])
        return index if index >= 0 else None
    except (ValueError, IndexError):
        return None


def _proc(pid):
    folder = Path('/proc')/str(pid)
    try:
        stat = (folder/'stat').read_text().split(') ', 1)[1].split()
        argv = [p.decode() for p in (folder/'cmdline').read_bytes().split(b'\0') if p]
        cwd = os.readlink(folder/'cwd') if stat[0] != 'Z' else None
        return dict(pid=pid, state=stat[0], start_ticks=int(stat[19]), argv=argv, cwd=cwd)
    except (FileNotFoundError, ProcessLookupError):
        return None


def _owner_matches(info, old, start_ticks):
    if not info or info['start_ticks'] != start_ticks or info['cwd'] != old['release_root']:
        return False
    argv=info['argv']
    valid_entry = (('experiments.pamf_phase2_practical.group2_c_backbone_schedule' in argv and 'dispatch' in argv) or
                   'experiments.pamf_phase2_practical.group2_c_backbone_fast_takeover' in argv)
    try:
        return valid_entry and argv[argv.index('--runtime')+1] == old['runtime_file']
    except (ValueError,IndexError):
        return False


def _children(old, pid):
    childfile = Path('/proc')/str(pid)/'task'/str(pid)/'children'
    children = [int(p) for p in childfile.read_text().split()]
    known = {}
    for file in (Path(old['run_root'])/'launches').glob('worker-*.json'):
        launch = read(file)
        known[int(launch['pid'])] = int(file.stem.split('-')[-1])
    found = {}
    for child in children:
        info = _proc(child)
        if info is None:
            continue
        if info['state'] == 'Z':
            require(child in known, 'Unknown exited parent child; resume parent')
            index = known[child]
        else:
            index = worker_index(info['argv'], old['runtime_file'])
            require(index is not None and info['cwd'] == old['release_root'],
                    'Unexpected parent child; resume parent')
        require(index not in found, 'Duplicate parent worker index')
        found[index] = dict(pid=child, start_ticks=info['start_ticks'])
    return found


def _launched_indices(old):
    indices = set()
    for file in (Path(old['run_root'])/'launches').glob('worker-*.json'):
        launch = read(file)
        index = worker_index(launch['argv'], old['runtime_file'])
        require(index is not None and index == int(file.stem.split('-')[-1]) and
                launch['cwd'] == old['release_root'] and index not in indices,
                'Invalid parent worker launch record')
        indices.add(index)
    return indices


def _copy_or_link(source, target):
    try: os.link(source, target)
    except OSError: shutil.copyfile(source, target)
    return target


def link_closed_predictions(parent_root, jobs):
    lookup = {job['key']: (index, job) for index,job in enumerate(jobs)}
    reused = set()
    directory = Path(parent_root)/'predictions'
    for source in directory.iterdir() if directory.exists() else ():
        require(source.is_dir() and source.name in lookup, 'Unexpected parent prediction output')
        index, job = lookup[source.name]
        old_job = dict(job, output=str(source))
        prediction_closed(old_job)
        target = Path(job['output'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, copy_function=_copy_or_link)
        prediction_closed(job)
        reused.add(index)
    return reused


def _smoke(rt, root):
    deadline = time.monotonic()+120
    gpu = None
    while gpu is None and time.monotonic() < deadline:
        gpu = next((g for g in rt['gpu_limits'] if _gpu_free(g) >= rt['minimum_free_MiB']), None)
        if gpu is None: time.sleep(rt['poll_seconds'])
    require(gpu is not None, 'No GPU available for source smoke; resume parent')
    cmd = [sys.executable,'-u','-m',__package__+'.group2_c_backbone_schedule','smoke','--runtime',rt['runtime_file']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'logs'/'smoke.log').open('x',encoding='utf-8') as log:
        proc = subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(root/'launches'/'smoke.json',dict(pid=proc.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
        require(proc.wait() == 0, 'Real checkpoint source-only smoke failed; resume parent')
    smoke = read(root/'source_smoke.json')
    require(smoke['status'] == 'PASS' and smoke['regularization_check']['status'] == 'PASS' and
            smoke['regularization_check']['loss_equal'] and smoke['regularization_check']['gradients_equal'],
            'Source-only accelerated penalty check did not close')


def dispatch(rt):
    validate_runtime(rt)
    old = read(rt['parent_runtime_file']); check_parent_runtime(old,rt)
    oldroot = Path(old['run_root']); oldowner = read(oldroot/'owner.json'); pid = int(oldowner['pid'])
    require(pid == int(rt['parent_controller_pid']), 'Parent owner PID differs')
    info = _proc(pid); require(info is not None, 'Parent owner is absent')
    ticks = info['start_ticks']
    require(_owner_matches(info,old,ticks), 'Parent owner identity differs')
    require(read(oldroot/'worker_state.json')['status'] == 'RUNNING' and
            not (oldroot/'predictions_complete.json').exists(), 'Parent no longer running prediction')
    jobs = tasks(rt); root = Path(rt['run_root']); root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','launches','jobs'): (root/name).mkdir()
    for index,job in enumerate(jobs): write(root/'jobs'/f'{index:05d}.json',job)
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    write(root/'takeover.json',dict(status='PREPARING',parent_pid=pid,parent_start_ticks=ticks),exclusive=False)
    require(_owner_matches(_proc(pid),old,ticks), 'Parent identity changed before pause')
    paused = False; transferred = False
    try:
        os.kill(pid,signal.SIGSTOP); paused = True
        for _ in range(100):
            info = _proc(pid)
            require(_owner_matches(info,old,ticks), 'Parent changed during pause')
            if info['state'] in ('T','t'): break
            time.sleep(.05)
        require(info['state'] in ('T','t'), 'Parent pause failed')
        parent_state = read(oldroot/'worker_state.json')
        require(parent_state['status'] == 'RUNNING' and not parent_state.get('failed'),
                'Parent stopped or failed before transfer')
        previously_active = {int(item['index']) for item in parent_state.get('active', [])}
        children = _children(old,pid)
        launched = _launched_indices(old)
        write(root/'takeover.json',dict(status='DRAINING_PARENT',parent_pid=pid,parent_start_ticks=ticks,
                                       parent_workers=children),exclusive=False)
        while True:
            active = []
            for index,child in children.items():
                proc = _proc(child['pid'])
                if proc and proc['start_ticks'] == child['start_ticks'] and proc['state'] != 'Z': active.append(index)
            if not active: break
            time.sleep(rt['poll_seconds'])
        for index in set(children) | previously_active | launched:
            require(index < len(jobs), 'Parent worker index out of bounds')
            oldjob = dict(jobs[index],output=str(oldroot/'predictions'/jobs[index]['key']))
            prediction_closed(oldjob)
        _smoke(rt,root)
        reused = link_closed_predictions(oldroot,jobs)
        require(len(reused) >= int(parent_state['completed']), 'Incomplete parent prediction reuse')
        require(_owner_matches(_proc(pid),old,ticks) and _proc(pid)['state'] in ('T','t'), 'Parent changed before transfer')
        write(root/'takeover.json',dict(status='READY_TO_TRANSFER',parent_pid=pid,parent_start_ticks=ticks,
                                       reused=len(reused),parent_workers=children),exclusive=False)
        os.kill(pid,signal.SIGKILL); transferred = True; paused = False
        write(root/'takeover.json',dict(status='TRANSFERRED',parent_pid=pid,parent_start_ticks=ticks,
                                       reused=len(reused),parent_workers=children),exclusive=False)
        _queue(rt,jobs,'worker',reused_indices=reused)
        for job in jobs: prediction_closed(job)
        count=sum(len(job['points']) for job in jobs)
        write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED',tasks=len(jobs),rows=count,
                                                    truth_read=False,query_fit=False))
        if rt.get('persistent_score_workers'):
            from .group2_c_backbone_score_pool import score_pool
            score_pool(rt,jobs)
        else:
            _queue(rt,jobs,'score')
        write(root/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(jobs),rows=count,
                                         truth_last=True,target_feedback=False,reused_prediction_tasks=len(reused)))
    finally:
        if paused and not transferred:
            info = _proc(pid)
            if _owner_matches(info,old,ticks) and info['state'] in ('T','t'):
                os.kill(pid,signal.SIGCONT)
                write(root/'takeover.json',dict(status='FAILED_PARENT_RESUMED',parent_pid=pid,
                                               parent_start_ticks=ticks),exclusive=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--runtime',required=True)
    args = parser.parse_args(); dispatch(read(args.runtime))
