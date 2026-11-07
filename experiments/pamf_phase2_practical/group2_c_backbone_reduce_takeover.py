"""Drain the verified quarantined dispatcher and resume under a lower GPU cap."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

from .io import read, require, write
from .group2_c_backbone_fast_takeover import (
    SCIENCE_KEYS, OPTIONAL_SCIENCE_KEYS, _children, _launched_indices,
    _proc, _smoke, link_closed_predictions,
)
from .group2_c_backbone_schedule import _queue, prediction_closed, tasks, validate_runtime


def validate_parent(old, rt):
    require(all(key in old and key in rt and old[key] == rt[key] for key in SCIENCE_KEYS)
            and all(old.get(key) == rt.get(key) for key in OPTIONAL_SCIENCE_KEYS),
            'Parent scientific matrix or input differs')
    require(old.get('minimum_free_MiB') == rt.get('minimum_free_MiB') and
            int(rt['minimum_free_MiB']) >= 6000 and
            set(old['gpu_limits']) == set(rt['gpu_limits']) == set(map(str, range(8))) and
            all(int(rt['gpu_limits'][g]) == 2 and int(old['gpu_limits'][g]) >= 2
                for g in rt['gpu_limits']), 'Invalid reduced GPU policy')
    require(rt.get('excluded_failed_indices') == old.get('excluded_failed_indices') == [36248, 36266]
            and rt.get('deferred_b_methods') == old.get('deferred_b_methods'),
            'Quarantine or dispatch priority differs')


def owner_matches(info, old, pid, ticks):
    if not info or info['pid'] != pid or info['start_ticks'] != ticks or info['cwd'] != old['release_root']:
        return False
    argv = info['argv']
    try:
        return ('experiments.pamf_phase2_practical.group2_c_backbone_resume' in argv and
                argv[argv.index('--runtime') + 1] == old['runtime_file'])
    except (ValueError, IndexError):
        return False


def verify_quarantine(old, jobs):
    oldroot = Path(old['run_root'])
    previous = read(oldroot/'resume.json')
    parent = Path(previous['parent_root'])
    parent_rt = read(old['parent_runtime_file'])
    require(previous['status'] == 'RESUMED' and
            set(previous['excluded']) == set(old['excluded_failed_indices']) and
            str(parent) == parent_rt['run_root'], 'Wrong quarantine parent')
    parent_resume = read(parent/'resume.json')
    ancestor = Path(parent_resume['parent_root'])
    require(parent_resume['status'] == 'RESUMED' and
            set(parent_resume['excluded']) == set(old['excluded_failed_indices']) and
            str(ancestor) == read(parent_rt['parent_runtime_file'])['run_root'], 'Wrong quarantine ancestor')
    for index in old['excluded_failed_indices']:
        partial = ancestor/'predictions'/jobs[index]['key']
        require(not (oldroot/'predictions'/jobs[index]['key']).exists() and
                not (parent/'predictions'/jobs[index]['key']).exists() and
                not (partial/'complete.json').exists() and any(partial.glob('*.npz')),
                'Inherited partial missing or changed')


def dispatch(rt):
    validate_runtime(rt)
    old = read(rt['parent_runtime_file'])
    validate_parent(old, rt)
    root, oldroot = Path(rt['run_root']), Path(old['run_root'])
    pid, ticks = int(rt['parent_controller_pid']), int(rt['parent_start_ticks'])
    require(owner_matches(_proc(pid), old, pid, ticks), 'Parent identity differs')
    require(owner_matches(_proc(pid), old, pid, ticks), 'Parent changed before pause')
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            info = _proc(pid)
            require(owner_matches(info, old, pid, ticks), 'Parent changed during pause')
            if info['state'] in ('T', 't'):
                break
            time.sleep(.05)
        require(info['state'] in ('T', 't'), 'Parent pause failed')
        state = read(oldroot/'worker_state.json')
        require(state['status'] == 'RUNNING' and not state['failed'] and
                int(state['excluded']) == len(rt['excluded_failed_indices']) and
                not (oldroot/'predictions_complete.json').exists(), 'Parent state differs')
        require(not root.exists(), 'Successor root already exists')
        claim = oldroot.parent/'.resume_successors'/(oldroot.name+'.json')
        write(claim, dict(parent_root=str(oldroot), successor_root=str(root),
                          successor_runtime=rt['runtime_file']))
        root.mkdir(parents=True, exist_ok=False)
        for name in ('logs', 'launches', 'jobs'):
            (root/name).mkdir()
        write(root/'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv))
        write(root/'takeover.json', dict(status='PREPARING', parent_pid=pid, parent_start_ticks=ticks), exclusive=False)
        jobs = tasks(rt)
        verify_quarantine(old, jobs)
        for index, job in enumerate(jobs):
            write(root/'jobs'/f'{index:05d}.json', job)
        parent_state = read(oldroot/'worker_state.json')
        require(parent_state['status'] == 'RUNNING' and not parent_state['failed'], 'Parent already failed')
        children = _children(old, pid)
        launched = _launched_indices(old)
        active_state = {int(item['index']) for item in parent_state['active']}
        write(root/'takeover.json', dict(status='DRAINING_PARENT', parent_pid=pid,
              parent_start_ticks=ticks, parent_workers=children), exclusive=False)
        while True:
            require(owner_matches(_proc(pid), old, pid, ticks) and _proc(pid)['state'] in ('T', 't'),
                    'Parent identity or paused state changed')
            active = [index for index, child in children.items()
                      if (p := _proc(child['pid'])) and p['start_ticks'] == child['start_ticks'] and p['state'] != 'Z']
            if not active:
                break
            time.sleep(int(rt['poll_seconds']))
        for index in set(children) | launched | active_state:
            require(index < len(jobs), 'Parent worker index out of bounds')
            prediction_closed(dict(jobs[index], output=str(oldroot/'predictions'/jobs[index]['key'])))
        verify_quarantine(old, jobs)
        _smoke(rt, root)
        reused = link_closed_predictions(oldroot, jobs)
        require(len(reused) >= int(parent_state['completed']), 'Incomplete parent prediction reuse')
        require(owner_matches(_proc(pid), old, pid, ticks) and _proc(pid)['state'] in ('T', 't'),
                'Parent changed before transfer')
        write(root/'takeover.json', dict(status='READY_TO_TRANSFER', parent_pid=pid,
              parent_start_ticks=ticks, reused=len(reused), excluded=rt['excluded_failed_indices']), exclusive=False)
        os.kill(pid, signal.SIGKILL)
        write(root/'takeover.json', dict(status='TRANSFERRED', parent_pid=pid,
              parent_start_ticks=ticks, reused=len(reused), excluded=rt['excluded_failed_indices']), exclusive=False)
        _queue(rt, jobs, 'worker', reused_indices=reused, excluded_indices=set(rt['excluded_failed_indices']))
        write(root/'resume_complete.json', dict(status='AWAITING_QUARANTINED_JOBS',
              completed=len(jobs)-len(rt['excluded_failed_indices']), total=len(jobs),
              excluded=rt['excluded_failed_indices'], truth_read=False, scoring_started=False))
    except BaseException as exc:
        if root.is_dir():
            write(root/'takeover_error.json', dict(error=repr(exc), parent_pid=pid,
                  parent_start_ticks=ticks, parent=_proc(pid), note='Do not resume the former three-per-GPU dispatcher.'), exclusive=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    dispatch(read(parser.parse_args().runtime))
