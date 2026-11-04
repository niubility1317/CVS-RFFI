"""Resume unaffected jobs after explicitly quarantining failed partial outputs.

The parent stays immutable. No truth barrier or scoring is opened while any
quarantined job remains unresolved.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .io import read, require, write
from .group2_c_backbone_fast_takeover import check_parent_runtime, link_closed_predictions, _smoke
from .group2_c_backbone_schedule import _queue, tasks, validate_runtime


def _old_processes(runtime_file):
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = (folder/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if runtime_file.encode() in argv and any(
                token in argv for token in (b'experiments.pamf_phase2_practical.group2_c_backbone_schedule',
                                            b'experiments.pamf_phase2_practical.group2_c_backbone_fast_takeover')):
            found.append(int(folder.name))
    return found


def dispatch(rt):
    validate_runtime(rt)
    old = read(rt['parent_runtime_file'])
    _dispatch_locked(rt, old)


def _dispatch_locked(rt, old):
    check_parent_runtime(old, rt)
    oldroot = Path(old['run_root'])
    state = read(oldroot/'worker_state.json')
    excluded = set(rt['excluded_failed_indices'])
    failed = {int(item['index']) for item in state.get('failed', [])}
    require(state['status'] == 'FAILED' and not state.get('active') and excluded == failed and excluded,
            'Parent failed task inventory differs')
    require(not _old_processes(old['runtime_file']) and not (oldroot/'predictions_complete.json').exists(),
            'Parent still running or predictions already frozen')
    jobs = tasks(rt)
    root = Path(rt['run_root'])
    require(not root.exists(), 'Successor root already exists')
    # The permanent parent-to-successor record is created atomically. If a
    # controller exits, future recovery must inspect its successor and outputs.
    claim = oldroot.parent/'.resume_successors'/(oldroot.name+'.json')
    write(claim, dict(parent_root=str(oldroot), successor_root=str(root),
                      successor_runtime=rt['runtime_file']))
    root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'launches', 'jobs'):
        (root/name).mkdir()
    for index, job in enumerate(jobs):
        write(root/'jobs'/f'{index:05d}.json', job)
    write(root/'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv))
    write(root/'resume.json', dict(status='PREPARING', parent_root=str(oldroot),
                                  excluded=sorted(excluded)))
    _smoke(rt, root)
    reused = link_closed_predictions(oldroot, jobs, excluded_indices=excluded)
    require(len(reused) == int(state['completed']), 'Parent closed prediction count differs')
    write(root/'resume.json', dict(status='RESUMED', parent_root=str(oldroot),
                                  reused=len(reused), excluded=sorted(excluded)), exclusive=False)
    _queue(rt, jobs, 'worker', reused_indices=reused, excluded_indices=excluded)
    write(root/'resume_complete.json', dict(status='AWAITING_QUARANTINED_JOBS',
          completed=len(jobs)-len(excluded), total=len(jobs), excluded=sorted(excluded),
          truth_read=False, scoring_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    dispatch(read(args.runtime))
