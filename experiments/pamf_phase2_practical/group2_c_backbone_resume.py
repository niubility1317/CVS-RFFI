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
                                            b'experiments.pamf_phase2_practical.group2_c_backbone_fast_takeover',
                                            b'experiments.pamf_phase2_practical.group2_c_backbone_resume')):
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
    inherited = set(old.get('excluded_failed_indices', []))
    expected_failed = set(rt.get('retry_cuda_init_indices', [])) if inherited else excluded
    require(state['status'] == 'FAILED' and not state.get('active') and
            excluded and failed == expected_failed and
            (not inherited or (excluded == inherited and len(failed) == 1 and not (failed & excluded))),
            'Parent failed task inventory differs')
    require(not _old_processes(old['runtime_file']) and not (oldroot/'predictions_complete.json').exists(),
            'Parent still running or predictions already frozen')
    jobs = tasks(rt)
    if inherited:
        require(old.get('deferred_b_methods') == rt.get('deferred_b_methods') and
                int(state.get('excluded', -1)) == len(excluded), 'Inherited quarantine differs')
        previous = read(oldroot/'resume.json')
        ancestor = Path(previous['parent_root'])
        require(set(previous['excluded']) == excluded and
                str(ancestor) == read(old['parent_runtime_file'])['run_root'], 'Wrong quarantine ancestor')
        for index in excluded:
            parent_partial = ancestor/'predictions'/jobs[index]['key']
            require(not (oldroot/'predictions'/jobs[index]['key']).exists() and
                    not (parent_partial/'complete.json').exists() and any(parent_partial.glob('*.npz')),
                    'Inherited partial prediction is missing or changed')
        for index in failed:
            log = (oldroot/'logs'/f'worker-{index:05d}.log').read_text(encoding='utf-8')
            require('CUDA driver initialization failed' in log and
                    not (oldroot/'predictions'/jobs[index]['key']).exists(),
                    'Failed job is not a pre-prediction CUDA initialization failure')
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
    reused = link_closed_predictions(oldroot, jobs, excluded_indices=() if inherited else excluded)
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
