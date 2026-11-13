"""Add staged reduced-weight Hybrid rows to the existing single GPU owner."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

from experiments.cric_dura_comparison.io import read
from experiments.practical_full_2x2.cache import atomic
from . import concurrency_takeover as scheduler

OLD_RELEASE = scheduler.PROJECT / 'releases/20261113-phase1-pamf-hybrid-tracked-failing-m26-r01-release'
OLD_MODULE = 'experiments.pamf_physics_tri.hybrid_tracked_dispatch'
NEW_RUN = '20261113-phase1-pamf-hybrid-structure-abcd-m16-r01'
NEW_MODULE = 'experiments.pamf_physics_tri.hybrid_structure_runner'
EXISTING = (('20261113-phase1-pamf-hybrid-confidence-low-m24-r01',
             'experiments.pamf_physics_tri.hybrid_low_runner', None),
            ('20261113-phase1-pamf-hybrid-tracked-failing-m26-r01',
             'experiments.pamf_physics_tri.hybrid_tracked_runner', None))


def queues():
    result = []
    old_pids = set()
    for run, module, _ in EXISTING:
        root = scheduler.PROJECT / 'runs' / run
        state = read(root / 'pipeline_state.json')
        if state['status'] == 'ANALYZED':
            continue
        if state['status'] != 'RUNNING':
            raise ValueError(f'Existing queue is not safely running: {run}')
        rt = read(scheduler.PROJECT / 'releases' / (run + '-release') / 'runtime.json')
        old_pids.add(state['pid'])
        result.append(dict(run=run, module=module, old_pid=state['pid'], rt=rt, root=root, state=state))
    if len(old_pids) > 1:
        raise ValueError('Expected one existing shared owner for all active queues')
    pid = old_pids.pop() if old_pids else None
    if pid is not None:
        info = scheduler.process_info(pid)
        expected = [result[0]['rt']['python'], '-u', '-m', OLD_MODULE]
        if info is None or info['argv'] != expected or info['cwd'] != str(OLD_RELEASE):
            raise ValueError('Existing shared owner identity differs')
    root = scheduler.PROJECT / 'runs' / NEW_RUN
    state = read(root / 'pipeline_state.json')
    rt = read(scheduler.PROJECT / 'releases' / (NEW_RUN + '-release') / 'runtime.json')
    if state['status'] != 'STAGED' or any(item['status'] != 'QUEUED' for item in state['jobs'].values()):
        raise ValueError('Hybrid queue is not cleanly staged')
    result.append(dict(run=NEW_RUN, module=NEW_MODULE, old_pid=None,
                       rt=rt, root=root, state=state, jobs=read(root / 'jobs.json')))
    return pid, result


def handoff(pid, all_queues):
    os.kill(pid, signal.SIGSTOP)
    stopped = True
    try:
        deadline = time.monotonic() + 2
        while Path(f'/proc/{pid}/stat').read_text().split()[2] not in ('T', 't') and time.monotonic() < deadline:
            time.sleep(.02)
        if Path(f'/proc/{pid}/stat').read_text().split()[2] not in ('T', 't'):
            raise ValueError('Existing shared owner did not pause')
        running = set()
        for q in all_queues:
            state = read(q['root'] / 'pipeline_state.json')
            if q['old_pid'] is None:
                if state['status'] != 'STAGED':
                    raise ValueError('New queue changed during handoff')
                state['status'] = 'RUNNING'
            else:
                if state['pid'] != pid or state['status'] != 'RUNNING':
                    raise ValueError('Existing queue changed owner during handoff')
                for item in state['jobs'].values():
                    if item['status'] == 'RUNNING':
                        running.add(item['pid'])
                        scheduler.adopt_worker(item)
                q['jobs'] = read(q['root'] / 'jobs.json')
            q['state'] = state
        if not scheduler.child_pids(pid) <= running:
            raise ValueError('Existing shared owner has an unrecorded live child')
        os.kill(pid, signal.SIGKILL)
        stopped = False
    finally:
        if stopped:
            os.kill(pid, signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    pid, all_queues = queues()
    if args.check:
        print('READY', flush=True)
        return
    if pid is not None:
        handoff(pid, all_queues)
    else:
        all_queues[0]['state']['status'] = 'RUNNING'
    scheduler.run(all_queues, read, atomic)


if __name__ == '__main__':
    main()
