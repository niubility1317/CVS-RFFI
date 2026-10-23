"""Replace only this task's unstarted CPU queue; preserve every output and GPU job."""
import argparse
import os
from pathlib import Path
import signal
import time

from experiments.cric_dura_comparison.io import read, write
from .view_anchor import RUN, PREVIOUS_RUN


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        state = (p / 'stat').read_text().split()[2]
        return dict(pid=pid, alive=state != 'Z', state=state, cwd=os.readlink(p / 'cwd'),
            argv=[x.decode() for x in (p / 'cmdline').read_bytes().split(bytes([0])) if x],
            children=(p / 'task' / str(pid) / 'children').read_text().split())
    except FileNotFoundError:
        return dict(pid=pid, alive=False)


def validate_queued(old, new, state, proc):
    rows = {row['row_id'] for row in old['rows']}
    root = Path(old['run_root'])
    if (old['run_id'] != PREVIOUS_RUN or new['run_id'] != RUN
            or old.get('launch_owner') != 'codex-pamf-view-anchor-20261023'
            or new.get('launch_owner') != old['launch_owner']
            or rows != {row['row_id'] for row in new['rows']} or len(rows) != 3
            or state.get('status') != 'WAITING_FOR_SMOKE_GPU' or set(state['jobs']) != rows
            or any(job.get('status') != 'QUEUED' or job.get('pid') for job in state['jobs'].values())
            or proc.get('alive') is not True or proc['pid'] != state['controller_pid']
            or proc.get('cwd') != old['release_root'] or proc.get('children') != []
            or proc.get('argv', [])[-2:] != ['--config', old['runtime_file']]
            or 'experiments.pamf_ssl_research.control' not in proc.get('argv', [])
            or any((root / row).exists() for row in rows)
            or (root / 'smoke.json').exists() or (root / 'logs/smoke.log').exists()):
        raise ValueError('Only the exact own queued controller with no smoke/children/GPU work may be replaced')


def replace(new):
    old = read(new['replaces_queued_runtime'])
    root = Path(old['run_root'])
    state = read(root / 'state.json')
    submit = read(Path(old['release_root']) / 'submit.json')
    pid = submit['pid']
    validate_queued(old, new, state, process(pid))
    stopped = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        deadline = time.monotonic() + 2
        while True:
            proc = process(pid)
            if proc.get('state') in ('T', 't'):
                break
            if time.monotonic() >= deadline or not proc.get('alive'):
                raise RuntimeError('Could not hold own queue; no replacement launched')
            time.sleep(.05)
        state = read(root / 'state.json')
        validate_queued(old, new, state, proc)
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        stopped = False
        deadline = time.monotonic() + 5
        while process(pid).get('alive'):
            if time.monotonic() >= deadline:
                raise RuntimeError('Own controller termination UNKNOWN; do not launch a duplicate')
            time.sleep(.05)
    finally:
        if stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
    state.update(status='REPLACED_BEFORE_GPU_RESOURCE_CHANGE', replacement_run_id=new['run_id'],
                 updated_unix=time.time(), controller_exit_independently_verified=True)
    write(root / 'state.json', state, exclusive=False)
    return dict(status='VERIFIED_REPLACED_BEFORE_ANY_GPU_WORK', old_run_id=old['run_id'],
        new_run_id=new['run_id'], old_controller_pid=pid, old_controller_alive=False,
        old_GPU_workers=0, old_smoke_started=False, old_outputs_preserved=True)


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    print(json.dumps(replace(read(parser.parse_args().runtime))))
