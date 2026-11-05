"""Prioritize unstarted B-backbone jobs while C workers finish naturally.

Only the verified C dispatcher is paused. A separate recovery process waits
for any surviving B processes before resuming that same C dispatcher.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import read, write, require
from .group2_c_backbone_fast_takeover import _proc, _children, _launched_indices
from .group2_c_backbone_schedule import prediction_closed
from .group2_resume import dispatch as resume_b


def peer_matches(info, peer):
    if not info or info['pid'] != peer['pid'] or info['start_ticks'] != peer['start_ticks'] or info['cwd'] != peer['release_root']:
        return False
    argv = info['argv']
    try:
        return ('experiments.pamf_phase2_practical.group2_c_backbone_resume' in argv and
                argv[argv.index('--runtime')+1] == peer['runtime_file'])
    except (ValueError, IndexError):
        return False


def resume_peer(peer):
    info = _proc(peer['pid'])
    if peer_matches(info, peer) and info['state'] in ('T', 't'):
        os.kill(peer['pid'], signal.SIGCONT)
        return True
    return False


def workers_drained(children, root):
    for index, child in children.items():
        info = _proc(child['pid'])
        if info and info['start_ticks'] == child['start_ticks'] and info['state'] != 'Z':
            return False
    for index in children:
        prediction_closed(read(root/'jobs'/f'{index:05d}.json'))
    return True


def b_processes(runtime_file):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if runtime_file.encode() in argv and any(module in argv for module in (
                b'experiments.pamf_phase2_practical.group2_resume',
                b'experiments.pamf_phase2_practical.group2_schedule')):
            found.append(int(path.name))
    return found


def watch(rt, control):
    data = read(control/'coordinator.json')
    write(control/'watcher.json', {'pid': os.getpid(), 'start_ticks': _proc(os.getpid())['start_ticks']})
    while True:
        info = _proc(data['pid'])
        if not info or info['start_ticks'] != data['start_ticks'] or info['state'] == 'Z':
            break
        time.sleep(5)
    while b_processes(rt['runtime_file']):
        time.sleep(5)
    resumed = resume_peer(rt['priority_peer'])
    write(control/'watcher_complete.json', {'peer_resumed': resumed, 'b_processes_remaining': []})


def dispatch(rt):
    peer = rt['priority_peer']
    old = read(peer['runtime_file'])
    require(old['run_root'] == peer['run_root'] and old['release_root'] == peer['release_root'], 'Peer runtime differs')
    info = _proc(peer['pid'])
    require(peer_matches(info, peer) and info['state'] not in ('T', 't', 'Z'), 'Peer is not the expected running controller')
    state = read(Path(old['run_root'])/'worker_state.json')
    require(state['status'] == 'RUNNING' and not state['failed'], 'Peer no longer healthy')
    require(not Path(rt['run_root']).exists(), 'B successor root already exists')
    control = Path(rt['run_root']+'_coordination')
    control.mkdir(parents=True, exist_ok=False)
    mine = _proc(os.getpid())
    write(control/'coordinator.json', {'pid': os.getpid(), 'start_ticks': mine['start_ticks']})
    command = [sys.executable, '-u', '-m', __package__+'.group2_priority', 'watch', '--runtime', rt['runtime_file']]
    with (control/'watcher.log').open('x', encoding='utf-8') as log:
        watcher = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    paused = False
    try:
        deadline = time.monotonic()+30
        while not (control/'watcher.json').exists() and watcher.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        require((control/'watcher.json').exists() and watcher.poll() is None, 'Recovery watcher did not start')
        require(peer_matches(_proc(peer['pid']), peer), 'Peer changed before pause')
        os.kill(peer['pid'], signal.SIGSTOP)
        paused = True
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            info = _proc(peer['pid'])
            require(peer_matches(info, peer), 'Peer changed during pause')
            if info['state'] in ('T', 't'):
                break
            time.sleep(.05)
        require(info['state'] in ('T', 't'), 'Peer dispatcher did not pause')
        state = read(Path(old['run_root'])/'worker_state.json')
        require(state['status'] == 'RUNNING' and not state['failed'], 'Peer failed before priority pause')
        children = _children(old, peer['pid'])
        launched = _launched_indices(old)
        for item in state.get('active', []):
            if item['index'] not in children:
                prediction_closed(read(Path(old['run_root'])/'jobs'/f"{item['index']:05d}.json"))
        write(control/'state.json', {'status': 'DRAINING_C_WORKERS', 'children': children, 'peer': peer})
        deadline = time.monotonic()+14400
        while not workers_drained(children, Path(old['run_root'])):
            require(watcher.poll() is None, 'Recovery watcher exited')
            require(time.monotonic() < deadline, 'C worker drain timed out')
            info = _proc(peer['pid'])
            require(peer_matches(info, peer) and info['state'] in ('T', 't'), 'Peer no longer paused')
            time.sleep(rt['poll_seconds'])
        # A worker can be reaped before it ever appears in worker_state. Every
        # recorded launch must close, including that admission/reaping window.
        for index in launched:
            prediction_closed(read(Path(old['run_root'])/'jobs'/f'{index:05d}.json'))
        require(not b_processes(rt['runtime_file']), 'B run already has a live process')
        write(control/'peer_handoff.json', {'B_runtime': rt['runtime_file'], 'C_runtime': old['runtime_file']})
        write(control/'state.json', {'status': 'B_PRIORITY_RUNNING', 'peer': peer}, exclusive=False)
        resume_b(rt)
        write(control/'state.json', {'status': 'B_PENDING_COMPLETE_QUARANTINES_RETAINED', 'peer': peer}, exclusive=False)
    except BaseException as exc:
        write(control/'state.json', {'status': 'FAILED_RETAINED', 'error': str(exc), 'peer': peer}, exclusive=False)
        raise
    finally:
        # The normal queue drains its workers on technical failure. On an abrupt
        # coordinator death the independent watcher waits for surviving B work.
        if paused and not b_processes(rt['runtime_file']):
            resume_peer(peer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'watch'])
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    runtime = read(args.runtime)
    if args.action == 'watch':
        watch(runtime, Path(runtime['run_root']+'_coordination'))
    else:
        dispatch(runtime)
