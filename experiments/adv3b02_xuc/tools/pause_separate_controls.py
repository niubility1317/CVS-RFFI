"""Scoped pause after checkpoint verification; native workers drain unchanged.

Run with the experiment Python on N607. Snapshot and stop are separate phases.
The immutable pause manifest is also the input for a later explicit continuation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start=stat[19], state=stat[0],
                    cwd=str((proc / 'cwd').resolve(strict=True)),
                    argv=(proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def checked(pid, cwd, argv=None, run=None):
    info = identity(pid)
    if info is None:
        raise RuntimeError(f'Process absent: {pid}')
    if info['cwd'] != cwd or (argv is not None and info['argv'] != argv):
        raise RuntimeError(f'Process identity mismatch: {pid}')
    if run is not None and str(run) not in info['argv'] and run.name not in info['argv']:
        raise RuntimeError(f'Run ownership mismatch: {pid}')
    return info


def same_process(saved):
    live = identity(saved['pid'])
    return live is not None and all(live[k] == saved[k] for k in ('start', 'cwd', 'argv'))


def complete(folder):
    path = folder / 'completion.json'
    if not path.is_file() or not (folder / 'final_ssdg.pth').is_file():
        return False
    value = json.loads(path.read_text())
    return value.get('status') == 'TRAINING_COMPLETE' and value.get('epochs') == 200 and value.get('steps') == 44400


def live_workers(run, release):
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        info = identity(int(proc.name))
        if info is None or info['cwd'] != release or '--output' not in info['argv']:
            continue
        output = Path(info['argv'][info['argv'].index('--output') + 1])
        if output.parent == run and output.name.startswith('PURE_'):
            result.append(info)
    return result


def snapshot(run, destination):
    import torch
    state = json.loads((run / 'pipeline_state.json').read_text())
    dispatcher = checked(state['pid'], state['release'], run=run)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    manifest = dict(run=str(run), created=time.time(), phase='SNAPSHOTTING',
                    dispatcher=dispatcher, original_state=state, rows={})
    write(destination / 'manifest.json', manifest)
    for rid, entry in state['rows'].items():
        if entry.get('family') != 'pure_game' or entry['status'] != 'RUNNING':
            continue
        if complete(run / rid):
            continue
        info = checked(entry['pid'], state['release'], entry['argv'])
        source = run / rid / 'latest_ssdg.pth'
        (destination / rid).mkdir()
        target = destination / rid / 'latest_ssdg.pth'
        # The trainer atomically replaces latest; a hardlink retains that exact inode.
        os.link(source, target)
        payload = torch.load(target, map_location='cpu', weights_only=False)
        contract = json.loads((run / 'source_contract.json').read_text())
        if payload.get('target_contact') is not False or payload.get('initialization') != 'scratch_only':
            raise ValueError('Invalid original lineage')
        if payload['source_info']['role_ids'] != contract['role_ids']:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        row = payload['row']
        if row['id'] != rid or row.get('pure_game') is not True or row['joint']['native_dr']:
            raise ValueError('Not the requested pure-game row')
        for key in ('model', 'optimizer', 'ema', 'prototype', 'solver', 'tickets', 'daot_rc4'):
            if payload.get(key) is None:
                raise ValueError('Missing persistent state: ' + key)
        if payload['step'] != payload['epoch'] * 222 or payload['solver']['steps'] != payload['step']:
            raise ValueError('Not a complete consistent epoch')
        if payload['epoch'] >= 200:
            raise RuntimeError('Final checkpoint is being written; reconcile completed row before snapshot')
        manifest['rows'][rid] = dict(process=info, checkpoint=str(target), epoch=payload['epoch'],
                                    step=payload['step'], gpu=entry['gpu'], validation='STRUCTURAL_ONLY')
        write(destination / 'manifest.json', manifest)
        del payload
    manifest['phase'] = 'AWAITING_RESTORE_VALIDATION'
    write(destination / 'manifest.json', manifest)


def stop(run, destination):
    manifest = json.loads((destination / 'manifest.json').read_text())
    approval = json.loads((destination / 'restore_validation.json').read_text())
    if manifest['run'] != str(run) or approval.get('status') != 'PASS':
        raise ValueError('Missing restore verification')
    if set(approval['rows']) != set(manifest['rows']):
        raise ValueError('Restore verification coverage differs')
    for rid, saved in manifest['rows'].items():
        result = approval['rows'][rid]
        if result.get('status') != 'PASS' or result.get('checkpoint') != saved['checkpoint']:
            raise ValueError('Unverified checkpoint: ' + rid)
        if not Path(saved['checkpoint']).is_file():
            raise FileNotFoundError(saved['checkpoint'])
    dispatcher = manifest['dispatcher']
    if not same_process(dispatcher):
        raise RuntimeError('Dispatcher changed; reconcile before stopping')
    # Freeze allocation first, then read state again. Never signal process groups.
    os.kill(dispatcher['pid'], signal.SIGSTOP)
    time.sleep(.2)
    suspended_workers = []
    try:
        frozen = identity(dispatcher['pid'])
        if frozen is None or not same_process(dispatcher) or frozen['state'] not in ('T', 't'):
            raise RuntimeError('Dispatcher freeze not observed')
        state = json.loads((run / 'pipeline_state.json').read_text())
        for rid, entry in state['rows'].items():
            if entry['family'] == 'pure_game' and complete(run / rid):
                entry['status'] = 'TRAINING_COMPLETE'
        running = {rid for rid, e in state['rows'].items()
                   if e['family'] == 'pure_game' and e['status'] == 'RUNNING'}
        if running - set(manifest['rows']):
            raise RuntimeError('New workers appeared; create a fresh snapshot')
        expected = {saved['process']['pid']: saved['process'] for saved in manifest['rows'].values()}
        for live in live_workers(run, state['release']):
            output = Path(live['argv'][live['argv'].index('--output') + 1])
            if complete(output):
                continue
            saved = expected.get(live['pid'])
            if saved is None or any(live[k] != saved[k] for k in ('start', 'cwd', 'argv')):
                raise RuntimeError('Unrecorded live worker; resnapshot before release')
        for rid in running:
            saved = manifest['rows'][rid]
            if not same_process(saved['process']):
                # A just-completed row may disappear between dispatcher polls.
                if complete(run / rid):
                    continue
                raise RuntimeError('Worker changed: ' + rid)
        # Prevent dispatcher from interpreting intentionally stopped rows as failures.
        os.kill(dispatcher['pid'], signal.SIGTERM)
        os.kill(dispatcher['pid'], signal.SIGCONT)
        time.sleep(.3)
        if same_process(dispatcher):
            raise RuntimeError('Dispatcher termination not observed')
        for rid in running:
            saved = manifest['rows'][rid]
            if not same_process(saved['process']):
                if not complete(run / rid):
                    raise RuntimeError('Worker vanished without completion: ' + rid)
                state['rows'][rid]['status'] = 'TRAINING_COMPLETE'
                continue
            # Freeze this verified worker before counting discarded, uncheckpointed steps.
            os.kill(saved['process']['pid'], signal.SIGSTOP)
            suspended_workers.append(saved['process'])
            actions = run / rid / 'actions.jsonl'
            with actions.open() as handle:
                count = sum(1 for line in handle if line.strip())
            os.kill(saved['process']['pid'], signal.SIGTERM)
            os.kill(saved['process']['pid'], signal.SIGCONT)
            saved['observed_steps_at_stop'] = count
            saved['steps_to_replay'] = count - saved['step']
            state['rows'][rid].update(status='PAUSED_RECOVERABLE', paused=time.time(),
                recovery_checkpoint=saved['checkpoint'], recovery_step=saved['step'],
                steps_to_replay=saved['steps_to_replay'])
        for entry in state['rows'].values():
            if entry['status'] == 'QUEUED':
                entry['status'] = 'HELD_USER_PAUSE'
        state.update(status='PARTIALLY_PAUSED_NATIVE_DRAINING', pause_manifest=str(destination / 'manifest.json'))
        manifest['phase'] = 'STOP_REQUESTED'
        write(run / 'pipeline_state.json', state)
        write(destination / 'manifest.json', manifest)
    except Exception:
        for worker in suspended_workers:
            if same_process(worker):
                os.kill(worker['pid'], signal.SIGCONT)
        if same_process(dispatcher):
            os.kill(dispatcher['pid'], signal.SIGCONT)
        manifest.update(phase='PARTIAL_PAUSE_FAILED', failed_at=time.time())
        write(destination / 'manifest.json', manifest)
        raise
    time.sleep(3)
    remaining = [rid for rid, saved in manifest['rows'].items() if same_process(saved['process'])]
    manifest.update(phase='VERIFIED_PAUSED' if not remaining else 'STOP_UNKNOWN', remaining=remaining,
                    verified_at=time.time(), native_workers='LEFT_RUNNING_UNCHANGED')
    write(destination / 'manifest.json', manifest)
    if remaining:
        raise RuntimeError('Workers still present: ' + repr(remaining))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('snapshot', 'stop'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if run.name != 'phase1_daot_rc4_pure_game_m3_20260917_r2' or run.parent.name != 'runs':
        raise ValueError('Unexpected run scope')
    if args.destination.resolve().parent != run:
        raise ValueError('Pause directory must be directly within the run')
    globals()[args.phase](run, args.destination)


if __name__ == '__main__':
    main()
