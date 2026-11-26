"""Keep already-admitted WiSig DURA rows and admit the remaining rows by memory."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep


def process(pid):
    folder = Path('/proc') / str(pid)
    try:
        status = (folder / 'status').read_text()
        state = next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
        if state == 'Z':
            return None
        return dict(pid=pid, state=state, uid=folder.stat().st_uid, cwd=(folder / 'cwd').resolve(),
                    argv=os.fsdecode((folder / 'cmdline').read_bytes()).rstrip('\0').split('\0'))
    except FileNotFoundError:
        return None


def cuda_device(pid):
    env = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
    key = b'CUDA_VISIBLE_DEVICES='
    return int(next(item[len(key):] for item in env if item.startswith(key)))


def available(allowed):
    value = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [tuple(map(int, line.split(','))) for line in value.splitlines()]
    return [(gpu, free) for gpu, free in sorted(rows, key=lambda row: -row[1])
            if gpu in allowed and free >= 12288]


def snapshots(rt, old):
    from experiments.pamf_unrolled_restore.wisig_dura_hardmix import ARMS, CASES

    root, release = map(Path, (rt['run_root'], rt['release_root']))
    children = Path('/proc') / str(old['pid']) / 'task' / str(old['pid']) / 'children'
    live = {}
    for raw_pid in children.read_text().split():
        pid = int(raw_pid)
        child = process(pid)
        if child is None:
            continue
        expected = [rt['python'], '-u', '-m', rt['runner'], 'arm', str(release / 'runtime.json')]
        if (child['uid'] != os.getuid() or child['cwd'] != release or child['argv'][:-1] != expected
                or child['argv'][-1] not in ARMS):
            raise ValueError('The old scheduler owns a non-row process. Leave it in charge.')
        arm = child['argv'][-1]
        gpu = cuda_device(pid)
        if arm in live or gpu in (item['gpu'] for item in live.values()):
            raise ValueError('Expected a single owner per row and physical GPU')
        live[arm] = dict(pid=pid, gpu=gpu)

    finished, pending = [], []
    for arm in ARMS:
        if arm in live:
            continue
        frozen = (root / 'models' / arm / 'complete.json').is_file()
        predicted = all((root / arm / 'predictions' / case / 'complete.json').is_file()
                        for case in CASES.values())
        if frozen and predicted:
            finished.append(arm)
        elif (root / 'logs' / f'arm-{arm}.log').exists():
            raise ValueError(f'{arm} has launch evidence without a live owner or completed output')
        else:
            pending.append(arm)
    return live, finished, pending


def handoff(rt):
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_unrolled_restore.wisig_dura_hardmix import CASES, RUN, complete, launch

    root, release = map(Path, (rt['run_root'], rt['release_root']))
    state = root / 'state.json'
    if (rt['run_id'] != RUN or read(state)['status'] != 'SOURCE_FIT_THEN_ROW_PREDICT'
            or sum(path.is_file() for path in (root / 'source').glob('*/complete.json')) != 30):
        raise ValueError('Handoff requires this run with its completed source cache and active scheduler')
    old = process(read(release / 'submit.json')['pid'])
    expected = [rt['python'], '-u', '-m', rt['runner'], 'dispatch', str(release / 'runtime.json')]
    if old is None or (old['uid'], old['cwd'], old['argv']) != (os.getuid(), release, expected):
        raise ValueError('The original owner identity changed')
    pid = old['pid']
    os.kill(pid, signal.SIGSTOP)
    assigned = {}
    try:
        deadline = monotonic() + 3
        while process(pid)['state'] != 'T':
            if monotonic() >= deadline:
                raise RuntimeError('Original scheduler did not stop for ownership handoff')
            sleep(.02)
        adopted, finished, pending = snapshots(rt, old)
        if not pending:
            print('ALL_ROWS_ALREADY_ADMITTED', flush=True)
            return
        occupied = {item['gpu'] for item in adopted.values()}
        if not any(gpu not in occupied for gpu, _ in available(rt['allowed_gpus'])):
            raise RuntimeError('No memory-qualified GPU available at handoff')
        write(root / 'gpu_memory_handoff.json', dict(status='HANDOFF', controller_pid=os.getpid(),
              previous_controller_pid=pid, adopted=adopted, finished=finished, pending=pending,
              minimum_free_mib=12288, max_workers=rt['max_workers']))
        os.kill(pid, signal.SIGTERM)
    finally:
        if process(pid) is not None:
            os.kill(pid, signal.SIGCONT)

    write(state, dict(status='SOURCE_FIT_THEN_ROW_PREDICT', allocation='gpu_memory_only'), exclusive=False)
    try:
        while pending or adopted or assigned:
            for arm, child in list(adopted.items()):
                identity = process(child['pid'])
                if identity is not None and identity['argv'][-2:] == [str(release / 'runtime.json'), arm]:
                    continue
                if not all((root / arm / 'predictions' / case / 'complete.json').is_file()
                           for case in CASES.values()):
                    raise RuntimeError(f'Adopted {arm} ended before complete prediction')
                del adopted[arm]
            for arm, (gpu, child) in list(assigned.items()):
                status = child.poll()
                if status is None:
                    continue
                if status:
                    raise RuntimeError(f'{arm} exited {status}. See the run-owned logs.')
                del assigned[arm]
            if pending and len(adopted) + len(assigned) < rt['max_workers']:
                occupied = {value['gpu'] for value in adopted.values()} | {value[0] for value in assigned.values()}
                for gpu, free in available(rt['allowed_gpus']):
                    if not pending or len(adopted) + len(assigned) >= rt['max_workers']:
                        break
                    if gpu not in occupied:
                        arm = pending.pop(0)
                        print('HARDMIX_GPU_ADMISSION', arm, gpu, free, flush=True)
                        assigned[arm] = (gpu, launch(rt, 'arm', arm, gpu))
            if pending or adopted or assigned:
                sleep(10)
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        complete(launch(rt, 'score', 'cpu'), 'independent truth-last scoring')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


if __name__ == '__main__':
    runtime = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(runtime.parent))
    handoff(json.loads(runtime.read_text()))
