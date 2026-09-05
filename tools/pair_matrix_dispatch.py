"""Linux source-only experiment queue. No resume, retries, shell, or process kills."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def _inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def _option(argv, name):
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return None


def validate_manifest(manifest):
    code, root = Path(manifest['code_root']), Path(manifest['output_root'])
    if not code.is_absolute() or not root.is_absolute() or not code.is_dir():
        raise ValueError('code_root and output_root must be absolute; code_root must exist')
    if not Path(manifest['checkpoint']).is_file() or not Path(manifest['python']).is_file():
        raise ValueError('checkpoint and python must exist')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', manifest['run_id']):
        raise ValueError('invalid run_id')
    gpus = manifest['gpus']
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 or g > 7 for g in gpus):
        raise ValueError('gpus must be unique integers in [0,7]')
    if type(manifest['max_active']) is not int or manifest['max_active'] < 1:
        raise ValueError('max_active must be a positive integer')
    for key, default in (('max_new_per_gpu', 1), ('max_per_gpu', 2), ('min_free_mib', 8192)):
        value = manifest.get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if manifest.get('max_new_per_gpu', 1) > manifest.get('max_per_gpu', 2):
        raise ValueError('max_new_per_gpu cannot exceed max_per_gpu')
    seen = set()
    if not manifest['rows']:
        raise ValueError('rows must not be empty')
    for row in manifest['rows']:
        name = row['row_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen or name == 'smoke':
            raise ValueError('row_id must be unique and path safe')
        seen.add(name)
        if type(row.get('assigned_gpu')) is not int or row['assigned_gpu'] not in gpus:
            raise ValueError('each row must have an assigned_gpu from manifest.gpus')
        if not _inside(row['cwd'], code) or not Path(row['cwd']).is_dir():
            raise ValueError('row cwd must exist inside code_root')
        argv = row['argv']
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv) or argv[0] != manifest['python']:
            raise ValueError('row argv must start with manifest.python')
        destination = _option(argv, '--output_dir')
        if destination is None or Path(destination).resolve() != (root / name).resolve():
            raise ValueError('row output_dir must equal output_root/row_id')
        for option in ('--metrics_csv', '--metrics_jsonl'):
            explicit = _option(argv, option)
            if explicit and not _inside(explicit, root / name):
                raise ValueError('metrics paths must remain inside row output_dir')
        if not isinstance(row['environment'], dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in row['environment'].items()):
            raise ValueError('environment must be a string mapping')
    smoke = manifest['smoke_argv']
    if not isinstance(smoke, list) or not smoke or smoke[0] != manifest['python'] or any(not isinstance(x, str) for x in smoke):
        raise ValueError('smoke_argv must start with manifest.python')
    destination = _option(smoke, '--output_dir')
    if destination and Path(destination).resolve() != (root / 'smoke').resolve():
        raise ValueError('smoke output_dir must equal output_root/smoke')


def create_runroot(root):
    # Atomic mkdir is the no-relaunch boundary. Never use exist_ok or resume here.
    Path(root).mkdir()


def _save(root, state):
    state['updated_unix'] = time.time()
    temporary = root / f'.status.{os.getpid()}.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / 'status.json')


def _cmdline(pid):
    try:
        return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', errors='replace').strip()
    except OSError:
        return None


def _is_training(command):
    # Unreadable compute processes count conservatively against the training cap.
    return command is None or bool(re.search(r'(?:train[\w.-]*|torchrun|deepspeed)', command, re.I))


def gpu_snapshot():
    def query(fields, kind):
        result = subprocess.run(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=15)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    devices = query('index,uuid,memory.free', 'gpu')
    gpu = {int(index): {'free_mib': int(free), 'training_pids': set()} for index, _, free in devices}
    indices = {uuid: int(index) for index, uuid, _ in devices}
    for uuid, raw_pid in query('gpu_uuid,pid', 'compute-apps'):
        if uuid in indices and raw_pid.strip().isdigit():
            pid = int(raw_pid)
            if _is_training(_cmdline(pid)):
                gpu[indices[uuid]]['training_pids'].add(pid)
    return gpu


def choose_gpu(gpus, snapshot, active, *, max_new_per_gpu=1, max_per_gpu=2, min_free_mib=8192):
    for gpu in gpus:
        owned = {row['pid'] for row in active if row['gpu'] == gpu}
        if len(owned) >= max_new_per_gpu:
            continue
        info = snapshot.get(gpu)
        if info and info['free_mib'] >= min_free_mib and len(set(info['training_pids']) | owned) < max_per_gpu:
            return gpu
    return None


def choose_pending_row(pending, manifest, snapshot, active):
    for index, row in enumerate(pending):
        gpu = choose_gpu([row['assigned_gpu']], snapshot, active,
                         max_new_per_gpu=manifest.get('max_new_per_gpu', 1),
                         max_per_gpu=manifest.get('max_per_gpu', 2),
                         min_free_mib=manifest.get('min_free_mib', 8192))
        if gpu is not None:
            return index, gpu
    return None, None


def failure_fingerprint(log_text, exit_code):
    errors = re.findall(r'^.*(?:[A-Za-z]+Error|[A-Za-z]+Exception):.*$', log_text, re.M)
    cause = errors[-1].strip() if errors else f'EXIT_{exit_code}'
    cause = re.sub(r'0x[0-9a-fA-F]+|\d+(?:\.\d+)?', '#', cause)
    return hashlib.sha256(cause.encode('utf-8')).hexdigest()[:16]


def validate_training_artifacts(runroot):
    root = Path(runroot)
    checkpoint = root / 'final_ssdg.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False, 'missing_final_ssdg'
    metrics = root / 'metrics_epoch.jsonl'
    try:
        with metrics.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not any(int(row.get('epoch', -1)) == 200 for row in rows):
            return False, 'missing_epoch200_metrics'
    except (OSError, ValueError, TypeError):
        return False, 'invalid_epoch_metrics'
    return True, 'TRAINING_COMPLETE_SOURCE_ONLY'


def _start(argv, cwd, environment, gpu, runroot):
    runroot.mkdir()
    env = dict(os.environ)
    env.update(environment)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
    log = runroot / 'process.log'
    with log.open('xb') as stream:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   shell=False, start_new_session=True)
    return process, dict(pid=process.pid, cwd=str(cwd), gpu=gpu, argv=argv,
                         runroot=str(runroot), log=str(log), status='RUNNING', started_unix=time.time())


def _tail(path):
    with Path(path).open('rb') as stream:
        stream.seek(max(0, Path(path).stat().st_size - 65536))
        return stream.read().decode('utf-8', errors='replace')


def run(manifest_path):
    if os.name != 'posix':
        raise RuntimeError('dispatcher run requires Linux with /proc and flock')
    import fcntl
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    if not _inside(manifest_path, manifest['code_root']):
        raise ValueError('manifest must reside inside readonly code release')
    root = Path(manifest['output_root'])
    create_runroot(root)
    with (root / '.dispatch.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(run_id=manifest['run_id'], manifest=str(Path(manifest_path).resolve()),
                     dispatcher_pid=os.getpid(), code_root=manifest['code_root'], checkpoint=manifest['checkpoint'],
                     output_root=str(root), status='WAITING_FOR_SMOKE_GPU', rows=[], smoke=None,
                     scheduling_limits={key: manifest.get(key, default) for key, default in
                         (('max_active', 24), ('max_new_per_gpu', 1), ('max_per_gpu', 2), ('min_free_mib', 8192))},
                     claim_boundary='SOURCE_ONLY_NO_TARGET_ARTIFACT_COMPLETION_CLAIM')
        active, fingerprints = {}, {}
        pending = list(manifest['rows'])
        state['unlaunched_rows'] = [row['row_id'] for row in pending]
        smoke_done, stopped = False, False
        _save(root, state)
        try:
            while True:
                for name, (process, record) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    record.update(exit_code=code, ended_unix=time.time())
                    if name == 'smoke':
                        smoke_done = code == 0
                        record['status'] = 'SMOKE_PASSED' if smoke_done else 'SMOKE_FAILED'
                        stopped = not smoke_done
                    else:
                        valid, evidence = validate_training_artifacts(record['runroot']) if code == 0 else (False, f'EXIT_{code}')
                        record.update(status=evidence if valid else 'TECHNICAL_FAILURE', artifact_evidence=evidence)
                        if not valid:
                            fingerprint = failure_fingerprint(_tail(record['log']), code) if code else evidence
                            record['failure_fingerprint'] = fingerprint
                            fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
                            stopped |= fingerprints[fingerprint] >= 2
                    del active[name]
                if stopped:
                    state['status'] = 'STOPPED_NEW_ROWS_WAITING_HEALTHY' if active else 'STOPPED_NEW_ROWS'
                elif smoke_done:
                    state['status'] = 'TRAINING' if active else 'WAITING_FOR_GPU'
                if stopped and not active or smoke_done and not pending and not active:
                    if not stopped:
                        state['status'] = 'QUEUE_FINISHED_SOURCE_ONLY'
                    state['unlaunched_rows'] = [row['row_id'] for row in pending]
                    _save(root, state)
                    return 1 if stopped or any(row['status'] == 'TECHNICAL_FAILURE' for row in state['rows']) else 0
                if not stopped and len(active) < manifest['max_active'] and (smoke_done or not active):
                    try:
                        snapshot = gpu_snapshot()
                        state.pop('gpu_poll_error', None)
                        active_records = [record for _, record in active.values()]
                        if smoke_done:
                            row_index, gpu = choose_pending_row(pending, manifest, snapshot, active_records)
                        else:
                            row_index = None
                            gpu = choose_gpu(manifest['gpus'], snapshot, active_records,
                                max_new_per_gpu=manifest.get('max_new_per_gpu', 1),
                                max_per_gpu=manifest.get('max_per_gpu', 2),
                                min_free_mib=manifest.get('min_free_mib', 8192))
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        state['gpu_poll_error'] = str(exc)
                        gpu = None
                    if gpu is not None and (not smoke_done or pending):
                        row = pending.pop(row_index) if smoke_done else dict(row_id='smoke', argv=manifest['smoke_argv'],
                                                                  cwd=manifest['code_root'], environment={})
                        state['unlaunched_rows'] = [item['row_id'] for item in pending]
                        try:
                            process, record = _start(row['argv'], row['cwd'], row['environment'], gpu, root / row['row_id'])
                            record['row_id'] = row['row_id']
                            active[row['row_id']] = (process, record)
                        except OSError as exc:
                            record = dict(row_id=row['row_id'], argv=row['argv'], cwd=row['cwd'], gpu=gpu,
                                          runroot=str(root / row['row_id']), status='TECHNICAL_FAILURE', error=str(exc))
                            key = f'SPAWN_{type(exc).__name__}'
                            fingerprints[key] = fingerprints.get(key, 0) + 1
                            stopped |= fingerprints[key] >= 2 or row['row_id'] == 'smoke'
                        if row['row_id'] == 'smoke':
                            state['smoke'] = record
                            state['status'] = 'SMOKE_RUNNING'
                        else:
                            state['rows'].append(record)
                        _save(root, state)
                        # Fill other GPUs immediately; a launched GPU is reserved before CUDA initialization.
                        continue
                _save(root, state)
                time.sleep(25)
        except BaseException as exc:
            state.update(status='DISPATCHER_INTERRUPTED_CHILDREN_NOT_KILLED', error=f'{type(exc).__name__}: {exc}')
            _save(root, state)
            raise


def status(root):
    state = json.loads((Path(root) / 'status.json').read_text(encoding='utf-8'))
    for row in ([state['smoke']] if state.get('smoke') else []) + state.get('rows', []):
        if 'pid' in row:
            row['observed_cmdline'] = _cmdline(row['pid'])
            try:
                row['observed_cwd'] = str(Path(f"/proc/{row['pid']}/cwd").resolve(strict=True))
            except OSError:
                row['observed_cwd'] = None
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'status', 'snapshot'])
    parser.add_argument('path', help='manifest for run; existing output_root for status/snapshot')
    options = parser.parse_args()
    if options.action == 'run':
        raise SystemExit(run(options.path))
    print(json.dumps(status(options.path), indent=2))
