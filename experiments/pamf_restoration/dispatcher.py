"""N607 CPU controller; at most one run-owned GPU worker per card.

No process termination, retries, query-driven changes or writes to any other
run. A technical failure leaves healthy active candidates running to closure.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .models import NAMES
from .data import IQSource, prepare_cache, write_json


def progress(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def capacity():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                         '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in gpu.strip().splitlines():
        index, uuid, free = [v.strip() for v in line.split(',')]
        result[int(index)] = dict(uuid=uuid, free_mb=int(free), pids=set())
    for line in processes.strip().splitlines():
        if not line.strip() or ',' not in line:
            continue
        pid, uuid = [v.strip() for v in line.split(',')]
        for row in result.values():
            if row['uuid'] == uuid:
                row['pids'].add(int(pid))
    return result


def available_cards(state, allowed, active):
    reserved = {row['gpu'] for row in active.values()}
    return [gpu for gpu in allowed if gpu not in reserved and gpu in state and
            state[gpu]['free_mb'] >= 12000 and len(state[gpu]['pids']) < 2]


def start_child(mode, candidate, gpu, args, root):
    command = [sys.executable, '-u', '-m', 'experiments.pamf_restoration.runner', mode,
               '--source-cache', str(args.source_cache), '--output', str(args.output),
               '--frozen-bundle', str(args.frozen_bundle), '--candidate', candidate,
               '--expected-commit', args.expected_commit]
    if mode == 'confirm':
        command.extend(['--target-capsule', str(args.target_capsule)])
    log_path = args.output/'logs'/(candidate+('-confirmation' if mode == 'confirm' else '')+'.log')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                       PYTHONUNBUFFERED='1')
    with log_path.open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(candidate=candidate, phase=mode, pid=child.pid, gpu=gpu, cwd=str(root),
                  argv=command, log=str(log_path), started=time.time())
    return child, record


def dispatch(args):
    import fcntl
    from .runner import ROOT, code_commit, array_tensor, source_decision
    from .frozen import load_readers
    allowed = [int(v) for v in args.gpus.split(',')]
    if allowed != d.configuration()['GPUs'] or len(set(allowed)) != 3:
        raise ValueError('Preregistered three-card resource assignment differs')
    if code_commit() != args.expected_commit:
        raise ValueError('Pinned release commit differs')
    for field in ('output', 'source_cache', 'target_capsule', 'frozen_bundle'):
        setattr(args, field, getattr(args, field).resolve())
    if args.output.exists():
        raise FileExistsError('Immutable run root exists; reconcile original state')
    args.output.mkdir(parents=True)
    (args.output/'logs').mkdir()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    write_json(args.output/'resolved_config.json', d.configuration())
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv,
               commit=args.expected_commit, controller='CPU only', physical_gpus=allowed,
               launch_owner='codex/root/pamf-restoration', started=time.time()))
    try:
        # First launcher step: real trained F weights, original data contract,
        # two real source IQ records, no TX label/query read. Continue on PASS.
        readers = load_readers(args.frozen_bundle, args.source_cache, 'cpu')
        source = IQSource(args.source_cache, 'L_s')
        x = array_tensor(source.iq[:2], 'cpu')
        with torch.no_grad():
            if any(not torch.isfinite(model(x)).all() for model in readers.values()):
                raise ValueError('Real F checkpoint no-query smoke failed')
        write_json(args.output/'reference_smoke.json', dict(status='PASS', readers=list(readers),
                   source_samples=2, target_reads=0, TX_label_reads=0,
                   exact_physical_roles=True, AsF_full_scratch20_own10_lineage=True))
        del readers, source, x
        print('REFERENCE_CHECKPOINT_SMOKE_PASS', flush=True)
        prepare_cache(args.source_cache, args.output/'cache', d.TRAIN_L_PER_SESSION,
                      d.TRAIN_U_PER_SESSION, d.VALIDATION_PER_SESSION, d.DATA_SEED,
                      workers=d.configuration()['data_generation_workers'])
        pending = list(NAMES)
        active, children, completed, failed, launches = {}, {}, [], [], []
        shared_lock = args.output.parent/'receiver_residual_capacity4.lock'
        while pending or active:
            for candidate, child in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                if code != 0 or not (args.output/candidate/'metrics.json').is_file():
                    failed.append(dict(candidate=candidate, exit_code=code, pid=child.pid))
                else:
                    completed.append(candidate)
                del active[candidate]
                del children[candidate]
            if pending and not failed:
                with shared_lock.open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    slots = available_cards(capacity(), allowed, active)
                    if slots:
                        gpu, candidate = slots[0], pending.pop(0)
                        child, record = start_child('worker', candidate, gpu, args, ROOT)
                        children[candidate], active[candidate] = child, record
                        launches.append(record)
                        progress(args.output/'worker_launches.json', dict(rows=launches))
                        # Keep the existing shared capacity lock through CUDA
                        # initialization so another cooperating launcher sees
                        # this process in nvidia-smi before reserving a slot.
                        deadline = time.monotonic()+45
                        while child.poll() is None and time.monotonic() < deadline:
                            if child.pid in capacity()[gpu]['pids']:
                                break
                            time.sleep(.5)
                    fcntl.flock(lock, fcntl.LOCK_UN)
            progress(args.output/'queue_state.json', dict(active=list(active.values()), pending=pending,
                     completed=completed, failed=failed, updated=time.time()))
            if failed and not active:
                break
            if pending or active:
                time.sleep(2)
        if failed:
            write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', failures=failed,
                       unstarted=pending, completed=completed, healthy_rows_preserved=True,
                       no_automatic_retry=True))
            raise RuntimeError('Technical failure; preserve all artifacts and unstarted rows')
        decision = source_decision(args.output)
        candidate = decision['selected']
        while True:
            with shared_lock.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                slots = available_cards(capacity(), allowed, {})
                if slots:
                    child, record = start_child('confirm', candidate, slots[0], args, ROOT)
                    launches.append(record)
                    progress(args.output/'worker_launches.json', dict(rows=launches))
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(5)
        while child.poll() is None:
            progress(args.output/'queue_state.json', dict(active=[record], pending=[], completed=completed,
                     failed=[], phase='TARGET_CONFIRMATION', source_qualified=decision['source_qualified'],
                     selected=candidate, updated=time.time()))
            time.sleep(5)
        target_done = args.output/'target_confirmation/complete.json'
        if child.returncode != 0 or not target_done.is_file():
            raise RuntimeError('Frozen target confirmation failed; preserve predictions and do not retry')
        write_json(args.output/'complete.json', dict(status='ANALYZED', source_rows=completed,
                   selected=candidate, source_qualified=decision['source_qualified'],
                   negative_confirmation=decision['negative_confirmation'], identity_accuracy_measured=True,
                   target_confirmation_complete=True, no_target_feedback=True))
        progress(args.output/'queue_state.json', dict(active=[], pending=[], completed=completed,
                 failed=[], phase='ANALYZED', updated=time.time()))
    except BaseException as error:
        if not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
                       traceback=traceback.format_exc(), automatic_retry=False, artifacts_preserved=True))
        raise
