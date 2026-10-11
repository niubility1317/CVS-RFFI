"""Three-seed source-only diagnostic; no training or target evaluation entry."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

from . import design as d
from .dispatch import choose_gpu
from experiments.cvs_multi_state_action.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock, process

RUN = '20261011-phase1-stage2-counterfactual-source-manysig-m3-r01'
BASE = Path(d.PROJECT) / 'runs' / RUN
OWNER = d.OWNER


def event(status, **values):
    with lock(BASE / 'registration.lock'):
        value = dict(at=time.time(), status=status, target_access=False, **values)
        with (BASE / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(value) + '\n')
        path = BASE / 'experiment.json'
        spec = d.read(path)
        if 'seed' in values:
            for row in spec['rows']:
                if row['seeds']['model'] == values['seed']:
                    row.update(status=status, latest_event=value)
        else:
            spec['status'] = status
        d.write(path, spec)


def validate_result(path, seed):
    value = d.read(path)
    if (value['status'] != 'SOURCE_COUNTERFACTUAL_AUDIT_COMPLETE' or value['seed'] != seed
            or value['count'] != 6300 or value['target_access'] or not value['model_state_restored']):
        raise ValueError('Source counterfactual diagnostic result contract differs')
    return value


def worker(seed):
    if seed not in d.SEEDS:
        raise ValueError('Unregistered model seed')
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    stable = 0
    while stable < 3:
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            observed = capacity()[gpu]
            owners = observed['pids'] | {os.getpid()}
            ready = len(owners) <= 4 and observed['free_mb'] >= 12000
            stable = stable + 1 if ready else 0
            d.write(BASE / 'capacity_leases' / ('s' + str(seed) + '.json'),
                dict(pid=os.getpid(), gpu=gpu, observed_pids=sorted(owners), stable_checks=stable,
                     max_processes_per_gpu=4, ready=stable == 3))
        if stable < 3:
            time.sleep(1 if ready else 5)
    out = BASE / ('s' + str(seed)) / 'audit.json'
    old = sys.argv
    sys.argv = ['source_counterfactual_audit', '--seed', str(seed), '--output', str(out), '--device', 'cuda:0']
    try:
        runpy.run_module('experiments.cvs_stage2_decouple.source_counterfactual_audit', run_name='__main__')
        validate_result(out, seed)
        event('ANALYZED', seed=seed, artifact=str(out))
    except Exception as exc:
        d.write(out.parent / 'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
        event('FAILED', seed=seed, error=repr(exc))
        raise
    finally:
        sys.argv = old


def run():
    with lock(d.ROOT / 'counterfactual_owner.lock', nonblocking=True):
        BASE.mkdir(parents=True, exist_ok=False)
        (BASE / 'logs').mkdir()
        for name in ('experiment.json', 'report.md', 'events.jsonl'):
            shutil.copy2(d.ROOT / 'counterfactual_registration' / name, BASE / name)
        spec = d.read(BASE / 'experiment.json')
        if spec['run_id'] != RUN or {r['seeds']['model'] for r in spec['rows']} != set(d.SEEDS):
            raise ValueError('Source-only preregistration differs')
        commit = (d.ROOT / 'release_commit.txt').read_text().strip()
        spec['code']['commit'] = commit
        d.write(BASE / 'experiment.json', spec)
        actual = process(os.getpid())
        d.write(BASE / 'controller.json', dict(pid=os.getpid(), start_ticks=actual['start_ticks'],
            cwd=str(d.ROOT), argv=sys.argv, launch_owner=OWNER, commit=commit, target_access=False))
        event('RUNNING', rows=3, identity_updates=0)
        pending, active, completed, failures, receipts = list(d.SEEDS), {}, [], [], []
        try:
            while pending or active:
                for seed, child in list(active.items()):
                    rc = child.poll()
                    if rc is None:
                        continue
                    artifact = BASE / ('s' + str(seed)) / 'audit.json'
                    if rc or not artifact.is_file():
                        failures.append(dict(seed=seed, exit_code=rc, artifact_exists=artifact.is_file()))
                    else:
                        validate_result(artifact, seed)
                        completed.append(seed)
                    del active[seed]
                while pending and not failures:
                    with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
                        gpu = choose_gpu(capacity())
                        if gpu is None:
                            break
                        seed = pending.pop(0)
                        command = [sys.executable, '-u', '-m', 'experiments.cvs_stage2_decouple.counterfactual_control', '--worker', str(seed)]
                        log = BASE / 'logs' / ('s' + str(seed) + '.log')
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                            PYTHONPATH=str(d.ROOT) + os.pathsep + str(d.ROOT / 'code'),
                            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                        with log.open('x', encoding='utf-8') as stream:
                            child = subprocess.Popen(command, cwd=d.ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        actual = process(child.pid)
                        if actual is None:
                            raise RuntimeError('Source audit process missing after launch')
                        receipt = dict(seed=seed, pid=child.pid, start_ticks=actual['start_ticks'],
                            cwd=str(d.ROOT), argv=command, gpu=gpu, log=str(log))
                        receipts.append(receipt)
                        active[seed] = child
                        d.write(BASE / 'launch.json', dict(rows=receipts))
                    event('RUNNING', seed=seed, pid=child.pid, gpu=gpu, log=str(log))
                d.write(BASE / 'queue_state.json', dict(active=list(active), pending=pending,
                    completed=completed, failures=failures, per_gpu_limit=4, target_access=False))
                if failures and not active:
                    raise RuntimeError('Source audit technical failure preserved: ' + repr(failures))
                if pending or active:
                    time.sleep(5)
            results = [validate_result(BASE / ('s' + str(seed)) / 'audit.json', seed) for seed in d.SEEDS]
            d.write(BASE / 'summary.json', dict(status='SOURCE_COUNTERFACTUAL_AUDIT_COMPLETE',
                seeds=list(d.SEEDS), rows=results, target_access=False, identity_updates=0))
            d.write(BASE / 'completion.json', dict(status='ANALYZED', rows=3, source_count_per_seed=6300,
                target_access=False, identity_updates=0, commit=commit))
            event('ANALYZED', rows=3, identity_updates=0)
        except Exception as exc:
            d.write(BASE / 'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
            event('FAILED', error=repr(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    worker(args.worker) if args.worker is not None else run()
