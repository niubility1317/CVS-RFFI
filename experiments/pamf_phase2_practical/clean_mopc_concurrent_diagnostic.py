"""Repeat one saved-B support diagnostic under bounded GPU concurrency."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .clean_scheduler import choose
from .io import atomic, require, write
from .runner import occupancy


def result_status(path, code):
    require(code == 0, 'Diagnostic process failed')
    events = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    results = [e for e in events if e.get('event') == 'DIAGNOSTIC_RESULT']
    require(len(results) == 1 and results[0].get('status') in ('COMPLETE', 'NONFINITE') and
            results[0].get('query_read') is False and results[0].get('truth_read') is False and
            results[0].get('predictions_written') is False, 'Invalid diagnostic result')
    if results[0]['status'] == 'COMPLETE':
        require(results[0].get('repeats') == 3, 'Incomplete diagnostic repetitions')
    return results[0]['status']


def run(runtime, job, output, k=10, n=20, variant='original_uninstrumented'):
    require(k in (1, 5, 10, 20) and n in (2, 5, 10, 20), 'Invalid K/N')
    require(variant in ('original_uninstrumented', 'clip1_uninstrumented', 'clip10_uninstrumented',
                        'clip1000_uninstrumented'),
            'Invalid diagnostic variant')
    root = Path(output).resolve()
    require(not root.exists(), 'Diagnostic root already exists')
    root.mkdir(parents=True)
    (root/'logs').mkdir()
    write(root/'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), runtime=runtime, job=job,
                                  gpus=[5,6,7], max_per_gpu=7, minimum_free_MiB=6000, k=k, n=n, variant=variant))
    pending = [dict(id=f'diagnostic-{i:02d}') for i in range(21)]
    active = {}; finished = []; failures = []
    while pending or active:
        for key, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['stream'].close()
            try:
                status = result_status(item['path'], code)
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
                status = 'FAILED'
                failures.append(dict(id=key, reason=str(exc)))
            if status == 'NONFINITE':
                failures.append(dict(id=key, reason='NONFINITE'))
            finished.append(dict(id=key, gpu=item['gpu'], status=status))
            del active[key]
        if failures:
            pending.clear()
        else:
            try:
                admitted = choose(pending, {k: dict(pid=v['process'].pid, gpu=v['gpu']) for k,v in active.items()},
                                  occupancy(), [5,6,7], 6000, 7, worker_reserve=3000, one_per_snapshot=True)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                failures.append(dict(id='occupancy', reason=str(exc)))
                admitted = []
            for task, gpu in admitted:
                path = root/'logs'/(task['id']+'.jsonl')
                command = [sys.executable, '-u', '-m', __package__+'.clean_mopc_savedb_diagnostic',
                           '--runtime', runtime, '--job', job, '--k', str(k), '--n', str(n),
                           '--device', 'cuda:0', '--variant', variant, '--repeats', '3']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                stream = None
                try:
                    stream = path.open('x', encoding='utf-8')
                    process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT)
                except OSError as exc:
                    if stream is not None:
                        stream.close()
                    failures.append(dict(id=task['id'], reason=str(exc)))
                    break
                active[task['id']] = dict(process=process, stream=stream, path=path, gpu=gpu)
                pending.remove(task)
        if failures:
            pending.clear()
        atomic(root/'state.json', dict(status='DRAINING' if failures else 'RUNNING',
               pending=len(pending), finished=finished, failures=failures,
               active=[dict(id=k, pid=v['process'].pid, gpu=v['gpu']) for k,v in active.items()],
               query_read=False, truth_read=False))
        if pending or active:
            time.sleep(1)
    write(root/'complete.json', dict(status='NONFINITE_OR_FAILED' if failures else 'COMPLETE',
          finished=finished, failures=failures, query_read=False, truth_read=False))
    atomic(root/'state.json', dict(status='NONFINITE_OR_FAILED' if failures else 'COMPLETE',
           pending=0, active=[], finished=finished, failures=failures, query_read=False, truth_read=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'job', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--variant', default='original_uninstrumented')
    a = p.parse_args()
    run(a.runtime, a.job, a.output, a.k, a.n, a.variant)


if __name__ == '__main__':
    main()
