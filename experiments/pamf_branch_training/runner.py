"""One owner; bounded shared-GPU dispatch; no mutation of other processes."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read, write


def worker(c, kind):
    if kind in ('score', 'weak_score'):
        if kind == 'score':
            from experiments.pamf_paper_completion.evaluation import score
        else:
            from .weak_evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        from .smoke import smoke
        return smoke(c)
    if kind == 'train':
        from .source import train
        return train(c)
    if kind == 'predict':
        from experiments.pamf_paper_completion.evaluation import predict
        return predict(c)
    if kind == 'weak_predict':
        from .weak_evaluation import predict
        return predict(c)
    raise ValueError('Unknown execution stage')


def launch(rt, job, gpu):
    path = Path(rt['run_root']) / 'logs' / (job['id'] + '.log')
    cmd = [rt['python'], '-u', '-m', 'experiments.pamf_branch_training.runner',
           'worker', '--kind', job['kind'], '--config', job['config']]
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    with path.open('x') as log:
        process = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, dict(status='RUNNING', pid=process.pid, gpu=gpu, cwd=rt['release_root'],
                         argv=cmd, log=str(path), started_at=time.time())


def admits(gpu, active, required_mib):
    from experiments.pamf_repl_completion.runner import capacity
    own = [(p, item) for p, item in active.values() if item['gpu'] == gpu]
    if len(own) >= 2:
        return False
    cuda, free = capacity(gpu)
    # Reserve enough memory for workers spawned but not yet visible to CUDA.
    pending = sum(p.pid not in cuda for p, _ in own)
    return free >= required_mib * (1 + pending)


def choose_gpu_job(available, prioritize_prediction=False):
    preferred = 'predict' if prioritize_prediction else 'train'
    return next((job for job in available if job['kind'] == preferred), available[0])


def analyze(rt):
    import csv
    root = Path(rt['run_root'])
    sources = read(root / 'sources.json')
    records = []
    for rid, entry in sources.items():
        for kind, key in (('practical', 'scores'), ('weak', 'weak_scores')):
            folder = Path(entry[key])
            status = 'SCORED_COMPLETE' if kind == 'practical' else 'SCORED_WEAK_COMPLETE'
            data = read(folder / 'scores.json')
            if read(folder / 'complete.json')['status'] != status or not data['truth_last']:
                raise ValueError('Complete independent scores required')
            for r in data['results']:
                if r['dimension'] == 'overall':
                    records.append(dict(row=rid, family=kind, test=r['test'], readout=r['readout'],
                                        view=r['view'], count=r['count'], accuracy=r['accuracy'], macro_f1=r['macro_f1']))
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    with (out / 'overall.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(out / 'complete.json', dict(status='ANALYZED', rows=len(sources), target_feedback_forbidden=True,
                                     results=len(records), n_seeds=1, paired_baselines=True))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done, ready, block_failed_dependencies
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    from .prepare import prepare
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    state = dict(status='PREPARING', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
                 code_commit=rt['code_commit'], owner=rt['launch_owner'], jobs={})
    atomic(root / 'pipeline_state.json', state)
    try:
        jobs = prepare(rt)
    except Exception as exc:
        state.update(status='FAILED_PREPARATION', error=repr(exc))
        atomic(root / 'pipeline_state.json', state)
        raise
    by = {j['id']: j for j in jobs}
    if len(by) != len(jobs):
        raise ValueError('Duplicate job identity')
    state['jobs'] = {j['id']: dict(status='QUEUED') for j in jobs}
    active = {}
    state['status'] = 'SMOKE_WAITING_CAPACITY'
    atomic(root / 'pipeline_state.json', state)
    while True:
        gpu = next((g for g in rt['allowed_gpus'] if admits(g, active, 6000)), None)
        if gpu is not None:
            break
        time.sleep(15)
    process, item = launch(rt, dict(id='smoke', kind='smoke', config=rt['runtime_file']), gpu)
    state.update(status='SMOKE_RUNNING', smoke=item)
    atomic(root / 'pipeline_state.json', state)
    code = process.wait()
    ok = code == 0 and (root / 'smoke.json').is_file() and read(root / 'smoke.json')['status'] == 'PASS'
    state['smoke'].update(exit_code=code, status='PASS' if ok else 'FAILED')
    if not ok:
        state['status'] = 'FAILED_NO_QUERY_SMOKE'
        atomic(root / 'pipeline_state.json', state)
        return
    peak = max(x['peak_allocated_bytes'] for x in read(root / 'smoke.json')['checks'])
    required_mib = max(6000, int(peak / 1024**2) + 2000)
    state.update(status='RUNNING', admission_mib=required_mib)
    failed, nonblocking_failed = [], []
    while True:
        for key, (process, item) in list(active.items()):
            if process.poll() is None:
                continue
            ok = process.returncode == 0 and done(by[key])
            state['jobs'][key] = dict(item, status='COMPLETE' if ok else 'FAILED',
                                      ended_at=time.time(), exit_code=process.returncode)
            if not ok:
                (nonblocking_failed if by[key].get('nonblocking') is True else failed).append(key)
            del active[key]
        block_failed_dependencies(jobs, state, [])
        pending = [j for j in jobs if state['jobs'][j['id']]['status'] == 'QUEUED']
        candidates = [j for j in pending if ready(j)] if not failed else []
        scorer = next((j for j in candidates if j['kind'] in ('score', 'weak_score')), None)
        if scorer and not any(item['gpu'] is None for _, item in active.values()) and not foreign_CPU_score({p.pid for p, _ in active.values()}):
            process, item = launch(rt, scorer, None)
            active[scorer['id']] = (process, item)
            state['jobs'][scorer['id']] = item
            candidates.remove(scorer)
        for gpu in rt['allowed_gpus']:
            if not admits(gpu, active, required_mib):
                continue
            available = [j for j in candidates if j['kind'] not in ('score', 'weak_score')]
            if not available:
                continue
            job = choose_gpu_job(available, rt.get('prioritize_prediction', False))
            process, item = launch(rt, job, gpu)
            active[job['id']] = (process, item)
            state['jobs'][job['id']] = item
            candidates.remove(job)
        state.update(updated_at=time.time(), counts=dict(Counter(v['status'] for v in state['jobs'].values())),
                     failed_jobs=failed, nonblocking_failed_jobs=nonblocking_failed, stop_new_dispatch=bool(failed))
        atomic(root / 'pipeline_state.json', state)
        if not active and (not pending or failed):
            break
        time.sleep(15)
    if failed or any(v['status'] != 'COMPLETE' for v in state['jobs'].values()):
        state['status'] = 'PARTIAL_TECHNICAL_FAILURE'
    else:
        state['status'] = 'ANALYZING'
        atomic(root / 'pipeline_state.json', state)
        analyze(rt)
        state['status'] = 'ANALYZED'
    atomic(root / 'pipeline_state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    dispatch(read(args.config)) if args.action == 'dispatch' else worker(read(args.config), args.kind)
