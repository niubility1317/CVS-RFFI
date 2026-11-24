"""One owner for source training, frozen prediction and independent CPU scoring."""

import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.pipeline import block_failed_dependencies, done, ready
from experiments.pamf_repl_view_shuffle.runner import activate, eligible, resources
from experiments.practical_full_2x2.cache import atomic

from .plan import ARMS, ORIGINAL_ROW, ORIGINAL_RUN, RUN, SEED, TASKS, row_id, source_config


def prepare(runtime):
    context = read(runtime['context'])
    root = Path(runtime['run_root'])
    original = context['original_prediction']
    jobs = []
    for arm in ARMS:
        row = row_id(arm)
        train = source_config(runtime, context, arm)
        folder = root / 'readouts' / row
        predict = dict(
            run_id=RUN, row_id=row, model_seed=SEED, source_output=train['output_root'],
            source_contract=train['source_contract'], classes=context['classes'],
            readouts=['raw'], tests={key: {'views_root': context['tests'][key]['views_root']} for key in TASKS},
            output_root=str(folder / 'predictions'), frozen_sources=runtime['frozen_sources'])
        score = dict(run_id=RUN, row_id=row, model_seed=SEED, classes=context['classes'],
                     readouts=['raw'], tests=context['tests'], source_days=train['source_days'],
                     prediction_output=predict['output_root'], output_root=str(folder / 'scores'),
                     reference_prediction=dict(run_id=ORIGINAL_RUN, row_id=ORIGINAL_ROW,
                         classes=context['classes'], readouts=['raw'], tests=predict['tests'],
                         prediction_output=original))
        for kind, c, status in (
            ('train', train, 'SOURCE_FROZEN'),
            ('predict', predict, 'ALL_PREDICTIONS_COMPLETE'),
            ('score', score, 'SCORED_COMPLETE'),
        ):
            key = kind + '__' + row
            path = root / 'configs' / (key + '.json')
            dependency = [] if kind == 'train' else [jobs[-1]['completion']]
            write(path, c)
            jobs.append(dict(id=key, kind=kind, arm=arm, config=str(path), dependencies=dependency,
                             completion=str(Path(c['output_root']) / 'complete.json'),
                             successful_statuses=[status]))
    write(root / 'jobs.json', jobs)
    return jobs


def launch(runtime, key, kind, config, gpu):
    cwd = runtime['release_root']
    argv = [runtime['python'], '-u', '-m', __spec__.name, 'worker', '--kind', kind, '--config', config]
    env = dict(os.environ, PYTHONPATH=cwd, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    path = Path(runtime['run_root']) / 'logs' / (key + '.log')
    with path.open('x') as stream:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return child, dict(status='RUNNING', pid=child.pid, gpu=gpu, cwd=cwd, argv=argv, log=str(path))


def failure_label(log):
    lines = Path(log).read_text(encoding='utf-8', errors='replace').splitlines()
    return lines[-1] if lines else 'Empty worker log'


def dispatch(runtime):
    from experiments.pamf_repl_completion.runner import foreign_CPU_score

    root = Path(runtime['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    jobs = prepare(runtime)
    by_id = {job['id']: job for job in jobs}
    state_path = root / 'pipeline_state.json'
    state = dict(status='WAITING_FOR_SMOKE_GPU', pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
                 code_commit=runtime['code_commit'], jobs={j['id']: {'status': 'QUEUED'} for j in jobs})
    atomic(state_path, state)
    print('WAITING_FOR_SMOKE_GPU', runtime['run_id'], flush=True)
    while True:
        available = resources()
        gpu = next((g for g in runtime['allowed_gpus'] if eligible(available[g], 0, 1)), None)
        if gpu is not None:
            break
        state['updated_at'] = time.time()
        state['gpu_availability'] = {str(g): dict(free_mib=available[g]['free'],
                                                  compute_slots=len(available[g]['pids']),
                                                  foreign_or_unknown=available[g]['blocked'])
                                     for g in runtime['allowed_gpus']}
        atomic(state_path, state)
        time.sleep(15)
    child, details = launch(runtime, 'smoke', 'smoke', runtime['runtime_file'], gpu)
    state.update(status='SMOKE_RUNNING', smoke=details)
    atomic(state_path, state)
    if child.wait() or read(root / 'smoke.json')['status'] != 'PASS':
        state.update(status='FAILED_NO_QUERY_SMOKE')
        atomic(state_path, state)
        return
    state['smoke']['status'] = 'PASS'

    active = {}
    failures = Counter()
    while True:
        for key, (child, details) in list(active.items()):
            status = child.poll()
            if status is None:
                continue
            success = status == 0 and done(by_id[key])
            entry = state['jobs'][key] = dict(details, status='COMPLETE' if success else 'FAILED',
                                              exit_code=status)
            if not success:
                label = failure_label(details['log'])
                failures[label] += 1
                entry['failure'] = label
            del active[key]
        block_failed_dependencies(jobs, state, [])
        systemic = any(n >= 2 for n in failures.values())
        queued = [j for j in jobs if state['jobs'][j['id']]['status'] == 'QUEUED']
        candidates = [j for j in queued if ready(j)] if not systemic else []
        scores = [j for j in candidates if j['kind'] == 'score']
        if (scores and not any(by_id[key]['kind'] == 'score' for key in active)
                and not foreign_CPU_score({p.pid for p, _ in active.values()})):
            job = scores[0]
            active[job['id']] = launch(runtime, job['id'], job['kind'], job['config'], None)
            state['jobs'][job['id']] = active[job['id']][1]
        gpu_jobs = sorted((j for j in candidates if j['kind'] != 'score'),
                          key=lambda j: j['kind'] != 'predict')
        available = resources() if gpu_jobs else {}
        for gpu in runtime['allowed_gpus']:
            if not gpu_jobs:
                break
            own = sum(details['gpu'] == gpu for _, details in active.values())
            if eligible(available[gpu], own, 1):
                job = gpu_jobs.pop(0)
                active[job['id']] = launch(runtime, job['id'], job['kind'], job['config'], gpu)
                state['jobs'][job['id']] = active[job['id']][1]
        state.update(status='RUNNING', updated_at=time.time(),
                     counts=dict(Counter(v['status'] for v in state['jobs'].values())),
                     failed_fingerprints=dict(failures), systemic_stop_new_launches=systemic)
        atomic(state_path, state)
        if not active and (not queued or systemic):
            break
        time.sleep(15)

    from .summary import analyze
    if any(done(j) for j in jobs if j['kind'] == 'score'):
        analyze(runtime, jobs)
    state['status'] = 'PARTIAL_TECHNICAL_FAILURE' if failures else 'ANALYZED'
    atomic(state_path, state)


def worker(c, kind):
    from experiments.pamf_repl_view_shuffle import evaluation
    if kind == 'score':
        return evaluation.score(c, reference=c['reference_prediction'])
    activate(c)
    from . import source
    if kind == 'smoke':
        jobs = read(Path(c['run_root']) / 'jobs.json')
        return source.smoke(c, [j['config'] for j in jobs if j['kind'] == 'train'])
    if kind == 'train':
        return source.train(c)
    return evaluation.predict(c, load_source=source.load_reader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'worker'])
    parser.add_argument('--kind', choices=['smoke', 'train', 'predict', 'score'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    if args.action == 'dispatch':
        dispatch(config)
    else:
        worker(config, args.kind)
