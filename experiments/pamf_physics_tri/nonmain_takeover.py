"""Finish nine already-trained non-Main rows after the user's Main stop.

Predictions and scoring use the old immutable release and its original configs
and output paths. The new controller owns only scheduling and its own logs.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_physics_branch.runner import admits
from experiments.pamf_repl_completion.runner import foreign_CPU_score

OLD = '20261111-phase1-pamf-tri-branchguard-m12-r01'
RUN = '20261111-phase1-pamf-branchguard-nonmain-takeover-r01'
SETTINGS = ('receiver_split_1', 'receiver_split_2', 'labeled_5pct')
SEEDS = (2026092701, 2026092702, 2026092703)


def dispatch(rt):
    project, root = Path(rt['project']), Path(rt['run_root'])
    old = project / 'runs' / OLD
    release = project / 'releases' / (OLD + '-release')
    old_state = read(old / 'pipeline_state.json')
    stopped = {f'train__main_split-tri-branchguard-s{s}' for s in SEEDS}
    if (rt['run_id'] != RUN or old_state['status'] != 'PARTIAL_TECHNICAL_FAILURE'
            or set(old_state['failed_jobs']) != stopped
            or any(old_state['jobs'][j]['exit_code'] != -15 for j in stopped)
            or Path('/proc').joinpath(str(old_state['pid'])).exists()):
        raise ValueError('Only the completed original owner after the three user-directed stops may be resumed')
    rows = [f'{setting}-tri-branchguard-s{seed}' for setting in SETTINGS for seed in SEEDS]
    all_jobs = {j['id']: j for j in read(old / 'jobs.json')}
    jobs = [all_jobs[f'{kind}__{rid}'] for kind in ('predict', 'score') for rid in rows]
    for rid in rows:
        if old_state['jobs'][f'train__{rid}']['status'] != 'COMPLETE':
            raise ValueError('All nine originally selected source rows must be frozen')
        pred = read(all_jobs[f'predict__{rid}']['config'])
        source = read(Path(pred['source_output']) / 'complete.json')
        if source['status'] != 'SOURCE_FROZEN' or source['target_access'] or source['U_truth_read']:
            raise ValueError('Only the original truth-blind source checkpoints may be evaluated')
    for job in jobs:
        c = read(job['config'])
        if c['run_id'] != OLD or Path(c['output_root']).exists():
            raise ValueError('Only unwritten original output paths may be resumed')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    if read(old / 'smoke.json')['status'] != 'PASS':
        raise ValueError('Original source-only checkpoint smoke must have passed')
    write(root / 'takeover.json', dict(old_run=OLD, old_release=str(release), rows=rows,
          main_stop='USER_DIRECTED_SIGTERM', original_smoke='PASS',
          all_nine_predictions_before_any_score=True, original_configs_preserved=True))
    state = dict(status='RUNNING', pid=os.getpid(), owner=rt['launch_owner'],
                 jobs={j['id']: dict(status='QUEUED') for j in jobs})
    active, failed = {}, []

    def save():
        from experiments.practical_full_2x2.cache import atomic
        atomic(root / 'pipeline_state.json', state)

    def launch(job, gpu):
        cmd = [rt['python'], '-u', '-m', 'experiments.pamf_physics_tri.pair_runner',
               'worker', '--kind', job['kind'], '--config', job['config']]
        env = dict(os.environ, PYTHONPATH=str(release), CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        log = root / 'logs' / (job['id'] + '.log')
        with log.open('x') as handle:
            proc = subprocess.Popen(cmd, cwd=release, env=env, stdin=subprocess.DEVNULL,
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        item = dict(status='RUNNING', pid=proc.pid, gpu=gpu, cwd=str(release), argv=cmd, log=str(log))
        state['jobs'][job['id']] = item
        active[job['id']] = proc, item

    by_id = {j['id']: j for j in jobs}
    save()
    while True:
        for job_id, (proc, item) in list(active.items()):
            if proc.poll() is None:
                continue
            job = by_id[job_id]
            marker = Path(job['completion'])
            evidence = read(marker) if marker.is_file() else {}
            detail = marker.parent / 'scores.json'
            truth_ok = (evidence.get('truth_read') is False if job['kind'] == 'predict'
                        else detail.is_file() and read(detail).get('truth_last') is True)
            valid = proc.returncode == 0 and evidence.get('status') in job['successful_statuses'] and truth_ok
            state['jobs'][job_id] = dict(item, status='COMPLETE' if valid else 'FAILED', exit_code=proc.returncode)
            if not valid:
                failed.append(job_id)
            del active[job_id]
        if not failed:
            pending = [j for j in jobs if state['jobs'][j['id']]['status'] == 'QUEUED']
            predictions = [j for j in pending if j['kind'] == 'predict']
            for gpu in (6, 7):
                if predictions and not any(item['gpu'] == gpu for _, item in active.values()) and admits(gpu, active, 6000):
                    launch(predictions.pop(0), gpu)
            frozen = all(state['jobs'][f'predict__{rid}']['status'] == 'COMPLETE' for rid in rows)
            if frozen and not any(item['gpu'] is None for _, item in active.values()):
                scores = [j for j in pending if j['kind'] == 'score']
                if scores and not foreign_CPU_score({proc.pid for proc, _ in active.values()}):
                    launch(scores[0], None)
        state.update(failed_jobs=failed, updated_at=time.time())
        save()
        if not active and (failed or all(v['status'] == 'COMPLETE' for v in state['jobs'].values())):
            break
        time.sleep(15)
    state['status'] = 'PARTIAL_TECHNICAL_FAILURE' if failed else 'SCORED_COMPLETE'
    save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config))
