"""Supplemental DURA queue; existing training and raw owners keep their files."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read, write
from .dura_reference import ready, validate_rows

WORKER = 'experiments.pamf_ssl_research.dura_reference_worker'


def has_capacity(gpu, active):
    args = ['nvidia-smi', '-i', str(gpu)]
    registered = {int(v.strip()) for v in subprocess.check_output(args + [
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines() if v.strip()}
    reserved = {item['child'].pid for item in active.values() if item['gpu'] == gpu}
    free = int(subprocess.check_output(args + [
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    # Popen returns before CUDA initialization; reserve that process immediately.
    return len(registered | reserved) < 2 and free >= 10000


def prior_scorer_finished(rt):
    """The already running raw owner retains the single CPU scorer until it exits."""
    status = read(rt['prior_score_owner_state'])['status']
    return status in ('ALL_SCORED', 'INCOMPLETE_TECHNICAL_FAILURE')


def run(rt):
    validate_rows(rt['rows'])
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    logs = root / 'logs'
    logs.mkdir()
    state = dict(status='WAITING_FOR_SMOKE_GPU', controller_pid=os.getpid(), code_commit=rt['code_commit'],
                 jobs={r['row_id']: dict(status='QUEUED') for r in rt['rows']})
    write(root / 'state.json', state)
    env = dict(os.environ, PYTHONPATH=rt['release_root'], OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    while True:
        selected = None
        for row in rt['rows']:
            if has_capacity(row['gpu'], {}) and ready(read(row['prediction_config'])):
                selected = row
                break
        if selected:
            break
        state['updated_unix'] = time.time()
        write(root / 'state.json', state, exclusive=False)
        time.sleep(20)
    state.update(status='SMOKE', smoke_gpu=selected['gpu'], smoke_row=selected['row_id'])
    write(root / 'state.json', state, exclusive=False)
    with (logs / 'smoke.log').open('x') as stream:
        result = subprocess.run([sys.executable, '-u', '-m', WORKER, 'smoke', '--config', selected['prediction_config']],
            cwd=rt['release_root'], env=dict(env, CUDA_VISIBLE_DEVICES=str(selected['gpu'])),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or read(root / 'smoke.json').get('status') != 'PASS':
        state['status'] = 'FAILED_SMOKE'
        write(root / 'state.json', state, exclusive=False)
        raise RuntimeError('Frozen F/G source smoke failed; retained log')
    active, scorer, failures = {}, None, Counter()

    def launch(row, action, gpu):
        path = logs / (row['row_id'] + '-' + action + '.log')
        config = row['prediction_config'] if action == 'predict' else row['score_config']
        with path.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-m', WORKER, action, '--config', config],
                cwd=rt['release_root'], env=dict(env, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu)),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        return dict(child=child, gpu=gpu, log=str(path), row=row)

    while True:
        for key, item in list(active.items()):
            if item['child'].poll() is None:
                continue
            complete = Path(read(item['row']['prediction_config'])['output_root']) / 'complete.json'
            okay = item['child'].returncode == 0 and complete.is_file() and read(complete).get('status') == 'ALL_PREDICTIONS_COMPLETE'
            state['jobs'][key].update(status='PREDICTIONS_COMPLETE' if okay else 'FAILED', exit_code=item['child'].returncode)
            if not okay:
                lines = Path(item['log']).read_text(errors='replace').splitlines()
                fingerprint = lines[-1] if lines else 'empty predictor log'
                failures[fingerprint] += 1
                state['jobs'][key]['failure'] = fingerprint
            del active[key]
        if scorer and scorer['child'].poll() is not None:
            row = scorer['row']
            complete = Path(read(row['score_config'])['output_root']) / 'complete.json'
            okay = scorer['child'].returncode == 0 and complete.is_file() and read(complete).get('status') == 'SCORED_COMPLETE'
            state['jobs'][row['row_id']].update(status='SCORED' if okay else 'FAILED_SCORE')
            if not okay:
                failures['independent scorer failed'] += 1
            scorer = None
        systemic = bool(failures and max(failures.values()) >= 2)
        if not systemic:
            for row in rt['rows']:
                key, gpu = row['row_id'], row['gpu']
                if state['jobs'][key]['status'] != 'QUEUED' or len(active) >= rt['max_total_workers']:
                    continue
                try:
                    source_ready = ready(read(row['prediction_config']))
                except (ValueError, KeyError) as error:
                    fingerprint = str(error)
                    failures[fingerprint] += 1
                    state['jobs'][key].update(status='FAILED_SOURCE', failure=fingerprint)
                    if failures[fingerprint] >= 2:
                        break
                    continue
                if not source_ready:
                    continue
                if not has_capacity(gpu, active) or sum(x['gpu'] == gpu for x in active.values()) >= rt['max_per_gpu']:
                    continue
                item = launch(row, 'predict', gpu)
                active[key] = item
                state['jobs'][key] = dict(status='RUNNING', pid=item['child'].pid, gpu=gpu, log=item['log'])
        if scorer is None and prior_scorer_finished(rt):
            pending = next((r for r in rt['rows'] if state['jobs'][r['row_id']]['status'] == 'PREDICTIONS_COMPLETE'), None)
            if pending:
                scorer = launch(pending, 'score', None)
                state['jobs'][pending['row_id']].update(status='SCORING', scorer_pid=scorer['child'].pid)
        systemic = bool(failures and max(failures.values()) >= 2)
        statuses = [j['status'] for j in state['jobs'].values()]
        state.update(status='STOP_NEW_SUBMISSIONS_TECHNICAL' if systemic else 'RUNNING',
                     failures=dict(failures), updated_unix=time.time())
        write(root / 'state.json', state, exclusive=False)
        pending = any(s in ('QUEUED', 'PREDICTIONS_COMPLETE', 'SCORING', 'RUNNING') for s in statuses)
        if not active and scorer is None and (systemic or not pending):
            state['status'] = 'ALL_SCORED' if all(s == 'SCORED' for s in statuses) else 'INCOMPLETE_TECHNICAL_FAILURE'
            write(root / 'state.json', state, exclusive=False)
            return
        time.sleep(20)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    run(read(p.parse_args().config))
