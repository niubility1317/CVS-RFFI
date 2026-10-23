"""Three seed-pinned cards, two workers/card; one separate truth-side scorer."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cric_dura_comparison.io import read, write
from .plan import GPU_BY_SEED, METHODS, SEEDS


def capacity(gpu):
    args = ['nvidia-smi', '-i', str(gpu)]
    apps = subprocess.check_output(args + ['--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    free = int(subprocess.check_output(args + ['--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    return len(apps), free


def validate_rows(rows, methods=METHODS, gpu_by_seed=None):
    methods = tuple(methods)
    gpu_by_seed = GPU_BY_SEED if gpu_by_seed is None else gpu_by_seed
    from .view_anchor import RESOURCE_GPU_BY_SEED
    if methods not in (METHODS, ('VIEWANCHOR',), ('PAIREDU',), ('UCRIC',), ('UMIX',), ('UGP',), ('UCMR',)):
        raise ValueError('A declared complete three-seed matrix is required')
    if gpu_by_seed != GPU_BY_SEED and (methods not in (('VIEWANCHOR',),('PAIREDU',),('UCRIC',),('UMIX',),('UGP',),('UCMR',)) or gpu_by_seed != RESOURCE_GPU_BY_SEED):
        raise ValueError('Only the declared three-card seed allocation is supported')
    expected = {(seed, method) for seed in SEEDS for method in methods}
    if (len(rows) != len(expected) or {(r['model_seed'], r['method']) for r in rows} != expected
            or any(r['gpu'] != gpu_by_seed[r['model_seed']] for r in rows)
            or len({r['row_id'] for r in rows}) != len(expected)):
        raise ValueError('Complete declared methods on each of three seed-pinned cards required')


def run(rt):
    methods = tuple(rt.get('methods', METHODS))
    gpu_by_seed = {int(seed): gpu for seed, gpu in rt.get('gpu_by_seed', GPU_BY_SEED).items()}
    validate_rows(rt['rows'], methods, gpu_by_seed)
    smoke_gpu = gpu_by_seed[SEEDS[0]]
    worker = 'experiments.pamf_ssl_research.' + {
        ('VIEWANCHOR',):'view_anchor_worker',('PAIREDU',):'paired_u_worker',('UCRIC',):'u_cric_worker',
        ('UMIX',):'u_mixstyle_worker',('UGP',):'u_gradprotect_worker',('UCMR',):'u_cmr_worker'}.get(methods,'worker')
    process_limit = 3 if methods in (('UCRIC',),('UMIX',),('UGP',),('UCMR',)) else 2
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    logs = root / 'logs'
    logs.mkdir()
    state = dict(status='SMOKE', controller_pid=os.getpid(), code_commit=rt['code_commit'],
                 gpu_by_seed=gpu_by_seed, jobs={r['row_id']: dict(status='QUEUED') for r in rt['rows']})
    write(root / 'state.json', state)
    env = dict(os.environ, PYTHONPATH=rt['release_root'], CUDA_VISIBLE_DEVICES=str(smoke_gpu),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    while True:
        predecessor_ready = True
        if methods == ('PAIREDU',):
            from .paired_u import reference_gpu_work_finished
            predecessor_ready = reference_gpu_work_finished(rt, SEEDS[0])
        elif methods == ('UCRIC',):
            from .u_cric import paired_dispatch_settled
            predecessor_ready = paired_dispatch_settled(rt, SEEDS[0])
        elif methods == ('UMIX',):
            from .u_mixstyle import second_slot_ready
            predecessor_ready = second_slot_ready(rt, SEEDS[0])
        elif methods == ('UGP',):
            from .u_gradprotect import second_slot_ready
            predecessor_ready = second_slot_ready(rt, SEEDS[0])
        elif methods == ('UCMR',):
            from .u_cmr import second_slot_ready
            predecessor_ready = second_slot_ready(rt, SEEDS[0])
        apps, free = capacity(smoke_gpu)
        if predecessor_ready and apps < process_limit and free >= 10000:
            break
        if methods not in (('VIEWANCHOR',),('PAIREDU',),('UCRIC',),('UMIX',),('UGP',),('UCMR',)):
            raise RuntimeError('Smoke GPU capacity changed; no workload modified')
        state.update(status='WAITING_FOR_SMOKE_GPU', predecessor_GPU_work_ready=predecessor_ready, observed_apps=apps,
                     observed_free_MiB=free, updated_unix=time.time())
        write(root / 'state.json', state, exclusive=False)
        time.sleep(20)
    state.update(status='SMOKE', updated_unix=time.time())
    write(root / 'state.json', state, exclusive=False)
    with (logs / 'smoke.log').open('x') as stream:
        result = subprocess.run([sys.executable, '-u', '-m', worker,
            'smoke', '--config', rt['runtime_file']], cwd=rt['release_root'], env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or read(root / 'smoke.json').get('status') != 'PASS':
        state['status'] = 'FAILED_SMOKE'
        write(root / 'state.json', state, exclusive=False)
        raise RuntimeError('Source-only smoke failed; inspect retained smoke.log')
    active, score, failures = {}, None, Counter()

    def launch(action, config, row_id, gpu):
        path = logs / (row_id + '-' + action + '.log')
        stream = path.open('x')
        e = dict(env, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
        child = subprocess.Popen([sys.executable, '-u', '-m', worker,
            action, '--config', config], cwd=rt['release_root'], env=e, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        return dict(child=child, gpu=gpu, log=str(path))

    while True:
        for row_id, item in list(active.items()):
            if item['child'].poll() is None:
                continue
            row = next(r for r in rt['rows'] if r['row_id'] == row_id)
            complete = Path(read(row['prediction_config'])['output_root']) / 'complete.json'
            okay = item['child'].returncode == 0 and complete.is_file() and read(complete).get('status') == 'ALL_PREDICTIONS_COMPLETE'
            state['jobs'][row_id].update(status='PREDICTIONS_COMPLETE' if okay else 'FAILED', exit_code=item['child'].returncode)
            if not okay:
                lines = Path(item['log']).read_text(errors='replace').splitlines()
                fingerprint = lines[-1] if lines else 'empty worker log'
                failures[fingerprint] += 1
                state['jobs'][row_id]['failure'] = fingerprint
            del active[row_id]
        if score and score['child'].poll() is not None:
            row_id = score['row_id']
            row = next(r for r in rt['rows'] if r['row_id'] == row_id)
            complete = Path(read(row['score_config'])['output_root']) / 'complete.json'
            okay = score['child'].returncode == 0 and complete.is_file() and read(complete).get('status') == 'SCORED_COMPLETE'
            state['jobs'][row_id].update(status='SCORED' if okay else 'FAILED_SCORE')
            if not okay:
                failures['independent scorer failed'] += 1
            score = None
        systemic = bool(failures and max(failures.values()) >= 2)
        if not systemic:
            for row in rt['rows']:
                row_id, gpu = row['row_id'], row['gpu']
                if state['jobs'][row_id]['status'] != 'QUEUED':
                    continue
                if methods == ('PAIREDU',) and not reference_gpu_work_finished(rt, row['model_seed']):
                    continue
                if methods == ('UCRIC',) and not paired_dispatch_settled(rt, row['model_seed']):
                    continue
                if methods == ('UMIX',) and not second_slot_ready(rt, row['model_seed']):
                    continue
                if methods == ('UGP',) and not second_slot_ready(rt, row['model_seed']):
                    continue
                if methods == ('UCMR',) and not second_slot_ready(rt, row['model_seed']):
                    continue
                apps, free = capacity(gpu)
                if apps >= process_limit or free < 10000 or sum(v['gpu'] == gpu for v in active.values()) >= 2:
                    continue
                item = launch('row', row['row_config'], row_id, gpu)
                active[row_id] = item
                state['jobs'][row_id] = dict(status='RUNNING', pid=item['child'].pid, gpu=gpu, log=item['log'])
        prior_scorer_finished = (methods not in (('PAIREDU',),('UCRIC',),('UMIX',),('UGP',),('UCMR',)) or read(rt['prior_score_owner_state'])['status']
                                 in ('ALL_SCORED','INCOMPLETE_TECHNICAL_FAILURE'))
        if score is None and prior_scorer_finished:
            pending = next((r for r in rt['rows'] if state['jobs'][r['row_id']]['status'] == 'PREDICTIONS_COMPLETE'), None)
            if pending:
                score = dict(launch('score', pending['score_config'], pending['row_id'], None), row_id=pending['row_id'])
                state['jobs'][pending['row_id']].update(status='SCORING', scorer_pid=score['child'].pid)
        statuses = [v['status'] for v in state['jobs'].values()]
        state['status'] = 'STOP_NEW_SUBMISSIONS_TECHNICAL' if systemic else 'RUNNING'
        state['failures'] = dict(failures)
        state['updated_unix'] = time.time()
        write(root / 'state.json', state, exclusive=False)
        if not active and score is None and (systemic or not any(s in ('QUEUED', 'PREDICTIONS_COMPLETE', 'SCORING', 'RUNNING') for s in statuses)):
            state['status'] = 'ALL_SCORED' if all(s == 'SCORED' for s in statuses) else 'INCOMPLETE_TECHNICAL_FAILURE'
            write(root / 'state.json', state, exclusive=False)
            return
        time.sleep(20)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    run(read(p.parse_args().config))
