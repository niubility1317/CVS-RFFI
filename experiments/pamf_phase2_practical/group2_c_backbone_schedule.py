"""Single-owner C-source/B/C transplant queue with global truth-last barrier."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read, require, write


BOUND_B = {'protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage_sda', 'radionet_ada_knn'}
BOUND_C = {'mopc_hr', 'orthogonal', 'lora_triplet_knn'}


def validate_runtime(rt):
    require(len(rt['c_methods']) == 3 and set(rt['c_methods']) == BOUND_C, 'Wrong C source methods')
    scope = rt.get('scope', 'full')
    require(scope in ('full', 'pilot', 'poster_pilot'), 'Unknown matrix scope')
    expected_b = {'poster_ft'} if scope == 'poster_pilot' else BOUND_B
    require(len(rt['b_methods']) == len(expected_b) and set(rt['b_methods']) == expected_b, 'Wrong bound B subset')
    seeds = [2026092701] if scope in ('pilot', 'poster_pilot') else [2026092701, 2026092702, 2026092703, 2026092704]
    require(rt['model_seeds'] == seeds, 'Wrong source seeds')
    require(rt['K'] == [1, 5, 10, 20] and rt['N'] == [2, 5, 10, 20], 'Wrong K/N')
    rx = rt['target_receivers']
    require(len(rx) == len(set(rx)) == 7, 'Wrong target receivers')
    episodes = [f'MIX{i:02d}' for i in range(1, 9)] + [f'{scene}{i:02d}' for scene in ('LS', 'MU', 'LU') for i in range(1, 6)]
    expected = {f'capsules/{pool}/{episode}/{receiver}' for pool in ('P01', 'P02', 'P03')
                for episode in episodes for receiver in rx}
    if scope in ('pilot', 'poster_pilot'):
        require(rt['capsules'] == ['capsules/P01/MIX01/1-1'] and rt['capsules'][0] in expected,
                'Wrong fixed pilot capsule')
    else:
        require(len(rt['capsules']) == len(expected) == 483 and set(rt['capsules']) == expected, 'Wrong capsule matrix')
    require(set(rt['c_source_roots']) == BOUND_C and all(rt['c_source_roots'].values()), 'Missing source roots')
    require(rt['gpu_limits'] and set(rt['gpu_limits']) <= set(str(i) for i in range(8)) and
            all(0 < int(value) <= 2 for value in rt['gpu_limits'].values()), 'Invalid GPU limits')
    require(rt['minimum_free_MiB'] >= 6000 and 1 <= rt['score_processes'] <= 4, 'Invalid resource budget')
    require(not rt.get('native_adaptation_overrides') and not rt.get('native_registration_overrides'),
            'Formal release cannot change fixed native budgets')


def tasks(rt):
    jobs = []
    for b in rt['b_methods']:
        for c in rt['c_methods']:
            for seed in rt['model_seeds']:
                for index, capsule in enumerate(rt['capsules']):
                    require(capsule.startswith('capsules/') and '..' not in Path(capsule).parts, 'Invalid capsule')
                    points = ([(5, 20)] if rt.get('scope', 'full') in ('pilot', 'poster_pilot') else
                              [(k, n) for k in rt['K'] for n in rt['N']] if '/MIX' in capsule else [(5, 20)])
                    key = f'{b}__{c}-s{seed}-c{index:04d}'
                    jobs.append(dict(key=key, b_method=b, c_method=c, seed=seed, capsule=capsule,
                                     points=points, output=str(Path(rt['run_root'])/'predictions'/key)))
    require(len({job['key'] for job in jobs}) == len(jobs), 'Duplicate task key')
    return jobs


def prediction_closed(job):
    root = Path(job['output'])
    done, index = read(root/'complete.json'), read(root/'predictions.json')
    require(done['status'] == 'PREDICTIONS_COMPLETE' and done['truth_read'] is False and done['query_fit'] is False,
            'Prediction not frozen')
    require(done['rows'] == len(index['rows']) == len(job['points']) and
            index['capsule'] == job['capsule'] and index['model_seed'] == job['seed'] and
            index['profile'] == job['b_method'] and index['c_method'] == job['c_method'] and
            done['capsule_id'] == index['capsule_id'], 'Prediction binding mismatch')
    require({(r['k'], r['new_count']) for r in index['rows']} == {tuple(p) for p in job['points']},
            'Missing K/N row')
    expected_arm = job['b_method']+'__'+job['c_method']+'__C_SOURCE'
    require(all(r['arm'] == expected_arm and r['input_path'] == 'P0' and
                (root/r['prediction']).is_file() for r in index['rows']), 'Missing or wrong input prediction')


def _gpu_free(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def _queue(rt, jobs, action):
    root = Path(rt['run_root']); pending = list(enumerate(jobs)); active = []; completed = 0; failed = []
    while pending or active:
        for record in list(active):
            index, gpu, proc, log = record
            code = proc.poll()
            if code is None:
                continue
            active.remove(record); log.close()
            if code:
                failed.append(dict(index=index, key=jobs[index]['key'], exit_code=code))
            else:
                try:
                    if action == 'worker':
                        prediction_closed(jobs[index])
                    else:
                        output = root/'scores'/jobs[index]['key']
                        require(read(output/'complete.json')['status'] == 'SCORED_COMPLETE', 'Score not closed')
                        if '/MIX' in jobs[index]['capsule']:
                            require((output/'common_query_metrics.json').is_file(), 'Missing common query score')
                    completed += 1
                except Exception as exc:
                    failed.append(dict(index=index, key=jobs[index]['key'], error=str(exc)))
        write(root/f'{action}_state.json', dict(status='FAILED' if failed else 'RUNNING', completed=completed,
            total=len(jobs), active=[dict(index=i, gpu=g, pid=p.pid) for i, g, p, _ in active],
            pending=len(pending), failed=failed), exclusive=False)
        if failed:
            pending.clear()
        slots = list(rt['gpu_limits'].items()) if action == 'worker' else [(None, rt['score_processes'])]
        for gpu, limit in slots:
            if not pending or failed:
                break
            if sum(g == gpu for _, g, _, _ in active) >= int(limit):
                continue
            if gpu is not None and _gpu_free(gpu) < int(rt['minimum_free_MiB']):
                continue
            index, _ = pending.pop(0)
            cmd = [sys.executable, '-u', '-m', __package__+'.group2_c_backbone_schedule', action,
                   '--runtime', rt['runtime_file'], '--index', str(index)]
            log = (root/'logs'/f'{action}-{index:05d}.log').open('x', encoding='utf-8')
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active.append((index, gpu, proc, log))
            write(root/'launches'/f'{action}-{index:05d}.json', dict(pid=proc.pid, gpu=gpu, cwd=str(Path.cwd()), argv=cmd))
        if pending or active:
            time.sleep(rt['poll_seconds'])
    require(not failed, 'Technical failure; retained output: '+str(failed))
    write(root/f'{action}_state.json', dict(status='COMPLETE', completed=completed, total=len(jobs),
          active=[], pending=0, failed=[]), exclusive=False)


def _score(rt, job):
    from .episodic_score import run
    from .milkr_r42_suite import common_query_score
    root = Path(rt['run_root'])
    require(read(root/'predictions_complete.json')['status'] == 'ALL_PREDICTIONS_FIXED', 'Truth barrier not closed')
    output = root/'scores'/job['key']
    run(rt, dict(prediction=job['output'], output=str(output)))
    if '/MIX' in job['capsule']:
        common_query_score(rt, job, Path(job['output']), read(Path(job['output'])/'predictions.json'), output)


def smoke(rt):
    """Real source checkpoints and source L_s only, no target/query/truth."""
    import numpy as np
    import torch
    from .io import activate
    from .group1 import source_scope
    from .group1_source import load_source, source_data
    from .group2_c_backbone_fit import fit_b_from_c_source, register_native_c

    activate(rt); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = source_scope(rt); data = source_data(source, 'L_s')
    labels = np.asarray(data.metadata['y'])
    indices = [int(i) for c in range(6) for i in np.flatnonzero(labels == c)[:2]]
    require(len(indices) == 12, 'Source smoke needs two labeled examples per class')
    x = data.x[indices].to('cuda:0'); y = torch.arange(6, device='cuda:0').repeat_interleave(2)
    test_rt = dict(rt)
    test_rt['native_adaptation_overrides'] = {p: dict(epochs=1, steps_per_epoch=1, estimate_steps=1,
        iterations=1, dann_epochs=1, lmmd_epochs=1, finetune_epochs=1, batch_size=6) for p in rt['b_methods']}
    test_rt['native_registration_overrides'] = {'mopc_hr': {'mopc_epochs': 1, 'mopc_batch_size': 4},
                                                'orthogonal': {'orthogonal_epochs': 1}}
    results = []
    for c in rt['c_methods']:
        source['run_root'] = rt['c_source_roots'][c]
        bundle = load_source(source, c, rt['model_seeds'][0], 'cuda:0')
        for b in rt['b_methods']:
            fitted = fit_b_from_c_source(test_rt, bundle, b, x, y, seed=123, model_seed=rt['model_seeds'][0],
                                          source_x=x, source_y=y)
            new_y = torch.tensor([6, 6, 7, 7], device='cuda:0')
            state = register_native_c(test_rt, fitted['c_state'], x[:4], new_y, [6, 7], seed=456)
            with torch.no_grad():
                result = state.score(x[:2])
            require(result.shape == (2, 8) and bool(torch.isfinite(result).all()), 'Source-only smoke failed')
            results.append(dict(B=b, C_source=c, source_checkpoint=bundle['source_checkpoint']))
            del fitted, state
        del bundle
    torch.cuda.synchronize()
    return dict(status='PASS', pairs=results, source_role='L_s', target_rows=0, query_rows=0, truth_read=False)


def dispatch(rt):
    validate_runtime(rt)
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'launches', 'jobs'):
        (root/name).mkdir()
    jobs = tasks(rt)
    for index, job in enumerate(jobs):
        write(root/'jobs'/f'{index:05d}.json', job)
    write(root/'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv))
    gpu = next((g for g in rt['gpu_limits'] if _gpu_free(g) >= rt['minimum_free_MiB']), None)
    while gpu is None:
        time.sleep(rt['poll_seconds'])
        gpu = next((g for g in rt['gpu_limits'] if _gpu_free(g) >= rt['minimum_free_MiB']), None)
    cmd = [sys.executable, '-u', '-m', __package__+'.group2_c_backbone_schedule', 'smoke', '--runtime', rt['runtime_file']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'logs'/'smoke.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root/'launches'/'smoke.json', dict(pid=proc.pid, gpu=gpu, cwd=str(Path.cwd()), argv=cmd))
        require(proc.wait() == 0, 'Real-checkpoint no-query smoke failed')
    _queue(rt, jobs, 'worker')
    for job in jobs:
        prediction_closed(job)
    count = sum(len(job['points']) for job in jobs)
    write(root/'predictions_complete.json', dict(status='ALL_PREDICTIONS_FIXED', tasks=len(jobs), rows=count,
                                                  truth_read=False, query_fit=False))
    _queue(rt, jobs, 'score')
    write(root/'complete.json', dict(status='SCORED_COMPLETE', tasks=len(jobs), rows=count,
                                     truth_last=True, target_feedback=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'smoke', 'worker', 'score'])
    parser.add_argument('--runtime', required=True); parser.add_argument('--index', type=int)
    args = parser.parse_args(); runtime = read(args.runtime)
    if args.action == 'dispatch':
        dispatch(runtime)
    elif args.action == 'smoke':
        write(Path(runtime['run_root'])/'source_smoke.json', smoke(runtime))
    else:
        require(args.index is not None and args.index >= 0, 'Invalid job index')
        task = read(Path(runtime['run_root'])/'jobs'/f'{args.index:05d}.json')
        if args.action == 'worker':
            from .group2_c_backbone_worker import predict
            predict(runtime, task)
        else:
            _score(runtime, task)
