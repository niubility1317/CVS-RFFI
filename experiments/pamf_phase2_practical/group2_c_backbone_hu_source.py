"""Single-owner source-only Hu auxiliary training for C source backbones."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read, require, write


C_METHODS = {'mopc_hr', 'orthogonal', 'lora_triplet_knn'}


def validate_runtime(rt):
    forbidden = {'data_root', 'target_root', 'target_truth', 'truth', 'capsules', 'query', 'target_support'}
    require(not (forbidden & set(rt)), 'Source run contains target/query handles')
    seeds = rt['model_seeds']
    require(len(rt['c_methods']) == 3 and set(rt['c_methods']) == C_METHODS and
            seeds and len(set(seeds)) == len(seeds) and
            set(seeds) <= {2026092701, 2026092702, 2026092703, 2026092704}, 'Wrong source pilot matrix')
    require(rt['source_epochs'] == 200 and rt['source_batch'] == 256, 'Wrong source auxiliary budget')
    require(set(rt['c_source_roots']) == C_METHODS and all(rt['c_source_roots'].values()) and
            rt['source_cache'] and rt['source_contract'], 'Missing source input')
    require(rt['gpu_limits'] and set(rt['gpu_limits']) <= set(str(i) for i in range(8)) and
            all(0 < int(v) <= 2 for v in rt['gpu_limits'].values()) and rt['minimum_free_MiB'] >= 6000,
            'Invalid GPU resource budget')


def tasks(rt):
    return [dict(key=f'{method}-s{seed}', c_method=method, seed=seed,
                 output=str(Path(rt['run_root'])/'source'/f'{method}-s{seed}'))
            for method in rt['c_methods'] for seed in rt['model_seeds']]


def source_closed(job):
    folder = Path(job['output']); done = read(folder/'complete.json'); meta = read(folder/'metadata.json')
    require(done['status'] == 'SOURCE_AUX_COMPLETE' and done['target_contact'] is False and
            meta['source_role'] == 'L_s' and meta['target_contact'] is False and
            meta['source_epochs'] == 200 and meta['model_seed'] == job['seed'] and
            meta['C_source_method'] == job['c_method'] and (folder/'hu_aux.pt').is_file(),
            'Wrong or incomplete source auxiliary')


def train(rt, job):
    import json
    import torch
    from .io import activate
    from .group1 import source_scope
    from .group1_source import load_source
    from .external_additional import _source_data
    from .group2_c_backbone_hu import train_hu_source_aux

    activate(rt); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = source_scope(rt); source['run_root'] = rt['c_source_roots'][job['c_method']]
    bundle = load_source(source, job['c_method'], job['seed'], 'cuda:0')
    bundle['source_model_seed'] = int(job['seed'])
    x, y, rx = _source_data(rt, 'L_s')
    def logger(row):
        print(json.dumps(row, allow_nan=False), flush=True)
    state = train_hu_source_aux(bundle, x, y, rx, model_seed=job['seed'],
        epochs=rt['source_epochs'], batch_size=rt['source_batch'], logger=logger)
    folder = Path(job['output']); folder.mkdir(parents=True, exist_ok=False)
    with (folder/'hu_aux.pt').open('xb') as handle:
        torch.save(state, handle)
    write(folder/'metadata.json', state['metadata'])
    write(folder/'complete.json', dict(status='SOURCE_AUX_COMPLETE', target_contact=False,
                                        c_method=job['c_method'], model_seed=job['seed']))


def smoke(rt):
    import numpy as np
    import torch
    from .io import activate
    from .group1 import source_scope
    from .group1_source import load_source
    from .external_additional import _source_data
    from .group2_c_backbone_hu import train_hu_source_aux

    activate(rt); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = source_scope(rt); x, y, rx = _source_data(rt, 'L_s')
    labels = y.numpy()
    indices = [int(i) for c in range(6) for i in np.flatnonzero(labels == c)[:2]]
    require(len(indices) == 12, 'Source smoke requires 2 samples per class')
    results = []
    for c in rt['c_methods']:
        source['run_root'] = rt['c_source_roots'][c]
        bundle = load_source(source, c, rt['model_seeds'][0], 'cuda:0')
        bundle['source_model_seed'] = int(rt['model_seeds'][0])
        state = train_hu_source_aux(bundle, x[indices], y[indices], rx[indices],
                                    model_seed=rt['model_seeds'][0], epochs=1, batch_size=12)
        require(state['metadata']['target_contact'] is False and state['metadata']['source_steps'] == 1,
                'Hu source-only smoke failed')
        results.append(dict(C_source=c, source_checkpoint=bundle['source_checkpoint']))
        del state, bundle
    torch.cuda.synchronize()
    return dict(status='PASS', sources=results, source_role='L_s', target_rows=0, query_rows=0, truth_read=False)


def _gpu_free(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())


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
    cmd = [sys.executable, '-u', '-m', __package__+'.group2_c_backbone_hu_source', 'smoke', '--runtime', rt['runtime_file']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'logs/smoke.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root/'launches/smoke.json', dict(pid=proc.pid, gpu=gpu, cwd=str(Path.cwd()), argv=cmd))
        require(proc.wait() == 0, 'Real-checkpoint no-query source smoke failed')
    pending = list(enumerate(jobs)); active = []; completed = 0; failed = []
    while pending or active:
        for item in list(active):
            index, gpu, proc, log = item; code = proc.poll()
            if code is None:
                continue
            active.remove(item); log.close()
            if code:
                failed.append(dict(index=index, key=jobs[index]['key'], exit_code=code))
            else:
                try:
                    source_closed(jobs[index]); completed += 1
                except Exception as exc:
                    failed.append(dict(index=index, key=jobs[index]['key'], error=str(exc)))
        write(root/'worker_state.json', dict(status='FAILED' if failed else 'RUNNING', completed=completed,
            total=len(jobs), active=[dict(index=i, gpu=g, pid=p.pid) for i,g,p,_ in active],
            pending=len(pending), failed=failed), exclusive=False)
        if failed:
            pending.clear()
        for gpu, limit in rt['gpu_limits'].items():
            if not pending or failed:
                break
            if sum(g == gpu for _,g,_,_ in active) >= int(limit) or _gpu_free(gpu) < rt['minimum_free_MiB']:
                continue
            index,_ = pending.pop(0)
            cmd = [sys.executable, '-u', '-m', __package__+'.group2_c_backbone_hu_source', 'worker',
                   '--runtime', rt['runtime_file'], '--index', str(index)]
            log = (root/'logs'/f'worker-{index:05d}.log').open('x', encoding='utf-8')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active.append((index, gpu, proc, log))
            write(root/'launches'/f'worker-{index:05d}.json', dict(pid=proc.pid, gpu=gpu, cwd=str(Path.cwd()), argv=cmd))
        if pending or active:
            time.sleep(rt['poll_seconds'])
    require(not failed, 'Hu source auxiliary technical failure; output retained: '+str(failed))
    for job in jobs:
        source_closed(job)
    write(root/'worker_state.json', dict(status='COMPLETE', completed=completed, total=len(jobs),
                                         active=[], pending=0, failed=[]), exclusive=False)
    write(root/'complete.json', dict(status='SOURCE_AUX_COMPLETE', tasks=len(jobs), target_contact=False,
                                     query_read=False, truth_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['dispatch', 'smoke', 'worker'])
    parser.add_argument('--runtime', required=True); parser.add_argument('--index', type=int)
    args = parser.parse_args(); rt = read(args.runtime)
    if args.action == 'dispatch':
        dispatch(rt)
    elif args.action == 'smoke':
        write(Path(rt['run_root'])/'source_smoke.json', smoke(rt))
    else:
        require(args.index is not None and args.index >= 0, 'Invalid worker index')
        train(rt, read(Path(rt['run_root'])/'jobs'/f'{args.index:05d}.json'))
