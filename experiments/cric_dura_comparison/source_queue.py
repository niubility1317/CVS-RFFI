"""Source-stage owner: train F now; retain explicit G/prediction dependencies."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read, write

F_ABLATION_RUN = '20261021-phase1-CRIC-DURA-F-ablation-m45-r01'


def row_configs(rt, plan, handles):
    result = []
    for row in plan['rows']:
        if row['status'] == 'FROZEN_RESULT_REUSE':
            continue
        p = plan['protocols'][row['protocol_id']]
        h = handles[row['protocol_id']]
        result.append(dict(row, run_id=plan['run_id'], num_classes=len(h['classes']),
            source_rxs=sorted(p['source_rxs']), source_days=p['source_days'],
            source_cache=h['source_cache'], source_contract=h['source_contract'],
            output_root=str(Path(rt['run_root']) / row['row_id'] / 'source'),
            code_commit=rt['code_commit'], frozen_sources=rt['frozen_sources'],
            resource_group='cric-dura-20261021', stage='F_source', ablation=None))
    p, h = plan['protocols']['F0'], handles['F0']
    native = next(r for r in plan['rows'] if r['protocol_id'] == 'F0' and r['regime'] == 'native_cric')
    for seed in (2026092701, 2026092702, 2026092703):
        for number in range(1, 16):
            ablation = 'F%02d' % number
            rid = ablation + '-s%d' % seed
            result.append(dict(native, row_id=rid, model_seed=seed, run_id=F_ABLATION_RUN,
                num_classes=6, source_rxs=p['source_rxs'], source_days=p['source_days'],
                source_cache=h['source_cache'], source_contract=h['source_contract'],
                output_root=str(Path(rt['project']) / 'runs' / F_ABLATION_RUN / rid / 'source'),
                code_commit=rt['code_commit'], frozen_sources=rt['frozen_sources'],
                resource_group='cric-dura-20261021', stage='F_source', ablation=ablation,
                status='PLANNED_NOT_LAUNCHED'))
    # Complete method and F mechanism studies both receive early slots; the
    # order depends only on registered protocol/seed, never target scores.
    result.sort(key=lambda c: (c['model_seed'], c['regime'] != 'native_cric',
        c['protocol_id'] != 'F0' if c.get('ablation') else False,
        c['protocol_id'], c.get('ablation') or '', c['regime']))
    if len(result) != 195 or len({(r['run_id'], r['row_id']) for r in result}) != 195:
        raise ValueError('Fixed 150 main F and 45 mechanism training rows required')
    return result


def capacity(gpu):
    apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    return len(apps), free


def allow_shared(apps, free):
    # Latest user explicitly permits joining existing workers using spare VRAM.
    return apps <= 1 and free >= 12000


def worker(c):
    from .bootstrap import activate, numerical_policy
    activate(c)
    import torch
    numerical_policy()
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(.45)
    reservation = torch.empty(1, device='cuda:0')
    write(Path(c['output_root']).parent / 'gpu_ready.json', dict(pid=os.getpid(), cwd=os.getcwd(),
        argv=sys.argv, gpu=os.environ['CUDA_VISIBLE_DEVICES'], reservation_bytes=reservation.numel() * reservation.element_size()))
    from .f_train import train
    train(c)
    write(Path(c['output_root']).parent / 'evaluation_dependency.json', dict(status='WAITING_MATCHING_G',
        row_id=c['row_id'], run_id=c['run_id'], source_checkpoint=str(Path(c['output_root']) / 'final_ssdg.pth'),
        required='All preregistered raw/G seven-view predictions then independent truth-last; no source-only completion claim',
        protocol_id=c['protocol_id'], regime=c['regime'], ablation=c.get('ablation'),
        source_contract=c['source_contract'], F_code_commit=c['code_commit']))


def run(rt):
    from .data import prepare
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    Path(rt['log_root']).mkdir(parents=True, exist_ok=False)
    write(root / 'launch.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        owner=rt['launch_owner'], code_commit=rt['code_commit'], stage='F_source', user_shared_GPU_authorization=True))
    plan = read(rt['matrix'])
    handles = read(rt['reused_source_handles']) if rt.get('reused_source_handles') else prepare(rt, plan)
    from .smoke import run as source_smoke
    source_smoke(rt, handles)
    rows = row_configs(rt, plan, handles)
    configs = root / 'F_source_configs'
    configs.mkdir()
    for row in rows:
        write(configs / (row['run_id'] + '__' + row['row_id'] + '.json'), row)
    write(root / 'F_source_rows.json', rows)
    state = dict(status='SOURCE_QUEUE_RUNNING', pid=os.getpid(), rows={}, total_new_F=195,
        main_F=150, mechanism_F=45, reused_main_F=3, predictions_status='DEPENDENT_ON_MATCHING_G',
        shared_GPU_policy='max one own worker/card; max two compute workers total; free>=12000MiB',
        no_existing_process_stopped=True)
    pending = list(rows)
    active = {}
    failures = Counter()
    while pending or active:
        for gpu, item in list(active.items()):
            if item['child'].poll() is None:
                continue
            row, child, log = item['row'], item['child'], item['log']
            log.close()
            key = row['run_id'] + '/' + row['row_id']
            done = Path(row['output_root']) / 'complete.json'
            good = child.returncode == 0 and done.is_file() and read(done).get('status') == 'SOURCE_FROZEN'
            state['rows'][key].update(status='SOURCE_FROZEN_WAITING_G' if good else 'FAILED', returncode=child.returncode)
            if not good:
                text = Path(item['path']).read_text(errors='replace')[-4000:]
                fingerprint = text.splitlines()[-1] if text.splitlines() else 'no log'
                failures[fingerprint] += 1
                state['rows'][key]['failure'] = fingerprint
            del active[gpu]
        if failures and max(failures.values()) >= 2:
            state['unstarted'] = [r['run_id'] + '/' + r['row_id'] for r in pending]
            pending = []
        for gpu in rt['allowed_gpus']:
            if gpu in active or not pending:
                continue
            apps, free = capacity(gpu)
            if not allow_shared(apps, free):
                continue
            row = pending.pop(0)
            key = row['run_id'] + '/' + row['row_id']
            path = Path(rt['log_root']) / (row['run_id'] + '__' + row['row_id'] + '.log')
            cfg = configs / (row['run_id'] + '__' + row['row_id'] + '.json')
            log = path.open('x')
            cmd = [rt['python'], '-u', '-m', 'experiments.cric_dura_comparison.source_queue', 'worker', '--config', str(cfg)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            child = subprocess.Popen(cmd, cwd=rt['release_root'], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[gpu] = dict(row=row, child=child, log=log, path=str(path))
            state['rows'][key] = dict(status='RUNNING', pid=child.pid, gpu=gpu, config=str(cfg), log=str(path))
            # Keep launches serial until CUDA reservation is visible. This
            # bounds the pre-CUDA gap; never kill a foreign process on a race.
            ready = Path(row['output_root']).parent / 'gpu_ready.json'
            deadline = time.monotonic() + 40
            while not ready.is_file() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.5)
        state.update(pending=len(pending), active=len(active), failures=sum(failures.values()))
        write(root / 'source_pipeline_state.json', state, exclusive=False)
        if pending or active:
            time.sleep(30)
    state['status'] = 'PARTIAL_SOURCE_FAILURE' if failures else 'ALL_F_FROZEN_WAITING_G_AND_EVALUATION'
    write(root / 'source_pipeline_state.json', state, exclusive=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dispatch', 'worker'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    (run if args.mode == 'dispatch' else worker)(read(args.config))
