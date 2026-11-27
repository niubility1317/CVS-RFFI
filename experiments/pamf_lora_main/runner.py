"""N607 owner for source calibration, frozen training and blind evaluation."""
import argparse
from collections import Counter, deque
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from experiments.cric_dura_comparison.io import read, write
from .comparison import predict, train_dura, train_resnet
from .recognition import FOLDS, METHODS, SEEDS, calibrate, choose_dose, config, segment, train_pamf
from .views import make_source_views, make_target_views


def activate(rt):
    from experiments.cric_dura_comparison.bootstrap import activate as native, numerical_policy
    native(rt)
    numerical_policy()


def worker(c, kind):
    activate(c)
    root = Path(c['run_root'])
    source = Path(c['source_root'])
    common = dict(c, source_cache=str(source), source_contract=str(source/'contract.json'),
                  source_practical=str(root/'source_views/L_practical.npy'))
    if kind == 'source_views':
        return make_source_views(source, root/'source_views')
    if kind == 'target_views':
        return make_target_views(Path(c['target_root']), root/'target_views')
    if kind == 'calibration':
        return calibrate(common, c['rx'], root/'calibration'/str(c['rx']))
    if kind == 'smoke':
        out = root/'source_smoke'
        sm = config(common, 'pamf_adaptive', SEEDS[0], out)
        evidence = segment(sm, 'source_checks', 0, 2, None, smoke=True)
        write(out/'complete.json', dict(status='PASS', source_steps=evidence['updates'], query_reads=0))
        return
    seed, method = c['seed'], c['method']
    name = f'{method}-s{seed}'
    if kind == 'dura':
        return train_dura(seed, source, root/'source_views/LU_DURA_practical.npy',
                          common['source_contract'], root/'G'/name)
    if kind == 'train':
        training_keys = ('run_id', 'code_commit', 'ssl', 'source_cache', 'source_contract', 'source_practical')
        train_args = {key: common[key] for key in training_keys}
        cfg = config(train_args, method, seed, root/'F'/name, weight=c.get('weight', 6.))
        return train_resnet(cfg) if method == 'resnet18' else train_pamf(cfg)
    if kind == 'predict':
        dura_folder = root/'G'/f'dura-s{seed}' if method == 'pamf_adaptive' else None
        return predict(root/'F'/name, Path(c['target_root']), root/'target_views', root/'predictions'/name,
                       source/'contract.json', dura_folder)
    raise ValueError('Unknown job kind')


def _validation_only(pid):
    try:
        process = Path('/proc') / pid
        if (process/'comm').read_text().strip() != 'MATLAB':
            return False
        command = (process/'cmdline').read_bytes()
    except OSError:
        return False
    return not re.search(rb'\b\w+_train\(', command) and bool(
        re.search(rb'\b\w+_validate\(', command) or b'/attempts/validate_' in command
    )


def available(allowed, max_compute_processes=2, *, training_cap=False):
    def nvsmi(field):
        return subprocess.check_output(['nvidia-smi', field, '--format=csv,noheader,nounits'], text=True)
    occupants = (line.replace(' ', '').split(',') for line in
                 nvsmi('--query-compute-apps=gpu_uuid,pid').splitlines())
    others = Counter(uuid for uuid, pid in occupants if not (training_cap and _validation_only(pid)))
    caps = (line.replace(' ', '').split(',') for line in
            nvsmi('--query-gpu=index,uuid,memory.free').splitlines())
    return [int(i) for i, uuid, mb in caps
            if int(i) in allowed and others[uuid] < max_compute_processes and int(mb) >= 12_288]


def job(rt, kind, name, gpu, **args):
    root = Path(rt['run_root'])
    cfg = root/'job_configs'/f'{kind}-{name}.json'
    body = {k: rt[k] for k in ('run_id', 'run_root', 'frozen_sources', 'code_commit', 'source_root')}
    if kind in ('calibration', 'smoke', 'train'):
        body['ssl'] = rt['ssl']
    if kind in ('target_views', 'predict'):
        body['target_root'] = rt['target_root']
    write(cfg, dict(body, **args))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    log = root/'logs'/f'{kind}-{name}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('x', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, kind, str(cfg)],
                                 env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    print('LORA_LAUNCHED', kind, name, child.pid, gpu, flush=True)
    return child


def watch(rt, tasks, calibrations=()):
    waiting, active = deque(tasks), {}
    while waiting or active:
        for gpu, (child, task) in list(active.items()):
            status = child.poll()
            if status is None:
                continue
            if status:
                raise RuntimeError(f'LORA_WORKER_FAILED {task[0]} {task[1]} status={status}')
            print('LORA_COMPLETED', task[0], task[1], flush=True)
            del active[gpu]
        calibration_active = any(task[0] == 'calibration' for _, task in active.values())
        calibration_pending = any(task[0] == 'calibration' for task in waiting)
        if calibrations and not (calibration_active or calibration_pending):
            decision = choose_dose(calibrations)
            write(Path(rt['run_root'])/'source_decision.json', decision)
            waiting.extendleft(reversed([('train', f'pamf_adaptive-s{seed}',
                dict(seed=seed, method='pamf_adaptive', weight=decision['weight'])) for seed in SEEDS]))
            calibrations = ()
        if waiting:
            for gpu in available(rt['allowed_gpus']):
                if gpu not in active and waiting:
                    task = waiting.popleft()
                    kind, name, params = task
                    active[gpu] = job(rt, kind, name, gpu, **params), task
        if waiting or active:
            time.sleep(20)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    print('LORA_DISPATCH', root.name, flush=True)
    status = root/'state.json'
    try:
        write(status, dict(status='PREPARING_SOURCE_PRACTICAL'))
        if job(rt, 'source_views', 'main', None).wait():
            raise RuntimeError('LoRa source practical generation failed')
        if job(rt, 'smoke', 'main', None).wait():
            raise RuntimeError('LoRa actual-source no-query smoke failed')
        write(status, dict(status='SOURCE_TRAINING', seeds=list(SEEDS)), exclusive=False)
        tasks = [('calibration', str(rx), dict(rx=rx)) for rx in FOLDS]
        for seed in SEEDS:
            tasks += [('dura', str(seed), dict(seed=seed, method='dura'))]
            tasks += [('train', f'{m}-s{seed}', dict(seed=seed, method=m))
                      for m in METHODS if m != 'pamf_adaptive']
        watch(rt, tasks, [root/'calibration'/str(rx)/'complete.json' for rx in FOLDS])
        decision = read(root/'source_decision.json')
        write(status, dict(status='SOURCE_FROZEN', decision=decision), exclusive=False)
        if job(rt, 'target_views', 'main', None).wait():
            raise RuntimeError('LoRa target practical generation failed')
        write(status, dict(status='PREDICTING', decision=decision), exclusive=False)
        watch(rt, [('predict', f'{m}-s{seed}', dict(seed=seed, method=m)) for seed in SEEDS for m in METHODS])
        write(status, dict(status='PREDICTIONS_FROZEN', decision=decision, target_truth_read=False), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['dispatch', 'source_views', 'target_views', 'calibration',
                         'smoke', 'train', 'dura', 'predict'])
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    runtime = read(args.config)
    if args.kind == 'dispatch':
        dispatch(runtime)
    else:
        worker(runtime, args.kind)
