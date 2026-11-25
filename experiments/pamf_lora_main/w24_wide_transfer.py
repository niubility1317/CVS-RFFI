"""Evaluate the frozen long-run W24 DURA with satellite-augmented W24 E104."""
import argparse
from functools import partial
import importlib
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from types import ModuleType

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, EDGE, W24_COMMIT, W24_RUN, classify_aperture,
)
from experiments.pamf_lora_main.w24_native_wiener import predict_pilot, receipt, score


RUN = '20261125-phase1-lora-w24-wide-aug-readout-s2026092702-r01'
TRAIN_RUN = '20261125-phase1-lora-dura-w24-train-s2026112991-r01'
TRAIN_COMMIT = '7615e8e4024e7f38688ffe69db3f632f5256f613'
STEPS = 40000
PILOT_SEED = 2026112581
MODES = ('direct', 'wide_dura_half')


def training_runtime(rt):
    archive = Path(rt['training_release_root'])
    parent = read(archive / 'runtime.json')
    if (parent['run_id'] != TRAIN_RUN or parent['code_commit'] != TRAIN_COMMIT
            or parent['source_root'] != rt['source_root']
            or Path(parent['run_root']) != Path(rt['project']) / 'runs' / TRAIN_RUN):
        raise ValueError('Unexpected long W24 DURA training source or release')
    return archive, Path(parent['run_root'])


def load_restorer(archive, root, contract, device):
    if read(root / 'state.json') != dict(status='SOURCE_FROZEN', steps=STEPS):
        raise ValueError('Long W24 DURA requires terminal source freeze')
    complete = read(root / 'w24-complete.json')
    isolation = ('F_reads', 'TX_reads', 'V_reads', 'target_reads', 'practical_reads', 'trace_reads')
    if (complete['status'] != 'SOURCE_FROZEN' or complete['steps'] != STEPS
            or complete['seed'] != 2026112991 or any(complete[k] != 0 for k in isolation)):
        raise ValueError('Long W24 DURA source provenance differs')

    # Use the original training release modules for the state and coordinate transform.
    package = ModuleType('lora_wide_dura_archive')
    package.__path__ = [str(archive / 'experiments' / 'pamf_lora_main')]
    sys.modules[package.__name__] = package
    imports = importlib.import_module('experiments').__path__
    imports.insert(0, str(archive / 'experiments'))
    try:
        training = importlib.import_module(f'{package.__name__}.dura_w24_train')
        runtime = importlib.import_module(f'{package.__name__}.spectral_block_source')
    finally:
        imports.pop(0)
    (model,), threshold = runtime.load_models(root, contract, device, {'w24': training.RECIPE})
    if threshold != complete['threshold']:
        raise ValueError('Long W24 DURA source gate differs')
    tracker = training.WideTracker().to(device).eval()
    return model, tracker, threshold, complete


def received_aperture(raw, device):
    from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, clean_iq

    iq = clean_iq(raw, 384, ALL_CHIRPS, native_edges=True, edge_points=EDGE)
    if iq.shape != (len(raw), 8, 2, 432):
        raise ValueError('Expected the registered W24 aperture')
    return tensor(iq, device)


@torch.no_grad()
def aperture_batch(classifier, model, tracker, threshold, raw, device):
    iq = received_aperture(raw, device)
    restored, active = model.restore(tracker, iq, threshold)
    base = classify_aperture(classifier, iq)
    predicted = base.copy()
    selected = np.asarray(active.tolist(), dtype=bool)
    if selected.any():
        predicted[selected] = classify_aperture(classifier, restored[active])
    return np.stack((base, predicted)), int(selected.sum())


@torch.no_grad()
def evaluate(runtime):
    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    archive, parent = training_runtime(rt)
    classifier = load_classifier(rt, contract, device)
    model, tracker, threshold, complete = load_restorer(archive, parent, contract, device)
    source_packets = np.array(native(source, 'U_s', contract['counts']['U_s'])[:4])
    rng = np.random.default_rng(PILOT_SEED)
    disturbance = (rng.standard_normal(source_packets.shape) + 1j*rng.standard_normal(source_packets.shape))
    strength = np.sqrt(np.mean(np.abs(source_packets)**2, axis=1, keepdims=True))
    noisy = (source_packets + .75 * strength * disturbance / np.sqrt(2)).astype(np.complex64)
    iq = received_aperture(np.concatenate((source_packets, noisy)), device)
    restored, active = model.restore(tracker, iq, threshold)
    if (not active[4:].any() or active[:4].all() or not torch.isfinite(restored).all()
            or not torch.equal(restored[~active], iq[~active])
            or not torch.equal(restored[..., -2*EDGE:], iq[..., -2*EDGE:])):
        raise ValueError('Long W24 DURA source smoke violates activity or unchanged edges')
    batch = partial(aperture_batch, classifier, model, tracker, threshold, device=device)
    labels, _ = batch(np.concatenate((source_packets, noisy)))
    inactive = ~np.asarray(active.tolist(), dtype=bool)
    if not np.array_equal(labels[0, inactive], labels[1, inactive]):
        raise ValueError('Inactive source predictions differ')
    write(root / 'smoke.json', dict(status='PASS', target_reads=0, source_packets=len(source_packets),
          restored=int(active.sum()), parent=TRAIN_RUN, source_checkpoint=complete))
    predict_pilot(rt, batch, modes=MODES, seed=PILOT_SEED, log_prefix='W24_WIDE_AUG_PILOT')


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir()
    state = root / 'state.json'

    def set_state(**values):
        write(state, values, exclusive=False)

    try:
        _, parent = training_runtime(rt)
        set_state(status='WAITING_FOR_FROZEN_TRAINING', parent=TRAIN_RUN)
        deadline = monotonic() + 7200
        while True:
            step = read(parent / 'state.json')['status']
            if step == 'SOURCE_FROZEN':
                break
            if step != 'SOURCE_TRAINING' or monotonic() >= deadline:
                raise ValueError(f'Long W24 DURA did not freeze from a valid training state: {step}')
            sleep(30)

        set_state(status='WAIT_GPU')
        deadline = monotonic() + 1800
        allowed = rt['allowed_gpus']
        while not (cards := available(allowed, max_compute_processes=8)):
            if monotonic() >= deadline:
                raise ValueError('GPU wait expired')
            sleep(20)
        gpu = next(i for i in allowed if i in cards)
        reference = Path(rt['w24_release_root'])
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Exact W24 reader release required')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=str(reference))
        with (root / 'evaluate.log').open('x', encoding='utf-8') as log:
            worker = subprocess.Popen([rt['python'], '-u', str(Path(__file__).resolve()), 'evaluate', str(runtime)],
                                      cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
        set_state(status='RUNNING', phase='evaluate', pid=worker.pid, gpu=gpu)
        if worker.wait() or read(root / 'predictions' / 'complete.json') != receipt(MODES, PILOT_SEED):
            raise ValueError('Long W24 DURA prediction matrix did not freeze')
        set_state(status='PREDICTIONS_FROZEN')
    except Exception as error:
        set_state(status='FAILED', error=str(error))
        raise


def publish(repo, context, ssh_config, output):
    from experiments.pamf_lora_main.publish import release, release_commit

    ctx = read(context)
    project = Path(ctx['project']).as_posix()
    runs, releases = f'{project}/runs', f'{project}/releases'
    data = f'{releases}/{DATA_RUN}-release/data'
    rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, code_commit=release_commit(repo), runner=__spec__.name,
              release_root=f'{releases}/{RUN}-release', run_root=f'{runs}/{RUN}',
              source_root=f'{data}/source', native_target_root=f'{data}/target',
              classifier_root=f'{runs}/{AUG_RUN}', w24_release_root=f'{releases}/{W24_RUN}-release',
              training_release_root=f'{releases}/{TRAIN_RUN}-release', allowed_gpus=[2, 7, 4],
              cpu_workers=8)
    rt['external_releases'] = [rt[k] for k in ('source_root', 'native_target_root', 'classifier_root',
                                              'w24_release_root', 'training_release_root')]
    release(repo, output, rt, ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'evaluate'):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    pub = sub.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--' + key, type=Path, required=True)
    scoring = sub.add_parser('score')
    for key in ('predictions', 'prior', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args.repo, args.context, args.ssh_config, args.output)
    elif args.action == 'score':
        score(args.predictions, args.prior, args.dataset, args.output, modes=MODES, seed=PILOT_SEED)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        evaluate(args.runtime)


if __name__ == '__main__':
    main()
