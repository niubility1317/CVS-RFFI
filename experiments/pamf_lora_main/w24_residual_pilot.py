"""Blind native-Residual W24 pilot against matched frozen R69."""
import argparse
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.packet_physics import chirp_samples
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, SEED, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify, project_delta,
)
from experiments.pamf_lora_main.w24_native_residual import ROUTE, native_residual
from experiments.pamf_lora_main.w24_native_wiener import predict_pilot, receipt, score
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, aperture_batch, load_restorer, training_runtime


RUN = '20261125-phase1-lora-w24-native-residual-r69-pilot-s2026092702-r01'
PILOT_SEED = 2026112592
MODES = ('direct', 'r69_gated_half', 'wide_dura_half')
R69_TRAIN = '20261123-phase1-lora-dura-residual-practical-m1-r01'
R69_GATE = '20261123-phase1-lora-dura-residual-gate-m1-r01'
R69_TRAIN_COMMIT = 'fd67669f3a3186a12a528f1f662fdca865a4fdde'
R69_GATE_COMMIT = 'c99556d506b583be3db15722c6ad820a719aab50'


def r69_baseline(rt, contract, device):
    from experiments.pamf_lora_main.full_dura_observed_shape import ObservedShape
    from experiments.pamf_lora_main.residual_dura_gate_run import frozen_models

    project = Path(rt['project'])
    for name, expected in ((R69_TRAIN, R69_TRAIN_COMMIT), (R69_GATE, R69_GATE_COMMIT)):
        if read(project / 'releases' / f'{name}-release' / 'submit.json')['code_commit'] != expected:
            raise ValueError('Original Residual R69 training and gate release required')
    artifact = dict(rt, parent_root=str(project / 'runs' / R69_TRAIN),
                    gate_path=str(project / 'runs' / R69_GATE / 'gate.json'))
    model, gate, matched = frozen_models(artifact, device, restoration_seed=SEED)
    if matched != contract:
        raise ValueError('Residual R69 and W24 do not share registered physical source roles')
    return gate, model, ObservedShape().to(device).eval()


@torch.no_grad()
def restore_r69(packets, gate, model, projector, device):
    centers = np.stack([chirp_samples(q) for q in packets])
    scale = np.sqrt(np.mean(np.abs(centers)**2, axis=(1, 2)))
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError('Invalid received six-window packet')
    channels = np.stack((centers.real, centers.imag), axis=2) / scale[:, None, None, None]
    iq = tensor(channels.astype(np.float32, copy=False), device)
    restored, active = gate.restore(iq, model, projector)
    if not torch.isfinite(restored).all():
        raise ValueError('Nonfinite frozen R69 output')
    delta = np.from_dlpack((restored - iq).cpu().contiguous())
    return project_delta(packets, delta, scale), active


@torch.no_grad()
def batch_labels(f, old, g, tracker, threshold, packets, device):
    wide, n_wide = aperture_batch(f, g, tracker, threshold, packets, device)
    restored, active = restore_r69(packets, *old, device)
    changed = np.any(restored != packets, axis=1)
    r69 = wide[0].copy()
    if changed.any():
        r69[changed] = classify(f, restored[changed], device)
    return np.stack((wide[0], r69, wide[1])), [int(active.sum()), n_wide]


def receive(scene, receiver, item):
    row, packet = item
    return native_residual(packet, scene, receiver, row, 'target')


@torch.no_grad()
def evaluate(runtime):
    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    f = load_classifier(rt, contract, device)
    archive, parent = training_runtime(rt)
    g, track, threshold, frozen = load_restorer(archive, parent, contract, device)
    old = r69_baseline(rt, contract, device)
    batch = partial(batch_labels, f, old, g, track, threshold, device=device)
    u = np.array(native(source, 'U_s', contract['counts']['U_s'])[:4])
    distorted = np.stack([native_residual(x, VIEWS[-1], SOURCE[0], i, RUN) for i, x in enumerate(u)])
    clean_labels, bypass_count = batch(u)
    noisy_labels, active_count = batch(distorted)
    if (clean_labels.shape != (len(MODES), len(u)) or noisy_labels.shape != clean_labels.shape
            or min(active_count) == 0 or max(bypass_count) == len(u)):
        raise ValueError('Actual source W24 and Residual R69 smoke failed')
    write(root / 'smoke.json', dict(status='PASS', source_packets=len(u), target_reads=0,
          clean_activation=bypass_count, disturbed_activation=active_count, threshold=threshold,
          r69_threshold=float(old[0].threshold), wide_parameters=frozen['parameters'], wide_steps=frozen['steps']))
    predict_pilot(rt, batch, modes=MODES, seed=PILOT_SEED, log_prefix='W24_RESIDUAL_R69_PILOT',
                  receive=receive, route=ROUTE)


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir()
    state = root / 'state.json'
    try:
        reference = Path(rt['w24_release_root'])
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Exact W24 source release required')
        write(state, dict(status='WAIT_GPU'))
        deadline = monotonic() + 1800
        allowed = rt['allowed_gpus']
        while not (cards := available(allowed, max_compute_processes=8)):
            if monotonic() >= deadline:
                raise ValueError('GPU wait expired')
            sleep(20)
        gpu = next(i for i in allowed if i in cards)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(reference))
        with (root / 'evaluate.log').open('x', encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'], '-u', str(Path(__file__).resolve()), 'evaluate', str(runtime)],
                                   cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(state, dict(status='RUNNING', phase='evaluate', pid=job.pid, gpu=gpu), exclusive=False)
        if job.wait() or read(root / 'predictions' / 'complete.json') != receipt(MODES, PILOT_SEED, route=ROUTE):
            raise ValueError('Native Residual pilot prediction matrix did not freeze')
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'evaluate'):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    publish = sub.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    scoring = sub.add_parser('score')
    for key in ('predictions', 'prior', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project / 'runs', project / 'releases'
        data = releases / f'{DATA_RUN}-release' / 'data'
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
                  source_root=str(data / 'source'), native_target_root=str(data / 'target'),
                  classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'), allowed_gpus=[0, 5, 3],
                  cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'native_target_root', 'classifier_root',
                                                  'w24_release_root', 'training_release_root')]
        rt['external_releases'] += [str(runs / name) for name in (R69_TRAIN, R69_GATE)]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'score':
        score(args.predictions, args.prior, args.dataset, args.output, modes=MODES, seed=PILOT_SEED, route=ROUTE)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        evaluate(args.runtime)


if __name__ == '__main__':
    main()
