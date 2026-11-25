"""Freeze a source-only native LoRa restorer, then test it with the original W24."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import numpy as np
import torch

# The prediction subprocess keeps the original W24 release on PYTHONPATH.
# Only the two experiment modules below are added from the new release.
if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, METHOD, SEED, W24_COMMIT, W24_RUN, classify, load_w24, predict, receipt, score,
)
from experiments.pamf_lora_main.w24_native_restorer import NativeRestorer, colored_observation, source_loss


RUN = '20261125-phase1-lora-w24-native-restore-s2026092702-r01'
MODES = ('direct', 'native_half', 'native_quarter')
SPLIT_SEED = 2026112508
STEPS = 1200
BATCH = 6
LR = 2e-4


def partition(source, contract):
    packets = native(source, 'U_s', contract['counts']['U_s'])
    with np.load(source / 'U_s_metadata.npz', allow_pickle=False) as index:
        if 'y' in index.files or index['ids'].tolist() != contract['role_ids']['U_s']:
            raise ValueError('Expected the registered TX-blind native source U role')
        receivers = index['rx_i'].copy()
    rng = np.random.default_rng(SPLIT_SEED)
    by_rx = [rng.permutation(np.flatnonzero(receivers == i))[:1600] for i in range(5)]
    if any(len(rows) != 1600 for rows in by_rx):
        raise ValueError('Insufficient native source U on a registered receiver')
    fit = np.concatenate([rows[:1000] for rows in by_rx])
    calibration = np.concatenate([rows[1000:1500] for rows in by_rx])
    held = np.concatenate([rows[1500:] for rows in by_rx])
    return packets, fit, calibration, held


def as_native(raw, device):
    raw = raw.reshape(-1, 8, 1024)
    return torch.complex(tensor(raw.real, device), tensor(raw.imag, device))


def checkpoint_record(contract, threshold):
    return dict(status='SOURCE_FROZEN', method='native_half_quarter', seed=SEED,
                split_seed=SPLIT_SEED, source_contract=contract, input_shape=[8, 1024],
                fit=5000, calibration=2500, held=500, updates=STEPS, threshold=threshold,
                F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


def fit(rt):
    activate(rt)
    source, root, device = Path(rt['source_root']), Path(rt['run_root']), 'cuda:0'
    contract = source_contract(source)
    packets, fit_ids, calibration, held = partition(source, contract)
    torch.manual_seed(SEED)
    rng = torch.Generator(device=device).manual_seed(SPLIT_SEED)
    sampler = np.random.default_rng(SEED)
    model = NativeRestorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=2e-5)
    with (root / 'steps.jsonl').open('x', encoding='utf-8') as log:
        for step in range(1, STEPS + 1):
            clean = as_native(packets[sampler.choice(fit_ids, BATCH, replace=False)], device)
            optimizer.zero_grad(set_to_none=True)
            loss = source_loss(model, clean, rng, identity=step % 5 == 0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native source U loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            row = dict(step=step, loss=float(loss.detach()), grad=float(gradient), lr=schedule.get_last_lr()[0])
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 50 == 0:
                log.flush()
                print('W24_NATIVE_G', json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad():
        clean_scores = torch.cat([corruption_score(model.frame(as_native(packets[ids], device))[0])
                                  for ids in np.array_split(calibration, 40)])
        threshold = float(torch.quantile(clean_scores, .999))
        if not np.isfinite(threshold):
            raise FloatingPointError('Invalid source calibration threshold')
        clean = as_native(packets[held[:8]], device)
        corrupted = colored_observation(clean, rng)
        restored = model.restore(corrupted, threshold)
        if not torch.isfinite(restored).all():
            raise FloatingPointError('Native source U holdout failed')
        holdout = dict(n=len(clean), clean_threshold=threshold,
                       changed=int((restored != corrupted).flatten(1).any(1).sum()))
    torch.save(model.state_dict(), root / 'G.pt')
    write(root / 'held-smoke.json', holdout)
    write(root / 'G-frozen.json', checkpoint_record(contract, threshold))


@torch.no_grad()
def render(packets, models, device):
    model, threshold = models
    received = as_native(packets, device)
    corrected = model.restore(received, threshold, amount=1.)
    restored = [np.from_dlpack(torch.lerp(received, corrected, amount).reshape(packets.shape)
                               .cpu().contiguous()).copy() for amount in (.50, .25)]
    return packets, *restored


def evaluate(rt):
    activate(rt)
    device, root = 'cuda:0', Path(rt['run_root'])
    frozen = read(root / 'G-frozen.json')
    w24, contract = load_w24(rt, device)
    threshold = frozen['threshold']
    if not np.isfinite(threshold) or frozen != checkpoint_record(contract, threshold):
        raise ValueError('Native source-only checkpoint provenance differs')
    model = NativeRestorer().to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(root / 'G.pt', map_location=device, weights_only=True), strict=True)
    raw = native(Path(rt['source_root']), 'U_s', contract['counts']['U_s'])[:4]
    for wave in render(raw, (model, threshold), device):
        classify(w24, wave, device)
    write(root / 'smoke.json', dict(status='PASS', source_U_packets=4, target_reads=0, recognizer=METHOD))
    predict(rt, (w24, model, threshold), device, render=render, modes=MODES)


def dispatch(rt, runtime):
    root = Path(rt['run_root'])
    root.mkdir()
    w24_root = Path(rt['w24_release_root'])
    if read(w24_root / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected the original W24 release')
    source_runtime = root / 'source-runtime.json'
    write(source_runtime, {k: rt[k] for k in ('run_root', 'source_root', 'frozen_sources')})
    write(root / 'state.json', dict(status='WAIT_GPU'))
    deadline = time.monotonic() + 3600
    while not (cards := available(rt['allowed_gpus'], max_compute_processes=rt['max_compute_processes'])):
        if time.monotonic() >= deadline:
            write(root / 'state.json', dict(status='FAILED', error='GPU wait expired'), exclusive=False)
            return
        time.sleep(30)
    gpu = next(i for i in rt['allowed_gpus'] if i in cards)
    release = Path(rt['release_root'])
    script = str(Path(__file__).resolve())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    for phase, path, cwd in (('fit', source_runtime, release), ('evaluate', runtime, w24_root)):
        with (root / (phase + '.log')).open('x', encoding='utf-8') as log:
            command = [rt['python'], '-u', script, phase, str(path)]
            proc = subprocess.Popen(command, cwd=cwd, env=dict(env, PYTHONPATH=str(cwd)),
                                    stdout=log, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status='RUNNING', phase=phase, pid=proc.pid, gpu=gpu), exclusive=False)
            if proc.wait():
                write(root / 'state.json', dict(status='FAILED', error=phase, gpu=gpu), exclusive=False)
                return
    if read(root / 'predictions' / 'complete.json') != receipt(MODES):
        raise ValueError('Frozen native prediction matrix differs')
    write(root / 'state.json', dict(status='PREDICTIONS_FROZEN', gpu=gpu), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit', 'evaluate'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    publish = commands.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    scoring = commands.add_parser('score')
    for key in ('predictions', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        context = read(args.context)
        rt = {key: context[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        native_data = project / 'releases' / f'{DATA_RUN}-release' / 'data'
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN), source_root=str(native_data / 'source'),
                  native_target_root=str(native_data / 'target'), w24_run_root=str(project / 'runs' / W24_RUN),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  allowed_gpus=[5, 0, 2, 7, 6, 1, 4], max_compute_processes=8, cpu_workers=12)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'score':
        score(args.predictions, args.dataset, args.output, modes=MODES)
    else:
        runtime = read(args.runtime)
        if args.action == 'dispatch':
            dispatch(runtime, args.runtime)
        elif args.action == 'fit':
            fit(runtime)
        else:
            evaluate(runtime)


if __name__ == '__main__':
    main()
